using BayesianSumProductNetworks
using StatsBase
using StatsFuns
using AxisArrays
using LinearAlgebra
using Random
using DelimitedFiles
using ArgParse
using ProgressMeter
using Logging
using SparseArrays
using Dates
using FileIO
using JLD2
using Serialization
using Dates

import BayesianSumProductNetworks.resample!

s = ArgParseSettings()
@add_arg_table s begin
    "--alpha"
        help = "Concentration parameter of sum nodes"
        default = 1.0
        arg_type = Float64
    "--beta"
        help = "Concentration parameter of product nodes"
        default = 10.0
        arg_type = Float64
    "--gamma"
        help = "Concentration parameter of leaf mixtures"
        default = 0.1
        arg_type = Float64
    "--Ksum"
        help = "Number of children under a sum node"
        default = 4
        arg_type = Int
    "--Kprod"
        help = "Number of children under a product node"
        default = 2
        arg_type = Int
    "--Kleaf"
        help = "Number of children under a product node"
        default = 10
        arg_type = Int
    "--J"
        help = "Number of partitions."
        default = 2
        arg_type = Int
    "--L"
        help = "Number of layers."
        default = 2
        arg_type = Int
    "--numsamples"
        help = "Write numsamples many samples to disk (default 100%)."
        default = 0
        arg_type = Int
    "--seed"
        help = "Random seed = seed * Int(first(dataset)) * chain."
        default = 42
        arg_type = Int
    "--chain"
        help = "Chain"
        default = 1
        arg_type = Int
    "dataset"
        help = "Dataset name, assuming naming convention of 20 terrible datasets."
        required = true
        arg_type = String
    "datafolder"
        help = "Path to folder containing datasets."
        required = true
        arg_type = String
    "outputfolder"
        help = "Path to folder in which output files should be stored."
        required = true
        arg_type = String
    "burnin"
        required = true
        arg_type = Int
    "iterations"
        help = "Number of iterations."
        required = true
        arg_type = Int
end

args = parse_args(s)

# Load the dataset
dataset = args["dataset"]
datafolder = args["datafolder"]

# results dir
resultsdir = args["outputfolder"]
if !isdir(resultsdir)
    mkpath(resultsdir)
end

donefile = joinpath(resultsdir, "done")
if isfile(donefile)
    @info "MCMC sampling already done, exiting."
    exit(-2)
end

# Output files.
burninfile = joinpath(resultsdir, "bspn_burnin.jld2")
modelfile = joinpath(resultsdir, "bspn_model.jld2")
llhfile = joinpath(resultsdir, "llh.jld2")
tracefile = joinpath(resultsdir, "trace.jld2")
nparamfile = joinpath(resultsdir, "nparam.jld2")
iterfile = joinpath(resultsdir, "iteration")

# check if results are locked and exit if this is the case
lockfile = joinpath(resultsdir, "filelock.lck")

# Check if we should continue the sampling...
if (isfile(modelfile) || isfile(burninfile)) && isfile(iterfile)
    d = now() - Dates.unix2datetime(stat(iterfile).mtime)
    if round(d, Minute) > Minute(60)
        # remove lock file
        if isfile(lockfile)
            rm(lockfile)
        end
    end
end

# Check if we should continue the sampling...
if isfile(lockfile)
    @info "MCMC sampling is locked, exiting."
    exit(-2)
else

    t = rand(1:15) + rand(1:15)
    sleep(t)

    if isfile(lockfile)
        @info "MCMC sampling is locked, exiting."
        exit(-2)
    end

    touch(lockfile)
end

# Write out the configuration of the run.
save(joinpath(resultsdir, "configuration.jld2"), Dict("configuration" => args))

# Start script
open(joinpath(resultsdir, "log.txt"), "w+") do io
    logger = SimpleLogger(io)
    with_logger(logger) do
        if Threads.nthreads() == 1
            @info "Run export JULIA_NUM_THREADS=nproc before running julia to use multithreading."
        else
            @info "Using $(Threads.nthreads()) threads."

            # Test if all threads are available.
            thrsa = zeros(Threads.nthreads())
            Threads.@threads for t in 1:Threads.nthreads()
                thrsa[t] = Threads.threadid()
            end

            avThreads = length(unique(thrsa[thrsa .> 0]))
            @assert avThreads == Threads.nthreads() "Only $(avThreads) threads available."
        end

        @info "Loading dataset: $dataset stored under: $datafolder"
        x_train = vcat(
            readdlm(joinpath(datafolder, dataset*".ts.data"), ','),
            readdlm(joinpath(datafolder, dataset*".valid.data"), ',')
        )

        # Set missing values to NaN
        x_train[x_train .== "missing"] .= NaN
        x_train = Float64.(x_train)

        x_test = readdlm(joinpath(datafolder, dataset*".test.data"), ',', Bool)

        # Set the random seed.
        rnd_seed = args["seed"]+args["chain"]
        rng = Random.seed!(rnd_seed)

        # Get data dimensions.
        (N_train, D) = size(x_train)
        (N_test, _) = size(x_test)

        @info "Sampling dataset of (train) size: N=$N_train, D=$D and (test) size: N=$N_test"

        # Define the base distributions
        priors = map(d -> Beta(1/2, 1/2), 1:D)
        likelihoods = map(d -> Bernoulli(rand(priors[d])), 1:D)
        sstats = map(d -> BetaSufficientStats(), 1:D)

        # Construct a second network which we use for sampling.
        @info "Constructing SPN for sampling."
        spn = templateRegion(args["alpha"],
                             args["beta"],
                             args["gamma"],
                             priors,
                             likelihoods,
                             sstats,
                             N_train, D,
                             args["Ksum"], args["Kprod"],
                             args["J"], args["Kleaf"],
                             1, args["L"], root = true)

        # Set the number of iterations.
        iterations = args["iterations"]
        burnin = args["burnin"]

        # ######### #
        # MCMC code #
        # ######### #
        ks = args["numsamples"] > 0 ? args["numsamples"] : iterations
        ks = min(iterations, ks)
        thinning = Int(iterations / ks)

        @info "Starting MCMC for $iterations iterations."

        startIter = 1
        storePredictions = true
        storeTrace = false

        # Initialize SPN.
        if isfile(modelfile)
            # Resume.
            spn = open(modelfile, "r") do fin
                deserialize(fin)
            end
            r_ = read(iterfile, Int)
            startIter = 1 + r_
            @info "Continue MCMC."
        elseif isfile(burninfile)
            spn = open(burninfile, "r") do fin
                deserialize(fin)
            end
            @info "Continue burnin or start using previously converged chain, not storing trace."
            r_ = read(iterfile, Int)
            startIter = 1 + r_
        else
            @warn "Starting from random position, not storing predictions!"
            randomscopes!(spn, x_train)
        end

        nodes = topoligicalorder(spn)

        for iteration in startIter:(burnin+iterations)

            storeTrace = iteration <= burnin
            storePredictions = iteration > burnin

            dateprint = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")

            @info "[$(dateprint)] iteration = $(iteration) "

            ancestralsampling!(spn, x_train)
            gibbssamplescopes!(spn, x_train)

            # Store the scopes, weights and parameters on the disk.
            if ((iteration-burnin) % thinning) == 0

                # Serialize model.
                fname = joinpath(resultsdir, "tmp_"*randstring()*".jld2")
                open(fname, "w") do fout
                    serialize(fout, spn)
                end
                if iteration > burnin
                    mv(fname, modelfile, force=true)
                else
                    mv(fname, burninfile, force=true)
                end

                if storeTrace
                    # Store parameters of root.
                    Nparam = length(spn.logweights[:])
                    θ = isfile(tracefile) ? load(tracefile, "weights") : zeros(0,Nparam)
                    fname = joinpath(resultsdir, "tmp_"*randstring()*".jld2")
                    save(fname, "weights", vcat(θ, spn.logweights[:]'))
                    mv(fname, tracefile, force=true)
                end

                if storePredictions
                    # Evaluate network.
                    lp_test = logpdf(spn, x_test)
                    lp_train = logpdf(spn, x_train)

                    b_ = isfile(llhfile)
                    llh_test = b_ ? load(llhfile, "test") : zeros(0,N_test)
                    llh_train = b_ ? load(llhfile, "train") : zeros(0,N_train)
                    fname = joinpath(resultsdir, "tmp_"*randstring()*".jld2")
                    save(fname, "train", vcat(llh_train, lp_train'), "test", vcat(llh_test, lp_test'))
                    mv(fname, llhfile, force=true)

                    nweights = isfile(nparamfile) ? load(nparamfile, "weights") : Vector{Int}()
                    nleaves = isfile(nparamfile) ? load(nparamfile, "leaves") : Vector{Int}()
                    n_ = 0
                    for n in filter(n -> n isa RegionGraphNode, nodes)
                        n_ += sum(n.active)
                    end
                    push!(nweights, n_)

                    n_ = 0
                    for n in filter(n -> n isa FactorizedDistributionGraphNode, nodes)
                        n_ += sum(any(n.obsVecs, dims=1)) * length(scope(n))
                    end
                    push!(nleaves, n_)

                    fname = joinpath(resultsdir, "tmp_"*randstring()*".jld2")
                    save(fname, "weights", nweights, "leaves", nleaves)
                    mv(fname, nparamfile, force=true)
                end

                write(iterfile, iteration)
            end
        end

        @info "Finished MCMC"

        # Save file indicating we are done.
        touch(donefile)
        rm(lockfile)

        tmp_files = filter(f -> isfile(joinpath(resultsdir,f)) && startswith(f, "tmp_"), readdir(resultsdir))
        for f in tmp_files
            rm(joinpath(resultsdir,f))
        end
    end
end
