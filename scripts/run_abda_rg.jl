using BayesianSumProductNetworks
using StatsBase
using StatsFuns
using AxisArrays
using LinearAlgebra
using Random
using DelimitedFiles
using JLD2
using FileIO
using ArgParse
using Logging
using SparseArrays
using Dates
using HDF5
using Serialization

include("helper.jl")

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
        help = "Concentration parameter of leaf distributions"
        default = 0.1
        arg_type = Float64
    "--Ksum"
        help = "Number of children under a sum node"
        default = 2
        arg_type = Int
    "--Kprod"
        help = "Number of children under a product node"
        default = 2
        arg_type = Int
    "--Kleaf"
        help = "Number of leaves in an atomic region"
        default = 2
        arg_type = Int
    "--J"
        help = "Number of partitions."
        default = 2
        arg_type = Int
    "--L"
        help = "Number of layers."
        default = 2
        arg_type = Int
    "--seed"
        help = "Random seed = seed * Int(first(dataset)) * chain."
        default = 42
        arg_type = Int
    "--chain"
        help = "Chain"
        default = 1
        arg_type = Int
    "--maxcategories"
        default = 20
        arg_type = Int
    "--numsamples"
        arg_type = Int
        default = 1000
        help = "Number of MCMC samples we want to store."
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
        help = "Number of burnin iterations."
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

@info "Processing $dataset in $datafolder"
@info "Storing results in $resultsdir"

donefile = joinpath(resultsdir, "done")
if isfile(donefile)
    @warn "Sampling already done."
    exit()
end

# check if results are locked and exit if this is the case
lockfile = joinpath(resultsdir, "filelock.lck")

# Output files.
modelfile = joinpath(resultsdir, "bspn_model.jld2")
burninfile = joinpath(resultsdir, "bspn_burnin.jld2")
llhfile = joinpath(resultsdir, "llhfile.jld2")
tracefile = joinpath(resultsdir, "tracefile.jld2")
iterfile = joinpath(resultsdir, "iteration")

# Check if we should continue the sampling...
if (isfile(modelfile) || isfile(burninfile)) && isfile(iterfile)
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

# Start script.
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

        x_train = map(Float64, h5read(joinpath(datafolder, dataset*"PP", "train.hdf5"), "train"))
        x_train = transpose(x_train)

        x_valid = map(Float64, h5read(joinpath(datafolder, dataset*"PP", "valid.hdf5"), "valid"))
        x_train = cat(x_train, transpose(x_valid), dims = 1)

        x_test = map(Float64, h5read(joinpath(datafolder, dataset*"PP", "test.hdf5"), "test"))
        x_test = transpose(x_test)

        # Set the number of iterations, burnin, etc.
        iterations = args["iterations"]
        burnin = args["burnin"]

        # ######### #
        # MCMC code #
        # ######### #
        ks = min(iterations, args["numsamples"])
        @assert isinteger(iterations / ks)
        thinning = Int(iterations / ks)

        (N_train, D) = size(x_train)
        (N_test, _) = size(x_test)

        @info "Processing dataset of size N=$N_train, D=$D."

        # Preprocessing
        dids = findall(map(d -> all(isinteger, filter(x -> !isnan(x), x_train[:,d])), 1:D))
        dids = filter(d -> length(filter(x -> !isnan(x), unique(x_train[:,d]))) < args["maxcategories"], 1:D)
        for d in dids
            ix = findall(!isnan, x_train[:,d])
            l = unique(x_train[ix,d])

            x_train[ix,d] = map(x -> findfirst(l .== x), x_train[ix,d])

            ix = findall(!isnan, x_test[:,d])
            x_test[ix,d] = map(x -> x ∈ l ? findfirst(l .== x) : length(l)+1, x_test[ix,d])

            if length(l) == 2
                # Make binary.
                x_train .-= 1
                x_test .-= 1
            end
        end

        priors = Vector{Vector{Distribution}}(undef, D)
        likelihoods = Vector{Vector{Distribution}}(undef, D)
        sstats = Vector{Vector{AbstractSufficientStats}}(undef, D)

        for d in 1:D
            priors[d] = Vector{Distribution}()
            likelihoods[d] = Vector{Distribution}()
            sstats[d] = Vector{Distribution}()
            ix = findall(!isnan, x_train[:,d])

            if d ∈ dids # is discrete ?
                if minimum(x_train[ix,d]) == 0
                    # Binary data
                    push!(priors[d], Beta(1/2, 1/2))
                    push!(likelihoods[d], Bernoulli())
                    push!(sstats[d], BetaSufficientStats())
                else
                    # Categorical data
                    l = unique(x_train[:,d])
                    push!(priors[d], Dirichlet(length(l), 0.1))
                    push!(likelihoods[d], Categorical(rand(last(priors[d]))))
                    push!(sstats[d], DirichletSufficientStats(length(l)))
                end

                # Poisson data
                push!(priors[d], Gamma())
                push!(likelihoods[d], Poisson())
                push!(sstats[d], GammaSufficienStats())
            else # continuous data
                if all((x) -> x >= 0, x_train[ix,d])
                    # Positive continuous data
                    push!(priors[d], Gamma())
                    push!(likelihoods[d], Exponential())
                    push!(sstats[d], GammaSufficienStats())
                end

                # Normal distributed data
                push!(priors[d], NormalInverseGamma(mean(x_train[ix,d]), 1.0/std(x_train[ix,d]), 2.0, 3.0))
                push!(likelihoods[d], Normal())
                push!(sstats[d], NormalInvGammaSufficientStats())
            end
        end

        # Set the random seed.
        rnd_seed = args["seed"]+args["chain"]
        rng = Random.seed!(rnd_seed)

        # Construct a second network which we use for sampling.
        @info "Constructing spn for sampling."
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

        for iteration in startIter:(burnin + iterations)
            dateprint = Dates.format(now(), "HH:MM:SS")

            storeTrace = iteration <= burnin
            storePredictions = iteration > burnin

            @info "[$(dateprint)] iteration = $(iteration) "

            ancestralsampling!(spn, x_train)
            gibbssamplescopes!(spn, x_train)

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
                end

                write(iterfile, iteration)
            end
        end
        # Save sampling results to disk.
        @info "Finished MCMC"

        # Create done file and release lock.
        touch(donefile)
        rm(lockfile)

        tmp_files = filter(f -> isfile(joinpath(resultsdir,f)) && startswith(f, "tmp_"), readdir(resultsdir))
        for f in tmp_files
            rm(joinpath(resultsdir,f))
        end
    end
end
