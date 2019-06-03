using BayesianSumProductNetworks
using Test, Random
using AxisArrays, StatsFuns

priors = [NormalInverseGamma(0.0, 1.0, 1.0, 1.0), Beta()]
likelihoods = [Normal(), Bernoulli()]
sstats = [NormalInvGammaSufficientStats(), BetaSufficientStats()]
prior_sums_alpha = 1.0
prior_prods_alpha = 10.0
prior_leaves_alpha = 0.01
J = 2
K = 10
K_sum = 4
K_prod = 2

N = 100
D = 2

Random.seed!(1)
x = hcat(randn(N,1), rand(0:1, N,1))

# SPN
spn = templateRegion(prior_sums_alpha, prior_prods_alpha, prior_leaves_alpha,
    priors, likelihoods, sstats, N, D, K_sum, K_prod, J, K, 1, 2, root = true)

# Initialization
randomscopes!(spn, x)

@test scope(spn) == [1,2]

# Single ancestral sampling step
ancestralsampling!(spn, x)

@test sum(exp.(spn.logweights)) ≈ 1

# SPN
spn = templateRegion(prior_sums_alpha,
    prior_prods_alpha, prior_leaves_alpha, priors, likelihoods,
    sstats, N, D, K_sum, K_prod, J, K, 1, 2, root = true)

# Initialization
Random.seed!(1)
randomscopes!(spn, x)

performance = Vector()
llhvals = ones(N) * -Inf
for i in 1:1000
    ancestralsampling!(spn, x)
    llhvals_ = logpdf(spn, x)
    for n in 1:N
        @inbounds begin
            llhvals[n] = logaddexp(llhvals[n], llhvals_[n])
        end
    end
    push!(performance, mean(llhvals) - log(i))
end

@test last(performance) ≈ -2.1 atol=0.1

# SPN
spn = templateRegion(prior_sums_alpha,
    prior_prods_alpha, prior_leaves_alpha, priors, likelihoods,
    sstats, N, D, 5, 2, 2, 5, 1, 4, root = true)

# Initialization
randomscopes!(spn, x)

ancestralsampling!(spn, x);

# Test region graph for heterogenuous data
priors = [[Beta(1, 1), Gamma()], [NormalInverseGamma(1.0, 1.0, 1.0, 1.0), Gamma()]]
likelihoods = [Distribution[Bernoulli(), Poisson()], Distribution[Normal(), Exponential()]]
sstats = [[BetaSufficientStats(), GammaSufficienStats()], [NormalInvGammaSufficientStats(), GammaSufficienStats()]]

Random.seed!(1)
x = hcat(rand(0:1,N,1), randn(N,1))
x[:,2] .-= minimum(x[:,2])

# SPN
spn = templateRegion(prior_sums_alpha, prior_prods_alpha, prior_leaves_alpha,
    priors, likelihoods, sstats, N, D, K_sum, K_prod, J, K, 1, 2, root = true)

# Initialization
Random.seed!(1)
randomscopes!(spn, x)

performance = Vector()
llhvals = ones(N) * -Inf
for i in 1:1000
    ancestralsampling!(spn, x)
    llhvals_ = logpdf(spn, x)
    for n in 1:N
        @inbounds begin
            llhvals[n] = logaddexp(llhvals[n], llhvals_[n])
        end
    end
    push!(performance, mean(llhvals) - log(i))
end

@test last(performance) ≈ -2.2 atol=0.1
