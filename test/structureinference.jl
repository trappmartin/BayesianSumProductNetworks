using BayesianSumProductNetworks
using Test, Random
using AxisArrays, StatsFuns

priors = [NormalInverseGamma(0.0, 1.0, 1.0, 1.0), Beta()]
likelihoods = [Normal(), Bernoulli()]
sstats = [NormalInvGammaSufficientStats(), BetaSufficientStats()]
prior_sums_alpha = 1.0
prior_prods = 1.0
prior_leaves = 0.1
J = 2
K = 10
K_sum = 4
K_prod = 2

N = 100
D = 2

Random.seed!(1)
x = hcat(randn(N,1), rand(0:1, N,1))

# SPN
spn = templateRegion(prior_sums_alpha, prior_prods_alpha, prior_leaves,
    priors, likelihoods, sstats, N, D, 
    K_sum, K_prod, J, K, 1, 2, root = true)

# Initialization
randomscopes!(spn, x)

@test scope(spn) == [1,2]

# Single ancestral sampling step
ancestralsampling!(spn, x)

@test sum(exp.(spn.logweights)) ≈ 1

# Single gibbs step
gibbssamplescopes!(spn, x)

@test scope(spn) == [1,2]

for child in children(spn)
    @test scope(child) == [1,2]
    scopes = map(n -> scope(n), children(child))
    @test length(reduce(vcat, scopes)) == 2
end

iterations = 1000

clusters = sort.([ [[1,2],[]], [[1],[2]] ])
lscopes = Vector{Vector}(undef, iterations)

for i in 1:iterations
    gibbssamplescopes!(spn, x)
    ancestralsampling!(spn, x)

    c = children(spn)[1]
    lscopes[i] = sort(map(n -> scope(n), children(c)))
end

cs = map(s -> findfirst(map(c -> c == s, clusters)), lscopes)

@test (sum(cs .== 1) / iterations) ≈ 0.5 atol=0.075
@test (sum(cs .== 2) / iterations) ≈ 0.5 atol=0.075

priors = [[NormalInverseGamma(0.0, 1.0, 1.0, 1.0), Gamma()], [Beta(), Gamma()]]
likelihoods = [Distribution[Normal(), Exponential()], Distribution[Bernoulli(), Poisson()]]
sstats = [[NormalInvGammaSufficientStats(), GammaSufficienStats()], [BetaSufficientStats(), GammaSufficienStats()]]

x[:,1] .-= minimum(x[:,1])

spn = templateRegion(prior_sums_alpha, prior_prods_alpha, prior_leaves,
    priors, likelihoods, sstats, N, D, 
    K_sum, K_prod, J, K, 1, 2, root = true)

# Initialization
randomscopes!(spn, x)

@test scope(spn) == [1,2]

# Single ancestral sampling step
ancestralsampling!(spn, x)

@test sum(exp.(spn.logweights)) ≈ 1

# Single gibbs step
gibbssamplescopes!(spn, x)

@test scope(spn) == [1,2]

for child in children(spn)
    @test scope(child) == [1,2]
    scopes = map(n -> scope(n), children(child))
    @test length(reduce(vcat, scopes)) == 2
end
