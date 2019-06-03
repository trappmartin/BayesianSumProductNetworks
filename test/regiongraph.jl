using BayesianSumProductNetworks
using Test, Random
using AxisArrays

priors = [NormalInverseGamma(0.0, 1.0, 1.0, 1.0), Beta()]
likelihoods = [Normal(), Bernoulli()]
sstats = [NormalInvGammaSufficientStats(), BetaSufficientStats()]
prior_sums_alpha = 1.0
prior_prods_alpha = 10.0
prior_leaf_alpha = 1.0
K = 10
K_sum = 2
K_prod = 2

N = 100
D = 2

Random.seed!(1)
x = hcat(randn(N,1), rand(0:1, N,1))

# Leaves with factorization over scope
leaves = templateLeaves(prior_leaf_alpha, priors, likelihoods, sstats, N, D, K)
@test length(leaves) == K

setscope!(leaves, [1,2])
@test scope(leaves) == [1,2]

@test size(logpdf(leaves,x)) == (N,K)
@test vec(sum(logpdf(leaves,x), dims=1)) ≈ ones(K) * -238.453 atol=0.01

# Partitions
partition = templatePartition(prior_leaf_alpha, prior_sums_alpha,
    prior_prods_alpha, priors, likelihoods,
    sstats, N, D, K_sum, K_prod, 1, K, 1, 1)

@test length(partition) == K^K_prod

for (d, child) in enumerate(children(partition))
    setscope!(child, [d])
end

updatescope!(partition)
@test scope(partition) == [1,2]

@test size(logpdf(partition, x)) == (N, K^K_prod)

# Test if logpdf is correct
lp_ = logpdf.(children(partition), Ref(x))
r_ = logpdf(partition, x)[1,:]
for j in 1:length(r_)
    j1, j2 = Base._ind2sub((K,K), j)
    @test r_[j] ≈ lp_[1][1,j1] + lp_[2][1,j2]
end

# Test in-place logpdf
nodes = topoligicalorder(partition)
out = AxisArray(Vector{Matrix}(undef, length(nodes)), map(id, nodes))

for n in nodes
    logpdf!(n, x, out)
end

# Regions
region = templateRegion(prior_leaf_alpha, prior_sums_alpha,
    prior_prods_alpha, priors, likelihoods,
    sstats, N, D, K_sum, K_prod, 2, K, 1, 1)
@test length(region) == K_sum

for child in children(region)
    setscope!(child, [1,2])
end

updatescope!(region)
@test scope(region) == [1,2]

@test size(logpdf(region, x)) == (N, K_sum)

# Test if logpdf is correct
lp_ = logpdf.(children(region), Ref(x))
lp_ = mapreduce(k -> lp_[k][1,:], vcat, 1:K_sum)
lp_ = lp_ .+ region.logweights

@test vec(logpdf(region, x)[1,:]) ≈ vec(lse(lp_, dims=1))

# Test in-place logpdf
nodes = topoligicalorder(region)
out = AxisArray(Vector{Matrix}(undef, length(nodes)), map(id, nodes))

for n in nodes
    logpdf!(n, x, out)
    @test out[id(n)] ≈ logpdf(n, x)
end

# Test util function
@test length(topoligicalorder(region)) == 3

# Test region graph for heterogenuous data
priors = [[Beta(1, 1), Gamma()], [NormalInverseGamma(1.0, 1.0, 1.0, 1.0), Gamma()]]
likelihoods = [Distribution[Bernoulli(), Poisson()], Distribution[Normal(), Exponential()]]
sstats = [[BetaSufficientStats(), GammaSufficienStats()], [NormalInvGammaSufficientStats(), GammaSufficienStats()]]

leaves = templateLeaves(prior_leaf_alpha, priors, likelihoods, sstats, N, D, K);
@test length(leaves) == K
@test size(leaves.obsVecs) == (N, K)
@test all((i) -> i == 0, logpdf(leaves, x))

setscope!(leaves, [1,2])
@test scope(leaves) == [1,2]
@test size(logpdf(leaves,x)) == (N, K)
@test all(isfinite, logpdf(leaves,x))
@test all((i) -> i != 0, logpdf(leaves, x))

partition = templatePartition(prior_leaf_alpha, prior_sums_alpha,
    prior_prods_alpha, priors, likelihoods,
    sstats, N, D, K_sum, K_prod, 1, K, 1, 1)

@test length(partition) == K^K_prod

# Test case where the mixture has only a single component.
priors = [Distribution[Beta(1, 1)], Distribution[NormalInverseGamma(1.0, 1.0, 1.0, 1.0)]]
likelihoods = [Distribution[Bernoulli()], Distribution[Normal()]]
sstats = [AbstractSufficientStats[BetaSufficientStats()], AbstractSufficientStats[NormalInvGammaSufficientStats()]]

leaves = templateLeaves(prior_leaf_alpha, priors, likelihoods, sstats, N, D, K);
setscope!(leaves, [1,2])

@test all((i) -> isfinite(i) && i != 0, logpdf(leaves, x))
