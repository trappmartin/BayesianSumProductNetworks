using BayesianSumProductNetworks
using Test
using SpecialFunctions

@testset "normal inverse gamma - mllh" begin

    x = [0.1, -0.1, 0.0]

    H = NormalInverseGamma(0, 1, 2, 3)
    sstat = NormalInvGammaSufficientStats(length(x), sum(x), sum( (x .- mean(x)).^2 ))

    n = length(x)
    (μ0, ν0, a0, b0) = params(H)
    (μn, νn, an, bn) = postparams(H, sstat)

    # Compute marginal likelihood according to Murphy.
    pD = (gamma(an) / gamma(a0)) * sqrt( (ν0/νn)) * ((2*b0)^a0 / (2*bn)^an) * (1/(π^(n/2)))
    logpD = log(pD)
    logpD2 = BayesianSumProductNetworks.invgmllh(μ0, ν0, a0, b0, μn, νn, an, bn, n)

    @test logpD ≈ logpD2
end

@testset "categorical - mllh" begin

    x = [0, 1, 0]

    γ = [0.5, 0.5]
    sstat = map(v -> sum(x .== v), 0:1)
    n = length(x)

    # Compute marginal likelihood according to Stephen Tu.
    pD = (gamma(sum(γ)) / prod(gamma.(γ))) * (prod(gamma.(map(j -> sum(γ[j] + sstat[j]), 1:2))) / gamma(n + sum(γ)))
    @test log(pD) ≈ BayesianSumProductNetworks.categoricalMLL(γ, sstat)
end
