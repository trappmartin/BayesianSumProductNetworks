using BayesianSumProductNetworks
using Test

@testset "distributions" begin

    # Normal with Normal and Inverse Gamma priors
    H0 = NormalInverseGamma(0.0, 1.0, 1.0, 1.0)
    @test params(H0) == (0.0, 1.0, 1.0, 1.0)

    T = NormalInvGammaSufficientStats()
    @test T.n == 0
    @test T.s == 0
    @test T.sdiff == 0

    # Categorical with Dirichlet prior
    H0 = Dirichlet([0.5, 0.5])
    @test H0.alpha == [0.5, 0.5]

    T = DirichletSufficientStats(2)
    @test T.counts == zeros(Int, 2)

    # Poisson with Gamma prior.
    H0 = Gamma(1.0, 1.0)
    @test H0.α == 1.0
    @test H0.θ == 1.0

    T = GammaSufficienStats()
    @test T.n == 0
    @test T.sn == 0

end
