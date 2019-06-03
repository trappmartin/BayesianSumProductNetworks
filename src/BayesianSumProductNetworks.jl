__precompile__()

module BayesianSumProductNetworks

    import Base.eltype
    import Base.length
    import Base.rand
    import Base.copy

    # Required packages
    using Reexport
    @reexport using Distributions
    using AxisArrays
    using SpecialFunctions
    using StatsFuns
    import StatsFuns.normlogpdf

    # Required base packages.
    using Statistics
    using Random
    using SparseArrays

    # Exports
    export NormalInverseGamma
    export DirichletSufficientStats, NormalInvGammaSufficientStats
    export GammaSufficienStats, BetaSufficientStats
    export AbstractSufficientStats
    export postparams, update!
    export RegionGraphNode, PartitionGraphNode
    export FactorizedDistributionGraphNode, AbstractRegionGraphNode
    export FactorizedMixtureGraphNode, InfiniteSumNode
    export haschildren, topoligicalorder, id
    export scope, randomscopes!, nscope, hasscope, setscope!, updatescope!
    export haschildren, children
    export logpdf!, logmllh!, logpdf

    ######################
    ## Type Definitions ##
    ######################

    abstract type AbstractSufficientStats end

    """
        Gamma prior sufficient stats.
    """
    mutable struct GammaSufficienStats <: AbstractSufficientStats
        n::Int
        sn::Float64
    end

    GammaSufficienStats() = GammaSufficienStats(0, 0.0)

    function update!(s::GammaSufficienStats, x::AbstractVector)
        if isempty(x)
            s.n = 0
            s.sn = 0.0
        else
            @assert all(x .>= 0) "Not all positive, $x"
            s.n = length(x)
            s.sn = sum(x)
        end
        return s
    end

    @inline postparams(d::Gamma, T::GammaSufficienStats) = (d.α+T.sn, (1/d.θ)+T.n)

    """
        Beta prior sufficient stats.
    """
    mutable struct BetaSufficientStats <: AbstractSufficientStats
        n::Int
        count::Int
    end

    function update!(s::BetaSufficientStats, x::T) where {T<:AbstractVector{<:Real}}
        if isempty(x)
            s.n = 0
            s.count = 0
        else
            @assert all((y) -> isone(y) | iszero(y), x)
            s.n = length(x)
            s.count = sum(x)
        end
        return s
    end
    @inline BetaSufficientStats() = BetaSufficientStats(0,0)
    @inline postparams(d::Beta, T::BetaSufficientStats) = (T.count+d.α,(T.n-T.count)+d.β)

    """
        Dirichlet prior sufficient stats.
    """
    struct DirichletSufficientStats <: AbstractSufficientStats
        counts::Vector{Int}
    end
    function update!(s::DirichletSufficientStats, x::AbstractVector)
        if isempty(x)
            fill!(s.counts, 0)
        else
            s.counts[:] = map(k -> sum(x .== k), 1:length(s.counts))
        end
        return s
    end
    @inline DirichletSufficientStats(n::Int) = DirichletSufficientStats(zeros(Int, n))
    @inline postparams(d::Dirichlet, T::DirichletSufficientStats) = T.counts.+d.alpha

    ## Gaussian with Normal Inverse Gamma prior ##
    struct NormalInverseGamma <: Distribution{Multivariate,Continuous}
        μ::Float64
        ν::Float64
        a::Float64
        b::Float64
    end

    length(d::NormalInverseGamma) = 2

    function logpdf(d::NormalInverseGamma, θ...)
        (μ, σ²) = θ
        l = _invgammalogpdf(d.a, d.b, σ²)
        l += normlogpdf(d.μ, sqrt(σ²) / d.ν, μ)
        return l
    end
    @inline _invgammalogpdf(a::Real, b::Real, x::Real) = a*log(b)-lgamma(a)-(a+1)*log(x)-b/x

    """
        Normal-Inverse-Gamma prior sufficient stats.
    """
    mutable struct NormalInvGammaSufficientStats <: AbstractSufficientStats
        n::Int
        s::Float64
        sdiff::Float64
    end

    NormalInvGammaSufficientStats() = NormalInvGammaSufficientStats(0,0.0,0.0)

    function update!(s::NormalInvGammaSufficientStats, x::AbstractVector)
        if isempty(x)
            s.n = 0
            s.s = 0
            s.sdiff = 0
        else
            s.n = length(x)
            s.s = sum(x)
            s.sdiff = sum((x .- mean(x)).^2)
        end
        s
    end

    function rand(d::NormalInverseGamma)
        σ² = rand(InverseGamma(d.a, d.b))
        μ = rand(Normal(d.μ, sqrt(σ² / d.ν )))
        return [μ, σ²]
    end

    @inline params(d::NormalInverseGamma) = (d.μ, d.ν, d.a, d.b)
    function postparams(d::NormalInverseGamma, T::NormalInvGammaSufficientStats)

        if T.n == 0
            return (d.μ, d.ν, d.a, d.b)
        end

        μ = (d.ν*d.μ + T.s) / (d.ν + T.n)
        ν = d.ν + T.n
        a = d.a + (T.n / 2)
        b = d.b
        b += T.sdiff/2 + ((T.n * d.ν) / (d.ν + T.n)) * (((T.s / T.n) - d.μ)^2 / 2)

        return (μ, ν, a, b)
    end

    # include implementations
    include("mathutil.jl")
    include("distUtils.jl")
    include("templateStructure.jl")
    include("regiongraph.jl")
    include("nonparametrics.jl")
    include("ancestralsampler.jl")
    include("initialization.jl")
    include("structureinference.jl")

end # module
