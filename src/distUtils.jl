#######################
## Log PDF functions ##
#######################

function _logpdf(::D, x::V, θ...) where {D<:Normal,V<:AbstractVector{<:Real}}
    (μ, σ) = θ
    return normlogpdf.(μ, σ, x)
end

function _logpdf(::D, x::V, w::T) where {D<:Bernoulli,V<:AbstractVector{<:Real},T<:Real}
    return bernlogpdf.(w, x)
end

function bernlogpdf(w::T, x::V) where {T<:Real,V<:Real}
    return isnan(x) ? 0.0 : (x == one(x) ? log(w) : log(1-w))
end

function catlogpdf(lw::T, x::V) where {T<:AbstractVector{<:Real},V<:Real}
    return 0 < x <= length(lw) ? (isinteger(x) ? lw[Int(x)] : (isnan(x) ? 0.0 : -Inf)) : -Inf
end

function _logpostpdf(d::Normal{T}, x::AbstractVector{<:Real}, θ::Tuple) where {T<:Real}

    (μ, ν, α, β) = θ

    dof = 2*α
    loc = μ
    scale = sqrt( (β*(1+1/ν)) / α )
    # Compute posterior predictive shifted and scaled, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    return @inbounds _logpostpdf.(Ref(d), x, dof, loc, scale)
end

function _logpostpdf(::Normal{T}, x::V, dof, loc, scale) where {T<:Real, V<:Real}
    return isnan(x) ? 0.0 : tdistlogpdf(dof, (x-loc)/scale) - log(scale)
end

function _logpostpdf(d::Categorical{T}, x::AbstractVector{<:Real}, θ::AbstractArray) where {T<:Real}
    α = [i for i in θ]
    lw = log.(α) .- log(sum(α))
    return _logpostpdf.(Ref(d), x, Ref(lw))
end

function _logpostpdf(::Categorical{T}, x::V, lw::AbstractVector) where {T<:Real,V<:Real}
    if isnan(x)
        return zero(V)
    else
        return isinteger(x) ? (length(lw) >= x > 0) ? lw[map(Int,x)] : map(V,-Inf) : map(V,-Inf)
    end
end

## Bayesian Poisson Gamma
function _logpostpdf(d::Poisson{V}, x::AbstractVector{<:Real}, θ::Tuple) where {V<:Real}
    α, β = θ
    dist = NegativeBinomial(α, β / (1+β))
    return _logpostpdf.(Ref(d), x, Ref(dist))
end

function _logpostpdf(::Poisson{V}, x::T, dist::NegativeBinomial) where {V<:Real,T<:Real}
    return isnan(x) ? zero(T) : Distributions.logpdf(dist, x)
end

## Bayesian Exponential Gamma
function _logpostpdf(d::Exponential{V}, x::AbstractVector{<:Real}, θ::Tuple) where {V<:Real}
    (λ, α) = θ
    return _logpostpdf.(Ref(d), x, α, λ)
end

function _logpostpdf(::Exponential{V}, x::T, α, β) where {V<:Real,T<:Real}
    return isnan(x) ? zero(T) : lomax(x, α, β)
end

## Bayesian Bernoulli Beta
function _logpostpdf(d::Bernoulli{V}, x::AbstractVector{<:Real}, θ::Tuple) where {V<:Real}
    (α, β) = θ
    p = α / (α+β)
    return _logpostpdf.(Ref(d), x, log(p), log(1-p))
end

function _logpostpdf(::D, x::T, lw0::V, lw1::V) where {T<:Real,V<:Real,D<:Bernoulli}
    return isnan(x) ? 0.0 : x == one(x) ? lw0 : lw1
end

### ################################# ###
### Marginal log likelihood functions ###
### ################################# ###
function _logmllh(d::Bernoulli{V}, prior::Distribution, x::AbstractVector{<:Real}, θ::Tuple) where {V<:Real}
    (α, β) = θ
    if isempty(x)
        θ_ = Distributions.rand(prior)
        return Distributions.logpdf(prior, θ_)
    end

    p = α / (α+β)
    return sum(_logpostpdf.(Ref(d), x, log(p), log(1-p)))
end

function _logmllh(d::Categorical{T}, prior::Distribution, x::AbstractVector{<:Real}, θ::AbstractArray) where {V<:Real, T<:Real}
    α = [i for i in θ]
    if isempty(x)
        θ_ = Distributions.rand(prior)
        return Distributions.logpdf(prior, θ_)
    end

    lw = log.(α) .- log(sum(α))
    return sum(_logpostpdf.(Ref(d), x, Ref(lw)))
end

function _logmllh(d::Normal{T}, prior::Distribution, x::AbstractVector{<:Real}, θ::Tuple) where {T<:Real}

    (μ, ν, α, β) = θ

    dof = 2*α
    loc = μ
    scale = sqrt( (β*(1+1/ν)) / α )

    if isempty(x)
        θ_ = rand(prior)
        return logpdf(prior, θ_...)
    end

    return @inbounds sum(_logpostpdf.(Ref(d), x, dof, loc, scale))
end

function _logmllh(d::Exponential{V}, prior::Distribution, x::AbstractVector{<:Real}, θ::Tuple) where {V<:Real}
    λ, α = θ
    if isempty(x)
        θ_ = Distributions.rand(prior)
        return Distributions.logpdf(prior, θ_...)
    end

    return @inbounds sum(_logpostpdf.(Ref(d), x, α, λ))
end

function _logmllh(d::Poisson{V}, prior::Distribution, x::AbstractVector{<:Real}, θ::Tuple) where {V<:Real}
    α, β = θ

    if isempty(x)
        θ_ = Distributions.rand(prior)
        return Distributions.logpdf(prior, θ_...)
    end

    dist = NegativeBinomial(α, 1 / (1+β))

    return @inbounds sum(_logpostpdf.(Ref(d), x, Ref(dist)))
end

function invgmllh(μ0, ν0, a0, b0, μn, νn, an, bn, n)
    r = ( lgamma(an) - lgamma(a0) ) + 1/2 * (log(ν0) - log(νn) ) + ( a0*log(2*b0) - an*log(2*bn) ) - ( n/2 * log(π))
    return r
end

function categoricalMLL(γ::AbstractVector{U}, T::AbstractVector) where {U<:Real}

    if length(γ) == 1
        lT = length(T)
        γsum = γ[1] * lT
        return (lgamma(γsum) - sum(lgamma.(ones(U, lT) * γ[1]))) + (sum(lgamma.(T .+ γ[1])) - lgamma(sum(T) + γsum))
    else
        return (lgamma(sum(γ)) - sum(lgamma.(γ))) + (sum(lgamma.(T + γ)) - lgamma(sum(T) + sum(γ)))
    end
end

function categoricalLPP(γ, T, x)
    return categoricalLPP(γ + T, x)
end

function categoricalLPP(γ::Vector, x::Int)
    return log( γ[x] ) - log( sum(γ) )
end

function dirichlet_postparam(α::AbstractVector, T::AbstractVector)
    return α + T
end

function dirichlet_postparam(α::AbstractFloat, T::AbstractVector)
    return T .+ (α/length(T))
end
