export lse, lomax, randomindex

# Log-sum exp.
function lse(x::AbstractMatrix{<:Real}; dims = 2)
    m = maximum(x, dims = dims)
    v = exp.(x .- m)
    return log.(sum(v, dims = dims)) + m
end

# see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
function tpdf(μ::T, σ::T, ν::T, x::V) where {T<:Real,V<:Real}
    c = gamma(ν/2 + 1/2) / gamma(ν/2)
    c *= 1/(sqrt(ν*π)*σ)
    return c * (1 + 1/ν * ((x-μ)/σ)^2)^((ν+1)/2)
end

function tlogpdf(μ::T, σ::T, ν::T, x::V) where {T<:Real,V<:Real}
    c = lgamma(ν/2 + 1/2) - lgamma(ν/2)
    c -= log(sqrt(ν*π)*σ)
    return c + ((ν+1)/2)*log(1 + 1/ν * ((x-μ)/σ)^2)
end

@inline lomax(x, α, λ) = log(α) - log(λ) + -(α + 1) * log(1 + (x / λ))

function randomindex(p::AbstractVector{T}) where T<: Real
    return randomindex(p, rand(T))
end

function randomindex(p::AbstractVector{T}, rnd::T) where T<: Real

    csum = T(0.0)
    thres = rnd * sum(p)

    for i in 1:length(p)
        @inbounds begin
            csum += p[i]
            if csum >= thres
                return Int(i)
            end
        end
    end

    return Int(length(p))
end