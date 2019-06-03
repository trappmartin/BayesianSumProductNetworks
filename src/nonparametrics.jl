export InfiniteSumNode, slicesample!

"""
    InfiniteSumNode

An infinite mixture over Bayesian SPNs realized using a stick-breaking representation of a Dirichlet process.

Parameters:

* id::Symbol                        Id
* α::Float64                        Concentration parameter
* w::Vector{<:Real}                 Infinite collection of weights
* z::Vector{Int}                    Assignments of observations
* children::Vector                  Infinite collection of Bayesian SPNs
* ustar::Float64                    Minimum slice variable
* istar::Int                        Observation inducing minimum slice variable
* kstar::Int                        Cluster inducing minimum slice variable
* logpdfbase::Function              Log pdf of base-case: f(x::Matrix) -> Vector{Float64}
* spnprior::Function                Function to draw a region-graph SPN.

"""
mutable struct InfiniteSumNode

    id::Symbol
    α::Float64
    w::Vector{<:Real}
    z::Vector{Int}
    children::Vector{<:AbstractRegionGraphNode}
    ustar::Float64
    istar::Int
    kstar::Int
    logpbase::Function
    spnprior::Function

end

function logpdf(n::InfiniteSumNode, x::AbstractMatrix{T}) where {T<:Real}
    @assert !isempty(n.w)
    lp_ = logpdf.(n.children, Ref(x))
    lp_ = reduce(hcat, lp_)
    return lse(lp_ .+ log.(n.w[1:end-1])', dims=2)
end

## ############## ##
## Slice sampling ##
## ############## ##

"""
    slicesample!(spn::InfiniteSumNode, x::AbstractMatrix{<:Real})
    
Distributed slice sampler for infinite mixtures of Bayesian SPNs based on Hong Ge et al. Distributed inference for Dirichlet process mixture models, ICML, 2015.
"""
function slicesample!(spn::InfiniteSumNode, x::AbstractMatrix{<:Real})

    (N, D) = size(x)

    if isempty(spn.children)
        # New SPN.
        spn.z[:] = ones(Int, N)
    else
        _slicesample!(spn, x)
    end

    # Update parameters
    K = max(length(spn.children), maximum(spn.z))
    n = map(k -> sum(spn.z .== k), 1:K)
    active = findall(n .!= 0)
    n = map(Float64, n)

    M = sum(n .== 0) + 1
    n[n .== 0] .= spn.α/M
    spn.w = rand(Dirichlet(vcat(n, spn.α/M)))

    # Add new children?
    if K > length(spn.children)
        push!(spn.children, spn.spnprior())
    end

    for (k, child) in enumerate(spn.children)
        ancestralsampling!(child, x; obs = findall(spn.z .== k))
    end

    b = map(k -> rand(Beta(1, n[k])), active)
    u = spn.w[active] .* b

    spn.ustar = minimum(u)
    spn.kstar = active[argmin(u)]
    spn.istar = rand(findall(spn.z .== spn.kstar))

    return spn
end

function _slicesample!(spn::InfiniteSumNode, x::AbstractMatrix{<:Real})
    (N,D) = size(x)

    lpbase = spn.logpbase(x)
    lp = map(child -> logpdf(child, x), spn.children)

    Threads.@threads for i in 1:N
        local w = spn.w[spn.z[i]]
        local u = i == spn.istar ? spn.ustar : rand(Uniform(spn.ustar, w))

        # Resample z_i.
        local ids = findall(spn.w .>= u)
        local K = length(spn.children)
        local p = zeros(length(ids))
        for (j, k) in enumerate(ids)
            if k > K
                p[j] = exp(lpbase[i])
            else
                p[j] = exp(lp[k][i])
            end
        end

        spn.z[i] = randomindex(p)
    end
end
