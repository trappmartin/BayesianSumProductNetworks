
abstract type AbstractRegionGraphNode end

function setscope!(n::AbstractRegionGraphNode, dims::Vector{Int})
    fill!(n.scopeVecs, false)
    n.scopeVecs[dims] .= true
end
nscope(d::AbstractRegionGraphNode) = sum(d.scopeVecs)
hasscope(d::AbstractRegionGraphNode) = any(d.scopeVecs)
scope(d::AbstractRegionGraphNode) = findall(d.scopeVecs)

function setscope!(d::AbstractRegionGraphNode, i::Int, s::Bool)
    d.scopeVecs[i] = s
end

length(d::AbstractRegionGraphNode) = size(d.obsVecs,2)
id(d::AbstractRegionGraphNode) = d.id

"""
    topoligicalorder(root::AbstractRegionGraphNode)

Return topological order of the nodes in the SPN.
"""
function topoligicalorder(root::AbstractRegionGraphNode)
  visitedNodes = Vector{AbstractRegionGraphNode}()
  visitNode!(root, visitedNodes)
  return visitedNodes
end

function visitNode!(node::AbstractRegionGraphNode, visitedNodes)
  # check if we have already visited this node
  if !(node in visitedNodes)

    # visit node
    if haschildren(node)
        for child in children(node)
            visitNode!(child, visitedNodes)
        end
    end
    push!(visitedNodes, node)
  end
end

"""
    Region node.

Parameters:

* ids::Symbol                                           Id
* scopeVecs::Vector{Bool}                               Active dimensions (D)
* obsVecs::Matrix{Bool}                                 Active observations (N x K)
* logweights::Matrix{<:Real}                            Log weights of sum nodes (Ch x K)
* prior::Dirichlet                                      Prior for sum nodes
* children::Vector{AbstractRegionGraphNode}             Children of region

"""
struct RegionGraphNode{T<:Real} <: AbstractRegionGraphNode
    id::Symbol
    scopeVecs::Vector{Bool}
    obsVecs::Matrix{Bool}
    logweights::Matrix{T}
    active::Matrix{Bool}
    prior::Dirichlet
    children::Vector{<:AbstractRegionGraphNode}
end

"""
    Partition node.

Parameters:

* ids::Symbol                                   Id
* scopeVecs::Vector{Bool}                       Active dimensions (D)
* obsVecs::Matrix{Bool}                         Active observations (N x K)
* prior::Dirichlet                              Prior on product nodes
* children::Vector{<:AbstractRegionGraphNode}   Child region nodes

"""
struct PartitionGraphNode <: AbstractRegionGraphNode
    id::Symbol
    scopeVecs::Vector{Bool}
    obsVecs::Matrix{Bool}
    prior::Dirichlet
    children::Vector{<:AbstractRegionGraphNode}
end

haschildren(d::RegionGraphNode) = !isempty(d.children)
children(d::RegionGraphNode) = d.children

function updatescope!(d::RegionGraphNode)
    updatescope!.(children(d))
    setscope!(d, scope(first(d.children)))
end

function _logsumexp(y::AbstractMatrix{T}, lw::AbstractMatrix{T}) where {T<:Real}
    r = zeros(T,size(lw,2),size(y,2))
    Threads.@threads for j in 1:size(y,2)
        for k in 1:size(lw,2)
            @inbounds begin
                yi_max = y[1,j] + lw[1,k]
                for i in 2:size(y,1)
                    yi_max = max(y[i,j]+lw[i,k], yi_max)
                end
                s = zero(T)
                for i in 1:size(y,1)
                    s += exp(y[i,j]+lw[i,k] - yi_max)
                end
                r[k,j] = log(s) + yi_max
            end
        end
    end
    return transpose(r)
end

function _getchildlogpdf(child::AbstractRegionGraphNode, out::AxisArray{V}) where {V<:AbstractMatrix}
    return out[id(child)]
end

function _getchildlogpdf(child::PartitionGraphNode, out::AxisArray{V}) where {V<:AbstractMatrix}
    childids = map(id, children(child))
    lp_ = out[childids]
    return reduce(_cross_prod, lp_)
end

"""
    logpdf(d::RegionGraphNode, x::AbstractMatrix{T})

Log pdf of a region node in a region graph.
"""
function logpdf(d::RegionGraphNode, x::AbstractMatrix{T}) where {T<:Real}
    lp_ = logpdf.(d.children, Ref(x))
    return _logsumexp(reduce(hcat, lp_)', d.logweights)
end

"""
    logpdf!(d::RegionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V})

Log pdf of a region node in a region graph. (in-place)
"""
function logpdf!(d::RegionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}) where {T<:Real,V<:AbstractMatrix}
    lp_ = _getchildlogpdf.(children(d), Ref(out))
    out[id(d)] = _logsumexp(reduce(hcat, lp_)', d.logweights)
    return out
end

"""
    logmllh!(n::RegionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}, y::AxisArray, d::Int)

Log marginal likelihood of a region node in a region graph for dimension d. (in-place)
"""
function logmllh!(n::RegionGraphNode,
                  x::AbstractMatrix{T},
                  out::AxisArray{V},
                  y::AxisArray,
                  d::Int) where {T<:Real,V<:AbstractVector}
    K = length(n)
    lp_ = out[id.(children(n))]
    lp_ = reduce(vcat, lp_) .* n.active
    out[id(n)] = vec(sum(lp_, dims = 1))
    return out
end

## Partition Graph Node

haschildren(d::PartitionGraphNode) = !isempty(d.children)
children(d::PartitionGraphNode) = d.children

function updatescope!(d::PartitionGraphNode)
    s_ = mapreduce(updatescope!, vcat, children(d))
    setscope!(d, unique(s_))
end

function _cross_prod(x1::AbstractMatrix{T}, x2::AbstractMatrix{T}) where {T<:Real}
    nx, ny = size(x1,2), size(x2,2)
    r = zeros(T, size(x1,1), nx*ny)
    Threads.@threads for j in 1:nx*ny
        @inbounds begin
            j1, j2 = Base._ind2sub((nx,ny), j)
            r[:,j] .= x1[:,j1] .+ x2[:,j2]
        end
    end
    return r
end

"""
    logpdf(d::PartitionGraphNode, x::AbstractMatrix{T})

Log pdf of a partition node in a region graph.
"""
function logpdf(d::PartitionGraphNode, x::AbstractMatrix{T}) where {T<:Real}
    childrn_ = children(d)
    lp_ = logpdf.(childrn_, Ref(x))
    return reduce(_cross_prod, lp_)
end

"""
    logpdf!(d::RegionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V})

Log pdf of a partition node in a region graph. (in-place)
"""
function logpdf!(d::PartitionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}) where {T<:Real,V<:AbstractMatrix}
    return out
end

"""
    logmllh!(n::PartitionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}, y::AxisArray, d::Int)

Log marginal likelihood of a partition node in a region graph for dimension d. (in-place)
"""
function logmllh!(n::PartitionGraphNode,
                  x::AbstractMatrix{T},
                  out::AxisArray{V},
                  y::AxisArray,
                  d::Int) where {T<:Real,V<:AbstractVector}
    # TODO: draw y_d ~ p(y_d | y_\d, x_d)
    cs = map(c -> nscope(c), children(n))
    α = postparams(n.prior, DirichletSufficientStats(cs))

    lp_ = map(c -> sum(out[id(c)]), children(n))
    lp_ += log.(α) .- sum(α)

    k = randomindex(map(exp, lp_))
    y[id(n)] = k
    cid = id(children(n)[k])

    out[id(n)] = zeros(length(n))
    dims = tuple(map(c -> length(c), children(n))...)

    for j in 1:length(n)
        @inbounds begin
            idxs = Base._ind2sub(dims, j)
            jc_ = idxs[k]
            out[id(n)][j] = out[cid][jc_]
        end
    end
    return out
end

"""
    Factorized distribution of the scope.

Parameters:

* ids::Symbol                               Id
* scopeVecs::Vector{Bool}                   Active dimensions (D)
* obsVecs::Matrix{Bool}                     Active observations (N x K)
* priors::Vector{Distribution}              Priors for each dimension (D)
* likelihoods::Vector{Distribution}         Likelihood functions (D)
* sstats::Matrix{AbstractSufficientStats}   Sufficient stats (D x K)

"""
struct FactorizedDistributionGraphNode <: AbstractRegionGraphNode
    id::Symbol
    scopeVecs::Vector{Bool}
    obsVecs::Matrix{Bool}
    priors::Vector{<:Distribution}
    likelihoods::Vector{<:Distribution}
    sstats::Matrix{<:AbstractSufficientStats}
end

haschildren(d::FactorizedDistributionGraphNode) = false
updatescope!(d::FactorizedDistributionGraphNode) = scope(d)
obs(d::FactorizedDistributionGraphNode, k::Int) = findall(d.obsVecs[:,k])

function _logpdf!(n::FactorizedDistributionGraphNode,
                  x::AbstractMatrix{T},
                  out::AbstractMatrix{V}) where {T<:Real,V<:Real}
    fill!(out, zero(V))
    ds = scope(n)
    #Threads.@threads
    for d in ds
        sstats_ = n.sstats[d,:]
        θ = postparams.(Ref(n.priors[d]), sstats_)
        lpd_ = _logpostpdf.(Ref(n.likelihoods[d]), Ref(view(x,:,d)), θ)
        out .+= reduce(hcat, lpd_)
    end
    return out
end

"""
    logpdf(d::FactorizedDistributionGraphNode, x::AbstractMatrix{T})

Log pdf of an atomic region node in a region graph.
"""
function logpdf(n::FactorizedDistributionGraphNode, x::AbstractMatrix{T}) where {T<:Real}
    (N, D) = size(x)
    K = length(n)
    lp_ = zeros(Float64, N, K)
    return _logpdf!(n, x, lp_)
end

"""
    logpdf!(d::FactorizedDistributionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V})

Log pdf of an atomic node in a region graph. (in-place)
"""
function logpdf!(n::FactorizedDistributionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}) where {T<:Real,V<:AbstractMatrix}
    (N, D) = size(x)
    K = length(n)

    i = findfirst(out.axes[1].val .== id(n))
    if !isassigned(out, i)
        out[id(n)] = zeros(Float64, N, K)
    end

    _logpdf!(n, x, out[id(n)])
    return out
end

function _logmllh!(n::FactorizedDistributionGraphNode,
                   x::AbstractMatrix{T},
                   out::AbstractVector{V},
                   d::Int) where {T<:Real,V<:Real}
    fill!(out, zero(V))
    K = length(n)

    sstats_ = n.sstats[d,:]
    θ = postparams.(Ref(n.priors[d]), sstats_)
    out .= map(k -> _logmllh(n.likelihoods[d], n.priors[d], x[obs(n,k),d], θ[k]), 1:K)
    return out
end

"""
    logmllh!(n::FactorizedDistributionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}, y::AxisArray, d::Int)

Log marginal likelihood of an atomic node in a region graph for dimension d. (in-place)
"""
function logmllh!(n::FactorizedDistributionGraphNode,
                  x::AbstractMatrix{T},
                  out::AxisArray{V},
                  y::AxisArray,
                  d::Int) where {T<:Real,V<:AbstractVector}
    (N,D) = size(x)
    K = length(n)

    i = findfirst(out.axes[1].val .== id(n))
    if !isassigned(out, i)
        out[id(n)] = zeros(Float64, K)
    end

    _logmllh!(n, x, out[id(n)], d)
    return out
end


"""
    Factorized mixture distribution of the scope.

Parameters:

* ids::Symbol                                       Id
* scopeVecs::Vector{Bool}                           Active dimensions (D)
* obsVecs::Vector{Matrix{Bool}}                     LV of active observations (N x K)
* priors::Vector{Vector{Distribution}}              Priors for each dimension D x (M)
* likelihoods::Vector{Vector{Distribution}}         Likelihood functions D x (m)
* sstats::Vector{Matrix{AbstractSufficientStats}}   Sufficient stats D x (M x K)

"""
struct FactorizedMixtureGraphNode <: AbstractRegionGraphNode
    id::Symbol
    scopeVecs::Vector{Bool}
    obsVecs::Matrix{Int}
    weightpriors::Vector{<:Dirichlet}
    logweights::Vector{Matrix{<:AbstractFloat}}
    priors::Vector{Vector{<:Distribution}}
    likelihoods::Vector{Vector{<:Distribution}}
    sstats::Vector{Matrix{<:AbstractSufficientStats}}
end

haschildren(d::FactorizedMixtureGraphNode) = false
updatescope!(d::FactorizedMixtureGraphNode) = scope(d)
length(d::FactorizedMixtureGraphNode) = size(first(d.sstats),2)

function obs(n::FactorizedMixtureGraphNode, k::Int, m::Int)
    if any(n.obsVecs[:,k] .== m)
        return findall(n.obsVecs[:,k] .== m)
    else
        return []
    end
end

"""
    In-place logsumexp for factorized mixture graph nodes.

Parameters:
* `y`: log-predictions of leaves in (N x K x M)
* `lw`: log weights in (M x K)
* `out`: intial out + log-sum-exp result in (N x K)

"""
function _logsumexp!(y::AbstractArray{T,3}, lw::AbstractMatrix{T}, out::AbstractMatrix{T}) where {T<:Real}
    Threads.@threads for j in 1:size(out,1)
        for k in 1:size(out,2)
            @inbounds begin
                yi_max = y[j,k,1] + lw[1,k]
                for i in 2:size(y,3)
                    yi_max = max(y[j,k,i]+lw[i,k], yi_max)
                end
                s = zero(T)
                for i in 1:size(y,3)
                    s += exp(y[j,k,i]+lw[i,k] - yi_max)
                end
                out[j,k] += log(s) + yi_max
            end
        end
    end
    return out
end

function _logpdf!(n::FactorizedMixtureGraphNode,
                  x::AbstractMatrix{T},
                  out::AbstractMatrix{V}) where {T<:Real,V<:Real}
    fill!(out, zero(V))
    ds = scope(n)
    for d in ds
        lp_ = Vector{Matrix}(undef, size(n.sstats[d],1))
        for m in 1:size(n.sstats[d],1)
            @inbounds begin
                sstats_ = n.sstats[d][m,:]
                θ = postparams.(Ref(n.priors[d][m]), sstats_)
                lpd_ = _logpostpdf.(Ref(n.likelihoods[d][m]), Ref(view(x,:,d)), θ)
                lp_[m] = reduce(hcat, lpd_)
            end
        end

        if length(lp_) > 1
            _logsumexp!(reduce((x,y) -> cat(x, y, dims=3), lp_), n.logweights[d], out)
        elseif length(lp_) == 1
            out[:,:] .= first(lp_)
        else
            out[:,:] .= -Inf
            @warn "No leaf distributions found..."
        end
    end
    return out
end

"""
    logpdf(d::FactorizedMixtureGraphNode, x::AbstractMatrix{T})

Log pdf of an atomic region node (mixture over likelihoods) in a region graph.
"""
function logpdf(n::FactorizedMixtureGraphNode, x::AbstractMatrix{T}) where {T<:Real}
    (N, D) = size(x)
    K = length(n)
    lp_ = zeros(Float64, N, K)
    return _logpdf!(n, x, lp_)
end

"""
    logpdf!(d::FactorizedMixtureGraphNode, x::AbstractMatrix{T}, out::AxisArray{V})

Log pdf of an atomic node (mixture over likelihoods) in a region graph. (in-place)
"""
function logpdf!(n::FactorizedMixtureGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}) where {T<:Real,V<:AbstractMatrix}
    (N, D) = size(x)
    K = length(n)

    i = findfirst(out.axes[1].val .== id(n))
    if !isassigned(out, i)
        out[id(n)] = zeros(Float64, N, K)
    end

    _logpdf!(n, x, out[id(n)])
    return out
end

function _logmllh!(n::FactorizedMixtureGraphNode,
                   x::AbstractMatrix{T},
                   out::AbstractVector{V},
                   d::Int) where {T<:Real,V<:Real}
    fill!(out, zero(V))
    K = length(n)

    sstats_ = n.sstats[d]
    for m in 1:size(sstats_,1)
        θ = postparams.(Ref(n.priors[d][m]), sstats_[m,:])
        out .+= map(k -> _logmllh(n.likelihoods[d][m], n.priors[d][m], x[obs(n,k,m),d], θ[k]), 1:K)
    end

    return out
end

"""
    logmllh!(n::FactorizedMixtureGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}, y::AxisArray, d::Int)

Log marginal likelihood of an atomic node (mixture over likelihoods) in a region graph for dimension d. (in-place)
"""
function logmllh!(n::FactorizedMixtureGraphNode,
                  x::AbstractMatrix{T},
                  out::AxisArray{V},
                  y::AxisArray,
                  d::Int) where {T<:Real,V<:AbstractVector}
    (N,D) = size(x)
    K = length(n)

    i = findfirst(out.axes[1].val .== id(n))
    if !isassigned(out, i)
        out[id(n)] = zeros(Float64, K)
    end

    _logmllh!(n, x, out[id(n)], d)
    return out
end
