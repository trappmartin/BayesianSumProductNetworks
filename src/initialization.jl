export randomscopes!

"""
    randomscopes!
    
Draw scopes from prior distribution.
"""
function randomscopes!(spn::AbstractRegionGraphNode, x::AbstractMatrix{<:Real})
    (N,D) = size(x)

    _randomscopes!(spn, collect(1:D))
end

function _randomscopes!(n::RegionGraphNode, dims::AbstractVector{Int})
    _randomscopes!.(children(n), Ref(dims))
    setscope!(n, dims)
end

function _randomscopes!(n::PartitionGraphNode, dims::AbstractVector{Int})
    K = length(children(n))
    α = postparams(n.prior, DirichletSufficientStats(K))
    prior = α / sum(α)

    z = map(d -> randomindex(prior), 1:length(dims))
    for (i,child) in enumerate(children(n))
        _randomscopes!(child, dims[z .== i])
    end
    setscope!(n, dims)
end

function _randomscopes!(n::FactorizedDistributionGraphNode, dims::AbstractVector{Int})
    setscope!(n, dims)
end

function _randomscopes!(n::FactorizedMixtureGraphNode, dims::AbstractVector{Int})
    setscope!(n, dims)
end