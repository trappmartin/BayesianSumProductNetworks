export gibbssamplescopes!

"""
    gibbssamplescopes!(spn::InfiniteSumNode, x::AbstractMatrix{T})
    
Apply Gibbs sampling of y_d assignments for each dimension at each components sub-SPN.
"""
function gibbssamplescopes!(spn::InfiniteSumNode,
                            x::AbstractMatrix{T}) where {T<:Real}
    for child in spn.children
        gibbssamplescopes!(child, x)
    end
end

"""
    gibbssamplescopes!(spn::AbstractRegionGraphNode, x::AbstractMatrix{T})
    
Gibbs sample all y_d assignments for each dimension and update all scopes in-place.
"""
function gibbssamplescopes!(spn::AbstractRegionGraphNode,
                            x::AbstractMatrix{T}) where {T<:Real}
    (N,D) = size(x)
    nodes = topoligicalorder(spn);
    partitionnodes = filter(n -> n isa PartitionGraphNode, nodes)
    dims = collect(1:D)

    shuffle!(dims)

    out = AxisArray(Vector{Vector}(undef, length(nodes)), map(id, nodes))
    y = AxisArray(zeros(Int, length(partitionnodes)), map(n -> id(n), partitionnodes))

    for d in dims

        # Remove dimension from scopes.
        @inbounds for n in nodes
            setscope!(n, d, false)
        end

        # Bottom-up path.
        for n in nodes
            logmllh!(n, x, out, y, d)
        end

        # Top-down assignment.
        _gibbsassignscope!(spn, y, d)

    end
end

function _gibbsassignscope!(n::RegionGraphNode,
                           y::AxisArray, d::Int)
    setscope!(n, d, true)
    for child in children(n)
        _gibbsassignscope!(child, y, d)
    end
end

function _gibbsassignscope!(n::PartitionGraphNode,
                           y::AxisArray, d::Int)
    setscope!(n, d, true)
    _gibbsassignscope!(children(n)[y[id(n)]], y, d)
end

function _gibbsassignscope!(n::FactorizedDistributionGraphNode,
                           y::AxisArray, d::Int)
    setscope!(n, d, true)
end

function _gibbsassignscope!(n::FactorizedMixtureGraphNode,
                           y::AxisArray, d::Int)
    setscope!(n, d, true)
end
