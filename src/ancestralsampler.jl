export ancestralsampling!

"""
    ancestralsampling!(root::AbstractRegionGraphNode, x::AbstractMatrix{T}; obs = 1:size(x,1))
    
Sample z_n assignments for each observation using ancestral sampling and update all parameters of the network in-place.
"""
function ancestralsampling!(root::AbstractRegionGraphNode, x::AbstractMatrix{T}; obs = 1:size(x,1)) where {T<:Real}

    (N,D) = size(x)
    nodes = topoligicalorder(root)
    nodes = filter(n -> !(n isa PartitionGraphNode), nodes)
    out = AxisArray(Vector{Matrix}(undef, length(nodes)), map(id, nodes))

    # Bottom-up path.
    for n in nodes
        logpdf!(n, x, out)
    end

    # Top-down path.
    _ancestralsampleassignments!(root, out, [collect(obs)], x)

    return root
end

function _ancestralsampleassignments!(n::RegionGraphNode,
                                      out::AxisArray{<:AbstractArray},
                                      obs::Vector{<:AbstractVector},
                                      x::AbstractMatrix)
    @assert length(obs) == length(n)

    (N, D) = size(x)
    couts = map(c -> _getchildlogpdf(c, out), children(n))
    Ch = size(n.logweights,1)
    T = eltype(out[n.id])
    clusters = Vector{Vector{Int}}(undef, Ch)
    for c in 1:length(clusters)
        clusters[c] = Vector{Int}()
    end

    out_ = _getchildlogpdf.(children(n), Ref(out))

    for k in 1:length(n)
        z = zeros(Int, length(obs[k]))
        Threads.@threads for i in 1:length(obs[k])
            @inbounds begin
                local ni = obs[k][i]
                local p_ = zeros(T, Ch)
                local j = 1
                for ccnt in 1:length(children(n))
                    local out_ = couts[ccnt]
                    for j_ in 1:size(out_,2)
                        p_[j] = hasscope(n) ? exp(out_[ni,j_] + n.logweights[j,k]) : exp(n.logweights[j,k])
                        j += 1
                    end
                end
                z[i] = randomindex(p_)
            end
        end

        for c in 1:length(clusters)
            obs_ = findall(z .== c)
            @assert obs_ isa Vector
            clusters[c] = obs[k][obs_]
        end

        @inbounds for j in 1:size(n.active,1)
            n.active[j,k] = j ∈ z
        end

        # Update weights.
        sstat = DirichletSufficientStats(map(j -> sum(z .== j), 1:size(n.logweights, 1)))
        w = rand(Dirichlet(postparams(n.prior, sstat)))
        w /= sum(w)
        @inbounds n.logweights[:,k] .= log.(w)
    end

    j = 1
    for child in children(n)
        obs_ = clusters[j:(j+length(child)-1)]
        @assert all(g -> all((i) -> (0 < i <= N), g), obs_) "$obs_"
        _ancestralsampleassignments!(child, out, obs_, x)
        j += length(child)
    end
end

function _ancestralsampleassignments!(n::PartitionGraphNode,
                                      out::AxisArray{<:AbstractArray},
                                      obs::Vector{<:AbstractVector},
                                      x::AbstractMatrix)

    (N,D) = size(x)
    clusters = Vector{Vector{Vector{Int}}}(undef, length(children(n)))
    for (c, child) in enumerate(children(n))
        clusters[c] = map(k -> Vector{Int}(), 1:length(child))
    end

    dims = tuple(map(c -> length(c), children(n))...)

    for k in 1:length(n)
        if !isempty(obs[k])
            @inbounds begin
                idxs = Base._ind2sub(dims, k)
                for (c, child) in enumerate(children(n))
                    append!(clusters[c][idxs[c]], obs[k])
                end
            end
        end
    end

    for (c, child) in enumerate(children(n))
        @assert all(g -> all((i) -> (0 < i <= N), g), clusters[c])
        _ancestralsampleassignments!(child, out, clusters[c], x)
    end
end

function _ancestralsampleassignments!(n::FactorizedDistributionGraphNode,
                                      out::AxisArray{<:AbstractArray},
                                      obs::Vector{<:AbstractVector},
                                      x::AbstractMatrix)
    (D,K) = size(n.sstats)
    fill!(n.obsVecs, false)

    #Threads.@threads 
    for d in 1:D
        for k in 1:length(n)
            if d == 1
                n.obsVecs[obs[k],k] .= true
            end
            x_ = x[obs[k],d]
            x_ = x_[.!isnan.(x_)]
            update!(n.sstats[d,k], x_)
        end
    end
end

function _ancestralsampleassignments!(n::FactorizedMixtureGraphNode,
                                      out::AxisArray{<:AbstractArray},
                                      obs::Vector{<:AbstractVector},
                                      x::AbstractMatrix)
    T = eltype(out[n.id])
    D = length(n.sstats)
    fill!(n.obsVecs, 0)

    for d in 1:D
        for k in 1:length(n)
            z = zeros(Int, length(obs[k]))
            for i in 1:length(obs[k])
                local ni = obs[k][i]
                @assert ni > 0 "$(obs), k: $k"
                local p_ = zeros(T, size(n.logweights[d],1))
                for m in 1:size(n.logweights[d],1)
                    θ = postparams(n.priors[d][m], n.sstats[d][m,k])
                    p_[m] = d ∈ scope(n) ? exp(_logpostpdf(n.likelihoods[d][m], vec(view(x, ni, d)), θ)[1]) : 1.0
                    p_[m] *= exp(n.logweights[d][m,k])
                end
                z[i] = randomindex(p_)
            end

            @inbounds n.obsVecs[obs[k],k] .= z

            for m in 1:size(n.sstats[d],1)
                obs_ = filter(i -> !isnan(x[i,d]), obs[k][z .== m])
                @inbounds update!(n.sstats[d][m,k], x[obs_,d])
            end

            sstat = DirichletSufficientStats(map(j -> sum(z .== j), 1:size(n.logweights[d], 1)))
            w = postparams(n.weightpriors[d], sstat)
            w /= sum(w)
            @inbounds n.logweights[d][:,k] .= log.(w)
        end
    end
end
