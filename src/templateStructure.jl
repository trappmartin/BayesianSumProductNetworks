export templateLeaves, templatePartition, templateRegion

function templateLeaves(alpha_leaf_prior::Float64,
                        priors::AbstractVector{<:Distribution},
                        likelihoods::AbstractVector{<:Distribution},
                        sstats::AbstractVector{<:AbstractSufficientStats},
                        N::Int, D::Int, K::Int)
    ids = map(k -> gensym("factorization"), 1:K)
    scopeVec = zeros(Bool, D)
    obsVec = zeros(Bool, N, K)
    priors_ = deepcopy(priors)
    likelihoods_ = deepcopy(likelihoods)
    sstats_ = mapreduce(k -> deepcopy(sstats), hcat, 1:K)

    return FactorizedDistributionGraphNode(
                                    gensym("fact"),
                                    scopeVec,
                                    obsVec,
                                    priors_,
                                    likelihoods_,
                                    sstats_
                                   )
end

function templateLeaves(alpha_leaf_prior::Float64,
                        priors::Vector{<:AbstractVector{<:Distribution}},
                        likelihoods::Vector{<:AbstractVector{<:Distribution}},
                        sstats::Vector{<:AbstractVector{<:AbstractSufficientStats}},
                        N::Int, D::Int, K::Int)
    ids = map(k -> gensym("factorization"), 1:K)
    scopeVec = zeros(Bool, D)
    obsVec = zeros(Int, N, K)
    weightpriors = map(d -> Dirichlet(length(likelihoods[d]), alpha_leaf_prior), 1:D)
    logweights = map(d -> log.(rand(weightpriors[d], K)), 1:D)
    priors_ = deepcopy(priors)
    likelihoods_ = deepcopy(likelihoods)
    sstats_ = map(d -> mapreduce(k -> deepcopy(sstats[d]), hcat, 1:K), 1:D)

    return FactorizedMixtureGraphNode(
                                    gensym("factmixture"),
                                    scopeVec,
                                    obsVec,
                                    weightpriors,
                                    logweights,
                                    priors_,
                                    likelihoods_,
                                    sstats_
                                   )
end

function templatePartition(alpha_region_prior::Float64,
                           alpha_partition_prior::Float64,
                           alpha_leaf_prior::Float64,
                           priors_leaf::AbstractVector,
                           likelihoods::AbstractVector,
                           sstats::AbstractVector,
                           N::Int, D::Int,
                           K_sum::Int, K_prod::Int,
                           J::Int, K::Int,
                           depth::Int, maxdepth::Int)

    children = if depth == maxdepth
        map(k -> templateLeaves(alpha_leaf_prior, priors_leaf, likelihoods, sstats, N, D, K), 1:K_prod)
    else
        map(k -> templateRegion(alpha_region_prior, alpha_partition_prior, alpha_leaf_prior,
                                priors_leaf, likelihoods, sstats, N, D, K_sum, K_prod, J, K, depth+1, maxdepth), 1:K_prod)
    end

    K_ = mapreduce(child -> length(child), *, children)
    scopeVec = zeros(Bool, D)
    obsVec = zeros(Bool, N, K_)
    prior = Dirichlet(K_prod, alpha_partition_prior)

    return PartitionGraphNode(
                              gensym("partition"),
                              scopeVec,
                              obsVec,
                              prior,
                              children
                             )
end

function templateRegion(alpha_region_prior::Float64,
                        alpha_partition_prior::Float64,
                        alpha_leaf_prior::Float64,
                        priors_leaf::AbstractVector,
                        likelihoods::AbstractVector,
                        sstats::AbstractVector,
                        N::Int, D::Int,
                        K_sum::Int, K_prod::Int,
                        J::Int, K::Int,
                        depth::Int, maxdepth::Int; root = false)

    K_ = root ? 1 : K_sum
    children = if depth == maxdepth
        map(k -> templateLeaves(alpha_leaf_prior, priors_leaf, likelihoods, sstats, N, D, K), 1:J)
    else
        map(k -> templatePartition(alpha_region_prior, alpha_partition_prior, alpha_leaf_prior,
                                   priors_leaf, likelihoods, sstats, N, D, K_sum, K_prod, J, K, depth+1, maxdepth), 1:J)
    end

    Ch = sum(length.(children))
    scopeVec = zeros(Bool, D)
    obsVec = zeros(Bool, N, K_)
    prior = Dirichlet(Ch, alpha_region_prior)
    logweights = convert(Matrix, reshape(mapreduce(_ -> rand(prior), hcat, 1:K_), Ch, K_))
    active = zeros(Bool, size(logweights)...)
    @assert size(logweights) == (Ch, K_)

    return RegionGraphNode(
                           gensym("region"),
                            scopeVec,
                            obsVec,
                            logweights,
                            active,
                            prior,
                            children
                          )

end