
"""
    meanfield([rng, ]model::Model)
Creates a mean-field approximation with multivariate normal as underlying distribution.
"""
function Turing.meanfield(rng::Random.AbstractRNG, model::DynamicPPL.Model)
    # setup
    varinfo = DynamicPPL.VarInfo(model)
    num_params = sum([size(varinfo.metadata[sym].vals, 1)
                      for sym ∈ keys(varinfo.metadata)])

    dists = vcat([varinfo.metadata[sym].dists for sym ∈ keys(varinfo.metadata)]...)

    num_ranges = sum([length(varinfo.metadata[sym].ranges)
                      for sym ∈ keys(varinfo.metadata)])
    ranges = Vector{UnitRange{Int}}(undef, num_ranges)
    idx = 0
    range_idx = 1
    for sym ∈ keys(varinfo.metadata)
        for r ∈ varinfo.metadata[sym].ranges
            ranges[range_idx] = idx .+ r
            range_idx += 1
        end

        # append!(ranges, [idx .+ r for r ∈ varinfo.metadata[sym].ranges])
        idx += varinfo.metadata[sym].ranges[end][end]
    end

    # initial params
    μ = randn(rng, num_params)
    σ = StatsFuns.softplus.(randn(rng, num_params))

    # construct variational posterior
    d = DistributionsAD.TuringDiagMvNormal(μ, σ)
    bs = inv.(Bijectors.bijector.(tuple(dists...)))
    b = Bijectors.Stacked(bs, ranges)

    return Bijectors.transformed(d, b)
end

# Overloading stuff from `AdvancedVI` to specialize for Turing
function AdvancedVI.update(d::DistributionsAD.TuringDiagMvNormal, μ, σ)
    return DistributionsAD.TuringDiagMvNormal(μ, σ)
end

function AdvancedVI.update(
    td::Bijectors.TransformedDistribution{<:DistributionsAD.TuringDiagMvNormal},
    θ::AbstractArray,
)
    μ, ω = θ[1:length(td)], θ[length(td) + 1:end]
    return AdvancedVI.update(td, μ, StatsFuns.softplus.(ω))
end
