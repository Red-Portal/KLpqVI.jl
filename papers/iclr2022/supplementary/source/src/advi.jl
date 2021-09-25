
initialize_state!(::AdvancedVI.ELBO, rng, q) = nothing

function AdvancedVI.grad!(
    rng::Random.AbstractRNG,
    vo::AdvancedVI.ELBO,
    alg::AdvancedVI.VariationalInference{<:AdvancedVI.ForwardDiffAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult
)
    f(θ_) = if (q isa Distribution)
        - vo(rng, alg, AdvancedVI.update(q, θ_), model, alg.samples_per_step)
    else
        - vo(rng, alg, q(θ_), model, alg.samples_per_step)
    end

    chunk_size = AdvancedVI.getchunksize(typeof(alg))
    chunk      = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config     = ForwardDiff.GradientConfig(f, θ, chunk)
    ForwardDiff.gradient!(out, f, θ, config)
end
