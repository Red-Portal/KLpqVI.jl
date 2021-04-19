
function AdvancedVI.grad!(
    rng,
    vo,
    alg::AdvancedVI.VariationalInference{<:AdvancedVI.ForwardDiffAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
)
    f(θ_) = if (q isa Distribution)
        - vo(rng, alg, AdvancedVI.update(q, θ_), model, alg.samples_per_step)
    else
        - vo(rng, alg, q(θ_), model, alg.samples_per_step)
    end

    chunk_size = AdvancedVI.getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, θ, chunk)
    ForwardDiff.gradient!(out, f, θ, config)
end

function (elbo::AdvancedVI.ELBO)(rng, alg, q, logπ, num_samples)
    return elbo(rng, alg, q, logπ, num_samples)
end
