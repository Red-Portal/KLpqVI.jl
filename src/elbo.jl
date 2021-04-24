
function AdvancedVI.grad!(
    rng,
    vo,
    alg::AdvancedVI.VariationalInference,
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
    gradient!(alg, f, θ, out)
end

function (elbo::AdvancedVI.ELBO)(rng, alg, q, logπ, num_samples)
    return elbo(rng, alg, q, logπ, num_samples)
end
