
init_state!(::AdvancedVI.ELBO, ::Random.AbstractRNG, q, logπ, n_mc) = nothing

function AdvancedVI.grad!(
    rng,
    vo,
    alg::AdvancedVI.VariationalInference,
    q,
    logπ,
    ∇logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
)
    f(θ_) = if (q isa Distribution)
        - vo(rng, alg, AdvancedVI.update(q, θ_), logπ, alg.samples_per_step)
    else
        - vo(rng, alg, q(θ_), logπ, alg.samples_per_step)
    end
    gradient!(alg, f, θ, out)
    NamedTuple()
end

function (elbo::AdvancedVI.ELBO)(rng, alg, q, logπ, num_samples)
    return elbo(rng, alg, q, logπ, num_samples)
end
