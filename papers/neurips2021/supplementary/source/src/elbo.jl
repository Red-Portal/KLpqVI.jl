
struct ELBO <: AdvancedVI.VariationalObjective end

init_state!(::ELBO, ::Random.AbstractRNG, q, logπ, n_mc) = nothing

function grad!(
    rng::Random.AbstractRNG,
    vo::ELBO,
    alg::AdvancedVI.VariationalInference,
    q,
    logπ,
    ∇logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult)

    n_samples = alg.samples_per_step
    f(θ_) = begin
        q′ = if (q isa Distribution)
            AdvancedVI.update(q, θ_)
        else
            q(θ_)
        end
        _, z, logjac, _ = Bijectors.forward(rng, q′)
        res = (logπ(z) + logjac) / n_samples
        
        if q′ isa Bijectors.TransformedDistribution
            res += entropy(q′.dist)
        else
            res += entropy(q′)
        end

        for i = 2:n_samples
            _, z, logjac, _ = Bijectors.forward(rng, q′)
            res += (logπ(z) + logjac) / n_samples
        end
        -res
    end
    gradient!(alg, f, θ, out)
    NamedTuple()
end