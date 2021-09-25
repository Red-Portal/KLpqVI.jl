
struct SNIS <: AdvancedVI.VariationalObjective end

init_state!(::SNIS, rng, q, logπ, n_mc) = nothing

function grad!(
    rng::Random.AbstractRNG,
    snis::SNIS,
    alg::AdvancedVI.VariationalInference,
    q,
    logπ,
    ∇logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
)
    n_samples = alg.samples_per_step
    q′        = if (q isa Distribution)
        AdvancedVI.update(q, θ)
    else
        q(θ)
    end
    z_tups = [Bijectors.forward(rng, q′) for i = 1:n_samples]
    z      = [z_tup[2] for z_tup ∈ z_tups]
    logws  = map(z_tups) do z_tup
        logqᵢ = z_tup[4]
        logπ(z_tup[2]) - logqᵢ
    end
    logZ = StatsFuns.logsumexp(logws)
    w    = exp.(logws .- logZ)
    ess  = 1/sum(w.^2)

    f(θ_) = if (q isa Distribution)
        q′      = AdvancedVI.update(q, θ_)
        nlogqs′ = [-logpdf(q′, z_tup[2]) for z_tup ∈ z_tups] 
        dot(w, nlogqs′)
    else
        q′      = q(θ_) 
        nlogqs′ = [-logpdf(q′, z_tup[2]) for z_tup ∈ z_tups] 
        dot(w, nlogqs′)
    end
    gradient!(alg, f, θ, out)
    (ess=ess,)
end
