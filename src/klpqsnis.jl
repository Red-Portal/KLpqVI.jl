
struct KLPQSNIS <: AdvancedVI.VariationalObjective end

init_state!(::KLPQSNIS, rng, q) = nothing

function AdvancedVI.grad!(
    rng::Random.AbstractRNG,
    klpq::KLPQSNIS,
    alg::AdvancedVI.VariationalInference{<:AdvancedVI.ForwardDiffAD},
    q,
    logπ,
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

    f(θ_) = if (q isa Distribution)
        q′      = AdvancedVI.update(q, θ_)
        nlogqs′ = [-logpdf(q′, z_tup[2]) for z_tup ∈ z_tups] 
        dot(w, nlogqs′)
    else
        q′      = q(θ_) 
        nlogqs′ = [-logpdf(q′, z_tup[2]) for z_tup ∈ z_tups] 
        dot(w, nlogqs′)
    end

    chunk_size = AdvancedVI.getchunksize(typeof(alg))
    chunk      = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config     = ForwardDiff.GradientConfig(f, θ, chunk)
    ForwardDiff.gradient!(out, f, θ, config)
end
