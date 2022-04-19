
struct SNIS <: AdvancedVI.VariationalObjective end

init_state!(::SNIS, rng, rand_q, λ0, ℓπ, n_mc) = nothing

function grad!(
    rng::Random.AbstractRNG,
    vo::SNIS,
    ℓq,
    ℓq_def,
    rand_q,
    rand_q_def,
    ℓjac,
    ℓπ,
    λ::AbstractVector{<:Real},
    n_mc::Int,
    out::DiffResults.MutableDiffResult
    )
    zs  = [rand_q_def(rng, λ) for i = 1:n_mc]
    ℓws = map(zs) do zᵢ
        ℓπ(zᵢ) - ℓq(λ, zᵢ)
    end
    ℓZ   = StatsFuns.logsumexp(ℓws)
    w   = exp.(ℓws .- ℓZ)
    ess = 1/sum(w.^2)

    f(λ_) = begin
        ∇ℓqs = [ℓq(λ_, zᵢ) for zᵢ ∈ zs] 
        -dot(w, ∇ℓqs)
    end
    turing_gradient!(f, λ, out)
    (ess=ess,)
end
