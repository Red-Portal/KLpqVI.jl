
struct ELBO <: AdvancedVI.VariationalObjective end

init_state!(::ELBO, ::Random.AbstractRNG, rand_q, λ0, ℓπ, n_mc) = nothing

function grad!(
    rng::Random.AbstractRNG,
    vo::ELBO,
    ℓq,
    rand_q,
    ℓjac,
    ℓπ,
    λ::AbstractVector{<:Real},
    n_mc::Int,
    out::DiffResults.MutableDiffResult)

    λ_stop = deepcopy(λ)
    f(λ′) = begin
        mapreduce(+, 1:n_mc) do _
            zᵢ = rand_q(rng, λ′)
            -ℓπ(zᵢ) + ℓq(λ_stop, zᵢ)
        end / n_mc
    end

    if (Turing.ADBackend() <: Turing.Core.ForwardDiffAD)
        ForwardDiff.gradient!(out, f, λ)
    elseif (Turing.ADBackend() <: Turing.Core.ReverseDiffAD)
        ReverseDiff.gradient!(out, f, λ)
    else
        elbo, back = Zygote.pullback(f, λ)
        ∇elbo      = back(one(elbo))[1]
        DiffResults.gradient!(out, ∇elbo)
        DiffResults.value!(   out, elbo)
    end

    nelbo = DiffResults.value(out)
    (elbo=-nelbo,)
end
