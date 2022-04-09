
mutable struct MSC_RB <: AdvancedVI.VariationalObjective
    z::RV{Float64}
    iter::Int
end

function MSC_CISRB()
    z = RV(Array{Float64}(undef, 0), -Inf)
    return MSC_RB(z, 1)
end

function init_state!(msc::MSC_RB, rng::Random.AbstractRNG, rand_q, λ0, ℓπ, n_mc)
    z     = rand_q(rng, λ0)
    msc.z = RV{Float64}(z, ℓπ(z))
end

function grad!(
    rng::Random.AbstractRNG,
    vo::MSC_RB,
    ℓq,
    rand_q,
    ℓjac,
    ℓπ,
    λ::AbstractVector{<:Real},
    n_mc::Int,
    out::DiffResults.MutableDiffResult)
#=
    Christian Naesseth, Fredrik Lindsten, David Blei
    "Markovian Score Climbing: Variational Inference with KL(p||q)"
    Advances in Neural Information Processing Systems 33 (NeurIPS 2020)
=##

    ess      = 0
    rej_rate = 0 

    z, w, ℓw, ℓp = cis(rng, vo.z, ℓπ, λ, ℓq, rand_q, n_mc)
    acc_idx  = rand(rng, Categorical(w))
    vo.z     = RV(z[:,acc_idx], ℓp[acc_idx])
    ess      = 1/sum(w.^2)
    rej_rate = 1 - w[1]
    cent     = -dot(w, ℓw)

    f(λ′) = begin
       logqs = map(zᵢ -> ℓq(λ′, zᵢ), eachcol(z))
       -dot(w, logqs)
    end

    turing_gradient!(f, λ, out)

    vo.iter += 1

    (ess      = ess,
     crossent = cent,
     rej_rate = rej_rate)
end
