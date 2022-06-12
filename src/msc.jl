
mutable struct MSC <: AdvancedVI.VariationalObjective
    z::RV{Float64}
    iter::Int
end

function MSC()
    z = RV(Array{Float64}(undef, 0), -Inf)
    return MSC(z, 1)
end

function init_state!(msc::MSC, rng::Random.AbstractRNG, rand_q, λ0, ℓπ, n_mc)
    z     = rand_q(rng, λ0)
    msc.z = RV{Float64}(z, ℓπ(z))
end

function grad!(
    rng::Random.AbstractRNG,
    vo::MSC,
    ℓq,
    ℓq_def,
    rand_q,
    rand_q_def,
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
    rejected = false
    rej_rate = 0 

    z, w, _, ℓp = cis(rng, vo.z, ℓπ, λ, ℓq_def, rand_q_def, n_mc)
    ess         = 1/sum(w.^2)
    acc_idx     = rand(rng, Categorical(w))
    rejected    = (acc_idx == 1)
    rej_rate    = w[1]
    vo.z        = RV(z[:,acc_idx], ℓp[acc_idx])
    cent        = -vo.z.prob + ℓq(λ, vo.z.val)

    f(λ′) = -ℓq(λ′, vo.z.val)

    turing_gradient!(f, λ, out)

    (ess      = ess,
     crossent = cent,
     rejected = rejected,
     rej_rate = rej_rate)
end
