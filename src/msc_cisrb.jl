
mutable struct MSC_RB <: AdvancedVI.VariationalObjective
    z::RV{Float64}
end

function MSC_CISRB()
    z = RV(Array{Float64}(undef, 0), -Inf)
    return MSC_RB(z)
end

function init_state!(msc::MSC_RB, rng::Random.AbstractRNG, q, logπ, n_mc)
    z     = rand(rng, q)
    msc.z = RV{Float64}(z, logπ(z))
end

function AdvancedVI.grad!(
    rng::Random.AbstractRNG,
    vo::MSC_RB,
    alg::AdvancedVI.VariationalInference,
    q,
    logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult)
#=
    Christian Naesseth, Fredrik Lindsten, David Blei
    "Markovian Score Climbing: Variational Inference with KL(p||q)"
    Advances in Neural Information Processing Systems 33 (NeurIPS 2020)
=##

    q′ = if (q isa Distribution)
        AdvancedVI.update(q, θ)
    else
        q(θ)
    end
    z, w, ℓp = cis(rng, vo.z, logπ, q′, alg.samples_per_step)
    acc_idx  = rand(rng, Categorical(w))
    vo.z     = RV(z[:,acc_idx], ℓp[acc_idx])

    f(θ) = begin
        q_θ = if (q isa Distribution)
            AdvancedVI.update(q, θ)
        else
            q(θ)
        end
        nlogq = map(zᵢ -> -logpdf(q_θ, zᵢ), eachcol(z))
        dot(nlogq, w)
    end
    gradient!(alg, f, θ, out)
end
