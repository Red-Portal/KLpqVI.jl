
mutable struct MSC_RB <: AdvancedVI.VariationalObjective
    z::RV{Float64}
    iter::Int
    hmc_freq::Int
    hmc_params::Union{Nothing, NamedTuple{(:ϵ, :L), Tuple{Float64, Int64}}}
end

function MSC_CISRB()
    z = RV(Array{Float64}(undef, 0), -Inf)
    return MSC_RB(z, 1, 0, nothing)
end

function MSC_CISRB(hmc_freq::Int, ϵ::Float64, L::Int64)
    z = RV(Array{Float64}(undef, 0), -Inf)
    return MSC_RB(z, 1, hmc_freq,  (ϵ=ϵ, L=L))
end

function init_state!(msc::MSC_RB, rng::Random.AbstractRNG, q, logπ, n_mc)
    z     = rand(rng, q)
    msc.z = RV{Float64}(z, logπ(z))
end

function grad!(
    rng::Random.AbstractRNG,
    vo::MSC_RB,
    alg::AdvancedVI.VariationalInference,
    q,
    logπ,
    ∇logπ,
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

    hmc_aug  = false
    ess      = 0
    hmc_acc  = 0
    rej_rate = 0 

    if(vo.hmc_freq > 0 && mod(vo.iter-1, vo.hmc_freq) == 0)
        vo.z, acc = hmc_step(rng, alg, q, logπ, ∇logπ, vo.z,
                             vo.hmc_params.ϵ, vo.hmc_params.L)
        hmc_aug = true
        hmc_acc = acc
    end

    z, w, ℓp = cis(rng, vo.z, logπ, q′, alg.samples_per_step)
    acc_idx  = rand(rng, Categorical(w))
    vo.z     = RV(z[:,acc_idx], ℓp[acc_idx])
    ess      = 1/sum(w.^2)
    rej_rate = 1 - w[1]

    ℓq   = logpdf.(Ref(q′), eachcol(z))
    cent = -dot(w, ℓp - ℓq)

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

    vo.iter += 1

    (ess      = ess,
     crossent = cent,
     hmc_aug  = hmc_aug,
     hmc_acc  = hmc_acc,
     rej_rate = rej_rate)
end
