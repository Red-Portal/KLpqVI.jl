
mutable struct MSC <: AdvancedVI.VariationalObjective
    z::RV{Float64}
    iter::Int
    hmc_freq::Int
    hmc_params::Union{Nothing, NamedTuple{(:ϵ, :L), Tuple{Float64, Int64}}}
end

function MSC_CIS()
    z = RV(Array{Float64}(undef, 0), -Inf)
    return MSC(z, 1, 0, nothing)
end

function MSC_CIS(hmc_freq::Int, ϵ::Float64, L::Int64)
    z = RV(Array{Float64}(undef, 0), -Inf)
    return MSC(z, 1, hmc_freq,  (ϵ=ϵ, L=L))
end

function MSC_HMC(ϵ::Float64, L::Int64)
    z = RV(Array{Float64}(undef, 0), -Inf)
    return MSC(z, 1, 1,  (ϵ=ϵ, L=L))
end

function init_state!(msc::MSC, rng::Random.AbstractRNG, q, logπ, n_mc)
    z     = rand(rng, q)
    msc.z = RV{Float64}(z, logπ(z))
end

function AdvancedVI.grad!(
    rng::Random.AbstractRNG,
    vo::MSC,
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

    if(vo.hmc_freq > 0 && mod(vo.iter-1, vo.hmc_freq) == 0)
        vo.z = hmc_step(rng, alg, q, logπ, vo.z,
                        vo.hmc_params.ϵ, vo.hmc_params.L)
    end

    vo.z = if vo.hmc_freq != 1
        z, w, ℓp = cis(rng, vo.z, logπ, q′, alg.samples_per_step)
        acc_idx  = rand(Categorical(w))
        RV(z[:,acc_idx], ℓp[acc_idx])
    else
        hmc_step(rng, alg, q, logπ, vo.z,
                 vo.hmc_params.ϵ, vo.hmc_params.L)
    end

    f(θ) = if (q isa Distribution)
        -(Bijectors.logpdf(AdvancedVI.update(q, θ), vo.z.val))
    else
        -Bijectors.logpdf(q(θ), vo.z.val)
    end
    gradient!(alg, f, θ, out)
    vo.iter += 1
end
