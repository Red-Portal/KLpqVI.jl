
mutable struct MSC_SIMH <: AdvancedVI.VariationalObjective
    z::RV{Float64}
    iter::Int
    hmc_freq::Int
    hmc_params::Union{Nothing, NamedTuple{(:ϵ, :L), Tuple{Float64, Int64}}}
end

function MSC_SIMH()
    MSC_SIMH(RV{Float64}(Vector{Float64}(), 0), 1, 0, nothing)
end

function MSC_SIMH(hmc_freq::Int, ϵ::Real, L::Int)
    MSC_SIMH(RV{Float64}(Vector{Float64}(), 0), 1, hmc_freq, (ϵ=ϵ, L=L,))
end

function init_state!(msc::MSC_SIMH, rng, q, logπ, n_mc)
    z        = rand(rng, q)
    ℓπ       = logπ(z) 
    msc.z    = RV{Float64}(z, ℓπ)
end

function grad!(
    rng::Random.AbstractRNG,
    vo::MSC_SIMH,
    alg::AdvancedVI.VariationalInference,
    q,
    logπ,
    ∇logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult)

    n_samples = alg.samples_per_step

    q′ = if (q isa Distribution)
        AdvancedVI.update(q, θ)
    else
        q(θ)
    end

    hmc_aug  = false
    hmc_acc  = 0
    rej_rate = 0 

    if(vo.hmc_freq > 0 && mod(vo.iter-1, vo.hmc_freq) == 0)
        vo.z, acc = hmc_step(rng, alg, q, logπ, ∇logπ, vo.z,
                             vo.hmc_params.ϵ, vo.hmc_params.L)
        hmc_acc = acc
        hmc_aug = true
    end

    rs = Vector{Float64}(    undef, n_samples)
    ws = Vector{Float64}(    undef, n_samples)
    zs = Vector{RV{Float64}}(undef, n_samples)
    for i = 1:n_samples
        vo.z, α, acc = imh_kernel(rng, vo.z, logπ, q′)
        ws[i]        = vo.z.prob - logpdf(q′, vo.z.val)
        zs[i]        = vo.z 
        rs[i]        = 1-α
    end
    rej_rate = mean(rs)
    cent     = -mean(ws)

    f(θ) = begin
        q_θ = if (q isa Distribution)
            AdvancedVI.update(q, θ)
        else
            q(θ)
        end
        nlogq = map(zᵢ_rv -> -logpdf(q_θ, zᵢ_rv.val), zs)
        mean(nlogq)
    end
    gradient!(alg, f, θ, out)
    vo.iter += 1

    (hmc_aug  = hmc_aug,
     hmc_acc  = hmc_acc,
     crossent = cent,
     rej_rate = rej_rate)
end
