
mutable struct MSC_PIMH <: AdvancedVI.VariationalObjective
    zs::Vector{RV{Float64}}
    iter::Int
    hmc_freq::Int
    hmc_params::Union{Nothing, NamedTuple{(:ϵ, :L), Tuple{Float64, Int64}}}
end

function MSC_PIMH()
    MSC_PIMH(Array{RV{Float64}}(undef, 0), 1, 0, nothing)
end

function MSC_PIMH(hmc_freq::Int, ϵ::Real, L::Int)
    MSC_PIMH(Array{RV{Float64}}(undef, 0), 1, hmc_freq, (ϵ=ϵ, L=L,))
end

function init_state!(msc::MSC_PIMH, rng, q, logπ, n_mc)
    zs     = [rand(rng, q) for i = 1:n_mc]
    ℓπs    = logπ.(zs) 
    msc.zs = RV{Float64}.(zs, ℓπs)
end


function imh_kernel(rng::Random.AbstractRNG,
                    z_rv::RV,
                    ℓπ::Function,
                    q)
    _, z′, _, ℓq′ = Bijectors.forward(rng, q)

    ℓw  = z_rv.prob - logpdf(q, z_rv.val)
    ℓp′ = ℓπ(z′)
    ℓw′ = ℓp′ - ℓq′
    α   = min(1.0, exp(ℓw′ - ℓw))
    if(rand(rng) < α)
        RV(z′, ℓp′), α, true
    else
        z_rv, α, false
    end
end

function AdvancedVI.grad!(
    rng::Random.AbstractRNG,
    vo::MSC_PIMH,
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
        hmc_acc_sum = 0.0
        for i = 1:length(vo.zs)
            vo.zs[i], acc = hmc_step(rng, alg, q, logπ, ∇logπ, vo.zs[i],
                                     vo.hmc_params.ϵ, vo.hmc_params.L)
            hmc_acc_sum += acc
        end
        hmc_acc = hmc_acc_sum / length(vo.zs)
        hmc_aug = true
    end

    rs = Vector{Float64}(undef, n_samples)
    for i = 1:length(vo.zs)
        z, α, acc   = imh_kernel(rng, vo.zs[i], logπ, q′)
        vo.zs[i]    = z 
        rs[i]       = 1-α
    end
    rej_rate = mean(rs)

    f(θ) = begin
        q_θ = if (q isa Distribution)
            AdvancedVI.update(q, θ)
        else
            q(θ)
        end
        nlogq = map(zᵢ_rv -> -logpdf(q_θ, zᵢ_rv.val), vo.zs)
        mean(nlogq)
    end
    gradient!(alg, f, θ, out)
    vo.iter += 1

    (hmc_aug  = hmc_aug,
     hmc_acc  = hmc_acc,
     rej_rate = rej_rate)
end
