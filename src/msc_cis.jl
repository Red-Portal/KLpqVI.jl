
mutable struct MSC <: AdvancedVI.VariationalObjective
    z::Vector{Float64}
    hmc_params::Union{Nothing, NamedTuple{(:ϵ, :L), Tuple{Float64, Int64}}}
end

function MSC()
    return MSC(Array{Float64}(undef, 0), nothing)
end

function MSC_HMC(ϵ::Float64, L::Int64)
    return MSC(Array{Float64}(undef, 0), (ϵ=ϵ, L=L))
end

function init_state!(msc::MSC, rng::Random.AbstractRNG, q, n_mc)
    msc.z = rand(rng, q)
end

function cir(rng::Random.AbstractRNG,
             z0::AbstractVector,
             ℓπ::Function,
             q,
             n_samples::Int)
    z_tups = map(1:n_samples) do i
        _, z, _, logq = Bijectors.forward(rng, q)
        (z=z, logq=logq)
    end
    z    = [z_tup.z    for z_tup ∈ z_tups]
    logq = [z_tup.logq for z_tup ∈ z_tups]

    z   = vcat([z0], z)
    ℓq  = vcat(logpdf(q, z0), logq)
    ℓp  = map(ℓπ, z)
    ℓw  = ℓp - ℓq
    ℓZ  = StatsFuns.logsumexp(ℓw)
    w   = exp.(ℓw .- ℓZ) 
    idx = rand(rng, Categorical(w))
    z[idx]
end

function hmc(rng::Random.AbstractRNG,
             z0::AbstractVector,
             ℓπ::Function,
             q,
             n_samples::Int)
    z_tups = map(1:n_samples) do i
        _, z, _, logq = Bijectors.forward(rng, q)
        (z=z, logq=logq)
    end
    z    = [z_tup.z    for z_tup ∈ z_tups]
    logq = [z_tup.logq for z_tup ∈ z_tups]

    z   = vcat([z0], z)
    ℓq  = vcat(logpdf(q, z0), logq)
    ℓp  = map(ℓπ, z)
    ℓw  = ℓp - ℓq
    ℓZ  = StatsFuns.logsumexp(ℓw)
    w   = exp.(ℓw .- ℓZ) 
    idx = rand(rng, Categorical(w))
    z[idx]
end

function AdvancedVI.grad!(
    rng::Random.AbstractRNG,
    vo::MSC,
    alg::AdvancedVI.VariationalInference,
    q,
    logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult)

    q′ = if (q isa Distribution)
        AdvancedVI.update(q, θ)
    else
        q(θ)
    end

    vo.z = if isnothing(vo.hmc_params)
        cir(rng, vo.z, logπ, q′, alg.samples_per_step)
    else
        bijection = q.transform
        η0        = inv(bijection)(vo.z)
        grad_buf  = DiffResults.GradientResult(η0)
        ∂ℓπ∂η(η)  = begin
            gradient!(alg, η_ -> logπ(bijection(η_)), η, grad_buf)
            f_η  = DiffResults.value(grad_buf)
            ∇f_η = DiffResults.gradient(grad_buf)
            f_η, ∇f_η
        end

        η′, acc = hmc(rng, ∂ℓπ∂η, η0, vo.hmc_params.ϵ, vo.hmc_params.L)
        println(acc)
        bijection(η′)
    end

    f(θ) = if (q isa Distribution)
        -(Bijectors.logpdf(AdvancedVI.update(q, θ), vo.z))
    else
        -Bijectors.logpdf(q(θ), vo.z)
    end
    gradient!(alg, f, θ, out)
end
