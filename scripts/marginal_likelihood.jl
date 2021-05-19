
using DrWatson
@quickactivate "KLpqVI"

using Memoization
using ReverseDiff

using Plots, StatsPlots
using Flux
using ForwardDiff
using Zygote
using OnlineStats
using Random123
using ProgressMeter
using DelimitedFiles
using ThermodynamicIntegration
using NestedSamplers
using Measurements
using DiffResults
#using Suppressor

include(srcdir("KLpqVI.jl"))
include("task/task.jl")

@inbounds function stochastic_volatility_prior(θ, y)
    ϵ     = 1e-10
    T     = length(y)
    ϕ     = θ[1]
    σ     = θ[2]
    μ     = θ[3]
    h_std = θ[4:3+T]
    
    ℓprior  = 0.0
    ℓprior += logpdf(Uniform(-1+ϵ, 1-ϵ),              ϕ)
    ℓprior += logpdf(truncated(Cauchy(0, 5), ϵ, Inf), σ)
    ℓprior += logpdf(Cauchy(0, 10),                   μ)
    ℓprior += logpdf(MvNormal(T, 1.0),                h_std)
    ℓprior
end

@inbounds function stochastic_volalitility_sample(prng, y)
    ϵ        = 1e-10
    T        = length(y)
    θ        = Array{Float64}(undef, T+3)
    θ[1]     = rand(prng, Uniform(-1+ϵ, 1-ϵ))
    θ[2]     = rand(prng, truncated(Cauchy(0, 5), ϵ, Inf))
    θ[3]     = rand(prng, Cauchy(0, 10))
    θ[4:end] = rand(prng, MvNormal(T, 1.0))
    θ
end

@inbounds function stochastic_volatility_like(θ, y)
    T = length(y)
    ϵ = 1e-10

    T     = length(y)
    ϕ     = θ[1]
    σ     = θ[2]
    μ     = θ[3]
    h_std = θ[4:3+T]

    if(1 - ϕ^2 <= 0.0 || σ <= 0.0)
        return -Inf
    end

    h     = σ*h_std
    h′    = Array{eltype(θ)}(undef, T)
    h′[1] = h[1] / sqrt(1 - ϕ^2)
    for t in 2:T
        h′[t]  = h[t] + ϕ*h′[t-1]
    end
    σ_y = exp.((h′ .+ μ) / 2)

    if(any(x -> x <= 0.0 || isinf(x) || isnan(x), σ_y))
        return -Inf
    end
    logpdf(MvNormal(σ_y), y)
end

function radon_prior(θ, county, x, y)
    ϵ      = eps(Float64)
    σ_a1   = θ[1]
    σ_a2   = θ[2]
    σ_y    = θ[3]
    μ_a1   = θ[4]
    μ_a2   = θ[5]
    a1     = θ[6:5+85]
    a2     = θ[6+85:5+85+85]

    ℓprior  = 0.0
    ℓprior += logpdf(Gamma(1, 50), σ_a1)
    ℓprior += logpdf(Gamma(1, 50), σ_a2)
    ℓprior += logpdf(Gamma(1, 50), σ_y)

    if(σ_a1 <= 0.0 || σ_a2 <= 0.0 || σ_y <= 0.0)
        return -Inf
    end

    ℓprior += logpdf(Normal(0,1),  μ_a1)
    ℓprior += logpdf(Normal(0,1),  μ_a2)
    ℓprior += logpdf(MvNormal(fill(μ_a1, 85), fill(σ_a1, 85)), a1)
    ℓprior += logpdf(MvNormal(fill(μ_a2, 85), fill(σ_a2, 85)), a2)
    ℓprior
end

function radon_sample(prng, county, x, y)
    θ      = Array{Float64}(undef, 85+85+5)

    θ[1] = rand(prng, Gamma(1, 50))
    θ[2] = rand(prng, Gamma(1, 50))
    θ[3] = rand(prng, Gamma(1, 50))
    θ[4] = rand(prng, Normal(0, 1))
    θ[5] = rand(prng, Normal(0, 1))

    σ_a1   = θ[1]
    σ_a2   = θ[2]
    μ_a1   = θ[4]
    μ_a2   = θ[5]
    θ[6:5+85]       = rand(prng, MvNormal(fill(μ_a1, 85), fill(σ_a1, 85)))   
    θ[6+85:5+85+85] = rand(prng, MvNormal(fill(μ_a2, 85), fill(σ_a2, 85)))
    θ
end

function radon_like(θ, county, x, y)
    ϵ    = eps(Float64)
    σ_a1 = θ[1]
    σ_a2 = θ[2]
    σ_y  = θ[3]
    μ_a1 = θ[4]
    μ_a2 = θ[5]
    a1   = θ[6:5+85]
    a2   = θ[6+85:5+85+85]
    μ_y  = a1[county] + a2[county].*x
    logpdf(MvNormal(μ_y, σ_y), y)
end

@eval ThermodynamicIntegration begin
    function ∂ℓπ∂θ_reversediff(ℓπ, θ::AbstractVector)
        res = DiffResults.GradientResult(θ)
        tp  = ReverseDiff.GradientTape(ℓπ, θ)
        ReverseDiff.gradient!(res, tp, θ)
        #ctp, _ = Turing.Core.memoized_taperesult(ℓπ, θ)
        ReverseDiff.gradient!(res, tp, θ)
        return DiffResults.value(res), DiffResults.gradient(res)
    end

    function get_hamiltonian(metric, ℓπ, ::ThermInt{:ReverseDiff})
        ∂ℓπ∂θ(θ::AbstractVecOrMat) = ∂ℓπ∂θ_reversediff(ℓπ, θ)
        return Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
    end

    function sample_powerlogπ(powerlogπ, alg::ThermInt, x_init)
        D = length(x_init)
        metric = DiagEuclideanMetric(D)
        hamiltonian = get_hamiltonian(metric, powerlogπ, alg)

        initial_ϵ = find_good_stepsize(hamiltonian, x_init)
        integrator = Leapfrog(initial_ϵ)

        proposal = AdvancedHMC.NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

        samples, stats = sample(
            alg.rng,
            hamiltonian,
            proposal,
            x_init,
            alg.n_samples,
            adaptor,
            alg.n_warmup;
            verbose=false,
            progress=true,
        )
        return samples
    end
end

function thermodynamic()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    Random123.set_counter!(prng, 0)
    Random.seed!(0)

    #Turing.Core.setrdcache(true)
    #Turing.Core._setadbackend(Val(:reversediff))

    #ThermodynamicIntegration.set_adbackend(:Zygote) 
    #ThermodynamicIntegration.set_adbackend(:ForwardDiff) 
    ThermodynamicIntegration.set_adbackend(:ReverseDiff) 
    results   = Dict{Symbol, Float64}()
    n_burn    = 2048
    n_samples = 2048
    n_steps   = 32
    alg       = ThermodynamicIntegration.ThermInt(
        prng, ((1:n_steps) ./ n_steps) .^ 5;
        n_samples=n_samples,
        n_warmup=n_burn)

    y            = load_dataset(Val(:sv))
    sv_prior(θ)  = stochastic_volatility_prior(θ, y)
    sv_like(θ)   = stochastic_volatility_like(θ, y)
    θ_init       = stochastic_volalitility_sample(prng, y)
    θ_init[1:3] .= 0.0
    θ_init[2]    = 1.0
    logZ         = alg(sv_prior, sv_like, θ_init)#, TIParallelThreads())
    results[:sv] = logZ
    @info "results" logZ = results

    ThermodynamicIntegration.set_adbackend(:Zygote) 
    alg       = ThermodynamicIntegration.ThermInt(
        prng, ((1:n_steps) ./ n_steps) .^ 5;
        n_samples=n_samples,
        n_warmup=n_burn)

    county, x, y   = load_dataset(Val(:radon))
    rd_prior(θ) = radon_prior(θ, county, x, y)
    rd_like(θ)  = radon_like(θ, county, x, y)
    θ_init      = radon_sample(prng, county, x, y)
    logZ  = alg(rd_prior, rd_like, θ_init, TIParallelThreads())
    results[:radon] = logZ

    @info "results" logZ = results
    results
end

function nested()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    Random123.set_counter!(prng, 0)
    Random.seed!(0)

    bounds    = Bounds.MultiEllipsoid
    prop      = Proposals.Slice(slices=10)
    n_samples = 1000

    y           = load_dataset(Val(:sv))
    sv_prior(θ) = stochastic_volatility_prior(θ, y)
    sv_like(θ)  = stochastic_volatility_like(θ, y)
    model       = NestedModel(sv_like, sv_prior)
    θ_init      = stochastic_volalitility_sample(prng, y)
    sampler     = Nested(length(θ_init), length(θ_init)*10)
    state       = sample(model, sampler; dlogz=0.2)
    @info "sv" logZ = state.logz ± state.logzerr

    county, x, y = load_data(Val(:radon))
    rd_prior(θ)  = radon_prior(θ, county, x, y)
    rd_like(θ)   = radon_like(θ, county, x, y)
    θ_init       = radon_sample(prng, county, x, y)
    sampler      = Nested(length(θ_init), length(θ_init)*10)
    model        = NestedModel(sv_like, sv_prior)
    state        = sample(model, sampler; dlogz=0.2)
    @info "neuron" logZ = state.logz ± state.logzerr
end
