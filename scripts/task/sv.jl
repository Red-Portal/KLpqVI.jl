
Turing.@model stochastic_volatility(y, ::Type{F} = Float64) where {F} = begin
    T = length(y)
    ϵ = 1e-10

    ϕ ~ Uniform(-1, 1)
    σ ~ truncated(Cauchy(0, 5), 0, Inf)
    μ ~ Cauchy(0, 10)

    if(abs(ϕ) > 1-ϵ || σ < ϵ)
        Turing.@addlogprob! -Inf
        return
    end

    h_std ~ MvNormal(T, 1.0)
    h     = σ*h_std

    σ_y    = Array{F}(undef, T)
    h′     = Array{F}(undef, T)
    @inbounds h′[1]  = h[1] / sqrt(1 - ϕ^2)
    @inbounds σ_y[1] = exp((h′[1] + μ) / 2)

    @inbounds for t in 2:T
        h′[t]  = h[t] + ϕ*h′[t-1]
        σ_y[t] = exp((h′[t] + μ) / 2)
    end

    if(any(x -> x < ϵ || isinf(x), σ_y))
        Turing.@addlogprob! -Inf
        return
    end
    y   ~ MvNormal(σ_y)
end

function load_dataset(::Val{:sv})
    data_raw, header = DelimitedFiles.readdlm(
        datadir("dataset", "stochastic-volatility.csv"), ',', header=true)
    y = map(x -> isa(x, Number) ? x : 0.1, data_raw[:,2])
    Float64.(y)
end

function hmc_params(task::Val{:sv})
    ϵ = 0.015
    L = 64
    ϵ, L
end

function run_task(prng::Random.AbstractRNG,
                  task::Val{:sv},
                  objective,
                  n_mc,
                  sleep_interval,
                  sleep_ϵ,
                  sleep_L;
                  show_progress=true)
    data  = load_dataset(task)
    model = stochastic_volatility(data)

    #AdvancedVI.setadbackend(:forwarddiff)
    #Turing.Core._setadbackend(Val(:forwarddiff))
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))
    #AdvancedVI.setadbackend(:zygote)
    #Turing.Core._setadbackend(Val(:zygote))

    k_hist  = []
    function plot_callback(logπ, q, objective_, klpq)
        NamedTuple()
    end

    varinfo     = DynamicPPL.VarInfo(model)
    varsyms     = keys(varinfo.metadata)
    n_params    = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ varsyms])
    θ           = 0.1*randn(prng, n_params*2)

    # Ensure Eq[σ] is sufficiently positive.
    # Initial σ needs to be feasible when using HMC 
    θ[2]       += 1.0 

    q_init      = Turing.Variational.meanfield(model)
    q_init      = AdvancedVI.update(q_init, θ)

    n_iter      = 10000
    θ, q, stats = vi(model, q_init;
                     objective       = objective,
                     n_mc            = n_mc,
                     n_iter          = n_iter,
                     tol             = 0.0005,
                     callback        = plot_callback,
                     rng             = prng,
                     rhat_interval   = 100,
                     paretok_samples = 128,
                     sleep_interval  = sleep_interval,
                     sleep_params    = (ϵ=sleep_ϵ, L=sleep_L,),
                     #optimizer      = AdvancedVI.TruncatedADAGrad(),
                     optimizer       = Flux.ADAM(0.01),
                     show_progress   = show_progress
                     )
    Dict.(pairs.(stats))
end

function sample_posterior(prng::Random.AbstractRNG,
                          task::Union{Val{:sv}})
    data  = load_dataset(task)
    model = stochastic_volatility(data)

    sampler = Turing.NUTS(1000, 0.95;
                          max_depth=10,
                          Δ_max=100.0,
                          metricT=AdvancedHMC.UnitEuclideanMetric)
    chain   = Turing.sample(model, sampler, 1000; progress=true)
    L       = median(chain[:n_steps][:,1])
    ϵ       = mean(chain[:step_size][:,1])
    @info "HMC Tuning Result on $(task)" ϵ=ϵ L=L
end
