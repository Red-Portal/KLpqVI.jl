
Turing.@model radon(county, x, y) = begin
    ϵ    = eps(Float64)
    σ_a1 ~ Gamma(1, 50)
    σ_a2 ~ Gamma(1, 50)
    σ_y  ~ Gamma(1, 50)
    μ_a1 ~ Normal(0,1)
    μ_a2 ~ Normal(0,1)

    if(σ_a1 <  ϵ || σ_a2 <  ϵ || σ_y <  ϵ)
        Turing.@addlogprob! -Inf
        return
    end

    a1   ~ MvNormal(fill(μ_a1, 85), fill(σ_a1, 85))
    a2   ~ MvNormal(fill(μ_a2, 85), fill(σ_a2, 85))
    μ_y  = a1[county] + a2[county].*x
    y    ~ MvNormal(μ_y, σ_y)
end

function load_dataset(::Val{:radon})
    data   = FileIO.load(datadir("dataset", "radon.jld2"))
    x      = data["x"]
    y      = data["y"]
    u      = data["county"]
    county = data["county"]
    N      = data["N"]
    J      = data["J"]
    county, x, y
end

function hmc_params(task::Val{:radon})
    ϵ = 0.02
    L = 64
    ϵ, L
end

function run_task(prng::Random.AbstractRNG,
                  task::Val{:radon},
                  objective,
                  n_mc,
                  sleep_interval,
                  sleep_ϵ,
                  sleep_L;
                  show_progress=true)
    county, x, y = load_dataset(task)
    model        = radon(county, x, y)

    #AdvancedVI.setadbackend(:forwarddiff)
    #Turing.Core._setadbackend(Val(:forwarddiff))
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))
    #AdvancedVI.setadbackend(:zygote)
    #Turing.Core._setadbackend(Val(:zygote))

    i      = 1
    k_hist = []
    function plot_callback(ℓπ, q, objective_, klpq)
        stat = if(mod(i-1, 100) == 0)
            N  = floor(Int, 2^12)
            zs = rand(prng, q, N)
            ℓw = mapslices(zᵢ -> ℓπ(zᵢ) - logpdf(q, zᵢ), zs, dims=1)[1,:]
            ℓZ = StatsFuns.logsumexp(ℓw) - log(N)
            if(isnan(ℓZ))
                ℓZ = -Inf
            end
            (mll = ℓZ,)
        else
            NamedTuple()
        end
        i += 1
        stat
    end

    varinfo     = DynamicPPL.VarInfo(model)
    varsyms     = keys(varinfo.metadata)
    n_params    = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ varsyms])
    θ           = 0.1*randn(prng, n_params*2)
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
                     rhat_interval   = 0,
                     paretok_samples = 1024,
                     sleep_interval  = sleep_interval,
                     sleep_params    = (ϵ=sleep_ϵ, L=sleep_L,),
                     #optimizer       = AdvancedVI.TruncatedADAGrad(),
                     optimizer       = Flux.ADAM(0.01),
                     show_progress   = show_progress
                     )
    Dict.(pairs.(stats))
end

function sample_posterior(prng::Random.AbstractRNG,
                          task::Union{Val{:radon}})
    county, x, y = load_dataset(task)
    model        = radon(county, x, y)

    sampler = Turing.NUTS(1000, 0.8;
                          max_depth=8,
                          Δ_max=100.0,
                          metricT=AdvancedHMC.UnitEuclideanMetric)
    chain   = Turing.sample(model, sampler, 1000; progress=true)
    L       = median(chain[:n_steps][:,1])
    ϵ       = mean(chain[:step_size][:,1])
    @info "HMC Tuning Result on $(task)" ϵ=ϵ L=L
    chain
end
