
Turing.@model eightschools(y, σ) = begin
    τ ~ truncated(Cauchy(0, 5), 0, Inf)
    μ ~ Normal(0, 10)
    θ ~ MvNormal(fill(μ, 8), fill(τ, 8))
    y ~ MvNormal(θ, σ)
end

function hmc_params(task::Val{:schools})
     ϵ = 0.15
     L = 64
     ϵ, L
end

function run_task(prng::Random.AbstractRNG,
                  task::Val{:schools},
                  objective,
                  n_mc,
                  sleep_interval,
                  sleep_ϵ,
                  sleep_L;
                  show_progress=true)
    AdvancedVI.setadbackend(:forwarddiff)
    Turing.Core._setadbackend(Val(:forwarddiff))

    y     = Float64[28,  8, -3,  7, -1,  1, 18, 12]
    σ     = Float64[15, 10, 16, 11,  9, 11, 10, 18]
    model = eightschools(y, σ)

    i      = 1
    function plot_callback(ℓπ, q, objective, klpq)
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

    n_iter      = 10000
    θ, q, stats = vi(model;
                     objective      = objective,
                     n_mc           = n_mc,
                     n_iter         = n_iter,
                     tol            = 0.0005,
                     callback       = plot_callback,
                     rng            = prng,
                     sleep_interval = sleep_interval,
                     sleep_params   = (ϵ=sleep_ϵ, L=sleep_L,),
                     rhat_interval   = 100,
                     paretok_samples = 128,
                     optimizer      = Flux.ADAM(0.01),
                     #optimizer      = AdvancedVI.TruncatedADAGrad(),
                     show_progress = show_progress
                     )
    # β = get_variational_mode(q, model, Symbol("β"))
    # α = get_variational_mode(q, model, Symbol("α"))
    # θ = vcat(β, α)
    Dict.(pairs.(stats))
end

function sample_posterior(prng::Random.AbstractRNG,
                          task::Val{:schools})
    y     = Float64[28,  8, -3,  7, -1,  1, 18, 12]
    σ     = Float64[15, 10, 16, 11,  9, 11, 10, 18]
    model = eightschools(y, σ)

    sampler = Turing.NUTS(1000, 0.8;
                          max_depth=8,
                          metricT=AdvancedHMC.UnitEuclideanMetric)
    chain   = Turing.sample(model, sampler, 1000)
    L       = median(chain[:n_steps][:,1])
    ϵ       = mean(chain[:step_size][:,1])
    @info "HMC Tuning Result on $(task)" ϵ=ϵ L=L
end
