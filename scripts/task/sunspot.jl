
Turing.@model sunspot(y, N) = begin
#=
    From: Bayesian Models for Astrophysical Data, Cambridge Univ. Press
    (c) 2017,  Joseph M. Hilbe, Rafael S. de Souza and Emille E. O. Ishida 
=##

    ϵ = 1e-15
    ϕ ~ MvNormal(zeros(2), 100)
    θ ~ Gamma(1e-3, 1e+3)

    μ        = Array{Real}(undef, N)
    μ[1]     = y[1]
    μ[2:end] = exp.(ϕ[1] .+ ϕ[2]*y[1:end-1])

    p  = θ ./ (μ .+ θ)
    r  = θ

    if(!all(pᵢ -> pᵢ > 0 && pᵢ <= 1, p) || !(r > 0))
        Turing.@addlogprob! -Inf
        return
    end
    
    y .~ NegativeBinomial.(r, p)
end

function load_data(task::Val{:sunspot})
    data = readdlm(datadir("dataset", "sunspot.txt"), ',', skipstart=1)   
    y    = data[:,2]
end

function hmc_params(task::Val{:sunspot})
     ϵ = 0.15
     L = 64
     ϵ, L
end

function run_task(prng::Random.AbstractRNG,
                  task::Val{:sunspot},
                  objective,
                  n_mc,
                  sleep_interval,
                  sleep_ϵ,
                  sleep_L;
                  show_progress=true)
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))

    y     = load_data(task)
    N     = length(y)
    model = sunspot(y, N)

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
                          task::Val{:sunspot})
    y     = load_data(task)
    N     = length(y)
    model = sunspot(y, N)

    sampler = Turing.NUTS(1000, 0.8;
                          max_depth=8,
                          metricT=AdvancedHMC.UnitEuclideanMetric)
    chain   = Turing.sample(model, sampler, 1000)
    L       = median(chain[:n_steps][:,1])
    ϵ       = mean(chain[:step_size][:,1])
    @info "HMC Tuning Result on $(task)" ϵ=ϵ L=L
end
