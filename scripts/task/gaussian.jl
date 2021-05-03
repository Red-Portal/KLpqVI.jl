
function create_gaussian(prng::Random.AbstractRNG,
                         ν::Real,
                         n_dims::Int;
                         correlated::Bool=false)
    Σ = if correlated
        Σdist = Wishart(n_dims*ν, diagm(fill(1/ν, n_dims)))
        rand(prng, Σdist)
    else
        diagm(exp.(randn(prng, n_dims)))
    end
    μ = randn(prng, n_dims)
    MvNormal(μ, Σ)
end

function load_dataset(task::Val{:gaussian})
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);

    n_dims = 100
    create_gaussian(prng, 0.0, n_dims; correlated=false)
end

function load_dataset(task::Val{:gaussian_correlated})
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);

    ν      = 10.0
    n_dims = 100
    create_gaussian(prng, ν, n_dims; correlated=true)
end

Turing.@model gaussian(μ, Σ) = begin
    z ~ MvNormal(μ, Σ)
end

function run_task(prng::Random.AbstractRNG,
                  task::Union{Val{:gaussian},Val{:gaussian_correlated}},
                  objective,
                  n_mc,
                  sleep_interval,
                  sleep_ϵ,
                  sleep_L;
                  show_progress=true)
    p     = load_dataset(task)
    model = gaussian(p.μ, p.Σ)

    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))

    k_hist  = []
    function plot_callback(logπ, q, objective_, klpq)
        μ  = mean(q.dist)
        Σ  = cov(q.dist)
        kl = kl_divergence(p, MvNormal(μ, Σ))
        (kl=kl,)
    end

    n_iter      = 10000
    θ, q, stats = vi(model;
                     objective       = objective,
                     n_mc            = n_mc,
                     n_iter          = n_iter,
                     tol             = 0.0005,
                     callback        = plot_callback,
                     rng             = prng,
                     sleep_interval  = sleep_interval,
                     sleep_params    = (ϵ=sleep_ϵ, L=sleep_L,),
                     rhat_interval   = 100,
                     paretok_samples = 128,
                     optimizer       = Flux.ADAM(0.01),
                     show_progress   = show_progress
                     )
    Dict.(pairs.(stats))
end

function hmc_params(task::Val{:gaussian_correlated})
     ϵ = 4.0
     L = 16
     ϵ, L
end

function sample_posterior(prng::Random.AbstractRNG,
                          task::Union{Val{:gaussian},
                                      Val{:gaussian_correlated},})
    p     = load_dataset(task)
    model = gaussian(p.μ, p.Σ)

    sampler = Turing.NUTS(1000, 0.8;
                          max_depth=8,
                          metricT=AdvancedHMC.UnitEuclideanMetric)
    chain   = Turing.sample(model, sampler, 1000)
    L       = median(chain[:n_steps][:,1])
    ϵ       = mean(chain[:step_size][:,1])
    @info "HMC Tuning Result on $(task)" ϵ=ϵ L=L
end
