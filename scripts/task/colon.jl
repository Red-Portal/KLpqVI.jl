 
@eval Distributions begin
    function truncated(d::UnivariateDistribution, l::T, u::T) where {T <: Real}
        l < u || error("lower bound should be less than upper bound.")
        lcdf = isinf(l) ? zero(T) : cdf(d, l)
        ucdf = isinf(u) ? one(T)  : cdf(d, u)
        tp = ucdf - lcdf
        Truncated(d, promote(l, u, lcdf, ucdf, tp, log(tp))...)
    end
end

Turing.@model horseshoe_fake(X, y, N, D) = begin
    # Model used for correctly inferring support
    # i.e. support of `truncated` is not 
    ϵ  = eps(Float64)
    s  = 2
    ν  = 4
    m₀ = 3
    μ   = mean(y)
    σ   = (1/μ)*(1/(1 - μ))
    τ₀  = m₀ / (D - m₀) * σ / sqrt(N)

    α   ~ Normal(0,10) 
    τ   ~ Gamma(1.0, τ₀)
    λ   ~ Turing.filldist(Gamma(1.0,2.0), D)
    c²  ~ InverseGamma(ν/2, ν/2*s^2)
    λ′  = sqrt.(c²)*λ ./ sqrt.(c² .+ τ.^2*λ.^2)
    σ_β = τ*λ′

    β  ~ MvNormal(zeros(D), σ_β)
    y .~ Turing.BernoulliLogit.(X*β .+ α)
end

Turing.@model horseshoe(X, y, N, D) = begin
    ϵ   = eps(Float64)
    s   = 2
    ν   = 4
    m₀  = 3
    μ   = mean(y)
    σ   = (1/μ)*(1/(1 - μ))
    τ₀  = m₀ / (D - m₀) * σ / sqrt(N)

    α   ~ Normal(0,10) 
    τ   ~ truncated(Cauchy(0.0, τ₀), 0.0, Inf)
    λ   ~ Turing.filldist(truncated(Cauchy(0.0, 1.0), 0.0, Inf), D)
    c²  ~ InverseGamma(ν/2, ν/2*s^2)
    λ′  = sqrt.(c²)*λ ./ sqrt.(c² .+ τ.^2*λ.^2)
    σ_β = τ*λ′

    # if(!all(σ_β .> ϵ))
    #     Turing.@addlogprob! -Inf
    #     return
    # end

    β  ~ MvNormal(zeros(D), σ_β)
    y .~ Turing.BernoulliLogit.(X*β .+ α)
end

function load_dataset(::Val{:colon})
    data   = MAT.matread(datadir("dataset", "colon.mat"))
    data_x = data["X"]
    data_y = (data["Y"][:,1] .+ 1) / 2
    data_x, data_y
end

function hmc_params(task::Val{:colon})
    ϵ = 0.0025
    L = 32
    ϵ, L
end

function run_task(prng::Random.AbstractRNG,
                  task::Val{:colon},
                  objective,
                  n_mc,
                  sleep_interval,
                  sleep_ϵ,
                  sleep_L;
                  show_progress=true)
    data_x, data_y = load_dataset(task)
    x_train, y_train, x_test, y_test =  prepare_dataset(prng, data_x, data_y; ratio=0.8)
    model = horseshoe(x_train, y_train, size(data_x,1), size(data_x,2))
    # posterior = FileIO.load(datadir("posterior", "colon.jld2"), "samples")
    # posterior = Array(posterior[1:4:end, ]')
    # ℓπ, _     = make_logjoint(prng, model)
    # ℓp        = ℓπ.(eachcol(posterior))
    # ∫pℓp      = mean(ℓp) 

    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))

    k_hist = []
    function plot_callback(ℓπ, q, objective_, klpq)
        β  = get_variational_mode(q, model, Symbol("β"))
        α  = get_variational_mode(q, model, Symbol("α"))
        λ  = get_variational_mode(q, model, Symbol("λ"))
        c² = get_variational_mode(q, model, Symbol("c²"))[1]
        τ  = get_variational_mode(q, model, Symbol("τ"))[1]

        λ′  = sqrt(c²)*λ ./ sqrt.(c² .+ τ.^2*λ.^2)
        σ_β = τ*λ′
        s   = x_test*β .+ α
        p   = StatsFuns.logistic.(s)
        ll  = sum(logpdf.(Turing.BernoulliLogit.(s), y_test))

        #klpq = ∫pℓp - mean(logpdf.(Ref(q), eachcol(posterior)))

        (ll=ll,)# klpq=klpq)# beta=β, sigma=σ_β)
    end

    model_fake  = horseshoe_fake(x_train, y_train, size(data_x, 1), size(data_x, 2))
    varinfo     = DynamicPPL.VarInfo(model_fake)
    varsyms     = keys(varinfo.metadata)
    n_params    = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ varsyms])
    θ           = 0.1*randn(prng, n_params*2)
    q_init      = Turing.Variational.meanfield(model_fake)
    q_init      = AdvancedVI.update(q_init, θ)

    n_iter      = 10000
    θ, q, stats = vi(model, q_init;
                     objective        = objective,
                     n_mc             = n_mc,
                     n_iter           = n_iter,
                     tol              = 0.0005,
                     callback         = plot_callback,
                     rng              = prng,
                     rhat_interval    = 100,
                     paretok_samples  = 0,
                     paretok_interval = 10,
                     sleep_interval   = sleep_interval,
                     sleep_params     = (ϵ=sleep_ϵ, L=sleep_L,),
                     optimizer        = Flux.ADAM(1e-3),
                     show_progress    = show_progress
                     )
    q
    #Dict.(pairs.(stats))
end

function sample_posterior(prng::Random.AbstractRNG,
                          task::Union{Val{:colon}})
    data_x, data_y = load_dataset(task)
    model          = horseshoe(data_x, data_y, size(data_x,1), size(data_x, 2))

    sampler = Turing.NUTS(1000, 0.99;
                          max_depth=10,
                          Δ_max=100.0,
                          metricT=AdvancedHMC.UnitEuclideanMetric)
    chain   = Turing.sample(model, sampler, 1000; progress=true)
    L       = median(chain[:n_steps][:,1])
    ϵ       = mean(chain[:step_size][:,1])
    @info "HMC Tuning Result on $(task)" ϵ=ϵ L=L
    chain
end

function sample_posterior_full(prng::Random.AbstractRNG,
                               task::Union{Val{:colon}})
    data_x, data_y = load_dataset(task)
    model          = horseshoe(data_x, data_y, size(data_x,1), size(data_x, 2))

    sampler = Turing.NUTS(4000, 0.99;
                          max_depth=15, 
                          metricT=AdvancedHMC.DiagEuclideanMetric)
    chain   = Turing.sample(model, sampler, 10000; progress=true)
    L       = median(chain[:n_steps][:,1])
    ϵ       = mean(chain[:step_size][:,1])
    @info "HMC Tuning Result on $(task)" ϵ=ϵ L=L
    chain
end
