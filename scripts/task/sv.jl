
Turing.@model stochastic_volatility(y) = begin
    T = length(y)
    ϵ = 1e-5

    ϕ ~ Uniform(-1+ϵ, 1-ϵ)
    σ ~ Gamma(1.2, 10.0)
    μ ~ Cauchy(0, 10)

    h_std ~ MvNormal(T, 1.0)
    h     = σ .* h_std
    h[1] /= sqrt(1 - ϕ^2)
    h   .+= μ
    for t in 2:T
        h[t] += ϕ * (h[t-1] - μ)
    end
    y ~ MvNormal(max.(exp.(h ./ 2), ϵ))
end

# Turing.@model stochastic_volatility(y) = begin
#     T = length(y)
#     ϵ = 1e-7

#     μ ~ Cauchy(0, 10)
#     ϕ ~ Uniform(-1+ϵ, 1-ϵ)
#     σ ~ Gamma(1.0, 10.0)

#     h = Vector{Real}(undef, T)
#     h[1] ~ Normal(μ, σ / sqrt(1 - ϕ^2))
#     y[1] ~ Normal(0, exp(h[1] / 2))
#     for t in 2:T
#         h[t] ~ Normal(μ + ϕ * (h[t-1] - μ), σ)
#         y[t] ~ Normal(0, exp(h[t] / 2))
#     end
# end

function load_dataset(::Val{:sv})
    data_raw, header = DelimitedFiles.readdlm(
        datadir("dataset", "stochastic-volatility.csv"), ',', header=true)
    y = map(x -> isa(x, Number) ? x : 0.1, data_raw[:,2])
    Float64.(y)
end

function hmc_params(task::Val{:sv})
    ϵ = 0.0001
    L = 16
    ϵ, L
end

function run_task(prng::Random.AbstractRNG,
                  task::Val{:sv},
                  adbackend,
                  objective,
                  n_mc,
                  sleep_freq,
                  sleep_ϵ,
                  sleep_L)
    data  = load_dataset(task)
    model = stochastic_volatility(data)

    AdvancedVI.setadbackend(adbackend)
    Turing.Core._setadbackend(Val(adbackend))

    k_hist  = []
    function plot_callback(logπ, q, objective_, klpq)
        zs   = [rand(prng, q) for i = 1:64]
        ℓw    = logπ.(zs) - logpdf.(Ref(q), zs)
        _, k = psis.psislw(ℓw)
        k    = k <= 10 ? k : 10
        push!(k_hist, k)
        (k=k,)
    end

    n_iter      = 10000
    θ, q, stats = vi(model;
                     objective    = objective,
                     n_mc         = n_mc,
                     n_iter       = n_iter,
                     tol          = 0.0005,
                     callback     = plot_callback,
                     rng          = prng,
                     sleep_freq   = sleep_freq,
                     uleep_params = (ϵ=sleep_ϵ, L=sleep_L,),
                     #optimizer    = AdvancedVI.TruncatedADAGrad(),
                     optimizer    = Flux.ADAM(0.01),
                     show_progress = true
                     )
end
