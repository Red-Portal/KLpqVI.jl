
Turing.@model logistic_regression(X, y, d) = begin
    ϵ  = 1e-10
    τ  ~ truncated(Normal(0, 1.0), 0, Inf)
    σ  ~ truncated(Normal(0, 1.0), 0, Inf)

    if(τ < ϵ || σ < ϵ)
        Turing.@addlogprob! -Inf   
        return
    end

    β  ~ MvNormal(d, τ)
    α  ~ Normal(0, σ)
    s  = X*β .+ α
    y .~ Turing.BernoulliLogit.(s)
end

function load_dataset(::Val{:german})
    dataset = DelimitedFiles.readdlm(
        datadir("dataset", "german.data-numeric"))
    data_x  = dataset[:, 1:end-1,]
    data_y  = dataset[:, end] .- 1
    data_x, data_y
end

function load_dataset(::Val{:pima})
    dataset = DelimitedFiles.readdlm(
        datadir("dataset", "pima-indians-diabetes.csv"), ',', skipstart=1)
    data_x  = dataset[:, 1:end-1,]
    data_y  = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:heart})
    dataset = DelimitedFiles.readdlm(
        datadir("dataset", "heart-disease.csv"), ',', skipstart=1)
    data_x  = dataset[:, 1:end-1,]
    data_y  = dataset[:, end]
    data_x, data_y
end


function prepare_dataset(prng::Random.AbstractRNG,
                         data_x::AbstractMatrix,
                         data_y::AbstractVector;
                         ratio::Real=0.9)
    n_data      = size(data_x, 1)
    shuffle_idx = Random.shuffle(1:n_data)
    data_x      = data_x[shuffle_idx,:]
    data_y      = data_y[shuffle_idx]

    n_train = floor(Int, n_data*ratio)
    x_train = data_x[1:n_train, :]
    y_train = data_y[1:n_train]
    x_test  = data_x[n_train+1:end, :]
    y_test  = data_y[n_train+1:end]
    x_train, y_train, x_test, y_test
end

# function sample_posterior(prng::Random.AbstractRNG, task::Val{:pima})
#     data_x, data_y = fetch_dataset(task)
#     n_dims = size(data_x, 2)
#     n_data = size(data_x, 1)

#     model  = logistic_regression(data_x,
#                                  data_y,
#                                  n_data, 
#                                  n_dims,
#                                  1.0)
#     sampler = Turing.NUTS(4096, 0.8; max_depth=8)
#     chain   = Turing.sample(model, sampler, 16_384)
#     L       = median(chain[:n_steps][:,1])
#     ϵ       = mean(chain[:step_size][:,1])
#     params  = Array(chain)
#     JLD.save(datadir("posterior", "pima.jld"),
#              "samples",       params,
#              "hmc_step_size", ϵ,
#              "hmc_n_steps",   L)
# end

function hmc_params(task::Union{Val{:pima}, Val{:heart}})
     ϵ = 5e-4
     L = 256
     ϵ, L
end

function hmc_params(task::Val{:german})
     ϵ = 0.001
     L = 256
     ϵ, L
end

function run_task(prng::Random.AbstractRNG,
                  task::Union{Val{:pima},
                              Val{:heart},
                              Val{:german},
                              },
                  objective,
                  n_mc,
                  sleep_interval,
                  sleep_ϵ,
                  sleep_L;
                  show_progress=true)
    data_x, data_y = load_dataset(task)
    x_train, y_train, x_test, y_test = prepare_dataset(prng, data_x, data_y)

    x_test = x_train
    y_test = y_train

    AdvancedVI.setadbackend(:forwarddiff)
    Turing.Core._setadbackend(Val(:forwarddiff))

    n_train = size(x_train,1)
    n_dims  = size(x_train,2)
    model   = logistic_regression(x_train, y_train, n_dims)

    function plot_callback(logπ, q, objective, klpq)
        #β   = get_variational_mode(q, model, Symbol("β"))
        #α   = get_variational_mode(q, model, Symbol("α"))

        μ_β, Σ_β = get_variational_mean_var(q, model, Symbol("β"))
        μ_α, Σ_α = get_variational_mean_var(q, model, Symbol("α"))

        f      = x_test*μ_β .+ μ_α[1]
        σ²     = Σ_α[1] .+  sum(x_test'*(x_test*Σ_β), dims=1)[1,:]
        λ⁻²    = 1/(π/8)
        p      = StatsFuns.normcdf.(f ./ sqrt.(λ⁻² .+ σ²))

        acc = mean((p .> 0.5) .== y_test)
        ll  = mean(logpdf.(Turing.Bernoulli.(p), y_test))
        (ll=ll, acc=acc,)
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
                          task::Union{Val{:pima},
                                      Val{:ionosphere},
                                      Val{:heart},
                                      Val{:sonar},
                                      Val{:german},
                                      })
    data_x, data_y = load_dataset(task)
    n_dims         = size(data_x,2)
    model          = logistic_regression(data_x,data_y, n_dims)

    sampler = Turing.NUTS(1000, 0.8;
                          max_depth=8,
                          metricT=AdvancedHMC.UnitEuclideanMetric)
    chain   = Turing.sample(model, sampler, 1000)
    L       = median(chain[:n_steps][:,1])
    ϵ       = mean(chain[:step_size][:,1])
    @info "HMC Tuning Result on $(task)" ϵ=ϵ L=L
end
