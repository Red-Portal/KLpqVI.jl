
Turing.@model logistic_regression(X, y, d) = begin
    ϵ  = 1e-10
    τ  ~ InverseGamma(1, 1)
    σ  ~ InverseGamma(1, 1)

    # if(τ < ϵ || σ < ϵ)
    #     Turing.@addlogprob! -Inf   
    #     return
    # end

    β  ~ MvNormal(d, τ)
    α  ~ Normal(0, σ)
    s  = X*β .+ α
    y .~ Turing.BernoulliLogit.(s)
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

# function run_task(prng::Random.AbstractRNG,
#                   task::Union{Val{:pima},
#                               Val{:heart},
#                               Val{:german},
#                               },
#                   objective,
#                   n_mc;
#                   defensive_weight=nothing,
#                   show_progress=true)
#     data_x, data_y = load_dataset(task)
#     x_train, y_train, x_test, y_test = prepare_dataset(prng, data_x, data_y)

#     x_test = x_train
#     y_test = y_train

#     Turing.Core._setadbackend(Val(:reversediff))

#     n_train = size(x_train,1)
#     n_dims  = size(x_train,2)
#     model   = logistic_regression(x_train, y_train, n_dims)
#     q       = Turing.Variational.meanfield(model)

#     function callback(logπ, λ)
#         q′       = AdvancedVI.update(q, λ)
#         μ_β, Σ_β = get_variational_mean_var(q′, model, Symbol("β"))
#         μ_α, Σ_α = get_variational_mean_var(q′, model, Symbol("α"))

#         μ      = x_test*μ_β .+ μ_α[1]
#         Σ_β_pd = PDMats.PDiagMat(Σ_β)
#         σ²     = Σ_α[1] .+ PDMats.quad.(Ref(Σ_β_pd), eachrow(x_test))
#         s      = μ ./ sqrt.(1 .+ π*σ²/8)
#         p      = StatsFuns.logistic.(s)
#         acc    = mean((p .> 0.5) .== y_test)
#         lpd    = mean(logpdf.(Turing.BernoulliLogit.(s), y_test))

#         if (isinf(lpd))
#             throw(OverflowError("predictive likelihood is inf"))
#         end

#         (lpd=lpd, acc=acc,)
#     end

#     n_params = length(q)
#     ν        = Distributions.Product(fill(Cauchy(), n_params))
#     n_iter   = 3000
#     θ, stats = vi(model;
#                   objective        = objective,
#                   n_mc             = n_mc,
#                   n_iter           = n_iter,
#                   callback         = callback,
#                   rng              = prng,
#                   defensive_dist   = ν,
#                   defensive_weight = defensive_weight,
#                   optimizer        = Flux.ADAM(0.01),
#                   show_progress    = show_progress
#                   )
#     # β = get_variational_mode(q, model, Symbol("β"))
#     # α = get_variational_mode(q, model, Symbol("α"))
#     # θ = vcat(β, α)
#     Dict.(pairs.(stats))
# end

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
