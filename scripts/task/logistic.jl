
Turing.@model logistic_regression(X, y, n, d, σ) = begin
    τ  ~ truncated(Normal(0, 2.0), 0, Inf)
    β  ~ MvNormal(d, τ)
    α  ~ Normal(0, τ)
    s  = X*β .+ α
    y .~ Turing.BernoulliLogit.(s)
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

function load_dataset(::Val{:ionosphere})
    dataset = DelimitedFiles.readdlm(
        datadir("dataset", "ionosphere.csv"), ',')
    data_x  = dataset[:, 1:end-1,]
    data_y  = dataset[:, end]

    data_y[data_y .== "g"] .= 1.0
    data_y[data_y .== "b"] .= 0.0

    data_x = Float64.(data_x)
    data_y = Float64.(data_y)
    data_x, data_y
end

function prepare_dataset(prng::Random.AbstractRNG,
                         data_x::AbstractMatrix,
                         data_y::AbstractVector)
    data_x_t       = data_x'
    data_x, data_y = MLDataUtils.shuffleobs((data_x_t, data_y); rng=prng)
    (train_x, train_y), (valid_x, valid_y) = MLDataUtils.splitobs(
        (data_x_t, data_y); at=0.9)
    (Array(train_x'), train_y), (Array(valid_x'), valid_y)
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
