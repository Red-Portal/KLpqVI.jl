
Turing.@model logistic_regression(X, y, n, d, σ) = begin
    β  ~ MvNormal(d, σ)
    α  ~ Normal(0, σ)
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

function sample_posterior(task::Val{:pima})
    data_x, data_y = fetch_dataset(task)
    n_dims = size(data_x, 2)
    n_data = size(data_x, 1)
    model  = logistic_regression(data_x,
                                 data_y,
                                 n_data, 
                                 n_dims,
                                 1.0)
    sampler = Turing.NUTS(4096, 0.8; max_depth=8)
    chain   = Turing.sample(model, sampler, 16_384)
    L       = median(chain[:n_steps][:,1])
    ϵ       = mean(chain[:step_size][:,1])
    params  = Array(chain)
    JLD.save(datadir("posterior", "pima.jld"),
             "samples",       params,
             "hmc_step_size", ϵ,
             "hmc_n_steps",   L)
end
