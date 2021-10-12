
import Tracker

ard_kernel(α², logℓ, σ²) =
    α²*(KernelFunctions.Matern52Kernel() ∘ KernelFunctions.ARDTransform(@. exp(-logℓ))) + σ² * KernelFunctions.WhiteKernel()

Turing.@model function logisticgp(X, y, jitter=1e-6)
    n_features = size(X, 1)

    logα  ~ Normal(0, 1)
    logσ  ~ Normal(0, 1)
    logℓ  ~ MvNormal(zeros(n_features), 1)

    α²     = exp(logα*2)
    σ²     = exp(logσ*2)
    kernel = ard_kernel(α², logℓ, σ²) 
    K      = KernelFunctions.kernelmatrix(kernel, X)

    f  ~ MvNormal(zeros(size(X,2)), K + jitter*I)
    y .~ Turing.BernoulliLogit.(f)
end

function load_dataset(::Val{:sonar})
    dataset = DelimitedFiles.readdlm(
        datadir("dataset", "sonar.csv"), ',', skipstart=1)
    data_x  = Float64.(dataset[:, 1:end-1,])
    data_y  = dataset[:, end]
    data_y  = map(data_y) do s
        if(s == "Rock")
            1.0
        else
            0.0
        end
    end
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

function load_dataset(::Val{:breast})
    dataset, _ = DelimitedFiles.readdlm(
        datadir("dataset", "breast.csv"), ',', header=true)
    data_x  = dataset[:, 3:end-1]
    data_y  = dataset[:, 2]

    data_y[data_y .== "M"] .= 1.0
    data_y[data_y .== "B"] .= 0.0

    data_x = Float64.(data_x)
    data_y = Float64.(data_y)
    data_x, data_y
end

function evaluate_metric(prng, X_test, y_test, X_data, q, model)
    n_samples = 32
    fs    = sample_variable(prng, q, model, Symbol("f"),    n_samples)
    logσs = sample_variable(prng, q, model, Symbol("logσ"), n_samples)
    logαs = sample_variable(prng, q, model, Symbol("logα"), n_samples)
    logℓs = sample_variable(prng, q, model, Symbol("logℓ"),  n_samples)

    p_res = Array{Float64}(undef, size(X_test, 2), n_samples)
    for i = 1:n_samples
        f    = view(fs, :, i)
        logℓ = view(logℓs, :, i)
        logσ = logσs[1, i]
        logα = logαs[1, i]
        α²   = exp(logα*2)
        σ²   = exp(logσ*2)

        kernel  = ard_kernel(α², logℓ, σ²) 
        gp      = AbstractGPs.GP(kernel)
        gp_post = AbstractGPs.posterior(gp(X_data, σ² + 1e-6), f)
        μ, σ²   = mean_and_var(gp_post(X_test))
        λ⁻²     = 1/(π/8)
        p       = StatsFuns.normcdf.(μ ./ sqrt.(λ⁻² .+ σ²))
        p_res[:,i] = p
    end
    p_μ = mean(p_res, dims=2)[:,1]
    y_pred  = p_μ .> 0.5
    nlpd    = mean(logpdf.(Bernoulli.(p_μ), y_test))
    acc     = mean(y_pred .== y_test)
    nlpd, acc
end

function run_task(prng::Random.AbstractRNG,
                  task::Union{Val{:sonar}, Val{:ionosphere}, Val{:breast}},
                  objective,
                  n_mc,
                  sleep_interval,
                  sleep_ϵ,
                  sleep_L;
                  show_progress=true)
    data_x, data_y = load_dataset(task)
    X_train, y_train, X_test, y_test = prepare_dataset(prng, data_x, data_y, ratio=0.9)
    X_train = Array(X_train')
    X_test  = Array(X_test')
    model   = logisticgp(X_train, y_train, 1e-4)

    μ = mean(X_train, dims=2)
    σ = std(X_train, dims=2)
    X_train .-= μ
    X_train ./= σ
    X_test .-= μ
    X_test ./= σ

    #AdvancedVI.setadbackend(:forwarddiff)
    #Turing.Core._setadbackend(Val(:forwarddiff))
    #AdvancedVI.setadbackend(:reversediff)
    #Turing.Core._setadbackend(Val(:reversediff))
    AdvancedVI.setadbackend(:zygote) 
    Turing.Core._setadbackend(Val(:zygote))

    i      = 1
    #k_hist = []
    function plot_callback(ℓπ, q, objective_, klpq)
        f    = get_variational_mode(q, model, Symbol("f"))
        logσ = get_variational_mode(q, model, Symbol("logσ"))
        logℓ = get_variational_mode(q, model, Symbol("logℓ"))
        logα = get_variational_mode(q, model, Symbol("logα"))
        σ²  = exp(logσ[1]*2)
        α²  = exp(logα[1]*2)
        ℓ   = exp.(logℓ)

        stat = if(mod(i-1, 1) == 0)
            kernel  = ard_kernel(α², logℓ, σ²) 
            gp      = AbstractGPs.GP(kernel)
            gp_post = AbstractGPs.posterior(gp(X_train, σ² + 1e-6), f)
            μ, σ²   = mean_and_var(gp_post(X_test))

            #μ, σ²   = mean_and_var(gp_post, X_test, dims=2)
            λ⁻²     = 1/(π/8)
            p       = StatsFuns.normcdf.(μ ./ sqrt.(λ⁻² .+ σ²))
            y_pred  = p .> 0.5
            nlpd    = mean(logpdf.(Bernoulli.(p), y_test))
            acc     = mean(y_pred .== y_test)

            #nlpd, acc = evaluate_metric(prng, X_test, y_test, X_train, q, model)
            (acc = acc, nlpd = nlpd)
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
                     #paretok_samples = 10,
                     sleep_interval  = sleep_interval,
                     sleep_params    = (ϵ=sleep_ϵ, L=sleep_L,),
                     #optimizer      = AdvancedVI.TruncatedADAGrad(),
                     optimizer       = Flux.ADAM(0.01),
                     show_progress   = show_progress
                     )
    Dict.(pairs.(stats))
end

# function sonar_test()
#     AdvancedVI.setadbackend(:forwarddiff)
#     Turing.Core._setadbackend(Val(:forwarddiff))

#     seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
#     prng = Random123.Philox4x(UInt64, seed, 8)
#     Random123.set_counter!(prng, 1)

#     #data_x, data_y = load_dataset(Val(:sonar))
#     #X_train, y_train, X_test, y_test = prepare_dataset(prng, data_x, data_y, ratio=0.9)
#     #X_train = Array(X_train')
#     #X_test  = Array(X_test')

#     X_train = @SMatrix randn(60, 150)
#     y_train = @SVector rand(50)
#     y_train = y_train .> 0.5

#     model   = logisticgp(X_train, y_train, 1e-4)
#     δ       = 1e-6
#     mcmc    = Turing.NUTS(200, 0.8; max_depth=5)
#     samples = Turing.sample(model, mcmc, 1_000)
#     samples
# end


function run_task(prng::Random.AbstractRNG,
                  task::Union{Val{:sonar}, Val{:ionosphere}, Val{:breast}},
                  ::ELBO,
                  n_mc,
                  sleep_interval,
                  sleep_ϵ,
                  sleep_L;
                  show_progress=true)
    data_x, data_y = load_dataset(task)
    X_train, y_train, X_test, y_test = prepare_dataset(prng, data_x, data_y, ratio=0.9)
    X_train = Array(X_train')
    X_test  = Array(X_test')

    μ = mean(X_train, dims=2)
    σ = std(X_train, dims=2)
    X_train .-= μ
    X_train ./= σ
    X_test .-= μ
    X_test ./= σ

    jitter  = 1e-6

    n_features = size(X_train, 1)
    n_data     = size(X_train, 2)
    n_samples  = n_mc
    n_params   = 2+n_features+n_data
    n_iter     = 10000
    opt        = Flux.ADAM(0.01)


    function joint_likelihood(z::AbstractVector)
        logα = z[1]
        logσ = z[2]
        logℓ = view(z, 3:2+n_features)
        f    = view(z, 3+n_features:length(z))

        p_logα = logpdf(Normal(0, 1), logα)
        p_logσ = logpdf(Normal(0, 1), logσ)
        p_logℓ = logpdf(MvNormal(zeros(n_features), 1), logℓ)

        α²     = exp(logα*2)
        σ²     = exp(logσ*2)
        kernel = ard_kernel(α², logℓ, σ²) 
        K      = KernelFunctions.kernelmatrix(kernel, X_train)
        p_f    = logpdf(MvNormal(zeros(n_data), K + jitter*I), f)
        
        loglike = mapreduce(+, 1:n_data) do i
            logpdf(Turing.BernoulliLogit(f[i]), y_train[i])
        end 
        p_logα + p_logσ + p_logℓ + p_f + loglike
    end

    # function joint_likelihood(z::AbstractVector)
    #     logpdf(MvNormal(ones(length(z)), ones(length(z))), z)
    # end

    function nelbo(λ, ϵ)
        μ = view(λ, 1:n_params)
        σ = exp.(view(λ, n_params+1:2*n_params))
        z = ϵ.*σ .+ μ

        λ_stop = Zygote.dropgrad(λ)
        μ_stop = view(λ_stop, 1:n_params)
        σ_stop = exp.(view(λ_stop, n_params+1:2*n_params))

        mapreduce(+, eachcol(z)) do zᵢ
            -joint_likelihood(zᵢ) + logpdf(MvNormal(μ_stop, σ_stop), zᵢ)
        end / n_samples
    end

    function performance(λ)
        logα = λ[1] 
        logσ = λ[2]
        logℓ = view(λ, 3:2+n_features)
        f    = view(λ, 3+n_features:2+n_features+n_data)
        α²   = exp(2*logα)
        σ²   = exp(2*logσ)
        
        kernel  = ard_kernel(α², logℓ, σ²) 
        gp      = AbstractGPs.GP(kernel)
        gp_post = AbstractGPs.posterior(gp(X_train, σ² + 1e-6), f)
        μ, σ²   = mean_and_var(gp_post(X_test))

        #μ, σ²   = mean_and_var(gp_post, X_test, dims=2)
        λ⁻²     = 1/(π/8)
        p       = StatsFuns.normcdf.(μ ./ sqrt.(λ⁻² .+ σ²))
        y_pred  = p .> 0.5
        nlpd    = mean(logpdf.(Bernoulli.(p), y_test))
        acc     = mean(y_pred .== y_test)
        (nlpd=nlpd, acc=acc)
    end

    λₜ          = randn(prng, n_params*2)*0.1
    diff_result = DiffResults.GradientResult(λₜ)
    stats       = Vector{NamedTuple}(undef, n_iter)
    elapsed_total = 0
    for t = 1:n_iter
        start_time = Dates.now()
        
        # sample gradient
        ϵ = randn(prng, n_params, n_samples)
        y, back = Zygote.pullback(λ -> nelbo(λ, ϵ), λₜ)
        g       = last(back(1.0))
        DiffResults.gradient!(diff_result, g)
        DiffResults.value!(   diff_result, y)

        # update parameter
        Δ  = DiffResults.gradient(diff_result)
        Flux.Optimise.apply!(opt, λₜ, Δ)
        λₜ -= Δ

        # metrics
        stat           = (iteration=t,)
        stat           = merge(stat, performance(λₜ))

        elapsed        = Dates.now() - start_time
        elapsed_total += elapsed.value
        stat           = merge(stat, (elapsed=elapsed_total,))

        stats[t] = stat
    end
    Dict.(pairs.(stats))
end
