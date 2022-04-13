
ard_kernel(α², logℓ) =
    α²*(KernelFunctions.Matern52Kernel() ∘ KernelFunctions.ARDTransform(@. exp(-logℓ)))

Turing.@model function logisticgp(X, y, jitter=1e-6)
    n_features = size(X, 1)

    logα  ~ Normal(0, 1)
    logσ  ~ Normal(0, 1)
    logℓ  ~ MvNormal(zeros(n_features), 1)

    logα_finite = clamp(logα, -1e-2, 1e+2)
    logσ_finite = clamp(logσ, -1e-2, 1e+2)
    logℓ_finite  = clamp.(logℓ, -1e-2, 1e+2)

    α²     = exp(logα_finite*2)
    σ²     = exp(logσ_finite*2)
    kernel = ard_kernel(α², logℓ_finite) 
    K      = KernelFunctions.kernelmatrix(kernel, X)
    K_ϵ    = K + (σ² + jitter)*I
    K_chol = cholesky(K_ϵ, check=false)

    if !LinearAlgebra.issuccess(K_chol)
        Turing.@addlogprob! -Inf
        return
    end

    f  ~ MvNormal(zeros(size(X,2)), PDMats.PDMat(K_ϵ, K_chol))
    y .~ Turing.BernoulliLogit.(f)
end

function load_dataset(::Val{:heart})
    dataset = DelimitedFiles.readdlm(
        datadir("dataset", "heart-disease.csv"), ',', skipstart=1)
    data_x  = dataset[:, 1:end-1,]
    data_y  = dataset[:, end]
    data_x, data_y
end

function load_dataset(::Val{:sonar})
    dataset = DelimitedFiles.readdlm(
        datadir("dataset", "sonar.csv"), ',', skipstart=1)
    feature_idx = 1:size(dataset, 2)-1
    data_x      = Float64.(dataset[:, setdiff(feature_idx, 2),])
    data_y      = dataset[:, end]
    data_y      = map(data_y) do s
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
    dataset = DelimitedFiles.readdlm(datadir("dataset", "wdbc.data"), ',')
    data_x  = dataset[:, 3:end]
    data_y  = dataset[:, 2]

    data_y[data_y .== "M"] .= 1.0
    data_y[data_y .== "B"] .= 0.0
    data_x = Float64.(data_x)
    data_y = Float64.(data_y)

    data_x, data_y
end

function load_dataset(::Val{:australian})
    dataset = DelimitedFiles.readdlm(datadir("dataset", "australian.dat"), ' ')
    data_x  = dataset[:, 1:end-1]
    data_y  = dataset[:, end]
    data_x, data_y
end

function run_task(prng::Random.AbstractRNG,
                  task::Union{Val{:sonar},
                              Val{:ionosphere},
                              Val{:australian},
                              Val{:breast},
                              Val{:heart}},
                  optimizer,
                  objective,
                  n_iter,
                  n_mc,
                  defensive_weight;
                  show_progress=true)
    data_x, data_y = load_dataset(task)
    X_train, y_train, X_test, y_test = prepare_dataset(prng, data_x, data_y, ratio=0.9)
    X_train = Array(X_train')
    X_test  = Array(X_test')
    jitter  = 1e-6
    model   = logisticgp(X_train, y_train, jitter)

    if task == Val(:breast) || task == Val(:heart) || task == Val(:australian)
        μ = mean(X_train, dims=2)
        σ = std( X_train, dims=2)

        X_train .-= μ
        X_train ./= σ
        X_test  .-= μ
        X_test  ./= σ
    end

    #AdvancedVI.setadbackend(:forwarddiff)
    #Turing.Core._setadbackend(Val(:forwarddiff))
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))
    # AdvancedVI.setadbackend(:zygote) 
    # Turing.Core._setadbackend(Val(:zygote))

    varinfo     = DynamicPPL.VarInfo(model)
    varsyms     = keys(varinfo.metadata)
    n_params    = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ varsyms])
    θ           = 0.1*randn(prng, n_params*2)

    @info("Logistic Gaussian process $(task)",
          train_data=size(X_train),
          test_data=size(X_test),
          n_params=n_params)

    q = Turing.Variational.meanfield(model)
    q = AdvancedVI.update(q, θ)

    i      = 1
    #k_hist = []
    function callback(ℓπ, λ)
        q′      = AdvancedVI.update(q, λ)
        f, Σ_f  = get_variational_mean_var(q′, model, Symbol("f"))
        logσ, _ = get_variational_mean_var(q′, model, Symbol("logσ"))
        logℓ, _ = get_variational_mean_var(q′, model, Symbol("logℓ"))
        logα, _ = get_variational_mean_var(q′, model, Symbol("logα"))

        logα_finite = clamp(logα[1], -1e-2, 1e+2)
        logσ_finite = clamp(logσ[1], -1e-2, 1e+2)
        logℓ_finite  = clamp.(logℓ, -1e-2, 1e+2)

        σ²_n    = exp(logσ_finite*2)
        α²      = exp(logα_finite*2)

        stat = if(mod(i-1, 1) == 0)
            kernel       = ard_kernel(α², logℓ_finite) 
            K            = KernelFunctions.kernelmatrix(kernel, X_train, obsdim=2)
            K_ϵ          = K + (σ²_n + jitter)*I
            K_test_train = KernelFunctions.kernelmatrix(kernel, X_test, X_train; obsdim=2)
            k_test       = KernelFunctions.kernelmatrix_diag(kernel, X_test; obsdim=2)

            K_pd   = PDMats.PDMat(K_ϵ)
            W⁻¹    = diagm(1 ./ Σ_f)
            KpW⁻¹  = PDMats.PDMat(K_ϵ + W⁻¹)
            μ      = K_test_train * (K_pd \ f)
            σ²     = (k_test .+ σ²_n) - PDMats.invquad.(Ref(KpW⁻¹), eachrow(K_test_train))

            s      = μ ./ sqrt.(1 .+ π*σ²/8)
            p      = StatsFuns.logistic.(s)
            y_pred = p .> 0.5
            lpd    = mean(logpdf.(Turing.BernoulliLogit.(s), y_test))
            acc    = mean(y_pred .== y_test)

            (acc = acc, lpd = lpd)
        else
            NamedTuple()
        end
        i += 1
        stat
    end

    n_params = length(q)
    ν        = Distributions.Product(fill(Cauchy(), n_params))
    θ, stats = vi(model, q;
                  objective        = objective,
                  n_mc             = n_mc,
                  n_iter           = n_iter,
                  callback         = callback,
                  rng              = prng,
                  defensive_dist   = ν,
                  defensive_weight = defensive_weight,
                  optimizer        = optimizer,
                  show_progress    = show_progress
                  )
    Dict.(pairs.(stats))
end

