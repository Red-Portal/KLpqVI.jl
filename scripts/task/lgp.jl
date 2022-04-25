
Turing.@model function logisticgp(X, y, jitter=1e-6)
    n_features = size(X, 1)

    logα  ~ Normal(0, 4)
    logσ  ~ Normal(0, 4)
    logℓ  ~ MvNormal(zeros(n_features), 4)

    logα_finite = clamp(logα, -1e-2, 1e+2)
    logσ_finite = clamp(logσ, -1e-2, 1e+2)
    logℓ_finite  = clamp.(logℓ, -1e-2, 1e+2)

    α²     = exp(logα_finite*2)
    σ²     = exp(logσ_finite*2)
    kernel = matern_kernel(α², logℓ_finite) 
    K      = KernelFunctions.kernelmatrix(kernel, X)
    K_ϵ    = K + (σ² + jitter)*I
    K_chol = cholesky(K_ϵ, check=false)

    if !LinearAlgebra.issuccess(K_chol)
        f  ~ MvNormal(zeros(size(X,2)), ones(size(X,2)))
        Turing.@addlogprob! -Inf
        return
    end

    f  ~ MvNormal(zeros(size(X,2)), PDMats.PDMat(K_ϵ, K_chol))
    y .~ Turing.BernoulliLogit.(f)
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

    if task == Val(:breast) || task == Val(:heart) || task == Val(:australian) || task == Val(:sonar)
        μ = mean(X_train, dims=2)
        σ = std( X_train, dims=2)

        X_train .-= μ
        X_train ./= σ
        X_test  .-= μ
        X_test  ./= σ
    end

    #AdvancedVI.setadbackend(:forwarddiff)
    #Turing.Core._setadbackend(Val(:forwarddiff))
    #AdvancedVI.setadbackend(:reversediff)
    #Turing.Core._setadbackend(Val(:reversediff))
    AdvancedVI.setadbackend(:zygote) 
    Turing.Core._setadbackend(Val(:zygote))

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
            kernel       = matern_kernel(α², logℓ_finite) 
            K            = KernelFunctions.kernelmatrix(kernel, X_train, obsdim=2)
            K_ϵ          = K + (σ²_n + jitter)*I
            K_test_train = KernelFunctions.kernelmatrix(kernel, X_test, X_train; obsdim=2)

            K_chol    = cholesky(K_ϵ; check = false)
            W_factor  = diagm(sqrt.(Σ_f))
            U_tru     = triu(K_chol.U)
            UW_factor = U_tru*W_factor
            B         = I + UW_factor'*UW_factor
            B_chol    = cholesky(B; check = false)
            v         = B_chol.L \ (W_factor*K_test_train')
            kᵀKpW⁻¹k  = Array(sum(v .* v, dims = 1)[1, :])
            μ         = Array(K_test_train * (K_chol \ f))
            σ²        = max.(α² .+ σ²_n .- kᵀKpW⁻¹k, 1e-10)

            s      = μ ./ sqrt.(1 .+ π*σ²/8)
            p      = StatsFuns.logistic.(s)
            y_pred = p .> 0.5
            lpd    = mean(logpdf.(Turing.BernoulliLogit.(s), y_test))
            acc    = mean(y_pred .== y_test)

            (iter = i, acc = acc, lpd = lpd)
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
