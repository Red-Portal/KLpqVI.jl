
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

ard_kernel(α², logℓ) =
    α²*(KernelFunctions.Matern52Kernel() ∘ KernelFunctions.ARDTransform(@. exp(-logℓ)))


Turing.@model function logisticgp(X, y, jitter=1e-6)
    n_features = size(X, 1)

    logσ  ~ Normal(0, 4)
    logα  ~ Normal(0, 4)
    logℓ  ~ MvNormal(zeros(n_features), 4)

    logα_finite = clamp(logα, -10, 10)
    logσ_finite = clamp(logσ, -10, 10)
    logℓ_finite  = clamp.(logℓ, -10, 10)

    α²     = exp(logα_finite*2)
    σ²     = exp(logσ_finite*2)
    kernel = ard_kernel(α², logℓ_finite) 
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

function run_task(
    prng::Random.AbstractRNG,
    task::Union{
        Val{:sonar},
        Val{:ionosphere},
        Val{:australian},
        Val{:german},
        Val{:breast},
        Val{:heart},
    },
    optimizer,
    objective,
    n_iter,
    n_mc,
    defensive_weight;
    show_progress = true,
)

    data_x, data_y = load_dataset(task)
    X_train, y_train, X_test, y_test = prepare_dataset(prng, data_x, data_y, ratio = 0.9)
    X_train = Array(X_train')
    X_test = Array(X_test')

    if task == Val(:breast) || task == Val(:heart) || task == Val(:australian) #||
        #task == Val(:sonar)
        μ = mean(X_train, dims = 2)
        σ = std(X_train, dims = 2)

        X_train .-= μ
        X_train ./= σ
        X_test  .-= μ
        X_test  ./= σ
    end

    AdvancedVI.setadbackend(:zygote)
    Turing.Core._setadbackend(Val(:zygote))

    X_train_dev = cu(X_train)
    X_test_dev  = cu(X_test)

    n_dims = size(X_train_dev, 1)
    n_data = size(X_train_dev, 2)

    n_params = 2 + n_dims + n_data
    ℓϵ_idx   = 1
    ℓσ_idx   = 2
    ℓℓ_idx   = 3:2+n_dims
    f_idx    = 3+n_dims:n_params

    function ℓπ(θ_)
        ℓϵ_ = θ_[ℓϵ_idx]
        ℓσ_ = θ_[ℓσ_idx]
        ℓℓ_ = θ_[ℓℓ_idx]
        f_ = θ_[f_idx]
        ϵ²_ = exp(clamp(2 * ℓϵ_, -20, 20))
        σ²_ = exp(clamp(2 * ℓσ_, -20, 20))
        ℓ²_dev_ = cu(exp.(clamp.(2 * ℓℓ_, -20, 20)))
        f_dev_ = cu(f_)

        ℓpϵ = logpdf(Normal(0, 4), ℓϵ_)
        ℓpσ = logpdf(Normal(0, 4), ℓσ_)
        ℓpℓ = logpdf(MvNormal(zeros(n_dims), fill(4, n_dims)), ℓℓ_)
        ℓpf = gp_likelihood(X_train_dev, f_dev_, σ²_, ϵ²_ + 1e-6, ℓ²_dev_)
        ℓpy = sum(logpdf.(Turing.BernoulliLogit.(f_), y_train))
        ℓpf + ℓpy + ℓpϵ + ℓpσ + ℓpℓ
    end

    function ℓq(λ_, θ_)
        μ_ = λ_[1:n_params]
        σ_ = StatsFuns.softplus.(λ_[n_params+1:end])
        logpdf(DistributionsAD.TuringDiagMvNormal(μ_, σ_), θ_)
    end

    function rand_q(prng_, λ_)
        μ_ = λ_[1:n_params]
        σ_ = StatsFuns.softplus.(λ_[n_params+1:end])
        rand(prng_, DistributionsAD.TuringDiagMvNormal(μ_, σ_))
    end

    @info(
        "Logistic Gaussian process $(task)",
        train_data = size(X_train),
        test_data = size(X_test),
        n_params = n_params
    )

    i = 0
    #k_hist = []
    function callback(_, λ_)
        μ = λ_[1:n_params]
        σ = StatsFuns.softplus.(λ_[n_params+1:end])

        ϵ² = exp(clamp(2 * μ[ℓϵ_idx], -20, 20))
        σ² = exp(clamp(2 * μ[ℓσ_idx], -20, 20))
        ℓ² = exp.(clamp.(2 * μ[ℓℓ_idx], -20, 20))

        μ_f = μ[f_idx]
        σ_f = σ[f_idx]

        ℓ²_dev = cu(ℓ²)
        μ_f_dev = cu(μ_f)

        R_train = distance_matrix_gpu(X_train_dev, X_train_dev, ℓ²_dev)
        K_unit  = matern52_gpu(R_train)
        K       = σ² * K_unit + (ϵ² + 1e-6) * I

        R_test_train      = distance_matrix_gpu(X_test_dev, X_train_dev, ℓ²_dev)
        K_unit_test_train = matern52_gpu(R_test_train)
        K_test_train      = σ² * K_unit_test_train

        K_chol     = cholesky(K; check = false)
        W⁻¹        = cu(diagm(1 ./ (σ_f .* σ_f)))
        KpW⁻¹      = K + W⁻¹
        KpW⁻¹_chol = cholesky(KpW⁻¹; check = false)
        U⁻¹k       = KpW⁻¹_chol.U \ K_test_train'
        kᵀKpW⁻¹k   = Array(sum(U⁻¹k .* U⁻¹k, dims = 1)[1, :])
        μ_f_pred   = Array(K_test_train * (K_chol \ μ_f_dev))
        σ²_f_pred  = max.((σ² + ϵ²) .- kᵀKpW⁻¹k, 1e-7)

        s      = μ_f_pred ./ sqrt.(1 .+ π * σ²_f_pred / 8)
        p      = StatsFuns.logistic.(s)
        y_pred = p .> 0.5
        lpd    = mean(logpdf.(Turing.BernoulliLogit.(s), y_test))
        acc    = mean(y_pred .== y_test)

        i += 1
        (iter = i, acc = acc, lpd = lpd)
    end

    ctx     = DynamicPPL.MiniBatchContext(DynamicPPL.DefaultContext(), 1)
    model   = logisticgp(X_train, y_train)
    vi_init = DynamicPPL.VarInfo(model, ctx)
    model(prng, vi_init, DynamicPPL.SampleFromUniform())

    λ = if objective isa ELBO
        λ_μ = randn(prng, n_params)
        λ_σ = fill(1.0, n_params)
        vcat(λ_μ, λ_σ)
    else
        λ_μ = randn(prng, n_params)
        λ_σ = fill(3.0, n_params)
        vcat(λ_μ, λ_σ)
    end

    θ, stats = vi(
        ℓπ,
        ℓq,
        rand_q,
        λ;
        objective     = objective,
        n_mc          = n_mc,
        n_iter        = n_iter,
        callback      = callback,
        rng           = prng,
        optimizer     = optimizer,
        show_progress = show_progress,
    )
    Dict.(pairs.(stats))
end

