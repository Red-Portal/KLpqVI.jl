
load_dataset(::Val{:sonar_gpu})      = load_dataset(Val(:sonar))
load_dataset(::Val{:ionosphere_gpu}) = load_dataset(Val(:ionosphere))
load_dataset(::Val{:australian_gpu}) = load_dataset(Val(:australian))
load_dataset(::Val{:german_gpu})     = load_dataset(Val(:german))
load_dataset(::Val{:breast_gpu})     = load_dataset(Val(:breast))
load_dataset(::Val{:heart_gpu})      = load_dataset(Val(:heart))

function run_task(
    prng::Random.AbstractRNG,
    task::Union{
        Val{:sonar_gpu},
        Val{:ionosphere_gpu},
        Val{:australian_gpu},
        Val{:german_gpu},
        Val{:breast_gpu},
        Val{:heart_gpu},
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

    if task == Val(:breast_gpu) ||
       task == Val(:heart_gpu) ||
       task == Val(:australian_gpu) ||
       task == Val(:german_gpu)
        μ = mean(X_train, dims = 2)
        σ = std(X_train, dims = 2)

        X_train .-= μ
        X_train ./= σ
        X_test  .-= μ
        X_test  ./= σ
    end

    AdvancedVI.setadbackend(:zygote)
    Turing.Core._setadbackend(Val(:zygote))

    X_train_dev = CuArray{Float32}(X_train)
    X_test_dev  = CuArray{Float32}(X_test)

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

        ℓ²_dev = CuArray{eltype(X_train_dev)}(ℓ²)
        μ_f_dev = CuArray{eltype(X_train_dev)}(μ_f)

        R_train = distance_matrix_gpu(X_train_dev, X_train_dev, ℓ²_dev)
        K_unit  = matern52_gpu(R_train)
        K       = eltype(K_unit)(σ²) * K_unit + eltype(K_unit)(1e-6 + ϵ²) * I
        K_chol  = cholesky(K; check = false)

        R_test_train      = distance_matrix_gpu(X_test_dev, X_train_dev, ℓ²_dev)
        K_unit_test_train = matern52_gpu(R_test_train)
        K_test_train      = eltype(K_unit_test_train)(σ²) * K_unit_test_train

        W_factor   = cu(diagm(eltype(K).(σ_f)))
        U_tru      = triu(K_chol.U)
        UW_factor  = U_tru*W_factor
        B          = I + UW_factor'*UW_factor
        B_chol     = cholesky(B; check = false)
        v          = B_chol.L \ (W_factor*K_test_train')
        kᵀKpW⁻¹k   = Array(sum(v .* v, dims = 1)[1, :])
        μ_f_pred   = Array(K_test_train * (K_chol \ μ_f_dev))
        σ²_f_pred  = max.(σ² .+ ϵ² .- kᵀKpW⁻¹k, eps(eltype(K)))

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

    λ_μ = randn(prng, n_params)
    λ_σ = fill(1.0, n_params)
    λ   = vcat(λ_μ, λ_σ)

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

