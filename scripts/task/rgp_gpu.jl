
load_dataset(::Val{:boston_gpu})   = load_dataset(Val(:boston))
load_dataset(::Val{:wine_gpu})     = load_dataset(Val(:wine))
load_dataset(::Val{:yacht_gpu})    = load_dataset(Val(:yacht))
load_dataset(::Val{:concrete_gpu}) = load_dataset(Val(:concrete))
load_dataset(::Val{:gas_gpu})      = load_dataset(Val(:gas))
load_dataset(::Val{:airfoil_gpu})  = load_dataset(Val(:airfoil))
load_dataset(::Val{:energy_gpu})   = load_dataset(Val(:energy))

function run_task(
    prng::Random.AbstractRNG,
    task::Union{Val{:wine_gpu},
                Val{:boston_gpu},
                Val{:yacht_gpu},
                Val{:concrete_gpu},
                Val{:airfoil_gpu},
                Val{:gas_gpu},
                Val{:energy_gpu},
                },
    optimizer,
    objective,
    n_iter,
    n_mc,
    defensive_weight;
    show_progress = true,
)
    jitter = 1e-6

    data_x, data_y = load_dataset(task)
    X_train, y_train, X_test, y_test = prepare_dataset(prng, data_x, data_y, ratio = 0.9)

    X_train   = Array(X_train')
    X_test    = Array(X_test')
    μ_X       = mean(X_train, dims = 2)[:, 1]
    σ_X       = std(X_train, dims = 2)[:, 1]
    valid_idx = σ_X .> 1e-6

    X_train .-= μ_X
    X_test  .-= μ_X
    X_train ./= σ_X
    X_test  ./= σ_X

    X_train = X_train[valid_idx,:]
    X_test  = X_test[valid_idx,:]

    μ_y = mean(y_train)
    σ_y = std(y_train)
    y_train .-= μ_y
    y_train /= σ_y

    AdvancedVI.setadbackend(:zygote)
    Turing.Core._setadbackend(Val(:zygote))

    X_train_dev = CuArray{Float32}(X_train)
    X_test_dev  = CuArray{Float32}(X_test)

    n_dims = size(X_train_dev, 1)
    n_data = size(X_train_dev, 2)

    n_params = 4 + n_dims + n_data
    ℓϵ_y_idx = 1
    ℓϵ_f_idx = 2
    ℓσ_idx   = 3
    ℓν_idx   = 4
    ℓℓ_idx   = 5:4+n_dims
    f_idx    = 5+n_dims:n_params

    function ℓπ(θ_)
        ℓϵ_y_ = θ_[ℓϵ_y_idx]
        ℓϵ_f_ = θ_[ℓϵ_f_idx]
        ℓσ_ = θ_[ℓσ_idx]
        ℓℓ_ = θ_[ℓℓ_idx]
        ℓν_ = θ_[ℓν_idx]
        f_ = θ_[f_idx]

        ϵ_y_   = exp(clamp(ℓϵ_y_, -10, 10))
        ϵ²_f_  = exp(clamp(2 * ℓϵ_f_, -20, 20))
        σ²_    = exp(clamp(2 * ℓσ_, -20, 20))
        ν_     = exp(clamp(ℓν_, 0, 10))
        ℓ²_dev_ = cu(exp.(clamp.(2 * ℓℓ_, -20, 20)))
        f_dev_ = cu(f_)
    
        ℓpϵ_y = logpdf(Normal(0, 4), ℓϵ_y_)
        ℓpϵ_f = logpdf(Normal(0, 4), ℓϵ_f_)
        ℓpσ = logpdf(Normal(0, 4), ℓσ_)
        ℓpν = logpdf(Gamma(4, 10), ν_)
        ℓpℓ = logpdf(MvNormal(fill(0.0, n_dims), fill(0.2, n_dims)), ℓℓ_)
        ℓpf = gp_likelihood(X_train_dev, f_dev_, σ²_, ϵ²_f_ + jitter, ℓ²_dev_)

        z_train = (y_train .- f_) ./ ϵ_y_
        ℓpy = mapreduce(+, z_train) do zᵢ
            logpdf(TDist(ν_), zᵢ) - ℓϵ_y_
        end
        ℓpf + ℓpy + ℓpϵ_y + ℓpϵ_f + ℓpσ + ℓpℓ + ℓpν
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
        i += 1

        μ = λ_[1:n_params]
        σ = StatsFuns.softplus.(λ_[n_params+1:end])
    
        ϵ²_y = exp(clamp(2 * μ[ℓϵ_y_idx], -20, 20))
        ϵ²_f = exp(clamp(2 * μ[ℓϵ_f_idx], -20, 20))
        σ² = exp(clamp(2 * μ[ℓσ_idx], -20, 20))
        ℓ² = exp.(clamp.(2 * μ[ℓℓ_idx], -20, 20))
        ν = exp.(clamp.(μ[ℓν_idx], 0, 10))
    
        μ_f = μ[f_idx]
        σ_f = σ[f_idx]
    
        ℓ²_dev = CuArray{eltype(X_train_dev)}(ℓ²)
        μ_f_dev = CuArray{eltype(X_train_dev)}(μ_f)
    
        R_train = distance_matrix_gpu(X_train_dev, X_train_dev, ℓ²_dev)
        K_unit  = se_gpu(R_train)
        K       = eltype(K_unit)(σ²) * K_unit + eltype(K_unit)(jitter + ϵ²_f) * I
        K_chol  = cholesky(K; check = false)

        R_test_train      = distance_matrix_gpu(X_test_dev, X_train_dev, ℓ²_dev)
        K_unit_test_train = se_gpu(R_test_train)
        K_test_train      = eltype(K_unit_test_train)(σ²) * K_unit_test_train
    
        W_factor   = cu(diagm(eltype(K).(σ_f)))
        U_tru      = triu(K_chol.U)
        UW_factor  = U_tru*W_factor
        B          = I + UW_factor'*UW_factor
        B_chol     = cholesky(B; check = false)
        v          = B_chol.L \ (W_factor*K_test_train')
        kᵀKpW⁻¹k   = Array(sum(v .* v, dims = 1)[1, :])
        μ_f_pred   = Array(K_test_train * (K_chol \ μ_f_dev))
        σ²_f_pred  = max.(σ² .+ ϵ²_y .- kᵀKpW⁻¹k, eps(eltype(K)))
    
        if issuccess(K_chol) && issuccess(K_chol)
            σ_y_pred = sqrt.(σ²_f_pred * σ_y*σ_y)
            μ_y_pred = μ_f_pred*σ_y .+ μ_y

            z_test = (y_test .- μ_y_pred) ./ σ_y_pred
            lpd    = mean(logpdf.(Ref(TDist(ν)), z_test) - log.(σ_y_pred))
            rmse   = sqrt(mean((y_test - μ_y_pred) .^ 2))
            (iter = i, rmse = rmse, lpd = lpd, ν = ν)
        else
            (iter = i, rmse = typemax(Float64), lpd = -Inf, ν=0)
        end
    end

    λ  = if objective isa ELBO
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

