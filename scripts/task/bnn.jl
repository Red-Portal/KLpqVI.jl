
using ExcelFiles
using MLDataUtils: splitobs
using DataFrames

Turing.@model bnn(X, y, hidden_dim) = begin
    λ ~ Gamma(6, 1/6)
    γ ~ Gamma(6, 1/6)

    λ⁻¹ = 1 ./ λ
    γ⁻¹ = 1 ./ γ

    W1 ~ MvNormal(hidden_dim*size(X,1), sqrt(λ⁻¹))
    W2 ~ MvNormal(hidden_dim+1,         sqrt(λ⁻¹))

    W1′ = reshape(W1, (hidden_dim, size(X,1)))
    W2′ = reshape(W2, (1, hidden_dim+1))

    X2  = relu.(W1′*X  / sqrt(size(X, 1)))
    X2′ = vcat(X2, ones(1, size(X2, 2)))
    ŷ   = W2′*X2′ / sqrt(size(X2′, 1))
    y ~ MvNormal(ŷ[1,:], sqrt.(γ⁻¹))
end

function load_dataset(::Val{:wine})
    fname   = datadir(joinpath("dataset", "winequality-red.csv"))
    vals, _ = readdlm(fname, ',', header=true)
    X = Array{Float32}(vals[:, 1:end-1])
    y = Array{Float32}(vals[:, end])
    X, y
end

function load_dataset(::Val{:boston})
    fname = datadir(joinpath("dataset", "housing.csv"))
    vals  = readdlm(fname)
    X = Array{Float32}(vals[:, 1:end-1])
    y = Array{Float32}(vals[:, end])
    X, y
end

function load_dataset(::Val{:concrete})
    fname= datadir(joinpath("dataset", "Concrete_Data.xls"))
    data = FileIO.load(fname, "Sheet1")  |> DataFrame |> Matrix
    
    X = Array{Float32}(data[:, 1:end-1])
    y = Array{Float32}(data[:, end])
    X, y
end

function load_dataset(::Val{:yacht})
    fname = datadir(joinpath("dataset", "yacht_hydrodynamics.data"))
    s     = open(fname, "r") do io
        s = read(io, String)
        replace(s, "  " => " ")
    end

    io   = IOBuffer(s)
    vals = readdlm(io, ' ', '\n', header=false)

    X = Array{Float32}(vals[:, 1:end-2])
    y = Array{Float32}(vals[:, end-1])
    X, y
end

function load_dataset(::Val{:naval})
    fname = datadir(joinpath("dataset", "naval_propulsion.txt"))
    s     = open(fname, "r") do io
        s = read(io, String)
        s = replace(s, "   " => " ")
    end

    io   = IOBuffer(s)
    vals = readdlm(io, ' ', '\n', header=false)
    vals = vals[:,2:end]

    X = Array{Float32}(vals[:, 1:end-2])
    y = Array{Float32}(vals[:, end-1])

    feature_idx = 1:size(X,2)
    X = Float64.(X[:, setdiff(feature_idx, (9, 12))])

    X, y
end

function load_dataset(::Val{:energy})
    fname= datadir(joinpath("dataset", "Concrete_Data.xls"))
    data = FileIO.load(fname, "Sheet1")  |> DataFrame |> Matrix
    
    X = Array{Float32}(data[:, 1:end-2])
    y = Array{Float32}(data[:, end])
    X, y
end

function propagate_linear(M, V, m_prev, v_prev)
    scaling = size(m_prev, 1)
    m_α     = M * m_prev / sqrt(scaling)
    v_α     = ((M.^2)*v_prev + V*(m_prev.^2) + V*v_prev) / scaling
    m_α, v_α
end

function propagate_relu(m_α, v_α)
    α  = m_α ./ sqrt.(v_α)
    γ  = normpdf.(-α) ./ normcdf.(α)

    unstable_idx    = α .< -30
    α_unstable      = α[unstable_idx]
    γ[unstable_idx] = -α_unstable - (1 ./ α_unstable) + (2 ./(α_unstable).^3)

    v′ = m_α + sqrt.(v_α).*γ

    m_b = StatsFuns.normcdf.(α) .* v′
    v_b = m_α .* v′ .* StatsFuns.normcdf.(-α) +
        StatsFuns.normcdf.(α) .* v_α .* (1 .- γ.*(γ .+ α))
    v_b = max.(v_b, 1e-12)

    m_b, v_b
end

function test()
    m_prev = randn(2, 2)*3
    v_prev = exp.(randn(2, 2)*2)
    M      = randn(2, 2)*3
    V      = exp.(randn(2, 2)*2)
    m, v   = propagate_linear(M, V, m_prev, v_prev)
    m      = m[:]
    v      = v[:]
    
    x = rand(MvNormal(m_prev[:], v_prev[:]), 4096)
    W = rand(MvNormal(M[:], V[:]), 4096)
    y = mapreduce(hcat, 1:size(W,2)) do i
        x′ = reshape(x[:,i], (2, 2))
        W′ = reshape(W[:,i], (2, 2))
        y  = W′ * x′
        reshape(y, :)
    end

    display(Plots.histogram(y[1,:], normed=true))
    display(Plots.plot!(Normal(m[1], sqrt(v[1]))))
end

# function full_covariance_meanfield(rng::Random.AbstractRNG, model::DynamicPPL.Model)
#     # Setup.
#     varinfo = DynamicPPL.VarInfo(model)
#     num_params = length(varinfo[DynamicPPL.SampleFromPrior()])

#     # initial params
#     μ = randn(rng, num_params)
#     σ = StatsFuns.softplus.(randn(rng, num_params))

#     # Construct the base family.
#     d = DistributionsAD.TuringMvNormal(μ, σ)

#     # Construct the bijector constrained → unconstrained.
#     b = Bijectors.bijector(model; varinfo=varinfo)

#     # We want to transform from unconstrained space to constrained,
#     # hence we need the inverse of `b`.
#     return Bijectors.transformed(d, inv(b))
# end

function run_task(prng::Random.AbstractRNG,
                  task::Union{Val{:wine},
                              Val{:concrete},
                              Val{:yacht},
                              Val{:naval},
                              Val{:boston},
                              Val{:toy}},
                  objective,
                  n_mc;
                  defensive_weight=nothing,
                  show_progress=true)
    X, y = load_dataset(task)
    X_train, y_train, X_test, y_test = prepare_dataset(prng, X, y)

    X_train    = Array{Float32}(X_train')
    X_test     = Array{Float32}(X_test')
    μ_X        = mean(X_train, dims=2)[:,1]
    σ_X        = std(X_train, dims=2)[:,1]
    X_train  .-= μ_X
    X_test   .-= μ_X
    X_train  ./= σ_X
    X_test   ./= σ_X

    X_train = vcat(X_train, ones(1, size(X_train, 2)))
    X_test  = vcat(X_test,  ones(1, size(X_test,  2)))

    μ_y       = mean(y_train)
    σ_y       = std(y_train)
    y_train .-= μ_y
    y_train  /= σ_y

    n_hidden = 50
    model    = bnn(X_train, y_train, n_hidden)

    #AdvancedVI.setadbackend(:forwarddiff)
    #Turing.Core._setadbackend(Val(:forwarddiff))
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))
    #AdvancedVI.setadbackend(:zygote)
    #Turing.Core._setadbackend(Val(:zygote))

    varinfo  = DynamicPPL.VarInfo(model)
    varsyms  = keys(varinfo.metadata)
    n_params = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ varsyms])
    θ        = randn(prng, 2*n_params)*0.1
    q        = Turing.Variational.meanfield(model)
    q        = AdvancedVI.update(q, θ)

    i      = 1
    #k_hist = []
    function plot_callback(ℓπ, λ)
        q′         = AdvancedVI.update(q, λ)
        W1_μ, W1_Σ = get_variational_mean_var(q′, model, Symbol("W1"))
        W2_μ, W2_Σ = get_variational_mean_var(q′, model, Symbol("W2"))
        γ_μ, γ_Σ   = get_variational_mean_var(q′, model, Symbol("γ"))

        γ_μ  = γ_μ[1]
        γ_σ² = γ_Σ[1]

        # Moment matching of Log-normal with Gamma
        γ_α = 1 / (exp(γ_σ²) - 1)
        γ_β = γ_α / exp(γ_μ + γ_σ²/2)

        W1′_μ = reshape(W1_μ, (n_hidden, size(X_test, 1)))
        W2′_μ = reshape(W2_μ, (1,        n_hidden+1))

        W1′_Σ = reshape(W1_Σ, (n_hidden, size(X_test, 1)))
        W2′_Σ = reshape(W2_Σ, (1,        n_hidden+1))

        m_1, v_1 = propagate_linear(W1′_μ, W1′_Σ, X_test, zeros(size(X_test)))
        m_1, v_1 = propagate_relu(m_1, v_1)
        m_1      = vcat(m_1, ones( 1, size(m_1, 2)))
        v_1      = vcat(v_1, zeros(1, size(v_1, 2)))
        m_2, v_2 = propagate_linear(W2′_μ, W2′_Σ, m_1, v_1)
        m_y      = m_2.*σ_y .+ μ_y
        v_y      = v_2*σ_y.*σ_y
        m_y      = m_y[1,:]
        v_y      = v_y[1,:]
        v_noise  = γ_β/γ_α.*σ_y.*σ_y

        # ∫ Normal(y, σ²_y) ∫ Normal(0, 1/γ) × LogNormal(1/γ; μ, σ) dγ
        # ≈ ∫ Normal(y, σ²_y) ∫ Normal(0, 1/γ) × Gamma(γ; α, β)
        # = ∫ Normal(y, σ²_y) × TDist(ν=2α, μ=0, σ²=β/α)
        # ≈ ∫ Normal(y, σ²_y) × Normal(0, β/α)
        # = Normal(y, σ²_y+β/α)
        lpd  = mean(logpdf.(Normal.(m_y, sqrt.(v_y .+ v_noise)), y_test))
        rmse = sqrt(Flux.Losses.mse(m_y, y_test, agg=mean))

        (rmse=rmse, lpd=lpd, σ=γ_β/γ_α)
    end

    ν        = Distributions.Product(fill(Cauchy(), n_params))
    n_iter   = 10000
    θ, stats = vi(model, q;
                  objective        = objective,
                  n_mc             = n_mc,
                  n_iter           = n_iter,
                  callback         = plot_callback,
                  rng              = prng,
                  #optimizer       = AdvancedVI.TruncatedADAGrad(),
                  defensive_dist   = ν,
                  defensive_weight = defensive_weight,
                  optimizer        = Flux.ADAM(1e-2),
                  # optimizer        = Flux.Nesterov(1e-2),
                  show_progress    = show_progress
                  )
    Dict.(pairs.(stats))
end

