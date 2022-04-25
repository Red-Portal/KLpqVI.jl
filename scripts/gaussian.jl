
using DrWatson
@quickactivate "KLpqVI"

include(srcdir("KLpqVI.jl"))
include("task/task.jl")

using Plots, StatsPlots
using Flux
using ForwardDiff
using Random123
using DataFramesMeta
using ProgressMeter
using Zygote
using HDF5

function imh(prng, z, ℓπ, q)
    z′ = rand(prng, q)
    ℓw′ = ℓπ(z′) - logpdf(q, z′)
    ℓw  = ℓπ(z)  - logpdf(q, z)
    ℓα  = min(0, ℓw′ - ℓw)
    ℓu  = log(rand(prng))
    if (ℓα > ℓu)
        z′
    else
        z
    end
end

function cis(prng, z, ℓπ, q, N)
    zs′ = rand(prng, q, N)
    ℓws = Array{Float64}(undef, N+1)

    ℓws[2:end] = mapslices(zᵢ -> ℓπ(zᵢ) - logpdf(q, zᵢ), zs′, dims=1)[1,:]
    ℓws[1]     = ℓπ(z) - logpdf(q, z)
    ℓZ         = StatsFuns.logsumexp(ℓws)
    w         = exp.(ℓws .- ℓZ)
    idx       = rand(prng, Categorical(w))
    if idx == 1
        z
    else
        zs′[:,idx-1]
    end
end

function init(prng, ::MSC_PIMH, n_dims, n_reps, n_samples)
    state      = Dict{Symbol,Any}()
    state[:zs] = randn(prng, n_dims, n_samples, n_reps)
    state
end

function init(prng, ::MSC_SIMH, n_dims, n_reps, n_samples)
    state     = Dict{Symbol,Any}()
    state[:z] = randn(prng, n_dims, n_reps)
    state
end

function init(prng, ::MSC, n_dims, n_reps, n_samples)
    state     = Dict{Symbol,Any}()
    state[:z] = randn(prng, n_dims, n_reps)
    state
end

function estimate!(prng, ::MSC, ℓπ, q, λ, s, E∇logq, n_samples, state)
    z      = state[:z] 
    n_reps = size(z, 2)

    for j = 1:n_reps
        z[:,j] = cis(prng, z[:,j], ℓπ, q, n_samples)
    end
    state[:z] = z

    Eηs = mapslices(z -> s(λ, z), z, dims=1)
    Es  = mean(Eηs, dims=2)[:,1]

    var  = mapreduce(+, 1:n_reps) do i
        sum((Eηs[:,i] - Es).^2) / n_reps
    end
    bias = norm(Es - E∇logq, Inf)

    bias, var
end

function estimate!(prng, ::MSC_PIMH, ℓπ, q, λ, s, E∇logq, n_samples, state)
    zs     = state[:zs] 
    n_dims = size(zs, 1)
    n_reps = size(zs, 3)

    for j = 1:n_reps
        for i = 1:n_samples
            zs[:,i,j] = imh(prng, zs[:,i,j], ℓπ, q)
        end
    end
    state[:zs] = zs

    Eηs = mean(mapslices(z -> s(λ, z), zs, dims=1), dims=2)[:,1,:]
    Es  = mean(Eηs, dims=2)[:,1]

    var  = mapreduce(+, 1:n_reps) do i
        sum((Eηs[:,i] - Es).^2) / n_reps
    end
    bias = norm(Es - E∇logq, Inf)

    bias, var
end

function estimate!(prng, ::MSC_SIMH, ℓπ, q, λ, s, E∇logq, n_samples, state)
    z_state = state[:z] 
    n_dims  = size(z_state, 1)
    n_reps  = size(z_state, 2)

    zs = Array{Float64}(undef, n_dims, n_samples, n_reps)
    for j = 1:n_reps
        z      = z_state[:, j]
        for i = 1:n_samples
            z = imh(prng, z, ℓπ, q)
            zs[:,i,j] = z
        end
        z_state[:,j] = z
    end
    state[:z] = z_state

    Eηs = mean(mapslices(z -> s(λ, z), zs, dims=1), dims=2)[:,1,:]
    Es  = mean(Eηs, dims=2)[:,1]

    var  = mapreduce(+, 1:n_reps) do i
        sum((Eηs[:,i] - Es).^2) / n_reps
    end
    bias = norm(Es - E∇logq, Inf)

    bias, var
end

function run_gaussian(seed_key, method, n_mc, σ_init=1.0, stat=true)
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    Random123.set_counter!(prng, seed_key)

    # AdvancedVI.setadbackend(:reversediff)
    # Turing.setadbackend(:reversediff)
    AdvancedVI.setadbackend(:forwarddiff)
    Turing.setadbackend(:forwarddiff)

    n_dims    = 100
    n_iter    = 10000
    n_mc      = n_mc
    n_reps    = 512
    objective = method
    state     = init(prng, method, n_dims, n_reps, n_mc)

    Random123.set_counter!(prng, seed_key)
    ν      = 100
    p      = load_dataset(Val(:gaussian_correlated), 100, ν)
    model  = gaussian(p.μ, p.Σ)
    z_true = rand(prng, p, n_reps)
    iter   = 1

    varinfo  = DynamicPPL.VarInfo(model)
    varsyms  = keys(varinfo.metadata)
    n_params = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ varsyms])
    θ_μ      = randn(prng, n_params)
    θ_σ      = fill(StatsFuns.invsoftplus(σ_init), n_params)
    θ        = vcat(θ_μ, θ_σ)
    q        = Turing.Variational.meanfield(model)
    q        = AdvancedVI.update(q, θ)

    s(λ, z)    = ForwardDiff.gradient(
        λ′ -> begin
            q′   = AdvancedVI.update(q, λ′)
            logpdf(q′, z)
        end, λ)

    iter_hist = []
    kl_hist   = []
    bias_hist = []
    var_hist  = []
    function plot_callback(ℓπ, λ)
        q′ = AdvancedVI.update(q, λ)
        μ  = mean(q′.dist)
        Σ  = cov(q′.dist)
        kl = kl_divergence(p, MvNormal(μ, Σ))

        if mod(iter, 100) == 1
            push!(kl_hist,   kl)
            push!(iter_hist, iter)

            if stat
                E∇logq    = mean(s.(Ref(λ), eachcol(z_true)))
                bias, var = estimate!(prng, objective, ℓπ, q′, λ, s, E∇logq, n_mc, state)           

                push!(bias_hist, bias)
                push!(var_hist,  var)
            end

            iter += 1
            (kl=kl,)
        else
            iter += 1
            (kl=kl,)
        end
    end

    ν        = Distributions.Product(fill(Cauchy(), n_params))
    λ, stats = vi(model, q;
                  objective   = objective,
                  #objective   = AdvancedVI.ELBO(),
                  #objective   = MSC_HMC(0.2, 16),
                  #objective   = SNIS(),
                  defensive_dist   = ν,
                  defensive_weight = nothing,#0.01,
                  n_mc        = n_mc,
                  n_iter      = n_iter,
                  callback    = plot_callback,
                  rng         = prng,
                  #optimizer    = ParameterSchedulers.Scheduler(Sqrt(stepsize), ADAM()),
                  optimizer    = Flux.ADAM(0.01),
                  show_progress = true
                  )

    z_true    = rand(prng, p, 4096)
    s_samples = mapslices(z′ -> s(λ, z′), z_true, dims=1)
    Es        = mean(s_samples, dims=2)[:,1]
    Vs        = mapreduce(+, 1:n_reps) do i
        sum((s_samples[:,i] - Es).^2) / n_reps
    end
    @info("Score Variance", Vs=Vs)

    if stat
        DataFrame(:iter => iter_hist,
                  :kl   => kl_hist,
                  :bias => bias_hist,
                  :var  => var_hist)
    else
        DataFrame(:iter => iter_hist,
                  :kl   => kl_hist)
    end
end

function bias_variance_simulation()
    n_rep = 20
    data  = DataFrame()
        
    method = MSC_SIMH()
    n_mc   = 16
    res    = reduce(vcat, @showprogress pmap(key -> run_gaussian(key, method, n_mc), 1:n_rep))
    res[:,:method] .= "SIMH"
    res[:,:n_mc]   .= n_mc
    data   = vcat(data, res)

    method = MSC_SIMH()
    n_mc   = 64
    res    = reduce(vcat, @showprogress pmap(key -> run_gaussian(key, method, n_mc), 1:n_rep))
    res[:,:method] .= "SIMH"
    res[:,:n_mc]   .= n_mc
    data   = vcat(data, res)

    method = MSC_PIMH()
    n_mc   = 16
    res    = reduce(vcat, @showprogress pmap(key -> run_gaussian(key, method, n_mc), 1:n_rep))
    res[:,:method] .= "PIMH"
    res[:,:n_mc]   .= n_mc
    data   = vcat(data, res)

    method = MSC_PIMH()
    n_mc   = 64
    res    = reduce(vcat, @showprogress pmap(key -> run_gaussian(key, method, n_mc), 1:n_rep))
    res[:,:method] .= "PIMH"
    res[:,:n_mc]   .= n_mc
    data   = vcat(data, res)

    method = MSC_CIS()
    n_mc   = 16
    res    = reduce(vcat, @showprogress pmap(key -> run_gaussian(key, method, n_mc), 1:n_rep))
    res[:,:method] .= "CIS"
    res[:,:n_mc]   .= n_mc
    data   = vcat(data, res)

    method = MSC_CIS()
    n_mc   = 64
    res    = reduce(vcat, @showprogress map(key -> run_gaussian(key, method, n_mc), 1:n_rep))
    res[:,:method] .= "CIS"
    res[:,:n_mc]   .= n_mc
    data   = vcat(data, res)
    JLD2.save("bias_variance.jld2", "data", data)
    data
end

function bias_variance_simulation()
    n_rep = 20
    data  = DataFrame()
        
    for σ_init ∈ exp.(range(log10(0.1), log10(8), length = 10))
        method          = MSC_SIMH()
        n_mc            = 10
        res             = reduce(vcat, @showprogress pmap(key -> run_gaussian(key, method, n_mc, σ_init, Inf), 1:n_rep))
        res[:, :method] .= "SIMH"
        res[:, :n_mc]   .= n_mc
        data            = vcat(data, res)
    end

    JLD2.save("initial_variance.jld2", "data", data)
    data
end

function process_gaussian_data(df)
    α = 0.8

    h5open(datadir("exp_pro", "gaussian.h5"), "w") do io
        write(io, "iteration",  collect(1:100:10000))
        for n_mc ∈ [16, 64]
            for method ∈ ["PIMH", "SIMH", "CIS"]
                df_subset = @chain df begin
                    @subset(:method .== method, :n_mc .== n_mc)
                    @select(:kl, :iter, :var)
                    groupby(:iter)
                    @combine(:kl = median(:kl),
                        :kl⁻ = quantile(:kl, (1 - α) / 2),
                        :kl⁺ = quantile(:kl, (1 / 2 + α / 2)),
                        :var = median(:var),
                        :var⁻ = quantile(:var, (1 - α) / 2),
                        :var⁺ = quantile(:var, (1 / 2 + α / 2)),
                    )
                end

                Δkl⁺  = df_subset[:, :kl⁺] - df_subset[:, :kl]
                Δkl⁻  = df_subset[:, :kl]  - df_subset[:, :kl⁻]
                Δvar⁺ = df_subset[:, :var⁺] - df_subset[:, :var]
                Δvar⁻ = df_subset[:, :var]  - df_subset[:, :var⁻]

                write(io, "$(method)_$(n_mc)_kl",  Array(hcat(df_subset[:, :kl],  Δkl⁺, Δkl⁻)'))
                write(io, "$(method)_$(n_mc)_var", Array(hcat(df_subset[:, :var], Δvar⁺, Δvar⁻)'))
            end
        end
    end
end
