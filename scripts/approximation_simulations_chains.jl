using DrWatson
@quickactivate "KLpqVI"

using DataFrames
using DataFramesMeta
using DelimitedFiles
using Distributed
using Distributions
using FileIO, JLD2
using LinearAlgebra
using Plots, StatsPlots
using ProgressMeter
using PDMats
using Random
using Random123
using StatsFuns
using StatsBase
using Zygote
using Base.Iterators
using QuantileRegressions
using HDF5

@inline function kl_divergence(p::MvNormal,
                               q::MvNormal)
    Σp = p.Σ
    Σq = q.Σ
    μp = p.μ
    μq = q.μ
    D  = length(μp)
    ((logdet(Σq) - logdet(Σp)) - D
    + tr(Σq \ Σp)
    + PDMats.invquad(Σq, μq - μp) )/2
end

@inline function cis(prng, zₜ₋₁, q, p, N)
    z′  = hcat(rand(prng, q, N), zₜ₋₁)
    ℓw  = logpdf.(Ref(p), eachcol(z′)) - logpdf.(Ref(q), eachcol(z′))
    ℓZ  = StatsFuns.logsumexp(ℓw)
    w   = exp.(ℓw .- ℓZ)
    w  /= sum(w)
    idx = rand(prng, Categorical(w))
    @inbounds zₜ = z′[:,idx]
    zₜ, idx == N
end

@inline function imh(prng, zₜ₋₁, q, p)
    z′  = rand(prng, q)
    ℓw′ = logpdf(p, z′)   - logpdf(q, z′)
    ℓw  = logpdf(p, zₜ₋₁) - logpdf(q, zₜ₋₁)
    α   = min(1, exp(ℓw′ - ℓw))
    if(rand(prng) < α)
        z′, true
    else
        zₜ₋₁, false
    end
end

@inline function cisrb(prng, zₜ₋₁, q, p, N)
    z′ = hcat(rand(prng, q, N), zₜ₋₁)
    ℓw = logpdf.(Ref(p), eachcol(z′)) - logpdf.(Ref(q), eachcol(z′))
    ℓZ = StatsFuns.logsumexp(ℓw)
    w  = exp.(ℓw .- ℓZ)
    w /= sum(w)
    z′, w
end

@inline @inbounds function score(q, z)
    Zygote.gradient(μ -> logpdf(MvNormal(μ, q.Σ), z), q.μ)[1]
end

function cis_estimate!(prng, q, p, p_z, N, n_iter, scores_out)
    z₀     = rand(prng, p_z)
    zₜ     = z₀
    accs   = 0
    q_z    = q
    @inbounds for t = 1:n_iter
        zₜ, acc = cis(prng, zₜ, q_z, p, N)
        accs   += acc
        ∇logqₜ  =  score(q, zₜ)
        scores_out[:,t] = ∇logqₜ
    end
    accs / n_iter
end

function cisrb_estimate!(prng, q, p, p_z, N, n_iter, scores_out)
    z₀     = rand(prng, p_z)
    zₜ     = z₀
    q_z    = q
    @inbounds for t = 1:n_iter
        z′, w  = cisrb(prng, zₜ, q_z, p, N)
        dqdzs  = hcat(score.(Ref(q), eachcol(z′))...)
        zₜ     = z′[:,rand(prng, Categorical(w))]
        scores_out[:,t] = dqdzs*w
    end
    1
end

function seqimh_estimate!(prng, q, p, p_z, N, n_iter, scores_out)
    z₀     = rand(prng, p_z)
    n_dims = length(z₀)
    zₜ     = z₀
    zs     = Array{eltype(z₀)}(undef, n_dims, N)
    accs   = 0
    q_z    = q
    @inbounds for t = 1:n_iter
        for i = 1:N
            zₜ, acc = imh(prng, zₜ, q_z, p)
            accs   += acc
            zs[:,i] = zₜ
        end
        scores_out[:,t] = mean(score.(Ref(q), eachcol(zs)))
    end
    accs / n_iter / N
end

function parimh_estimate!(prng, q, p, p_z, N, n_iter, scores_out)
    z₀s    = rand(prng, p_z, N)
    zₜs    = z₀s
    accs   = 0
    q_z    = q
    for t = 1:n_iter
        for i = 1:N
            z, acc   = imh(prng, zₜs[:,i], q_z, p)
            zₜs[:,i] = z
            accs    += acc
        end
        scores_out[:,t] = mean(score.(Ref(q), eachcol(zₜs)))
    end
    accs / n_iter / N
end

function score_expectation(p, q)
    q.Σ \ (p.μ - q.μ)
end

function score_variance(scores)
    diag_cov = var(scores)
    norm(diag_cov, 1)
end

function run_simulation(prng, estimate!, N, n_points, biased_init, name)
    n_iter    = 20
    n_dims    = 10
    n_samples = 128
    q_μs      = rand(prng, MvTDist(10, zeros(n_dims), diagm(fill(0.5, n_dims))), n_points) #randn(prng, n_dims, n_points)*2
    q_logσs   = zeros(n_dims) .+ log(1.5)
    qs        = MvNormal.(eachcol(q_μs), eachcol(exp.(q_logσs)))
    
    p_μ    = zeros(n_dims)
    p_logσ = zeros(n_dims)
    p      = MvNormal(p_μ, exp.(p_logσ))

    μs    = randn(prng, n_dims, n_points)
    logσs = zeros(n_dims) .+ log(2)
    q0s   = if biased_init
        MvNormal.(eachcol(μs), eachcol(exp.(logσs)))
    else
        p
    end

    data = @showprogress pmap(enumerate(qs)) do (i, q)
        samples   = Array{Float64}(undef, n_dims, n_samples, n_iter)
        acc_rates = Array{Float64}(undef, n_samples)
        for i = 1:n_samples
            q0 = if biased_init
                q0s[i]
            else
                q0s
            end
            acc_rate     = estimate!(prng, q, p, q0, N, n_iter, view(samples, :,i,:))
            acc_rates[i] = acc_rate
        end
        avg_acc  = mean(acc_rates)
        E∇logq   = score_expectation(p, q)
        bias     = mapslices(Δx -> norm(mean(Δx), 1), samples .- E∇logq, dims=(1,2))[1,1,:]
        variance = mapslices(x  -> score_variance(x), samples,           dims=(1,2))[1,1,:]
        klpq     = kl_divergence(p, q)

        #println(klpq, ' ', avg_acc)

        [Dict(:bias        => bias[t],
              :variance    => variance[t],
              :klpq        => klpq,
              :N           => N,
              :biased_init => biased_init,
              :iteration   => t,
              :acc         => avg_acc,
              :name        => name) for t = 1:n_iter]
    end
    Iterators.flatten(data)
end

function main()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    Random123.set_counter!(prng, 0)

    # n_reps = 1024
    # d = Dict[]
    # append!(d, run_simulation(prng, seqimh_estimate!, 16, n_reps, false, :parimh))
    # append!(d, run_simulation(prng, parimh_estimate!, 16, n_reps, false, :seqimh))
    # append!(d, run_simulation(prng, cis_estimate!,     16, n_reps, false, :parimh))
    # append!(d, run_simulation(prng, cisrb_estimate!,   16, n_reps, false, :parimh))

    # display(scatter( d1[:, :klpq], d1[:, :variance], xscale=:log10, yscale=:log10, alpha=0.5))
    # display(scatter!(d2[:, :klpq], d2[:, :variance], xscale=:log10, yscale=:log10, alpha=0.5))
    # return

    n_rep = 2048

    all_data = Dict[]
    #for biased_init ∈ [true, false]
        biased_init = true
        for N ∈ [4, 8, 16]
            prng = Random123.Philox4x(UInt64, seed, 8);
            Random123.set_counter!(prng, 0)
            data = run_simulation(prng, seqimh_estimate!, N, n_rep, biased_init, :seqimh)
            append!(all_data, data)
        end

        for N ∈ [4, 8, 16]
            prng = Random123.Philox4x(UInt64, seed, 8);
            Random123.set_counter!(prng, 0)
            data = run_simulation(prng, parimh_estimate!, N, n_rep, biased_init, :parimh)
            append!(all_data, data)
        end

        for N ∈ [4, 8, 16]
            prng = Random123.Philox4x(UInt64, seed, 8);
            Random123.set_counter!(prng, 0)
            data = run_simulation(prng, cis_estimate!, N, n_rep, biased_init, :cis)
            append!(all_data, data)
        end

        for N ∈ [4, 8, 16]
            prng = Random123.Philox4x(UInt64, seed, 8);
            Random123.set_counter!(prng, 0)
            data = run_simulation(prng, cisrb_estimate!, N, n_rep, biased_init, :cisrb)
            append!(all_data, data)
        end
    #end
    FileIO.save(datadir("exp_raw", "approximation_simulation_chain.jld2"), "data", DataFrame(all_data))
end

function process_data()
    df = FileIO.load(datadir("approximation_simulation_chain.jld2"), "data")
    display(df)

    

    h5open(datadir("exp_pro", "simulations.h5"), "w") do io
        t = 20
        for name ∈ ["parimh", "seqimh", "cis", "cisrb"]
            for N ∈ [4, 8, 16]
                data = @chain df begin
                    @subset(:name      .== Symbol(name))
                    @subset(:N         .== N)
                    @subset(:iteration .== t)
                    @select(:klpq, :variance, :bias)
                end

                res = qreg(@formula( variance ~ klpq ), data, 0.5)
                β   = StatsBase.coef(res)
                write(io, "$(name)_$(N)_klpq",      data[:,:klpq])
                write(io, "$(name)_$(N)_variance",  data[:,:variance])
                write(io, "$(name)_$(N)_bias",      data[:,:bias])
                write(io, "$(name)_$(N)_reg_alpha", β[1:1])
                write(io, "$(name)_$(N)_reg_beta",  β[2:2])
            end
        end
    end

    # fetch_bias(N, name) = @chain df begin
    #     @subset(:name .== name)
    #     @subset(:N    .== N)
    #     @subset((:klpq .> 2.0) .& (:klpq .< 2.3))
    #     @select(:iteration, :bias)
    # end

    # N           = 16
    # parimh_bias = fetch_bias(N, :parimh)
    # seqimh_bias = fetch_bias(N, :seqimh)
    # cis_bias    = fetch_bias(N, :cis)
    # cisrb_bias  = fetch_bias(N, :cisrb)

    # display(scatter( parimh_bias[:,:iteration], seqimh_bias[:,:bias], label="parimh", yscale=:log10, alpha=0.2))
    # display(scatter!(seqimh_bias[:,:iteration], parimh_bias[:,:bias], label="cisimh", yscale=:log10, alpha=0.2))
    # display(scatter!(cis_bias[   :,:iteration], cis_bias[   :,:bias], label="cis",    yscale=:log10, alpha=0.2))
    # display(scatter!(cisrb_bias[ :,:iteration], cisrb_bias[ :,:bias], label="cisrb",  yscale=:log10, alpha=0.2))
end
