
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
using Zygote
using Base.Iterators

function kl_divergence(p::MvNormal,
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

function cis(prng, zₜ₋₁, q, p, N)
    z′ = hcat(rand(prng, q, N), zₜ₋₁)
    ℓw = logpdf.(Ref(p), eachcol(z′)) - logpdf.(Ref(q), eachcol(z′))
    ℓZ = StatsFuns.logsumexp(ℓw)
    w  = exp.(ℓw .- ℓZ)
    w /= sum(w)
    zₜ = z′[:,rand(prng, Categorical(w))]
    r  = w[end]
    zₜ, r
end

function imh(prng, zₜ₋₁, q, p)
    z′  = rand(prng, q)
    ℓw′ = logpdf(p, z′)   - logpdf(q, z′)
    ℓw  = logpdf(p, zₜ₋₁) - logpdf(q, zₜ₋₁)
    α   = min(1, exp(ℓw′ - ℓw))
    if(rand(prng) < α)
        z′, 1 - α
    else
        zₜ₋₁, 1 - α
    end
end

function cisrb(prng, zₜ₋₁, q, p, N)
    z′ = hcat(rand(prng, q, N), zₜ₋₁)
    ℓw = logpdf.(Ref(p), eachcol(z′)) - logpdf.(Ref(q), eachcol(z′))
    ℓZ = StatsFuns.logsumexp(ℓw)
    w  = exp.(ℓw .- ℓZ)
    w /= sum(w)
    r  = w[end]
    z′, w, r
end

function score(q, z)
    Zygote.gradient(μ -> logpdf(MvNormal(μ, q.Σ), z), q.μ)[1]
end

function cis_estimate(prng, q, p, p_z, N, n_iter)
    z₀     = rand(prng, p_z)
    n_dims = length(z₀)
    scores = Array{Float64}(undef, n_dims, n_iter)
    zₜ     = z₀
    for t = 1:n_iter
        q_z    = MvNormal(randn(n_dims), exp.(randn(n_dims) .+ 1))
        zₜ, _  = cis(prng, zₜ, q_z, p, N)
        ∇logqₜ =  score(q, zₜ)
        scores[:,t] = ∇logqₜ
    end
    scores
end

function cisrb_estimate(prng, q, p, p_z, N, n_iter)
    z₀     = rand(prng, p_z)
    n_dims = length(z₀)
    scores = Array{Float64}(undef, n_dims, n_iter)
    zₜ     = z₀
    for t = 1:n_iter
        q_z         = MvNormal(randn(n_dims), exp.(randn(n_dims) .+ 1))
        z′, w, _    = cisrb(prng, zₜ, q_z, p, N)
        dqdzs       = hcat(score.(Ref(q), eachcol(z′))...)
        zₜ          = z′[:,rand(prng, Categorical(w))]
        scores[:,t] = dqdzs*w
    end
    scores
end

function seqimh_estimate(prng, q, p, p_z, N, n_iter)
    z₀     = rand(prng, p_z)
    n_dims = length(z₀)
    scores = Array{Float64}(undef, n_dims, n_iter)
    zₜ     = z₀
    zs     = Array{eltype(z₀)}(undef, n_dims, N)
    for t = 1:n_iter
        q_z    = MvNormal(randn(n_dims), exp.(randn(n_dims) .+ 1))
        for i = 1:N
            zₜ, _   = imh(prng, zₜ, q_z, p)
            zs[:,i] = zₜ
        end
        scores[:,t] = mean(score.(Ref(q), eachcol(zs)))
    end
    scores
end

function parimh_estimate(prng, q, p, p_z, N, n_iter)
    z₀s    = rand(prng, p_z, N)
    n_dims = size(z₀s, 1)
    scores = Array{Float64}(undef, n_dims, n_iter)
    zₜs    = z₀s
    for t = 1:n_iter
        q_z = MvNormal(randn(n_dims), exp.(randn(n_dims) .+ 1))
        for i = 1:N
            z, _        = imh(prng, zₜs[:,i], q_z, p)
            zₜs[:,i]    = z
        end
        scores[:,t] = mean(score.(Ref(q), eachcol(zₜs)))
    end
    scores
end

function score_expectation(p, q)
    q.Σ \ (p.μ - q.μ)
end

function score_variance(scores)
    diag_cov = var(scores)
    norm(diag_cov, 1)
end

function run_simulation(prng, estimate, N, n_points, biased_init, name)
    n_iter    = 20
    n_dims    = 10
    n_samples = 128
    #n_samples = 32
    q_μs      = randn(prng, n_dims, n_points)
    q_logσs   = randn(prng, n_dims, n_points)
    qs        = MvNormal.(eachcol(q_μs), eachcol(exp.(q_logσs)))
    
    p_μ = randn(prng, n_dims)
    p_σ = exp.(randn(prng, n_dims))
    p   = MvNormal(p_μ, p_σ)

    μs    = randn(prng, n_dims, n_points)
    logσs = randn(prng, n_dims, n_points)
    q0s   = if biased_init
        MvNormal.(eachcol(μs), eachcol(exp.(logσs)))
    else
        p
    end

    data = @showprogress pmap(enumerate(qs)) do (i, q)
        samples = Array{Float64}(undef, n_dims, n_samples, n_iter)
        for i = 1:n_samples
            q0 = if biased_init
                q0s[i]
            else
                q0s
            end
            samples[:,i,:] = estimate(prng, q, p, q0, N, n_iter)
        end

        E∇logq   = score_expectation(p, q)
        bias     = mapslices(Δx -> norm(mean(Δx), 1), samples .- E∇logq, dims=(1,2))[1,1,:]
        variance = mapslices(x  -> score_variance(x), samples,           dims=(1,2))[1,1,:]
        klpq     = kl_divergence(p, q)

        [Dict(:bias        => bias[t],
              :variance    => variance[t],
              :klpq        => klpq,
              :N           => N,
              :biased_init => biased_init,
              :iteration   => t,
              :name        => name) for t = 1:n_iter]
    end
    Iterators.flatten(data)
end

function main()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    Random123.set_counter!(prng, 0)

    # n_reps = 4
    # d = Dict[]
    # append!(d, run_simulation(prng, seqimh_estimate, 16, n_reps, false, :parimh))
    # append!(d, run_simulation(prng, parimh_estimate, 16, n_reps, false, :seqimh))
    # return DataFrame(d)

    #display(scatter( d1[:, :klpq], d1[:, :variance], xscale=:log10, yscale=:log10, alpha=0.5))
    #display(scatter!(d2[:, :klpq], d2[:, :variance], xscale=:log10, yscale=:log10, alpha=0.5))
    #return

    n_rep = 2048

    all_data = Dict[]
    #for biased_init ∈ [true, false]
    biased_init = false
        for N ∈ [4, 8, 16]
            data = run_simulation(prng, parimh_estimate, N, n_rep, biased_init, :parimh)
            append!(all_data, data)
        end

        for N ∈ [4, 8, 16]
            data = run_simulation(prng, parimh_estimate, N, n_rep, biased_init, :seqimh)
            append!(all_data, data)
        end

        for N ∈ [4, 8, 16]
            data = run_simulation(prng, cis_estimate, N, n_rep, biased_init, :cis)
            append!(all_data, data)
        end

        for N ∈ [4, 8, 16]
            data = run_simulation(prng, cisrb_estimate, N, n_rep, biased_init, :cisrb)
            append!(all_data, data)
        end
    #end
    FileIO.save(datadir("exp_raw", "approximation_simulation_chain.jld2"), "data", DataFrame(all_data))
end
