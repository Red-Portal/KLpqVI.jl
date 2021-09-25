
using DrWatson
@quickactivate "KLpqVI"

using Memoization
using ReverseDiff
using Plots, StatsPlots
using Flux
using ForwardDiff
using Zygote
using OnlineStats
using Random123
using ProgressMeter
using DelimitedFiles
using DataFrames
using FileIO, JLD2
using DataFramesMeta

include(srcdir("KLpqVI.jl"))
include("task/task.jl")

function kl(p, q)
    σ₁ = p.σ
    μ₁ = p.μ
    σ₂ = q.σ
    μ₂ = q.μ
    log(σ₂ / σ₁) + (σ₁.^2 + (μ₁ - μ₂).^2)/(2*σ₂.^2) - 1/2
end

function cis(prng, zₜ₋₁, q, p, N)
    z′ =  vcat(rand(prng, q, N), zₜ₋₁)
    ℓw = logpdf.(Ref(p), z′) - logpdf.(Ref(q), z′)
    ℓZ = StatsFuns.logsumexp(ℓw)
    w  = exp.(ℓw .- ℓZ)
    w /= sum(w)
    zₜ = rand(prng, Categorical(w))
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
    z′ = vcat(rand(prng, q, N), zₜ₋₁)
    ℓw = logpdf.(p, z′) - logpdf.(q, z′)
    ℓZ = StatsFuns.logsumexp(ℓw)
    w  = exp.(ℓw .- ℓZ)
    w /= sum(w)
    r  = w[end]
    z′, w, r
end

function score(q, z)
    Zygote.gradient(μ -> logpdf(Normal(μ, q.σ), z), q.μ)[1]
end

function cis_estimate(prng, zₜ₋₁, q, p, N)
    zₜ, r = cis(prng, zₜ₋₁, q, p, N)
    score(q, zₜ), r
end

function cisrb_estimate(prng, zₜ₋₁, q, p, N)
    z′, w, r = cisrb(prng, zₜ₋₁, q, p, N)
    dqdzs = score.(Ref(q), z′)
    dot(dqdzs, w), r
end

function imh_estimate(prng, zₜ₋₁, q, p, N)
    chains = imh.(Ref(prng), zₜ₋₁, Ref(q), Ref(p))
    scores = [score(q, state[1]) for state ∈ chains]
    mean(scores), 0
end

function run_simulation(prng, estimator, zₜ₋₁, p, q, N, name)
    n_iters = 2^14
    res     = map(1:n_iters) do i
        estimator(prng, zₜ₋₁, q, p, N)   
    end
    scores   = [resᵢ[1] for resᵢ ∈ res]
    rejrates = [resᵢ[2] for resᵢ ∈ res]

    bias     = mean(scores) - p.μ
    variance = var(scores)
    klpq     = kl(p, q)

    Dict(:bias      => bias,
         :variance  => variance,
         :N         => N,
         :kl        => klpq,
         :rejection => mean(rejrates),
         :q_mean    => q.μ,
         :name      => name,
         )
end

function main()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    Random123.set_counter!(prng, 0)

    data = Dict[]
    p    = Normal(0.0, 1.0)

    ProgressMeter.@showprogress for q_μ  ∈ collect(0.0:1.0:5.0)
        q    = Normal(q_μ, 2.0)
        for N ∈ [1, 2, 4, 8, 16, 32, 64, 128]
            zₜ₋₁ = rand(prng, Normal(0.0, 8.0), N)
            res  = run_simulation(prng, imh_estimate, zₜ₋₁, p, q, N, "pimh_diffuse")
            push!(data, res)
        end
    end

    ProgressMeter.@showprogress for q_μ  ∈ collect(0.0:1.0:5.0)
        q    = Normal(q_μ, 2.0)
        for N ∈ [1, 2, 4, 8, 16, 32, 64, 128]
            zₜ₋₁ = rand(prng, p, N)
            res  = run_simulation(prng, imh_estimate, zₜ₋₁, p, q, N, "pimh")
            push!(data, res)
        end
    end

    zₜ₋₁ = [rand(prng, p)]
    ProgressMeter.@showprogress for q_μ  ∈ collect(0.0:1.0:5.0)
        q    = Normal(q_μ, 2.0)
        for N ∈ [1, 2, 4, 8, 16, 32, 64, 128]
            res = run_simulation(prng, cisrb_estimate, zₜ₋₁, p, q, N, "cisrb")
            push!(data, res)
        end
    end

    ProgressMeter.@showprogress for q_μ  ∈ collect(0.0:1.0:5.0)
        q    = Normal(q_μ, 2.0)
        for N ∈ [1, 2, 4, 8, 16, 32, 64, 128]
            res = run_simulation(prng, cis_estimate, zₜ₋₁, p, q, N, "cis")
            push!(data, res)
        end
    end
    df     = DataFrame(data)
    q_μs   = unique(df.q_mean)
    names  = unique(df.name)

    processed = Dict()
    for name ∈ names
        for q_μ ∈ q_μs
            elem_name = "$(name)_$(q_μ)"
            filtered  = @where(df, (:name .== name) .& (:q_mean .== q_μ))

            processed[elem_name * "_x"]    = filtered.N
            processed[elem_name * "_var"]  = filtered.variance
            processed[elem_name * "_bias"] = filtered.bias
        end
    end
    FileIO.save(datadir("exp_pro", "approximation_simulation.jld2"), processed)
end
