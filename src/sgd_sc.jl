
mutable struct SGD_SC <: AdvancedVI.VariationalObjective
    z::Matrix{Float64}
end

function SGD_SC()
    return SGD_SC(Array{Float64}(undef, 0, 0))
end

function init_state!(msc::SGD_SC, rng, q)
    msc.z = rand(rng, q, 1)
end

function mcmc_transition(rng::Random.AbstractRNG,
                         ℓπ::Function,
                         z,
                         q)
    μ, σ = StatsBase.params(q)
    h    = 0.08
    M    = I#diagm(σ.^2)
    T    = q.transform
    ϕ    = inv(T)(z)

    ℓπT(η) = ℓπ(T(η)) + Bijectors.logabsdetjac(T, η)
    
    ∇ℓπ = ForwardDiff.gradient(ℓπT, ϕ)
    p   = MvNormal(ϕ + h^2/2*M*∇ℓπ,  h^2*M)

    ϕ′   = rand(rng, p)
    z′   = T(ϕ′)
    ∇ℓπ′ = ForwardDiff.gradient(ℓπT, ϕ′)
    p′   = MvNormal(ϕ′ + h.^2/2*∇ℓπ′, h.^2*M)

    ℓα = min(0, (ℓπT(ϕ′) - logpdf(p, ϕ′)) - (ℓπT(ϕ) - logpdf(p′, ϕ)))
    if(log(rand(rng)) < ℓα)
        z′
    else
        z
    end
end

function systematic_sampling(prng,
                             weights::AbstractVector,
                             n_resample=length(weights))
    N  = length(weights)
    Δs = 1/n_resample
    u  = rand(prng, Uniform(0.0, Δs))
    s  = 1

    resample_idx = zeros(Int64, n_resample)
    stratas      = cumsum(weights)
    @inbounds for i = 1:n_resample
        while(u > stratas[s] && s < N)
            s += 1
        end
        resample_idx[i] = s
        u += Δs
    end
    resample_idx
end

function cis_coupled(rng::Random.AbstractRNG,
                     ℓπ::Function,
                     q,
                     n_samples::Int)
    T = q.transform
    z_tups₊ = map(1:div(n_samples, 2)) do i
        ϕ, z, _, logq = Bijectors.forward(rng, q)
        (ϕ=ϕ, z=z, logq=logq)
    end

    ϕμ     = mean(q.dist)
    ϕΣ     = cov(q.dist)
    L      = cholesky(ϕΣ).L
    center = (L + I)*ϕμ

    z_tups₋ = map(z_tups₊) do z_tup₊
        ϕ₊ = z_tup₊.ϕ
        ϕ₋ = center - ϕ₊
        z₋ = T(ϕ₋)
        println(Bijectors.logpdf(q.dist, ϕ₊))
        println(Bijectors.logpdf(q.dist, ϕ₋))
        (z=z₋, logq=logpdf(q, z₋))
    end
    z_tups = vcat(z_tups₊, z_tups₋)

    z  = [z_tup.z    for z_tup ∈ z_tups]
    ℓq  = [z_tup.logq for z_tup ∈ z_tups]
    ℓp  = map(ℓπ, z)
    ℓw  = ℓp - ℓq
    ℓZ  = StatsFuns.logsumexp(ℓw)
    w   = exp.(ℓw .- ℓZ) 
    println(1/sum(w.^2))
    idx = rand(rng, Categorical(w))
    z[idx]
end

function cis(rng::Random.AbstractRNG,
             z0s::AbstractMatrix,
             ℓπ::Function,
             q,
             n_samples::Int)
    z_tups = map(1:n_samples) do i
        _, z, _, logq = Bijectors.forward(rng, q)
        (z=z, logq=logq)
    end
    z    = [z_tup.z    for z_tup ∈ z_tups]
    logq = [z_tup.logq for z_tup ∈ z_tups]

    z   = hcat(z0s, hcat(z...))
    ℓq0 = mapslices(z0ᵢ -> logpdf(q, z0ᵢ), dims=1, z0s)[1,:]
    ℓq  = vcat(ℓq0, logq)
    ℓp  = mapslices(ℓπ, z, dims=1)[1,:]
    ℓw  = ℓp - ℓq
    ℓZ  = StatsFuns.logsumexp(ℓw)
    w   = exp.(ℓw .- ℓZ) 
    println(1/sum(w.^2))
    w, z
end

function AdvancedVI.grad!(
    rng::Random.AbstractRNG,
    vo::SGD_SC,
    alg::AdvancedVI.VariationalInference{<:AdvancedVI.ForwardDiffAD},
    q,
    logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult)

    n_samples = alg.samples_per_step
    q′        = if (q isa Distribution)
        AdvancedVI.update(q, θ)
    else
        q(θ)
    end

    z = mapslices(vo.z, dims=1) do zᵢ
        mcmc_transition(rng, logπ, zᵢ, q′)
    end
    w, z = cis(rng, z, logπ, q′, n_samples)

    idx       = systematic_sampling(rng, w, size(vo.z,2))
    vo.z[:,:] = z[:,idx]
    #vo.z[:,i] = z[rand(rng, Categorical(w))]

    #Z = zeros(length(z), 10000)
    #Z[:,i] = z
    #end
    #println(mean(Z[:,1000:end], dims=2))
    #display(density(Z[2,:]))
    #display(plot(Z[2,1000:end]))
    #throw()

    f(θ) = if (q isa Distribution)
        q_θ   = AdvancedVI.update(q, θ)
        nlogq = mapslices(zᵢ -> -logpdf(q_θ, zᵢ), dims=1, z)[1,:]
        dot(w, nlogq)
    else
        q_θ = q(θ)
        nlogq = mapslices(zᵢ -> -logpdf(q_θ, zᵢ), dims=1, z)[1,:]
        dot(w, nlogq)
    end

    chunk_size = AdvancedVI.getchunksize(typeof(alg))
    chunk      = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config     = ForwardDiff.GradientConfig(f, θ, chunk)
    ForwardDiff.gradient!(out, f, θ, config)
end
