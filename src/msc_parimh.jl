
mutable struct MSC_PIMH <: AdvancedVI.VariationalObjective
    zs::Matrix{Float64}
end

function MSC_PIMH()
    MSC_PIMH(Array{Float64}(undef, 0, 0))
end

function init_state!(msc::MSC_PIMH, rng, q, n_mc)
    msc.zs = rand(rng, q, n_mc)
end

function cis(rng::Random.AbstractRNG,
             z0::AbstractVector,
             ℓπ::Function,
             q,
             n_samples::Int)
    z_tups = map(1:n_samples) do i
        _, z, _, logq = Bijectors.forward(rng, q)
        (z=z, logq=logq)
    end
    z  = [z_tup.z    for z_tup ∈ z_tups]
    z  = vcat([z0], z)

    ℓq  = [z_tup.logq for z_tup ∈ z_tups]
    ℓq  = vcat(logpdf(q, z0), ℓq)
    ℓp  = map(ℓπ, z)
    ℓw  = ℓp - ℓq
    ℓZ  = StatsFuns.logsumexp(ℓw)

    w = exp.(ℓw .- ℓZ) 
    z = hcat(z...)
    z, w
end

function imh_kernel(rng::Random.AbstractRNG,
                    z,
                    ℓπ::Function,
                    q)
    weight(x_) = ℓπ(x_) - logpdf(q, x_)
    z′  = rand(rng, q)
    ℓw  = weight(z)
    ℓw′ = weight(z′)
    α   = min(1.0, exp(ℓw′ - ℓw))
    if(rand(rng) < α)
        z′, α
    else
        z, α
    end
end

function AdvancedVI.grad!(
    rng::Random.AbstractRNG,
    vo::MSC_PIMH,
    alg::AdvancedVI.VariationalInference,
    q,
    logπ,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult)

    n_samples = alg.samples_per_step

    q′ = if (q isa Distribution)
        AdvancedVI.update(q, θ)
    else
        q(θ)
    end

    res = map(eachcol(vo.zs)) do zᵢ
        z, α = imh_kernel(rng, zᵢ, logπ, q′)
        (z=z, r=1-α)
    end

    r     = mean([resᵢ.r for resᵢ ∈ res])
    z     = hcat([resᵢ.z for resᵢ ∈ res]...)
    vo.zs = z

    f(θ) = begin
        q_θ = if (q isa Distribution)
            AdvancedVI.update(q, θ)
        else
            q(θ)
        end
        nlogq = mapslices(zᵢ -> -logpdf(q_θ, zᵢ), dims=1, z)[1,:]
        mean(nlogq)
    end
    gradient!(alg, f, θ, out)
end
