
mutable struct MSC_VR <: AdvancedVI.VariationalObjective
    z::Matrix{Float64}
    method::Symbol
end

function MSC_VR(method)
    return MSC_VR(Array{Float64}(undef, 0, 0), method)
end

function init_state!(msc::MSC_VR, rng, q)
    msc.z = if msc.method == :imh_par
        rand(rng, q, 16)
    else
        rand(rng, q, 1)
    end
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
    #weight(x_) = -logaddexp(log(0.5), logpdf(q, x_) - ℓπ(x_))
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

function evaluate_gradient!(alg::AdvancedVI.VariationalInference{<:AdvancedVI.ForwardDiffAD},
                            f::Function,
                            θ::AbstractVector,
                            out::DiffResults.MutableDiffResult)
    chunk_size = AdvancedVI.getchunksize(typeof(alg))
    chunk      = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config     = ForwardDiff.GradientConfig(f, θ, chunk)
    ForwardDiff.gradient!(out, f, θ, config)
end

function AdvancedVI.grad!(
    rng::Random.AbstractRNG,
    vo::MSC_VR,
    alg::AdvancedVI.VariationalInference{<:AdvancedVI.ForwardDiffAD},
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

    if vo.method == :imh
        z         = vo.z[:,1]
        z, α      = imh_kernel(rng, z, logπ, q′)
        vo.z[:,1] = z
        println(1-α)

        f_imh(θ) = begin
            q_θ = if (q isa Distribution)
                AdvancedVI.update(q, θ)
            else
                q(θ)
            end
            -logpdf(q_θ, z)
        end
        evaluate_gradient!(alg, f_imh, θ, out)

    elseif vo.method == :imh_par
        res = map(eachcol(vo.z)) do zᵢ
            z, α = imh_kernel(rng, zᵢ, logπ, q′)
            (z=z, r=1-α)
        end

        z = hcat([resᵢ.z for resᵢ ∈ res]...)
        r = mean([resᵢ.r for resᵢ ∈ res])

        vo.z = z

        f_imh_par(θ) = begin
            q_θ = if (q isa Distribution)
                AdvancedVI.update(q, θ)
            else
                q(θ)
            end
            nlogq = mapslices(zᵢ -> -logpdf(q_θ, zᵢ), dims=1, z)[1,:]
            mean(nlogq)
        end
        evaluate_gradient!(alg, f_imh_par, θ, out)

    elseif vo.method == :default
        z         = vo.z[:,1]
        z, w      = cis(rng, z, logπ, q′, n_samples)
        idx       = rand(rng, Categorical(w))
        vo.z[:,1] = z[:,idx]

        println(w)

        ess = 1 / sum(w.^2)
        println(ess)

        f_cis(θ) = begin
            q_θ = if (q isa Distribution)
                AdvancedVI.update(q, θ)
            else
                q(θ)
            end
            nlogq = mapslices(zᵢ -> -logpdf(q_θ, zᵢ), dims=1, z)[1,:]
            dot(w, nlogq)
        end
        evaluate_gradient!(alg, f_cis, θ, out)
    end
end
