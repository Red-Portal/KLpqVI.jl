
mutable struct PMCSA <: AdvancedVI.VariationalObjective
    zs::Vector{RV{Float64}}
    iter::Int
end

function PMCSA()
    PMCSA(Array{RV{Float64}}(undef, 0), 1)
end

function init_state!(msc::PMCSA, rng::Random.AbstractRNG, rand_q, λ0, ℓπ, n_mc)
    zs     = [rand_q(rng, λ0) for i = 1:n_mc]
    ℓπs    = ℓπ.(zs) 
    msc.zs = RV{Float64}.(zs, ℓπs)
end

function grad!(
    rng::Random.AbstractRNG,
    vo::PMCSA,
    ℓq,
    ℓq_def,
    rand_q,
    rand_q_def,
    ℓjac,
    ℓπ,
    λ::AbstractVector{<:Real},
    n_mc::Int,
    out::DiffResults.MutableDiffResult)
    rej_rate = 0 

    rs  = Vector{Float64}(undef, n_mc)
    ℓws = Vector{Float64}(undef, n_mc)
    for i = 1:length(vo.zs)
        z, α, ℓw, acc = imh_kernel(rng, vo.zs[i], ℓπ, λ, rand_q_def, ℓq_def)
        vo.zs[i] = z 
        ℓws[i]   = ℓw
        rs[i]    = 1-α
    end
    rej_rate = mean(rs)
    cent     = -mean(ℓws)

    f(λ′) = mapreduce(z_rvᵢ -> -ℓq(λ′, z_rvᵢ.val), +, vo.zs) / n_mc

    turing_gradient!(f, λ, out)

    vo.iter += 1

    (crossent = cent,
     rej_rate = rej_rate)
end
