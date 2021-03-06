
mutable struct JSA <: AdvancedVI.VariationalObjective
    z::RV{Float64}
    iter::Int
end

function JSA()
    JSA(RV{Float64}(Vector{Float64}(), 0), 1)
end

function init_state!(msc::JSA, rng::Random.AbstractRNG, rand_q, λ0, ℓπ, n_mc)
    z        = rand_q(rng, λ0)
    msc.z    = RV{Float64}(z, ℓπ(z))
end

function grad!(
    rng::Random.AbstractRNG,
    vo::JSA,
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

    rs  = Vector{Float64}(    undef, n_mc)
    ℓws = Vector{Float64}(    undef, n_mc)
    zs  = Vector{RV{Float64}}(undef, n_mc)
    for i = 1:n_mc
        vo.z, α, ℓw, acc = imh_kernel(rng, vo.z, ℓπ, λ, rand_q_def, ℓq_def)
        ℓws[i] = ℓw
        zs[i]  = vo.z 
        rs[i]  = 1-α
    end
    rej_rate = mean(rs)
    cent     = -mean(ℓws)

    f(λ′) = mapreduce(z_rvᵢ -> -ℓq(λ′, z_rvᵢ.val), +, zs) / n_mc

    turing_gradient!(f, λ, out)

    vo.iter += 1

    (crossent = cent,
     rej_rate = rej_rate)
end
