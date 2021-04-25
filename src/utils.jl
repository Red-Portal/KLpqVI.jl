
struct RV{T<:Real}
    val::Vector{T}
    prob::T
end

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
