
struct VIADBijector{AD, N, B <: Bijectors.Bijector{N}} <: Bijectors.ADBijector{AD, N}
    b::B
end
VIADBijector(d::Distribution) = VIADBijector{Bijectors.ADBackend()}(d)
VIADBijector{AD}(d::Distribution) where {AD} = VIADBijector{AD}(Bijectors.bijector(d))
VIADBijector{AD}(b::B) where {AD, N, B <: Bijectors.Bijector{N}} = VIADBijector{AD, N, B}(b)
(b::VIADBijector)(x) = b.b(x)
(b::Bijectors.Inverse{<:VIADBijector})(x) = inv(b.orig.b)(x)

function Bijectors.jacobian(
    b::Union{<:Bijectors.ADBijector{<:Bijectors.ZygoteAD},
             Bijectors.Inverse{<:Bijectors.ADBijector{<:Bijectors.ZygoteAD}}},
    x::AbstractVector{<:Real}
)
    return Zygote.jacobian(b, x)[1]
end

# function jacobian(
#     b::Union{<:Bijectors.ADBijector{<:Bijectors.TrackerAD},
#              Bijectors.Inverse{<:Bijectors.ADBijector{<:Bijectors.TrackerAD}}},
#     x::Real
# )
#     return Bijectors.data(Bijectors.Tracker.gradient(b, x)[1])
# end

# function jacobian(
#     b::Union{<:Bijectors.ADBijector{<:Bijectors.TrackerAD},
#              Bijectors.Inverse{<:Bijectors.ADBijector{<:Bijectors.TrackerAD}}},
#     x::AbstractVector{<:Real}
# )
#     # We extract `data` so that we don't return a `Tracked` type
#     return Bijectors.data(Bijectors.Tracker.jacobian(b, x))
# end

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

function kl_divergence(δ::AbstractMatrix, q)
    ℓp = -log(size(δ, 2))
    ℓq = map(x -> logpdf(q, x), eachcol(δ))
    mean(ℓp .- ℓq)
end

