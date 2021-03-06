
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

function sample_variable(prng, q, model, varsym)
    vi  = DynamicPPL.VarInfo(model)
    z  = rand(prng, q)
    DynamicPPL.setall!(vi, z)
    vi.metadata[varsym].vals
end

function sample_variable(prng, q, model, varsym, n_samples)
    hcat([sample_variable(prng, q, model, varsym) for i = 1:n_samples]...)
end

function get_variational_mode(q, model, varsym)
    vi  = DynamicPPL.VarInfo(model)
    μ_η = mean(q.dist)
    μ_z = q.transform(μ_η)
    DynamicPPL.setall!(vi, μ_z)
    vi.metadata[varsym].vals
end

function get_variational_mean_var(q, model, varsym)
    vi  = DynamicPPL.VarInfo(model)
    μ_η = mean(q.dist)
    Σ_η = var(q.dist)
    DynamicPPL.setall!(vi, μ_η)
    μ = vi.metadata[varsym].vals

    vi  = DynamicPPL.VarInfo(model)
    DynamicPPL.setall!(vi, Σ_η)
    Σ = vi.metadata[varsym].vals
    μ, Σ
end

function make_defensive(q, ν, ϵ)
    n_params = length(q)
    q0       = Bijectors.TransformedDistribution(ν, q.transform)

    ℓq(λ_, z_) = begin
        q′ = AdvancedVI.update(q, λ_)
        logaddexp(log(1 - ϵ) + logpdf(q′, z_), log(ϵ) + logpdf(q0, z_))
    end
    rand_q(prng_, λ_) = begin
        if rand(prng_, Bernoulli(ϵ))
            rand(prng_, q0)
        else
            rand(prng_, AdvancedVI.update(q, λ_))
        end
    end
    ℓq, rand_q
end

