
struct LDAMeanField <: Distributions.ContinuousMultivariateDistribution
    θs::Vector{Dirichlet}
    ϕs::Vector{Dirichlet}
    M::Int
    K::Int
    V::Int
end

function LDAMeanField(θs::AbstractVector{<:Dirichlet},
                      ϕs::AbstractVector{<:Dirichlet})
    @assert length(θs[1]) == length(ϕs)
    return LDAMeanField(θs, ϕs, length(θs), length(ϕs), length(ϕs[1]))
end

function LDAMeanField(λ::AbstractVector, M::Int, K::Int, V::Int)
    @assert length(λ) == M*K + K*V
    α_θ = reshape(view(λ, 1:M*K), (M,K))
    α_ϕ = reshape(view(λ, M*K+1:length(λ)), (K,V))
    θs  = map(x -> Dirichlet(Array(x)), eachrow(α_θ))
    ϕs  = map(x -> Dirichlet(Array(x)), eachrow(α_ϕ))
    return LDAMeanField(θs, ϕs)
end

Base.length(q::LDAMeanField) = q.M*q.K + q.K*q.V

function Distributions._rand!(prng::Random.AbstractRNG,
                              q::LDAMeanField,
                              z::AbstractVector{<:Real})
    K = q.K
    M = q.M
    V = q.V

    @simd for m = 1:M
        begin_idx = (m-1)*K+1
        end_idx   = m*K
        @inbounds z[begin_idx:end_idx] = rand(prng, q.θs[m])
    end
    start = M*K
    @simd for k = 1:K
        begin_idx = start + (k-1)*V + 1
        end_idx   = start + k*V
        @inbounds z[begin_idx:end_idx] = rand(prng, q.ϕs[k])
    end
    z
end

function StatsBase.params(q::LDAMeanField)
    λ = Array{Float64}(undef, length(q))
    K = q.K
    M = q.M
    V = q.V
    @simd for m = 1:M
        @inbounds α = q.θs[m].alpha
        begin_idx = (m-1)*K + 1
        end_idx   = m*K
        @inbounds λ[begin_idx:end_idx] = StatsFuns.invsoftplus.(α)
    end
    start = M*K
    @simd for k = 1:K
        @inbounds α = q.ϕs[k].alpha
        begin_idx = start + (k-1)*V + 1 
        end_idx   = start + k*V
        @inbounds λ[begin_idx:end_idx] = StatsFuns.invsoftplus.(α)
    end
    λ
end

function AdvancedVI.update(q::LDAMeanField,
                           λ::AbstractVector{<:Real})
    K  = q.K
    M  = q.M
    V  = q.V
    θs = Array{Dirichlet}(undef, M)
    @simd for m = 1:M
        begin_idx = (m-1)*K + 1
        end_idx   = m*K
        @inbounds θs[m] = Dirichlet(StatsFuns.softplus.(λ[begin_idx:end_idx]),
                                    check_args=true)
    end
    start = M*K
    ϕs    = Array{Dirichlet}(undef, K)
    @simd for k = 1:K
        begin_idx = start + (k-1)*V + 1
        end_idx   = start + k*V
        @inbounds ϕs[k] = Dirichlet(StatsFuns.softplus.(λ[begin_idx:end_idx]),
                                    check_args=true)
    end
    LDAMeanField(θs, ϕs)
end

function Distributions._logpdf(q::LDAMeanField,
                               z::AbstractVector{<:Real})
    res = 0
    K   = q.K
    M   = q.M
    V   = q.V
    @simd for m = 1:M
        begin_idx = (m-1)*K + 1
        end_idx   = m*K
        @inbounds res += logpdf(q.θs[m], z[begin_idx:end_idx])
    end
    start = M*K
    @simd for k = 1:K
        begin_idx = start + (k-1)*V + 1
        end_idx   = start + k*V
        @inbounds res += logpdf(q.ϕs[k], z[begin_idx:end_idx])
    end
    res
end

function get_ϕ_latent(q::LDAMeanField)
    ϕ = Array{Float64}(undef, q.V, q.K)
    for k = 1:q.K
        ϕ[:,k] = q.ϕs[k].alpha
    end
    ϕ
end
