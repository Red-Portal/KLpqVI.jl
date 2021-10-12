
function cis(rng::Random.AbstractRNG,
             z0::RV,
             ℓπ::Function,
             q,
             n_samples::Int)
    n_dims = length(z0.val)
    zs     = Matrix{Float64}(undef, n_dims, n_samples+1)
    ℓqs    = Vector{Float64}(undef, n_samples+1)
    ℓps    = Vector{Float64}(undef, n_samples+1)

    zs[:,1] = z0.val 
    ℓps[1]  = z0.prob
    ℓqs[1]  = logpdf(q, z0.val)
    for i = 2:n_samples+1
        _, z, _, logq = Bijectors.forward(rng, q)
        zs[:,i] = z
        ℓqs[i]  = logq
        ℓps[i]  = ℓπ(z)
    end
    ℓw = ℓps - ℓqs
    ℓZ = StatsFuns.logsumexp(ℓw)
    w  = exp.(ℓw .- ℓZ) 
    w  = w / sum(w) # Necessary because of numerical accuracy
    zs, w, ℓw, ℓps
end

# function cis(rng::Random.AbstractRNG,
#              z0::RV,
#              ℓπ::Function,
#              q,
#              n_samples::Int)
#     n_dims = length(z0.val)
#     zs     = Matrix{Float64}(undef, n_dims, n_samples+1)
#     ℓqs    = Vector{Float64}(undef, n_samples+1)
#     ℓps    = Vector{Float64}(undef, n_samples+1)

#     ϵ  = 0.01
#     ν  = 3
#     q0 = Bijectors.TransformedDistribution(
#         Distributions.IsoTDist(ν, length(z0.val), 3.0),
#         q.transform)

#     zs[:,1] = z0.val 
#     ℓps[1]  = z0.prob
#     ℓqs[1]  = logaddexp(log(1 - ϵ) + logpdf(q, z0.val),  log(ϵ) + logpdf(q0, z0.val))
#     for i = 2:n_samples+1
#         _, z, _, logq = Bijectors.forward(rng, q)

#         z_prev = nothing
#         if rand(rng, Bernoulli(ϵ))
#             z_prev = z
#             z = rand(rng, q0)
#         end
#         logq = logaddexp(log(1 - ϵ) + logpdf(q, z),  log(ϵ) + logpdf(q0, z))

#         if isinf(logq)
#             println(logpdf(q.dist,  x))
#             println(logpdf(q0.dist, x))
#             println(logpdf(q,  q.transform(x)))
#             println(logpdf(q0, q.transform(x)))
            
#             println(log(1 - ϵ) + logpdf(q, z))
#             println(log(ϵ) + logpdf(q0, z))
#             println(logpdf(q, z))
#             println(logpdf(q0, z))
#             throw()
#         end

#         zs[:,i] = z
#         ℓqs[i]  = logq
#         ℓps[i]  = ℓπ(z)
#     end
#     ℓw = ℓps - ℓqs
#     ℓZ = StatsFuns.logsumexp(ℓw)
#     w  = exp.(ℓw .- ℓZ) 
#     w  = w / sum(w) # Necessary because of numerical accuracy
#     zs, w, ℓw, ℓps
# end
