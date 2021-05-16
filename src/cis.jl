
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
