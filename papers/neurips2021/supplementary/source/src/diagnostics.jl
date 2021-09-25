
function rhat(chains::AbstractArray{<:Real,3})
    # n_iter × n_var × n_chain 
    n_vars = size(chains, 2)
    N      = size(chains, 1)
    R̂      = Array{Float64}(undef, n_vars)
    for v = 1:n_vars
        chain_slice = view(chains, :, v, :)
        μ_chain     = mean(chain_slice, dims=1)
        σ²_chain    = var(chain_slice, mean=μ_chain, dims=1, corrected=true)
        W           = mean(σ²_chain)
        var₊        = ((N-1)/N)*W + var(μ_chain; corrected=true)
        R̂[v]        = √(var₊ / W)
    end
    R̂
end

split_rhat(chain::AbstractArray{<:Real,2}) = split_rhat(reshape(chain, (size(chain)..., 1)))

function split_rhat(chains::AbstractArray{<:Real,3})
    # n_iter × n_var × n_chain 
    n_iters  = div(size(chains, 1), 2)
    splitted = cat(view(chains, 1:n_iters, :, :),
                   view(chains, n_iters+1:n_iters*2, :, :), dims=3)
    rhat(splitted)
end

function paretok(prng::Random.AbstractRNG, q, ℓπ, paretok_samples::Int)
    zs   = rand(prng, q, paretok_samples)
    ℓw    = map(zᵢ -> ℓπ(zᵢ) - logpdf(q, zᵢ), eachcol(zs))
    _, k  = psis.psislw(ℓw)
    k
end
