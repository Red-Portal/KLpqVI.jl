
function gaussian_data(prng::Random.AbstractRNG,
                       n_dims::Int;
                       correlated::Bool=false)
    Σ = if correlated
        Σdist = Wishart(n_dims, diagm(ones(n_dims)))
        rand(prng, Σdist)
    else
        diagm(exp.(randn(prng, n_dims)))
    end
    μ = randn(prng, n_dims)
    MvNormal(μ, Σ)
end

function studentt_data(prng::Random.AbstractRNG,
                       ν::Real,
                       n_dims::Int;
                       correlated::Bool=false)
    Σ = if correlated
        Σdist = Wishart(n_dims, diagm(ones(n_dims)))
        rand(prng, Σdist)
    else
        diagm(exp.(randn(prng, n_dims)))
    end
    μ = randn(prng, n_dims)
    Distributions.mvtdist(ν, μ, Σ)
end

Turing.@model gaussian(μ, Σ) = begin
    z ~ MvNormal(μ, Σ)
end
