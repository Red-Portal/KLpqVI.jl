
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

Turing.@model gaussian(z, d) = begin
    μ  = Vector{Real}(undef, d)
    s  = Vector{Real}(undef, d)
    s .~ Gamma(1, 1)
    σ  = sqrt.(s)
    μ .~ Normal.(0.0, σ)
    z .~ MvNormal(μ, σ)
end
