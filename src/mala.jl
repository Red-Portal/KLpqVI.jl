
function mala(prng::Random.AbstractRNG,
              ∂ℓπ∂θ::Function,
              x::AbstractVector,
              ϵ::Real)
    n_dims = length(x)
    M⁻¹    = Diagonal(I, n_dims)
    ℓq, ∇ℓq = ∂ℓπ∂θ(x)
    q      = MvNormal(x + ϵ/2*∇ℓq, ϵ*M⁻¹)
    x′     = rand(prng, q)

    ℓq′, ∇ℓq′ = ∂ℓπ∂θ(x′)
    q′       = MvNormal(x′ + ϵ/2*∇ℓq′, ϵ*M⁻¹)

    ℓα = min(0, ℓq′ - ℓq - logpdf(q, x′) + logpdf(q′, x))
    if(log(rand(prng)) < ℓα)
        x′, exp(ℓα)
    else
        x, exp(ℓα)
    end
end

function main()
    prng   = MersenneTwister(1)
    π      = MvNormal(zeros(2), [1.0 0.2; 0.2 2.0])

    ∂ℓπ∂θ(x) = begin
        ℓπ  = logpdf(π, x)
        ∇ℓπ = Zygote.gradient(y->logpdf(π, y), x)[1]
        ℓπ, ∇ℓπ
    end

    x = rand(prng, π)
    n = 10000
    X = zeros(2, n)
    μ = Mean()
    for i = 1:n
        x, α  = mala(prng, ∂ℓπ∂θ, x, 1.0)
        fit!(μ, α)
        X[:,i] = x
    end
    println(μ.μ)
    display(scatter(X[1,:], X[2,:]))

    MCMCChains.Chains(reshape(X', (:,2,1)))

    #y = rand(prng, π, n)
    #scatter!(y[1,:], y[2,:])
end
