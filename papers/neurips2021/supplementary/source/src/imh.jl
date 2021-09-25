
function imh_kernel(rng::Random.AbstractRNG,
                    z_rv::RV,
                    ℓπ::Function,
                    q)
    _, z′, _, ℓq′ = Bijectors.forward(rng, q)

    ℓw  = z_rv.prob - logpdf(q, z_rv.val)
    ℓp′ = ℓπ(z′)
    ℓw′ = ℓp′ - ℓq′
    α   = min(1.0, exp(ℓw′ - ℓw))
    if(rand(rng) < α)
        RV(z′, ℓp′), α, true
    else
        z_rv, α, false
    end
end
