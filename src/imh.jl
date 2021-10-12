
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

# function imh_kernel(rng::Random.AbstractRNG,
#                     z_rv::RV,
#                     ℓπ::Function,
#                     q)
#     _, z′, _, ℓq′ = Bijectors.forward(rng, q)

#     ϵ  = 0.01
#     ν  = 3
#     # q0 = Bijectors.TransformedDistribution(
#     #     Distributions.IsoTDist(ν, length(z′), 3.0),
#     #     q.transform)
#     # if rand(rng, Bernoulli(ϵ))
#     #     z′ = rand(rng, q0)
#     # end

#     ℓw  = z_rv.prob - logaddexp(
#         log(1 - ϵ) + logpdf(q, z_rv.val),  log(ϵ) + logpdf(q0, z_rv.val))
#     ℓp′ = ℓπ(z′)
#     ℓw′ = ℓp′ - logaddexp(
#         log(1 - ϵ) + logpdf(q, z′),  log(ϵ) + logpdf(q0, z′))
#     α   = min(1.0, exp(ℓw′ - ℓw))
#     if(rand(rng) < α)
#         RV(z′, ℓp′), α, true
#     else
#         z_rv, α, false
#     end
# end
