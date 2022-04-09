
function imh_kernel(rng::Random.AbstractRNG,
                    z_rv::RV,
                    ℓπ::Function,
                    λ::AbstractVector,
                    rand_q::Function,
                    ℓq::Function)
    z′  = rand_q(rng, λ)
    ℓq′ = ℓq(λ, z′)
    ℓp′ = ℓπ(z′)
    ℓw′ = ℓp′ - ℓq′
    ℓw  = z_rv.prob - ℓq(λ, z_rv.val)
    ℓα   = min(0.0, ℓw′ - ℓw)
    if(log(rand(rng)) < ℓα)
        RV(z′, ℓp′), exp(ℓα), ℓw′, true
    else
        z_rv, exp(ℓα), ℓw, false
    end
end

# function imh_kernel(rng::Random.AbstractRNG,
#                     z_rv::RV,
#                     ℓπ::Function,
#                     q)
#     _, z′, _, ℓq′ = Bijectors.forward(rng, q)


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
