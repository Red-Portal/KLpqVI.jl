
mutable struct WakeSleep
    z::RV{Float64}
    hmc_params::NamedTuple{(:ϵ, :L), Tuple{Float64, Int64}}
end

function sleep_phase!(
    rng::Random.AbstractRNG,
    ws::WakeSleep,
    alg::AdvancedVI.VariationalInference,
    q,
    logπ,
    θ::AbstractVector,
    out::DiffResults.MutableDiffResult
    )
#=
    Jorg B\"ornschein and Yoshua Bengio.
    "Reweighted Wake-sleep."
    ICLR 2015
=##
    z′, acc = hmc_step(rng, alg, q, logπ, ws.z,
                       ws.hmc_params.ϵ, ws.hmc_params.L)
    ws.z = z′
    f(θ_) = if (q isa Distribution)
        -(Bijectors.logpdf(AdvancedVI.update(q, θ_), ws.z.val))
    else
        -Bijectors.logpdf(q(θ_), ws.z.val)
    end
    gradient!(alg, f, θ, out)

    (sleep=true,
     sleep_acc=acc)
end

