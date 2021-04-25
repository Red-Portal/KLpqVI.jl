
function pm_next!(pm, stat::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stat)])
end

function sgd_step!(optimizer, θ::AbstractVector, ∇_buf)
    ∇ = DiffResults.gradient(∇_buf)
    Δ = AdvancedVI.apply!(optimizer, θ, ∇)
    @. θ = θ - Δ
end

function vi(model,
            q_init=nothing;
            n_mc,
            n_iter,
            tol,
            objective=AdvancedVI.ELBO(),
            optimizer,
            rng=Random.GLOBAL_RNG,
            callback=nothing,
            sleep_freq=0,
            sleep_params=nothing,
            show_progress::Bool=false)
    varinfo  = DynamicPPL.VarInfo(model)
    n_params = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ keys(varinfo.metadata)])
    logπ     = Turing.Variational.make_logjoint(model)
    alg      = AdvancedVI.ADVI(n_mc, n_iter)
    stats    = Vector{NamedTuple}(undef, n_iter)

    if isnothing(q_init)
        μ = randn(rng, n_params)
        σ = StatsFuns.softplus.(randn(rng, n_params))
    else
        μ, σs = StatsBase.params(q_init)
        σ     = StatsFuns.invsoftplus.(σs)
    end
    θ     = vcat(μ, σ)
    q     = Turing.Variational.meanfield(model)
    ∇_buf = DiffResults.GradientResult(θ)

    init_state!(objective, rng, q, logπ, n_mc)

    ws_aug = if (sleep_freq > 0)
        z0_ws = rand(rng, q)
        WakeSleep(RV(z0_ws, logπ(z0_ws)), sleep_params)
    else
        nothing
    end

    prog = if(show_progress)
        ProgressMeter.Progress(n_iter)
    else
        nothing
    end
    start_time = Dates.now()
    
    for t = 1:n_iter
        stat  = (iteration=t,)
        stat′ = AdvancedVI.grad!(rng, objective, alg, q, logπ, θ, ∇_buf)
        stat  = merge(stat, stat′)
        sgd_step!(optimizer, θ, ∇_buf)

        if(sleep_freq > 0 && mod(t-1, sleep_freq) == 0)
            stat′ = sleep_phase!(rng, ws_aug, alg, q, logπ, θ, ∇_buf)
            stat  = merge(stat, stat′)
            sgd_step!(optimizer, θ, ∇_buf)
        end

        elapsed  = Dates.now() - start_time
        stat     = merge(stat, (elapsed=elapsed,))

        if(!isnothing(callback))
            q′    = (q isa Distribution) ?  AdvancedVI.update(q, θ) : q(θ)
            stat′ = callback(logπ, q′, objective, DiffResults.value(∇_buf))
            stat  = merge(stat, stat′)
        end

        if(show_progress)
            pm_next!(prog, stat)
        end
        stats[t] = stat
    end
    q = AdvancedVI.update(q, θ)
    return θ, q, stats
end

