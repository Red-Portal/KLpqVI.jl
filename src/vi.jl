
function pm_next!(pm, stat::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stat)])
end

function sgd_step!(optimizer, θ::AbstractVector, ∇_buf)
    ∇ = DiffResults.gradient(∇_buf)
    Δ = AdvancedVI.apply!(optimizer, θ, ∇)
    @. θ = θ - Δ
end

function make_logjoint(rng, model::DynamicPPL.Model, weight::Real=1.0)
    # setup
    ctx = DynamicPPL.MiniBatchContext(
        DynamicPPL.DefaultContext(),
        weight
    )

    #sampler_init = DynamicPPL.initialsampler(sampler_hmc)
    sampler_hmc  = DynamicPPL.Sampler(Turing.NUTS{Turing.Core.ADBackend()}(), model)
    vi_init      = DynamicPPL.VarInfo(model, ctx)
    model(rng, vi_init, DynamicPPL.SampleFromUniform())

    function logπ(z)
        varinfo = DynamicPPL.VarInfo(vi_init, DynamicPPL.SampleFromUniform(), z)
        model(varinfo)
        return DynamicPPL.getlogp(varinfo)
    end
    ∇logπ = Turing.Inference.gen_∂logπ∂θ(vi_init, sampler_hmc, model)
    logπ, ∇logπ
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
            sleep_interval=0,
            sleep_params=nothing,
            rhat_interval=0,
            paretok_samples=0,
            show_progress::Bool=false)
    varinfo     = DynamicPPL.VarInfo(model)
    varsyms     = keys(varinfo.metadata)
    n_params    = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ varsyms])
    logπ, ∇logπ = make_logjoint(rng, model)
    alg         = AdvancedVI.ADVI(n_mc, n_iter)
    stats       = Vector{NamedTuple}(undef, n_iter)

    if isnothing(q_init)
        μ = randn(rng, n_params)
        σ = StatsFuns.softplus.(randn(rng, n_params))
    else
        μ, σs = StatsBase.params(q_init)
        σ     = StatsFuns.invsoftplus.(σs)
    end
    θ        = vcat(μ, σ)
    q        = Turing.Variational.meanfield(model)
    ∇_buf    = DiffResults.GradientResult(θ)

    pimh_rhat_win = if(objective isa MSC_PIMH)
        Array{Float64}(undef, rhat_interval, n_params, n_mc)
    else
        nothing
    end

    rhat_win = if(rhat_interval > 0)
        Array{Float64}(undef, rhat_interval, n_params*2)
    else
        nothing
    end

    ws_aug = if (sleep_interval > 0)
        z0_ws = rand(rng, q)
        WakeSleep(RV(z0_ws, logπ(z0_ws)), sleep_params)
    else
        nothing
    end

    init_state!(objective, rng, q, logπ, n_mc)

    prog = if(show_progress)
        ProgressMeter.Progress(n_iter)
    else
        nothing
    end
    
    elapsed_total = 0
    for t = 1:n_iter
        start_time = Dates.now()

        stat  = (iteration=t,)
        stat′ = grad!(rng, objective, alg, q, logπ, ∇logπ, θ, ∇_buf)
        stat  = merge(stat, stat′)
        sgd_step!(optimizer, θ, ∇_buf)

        if(sleep_interval > 0 && mod(t-1, sleep_interval) == 0)
            stat′ = sleep_phase!(rng, ws_aug, alg, q, logπ, ∇logπ, θ, ∇_buf)
            stat  = merge(stat, stat′)
            sgd_step!(optimizer, θ, ∇_buf)
        end
        q′ = (q isa Distribution) ?  AdvancedVI.update(q, θ) : q(θ)

        elapsed        = Dates.now() - start_time
        elapsed_total += elapsed.value
        stat           = merge(stat, (elapsed=elapsed_total,))

        if(!isnothing(callback))
            stat′ = callback(logπ, q′, objective, DiffResults.value(∇_buf))
            stat  = merge(stat, stat′)
        end

        if(rhat_interval > 0)
            rhat_idx = mod(t-1, rhat_interval) + 1
            rhat_win[rhat_idx,:] = θ

            if(objective isa MSC_PIMH)
                zs = hcat([z.val for z ∈ objective.zs]...)
                pimh_rhat_win[rhat_idx,:,:] = zs
            end

            if(rhat_idx == rhat_interval)
                R̂    = split_rhat(rhat_win)
                stat = merge(stat, (rhat=maximum(R̂),))

                if(objective isa MSC_PIMH)
                    pimh_R̂ = split_rhat(pimh_rhat_win)
                    stat   = merge(stat, (pimh_rhat=maximum(pimh_R̂),))
                end
            end
        end

        if(paretok_samples > 0)
            k    = paretok(rng, q′, logπ, paretok_samples)
            stat = merge(stat, (paretok=k,))
        end

        if(show_progress)
            pm_next!(prog, stat)
        end
        stats[t] = stat
    end
    q = AdvancedVI.update(q, θ)
    return θ, q, stats
end

