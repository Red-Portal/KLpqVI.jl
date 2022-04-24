
function pm_next!(pm, stat::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stat)])
end

function sgd_step!(optimizer, θ::AbstractVector, ∇_buf)
    ∇ = DiffResults.gradient(∇_buf)
    Δ = AdvancedVI.apply!(optimizer, θ, ∇)
    @. θ = θ - Δ
    (grad_norm=norm(∇),)
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

function vi(model::DynamicPPL.Model, 
            q_init=nothing;
            n_mc,
            n_iter,
            objective=AdvancedVI.ELBO(),
            optimizer=Flux.ADAM(),
            rng=Random.GLOBAL_RNG,
            defensive_weight=nothing,
            defensive_dist=nothing,
            callback=nothing,
            show_progress::Bool=false)
    varinfo     = DynamicPPL.VarInfo(model)
    varsyms     = keys(varinfo.metadata)
    n_params    = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ varsyms])
    logπ, ∇logπ = make_logjoint(rng, model)

    θ, q = if isnothing(q_init)
        μ = randn(rng, n_params)
        σ = StatsFuns.softplus.(randn(rng, n_params))
        θ = vcat(μ, σ)
        q = Turing.Variational.meanfield(model)
        θ, q
    else
        θ = StatsBase.params(q_init)
        vcat(θ...), q_init
    end

    b   = Bijectors.bijector(q)
    b⁻¹ = inv(b)
    ℓjac(z_)          = Bijectors.logabsdetjac(b⁻¹, b(z_))
    ℓq(λ_, z_)        = logpdf(AdvancedVI.update(q, λ_), z_)
    rand_q(prng_, λ_) = rand(prng_, AdvancedVI.update(q, λ_))

    ℓq_def, rand_q_def = if isnothing(defensive_dist) || isnothing(defensive_weight)
        ℓq, rand_q
    else
        make_defensive(q, defensive_dist, defensive_weight)
    end

    vi(logπ, ℓq, rand_q, θ;
       ℓq_def        = ℓq_def,
       rand_q_def    = rand_q_def,
       objective     = objective,
       n_mc          = n_mc,
       n_iter        = n_iter,
       optimizer     = optimizer,
       rng           = rng,
       ℓjac          = ℓjac,
       show_progress = show_progress,
       callback      = callback)
end

function vi(ℓπ,
            ℓq::Function,
            rand_q::Function,
            λ0::AbstractVector;
            ℓq_def              = ℓq,
            rand_q_def          = rand_q,
            n_mc::Int           = 10,
            n_iter::Int         = 10000,
            optimizer           = Flux.ADAM(),
            objective           = AdvancedVI.ELBO(),
            rng                 = Random.GLOBAL_RNG,
            callback            = nothing,
            show_progress::Bool = false,
            ℓjac                = z′ -> 0)
    prog   = if(show_progress)
        ProgressMeter.Progress(n_iter)
    else
        nothing
    end
    stats   = Vector{NamedTuple}(undef, n_iter)
    ∇KL_buf = DiffResults.GradientResult(λ0)

    init_state!(objective, rng, rand_q, λ0, ℓπ, n_mc)

    λ = λ0
    elapsed_total = 0
    for t = 1:n_iter
        start_time = Dates.now()
        stat       = (iteration=t,)

        stat′ = grad!(rng, objective, ℓq, ℓq_def, rand_q, rand_q_def, ℓjac, ℓπ, λ, n_mc, ∇KL_buf)
        stat  = merge(stat, stat′)

        stat′ = sgd_step!(optimizer, λ, ∇KL_buf)
        stat  = merge(stat, stat′)

        elapsed        = Dates.now() - start_time
        elapsed_total += elapsed.value
        stat           = merge(stat, (elapsed=elapsed_total,))

        if(!isnothing(callback))
            stat′ = callback(ℓπ, λ)
            #stat′ = callback(ℓπ,  t > T_polyak ? λ_avg : λ)
            stat  = merge(stat, stat′)
        end

        if(show_progress)
            pm_next!(prog, stat)
        end
        stats[t] = stat
    end
    λ, stats
end
