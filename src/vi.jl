
function vi(model,
            q_init=nothing;
            n_mc,
            n_iter,
            tol,
            objective=AdvancedVI.ELBO(),
            optimizer,
            rng=Random.GLOBAL_RNG,
            callback=nothing,
            show_progress::Bool=false)
    varinfo  = DynamicPPL.VarInfo(model)
    n_params = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ keys(varinfo.metadata)])
    logπ     = Turing.Variational.make_logjoint(model)
    alg      = AdvancedVI.ADVI(n_mc, n_iter)

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

    init_state!(objective, rng, q, n_mc)

    prog = if(show_progress)
        ProgressMeter.Progress(n_iter)
    else
        nothing
    end
    
    for step = 1:n_iter
        AdvancedVI.grad!(rng, objective, alg, q, logπ, θ, ∇_buf)
        ∇ = DiffResults.gradient(∇_buf)
        Δ = AdvancedVI.apply!(optimizer, θ, ∇)
        @. θ = θ - Δ

        if(!isnothing(callback))
            q′ = (q isa Distribution) ?  AdvancedVI.update(q, θ) : q(θ)
            callback(logπ, q′, objective, DiffResults.value(∇_buf))
        end
        
        if(show_progress)
            ProgressMeter.next!(prog)
        end
    end
    q = AdvancedVI.update(q, θ)
    return θ, q
end

