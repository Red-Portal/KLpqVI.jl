
@inline function dirichlet_expectation(α::AbstractVector)
    SpecialFunctions.digamma.(α) .- SpecialFunctions.digamma(sum(α))
end

function update_local(prng::Random.AbstractRNG,
                      α0::Real,
                      Eℓβ::AbstractMatrix,
                      K::Int,
                      n_iter::Int,
                      doc)
    ϵ      = eps(Float64)
    expEℓβ = exp.(Eℓβ)
    N      = length(doc)
    ϕ      = zeros(K, N)
    γ      = rand(prng, Gamma(1e+2, 1e-2), K)
    Eℓθ    = dirichlet_expectation(γ)
    expEℓθ = exp.(Eℓθ)

    for t = 1:n_iter
        @simd for n in 1:length(doc)
            @inbounds ϕᵢ      = expEℓθ .* view(expEℓβ,doc[n],:)
            @inbounds ϕ[:, n] = ϕᵢ ./ (sum(ϕᵢ) + ϵ)
        end
        γ′     = α0 .+ sum(ϕ, dims=2)[:,1]
        Δγ     = norm(γ′ - γ)
        if(Δγ < 0.01)
            break
        end
        γ      = γ′
        Eℓθ    = dirichlet_expectation(γ)
        expEℓθ = exp.(Eℓθ)
    end
    #γ / sum(γ)
    ϕ, γ
end

function update_global(λₜ₋₁, ρ, β0, M, K, V, ϕ, doc)
    λ = zeros(V,K)
    for k = 1:K
        ϕₖ = zeros(V)
        @simd for n = 1:length(doc)
            @inbounds ϕₖ[doc[n]] += ϕ[k, n]
        end
        @inbounds λ[:,k] = β0 .+ M*ϕₖ
    end
    λₜ  = (1-ρ)*λₜ₋₁ + ρ*λ
end

function predictive(prng, α, K, λ, Eℓβ, n_iters, docs)
    Eλ = λ ./ sum(λ, dims=1)
    mapreduce(+, enumerate(docs)) do (m, doc)
        n_w      = length(doc)
        n_obs    = floor(Int, n_w/2)
        w_obs    = view(doc, 1:n_obs)
        w_ho     = view(doc, n_obs+1:n_w)
        _, γₘ    = update_local(prng, α, Eℓβ, K, n_iters, w_obs)
        Eγₘ      = γₘ / sum(γₘ)
        p_w      = Eλ * Eγₘ
        mapreduce(+, w_ho) do w
            log.(p_w[w])
        end / length(w_ho)
    end / length(docs)
end

function lda_svi(prng::Random.AbstractRNG,
                 words,
                 train_docs,
                 test_docs,
                 n_epochs::Int,
                 n_local_iters::Int,
                 K::Int,
                 V::Int,
                 show_progress=true)
    M   = length(train_docs)
    α0  = 0.1
    β0  = 1.0
    τ   = 1024
    κ   = 0.7
    λ   = rand(prng, Gamma(1e+2, 1e-2), V, K)
    Eℓβ = mapslices(dirichlet_expectation, λ, dims=1)

    n_iters = ceil(Int, M/1000)*n_epochs
    prog    = if(show_progress)
        ProgressMeter.Progress(n_iters)
    else
        nothing
    end
    stats         = Vector{NamedTuple}()
    elapsed_total = 0

    t = 0
    for epoch_idx = 1:n_epochs
        doc_epoch  = Random.shuffle(prng, 1:M)
        for batch ∈ Iterators.partition(doc_epoch, 1000)
            start_time = Dates.now()
            for doc_idx ∈ batch
                doc     = train_docs[doc_idx]
                ϕₘ,  _  = update_local(prng, α0, Eℓβ, K, n_local_iters, doc)
                ρ       = (t + τ).^(-κ)
                λ       = update_global(λ, ρ, β0, M, K, V, ϕₘ, doc)
                Eℓβ     = mapslices(dirichlet_expectation, λ, dims=1)
                t      += 1
            end
            elapsed        = Dates.now() - start_time
            elapsed_total += elapsed.value
            pll            = predictive(prng, α0, K, λ, Eℓβ, n_local_iters, test_docs)

            best_idx   = [sortperm(λ[:,i])[end-10:end] for i = 1:K]
            best_words = [words[best_idx_cat] for best_idx_cat in best_idx]
            stat       = (epoch          = epoch_idx,
                          iteration      = t,
                          log_predictive = pll,
                          elapsed        = elapsed_total,
                          best_words_1   = best_words[1],
                          best_words_2   = best_words[2],
                          best_words_3   = best_words[3],
                          )
            push!(stats, stat)
            display(plot([stat.iteration for stat in stats],
                         [stat.log_predictive for stat in stats]))

            if(!isnothing(prog))
                pm_next!(prog, stat)
            end
        end
    end
end
