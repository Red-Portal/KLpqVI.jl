
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
        for (i, w) in enumerate(doc)
            ϕᵢ      = expEℓθ .* view(expEℓβ,:,w)
            ϕ[:, i] = ϕᵢ ./ (sum(ϕᵢ) + ϵ)
        end
        γ      = α0 .+ sum(ϕ, dims=2)[:,1]
        Eℓθ    = dirichlet_expectation(γ)
        expEℓθ = exp.(Eℓθ)
    end
    #γ / sum(γ)
    ϕ, γ
end

function update_global(λₜ₋₁, ρ, β0, M, K, V, ϕ, doc)
    λ = zeros(K,V)
    for k = 1:K
        ϕₖ = zeros(V)
        for (m, w) ∈ enumerate(doc)
            ϕₖ[w] += ϕ[k, m]
        end
        λ[k,:] = β0 .+ M*ϕₖ
    end
    λₜ  = (1-ρ)*λₜ₋₁ + ρ*λ
end

function perplexity(prng, α, K, λ, Eℓβ, docs)
    γ  = zeros(K, length(docs))
    ll = mapreduce(+, enumerate(docs)) do (m, doc)
        _, γₘ    = update_local(prng, α, Eℓβ, K, 100, doc)
        mixture  = γₘ'*λ
        mixture /= sum(mixture)
        γ[:, m]  = γₘ
        mapreduce(+, doc) do w
            log.(mixture[w])
        end
    end
    pll  = ll / sum(length.(docs))
    pplx = exp(-pll)
    pll, pplx
end

function lda_svi(prng::Random.AbstractRNG,
                 words,
                 train_docs,
                 test_docs,
                 n_iters::Int,
                 K::Int,
                 V::Int,
                 show_progress=true)
    M   = length(train_docs)
    ρ0  = 0.2
    α0  = 0.1
    β0  = 1.0
    τ   = 1024
    κ   = 0.7
    λ   = rand(prng, Gamma(1e+2, 1e-2), K, V)
    Eℓβ = dirichlet_expectation(λ)

    prog = if(show_progress)
        ProgressMeter.Progress(n_iters)
    else
        nothing
    end
    stats      = Vector{NamedTuple}(undef, n_iters)
    start_time = Dates.now()

    for t = 1:n_iters
        doc_epoch = Random.shuffle(prng, 1:M)
        for doc_idx ∈ doc_epoch
            doc     = train_docs[doc_idx]
            ϕₘ,  _  = update_local(prng, α0, Eℓβ, K, 1000, doc)
            ρ       = ρ0*(t + τ).^(-κ)
            λ       = update_global(λ, ρ, β0, M, K, V, ϕₘ, doc)
            Eℓβ     = dirichlet_expectation(λ)
        end

        pll, pplx  = perplexity(prng, α0, K, λ, Eℓβ, test_docs)
        elapsed    = Dates.now() - start_time
        best_idx   = [sortperm(λ[i,:])[end-10:end] for i = 1:K]
        best_words = [words[best_idx_cat] for best_idx_cat in best_idx]

        stat     = (log_predictive = pll,
                    perplexity     = pplx,
                    elapsed        = elapsed,
                    best_words_1   = best_words[1],
                    best_words_2   = best_words[2],
                    best_words_3   = best_words[3],
                    )
        stats[t] = stat
        display(plot([stat.perplexity for stat in stats[1:t]]))

        if(!isnothing(prog))
            pm_next!(prog, stat)
        end
    end
end
