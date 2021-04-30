
Turing.@model lda(K, M, N, docs, words, counts, α, β) = begin
    θ = Vector{Vector{Real}}(undef, M)
    @simd for m = 1:M
        @inbounds θ[m] ~ Dirichlet(α)
    end

    ϕ = Vector{Vector{Real}}(undef, K)
    @simd for k = 1:K
        @inbounds ϕ[k] ~ Dirichlet(β)
    end

    ℓϕmθ = log.(hcat(ϕ...) * hcat(θ...))

    Turing.@addlogprob! mapreduce(+, 1:N) do i
        counts[i]*ℓϕmθ[words[i],docs[i]]
    end
end

function dirichlet_expectation(α)
    return SpecialFunctions.digamma.(α) .- SpecialFunctions.digamma(sum(α))
end

function update_local(prng::Random.AbstractRNG,
                      α::AbstractVector,
                      Eℓβ::AbstractMatrix,
                      K::Int,
                      n_iter::Int,
                      words)
    ϵ      = 1e-10
    expEℓβ = exp.(Eℓβ)

    N      = length(words)
    ϕ      = zeros(K, N)
    γ      = rand(prng, Gamma(1e+2, 1e-2), K)
    Eℓθ    = dirichlet_expectation(γ)
    expEℓθ = exp.(Eℓθ)

    for t = 1:n_iter
        for (i, w) in enumerate(words)
            ϕᵢ      = expEℓθ .* view(expEℓβ,:,w)
            ϕ[:, i] = ϕᵢ ./ (sum(ϕᵢ) + ϵ)
        end
        γ      = α .+ sum(ϕ, dims=2)[:,1]
        Eℓθ    = dirichlet_expectation(γ)
        expEℓθ = exp.(Eℓθ)
    end
    γ / sum(γ)
end

function prepare_valid(mat_valid)
    word_arrs_valid = []
    for i = 1:size(mat_valid,1)
        word_idx, word_count = SparseArrays.findnz(mat_valid[i,:])
        word_count           = word_count
        word_arr             = vcat(fill.(word_idx, word_count)...)
        push!(word_arrs_valid, word_arr)
    end
    word_arrs_valid
end

function run_task(task::Val{:lda})
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    Random123.set_counter!(prng, 0)
    Random.seed!(0)

    mat_train, mat_valid, words = load_dataset(prng, task)

    # Prepare train dataset
    (d_train, w_train, c_train) = SparseArrays.findnz(mat_train)
    
    # Prepare valid dataset
    word_arrs_valid = prepare_valid(mat_valid)

    K     = 3
    N     = length(w_train)
    M     = size(mat_train,1)
    V     = size(mat_train,2)
    α     = fill(0.1, K)
    β     = fill(1.0, V)
    model = lda(K, M, N, d_train, w_train, c_train, α, β)

    #AdvancedVI.setadbackend(:forwarddiff)
    #Turing.setadbackend(:forwarddiff)
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))
    #AdvancedVI.setadbackend(:zygote)
    #Turing.Core._setadbackend(Val(:zygote))

    pll_hist = []
    function plot_callback(logπ, q, objective, klpq)
        z    = rand(prng, q, 256)
        z_β  = z[end-K*V+1:end,:]
        #z_β  = z[1:K*V,:]

        μ_β  = mean(z_β, dims=2)[:,1]
        μ_β  = Array(reshape(μ_β, (V,K))')

        best_idx   = [sortperm(μ_β[i,:])[end-5:end] for i = 1:K]
        best_words = [words[best_idx_cat] for best_idx_cat in best_idx]

        μ_ℓβ = mean(log.(z_β), dims=2)[:,1]
        μ_ℓβ = Array(reshape(μ_ℓβ, (V,K))')

        ll = mapreduce(+, word_arrs_valid) do word_arr
            θₘ  = update_local(prng, α, μ_ℓβ, K, 100, word_arr)
            sum(log.(θₘ'*view(μ_β, :, word_arr)))
        end
        pll = ll / sum(length.(word_arrs_valid))
        ppx = exp(-pll)
        push!(pll_hist, ppx)

        display(plot(pll_hist))
        (pll=ppx,
         best_1=best_words[1],
         best_2=best_words[2],
         best_3=best_words[3],
         )
    end
    
    n_iter      = 10
    n_mc        = 8
    θ, q, stats = vi(model;
                     #objective   = MSC_CIS(),
                     #objective   = MSC_PIMH(),
                     objective   = ELBO(),
                     n_mc        = n_mc,
                     n_iter      = n_iter,
                     tol         = 0.0005,
                     callback    = plot_callback,
                     rng         = prng,
                     #sleep_freq   = 5,
                     #sleep_params = (ϵ=hmc_ϵ, L=hmc_L,),
                     #optimizer   = AdvancedVI.TruncatedADAGrad(),
                     optimizer    = Flux.ADAM(0.01),
                     #optimizer    = Flux.ADAGrad(),
                     show_progress = true
                     )
end

function load_dataset(prng::Random.AbstractRNG,
                      task::Val{:lda})
    data      = MAT.matread(datadir("dataset", "classic400.mat"))
    data_mat  = data["classic400"]
    n_docs    = size(data_mat, 1)
    n_words   = size(data_mat, 2)
    n_train   = floor(Int, n_docs*0.9)
    words     = data["classicwordlist"]

    idx_shfl  = Random.shuffle(prng, 1:n_docs)
    data_mat  = data_mat[idx_shfl,:]
    mat_train = Int.(data_mat[1:n_train,:])
    mat_valid = Int.(data_mat[n_train+1:end,:])
    mat_train, mat_valid, words
end


