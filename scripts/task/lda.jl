
Turing.@model lda(K, M, V, N, docs, words, counts, α, β, ::Type{T}=Float64) where {T} = begin
    θ = Vector{Vector{T}}(undef, M)
    @simd for m = 1:M
        @inbounds θ[m] ~ Dirichlet(K, α)
    end

    ϕ = Vector{Vector{T}}(undef, K)
    @simd for k = 1:K
        @inbounds ϕ[k] ~ Dirichlet(V, β)
    end
    ℓϕmθ = log.(hcat(ϕ...)*hcat(θ...))
    Turing.@addlogprob! mapreduce(+, 1:N) do i
        counts[i]*ℓϕmθ[words[i], docs[i]]
    end
end

function dirichlet_expectation(α)
    return SpecialFunctions.digamma.(α) .- SpecialFunctions.digamma(sum(α))
end


function prepare_valid(mat_valid)
    word_arrs_valid = []
    for i = 1:size(mat_valid,1)
        word_idx, word_count = SparseArrays.findnz(mat_valid[i,:])
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
    word_arrs_train = prepare_valid(mat_train)

    K     = 3
    N     = length(w_train)
    M     = size(mat_train,1)
    V     = size(mat_train,2)
    α     = 0.1
    β     = 1.0
    #model = lda(K, M, V, N, d_train, w_train, c_train, α, β)

    lda_svi(prng, words, word_arrs_train, word_arrs_valid, 100, K, V)
    throw()

    #AdvancedVI.setadbackend(:forwarddiff)
    #Turing.setadbackend(:forwarddiff)
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))
    #AdvancedVI.setadbackend(:zygote)
    #Turing.Core._setadbackend(Val(:zygote))

    pll_hist = []
    function plot_callback(logπ, q, objective, klpq)
        z_β  = sample_variable(prng, q, model, Symbol("ϕ"), 128)
        z_β  = reshape(z_β, (K,V,:))
        #z_β  = permutedims(z_β, (2,1,3))
        μ_β  = mean(z_β, dims=3)[:,:,1]
        μ_ℓβ = mean(log.(z_β), dims=3)[:,:,1]

        best_idx   = [sortperm(μ_β[i,:])[end-10:end] for i = 1:K]
        best_words = [words[best_idx_cat] for best_idx_cat in best_idx]

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
    
    n_iter      = 500
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
    data           = MAT.matread(datadir("dataset", "classic400.mat"))
    data_mat       = data["classic400"]
    labels         = data["truelabels"][1,:]
    words          = data["classicwordlist"]

    n_docs         = size(data_mat, 1)
    n_words        = size(data_mat, 2)
    doc_idxs       = collect(1:n_docs)
    docs_labeled   = [doc_idxs[labels .== i] for i = 1:3]
    n_train        = floor.(Int, length.(docs_labeled)*0.9)
    doc_idx_train  = [StatsBase.sample(docs_labeled[i], n_train[i], replace=false) for i = 1:3]
    doc_idx_test   = [doc_idxs[doc_idxs .∉ Ref(doc_idx_train[i])] for i = 1:3]

    doc_idx_train  = vcat(doc_idx_train...)
    doc_idx_test   = vcat(doc_idx_test...)
    doc_label_test = labels[doc_idx_test]

    mat_train = Int.(data_mat[doc_idx_train,:])
    mat_valid = Int.(data_mat[doc_idx_test,:])
    mat_train, mat_valid, words
end


