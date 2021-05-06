

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

function prepare_valid(prng, mat_valid)
    word_arrs_valid = []
    for i = 1:size(mat_valid,1)
        word_idx, word_count = SparseArrays.findnz(mat_valid[i,:])
        word_arr             = vcat(fill.(word_idx, word_count)...)
        Random.shuffle!(prng, word_arr)
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
    docs_valid = prepare_valid(prng, mat_valid)

    K     = 3
    N     = length(w_train)
    M     = size(mat_train,1)
    V     = size(mat_train,2)
    α     = 0.1
    β     = 1.0
    model = lda(K, M, V, N, d_train, w_train, c_train, α, β)

    #AdvancedVI.setadbackend(:forwarddiff)
    #Turing.setadbackend(:forwarddiff)
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))
    #AdvancedVI.setadbackend(:zygote)
    #Turing.Core._setadbackend(Val(:zygote))

    klpq_hist = []
    pll_hist  = []
    function plot_callback(logπ, q, objective, klpq)
        λ   = get_ϕ_latent(q)
        Eℓβ = mapslices(dirichlet_expectation, λ, dims=1)
        pll = predictive(prng, α, K, λ, Eℓβ, 1000, docs_valid)

        best_idx   = [sortperm(λ[:,i])[end-10:end] for i = 1:K]
        best_words = [words[best_idx_cat] for best_idx_cat in best_idx]

        push!(pll_hist, pll)
        push!(klpq_hist, klpq)
        display(plot(pll_hist))

        (pll=pll,
         klpq=klpq,
         best_1=best_words[1],
         best_2=best_words[2],
         best_3=best_words[3],
         )
    end

    θ0          = StatsFuns.softplus.(randn(prng, M*K+K*V))
    q_init      = LDAMeanField(θ0, M, K, V)
    n_iter      = 10000
    n_mc        = 32
    θ, q, stats = vi(model, q_init;
                     #objective   = MSC_CIS(),
                     objective   = MSC_PIMH(),
                     #objective   = ELBO(),
                     n_mc        = n_mc,
                     n_iter      = n_iter,
                     tol         = 0.0005,
                     callback    = plot_callback,
                     rng         = prng,
                     #sleep_freq   = 5,
                     #sleep_params = (ϵ=hmc_ϵ, L=hmc_L,),
                     #optimizer   = AdvancedVI.TruncatedADAGrad(),
                     optimizer    = Flux.ADAM(1.0),
                     #optimizer    = Flux.ADAGrad(),
                     show_progress = true
                     )
end

function load_dataset(prng::Random.AbstractRNG,
                      task::Val{:lda})
    data           = MAT.matread(datadir("dataset", "reuter21578.mat"))
    data_mat       = transpose(data["matrix"])
    words          = data["words"]

    n_docs         = size(data_mat, 1)
    doc_idxs       = (1:n_docs)
    doc_idxs       = filter(i -> SparseArrays.nnz(data_mat[i,:]) > 10, doc_idxs)
    n_docs         = length(doc_idxs)
    n_words        = size(data_mat, 2)
    println(n_docs, " documents")

    Random.shuffle!(doc_idxs)

    n_train   = floor(Int, n_docs*0.9)
    mat_train = Int.(data_mat[doc_idxs[1:n_train]    ,:])
    mat_valid = Int.(data_mat[doc_idxs[n_train+1:end],:])
    mat_train, mat_valid, words

    # data           = MAT.matread(datadir("dataset", "classic400.mat"))
    # data_mat       = data["classic400"]
    # labels         = data["truelabels"][1,:]
    # words          = data["classicwordlist"]

    # n_docs         = size(data_mat, 1)
    # n_words        = size(data_mat, 2)
    # doc_idxs       = collect(1:n_docs)
    # docs_labeled   = [doc_idxs[labels .== i] for i = 1:3]
    # n_train        = floor.(Int, length.(docs_labeled)*0.9)
    # doc_idx_train  = [StatsBase.sample(docs_labeled[i], n_train[i], replace=false) for i = 1:3]
    # doc_idx_test   = [doc_idxs[doc_idxs .∉ Ref(doc_idx_train[i])] for i = 1:3]

    # doc_idx_train  = vcat(doc_idx_train...)
    # doc_idx_test   = vcat(doc_idx_test...)
    # doc_label_test = labels[doc_idx_test]

    # mat_train = Int.(data_mat[doc_idx_train,:])
    # mat_valid = Int.(data_mat[doc_idx_test,:])
    # mat_train, mat_valid, words
end

