
using DrWatson
@quickactivate "KLpqVI"

using Base.GC
using BSON
using CUDA
using DelimitedFiles
using Distributions
using DrWatson: struct2dict
using Flux
using Flux: @functor, chunk
using Flux.Losses: logitbinarycrossentropy, binarycrossentropy
using Flux.Data: DataLoader
using Images
using Logging: with_logger
using MLDataUtils: splitobs
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Random
using Statistics
using StatsFuns
using Plots
using Zygote
using LinearAlgebra
using JLD2
using FileIO

using KernelAbstractions
using CUDAKernels

# load MNIST images and return loader
function get_data(batch_size)
    xtrain = Array{Float32}(readdlm(datadir("dataset", "binarized_mnist_train.amat"))')
    xvalid = Array{Float32}(readdlm(datadir("dataset", "binarized_mnist_valid.amat"))')
    xtest  = Array{Float32}(readdlm(datadir("dataset", "binarized_mnist_test.amat"))')

    ytrain = collect(1:size(xtrain, 2))
    yvalid = collect(1:size(xvalid, 2))
    ytest  = collect(1:size(xtest,  2))

    @info "Dataset Organization" train=size(xtrain) valid=size(xvalid) test=size(xtest)

    DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true),
    DataLoader((xvalid, yvalid), batchsize=1,          shuffle=false),
    DataLoader((xtest,  ytest),  batchsize=1,          shuffle=false)
end

struct Encoder
    layer1::Dense
    layer2::Dense
end
@functor Encoder
Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Encoder(
    Dense(input_dim,  hidden_dim, initb=x -> fill(-1f0, x)),
    Dense(hidden_dim, latent_dim, initb=x -> fill(-1f0, x)),
)

struct Decoder
    layer1::Dense
    layer2::Dense
end
@functor Decoder
Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Decoder(
    Dense(latent_dim, hidden_dim, initb=x -> fill(-1f0, x)),
    Dense(hidden_dim, input_dim,  initb=x -> fill(-1f0, x)),
)

function sample_layer(layer::Dense, x, device)
    logit = layer(x)
    u     = rand(Float32, size(logit)) |> device
    p     = sigmoid.(logit) 
    z     = Float32.(p .> u)
    z, logit
end

function sample_z(encoder::Encoder, x, device)
    z1, logit1 = sample_layer(encoder.layer1, x,  device)
    z2, logit2 = sample_layer(encoder.layer2, z1, device)
    vcat(z1, z2), vcat(logit1, logit2)
end

function reconstuct(encoder::Encoder, decoder::Decoder, x, device)
    _, logit1  = sample_layer(encoder.layer1, x,  device)
    _, logit2  = sample_layer(encoder.layer2, sigmoid.(logit1), device)
    _, logit1  = sample_layer(decoder.layer1, sigmoid.(logit2), device)
    _, x_logit = sample_layer(decoder.layer2, sigmoid.(logit1), device)
    sigmoid.(x_logit) |> cpu
end

function joint_density(decoder::Decoder, z, x, device, settings)
    n_latent = settings[:latent_dim]
    n_hidden = settings[:hidden_dim]
    z1       = view(z, 1:n_hidden, :)
    z2       = view(z, n_hidden+1:n_hidden+n_latent, :)

    p0      = fill(0f0, size(z2)) |> device
    logp_z2 = -logitbinarycrossentropy(p0,  z2, agg=xᵢ->sum(xᵢ, dims=1))[1,:]

    p1      = decoder.layer1(z2)
    x_recon = decoder.layer2(z1)
    logp_z1 = -logitbinarycrossentropy(p1, z1,     agg=xᵢ->sum(xᵢ, dims=1))[1,:]
    logp_x  = -logitbinarycrossentropy(x_recon, x, agg=xᵢ->sum(xᵢ, dims=1))[1,:]

    logp_z2 + logp_z1 + logp_x
end

function variational_density(z, q, settings::Dict)
    n_latent = settings[:latent_dim]
    n_hidden = settings[:hidden_dim]
    z1       = view(z, 1:n_hidden, :)
    z2       = view(z, n_hidden+1:n_hidden+n_latent, :)
    q1       = view(q, 1:n_hidden, :)
    q2       = view(q, n_hidden+1:n_hidden+n_latent, :)

    logq1 = -logitbinarycrossentropy(q1, z1, agg=xᵢ->sum(xᵢ, dims=1))[1,:]
    logq2 = -logitbinarycrossentropy(q2, z2, agg=xᵢ->sum(xᵢ, dims=1))[1,:]
    logq1 + logq2
end

function variational_density_defensive(z, q, settings::Dict)
    n_latent = settings[:latent_dim]
    n_hidden = settings[:hidden_dim]
    z1       = view(z, 1:n_hidden, :)
    z2       = view(z, n_hidden+1:n_hidden+n_latent, :)
    q1       = view(q, 1:n_hidden, :)
    q2       = view(q, n_hidden+1:n_hidden+n_latent, :)

    p      = 0.1f0
    q1_def = clamp.(sigmoid.(q1), p, 1-p)
    q2_def = clamp.(sigmoid.(q2), p, 1-p)
    logq1  = -binarycrossentropy(q1_def, z1, agg=xᵢ->sum(xᵢ, dims=1))[1,:]
    logq2  = -binarycrossentropy(q2_def, z2, agg=xᵢ->sum(xᵢ, dims=1))[1,:]
    logq1 + logq2
end

function variational_density(encoder::Encoder, z, x, settings::Dict)
    n_latent = settings[:latent_dim]
    n_hidden = settings[:hidden_dim]
    z1       = view(z, 1:n_hidden, :)
    z2       = view(z, n_hidden+1:n_hidden+n_latent, :)

    q1    = encoder.layer1(x)
    q2    = encoder.layer2(z1)
    logq1 = -logitbinarycrossentropy(q1, z1, agg=xᵢ->sum(xᵢ, dims=1))[1,:]
    logq2 = -logitbinarycrossentropy(q2, z2, agg=xᵢ->sum(xᵢ, dims=1))[1,:]
    logq1 + logq2
end

function compute_log_weight(encoder, decoder, x, device, settings)
    z, q     = sample_z(encoder, x, device)
    logp_x_z = joint_density(decoder, z, x, device, settings) |> cpu
    logq_z   = variational_density(z, q, settings)            |> cpu
    logp_x_z - logq_z
end

function compute_log_weight(decoder, x, z, q, device, settings)
    logp_x_z = joint_density(decoder, z, x, device, settings)  |> cpu
    logq_z   = variational_density(z, q, settings)             |> cpu
    logp_x_z - logq_z
end

function rws_wake_loss(encoder, decoder, x, w, z, n_batch, device, settings)
    logp_x_z = joint_density(decoder, z, x, device, settings)
    logq_z   = variational_density(encoder, z, x, settings)
    -dot(logp_x_z + logq_z, w) / n_batch
end

function rws_sleep_loss(encoder, x, z, settings)
    logq_z   = variational_density(encoder, z, x, settings)
    -mean(logq_z)
end

function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

function init_state!(::Val{:RWS}, encoder, decoder, train_loader, device, settings)
    state             = Dict{Symbol, Any}()
    state[:opt_sleep] = ADAM(settings[:η])
    state[:opt_wake]  = ADAM(settings[:η])
    state
end

function init_state!(::Val{:ELBO}, encoder, decoder, train_loader, device, settings)
    state       = Dict{Symbol, Any}()
    state[:opt] = ADAM(settings[:η])
    state
end

function init_state!(::Val{:JSA}, prng, encoder, decoder, train_loader, device, settings)
    state           = Dict{Symbol, Any}()
    state[:opt]     = ADAM(settings[:η])
    state[:samples] = [rand(Bernoulli(0.5f0),
                            settings[:latent_dim] + settings[:hidden_dim])
                       for i = 1:train_loader.nobs]
    state
end

function init_state!(::Val{:PIMH}, encoder, decoder, train_loader, device, settings)
    state           = Dict{Symbol, Any}()
    state[:opt]     = ADAM(settings[:η])
    state[:samples] = [rand(Bernoulli(0.5f0),
                            settings[:latent_dim] + settings[:hidden_dim],
                            settings[:n_samples])
                       for i = 1:train_loader.nobs]
    state
end

function init_state!(::Val{:PIMH_SIR}, encoder, decoder, train_loader, device, settings)
    state            = Dict{Symbol, Any}()
    state[:opt]      = ADAM(settings[:η])
    state[:samples]  = [rand(Bernoulli(0.5f0),
                             settings[:latent_dim] + settings[:hidden_dim],
                             settings[:n_samples])
                        for i = 1:train_loader.nobs]
    state[:logjoint] = [fill(Inf, settings[:n_samples]) for i = 1:train_loader.nobs]
    state
end

function jsa_loss(encoder, decoder, x, z, device, settings)
    logp_x_z = joint_density(decoder, z, x, device, settings)
    logq_z   = variational_density(encoder, z, x, settings)
    -mean(logp_x_z + logq_z)
end

@kernel function repeat_inner_kernel!(a::AbstractArray{<:Any, N}, inner::NTuple{N}, out) where {N}
    inds = @index(Global, NTuple)
    inds_a = ntuple(i -> (inds[i] - 1) ÷ inner[i] + 1, N)

    @inbounds out[inds...] = a[inds_a...]
end

function repeat_inner(a::TV, inner) where {TV<:AbstractArray}
    out = TV(undef, inner .* size(a))

    kernel! = if out isa CuArray
        repeat_inner_kernel!(CUDADevice(), 64)
    else
        repeat_inner_kernel!(CPU(), Threads.nthreads())
    end

    ev = kernel!(a, inner, out, ndrange=size(out))
    wait(ev)
    return out
end

function adaptive_restart(::Val{:JSA}, epoch, idx, decoder, x_dev, logit, z_prop, logw_prop, device, state, settings)
    γ          = settings[:gamma]
    z_prev     = hcat(state[:samples][idx]...)
    z_prev_dev = z_prev |> device
    logw_prev  = compute_log_weight(decoder, x_dev, z_prev_dev, logit, device, settings)

    if settings[:adaptive]
        R = mean(min.(0, logw_prop - logw_prev))
        p = tanh(-γ*R)
        if (rand(Bernoulli(p)))
            z_prop, logw_prop
        else
            z_prev, logw_prev
        end
    elseif epoch > 600
        z_prev, logw_prev
    else
        z_prop, logw_prop
    end
end

function cache_samples!(::Val{:JSA}, encoder::Encoder, decoder::Decoder, x_idx, x_dev, z, device, state, settings)
    state[:samples][x_idx] = [z[:,i] for i = 1:size(z, 2)]
end

function step!(type::Union{Val{:JSA}, Val{:JSA_MC}},
               idx, epoch, encoder, decoder, params, x, x_idx, device, state, settings)
    n_batch   = size(x, 2)
    n_samples = settings[:n_samples]
    n_total   = (n_samples + 1)*n_batch

    x_dev        = x |> device
    x_rep_dev    = repeat_inner(x_dev, (1, n_samples + 1)) 
    z_dev, q_dev = sample_z(encoder, x_rep_dev, device)
    logw         = compute_log_weight(decoder, x_rep_dev, z_dev, q_dev, device, settings)
    z            = z_dev |> cpu

    if epoch > 1
        z_prop    = view(z,     :, 1:n_samples+1:n_total)
        logw_prop = view(logw,     1:n_samples+1:n_total)
        q_prop    = view(q_dev, :, 1:n_samples+1:n_total)
        z_init, logw_init = adaptive_restart(type, epoch, x_idx, decoder, x_dev, q_prop,
                                             z_prop, logw_prop, device, state, settings)
        z[:, 1:n_samples+1:n_total] = z_init
        logw[1:n_samples+1:n_total] = logw_init
    end

    u       = rand(Float32, n_total)
    logu    = log.(u)
    acc_sum = 0
    for offset = 1:n_samples
        batch_idx = 1:n_samples+1:n_total
        state_idx = batch_idx .+ offset
        prev_idx  = state_idx .- 1
        logα      = logw[state_idx] - logw[prev_idx]
        acc_flag  = logu[state_idx] .< logα
        acc_sum  += sum(acc_flag)
        rej_flag  = .!acc_flag

        z[:, state_idx[rej_flag]] = z[:, prev_idx[rej_flag]]
        logw[state_idx[rej_flag]] = logw[prev_idx[rej_flag]]
    end
    acc_avg   = acc_sum / (n_samples*n_batch)
    x_rep_dev = repeat_inner(x_dev, (1, n_samples)) 
    z         = z[    :, setdiff(1:n_total, 1:n_samples+1:n_total)]
    loss, back = Flux.pullback(params) do
        jsa_loss(encoder, decoder, x_rep_dev, z |> device, device, settings)
    end
    z_last    = view(z,    :, n_samples:n_samples:n_samples*n_batch)
    cache_samples!(type, encoder, decoder, x_idx, x_dev, z_last, device, state, settings)

    grad = back(1f0)
    Flux.Optimise.update!(state[:opt], params, grad)
    loss, acc_avg
end

function adaptive_restart(::Val{:PIMH}, epoch, idx, decoder, x_dev, logit, z_prop, logw_prop, device, state, settings)
    γ         = settings[:gamma]
    n_samples = settings[:n_samples]
    x_rep_dev = repeat_inner(x_dev, (1, n_samples))
    
    z_prev     = hcat(state[:samples][idx]...)
    z_prev_dev = z_prev |> device
    logw_prev  = compute_log_weight(decoder, x_rep_dev, z_prev_dev, logit, device, settings)

    if settings[:adaptive]
        logα = min.(0, logw_prop - logw_prev)
        logα = reshape(logα, (n_samples, :))
        R    = mean(logα, dims=1)[1,:]
        p    = tanh.(-γ*R)
        acc  = rand.(Bernoulli.(p))
        acc  = repeat(acc, inner=n_samples)
        rej  = .!acc
        z_prop[:,rej]  = z_prev[:,rej]
        logw_prop[rej] = logw_prev[rej]
        z_prop, logw_prop
    elseif epoch > 600
        z_prev, logw_prev
    else
        z_prop, logw_prop
    end
end

function systematic_sampling(weights::AbstractVector,
                             n_resample=length(weights))
    N  = length(weights)
    Δs = 1/n_resample
    u  = rand(Uniform(0.0, Δs))
    s  = 1

    resample_idx = zeros(Int64, n_resample)
    stratas      = cumsum(weights)
    @inbounds for i = 1:n_resample
        while(u > stratas[s] && s < N)
            s += 1
        end
        resample_idx[i] = s
        u += Δs
    end
    resample_idx
end

# function adaptive_restart(::Val{:PIMH_SIR}, idx, decoder, x_dev, logit, z_prop, logw_prop, device, state, settings)
#     n_samples = settings[:n_samples]
#     x_rep_dev = repeat_inner(x_dev, (1, n_samples))

#     z_prev        = hcat(state[:samples][idx]...)
#     logjoint_prev = vcat(state[:logjoint][idx]...)

#     z_prev_dev    = z_prev |> device
#     logjoint_curr = joint_density(decoder, z_prev_dev, x_rep_dev, device, settings) |> cpu
#     logw          = logjoint_curr - logjoint_prev
#     logw          = reshape(logw, (n_samples, :))
#     logZ          = StatsFuns.logsumexp(logw, dims=1)
#     w             = exp.(logw .- logZ)

#     println(mean(1 ./ sum(w.^2, dims=1)[1,:]))

#     resample_idx  = vcat([systematic_sampling(w[:,i]) for i = 1:size(w, 2)]...)

#     z_acc     = z_prev[:, resample_idx]
#     z_acc_dev = z_acc |> device
#     logw_acc  = compute_log_weight(decoder, x_rep_dev, z_acc_dev, logit, device, settings)
#     z_acc, logw_acc
# end

function cache_samples!(::Val{:PIMH}, decoder, x_idx, z, logjoint, state, settings)
    n_samples              = settings[:n_samples]
    n_batch                = length(x_idx)
    state[:samples][x_idx] = [z[:,(i-1)*(n_samples)+1:i*(n_samples)] for i = 1:n_batch]
end

function cache_samples!(::Val{:PIMH_SIR}, decoder, x_idx, z, logjoint, state, settings)
    n_samples               = settings[:n_samples]
    n_batch                 = length(x_idx)
    state[:samples][x_idx]  = [z[:,(i-1)*(n_samples)+1:i*(n_samples)]      for i = 1:n_batch]
    state[:logjoint][x_idx] = [logjoint[(i-1)*(n_samples)+1:i*(n_samples)] for i = 1:n_batch]
end

function step!(type::Union{Val{:PIMH}, Val{:PIMH_SIR}},
               idx, epoch, encoder, decoder, params, x, x_idx, device, state, settings)
    n_batch   = size(x, 2)
    n_samples = settings[:n_samples]
    n_total   = (n_samples*2)*n_batch

    x_dev        = x |> device
    x_rep_dev    = repeat_inner(x_dev, (1, n_samples*2)) 
    z_dev, q_dev = sample_z(encoder, x_rep_dev, device)
    logw         = compute_log_weight(decoder, x_rep_dev, z_dev, q_dev, device, settings)
    z            = z_dev |> cpu

    prev_idx  = 1:2:n_total
    state_idx = 2:2:n_total

    if epoch > 1
        z_prop    = view(z,    :,   prev_idx)
        logw_prop = view(logw,      prev_idx)
        q_prop    = view(q_dev, :, prev_idx)
        z_init, logw_init = adaptive_restart(type, epoch, x_idx, decoder, x_dev, q_prop,
                                             z_prop, logw_prop, device, state, settings)
        z[:, prev_idx] = z_init
        logw[prev_idx] = logw_init
    end

    u       = rand(Float32, Int(n_total/2))
    logu    = log.(u)
    acc_sum = 0

    logα      = logw[state_idx] - logw[prev_idx]
    acc_flag  = logu .< logα
    acc_sum   = sum(acc_flag)
    rej_flag  = .!acc_flag
    acc_avg   = acc_sum / (n_samples*n_batch)

    z[:, state_idx[rej_flag]]        = z[:, prev_idx[rej_flag]]

    x_rep_dev = repeat_inner(x_dev, (1, n_samples)) 
    z         = z[:,state_idx]
    z_dev     = z |> device
    loss, back = Flux.pullback(params) do
        jsa_loss(encoder, decoder, x_rep_dev, z_dev, device, settings)
    end
    logp_x_z = joint_density(decoder, z_dev, x_rep_dev, device, settings) |> cpu
    cache_samples!(type, decoder, x_idx, z, logp_x_z, state, settings)
    
    grad = back(1f0)
    Flux.Optimise.update!(state[:opt], params, grad)
    loss, acc_avg
end

function step!(::Val{:RWS}, idx, epoch, encoder, decoder, params, x, x_idx, device, state, settings)
    n_samples = settings[:n_samples]
    n_batch   = size(x, 2)

    x_dev     = x |> device
    x_rep_dev = repeat_inner(x_dev, (1, settings[:n_samples]))
    z, q      = sample_z(encoder, x_rep_dev, device)
    logw      = compute_log_weight(decoder, x_rep_dev, z, q, device, settings) |> cpu
    for i = 1:n_batch
        idx_start = (i - 1)*n_samples + 1
        idx_stop  = i*n_samples
        logZ      = StatsFuns.logsumexp(logw[idx_start:idx_stop])
        logw[idx_start:idx_stop] .-= logZ
    end
    w = exp.(logw)

    loss, back = Flux.pullback(params) do
        rws_wake_loss(encoder,
                      decoder,
                      x_rep_dev,
                      w  |> device,
                      z,
                      n_batch,
                      device,
                      settings)
    end
    grad = back(1f0)
    Flux.Optimise.update!(state[:opt_wake], params, grad)

    if mod(idx, 5) == 0
        z, _       = sample_z(encoder, x_dev, device)
        loss, back = Flux.pullback(params) do
            rws_sleep_loss(encoder, x_dev, z, settings)
        end
        grad = back(1f0)
        Flux.Optimise.update!(state[:opt_sleep], params, grad)
    end
    loss, 1
end

function step!(::Val{:ELBO}, idx, epoch, encoder, decoder, params, x, x_idx, device, state, settings)
    loss, back = Flux.pullback(params) do
        elbo_loss(encoder, decoder, x |> device, device, state)
    end
    grad = back(1f0)
    Flux.Optimise.update!(state[:opt], params, grad)
    loss
end

function train(type::Val, settings)
    use_cuda = true
    # load hyperparamters
    settings[:seed] > 0 && Random.seed!(settings[:seed])

    # GPU config
    if use_cuda && CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load MNIST images
    train_loader, valid_loader, test_loader = get_data(settings[:n_batch])

    encoder = Encoder(settings[:input_dim], settings[:latent_dim], settings[:hidden_dim]) |> device
    decoder = Decoder(settings[:input_dim], settings[:latent_dim], settings[:hidden_dim]) |> device

    # initialize encoder and decode    # parameters
    params = Flux.params(encoder, decoder)
    state  = init_state!(type, encoder, decoder, train_loader, device, settings)

    !ispath(settings[:save_path]) && mkpath(settings[:save_path])

    # fixed input
    original, _ = first(first(get_data(10^2)))
    original    = original |> device
    image = convert_to_image(original, 10)
    image_path = joinpath(settings[:save_path], "original.png")
    save(image_path, image)

    valid_hist = []

    # training
    train_steps = 0
    @info "Start Training, total $(settings[:n_epochs]) epochs"
    for epoch = 1:settings[:n_epochs]
        @info "Epoch $(epoch)"
        progress = Progress(length(train_loader))

        for (x, x_idx) in train_loader
            loss, acc_avg = step!(type, train_steps, epoch, encoder, decoder, params, x, x_idx, device, state, settings)

            # progress meter
            next!(progress; showvalues=[(:loss, loss), (:acceptance_rate, acc_avg)]) 

            train_steps += 1
        end

        # save image
        rec_original = reconstuct(encoder, decoder, original, device)
        image        = convert_to_image(rec_original, 10)
        image_path   = joinpath(settings[:save_path], "epoch_$(epoch).png")
        save(image_path, image)
        @info "Image saved: $(image_path)"

        valid_period = settings[:valid_period]
        if mod(epoch, valid_period) == 0
            valid_mll = mapreduce(+, test_loader) do (x, _)
                x_dev     = x |> device
                x_rep_dev = repeat_inner(x_dev, (1, 512)) 
                logw      = compute_log_weight(encoder, decoder, x_rep_dev, device, settings)
                StatsFuns.logsumexp(logw) - log(length(logw))
            end / length(test_loader)

            push!(valid_hist, valid_mll)
            @info "Validation" mll=valid_mll

            t = 1:valid_period:valid_period*length(valid_hist)
            y = valid_hist
            display(Plots.plot(t, y))
        end
        GC.gc()
    end

    # save model
    model_path = joinpath(settings[:save_path], "model.bson") 
    let encoder = cpu(encoder), decoder = cpu(decoder)
        BSON.@save model_path encoder decoder
        @info "Model saved: $(model_path)"
    end
    valid_hist
end

function main()
    CUDA.allowscalar(false)

    settings = Dict{Symbol, Any}()
    settings[:η]            = 3e-4
    settings[:n_batch]      = 50
    settings[:n_epochs]     = 800
    settings[:valid_period] = 10
    settings[:input_dim]    = 28^2
    settings[:latent_dim]   = 200
    settings[:hidden_dim]   = 200
    settings[:save_path]    = "output"

    #settings[:defensive] = true
    settings[:defensive] = false
    settings[:n_samples] = 4
    settings[:seed]      = 1
    settings[:gamma]     = 0.05
    settings[:adaptive]  = false
    #train(Val(:JSA_MC), settings)
    #train(Val(:JSA), settings)
    #train(Val(:RWS), settings)
    train(Val(:PIMH), settings)
    #train(Val(:PIMH_SIR), settings)

    # for i = 1:5
    #     for γ ∈ [0.05]
    #         for n_samples ∈ [4, 2]
    #             for method ∈ ["JSA", "PIMH"]
    #                 fname = "SBN_$(method)_gamma=$(γ)_samples=$(n_samples).jld2"
    #                 data  = if isfile(fname)
    #                     FileIO.load(fname)
    #                 else
    #                     Dict{String, Any}()
    #                 end
    #                 if string(i) ∈ keys(data)
    #                     @info "skipping seed $(i)"
    #                     continue
    #                 end
    #                 settings[:seed]      = i
    #                 settings[:gamma]     = γ
    #                 settings[:n_samples] = n_samples
    #                 valid_hist           = train(Val(Symbol(method)), settings)
    #                 data[string(i)]      = valid_hist
    #                 FileIO.save(fname, data)
    #             end
    #         end
    #     end
    # end
end
