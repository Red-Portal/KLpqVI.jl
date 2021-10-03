# Variational Autoencoder(VAE)
#
# Auto-Encoding Variational Bayes
# Diederik P Kingma, Max Welling
# https://arxiv.org/abs/1312.6114

using DrWatson
@quickactivate "KLpqVI"

using BSON
using CUDA
using Distributions
using DrWatson: struct2dict
using Flux
using Flux: @functor, chunk
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Images
using Logging: with_logger
using MLDatasets
using MLDataUtils: splitobs
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Random
using Statistics
using Plots
using Zygote
using LinearAlgebra
using JLD2
using FileIO
using Base.GC
using StatsFuns

using KernelAbstractions
using CUDAKernels

# load MNIST images and return loader
function get_data(batch_size)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest,  ytest  = MLDatasets.MNIST.testdata(Float32)

    xtrain = reshape(xtrain, 28^2, :)
    xtest  = reshape(xtest,  28^2, :)

    n_data = size(xtrain, 2)
    idxs   = shuffle(1:n_data)
    xtrain = xtrain[:,idxs]
    ytrain = collect(1:size(xtrain, 2))

    xtrain, xvalid = splitobs(xtrain, at=0.83333)
    ytrain, yvalid = splitobs(ytrain, at=0.83333)

    @info "Dataset Organization" train=size(xtrain) valid=size(xvalid) test=size(xtest)

    DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true),
    DataLoader((xvalid, yvalid), batchsize=1, shuffle=false),
    DataLoader((xtest,  ytest),  batchsize=1, shuffle=false)
end

Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = 
    Chain(
        Dense(input_dim,  hidden_dim, NNlib.leakyrelu, init=Flux.kaiming_uniform), # linear
        Dense(hidden_dim, hidden_dim, NNlib.leakyrelu, init=Flux.kaiming_uniform), # linear
        Dense(hidden_dim, latent_dim, init=Flux.kaiming_uniform)
    )

Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = 
    Chain(
        Dense(latent_dim, hidden_dim, NNlib.leakyrelu, init=Flux.kaiming_uniform),
        Dense(hidden_dim, hidden_dim, NNlib.leakyrelu, init=Flux.kaiming_uniform),
        Dense(hidden_dim, input_dim, init=Flux.kaiming_uniform)
    )

function sample_z(encoder, x, device)
    logit = encoder(x)
    p     = sigmoid.(logit)
    u     = rand(Float32, size(logit)) |> device
    z     = Float32.(p .> u)
    z, logit
end

function sample_z(encoder, x, u, device)
    logit = encoder(x)
    p     = sigmoid.(logit) 
    z     = Float32.(p .> u)
    z, logit
end

function sample_z(logit, device)
    p = sigmoid.(logit)
    u = rand(Float32, size(logit)) |> device
    z = Float32.(p .> u)
    z
end

function reconstuct(encoder, decoder, x, device)
    z, logit = sample_z(encoder, x, device)
    logit, decoder(z)
end

function joint_density(z, x, x_recon, device)
    logit0   = fill(0f0, size(z)) |> device
    logp_z   = -logitbinarycrossentropy(logit0,  z, agg=xᵢ->sum(xᵢ, dims=1))[1,:]
    logp_x_z = -logitbinarycrossentropy(x_recon, x, agg=xᵢ->sum(xᵢ, dims=1))[1,:]
    logp_x_z + logp_z
end

function variational_density(z, logit)
    -logitbinarycrossentropy(logit, z, agg=xᵢ->sum(xᵢ, dims=1))[1,:]
end

function compute_log_weight(encoder, decoder, x, device)
    z, logit = sample_z(encoder, x, device)
    x_recon  = decoder(z)
    logp_x_z = joint_density(z, x, x_recon, device)  |> cpu
    logq_z   = variational_density(z, logit) |> cpu
    logp_x_z - logq_z
end

function compute_log_weight(decoder, x, z, logit, device)
    x_recon  = decoder(z)
    logp_x_z = joint_density(z, x, x_recon, device)  |> cpu
    logq_z   = variational_density(z, logit)         |> cpu
    logp_x_z - logq_z
end

function rws_wake_loss(encoder, decoder, x, w, z, n_batch, device)
    logit    = encoder(x)
    z_stop   = Zygote.dropgrad(z)
    x_recon  = decoder(z_stop)
    logp_x_z = joint_density(z_stop, x, x_recon, device)
    logq_z   = variational_density(z_stop, logit)
    -dot(logp_x_z + logq_z, w) / n_batch
end

function rws_sleep_loss(encoder, x, z)
    logit  = encoder(x)
    logq_z = variational_density(z, logit)
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

function init_state!(::Val{:JSA}, encoder, decoder, train_loader, device, settings)
    state           = Dict{Symbol, Any}()
    state[:opt]     = ADAM(settings[:η])
    state[:samples] = [randn(Float32, settings[:latent_dim]) for i = 1:train_loader.nobs]
    state
end

function init_state!(::Val{:JSA_MC}, encoder, decoder, train_loader, device, settings)
    state            = Dict{Symbol, Any}()
    state[:opt]      = ADAM(settings[:η])
    state[:samples]  = [randn(Float32, settings[:latent_dim]) for i = 1:train_loader.nobs]
    state[:logjoint] = [Inf                                   for i = 1:train_loader.nobs]
    state
end

function init_state!(::Val{:PIMH}, encoder, decoder, train_loader, device, settings)
    state           = Dict{Symbol, Any}()
    state[:opt]     = ADAM(settings[:η])
    state[:samples] = [randn(Float32, settings[:latent_dim], settings[:n_samples])
                       for i = 1:train_loader.nobs]
    state
end

function init_state!(::Val{:PIMH_MC}, encoder, decoder, train_loader, device, settings)
    state           = Dict{Symbol, Any}()
    state[:opt]     = ADAM(settings[:η])
    state[:samples] = [randn(Float32, settings[:latent_dim], settings[:n_samples])
                       for i = 1:train_loader.nobs]
    state[:logjoint] = [fill(Inf, settings[:n_samples]) for i = 1:train_loader.nobs]
    state
end

function jsa_loss(encoder, decoder, x, z, device)
    logit    = encoder(x)
    x_recon  = decoder(z)
    logp_x_z = joint_density(z, x, x_recon, device)
    logq_z   = variational_density(z, logit)
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

function adaptive_restart(::Val{:JSA}, idx, decoder, x_dev, logit, z_prop, logw_prop, device, state, settings)
    γ          = settings[:gamma]
    z_prev     = hcat(state[:samples][idx]...)
    z_prev_dev = z_prev |> device
    logw_prev  = compute_log_weight(decoder, x_dev, z_prev_dev, logit, device)

    R = mean(min.(0, logw_prop - logw_prev))
    p = tanh(-γ*R)
    if (rand(Bernoulli(p)))
        z_prop, logw_prop
    else
        z_prev, logw_prev
    end
end

function adaptive_restart(::Val{:JSA_MC}, idx, decoder, x_dev, logit, z_prop, logw_prop, device, state, settings)
    z_prev  = hcat(state[:samples ][idx]...)
    logprop_prev = vcat(state[:logjoint][idx]...)
    n_dims  = size(z_prev, 1)

    z_prev_dev   = z_prev |> device
    x_prev_recon = decoder(z_prev_dev)
    logp_prev    = joint_density(z_prev_dev, x_dev, x_prev_recon, device) |> cpu
    logq_prev    = variational_density(z_prev_dev, logit) |> cpu
    logw_prev    = logp_prev - logq_prev
    logr_prev    = logp_prev - (logaddexp.(logq_prev, logprop_prev) .- log(2))

    logα     = logw_prop .- logaddexp.(logr_prev, logw_prop)
    logu     = log.(rand(size(x_dev, 2)))
    acc_flag = logu .< logα
    rej_flag = .!acc_flag

    z_prop[:,rej_flag]  = z_prev[:,rej_flag]
    logw_prop[rej_flag] = logw_prev[rej_flag]
    z_prop, logw_prop
end

function cache_samples!(::Val{:JSA}, encoder, x_idx, x_dev, z, device, state)
    state[:samples][x_idx] = [z[:,i] for i = 1:size(z, 2)]
end

function cache_samples!(::Val{:JSA_MC}, encoder, x_idx, x_dev, z, device, state)
    state[:samples][x_idx]  = [z[:,i]  for i = 1:size(z, 2)]

    logit = encoder(x_dev)
    logq  = variational_density(z |> device, logit) |> cpu
    state[:logjoint][x_idx] = [logq[i] for i = 1:size(z, 2)]
end

function step!(type::Union{Val{:JSA}, Val{:JSA_MC}},
               idx, epoch, encoder, decoder, params, x, x_idx, device, state, settings)
    n_batch   = size(x, 2)
    n_samples = settings[:n_samples]
    n_total   = (n_samples + 1)*n_batch

    x_dev     = x |> device
    logit     = encoder(x_dev)
    logit_rep = repeat_inner(logit, (1, n_samples + 1))
    z_dev     = sample_z(logit_rep, device)
    x_rep_dev = repeat_inner(x_dev, (1, n_samples + 1)) 
    logw      = compute_log_weight(decoder, x_rep_dev, z_dev, logit_rep, device)
    z         = z_dev |> cpu

    if epoch > 1
        z_prop            = view(z,    :, 1:n_samples+1:n_total)
        logw_prop         = view(logw,    1:n_samples+1:n_total)
        z_init, logw_init = adaptive_restart(type, x_idx, decoder, x_dev, logit, z_prop, logw_prop, device, state, settings)
        z[:, 1:n_samples+1:n_total]  = z_init
        logw[ 1:n_samples+1:n_total] = logw_init
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
        jsa_loss(encoder, decoder, x_rep_dev, z |> device, device)
    end
    z_last    = view(z,    :, n_samples:n_samples:n_samples*n_batch)
    cache_samples!(type, encoder, x_idx, x_dev, z_last, device, state)

    grad = back(1f0)
    Flux.Optimise.update!(state[:opt], params, grad)
    loss, acc_avg
end

function adaptive_restart(::Val{:PIMH}, idx, decoder, x_dev, logit, z_prop, logw_prop, device, state, settings)
    γ         = settings[:gamma]
    n_samples = settings[:n_samples]
    logit_rep = repeat_inner(logit, (1, n_samples))
    x_rep_dev = repeat_inner(x_dev, (1, n_samples))
    
    z_prev     = hcat(state[:samples][idx]...)
    z_prev_dev = z_prev |> device
    logw_prev  = compute_log_weight(decoder, x_rep_dev, z_prev_dev, logit_rep, device)

    R = mean(min.(0, logw_prop - logw_prev))
    p = tanh(-γ*R)
    #println(mean(min.(1, exp.(logw_prop - logw_prev))), " ", p)
    if (rand(Bernoulli(p)))
       z_prop, logw_prop
    else
       z_prev, logw_prev
    end
end

function adaptive_restart(::Val{:PIMH_MC}, idx, decoder, x_dev, logit, z_prop, logw_prop, device, state, settings)
    n_samples = settings[:n_samples]
    logit_rep = repeat_inner(logit, (1, n_samples))
    x_rep_dev = repeat_inner(x_dev, (1, n_samples))

    # z_prev_dev   = z_prev |> device
    # x_prev_recon = decoder(z_prev_dev)
    # logp_prev    = joint_density(z_prev_dev, x_dev, x_prev_recon, device) |> cpu
    # logq_prev    = variational_density(z_prev_dev, logit) |> cpu
    # logw_prev    = logp_prev - logq_prev
    # logr_prev    = logp_prev - (logaddexp.(logq_prev, logprop_prev) .- log(2))

    # logα     = logw_prop .- logaddexp.(logr_prev, logw_prop)
    # logu     = log.(rand(size(x_dev, 2)))
    # acc_flag = logu .< logα
    # rej_flag = .!acc_flag
    
    z_prev       = hcat(state[:samples][ idx]...)
    logprop_prev = vcat(state[:logjoint][idx]...)
    z_prev_dev   = z_prev |> device
    x_recon      = decoder(z_prev_dev)
    logp_prev    = joint_density(z_prev_dev, x_rep_dev, x_recon, device)  |> cpu
    logq_prev    = variational_density(z_prev_dev, logit_rep) |> cpu
    logr_prev    = logp_prev - (logaddexp.(logq_prev, logprop_prev) .- log(2))
    logw_prev    = logp_prev - logq_prev

    logα     = logw_prop .- logaddexp.(logr_prev, logw_prop)
    logu     = log.(rand(size(x_rep_dev, 2)))
    acc_flag = logu .< logα
    rej_flag = .!acc_flag

    z_prop[:,rej_flag]  = z_prev[:,rej_flag]
    logw_prop[rej_flag] = logw_prev[rej_flag]
    z_prop, logw_prop
end

function cache_samples!(::Val{:PIMH}, encoder, x_idx, x, z, state, settings)
    n_samples              = settings[:n_samples]
    n_batch                = length(x_idx)
    state[:samples][x_idx] = [z[:,(i-1)*(n_samples)+1:i*(n_samples)] for i = 1:n_batch]
end

function cache_samples!(::Val{:PIMH_MC}, encoder, x_idx, x, z, state, settings)
    n_samples               = settings[:n_samples]
    n_batch                 = length(x_idx)
    state[:samples][x_idx]  = [z[   :,(i-1)*(n_samples)+1:i*(n_samples)] for i = 1:n_batch]

    logit = encoder(x)
    logq  = variational_density(z, logit)
    state[:logjoint][x_idx] = [logq[  (i-1)*(n_samples)+1:i*(n_samples)] for i = 1:n_batch]
end

function step!(type::Union{Val{:PIMH}, Val{:PIMH_MC}},
               idx, epoch, encoder, decoder, params, x, x_idx, device, state, settings)
    n_batch   = size(x, 2)
    n_samples = settings[:n_samples]
    n_total   = (n_samples*2)*n_batch

    x_dev     = x |> device
    logit     = encoder(x_dev)
    logit_rep = repeat_inner(logit, (1, n_samples*2))
    z_dev     = sample_z(logit_rep, device)
    x_rep_dev = repeat_inner(x_dev, (1, n_samples*2)) 
    logw      = compute_log_weight(decoder, x_rep_dev, z_dev, logit_rep, device)
    z         = z_dev |> cpu

    prev_idx  = 1:2:n_total
    state_idx = 2:2:n_total

    if epoch > 1
        z_prop    = view(z,    :, prev_idx)
        logw_prop = view(logw,    prev_idx)
        z_init, logw_init = adaptive_restart(type, x_idx, decoder, x_dev, logit, z_prop, logw_prop, device, state, settings)
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

    z[:, state_idx[rej_flag]] = z[:, prev_idx[rej_flag]]

    x_rep_dev = repeat_inner(x_dev, (1, n_samples)) 
    z         = z[:,state_idx]
    z_dev     = z |> device
    loss, back = Flux.pullback(params) do
        jsa_loss(encoder, decoder, x_rep_dev, z_dev, device)
    end
    cache_samples!(type, encoder, x_idx, x_rep_dev, z_dev, state, settings)
    
    grad = back(1f0)
    Flux.Optimise.update!(state[:opt], params, grad)
    loss, acc_avg
end

function step!(::Val{:RWS}, idx, epoch, encoder, decoder, params, x, x_idx, device, state, settings)
    x_repeat  = repeat(x, inner=(1, settings[:n_samples]))
    n_latent  = settings[:latent_dim]
    n_samples = settings[:n_samples]
    n_batch   = size(x, 2)

    x_rep_dev = x_repeat |> device
    u         = rand(Float32, (n_latent, size(x_repeat,2)))
    z, logit  = sample_z(encoder, x_rep_dev, u |> device, device)
    logw      = compute_log_weight(decoder, x_rep_dev, z, logit, device) |> cpu
    for i = 1:n_batch
        idx_start = (i - 1)*n_samples + 1
        idx_stop  = i*n_samples
        logZ      = logsumexp(logw[idx_start:idx_stop])
        logw[idx_start:idx_stop] .-= logZ
    end
    w = exp.(logw)

    loss, back = Flux.pullback(params) do
        rws_wake_loss(encoder,
                      decoder,
                      x_repeat |> device,
                      w        |> device,
                      z        |> device,
                      n_batch,
                      device)
    end
    grad = back(1f0)
    Flux.Optimise.update!(state[:opt_wake], params, grad)

    if mod(idx, 5) == 0
        x_dev = x |> device
        z, _  = sample_z(encoder, x_dev, device)
        loss, back = Flux.pullback(params) do
            rws_sleep_loss(encoder, x_dev, z)
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
        # _, rec_original = reconstuct(encoder, decoder, original, device)
        # rec_original = sigmoid.(rec_original)
        # image = convert_to_image(rec_original, 10)
        # image_path = joinpath(settings[:save_path], "epoch_$(epoch).png")
        # save(image_path, image)
        # @info "Image saved: $(image_path)"

        valid_period = settings[:valid_period]
        if mod(epoch, valid_period) == 0
            valid_mll = mapreduce(+, test_loader) do (x, _)
                x_dev = x |> device
                logit = encoder(x_dev)
                p     = sigmoid.(logit)
                u     = rand(Float32, (settings[:latent_dim], 256))  |> device
                z_dev = Float32.(p .> u)
                logw  = compute_log_weight(decoder, x_dev, z_dev, logit, device) |> cpu
                StatsFuns.logsumexp(logw) - log(length(logw))
            end / length(valid_loader)

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
    settings[:n_epochs]     = 1000
    settings[:valid_period] = 10
    settings[:input_dim]    = 28^2
    settings[:latent_dim]   = 200
    settings[:hidden_dim]   = 200
    settings[:save_path]    = "output"

    settings[:n_epochs]  = 100
    settings[:n_samples] = 2
    settings[:seed]      = 1
    settings[:gamma]     = 0.05
    #train(Val(:JSA), settings)
    train(Val(:JSA_MC), settings)
    train(Val(:PIMH_MC), settings)

    # for i = 1:5
    #     for γ ∈ [0.05]
    #         for n_samples ∈ [2, 4]
    #             for method ∈ ["JSA", "PIMH"]
    #                 fname = "VAE_$(method)_gamma=$(γ)_samples=$(n_samples).jld2"
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
