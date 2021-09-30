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

struct Encoder
    linear
    μ
    logσ
end
@functor Encoder
    
Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Encoder(
    Chain(
        Dense(input_dim, hidden_dim,  tanh), # linear
        Dense(hidden_dim, hidden_dim, tanh), # linear
    ),
    Dense(hidden_dim, latent_dim), # μ
    Dense(hidden_dim, latent_dim), # logσ
)

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Chain(
    Chain(
        Dense(latent_dim, hidden_dim, tanh),
        Dense(hidden_dim, hidden_dim, tanh),
    ),
    Dense(hidden_dim, hidden_dim, tanh), # linear
    Dense(hidden_dim, input_dim)
)

function reconstuct(encoder, decoder, x, device)
    μ, logσ = encoder(x)
    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    μ, logσ, decoder(z)
end

function elbo_loss(encoder, decoder, x, device, state)
    μ, logσ, decoder_z = reconstuct(encoder, decoder, x, device)
    len      = size(x)[end]
    kl_q_p   = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len
    logp_x_z = -logitbinarycrossentropy(decoder_z, x, agg=sum) / len
    -logp_x_z + kl_q_p
end

function joint_density(z, x, x_recon)
    logp_z   = sum(z.^2/-2 .+ Float32(log(2*π)/-2), dims=1)[1,:]
    logp_x_z = -logitbinarycrossentropy(x_recon, x, agg=xᵢ->sum(xᵢ, dims=1))[1,:] + logp_z
end

function variational_density(z, μ, logσ)
    σ²     = max.(exp.(2*logσ), 1f-7)
    logq_z = sum((z .- μ).^(2)./σ²/-2 .+ Float32(log(2*π)/-2) .+ -logσ, dims=1)[1,:]
end

function variational_density(ϵ, logσ)
    logq_z = sum(ϵ.^2/-2 .+ Float32(log(2*π)/-2) .+ -logσ, dims=1)[1,:]
end

function compute_log_weight(encoder, decoder, ϵ, x)
    μ, logσ  = encoder(x)
    z        = μ .+ ϵ .* exp.(logσ)
    x_recon  = decoder(z)
    logp_x_z = joint_density(z, x, x_recon)
    logq_z   = variational_density(ϵ, logσ)
    logp_x_z - logq_z
end

function rws_wake_loss(encoder, decoder, x, w, ϵ, n_batch, device, state)
    μ, logσ  = encoder(x)
    z        = μ .+ ϵ.*exp.(logσ)
    z_stop   = Zygote.dropgrad(z)
    x_recon  = decoder(z_stop)

    logp_x_z = joint_density(z_stop, x, x_recon)
    logq_z   = variational_density(z_stop, μ, logσ)
    -dot(logp_x_z + logq_z, w) / n_batch
end

function rws_sleep_loss(encoder, decoder, x, device, n_latent, state)
    ϵ       = randn(Float32, n_latent) |> device
    μ, logσ = encoder(x)
    z       = μ .+ ϵ.*exp.(logσ)
    z_stop  = Zygote.dropgrad(z)
    logq_z   = variational_density(z_stop, μ, logσ)
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

function init_state!(::Val{:PIMH}, encoder, decoder, train_loader, device, settings)
    state           = Dict{Symbol, Any}()
    state[:opt]     = ADAM(settings[:η])
    state[:samples] = [randn(Float32, settings[:latent_dim], settings[:n_samples])
                       for i = 1:train_loader.nobs]
    state
end

function jsa_loss(encoder, decoder, x, z, device)
    μ, logσ  = encoder(x)
    x_recon  = decoder(z)
    logp_x_z = joint_density(z, x, x_recon)
    logq_z   = variational_density(z, μ, logσ)
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

function adaptive_restart(::Val{:JSA}, idx, decoder, x_dev, μ, logσ, z_prop, logw_prop, device, state, settings)
    γ            = 0.05
    z_prev       = hcat(state[:samples][idx]...)
    z_prev_dev   = z_prev |> device
    x_recon_prev = decoder(z_prev_dev)

    logp_x_z_prev = joint_density(z_prev_dev, x_dev, x_recon_prev) |> cpu
    logq_z_prev   = variational_density(z_prev_dev, μ, logσ) |> cpu
    logw_prev     = logp_x_z_prev - logq_z_prev

    R = mean(min.(0, logw_prop - logw_prev))
    p = tanh(-γ*R)
    #println(mean(min.(1, exp.(logw_prop - logw_prev))))
    if (rand(Bernoulli(p)))
        z_prop, logw_prop
    else
        z_prev, logw_prev
    end
end

function step!(::Val{:JSA}, idx, epoch, encoder, decoder, params, x, x_idx, device, state, settings)
    n_batch   = size(x, 2)
    n_samples = settings[:n_samples]
    n_total   = (n_samples + 1)*n_batch

    x_dev     = x |> device
    μ, logσ   = encoder(x_dev)
    μ_rep     = repeat_inner(μ,    (1, n_samples + 1)) 
    logσ_rep  = repeat_inner(logσ, (1, n_samples + 1)) 

    ϵ         = randn(Float32, (settings[:latent_dim], size(μ_rep, 2))) |> device
    z_dev     = μ_rep .+ ϵ.*exp.(logσ_rep)
    x_recon   = decoder(z_dev)
    x_rep_dev = repeat_inner(x_dev, (1, n_samples + 1)) 

    logp_x_z = joint_density(z_dev, x_rep_dev, x_recon)    |> cpu
    logq_z   = variational_density(z_dev, μ_rep, logσ_rep) |> cpu
    logw     = logp_x_z - logq_z
    z        = z_dev |> cpu

    z_prop            = view(z,    :, 1:n_samples+1:n_total)
    logw_prop         = view(logw,    1:n_samples+1:n_total)
    z_init, logw_init = adaptive_restart(Val(:JSA), x_idx, decoder, x_dev, μ, logσ, z_prop, logw_prop, device, state, settings)
    z[:, 1:n_samples+1:n_total]  = z_init
    logw[ 1:n_samples+1:n_total] = logw_init

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
    z_last    = view(z, :, n_samples:n_samples:n_samples*n_batch)
    state[:samples][x_idx] = [z_last[:,i] for i = 1:size(z_last, 2)]

    grad = back(1f0)
    Flux.Optimise.update!(state[:opt], params, grad)
    loss, acc_avg
end

function adaptive_restart(::Val{:PIMH}, idx, decoder, x_dev, μ, logσ, z_prop, logw_prop, device, state, settings)
    γ            = 0.05
    #γ            = 0.1

    n_samples = settings[:n_samples]
    μ_rep     = repeat_inner(μ,     (1, n_samples))
    logσ_rep  = repeat_inner(logσ,  (1, n_samples))
    x_dev_rep = repeat_inner(x_dev, (1, n_samples))
    
    #ϵ′            = randn(Float32, (settings[:latent_dim], size(μ_rep, 2))) |> device
    #z_prev′       = μ_rep + ϵ′.*exp.(logσ_rep) |> cpu
    #state[:samples][idx] = [z_prev′[:,(i-1)*n_samples+1:i*n_samples] for i = 1:size(x_dev, 2)]

    z_prev       = hcat(state[:samples][idx]...)
    z_prev_dev   = z_prev |> device
    x_recon_prev = decoder(z_prev_dev)
    
    logp_x_z_prev = joint_density(z_prev_dev, x_dev_rep, x_recon_prev) |> cpu
    logq_z_prev   = variational_density(z_prev_dev, μ_rep, logσ_rep)   |> cpu
    logw_prev     = logp_x_z_prev - logq_z_prev

    R = mean(min.(0, logw_prop - logw_prev))
    p = tanh(-γ*R)
    #println(mean(min.(1, exp.(logw_prop - logw_prev))), " ", p)
    if (rand(Bernoulli(p)))
       z_prop, logw_prop
    else
       z_prev, logw_prev
    end
end

function step!(::Val{:PIMH}, idx, epoch, encoder, decoder, params, x, x_idx, device, state, settings)
    n_batch   = size(x, 2)
    n_samples = settings[:n_samples]
    n_total   = (n_samples*2)*n_batch

    x_dev     = x |> device
    μ, logσ   = encoder(x_dev)
    μ_rep     = repeat_inner(μ,    (1, n_samples*2)) 
    logσ_rep  = repeat_inner(logσ, (1, n_samples*2)) 

    ϵ         = randn(Float32, (settings[:latent_dim], size(μ_rep, 2))) |> device
    z_dev     = μ_rep .+ ϵ.*exp.(logσ_rep)
    x_recon   = decoder(z_dev)
    x_rep_dev = repeat_inner(x_dev, (1, n_samples*2)) 

    logp_x_z = joint_density(z_dev, x_rep_dev, x_recon)    |> cpu
    logq_z   = variational_density(z_dev, μ_rep, logσ_rep) |> cpu
    logw     = logp_x_z - logq_z
    z        = z_dev |> cpu

    prev_idx  = 1:2:n_total
    state_idx = 2:2:n_total

    z_prop    = view(z,    :, prev_idx)
    logw_prop = view(logw,    prev_idx)
    z_init, logw_init = adaptive_restart(Val(:PIMH), x_idx, decoder, x_dev, μ, logσ, z_prop, logw_prop, device, state, settings)
    z[:, prev_idx] = z_init
    logw[prev_idx] = logw_init

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
    loss, back = Flux.pullback(params) do
        jsa_loss(encoder, decoder, x_rep_dev, z |> device, device)
    end

    state[:samples][x_idx] = [z[:,(i-1)*(n_samples)+1:i*(n_samples)] for i = 1:n_batch]

    #
    # μ_rep     = (μ_rep     |> cpu)[:, 1:2:n_total] |> device
    # logσ_rep  = (logσ_rep  |> cpu)[:, 1:2:n_total] |> device #repeat_inner(logσ,  (1, n_samples))
    # x_dev_rep = x_rep_dev #repeat_inner(x_dev, (1, n_samples))
    # z_prev       = hcat(state[:samples][x_idx]...)
    # z_prev_dev   = z_prev |> device
    # x_recon_prev = decoder(z_prev_dev)
    # logp_x_z_prev = joint_density(z_prev_dev, x_dev_rep, x_recon_prev) |> cpu
    # logq_z_prev   = variational_density(z_prev_dev, μ_rep, logσ_rep)   |> cpu
    # logw_prev     = logp_x_z_prev - logq_z_prev
    # R = mean(logw_prev)

    #state[:samples][x_idx] = [z[:,(i-1)*n_samples+1:i*n_samples] for i = 1:n_batch]

    #μ_rep     = repeat_inner(μ,     (1, n_samples))
    #logσ_rep  = repeat_inner(logσ,  (1, n_samples))
    #x_dev_rep = repeat_inner(x_dev, (1, n_samples))
    #μ_rep     = (μ     |> cpu)[:, 1:2:n_total] |> device
    #logσ_rep  = (logσ  |> cpu)[:, 1:2:n_total] |> device #repeat_inner(logσ,  (1, n_samples))
    #x_dev_rep = (x_dev |> cpu)[:, 1:2:n_total] |> device #repeat_inner(x_dev, (1, n_samples))
    # z_prev       = z
    # z_prev_dev   = z_prev |> device
    # x_recon_prev = decoder(z_prev_dev)
    # logp_x_z_prev = joint_density(z_prev_dev, x_dev_rep, x_recon_prev) |> cpu
    # logq_z_prev   = variational_density(z_prev_dev, μ_rep, logσ_rep)   |> cpu
    # logw_prev     = logp_x_z_prev - logq_z_prev
    #
    
    #R_now = mean(logw_prev)
    #println("$(R) $(R_now)")

    grad = back(1f0)
    Flux.Optimise.update!(state[:opt], params, grad)
    loss, acc_avg
end

function sample_z(encoder, decoder, x, n_latent, device)
    μ, logσ  = encoder(x)
    ϵ        = randn(Float32, (n_latent, size(x,2))) |> device
    z        = μ + ϵ.*exp.(logσ)
    x_recon  = decoder(z)
    logp_x_z = joint_density(z, x, x_recon)
    logq_z   = variational_density(z, μ, logσ)
    logw     = logp_x_z - logq_z
    z, μ, logσ, logw
end

function step!(::Val{:RWS}, idx, epoch, encoder, decoder, params, x, x_idx, device, state, settings)
    x_repeat  = repeat(x, inner=(1, settings[:n_samples]))
    n_latent  = settings[:latent_dim]
    n_samples = settings[:n_samples]
    n_batch   = size(x, 2)

    ϵ    = randn(Float32, (n_latent, size(x_repeat,2)))
    logw = compute_log_weight(encoder, decoder, ϵ |> device, x_repeat |> device) |> cpu
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
                      ϵ        |> device,
                      n_batch,
                      device,
                      state)
    end
    grad = back(1f0)
    Flux.Optimise.update!(state[:opt_wake], params, grad)

    if mod(idx, 5) == 0
        loss, back = Flux.pullback(params) do
            rws_sleep_loss(encoder,
                           decoder,
                           x |> device,
                           device,
                           settings[:latent_dim],
                           state)
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
    params = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)
    state  = init_state!(type, encoder, decoder, train_loader, device, settings)

    !ispath(settings[:save_path]) && mkpath(settings[:save_path])

    # fixed input
    original, _ = first(first(get_data(10^2)))
    original    = original |> device
    image = convert_to_image(original, 10)
    image_path = joinpath(settings[:save_path], "original.png")
    save(image_path, image)

    valid_hist = Float64[]

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
        _, _, rec_original = reconstuct(encoder, decoder, original, device)
        rec_original = sigmoid.(rec_original)
        image = convert_to_image(rec_original, 10)
        image_path = joinpath(settings[:save_path], "epoch_$(epoch).png")
        save(image_path, image)
        @info "Image saved: $(image_path)"

        #if mod(epoch, 5) == 0
        valid_mll = mapreduce(+, valid_loader) do (x, _)
            ϵ    = randn(Float32, (settings[:latent_dim], settings[:n_batch]))
            logw = compute_log_weight(encoder, decoder, ϵ |> device, x |> device)
            logsumexp(logw) - log(length(logw))
        end / length(valid_loader)

        push!(valid_hist, valid_mll)
        @info "Validation" mll=valid_mll

        display(Plots.plot(valid_hist))
        #end
    end

    # save model
    model_path = joinpath(settings[:save_path], "model.bson") 
    let encoder = cpu(encoder), decoder = cpu(decoder)
        BSON.@save model_path encoder decoder
        @info "Model saved: $(model_path)"
    end
end

function main()
    CUDA.allowscalar(false)

    settings = Dict{Symbol, Any}()
    settings[:η]          = 3e-4
    settings[:n_batch]    = 64
    settings[:n_samples]  = 4
    settings[:n_epochs]   = 10
    settings[:seed]       = 1
    settings[:input_dim]  = 28^2
    settings[:latent_dim] = 10
    settings[:hidden_dim] = 600
    settings[:save_path]  = "output"

    #train(Val(:PIMH), settings)
    train(Val(:PIMH), settings)
end
