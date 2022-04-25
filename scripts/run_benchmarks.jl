
using Distributed

@everywhere using DrWatson
@everywhere @quickactivate "KLpqVI"

@everywhere using ReverseDiff
using Plots, StatsPlots
@everywhere using Flux
@everywhere using DataFrames
@everywhere using ForwardDiff
@everywhere using Zygote
@everywhere using Random123
@everywhere using ProgressMeter
@everywhere using DelimitedFiles
@everywhere using FileIO
@everywhere using UnPack
@everywhere using CUDA

@everywhere include(srcdir("KLpqVI.jl"))
@everywhere include("task/task.jl")

dispatch_optimizer(opt::String, stepsize) = begin
    if (opt == "ADAM")
        ADAM(stepsize)
    elseif (opt == "NADAM")
        NADAM(stepsize)
    elseif (opt == "Nesterov")
        Nesterov(stepsize)
    elseif (opt == "RMSProp")
        RMSProp(stepsize)
    elseif (opt == "ADAGrad")
        ADAGrad(stepsize)
    elseif (opt == "SGD")
        Descent(stepsize)
    elseif (opt == "Momentum")
        Momentum(stepsize)
    end
end

dispatch_inference_method(method::String) = begin
    if (method == "MSC_PIMH")
        MSC_PIMH()
    elseif (method == "MSC_SIMH")
        MSC_SIMH()
    elseif (method == "ELBO")
        ELBO()
    elseif (method == "MSC_CIS")
        MSC_CIS()
    elseif (method == "MSC_CISRB")
        MSC_CISRB()
    elseif (method == "SNIS")
        SNIS()
    end
end

@everywhere function run_experiment(settings::Dict)
    @unpack method, decay, optimizer, task, stepsize, defensive, n_iter, n_samples, n_reps =
        settings

    task = Val(Symbol(task))
    method = dispatch_inference_method(method)
    optimizer = dispatch_optimizer(optimizer, stepsize)

    stats = ProgressMeter.@showprogress pmap(1:n_reps) do seed_key
        seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
        prng = Random123.Philox4x(UInt64, seed, 8)
        Random123.set_counter!(prng, seed_key)
        Random.seed!(seed_key)
        run_task(
            prng,
            task,
            optimizer,
            method,
            n_iter,
            n_samples,
            defensive;
            show_progress = false,
        )
    end
    df = vcat(DataFrame.(stats)...)

    for (k, v) ∈ settings
        df[:, k] .= v
    end

    Dict("result" => df, "settings" => settings)
end

@everywhere function run_experiment_gpu(settings::Dict)
    @unpack method, decay, optimizer, task, stepsize, defensive, n_iter, n_samples, n_reps =
        settings

    task = Val(Symbol(task))
    method = dispatch_inference_method(method)
    optimizer = dispatch_optimizer(optimizer, stepsize)

    device_list = collect(CUDA.devices())
    n_devices   = length(devices())
    cpu_list    = pmap(x -> myid(), 1:n_devices) 
    key_to_id   = Dict(enumerate(cpu_list))
    id_to_key   = Dict(value => key for (key, value) in key_to_id)

    @assert length(cpu_list) == n_devices

    stats = ProgressMeter.@showprogress pmap(1:n_reps) do seed_key
        seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
        prng = Random123.Philox4x(UInt64, seed, 8)
        Random123.set_counter!(prng, seed_key)
        Random.seed!(seed_key)

        proc_id   = myid()
        proc_key  = proc_id[id_to_key]
        device_id = device_list[proc_key]
        CUDA.device!(device_id)

        run_task(
            prng,
            task,
            optimizer,
            method,
            n_iter,
            n_samples,
            defensive;
            show_progress = false,
        )
    end
    df = vcat(DataFrame.(stats)...)

    for (k, v) ∈ settings
        df[:, k] .= v
    end

    Dict("result" => df, "settings" => settings)
end

function gaussian_stepsize()
    ν = 400
    for ϵ ∈ exp10.(range(log10(0.001), log10(1.0); length = 20))
        for decay ∈ [true, false]
            for defensive ∈ [0.0, 0.001]
                for method ∈ ["MSC_SIMH", "MSC_PIMH", "MSC_CIS"]
                    for optimizer ∈ ["ADAM", "RMSProp", "Momentum", "Nesterov", "SGD"]
                        seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
                        prng = Random123.Philox4x(UInt64, seed, 8)
                        n_iter = 20000
                        n_mc = 10

                        settings = Dict{Symbol,Any}()
                        @info "starting epxeriment" settings = settings
                        produce_or_load(
                            datadir("exp_raw"),
                            settings,
                            run_experiment,
                            suffix = "jld2",
                            loadfile = false,
                            tag = false,
                        )
                    end
                end
            end
        end
    end
end

function general_benchmarks()
    defensive = nothing
    stepsize  = 0.01
    decay     = false
    optimizer = "ADAM"
    n_samples = 10
    n_reps    = 80
    n_iter    = 20000

    for task ∈ [
        "wine",
        "concrete",
        "yacht",
        "naval",
        "boston",
        "sonar",
        "ionosphere",
        "heart",
        #"australian",
        #"breast",
    ]
        for (method, n_samples) ∈ [
            ("MSC_PIMH",  10),
            ("MSC_CIS",   10),
            ("MSC_CISRB", 10),
            ("MSC_SIMH",  10),
            ("SNIS",      10),
            ("ELBO",       1),
            ("ELBO",      10)
            ]
            settings = Dict{Symbol,Any}()
            settings[:method] = method
            settings[:defensive] = defensive
            settings[:stepsize] = stepsize
            settings[:task] = task
            settings[:decay] = decay
            settings[:optimizer] = optimizer
            settings[:n_samples] = n_samples
            settings[:n_reps] = n_reps
            settings[:n_iter] = n_iter
            @info "starting epxeriment" settings = settings
            produce_or_load(
                datadir("exp_raw"),
                settings,
                run_experiment,
                suffix = "jld2",
                loadfile = false,
                tag = false,
            )
        end
    end
end

function general_benchmarks_gpu()
    defensive = nothing
    stepsize  = 0.01
    decay     = false
    optimizer = "ADAM"
    n_reps    = 20
    n_iter    = 20000

    for task ∈ [
        "australian_gpu",
        "german_gpu",
        "wine_gpu",
        "concrete_gpu",
        "yacht_gpu",
        "boston_gpu",
    ]
        for (method, n_samples) ∈ [("MSC_PIMH", 10),
                                   ("MSC_CIS", 10),
                                   ("MSC_CISRB", 10),
                                   ("MSC_SIMH", 10),
                                   ("SNIS", 10),
                                   ("ELBO", 1),
                                   ("ELBO", 10),
                                   ]
            settings = Dict{Symbol,Any}()
            settings[:method]    = method
            settings[:defensive] = defensive
            settings[:stepsize]  = stepsize
            settings[:task]      = task
            settings[:decay]     = decay
            settings[:optimizer] = optimizer
            settings[:n_samples] = n_samples
            settings[:n_reps]    = n_reps
            settings[:n_iter]    = n_iter
            @info "starting epxeriment" settings = settings
            produce_or_load(
                datadir("exp_raw"),
                settings,
                run_experiment_gpu,
                suffix = "jld2",
                loadfile = false,
                tag = false,
            )
        end
    end
end

function hyperparameter_tuning()
    for method ∈ ["MSC_PIMH"]
        for task ∈ ["boston", "sonar"]
            for defensive ∈ [nothing, 0.001]
                for decay ∈ [true, false]
                    for optimizer ∈ ["NADAM", "ADAM", "RMSProp", "ADAGrad", "Nesterov"]
                        for stepsize ∈ exp10.(range(log10(0.005), log10(0.5); length = 15))
                            settings = Dict{Symbol,Any}()

                            @info "starting epxeriment" settings = settings
                            settings[:defensive] = defensive
                            settings[:stepsize] = stepsize
                            settings[:method] = method
                            settings[:task] = task
                            settings[:decay] = decay
                            settings[:optimizer] = optimizer
                            settings[:n_samples] = 10
                            settings[:n_reps] = 8
                            settings[:n_iter] = 10000
                            produce_or_load(
                                datadir("exp_raw"),
                                settings,
                                run_experiment,
                                suffix = "jld2",
                                loadfile = false,
                                tag = false,
                            )
                        end
                    end
                end
            end
        end
    end
end

function gaussian_stepsize()
    ν = 400
    for ϵ ∈ exp10.(range(log10(0.001), log10(1.0); length = 20))
        for decay ∈ [true, false]
            for defensive ∈ [0.0, 0.001]
                for method ∈ ["MSC_SIMH", "MSC_PIMH", "MSC_CIS"]
                    for optimizer ∈ ["ADAM", "RMSProp", "Momentum", "Nesterov", "SGD"]
                        seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
                        prng = Random123.Philox4x(UInt64, seed, 8)
                        n_iter = 20000
                        n_mc = 10

                        settings = Dict{Symbol,Any}()
                        @info "starting epxeriment" settings = settings
                        settings[:defensive] = defensive
                        settings[:stepsize] = ϵ
                        settings[:method] = method
                        settings[:task] = "gaussian_correlated"
                        settings[:decay] = decay
                        settings[:optimizer] = optimizer
                        settings[:n_samples] = 10
                        settings[:n_reps] = 20
                        settings[:n_iter] = 10000
                        produce_or_load(
                            datadir("exp_raw"),
                            settings,
                            run_experiment,
                            suffix = "jld2",
                            loadfile = false,
                            tag = false,
                        )
                    end
                end
            end
        end
    end
end
