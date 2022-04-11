
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
@everywhere using ParameterSchedulers: Scheduler

@everywhere include(srcdir("KLpqVI.jl"))
@everywhere include("task/task.jl")

@everywhere begin
    using ParameterSchedulers
    @eval ParameterSchedulers begin
        struct Sqrt <: AbstractSchedule{false}
            start::Float64
        end
        Base.eltype(::Type{Sqrt}) = Float64
        (schedule::Sqrt)(t) = schedule.start / sqrt(t)

        export Sqrt
    end
end

@everywhere function run_experiment(settings::Dict)
    @unpack method, decay, optimizer, task, stepsize, defensive, n_iter, n_samples, n_reps = settings

    task   = Val(Symbol(task))
    method = if(method == "MSC_PIMH")
        MSC_PIMH()
    elseif(method == "MSC_SIMH")
        MSC_SIMH()
    elseif(method == "ELBO")
        ELBO()
    elseif(method == "MSC_CIS")
        MSC_CIS()
    elseif(method == "MSC_CISRB")
        MSC_CISRB()
    elseif(method == "SNIS")
        SNIS()
    end

    optimizer = if(optimizer == "ADAM")
        ADAM(stepsize)
    elseif(optimizer == "NADAM")
        NADAM(stepsize)
    elseif(optimizer == "Nesterov")
        Nesterov(stepsize)
    elseif(optimizer == "RMSProp")
        RMSProp(stepsize)
    elseif(optimizer == "ADAGrad")
        ADAGrad(stepsize)
    end

    optimizer = if(decay)
        Scheduler(Sqrt(stepsize), optimizer)
    else
        optimizer
    end

    stats   = ProgressMeter.@showprogress pmap(1:n_reps) do seed_key
        seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
        prng = Random123.Philox4x(UInt64, seed, 8)
        Random123.set_counter!(prng, seed_key)
        Random.seed!(seed_key)
        run_task(prng,
                 task,
                 optimizer,
                 method,
                 n_iter,
                 n_samples,
                 defensive;
                 show_progress=false)
    end
    df = vcat(DataFrame.(stats)...)

    for (k,v) ∈ settings
        df[:,k] .= v
    end

    Dict("result"=>df, "settings"=>settings)
end

# function general_benchmarks()
#     for task ∈ ["pima",
#                 "heart",
#                 "german"]
#         for settings ∈ [Dict(:method=>"MSC_PIMH",
#                              :task  =>task,
#                              :n_samples=>10),

#                         Dict(:method=>"MSC_CIS",
#                              :task  =>task,
#                              :n_samples=>10),

#                         Dict(:method=>"MSC_CISRB",
#                              :task  =>task,
#                              :n_samples=>10),

#                         Dict(:method=>"MSC_SIMH",
#                              :task  =>task,
#                              :n_samples=>10),

#                         Dict(:method=>"SNIS",
#                              :task  =>task,
#                              :n_samples=>10),

#                         Dict(:method=>"ELBO",
#                              :task  =>task,
#                              :n_samples=>10),
#                         ]
#             @info "starting epxeriment" settings=settings
#             settings[:n_reps] = 100
#             produce_or_load(datadir("exp_raw"),
#                             settings,
#                             run_experiment,
#                             suffix="jld2",
#                             loadfile=false,
#                             tag=false)
#         end
#     end

#     n_samples = 10
#     for task ∈ ["sonar", "ionosphere", "breast"]
#         for settings ∈ [Dict(:method=>"MSC_PIMH",
#                              :task  =>task,
#                              :n_samples=>n_samples),

#                         Dict(:method=>"MSC_SIMH",
#                             :task  =>task,
#                             :n_samples=>n_samples),

#                         Dict(:method=>"MSC_CIS",
#                              :task  =>task,
#                              :n_samples=>n_samples),

#                         Dict(:method=>"MSC_CISRB",
#                              :task  =>task,
#                              :n_samples=>n_samples),

#                         Dict(:method=>"SNIS",
#                              :task  =>task,
#                              :n_samples=>n_samples),

#                         Dict(:method=>"ELBO",
#                              :task  =>task,
#                              :n_samples=>n_samples),
#                         ]
#             settings[:n_reps] = 30
#             @info "starting epxeriment" settings=settings
#             produce_or_load(datadir("exp_raw"),
#                             settings,
#                             run_experiment,
#                             suffix="jld2",
#                             loadfile=false,
#                             tag=false)
#         end
#     end

#     n_samples = 1
#     for task ∈ ["sonar", "ionosphere", "breast"]
#         for settings ∈ [Dict(:method=>"ELBO",
#                              :task  =>task,
#                              :n_samples=>n_samples),]
#             settings[:n_reps] = 30
#             @info "starting epxeriment" settings=settings
#             produce_or_load(datadir("exp_raw"),
#                             settings,
#                             run_experiment,
#                             suffix="jld2",
#                             loadfile=false,
#                             tag=false)
#         end
#     end

#     for n_samples ∈ [20, 50]
#     for task ∈ ["sonar"]
#         for settings ∈ [Dict(:method=>"MSC_PIMH",
#                              :task  =>task,
#                              :n_samples=>n_samples),

#                         Dict(:method=>"MSC_SIMH",
#                             :task  =>task,
#                             :n_samples=>n_samples),

#                         Dict(:method=>"MSC_CIS",
#                              :task  =>task,
#                              :n_samples=>n_samples),

#                         Dict(:method=>"MSC_CISRB",
#                              :task  =>task,
#                              :n_samples=>n_samples),

#                         Dict(:method=>"SNIS",
#                              :task  =>task,
#                              :n_samples=>n_samples),
#                         ]
#             settings[:n_reps] = 30
#             @info "starting epxeriment" settings=settings
#             produce_or_load(datadir("exp_raw"),
#                             settings,
#                             run_experiment,
#                             suffix="jld2",
#                             loadfile=false,
#                             tag=false)
#         end
#     end
#     end
# end

function hyperparameter_tuning()
    for method ∈ ["MSC_PIMH"]
        for task ∈ ["boston", "sonar"]
            for defensive ∈ [nothing, 0.001]
                for decay ∈ [true, false]
                    for optimizer ∈ ["NADAM", "ADAM", "RMSProp", "ADAGrad", "Nesterov"]
                        for stepsize ∈ exp10.(range(log10(0.005), log10(0.5); length=15))
                            settings = Dict{Symbol, Any}()

                            @info "starting epxeriment" settings=settings
                            settings[:defensive] = defensive
                            settings[:stepsize]  = stepsize
                            settings[:method]    = method
                            settings[:task]      = task
                            settings[:decay]     = decay
                            settings[:optimizer] = optimizer
                            settings[:n_samples] = 10
                            settings[:n_reps]    = 8
                            settings[:n_iter]    = 10000
                            produce_or_load(datadir("exp_raw"),
                                            settings,
                                            run_experiment,
                                            suffix="jld2",
                                            loadfile=false,
                                            tag=false)
                        end
                    end
                end
            end
        end
    end
end

# function main()
#     seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
#     prng = Random123.Philox4x(UInt64, seed, 8)
#     Random123.set_counter!(prng, seed_key)

#     Random.seed!(seed_key)
#     run_task(prng,
#                  task,
#                  method,
#                  n_samples,
#                  sleep_itvl,
#                  sleep_ϵ,
#                  sleep_L;
#                  show_progress=false)
#     end
#     Dict("result"=>stats, "settings"=>settings)
# end
