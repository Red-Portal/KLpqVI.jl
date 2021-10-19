
using Distributed

@everywhere using DrWatson
@everywhere @quickactivate "KLpqVI"

@everywhere using ReverseDiff
            using Plots, StatsPlots
@everywhere using Flux
@everywhere using ForwardDiff
@everywhere using Zygote
@everywhere using OnlineStats
@everywhere using Random123
@everywhere using ProgressMeter
@everywhere using DelimitedFiles
@everywhere using FileIO

@everywhere include(srcdir("KLpqVI.jl"))
@everywhere include("task/task.jl")

@everywhere function run_experiment(settings::Dict)
    @unpack method, task, n_samples, n_reps = settings

    task = Val(Symbol(task))
    method, sleep_itvl = begin
        if(method== "MSC_PIMH")
            MSC_PIMH(), 0
        elseif(method== "MSC_SIMH")
            MSC_SIMH(), 0
        elseif(method == "ELBO")
            ELBO(), 0
        elseif(method == "MSC_CIS")
            MSC_CIS(), 0
        elseif(method == "MSC_HMC")
            ϵ, L = hmc_params(task)
            MSC_HMC(ϵ, L), 0
        elseif(method == "MSC_CISRB")
            MSC_CISRB(), 0
        elseif(method == "MSC_CISRB_HMC")
            MSC_CISRB(), 5
        elseif(method == "MSC_PIMH_HMC")
            MSC_CISRB(), 5
        elseif(method == "RWS")
            SNIS(), 5
        elseif(method == "SNIS")
            SNIS(), 0
        end
    end
    sleep_ϵ, sleep_L = 0.1, 16#hmc_params(task)

    stats   = ProgressMeter.@showprogress pmap(1:n_reps) do seed_key
        seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
        prng = Random123.Philox4x(UInt64, seed, 8)
        Random123.set_counter!(prng, seed_key)
        Random.seed!(seed_key)
        run_task(prng,
                 task,
                 method,
                 n_samples,
                 sleep_itvl,
                 sleep_ϵ,
                 sleep_L;
                 show_progress=false)
    end
    Dict("result"=>stats, "settings"=>settings)
end

function general_benchmarks()
    for task ∈ ["pima",
                "heart",
                "german"]
        for settings ∈ [Dict(:method=>"MSC_PIMH",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"MSC_CIS",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"MSC_CISRB",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"MSC_SIMH",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"SNIS",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"ELBO",
                             :task  =>task,
                             :n_samples=>10),
                        ]
            @info "starting epxeriment" settings=settings
            settings[:n_reps] = 100
            produce_or_load(datadir("exp_raw"),
                            settings,
                            run_experiment,
                            suffix="jld2",
                            loadfile=false,
                            tag=false)
        end
    end

    n_samples = 10
    for task ∈ ["radon"]
        for settings ∈ [# Dict(:method=>"MSC_PIMH",
                        #      :task  =>task,
                        #      :n_samples=>n_samples),

                        # Dict(:method=>"MSC_SIMH",
                        #     :task  =>task,
                        #     :n_samples=>n_samples),

                        # Dict(:method=>"MSC_CIS",
                        #      :task  =>task,
                        #      :n_samples=>n_samples),

                        # Dict(:method=>"MSC_CISRB",
                        #      :task  =>task,
                        #      :n_samples=>n_samples),

                        # Dict(:method=>"SNIS",
                        #      :task  =>task,
                        #      :n_samples=>n_samples),

                        # Dict(:method=>"RWS",
                        #      :task  =>task,
                        #      :n_samples=>n_samples),

                        # Dict(:method=>"ELBO",
                        #      :task  =>task,
                        #      :n_samples=>n_samples),

                        # Dict(:method=>"MSC_HMC",
                        #      :task  =>task,
                        #      :n_samples=>n_samples),

                        ]
            settings[:n_reps] = 100
            @info "starting epxeriment" settings=settings
            produce_or_load(datadir("exp_raw"),
                            settings,
                            run_experiment,
                            suffix="jld2",
                            loadfile=false,
                            tag=false)
        end
    end

    n_samples = 10
    for task ∈ ["sonar", "ionosphere", "breast"]
        for settings ∈ [Dict(:method=>"MSC_PIMH",
                             :task  =>task,
                             :n_samples=>n_samples),

                        Dict(:method=>"MSC_SIMH",
                            :task  =>task,
                            :n_samples=>n_samples),

                        Dict(:method=>"MSC_CIS",
                             :task  =>task,
                             :n_samples=>n_samples),

                        Dict(:method=>"MSC_CISRB",
                             :task  =>task,
                             :n_samples=>n_samples),

                        Dict(:method=>"SNIS",
                             :task  =>task,
                             :n_samples=>n_samples),

                        Dict(:method=>"ELBO",
                             :task  =>task,
                             :n_samples=>n_samples),
                        ]
            settings[:n_reps] = 30
            @info "starting epxeriment" settings=settings
            produce_or_load(datadir("exp_raw"),
                            settings,
                            run_experiment,
                            suffix="jld2",
                            loadfile=false,
                            tag=false)
        end
    end

    n_samples = 1
    for task ∈ ["sonar", "ionosphere", "breast"]
        for settings ∈ [Dict(:method=>"ELBO",
                             :task  =>task,
                             :n_samples=>n_samples),]
            settings[:n_reps] = 30
            @info "starting epxeriment" settings=settings
            produce_or_load(datadir("exp_raw"),
                            settings,
                            run_experiment,
                            suffix="jld2",
                            loadfile=false,
                            tag=false)
        end
    end

    for n_samples ∈ [20, 50]
    for task ∈ ["sonar"]
        for settings ∈ [Dict(:method=>"MSC_PIMH",
                             :task  =>task,
                             :n_samples=>n_samples),

                        Dict(:method=>"MSC_SIMH",
                            :task  =>task,
                            :n_samples=>n_samples),

                        Dict(:method=>"MSC_CIS",
                             :task  =>task,
                             :n_samples=>n_samples),

                        Dict(:method=>"MSC_CISRB",
                             :task  =>task,
                             :n_samples=>n_samples),

                        Dict(:method=>"SNIS",
                             :task  =>task,
                             :n_samples=>n_samples),
                        ]
            settings[:n_reps] = 30
            @info "starting epxeriment" settings=settings
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

function gaussian_benchmarks()
    for task ∈ ["gaussian_correlated", "gaussian"]
        for n_samples ∈ [4, 16, 64]
            for settings ∈ [Dict(:method=>"MSC_PIMH",
                                 :task  =>task,
                                 :n_samples=>n_samples),

                            Dict(:method=>"MSC_CIS",
                                 :task  =>task,
                                 :n_samples=>n_samples),

                            Dict(:method=>"MSC_CISRB",
                                 :task  =>task,
                                 :n_samples=>n_samples),

                            Dict(:method=>"MSC_HMC",
                                 :task  =>task,
                                 :n_samples=>n_samples),

                            Dict(:method=>"RWS",
                                 :task  =>task,
                                 :n_samples=>n_samples),

                            Dict(:method=>"SNIS",
                                 :task  =>task,
                                 :n_samples=>n_samples),
                            ]
                @info "starting epxeriment" settings=settings

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
