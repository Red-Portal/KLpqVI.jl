
using DrWatson
@quickactivate "KLpqVI"

using ReverseDiff
using Plots, StatsPlots
using Flux
using ForwardDiff
using Zygote
using OnlineStats
using Random123
using ProgressMeter
using DelimitedFiles

include(srcdir("KLpqVI.jl"))
include("task/task.jl")


function run_experiment(settings::Dict)
    @unpack method, task, n_samples = settings

    task = Val(Symbol(task))
    method, sleep_itvl = begin
        if(method== "MSC_PIMH")
            MSC_PIMH(), 0
        elseif(method == "ELBO")
            ELBO(), 0
        elseif(method == "MSC_CIS")
            MSC_CIS(), 0
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
    sleep_ϵ, sleep_L = hmc_params(task)

    n_iters = 100
    stats   = map(1:n_iters) do seed_key
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
                 sleep_L)
    end
    Dict("result" => stats)
end

function general_benchmarks()
    for task ∈ ["gaussian_correlated",
                "pima",
                "heart",
                "ionosphere",
                "sv",]
        for settings ∈ [Dict(:method=>"MSC_PIMH",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"MSC_CIS",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"MSC_CISRB",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"MSC_CISRB_HMC",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"MSC_PIMH_HMC",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"RWS",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"SNIS",
                             :task  =>task,
                             :n_samples=>10),

                        Dict(:method=>"ELBO",
                             :task  =>task,
                             :n_samples=>10),
                        ]

            @info "starting epxeriment" settings...

            produce_or_load(datadir("exp_raw"),
                            settings,
                            run_experiment,
                            suffix="jld",
                            loadfile=false)
        end
    end
end

function gaussian_benchmarks()
    task = "gaussian_correlated"
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

                        Dict(:method=>"MSC_CISRB_HMC",
                             :task  =>task,
                             :n_samples=>n_samples),

                        Dict(:method=>"MSC_PIMH_HMC",
                             :task  =>task,
                             :n_samples=>n_samples),

                        Dict(:method=>"RWS",
                             :task  =>task,
                             :n_samples=>n_samples),

                        Dict(:method=>"SNIS",
                             :task  =>task,
                             :n_samples=>n_samples),
                        ]
            @info "starting epxeriment" settings...

            produce_or_load(datadir("exp_raw"),
                            settings,
                            run_experiment,
                            suffix="jld",
                            loadfile=false)
        end
    end
end
