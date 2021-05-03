
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
@everywhere using JLD

@everywhere include(srcdir("KLpqVI.jl"))
@everywhere include("task/task.jl")

@everywhere function run_experiment(settings::Dict)
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
    stats   = ProgressMeter.@showprogress pmap(1:n_iters) do seed_key
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
    Dict("result" => stats)
end

function general_benchmarks()
    for task ∈ ["pima",
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
            fname = savename(settings, "jld")
            if(isfile(datadir("exp_raw", fname)))
                @info "Skipping $fname" settings=settings 
                continue
            else
                res   = run_experiment(settings)
                JLD.save(datadir("exp_raw", fname), res...)
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
                @info "starting epxeriment" settings=settings

                fname = savename(settings, "jld")
                if(isfile(datadir("exp_raw", fname)))
                    @info "Skipping $fname"
                    continue
                else
                    res   = run_experiment(settings)
                    JLD.save(datadir("exp_raw", fname), res...)
                end
            end
        end
    end
end
