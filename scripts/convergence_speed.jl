
using DrWatson
@quickactivate "KLpqVI"

include(srcdir("KLpqVI.jl"))
include("task/lda.jl")
include("task/gaussian.jl")

using DelimitedFiles
using Plots, StatsPlots
using Flux
using ForwardDiff
using OnlineStats
using Random123
using ProgressMeter

function run_vi(seed_int, method, n_dims, n_mc)
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    Random123.set_counter!(prng, seed_int)

    AdvancedVI.setadbackend(:forwarddiff)

    n_data = 200
    p      = gaussian_data(prng, n_dims, correlated=false)
    z      = rand(p, n_data);
    model  = gaussian(z, n_dims)

    crossent = []
    paretok  = []
    function plot_callback(logπ, q, objective, klpq)
        zs   = [rand(prng, q) for i = 1:128]
        ℓw    = logπ.(zs) - logpdf.(Ref(q), zs)
        _, k = psis.psislw(ℓw)

        push!(crossent, klpq)
        push!(paretok, min(k, 15))
    end

    n_iter = 5000
    θ, q = vi(model;
              objective = method,
              n_mc      = n_mc,
              n_iter    = n_iter,
              tol       = 0.0005,
              callback  = plot_callback,
              rng       = prng,
              optimizer = AdvancedVI.TruncatedADAGrad(),
              show_progress = true,
              )
    crossent, paretok
end

function run_experiment(settings::Dict)
    @unpack method, n_dims, n_samples = settings

    seed   = 0
    method = if(method== "MSC_HMC")
        if(n_dims == 1)
            MSC_HMC(0.03, 16)
        elseif(n_dims == 10)
            MSC_HMC(0.02, 16)
        elseif(n_dims == 20)
            MSC_HMC(0.01, 16)
        end
    elseif(method == "MSC")
        MSC()
    elseif(method == "KLPQSNIS")
        KLPQSNIS()
    end
    fname        = savename(settings)
    @info "starting epxeriment" settings...

    cent, pareto = run_vi(seed, method, n_dims,  n_samples)
    open(datadir("convergence", fname*".txt"), "w") do io
        write(io, "crossentropy,paretok\n")
        writedlm(io, hcat(cent, pareto), ',')
    end
end

function main()
    for n_dims ∈ [1, 10, 20]
        for settings ∈ [Dict(:method=>"MSC_HMC",
                             :n_dims=>n_dims,
                             :n_samples=>8),

                        Dict(:method=>"KLPQSNIS",
                             :n_dims=>n_dims,
                             :n_samples=>8),

                        Dict(:method=>"MSC",
                             :n_dims=>n_dims,
                             :n_samples=>1),

                        Dict(:method=>"MSC",
                             :n_dims=>n_dims,
                             :n_samples=>8),

                        Dict(:method=>"MSC",
                             :n_dims=>n_dims,
                             :n_samples=>32),
                        ]
            run_experiment(settings)
        end
    end
end
