
using DrWatson
@quickactivate "KLpqVI"

using Memoization
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

function find_hmc_params()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    Random123.set_counter!(prng, 0)
    Random.seed!(0)

    Turing.Core.setrdcache(true)
    #AdvancedVI.setadbackend(:zygote)
    #Turing.Core._setadbackend(Val(:zygote))
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))
    #AdvancedVI.setadbackend(:forwarddiff)
    #Turing.Core._setadbackend(Val(:forwarddiff))

    #sample_posterior(prng, Val(:schools))

    #sample_posterior(prng, Val(:pima))
    #sample_posterior(prng, Val(:heart))
    #sample_posterior(prng, Val(:ionosphere))
    #sample_posterior(prng, Val(:german))
    #sample_posterior(prng, Val(:sonar))
    sample_posterior(prng, Val(:radon))

    #AdvancedVI.setadbackend(:zygote)
    #Turing.Core._setadbackend(Val(:zygote))
    #sample_posterior(prng, Val(:colon))
    #sample_posterior(prng, Val(:radon))

    #AdvancedVI.setadbackend(:zygote)
    #Turing.Core._setadbackend(Val(:zygote))

    #sample_posterior(prng, Val(:gaussian_correlated))
    #sample_posterior(prng, Val(:gaussian))

    #sample_posterior_full(prng, Val(:colon))
end
