
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
using ThermodynamicIntegration
using Suppressor

include(srcdir("KLpqVI.jl"))
include("task/task.jl")

# @eval ThermodynamicIntegration begin
#     function sample_powerlogπ(powerlogπ, alg::ThermInt, x_init)
#         D = length(x_init)
#         metric = DiagEuclideanMetric(D)
#         hamiltonian = get_hamiltonian(metric, powerlogπ, alg)

#         initial_ϵ = find_good_stepsize(hamiltonian, x_init)
#         integrator = Leapfrog(initial_ϵ)

#         proposal = AdvancedHMC.NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
#         adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

#         samples, stats = sample(
#             alg.rng,
#             hamiltonian,
#             proposal,
#             x_init,
#             alg.n_samples,
#             adaptor,
#             alg.n_warmup;
#             verbose=false,
#             progress=true,
#         )
#         return samples
#     end
# end

function marginal_likelihood()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    Random123.set_counter!(prng, 0)
    Random.seed!(0)

    ThermodynamicIntegration.set_adbackend(:ForwardDiff) 
    Turing.Core.setrdcache(true)
    Turing.Core._setadbackend(Val(:reversediff))

    n_burn    = 2048
    n_samples = 2048
    n_steps   = 32
    alg       = ThermodynamicIntegration.ThermInt(
        prng, ((1:n_steps) ./ n_steps) .^ 5;
        n_samples=n_samples,
        n_warmup=n_burn)

    y      = load_dataset(Val(:sv))
    model  = stochastic_volatility(y)
    logZ   = @suppress begin
        alg(model, TIParallelThreads())
    end
    @info "sv" logZ = logZ

    ThermodynamicIntegration.set_adbackend(:ReverseDiff) 

    mat, N = load_data(Val(:neuron))
    model  = neuron(mat, N, 3)
    logZ   = @suppress begin
        alg(model, TIParallelThreads())
    end
    @info "neuron" logZ = logZ
end
