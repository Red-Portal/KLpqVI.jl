
#using Flux
using Distributions
using StatsBase
using StatsFuns
using Zygote
using LinearAlgebra

import SpecialFunctions
import Tracker
import Dates
import PDMats
import AdvancedVI
import Bijectors
import DiffResults
import DistributionsAD
import DynamicPPL
import ForwardDiff
import PyCall
import Random
import Turing

#@eval Zygote begin
#end

# function DynamicPPL.dot_observe(
#     spl::Union{DynamicPPL.SampleFromPrior,
#                DynamicPPL.SampleFromUniform},
#     dists::AbstractArray{<:Distribution},
#     value::AbstractArray,
#     vi,
# )
#     #return sum(Distributions.loglikelihood.(dists, value))
#     return mapreduce(Distributions.loglikelihood, +, dists, value)
# end

include("utils.jl")
include("cis.jl")
include("hmc.jl")
include("gradient.jl")
include("vi.jl")
include("snis.jl")
include("elbo.jl")
include("advi.jl")
include("msc_cis.jl")
include("msc_cisrb.jl")
include("msc_parimh.jl")
include("rws.jl")
include("psis.jl")
include("lda_svi.jl")
include("diagnostics.jl")
