
#using Flux
using Distributions
using StatsBase
using StatsFuns
using Zygote
using LinearAlgebra
using ProgressMeter

import Base.Iterators
import SpecialFunctions
import Dates
import PDMats
import AdvancedVI
import Bijectors
import DiffResults
import DistributionsAD
import DynamicPPL
import ForwardDiff
import Random
import RDatasets
import Turing

import HDF5
import KernelFunctions
import AbstractGPs

#@eval Zygote begin
#end

# function DynamicPPL.dot_observe(
#     spl::Union{DynamicPPL.rior,
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
include("imh.jl")
include("hmc.jl")
include("gradient.jl")
include("vi.jl")
include("snis.jl")
include("elbo.jl")
include("advi.jl")
include("msc.jl")
include("msc_rb.jl")
include("pmcsa.jl")
include("jsa.jl")
include("diagnostics.jl")
