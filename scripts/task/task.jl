
using ParserCombinator
using SparseArrays
using JLD2

import FileIO
import Distances
import JLD
import MAT
import DelimitedFiles
import Random
import PyCall
import AdvancedHMC
import GraphIO
import LightGraphs
import SimpleWeightedGraphs

include("gaussian.jl")
include("logistic.jl")
#include("gmm.jl")
include("eightschools.jl")
include("sv.jl")
include("lda.jl")
include("lda_meanfield.jl")
include("neuron.jl")
include("sunspot.jl")
include("colon.jl")
include("radon.jl")
