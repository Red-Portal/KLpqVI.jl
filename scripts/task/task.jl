
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

include("gaussian.jl")
include("logistic.jl")
include("eightschools.jl")
include("radon.jl")
include("bgp.jl")
include("bnn.jl")
