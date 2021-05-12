
import JLD
import MAT
import DelimitedFiles
import Random
import PyCall
import AdvancedHMC

using SparseArrays

include("gaussian.jl")
include("logistic.jl")
#include("gmm.jl")
include("eightschools.jl")
include("sv.jl")
include("lda.jl")
include("lda_meanfield.jl")
