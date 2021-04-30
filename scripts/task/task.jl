
import JLD
import MAT
import DelimitedFiles
import MLDataUtils
import Random
import PyCall

using SparseArrays

load_dataset(prng::Random.AbstractRNG, task::Symbol) = load_dataset(Val(task))
sample_posterior(task::Symbol) = sample_posterior(Val(task))

include("gaussian.jl")
include("logistic.jl")
#include("gmm.jl")
include("sv.jl")
include("lda.jl")
