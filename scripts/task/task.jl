
import JLD
import DelimitedFiles

load_dataset(task::Symbol)     = fetch_dataset(Val(task))
sample_posterior(task::Symbol) = sample_posterior(Val(task))

include("gaussian.jl")
include("logistic.jl")
