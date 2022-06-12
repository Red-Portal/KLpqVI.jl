
# KLpqVI.jl

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> Markov Chain Score Ascent: A Unifying Framework of Variational Inference with Markovian Gradients

## Project Structure
| Path  | Description  |
|:--|:--|
| scripts/ | Scripts for executing the experiments  |
| scripts/task | Code specific to each benchmark problems  |
| src/ | Code for our implemented algorithms |
| stan/ | Code for estimating the marginal likelihoods using Stan |


## Installation
To (locally) reproduce this project, do the following:

0. Download this code base. Notice datasets are not included and need to be downloaded independently from the [UCI repository](https://archive.ics.uci.edu/ml/datasets.php).
Please refer to the `load_dataset(task)` function in `scripts/task/datasets.jl` to see how to set up the datasets.

1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```
This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.

2. To reproduce our benchmarks, run the script:
   ```
   julia> include("scripts/run_benchmarks.jl")
   julia> general_benchmarks()
   ```
