
# KLpqVI.jl

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
>  Inclusive Variational Inference with Independent Metropolis­Hastings

## Project Structure
| Path  | Description  |
|:--|:--|
| scripts/ | Scripts for executing the experiments  |
| scripts/task | Code specific to each benchmark problems  |
| src/ | Code for our implemented algorithms (MSC, RWS, SNIS, etc...) |
| stan/ | Code for estimating the marginal likelihoods using Stan |


## Installation
To (locally) reproduce this project, do the following:

0. Download this code base. Notice datasets are not included and need to be downloaded independently.
Please refer to the `load_dataset(task)` function in the respective task in `scripts/task` to see how to set up the datasets.
The datasets used in this work are as follows: 
* [German credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
* [Pima indians diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
* [Heart disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)
* [Radon](https://github.com/stan-dev/example-models/blob/master/ARM/Ch.19/radon.data.R)
* [Stochastic volatility](https://github.com/TuringLang/TuringExamples/blob/master/benchmarks/sto_volatility/data.csv)

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
