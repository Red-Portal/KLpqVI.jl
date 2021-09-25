
using DrWatson
using Distributed
using ProgressMeter
using DataFrames
using CSV
using DataFramesMeta
using Trapz

@everywhere function execute(cmd::Cmd;
                             logfile::Union{String, Nothing}=nothing,
                             switch_stream::Bool=false)
    out = IOBuffer()
    err = IOBuffer()

    code = begin
        if(!isnothing(logfile))
            proc = begin
                if(switch_stream)
                    run(pipeline(ignorestatus(cmd),
                                 stdout=err,
                                 stderr=pipeline(`tee $(logfile)`, out)))
                else
                    run(pipeline(ignorestatus(cmd),
                                 stdout=pipeline(`tee $(logfile)`, out),
                                 stderr=err))
                end
            end
            proc.processes[1].exitcode + proc.processes[2].exitcode
        else
            proc = begin
                if(switch_stream)
                    run(pipeline(ignorestatus(cmd), stdout=err, stderr=out))
                else
                    run(pipeline(ignorestatus(cmd), stdout=out, stderr=err))
                end
            end
            proc.exitcode
        end
    end
    stdout  = @async String(take!(out))
    stderr  = @async String(take!(err))

    return (
        stdout = fetch(stdout),
        stderr = fetch(stderr),
        code   = code
    )
end

@everywhere function run_mcmc(model_name::String,
                              chain_id::Int,
                              β::Real,
                              β_id::Int,
                              num_samples::Int,
                              num_warmup::Int,
                              seed::Int64,
                              ill_conditioned::Bool)
    # sample from model
    #@info("MCMC Sampling from model $(model_name) chain $(chain_id) temp. $(β)")
    cmd_args  = ["method=sample",
                 "algorithm=hmc",
                 "engine=nuts",
                 "max_depth=15",
                 "num_samples=$(num_samples)",
                 "num_warmup=$(num_warmup)"]

    data_base_name = model_name * ".data.R" 
    data_name      = model_name * "_$(β_id).data.R" 
    data           = open(data_base_name, "r") do io
        read(io, String)
    end
    data_with_beta  = data * "\n beta <- $(β)"
    open(data_name, "w") do io
        write(io, data_with_beta)
    end
    
    outfile_name = joinpath("$(model_name)_posterior", "$(model_name)_$(β_id)_$(chain_id).csv")
    cmd          = begin
        if(ill_conditioned)
            `./$(model_name) $cmd_args adapt delta=0.99 data file=$(data_name) random seed=$(seed) output file=$outfile_name`
        else
            `./$(model_name) $cmd_args data file=$(data_name) random seed=$(seed) output file=$outfile_name`
        end
    end 

    out, err, errno = execute(cmd)
    if(errno == 0)
        throw(ErrorException("MCMC Couldn't sample from model $(model_name)\n$(err)"))
    end
end

@everywhere function β_schedule(n_β)
    range(0, 1, length=n_β).^6
end

@everywhere function compuate_summary!(stan_path::String,
                                       summary_name::String,
                                       model_name::String,
                                       n_chains::Int,
                                       β_id::Int)
    samples_name = map(
        chain_id -> joinpath("$(model_name)_posterior",
                             "$(model_name)_$(β_id)_$(chain_id).csv"), 1:n_chains)
    summary_cmd  = joinpath(stan_path, "stansummary")
    execute(`$(summary_cmd) --csv_filename $summary_name $samples_name`)
end

function run_parallel_chains(model_name::String, n_chains::Int, ill_conditioned::BOol)
    n_samples  = 409
    n_warmup   = 2048
    n_β        = 32
    @showprogress pmap(1:n_β) do β_id
        β = β_schedule(n_β)[β_id]
        for i = 1:n_chains
            run_mcmc(model_name, i, β, β_id, n_samples, n_warmup, i, ill_conditioned)
        end
    end
end

function compute_marginal_likelihood(stan_path::String,
                                     model_name::String,
                                     n_chains::Int,
                                     n_β::Int)
    @showprogress pmap(1:n_β) do β_id
        summary_name = "$(model_name)_$(β_id)_summary.csv" 
        if(!isfile(summary_name))
            compuate_summary!(stan_path, summary_name, model_name, n_chains, β_id)
        end
    end

    result = pmap(1:n_β) do β_id
        summary_name = "$(model_name)_$(β_id)_summary.csv" 
        diagnostic   = CSV.File(summary_name,
                                ignoreemptylines=true,
                                comment="#",
        		        threaded=false) |> DataFrames.DataFrame

        @info "$(model_name)_$(β_id)_summary.csv" 

        result = @linq diagnostic |>
            where(:name .== "loglikelihood") |>
            select(:Mean, :MCSE)

        β = β_schedule(n_β)[β_id]
        Dict(:μ    => result.Mean[1],
             :mcse => result.MCSE[1],
             :β    => β)
    end

    β    = [resultᵢ[:β]    for resultᵢ ∈ result[2:end]] # β = 0 contributes too much error
    mcse = [resultᵢ[:mcse] for resultᵢ ∈ result[2:end]]
    println(mcse)
    μ    = [resultᵢ[:μ]    for resultᵢ ∈ result[2:end]]
    ℓZ   = trapz(β, μ) 
   @info("Marginal Likelihood Estimate", ℓZ = ℓZ)
end
