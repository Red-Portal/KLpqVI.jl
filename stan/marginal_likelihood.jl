
using Distributed

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

function run_mcmc(model_name::String,
                  chain_id::Int,
                  β::Real,
                  num_samples::Int,
                  num_warmup::Int,
                  seed::Int64)
    # sample from model
    @info("MCMC Sampling from model $(model_name)")
    cmd_args  = ["method=sample",
                 "algorithm=hmc",
                 "engine=nuts",
                 "max_depth=10",
                 "num_samples=$(num_samples)",
                 "num_warmup=$(num_warmup)"]
    β = 0.01

    data_name    = model_name * ".data.R" 
    outfile_name = model_name * "_$(β)_$(chain_id).csv"
    cmd          = `./$(model_name) $cmd_args data file=$(data_name) random seed=$(seed) output file=$outfile_name`

    out, err, errno = execute(cmd)
    if(errno == 0)
        @info("MCMC Successfully sampled from model $(model_name)")
    else
        throw(ErrorException("MCMC Couldn't sample from model $(model_name)\n$(err)"))
    end
end

function run_parallel_chains(model_name::String, n_chains::Int)
    n_samples  = 4096
    n_warmup   = 2048
    β_schedule = []
    pmap(1:n_chains, β_schedule) do i, β
        run_mcmc(model_name, i, β, n_samples, n_warmup, i)
    end
    #@info("MCMC Diagnosing samples from model $(model_name)")
    #summaryfile_name = "$(model_name)_$(β)_diagnostic.csv"
    #execute(`$summary_path --input_files $outfile_name --csv_filename $summaryfile_name`)
    #nothing
end
