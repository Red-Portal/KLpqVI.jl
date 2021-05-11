
            using Distributed
@everywhere using DrWatson
@everywhere @quickactivate "KLpqVI"

@everywhere using FileIO
@everywhere using JLD2
@everywhere using Statistics
@everywhere using CSVFiles
            using ProgressMeter
@everywhere using DataFrames

@everywhere function process_data(ks, stats)::Dict
    mapreduce(merge, ks) do k
        filt_stats    = filter(stat -> k ∈ keys(stat), stats)
        y             = [stat[k] for stat ∈ filt_stats]
        x             = [stat[k] for stat ∈ filt_stats]
        t             = [stat[:elapsed] for stat ∈ filt_stats]
        if(k == :paretok)
            y[isinf.(y)] .= 1e+6
        end
        if(length(y) > 1000)
            y = y[1:10:end]
            x = x[1:10:end]
            t = t[1:10:end]
        end
        name_y     = string(k) * "_y"
        name_x     = string(k) * "_x"
        name_t     = string(k) * "_t"
        Dict(name_y => y, name_x => x, name_t => t)
    end
end

@everywhere function process_batches(batches, path)
    allkeys = vcat(collect.(keys.(batches[1]))...)
    ks      = collect(Set(allkeys))

    @info "Finding y values"
    data_batches = map(batches) do stats
        process_data(ks, stats)
    end

    @info "Computing statistics"
    for k ∈ ks
        name_y = string(k) * "_y"
        name_t = string(k) * "_t"
        name_x = string(k) * "_x"

        ts = map(data_batches) do batch
            batch[name_t]
        end
        xs = map(data_batches) do batch
            batch[name_x]
        end
        ys = map(data_batches) do batch
            batch[name_y]
        end

        name_μ = string(k) * "_mean"
        μ_t    = mean(ts)
        μ_y    = mean(ys)
        summ   = DataFrame("t" => ts[1],
                           "x" => xs[1],
                           "mean_t" => μ_t,
                           "mean_y" => μ_y,
                         )
        for i = 1:length(ys)
            insertcols!(summ, Symbol("y$(i)") => ys[i])
        end
        name = string(k)*".csv"
        FileIO.save(joinpath(path, name), summ)
    end
end

function main()
    @showprogress pmap(datadir("exp_raw") |> readdir) do fname
        data_name = fname[1:end-5]
        data      = FileIO.load(datadir("exp_raw", fname))
        result    = data["result"]
        settings  = data["settings"]
        task      = string(settings[:task])
        method    = string(settings[:method])
        n_samples = string(settings[:n_samples])
        outpath   = datadir("exp_pro", task, method, n_samples)

        @info "$(fname)" settings=merge(
            settings, Dict(:path=>outpath))
        process_batches(result, outpath)
    end
end
