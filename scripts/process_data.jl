
using DrWatson
@quickactivate "KLpqVI"

using UnicodePlots
using FileIO
using JLD2
using Statistics
using ProgressMeter

function process_data(ks, stats)::Dict
    mapreduce(merge, ks) do k
        filt_stats   = filter(stat -> k ∈ keys(stat), stats)
        y            = [stat[k] for stat ∈ filt_stats]
        y[isinf.(y)] = 1e+6
        name_y     = string(k) * "_y"
        Dict(name_y => y)
    end
end

function process_batches(batches)
    ks     = keys(batches[1][1])
    @info "Finding x values"
    data_x = map(collect(ks)) do k
        filt_stats = filter(stat -> k ∈ keys(stat), batches[1])
        x          = [stat[:iteration] for stat ∈ filt_stats]
        name       = string(k) * "_x"
        Dict(name => x)
    end

    @info "Finding y values"
    data_batches = map(batches) do stats
        process_data(ks, stats)
    end

    @info "Computing statistics"
    data_y = map(collect(ks)) do k
        name_y = string(k) * "_y"
        ys     = map(data_batches) do batch
            batch[name_y]
        end
        μ      = mean(ys)
        Dict(name_y => μ)
    end
    res = merge(data_x..., data_y...)
    res
end

function main()
    @showprogress map(datadir("exp_raw") |> readdir) do fname
        data_name = fname[1:end-5]
        data      = FileIO.load(datadir("exp_raw", fname))
        result    = data["result"]
        settings  = data["settings"]

        @info "$(fname)" settings
        processed = process_batches(result)

        FileIO.save(datadir("exp_pro", fname), processed)
    end
end
