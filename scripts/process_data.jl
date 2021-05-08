
using DrWatson
@quickactivate "KLpqVI"

using UnicodePlots
using FileIO
using JLD2

function process_data(ks, stats)::Dict
    mapreduce(merge, ks) do k
        filt_stats = filter(stat -> k ∈ keys(stat))
        y          = [stat[k] for filt_stats]
        name       = string(k) * "_y"
        (name_y => y)
    end
end

function process_batch(batches)
    ks     = keys(stats[1])
    @info "Finding x values"
    data_x = mapreduce(merge, ks) do k
        filt_stats = filter(stat -> k ∈ keys(stat))
        x          = [stat[:iteration] for filt_stats]
        name       = string(k) * "_x"
        Dict(name => x)
    end

    @info "Finding y values"
    data_batches = map(batches) do stats
        process_data(ks, stats)
    end

    @info "Computing statistics"
    data_y = map(ks) do k
        ys = map(data_batches) do batch
            batch[k]
        end
        μ      = mean(ys)
        name_y = string(k) * "_x"
        Dict(name_y => μ)
    end
    res = merge(data_x, data_y)
    println(keys(res))
    res
end

function main()
    map(datadir("exp_raw")) do fname
        data_name = fname[1:end-5]
        data      = FileIO.load(fname)
        data      = data["result"]
        settings  = data["settings"]

        @info "$(fname)" settings
        process_data(data)
    end
end
