
using DrWatson
@quickactivate "KLpqVI"

using UnicodePlots
using FileIO
using JLD2
using Statistics
using ProgressMeter

function process_data(ks, stats)::Dict
    mapreduce(merge, ks) do k
        filt_stats = filter(stat -> k ∈ keys(stat), stats)
        y          = [stat[k]          for stat ∈ filt_stats]
        x          = [stat[:iteration] for stat ∈ filt_stats]
        t          = [stat[:elapsed]   for stat ∈ filt_stats]
        if(k == :paretok)
            y[isinf.(y)] .= 1e+6
        end
        name_y     = string(k) * "_y"
        name_x     = string(k) * "_x"
        name_t     = string(k) * "_t"
        Dict(name_y => y, name_x => x, name_t => t)
    end
end

function process_batches(batches, path)
    allkeys = vcat(keys.(batches[1]))
    ks      = collect(Set(allkeys))

    @info "Finding y values"
    data_batches = map(batches) do stats
        process_data(ks, stats)
    end

    @info "Computing statistics"
    for k ∈ collect(ks)
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
        μtxμy  = hcat(μ_t, xs[1], μ_y)
        FileIO.save(joinpath(path, name_μ*".csv"), μtxμy)

        for i = 1:length(ys)
            name = string(k)*string(i)*".csv"
            txy  = hcat(ts[i], xs[i], ys[i])
            FileIO.save(joinpath(path, name), txy)
        end
    end
end

function main()
    @showprogress map(datadir("exp_raw") |> readdir) do fname
        data_name = fname[1:end-5]
        data      = FileIO.load(datadir("exp_raw", fname))
        result    = data["result"]
        settings  = data["settings"]
        task      = string(settings[:task])
        method    = string(settings[:method])
        outpath   = datadir("exp_pro", task, method)

        @info "$(fname)" settings=merge(
            settings, Dict(:path=>outpath))
        process_batches(result, outpath)
    end
end
