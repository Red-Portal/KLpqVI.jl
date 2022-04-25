
            using Distributed
@everywhere using DrWatson
@everywhere @quickactivate "KLpqVI"

@everywhere using FileIO
@everywhere using JLD2
@everywhere using Statistics
@everywhere using CSVFiles
            using ProgressMeter
@everywhere using DataFrames
@everywhere using Bootstrap

using HDF5
using DataFramesMeta
using Printf
using PrettyTables

@everywhere function process_data(ks, stats)::Dict
    mapreduce(merge, ks) do k
        filt_stats = filter(stat -> k ∈ keys(stat), stats)
        y          = [stat[k]          for stat ∈ filt_stats]
        x          = [stat[:iteration] for stat ∈ filt_stats]
        t          = [stat[:elapsed]   for stat ∈ filt_stats]

        y[isnan.(y)] .= typemax(eltype(y))
        if(k == :paretok)
            y[isinf.(y)] .= 1e+6
        end

        if(length(y) > 1000)
            y = y[1:10:end]
            x = x[1:10:end]
            t = t[1:10:end]
        end
        name_y = string(k) * "_y"
        name_x = string(k) * "_x"
        name_t = string(k) * "_t"
        Dict(name_y => y, name_x => x, name_t => t)
    end
end

@everywhere function process_batches(batches)
    allkeys = vcat(collect.(keys.(batches[1]))...)
    ks      = collect(Set(allkeys))

    @info "Finding y values"
    data_batches = map(batches) do stats
        process_data(ks, stats)
    end

    @info "Computing statistics"
    mapreduce(merge, ks) do k
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

        CIs = map(1:length(ys[1])) do  t
            boot = bootstrap(mean, [ys[i][t] for i = 1:length(ys)], AntitheticSampling(1024))
            confint(boot, PercentileConfInt(0.8))
        end

        μ_y    = [CI[1][1] for CI in CIs]
        e_l    = [CI[1][2] for CI in CIs]
        e_h    = [CI[1][3] for CI in CIs]
            #[median(  [ys[i][t] for i = 1:length(ys)])      for t = 1:length(ys[1])]
        #e_l    = #[quantile([ys[i][t] for i = 1:length(ys)], 0.2) for t = 1:length(ys[1])]
        #e_h    = #[quantile([ys[i][t] for i = 1:length(ys)], 0.8) for t = 1:length(ys[1])]

        y_stat = Array(hcat(μ_y, e_h - μ_y, e_l - μ_y)')

        Dict(name_t   => μ_t,
             name_x   => xs[1],
             name_y   => y_stat,
             )
        # for i = 1:length(ys)
        #     insertcols!(summ, Symbol("y$(i)") => ys[i])
        # end
        #name = string(k)*".csv"
        #FileIO.save(joinpath(path, name), summ)
    end
end

function convergence_diagnostic()
    data         = Dict()
    data[:CIS]   = FileIO.load(datadir("exp_raw", "method=MSC_CIS_n_samples=16_task=gaussian.jld2")) 
    data[:CISRB] = FileIO.load(datadir("exp_raw", "method=MSC_CISRB_n_samples=16_task=gaussian.jld2")) 
    data[:PIMH]  = FileIO.load(datadir("exp_raw", "method=MSC_PIMH_n_samples=16_task=gaussian.jld2")) 

    p    = load_dataset(Val(:gaussian))
    ∫pℓp = entropy(p)

    result             = Dict()
    result["cis"]       = [stat[:kl] for stat ∈ data[:CIS]["result"][1]]         .+ ∫pℓp
    result["cisrb"]     = [stat[:kl] for stat ∈ data[:CISRB]["result"][1]]       .+ ∫pℓp
    result["pimh"]      = [stat[:kl] for stat ∈ data[:PIMH]["result"][1]]        .+ ∫pℓp

    result["pimh_est"]  = [stat[:crossent] for stat ∈ data[:PIMH]["result"][1]]  .+ ∫pℓp
    result["cis_est"]   = [stat[:crossent] for stat ∈ data[:CIS]["result"][1]] .+ ∫pℓp
    result["cisrb_est"] = [stat[:crossent] for stat ∈ data[:CISRB]["result"][1]]   .+ ∫pℓp
    FileIO.save(datadir("exp_pro", "convergence_diagnostic.jld2"), result)
end

function draw_table_logistic()
    entries = []
    for (method, name) ∈ [("ELBO",      "ELBO"),
                          ("MSC_PIMH",  "par.-IMH"),
                          ("MSC_SIMH",  "seq.-IMH"),
                          ("MSC_CIS",   "single-CIS"),
                          ("MSC_CISRB", "single-CISRB"),
                          ("MSC_HMC",   "single-HMC"),
                          ("SNIS",      "SNIS"),
                          ]
        row_entries = String[]
        push!(row_entries, name)

        for (i, problem) ∈ enumerate(["pima", "heart", "german"])
            data = FileIO.load(datadir("exp_pro", "method=$(method)_n_reps=100_n_samples=10_task=$(problem).jld2"))
            acc_y, acc_Δ₊, acc_Δ₋ = data["acc_y"][:,end]
            acc_y₊ = acc_y + acc_Δ₊ 
            acc_y₋ = acc_y + acc_Δ₋
            push!(row_entries, Printf.@sprintf("%.2f {\\scriptsize(%.2f, %.2f)}", acc_y, acc_y₋, acc_y₊))

            ll_y, ll_Δ₊, ll_Δ₋ = data["ll_y"][:,end]
            ll_y₊ = ll_y + ll_Δ₊ 
            ll_y₋ = ll_y + ll_Δ₋
            push!(row_entries, Printf.@sprintf("%.2f {\\scriptsize(%.2f, %.2f)}", ll_y, ll_y₋, ll_y₊))
        end
        push!(entries, row_entries)
    end
    table = permutedims(hcat(entries...))
    display(PrettyTables.pretty_table(table;
                                      backend = Val(:latex),
                                      tf = PrettyTables.tf_latex_booktabs,
                                      header=(["",
                                               "pima acc",  "pima ll",
                                               "heart_acc", "heart_ll",
                                               "german_acc", "german_ll"],
                                              ["",
                                               "acc", "l",
                                               "acc", "ll",
                                               "acc", "ll"])))
end

function draw_table_gp()
    entries = []
    for (method, name, N) ∈ [("ELBO",      "ELBO", 1),
                             ("MSC_PIMH",  "par.-IMH", 10),
                             ("MSC_SIMH",  "seq.-IMH", 10),
                             ("MSC_CIS",   "single-CIS", 10),
                             ("MSC_CISRB", "single-CISRB", 10),
                             ("SNIS",      "SNIS", 10),
                             ]
        row_entries = String[]
        push!(row_entries, name)

        for (i, problem) ∈ enumerate(["sonar", "ionosphere", "breast"])
            data = FileIO.load(datadir("exp_pro", "method=$(method)_n_reps=30_n_samples=$(N)_task=$(problem).jld2"))
            acc_y, acc_Δ₊, acc_Δ₋ = data["acc_y"][:,end]
            acc_y₊ = acc_y + acc_Δ₊ 
            acc_y₋ = acc_y + acc_Δ₋
            push!(row_entries, Printf.@sprintf("%.2f {\\scriptsize(%.2f, %.2f)}", acc_y, acc_y₋, acc_y₊))

            ll_y, ll_Δ₊, ll_Δ₋ = data["nlpd_y"][:,end]
            ll_y₊ = ll_y + ll_Δ₊ 
            ll_y₋ = ll_y + ll_Δ₋
            push!(row_entries, Printf.@sprintf("%.2f {\\scriptsize(%.2f, %.2f)}", ll_y, ll_y₋, ll_y₊))
        end
        push!(entries, row_entries)
    end
    table = permutedims(hcat(entries...))
    display(PrettyTables.pretty_table(table;
                                      backend = Val(:latex),
                                      tf = PrettyTables.tf_latex_booktabs,
                                      header=(["",
                                               "sonar_acc",  "sonar_ll",
                                               "ionosphere_acc", "ionosphere_ll",
                                               "breast_acc", "breast_ll"],
                                              ["",
                                               "acc", "ll",
                                               "acc", "ll",
                                               "acc", "ll"])))
end

function stepsize_plot()
    defensive = 0.0#01
    decay     = false
    ν         = 500
    data      = load(datadir("exp_pro", "stepsize_nu=$(ν).jld2"), "data")
    α         = 0.8


    h5open(datadir("exp_pro/stepsize_data.h5"), "w") do io
        for optimizer ∈ ["ADAM", "Nesterov", "Momentum", "RMSProp", "SGD"]
            for method ∈ ["MSC_PIMH", "MSC_SIMH", "MSC_CIS"]
                df_subset = @chain data begin
                    @subset((:decay     .== decay)     .&
                            (:defensive .== defensive) .& 
                            (:optimizer .== optimizer) .&
                            (:method    .== method))
                    @select(:kl, :stepsize)
                    groupby(:stepsize)
                    @combine(:kl = median(:kl), :kl⁻ = quantile(:kl, (1 - α) / 2), :kl⁺ = quantile(:kl, (1 / 2 + α / 2)))
                end
                @info("", optimizer, method)
                Δkl⁺ = df_subset[:, :kl⁺] - df_subset[:, :kl]
                Δkl⁻ = df_subset[:, :kl]  - df_subset[:, :kl⁻]
                write(io, "$(method)_$(optimizer)_kl", Array(hcat(df_subset[:, :kl], Δkl⁺, Δkl⁻)'))
                if (optimizer == "ADAM") && (method == "MSC_PIMH")
                    write(io, "stepsize", df_subset[:,:stepsize])
                end
            end
        end
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

        #outpath   = datadir("exp_pro", task, method, n_samples)
        @info "$(fname)" settings=settings
        summary = process_batches(result)
        
        FileIO.save(datadir("exp_pro", fname), summary)
    end
end
