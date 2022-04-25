
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
using UnicodePlots

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

function process_data(io, df, settings)
    α = 0.8

    #iter        = repeat(1:settings[:n_iter]; inner=settings[:n_reps])
    iter        = repeat(1:settings[:n_iter], settings[:n_reps])
    df[:,:iter] = iter

    cols       = names(df)
    valid_cols = setdiff(cols, ["iter"])
    valid_cols = filter(col -> eltype(df[:,col]) <: Real, valid_cols)
    # @info("Processing Data",
    #       n_iter        = settings[:n_iter],
    #       n_reps        = settings[:n_reps],
    #       optimizer     = settings[:optimizer],
    #       method        = settings[:method],
    #       valid_columns = valid_cols,
    #       )

    for valid_col ∈ valid_cols
        valid_col⁺ = valid_col * "⁺"
        valid_col⁻ = valid_col * "⁻"
        stats      = @chain df begin
            @select(:iter, $valid_col)
            groupby(:iter)
            @combine($valid_col  = median($valid_col),
                     $valid_col⁺ = quantile($valid_col, (1 - α) / 2),
                     $valid_col⁻ = quantile($valid_col, (1 / 2 + α / 2)))
        end
        x  = stats[:,:iter]
        y  = stats[:,valid_col]
        y⁺ = stats[:,valid_col⁺] - stats[:,valid_col]
        y⁻ = stats[:,valid_col]  - stats[:,valid_col⁺]
        write(io, "$(valid_col)_x", x)
        write(io, "$(valid_col)_y", Array(hcat(y, y⁺, y⁻)'))
    end
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
    @showprogress map(datadir("exp_raw") |> readdir) do fname
        data      = FileIO.load(datadir("exp_raw", fname))
        result    = data["result"]
        settings  = data["settings"]

        @info "$(fname)" settings=settings
        h5open(datadir("exp_pro", fname), "w") do io
            process_data(io, result, settings)
        end
    end
end
