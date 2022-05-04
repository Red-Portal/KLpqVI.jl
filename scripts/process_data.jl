
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

function compress_data()
    files = filter(fname -> occursin("jld2", fname), readdir(datadir("exp_raw"), join=true))
    ProgressMeter.@showprogress for file ∈ files
        data       = JLD2.load(file)
        df         = data["result"]
        cols       = names(df)
        cols_valid = intersect(cols, ["rmse", "iter", "lpd", "elapsed", "acc", "elbo"])
        df         = @select(df, $(cols_valid))
        data["result"] = df
        JLD2.save(file, data)
    end
end

function process_data(io, df, settings)
    α = 0.95

    #iter        = repeat(1:settings[:n_iter]; inner=settings[:n_reps])
    iter        = repeat(1:settings[:n_iter], settings[:n_reps])
    df[:,:iter] = iter

    method     = settings[:method]
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

    function apply_bootstrap(arr)
        boot = bootstrap(mean, arr, AntitheticSampling(245))
        ci   = confint(boot, NormalConfInt(α))
        (y = ci[1][1], y⁺⁻ = ci[1][3])#, y⁻ = ci[1][2])
    end

    for valid_col ∈ valid_cols
        stats      = @chain df begin
            @select(:iter, $valid_col)
            groupby(:iter)
            # @combine(y  = median($valid_col),
            #          y⁺ = quantile($valid_col, (1 - α) / 2),
            #          y⁻ = quantile($valid_col, (1 / 2 + α / 2)))
            @combine($AsTable = apply_bootstrap($valid_col))
        end

        x  = stats[:,:iter]
        y  = stats[:,:y]
        #y⁺ = stats[:,:y⁺] - stats[:,:y]
        #y⁻ = stats[:,:y]  - stats[:,:y⁺]
        y⁺⁻ = stats[:,:y]  - stats[:,:y⁺⁻]
        write(io, "$(method)_$(valid_col)_x", x)
        write(io, "$(method)_$(valid_col)_y", Array(hcat(y, y⁺⁻)'))
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

    for (problem, n_reps) ∈ [("sonar",          80),
                             ("ionosphere",     80),
                             ("breast",         80),
                             ("heart",          80),
                             ("german_gpu",     20),
                             ("australian_gpu", 20),
                             ]
        row_entries = String[]
        push!(row_entries, problem)

        for (method, name, N) ∈ [("ELBO",      "ELBO", 1),
                                 ("MSC_PIMH",  "par.-IMH", 10),
                                 ("MSC_SIMH",  "seq.-IMH", 10),
                                 ("MSC_CIS",   "single-CIS", 10),
                                 ("MSC_CISRB", "single-CISRB", 10),
                                 ("SNIS",      "SNIS", 10),
                                 ]

            data   = FileIO.load(datadir("exp_pro", "decay=false_method=$(method)_n_iter=20000_n_reps=$(n_reps)_n_samples=$(N)_optimizer=ADAM_stepsize=0.01_task=$(problem).jld2"))
            # acc_y  = data["$(method)_acc_y"][1,end]
            # acc_Δy = abs(data["$(method)_acc_y"][2,end])
            # push!(row_entries, Printf.@sprintf("%.2f {\\scriptsize{\\(\\pm %.2f\\)}}", acc_y, acc_Δy))

            lpd_y  = data["$(method)_lpd_y"][1,end]
            lpd_Δy = abs(data["$(method)_lpd_y"][2,end])
            push!(row_entries, Printf.@sprintf("{%.2f {\\scriptsize{\\(\\pm %.2f\\)}}}", lpd_y, lpd_Δy))
        end
        push!(entries, row_entries)
    end
    table = permutedims(hcat(entries...), (2,1))
    println(size(table))
    display(PrettyTables.pretty_table(table;
                                      backend = Val(:latex),
                                      tf = PrettyTables.tf_latex_booktabs,
                                      header=(["",
                                               "ELBO",
                                               "pMCSA",
                                               "JSA",
                                               "CIS",
                                               "CIS-RB",
                                               "SNIS",
                                               ],
                                              )))
end

function draw_table_bnn()
    entries = []
    for (problem, n_reps) ∈ [("wine",     80),
                             ("concrete", 80),
                             ("boston",   80),
                             ("yacht",    80),
                             ("airfoil",  80),
                             ("gas",      80),
                             ("energy",   80),
                             ("sml",      80),
                             ]
        row_entries = String[]
        push!(row_entries, problem)

        for (method, name, N) ∈ [("ELBO",      "ELBO", 1),
                                 ("MSC_PIMH",  "par.-IMH", 10),
                                 ("MSC_SIMH",  "seq.-IMH", 10),
                                 ("MSC_CIS",   "single-CIS", 10),
                                 ("MSC_CISRB", "single-CISRB", 10),
                                 ("SNIS",      "SNIS", 10),
                                 ]
            data   = FileIO.load(datadir("exp_pro", "decay=false_method=$(method)_n_iter=20000_n_reps=$(n_reps)_n_samples=$(N)_optimizer=ADAM_stepsize=0.01_task=$(problem).jld2"))
            # acc_y  = data["$(method)_acc_y"][1,end]
            # acc_Δy = abs(data["$(method)_acc_y"][2,end])
            # push!(row_entries, Printf.@sprintf("%.2f {\\scriptsize{\\(\\pm %.2f\\)}}", acc_y, acc_Δy))

            lpd_y  = data["$(method)_lpd_y"][1,end]
            lpd_Δy = abs(data["$(method)_lpd_y"][2,end])
            push!(row_entries, Printf.@sprintf("{%.2f {\\scriptsize{\\(\\pm %.2f\\)}}}", lpd_y, lpd_Δy))
        end
        push!(entries, row_entries)
    end
    table = permutedims(hcat(entries...), (2,1))
    println(size(table))
    display(PrettyTables.pretty_table(table;
                                      backend = Val(:latex),
                                      tf = PrettyTables.tf_latex_booktabs,
                                      header=(["",
                                               "ELBO",
                                               "pMCSA",
                                               "JSA",
                                               "CIS",
                                               "CIS-RB",
                                               "SNIS",
                                               ],
                                              )))
end

function draw_table_pgp()
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

        for (problem, n_reps) ∈ [("wine_gpu", 20),
                                 ("concrete_gpu", 20),
                                 ("boston_gpu", 20),
                                 ("yacht_gpu", 20),
                                 ]
            data   = FileIO.load(datadir("exp_pro", "decay=false_method=$(method)_n_iter=20000_n_reps=$(n_reps)_n_samples=$(N)_optimizer=ADAM_stepsize=0.01_task=$(problem).jld2"))
            # acc_y  = data["$(method)_acc_y"][1,end]
            # acc_Δy = abs(data["$(method)_acc_y"][2,end])
            # push!(row_entries, Printf.@sprintf("%.2f {\\scriptsize{\\(\\pm %.2f\\)}}", acc_y, acc_Δy))

            lpd_y  = data["$(method)_lpd_y"][1,end]
            lpd_Δy = abs(data["$(method)_lpd_y"][2,end])
            push!(row_entries, Printf.@sprintf("{%.2f {\\scriptsize{\\(\\pm %.2f\\)}}}", lpd_y, lpd_Δy))
        end
        push!(entries, row_entries)
    end
    table = permutedims(hcat(entries...), (2,1))
    println(size(table))
    display(PrettyTables.pretty_table(table;
                                      backend = Val(:latex),
                                      tf = PrettyTables.tf_latex_booktabs,
                                      header=(["",
                                               "wine",
                                               "concrete",
                                               "boston",
                                               "yacth",
                                               ],
                                              )))
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
    files = readdir(datadir("exp_raw"))
    files = filter(fname -> occursin(".jld2", fname), files)
    @showprogress map(files) do fname
        data      = FileIO.load(datadir("exp_raw", fname))
        result    = data["result"]
        settings  = data["settings"]

        @info "$(fname)" settings=settings
        h5open(datadir("exp_pro", fname), "w") do io
            process_data(io, result, settings)
        end
    end
end
