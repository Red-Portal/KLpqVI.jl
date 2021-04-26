
@inline function gradient!(alg::AdvancedVI.VariationalInference{<:AdvancedVI.ForwardDiffAD},
                           f::Function,
                           x::AbstractVector,
                           out::DiffResults.MutableDiffResult)
    chunk_size = AdvancedVI.getchunksize(typeof(alg))
    chunk      = ForwardDiff.Chunk(min(length(x), chunk_size))
    config     = ForwardDiff.GradientConfig(f, x, chunk)
    ForwardDiff.gradient!(out, f, x, config)
end

@inline function gradient!(::AdvancedVI.VariationalInference{<:AdvancedVI.ZygoteAD},
                           f::Function,
                           x::AbstractVector,
                           out::DiffResults.MutableDiffResult)
    y, back = Zygote.pullback(f, x)
    dy      = first(back(1.0))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, dy)
end

@inline function gradient!(::AdvancedVI.VariationalInference{<:AdvancedVI.TrackerAD},
                           f::Function,
                           x::AbstractVector,
                           out::DiffResults.MutableDiffResult)
    y, back = Tracker.forward(f, x)
    dy      = back(1.0)
    DiffResults.value!(out,    Tracker.data(y))
    DiffResults.gradient!(out, Tracker.data(back(dy)))
end
