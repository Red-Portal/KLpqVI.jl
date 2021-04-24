struct ADBijector{AD, N, B <: Bijectors.Bijector{N}} <: Bijectors.ADBijector{AD, N}
    b::B
end
ADBijector(d::Distribution) = ADBijector{Bijectors.ADBackend()}(d)
ADBijector{AD}(d::Distribution) where {AD} = ADBijector{AD}(Bijectors.bijector(d))
ADBijector{AD}(b::B) where {AD, N, B <: Bijectors.Bijector{N}} = ADBijector{AD, N, B}(b)

(b::ADBijector)(x) = b.b(x)
(b::Bijectors.Inverse{<:ADBijector})(x) = inv(b.orig.b)(x)

function gradient!(alg::AdvancedVI.VariationalInference{<:AdvancedVI.ForwardDiffAD},
                   f::Function,
                   x::AbstractVector,
                   out::DiffResults.MutableDiffResult)
    chunk_size = AdvancedVI.getchunksize(typeof(alg))
    chunk      = ForwardDiff.Chunk(min(length(x), chunk_size))
    config     = ForwardDiff.GradientConfig(f, x, chunk)
    ForwardDiff.gradient!(out, f, x, config)
end

function gradient!(::AdvancedVI.VariationalInference{<:AdvancedVI.ZygoteAD},
                   f::Function,
                   x::AbstractVector,
                   out::DiffResults.MutableDiffResult)
    y, back = Zygote.pullback(f, x)
    dy      = first(back(1.0))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, dy)
end
