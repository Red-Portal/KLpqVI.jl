
function sample_sphere(prng::Random.AbstractRNG,
                       n_dims::Int,
                       n_samples::Int)
    x = randn(prng, n_dims, n_samples)
    u = x ./ mapslices(norm, x, dims=1)
    u
end

function randn_raoblackwell(prng::Random.AbstractRNG,
                            n_dims::Int,
                            n_samples::Int)
    #r = rand(prng, Chi(n_dims))
    #u = sample_sphere(prng, n_dims, n_samples)#div(n_samples, 2))
    u = randn(prng, n_dims, div(n_samples,2))
    u = hcat(u, -u)
    #r*u
end

function rand_raoblackwell(rng::Random.AbstractRNG,
                           d::DistributionsAD.TuringDenseMvNormal,
                           n_samples::Int)
    return d.m .+ d.C.U' * randn_raoblackwell(rng, length(d), n_samples)
end

function rand_raoblackwell(rng::Random.AbstractRNG,
                           d::DistributionsAD.TuringScalMvNormal,
                           n_samples::Int)
    return d.m .+ d.σ .* randn_raoblackwell(rng, length(d), n_samples)
end

function rand_raoblackwell(rng::Random.AbstractRNG,
                           d::DistributionsAD.TuringDiagMvNormal,
                           n_samples::Int)
    return d.m .+ d.σ .* randn_raoblackwell(rng, length(d), n_samples)
end
