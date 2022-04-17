
function create_gaussian(prng::Random.AbstractRNG,
                         ν::Real,
                         n_dims::Int;
                         correlated::Bool=false)
    Σ = if correlated
        n     = n_dims + ν
        Σdist = Wishart(n, diagm(1/n_dims:1/n_dims:1))
        rand(prng, Σdist)
    else
        diagm(exp.(randn(prng, n_dims)))
    end
    # μ = [-5.0, 2.0]
    μ = randn(prng, n_dims)
    MvNormal(μ, Σ)
end

function load_dataset(task::Val{:gaussian}, n_dims, ν)
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    create_gaussian(prng, 0.0, n_dims; correlated=false)
end

function load_dataset(task::Val{:gaussian_correlated}, n_dims, ν)
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    prng = Random123.Philox4x(UInt64, seed, 8);
    create_gaussian(prng, ν, n_dims; correlated=true)
end

Turing.@model gaussian(μ, Σ) = begin
    z ~ MvNormal(μ, Σ)
end

function run_task(prng::Random.AbstractRNG,
                  task::Union{Val{:gaussian},Val{:gaussian_correlated}},
                  optimizer,
                  objective,
                  n_iter,
                  n_mc,
                  defensive;
                  n_dims=100,
                  ν=150,
                  show_progress=true)
    p     = load_dataset(task, n_dims, ν)
    model = gaussian(p.μ, p.Σ)

    #AdvancedVI.setadbackend(:zygote)
    #Turing.Core._setadbackend(Val(:zygote))
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))

    varinfo  = DynamicPPL.VarInfo(model)
    varsyms  = keys(varinfo.metadata)
    n_params = sum([size(varinfo.metadata[sym].vals, 1) for sym ∈ varsyms])
    θ        = randn(prng, 2*n_params)*0.1
    q        = Turing.Variational.meanfield(model)
    q        = AdvancedVI.update(q, θ)

    kl_hist  = []
    function callback(logπ, λ)
        q′ = AdvancedVI.update(q, λ)
        μ  = mean(q′.dist)
        Σ  = cov(q′.dist)
        kl = kl_divergence(p, MvNormal(μ, Σ))
        #push!(kl_hist, kl)
        # display(Plots.plot(kl_hist))
        (kl=kl,)
    end

    ν        = Distributions.Product(fill(Cauchy(), n_params))
    θ, stats = vi(model;
                  objective        = objective,
                  n_mc             = n_mc,
                  n_iter           = n_iter,
                  callback         = callback,
                  rng              = prng,
                  defensive_dist   = ν,
                  defensive_weight = defensive,
                  optimizer        = optimizer,
                  show_progress    = show_progress
                  )
    Dict.(pairs.(stats))
end

