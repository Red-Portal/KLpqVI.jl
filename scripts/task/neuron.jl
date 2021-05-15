
Turing.@model neuron(graph, N, K) = begin
    ϵ     = 1e-10
    z     ~ MvNormal(N*K, 1)
    z′    = reshape(z, (K,N))
    d_z   = Distances.pairwise(Distances.Euclidean(), z′, dims=2)
    d_z⁻¹ = 1 ./ (d_z .+ ϵ)
    p_z   = Poisson.(d_z⁻¹)

    Turing.@addlogprob! mapreduce(+, CartesianIndices(d_z)) do idx
        logpdf(p_z[idx], graph[idx])
    end
end

function GraphIO.GML._gml_read_one_graph(gs, dir)
    nodes = [x[:id] for x in gs[:node]]
    g     = SimpleWeightedGraphs.SimpleWeightedDiGraph(length(nodes))
    mapping = Dict{Int,Int}()
    for (i, n) in enumerate(nodes)
        mapping[n] = i
    end
    sds = [(Int(x[:source]), Int(x[:target]), Int(x[:value])) for x in gs[:edge]]
    for (s, d, v) in (sds)
        SimpleWeightedGraphs.add_edge!(g, mapping[s], mapping[d], v)
    end
    return g
end

function load_data(task::Val{:neuron})
    # D. J. Watts and S. H. Strogatz, Nature 393, 440-442 (1998). 
    graph = open(datadir("dataset", "celegansneural.gml"), "r") do io
        LightGraphs.loadgraph(io, "digraph", GraphIO.GMLFormat())
    end
    es   = collect(LightGraphs.edges(graph))
    mat  = LightGraphs.weights(graph)'
    N    = size(mat, 1)
    mat, N
end

function hmc_params(task::Val{:neuron})
     ϵ = 0.15
     L = 64
     ϵ, L
end

function run_task(prng::Random.AbstractRNG,
                  task::Val{:neuron},
                  objective,
                  n_mc,
                  sleep_interval,
                  sleep_ϵ,
                  sleep_L;
                  show_progress=true)
    AdvancedVI.setadbackend(:reversediff)
    Turing.Core._setadbackend(Val(:reversediff))

    cnt, N = load_data(task)
    K      = 3
    model  = neuron(cnt, N, K)

    i      = 1
    function plot_callback(ℓπ, q, objective, klpq)
        stat = if(mod(i-1, 100) == 0)
            N  = floor(Int, 2^12)
            zs = rand(prng, q, N)
            ℓw = mapslices(zᵢ -> ℓπ(zᵢ) - logpdf(q, zᵢ), zs, dims=1)[1,:]
            ℓZ = StatsFuns.logsumexp(ℓw) - log(N)
            if(isnan(ℓZ))
                ℓZ = -Inf
            end
            (mll = ℓZ,)
        else
            NamedTuple()
        end
        i += 1
        stat
    end

    n_iter      = 10000
    θ, q, stats = vi(model;
                     objective      = objective,
                     n_mc           = n_mc,
                     n_iter         = n_iter,
                     tol            = 0.0005,
                     callback       = plot_callback,
                     rng            = prng,
                     sleep_interval = sleep_interval,
                     sleep_params   = (ϵ=sleep_ϵ, L=sleep_L,),
                     rhat_interval    = 100,
                     paretok_samples  = 64,
                     paretok_interval = 50,
                     optimizer        = Flux.ADAM(0.01),
                     #optimizer      = AdvancedVI.TruncatedADAGrad(),
                     show_progress = show_progress
                     )
    # β = get_variational_mode(q, model, Symbol("β"))
    # α = get_variational_mode(q, model, Symbol("α"))
    # θ = vcat(β, α)
    Dict.(pairs.(stats))
end

function sample_posterior(prng::Random.AbstractRNG,
                          task::Val{:neuron})
    K      = 3
    mat, N = load_data(task)
    model  = neuron(mat, N, K)

    sampler = Turing.NUTS(1000, 0.8;
                          max_depth=8,
                          metricT=AdvancedHMC.UnitEuclideanMetric)
    chain   = Turing.sample(model, sampler, 1000)
    L       = median(chain[:n_steps][:,1])
    ϵ       = mean(chain[:step_size][:,1])
    @info "HMC Tuning Result on $(task)" ϵ=ϵ L=L
end
