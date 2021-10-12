
function hamiltonian_energy(ℓπ, ν, M⁻¹)
    M⁻¹ν = M⁻¹*ν
    -ℓπ + dot(ν, M⁻¹ν)/2
end

mutable struct PhasePoint
    x::Vector{Float64}
    ν::Vector{Float64}
    ℓπ::Float64
end

@inline function leapfrog_recycled!(∂ℓπ∂θ::Function,
                                    z::PhasePoint,
                                    ϵ::Real,
                                    L::Int,
                                    M⁻¹::AbstractMatrix,
                                    z_recycled_buf::AbstractVector)
    ℓπ_x, ∇ℓπ_x = ∂ℓπ∂θ(z.x)
    for t = 1:L
        ν  = z.ν + (ϵ/2)*∇ℓπ_x
        x  = z.x + ϵ*M⁻¹*ν
        ℓπ_x, ∇ℓπ_x = ∂ℓπ∂θ(x)
        ν += (ϵ/2)*∇ℓπ_x
        z  = PhasePoint(x, ν, ℓπ_x)
        z_recycled_buf[t] = z
    end
    z
end

function simulate_hamiltonian!(∂ℓπ∂θ::Function,
                               z::PhasePoint,
                               l::Int,
                               ϵ::Real,
                               L::Int,
                               M⁻¹::AbstractMatrix,
                               z_recycled_buf::AbstractVector)
    H = hamiltonian_energy(z.ℓπ, z.ν, M⁻¹)

    z_recycled_buf[l+1] = z
    z_L = leapfrog_recycled!(∂ℓπ∂θ, z, ϵ, l, M⁻¹,
                             view(z_recycled_buf, 1:l))
    z_0 = leapfrog_recycled!(∂ℓπ∂θ, z, -ϵ, L-l, M⁻¹,
                             view(z_recycled_buf, l+2:L+1))

    ℓw = map(1:L+1) do i
        z_i = z_recycled_buf[i]
        if(isnan(z_i.ℓπ) || isinf(z_i.ℓπ))
            -Inf
        else
            -hamiltonian_energy(z_i.ℓπ, z_i.ν, M⁻¹)
        end
    end

    ℓZ  = StatsFuns.logsumexp(ℓw)
    w   = exp.(ℓw .- ℓZ)
    acc = mean(exp.(-abs.(H .+ ℓw)))
    w, acc
end

function hmc(prng::Random.AbstractRNG,
             ∂ℓπ∂θ::Function,
             x::AbstractVector,
             ϵ::Real,
             L::Int)
    z_recycled_buf = Array{PhasePoint}(undef, L+1)
    ℓπ_x, ∇ℓπ_x     = ∂ℓπ∂θ(x)

    n_dims  = length(x)
    M⁻¹     = Diagonal(I, n_dims)
    p_l     = DiscreteUniform(0, L)
    l       = rand(prng, p_l)
    ν       = randn(prng, n_dims)
    z       = PhasePoint(x, ν, ℓπ_x)
    w, acc  = simulate_hamiltonian!(∂ℓπ∂θ, z, l, ϵ, L, M⁻¹, z_recycled_buf)
    acc_idx = try
       rand(prng, Categorical(w))
    catch
       println(∂ℓπ∂θ(x))
       println(x)
       println(w)
    end
    z_acc   = z_recycled_buf[acc_idx]
    z_acc.x, acc
end

# function hmc_step(rng::Random.AbstractRNG,
#                   alg::AdvancedVI.VariationalInference,
#                   q,
#                   logπ,
#                   ∇logπ,
#                   z::RV,
#                   ϵ::Real,
#                   L::Int)
#     bijection = VIADBijector(q.dist)
#     η0        = inv(bijection)(z.val)
#     grad_buf  = DiffResults.GradientResult(η0)
#     ∂ℓπ∂η(η)  = begin
#         J         = Bijectors.jacobian(bijection, η)
#         f_z, ∇f_η = ∇logπ(bijection(η))
#         f_z, J*∇f_η
#     end
#     η′, acc = hmc(rng, ∂ℓπ∂η, η0, ϵ, L)
#     z′ = bijection(η′)
#     RV(z′, logπ(z′)), acc
# end
