module ScoreMatchingVI

export fit, GSMVI

using LogDensityProblems
using Distributions
using LinearAlgebra
using Random

# Write your package code here.

struct GSMVI{I<:Integer}
    dim::I
    batch::I
end

function GSMVI(dimension; batch=2)
    return GSMVI(dimension, batch)
end

function _gsm_vi_inner!(v, θ, μ, Σ)
    Σv = Σ*v
    vΣv = dot(v,Σv)
    μdv = dot(μ - θ, v)
    ρ   = 1/2*sqrt(1 + 4*(vΣv + μdv^2)) - 1/2
    ϵ   = Σv - μ + θ

    # update μ
    μv = (μ - θ)*v'
    δμ = inv(1+ρ)*(I - μv*inv(1 + ρ + μdv))*ϵ
    μ2 = μ + δμ

    # update Σ
    δΣ = (μ - θ)*(μ - θ)' - (μ2 - θ)*(μ2 - θ)'
    return δμ, δΣ
end

function gsm_vi_step!(rng, ℓ, μ, Σ, θ, gsm::GSMVI)
    (; batch) = gsm
    δμ = zero(θ)
    δΣ = fill!(similar(μ, gsm.dim, gsm.dim), 0)
    for _ in 1:batch
        rand!(rng, MvNormal(μ, Σ), θ)
        _, g = LogDensityProblems.logdensity_and_gradient(ℓ, θ)
        dμ, dΣ =  _gsm_vi_inner!(g, θ, μ, Σ)
        δμ .+= dμ
        δΣ .+= dΣ
    end
    μ .+= δμ./batch
    Σ .+= δΣ./batch
    if !isposdef(Σ)
        @warn "Cholesky decomp failed reverting to last step"
        Σ .-= Σ′./batch
    end

    return μ, Σ
end

function Distributions.fit(rng::Random.AbstractRNG, ℓ, gsm::GSMVI, iterations; d0 = MvNormal(gsm.dim, 1.0))
    μ = mean(d0)
    Σ = cov(d0)
    θ = similar(μ)
    for i in 1:iterations
        @info "On step $i/$iterations"
        gsm_vi_step!(rng, ℓ, μ, Σ, θ, gsm)
    end
    return MvNormal(μ, Σ)
end


end
