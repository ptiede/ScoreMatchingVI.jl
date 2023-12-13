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

function _gsm_vi_inner!(δμ, δΣ, v, θ, μ, Σ)
    Σv = Σ*v
    vΣv = dot(v,Σv)
    z = μ - θ
    μdv = dot(z, v)
    ρ   = 1/2*sqrt(1 + 4*(vΣv + μdv^2)) - 1/2
    ϵ   = Σv - z

    # update μ
    δμ .= inv(1+ρ).*(ϵ .- inv(1 + ρ + μdv).*dot(v',ϵ).*z)
    μ2 = μ + δμ

    # update Σ
    z2 = μ2 - θ
    δΣ .= z.*z' .- z2.*z2'
    return δμ, δΣ
end

function gsm_vi_step!(rng, ℓ, μ, Σ, θ, gsm::GSMVI)
    (; batch) = gsm
    Δμ = zero(θ)
    ΔΣ = fill!(similar(μ, gsm.dim, gsm.dim), 0)

    δμ = zero(θ)
    δΣ = fill!(similar(μ, gsm.dim, gsm.dim), 0)

    for _ in 1:batch
        # TODO remove dependency on Distributions?
        rand!(rng, MvNormal(μ, Σ), θ)
        _, g = LogDensityProblems.logdensity_and_gradient(ℓ, θ)
        _gsm_vi_inner!(δμ, δΣ, g, θ, μ, Σ)
        Δμ .+= δμ
        ΔΣ .+= δΣ
    end
    μ .+= δμ./batch
    Σ .+= δΣ./batch

    # TODO fix this so we only have a single cholesky decomp
    if !isposdef(Σ)
        @warn "Cholesky decomp failed reverting to last step"
        Σ .-= δΣ./batch
    end

    return μ, Σ
end

function Distributions.fit(rng::Random.AbstractRNG, ℓ, gsm::GSMVI, iterations; d0 = MvNormal(gsm.dim, 1.0), logging=false)
    μ = mean(d0)
    Σ = cov(d0)
    θ = similar(μ)
    for i in 1:iterations
        if logging
            @info "On step $i/$iterations"
        end
        gsm_vi_step!(rng, ℓ, μ, Σ, θ, gsm)
    end
    return MvNormal(μ, Σ)
end


end
