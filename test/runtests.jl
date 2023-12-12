using ScoreMatchingVI
using Test
using LogDensityProblems, LogDensityProblemsAD
using Random
using LinearAlgebra

struct LogTargetDensity{M}
    dim::Int
    Σ::M
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = -sum(θ.^2 ./p.Σ)/2  # standard multivariate normal
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()


@testset "ScoreMatchingVI.jl" begin
    using ForwardDiff


    ndim = 6000
    Σ = rand(ndim)
    ℓ = LogTargetDensity(length(Σ), collect(Float64, Σ))

    vi = GSMVI(ndim)

    rng = Random.default_rng()
    d = fit(rng, ADgradient(:ForwardDiff, ℓ), vi, 1000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))))
    end
