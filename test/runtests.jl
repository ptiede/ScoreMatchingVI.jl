using ScoreMatchingVI
using Test
using LogDensityProblems, LogDensityProblemsAD
using Random
using LinearAlgebra
using Distributions
using ForwardDiff


struct LogTargetDensity{M}
    dim::Int
    Σ::M
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = -sum(θ.^2 ./p.Σ)/2  # standard multivariate normal
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()


@testset "ScoreMatchingVI.jl" begin


    ndim = 5
    Σ = rand(ndim)
    ℓ = LogTargetDensity(length(Σ), collect(Float64, Σ))
    @testset "batch size 1" begin
        vi = GSMVI(ndim; batch=1)

        rng = Random.default_rng()
        d = fit(rng, ADgradient(:ForwardDiff, ℓ), vi, 1000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))))

        @test isapprox(d.μ, zeros(ndim), atol=1e-6)
        @test isapprox(d.Σ, diagm(ℓ.Σ), atol=1e-6)
    end

    @testset "batch size 2" begin
        vi = GSMVI(ndim; batch=2)

        rng = Random.default_rng()
        d = fit(rng, ADgradient(:ForwardDiff, ℓ), vi, 1000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))))

        @test isapprox(d.μ, zeros(ndim), atol=1e-6)
        @test isapprox(d.Σ, diagm(ℓ.Σ), atol=1e-6)
    end

    @testset "batch size 3" begin
        vi = GSMVI(ndim; batch=2)

        rng = Random.default_rng()
        d = fit(rng, ADgradient(:ForwardDiff, ℓ), vi, 1000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))))

        @test isapprox(d.μ, zeros(ndim), atol=1e-6)
        @test isapprox(d.Σ, diagm(ℓ.Σ), atol=1e-6)
    end


end
