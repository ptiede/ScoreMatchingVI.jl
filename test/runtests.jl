using ScoreMatchingVI
using Test
using LogDensityProblems, LogDensityProblemsAD
using Random
using LinearAlgebra
using Distributions
using ForwardDiff


struct GaussDensity{M}
    d::M
end
LogDensityProblems.logdensity(p::GaussDensity, θ) = logpdf(p.d, θ)  # standard multivariate normal
LogDensityProblems.dimension(p::GaussDensity) = length(p.d)
LogDensityProblems.capabilities(::Type{GaussDensity}) = LogDensityProblems.LogDensityOrder{0}()


# struct BannanaDensity{T}
#     a::T
#     b::T
# end
# LogDensityProblems.logdensity(p::BannanaDensity, x) = -(p.a - x[1])^2 - p.b*(x[2] - x[1]^2)^2  # standard multivariate normal
# LogDensityProblems.dimension(p::BannanaDensity) = 2
# LogDensityProblems.capabilities(::Type{BannanaDensity}) = LogDensityProblems.LogDensityOrder{0}()


@testset "ScoreMatchingVI.jl" begin

    rng = Xoshiro(42)
    ndim = 5
    σ = randn(ndim)
    Σ = σ'.*σ .+ 0.1*Diagonal(ones(ndim))
    μ = randn(ndim)

    @testset "batch size 1" begin
        vi = GSMVI(ndim; batch=1)

        d = fit(rng, ADgradient(:ForwardDiff, GaussDensity(MvNormal(Σ))), vi, 1000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))))
        @test isapprox(d.μ, zeros(ndim), atol=1e-6)
        @test isapprox(d.Σ, Σ, atol=1e-6)

        d = fit(rng, ADgradient(:ForwardDiff, GaussDensity(MvNormal(μ, Σ))), vi, 1000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))))
        @test isapprox(d.μ, μ, atol=1e-6)
        @test isapprox(d.Σ, Σ, atol=1e-6)

    end

    @testset "batch size 2" begin
        vi = GSMVI(ndim; batch=2)


        d = fit(rng, ADgradient(:ForwardDiff, GaussDensity(MvNormal(Σ))), vi, 1000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))))
        @test isapprox(d.μ, zeros(ndim), atol=1e-6)
        @test isapprox(d.Σ, Σ, atol=1e-6)

        d = fit(rng, ADgradient(:ForwardDiff, GaussDensity(MvNormal(μ, Σ))), vi, 1000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))))
        @test isapprox(d.μ, μ, atol=1e-6)
        @test isapprox(d.Σ, Σ, atol=1e-6)
    end

    @testset "batch size 3" begin
        vi = GSMVI(ndim; batch=2)

        d = fit(rng, ADgradient(:ForwardDiff, GaussDensity(MvNormal(Σ))), vi, 1000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))))
        @test isapprox(d.μ, zeros(ndim), atol=1e-6)
        @test isapprox(d.Σ, Σ, atol=1e-6)

        d = fit(rng, ADgradient(:ForwardDiff, GaussDensity(MvNormal(μ, Σ))), vi, 1000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))), logging=false)
        @test isapprox(d.μ, μ, atol=1e-6)
        @test isapprox(d.Σ, Σ, atol=1e-6)
    end

    @testset "High dim problem" begin
        ndim = 200
        σ = randn(ndim)
        Σ = σ'.*σ .+ 0.1*Diagonal(ones(ndim))
        μ = randn(ndim)

        vi = GSMVI(ndim; batch=2)


        d = fit(rng, ADgradient(:ForwardDiff, GaussDensity(MvNormal(Σ))), vi, 50_000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))))
        @test isapprox(d.μ, zeros(ndim), atol=1e-6)
        @test isapprox(d.Σ, Σ, atol=1e-6)

        d = fit(rng, ADgradient(:ForwardDiff, GaussDensity(MvNormal(μ, Σ))), vi, 50_000; d0 = MvNormal(zeros(ndim), collect(Float64, I(ndim))), logging=true)
        @test isapprox(d.μ, μ, atol=1e-6)
        @test isapprox(d.Σ, Σ, atol=1e-6)

    end


end
