using Test
using ITensors
using T4AITensorCompat

@testset "itensor_compat" begin
    @testset "random_mpo basic properties" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))

        mpo = T4AITensorCompat.random_mpo(sites)
        @test length(mpo) == N

        # Check site indices structure
        ssites = T4AITensorCompat.siteinds(mpo)
        @test length(ssites) == N
        @test all(length(ssites[n]) == 2 for n in 1:N)
        @test all(dim.(ssites[n]) == dim.(sites[n]) for n in 1:N)
    end

    @testset "addition using compat TT" begin
        N = 3
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))

        A = T4AITensorCompat.random_mpo(sites)
        B = T4AITensorCompat.random_mpo(sites)
        C = A + B

        # Basic structural checks
        @test length(C) == N
        @test all(length(T4AITensorCompat.siteinds(C)[n]) == 2 for n in 1:N)
    end
end
