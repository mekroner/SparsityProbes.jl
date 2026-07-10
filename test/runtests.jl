import SparsityProbes
using SparsityProbes
using Test

@testset verbose=true "SparsityProbes.jl Tests" begin

    include("util_functions.jl")

    @testset "Chunked Detector Tests" begin
        include("test_chunked_detector.jl")
    end

    @testset "Bloom Filter Detector Tests" begin
        include("test_bloom_filter.jl")
    end

    @testset "Hierarchical Bloom Filter Detector Tests" begin
        include("test_hierarchical_bloom_filter.jl")
    end

    @testset "Fuzzy Tests" begin    
        include("test_fuzzy.jl")
    end
end