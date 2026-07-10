using ADTypes: jacobian_sparsity
using SparsityProbes: BloomFilterDetector
using Test
using SparseConnectivityTracer: TracerSparsityDetector

@testset "Bloom Filter Detector Tests" begin
    @testset "Validates filter parameters" begin
        @test_throws ArgumentError BloomFilterDetector(0, 1)
        @test_throws ArgumentError BloomFilterDetector(3, 0)
        @test BloomFilterDetector(3, 3) isa BloomFilterDetector
    end

    @testset "Matches default detector when filter is collision-free for dependencies" begin
        x = [1.0, -2.0, 3.0, -4.0, 5.0]
        expected = jacobian_sparsity(cross_chunk_function, x, TracerSparsityDetector())
        got = jacobian_sparsity(cross_chunk_function, x, BloomFilterDetector(128, 3))

        @test size(got) == size(expected)
        @test got == expected
    end

    @testset "Only false positives" begin
        x = [2.0, 0.5, -1.0, 4.0, 3.0]
        expected = jacobian_sparsity(mixed_dependency_function, x, TracerSparsityDetector())
        got = jacobian_sparsity(mixed_dependency_function, x, BloomFilterDetector(16, 2))

        @test size(got) == size(expected)
        @test all(got .>= expected)
    end

    @testset "Remains an over-approximation under maximal hash collisions" begin
        x = [2.0, 0.5, -1.0, 4.0, 3.0]
        expected = jacobian_sparsity(mixed_dependency_function, x, TracerSparsityDetector())
        got = jacobian_sparsity(mixed_dependency_function, x, BloomFilterDetector(1, 2))

        @test size(got) == size(expected)
        @test all(got .>= expected)
    end

    @testset "Does not report dependencies for constant outputs" begin
        constant_function(x) = [1.0, 2.0]
        x = [3.0, -1.0, 4.0]
        got = jacobian_sparsity(constant_function, x, BloomFilterDetector(4, 3))

        @test size(got) == (2, length(x))
        @test !any(got)
    end

    @testset "Rejects non-one-based inputs" begin
        x_offset = ZeroBasedVector([1.0, -2.0, 3.0, -4.0, 5.0])
        @test_throws ArgumentError jacobian_sparsity(
            cross_chunk_function, x_offset, BloomFilterDetector(128, 3)
        )
    end
end
