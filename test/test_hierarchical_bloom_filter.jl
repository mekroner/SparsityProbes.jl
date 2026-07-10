using ADTypes: jacobian_sparsity
using SparsityProbes: HierarchicalBloomFilterDetector
using Test
using SparseConnectivityTracer: TracerSparsityDetector

@testset "Hierarchical Bloom Filter Detector Tests" begin
    @testset "Validates filter parameters" begin
        @test_throws ArgumentError HierarchicalBloomFilterDetector(0, 1)
        @test_throws ArgumentError HierarchicalBloomFilterDetector(3, 0)
        @test HierarchicalBloomFilterDetector(3, 3) isa HierarchicalBloomFilterDetector
    end

    @testset "Matches default detector when the first Bloom pass is collision-free" begin
        x = [1.0, -2.0, 3.0, -4.0, 5.0]
        expected = jacobian_sparsity(cross_chunk_function, x, TracerSparsityDetector())
        got = jacobian_sparsity(cross_chunk_function, x, HierarchicalBloomFilterDetector(128, 3))

        @test size(got) == size(expected)
        @test got == expected
    end

    @testset "Removes Bloom false positives even under maximal hash collisions" begin
        x = [2.0, 0.5, -1.0, 4.0, 3.0]
        expected = jacobian_sparsity(mixed_dependency_function, x, TracerSparsityDetector())
        got = jacobian_sparsity(mixed_dependency_function, x, HierarchicalBloomFilterDetector(1, 2))

        @test size(got) == size(expected)
        @test got == expected
    end

    @testset "Does not report dependencies for constant outputs" begin
        constant_function(x) = [1.0, 2.0]
        x = [3.0, -1.0, 4.0]
        expected = jacobian_sparsity(constant_function, x, TracerSparsityDetector())
        got = jacobian_sparsity(constant_function, x, HierarchicalBloomFilterDetector(128, 3))

        @test size(got) == (2, length(x))
        @test got == expected
        @test !any(got)
    end
end
