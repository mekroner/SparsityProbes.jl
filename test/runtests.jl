using ADTypes: jacobian_sparsity
import SparsityProbes
using SparsityProbes:
    BloomFilterDetector,
    ChunkedDetector,
    HierarchicalBloomFilterDetector
using Test
using SparseConnectivityTracer: GradientTracer, TracerSparsityDetector, gradient
using SparseArrays: SparseMatrixCSC, sprand


function toy_function(x)
    y1 = x[1] * x[2]
    y2 = x[2] + 0.0
    return [y1, y2]
end

function assert_chunked_matches_default(f, x; chunk_sizes=[1, 2, 3, length(x), length(x) + 2])
    expected = jacobian_sparsity(f, x, TracerSparsityDetector())

    @testset "Chunk-size $chunk_size" for chunk_size in chunk_sizes
        got = jacobian_sparsity(f, x, ChunkedDetector(chunk_size))
        @test size(got) == size(expected)
        @test got == expected
    end
end

nothing

function cross_chunk_function(x)
    y1 = x[1] * x[3] + x[5]
    y2 = x[2] - x[4]
    y3 = x[1] + x[4] * x[5]
    y4 = 7.0
    return [y1, y2, y3, y4]
end

function mixed_dependency_function(x)
    y1 = x[1] + x[2] * x[4]
    y2 = x[3]
    y3 = x[2] * x[5] + x[1]
    y4 = 0.0
    return [y1, y2, y3, y4]
end

@testset "Chunked Detector Helpers Unit Tests" begin

    x_test = [10.0, 20.0, 30.0, 40.0]
    @testset "Create Chunks (1)" begin
        chunks_2 = SparsityProbes._create_chunks(x_test, 2)
        @test chunks_2 == [1:2, 3:4]
       
        # this is uneven
        chunks_3 = SparsityProbes._create_chunks(x_test, 3)
        @test chunks_3 == [1:3, 4:4]

        chunks_large = SparsityProbes._create_chunks(x_test, 10)
        @test chunks_large == [1:4]
    end

    @testset "Trace Input Chunk (2)" begin
        chunk = 1:2
        T = GradientTracer{Int, BitSet}
        xt = SparsityProbes._trace_input_chunk(T, x_test, chunk)

        @test eltype(xt) == T
        @test gradient(xt[1]) == BitSet([1])
        @test gradient(xt[2]) == BitSet([2])

        @test gradient(xt[3]) == BitSet()
        @test gradient(xt[4]) == BitSet()
    end

    @testset "Combine Patterns (3)" begin
        patterns = [
            [true  false; false false],
            [false true ; false false],
            [false false; true  false]
        ]

        combined = SparsityProbes._combine_patterns(patterns)
        
        @test combined == [true  true ; true  false]
    end

end

@testset "Chunked Detector Global Comparison Tests" begin
    @testset "Matches default detector across chunk sizes" begin
        x = [1.0, -2.0, 3.0, -4.0, 5.0]
        assert_chunked_matches_default(cross_chunk_function, x)
    end

    @testset "Handles single dependency and constant rows" begin
        x = [2.0, 0.5, -1.0, 4.0, 3.0]
        assert_chunked_matches_default(mixed_dependency_function, x)
    end
end

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
end

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


# Fuzzy tests for all detectors, to ensure they don't crash on random inputs.

struct RandomSparseNetwork
    dependency_matrix::SparseMatrixCSC{Float64, Int}
end

# Make struct callable
function (rsn::RandomSparseNetwork)(x::AbstractVector)
    W = rsn.dependency_matrix
    return sin.(W * x) .^ cos.(W * (x .^ 2))
end

# Allows us the bloom filters to over-approximate the sparsity pattern, but not under-approximate it.
function equal_or_over_approximation(A::AbstractMatrix{Bool}, B::AbstractMatrix{Bool})
    return size(A) == size(B) && all(A .>= B)
end

function fuzzy_test_detector(detector; n_tests::Int=1000, max_inputs::Int=100, max_outputs::Int=100)
    @testset "Fuzzy Tests for $(typeof(detector)): $n_tests iterations" begin
        for i in 1:n_tests
            # Random dimension 
            n_inputs = rand(1:max_inputs)
            n_outputs = rand(1:max_outputs)

            # random density between 1% and 50%
            density = rand(0.01:0.01:0.5) 
            W = sprand(n_outputs, n_inputs, density)
            f_rand = RandomSparseNetwork(W)
            x = randn(n_inputs)

            expected = jacobian_sparsity(f_rand, x, TracerSparsityDetector())
            got = jacobian_sparsity(f_rand, x, detector)

            @test size(got) == size(expected)
            @test equal_or_over_approximation(got, expected)
        end
    end
end

@testset "Fuzzy Tests" begin
    fuzzy_test_detector(HierarchicalBloomFilterDetector(128, 3); n_tests=1000)
    fuzzy_test_detector(BloomFilterDetector(128, 3); n_tests=1000)
    fuzzy_test_detector(ChunkedDetector(10); n_tests=1000)
end
