using ADTypes: jacobian_sparsity
using SparsityProbes: ChunkedDetector
using Test
using SparseConnectivityTracer: GradientTracer, TracerSparsityDetector, gradient

function assert_chunked_matches_default(f, x; chunk_sizes=[1, 2, 3, length(x), length(x) + 2])
    expected = jacobian_sparsity(f, x, TracerSparsityDetector())

    @testset "Chunk-size $chunk_size" for chunk_size in chunk_sizes
        got = jacobian_sparsity(f, x, ChunkedDetector(chunk_size))
        @test size(got) == size(expected)
        @test got == expected
    end
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

    @testset "Rejects non-one-based arrays" begin
        chunk = 1:2
        T = GradientTracer{Int, BitSet}
        x_offset = ZeroBasedVector([10.0, 20.0, 30.0, 40.0])

        @test_throws ArgumentError SparsityProbes._create_chunks(x_offset, 2)
        @test_throws ArgumentError SparsityProbes._trace_input_chunk(T, x_offset, chunk)
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
    @testset "Validates chunk size parameter in ChunkedDetector" begin
        @test_throws ArgumentError ChunkedDetector(0)
        @test_throws ArgumentError ChunkedDetector(-1)
        @test_throws ArgumentError ChunkedDetector(-5)
        @test ChunkedDetector(1) isa ChunkedDetector
        @test ChunkedDetector(5) isa ChunkedDetector
    end

    @testset "Matches default detector across chunk sizes" begin
        x = [1.0, -2.0, 3.0, -4.0, 5.0]
        assert_chunked_matches_default(cross_chunk_function, x)
    end

    @testset "Handles single dependency and constant rows" begin
        x = [2.0, 0.5, -1.0, 4.0, 3.0]
        assert_chunked_matches_default(mixed_dependency_function, x)
    end

    @testset "Rejects non-one-based inputs at the public API" begin
        x_offset = ZeroBasedVector([1.0, -2.0, 3.0, -4.0, 5.0])
        @test_throws ArgumentError jacobian_sparsity(cross_chunk_function, x_offset, ChunkedDetector(2))
    end
end
