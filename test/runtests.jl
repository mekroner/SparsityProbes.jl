using ADTypes: jacobian_sparsity
using SparsityProbes: ChunkedDetector, create_chunks, trace_input_chunk, combine_patterns
using Test
using SparseConnectivityTracer: GradientTracer, TracerSparsityDetector


function toy_function(x)
    y1 = x[1] * x[2]
    y2 = x[2] + 0.0
    return [y1, y2]
end

function assert_chunked_matches_default(f, x; chunk_sizes=[1, 2, 3, length(x), length(x) + 2])
    expected = jacobian_sparsity(f, x, TracerSparsityDetector())

    for chunk_size in chunk_sizes
        got = jacobian_sparsity(f, x, ChunkedDetector(chunk_size))
        @test size(got) == size(expected)
        @test got == expected
    end
end

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
        chunks_2 = create_chunks(x_test, 2)
        @test chunks_2 == [1:2, 3:4]
       
        # this is uneven
        chunks_3 = create_chunks(x_test, 3)
        @test chunks_3 == [1:3, 4:4]

        chunks_large = create_chunks(x_test, 10)
        @test chunks_large == [1:4]
    end

    @testset "Trace Input Chunk (2)" begin
        chunk = 1:2
        T = GradientTracer{Int, BitSet}
        xt = trace_input_chunk(T, x_test, chunk)

        @test eltype(xt) == T
        @test getfield(xt[1], 1) == BitSet([1])
        @test getfield(xt[2], 1) == BitSet([2])

        @test getfield(xt[3], 1) == BitSet()
        @test getfield(xt[4], 1) == BitSet()
    end

    @testset "Combine Patterns (3)" begin
        patterns = [
            [true  false; false false],
            [false true ; false false],
            [false false; true  false]
        ]

        combined = combine_patterns(patterns)
        
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
