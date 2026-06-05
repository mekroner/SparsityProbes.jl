using SparsityProbes: create_chunks, trace_input_chunk, combine_patterns
using Test
using SparseConnectivityTracer: GradientTracer


function toy_function(x)
    y1 = x[1] * x[2]
    y2 = x[2] + 0.0
    return [y1, y2]
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
        pattern1 = [1 2; 3 4]
        pattern2 = [5 6; 7 8]
        combined = combine_patterns([pattern1, pattern2])
        
        @test combined == [1 2 5 6; 3 4 7 8]
    end

end
