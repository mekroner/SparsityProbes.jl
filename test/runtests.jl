using SparsityProbes: create_chunks, trace_input_chunk, combine_patterns
using Test



@testset "Chunked Detector Fuzzy Test" begin
end


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
    
    @testset "Trace Input Chunk (1)" begin
        # T_Tracer = GradientTracer{Int, BitSet}[cite: 25]
        
        # # Test chunk 1:2
        # chunk = 1:2
        # xt = trace_input_chunk(T_Tracer, x_test, chunk)
        # expected_xt1 = [
        #     T_Tracer(BitSet(1)), 
        #     T_Tracer(BitSet(2)), 
        #     myempty(T_Tracer), 
        #     myempty(T_Tracer)
        # ]
        
        # # Verify structure
        # @test size(xt) == size(x_test)
        # @test eltype(xt) == T_Tracer
        # @test xt = expected_xt1
        
    end

    @testset "Combine Patterns (1)" begin
        
    end
end

