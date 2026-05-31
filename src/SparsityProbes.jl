module SparsityProbes

    using ADTypes: ADTypes, jacobian_sparsity
    using SparseConnectivityTracer: GradientTracer, jacobian_tracers_to_matrix, to_array
    
    const T = GradientTracer{Int, BitSet}

    struct ChunkedDetector
        chunk_size::Int
    end

    function ADTypes.jacobian_sparsity(f, x, detector::ChunkedDetector)
        return _jacobian_sparsity_chunked(f,x, detector.chunk_size)
    end
    
    function _jacobian_sparsity_chunked(f, x, chunk_size)
        chunks = create_chunks(x, chunk_size)
        patterns = map(chunks) do chunk
            xt = trace_input_chunk(T, x, chunk)
            yt = f(xt)
            jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
        end
        return combine_patterns(patterns)
    end
    
    function create_chunks(x::AbstractArray, chunk_size::Int)::Vector{UnitRange{Int}}
        """
        Splits the linear indices of array `x` int a vector of ranges
        based on the `chunk_size`. Handles uneven edgecase and to large chunk edgecase
        """
        n = length(x)
        return [i:min(i + chunk_size - 1, n) for i in 1:chunk_size:n]
    end
    
    function trace_input_chunk(::Type{Tracer}, x::AbstractArray, chunk::UnitRange{Int}) where {Tracer}
        # TODO: seed only indices within this chunk, {}
        @warn "Not Implemented yet"
        return
    end
    
    function combine_patterns(patterns::AbstractVector{<:AbstractMatrix})::AbstractMatrix
        # TODO: merge the partial patterns
        return
    end

end
