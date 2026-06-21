module SparsityProbes

    using ADTypes: ADTypes, jacobian_sparsity
    using SparseConnectivityTracer: GradientTracer, myempty, jacobian_tracers_to_matrix, to_array
    
    const T = GradientTracer{Int, BitSet}

    """
        ChunkedDetector(chunk_size::Int)

    A detector configuration for `ADTypes.jacobian_sparsity` that computes the 
    sparsity pattern in blocks of a given `chunk_size` to optimize memory overhead.
    """
    struct ChunkedDetector
        chunk_size::Int
    end

    """
    ADTypes.jacobian_sparsity(f, x, detector::ChunkedDetector)

    Compute the Jacobian sparsity pattern of a function `f` at input `x` using 
    a chunked tracking strategy.
    """
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
    
    """
        create_chunks(x::AbstractArray, chunk_size::Int)

    Split the linear indices of array `x` into a vector of ranges based on `chunk_size`. 
    Handles uneven edge cases and chunk sizes larger than the array length.
    """
    function create_chunks(x::AbstractArray, chunk_size::Int)::Vector{UnitRange{Int}}
        n = length(x)
        return [i:min(i + chunk_size - 1, n) for i in 1:chunk_size:n]
    end
    

    """
        trace_input_chunk(T::Type{<:GradientTracer}, x::AbstractArray, chunk::UnitRange{Int})

    Initialize a dual-number vector `xt` where only the indices belonging to `chunk` 
    are seeded with active tracers. All other indices are filled with empty tracers.
    """
    function trace_input_chunk(T::Type{GradientTracer{Int, BitSet}}, x::AbstractArray, chunk::UnitRange{Int})
        xt = Vector{T}(undef, length(x))
        for i in 1:length(x)
            if i in chunk
                xt[i] = T(BitSet(i))
            else
                xt[i] = myempty(T)
            end
        end
        return xt
    end
    
    """
        combine_patterns(patterns::AbstractVector{<:AbstractMatrix})

    Merge a sequence of partial sparsity pattern matrices into a single
    sparsity pattern matrix by performing an element-wise logical OR
    operation across all matrices.
    """
    function combine_patterns(patterns::AbstractVector{<:AbstractMatrix})::AbstractMatrix
        return reduce((a, b) -> a .| b, patterns)
    end
    export jacobian_sparsity

end # module
