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
    return _jacobian_sparsity_chunked(f, x, detector.chunk_size)
end

"""
    _jacobian_sparsity_chunked(f, x, chunk_size::Int)

Internal logic to execute the chunked sparsity tracking over the function `f`.
"""
function _jacobian_sparsity_chunked(f, x, chunk_size)
    chunks = _create_chunks(x, chunk_size)
    patterns = map(chunks) do chunk
        xt = _trace_input_chunk(DEFAULT_TRACER_TYPE, x, chunk)
        yt = f(xt)
        jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
    end
    return _combine_patterns(patterns)
end

"""
    _create_chunks(x::AbstractArray, chunk_size::Int)

Split the linear indices of array `x` into a vector of ranges based on `chunk_size`. 
Handles uneven edge cases and chunk sizes larger than the array length.
"""
function _create_chunks(x::AbstractArray, chunk_size::Int)
    n = length(x)
    return [i:min(i + chunk_size - 1, n) for i in 1:chunk_size:n]
end

"""
    _trace_input_chunk(T::Type{<:GradientTracer}, x::AbstractArray, chunk::UnitRange{Int})

Initialize a dual-number vector `xt` where only the indices belonging to `chunk` 
are seeded with active tracers. All other indices are filled with empty tracers.
"""
function _trace_input_chunk(T::Type{<:GradientTracer}, x::AbstractArray, chunk::UnitRange{Int})
    xt = Vector{T}(undef, length(x))
    for i in eachindex(x)
        xt[i] = myempty(T)
        if i in chunk
            xt[i] = T(BitSet(i))
        end
    end
    return xt
end

"""
    _combine_patterns(patterns::AbstractVector{<:AbstractMatrix})

Merge a sequence of partial sparsity pattern matrices into a single
sparsity pattern matrix by performing an element-wise logical OR
operation across all matrices.
"""
function _combine_patterns(patterns::AbstractVector{<:AbstractMatrix})
    return reduce((a, b) -> a .| b, patterns)
end