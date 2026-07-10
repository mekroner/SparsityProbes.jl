"""
    ChunkedDetector(chunk_size::Int)

A detector configuration for `ADTypes.jacobian_sparsity` that computes the 
sparsity pattern in blocks of a given `chunk_size` to optimize memory overhead.

# Arguments
- `chunk_size::Int`: The number of input indices to track simultaneously in a single pass. Must be a positive integer.

# Returns
- `ChunkedDetector`: A configuration instance for chunk-based sparsity detection.
"""
struct ChunkedDetector
    chunk_size::Int
    
    function ChunkedDetector(chunk_size::Int)
        chunk_size > 0 ||
            throw(ArgumentError("ChunkedDetector requires chunk_size > 0"))
        return new(chunk_size)
    end
end

"""
    ADTypes.jacobian_sparsity(f, x, detector::ChunkedDetector)

Compute the Jacobian sparsity pattern of a function `f` at input `x` using 
a chunked tracking strategy.

# Arguments
- `f`: The target function whose Jacobian sparsity is being analyzed.
- `x`: The input argument (usually an array) where the sparsity is evaluated.
- `detector::ChunkedDetector`: The configuration specifying the block/chunk size.

# Returns
- `AbstractMatrix`: The final combined Jacobian sparsity pattern matrix.

"""
function ADTypes.jacobian_sparsity(f, x, detector::ChunkedDetector)
    return _jacobian_sparsity_chunked(f, x, detector.chunk_size)
end

"""
    _jacobian_sparsity_chunked(f, x, chunk_size::Int)

Internal logic to execute the chunked sparsity tracking over the function `f`.

# Arguments
- `f`: The target function.
- `x`: The input argument.
- `chunk_size::Int`: The size of each index block.

# Returns
- `AbstractMatrix`: The combined sparsity matrix aggregated across all chunks.
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

# Arguments
- `x::AbstractArray`: The input array whose indices need partitioning.
- `chunk_size::Int`: The maximum size of each chunk.

# Returns
- `Vector{UnitRange{Int}}`: A collection of index ranges partitioning the array.
"""
function _create_chunks(x::AbstractArray, chunk_size::Int)
    n = length(x)
    return [i:min(i + chunk_size - 1, n) for i in 1:chunk_size:n]
end

"""
    _trace_input_chunk(T::Type{<:GradientTracer}, x::AbstractArray, chunk::UnitRange{Int})

Initialize a dual-number vector `xt` where only the indices belonging to `chunk` 
are seeded with active tracers. All other indices are filled with empty tracers.

# Arguments
- `T::Type{<:GradientTracer}`: The concrete tracer type to instantiate.
- `x::AbstractArray`: The baseline input array providing the structural layout.
- `chunk::UnitRange{Int}`: The range of linear indices to mark as active.

# Returns
- `Vector{T}`: A tracer-wrapped vector ready for automatic differentiation tracking.
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

# Arguments
- `patterns::AbstractVector{<:AbstractMatrix}`: A vector of intermediate sparsity matrices.

# Returns
- `AbstractMatrix`: A single merged matrix representing the global sparsity layout.
"""
function _combine_patterns(patterns::AbstractVector{<:AbstractMatrix})
    return reduce((a, b) -> a .| b, patterns)
end