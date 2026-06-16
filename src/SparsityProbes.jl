module SparsityProbes

    using ADTypes: ADTypes, jacobian_sparsity
    using SparseConnectivityTracer: GradientTracer, myempty, jacobian_tracers_to_matrix, to_array
    using XXhash: xxh32
    
    const T = GradientTracer{Int, BitSet}
    
    struct ChunkedDetector
        chunk_size::Int
    end

    struct BloomFilterDetector
        filter_size_m::Int
        num_hashes_k::Int

        function BloomFilterDetector(filter_size_m::Int, num_hashes_k::Int)
            filter_size_m > 0 ||
                throw(ArgumentError("BloomFilterDetector requires filter_size_m > 0"))
            num_hashes_k > 0 ||
                throw(ArgumentError("BloomFilterDetector requires num_hashes_k > 0"))
            return new(filter_size_m, num_hashes_k)
        end
    end

    function ADTypes.jacobian_sparsity(f, x, detector::Union{ChunkedDetector, BloomFilterDetector})
        if detector isa ChunkedDetector
            return _jacobian_sparsity_chunked(f, x, detector.chunk_size)
        elseif detector isa BloomFilterDetector
            return _jacobian_sparsity_bloom_filter(f, x, detector.filter_size_m, detector.num_hashes_k)
        end
    end
    
    function _jacobian_sparsity_chunked(f, x, chunk_size)
        chunks = _create_chunks(x, chunk_size)
        patterns = map(chunks) do chunk
            xt = _trace_input_chunk(T, x, chunk)
            yt = f(xt)
            jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
        end
        return _combine_patterns(patterns)
    end
    
    function _create_chunks(x::AbstractArray, chunk_size::Int)::Vector{UnitRange{Int}}
        """
        Splits the linear indices of array `x` int a vector of ranges
        based on the `chunk_size`. Handles uneven edgecase and to large 
        chunk edgecase.
        """
        n = length(x)
        return [i:min(i + chunk_size - 1, n) for i in 1:chunk_size:n]
    end
    
    function _trace_input_chunk(T::Type{GradientTracer{Int, BitSet}}, x::AbstractArray, chunk::UnitRange{Int})
        xt = Vector{T}(undef, length(x))
        for i in 1:length(x)
            xt[i] = myempty(T)
            if i in chunk
                xt[i] = T(BitSet(i))
            end
        end
        return xt
    end
    
    function _combine_patterns(patterns::AbstractVector{<:AbstractMatrix})::AbstractMatrix
        return reduce((a, b) -> a .| b, patterns)
    end

    function _hash(i::Integer, j::Integer, filter_size_m::Integer)
        h = xxh32([UInt32(i)], UInt32(j))
        return (h % filter_size_m) + 1
    end

    function _bloomseed(n::Integer, num_hashes_k::Integer, filter_size_m::Integer)
        S = zeros(Bool, n, filter_size_m)
        for i in 1:n
            for j in 1:num_hashes_k
                l = _hash(i, j, filter_size_m)
                S[i, l] = 1
            end
        end
        return S
    end

    function _probe(f, S::AbstractMatrix{Bool})
        n, filter_size_m = size(S)
        xt = Vector{T}(undef, n)
        for i in 1:n
            xt[i] = T(BitSet(findall(@view S[i, :])))
        end
        yt = f(xt)
        yt_array = to_array(yt)
        Q = zeros(Bool, length(yt_array), filter_size_m)
        for (row, y) in enumerate(yt_array)
            y isa T || continue
            for column in getfield(y, :gradient)
                Q[row, column] = true
            end
        end
        return Q
    end

    function _bloomharvest(Q::AbstractMatrix{Bool}, S::AbstractMatrix{Bool}, num_hashes_k::Integer)
        W = Int.(Q) * transpose(Int.(S))
        return W .== num_hashes_k
    end

    function _jacobian_sparsity_bloom_filter(f, x, filter_size_m::Integer, num_hashes_k::Integer)
        n = length(x)
        S = _bloomseed(n, num_hashes_k, filter_size_m)
        Q = _probe(f, S)
        R = _bloomharvest(Q, S, num_hashes_k)
        return R
    end

end
