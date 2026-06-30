"""
    BloomFilterDetector(filter_size_m::Int, num_hashes_k::Int)

A detector configuration for `ADTypes.jacobian_sparsity` that uses a probabilistic 
Bloom filter approach to determine the sparsity pattern, reducing memory footprint.
Requires both `filter_size_m` and `num_hashes_k` to be strictly positive.
"""
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

"""
    ADTypes.jacobian_sparsity(f, x, detector::BloomFilterDetector)

Compute the Jacobian sparsity pattern of a function `f` at input `x` using 
a probabilistic Bloom filter strategy.
"""
function ADTypes.jacobian_sparsity(f, x, detector::BloomFilterDetector)
    return _jacobian_sparsity_bloom_filter(f, x, detector.filter_size_m, detector.num_hashes_k)
end

"""
    _jacobian_sparsity_bloom_filter(f, x, filter_size_m::Integer, num_hashes_k::Integer)

Internal logic to execute the probabilistic Bloom filter sparsity tracking.
"""
function _jacobian_sparsity_bloom_filter(f, x, filter_size_m::Integer, num_hashes_k::Integer)
    n = length(x)
    S = _bloomseed(n, num_hashes_k, filter_size_m)
    Q = _probe(f, S)
    R = _bloomharvest(Q, S, num_hashes_k)
    return R
end

"""
    _hash(i::Integer, j::Integer, filter_size_m::Integer)

Compute a hash value for index `i` using seed `j`, returning a 1-based 
index bounded by `filter_size_m` for Bloom filter assignments.
"""
function _hash(i::Integer, j::Integer, filter_size_m::Integer)
    h = xxh32([UInt32(i)], UInt32(j))
    return (h % filter_size_m) + 1
end

"""
    _bloomseed(n::Integer, num_hashes_k::Integer, filter_size_m::Integer)

Generate a Bloom filter seed matrix `S` mapping `n` inputs into a 
filter of size `filter_size_m`, using `num_hashes_k` hash functions.
"""
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

"""
    _probe(f, S::AbstractMatrix{Bool})

Evaluate the function `f` using sparse tracers initialized from the Bloom filter 
seed matrix `S`, returning the resulting observation matrix `Q`.
"""
function _probe(f, S::AbstractMatrix{Bool})
    n, filter_size_m = size(S)
    xt = Vector{DEFAULT_TRACER_TYPE}(undef, n)
    for i in 1:n
        xt[i] = DEFAULT_TRACER_TYPE(BitSet(findall(@view S[i, :])))
    end
    yt = f(xt)
    yt_array = to_array(yt)
    Q = zeros(Bool, length(yt_array), filter_size_m)
    for (row, y) in enumerate(yt_array)
        y isa DEFAULT_TRACER_TYPE || continue
        for column in getfield(y, :gradient)
            Q[row, column] = true
        end
    end
    return Q
end

"""
    _bloomharvest(Q::AbstractMatrix{Bool}, S::AbstractMatrix{Bool}, num_hashes_k::Integer)

Recover the Jacobian sparsity pattern by matching output observations `Q` 
against the input seed matrix `S`. An interaction is recorded only if all 
`num_hashes_k` bits match.
"""
function _bloomharvest(Q::AbstractMatrix{Bool}, S::AbstractMatrix{Bool}, num_hashes_k::Integer)
    W = Int.(Q) * transpose(Int.(S))
    required_matches = vec(sum(S; dims=2))
    return W .== transpose(required_matches)
end
