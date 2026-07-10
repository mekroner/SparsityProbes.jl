"""
    HierarchicalBloomFilterDetector(filter_size_m::Int, num_hashes_k::Int)

Like `BloomFilterDetector`, but refines the (over-approximated) Bloom pattern with a
second, color-seeded pass that removes false positives ([Hovland2026](@cite), Section 4).

# Arguments
- `filter_size_m::Int`: The size of the Bloom filter bit array.
- `num_hashes_k::Int`: The number of independent hash functions to use.

# Returns
- `HierarchicalBloomFilterDetector`: An instance of the hierarchical detector configuration.
"""
struct HierarchicalBloomFilterDetector
    filter_size_m::Int
    num_hashes_k::Int

    function HierarchicalBloomFilterDetector(filter_size_m::Int, num_hashes_k::Int)
        filter_size_m > 0 ||
            throw(ArgumentError("HierarchicalBloomFilterDetector requires filter_size_m > 0"))
        num_hashes_k > 0 ||
            throw(ArgumentError("HierarchicalBloomFilterDetector requires num_hashes_k > 0"))
        return new(filter_size_m, num_hashes_k)
    end
end

"""
    ADTypes.jacobian_sparsity(f, x, detector::HierarchicalBloomFilterDetector)

Compute the Jacobian sparsity pattern of a function `f` at input `x` using 
a multi-level hierarchical Bloom filter strategy to eliminate false positives.

# Arguments
- `f`: The target function whose Jacobian sparsity is being analyzed.
- `x`: The input argument where the sparsity is evaluated.
- `detector::HierarchicalBloomFilterDetector`: The hierarchical Bloom configuration.

# Returns
- `AbstractMatrix{Bool}`: The final, refined Jacobian sparsity pattern matrix.
"""
function ADTypes.jacobian_sparsity(f, x, detector::HierarchicalBloomFilterDetector)
    return _jacobian_sparsity_hierarchical_bloom(
        f, x, detector.filter_size_m, detector.num_hashes_k,
    )
end

"""
    _jacobian_sparsity_hierarchical_bloom(f, x, filter_size_m, num_hashes_k)

Two-level detection:
1. a coarse Bloom pass yields an over-approximation `Rbar` (a superset of the true
   pattern: never misses an entry, but may contain false positives);
2. the columns of `Rbar` are colored so that two inputs sharing an output row get
   different colors; a second pass re-seeds each input with a single bit at its color
   and keeps a candidate `(output, input)` only if that color is observed in the output.

    # Arguments
- `f`: The target function.
- `x`: The input argument array.
- `filter_size_m::Integer`: The size of the underlying Bloom filter bit array.
- `num_hashes_k::Integer`: The number of hash functions for the first-level pass.

# Returns
- `AbstractMatrix{Bool}`: The exact sparsity matrix after filtering out false positives.
"""
function _jacobian_sparsity_hierarchical_bloom(f, x, filter_size_m::Integer, num_hashes_k::Integer)
    n = length(x)

    # level 1:
    # coarse Bloom pass -> over-approximation
    S = _bloomseed(n, num_hashes_k, filter_size_m)
    Q = _probe(f, S)
    Rbar = _bloomharvest(Q, S, num_hashes_k)

    # color the over-approximation
    colors = _color_columns(Rbar)
    num_colors = maximum(colors; init = 0)
    num_colors == 0 && return Rbar # no inputs seeded

    # level 2:
    # re-seed by color (exact, one bit per input)
    S_color = _colorseed(colors, num_colors)
    Q_color = _probe(f, S_color)
    confirmed = _bloomharvest(Q_color, S_color, 1) # color(input) ∈ colors(output)?

    # keep only candidates whose color was confirmed for that output.
    return Rbar .& confirmed
end

"""
    _color_columns(P::AbstractMatrix{Bool}) -> Vector{Int}

Greedy distance-1 coloring of the column-intersection graph of `P`
(rows = outputs, columns = inputs). Two columns are adjacent iff they share a `true`
entry in some row; adjacent columns get different colors. Colors are `1:num_colors`.

# Arguments
- `P::AbstractMatrix{Bool}`: The over-approximated sparsity pattern matrix to color.

# Returns
- `Vector{Int}`: A vector assigning a color ID to each column index.
"""
function _color_columns(P::AbstractMatrix{Bool})
    num_outputs, n = size(P)
    colors = zeros(Int, n)
    neighbor_colors = BitSet()
    for i in 1:n
        empty!(neighbor_colors)
        for j in 1:num_outputs
            @inbounds P[j, i] || continue
            for i2 in 1:(i - 1)
                @inbounds P[j, i2] && push!(neighbor_colors, colors[i2])
            end
        end
        ci = 1
        while ci in neighbor_colors
            ci += 1
        end
        colors[i] = ci
    end
    return colors
end

"""
    _colorseed(colors::AbstractVector{<:Integer}, num_colors::Integer) -> Matrix{Bool}

Seed matrix for the color pass: row `i` has a single `true` at column `colors[i]`,
so each input carries the exact singleton set `{color(i)}` (no hashing, no collisions).

# Arguments
- `colors::AbstractVector{<:Integer}`: The color assignments for each input index.
- `num_colors::Integer`: The total number of unique colors used.

# Returns
- `Matrix{Bool}`: An `n × num_colors` seed matrix for the secondary color trace.
"""
function _colorseed(colors::AbstractVector{<:Integer}, num_colors::Integer)
    n = length(colors)
    S = zeros(Bool, n, num_colors)
    for i in 1:n
        @inbounds S[i, colors[i]] = true
    end
    return S
end
