"""
    HierarchicalBloomFilterDetector(filter_size_m::Int, num_hashes_k::Int)

Like `BloomFilterDetector`, but refines the (over-approximated) Bloom pattern with a
second, color-seeded pass that removes false positives (Hovland 2026, Section 4).
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
"""
function _colorseed(colors::AbstractVector{<:Integer}, num_colors::Integer)
    n = length(colors)
    S = zeros(Bool, n, num_colors)
    for i in 1:n
        @inbounds S[i, colors[i]] = true
    end
    return S
end
