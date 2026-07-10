struct ZeroBasedVector{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.IndexStyle(::Type{<:ZeroBasedVector}) = IndexLinear()
Base.size(x::ZeroBasedVector) = size(x.data)
Base.axes(x::ZeroBasedVector) = (0:length(x.data) - 1,)
Base.getindex(x::ZeroBasedVector, i::Int) = x.data[i + 1]
Base.eltype(::Type{ZeroBasedVector{T}}) where {T} = T

function cross_chunk_function(x)
    x1, x2, x3, x4, x5 = x
    y1 = x1 * x3 + x5
    y2 = x2 - x4
    y3 = x1 + x4 * x5
    y4 = 7.0
    return [y1, y2, y3, y4]
end

function mixed_dependency_function(x)
    x1, x2, x3, x4, x5 = x
    y1 = x1 + x2 * x4
    y2 = x3
    y3 = x2 * x5 + x1
    y4 = 0.0
    return [y1, y2, y3, y4]
end
