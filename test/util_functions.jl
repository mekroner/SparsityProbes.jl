function cross_chunk_function(x)
    y1 = x[1] * x[3] + x[5]
    y2 = x[2] - x[4]
    y3 = x[1] + x[4] * x[5]
    y4 = 7.0
    return [y1, y2, y3, y4]
end

function mixed_dependency_function(x)
    y1 = x[1] + x[2] * x[4]
    y2 = x[3]
    y3 = x[2] * x[5] + x[1]
    y4 = 0.0
    return [y1, y2, y3, y4]
end
