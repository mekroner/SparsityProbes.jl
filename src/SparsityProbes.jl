module SparsityProbes

using ADTypes: ADTypes, jacobian_sparsity
using SparseConnectivityTracer: GradientTracer, myempty, jacobian_tracers_to_matrix, to_array
using XXhash: xxh32

const DEFAULT_TRACER_TYPE = GradientTracer{Int, BitSet}

export jacobian_sparsity, ChunkedDetector, BloomFilterDetector, HierarchicalBloomFilterDetector

@inline function _require_one_based_array(x)
    x isa AbstractArray && Base.require_one_based_indexing(x)
    return x
end

include("chunked_detector.jl")
include("bloom_filter_detector.jl")
include("hierarchical_bloom_filter_detector.jl")

    function f(x)
        return x .^2
    end
    
end
