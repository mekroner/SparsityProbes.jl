module SparsityProbes

    using ADTypes: ADTypes, jacobian_sparsity
    using SparseConnectivityTracer: GradientTracer, myempty, jacobian_tracers_to_matrix, to_array
    using XXhash: xxh32
    
    const T = GradientTracer{Int, BitSet}

    export jacobian_sparsity, ChunkedDetector, BloomFilterDetector, HierarchicalBloomFilterDetector

    include("chunked_detector.jl")
    include("bloom_filter_detector.jl")
    include("hierarchical_bloom_filter_detector.jl")

end # module
