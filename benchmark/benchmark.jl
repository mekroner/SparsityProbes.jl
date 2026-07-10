using BenchmarkTools
using SparsityProbes
using ADTypes
using SparseConnectivityTracer

const SUITE = BenchmarkGroup()
SUITE["SparsityProbes"] = BenchmarkGroup()
SUITE["SparsityProbes"]["baseline"] = BenchmarkGroup()
SUITE["SparsityProbes"]["chunked"] = BenchmarkGroup()
SUITE["SparsityProbes"]["bloom"] = BenchmarkGroup()
SUITE["SparsityProbes"]["hierarchical_bloom"] = BenchmarkGroup()

function tridiagonal_test_function(x)
    y = similar(x)
    y[1] = x[1] * x[2]
    for i in 2:length(x)-1
        y[i] = x[i-1] + x[i]^2 + x[i+1]
    end
    y[end] = x[end-1] * x[end]
    return y
end

function benchmark_baseline(x::Vector{Float64})
    baseline_detector = TracerSparsityDetector()
    res = @benchmarkable ADTypes.jacobian_sparsity(
        tridiagonal_test_function, 
        $x, 
        $baseline_detector
    )
    SUITE["baseline"]["jacobian_sparsity"] = res
end

function benchmark_chunked_detector(chunk_sizes::Vector{Int}, x::Vector{Float64})
    for cs in chunk_sizes
        detector = ChunkedDetector(cs)
        res = @benchmarkable ADTypes.jacobian_sparsity(
            tridiagonal_test_function, 
            $x, 
            $detector
        )
        SUITE["chunked"]["cs=$cs"] = res
    end
end

function benchmark_bloom_filter_detectors(filter_size_m, num_hashes_k, x::Vector{Float64})
    for m in filter_size_m, k in num_hashes_k
        name = "m=$m,k=$k"
        bloom_detector = BloomFilterDetector(m, k)
        res = @benchmarkable ADTypes.jacobian_sparsity(
            tridiagonal_test_function, 
            $x, 
            $bloom_detector
        )
        SUITE["bloom"]["$name"] = res

        hierarchical_bloom_detector = BloomFilterDetector(m, k)
        res_hierarchical = @benchmarkable ADTypes.jacobian_sparsity(
            tridiagonal_test_function, 
            $x, 
            $hierarchical_bloom_detector
        )
        SUITE["hierarchical_bloom"]["$name"] = res_hierarchical
    end
end

const N = 1000
const x = rand(N)
chunk_sizes = [10, 50, 100, 200, 500, 1000]

benchmark_baseline(x)
benchmark_chunked_detector(chunk_sizes, x)
benchmark_bloom_filter_detectors([1000, 5000, 10000, 50000], [3, 5], x)


tune!(SUITE)
results = run(SUITE, verbose=true)

BenchmarkTools.save("benchmark/benchmark_results.json", results)
println("\nBenchmarks complete! Results saved to benchmark_results.json")

# if abspath(PROGRAM_FILE) == @__FILE__
#     # Run the suite with a tuned configuration
#     tune!(SUITE)
#     results = run(SUITE, verbose=true)
    
#     # Display the results neatly
#     for cs in chunk_sizes
#         println("\n--- Results for Chunk Size: $cs ---")
#         show(stdout, MIME("text/plain"), results["jacobian_sparsity"]["chunk_size_$cs"])
#         println()
#     end
# end