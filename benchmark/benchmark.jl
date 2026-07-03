using BenchmarkTools
using SpasityProbes
using ADTypes

const SUITE = BenchmarkGroup()
SUITE["chunked_detector"] = BenchmarkGroup()

function tridiagonal_test_function(x)
    y = similar(x)
    y[1] = x[1] * x[2]
    for i in 2:length(x)-1
        y[i] = x[i-1] + x[i]^2 + x[i+1]
    end
    y[end] = x[end-1] * x[end]
    return y
end

const N = 1000
const x = rand(N)

chunk_sizes = [10, 50, 100, 200, 500, 1000]

for cs in chunk_sizes
    detector = ChunkedDetector(cs)
    res = @benchmarkable ADTypes.jacobian_sparsity(
        tridiagonal_test_function, 
        $x, 
        $detector
    )
    SUITE["chunked_detector"]["chunk_size=$cs"] = res
end


if abspath(PROGRAM_FILE) == @__FILE__
    # Run the suite with a tuned configuration
    tune!(SUITE)
    results = run(SUITE, verbose=true)
    
    # Display the results neatly
    for cs in chunk_sizes
        println("\n--- Results for Chunk Size: $cs ---")
        show(stdout, MIME("text/plain"), results["jacobian_sparsity"]["chunk_size_$cs"])
        println()
    end
end