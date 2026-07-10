using ADTypes: jacobian_sparsity
using SparsityProbes:
    BloomFilterDetector,
    ChunkedDetector,
    HierarchicalBloomFilterDetector
using Test
using Random: MersenneTwister
using SparseConnectivityTracer: TracerSparsityDetector
using SparseArrays: SparseMatrixCSC, sprand

# Fuzzy tests for all detectors, to ensure they don't crash on random inputs.

struct RandomSparseNetwork
    dependency_matrix::SparseMatrixCSC{Float64, Int}
end

# Make struct callable
function (rsn::RandomSparseNetwork)(x::AbstractVector)
    W = rsn.dependency_matrix
    return sin.(W * x) .^ cos.(W * (x .^ 2))
end

# Allows us the bloom filters to over-approximate the sparsity pattern, but not under-approximate it.
function equal_or_over_approximation(A::AbstractMatrix{Bool}, B::AbstractMatrix{Bool})
    return size(A) == size(B) && all(A .>= B)
end

function fuzzy_test_detector(detector; n_tests::Int=1000, max_inputs::Int=100, max_outputs::Int=100)
    rng = MersenneTwister(42)
    
    @testset "Fuzzy Tests for $(typeof(detector)): $n_tests iterations" begin
        for i in 1:n_tests
            # Random dimension 
            n_inputs = rand(rng, 1:max_inputs)
            n_outputs = rand(rng, 1:max_outputs)

            # random density between 1% and 50%
            density = rand(rng, 0.01:0.01:0.5) 
            W = sprand(rng, n_outputs, n_inputs, density)
            f_rand = RandomSparseNetwork(W)
            x = randn(rng, n_inputs)

            expected = jacobian_sparsity(f_rand, x, TracerSparsityDetector())
            got = jacobian_sparsity(f_rand, x, detector)

            @test size(got) == size(expected)
            @test equal_or_over_approximation(got, expected)
        end
    end
end

@testset "Fuzzy Tests" begin
    fuzzy_test_detector(HierarchicalBloomFilterDetector(128, 3); n_tests=1000)
    fuzzy_test_detector(BloomFilterDetector(128, 3); n_tests=1000)
    fuzzy_test_detector(ChunkedDetector(10); n_tests=1000)
end
