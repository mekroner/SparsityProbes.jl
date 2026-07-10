```@meta
CurrentModule = SparsityProbes
```

# SparsityProbes

Welcome to the documentation for [SparsityProbes.jl](https://github.com/mekroner/SparsityProbes.jl).
This package provides detector configurations for Jacobian sparsity detection through chunking and probabilistic Bloom filter strategies.

## Motivation

Automatic differentiation (AD) is a crucial method in modern computing, especially in machine learning applications. 
However, these computations can be vastly accelerated when Jacobians exhibit sparsity [ASDGuide](@cite). Despite its potential, automatic sparse differentiation (ASD) remains largely underutilized and is a topic of cutting-edge research. 
A critical prerequisite for unlocking these performance gains is the efficient determination of the exact Jacobian sparsity structure [Hovland2026](@cite). 
[SparsityProbes.jl](https://github.com/mekroner/SparsityProbes.jl) extends [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl) by introducing new topology discovery techniques, including probabilistic Bloom filters [Hovland2026](@cite). 
By minimizing the computational overhead of structure detection, this package eliminates redundant derivative calculations to directly accelerate downstream sparse AD pipelines for large-scale models.

## Installation

```julia
julia> ]add https://github.com/mekroner/SparsityProbes.jl
julia> using SparsityProbes
```

## Getting started
This package provides three detection algorithms, a `ChunkedDetector` that splits the work into chunks, a `BloomFilterDetector` that follows [Hovland2026](@cite), trading absolute precision for memory reduction, and `HierarchicalBloomFilterDetector` that dynamically refines the probabilistic estimation in a second pass.

### Chunked Detector
The `ChunkedDetector` splits the sparsity tracking into mutliple smaller blocks.
By splitting the input space into smaller, sequential chunks, it reduces the active memory footprint at the expense of running the tracer multiple times.
The `chunk_size` is a trade of between runtime and memory footprint, where small values cause low memory usage and high runtime. 
```@example
using SparsityProbes, ADTypes
f(x) = [x[1]^2 + x[2], x[2] * x[3], x[3] - x[1]]
x = [1.0, 2.0, 3.0]
detector = ChunkedDetector(2)
jacobian_sparsity(f, x, detector)
```


### Bloom Filter Detector
The `BloomFilterDetector` compresses the bit representation itself. 
It replaces the exact tracking with a fixed-length bit pattern of size `m` (`m << n`) using `k` hash functions.
This results in a fast, single-pass computation that contains a rate of false positives but guarantees zero false negatives.
The `filter_size_m` is a trade-off between memory footprint and accuracy, where small values cause low memory usage and a higher false-positive rate.
The `num_hashes_k` is a trade-off between hashing runtime and filter saturation, where small values lower computational overhead but large values risk prematurely filling the filter with ones and increasing false positives.

```@example
using SparsityProbes, ADTypes
f(x) = [x[1]^2 + x[2], x[2] * x[3], x[3] - x[1]]
x = [1.0, 2.0, 3.0]
detector = BloomFilterDetector(10, 2)
jacobian_sparsity(f, x, detector)
```

### Hierarchical Bloom Filter Detector
The `HierarchicalBloomFilterDetector` extends the Bloom filter method by adding a second refinemend pass.
It takes the initial over-approximation, colors the resulting graph topology, and re-seeds the tracer by color to reduce the false-positive rate.
This method uses the same parameters as the standart Bloom filter.

```@example
using SparsityProbes, ADTypes
f(x) = [x[1]^2 + x[2], x[2] * x[3], x[3] - x[1]]
x = [1.0, 2.0, 3.0]
detector = HierarchicalBloomFilterDetector(10, 2)
jacobian_sparsity(f, x, detector)
```
