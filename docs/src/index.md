```@meta
CurrentModule = SparsityProbes
```

# SparsityProbes

Welcome to the documentation for [SparsityProbes](https://github.com/mekroner/SparsityProbes.jl).
This package provides detector configurations for Jacobian sparsity detection through chunking and probabilistic Bloom filter strategies.

## Motivation
Automatic differentiation (AD) is a crucial method in modern computing, especially in machine learning applications. However, these computations can be vastly accelerated when Jacobians exhibit sparsity [ASDGuide](@cite). Despite its potential, automatic sparse differentiation (ASD) remains largely underutilized and is a topic of cutting-edge research. A critical prerequisite for unlocking these performance gains is the efficient determination of the exact Jacobian sparsity structure [Hovland2026](@cite). `SparsityProbes.jl` extends [`SparseConnectivityTracer.jl`](https://github.com/adrhill/SparseConnectivityTracer.jl) by introducing new topology discovery techniques, including probabilistic Bloom filters [Hovland2026](@cite). By minimizing the computational overhead of structure detection, this package eliminates redundant derivative calculations to directly accelerate downstream sparse AD pipelines for large-scale models.

## Installation

```julia
julia> ]add https://github.com/mekroner/SparsityProbes.jl
julia> using SparsityProbes
```

## Getting started

The `ChunkedDetector` splits the sparsity tracking into manageable blocks and is a good fit when memory usage matters.

```@example
using SparsityProbes, ADTypes
f(x) = [x[1]^2 + x[2], x[2] * x[3], x[3] - x[1]]
x = [1.0, 2.0, 3.0]
detector = ChunkedDetector(2)
jacobian_sparsity(f, x, detector)
```

The package also provides probabilistic detectors for large systems where a small amount of over-approximation is acceptable.

```@example
using SparsityProbes, ADTypes
f(x) = [x[1]^2 + x[2], x[2] * x[3], x[3] - x[1]]
x = [1.0, 2.0, 3.0]
bloom_detector = BloomFilterDetector(10, 2)
jacobian_sparsity(f, x, bloom_detector)
```

## API Reference

### Public API

```@autodocs
Modules = [SparsityProbes]
Private = false
Public = true
```


### Internals

```@autodocs
Modules = [SparsityProbes]
Private = true
Public = false
```