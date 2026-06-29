```@meta
CurrentModule = SparsityProbes
```

# SparsityProbes

Welcome to the documentation for [SparsityProbes](https://github.com/mekroner/SparsityProbes.jl).
This package provides detector configurations for Jacobian sparsity detection through chunking and probabilistic Bloom filter strategies.

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

```@docs
SparsityProbes.jacobian_sparsity
```

```@docs
SparsityProbes.ChunkedDetector
```

```@docs
SparsityProbes.BloomFilterDetector
```

```@docs
SparsityProbes.HierarchicalBloomFilterDetector
```

### Internals

```@docs
SparsityProbes._create_chunks
```

```@docs
SparsityProbes._trace_input_chunk
```

```@docs
SparsityProbes._combine_patterns
```
