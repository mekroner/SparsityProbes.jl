```@meta
CurrentModule = SparsityProbes
```

# SparsityProbes

Welcome to the documentation for [SparsityProbes](https://github.com/mekroner/SparsityProbes.jl).
This package provides advanced detector configurations to compute Jacobian sparsity patterns, through chunking and probabilistic Bloom filter strategies.


## Installation

You can install SparsityProbes.jl using Julia's built-in package manager. 
Open the Julia REPL, type ] to enter Pkg mode, and run:
```julia
pkg> add SparsityProbes
```

## Quick Start: Chunked Detector
The `ChunkedDetector` splits the sparsity tracking into manageable blocks, which is excellent for strict memory limits.
```@example
using SparsityProbes, ADTypes
f(x) = [x[1]^2 + x[2], x[2] * x[3], x[3] - x[1]];
x = [1.0, 2.0, 3.0];
detector = ChunkedDetector(2);
jacobian_sparsity(f, x, detector)
```

## Quick Start: Bloom Filter Detector
The `BloomFilterDetector` uses a probabilistic hashing approach. 
Define the filter size (`m`) and the number of hash functions (`k`). 
It is effective for extremely large, sparse systems.

```@example
using SparsityProbes, ADTypes
f(x) = [x[1]^2 + x[2], x[2] * x[3], x[3] - x[1]];
x = [1.0, 2.0, 3.0];
bloom_detector = BloomFilterDetector(10, 2);
jacobian_sparsity(f, x, bloom_detector)
``` 

## API Reference
```@index
```

```@autodocs
Modules = [SparsityProbes]
```
