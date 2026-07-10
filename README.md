# SparsityProbes.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mekroner.github.io/SparsityProbes.jl/dev/)
[![Build Status](https://github.com/mekroner/SparsityProbes.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/mekroner/SparsityProbes.jl/actions/workflows/test.yaml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mekroner/SparsityProbes.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mekroner/SparsityProbes.jl)

`SparsityProbes` extends the `ADTypes` interface to compute Jacobian sparsity patterns in isolated chunks. By dividing the input array into subsets and tracing them individually using `SparseConnectivityTracer`, it allows for piece-wise sparsity detection.

## Installation

```julia
julia> ]add https://github.com/mekroner/SparsityProbes.jl
julia> using SparsityProbes
```

## Usage

This package provides three detection algorithms, a `ChunkedDetector` that splits the work into chunks, a `BloomFilterDetector`, trading absolute precision for memory reduction, and `HierarchicalBloomFilterDetector` that dynamically refines the probabilistic estimation of the Bloom filter in a second pass.
To calculate the Jacobian sparsity of a function, use the `ChunkedDetector` alongside `ADTypes.jacobian_sparsity`.

### Basic Example

```julia
using ADTypes: jacobian_sparsity
using SparsityProbes: ChunkedDetector, BloomFilterDetector

# Define your target function
f(x) = [x[1]^2 + x[2], x[2] * x[3], x[3] - x[1]]

# Define your input array
x = [1.0, 2.0, 3.0]

# Initialize the detector with a specific `chunk_size` (e.g., 2)
detector = ChunkedDetector(2)

# Compute the sparsity pattern matrix
jacobian_sparsity(f, x, detector)
# Produces a sparse matrix with 6 stored entries:
# 1  1  ⋅
# ⋅  1  1
# 1  ⋅  1

# A BloomFilter can be used with `filter_size`(10) and the number of independent hash functions `num_hashes_k`(2).
bloom_detector = BloomFilterDetector(10, 2);
jacobian_sparsity(f, x, bloom_detector)
# Produces a 3x3 BitMatrix:
# 1  1  0
# 0  1  1
# 1  0  1
```
