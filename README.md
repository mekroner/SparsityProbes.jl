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

To calculate the Jacobian sparsity of a function, use the `ChunkedDetector` alongside `ADTypes.jacobian_sparsity`.

### Basic Example

```julia
using ADTypes: jacobian_sparsity
using SparsityProbes: ChunkedDetector

# Define your target function
function toy_function(x)
    y1 = x[1] * x[2]
    y2 = x[2] + 0.0
    return [y1, y2]
end

# Define your input array
x = [10.0, 20.0, 30.0, 40.0]

# Initialize the detector with a specific chunk size (e.g., 2)
detector = ChunkedDetector(2)

# Compute the sparsity pattern matrix
sparsity_pattern = jacobian_sparsity(toy_function, x, detector)
```