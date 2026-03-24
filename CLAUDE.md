# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolynomialQTT.jl is a Julia package implementing multiscale polynomial interpolation for Quantized Tensor Trains (QTT). It provides efficient tensor-train representation of functions, especially those with singularities or high oscillation. See [arXiv:2311.12554](https://arxiv.org/abs/2311.12554) for the underlying algorithm.

## Commands

```bash
# Install/update dependencies
julia -e 'using Pkg; Pkg.instantiate()'

# Run all tests
julia -e 'using Pkg; Pkg.test()'

# Run tests directly
julia test/runtests.jl

# Run a single test file
julia -e 'include("test/test_interpolation.jl")'

# Build documentation
julia docs/make.jl
```

CI uses a custom Julia registry that must be added for local development:
```julia
using Pkg
pkg"registry add https://github.com/tensor4all/T4ARegistry.git"
```

## Architecture

The package has three source files:

**[src/interval.jl](src/interval.jl)** — `Interval{V}` (1D) and `NInterval{N,V}` (N-dimensional) types with `split`, `midpoint`, and membership operations. These represent the recursive subdivision domains used in multiscale methods.

**[src/interpolation.jl](src/interpolation.jl)** — All core algorithms:
- `LagrangePolynomials` struct with barycentric weights for stable evaluation
- `interpolatesinglescale` — Chebyshev-node polynomial interpolation on a uniform QTT grid. Constructs tensor train cores via Lagrange interpolation; multi-dimensional case uses a "fused unfolding" scheme to build Kronecker-product cores.
- `interpolatemultiscale` — Recursive subdivision at known `cusplocations`; assembles separate QTT cores per resolution level.
- `interpolateadaptive` — Detects "dangerous" intervals via `estimate_interpolation_error` / `detect_dangerous_intervals!`, then drives `interpolatemultiscale` automatically.
- Internal helpers: `_compress_train` (SVD truncation), `_direct_product_coretensors` (Kronecker for N-D), `_evalf` (grid evaluation).

**[src/PolynomialQTT.jl](src/PolynomialQTT.jl)** — Module entry point; includes the two files above and declares exports.

### QTT representation

Functions are stored as tensor trains where each core has shape `(left_bond, 2^N, right_bond)` — the physical dimension `2^N` encodes N-dimensional local structure. Bond dimensions are controlled by SVD compression with a user-supplied `tolerance`.

### Key dependency

`TensorCrossInterpolation.jl` provides the tensor train data structure and SVD/compression utilities used throughout. It must be installed via T4ARegistry (not the General registry).

## Current Branch

The `adaptive` branch adds `interpolateadaptive` and is the active development branch. Main is stable/released.
