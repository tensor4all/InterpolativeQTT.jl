# InterpolativeQTT.jl

**InterpolativeQTT.jl** implements the multiscale interpolative construction of Quantized Tensor Trains (QTTs) described in [Lindsey (2023)](https://arxiv.org/abs/2311.12554). 

## Content

- **Single-scale interpolation**
- **Multiscale interpolation**
- **Adaptive interpolation**
- **Sparse interpolation** 
- **Inversion**

## Installation

```julia
julia> ]
pkg> add InterpolativeQTT
```

## Quick start

```@example simple
import InterpolativeQTT
import TensorCrossInterpolation as TCI

f(x) = exp(-x^2)
K = 10
R = 8
tt = InterpolativeQTT.interpolatesinglescale(f, -2.0, 2.0, R, K)
println("Rank: ", TCI.rank(tt))
```

```@example simple
g(x) = x == 0.0 ? 0.0 : 1 / x
tt_ms = InterpolativeQTT.interpolatemultiscale(g, 0.0, 1.0, 12, 25, [0.0])
println("Rank: ", TCI.rank(tt_ms))
```

## Examples

The [Examples](examples.md) page walks through four worked examples:

| Example | Function | Method |
|---------|----------|--------|
| [Highly oscillatory function](examples.md#Highly-oscillatory-function-interpolation) | ``\cos(x^2) + \sin(\pi x)`` | `interpolatesinglescale` |
| [Narrow Gaussian](examples.md#Adaptive-interpolation-of-a-narrow-Gaussian) | ``e^{-x^2/(2\alpha^2)},\ \alpha=0.01`` | `interpolateadaptive` |
| [Lorentzian peak](examples.md#Sparse-construction-for-a-Lorentzian-peak) | ``\alpha/\sqrt{\alpha^2+(x-\tfrac{1}{2})^2}`` | `interpolatesinglescale_sparse` |
| [QTT inversion](examples.md#Inverting-a-QTT-to-a-multiresolution-Chebyshev-grid) | ``e^{-x^2+\cos x}`` | `invertqtt` |

## API reference

See the [API Reference](apireference.md) page for the full list of exported functions.
The main entry points are:

| Function | Description |
|----------|-------------|
| [`interpolatesinglescale`](@ref InterpolativeQTT.interpolatesinglescale) | Single-scale Chebyshev QTT interpolation |
| [`interpolatemultiscale`](@ref InterpolativeQTT.interpolatemultiscale) | Multiscale QTT with explicit singularities |
| [`interpolateadaptive`](@ref InterpolativeQTT.interpolateadaptive) | Adaptive refinement QTT |
| [`interpolatesinglescale_sparse`](@ref InterpolativeQTT.interpolatesinglescale_sparse) | Sparse (banded) single-scale QTT |
| [`invertqtt`](@ref InterpolativeQTT.invertqtt) | Invert a QTT to a multiresolution Chebyshev grid |
| [`getChebyshevGrid`](@ref InterpolativeQTT.getChebyshevGrid) | Chebyshev-Lobatto nodes and Lagrange basis |
