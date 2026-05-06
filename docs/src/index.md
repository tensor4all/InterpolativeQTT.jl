# InterpolativeQTT.jl

This is the documentation for [InterpolativeQTT.jl](https://github.com/tensor4all/InterpolativeQTT.jl). The code is based on the paper [Multiscale interpolative construction of quantized tensor trains](https://arxiv.org/abs/2311.12554) by  Michael Lindsey. 

## Installation

To install InterpolativeQTT.jl enter the following commands:

```julia
julia 
] 
add https://github.com/tensor4all/InterpolativeQTT.jl
```

### Usage Example

```@example simple
import TensorCrossInterpolation as TCI
import InterpolativeQTT

# Single-scale interpolation
f(x) = exp(-x^2)
tt = InterpolativeQTT.interpolatesinglescale(f, -2.0, 2.0, 8, 20) # TensorCrossInterpolation.TensorTrain{Float64, 3}

# Multi-scale interpolation with singularity
g(x) = x == 0.0 ? 0.0 : 1/x
tt_multiscale = InterpolativeQTT.interpolatemultiscale(g, 0.0, 1.0, 12, 25, [0.0])
```