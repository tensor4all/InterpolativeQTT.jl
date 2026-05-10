module InterpolativeQTT

export LagrangePolynomials, getChebyshevGrid
export interpolatesinglescale, interpolatemultiscale, interpolateadaptive
export interpolatesinglescale_sparse
export invertqtt, estimate_interpolation_error

import TensorCrossInterpolation as TCI
using LinearAlgebra

import Base: in

include("interval.jl")
include("interpolation.jl")
include("angular_lagrange.jl")

end
