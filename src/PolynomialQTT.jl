module PolynomialQTT

import TensorCrossInterpolation as TCI
using LinearAlgebra

import Base: in

include("interval.jl")
include("interpolation.jl")
include("angular_lagrange.jl")

end
