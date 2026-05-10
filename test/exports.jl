using Test
import InterpolativeQTT

@testset "exported symbols" begin
    exported = names(InterpolativeQTT)
    for sym in [
        :LagrangePolynomials, :getChebyshevGrid,
        :interpolatesinglescale, :interpolatemultiscale, :interpolateadaptive,
        :interpolatesinglescale_sparse, :invertqtt, :estimate_interpolation_error,
    ]
        @test sym ∈ exported
    end
    for sym in [:Interval, :NInterval, :midpoint, :intervallength, :angular_local_lagrange]
        @test sym ∉ exported
    end
end
