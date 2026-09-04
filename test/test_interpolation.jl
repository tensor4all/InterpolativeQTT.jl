using Test
import InterpolativeQTT
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI
import LinearAlgebra: norm

@testset "single-scale interpolation" begin
    R = 4
    a, b = -2.8, float(pi)
    f(x) = exp(-x^2)
    K = 20

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, K)
    @test TCI.rank(tt) <= K + 1

    grid = QG.DiscretizedGrid{1}(R, a, b)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:(2^R))
    xs = QG.grididx_to_origcoord.(Ref(grid), 1:(2^R))
    origdata = f.(xs)
    ttdata = tt.(quanticsinds)
    @test all(abs.(ttdata .- origdata) .< 1.0e-10)
end

@testset "single-scale interpolation (N=1)" begin
    R = 4
    a, b = -2.8, float(pi)
    f(x) = exp(-x^2)
    K = 20

    tt = InterpolativeQTT.interpolatesinglescale(f, (a,), (b,), R, K)
    @test TCI.rank(tt) <= K + 1

    grid = QG.DiscretizedGrid{1}(R, a, b)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:(2^R))
    xs = QG.grididx_to_origcoord.(Ref(grid), 1:(2^R))
    origdata = f.(xs)
    ttdata = tt.(quanticsinds)
    @test all(abs.(ttdata .- origdata) .< 1.0e-10)
end


@testset "single-scale interpolation (N=2)" begin
    R = 4
    K = 20
    a, b = (-1.0, -1.0), (2.0, 2.0)
    f(x, y) = exp(-x^2 - y^3)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, K)
    @test TCI.rank(tt) <= (K + 1)^2
    @test length(tt) == R

    grid = QG.DiscretizedGrid{2}(R, a, b; unfoldingscheme = :fused)

    quanticsinds = [QG.grididx_to_quantics(grid, (i, j)) for i in 1:(2^R), j in 1:(2^R)]
    xs = [QG.grididx_to_origcoord(grid, (i, j)) for i in 1:(2^R), j in 1:(2^R)]

    origdata = [f(x...) for x in xs]
    ttdata = tt.(quanticsinds)

    @test all(abs.(ttdata .- origdata) .< 1.0e-10)
end


@testset "single-scale interpolation (N=3)" begin
    R = 4
    K = 15
    a, b = (-1.0, -1.0, 0.0), (2.0, 2.0, 1.0)
    f(x, y, z) = exp(-x^2 - y^3 - 2 * z^2)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, K)
    @test TCI.rank(tt) <= (K + 1)^3
    @test length(tt) == R

    grid = QG.DiscretizedGrid{3}(R, a, b; unfoldingscheme = :fused)

    quanticsinds = [QG.grididx_to_quantics(grid, (i, j, k)) for i in 1:(2^R), j in 1:(2^R), k in 1:(2^R)]
    xs = [QG.grididx_to_origcoord(grid, (i, j, k)) for i in 1:(2^R), j in 1:(2^R), k in 1:(2^R)]

    origdata = [f(x...) for x in xs]
    ttdata = tt.(quanticsinds)

    @test all(abs.(ttdata .- origdata) .< 1.0e-7)
end


@testset "multiscale interpolation" begin
    R = 4
    a, b = -2.0, sqrt(2)
    f(x) = exp(-x^2) + abs(x)
    K = 25

    tt = InterpolativeQTT.interpolatemultiscale(f, a, b, R, K, Float64[0])
    @test TCI.rank(tt) <= K + 2

    grid = QG.DiscretizedGrid{1}(R, a, b)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:(2^R))
    xs = QG.grididx_to_origcoord.(Ref(grid), 1:(2^R))
    origdata = f.(xs)
    ttdata = tt.(quanticsinds)
    @test all(abs.(ttdata .- origdata) .< 1.0e-12)
end


@testset "multiscale interpolation (N=2)" begin
    R = 4
    N = 2

    a, b = (0.0, 0.0), (1.0, 1.0)
    f(x, y) = exp(-x^2 - 2 * y^2)
    K = 10

    tt = InterpolativeQTT.interpolatemultiscale(f, a, b, R, K, [(0.0, 0.0)])

    @test TCI.rank(tt) <= (K + 2)^N

    grid = QG.DiscretizedGrid{N}(R, a, b)

    quanticsinds = [QG.grididx_to_quantics(grid, (i, j)) for i in 1:(2^R), j in 1:(2^R)]
    xs = [QG.grididx_to_origcoord(grid, (i, j)) for i in 1:(2^R), j in 1:(2^R)]

    origdata = [f(x...) for x in xs]
    ttdata = tt.(quanticsinds)

    @test maximum(abs, ttdata .- origdata) < 1.0e-10
end

@testset "NInterval" begin
    @testset "split" begin

        interval = InterpolativeQTT.NInterval{2, Float64}((-1.0, -1.0), (1.0, 1.0))

        @test InterpolativeQTT.split(interval) == [
            InterpolativeQTT.NInterval{2, Float64}((-1.0, -1.0), (0.0, 0.0)),
            InterpolativeQTT.NInterval{2, Float64}((0.0, -1.0), (1.0, 0.0)),
            InterpolativeQTT.NInterval{2, Float64}((-1.0, 0.0), (0.0, 1.0)),
            InterpolativeQTT.NInterval{2, Float64}((0.0, 0.0), (1.0, 1.0)),
        ]
    end
end

@testset "_direct_product_coretensors (two tensors)" begin
    χ = 3
    coretensors = [randn(χ, 2, χ) for _ in 1:2]
    c12 = InterpolativeQTT._direct_product_coretensors(coretensors)
    c12_ref = Array{Float64, 6}(undef, χ, χ, 2, 2, χ, χ)
    for i in 1:χ, j in 1:χ, k in 1:2, l in 1:2, m in 1:χ, n in 1:χ
        c12_ref[i, j, k, l, m, n] = coretensors[1][i, k, m] * coretensors[2][j, l, n]
    end
    @test vec(c12) ≈ vec(c12_ref)
end


@testset "_direct_product_coretensors (three tensors)" begin
    χ = 3
    coretensors = [randn(χ, 2, χ) for _ in 1:3]
    c123 = InterpolativeQTT._direct_product_coretensors(coretensors)
    c123_ref = Array{Float64, 9}(undef, χ, χ, χ, 2, 2, 2, χ, χ, χ)
    for i in 1:χ, j in 1:χ, k in 1:χ, l in 1:2, m in 1:2, n in 1:2, o in 1:χ, p in 1:χ, q in 1:χ
        c123_ref[i, j, k, l, m, n, o, p, q] =
            coretensors[1][i, l, o] * coretensors[2][j, m, p] * coretensors[3][k, n, q]
    end
    @test vec(c123) ≈ vec(c123_ref)
end

@testset "high-degree Chebyshev basis stays finite" begin
    K = 600
    P = InterpolativeQTT.getChebyshevGrid(K)

    @test length(P.grid) == K + 1
    @test all(isfinite, P.baryweights)
    @test P(0, P.grid[1]) ≈ 1.0
    @test P(1, P.grid[1]) ≈ 0.0 atol = 1.0e-12

    A = InterpolativeQTT.interpolationtensor(P)
    @test all(isfinite, A)
end

@testset "high-degree single-scale interpolation remains accurate" begin
    R = 3
    K = 600
    f(x) = sin(2 * π * x)

    tt = InterpolativeQTT.interpolatesinglescale(f, 0.0, 1.0, R, K)
    grid = QG.DiscretizedGrid{1}(R, 0.0, 1.0)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:(2^R))
    xs = QG.grididx_to_origcoord.(Ref(grid), 1:(2^R))
    maxerr = maximum(abs.(tt.(quanticsinds) .- f.(xs)))

    @test maxerr < 1.0e-10
end


@testset "multiscale interpolation (1/x)" begin
    R = 12
    a, b = 0.0, 1.0
    f(x) = x == 0.0 ? 0.0 : 1 / x
    K = 25

    tt = InterpolativeQTT.interpolatemultiscale(f, a, b, R, K, Float64[0], tolerance = 1.0e-12)
    @test TCI.rank(tt) <= K + 2

    grid = QG.DiscretizedGrid{1}(R, a, b)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:(2^R))
    xs = QG.grididx_to_origcoord.(Ref(grid), 1:(2^R))
    origdata = f.(xs)
    ttdata = tt.(quanticsinds)

    ref = origdata
    diff = abs.(ttdata .- origdata)
    @test norm(diff) / norm(ref) < 1.0e-11
end

@testset "estimate_interpolation_error (N=1)" begin
    a, b = 0.0, 1.0
    f(x) = sin(2 * π * x)
    K = 5
    P = InterpolativeQTT.getChebyshevGrid(K)
    interval = InterpolativeQTT.Interval{Float64}(a, b)

    err = InterpolativeQTT.estimate_interpolation_error(f, interval, P)
    @test err >= 0.0
    @test err < 1.0
end

@testset "estimate_interpolation_error (N=2)" begin
    a, b = (-1.0, -1.0), (1.0, 1.0)
    f(x, y) = sin(π * x) * cos(π * y)
    K = 5
    P = InterpolativeQTT.getChebyshevGrid(K)
    interval = InterpolativeQTT.NInterval{2, Float64}(a, b)

    err = InterpolativeQTT.estimate_interpolation_error(f, interval, P)
    @test err >= 0.0
    @test err < 1.0
end

@testset "estimate_interpolation_error (N=3)" begin
    a, b = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    f(x, y, z) = exp(-x^2 - y^2 - z^2)
    K = 10
    P = InterpolativeQTT.getChebyshevGrid(K)
    interval = InterpolativeQTT.NInterval{3, Float64}(a, b)

    err = InterpolativeQTT.estimate_interpolation_error(f, interval, P)
    @test err >= 0.0
    @test err < 1.0e-8
end

@testset "invertqtt (uncompressed, Stage 1)" begin
    R = 12
    N = 10
    a, b = 0.0, 1.0
    f(x) = exp(-x^2)
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = 0.0)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    K_out = R - 1
    S = result[K_out]
    @test size(S, 1) == 2^K_out
    @test size(S, 2) == N + 1

    max_err = 0.0
    for i in 1:(2^K_out)
        for β in 0:N
            x_ref = a + (b - a) * (i - 1 + P.grid[β + 1]) / 2^K_out
            max_err = max(max_err, abs(S[i, β + 1] - f(x_ref)))
        end
    end
    @test max_err < 1.0e-6
end

@testset "invertqtt (compressed)" begin
    R = 12
    N = 10
    a, b = 0.0, 1.0
    f(x) = exp(-x^2)
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = 1.0e-10)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    K_out = R - 1
    S = result[K_out]
    max_err = 0.0
    for i in 1:(2^K_out)
        for β in 0:N
            x_ref = a + (b - a) * (i - 1 + P.grid[β + 1]) / 2^K_out
            max_err = max(max_err, abs(S[i, β + 1] - f(x_ref)))
        end
    end
    @test max_err < 1.0e-6
end

@testset "invertqtt round trip (uncompressed)" begin
    R = 8
    N = 10
    a, b = 0.0, 1.0
    f(x) = exp(-x^2)
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = 0.0)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    K_out = R - 1
    S = result[K_out]

    grid = QG.DiscretizedGrid{1}(R, a, b)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:(2^R))
    ttvals = tt.(quanticsinds)

    max_err = 0.0
    for i in 1:(2^K_out)
        v_left = sum(P(β, 0.0) * S[i, β + 1] for β in 0:N)
        v_right = sum(P(β, 0.5) * S[i, β + 1] for β in 0:N)
        max_err = max(max_err, abs(v_left - ttvals[2i - 1]))
        max_err = max(max_err, abs(v_right - ttvals[2i]))
    end
    @test max_err < 1.0e-11
end

@testset "invertqtt round trip (compressed)" begin
    R = 10
    N = 10
    a, b = 0.0, 1.0
    f(x) = exp(-x^2)
    tol = 1.0e-8
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = tol)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    K_out = R - 1
    S = result[K_out]

    grid = QG.DiscretizedGrid{1}(R, a, b)
    quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:(2^R))
    ttvals = tt.(quanticsinds)

    max_err = 0.0
    for i in 1:(2^K_out)
        v_left = sum(P(β, 0.0) * S[i, β + 1] for β in 0:N)
        v_right = sum(P(β, 0.5) * S[i, β + 1] for β in 0:N)
        max_err = max(max_err, abs(v_left - ttvals[2i - 1]))
        max_err = max(max_err, abs(v_right - ttvals[2i]))
    end
    @test max_err < 1.0e-6
end

# Fused multivariate QTT uses Z-order (Morton) cell indexing: at each resolution
# level k, all N bits σ_{k,n} are packed into one physical index as
# σ_flat = σ_{k,1} + 2*σ_{k,2} + ... (dim 1 fastest, column-major within a level).
# The flat row of S after K_out levels is therefore a bit-interleaved Morton code.
function _morton_cell_flat(cell_ixs, K_out)
    ndims = length(cell_ixs)
    row = 0
    for p in 0:(K_out - 1)
        for n in 1:ndims
            bit = (cell_ixs[n] >> p) & 1
            row |= bit << (ndims * p + (n - 1))
        end
    end
    return row + 1  # 1-indexed
end

@testset "invertqtt 2D (uncompressed, Stage 1 values)" begin
    # Use a bilinear function f(x,y) = x*y so that the bilinear Lagrange Stage 1
    # reconstruction is algebraically exact — this precisely validates the Morton
    # cell ordering and the L_mv Kronecker weight structure.
    R = 6
    N = 6
    a, b = (0.0, 0.0), (1.0, 1.0)
    f(x, y) = x * y
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = 0.0)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    K_out = R - 1
    S = result[K_out]
    @test size(S, 1) == 4^K_out        # (2^2)^{K_out} sub-cells
    @test size(S, 2) == (N + 1)^2      # (N+1)^2 Chebyshev nodes

    cells_per_dim = 2^K_out
    h = 1.0 / cells_per_dim
    max_err = 0.0
    for ix in 0:min(7, cells_per_dim - 1), iy in 0:min(7, cells_per_dim - 1)
        cell_flat = _morton_cell_flat((ix, iy), K_out)
        for βx in 0:N, βy in 0:N
            β_flat = βx + 1 + (N + 1) * βy
            x_ref = a[1] + (ix + P.grid[βx + 1]) * h
            y_ref = a[2] + (iy + P.grid[βy + 1]) * h
            max_err = max(max_err, abs(S[cell_flat, β_flat] - f(x_ref, y_ref)))
        end
    end
    @test max_err < 1.0e-10
end

@testset "invertqtt 2D (compressed)" begin
    # Round-trip test on a compressed QTT: the Chebyshev interpolation at the
    # fine-level dyadic positions must reproduce the (compressed) QTT values exactly
    # (algebraically), regardless of function. This checks invertqtt on non-trivial TTs.
    R = 7
    N = 8
    a, b = (-1.0, -1.0), (1.0, 1.0)
    f(x, y) = cos(π * x) * sin(π * y)
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = 1.0e-10)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    K_out = R - 1
    S = result[K_out]
    @test size(S, 1) == 4^K_out
    @test size(S, 2) == (N + 1)^2

    cells_per_dim = 2^K_out
    grid = QG.DiscretizedGrid{2}(R, a, b; unfoldingscheme = :fused)
    max_err = 0.0
    for ix in 0:min(4, cells_per_dim - 1), iy in 0:min(4, cells_per_dim - 1)
        cell_flat = _morton_cell_flat((ix, iy), K_out)
        for (σx, σy) in ((0, 0), (1, 0), (0, 1), (1, 1))
            qinds = QG.grididx_to_quantics(grid, (2 * ix + σx + 1, 2 * iy + σy + 1))
            tt_val = tt(qinds)
            tx, ty = σx / 2.0, σy / 2.0
            interp_val = sum(
                P(βx, tx) * P(βy, ty) * S[cell_flat, βx + 1 + (N + 1) * βy]
                    for βx in 0:N, βy in 0:N
            )
            max_err = max(max_err, abs(interp_val - tt_val))
        end
    end
    @test max_err < 1.0e-8
end

@testset "invertqtt 2D round trip" begin
    # For q=1 in 2D, within each sub-cell at level K_out the inversion uses
    # linear Lagrange interpolation over the 4 fine-level (K=K_out+1) dyadic points.
    # At those 4 points (σx/2, σy/2) the Chebyshev interpolation should reproduce
    # the QTT values up to the stage-1 approximation error O(2^{-2K}).
    R = 8
    N = 10
    a, b = (0.0, 0.0), (1.0, 1.0)
    f(x, y) = exp(-x^2 - y^2)
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = 0.0)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    K_out = R - 1
    S = result[K_out]
    cells_per_dim = 2^K_out
    h = 1.0 / cells_per_dim

    grid = QG.DiscretizedGrid{2}(R, a, b; unfoldingscheme = :fused)
    max_err = 0.0
    for ix in 0:min(3, cells_per_dim - 1), iy in 0:min(3, cells_per_dim - 1)
        cell_flat = _morton_cell_flat((ix, iy), K_out)
        for (σx, σy) in ((0, 0), (1, 0), (0, 1), (1, 1))
            # QTT value at the fine-grid dyadic point (2*ix+σx, 2*iy+σy)
            qinds = QG.grididx_to_quantics(grid, (2 * ix + σx + 1, 2 * iy + σy + 1))
            tt_val = tt(qinds)
            # Chebyshev interpolation at relative position (σx/2, σy/2)
            tx, ty = σx / 2.0, σy / 2.0
            interp_val = sum(
                P(βx, tx) * P(βy, ty) * S[cell_flat, βx + 1 + (N + 1) * βy]
                    for βx in 0:N, βy in 0:N
            )
            max_err = max(max_err, abs(interp_val - tt_val))
        end
    end
    @test max_err < 1.0e-8
end

@testset "invertqtt 2D Stage 2 coarse levels" begin
    # Use f(x,y) = x*y (bilinear) so that both Stage 1 and Stage 2 restriction
    # are algebraically exact. This verifies the restriction operator at all levels
    # including multiple cells per level (not just cell 1).
    R = 6
    N = 6
    a, b = (0.0, 0.0), (1.0, 1.0)
    f(x, y) = x * y
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = 0.0)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    K_out = R - 1
    for (k, S) in enumerate(result)
        cells_per_dim = 2^k
        h = 1.0 / cells_per_dim
        max_err = 0.0
        for ix in 0:min(3, cells_per_dim - 1), iy in 0:min(3, cells_per_dim - 1)
            cell_flat = _morton_cell_flat((ix, iy), k)
            for βx in 0:N, βy in 0:N
                β_flat = βx + 1 + (N + 1) * βy
                x_ref = a[1] + (ix + P.grid[βx + 1]) * h
                y_ref = a[2] + (iy + P.grid[βy + 1]) * h
                max_err = max(max_err, abs(S[cell_flat, β_flat] - f(x_ref, y_ref)))
            end
        end
        @test max_err < 1.0e-8
    end
end

@testset "invertqtt 3D (uncompressed, Stage 1 values)" begin
    # Use f(x,y,z) = x*y*z (trilinear) so that the trilinear Lagrange Stage 1 is
    # algebraically exact — validates Morton ordering and L_mv Kronecker structure in 3D.
    R = 5
    N = 5
    a, b = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    f(x, y, z) = x * y * z
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = 0.0)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    K_out = R - 1
    S = result[K_out]
    @test size(S, 1) == 8^K_out        # (2^3)^{K_out}
    @test size(S, 2) == (N + 1)^3

    cells_per_dim = 2^K_out
    h = 1.0 / cells_per_dim
    max_err = 0.0
    for ix in 0:min(2, cells_per_dim - 1), iy in 0:min(2, cells_per_dim - 1), iz in 0:min(2, cells_per_dim - 1)
        cell_flat = _morton_cell_flat((ix, iy, iz), K_out)
        for βx in 0:N, βy in 0:N, βz in 0:N
            β_flat = βx + 1 + (N + 1) * βy + (N + 1)^2 * βz
            x_ref = a[1] + (ix + P.grid[βx + 1]) * h
            y_ref = a[2] + (iy + P.grid[βy + 1]) * h
            z_ref = a[3] + (iz + P.grid[βz + 1]) * h
            max_err = max(max_err, abs(S[cell_flat, β_flat] - f(x_ref, y_ref, z_ref)))
        end
    end
    @test max_err < 1.0e-10
end

@testset "invertqtt 2D output dimensions" begin
    R = 5
    N = 4
    a, b = (0.0, 0.0), (1.0, 1.0)
    f(x, y) = x * y
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = 0.0)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    @test length(result) == R - 1
    for k in 1:(R - 1)
        @test size(result[k], 1) == 4^k
        @test size(result[k], 2) == (N + 1)^2
    end
end

@testset "invertqtt (Stage 2 coarse levels)" begin
    R = 10
    N = 8
    a, b = -1.0, 1.0
    f(x) = cos(π * x)
    P = InterpolativeQTT.getChebyshevGrid(N)

    tt = InterpolativeQTT.interpolatesinglescale(f, a, b, R, N; tolerance = 0.0)
    result = InterpolativeQTT.invertqtt(tt, P; q = 1)

    for (k, S) in enumerate(result)
        max_err = 0.0
        for i in axes(S, 1)
            for β in 0:N
                x_ref = a + (b - a) * (i - 1 + P.grid[β + 1]) / 2^k
                max_err = max(max_err, abs(S[i, β + 1] - f(x_ref)))
            end
        end
        @test max_err < 1.0e-4
    end
end
