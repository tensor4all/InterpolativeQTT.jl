using PolynomialQTT
using CairoMakie
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI

α   = 0.1
f(x) = α / sqrt(α^2 + (x - 0.5)^2)
a, b = 0.0, 1.0
R    = 10       
N_dense  = 100     
N_sparse = 200

grid        = QG.DiscretizedGrid{1}(R, a, b)
quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:(2^R))
plotx       = QG.grididx_to_origcoord.(Ref(grid), 1:(2^R))
origdata    = f.(plotx)

tt_dense = PolynomialQTT.interpolatesinglescale(f, a, b, R, N_dense)
err_dense = maximum(abs.(tt_dense.(quanticsinds) .- origdata))

M_values = [5, 10, 15, 20, 30]
tt_sparse = [PolynomialQTT.interpolatesinglescale_sparse(f, a, b, R, N_sparse, M) for M in M_values]
errs_sparse = [maximum(abs.(tt.(quanticsinds) .- origdata)) for tt in tt_sparse]

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = L"x", ylabel = L"f(x)",
        title = L"Lorentzian peak: $f(x) = \alpha / \sqrt{\alpha^2 + (x-\frac{1}{2})^2}$, $\alpha = %$α$")
    lines!(ax, plotx, origdata, linewidth = 2)
    fig
end

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = L"M \text{ (bandwidth)}", ylabel = "Max absolute error",
        title = "Sparse construction error vs bandwidth  (N=$N_sparse)",
        yscale = log10)
    scatterlines!(ax, M_values, errs_sparse, label = "sparse  N=$N_sparse", linewidth = 2)
    hlines!(ax, [err_dense], color = :black, linestyle = :dash,
        label = "dense  N=$N_dense")
    axislegend(ax, pos = :topright)
    fig
end

M_best = 10
tt_best = tt_sparse[findfirst(==(M_best), M_values)]

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = L"\ell", ylabel = L"\chi_\ell",
        title = "Bond dimensions  (α=$α)")
    scatterlines!(ax, 1:(R - 1), TCI.linkdims(tt_dense),
        label = "dense  N=$N_dense", linewidth = 2, marker = :circle)
    scatterlines!(ax, 1:(R - 1), TCI.linkdims(tt_best),
        label = "sparse  N=$N_sparse, M=$M_best", linewidth = 2, marker = :diamond)
    axislegend(ax, pos = :topright)
    fig
end

println("Dense  N=$N_dense:  max err = $err_dense  rank = $(TCI.rank(tt_dense))")
for (M, tt, err) in zip(M_values, tt_sparse, errs_sparse)
    println("Sparse N=$N_sparse M=$M:  max err = $err  rank = $(TCI.rank(tt))")
end
