"""
For sparse interpolation
"""
function angular_local_lagrange(P::LagrangePolynomials{Float64}, M::Int)
    N = length(P.grid) - 1
    @assert N >= 2M "Need N ≥ 2M (got N=$N, M=$M)"

    Ā = zeros(N + 1, 2, N + 1)   # indices: [α+1, σ+1, β+1]

    for σ in 0:1, β in 0:N
        x = (Float64(σ) + P.grid[β + 1]) / 2
        θ_x = acos(clamp(1 - 2x, -1.0, 1.0))

        # Nearest angular grid index (grid is uniform: θ^γ = πγ/N)
        ι = clamp(round(Int, θ_x * N / π), 0, N)

        # Shift window to stay within [0, N] while keeping size 2M+1
        ι_lo = clamp(ι - M, 0, N - 2M)
        window = ι_lo:(ι_lo + 2M)          # exactly 2M+1 indices

        # Angular positions of the window nodes
        θ_win = [π * γ / N for γ in window]

        # Evaluate the local Lagrange basis L^α(θ_x) for each α in the window
        for (k, α) in enumerate(window)
            L = one(Float64)
            for j in eachindex(θ_win)
                j == k && continue
                L *= (θ_x - θ_win[j]) / (θ_win[k] - θ_win[j])
            end
            Ā[α + 1, σ + 1, β + 1] = L
        end
    end

    return Ā
end

"""
    interpolatesinglescale_sparse(f, a, b, numbits, polynomialdegree, M; ...)

Sparse variant of `interpolatesinglescale` (Section 4.3).
"""
function interpolatesinglescale_sparse(
        f,
        a::Float64, b::Float64,
        numbits::Int,
        polynomialdegree::Int,
        M::Int;
        tolerance = 1.0e-12,
        maxbonddim = typemax(Int)
    )
    _scale(x) = a + (b - a) * x
    P = getChebyshevGrid(polynomialdegree)

    Aleft = [
        f(_scale((sigma + P.grid[beta + 1]) / 2))
            for sigma in [0, 1], beta in 0:polynomialdegree
    ]
    Acenter = angular_local_lagrange(P, M)
    Aright = [
        P(alpha, sigma / 2)
            for alpha in 0:polynomialdegree, sigma in [0, 1]
    ]

    train_ = [
        reshape(Aleft, 1, 2, size(Aleft, 2)),
        fill(Acenter, numbits - 2)...,
        reshape(Aright, size(Aright, 1), 2, 1),
    ]

    return _compress_train(train_, tolerance, maxbonddim)
end


"""
    interpolatesinglescale_sparse(f, a, b, numbits, polynomialdegree, M; ...)

N-dimensional version of the sparse single-scale construction.  The sparse
1-D core replaces `interpolationtensor(P)` inside the Kronecker product built
by `_direct_product_coretensors`.
"""
function interpolatesinglescale_sparse(
        f,
        a::NTuple{D, Float64}, b::NTuple{D, Float64},
        numbits::Int,
        polynomialdegree::Int,
        M::Int;
        tolerance = 1.0e-12,
        maxbonddim = typemax(Int)
    ) where {D}
    _scale(x::NTuple{D, Float64})::NTuple{D, Float64} = tuple((a .+ (b .- a) .* x)...)

    P = getChebyshevGrid(polynomialdegree)

    Aleft = Array{Float64, 2D}(
        undef, ntuple(_ -> 2, D)..., ntuple(_ -> polynomialdegree + 1, D)...
    )
    for sigmas in Iterators.product(ntuple(_ -> 0:1, D)...)
        for betas in Iterators.product(ntuple(_ -> 0:polynomialdegree, D)...)
            point = tuple(((sigmas[n] + P.grid[betas[n] + 1]) / 2 for n in 1:D)...)
            Aleft[(sigmas .+ 1)..., (betas .+ 1)...] = f(_scale(point)...)
        end
    end

    # Only this line differs from interpolatesinglescale: sparse core instead of dense
    Acenter_ = angular_local_lagrange(P, M)
    Acenter = _direct_product_coretensors(fill(Acenter_, D))

    Aright_ = [
        P(alpha, sigma / 2)
            for alpha in 0:polynomialdegree, sigma in [0, 1]
    ]
    Aright = _direct_product_coretensors(fill(reshape(Aright_, size(Aright_)..., 1), D))

    train_ = [
        reshape(Aleft, 1, 2^D, :),
        fill(Acenter, numbits - 2)...,
        Aright,
    ]

    return _compress_train(train_, tolerance, maxbonddim)
end
