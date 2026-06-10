# invertqtt for interleaved and serial QTT orderings

**Date:** 2026-06-10
**Status:** Approved by Martin (Approach 1: normalize-then-share)

## Goal

Extend `invertqtt` (Lindsey arXiv:2311.12554 §5) to multivariate QTTs in
**interleaved** and **serial** orderings, in addition to the existing **fused**
ordering. No forward constructions are added in this work — interleaved QTTs
come from TCI/QuanticsTCI in practice; serial test inputs are built in the test
file.

## API

```julia
invertqtt(tt, P; q = 1, unfoldingscheme = :fused, ndims = nothing)
```

- `unfoldingscheme ∈ (:fused, :interleaved, :serial)`.
- `:fused` — `ndims` auto-detected from the physical dimension of the first
  core (`log2(phys_dim)`), preserving current behavior. If `ndims` is supplied
  it must agree with the detected value.
- `:interleaved`, `:serial` — every core has physical dimension 2, so `ndims`
  is **required** (`ArgumentError` otherwise). Bits per dimension
  `K = length(tt) ÷ ndims`; require `length(tt) % ndims == 0`.
- `q` — levels collapsed per dimension by Stage 1, `1 ≤ q < K`. Same semantics
  as the existing fused/1D paths.

### Validation (ArgumentError)

- `:interleaved`/`:serial` without `ndims`.
- `length(tt)` not divisible by `ndims`.
- Core physical dimension ≠ 2 for `:interleaved`/`:serial`; not a power of 2
  for `:fused`.
- `q` out of range relative to per-dimension `K`.

## Output convention (breaking change)

All schemes return `[S₁, …, S_{K−q}]` where `Sₖ` has shape
`((2^k)^ndims, (N+1)^ndims)` with **uniform column-major indexing**:

- Row: linear cell index `i = 1 + i₁ + 2^k·i₂ + 4^k·i₃ + …` (0-indexed cell
  coordinates, dimension 1 fastest).
- Column: `β_flat = 1 + β₁ + (N+1)·β₂ + …` (β₁ fastest, unchanged).

This **replaces the Morton (Z-order) row convention** that the just-added
fused multivariate path uses. The change lands before any release ships the
Morton convention. The 1D path is untouched (all conventions coincide in 1D).

## Architecture (Approach 1: normalize-then-share)

```
:fused        ──────────────► fused Stage 1 ──► Morton rows ──┐
:interleaved ──_fuse_cores──► fused Stage 1 ──► Morton rows ──┤── permute to
:serial      ──► _invert_stage1_serial ──► dim-major rows  ───┘   column-major
                                                                     │
                                              shared column-major Stage 2
                                              (levels K_out−1 … 1)
```

### Components

1. **`_fuse_cores(tt, ndims)`** — contracts each group of `ndims` adjacent
   cores of an interleaved TT into one core of physical dimension `2^ndims`.
   Within a group the dimension-1 bit is the fastest index of the fused
   physical index (`σ_flat = σ₁ + 2σ₂ + …`), matching the QuanticsGrids
   interleaved↔fused correspondence. Exact contraction — no compression, bond
   dimensions between groups unchanged.

2. **`_invert_stage1_serial(tt, P, q, ndims)`** — single left-to-right sweep
   maintaining `current` of shape `(rows, cols, bond)`. For each dimension
   block `n = 1…ndims`: the first `K−q` cores expand the row index (cell bits,
   coarse-to-fine); the last `q` cores are contracted with the 1D Lagrange
   matrix `L`, appending `βₙ` to the column index as the *slower* index so the
   final columns come out `β₁`-fastest with no permutation. Resulting rows are
   dimension-major (dimension 1 most significant).

3. **Row permutations** — `_morton_perm(ndims, k)` and `_serial_perm(ndims, k)`
   return permutation vectors mapping the native Stage-1 row order to
   column-major; applied once after Stage 1 (`S = S_native[p, :]`). Cost is
   O(size of S), negligible versus the contraction.

4. **`_apply_stage2_mv` (rewritten)** — single column-major multivariate
   restriction shared by all schemes. For each coarse cell `i` (column-major
   coords `i₁,…,i_N`) and each `σ ∈ {0,1}^N`: fine row index
   `j = Σₙ (2iₙ + σₙ)·2^{(n−1)(k+1)}`, restriction matrix
   `R_σ = foldl(kron, reverse([σₙ == 0 ? R_left : R_right for n]))`, accumulate
   `S_coarse[i, :] += R_σ * S_fine[j, :]`. Replaces the current Morton-based
   implementation.

5. **`invertqtt` dispatch** — normalizes per scheme, runs Stage 1, permutes to
   column-major, then runs the shared Stage 2 loop.

## Testing

- **Update existing fused multivariate tests** from Morton to linear cell
  indexing; delete the `_morton_cell_flat` test helper.
- **Interleaved (≥3 tests):**
  - Exact-values: TCI-built interleaved sample QTT of bilinear `f(x,y) = x·y`
    (linear Lagrange Stage 1 is algebraically exact for bilinear samples);
    check S against f at Chebyshev nodes with column-major indexing, tol 1e-10.
  - Round trip: Chebyshev interpolation at fine dyadic points reproduces TT
    values for a smooth compressed function, tol 1e-8.
  - Cross-scheme consistency: TCI sample QTTs of the same f on fused and
    interleaved grids → S matrices agree to TCI tolerance.
- **Serial (≥2 tests):** serial QTTs built in the test file via
  `TCI.crossinterpolate2` over a hand-decoded serial quantics function
  (first K bits = x₁ binary fraction, next K = x₂, …), small R, tol 1e-12.
  - Exact-values with bilinear f, column-major indexing.
  - Stage 2 coarse levels with bilinear f, all levels.
- **Validation tests:** missing `ndims`, non-divisible core count, q range.

## Documentation

Update the `invertqtt` docstring: new keywords, ordering semantics, the
column-major output convention, and the serial/interleaved input requirements.

## Out of scope

- Interleaved/serial forward constructions (`interpolatesinglescale`); §7.2
  serial construction remains a future feature.
- `q > 1` behavior beyond mirroring the existing fused/1D semantics.
