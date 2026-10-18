# Finite-cell-method (FCM) machinery: the quadrature of cells crossed by ∂Ω.
#
# Algorithm
#
#   A cut cell's rule is built in two stages, each a field of the `PhysicalDomain`:
#
#   1. The cut rule integrates Ω ∩ R accurately and with however many points that takes. The package's own,
#      `implicit_rule`, is the Saye dimension-reduction implicit quadrature of `src/implicit.jl` on the CSG level set of
#      `src/physical.jl` — exact for polynomials where the leaves are linear, polytope corners included, and spectrally
#      accurate on smooth curved boundaries — at a Gauss count that integrates the moments below exactly where it can.
#   2. The compression replaces that rule by one with O(nbasis) points that reproduces its moments over a tensor
#      Legendre basis on R,
#
#        mₐ = Σ_q w_q ψₐ(ξ(x_q)) = ∫_{Ω ∩ R} ψₐ(ξ(x)) dx,
#
#      so it integrates every polynomial of the moment order as accurately as the cut rule did — the moment-fitting idea
#      of
#
#        B. Müller, F. Kummer, M. Oberlack, "Highly accurate surface and volume
#        integration on implicit domains by means of moment-fitting", Int. J.
#        Numer. Methods Engng. 96 (2013) 512–528, doi:10.1002/nme.4569,
#
#      with the cut rule's points as the candidates and its moments as the target. The package carries two compressions,
#      both returning non-negative weights at a subset of the cut rule's points:
#
#        - `Caratheodory` (default) prunes the cut rule to at most rank-many points by Carathéodory's theorem, made
#          constructive by null-space elimination and made affordable by the recombination of Litterer & Lyons (both
#          cited at `Caratheodory`). It reproduces the moments by construction.
#        - `NonnegativeFit` fits non-negative weights at candidates drawn from the cut rule, with the structure of
#          QuESo's `QuadratureTrimmedElement` (M. Meßmer et al., Comput. Methods Appl. Mech. Engrg. 400 (2022) 115584,
#          doi:10.1016/j.cma.2022.115584; https://github.com/manuelmessmer/QuESo, BSD-4-Clause), solved by the
#          classical Lawson–Hanson algorithm (C. L. Lawson, R. J. Hanson, "Solving Least Squares Problems",
#          Prentice-Hall (1974), Ch. 23; SIAM reprint doi:10.1137/1.9781611971217).
#
#   `moment_fit_rule` runs the two and owns everything they share: the moments, the truncation of round-off weights,
#   the residual of the rule handed back, and the fallback to the cut rule itself when a compression misses.
#
# Deviations from QuESo
#
#   QuESo classifies axis-aligned boxes against a triangulated B-rep and computes moments from a divergence-theorem
#   surface integral. This package classifies against a CSG level set (`PhysicalDomain`) and takes the moments from the
#   implicit-quadrature kernel, which is accurate on cut cells without an octree, so QuESo's stair-step moment
#   integrator, its `PointElimination` inner loop and its accuracy-driven `AssembleIPs` retry are all absent: they
#   worked around an accuracy floor the kernel does not have. Non-graph-like cells are handled by the kernel's own
#   bounded subdivision.
#
#   `NonnegativeFit` does carry a retry of its own, and it is a different mechanism from QuESo's: up to three solves on
#   ever denser candidate clouds drawn from the same cut rule, there because a subset of the rule need not admit an
#   exact non-negative fit (`_candidate_budget`). It never eliminates points from a fitted rule, and never changes the
#   rule or its moments.
#
#   Public entry points: `moment_fit_rule(physical, region_box, moment_order)`, `implicit_rule`, `Caratheodory`,
#   `NonnegativeFit`.

# ── 1. Tensor Legendre moments on a region box ────────────────────────────────

# Flat lexicographic ordering of tensor-product Legendre indices up to `moment_order` per axis:
# `vec(CartesianIndices(0:p₁, …, 0:p_D))`. The resulting `Vector{CartesianIndex{D}}` is the moment-basis index list
# `α = (α₁, …, α_D)` used by the moment vector and the moment-fit matrix. Its first entry is α = 0, whose basis function
# is the constant 1, so the first moment of a rule is the measure it integrates.
function _moment_basis_indices(moment_order::NTuple{D,Int}) where {D}
    return vec(collect(CartesianIndices(ntuple(d -> 0:moment_order[d], D))))
end

# Cardinality of the tensor-product moment basis, `prod(moment_order .+ 1)`.
_moment_basis_count(moment_order::NTuple{D,Int}) where {D} = prod(moment_order .+ 1)

# Fill caller-owned per-axis 1D Legendre value tables for the point `x` mapped into the region's reference frame
# `[−1, 1]ᴰ`:
#
#     factors[d][p + 1] = L_p(ξ_d),    ξ_d = (2 x[d] − lower[d] − upper[d]) / (upper[d] − lower[d]),
#
# via one Legendre three-term recurrence per axis, so the per-point cost is `D · (order + 1)` instead of
# `D · (order + 1)^D`.
function _fill_legendre_factors!(factors, x::SVector{D,T}, lower::SVector{D,T}, upper::SVector{D,T},
                                 moment_order::NTuple{D,Int}) where {D,T}
    @inbounds for d in 1:D
        xi = (2 * x[d] - lower[d] - upper[d]) / (upper[d] - lower[d])
        p = moment_order[d]
        fd = factors[d]
        fd[1] = one(T)
        p >= 1 && (fd[2] = xi)
        if p >= 2
            pm2 = one(T)
            pm1 = xi
            for m in 2:p
                pm = ((2m - 1) * xi * pm1 - (m - 1) * pm2) / m
                fd[m + 1] = pm
                pm2, pm1 = pm1, pm
            end
        end
    end
    return nothing
end

# Caller-owned per-axis 1D scratch buffers used by `_fill_legendre_factors!` and the tensor-product evaluation: one
# `Vector{T}` per axis, length `moment_order[d] + 1`. Allocated once per call, then reused across points.
function _legendre_factor_buffers(moment_order::NTuple{D,Int}, ::Type{T}) where {D,T}
    return ntuple(d -> Vector{T}(undef, moment_order[d] + 1), D)
end

# Tensor-product basis value at the point whose per-axis Legendre factors are loaded into `factors`:
# `ψ_α(ξ) = ∏_d L_{α_d}(ξ_d) = ∏_d factors[d][α_d + 1]`.
function _tensor_legendre_value(factors, idx::CartesianIndex{D}) where {D}
    v = factors[1][idx.I[1] + 1]
    @inbounds for d in 2:D
        v *= factors[d][idx.I[d] + 1]
    end
    return v
end

# The moment-fit design matrix
#
#     A[a, j] = ψ_α_a(ξ(x_j)),
#
# columns indexed by points `x_j`, rows by moment-basis multi-indices `α_a`. One Legendre table fill per point, then a
# tensor-product evaluation per (point, basis-index) pair. The matrix is `nbasis × npoints`.
function _build_moment_matrix(points::AbstractVector{SVector{D,T}}, region_box::AxisBox{D,T},
                              moment_order::NTuple{D,Int}) where {D,T}
    indices = _moment_basis_indices(moment_order)
    A = Matrix{T}(undef, length(indices), length(points))
    factors = _legendre_factor_buffers(moment_order, T)
    @inbounds for (j, x) in pairs(points)
        _fill_legendre_factors!(factors, x, region_box.lower, region_box.upper, moment_order)
        for (a, idx) in pairs(indices)
            A[a, j] = _tensor_legendre_value(factors, idx)
        end
    end
    return A
end

# The tensor Legendre moments `mₐ = Σ_q w_q ψₐ(x_q)` of a rule, `A w` without forming `A`: one Legendre table fill per
# point, so a rule of any size costs `nbasis` storage.
function _moments_from_rule(points::AbstractVector{SVector{D,T}}, weights::AbstractVector{T}, region_box::AxisBox{D,T},
                            moment_order::NTuple{D,Int}) where {D,T}
    indices = _moment_basis_indices(moment_order)
    moments = zeros(T, length(indices))
    factors = _legendre_factor_buffers(moment_order, T)
    @inbounds for q in eachindex(points, weights)
        _fill_legendre_factors!(factors, points[q], region_box.lower, region_box.upper, moment_order)
        wq = weights[q]
        for (a, idx) in pairs(indices)
            moments[a] += wq * _tensor_legendre_value(factors, idx)
        end
    end
    return moments
end

# ── 2. The cut rule ───────────────────────────────────────────────────────────

"""
    implicit_rule(physical, region_box, moment_order) -> (points, weights)

The package's cut rule, and the default `cut_rule` of [`physical_domain`](@ref): Saye's implicit volume rule for
`region_box ∩ Ω` ([`implicit_volume_quadrature`](@ref)) on the CSG leaves of `physical`, in physical coordinates with
non-negative weights.

Its accuracy is set by `moment_order`, the per-axis order of the tensor Legendre moments the rule is compressed against:
the fiber Gauss count (`_implicit_gauss_points`) makes the rule exact for those moments wherever the leaves are linear,
polytope corners included, and spectrally accurate on a smooth curved boundary. The subdivision budget for a cell that
is not graph-like is the domain's (`subcell_length_scale`, `max_depth`). Each leaf's Lipschitz constant is forwarded to
the kernel, so a geometric feature smaller than `region_box` is certified rather than sampled for — leaves left at the
[`leaf`](@ref) default `Inf` keep the kernel's point-sampling heuristic and its sub-cell blind spot.

The rule is accurate, not small: a few hundred to a few hundred thousand points per cell, growing with the moment
order and with the geometric features the cell holds. Making it small is the compression's job (see
[`moment_fit_rule`](@ref)).
"""
function implicit_rule(physical::PhysicalDomain, region_box::AxisBox{D,T}, moment_order::NTuple{D,Int}) where {D,T}
    leaves = _leaves(physical.geometry)
    return implicit_volume_quadrature(Any[l.f for l in leaves], x -> _inside(physical.geometry, x), region_box;
                                      gauss_points=_implicit_gauss_points(moment_order),
                                      max_subdiv=_effective_subcell_depth(physical, region_box),
                                      lipschitz=Float64[l.lipschitz for l in leaves])
end

# Per-fiber Gauss count for the implicit moment rule, the maximum of two needs:
#
#   1. Exactness on a *tilted* linear cut. The inside-interval endpoints are then linear in the base coordinates, so
#      each of the D fiber integrations raises the accumulated polynomial degree by one; the worst case (a corner
#      beveled by a linear leaf, e.g. the simplex Σx ≤ 1) reaches total degree `sum(moment_order) + D` at the innermost
#      fiber, needing `q ≥ (sum(moment_order) + D) / 2`. An axis-aligned cut needs only `⌈(maxorder+1)/2⌉`, but the
#      default must serve the worst case — a constant offset silently undershoots tilted high-order 3D cuts by ~1e-5.
#      (Verified against tilted simplices in 2D–4D as the smallest exact q.)
#   2. Accuracy on a *curved* cut. A curved boundary is never polynomial, so the moments are approximated, not exact; a
#      generous floor of `maxorder + 2` keeps that high-order approximation sharp (it also dominates (1) in 2D).
function _implicit_gauss_points(moment_order::NTuple{D,Int}) where {D}
    max(cld(sum(moment_order) + D, 2), maximum(moment_order) + 2)
end

# ── 3. Non-negative least-squares compression ─────────────────────────────────

"""
    nnls(A, b) -> (x, residual_norm)

Solve the non-negativity-constrained linear least-squares problem

    minimise  ‖A x − b‖₂  subject to  x ≥ 0,

returning the solution vector and the L² residual `‖A x − b‖₂`.

Wraps `NonNegLeastSquares.nonneg_lsq` with `alg = :nnls`, the classical Lawson–Hanson (1974) active-set algorithm. It
operates directly on the `nbasis × npoints` design matrix and converges in roughly `nbasis` outer iterations, each
solving a small `|P| × |P|` linear system with `|P| ≤ nbasis`. The moment-fit matrix is short and wide
(`nbasis ≪ npoints`), so the matrix the algorithm walks IS the small one.

The Bro & de Jong (1997) Fast NNLS variant (`alg = :fnnls`) works on the `npoints × npoints` Gram matrix instead, which
is the wrong shape here: our candidate cloud is `6 · nbasis` wide, so the Gram matrix is the large one. Measured on one
cut cell of the space-time cavity geometry, `:fnnls` against `:nnls` at a candidate count of 162 / 750 / 2058: 0.000 s
vs 0.004 s, 0.014 s vs 0.005 s, and 29.7 s vs 0.124 s — it wins only on the smallest cloud and is 240× slower by
`nbasis = 343`. It is also markedly less accurate, the relative moment residual running 2.7e-15 / 6.2e-05 / 1.4e-03
against `:nnls`'s uniform ~1e-15, so the Gram formulation loses the conditioning the Householder walk keeps. `:nnls` is
the right choice on both counts.

`max_iter` is raised to `10 · npoints` from the library default of `3 · npoints`. It counts the algorithm's *inner*
iterations, not its outer ones (`NonNegLeastSquares/src/nnls.jl`), so it is not a bound on the `nbasis` columns the
active set accepts. Lawson–Hanson stays feasible at every step, so reaching the cap is not a wrong answer — but the
library announces it by printing `NNLS quitting on iteration count` to stdout, and a library call has no business
writing there. Raising a cap cannot change a solve that converged under the old one: the cap only ever truncates.

How far cells overrun is not a few percent. At moment order `(8, 8, 8)` on the space-time cavity geometry the cap is
still reached, at ten times the library default, on cells whose candidate cloud is degenerate enough to make the walk
cycle. Those are the cells `_cap_candidates` used to manufacture with an integer stride; the fractional spacing it now
uses removes most of them, and the ones that remain are a signal that the cell is under-resolved for its geometric
complexity rather than evidence the cap is too small.

Allocations are owned by `NonNegLeastSquares.jl`; `A` and `b` are not mutated, so the name carries no `!`.
"""
function nnls(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:Real}
    x = vec(nonneg_lsq(A, b; alg=:nnls, max_iter=10 * size(A, 2)))
    residual = norm(A * x - b)
    return x, residual
end

"""
    NonnegativeFit(; target_residual = 1e-6)

A compression for [`physical_domain`](@ref), the alternative to [`Caratheodory`](@ref): non-negative least squares on
the moments of the cut rule. One Lawson–Hanson solve ([`nnls`](@ref)) picks non-negative weights at candidate points
drawn from the cut rule so that they reproduce its tensor Legendre moments; its active set, and so the rule it returns,
holds at most `nbasis` points.

The candidates are `6 · nbasis` points of the cut rule drawn by even fractional spacing (`_cap_candidates`). The cut
rule itself is an exact non-negative solution over all of its points, but a subset of them admits one only if the
moments lie in the cone the subset spans (`_candidate_budget`), so a fit whose residual exceeds `target_residual · ‖m‖`
is retried on a cloud eight times denser, at most twice. The best of the attempts is returned; the cut rule and its
moments are the same for all of them, and a cloud that already holds the whole rule is not drawn again.

`target_residual` is relative to the norm of the moment vector and so independent of the cell's size; it decides only
whether to retry. Whether the returned rule is used is decided by [`moment_fit_rule`](@ref), against its own failure
threshold.
"""
struct NonnegativeFit{T<:Real}
    target_residual::T
end

function NonnegativeFit(; target_residual::Real=1.0e-6)
    target_residual > 0 || throw(ArgumentError("target_residual must be positive; got $target_residual"))
    return NonnegativeFit(float(target_residual))
end

function (fit::NonnegativeFit)(points::Vector{SVector{D,T}}, weights::Vector{T}, moments::Vector{T},
                               region_box::AxisBox{D,T}, moment_order::NTuple{D,Int}) where {D,T}
    target = fit.target_residual * norm(moments)
    best = (points, weights, T(Inf))
    for attempt in 1:_MAX_FIT_ATTEMPTS
        candidates = _cap_candidates(points, moment_order, _candidate_budget(attempt))
        x, residual = nnls(_build_moment_matrix(candidates, region_box, moment_order), moments)
        residual < best[3] && (best = (candidates, x, residual))
        (residual <= target || length(candidates) == length(points)) && break
    end
    candidates, x, _ = best
    kept = findall(>(0), x)
    return candidates[kept], x[kept]
end

# Number of NNLS solves `NonnegativeFit` may spend on one cell.
const _MAX_FIT_ATTEMPTS = 3

# Ceiling on the NNLS design matrix, in entries (`nbasis × ncandidates`). Lawson–Hanson sweeps the whole matrix once per
# accepted column, so a solve's memory is proportional to this product and its time to the product times the number of
# accepted columns (≤ nbasis). 6·10⁶ entries is 48 MB — and nothing like a fraction of a second at the top of the range.
# Measured serially at `-O2` on the space-time cavity geometry, one cut cell's fit takes 0.2 ms at `nbasis = 27`, 7 ms
# at 125, 215 ms at 343 and ≈ 2.4 s (median; 17 s worst) at 729, where the matrix is 729 × 4374 = 3.2·10⁶ entries, 25
# MB. The growth is steeper than the `nbasis³` the flop count predicts because past ~10 MB the matrix no longer fits in
# per-core cache and the sweep becomes bandwidth-bound rather than compute-bound. Because the candidate budget below is
# a multiple of `nbasis`, the entry count grows as `budget · nbasis²` — quadratically in the basis size — so the retry
# needs an absolute ceiling and not just a relative one. It binds retries only (the `max` in `_cap_candidates` keeps the
# first attempt exempt), and there only once `nbasis` exceeds `√(_MAX_FIT_MATRIX_ENTRIES / budget)`: ≈ 354 on the second
# attempt, ≈ 126 on the third.
const _MAX_FIT_MATRIX_ENTRIES = 6_000_000

# Cap the candidate cloud so the NNLS design matrix stays bounded regardless of the cut rule's point count: at most
# `budget · nbasis` candidates, and never more than `_MAX_FIT_MATRIX_ENTRIES` matrix entries.
#
# The survivors are `cap` indices spread across the rule by FRACTIONAL spacing, `(n − 1) / (cap − 1)`, rather than by an
# integer stride. The integer stride this replaces carried two separate defects, both measured on the space-time cone
# fixture of `examples/applications/spacetime_cavity_2d` at cells (8, 8, 8):
#
#   1. It drew the wrong NUMBER of points. `cld(n, cap)` rounds the stride up, so the draw retained `cld(n, stride)`
#      candidates and not `cap` — 2744 of an allowed 4374 on a 5488-point rule, throwing away 37% of a budget the solve
#      was already sized for.
#   2. It aliased. The volume rule emits its points fiber by fiber in blocks of `gauss_points`, so an integer stride
#      sharing a factor with that block length reaches only `gauss_points / gcd` of the per-fiber node positions. At
#      moment order (8, 8, 8) the fiber count is 14, and the 5488- and 8232-point rules both take stride 2: 82 of 146
#      cut cells then fitted to a relative residual above 1e-12 and 57 of them above 1e-3. Drawn without the stride at
#      the *same* candidate count, those cells reach ~2e-15. Shifting the stride's starting index does not help, so what
#      matters is the periodicity and not its phase.
#
# Fractional spacing addresses both at once: it spends the whole budget, and its step is not an integer, so successive
# indices sweep the per-fiber positions instead of locking onto a fixed subset of them. It is deterministic, which
# everything in this path owes `test/characterize.jl`, and it is dimension-generic — the rule is a flat
# `Vector{SVector{D,T}}` and nothing here inspects `D`.
#
# `budget` comes from `_candidate_budget(attempt)` and grows with the retry, so each attempt draws a genuinely denser
# cloud. The `max` keeps the first attempt's `6 · nbasis` budget exempt from the ceiling, so the entry point of the
# pipeline behaves identically at every moment order.
function _cap_candidates(points::Vector{SVector{D,T}}, moment_order::NTuple{D,Int}, budget::Int) where {D,T}
    nbasis = _moment_basis_count(moment_order)
    cap = min(budget * nbasis, max(6 * nbasis, cld(_MAX_FIT_MATRIX_ENTRIES, nbasis)))
    n = length(points)
    n <= cap && return points
    # `n > cap` holds here, so the spacing exceeds 1 and the rounded indices are strictly increasing: the draw cannot
    # repeat a candidate.
    return points[round.(Int, range(1, n; length=cap))]
end

# Candidate-cloud budget of `attempt`, as a multiple of the moment-basis size `nbasis`: 6·nbasis on the first attempt,
# eight times that on each retry.
#
# The budget has to grow for the retry to mean anything, and what it buys is cone membership rather than conditioning.
# Lawson–Hanson is greedy and terminates as soon as its passive set holds `nbasis` columns (`nsetp ≥ m` in the classical
# algorithm), so the fit it returns is decided by which candidates the walk was offered — but what decides whether a
# zero-residual non-negative fit EXISTS over those candidates is narrower: the moments admit one only if `m` lies in the
# convex cone spanned by the selected columns' moment vectors. The full cut rule always satisfies that by construction,
# its own positive weights being such a solution. A subset need not, and a subset that does not cannot be rescued by any
# solver, however well conditioned.
#
# That distinction is measurable, and the measurement picks cone membership. At moment order (8,8,8) on the space-time
# cone geometry, with the draw held non-degenerate so that only the COUNT varies, the relative residual runs 1.9e-2 at
# 1.5·nbasis candidates, 9.2e-3 at 2·nbasis and 1.9e-3 at 3·nbasis, then falls to 3.1e-15 at the 6·nbasis the first
# attempt uses. A ×2 change in count moves the residual by orders of magnitude while leaving the conditioning of any one
# selection untouched — the signature of the cone widening, not of a triangular solve improving. The first attempt's
# 6·nbasis is therefore load-bearing and cannot be trimmed to buy speed.
#
# An earlier measurement of the same effect, on a cell carrying many distinct boundary pieces (nbasis = 125, 25216 rule
# points, the moments held fixed): 742 candidates → residual 4.7e-2, 1484 → 2.8e-2, 2802 → 3.0e-2, but 5044 → 3e-16 and
# 8406 → 6e-16. The growth factor is large on purpose: a ×8 step moves each retry to a plainly different scale, so a
# cell whose cone was too narrow gets a materially wider one instead of another draw of much the same extent.
#
# The retry used to raise the cut rule's fiber Gauss order as well, regenerating the rule and its moments on every
# attempt. That bought nothing the budget did not, and it was not free: on a cell holding eight spherical voids (moment
# order (4, 4, 4), 3.4·10⁵ rule points) the fit that regenerated took 0.93 s and the one that drew a denser cloud from
# the same rule 0.042 s, the latter reaching a relative residual of 9e-16. It also made the rule a cell received depend
# on how its first fit went.
_candidate_budget(attempt::Int) = 6 * 8^(attempt - 1)

# ── 4. Carathéodory compression ───────────────────────────────────────────────

"""
    Caratheodory()

A compression for [`physical_domain`](@ref), and its default: Carathéodory pruning of the cut rule. The cut rule is a
non-negative combination of point evaluations whose moments are `m = A w`, with `A` the `nbasis × npoints` matrix of
the tensor Legendre basis at its points. By Carathéodory's theorem a combination of more points than `rank A` can be
reduced to one of at most `rank A ≤ nbasis` of them with the same moments and non-negative weights, and the reduction is
constructive: any null vector of `A` on the current points moves the weights without moving the moments, so walking
along one until the first weight reaches zero removes a point. The rule returned is a subset of the cut rule's points
with non-negative weights, reproducing its moments to round-off — by construction, with no target, retry or fallback
to tune.

Pruning the points one null vector at a time costs a factorisation per point, so the cut rule is first reduced by
recombination (Litterer & Lyons): its points are grouped into `2 · nbasis` clusters, the clusters' moment vectors are
pruned instead — at most `nbasis` survive — and each surviving cluster's weights are rescaled by its coefficient,
which halves the points at the cost of one moment sum over them and one factorisation of a `nbasis × 2 nbasis`
matrix. Repeated until at most `2 · nbasis` points remain, which are then pruned directly, it costs
O(npoints · nbasis + nbasis³ log(npoints / nbasis)) and never forms a matrix larger than `nbasis × 2 nbasis`, however
many points the cut rule carries.

Measured against [`NonnegativeFit`](@ref) on the same cut rules — 34 cells in 2D and 3D at moment orders 2–12: disc,
sliver, polytope-corner and perforated cells — Carathéodory took 2.4 s altogether against 7.8 s, and 0.77 s against
5.3 s at moment order (8, 8, 8) where both cost most. It returned 5% fewer points, fewer wherever the cell's moment
matrix is numerically rank-deficient: 560 against 660 of the 729 a ball's cut cell admits. Both reproduce the moments
to round-off, the largest relative residual being 3e-13 against NNLS's 2e-15. NNLS, which solves on a capped candidate
cloud, is the faster on a cut rule of 10⁵ points at low moment order: 0.04 s against 0.11 s at moment order (4, 4, 4)
on 3.4·10⁵ points.

  - C. Carathéodory, "Über den Variabilitätsbereich der Fourier'schen Konstanten von positiven harmonischen
    Funktionen", Rend. Circ. Mat. Palermo 32 (1911) 193–217, doi:10.1007/BF03014795.
  - C. Litterer, T. Lyons, "High order recombination and an application to cubature on Wiener space", Ann. Appl.
    Probab. 22 (2012) 1301–1327, doi:10.1214/11-AAP786.
  - F. Bělík, J. Chan, A. Narayan, "Efficient and robust Carathéodory–Steinitz pruning of positive discrete measures",
    arXiv:2510.14916 (2025), for the same reduction applied to cut-cell quadrature.
"""
struct Caratheodory end

function (::Caratheodory)(points::Vector{SVector{D,T}}, weights::Vector{T}, moments::Vector{T},
                          region_box::AxisBox{D,T}, moment_order::NTuple{D,Int}) where {D,T}
    nbasis = length(moments)
    w = copy(weights)
    support = findall(>(0), w)
    # Recombination: prune the moment vectors of 2·nbasis clusters, and scale each cluster's weights by what survives.
    while length(support) > 2nbasis
        clusters = Iterators.partition(support, cld(length(support), 2nbasis))
        S = stack(_moments_from_rule(view(points, c), view(w, c), region_box, moment_order) for c in clusters)
        for (c, λ) in zip(clusters, _prune!(ones(T, size(S, 2)), S))
            w[c] .*= λ
        end
        filter!(i -> w[i] > 0, support)
    end
    # The at most 2·nbasis points left, pruned directly.
    ws = _prune!(w[support], _build_moment_matrix(view(points, support), region_box, moment_order))
    kept = findall(>(0), ws)
    return points[support[kept]], ws[kept]
end

# Carathéodory's theorem, constructively: reduce the non-negative coefficients `λ` of the combination `S λ` of the
# columns of `S`, in place, to at most `rank S` positive ones, leaving `S λ` unchanged to round-off.
#
# One pass factors `S` on the current support for an orthonormal basis of its null space. Each null vector `v` moves
# the coefficients without moving `S λ`; walking `λ − t v` from `t = 0` until the first coefficient reaches zero keeps
# every coefficient non-negative and removes one column, and eliminating that column's entry from the remaining null
# vectors keeps them null vectors of the reduced support, so one factorisation serves the whole pass. Those updates
# accumulate round-off, and a vector worn down to round-off — no entry above `size(V, 1) · eps` on the scale of the
# orthonormal basis it started in — is not walked, so a pass may end with columns left over; the next one factors
# afresh, and the pruning ends when the support has no null space. Each vector is walked in whichever of its two
# directions has the larger positive entry, so every pass removes at least one column: the first vector of a fresh
# basis is a unit vector. There is therefore at most one pass per column, and in practice there are two or three.
#
# A coefficient below `eps · max λ` is round-off left by a walk, and is dropped before the support is factored: kept, it
# would be one more column for the null space to find at that scale.
function _prune!(λ::AbstractVector{T}, S::AbstractMatrix{T}) where {T}
    while true
        λmax = maximum(λ)
        λ .= ifelse.(λ .> eps(T) * λmax, λ, zero(T))
        support = findall(>(0), λ)
        V = nullspace(S[:, support])
        size(V, 2) == 0 && return λ
        tol = size(V, 1) * eps(T)
        for j in axes(V, 2)
            v = view(V, :, j)
            maximum(v) < -minimum(v) && (v .= .-v)
            t, i = findmin(r -> v[r] > tol ? λ[support[r]] / v[r] : T(Inf), eachindex(v))
            isfinite(t) || continue
            for (r, s) in pairs(support)
                λ[s] = max(λ[s] - t * v[r], zero(T))
            end
            λ[support[i]] = zero(T)
            for l in (j+1):size(V, 2)
                @views V[:, l] .-= (V[i, l] / v[i]) .* v
            end
        end
    end
end

# ── 5. The cut-cell rule ──────────────────────────────────────────────────────

# Truncation cutoff for a compressed rule's weights, relative to the cut volume: a weight within it of zero is round-off
# around an exact zero of the compression and is dropped. The number is dimensionless because a quadrature weight
# carries the units of a volume, so the cutoff is taken *relative to the cut volume* `∫_{Ω∩R} 1 dx`, the first moment.
# Every weight scales as (length scale)ᴰ, so an absolute constant instead keeps every weight on a geometry of unit size
# and empties the rule on a small one: at a domain length scale of 1e-5 in 3D the whole cut volume is ~1e-15 — three
# orders of magnitude *under* a 1e-12 cutoff — so a perfectly good rule truncates to nothing. The breakpoint tolerance
# of `src/implicit.jl` and the mesh-SDF length tolerance of `ext/UnfittedMeshIOExt.jl` are scaled the same way, and for
# the same reason: the kernel has to behave identically under a rescaling of the geometry. It is the cut volume and not
# `volume(region_box)` because on a sliver cut the two differ by orders of magnitude, and only the former tracks the
# weights being tested.
const _WEIGHT_TOL = 1.0e-12

# Failure threshold for a compressed rule's moment residual, RELATIVE to the moment vector's norm `‖m‖`. A residual
# above `_FIT_FAILURE_RESIDUAL · ‖m‖` means the compression did not come anywhere near reproducing the moments;
# `moment_fit_rule` then returns the cut rule itself, and the integration-plan dispatcher tags such a region
# `:cut_fallback`.
#
# It is dimensionless for the reason `_WEIGHT_TOL` is: the residual carries the units of a moment, (length)ᴰ. An
# absolute threshold made the verdict a statement about the cell's size — the same four-holes Float32 cell, fitted to
# the same 1e-7 relative residual, came back `:fitted` at a cell extent of 1 and `:fallback` at 2000, where
# `1e-7 · ‖m‖ ≈ 0.35` exceeds 1e-2.
const _FIT_FAILURE_RESIDUAL = 1.0e-2

"""
    moment_fit_rule(physical, region_box, moment_order) -> (points, weights, residual, status)

The quadrature rule of `region_box ∩ Ω` that assembly uses on a cut region: the domain's `cut_rule` integrates the cut
volume, and its `compression` reduces that rule to one that reproduces its tensor Legendre moments

    m_α = Σ_q w_q ψ_α(ξ(x_q)) = ∫_{region_box ∩ Ω} ψ_α(ξ(x)) dx,    α ∈ {0,…,p₁} × … × {0,…,p_D},

with far fewer points (see [`physical_domain`](@ref) for both stages and their contracts). Points are in physical
coordinates and weights are physical, so `Σ w_q` is the cut volume.

Between the two stages this function owns what every compression shares. It sums the moments from the cut rule,
drops the compressed weights within `_WEIGHT_TOL` of zero relative to the cut volume — round-off around the exact zeros
a compression leaves — and measures the L² moment residual `‖Σ w_q ψ(x_q) − m‖` of the rule it hands back, after that
truncation, so the residual never describes a denser rule than the caller receives. `compression = nothing` hands the
cut rule back as it is, with a zero residual.

`status` reports which rule came back, and the integration-region dispatcher in `intersections.jl` turns it into the
region's quadrature kind:

  - `:fitted` — the compressed rule's residual is within the failure threshold `_FIT_FAILURE_RESIDUAL · ‖m‖` (1e-2,
    relative to the moment vector, so the verdict does not depend on the cell's size). Region kind `:cut_fitted`.
  - `:fallback` — the compressed rule missed the failure threshold, so the returned rule is the cut rule itself, the
    very data the moments were summed from, uncompressed. Region kind `:cut_fallback`.
  - `:empty` — `Ω ∩ region_box` carries no cut rule at all, or every compressed weight falls below the truncation
    cutoff so the rule would hold no points; `points` and `weights` are empty and `residual` is zero. Region kind
    `:cut_failed` (`:cut_alpha_failed` under α-FCM).

The fallback loses no accuracy: the cut rule is by construction at least as accurate as a successful compression of it,
and the package's cut rule has non-negative weights — each a product of a positive Gauss weight, a positive fiber
half-length and a positive base weight (see `_emit_fiber!` in `src/implicit.jl`). What it costs is points: 10³–10⁵ per
cell where a compressed rule carries at most `nbasis`, a 50–200× per-cell blow-up in assembly work on the many-feature
cells where a fallback is plausible at all. `AssemblyDiagnostics.cut_fallback_count` and `cut_fallback_points` report
how often it fired and what it cost.
"""
function moment_fit_rule(physical::PhysicalDomain, region_box::AxisBox{D,T}, moment_order::NTuple{D,Int}) where {D,T}
    points, weights = physical.cut_rule(physical, region_box, moment_order)
    isempty(weights) && return SVector{D,T}[], T[], zero(T), :empty
    physical.compression === nothing && return points, weights, zero(T), :fitted
    moments = _moments_from_rule(points, weights, region_box, moment_order)
    fitted, fitted_weights = physical.compression(points, weights, moments, region_box, moment_order)
    # A compression that truncates to nothing has no rule to offer, and a denser cut rule would not change that: hand
    # back the same `:empty` an empty cut rule gives, so the caller drops the region (or keeps its α-stabilised tensor
    # part) instead of consuming a zero-point rule as a successful fit.
    kept = findall(w -> abs(w) > _WEIGHT_TOL * moments[1], fitted_weights)
    isempty(kept) && return SVector{D,T}[], T[], zero(T), :empty
    fitted, fitted_weights = fitted[kept], fitted_weights[kept]
    residual = norm(_moments_from_rule(fitted, fitted_weights, region_box, moment_order) - moments)
    residual <= _FIT_FAILURE_RESIDUAL * norm(moments) && return fitted, fitted_weights, residual, :fitted
    return points, weights, residual, :fallback
end

# Whether a stage of the cut-cell rule may run on several threads at once. The package's own are pure functions of
# their arguments; a caller's callable is under no such contract, so a plan that meets one runs its fits serially.
_threadsafe(_) = false
_threadsafe(::Union{Nothing,typeof(implicit_rule),NonnegativeFit,Caratheodory}) = true
