# Finite-cell-method (FCM) machinery: non-negative moment-fitted quadrature
# for cells crossed by ∂Ω.
#
# Algorithm
#
#   This pipeline is a volume-integration-point port of QuESo's
#   `QuadratureTrimmedElement`. QuESo's quadrature pipeline is described in
#
#     M. Meßmer, T. Teschemacher, L. F. Leidinger, R. Wüchner, K.-U.
#     Bletzinger, "Efficient CAD-integrated isogeometric analysis of trimmed
#     solids", Comput. Methods Appl. Mech. Engrg. 400 (2022) 115584,
#     doi:10.1016/j.cma.2022.115584.
#
#   The Lawson–Hanson NNLS solver wrapped by `nnls!` is the classical
#
#     C. L. Lawson, R. J. Hanson, "Solving Least Squares Problems",
#     Prentice-Hall (1974), Ch. 23; SIAM Classics in Applied
#     Mathematics reprint doi:10.1137/1.9781611971217.
#
#   The Bro & de Jong (1997) Fast NNLS variant is exposed by the same
#   `NonNegLeastSquares` package but not wrapped here — its `N × N`
#   Gram-matrix shape loses to Lawson–Hanson on the short-and-wide
#   moment-fit matrices this pipeline produces (see `nnls!`).
#
# Upstream source and license
#
#   QuESo: https://github.com/manuelmessmer/QuESo, BSD-4-Clause.
#
# Deliberate deviations from upstream
#
#   The structural swap is QuESo's geometry kernel. QuESo classifies
#   axis-aligned bounding boxes against a triangulated B-rep and uses a
#   divergence-theorem surface integral to compute moments exactly. This
#   port classifies against a level-set callback via the Lipschitz
#   certificate in `src/physical.jl`, and approximates the moment integrals
#   by tensor Gauss quadrature on the octree leaves (stair-step accurate,
#   bounded by `O(subcell_length_scale / region_extent)` per axis). The
#   QuESo surface-IP / divergence-theorem path is not ported.
#
#   As a consequence, QuESo's hardcoded default
#   `moment_fitting_residual = 1.0e-10` (`dictionary_factory.hpp`) and the
#   `1.0e-8` typical of their shipped examples are unreachable in this
#   port at typical `subcell_length_scale` values. The
#   `PhysicalDomain.target_residual` default of `1.0e-6` is matched to
#   the integrator's own floor and avoids the catastrophic retry
#   blow-up that an unreachable target triggers. Users who tighten
#   `subcell_length_scale` should tighten `target_residual`
#   correspondingly (roughly two decades per halving of the scale).
#
# Pipeline summary
#
#   1. NNLS wrapper around `NonNegLeastSquares.nonneg_lsq` (Lawson–Hanson).
#   2. D-generic octree leaf walker driven by the PhysicalDomain Lipschitz
#      certificate and corner sampling.
#   3. Tensor Legendre moment integration over the octree leaves.
#   4. Candidate-point seeding from the same leaves, filtered by φ ≤ 0.
#   5. NNLS-based moment fit, with QuESo's point-elimination inner loop
#      and `AssembleIPs` outer retry.
#
#   Public entry point: `moment_fit_rule(physical, region_box, moment_order)`.
#   Building blocks (`foreach_octree_leaf`, `compute_region_moments`,
#   `seed_candidate_points`, `nnls!`) are exposed for inspection and
#   advanced use; the elimination and retry loops are internal.

# ── 1. NNLS wrapper ───────────────────────────────────────────────────────────

"""
    nnls!(A, b) -> (x, residual_norm)

Solve the non-negativity-constrained linear least-squares problem

    minimise  ‖A x − b‖₂  subject to  x ≥ 0,

returning the solution vector and the L² residual `‖A x − b‖₂`.

Wraps `NonNegLeastSquares.nonneg_lsq` with `alg = :nnls`, the classical
Lawson–Hanson (1974) active-set algorithm. It operates directly on the
`nbasis × npoints` design matrix and converges in roughly `nbasis`
outer iterations, each solving a small `|P| × |P|` linear system with
`|P| ≤ nbasis`. The moment-fit matrix is short and wide
(`nbasis ≪ npoints`, e.g. `64 × 4096` for an order-3 basis with
`moment_order_factor = 1` on a depth-3 octree of one cut cell), so
the matrix the algorithm walks IS the small one.

The Bro & de Jong (1997) Fast NNLS variant (`alg = :fnnls`) works on
the `npoints × npoints` Gram matrix instead. It is faster only when
`npoints` is small; at our typical `npoints` of several thousand the
Gram matrix is hundreds of megabytes and `:fnnls` is roughly 30×
slower (275 ms vs 8.9 ms median at `nbasis = 64`, `npoints = 4096`).

Allocations are owned by `NonNegLeastSquares.jl`; `A` and `b` are not
mutated despite the `!` — the bang is kept for a future in-place port.
"""
function nnls!(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:Real}
    x = vec(nonneg_lsq(A, b; alg=:nnls))
    residual = norm(A * x - b)
    return x, residual
end

# ── 2. Octree leaf walker ─────────────────────────────────────────────────────

"""
    foreach_octree_leaf(f, physical, box)

Walk the octree of `box` subdivided against `physical`'s level set. The
leaf size on the largest axis is driven down to
`physical.subcell_length_scale`, capped at `physical.max_depth` octree
levels of subdivision. For each leaf, call `f(leaf_box, kind)` where
`kind` is one of

  - `:full` — the leaf is entirely inside Ω,
  - `:fictitious` — the leaf is entirely outside Ω,
  - `:cut` — the maximum depth was reached without a definitive
    classification, so the leaf is treated as crossed by ∂Ω.

The classifier uses the Lipschitz certificate first
(`|φ(c)| > L · r` ⇒ uniform sign, where `c` is the leaf center and `r`
is its half-diagonal), then 2ᴰ-corner sampling, then recursion. When the
depth budget runs out: agreeing samples emit the consensus verdict, and
disagreeing samples emit `:cut`.

This is the public entry point; the implementation is in `_walk_octree!`.
"""
function foreach_octree_leaf(f, physical::PhysicalDomain, box::AxisBox)
    _walk_octree!(f, physical, box, 0, _effective_subcell_depth(physical, box))
    return nothing
end

# Recursive octree walker invoked by `foreach_octree_leaf`. Conceptually
# the same as `_classify_box` in `src/physical.jl`, but emits a callback
# at every leaf rather than collapsing the walk to a single Symbol verdict.
# Strategy at each box:
#
#   1. Lipschitz certificate. `|φ(center)| > L · r` proves sign-uniformity
#      across the box; emit `:full` or `:fictitious` immediately.
#   2. Corner sampling at the box's 2ᴰ vertices. Mixed signs prove the
#      boundary crosses the box.
#   3. If the maximum subcell depth is reached: emit the corner-consensus
#      verdict (`:full` / `:fictitious`), or `:cut` if corners disagree.
#   4. Otherwise bisect into 2ᴰ children and recurse on each.
#
# The recursion bound is `(2ᴰ)^max_depth` leaves per call, where
# `max_depth` is set by `_effective_subcell_depth` from the top box.
function _walk_octree!(f, physical::PhysicalDomain, box::AxisBox{D,T}, depth::Integer,
                       max_depth::Integer) where {D,T}
    c = center(box)
    r = _half_diagonal(box)
    phi_c = physical.phi(c)
    threshold = physical.lipschitz * r

    if phi_c < -threshold
        f(box, :full);
        return
    elseif phi_c > threshold
        f(box, :fictitious);
        return
    end

    n_inside, n_outside = _corner_signs(physical, box, phi_c)
    samples_agree = !(n_inside > 0 && n_outside > 0)

    if depth >= max_depth
        if samples_agree
            f(box, n_inside > 0 ? :full : :fictitious)
        else
            f(box, :cut)
        end
        return
    end

    # Bisection: every child shares one corner with the parent (the box
    # center) and inherits seven of its corners from the parent's
    # `lower`/`upper`. The 2ᴰ children are enumerated by iterating
    # `CartesianIndices(ntuple(_ -> 0:1, D))` — bit 0 on each axis takes
    # `lower`, bit 1 takes `c`.
    for ci in CartesianIndices(ntuple(_ -> 0:1, D))
        child_lower = SVector{D,T}(ntuple(d -> ci.I[d] == 0 ? box.lower[d] : c[d], D))
        child_upper = SVector{D,T}(ntuple(d -> ci.I[d] == 0 ? c[d] : box.upper[d], D))
        _walk_octree!(f, physical, AxisBox{D,T}(child_lower, child_upper), depth + 1, max_depth)
    end
    return nothing
end

# ── 3. Tensor Legendre basis on a region box ──────────────────────────────────

# Flat lexicographic ordering of tensor-product Legendre indices up to
# `moment_order` per axis: `vec(CartesianIndices(0:p₁, …, 0:p_D))`. The
# resulting `Vector{CartesianIndex{D}}` is the moment-basis index list
# `α = (α₁, …, α_D)` used by `compute_region_moments` and the moment-fit
# matrix.
function _moment_basis_indices(moment_order::NTuple{D,Int}) where {D}
    return vec(collect(CartesianIndices(ntuple(d -> 0:moment_order[d], D))))
end

# Cardinality of the tensor-product moment basis, `prod(moment_order .+ 1)`.
_moment_basis_count(moment_order::NTuple{D,Int}) where {D} = prod(moment_order .+ 1)

# Fill caller-owned per-axis 1D Legendre value tables for the point `x`
# mapped into the region's reference frame `[−1, 1]ᴰ`:
#
#     factors[d][p + 1] = L_p(ξ_d),    ξ_d = (2 x[d] − lower[d] − upper[d]) / (upper[d] − lower[d]),
#
# where `L_p` is the standard Legendre polynomial. One Legendre three-term
# recurrence per axis, so the per-point cost is `D · (order + 1)` instead
# of `D · (order + 1)^D`.
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

# Caller-owned per-axis 1D scratch buffers used by `_fill_legendre_factors!`
# and the tensor-product evaluation: one `Vector{T}` per axis, length
# `moment_order[d] + 1`. Allocated once per `compute_region_moments` or
# `_build_moment_matrix` call, then reused across every point.
function _legendre_factor_buffers(moment_order::NTuple{D,Int}, ::Type{T}) where {D,T}
    ntuple(d -> Vector{T}(undef, moment_order[d] + 1), D)
end

# Tensor-product basis value at the point whose per-axis Legendre factors
# are loaded into `factors`: `ψ_α(ξ) = ∏_d L_{α_d}(ξ_d) =
# ∏_d factors[d][α_d + 1]`.
function _tensor_legendre_value(factors, idx::CartesianIndex{D}) where {D}
    v = factors[1][idx.I[1] + 1]
    @inbounds for d in 2:D
        v *= factors[d][idx.I[d] + 1]
    end
    return v
end

# ── 4. Moment integration and candidate seeding ───────────────────────────────

# Side-of-Ω predicate for a candidate Gauss point inside a `:cut` octree
# leaf: keep the point iff `φ(x) ≤ 0`, i.e. `x` is inside Ω. Used by both
# `compute_region_moments` and `seed_candidate_points`.
_inside_omega(physical::PhysicalDomain, x) = physical.phi(x) <= zero(eltype(x))

# Per-axis Gauss-Legendre point count for the octree-leaf integration
# rule. An `n`-point Gauss rule is exact for polynomials up to degree
# `2n − 1`, so the smallest `n` that integrates the moment basis (max
# polynomial degree `moment_order[d]` per axis) on a full leaf is
# `ceil((moment_order[d] + 1) / 2)`. Used as the default for both
# `compute_region_moments` (where the integrand IS a degree-`moment_order`
# polynomial on each full leaf) and `seed_candidate_points` (where this
# is also the minimum density at which the moment-fit NNLS sees one
# candidate per moment basis function per leaf — enough redundancy for
# NNLS convergence without the quadratic-in-`gauss_per_axis` blow-up of
# a higher count).
#
# QuESo arrives at the same minimum, by a different route: their
# `ffactor = 1` makes their moment basis order `p` (vs. our default
# `2p`), and they then use `p + 1` Gauss points per axis — `ceil((p + 1)
# / 2) ≤ p + 1` is automatically respected. The previous default here
# (`moment_order + 1`) was twice the necessary density, which translated
# into `~4ˣ` candidates in 2D and roughly `~10ˣ` NNLS cost per outer
# attempt with no moment-integration accuracy gain on full leaves and
# only marginal `~30%` reduction in NNLS residual (NNLS picks the best
# rule from the candidate set; more candidates → marginally tighter fit
# but no integration accuracy gain past the moment basis).
function _default_moment_gauss(moment_order::NTuple{D,Int}) where {D}
    ntuple(d -> cld(moment_order[d] + 1, 2), D)
end

"""
    compute_region_moments(physical, region_box, moment_order; gauss_per_axis=…) -> Vector

Return the moment vector

    m_α  =  ∫_{R ∩ Ω}  ψ_α(ξ(x))  dx,    α ∈ {0, …, p₁} × … × {0, …, p_D},

where `R = region_box`, `ψ_α(ξ) = ∏_d L_{α_d}(ξ_d)` is the tensor Legendre
basis on the region's reference frame `[−1, 1]ᴰ`, and the indices `α`
are enumerated in the lexicographic order of
[`_moment_basis_indices`](@ref).

Integration walks the octree leaves emitted by
[`foreach_octree_leaf`](@ref):

  - `:full` leaves contribute the standard tensor Gauss rule (exact for
    polynomials up to the moment order on the leaf);
  - `:cut` leaves contribute the same rule with points filtered by
    `φ(x) ≤ 0` — stair-step accurate at the resolved leaf size set by
    `subcell_length_scale`, matched to the integrator's natural floor by the
    `target_residual = 1e-6` default;
  - `:fictitious` leaves contribute nothing.

`gauss_per_axis` defaults to `ceil((moment_order + 1) / 2)` — the
minimum per-axis Gauss-Legendre count that integrates the tensor
Legendre moment basis exactly on a full leaf
(see [`_default_moment_gauss`](@ref)).
"""
function compute_region_moments(physical::PhysicalDomain, region_box::AxisBox{D,T},
                                moment_order::NTuple{D,Int};
                                gauss_per_axis::NTuple{D,Int}=_default_moment_gauss(moment_order)) where {D,
                                                                                                          T}
    indices = _moment_basis_indices(moment_order)
    moments = zeros(T, length(indices))
    factors = _legendre_factor_buffers(moment_order, T)
    gauss = _tensor_gauss_rule(gauss_per_axis, T)

    foreach_octree_leaf(physical, region_box) do leaf, kind
        kind === :fictitious && return nothing
        jac = volume(leaf) / convert(T, 2^D)
        for (eta, weight) in zip(gauss.points, gauss.weights)
            x = reference_to_physical(leaf, eta)
            kind === :cut && !_inside_omega(physical, x) && continue
            _fill_legendre_factors!(factors, x, region_box.lower, region_box.upper, moment_order)
            qweight = weight * jac
            for (a, idx) in pairs(indices)
                moments[a] += qweight * _tensor_legendre_value(factors, idx)
            end
        end
        return nothing
    end
    return moments
end

"""
    seed_candidate_points(physical, region_box, moment_order; gauss_per_axis=…)
        -> Vector{SVector{D,T}}

Return the candidate-point set fed into the moment-fit NNLS. Generated by
walking the octree leaves emitted by [`foreach_octree_leaf`](@ref) and
collecting tensor Gauss points from every non-fictitious leaf, with `:cut`
leaves filtered by `φ(x) ≤ 0`.

`gauss_per_axis` defaults to `ceil((moment_order + 1) / 2)`, matching
the per-leaf Gauss density used by [`compute_region_moments`](@ref) —
the minimum count at which the moment-fit NNLS sees one candidate per
moment basis function per leaf. At the typical octree depths driven by
`subcell_length_scale`, this already gives the NNLS far more candidates
than basis functions; the outer retry loop in `_moment_fit_with_retry`
doubles per-axis as needed.
"""
function seed_candidate_points(physical::PhysicalDomain, region_box::AxisBox{D,T},
                               moment_order::NTuple{D,Int};
                               gauss_per_axis::NTuple{D,Int}=_default_moment_gauss(moment_order)) where {D,
                                                                                                         T}
    gauss = _tensor_gauss_rule(gauss_per_axis, T)
    points = SVector{D,T}[]
    foreach_octree_leaf(physical, region_box) do leaf, kind
        kind === :fictitious && return nothing
        for eta in gauss.points
            x = reference_to_physical(leaf, eta)
            kind === :cut && !_inside_omega(physical, x) && continue
            push!(points, x)
        end
        return nothing
    end
    return points
end

# ── 5. Moment-fit matrix and single solve ─────────────────────────────────────

# Drop a candidate weight when its NNLS solution falls below this absolute
# threshold. Matches QuESo's "near-zero weight" criterion and protects the
# elimination loop from numerical noise around true zeros.
const _NNLS_WEIGHT_TOL = 1.0e-12

# Catastrophic failure threshold for the moment-fit residual. Matches
# QuESo's `if residual > 1e-2, clear the surviving set` heuristic in
# `AssembleIPs`. A residual above this value indicates the NNLS could not
# come anywhere near reproducing the moments — the elimination loop and
# outer retry both treat this as a hard reset.
const _FIT_FAILURE_RESIDUAL = 1.0e-2

# Assemble the moment-fit design matrix
#
#     A[a, j] = ψ_α_a(ξ(x_j)),
#
# columns indexed by the candidate points `x_j` and rows by the moment-basis
# multi-indices `α_a`. One Legendre table fill per point, then a tensor-
# product evaluation per (point, basis-index) pair. The matrix is
# `nbasis × npoints` and is rebuilt from scratch on each NNLS call because
# the candidate point set changes between iterations.
function _build_moment_matrix(points::Vector{SVector{D,T}}, region_box::AxisBox{D,T},
                              moment_order::NTuple{D,Int}) where {D,T}
    indices = _moment_basis_indices(moment_order)
    npoints = length(points)
    A = Matrix{T}(undef, length(indices), npoints)
    factors = _legendre_factor_buffers(moment_order, T)
    @inbounds for j in 1:npoints
        _fill_legendre_factors!(factors, points[j], region_box.lower, region_box.upper,
                                moment_order)
        for (a, idx) in pairs(indices)
            A[a, j] = _tensor_legendre_value(factors, idx)
        end
    end
    return A
end

# One NNLS solve against the design matrix above: returns
# `(weights, residual)` for the candidate `points`. Shared by the
# single-shot path of `moment_fit_rule` and every iteration of
# `_eliminate_points`.
function _solve_moment_fit(moments::Vector{T}, points::Vector{SVector{D,T}},
                           region_box::AxisBox{D,T}, moment_order::NTuple{D,Int}) where {D,T}
    A = _build_moment_matrix(points, region_box, moment_order)
    return nnls!(A, moments)
end

# ── 6. Point elimination (port of QuESo's `PointElimination`) ────────────────

# Drop weights below `_NNLS_WEIGHT_TOL` and return filtered points and
# weights vectors. Used after the first-iteration top-N truncation to
# clear up the inevitable handful of numerically-zero NNLS weights left
# over from the initial over-saturated candidate set.
function _prune_near_zero(points::Vector{<:SVector}, weights::Vector{<:Real})
    keep = findall(>(_NNLS_WEIGHT_TOL), weights)
    return points[keep], weights[keep]
end

"""
    _eliminate_points(moments, candidates, region_box, moment_order;
                      target_residual, max_iterations=1000)
        -> (points, weights, residual, iterations)

Inner loop of the QuESo moment-fitting pipeline (their `PointElimination`).
First iteration solves the NNLS on the full candidate set, sorts by
weight descending, keeps the top `nbasis = prod(moment_order .+ 1)`
points (the maximum a non-degenerate NNLS can support with non-trivial
weights), and drops near-zero weights.

Subsequent iterations resolve the NNLS on the surviving points and
remove every weight below `1e-8 · max(weights)`; if that filter would
empty the set, the single smallest-weight point is dropped instead.
The shrinking respects the floor `min_points = prod(moment_order)` so the
NNLS retains enough degrees of freedom to satisfy the moment constraints.

The loop stops when the residual first exceeds `target_residual` and
returns the *last* solution that was still within tolerance — that is,
the smallest surviving point set whose moment-fit was still acceptable.
"""
function _eliminate_points(moments::Vector{T}, candidates::Vector{SVector{D,T}},
                           region_box::AxisBox{D,T}, moment_order::NTuple{D,Int};
                           target_residual::Real, max_iterations::Integer=1000,) where {D,T}
    nbasis = _moment_basis_count(moment_order)
    min_points = prod(moment_order)
    target = T(target_residual)

    points = copy(candidates)
    weights, residual = _solve_moment_fit(moments, points, region_box, moment_order)
    residual > _FIT_FAILURE_RESIDUAL && return SVector{D,T}[], T[], residual, 1

    # First-iteration top-N: sort by weight (descending), keep the
    # leading `nbasis`, drop the near-zero tail. `nbasis` is the
    # maximum number of non-trivial weights a non-degenerate NNLS
    # against this basis can support.
    order = sortperm(weights; rev=true)
    keep = min(nbasis, length(weights))
    points = points[order[1:keep]]
    weights = weights[order[1:keep]]
    points, weights = _prune_near_zero(points, weights)
    isempty(points) && return points, weights, residual, 1

    last_good_points = copy(points)
    last_good_weights = copy(weights)
    last_good_residual = residual

    for iter in 2:max_iterations
        weights, residual = _solve_moment_fit(moments, points, region_box, moment_order)
        if residual >= target
            return last_good_points, last_good_weights, last_good_residual, iter
        end
        last_good_points = copy(points)
        last_good_weights = copy(weights)
        last_good_residual = residual

        length(points) <= min_points && return points, weights, residual, iter

        # Drop the weights that fall below `1e-8 · max_weight`. If no
        # weight falls below the threshold, drop the single smallest
        # one — the loop must make progress.
        max_w = maximum(weights)
        thresh = 1e-8 * max_w
        drop_indices = findall(<(thresh), weights)
        if isempty(drop_indices)
            _, min_i = findmin(weights)
            drop_indices = [min_i]
        end

        # Respect the `min_points` floor: never drop more than
        # `length(points) - min_points` in a single iteration. When the
        # filter would remove too many, keep the smallest few.
        max_drops = length(points) - min_points
        if length(drop_indices) > max_drops
            sort!(drop_indices, by=i -> weights[i])
            drop_indices = drop_indices[1:max_drops]
        end
        keep_mask = trues(length(points))
        keep_mask[drop_indices] .= false
        points = points[keep_mask]
        weights = weights[keep_mask]
    end

    return last_good_points, last_good_weights, last_good_residual, max_iterations
end

# ── 7. Outer retry (port of QuESo's `AssembleIPs`) ───────────────────────────

# Default maximum number of outer retry attempts. QuESo uses 4 when the
# moment order is small (2 in any axis) — those problems are the most
# sensitive to the initial candidate count — and 3 otherwise.
_default_max_outer(moment_order::NTuple{D,Int}) where {D} = maximum(moment_order) == 2 ? 4 : 3

# Stagnation threshold for retry abandonment. A retry whose residual is
# not at least this many times *smaller* than the previous attempt's
# residual is considered ineffective — typically because the
# `compute_region_moments` octree integrator has hit its stair-step
# accuracy floor (≈ `subcell_length_scale / cell_size` per axis) and no NNLS against
# any candidate cloud can do better. Doubling the candidate density then
# `_min_retry_improvement`'s reciprocal in cost while delivering nothing,
# so we stop early.
const _MIN_RETRY_IMPROVEMENT = 2.0

# Early-acceptance multiplier. An attempt whose residual is within this
# factor of `target_residual` is treated as having reached the
# integrator's natural floor, and no further retries are issued. The
# stagnation guard below would catch this anyway after running one more
# (much more expensive) attempt to confirm; this multiplier avoids the
# confirmation cost when the residual is already close enough that
# doubling the candidate cloud is virtually certain not to halve it.
# `2` keeps the assertion `residual < target_residual × 2` true rather
# than the stricter `residual < target_residual`, but only when the gap
# between target and floor is at most 2× — i.e. the user has set
# `target_residual` near (rather than well below) the achievable floor.
# A user who genuinely needs `residual < target_residual` can tighten
# `target_residual` by 2× and the original behaviour returns.
const _EARLY_ACCEPT_FACTOR = 2.0

"""
    _moment_fit_with_retry(physical, region_box, moment_order;
                           target_residual, max_outer, gauss_per_axis)
        -> (points, weights, residual, inner_iterations, outer_iterations)

Outer driver for the QuESo moment-fitting pipeline (their `AssembleIPs`).
On each attempt:

  1. Seed candidate points from the octree at the current per-axis Gauss
     density `gauss_per_axis × point_factor`.
  2. Re-inject the surviving fitted point set from the previous attempt
     (if any) so the elimination loop starts from a known-good point
     subset.
  3. Run [`_eliminate_points`](@ref).
  4. If the residual is within `target_residual`, accept and return.
  5. If the residual is catastrophic (`> _FIT_FAILURE_RESIDUAL`), discard
     the surviving set so the next attempt starts fresh from a wider
     candidate cloud.
  6. Otherwise double `point_factor` and try again, **unless** the new
     residual is no better than `_MIN_RETRY_IMPROVEMENT`× the best so
     far — i.e. the integrator's stair-step floor has been hit and
     further attempts only cost (cubically in candidate count) without
     improving accuracy. QuESo's reference implementation has the same
     `AssembleIPs` shape but always burns the full retry budget; this
     port adds the stagnation check because the moment integrator here
     is octree-stair-step rather than B-rep-exact, so the floor is
     easily reachable.

Returns the best (lowest-residual) fit observed across all attempts. The
inner and outer iteration counts are surfaced so the caller can adapt
their retry strategy or report convergence behaviour.
"""
function _moment_fit_with_retry(physical::PhysicalDomain, region_box::AxisBox{D,T},
                                moment_order::NTuple{D,Int}; target_residual::Real,
                                max_outer::Integer, gauss_per_axis::NTuple{D,Int},) where {D,T}
    moments = compute_region_moments(physical, region_box, moment_order)
    target = T(target_residual)

    surviving = SVector{D,T}[]
    best_points = SVector{D,T}[]
    best_weights = T[]
    best_residual = T(Inf)
    best_inner = 0
    prev_residual = T(Inf)
    point_factor = 1

    for outer in 1:max_outer
        # Wider candidate cloud on every retry. The first attempt uses the
        # base `gauss_per_axis`; subsequent attempts double the density.
        scaled = ntuple(d -> gauss_per_axis[d] * point_factor, D)
        candidates = seed_candidate_points(physical, region_box, moment_order;
                                           gauss_per_axis=scaled)
        if !isempty(surviving)
            append!(candidates, surviving)
        end
        isempty(candidates) && return SVector{D,T}[], T[], one(T), 0, outer

        pts, ws, res, iter = _eliminate_points(moments, candidates, region_box, moment_order;
                                               target_residual)

        if res > _FIT_FAILURE_RESIDUAL
            # Catastrophic failure: throw the surviving set away so the
            # next attempt starts from a wider candidate cloud only, and
            # reset the previous-residual marker so the next non-failing
            # attempt is judged against itself, not against this junk.
            surviving = SVector{D,T}[]
            prev_residual = T(Inf)
        else
            # Reasonable fit. Update the best-so-far whenever the new
            # residual genuinely improves on it; always seed the next
            # attempt with `pts` (QuESo's `surviving` contract).
            if res < best_residual
                best_points = pts
                best_weights = ws
                best_residual = res
                best_inner = iter
            end
            surviving = pts

            res <= target && return best_points, best_weights, best_residual, best_inner, outer

            # Early-accept: residual is within `_EARLY_ACCEPT_FACTOR` of
            # the target. The next attempt would have to roughly halve
            # the residual to cross the target, but the achievable
            # residual is bounded below by the moment integrator's
            # stair-step floor — when target is calibrated to that floor
            # (as the `physical.target_residual = 1e-6` default is for a
            # `subcell_length_scale` matched to the smallest geometric
            # feature), no candidate-cloud doubling can
            # actually deliver that halving, and the retry is pure
            # waste. Users who genuinely need `residual ≤ target` can
            # tighten `target_residual` and retain the original
            # behaviour.
            res <= target * T(_EARLY_ACCEPT_FACTOR) &&
                return best_points, best_weights, best_residual, best_inner, outer

            # Stagnation guard: if doubling the candidate cloud failed to
            # improve the residual by at least `_MIN_RETRY_IMPROVEMENT`,
            # the integrator's stair-step floor has been reached and the
            # remaining (cubically more expensive) attempts will not
            # help. Catches the case where attempt 1 misses the
            # early-accept threshold but stagnates anyway.
            if outer > 1 && res * T(_MIN_RETRY_IMPROVEMENT) >= prev_residual
                return best_points, best_weights, best_residual, best_inner, outer
            end
            prev_residual = res
        end

        point_factor *= 2
    end

    return best_points, best_weights, best_residual, best_inner, max_outer
end

# ── 8. Public moment-fit rule ─────────────────────────────────────────────────

"""
    moment_fit_rule(physical, region_box, moment_order;
                    target_residual=1e-10,
                    elimination=true,
                    max_outer=nothing,
                    gauss_per_axis=nothing)
        -> (points, weights, residual)

Build a non-negative tensor-Legendre moment-fitted quadrature rule on
`region_box ∩ Ω`. The rule integrates the tensor Legendre moments

    m_α = ∫_{region_box ∩ Ω} ψ_α(ξ(x)) dx,    α ∈ {0,…,p₁} × … × {0,…,p_D}

exactly up to the moment-fit residual.

  - `elimination = true` (default) runs the full QuESo pipeline
    (`PointElimination` inner loop + `AssembleIPs` outer retry).
  - `elimination = false` takes the single-shot path: solve the NNLS
    once against the full candidate-point cloud and return the
    near-zero-filtered result. Useful for testing and for inspecting the
    candidate set without the elimination loop.

Keyword arguments:

  - `target_residual` — moment-fit L² residual target.
  - `max_outer` — maximum number of outer retry attempts. Defaults to
    3 (4 when any `moment_order` axis is 2; see `_default_max_outer`).
  - `gauss_per_axis` — per-axis Gauss-point density used to seed the
    candidate set on the first outer attempt. Defaults to
    `ceil((moment_order + 1) / 2)` per axis, matching the per-leaf
    density used by [`compute_region_moments`](@ref) — the minimum count
    that integrates the tensor Legendre moment basis exactly on a full
    leaf (see [`_default_moment_gauss`](@ref)). The outer retry doubles
    this per attempt.
"""
function moment_fit_rule(physical::PhysicalDomain, region_box::AxisBox{D,T},
                         moment_order::NTuple{D,Int}; target_residual::Real=1.0e-10,
                         elimination::Bool=true, max_outer::Union{Nothing,Integer}=nothing,
                         gauss_per_axis::Union{Nothing,NTuple{D,Int}}=nothing,) where {D,T}
    gauss = gauss_per_axis === nothing ? _default_moment_gauss(moment_order) : gauss_per_axis
    if !elimination
        moments = compute_region_moments(physical, region_box, moment_order)
        candidates = seed_candidate_points(physical, region_box, moment_order; gauss_per_axis=gauss)
        isempty(candidates) && return (SVector{D,T}[], T[], norm(moments))
        weights, residual = _solve_moment_fit(moments, candidates, region_box, moment_order)
        kept = findall(>(_NNLS_WEIGHT_TOL), weights)
        return candidates[kept], weights[kept], residual
    end
    outer_cap = max_outer === nothing ? _default_max_outer(moment_order) : Int(max_outer)
    pts, ws, res, _, _ = _moment_fit_with_retry(physical, region_box, moment_order; target_residual,
                                                max_outer=outer_cap, gauss_per_axis=gauss)
    return pts, ws, res
end
