# Finite-cell-method (FCM) machinery: non-negative moment-fitted quadrature
# for cells crossed by ∂Ω.
#
# Algorithm
#
#   For a cut cell the moments of a tensor Legendre basis over Ω ∩ R,
#
#     mₐ = ∫_{Ω ∩ R} ψₐ(ξ(x)) dx,
#
#   are computed exactly by the Saye dimension-reduction implicit-quadrature
#   kernel in `src/implicit.jl` (the CSG level set in
#   `src/physical.jl` supplies the leaves and the membership predicate). A
#   non-negative least-squares (NNLS) moment fit then selects quadrature
#   weights at the kernel's volume-rule points so the weighted point rule
#   reproduces the moments — the moment-fitting idea of
#
#     B. Müller, F. Kummer, M. Oberlack, "Highly accurate surface and volume
#     integration on implicit domains by means of moment-fitting", Int. J.
#     Numer. Methods Engng. 96 (2013) 512–528, doi:10.1002/nme.4569.
#
#   The NNLS moment-fit structure follows QuESo's `QuadratureTrimmedElement`
#   (M. Meßmer et al., Comput. Methods Appl. Mech. Engrg. 400 (2022) 115584,
#   doi:10.1016/j.cma.2022.115584; https://github.com/manuelmessmer/QuESo,
#   BSD-4-Clause). The Lawson–Hanson NNLS solver wrapped by `nnls` is the
#   classical C. L. Lawson, R. J. Hanson, "Solving Least Squares Problems",
#   Prentice-Hall (1974), Ch. 23 (SIAM reprint doi:10.1137/1.9781611971217).
#
# Deviations from QuESo
#
#   QuESo classifies axis-aligned boxes against a triangulated B-rep and
#   computes moments from a divergence-theorem surface integral. This package
#   classifies against a CSG level set (`PhysicalDomain`) and computes the
#   moments from the Saye implicit-quadrature kernel — exact and
#   octree-depth-independent on smooth (graph-like) cut cells, machine precision
#   for linear leaves and polytope corners. There is a single integrator: the
#   octree stair-step moment integrator, QuESo's `PointElimination` inner loop
#   and its accuracy-driven `AssembleIPs` retry are not used, because the exact
#   kernel removes the stair-step accuracy floor those loops worked around.
#   Non-graph-like cells (high curvature relative to the cell) are handled by the
#   kernel's bounded subdivision, so no separate fallback path is needed.
#
#   `moment_fit_rule` does carry an outer retry of its own, and it is a different
#   mechanism from QuESo's: a bounded three-attempt densification of the NNLS
#   candidate cloud, there purely for the conditioning of the least-squares
#   solve. It never searches for points to eliminate from a fitted rule, and
#   never deepens the subdivision the kernel is allowed.
#
# Pipeline summary
#
#   1. NNLS wrapper around `NonNegLeastSquares.nonneg_lsq` (Lawson–Hanson).
#   2. Saye implicit volume rule (`implicit_volume_quadrature`) over Ω ∩ R.
#   3. Exact moments by `mₐ = Σ_q w_q ψₐ(x_q)`; the rule's points are the
#      moment-fit candidates, capped at a multiple of nbasis that grows with the
#      retry index.
#   4. A single NNLS moment fit (Lawson–Hanson naturally yields ≤ nbasis
#      non-negative weights; a zero-residual non-negative solution exists, so
#      one solve reaches machine-level residual), then a truncation of the
#      near-zero weights at a cutoff relative to the cut volume, with the
#      residual re-measured on the weights that survive it.
#   5. If no attempt fits, fall back to the volume rule of step 2 itself — the
#      moments' own source data, correct but uncompressed.
#
#   Public entry point: `moment_fit_rule(physical, region_box, moment_order)`.

# ── 1. NNLS wrapper ───────────────────────────────────────────────────────────

"""
    nnls(A, b) -> (x, residual_norm)

Solve the non-negativity-constrained linear least-squares problem

    minimise  ‖A x − b‖₂  subject to  x ≥ 0,

returning the solution vector and the L² residual `‖A x − b‖₂`.

Wraps `NonNegLeastSquares.nonneg_lsq` with `alg = :nnls`, the classical
Lawson–Hanson (1974) active-set algorithm. It operates directly on the
`nbasis × npoints` design matrix and converges in roughly `nbasis` outer
iterations, each solving a small `|P| × |P|` linear system with `|P| ≤ nbasis`.
The moment-fit matrix is short and wide (`nbasis ≪ npoints`), so the matrix the
algorithm walks IS the small one.

The Bro & de Jong (1997) Fast NNLS variant (`alg = :fnnls`) works on the
`npoints × npoints` Gram matrix instead and is faster only when `npoints` is
small; at our typical `npoints` of several thousand it is far slower.

`max_iter` is raised to `10 · npoints` from the library default of
`3 · npoints`. Lawson–Hanson stays feasible at every step, so reaching the cap
is not a wrong answer — but the library announces it by printing
`NNLS quitting on iteration count` to stdout, and a library call has no business
writing there. The cells that reach it (a 3-D order-3 sphere is the case this
was found on) overrun the default by a few percent, so the wider cap lets the
active-set walk finish instead of being cut short. Raising a cap cannot change a
solve that converged under the old one: the cap only ever truncates.

Allocations are owned by `NonNegLeastSquares.jl`; `A` and `b` are not mutated,
so the name carries no `!`.
"""
function nnls(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:Real}
    x = vec(nonneg_lsq(A, b; alg=:nnls, max_iter=10 * size(A, 2)))
    residual = norm(A * x - b)
    return x, residual
end

# ── 2. Tensor Legendre basis on a region box ──────────────────────────────────

# Flat lexicographic ordering of tensor-product Legendre indices up to
# `moment_order` per axis: `vec(CartesianIndices(0:p₁, …, 0:p_D))`. The
# resulting `Vector{CartesianIndex{D}}` is the moment-basis index list
# `α = (α₁, …, α_D)` used by the moment vector and the moment-fit matrix.
function _moment_basis_indices(moment_order::NTuple{D,Int}) where {D}
    return vec(collect(CartesianIndices(ntuple(d -> 0:moment_order[d], D))))
end

# Cardinality of the tensor-product moment basis, `prod(moment_order .+ 1)`.
_moment_basis_count(moment_order::NTuple{D,Int}) where {D} = prod(moment_order .+ 1)

# Fill caller-owned per-axis 1D Legendre value tables for the point `x` mapped
# into the region's reference frame `[−1, 1]ᴰ`:
#
#     factors[d][p + 1] = L_p(ξ_d),    ξ_d = (2 x[d] − lower[d] − upper[d]) / (upper[d] − lower[d]),
#
# via one Legendre three-term recurrence per axis, so the per-point cost is
# `D · (order + 1)` instead of `D · (order + 1)^D`.
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

# Caller-owned per-axis 1D scratch buffers used by `_fill_legendre_factors!` and
# the tensor-product evaluation: one `Vector{T}` per axis, length
# `moment_order[d] + 1`. Allocated once per call, then reused across points.
function _legendre_factor_buffers(moment_order::NTuple{D,Int}, ::Type{T}) where {D,T}
    return ntuple(d -> Vector{T}(undef, moment_order[d] + 1), D)
end

# Tensor-product basis value at the point whose per-axis Legendre factors are
# loaded into `factors`: `ψ_α(ξ) = ∏_d L_{α_d}(ξ_d) = ∏_d factors[d][α_d + 1]`.
function _tensor_legendre_value(factors, idx::CartesianIndex{D}) where {D}
    v = factors[1][idx.I[1] + 1]
    @inbounds for d in 2:D
        v *= factors[d][idx.I[d] + 1]
    end
    return v
end

# ── 3. Moment-fit matrix and single solve ─────────────────────────────────────

# Drop a candidate weight when its NNLS solution falls below this threshold
# (numerical noise around true zeros). The number is dimensionless: a quadrature
# weight carries the units of a volume, so the cutoff is taken *relative to the
# cut volume* `∫_{Ω∩R} 1 dx` in `_solve_moment_fit` below. Compared against an
# absolute constant instead, the same truncation keeps every weight on a geometry
# of unit size and empties the rule on a small one — at a domain length scale of
# 1e-5 in 2D a correct cut weight is ~1e-11, an order of magnitude under a 1e-12
# cutoff. The breakpoint tolerance of `src/implicit.jl` and the mesh-SDF length
# tolerance of `ext/UnfittedMeshIOExt.jl` are scaled the same way, and for the
# same reason: the kernel has to behave identically under a rescaling of the
# geometry.
const _NNLS_WEIGHT_TOL = 1.0e-12

# Catastrophic failure threshold for the moment-fit residual. A residual above
# this value indicates the NNLS could not come anywhere near reproducing the
# moments; `moment_fit_rule` then returns the raw Saye volume rule instead of the
# fitted one, and the integration-plan dispatcher tags such a region
# `:cut_fallback`.
const _FIT_FAILURE_RESIDUAL = 1.0e-2

# Assemble the moment-fit design matrix
#
#     A[a, j] = ψ_α_a(ξ(x_j)),
#
# columns indexed by candidate points `x_j`, rows by moment-basis multi-indices
# `α_a`. One Legendre table fill per point, then a tensor-product evaluation per
# (point, basis-index) pair. The matrix is `nbasis × npoints`.
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

# One NNLS solve against the design matrix above, followed by the truncation of
# the near-zero weights: returns `(kept, weights, residual)` — the indices into
# `points` that survive the truncation, the weights at those points, and the L²
# moment residual *of that truncated rule*.
#
# `cut_volume` is the measure of `Ω ∩ region_box`, the scale the weight cutoff is
# taken relative to (see `_NNLS_WEIGHT_TOL`). It is deliberately not
# `volume(region_box)`: on a sliver cut the bounding box and the cut region differ
# by orders of magnitude, and only the latter tracks the weights being tested.
#
# The residual is re-measured after the truncation because what `nnls` reports
# describes the untruncated weight vector, which is not the rule handed back — a
# rule truncated to nothing kept a machine-zero residual and read as a perfect
# fit. Zeroing the dropped entries in place, rather than slicing the matrix down
# to `kept`, keeps that re-measurement bit-identical to the NNLS residual in the
# ordinary case, where the truncation drops only the exact zeros of the
# Lawson–Hanson inactive set.
#
# "In place" is literal: the zeroing happens in the NNLS solution vector itself.
# `nnls` returns a vector nothing else holds a reference to, and the surviving
# entries are copied out before the loop runs, so no second full-length array is
# needed — which matters, because the candidate cloud is thousands of points
# wide while the rule handed back is `nbasis` wide.
function _solve_moment_fit(moments::Vector{T}, points::Vector{SVector{D,T}},
                           region_box::AxisBox{D,T}, moment_order::NTuple{D,Int},
                           cut_volume::T) where {D,T}
    A = _build_moment_matrix(points, region_box, moment_order)
    weights, _ = nnls(A, moments)
    cutoff = _NNLS_WEIGHT_TOL * cut_volume
    kept = findall(>(cutoff), weights)
    kept_weights = weights[kept]
    for i in eachindex(weights)
        weights[i] > cutoff || (weights[i] = zero(T))
    end
    residual = A * weights
    residual .-= moments
    return kept, kept_weights, norm(residual)
end

# ── 4. Exact moments and candidates from the implicit kernel ──────────────────

# Tensor Legendre moments mₐ = ∫_{Ω∩R} ψₐ dx recovered from an implicit volume
# rule by direct summation; reuses the per-axis Legendre tables of section 2.
function _moments_from_rule(points::Vector{SVector{D,T}}, weights::Vector{T},
                            region_box::AxisBox{D,T}, moment_order::NTuple{D,Int}) where {D,T}
    indices = _moment_basis_indices(moment_order)
    moments = zeros(T, length(indices))
    factors = _legendre_factor_buffers(moment_order, T)
    @inbounds for q in eachindex(weights)
        _fill_legendre_factors!(factors, points[q], region_box.lower, region_box.upper,
                                moment_order)
        wq = weights[q]
        for (a, idx) in pairs(indices)
            moments[a] += wq * _tensor_legendre_value(factors, idx)
        end
    end
    return moments
end

# Per-fiber Gauss count for the implicit moment rule, the maximum of two needs:
#
#   1. Exactness on a *tilted* linear cut. The inside-interval endpoints are then
#      linear in the base coordinates, so each of the D fiber integrations raises
#      the accumulated polynomial degree by one; the worst case (a corner beveled
#      by a linear leaf, e.g. the simplex Σx ≤ 1) reaches total degree
#      `sum(moment_order) + D` at the innermost fiber, needing
#      `q ≥ (sum(moment_order) + D) / 2`. An axis-aligned cut needs only
#      `⌈(maxorder+1)/2⌉`, but the default must serve the worst case — a constant
#      offset silently undershoots tilted high-order 3D cuts by ~1e-5. (Verified
#      against tilted simplices in 2D–4D as the smallest exact q.)
#   2. Accuracy on a *curved* cut. A curved boundary is never polynomial, so the
#      moments are approximated, not exact; a generous floor of `maxorder + 2`
#      keeps that high-order approximation sharp (it also dominates (1) in 2D).
function _implicit_gauss_points(moment_order::NTuple{D,Int}) where {D}
    max(cld(sum(moment_order) + D, 2), maximum(moment_order) + 2)
end

# Ceiling on the NNLS design matrix, in entries (`nbasis × ncandidates`).
# Lawson–Hanson sweeps the whole matrix once per accepted column, so both a
# solve's memory and its time are proportional to this product; 6·10⁶ entries is
# 48 MB and a fraction of a second at the moment orders in use here. Because the
# candidate budget below is a multiple of `nbasis`, the entry count grows as
# `budget · nbasis²` — quadratically in the basis size — so the retry needs an
# absolute ceiling and not just a relative one. It binds retries only (the `max`
# in `_cap_candidates` keeps the first attempt exempt), and only bites at all
# once `nbasis` passes ≈ 1000.
const _MAX_FIT_MATRIX_ENTRIES = 6_000_000

# Cap the candidate cloud so the NNLS design matrix stays bounded regardless of
# the volume rule's point count: at most `budget · nbasis` candidates, and never
# more than `_MAX_FIT_MATRIX_ENTRIES` matrix entries. A uniform stride over the
# (base-then-fiber ordered) rule keeps the survivors well spread through Ω ∩ R.
#
# `budget` comes from `_candidate_budget(attempt)` and grows with the retry, so
# each attempt draws a genuinely denser cloud. The `max` keeps the first
# attempt's `6 · nbasis` budget exempt from the ceiling, so the entry point of
# the pipeline behaves identically at every moment order.
function _cap_candidates(points::Vector{SVector{D,T}}, moment_order::NTuple{D,Int},
                         budget::Int) where {D,T}
    nbasis = _moment_basis_count(moment_order)
    cap = min(budget * nbasis, max(6 * nbasis, cld(_MAX_FIT_MATRIX_ENTRIES, nbasis)))
    length(points) <= cap && return points
    return points[1:cld(length(points), cap):end]
end

# ── 5. Public moment-fit rule ─────────────────────────────────────────────────

# Number of attempts. With exact moments the retry exists only for NNLS
# conditioning (a denser candidate cloud), never for moment accuracy.
const _MAX_IMPLICIT_ATTEMPTS = 3

# Candidate-cloud budget of `attempt`, as a multiple of the moment-basis size
# `nbasis`: 6·nbasis on the first attempt, eight times that on each retry.
#
# The budget has to grow for the retry to mean anything. Lawson–Hanson is greedy
# and terminates as soon as its passive set holds `nbasis` columns (`nsetp ≥ m`
# in the classical algorithm), so the fit it returns is decided entirely by which
# candidates the walk was offered: when the columns it picked are only marginally
# independent, the closing triangular solve is ill-conditioned and the residual
# lands near 1e-2 instead of 1e-16. A cell carrying many distinct boundary pieces
# — a coarse mesh cell spanning a dozen holes — is where that happens. Holding the
# budget at 6·nbasis made all three attempts draw a cloud of the same size, so
# they failed together and the "denser candidate cloud" the retry promises never
# materialised. Measured on one such cell (nbasis = 125, 25216 rule points, the
# moments held fixed): 742 candidates → residual 4.7e-2, 1484 → 2.8e-2,
# 2802 → 3.0e-2, but 5044 → 3e-16 and 8406 → 6e-16.
#
# The growth factor is large on purpose. `_cap_candidates` selects by a uniform
# stride, and the volume rule emits its points fiber by fiber in blocks of
# `gauss_points`, so a stride sharing a factor with that block length reaches
# only `gauss_points / gcd` of the per-fiber node positions and draws a
# systematically degenerate cloud. On the cell above (`gauss_points = 8`) the
# residual at ≈ 750 candidates runs 5e-1 at gcd 8, 2.7e-1 at gcd 4, and 1e-2–5e-2
# at gcd 2 or 1. A modest bump therefore only buys another unreliable draw; a ×8
# step moves each retry to a plainly different scale, where the stride is small
# enough that the aliasing no longer decides the outcome.
#
# The first attempt's budget is deliberately unchanged, so every cell that
# already fits keeps its rule unchanged and pays nothing extra; only cells that
# fail reach the larger solves.
_candidate_budget(attempt::Int) = 6 * 8^(attempt - 1)

"""
    moment_fit_rule(physical, region_box, moment_order; target_residual=1e-10)
        -> (points, weights, residual, status)

Build a non-negative tensor-Legendre moment-fitted quadrature rule on
`region_box ∩ Ω`. The rule reproduces the tensor Legendre moments

    m_α = ∫_{region_box ∩ Ω} ψ_α(ξ(x)) dx,    α ∈ {0,…,p₁} × … × {0,…,p_D}

to within the returned residual.

The moments and candidate points come from the Saye implicit-quadrature kernel
([`implicit_volume_quadrature`](@ref)) applied to `physical`'s CSG level set:
the moments are exact (machine precision for linear leaves and polytope
corners) and octree-depth-independent, and the volume-rule points serve as the
candidate cloud. Each leaf's Lipschitz constant is forwarded to the kernel, so a
geometric feature smaller than `region_box` is certified rather than sampled
for — leaves left at the `leaf` default `Inf` keep the kernel's point-sampling
heuristic and its sub-cell blind spot.

A single Lawson–Hanson NNLS solve then picks the non-negative
weights — its active-set solution carries at most `nbasis` non-zeros, so the
rule is already compressed to O(nbasis) points, and because the volume rule
reproduces the moments with strictly positive weights a zero-residual
non-negative solution exists, so one solve reaches machine-level residual.

`target_residual` bounds a small conditioning retry — up to
`_MAX_IMPLICIT_ATTEMPTS` attempts, each with a denser candidate cloud, never with
more octree depth. Both retry knobs act on the cloud alone: the fiber Gauss order
rises by one per attempt, emitting more volume-rule points, and the candidate
budget of `_candidate_budget` grows by a factor of eight per attempt, letting
more of them past the cap. The best (lowest-residual) fit observed is the one
returned. In the assembly pipeline this function is called with
`physical.target_residual` (the `physical_domain` default is `1e-6`), which
overrides the `1e-10` default here.

`status` reports which rule came back, and the integration-region dispatcher in
`intersections.jl` turns it into the region's quadrature kind:

  - `:fitted` — the moment fit met `_FIT_FAILURE_RESIDUAL`; `points`/`weights`
    are the compressed O(nbasis) fitted rule. Region kind `:cut_fitted`.
  - `:fallback` — no attempt met `_FIT_FAILURE_RESIDUAL`, so the returned rule is
    the highest-order attempt's *raw Saye volume rule* — the very data the
    moments were computed from, uncompressed. Region kind `:cut_fallback`.
  - `:empty` — `Ω ∩ region_box` carries no volume rule at all, or every fitted
    weight falls below the cut-volume-relative truncation cutoff so the
    compressed rule would hold no points; `points` and `weights` are empty and
    `residual` is zero. Region kind `:cut_failed` (`:cut_alpha_failed` under
    α-FCM).

The returned `residual` measures the rule that is returned: it is taken after the
near-zero weights have been truncated away, so it never describes a denser rule
than the caller receives.

The fallback loses no accuracy: on a cell where the fit fails, the volume rule is
by construction at least as accurate as a successful fit would have been, and its
weights are non-negative for the same reason the fit's are — each is a product of
a positive Gauss weight, a positive fiber half-length and a positive base weight
(see `_emit_fiber!` in `src/implicit.jl`). What it costs is points: the failing
cells measured on a 96-hole plate carry 7·10³–2.5·10⁴ volume-rule points where a
successful fit yields ≈ nbasis ≈ 125, a 50–200× per-cell blow-up in assembly
work. A fired fallback is therefore a signal that the cell is under-resolved for
its geometric complexity and is the condition that should drive refinement — it
is a safety net against the silently wrong answer an empty rule would give, not a
substitute for an adequate mesh. `AssemblyDiagnostics.cut_fallback_count` and
`cut_fallback_points` report how often it fired and what it cost.
"""
function moment_fit_rule(physical::PhysicalDomain, region_box::AxisBox{D,T},
                         moment_order::NTuple{D,Int}; target_residual::Real=1.0e-10) where {D,T}
    target = T(target_residual)
    q0 = _implicit_gauss_points(moment_order)
    max_subdiv = _effective_subcell_depth(physical, region_box)
    # Hand the kernel each leaf's Lipschitz constant, not just its callback: the
    # region box is a whole mesh cell, which may be much coarser than the
    # geometry, and only the certificate keeps a sub-cell feature (a hole well
    # inside the cell) from being pruned away and the cell integrated as solid.
    # Leaves built with the `leaf` default `Inf` fall back to point sampling.
    leaves = _leaves(physical.geometry)
    leaf_fs = Any[l.f for l in leaves]
    leaf_ls = Float64[l.lipschitz for l in leaves]
    membership = x -> _inside(physical.geometry, x)

    best_pts = SVector{D,T}[]
    best_ws = T[]
    best_res = T(Inf)
    # The raw volume rule of the attempt last run. It has to outlive the loop
    # because it is the fallback below, and keeping the *last* attempt's rule
    # keeps the highest fiber Gauss order — the most accurate of the three.
    saye_pts = SVector{D,T}[]
    saye_ws = T[]
    for attempt in 1:_MAX_IMPLICIT_ATTEMPTS
        # Exact moments at q0; denser candidate clouds on retry improve NNLS
        # conditioning without changing the (already exact) moments. Density
        # comes from both knobs at once — a higher fiber Gauss order emits more
        # rule points, a larger budget lets more of them past `_cap_candidates`.
        q = q0 + (attempt - 1)
        saye_pts, saye_ws = implicit_volume_quadrature(leaf_fs, membership, region_box;
                                                       gauss_points=q, max_subdiv=max_subdiv,
                                                       lipschitz=leaf_ls)
        isempty(saye_pts) && return SVector{D,T}[], T[], zero(T), :empty
        moments = _moments_from_rule(saye_pts, saye_ws, region_box, moment_order)
        candidates = _cap_candidates(saye_pts, moment_order, _candidate_budget(attempt))
        kept, kept_ws, residual = _solve_moment_fit(moments, candidates, region_box,
                                                    moment_order, sum(saye_ws))
        # A fit whose every weight falls under the truncation cutoff is no rule at
        # all: hand back the same `:empty` an empty kernel rule returns, so the
        # caller drops the region (or keeps its α-stabilised tensor part) instead
        # of consuming a zero-point rule as a successful fit. A denser candidate
        # cloud cannot resurrect a fit that collapsed to zero, so there is nothing
        # for the retry to do here.
        isempty(kept) && return SVector{D,T}[], T[], zero(T), :empty
        if residual < best_res
            best_pts, best_ws, best_res = candidates[kept], kept_ws, residual
        end
        best_res <= target && return best_pts, best_ws, best_res, :fitted
    end
    # Short of `target` but inside the failure threshold, the compressed fit is
    # still the rule to use — that band is ordinary approximation error on a
    # curved cut, not a broken fit.
    best_res <= _FIT_FAILURE_RESIDUAL && return best_pts, best_ws, best_res, :fitted
    # Beyond it the fit is not usable at any price. Hand back the volume rule the
    # moments themselves were summed from: correct and non-negative by
    # construction, merely uncompressed. Returning an empty rule here instead —
    # the pre-fallback behaviour — dropped the cell's entire contribution.
    return saye_pts, saye_ws, best_res, :fallback
end
