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
#   BSD-4-Clause). The Lawson–Hanson NNLS solver wrapped by `nnls!` is the
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
#   octree stair-step moment integrator and QuESo's point-elimination /
#   `AssembleIPs` retry are not used, because the exact kernel removes the
#   stair-step accuracy floor those loops worked around. Non-graph-like cells
#   (high curvature relative to the cell) are handled by the kernel's bounded
#   subdivision, so no separate fallback path is needed.
#
# Pipeline summary
#
#   1. NNLS wrapper around `NonNegLeastSquares.nonneg_lsq` (Lawson–Hanson).
#   2. Saye implicit volume rule (`implicit_volume_quadrature`) over Ω ∩ R.
#   3. Exact moments by `mₐ = Σ_q w_q ψₐ(x_q)`; the rule's points are the
#      moment-fit candidates (capped at O(nbasis)).
#   4. A single NNLS moment fit (Lawson–Hanson naturally yields ≤ nbasis
#      non-negative weights; a zero-residual non-negative solution exists, so
#      one solve reaches machine-level residual).
#
#   Public entry point: `moment_fit_rule(physical, region_box, moment_order)`.

# ── 1. NNLS wrapper ───────────────────────────────────────────────────────────

"""
    nnls!(A, b) -> (x, residual_norm)

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

Allocations are owned by `NonNegLeastSquares.jl`; `A` and `b` are not mutated
despite the `!` — the bang is kept for a future in-place port.
"""
function nnls!(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:Real}
    x = vec(nonneg_lsq(A, b; alg=:nnls))
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

# Drop a candidate weight when its NNLS solution falls below this absolute
# threshold (numerical noise around true zeros).
const _NNLS_WEIGHT_TOL = 1.0e-12

# Catastrophic failure threshold for the moment-fit residual. A residual above
# this value indicates the NNLS could not come anywhere near reproducing the
# moments; the integration-plan dispatcher tags such a region `:cut_failed`.
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

# One NNLS solve against the design matrix above: returns `(weights, residual)`
# for the candidate `points`.
function _solve_moment_fit(moments::Vector{T}, points::Vector{SVector{D,T}},
                           region_box::AxisBox{D,T}, moment_order::NTuple{D,Int}) where {D,T}
    A = _build_moment_matrix(points, region_box, moment_order)
    return nnls!(A, moments)
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
        _fill_legendre_factors!(factors, points[q], region_box.lower, region_box.upper, moment_order)
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
_implicit_gauss_points(moment_order::NTuple{D,Int}) where {D} =
    max(cld(sum(moment_order) + D, 2), maximum(moment_order) + 2)

# Cap the candidate cloud at a small multiple of the moment-basis size so the
# NNLS design matrix stays O(nbasis) wide regardless of the volume rule's point
# count. A uniform stride over the (base-then-fiber ordered) rule keeps the
# survivors well spread through Ω ∩ R.
function _cap_candidates(points::Vector{SVector{D,T}}, moment_order::NTuple{D,Int}) where {D,T}
    cap = 6 * _moment_basis_count(moment_order)
    length(points) <= cap && return points
    return points[1:cld(length(points), cap):end]
end

# ── 5. Public moment-fit rule ─────────────────────────────────────────────────

# Number of attempts. With exact moments the retry exists only for NNLS
# conditioning (a denser candidate cloud), never for moment accuracy.
const _MAX_IMPLICIT_ATTEMPTS = 3

"""
    moment_fit_rule(physical, region_box, moment_order; target_residual=1e-10)
        -> (points, weights, residual)

Build a non-negative tensor-Legendre moment-fitted quadrature rule on
`region_box ∩ Ω`. The rule reproduces the tensor Legendre moments

    m_α = ∫_{region_box ∩ Ω} ψ_α(ξ(x)) dx,    α ∈ {0,…,p₁} × … × {0,…,p_D}

to within the returned residual.

The moments and candidate points come from the Saye implicit-quadrature kernel
([`implicit_volume_quadrature`](@ref)) applied to `physical`'s CSG level set:
the moments are exact (machine precision for linear leaves and polytope
corners) and octree-depth-independent, and the volume-rule points serve as the
candidate cloud. A single Lawson–Hanson NNLS solve then picks the non-negative
weights — its active-set solution carries at most `nbasis` non-zeros, so the
rule is already compressed to O(nbasis) points, and because the volume rule
reproduces the moments with strictly positive weights a zero-residual
non-negative solution exists, so one solve reaches machine-level residual.

`target_residual` bounds a small conditioning retry (a denser candidate cloud,
not more octree depth). Returns the best (lowest-residual) fit observed; an
empty region returns empty point/weight vectors. In the assembly pipeline this
function is called with `physical.target_residual` (the `physical_domain`
default is `1e-6`), which overrides the `1e-10` default here.
"""
function moment_fit_rule(physical::PhysicalDomain, region_box::AxisBox{D,T},
                         moment_order::NTuple{D,Int}; target_residual::Real=1.0e-10) where {D,T}
    target = T(target_residual)
    q0 = _implicit_gauss_points(moment_order)
    max_subdiv = _effective_subcell_depth(physical, region_box)
    leaf_fs = Any[l.f for l in _leaves(physical.geometry)]
    membership = x -> _inside(physical.geometry, x)

    best_pts = SVector{D,T}[]
    best_ws = T[]
    best_res = T(Inf)
    for attempt in 1:_MAX_IMPLICIT_ATTEMPTS
        # Exact moments at q0; denser candidate clouds on retry improve NNLS
        # conditioning without changing the (already exact) moments.
        q = q0 + (attempt - 1)
        pts_vol, w_vol = implicit_volume_quadrature(leaf_fs, membership, region_box; gauss_points=q,
                                                    max_subdiv=max_subdiv)
        isempty(pts_vol) && return SVector{D,T}[], T[], zero(T)
        moments = _moments_from_rule(pts_vol, w_vol, region_box, moment_order)
        candidates = _cap_candidates(pts_vol, moment_order)
        weights, residual = _solve_moment_fit(moments, candidates, region_box, moment_order)
        kept = findall(>(_NNLS_WEIGHT_TOL), weights)
        if residual < best_res
            best_pts, best_ws, best_res = candidates[kept], weights[kept], residual
        end
        best_res <= target && return best_pts, best_ws, best_res
    end
    return best_pts, best_ws, best_res
end
