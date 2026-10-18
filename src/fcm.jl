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
#   2. The compression replaces that rule by one with at most as many points as the basis has functions, reproducing
#      its moments over the tensor Legendre basis on R (`MomentBasis`),
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
#   the refusal of negative weights, the residual of the rule handed back, and the fallback to the cut rule itself when
#   a compression misses.
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
#   Public entry points: `moment_fit_rule(physical, region_box, moment_order)`, `implicit_rule`, `MomentBasis`,
#   `Caratheodory`, `NonnegativeFit`.

# ── 1. The moment basis ───────────────────────────────────────────────────────

"""
    MomentBasis(box, order)

The tensor Legendre basis on `box` up to `order` per axis, which a cut cell's rule is compressed against:

    ψ_α(x) = ∏_d L_{α_d}(ξ_d(x)),    α ∈ {0,…,p₁} × … × {0,…,p_D},

with `L_n` the Legendre polynomial of degree `n` and `ξ(x) ∈ [−1, 1]ᴰ` the box's reference coordinate. A compression is
handed the basis of the cell it compresses (see [`physical_domain`](@ref)) and evaluates it through two calls:

  - `basis(points, weights)` is the moment vector `m_α = Σ_q w_q ψ_α(x_q)` of a rule;
  - `basis(points)` is the matrix `A[α, q] = ψ_α(x_q)`, `length(basis) × length(points)`, so that
    `basis(points) * weights` is `basis(points, weights)` up to round-off.

`length(basis)` is the number of functions, `∏(order .+ 1)`, and `basis.box` and `basis.order` are the cell and the
order. The first moment, of the constant `ψ_0 = 1`, is the measure a rule integrates. Which of the other entries belongs
to which `α`, and how `L_n` is normalised, are not part of the contract: a compression compares moment vectors this
object computes and never needs to identify an entry.

The object is immutable and allocates its scratch per call, so one basis may be evaluated on several threads at once.
"""
struct MomentBasis{D,T}
    box::AxisBox{D,T}
    order::NTuple{D,Int}
    indices::Vector{CartesianIndex{D}}
end

function MomentBasis(box::AxisBox{D,T}, order::NTuple{D,Int}) where {D,T}
    return MomentBasis(box, order, vec(collect(CartesianIndices(ntuple(d -> 0:order[d], D)))))
end

Base.length(basis::MomentBasis) = length(basis.indices)

function (basis::MomentBasis{D,T})(points::AbstractVector{SVector{D,T}}) where {D,T}
    A = Matrix{T}(undef, length(basis), length(points))
    factors = _legendre_factor_buffers(basis.order, T)
    @inbounds for (j, x) in pairs(points)
        _fill_legendre_factors!(factors, x, basis.box.lower, basis.box.upper, basis.order)
        for (a, α) in pairs(basis.indices)
            A[a, j] = _tensor_legendre_value(factors, α)
        end
    end
    return A
end

function (basis::MomentBasis{D,T})(points::AbstractVector{SVector{D,T}}, weights::AbstractVector{T}) where {D,T}
    return _add_moments!(zeros(T, length(basis)), basis, points, weights, _legendre_factor_buffers(basis.order, T))
end

# Add the moments of the rule `(points, weights)` to `m`, with the caller's factor buffers: the kernel behind
# `basis(points, weights)`, and the one `Caratheodory` runs once per cluster without allocating. `A w` without forming
# `A`, so a rule of any size costs `length(basis)` storage.
function _add_moments!(m::AbstractVector{T}, basis::MomentBasis{D,T}, points, weights, factors) where {D,T}
    @inbounds for q in eachindex(points, weights)
        _fill_legendre_factors!(factors, points[q], basis.box.lower, basis.box.upper, basis.order)
        wq = weights[q]
        for (a, α) in pairs(basis.indices)
            m[a] += wq * _tensor_legendre_value(factors, α)
        end
    end
    return m
end

# Fill caller-owned per-axis 1D Legendre value tables for the point `x` mapped into the box's reference frame
# `[−1, 1]ᴰ`:
#
#     factors[d][p + 1] = L_p(ξ_d),    ξ_d = (2 x[d] − lower[d] − upper[d]) / (upper[d] − lower[d]),
#
# via one Legendre three-term recurrence per axis, so the per-point cost is `D · (order + 1)` instead of
# `D · (order + 1)^D`.
function _fill_legendre_factors!(factors, x::SVector{D,T}, lower::SVector{D,T}, upper::SVector{D,T},
                                 order::NTuple{D,Int}) where {D,T}
    @inbounds for d in 1:D
        xi = (2 * x[d] - lower[d] - upper[d]) / (upper[d] - lower[d])
        p = order[d]
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

# Caller-owned per-axis 1D scratch buffers for `_fill_legendre_factors!`: one `Vector{T}` per axis, of length
# `order[d] + 1`.
_legendre_factor_buffers(order::NTuple{D,Int}, ::Type{T}) where {D,T} = ntuple(d -> Vector{T}(undef, order[d] + 1), D)

# Tensor-product basis value at the point whose per-axis Legendre factors are loaded into `factors`:
# `ψ_α(ξ) = ∏_d L_{α_d}(ξ_d) = ∏_d factors[d][α_d + 1]`.
function _tensor_legendre_value(factors, α::CartesianIndex{D}) where {D}
    v = factors[1][α.I[1] + 1]
    @inbounds for d in 2:D
        v *= factors[d][α.I[d] + 1]
    end
    return v
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
drawn from the cut rule so that they reproduce its moments. It returns every candidate with its weight, and the active
set — the at most `length(basis)` candidates of positive weight — is the rule [`moment_fit_rule`](@ref) keeps.

The candidates are `6 · length(basis)` points of the cut rule, evenly spread over it (`_cap_candidates`). The cut rule
itself is an exact non-negative solution over all of its points, but a subset admits one only if the moments lie in the
cone the subset spans (`_candidate_budget`), so a fit whose residual exceeds `target_residual · ‖m‖` is retried on a
cloud eight times denser, at most twice. The best attempt is returned; the cut rule and its moments are the same for all
of them, and a cloud that already holds the whole rule is not drawn again. `target_residual` decides only whether to
retry; whether the rule is used is decided by [`moment_fit_rule`](@ref) against its failure threshold.

Choose it over the default where its solve on a bounded cloud pays: on a cut rule of 10⁵ points or more at a low moment
order it is the faster of the two (0.04 s against 0.10 s at moment order (4, 4, 4) on 3.4·10⁵ points), and it reproduces
the moments to 1e-15 rather than 1e-13. Wherever else a cell takes more than 10 ms, `Caratheodory` is the faster — 3×
at moment order (8, 8, 8) — returns fewer points and needs no target.
"""
struct NonnegativeFit
    target_residual::Float64
    function NonnegativeFit(; target_residual::Real=1.0e-6)
        target_residual > 0 || throw(ArgumentError("target_residual must be positive; got $target_residual"))
        return new(target_residual)
    end
end

function (fit::NonnegativeFit)(points::Vector{SVector{D,T}}, weights::Vector{T}, moments::Vector{T},
                               basis::MomentBasis{D,T}) where {D,T}
    target = fit.target_residual * norm(moments)
    best = nothing
    for attempt in 1:_MAX_FIT_ATTEMPTS
        candidates = _cap_candidates(points, length(basis), _candidate_budget(attempt))
        x, residual = nnls(basis(candidates), moments)
        (best === nothing || residual < best[3]) && (best = (candidates, x, residual))
        (residual <= target || length(candidates) == length(points)) && break
    end
    return best[1], best[2]
end

# Number of NNLS solves `NonnegativeFit` may spend on one cell.
const _MAX_FIT_ATTEMPTS = 3

# Ceiling on the NNLS design matrix, in entries (`nbasis × ncandidates`): 6·10⁶ entries, 48 MB. Lawson–Hanson sweeps the
# whole matrix once per accepted column, so a solve's memory grows with this product and its time with the product
# times `nbasis`; measured serially at `-O2` on the space-time cavity geometry, one fit took 0.2 ms at `nbasis = 27` and
# ≈ 2.4 s at 729 (a 729 × 4374 matrix, past per-core cache). The budget is a multiple of `nbasis`, so the entry count
# grows quadratically in it and the retry needs an absolute ceiling. It binds retries only — `_cap_candidates` keeps the
# first attempt exempt — and only once `nbasis` exceeds ≈ 354 on the second attempt and ≈ 126 on the third.
const _MAX_FIT_MATRIX_ENTRIES = 6_000_000

# The candidate cloud of an attempt: at most `budget · nbasis` points of the cut rule and at most
# `_MAX_FIT_MATRIX_ENTRIES` matrix entries, the first attempt's `6 · nbasis` exempt from the latter.
#
# The candidates are spread by fractional spacing, `(n − 1) / (cap − 1)`, not by an integer stride. A stride spent only
# `cld(n, stride)` of the budget (2744 of 4374 candidates on a 5488-point rule), and it aliased against the blocks of
# `gauss_points` the kernel emits per fiber: on the space-time cone fixture at cells (8, 8, 8), 57 of 146 cut cells then
# fitted only to above 1e-3, where the same number of candidates spread fractionally reaches ~2e-15. Fractional spacing
# spends the whole budget and sweeps the per-fiber positions instead of locking onto some of them, deterministically.
function _cap_candidates(points::Vector{SVector{D,T}}, nbasis::Int, budget::Int) where {D,T}
    cap = min(budget * nbasis, max(6 * nbasis, cld(_MAX_FIT_MATRIX_ENTRIES, nbasis)))
    n = length(points)
    n <= cap && return points
    # `n > cap`, so the spacing exceeds 1 and the rounded indices are strictly increasing: no candidate repeats.
    return points[round.(Int, range(1, n; length=cap))]
end

# Candidate budget of `attempt`, as a multiple of `nbasis`: 6 on the first attempt, eight times more on each retry.
#
# What a larger cloud buys is cone membership, not conditioning: an exact non-negative fit exists over the candidates
# only if the moments lie in the cone of their columns, which the whole cut rule's does by construction and a subset's
# need not, whatever the solver. Measured at moment order (8, 8, 8) on the space-time cone geometry with only the
# candidate count varied, the relative residual runs 1.9e-2 at 1.5·nbasis candidates, 1.9e-3 at 3·nbasis and 3.1e-15 at
# 6·nbasis — so the first attempt's 6·nbasis is load-bearing — and on a cell with many boundary pieces a cloud of 2802
# still missed by 3.0e-2 where one of 5044 fitted to 3e-16. The ×8 step makes each retry's cone materially wider rather
# than a redraw of much the same extent.
_candidate_budget(attempt::Int) = 6 * 8^(attempt - 1)

# ── 4. Carathéodory compression ───────────────────────────────────────────────

"""
    Caratheodory()

A compression for [`physical_domain`](@ref), and its default: Carathéodory pruning of the cut rule. The cut rule is a
non-negative combination of point evaluations whose moments are `m = A w`, with `A = basis(points)`. By Carathéodory's
theorem a combination of more points than `rank A` can be reduced to one of at most `rank A ≤ length(basis)` of them
with the same moments and non-negative weights, and the reduction is constructive: a null vector of `A` on the current
points moves the weights without moving the moments, so walking along one until the first weight reaches zero removes a
point. The rule returned is a subset of the cut rule's points with non-negative weights, reproducing its moments to
round-off, with no target, retry or tolerance for a caller to tune.

The null space of `A` on all of a cut rule's points would take a factorisation of an `npoints × npoints` matrix, so the
cut rule is first reduced by recombination (Litterer & Lyons): its points are grouped into `2 · length(basis)` clusters,
contiguous runs of a rule the kernel emits fiber by fiber and so spatially coherent; the clusters' moment vectors are
pruned instead — at most `length(basis)` survive — and each surviving cluster's weights are scaled by its coefficient.
That halves the points at the cost of one moment sum over them and a few factorisations of a `length(basis) ×
2 length(basis)` matrix, and repeats until at most `2 · length(basis)` points remain, which are pruned directly. No
matrix it forms is larger than `2 length(basis)` square, however many points the cut rule has, and none of its work
calls BLAS, so it runs on every thread of a plan build without competing with BLAS threads for the cores.

Measured against [`NonnegativeFit`](@ref) on the same cut rules — 34 cells in 2D and 3D at moment orders 2–12: disc,
sliver, polytope-corner and perforated cells — Carathéodory took 3.4 s altogether against 7.8 s, and 1.7 s against
5.2 s at moment order (8, 8, 8) where both cost most. It returned 3% fewer points, fewer wherever the cell's moment
matrix is numerically rank-deficient: 594 against 660 of the 729 a ball's cut cell admits. Its moments matched to
3e-13 at worst, and no pruning pass of the 182 those cells took drifted far enough to be undone.

  - C. Carathéodory, "Über den Variabilitätsbereich der Fourier'schen Konstanten von positiven harmonischen
    Funktionen", Rend. Circ. Mat. Palermo 32 (1911) 193–217, doi:10.1007/BF03014795.
  - C. Litterer, T. Lyons, "High order recombination and an application to cubature on Wiener space", Ann. Appl.
    Probab. 22 (2012) 1301–1327, doi:10.1214/11-AAP786.
  - F. Bělík, J. Chan, A. Narayan, "Efficient and robust Carathéodory–Steinitz pruning of positive discrete measures",
    arXiv:2510.14916 (2025), for the same reduction applied to cut-cell quadrature.
"""
struct Caratheodory end

function (::Caratheodory)(points::Vector{SVector{D,T}}, weights::Vector{T}, moments::Vector{T},
                          basis::MomentBasis{D,T}) where {D,T}
    n = length(basis)
    w = copy(weights)
    support = findall(>(0), w)
    factors = _legendre_factor_buffers(basis.order, T)
    # Recombination: prune the moment vectors of 2n clusters, and scale each cluster's weights by what survives.
    while length(support) > 2n
        clusters = Iterators.partition(support, cld(length(support), 2n))
        S = zeros(T, n, length(clusters))
        for (j, c) in enumerate(clusters)
            _add_moments!(view(S, :, j), basis, view(points, c), view(w, c), factors)
        end
        for (c, λ) in zip(clusters, _prune!(ones(T, size(S, 2)), S))
            @views w[c] .*= λ
        end
        filter!(i -> w[i] > 0, support)
    end
    # The at most 2n points left, pruned directly.
    return points[support], _prune!(w[support], basis(view(points, support)))
end

# Carathéodory's theorem, constructively: reduce the non-negative coefficients `λ` of the combination `S λ` of the
# columns of `S`, in place, to at most `rank S` positive ones, leaving `S λ` unchanged to round-off.
#
# Each pass factors `S` on the current support for a basis of its null space (`_nullspace`) and walks the coefficients
# along it, checked (`_prune_pass!`). Every pass removes at least one column, so the pruning ends, at the latest when
# the support has no null space left.
#
# A coefficient below `eps · max λ` is round-off left by a walk, and is dropped before the support is factored: kept, it
# would be one more column for the null space to find at that scale.
function _prune!(λ::AbstractVector{T}, S::AbstractMatrix{T}) where {T}
    while true
        λ .= ifelse.(λ .> eps(T) * maximum(λ; init=zero(T)), λ, zero(T))
        support = findall(>(0), λ)
        isempty(support) && return λ
        A = S[:, support]
        V = _nullspace(A)
        size(V, 2) == 0 && return λ
        λ[support] .= _prune_pass!(λ[support], A, V)
    end
end

# One pass of `_prune!`: walk `μ` along every column of `V`, a basis of the null space of `A` (`_walk!`). The walks
# share one factorisation, which is what makes a pass cheap and also what can make it inexact: eliminating a removed
# coefficient from the vectors still to come compounds round-off, the more so at a small pivot. The pass is therefore
# checked: if it moved `A μ` by more than `_PRUNE_DRIFT` of its norm, it is undone and only the first column of `V` is
# walked — a null vector no elimination has touched, whose walk cannot drift.
function _prune_pass!(μ::AbstractVector{T}, A::AbstractMatrix{T}, V::AbstractMatrix{T}) where {T}
    μ₀ = copy(μ)
    before = _matvec(A, μ)
    _walk!(μ, V)
    norm(_matvec(A, μ) - before) <= _PRUNE_DRIFT * norm(before) && return μ
    μ .= μ₀
    return _walk!(μ, view(V, :, 1:1))
end

# The largest change a pruning pass may make to the combination it prunes, relative to its norm, before the pass is
# replaced by a single walk. Above the round-off of a sound pass — none of the 182 passes on the 34 cells measured at
# `Caratheodory` reached it, their rules matching to 3e-13 at worst — and below what an ill-conditioned matrix makes of
# one: over monomials of random points, 64 of them at 128 points crowded towards one end, single passes moved the
# combination by up to 3.3e-9, and the pruning with the guard ended within 1.0e-11 of where it began throughout.
const _PRUNE_DRIFT = 1.0e-11

# A basis of the null space of `A` (n × m): the columns of an m × (m − r) matrix, r the numerical rank of `A`.
#
# Householder QR with column pivoting, A P = Q [R₁₁ R₁₂; 0 R₂₂] (P. Businger, G. H. Golub, "Linear least squares
# solutions by Householder transformations", Numer. Math. 7 (1965) 269–276, doi:10.1007/BF01436084), stopped where the
# largest column left has norm at most min(n, m) · eps times the largest of `A` — the rank cut LinearAlgebra's
# `nullspace` makes on the singular values. Q is never formed: `A` and `[R₁₁ R₁₂]` share their null space, which the
# columns of P [−R₁₁⁻¹ R₁₂; I] span. The walks need null vectors, not orthonormal ones, so this basis serves them as an
# SVD's would, without the SVD's cost: to read off the last m − r columns of V, it forms all of V, m × m. The norms of
# the columns left are recomputed in the sweep that applies each reflector, rather than downdated, so no pivot is chosen
# on a cancelled norm.
#
# Written out rather than taken from LAPACK because `_prune!` runs on every thread of a plan build: LAPACK's `gesdd`
# calls threaded BLAS kernels on matrices this small, whose workers then compete with the plan's threads for the cores.
function _nullspace(A::AbstractMatrix{T}) where {T}
    n, m = size(A)
    R = Matrix{T}(A)
    perm = collect(1:m)
    colnorm = [sqrt(sum(abs2, view(R, :, l))) for l in 1:m]
    tol = min(n, m) * eps(T) * maximum(colnorm; init=zero(T))
    r = 0
    @inbounds for j in 1:min(n, m)
        c = j - 1 + argmax(view(colnorm, j:m))
        colnorm[c] <= tol && break
        if c != j
            for i in 1:n
                R[i, j], R[i, c] = R[i, c], R[i, j]
            end
            perm[j], perm[c] = perm[c], perm[j]
            colnorm[j], colnorm[c] = colnorm[c], colnorm[j]
        end
        # The reflector I − v vᵀ / (α v₁), v = x + α e₁ with α = sign(x₁) ‖x‖, takes x = R[j:n, j] to −α e₁; v is stored
        # in place of x while the columns to its right are reflected.
        α = copysign(colnorm[j], R[j, j])
        R[j, j] += α
        β = α * R[j, j]
        for l in (j+1):m
            s = zero(T)
            @simd for i in j:n
                s += R[i, j] * R[i, l]
            end
            s /= β
            R[j, l] -= s * R[j, j]
            left = zero(T)
            @simd for i in (j+1):n
                R[i, l] -= s * R[i, j]
                left += abs2(R[i, l])
            end
            colnorm[l] = sqrt(left)
        end
        R[j, j] = -α
        r = j
    end
    # Back-substitute R₁₁⁻¹ R₁₂ in place of R₁₂, a column of R₁₁ at a time, and scatter it through P.
    N = zeros(T, m, m - r)
    @inbounds for c in axes(N, 2)
        x = view(R, 1:r, r + c)
        for i in r:-1:1
            x[i] /= R[i, i]
            @simd for h in 1:(i-1)
                x[h] -= x[i] * R[h, i]
            end
        end
        N[perm[r+c], c] = one(T)
        for i in 1:r
            N[perm[i], c] = -x[i]
        end
    end
    return N
end

# `A x`, in pure Julia for the reason `_nullspace` gives: at the sizes `_prune!` meets, BLAS runs it threaded.
function _matvec(A::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    y = zeros(T, size(A, 1))
    @inbounds for j in axes(A, 2)
        xj = x[j]
        @simd for i in axes(A, 1)
            y[i] += A[i, j] * xj
        end
    end
    return y
end

# Walk the coefficients `μ` along the columns of `V`, a basis of the null space on `μ`'s support: along each, in the
# direction with the larger positive entry, until the first coefficient reaches zero — the ratio test over the live
# coefficients, so every other stays non-negative to round-off — then eliminate that coefficient's entry from the
# columns still to come, so they stay null vectors of the shrinking support and leave it at exactly zero. The walk ends
# early when a step takes a second coefficient to round-off along with the first: only one can be eliminated, and the
# rest would be moved again by the next vector.
function _walk!(μ::AbstractVector{T}, V::AbstractMatrix{T}) where {T}
    for j in axes(V, 2)
        v = view(V, :, j)
        maximum(v) < -minimum(v) && (v .= .-v)
        t, i = findmin(r -> μ[r] > 0 && v[r] > 0 ? μ[r] / v[r] : T(Inf), eachindex(μ))
        isfinite(t) || continue
        μ .-= t .* v
        μ[i] = zero(T)
        floor = eps(T) * maximum(μ)
        dead = (μ .!= 0) .& (μ .<= floor)
        μ[dead] .= zero(T)
        any(dead) && return μ
        for l in (j+1):size(V, 2)
            @views V[:, l] .-= (V[i, l] / v[i]) .* v
            V[i, l] = zero(T)
        end
    end
    return μ
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
volume, and its `compression` reduces that rule to one with far fewer points that reproduces its moments over
`MomentBasis(region_box, moment_order)` ([`MomentBasis`](@ref)),

    m_α = Σ_q w_q ψ_α(x_q) ≈ ∫_{region_box ∩ Ω} ψ_α(x) dx,

see [`physical_domain`](@ref) for both stages and their contracts. Points are in physical coordinates and weights are
physical, so `Σ w_q` is the cut volume.

Between the two stages this function owns what every compression shares. It sums the moments of the cut rule, refuses
a rule from either stage that carries a weight below zero beyond round-off (an `ArgumentError`), drops the compressed
weights within `_WEIGHT_TOL` of zero relative to the cut volume — round-off around the exact zeros a compression
leaves — and measures the L² moment residual `‖basis(points, weights) − m‖` of the rule it hands back, after that
truncation, so the residual never describes a denser rule than the caller receives.

`status` reports which rule came back, and the integration-region dispatcher in `intersections.jl` turns it into the
region's quadrature kind:

  - `:fitted` — the returned rule reproduces the cut rule's moments within the failure threshold
    `_FIT_FAILURE_RESIDUAL · ‖m‖` (1e-2, relative to the moment vector, so the verdict does not depend on the cell's
    size). Region kind `:cut_fitted`. "Fitted" means exactly this throughout — in this status, that kind and the
    diagnostics `fit_failure_count` and `moment_fit_residual_max` — whichever compression produced the rule; under
    `compression = nothing` the cut rule itself comes back, fitted to its own moments with a zero residual.
  - `:fallback` — the compressed rule missed the failure threshold, so the returned rule is the cut rule itself, the
    very data the moments were summed from, uncompressed. Region kind `:cut_fallback`.
  - `:empty` — the cut rule integrates no measure over `region_box`, or every compressed weight falls below the
    truncation cutoff so the rule would hold no points; `points` and `weights` are empty and `residual` is zero. Region
    kind `:cut_failed` (`:cut_alpha_failed` under α-FCM).

The fallback loses no accuracy: the cut rule is by construction at least as accurate as a successful compression of it,
and the package's cut rule has non-negative weights — each a product of a positive Gauss weight, a positive fiber
half-length and a positive base weight (see `_emit_fiber!` in `src/implicit.jl`). What it costs is points: 10³–10⁵ per
cell where a compressed rule carries at most `length(basis)`, a 50–200× per-cell blow-up in assembly work on the
many-feature cells where a fallback is plausible at all. `AssemblyDiagnostics.cut_fallback_count` and
`cut_fallback_points` report how often it fired and what it cost.
"""
function moment_fit_rule(physical::PhysicalDomain, region_box::AxisBox{D,T}, moment_order::NTuple{D,Int}) where {D,T}
    points, weights = physical.cut_rule(physical, region_box, moment_order)
    measure = sum(abs, weights; init=zero(T))
    _check_weights(weights, measure, "cut rule")
    measure > 0 || return SVector{D,T}[], T[], zero(T), :empty
    physical.compression === nothing && return points, weights, zero(T), :fitted
    basis = MomentBasis(region_box, moment_order)
    moments = basis(points, weights)
    fitted, fitted_weights = physical.compression(points, weights, moments, basis)
    _check_weights(fitted_weights, measure, "compression")
    # A compression that truncates to nothing has no rule to offer, and a denser cut rule would not change that: hand
    # back the same `:empty` a measureless cut rule gives, so the caller drops the region (or keeps its α-stabilised
    # tensor part) instead of consuming a zero-point rule as a successful fit.
    kept = findall(>(_WEIGHT_TOL * measure), fitted_weights)
    isempty(kept) && return SVector{D,T}[], T[], zero(T), :empty
    fitted, fitted_weights = fitted[kept], fitted_weights[kept]
    residual = norm(basis(fitted, fitted_weights) - moments)
    residual <= _FIT_FAILURE_RESIDUAL * norm(moments) && return fitted, fitted_weights, residual, :fitted
    return points, weights, residual, :fallback
end

# Refuse a stage's weights if one lies below zero by more than the round-off `_WEIGHT_TOL` allows. Every consumer of a
# cut-cell rule relies on w ≥ 0 — the α blend, the Dirichlet trace solve's test for unsupported dofs, any squared norm
# integrated over the region — and a signed rule breaks them without a trace. The package's own stages never produce
# one.
function _check_weights(weights, measure, stage::String)
    w = minimum(weights; init=zero(measure))
    w >= -_WEIGHT_TOL * measure ||
        throw(ArgumentError("the $stage returned the weight $w; a cut-cell rule must have non-negative weights " *
                            "(see `PhysicalDomain`)"))
    return nothing
end
