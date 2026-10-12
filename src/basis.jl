# ── Basis family interface ────────────────────────────────────────────────────

"""
    BasisFamily

Abstract supertype for every basis family supported by the package. A basis family answers, for each cell or span, the
questions assembly and the dof layer ask:

  - how many local basis functions live on a cell;
  - which multi-indices identify them and in what tensor-product order;
  - what their values and gradients are at a reference point, possibly pulled back from a physical cell;
  - which of them touch a given axis-side of the cell (used to pick the facet-incident modes for the Dirichlet boundary
    trace in `src/dirichlet.jl`; the dof layer's own "is this dof on a face" test is the separate `_key_on_level_side`
    hook, which works on dof keys rather than local mode indices);
  - what quadrature order integrates products of two basis functions exactly on a `:full` region.

The concrete implementation shipped here is [`IntegratedLegendre`](@ref); the `BasicBSpline` package extension adds a
second family. Adding a family means implementing this interface plus the corresponding dof/constraint behaviour —
geometry, intersections, assembly, projection, and solvers stay untouched.

A new family `F <: BasisFamily` must provide:

  - `basis_name(::F) -> Symbol` — the short identifier error messages and diagnostics print. There is no `::BasisFamily`
    default, so a family that omits it turns the dof layer's own "not implemented for basis family" message into an
    unrelated `MethodError`.
  - `_fill_factor_tables!(::F, val1d, der1d, order, ξ, cell)` — the per-axis 1D value tables at one reference point on
    one cell, and the derivative tables unless `der1d === nothing`. One method serves both: `der1d` is `nothing` by
    type on the values-only path, so the derivative work folds away at compile time. This is the *only*
    family-specific step on the hot path: tensor-product evaluation, assembly, projection, field evaluation, and the
    boundary trace all reach the family through it and through nothing else.
  - `is_boundary_basis(::F, id, axis, side)` — per-axis facet incidence.
  - the dof-key and overlay-constraint hooks the dof layer needs: `_tensor_dof_key`, `_key_on_level_side`, and either
    the per-raw `_has_overlay_constraint` predicate or a whole-level `_overlay_constraints!` method that records into
    the dof layer's `EliminationLedger` (see `src/dofs.jl`), plus `_resolve_rows!` if the family records
    linear-constraint rows: the dof layer stores rows but resolves none itself, and its default raises. The B-spline
    extension's cascade is the one shipped resolver. A key `_tensor_dof_key(level, …)` returns must carry `level.id`
    (`TensorDofKey(level.id, …)`): no key is then shared between two levels, which is what lets the dof layer hold each
    level's raw dofs as one contiguous range and run every per-level pass over that range alone.
  - `instantiate_basis(::F, mesh, order)` — only when the concrete family depends on the level's mesh. The
    `::BasisFamily` default returns the family unchanged; see [`instantiate_basis`](@ref) for the contract and why the
    hook exists.
  - `_supports_cell_order(::F)` — `false` by default, and a *loud* default: a non-uniform `order =` on a family that
    declines it raises rather than being silently flattened. Answer `true` only where a per-cell degree names a set of
    functions and the minimum rule has a shared entity to act on, and supply the matching `_cell_modes` method
    (`src/dofs.jl`) in the same change — `_index_admissible(::F, order, id)` below is the per-axis filter that rule is
    expressed over (it has no `::BasisFamily` default, so the family supplies that method too), and
    `test_cell_order.jl` asserts its monotonicity in the order, which is what makes the rule correct.

It inherits the basis-agnostic `::BasisFamily` defaults and overrides only the ones that do not fit:
`recommended_quadrature_order` (`order .+ 1`), `local_basis_indices` (the lexicographic tensor index set
`∏_d {0, …, order[d]}`), `_with_mode` (`:tensor` only), `is_facet_basis` and `boundary_basis_indices`;
`local_basis_count` is the size of `local_basis_indices` and has nothing to override. The point evaluators
[`basis_values`](@ref) / [`physical_basis_gradients`](@ref) take a [`Level`](@ref) and a cell rather than a family: they
read the cell's order and index set from the level and reach the family through `_fill_factor_tables!` alone, so they
serve every family with nothing to override.

**The family value is the whole index-set rule.** A basis mode (the `mode` keyword of [`space`](@ref),
[`overlay`](@ref) and [`ladder`](@ref)) is not a second argument carried beside the family: it is applied once, when a
level is built, by `_with_mode(family, mode)`, which returns the family value that serves that index set or refuses a
mode the family does not carry. Every index-set question — `local_basis_indices(F, order)`, `local_basis_count`,
`boundary_basis_indices`, `_index_admissible` — then takes the family alone, and `_mode(F)` reads the mode back for
display and diagnostics. A family that offers a mode beyond `:tensor` overrides `_with_mode`, `_mode` and
`local_basis_indices`, as integrated Legendre does for `:trunk`; the default `_with_mode` accepts `:tensor` alone.

One further default is silent rather than fatal, so a new family should decide about it deliberately: the burial hooks
of `_coverage_constraints!` (`src/dofs.jl`), `_sheds_any` and `_reproduces` above all — `false` by default, so on a
family that does not opt in the space's leaf semantics stay on and do nothing.
Opting in is not only about shedding dofs: where a covering level's span *contains* one of this level's functions, the
two are linearly dependent and the superposed operator is exactly singular until one of them is eliminated there. Both
shipped families opt in — integrated Legendre in `src/dofs.jl`, B-splines in the extension.

Nothing gates an immersed [`PhysicalDomain`](@ref) by family, so a family's overlay constraints owe it one exemption, on
internal faces: a face between an active and an inactive cell of the level's own grid, whose inactive cell holds no
material (`_is_fictitious`, `src/coverage.jl`), has no trace to vanish on, and constraining it would over-constrain the
adjacent cut cell's modes. It is a single test on the inactive neighbour of the face, and both shipped families make it
(`_has_overlay_constraint` in `src/dofs.jl`; `_active_cell_at_face` in the extension, and `_cell_admissible` at
`:maximal` continuity). A level's own box face follows the box-face rule instead: next to an active cell it is
constrained unless that side of the level lies on the boundary of the space's domain (`_level_side_is_physical`),
whatever lies beyond it. That holds for integrated Legendre and for B-splines below `:maximal` continuity; the
`:maximal` B-spline support selection asks the material question past the box face too (`_cell_admissible`, case 3), and
keeps a function whose support reaches beyond an artificial box face only into cells that hold no material.

The integrated Legendre family (this file) and the B-spline extension are the two worked examples.
"""
abstract type BasisFamily end

"""
    IntegratedLegendre()

Default basis family: hierarchical integrated Legendre tensor-product basis on axis-aligned Cartesian cells.

In each axis the 1D modes are

    N̂₀(ξ) = (1 − ξ) / 2
    N̂₁(ξ) = (1 + ξ) / 2
    N̂ₘ(ξ) = (Lₘ(ξ) − Lₘ₋₂(ξ)) / √(4m − 2)        for m ≥ 2

where Lₘ is the standard Legendre polynomial. Modes 0 and 1 are the linear endpoint shape functions; modes m ≥ 2 are
bubble functions that vanish at the cell boundaries (`N̂ₘ(±1) = 0`). Each derivative `N̂ₘ′` is a scaled Legendre
polynomial, so the bubble block of the 1D stiffness matrix is diagonal for any scaling; the `√(4m − 2)` normalisation is
what fixes that diagonal at `⟨N̂ₘ′, N̂ₙ′⟩_{L²(−1,1)} = δₘₙ`, keeping the block identically scaled at high order.

`D`-dimensional basis functions are tensor products of the 1D modes:

    N_α(ξ) = ∏_{d=1}^{D} N̂_{α_d}(ξ_d),   α = (α₁, …, α_D),   0 ≤ α_d ≤ p_d

Two index sets are available, set by the `mode` keyword on [`space`](@ref) / [`overlay`](@ref) / [`ladder`](@ref):

  - `:tensor` — the full tensor product `∏_d {0, …, p_d}` (default).
  - `:trunk` — the Szabó–Babuška trunk (serendipity) space at isotropic order `p`. Keep every multi-index whose *trunk
    degree* `Σ_d t(α_d) ≤ p`, where `t(α_d) = α_d` for a bubble mode `α_d ≥ 2` and `t(α_d) = 0` for the two linear
    endpoint modes `α_d ∈ {0, 1}`. This retains all vertex modes and every edge mode up to degree `p` (so `C⁰`
    inter-cell conformity is preserved) while trimming the high-degree interior face/volume modes to `Σ` of their bubble
    degrees `≤ p`. Requires isotropic order. The saving over `:tensor` grows with dimension: e.g. for `p = 4` a
    hexahedral cell carries 50 modes instead of 125.

`IntegratedLegendre()` is the one public spelling. The family value also carries the index set it serves, as an
internal flag the `mode` keyword sets when a level is built, so a level's family is the whole answer to "which
functions does a cell of this order carry" (see [`BasisFamily`](@ref)); pass `mode = :trunk` rather than building the
trunk family by hand. It displays as `IntegratedLegendre()` in `:tensor` mode and is marked in `:trunk` mode.

All integrated-Legendre-specific dof and constraint behaviour (endpoint-vs-bubble identification, boundary and facet
incidence through `is_boundary_basis` / `is_facet_basis`) lives in this file and `src/dofs.jl`; the rest of the codebase
queries this family through the [`BasisFamily`](@ref) interface.
"""
struct IntegratedLegendre <: BasisFamily
    trunk::Bool    # the `:trunk` index set; internal, set from the `mode` keyword by `_with_mode`
end

IntegratedLegendre() = IntegratedLegendre(false)

Base.show(io::IO, b::IntegratedLegendre) = print(io, "IntegratedLegendre()", b.trunk ? " (mode=:trunk)" : "")

"""
    basis_name(basis) -> Symbol

Short identifier used in error messages and diagnostics output.
"""
basis_name(::IntegratedLegendre) = :integrated_legendre

# ── Quadrature rules ──────────────────────────────────────────────────────────

"""
    TensorQuadrature{D,T}(points, weights)

Tensor-product Gauss–Legendre quadrature on the reference cube `[−1, 1]ᴰ`. Built once per `(per_axis_counts, T)` pair by
`_tensor_gauss_rule` and cached, so all `:full` regions of the same per-axis order share the same `points`/`weights`
arrays (see [`Region`](@ref)).
"""
struct TensorQuadrature{D,T<:Real}
    points::Vector{SVector{D,T}}
    weights::Vector{T}
end

# The tensor product of per-axis 1D rules, `axis_rules[d] = (nodes, weights)`: the points in `CartesianIndices` order
# (axis 1 fastest) and each weight the left-fold product of its per-axis weights. The one tensor-rule constructor; the
# Gauss rule below and every rule built from per-axis pieces go through it.
function _tensor_rule(axis_rules::NTuple{D,Tuple{Vector{T},Vector{T}}}) where {D,T}
    counts = map(rule -> length(first(rule)), axis_rules)
    points = Vector{SVector{D,T}}(undef, prod(counts))
    weights = Vector{T}(undef, prod(counts))
    for (k, index) in enumerate(CartesianIndices(counts))
        points[k] = SVector{D,T}(ntuple(d -> axis_rules[d][1][index.I[d]], D))
        weights[k] = prod(axis_rules[d][2][index.I[d]] for d in 1:D)
    end
    return TensorQuadrature{D,T}(points, weights)
end

# A fresh tensor Gauss–Legendre rule with `counts[d]` points along axis `d` (FastGaussQuadrature's 1D rules). A count of
# 0 is an axis the rule does not extend along — a boundary facet's pinned axis — and carries the single node 0 with
# weight 1, so a facet's rule lives in the full `D`-dimensional frame with the free-axis points and weights unchanged.
# Always called through a cache (`_cached_tensor_quadrature!`), never on the hot path.
function _tensor_gauss_rule(counts::NTuple{D,Int}, ::Type{T}) where {D,T<:Real}
    return _tensor_rule(ntuple(D) do d
                            counts[d] == 0 && return (T[0], T[1])
                            points, weights = gausslegendre(counts[d])
                            return (T.(points), T.(weights))
                        end)
end

"""
    recommended_quadrature_order(basis, order) -> NTuple{D,Int}

Per-axis number of Gauss–Legendre points that integrates products of two basis functions of polynomial order `order[d]`
exactly. The `::BasisFamily` default is `order[d] + 1`: an `n`-point Gauss rule is exact for polynomials up to degree
`2n − 1`, which covers the degree `2 · order[d]` of a product of two degree-`order[d]` modes. A family whose functions
are not degree-`order[d]` polynomials overrides this.
"""
recommended_quadrature_order(::BasisFamily, order::NTuple{D,Int}) where {D} = ntuple(i -> order[i] + 1, D)

# ── Basis modes and local mode counts ─────────────────────────────────────────

"""
    local_basis_count(basis, order) -> Int

Number of local basis functions on a single cell: the size of the matching [`local_basis_indices`](@ref) set, so the
count validates exactly what the set validates and cannot describe a set the family does not have — integrated Legendre
refuses an order below 1 on any axis here as it does there. The family value carries its mode, so on a `:tensor` family
this is `prod(order .+ 1)` on every shipped family, and on integrated Legendre's `:trunk` family it counts the
multi-indices passing the trunk-degree filter (see [`IntegratedLegendre`](@ref)).
"""
local_basis_count(basis::BasisFamily, order::NTuple{D,Int}) where {D} = length(local_basis_indices(basis, order))

# The family value that serves the index set `mode` names, applied once where a level is built (`space`, `overlay`, and
# through them `ladder`): the `mode` keyword is folded into the family here, and every index-set question after it takes
# the family alone. `:tensor` — the full tensor-product set — is the only mode every tensor-product family serves, so
# the default accepts it and returns the family unchanged; a family offering more overrides this, and integrated
# Legendre's method sets its `trunk` flag. A symbol the package does not define is refused as unknown, and a defined
# mode the family does not carry is refused naming the family and the one mode it serves, so a family needs no
# hand-written mode guard of its own. The keyword always wins: an integrated-Legendre family handed in with one mode and
# asked for the other gets the other. `_mode` reads the mode back, for display, the pin and the diagnostics report.
_known_mode(m::Symbol) = m in (:tensor, :trunk) ? m : throw(ArgumentError("basis mode must be :tensor or :trunk"))

function _with_mode(basis::BasisFamily, mode::Symbol)
    _known_mode(mode) === :tensor && return basis
    throw(ArgumentError("basis family $(basis_name(basis)) does not support mode=:$mode; supported: tensor"))
end

_with_mode(::IntegratedLegendre, mode::Symbol) = IntegratedLegendre(_known_mode(mode) === :trunk)

_mode(::BasisFamily) = :tensor
_mode(b::IntegratedLegendre) = b.trunk ? :trunk : :tensor

# ── 1D Legendre primitives ────────────────────────────────────────────────────

"""
    legendre_value(n, x) -> Real

Standard Legendre polynomial `Lₙ(x)` evaluated via the three-term recurrence

    L₀(ξ) = 1
    L₁(ξ) = ξ
    Lₘ(ξ) = ((2m − 1) ξ Lₘ₋₁(ξ) − (m − 1) Lₘ₋₂(ξ)) / m,     m ≥ 2.

Used by the per-mode value/derivative wrappers below for testing and one-off evaluation. The hot-path assembly route
does not call this — see `_fill_factor_tables!` for the fused per-axis recurrence that serves it.
"""
function legendre_value(n::Integer, x::Real)
    n >= 0 || throw(ArgumentError("Legendre index must be nonnegative"))
    T = typeof(float(x))
    ξ = convert(T, x)

    n == 0 && return one(T)
    n == 1 && return ξ

    pnm2 = one(T)
    pnm1 = ξ
    pn = zero(T)
    for k in 2:n
        pn = ((2k - 1) * ξ * pnm1 - (k - 1) * pnm2) / k
        pnm2, pnm1 = pnm1, pn
    end

    return pn
end

"""
    integrated_legendre_value(i, ξ) -> Real

Integrated Legendre 1D basis value `N̂ᵢ(ξ)` for `i ≥ 0` on the reference interval `[−1, 1]`:

    N̂₀(ξ) = (1 − ξ) / 2
    N̂₁(ξ) = (1 + ξ) / 2
    N̂ᵢ(ξ) = (Lᵢ(ξ) − Lᵢ₋₂(ξ)) / √(4i − 2),     i ≥ 2.

Modes 0 and 1 are the linear endpoint shape functions; modes i ≥ 2 are bubbles that vanish at ξ = ±1.
"""
function integrated_legendre_value(i::Integer, ξ::Real)
    i >= 0 || throw(ArgumentError("basis index must be nonnegative"))
    T = typeof(float(ξ))
    x = convert(T, ξ)

    i == 0 && return (one(T) - x) / 2
    i == 1 && return (one(T) + x) / 2

    return (legendre_value(i, x) - legendre_value(i - 2, x)) / sqrt(convert(T, 4i - 2))
end

"""
    integrated_legendre_derivative(i, ξ) -> Real

Derivative `N̂ᵢ′(ξ)` of the integrated Legendre 1D basis function with respect to the reference coordinate:

    N̂₀′(ξ) = −1/2
    N̂₁′(ξ) = +1/2
    N̂ᵢ′(ξ) = √((2i − 1) / 2) · Lᵢ₋₁(ξ),     i ≥ 2.

The closed form for `i ≥ 2` follows from differentiating `N̂ᵢ = (Lᵢ − Lᵢ₋₂) / √(4i − 2)` and applying the standard
Legendre identity `Lₘ′ − Lₘ₋₂′ = (2m − 1) Lₘ₋₁`.
"""
function integrated_legendre_derivative(i::Integer, ξ::Real)
    i >= 0 || throw(ArgumentError("basis index must be nonnegative"))
    T = typeof(float(ξ))
    x = convert(T, ξ)

    i == 0 && return -one(T) / 2
    i == 1 && return one(T) / 2

    return sqrt(convert(T, 2i - 1) / 2) * legendre_value(i - 1, x)
end

# ── Local basis indexing ──────────────────────────────────────────────────────

# The lexicographic tensor index set `∏_d {0, …, order[d]}` with axis 1 varying fastest. This ordering is the package's
# tensor-product convention (see `CONTRIBUTING.md`, "Coordinate conventions"), so it lives in one place and every
# family's `local_basis_indices` is a filter over it, never a re-derivation of it.
_tensor_index_set(order::NTuple{D,Int}) where {D} = vec(collect(CartesianIndices(ntuple(d -> 0:order[d], D))))

"""
    local_basis_indices(basis, order) -> Vector{CartesianIndex{D}}

Multi-indices `α = (α₁, …, α_D)` enumerating the local basis functions on a cell of polynomial order `order`.
Lexicographic over `CartesianIndices(ntuple(d -> 0:order[d], D))`, axis 1 varying fastest — this is the canonical
tensor-product ordering used by every basis consumer (assembly, projection, post-processing, the dof layer).

The `::BasisFamily` default is that full tensor set, which is what an open-knot B-spline span and any other
tensor-product family need; a family overrides it only to *filter* the set, and inherits the ordering either way. The
family value carries its basis mode (see [`BasisFamily`](@ref)), so the set is the one that mode selects. Integrated
Legendre overrides it to require order ≥ 1 per axis (order 0 would leave a single endpoint mode, not a partition of
unity) and, in `:trunk` mode, to keep only the indices with trunk degree `Σ_d t(α_d) ≤ p`, where `t(α_d) = α_d` for a
bubble mode (`α_d ≥ 2`) and `t(α_d) = 0` for the two linear endpoint modes (`α_d ∈ {0, 1}`) — the trunk space of

> B. Szabó, I. Babuška, *Finite Element Analysis*, Wiley, New York (1991), ISBN 978-0-471-50273-9,

which is also the source of the minimum rule `src/dofs.jl` applies across an order jump (isotropic order `p` required,
see [`IntegratedLegendre`](@ref)).
"""
function local_basis_indices(::BasisFamily, order::NTuple{D,Int}) where {D}
    all(o -> o >= 0, order) || throw(ArgumentError("basis order must be nonnegative in every axis"))
    return _tensor_index_set(order)
end

# Trunk degree `Σ_d t(α_d)` of a multi-index `α`, where `t(α_d) = α_d` for a bubble mode `α_d ≥ 2` and `t(α_d) = 0` for
# the two endpoint modes 0 and 1, the linear vertex shape functions. This is what the `:trunk` mode compares against the
# isotropic polynomial order `p`: edge modes (one bubble axis) survive up to degree p, while face/volume modes survive
# only while their bubble degrees sum to ≤ p. Counting the vertex factors as 0 is what lets the filter keep every edge
# mode up to degree p while still trimming the interior — a pure polynomial-total-degree filter would instead drop the
# top edge modes and break C⁰ conformity.
_trunk_degree(id::CartesianIndex) = sum(a -> a ≥ 2 ? a : 0, Tuple(id))

# The trunk filter compares against one scalar `p`, so a trunk family refuses an anisotropic order first, then — as in
# either mode — an order below 1 on some axis.
function local_basis_indices(basis::IntegratedLegendre, order::NTuple{D,Int}) where {D}
    basis.trunk && any(!=(order[1]), order) && throw(ArgumentError("mode=:trunk requires isotropic order"))
    all(o -> o >= 1, order) || throw(ArgumentError("integrated Legendre order must be at least 1 in every axis"))
    indices = _tensor_index_set(order)
    return basis.trunk ? filter!(id -> _trunk_degree(id) <= order[1], indices) : indices
end

# ── Family capability traits ──────────────────────────────────────────────────

# Whether a basis family can carry a polynomial order that varies from cell to cell. Defaults to `false` and is answered
# `true` only by integrated Legendre, whose hierarchic 1D modes make a mixed-order interface a pure question of which
# shared-entity modes to generate.
#
# B-splines answer `false`, and the reason is structural rather than a missing implementation: the degree is a type
# parameter of the whole-axis knot vector (`BSplineSpace{p}(kv)`), the clamped end multiplicity is `p + 1` so changing p
# on one cell rewrites the vector's endpoints, the 1D dimension is `cells + p` — a global count — and a function's
# support is `p + 1` cells wide, so no set of functions belongs to one cell for a per-cell degree to name. There is
# likewise no shared entity for a minimum rule to attach to: continuity at an interior knot is C^(p−1), a property of
# the knot rather than of a face between two cells with independent degrees.
#
# The trait is consulted only when the normalised order field is genuinely non-uniform. A B-spline level handed a
# uniform per-cell field is still a perfectly well defined level and must keep working. A family that answers `true`
# owes the dof layer a `_cell_modes` method as well: the generic one (`mesh.jl`) refuses a real palette rather than
# assembling every cell at the first palette entry, because the minimum rule is the family's to state.
_supports_cell_order(::BasisFamily) = false
_supports_cell_order(::IntegratedLegendre) = true

# How far, in cells, a basis function of this family reaches beyond the cell that generates it — the radius
# [`support_extension`](@ref) dilates a marked region by, per axis.
#
# The default is zero, and it is the right answer for every family whose functions are element-local or shared only
# across the face they sit on: integrated Legendre's bubbles live in one cell and its endpoint modes are shared by the
# two cells meeting at the node, so a marked cell's own functions are generated by that cell and activating it is
# enough.
#
# It is *not* the right answer for a family with wider support. A degree-`p` B-spline spans `p + 1` cells, and the
# maximal-continuity selection rule keeps only the functions whose whole support lies inside the active region, so a
# level contributes nothing until the region is more than `p` cells wide in every axis and an activation front must
# advance `p` cells before it carries a new unknown. That is a property of the spline space rather than of the selection
# — minimal support needs `p + 1` intervals — so the region a caller marks has to be dilated rather than the rule
# relaxed. The requirement is the support-extension condition behind the admissible hierarchical meshes of
#
# > A. Buffa, C. Giannelli, *Adaptive isogeometric methods with hierarchical splines: Error estimator and convergence*,
# > Math. Models Methods Appl. Sci. **26** (2016) 1–25,
# > [doi:10.1142/S0218202516500019](https://doi.org/10.1142/S0218202516500019).
_support_radius(::BasisFamily, order::NTuple{D,Int}) where {D} = ntuple(_ -> 0, D)

"""
    _index_admissible(basis, order, id) -> Bool

Whether the multi-index `id` belongs to `local_basis_indices(basis, order)`, answered in `O(D)` instead of by scanning
the set.

This is the family's own index-set filter, factored out so the dof layer's minimum rule (`_entity_carries` in
`src/dofs.jl`) can ask, per active incident cell, whether *that cell's own* order generates the mode. The intersection
over the incident cells is the classical componentwise-minimum rule, because this predicate is monotone non-decreasing
in the order — raising an order never removes a mode from a cell's set — so the minimum need never be formed and no
question is ever asked at an order no cell carries. The "WHY THE INTERSECTION IS THE MINIMUM RULE" block in
`src/dofs.jl` states that argument in full.

Asking the *filter* rather than comparing modes per axis is what makes the rule family-generic: for `:tensor` the two
agree, and for `:trunk` they do not, because a 3D face mode with bubble degrees `(3, 2)` passes a per-axis test at order
4 while its trunk degree 5 puts it outside the order-4 trunk set. A family that adds a mode adds it here and the minimum
rule follows.

Integrated Legendre, the one family declaring `_supports_cell_order`, has the only method: the per-axis box, plus the
trunk-degree filter on a `:trunk` family. `test_cell_order.jl` asserts that this predicate agrees with
`local_basis_indices` over a range of orders, modes and dimensions, that it is monotone in the order for every family
declaring `_supports_cell_order`, and that the intersection form reproduces the componentwise-minimum form cell for cell
— the three contracts that keep the two layers from drifting.
"""
function _index_admissible(basis::IntegratedLegendre, order::NTuple{D,Int}, id::CartesianIndex{D}) where {D}
    all(d -> 0 <= id.I[d] <= order[d], 1:D) || return false
    return !basis.trunk || _trunk_degree(id) <= order[1]
end

# ── Hot-path tensor-product evaluation ────────────────────────────────────────

# Caller-owned per-axis 1D scratch buffers used by `_fill_factor_tables!` and the tensor-product kernel. One `Vector{T}`
# per axis, length `order[d] + 1`. The assembly workspace pre-allocates one of these per level so the hot loop never
# reallocates.
#
# The per-axis size requirement `order[d] + 1` is the count of 1D modes active on a single cell for *every* basis family
# the package currently supports: integrated Legendre carries one endpoint per side plus `order − 1` bubbles, and an
# open-knot B-spline of degree `p` has exactly `p + 1` 1D functions non-zero on any single span. So the buffer
# allocation is family-agnostic; only the per-axis evaluator (`_fill_factor_tables!`) dispatches.
_factor_buffers(order::NTuple{D,Int}, ::Type{T}) where {D,T} = ntuple(d -> Vector{T}(undef, order[d] + 1), D)

# Fill caller-owned per-axis 1D tables for `basis` at one reference point: `val1d[d][i + 1] = N̂ᵢ(ξ[d])` for the
# `order[d] + 1` 1D modes active on the cell along axis `d`, and, unless `der1d === nothing`, also
# `der1d[d][i + 1] = N̂ᵢ′(ξ[d])`. One method per family serves both: `der1d` is `nothing` or a tuple of vectors by type,
# so the derivative work folds away at compile time on the values-only path (the transfer right-hand side and the
# Dirichlet trace) and no second method can drift from the first. The recurrence is the family's own (Legendre
# three-term for integrated Legendre, de Boor for B-splines); both cost `O(D · (order + 1))` per point, so the
# tensor-product kernel downstream stays family-agnostic.
#
# `cell::CartesianIndex{D}` is the parent cell's mesh index along each axis. Integrated Legendre ignores it (its 1D
# modes are cell-local in the reference frame, identical on every cell); the B-spline family in the extension uses it as
# the knot span the parent cell occupies. Callers always pass the same `parent.cell` the rest of the assembly loop uses,
# so the integrated-Legendre overhead is zero and the B-spline method gets the span for free.
#
# The integrated-Legendre recurrence below is the in-package method; the B-spline one lives in
# `ext/UnfittedBasicBSplineExt.jl`. Each step of the three-term recurrence yields `Lₘ` from `Lₘ₋₁` and `Lₘ₋₂`, which is
# all that `N̂ₘ = (Lₘ − Lₘ₋₂) / √(4m − 2)` and `N̂ₘ′ = √((2m − 1)/2) · Lₘ₋₁` need. The derivative's constant is taken
# from the value's: `√((2m − 1)/2) = √(4m − 2) / 2` holds bit for bit under a correctly rounded square root, because
# `4m − 2 = 4 · (2m − 1)/2` with both scalings exact powers of two, so the gradient path pays one square root per mode
# instead of two and returns exactly what the closed forms `integrated_legendre_value` / `_derivative` return.
function _fill_factor_tables!(::IntegratedLegendre, val1d, der1d, order::NTuple{D,Int}, ξ::SVector{D,T},
                              ::CartesianIndex{D}) where {D,T}
    @inbounds for d in 1:D
        x = ξ[d]
        p = order[d]
        vd = val1d[d]
        dd = der1d === nothing ? nothing : der1d[d]
        vd[1] = (one(T) - x) / 2
        dd === nothing || (dd[1] = -one(T) / 2)
        if p >= 1
            vd[2] = (one(T) + x) / 2
            dd === nothing || (dd[2] = one(T) / 2)
        end
        ℓm2, ℓm1 = one(T), x
        for m in 2:p
            ℓm = ((2m - 1) * x * ℓm1 - (m - 1) * ℓm2) / m
            s = sqrt(T(4m - 2))
            vd[m + 1] = (ℓm - ℓm2) / s
            dd === nothing || (dd[m + 1] = s / 2 * ℓm1)
            ℓm2, ℓm1 = ℓm1, ℓm
        end
    end
    return nothing
end

# The tensor-product kernel, the one evaluator every hot loop calls: fills the per-axis tables through
# `_fill_factor_tables!(basis, …)` (the only family-specific step), then walks each multi-index `α` in `indices` and
# writes
#
#     values[a]       = N_α(ξ)    = ∏_d val1d[d][α_d + 1]
#     gradients[a][d] = ∂N_α/∂x_d = der1d[d][α_d + 1] · ∏_{j ≠ d} val1d[j][α_j + 1] · scale[d]     (when asked)
#
# `grad` is `nothing` for values alone (the transfer right-hand side and the Dirichlet trace, which never read a
# gradient) or the tuple `(gradients, scale, der1d)`: the gradient buffer, the per-axis chain-rule factor from reference
# to physical coordinates (`2 / h_d` on an axis-aligned cell of edge lengths `h_d`, ones for the bare reference
# gradient), and the derivative tables. It is a type either way, so the gradient work folds away at compile time and one
# loop serves both paths without a second copy to keep in step. The loop is family-agnostic: it only assumes the
# per-axis tables are indexed by the 1D mode number on the cell, the convention every family here follows. The caller
# owns every buffer. The "product except one" is `O(D)` per gradient component, `O(D²)` per multi-index, on `NTuple`
# scratch and allocation-free.
#
# The entry point is a thin `@inline` method that unpacks `grad` into plain arguments of the loop's body,
# `_tensor_kernel!`, so the tuple is taken apart in the caller and never built. Handing the tuple whole to a body that
# is not inlined was measured 12–15 % slower on the gradient path of a 3D integrated-Legendre level: the caller then
# stores the tuple to memory on every call, and the body loads the gradient buffer, scale and derivative tables back out
# of it and roots them in a GC frame twice the size.
#
# The buffer checks are `>=`, not `==`. Under a per-cell polynomial order the workspace banks stay sized at the level's
# longest minimum-rule set while `indices` is the cell's own set, which is shorter on every cell below the maximum.
# Writing into a prefix of the longer buffer is transparent to every consumer, because they all bound their loop by
# `eachindex(raw_dofs)` and read `values[a]` positionally — the pairing is with the index list, not with the buffer's
# length. Length-matching with a `view` instead was measured at +24–30% on this kernel and is not worth it. What `>=`
# gives up is the guard against a *short* buffer, so `test_cell_order.jl` asserts the surviving checks still fire.
#
# The 1D factor tables are likewise filled at the nominal order and read only up to the cell's own: for integrated
# Legendre `_fill_factor_tables!` computes mode `m` from `ξ` and `m` alone, with no `p` anywhere in the recurrence, so
# slots `1:p_cell+1` hold bit-identical values whether the table was filled at `p_cell` or at `p_max`. That identity is
# what makes the nominal-order banks correct, and it is *false* for a family whose 1D modes depend on the degree — de
# Boor at degree p yields p+1 span-specific functions — which is one more reason `_supports_cell_order` is false for
# B-splines.
@inline function _tensor_values!(basis::BasisFamily, values::AbstractVector, indices::AbstractVector{CartesianIndex{D}},
                                 order::NTuple{D,Int}, ξ::SVector{D}, val1d, cell::CartesianIndex{D},
                                 grad=nothing) where {D}
    gradients, scale, der1d = grad === nothing ? (nothing, nothing, nothing) : grad
    return _tensor_kernel!(basis, values, gradients, indices, order, ξ, scale, val1d, der1d, cell)
end

# The body of `_tensor_values!` above, with the gradient buffer, scale and derivative tables as plain arguments, each
# `nothing` on the values-only path.
function _tensor_kernel!(basis::BasisFamily, values::AbstractVector, gradients,
                         indices::AbstractVector{CartesianIndex{D}}, order::NTuple{D,Int}, ξ::SVector{D,T}, scale,
                         val1d, der1d, cell::CartesianIndex{D}) where {D,T}
    length(values) >= length(indices) || throw(DimensionMismatch("basis value buffer has wrong length"))
    gradients === nothing ||
        length(gradients) >= length(indices) ||
        throw(DimensionMismatch("basis gradient buffer has wrong length"))
    _fill_factor_tables!(basis, val1d, der1d, order, ξ, cell)
    @inbounds for (a, id) in pairs(indices)
        vfac = ntuple(d -> val1d[d][id.I[d] + 1], D)
        values[a] = prod(vfac)
        gradients === nothing && continue
        gradients[a] = SVector{D,T}(ntuple(k -> der1d[k][id.I[k] + 1] * _product_except(vfac, k) * scale[k], D))
    end
    return values
end

# Product `∏_{d ≠ k} values[d]`. Used by the gradient kernel to compute the "value part" of the gradient component along
# axis `k` without recomputing the full product.
function _product_except(values::NTuple{D}, k::Integer) where {D}
    result = one(values[1])
    for d in 1:D
        d == k || (result *= values[d])
    end
    return result
end

# ── Boundary and facet basis identification ───────────────────────────────────

# True iff the tensor-product basis function with multi-index `id` carries support on the codim-1 face specified by
# `(axis, side)`. For integrated Legendre this is a test of the single 1D mode along the requested axis: mode 0 lives on
# the lower endpoint (`ξ = −1`), mode 1 on the upper endpoint (`ξ = +1`), and modes m ≥ 2 (bubbles) vanish at both
# endpoints — they never sit on either side.
function is_boundary_basis(::IntegratedLegendre, id::CartesianIndex{D}, axis::Integer, side::Symbol) where {D}
    1 <= axis <= D || throw(ArgumentError("axis out of bounds"))
    side === :lower && return id.I[axis] == 0
    side === :upper && return id.I[axis] == 1
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# Codim-K facet membership for a tensor-product multi-index: the basis function sits on the facet specified by `sides`
# (a list of `(axis, side)` pairs) iff its 1D mode along *every* listed axis touches the requested side. Codim 1 reduces
# to a single `is_boundary_basis` check; codim K requires every listed axis-side match. This `::BasisFamily` default
# delegates the per-axis test to the family's `is_boundary_basis`, so it needs no family-specific override.
function is_facet_basis(basis::BasisFamily, id::CartesianIndex{D},
                        sides::AbstractVector{<:Tuple{Integer,Symbol}}) where {D}
    for (axis, side) in sides
        is_boundary_basis(basis, id, axis, side) || return false
    end
    return true
end

"""
    boundary_basis_indices(basis, order; axis, side) -> Vector{CartesianIndex{D}}

Multi-indices of the local basis functions whose support touches the codim-1 face at the requested `(axis, side)`. Used
by the Dirichlet projection in `src/dofs.jl` to identify which trial / test modes contribute on each physical facet.
This `::BasisFamily` default filters `local_basis_indices(basis, order)` — the set of the family's own mode — through
the family's `is_boundary_basis`, so it needs no override.
"""
function boundary_basis_indices(basis::BasisFamily, order::NTuple{D,Int}; axis::Integer, side::Symbol) where {D}
    return [id for id in local_basis_indices(basis, order) if is_boundary_basis(basis, id, axis, side)]
end

# ── Public quadrature rule ────────────────────────────────────────────────────

"""
    gauss_rule(basis::IntegratedLegendre, order, [T=Float64]) -> TensorQuadrature

Tensor Gauss–Legendre rule on the reference cube `[−1, 1]ᴰ` sized by [`recommended_quadrature_order`](@ref) for the
given basis and polynomial order. Useful for one-shot reference-cell quadrature in tests and examples; the assembly path
uses the cached `_tensor_gauss_rule` directly.
"""
function gauss_rule(basis::IntegratedLegendre, order::NTuple{D,Int}, ::Type{T}=Float64) where {D,T<:Real}
    _tensor_gauss_rule(recommended_quadrature_order(basis, order), T)
end
