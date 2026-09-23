# ── Basis family interface ────────────────────────────────────────────────────

"""
    BasisFamily

Abstract supertype for every basis family supported by the package. A basis
family answers, for each cell or span, the questions assembly and the dof
layer ask:

  - how many local basis functions live on a cell;
  - which multi-indices identify them and in what tensor-product order;
  - what their values and gradients are at a reference point, possibly
    pulled back from a physical cell;
  - which of them touch a given axis-side of the cell (used to pick the
    facet-incident modes for the Dirichlet boundary trace in
    `src/dirichlet.jl`; the dof layer's own "is this dof on a face" test
    is the separate `_key_on_level_side` hook, which works on dof keys
    rather than local mode indices);
  - what quadrature order integrates products of two basis functions
    exactly on a `:full` region.

The concrete implementation shipped here is [`IntegratedLegendre`](@ref); the
`BasicBSpline` package extension adds a second family. Adding a family means
implementing this interface plus the corresponding dof/constraint behaviour —
geometry, intersections, assembly, projection, and solvers stay untouched.

A new family `F <: BasisFamily` must provide:

  - `basis_name(::F) -> Symbol` — the short identifier error messages and
    diagnostics print. There is no `::BasisFamily` default, so a family that
    omits it turns the dof layer's own "not implemented for basis family"
    message into an unrelated `MethodError`.
  - `_fill_factor_tables!(::F, val1d[, der1d], order, ξ, cell)` — the per-axis
    1D value (and derivative) tables at one reference point on one cell. This
    is the *only* family-specific step on the hot path: tensor-product
    evaluation, assembly, projection, field evaluation, and the boundary trace
    all reach the family through it and through nothing else.
  - `is_boundary_basis(::F, id, axis, side)` — per-axis facet incidence.
  - the dof-key and overlay-constraint hooks the dof layer needs:
    `_tensor_dof_key`, `_key_on_level_side`, and either the per-raw
    `_has_overlay_constraint` predicate or a whole-level `_overlay_constraints`
    generator (see `src/dofs.jl`).
  - `instantiate_basis(::F, mesh, order, mode, mask)` — only when the concrete
    family depends on the level's mesh. The `::BasisFamily` default returns the
    family unchanged; see [`instantiate_basis`](@ref) for the five-argument
    contract and why the hook exists.
  - `_supports_cell_order(::F)` — `false` by default, and a *loud* default: a
    non-uniform `order =` on a family that declines it raises rather than being
    silently flattened. Answer `true` only where a per-cell degree names a set of
    functions and the minimum rule has a shared entity to act on, and supply the
    matching `_cell_modes` method (`src/dofs.jl`) in the same change —
    `_index_admissible` below is the per-axis filter that rule is expressed over,
    and `test_cell_order.jl` asserts its monotonicity in the order, which is what
    makes the rule correct.

It inherits the basis-agnostic `::BasisFamily` defaults and overrides only the
ones that do not fit: `recommended_quadrature_order` (`order .+ 1`),
`_supported_modes` (`(:tensor,)`), `local_basis_indices` / `local_basis_count`
(the lexicographic tensor index set `∏_d {0, …, order[d]}` and its size),
`is_facet_basis`, `boundary_basis_indices`, and the point evaluators
[`basis_values`](@ref) / [`physical_basis_gradients`](@ref), which are built on
`_fill_factor_tables!` and therefore already serve every family.

Two further defaults are silent rather than fatal, so a new family should decide
about them deliberately:

  - `_supports_physical_domain(::F)` — `true` by default, i.e. the family claims
    it can carry an immersed [`PhysicalDomain`](@ref). Answer `false` when the
    family's overlay constraints would over-constrain cut-cell modes on
    fully-fictitious fold faces; [`space`](@ref) / [`overlay`](@ref) then reject
    the pairing with an `ArgumentError` instead of silently degrading the FCM
    solution. Both shipped families answer `true`: the fold exemption is the one
    thing a family must supply, and it is a single test on the inactive
    neighbour of a face.
  - `_coverage_constraints` (`src/dofs.jl`) — the generic method returns no
    constraints, so on a family that does not override it the space's leaf
    semantics stay on and do nothing. Overriding it is not only about
    shedding dofs: where a covering level's span *contains* one of this level's
    functions, the two are linearly dependent and the superposed operator is
    exactly singular until one of them is eliminated here. Both shipped
    families override it — integrated Legendre in `src/dofs.jl`, B-splines in
    the extension.

The integrated Legendre family (this file) and the B-spline extension are the
two worked examples.
"""
abstract type BasisFamily end

"""
    IntegratedLegendre()

Default basis family: hierarchical integrated Legendre tensor-product basis
on axis-aligned Cartesian cells.

In each axis the 1D modes are

    N̂₀(ξ) = (1 − ξ) / 2
    N̂₁(ξ) = (1 + ξ) / 2
    N̂ₘ(ξ) = (Lₘ(ξ) − Lₘ₋₂(ξ)) / √(4m − 2)        for m ≥ 2

where Lₘ is the standard Legendre polynomial. Modes 0 and 1 are the
linear endpoint shape functions; modes m ≥ 2 are bubble functions that
vanish at the cell boundaries (`N̂ₘ(±1) = 0`). Each derivative `N̂ₘ′` is a
scaled Legendre polynomial, so the bubble block of the 1D stiffness matrix
is diagonal for any scaling; the `√(4m − 2)` normalisation is what fixes
that diagonal at `⟨N̂ₘ′, N̂ₙ′⟩_{L²(−1,1)} = δₘₙ`, keeping the block
identically scaled at high order.

`D`-dimensional basis functions are tensor products of the 1D modes:

    N_α(ξ) = ∏_{d=1}^{D} N̂_{α_d}(ξ_d),   α = (α₁, …, α_D),   0 ≤ α_d ≤ p_d

Two index sets are available, set by the `mode` keyword on
[`space`](@ref) / [`overlay`](@ref):

  - `:tensor` — the full tensor product `∏_d {0, …, p_d}` (default).
  - `:trunk` — the Szabó–Babuška trunk (serendipity) space at isotropic
    order `p`. Keep every multi-index whose *trunk degree*
    `Σ_d t(α_d) ≤ p`, where `t(α_d) = α_d` for a bubble mode `α_d ≥ 2`
    and `t(α_d) = 0` for the two linear endpoint modes `α_d ∈ {0, 1}`.
    This retains all vertex modes and every edge mode up to degree `p`
    (so `C⁰` inter-cell conformity is preserved) while trimming the
    high-degree interior face/volume modes to `Σ` of their bubble
    degrees `≤ p`. Requires isotropic order. The saving over `:tensor`
    grows with dimension: e.g. for `p = 4` a hexahedral cell carries 50
    modes instead of 125.

All integrated-Legendre-specific dof and constraint behaviour
(endpoint-vs-bubble identification, boundary-mode dispatch, facet
detection) lives in this file and `src/dofs.jl`; the rest of the codebase
queries this family through the [`BasisFamily`](@ref) interface.
"""
struct IntegratedLegendre <: BasisFamily end

"""
    basis_name(basis) -> Symbol

Short identifier used in error messages and diagnostics output.
"""
basis_name(::IntegratedLegendre) = :integrated_legendre

# ── Quadrature rules ──────────────────────────────────────────────────────────

"""
    TensorQuadrature{D,T}(points, weights)

Tensor-product Gauss–Legendre quadrature on the reference cube `[−1, 1]ᴰ`.
Built once per `(per_axis_counts, T)` pair by `_tensor_gauss_rule`
and cached, so all `:full` regions of the same per-axis order share the
same `points`/`weights` arrays.
"""
struct TensorQuadrature{D,T<:Real}
    points::Vector{SVector{D,T}}
    weights::Vector{T}
end

"""
    RegionQuadrature{D,T}(kind, points, weights)

Per-region quadrature rule consumed by the assembly hot loop. `kind` is an
informational tag set by the integration-region dispatcher in
`intersections.jl`:

  - `:full` — region is entirely inside Ω (or no `PhysicalDomain` is
    attached). `points`/`weights` alias the shared cached tensor Gauss
    rule; no copy is made.
  - `:fictitious_alpha` — region is entirely outside Ω under α-FCM;
    weights are the tensor Gauss weights pre-multiplied by `α` and
    shared across regions of the same per-axis order via a per-plan
    cache.
  - `:cut_fitted` — region is crossed by ∂Ω; weights come from the NNMF
    moment-fit rule (see `src/fcm.jl`) and are unique per region. Under
    α-FCM (`α > 0`) the rule additionally carries the α-scaled full-cell
    tensor part (`(1−α)·moment-fit ∪ α·tensor`).
  - `:cut_fallback` — the moment-fit residual exceeded the catastrophic
    threshold, so the region carries the raw Saye volume rule the moments
    were summed from (see [`moment_fit_rule`](@ref)) instead of a fitted
    one: correct and non-negative, but with 50–200× the points. Under
    α-FCM the α-scaled tensor part is appended as for `:cut_fitted`.
  - `:cut_failed` — strict-cut (`α = 0`) region whose `Ω ∩ box` carries no
    volume rule at all; the region was emitted with empty
    `points`/`weights` so it contributes zero quadrature.
  - `:cut_alpha_failed` — same under α-FCM (`α > 0`); the physical part is
    empty but the region carries the α-scaled tensor rule alone so the
    cell's dofs stay α-stabilised rather than singular.
  - `:cut_custom` — region is crossed by ∂Ω and its rule came from the
    domain's own `cut_quadrature` callable instead of the moment fit (see
    the [`PhysicalDomain`](@ref) docstring). Under α-FCM the α-scaled
    tensor part is appended as for `:cut_fitted`.

All three non-fitted kinds are reported in
`AssemblyDiagnostics.fit_failure_count`; `:cut_fallback` is additionally
counted, with its point cost, in `cut_fallback_count` and
`cut_fallback_points`. `:cut_custom` is not among them — no fit ran, so
there is no residual to report on.

Assembly iterates `zip(points, weights)` without dispatching on `kind`,
so the hot loop's cost is the same regardless of the quadrature source.
"""
struct RegionQuadrature{D,T<:Real}
    kind::Symbol
    points::Vector{SVector{D,T}}
    weights::Vector{T}
end

# Build a fresh tensor Gauss–Legendre rule with `counts[d]` points along
# axis `d`. The per-axis 1D rules come from FastGaussQuadrature; we then
# enumerate `CartesianIndices(counts)` to build the `prod(counts)` tensor
# points and the corresponding product weights. Always called through the
# cache in `intersections.jl`, never on the hot path.
function _tensor_gauss_rule(counts::NTuple{D,Int}, ::Type{T}) where {D,T<:Real}
    axes = ntuple(D) do d
        points, weights = gausslegendre(counts[d])
        (T.(points), T.(weights))
    end

    npoints = prod(counts)
    points = Vector{SVector{D,T}}(undef, npoints)
    weights = Vector{T}(undef, npoints)
    counter = 1

    for index in CartesianIndices(counts)
        points[counter] = SVector{D,T}(ntuple(d -> axes[d][1][index.I[d]], D))
        weights[counter] = prod(axes[d][2][index.I[d]] for d in 1:D)
        counter += 1
    end

    return TensorQuadrature{D,T}(points, weights)
end

"""
    recommended_quadrature_order(basis, order) -> NTuple{D,Int}

Per-axis number of Gauss–Legendre points that integrates products of two
basis functions of polynomial order `order[d]` exactly. The `::BasisFamily`
default is `order[d] + 1`: an `n`-point Gauss rule is exact for polynomials up
to degree `2n − 1`, which covers the degree `2 · order[d]` of a product of two
degree-`order[d]` modes. A family whose functions are not degree-`order[d]`
polynomials overrides this.
"""
function recommended_quadrature_order(::BasisFamily, order::NTuple{D,Int}) where {D}
    return ntuple(i -> order[i] + 1, D)
end

# ── Basis modes and local mode counts ─────────────────────────────────────────

"""
    local_basis_count(basis, order)         -> Int
    local_basis_count(basis, order, mode)   -> Int

Number of local basis functions on a single cell. The two-argument form is the
tensor count `prod(order .+ 1)`, a `::BasisFamily` default for tensor-product
families. The three-argument form takes a basis mode and counts the index set
that mode selects, so it tracks [`local_basis_indices`](@ref) by construction:
for integrated Legendre `:trunk` counts the multi-indices passing the
trunk-degree filter (see [`IntegratedLegendre`](@ref)). A family that supports
only some modes says so through `_supported_modes`, not by counting.
"""
function local_basis_count(::BasisFamily, order::NTuple{D,Int}) where {D}
    return prod(ntuple(i -> order[i] + 1, D))
end

function local_basis_count(basis::BasisFamily, order::NTuple{D,Int}, mode::Symbol) where {D}
    return length(local_basis_indices(basis, order, mode))
end

# The basis modes a family carries. `:tensor` — the full tensor-product index
# set — is the only mode every tensor-product family can serve, so it is the
# default; a family offering more (integrated Legendre's `:trunk`) overrides
# this, and `_check_basis_mode` is the single place the answer is consulted.
_supported_modes(::BasisFamily) = (:tensor,)
_supported_modes(::IntegratedLegendre) = (:tensor, :trunk)

# Validate a basis-mode symbol against a concrete family and a per-axis
# polynomial order. Three conditions, in one place because there is only ever
# one caller position for them — the family's own index-set constructor:
#
#   * the symbol must be a mode the package defines at all;
#   * `:trunk` requires isotropic order, since the trunk-degree filter compares
#     against a single scalar `p`;
#   * the family must list the mode in its `_supported_modes`, which is what
#     rejects a mode the package defines but *this* family cannot serve, so a
#     family needs no hand-written mode guard of its own.
#
# There used to be a second, family-blind two-argument form, run by `space` and
# `overlay` before `instantiate_basis` had produced a concrete family. Nothing
# needed the answer that early: every `Level` construction reaches
# `local_basis_indices` — through `_build_cell_orders` on a graded level and
# through `_cell_modes` on a uniform one — and that is where this check lives.
function _check_basis_mode(basis::BasisFamily, mode::Symbol, order::NTuple{D,Int}) where {D}
    mode in (:tensor, :trunk) || throw(ArgumentError("basis mode must be :tensor or :trunk"))
    if mode === :trunk && any(!=(order[1]), order)
        throw(ArgumentError("mode=:trunk requires isotropic order"))
    end
    supported = _supported_modes(basis)
    mode in supported || throw(ArgumentError("basis family $(basis_name(basis)) does not support " *
                                             "mode=:$mode; supported: $(join(supported, ", "))"))
    return mode
end

# ── 1D Legendre primitives ────────────────────────────────────────────────────

"""
    legendre_value(n, x) -> Real

Standard Legendre polynomial `Lₙ(x)` evaluated via the three-term
recurrence

    L₀(ξ) = 1
    L₁(ξ) = ξ
    Lₘ(ξ) = ((2m − 1) ξ Lₘ₋₁(ξ) − (m − 1) Lₘ₋₂(ξ)) / m,     m ≥ 2.

Used by the per-mode value/derivative wrappers below for testing and one-
off evaluation. The hot-path assembly route does not call this — see
`_fill_factor_tables!` for the fused per-axis recurrence that
serves it.
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

Integrated Legendre 1D basis value `N̂ᵢ(ξ)` for `i ≥ 0` on the reference
interval `[−1, 1]`:

    N̂₀(ξ) = (1 − ξ) / 2
    N̂₁(ξ) = (1 + ξ) / 2
    N̂ᵢ(ξ) = (Lᵢ(ξ) − Lᵢ₋₂(ξ)) / √(4i − 2),     i ≥ 2.

Modes 0 and 1 are the linear endpoint shape functions; modes i ≥ 2 are
bubbles that vanish at ξ = ±1.
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

Derivative `N̂ᵢ′(ξ)` of the integrated Legendre 1D basis function with
respect to the reference coordinate:

    N̂₀′(ξ) = −1/2
    N̂₁′(ξ) = +1/2
    N̂ᵢ′(ξ) = √((2i − 1) / 2) · Lᵢ₋₁(ξ),     i ≥ 2.

The closed form for `i ≥ 2` follows from differentiating
`N̂ᵢ = (Lᵢ − Lᵢ₋₂) / √(4i − 2)` and applying the standard Legendre
identity `Lₘ′ − Lₘ₋₂′ = (2m − 1) Lₘ₋₁`.
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

# The lexicographic tensor index set `∏_d {0, …, order[d]}` with axis 1
# varying fastest. This ordering is the package's tensor-product convention
# (see `CONTRIBUTING.md`, "Coordinate conventions"), so it lives in one place
# and every family's `local_basis_indices` is a filter over it, never a
# re-derivation of it.
function _tensor_index_set(order::NTuple{D,Int}) where {D}
    return vec(collect(CartesianIndices(ntuple(d -> 0:order[d], D))))
end

"""
    local_basis_indices(basis, order)         -> Vector{CartesianIndex{D}}
    local_basis_indices(basis, order, mode)   -> Vector{CartesianIndex{D}}

Multi-indices `α = (α₁, …, α_D)` enumerating the local basis functions on
a cell of polynomial order `order`. Lexicographic over
`CartesianIndices(ntuple(d -> 0:order[d], D))`, axis 1 varying fastest —
this is the canonical tensor-product ordering used by every basis
consumer (assembly, projection, post-processing, the dof layer).

The `::BasisFamily` default is that full tensor set, which is what an
open-knot B-spline span and any other tensor-product family need; a family
overrides it only to *filter* the set, and inherits the ordering either way.
Integrated Legendre overrides both forms: the two-argument form to require
order ≥ 1 per axis (order 0 would leave a single endpoint mode, not a
partition of unity), the three-argument form to add `:trunk`.

The two-argument form is the `:tensor` set (no filter); `mode = :trunk` keeps only
indices with trunk degree `Σ_d t(α_d) ≤ p`, where `t(α_d) = α_d` for a
bubble mode (`α_d ≥ 2`) and `t(α_d) = 0` for the two linear endpoint
modes (`α_d ∈ {0, 1}`) — the trunk space of

> B. Szabó, I. Babuška, *Finite Element Analysis*, Wiley, New York (1991),
> ISBN 978-0-471-50273-9,

which is also the source of the minimum rule `src/dofs.jl` applies across an
order jump (isotropic order `p` required, see [`IntegratedLegendre`](@ref)).
"""
function local_basis_indices(::BasisFamily, order::NTuple{D,Int}) where {D}
    all(o -> o >= 0, order) || throw(ArgumentError("basis order must be nonnegative in every axis"))
    return _tensor_index_set(order)
end

function local_basis_indices(basis::BasisFamily, order::NTuple{D,Int}, mode::Symbol) where {D}
    _check_basis_mode(basis, mode, order)
    return local_basis_indices(basis, order)
end

function local_basis_indices(::IntegratedLegendre, order::NTuple{D,Int}) where {D}
    all(o -> o >= 1, order) ||
        throw(ArgumentError("integrated Legendre order must be at least 1 in every axis"))
    return _tensor_index_set(order)
end

# Trunk degree of a single 1D integrated-Legendre mode index. The two
# endpoint modes 0 and 1 are the linear vertex shape functions and count
# as 0; a bubble mode m ≥ 2 is an integrated Legendre polynomial of degree
# m and counts as m. Counting the vertex factors as 0 is what lets the
# trunk filter keep every edge mode up to degree p while still trimming
# the interior — a pure polynomial-total-degree filter would instead drop
# the top edge modes and break C⁰ conformity.
_trunk_degree(mode::Integer) = mode ≥ 2 ? Int(mode) : 0

# Trunk degree of a D-dim multi-index `α` is `Σ_d t(α_d)`. This is what the
# `:trunk` mode compares against the isotropic polynomial order `p`: edge
# modes (one bubble axis) survive up to degree p, while face/volume modes
# survive only while their bubble degrees sum to ≤ p.
function _trunk_degree(id::CartesianIndex{D}) where {D}
    degree = 0
    for axis in 1:D
        degree += _trunk_degree(id.I[axis])
    end
    return degree
end

function local_basis_indices(basis::IntegratedLegendre, order::NTuple{D,Int},
                             mode::Symbol) where {D}
    _check_basis_mode(basis, mode, order)
    indices = local_basis_indices(basis, order)
    mode === :tensor && return indices
    return [id for id in indices if _trunk_degree(id) <= order[1]]
end

# ── Family capability traits ──────────────────────────────────────────────────

# Whether a basis family can carry a polynomial order that varies from cell to
# cell. Defaults to `false` and is answered `true` only by integrated Legendre,
# whose hierarchic 1D modes make a mixed-order interface a pure question of which
# shared-entity modes to generate.
#
# B-splines answer `false`, and the reason is structural rather than a missing
# implementation: the degree is a type parameter of the whole-axis knot vector
# (`BSplineSpace{p}(kv)`), the clamped end multiplicity is `p + 1` so changing p
# on one cell rewrites the vector's endpoints, the 1D dimension is `cells + p` —
# a global count — and a function's support is `p + 1` cells wide, so no set of
# functions belongs to one cell for a per-cell degree to name. There is likewise
# no shared entity for a minimum rule to attach to: continuity at an interior
# knot is C^(p−1), a property of the knot rather than of a face between two cells
# with independent degrees.
#
# The trait is consulted only when the normalised order field is genuinely
# non-uniform. A B-spline level handed a uniform per-cell field is still a
# perfectly well defined level and must keep working. A family that answers
# `true` owes the dof layer a `_cell_modes` method as well: the generic one
# (`mesh.jl`) refuses a real palette rather than assembling every cell at the
# first palette entry, because the minimum rule is the family's to state.
_supports_cell_order(::BasisFamily) = false
_supports_cell_order(::IntegratedLegendre) = true

# Whether a basis family supports an immersed `PhysicalDomain` — the α-FCM fold,
# cut-cell moment-fit quadrature, and the fictitious-fold exemption that keeps a
# cut cell's boundary modes on its fully-fictitious fold faces. Defaults to
# `true`, and both shipped families answer it: the exemption is the only piece a
# family has to get right, and each has it. A family whose constraint generator
# lacks one must override this to `false` rather than degrade the FCM solution
# silently. `_check_physical_basis` in `mesh.jl` is the space-construction guard
# that consults it.
_supports_physical_domain(::BasisFamily) = true

"""
    _index_admissible(basis, order, mode, id) -> Bool

Whether the multi-index `id` belongs to `local_basis_indices(basis, order, mode)`,
answered in `O(D)` instead of by scanning the set.

This is the family's own index-set filter, factored out so the dof layer's
minimum rule (`_entity_carries` in `src/dofs.jl`) can ask, per active incident
cell, whether *that cell's own* order generates the mode. The intersection over
the incident cells is the classical componentwise-minimum rule, because this
predicate is monotone non-decreasing in the order — raising an order never
removes a mode from a cell's set — so the minimum need never be formed and no
question is ever asked at an order no cell carries. The "WHY THE INTERSECTION IS
THE MINIMUM RULE" block in `src/dofs.jl` states that argument in full.

Asking the *filter* rather than comparing modes per axis is what makes the rule
family-generic: for `:tensor` the two agree, and for `:trunk` they do not,
because a 3D face mode with bubble degrees `(3, 2)` passes a per-axis test at
order 4 while its trunk degree 5 puts it outside the order-4 trunk set. A family
that adds a mode adds it here and the minimum rule follows.

The `::BasisFamily` default is the per-axis box every tensor-product family
carries; integrated Legendre adds the trunk-degree filter. `test_cell_order.jl`
asserts that this predicate agrees with `local_basis_indices` over a range of
orders, modes and dimensions, that it is monotone in the order for every family
declaring `_supports_cell_order`, and that the intersection form reproduces the
componentwise-minimum form cell for cell — the three contracts that keep the two
layers from drifting.
"""
function _index_admissible(::BasisFamily, order::NTuple{D,Int}, ::Symbol,
                           id::CartesianIndex{D}) where {D}
    return all(d -> 0 <= id.I[d] <= order[d], 1:D)
end

function _index_admissible(::IntegratedLegendre, order::NTuple{D,Int}, mode::Symbol,
                           id::CartesianIndex{D}) where {D}
    all(d -> 0 <= id.I[d] <= order[d], 1:D) || return false
    mode === :tensor && return true
    return _trunk_degree(id) <= order[1]
end

# ── Hot-path tensor-product evaluation ────────────────────────────────────────

# Promote a `PointLike` input (NTuple or SVector) to an `SVector{D,T}` whose
# scalar type is the common floating-point promotion of the inputs.
function _reference_coordinate(xi::PointLike{D}) where {D}
    T = float(promote_type(map(typeof, Tuple(xi))...))
    return SVector{D,T}(xi)
end

# Caller-owned per-axis 1D scratch buffers used by `_fill_factor_tables!`
# and the tensor-product evaluation kernels. One `Vector{T}` per axis,
# length `order[d] + 1`. The assembly workspace pre-allocates one of
# these per level so the hot loop never reallocates.
#
# The per-axis size requirement `order[d] + 1` is the count of 1D modes
# active on a single cell for *every* basis family the package currently
# supports: integrated Legendre carries one endpoint per side plus
# `order − 1` bubbles, and an open-knot B-spline of degree `p` has
# exactly `p + 1` 1D functions non-zero on any single span. So the
# buffer allocation is family-agnostic; only the per-axis evaluator
# (`_fill_factor_tables!`) dispatches.
function _factor_buffers(order::NTuple{D,Int}, ::Type{T}) where {D,T}
    ntuple(d -> Vector{T}(undef, order[d] + 1), D)
end

# Fill caller-owned per-axis 1D tables for `basis` at one reference
# point: `val1d[d][i + 1] = N̂ᵢ(ξ[d])` for the `order[d] + 1` 1D modes
# active on the cell along axis `d`. The 5-arg form fills only values;
# the 6-arg form additionally fills `der1d[d][i + 1] = N̂ᵢ′(ξ[d])`. The
# specific recurrence is basis-family-specific (Legendre three-term for
# integrated Legendre, de Boor for B-splines); both share the same
# `O(D · (order + 1))` per-point cost so the tensor-product evaluators
# downstream stay basis-agnostic.
#
# `cell::CartesianIndex{D}` is the parent cell's mesh-index along each
# axis. Integrated Legendre ignores it (its 1D modes are cell-local in
# the reference frame, identical on every cell); the B-spline family in
# the extension uses it to look up the knot-vector span the parent cell
# corresponds to. Callers always pass the same `parent.cell` the rest
# of the assembly loop uses, so the integrated-Legendre overhead is
# zero and the B-spline overload gets the lookup for free.
#
# The integrated-Legendre overloads below are the in-package default;
# additional basis families plug in via further overloads. The B-spline
# overloads live in `ext/UnfittedBasicBSplineExt.jl`.
function _fill_factor_tables!(::IntegratedLegendre, val1d, order::NTuple{D,Int}, ξ::SVector{D,T},
                              ::CartesianIndex{D}) where {D,T}
    @inbounds for d in 1:D
        x = ξ[d]
        p = order[d]
        vd = val1d[d]
        vd[1] = (one(T) - x) / 2
        if p >= 1
            vd[2] = (one(T) + x) / 2
        end
        if p >= 2
            ℓm2 = one(T)
            ℓm1 = x
            for m in 2:p
                ℓm = ((2m - 1) * x * ℓm1 - (m - 1) * ℓm2) / m
                vd[m + 1] = (ℓm - ℓm2) / sqrt(T(4m - 2))
                ℓm2, ℓm1 = ℓm1, ℓm
            end
        end
    end
    return nothing
end

function _fill_factor_tables!(::IntegratedLegendre, val1d, der1d, order::NTuple{D,Int},
                              ξ::SVector{D,T}, ::CartesianIndex{D}) where {D,T}
    @inbounds for d in 1:D
        x = ξ[d]
        p = order[d]
        vd = val1d[d]
        dd = der1d[d]
        vd[1] = (one(T) - x) / 2
        dd[1] = -one(T) / 2
        if p >= 1
            vd[2] = (one(T) + x) / 2
            dd[2] = one(T) / 2
        end
        if p >= 2
            ℓm2 = one(T)
            ℓm1 = x
            for m in 2:p
                ℓm = ((2m - 1) * x * ℓm1 - (m - 1) * ℓm2) / m
                vd[m + 1] = (ℓm - ℓm2) / sqrt(T(4m - 2))
                dd[m + 1] = sqrt(T(2m - 1) / 2) * ℓm1
                ℓm2, ℓm1 = ℓm1, ℓm
            end
        end
    end
    return nothing
end

# Tensor-product values from precomputed 1D tables. Fills the per-axis
# tables via `_fill_factor_tables!(basis, …)` (the only family-specific
# step), then walks each multi-index `α` in `indices` and assembles the
# product `N_α(ξ) = ∏_d val1d[d][α_d + 1]`. The tensor-product loop is
# basis-agnostic: it only assumes the per-axis tables are indexed by the
# 1D mode number on the cell, which is the canonical convention every
# family in this package follows. The caller owns `val1d`.
#
# The buffer check is `>=`, not `==`. Under a per-cell polynomial order the
# workspace banks stay sized at the level's *nominal* (maximum) order while
# `indices` is the cell's own minimum-rule set, which is shorter on every cell
# below the maximum. Writing into a prefix of the longer buffer is transparent to
# every consumer, because they all bound their loop by `eachindex(raw_dofs)` and
# read `values[a]` positionally — the pairing is with the index list, not with
# the buffer's length. Length-matching with a `view` instead was measured at
# +24–30% on this kernel and is not worth it. What `>=` gives up is the guard
# against a *short* buffer, so `test_cell_order.jl` asserts the surviving check
# still fires.
#
# The 1D factor tables are likewise filled at the nominal order and read only up
# to the cell's own: for integrated Legendre `_fill_factor_tables!` computes mode
# `m` from `ξ` and `m` alone, with no `p` anywhere in the recurrence, so slots
# `1:p_cell+1` hold bit-identical values whether the table was filled at `p_cell`
# or at `p_max`. That identity is what makes the nominal-order banks correct, and
# it is *false* for a family whose 1D modes depend on the degree — de Boor at
# degree p yields p+1 span-specific functions — which is one more reason
# `_supports_cell_order` is false for B-splines.
function _tensor_values!(basis::BasisFamily, values::AbstractVector,
                         indices::AbstractVector{CartesianIndex{D}}, order::NTuple{D,Int},
                         ξ::SVector{D,T}, val1d, cell::CartesianIndex{D}) where {D,T}
    length(values) >= length(indices) ||
        throw(DimensionMismatch("basis value buffer has wrong length"))
    _fill_factor_tables!(basis, val1d, order, ξ, cell)
    @inbounds for (a, id) in pairs(indices)
        values[a] = prod(ntuple(d -> val1d[d][id.I[d] + 1], D))
    end
    return values
end

# Tensor-product values and physically-scaled gradients from precomputed 1D
# tables. For each multi-index `α`:
#
#     N_α(ξ)     = ∏_d val1d[d][α_d + 1]
#     ∂N_α/∂x_d  = der1d[d][α_d + 1] · ∏_{j ≠ d} val1d[j][α_j + 1] · scale[d]
#
# `scale[d]` is the chain-rule factor from reference to physical
# coordinates. For the bare reference gradient pass `scale = (1, …, 1)`;
# for the physical gradient on an axis-aligned cell with edge lengths
# `h_d` pass `scale[d] = 2 / h_d` (the reference cell has edge length 2).
#
# The "product except one" trick is `O(D)` per gradient component for
# `O(D²)` per multi-index — for typical `D ≤ 4` and modest mode counts
# this is comfortably allocation-free with `_product_except` operating
# on `NTuple` scratch.
function _tensor_values_grads!(basis::BasisFamily, values::AbstractVector,
                               gradients::AbstractVector,
                               indices::AbstractVector{CartesianIndex{D}}, order::NTuple{D,Int},
                               ξ::SVector{D,T}, scale::SVector{D,T}, val1d, der1d,
                               cell::CartesianIndex{D}) where {D,T}
    length(values) >= length(indices) ||
        throw(DimensionMismatch("basis value buffer has wrong length"))
    length(gradients) >= length(indices) ||
        throw(DimensionMismatch("basis gradient buffer has wrong length"))
    _fill_factor_tables!(basis, val1d, der1d, order, ξ, cell)
    @inbounds for (a, id) in pairs(indices)
        vfac = ntuple(d -> val1d[d][id.I[d] + 1], D)
        values[a] = prod(vfac)
        gradients[a] = SVector{D,T}(ntuple(k -> der1d[k][id.I[k] + 1] *
                                                _product_except(vfac, k) *
                                                scale[k], D))
    end
    return values, gradients
end

# Product `∏_{d ≠ k} values[d]`. Used by the gradient kernel to compute
# the "value part" of the gradient component along axis `k` without
# recomputing the full product.
function _product_except(values::NTuple{D}, k::Integer) where {D}
    result = one(values[1])
    for d in 1:D
        d == k || (result *= values[d])
    end
    return result
end

# ── Public value and gradient evaluation ──────────────────────────────────────

"""
    basis_values(basis, order, mode, xi, cell) -> Vector

Local basis values at reference point `xi ∈ [−1, 1]ᴰ` on the parent `cell`,
in [`local_basis_indices`](@ref) order. Returns a fresh vector; the hot loops
in `src/assembly.jl` call `_tensor_values!` with reused scratch instead.

`cell::CartesianIndex{D}` is the parent cell's mesh index along each axis.
Families whose 1D modes are cell-local in the reference frame (integrated
Legendre) ignore it; families whose modes are global (the B-spline extension's
knot-vector spans) need it to pick the right span, which is why it is part of
the signature for *every* family rather than an overload some families add.

This `::BasisFamily` method is built on `_tensor_values!` and therefore serves
every family through its `_fill_factor_tables!` hook — there is nothing here
for a family to override.

!!! warning "Pairing with `cell_dofs`"
    The result is in `local_basis_indices(basis, order, mode)` order, which is
    the set the *order* names. On a level whose order varies from cell to cell
    that is not the set the cell generates: the minimum rule removes the
    shared-entity modes a lower-order neighbour cannot match, and the survivors
    are in general not the index set of any order. Pairing this vector
    positionally with `cell_dofs(layout, level, cell)` then attaches values to
    the wrong unknowns. Use `basis_values(level::Level, cell, xi)`, which takes
    the cell rather than an order and reads `cell_basis_indices`.
"""
function basis_values(basis::BasisFamily, order::NTuple{D,Int}, mode::Symbol, xi::PointLike{D},
                      cell::CartesianIndex{D}) where {D}
    indices = local_basis_indices(basis, order, mode)
    ξ = _reference_coordinate(xi)
    T = eltype(ξ)
    values = Vector{T}(undef, length(indices))
    return _tensor_values!(basis, values, indices, order, ξ, _factor_buffers(order, T), cell)
end

"""
    physical_basis_gradients(basis, order, mode, cell_box, xi, cell) -> Vector{SVector{D}}

Local basis gradients with respect to the *physical* coordinate on the
axis-aligned cell `cell_box`, at reference point `xi ∈ [−1, 1]ᴰ`, in
[`local_basis_indices`](@ref) order. The chain rule for an axis-aligned cell
with edge lengths `h_d = edge_lengths(cell_box)[d]` gives

    ∂N_α/∂x_d = (2 / h_d) · ∂N_α/∂ξ_d,

so the physical gradient is the reference gradient scaled per axis by
`scale[d] = 2 / h_d`; pass a box of edge length 2 per axis to recover the bare
reference gradient. Map a physical point through [`physical_to_reference`](@ref)
first if you have `x` rather than `ξ`.

`cell_box` is the cell's *geometry* and `cell` its *mesh index* — the two are
independent arguments because the chain-rule scaling needs the former while a
span-based family needs the latter. As with [`basis_values`](@ref), this
`::BasisFamily` method is built on `_tensor_values_grads!` and serves every
family unchanged.

!!! warning "Pairing with `cell_dofs`"
    The result is in `local_basis_indices(basis, order, mode)` order and carries
    the same caveat [`basis_values`](@ref) does: on a level whose order varies
    from cell to cell the minimum rule has already removed modes from the cell's
    set, so this vector does not line up with `cell_dofs`. Use
    `physical_basis_gradients(level::Level, cell, cell_box, xi)` there.
"""
function physical_basis_gradients(basis::BasisFamily, order::NTuple{D,Int}, mode::Symbol,
                                  cell_box::AxisBox{D,T}, xi::PointLike{D},
                                  cell::CartesianIndex{D}) where {D,T}
    indices = local_basis_indices(basis, order, mode)
    # The cell's scalar type joins the promotion: `scale` is derived from the
    # box, so a `Float32` reference point on a `Float64` cell must still
    # compute — and store — the chain rule at the wider precision.
    R = float(promote_type(map(typeof, Tuple(xi))..., T))
    ξ = SVector{D,R}(xi)
    values = Vector{R}(undef, length(indices))
    gradients = Vector{SVector{D,R}}(undef, length(indices))
    scale = SVector{D,R}(2 ./ edge_lengths(cell_box))
    _tensor_values_grads!(basis, values, gradients, indices, order, ξ, scale,
                          _factor_buffers(order, R), _factor_buffers(order, R), cell)
    return gradients
end

# ── Boundary and facet basis identification ───────────────────────────────────

# True iff the 1D integrated Legendre mode index `mode` carries support on
# the requested cell side: mode 0 lives on the lower endpoint (`ξ = −1`),
# mode 1 on the upper endpoint (`ξ = +1`), and modes m ≥ 2 (bubbles)
# vanish at both endpoints — they never sit on either side.
function boundary_mode(mode::Integer, side::Symbol)
    side === :lower && return mode == 0
    side === :upper && return mode == 1
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# True iff the tensor-product basis function with multi-index `id` carries
# support on the codim-1 face specified by `(axis, side)`. For integrated
# Legendre this reduces to checking the single 1D mode along the requested
# axis.
function is_boundary_basis(::IntegratedLegendre, id::CartesianIndex{D}, axis::Integer,
                           side::Symbol) where {D}
    1 <= axis <= D || throw(ArgumentError("axis out of bounds"))
    return boundary_mode(id.I[axis], side)
end

# Codim-K facet membership for a tensor-product multi-index: the basis function
# sits on the facet specified by `sides` (a list of `(axis, side)` pairs) iff
# its 1D mode along *every* listed axis touches the requested side. Codim 1
# reduces to a single `is_boundary_basis` check; codim K requires every listed
# axis-side match. This `::BasisFamily` default delegates the per-axis test to
# the family's `is_boundary_basis`, so it needs no family-specific override.
function is_facet_basis(basis::BasisFamily, id::CartesianIndex{D},
                        sides::AbstractVector{<:Tuple{Integer,Symbol}}) where {D}
    for (axis, side) in sides
        is_boundary_basis(basis, id, axis, side) || return false
    end
    return true
end

"""
    boundary_basis_indices(basis, order; axis, side, mode=:tensor)
        -> Vector{CartesianIndex{D}}

Multi-indices of the local basis functions whose support touches the codim-1
face at the requested `(axis, side)`. Used by the Dirichlet projection in
`src/dofs.jl` to identify which trial / test modes contribute on each physical
facet. This `::BasisFamily` default filters `local_basis_indices(basis, order,
mode)` through the family's `is_boundary_basis`, so it needs no override.
"""
function boundary_basis_indices(basis::BasisFamily, order::NTuple{D,Int}; axis::Integer,
                                side::Symbol, mode::Symbol=:tensor) where {D}
    return [id
            for id in local_basis_indices(basis, order, mode)
            if is_boundary_basis(basis, id, axis, side)]
end

# ── Public quadrature rule ────────────────────────────────────────────────────

"""
    gauss_rule(basis::IntegratedLegendre, order, [T=Float64]) -> TensorQuadrature

Tensor Gauss–Legendre rule on the reference cube `[−1, 1]ᴰ` sized by
[`recommended_quadrature_order`](@ref) for the given basis and polynomial
order. Useful for one-shot reference-cell quadrature in tests and
examples; the assembly path uses the cached `_tensor_gauss_rule` directly.
"""
function gauss_rule(basis::IntegratedLegendre, order::NTuple{D,Int},
                    ::Type{T}=Float64) where {D,T<:Real}
    _tensor_gauss_rule(recommended_quadrature_order(basis, order), T)
end
