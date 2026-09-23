"""
    UnfittedBasicBSplineExt

Package extension that adds an open-knot, tensor-product B-spline basis
family to Unfitted on top of [`BasicBSpline.jl`](https://github.com/hyrodium/BasicBSpline.jl).
Loading `BasicBSpline` next to `Unfitted` installs the family and the
required basis-interface methods; without `BasicBSpline`, the package
functions exactly as before with the default [`IntegratedLegendre`](@ref)
family.

# Mathematical setting

Per axis `d` the family carries a `BasicBSpline.BSplineSpace{p_d, T, K_d}`
on an open knot vector aligned with the level's mesh cell coordinates:

  - End knots have multiplicity `p_d + 1` (clamped open knots), so the
    first 1D function evaluates to 1 at the lower edge and the last
    function evaluates to 1 at the upper edge — every other 1D function
    vanishes at both endpoints.
  - Interior knots all have multiplicity 1 (max-regularity), giving
    `C^{p_d − 1}` smoothness across every cell boundary — uniform knots,
    no insertions.

Boundary smoothness on the artificial overlay boundary (the mesh-edge faces
and any mask-induced internal faces) is imposed by homogeneous linear
*trace-vanishing constraints* on the eliminated boundary functions, not by
knot multiplicity. The `continuity_order = m` parameter requests `C^m` there:
the trace and its derivatives `k = 0 … m` are constrained to vanish,
eliminating `m + 1` boundary functions per face. `m` may be anything from `0`
(`C⁰`, the default) up to `p_d − 1`, and arbitrary mask geometries are
supported because the constraints need no separability of the mask.

The `D`-dimensional basis is the tensor product of these per-axis 1D
spaces. Adjacent cells of the level share `p_d` of `p_d + 1` 1D
functions per axis — the shared dofs fall out of the `TensorDofKey`
cache in `src/dofs.jl` once each cell-local mode index is mapped to its
global 1D function index via the `_AXIS_BSPLINE` tag.

# What this extension installs

  - `BSplineFamily{D,T,…}` (a `BasisFamily` subtype) plus a deferred
    `_BSplineSpec` returned by the public [`bspline`](@ref) factory.
    The deferred form lives only between `bspline(...)` and the
    [`instantiate_basis`](@ref) hook, which materialises the per-axis
    `BSplineSpace`s once the mesh axes are known. The knot vectors depend
    on the mesh alone — never on the mask, whose faces are handled by the
    constraints described below.
  - The family-specific basis-interface overloads — `basis_name` and
    `is_boundary_basis`. Everything else in the interface is inherited from the
    `::BasisFamily` defaults in `src/basis.jl`: the tensor `local_basis_indices`
    / `local_basis_count` (an open-knot span carries exactly the full
    `∏_d {0, …, p_d}` set), `recommended_quadrature_order`, `is_facet_basis`,
    `boundary_basis_indices`, `basis_values`, `physical_basis_gradients`, and
    the boundary trace. The family declares no mode support of its own either:
    the `::BasisFamily` default is `:tensor` only, which is exactly what this
    family serves, so `_check_basis_mode` rejects `:trunk` for it.
  - Hot-path `_fill_factor_tables!` overloads (values, values +
    derivatives) using `BasicBSpline.bsplinebasisall` per axis after
    mapping the cell-local reference coordinate `ξ ∈ [−1, 1]` to the
    knot-vector parameter.
  - `_tensor_dof_key`, `_key_on_level_side`, and `_overlay_constraints`
    overloads that route through the `_AXIS_BSPLINE` tag.
  - The `_overlay_constraints` overload emits the homogeneous
    trace-vanishing [`LinearConstraint`](@ref)s (orders `k = 0 … m`) on
    every overlay / mask face, so masked B-spline levels of any geometry
    remain eliminable through the existing constraint machinery.
  - A `_coverage_constraints` overload. Covered-mode pruning for this family is
    the dedup and nothing else: a buried function a covering level's span
    already contains exactly is eliminated, because the two are linearly
    dependent and the superposition is otherwise singular. See
    [`bspline`](@ref) under "Nested levels".
  - The fictitious-fold exemption inside that generator: a face whose
    inactive side is a fully-fictitious cell carries no trace to vanish on
    and emits no constraint, while a user-masked face still does. That one
    test is all the family needs to take the `true` default of
    `_supports_physical_domain` and carry an immersed
    [`PhysicalDomain`](@ref), so the finite-cell workflow is open to it.

The hot loops in `src/assembly.jl` and `src/projection.jl` are
unchanged — they dispatch on `level.basis` (via the workspace's
`bases` vector) and the B-spline overloads fire naturally.
"""
module UnfittedBasicBSplineExt

using Unfitted
# Only the names this module uses *unqualified* are imported; every method it
# installs on an Unfitted function is written `Unfitted.f(...)` at its
# definition, which needs no import. The list is therefore an honest measure of
# how far the extension reaches into the package.
using Unfitted: BasisFamily, IntegratedLegendre, Level, CartesianMesh, LevelMask, AxisDofKey,
                TensorDofKey, GeometryTolerance, LinearConstraint, Space, Coverage, cell_box,
                is_active, _AXIS_BSPLINE, _check_basis_mode, _level_side_is_physical,
                _tensor_dof_key, _ClassifyCache, _is_fictitious, _nested_over,
                _reproduced_on_domain, classify_cell
using StaticArrays: SVector
using BasicBSpline: BSplineSpace, BSplineDerivativeSpace, KnotVector, bsplinebasisall, degree, dim

# ── Public factory and deferred spec ──────────────────────────────────────────

"""
    bspline(; continuity_order=0) -> _BSplineSpec

Mark a [`space`](@ref) / [`overlay`](@ref) call as using the open-knot
tensor B-spline basis family. The per-axis polynomial degree is taken
from the `order` keyword on `space` / `overlay` (matching the
integrated Legendre convention), so a typical use looks like

```julia
using Unfitted, BasicBSpline
V = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, basis=bspline())
```

# Keyword arguments

  - `continuity_order` — boundary smoothness `m` on the artificial overlay
    boundary, imposed by homogeneous trace-vanishing constraints (orders
    `k = 0 … m`). Defaults to `m = 0` (`C⁰`); may be any value from `0` up
    to `p_d − 1` on the lowest-degree axis, provided the level is thick
    enough (`cells + p ≥ 2m + 3` per axis). Higher `m` eliminates more
    boundary functions, lowering the active dof count for smooth problems.

# Nested levels

A B-spline overlay of the same degree whose cell boundaries include the
base's reproduces, exactly, every base function buried underneath it — so
the two levels carry the same function twice and the superposed operator is
exactly singular. Nesting is the natural thing to reach for, so the family
answers for it: leaf semantics are unconditional on every level and there is no
keyword to reach for, and they eliminate the duplicate everywhere but the one
fold configuration named below. That costs nothing here — the discrete space is
unchanged, unlike integrated Legendre's shedding of buried high-order modes,
which trades accuracy for dofs. A `:tensor` integrated-Legendre overlay of
order ≥ the base degree reproduces the same functions and is deduped the
same way. The converse also holds and the family answers for it: a *degree-1*
B-spline overlay is the C⁰ multilinear hat space of its own mesh, so it
deduplicates the buried vertex functions of an integrated-Legendre level
below; at degree ≥ 2 it is `C^{p−1}` at a simple interior knot, too smooth to
carry the hat's kink, and nothing is deduplicated.

Under an immersed [`PhysicalDomain`](@ref) the fold takes part on both sides
of the question. A fictitious cell of the *covering* level does not stop it
covering, and a fictitious cell under the buried function's *support* neither
has to be covered nor blocks the dedup: nothing is integrated there, so on Ω
the truncated and untruncated functions coincide. A user-masked cell is the
opposite case and still blocks, because material there is carried by the
coarser level alone. What decides the remaining question is the overlay's own
trace condition: a face of the overlay's box that cuts the interior of the
support carries the function's trace, and the overlay is clamped there exactly
where its boundary cells are active — so the dedup fires iff every overlay
boundary cell along such a face was folded away.

One fold configuration is still left singular, and breaking the nesting is
the only way out of it today. When the overlay's box face falls on a knot of
the level below *inside* a band of cut cells — so the overlay's boundary cells
there are active and clamped — the clamped overlay can reproduce a
*combination* of two buried functions (their difference vanishes at the face,
and the part of it beyond the face is fictitious) while reproducing neither on
its own. A dedup keyed on single buried functions cannot see that, and the
operator keeps one exact null mode per such combination: measured at 8 on a
6×6 degree-2 base over Ω = {x₁ ≤ 0.9} in [-1, 2]² with a quarter-cell overlay
ending at x = 1. Closing it needs a per-face rank repair — of the `p` buried
functions straddling the face knot with the same perpendicular factor, keep the
`m + 1` trace orders and strongly eliminate the rest — which is a change to what
the dedup is allowed to eliminate, not to the burial test. Move the overlay's
face off the cut-cell band, or break the nesting, until it lands.

The only stack that carries both copies is the **unreduced twin**
[`prepare`](@ref)`(problem; prune = false)` builds, and on a nested stack that
twin is exactly singular by construction — which is what it is for. It exists
to be measured against, not to be solved with. If a configuration must keep
every mode in a system that can actually be factorised, break the nesting
instead: a cell count that does not divide the base's (`cells=5` under a base
of 8, say), or a different degree on the overlay, leaves nothing to
deduplicate.

# Why a deferred spec

The actual per-axis `BSplineSpace`s depend on the level's mesh-cell
coordinates, which do not exist yet when the `basis=` value is chosen.
`bspline(...)` therefore returns a deferred [`_BSplineSpec`](@ref); the
concrete [`BSplineFamily`](@ref) is built by the
[`instantiate_basis`](@ref) hook in this extension once the mesh is known.
A mask changes nothing about the knot vectors — mask faces are handled by
the trace-vanishing constraints, not by knot insertion — so a level's
family is rebuilt on a move but is the same family before and after a
mask update. The spec is itself a [`BasisFamily`](@ref) subtype
so the existing `space` / `overlay` code path accepts it without special
casing.
"""
function Unfitted.bspline(; continuity_order::Integer=0)
    continuity_order >= 0 ||
        throw(ArgumentError("bspline: continuity_order must be nonnegative; got $continuity_order"))
    return _BSplineSpec(Int(continuity_order))
end

"""
    _BSplineSpec(continuity_order) <: BasisFamily

Deferred B-spline family specification returned by [`bspline`](@ref).
Subtypes `BasisFamily` so [`space`](@ref) accepts it as a `basis=`
value, but carries no per-axis knot vectors yet — the
[`instantiate_basis`](@ref) hook in this extension materialises a real
[`BSplineFamily`](@ref) from the mesh axes. Once the level is built, the
spec is gone; every hot-path method dispatches on the concrete
`BSplineFamily` type.
"""
struct _BSplineSpec <: BasisFamily
    continuity_order::Int
end

# ── BSplineFamily struct and accessors ────────────────────────────────────────

"""
    BSplineFamily{D,T,S,DS}

Open-knot tensor-product B-spline basis family.

Fields:

  - `spaces::S` — `D`-tuple of `BasicBSpline.BSplineSpace` objects, one
    per axis. Built from the level's mesh axes as clamped open uniform
    knot vectors (interior knot multiplicity 1, `C^{p − 1}`).
  - `derivative_spaces::DS` — `D`-tuple of `BSplineDerivativeSpace{1, _}`
    objects, one per axis, precomputed for the derivative hot path so
    every quadrature point uses a stable type.
  - `axis_coords::NTuple{D,Vector{T}}` — per-axis mesh cell-corner
    coordinates (a copy of `mesh.axes[d]`). Stored on the family so
    the hot kernel can do the `ξ ∈ [−1, 1]` → `t` affine map without
    looking up the level.
  - `cell_to_span::NTuple{D,Vector{Int}}` — per-axis map from a cell index
    `c ∈ 1:cells[d]` to the BasicBSpline span index. With the uniform
    max-regularity knots used here it is the identity
    (`cell_to_span[d][c] = c`); it is kept as an explicit field because it
    also doubles as the per-cell first global 1D function index —
    `bsplinebasisall(P, span, t)` returns the functions with global
    indices `span, span + 1, …, span + p`.
  - `continuity_order::Int` — boundary-smoothness parameter `m`, enforced
    by trace-vanishing constraints (not knot multiplicity); `0 ≤ m ≤ p_d − 1`.

Constructed automatically from a [`_BSplineSpec`](@ref) by the
[`instantiate_basis`](@ref) hook in this extension — users do not
build it directly.
"""
struct BSplineFamily{D,T,S<:Tuple,DS<:Tuple} <: BasisFamily
    spaces::S
    derivative_spaces::DS
    axis_coords::NTuple{D,Vector{T}}
    cell_to_span::NTuple{D,Vector{Int}}
    continuity_order::Int
end

# ── Knot-vector construction ──────────────────────────────────────────────────

# Build a clamped open uniform knot vector for one axis from the mesh's
# per-axis coordinates and the requested degree `p`. End knots have
# multiplicity `p + 1` (clamped open knots); every interior knot has
# multiplicity 1 (max-regularity `C^{p − 1}` continuity across cell
# boundaries). The resulting knot vector aligns each cell with one span
# of the resulting `BSplineSpace`, so the per-cell span index in the
# B-spline parameter coincides with the cell's mesh index.
#
# Boundary conditions on the artificial overlay boundary (mesh-edge
# faces and mask-induced internal faces) are imposed via the
# linear-constraint system in `_overlay_constraints` below — *not*
# via knot multiplicity. This keeps the basis globally `C^{p − 1}`
# inside the active region regardless of mask geometry.
function _open_knot_vector(axis_coords::AbstractVector{T}, p::Integer) where {T<:Real}
    knots = T[]
    sizehint!(knots, length(axis_coords) + 2 * p)
    for _ in 1:(p+1)
        push!(knots, axis_coords[1])
    end
    for i in 2:(length(axis_coords)-1)
        push!(knots, axis_coords[i])
    end
    for _ in 1:(p+1)
        push!(knots, axis_coords[end])
    end
    return KnotVector(knots)
end

# Build the concrete `BSplineFamily` from a deferred `_BSplineSpec`,
# the mesh, and the per-axis polynomial degree. Uniform knots only:
# the cell-to-span map is the identity (`cell_to_span[d][c] = c`),
# and `dim(spaces[d]) = cells[d] + p[d]`.
function _materialize(spec::_BSplineSpec, mesh::CartesianMesh{D,T}, p::NTuple{D,Int}) where {D,T}
    m = spec.continuity_order
    spaces = ntuple(D) do d
        kv = _open_knot_vector(mesh.axes[d], p[d])
        BSplineSpace{p[d]}(kv)
    end
    derivative_spaces = ntuple(d -> BSplineDerivativeSpace{1}(spaces[d]), D)
    axis_coords = ntuple(d -> copy(mesh.axes[d]), D)
    cell_to_span = ntuple(D) do d
        collect(1:mesh.cells[d])  # uniform knots: span = cell index
    end
    return BSplineFamily{D,T,typeof(spaces),typeof(derivative_spaces)}(spaces, derivative_spaces,
                                                                       axis_coords, cell_to_span, m)
end

# ── Basis instantiation: materialise the deferred spec onto a mesh ────────────

# `instantiate_basis` hook for the deferred `_BSplineSpec`. Invoked by
# the core `space` / `overlay` / `moved_space` / mask-mutator paths once
# the level's mesh, order, and mask are known. Validates the request and
# materialises the concrete `BSplineFamily` from the mesh axes; the spec
# itself only ever exists between `bspline(...)` and this call, so every
# downstream method dispatches on the concrete family type.
#
# Arbitrary `LevelMask` geometries are accepted: trace-vanishing at
# mask-induced internal faces is enforced via the linear-constraint
# system in `_overlay_constraints`, which doesn't require any
# separability of the mask.
function Unfitted.instantiate_basis(spec::_BSplineSpec, mesh::CartesianMesh{D,T},
                                    order::NTuple{D,Int}, mode::Symbol,
                                    mask::Union{Nothing,LevelMask{D}}) where {D,T<:Real}
    _check_basis_mode(spec, mode, order)
    all(p -> p >= 1, order) ||
        throw(ArgumentError("bspline order (per-axis polynomial degree) must be ≥ 1"))
    m = spec.continuity_order
    max_m = minimum(p -> p - 1, order)
    m <= max_m || throw(ArgumentError("bspline continuity_order $m exceeds p_d − 1 = $max_m " *
                                      "on the lowest-degree axis; need order > continuity_order"))
    # Linear constraints need enough interior 1D dofs to satisfy:
    # eliminating `m + 1` boundary functions per side leaves room for
    # at least one interior dof, i.e. `dim ≥ 2(m+1) + 1`, equivalently
    # `cells + p ≥ 2m + 3`.
    for d in 1:D
        cells_plus_p = mesh.cells[d] + order[d]
        cells_plus_p >= 2m + 3 || throw(ArgumentError("bspline level too thin to support " *
                                                      "continuity_order $m on axis $d: cells + p = " *
                                                      "$cells_plus_p, need ≥ $(2m + 3). Increase " *
                                                      "cells or decrease continuity_order."))
    end
    return _materialize(spec, mesh, order)
end

# Re-instantiate an already-concrete `BSplineFamily` against a
# (possibly new) mesh. Invoked when an overlay carrying this family is
# moved — where the per-axis knot vectors must be rebuilt for the new cell
# coordinates, since reusing the old family would leave stale `axis_coords`
# / `cell_to_span` — and, for uniformity, when it is re-masked, where the
# mesh is unchanged and the rebuild reproduces an identical family (the
# mask never enters a knot vector). Cell counts and order are unchanged in
# both cases, so the thinness / continuity validation already passed at
# first construction and is not repeated.
function Unfitted.instantiate_basis(family::BSplineFamily, mesh::CartesianMesh{D,T},
                                    order::NTuple{D,Int}, mode::Symbol,
                                    mask::Union{Nothing,LevelMask{D}}) where {D,T<:Real}
    _check_basis_mode(family, mode, order)
    return _materialize(_BSplineSpec(family.continuity_order), mesh, order)
end

# ── Basis-interface methods ───────────────────────────────────────────────────

Unfitted.basis_name(::BSplineFamily) = :bspline
# The deferred spec answers to the same name: it is the family the user asked
# for, and `_check_basis_mode` reports through `basis_name` while validating a
# `space(...; basis=bspline(), mode=…)` call, before the spec is materialised.
Unfitted.basis_name(::_BSplineSpec) = :bspline

# ── Boundary / facet incidence ────────────────────────────────────────────────

# True iff cell-local 1D mode index `i ∈ 0:p` corresponds to a function that
# is non-zero on the requested side of the *level's* mesh box, where the
# clamped end knots of multiplicity `p + 1` leave exactly one non-zero 1D
# function: local index `0` (global index 1) at the lower edge, local index
# `p` (global index `dim`) at the upper edge.
#
# Read it as an outer-boundary predicate, not as a general per-cell support
# test. In-tree it is only ever asked about a boundary cell's outer face —
# `is_facet_basis` filters the Dirichlet boundary trace in `dirichlet.jl`,
# whose facets lie on the physical boundary. At an *interior* cell face the
# answer would be different: with simple interior knots, `p` of the `p + 1`
# local functions are non-zero at the face (only local index `p` vanishes at
# the cell's lower edge, only local index `0` at its upper edge), so a caller
# that needs interior-face support must ask the knot vector, not this.
function Unfitted.is_boundary_basis(family::BSplineFamily, id::CartesianIndex{D}, axis::Integer,
                                    side::Symbol) where {D}
    1 <= axis <= D || throw(ArgumentError("axis out of bounds"))
    p = degree(family.spaces[axis])
    side === :lower && return id.I[axis] == 0
    side === :upper && return id.I[axis] == p
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# ── Hot kernels: per-axis 1D B-spline value and derivative tables ─────────────

# Affine reference-to-parameter map for axis `d` on cell `c`:
# ξ ∈ [−1, 1] ↦ t ∈ [axis_coords[d][c], axis_coords[d][c + 1]]. Returns
# `(t, half_width)` where `half_width = (t_upper − t_lower) / 2`. Stored
# half-width is reused for the derivative-axis chain rule: a B-spline
# basis function `N_i(t)` satisfies `dN/dξ = dN/dt · half_width`, so the
# reference-frame derivative the kernel needs to fill into `der1d[d]`
# already includes that factor.
@inline function _axis_parameter(family::BSplineFamily{D,T}, axis::Integer, cell::Integer,
                                 ξ::Real) where {D,T}
    coords = family.axis_coords[axis]
    t_lower = coords[cell]
    t_upper = coords[cell + 1]
    half_width = (t_upper - t_lower) / 2
    t = t_lower + (ξ + one(ξ)) * half_width
    return t, half_width
end

# Per-axis B-spline values into `val1d[d]` for one quadrature point.
# Calls `bsplinebasisall(spaces[d], span, t)` which returns the `p_d + 1`
# values on `span` as a stack-allocated `SVector{p_d + 1}` — no
# allocation; the SVector entries are then unrolled into the per-axis
# buffer.
function Unfitted._fill_factor_tables!(family::BSplineFamily{D,T,S,DS}, val1d, order::NTuple{D,Int},
                                       ξ::SVector{D,Tξ},
                                       cell::CartesianIndex{D}) where {D,T,S,DS,Tξ}
    @inbounds for d in 1:D
        c = cell.I[d]
        span = family.cell_to_span[d][c]
        t, _ = _axis_parameter(family, d, c, ξ[d])
        values = bsplinebasisall(family.spaces[d], span, t)
        vd = val1d[d]
        for i in 1:(order[d]+1)
            vd[i] = values[i]
        end
    end
    return nothing
end

# Per-axis B-spline values + reference-frame derivatives. Values from
# `bsplinebasisall(spaces[d], span, t)`; raw `t`-derivatives from
# `bsplinebasisall(derivative_spaces[d], span, t)` (a fully unrolled
# `BasicBSpline.BSplineDerivativeSpace{1, …}` call); reference-frame
# `ξ`-derivatives obtained by multiplying with `half_width`. The
# downstream tensor-product kernel multiplies by the assembly-loop
# `scale[d] = 2 / edge_lengths[d]` factor; the two compose to
# `dN/dx = dN/dt · (h_d / 2) · (2 / h_d) = dN/dt`, the physical
# derivative — consistent with the integrated-Legendre chain-rule
# convention.
function Unfitted._fill_factor_tables!(family::BSplineFamily{D,T,S,DS}, val1d, der1d,
                                       order::NTuple{D,Int}, ξ::SVector{D,Tξ},
                                       cell::CartesianIndex{D}) where {D,T,S,DS,Tξ}
    @inbounds for d in 1:D
        c = cell.I[d]
        span = family.cell_to_span[d][c]
        t, half_width = _axis_parameter(family, d, c, ξ[d])
        values = bsplinebasisall(family.spaces[d], span, t)
        derivs = bsplinebasisall(family.derivative_spaces[d], span, t)
        vd = val1d[d]
        dd = der1d[d]
        for i in 1:(order[d]+1)
            vd[i] = values[i]
            dd[i] = derivs[i] * half_width
        end
    end
    return nothing
end

# ── Dof key construction ──────────────────────────────────────────────────────

# Build the per-axis dof key for the B-spline family. A cell-local 1D
# mode `m ∈ 0:p_d` on cell `c` maps to the global 1D function index
# `cell_to_span[d][c] + m`; that index uniquely identifies the function
# across the level, so adjacent cells naturally reuse the dof slot for
# every shared function through the `TensorDofKey` cache in
# `src/dofs.jl`. Wrapped in the `_AXIS_BSPLINE` tag so it lives in
# a disjoint key range from the integrated-Legendre `_AXIS_NODE` /
# `_AXIS_SPAN` keys.
@inline function _bspline_axis_dof_key(family::BSplineFamily, axis::Integer, cell_axis::Integer,
                                       mode::Integer)
    return AxisDofKey(_AXIS_BSPLINE, family.cell_to_span[axis][cell_axis] + mode, 0)
end

function Unfitted._tensor_dof_key(level::Level{D,T,<:BSplineFamily}, cell::CartesianIndex{D},
                                  local_id::CartesianIndex{D}) where {D,T}
    family = level.basis
    axes = ntuple(d -> _bspline_axis_dof_key(family, d, cell.I[d], local_id.I[d]), D)
    return TensorDofKey{D}(level.id, axes)
end

# ── Boundary classification and Dirichlet trace ───────────────────────────────

# True iff the per-axis B-spline factor of `key` anchors at the level's
# lower or upper mesh edge along `axis`. Used by the **physical
# Dirichlet** detection in `dirichlet.jl::_key_on_physical_side`: only
# the function that is interpolatory at the boundary (index 1 at the
# lower edge, index `dim` at the upper edge for an open clamped knot
# vector) carries the Dirichlet value. Higher-derivative constraints
# (`continuity_order ≥ 1`) at the **overlay-artificial** boundary are
# imposed via the linear-constraint system in
# [`_overlay_constraints`](@ref), not through this predicate.
function Unfitted._key_on_level_side(key::TensorDofKey{D}, level::Level{D,T,<:BSplineFamily},
                                     axis::Integer, side::Symbol) where {D,T}
    axis_key = key.axes[axis]
    axis_key.kind == _AXIS_BSPLINE || return false
    family = level.basis
    n = dim(family.spaces[axis])
    side === :lower && return axis_key.index == 1
    side === :upper && return axis_key.index == n
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# Generate the homogeneous linear constraints encoding the
# trace-vanishing condition `∂^k u^(level) / ∂n^k = 0` for
# `k = 0, …, m` on every artificial face of the level's active
# region. Replaces the per-raw `_has_overlay_constraint` predicate
# the integrated-Legendre family uses with a multi-raw constraint
# generator that exploits the tensor-product structure of the
# B-spline basis.
#
# Mathematics. At an artificial face perpendicular to axis `d` at
# physical coordinate `t* = axes[d][cell]`, the trace condition
# factors:
#
#     u(t*, x') = Σ_{α_d, α'} c_α N_{α_d}(t*) N_{α'}(x') = 0
#
# By linear independence of the `N_{α'}(x')` functions on the active
# face region, this reduces to one constraint per perpendicular
# multi-index `α'`:
#
#     Σ_{α_d = 0..p_d} N^{(k)}_{α_d}(t*) · c_{(α_d, α')} = 0
#
# The axis-d coefficients `N^{(k)}_{α_d}(t*)` are computed once per
# face via `bsplinebasisall(BSplineDerivativeSpace{k}, span, t*)` and
# reused across every perpendicular multi-index. The raw lookup for
# each `(α_d, α')` resolves through the standard
# `TensorDofKey → raw` map, with a representative active cell chosen
# on the active side of the face. For arbitrary mask geometries the
# *same* α' may produce *identical* constraints from multiple
# `(perp_cell, perp_local_mode)` iterations — the cascade resolver in
# `_resolve_constraints!` treats subsequent identical emissions as
# trivially satisfied (the constraint collapses to `0 = 0` after
# substitution), so the redundancy is correctness-preserving and the
# performance overhead is small for typical mask layouts.
#
# `continuity_order = 0` (`m = 0`) generates the classical C⁰
# vanishing-trace constraint — the same condition a C⁰ knot insertion at
# the face would impose, but reached through constraints, which puts no
# restriction on the shape of the mask. `m > 0` adds vanishing normal
# derivatives up to order `m`, the trace structure an `H^{m+1}`-conforming
# essential boundary condition needs; for the B-spline/FCM setting see
#
#     M. Ruess, D. Schillinger, Y. Bazilevs, V. Varduhn, E. Rank,
#     *Weakly Enforced Essential Boundary Conditions for NURBS-embedded and
#     trimmed NURBS geometries on the basis of the Finite Cell Method*,
#     Int. J. Numer. Methods Engng. 95 (2013) 811–846.
function Unfitted._overlay_constraints(level::Level{D,T,<:BSplineFamily}, V::Space{D,T},
                                       tol::GeometryTolerance{T},
                                       raw_by_key::AbstractDict{TensorDofKey{D},Int},
                                       level_keys::AbstractVector{Pair{TensorDofKey{D},Int}},
                                       classify_cache::Unfitted._ClassifyCache{D,T}) where {D,T}
    # `level_keys` (this level's pre-bucketed raws) is unused: the
    # B-spline constraint generator walks the mesh face/perp grids and
    # looks up raws through `raw_by_key` directly. `classify_cache` goes to
    # `_active_cell_at_face`, which reads it to tell a fictitious fold face
    # from a user-mask face; with `V.physical === nothing` it is never touched.
    family = level.basis
    m = family.continuity_order
    n_cells = level.mesh.cells
    constraints = LinearConstraint{T}[]

    for d in 1:D
        p_d = degree(family.spaces[d])
        P_d = family.spaces[d]
        # Perp cell and perp local-mode grids are invariant per axis d.
        perp_cells = CartesianIndices(ntuple(e -> e == d ? (1:1) : (1:n_cells[e]), D))
        perp_modes = CartesianIndices(ntuple(e -> e == d ? (0:0) : (0:degree(family.spaces[e])), D))

        for j in 0:n_cells[d]
            # Skip physical-boundary edges — the physical-Dirichlet
            # branch in `dof_layout` handles them, not the overlay
            # constraint.
            j == 0 && _level_side_is_physical(level, V.domain, d, :lower, tol) && continue
            j == n_cells[d] && _level_side_is_physical(level, V.domain, d, :upper, tol) && continue

            t_star = j == 0 ? family.axis_coords[d][1] :
                     j == n_cells[d] ? family.axis_coords[d][end] : family.axis_coords[d][j + 1]

            # Two possible spans (= active-cell axis-d index) per face:
            # `j` if the lower-cell is active, `j+1` if the upper is.
            # Boundary faces have a single span (lower_span == upper_span).
            # Precompute the `(p_d + 1)`-vectors of axis-d trace values
            # per derivative order per span. One `bsplinebasisall` call
            # per (k, span) — at most `2(m + 1)` calls per face — instead
            # of one per (perp_cell, perp_local, k) iteration.
            lower_span = j == 0 ? 1 : j
            upper_span = j == n_cells[d] ? n_cells[d] : j + 1
            coeffs_lower, coeffs_upper = _trace_coeffs(P_d, lower_span, upper_span, t_star, Val(m))

            for perp_cell in perp_cells
                located = _active_cell_at_face(level, d, j, perp_cell, n_cells, Val(D), V.physical,
                                               classify_cache)
                located === nothing && continue
                active_cell, span = located
                coeffs_for_span = span == lower_span ? coeffs_lower : coeffs_upper

                for perp_local in perp_modes
                    for kp1 in 1:(m+1)
                        coeffs = coeffs_for_span[kp1]
                        raws = Int[]
                        cs = T[]
                        sizehint!(raws, p_d + 1)
                        sizehint!(cs, p_d + 1)
                        for i in 0:p_d
                            local_id = CartesianIndex(ntuple(e -> e == d ? i : perp_local[e], D))
                            key = _tensor_dof_key(level, active_cell, local_id)
                            raw = get(raw_by_key, key, 0)
                            raw == 0 && continue
                            push!(raws, raw)
                            push!(cs, T(coeffs[i + 1]))
                        end
                        isempty(raws) || push!(constraints, LinearConstraint{T}(raws, cs))
                    end
                end
            end
        end
    end
    return constraints
end

# Per-face trace coefficients (axis-d B-spline values and derivatives at
# `t_star`) for both potential spans, indexed by derivative order `k+1`.
# Returns `(coeffs_lower, coeffs_upper)` where each is a tuple of length
# `m+1` holding the `(p_d + 1)`-element `SVector` of trace values. When
# the two spans coincide (boundary faces), reuses the lower-span vectors
# directly to avoid the second `bsplinebasisall` call.
function _trace_coeffs(P_d, lower_span::Int, upper_span::Int, t_star, ::Val{m}) where {m}
    coeffs_lower = ntuple(kp1 -> _trace_at(P_d, lower_span, t_star, kp1 - 1), m + 1)
    coeffs_upper = lower_span == upper_span ? coeffs_lower :
                   ntuple(kp1 -> _trace_at(P_d, upper_span, t_star, kp1 - 1), m + 1)
    return coeffs_lower, coeffs_upper
end

@inline _trace_at(P_d, span, t_star, k::Int) = k == 0 ? bsplinebasisall(P_d, span, t_star) :
                                               bsplinebasisall(BSplineDerivativeSpace{k}(P_d), span,
                                                               t_star)

# Returns `(active_cell::CartesianIndex{D}, span::Int)` if the face at
# axis `d`, position `j`, perpendicular cell `perp_cell` is artificial
# (i.e., contributes a trace constraint), `nothing` otherwise. For mesh
# edges, "artificial" reduces to "the boundary cell is active"; for
# internal faces, it's "the two adjacent cells differ in activity".
#
# With one exemption, and it is what lets this family carry an immersed
# `PhysicalDomain`. A `LevelMask` merges two deactivations that mean opposite
# things: a *user-masked* cell holds material only the coarser level represents,
# so the face between it and the active side is a real artificial boundary and
# the trace must vanish there; a *fictitious* cell holds no material at all, so
# there is no trace to vanish on, and constraining it would delete the cut cell's
# boundary modes — the very over-constraint the FCM solution must not suffer.
# `_internal_face_is_physical` in `src/dofs.jl` draws the same line for the
# integrated-Legendre family. With `physical === nothing` there is no fold and
# `cache` is never touched.
function _active_cell_at_face(level::Level{D}, d::Int, j::Int, perp_cell::CartesianIndex{D},
                              n_cells::NTuple{D,Int}, ::Val{D}, physical,
                              cache::_ClassifyCache{D}) where {D}
    if j == 0 || j == n_cells[d]
        axis_d_idx = j == 0 ? 1 : n_cells[d]
        cell = CartesianIndex(ntuple(e -> e == d ? axis_d_idx : perp_cell[e], D))
        return is_active(level.mask, cell) ? (cell, axis_d_idx) : nothing
    end
    cell_j = CartesianIndex(ntuple(e -> e == d ? j : perp_cell[e], D))
    cell_jp1 = CartesianIndex(ntuple(e -> e == d ? j + 1 : perp_cell[e], D))
    a_j = is_active(level.mask, cell_j)
    a_jp1 = is_active(level.mask, cell_jp1)
    a_j == a_jp1 && return nothing
    inactive = a_j ? cell_jp1 : cell_j
    if physical !== nothing &&
       classify_cell(physical, cell_box(level.mesh, inactive), cache) === :fictitious
        return nothing
    end
    return a_j ? (cell_j, j) : (cell_jp1, j + 1)
end

# ── Covered-mode pruning: dedup of what a covering level already contains ──────────

# The cells this level's axis-`d` 1D function with global index `i` is supported
# on. Cell `c` carries the global indices `c … c + p` (uniform knots,
# `cell_to_span` the identity), so function `i` lives on
# `max(1, i − p) … min(ncells, i)` — up to `p + 1` cells. The core's
# `_axis_incidence` cannot answer this: it is written for the integrated-Legendre
# node / span keys and reports a single cell for any other tag.
@inline _axis_support(i::Integer, p::Integer, ncells::Integer) = max(1, i-p):min(ncells, i)

function _support_cells(key::TensorDofKey{D}, family::BSplineFamily{D}, n::NTuple{D,Int}) where {D}
    return CartesianIndices(ntuple(d -> _axis_support(key.axes[d].index, degree(family.spaces[d]),
                                                      n[d]), D))
end

# True iff level `k`, standing above a B-spline level of per-axis degree `p`,
# reproduces that level's buried functions exactly — given the meshes nest, which
# the caller tests separately with `_nested_over`.
#
# A buried function's support lies inside `k`'s active region, and it vanishes
# there to order `p − 1` (its own end knots are simple), so `k`'s trace conditions
# — orders `0 … m ≤ p − 1` — never exclude it. What is left is a question about
# the two spans:
#
#   * a B-spline `k` of the *same* degree. Interior knots are simple on both
#     sides, so `k` is `C^{p−1}` exactly where the buried function kinks and,
#     under nesting, that function is one of `k`'s own splines. A *higher* degree
#     does not help — at a shared simple knot `k` would be `C^p`, too smooth to
#     carry the kink — and a lower one cannot carry the polynomial degree.
#   * an integrated-Legendre `k` in `:tensor` mode of order ≥ `p`. Its span is
#     every `C⁰` tensor polynomial of that degree on its cells, which contains a
#     `C^{p−1}` spline of degree `p`. `:trunk` drops the mixed high-order terms,
#     so it does not.
#
# Every other family answers `false`. That is the safe direction: skipping a
# legitimate dedup leaves a rank-deficient operator, whereas deduping a function
# nothing reproduces would delete part of the space.
function _reproduces(k::Level{D}, p::NTuple{D,Int}) where {D}
    family = k.basis
    family isa BSplineFamily && return all(d -> degree(family.spaces[d]) == p[d], 1:D)
    family isa IntegratedLegendre || return false
    k.mode === :tensor || return false
    # `nominal_order(k)` is the level's per-axis maximum, which is not enough when `k` carries
    # a per-cell order: a single cell below `p` anywhere under the buried function's
    # support cannot reproduce it. Requiring every palette entry to clear `p` is the
    # conservative reading — it can only skip a legitimate dedup, never delete a
    # function nothing reproduces.
    return all(o -> all(d -> o[d] >= p[d], 1:D), k.orders.palette)
end

# The reverse direction of `_reproduces`: whether *this* family, standing above an
# integrated-Legendre level, spans that level's C⁰ hats and may therefore deduplicate
# them. At degree 1 the open knot vector is the Q1 hat basis outright — simple interior
# knots give exactly C⁰, the clamped ends give the two boundary hats — so on a nested
# mesh the buried hat is one of this level's own functions. Any axis at degree ≥ 2 is
# C^(p−1) with p − 1 ≥ 1 at a simple interior knot, too smooth to carry the kink, and
# the whole tensor product goes with it. Answering per axis rather than per level is
# what makes this a mixed-family question: a degree-1 B-spline cover over an
# integrated-Legendre base is the C⁰ hat space and was previously refused the dedup by a
# family type test, leaving the stack exactly singular.
Unfitted._spans_hats(f::BSplineFamily{D}) where {D} = all(d -> degree(f.spaces[d]) == 1, 1:D)

# `_coverage_constraints` for a B-spline level — the dedup half only, and it is
# the whole of covered-mode pruning for this family.
#
# Integrated Legendre splits into a linear skeleton plus bubble modes, so it can
# shed the bubbles of a covered cell on their own and trade accuracy for dofs. A
# B-spline basis has no such split: its only reducible mode is one the covering
# level reproduces *exactly*, which leaves the discrete space unchanged and is
# therefore free. What it buys is not dof count but well-posedness — the
# reproduced function and its copy above are linearly dependent, so leaving both
# active makes the superposed operator exactly singular. The elimination is
# therefore unconditional here as everywhere: the only stack that keeps both
# copies is the unreduced twin `prepare(problem; prune = false)` builds, which
# opts out of the repair along with the reduction, exactly as it does for
# integrated Legendre.
function Unfitted._coverage_constraints(level::Level{D,T,<:BSplineFamily}, V::Space{D,T},
                                        coverage::Coverage{D}, tol::GeometryTolerance{T},
                                        level_keys::AbstractVector{Pair{TensorDofKey{D},Int}},
                                        classify_cache::_ClassifyCache{D,T}) where {D,T}
    out = Tuple{LinearConstraint{T},Symbol}[]
    cov = coverage.covered[level.id]
    any(cov) || return out
    family = level.basis
    n = level.mesh.cells
    p = ntuple(d -> degree(family.spaces[d]), D)
    above = [k
             for k in V.levels
             if k.id > level.id && _reproduces(k, p) && _nested_over(level, k, tol)]
    isempty(above) && return out
    for (key, raw) in level_keys
        cells = _support_cells(key, family, n)
        # Buried, and buried as a whole function — but the two halves of "whole" are
        # not the same test, and conflating them is what left a folded stack singular.
        # A *user-masked* support cell blocks: what a masked level assembles is the
        # function truncated to its own active cells, which is not the function `k`
        # reproduces. A *fictitious* one does not: nothing is integrated there, so the
        # truncation is invisible on Ω. It must also bypass `cov`, which is computed
        # only on the active cells dilated by one (`_coverage_cells`) — a support cell
        # two cells past the active front holds the `false` default, not a verdict.
        # What is left of the question `_reproduced_on_domain` settles, including
        # whether a face of `k`'s box cuts the support and carries a trace condition
        # there.
        all(ci -> is_active(level.mask, ci) ? cov[ci] :
                  _is_fictitious(level, ci, V.physical, classify_cache), cells) || continue
        any(k -> _reproduced_on_domain(level, k, cells, V.physical, tol, classify_cache), above) ||
            continue
        push!(out, (LinearConstraint{T}([raw], [one(T)]), :dedup))
    end
    return out
end

end # module UnfittedBasicBSplineExt
