# Physical Dirichlet boundary conditions: the user-facing spec types,
# the per-key boundary-face detection used by `dof_layout` to populate
# the `physical_dirichlet` matrix, the codim-K facet integration
# machinery (regions, tensor Gauss quadrature, basis traces), and the
# L² boundary projection that turns nonzero Dirichlet data into the
# `constrained_values` stored on the layout.
#
# Loads after `dofs.jl` so the dof keys and `DofLayout` it operates on
# already exist. The `dof_layout` constructor in `dofs.jl` forward-
# references `_has_physical_dirichlet` and `_project_dirichlet_values!`
# defined here — they are resolved at call time (when the user runs
# `prepare(problem)`), by which point every source file is loaded.

# ── Dirichlet condition spec ──────────────────────────────────────────────────

"""
    BoundarySelector(selector, sides)

Specification of a facet on the physical boundary `∂Ω` used by Dirichlet
conditions and the dof-layer's constraint detection. Two `selector`
shapes are supported:

  - `:all` — the whole physical boundary `∂Ω`, encoded as the union of
    all codim-1 faces. `sides` is empty in this case.
  - `:sides` — the intersection of one or more `(axis, side)` pairs
    listed in `sides`. A single pair picks a codim-1 face; two pairs in
    2D pick a corner; two pairs in 3D pick an edge; three pairs in 3D
    pick a vertex. Each axis may appear at most once and `side` must be
    `:lower` or `:upper`.

Use the public [`boundary`](@ref) constructor instead of building this
directly.
"""
struct BoundarySelector
    selector::Symbol
    sides::Vector{Tuple{Int,Symbol}}
end

# Value semantics for `BoundarySelector`. A selector is a *description* of a
# facet, not an object with an identity: `boundary(axis=1, side=:upper)` written
# twice names the same face both times, and every consumer that reasons about
# selectors — `_selector_matches`, `_facet_regions_for_selector`,
# `update_dirichlet!`'s structural check — already compares them by value.
#
# The default `==` on a struct carrying a `Vector` falls back to `===`, and the
# default `hash` to `objectid`, so without these two methods a selector is
# identity-keyed in every `Dict`: two value-equal selectors are distinct keys,
# a freshly built selector can never hit a cache populated at `prepare`, and the
# identity-based and value-based halves of region resolution disagree about
# which selectors are the same one. `sides` is hashed by value (Julia hashes a
# `Vector` element-wise), matching the element-wise `==` below; the elements are
# isbits tuples, so both are exact.
Base.:(==)(a::BoundarySelector, b::BoundarySelector) = a.selector === b.selector &&
                                                       a.sides == b.sides
Base.hash(s::BoundarySelector, h::UInt) = hash(s.sides, hash(s.selector, hash(:BoundarySelector, h)))

"""
    DirichletCondition(value, boundary, field, component)

One physical Dirichlet condition. Built by [`dirichlet`](@ref):

  - `value` — Dirichlet datum. Either a scalar / indexable value applied
    everywhere on the selected facet, or a callback `value(x)` evaluated
    at physical coordinates.
  - `boundary::BoundarySelector` — the facet to constrain.
  - `field::Union{Nothing,Symbol}` — name of the field this condition
    applies to. `nothing` applies it to the only field of a single-field
    problem; multi-field problems must name a field explicitly.
  - `component::Union{Nothing,Int}` — component index when constraining
    only one component of a vector field (e.g. a roller boundary that
    pins the normal component); `nothing` constrains every component.
"""
struct DirichletCondition{F}
    value::F
    boundary::BoundarySelector
    field::Union{Nothing,Symbol}
    component::Union{Nothing,Int}
end

# ── Physical-boundary key detection ───────────────────────────────────────────

# True iff the dof key sits on the requested physical-domain face.
# Composed from two family-aware primitives so it works for any basis
# family that implements [`_key_on_level_side`](@ref):
#
#   1. The key's per-axis factor must anchor on the level's mesh edge
#      at (`axis`, `side`). Family-specific check.
#   2. The level's mesh edge at (`axis`, `side`) must coincide with the
#      physical-domain edge up to `tol.contain`. Pure geometry check.
#
# Equivalent to the previous coordinate-matching formulation for the
# integrated Legendre family (mesh-axis-node coordinate equals
# `mesh.axes[axis][1]` or `[end]`, and those coincide with the domain
# edge iff `_level_side_is_physical`); the new formulation generalizes
# to any family whose boundary-mode classification lives in
# `_key_on_level_side`.
function _key_on_physical_side(key::TensorDofKey{D}, level::Level{D,T}, domain::AxisBox{D,T},
                               axis::Integer, side::Symbol, tol::GeometryTolerance{T}) where {D,T}
    _key_on_level_side(key, level, axis, side) || return false
    return _level_side_is_physical(level, domain, axis, side, tol)
end

# True iff the dof key sits on *any* of the 2D codim-1 physical faces.
# Used to decide whether a key matches a `boundary(:all)` selector.
function _key_on_any_physical_side(key::TensorDofKey{D}, level::Level{D,T}, domain::AxisBox{D,T},
                                   tol::GeometryTolerance{T}) where {D,T}
    return any(_key_on_physical_side(key, level, domain, axis, side, tol) for axis in 1:D
               for side in (:lower, :upper))
end

# True iff the dof key matches the facet picked out by a `BoundarySelector`.
# For `:all` the predicate is "on any physical face"; for `:sides` it is
# "on every listed (axis, side) face" (i.e. the intersection of the listed
# codim-1 faces).
function _selector_matches(key::TensorDofKey{D}, level::Level{D,T}, domain::AxisBox{D,T},
                           selector::BoundarySelector, tol::GeometryTolerance{T}) where {D,T}
    selector.selector === :all && return _key_on_any_physical_side(key, level, domain, tol)

    if selector.selector === :sides
        for (axis, side) in selector.sides
            1 <= axis <= D ||
                throw(ArgumentError("boundary axis $axis is out of bounds for dimension $D"))
            _key_on_physical_side(key, level, domain, axis, side, tol) || return false
        end
        return true
    end

    throw(ArgumentError("unsupported boundary selector $(selector.selector)"))
end

# True iff any of `conditions` constrains component `component` of `key`.
# Used during `dof_layout` to fill the `physical_dirichlet` matrix.
function _has_physical_dirichlet(key::TensorDofKey{D}, level::Level{D,T}, domain::AxisBox{D,T},
                                 conditions, tol::GeometryTolerance{T},
                                 component::Integer) where {D,T}
    return any(conditions) do condition
        (condition.component === nothing || condition.component == component) &&
            _selector_matches(key, level, domain, condition.boundary, tol)
    end
end

# ── Boundary facet integration ────────────────────────────────────────────────

# Decompose a `BoundarySelector` into the list of facets it covers. For
# `:all` this is one codim-1 facet per physical face (their union is
# `∂Ω`); for `:sides` it is the single facet identified by the listed
# `(axis, side)` constraints. Used by `_project_dirichlet_values!` to
# iterate over the support of each Dirichlet condition.
function _facets(selector::BoundarySelector, ::Val{D}) where {D}
    if selector.selector === :all
        return [Tuple{Int,Symbol}[(axis, side)] for axis in 1:D for side in (:lower, :upper)]
    elseif selector.selector === :sides
        for (axis, _) in selector.sides
            1 <= axis <= D ||
                throw(ArgumentError("boundary axis $axis is out of bounds for dimension $D"))
        end
        return [selector.sides]
    end

    throw(ArgumentError("unsupported boundary selector $(selector.selector)"))
end

# Component extraction from a Dirichlet value: scalars apply to every
# component, anything indexable supplies one entry per component.
function _component_value(value, component::Integer)
    value isa Number && return value
    return value[component]
end

# Evaluate a `DirichletCondition` at physical coordinate `x` for the
# given component. Handles both constant and callback `value` fields.
function _condition_value(condition::DirichletCondition, x, component::Integer=1)
    value = condition.value isa Function ? condition.value(x) : condition.value
    return _component_value(value, component)
end

# True iff `cell` sits on the requested mesh-boundary side of `level`
# along `axis`. Used to filter the parent cells of a boundary facet
# region to only the ones whose own face along the constrained axis
# coincides with the facet.
function _cell_on_side(level::Level, cell::CartesianIndex, axis::Integer, side::Symbol)
    side === :lower && return cell.I[axis] == 1
    side === :upper && return cell.I[axis] == level.mesh.cells[axis]
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

"""
    FacetParent{D,T}(level, cell, parent_box)

Reference to one parent cell touching a boundary facet. `level` and
`cell` identify the cell; `parent_box` is the cell's physical box,
cached at region construction so the consumer's per-quadrature-point
inner loop never re-derives it from `level.mesh`.

The facet's side specification (the `(axis, side)` constraints picking
out which basis modes are facet-incident) lives on the enclosing
[`FacetRegion`](@ref) — every parent in a region shares the same
sides, so storing them per-parent would be redundant.
"""
struct FacetParent{D,T<:Real}
    level::Int
    cell::CartesianIndex{D}
    parent_box::AxisBox{D,T}
end

"""
    FacetRegion{D,T}(sides, parents, points, weights, normal)

One admissible integration region on a codim-`K` facet of the physical
domain (where `K = length(sides)`). The region carries the precomputed
physical-frame Gauss rule, the constant covering-parent list, and the
facet's outward normal — all consumers (the Dirichlet projection,
[`boundary_integral`](@ref), and the assembly path) iterate this
precomputed data directly rather than re-deriving Jacobians and
reference-to-physical maps per quadrature point.

Fields:

  - `sides::Vector{Tuple{Int,Symbol}}` — the codim-`K` facet identifier
    (one `(axis, side)` pair per constrained axis). For codim-1 facets
    this is a single pair; for codim-`K > 1` facets the intersection
    of `K` codim-1 faces. Used by the basis-trace machinery
    ([`boundary_trace_data`](@ref) and `is_facet_basis`) to filter
    facet-incident basis modes.
  - `parents::Vector{FacetParent{D,T}}` — every level cell whose own
    facet on `sides` coincides with this region. The parent set is
    constant across the region's quadrature points (the contract that
    makes "evaluate basis values once per parent" amortizable).
  - `points::Vector{SVector{D,T}}` — physical-frame quadrature
    coordinates on the facet.
  - `weights::Vector{T}` — physical-frame quadrature weights, already
    multiplied by the facet's reference-to-physical Jacobian
    (`vol(facet) / 2^(D-K)`).
  - `normal::SVector{D,T}` — outward normal to the facet. For a codim-1
    face this is `±eₐ` with the sign set by `side`. For codim-`K > 1`
    facets it is the unit average of the constrained-face outward
    normals — well-defined and finite, though not geometrically as
    sharp as the codim-1 case.
"""
struct FacetRegion{D,T<:Real}
    sides::Vector{Tuple{Int,Symbol}}
    parents::Vector{FacetParent{D,T}}
    points::Vector{SVector{D,T}}
    weights::Vector{T}
    normal::SVector{D,T}
end

# Axes *not* constrained by `sides`. Each codim-1 face has `D − 1` free
# axes; a codim-D vertex has zero free axes. Used to size the
# free-axis tensor quadrature and to enumerate which axes carry physical
# extent on the boundary region.
function _free_axes(sides::AbstractVector{<:Tuple{Integer,Symbol}}, ::Val{D}) where {D}
    constrained = Set{Int}()
    for (axis, _) in sides
        push!(constrained, Int(axis))
    end
    return [d for d in 1:D if !(d in constrained)]
end

# Fixed coordinate value for axis `d` of a facet specified by `sides`.
# Returns the side's coordinate (`domain.lower[axis]` or
# `domain.upper[axis]`) when `d` is constrained by some entry of `sides`
# and `zero(T)` otherwise — the caller overwrites the free-axis entries
# with actual interval bounds.
function _facet_fixed_coord(domain::AxisBox{D,T}, sides::AbstractVector{<:Tuple{Integer,Symbol}},
                            d::Integer) where {D,T}
    for (axis, side) in sides
        axis == d || continue
        return side === :lower ? domain.lower[axis] : domain.upper[axis]
    end
    return zero(T)
end

# Per-free-axis Gauss-Legendre point counts for a boundary sub-rectangle.
# Takes the per-axis maximum recommended quadrature order over the
# covering parents so the trace integration is exact for every covering
# parent simultaneously.
function _facet_quadrature_counts(V::Space{D,T}, parents::Vector{FacetParent{D,T}},
                                  free_axes::Vector{Int}) where {D,T}
    return [maximum(parents) do parent
                level = _level_by_id(V, parent.level)
                recommended_quadrature_order(level.basis, level.order)[d]
            end for d in free_axes]
end

# Tensor Gauss quadrature on the free axes of a facet sub-rectangle. The
# returned samples are `(reference_eta_on_free_axes, weight)` pairs in
# the facet's reference frame `[−1, 1]^(D−K)`. The codim-D vertex case
# (`isempty(counts)`) collapses to a single sample with unit weight, an
# empty reference vector, and the boundary "integration" reduces to a
# point evaluation.
function _facet_reference_quadrature(counts::Vector{Int}, ::Type{T}) where {T}
    if isempty(counts)
        return [(T[], one(T))]
    end

    axes_rules = map(counts) do n
        points, weights = gausslegendre(n)
        (T.(points), T.(weights))
    end
    samples = Tuple{Vector{T},T}[]

    for index in CartesianIndices(Tuple(counts))
        eta = T[axes_rules[j][1][index.I[j]] for j in eachindex(counts)]
        weight = prod(axes_rules[j][2][index.I[j]] for j in eachindex(counts))
        push!(samples, (eta, weight))
    end

    return samples
end

# Outward unit normal of a facet identified by `sides`. For codim-1
# facets this is `±eₐ` with the sign set by the single `side`. For
# codim-`K > 1` facets the constrained-face outward normals are summed
# and normalised; well-defined for every `K ≤ D` and continuous in the
# face geometry, though not geometrically as sharp as the codim-1 case
# (a codim-2 corner in 2D returns the diagonal-out direction).
function _facet_outward_normal(sides::AbstractVector{<:Tuple{Integer,Symbol}}, ::Val{D},
                               ::Type{T}) where {D,T}
    accumulator = MVector{D,T}(ntuple(_ -> zero(T), D))
    for (axis, side) in sides
        sign = side === :lower ? -one(T) :
               side === :upper ? one(T) :
               throw(ArgumentError("boundary side must be :lower or :upper"))
        accumulator[axis] += sign
    end
    n = SVector{D,T}(accumulator)
    norm_n = sqrt(sum(x -> x * x, n))
    return iszero(norm_n) ? n : n / norm_n
end

# Build the admissible boundary regions on the facet identified by
# `sides`. The algorithm mirrors the volume-region construction in
# `intersections.jl`, restricted to the facet:
#
#   1. Identify the levels whose own mesh face on `sides` coincides with
#      the physical-domain face — only these levels contribute boundary
#      trace dofs on this facet.
#   2. Per free axis, collect every touching level's boundary
#      coordinates along that axis, merge them, and form non-degenerate
#      intervals.
#   3. Take the Cartesian product of the intervals to enumerate the
#      candidate sub-rectangles. Fix the constrained-axis coordinates to
#      the facet's coordinates.
#   4. For each candidate, find the touching levels' parent cells whose
#      own face on `sides` contains the candidate's midpoint. Drop
#      candidates with no parents.
#   5. Precompute the per-sub-rectangle Gauss rule and bake the
#      `vol / 2^(D-K)` Jacobian into the weights. The returned
#      `FacetRegion` is ready for direct consumption — every Q-point is
#      in physical coordinates with the physical-frame weight already
#      applied.
function _boundary_facet_regions(V::Space{D,T}, sides::Vector{Tuple{Int,Symbol}},
                                 tol::GeometryTolerance{T}) where {D,T}
    touching_levels = [level
                       for level in V.levels
                       if all(_level_side_is_physical(level, V.domain, axis, side, tol)
                              for (axis, side) in sides)]
    isempty(touching_levels) && return FacetRegion{D,T}[]

    free_axes = _free_axes(sides, Val(D))
    normal = _facet_outward_normal(sides, Val(D), T)

    # Per-axis intervals along the free axes, mirroring
    # `_axis_intervals` from intersections.jl.
    intervals = map(free_axes) do d
        _intervals_from_coordinates(_merged_axis_coordinates(touching_levels, d, tol), tol)
    end

    ranges = Tuple(length(intervals[i]) for i in eachindex(intervals))
    any(==(0), ranges) && return FacetRegion{D,T}[]
    regions = FacetRegion{D,T}[]

    for index in CartesianIndices(ranges)
        # Build the sub-rectangle's corner coordinates: fixed coords on
        # constrained axes, interval endpoints on free axes.
        lower = MVector{D,T}(ntuple(d -> _facet_fixed_coord(V.domain, sides, d), D))
        upper = MVector{D,T}(ntuple(d -> _facet_fixed_coord(V.domain, sides, d), D))
        for (j, d) in pairs(free_axes)
            lower[d] = intervals[j][index.I[j]][1]
            upper[d] = intervals[j][index.I[j]][2]
        end
        lower_s = SVector{D,T}(lower)
        upper_s = SVector{D,T}(upper)
        midpoint = SVector{D,T}(ntuple(d -> (lower_s[d] + upper_s[d]) / 2, D))
        parents = FacetParent{D,T}[]

        # Find every touching level whose own face on `sides` contains
        # this candidate's midpoint.
        for level in touching_levels
            cell = locate_cell(level.mesh, midpoint; tol)
            cell === nothing && continue
            all(_cell_on_side(level, cell, axis, side) for (axis, side) in sides) || continue
            is_active(level.mask, cell) || continue
            push!(parents, FacetParent{D,T}(level.id, cell, cell_box(level.mesh, cell)))
        end

        isempty(parents) && continue

        # Precompute the physical Q-points and physical-weighted
        # samples. The Jacobian for a codim-K facet sub-rectangle is
        # `vol_free / 2^(D-K)`, matching the volume convention
        # `vol / 2^D` from `reference_to_physical`.
        counts = _facet_quadrature_counts(V, parents, free_axes)
        jacobian = if isempty(free_axes)
            one(T)
        else
            prod(upper_s[d] - lower_s[d] for d in free_axes) / convert(T, 2^length(free_axes))
        end

        physical_points = SVector{D,T}[]
        physical_weights = T[]
        for (eta, weight) in _facet_reference_quadrature(counts, T)
            x = MVector{D,T}(lower_s)
            for (j, d) in pairs(free_axes)
                axis_mid = (lower_s[d] + upper_s[d]) / 2
                axis_half = (upper_s[d] - lower_s[d]) / 2
                x[d] = axis_mid + axis_half * eta[j]
            end
            push!(physical_points, SVector{D,T}(x))
            push!(physical_weights, weight * jacobian)
        end

        push!(regions, FacetRegion{D,T}(sides, parents, physical_points, physical_weights, normal))
    end

    return regions
end

# ── Boundary trace evaluation ─────────────────────────────────────────────────

"""
    boundary_trace_data(level, raw_dofs, sides, xi, cell) -> NamedTuple

Evaluate the basis traces of `level` on the codim-K facet identified by
`sides`. Returns a `NamedTuple` with two fields:

  - `raw_dofs::Vector{Int}` — the subset of `raw_dofs` whose basis
    functions have support on the facet (filtered by
    [`is_facet_basis`](@ref)). Listed in the original local-basis order.
  - `values::Vector{T}` — corresponding tensor-product basis values at
    reference point `xi` on the parent cell.

`cell::CartesianIndex{D}` is the parent cell's mesh-index along each
axis. Integrated Legendre ignores it (the 1D modes are cell-local in
the reference frame); the B-spline family in the extension uses it to
pick the right knot-vector span when computing the trace. Callers
should pass the parent's own cell — `_project_dirichlet_values!`
already has it as `parent.cell`.

For integrated Legendre, `is_facet_basis` reduces to "every 1D mode
along a constrained axis equals the boundary mode" — the trace is the
straightforward tensor product of `integrated_legendre_value` factors.
The fallback method throws for any other basis family.
"""
function boundary_trace_data(level::Level{D,T,<:IntegratedLegendre}, raw_dofs::Vector{Int},
                             sides::Vector{Tuple{Int,Symbol}}, xi::SVector{D,T},
                             ::CartesianIndex{D}) where {D,T}
    local_ids = local_basis_indices(level.basis, level.order, level.mode)
    raws = Int[]
    values = T[]

    for (i, local_id) in pairs(local_ids)
        is_facet_basis(level.basis, local_id, sides) || continue
        push!(raws, raw_dofs[i])
        push!(values, prod(integrated_legendre_value(local_id.I[d], xi[d]) for d in 1:D))
    end

    return (; raw_dofs=raws, values)
end

function boundary_trace_data(level::Level{D,T,B}, raw_dofs::Vector{Int},
                             sides::Vector{Tuple{Int,Symbol}}, xi::SVector{D,T},
                             ::CartesianIndex{D}) where {D,T,B}
    throw(ArgumentError("boundary trace projection is not implemented for basis family $(basis_name(level.basis))"))
end

# ── Dirichlet projection ──────────────────────────────────────────────────────

# Project nonzero physical Dirichlet data onto the boundary trace space
# by an L² mass-matrix solve. For each physical Dirichlet constrained
# dof we want a value `g_h` such that
#
#     ∫_∂Ω g_h v dx  =  ∫_∂Ω g v dx
#
# for every test trace `v` from the constrained-dof subspace. This is
# the classical L² projection; the resulting linear system is
#
#     M c = b,   M_ij = ∫_∂Ω φ_i φ_j dx,   b_i = ∫_∂Ω g φ_i dx,
#
# where `φ_i` are the boundary traces of the physically-constrained,
# non-overlay-constrained raw dofs. The mass matrix is symmetric
# positive (semi-)definite, so we try a Cholesky factorisation first
# and fall back to a pseudoinverse on the rare cases where Cholesky
# reports indefiniteness (degenerate facets, zero-area boundaries under
# heavy masking).
#
# `M` and the boundary walk that builds it depend on the mesh alone; only
# `b` depends on the prescribed data. A load-stepping driver calls
# [`update_dirichlet!`](@ref) once per increment on a fixed mesh, so the
# walk is split in two: [`DirichletProjection`](@ref) holds everything the
# datum does not enter — the per-component unknown sets, the factorised
# mass, and the sampled boundary traces — and is built once and reused,
# while each increment re-accumulates `b` from the samples and
# back-substitutes. See `DirichletProjection` for why that reuse is exact
# and when it is invalidated.
#
# The `_test_value_contribution` and `_as_test_channels` helpers are
# defined in `assembly.jl`'s channel layer and reused here so the
# Dirichlet projection rides the same channel calculus the volume
# assembly uses; the forward reference resolves at call time
# (`prepare(problem)` loads every source file first).

"""
    FacetTraceSamples{D,T}(points, weights, offsets, values, rows)

The quadrature sampling of one Dirichlet condition's facet: every
boundary quadrature point on that facet together with the boundary
traces of the constrained dofs evaluated there.

The right-hand side `b_i = ∫_∂Ω g φ_i dx` is the only part of the L²
projection the prescribed data enter, and it is a quadrature sum over
exactly these samples. Recording them lets a load increment re-evaluate
`g` and re-accumulate `b` without re-deriving the facet regions or the
basis traces — the two dominant costs of the walk, both of which scale
with the finest level's *background* boundary grid rather than with the
active cells on the boundary.

Fields:

  - `points::Vector{SVector{D,T}}` — the facet's physical-frame
    quadrature coordinates, in the order the region walk visits them.
  - `weights::Vector{T}` — the matching physical-frame weights, with the
    facet Jacobian already applied (see [`FacetRegion`](@ref)).
  - `offsets::Vector{Int}` — sample pointers of length
    `length(points) + 1`: the trace entries of sample `k` are
    `offsets[k] : offsets[k+1] - 1`.
  - `values::Vector{T}` — each entry's trace value `φ_i(x_k)`.
  - `rows::Vector{Vector{Int}}` — per component, each entry's row in that
    component's unknown set, or `0` when the entry's dof is not an
    unknown of that component. Components the condition does not
    constrain carry an empty vector and are never indexed.
"""
struct FacetTraceSamples{D,T<:Real}
    points::Vector{SVector{D,T}}
    weights::Vector{T}
    offsets::Vector{Int}
    values::Vector{T}
    rows::Vector{Vector{Int}}
end

# One component's solved boundary mass: the Cholesky factorisation when the
# mass is positive definite, its pseudoinverse when Cholesky reports
# indefiniteness, and `nothing` when the component has no unknowns at all.
const DirichletFactor{T} = Union{Nothing,Cholesky{T,Matrix{T}},Matrix{T}}

"""
    DirichletProjection{D,T}(unknowns, factors, samples)

Everything in the L² Dirichlet projection `M c = b` that the prescribed
data `g` do not enter: the per-component unknown sets, the per-component
factorised boundary mass, and the per-facet trace samples a right-hand
side is accumulated from.

Fields:

  - `facets::Int` — the number of (condition, facet) pairs the
    projection was built for. A condition list of a different shape
    cannot reuse it.
  - `unknowns::Vector{Vector{Int}}` — per component, the raw dofs the
    projection solves for: the physically-constrained,
    non-overlay-constrained dofs of that component.
  - `factors::Vector{DirichletFactor{T}}` — per component, the solved
    boundary mass (see `DirichletFactor`).
  - `samples::Vector{FacetTraceSamples{D,T}}` — one entry per
    (condition, facet) pair, in the order
    `condition × _facets(condition.boundary)` enumerates them. Empty
    when no component has an unknown, since then nothing is integrated.

# Validity

A projection is valid for the `(layout, space, condition-structure)`
triple it was built from. The unknown sets, the facets and the mass
depend on the mesh, the masks and each condition's boundary selector,
field and component — never on a condition's *value*. Those are exactly
the quantities [`update_dirichlet!`](@ref) pins with
`_check_dirichlet_update_compatibility`, which is why an increment may
reuse the projection; anything else that moves goes through
[`prepare`](@ref), `move!`, `activate!` or `deactivate!`, each of which
builds a fresh dof layout and drops the cached projection with it.

# Exactness

Reuse is bit-exact, not merely accurate to roundoff. The samples are
recorded in the region walk's own order and the right-hand side is
re-accumulated term by term in that order, so a reused projection
produces the same floating-point `b` — and, from the same stored
factorisation, the same `c` — as a full rebuild would.
"""
struct DirichletProjection{D,T<:Real}
    facets::Int
    unknowns::Vector{Vector{Int}}
    factors::Vector{DirichletFactor{T}}
    samples::Vector{FacetTraceSamples{D,T}}
end

# True iff `condition` constrains `component`: an unscoped condition
# constrains every component of its field, a scoped one only its own.
function _constrains(condition::DirichletCondition, component::Integer)
    return condition.component === nothing || condition.component == component
end

# Number of (condition, facet) pairs a Dirichlet list covers. Recorded on a
# `DirichletProjection` as its `facets` field so a cached projection can be
# rejected when the condition list no longer has the shape it was built for.
function _dirichlet_facet_count(dirichlet, ::Val{D}) where {D}
    return sum(c -> length(_facets(c.boundary, Val(D))), dirichlet; init=0)
end

# Build the datum-independent half of the projection.
#
# The projection is done PER COMPONENT. Two boundary traces couple in the L²
# projection only through the part of ∂Ω where BOTH are constrained in the
# SAME displacement component; a dof constrained in one component must not
# enter another component's mass matrix. A single shared mass over the union
# of all component-constrained dofs (integrated over every condition's facet,
# component-blind) is wrong on two counts: two conditions on the same facet
# (e.g. u_x = 0 and u_y = ḡ on one edge) double-count the mass and halve the
# projected value, and a component-1-only condition (e.g. a lateral u_x = 0)
# adds mass to a shared corner's component-2 row with no matching right-hand
# side, pulling that u_y toward zero. Per-component unknown sets, mass
# matrices, and right-hand sides — each accumulated only over the facets
# where a condition actually constrains that component — remove both. For
# all-component (vector) conditions this reduces to the previous behaviour.
function _dirichlet_projection(layout::DofLayout{D,T}, V::Space{D,T}, dirichlet) where {D,T}
    ncomp = layout.components
    facets = _dirichlet_facet_count(dirichlet, Val(D))
    unknowns = [[raw
                 for raw in eachindex(layout.raw_keys)
                 if layout.elimination_source[raw] === :free && layout.physical_dirichlet[raw, c]]
                for c in 1:ncomp]
    empty_factors = DirichletFactor{T}[nothing for _ in 1:ncomp]
    all(isempty, unknowns) &&
        return DirichletProjection{D,T}(facets, unknowns, empty_factors,
                                        FacetTraceSamples{D,T}[])

    index = [Dict(raw => i for (i, raw) in pairs(unknowns[c])) for c in 1:ncomp]
    # The constrained-boundary mass couples only the physically-Dirichlet dofs
    # of one component and is dense-solved below, so each component's is
    # accumulated directly into a dense matrix — no sparse intermediate.
    mass = [zeros(T, length(unknowns[c]), length(unknowns[c])) for c in 1:ncomp]

    samples = FacetTraceSamples{D,T}[]
    for condition in dirichlet
        for sides in _facets(condition.boundary, Val(D))
            push!(samples, _sample_dirichlet_facet!(mass, index, layout, V, condition, sides))
        end
    end

    # Solve `M c = b` for each component. Cholesky first; pseudoinverse fallback
    # on the rare indefinite case (degenerate / zero-area facets under heavy
    # masking, or a codim-D point pin whose facet carries no measure). Both are
    # stored rather than applied, so every later increment back-substitutes
    # against the same factorisation instead of rebuilding it.
    factors = DirichletFactor{T}[]
    for component in 1:ncomp
        if isempty(unknowns[component])
            push!(factors, nothing)
            continue
        end
        factor = cholesky(Symmetric(mass[component]); check=false)
        push!(factors, issuccess(factor) ? factor : pinv(mass[component]))
    end

    return DirichletProjection{D,T}(facets, unknowns, factors, samples)
end

# Walk one condition's facet: the boundary regions on that facet × the
# per-region quadrature points. At each quadrature point, evaluate the
# boundary traces of every covering parent once, record them as a sample, and
# accumulate the mass-matrix contribution of every component the condition
# constrains.
function _sample_dirichlet_facet!(mass::Vector{Matrix{T}}, index::Vector{Dict{Int,Int}},
                                  layout::DofLayout{D,T}, V::Space{D,T},
                                  condition::DirichletCondition,
                                  sides::Vector{Tuple{Int,Symbol}}) where {D,T}
    ncomp = layout.components
    points = SVector{D,T}[]
    weights = T[]
    offsets = Int[1]
    values = T[]
    rows = [Int[] for _ in 1:ncomp]

    for region in _boundary_facet_regions(V, sides, layout.tolerance)
        for (qp, x) in pairs(region.points)
            qweight = region.weights[qp]

            # Evaluate every parent's boundary traces at this point once and
            # reuse them for the sample and for every component's mass.
            # `parent.parent_box` is precomputed on the region, so the only
            # per-Q-point geometry work is the affine `physical_to_reference`.
            traces = map(region.parents) do parent
                level = _level_by_id(V, parent.level)
                raw_dofs = cell_dofs(layout, parent.level, parent.cell)
                xi = physical_to_reference(parent.parent_box, x)
                boundary_trace_data(level, raw_dofs, sides, xi, parent.cell)
            end

            # Record the sample. Entries are pushed in the same
            # parent-then-mode order the right-hand-side loop below would visit
            # them in, which is what makes a re-accumulated `b` bit-identical.
            push!(points, x)
            push!(weights, qweight)
            for trace in traces
                for a in eachindex(trace.raw_dofs)
                    push!(values, trace.values[a])
                    for component in 1:ncomp
                        _constrains(condition, component) || continue
                        push!(rows[component], get(index[component], trace.raw_dofs[a], 0))
                    end
                end
            end
            push!(offsets, length(values) + 1)

            # Mass: ∫_∂Ω φ_i φ_j dx over this component's constrained dofs.
            # Both indices iterate the same `traces`, so it is symmetric.
            for component in 1:ncomp
                _constrains(condition, component) || continue
                index_c = index[component]
                mass_c = mass[component]
                for trial_trace in traces
                    for b in eachindex(trial_trace.raw_dofs)
                        col = get(index_c, trial_trace.raw_dofs[b], 0)
                        col == 0 && continue
                        mass_channels = _as_test_channels(trial_trace.values[b], Val(D), T)
                        for test_trace in traces
                            for a in eachindex(test_trace.raw_dofs)
                                row = get(index_c, test_trace.raw_dofs[a], 0)
                                row == 0 && continue
                                mass_c[row, col] += qweight *
                                                    _test_value_contribution(mass_channels,
                                                                             test_trace.values[a])
                            end
                        end
                    end
                end
            end
        end
    end

    return FacetTraceSamples{D,T}(points, weights, offsets, values, rows)
end

# Accumulate one facet's contribution to the per-component right-hand sides
# `b_i = ∫_∂Ω g φ_i dx`. `condition` is deliberately left to dispatch: the
# Dirichlet list is heterogeneous (each condition carries its own value type),
# so this call is the function barrier that specializes the sample loop on the
# concrete condition and keeps the `g` evaluation free of dynamic dispatch.
#
# The `@inbounds` rests on two construction invariants of
# [`FacetTraceSamples`](@ref): `offsets` never leaves `1 : length(values) + 1`
# and `rows` is as long as `values`, so `e` indexes both; and every nonzero
# `row` came from the component's own unknown index map, so it indexes `rhs_c`.
function _accumulate_dirichlet_rhs!(rhs::Vector{Vector{T}}, samples::FacetTraceSamples{D,T},
                                    condition::DirichletCondition, ncomp::Integer) where {D,T}
    for k in eachindex(samples.points)
        x = samples.points[k]
        qweight = samples.weights[k]
        first_entry = samples.offsets[k]
        last_entry = samples.offsets[k + 1] - 1
        for component in 1:ncomp
            _constrains(condition, component) || continue
            g = convert(T, _condition_value(condition, x, component))
            channels = _as_test_channels(g, Val(D), T)
            rows = samples.rows[component]
            rhs_c = rhs[component]
            @inbounds for e in first_entry:last_entry
                row = rows[e]
                row == 0 && continue
                rhs_c[row] += qweight * _test_value_contribution(channels, samples.values[e])
            end
        end
    end
    return rhs
end

# Refill `layout.constrained_values` from `dirichlet`, returning the
# [`DirichletProjection`](@ref) used so a caller that re-projects the same
# layout can hand it back and skip the boundary walk. `projection` is reused
# when given; the facet-count check is a cheap guard for a caller that reaches
# past `update_dirichlet!`'s structural check, which is what actually
# establishes that a cached projection still matches the condition list.
function _project_dirichlet_values!(layout::DofLayout{D,T}, V::Space{D,T}, dirichlet,
                                    projection::Union{Nothing,
                                                      DirichletProjection{D,T}}=nothing) where {D,T}
    fill!(layout.constrained_values, zero(T))
    ncomp = layout.components
    plan = if projection === nothing ||
              projection.facets != _dirichlet_facet_count(dirichlet, Val(D))
        _dirichlet_projection(layout, V, dirichlet)
    else
        projection
    end
    all(isempty, plan.unknowns) && return plan

    rhs = [zeros(T, length(plan.unknowns[c])) for c in 1:ncomp]
    facet = 0
    for condition in dirichlet
        for _ in _facets(condition.boundary, Val(D))
            facet += 1
            _accumulate_dirichlet_rhs!(rhs, plan.samples[facet], condition, ncomp)
        end
    end

    for component in 1:ncomp
        factor = plan.factors[component]
        factor === nothing && continue
        projected = factor isa Cholesky ? factor \ rhs[component] : factor * rhs[component]
        for (i, raw) in pairs(plan.unknowns[component])
            layout.constrained_values[raw, component] = projected[i]
        end
    end

    return plan
end

# True iff `dirichlet` carries at least one nonzero condition. The
# homogeneous-zero case skips the projection: every constrained value
# already equals zero from the layout's zero-initialised
# `constrained_values` matrix.
function _needs_dirichlet_projection(dirichlet)
    return any(condition -> !(condition.value isa Number && iszero(condition.value)), dirichlet)
end

# True iff `a` and `b` pick out the same physical facet. Named alias for the
# value `==` defined next to `BoundarySelector` above, kept because it reads as
# a predicate at the call sites that ask the question in those words (e.g.
# `update_dirichlet!`'s structural check, whose function lives in
# `src/model.jl`).
_selectors_equal(a::BoundarySelector, b::BoundarySelector) = a == b

# Cheap structural compatibility check. Re-projecting Dirichlet values
# in place is only valid when the *set* of constrained dofs is
# unchanged; that set depends on the boundary selector, field, and
# component of each condition, but not on its value. Anything else
# moving means the caller has to go through `prepare(problem)` instead.
function _check_dirichlet_update_compatibility(old, new)
    length(new) == length(old) || throw(ArgumentError("update_dirichlet! expects $(length(old)) " *
                                                      "conditions (the count the model was prepared " *
                                                      "with), got $(length(new))"))
    for (i, (o, n)) in enumerate(zip(old, new))
        _selectors_equal(o.boundary, n.boundary) ||
            throw(ArgumentError("update_dirichlet! condition $i: boundary selector changed; " *
                                "rebuild the model with prepare(problem) instead"))
        o.field === n.field || throw(ArgumentError("update_dirichlet! condition $i: field name " *
                                                   "changed; rebuild the model with " *
                                                   "prepare(problem) instead"))
        o.component === n.component ||
            throw(ArgumentError("update_dirichlet! condition $i: component changed; rebuild the " *
                                "model with prepare(problem) instead"))
    end
    return nothing
end
