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
conditions and the dof-layer's constraint detection. Three `selector`
shapes are supported:

  - `:all` — the whole physical boundary `∂Ω`, encoded as the union of
    all codim-1 faces. `sides` is empty in this case.
  - `:sides` — the intersection of one or more `(axis, side)` pairs
    listed in `sides`. A single pair picks a codim-1 face; two pairs in
    2D pick a corner; two pairs in 3D pick an edge; three pairs in 3D
    pick a vertex. Each axis may appear at most once and `side` must be
    `:lower` or `:upper`.
  - `:except` — the union of every codim-1 face *except* the ones listed
    in `sides`. Here `sides` is a complement rather than a facet
    specification, and an axis may appear twice (both of its faces
    excluded). The kept faces are taken closed, so a dof on the edge
    between a kept and an excluded face still matches.

`sides` is therefore read against `selector` and never on its own. The
kind lives in `selector` rather than in a fourth field because the
facet-region cache keys on a selector's *value* (see the `==` / `hash`
methods below), and a `:sides` selector's `sides` can never be mistaken
for an `:except` one's: the discriminant is already there.

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
function Base.:(==)(a::BoundarySelector, b::BoundarySelector)
    a.selector === b.selector && a.sides == b.sides
end
function Base.hash(s::BoundarySelector, h::UInt)
    hash(s.sides, hash(s.selector, hash(:BoundarySelector, h)))
end

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
# family that implements `_key_on_level_side`:
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

# True iff the dof key sits on *any* of the 2·D codim-1 physical faces.
# Used to decide whether a key matches a `boundary(:all)` selector.
function _key_on_any_physical_side(key::TensorDofKey{D}, level::Level{D,T}, domain::AxisBox{D,T},
                                   tol::GeometryTolerance{T}) where {D,T}
    return any(_key_on_physical_side(key, level, domain, axis, side, tol) for axis in 1:D
               for side in (:lower, :upper))
end

# The codim-1 faces an `:except` selector keeps: every face of the physical box
# that its `sides` list — the *excluded* faces — does not name. The order is
# `boundary(:all)`'s own, axis-major with `:lower` before `:upper`, so an
# `:except` selector enumerates its facets in exactly the order the whole
# boundary would, minus the dropped entries. That is what makes the Dirichlet
# projection through one `:except` condition bit-identical to the same faces
# written out individually in that order, rather than merely equal to roundoff:
# the projection accumulates its mass and right-hand side facet by facet in this
# walk's order, and floating-point addition is not associative.
#
# Both checks need `D` and so cannot live in `boundary`, which builds a selector
# without reference to a space — exactly as the bound check on
# `boundary(axis=4, side=:lower)` waits for one. Excluding every face is an
# error, not an empty facet list: nothing selected is never what a caller meant,
# and silently constraining nothing is the failure mode this selector exists to
# prevent.
function _kept_faces(excluded::Vector{Tuple{Int,Symbol}}, ::Val{D}) where {D}
    for (axis, _) in excluded
        1 <= axis <= D ||
            throw(ArgumentError("boundary axis $axis is out of bounds for dimension $D"))
    end
    kept = [(axis, side) for axis in 1:D for side in (:lower, :upper) if (axis, side) ∉ excluded]
    isempty(kept) && throw(ArgumentError("boundary(:all; except=…) excludes all $(2D) faces of " *
                                         "the physical boundary in $(D)D, selecting nothing"))
    return kept
end

# True iff the dof key matches the facet picked out by a `BoundarySelector`.
# For `:all` the predicate is "on any physical face"; for `:except` it is "on any
# kept physical face", which is the same union over a subset; for `:sides` it is
# "on every listed (axis, side) face" (i.e. the intersection of the listed
# codim-1 faces).
function _selector_matches(key::TensorDofKey{D}, level::Level{D,T}, domain::AxisBox{D,T},
                           selector::BoundarySelector, tol::GeometryTolerance{T}) where {D,T}
    selector.selector === :all && return _key_on_any_physical_side(key, level, domain, tol)

    if selector.selector === :except
        return any(_key_on_physical_side(key, level, domain, axis, side, tol)
                   for (axis, side) in _kept_faces(selector.sides, Val(D)))
    end

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
# `∂Ω`); for `:except` it is one per *kept* face, the same list with the
# excluded entries removed; for `:sides` it is the single facet identified
# by the listed `(axis, side)` constraints. Used by
# `_project_dirichlet_values!` to iterate over the support of each Dirichlet
# condition, and by `_facet_regions_for_selector` to assemble the region
# union — which is why a union selector needs nothing else to integrate.
function _facets(selector::BoundarySelector, ::Val{D}) where {D}
    if selector.selector === :all
        return [Tuple{Int,Symbol}[(axis, side)] for axis in 1:D for side in (:lower, :upper)]
    elseif selector.selector === :except
        return [Tuple{Int,Symbol}[face] for face in _kept_faces(selector.sides, Val(D))]
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
    FacetRegion{D,T}(sides, lower, upper, parents, kind, points, weights, normal)

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
    ([`boundary_trace_indices`](@ref) and `is_facet_basis`) to filter
    facet-incident basis modes.
  - `lower::SVector{D,T}`, `upper::SVector{D,T}` — the region's own corner
    coordinates: the facet's fixed coordinate on every constrained axis
    (so `lower[axis] == upper[axis]` there), and the region's merged
    interval bounds on the free axes. They are the region's *own* extent,
    which after the greedy merge is in general a strict subset of the
    intersection of its parents' cell faces, so a consumer that needs the
    region's geometry — rather than a bound on it — has to read them here.
    Held as a corner pair rather than an [`AxisBox`](@ref) because a facet
    box is degenerate on its constrained axes and `AxisBox` requires
    strictly positive extent on every axis.
  - `parents::Vector{FacetParent{D,T}}` — every level cell whose own
    facet on `sides` coincides with this region. The parent set is
    constant across the region's quadrature points (the contract that
    makes "evaluate basis values once per parent" amortizable).
  - `kind::Symbol` — how `Ω` meets the region's own face, in
    [`classify_cell`](@ref)'s own three-valued vocabulary: `:full` (the
    face lies inside `Ω`), `:cut` (`∂Ω` crosses it), or `:fictitious`
    (the face lies outside `Ω`). It is the verdict of the classifier on
    the region's `(D − K)`-dimensional face box against the level set
    restricted to the facet's affine slice, so it carries exactly
    [`classify_cell`](@ref)'s own resolution — `subcell_length_scale` and
    `max_depth` — rather than the rule's point spacing: a fictitious sliver
    of face is `:cut` whenever the classifier resolves it, however the
    rule's points below happen to fall, and one finer than that budget is
    invisible here exactly as it is to the cell classification behind the
    face. A space carrying no [`PhysicalDomain`](@ref) has nothing outside
    `Ω`, so every region of it is `:full`. The rule below covers the whole
    face whatever the kind — `kind` is what says how much of that face is
    physical, and `AssemblyDiagnostics.cut_facet_region_count` counts the
    regions whose kind is not `:full`.
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
    lower::SVector{D,T}
    upper::SVector{D,T}
    parents::Vector{FacetParent{D,T}}
    kind::Symbol
    points::Vector{SVector{D,T}}
    weights::Vector{T}
    normal::SVector{D,T}
end

# Key of the per-*face* memo a `FacetResolver` holds: one entry per
# `(space, facet)` pair, where the facet is the `(axis, side)` list a
# `BoundarySelector` decomposes into (`_facets`). This is one level below
# `RegionKey` (`model.jl`), which keys whole selectors.
#
# The distinction matters because a selector is a *union* of faces while
# `_boundary_facet_regions` resolves *one* face, so two selectors that are not
# value-equal can still overlap: `boundary(:all)` covers every face, hence every
# face any other selector names. Keyed on the selector, each overlap resolves the
# shared face again; keyed on the face, each face of each space is resolved
# exactly once and the selectors' region vectors are assembled from the shared
# results. That is worth doing because the overlap is not rare — it is what any
# spelling of "the whole boundary except one face" produces for a coupled
# problem, one such selector per field.
#
# The space half is identity-keyed, because `problem_spaces` already deduplicates
# spaces by `===`: "a distinct discretisation" is package-wide synonymous with
# "a distinct `Space` object". The `sides` half is a `Vector{Tuple{Int,Symbol}}`,
# which `Tuple` hashes and compares element-wise, so the same face spelled by two
# different selectors is one key.
const FacetKey = Tuple{Any,Vector{Tuple{Int,Symbol}}}

"""
    FacetResolver{D,T}(tolerance)
    FacetResolver{D,T}(tolerance, faces)

The package's one route from a boundary face to its [`FacetRegion`](@ref)s: the
geometry tolerance the resolution runs at, together with the per-face memo
(`FacetKey` → region list) it fills as faces are asked for.

One resolver belongs to a [`Model`](@ref) and lives as long as it does, so every
consumer of grid-aligned boundary integration shares one resolution per
`(space, face)` pair — the per-selector cache `prepare` builds
(`_resolve_facet_regions`), the one-shot lookup `_resolve_on_regions` falls back
to for a selector no `prepare` saw, and the L² Dirichlet projection
(`_dirichlet_projection`), which reaches the same faces from the dof layer.
Sharing is worth arranging because `_boundary_facet_regions` is a pure function
of `(V, sides, tolerance)`: a second resolution of a face can only ever
reproduce the first one, at full price.

The memo is filled lazily and so is mutated by lookups, which is safe for the
same reason the integration plan's moment-fit cache is: every resolution happens
on the task that called the public API — `prepare`, `update_dirichlet!`,
`boundary_integral`, or an `assemble*` call resolving its passes — and never
inside an assembly worker, which is handed a region list that is already built.

Fields:

  - `tolerance::GeometryTolerance{T}` — the tolerance every resolution through
    this memo runs at. It travels with the memo rather than with each call
    because it is part of what a cached entry *means*: the coordinates a face is
    partitioned at are merged against it, so regions built at one tolerance are
    not the regions another would have produced, and a consumer holding a
    different one must not read these entries. `_project_dirichlet_values!`
    raises on the mismatch rather than resolving against the wrong memo.
  - `faces::Dict{FacetKey,Vector{FacetRegion{D,T}}}` — the memo itself. The
    space half of a key is matched by identity, so an entry for a space that is
    no longer referenced is dead weight and never a stale answer; every mutator
    replaces the model's resolver along with the dof layout, which is what keeps
    such entries from accumulating across a move or a mask flip. Within one space
    the memo is bounded by construction: a `BoundarySelector` decomposes into
    codim-`K` facets of the background box, of which there are at most `3^D − 1`
    (each axis is unconstrained, pinned low, or pinned high), so no sequence of
    one-shot lookups can grow it without limit.
"""
struct FacetResolver{D,T<:Real}
    tolerance::GeometryTolerance{T}
    faces::Dict{FacetKey,Vector{FacetRegion{D,T}}}
end

function FacetResolver{D,T}(tolerance::GeometryTolerance{T}) where {D,T<:Real}
    return FacetResolver{D,T}(tolerance, Dict{FacetKey,Vector{FacetRegion{D,T}}}())
end

# Resolve one face of `V` through `facets`, building it with
# `_boundary_facet_regions` (defined below) on a miss. This is that function's
# only `get!` call site anywhere in the package, which is what makes "one
# resolution per (space, face) per model lifetime" a property of the code rather
# than a convention several independent call sites happen to honour.
function _resolve_face(facets::FacetResolver{D,T}, V::Space{D,T},
                       sides::Vector{Tuple{Int,Symbol}}) where {D,T}
    return get!(() -> _boundary_facet_regions(V, sides, facets.tolerance), facets.faces, (V, sides))
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
                recommended_quadrature_order(level.basis, cell_order(level, parent.cell))[d]
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

# The cells of `level` whose own faces all lie on `sides`: index `1` or the last
# index on each constrained axis, the whole range on the free ones. These are
# the only cells `_facet_signature` can accept as parents, so restricting the
# coordinate projection to them makes the facet partition follow the *active
# boundary layer* rather than the level's whole active region — an overlay live
# deep in the interior then contributes nothing to a facet it never touches.
function _side_cells(level::Level{D}, sides::Vector{Tuple{Int,Symbol}}) where {D}
    counts = level.mesh.cells
    return CartesianIndices(ntuple(D) do d
                                for (axis, side) in sides
                                    axis == d || continue
                                    side === :lower && return 1:1
                                    side === :upper && return counts[d]:counts[d]
                                    throw(ArgumentError("boundary side must be :lower or :upper"))
                                end
                                return 1:counts[d]
                            end)
end

# Build the admissible boundary regions on the facet identified by
# `sides`.
#
# The **partition** is grid-aligned and level-set-blind, and that is the one
# property to know before using them. No `PhysicalDomain` reaches steps 1–5:
# step 1 selects levels by comparing mesh edges against the space's `AxisBox`
# (`_level_side_is_physical`), step 2 partitions over `_side_cells`, step 4 asks
# `locate_cell` and the `LevelMask`. So a region is the whole face of its parent
# cells, and the rule in step 6 covers all of it — where `∂Ω` crosses that face
# the rule integrates the fictitious part along with the physical part. Every
# consumer inherits it: the L² Dirichlet mass and right-hand side below, a
# Neumann / Robin / Nitsche term tagged `on::BoundarySelector`, and
# `boundary_integral`. A homogeneous Dirichlet datum is unaffected (∫ 0 = 0, and
# `_needs_dirichlet_projection` skips the walk for one), which is why the
# property is easy to miss. What *does* reach the level set is step 7, which
# classifies each region's own face against Ω and records the verdict on
# `FacetRegion.kind`; `cut_facet_region_count` in the assembly diagnostics counts
# the regions whose kind is not `:full` — exactly the faces part of whose weight
# falls outside Ω. Integration over the immersed boundary itself goes
# through a `BoundaryMesh` (`surface.jl`), which *is* cut against the grid. The
# whole-cell fictitious fold reaches here too, through the mask step 4 reads: a
# fully fictitious cell is inactive and parents nothing, so the face shrinks by
# whole cells, never within one.
#
# The geometry arrives through `V`, not through an argument of its own, so this
# stays a pure function of `(V, sides, tol)` — which is what lets the per-face
# memo of a [`FacetResolver`](@ref) key on those three alone and what keeps the
# two consumers above reading the same faces automatically.
#
# The algorithm mirrors the volume-region construction in `intersections.jl`,
# restricted to the facet:
#
#   1. Identify the levels whose own mesh face on `sides` coincides with
#      the physical-domain face — only these levels contribute boundary
#      trace dofs on this facet.
#   2. Per free axis, collect every touching level's boundary coordinates
#      along that axis that bound an *active* cell whose own face lies on
#      `sides` — the only cells that can become parents here — merge them,
#      and form non-degenerate intervals. An unmasked level contributes its
#      whole grid, since every cell of it is active.
#   3. Take the Cartesian product of the intervals to enumerate the
#      candidate sub-rectangles. Fix the constrained-axis coordinates to
#      the facet's coordinates.
#   4. For each candidate, compute the per-level *signature* at its
#      midpoint: the linear index of the touching level's on-side active
#      cell containing it, or `0` where the level contributes nothing —
#      a cell off the side, or one the `LevelMask` deactivated, has no
#      dofs to constrain. Drop candidates whose signature is all zero.
#   5. Greedily merge axis-adjacent candidates that share a signature,
#      exactly as `_merged_boxes` does over the volume (steps 3–4 there,
#      through the same `_extend_axis`). Every point of a merged region
#      still lies inside one cell of every parent level, so a product of
#      boundary traces is still polynomial on it and the rule below is
#      still exact for it.
#   6. Precompute the per-sub-rectangle Gauss rule and bake the
#      `vol / 2^(D-K)` Jacobian into the weights. The returned
#      `FacetRegion` is ready for direct consumption — every Q-point is
#      in physical coordinates with the physical-frame weight already
#      applied.
#   7. Classify each region's own face against Ω — `:full`, `:cut` or
#      `:fictitious` — by restricting the space's level set to the facet's
#      affine slice and handing `classify_cell` the region's face box, the
#      facet counterpart of the volume path's step 7 (`CONTRIBUTING.md`,
#      "Integration regions"). The verdict lands on `FacetRegion.kind`.
#
# Steps 2 and 5 together make the facet partition follow the *active* cells
# on the boundary rather than the finest touching level's background grid.
# One consequence is worth stating plainly, because it is the one thing the
# merge changes rather than merely makes cheaper. The exactness argument
# covers products of boundary traces: the Dirichlet mass matrix, and the
# Neumann / `boundary_integral` terms. It does not cover a *non-polynomial*
# datum `g` in the Dirichlet right-hand side `b_i = ∫_∂Ω g φ_i`. The finer
# slicing integrated that with an incidental composite rule at the finest
# touching level's boundary spacing; it is now integrated with the parent
# cell's own rule — the same convention `assembly.jl` already uses for a
# non-polynomial volume source. Projected Dirichlet values for such a datum
# therefore move at the boundary-quadrature-error level, not at roundoff.
function _boundary_facet_regions(V::Space{D,T}, sides::Vector{Tuple{Int,Symbol}},
                                 tol::GeometryTolerance{T}) where {D,T}
    touching_levels = [level
                       for level in V.levels
                       if all(_level_side_is_physical(level, V.domain, axis, side, tol)
                              for (axis, side) in sides)]
    isempty(touching_levels) && return FacetRegion{D,T}[]

    free_axes = _free_axes(sides, Val(D))

    # Per-axis intervals along the free axes, mirroring `_axis_intervals` from
    # intersections.jl but scanning only the cells that can parent a region
    # here (`_side_cells`).
    merged = _merged_axis_coordinates(touching_levels, Val(D), tol,
                                      level -> _side_cells(level, sides))
    intervals = map(d -> _intervals_from_coordinates(merged[d], tol), free_axes)
    any(isempty, intervals) && return FacetRegion{D,T}[]

    return _merged_facet_regions(V, touching_levels, sides, free_axes, intervals, tol,
                                 Val(length(free_axes)))
end

# Physical corners of the facet sub-rectangle spanned by the candidate index
# range `start:hi`: the facet's own coordinate on every constrained axis, the
# interval endpoints on the free axes. `start == hi` gives one candidate.
function _facet_corners(domain::AxisBox{D,T}, sides, free_axes, intervals, start::NTuple{F,Int},
                        hi::NTuple{F,Int}) where {D,T,F}
    lower = MVector{D,T}(ntuple(d -> _facet_fixed_coord(domain, sides, d), D))
    upper = MVector{D,T}(ntuple(d -> _facet_fixed_coord(domain, sides, d), D))
    for (j, d) in pairs(free_axes)
        lower[d] = intervals[j][start[j]][1]
        upper[d] = intervals[j][hi[j]][2]
    end
    return SVector{D,T}(lower), SVector{D,T}(upper)
end

# On-side coverage signature at a facet point: per touching level, the linear
# index of the active cell whose own face on `sides` contains the point, and
# `0` where the level contributes no boundary dof there. The facet counterpart
# of `_coverage_signature` (intersections.jl), with that one's "off the mesh or
# inactive" rule widened by the side filter, and it carries everything the
# parent list does — `_facet_parents` decodes it once per *emitted* region
# instead of building a parent list once per candidate.
function _facet_signature(levels, linmaps, sides, point::SVector{D,T},
                          tol::GeometryTolerance{T}) where {D,T}
    sig = zeros(Int, length(levels))
    for (i, level) in pairs(levels)
        cell = locate_cell(level.mesh, point; tol)
        cell === nothing && continue
        all(_cell_on_side(level, cell, axis, side) for (axis, side) in sides) || continue
        is_active(level.mask, cell) || continue
        sig[i] = linmaps[i][cell]
    end
    return sig
end

# The parents a signature stands for, in `touching_levels` order.
function _facet_parents(levels, sig::Vector{Int}, ::Val{D}, ::Type{T}) where {D,T}
    parents = FacetParent{D,T}[]
    for (i, level) in pairs(levels)
        sig[i] == 0 && continue
        cell = CartesianIndices(level.mesh.cells)[sig[i]]
        push!(parents, FacetParent{D,T}(level.id, cell, cell_box(level.mesh, cell)))
    end
    return parents
end

# Candidate enumeration, greedy merge and quadrature for one facet, behind a
# `Val(F)` barrier on the number of free axes. `F` is a property of the facet's
# codimension and so a runtime value at the call site; the barrier is what makes
# `ranges`, the signature grid and the `visited` bitmap concretely typed, which
# the merge's inner slab test (`_extend_axis`) needs to stay a static dispatch.
# The codim-`D` vertex facet is `F = 0`, where the grid degenerates to the
# single candidate `CartesianIndex()`, no axis is ever extended, and the region
# is the point evaluation such a facet should be.
function _merged_facet_regions(V::Space{D,T}, touching_levels, sides::Vector{Tuple{Int,Symbol}},
                               free_axes::Vector{Int}, intervals, tol::GeometryTolerance{T},
                               ::Val{F}) where {D,T,F}
    axis_intervals = ntuple(j -> intervals[j], Val(F))
    ranges = ntuple(j -> length(axis_intervals[j]), Val(F))
    normal = _facet_outward_normal(sides, Val(D), T)
    linmaps = [LinearIndices(level.mesh.cells) for level in touching_levels]
    # Ω restricted to this facet's affine slice, built once per face rather than
    # once per region: the pinned coordinates are the face's own, so every region
    # on the face is classified against the same restricted tree.
    slice = V.physical === nothing ? nothing :
            _restrict_domain(V.physical, _facet_pins(V.domain, sides))

    sigs = map(CartesianIndices(ranges)) do index
        lower, upper = _facet_corners(V.domain, sides, free_axes, axis_intervals, index.I, index.I)
        midpoint = SVector{D,T}(ntuple(d -> (lower[d] + upper[d]) / 2, D))
        _facet_signature(touching_levels, linmaps, sides, midpoint, tol)
    end

    visited = falses(ranges)
    regions = FacetRegion{D,T}[]
    for start in CartesianIndices(ranges)
        visited[start] && continue
        sig = sigs[start]
        # No active on-side cell anywhere: nothing to constrain, and nothing a
        # neighbouring region may absorb either, since a merged region must
        # share this candidate's (empty) parent set.
        all(iszero, sig) && continue

        hi = start.I
        for j in 1:F
            hi = _extend_axis(sigs, visited, ranges, start.I, hi, sig, j)
        end
        for c in CartesianIndices(ntuple(e -> start.I[e]:hi[e], Val(F)))
            visited[c] = true
        end

        lower, upper = _facet_corners(V.domain, sides, free_axes, axis_intervals, start.I, hi)
        parents = _facet_parents(touching_levels, sig, Val(D), T)
        push!(regions,
              _facet_region(V, sides, parents, free_axes, lower, upper, normal, slice, Val(F)))
    end
    return regions
end

# ── Facet slices ──────────────────────────────────────────────────────────────
#
# A codim-`K` facet is the affine slice `⋂ⱼ {x_{kⱼ} = vⱼ}` of the background box,
# so the geometry of Ω *on* that facet is the level set restricted to the slice —
# a `PhysicalDomain` of dimension `F = D − K` (`_restrict_domain` in
# `src/physical.jl`). What follows is the coordinate half of that correspondence —
# the pins that name the slice, the facet region's extent as a box in the slice's
# own `F` coordinates, and the map back to ℝᴰ — and then the classification those
# three feed.

# The facet's pinned axes as `(axis, coordinate)` pairs in DESCENDING axis order:
# the order `_restrict_domain` must consume them in, and the reverse of the one
# `_lift` splices them back in. The coordinate is the facet's own, so the pins are
# a property of `(domain, sides)` alone and are shared by every region on the face.
function _facet_pins(domain::AxisBox{D,T}, sides::Vector{Tuple{Int,Symbol}}) where {D,T}
    return sort!([(axis, _facet_fixed_coord(domain, sides, axis)) for (axis, _) in sides]; by=first,
                 rev=true)
end

# The facet region's own extent as an `AxisBox` over its `F` free axes — the box
# the slice is classified and integrated on. Built by *selecting* the free axes
# rather than by dropping the pinned ones one at a time, because the intermediate
# box of a codim-`K > 1` drop is degenerate on the axes still pinned and
# `AxisBox`'s inner constructor rejects that (which is also why `FacetRegion`
# stores a corner pair instead of a box).
function _facet_box(lower::SVector{D,T}, upper::SVector{D,T}, free_axes::Vector{Int},
                    ::Val{F}) where {D,T,F}
    return AxisBox(SVector{F,T}(ntuple(j -> lower[free_axes[j]], Val(F))),
                   SVector{F,T}(ntuple(j -> upper[free_axes[j]], Val(F))))
end

# Splice the pinned coordinates back into a point of the slice, the inverse of the
# free-axis selection above and the map that returns anything built on the slice to
# the coordinates a facet's consumers work in. The pins are consumed in ASCENDING
# axis order — `_facet_pins` holds them descending, hence the reverse — because an
# insertion at index `k` lands on axis `k` of the result only once every axis below
# `k` is already in place.
function _lift(x::SVector, pins)
    return foldl((y, pin) -> _insert_axis(y, pin[1], pin[2]), Iterators.reverse(pins); init=x)
end

# How Ω meets one facet region's own face: `:full`, `:cut` or `:fictitious`, the
# same three verdicts `classify_cell` returns on a cell box, reached on the
# region's face box against `slice`. It is the classifier's verdict with exactly
# the classifier's resolution — the Lipschitz certificate where a leaf carries a
# constant, octree-bounded corner sampling where it does not — and in particular it
# is not a sampling of the *rule* on the region: a fictitious sliver of face is seen
# whenever the classifier resolves it, however the rule's points happen to fall, and
# missed when it is finer than the octree budget, which is the same blind spot the
# cell classification behind the face has.
#
# Three cases, by dispatch rather than by branch. A space with no `PhysicalDomain`
# has nothing outside Ω to find. A codim-`D` facet is a single point, with no box
# to classify, so its membership in Ω *is* the verdict — the slice's coordinate
# space is `ℝ⁰` and the lone point of it is the facet's own corner. Everything else
# classifies the face box.
_classify_facet(::Nothing, lower, upper, free_axes, ::Val) = :full

function _classify_facet(slice::PhysicalDomain, lower::SVector{D,T}, upper::SVector{D,T},
                         free_axes::Vector{Int}, ::Val{F}) where {D,T,F}
    return classify_cell(slice, _facet_box(lower, upper, free_axes, Val(F)))
end

function _classify_facet(slice::PhysicalDomain, ::SVector{D,T}, ::SVector{D,T}, ::Vector{Int},
                         ::Val{0}) where {D,T}
    return _inside(slice.geometry, SVector{0,T}()) ? :full : :fictitious
end

# The quadrature of one merged facet region, and the classification of the face it
# covers. The Jacobian for a codim-K facet sub-rectangle is `vol_free / 2^(D-K)`,
# matching the volume convention `vol / 2^D` from `reference_to_physical`. The rule
# comes from the region's own parents through `_facet_quadrature_counts`, exactly
# as it did per candidate — a merged region carries the parent set its candidates
# shared, so the merge changes which points exist, never how they are sized. It
# covers the whole face whatever `kind` says about how much of that face is in Ω.
function _facet_region(V::Space{D,T}, sides::Vector{Tuple{Int,Symbol}},
                       parents::Vector{FacetParent{D,T}}, free_axes::Vector{Int},
                       lower::SVector{D,T}, upper::SVector{D,T}, normal::SVector{D,T},
                       slice::Union{Nothing,PhysicalDomain}, ::Val{F}) where {D,T,F}
    kind = _classify_facet(slice, lower, upper, free_axes, Val(F))
    counts = _facet_quadrature_counts(V, parents, free_axes)
    jacobian = if isempty(free_axes)
        one(T)
    else
        prod(upper[d] - lower[d] for d in free_axes) / convert(T, 2^length(free_axes))
    end

    physical_points = SVector{D,T}[]
    physical_weights = T[]
    for (eta, weight) in _facet_reference_quadrature(counts, T)
        x = MVector{D,T}(lower)
        for (j, d) in pairs(free_axes)
            axis_mid = (lower[d] + upper[d]) / 2
            axis_half = (upper[d] - lower[d]) / 2
            x[d] = axis_mid + axis_half * eta[j]
        end
        push!(physical_points, SVector{D,T}(x))
        push!(physical_weights, weight * jacobian)
    end
    return FacetRegion{D,T}(sides, lower, upper, parents, kind, physical_points, physical_weights,
                            normal)
end

# ── Boundary trace evaluation ─────────────────────────────────────────────────

"""
    boundary_trace_indices(level, cell, raw_dofs, sides) -> NamedTuple

The datum-independent, *point*-independent half of a boundary trace: which of
`level`'s local basis functions have support on the codim-K facet identified by
`sides`. Returns a `NamedTuple` with four fields:

  - `raw_dofs::Vector{Int}` — the subset of `raw_dofs` whose basis functions
    touch the facet (filtered by `is_facet_basis`), in the original
    local-basis order.
  - `local_ids::Vector{CartesianIndex{D}}` — their tensor-product multi-indices,
    in the same order.
  - `values::Vector{T}` — the buffer [`boundary_trace_values!`](@ref) refills
    at each quadrature point; undefined until it does.
  - `factors::NTuple{D,Vector{T}}` — the per-axis 1D scratch that same call
    refills; sized for the level's full mode range, not the facet subset,
    because the family fills whole axes.

The subset depends only on `(level, sides)`, and a [`FacetRegion`](@ref)'s
parent list is constant across its quadrature points, so one call per parent
covers a whole region — the trace's per-point cost is then the tensor product
alone. The per-axis scratch the tensor product needs is allocated here too, for
the same reason.

This half is basis-family-agnostic: it asks only for [`cell_basis_indices`](@ref)
and `is_facet_basis`, which every family supplies. The per-axis scratch is sized
from the level's nominal order, which is an upper bound on every cell's.
"""
function boundary_trace_indices(level::Level{D,T}, cell::CartesianIndex{D}, raw_dofs::Vector{Int},
                                sides::Vector{Tuple{Int,Symbol}}) where {D,T}
    # The parent `cell` selects the level's minimum-rule index list for that cell,
    # which is what `raw_dofs` was built from. Taking the level-wide list instead
    # would pair the trace modes with the wrong raws on a per-cell-order level —
    # silently, because the pairing is positional and unchecked.
    local_ids = cell_basis_indices(level, cell)
    raws = Int[]
    ids = CartesianIndex{D}[]

    for (i, local_id) in pairs(local_ids)
        is_facet_basis(level.basis, local_id, sides) || continue
        push!(raws, raw_dofs[i])
        push!(ids, local_id)
    end

    return (; raw_dofs=raws, local_ids=ids, values=Vector{T}(undef, length(ids)),
            factors=_factor_buffers(nominal_order(level), T))
end

"""
    boundary_trace_values!(level, trace, xi) -> trace

The per-point half of a boundary trace: refill `trace.values` with the basis
traces at reference point `xi` on the parent cell, for the facet-incident modes
[`boundary_trace_indices`](@ref) selected. `trace` is that call's record,
extended by the caller with the parent's `cell`. Mirrors the
`_parent_basis_data` / `_update_parent_basis_values!` pair the volume
evaluation paths use.

The trace of a tensor-product basis function on a facet is the volume tensor
product evaluated at a point that happens to lie on the facet, so this is
`_tensor_values!` restricted to the facet-incident multi-indices — one code
path for every basis family, reaching the family only through its
`_fill_factor_tables!` hook. `sides` is already spent by
[`boundary_trace_indices`](@ref) and plays no part here.
"""
function boundary_trace_values!(level::Level{D,T}, trace, xi::SVector{D,T}) where {D,T}
    _tensor_values!(level.basis, trace.values, trace.local_ids, nominal_order(level), xi,
                    trace.factors, trace.cell)
    return trace
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
# where `φ_i` are the boundary traces of the physically-constrained raw
# dofs that survived every artificial elimination — the overlay-boundary
# condition and, under leaf semantics, the shedding of buried high-order modes
# and the linear dedup (`elimination_source === :free`). An eliminated dof has
# no value to project: its coefficient is not a solved unknown of the
# system. The mass matrix is symmetric positive (semi-)definite, so we
# try a Cholesky factorisation first and fall back to a pseudoinverse on
# the rare cases where Cholesky reports indefiniteness (degenerate
# facets, zero-area boundaries under heavy masking).
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
with the active cells on the boundary.

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
    DirichletProjection{D,T}(facets, unknowns, factors, samples)

Everything in the L² Dirichlet projection `M c = b` that the prescribed
data `g` do not enter: the per-component unknown sets, the per-component
factorised boundary mass, and the per-facet trace samples a right-hand
side is accumulated from.

Fields:

  - `facet_count::Int` — the number of (condition, facet) pairs the
    projection was built for. A condition list of a different shape
    cannot reuse it. Named for the count rather than for the facets
    themselves because `facets` is, everywhere else on this path, the
    [`FacetResolver`](@ref) the facets are resolved through.
  - `unknowns::Vector{Vector{Int}}` — per component, the raw dofs the
    projection solves for: the dofs of that component carrying a
    physical Dirichlet condition and no artificial elimination
    (`elimination_source === :free`, which excludes the overlay boundary
    as well as covered-mode pruning and the linear dedup).
  - `factors::Vector{DirichletFactor{T}}` — per component, the solved
    boundary mass (see `DirichletFactor`).
  - `samples::Vector{FacetTraceSamples{D,T}}` — one entry per
    (condition, facet) pair, in the order
    `condition × _facets(condition.boundary)` enumerates them. Empty
    when no component has an unknown, since then nothing is integrated.

# Validity

A projection is valid for the `(layout, space, condition-structure)`
triple it was built from. The unknown sets, the facet count and the mass
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
    facet_count::Int
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
#
# `facets` is the model's [`FacetResolver`](@ref): the boundary walk below reads
# the *same* resolved faces the assembly path and `boundary_integral` read, so
# the operator and the projected datum can never integrate different facets.
function _dirichlet_projection(layout::DofLayout{D,T}, V::Space{D,T}, facets::FacetResolver{D,T},
                               dirichlet) where {D,T}
    ncomp = layout.components
    facet_count = _dirichlet_facet_count(dirichlet, Val(D))
    unknowns = [[raw
                 for raw in eachindex(layout.raw_keys)
                 if layout.elimination_source[raw] === :free && layout.physical_dirichlet[raw, c]]
                for c in 1:ncomp]
    # Nothing is constrained anywhere: no mass to accumulate and no facet to walk,
    # so the projection is the empty one its `facet_count` still has to carry.
    if all(isempty, unknowns)
        return DirichletProjection{D,T}(facet_count, unknowns,
                                        DirichletFactor{T}[nothing for _ in 1:ncomp],
                                        FacetTraceSamples{D,T}[])
    end

    index = [Dict(raw => i for (i, raw) in pairs(unknowns[c])) for c in 1:ncomp]
    # The constrained-boundary mass couples only the physically-Dirichlet dofs
    # of one component and is dense-solved below, so each component's is
    # accumulated directly into a dense matrix — no sparse intermediate.
    mass = [zeros(T, length(unknowns[c]), length(unknowns[c])) for c in 1:ncomp]

    samples = FacetTraceSamples{D,T}[]
    for condition in dirichlet
        for sides in _facets(condition.boundary, Val(D))
            push!(samples,
                  _sample_dirichlet_facet!(mass, index, layout, V, facets, condition, sides))
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

    return DirichletProjection{D,T}(facet_count, unknowns, factors, samples)
end

# Walk one condition's facet: the boundary regions on that facet × the
# per-region quadrature points. At each quadrature point, evaluate the
# boundary traces of every covering parent once, record them as a sample, and
# accumulate the mass-matrix contribution of every component the condition
# constrains.
function _sample_dirichlet_facet!(mass::Vector{Matrix{T}}, index::Vector{Dict{Int,Int}},
                                  layout::DofLayout{D,T}, V::Space{D,T}, facets::FacetResolver{D,T},
                                  condition::DirichletCondition,
                                  sides::Vector{Tuple{Int,Symbol}}) where {D,T}
    ncomp = layout.components
    points = SVector{D,T}[]
    weights = T[]
    offsets = Int[1]
    values = T[]
    rows = [Int[] for _ in 1:ncomp]

    for region in _resolve_face(facets, V, sides)
        # Everything about a parent's trace except the point: the level lookup,
        # the cell's dof list, and the facet-incident subset of its local basis.
        # `region.parents` is constant across the region's quadrature points by
        # the `FacetRegion` contract, and the subset depends only on
        # `(level, sides)`, so all of it is resolved once here instead of at
        # every point below. `box` is the parent's precomputed physical box, so
        # the only per-Q-point geometry work left is the affine
        # `physical_to_reference`. The levels stay in their own vector: a space
        # may mix basis families across levels, and holding one in the record
        # would make the record type — and with it the mass double loop below,
        # which reads nothing family-specific — vary per parent.
        levels = map(parent -> _level_by_id(V, parent.level), region.parents)
        traces = map(levels, region.parents) do level, parent
            all_raw_dofs = cell_dofs(layout, parent.level, parent.cell)
            (; cell=parent.cell, box=parent.parent_box,
             boundary_trace_indices(level, parent.cell, all_raw_dofs, sides)...)
        end

        for (qp, x) in pairs(region.points)
            qweight = region.weights[qp]

            # Refresh every parent's trace values at this point. The buffers
            # live on `traces`, so the sample record and the mass double loop
            # below all read the same point from every parent.
            for (i, trace) in pairs(traces)
                boundary_trace_values!(levels[i], trace, physical_to_reference(trace.box, x))
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

# The [`FacetResolver`](@ref) a Dirichlet projection resolves its faces through:
# the caller's when it supplied one, a private throw-away keyed on the layout's
# own tolerance otherwise (the standalone `dof_layout(V; dirichlet)` call, which
# has no model to share with).
#
# A tolerance mismatch is a caller bug and is raised as one. A memo's entries
# were merged against *its* tolerance, so reading them with a layout built at
# another one would project the datum over a different partition of ∂Ω than the
# one the layout's constrained dofs were detected on. Quietly substituting a
# fresh resolver instead would hide that behind a second resolution of every
# face — exactly the cost sharing the resolver exists to remove.
function _dirichlet_resolver(layout::DofLayout{D,T},
                             facets::Union{Nothing,FacetResolver{D,T}}) where {D,T}
    facets === nothing && return FacetResolver{D,T}(layout.tolerance)
    facets.tolerance == layout.tolerance ||
        throw(ArgumentError("facet resolver tolerance $(facets.tolerance) does not match the dof " *
                            "layout's $(layout.tolerance); a layout and the resolver it projects " *
                            "through must come from the same prepare"))
    return facets
end

# Refill `layout.constrained_values` from `dirichlet`, returning the
# [`DirichletProjection`](@ref) used so a caller that re-projects the same
# layout can hand it back and skip the boundary walk. `projection` is reused
# when given; the facet-count check is a cheap guard for a caller that reaches
# past `update_dirichlet!`'s structural check, which is what actually
# establishes that a cached projection still matches the condition list.
#
# `facets` is the resolver the boundary walk resolves faces through, or `nothing`
# to resolve through a private one; see `_dirichlet_resolver`.
function _project_dirichlet_values!(layout::DofLayout{D,T}, V::Space{D,T},
                                    facets::Union{Nothing,FacetResolver{D,T}}, dirichlet,
                                    projection::Union{Nothing,DirichletProjection{D,T}}=nothing) where {D,
                                                                                                        T}
    ncomp = layout.components
    plan = if projection === nothing ||
              projection.facet_count != _dirichlet_facet_count(dirichlet, Val(D))
        _dirichlet_projection(layout, V, _dirichlet_resolver(layout, facets), dirichlet)
    else
        projection
    end
    # Only now clear the old values. Everything that can reject the call — the
    # resolver's tolerance above all — has happened, so a rejected call leaves the
    # layout exactly as it found it instead of zeroed and unprojected.
    fill!(layout.constrained_values, zero(T))
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
