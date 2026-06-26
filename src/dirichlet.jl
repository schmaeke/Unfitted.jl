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
        coords = T[]
        for level in touching_levels
            append!(coords, boundary_coordinates(level.mesh)[d])
        end
        _intervals_from_coordinates(merge_coordinates(coords, tol), tol)
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
# Walk every Dirichlet condition × its facets × the boundary regions on
# that facet × the per-region quadrature points. At each quadrature
# point: assemble the mass-matrix and RHS contributions over the
# constrained dofs from every covering parent. The
# `_test_value_contribution` and `_as_test_channels` helpers are
# defined in `assembly.jl`'s channel layer and reused here so the
# Dirichlet projection rides the same channel calculus the volume
# assembly uses; the forward reference resolves at call time
# (`prepare(problem)` loads every source file first).
function _project_dirichlet_values!(layout::DofLayout{D,T}, V::Space{D,T}, dirichlet) where {D,T}
    fill!(layout.constrained_values, zero(T))

    # Index the raw dofs that need a Dirichlet value (physically
    # constrained on at least one component, and not eliminated by an
    # overlay constraint — the overlay-constrained dofs are homogeneous
    # by construction).
    unknown_raws = [raw
                    for raw in eachindex(layout.raw_keys)
                    if !layout.overlay_constraint[raw] &&
                       any(c -> layout.physical_dirichlet[raw, c], 1:layout.components)]
    isempty(unknown_raws) && return layout

    raw_to_projection = Dict(raw => i for (i, raw) in pairs(unknown_raws))
    nproj = length(unknown_raws)
    # The constrained-boundary mass couples only the physically-Dirichlet
    # dofs (a small set) and is dense-solved below, so it is accumulated
    # directly into a dense matrix — no sparse intermediate.
    mass = zeros(T, nproj, nproj)
    rhs = zeros(T, nproj, layout.components)

    for condition in dirichlet
        for sides in _facets(condition.boundary, Val(D))
            for region in _boundary_facet_regions(V, sides, layout.tolerance)
                for (qp, x) in pairs(region.points)
                    qweight = region.weights[qp]

                    # Evaluate every parent's boundary traces at this
                    # point once and reuse them for both the RHS and
                    # mass-matrix contributions below. `parent.parent_box`
                    # is precomputed on the region, so the only per-Q-point
                    # geometry work is the affine `physical_to_reference`.
                    traces = map(region.parents) do parent
                        level = _level_by_id(V, parent.level)
                        raw_dofs = cell_dofs(layout, parent.level, parent.cell)
                        xi = physical_to_reference(parent.parent_box, x)
                        boundary_trace_data(level, raw_dofs, sides, xi, parent.cell)
                    end

                    # RHS: ∫_∂Ω g v dx contributions for every component
                    # this condition applies to.
                    for component in 1:layout.components
                        (condition.component === nothing || condition.component == component) ||
                            continue
                        g = convert(T, _condition_value(condition, x, component))
                        rhs_channels = _as_test_channels(g, Val(D), T)

                        for test_trace in traces
                            for a in eachindex(test_trace.raw_dofs)
                                row = get(raw_to_projection, test_trace.raw_dofs[a], 0)
                                row == 0 && continue
                                rhs[row, component] += qweight *
                                                       _test_value_contribution(rhs_channels,
                                                                                test_trace.values[a])
                            end
                        end
                    end

                    # Mass matrix: ∫_∂Ω φ_i φ_j dx contributions over
                    # the constrained-dof subspace. Both indices iterate
                    # over the same `traces`, so the resulting matrix is
                    # symmetric by construction.
                    for trial_trace in traces
                        for b in eachindex(trial_trace.raw_dofs)
                            col = get(raw_to_projection, trial_trace.raw_dofs[b], 0)
                            col == 0 && continue
                            mass_channels = _as_test_channels(trial_trace.values[b], Val(D), T)

                            for test_trace in traces
                                for a in eachindex(test_trace.raw_dofs)
                                    row = get(raw_to_projection, test_trace.raw_dofs[a], 0)
                                    row == 0 && continue
                                    mass[row, col] += qweight *
                                                      _test_value_contribution(mass_channels,
                                                                               test_trace.values[a])
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # Solve `M c = b` for every component. Cholesky first; pseudoinverse
    # fallback on the rare indefinite case (degenerate / zero-area facets
    # under heavy masking).
    factor = cholesky(Symmetric(mass); check=false)
    projected = issuccess(factor) ? factor \ rhs : pinv(mass) * rhs

    for (i, raw) in pairs(unknown_raws)
        for component in 1:layout.components
            layout.constrained_values[raw, component] = projected[i, component]
        end
    end

    return layout
end

# True iff `dirichlet` carries at least one nonzero condition. The
# homogeneous-zero case skips the projection: every constrained value
# already equals zero from the layout's zero-initialised
# `constrained_values` matrix.
function _needs_dirichlet_projection(dirichlet)
    return any(condition -> !(condition.value isa Number && iszero(condition.value)), dirichlet)
end

# True iff `a` and `b` pick out the same physical facet. The default
# `==` on a struct that carries a `Vector` field falls back to `===`,
# so we unfold by hand: Symbol `===` short-circuits, then `Vector ==`
# does the element-by-element compare on the `(axis, side)` tuples
# (which are isbits, so `==` is bit-equality). Used by
# `update_dirichlet!`'s structural check; the function itself lives
# in `src/model.jl`.
function _selectors_equal(a::BoundarySelector, b::BoundarySelector)
    return a.selector === b.selector && a.sides == b.sides
end

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
