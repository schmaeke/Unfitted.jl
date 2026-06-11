# Degree-of-freedom layout for a superposition `Space`. The dof layer
# answers, for each cell, two questions assembly needs:
#
#   * which local basis functions map to which raw (pre-constraint) dof
#     id — produced by walking every level's cells and reusing dof ids
#     across adjacent cells through a `TensorDofKey` lookup;
#   * which raw dofs are constrained, by what kind of constraint, and
#     (for nonzero physical Dirichlet data) to what value.
#
# Two kinds of constraint are tracked separately and never mixed:
#
#   * Physical Dirichlet — user-imposed boundary conditions on the
#     physical domain `∂Ω`. Tracked per-component so vector fields can
#     pin only one component along a face (rollers, symmetry, …).
#     Nonzero data are projected onto the boundary trace space by an
#     L² mass-matrix solve so the constrained value list reflects the
#     correct trace at every constrained dof.
#   * Artificial overlay constraint — homogeneous Dirichlet on the
#     overlay's artificial boundary `Γ_o^(k) = ∂Ω^(k) \ ∂Ω`. Strongly
#     eliminated for the integrated Legendre basis by removing the
#     corresponding raw dof from active enumeration. Also applied at
#     internal active/inactive cell faces of the same level when a
#     `LevelMask` is in play.
#
# Active dofs are enumerated component-major (component 1 of every raw
# first, then component 2, …) so the all-or-nothing constrained case
# matches the natural block ordering, and per-component constraints
# simply leave holes in the enumeration.
#
# This file owns the dof keys, the layout structs, the structural
# overlay-constraint detection, and the active-enumeration constructor.
# The physical-Dirichlet spec types, per-key Dirichlet detection, and
# the L² boundary projection live in `dirichlet.jl`. `dof_layout`
# forward-references `_has_physical_dirichlet` and
# `_project_dirichlet_values!`; the references resolve at call time
# (when `prepare(problem)` runs, after every source file is loaded).

# ── Dof keys and layout struct ────────────────────────────────────────────────

# Per-axis component of a `TensorDofKey`. Together with the axis index it
# uniquely identifies one factor of a tensor-product basis function:
#
#   * `kind = _AXIS_NODE` ⇒ an endpoint-mode contribution. `index` is the
#     1-based node coordinate index inside the level's `mesh.axes[axis]`
#     vector; `mode` is `0` (the endpoint contribution is the same shape
#     function regardless of which mode triggered it).
#   * `kind = _AXIS_SPAN` ⇒ a bubble-mode contribution. `index` is the
#     1-based cell index along the axis; `mode ≥ 2` selects which bubble.
#
# Stored as a flat `UInt8 + Int + Int` rather than a sum type so the key
# is fully isbits and hashes cheaply for the cross-cell dof reuse cache.
struct AxisDofKey
    kind::UInt8
    index::Int
    mode::Int
end

# Tensor-product dof key: one `AxisDofKey` per axis plus the source
# level's id. Two basis functions share a raw dof iff they have the same
# `TensorDofKey` — i.e. they live on the same level and reduce to the
# same per-axis (kind, index, mode) along every axis. The node/span
# distinction handles the dof reuse correctly: adjacent cells share the
# right-endpoint of cell `i` with the left-endpoint of cell `i + 1` (both
# resolve to the same `_AXIS_NODE, index = i + 1` factor), while bubble
# modes are cell-local (each cell's `_AXIS_SPAN` index is unique).
struct TensorDofKey{D}
    level::Int
    axes::NTuple{D,AxisDofKey}
end

"""
    DofLayout{D,T}

Basis-aware global dof numbering for a single field over a superposition
[`Space`](@ref). Constrained and active dofs are kept explicitly
distinguishable: raw dofs are enumerated first (every basis function
across every level and every active cell), then constraints are detected
and the active enumeration assigns positive ids to the unconstrained
dofs only.

Fields:

  - `components::Int` — number of scalar channels the field carries
    (1 for scalar fields, > 1 for vector fields). Constraints are
    tracked per-component so vector boundary conditions can pin
    individual channels.
  - `cell_dofs_by_level::Vector{Array{Vector{Int},D}}` — one entry per
    level; each entry is a `D`-dimensional array of cell-local raw dof
    id vectors, indexed by the level's `CartesianIndex`. Inactive cells
    carry an empty vector.
  - `raw_keys::Vector{TensorDofKey{D}}` — `raw_keys[i]` is the
    `TensorDofKey` that produced raw dof `i`. The inverse of the
    construction dictionary.
  - `active_component::Matrix{Int}` — `(raw, component) → active id`.
    Entry is `0` when component `component` of raw dof `raw` is
    constrained.
  - `physical_dirichlet::Matrix{Bool}` — `(raw, component) → true` iff
    component `component` of raw dof `raw` has a physical Dirichlet
    constraint.
  - `overlay_constraint::Vector{Bool}` — `raw → true` iff raw dof `raw`
    carries an artificial overlay constraint. Overlay constraints
    apply to every component simultaneously (the overlay function is
    constrained to vanish on its artificial boundary).
  - `constrained_values::Matrix{T}` — `(raw, component) → value` for
    constrained dofs. Filled by [`_project_dirichlet_values!`](@ref)
    for nonzero physical Dirichlet data; zero otherwise.
  - `active_count::Int` — total number of active (i.e. enumerated) dofs.
  - `tolerance::GeometryTolerance{T}` — tolerance used during boundary
    detection.

Use [`dof_layout`](@ref) to construct.
"""
struct DofLayout{D,T<:Real}
    components::Int
    cell_dofs_by_level::Vector{Array{Vector{Int},D}}
    raw_keys::Vector{TensorDofKey{D}}
    active_component::Matrix{Int}
    physical_dirichlet::Matrix{Bool}
    overlay_constraint::Vector{Bool}
    constrained_values::Matrix{T}
    active_count::Int
    tolerance::GeometryTolerance{T}
end

# Tag values for `AxisDofKey.kind`. `UInt8` keeps the key compact and
# fast to hash; the values themselves are arbitrary.
const _AXIS_NODE = UInt8(0)
const _AXIS_SPAN = UInt8(1)

# Build the per-axis dof key for cell `cell_axis` and 1D mode `mode` in
# the integrated Legendre basis. Modes 0 and 1 are the two endpoint
# shape functions — they map to `_AXIS_NODE` factors at the cell's
# lower-endpoint coordinate index (`cell_axis`) and upper-endpoint index
# (`cell_axis + 1`) respectively. Mode m ≥ 2 is a cell-local bubble and
# maps to an `_AXIS_SPAN` factor at index `cell_axis` carrying the mode
# number for cross-cell distinguishability.
function _axis_dof_key(cell_axis::Int, mode::Int)
    mode == 0 && return AxisDofKey(_AXIS_NODE, cell_axis, 0)
    mode == 1 && return AxisDofKey(_AXIS_NODE, cell_axis + 1, 0)
    return AxisDofKey(_AXIS_SPAN, cell_axis, mode)
end

# Build the full `TensorDofKey` for one tensor-product basis function
# `local_id` on `cell` of `level`. The integrated Legendre specialisation
# above produces the per-axis keys; a fallback method throws for any
# other basis family so the dof layer fails early instead of producing
# silently-wrong keys.
function _tensor_dof_key(level::Level{D,T,<:IntegratedLegendre}, cell::CartesianIndex{D},
                         local_id::CartesianIndex{D}) where {D,T}
    axes = ntuple(d -> _axis_dof_key(cell.I[d], local_id.I[d]), D)
    return TensorDofKey{D}(level.id, axes)
end

function _tensor_dof_key(level::Level{D,T,B}, cell::CartesianIndex{D},
                         local_id::CartesianIndex{D}) where {D,T,B}
    throw(ArgumentError("dof layout is not implemented for basis family $(basis_name(level.basis))"))
end

# Build the per-cell raw dof id vector for one cell. For every local
# basis function on the cell, compute its `TensorDofKey` and either
# reuse the raw id already registered for that key (the cross-cell dof
# reuse path: adjacent cells share endpoint nodes, identical levels
# share interior modes) or register a fresh id by appending the key to
# `raw_keys`.
function _build_cell_dofs!(raw_by_key::Dict{TensorDofKey{D},Int}, raw_keys::Vector{TensorDofKey{D}},
                           level::Level{D}, cell::CartesianIndex{D}) where {D}
    local_ids = local_basis_indices(level.basis, level.order, level.mode)
    raw_ids = Vector{Int}(undef, length(local_ids))

    for (a, local_id) in pairs(local_ids)
        key = _tensor_dof_key(level, cell, local_id)
        raw_ids[a] = get!(raw_by_key, key) do
            push!(raw_keys, key)
            length(raw_keys)
        end
    end

    return raw_ids
end

# ── Geometric helpers shared with `dirichlet.jl` ──────────────────────────────

# Tolerance-aware coordinate equality used throughout boundary
# detection. Pulled out so the `<= tol.contain` convention is in one
# place; consumed both by the overlay-constraint detection below and by
# the physical-Dirichlet detection in `dirichlet.jl`.
_coordinate_matches(a::Real, b::Real, tol::GeometryTolerance) = abs(a - b) <= tol.contain

# Coordinate of a `_AXIS_NODE` dof key along `axis`. Returns `nothing`
# for `_AXIS_SPAN` keys (which don't sit at a specific axis coordinate).
# Used by the physical-side detection in `dirichlet.jl`.
function _node_coordinate(level::Level, key::TensorDofKey{D}, axis::Integer) where {D}
    axis_key = key.axes[axis]
    axis_key.kind == _AXIS_NODE || return nothing
    return level.mesh.axes[axis][axis_key.index]
end

# ── Artificial overlay-boundary detection ─────────────────────────────────────

# True iff the `_AXIS_NODE` factor of the key sits at the level mesh's
# lower (`side = :lower`, index 1) or upper (`side = :upper`, index
# `cells[axis] + 1`) extent along `axis`.
function _key_on_level_side(key::TensorDofKey{D}, level::Level{D}, axis::Integer,
                            side::Symbol) where {D}
    axis_key = key.axes[axis]
    axis_key.kind == _AXIS_NODE || return false
    side === :lower && return axis_key.index == 1
    side === :upper && return axis_key.index == level.mesh.cells[axis] + 1
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# True iff the level's mesh-domain boundary on the given axis-side
# coincides with the physical-domain boundary up to `tol.contain`. Used
# by `_has_overlay_constraint` to decide whether a mesh-boundary face is
# a physical face (no overlay constraint) or an artificial overlay face
# (homogeneous Dirichlet constraint).
function _level_side_is_physical(level::Level{D,T}, domain::AxisBox{D,T}, axis::Integer,
                                 side::Symbol, tol::GeometryTolerance{T}) where {D,T}
    side === :lower &&
        return _coordinate_matches(level.mesh.domain.lower[axis], domain.lower[axis], tol)
    side === :upper &&
        return _coordinate_matches(level.mesh.domain.upper[axis], domain.upper[axis], tol)
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# True iff the cell whose axis-`axis` index is `pos` (with the other-axis
# indices drawn from `outer`, ignoring the axis-`axis` slot) is in bounds
# and active under the level's `LevelMask`. Used by `_on_active_face` to
# probe both sides of a candidate face for active/inactive transitions.
function _cell_active(level::Level{D}, axis::Integer, pos::Integer, outer::NTuple{D,Int},
                      n::NTuple{D,Int}) where {D}
    1 <= pos <= n[axis] || return false
    cell_tuple = ntuple(d -> d == axis ? pos : outer[d], D)
    return is_active(level.mask, CartesianIndex(cell_tuple))
end

# True iff the `_AXIS_NODE` factor of `key` at axis `axis`, node index
# `i`, lies on a face that separates an active cell from an
# off-mesh-or-inactive cell on the other side. "Perpendicular" positions
# in the other axes come from the key's own incidence:
#
#   * a `_AXIS_NODE` factor at index `j` contributes cells `j − 1` and
#     `j` (whichever are in bounds), since a node sits between two
#     adjacent cells along its axis;
#   * a `_AXIS_SPAN` factor at index `k` contributes only cell `k`,
#     since a span (bubble) mode is cell-local.
#
# For each perpendicular `outer` position, sample the activity of the
# cells immediately below (`i − 1`) and above (`i`) the candidate face
# along `axis`. A change in activity ⇒ the face is between an active
# and an inactive cell of the level, i.e. an active-region face.
function _on_active_face(key::TensorDofKey{D}, level::Level{D}, axis::Integer, i::Integer,
                         n::NTuple{D,Int}) where {D}
    ranges = ntuple(D) do d
        if d == axis
            1:1
        else
            kd = key.axes[d]
            if kd.kind == _AXIS_NODE
                j = kd.index
                max(1, j-1):min(n[d], j)
            else
                kd.index:kd.index
            end
        end
    end
    for outer in CartesianIndices(ranges)
        below_active = _cell_active(level, axis, i - 1, outer.I, n)
        above_active = _cell_active(level, axis, i, outer.I, n)
        below_active != above_active && return true
    end
    return false
end

# True iff `key` sits on the artificial boundary of the level's *active
# region* and therefore carries a homogeneous overlay constraint. The
# active region is the union of active cells; its boundary includes
#
#   * the mesh-box face on every axis where the level's mesh boundary
#     does not coincide with the physical-domain boundary (i.e. the
#     artificial overlay boundary `Γ_o^(k) = ∂Ω^(k) \ ∂Ω`);
#   * internal faces between active and inactive cells of the *same*
#     level (the `LevelMask` boundary inside the level).
#
# With `mask === nothing` the active region equals the level's mesh
# box, only mesh-box faces matter, and this reduces to the original
# overlay-boundary rule from the paper.
function _has_overlay_constraint(key::TensorDofKey{D}, level::Level{D,T}, domain::AxisBox{D,T},
                                 tol::GeometryTolerance{T}) where {D,T}
    n = level.mesh.cells
    for axis in 1:D
        axis_key = key.axes[axis]
        axis_key.kind == _AXIS_NODE || continue
        i = axis_key.index

        # Only nodes that actually sit on an active-region face matter.
        _on_active_face(key, level, axis, i, n) || continue

        # Three cases for an active-region face along this axis:
        #   i == 1            — mesh's lower-axis face. Physical iff the
        #                       level's lower bound coincides with the
        #                       physical domain's lower bound; otherwise
        #                       artificial.
        #   i == n[axis] + 1  — mesh's upper-axis face, symmetric.
        #   otherwise         — internal face between active and inactive
        #                       cells of the same level, always artificial.
        if i == 1
            _level_side_is_physical(level, domain, axis, :lower, tol) || return true
        elseif i == n[axis] + 1
            _level_side_is_physical(level, domain, axis, :upper, tol) || return true
        else
            return true
        end
    end
    return false
end

# ── DofLayout construction and accessors ──────────────────────────────────────

# Active global id of component `component` of raw dof `raw`, or 0 if
# the (raw, component) is constrained.
function _active_component_dof(layout::DofLayout, raw::Integer, component::Integer)
    layout.active_component[raw, component]
end

"""
    dof_layout(V::Space; dirichlet=[], tolerance=GeometryTolerance(T), components=1) -> DofLayout

Construct the basis-aware global dof layout for a superposition
[`Space`](@ref). The construction proceeds in three stages:

  1. Walk every level's cells in order and build the cell-local raw dof
     id vector through `TensorDofKey` lookup. Adjacent cells share
     endpoint nodes; bubble (span) modes are cell-local. Inactive cells
     get an empty vector.
  2. For every raw dof, run the constraint detectors
     ([`_has_overlay_constraint`](@ref) and
     [`_has_physical_dirichlet`](@ref) per component) to fill the
     `overlay_constraint` and `physical_dirichlet` matrices.
  3. Enumerate active dofs component-major (component 1 first, then
     component 2, …) skipping every (raw, component) entry that is
     either physically Dirichlet-constrained or overlay-constrained.
     Project nonzero Dirichlet data onto the boundary trace space via
     [`_project_dirichlet_values!`](@ref).

Keyword arguments:

  - `dirichlet` — iterable of [`DirichletCondition`](@ref)s.
  - `tolerance` — `GeometryTolerance` used by boundary detection.
  - `components` — scalar channels per field (≥ 1).

The current implementation supports the integrated Legendre basis and
dof-wise homogeneous overlay constraints; other basis families are
expected to plug in via the [`_tensor_dof_key`](@ref) dispatch.
"""
function dof_layout(V::Space{D,T}; dirichlet=[], tolerance=GeometryTolerance(T),
                    components::Integer=1) where {D,T}
    components > 0 || throw(ArgumentError("dof layout components must be positive"))
    raw_by_key = Dict{TensorDofKey{D},Int}()
    raw_keys = TensorDofKey{D}[]
    cell_dofs_by_level = Vector{Array{Vector{Int},D}}(undef, length(V.levels))

    # Stage 1: walk every level's cells and assign raw dofs through the
    # `TensorDofKey` cache (cross-cell endpoint sharing happens here).
    for (level_index, level) in pairs(V.levels)
        level_cells = Array{Vector{Int},D}(undef, level.mesh.cells)
        for cell in cell_indices(level.mesh)
            if is_active(level.mask, cell)
                level_cells[cell] = _build_cell_dofs!(raw_by_key, raw_keys, level, cell)
            else
                level_cells[cell] = Int[]
            end
        end
        cell_dofs_by_level[level_index] = level_cells
    end

    # Stage 2: classify every raw dof against the two constraint kinds.
    ncomp = Int(components)
    nraw = length(raw_keys)
    physical_dirichlet = falses(nraw, ncomp)
    overlay_constraint = falses(nraw)

    for (raw, key) in pairs(raw_keys)
        level = _level_by_id(V, key.level)
        overlay_constraint[raw] = _has_overlay_constraint(key, level, V.domain, tolerance)
        for component in 1:ncomp
            physical_dirichlet[raw, component] = _has_physical_dirichlet(key, level, V.domain,
                                                                         dirichlet, tolerance,
                                                                         component)
        end
    end

    # Stage 3: enumerate active dofs component-major. For all-or-nothing
    # constraints this matches the natural block ordering; per-component
    # constraints simply leave holes in the enumeration.
    active_component = zeros(Int, nraw, ncomp)
    active_count = 0
    for component in 1:ncomp, raw in 1:nraw
        if !(physical_dirichlet[raw, component] || overlay_constraint[raw])
            active_count += 1
            active_component[raw, component] = active_count
        end
    end

    layout = DofLayout{D,T}(ncomp, cell_dofs_by_level, raw_keys, active_component,
                            physical_dirichlet, overlay_constraint, zeros(T, nraw, ncomp),
                            active_count, tolerance)

    # Stage 3b: project nonzero Dirichlet data onto the boundary trace
    # space. Zero-only conditions skip this — `constrained_values` is
    # already zero from the construction above.
    _needs_dirichlet_projection(dirichlet) && _project_dirichlet_values!(layout, V, dirichlet)
    return layout
end

"""
    raw_dof_count(layout::DofLayout) -> Int

Total number of *raw* dof slots (including constrained), counted as
`length(raw_keys) * components`.
"""
raw_dof_count(layout::DofLayout) = length(layout.raw_keys) * layout.components

"""
    active_unknowns(layout::DofLayout) -> Int

Number of *active* (post-constraint) dofs — the size of the global
system the solver sees.
"""
active_unknowns(layout::DofLayout) = layout.active_count

"""
    cell_dofs(layout, level::Integer, cell::CartesianIndex) -> Vector{Int}

Pre-constraint raw dof ids for the cell at `cell` on level `level`, in
the canonical local-basis order. Inactive cells return an empty vector.
"""
function cell_dofs(layout::DofLayout{D}, level::Integer, cell::CartesianIndex{D}) where {D}
    1 <= level <= length(layout.cell_dofs_by_level) ||
        throw(ArgumentError("unknown level id $level"))
    return layout.cell_dofs_by_level[level][cell]
end

"""
    active_cell_dofs(layout, level::Integer, cell::CartesianIndex, component=1) -> Vector{Int}

Active global ids for component `component` of the cell-local raw dofs,
with `0` standing for any dof that was eliminated by a constraint. The
order matches the canonical local-basis ordering (same length as
`cell_dofs(layout, level, cell)`).
"""
function active_cell_dofs(layout::DofLayout{D}, level::Integer, cell::CartesianIndex{D},
                          component::Integer=1) where {D}
    return [_active_component_dof(layout, raw, component) for raw in cell_dofs(layout, level, cell)]
end

"""
    constraint_kind(layout::DofLayout, raw, component=1) -> Symbol

Classify a single raw dof. Returns one of:

  - `:free`       — no constraint, the dof is enumerated.
  - `:dirichlet`  — physical Dirichlet only.
  - `:overlay`    — artificial overlay constraint only.
  - `:mixed`      — both. The overlay rule wins (the dof is eliminated
                    and its value is held in `constrained_values` from
                    the Dirichlet projection).
"""
function constraint_kind(layout::DofLayout, raw::Integer, component::Integer=1)
    physical = layout.physical_dirichlet[raw, component]
    overlay = layout.overlay_constraint[raw]
    physical && overlay && return :mixed
    physical && return :dirichlet
    overlay && return :overlay
    return :free
end

"""
    constrained_value(layout::DofLayout, raw, component=1) -> T

Stored value of a constrained (raw, component). Zero for overlay
constraints and homogeneous Dirichlet conditions; the projected value
for nonzero Dirichlet data.
"""
function constrained_value(layout::DofLayout, raw::Integer, component::Integer=1)
    layout.constrained_values[raw, component]
end

"""
    dof_value(layout::DofLayout, coefficients, raw, component=1) -> T

Recover the value of component `component` of raw dof `raw`. Returns the
constrained value when the dof is constrained, and `coefficients[active]`
otherwise — i.e. the unified accessor used by assembly and
post-processing when a basis-function contribution needs the actual dof
value regardless of constraint status.
"""
function dof_value(layout::DofLayout, coefficients, raw::Integer, component::Integer=1)
    active = _active_component_dof(layout, raw, component)
    return active == 0 ? constrained_value(layout, raw, component) : coefficients[active]
end

# ── FieldLayout and SystemLayout (multi-field problems) ───────────────────────

"""
    FieldLayout{D,T}(name, components, dofs, offset)

One field's slot inside a multi-field [`SystemLayout`](@ref). `dofs` is
the field's own [`DofLayout`](@ref); `offset` is the field's starting
column in the global active-dof vector (the active dofs of all earlier
fields take ids 1 through `offset`, and this field's active dofs take
ids `offset + 1` through `offset + active_unknowns(dofs)`).
"""
struct FieldLayout{D,T<:Real}
    name::Symbol
    components::Int
    dofs::DofLayout{D,T}
    offset::Int
end

"""
    SystemLayout{D,T}(fields, by_name, active_unknowns, tolerance)

Top-level layout for a problem with one or more fields. `fields` lists
the per-field [`FieldLayout`](@ref)s in declaration order; `by_name`
maps field names to their indices in `fields`; `active_unknowns` is the
sum of every field's active count (the global system size); `tolerance`
is shared across the fields.

Most consumers go through [`active_cell_dofs`](@ref) and friends, which
delegate to the per-field layout transparently. Single-field problems
take the implicit-field shortcut where the field name does not need to
be passed.
"""
struct SystemLayout{D,T<:Real}
    fields::Vector{FieldLayout{D,T}}
    by_name::Dict{Symbol,Int}
    active_unknowns::Int
    tolerance::GeometryTolerance{T}
end

# Look up the per-field layout by name. Throws on unknown name.
function _field_layout(system::SystemLayout, name::Symbol)
    index = get(system.by_name, name, 0)
    index == 0 && throw(ArgumentError("unknown field $name"))
    return system.fields[index]
end

# Global active dof id for component `component` of raw `raw` in this
# field, or 0 if the (raw, component) is constrained. Per-field active
# ids are shifted by the field's `offset` so they live in disjoint
# blocks of the global active enumeration.
function _field_component_dof(layout::FieldLayout, raw::Integer, component::Integer)
    (active=_active_component_dof(layout.dofs, raw, component);
     active == 0 ? 0 : layout.offset + active)
end

"""
    dof_value(layout::FieldLayout, coefficients, raw, component=1) -> T

Recover the value of component `component` of raw `raw` in this field,
using the global active enumeration. Equivalent to the
[`DofLayout`](@ref) version but resolves the per-field offset so
`coefficients` is the global active vector.
"""
function dof_value(layout::FieldLayout, coefficients, raw::Integer, component::Integer=1)
    global_dof = _field_component_dof(layout, raw, component)
    return global_dof == 0 ? constrained_value(layout.dofs, raw, component) :
           coefficients[global_dof]
end

"""
    active_unknowns(layout::SystemLayout) -> Int

Total number of active dofs across every field — the size of the global
system the solver sees.
"""
active_unknowns(layout::SystemLayout) = layout.active_unknowns

"""
    raw_dof_count(layout::SystemLayout) -> Int

Total number of raw dof slots across every field (sum of
`raw_dof_count(field.dofs)`).
"""
raw_dof_count(layout::SystemLayout) = sum(raw_dof_count(field.dofs) for field in layout.fields)

# Resolve the unique field of a single-field layout. Used by the
# `*_dofs` delegates below so single-field problems do not have to
# carry a field name argument.
function _only_field_layout(layout::SystemLayout)
    length(layout.fields) == 1 ||
        throw(ArgumentError("field argument is required for multi-field layouts"))
    return only(layout.fields)
end

# `cell_dofs`/`active_cell_dofs`/`constraint_kind`/`constrained_value`
# delegates that thread through `_only_field_layout` for single-field
# problems or through `_field_layout(name)` for the multi-field call
# forms. The explicit overloads keep grep-friendly call sites and
# preserve the per-field offset translation.

function cell_dofs(layout::SystemLayout, level::Integer, cell::CartesianIndex)
    cell_dofs(_only_field_layout(layout).dofs, level, cell)
end
function active_cell_dofs(layout::SystemLayout, level::Integer, cell::CartesianIndex)
    active_cell_dofs(_only_field_layout(layout).dofs, level, cell)
end

function active_cell_dofs(layout::SystemLayout, level::Integer, cell::CartesianIndex,
                          component::Integer)
    field = _only_field_layout(layout)
    return [_field_component_dof(field, raw, component)
            for raw in cell_dofs(field.dofs, level, cell)]
end

function active_cell_dofs(layout::SystemLayout, field_name::Symbol, level::Integer,
                          cell::CartesianIndex, component::Integer=1)
    field = _field_layout(layout, field_name)
    return [_field_component_dof(field, raw, component)
            for raw in cell_dofs(field.dofs, level, cell)]
end

function constraint_kind(layout::SystemLayout, raw::Integer)
    constraint_kind(_only_field_layout(layout).dofs, raw)
end
function constrained_value(layout::SystemLayout, raw::Integer, component::Integer=1)
    constrained_value(_only_field_layout(layout).dofs, raw, component)
end
