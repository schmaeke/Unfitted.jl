# Degree-of-freedom layout for a superposition `Space`. The dof layer
# answers, for each cell, two questions assembly needs:
#
#   * which local basis functions map to which raw (pre-constraint) dof
#     id — produced by walking every level's cells and reusing dof ids
#     across adjacent cells through a `TensorDofKey` lookup;
#   * which raw dofs are constrained, by what kind of constraint, and
#     (for nonzero physical Dirichlet data) to what value.
#
# Physical and artificial constraints are tracked separately and never
# mixed — the physical ones in `physical_dirichlet`, the artificial ones
# in `elimination_source`:
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
#     `LevelMask` is in play — except where the inactive side is fully
#     fictitious, since a fold face carries no physical trace to
#     vanish on (`_internal_face_is_physical`).
#   * Covered-mode pruning — under leaf semantics, the high-order modes
#     buried under a finer level, plus the buried linear modes a nested
#     finer level reproduces exactly. Also a strong elimination, and also
#     artificial: it removes modes the superposition already carries,
#     never a physical condition.
#
# Active dofs are enumerated component-major (component 1 of every raw
# first, then component 2, …) so the all-or-nothing constrained case
# matches the natural block ordering, and per-component constraints
# simply leave holes in the enumeration.
#
# This file owns the dof keys, the layout structs, the structural
# overlay-constraint detection, the pruning constraint source
# (`_coverage_constraints`, over the masks `coverage.jl` computes), and
# the active-enumeration constructor.
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
    LinearConstraint{T}

A homogeneous linear constraint among raw dofs of one level:

    Σᵢ coefficientsᵢ · u_{rawsᵢ} = 0

Applies uniformly to every field component (neither constraint source
distinguishes components). Emitted by [`_overlay_constraints`](@ref) for
the trace-vanishing conditions at artificial boundaries and by
[`_coverage_constraints`](@ref) for pruning eliminations; the dof
layer resolves the resulting constraint system into a `raw_expansion`
table via cascade elimination in [`_resolve_constraints!`](@ref).

A single-raw constraint (`length(raws) == 1`, coefficient `1`) is
strong elimination of that raw — the regime integrated Legendre uses.
Multi-raw constraints (typical for B-splines with arbitrary masks or
higher continuity) couple `p + 1` raws per perpendicular line per
derivative order.
"""
struct LinearConstraint{T<:Real}
    raws::Vector{Int}
    coefficients::Vector{T}
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
  - `cell_dofs_by_level::Vector{Array{Vector{Int},D}}` — indexed by
    level *id*, not by position in `V.levels`, so that it matches the
    `parent.level` id assembly carries; a subdomain whose ids were
    reindexed into a higher block leaves the leading slots undefined.
    Each entry is a `D`-dimensional array of cell-local raw dof id
    vectors, indexed by the level's `CartesianIndex`. Inactive cells
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
  - `elimination_source::Vector{Symbol}` — `raw → :free / :overlay /
    :coverage / :dedup`. `:free` iff the raw survives; otherwise the source
    that eliminated it: the artificial overlay boundary (`:overlay`) or, from
    the pruning extension, a covered high-order mode (`:coverage`) or a
    deduped covered vertex (`:dedup`). Applies to every component
    simultaneously; derived from `raw_expansion`. Serves as both the
    elimination flag (`!== :free`) and its provenance, read by
    [`constraint_kind`](@ref), the active enumeration, and diagnostics.
  - `raw_expansion::Vector{Vector{Tuple{Int,T}}}` — for each raw, the
    list of `(other_raw, weight)` pairs that express the raw's value
    in terms of *non-pivot* raws after constraint resolution. Three
    regimes:
      - free raw → `[(raw, 1)]` (identity);
      - strongly eliminated raw (single-raw constraint, the
        integrated-Legendre case) → `[]`;
      - linear-constraint pivot → multi-element list; B-spline overlay
        boundaries with continuity order `m` produce `m + 1`-element
        lists per perpendicular line.
    The assembly hot loop distributes each emitted matrix entry
    through both the test and trial expansions, so the strong-
    elimination case (empty list) skips emission and the free case is
    the identity. See [`_resolve_constraints!`](@ref) for the
    construction algorithm.
  - `has_linear_constraints::Bool` — `true` iff any raw has a
    *non-trivial* expansion (neither the identity `[(raw, 1)]` nor the
    empty strong-elimination list `[]`), i.e. a linear-constraint pivot
    that redistributes a raw onto one or more *other* raws. The
    integrated Legendre family and the C⁰ B-spline mesh-edge path never
    produce such expansions, so this is `false` for them and assembly
    takes the lightweight single-target path; B-spline linear
    constraints (masks, `continuity_order ≥ 1`, overlay interiors) set
    it `true` and assembly takes the expansion-distributing path. The
    flag lets the hot loop pick the cheaper path without paying the
    general machinery's per-emission indirection on the common case.
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
    elimination_source::Vector{Symbol}
    raw_expansion::Vector{Vector{Tuple{Int,T}}}
    has_linear_constraints::Bool
    constrained_values::Matrix{T}
    active_count::Int
    tolerance::GeometryTolerance{T}
end

# Tag values for `AxisDofKey.kind`. `UInt8` keeps the key compact and
# fast to hash; the values themselves are arbitrary.
#
#   _AXIS_NODE   — integrated Legendre endpoint mode at a cell-corner
#                  node coordinate.
#   _AXIS_SPAN   — integrated Legendre interior (bubble) mode, cell-local.
#   _AXIS_BSPLINE — global 1D B-spline function index along the axis;
#                  used by the B-spline family in
#                  `ext/UnfittedBasicBSplineExt.jl`. Reserved here so
#                  the integrated-Legendre keys and B-spline keys live
#                  in disjoint tag ranges and the dof-key cache can
#                  share its hash table across mixed-basis problems.
const _AXIS_NODE = UInt8(0)
const _AXIS_SPAN = UInt8(1)
const _AXIS_BSPLINE = UInt8(2)

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
    local_ids = cell_basis_indices(level, cell)
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

# ── Artificial overlay-boundary detection ─────────────────────────────────────

# True iff the per-axis factor of `key` along `axis` anchors at the
# level's lower (`side = :lower`) or upper (`side = :upper`) mesh edge.
# Dispatched on the level's basis family so each family's "boundary
# mode" convention is local to its own implementation. Used by
# `_key_on_physical_side` and `_has_overlay_constraint` to translate
# "is this dof at a face" into a basis-family-specific predicate without
# leaking the family's mode-index conventions into the rest of the dof
# layer.
#
# Integrated Legendre: only `_AXIS_NODE` keys touch a mesh edge —
# `_AXIS_NODE.index == 1` (lower) or `cells[axis] + 1` (upper). Bubbles
# never touch any edge. B-spline overloads live in
# `ext/UnfittedBasicBSplineExt.jl`.
function _key_on_level_side(key::TensorDofKey{D}, level::Level{D,T,<:IntegratedLegendre},
                            axis::Integer, side::Symbol) where {D,T}
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

# Cells of `level` along `axis` incident to the axis-`axis` factor of a dof key,
# clipped to the `n[axis]` cells the level has: a `_AXIS_NODE` factor at index
# `i` touches cells `i − 1` and `i`, since a node sits between two adjacent
# cells along its axis; a `_AXIS_SPAN` (bubble) factor at index `k` touches
# only cell `k`, since a bubble mode is cell-local.
#
# This is the integrated-Legendre node-to-cell support map, and it is the one
# place it is written down: `_incident_cells` applies it on every axis, and
# `_perp_ranges` applies it on every axis but one. The B-spline family in the
# extension has its own support map — a function's span can cover many cells —
# and never routes through here.
function _axis_incidence(axis_key::AxisDofKey, ncells::Int)
    axis_key.kind == _AXIS_NODE || return axis_key.index:axis_key.index
    return max(1, axis_key.index-1):min(ncells, axis_key.index)
end

# Incidence ranges of `key` on every axis *except* `axis`, which is pinned to
# `1:1` so a `CartesianIndices` over the result enumerates the perpendicular
# positions of a candidate face along `axis` exactly once. The `axis` slot of
# each enumerated index is a placeholder the caller replaces with the cell
# position on either side of the face.
function _perp_ranges(key::TensorDofKey{D}, axis::Integer, n::NTuple{D,Int}) where {D}
    return ntuple(d -> d == axis ? (1:1) : _axis_incidence(key.axes[d], n[d]), D)
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
# off-mesh-or-inactive cell on the other side. The perpendicular positions
# come from the key's own incidence (`_perp_ranges`); for each of them,
# sample the activity of the cells immediately below (`i − 1`) and above
# (`i`) the candidate face along `axis`. A change in activity ⇒ the face is
# between an active and an inactive cell of the level, i.e. an
# active-region face.
#
# Integrated-Legendre-specific, because the incidence rule is. The B-spline
# family in the extension rolls its own active-face check tied to the
# function's actual support span, which can spread across many cells.
function _on_active_face(key::TensorDofKey{D}, level::Level{D,T,<:IntegratedLegendre},
                         axis::Integer, i::Integer, n::NTuple{D,Int}) where {D,T}
    for outer in CartesianIndices(_perp_ranges(key, axis, n))
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
#
# Dispatched on the level's basis family. The integrated-Legendre body
# below uses the family's node/span index conventions directly; the
# B-spline overload in `ext/UnfittedBasicBSplineExt.jl` does the same
# job with B-spline function-index conventions and the C⁰-knot
# treatment of mask-induced internal junctions.
function _has_overlay_constraint(key::TensorDofKey{D}, level::Level{D,T,<:IntegratedLegendre},
                                 physical, domain::AxisBox{D,T}, tol::GeometryTolerance{T},
                                 cache::_ClassifyCache{D,T}) where {D,T}
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
        #                       cells of the same level. Artificial only when it
        #                       carries physical material: a fully-fictitious
        #                       fold face leaves the mode active (see
        #                       `_internal_face_is_physical`).
        if i == 1
            _level_side_is_physical(level, domain, axis, :lower, tol) || return true
        elseif i == n[axis] + 1
            _level_side_is_physical(level, domain, axis, :upper, tol) || return true
        else
            _internal_face_is_physical(key, level, physical, axis, i, n, cache) && return true
        end
    end
    return false
end

# True iff the internal active/inactive face along `axis` at node index `i`
# carries physical material — i.e. at least one inactive cell across the face
# is not fully fictitious. Mirrors `_on_active_face`'s perpendicular incidence
# so it visits exactly the cells the face separates, and classifies only the
# inactive neighbour on each such face.
#
# The fictitious fold deactivates fully-fictitious cells, so a fold face sees
# only fictitious material on its inactive side and returns `false`: its
# boundary modes stay active. They vanish on every physical face — so they
# cannot break the C⁰ trace condition the overlay constraint enforces — and
# instead carry the adjacent cut cell's approximation up to ∂Ω. A user mask
# that excludes *physical* cells returns `true`: those modes are non-zero on a
# physical face and must be eliminated to keep the truncated solution
# continuous. With no physical domain there is no fictitious material, so every
# internal active/inactive face is physical (the original overlay/user-mask
# behaviour) and the `cache` is never touched.
function _internal_face_is_physical(key::TensorDofKey{D}, level::Level{D,T,<:IntegratedLegendre},
                                    physical, axis::Integer, i::Integer, n::NTuple{D,Int},
                                    cache::_ClassifyCache{D,T}) where {D,T}
    physical === nothing && return true
    for outer in CartesianIndices(_perp_ranges(key, axis, n))
        below_active = _cell_active(level, axis, i - 1, outer.I, n)
        above_active = _cell_active(level, axis, i, outer.I, n)
        below_active == above_active && continue
        inactive_pos = below_active ? i : i - 1
        cell = CartesianIndex(ntuple(d -> d == axis ? inactive_pos : outer[d], D))
        classify_cell(physical, cell_box(level.mesh, cell), cache) === :fictitious || return true
    end
    return false
end

# ── Linear-constraint collection and resolution ──────────────────────────────

"""
    _overlay_constraints(level, V, tol, raw_by_key, level_keys, classify_cache)
        -> Vector{LinearConstraint{T}}

Produce the homogeneous linear constraints that encode `level`'s
artificial-overlay-boundary trace condition. Dispatched on the level's
basis family so each family can exploit its own boundary mode structure.
A new family must implement all six arguments, in this order — the hook
is called positionally from [`dof_layout`](@ref) stage 2.

The default fallback walks `level_keys` (the `(key, raw)` pairs that
belong to this level, pre-bucketed by [`dof_layout`](@ref) so the scan
is `O(this level's raws)` rather than `O(all raws)` per level), asks the
family's `_has_overlay_constraint` predicate per raw, and emits a
*single-raw* constraint `1 · u_raw = 0` for every boundary raw. This is
strong elimination of the boundary node — the regime the integrated
Legendre family relies on.

The B-spline family overrides this fallback (dispatching on
`Level{…,<:BSplineFamily}`) to produce *multi-raw* constraints expressing
the C^m trace-vanishing condition along every artificial perpendicular
line, exploiting the tensor-product structure of the B-spline basis to
factor `m + 1` constraints per perpendicular line at each artificial
face. The B-spline family therefore never uses this fallback or the
per-raw `_has_overlay_constraint`; it reaches through `raw_by_key`
directly to look up raw ids by (cell, mode), and ignores `level_keys`.

`raw_by_key` is the global `TensorDofKey` → raw-id map built in stage 1
of [`dof_layout`](@ref). `classify_cache` is the space's
cell-classification cache, threaded through for whatever the family
needs it for; the fallback hands it to `_has_overlay_constraint`, which
consults it only to tell a fully-fictitious fold face from a user-mask
face. It is empty, and untouched, when `V.physical === nothing`.
"""
function _overlay_constraints(level::Level{D,T,B}, V::Space{D,T}, tol::GeometryTolerance{T},
                              raw_by_key::AbstractDict{TensorDofKey{D},Int},
                              level_keys::AbstractVector{Pair{TensorDofKey{D},Int}},
                              classify_cache::_ClassifyCache{D,T}) where {D,T,B}
    constraints = LinearConstraint{T}[]
    # `classify_cache` is the space's fold cache: the internal-face predicate
    # below re-classifies inactive neighbours to tell a fully-fictitious fold
    # face (mode stays active) from a physical user-mask face (mode eliminated),
    # and every fold-boundary cell box is already in the cache. Empty and unused
    # when `V.physical === nothing`.
    for (key, raw) in level_keys
        _has_overlay_constraint(key, level, V.physical, V.domain, tol, classify_cache) || continue
        push!(constraints, LinearConstraint{T}([raw], [one(T)]))
    end
    return constraints
end

# ── Pruning (coverage) constraint source ─────────────────────────────

# Cells of `level` incident to the entity of `key`, from the per-axis incidence
# rule of `_axis_incidence` applied on every axis.
function _incident_cells(key::TensorDofKey{D}, n::NTuple{D,Int}) where {D}
    return CartesianIndices(ntuple(d -> _axis_incidence(key.axes[d], n[d]), D))
end

# ── The minimum rule: per-cell order across a shared entity ──────────────────
#
# A cell at order 5 beside a cell at order 2 generates dof keys the neighbour
# does not. Interior bubble keys are cell-local (`_AXIS_SPAN` on every axis) and
# vertex keys carry mode 0 on every axis, so neither is a problem. The keys that
# are shared *and* carry a mode — edge keys in 2D, edge and face keys in 3D —
# are the whole difficulty: a raw generated by only one of the two cells is
# nonzero on the face they share, so the superposed space is no longer C⁰. It
# assembles, it solves, and it returns a plausible residual; only a two-sided
# trace evaluation sees it.
#
# The classical hp-FEM fix is the *minimum rule*: a shared entity carries the
# minimum of the orders of the cells sharing it. It is due to
#
#   B. Szabó, I. Babuška, "Finite Element Analysis", Wiley, New York (1991),
#     ISBN 978-0-471-50273-9 — cited in full above `local_basis_indices` in
#     `basis.jl`, where the same work's trunk space is defined; and
#   L. Demkowicz, "Computing with hp-Adaptive Finite Elements, Vol. 1: One and
#     Two Dimensional Elliptic and Maxwell Problems", Chapman & Hall/CRC (2006),
#     doi:10.1201/9781420011685 — the constrained-approximation treatment this
#     package's key-keyed statement of the rule follows.
#
# Three properties of the statement below are load-bearing and none of them is
# cosmetic:
#
#   * It is a predicate on the **key**, not on the generating cell. The minimum
#     over `_incident_cells(key, n)` is the same number whichever incident cell
#     asks, so both sides of a face agree by construction and the one-sided mode
#     is not expressible. A per-cell filter would have to be kept symmetric by
#     hand.
#   * Every question is asked at an order some cell **actually carries**. The
#     classical statement forms the componentwise minimum first and then asks
#     the family whether the mode survives *there* — at an order that in general
#     no cell has, and where the family's index set therefore has to be defined
#     by extension. That extension is where the rule goes wrong: read per axis
#     it admits a 3D `:trunk` face key with bubble modes (3, 2), whose trunk
#     degree is 5, at order 4 — a mode the order-4 cell never generates, and a
#     measured 0.186 trace jump. Asking each incident cell about its own index
#     set removes the fiction and with it the ambiguity.
#   * Inactive incident cells are **skipped, not minimised over**. An inactive
#     cell generates nothing, so it cannot disagree; letting it lower the entity
#     would delete dofs nothing replaces. That is not hypothetical: under a
#     fictitious fold `_internal_face_is_physical` deliberately keeps the modes
#     on a fold face free, precisely so the adjacent cut cell's approximation
#     reaches ∂Ω.
#
# One pass suffices — no fixed point. The incident cells of a key form a
# face-connected `2^(#NODE axes)` hypercube, so agreement across every face of
# that block already implies agreement with every cell in it.
#
# WHY THE INTERSECTION IS THE MINIMUM RULE. `_index_admissible` is, for every
# family that opts into a per-cell order, a conjunction of upper bounds on the
# order — `0 ≤ αᵈ ≤ pᵈ` for `:tensor`, plus `Σ t(αᵈ) ≤ p₁` for `:trunk`. Raising
# an order therefore never removes a mode from a cell's set, only adds: the
# predicate is monotone non-decreasing in `p`. Given that,
#
#     admissible(min over incident cells, α)  ⟺  admissible(order(c), α) ∀ c
#
# so the two statements are the same rule and the minimum need never be formed.
# The equivalence *rests* on that monotonicity, which is a property of the
# family rather than of this file, so `test_cell_order.jl` asserts it directly
# for every family answering `_supports_cell_order() == true`. A family that
# broke it would silently get a non-conforming space from this code.
#
# What the rule cannot express, and what therefore has to be checked elsewhere:
# every cell needs `order ≥ 1` on every axis. `_axis_dof_key` deliberately
# collapses the two endpoint modes 0 and 1 into one `_AXIS_NODE` factor — that
# collapse is exactly what makes an endpoint mode shareable across a cell
# boundary — so no filter over keys can notice that a cell is missing one of its
# two endpoint modes. `_build_cell_orders` in `mesh.jl` validates every palette
# entry through `local_basis_indices` for that reason.

# Whether the shared entity of `key` carries the mode `id`: every *active* cell
# incident to it generates `id` from its own order. The generating cell is itself
# incident, and `id` came from its own index set, so the loop is never empty and
# the unconstrained case needs no separate answer.
function _entity_carries(basis::BasisFamily, orders::CellOrders{D}, mask, key::TensorDofKey{D},
                         n::NTuple{D,Int}, mode::Symbol, id::CartesianIndex{D}) where {D}
    for ci in _incident_cells(key, n)
        is_active(mask, ci) || continue
        _index_admissible(basis, orders.palette[orders.class[ci]], mode, id) || return false
    end
    return true
end

# Integrated Legendre's method of the `_cell_modes` hook whose generic form is in
# `mesh.jl`. It lives here, with `_entity_carries` and the `TensorDofKey` factors
# the minimum rule is expressed over, the same way the B-spline family's
# `_coverage_constraints` lives in its extension.
#
# The per-order index sets are hoisted out of the cell loop: there are only
# `length(palette)` distinct answers and `local_basis_indices` allocates a fresh
# vector on every call, so asking it per cell allocated one vector per active
# cell of the level. Measured on a 16³ level carrying 36 distinct anisotropic
# orders, 36 973 allocations and 30.6 ms against 23 912 and 28.5 ms.
function _cell_modes(basis::IntegratedLegendre, orders::CellOrders{D}, mesh::CartesianMesh{D},
                     mode::Symbol, mask) where {D}
    length(orders.palette) == 1 && return _uniform_cell_modes(basis, orders, mesh, mode, mask)
    n = mesh.cells
    sets = [CartesianIndex{D}[]]
    index = Dict{Vector{CartesianIndex{D}},UInt16}(sets[1] => UInt16(1))
    kind = Array{UInt16,D}(undef, n)
    sets_by_class = [local_basis_indices(basis, o, mode) for o in orders.palette]
    for cell in cell_indices(mesh)
        if !is_active(mask, cell)
            kind[cell] = UInt16(1)
            continue
        end
        kept = CartesianIndex{D}[]
        for id in sets_by_class[orders.class[cell]]
            key = TensorDofKey{D}(0, ntuple(d -> _axis_dof_key(cell.I[d], id.I[d]), D))
            _entity_carries(basis, orders, mask, key, n, mode, id) && push!(kept, id)
        end
        kind[cell] = get!(index, kept) do
            length(sets) == typemax(UInt16) &&
                throw(ArgumentError("a level carries at most $(typemax(UInt16)) distinct " *
                                    "per-cell mode sets, and this one asks for more"))
            push!(sets, kept)
            UInt16(length(sets))
        end
    end
    return CellModes{D}(sets, kind)
end

# Whether a family's span on a mesh contains that mesh's C⁰ multilinear vertex
# functions — the hats. This is the basis half of the linear dedup in
# `_coverage_constraints`; `_nested_over` (`coverage.jl`) is the geometric half, and a
# cover must pass both before a buried hat of the level below is eliminated.
#
# Integrated Legendre qualifies in either mode at any per-cell order ≥ 1: it is exactly
# C⁰ across a cell boundary, and the two endpoint modes it collapses into one node
# factor have trunk degree 0, so Q1 is in every cell's index set. The default is `false`,
# which is the safe direction — skipping a legitimate dedup leaves a duplicate, while
# deduping against a span that does not contain the hat deletes part of the space.
#
# It is a family trait rather than a type test on the cover because smoothness is a
# property of the family's own construction: a maximal-regularity B-spline is C^(p−1) at
# a simple interior knot, so it carries the kink at degree 1 and at no higher degree —
# the extension answers exactly that, and a `k.basis isa IntegratedLegendre` gate cannot.
_spans_hats(::BasisFamily) = false
_spans_hats(::IntegratedLegendre) = true

"""
    _coverage_constraints(level, V, coverage, tol, level_keys, classify_cache)
        -> Vector{Tuple{LinearConstraint{T},Symbol}}

Pruning constraint source, a peer of [`_overlay_constraints`](@ref). Under the
leaf semantics (`dof_layout`'s `prune`, on by default), emit a single-raw
strong elimination for

  * every **buried high-order** mode (at least one bubble axis, every active incident
    cell covered by and every fictitious one reached by a single finer level) —
    covered-mode pruning, source `:coverage`; and
  * every **buried linear** mode a single nested level above whose basis spans that
    mesh's hats reproduces exactly — dedup, source `:dedup`.

The linear skeleton is otherwise retained, which is what makes the reduced space
complete. Each returned pair carries its elimination source for `constraint_kind` /
diagnostics.

Both halves ask `_reproduced_on_domain` (`coverage.jl`), which applies the
fictitious-fold rule on the covering level *and* on this one — a support cell folded
away carries nothing to reproduce, so it neither has to be covered nor may veto the
verdict. `classify_cache` is threaded through for the classification that tells a fold
from a user mask, the same one [`build_coverage`](@ref) used.

The dedup half is not an optimisation: a mode a covering level reproduces exactly is
linearly dependent on that level's own modes, so leaving both active makes the
superposed operator exactly singular. The B-spline extension overrides this hook for
the same reason, and emits nothing but dedups — that family has no bubble modes to
shed. Every other family takes the generic fallback below and reduces nothing.
"""
function _coverage_constraints(level::Level{D,T,<:IntegratedLegendre}, V::Space{D,T},
                               coverage::Coverage{D}, tol::GeometryTolerance{T},
                               level_keys::AbstractVector{Pair{TensorDofKey{D},Int}},
                               classify_cache::_ClassifyCache{D,T}) where {D,T}
    out = Tuple{LinearConstraint{T},Symbol}[]
    cov = coverage.covered[level.id]
    any(cov) || return out
    n = level.mesh.cells
    # A buried vertex function is a C⁰ hat: multilinear on each of `level`'s cells,
    # kinked across every cell boundary. Mesh nesting (`_nested_over`) only buys the
    # multilinear half; the kink is the cover basis's half, and `_spans_hats` is where
    # each family answers for it. Both halves are necessary and neither is about this
    # level's family — a cover that spans the hats deduplicates them whoever generated
    # them.
    nested_above = [k
                    for k in V.levels
                    if k.id > level.id && _spans_hats(k.basis) && _nested_over(level, k, tol)]
    reproduces(k, cells) = _reproduced_on_domain(level, k, cells, V.physical, tol, classify_cache)
    # Covered-mode pruning asks the same *single-level* question the dedup half asks.
    # `cov[ci]` says only that *some* higher level contains cell `ci`, and a mode
    # whose incident cells are covered by two different levels is reproduced by
    # neither: each of them is clamped to zero on the face they share, so at that
    # seam nothing carries what the elimination removes. Requiring one level to
    # cover the whole stencil is the same condition the dedup half already
    # applies, and for the same reason.
    #
    # It deliberately does NOT ask the covering level to carry this level's
    # order. Shedding a buried high-order mode is not a claim that something
    # reproduces it — that is the dedup half's job, for linear modes. It is the
    # what leaf semantics mean: over a region a finer level
    # resolves, a coarse cell's high-order modes buy almost nothing in L² and
    # carry the oscillation, so a *high-order base with a low-order fine overlay
    # over a non-smooth feature* is exactly the configuration the rule exists to
    # serve. Gating on the cover's order turns leaf semantics off exactly there:
    # measured on a tanh layer with base p = 5 and
    # a p = 2 overlay, the gate cut the base's shed modes from 189 to 5 and left
    # far-field oscillation and overshoot bit-identical to switching the rule
    # off (9.37e-3 both), against 9.95e-4 and 7.03e-4 without it.
    above = [k for k in V.levels if k.id > level.id]
    resolves(cells) = any(k -> reproduces(k, cells), above)
    # The `cov` prefilter is a fast path over the *active* support only, which is both
    # the set `_coverage_cells` computed and the only set on which a `false` entry is a
    # verdict rather than a default. An inactive incident cell is settled by
    # `reproduces`: fictitious and it is skipped, user-masked and it refuses.
    for (key, raw) in level_keys
        cells = _incident_cells(key, n)
        all(ci -> !is_active(level.mask, ci) || cov[ci], cells) || continue     # buried?
        if any(a -> a.kind == _AXIS_SPAN, key.axes)                 # high-order → covered-mode pruning
            resolves(cells) && push!(out, (LinearConstraint{T}([raw], [one(T)]), :coverage))
        elseif any(k -> reproduces(k, cells), nested_above)          # linear reproduced above
            push!(out, (LinearConstraint{T}([raw], [one(T)]), :dedup))
        end
    end
    return out
end

# Generic fallback: no covered-mode pruning for a family that has not opted in.
function _coverage_constraints(::Level{D,T,B}, ::Space{D,T}, ::Coverage{D}, ::GeometryTolerance{T},
                               ::AbstractVector{Pair{TensorDofKey{D},Int}},
                               ::_ClassifyCache{D,T}) where {D,T,B}
    return Tuple{LinearConstraint{T},Symbol}[]
end

# Combine repeated raws in a list of `(raw, coefficient)` pairs by
# summing the coefficients, dropping terms whose coefficient falls
# below the working tolerance. Pure-data helper consumed by the
# cascade resolver below; sorts in place for compactness.
#
# A one-term list is not a special case, and must not be short-circuited past the
# drop test. A multi-raw constraint re-emitted after its raws were pivoted away can
# leave exactly one surviving term whose coefficient is zero — `0·u = 0`, which
# constrains nothing — and returning that unfiltered makes the resolver pivot on it
# and eliminate `u` outright.
function _combine_terms(terms::Vector{Tuple{Int,T}}) where {T}
    isempty(terms) && return terms
    sort!(terms; by=first)
    write = 1
    @inbounds for read in 2:length(terms)
        if terms[read][1] == terms[write][1]
            terms[write] = (terms[write][1], terms[write][2] + terms[read][2])
        else
            write += 1
            terms[write] = terms[read]
        end
    end
    resize!(terms, write)
    drop_tol = sqrt(eps(T)) * maximum(t -> abs(t[2]), terms; init=one(T))
    filter!(t -> abs(t[2]) > drop_tol, terms)
    return terms
end

# Back-substitute a freshly created pivot's expansion into every other
# pivot's expansion that mentions the new pivot raw. Necessary because
# our greedy pivot ordering doesn't guarantee that a constraint's pivot
# choice is downstream of all its references — when a later constraint
# pivots a raw that an earlier pivot's expansion still mentions, the
# earlier expansion needs the substitution to stay in "free raws only"
# form.
#
# `mentions` maps a raw to the pivots whose expansion currently contains it,
# so this visits only the expansions that can possibly need rewriting. It
# used to scan all `nraw` expansions per constraint, which made the pass
# `O(K · nraw)` — quadratic in the problem — while the comment here claimed
# `O(K · D · p)`, "negligible at problem scale". It was not negligible: on a
# covered stack the dof layout reached 98.7% of `prepare` and 2.8× the cost
# of `assemble!`, growing ×12.7 and ×14.8 per ×4 in unknowns. The scan's
# answer was almost always "nothing to do", because every integrated
# Legendre elimination is a single-raw strong constraint whose pivot
# expansion is empty and which therefore mentions no raw at all.
#
# The index is allowed to over-report: an expansion rewritten by an earlier
# substitution may no longer contain the raw that registered it, so the
# membership test below stays. It must never under-report, which is why an
# entry is added at *every* site that writes an expansion — here and in
# `_resolve_constraints!` step 4. The identity expansions set up before the
# loop need no entry: a raw is skipped until it is pivoted, and pivoting
# overwrites its expansion.
function _back_substitute!(raw_expansion::Vector{Vector{Tuple{Int,T}}}, pivot_raw::Int,
                           pivot_expansion::Vector{Tuple{Int,T}}, pivoted::AbstractVector{Bool},
                           mentions::Dict{Int,Vector{Int}}) where {T}
    holders = get(mentions, pivot_raw, nothing)
    holders === nothing && return raw_expansion
    @inbounds for raw in holders
        raw == pivot_raw && continue
        pivoted[raw] || continue
        expansion = raw_expansion[raw]
        # The index over-reports; this is the exact test.
        any(t -> first(t) == pivot_raw, expansion) || continue
        # Rebuild with substitution.
        new_terms = Tuple{Int,T}[]
        sizehint!(new_terms, length(expansion) + length(pivot_expansion))
        for (other, w) in expansion
            if other == pivot_raw
                for (sub_other, sub_w) in pivot_expansion
                    push!(new_terms, (sub_other, w * sub_w))
                end
            else
                push!(new_terms, (other, w))
            end
        end
        combined = _combine_terms(new_terms)
        raw_expansion[raw] = combined
        # This expansion now mentions the pivot's own references. `pivot_raw`
        # cannot be among them — `pivot_expansion` was built excluding it — so
        # the list being iterated is never the one appended to.
        _register_mentions!(mentions, raw, combined)
    end
    return raw_expansion
end

# Record that `raw`'s expansion contains each of `terms`' raws.
function _register_mentions!(mentions::Dict{Int,Vector{Int}}, raw::Int,
                             terms::Vector{Tuple{Int,T}}) where {T}
    for (other, _) in terms
        push!(get!(() -> Int[], mentions, other), raw)
    end
    return mentions
end

"""
    _resolve_constraints!(raw_expansion, constraints, nraw) -> raw_expansion

Resolve a list of homogeneous linear constraints into the per-raw
expansion table `raw_expansion`, mutated in place.

After resolution every entry is either

  - `[(raw, 1)]` — the raw is *free*, identity expansion,
  - `[]` — the raw is *strongly eliminated* (single-raw constraint),
  - `[(rᵢ, wᵢ), …]` — the raw is a *linear-constraint pivot*, expressed
    in terms of currently-free raws via `u_raw = Σᵢ wᵢ u_rᵢ`.

The algorithm is the classical sparse Gauss elimination, adapted to
exploit the small constraint matrix's structure:

  1. For each constraint, substitute already-resolved pivot expressions
     into the constraint's terms (so the constraint is in free-raws +
     yet-to-resolve raws).
  2. Combine like terms and drop near-zeros; skip the constraint
     entirely if it collapses to `0 = 0`.
  3. Pick a pivot by largest absolute coefficient (numerical stability).
  4. Express the pivot as a linear combination of the remaining
     coefficients' raws.
  5. Back-substitute the pivot into every earlier pivot's expansion
     that mentions the new pivot raw.

Step 5 keeps the invariant: at any point after a constraint has been
processed, *every pivot's expansion is in terms of currently-free raws
only*. Assembly emission then never needs to chase chains.

The cascade depth of overlay-boundary constraints is bounded by the
spatial dimension `D` — a corner raw of the active region touches at
most `D` boundary axes — so total resolution cost is `O(K · D · p)`
where `K` is the constraint count and `p` is the per-constraint raw
count.
"""
function _resolve_constraints!(raw_expansion::Vector{Vector{Tuple{Int,T}}},
                               constraints::Vector{LinearConstraint{T}}, nraw::Int) where {T}
    # Identity start: every raw is its own expansion.
    for raw in 1:nraw
        raw_expansion[raw] = [(raw, one(T))]
    end
    pivoted = falses(nraw)
    # raw -> the pivots whose expansion mentions it. Empty for every
    # single-raw strong elimination, which is all of them for integrated
    # Legendre, so this costs nothing on the common path.
    mentions = Dict{Int,Vector{Int}}()

    for constraint in constraints
        # 1) Substitute already-pivoted raws.
        terms = Tuple{Int,T}[]
        sizehint!(terms, length(constraint.raws))
        for (raw, coeff) in zip(constraint.raws, constraint.coefficients)
            for (other, w) in raw_expansion[raw]
                push!(terms, (other, coeff * w))
            end
        end

        # 2) Combine like terms; trivially satisfied constraints fall away.
        combined = _combine_terms(terms)
        isempty(combined) && continue

        # 3) Pick pivot by largest |coefficient| (numerical stability).
        _, pivot_pos = findmax(t -> abs(t[2]), combined)
        pivot_raw, pivot_coeff = combined[pivot_pos]

        # 4) Build the pivot expansion (in terms of remaining raws).
        new_expansion = Tuple{Int,T}[(raw, -c / pivot_coeff)
                                     for (k, (raw, c)) in pairs(combined) if k != pivot_pos]

        raw_expansion[pivot_raw] = new_expansion
        pivoted[pivot_raw] = true
        _register_mentions!(mentions, pivot_raw, new_expansion)

        # 5) Back-substitute into earlier pivots that mention this raw.
        _back_substitute!(raw_expansion, pivot_raw, new_expansion, pivoted, mentions)
    end
    return raw_expansion
end

# ── DofLayout construction and accessors ──────────────────────────────────────

# Active global id of component `component` of raw dof `raw`, or 0 if
# the (raw, component) is constrained.
function _active_component_dof(layout::DofLayout, raw::Integer, component::Integer)
    layout.active_component[raw, component]
end

"""
    dof_layout(V::Space; dirichlet=[], tolerance=GeometryTolerance(T), components=1,
                         prune=true, classify_cache=_ClassifyCache{D,T}()) -> DofLayout

Construct the basis-aware global dof layout for a superposition
[`Space`](@ref). The construction proceeds in four stages:

  1. Walk every level's cells in order and build the cell-local raw dof
     id vector through `TensorDofKey` lookup. Adjacent cells share
     endpoint nodes; bubble (span) modes are cell-local. Inactive cells
     get an empty vector.
  2. Collect homogeneous linear constraints from every level, from two
     sources: the family-dispatched [`_overlay_constraints`](@ref) hook
     on every level, and [`_coverage_constraints`](@ref) on every level
     under leaf semantics (which needs the per-level masks
     [`build_coverage`](@ref) computes, so those are built first, and
     skipped entirely when `prune = false`). A raw the overlay condition already eliminates
     is not re-constrained by covered-mode pruning, so `elimination_source`
     credits it to the source that actually removed it. Resolve the
     collected constraints into the per-raw expansion table via
     [`_resolve_constraints!`](@ref). The resulting `raw_expansion[raw]`
     is either the identity `[(raw, 1)]` for free raws, the empty list
     `[]` for strongly eliminated raws, or a multi-element list for
     linear-constraint pivots.
  3. Per-component physical Dirichlet detection via
     `_has_physical_dirichlet`.
  4. Enumerate active dofs component-major (component 1 first, then
     component 2, …) skipping every (raw, component) entry that is
     either physically Dirichlet-constrained or a constraint pivot.
     Project nonzero Dirichlet data onto the boundary trace space via
     [`_project_dirichlet_values!`](@ref).

Keyword arguments:

  - `dirichlet` — iterable of [`DirichletCondition`](@ref)s.
  - `tolerance` — `GeometryTolerance` used by boundary detection.
  - `components` — scalar channels per field (≥ 1).
  - `prune` — leaf semantics, on by default. `false` retains every covered
    mode and is the unreduced twin the reduction is measured against; see
    [`prepare`](@ref), which is where a caller reaches it.
  - `classify_cache` — the space's cell-classification cache (shared with the
    `PhysicalDomain` fold). The fictitious-fold constraint predicate and the
    coverage rule both reuse it instead of re-classifying fold-boundary cells;
    defaults to a fresh empty cache for standalone calls.

The integrated Legendre family produces single-raw constraints,
reducing the resolved expansion to strong elimination (`raw_expansion =
[]` for the pivoted raw); the assembly path then behaves identically
to the pre-constraint-primitive code. B-spline families produce
multi-raw constraints encoding the C^m trace-vanishing condition on
artificial boundaries; the assembly path distributes entries through
the expansion automatically.
"""
function dof_layout(V::Space{D,T}; dirichlet=[], tolerance=GeometryTolerance(T),
                    components::Integer=1, prune::Bool=true,
                    classify_cache::_ClassifyCache{D,T}=_ClassifyCache{D,T}()) where {D,T}
    components > 0 || throw(ArgumentError("dof layout components must be positive"))
    raw_by_key = Dict{TensorDofKey{D},Int}()
    raw_keys = TensorDofKey{D}[]
    # Indexed by level *id*, not by position: for a single-domain space the two
    # coincide (base = 1, overlays 2, 3, …), but a subdomain of a coupled model
    # is reindexed into a disjoint id block by `_reindex_space_levels`, so this
    # layout's ids run over that block and match the `parent.level` id the
    # assembly workspace and `cell_dofs` key by (leading slots below the block
    # stay `undef` and are never addressed by this field's parents).
    cell_dofs_by_level = Vector{Array{Vector{Int},D}}(undef, maximum(l -> l.id, V.levels))

    # Stage 1: walk every level's cells and assign raw dofs through the
    # `TensorDofKey` cache (cross-cell endpoint sharing happens here).
    #
    # Every inactive cell of every level shares one empty vector. A ladder makes
    # inactive cells the overwhelming majority of the grid — measured on a
    # depth-5 2D stack over a 6² base with one base cell refined to the bottom,
    # 49 140 cells of which 1 400 are active — and a fresh `Int[]` per inactive
    # cell put one heap allocation on each of them, on every rebuild, and
    # `estimate` rebuilds. On that layout it is 74 856 allocations against
    # 27 117. It is per call rather than a module-level constant because a
    # `const` empty vector is one stray `push!` away from corrupting every
    # inactive cell of every layout ever built.
    none = Int[]
    for level in V.levels
        level_cells = Array{Vector{Int},D}(undef, level.mesh.cells)
        for cell in cell_indices(level.mesh)
            level_cells[cell] = is_active(level.mask, cell) ?
                                _build_cell_dofs!(raw_by_key, raw_keys, level, cell) : none
        end
        cell_dofs_by_level[level.id] = level_cells
    end

    # Stage 2: collect linear constraints per level and resolve to expansions.
    # Bucket the raw keys by level once so each level's constraint hook
    # scans only its own raws (O(total raws) overall, not O(levels × raws)).
    nraw = length(raw_keys)
    keys_by_level = Dict{Int,Vector{Pair{TensorDofKey{D},Int}}}()
    for (key, raw) in raw_by_key
        push!(get!(() -> Pair{TensorDofKey{D},Int}[], keys_by_level, key.level), key => raw)
    end
    empty_keys = Pair{TensorDofKey{D},Int}[]
    constraints = LinearConstraint{T}[]
    # Track only the COVERAGE/DEDUP elimination sources, so `constraint_kind` /
    # diagnostics can pick a pruned or deduped raw out. Overlay-boundary
    # (and multi-raw B-spline) eliminations are not recorded here — they fall
    # through to the `:overlay` default when `elimination_source` is built below.
    # The loop below keeps that split honest: a raw the overlay condition already
    # eliminates never reaches this map, whichever source names it second.
    source_of = Dict{Int,Symbol}()
    coverage = prune ? build_coverage(V, tolerance, classify_cache) :
               Coverage{D}(Dict{Int,BitArray{D}}())
    for level in V.levels
        level_keys = get(keys_by_level, level.id, empty_keys)
        # Source 1: the artificial-overlay-boundary trace condition (every family).
        overlay_constraints = _overlay_constraints(level, V, tolerance, raw_by_key, level_keys,
                                                   classify_cache)
        append!(constraints, overlay_constraints)
        # Source 2: covered-mode pruning in covered regions (opt-in per level).
        #
        # A raw on Γ_o that is also buried carries both constraints. The overlay
        # one is queued first and strongly eliminates the raw, which leaves the
        # coverage constraint trivially satisfied — so recording it as the
        # elimination source would credit covered-mode pruning with a raw it did not
        # save, and `reduced_mode_counts` would over-report. Skipping the
        # redundant constraint leaves `raw_expansion` untouched:
        # `_resolve_constraints!` substitutes the already-eliminated raw, gets an
        # empty term list, and drops the constraint anyway. Only single-raw
        # overlay constraints eliminate a named raw outright; a multi-raw
        # (B-spline) constraint picks its pivot during resolution, and that
        # family emits no coverage constraints at all.
        if prune
            reductions = _coverage_constraints(level, V, coverage, tolerance, level_keys,
                                               classify_cache)
            overlay_raws = isempty(reductions) ? Set{Int}() :
                           Set{Int}(c.raws[1] for c in overlay_constraints if length(c.raws) == 1)
            for (c, src) in reductions
                c.raws[1] in overlay_raws && continue
                push!(constraints, c)
                source_of[c.raws[1]] = src
            end
        end
    end
    raw_expansion = Vector{Vector{Tuple{Int,T}}}(undef, nraw)
    _resolve_constraints!(raw_expansion, constraints, nraw)
    # A raw is "pivoted" (eliminated) iff its resolved expansion is anything other
    # than the trivial identity `[(raw, 1)]`. This local vector is construction
    # scratch for the active enumeration and for `elimination_source` below; the
    # layout exposes elimination only through `elimination_source` (`!== :free`).
    eliminated = [length(e) != 1 || e[1] != (raw, one(T)) for (raw, e) in pairs(raw_expansion)]
    # Per-raw elimination source: the tracked coverage/dedup source, `:overlay` for
    # any other eliminated raw (overlay boundary; multi-raw B-spline pivots), and
    # `:free` when the raw survives. This doubles as the elimination flag.
    elimination_source = Symbol[eliminated[raw] ? get(source_of, raw, :overlay) : :free
                                for raw in 1:nraw]
    # A raw is "simple" if its expansion is either the identity
    # `[(raw, 1)]` (free) or empty `[]` (strongly eliminated) — both
    # the lightweight assembly path represents directly as a single
    # active-or-constrained local target. Any other expansion (a
    # redirect onto other raws) is a genuine linear constraint and
    # forces the expansion-distributing assembly path.
    has_linear_constraints = any(pairs(raw_expansion)) do (raw, e)
        !(isempty(e) || (length(e) == 1 && e[1] == (raw, one(T))))
    end

    # Stage 3: per-component physical Dirichlet detection.
    ncomp = Int(components)
    physical_dirichlet = falses(nraw, ncomp)
    for (raw, key) in pairs(raw_keys)
        level = _level_by_id(V, key.level)
        for component in 1:ncomp
            physical_dirichlet[raw, component] = _has_physical_dirichlet(key, level, V.domain,
                                                                         dirichlet, tolerance,
                                                                         component)
        end
    end

    # Stage 4: enumerate active dofs component-major. For all-or-nothing
    # constraints this matches the natural block ordering; per-component
    # constraints simply leave holes in the enumeration.
    active_component = zeros(Int, nraw, ncomp)
    active_count = 0
    for component in 1:ncomp, raw in 1:nraw
        if !(physical_dirichlet[raw, component] || eliminated[raw])
            active_count += 1
            active_component[raw, component] = active_count
        end
    end

    layout = DofLayout{D,T}(ncomp, cell_dofs_by_level, raw_keys, active_component,
                            physical_dirichlet, elimination_source, raw_expansion,
                            has_linear_constraints, zeros(T, nraw, ncomp), active_count, tolerance)

    # Stage 4b: project nonzero Dirichlet data onto the boundary trace
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

The vector is the layout's own and must not be mutated: every inactive cell of
every level shares one empty vector, so a `push!` into a returned empty result
would reach all of them. Copy before editing.
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
  - `:overlay`    — artificial overlay-boundary elimination only.
  - `:coverage`   — pruning elimination (covered high-order mode).
  - `:dedup`      — linear-dedup elimination (covered vertex reproduced by
                    a nested finer level).
  - `:mixed`      — physical Dirichlet plus an elimination. The
                    elimination wins (the dof is eliminated and its value
                    is held in `constrained_values` from the Dirichlet
                    projection).
"""
function constraint_kind(layout::DofLayout, raw::Integer, component::Integer=1)
    physical = layout.physical_dirichlet[raw, component]
    source = layout.elimination_source[raw]
    physical && source != :free && return :mixed
    physical && return :dirichlet
    return source
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
    FieldLayout{D,T}(name, components, dofs, offset, level_ids)

One field's slot inside a multi-field [`SystemLayout`](@ref). `dofs` is
the field's own [`DofLayout`](@ref); `offset` is the field's starting
column in the global active-dof vector (the active dofs of all earlier
fields take ids 1 through `offset`, and this field's active dofs take
ids `offset + 1` through `offset + active_unknowns(dofs)`).

`level_ids` is the contiguous block of (globally reindexed) level ids the
field's [`Space`](@ref) owns. Assembly uses it to route each integration
region to the field that owns it: a region whose parents live on a level
outside this range belongs to another subdomain and this field skips it
(`region_parents`). For a single-domain problem every field's range covers
every level, so every field evaluates on every region.
"""
struct FieldLayout{D,T<:Real}
    name::Symbol
    components::Int
    dofs::DofLayout{D,T}
    offset::Int
    level_ids::UnitRange{Int}
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

# True iff any field's dof layout carries a non-trivial linear
# constraint (a [`DofLayout`](@ref) `raw_expansion` that redistributes a
# raw onto other raws). Assembly consults this once per region to choose
# between the lightweight single-target path and the general
# expansion-distributing path; see [`_accumulate_qpoint!`](@ref).
function has_linear_constraints(layout::SystemLayout)
    any(field -> field.dofs.has_linear_constraints, layout.fields)
end

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
