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
    LinearConstraint{T}

A homogeneous linear constraint among raw dofs of one level:

    Σᵢ coefficientsᵢ · u_{rawsᵢ} = 0

Applies uniformly to every field component (overlay constraints don't
distinguish components). Used by [`_overlay_constraints`](@ref) to
encode trace-vanishing conditions at artificial boundaries; the dof
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
  - `elimination_source::Vector{Symbol}` — `raw → :free / :overlay /
    :coverage / :dedup`. `:free` iff the raw survives; otherwise the source
    that eliminated it: the artificial overlay boundary (`:overlay`) or, from
    the order-reduction extension, a covered high-order mode (`:coverage`) or a
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
#
# Integrated-Legendre-specific: the node-to-cell support map and the
# `_AXIS_NODE` / `_AXIS_SPAN` perpendicular incidence rules are
# integrated-Legendre conventions. The B-spline family in the extension
# rolls its own active-face check tied to the function's actual support
# span, which can spread across many cells.
function _on_active_face(key::TensorDofKey{D}, level::Level{D,T,<:IntegratedLegendre},
                         axis::Integer, i::Integer, n::NTuple{D,Int}) where {D,T}
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
    ranges = ntuple(D) do d
        if d == axis
            1:1
        else
            kd = key.axes[d]
            kd.kind == _AXIS_NODE ? (max(1, kd.index-1):min(n[d], kd.index)) : (kd.index:kd.index)
        end
    end
    for outer in CartesianIndices(ranges)
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
    _overlay_constraints(level, V, tol, raw_by_key, level_keys) -> Vector{LinearConstraint{T}}

Produce the homogeneous linear constraints that encode `level`'s
artificial-overlay-boundary trace condition. Dispatched on the level's
basis family so each family can exploit its own boundary mode structure.

The default fallback walks `level_keys` (the `(key, raw)` pairs that
belong to this level, pre-bucketed by [`dof_layout`](@ref) so the scan
is `O(this level's raws)` rather than `O(all raws)` per level), asks the
family's [`_has_overlay_constraint`](@ref) predicate per raw, and emits a
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
of [`dof_layout`](@ref).
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

# ── Order-reduction (coverage) constraint source ─────────────────────────────

# Cells of `level` incident to the entity of `key`, reconstructed from the per-axis
# node/span structure: a NODE factor at index i touches cells i-1 and i (whichever are
# in bounds); a SPAN (bubble) factor at index k touches only cell k. Mirrors the
# perpendicular incidence used by `_on_active_face`.
function _incident_cells(key::TensorDofKey{D}, n::NTuple{D,Int}) where {D}
    ranges = ntuple(D) do d
        a = key.axes[d]
        a.kind == _AXIS_NODE ? (max(1, a.index-1):min(n[d], a.index)) : (a.index:a.index)
    end
    return CartesianIndices(ranges)
end

"""
    _coverage_constraints(level, V, coverage, tol, level_keys)
        -> Vector{Tuple{LinearConstraint{T},Symbol}}

Order-reduction constraint source, a peer of [`_overlay_constraints`](@ref). For a
level opted into `reduce_order`, emit a single-raw strong elimination for

  * every **buried high-order** mode (at least one bubble axis, every incident cell
    covered) — order reduction, source `:coverage`; and
  * every **buried linear** mode a single nested level above reproduces exactly —
    dedup, source `:dedup`.

The linear skeleton is otherwise retained, which is what makes the reduced space
complete (see `docs/design/covered-cell-deactivation.md`). Each returned pair carries
its elimination source for `constraint_kind` / diagnostics.

Integrated-Legendre only; the generic fallback returns nothing (order reduction is out
of scope for the B-spline family).
"""
function _coverage_constraints(level::Level{D,T,<:IntegratedLegendre}, V::Space{D,T},
                               coverage::Coverage{D}, tol::GeometryTolerance{T},
                               level_keys::AbstractVector{Pair{TensorDofKey{D},Int}}) where {D,T}
    out = Tuple{LinearConstraint{T},Symbol}[]
    cov = coverage.covered[level.id]
    any(cov) || return out
    n = level.mesh.cells
    nested_above = [k for k in V.levels if k.id > level.id && _nested_over(level, k, tol)]
    for (key, raw) in level_keys
        cells = _incident_cells(key, n)
        all(ci -> cov[ci], cells) || continue                       # buried?
        if any(a -> a.kind == _AXIS_SPAN, key.axes)                 # high-order → order reduction
            push!(out, (LinearConstraint{T}([raw], [one(T)]), :coverage))
        elseif any(k -> all(ci -> _covered_by_level(cell_box(level.mesh, ci), k, tol), cells),
                   nested_above)                                    # linear reproduced by a nested level
            push!(out, (LinearConstraint{T}([raw], [one(T)]), :dedup))
        end
    end
    return out
end

# Generic fallback: no order reduction (e.g. the B-spline family).
function _coverage_constraints(::Level{D,T,B}, ::Space{D,T}, ::Coverage{D}, ::GeometryTolerance{T},
                               ::AbstractVector{Pair{TensorDofKey{D},Int}}) where {D,T,B}
    return Tuple{LinearConstraint{T},Symbol}[]
end

# Combine repeated raws in a list of `(raw, coefficient)` pairs by
# summing the coefficients, dropping terms whose coefficient falls
# below the working tolerance. Pure-data helper consumed by the
# cascade resolver below; sorts in place for compactness.
function _combine_terms(terms::Vector{Tuple{Int,T}}) where {T}
    length(terms) <= 1 && return terms
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
# form. Cascade depth is bounded by the spatial dimension `D` for our
# overlay-boundary constraints, so the total back-substitution work is
# `O(K · D · p)` across all constraints — negligible at problem scale.
function _back_substitute!(raw_expansion::Vector{Vector{Tuple{Int,T}}}, pivot_raw::Int,
                           pivot_expansion::Vector{Tuple{Int,T}},
                           pivoted::AbstractVector{Bool}) where {T}
    @inbounds for raw in eachindex(raw_expansion)
        raw == pivot_raw && continue
        pivoted[raw] || continue
        expansion = raw_expansion[raw]
        # Quick check: does this expansion mention the new pivot?
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
        raw_expansion[raw] = _combine_terms(new_terms)
    end
    return raw_expansion
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

        # 5) Back-substitute into earlier pivots that mention this raw.
        _back_substitute!(raw_expansion, pivot_raw, new_expansion, pivoted)
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
    dof_layout(V::Space; dirichlet=[], tolerance=GeometryTolerance(T), components=1) -> DofLayout

Construct the basis-aware global dof layout for a superposition
[`Space`](@ref). The construction proceeds in four stages:

  1. Walk every level's cells in order and build the cell-local raw dof
     id vector through `TensorDofKey` lookup. Adjacent cells share
     endpoint nodes; bubble (span) modes are cell-local. Inactive cells
     get an empty vector.
  2. Collect homogeneous linear constraints from every level via the
     family-dispatched [`_overlay_constraints`](@ref) hook, then resolve
     them into the per-raw expansion table via
     [`_resolve_constraints!`](@ref). The resulting
     `raw_expansion[raw]` is either the identity `[(raw, 1)]` for free
     raws, the empty list `[]` for strongly eliminated raws, or a
     multi-element list for linear-constraint pivots.
  3. Per-component physical Dirichlet detection via
     [`_has_physical_dirichlet`](@ref).
  4. Enumerate active dofs component-major (component 1 first, then
     component 2, …) skipping every (raw, component) entry that is
     either physically Dirichlet-constrained or a constraint pivot.
     Project nonzero Dirichlet data onto the boundary trace space via
     [`_project_dirichlet_values!`](@ref).

Keyword arguments:

  - `dirichlet` — iterable of [`DirichletCondition`](@ref)s.
  - `tolerance` — `GeometryTolerance` used by boundary detection.
  - `components` — scalar channels per field (≥ 1).
  - `classify_cache` — the space's cell-classification cache (shared with the
    `PhysicalDomain` fold). The fictitious-fold constraint predicate reuses it
    instead of re-classifying fold-boundary cells; defaults to a fresh empty
    cache for standalone calls.

The integrated Legendre family produces single-raw constraints,
reducing the resolved expansion to strong elimination (`raw_expansion =
[]` for the pivoted raw); the assembly path then behaves identically
to the pre-constraint-primitive code. B-spline families produce
multi-raw constraints encoding the C^m trace-vanishing condition on
artificial boundaries; the assembly path distributes entries through
the expansion automatically.
"""
function dof_layout(V::Space{D,T}; dirichlet=[], tolerance=GeometryTolerance(T),
                    components::Integer=1,
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
    for level in V.levels
        level_cells = Array{Vector{Int},D}(undef, level.mesh.cells)
        for cell in cell_indices(level.mesh)
            if is_active(level.mask, cell)
                level_cells[cell] = _build_cell_dofs!(raw_by_key, raw_keys, level, cell)
            else
                level_cells[cell] = Int[]
            end
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
    # diagnostics can pick an order-reduced or deduped raw out. Overlay-boundary
    # (and multi-raw B-spline) eliminations are not recorded here — they fall
    # through to the `:overlay` default when `elimination_source` is built below.
    source_of = Dict{Int,Symbol}()
    reduce_any = any(level -> level.reduce_order, V.levels)
    coverage = reduce_any ? build_coverage(V, tolerance) : Coverage{D}(Dict{Int,BitArray{D}}())
    for level in V.levels
        level_keys = get(keys_by_level, level.id, empty_keys)
        # Source 1: the artificial-overlay-boundary trace condition (every family).
        append!(constraints,
                _overlay_constraints(level, V, tolerance, raw_by_key, level_keys, classify_cache))
        # Source 2: order reduction in covered regions (opt-in per level).
        if level.reduce_order
            for (c, src) in _coverage_constraints(level, V, coverage, tolerance, level_keys)
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
  - `:coverage`   — order-reduction elimination (covered high-order mode).
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
