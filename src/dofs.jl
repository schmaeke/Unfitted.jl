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
#     vanish on (`_has_overlay_constraint`).
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
# A layout has two halves. The *space block* — the raw dofs, each level's
# cell → raw tables and raw range, every raw's artificial elimination
# source and the linear-constraint pivots — is a function of the space
# alone, so it is built once per distinct space and every field over that
# space holds the same arrays. The *field block* — the per-component
# physical-Dirichlet flags, the active numbering and the constrained
# values — depends on one field's components and Dirichlet data. A layout
# is built in four steps, each reading only what the steps before it
# produced:
#
#   1. Topology (`_dof_topology`, here): raw enumeration, then the
#      artificial eliminations, recorded level by level in an
#      `EliminationLedger` through the basis family's hooks. Its product,
#      a `_DofTopology`, is exactly the space block.
#   2. Physical-Dirichlet detection (`_physical_dirichlet`, `dirichlet.jl`).
#   3. Numbering (the `DofLayout(topology, physical_dirichlet)` constructor,
#      here).
#   4. Projection of nonzero Dirichlet data onto the boundary trace space
#      (`_project_dirichlet_values!`, `dirichlet.jl`).
#
# `model.jl` composes the four, for one field (`dof_layout(V; …)`) and for
# every field of a problem (`system_layout`, step 1 once per space), and
# reads the projection's statistics into the diagnostics there.
#
# The file is in two parts. The first is family-agnostic: the dof keys and
# the layout structs; the elimination ledger, the raising `_resolve_rows!`
# default and the hooks' typed context; steps 1 and 3; the boundary-side
# geometry shared with `dirichlet.jl`; the generic defaults of the
# artificial-boundary and pruning hooks; and the accessors. The second,
# from the "Integrated Legendre family" divider to the end, is that
# family's side of the dof-layer interface: its key tags and keys, its
# boundary association, its artificial-boundary predicate, the hp minimum
# rule, its covered-mode pruning and dedup, and the `_spans_hats` trait.
# The fallbacks of `_tensor_dof_key` and `_spans_hats` sit beside this
# family's methods. The B-spline family implements the same hooks in
# `ext/UnfittedBasicBSplineExt.jl`. The file calls nothing defined in a
# later file.

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
# is fully isbits and, hashed field by field (the `Base.hash` methods
# below `TensorDofKey`), cheap to look up in the cross-cell dof reuse cache.
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

# The default hash of an isbits struct hashes its bytes through a generic
# path that costs ~91 ns per `TensorDofKey{3}`; hashing the fields costs
# ~10 ns, and it was 55 % of the layout's stage-1 profile. Equality stays
# the default `===`, which these methods agree with.
Base.hash(k::AxisDofKey, h::UInt) = hash(k.mode, hash(k.index, hash(k.kind, h)))
Base.hash(k::TensorDofKey, h::UInt) = hash(k.axes, hash(k.level, h))

"""
    DofLayout{D,T}

Basis-aware global dof numbering for a single field over a superposition
[`Space`](@ref). Constrained and active dofs are kept explicitly
distinguishable: raw dofs are enumerated first (every basis function
across every level and every active cell), then constraints are detected
and the active enumeration assigns positive ids to the unconstrained
dofs only.

The fields fall into two blocks. The **space block** is a function of the
space alone, never of the field's components or Dirichlet data. It is what
the first step of the build, the dof topology, produced: the layout holds
the topology's arrays themselves, not copies, and nothing mutates them
once they are built. `system_layout` builds one topology per distinct
space, so the layouts of all fields over one space hold the *same* arrays
(`===`) — a write through one of them would change every other field over
the space, which is why they are never written after the build, by the
package or by a caller. The **field block** is the field's own, never
shared: `physical_dirichlet`, `active_component` and `constrained_values`
are separate arrays for every field, even two over one space. Build a
layout with [`dof_layout`](@ref) for one field, or with `system_layout`
for every field of a problem.

The space block:

  - `cell_dofs_by_level::Vector{Array{Vector{Int},D}}` — indexed by
    level *id*, not by position in `V.levels`, so that it matches the
    `parent.level` id assembly carries; a subdomain whose ids were
    reindexed into a higher block leaves the leading slots undefined,
    and [`cell_dofs`](@ref) refuses them with an `ArgumentError`.
    Each entry is a `D`-dimensional array of cell-local raw dof id
    vectors, indexed by the level's `CartesianIndex`. Inactive cells
    carry an empty vector.
  - `raw_keys::Vector{TensorDofKey{D}}` — `raw_keys[i]` is the
    `TensorDofKey` that produced raw dof `i`. The inverse of the
    construction dictionary.
  - `level_raws::Vector{UnitRange{Int}}` — indexed by level *id*, like
    `cell_dofs_by_level`: the raws level `id` enumerated, as one
    contiguous range, so `raw_keys[r].level == id` exactly for the `r` in
    `level_raws[id]`. Contiguous by construction: a key carries its level
    id, so walking one level's cells can only create or reuse that level's
    keys. A dormant level (no active cell) has an empty range, and an id
    the space does not hold reads `1:0`. Every per-level pass — the
    constraint hooks, physical-Dirichlet detection, the per-level
    diagnostics — reads its level's raws off this range instead of
    scanning, bucketing or looking up the level of every raw.
  - `elimination_source::Vector{Symbol}` — `raw → :free / :overlay /
    :coverage / :dedup`. `:free` iff the raw survives; otherwise the source
    that eliminated it: the artificial overlay boundary (`:overlay`) or, from
    the pruning extension, a covered high-order mode (`:coverage`) or a
    deduped covered vertex (`:dedup`). Applies to every component
    simultaneously. It is the `source` vector of the
    [`EliminationLedger`](@ref) the layout's artificial eliminations were
    recorded in, so a raw is credited to the source that actually removed
    it. Serves as both the elimination flag (`!== :free`) and its
    provenance, read by [`constraint_kind`](@ref), the active enumeration,
    and diagnostics.
  - `pivots::Vector{Vector{Tuple{Int,T}}}` — the linear-constraint pivots,
    and nothing else. A pivot is a raw that constraint resolution expressed
    through *other* raws, `u_raw = Σᵢ wᵢ u_{rᵢ}`, and `pivots[raw]` is that
    expansion `[(r₁, w₁), …]` in resolution order. It has depth one: every
    `rᵢ` is free with respect to the artificial constraints, so a reader
    never chases a chain, and the topology build checks this once. Like
    `elimination_source`, it comes from the ledger. The table is empty when
    the layout has no pivot, which
    is every integrated-Legendre and every maximal-continuity B-spline
    layout: only the multi-raw trace constraints of a B-spline level below
    maximal continuity (on a `LevelMask` face, for instance) produce one.
    Otherwise it holds one entry per raw, and every raw that is not a pivot
    — free or strongly eliminated alike, which `elimination_source` tells
    apart — shares one empty vector; like the shared empty `cell_dofs`
    vector, it must never be pushed into. Assembly keeps a pivot's own
    local row and column while it integrates a region and condenses them
    onto the pivot's active branches once per region (`K ← PᵀKP`,
    `b ← Pᵀb − lift`); [`dof_value`](@ref) reconstructs a pivot from the
    same expansion. Both read it through `_fold_pivot`, which returns at
    once on an empty table, so a layout without pivots never reads an
    entry. [`has_linear_constraints`](@ref) is `!isempty(pivots)`.
  - `tolerance::GeometryTolerance{T}` — the tolerance the topology was
    built at and boundary detection ran at. A `FacetResolver` the layout's
    Dirichlet data are projected through must carry the same one.

The field block:

  - `components::Int` — number of scalar channels the field carries
    (1 for scalar fields, > 1 for vector fields). Constraints are
    tracked per-component so vector boundary conditions can pin
    individual channels.
  - `physical_dirichlet::Matrix{Bool}` — `(raw, component) → true` iff
    component `component` of raw dof `raw` has a physical Dirichlet
    constraint.
  - `active_component::Matrix{Int}` — `(raw, component) → active id`.
    Entry is `0` when component `component` of raw dof `raw` is
    constrained, physically or artificially.
  - `active_count::Int` — total number of active (i.e. enumerated) dofs.
  - `constrained_values::Matrix{T}` — `(raw, component) → value` for
    constrained dofs. Filled by [`_project_dirichlet_values!`](@ref)
    for nonzero physical Dirichlet data; zero otherwise. The one array of
    a layout that is rewritten after it is built: [`update_dirichlet!`](@ref)
    re-projects into it in place.

What the L² Dirichlet trace solve concluded — which branch each component's
solve took, and which dofs it left undetermined — is not part of the
layout. Both describe the projection rather than the numbering, so
[`diagnostics`](@ref) reports them (`dirichlet_trace_factors`,
`unsupported_dirichlet_dofs`) from the projection that produced them.
"""
struct DofLayout{D,T<:Real}
    cell_dofs_by_level::Vector{Array{Vector{Int},D}}
    raw_keys::Vector{TensorDofKey{D}}
    level_raws::Vector{UnitRange{Int}}
    elimination_source::Vector{Symbol}
    pivots::Vector{Vector{Tuple{Int,T}}}
    tolerance::GeometryTolerance{T}
    components::Int
    physical_dirichlet::Matrix{Bool}
    active_component::Matrix{Int}
    active_count::Int
    constrained_values::Matrix{T}
end

# The product of the first build step, `_dof_topology`: exactly the space block of
# `DofLayout`, as one value. Physical-Dirichlet detection and numbering both read it
# before any layout exists, which is why it is a type of its own rather than a
# half-built layout. It is transient — nothing keeps it past the build that made
# it, which shares it between the fields over its space — and the numbering
# constructor `DofLayout(topology, physical_dirichlet)` is the one place its arrays
# are handed on, so a field added to the space block cannot reach some layouts and
# miss others.
struct _DofTopology{D,T<:Real}
    cell_dofs_by_level::Vector{Array{Vector{Int},D}}
    raw_keys::Vector{TensorDofKey{D}}
    level_raws::Vector{UnitRange{Int}}
    elimination_source::Vector{Symbol}
    pivots::Vector{Vector{Tuple{Int,T}}}
    tolerance::GeometryTolerance{T}
end

# ── The elimination ledger ────────────────────────────────────────────────────

"""
    EliminationLedger{T}

The record a basis family's dof-layer hooks write a space's artificial
eliminations into: per raw dof, whether it is eliminated and by which source; the
current level's pending linear-constraint rows; and the pivot table the resolved
rows produce. The topology build (`_dof_topology`) makes one, sized to its
space's raw count, runs every level's hooks into it, and keeps its `source` and
`pivots` as the topology's, hence the layout's, `elimination_source` and
`pivots`.

Fields:

  - `source::Vector{Symbol}` — per raw, `:free` until something eliminates it,
    then the source credited with the elimination: `:overlay` (the artificial
    overlay boundary), `:coverage` (a covered high-order mode) or `:dedup` (a
    buried function a covering level reproduces).
  - `rows::Vector{Tuple{Vector{Int},Vector{T},Symbol}}` — the current level's
    homogeneous linear constraints `Σᵢ cᵢ u_{rᵢ} = 0`, as `(raws, coefficients,
    source)` in emission order. Empty between levels.
  - `pivots::Vector{Vector{Tuple{Int,T}}}` — empty until the first pivot, then one
    entry per raw: a pivot's depth-one expansion, and one shared empty vector for
    every other raw, which nothing pushes into. See [`DofLayout`](@ref).

A family writes through three verbs and nothing else: `_eliminate!(ledger, raw,
source)` for a dof-wise strong elimination, `_constrain!(ledger, raws,
coefficients, source)` for a row, and, only from a `_resolve_rows!` method,
`_pivot!(ledger, raw, expansion, source)` for each raw the rows eliminate.

The order is the core's, and it is what makes the attribution exact:

  - Levels are processed one at a time, in `V.levels` order: the level's
    `_overlay_constraints!`, then, under leaf semantics, its
    `_coverage_constraints!`, then the family's `_resolve_rows!` on whatever rows
    the two recorded (the core stores rows but resolves none). A row names only
    its own level's raws, so resolving level by level is the same as resolving
    every row of the space in one pass.
  - Within a level, the direct eliminations from both hooks are applied as they
    arrive, and the first writer wins: a raw is credited to the first source that
    eliminates it. The artificial boundary is recorded first, so a raw that both
    it and pruning remove is credited to the boundary.
  - The level's rows are resolved *after* both hooks, against everything
    eliminated so far; an eliminated raw contributes no term to a row. A row that
    collapses to `0 = 0` eliminates nothing, so it is credited with nothing.
  - A family that needs an elimination ordered *among* its rows records it as the
    one-term row `_constrain!(ledger, [raw], [one(T)], source)`. The resolution
    then decides whether it eliminates anything, and which row is credited. The
    dedup of a B-spline level below maximal continuity is that case: a buried
    function there can also be a term of a trace row, and eliminating it directly
    would change which raws the trace rows pivot.
"""
struct EliminationLedger{T<:Real}
    source::Vector{Symbol}
    rows::Vector{Tuple{Vector{Int},Vector{T},Symbol}}
    pivots::Vector{Vector{Tuple{Int,T}}}
end

# A ledger over `nraw` raws with nothing eliminated, no row pending and no pivot.
function EliminationLedger{T}(nraw::Integer) where {T}
    return EliminationLedger{T}(fill(:free, nraw), Tuple{Vector{Int},Vector{T},Symbol}[],
                                Vector{Tuple{Int,T}}[])
end

# A dof-wise strong elimination of `raw` by `source`. The first source to eliminate a
# raw is the one it is credited to, which is what makes the artificial boundary,
# recorded first, win over pruning on a raw both remove.
function _eliminate!(ledger::EliminationLedger, raw::Integer, source::Symbol)
    ledger.source[raw] === :free && (ledger.source[raw] = source)
    return ledger
end

# One homogeneous linear constraint Σᵢ cᵢ u_{rawsᵢ} = 0 among the current level's raws,
# credited to `source`. It is held until both of the level's hooks have run and is then
# handed to `_resolve_rows!`. The vectors are stored, not copied.
function _constrain!(ledger::EliminationLedger{T}, raws::Vector{Int}, coefficients::Vector{T},
                     source::Symbol) where {T}
    push!(ledger.rows, (raws, coefficients, source))
    return ledger
end

# The resolver's one write: `raw` is eliminated by the row credited `source`, either
# strongly (`expansion` empty) or as the depth-one pivot u_raw = Σᵢ wᵢ u_{rᵢ}.
#
# The pivot table is allocated on the first pivot, as one shared empty vector per raw,
# so a space without pivots, which is every integrated-Legendre and every
# maximal-continuity B-spline space, never allocates it. A raw already eliminated is
# refused: a resolver that pivoted it again would silently overwrite its source and,
# for a pivot, the expansion other readers already rely on.
function _pivot!(ledger::EliminationLedger{T}, raw::Integer, expansion::Vector{Tuple{Int,T}},
                 source::Symbol) where {T}
    ledger.source[raw] === :free ||
        throw(ArgumentError("raw $raw is already eliminated; a resolver records each raw it " *
                            "eliminates exactly once"))
    ledger.source[raw] = source
    isempty(expansion) && return ledger
    if isempty(ledger.pivots)
        none = Tuple{Int,T}[]           # shared by every non-pivot raw; never pushed into
        fill!(resize!(ledger.pivots, length(ledger.source)), none)
    end
    ledger.pivots[raw] = expansion
    return ledger
end

"""
    _resolve_rows!(ledger::EliminationLedger, level::Level) -> ledger

Resolve the linear-constraint rows `level`'s hooks recorded with `_constrain!`.
Called once per level, after `_overlay_constraints!` and `_coverage_constraints!`,
and only if a row is pending. A method receives `ledger.rows` in emission order,
every row naming only `level`'s raws; a raw with `ledger.source[raw] !== :free` is
already eliminated and contributes no term. It must record every raw the rows
eliminate exactly once, with `_pivot!` — an empty expansion for a strong
elimination, else a depth-one expansion over raws that remain free — credited to
the row that eliminated it. It must not otherwise write `ledger.source`, and it
must not clear `ledger.rows`: the caller does.

The core resolves nothing itself: a family that records rows implements this, and
this default raises. The integer-continuity B-spline family in
`ext/UnfittedBasicBSplineExt.jl` is the only one today; its method is the cascade
Gauss elimination `_cascade!`, which lives there with it. The default is written
over `Level{D,T}` with both parameters free, as a family method naturally is
(`Level{D,T,<:F} where {D,T}`): over a bare `Level`, whose `T` is bounded by
`Real`, such a method would be ambiguous with it rather than more specific.
"""
function _resolve_rows!(ledger::EliminationLedger, level::Level{D,T}) where {D,T}
    throw(ArgumentError("the $(basis_name(level.basis)) family recorded $(length(ledger.rows)) " *
                        "linear-constraint row(s) on level $(level.id) but implements no " *
                        "`_resolve_rows!` method to resolve them"))
end

# Every pivot must be expressed over raws that stay free: `_fold_pivot`, `dof_value`
# and assembly's condensation read ONE level of expansion and never chase a chain. A
# family resolver that broke this would make assembly and reconstruction silently
# disagree, so it is checked once per build. The cost is one read per pivot term, and
# nothing at all without pivots.
function _assert_resolved(ledger::EliminationLedger)
    for expansion in ledger.pivots, (branch, _) in expansion
        ledger.source[branch] === :free ||
            throw(ArgumentError("pivot branch $branch is itself eliminated: `_resolve_rows!` " *
                                "must leave every pivot expressed over free raws"))
    end
    return nothing
end

# Everything a dof-layer hook may read about the level it is called for: the space,
# its physical domain and tolerance, the space's cell-classification cache, the
# coverage masks (empty when the build does not prune), the raw keys and the key → raw
# map of stage 1 (for a family that addresses functions by `(cell, mode)`), and the
# level's own raws.
#
# `P` is the CONCRETE type of `V.physical`. `Space.physical` is abstractly typed, so a
# hook that read it off the space dispatched dynamically on every per-raw predicate
# call it fed it to. The struct is built from the run-time value of `V.physical`,
# which gives it the concrete `P` wherever it is built, and the hook it is handed
# to is the function barrier: the hooks, and every predicate they call, specialise
# on `P` instead. Measured on the integrated-Legendre artificial-boundary pass over
# an immersed disc at 128² p2 (29 281 raws): 304 bytes per raw and 1.26 ms with
# `P = Any`, nothing and 0.24 ms typed.
struct _LevelContext{D,T,S<:Space{D,T},P}
    space::S
    physical::P
    tolerance::GeometryTolerance{T}
    classify_cache::_ClassifyCache{D,T}
    coverage::Coverage{D}
    raw_keys::Vector{TensorDofKey{D}}
    raw_by_key::Dict{TensorDofKey{D},Int}
    raws::UnitRange{Int}
end

# ── DofLayout construction ────────────────────────────────────────────────────

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

# Stage 1 of the layout: walk every level's cells and assign raw dofs through
# the `TensorDofKey` cache (cross-cell endpoint sharing happens here). Levels go
# in `V.levels` order, cells in `cell_indices` order, local functions in
# `cell_basis_indices` order, and a raw's id is the order of its key's first
# appearance. Returns the key → raw map, `raw_keys`, the per-level cell tables
# and the per-level raw ranges.
#
# The ranges cost one `UnitRange` per level and replace a pass that bucketed
# every key by level through a `Dict` — in that `Dict`'s hash order, which is
# also the order the constraint hooks then visited a level's raws in. They are
# exact because a key carries its level id: walking level L can only create or
# reuse L's keys, so L's raws are precisely the ids appended while L is walked,
# and visiting them is visiting them in raw order.
function _enumerate_raws(V::Space{D}) where {D}
    raw_by_key = Dict{TensorDofKey{D},Int}()
    raw_keys = TensorDofKey{D}[]
    # Indexed by level *id*, not by position: for a single-domain space the two
    # coincide (base = 1, overlays 2, 3, …), but a subdomain of a coupled model
    # is reindexed into a disjoint id block by `_reindex_space_levels`, so this
    # layout's ids run over that block and match the `parent.level` id the
    # assembly workspace and `cell_dofs` key by (leading slots below the block
    # stay `undef` and are never addressed by this field's parents).
    nid = maximum(l -> l.id, V.levels)
    cell_dofs_by_level = Vector{Array{Vector{Int},D}}(undef, nid)
    level_raws = fill(1:0, nid)

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
        first_raw = length(raw_keys) + 1
        level_cells = Array{Vector{Int},D}(undef, level.mesh.cells)
        for cell in cell_indices(level.mesh)
            level_cells[cell] = is_active(level.mask, cell) ?
                                _build_cell_dofs!(raw_by_key, raw_keys, level, cell) : none
        end
        cell_dofs_by_level[level.id] = level_cells
        level_raws[level.id] = first_raw:length(raw_keys)
    end
    return raw_by_key, raw_keys, cell_dofs_by_level, level_raws
end

# Step 1 of a layout: the space block, built from the space alone. Raw enumeration
# (`_enumerate_raws`), then every level's artificial eliminations recorded in one
# `EliminationLedger`, level by level in `V.levels` order: the family's
# `_overlay_constraints!`, then, under leaf semantics, its `_coverage_constraints!`
# against the masks `build_coverage` computes (so those are built first, and not at
# all when `prune = false`), then `_resolve_rows!` on whatever linear-constraint rows
# the two recorded — and finally the depth-one check on the pivots the rows resolved
# to. The ledger's per-raw sources and pivot table become the topology's. The core
# owns this order and the pruning gate (`prune`, `prune_exempt`), so no family
# re-implements either: a family that forgot the gate would prune the unreduced twin.
#
# Each hook call is the function barrier its per-raw loop runs behind. The level's
# context is built from the run-time value of `V.physical`, so its `P` is concrete
# (see `_LevelContext`) even though the field is not, and `V.levels` may mix basis
# families; a hook therefore runs specialised on both, and what dispatches at run
# time is at most the context's construction and the three calls per level.
#
# A topology reads neither a field's components nor its Dirichlet data, so one is
# the same for every field over the space, and `_system_layout` builds it once per
# space and shares it between them.
function _dof_topology(V::Space{D,T}; tolerance::GeometryTolerance{T}=GeometryTolerance(T),
                       prune::Bool=true, prune_exempt=(),
                       classify_cache::_ClassifyCache{D,T}=_ClassifyCache{D,T}()) where {D,T}
    raw_by_key, raw_keys, cell_dofs_by_level, level_raws = _enumerate_raws(V)
    coverage = prune ? build_coverage(V, tolerance, classify_cache) :
               Coverage{D}(Dict{Int,BitArray{D}}())
    ledger = EliminationLedger{T}(length(raw_keys))
    for level in V.levels
        ctx = _LevelContext(V, V.physical, tolerance, classify_cache, coverage, raw_keys,
                            raw_by_key, level_raws[level.id])
        _overlay_constraints!(ledger, level, ctx)
        prune && !(level.id in prune_exempt) && _coverage_constraints!(ledger, level, ctx)
        isempty(ledger.rows) && continue
        _resolve_rows!(ledger, level)
        empty!(ledger.rows)
    end
    _assert_resolved(ledger)
    return _DofTopology{D,T}(cell_dofs_by_level, raw_keys, level_raws, ledger.source, ledger.pivots,
                             tolerance)
end

# Step 3 of a layout: number one field over a topology, given its physical-Dirichlet
# flags (step 2, `_physical_dirichlet`). Active ids run component-major — component
# 1 of every raw, then component 2, … — skipping every (raw, component) a physical
# Dirichlet condition constrains or an artificial elimination removed, so an
# all-or-nothing constraint leaves the natural block ordering and a per-component
# one leaves holes. The constrained values start at zero, which is already the
# answer for homogeneous data; nonzero data are projected into them afterwards
# (step 4, `_project_dirichlet_values!`).
#
# The topology's arrays are handed on, not copied: the space block is never mutated
# once built, and this is the one place a layout receives it.
function DofLayout(t::_DofTopology{D,T}, physical_dirichlet::Matrix{Bool}) where {D,T}
    nraw, ncomp = size(physical_dirichlet)
    active_component = zeros(Int, nraw, ncomp)
    active_count = 0
    for c in 1:ncomp, raw in 1:nraw
        (physical_dirichlet[raw, c] || t.elimination_source[raw] !== :free) && continue
        active_component[raw, c] = (active_count += 1)
    end
    return DofLayout{D,T}(t.cell_dofs_by_level, t.raw_keys, t.level_raws, t.elimination_source,
                          t.pivots, t.tolerance, ncomp, physical_dirichlet, active_component,
                          active_count, zeros(T, nraw, ncomp))
end

# ── Geometric helpers shared with `dirichlet.jl` ──────────────────────────────

# Tolerance-aware coordinate equality used throughout boundary
# detection. Pulled out so the `<= tol.contain` convention is in one
# place; consumed both by the overlay-constraint detection below and by
# the physical-Dirichlet detection in `dirichlet.jl`.
_coordinate_matches(a::Real, b::Real, tol::GeometryTolerance) = abs(a - b) <= tol.contain

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

# ── The artificial-boundary and pruning hooks ─────────────────────────────────

"""
    _overlay_constraints!(ledger::EliminationLedger, level::Level, ctx::_LevelContext) -> ledger

Record `level`'s artificial-overlay-boundary condition in `ledger`. Dispatched on
the level's basis family, so each family can exploit its own boundary mode
structure. Called once per level, before `_coverage_constraints!` (see
[`EliminationLedger`](@ref) for the order). A method names only raws in
`ctx.raws`, reads only `ctx` (see `_LevelContext`), and records through
`_eliminate!` and `_constrain!`. Every method ties `{D,T}` across `Level{D,T}` and
`_LevelContext{D,T}`, as this default does, so that no two methods are ambiguous.

This default walks the level's raws, asks the family's per-raw
`_has_overlay_constraint` predicate about each, and strongly eliminates every raw
it accepts with `_eliminate!(ledger, raw, :overlay)`: the regime the integrated
Legendre family relies on. The predicate consults `ctx.classify_cache`, the
space's cell-classification cache, only to tell a fully-fictitious fold face from
a user-mask face; the cache is untouched when the space has no physical domain.

The B-spline family in the extension overrides this default. At `continuity =
:maximal` its method is a support selection, which again eliminates strongly. At
an integer continuity `m` it records multi-raw rows instead: the `Cᵐ`
trace-vanishing condition along every artificial perpendicular line, `m + 1` rows
per line per face, factored through the tensor-product structure of the basis and
looking raw ids up by `(cell, mode)` in `ctx.raw_by_key`. That family therefore
never reaches this default or the per-raw `_has_overlay_constraint`.
"""
function _overlay_constraints!(ledger::EliminationLedger, level::Level{D,T},
                               ctx::_LevelContext{D,T}) where {D,T}
    for raw in ctx.raws
        _has_overlay_constraint(ctx.raw_keys[raw], level, ctx.physical, ctx.space.domain,
                                ctx.tolerance, ctx.classify_cache) &&
            _eliminate!(ledger, raw, :overlay)
    end
    return ledger
end

# Generic fallback: no covered-mode pruning for a family that has not opted in.
function _coverage_constraints!(ledger::EliminationLedger, ::Level{D,T},
                                ::_LevelContext{D,T}) where {D,T}
    ledger
end

# ── DofLayout accessors and the pivot rule ────────────────────────────────────

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

Pre-constraint raw dof ids for the cell at `cell` on the level with id `level`,
in the canonical local-basis order. Inactive cells return an empty vector.
`layout` is a [`DofLayout`](@ref), or the `SystemLayout` of a single-field
problem.

`level` is a level *id*. For a single-domain space that is the level's position
in `V.levels`, but each subdomain of a coupled problem has its ids reindexed
into a block of its own, and its layout holds no cells for an id outside that
block. Such an id, or one no level carries, raises an `ArgumentError`.

The vector is the layout's own and must not be mutated: every inactive cell of
every level shares one empty vector, so a `push!` into a returned empty result
would reach all of them. Copy before editing.
"""
function cell_dofs(layout::DofLayout{D}, level::Integer, cell::CartesianIndex{D}) where {D}
    # One `isassigned` covers both an id past the end of the table and a slot
    # below a reindexed subdomain block, which would otherwise surface as an
    # `UndefRefError` from inside the table. It replaces the range check rather
    # than adding to it, so assembly's `_frame!`, which calls this once per
    # (region, parent, field), pays one test as before, and the read after it is
    # in bounds.
    tables = layout.cell_dofs_by_level
    isassigned(tables, level) ||
        throw(ArgumentError("level id $level is not a level of this layout's space"))
    return @inbounds(tables[level])[cell]
end

"""
    active_cell_dofs(layout::DofLayout, level::Integer, cell::CartesianIndex, component=1) -> Vector{Int}
    active_cell_dofs(layout::SystemLayout, level::Integer, cell::CartesianIndex, component=1) -> Vector{Int}
    active_cell_dofs(layout::SystemLayout, field::Symbol, level::Integer, cell::CartesianIndex,
                     component=1) -> Vector{Int}

Active ids for component `component` of the cell-local raw dofs, with `0`
standing for any dof that was eliminated by a constraint. The order matches the
canonical local-basis ordering (same length as `cell_dofs(layout, level, cell)`),
and `level` is a level id, as there.

On a `DofLayout` the ids are the field's own, in `1:active_unknowns(layout)`. On
a [`SystemLayout`](@ref) they are global: shifted by the field's `offset` into
the system's active-dof vector, so they index a [`Solution`](@ref)'s
coefficients directly. The second form reads the only field of a single-field
problem; the third names the field.
"""
function active_cell_dofs(layout::DofLayout{D}, level::Integer, cell::CartesianIndex{D},
                          component::Integer=1) where {D}
    return [layout.active_component[raw, component] for raw in cell_dofs(layout, level, cell)]
end

"""
    constraint_kind(layout, raw, component=1) -> Symbol

Classify component `component` of a single raw dof. `layout` is a
[`DofLayout`](@ref), or the `SystemLayout` of a single-field problem. Returns
one of:

  - `:free`       — no constraint, the dof is enumerated.
  - `:dirichlet`  — physical Dirichlet only.
  - `:overlay`    — artificial overlay-boundary elimination only.
  - `:coverage`   — pruning elimination (covered high-order mode).
  - `:dedup`      — linear-dedup elimination (covered vertex reproduced by
                    a nested finer level).
  - `:mixed`      — physical Dirichlet plus an elimination. The
                    elimination wins: the dof is eliminated, and its
                    [`constrained_value`](@ref) is zero, the homogeneous
                    artificial constraint's, whatever the Dirichlet data.
                    The Dirichlet projection solves only for the physical
                    dofs no artificial elimination removed, so it never
                    writes a value here.
"""
function constraint_kind(layout::DofLayout, raw::Integer, component::Integer=1)
    physical = layout.physical_dirichlet[raw, component]
    source = layout.elimination_source[raw]
    physical && source != :free && return :mixed
    physical && return :dirichlet
    return source
end

"""
    constrained_value(layout, raw, component=1) -> T

Stored value of a constrained (raw, component). Zero for every artificial
elimination (overlay boundary, pruning, and a `:mixed` dof, see
[`constraint_kind`](@ref)) and for homogeneous Dirichlet conditions; the
projected value for nonzero Dirichlet data. `layout` is a
[`DofLayout`](@ref), or the `SystemLayout` of a single-field problem.
"""
function constrained_value(layout::DofLayout, raw::Integer, component::Integer=1)
    layout.constrained_values[raw, component]
end

"""
    has_linear_constraints(layout::DofLayout) -> Bool

Whether the layout carries a linear-constraint pivot: a raw that constraint
resolution expressed through other raws (see [`DofLayout`](@ref)'s `pivots`),
which assembly condenses and [`dof_value`](@ref) reconstructs. `false` for every
integrated-Legendre and every maximal-continuity B-spline layout, whose
eliminations are all strong; assembly reads it per field and skips the pivot
search when it is `false`.
"""
has_linear_constraints(d::DofLayout) = !isempty(d.pivots)

# The pivot rule, stated once. A linear-constraint pivot is
#
#     u(raw, c) = Σₖ wₖ · u(oₖ, c)
#
# over constraint-free raws `oₖ`: `_resolve_rows!` leaves every pivot expressed
# at depth one, and `_assert_resolved` checks it once per build. Two readers
# share this fold — `dof_value`, which reconstructs a pivot from it, and
# `_slots!` in assembly, which detects a region's pivots, widens its active set
# to their branches and records those branches for `_condense!` to fold the
# pivot's local row and column onto. When reconstruction and emission each
# carried their own encoding they could disagree, and a field was then solved
# correctly but reconstructed wrong; one definition rules that out.
#
# `f(acc, g, w, v)` is folded over the branches in expansion order, where
# `g = id(oₖ, c)` is the branch's active id (0 when it is constrained) and `v` its
# constrained value when `g == 0`, zero otherwise. `id` maps `(raw, component)` to
# the active id in whichever numbering the caller needs; both readers pass a
# field's global one, `_field_component_dof`. Returns
# `(acc, true)` for a pivot. A non-pivot returns `(acc, false)` without calling
# `f`, and needs no test of its own: the `pivots` table holds pivots only, so a
# free raw, a strongly eliminated one and a constraint-free raw a Dirichlet
# condition eliminated on `c` all read an empty entry — or an empty table, when
# the layout has no pivot at all.
@inline function _fold_pivot(f::F, acc, d::DofLayout, id::I, raw::Integer, c::Integer) where {F,I}
    isempty(d.pivots) && return acc, false
    expansion = d.pivots[raw]
    isempty(expansion) && return acc, false
    for (other, w) in expansion
        g = id(other, c)
        acc = f(acc, g, w, g == 0 ? constrained_value(d, other, c) : zero(w))
    end
    return acc, true
end

# The `_fold_pivot` step of `dof_value`: add a branch's weighted value, read from
# the coefficients when the branch is active and from its stored constrained
# value otherwise — `value += wₖ · u(oₖ, c)`, in expansion order.
_branch_value(coefficients) = (acc, g, w, v) -> acc + w * (g == 0 ? v : coefficients[g])

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

# Resolve a field name to its index inside a `SystemLayout`, raising on an
# unknown name. How the assembly kernel's blocks and loads, an interface's region
# build and a `FormState` read find a field in the layout's field-major data.
function _field_index(system::SystemLayout, name::Symbol)
    index = get(system.by_name, name, 0)
    index == 0 && throw(ArgumentError("unknown field $name"))
    return index
end

# Look up the per-field layout by name. Throws on unknown name.
_field_layout(system::SystemLayout, name::Symbol) = system.fields[_field_index(system, name)]

# Global active dof id for component `component` of raw `raw` in this
# field, or 0 if the (raw, component) is constrained. Per-field active
# ids are shifted by the field's `offset` so they live in disjoint
# blocks of the global active enumeration. Read once per slot by
# assembly's `_frame!` and per branch by `_slots!`, so it stays one table
# read and one add.
function _field_component_dof(layout::FieldLayout, raw::Integer, component::Integer)
    active = layout.dofs.active_component[raw, component]
    return active == 0 ? 0 : layout.offset + active
end

"""
    dof_value(layout::FieldLayout, coefficients, raw, component=1) -> T

Recover the value of component `component` of raw dof `raw` in this field — the
unified accessor every reconstruction path uses when a basis-function
contribution needs the actual dof value regardless of constraint status.
`coefficients` is the system's global active vector, a [`Solution`](@ref)'s
coefficients, and the field's `offset` is resolved here.

An active dof reads `coefficients[offset + active]`. A constrained one reads its
[`constrained_value`](@ref) — zero for an artificial-boundary or pruning
elimination, the projected datum for a nonzero Dirichlet condition — and, if it is
a **linear-constraint pivot**, adds the pivot's expansion from the field's
`pivots` table (see [`DofLayout`](@ref)):

    u_raw = constrained_value(raw) + Σᵢ wᵢ · u_{rᵢ} .

That pivot term is not an embellishment. A pivot is precisely the mechanism a
basis family uses to impose C^m continuity or an arbitrary-mask trace condition,
so reading `constrained_value` for it — which is zero — silently drops the dof's
whole contribution from `value`, `field_gradient`, `l2_error`, `write_vtk` and
`L2Projection`'s source reconstruction, while the *solve* is correct because
assembly condenses every pivot through the same expansion. Measured on a
6×6 degree-3 C¹ B-spline base under an L-masked 4×4 C¹ overlay (8 non-trivial
expansions of 126 raws), the dropped term reached 4.79e-7 against a field of
magnitude 7e-2 — silent, and small enough to pass a hand-written tolerance.

One level of expansion is enough, because `_resolve_rows!` leaves every pivot
expressed over raws the artificial constraints leave free, and the topology build
checks it once (`_assert_resolved`). Such a branch may still be constrained on
`component` by a physical Dirichlet condition, which is resolved later; it then
contributes its constrained value rather than a coefficient.

A layout without pivots — every integrated-Legendre layout — stops at the test
for an empty table, before any per-raw read.

The expansion is read through `_fold_pivot`, the one statement of the pivot rule
that assembly reads as well, so a pivot cannot be reconstructed one way and
assembled another.
"""
function dof_value(layout::FieldLayout, coefficients, raw::Integer, component::Integer=1)
    global_dof = _field_component_dof(layout, raw, component)
    global_dof == 0 || return coefficients[global_dof]
    value, _ = _fold_pivot(_branch_value(coefficients),
                           constrained_value(layout.dofs, raw, component), layout.dofs,
                           (other, c) -> _field_component_dof(layout, other, c), raw, component)
    return value
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

# True iff any field's dof layout carries a linear-constraint pivot (a raw
# redistributed onto other raws; see [`DofLayout`](@ref)'s `pivots`). Assembly
# itself asks per field; this system-wide form serves tests and diagnostics.
function has_linear_constraints(layout::SystemLayout)
    any(field -> has_linear_constraints(field.dofs), layout.fields)
end

# Resolve the unique field of a single-field layout. Used by the
# `*_dofs` delegates below so single-field problems do not have to
# carry a field name argument.
function _only_field_layout(layout::SystemLayout)
    length(layout.fields) == 1 ||
        throw(ArgumentError("field argument is required for multi-field layouts"))
    return only(layout.fields)
end

# The system forms of the per-dof accessors, one method per verb, documented
# with the `DofLayout` forms above. Without a field name they read the only
# field of a single-field problem, so its callers need not name it; the
# active-id form also takes a name, for a multi-field problem. Each takes the
# same `component` as the `DofLayout` form it forwards to, and the active ids
# come back global, shifted by the field's offset — the numbering a
# `Solution`'s coefficients use.
function cell_dofs(layout::SystemLayout, level::Integer, cell::CartesianIndex)
    cell_dofs(_only_field_layout(layout).dofs, level, cell)
end

function active_cell_dofs(layout::SystemLayout, level::Integer, cell::CartesianIndex,
                          component::Integer=1)
    active_cell_dofs(layout, _only_field_layout(layout).name, level, cell, component)
end

function active_cell_dofs(layout::SystemLayout, name::Symbol, level::Integer, cell::CartesianIndex,
                          component::Integer=1)
    field = _field_layout(layout, name)
    return [_field_component_dof(field, raw, component)
            for raw in cell_dofs(field.dofs, level, cell)]
end

function constraint_kind(layout::SystemLayout, raw::Integer, component::Integer=1)
    constraint_kind(_only_field_layout(layout).dofs, raw, component)
end

function constrained_value(layout::SystemLayout, raw::Integer, component::Integer=1)
    constrained_value(_only_field_layout(layout).dofs, raw, component)
end

# ── Integrated Legendre family ────────────────────────────────────────────────

# Tag values for `AxisDofKey.kind`. `UInt8` keeps the key compact and
# fast to hash; the values themselves are arbitrary.
#
#   _AXIS_NODE   — integrated Legendre endpoint mode at a cell-corner
#                  node coordinate.
#   _AXIS_SPAN   — integrated Legendre interior (bubble) mode, cell-local.
#
# A tag need only be unique within one basis family, so each family owns
# its tags: the B-spline extension defines its `_AXIS_BSPLINE` next to its
# own key construction. Keys of two families can never be confused even
# where their tags coincide. A key carries its level id and one level has
# one family, so two keys of different families differ in their level and
# the dof-key cache of a mixed-family space never takes one for the other;
# and `_assert_keys_comparable` (`projection.jl`) refuses to match keys
# across a change of family between two models. The shipped values happen
# to be distinct (the extension's is `0x02`); nothing relies on it.
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

# True iff the per-axis factor of `key` along `axis` anchors at the
# level's lower (`side = :lower`) or upper (`side = :upper`) mesh edge.
# Dispatched on the level's basis family so each family's "boundary
# mode" convention is local to its own implementation. Used by the
# physical-Dirichlet detection (`_key_faces` in `dirichlet.jl`) to
# translate "is this dof at a face" into a basis-family-specific predicate
# without leaking the family's mode-index conventions into the rest of the
# dof layer. The integrated-Legendre `_has_overlay_constraint` below does
# not route through it: it already walks the key's node factors and reads
# the mesh-edge indices (`1` and `cells[axis] + 1`) off them directly.
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

# ── Artificial overlay-boundary detection ─────────────────────────────────────

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
# and active under the level's `LevelMask`. Used by `_has_overlay_constraint`
# to probe both sides of a candidate face for an active/inactive transition.
# A position off the mesh reads as inactive, so a mesh-box face is a
# transition wherever the cell just inside it is active.
function _cell_active(level::Level{D}, axis::Integer, pos::Integer, outer::NTuple{D,Int},
                      n::NTuple{D,Int}) where {D}
    1 <= pos <= n[axis] || return false
    cell_tuple = ntuple(d -> d == axis ? pos : outer[d], D)
    return is_active(level.mask, CartesianIndex(cell_tuple))
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
# One walk per node axis decides it. Only an `_AXIS_NODE` factor sits on a
# face: at node index `i` it is the face between cells `i − 1` and `i` along
# `axis`, and the key's incidence on the other axes (`_perp_ranges`) gives
# the perpendicular positions the face spans under the key's support. Every
# position where activity changes across the face is a face of the active
# region, and
#
#   * on a mesh-box face (`i == 1` or `i == n[axis] + 1`) the first such
#     position settles the axis: the face constrains unless the level side
#     lies on the physical boundary (`_level_side_is_physical`), and the rest
#     of the face cannot say otherwise;
#   * on an internal face, a position constrains when the inactive cell
#     across it is not `_is_fictitious`.
#
# The second case is the fold exemption. The fictitious fold deactivates
# fully-fictitious cells, so a fold face sees no material on its inactive
# side, and its boundary modes stay active. They vanish on every physical
# face — so they cannot break the C⁰ trace condition the overlay constraint
# enforces — and instead carry the adjacent cut cell's approximation up to
# ∂Ω. A cell the *user* masked out still holds material: the modes on its
# face are non-zero on a physical face and must be eliminated to keep the
# truncated solution continuous. With no physical domain there is no
# fictitious material, so every internal active/inactive face constrains
# (the original overlay/user-mask behaviour) and `cache` is never touched.
# The walk stops at the first constraining position, so a fold face
# classifies its inactive neighbours only until one of them holds material.
#
# This runs once per raw on every layout build, so two measured choices keep
# it as cheap as the two-walk predicate it replaced: `@inline`, without which
# the call stays out of `_overlay_constraints!`'s per-raw loop (1.1–1.8× the
# pass time on the layout probe's spaces), and the side test at the first
# transition rather than ahead of the walk, so an internal face without one
# pays for nothing but the walk (ahead of it: 1.2× on 2D overlays and ladders).
#
# Integrated-Legendre-specific, because the incidence rule is. The B-spline
# family in `ext/UnfittedBasicBSplineExt.jl` does the same job in its own
# `_overlay_constraints!`, with a support map that spans several cells and
# the same fold exemption (`_active_cell_at_face`).
@inline function _has_overlay_constraint(key::TensorDofKey{D},
                                         level::Level{D,T,<:IntegratedLegendre}, physical,
                                         domain::AxisBox{D,T}, tol::GeometryTolerance{T},
                                         cache::_ClassifyCache{D,T}) where {D,T}
    n = level.mesh.cells
    for axis in 1:D
        axis_key = key.axes[axis]
        axis_key.kind == _AXIS_NODE || continue
        i = axis_key.index
        for outer in CartesianIndices(_perp_ranges(key, axis, n))
            below_active = _cell_active(level, axis, i - 1, outer.I, n)
            below_active == _cell_active(level, axis, i, outer.I, n) && continue
            if i == 1 || i == n[axis] + 1
                side = i == 1 ? :lower : :upper
                _level_side_is_physical(level, domain, axis, side, tol) && break
                return true
            end
            across = below_active ? i : i - 1          # the inactive cell's axis position
            inactive = CartesianIndex(ntuple(d -> d == axis ? across : outer[d], D))
            _is_fictitious(level, inactive, physical, cache) || return true
        end
    end
    return false
end

# ── Pruning (coverage) constraint source ──────────────────────────────────────

# Cells of `level` incident to the entity of `key`, from the per-axis incidence
# rule of `_axis_incidence` applied on every axis.
function _incident_cells(key::TensorDofKey{D}, n::NTuple{D,Int}) where {D}
    return CartesianIndices(ntuple(d -> _axis_incidence(key.axes[d], n[d]), D))
end

# ── The minimum rule: per-cell order across a shared entity ───────────────────
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
#     fictitious fold `_has_overlay_constraint` deliberately keeps the modes
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
# `_coverage_constraints!` lives in its extension.
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
# `_coverage_constraints!`; `_nested_over` (`coverage.jl`) is the geometric half, and a
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
    _coverage_constraints!(ledger::EliminationLedger, level::Level, ctx::_LevelContext) -> ledger

Pruning constraint source, the peer of [`_overlay_constraints!`](@ref) and bound by
the same contract. Called only under leaf semantics (`dof_layout`'s `prune`, on by
default) and only on a level outside `prune_exempt`, after the level's
`_overlay_constraints!`. This integrated-Legendre method strongly eliminates

  * every **buried high-order** mode (at least one bubble axis, every active incident
    cell covered by and every fictitious one reached by a single finer level) —
    covered-mode pruning, source `:coverage`; and
  * every **buried linear** mode a single nested level above whose basis spans that
    mesh's hats reproduces exactly — dedup, source `:dedup`.

The linear skeleton is otherwise retained, which is what makes the reduced space
complete. Each elimination is credited with its source for `constraint_kind` /
diagnostics, unless the artificial boundary already eliminated the raw: the first
writer wins (see [`EliminationLedger`](@ref)), so a buried raw on `Γ_o` stays
credited to `:overlay`.

Both halves ask `_reproduced_on_domain` (`coverage.jl`), which applies the
fictitious-fold rule on the covering level *and* on this one — a support cell folded
away carries nothing to reproduce, so it neither has to be covered nor may veto the
verdict. `ctx.classify_cache` is the classification that tells a fold from a user
mask, the same one [`build_coverage`](@ref) used.

The dedup half is not an optimisation: a mode a covering level reproduces exactly is
linearly dependent on that level's own modes, so leaving both active makes the
superposed operator exactly singular. The B-spline extension overrides this hook for
the same reason, and records nothing but dedups — that family has no bubble modes to
shed. Every other family takes the generic fallback above and records nothing.
"""
function _coverage_constraints!(ledger::EliminationLedger, level::Level{D,T,<:IntegratedLegendre},
                                ctx::_LevelContext{D,T}) where {D,T}
    cov = ctx.coverage.covered[level.id]
    any(cov) || return ledger
    V, tol, n = ctx.space, ctx.tolerance, level.mesh.cells
    # A buried vertex function is a C⁰ hat: multilinear on each of `level`'s cells,
    # kinked across every cell boundary. Mesh nesting (`_nested_over`) only buys the
    # multilinear half; the kink is the cover basis's half, and `_spans_hats` is where
    # each family answers for it. Both halves are necessary and neither is about this
    # level's family — a cover that spans the hats deduplicates them whoever generated
    # them.
    nested_above = [k
                    for k in V.levels
                    if k.id > level.id && _spans_hats(k.basis) && _nested_over(level, k, tol)]
    reproduces(k, cells) = _reproduced_on_domain(level, k, cells, ctx.physical, tol,
                                                 ctx.classify_cache)
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
    for raw in ctx.raws
        key = ctx.raw_keys[raw]
        cells = _incident_cells(key, n)
        all(ci -> !is_active(level.mask, ci) || cov[ci], cells) || continue     # buried?
        if any(a -> a.kind == _AXIS_SPAN, key.axes)                 # high-order → covered-mode pruning
            resolves(cells) && _eliminate!(ledger, raw, :coverage)
        elseif any(k -> reproduces(k, cells), nested_above)          # linear reproduced above
            _eliminate!(ledger, raw, :dedup)
        end
    end
    return ledger
end
