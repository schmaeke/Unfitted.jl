# Degree-of-freedom layout for a superposition `Space`. The dof layer answers, for each cell, two questions assembly
# needs:
#
#   * which local basis functions map to which raw (pre-constraint) dof id — produced by walking every level's cells and
#     reusing dof ids across adjacent cells through a `TensorDofKey` lookup;
#   * which raw dofs are constrained, by what kind of constraint, and (for nonzero physical Dirichlet data) to what
#     value.
#
# Physical and artificial constraints are tracked separately and never mixed — the physical ones in
# `physical_dirichlet`, the artificial ones in `elimination_source`:
#
#   * Physical Dirichlet — user-imposed boundary conditions on the physical domain `∂Ω`. Tracked per-component so vector
#     fields can pin only one component along a face (rollers, symmetry, …). Nonzero data are projected onto the
#     boundary trace space by an L² mass-matrix solve so the constrained value list reflects the correct trace at every
#     constrained dof.
#   * Artificial overlay constraint — homogeneous Dirichlet on the overlay's artificial boundary
#     `Γ_o^(k) = ∂Ω^(k) \ ∂Ω`, which on an immersed space is the part of the overlay's box boundary that meets Ω: a box
#     face lying wholly outside Ω carries no trace to vanish on (`_clamps_box_face`). Strongly eliminated for the
#     integrated Legendre basis by removing the corresponding raw dof from active enumeration. Also applied at internal
#     active/inactive cell faces of the same level when a `LevelMask` is in play — except where the inactive side is
#     fully fictitious, since a fold face carries no physical trace to vanish on either (`_has_overlay_constraint`).
#   * Covered-mode pruning — under leaf semantics, the high-order modes buried under a finer level, plus the buried
#     linear modes a nested finer level reproduces exactly. Also a strong elimination, and also artificial: it removes
#     modes the superposition already carries, never a physical condition.
#
# Active dofs are enumerated component-major (component 1 of every raw first, then component 2, …) so the all-or-nothing
# constrained case matches the natural block ordering, and per-component constraints simply leave holes in the
# enumeration.
#
# A layout has two halves. The *space block* — the raw dofs, each level's cell → raw tables and raw range, every raw's
# artificial elimination source and the linear-constraint pivots — is a function of the space alone, so it is built once
# per distinct space and every field over that space holds the same arrays. The *field block* — the per-component
# physical-Dirichlet flags, the active numbering and the constrained values — depends on one field's components and
# Dirichlet data. A layout is built in four steps, each reading only what the steps before it produced:
#
#   1. Topology (`_dof_topology`, here): raw enumeration, then the artificial eliminations, recorded level by level in
#      an `EliminationLedger` through the basis family's hooks. Its product, a `_DofTopology`, is exactly the space
#      block.
#   2. Physical-Dirichlet detection (`_physical_dirichlet`, `dirichlet.jl`).
#   3. Numbering (the `DofLayout(topology, physical_dirichlet)` constructor, here).
#   4. Projection of nonzero Dirichlet data onto the boundary trace space (`_project_dirichlet_values!`,
#      `dirichlet.jl`).
#
# `model.jl` composes the four, for one field (`dof_layout(V; …)`) and for every field of a problem (`system_layout`,
# step 1 once per space), and reads the projection's statistics into the diagnostics there.
#
# The file is in two parts. The first is family-agnostic: the dof keys and the layout structs; the elimination ledger,
# the raising `_resolve_rows!` default and the hooks' typed context; steps 1 and 3; the boundary-side geometry shared
# with `dirichlet.jl`; the generic default of the artificial-boundary hook, the pruning hook — one burial driver for
# every family — and its family hooks' defaults; and the accessors. The second, from the "Integrated Legendre family"
# divider to the end, is that family's side of the dof-layer interface: its key tags and keys, its boundary association,
# its artificial-boundary predicate, the hp minimum rule, and its burial hooks (covered-mode pruning and dedup). The
# fallback of `_tensor_dof_key` sits beside this family's methods. The B-spline family implements the same hooks in
# `ext/UnfittedBasicBSplineExt.jl`. The file calls nothing defined in a later file.

# ── Dof keys and layout structs ───────────────────────────────────────────────

# Per-axis component of a `TensorDofKey`. Together with the axis index it uniquely identifies one factor of a
# tensor-product basis function. `kind` is a tag the basis family owns, and the family reads `index` and `mode` by its
# own convention: integrated Legendre tags an endpoint factor at a node coordinate and a cell-local bubble factor
# (`_AXIS_NODE`, `_AXIS_SPAN`, defined with the family's other hooks under the "Integrated Legendre family" divider),
# and the B-spline extension one factor per univariate B-spline (`_AXIS_BSPLINE`).
#
# Stored as a flat `UInt8 + Int + Int` rather than a sum type so the key is fully isbits and, hashed field by field (the
# `Base.hash` methods below `TensorDofKey`), cheap to look up in the cross-cell dof reuse cache.
struct AxisDofKey
    kind::UInt8
    index::Int
    mode::Int
end

# Tensor-product dof key: one `AxisDofKey` per axis plus the source level's id. Two basis functions share a raw dof iff
# they have the same `TensorDofKey` — i.e. they live on the same level and reduce to the same per-axis (kind, index,
# mode) along every axis. That is how adjacent cells share a function: a family's key names a factor by what it is, not
# by the cell asking, so in integrated Legendre the right endpoint of cell `i` and the left endpoint of cell `i + 1` are
# one `_AXIS_NODE` factor at node `i + 1`, while each bubble stays cell-local.
struct TensorDofKey{D}
    level::Int
    axes::NTuple{D,AxisDofKey}
end

# The default `hash` of a struct goes through `objectid`, which the runtime computes generically — field by field, for
# these padded structs — at ~90 ns per `TensorDofKey{3}`; hashing the fields directly costs ~10 ns. The default was 55 %
# of the raw enumeration's profile. Equality stays the default `===`, which these methods agree with.
Base.hash(k::AxisDofKey, h::UInt) = hash(k.mode, hash(k.index, hash(k.kind, h)))
Base.hash(k::TensorDofKey, h::UInt) = hash(k.axes, hash(k.level, h))

"""
    DofLayout{D,T}

Basis-aware global dof numbering for a single field over a superposition [`Space`](@ref). Constrained and active dofs
are kept explicitly distinguishable: raw dofs are enumerated first (every basis function across every level and every
active cell), then constraints are detected and the active enumeration assigns positive ids to the unconstrained dofs
only.

The fields fall into two blocks. The **space block** is a function of the space alone, never of the field's components
or Dirichlet data. It is what the first step of the build, the dof topology, produced: the layout holds the topology's
arrays themselves, not copies, and nothing mutates them once they are built. `system_layout` builds one topology per
distinct space, so the layouts of all fields over one space hold the *same* arrays (`===`) — a write through one of them
would change every other field over the space, which is why they are never written after the build, by the package or by
a caller. The **field block** is the field's own, never shared: `physical_dirichlet`, `active_component` and
`constrained_values` are separate arrays for every field, even two over one space. Build a layout with
[`dof_layout`](@ref) for one field, or with `system_layout` for every field of a problem.

The space block:

  - `cell_dofs_by_level::Vector{Array{Vector{Int},D}}` — indexed by level *id*, not by position in `V.levels`, so that
    it matches the `parent.level` id assembly carries; a subdomain whose ids were reindexed into a higher block leaves
    the leading slots undefined, and [`cell_dofs`](@ref) refuses them with an `ArgumentError`. Each entry is a
    `D`-dimensional array of cell-local raw dof id vectors, indexed by the level's `CartesianIndex`. Inactive cells
    carry an empty vector.
  - `raw_keys::Vector{TensorDofKey{D}}` — `raw_keys[i]` is the `TensorDofKey` that produced raw dof `i`. The inverse of
    the construction dictionary.
  - `level_raws::Vector{UnitRange{Int}}` — indexed by level *id*, like `cell_dofs_by_level`: the raws level `id`
    enumerated, as one contiguous range, so `raw_keys[r].level == id` exactly for the `r` in `level_raws[id]`.
    Contiguous by construction: a key carries its level id, so walking one level's cells can only create or reuse that
    level's keys. A dormant level (no active cell) has an empty range, and an id the space does not hold reads `1:0`.
    Every per-level pass — the constraint hooks, physical-Dirichlet detection, the per-level diagnostics — reads its
    level's raws off this range instead of scanning, bucketing or looking up the level of every raw.
  - `elimination_source::Vector{Symbol}` — `raw → :free / :overlay / :coverage / :dedup`. `:free` iff the raw survives;
    otherwise the source that eliminated it: the artificial overlay boundary (`:overlay`) or, from the pruning
    extension, a covered high-order mode (`:coverage`) or a deduped covered vertex (`:dedup`). Applies to every
    component simultaneously. It is the `source` vector of the [`EliminationLedger`](@ref) the layout's artificial
    eliminations were recorded in, so a raw is credited to the source that actually removed it. Serves as both the
    elimination flag (`!== :free`) and its provenance, read by [`constraint_kind`](@ref), the active enumeration, and
    diagnostics.
  - `pivots::Vector{Vector{Tuple{Int,T}}}` — the linear-constraint pivots, and nothing else. A pivot is a raw that
    constraint resolution expressed through *other* raws, `u_raw = Σᵢ wᵢ u_{rᵢ}`, and `pivots[raw]` is that expansion
    `[(r₁, w₁), …]` in resolution order. It has depth one: every `rᵢ` is free with respect to the artificial
    constraints, so a reader never chases a chain, and the topology build checks this once. Like `elimination_source`,
    it comes from the ledger. The table is empty when the layout has no pivot, which is every integrated-Legendre and
    every maximal-continuity B-spline layout: only the multi-raw trace constraints of a B-spline level below maximal
    continuity (on a `LevelMask` face, for instance) produce one. Otherwise it holds one entry per raw, and every raw
    that is not a pivot — free or strongly eliminated alike, which `elimination_source` tells apart — shares one empty
    vector; like the shared empty `cell_dofs` vector, it must never be pushed into. Assembly keeps a pivot's own local
    row and column while it integrates a region and condenses them onto the pivot's active branches once per region
    (`K ← PᵀKP`, `b ← Pᵀb − lift`); [`dof_value`](@ref) reconstructs a pivot from the same expansion. Both read it
    through `_fold_pivot`, which returns at once on an empty table, so a layout without pivots never reads an entry.
    [`has_linear_constraints`](@ref) is `!isempty(pivots)`.
  - `tolerance::GeometryTolerance{T}` — the tolerance the topology was built at and boundary detection ran at. A
    `FacetResolver` the layout's Dirichlet data are projected through must carry the same one.

The field block:

  - `components::Int` — number of scalar channels the field carries (1 for scalar fields, > 1 for vector fields).
    Constraints are tracked per-component so vector boundary conditions can pin individual channels.
  - `physical_dirichlet::Matrix{Bool}` — `(raw, component) → true` iff component `component` of raw dof `raw` has a
    physical Dirichlet constraint.
  - `active_component::Matrix{Int}` — `(raw, component) → active id`. Entry is `0` when component `component` of raw dof
    `raw` is constrained, physically or artificially.
  - `active_count::Int` — total number of active (i.e. enumerated) dofs.
  - `constrained_values::Matrix{T}` — `(raw, component) → value` for constrained dofs. Filled by
    [`_project_dirichlet_values!`](@ref) for nonzero physical Dirichlet data; zero otherwise. The one array of a layout
    that is rewritten after it is built: [`update_dirichlet!`](@ref) re-projects into it in place.

What the L² Dirichlet trace solve concluded — which branch each component's solve took, and which dofs it left
undetermined — is not part of the layout. Both describe the projection rather than the numbering, so
[`diagnostics`](@ref) reports them (`dirichlet_trace_factors`, `unsupported_dirichlet_dofs`) from the projection that
produced them.
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

# The product of the first build step, `_dof_topology`: exactly the space block of `DofLayout`, as one value.
# Physical-Dirichlet detection and numbering both read it before any layout exists, which is why it is a type of its own
# rather than a half-built layout. It is transient — nothing keeps it past the build that made it, which shares it
# between the fields over its space — and the numbering constructor `DofLayout(topology, physical_dirichlet)` is the one
# place its arrays are handed on, so a field added to the space block cannot reach some layouts and miss others.
struct _DofTopology{D,T<:Real}
    cell_dofs_by_level::Vector{Array{Vector{Int},D}}
    raw_keys::Vector{TensorDofKey{D}}
    level_raws::Vector{UnitRange{Int}}
    elimination_source::Vector{Symbol}
    pivots::Vector{Vector{Tuple{Int,T}}}
    tolerance::GeometryTolerance{T}
end

"""
    FieldLayout{D,T}(name, components, dofs, offset, level_ids)

One field's slot inside a multi-field [`SystemLayout`](@ref). `dofs` is the field's [`DofLayout`](@ref): its field block
is the field's own, and its space block is shared with every other field over the same space (see [`DofLayout`](@ref)).
`offset` is the field's starting column in the global active-dof vector (the active dofs of all earlier fields take ids
1 through `offset`, and this field's active dofs take ids `offset + 1` through `offset + active_unknowns(dofs)`).

`level_ids` is the contiguous block of (globally reindexed) level ids the field's [`Space`](@ref) owns. Assembly uses it
to route each integration region to the field that owns it: a region whose parents live on a level outside this range
belongs to another subdomain and this field skips it (`_parent_range`). For a single-domain problem every field's range
covers every level, so every field evaluates on every region.
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

Top-level layout for a problem with one or more fields. `fields` lists the per-field [`FieldLayout`](@ref)s in
declaration order; `by_name` maps field names to their indices in `fields`; `active_unknowns` is the sum of every
field's active count (the global system size); `tolerance` is shared across the fields.

Most consumers go through [`active_cell_dofs`](@ref) and friends, which delegate to the per-field layout transparently.
Single-field problems take the implicit-field shortcut where the field name does not need to be passed.
"""
struct SystemLayout{D,T<:Real}
    fields::Vector{FieldLayout{D,T}}
    by_name::Dict{Symbol,Int}
    active_unknowns::Int
    tolerance::GeometryTolerance{T}
end

# ── The elimination ledger ────────────────────────────────────────────────────

"""
    EliminationLedger{T}

The record a basis family's dof-layer hooks write a space's artificial eliminations into: per raw dof, whether it is
eliminated and by which source; the current level's pending linear-constraint rows; and the pivot table the resolved
rows produce. The topology build (`_dof_topology`) makes one, sized to its space's raw count, runs every level's hooks
into it, and keeps its `source` and `pivots` as the topology's, hence the layout's, `elimination_source` and `pivots`.

Fields:

  - `source::Vector{Symbol}` — per raw, `:free` until something eliminates it, then the source credited with the
    elimination: `:overlay` (the artificial overlay boundary), `:coverage` (a covered high-order mode) or `:dedup` (a
    buried function a covering level reproduces).
  - `rows::Vector{Tuple{Vector{Int},Vector{T},Symbol}}` — the current level's homogeneous linear constraints
    `Σᵢ cᵢ u_{rᵢ} = 0`, as `(raws, coefficients, source)` in emission order. Empty between levels.
  - `pivots::Vector{Vector{Tuple{Int,T}}}` — empty until the first pivot, then one entry per raw: a pivot's depth-one
    expansion, and one shared empty vector for every other raw, which nothing pushes into. See [`DofLayout`](@ref).

A family writes through three verbs and nothing else: `_eliminate!(ledger, raw, source)` for a dof-wise strong
elimination, `_constrain!(ledger, raws, coefficients, source)` for a row, and, only from a `_resolve_rows!` method,
`_pivot!(ledger, raw, expansion, source)` for each raw the rows eliminate.

The order is the core's, and it is what makes the attribution exact:

  - Levels are processed one at a time, in `V.levels` order: the level's `_overlay_constraints!`, then, under leaf
    semantics, its `_coverage_constraints!`, then the family's `_resolve_rows!` on whatever rows the two recorded (the
    core stores rows but resolves none). A row names only its own level's raws, so resolving level by level is the same
    as resolving every row of the space in one pass.
  - Within a level, the direct eliminations from both hooks are applied as they arrive, and the first writer wins: a raw
    is credited to the first source that eliminates it. The artificial boundary is recorded first, so a raw that both it
    and pruning remove is credited to the boundary.
  - The level's rows are resolved *after* both hooks, against everything eliminated so far; an eliminated raw
    contributes no term to a row. A row that collapses to `0 = 0` eliminates nothing, so it is credited with nothing.
  - A family that needs an elimination ordered *among* its rows records it as the one-term row
    `_constrain!(ledger, [raw], [one(T)], source)`. The resolution then decides whether it eliminates anything, and
    which row is credited. The dedup of a B-spline level below maximal continuity is that case: a buried function there
    can also be a term of a trace row, and eliminating it directly would change which raws the trace rows pivot.
"""
struct EliminationLedger{T<:Real}
    source::Vector{Symbol}
    rows::Vector{Tuple{Vector{Int},Vector{T},Symbol}}
    pivots::Vector{Vector{Tuple{Int,T}}}
end

# A ledger over `nraw` raws with nothing eliminated, no row pending and no pivot.
function EliminationLedger{T}(nraw::Integer) where {T}
    return EliminationLedger{T}(fill(:free, nraw), Tuple{Vector{Int},Vector{T},Symbol}[], Vector{Tuple{Int,T}}[])
end

# A dof-wise strong elimination of `raw` by `source`. The first source to eliminate a raw is the one it is credited to,
# which is what makes the artificial boundary, recorded first, win over pruning on a raw both remove.
function _eliminate!(ledger::EliminationLedger, raw::Integer, source::Symbol)
    ledger.source[raw] === :free && (ledger.source[raw] = source)
    return ledger
end

# One homogeneous linear constraint Σᵢ cᵢ u_{rawsᵢ} = 0 among the current level's raws, credited to `source`. It is held
# until both of the level's hooks have run and is then handed to `_resolve_rows!`. The vectors are stored, not copied.
function _constrain!(ledger::EliminationLedger{T}, raws::Vector{Int}, coefficients::Vector{T}, source::Symbol) where {T}
    push!(ledger.rows, (raws, coefficients, source))
    return ledger
end

# The resolver's one write: `raw` is eliminated by the row credited `source`, either strongly (`expansion` empty) or as
# the depth-one pivot u_raw = Σᵢ wᵢ u_{rᵢ}.
#
# The pivot table is allocated on the first pivot, as one shared empty vector per raw, so a space without pivots, which
# is every integrated-Legendre and every maximal-continuity B-spline space, never allocates it. A raw already eliminated
# is refused: a resolver that pivoted it again would silently overwrite its source and, for a pivot, the expansion other
# readers already rely on.
function _pivot!(ledger::EliminationLedger{T}, raw::Integer, expansion::Vector{Tuple{Int,T}}, source::Symbol) where {T}
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

Resolve the linear-constraint rows `level`'s hooks recorded with `_constrain!`. Called once per level, after
`_overlay_constraints!` and `_coverage_constraints!`, and only if a row is pending. A method receives `ledger.rows` in
emission order, every row naming only `level`'s raws; a raw with `ledger.source[raw] !== :free` is already eliminated
and contributes no term. It must record every raw the rows eliminate exactly once, with `_pivot!` — an empty expansion
for a strong elimination, else a depth-one expansion over raws that remain free — credited to the row that eliminated
it. It must not otherwise write `ledger.source`, and it must not clear `ledger.rows`: the caller does.

The core resolves nothing itself: a family that records rows implements this, and this default raises. The
integer-continuity B-spline family in `ext/UnfittedBasicBSplineExt.jl` is the only one today; its method is the cascade
Gauss elimination `_cascade!`, which lives there with it. The default is written over `Level{D,T}` with both parameters
free, as a family method naturally is (`Level{D,T,<:F} where {D,T}`): over a bare `Level`, whose `T` is bounded by
`Real`, such a method would be ambiguous with it rather than more specific.
"""
function _resolve_rows!(ledger::EliminationLedger, level::Level{D,T}) where {D,T}
    throw(ArgumentError("the $(basis_name(level.basis)) family recorded $(length(ledger.rows)) " *
                        "linear-constraint row(s) on level $(level.id) but implements no " *
                        "`_resolve_rows!` method to resolve them"))
end

# Every pivot must be expressed over raws that stay free: `_fold_pivot`, `dof_value` and assembly's condensation read
# ONE level of expansion and never chase a chain. A family resolver that broke this would make assembly and
# reconstruction silently disagree, so it is checked once per build: one pass over the pivot table, a read per raw and
# one per pivot term, and nothing at all without pivots. No shipped resolver breaks the rule, so no build reaches the
# raise; the guard is there for a family resolver yet to be written, and test_constraints.jl pins its verdict on a
# hand-built ledger.
function _assert_resolved(ledger::EliminationLedger)
    for expansion in ledger.pivots, (branch, _) in expansion
        ledger.source[branch] === :free ||
            throw(ArgumentError("pivot branch $branch is itself eliminated: `_resolve_rows!` " *
                                "must leave every pivot expressed over free raws"))
    end
    return nothing
end

# Everything a dof-layer hook may read about the level it is called for: the space, the build's material oracle (`_Fold`
# in `coverage.jl`, which answers whether a cell holds material), the tolerance, the coverage masks (empty when the
# build does not prune), the raw keys and the key → raw map the raw enumeration built (for a family that addresses
# functions by `(cell, mode)`), and the level's own raws. One oracle serves every level of the build, so a cell is
# classified at most once per build whichever hook asks.
#
# `F` is the CONCRETE type of the oracle, whose own parameter is the run-time type of `V.physical`. `Space.physical` is
# abstractly typed, so a hook that read it off the space dispatched dynamically on every per-raw predicate call it fed
# it to. The oracle is built from the run-time value of `V.physical`, which gives it, and this struct, a concrete type
# wherever they are built, and the hook the context is handed to is the function barrier: the hooks, and every predicate
# they call, specialise on it instead. Measured on the integrated-Legendre artificial-boundary pass over an immersed
# disc at 128² p2 (29 281 raws): 304 bytes per raw and 1.26 ms with the physical domain typed `Any`, nothing and 0.24 ms
# typed.
struct _LevelContext{D,T,S<:Space{D,T},F<:_Fold{D,T}}
    space::S
    fold::F
    tolerance::GeometryTolerance{T}
    coverage::Coverage{D}
    raw_keys::Vector{TensorDofKey{D}}
    raw_by_key::Dict{TensorDofKey{D},Int}
    raws::UnitRange{Int}
end

# ── Topology (step 1) and numbering (step 3) ──────────────────────────────────

# Step 1 of a layout: the space block, built from the space alone. Raw enumeration (`_enumerate_raws`), then every
# level's artificial eliminations recorded in one `EliminationLedger`, level by level in `V.levels` order: the family's
# `_overlay_constraints!`, then, under leaf semantics, its `_coverage_constraints!` against the masks `build_coverage`
# computes (so those are built first, and not at all when `prune = false`), then `_resolve_rows!` on whatever
# linear-constraint rows the two recorded — and finally the depth-one check on the pivots the rows resolved to. The
# ledger's per-raw sources and pivot table become the topology's. The core owns this order and the pruning gate
# (`prune`, `prune_exempt`), so no family re-implements either: a family that forgot the gate would prune the unreduced
# twin.
#
# The material oracle (`_Fold`) is built here, once, over `classify_cache` — the cache `prepare` filled during the fold,
# or a fresh one for a standalone call — and every consumer of the build reads it: `build_coverage` and every level's
# context. It lives for this call only.
#
# Each hook call is the function barrier its per-raw loop runs behind. The oracle is built from the run-time value of
# `V.physical`, so its type, and with it the level's context, is concrete (see `_LevelContext`) even though the field is
# not, and `V.levels` may mix basis families; a hook therefore runs specialised on both, and what dispatches at run time
# is at most the context's construction and the three calls per level.
#
# A topology reads neither a field's components nor its Dirichlet data, so one is the same for every field over the
# space, and `_system_layout` builds it once per space and shares it between them.
function _dof_topology(V::Space{D,T}; tolerance::GeometryTolerance{T}=GeometryTolerance(T), prune::Bool=true,
                       prune_exempt=(), classify_cache::_ClassifyCache{D,T}=_ClassifyCache{D,T}()) where {D,T}
    raw_by_key, raw_keys, cell_dofs_by_level, level_raws = _enumerate_raws(V)
    fold = _Fold(V, classify_cache)
    coverage = prune ? build_coverage(V, tolerance, fold) : Coverage{D}(Dict{Int,BitArray{D}}())
    ledger = EliminationLedger{T}(length(raw_keys))
    for level in V.levels
        ctx = _LevelContext(V, fold, tolerance, coverage, raw_keys, raw_by_key, level_raws[level.id])
        _overlay_constraints!(ledger, level, ctx)
        prune && !(level.id in prune_exempt) && _coverage_constraints!(ledger, level, ctx)
        isempty(ledger.rows) && continue
        _resolve_rows!(ledger, level)
        empty!(ledger.rows)
    end
    _assert_resolved(ledger)
    return _DofTopology{D,T}(cell_dofs_by_level, raw_keys, level_raws, ledger.source, ledger.pivots, tolerance)
end

# The raw enumeration, the first half of step 1 (`_dof_topology`): walk every level's cells and assign raw dofs through
# the `TensorDofKey` cache (cross-cell endpoint sharing happens here). Levels go in `V.levels` order, cells in
# `cell_indices` order, local functions in `cell_basis_indices` order, and a raw's id is the order of its key's first
# appearance. Returns the key → raw map, `raw_keys`, the per-level cell tables and the per-level raw ranges.
#
# The ranges cost one `UnitRange` per level, and every per-level pass reads its level's raws off one instead of scanning
# or bucketing every raw. They are exact because a key carries its level id: walking level L can only create or reuse
# L's keys, so L's raws are precisely the ids appended while L is walked, and visiting them is visiting them in raw
# order.
function _enumerate_raws(V::Space{D}) where {D}
    raw_by_key = Dict{TensorDofKey{D},Int}()
    raw_keys = TensorDofKey{D}[]
    # Indexed by level *id*, not by position: for a single-domain space the two coincide (base = 1, overlays 2, 3, …),
    # but a subdomain of a coupled model is reindexed into a disjoint id block by `_reindex_space_levels`, so this
    # layout's ids run over that block and match the `parent.level` id the assembly workspace and `cell_dofs` key by
    # (leading slots below the block stay `undef` and are never addressed by this field's parents).
    nid = maximum(l -> l.id, V.levels)
    cell_dofs_by_level = Vector{Array{Vector{Int},D}}(undef, nid)
    level_raws = fill(1:0, nid)

    # Every inactive cell of every level shares one empty vector. A ladder makes inactive cells the overwhelming
    # majority of the grid — measured on a depth-5 2D stack over a 6² base with one base cell refined to the bottom, 49
    # 140 cells of which 1 400 are active — and a fresh `Int[]` per inactive cell put one heap allocation on each of
    # them, on every rebuild, and `estimate` rebuilds. On that layout it is 74 856 allocations against 27 117. It is per
    # call rather than a module-level constant because a `const` empty vector is one stray `push!` away from corrupting
    # every inactive cell of every layout ever built.
    none = Int[]
    for level in V.levels
        first_raw = length(raw_keys) + 1
        level_cells = Array{Vector{Int},D}(undef, level.mesh.cells)
        for cell in cell_indices(level.mesh)
            level_cells[cell] = is_active(level.mask, cell) ? _build_cell_dofs!(raw_by_key, raw_keys, level, cell) :
                                none
        end
        cell_dofs_by_level[level.id] = level_cells
        level_raws[level.id] = first_raw:length(raw_keys)
    end
    return raw_by_key, raw_keys, cell_dofs_by_level, level_raws
end

# Build the per-cell raw dof id vector for one cell. For every local basis function on the cell, compute its
# `TensorDofKey` and either reuse the raw id already registered for that key (the cross-cell dof reuse path: adjacent
# cells of one level share every function their keys name alike, such as an endpoint node, while a key carries its level
# id and so is never shared across levels) or register a fresh id by appending the key to `raw_keys`.
function _build_cell_dofs!(raw_by_key::Dict{TensorDofKey{D},Int}, raw_keys::Vector{TensorDofKey{D}}, level::Level{D},
                           cell::CartesianIndex{D}) where {D}
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

# Step 3 of a layout: number one field over a topology, given its physical-Dirichlet flags (step 2,
# `_physical_dirichlet`). Active ids run component-major — component 1 of every raw, then component 2, … — skipping
# every (raw, component) a physical Dirichlet condition constrains or an artificial elimination removed, so an
# all-or-nothing constraint leaves the natural block ordering and a per-component one leaves holes. The constrained
# values start at zero, which is already the answer for homogeneous data; nonzero data are projected into them
# afterwards (step 4, `_project_dirichlet_values!`).
#
# The topology's arrays are handed on, not copied: the space block is never mutated once built, and this is the one
# place a layout receives it.
function DofLayout(t::_DofTopology{D,T}, physical_dirichlet::Matrix{Bool}) where {D,T}
    nraw, ncomp = size(physical_dirichlet)
    active_component = zeros(Int, nraw, ncomp)
    active_count = 0
    for c in 1:ncomp, raw in 1:nraw
        (physical_dirichlet[raw, c] || t.elimination_source[raw] !== :free) && continue
        active_component[raw, c] = (active_count += 1)
    end
    return DofLayout{D,T}(t.cell_dofs_by_level, t.raw_keys, t.level_raws, t.elimination_source, t.pivots, t.tolerance,
                          ncomp, physical_dirichlet, active_component, active_count, zeros(T, nraw, ncomp))
end

# ── Geometric helpers shared with `dirichlet.jl` ──────────────────────────────

# Tolerance-aware coordinate equality used throughout boundary detection. Pulled out so the `<= tol.contain` convention
# is in one place; consumed both by the overlay-constraint detection below and by the physical-Dirichlet detection in
# `dirichlet.jl`.
_coordinate_matches(a::Real, b::Real, tol::GeometryTolerance) = abs(a - b) <= tol.contain

# True iff the level's mesh-domain boundary on the given axis-side coincides with the physical-domain boundary up to
# `tol.contain`. Used by the artificial-boundary hooks to decide whether a mesh-boundary face is a physical face (no
# overlay constraint) or part of the artificial overlay boundary, whose positions the box-face rule `_clamps_box_face`
# then decides.
function _level_side_is_physical(level::Level{D,T}, domain::AxisBox{D,T}, axis::Integer, side::Symbol,
                                 tol::GeometryTolerance{T}) where {D,T}
    side === :lower && return _coordinate_matches(level.mesh.domain.lower[axis], domain.lower[axis], tol)
    side === :upper && return _coordinate_matches(level.mesh.domain.upper[axis], domain.upper[axis], tol)
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# True iff `level` is clamped to zero on the `side` face of its own box where boundary cell `cell` meets it — the
# box-face rule, written once for every family that clamps (`_has_overlay_constraint`, and `_active_cell_at_face` in the
# B-spline extension) and for the burial test that has to predict the clamp (`_reproduced_on_domain`). Asked only of a
# side that is not physical (`_level_side_is_physical`); there the face is a candidate for the artificial boundary
# Γ_o = ∂Ω^(k) \ ∂Ω, and it belongs to it where the cell is active and its face holds material (`_is_void_face`).
#
# The material half reads Ω^(k) as the overlay's box intersected with Ω, so that Γ_o is the part of the box face that
# meets Ω. A face lying wholly outside Ω has nothing on either side of it to keep continuous: the level's functions jump
# to zero there, and nothing is integrated across the jump. Clamping such a face deleted the functions a cut boundary
# cell needs near ∂Ω, and it left exact null modes wherever the coarse cell behind the face was covered on its material
# part, because the clamped level still reproduced on Ω every coarse function vanishing on the face. It is the box-face
# counterpart of the internal-face exemption, which asks the inactive neighbour (`_is_fictitious`); beyond a box face
# there is no cell of the level to ask, so the face itself is asked.
#
# A face on which ∂Ω lies exactly counts as holding material, because the classifier's verdicts are taken on closed
# sets, so it stays clamped. Measured on Ω = {x₁ ≤ ω} under a cover ending at x₁ = ω: 7 and 20 exact null modes remain
# at p = 2 and 3 when ω lands on the face, and none with ω moved 1e-3 off it. A boundary that is not built from planes
# on the grid essentially never lies on a face.
@inline function _clamps_box_face(fold::_Fold{D,T}, level::Level{D,T}, cell::CartesianIndex{D}, axis::Int,
                                  side::Symbol) where {D,T}
    return is_active(level.mask, cell) && !_is_void_face(fold, level, cell, axis, side)
end

# ── The artificial-boundary hook ──────────────────────────────────────────────

"""
    _overlay_constraints!(ledger::EliminationLedger, level::Level, ctx::_LevelContext) -> ledger

Record `level`'s artificial-overlay-boundary condition in `ledger`. Dispatched on the level's basis family, so each
family can exploit its own boundary mode structure. Called once per level, before `_coverage_constraints!` (see
[`EliminationLedger`](@ref) for the order). A method names only raws in `ctx.raws`, reads only `ctx` (see
`_LevelContext`), and records through `_eliminate!` and `_constrain!`. Every method ties `{D,T}` across `Level{D,T}` and
`_LevelContext{D,T}`, as this default does, so that no two methods are ambiguous.

This default walks the level's raws, asks the family's per-raw `_has_overlay_constraint` predicate about each, and
strongly eliminates every raw it accepts with `_eliminate!(ledger, raw, :overlay)`: the regime the integrated Legendre
family relies on. The predicate consults `ctx.fold`, the build's material oracle, only to ask whether the inactive cell
across an internal face holds material; nothing is classified when the space has no physical domain.

The B-spline family in the extension overrides this default. At `continuity = :maximal` its method is a support
selection, which again eliminates strongly. At an integer continuity `m` it records multi-raw rows instead: the `Cᵐ`
trace-vanishing condition along every artificial perpendicular line, `m + 1` rows per line per face, factored through
the tensor-product structure of the basis and looking raw ids up by `(cell, mode)` in `ctx.raw_by_key`. That family
therefore never reaches this default or the per-raw `_has_overlay_constraint`.
"""
function _overlay_constraints!(ledger::EliminationLedger, level::Level{D,T}, ctx::_LevelContext{D,T}) where {D,T}
    for raw in ctx.raws
        _has_overlay_constraint(ctx.raw_keys[raw], level, ctx) && _eliminate!(ledger, raw, :overlay)
    end
    return ledger
end

# ── The pruning hook: one burial driver ───────────────────────────────────────

"""
    _coverage_constraints!(ledger::EliminationLedger, level::Level, ctx::_LevelContext) -> ledger

Pruning constraint source, the peer of [`_overlay_constraints!`](@ref) and bound by the same contract. Called only under
leaf semantics (`dof_layout`'s `prune`, on by default) and only on a level outside `prune_exempt`, after the level's
`_overlay_constraints!`. It is one driver for every family — the *burial test* — and a family takes part through the
burial hooks below rather than through a method of its own. It strongly eliminates

  * every **buried high-order** mode of a family that sheds them (integrated Legendre: at least one bubble axis, every
    active incident cell covered on its material part by, and every fictitious one reached by, a single finer level) —
    covered-mode pruning, source `:coverage`; and
  * every **buried** function a single nested level above whose span contains it reproduces exactly — dedup, source
    `:dedup`. The cover's mesh must nest over this one (`_nested_over`) and its span must contain the function
    (`_reproduces`): for integrated Legendre the linear modes, under a cover whose basis spans that mesh's hats; for a
    B-spline any buried function, under a cover that carries its degree and smoothness.

The linear skeleton is otherwise retained, which is what makes the reduced space complete. Each elimination is credited
with its source for `constraint_kind` / diagnostics, unless the artificial boundary already eliminated the raw: the
first writer wins (see [`EliminationLedger`](@ref)), so a buried raw on `Γ_o` stays credited to `:overlay`.

Both halves ask `_reproduced_on_domain` (`coverage.jl`) of a cover judged geometrically, which applies the material rule
on the covering level *and* on this one — an inactive support cell that holds no material (a folded cell, or a user mask
entirely outside Ω) carries nothing to reproduce, so it neither has to be covered nor may veto the verdict, and an
active one has to be covered only on its share of Ω. `ctx.fold` is the material oracle that answers it, the same one
[`build_coverage`](@ref) used, and its memo holds `build_coverage`'s answers to the per-cover coverage question that
test asks again. A B-spline level under a cover that selects by support is judged exactly instead, by subdivision
against the cover's own admissible region (see `bspline` in the extension).

The dedup half is not an optimisation: a mode a covering level reproduces exactly is linearly dependent on that level's
own modes, so leaving both active makes the superposed operator exactly singular. The B-spline family takes part for
that reason and records nothing but dedups — it has no bubble modes to shed. A family that does not opt in — that sheds
nothing (`_sheds_any`) and has no cover that reproduces its functions (`_reproduces`) — records nothing.

The driver runs in three steps, all behind a dispatch barrier (`_bury!`):

  1. The **candidates**, in raw order: the level's functions whose support passes the prefilter of the test that will
     judge them — every active cell of it covered by some level above, for the geometric test; every inactive cell of it
     void, for the exact one — each marked as a mode to shed or a function to dedup (`_sheds`).
  2. One **pass per cover**, in `V.levels` order, over the candidates no earlier cover decided. A shed mode is decided
     by any cover that reproduces it geometrically, a dedup candidate only by a cover that may dedup it (nested, and
     spanning it), with that cover's exact test when the family has one (`_cover_test`). A candidate thus meets the
     covers in order up to the first that decides it, as a per-function loop over the covers would ask them.
  3. The **record**, in raw order: `:coverage` for a shed mode, the family's `_dedup!` for a deduped function — a direct
     elimination, or the one-term row a clamped B-spline level resolves among its trace rows.

The family hooks, each defaulting to "takes no part":

  - `_sheds_any(level)` / `_sheds(level, key)` — whether the level has, and the key names, a high-order mode to shed
    rather than a function to dedup;
  - `_burial_cells(level, key)` — the cells the function named by `key` lives on;
  - `_cover_test(level, k, ctx)` — the exact reproduction record of cover `k` for this level's functions, read by
    `_reproduced_by`, or `nothing` to judge `k` geometrically;
  - `_dedup!(ledger, level, raw)` — how a dedup is recorded;
  - `_reproduces(k, level)` — whether `k`'s span contains this level's buried functions on a mesh `k` nests over.

Every hook method over a `Level` carries `{D,T<:Real}` in its signature, as the defaults do, so a family method is
always more specific than the default rather than ambiguous with it.
"""
function _coverage_constraints!(ledger::EliminationLedger, level::Level{D,T}, ctx::_LevelContext{D,T}) where {D,T}
    # A level none of whose cells is covered has nothing to bury, under either test. A function a cover reproduces has
    # every active support cell covered on its share of Ω: the geometric test asks exactly that (condition 1 of
    # `_reproduced_on_domain`), and the exact test of a selecting B-spline cover asks it through the cover's virtual
    # cells instead, wanting each support cell's part past the cover's box void. That made the gate sound only once the
    # flags looked at the material part. While they tested whole cell boxes they skipped the exact test wherever no cell
    # of the level lay inside a cover's box: a degree-2 B-spline base of 3² over Ω = {x₁ ≤ 0.91} under a cover ending at
    # x₁ = 0.95 kept the 5 functions the cover reproduces as exact null modes.
    any(ctx.coverage.covered[level.id]) || return ledger
    covers = Level[]
    for k in ctx.space.levels
        k.id > level.id && push!(covers, k)
    end
    # `inferencebarrier` hides the callee from inference: the call is resolved at run time, on the level's concrete
    # type, so the burial test is inferred and compiled when a level first reaches it and not before. A level whose
    # coverage flags are all false — every level of a single-level space — never does, so a build in which nothing is
    # covered compiles none of it. Hiding the level's type instead is not enough: `_bury!` is the only method that can
    # match, so inference still descends into it with the level widened to an abstract `Level` and infers every family's
    # hooks there — measured on the first `prepare` of an immersed single-level plate, 146 ms of inference in a `_bury!`
    # that never runs. `Base.inferencebarrier` is a Base internal, not declared public, and its effect was measured on
    # Julia 1.12: a Julia that changed it would show here as more compile work, never as another answer, since the call
    # it wraps is the same either way.
    Base.inferencebarrier(_bury!)(ledger, level, ctx, covers)
    return ledger
end

# The burial test of one level (`_coverage_constraints!`), entered through the dispatch barrier and so specialised on
# the level's concrete type. `covers` holds the levels above, abstractly typed: each cover's pass is one run-time
# dispatch, on the cover and its record, and the per-candidate loop inside it runs specialised on them, whatever mix of
# families the space has. The records are handed to the pass as an argument and never kept in a vector: one read back
# out of a `Vector{Union{Nothing, _Admissibility}}` was re-boxed for every call, 272 bytes per candidate measured on a
# 48² degree-3 B-spline level under a nested maximal-continuity cover.
#
# `dedups[i]` says whether `covers[i]` may dedup: its span contains this level's buried functions (`_reproduces`) and
# its mesh nests over this one (`_nested_over`). Only those covers may dedup; any cover may shed. A level that sheds
# nothing and has no cover to dedup it is done before a candidate is looked at.
#
# `geometric` is whether every cover that may dedup is judged geometrically, which picks the candidates' prefilter
# (`_BurialMarks`): the coverage flags, a fast path for the geometric test only, or else the level side the exact test
# needs. It is decided before the passes, so a family whose `_cover_test` builds a record builds the first one here only
# to learn that it is not `nothing`, and that cover's pass builds it again; that record is O(1) to build.
#
# Step 3, the record, follows the passes: every decided candidate, in raw order. The geometric and exact tests read the
# mask and the material oracle with its memo but never the ledger, so recording after the passes is recording as each
# candidate is decided; raw order is the order a clamped B-spline level's dedup rows must keep (see `_dedup!` in the
# extension).
function _bury!(ledger::EliminationLedger, level::Level{D,T}, ctx::_LevelContext{D,T},
                covers::Vector{Level}) where {D,T<:Real}
    dedups = falses(length(covers))
    geometric = true
    for (i, k) in pairs(covers)
        dedups[i] = _reproduces(k, level) && _nested_over(level, k, ctx.tolerance)
        geometric && dedups[i] && (geometric = _cover_test(level, k, ctx) === nothing)
    end
    _sheds_any(level) || any(dedups) || return ledger
    marks = _BurialMarks(level, ctx, geometric)
    for (k, dedup) in zip(covers, dedups)
        # A pass that can decide nothing is skipped before the cover's record is built.
        _decides(marks, dedup) || continue
        _mark_buried!(marks, level, k, dedup ? _cover_test(level, k, ctx) : nothing, dedup, ctx)
    end
    for (i, raw) in enumerate(ctx.raws)
        marks.done[i] || continue
        marks.sheds[i] ? _eliminate!(ledger, raw, :coverage) : _dedup!(ledger, level, raw)
    end
    return ledger
end

# The burial test's working set, one bit per raw of the level, in raw order: whether the raw is a candidate, whether it
# is a mode to shed (`:coverage`) rather than a function to dedup, and whether a cover has decided it yet. Bits over the
# level's raw range rather than a list of the candidates, so the set costs the same few allocations whatever the number
# of candidates.
struct _BurialMarks
    candidate::BitVector
    sheds::BitVector
    done::BitVector
end

# The candidates of `level` (step 1 of `_coverage_constraints!`): every raw whose support passes the prefilter of the
# test that will judge it. Each half of "buried as a whole function" belongs to one of the two tests.
#
# The coverage half — every active support cell covered by some level above — is the prefilter of the geometric test,
# used when every cover that may dedup is judged geometrically (`geometric`; always, for an integrated-Legendre level):
# a cover that reproduces the function covers each active support cell, so the flags turn away only functions no
# geometric cover could decide. It is not asked when a selecting cover may dedup. Its exact test asks the same material
# question — does the part of a support cell past the cover's box hold anything? — through other boxes, the cover's
# virtual cells rather than the slabs `_material_inside` classifies, so on a feature finer than the classifier resolves
# the two answers can differ, and a prefilter could then only cost a dedup. Nor does the exact test need one for speed:
# its block test stops at the first inadmissible cell, which for a function nowhere near the cover is one of the first
# it asks — what keeps the walk off the level's bulk. The flags are read on the active support cells only, the ones
# `_coverage_cells` computed: a support cell two cells past the active front holds the `false` default, not a verdict.
#
# The level half — every inactive support cell void — is the exact test's. An inactive support cell that holds material
# (a user mask reaching into Ω) blocks: what a masked level assembles is the function truncated to its own active cells,
# which is not the function a cover reproduces. One that holds none (a folded cell, or a user mask entirely outside Ω)
# does not: nothing is integrated there, so the truncation is invisible on Ω. The geometric test asks this itself
# (`_reproduced_on_domain`, condition 2, the same for every cover), but the exact test of a selecting B-spline cover
# looks only at the cover, and without the level half it deleted real approximation rather than a null mode: measured on
# a `continuity = 0` base of 4² masked at one interior cell under a nested maximal-continuity 6² cover, the pruned space
# carried 58 unknowns against an unreduced rank of 59. Where both tests are in play the level half is asked, and a cover
# judged geometrically still applies its own condition 2.
function _BurialMarks(level::Level{D,T}, ctx::_LevelContext{D,T}, geometric::Bool) where {D,T<:Real}
    cov, n = ctx.coverage.covered[level.id], length(ctx.raws)
    candidate, sheds = falses(n), falses(n)
    for (i, raw) in enumerate(ctx.raws)
        key = ctx.raw_keys[raw]
        cells = _burial_cells(level, key)
        buried = geometric ? all(ci -> !is_active(level.mask, ci) || cov[ci], cells) :
                 all(ci -> is_active(level.mask, ci) || _is_void(ctx.fold, level, ci), cells)
        buried || continue
        candidate[i] = true
        sheds[i] = _sheds(level, key)
    end
    return _BurialMarks(candidate, sheds, falses(n))
end

# Whether a pass over a cover can decide anything: an undecided shed mode, or, when the cover may dedup, any undecided
# candidate. Its scan of the bits costs what an empty pass does; what skipping buys is the pass's run-time dispatch with
# its boxed arguments and, for a selecting B-spline cover, the record: measured single-threaded, running every pass
# added 24 to 55 allocations (10 to 34 KB) to one layout of the ladders and band stacks over an immersed disc and of a
# depth-5 ladder, with no change in time.
function _decides(marks::_BurialMarks, dedup::Bool)
    any(i -> marks.candidate[i] && !marks.done[i] && (dedup || marks.sheds[i]), eachindex(marks.done))
end

# One cover's pass (step 2 of `_coverage_constraints!`): mark every undecided candidate the cover `k` reproduces. `test`
# is `k`'s exact record from `_cover_test`, or `nothing` to judge geometrically (`_reproduced_on_domain`); `dedup`
# whether `k` may dedup at all.
#
# A shed mode is judged geometrically by every cover, and the question is the same *single-level* one the dedup asks. A
# mode whose incident cells are covered by two different levels is reproduced by neither: each of them is clamped to
# zero on the face they share, so at that seam nothing carries what the elimination removes — the coverage flags say
# only that *some* level covers each cell, which is why they are a prefilter and not the verdict.
#
# The shed deliberately does NOT ask the cover to carry this level's order (nor to nest, nor to span). Shedding a buried
# high-order mode is not a claim that something reproduces it — that is the dedup's job, for the functions a cover does
# reproduce. It is what leaf semantics mean: over a region a finer level resolves, a coarse cell's high-order modes buy
# almost nothing in L² and carry the oscillation, so a high-order base with a low-order fine overlay over a non-smooth
# feature is exactly the configuration the rule exists to serve. Gating on the cover's order turns leaf semantics off
# exactly there: measured on a tanh layer with base p = 5 and a p = 2 overlay, the gate cut the base's shed modes from
# 189 to 5 and left far-field oscillation and overshoot bit-identical to switching the rule off (9.37e-3 both), against
# 9.95e-4 and 7.03e-4 without it.
function _mark_buried!(marks::_BurialMarks, level::Level{D,T}, k::Level{D,T}, test, dedup::Bool,
                       ctx::_LevelContext{D,T}) where {D,T<:Real}
    for (i, raw) in enumerate(ctx.raws)
        marks.candidate[i] && !marks.done[i] || continue
        shed = marks.sheds[i]
        shed || dedup || continue
        key = ctx.raw_keys[raw]
        marks.done[i] = if shed || test === nothing
            _reproduced_on_domain(level, k, _burial_cells(level, key), ctx.tolerance, ctx.fold)
        else
            _reproduced_by(test, level, key)
        end
    end
    return marks
end

# The burial hooks' defaults: a family takes no part until it says so. The integrated-Legendre methods are under the
# "Integrated Legendre family" divider below, the B-spline ones in the extension. `_burial_cells` and `_reproduced_by`
# have no default: a family that answers `_sheds_any` or `_reproduces` (or `_cover_test`) with something other than the
# default supplies them.
_sheds_any(::Level{D,T}) where {D,T<:Real} = false
_sheds(::Level{D,T}, ::TensorDofKey{D}) where {D,T<:Real} = false
function _burial_cells end
_cover_test(::Level{D,T}, ::Level{D,T}, ::_LevelContext{D,T}) where {D,T<:Real} = nothing
function _reproduced_by end
_dedup!(ledger::EliminationLedger, ::Level{D,T}, raw::Int) where {D,T<:Real} = _eliminate!(ledger, raw, :dedup)

# Whether the cover `k`'s span contains `level`'s buried functions, given that `k`'s mesh nests over `level`'s
# (`_nested_over`, the geometric half, which the driver asks separately). Dispatched on both families, so every pair has
# exactly one answer:
#
#   level \ k             integrated Legendre        B-spline
#   integrated Legendre   true                       every axis of degree 1
#   B-spline, degree p    `:tensor`, every palette    the same degree p on
#                         entry ≥ p componentwise     every axis
#
# A buried integrated-Legendre function that a cover may dedup is a C⁰ hat: multilinear on each of `level`'s cells,
# kinked across every cell boundary. Nesting buys the multilinear half; the kink is the cover basis's half, and neither
# half is about `level`'s family — a cover that spans the hats dedups them whoever generated them. The B-spline rows are
# the extension's.
#
# The default is `false`, which is the safe direction: skipping a legitimate dedup leaves a duplicate, while deduping
# against a span that does not contain the function deletes part of the space. It is a trait of the pair rather than a
# type test on the cover because smoothness is a property of a family's own construction: a maximal-regularity B-spline
# is C^(p−1) at a simple interior knot, so it carries the hat's kink at degree 1 and at no higher degree, which a
# `k.basis isa IntegratedLegendre` gate cannot say.
_reproduces(::Level{D,T}, ::Level{D,T}) where {D,T<:Real} = false

# ── DofLayout accessors ───────────────────────────────────────────────────────

"""
    raw_dof_count(layout::DofLayout) -> Int

Total number of *raw* dof slots (including constrained), counted as `length(raw_keys) * components`.
"""
raw_dof_count(layout::DofLayout) = length(layout.raw_keys) * layout.components

"""
    active_unknowns(layout::DofLayout) -> Int

Number of *active* (post-constraint) dofs — the size of the global system the solver sees.
"""
active_unknowns(layout::DofLayout) = layout.active_count

"""
    cell_dofs(layout, level::Integer, cell::CartesianIndex) -> Vector{Int}

Pre-constraint raw dof ids for the cell at `cell` on the level with id `level`, in the canonical local-basis order.
Inactive cells return an empty vector. `layout` is a [`DofLayout`](@ref), or the `SystemLayout` of a single-field
problem.

`level` is a level *id*. For a single-domain space that is the level's position in `V.levels`, but each subdomain of a
coupled problem has its ids reindexed into a block of its own, and its layout holds no cells for an id outside that
block. Such an id, or one no level carries, raises an `ArgumentError`.

The vector is the layout's own and must not be mutated: every inactive cell of every level shares one empty vector, so a
`push!` into a returned empty result would reach all of them. Copy before editing.
"""
function cell_dofs(layout::DofLayout{D}, level::Integer, cell::CartesianIndex{D}) where {D}
    # One `isassigned` covers both an id past the end of the table and a slot below a reindexed subdomain block, which
    # would otherwise surface as an `UndefRefError` from inside the table. It replaces the range check rather than
    # adding to it, so assembly's `_frame!`, which calls this once per (region, parent, field), pays one test as before,
    # and the read after it is in bounds.
    tables = layout.cell_dofs_by_level
    isassigned(tables, level) || throw(ArgumentError("level id $level is not a level of this layout's space"))
    return @inbounds(tables[level])[cell]
end

"""
    active_cell_dofs(layout::DofLayout, level::Integer, cell::CartesianIndex, component=1) -> Vector{Int}
    active_cell_dofs(layout::SystemLayout, level::Integer, cell::CartesianIndex, component=1) -> Vector{Int}
    active_cell_dofs(layout::SystemLayout, field::Symbol, level::Integer, cell::CartesianIndex,
                     component=1) -> Vector{Int}

Active ids for component `component` of the cell-local raw dofs, with `0` standing for any dof that was eliminated by a
constraint. The order matches the canonical local-basis ordering (same length as `cell_dofs(layout, level, cell)`), and
`level` is a level id, as there.

On a `DofLayout` the ids are the field's own, in `1:active_unknowns(layout)`. On a [`SystemLayout`](@ref) they are
global: shifted by the field's `offset` into the system's active-dof vector, so they index a [`Solution`](@ref)'s
coefficients directly. The second form reads the only field of a single-field problem; the third names the field.
"""
function active_cell_dofs(layout::DofLayout{D}, level::Integer, cell::CartesianIndex{D}, component::Integer=1) where {D}
    return [layout.active_component[raw, component] for raw in cell_dofs(layout, level, cell)]
end

"""
    constraint_kind(layout, raw, component=1) -> Symbol

Classify component `component` of a single raw dof. `layout` is a [`DofLayout`](@ref), or the `SystemLayout` of a
single-field problem. Returns one of:

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

Stored value of a constrained (raw, component). Zero for every artificial elimination (overlay boundary, pruning, and a
`:mixed` dof, see [`constraint_kind`](@ref)) and for homogeneous Dirichlet conditions; the projected value for nonzero
Dirichlet data. `layout` is a [`DofLayout`](@ref), or the `SystemLayout` of a single-field problem.
"""
constrained_value(layout::DofLayout, raw::Integer, component::Integer=1) = layout.constrained_values[raw, component]

"""
    has_linear_constraints(layout::DofLayout) -> Bool

Whether the layout carries a linear-constraint pivot: a raw that constraint resolution expressed through other raws (see
[`DofLayout`](@ref)'s `pivots`), which assembly condenses and [`dof_value`](@ref) reconstructs. `false` for every
integrated-Legendre and every maximal-continuity B-spline layout, whose eliminations are all strong; assembly reads it
per field and skips the pivot search when it is `false`.
"""
has_linear_constraints(d::DofLayout) = !isempty(d.pivots)

# ── SystemLayout accessors (multi-field problems) ─────────────────────────────

"""
    active_unknowns(layout::SystemLayout) -> Int

Total number of active dofs across every field — the size of the global system the solver sees.
"""
active_unknowns(layout::SystemLayout) = layout.active_unknowns

"""
    raw_dof_count(layout::SystemLayout) -> Int

Total number of raw dof slots across every field (sum of `raw_dof_count(field.dofs)`).
"""
raw_dof_count(layout::SystemLayout) = sum(raw_dof_count(field.dofs) for field in layout.fields)

# True iff any field's dof layout carries a linear-constraint pivot (a raw redistributed onto other raws; see
# [`DofLayout`](@ref)'s `pivots`). Assembly itself asks per field; this system-wide form serves tests and diagnostics.
has_linear_constraints(layout::SystemLayout) = any(field -> has_linear_constraints(field.dofs), layout.fields)

# Resolve the unique field of a single-field layout. Used by the single-field system forms below, so a single-field
# problem need not name its field.
function _only_field_layout(layout::SystemLayout)
    length(layout.fields) == 1 || throw(ArgumentError("field argument is required for multi-field layouts"))
    return only(layout.fields)
end

# The system forms of the per-dof accessors, one method per verb, documented with the `DofLayout` forms above. Without a
# field name they read the only field of a single-field problem, so its callers need not name it; the active-id form
# also takes a name, for a multi-field problem. Each per-component form takes the same `component` as the `DofLayout`
# form it forwards to, and the active ids come back global, shifted by the field's offset — the numbering a `Solution`'s
# coefficients use.
function cell_dofs(layout::SystemLayout, level::Integer, cell::CartesianIndex)
    cell_dofs(_only_field_layout(layout).dofs, level, cell)
end

function active_cell_dofs(layout::SystemLayout, level::Integer, cell::CartesianIndex, component::Integer=1)
    active_cell_dofs(layout, _only_field_layout(layout).name, level, cell, component)
end

function active_cell_dofs(layout::SystemLayout, name::Symbol, level::Integer, cell::CartesianIndex,
                          component::Integer=1)
    field = _field_layout(layout, name)
    return [_field_component_dof(field, raw, component) for raw in cell_dofs(field.dofs, level, cell)]
end

function constraint_kind(layout::SystemLayout, raw::Integer, component::Integer=1)
    constraint_kind(_only_field_layout(layout).dofs, raw, component)
end

function constrained_value(layout::SystemLayout, raw::Integer, component::Integer=1)
    constrained_value(_only_field_layout(layout).dofs, raw, component)
end

# Look up the per-field layout by name. Throws on unknown name.
_field_layout(system::SystemLayout, name::Symbol) = system.fields[_field_index(system, name)]

# Resolve a field name to its index inside a `SystemLayout`, raising on an unknown name. How the assembly kernel's
# blocks and loads, an interface's region build and a `FormState` read find a field in the layout's field-major data.
function _field_index(system::SystemLayout, name::Symbol)
    index = get(system.by_name, name, 0)
    index == 0 && throw(ArgumentError("unknown field $name"))
    return index
end

# Global active dof id for component `component` of raw `raw` in this field, or 0 if the (raw, component) is
# constrained. Per-field active ids are shifted by the field's `offset` so they live in disjoint blocks of the global
# active enumeration. Read once per slot by assembly's `_frame!` and per branch by `_slots!`, so it stays one table read
# and one add.
function _field_component_dof(layout::FieldLayout, raw::Integer, component::Integer)
    active = layout.dofs.active_component[raw, component]
    return active == 0 ? 0 : layout.offset + active
end

# ── Dof values and the pivot rule ─────────────────────────────────────────────

"""
    dof_value(layout::FieldLayout, coefficients, raw, component=1) -> T

Recover the value of component `component` of raw dof `raw` in this field — the unified accessor every reconstruction
path uses when a basis-function contribution needs the actual dof value regardless of constraint status. `coefficients`
is the system's global active vector, a [`Solution`](@ref)'s coefficients, and the field's `offset` is resolved here.

An active dof reads `coefficients[offset + active]`. A constrained one reads its [`constrained_value`](@ref) — zero for
an artificial-boundary or pruning elimination, the projected datum for a nonzero Dirichlet condition — and, if it is a
**linear-constraint pivot**, adds the pivot's expansion from the field's `pivots` table (see [`DofLayout`](@ref)):

    u_raw = constrained_value(raw) + Σᵢ wᵢ · u_{rᵢ} .

That pivot term is not an embellishment. A pivot is precisely the mechanism a basis family uses to impose C^m continuity
or an arbitrary-mask trace condition, so reading `constrained_value` for it — which is zero — silently drops the dof's
whole contribution from `value`, `field_gradient`, `l2_error`, `write_vtk` and `L2Projection`'s source reconstruction,
while the *solve* is correct because assembly condenses every pivot through the same expansion. Measured on a 6×6
degree-3 C¹ B-spline base under an L-masked 4×4 C¹ overlay (8 non-trivial expansions of 126 raws), the dropped term
reached 4.79e-7 against a field of magnitude 7e-2 — silent, and small enough to pass a hand-written tolerance.

One level of expansion is enough, because `_resolve_rows!` leaves every pivot expressed over raws the artificial
constraints leave free, and the topology build checks it once (`_assert_resolved`). Such a branch may still be
constrained on `component` by a physical Dirichlet condition, which is resolved later; it then contributes its
constrained value rather than a coefficient.

A layout without pivots — every integrated-Legendre layout — stops at the test for an empty table and never reads a
pivot entry.

The expansion is read through `_fold_pivot`, the one statement of the pivot rule that assembly reads as well, so a pivot
cannot be reconstructed one way and assembled another.
"""
function dof_value(layout::FieldLayout, coefficients, raw::Integer, component::Integer=1)
    global_dof = _field_component_dof(layout, raw, component)
    global_dof == 0 || return coefficients[global_dof]
    value, _ = _fold_pivot(_branch_value(coefficients), constrained_value(layout.dofs, raw, component), layout.dofs,
                           (other, c) -> _field_component_dof(layout, other, c), raw, component)
    return value
end

# The pivot rule, stated once. A linear-constraint pivot is
#
#     u(raw, c) = Σₖ wₖ · u(oₖ, c)
#
# over constraint-free raws `oₖ`: `_resolve_rows!` leaves every pivot expressed at depth one, and `_assert_resolved`
# checks it once per build. Two readers share this fold — `dof_value`, which reconstructs a pivot from it, and `_slots!`
# in assembly, which detects a region's pivots, widens its active set to their branches and records those branches for
# `_condense!` to fold the pivot's local row and column onto. When reconstruction and emission each carried their own
# encoding they could disagree, and a field was then solved correctly but reconstructed wrong; one definition rules that
# out.
#
# `f(acc, g, w, v)` is folded over the branches in expansion order, where `g = id(oₖ, c)` is the branch's active id (0
# when it is constrained) and `v` its constrained value when `g == 0`, zero otherwise. `id` maps `(raw, component)` to
# the active id in whichever numbering the caller needs; both readers pass a field's global one, `_field_component_dof`.
# Returns `(acc, true)` for a pivot. A non-pivot returns `(acc, false)` without calling `f`, and needs no test of its
# own: the `pivots` table holds pivots only, so a free raw, a strongly eliminated one and a constraint-free raw a
# Dirichlet condition eliminated on `c` all read an empty entry — or an empty table, when the layout has no pivot at
# all.
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

# The `_fold_pivot` step of `dof_value`: add a branch's weighted value, read from the coefficients when the branch is
# active and from its stored constrained value otherwise — `value += wₖ · u(oₖ, c)`, in expansion order.
_branch_value(coefficients) = (acc, g, w, v) -> acc + w * (g == 0 ? v : coefficients[g])

# ── Integrated Legendre family ────────────────────────────────────────────────

# Tag values for `AxisDofKey.kind`. `UInt8` keeps the key compact and fast to hash; the values themselves are arbitrary.
#
#   _AXIS_NODE   — an endpoint-mode factor at a cell-corner node coordinate.
#                  `index` is the 1-based node coordinate index inside the
#                  level's `mesh.axes[axis]`; `mode` is `0`, the endpoint
#                  contribution being the same shape function whichever of
#                  the two endpoint modes produced it.
#   _AXIS_SPAN   — an interior (bubble) factor, cell-local. `index` is the
#                  1-based cell index along the axis; `mode ≥ 2` selects
#                  which bubble.
#
# A tag need only be unique within one basis family, so each family owns its tags: the B-spline extension defines its
# `_AXIS_BSPLINE` next to its own key construction. Keys of two families can never be confused even where their tags
# coincide. A key carries its level id and one level has one family, so two keys of different families differ in their
# level and the dof-key cache of a mixed-family space never takes one for the other; and `_assert_keys_comparable`
# (`projection.jl`) refuses to match keys across a change of family between two models. The shipped values happen to be
# distinct (the extension's is `0x02`); nothing relies on it.
const _AXIS_NODE = UInt8(0)
const _AXIS_SPAN = UInt8(1)

# Build the per-axis dof key for cell `cell_axis` and 1D mode `mode` in the integrated Legendre basis. Modes 0 and 1 are
# the two endpoint shape functions — they map to `_AXIS_NODE` factors at the cell's lower-endpoint coordinate index
# (`cell_axis`) and upper-endpoint index (`cell_axis + 1`) respectively. Mode m ≥ 2 is a cell-local bubble and maps to
# an `_AXIS_SPAN` factor at index `cell_axis` carrying the mode number for cross-cell distinguishability.
function _axis_dof_key(cell_axis::Int, mode::Int)
    mode == 0 && return AxisDofKey(_AXIS_NODE, cell_axis, 0)
    mode == 1 && return AxisDofKey(_AXIS_NODE, cell_axis + 1, 0)
    return AxisDofKey(_AXIS_SPAN, cell_axis, mode)
end

# Build the full `TensorDofKey` for one tensor-product basis function `local_id` on `cell` of `level`. The integrated
# Legendre specialisation above produces the per-axis keys; a fallback method throws for any other basis family so the
# dof layer fails early instead of producing silently-wrong keys.
function _tensor_dof_key(level::Level{D,T,<:IntegratedLegendre}, cell::CartesianIndex{D},
                         local_id::CartesianIndex{D}) where {D,T}
    axes = ntuple(d -> _axis_dof_key(cell.I[d], local_id.I[d]), D)
    return TensorDofKey{D}(level.id, axes)
end

function _tensor_dof_key(level::Level{D,T,B}, cell::CartesianIndex{D}, local_id::CartesianIndex{D}) where {D,T,B}
    throw(ArgumentError("dof layout is not implemented for basis family $(basis_name(level.basis))"))
end

# True iff the per-axis factor of `key` along `axis` anchors at the level's lower (`side = :lower`) or upper
# (`side = :upper`) mesh edge. Dispatched on the level's basis family so each family's "boundary mode" convention is
# local to its own implementation. Used by the physical-Dirichlet detection (`_mark_physical_dirichlet!` in
# `dirichlet.jl`) to translate "is this dof at a face" into a basis-family-specific predicate without leaking the
# family's mode-index conventions into the rest of the dof layer. The integrated-Legendre `_has_overlay_constraint`
# below does not route through it: it already walks the key's node factors and reads the mesh-edge indices (`1` and
# `cells[axis] + 1`) off them directly.
#
# Integrated Legendre: only `_AXIS_NODE` keys touch a mesh edge — `_AXIS_NODE.index == 1` (lower) or `cells[axis] + 1`
# (upper). Bubbles never touch any edge. B-spline overloads live in `ext/UnfittedBasicBSplineExt.jl`.
function _key_on_level_side(key::TensorDofKey{D}, level::Level{D,T,<:IntegratedLegendre}, axis::Integer,
                            side::Symbol) where {D,T}
    axis_key = key.axes[axis]
    axis_key.kind == _AXIS_NODE || return false
    side === :lower && return axis_key.index == 1
    side === :upper && return axis_key.index == level.mesh.cells[axis] + 1
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# ── Artificial overlay-boundary detection ─────────────────────────────────────

# True iff `key` sits on the artificial boundary of the level's *active region* and therefore carries a homogeneous
# overlay constraint. The active region is the union of active cells; its boundary includes
#
#   * the mesh-box face on every axis where the level's mesh boundary does not coincide with the physical-domain
#     boundary, where it meets Ω (i.e. the artificial overlay boundary `Γ_o^(k) = ∂Ω^(k) \ ∂Ω`);
#   * internal faces between active and inactive cells of the *same* level (the `LevelMask` boundary inside the level).
#
# With `mask === nothing` and no physical domain the active region equals the level's mesh box, only mesh-box faces
# matter, and this reduces to the original overlay-boundary rule from the paper.
#
# One walk per node axis decides it. Only an `_AXIS_NODE` factor sits on a face: at node index `i` it is the face
# between cells `i − 1` and `i` along `axis`, and the key's incidence on the other axes (`_perp_ranges`) gives the
# perpendicular positions the face spans under the key's support. Every position where activity changes across the face
# is a face of the active region, and
#
#   * on a mesh-box face (`i == 1` or `i == n[axis] + 1`) the whole face is exempt when the level side lies on the
#     physical boundary (`_level_side_is_physical`), and otherwise a position constrains when the box-face rule clamps
#     it (`_clamps_box_face`: the active cell's face holds material);
#   * on an internal face, a position constrains when the inactive cell across it holds material (it is not
#     `_is_fictitious`).
#
# Both material tests are one exemption. A face with no material on one side of it — the face of a cell the fictitious
# fold switched off, of a user-masked cell lying entirely outside Ω, or a box face lying wholly outside Ω — has nothing
# there to stay continuous with, and its boundary modes stay active. They vanish on every physical face — so they cannot
# break the C⁰ trace condition the overlay constraint enforces — and instead carry the adjacent active cell's
# approximation (a cut cell's, up to ∂Ω, whenever the fold runs). An inactive cell that holds material — a user mask
# reaching into Ω — is represented by a coarser level only: the modes on its face are non-zero on a physical face and
# must be eliminated to keep the truncated solution continuous. With no physical domain everything holds material, so
# every face of the active region constrains (the original overlay/user-mask behaviour) and `ctx.fold` classifies
# nothing. The walk stops at the first constraining position, so a face classifies its inactive neighbours, or its own
# cell faces, only until one of them holds material.
#
# This runs once per raw on every layout build, and two measured choices keep it cheap: `@inline`, without which the
# call stays out of `_overlay_constraints!`'s per-raw loop (1.1–1.8× the pass time across the overlay, ladder and
# immersed spaces it was measured on), and the side test at the first transition rather than ahead of the walk, so an
# internal face without one pays for nothing but the walk (ahead of it: 1.2× on 2D overlays and ladders).
#
# Integrated-Legendre-specific, because the incidence rule is. The B-spline family in `ext/UnfittedBasicBSplineExt.jl`
# does the same job in its own `_overlay_constraints!`, with a support map that spans several cells and the same
# material exemptions (`_active_cell_at_face`, which asks the same box-face rule).
@inline function _has_overlay_constraint(key::TensorDofKey{D}, level::Level{D,T,<:IntegratedLegendre},
                                         ctx::_LevelContext{D,T}) where {D,T}
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
                _level_side_is_physical(level, ctx.space.domain, axis, side, ctx.tolerance) && break
                boundary = CartesianIndex(ntuple(d -> d == axis ? (below_active ? i - 1 : i) : outer[d], D))
                _clamps_box_face(ctx.fold, level, boundary, axis, side) && return true
                continue
            end
            across = below_active ? i : i - 1          # the inactive cell's axis position
            inactive = CartesianIndex(ntuple(d -> d == axis ? across : outer[d], D))
            _is_fictitious(ctx.fold, level, inactive) || return true
        end
    end
    return false
end

# Cells of `level` along `axis` incident to the axis-`axis` factor of a dof key, clipped to the `n[axis]` cells the
# level has: a `_AXIS_NODE` factor at index `i` touches cells `i − 1` and `i`, since a node sits between two adjacent
# cells along its axis; a `_AXIS_SPAN` (bubble) factor at index `k` touches only cell `k`, since a bubble mode is
# cell-local.
#
# This is the integrated-Legendre node-to-cell support map, and it is the one place it is written down:
# `_incident_cells` applies it on every axis, and `_perp_ranges` applies it on every axis but one. The B-spline family
# in the extension has its own support map — a function's span can cover many cells — and never routes through here.
function _axis_incidence(axis_key::AxisDofKey, ncells::Int)
    axis_key.kind == _AXIS_NODE || return axis_key.index:axis_key.index
    return max(1, axis_key.index-1):min(ncells, axis_key.index)
end

# Cells of `level` incident to the entity of `key`, from the per-axis incidence rule of `_axis_incidence` applied on
# every axis.
function _incident_cells(key::TensorDofKey{D}, n::NTuple{D,Int}) where {D}
    return CartesianIndices(ntuple(d -> _axis_incidence(key.axes[d], n[d]), D))
end

# Incidence ranges of `key` on every axis *except* `axis`, which is pinned to `1:1` so a `CartesianIndices` over the
# result enumerates the perpendicular positions of a candidate face along `axis` exactly once. The `axis` slot of each
# enumerated index is a placeholder the caller replaces with the cell position on either side of the face.
function _perp_ranges(key::TensorDofKey{D}, axis::Integer, n::NTuple{D,Int}) where {D}
    return ntuple(d -> d == axis ? (1:1) : _axis_incidence(key.axes[d], n[d]), D)
end

# True iff the cell whose axis-`axis` index is `pos` (with the other-axis indices drawn from `outer`, ignoring the
# axis-`axis` slot) is in bounds and active under the level's `LevelMask`. Used by `_has_overlay_constraint` to probe
# both sides of a candidate face for an active/inactive transition. A position off the mesh reads as inactive, so a
# mesh-box face is a transition wherever the cell just inside it is active.
function _cell_active(level::Level{D}, axis::Integer, pos::Integer, outer::NTuple{D,Int}, n::NTuple{D,Int}) where {D}
    1 <= pos <= n[axis] || return false
    cell_tuple = ntuple(d -> d == axis ? pos : outer[d], D)
    return is_active(level.mask, CartesianIndex(cell_tuple))
end

# ── The minimum rule: per-cell order across a shared entity ───────────────────
#
# A cell at order 5 beside a cell at order 2 generates dof keys the neighbour does not. Interior bubble keys are
# cell-local (`_AXIS_SPAN` on every axis) and vertex keys carry mode 0 on every axis, so neither is a problem. The keys
# that are shared *and* carry a mode — edge keys in 2D, edge and face keys in 3D — are the whole difficulty: a raw
# generated by only one of the two cells is nonzero on the face they share, so the superposed space is no longer C⁰. It
# assembles, it solves, and it returns a plausible residual; only a two-sided trace evaluation sees it.
#
# The classical hp-FEM fix is the *minimum rule*: a shared entity carries the minimum of the orders of the cells sharing
# it. It is due to
#
#   B. Szabó, I. Babuška, "Finite Element Analysis", Wiley, New York (1991),
#     ISBN 978-0-471-50273-9 — cited in full above `local_basis_indices` in
#     `basis.jl`, where the same work's trunk space is defined; and
#   L. Demkowicz, "Computing with hp-Adaptive Finite Elements, Vol. 1: One and
#     Two Dimensional Elliptic and Maxwell Problems", Chapman & Hall/CRC (2006),
#     doi:10.1201/9781420011685 — the constrained-approximation treatment this
#     package's key-keyed statement of the rule follows.
#
# Three properties of the statement below are load-bearing and none of them is cosmetic:
#
#   * It is a predicate on the **key**, not on the generating cell. The minimum over `_incident_cells(key, n)` is the
#     same number whichever incident cell asks, so both sides of a face agree by construction and the one-sided mode is
#     not expressible. A per-cell filter would have to be kept symmetric by hand.
#   * Every question is asked at an order some cell **actually carries**. The classical statement forms the
#     componentwise minimum first and then asks the family whether the mode survives *there* — at an order that in
#     general no cell has, and where the family's index set therefore has to be defined by extension. That extension is
#     where the rule goes wrong: read per axis it admits a 3D `:trunk` face key with bubble modes (3, 2), whose trunk
#     degree is 5, at order 4 — a mode the order-4 cell never generates, and a measured 0.186 trace jump. Asking each
#     incident cell about its own index set removes the fiction and with it the ambiguity.
#   * Inactive incident cells are **skipped, not minimised over**. An inactive cell generates nothing, so it cannot
#     disagree; letting it lower the entity would delete dofs nothing replaces. That is not hypothetical: under a
#     fictitious fold `_has_overlay_constraint` deliberately keeps the modes on a fold face free, precisely so the
#     adjacent cut cell's approximation reaches ∂Ω.
#
# One pass suffices — no fixed point. The incident cells of a key form a face-connected `2^(#NODE axes)` hypercube, so
# agreement across every face of that block already implies agreement with every cell in it.
#
# WHY THE INTERSECTION IS THE MINIMUM RULE. `_index_admissible` is, for every family that opts into a per-cell order, a
# conjunction of upper bounds on the order — `0 ≤ αᵈ ≤ pᵈ` for `:tensor`, plus `Σ t(αᵈ) ≤ p₁` for `:trunk`. Raising an
# order therefore never removes a mode from a cell's set, only adds: the predicate is monotone non-decreasing in `p`.
# Given that,
#
#     admissible(min over incident cells, α)  ⟺  admissible(order(c), α) ∀ c
#
# so the two statements are the same rule and the minimum need never be formed. The equivalence *rests* on that
# monotonicity, which is a property of the family rather than of this file, so `test_cell_order.jl` asserts it directly
# for every family answering `_supports_cell_order() == true`. A family that broke it would silently get a
# non-conforming space from this code.
#
# What the rule cannot express, and what therefore has to be checked elsewhere: every cell needs `order ≥ 1` on every
# axis. `_axis_dof_key` deliberately collapses the two endpoint modes 0 and 1 into one `_AXIS_NODE` factor — that
# collapse is exactly what makes an endpoint mode shareable across a cell boundary — so no filter over keys can notice
# that a cell is missing one of its two endpoint modes. `_build_cell_orders` in `mesh.jl` validates every palette entry
# through `local_basis_indices` for that reason.

# Integrated Legendre's method of the `_cell_modes` hook whose generic form is in `mesh.jl`. It lives here, with
# `_entity_carries` and the `TensorDofKey` factors the minimum rule is expressed over, the same way the B-spline
# family's burial hooks live in its extension. Only a graded level reaches it — `Level`'s constructor answers a uniform
# one, whose table has no per-cell map — so the order record carries its class array here.
#
# The per-order index sets are hoisted out of the cell loop: there are only `length(palette)` distinct answers and
# `local_basis_indices` allocates a fresh vector on every call, so asking it per cell allocated one vector per active
# cell of the level. Measured on a 16³ level carrying 36 distinct anisotropic orders, 36 973 allocations and 30.6 ms
# against 23 912 and 28.5 ms. The distinct kept sets are interned (`_intern!`) the way the order palette is.
function _cell_modes(basis::IntegratedLegendre, orders::CellOrders{D}, mesh::CartesianMesh{D}, mask) where {D}
    n = mesh.cells
    class = orders.class::Array{UInt16,D}
    sets = [CartesianIndex{D}[]]
    slot = Dict(sets[1] => UInt16(1))
    kind = Array{UInt16,D}(undef, n)
    sets_by_class = [local_basis_indices(basis, o) for o in orders.palette]
    for cell in cell_indices(mesh)
        if !is_active(mask, cell)
            kind[cell] = UInt16(1)
            continue
        end
        kept = CartesianIndex{D}[]
        for id in sets_by_class[class[cell]]
            key = TensorDofKey{D}(0, ntuple(d -> _axis_dof_key(cell.I[d], id.I[d]), D))
            _entity_carries(basis, orders, mask, key, n, id) && push!(kept, id)
        end
        kind[cell] = _intern!(slot, sets, kept, "per-cell mode sets")
    end
    return CellModes{D}(sets, kind)
end

# Whether the shared entity of `key` carries the mode `id`: every *active* cell incident to it generates `id` from its
# own order. The generating cell is itself incident, and `id` came from its own index set, so the loop is never empty
# and the unconstrained case needs no separate answer.
function _entity_carries(basis::BasisFamily, orders::CellOrders{D}, mask, key::TensorDofKey{D}, n::NTuple{D,Int},
                         id::CartesianIndex{D}) where {D}
    for ci in _incident_cells(key, n)
        is_active(mask, ci) || continue
        _index_admissible(basis, _order_at(orders, ci), id) || return false
    end
    return true
end

# ── The burial hooks ──────────────────────────────────────────────────────────

# Integrated Legendre's side of the burial driver `_coverage_constraints!` (see its docstring for the hooks' contract):
# a buried mode with a bubble axis is shed, and a buried linear mode — every factor a node, a C⁰ hat — is deduped by a
# cover that reproduces it.
_sheds_any(::Level{D,T,<:IntegratedLegendre}) where {D,T<:Real} = true

function _sheds(::Level{D,T,<:IntegratedLegendre}, key::TensorDofKey{D}) where {D,T<:Real}
    return any(a -> a.kind == _AXIS_SPAN, key.axes)
end

function _burial_cells(level::Level{D,T,<:IntegratedLegendre}, key::TensorDofKey{D}) where {D,T<:Real}
    return _incident_cells(key, level.mesh.cells)
end

# An integrated-Legendre cover spans the hats of any mesh it nests over, in either mode at any per-cell order ≥ 1: it is
# exactly C⁰ across a cell boundary, and the two endpoint modes it collapses into one node factor have trunk degree 0,
# so Q1 is in every cell's index set. The other rows of the `_reproduces` table involve a B-spline level and are the
# extension's.
_reproduces(::Level{D,T,<:IntegratedLegendre}, ::Level{D,T,<:IntegratedLegendre}) where {D,T<:Real} = true
