# Physical Dirichlet boundary conditions: the user-facing spec types, the per-level physical-Dirichlet detection (one
# face-mask pass over each level's raws) that fills the layout's `physical_dirichlet` matrix, the codim-K facet
# integration machinery (the per-face resolver, regions, the affine-slice restriction that trims a rule to `face ∩ Ω`,
# tensor Gauss quadrature, basis traces), and the L² boundary projection that turns nonzero Dirichlet data into the
# `constrained_values` stored on the layout.
#
# The detection and the projection are steps 2 and 4 of a dof layout's build (see `dofs.jl`): detection reads the dof
# topology of step 1, and the projection fills the layout that step 3's numbering produced. `model.jl` runs the four
# steps in order, so this file loads after `dofs.jl`, whose dof keys, topology and `DofLayout` it operates on, and
# before `model.jl`, which calls it.

# ── Dirichlet condition spec ──────────────────────────────────────────────────

"""
    BoundarySelector(selector, sides)

Specification of a facet on the physical boundary `∂Ω` used by Dirichlet conditions and the dof-layer's constraint
detection. Three `selector` shapes are supported:

  - `:all` — the whole physical boundary `∂Ω`, encoded as the union of all codim-1 faces. `sides` is empty in this case.
  - `:sides` — the intersection of one or more `(axis, side)` pairs listed in `sides`. A single pair picks a codim-1
    face; two pairs in 2D pick a corner; two pairs in 3D pick an edge; three pairs in 3D pick a vertex. Each axis may
    appear at most once and `side` must be `:lower` or `:upper`.
  - `:except` — the union of every codim-1 face *except* the ones listed in `sides`. Here `sides` is a complement rather
    than a facet specification, and an axis may appear twice (both of its faces excluded). The kept faces are taken
    closed, so a dof on the edge between a kept and an excluded face still matches.

`sides` is therefore read against `selector` and never on its own. The kind lives in `selector` rather than in a fourth
field because the facet-region cache keys on a selector's *value* (see the `==` / `hash` methods below), and a `:sides`
selector's `sides` can never be mistaken for an `:except` one's: the discriminant is already there.

Use the public [`boundary`](@ref) constructor instead of building this directly.
"""
struct BoundarySelector
    selector::Symbol
    sides::Vector{Tuple{Int,Symbol}}
end

# Value semantics for `BoundarySelector`. A selector is a *description* of a facet, not an object with an identity:
# `boundary(axis=1, side=:upper)` written twice names the same face both times, and every consumer that reasons about
# selectors — `_physical_dirichlet` and `_facet_regions_for_selector`, which read one only through its `_facets`
# decomposition, and `update_dirichlet!`'s structural check, which compares two — already treats them by value.
#
# The default `==` on a struct carrying a `Vector` falls back to `===`, and the default `hash` to `objectid`, so without
# these two methods a selector is identity-keyed in every `Dict`: two value-equal selectors are distinct keys, a freshly
# built selector can never hit a cache populated at `prepare`, and the identity-based and value-based halves of region
# resolution disagree about which selectors are the same one. `sides` is hashed by value (Julia hashes a `Vector`
# element-wise), matching the element-wise `==` below; the elements are isbits tuples, so both are exact.
Base.:(==)(a::BoundarySelector, b::BoundarySelector) = a.selector === b.selector && a.sides == b.sides
Base.hash(s::BoundarySelector, h::UInt) = hash(s.sides, hash(s.selector, hash(:BoundarySelector, h)))

"""
    DirichletCondition(value, boundary, field, component)

One physical Dirichlet condition. Built by [`dirichlet`](@ref):

  - `value` — Dirichlet datum, applied everywhere on the selected facet: a scalar, a tuple or array with one entry per
    component of the field, or a callback `value(x)` evaluated at physical coordinates.
  - `boundary::BoundarySelector` — the facet to constrain.
  - `field::Union{Nothing,Symbol}` — name of the field this condition applies to. `nothing` applies it to the only field
    of a single-field problem; multi-field problems must name a field explicitly.
  - `component::Union{Nothing,Int}` — component index when constraining only one component of a vector field (e.g. a
    roller boundary that pins the normal component); `nothing` constrains every component.
"""
struct DirichletCondition{F}
    value::F
    boundary::BoundarySelector
    field::Union{Nothing,Symbol}
    component::Union{Nothing,Int}
end

# Reject a constant per-component datum — a tuple or array — whose length is not `components`, the component count of
# the field `name` the condition constrains. The projection reads entry `c` of such a datum for component `c`
# (`_component_value`), and does so under `component = c` too, so the datum carries one entry per component whether or
# not the condition is scoped to one. Unchecked, a wrong length went three ways, none of them an error naming the
# mistake: a short all-zero datum passed silently, because homogeneous data are never projected
# (`_needs_dirichlet_projection`) and so never indexed; a short nonzero one failed inside the projection with an
# incidental `BoundsError`, or not at all when every entry it read was in range; and a long one was accepted with its
# extra entries ignored. A callback is not checked: only evaluating it could say what it returns. The condition, rather
# than its datum, is the argument, so a caller walking an abstractly typed condition list dispatches once per condition
# and never boxes an isbits datum to pass it on.
function _check_dirichlet_datum(condition::DirichletCondition, components::Integer, name::Union{Nothing,Symbol})
    value = condition.value
    (value isa Union{Tuple,AbstractArray} && length(value) != components) || return nothing
    owner = name === nothing ? "its field" : "field $name"
    scoped = condition.component === nothing ? "" :
             "; a condition scoped to component $(condition.component) still reads entry " *
             "$(condition.component) of a per-component value, or takes a scalar"
    throw(ArgumentError("a constant tuple or array Dirichlet datum carries one entry per " *
                        "component of $owner, $components, but this one has length " *
                        "$(length(value))$scoped"))
end

# ── Physical-Dirichlet detection ──────────────────────────────────────────────

# Physical-Dirichlet detection: `marks[raw, c]` is true iff some condition scoped to component `c`
# (`component = nothing` scopes it to every component) selects a facet that raw's basis function lies on. These are the
# flags the active numbering skips and the Dirichlet projection solves for.
#
# A function lies on a facet iff it lies on every codim-1 face the facet is the intersection of. So the pass finds, once
# per raw, the physical faces the raw lies on as a face mask `on`, and a condition matches iff the mask of one of its
# facets is a subset of `on`:
#
#   * `:all` and `:except` decompose into single faces, so this reads "on some (kept) physical face";
#   * `:sides` decomposes into one facet listing every given side, so it reads "on all of them";
#   * an empty `:sides` list is the empty mask, which every raw covers — on every level, including one with no physical
#     side, so no level or raw may be skipped on the grounds that it touches no face.
#
# Selector semantics therefore live in `_facets` alone, the same decomposition the projection and the facet regions
# integrate over, and a malformed selector raises from there with its usual message before any raw is visited, whichever
# component it is scoped to. The masks are built once per condition, so the per-raw loop does no selector work: it
# decomposes no facet, allocates nothing, and asks the family about each of the level's physical sides once per raw
# rather than once per raw, component and condition.
#
# `t` is anything carrying `raw_keys`, `level_raws` and `tolerance` — the build's `_DofTopology`, or a finished
# `DofLayout` — so the pass runs before the layout it feeds exists. Two bits per axis in a `UInt64` hold up to `D = 32`;
# beyond that a face's bit would shift out of the word and the face drop out of its mask unnoticed, so it refuses
# instead.
function _physical_dirichlet(t, V::Space{D,T}, conditions, components::Integer) where {D,T}
    marks = zeros(Bool, length(t.raw_keys), components)
    isempty(conditions) && return marks
    2D <= 64 || throw(ArgumentError("physical-Dirichlet face masks hold at most 32 axes; got D = $D"))
    # Per condition, the columns of `marks` it sets and its facets' masks — all the per-raw loop reads of it, concretely
    # typed, since `conditions` comes from `_dirichlet_for_field` as an `Any[]`. The columns are every component for
    # `component = nothing`, else that one, or none if the field has no such component. Resolving the scope to a range
    # here, rather than carrying the `Union{Nothing,Int}` into the loop, keeps the entries stored inline and the loop
    # free of a per-component test: 1.33 → 0.49 ms for this pass on a 3D 12³ p3 field under `boundary(:all)` (Apple M2
    # Pro).
    every = 1:components
    masks = Tuple{UnitRange{Int},Vector{UInt64}}[(c.component === nothing ? every :
                                                  intersect(c.component:c.component, every),
                                                  UInt64[_face_mask(f) for f in _facets(c.boundary, Val(D))])
                                                 for c in conditions]
    for level in V.levels
        _mark_physical_dirichlet!(marks, t, level, V.domain, masks)
    end
    return marks
end

# Decompose a `BoundarySelector` into the list of facets it covers. For `:all` this is one codim-1 facet per physical
# face (their union is `∂Ω`); for `:except` it is one per *kept* face, the same list with the excluded entries removed;
# for `:sides` it is the single facet identified by the listed `(axis, side)` constraints. Used by `_physical_dirichlet`
# to find the dofs each condition constrains, by `_project_dirichlet_values!` to iterate over the support of each
# Dirichlet condition, and by `_facet_regions_for_selector` to assemble the region union — which is why a union selector
# needs nothing else to be detected or integrated.
function _facets(selector::BoundarySelector, ::Val{D}) where {D}
    if selector.selector === :all
        return [Tuple{Int,Symbol}[(axis, side)] for axis in 1:D for side in (:lower, :upper)]
    elseif selector.selector === :except
        return [Tuple{Int,Symbol}[face] for face in _kept_faces(selector.sides, Val(D))]
    elseif selector.selector === :sides
        for (axis, side) in selector.sides
            1 <= axis <= D || throw(ArgumentError("boundary axis $axis is out of bounds for dimension $D"))
            # Reachable only through a hand-built `BoundarySelector`: `boundary` validates its sides. Refused here,
            # where every consumer decomposes a selector, with the message every side check in the package uses.
            side in (:lower, :upper) || throw(ArgumentError("boundary side must be :lower or :upper"))
        end
        return [selector.sides]
    end

    throw(ArgumentError("unsupported boundary selector $(selector.selector)"))
end

# The codim-1 faces an `:except` selector keeps: every face of the physical box that its `sides` list — the *excluded*
# faces — does not name. The order is `boundary(:all)`'s own, axis-major with `:lower` before `:upper`, so an `:except`
# selector enumerates its facets in exactly the order the whole boundary would, minus the dropped entries. That is what
# makes the Dirichlet projection through one `:except` condition bit-identical to the same faces written out
# individually in that order, rather than merely equal to roundoff: the projection accumulates its mass and right-hand
# side facet by facet in this walk's order, and floating-point addition is not associative.
#
# Both checks need `D` and so cannot live in `boundary`, which builds a selector without reference to a space — exactly
# as the bound check on `boundary(axis=4, side=:lower)` waits for one. Excluding every face is an error, not an empty
# facet list: nothing selected is never what a caller meant, and silently constraining nothing is the failure mode this
# selector exists to prevent.
function _kept_faces(excluded::Vector{Tuple{Int,Symbol}}, ::Val{D}) where {D}
    for (axis, _) in excluded
        1 <= axis <= D || throw(ArgumentError("boundary axis $axis is out of bounds for dimension $D"))
    end
    kept = [(axis, side) for axis in 1:D for side in (:lower, :upper) if (axis, side) ∉ excluded]
    isempty(kept) && throw(ArgumentError("boundary(:all; except=…) excludes all $(2D) faces of " *
                                         "the physical boundary in $(D)D, selecting nothing"))
    return kept
end

# The face mask of one `_facets` entry: the OR of the bits of the codim-1 faces whose intersection the facet is. An
# empty facet — the one an empty `:sides` list decomposes into — is mask 0, which every dof covers.
_face_mask(facet) = mapreduce(((axis, side),) -> _face_bit(axis, side), |, facet; init=UInt64(0))

# A codim-1 face of the physical box as one bit of a `UInt64`: axis `a`'s lower face is bit `2(a − 1)` and its upper
# face the bit above. A set of faces is then the OR of their bits, and "lies on every face of the set" is a subset test
# on two words. `side` is `:lower` or `:upper`: `_facets` has refused any other.
_face_bit(axis::Integer, side::Symbol) = UInt64(1) << (2(axis - 1) + (side === :upper))

# One level of `_physical_dirichlet`, behind a function barrier: `V.levels` is a tuple that may mix basis families, and
# receiving the level as an argument compiles this loop, and the family's `_key_on_level_side` inside it, for the
# level's concrete type instead of dispatching per raw. The level's physical sides — those of its mesh within
# `tol.contain` of the domain's — are found once, so a raw only asks the family which of them its function lies on, and
# the answers accumulate into the raw's face mask `on` in an explicit loop (a `foldl` over the sides takes 1.7× as
# long).
function _mark_physical_dirichlet!(marks::Matrix{Bool}, t, level::Level{D}, domain::AxisBox{D},
                                   masks::Vector{Tuple{UnitRange{Int},Vector{UInt64}}}) where {D}
    sides = [(axis, side) for axis in 1:D
             for side in (:lower, :upper) if _level_side_is_physical(level, domain, axis, side, t.tolerance)]
    for raw in t.level_raws[level.id]
        key, on = t.raw_keys[raw], UInt64(0)
        for (axis, side) in sides
            _key_on_level_side(key, level, axis, side) && (on |= _face_bit(axis, side))
        end
        for (components, facets) in masks
            _covers_a_facet(on, facets) || continue
            for c in components
                marks[raw, c] = true
            end
        end
    end
    return marks
end

# True iff the faces in `on` include every face of one of `facets`' masks. A function of its own on purpose: its closure
# captures this function's argument, which nothing reassigns. Written inline in `_mark_physical_dirichlet!`'s per-raw
# loop, the closure would capture that loop's `on`, which the loop reassigns, and box it — 4.9 ms and 0.8 MB against
# 0.49 ms and nothing per raw for the pass on a 3D 12³ p3 field under `boundary(:all)` (Apple M2 Pro).
_covers_a_facet(on::UInt64, facets::Vector{UInt64}) = any(f -> on & f == f, facets)

# ── Boundary facet integration ────────────────────────────────────────────────

"""
    FacetRegion{D,T} = Region{D,T,SVector{D,T},Vector{Tuple{Int,Symbol}}}

One admissible integration region on a codim-`K` facet of the physical domain (where `K = length(sides)`): a
[`Region`](@ref) whose frame is the region's own face box, degenerate on the `K` pinned axes. It carries the rule over
`face ∩ Ω`, the constant covering-parent list and the facet's outward normal, and every consumer — the Dirichlet
projection, [`boundary_integral`](@ref) and the assembly path — reads it through the region accessors, which map the
frame rule to physical points and weights.

What its fields hold on a facet:

  - `sides::Vector{Tuple{Int,Symbol}}` — the codim-`K` facet identifier (one `(axis, side)` pair per constrained axis).
    For codim-1 facets this is a single pair; for codim-`K > 1` facets the intersection of `K` codim-1 faces. Used by
    the basis-trace machinery ([`boundary_trace_indices`](@ref) and `is_facet_basis`) to filter facet-incident basis
    modes.
  - `lower::SVector{D,T}`, `upper::SVector{D,T}` — the region's own corner coordinates: the facet's fixed coordinate on
    every constrained axis (so `lower[axis] == upper[axis]` there), and the region's merged interval bounds on the free
    axes. They are the region's *own* extent, which after the greedy merge is in general a strict subset of the
    intersection of its parents' cell faces, so a consumer that needs the region's geometry — rather than a bound on it
    — has to read them here. Held as a corner pair rather than an [`AxisBox`](@ref) because a facet box is degenerate on
    its constrained axes and `AxisBox` requires strictly positive extent on every axis.
  - `parents::Vector{Parent{D,T}}` — every level cell whose own facet on `sides` coincides with this region, each with
    its map from the region's frame, constant (`dξ = 0`) on the pinned axes. The parent set is constant across the
    region's quadrature points (the contract that makes "evaluate basis values once per parent" amortizable).
  - `kind::Symbol` — which rule `points` / `weights` carry, in the vocabulary every region's `kind` uses
    (`CONTRIBUTING.md`, "Region quadrature kinds"): `:full` (tensor Gauss over the whole face, because the whole face is
    in `Ω`), `:fictitious` (the face is outside `Ω`, and under strict `α = 0` the rule is empty), `:fictitious_alpha`
    (the same face under `α > 0`, α-scaled tensor Gauss), `:cut_fitted` / `:cut_fallback` (`∂Ω` crosses the face and
    the moment fit on the facet's affine slice succeeded / fell back to the raw Saye rule), `:cut_failed` /
    `:cut_alpha_failed` (the slice carries no rule at all, under `α = 0` / `α > 0` — a face `∂Ω` is merely tangent to,
    or one it crosses in a sliver too thin for the kernel to resolve).

    The geometry is readable straight off it, because the kind is the classifier's own verdict refined by what the fit
    returned: `:full` is a face inside `Ω`, any `:fictitious*` kind a face outside it, any `:cut_*` kind a face `∂Ω`
    crosses. That verdict is [`classify_cell`](@ref) applied to the region's `(D − K)`-dimensional face box against the
    level set restricted to the facet's affine slice, so it carries exactly the classifier's own resolution —
    `subcell_length_scale` and `max_depth` — rather than the rule's point spacing: a fictitious sliver of face is seen
    whenever the classifier resolves it, however the rule's points happen to fall, and one finer than that budget is
    invisible here exactly as it is to the cell classification behind the face. A space carrying no
    [`PhysicalDomain`](@ref) has nothing outside `Ω`, so every region of it is `:full`, and
    `AssemblyDiagnostics.cut_facet_region_count` counts the regions whose kind is not `:full`.
  - `points::Vector{SVector{D,T}}`, `weights::Vector{T}` — the rule in the region's frame, `0` on every pinned axis; the
    physical weight is `weights[k] · J` with `J = vol(face) / 2^(D−K)` (1 on the codim-`D` vertex), the frame
    convention of every region. The rule covers `face ∩ Ω` rather than the whole face, so its physical points lie inside
    `Ω` and its weights sum to the measure of `face ∩ Ω` — up to the α-FCM blend when `α > 0`, which appends the
    full-face tensor points to every trimmed kind and is the one configuration in which a point of a facet rule lies
    outside `Ω`. A `:full` region aliases its resolver's cached tensor rule, and the α-scaled kinds of one face share
    one α-scaled weight vector. **Empty** on a zero-measure region: such a region is kept in the list because its
    parents still carry the trace dofs the dof layer constrains, so `points` is not a proxy for the region's extent —
    read `lower` / `upper` for that.
  - `normal::SVector{D,T}` — outward normal to the facet. For a codim-1 face this is `±eₐ` with the sign set by `side`.
    For codim-`K > 1` facets it is the unit average of the constrained-face outward normals — well-defined and finite,
    though not geometrically as sharp as the codim-1 case.
  - `residual::T` — the moment-fit L² residual on a `:cut_*` region, zero on every kind where no fit ran. Reported
    through `AssemblyDiagnostics.facet_moment_fit_residual_max`.
"""
const FacetRegion{D,T} = Region{D,T,SVector{D,T},Vector{Tuple{Int,Symbol}}}

# Key of the per-*face* memo a `FacetResolver` holds: one entry per `(space, facet)` pair, where the facet is the
# `(axis, side)` list a `BoundarySelector` decomposes into (`_facets`). This is one level below `RegionKey`
# (`model.jl`), which keys whole selectors.
#
# The distinction matters because a selector is a *union* of faces while `_boundary_facet_regions` resolves *one* face,
# so two selectors that are not value-equal can still overlap: `boundary(:all)` covers every face, hence every face any
# other selector names. Keyed on the selector, each overlap resolves the shared face again; keyed on the face, each face
# of each space is resolved exactly once and the selectors' region vectors are assembled from the shared results. That
# is worth doing because the overlap is not rare — it is what any spelling of "the whole boundary except one face"
# produces for a coupled problem, one such selector per field.
#
# The space half is identity-keyed, because `problem_spaces` already deduplicates spaces by `===`: "a distinct
# discretisation" is package-wide synonymous with "a distinct `Space` object". The `sides` half is a
# `Vector{Tuple{Int,Symbol}}`, which `Tuple` hashes and compares element-wise, so the same face spelled by two different
# selectors is one key.
const FacetKey = Tuple{Any,Vector{Tuple{Int,Symbol}}}

"""
    FacetResolver{D,T}(tolerance)
    FacetResolver{D,T}(tolerance, faces, rules)

The package's one route from a boundary face to its [`FacetRegion`](@ref)s: the geometry tolerance the resolution runs
at, together with the per-face memo (`FacetKey` → region list) it fills as faces are asked for and the tensor rules
those regions share.

One resolver belongs to a [`Model`](@ref) and lives as long as it does, so every consumer of grid-aligned boundary
integration shares one resolution per `(space, face)` pair — the per-selector cache `prepare` builds
(`_resolve_facet_regions`), the one-shot lookup `_region_list` falls back to for a selector no `prepare` saw, and the L²
Dirichlet projection (`_dirichlet_projection`), which reaches the same faces from the dof layer. Sharing is worth
arranging because `_boundary_facet_regions` is a pure function of `(V, sides, tolerance)`: a second resolution of a face
can only ever reproduce the first one, at full price.

The memo is filled lazily and so is mutated by lookups, which is safe for the same reason the integration plan's
moment-fit cache is: every resolution happens on the task that called the public API — `prepare`, `update_dirichlet!`,
`boundary_integral`, or an `assemble*` call resolving its passes — and never inside an assembly worker, which is handed
a region list that is already built.

Fields:

  - `tolerance::GeometryTolerance{T}` — the tolerance every resolution through this memo runs at. It travels with the
    memo rather than with each call because it is part of what a cached entry *means*: the coordinates a face is
    partitioned at are merged against it, so regions built at one tolerance are not the regions another would have
    produced, and a consumer holding a different one must not read these entries. `_project_dirichlet_values!` raises on
    the mismatch rather than resolving against the wrong memo.
  - `faces::Dict{FacetKey,Vector{FacetRegion{D,T}}}` — the memo itself. The space half of a key is matched by identity,
    so an entry for a space that is no longer referenced is dead weight and never a stale answer; every mutator replaces
    the model's resolver along with the dof layout, which is what keeps such entries from accumulating across a move or
    a mask flip. Within one space the memo is bounded by construction: a `BoundarySelector` decomposes into codim-`K`
    facets of the background box, of which there are at most `3^D − 1` (each axis is unconstrained, pinned low, or
    pinned high), so no sequence of one-shot lookups can grow it without limit.
  - `rules::Dict{NTuple{D,Int},TensorQuadrature{D,T}}` — the tensor Gauss rules of the resolved regions, keyed by their
    per-axis counts (0 on a pinned axis). Every `:full` region aliases its rule from here, so a boundary of many faces
    holds one array per distinct order rather than one per region. The rules depend on the counts alone, so the cache
    is shared by every face and space the resolver serves and is bounded by the distinct orders they reach.
"""
struct FacetResolver{D,T<:Real}
    tolerance::GeometryTolerance{T}
    faces::Dict{FacetKey,Vector{FacetRegion{D,T}}}
    rules::_TensorRuleCache{D,T}
end

function FacetResolver{D,T}(tolerance::GeometryTolerance{T}) where {D,T<:Real}
    return FacetResolver{D,T}(tolerance, Dict{FacetKey,Vector{FacetRegion{D,T}}}(), _TensorRuleCache{D,T}())
end

# Resolve one face of `V` through `facets`, building it with `_boundary_facet_regions` (defined below) on a miss. This
# is that function's only `get!` call site anywhere in the package, which is what makes "one resolution per (space,
# face) per model lifetime" a property of the code rather than a convention several independent call sites happen to
# honour.
function _resolve_face(facets::FacetResolver{D,T}, V::Space{D,T}, sides::Vector{Tuple{Int,Symbol}}) where {D,T}
    return get!(() -> _boundary_facet_regions(V, sides, facets.tolerance, facets.rules), facets.faces, (V, sides))
end

# Build the admissible boundary regions on the facet identified by `sides`.
#
# The **partition** is grid-aligned and level-set-blind, and that is the one property to know before using them. No
# `PhysicalDomain` reaches steps 1–3: step 2 selects levels by comparing mesh edges against the space's `AxisBox`
# (`_level_side_is_physical`), partitions over `_side_cells` and asks `locate_cell` and the `LevelMask` for every
# signature. So a region *is* the whole face of its parent cells — but the **rule** on it is not. Step 4 restricts the
# level set to the facet's affine slice and integrates `face ∩ Ω`, so a region's weights sum to the measure of the
# physical part of its face and never to the face's full geometric measure. Every consumer inherits that: the L²
# Dirichlet mass and right-hand side below, a Neumann / Robin / Nitsche term tagged `on::BoundarySelector`, and
# `boundary_integral`. `FacetRegion.kind` names which rule the region ended up with, and `cut_facet_region_count` in the
# assembly diagnostics counts the regions whose kind is not `:full` — exactly the faces part of whose area falls outside
# Ω. Integration over the immersed boundary itself goes through a `BoundaryMesh` (`surface.jl`), which is cut against
# the grid in the same spirit. The whole-cell fictitious fold reaches here too, through the mask step 2 reads: a fully
# fictitious cell is inactive and parents nothing, so a face loses whole cells through the mask and the part of a
# surviving cell's face that lies outside Ω through the rule.
#
# Two things the partition's level-set-blindness still costs, and they are the reason the distinction above is worth
# stating. A region whose face lies wholly outside Ω is **kept**, with an empty rule, because its parents carry trace
# dofs the dof layer constrains on the same grid-aligned test; and a region's extent is its whole face, so `points` is
# not a proxy for the geometry the region covers.
#
# The geometry arrives through `V`, not through an argument of its own, so this stays a pure function of
# `(V, sides, tol)` — which is what lets the per-face memo of a [`FacetResolver`](@ref) key on those three alone and
# what keeps the two consumers above reading the same faces automatically. `rules` is the resolver's tensor-rule cache,
# which only decides whether a region's rule is a shared array or a fresh one with the same values.
#
# The partition is the volume plan's own engine (`_partition` in `intersections.jl`) restricted to the facet:
#
#   1. The facet's free axes are the engine's free axes, and its pinned coordinates ride in the anchor, so every
#      candidate midpoint lies on the facet. The codim-`D` vertex facet has no free axis at all: its candidate grid is
#      the single empty index, and its region is the point evaluation such a facet should be.
#   2. Per level, the eligible cells are those whose own face lies on `sides` (`_side_cells`) — the only cells that
#      carry trace dofs here — and none at all on a level whose own mesh face does not coincide with the physical-domain
#      face (`_level_side_is_physical`). Each free axis's coordinates are those that bound an *active* eligible cell,
#      and a candidate's signature is `0` on every level that contributes nothing there: a cell off the side, or one
#      the `LevelMask` deactivated, has no dofs to constrain. A candidate no level covers seeds no region.
#   3. The greedy merge fuses axis-adjacent candidates that share a signature, exactly as over the volume. Every point
#      of a merged region still lies inside one cell of every parent level, so a product of boundary traces is still
#      polynomial on it and the rule below is still exact for it. The parents are decoded from the signature
#      (`_parents`), each with its map from the region's frame, the face box.
#   4. Restrict the space's level set to the facet's affine slice, classify each region's own face box against it
#      (`:full`, `:cut`, `:fictitious`), and build the rule that verdict calls for (`_facet_regions`): the full-face
#      tensor Gauss rule, an α-scaled copy of it, or the moment fit on the slice, all of the face's fits at once on
#      every thread. This is the facet counterpart of the volume path's step 7 (`CONTRIBUTING.md`, "Integration
#      regions"), sharing its records (`_region_record`), its rule builder (`_trimmed_rule`), its α-FCM blend and its
#      kind vocabulary one dimension down. Every rule lives in the region's own frame — the face box, `0` on the pinned
#      axes — as every region's does, so the region accessors give its consumers physical points and weights.
#
# Steps 2 and 3 together make the facet partition follow the *active* cells on the boundary rather than the finest
# touching level's background grid. One consequence is worth stating plainly, because it is the one thing the merge
# changes rather than merely makes cheaper. The exactness argument covers products of boundary traces: the Dirichlet
# mass matrix, and the Neumann / `boundary_integral` terms. It does not cover a *non-polynomial* datum `g` in the
# Dirichlet right-hand side `b_i = ∫_∂Ω g φ_i`. The finer slicing integrated that with an incidental composite rule at
# the finest touching level's boundary spacing; it is now integrated with the parent cell's own rule — the same
# convention `assembly.jl` already uses for a non-polynomial volume source. Projected Dirichlet values for such a datum
# therefore move at the boundary-quadrature-error level, not at roundoff.
#
# On a *cut* face the exactness argument is the moment fit's rather than tensor Gauss's, and it reaches the same degree
# by construction: the order is the volume path's own `_moment_orders` read over the free axes, so with the default
# `moment_order_factor = 2` the fitted rule reproduces the moments of products of traces. The maximum over parents
# `_axis_max` takes is exactly what a merged region needs, for the same reason the tensor counts take one — `adapt` and
# per-cell order let a region span cells of different orders, and anything less would under-integrate the finest of
# them.
#
# That makes the rule exact where the restricted level set is piecewise affine — one planar leaf, or several meeting at
# a corner strictly *inside* the face box, where the quadrature kernel's pair leaves put a base breakpoint (`_PairLeaf`
# in `implicit.jl`; a face cut by two planes meeting inside it comes out exact to round-off) — and accurate to the
# reported residual and the kernel's approximation where it curves: never exact on a curved cut, which is why nothing
# here promises exactness there. `boundary`'s docstring states the condition for the public reader.
function _boundary_facet_regions(V::Space{D,T}, sides::Vector{Tuple{Int,Symbol}}, tol::GeometryTolerance{T},
                                 rules::_TensorRuleCache{D,T}=_TensorRuleCache{D,T}()) where {D,T}
    free_axes = _free_axes(sides, Val(D))
    return _boundary_facet_regions(V, sides, free_axes, tol, rules, Val(length(free_axes)))
end

# The partition and the records of one facet, behind a `Val(F)` barrier on its number of free axes. `F` is a property of
# the facet's codimension and so a runtime value at the call site; the barrier is what makes the engine's candidate
# grid, signature grid and `visited` bitmap concretely typed, which its inner slab test needs to stay a static dispatch.
function _boundary_facet_regions(V::Space{D,T}, sides::Vector{Tuple{Int,Symbol}}, free_axes::Vector{Int},
                                 tol::GeometryTolerance{T}, rules::_TensorRuleCache{D,T}, ::Val{F}) where {D,T,F}
    anchor = SVector{D,T}(ntuple(d -> _facet_fixed_coord(V.domain, sides, d), Val(D)))
    cells = map(V.levels) do level
        touching = all(_level_side_is_physical(level, V.domain, axis, side, tol) for (axis, side) in sides)
        return touching ? _side_cells(level, sides) : CartesianIndices(ntuple(_ -> 1:0, Val(D)))
    end
    # Ω restricted to this facet's affine slice, built once per face rather than once per region: the pinned coordinates
    # are the face's own, so every region on the face is classified and integrated against the same restricted tree.
    pins = _facet_pins(V.domain, sides)
    slice = V.physical === nothing ? nothing : _restrict_domain(V.physical, pins)
    records = _RegionRecord{D,T}[]
    _partition(V.levels, ntuple(j -> free_axes[j], Val(F)), anchor, tol, cells) do lower, upper, sig
        verdict = _classify_facet(slice, lower, upper, free_axes, Val(F))
        push!(records, _region_record(V, lower, upper, _parents(V.levels, sig, lower, upper), verdict))
        return nothing
    end
    return _facet_regions(records, slice, sides, free_axes, pins, rules, Val(F))
end

# The regions of one face's records: every cut region's moment fit on the slice, then each record's rule through
# `_trimmed_rule` (`intersections.jl`). A function barrier on `slice`, which is not concretely typed where it is built,
# so the per-region rule dispatch below is static.
#
#   - The base tensor rule comes from the resolver's cache, so a `:full` region — the overwhelming majority: 28 of
#     kirsch's 30 cached ones, 44 of the 54 on the space-time cavity's six faces at `SC_CELLS = 3` — aliases it and
#     costs no allocation. Its counts are 0 on the pinned axes, so its points are the free-axis Gauss points with 0 on
#     every pinned axis and its weights the free-axis weight products, unchanged.
#   - The α-scaled weights are cached per face, not per resolver: one resolver serves every space of a model, and two
#     spaces may carry different `α`.
#   - The fits run through `_fit_rules`, on every thread, and come back in the face box's own reference frame; a `0` on
#     each pinned axis embeds them in the region's frame. The domain's `cut_quadrature` is **not** consulted: it is
#     documented as a rule on a cell box in the model's own dimension, so handing a user's callable a face box one
#     dimension down would silently change what it is being asked for — a facet cut region always goes through the
#     package's own `moment_fit_rule`.
#
# One deliberate departure from the volume plan: a zero-measure region — `:fictitious` under strict `α = 0`, or
# `:cut_failed` — is kept rather than dropped. Its parents carry trace dofs the dof layer constrains on the same
# grid-aligned test, and dropping it would move `facet_region_count` without changing one integral. The codim-`D` vertex
# facet (`F = 0`) never reaches the cut branch: `_classify_facet` decides a single point by membership.
#
# The fits are entered through `Base.inferencebarrier` (the idiom of `_coverage_constraints!` in `dofs.jl`), so
# inference never descends from here into the moment-fit chain — `_fit_rules`, `moment_fit_rule`, the implicit kernel,
# NNLS. The call is resolved at run time on the slice's concrete type, and the chain is inferred and compiled when a
# face first has a cut region, not before. Without the barrier, the abstract inference `_regions` (`assembly.jl`) starts
# on its unspecialised `space` reaches this call with the face's dimension and number type unknown and infers the whole
# chain for them, in code that never runs — on the Kirsch plate's first assemble (12² cells, p = 3, no facet region
# cut), 2726 of the 8249 method instances it inferred; with the barrier it infers 4730 and none of the chain. The
# result is asserted to its concrete type, so the rule loop below stays statically dispatched.
function _facet_regions(records::Vector{_RegionRecord{D,T}}, slice, sides::Vector{Tuple{Int,Symbol}},
                        free_axes::Vector{Int}, pins, rules::_TensorRuleCache{D,T}, ::Val{F}) where {D,T,F}
    function key(record)
        face = _facet_box(record.lower, record.upper, free_axes, Val(F))
        return (face.lower, face.upper, ntuple(j -> record.order[free_axes[j]], Val(F)))
    end
    keys = _MomentFitKey{F,T}[key(record) for record in records if record.verdict === :cut]
    fits = isempty(keys) ? _MomentFitValue{F,T}[] :
           Base.inferencebarrier(_fit_rules)(slice, keys, nothing)::Vector{_MomentFitValue{F,T}}
    frame_pins = [(axis, zero(T)) for (axis, _) in pins]
    alpha = slice === nothing ? zero(T) : slice.alpha
    alpha_cache = Dict{NTuple{D,Int},Vector{T}}()
    normal = _facet_outward_normal(sides, Val(D), T)
    regions = FacetRegion{D,T}[]
    next = 0
    for record in records
        fit = nothing
        if record.verdict === :cut
            points, weights, residual, status = fits[next += 1]
            fit = (SVector{D,T}[_lift(p, frame_pins) for p in points], weights, residual, status)
        end
        kind, points, weights, residual = _trimmed_rule(record.verdict, record.counts, rules, alpha_cache, alpha, fit,
                                                        false)
        push!(regions,
              Region(record.lower, record.upper, record.parents, kind, points, weights; normal, sides, residual))
    end
    return regions
end

# Axes *not* constrained by `sides`. Each codim-1 face has `D − 1` free axes; a codim-D vertex has zero free axes. Used
# to size the free-axis tensor quadrature and to enumerate which axes carry physical extent on the boundary region.
function _free_axes(sides::AbstractVector{<:Tuple{Integer,Symbol}}, ::Val{D}) where {D}
    constrained = Set{Int}()
    for (axis, _) in sides
        push!(constrained, Int(axis))
    end
    return [d for d in 1:D if !(d in constrained)]
end

# Fixed coordinate value for axis `d` of a facet specified by `sides`. Returns the side's coordinate
# (`domain.lower[axis]` or `domain.upper[axis]`) when `d` is constrained by some entry of `sides` and `zero(T)`
# otherwise — the caller overwrites the free-axis entries with actual interval bounds.
function _facet_fixed_coord(domain::AxisBox{D,T}, sides::AbstractVector{<:Tuple{Integer,Symbol}},
                            d::Integer) where {D,T}
    for (axis, side) in sides
        axis == d || continue
        return side === :lower ? domain.lower[axis] : domain.upper[axis]
    end
    return zero(T)
end

# The cells of `level` whose own faces all lie on `sides`: index `1` or the last index on each constrained axis, the
# whole range on the free ones. These are the only cells that carry trace dofs on the facet, so they are the level's
# eligible cells in the facet's partition (`_partition`): its signature accepts no other parent, and its coordinate
# projection follows the *active boundary layer* rather than the level's whole active region — an overlay live deep in
# the interior then contributes nothing to a facet it never touches.
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

# Outward unit normal of a facet identified by `sides`. For codim-1 facets this is `±eₐ` with the sign set by the single
# `side`. For codim-`K > 1` facets the constrained-face outward normals are summed and normalised; well-defined for
# every `K ≤ D` and continuous in the face geometry, though not geometrically as sharp as the codim-1 case (a codim-2
# corner in 2D returns the diagonal-out direction).
function _facet_outward_normal(sides::AbstractVector{<:Tuple{Integer,Symbol}}, ::Val{D}, ::Type{T}) where {D,T}
    accumulator = MVector{D,T}(ntuple(_ -> zero(T), D))
    for (axis, side) in sides
        sign = side === :lower ? -one(T) :
               side === :upper ? one(T) : throw(ArgumentError("boundary side must be :lower or :upper"))
        accumulator[axis] += sign
    end
    n = SVector{D,T}(accumulator)
    norm_n = sqrt(sum(x -> x * x, n))
    return iszero(norm_n) ? n : n / norm_n
end

# ── Facet slices ──────────────────────────────────────────────────────────────
#
# A codim-`K` facet is the affine slice `⋂ⱼ {x_{kⱼ} = vⱼ}` of the background box, so the geometry of Ω *on* that facet
# is the level set restricted to the slice — a `PhysicalDomain` of dimension `F = D − K` (`_restrict_domain` in
# `src/physical.jl`). What follows is the coordinate half of that correspondence — the pins that name the slice, the
# facet region's extent as a box in the slice's own `F` coordinates, and the map back to ℝᴰ — and then the
# classification those three feed.

# The facet's pinned axes as `(axis, coordinate)` pairs in DESCENDING axis order: the order `_restrict_domain` must
# consume them in, and the reverse of the one `_lift` splices them back in. The coordinate is the facet's own, so the
# pins are a property of `(domain, sides)` alone and are shared by every region on the face.
function _facet_pins(domain::AxisBox{D,T}, sides::Vector{Tuple{Int,Symbol}}) where {D,T}
    return sort!([(axis, _facet_fixed_coord(domain, sides, axis)) for (axis, _) in sides]; by=first, rev=true)
end

# The facet region's own extent as an `AxisBox` over its `F` free axes — the box the slice is classified and integrated
# on. Built by *selecting* the free axes rather than by dropping the pinned ones one at a time, because the intermediate
# box of a codim-`K > 1` drop is degenerate on the axes still pinned and `AxisBox`'s inner constructor rejects that
# (which is also why `FacetRegion` stores a corner pair instead of a box).
function _facet_box(lower::SVector{D,T}, upper::SVector{D,T}, free_axes::Vector{Int}, ::Val{F}) where {D,T,F}
    return AxisBox(SVector{F,T}(ntuple(j -> lower[free_axes[j]], Val(F))),
                   SVector{F,T}(ntuple(j -> upper[free_axes[j]], Val(F))))
end

# Splice the pinned coordinates back into a point of the slice, the inverse of the free-axis selection above and the map
# that returns anything built on the slice to `D` coordinates — the region's frame when the pins carry the frame
# coordinate `0` (`_facet_regions`), physical space when they carry the face's own. The pins are consumed in
# ASCENDING axis order — `_facet_pins` holds them descending, hence the reverse — because an insertion at index `k`
# lands on axis `k` of the result only once every axis below `k` is already in place.
_lift(x::SVector, pins) = foldl((y, pin) -> _insert_axis(y, pin[1], pin[2]), Iterators.reverse(pins); init=x)

# How Ω meets one facet region's own face: `:full`, `:cut` or `:fictitious`, the same three verdicts `classify_cell`
# returns on a cell box, reached on the region's face box against `slice`. `_facet_regions` turns the verdict into the
# rule it calls for, and into the finer `kind` that names which rule came back. It is the classifier's verdict with
# exactly the classifier's resolution — the Lipschitz certificate where a leaf carries a constant, octree-bounded corner
# sampling where it does not — and in particular it is not a sampling of the *rule* on the region. `FacetRegion.kind`'s
# own docstring states what that resolution does and does not see.
#
# Three cases, by dispatch rather than by branch. A space with no `PhysicalDomain` has nothing outside Ω to find. A
# codim-`D` facet is a single point, with no box to classify, so its membership in Ω *is* the verdict — the slice's
# coordinate space is `ℝ⁰` and the lone point of it is the facet's own corner. Everything else classifies the face box.
_classify_facet(::Nothing, lower, upper, free_axes, ::Val) = :full

function _classify_facet(slice::PhysicalDomain, lower::SVector{D,T}, upper::SVector{D,T}, free_axes::Vector{Int},
                         ::Val{F}) where {D,T,F}
    return classify_cell(slice, _facet_box(lower, upper, free_axes, Val(F)))
end

function _classify_facet(slice::PhysicalDomain, ::SVector{D,T}, ::SVector{D,T}, ::Vector{Int}, ::Val{0}) where {D,T}
    return _inside(slice.geometry, SVector{0,T}()) ? :full : :fictitious
end

# ── Boundary trace evaluation ─────────────────────────────────────────────────

"""
    boundary_trace_indices(level, cell, raw_dofs, sides) -> NamedTuple

The datum-independent, *point*-independent half of a boundary trace: which of `level`'s local basis functions have
support on the codim-K facet identified by `sides`. Returns a `NamedTuple` with four fields:

  - `raw_dofs::Vector{Int}` — the subset of `raw_dofs` whose basis functions touch the facet (filtered by
    `is_facet_basis`), in the original local-basis order.
  - `local_ids::Vector{CartesianIndex{D}}` — their tensor-product multi-indices, in the same order.
  - `values::Vector{T}` — the buffer [`boundary_trace_values!`](@ref) refills at each quadrature point; undefined until
    it does.
  - `factors::NTuple{D,Vector{T}}` — the per-axis 1D scratch that same call refills; sized for the level's full mode
    range, not the facet subset, because the family fills whole axes.

The subset depends only on `(level, sides)`, and a [`FacetRegion`](@ref)'s parent list is constant across its quadrature
points, so one call per parent covers a whole region — the trace's per-point cost is then the tensor product alone. The
per-axis scratch the tensor product needs is allocated here too, for the same reason.

This half is basis-family-agnostic: it asks only for [`cell_basis_indices`](@ref) and `is_facet_basis`, which every
family supplies. The per-axis scratch is sized from the level's nominal order, which is an upper bound on every cell's.
"""
function boundary_trace_indices(level::Level{D,T}, cell::CartesianIndex{D}, raw_dofs::Vector{Int},
                                sides::Vector{Tuple{Int,Symbol}}) where {D,T}
    # The parent `cell` selects the level's minimum-rule index list for that cell, which is what `raw_dofs` was built
    # from. Taking the level-wide list instead would pair the trace modes with the wrong raws on a per-cell-order level
    # — silently, because the pairing is positional and unchecked.
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

The per-point half of a boundary trace: refill `trace.values` with the basis traces at reference point `xi` on the
parent cell, for the facet-incident modes [`boundary_trace_indices`](@ref) selected. `trace` is that call's record,
extended by the caller with the parent's `cell`. Mirrors the per-level evaluation buffers an assembly workspace
refreshes for the volume (`_refresh!` in `assembly.jl`).

The trace of a tensor-product basis function on a facet is the volume tensor product evaluated at a point that happens
to lie on the facet, so this is `_tensor_values!` restricted to the facet-incident multi-indices — one code path for
every basis family, reaching the family only through its `_fill_factor_tables!` hook. `sides` is already spent by
[`boundary_trace_indices`](@ref) and plays no part here.
"""
function boundary_trace_values!(level::Level{D,T}, trace, xi::SVector{D,T}) where {D,T}
    _tensor_values!(level.basis, trace.values, trace.local_ids, nominal_order(level), xi, trace.factors, trace.cell)
    return trace
end

# ── Dirichlet projection ──────────────────────────────────────────────────────

# Project nonzero physical Dirichlet data onto the boundary trace space by an L² mass-matrix solve. For each physical
# Dirichlet constrained dof we want a value `g_h` such that
#
#     ∫_∂Ω g_h v dx  =  ∫_∂Ω g v dx
#
# for every test trace `v` from the constrained-dof subspace. This is the classical L² projection; the resulting linear
# system is
#
#     M c = b,   M_ij = ∫_∂Ω φ_i φ_j dx,   b_i = ∫_∂Ω g φ_i dx,
#
# where `φ_i` are the boundary traces of the physically-constrained raw dofs that survived every artificial elimination
# — the overlay-boundary condition and, under leaf semantics, the shedding of buried high-order modes and the linear
# dedup (`elimination_source === :free`). An eliminated dof has no value to project: its coefficient is not a solved
# unknown of the system. The mass matrix is symmetric positive (semi-)definite, so we try a Cholesky factorisation first
# and fall back to a pseudoinverse on the rare cases where Cholesky reports indefiniteness (degenerate facets, zero-area
# boundaries under heavy masking).
#
# `M` and the boundary walk that builds it depend on the mesh alone; only `b` depends on the prescribed data. A
# load-stepping driver calls [`update_dirichlet!`](@ref) once per increment on a fixed mesh, so the walk is split in
# two: [`DirichletProjection`](@ref) holds everything the datum does not enter — the per-component unknown sets, the
# factorised mass, and the sampled boundary traces — and is built once and reused, while each increment re-accumulates
# `b` from the samples and back-substitutes. See `DirichletProjection` for why that reuse is exact and when it is
# invalidated.
#
# The `_test_value_contribution` and `_as_test_channels` helpers are defined in `problems.jl`'s channel layer and reused
# here so the Dirichlet projection rides the same channel calculus the volume assembly uses; the forward reference
# resolves at call time (`prepare(problem)` loads every source file first).

"""
    FacetTraceSamples{D,T}(points, weights, offsets, values, rows)

The quadrature sampling of one Dirichlet condition's facet: every boundary quadrature point on that facet together with
the boundary traces of the constrained dofs evaluated there.

The right-hand side `b_i = ∫_∂Ω g φ_i dx` is the only part of the L² projection the prescribed data enter, and it is a
quadrature sum over exactly these samples. Recording them lets a load increment re-evaluate `g` and re-accumulate `b`
without re-deriving the facet regions or the basis traces — the two dominant costs of the walk, both of which scale with
the active cells on the boundary.

Fields:

  - `points::Vector{SVector{D,T}}` — the facet's physical-frame quadrature coordinates, in the order the region walk
    visits them.
  - `weights::Vector{T}` — the matching physical-frame weights, with the facet Jacobian already applied (see
    [`FacetRegion`](@ref)).
  - `offsets::Vector{Int}` — sample pointers of length `length(points) + 1`: the trace entries of sample `k` are
    `offsets[k] : offsets[k+1] - 1`.
  - `values::Vector{T}` — each entry's trace value `φ_i(x_k)`.
  - `rows::Vector{Vector{Int}}` — per component, each entry's row in that component's unknown set, or `0` when the
    entry's dof is not an unknown of that component. Components the condition does not constrain carry an empty vector
    and are never indexed.
"""
struct FacetTraceSamples{D,T<:Real}
    points::Vector{SVector{D,T}}
    weights::Vector{T}
    offsets::Vector{Int}
    values::Vector{T}
    rows::Vector{Vector{Int}}
end

# One component's solved boundary mass: the Cholesky factorisation when the mass is positive definite, its pseudoinverse
# when Cholesky reports indefiniteness, and `nothing` when the component has no unknowns at all.
const DirichletFactor{T} = Union{Nothing,Cholesky{T,Matrix{T}},Matrix{T}}

# Which of those three a component's factor is, as a short tag. The pseudoinverse branch is the only externally visible
# trace of a boundary trace space that has gone singular — `cholesky(…; check = false)` reports the indefiniteness and
# the projection carries on against the minimum-norm solution — so the branch is reported rather than merely taken; see
# `AssemblyDiagnostics.dirichlet_trace_factors`. A tag and not the type itself, so neither a reader nor a recorded
# report is reading a Julia version's choice of type parameters.
_trace_branch(::Nothing) = :none
_trace_branch(::Cholesky) = :cholesky
_trace_branch(::AbstractMatrix) = :pseudoinverse

"""
    UnsupportedDirichletDof(field, component, raw)

Diagnostic record for a dof that a Dirichlet condition constrains with no measure anywhere on its facet support. Which
dofs a condition constrains is decided per dof key by the grid-aligned face test alone, while what is *integrated* over
a face is only the part inside `Ω` (see [`boundary`](@ref)), so on an immersed space the two can come apart: every facet
region this dof's trace reaches is trimmed down to nothing, its row and column of the boundary mass `∫ φᵢ φⱼ ds` are
identically zero, and the L² projection determines no value for it. The pseudoinverse the trace solve falls back to then
assigns it the minimum-norm value — zero, to the roundoff of the singular-value decomposition.

The configuration is reachable only inside one cell, and no whole-cell fold can stand in for it: a cell whose own faces
lie entirely outside `Ω` while the cell itself is cut stays active and keeps parenting regions on those faces. Measured
on `Ω = (0,1)² ∖ {‖x‖ ≤ 0.3}` at 4×4, order 2, with a nonzero datum on `boundary(:all)` — the corner cell `[0, ¼]²` is
cut (its far corner sits at `r = 0.354 > 0.3`) while both of its boundary faces lie at `r ≤ 0.25`: three dofs report
here, the node at the origin and the two face bubbles of that cell.

Reported, not resolved. What a Dirichlet condition *ought* to mean on such a dof is an open semantic question rather
than a defect in the rule: its *volume* support still reaches into `Ω`, so leaving it free instead is a defensible
answer — and a different discretisation, since it moves `active_unknowns` and the size of the system. The package
therefore pins the value it does assign and names the dofs it assigned it to. Fields:

  - `field::Symbol` — the field whose dof layout the dof belongs to.
  - `component::Int` — the component the condition constrains it in. A component-scoped condition integrates only its
    own facets, so one component of a dof can be unsupported while another is fitted normally.
  - `raw::Int` — the dof's raw index, i.e. its position in that field's [`DofLayout`](@ref) `raw_keys`, which is where
    its level and its per-axis modes are read back from.

The records are built from the projection itself, out of its per-component `unsupported` lists (see
[`DirichletProjection`](@ref)), by `prepare` for the projections it builds and by [`update_dirichlet!`](@ref) for the
ones it caches. The dof layout keeps no copy, so they are read through [`diagnostics`](@ref).
"""
struct UnsupportedDirichletDof
    field::Symbol
    component::Int
    raw::Int
end

"""
    DirichletProjection{D,T}(facet_count, unknowns, unsupported, factors, samples)

Everything in the L² Dirichlet projection `M c = b` that the prescribed data `g` do not enter: the per-component unknown
sets, which of those the mass leaves undetermined, the per-component factorised boundary mass, and the per-facet trace
samples a right-hand side is accumulated from.

Fields:

  - `facet_count::Int` — the number of (condition, facet) pairs the projection was built for. A condition list of a
    different shape cannot reuse it. Named for the count rather than for the facets themselves because `facets` is,
    everywhere else on this path, the [`FacetResolver`](@ref) the facets are resolved through.
  - `unknowns::Vector{Vector{Int}}` — per component, the raw dofs the projection solves for: the dofs of that component
    carrying a physical Dirichlet condition and no artificial elimination (`elimination_source === :free`, which
    excludes the overlay boundary as well as covered-mode pruning and the linear dedup).
  - `unsupported::Vector{Vector{Int}}` — per component, the subset of `unknowns` whose row of the boundary mass is
    identically zero: dofs the conditions constrain over a facet support that carries no measure, so the solve assigns
    them the minimum-norm value instead of a fitted one. Empty on every untrimmed boundary. Reported through
    [`diagnostics`](@ref) as [`UnsupportedDirichletDof`](@ref) records; see that docstring for what the package does and
    does not decide about them.
  - `factors::Vector{DirichletFactor{T}}` — per component, the solved boundary mass (see `DirichletFactor`).
  - `samples::Vector{FacetTraceSamples{D,T}}` — one entry per (condition, facet) pair, in the order
    `condition × _facets(condition.boundary)` enumerates them. Empty when no component has an unknown, since then
    nothing is integrated.

# Validity

A projection is valid for the `(layout, space, condition-structure)` triple it was built from. The unknown sets, the
facet count and the mass depend on the mesh, the masks and each condition's boundary selector, field and component —
never on a condition's *value*. Those are exactly the quantities [`update_dirichlet!`](@ref) pins with
`_check_dirichlet_update_compatibility`, which is why an increment may reuse the projection; anything else that moves
goes through [`prepare`](@ref), `move!`, `activate!` or `deactivate!`, each of which builds a fresh dof layout and drops
the cached projection with it.

# Exactness

Reuse is bit-exact, not merely accurate to roundoff. The samples are recorded in the region walk's own order and the
right-hand side is re-accumulated term by term in that order, so a reused projection produces the same floating-point
`b` — and, from the same stored factorisation, the same `c` — as a full rebuild would.
"""
struct DirichletProjection{D,T<:Real}
    facet_count::Int
    unknowns::Vector{Vector{Int}}
    unsupported::Vector{Vector{Int}}
    factors::Vector{DirichletFactor{T}}
    samples::Vector{FacetTraceSamples{D,T}}
end

# True iff `dirichlet` carries at least one nonzero condition. The homogeneous case skips the projection: every
# constrained value already equals zero from the layout's zero-initialised `constrained_values` matrix, and projecting
# zero data walks the boundary and factorises the mass only to write those zeros again: a 3-component 8³ p3 `dof_layout`
# under `SVector` zero data took 369 ms with the projection and 2.8 ms without it (Ryzen 9 9950X).
_needs_dirichlet_projection(dirichlet) = !all(c -> _is_zero_datum(c.value), dirichlet)

# A Dirichlet datum that is zero everywhere: a zero number, or a per-component value (`SVector`, `Tuple`, vector) whose
# every entry is zero. `SVector(0.0, 0.0)` names the same homogeneous condition as `0.0`; any other datum — a function
# above all — is taken to be nonzero, since only evaluating it could say otherwise.
_is_zero_datum(value) = value isa Number ? iszero(value) : value isa Union{Tuple,AbstractArray} && all(iszero, value)

# Refill `layout.constrained_values` from `dirichlet`, returning the [`DirichletProjection`](@ref) used. The caller
# reads the trace solve's verdicts — `_trace_branch` of each factor and the `unsupported` dofs — off the returned
# projection, which is the only place they live, and a caller that re-projects the same layout can hand it back and skip
# the boundary walk. `projection` is reused when given; the facet-count check is a cheap guard for a caller that reaches
# past `update_dirichlet!`'s structural check, which is what actually establishes that a cached projection still matches
# the condition list.
#
# `facets` is the resolver the boundary walk resolves faces through: the model's under `prepare` and
# `update_dirichlet!`, a private one for a standalone `dof_layout`. It must carry the layout's tolerance, checked before
# a projection is built.
function _project_dirichlet_values!(layout::DofLayout{D,T}, V::Space{D,T}, facets::FacetResolver{D,T}, dirichlet,
                                    projection::Union{Nothing,DirichletProjection{D,T}}=nothing) where {D,T}
    ncomp = layout.components
    plan = if projection === nothing || projection.facet_count != _dirichlet_facet_count(dirichlet, Val(D))
        # A tolerance mismatch is a caller bug and is raised as one. A memo's entries were merged against *its*
        # tolerance, so reading them with a layout built at another one would project the datum over a different
        # partition of ∂Ω than the one the layout's constrained dofs were detected on. Quietly substituting a fresh
        # resolver instead would hide that behind a second resolution of every face — exactly the cost sharing the
        # resolver exists to remove.
        facets.tolerance == layout.tolerance ||
            throw(ArgumentError("facet resolver tolerance $(facets.tolerance) does not match " *
                                "the dof layout's $(layout.tolerance); a layout and the " *
                                "resolver it projects through must come from the same prepare"))
        _dirichlet_projection(layout, V, facets, dirichlet)
    else
        projection
    end
    # Only now clear the old values. Everything that can reject the call — the resolver's tolerance above all — has
    # happened, so a rejected call leaves the layout exactly as it found it instead of zeroed and unprojected.
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

# Number of (condition, facet) pairs a Dirichlet list covers. Recorded on a `DirichletProjection` as its `facet_count`
# field so a cached projection can be rejected when the condition list no longer has the shape it was built for.
_dirichlet_facet_count(dirichlet, ::Val{D}) where {D} = sum(c -> length(_facets(c.boundary, Val(D))), dirichlet; init=0)

# Build the datum-independent half of the projection.
#
# The projection is done PER COMPONENT. Two boundary traces couple in the L² projection only through the part of ∂Ω
# where BOTH are constrained in the SAME displacement component; a dof constrained in one component must not enter
# another component's mass matrix. A single shared mass over the union of all component-constrained dofs (integrated
# over every condition's facet, component-blind) is wrong on two counts: two conditions on the same facet (e.g. u_x = 0
# and u_y = ḡ on one edge) double-count the mass and halve the projected value, and a component-1-only condition (e.g. a
# lateral u_x = 0) adds mass to a shared corner's component-2 row with no matching right-hand side, pulling that u_y
# toward zero. Per-component unknown sets, mass matrices, and right-hand sides — each accumulated only over the facets
# where a condition actually constrains that component — remove both. For all-component (vector) conditions this reduces
# to the previous behaviour.
#
# `facets` is the [`FacetResolver`](@ref) the caller projects through. Under a model it is the model's, so the boundary
# walk below reads the *same* resolved faces the assembly path and `boundary_integral` read, and the operator and the
# projected datum can never integrate different facets; a standalone `dof_layout` hands in a private one at the layout's
# tolerance.
function _dirichlet_projection(layout::DofLayout{D,T}, V::Space{D,T}, facets::FacetResolver{D,T}, dirichlet) where {D,T}
    ncomp = layout.components
    facet_count = _dirichlet_facet_count(dirichlet, Val(D))
    unknowns = [[raw
                 for raw in eachindex(layout.raw_keys)
                 if layout.elimination_source[raw] === :free && layout.physical_dirichlet[raw, c]] for c in 1:ncomp]
    # Nothing is constrained anywhere: no mass to accumulate and no facet to walk, so the projection is the empty one
    # its `facet_count` still has to carry.
    if all(isempty, unknowns)
        return DirichletProjection{D,T}(facet_count, unknowns, [Int[] for _ in 1:ncomp],
                                        DirichletFactor{T}[nothing for _ in 1:ncomp], FacetTraceSamples{D,T}[])
    end

    index = [Dict(raw => i for (i, raw) in pairs(unknowns[c])) for c in 1:ncomp]
    # The constrained-boundary mass couples only the physically-Dirichlet dofs of one component and is dense-solved
    # below, so each component's is accumulated directly into a dense matrix — no sparse intermediate.
    mass = [zeros(T, length(unknowns[c]), length(unknowns[c])) for c in 1:ncomp]

    samples = FacetTraceSamples{D,T}[]
    for condition in dirichlet
        for sides in _facets(condition.boundary, Val(D))
            push!(samples, _sample_dirichlet_facet!(mass, index, layout, V, facets, condition, sides))
        end
    end

    # The dofs this projection cannot determine, read off the mass it just accumulated. `M` is the Gram matrix of the
    # boundary traces over the part of ∂Ω these conditions integrate and every facet weight is non-negative, so
    # `M[i,i] = ∫ φᵢ² ds` vanishes exactly when φᵢ vanishes almost everywhere there — and the whole row and column
    # vanish with it. The diagonal is therefore an exact test rather than a thresholded one, and deliberately so: it
    # reads zero only where the rule integrates nothing, which on a trimmed face is structural (an empty rule, or a
    # trace vanishing at every point of one), while a merely *small* diagonal is the conditioning question
    # `min_relative_facet_measure` already answers and not a missing constraint.
    unsupported = [Int[raw for (i, raw) in pairs(unknowns[c]) if iszero(mass[c][i, i])] for c in 1:ncomp]

    # Solve `M c = b` for each component. Cholesky first; pseudoinverse fallback on the rare indefinite case (degenerate
    # / zero-area facets under heavy masking, or a codim-D point pin whose facet carries no measure). Both are stored
    # rather than applied, so every later increment back-substitutes against the same factorisation instead of
    # rebuilding it.
    factors = DirichletFactor{T}[]
    for component in 1:ncomp
        if isempty(unknowns[component])
            push!(factors, nothing)
            continue
        end
        factor = cholesky(Symmetric(mass[component]); check=false)
        push!(factors, issuccess(factor) ? factor : pinv(mass[component]))
    end

    return DirichletProjection{D,T}(facet_count, unknowns, unsupported, factors, samples)
end

# Walk one condition's facet: the boundary regions on that facet × the per-region quadrature points. At each quadrature
# point, evaluate the boundary traces of every covering parent once, record them as a sample, and accumulate the
# mass-matrix contribution of every component the condition constrains.
function _sample_dirichlet_facet!(mass::Vector{Matrix{T}}, index::Vector{Dict{Int,Int}}, layout::DofLayout{D,T},
                                  V::Space{D,T}, facets::FacetResolver{D,T}, condition::DirichletCondition,
                                  sides::Vector{Tuple{Int,Symbol}}) where {D,T}
    ncomp = layout.components
    points = SVector{D,T}[]
    weights = T[]
    offsets = Int[1]
    values = T[]
    rows = [Int[] for _ in 1:ncomp]

    for region in _resolve_face(facets, V, sides)
        # Everything about a parent's trace except the point: the level lookup, the cell's dof list, and the
        # facet-incident subset of its local basis. `region.parents` is constant across the region's quadrature points
        # by the `FacetRegion` contract, and the subset depends only on `(level, sides)`, so all of it is resolved once
        # here instead of at every point below. The levels stay in their own vector: a space may mix basis families
        # across levels, and holding one in the record would make the record type — and with it the mass double loop
        # below, which reads nothing family-specific — vary per parent.
        levels = map(parent -> _level_by_id(V, parent.level), region.parents)
        traces = map(levels, region.parents) do level, parent
            all_raw_dofs = cell_dofs(layout, parent.level, parent.cell)
            (; cell=parent.cell, box=parent.parent_box,
             boundary_trace_indices(level, parent.cell, all_raw_dofs, sides)...)
        end

        # The physical point and weight come from the region accessor every consumer reads (`_region_point`), and each
        # trace is evaluated at that point mapped through its parent's own box, `physical_to_reference(box, x)`, rather
        # than through the parent's frame map: the projected Dirichlet values are a property of the dof layer, and on an
        # uncut face the accessor's point and weight are exactly the tensor rule's, so they stay independent of how the
        # region stores its rule.
        jacobian = _region_jacobian(region)
        for k in 1:_region_qpoint_count(region)
            x, qweight, _ = _region_point(region, k, jacobian)

            # Refresh every parent's trace values at this point. The buffers live on `traces`, so the sample record and
            # the mass double loop below all read the same point from every parent.
            for (i, trace) in pairs(traces)
                boundary_trace_values!(levels[i], trace, physical_to_reference(trace.box, x))
            end

            # Record the sample. Entries are pushed in the same parent-then-mode order the right-hand-side loop below
            # would visit them in, which is what makes a re-accumulated `b` bit-identical.
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

            # Mass: ∫_∂Ω φ_i φ_j dx over this component's constrained dofs. Both indices iterate the same `traces`, so
            # it is symmetric.
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
                                                    _test_value_contribution(mass_channels, test_trace.values[a])
                            end
                        end
                    end
                end
            end
        end
    end

    return FacetTraceSamples{D,T}(points, weights, offsets, values, rows)
end

# True iff `condition` constrains `component`: an unscoped condition constrains every component of its field, a scoped
# one only its own.
function _constrains(condition::DirichletCondition, component::Integer)
    return condition.component === nothing || condition.component == component
end

# Accumulate one facet's contribution to the per-component right-hand sides `b_i = ∫_∂Ω g φ_i dx`. `condition` is
# deliberately left to dispatch: the Dirichlet list is heterogeneous (each condition carries its own value type), so
# this call is the function barrier that specializes the sample loop on the concrete condition and keeps the `g`
# evaluation free of dynamic dispatch.
#
# The `@inbounds` rests on two construction invariants of [`FacetTraceSamples`](@ref): `offsets` never leaves
# `1 : length(values) + 1` and `rows` is as long as `values`, so `e` indexes both; and every nonzero `row` came from the
# component's own unknown index map, so it indexes `rhs_c`.
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

# Evaluate a `DirichletCondition` at physical coordinate `x` for the given component. Handles both constant and callback
# `value` fields.
function _condition_value(condition::DirichletCondition, x, component::Integer=1)
    value = condition.value isa Function ? condition.value(x) : condition.value
    return _component_value(value, component)
end

# Component extraction from a Dirichlet value: scalars apply to every component, anything indexable supplies one entry
# per component.
function _component_value(value, component::Integer)
    value isa Number && return value
    return value[component]
end

# Cheap structural compatibility check. Re-projecting Dirichlet values in place is only valid when the *set* of
# constrained dofs is unchanged; that set depends on the boundary selector, field, and component of each condition, but
# not on its value. Anything else moving means the caller has to go through `prepare(problem)` instead.
function _check_dirichlet_update_compatibility(old, new)
    length(new) == length(old) || throw(ArgumentError("update_dirichlet! expects $(length(old)) " *
                                                      "conditions (the count the model was prepared " *
                                                      "with), got $(length(new))"))
    for (i, (o, n)) in enumerate(zip(old, new))
        o.boundary == n.boundary || throw(ArgumentError("update_dirichlet! condition $i: boundary selector changed; " *
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
