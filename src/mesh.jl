# ── Cartesian mesh ────────────────────────────────────────────────────────────

"""
    CartesianMesh(domain; cells)

Tensor-product Cartesian mesh over the axis-aligned `domain`. `cells` is either a single positive integer used in every
direction or an `NTuple{D,Int}` of per-axis cell counts. The mesh stores the canonical per-axis coordinate vectors
(`length(axes[d]) == cells[d] + 1`) so that cell boxes, parent lookup, and integration-region construction never
recompute the same coordinates.

The mesh discretization is uniform: axis `d` is split into `cells[d]` equal sub-intervals. Non-uniform Cartesian meshes
are not currently supported.

The coordinate type is the domain's. It must be able to hold a fraction of the extent, so a domain with integer corners
is refused with an `ArgumentError`: write `box((0, 0), (1, 1))` as `box((0.0, 0.0), (1.0, 1.0))`.
"""
struct CartesianMesh{D,T<:Real}
    domain::AxisBox{D,T}
    cells::NTuple{D,Int}
    axes::NTuple{D,Vector{T}}
end

# Multi-method dispatcher: coerce a user-supplied "cell count" / "polynomial order" specification into a canonical
# `NTuple{D,Int}`. Accepts a single positive integer (replicated across every axis) or a `D`-tuple of integers (one per
# axis). The fallback method gives a clear error for any other input shape. `name` is the user-visible parameter name
# used in error messages (`:cells`, `:order`, …).
function _axis_int_tuple(n::Integer, ::Val{D}, name::Symbol) where {D}
    n > 0 || throw(ArgumentError("$name entries must be positive"))
    return ntuple(_ -> Int(n), D)
end

function _axis_int_tuple(t::NTuple{D,<:Integer}, ::Val{D}, name::Symbol) where {D}
    all(>(0), t) || throw(ArgumentError("$name entries must be positive"))
    return Int.(t)
end

function _axis_int_tuple(_, ::Val{D}, name::Symbol) where {D}
    throw(ArgumentError("$name must be a positive integer or an NTuple{$D,Int}"))
end

# Per-axis coordinate samples for a uniform Cartesian mesh: axis `d` carries `cells[d] + 1` values evenly spaced from
# `lo = domain.lower[d]` to `hi = domain.upper[d]`,
#
#     xᵢ = lo + (hi − lo) · fl(i / n)   for i < n,     xₙ = hi,
#
# with `fl(i / n)` the correctly rounded quotient. Two bit-level properties hang on evaluating it in exactly that way.
#
#   * The endpoints are the domain's own corners. `x₀` is `lo` because the quotient is zero, and `xₙ` is `hi` by
#     definition rather than by arithmetic: the sum `lo + (hi − lo)` rounds, and on `[0, 0.7]` in three cells the
#     scale-first order below put the last node one ulp short of `0.7`.
#   * Meshes of the same domain whose cell counts are multiples of each other share their nodes. A node depends on `i`
#     and `n` only through `fl(i / n)`, and the correct rounding of a rational does not depend on how it is written, so
#     node `j·s` of an `n·s`-cell axis is node `j` of an `n`-cell axis, bit for bit. A `ladder` is built on that: a
#     node of one level either is a node of a finer level or lies a whole cell away from every one, so its cross-level
#     cell map (`_cell_block`, `ladder.jl`) is exact whatever rounding margin it reads with. Scaling first,
#     `(hi − lo)·i / n`, rounds a different product `(hi − lo)·i` on each level before dividing, and the levels part by
#     an ulp: on `[0, 0.7]` in three cells refined by three, an exact map gave the last coarse cell three fine cells
#     and the other two four each.
#
# Where `hi − lo` is exactly a power of two, the scale-first order gave the same nodes bit for bit, so the dyadic
# domains of most tests and examples are unchanged by the switch; a power-of-two cell count keeps every interior node
# too. Elsewhere a node can move by an ulp, and it is then no longer the correctly rounded value of `lo + (hi − lo)·i/n`
# even on an integer extent — `[0, 3]` in five cells puts the second node at `0.6000000000000001` — which is within
# rounding of the exact coordinate and is the price of nesting.
function _mesh_axes(domain::AxisBox{D,T}, cells::NTuple{D,Int}) where {D,T}
    return ntuple(D) do d
        lo = domain.lower[d]
        hi = domain.upper[d]
        n = cells[d]
        [i == n ? hi : lo + (hi - lo) * (convert(T, i) / convert(T, n)) for i in 0:n]
    end
end

function CartesianMesh(domain::AxisBox{D,T}; cells) where {D,T}
    # Interior nodes are fractions of the extent, which an integer type cannot hold: `box((0, 0), (1, 1))` used to fail
    # deep inside `_mesh_axes` with an `InexactError` at two cells, and to build at one cell a mesh no tolerance could
    # be formed for (`GeometryTolerance(Int)` has no `eps`). Refuse it here, where the fix can be named.
    T <: Integer && throw(ArgumentError("a mesh needs floating-point coordinates, but the domain's corners are " *
                                        "$T; write box$((Tuple(domain.lower), Tuple(domain.upper))) as " *
                                        "box$((Tuple(float.(domain.lower)), Tuple(float.(domain.upper))))"))
    cell_tuple = _axis_int_tuple(cells, Val(D), :cells)
    return CartesianMesh{D,T}(domain, cell_tuple, _mesh_axes(domain, cell_tuple))
end

"""
    mesh(domain::AxisBox; cells) -> CartesianMesh

Public-API thin wrapper over the [`CartesianMesh`](@ref) constructor. Use it when you want a standalone mesh; for
building a discretization for a problem, prefer [`space`](@ref).
"""
mesh(domain::AxisBox; cells) = CartesianMesh(domain; cells)

"""
    cell_count(m::CartesianMesh) -> Int

Total number of cells in the mesh, `prod(m.cells)`.
"""
cell_count(m::CartesianMesh) = prod(m.cells)

"""
    cell_indices(m::CartesianMesh) -> CartesianIndices{D}

Iterator over the `D`-dimensional cell index space of `m`. Equivalent to `CartesianIndices(m.cells)` but routed through
this accessor so callers do not reach into the struct.
"""
cell_indices(m::CartesianMesh{D}) where {D} = CartesianIndices(m.cells)

"""
    boundary_coordinates(m::CartesianMesh) -> NTuple{D,Vector{T}}

Per-axis element-boundary coordinates of the mesh: `axes[d]` carries the `cells[d] + 1` coordinates separating the cells
along axis `d`. Used by `intersections.jl` to collect every participating mesh's boundary coordinates before merging
them into the admissible-box partition.
"""
boundary_coordinates(m::CartesianMesh) = m.axes

"""
    cell_box(m::CartesianMesh, index::CartesianIndex) -> AxisBox

Axis-aligned box for the cell at the given multi-index. Returns the cell in the mesh's physical frame; combine with
[`physical_to_reference`](@ref) and [`reference_to_physical`](@ref) to map between the cell, its parent level's
reference frame, and a region's reference frame. Throws `BoundsError` if `index` is outside `cell_indices(m)`.
"""
function cell_box(m::CartesianMesh{D,T}, index::CartesianIndex{D}) where {D,T}
    # Base's own refusal, against the cell index space. A `BoundsError(m, index)` built by hand carries the bare
    # `CartesianIndex` as its index, and `showerror` iterates that field, which a `CartesianIndex` refuses: the error
    # could be thrown but not printed.
    checkbounds(cell_indices(m), index)
    lower = SVector{D,T}(ntuple(d -> m.axes[d][index.I[d]], D))
    upper = SVector{D,T}(ntuple(d -> m.axes[d][index.I[d] + 1], D))
    return AxisBox{D,T}(lower, upper)
end

"""
    locate_cell(m::CartesianMesh, point; tol=GeometryTolerance(T)) -> Union{CartesianIndex,Nothing}

Find the cell of `m` containing `point`, or `nothing` if `point` is outside the mesh's `domain` (up to `tol.contain`).
For a point lying exactly on an interior element boundary the convention is "upper cell wins": the cell whose lower face
is at the boundary — the larger index along that axis — is returned, because `searchsortedlast` maps a coordinate
sitting on node `axis[i]` to cell `i`. Points right at the upper boundary of an axis are snapped to the last cell along
that axis so a point sitting exactly on `domain.upper[d]` still returns a valid cell.

Used by the coverage signature of the admissible-box partition (`_signature` in `intersections.jl`) and by
post-processing evaluation (`_level_field`).
"""
function locate_cell(m::CartesianMesh{D,T}, point::PointLike{D}; tol=GeometryTolerance(T)) where {D,T}
    contains_point(point, m.domain, tol) || return nothing

    ids = ntuple(D) do d
        axis = m.axes[d]
        x = convert(T, point[d])
        if x >= last(axis) - tol.contain
            length(axis) - 1
        else
            clamp(searchsortedlast(axis, x), 1, length(axis) - 1)
        end
    end

    return CartesianIndex(ids)
end

# Index range of the cells of `axis` that the coordinate interval `[lo, hi]` overlaps by a positive length; empty
# (`1:0`) when the interval misses the axis entirely. The point form of the same lookup is `locate_cell` above.
#
# Exact by construction, and the two searches are deliberately different: `searchsortedlast` puts a lower endpoint
# sitting exactly on a node into the cell *above* it, and `searchsortedfirst` excludes the cell an upper endpoint merely
# touches. A block that shares only a face with the interval is therefore not claimed, which is what every caller means
# by "overlaps".
#
# Whether the endpoints need a tolerance is the caller's decision, not this function's, and the two in-tree callers
# answer it differently for reasons that belong to them:
#
#   * the coverage rule nudges a box's faces inward by the absolute `tol.contain` (`_cells_under_box`, `coverage.jl`):
#     an `overlay` box whose axes were built by `_mesh_axes` from its own corners does not reproduce a coarser
#     level's node coordinates bit-for-bit;
#   * the cross-level cell map nudges a cell's faces inward by a few ulps of the two axes' own magnitude
#     (`_axis_overlap`, behind `overlapping_cells` and `_cell_block`, `ladder.jl`). That margin is scale-free: it
#     absorbs the same rounding at any offset and extent, and changes nothing on a ladder, whose levels share their node
#     coordinates bit-for-bit. `GeometryTolerance` would be wrong twice over there — it is absolute, so on a domain far
#     from the origin it is below the coordinates' own ulp, and on a fine level it is a sizeable fraction of a cell.
function _axis_cells(axis::Vector{T}, lo::T, hi::T) where {T}
    ncells = length(axis) - 1
    i = searchsortedlast(axis, lo)
    j = searchsortedfirst(axis, hi) - 1
    (j < 1 || i > ncells) && return 1:0                     # disjoint from this axis
    return max(i, 1):min(j, ncells)
end

# ── Per-cell activation mask ──────────────────────────────────────────────────

"""
    LevelMask{D}(on::BitArray{D})

Per-cell activation mask for a level. `on[cell]` is `true` when the cell participates in assembly and dof enumeration.
Carried as an optional field on [`Level`](@ref); a `nothing` mask means "every cell active" and keeps the no-mask hot
path type-stable and allocation-free.

The mask interacts with the dof layer like a geometric overlay boundary: faces between active and inactive cells of the
*same* level are treated as artificial overlay constraints (analogous to faces between the level and its physical-domain
complement). Span-mode dofs whose support is entirely inside inactive cells are not enumerated.

One exception, and it is a physical one: when the inactive cell across such a face holds no material — the classifier
finds its box *fully fictitious*, entirely outside `Ω` — the face carries no physical material and the modes on it stay
active: they vanish on every physical face anyway and instead carry the adjacent active cell's approximation — a cut
cell's, up to `∂Ω`, whenever the fold runs. Every cell the finite-cell fold switched off is such a cell. A cell the
*user* masked out constrains when it holds material, which is the case unless it lies entirely outside `Ω` (always,
without a [`PhysicalDomain`](@ref)). See `_has_overlay_constraint` in `src/dofs.jl`.
"""
struct LevelMask{D}
    on::BitArray{D}
end

"""
    is_active(mask, cell) -> Bool

Whether `cell` is active under the level's mask. `nothing` (the default mask) treats every cell as active.
"""
is_active(::Nothing, ::CartesianIndex) = true
is_active(mask::LevelMask{D}, cell::CartesianIndex{D}) where {D} = mask.on[cell]

# Fold an all-active bit field back into the no-mask default it is. `nothing` is not merely a shorthand:
# `is_active(::Nothing, _)` is a compile-time `true`, `_covered_by_level` takes an O(1) box-containment answer instead
# of scanning the block under every coarse cell, and `_coverage_cells` returns `trues` instead of dilating the active
# front. Routing every construction through this one function is what makes the invariant "a `LevelMask` always excludes
# at least one cell" hold for `space`, `overlay`, `adapt` and the mutators alike, rather than for the mutators alone.
_collapse_mask(on::BitArray{D}) where {D} = all(on) ? nothing : LevelMask{D}(on)

# A cell selection, as a fresh `BitArray` over `mesh`'s cells, from every shape the `active =` keyword documents:
#
#   * `nothing` — every cell, the reading `nothing` has in `active =`, `dilate`, `ladder` and the mutators alike;
#   * a `LevelMask{D}`, or an `AbstractArray{Bool,D}` shaped like the cell grid — copied, so a later mutation of the
#     argument cannot reach a level;
#   * a predicate `(cell_box, cell_index) -> Bool` — evaluated once per cell;
#   * an iterable of `CartesianIndex{D}` — the listed cells, each type- and bounds-checked.
#
# This is the one definition of what a selection is. `_normalize_mask` (`active =`, `adapt`), the mutators `activate!`
# / `deactivate!`, `dilate` / `support_extension` and the ladder's `overlapping_cells` all read it, so a shape one of
# them accepts every one accepts, with the same refusals. Each caller used to carry its own copy of this dispatch, and
# the copies drifted: the mutators' copy accepted neither `nothing` nor a `LevelMask`. The result is always a new array
# the caller may keep or mutate.
_selection(::Nothing, mesh::CartesianMesh) = trues(mesh.cells)

_selection(mask::LevelMask, mesh::CartesianMesh) = _selection(mask.on, mesh)

function _selection(bits::AbstractArray{Bool,D}, mesh::CartesianMesh{D}) where {D}
    size(bits) == mesh.cells ||
        throw(DimensionMismatch("selector shape $(size(bits)) does not match mesh cells $(mesh.cells)"))
    on = BitArray(undef, mesh.cells)
    copyto!(on, bits)
    return on
end

function _selection(f::Function, mesh::CartesianMesh{D}) where {D}
    on = BitArray(undef, mesh.cells)
    for ci in cell_indices(mesh)
        on[ci] = f(cell_box(mesh, ci), ci)::Bool
    end
    return on
end

# The fallback consumes any other iterable, and is also the type-check error path: a wrong-typed selection still gets a
# clear message rather than a `MethodError` from deep inside.
function _selection(cells, mesh::CartesianMesh{D}) where {D}
    on = falses(mesh.cells)
    for ci in cells
        ci isa CartesianIndex{D} ||
            throw(ArgumentError("cells iterable entries must be CartesianIndex{$D}; got $(typeof(ci))"))
        checkbounds(Bool, on, ci) ||
            throw(ArgumentError("cell index $ci is out of bounds for mesh cells $(mesh.cells)"))
        on[ci] = true
    end
    return on
end

# A level's stored mask for an `active =` selection: the selection, collapsed to the no-mask default when it names
# every cell, so that holds whichever shape the selection arrived in. `nothing` is answered without building the
# all-true array it would collapse from.
_normalize_mask(::Nothing, ::CartesianMesh) = nothing
_normalize_mask(cells, mesh::CartesianMesh) = _collapse_mask(_selection(cells, mesh))

# ── Per-cell polynomial order ─────────────────────────────────────────────────

"""
    CellOrders{D}

The per-cell polynomial order of a [`Level`](@ref), as a palette of the distinct orders plus one palette index per cell.

A level whose order is one tuple for every cell carries a palette of one entry — the same representation, so the uniform
case has no second code path and the lookup stays allocation-free. Fields:

  - `palette::Vector{NTuple{D,Int}}` — the distinct per-axis orders occurring on the level, in first-occurrence order.
    Stored as a palette rather than a dense `Array{NTuple{D,Int},D}` because a graded p-field carries a handful of
    distinct orders however many cells it has, and the palette doubles as the list a per-order validation walks.
  - `class::Array{UInt16,D}` — two bytes per cell, indexing `palette`. An anisotropic hp estimator reaches 255 distinct
    orders on a modest 3D grid (an 8³ grid admits 512), so the index is wide enough that the cap is not a shape a real
    p-field runs into; the normaliser still rejects rather than wraps.
  - `nominal::NTuple{D,Int}` — the per-axis maximum over `palette`, computed once at construction and read through
    [`nominal_order`](@ref). It sizes buffers and 1D factor tables, where over-sizing is safe; it is not the question
    anything that integrates, evaluates or subdivides should ask. That one is [`cell_order`](@ref).

This record is what the caller asked for, and it depends on nothing else — not on the mask, not on the basis mode. The
index sets those orders actually generate, once the minimum rule has trimmed the modes a shared entity cannot carry,
live in [`CellModes`](@ref), which `Level`'s inner constructor derives from this record together with the mesh, the
family (which carries the basis mode) and the mask. `CellModes` is what makes the space C⁰, and it is the table every
positional pairing of raw dofs with basis values must read; go through [`cell_basis_indices`](@ref) rather than
re-deriving the set.
"""
struct CellOrders{D}
    palette::Vector{NTuple{D,Int}}
    class::Array{UInt16,D}
    nominal::NTuple{D,Int}
end

"""
    CellModes{D}

The multi-index sets a level's cells actually generate, after the minimum rule has trimmed the modes a shared entity
cannot carry. Two fields, the same shape as [`CellOrders`](@ref): `sets` holds the distinct index sets (entry 1 is the
empty set every inactive cell points at) and `kind` maps a cell to one of them.

**This is derived, never supplied.** It is a function of the level's orders, its mesh, its basis family (which carries
the basis mode) and its activation mask, and it is computed by `Level`'s inner constructor from exactly those. That is
not a stylistic choice: the mask half of that dependency is what a caller used to be able to get wrong, by handing in a
table built against one mask alongside a different one, and the failure was silent — the dof layout is built *from* this
table, so layout and table went stale together and nothing downstream could notice. A table that cannot be passed in
cannot be passed in stale.
"""
struct CellModes{D}
    sets::Vector{Vector{CartesianIndex{D}}}
    kind::Array{UInt16,D}
end

# One shared index set for every active cell, and the empty set for the rest. This serves two cases that are the same
# case: a family with no per-cell order, and a level whose palette has a single entry. Neither can lose a mode to the
# minimum rule — every active cell asks the same question of neighbours carrying the same order — so the filter would be
# a no-op, and a uniform level pays nothing for the existence of the graded path.
function _uniform_cell_modes(basis::BasisFamily, orders::CellOrders{D}, mesh::CartesianMesh{D}, mask) where {D}
    sets = [CartesianIndex{D}[], local_basis_indices(basis, orders.palette[1])]
    kind = Array{UInt16,D}(undef, mesh.cells)
    for cell in cell_indices(mesh)
        kind[cell] = is_active(mask, cell) ? UInt16(2) : UInt16(1)
    end
    return CellModes{D}(sets, kind)
end

# The multi-index set each cell of a level generates — the minimum rule — as a basis-family hook. This generic method is
# the whole interface: it is what `Level`'s constructor below calls, and it is bound here so the include order stays the
# dependency map.
#
# A family that declares `_supports_cell_order` supplies its own method where its dof-key incidence machinery lives,
# exactly as a family supplies its own burial hooks for `_coverage_constraints!`: integrated Legendre's is in `dofs.jl`,
# next to `_entity_carries` and the `TensorDofKey` factors it is expressed over, and the B-spline family's would live in
# its extension. A family that has not supplied one may only be handed a single-entry palette, and the refusal below is
# loud on purpose. Falling through to `_uniform_cell_modes` with a real palette would assemble every cell at
# `palette[1]` while `cell_order` reported the field the caller asked for — a non-conforming space with no error
# anywhere, which is the exact failure the minimum rule exists to prevent.
function _cell_modes(basis::BasisFamily, orders::CellOrders{D}, mesh::CartesianMesh{D}, mask) where {D}
    length(orders.palette) == 1 ||
        throw(ArgumentError("the $(basis_name(basis)) basis family answers _supports_cell_order " *
                            "but supplies no _cell_modes method: the minimum rule for a graded " *
                            "level is the family's to state, and the uniform fallback would " *
                            "silently assemble every cell at $(orders.palette[1])."))
    return _uniform_cell_modes(basis, orders, mesh, mask)
end

# Translate a user-supplied `order` specification into either an `NTuple{D,Int}` (uniform) or a flat
# `Vector{NTuple{D,Int}}` of per-cell orders in `cell_indices` order. Both are intermediate forms: `_build_cell_orders`
# folds the first into a palette of one entry and the second into a palette of its distinct entries, so neither shape
# survives into a `Level`. Accepted shapes, dispatched on type and deliberately parallel to `_normalize_mask`:
#
#   * `Integer`                                 — isotropic, uniform.
#   * `NTuple{D,Integer}`                       — anisotropic, uniform.
#   * `AbstractArray{<:Integer,D}` of cell shape — isotropic per cell.
#   * `AbstractArray{<:NTuple{D,Integer},D}`    — anisotropic per cell.
#   * predicate `(cell_box, cell_index) -> Integer | NTuple{D,Integer}`.
#
# A per-cell array whose entries are all equal collapses back to the uniform tuple, so a nominally graded field that
# happens to be flat costs nothing — the same discipline `_collapse_mask` applies to an all-active mask.
_normalize_order(n::Integer, ::CartesianMesh{D}) where {D} = _axis_int_tuple(n, Val(D), :order)

_normalize_order(t::NTuple{D,<:Integer}, ::CartesianMesh{D}) where {D} = _axis_int_tuple(t, Val(D), :order)

function _normalize_order(a::AbstractArray{<:Integer,D}, mesh::CartesianMesh{D}) where {D}
    size(a) == mesh.cells || throw(DimensionMismatch("order shape $(size(a)) does not match mesh cells $(mesh.cells)"))
    return _collapse_order(vec([_axis_int_tuple(a[ci], Val(D), :order) for ci in cell_indices(mesh)]))
end

function _normalize_order(a::AbstractArray{<:NTuple{D,<:Integer},D}, mesh::CartesianMesh{D}) where {D}
    size(a) == mesh.cells || throw(DimensionMismatch("order shape $(size(a)) does not match mesh cells $(mesh.cells)"))
    return _collapse_order(vec([_axis_int_tuple(a[ci], Val(D), :order) for ci in cell_indices(mesh)]))
end

function _normalize_order(f::Function, mesh::CartesianMesh{D}) where {D}
    entries = vec([_axis_int_tuple(f(cell_box(mesh, ci), ci), Val(D), :order) for ci in cell_indices(mesh)])
    return _collapse_order(entries)
end

function _normalize_order(_, ::CartesianMesh{D}) where {D}
    throw(ArgumentError("order must be a positive integer, an NTuple{$D,Int}, an array of either " *
                        "shaped like the level's cell grid, or a predicate " *
                        "(cell_box, cell_index) -> order"))
end

# Fold a flat list of per-cell orders back into the level's cell grid, collapsing an everywhere-equal field to the
# uniform tuple it is.
function _collapse_order(entries::Vector{NTuple{D,Int}}) where {D}
    all(==(first(entries)), entries) && return first(entries)
    return entries
end

# The level's *nominal* order: the uniform order, or the per-axis maximum over a per-cell field. It is the `nominal`
# field of `CellOrders` — there is no `Level.order` any more — and `nominal_order` is its public read.
#
# Sizing a quadrature rule from it is always SAFE, because it bounds every cell of the level from above and
# over-integrating costs points rather than accuracy. Safe is not the same as right, though, and the consumers have
# since split on that: `_axis_max` (tensor counts and moment orders, volume, facet and immersed simplex) and
# `_subdivision_counts` each size from the parent CELL's own order, so one loud cell no longer sets the rule for the
# level.
_nominal_order(o::NTuple{D,Int}) where {D} = o

function _nominal_order(entries::Vector{NTuple{D,Int}}) where {D}
    out = first(entries)
    for e in entries
        out = ntuple(d -> max(out[d], e[d]), D)
    end
    return out
end

# Build the `CellOrders` record a level carries. Both methods run the same per-palette-entry validation —
# `local_basis_indices(family, o)`, which raises on an order the family, in its mode, rejects — so the per-*cell*
# `order ≥ 1` requirement and the `:trunk` isotropy requirement are enforced at the granularity a graded level needs,
# where a per-level check can no longer see a single offending cell. The graded method adds two checks a uniform level
# gets for free: the family must declare `_supports_cell_order`, and the palette must fit the `UInt16` class index.
#
# This record is what the caller ASKED FOR, and it depends on nothing else — not on the mask, not on the mode. What the
# basis then emits per cell is derived from it by `_cell_modes`, whose generic method is defined above and whose
# per-family methods live where each family's key-incidence machinery does.
function _build_cell_orders(family::BasisFamily, order::NTuple{D,Int}, mesh::CartesianMesh{D}) where {D}
    local_basis_indices(family, order)    # raises on an order the family rejects
    return CellOrders{D}([order], fill(UInt16(1), mesh.cells), order)
end

function _build_cell_orders(family::BasisFamily, entries::Vector{NTuple{D,Int}}, mesh::CartesianMesh{D}) where {D}
    _supports_cell_order(family) ||
        throw(ArgumentError("the $(basis_name(family)) basis family does not support a per-cell " *
                            "polynomial order: its degree belongs to a whole-axis knot vector, not " *
                            "to a cell, so there is no set of basis functions a per-cell order " *
                            "could name and no shared entity for the minimum rule to act on. Use " *
                            "the default integrated-Legendre basis, or pass one order for the level."))
    palette = NTuple{D,Int}[]
    # A dictionary rather than a scan over `palette`: the class index is `UInt16` because an anisotropic hp field
    # reaches hundreds of distinct orders on a modest grid, and `findfirst` over the palette made the build O(cells ×
    # palette) — measured on a 512² field carrying 500 distinct orders, 284 ms against 174 ms. It is also the idiom
    # `_cell_modes` already uses for its own dedup, so the two palette builders now read the same way.
    slot = Dict{NTuple{D,Int},UInt16}()
    class = Array{UInt16,D}(undef, mesh.cells)
    for (i, ci) in enumerate(cell_indices(mesh))
        o = entries[i]
        class[ci] = get!(slot, o) do
            length(palette) == typemax(UInt16) &&
                throw(ArgumentError("a level carries at most $(typemax(UInt16)) distinct per-cell " *
                                    "orders, and this one asks for more. That is a palette index " *
                                    "limit, not a statement about your p-field."))
            push!(palette, o)
            UInt16(length(palette))
        end
    end
    for o in palette
        local_basis_indices(family, o)     # raises on an order the family rejects
    end
    return CellOrders{D}(palette, class, _nominal_order(palette))
end

# ── Levels and superposition spaces ───────────────────────────────────────────

"""
    Level

One base or overlay discretization level of a [`Space`](@ref). Carries the level's mesh, basis family (which carries
the basis mode), polynomial-order metadata and an optional activation mask. Fields:

  - `id::Int` — level identifier, unique within a `Space`. `1` for the base level, `2, 3, …` for overlays in the order
    they were added. In a coupled multi-domain model every subdomain's ids are shifted into a disjoint block by
    `_reindex_space_levels`, so only the *ordering* of the ids is guaranteed there, not the value `1` for a base level.
  - `role::Symbol` — `:base` or `:overlay`. The base level covers the full physical domain; overlay levels live inside
    it.
  - `mesh::CartesianMesh{D,T}` — the Cartesian mesh that discretizes the level's domain.
  - `basis::B` — basis family (see [`BasisFamily`](@ref)). All cells of the level share the same family. The family
    value is the whole index-set rule: it carries the basis mode the level was built with (`:tensor`, the full tensor
    product, or integrated Legendre's `:trunk`, the Szabó–Babuška trunk space filtered by trunk degree), which
    `_mode(level.basis)` reads back.
  - `orders::CellOrders{D}` — the per-cell polynomial order, as a palette of the distinct orders plus one index per
    cell. A level whose order is uniform carries a palette of one entry, so there is no second representation and no
    second code path for the common case. Ask [`cell_order`](@ref) for a cell's order and [`nominal_order`](@ref) for
    the per-axis maximum, which is a sizing quantity only. Both take a [`Space`](@ref) and a `level` keyword as well,
    which is how a caller who is not inside the package should reach them.
  - `modes::CellModes{D}` — the multi-index sets those orders actually generate, after the minimum rule. **Derived**,
    never supplied: see [`CellModes`](@ref) for why that is the point rather than a detail.
  - `mask::Union{Nothing,LevelMask{D}}` — optional per-cell activation mask. `nothing` keeps every cell active and is
    the type-stable no-mask hot path; the `Union` is small so the `Level` type stays stable across `activate!` /
    `deactivate!` transitions between the masked and unmasked states. An `active =` selection that happens to name every
    cell is stored as `nothing`, so the fast path survives a round trip through [`adapt`](@ref) as well as through the
    mutators.

Leaf semantics — which modes a covered level sheds — are a property of the whole superposition stack rather than of one
level, so they are applied once, by `dof_layout`, over the whole stack. See leaf semantics under [`space`](@ref), and
[`prepare`](@ref)'s `prune` keyword for the unreduced twin the rule is measured against.
"""
struct Level{D,T<:Real,B<:BasisFamily}
    id::Int
    role::Symbol
    mesh::CartesianMesh{D,T}
    basis::B
    orders::CellOrders{D}
    modes::CellModes{D}
    mask::Union{Nothing,LevelMask{D}}

    # `modes` is DERIVED, and deriving it here is the whole point. It depends on the orders, the mesh, the family (with
    # its mode) and the mask; when it was a constructor argument a caller could hand in a table built against one mask
    # alongside a different one, and the failure was silent — `_build_cell_dofs!` builds the dof layout *from* the
    # table, so layout and table went stale together and nothing downstream could see it. Measured, a fold carrying its
    # pre-fold table dropped 24 of 780 unknowns while reporting `symmetry_residual = 0.0` and a 1e-16 residual. That
    # mismatch used to be guarded by a witness field recording which mask the table was built against; a guard is a
    # confession that the invariant can be violated. Computing the table from the arguments that determine it makes the
    # mismatch unspellable instead, and the construction sites — which now all go through `_level` / `_new_level` below
    # — stop having to remember anything.
    function Level{D,T,B}(id, role, mesh, basis, orders, mask) where {D,T<:Real,B<:BasisFamily}
        return new{D,T,B}(id, role, mesh, basis, orders, _cell_modes(basis, orders, mesh, mask), mask)
    end
end

"""
    nominal_order(V::Space; level) -> NTuple{D,Int}
    nominal_order(level::Level) -> NTuple{D,Int}

The per-axis **maximum** order over the level's cells — a sizing quantity, and nothing else.

It is spelled out rather than stored under the name `order` because the two are not interchangeable and the difference
is invisible at the call site. Reading a level-wide maximum where a per-cell order was meant is a defect this package
has shipped three times: the immersed-surface rule sized by the loudest cell anywhere (measured 8.34× on `assemble!`),
`write_vtk` subdividing every region at the maximum (200× in time, 91× on disk), and an error estimator enriching the
whole domain to one above the hottest cell (3.6× end to end). Each was found by measurement rather than by the compiler,
because the read looked correct.

Use it for buffer lengths and 1D factor tables, where over-sizing is safe. For anything that integrates, evaluates or
subdivides, the question you want is [`cell_order`](@ref).

The `Space` form takes the level by keyword, as every other space accessor does; the `Level` form is the one the
package's own kernels call once they have resolved the level.
"""
nominal_order(level::Level) = level.orders.nominal

"""
    cell_order(V::Space, cell::CartesianIndex; level) -> NTuple{D,Int}
    cell_order(level::Level, cell::CartesianIndex) -> NTuple{D,Int}

Per-axis polynomial order of one cell of `level`. A level whose order is uniform carries a palette of one entry, so this
is the same lookup either way and there is no second code path for the common case.

The `Space` form is the public spelling — a marking loop that already asks [`cell_box`](@ref)`(V, ci; level)` and
[`cell_orders`](@ref)`(V; level)` asks this the same way, without reaching into `V.levels`. It costs one bounds check on
the level index over the `Level` form, which is the one the kernels use.
"""
@inline function cell_order(level::Level{D}, cell::CartesianIndex{D}) where {D}
    return level.orders.palette[level.orders.class[cell]]
end

"""
    cell_basis_indices(level::Level, cell::CartesianIndex) -> Vector{CartesianIndex{D}}

Multi-indices of the local basis functions cell `cell` of `level` generates, in the canonical
[`local_basis_indices`](@ref) order. An inactive cell generates none and gets the empty set.

The answer is a lookup into the level's minimum-rule table ([`CellModes`](@ref)) on every level, and allocates
nothing. On a uniform level every active cell shares one set, the level's whole index set
`local_basis_indices(level.basis, nominal_order(level))`. On a level carrying a per-cell order the set is
*not* the index set at the cell's own order: the modes a lower-order neighbour cannot match are gone, which is what
keeps the space C⁰ across an order jump.

The result is the level's own vector — the same object for every cell with the same set, and the one the dof layout and
the assembly banks pair raw dofs against — so treat it as read-only and `copy` it before modifying it. Mutating it
corrupts that table for every cell that shares it.

This is the one place the per-cell index set is defined. Every consumer that pairs `cell_dofs(layout, level, cell)` with
basis values positionally must obtain its multi-indices here, or from a bank filled from `level.modes`.
"""
@inline function cell_basis_indices(level::Level{D}, cell::CartesianIndex{D}) where {D}
    return level.modes.sets[level.modes.kind[cell]]
end

"""
    basis_values(level::Level, cell::CartesianIndex, xi) -> Vector

Values of exactly the functions cell `cell` of `level` generates, at reference point `xi ∈ [−1, 1]ᴰ`, in
[`cell_basis_indices`](@ref) order, in the floating-point promotion of `xi`'s coordinate type.

This is the package's per-cell point evaluator, and the one to reach for whenever the result will be paired with
`cell_dofs`: it reads the level's own minimum-rule table, so the values pair positionally with
`cell_dofs(layout, level.id, cell)` on a uniform and a graded level alike. An index set derived from an order instead
would be wrong on a level whose order varies from cell to cell, where the minimum rule has removed the shared-entity
modes a lower-order neighbour cannot match and what is left is in general the index set of no order at all.

`cell` is the parent cell's mesh index as well as the key into that table. Integrated Legendre's 1D modes are cell-local
in the reference frame and ignore it; the B-spline extension's are global and pick their knot-vector span from it. The
cell is therefore looked up on the level's own grid, and an index outside it raises `BoundsError`. The evaluator reaches
the family through its `_fill_factor_tables!` hook alone, so it serves every family unchanged. To evaluate on the
reference cell itself, use a one-cell level over `[−1, 1]ᴰ`, e.g.
`space(box((-1.0, -1.0), (1.0, 1.0)); cells=1, order=p).levels[1]`.

Allocates a fresh vector per call. The assembly hot loops go through the workspace banks instead and allocate nothing.
"""
function basis_values(level::Level{D}, cell::CartesianIndex{D}, xi::PointLike{D}) where {D}
    return _cell_basis(level, cell, xi, float(promote_type(map(typeof, Tuple(xi))...)), nothing)
end

"""
    physical_basis_gradients(level::Level, cell::CartesianIndex, cell_box, xi) -> Vector{SVector{D}}

Physical gradients of exactly the functions cell `cell` of `level` generates, on the axis-aligned `cell_box`, at
reference point `xi ∈ [−1, 1]ᴰ`, in [`cell_basis_indices`](@ref) order — the gradient counterpart of
[`basis_values`](@ref)`(level, cell, xi)`, with the same pairing with `cell_dofs` and the same contract on `cell`.

The chain rule for an axis-aligned cell with edge lengths `h_d = edge_lengths(cell_box)[d]` gives

    ∂N_α/∂x_d = (2 / h_d) · ∂N_α/∂ξ_d,

so the physical gradient is the reference gradient scaled per axis by `2 / h_d`; on a cell of edge length 2 per axis it
is the bare reference gradient. `cell_box` is the cell's *geometry* and `cell` its *mesh index*, independent arguments
because the chain rule needs the former while a span-based family needs the latter: pass
[`cell_box`](@ref)`(level.mesh, cell)` for the cell's own box, and map a physical point through
[`physical_to_reference`](@ref) first if you have `x` rather than `ξ`. The result is in the promotion of `xi`'s
coordinate type with the box's, so a `Float32` point on a `Float64` cell computes and stores the chain rule at the wider
precision.
"""
function physical_basis_gradients(level::Level{D}, cell::CartesianIndex{D}, cell_box::AxisBox{D,T},
                                  xi::PointLike{D}) where {D,T}
    return _cell_basis(level, cell, xi, float(promote_type(map(typeof, Tuple(xi))..., T)), cell_box)
end

# The one body of the two per-cell evaluators: the cell's own index set and order from the level, fresh buffers in the
# working precision `R`, and one kernel call — values alone without a box, values and physical gradients (the chain-rule
# scale `2 / h_d` of `cell_box`) with one. Each public method states its own precision in its signature, because the
# two differ: values in the promotion of the point's coordinate types, gradients in its promotion with the box's.
function _cell_basis(level::Level{D}, cell::CartesianIndex{D}, xi::PointLike{D}, ::Type{R}, cell_box) where {D,R}
    indices = cell_basis_indices(level, cell)
    p = cell_order(level, cell)
    values = Vector{R}(undef, length(indices))
    grad = cell_box === nothing ? nothing :
           (Vector{SVector{D,R}}(undef, length(indices)), SVector{D,R}(2 ./ edge_lengths(cell_box)),
            _factor_buffers(p, R))
    _tensor_values!(level.basis, values, indices, p, SVector{D,R}(xi), _factor_buffers(p, R), cell, grad)
    return grad === nothing ? values : grad[1]
end

# The per-cell minimum-rule table of a level: one `CellModes` always, a palette of one set on a uniform level. Assembly
# and transfer workspaces bank it per level id so a hot loop reads `sets[kind[cell]]` — two array reads, no branch and
# no per-cell derivation — and so that the table they pair raw dofs against is literally the one `_build_cell_dofs!`
# walked.
_cell_locals(level::Level) = level.modes

"""
    Space

Superposition space over one base level and zero or more overlay levels. Built incrementally via [`space`](@ref) and
[`overlay`](@ref).

The approximation it represents is the sum of overlay contributions

    u_h(x) = Σₖ u_h^(k)(x),

where each overlay contribution `u_h^(k)` is extended by zero outside its overlay domain `Ω^(k)`. The `levels` field is
a heterogeneous tuple so each level can carry its own basis-family type without losing type stability.

Fields:

  - `domain::AxisBox{D,T}` — the physical bounding box. Every overlay level's mesh domain must lie inside this box.
  - `levels::L` — tuple of `Level{D,T,…}`, base first then overlays in insertion order.
  - `physical::Union{Nothing,PhysicalDomain}` — optional level-set description of the physical domain `Ω`. `nothing`
    keeps the no-FCM hot path: integration is over the bounding box `domain` and every cell is `:full`. With a
    `PhysicalDomain`, cut and fictitious cells are handled per the [`PhysicalDomain`](@ref) docstring.

Leaf semantics — a cell carries basis functions only where no finer level has taken its region over — are not a field
here. They are a property of the whole stack rather than of one level, and the level that sheds a mode is not the one
that decides it, so the question is answered once, in `dof_layout`, where the modes are enumerated. [`prepare`](@ref)'s
`prune` keyword is the one way to build a space's unreduced twin, and it exists to be measured against rather than
solved; see [`space`](@ref) for the rule itself.
"""
struct Space{D,T<:Real,L<:Tuple}
    domain::AxisBox{D,T}
    levels::L
    physical::Union{Nothing,PhysicalDomain}
end

"""
    level_count(V::Space) -> Int

Number of levels (base + overlays) in the space.
"""
level_count(V::Space) = length(V.levels)

# Linear scan over `V.levels` resolving a level id to its `Level` value. Linear is fine because typical problems carry
# only a handful of levels; downstream callers cache the resolved levels when they iterate hot loops.
#
# The scan is a recursion over the level tuple, not a `for` loop: peeling the tuple with `Base.tail` unrolls it at
# compile time, so each step compares one statically known level's id and the match is returned as is. A `for` loop
# indexes the tuple at run time and materialises every `Level` it passes (144 bytes each in 2D) — about 40 ns per
# lookup on a six-level ladder, which made this lookup most of the cost of sizing the rules of an integration plan
# (`_axis_max`, one lookup per region parent).
_level_by_id(V::Space, id::Integer) = _level_by_id(V.levels, id)
@inline function _level_by_id(levels::Tuple, id::Integer)
    return first(levels).id == id ? first(levels) : _level_by_id(Base.tail(levels), id)
end
_level_by_id(::Tuple{}, id::Integer) = throw(ArgumentError("unknown level id $id"))

# Bounds check shared by every entry point that takes a level index, returning the index as an `Int` so the caller can
# use the result directly. Written once because every site that spelled it out chose a different message, and two of
# them raised a `BoundsError` naming the internal level tuple. `name` is the user-visible name of the argument
# (`"from"`, `"marked level"`, …).
function _check_level(V::Space, level::Integer, name::AbstractString="level")
    1 <= level <= length(V.levels) ||
        throw(ArgumentError("$name index $level is out of bounds for a space with " * "$(length(V.levels)) levels"))
    return Int(level)
end

# The guard the `level => spec` verbs share: every named level exists, and none is named twice. Naming a level twice is
# always a mistake rather than a last-one-wins composition — the two specs would be applied in argument order and the
# first silently discarded — and `verb` names the call in the message.
function _check_named_once(V::Space, pairs, verb::AbstractString)
    named = falses(length(V.levels))
    for (level, _) in pairs
        k = _check_level(V, level)
        named[k] && throw(ArgumentError("level $k is named more than once in one `$verb` call"))
        named[k] = true
    end
    return nothing
end

# `V` with its level tuple replaced, and with everything else — domain, physical domain — carried over. Every rebuild in
# this file funnels through these two, which is what keeps a new `Space` or `Level` field from having to be remembered
# at eight call sites; before them, each site spelled out the full positional constructor and a `ntuple` splice of its
# own.
_with_levels(V::Space{D,T}, levels::Tuple) where {D,T} = Space{D,T,typeof(levels)}(V.domain, levels, V.physical)

function _with_level(V::Space, k::Int, level::Level)
    return _with_levels(V, ntuple(i -> i == k ? level : V.levels[i], length(V.levels)))
end

# ── Basis-family hooks ────────────────────────────────────────────────────────

"""
    instantiate_basis(basis, mesh, order) -> BasisFamily

Finalize a basis family for a concrete level, just before the [`Level`](@ref) is built. The default returns `basis`
unchanged: most families — including the default [`IntegratedLegendre`](@ref) — are fully specified before any mesh
exists.

Families whose concrete form depends on the level's mesh overload this hook. The B-spline family in the `BasicBSpline`
extension is the motivating case: its per-axis knot vectors are built from the mesh cell coordinates, so
[`bspline`](@ref) returns a *deferred* specification and the real family is materialised here once the mesh is known.
Keeping the two-phase "spec → concrete family" handshake in one named hook (rather than buried in a `Level` constructor
overload) makes the extension point explicit and means every level-building path treats mesh-dependent families
uniformly.

Every level-building path routes the basis through this hook — there are exactly two, `_new_level` for a level built
from a user specification ([`space`](@ref), [`overlay`](@ref), [`elevate`](@ref)) and `_level` for a copy of an existing
one ([`moved_space`](@ref), [`adapt`](@ref), the mask mutators, the fictitious fold, level-id reindexing) — so a
mesh-dependent family is rebuilt whenever the mesh changes (e.g. an overlay move) and is validated against the level's
`order` at construction time. The basis mode is not an argument either: it is already folded into `basis` (see
[`BasisFamily`](@ref)), and a family that serves `:tensor` alone has refused any other mode before this hook runs. The
level's activation mask is not an argument: a mask face selects or constrains functions in the dof layer rather than
changing the family, which is how the shipped B-spline family handles every mask geometry.

A concrete family handed to this hook is a request like any other, and a family that validates its request must do so
on every call, not only on the first. The B-spline family re-judges its requested continuity against the new mesh and
order each time, so a level built by inheritance ([`overlay`](@ref)'s default `basis` is the base level's concrete
family), by [`elevate`](@ref), by a move or by a mask edit is validated and given its mechanism exactly as a fresh build
of the same request is.
"""
instantiate_basis(basis::BasisFamily, mesh::CartesianMesh{D,T}, order::NTuple{D,Int}) where {D,T} = basis

# ── Space construction ────────────────────────────────────────────────────────

"""
    space(domain::AxisBox; cells, order=1, basis=IntegratedLegendre(),
                          mode=:tensor, active=nothing, physical=nothing) -> Space

Build the base discretization of a [`Space`](@ref) over `domain`. The resulting space has one base level
(`role = :base`, `id = 1`); add overlay levels with [`overlay`](@ref).

Keyword arguments:

  - `cells` — base mesh resolution. Either a single positive integer (replicated across axes) or an `NTuple{D,Int}` of
    per-axis cell counts.
  - `order` — polynomial order of the basis, uniform over the level or varying from cell to cell. Accepted shapes:

      * a single positive integer (isotropic, uniform);
      * an `NTuple{D,Int}` (anisotropic, uniform; requires `mode = :tensor`);
      * an `AbstractArray{<:Integer,D}` shaped exactly like the level's cell grid (isotropic per cell);
      * an `AbstractArray{<:NTuple{D,Integer},D}` of the same shape (anisotropic per cell);
      * a predicate `(cell_box, cell_index) -> order` returning either of the uniform shapes, evaluated once per cell.

    A per-cell field whose entries are all equal collapses back to the uniform case, so the last three shapes cost
    nothing when they happen to be flat. Where two cells of different order share a face, the shared entity carries the
    *minimum* of the two orders (the classical hp-FEM minimum rule), which is what keeps the space C⁰ — see
    [`CellModes`](@ref). `nominal_order` then reports the per-axis maximum, and every cell keeps `order ≥ 1` on every
    axis, which is checked per cell rather than per level.

    Per-cell order is a basis-family capability: only integrated Legendre declares it, and a non-uniform field on any
    other family raises `ArgumentError` rather than being silently ignored.
  - `basis` — basis family. Defaults to [`IntegratedLegendre`](@ref).
  - `mode` — basis index set. `:tensor` is the full tensor product; `:trunk` is the Szabó–Babuška trunk space (filtered
    by trunk degree) and requires isotropic `order`. It is folded into the level's family (see [`BasisFamily`](@ref)),
    and a family that does not carry the mode refuses it.
  - `active` — per-cell activation mask, `nothing` (default) meaning every cell is active. The accepted shapes, all
    normalised to a [`LevelMask`](@ref) over the level's cells, are:

      * a `LevelMask{D}`, or an `AbstractArray{Bool,D}` shaped exactly like the level's cell grid, `true` where the cell
        participates. Both are copied, so mutating the argument afterwards does not reach the level. A selection naming
        every cell is stored as the no-mask default, which is the same level;
      * a predicate `(cell_box, cell_index) -> Bool` evaluated once per cell, with `cell_box` the cell's
        [`AxisBox`](@ref) and `cell_index` its `CartesianIndex{D}`;
      * an iterable of `CartesianIndex{D}` listing the active cells — every cell not listed is inactive.

    Inactive cells are dropped from dof enumeration, and faces between active and inactive cells of one level are
    treated like that level's artificial overlay boundary; see [`LevelMask`](@ref).
  - `physical` — optional [`PhysicalDomain`](@ref) describing an immersed `Ω ⊂ domain`. With `nothing` (default) the
    bounding box is the physical domain. Both shipped basis families support an immersed domain.

# Leaf semantics

A cell carries basis functions only where it is a *leaf* — where no finer level has taken the region over. A covered
cell is the parent of a leaf, and a parent carries no unknowns where its children do. There is nothing to configure: it
is what makes superposition equivalent to ordinary refinement, in which h-refining a cell *replaces* it rather than
adding to it. [`prepare`](@ref)`(problem; prune = false)` builds the unreduced twin as a diagnostic — a space that keeps
every covered mode, and that is exactly singular on a nested stack.

Concretely a level sheds a mode wherever a *single* finer level has taken over the region that mode lives on. On an
integrated Legendre level there are two eliminations, both per-mode rather than per-cell, and both asking first that the
mode be *buried* — every cell it is incident to covered by one finer level:

  * a buried **high-order** mode (one with at least one bubble axis) is dropped, so a fully covered cell keeps only its
    linear skeleton. An edge or face mode straddling the boundary of the covered region survives, because its own
    stencil is not buried.
  * a buried **linear** mode is dropped when one finer level reproduces it exactly, which takes both halves of a test:
    that level's mesh must refine this one's (`_nested_over` in `src/coverage.jl`) *and* its basis must span that mesh's
    hats (`_reproduces` in `src/dofs.jl`). A buried vertex function is a C⁰ hat, and a basis smoother than C⁰ across its
    own cell boundaries — a B-spline of degree ≥ 2 — carries nothing that reproduces the kink.

Leaving a covered mode active makes the superposed operator singular, because it and the covering level's reproduction
of it are linearly dependent: 169 to 1378 null directions were measured on nested ladders with the rule switched off.

The high-order half does not require the covering level to reproduce what it displaces, and that is deliberate rather
than an oversight. It is what lets a low-order cover sit under a high-order base and behave the way small low-order
elements behave near a singularity in any hp code: the covered region is then resolved at the cover's order, which is
what asking for a low-order cover *means*. Once the cover carries at least the base's order the elimination is exactly
lossless — measured, the reduced and unreduced errors agree to every printed digit from cover order 4 upward under a p=5
base.

A B-spline level has no bubble/skeleton split and so has only the second elimination: a buried function is dropped
exactly when a nested level above reproduces it, which for that family is not an accuracy trade at all but the thing
that keeps a nested stack non-singular — see `bspline` in the `BasicBSpline` extension. Any further family takes no part
in the burial test of `_coverage_constraints!` unless it opts in, so on such a space leaf semantics eliminate nothing.
See `src/coverage.jl` and `_coverage_constraints!` in `src/dofs.jl`; `diagnostics(...).reduced_mode_counts` reports the
per-level count.
"""
function space(domain::AxisBox{D,T}; cells, order=1, basis=IntegratedLegendre(), mode::Symbol=:tensor, active=nothing,
               physical=nothing) where {D,T}
    base_mesh = CartesianMesh(domain; cells)
    base_level = _new_level(1, :base, base_mesh, _with_mode(basis, mode), _normalize_order(order, base_mesh),
                            _normalize_mask(active, base_mesh))
    return Space{D,T,Tuple{typeof(base_level)}}(domain, (base_level,), physical)
end

"""
    overlay(V::Space, domain::AxisBox; cells, order=nominal_order(V.levels[1]),
                                       basis=V.levels[1].basis,
                                       mode=<the base level's mode>,
                                       tolerance=GeometryTolerance(T),
                                       active=nothing) -> Space

Add an overlay level on the sub-box `domain` to an existing [`Space`](@ref) `V`, returning the extended space. The
overlay's domain must lie inside `V.domain` up to `tolerance.contain` (see [`is_inside`](@ref)).

Defaults inherit from the base level so a quick overlay only needs `cells` and possibly `order`. Keyword arguments:

  - `cells` — per-axis cell counts for the overlay mesh.
  - `order` — polynomial order; defaults to the base level's nominal order. Takes every shape [`space`](@ref)'s `order`
    takes, including the per-cell ones, resolved against *this* overlay's cell grid.
  - `basis` — basis family; defaults to the base level's family.
  - `mode` — basis index set; defaults to the base level's mode. It always applies, to an explicit `basis` as to the
    inherited one, so an overlay of a `:trunk` base is `:trunk` unless `mode` says otherwise.
  - `tolerance` — slack on the inside-domain check.
  - `active` — optional per-cell mask, same shapes as [`space`](@ref)'s `active`.

Leaf semantics apply to this overlay exactly as to any level, and there is no keyword for them here either: where the
overlay is itself covered by something finer it carries no unknowns, and where it covers the level below, that level
carries none. See [`space`](@ref) for the rule and [`prepare`](@ref)'s `prune` for the unreduced twin it is measured
against.

The new level's `id` is `length(V.levels) + 1`. Overlay placement is independent of any existing overlay: overlay
boundaries need not coincide with lower-level element boundaries, and there is no topological merging with parents
(every overlay imposes homogeneous Dirichlet data on its artificial boundary, see `CONTRIBUTING.md`'s "The superposition
model" section).
"""
function overlay(V::Space{D,T}, domain::AxisBox{D,T}; cells, order=nominal_order(V.levels[1]), basis=V.levels[1].basis,
                 mode::Symbol=_mode(V.levels[1].basis), tolerance=GeometryTolerance(T), active=nothing) where {D,T}
    is_inside(domain, V.domain, tolerance) || throw(ArgumentError("overlay domain must lie inside the physical domain"))
    overlay_mesh = CartesianMesh(domain; cells)
    level = _new_level(length(V.levels) + 1, :overlay, overlay_mesh, _with_mode(basis, mode),
                       _normalize_order(order, overlay_mesh), _normalize_mask(active, overlay_mesh))
    return _with_levels(V, (V.levels..., level))
end

# A brand-new `Level` from a user-facing specification: the normalise → check → instantiate → derive pipeline `space`,
# `overlay` and `_reordered_space` each used to spell out. `field` is the already-normalised order (a tuple or a
# per-cell vector) and `mask` the already-normalised activation, because `_reordered_space` reuses the level's stored
# mask and `_normalize_mask` would copy it for nothing.
#
# The step order matters and is not free to rearrange: `instantiate_basis` must run before `_build_cell_orders`, because
# a deferred family (the B-spline spec) is not a family yet and `_supports_cell_order` has to be asked of the concrete
# one.
function _new_level(id::Int, role::Symbol, mesh::CartesianMesh{D,T}, basis, field, mask) where {D,T}
    family = instantiate_basis(basis, mesh, _nominal_order(field))
    orders = _build_cell_orders(family, field, mesh)
    return Level{D,T,typeof(family)}(id, role, mesh, family, orders, mask)
end

"""
    moved_space(V::Space; level, to, tolerance=GeometryTolerance(T)) -> Space

Return a copy of `V` with overlay `level` (its `level`-th level, which must be an overlay) relocated to the axis box
`to`, keeping the same cell counts, basis family, order, level id, and activation mask. The base level (level 1) cannot
be moved. This is the space-level building block behind [`move!`](@ref) and [`moved`](@ref).
"""
function moved_space(V::Space{D,T}; level::Integer, to::AxisBox{D,T}, tolerance=GeometryTolerance(T)) where {D,T}
    # The range check first, so an index past the last level is named as such rather than reported as the base level.
    k = _check_level(V, level)
    k > 1 || throw(ArgumentError("only overlay levels can be moved"))
    is_inside(to, V.domain, tolerance) || throw(ArgumentError("overlay domain must lie inside the physical domain"))
    old = V.levels[k]
    # The order field is indexed by cell and a move keeps the cell grid identical — only the coordinates change — so it
    # carries over verbatim. `_level` re-instantiates the family against the moved mesh, which is what a mesh-dependent
    # family (B-splines) needs to rebuild its knot vectors.
    return _with_level(V, k, _level(old; mesh=CartesianMesh(to; cells=old.mesh.cells)))
end

# Rebuild `V` with the mask of one level replaced. Mirrors `moved_space` but keeps the mesh / basis / order fixed and
# swaps only the mask. `mask` must already be normalised. Used by [`adapt`](@ref) and by the `activate!` / `deactivate!`
# model-level mutators in `model.jl`.
#
# A mask edit changes which shared-entity modes survive, and nothing has to be rebuilt for that: the order field is the
# caller's and is unaffected, and `Level`'s constructor re-derives the modes from the new mask.
function _remasked_space(V::Space, level_index::Integer, mask)
    k = _check_level(V, level_index)
    return _with_level(V, k, _level(V.levels[k]; mask=mask))
end

# A copy of an existing `Level` with one thing changed, by keyword. Every rebuild in this file is one of these: a move
# changes the mesh, `activate!` the mask, `_apply_physical_fold` the mask, coupling the id.
#
# The family is re-instantiated unconditionally. That is behaviour-preserving rather than merely convenient:
# `instantiate_basis` consumes the mesh and the nominal order, the only things a family's concrete form can depend on —
# integrated Legendre returns itself, and the B-spline hook rebuilds knot vectors from the mesh axes and the degree — so
# a rebuild that keeps the mesh and the order, such as a mask update, re-instantiates an equal family (for B-splines the
# same knot vectors, rebuilt). Before this helper the sites disagreed about it — `moved_space` and the mask rebuilds
# re-instantiated while the fold and the reindex did not — and the disagreement was invisible because the answers
# coincide.
function _level(old::Level{D,T}; id=old.id, mesh=old.mesh, orders=old.orders, mask=old.mask) where {D,T}
    family = instantiate_basis(old.basis, mesh, orders.nominal)
    return Level{D,T,typeof(family)}(id, old.role, mesh, family, orders, mask)
end

# ── Reading a level ───────────────────────────────────────────────────────────

"""
    active_cells(V::Space; level) -> BitArray

Which cells of `level` are active, as a copy. A level with no mask returns an all-true array. The `Space` counterpart of
`active_cells(::Model; level)`, and the read half of an adaptive step: fetch a level's mask, edit it per cell, hand it
back to [`adapt`](@ref).

On a `Space` this is the mask as written. The `Model` method reports the *effective* mask by default, which on a space
carrying a [`PhysicalDomain`](@ref) already has the fictitious fold applied; pass `effective = false` there to read back
what the caller asked for.
"""
function active_cells(V::Space{D}; level::Integer) where {D}
    lvl = V.levels[_check_level(V, level)]
    lvl.mask === nothing && return trues(lvl.mesh.cells)
    return copy(lvl.mask.on)
end

"""
    support_extension(V::Space, cells; level) -> BitArray{D}

The cells of `level` that must be active for the level to actually carry the functions the marked `cells` were meant to
wake: the marked set dilated, per axis, by the basis family's support radius.

`cells` takes every shape [`space`](@ref)'s `active =` keyword takes — a `BitArray`/`Array{Bool,D}` over the cell grid,
an iterable of `CartesianIndex{D}`, or a predicate `(cell_box, cell_index) -> Bool` — and the result is a `BitArray{D}`,
which is what [`adapt`](@ref), `active =` and [`activate!`](@ref) accept, so marking composes without a conversion.

For the default integrated Legendre family the radius is zero and this is the identity: its bubbles are cell-local and
its endpoint modes belong to the node two cells share, so activating a cell is enough to generate that cell's own
functions.

For a family with wider support it is not. A degree-`p` B-spline spans `p + 1` cells, and a maximal-continuity level
keeps exactly the functions whose whole support lies inside its active region, so a region `p` cells wide or less
carries *no unknowns at all* and an activation front has to advance `p` cells before it buys one. Dilating the marked
region is the fix, and it is the support-extension condition behind admissible hierarchical meshes (A. Buffa, C.
Giannelli, Math. Models Methods Appl. Sci. **26** (2016) 1–25,
[doi:10.1142/S0218202516500019](https://doi.org/10.1142/S0218202516500019)); the alternative — lowering the level's
`continuity` — buys unknowns with the smoothness of the whole superposition, which is a different decision.

Measured on a tanh layer of width 1/40 with a degree-3 B-spline base of 8 cells and a band overlay at constant
`h = 1/128`, widening the band from 1 to 4 base cells per side took the relative `L²` error from 7.35e-2 to 3.85e-4
without refining anything.

```julia
marked = [CartesianIndex(6, 6), CartesianIndex(6, 7)]
V = adapt(V, 2 => support_extension(V, marked; level = 2))
```

It is exactly [`dilate`](@ref) with the family's own radius, and the two are worth keeping apart: `dilate` is a
geometric operation the caller chooses the width of, while this one asks the basis how far it reaches and is therefore
the right call whenever the reason for widening is *that the functions need the room*. Nothing else in the package
computes that radius for you.

The dilation clips to the level's own cell grid; it does not widen the level's box. A marked region against a box face
therefore stays against it, and the functions that would have reached past the face are the ones the face's own
condition removes — unless that face is physical, where nothing is removed.

Widening is the caller's job everywhere except inside [`refine`](@ref), whose h-step applies this to the cells it wakes.
`space` and `overlay` build exactly the level they are given: a level too thin to hold a support is accepted and carries
no functions, which `diagnostics(…).levels[k]` reports as `active_functions = 0` against a nonzero `raw_functions`. That
signature narrows the cause without naming it — `active_functions` counts what survived *every* elimination, so physical
Dirichlet data produces it too — but on an overlay, whose faces are artificial by construction, the thin case is what it
means.
"""
function support_extension(V::Space{D}, cells; level::Integer) where {D}
    lvl = V.levels[_check_level(V, level)]
    return dilate(V, cells; level=level, by=_support_radius(lvl.basis, nominal_order(lvl)))
end

"""
    dilate(V::Space, cells; level, by) -> BitArray{D}
    dilate(mask::AbstractArray{Bool,D}, by) -> BitArray{D}

Grow a set of cells by `by` cells in every direction, clipped to the grid.

`by` is an `Integer` applied to every axis or an `NTuple{D,Integer}` applied per axis, and must be non-negative. The
`Space` form takes `cells` in every shape [`space`](@ref)'s `active =` keyword takes — a `BitArray`/`Array{Bool,D}` over
the cell grid, an iterable of `CartesianIndex{D}`, or a predicate `(cell_box, cell_index) -> Bool` — and resolves it
against `level`'s mesh; the array form is the bare operation for a mask already in hand. Both return a `BitArray{D}`,
which is what [`adapt`](@ref), `active =` and [`activate!`](@ref) accept, so a marking loop composes without a
conversion.

The dilation is a **box**, not a cross: a cell survives when it is within `by[d]` along *each* axis independently, so a
single cell grows to `∏_d (2·by[d] + 1)`. That is the shape a tensor-product support has, and it is the reason
[`support_extension`](@ref) — which is this function with the basis family's own radius — is written in terms of it.

Two uses, and they are different enough to be worth naming. A **margin** around a feature that is about to move: a
transient loop that refines a travelling front widens the band so the front does not leave it before the next estimate,
and `by` is then a statement about the time step rather than about the basis. And a **halo** a caller wants for its own
reasons — a region to hold a coefficient transition, or a buffer a downstream tool needs. Where the reason is that the
basis functions need room, reach for [`support_extension`](@ref) instead, so the width follows the order rather than
being pinned by hand.

```julia
front = findall(c -> band(cell_box(V, c; level = 2)), cell_indices(V; level = 2))
V = adapt(V, 2 => dilate(V, front; level = 2, by = 3))
```

It never widens the level's box — a region against a face stays against it — and it is monotone: dilating an
already-dilated set by `a` then `b` is the same as dilating once by `a + b`.
"""
function dilate(V::Space{D}, cells; level::Integer, by) where {D}
    lvl = V.levels[_check_level(V, level)]
    # `by` is validated first, so a bad width is reported whatever the selection is — including one naming every cell,
    # which an earlier early return let through while the array method rejected it.
    radius = _dilation_radius(by, Val(D))
    return _dilate_cells(_selection(cells, lvl.mesh), radius)
end

function dilate(mask::AbstractArray{Bool,D}, by) where {D}
    on = BitArray(undef, size(mask))
    copyto!(on, mask)
    return _dilate_cells(on, _dilation_radius(by, Val(D)))
end

# Coerce a `by =` specification into a per-axis radius. Deliberately parallel to `_axis_int_tuple`, which it cannot
# reuse: cell counts and polynomial orders must be positive, and a dilation width of zero is the perfectly sensible
# request to leave one axis alone.
function _dilation_radius(by::Integer, ::Val{D}) where {D}
    by >= 0 || throw(ArgumentError("dilate: `by` must be non-negative; got $by"))
    return ntuple(_ -> Int(by), D)
end

function _dilation_radius(by::NTuple{D,<:Integer}, ::Val{D}) where {D}
    all(>=(0), by) || throw(ArgumentError("dilate: every entry of `by` must be non-negative; " * "got $by"))
    return Int.(by)
end

function _dilation_radius(by, ::Val{D}) where {D}
    throw(ArgumentError("dilate: `by` must be a non-negative integer or an NTuple{$D,Int}; " * "got $(typeof(by))"))
end

# Erode a cell mask by `radius[d]` cells along each axis: the cells whose whole neighbourhood, clipped to the grid, is
# inside the mask. Written as the dual of the dilation rather than as its own pass, which is both shorter and exactly
# consistent with it — in particular they agree on the clipping at a box face.
#
# Used by `coarsen` to recover the set `refine` dilated, which is the one thing a mask does not record: erosion is a
# right inverse of dilation on a dilated set, `dilate(erode(dilate(X, r), r), r) == dilate(X, r)`.
_erode_cells(marked::BitArray{D}, radius::NTuple{D,Int}) where {D} = .!_dilate_cells(.!marked, radius)

# Dilate a cell mask by `radius[d]` cells along each axis, clipped to the grid.
#
# Done as `D` successive one-axis passes rather than one `∏_d (2r_d + 1)`-cell stencil per marked cell: the separable
# form writes `Σ_d (2r_d + 1)` cells per marked cell against `∏_d (2r_d + 1)`, which at `D = 3` and `r = 3` is 21
# instead of 343. Each pass reads the previous pass's result, which is what makes the composition the full box dilation
# rather than a cross.
function _dilate_cells(marked::BitArray{D}, radius::NTuple{D,Int}) where {D}
    n = size(marked)
    out = copy(marked)
    scratch = similar(out)
    for d in 1:D
        radius[d] == 0 && continue
        fill!(scratch, false)
        for ci in CartesianIndices(out)
            out[ci] || continue
            lo = max(1, ci.I[d] - radius[d])
            hi = min(n[d], ci.I[d] + radius[d])
            for j in lo:hi
                scratch[CartesianIndex(ntuple(e -> e == d ? j : ci.I[e], D))] = true
            end
        end
        out, scratch = scratch, out
    end
    return out
end

"""
    cell_orders(V::Space; level) -> Array{NTuple{D,Int},D}

Per-axis polynomial order of every cell of `level`, as a fresh array. A level whose order is uniform returns that order
repeated over the cell grid, so the result has the same shape and meaning whether or not the level carries a per-cell
field.

The read half of a p-adaptive step, and the `Space` counterpart of `cell_orders(::Model; level)`: fetch a level's
orders, edit them per cell, hand them back to [`elevate`](@ref). Round-tripping through `elevate` is an identity.

```julia
p = cell_orders(V; level=1)
p[marked] .+= 1
V = elevate(V, 1 => p)
```
"""
function cell_orders(V::Space{D}; level::Integer) where {D}
    lvl = V.levels[_check_level(V, level)]
    return [lvl.orders.palette[c] for c in lvl.orders.class]
end

nominal_order(V::Space; level::Integer) = nominal_order(V.levels[_check_level(V, level)])

function cell_order(V::Space{D}, cell::CartesianIndex{D}; level::Integer) where {D}
    return cell_order(V.levels[_check_level(V, level)], cell)
end

"""
    cell_indices(V::Space; level) -> CartesianIndices

Cell index space of `level`. Together with [`cell_box`](@ref)'s `Space` method this is what lets a caller build a cell
selection — evaluate an error indicator per cell, mark, [`adapt`](@ref) — without reaching into `V.levels[k].mesh`.
"""
cell_indices(V::Space; level::Integer) = cell_indices(V.levels[_check_level(V, level)].mesh)

"""
    cell_box(V::Space, index::CartesianIndex; level) -> AxisBox

Axis-aligned box of one cell of `level`, in the physical frame.
"""
function cell_box(V::Space{D}, index::CartesianIndex{D}; level::Integer) where {D}
    return cell_box(V.levels[_check_level(V, level)].mesh, index)
end

# ── Writing a level: the `elevate` and `adapt` verbs ──────────────────────────

# The order spec shape that only `elevate` can serve: `cell => order` pairs against the level's current field.
# Normalised here rather than in `_normalize_order` because it needs a base to fill the unnamed cells from, and `space`
# / `overlay` have none.
function _pair_orders(V::Space{D}, k::Int, pairs) where {D}
    lvl = V.levels[k]
    out = cell_orders(V; level=k)
    for entry in pairs
        entry isa Pair || throw(ArgumentError("an order pair list must contain " *
                                              "`CartesianIndex{$D} => order` entries; got $(typeof(entry))"))
        cell, order = entry
        cell isa CartesianIndex{D} ||
            throw(ArgumentError("an order pair list must be keyed by CartesianIndex{$D}; " * "got $(typeof(cell))"))
        checkbounds(Bool, out, cell) ||
            throw(ArgumentError("cell index $cell is out of bounds for mesh cells $(lvl.mesh.cells)"))
        out[cell] = _axis_int_tuple(order, Val(D), :order)
    end
    return out
end

# Which spellings are a pair list. Every container the docstring names goes to `_pair_orders`, which owns the entry
# checks; a generator is collected first, so a generator of pairs takes the vector path and anything else it yields is
# judged as the order spec it then is. Everything that is not a pair list is an ordinary order spec. A vector must be
# typed as a vector of pairs, because a vector of integers is a per-cell order field on a 1D level.
const _PairList = Union{AbstractVector{<:Pair},Tuple{Vararg{Pair}},AbstractDict}
_normalize_elevate_spec(V::Space, k::Int, spec::Pair) = _pair_orders(V, k, (spec,))
_normalize_elevate_spec(V::Space, k::Int, spec::_PairList) = _pair_orders(V, k, spec)
_normalize_elevate_spec(V::Space, k::Int, spec::Base.Generator) = _normalize_elevate_spec(V, k, collect(spec))
_normalize_elevate_spec(::Space, ::Int, spec) = spec

# Rebuild `V` with the polynomial order of one level replaced. Mirrors `_remasked_space` but keeps the mesh / basis /
# mask fixed and swaps only the order field, so the level's type — and with it the `Space` type and the compiled
# assembly pipeline — is unchanged. The level's own mask is passed through without re-normalising, which is why this
# goes to `_new_level` rather than `_level`: the order field is new and the whole pipeline has to run for it.
function _reordered_space(V::Space, level_index::Integer, spec)
    k = _check_level(V, level_index)
    old = V.levels[k]
    field = _normalize_order(_normalize_elevate_spec(V, k, spec), old.mesh)
    return _with_level(V, k, _new_level(old.id, old.role, old.mesh, old.basis, field, old.mask))
end

"""
    elevate(V::Space, level => order, ...) -> Space

Return `V` with the named levels' polynomial orders replaced. Meshes, level boxes, activation masks and the basis
families' requests are untouched; a family whose concrete form depends on the order — the B-spline family's knot vectors
— is rebuilt at the new order and validated there, as a fresh build of the same request would be.

For a family whose degree is not a type parameter, integrated Legendre's, the result has the *same type* as `V` and a
model rebuilt from it does not recompile the assembly pipeline: the per-cell order lives in a `CellOrders` field whose
palette length is a value rather than a type parameter, so a p-step never changes `typeof(V)`. A B-spline level's degree
is a type parameter of its knot-vector spaces (`BSplineSpace{p}`), so raising it changes the level's type, and the
pipeline is compiled for the new one.

`order` takes every shape [`space`](@ref)'s `order` keyword takes — an integer, an `NTuple{D,Int}`, an array of either
shaped like the level's cell grid, or a predicate `(cell_box, cell_index) -> order` — plus one shape that only makes
sense against an existing level:

  * `CartesianIndex{D} => order` pairs — one `Pair`, or a vector, tuple, dictionary or generator of them — raising the
    listed cells and leaving every other cell at the order it already has. It is the shape a marking loop produces when
    it changes a dozen cells: the caller states only what changed and never has to read, copy and hand back the whole
    field. The level's current field is still read once here, to fill the cells the list does not name.

Where two cells of different order share a face, the shared entity carries the minimum of the two orders; see
[`CellModes`](@ref) for why that is what keeps the space C⁰, and [`space`](@ref) for the per-cell `order ≥ 1` and
basis-family requirements, which are checked here too.

This is a separate verb from [`adapt`](@ref) rather than another shape of its pair form, and the reason is dispatch
rather than taste: a mask spec and an order spec collide irreducibly on `nothing` ("every cell active" versus "uniform
order") and on a predicate (`-> Bool` versus `-> Int`), and neither collision can be detected before the value is used.
Compose them instead — `elevate(adapt(V, h), p)` costs two cheap `Space` rebuilds and one [`prepare`](@ref), and the dof
layout only ever sees the final hp state.
"""
function elevate(V::Space, pairs::Pair{<:Integer}...)
    _check_named_once(V, pairs, "elevate")
    out = V
    for (level, spec) in pairs
        out = _reordered_space(out, level, spec)
    end
    return out
end

"""
    adapt(V::Space, level => cells, ...) -> Space

Return `V` with the named levels' activation masks replaced. Level boxes, cell counts, orders and basis families are
untouched, so the result has the *same type* as `V` and a model rebuilt from it does not recompile the assembly
pipeline.

The pair form sets each named level's mask outright and leaves every other level alone. `cells` takes the shapes
`active =` accepts, so a mask written for [`overlay`](@ref) can be handed straight here:

```julia
m = active_cells(V; level=4)
m[marked] .= true
V = adapt(V, 4 => m)
```

Setting several levels in one call is one integration-plan and dof-layout rebuild; the same edit through
[`activate!`](@ref) is one of each per level.

Nothing requires a level to be active only where its parent is, or its active cells to tile whole parent cells. Both are
admissible, and a feature far finer than a base cell can be resolved without paying for the intervening levels. What
they cost is conditioning: where a level's active region boundary does not fall on the cell boundaries of the level
below, the condition number of the assembled operator rises — measured at one to three orders of magnitude on stacked
ragged masks, against a parent-aligned stack carrying several times the unknowns. Prefer a mask whose boundary follows
the coarser level's cells when there is a choice.
"""
function adapt(V::Space, pairs::Pair{<:Integer}...)
    _check_named_once(V, pairs, "adapt")
    out = V
    for (level, cells) in pairs
        k = _check_level(V, level)
        out = _remasked_space(out, k, _normalize_mask(cells, V.levels[k].mesh))
    end
    return out
end

# ── Fictitious-cell fold ──────────────────────────────────────────────────────

# Apply the PhysicalDomain fold to `V`. For each level, classifies cells against `V.physical` and merges the resulting
# fictitious mask into the level's user mask, so cells fully outside Ω are dropped from the dof layout. This happens
# *regardless of α* — the α-FCM stabilization enriches cut-cell quadrature (in `intersections.jl`), it does not keep
# whole fictitious cells around. Returns `V` unchanged only when there is no physical domain, or when `keep_fictitious`
# opts into the classic α-FCM treatment (fully-fictitious cells kept active with α-scaled quadrature). The
# `_ClassifyCache` carries the per-box classification verdicts so the region-level dispatcher in `intersections.jl` can
# reuse them.
#
# Contract: `V` must be a *pre-fold* space — one whose level masks still mean "what the caller asked for". The fold is
# an intersection (`active .& .!fictitious`) and is not invertible, so applying it to its own output can only remove
# cells: a cell dropped as fictitious at one overlay position stays dropped after the overlay moves somewhere it would
# be inside Ω. Callers that fold repeatedly over a model's lifetime must therefore keep the pre-fold space and re-derive
# from it — see `Model.prefold_space` and `_remodel!` in `model.jl`.
function _apply_physical_fold(V::Space{D,T}, cache::_ClassifyCache{D,T}) where {D,T}
    V.physical === nothing && return V
    V.physical.keep_fictitious && return V

    new_levels = ntuple(length(V.levels)) do i
        level = V.levels[i]
        mask = _folded_mask(level, V.physical, cache)
        mask === nothing ? level : _level(level; mask)
    end
    return _with_levels(V, new_levels)
end

# The fold of one level: its user mask with every cell the classifier reports as fully outside Ω (`:fictitious`)
# switched off as well, or `nothing` when the classifier reports none. A cell is deactivated when *either* source asks
# for it: the user explicitly excluded it via `active=`, or it lies entirely outside Ω; a `nothing` user mask is treated
# as "all active". Cells classified as `:full` or `:cut` stay active: `:cut` cells will get their quadrature replaced by
# the NNMF moment-fit rule at region-build time (see `intersections.jl`), while `:fictitious` cells are dropped from the
# dof layout — at every α, unless `keep_fictitious` asks for the classic α-FCM fill instead.
#
# Only cells the level's own mask already keeps active are classified: a cell the user already deactivated stays
# deactivated whatever the classifier would say about it, so skipping it is bit-identical to classifying it, and the
# guard turns the pass from O(cells of the level) into O(active cells of the level).
#
# That distinction is what makes a deep, sparsely-populated overlay affordable. A refinement level whose grid is Nᴰ
# cells but which carries a handful of active ones — the normal state of an adaptive front — otherwise pays a full-grid
# classification sweep every time the mesh is folded, and `classify_cell` is not cheap: it walks a bounded octree
# whenever the Lipschitz certificate cannot settle the box from its centre sample.
#
# Whether an *inactive* cell holds material is still needed further down the pipeline — coverage (`_covered_by_level`,
# `coverage.jl`) and the artificial-face tests of the dof layer (`_has_overlay_constraint`, `dofs.jl`) all ask it — but
# each asks only about the cells it meets, through the dof layer's material oracle (`_Fold`, `coverage.jl`) over the
# same `cache`. Skipping them here turns those queries from cache hits into cache misses; it does not change a single
# verdict, because `classify_cell` is a pure function of the box and the geometry.
#
# Returning `nothing` when nothing folds is what lets `_apply_physical_fold` keep such a level as it is, and with it a
# `nothing` mask — the no-mask hot path.
#
# One pass that clears bits of a copy of the mask, rather than classifying into a separate `BitArray` and merging the
# two by the broadcast `mask.on .& .!fictitious`. The bit reads and writes ride on the classification loop, which visits
# every cell anyway, so the broadcast's word-at-a-time merge would only add a pass; and the loop is the cheaper form to
# compile. Measured on the earlier two-pass form, where only the merge was a loop or a broadcast, a fresh process's
# first `prepare` was 0.3–1 % slower with the broadcast, both on spaces that compile the fold without running it (a
# two-level overlay, a masked clamped B-spline space) and on an immersed FCM plate; the one-pass form compiles one
# method fewer than that loop.
function _folded_mask(level::Level{D,T}, physical::PhysicalDomain, cache::_ClassifyCache{D,T}) where {D,T}
    on = level.mask === nothing ? trues(level.mesh.cells) : copy(level.mask.on)
    folded = false
    for ci in cell_indices(level.mesh)
        on[ci] || continue
        classify_cell(physical, cell_box(level.mesh, ci), cache) === :fictitious || continue
        on[ci] = false
        folded = true
    end
    return folded ? LevelMask{D}(on) : nothing
end

# ── Level-id reindexing ───────────────────────────────────────────────────────

# Rebuild `V` with every level id shifted by `offset`, giving the space a disjoint level-id block
# `[offset+1, offset+level_count]`.
#
# Level ids are unique only *within* one space (base = 1, overlays 2, 3, …). When several independent discretisations
# are coupled in one multi-domain model, the assembly workspace banks basis values in a single flat vector indexed by
# level id, and each field's dof layout keys `cell_dofs` by the same id — so before those are built, each participating
# space is reindexed into its own contiguous id block. `offset == 0` returns `V` unchanged (the single-domain fast
# path). Positional level addressing (`move!` / `activate!`, which index `V.levels` by position) is unaffected;
# `_level_by_id` scans by id and works with any id assignment.
function _reindex_space_levels(V::Space, offset::Int)
    offset == 0 && return V
    return _with_levels(V, ntuple(i -> _level(V.levels[i]; id=V.levels[i].id + offset), length(V.levels)))
end

# ── Display ───────────────────────────────────────────────────────────────────

function Base.show(io::IO, level::Level)
    print(io, "Level(id=", level.id, ", role=:", level.role, ", cells=", level.mesh.cells, ", order=",
          nominal_order(level), ", mode=:", _mode(level.basis), ", basis=:", basis_name(level.basis))
    # A per-cell field is summarised by its palette size, never dumped: a graded level carries one order per cell and
    # printing them all is unreadable. A uniform level's palette holds exactly one entry and says nothing worth
    # printing, so the suffix appears only where the order actually varies.
    length(level.orders.palette) > 1 && print(io, ", cell_orders=", length(level.orders.palette))
    if level.mask !== nothing
        print(io, ", active=", count(level.mask.on), "/", length(level.mask.on))
    end
    print(io, ")")
end

function Base.show(io::IO, V::Space{D}) where {D}
    print(io, "Space(D=", D, ", levels=", length(V.levels), ", domain=", V.domain, ")")
end
