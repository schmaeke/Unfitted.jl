# ── Cartesian mesh ────────────────────────────────────────────────────────────

"""
    CartesianMesh(domain; cells)

Tensor-product Cartesian mesh over the axis-aligned `domain`. `cells` is
either a single positive integer used in every direction or an
`NTuple{D,Int}` of per-axis cell counts. The mesh stores the canonical
per-axis coordinate vectors (`length(axes[d]) == cells[d] + 1`) so that
cell boxes, parent lookup, and integration-region construction never
recompute the same coordinates.

The mesh discretization is uniform: axis `d` is split into `cells[d]`
equal sub-intervals. Non-uniform Cartesian meshes are not currently
supported.
"""
struct CartesianMesh{D,T<:Real}
    domain::AxisBox{D,T}
    cells::NTuple{D,Int}
    axes::NTuple{D,Vector{T}}
end

# Multi-method dispatcher: coerce a user-supplied "cell count" / "polynomial
# order" specification into a canonical `NTuple{D,Int}`. Accepts a single
# positive integer (replicated across every axis) or a `D`-tuple of integers
# (one per axis). The fallback method gives a clear error for any other
# input shape. `name` is the user-visible parameter name used in error
# messages (`:cells`, `:order`, …).
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

# Per-axis coordinate samples for a uniform Cartesian mesh: axis `d` carries
# `cells[d] + 1` values evenly spaced from `domain.lower[d]` to
# `domain.upper[d]`. The endpoints come out bit-exact — `i = 0` is `lo`
# itself and `i = n` reduces to `lo + (hi − lo)` — which the `Base.:(==)`
# comparison on `AxisBox` corner coordinates relies on.
function _mesh_axes(domain::AxisBox{D,T}, cells::NTuple{D,Int}) where {D,T}
    return ntuple(D) do d
        lo = domain.lower[d]
        hi = domain.upper[d]
        n = cells[d]
        [lo + (hi - lo) * convert(T, i) / convert(T, n) for i in 0:n]
    end
end

function CartesianMesh(domain::AxisBox{D,T}; cells) where {D,T}
    cell_tuple = _axis_int_tuple(cells, Val(D), :cells)
    return CartesianMesh{D,T}(domain, cell_tuple, _mesh_axes(domain, cell_tuple))
end

"""
    mesh(domain::AxisBox; cells) -> CartesianMesh

Public-API thin wrapper over the [`CartesianMesh`](@ref) constructor. Use it
when you want a standalone mesh; for building a discretization for a
problem, prefer [`space`](@ref).
"""
mesh(domain::AxisBox; cells) = CartesianMesh(domain; cells)

"""
    cell_count(m::CartesianMesh) -> Int

Total number of cells in the mesh, `prod(m.cells)`.
"""
cell_count(m::CartesianMesh) = prod(m.cells)

"""
    cell_indices(m::CartesianMesh) -> CartesianIndices{D}

Iterator over the `D`-dimensional cell index space of `m`. Equivalent to
`CartesianIndices(m.cells)` but routed through this accessor so callers do
not reach into the struct.
"""
cell_indices(m::CartesianMesh{D}) where {D} = CartesianIndices(m.cells)

"""
    boundary_coordinates(m::CartesianMesh) -> NTuple{D,Vector{T}}

Per-axis element-boundary coordinates of the mesh: `axes[d]` carries the
`cells[d] + 1` coordinates separating the cells along axis `d`. Used by
`intersections.jl` to collect every participating mesh's boundary
coordinates before merging them into the admissible-box partition.
"""
boundary_coordinates(m::CartesianMesh) = m.axes

"""
    cell_box(m::CartesianMesh, index::CartesianIndex) -> AxisBox

Axis-aligned box for the cell at the given multi-index. Returns the cell
in the mesh's physical frame; combine with [`physical_to_reference`](@ref)
and [`reference_to_physical`](@ref) to map between the cell, its parent
level's reference frame, and a region's reference frame. Throws
`BoundsError` if `index` is outside `cell_indices(m)`.
"""
function cell_box(m::CartesianMesh{D,T}, index::CartesianIndex{D}) where {D,T}
    checkbounds(Bool, cell_indices(m), index) || throw(BoundsError(m, index))
    lower = SVector{D,T}(ntuple(d -> m.axes[d][index.I[d]], D))
    upper = SVector{D,T}(ntuple(d -> m.axes[d][index.I[d] + 1], D))
    return AxisBox{D,T}(lower, upper)
end

"""
    locate_cell(m::CartesianMesh, point; tol=GeometryTolerance(T)) -> Union{CartesianIndex,Nothing}

Find the cell of `m` containing `point`, or `nothing` if `point` is outside
the mesh's `domain` (up to `tol.contain`). For a point lying exactly on an
interior element boundary the convention is "upper cell wins": the cell
whose lower face is at the boundary — the larger index along that axis — is
returned, because `searchsortedlast` maps a coordinate sitting on node
`axis[i]` to cell `i`. Points right at the upper boundary of an axis are
snapped to the last cell along that axis so a point sitting exactly on
`domain.upper[d]` still returns a valid cell.

Used by region-to-parent coverage (`_parents_covering` in
`intersections.jl`) and by post-processing evaluation
(`_level_value`, `_level_gradient`).
"""
function locate_cell(m::CartesianMesh{D,T}, point::PointLike{D};
                     tol=GeometryTolerance(T)) where {D,T}
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

# ── Per-cell activation mask ──────────────────────────────────────────────────

"""
    LevelMask{D}(on::BitArray{D})

Per-cell activation mask for a level. `on[cell]` is `true` when the cell
participates in assembly and dof enumeration. Carried as an optional field
on [`Level`](@ref); a `nothing` mask means "every cell active" and keeps
the no-mask hot path type-stable and allocation-free.

The mask interacts with the dof layer like a geometric overlay boundary:
faces between active and inactive cells of the *same* level are treated
as artificial overlay constraints (analogous to faces between the level
and its physical-domain complement). Span-mode dofs whose support is
entirely inside inactive cells are not enumerated.

One exception, and it is a physical one: when the inactive side of such a
face is inactive only because the finite-cell fold found it *fully
fictitious*, the face carries no physical material and the modes on it stay
active — they vanish on every physical face anyway and instead carry the
adjacent cut cell's approximation up to `∂Ω`. A cell the *user* masked out
is treated as physical and does constrain. See `_internal_face_is_physical`
in `src/dofs.jl`.
"""
struct LevelMask{D}
    on::BitArray{D}
end

"""
    is_active(mask, cell) -> Bool

Whether `cell` is active under the level's mask. `nothing` (the default
mask) treats every cell as active.
"""
is_active(::Nothing, ::CartesianIndex) = true
is_active(mask::LevelMask{D}, cell::CartesianIndex{D}) where {D} = mask.on[cell]

# Translate a user-supplied `active` specification into either `nothing`
# (no mask, default) or a `LevelMask{D}` whose shape matches the mesh's
# cell grid. Five accepted input shapes, dispatched on type:
#
#   * `nothing`                            — no mask, default fast path.
#   * `LevelMask{D}` with matching shape   — defensively copied so user
#                                            mutation of the BitArray does
#                                            not leak into the level.
#   * `AbstractArray{Bool,D}` of cell shape — copied into a fresh BitArray.
#   * predicate `(cell_box, cell_index) -> Bool` — evaluated per cell.
#   * iterable of `CartesianIndex{D}`      — listed cells are active, the
#                                            rest inactive.
#
# The fallback signature `_normalize_mask(cells, mesh::CartesianMesh{D})`
# acts as a generic iterable consumer plus a type-check error path so a
# wrong-typed `active=` keyword still gives a clear message.
_normalize_mask(::Nothing, ::CartesianMesh) = nothing

function _normalize_mask(mask::LevelMask{D}, mesh::CartesianMesh{D}) where {D}
    size(mask.on) == mesh.cells ||
        throw(DimensionMismatch("mask shape $(size(mask.on)) does not match mesh cells $(mesh.cells)"))
    # Copy: a user mutation of `mask.on` after construction must not leak
    # into the level. Matches `active_cells`'s return-a-copy contract.
    return LevelMask{D}(copy(mask.on))
end

function _normalize_mask(bits::AbstractArray{Bool,D}, mesh::CartesianMesh{D}) where {D}
    size(bits) == mesh.cells ||
        throw(DimensionMismatch("mask shape $(size(bits)) does not match mesh cells $(mesh.cells)"))
    on = BitArray(undef, mesh.cells)
    copyto!(on, bits)
    return LevelMask{D}(on)
end

function _normalize_mask(f::Function, mesh::CartesianMesh{D}) where {D}
    on = BitArray(undef, mesh.cells)
    for ci in cell_indices(mesh)
        on[ci] = f(cell_box(mesh, ci), ci)::Bool
    end
    return LevelMask{D}(on)
end

function _normalize_mask(cells, mesh::CartesianMesh{D}) where {D}
    on = falses(mesh.cells)
    for ci in cells
        ci isa CartesianIndex{D} ||
            throw(ArgumentError("active cell list must contain CartesianIndex{$D} entries; got $(typeof(ci))"))
        checkbounds(Bool, on, ci) ||
            throw(ArgumentError("active cell index $ci is out of bounds for mesh cells $(mesh.cells)"))
        on[ci] = true
    end
    return LevelMask{D}(on)
end

# ── Per-cell polynomial order ─────────────────────────────────────────────────

"""
    CellOrders{D}

Per-cell polynomial order of a [`Level`](@ref), together with the local basis
index set each cell actually generates under the hp-FEM *minimum rule*.

A level whose order is one tuple for every cell carries `nothing` here — that is
the representation this type extends, not one it replaces, and it stays the
allocation-free hot path. Fields:

  - `palette::Vector{NTuple{D,Int}}` — the distinct per-axis orders occurring on
    the level, in first-occurrence order. Stored as a palette rather than a dense
    `Array{NTuple{D,Int},D}` because a graded p-field carries a handful of
    distinct orders however many cells it has, and the palette doubles as the
    list a per-order validation walks.
  - `class::Array{UInt16,D}` — two bytes per cell, indexing `palette`. An
    anisotropic hp estimator reaches 255 distinct orders on a modest 3D grid
    (an 8³ grid admits 512), so the index is wide enough that the cap is not a
    shape a real p-field runs into; the normaliser still rejects rather than
    wraps.
  - `locals::Array{Vector{CartesianIndex{D}},D}` — per cell, the multi-indices of
    the local basis functions that cell generates, in the canonical
    [`local_basis_indices`](@ref) order. This is *not* `local_basis_indices` at
    the cell's own order: the minimum rule removes the shared-entity modes a
    lower-order neighbour cannot match, and the surviving set is in general not a
    tensor product of per-axis ranges. Cells carrying the same set share one
    vector. Inactive cells carry an empty vector.

The `locals` table is what makes the space C⁰. Every consumer that pairs a cell's
raw dof ids with its local basis functions positionally — the dof walk, the
assembly and transfer workspaces, the Dirichlet boundary trace, field evaluation
— must read this one table, because the pairing is the only link between the dof
layer and the basis layer and nothing checks it at runtime. Go through
[`cell_basis_indices`](@ref) rather than re-deriving the set.
"""
struct CellOrders{D}
    palette::Vector{NTuple{D,Int}}
    class::Array{UInt16,D}
    nominal::NTuple{D,Int}
end

"""
    CellModes{D}

The multi-index sets a level's cells actually generate, after the minimum rule
has trimmed the modes a shared entity cannot carry. Two fields, the same shape
as [`CellOrders`](@ref): `sets` holds the distinct index sets (entry 1 is the
empty set every inactive cell points at) and `kind` maps a cell to one of them.

**This is derived, never supplied.** It is a function of the level's orders, its
mesh, its mode and its activation mask, and it is computed by `Level`'s inner
constructor from exactly those. That is not a stylistic choice: the mask half of
that dependency is what a caller used to be able to get wrong, by handing in a
table built against one mask alongside a different one, and the failure was
silent — the dof layout is built *from* this table, so layout and table went
stale together and nothing downstream could notice. A table that cannot be
passed in cannot be passed in stale.
"""
struct CellModes{D}
    sets::Vector{Vector{CartesianIndex{D}}}
    kind::Array{UInt16,D}
end

# Whether a basis family can carry a polynomial order that varies from cell to
# cell. Defaults to `false` and is answered `true` only by integrated Legendre,
# whose hierarchic 1D modes make a mixed-order interface a pure question of which
# shared-entity modes to generate.
#
# B-splines answer `false`, and the reason is structural rather than a missing
# implementation: the degree is a type parameter of the whole-axis knot vector
# (`BSplineSpace{p}(kv)`), the clamped end multiplicity is `p + 1` so changing p
# on one cell rewrites the vector's endpoints, the 1D dimension is `cells + p` —
# a global count — and a function's support is `p + 1` cells wide, so no set of
# functions belongs to one cell for a per-cell degree to name. There is likewise
# no shared entity for a minimum rule to attach to: continuity at an interior
# knot is C^(p−1), a property of the knot rather than of a face between two cells
# with independent degrees.
#
# The trait is consulted only when the normalised order field is genuinely
# non-uniform. A B-spline level handed a uniform per-cell field is still a
# perfectly well defined level and must keep working.
_supports_cell_order(::BasisFamily) = false
_supports_cell_order(::IntegratedLegendre) = true

# Translate a user-supplied `order` specification into either an `NTuple{D,Int}`
# (uniform — today's representation, and the one the hot path keeps) or a dense
# `Array{NTuple{D,Int},D}` of per-cell orders. Accepted shapes, dispatched on
# type and deliberately parallel to `_normalize_mask`:
#
#   * `Integer`                                 — isotropic, uniform.
#   * `NTuple{D,Integer}`                       — anisotropic, uniform.
#   * `AbstractArray{<:Integer,D}` of cell shape — isotropic per cell.
#   * `AbstractArray{<:NTuple{D,Integer},D}`    — anisotropic per cell.
#   * predicate `(cell_box, cell_index) -> Integer | NTuple{D,Integer}`.
#
# A per-cell array whose entries are all equal collapses back to the uniform
# tuple, so a nominally graded field that happens to be flat costs nothing —
# the same discipline `_apply_mask_update` applies to an all-active mask.
_normalize_order(n::Integer, ::CartesianMesh{D}) where {D} = _axis_int_tuple(n, Val(D), :order)

function _normalize_order(t::NTuple{D,<:Integer}, ::CartesianMesh{D}) where {D}
    return _axis_int_tuple(t, Val(D), :order)
end

function _normalize_order(a::AbstractArray{<:Integer,D}, mesh::CartesianMesh{D}) where {D}
    size(a) == mesh.cells ||
        throw(DimensionMismatch("order shape $(size(a)) does not match mesh cells $(mesh.cells)"))
    return _collapse_order(vec([_axis_int_tuple(a[ci], Val(D), :order) for ci in cell_indices(mesh)]))
end

function _normalize_order(a::AbstractArray{<:NTuple{D,<:Integer},D},
                          mesh::CartesianMesh{D}) where {D}
    size(a) == mesh.cells ||
        throw(DimensionMismatch("order shape $(size(a)) does not match mesh cells $(mesh.cells)"))
    return _collapse_order(vec([_axis_int_tuple(a[ci], Val(D), :order) for ci in cell_indices(mesh)]))
end

function _normalize_order(f::Function, mesh::CartesianMesh{D}) where {D}
    entries = vec([_axis_int_tuple(f(cell_box(mesh, ci), ci), Val(D), :order)
                   for ci in cell_indices(mesh)])
    return _collapse_order(entries)
end

function _normalize_order(_, ::CartesianMesh{D}) where {D}
    throw(ArgumentError("order must be a positive integer, an NTuple{$D,Int}, an array of either " *
                        "shaped like the level's cell grid, or a predicate " *
                        "(cell_box, cell_index) -> order"))
end

# Fold a flat list of per-cell orders back into the level's cell grid, collapsing
# an everywhere-equal field to the uniform tuple it is.
function _collapse_order(entries::Vector{NTuple{D,Int}}) where {D}
    all(==(first(entries)), entries) && return first(entries)
    return entries
end

# The level's *nominal* order: the uniform order, or the per-axis maximum over a
# per-cell field. This is what `Level.order` stores, and it is the reason the
# quadrature-sizing consumers (`_parent_quadrature_counts`,
# `_moment_order_for_region`, `_facet_quadrature_counts`,
# `_surface_quadrature_order`, `_subdivision_counts`) need no change: they size a
# rule from `level.order` and over-integrating is safe where under-integrating is
# not.
_nominal_order(o::NTuple{D,Int}) where {D} = o

function _nominal_order(entries::Vector{NTuple{D,Int}}) where {D}
    out = first(entries)
    for e in entries
        out = ntuple(d -> max(out[d], e[d]), D)
    end
    return out
end

# Build the `CellOrders` record a level carries, or `nothing` when the normalised
# field is uniform. Runs three checks a uniform level gets for free:
#
#   1. the family must declare `_supports_cell_order`;
#   2. every palette entry must be a legal order for the family and mode — this
#      is where the per-*cell* `order ≥ 1` requirement and the `:trunk` isotropy
#      requirement are enforced, since `local_basis_indices` raises on both and a
#      per-level check can no longer see a single offending cell;
#   3. the palette must fit the `UInt16` class index.
#
# This record is what the caller ASKED FOR, and it depends on nothing else — not
# on the mask, not on the mode. What the basis then emits per cell is derived
# from it by `_cell_modes` in the dof layer, which owns the key-incidence rule
# the minimum rule is expressed over, and which `Level`'s constructor calls. The
# forward reference resolves at call time, as `dof_layout`'s references to
# `dirichlet.jl` do.
function _build_cell_orders(::BasisFamily, order::NTuple{D,Int}, mesh::CartesianMesh{D},
                            ::Symbol) where {D}
    return CellOrders{D}([order], fill(UInt16(1), mesh.cells), order)
end

function _build_cell_orders(family::BasisFamily, entries::Vector{NTuple{D,Int}},
                            mesh::CartesianMesh{D}, mode::Symbol) where {D}
    _supports_cell_order(family) ||
        throw(ArgumentError("the $(basis_name(family)) basis family does not support a per-cell " *
                            "polynomial order: its degree belongs to a whole-axis knot vector, not " *
                            "to a cell, so there is no set of basis functions a per-cell order " *
                            "could name and no shared entity for the minimum rule to act on. Use " *
                            "the default integrated-Legendre basis, or pass one order for the level."))
    palette = NTuple{D,Int}[]
    class = Array{UInt16,D}(undef, mesh.cells)
    for (i, ci) in enumerate(cell_indices(mesh))
        o = entries[i]
        k = findfirst(==(o), palette)
        if k === nothing
            length(palette) == typemax(UInt16) &&
                throw(ArgumentError("a level carries at most $(typemax(UInt16)) distinct per-cell " *
                                    "orders, and this one asks for more. That is a palette index " *
                                    "limit, not a statement about your p-field."))
            push!(palette, o)
            k = length(palette)
        end
        class[ci] = UInt16(k)
    end
    for o in palette
        _check_basis_mode(family, mode, o)
        local_basis_indices(family, o, mode)     # raises on an order the family rejects
    end
    return CellOrders{D}(palette, class, _nominal_order(palette))
end

# ── Levels and superposition spaces ───────────────────────────────────────────

"""
    Level

One base or overlay discretization level of a [`Space`](@ref). Carries the
level's mesh, basis family, polynomial-order metadata, basis mode, and an
optional activation mask. Fields:

  - `id::Int` — level identifier, unique within a `Space`. `1` for the
    base level, `2, 3, …` for overlays in the order they were added. In a
    coupled multi-domain model every subdomain's ids are shifted into a
    disjoint block by `_reindex_space_levels`, so only the *ordering* of
    the ids is guaranteed there, not the value `1` for a base level.
  - `role::Symbol` — `:base` or `:overlay`. The base level covers the
    full physical domain; overlay levels live inside it.
  - `mesh::CartesianMesh{D,T}` — the Cartesian mesh that discretizes the
    level's domain.
  - `basis::B` — basis family (see [`BasisFamily`](@ref)). All cells of
    the level share the same family.
  - `orders::CellOrders{D}` — the per-cell polynomial order, as a palette
    of the distinct orders plus one index per cell. A level whose order is
    uniform carries a palette of one entry, so there is no second
    representation and no second code path for the common case. Ask
    [`cell_order`](@ref) for a cell's order and [`nominal_order`](@ref) for
    the per-axis maximum, which is a sizing quantity only.
  - `modes::CellModes{D}` — the multi-index sets those orders actually
    generate, after the minimum rule. **Derived**, never supplied: see
    [`CellModes`](@ref) for why that is the point rather than a detail.
  - `mode::Symbol` — basis index-set mode: `:tensor` (full tensor
    product) or `:trunk` (the Szabó–Babuška trunk space, filtered by
    trunk degree).
  - `mask::Union{Nothing,LevelMask{D}}` — optional per-cell activation
    mask. `nothing` keeps every cell active and is the type-stable
    no-mask hot path; the `Union` is small so the `Level` type stays
    stable across `activate!` / `deactivate!` transitions between the
    masked and unmasked states.
  - `prune_covered::Bool` — internal. Always `true` from the public API;
    see leaf semantics under [`space`](@ref). When `true`, this level sheds every high-order
    mode all of whose incident cells a finer level covers (order
    reduction), and additionally sheds a buried *linear* mode that a
    nested finer level reproduces exactly (dedup). Integrated Legendre
    only; every other family takes the empty generic fallback. See
    `_coverage_constraints` in `dofs.jl` and [`space`](@ref).
"""
struct Level{D,T<:Real,B<:BasisFamily}
    id::Int
    role::Symbol
    mesh::CartesianMesh{D,T}
    basis::B
    orders::CellOrders{D}
    modes::CellModes{D}
    mode::Symbol
    mask::Union{Nothing,LevelMask{D}}
    prune_covered::Bool

    # `modes` is DERIVED, and deriving it here is the whole point. It depends on
    # the orders, the mesh, the mode and the mask; when it was a constructor
    # argument a caller could hand in a table built against one mask alongside a
    # different one, and the failure was silent — `_build_cell_dofs!` builds the
    # dof layout *from* the table, so layout and table went stale together and
    # nothing downstream could see it. Measured, a fold carrying its pre-fold
    # table dropped 24 of 780 unknowns while reporting `symmetry_residual = 0.0`
    # and a 1e-16 residual. That mismatch used to be guarded by a witness field
    # recording which mask the table was built against; a guard is a confession
    # that the invariant can be violated. Computing the table from the arguments
    # that determine it makes the mismatch unspellable instead, and the seven
    # construction sites stop having to remember anything.
    function Level{D,T,B}(id, role, mesh, basis, orders, mode, mask,
                          prune_covered) where {D,T<:Real,B<:BasisFamily}
        return new{D,T,B}(id, role, mesh, basis, orders,
                          _cell_modes(basis, orders, mesh, mode, mask), mode, mask, prune_covered)
    end
end

"""
    nominal_order(level::Level) -> NTuple{D,Int}

The per-axis **maximum** order over the level's cells — a sizing quantity, and
nothing else.

It is spelled out rather than stored under the name `order` because the two are
not interchangeable and the difference is invisible at the call site. Reading a
level-wide maximum where a per-cell order was meant is a defect this package has
shipped three times: the immersed-surface rule sized by the loudest cell anywhere
(measured 8.34× on `assemble!`), `write_vtk` subdividing every region at the
maximum (200× in time, 91× on disk), and an error estimator enriching the whole
domain to one above the hottest cell (3.6× end to end). Each was found by
measurement rather than by the compiler, because the read looked correct.

Use it for buffer lengths and 1D factor tables, where over-sizing is safe. For
anything that integrates, evaluates or subdivides, the question you want is
[`cell_order`](@ref).
"""
nominal_order(level::Level) = level.orders.nominal

"""
    cell_order(level::Level, cell::CartesianIndex) -> NTuple{D,Int}

Per-axis polynomial order of one cell of `level`. A level whose order is uniform
carries a palette of one entry, so this is the same lookup either way and there
is no second code path for the common case.
"""
@inline function cell_order(level::Level{D}, cell::CartesianIndex{D}) where {D}
    return level.orders.palette[level.orders.class[cell]]
end

"""
    cell_basis_indices(level::Level, cell::CartesianIndex) -> Vector{CartesianIndex{D}}

Multi-indices of the local basis functions cell `cell` of `level` generates, in
the canonical [`local_basis_indices`](@ref) order.

On a uniform level this is `local_basis_indices(level.basis, nominal_order(level),
level.mode)` and allocates a fresh vector per call, exactly as the dof walk has
always done. On a level carrying a per-cell order it is a lookup into the level's
precomputed minimum-rule table and allocates nothing — and it is *not* the index
set at the cell's own order: the modes a lower-order neighbour cannot match are
gone, which is what keeps the space C⁰ across an order jump.

This is the one place the per-cell index set is defined. Every consumer that
pairs `cell_dofs(layout, level, cell)` with basis values positionally must obtain
its multi-indices here, or from a bank filled from `level.modes`.
"""
@inline function cell_basis_indices(level::Level{D}, cell::CartesianIndex{D}) where {D}
    return level.modes.sets[level.modes.kind[cell]]
end

# Basis values / physical gradients of exactly the functions cell `cell` of
# `level` generates, at reference point `xi`. The family-level `basis_values` and
# `physical_basis_gradients` take an order and re-derive the index set from it,
# which is the wrong set on a per-cell-order level — the minimum rule's kept set
# is not the index set of any order. These two take the cell instead, so the
# values they return pair positionally with `cell_dofs(layout, level.id, cell)`.
#
# Used by the point-evaluation paths in `postprocessing.jl`; the assembly hot
# loops go through the workspace banks instead and never allocate here.
function _cell_basis_values(level::Level{D}, cell::CartesianIndex{D}, xi::PointLike{D}) where {D}
    indices = cell_basis_indices(level, cell)
    ξ = _reference_coordinate(xi)
    R = eltype(ξ)
    values = Vector{R}(undef, length(indices))
    p = cell_order(level, cell)
    return _tensor_values!(level.basis, values, indices, p, ξ, _factor_buffers(p, R), cell)
end

function _cell_basis_gradients(level::Level{D}, cell::CartesianIndex{D}, cell_box::AxisBox{D,T},
                               xi::PointLike{D}) where {D,T}
    indices = cell_basis_indices(level, cell)
    R = float(promote_type(map(typeof, Tuple(xi))..., T))
    ξ = SVector{D,R}(xi)
    values = Vector{R}(undef, length(indices))
    gradients = Vector{SVector{D,R}}(undef, length(indices))
    scale = SVector{D,R}(2 ./ edge_lengths(cell_box))
    p = cell_order(level, cell)
    _tensor_values_grads!(level.basis, values, gradients, indices, p, ξ, scale,
                          _factor_buffers(p, R), _factor_buffers(p, R), cell)
    return gradients
end

# The per-cell minimum-rule table of a level, or `nothing` when the level is
# uniform and every cell shares the level-wide index list. Assembly and transfer
# workspaces bank this per level id so a hot loop pays one `=== nothing` branch
# instead of a per-cell lookup on the common path.
_cell_locals(level::Level) = level.modes

"""
    Space

Superposition space over one base level and zero or more overlay levels.
Built incrementally via [`space`](@ref) and [`overlay`](@ref).

The approximation it represents is the sum of overlay contributions

    u_h(x) = Σₖ u_h^(k)(x),

where each overlay contribution `u_h^(k)` is extended by zero outside its
overlay domain `Ω^(k)`. The `levels` field is a heterogeneous tuple so
each level can carry its own basis-family type without losing type
stability.

Fields:

  - `domain::AxisBox{D,T}` — the physical bounding box. Every overlay
    level's mesh domain must lie inside this box.
  - `levels::L` — tuple of `Level{D,T,…}`, base first then overlays in
    insertion order.
  - `physical::Union{Nothing,PhysicalDomain}` — optional level-set
    description of the physical domain `Ω`. `nothing` keeps the
    no-FCM hot path: integration is over the bounding box `domain` and
    every cell is `:full`. With a `PhysicalDomain`, cut and fictitious
    cells are handled per the [`PhysicalDomain`](@ref) docstring.
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

# Linear scan over `V.levels` resolving a level id to its `Level` value.
# Linear is fine because typical problems carry only a handful of levels;
# downstream callers cache the resolved levels when they iterate hot loops.
function _level_by_id(V::Space, id::Integer)
    for level in V.levels
        level.id == id && return level
    end
    throw(ArgumentError("unknown level id $id"))
end

# ── Basis-family hooks ────────────────────────────────────────────────────────

"""
    instantiate_basis(basis, mesh, order, mode, mask) -> BasisFamily

Finalize a basis family for a concrete level, just before the
[`Level`](@ref) is built. The default returns `basis` unchanged: most
families — including the default [`IntegratedLegendre`](@ref) — are fully
specified before any mesh exists.

Families whose concrete form depends on the level's mesh overload this
hook. The B-spline family in the `BasicBSpline` extension is the
motivating case: its per-axis knot vectors are built from the mesh cell
coordinates, so [`bspline`](@ref) returns a *deferred* specification and
the real family is materialised here once the mesh is known. Keeping the
two-phase "spec → concrete family" handshake in one named hook (rather
than buried in a `Level` constructor overload) makes the extension point
explicit and means every level-building path treats mesh-dependent
families uniformly.

All of [`space`](@ref), [`overlay`](@ref), [`moved_space`](@ref), and the
mask mutators route the basis through this hook, so a mesh-dependent
family is rebuilt whenever the mesh changes (e.g. an overlay move) and is
validated against the level's `order` and `mode` at construction time.
`mask` is the normalised [`LevelMask`](@ref) (`nothing` for an all-active
level); it is passed so a family whose concrete form depends on where the
active region ends can use it, and the shipped B-spline family — which
handles arbitrary mask geometry through linear constraints instead —
accepts every mask.
"""
function instantiate_basis(basis::BasisFamily, mesh::CartesianMesh{D,T}, order::NTuple{D,Int},
                           mode::Symbol, mask) where {D,T}
    basis
end

# Whether a basis family supports an immersed `PhysicalDomain` — the α-FCM fold,
# cut-cell moment-fit quadrature, and the fictitious-fold exemption that keeps a
# cut cell's boundary modes on its fully-fictitious fold faces. Defaults to
# `true`, and both shipped families answer it: the exemption is the only piece a
# family has to get right, and each has it. A family whose constraint generator
# lacks one must override this to `false` rather than degrade the FCM solution
# silently.
_supports_physical_domain(::BasisFamily) = true

# Reject a `PhysicalDomain` on a basis family that cannot integrate it yet.
# Called from every space-building entry point once the concrete family exists.
function _check_physical_basis(family::BasisFamily, physical)
    physical === nothing ||
        _supports_physical_domain(family) ||
        throw(ArgumentError("the $(basis_name(family)) basis family does not yet support an immersed physical " *
                            "domain (finite-cell method): its overlay constraints would over-constrain cut-cell " *
                            "modes on fold faces. Use the default integrated-Legendre basis for FCM problems."))
    return nothing
end

# ── Space construction ────────────────────────────────────────────────────────

"""
    space(domain::AxisBox; cells, order=1, basis=IntegratedLegendre(),
                          mode=:tensor, active=nothing, physical=nothing,
                          )

Build the base discretization of a [`Space`](@ref) over `domain`. The
resulting space has one base level (`role = :base`, `id = 1`); add overlay
levels with [`overlay`](@ref).

Keyword arguments:

  - `cells` — base mesh resolution. Either a single positive integer
    (replicated across axes) or an `NTuple{D,Int}` of per-axis cell
    counts.
  - `order` — polynomial order of the basis, uniform over the level or
    varying from cell to cell. Accepted shapes:

      * a single positive integer (isotropic, uniform);
      * an `NTuple{D,Int}` (anisotropic, uniform; requires `mode = :tensor`);
      * an `AbstractArray{<:Integer,D}` shaped exactly like the level's cell
        grid (isotropic per cell);
      * an `AbstractArray{<:NTuple{D,Integer},D}` of the same shape
        (anisotropic per cell);
      * a predicate `(cell_box, cell_index) -> order` returning either of the
        uniform shapes, evaluated once per cell.

    A per-cell field whose entries are all equal collapses back to the uniform
    case, so the last three shapes cost nothing when they happen to be flat.
    Where two cells of different order share a face, the shared entity carries
    the *minimum* of the two orders (the classical hp-FEM minimum rule), which
    is what keeps the space C⁰ — see [`CellModes`](@ref). `nominal_order` then
    reports the per-axis maximum, and every cell keeps `order ≥ 1` on every
    axis, which is checked per cell rather than per level.

    Per-cell order is a basis-family capability: only integrated Legendre
    declares it, and a non-uniform field on any other family raises
    `ArgumentError` rather than being silently ignored.
  - `basis` — basis family. Defaults to [`IntegratedLegendre`](@ref).
  - `mode` — basis index set. `:tensor` is the full tensor product;
    `:trunk` is the Szabó–Babuška trunk space (filtered by trunk degree)
    and requires isotropic `order`.
  - `active` — per-cell activation mask, `nothing` (default) meaning every
    cell is active. The accepted shapes, all normalised to a
    [`LevelMask`](@ref) over the level's cells, are:

      * a `LevelMask{D}`, or an `AbstractArray{Bool,D}` shaped exactly like
        the level's cell grid, `true` where the cell participates. Both are
        copied, so mutating the argument afterwards does not reach the level;
      * a predicate `(cell_box, cell_index) -> Bool` evaluated once per
        cell, with `cell_box` the cell's [`AxisBox`](@ref) and `cell_index`
        its `CartesianIndex{D}`;
      * an iterable of `CartesianIndex{D}` listing the active cells — every
        cell not listed is inactive.

    Inactive cells are dropped from dof enumeration, and faces between
    active and inactive cells of one level are treated like that level's
    artificial overlay boundary; see [`LevelMask`](@ref).
  - `physical` — optional [`PhysicalDomain`](@ref) describing an
    immersed `Ω ⊂ domain`. With `nothing` (default) the bounding box is
    the physical domain. Not every basis family can integrate one: the
    call raises `ArgumentError` for a family that declares
    `_supports_physical_domain` false. Both shipped families declare it
    true.
  **Leaf semantics.** A cell carries basis functions only where it is a
  *leaf* — where no finer level has taken the region over. A covered cell
  is the parent of a leaf, and a parent carries no unknowns where its
  children do. This is not an option and there is nothing to configure:
  it is what makes superposition equivalent to ordinary refinement, in
  which h-refining a cell *replaces* it rather than adding to it.

  Concretely a level sheds a mode wherever a *single* finer level has
  taken over the region that mode lives on. On an integrated Legendre
  level there are two eliminations, both per-mode rather than per-cell,
  and both asking first that the mode be *buried* — every cell it is
  incident to covered by one finer level:

      * a buried **high-order** mode (one with at least one bubble axis)
        is dropped, so a fully covered cell keeps only its linear
        skeleton. An edge or face mode straddling the boundary of the
        covered region survives, because its own stencil is not buried.
      * a buried **linear** mode is dropped when one finer level
        reproduces it exactly, which takes both halves of a test: that
        level's mesh must refine this one's (`_nested_over` in
        `src/coverage.jl`) *and* its basis must be integrated Legendre.
        A buried vertex function is a C⁰ hat, and a basis smoother than
        C⁰ across its own cell boundaries — a B-spline of degree ≥ 2 —
        carries nothing that reproduces the kink.

  Leaving a covered mode active makes the superposed operator singular,
  because it and the covering level's reproduction of it are linearly
  dependent: 169 to 1378 null directions were measured on nested ladders
  with the rule switched off.

  The high-order half does not require the covering level to reproduce
  what it displaces, and that is deliberate rather than an oversight. It
  is what lets a low-order cover sit under a high-order base and behave
  the way small low-order elements behave near a singularity in any hp
  code: the covered region is then resolved at the cover's order, which
  is what asking for a low-order cover *means*. Once the cover carries at
  least the base's order the elimination is exactly lossless — measured,
  the reduced and unreduced errors agree to every printed digit from
  cover order 4 upward under a p=5 base.

    A B-spline level has no bubble/skeleton split and so has only the
    second elimination: a buried function is dropped exactly when a nested
    level above reproduces it, which for that family is not an accuracy
    trade at all but the thing that keeps a nested stack non-singular —
    see `bspline` in the `BasicBSpline` extension. Any further family
    takes the generic `_coverage_constraints` fallback, which returns no
    constraints, so on such a space the default `true` is a no-op. See
    `src/coverage.jl` and `_coverage_constraints` in `src/dofs.jl`;
    `diagnostics(...).reduced_mode_counts` reports the per-level count.
"""
function space(domain::AxisBox{D,T}; cells, order=1, basis=IntegratedLegendre(),
               mode::Symbol=:tensor, active=nothing, physical=nothing) where {D,T}
    base_mesh = CartesianMesh(domain; cells)
    field = _normalize_order(order, base_mesh)
    orders = _nominal_order(field)
    _check_basis_mode(mode, orders)
    mask = _normalize_mask(active, base_mesh)
    family = instantiate_basis(basis, base_mesh, orders, mode, mask)
    _check_physical_basis(family, physical)
    cell_orders = _build_cell_orders(family, field, base_mesh, mode)
    base_level = Level{D,T,typeof(family)}(1, :base, base_mesh, family, cell_orders, mode, mask,
                                           true)
    return Space{D,T,Tuple{typeof(base_level)}}(domain, (base_level,), physical)
end

"""
    overlay(V::Space, domain::AxisBox; cells, order=nominal_order(V.levels[1]),
                                       basis=V.levels[1].basis,
                                       mode=V.levels[1].mode,
                                       tolerance=GeometryTolerance(T),
                                       active=nothing) -> Space

Add an overlay level on the sub-box `domain` to an existing [`Space`](@ref)
`V`, returning the extended space. The overlay's domain must lie inside
`V.domain` up to `tolerance.contain` (see [`is_inside`](@ref)).

Defaults inherit from the base level so a quick overlay only needs `cells`
and possibly `order`. Keyword arguments:

  - `cells` — per-axis cell counts for the overlay mesh.
  - `order` — polynomial order; defaults to the base level's nominal order.
    Takes every shape [`space`](@ref)'s `order` takes, including the per-cell
    ones, resolved against *this* overlay's cell grid.
  - `basis` — basis family; defaults to the base level's family.
  - `mode` — basis index set; defaults to the base level's mode.
  - `tolerance` — slack on the inside-domain check.
  - `active` — optional per-cell mask, same shapes as [`space`](@ref)'s
    `active`.
  Leaf semantics apply to this overlay exactly as to any level: where it
  is itself covered by something finer it carries no unknowns, and where
  it covers the level below, that level carries none. See [`space`](@ref).

The new level's `id` is `length(V.levels) + 1`. Overlay placement is
independent of any existing overlay: overlay boundaries need not coincide
with lower-level element boundaries, and there is no topological merging
with parents (every overlay imposes homogeneous Dirichlet data on its
artificial boundary, see `CONTRIBUTING.md`'s "The superposition model"
section).
"""
function overlay(V::Space{D,T}, domain::AxisBox{D,T}; cells, order=nominal_order(V.levels[1]),
                 basis=V.levels[1].basis, mode::Symbol=V.levels[1].mode,
                 tolerance=GeometryTolerance(T), active=nothing) where {D,T}
    is_inside(domain, V.domain, tolerance) ||
        throw(ArgumentError("overlay domain must lie inside the physical domain"))
    overlay_mesh = CartesianMesh(domain; cells)
    field = _normalize_order(order, overlay_mesh)
    orders = _nominal_order(field)
    _check_basis_mode(mode, orders)
    mask = _normalize_mask(active, overlay_mesh)
    id = length(V.levels) + 1
    family = instantiate_basis(basis, overlay_mesh, orders, mode, mask)
    _check_physical_basis(family, V.physical)
    cell_orders = _build_cell_orders(family, field, overlay_mesh, mode)
    level = Level{D,T,typeof(family)}(id, :overlay, overlay_mesh, family, cell_orders, mode, mask,
                                      true)
    levels = (V.levels..., level)
    return Space{D,T,typeof(levels)}(V.domain, levels, V.physical)
end

# Rebuild `V` with leaf semantics DISABLED on every level, so a covered cell keeps
# the modes it would otherwise shed. Internal, and deliberately not reachable from
# the public API: the resulting space is never what a caller wants. On a stack
# whose covers are aligned it is exactly singular — the coarse modes and the fine
# level's reproductions of them are linearly dependent, and 169 to 1378 null
# directions were measured on nested ladders — and where it is not singular it
# costs 11–25% more unknowns for an advantage that decays to 1.02× as the cover
# refines.
#
# It exists for one reason: it is the reference the losslessness assertion is made
# against. `size(gram(V), 1) == rank(gram(_unpruned(V)))` says the elimination
# removed redundancy and nothing else, and that assertion is the only instrument
# that can see a wrong elimination — the reduced operator stays full rank at a
# healthy condition number either way.
function _unpruned(V::Space{D,T}) where {D,T}
    levels = map(V.levels) do l
        Level{D,T,typeof(l.basis)}(l.id, l.role, l.mesh, l.basis, l.orders, l.mode, l.mask, false)
    end
    return Space{D,T,typeof(levels)}(V.domain, levels, V.physical)
end

"""
    moved_space(V::Space; level, to, tolerance=GeometryTolerance(T)) -> Space

Return a copy of `V` with overlay `level` (its `level`-th level, which must
be an overlay) relocated to the axis box `to`, keeping the same cell
counts, basis family, order, level id, and activation mask. The base level
(level 1) cannot be moved. This is the space-level building block behind
[`move!`](@ref) and [`moved`](@ref).
"""
function moved_space(V::Space{D,T}; level::Integer, to::AxisBox{D,T},
                     tolerance=GeometryTolerance(T)) where {D,T}
    1 < level <= length(V.levels) || throw(ArgumentError("only overlay levels can be moved"))
    is_inside(to, V.domain, tolerance) ||
        throw(ArgumentError("overlay domain must lie inside the physical domain"))
    old = V.levels[level]
    new_mesh = CartesianMesh(to; cells=old.mesh.cells)
    # Re-instantiate the family against the moved mesh: mesh-dependent
    # families (B-splines) must rebuild their knot vectors for the new
    # cell coordinates; mesh-independent families return themselves.
    family = instantiate_basis(old.basis, new_mesh, nominal_order(old), old.mode, old.mask)
    # The order field is indexed by cell and a move keeps the cell grid identical
    # — only the coordinates change — so it carries over verbatim.
    new_level = Level{D,T,typeof(family)}(old.id, old.role, new_mesh, family, old.orders, old.mode,
                                          old.mask, old.prune_covered)
    levels = ntuple(i -> i == level ? new_level : V.levels[i], length(V.levels))
    return Space{D,T,typeof(levels)}(V.domain, levels, V.physical)
end

# Rebuild `V` with the mask of one level replaced. Mirrors `moved_space`
# but keeps the mesh / basis / order fixed and swaps only the mask. Used by
# the `activate!` / `deactivate!` model-level mutators in `assembly.jl`.
function _remasked_space(V::Space{D,T}, level_index::Integer, mask) where {D,T}
    1 <= level_index <= length(V.levels) ||
        throw(ArgumentError("level index $level_index out of bounds"))
    old = V.levels[level_index]
    family = instantiate_basis(old.basis, old.mesh, nominal_order(old), old.mode, mask)
    # A mask edit changes which shared-entity modes survive. Nothing is rebuilt
    # here: the order field is the caller's and is unaffected, and the constructor
    # re-derives the modes from the new mask.
    new_level = Level{D,T,typeof(family)}(old.id, old.role, old.mesh, family, old.orders, old.mode,
                                          mask, old.prune_covered)
    levels = ntuple(i -> i == level_index ? new_level : V.levels[i], length(V.levels))
    return Space{D,T,typeof(levels)}(V.domain, levels, V.physical)
end

# Rebuild `V` with the polynomial order of one level replaced. Mirrors
# `_remasked_space` but keeps the mesh / basis / mask fixed and swaps only the
# order field, so the level's type — and with it the `Space` type and the
# compiled assembly pipeline — is unchanged. Used by the `elevate` verb in
# `ladder.jl`.
function _reordered_space(V::Space{D,T}, level_index::Integer, spec) where {D,T}
    1 <= level_index <= length(V.levels) ||
        throw(ArgumentError("level index $level_index out of bounds"))
    old = V.levels[level_index]
    field = _normalize_order(_normalize_elevate_spec(V, Int(level_index), spec), old.mesh)
    orders = _nominal_order(field)
    _check_basis_mode(old.mode, orders)
    family = instantiate_basis(old.basis, old.mesh, orders, old.mode, old.mask)
    cell_orders = _build_cell_orders(family, field, old.mesh, old.mode)
    new_level = Level{D,T,typeof(family)}(old.id, old.role, old.mesh, family, cell_orders, old.mode,
                                          old.mask, old.prune_covered)
    levels = ntuple(i -> i == level_index ? new_level : V.levels[i], length(V.levels))
    return Space{D,T,typeof(levels)}(V.domain, levels, V.physical)
end

# ── Fictitious-cell fold ──────────────────────────────────────────────────────

# Per-cell classification of a level's mesh against `physical`. Returns a
# `BitArray{D}` marking the cells the classifier reports as fully outside
# Ω (`:fictitious`). Cells classified as `:full` or `:cut` are kept active:
# `:cut` cells will get their quadrature replaced by the NNMF moment-fit
# rule at region-build time (see `intersections.jl`), while `:fictitious`
# cells are dropped from the dof layout via the fold below — at every α,
# unless `keep_fictitious` asks for the classic α-FCM fill instead.
#
# Only cells the level's own mask already keeps active are classified. The fold
# below combines the two masks with `active .& .!fictitious` (`_fold_fictitious`),
# so a cell the user already deactivated stays deactivated whatever the classifier
# would say about it: `false & x == false`. Leaving `fict` at its `false` default
# there is therefore bit-identical to classifying it, and the guard turns the pass
# from O(cells of the level) into O(active cells of the level).
#
# That distinction is what makes a deep, sparsely-populated overlay affordable.
# A refinement level whose grid is Nᴰ cells but which carries a handful of active
# ones — the normal state of an adaptive front — otherwise pays a full-grid
# classification sweep every time the mesh is folded, and `classify_cell` is not
# cheap: it walks a bounded octree whenever the Lipschitz certificate cannot
# settle the box from its centre sample.
#
# The classification verdicts of *inactive* cells are still needed further down
# the pipeline — `_covered_by_level` in `coverage.jl` and `_internal_face_is_physical`
# in `dofs.jl` both have to tell a fictitious fold from a user mask — but each asks
# only about the cells on its own fold/mask boundary, and each goes through the same
# `cache`. Skipping them here turns those queries from cache hits into cache misses;
# it does not change a single verdict, because `classify_cell` is a pure function of
# the box and the geometry.
function _level_fictitious_cells(level::Level{D,T}, physical::PhysicalDomain,
                                 cache::_ClassifyCache{D,T}) where {D,T}
    fict = falses(level.mesh.cells)
    for ci in cell_indices(level.mesh)
        is_active(level.mask, ci) || continue
        if classify_cell(physical, cell_box(level.mesh, ci), cache) === :fictitious
            fict[ci] = true
        end
    end
    return fict
end

# Combine the user-supplied mask with the fictitious mask. A cell is
# deactivated when *either* source asks for it: the user explicitly
# excluded it via `active=`, or the geometry classifier reports the cell
# entirely outside Ω. A `nothing` user mask is treated as "all active".
# The result collapses back to `nothing` when the fold leaves every cell
# active, preserving the no-mask hot path.
function _fold_fictitious(::Nothing, fictitious::BitArray{D}) where {D}
    any(fictitious) || return nothing
    return LevelMask{D}(.!fictitious)
end

function _fold_fictitious(user::LevelMask{D}, fictitious::BitArray{D}) where {D}
    !any(fictitious) && return user
    return LevelMask{D}(user.on .& .!fictitious)
end

# Apply the PhysicalDomain fold to `V`. For each level, classifies cells
# against `V.physical` and merges the resulting fictitious mask into the
# level's user mask, so cells fully outside Ω are dropped from the dof layout.
# This happens *regardless of α* — the α-FCM stabilization enriches cut-cell
# quadrature (in `intersections.jl`), it does not keep whole fictitious cells
# around. Returns `V` unchanged only when there is no physical domain, or when
# `keep_fictitious` opts into the classic α-FCM treatment (fully-fictitious
# cells kept active with α-scaled quadrature). The `_ClassifyCache` carries the
# per-box classification verdicts so the region-level dispatcher in
# `intersections.jl` can reuse them.
#
# Contract: `V` must be a *pre-fold* space — one whose level masks still mean
# "what the caller asked for". The fold is an intersection
# (`active .& .!fictitious`) and is not invertible, so applying it to its own
# output can only remove cells: a cell dropped as fictitious at one overlay
# position stays dropped after the overlay moves somewhere it would be inside
# Ω. Callers that fold repeatedly over a model's lifetime must therefore keep
# the pre-fold space and re-derive from it — see `Model.prefold_space` and
# `_moved_problem` in `model.jl`.
function _apply_physical_fold(V::Space{D,T}, cache::_ClassifyCache{D,T}) where {D,T}
    V.physical === nothing && return V
    V.physical.keep_fictitious && return V

    new_levels = ntuple(length(V.levels)) do i
        level = V.levels[i]
        fict = _level_fictitious_cells(level, V.physical, cache)
        if any(fict)
            new_mask = _fold_fictitious(level.mask, fict)
            Level{D,T,typeof(level.basis)}(level.id, level.role, level.mesh, level.basis,
                                           level.orders, level.mode, new_mask, level.prune_covered)
        else
            level
        end
    end
    return Space{D,T,typeof(new_levels)}(V.domain, new_levels, V.physical)
end

# Convenience overload: allocate a one-shot classification cache, fold
# once, discard. Used by `space(...)` and the activation mutators when
# nobody is going to reuse the cache.
_apply_physical_fold(V::Space{D,T}) where {D,T} = _apply_physical_fold(V, _ClassifyCache{D,T}())

# ── Level-id reindexing ───────────────────────────────────────────────────────

# Rebuild `V` with every level id shifted by `offset`, giving the space a
# disjoint level-id block `[offset+1, offset+level_count]`.
#
# Level ids are unique only *within* one space (base = 1, overlays 2, 3, …).
# When several independent discretisations are coupled in one multi-domain
# model, the assembly workspace banks basis values in a single flat vector
# indexed by level id, and each field's dof layout keys `cell_dofs` by the
# same id — so before those are built, each participating space is reindexed
# into its own contiguous id block. `offset == 0` returns `V` unchanged (the
# single-domain fast path). Positional level addressing (`move!` / `activate!`,
# which index `V.levels` by position) is unaffected; `_level_by_id` scans by
# id and works with any id assignment.
function _reindex_space_levels(V::Space{D,T}, offset::Int) where {D,T}
    offset == 0 && return V
    new_levels = ntuple(length(V.levels)) do i
        lvl = V.levels[i]
        Level{D,T,typeof(lvl.basis)}(lvl.id + offset, lvl.role, lvl.mesh, lvl.basis, lvl.orders,
                                     lvl.mode, lvl.mask, lvl.prune_covered)
    end
    return Space{D,T,typeof(new_levels)}(V.domain, new_levels, V.physical)
end

# ── Mask updates ──────────────────────────────────────────────────────────────

# Flip cells in `on` selected by `cells` to `value`. Accepts the same
# shapes as `_normalize_mask`'s selector branches: `AbstractArray{Bool,D}`,
# predicate `(cell_box, cell_index) -> Bool`, or iterable of
# `CartesianIndex{D}`. For Bool-array selectors we iterate the `findall`
# index list — `O(#selected)` instead of `O(#cells)`, important when the
# selector is sparse on a large mesh.
function _flip_cells!(on::BitArray{D}, bits::AbstractArray{Bool,D}, mesh::CartesianMesh{D},
                      value::Bool) where {D}
    size(bits) == size(on) ||
        throw(DimensionMismatch("selector shape $(size(bits)) does not match mesh cells $(mesh.cells)"))
    for ci in findall(bits)
        on[ci] = value
    end
    return on
end

function _flip_cells!(on::BitArray{D}, f::Function, mesh::CartesianMesh{D}, value::Bool) where {D}
    for ci in cell_indices(mesh)
        f(cell_box(mesh, ci), ci)::Bool && (on[ci] = value)
    end
    return on
end

function _flip_cells!(on::BitArray{D}, cells, mesh::CartesianMesh{D}, value::Bool) where {D}
    for ci in cells
        ci isa CartesianIndex{D} ||
            throw(ArgumentError("cells iterable entries must be CartesianIndex{$D}; got $(typeof(ci))"))
        checkbounds(Bool, on, ci) ||
            throw(ArgumentError("cell index $ci is out of bounds for mesh cells $(mesh.cells)"))
        on[ci] = value
    end
    return on
end

# Build the new mask after activating or deactivating `cells` on a level
# whose current mask is `old`. Starts from the current active set
# (all-true if `old === nothing`), flips the selected cells, and collapses
# an all-active result back to `nothing` so the no-mask hot path survives
# bouncing in and out of the mask. `_flip_cells!` always runs (even when
# the starting mask is all-on) so out-of-bounds and wrong-typed selectors
# raise the same error regardless of the starting mask state.
function _apply_mask_update(old::Union{Nothing,LevelMask{D}}, mesh::CartesianMesh{D}, cells,
                            value::Bool) where {D}
    on = old === nothing ? trues(mesh.cells) : copy(old.on)
    _flip_cells!(on, cells, mesh, value)
    return all(on) ? nothing : LevelMask{D}(on)
end

# ── Display ───────────────────────────────────────────────────────────────────

function Base.show(io::IO, level::Level)
    print(io, "Level(id=", level.id, ", role=:", level.role, ", cells=", level.mesh.cells,
          ", order=", nominal_order(level), ", mode=:", level.mode, ", basis=:",
          basis_name(level.basis))
    # A per-cell field is summarised by its palette size, never dumped: a graded
    # level carries one order per cell and printing them all is unreadable.
    if level.orders !== nothing
        print(io, ", cell_orders=", length(level.orders.palette))
    end
    if level.mask !== nothing
        print(io, ", active=", count(level.mask.on), "/", length(level.mask.on))
    end
    print(io, ")")
end

function Base.show(io::IO, V::Space{D}) where {D}
    print(io, "Space(D=", D, ", levels=", length(V.levels), ", domain=", V.domain, ")")
end
