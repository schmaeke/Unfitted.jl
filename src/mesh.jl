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
# `domain.upper[d]`. The explicit form (rather than `range(...)`) keeps the
# endpoints bit-exact at `domain.lower[d]` and `domain.upper[d]`, which the
# `Base.:(==)` comparison on `AxisBox` corner coordinates relies on.
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

"""
    Level

One base or overlay discretization level of a [`Space`](@ref). Carries the
level's mesh, basis family, polynomial-order metadata, basis mode, and an
optional activation mask. Fields:

  - `id::Int` — level identifier, unique within a `Space`. `1` for the
    base level, `2, 3, …` for overlays in the order they were added.
  - `role::Symbol` — `:base` or `:overlay`. The base level covers the
    full physical domain; overlay levels live inside it.
  - `mesh::CartesianMesh{D,T}` — the Cartesian mesh that discretizes the
    level's domain.
  - `basis::B` — basis family (see [`BasisFamily`](@ref)). All cells of
    the level share the same family.
  - `order::NTuple{D,Int}` — per-axis polynomial order. May be
    anisotropic for `mode = :tensor`; must be isotropic for
    `mode = :trunk`.
  - `mode::Symbol` — basis index-set mode: `:tensor` (full tensor
    product) or `:trunk` (the Szabó–Babuška trunk space, filtered by
    trunk degree).
  - `mask::Union{Nothing,LevelMask{D}}` — optional per-cell activation
    mask. `nothing` keeps every cell active and is the type-stable
    no-mask hot path; the `Union` is small so the `Level` type stays
    stable across `activate!` / `deactivate!` transitions between the
    masked and unmasked states.
"""
struct Level{D,T<:Real,B<:BasisFamily}
    id::Int
    role::Symbol
    mesh::CartesianMesh{D,T}
    basis::B
    order::NTuple{D,Int}
    mode::Symbol
    mask::Union{Nothing,LevelMask{D}}
end

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
validated against the level's `order` / `mode` / `mask` at construction
time. `mask` is the normalised [`LevelMask`](@ref) (`nothing` for an
all-active level).
"""
function instantiate_basis(basis::BasisFamily, mesh::CartesianMesh{D,T}, order::NTuple{D,Int},
                           mode::Symbol, mask) where {D,T}
    basis
end

# Whether a basis family supports an immersed `PhysicalDomain` — the α-FCM fold,
# cut-cell moment-fit quadrature, and the fictitious-fold C⁰ constraint rule that
# keeps a cut cell's boundary modes on its fully-fictitious fold faces. Defaults
# to `true`. The B-spline extension overrides it to `false`: its overlay-
# constraint generator emits trace-vanishing constraints on every active/inactive
# face with no fictitious-fold exemption, so it would over-constrain cut-cell
# modes on fold faces (silently degrading the FCM solution).
_supports_physical_domain(::BasisFamily) = true

# Reject a `PhysicalDomain` on a basis family that cannot integrate it yet.
# Called from every space-building entry point once the concrete family exists.
function _check_physical_basis(family::BasisFamily, physical)
    physical === nothing || _supports_physical_domain(family) ||
        throw(ArgumentError(
            "the $(basis_name(family)) basis family does not yet support an immersed physical " *
            "domain (finite-cell method): its overlay constraints would over-constrain cut-cell " *
            "modes on fold faces. Use the default integrated-Legendre basis for FCM problems."))
    return nothing
end

"""
    space(domain::AxisBox; cells, order=1, basis=IntegratedLegendre(),
                          mode=:tensor, active=nothing, physical=nothing)

Build the base discretization of a [`Space`](@ref) over `domain`. The
resulting space has one base level (`role = :base`, `id = 1`); add overlay
levels with [`overlay`](@ref).

Keyword arguments:

  - `cells` — base mesh resolution. Either a single positive integer
    (replicated across axes) or an `NTuple{D,Int}` of per-axis cell
    counts.
  - `order` — polynomial order of the basis. Either a single positive
    integer (isotropic) or an `NTuple{D,Int}` (anisotropic, requires
    `mode = :tensor`).
  - `basis` — basis family. Defaults to [`IntegratedLegendre`](@ref).
  - `mode` — basis index set. `:tensor` is the full tensor product;
    `:trunk` is the Szabó–Babuška trunk space (filtered by trunk degree)
    and requires isotropic `order`.
  - `active` — per-cell activation mask. See
    [`LevelMask`](@ref) and [`_normalize_mask`](@ref) for the accepted
    shapes.
  - `physical` — optional [`PhysicalDomain`](@ref) describing an
    immersed `Ω ⊂ domain`. With `nothing` (default) the bounding box is
    the physical domain.
"""
function space(domain::AxisBox{D,T}; cells, order=1, basis=IntegratedLegendre(),
               mode::Symbol=:tensor, active=nothing, physical=nothing) where {D,T}
    orders = _axis_int_tuple(order, Val(D), :order)
    _check_basis_mode(mode, orders)
    base_mesh = CartesianMesh(domain; cells)
    mask = _normalize_mask(active, base_mesh)
    family = instantiate_basis(basis, base_mesh, orders, mode, mask)
    _check_physical_basis(family, physical)
    base_level = Level{D,T,typeof(family)}(1, :base, base_mesh, family, orders, mode, mask)
    return Space{D,T,Tuple{typeof(base_level)}}(domain, (base_level,), physical)
end

"""
    overlay(V::Space, domain::AxisBox; cells, order=…, basis=…, mode=…,
                                       tolerance=GeometryTolerance(T),
                                       active=nothing) -> Space

Add an overlay level on the sub-box `domain` to an existing [`Space`](@ref)
`V`, returning the extended space. The overlay's domain must lie inside
`V.domain` up to `tolerance.contain` (see [`is_inside`](@ref)).

Defaults inherit from the base level so a quick overlay only needs `cells`
and possibly `order`. Keyword arguments:

  - `cells` — per-axis cell counts for the overlay mesh.
  - `order` — polynomial order; defaults to the base level's order.
  - `basis` — basis family; defaults to the base level's family.
  - `mode` — basis index set; defaults to the base level's mode.
  - `tolerance` — slack on the inside-domain check.
  - `active` — optional per-cell mask, same shapes as [`space`](@ref)'s
    `active`.

The new level's `id` is `length(V.levels) + 1`. Overlay placement is
independent of any existing overlay: overlay boundaries need not coincide
with lower-level element boundaries, and there is no topological merging
with parents (every overlay imposes homogeneous Dirichlet data on its
artificial boundary, see `CONTRIBUTING.md`'s "Approximation space"
section).
"""
function overlay(V::Space{D,T}, domain::AxisBox{D,T}; cells, order=V.levels[1].order,
                 basis=V.levels[1].basis, mode::Symbol=V.levels[1].mode,
                 tolerance=GeometryTolerance(T), active=nothing) where {D,T}
    is_inside(domain, V.domain, tolerance) ||
        throw(ArgumentError("overlay domain must lie inside the physical domain"))
    orders = _axis_int_tuple(order, Val(D), :order)
    _check_basis_mode(mode, orders)
    overlay_mesh = CartesianMesh(domain; cells)
    mask = _normalize_mask(active, overlay_mesh)
    id = length(V.levels) + 1
    family = instantiate_basis(basis, overlay_mesh, orders, mode, mask)
    _check_physical_basis(family, V.physical)
    level = Level{D,T,typeof(family)}(id, :overlay, overlay_mesh, family, orders, mode, mask)
    levels = (V.levels..., level)
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
    family = instantiate_basis(old.basis, new_mesh, old.order, old.mode, old.mask)
    new_level = Level{D,T,typeof(family)}(old.id, old.role, new_mesh, family, old.order, old.mode,
                                          old.mask)
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
    family = instantiate_basis(old.basis, old.mesh, old.order, old.mode, mask)
    new_level = Level{D,T,typeof(family)}(old.id, old.role, old.mesh, family, old.order, old.mode,
                                          mask)
    levels = ntuple(i -> i == level_index ? new_level : V.levels[i], length(V.levels))
    return Space{D,T,typeof(levels)}(V.domain, levels, V.physical)
end

# Per-cell classification of a level's mesh against `physical`. Returns a
# `BitArray{D}` marking the cells the classifier reports as fully outside
# Ω (`:fictitious`). Cells classified as `:full` or `:cut` are kept active:
# `:cut` cells will get their quadrature replaced by the NNMF moment-fit
# rule at region-build time (see `intersections.jl`), while `:fictitious`
# cells are dropped from the dof layout via the fold below (strict-α path
# only, i.e. `physical.alpha == 0`).
function _level_fictitious_cells(level::Level{D,T}, physical::PhysicalDomain,
                                 cache::_ClassifyCache{D,T}) where {D,T}
    fict = falses(level.mesh.cells)
    for ci in cell_indices(level.mesh)
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
function _apply_physical_fold(V::Space{D,T}, cache::_ClassifyCache{D,T}) where {D,T}
    V.physical === nothing && return V
    V.physical.keep_fictitious && return V

    new_levels = ntuple(length(V.levels)) do i
        level = V.levels[i]
        fict = _level_fictitious_cells(level, V.physical, cache)
        if any(fict)
            new_mask = _fold_fictitious(level.mask, fict)
            Level{D,T,typeof(level.basis)}(level.id, level.role, level.mesh, level.basis,
                                           level.order, level.mode, new_mask)
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
        Level{D,T,typeof(lvl.basis)}(lvl.id + offset, lvl.role, lvl.mesh, lvl.basis, lvl.order,
                                     lvl.mode, lvl.mask)
    end
    return Space{D,T,typeof(new_levels)}(V.domain, new_levels, V.physical)
end

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

function Base.show(io::IO, level::Level)
    print(io, "Level(id=", level.id, ", role=:", level.role, ", cells=", level.mesh.cells,
          ", order=", level.order, ", mode=:", level.mode, ", basis=:", basis_name(level.basis))
    if level.mask !== nothing
        print(io, ", active=", count(level.mask.on), "/", length(level.mask.on))
    end
    print(io, ")")
end

function Base.show(io::IO, V::Space{D}) where {D}
    print(io, "Space(D=", D, ", levels=", length(V.levels), ", domain=", V.domain, ")")
end
