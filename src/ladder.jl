# Nested refinement ladders on the superposition stack.
#
# `overlay` places a level anywhere inside the space's bounding box. That freedom
# is the method; it also has a price the package charges silently. The
# pruning rule in `dofs.jl` sheds a level's buried high-order modes
# (`:coverage`) and the buried linear modes a nested level reproduces (`:dedup`).
# Where the levels are nested — `_nested_over` in `coverage.jl` — reduction
# removes exactly the redundancy. Where they are not, the same default deletes
# modes nothing replaces, and it does so at full rank and a healthy condition
# number, so no residual, no `isposdef` test and no assembly diagnostic reports
# it.
#
# A level nests over its parent exactly when
#
#   (i)  every corner of its box is a parent node coordinate, and
#   (ii) its per-axis cell count is an integer multiple of the parent cells the
#        box spans.
#
# Both are arithmetic, so a constructor can guarantee them. [`ladder`](@ref)
# builds a stack that satisfies them by construction — every level spans the
# whole domain at an integer multiple of the previous level's resolution — and
# [`adapt`](@ref) changes which cells of each level are live without touching the
# geometry that makes the guarantee hold.
#
# What the ladder deliberately does NOT impose is any relation between the active
# regions of consecutive levels. A level may be live where the level below it is
# dormant, and on a cell set that cuts across the level below. Superposition has
# no hanging nodes and no 2:1 balance to maintain, so neither is a well-posedness
# condition, and both are useful: a feature far finer than the base cell can be
# resolved without paying for the levels in between. The cost is conditioning —
# see [`adapt`](@ref) — not admissibility.

# Bounds check shared by every entry point that takes a level index. Written once
# because three of them used to spell it out and one of them used to raise a
# `BoundsError` naming the internal level tuple.
function _check_level(V::Space, level::Integer, name::AbstractString="level")
    1 <= level <= length(V.levels) ||
        throw(ArgumentError("$name index $level is out of bounds for a space with " *
                            "$(length(V.levels)) levels"))
    return Int(level)
end

# Per-level split specification. A `Tuple` is per-axis and applies to every level;
# a `Vector` is per-level, one entry per overlay, each itself a positive integer
# or a `D`-tuple. The two readings collide — in 2D `(2, 2)` could be "two axes" or
# "two levels" — and the tuple/vector distinction settles it without a second
# keyword, reusing the fact that `_axis_int_tuple` already rejects a `Vector`.
_level_splits(s, depth::Int, ::Val{D}) where {D} = fill(_axis_int_tuple(s, Val(D), :splits), depth)

function _level_splits(s::AbstractVector, depth::Int, ::Val{D}) where {D}
    length(s) == depth ||
        throw(ArgumentError("a per-level `splits` vector must have one entry per overlay " *
                            "level: got $(length(s)) for depth $depth"))
    return [_axis_int_tuple(e, Val(D), :splits) for e in s]
end

"""
    ladder(domain::AxisBox; cells, depth, splits=2, order=1, basis=IntegratedLegendre(),
                            mode=:tensor, physical=nothing, active=nothing,
                            prune_covered=true, tolerance=GeometryTolerance(T)) -> Space

Declare a nested refinement ladder over `domain`: a base level plus `depth`
overlays, each spanning the whole domain at a per-axis multiple of the previous
level's resolution. Every level nests over every level below it, so the default
`prune_covered = true` removes exactly the redundancy rather than deleting modes
nothing replaces.

Every overlay is created with **no active cell**, so the declared stack carries
the unknowns of the base level alone. Populate it with [`adapt`](@ref).

Keyword arguments:

  - `cells`, `order`, `basis`, `mode`, `physical`, `active` — as on
    [`space`](@ref); they describe the base level. `basis`, `mode` and `order`
    are carried unchanged to every overlay, and `active` applies to the base
    level only.
  - `depth` — the number of overlay levels. The space has `depth + 1` levels, at
    positions `1:(depth + 1)`, deepest last. Coverage searches strictly upward in
    level id, so the emission order matters and this is the only order that is
    correct.
  - `splits` — the per-axis refinement factor between consecutive levels. A
    positive integer or an `NTuple{D,Int}` applies to every level; a `Vector` of
    those gives one entry per level, so `splits = [(2, 2), (3, 1)]` splits both
    axes on the first overlay and only axis 1 on the second. A factor of `1`
    inherits the previous level's partition on that axis — it never means "one
    cell", which would destroy nesting.
  - `prune_covered` — forwarded to every level. Leave it `true`: on a nested stack
    the shed modes are exactly linearly dependent, so `false` gives a singular
    operator rather than a more accurate one. It is exposed because measuring the
    redundancy requires building the unreduced twin.

# Order across the ladder

`order` is one value for the whole stack rather than a per-level schedule,
because that is the safe default rather than the only correct choice.

On a ladder whose levels genuinely cover each other, letting the order *descend*
with depth makes each parent shed buried high-order modes its lower-order child
does not reproduce. Whether that is a loss or the point depends on the problem.
For a smooth solution it costs approximation power: measured on a covering band
ladder, constant order reached 2.22e-5 where a 4→3→2→1 schedule reached 5.54e-3
and got worse with depth. For a *non-smooth* feature it is the reason order
reduction exists — the coarse high-order modes contribute little in L² and carry
the oscillation, so shedding them under a fine low-order overlay cut far-field
oscillation and overshoot by roughly an order of magnitude on a tanh layer.

`ladder` takes the constant-order default and leaves the trade to
[`overlay`](@ref), which accepts a per-level order. Raising the order with depth
is likewise sound and likewise an hp choice made there.

# Cost

Every level spans the whole domain, which is what keeps all their grids mutually
aligned and lets [`adapt`](@ref) address any level by cell index. The price is
paid in `integration_plan`: its candidate grid is the `D`-fold product of the
*finest* level's resolution and is independent of how few cells are active, so it
is re-paid on every [`adapt`](@ref), not once at setup. Measured on an empty
stack, base 8 in 2D: depth 3 → 1.5 ms, depth 4 → 7.4 ms, depth 5 → 37 ms per
plan rebuild. In 3D at base 4 the wall arrives around `depth = 5`.

The finest cell is `domain / (cells .* prod(splits))`, so a feature needing a
geometric grading of many orders of magnitude toward a point — a corner
singularity — is still [`overlay`](@ref)'s job: a ladder would need a depth the
candidate grid cannot afford.

Note that `order` defaults to `1`, at which an overlay contributes only vertex
functions and a small patch contributes none at all: every vertex of a one-cell
patch lies on its own artificial boundary, where the level is clamped to zero. A
refinement that appears to do nothing is usually this. Pass `order ≥ 2` unless
the base discretisation is deliberately multilinear.

# Example

```julia
V = ladder(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, depth=3, splits=2)
m = falses(64, 64)
m[29:36, 29:36] .= true
V = adapt(V, 4 => m)          # the finest level alone, live on a small patch
```
"""
function ladder(domain::AxisBox{D,T}; cells, depth::Integer, splits=2, order=1,
                basis=IntegratedLegendre(), mode::Symbol=:tensor, physical=nothing, active=nothing,
                prune_covered::Bool=true,
                tolerance::GeometryTolerance{T}=GeometryTolerance(T)) where {D,T}
    depth >= 0 || throw(ArgumentError("ladder depth must be non-negative; got $depth"))
    base_cells = _axis_int_tuple(cells, Val(D), :cells)
    factors = _level_splits(splits, Int(depth), Val(D))
    counts = _ladder_counts(base_cells, factors, domain, tolerance)

    V = space(domain; cells=base_cells, order=order, basis=basis, mode=mode, physical=physical,
              active=active, prune_covered=prune_covered)
    for k in 1:depth
        V = overlay(V, domain; cells=counts[k], order=order, basis=basis, mode=mode,
                    active=CartesianIndex{D}[], tolerance=tolerance, prune_covered=prune_covered)
    end
    return V
end

# Per-level cell counts, with the two failure modes a `depth` keyword invites
# checked before anything is allocated. Both are otherwise diagnosed far away
# from their cause: a spacing below `tolerance.merge` surfaces as
# `_check_axis_spacing` naming a level number at `prepare` time, and an
# unaffordable grid surfaces as the process running out of memory inside
# `_mesh_axes`.
function _ladder_counts(base::NTuple{D,Int}, factors::Vector{NTuple{D,Int}}, domain::AxisBox{D,T},
                        tol::GeometryTolerance{T}) where {D,T}
    counts = Vector{NTuple{D,Int}}(undef, length(factors))
    current = base
    extent = domain.upper - domain.lower
    for (k, f) in pairs(factors)
        all(isone, f) &&
            throw(ArgumentError("ladder level $k has `splits` of $f, which repeats the grid below it. A level with " *
                                "the same partition as its parent adds no resolution: covered-mode pruning deduplicates " *
                                "it away again, and it costs a plan, a dof layout and an assembly pass. Use a " *
                                "factor above 1 on at least one axis, or reduce `depth`."))
        current = ntuple(d -> current[d] * f[d], D)
        for d in 1:D
            h = extent[d] / current[d]
            h > tol.merge ||
                throw(ArgumentError("ladder depth $k puts the cell spacing on axis $d at $h, at or below the " *
                                    "geometry merge tolerance $(tol.merge); the mesh coordinates would be " *
                                    "collapsed. Reduce `depth` or `splits`, or pass a smaller `tolerance`."))
        end
        prod(Float64.(current)) <= 1e9 ||
            throw(ArgumentError("ladder depth $k would need $(prod(Float64.(current))) cells on one level. " *
                                "Reduce `depth` or `splits`."))
        counts[k] = current
    end
    return counts
end

"""
    is_nested(V::Space; tolerance=GeometryTolerance(T)) -> Bool

Whether every level of `V` is nested over every level above it: each level's node
coordinates that fall inside a higher level's domain coincide with that level's
node coordinates, per axis.

This is the **geometric** half of the condition the `:dedup` elimination tests,
and nothing more. It is not a witness that covered-mode pruning is lossless on `V`:

  - it does not look at polynomial order, and a nested cover of *lower* order
    cannot reproduce what it displaces;
  - it does not look at which levels cover which cells, and a mode buried under
    two *different* levels is reproduced by neither;
  - it does not consult `prune_covered`, so it reports `false` — with all the
    alarm this docstring might suggest — on a stack that eliminates nothing.

It is still the cheapest check that a hand-built [`overlay`](@ref) stack has the
mesh alignment the reduction rule wants, which is the failure that is otherwise
invisible. A [`ladder`](@ref) satisfies it by construction; [`move!`](@ref) can
void it, and nothing re-checks.

Note that it can be satisfied vacuously: a level whose box contains no interior
node of the level below has nothing to match, so it passes.
"""
function is_nested(V::Space{D,T}; tolerance::GeometryTolerance{T}=GeometryTolerance(T)) where {D,T}
    for i in eachindex(V.levels), j in eachindex(V.levels)
        V.levels[j].id > V.levels[i].id || continue
        _nested_over(V.levels[i], V.levels[j], tolerance) || return false
    end
    return true
end

# ── Reading a level ───────────────────────────────────────────────────────────

"""
    active_cells(V::Space; level) -> BitArray

Which cells of `level` are active, as a copy. A level with no mask returns an
all-true array. The `Space` counterpart of `active_cells(::Model; level)`, and
the read half of an adaptive step: fetch a level's mask, edit it per cell, hand
it back to [`adapt`](@ref).

On a `Space` this is the mask as written. The `Model` method reports the
*effective* mask by default, which on a space carrying a [`PhysicalDomain`](@ref)
already has the fictitious fold applied; pass `effective = false` there to read
back what the caller asked for.
"""
function active_cells(V::Space{D}; level::Integer) where {D}
    lvl = V.levels[_check_level(V, level)]
    lvl.mask === nothing && return trues(lvl.mesh.cells)
    return copy(lvl.mask.on)
end

"""
    cell_orders(V::Space; level) -> Array{NTuple{D,Int},D}

Per-axis polynomial order of every cell of `level`, as a fresh array. A level
whose order is uniform returns that order repeated over the cell grid, so the
result has the same shape and meaning whether or not the level carries a per-cell
field.

The read half of a p-adaptive step, and the `Space` counterpart of
`cell_orders(::Model; level)`: fetch a level's orders, edit them per cell, hand
them back to [`elevate`](@ref). Round-tripping through `elevate` is an identity.

```julia
p = cell_orders(V; level=1)
p[marked] .+= 1
V = elevate(V, 1 => p)
```
"""
function cell_orders(V::Space{D}; level::Integer) where {D}
    lvl = V.levels[_check_level(V, level)]
    lvl.orders === nothing && return fill(lvl.order, lvl.mesh.cells)
    return [lvl.orders.palette[c] for c in lvl.orders.class]
end

"""
    elevate(V::Space, level => order, ...) -> Space

Return `V` with the named levels' polynomial orders replaced. Meshes, level
boxes, activation masks and basis families are untouched, so the result has the
*same type* as `V` and a model rebuilt from it does not recompile the assembly
pipeline — the per-cell order lives in a `Union{Nothing,CellOrders}` field, not
in a type parameter.

`order` takes every shape [`space`](@ref)'s `order` keyword takes — an integer,
an `NTuple{D,Int}`, an array of either shaped like the level's cell grid, or a
predicate `(cell_box, cell_index) -> order` — plus one shape that only makes
sense against an existing level:

  * an iterable of `CartesianIndex{D} => order` pairs, raising the listed cells
    and leaving every other cell at the order it already has. That is the shape a
    marking loop produces, so a p-adaptive step does not have to materialise the
    whole field to change twelve cells.

Where two cells of different order share a face, the shared entity carries the
minimum of the two orders; see [`CellOrders`](@ref) for why that is what keeps
the space C⁰, and [`space`](@ref) for the per-cell `order ≥ 1` and
basis-family requirements, which are checked here too.

This is a separate verb from [`adapt`](@ref) rather than another shape of its
pair form, and the reason is dispatch rather than taste: a mask spec and an order
spec collide irreducibly on `nothing` ("every cell active" versus "uniform
order") and on a predicate (`-> Bool` versus `-> Int`), and neither collision can
be detected before the value is used. Compose them instead —
`elevate(adapt(V, h), p)` costs two cheap `Space` rebuilds and one
[`prepare`](@ref), and the dof layout only ever sees the final hp state.
"""
function elevate(V::Space{D,T}, pairs::Pair{<:Integer}...) where {D,T}
    n = length(V.levels)
    named = falses(n)
    for (level, _) in pairs
        k = _check_level(V, level)
        named[k] && throw(ArgumentError("level $k is named more than once in one `elevate` call"))
        named[k] = true
    end
    out = V
    for (level, spec) in pairs
        out = _reordered_space(out, _check_level(V, level), spec)
    end
    return out
end

# The order spec shape that only `elevate` can serve: `cell => order` pairs
# against the level's current field. Normalised here rather than in
# `_normalize_order` because it needs a base to fill the unnamed cells from, and
# `space` / `overlay` have none.
function _pair_orders(V::Space{D}, k::Int, pairs) where {D}
    lvl = V.levels[k]
    out = cell_orders(V; level=k)
    for entry in pairs
        entry isa Pair || throw(ArgumentError("an order pair list must contain " *
                                              "`CartesianIndex{$D} => order` entries; got $(typeof(entry))"))
        cell, order = entry
        cell isa CartesianIndex{D} ||
            throw(ArgumentError("an order pair list must be keyed by CartesianIndex{$D}; " *
                                "got $(typeof(cell))"))
        checkbounds(Bool, out, cell) ||
            throw(ArgumentError("cell index $cell is out of bounds for mesh cells $(lvl.mesh.cells)"))
        out[cell] = _axis_int_tuple(order, Val(D), :order)
    end
    return out
end

_normalize_elevate_spec(V::Space, k::Int, spec::Pair) = _pair_orders(V, k, (spec,))
_normalize_elevate_spec(V::Space, k::Int, spec::AbstractVector{<:Pair}) = _pair_orders(V, k, spec)
_normalize_elevate_spec(::Space, ::Int, spec) = spec

"""
    cell_indices(V::Space; level) -> CartesianIndices

Cell index space of `level`. Together with [`cell_box`](@ref)'s `Space` method
this is what lets a caller build a cell selection — evaluate an error indicator
per cell, mark, [`adapt`](@ref) — without reaching into `V.levels[k].mesh`.
"""
cell_indices(V::Space; level::Integer) = cell_indices(V.levels[_check_level(V, level)].mesh)

"""
    cell_box(V::Space, index::CartesianIndex; level) -> AxisBox

Axis-aligned box of one cell of `level`, in the physical frame.
"""
function cell_box(V::Space{D}, index::CartesianIndex{D}; level::Integer) where {D}
    return cell_box(V.levels[_check_level(V, level)].mesh, index)
end

# ── Mapping cells between levels ──────────────────────────────────────────────

# Index range of `from`'s cells that cell `j` of `to` overlaps along one axis, or
# an empty range where `j` lies outside `from`'s extent.
#
# Deliberately tolerance-free. An additive epsilon on the node coordinates is
# wrong twice over: `GeometryTolerance` is absolute, so on a domain far from the
# origin it is smaller than one ulp and does nothing, while on a domain of small
# extent it is larger than a whole cell and empties every range — both silently.
# The two searches below are exact instead, and they express the right convention
# on their own: `searchsortedlast` puts a lower face sitting exactly on a node
# into the cell above it, `searchsortedfirst` excludes the cell an upper face
# merely touches. Levels of a ladder share their node coordinates bit-for-bit
# (`_mesh_axes` builds endpoints exactly), so nothing here has to absorb a
# rounding difference.
function _overlap_range(to_axis::Vector{T}, from_axis::Vector{T}, j::Int) where {T}
    ncells = length(from_axis) - 1
    lo = searchsortedlast(from_axis, to_axis[j])
    hi = searchsortedfirst(from_axis, to_axis[j + 1]) - 1
    (hi < 1 || lo > ncells) && return 1:0                   # disjoint from `from`'s extent
    return max(lo, 1):min(hi, ncells)
end

"""
    overlapping_cells(V::Space, cells; from, to) -> BitArray

The cells of level `to` that overlap the `cells` selected on level `from`. This
is how a marking made where an error indicator lives becomes an activation on
another level: downward it names the children of the marked cells, upward the
cells that contain any of them. It is the same rule in both directions —
geometric overlap, per axis — so a ladder refined anisotropically, finer than
`from` on one axis and coarser on another, needs no special handling.

Cells that merely touch across a shared face do not overlap. A level of `to` that
lies outside `from`'s box selects nothing, rather than being credited to `from`'s
edge cell.

`cells` takes the shapes `active =` accepts: an `AbstractArray{Bool,D}` over
level `from`'s cell grid, an iterable of `CartesianIndex{D}`, a predicate
`(cell_box, cell_index) -> Bool`, or a [`LevelMask`](@ref).

# Example

```julia
marked = [ci for ci in cell_indices(V; level=3) if indicator[ci] > θ]
deeper = active_cells(V; level=4) .| overlapping_cells(V, marked; from=3, to=4)
V = adapt(V, 4 => deeper)
```
"""
function overlapping_cells(V::Space{D,T}, cells; from::Integer, to::Integer) where {D,T}
    src_level = V.levels[_check_level(V, from, "from")]
    dst_level = V.levels[_check_level(V, to, "to")]
    src = _level_selection(cells, src_level)
    from == to && return copy(src)

    from_mesh, to_mesh = src_level.mesh, dst_level.mesh
    ranges = ntuple(d -> [_overlap_range(to_mesh.axes[d], from_mesh.axes[d], j)
                          for j in 1:to_mesh.cells[d]], D)
    out = falses(to_mesh.cells)
    for ci in cell_indices(to_mesh)
        block = CartesianIndices(ntuple(d -> ranges[d][ci.I[d]], D))
        out[ci] = any(@view src[block])
    end
    return out
end

# A cell selection as a plain `BitArray` over a level's cells. `_normalize_mask`
# already accepts every shape `active =` documents and copies defensively, so it
# is the single definition of what a selection is; this only unwraps the mask it
# returns, and gives `nothing` the "every cell" reading it has everywhere else.
function _level_selection(cells, level::Level{D}) where {D}
    mask = _normalize_mask(cells, level.mesh)
    mask === nothing && return trues(level.mesh.cells)
    return mask.on
end

# ── Writing the activation of a whole stack ───────────────────────────────────

# Private "not named by the caller" sentinel. `nothing` cannot serve: it is a
# legal user value meaning "every cell", so using it here would make
# `adapt(V, 2 => nothing, 2 => m)` slip past the duplicate-level guard.
struct _Unset end

# Rebuild `V` with the given per-level masks, `_Unset()` leaving a level alone.
# One pass, so one integration plan and one dof layout downstream, where the same
# edit through `activate!` costs one of each per level touched. `_remasked_space`
# owns the per-level rebuild, including the `instantiate_basis` call that a
# mesh-dependent family needs.
function _with_masks(V::Space{D,T}, masks::NTuple{N,Any}) where {D,T,N}
    out = V
    for k in 1:N
        masks[k] isa _Unset && continue
        out = _remasked_space(out, k, _normalize_mask(masks[k], V.levels[k].mesh))
    end
    return out
end

"""
    adapt(V::Space, level => cells, ...) -> Space
    adapt(V::Space, depths::AbstractArray{<:Integer,D}; grade=0) -> Space

Return `V` with the named levels' activation masks replaced. Level boxes, cell
counts, orders and basis families are untouched, so the result has the *same
type* as `V` and a model rebuilt from it does not recompile the assembly
pipeline.

The pair form sets each named level's mask outright and leaves every other level
alone. `cells` takes the shapes `active =` accepts, so a mask written for
[`overlay`](@ref) can be handed straight here:

```julia
m = active_cells(V; level=4)
m[marked] .= true
V = adapt(V, 4 => m)
```

Setting several levels in one call is one integration-plan and dof-layout
rebuild; the same edit through [`activate!`](@ref) is one of each per level.

Nothing requires a level to be active only where its parent is, or its active
cells to tile whole parent cells. Both are admissible, and a feature far finer
than a base cell can be resolved without paying for the intervening levels. What
they cost is conditioning: where a level's active region boundary does not fall
on the cell boundaries of the level below, the condition number of the assembled
operator rises — measured at one to three orders of magnitude on stacked ragged
masks, against a parent-aligned stack carrying several times the unknowns. Prefer
a mask whose boundary follows the coarser level's cells when there is a choice.

The array form is the shorthand for the structured case: it replaces every
*overlay* level's mask with the one implied by a per-base-cell refinement depth,
as [`depth_masks`](@ref) builds them, and leaves the base level alone. It
overwrites masks set by the pair form rather than merging with them, so a stack
maintained per cell should not be edited through it.

`grade` bounds how fast the depth may change between neighbouring base cells and
applies to the array form only. It defaults to `0`, meaning the array is applied
verbatim; see [`depth_masks`](@ref) for what a positive value buys and costs.
"""
function adapt(V::Space{D,T}, pairs::Pair{<:Integer}...) where {D,T}
    n = length(V.levels)
    named = falses(n)
    for (level, _) in pairs
        k = _check_level(V, level)
        named[k] && throw(ArgumentError("level $k is named more than once in one `adapt` call"))
        named[k] = true
    end
    masks = ntuple(k -> _pair_mask(pairs, k), Val(n))
    return _with_masks(V, masks)
end

# The mask a pair list gives level `k`, or the sentinel. A loop rather than a
# `Dict` because a stack carries a handful of levels and this runs inside
# `ntuple`, where a type-stable return matters more than the asymptotics.
function _pair_mask(pairs::Tuple, k::Int)
    for (level, cells) in pairs
        Int(level) == k && return cells
    end
    return _Unset()
end

function adapt(V::Space{D,T}, depths::AbstractArray{<:Integer,D}; grade::Int=0) where {D,T}
    eltype(depths) === Bool &&
        throw(ArgumentError("adapt received a Bool array where a per-base-cell depth map was expected. To set one " *
                            "level's mask, name the level: `adapt(V, k => mask)`."))
    return _with_masks(V, depth_masks(V, depths; grade=grade))
end

"""
    depth_masks(V::Space, depths; grade=0) -> NTuple

The per-level masks implied by a per-base-cell refinement `depths` array: overlay
level `k` is active exactly where `depths ≥ k - 1`. A convenience constructor for
the structured case, not the state of the stack — the masks it returns can be
edited per cell before being handed to [`adapt`](@ref).

`depths` is shaped like the base level's cell grid, with values in
`0:(length(V.levels) - 1)`. A value outside that range raises rather than being
clamped: a depth the ladder cannot represent means the caller is refining less
than it believes. The base level's mask is never derived from `depths`; the first
entry of the returned tuple is always `nothing`.

Because a depth is per *base* cell, a marked base cell is refined across its
whole child block on every deeper level. That is coarser than the stack can
express — [`adapt`](@ref)'s pair form reaches individual cells of any level — and
on a point feature it is the difference between refining a base cell and refining
the few cells that need it.

`grade` bounds how fast the depth may change between neighbouring base cells:
with `grade = k`, a cell adjacent to one at depth `d` is raised to at least
`d - k`. Grading only ever raises depths, so it cannot undo a request. It
defaults to `0` — off — because superposition needs no 2:1 balance for
well-posedness, so this is a convenience rather than a requirement of the method,
and it is expensive: it inflates the refined region by roughly `(2·grade + 1)^D`
per level, which in 3D was measured at 2.2× the unknowns and 2.9× the assembly
time for a single deeply refined corner cell. Use it when a level's artificial
boundary Γ_o, where its contribution is clamped to zero, would otherwise fall on
the feature that marked it.
"""
function depth_masks(V::Space{D,T}, depths::AbstractArray{<:Integer,D}; grade::Int=0) where {D,T}
    base = V.levels[1].mesh
    size(depths) == base.cells ||
        throw(DimensionMismatch("depth map shape $(size(depths)) does not match the base " *
                                "level's cells $(base.cells)"))
    maxdepth = length(V.levels) - 1
    all(dk -> 0 <= dk <= maxdepth, depths) ||
        throw(ArgumentError("depths must lie in 0:$maxdepth for a ladder with " *
                            "$(length(V.levels)) levels; got $(extrema(depths))"))
    grade >= 0 || throw(ArgumentError("grade must be non-negative; got $grade"))

    d = Array{Int,D}(depths)
    grade > 0 && _grade!(d, grade)
    return ntuple(k -> k == 1 ? nothing : overlapping_cells(V, findall(>=(k - 1), d); from=1, to=k),
                  Val(length(V.levels)))
end

# Raise every depth that sits more than `by` below a neighbour, to the fixed point
#
#     d[c] ← max over x of (d[x] − by · ‖c − x‖_∞).
#
# One iteration is `d ← max(d, dilate(d) − by)` with `dilate` the ∞-norm unit-ball
# dilation. That ball is a hypercube, and a hypercube is the Minkowski sum of unit
# intervals, so the dilation separates into `D` one-dimensional maximum filters
# rather than a sweep over the `3^D` neighbourhood. Writing it as a whole-array
# step also makes the result independent of iteration order, which a
# neighbour-by-neighbour sweep is not — the same input graded from opposite
# corners differed by a factor of four in cost, and the fixed point is a property
# of the input, not of the traversal.
function _grade!(d::Array{Int,D}, by::Int) where {D}
    buffer = Vector{Int}(undef, maximum(size(d); init=0))
    dilated = similar(d)
    while true
        copyto!(dilated, d)
        for axis in 1:D
            _max_filter3!(dilated, axis, buffer)
        end
        dilated .-= by
        any(k -> dilated[k] > d[k], eachindex(d)) || return d
        d .= max.(d, dilated)
    end
end

# In-place maximum filter of width three along one axis. `buffer` holds the row
# being filtered so the filter reads the original values rather than the ones it
# has just written.
function _max_filter3!(a::Array{Int,D}, axis::Int, buffer::Vector{Int}) where {D}
    n = size(a, axis)
    n > 1 || return a
    for slice in CartesianIndices(ntuple(d -> d == axis ? (1:1) : axes(a, d), D))
        for i in 1:n
            buffer[i] = a[_along(slice, axis, i)]
        end
        for i in 1:n
            a[_along(slice, axis, i)] = max(buffer[max(i - 1, 1)], buffer[i], buffer[min(i + 1, n)])
        end
    end
    return a
end

# `slice` with its `axis` component replaced by `i`.
function _along(slice::CartesianIndex{D}, axis::Int, i::Int) where {D}
    CartesianIndex(ntuple(d -> d == axis ? i : slice.I[d], D))
end
