# Nested refinement ladders on the superposition stack.
#
# `overlay` places a level anywhere inside the space's bounding box. That freedom is the method; it also has a price the
# package charges silently. The pruning rule in `dofs.jl` sheds a level's buried high-order modes (`:coverage`) and the
# buried linear modes a nested level reproduces (`:dedup`). Where the levels are nested — `_nested_over` in
# `coverage.jl` — reduction removes exactly the redundancy. Where they are not, the same default deletes modes nothing
# replaces, and it does so at full rank and a healthy condition number, so no residual, no `isposdef` test and no
# assembly diagnostic reports it.
#
# A level nests over its parent exactly when
#
#   (i)  every corner of its box is a parent node coordinate, and
#   (ii) its per-axis cell count is an integer multiple of the parent cells the box spans.
#
# Both are arithmetic, so a constructor can guarantee them. [`ladder`](@ref) builds a stack that satisfies them by
# construction — every level spans the whole domain at an integer multiple of the previous level's resolution, and
# `_mesh_axes` (`mesh.jl`) computes a node from the correctly rounded ratio `i / n` alone, so a level's node coordinates
# are its parent's, bit for bit, wherever the two meet — and [`adapt`](@ref) changes which cells of each level are live
# without touching the geometry that makes the guarantee hold.
#
# What the ladder deliberately does NOT impose is any relation between the active regions of consecutive levels. A level
# may be live where the level below it is dormant, and on a cell set that cuts across the level below. Superposition has
# no hanging nodes and no 2:1 balance to maintain, so neither is a well-posedness condition, and both are useful: a
# feature far finer than the base cell can be resolved without paying for the levels in between. The cost is
# conditioning — see [`adapt`](@ref) — not admissibility.

# ── Ladders and nesting ───────────────────────────────────────────────────────

"""
    ladder(domain::AxisBox; cells, depth, splits=2, order=1, basis=IntegratedLegendre(),
                            mode=:tensor, physical=nothing, active=nothing,
                            tolerance=GeometryTolerance(T)) -> Space

Declare a nested refinement ladder over `domain`: a base level plus `depth` overlays, each spanning the whole domain at
a per-axis multiple of the previous level's resolution. Every level nests over every level below it, so leaf semantics
remove exactly the redundancy rather than deleting modes nothing replaces.

Every overlay is created with **no active cell**, so the declared stack carries the unknowns of the base level alone.
Populate it with [`adapt`](@ref).

Keyword arguments:

  - `cells`, `order`, `basis`, `mode`, `physical`, `active` — as on [`space`](@ref); they describe the base level.
    `basis` and `mode` are carried unchanged to every overlay, and `active` applies to the base level only. `order`
    reaches every overlay too: an integer or a per-axis tuple unchanged; a per-cell array, shaped like the base level's
    cells, by giving every overlay cell the order of the base cell it lies in — what an h-step of [`refine`](@ref)
    gives a child; a function `(cell_box, cell_index) -> order` by asking it again on every level's own cells.
  - `depth` — the number of overlay levels. The space has `depth + 1` levels, at positions `1:(depth + 1)`, deepest
    last. Coverage searches strictly upward in level id, so the emission order matters and this is the only order that
    is correct.
  - `splits` — the per-axis refinement factor between consecutive levels. A positive integer or an `NTuple{D,Int}`
    applies to every level; a `Vector` of those gives one entry per level, so `splits = [(2, 2), (3, 1)]` splits both
    axes on the first overlay and only axis 1 on the second. A factor of `1` inherits the previous level's partition on
    that axis — it never means "one cell", which would destroy nesting.
  - `tolerance` — the [`GeometryTolerance`](@ref) each overlay's placement is checked against, and whose `merge` field
    is the floor `_ladder_counts` holds a level's cell spacing above, so a `depth` that would collapse the mesh
    coordinates raises here rather than at `prepare` time. Defaults to `GeometryTolerance(T)`.
  - leaf semantics apply to every level of the ladder, and there is nothing to configure. On a nested stack the shed
    modes are exactly linearly dependent, so keeping them would give a singular operator rather than a more accurate
    one. See [`space`](@ref).

# Order across the ladder

`order` is one value for the whole stack rather than a per-level schedule, because that is the safe default rather than
the only correct choice.

On a ladder whose levels genuinely cover each other, letting the order *descend* with depth makes each parent shed
buried high-order modes its lower-order child does not reproduce. Whether that is a loss or the point depends on the
problem. For a smooth solution it costs approximation power: measured on a covering band ladder, constant order reached
2.22e-5 where a 4→3→2→1 schedule reached 5.54e-3 and got worse with depth. For a *non-smooth* feature it is the reason
order reduction exists — the coarse high-order modes contribute little in L² and carry the oscillation, so shedding them
under a fine low-order overlay cut far-field oscillation and overshoot by roughly an order of magnitude on a tanh layer.

`ladder` takes the constant-order default and leaves the trade to the verbs that exist for it: [`elevate`](@ref)
rewrites the order of any level — or of named cells of it — on a stack that already exists, and [`refine`](@ref) makes
the same choice per cell inside the adaptive loop. A hand-built stack can also set a per-level order at
[`overlay`](@ref) time. Raising the order with depth is likewise sound and likewise an hp choice made there, not here.

# Cost

Every level spans the whole domain, which is what keeps all their grids mutually aligned and lets [`adapt`](@ref)
address any level by cell index. What that costs is paid in two places, and only one of them scales with `depth` on its
own.

`integration_plan` follows the **live** cells, not the declared resolution: the candidate grid is built from the
element-boundary coordinates that bound an active cell, so an inert level contributes none at all and a level live on a
few cells contributes a few. A declared but unpopulated ladder therefore costs what its base costs, whatever depth it
carries. Measured at `-O0` on four threads as the minimum of seven rebuilds, base 8 in 2D with every overlay inert:
0.084 / 0.107 / 0.139 / 0.165 ms per plan rebuild at depth 3 / 4 / 5 / 6, for 64 regions throughout — the base's own
partition, while the finest level grows from 64² to 512². In 3D at base 4, 0.167 ms at depth 3 and 0.188 ms at depth 5.

Populating it is what costs, and it costs in proportion to the refined region rather than to the stack. The live cells'
coordinates cut the whole domain along each axis, so the candidate grid is the `D`-fold product of what they contribute.
With one base cell refined to the bottom, base 8 in 2D: 0.24 / 0.94 / 3.8 / 17.6 ms at depth 3 / 4 / 5 / 6, for 127 /
319 / 1087 / 4159 regions; in 3D at base 4, 1.8 ms at depth 3 and 164 ms at depth 5. That is re-paid on every
[`adapt`](@ref), not once at setup, so a cycle that refines pays it per cycle.

What does scale with `depth` regardless of activity is the per-level dense cell grid and its mask: every level spans the
whole domain, so level `k` allocates `prod(counts[k])` cells whether or not any of them is live. At base 8 in 2D that is
22.4 million cells across a depth-9 stack, where [`adapt`](@ref) alone costs 142 ms. `_ladder_counts` refuses a level
above 1e9 cells for that reason.

The finest cell is `domain / (cells .* prod(splits))`, so a feature needing a geometric grading of many orders of
magnitude toward a point — a corner singularity — is still [`overlay`](@ref)'s job: a ladder would need a depth whose
dense per-level grids do not fit, even though only a handful of their cells would ever be live.

Note that `order` defaults to `1`, at which an overlay contributes only vertex functions and a small patch contributes
none at all: every vertex of a one-cell patch lies on its own artificial boundary, where the level is clamped to zero. A
refinement that appears to do nothing is usually this. Pass `order ≥ 2` unless the base discretisation is deliberately
multilinear.

# Example

```julia
V = ladder(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, depth=3, splits=2)
m = falses(64, 64)
m[29:36, 29:36] .= true
V = adapt(V, 4 => m)          # the finest level alone, live on a small patch
```
"""
function ladder(domain::AxisBox{D,T}; cells, depth::Integer, splits=2, order=1, basis=IntegratedLegendre(),
                mode::Symbol=:tensor, physical=nothing, active=nothing,
                tolerance::GeometryTolerance{T}=GeometryTolerance(T)) where {D,T}
    depth >= 0 || throw(ArgumentError("ladder depth must be non-negative; got $depth"))
    base_cells = _axis_int_tuple(cells, Val(D), :cells)
    factors = _level_splits(splits, Int(depth), Val(D))
    counts = _ladder_counts(base_cells, factors, domain, tolerance)

    V = space(domain; cells=base_cells, order=order, basis=basis, mode=mode, physical=physical, active=active)
    for k in 1:depth
        # A per-cell order is shaped like the base grid, which no overlay is. Every level spans the whole domain at an
        # integer multiple of the base resolution, so repeating each entry over its own child block is exactly "each
        # cell inherits its base parent's order" — the order an h-step of `refine` hands a child.
        o = order isa AbstractArray ? repeat(order; inner=counts[k] .÷ base_cells) : order
        V = overlay(V, domain; cells=counts[k], order=o, basis=basis, mode=mode, active=CartesianIndex{D}[],
                    tolerance=tolerance)
    end
    return V
end

# Per-level split specification. A `Tuple` is per-axis and applies to every level; a `Vector` is per-level, one entry
# per overlay, each itself a positive integer or a `D`-tuple. The two readings collide — in 2D `(2, 2)` could be "two
# axes" or "two levels" — and the tuple/vector distinction settles it without a second keyword, reusing the fact that
# `_axis_int_tuple` already rejects a `Vector`.
_level_splits(s, depth::Int, ::Val{D}) where {D} = fill(_axis_int_tuple(s, Val(D), :splits), depth)

function _level_splits(s::AbstractVector, depth::Int, ::Val{D}) where {D}
    length(s) == depth || throw(ArgumentError("a per-level `splits` vector must have one entry per overlay " *
                                              "level: got $(length(s)) for depth $depth"))
    return [_axis_int_tuple(e, Val(D), :splits) for e in s]
end

# Per-level cell counts, with the two failure modes a `depth` keyword invites checked before anything is allocated. Both
# are otherwise diagnosed far away from their cause: a spacing below `tolerance.merge` surfaces as `_check_axis_spacing`
# naming a level number at `prepare` time, and an unaffordable grid surfaces as the process running out of memory inside
# `_mesh_axes`.
function _ladder_counts(base::NTuple{D,Int}, factors::Vector{NTuple{D,Int}}, domain::AxisBox{D,T},
                        tol::GeometryTolerance{T}) where {D,T}
    counts = Vector{NTuple{D,Int}}(undef, length(factors))
    current = base
    extent = domain.upper - domain.lower
    for (k, f) in pairs(factors)
        all(isone, f) &&
            throw(ArgumentError("ladder level $k has `splits` of $f, which repeats the grid below it. A level with " *
                                "the same partition as its parent adds no resolution: covered-mode pruning " *
                                "deduplicates it away again, and it costs a plan, a dof layout and an assembly pass. " *
                                "Use a factor above 1 on at least one axis, or reduce `depth`."))
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

Whether every level of `V` is nested over every level above it: each level's node coordinates that fall inside a higher
level's domain coincide with that level's node coordinates, per axis.

This is the **geometric** half of the condition the `:dedup` elimination tests, and nothing more. It is not a witness
that covered-mode pruning is lossless on `V`:

  - it does not look at polynomial order, and a nested cover of *lower* order cannot reproduce what it displaces;
  - it does not look at which levels cover which cells, and a mode buried under two *different* levels is reproduced by
    neither;
  - it is purely geometric, so it reports `false` — with all the alarm this docstring might suggest — on a stack that
    eliminates nothing.

It is still the cheapest check that a hand-built [`overlay`](@ref) stack has the mesh alignment the reduction rule
wants, which is the failure that is otherwise invisible. A [`ladder`](@ref) satisfies it by construction;
[`move!`](@ref) can void it, and nothing re-checks. `diagnostics(model, solution).levels[k].nested` reports the same
question per level, which is where a run records the answer without being asked.

Note that it can be satisfied vacuously: a level whose box contains no interior node of the level below has nothing to
match, so it passes.
"""
function is_nested(V::Space{D,T}; tolerance::GeometryTolerance{T}=GeometryTolerance(T)) where {D,T}
    for i in eachindex(V.levels), j in eachindex(V.levels)
        V.levels[j].id > V.levels[i].id || continue
        _nested_over(V.levels[i], V.levels[j], tolerance) || return false
    end
    return true
end

# ── Mapping cells between levels ──────────────────────────────────────────────

"""
    overlapping_cells(V::Space, cells; from, to) -> BitArray

The cells of level `to` that overlap the `cells` selected on level `from`. This is how a marking made where an error
indicator lives becomes an activation on another level: downward it names the children of the marked cells, upward the
cells that contain any of them. It is the same rule in both directions — geometric overlap, per axis — so a ladder
refined anisotropically, finer than `from` on one axis and coarser on another, needs no special handling.

Cells that merely touch across a shared face do not overlap. A level of `to` that lies outside `from`'s box selects
nothing, rather than being credited to `from`'s edge cell.

`cells` takes the shapes `active =` accepts: an `AbstractArray{Bool,D}` over level `from`'s cell grid, an iterable of
`CartesianIndex{D}`, a predicate `(cell_box, cell_index) -> Bool`, or a [`LevelMask`](@ref).

# Example

```julia
marked = [ci for ci in cell_indices(V; level=3) if indicator[ci] > θ]
deeper = active_cells(V; level=4) .| overlapping_cells(V, marked; from=3, to=4)
V = adapt(V, 4 => deeper)
```

# Example: reading a level's cover upward

Called with `from` the finer level and `to` the coarser one, the same function answers the question a coarsening step
asks — *which parents cover a live child, and which of those cover only cold ones* — for a whole level at once. The
overlap relation is symmetric, so nothing has to be said differently in this direction:

```julia
# `indicator` holds the error indicator over level k+1; `peak` its maximum, and
# `fraction` the share of it below which a child counts as cold.
live = active_cells(V; level=k + 1)
hot = live .& (indicator .> fraction * peak)
releasable = overlapping_cells(V, live; from=k + 1, to=k) .&
             .!overlapping_cells(V, hot; from=k + 1, to=k)
```

Two calls per level, rather than one call per cell of level `k` asking what it covers below. Both compute the same set,
but a single call scans the whole source level and allocates a mask over the whole destination level, so per-cell use
pays for both once per cell where a set-wise call pays once per level. On a 2D ladder of 4×4 base cells and five
overlays (levels up to 128×128, 42–50 live cells on each overlay), the sweep above took **0.058 ms against 56 ms** for
the per-cell loop — the same answer, three orders of magnitude apart.
"""
function overlapping_cells(V::Space{D,T}, cells; from::Integer, to::Integer) where {D,T}
    src_level = V.levels[_check_level(V, from, "from")]
    dst_level = V.levels[_check_level(V, to, "to")]
    src = _selection(cells, src_level.mesh)
    from == to && return src

    # The union of the selected cells' blocks, which is the same set as "every destination cell that sees a selected
    # source cell" because the overlap relation is symmetric. Reading it this way costs one block per *selected* cell
    # instead of one pass over the whole destination grid, which is the difference between 25 µs and 7.58 ms when 64
    # base cells are mapped onto a 64³ level, and parity (1.95 ms against 1.91 ms) in the other direction with 15 625
    # sources. The per-axis blocks are tabulated once over the source axes so no cell repeats the binary search.
    f, t = src_level.mesh, dst_level.mesh
    ranges = ntuple(d -> [_axis_overlap(t.axes[d], f.axes[d], i) for i in 1:f.cells[d]], D)
    out = falses(t.cells)
    for ci in CartesianIndices(src)
        src[ci] || continue
        out[CartesianIndices(ntuple(d -> ranges[d][ci.I[d]], D))] .= true
    end
    return out
end

# The block of level-`to` cells that cell `cell` of level `from` overlaps. The answer is always an axis-aligned block,
# because both meshes are tensor products of sorted axes, so `D` binary searches settle it: O(D·log N + block) rather
# than the O(N_to) pass over the destination grid a dense answer would make.
#
# That difference is why this exists. Adaptivity asks the question one cell at a time — is anything live below this
# cell, which cells does an h-step activate, how many children does this parent have — and paying for a whole level per
# marked cell dominated the cycle: measured mapping one base cell of a depth-3 3D ladder onto its 64³ level, 7.65 ms
# through `overlapping_cells` against 0.29 µs for this block, and a cycle asks it once per mark from four sites.
#
# That ratio holds per mark and inverts over a sweep, which is worth saying here because the figures above invite the
# wrong generalisation. Asking the question for *every* cell of a level is faster through the public verb, called once
# on the whole set, than through this one called once per cell: one set-wise call amortises the per-axis tabulation, the
# source scan and the destination mask over the entire selection. Measured over a 2D ladder of 4×4 base cells and five
# overlays, the release sweep costs 0.058 ms as two `overlapping_cells` calls per level against 0.36 ms as one
# `_cell_block` per cell — and 56 ms as one `overlapping_cells` per cell, which is the trap. Per mark, this function;
# per level, the public one.
#
# The per-axis rule, rounding margin included, is `_axis_overlap` below, and `overlapping_cells` tabulates the same
# rule, so the per-cell and the set-wise map cannot disagree.
#
# The overlap relation is symmetric — cell i of one axis overlaps cell j of the other exactly when the reverse holds —
# so this one function answers the cross-level map in both directions and nothing here has to know which level is the
# finer one.
#
# An empty block is a legitimate answer rather than an error: a level whose box does not reach this cell covers none of
# it, which is the ordinary state of affairs on a sub-box overlay. Every caller must read it that way.
function _cell_block(V::Space{D}, from::Integer, to::Integer, cell::CartesianIndex{D}) where {D}
    f, t = V.levels[from].mesh, V.levels[to].mesh
    return CartesianIndices(ntuple(d -> _axis_overlap(t.axes[d], f.axes[d], cell.I[d]), D))
end

# The cells of axis `target` that cell `i` of axis `source` overlaps by more than rounding: the one per-axis relation
# both cross-level maps read.
#
# Each endpoint is pulled inward by δ = 4·eps(m) before the exact lookup, with m the largest coordinate magnitude on
# either axis — the coordinates' own round-off, at their own scale. Two meshes built from different boxes compute their
# nodes from different corners, so a node one of them places on the other's face can miss it by an ulp. Read exactly, a
# miss of 5.6e-17 on the wrong side made a cell that merely touches the face an overlapping one: a child an h-step woke,
# a child `_predicted` counted, a parent that was no longer a leaf — against `overlapping_cells`' own contract. On a
# ladder the margin changes nothing, because its levels share their node coordinates bit for bit (`_mesh_axes` makes a
# node a function of the ratio `i / n` alone), so a node either coincides or lies a whole cell away.
#
# The margin is relative, never `GeometryTolerance`'s absolute `contain`, which is wrong twice over: on a domain far
# from the origin it is below one ulp and does nothing, and on a domain of small extent it is larger than a whole cell
# and empties every range — both silently. A relative δ only has to stay far below a cell, and it does wherever every
# cell is wider than about 8·eps(m). That m is the larger of the two axes' magnitudes, not a cell's own coordinates, so
# a fine cell near a node of small magnitude is held to the scale of the larger axis. Under the default
# `GeometryTolerance`, whose `merge` admits no cell narrower than √eps(T) (`_check_axis_spacing`), it holds for every
# admitted mesh with coordinates below about 1e6; a custom tolerance on a deeply graded stack far from the origin is
# where to look for the limit. The coverage rule nudges by the absolute tolerance instead; see `_cells_under_box`.
function _axis_overlap(target::Vector{T}, source::Vector{T}, i::Integer) where {T}
    δ = 4 * eps(max(abs(target[1]), abs(target[end]), abs(source[1]), abs(source[end])))
    return _axis_cells(target, source[i] + δ, source[i + 1] - δ)
end

# ── Writing the activation of a whole stack ───────────────────────────────────

"""
    adapt(V::Space, depths::AbstractArray{<:Integer,D}; grade=0) -> Space

The structured shorthand for the pair form: replace every *overlay* level's mask with the one implied by a per-base-cell
refinement depth, and leave the base level alone. It is literally `adapt(V, Unfitted.depth_masks(V, depths; grade)...)`
— that builder is internal and not exported, so the composition above is the spelling to reach for — and it therefore
overwrites masks set by the pair form rather than merging with them, so a stack maintained per cell should not be edited
through it.

The base level is left alone because `depths` says nothing about it: a base cell at depth 0 is one that no overlay
refines, not one the base itself drops. A base mask set through [`space`](@ref)'s or [`ladder`](@ref)'s `active =`
therefore survives this call.

`grade` bounds how fast the depth may change between neighbouring base cells and applies to this form only. It defaults
to `0`, meaning the array is applied verbatim; see the internal `depth_masks` for what a positive value buys and costs.
"""
function adapt(V::Space{D}, depths::AbstractArray{<:Integer,D}; grade::Int=0) where {D}
    eltype(depths) === Bool &&
        throw(ArgumentError("adapt received a Bool array where a per-base-cell depth map was expected. To set one " *
                            "level's mask, name the level: `adapt(V, k => mask)`."))
    return adapt(V, depth_masks(V, depths; grade=grade)...)
end

"""
    depth_masks(V::Space, depths; grade=0) -> Vector{Pair{Int,BitArray{D}}}

The per-overlay-level masks implied by a per-base-cell refinement `depths` array: overlay level `k` is active exactly
where `depths ≥ k - 1`. The internal behind [`adapt`](@ref)'s array form, and the pairs it returns splat straight into
[`adapt`](@ref)'s pair form — `adapt(V, depth_masks(V, d)...)` is exactly what the array form does. To edit a level per
cell first, compose the two verbs: `adapt(adapt(V, depths), k => edited)`.

`depths` is shaped like the base level's cell grid, with values in `0:(length(V.levels) - 1)`. A value outside that
range raises rather than being clamped: a depth the ladder cannot represent means the caller is refining less than it
believes. The base level is never named: `depths` describes which overlays refine a base cell, which is not a statement
about the base's own mask.

Because a depth is per *base* cell, a marked base cell is refined across its whole child block on every deeper level.
That is coarser than the stack can express — [`adapt`](@ref)'s pair form reaches individual cells of any level — and on
a point feature it is the difference between refining a base cell and refining the few cells that need it.

`grade` bounds how fast the depth may change between neighbouring base cells: with `grade = k`, a cell adjacent to one
at depth `d` is raised to at least `d - k`. Grading only ever raises depths, so it cannot undo a request. It defaults to
`0` — off — because superposition needs no 2:1 balance for well-posedness, so this is a convenience rather than a
requirement of the method, and it is expensive: it inflates the refined region by roughly `(2·grade + 1)^D` per level,
which in 3D was measured at 2.2× the unknowns and 2.9× the assembly time for a single deeply refined corner cell. Use it
when a level's artificial boundary Γ_o, where its contribution is clamped to zero, would otherwise fall on the feature
that marked it.
"""
function depth_masks(V::Space{D}, depths::AbstractArray{<:Integer,D}; grade::Int=0) where {D}
    base = V.levels[1].mesh
    size(depths) == base.cells || throw(DimensionMismatch("depth map shape $(size(depths)) does not match the base " *
                                                          "level's cells $(base.cells)"))
    maxdepth = length(V.levels) - 1
    all(dk -> 0 <= dk <= maxdepth, depths) || throw(ArgumentError("depths must lie in 0:$maxdepth for a ladder with " *
                                                                  "$(length(V.levels)) levels; got $(extrema(depths))"))
    grade >= 0 || throw(ArgumentError("grade must be non-negative; got $grade"))

    # Overlay level k is live where the graded depth is at least k − 1, so only the threshold sets G[t] = {g ≥ t} of
    # the graded depth g are ever needed, and they are built directly. Grading raises every cell to
    #
    #     g(c) = max over x of (d(x) − grade · ‖c − x‖_∞),
    #
    # and for `grade > 0` its threshold sets obey
    #
    #     G[t] = {d ≥ t} ∪ dilate₁(G[t + grade]),
    #
    # with dilate₁ the unit-ball dilation of the ∞-norm, clipped to the grid (`_dilate_cells` at radius 1).
    #
    # ⊇: g ≥ d, and g is grade-Lipschitz in the ∞-norm, so a king neighbour of a cell where g ≥ t + grade has g ≥ t.
    # ⊆: let g(c) ≥ t be attained at x. If x = c, then d(c) ≥ t. Otherwise the king move from c toward x stays on the
    # grid and brings x one step closer, so that neighbour has g ≥ t + grade. Above `maxdepth − grade` the dilated set
    # is empty, since g never exceeds the largest depth. So the sets come top-down, one radius-1 dilation each, where
    # iterating g to its fixed point took whole-array max-filter sweeps until nothing moved.
    #
    # The selections are `BitArray`s over the base grid, which is a cell selection in every dimension. `findall` is
    # not: on a `Vector` it returns plain `Int` indices rather than `CartesianIndex{1}`, which `overlapping_cells`
    # rejects, so a 1D ladder with an overlay could not take a depth map at all. The element type is stated, not left
    # to the broadcast: `.>=` keeps the input's container, so a sparse or static depth map would give sets the
    # dilation does not take. A dense array already broadcasts to a `BitArray`, which the conversion passes through.
    G = BitArray{D}[depths .>= t for t in 1:maxdepth]
    if grade > 0
        for t in (maxdepth-grade):-1:1
            G[t] .|= _dilate_cells(G[t + grade], ntuple(_ -> 1, D))
        end
    end
    return [k => overlapping_cells(V, G[k - 1]; from=1, to=k) for k in 2:length(V.levels)]
end
