# Per-level coverage of the superposition stack, and the material oracle that coverage and the dof layer consult about
# inactive cells, about the part of a cell outside the level covering it, and about a level's box faces.
#
# The pruning rule sheds a level's high-order modes wherever a finer level fully covers it, keeping only its linear
# skeleton:
#
#   a level-j cell is *covered* iff the part of it that holds material lies inside the covering region of a *single*
#   level above it (higher id) — that level's domain, minus the cells it masks off that still carry material.
#
# The rule is per-covering-level on purpose: a cell covered only by the *union* of two abutting overlays counts as
# uncovered. See `build_coverage`.
#
# Without a physical domain the part that holds material is the whole cell box. On an immersed space it is the cell's
# share of Ω, so a cut cell whose material lies inside a cover's box is covered even where the rest of its box sticks
# out past the cover's face: nothing is integrated over the part outside Ω, and the cover represents everything the cell
# has to. Asking for the whole box there left the cell's high-order modes active next to the cover's copies of them on
# Ω, and every such copy was an exact null mode of the operator. See `_material_inside`.
#
# Coverage is mask-aware, so a deactivated overlay does not cover — which is what keeps "a fully-deactivated overlay
# behaves like no overlay" true under covered-mode pruning. The one inactive cell that still covers is one that holds no
# material at all: every cell the fictitious fold switched off, and a user-masked cell lying entirely outside Ω. See the
# comment on `_covered_by_level`.
#
# A cell without material is exempt on *both* sides of the question, and the two exemptions are separate functions:
#
#   * on the covering level, an inactive cell of the cover that holds no material does not stop it covering
#     (`_covered_by_level`);
#   * on the level being pruned, an inactive *support* cell that holds no material neither has to be covered nor may
#     block the verdict, because nothing is integrated there and the function the level assembles is the same on Ω with
#     it or without it (`_reproduced_on_domain`).
#
# Reading the rule with only the first exemption in mind is how a nested cover ending inside a folded coarse cell came
# to leave both copies of an exactly reproduced mode active; `_reproduced_on_domain` is the shared answer, which the
# burial driver in `dofs.jl` (`_coverage_constraints!`) asks for every family and every cover it judges geometrically.
#
# Both exemptions, the part of a cell outside its cover, and the artificial-face tests of the dof layer ask one question
# — does this hold material? — and `_Fold` below is the one value that answers it.

# ── The material oracle ───────────────────────────────────────────────────────

# Whether a cell, a box or a box face holds material, answered for one dof-layout build.
#
# A `LevelMask` merges two deactivations, the caller's `active =` selection and the fictitious fold
# (`_apply_physical_fold`, `mesh.jl`). Every site of the dof layer that looks behind an inactive cell — coverage, the
# burial test's level side, the integrated-Legendre artificial face, the B-spline admissible region and trace faces —
# asks the same question of it: does it hold material? A cell without any carries nothing to represent and no trace to
# keep continuous; a cell with some is represented by a coarser level only. The answer is a fact about the geometry and
# the cell box — `classify_cell` reports the box `:fictitious` — and not about which mask switched the cell off: every
# folded cell is void, and so is a user-masked cell lying entirely outside Ω.
#
# The same question is asked of two other things. Of a box that is no cell of the level's grid: a B-spline level's
# virtual cell past its own face, and the part of a coarse cell lying outside a cover's box (`_material_inside`). And of
# a level's own box face, which carries the artificial trace condition only where it holds material (`_is_void_face`,
# read by the box-face rule `_clamps_box_face` in `dofs.jl`).
#
# Fields:
#
#   * `physical` — the space's physical domain under its CONCRETE type `P` (`Nothing` without one). `Space.physical` is
#     abstractly typed; building a `_Fold` from its run-time value is the function barrier behind which every query
#     below is a static call, and with `P = Nothing` every query's `physical === nothing` test is a compile-time
#     constant, so it answers without touching the fields below.
#   * `boxes` — the box memo `classify_cell` fills, keyed by the box corners. It is the caller's: `prepare` hands in the
#     one its fold filled and its integration plan read, so a cell the fold classified costs a lookup here; a standalone
#     call passes a fresh one.
#   * `cells` — per level id, a per-cell verdict table (`0x00` not asked yet, `0x01` void, `0x02` material), allocated
#     on the level's first miss. A build asks about the same cell many times — per buried function, per cover, per face
#     position — and a table read is cheaper than hashing the box into `boxes` again.
#   * `covered_by` — per (level id, cover id) pair, the same kind of table for `_covered_by_level` of the level's cells
#     against the cover (`0x01` covered, `0x02` not), which `build_coverage` fills and the burial test reads back; see
#     `_covered_by!`.
#   * `faces` — per (level id, axis, side) — side `1` the lower, `2` the upper — the same kind of verdict table for that
#     box face of the level (`0x01` void, `0x02` material), over the level's grid with the axis collapsed to the one
#     boundary layer, allocated on the face's first ask. The box-face rule asks it once per raw on the face, and a
#     cell's face carries up to `(p + 1)^(D − 1)` of them.
#
# All three are indexed by level id. Ids are unique within the space the fold was built for, the only space it is asked
# about, and bounded by the largest of them, which sizes them up front. `covered_by` is therefore a (largest id)² matrix
# of which a build uses at most the pairs above the diagonal, and in a coupled model, where `_prepare_spaces` numbers a
# later subdomain's levels after the earlier ones', it also spans the ids below the space's own; `faces` holds
# (largest id) · 2D slots. Their slots are unassigned references until asked — a pointer each, in one allocation per
# layout build, pruning or not — which at the level counts in use is a few hundred bytes to a few KB.
#
# The verdicts are exactly `classify_cell`'s: a table miss makes the same
# `classify_cell(physical, cell_box(mesh, c), boxes)` call with the same box and memo, a hit returns the stored result
# of that call, and the call is a pure function of the geometry and the box. Only how often a cell is asked changes. The
# same holds for `covered_by`, a pure function of the cell box, the cover, the tolerance — one per build — and these
# verdicts, and for `faces`, whose verdict is the one a facet region on the same face gets (`_classify_facet`).
#
# A `_Fold` is built by `_dof_topology` or `build_coverage` and lives for that one call: no `Space`, `Model` or layout
# keeps it, so its tables are never shared between two public calls, and the box memo keeps the owner it has without it.
#
# It lives here rather than beside `_ClassifyCache` in `physical.jl` because its queries are about level grids (`Level`,
# `cell_box`), which load after `physical.jl`, and its users are this file, `dofs.jl` and the B-spline extension.
struct _Fold{D,T<:Real,P}
    physical::P
    boxes::_ClassifyCache{D,T}
    cells::Vector{Array{UInt8,D}}
    covered_by::Matrix{Array{UInt8,D}}
    faces::Array{Array{UInt8,D},3}
end

function _Fold(V::Space{D,T}, boxes::_ClassifyCache{D,T}=_ClassifyCache{D,T}()) where {D,T<:Real}
    n = maximum(level -> level.id, V.levels)
    return _Fold{D,T,typeof(V.physical)}(V.physical, boxes, Vector{Array{UInt8,D}}(undef, n),
                                         Matrix{Array{UInt8,D}}(undef, n, n), Array{Array{UInt8,D},3}(undef, n, D, 2))
end

# Whether `box`, an arbitrary axis-aligned box, holds no material: the void query of the method below for a box that is
# not a cell of one of the space's grids — a B-spline level's virtual cell past its own face — and so the box memo
# alone, with no table.
function _is_void(f::_Fold{D,T}, box::AxisBox{D,T}) where {D,T<:Real}
    f.physical === nothing && return false
    return classify_cell(f.physical, box, f.boxes) === :fictitious
end

# Whether cell `c` of `level` holds no material: its box lies entirely outside Ω. A fact about the geometry and the box,
# whatever the cell's activity. Inlined, so the hit path a per-cell loop runs — the B-spline admissibility walk above
# all — is an `isassigned` test, a table load and a compare; a level's table is allocated on its first miss, so a level
# nobody asks about, and every level of a space without a physical domain, gets none.
@inline function _is_void(f::_Fold{D,T}, level::Level{D,T}, c::CartesianIndex{D}) where {D,T<:Real}
    f.physical === nothing && return false
    isassigned(f.cells, level.id) || return _classify_void!(f, level, c)
    verdict = f.cells[level.id][c]
    verdict == 0x00 || return verdict == 0x01
    return _classify_void!(f, level, c)
end

# A table miss: classify the cell box through the box memo and record the verdict. Out of line, so the hit path a
# per-cell loop inlines stays the few loads and the compare `_is_void` describes.
@noinline function _classify_void!(f::_Fold{D,T}, level::Level{D,T}, c::CartesianIndex{D}) where {D,T<:Real}
    isassigned(f.cells, level.id) || (f.cells[level.id] = zeros(UInt8, level.mesh.cells))
    void = classify_cell(f.physical, cell_box(level.mesh, c), f.boxes) === :fictitious
    f.cells[level.id][c] = void ? 0x01 : 0x02
    return void
end

# Whether cell `c` of `level` is inactive and holds no material: every cell the fold switched off, and a user-masked
# cell lying entirely outside Ω. An inactive cell for which this is false holds material that only a coarser level
# represents.
function _is_fictitious(f::_Fold{D,T}, level::Level{D,T}, c::CartesianIndex{D}) where {D,T<:Real}
    is_active(level.mask, c) && return false
    return _is_void(f, level, c)
end

# Whether the `side` face of `level`'s own box holds no material where boundary cell `c` meets it: no point of that cell
# face lies in Ω. Asked by the box-face rule (`_clamps_box_face`, `dofs.jl`), once per raw on the face. Inlined like the
# cell query above, so the hit path is an `isassigned` test, a table load and a compare; a face's table is allocated on
# its first miss, and without a physical domain the answer is a compile-time `false`.
@inline function _is_void_face(f::_Fold{D,T}, level::Level{D,T}, c::CartesianIndex{D}, axis::Int,
                               side::Symbol) where {D,T<:Real}
    f.physical === nothing && return false
    slot = side === :lower ? 1 : 2
    isassigned(f.faces, level.id, axis, slot) || return _classify_face!(f, level, c, axis, slot)
    verdict = f.faces[level.id, axis, slot][_face_index(c, axis)]
    verdict == 0x00 || return verdict == 0x01
    return _classify_face!(f, level, c, axis, slot)
end

# A face-table miss: classify the cell's face as a facet region on that face is classified — Ω restricted to the face's
# affine slice, on the face's own (D − 1)-dimensional box (`_classify_facet`, `dirichlet.jl`) — and record the verdict.
@noinline function _classify_face!(f::_Fold{D,T}, level::Level{D,T}, c::CartesianIndex{D}, axis::Int,
                                   slot::Int) where {D,T<:Real}
    isassigned(f.faces, level.id, axis, slot) ||
        (f.faces[level.id, axis, slot] = zeros(UInt8, ntuple(d -> d == axis ? 1 : level.mesh.cells[d], D)))
    box = cell_box(level.mesh, c)
    slice = _restrict_domain(f.physical, ((axis, (slot == 1 ? box.lower : box.upper)[axis]),))
    void = _classify_facet(slice, box.lower, box.upper, [d for d in 1:D if d != axis], Val(D - 1)) === :fictitious
    f.faces[level.id, axis, slot][_face_index(c, axis)] = void ? 0x01 : 0x02
    return void
end

# The slot of cell `c` in a face table, whose `axis` extent is the single boundary layer.
_face_index(c::CartesianIndex{D}, axis::Int) where {D} = CartesianIndex(ntuple(d -> d == axis ? 1 : c[d], D))

# Whether every part of `box` that holds material lies inside `dom`. This is the immersed half of the containment test
# `_covered_by_level` makes, asked of a cell box that overlaps a cover's box without lying inside it. Without a physical
# domain every part of a box holds material, and the answer is the containment test the caller has already failed.
#
# Only a cut box can pass: a `:full` one holds material outside `dom` as well, and a `:fictitious` one is never a cell
# whose coverage a consumer reads — an inactive cell is settled by its own material test. That verdict is the fold's, a
# box-memo hit on any cell the fold classified. What remains is `box \ dom`, the disjoint slabs `_any_outside_slab`
# walks, asked cheapest first:
#
#   1. a sample at each slab's centre and corners. A cut cell whose material crosses the cover's face — the common case
#      wherever the face runs through Ω — is settled by one of them;
#   2. a proof that each slab holds no material (`_is_void_slab`), asked only once every sample came back outside Ω.
#
# Measured on immersed discs and balls under a cover whose face runs through the band of cut cells (`_dof_topology` of a
# folded space over a warm memo): the samples settled every cell whose material crossed the face, and a proof ran only
# on a cell that came out covered. Where the geometry carries a Lipschitz constant the certificate settles those proofs,
# and the whole test was lost in the noise of a 3D build of 30 ms. Without one, the proofs added 17 and 44 ms to such
# builds, against folds of 1.5–1.6 s; asking for the material part costs nothing where the face crosses Ω.
function _material_inside(f::_Fold{D,T}, box::AxisBox{D,T}, dom::AxisBox{D,T}, tol::GeometryTolerance{T}) where {D,T}
    physical = f.physical
    physical === nothing && return false
    box_intersection(box, dom) === nothing && return false
    classify_cell(physical, box, f.boxes) === :cut || return false
    sampled(slab) = any(x -> _inside(physical.geometry, x), (center(slab), _box_vertices(slab)...))
    _any_outside_slab(sampled, box, dom, tol) && return false
    return !_any_outside_slab(slab -> !_is_void_slab(f, slab), box, dom, tol)
end

# Whether `pred` holds for any of the slabs that make up `box \ dom`, `box` overlapping `dom`: on each axis in turn, the
# part of `box` below `dom` and the part above it, every earlier axis already clipped to `dom`, so the slabs are
# disjoint and each point of `box` outside `dom` lies in exactly one. A side `box` overhangs by no more than
# `tol.contain` has no slab, as `is_inside` reads the same overhang. Stops at the first slab for which `pred` is `true`.
function _any_outside_slab(pred::F, box::AxisBox{D,T}, dom::AxisBox{D,T}, tol::GeometryTolerance{T}) where {F,D,T}
    lo, hi = box.lower, box.upper
    for d in 1:D
        if lo[d] < dom.lower[d] - tol.contain
            pred(AxisBox{D,T}(lo, Base.setindex(hi, dom.lower[d], d))) && return true
            lo = Base.setindex(lo, dom.lower[d], d)
        end
        if hi[d] > dom.upper[d] + tol.contain
            pred(AxisBox{D,T}(Base.setindex(lo, dom.upper[d], d), hi)) && return true
            hi = Base.setindex(hi, dom.upper[d], d)
        end
    end
    return false
end

# Whether the box `slab` — the part of a cell outside its cover, typically far thinner than it is wide — holds no
# material.
#
# The classifier sets its octree depth from a box's longest side and bisects every axis, so a slab handed to it whole is
# resolved across its thickness as finely as along its length, which wastes the factor length / thickness: up to 25× on
# the configurations measured at `_material_inside`, where tiling cut the 3D proofs from 267 to 44 ms. The slab is
# therefore offered whole to the Lipschitz certificate, which settles it in one evaluation wherever it lies clear of ∂Ω,
# and otherwise classified as near-cubic tiles of side max(thickness, `subcell_length_scale`), each at the depth its own
# size calls for, stopping at the first tile that holds material.
#
# The tiles bypass the box memo. None of them is asked twice — what the build reuses is the cell's coverage verdict,
# which `_covered_by!` stores — and the memo is the one `prepare`'s integration plan reads afterwards.
function _is_void_slab(f::_Fold{D,T}, slab::AxisBox{D,T}) where {D,T}
    physical = f.physical
    _tri(physical.geometry, slab) == -1 && return true
    extent = slab.upper - slab.lower
    side = max(minimum(extent), physical.subcell_length_scale)
    n = ntuple(d -> max(1, round(Int, extent[d] / side)), Val(D))
    # The `j`-th tile boundary on axis `d`. The last is the slab's own face, so the tiles close the slab exactly.
    at(d, j) = j == n[d] ? slab.upper[d] : slab.lower[d] + extent[d] * j / n[d]
    for i in CartesianIndices(n)
        lower = SVector{D,T}(ntuple(d -> at(d, i[d] - 1), Val(D)))
        upper = SVector{D,T}(ntuple(d -> at(d, i[d]), Val(D)))
        classify_cell(physical, AxisBox{D,T}(lower, upper)) === :fictitious || return false
    end
    return true
end

# ── Coverage ──────────────────────────────────────────────────────────────────

"""
    Coverage{D}

Per-level covered-cell masks for a [`Space`](@ref). `covered[level_id]` is a `BitArray{D}` over the level's cells,
`true` where the part of the cell that holds material — the whole cell, or on an immersed space its share of Ω — lies
inside the active region of a single higher-id level. Built by [`build_coverage`](@ref) and consumed by the pruning
constraint source `_coverage_constraints!` in `dofs.jl`.

The flag is only computed where that consumer can read it — on a masked level, the active cells and their one-cell
∞-norm halo (see `_coverage_cells`). Outside that set the entry keeps its `false` default and carries no information; it
is not a claim that the cell is uncovered. The level-mesh records of post-processing (`_level_grid` in
`postprocessing.jl`, which `write_vtk` writes) also read the flag, for *every* cell of a level, so their `covered` cell
array shows `0` on the masked level's far-away inactive cells whatever their true coverage; read it alongside the
`active` array it is written next to.
"""
struct Coverage{D}
    covered::Dict{Int,BitArray{D}}
end

"""
    build_coverage(V::Space, tol) -> Coverage
    build_coverage(V::Space, tol, fold) -> Coverage

Compute, for every level of `V`, the cells whose material a higher-id level's active region contains: the whole cell
without a physical domain, the cell's share of Ω with one. Containment is tested one covering level at a time (a cell
counts as covered when a single level above contains it); a cell covered only by the *union* of two abutting overlays
is treated as uncovered — a conservative, safe choice, since its high-order modes then stay active.

`fold` is the material oracle of `V` (`_Fold`, built over a cell-classification cache). It is consulted for two things,
both about material the covering level may not reach: an inactive overlapped cell of a masked level, to tell one that
holds no material (which covers) from one that does (which does not), and the part of a cut cell lying outside the
covering level's box, which must hold none — see `_covered_by_level`. The dof layer passes the oracle it built over the
cache `prepare` already filled during the fold, so no cell box is classified twice; the two-argument form builds one
over a fresh cache for a standalone call, and nothing is classified when `V.physical === nothing`.

Only the cells `_coverage_cells` selects are evaluated; the rest keep the `false` default, which the pruning rule never
reads. On a masked level that is what keeps the pass proportional to the *active* region rather than to the level's
grid.

Every answer computed on the way that took more than a box test — against a masked cover, or against any cover of an
immersed space — is kept in `fold`'s coverage memo, which the burial test of the same build reads back. The memo is not
keyed by the tolerance, so a `fold` belongs to one build and one `tol`: pass a fresh oracle for another tolerance, or
the stored answers of the first are returned for the second.
"""
function build_coverage(V::Space{D,T}, tol::GeometryTolerance{T}, fold::_Fold{D,T}=_Fold(V)) where {D,T}
    covered = Dict{Int,BitArray{D}}()
    for level in V.levels
        cov = falses(level.mesh.cells)
        above = [k for k in V.levels if k.id > level.id]
        # Behind a barrier, like the burial test (`_coverage_constraints!`): the per-cell pass is inferred and compiled
        # when a level with something above it first reaches it, so a single-level space compiles none of the coverage
        # test — measured on the first `prepare` of an immersed single-level plate, 84 ms of codegen that never runs
        # there (`_covered_by!`, `_covered_by_level`, `_coverage_cells` and the loop).
        isempty(above) || Base.inferencebarrier(_cover_cells!)(cov, level, above, tol, fold)
        covered[level.id] = cov
    end
    return Coverage{D}(covered)
end

# The coverage flags of `level` against the levels `above` it, on the cells `_coverage_cells` selects; every per-cover
# answer goes into `fold`'s coverage memo.
function _cover_cells!(cov::BitArray{D}, level::Level{D,T}, above, tol::GeometryTolerance{T},
                       fold::_Fold{D,T}) where {D,T}
    wanted = _coverage_cells(level)
    for ci in cell_indices(level.mesh)
        wanted[ci] || continue
        cov[ci] = any(k -> _covered_by!(fold, level, ci, k, tol), above)
    end
    return cov
end

# The cells of `level` whose coverage flag can ever be read, as a `BitArray{D}`.
#
# `_coverage_constraints!` (in `dofs.jl`) is the consumer this set is cut for: its candidate test (`_BurialMarks`) reads
# `cov`, on an integrated-Legendre level, exactly on `_incident_cells(key, n)` for the level's raw dof keys. Raw dofs
# are enumerated on *active* cells only (`_enumerate_raws`), and a key born on cell `c` is incident to `c` and — through
# its node factors — to the cells sharing a face, edge, or vertex with `c`. The read set is therefore the active set
# dilated by one cell in the ∞-norm, and computing `cov` anywhere else is work whose answer nobody looks at. The
# remaining `any(cov)` fast-exit in `_coverage_constraints!` is unaffected: it can only lose a `true` that belonged to a
# cell no key is incident to, in which case the loop it guards would have emitted nothing anyway.
#
# An unmasked level is all-active, so its dilation is the whole grid and the pass is the original one.
#
# On a B-spline level the same test reads `cov` over a wider stencil — a spline of degree `p` lives on up to `p + 1`
# cells per axis. On either family it reads only the *active* cells of the stencil, which are the seed of the dilation
# above, and skips the inactive ones outright: a fictitious support cell can lie several cells past the active front,
# where the entry is the uninformative `false` default, and `_reproduced_on_domain` settles it by classification
# instead. So it reads only computed entries.
#
# The dof layer is not the only reader: `_level_grid` in `postprocessing.jl` reads `cov` on every cell of the level for
# its `covered` diagnostic array. That reader is not what the set is sized for, and outside it sees the `false`
# default rather than a computed verdict — a diagnostic gap, documented on `Coverage`, not a correctness one, since no
# constraint is ever derived from an entry nobody computed.
function _coverage_cells(level::Level{D}) where {D}
    n = level.mesh.cells
    level.mask === nothing && return trues(n)
    wanted = falses(n)
    lo, hi = oneunit(CartesianIndex{D}), CartesianIndex(n)
    for c in cell_indices(level.mesh)
        is_active(level.mask, c) || continue
        for nb in max(lo, c-lo):min(hi, c+lo)
            wanted[nb] = true
        end
    end
    return wanted
end

# `_covered_by_level` of cell `c` of `level` against the single level `k`, through the oracle's `covered_by` memo (see
# `_Fold`). The answer is a pure function of the cell box, `k`, the tolerance and the oracle's verdicts, all fixed for
# the build the oracle serves, so a stored verdict is the one a fresh call returns. `build_coverage` fills the memo as
# it decides its flags, and the burial test's condition 1 (`_reproduced_on_domain`) asks the same (cell, cover) pairs
# again once per buried function and cover: on an integrated-Legendre depth-5 ladder (8² p2) over an immersed disc the
# memo cut the `_covered_by_level` evaluations of one layout from 88 440 to 10 492, and on a base under five band-shaped
# overlays around a hole from 13 204 to 5 092.
#
# The pair's table is allocated, every cell "not asked yet", on the pair's first ask. An unmasked cover of a space
# without a physical domain gets none: its answer is a single box test. On an immersed space it is not, for a cut cell
# straddling the cover's face, whose material part the oracle has to settle (`_material_inside`), and that answer is
# the one worth keeping.
function _covered_by!(fold::_Fold{D,T}, level::Level{D,T}, c::CartesianIndex{D}, k::Level{D,T},
                      tol::GeometryTolerance{T}) where {D,T}
    k.mask === nothing && fold.physical === nothing && return _covered_by_level(cell_box(level.mesh, c), k, tol, fold)
    memo = fold.covered_by
    isassigned(memo, level.id, k.id) || (memo[level.id, k.id] = zeros(UInt8, level.mesh.cells))
    table = memo[level.id, k.id]
    verdict = table[c]
    verdict == 0x00 || return verdict == 0x01
    covered = _covered_by_level(cell_box(level.mesh, c), k, tol, fold)
    table[c] = covered ? 0x01 : 0x02
    return covered
end

# The cells of `mesh` that `box` overlaps, as an axis-aligned block. `_axis_cells` (`mesh.jl`) does the per-axis binary
# search, so the cost is O(D·log N + overlapping cells) rather than O(N), and the exactness convention is stated once,
# there.
#
# What belongs here is the ± tol.contain nudge. `box` comes from *another* level's mesh, and an `overlay` box whose axes
# `_mesh_axes` built from its own corners does not reproduce a coarser level's node coordinates bit-for-bit, so a face
# that should land on a node can land a rounding step either side of it. Pulling the interval inward by `tol.contain`
# keeps a box whose face merely touches a cell boundary — or misses it by a rounding step — from claiming the
# neighbouring cell. The ladder's own cross-level map nudges by a few ulps of the coordinates' magnitude instead, a
# margin that must hold at any scale; see `_axis_overlap` in `ladder.jl`.
#
# The name is deliberately not `_overlapping_cells`: that was one underscore from `overlapping_cells`, a public function
# with a different contract (which cells of one *level* overlap a selection on another, free of `GeometryTolerance`: a
# relative rounding margin only).
#
# Used by `_covered_by_level` below, which asks whether every overlapped cell of a masked covering level is active or
# fictitious — the "for every cell of the finer level under this coarser cell" question the block resolution answers —
# and by `_reproduced_on_domain`, which asks it of a support hull. Both may hand in a box that sticks out past the
# covering level's own — a cut cell whose fictitious part does, a hull over a void support cell — and `_axis_cells`
# clips the block to the grid, which is the part of the box the level can say anything about.
function _cells_under_box(mesh::CartesianMesh{D,T}, box::AxisBox{D,T}, tol::GeometryTolerance{T}) where {D,T}
    return CartesianIndices(ntuple(d -> _axis_cells(mesh.axes[d], box.lower[d] + tol.contain,
                                                    box.upper[d] - tol.contain), D))
end

# True iff the part of `box` (a coarser-level cell) that holds material lies inside the covering region of the single
# level `k`: inside k's domain — the whole box, or on an immersed space every part of it holding material
# (`_material_inside`) — and, when k is masked, with every k-cell it overlaps either active or holding no material at
# all. A box inside an unmasked k's domain is settled by the containment test alone.
#
# Why the material part of the box, and not the box. A cut cell whose share of Ω lies inside k's box, while the rest of
# its box sticks out past k's face into fictitious material, has nothing integrated over the part that sticks out, so k
# represents everything on Ω the cell has to. Asking for the whole box left the cell's high-order modes active next to
# k's copies of them on Ω — measured on Ω = {x₁ ≤ 0.91} under a cover ending at x₁ = 0.95, 14, 30 and 52 exact null
# modes at p = 2, 3, 4, and none with the cover's box extended over the cell, which leaves its active cells unchanged.
#
# Why an inactive cell without material still counts as covering. A `LevelMask` merges two deactivation sources
# (`_apply_physical_fold` in `mesh.jl`): the user's `active =` selection, and the fictitious fold that drops cells lying
# entirely outside Ω. What decides here is not which source switched a cell off but whether the cell holds material,
# which `fold` answers (`_Fold`).
#
#   * A cell that holds material — a user mask reaching into Ω — is represented only by the coarser level, so it blocks
#     coverage and the parent keeps its high-order modes.
#   * A cell that holds none — every folded cell, and a user mask lying entirely outside Ω — has nothing integrated over
#     it, so there is nothing there for the parent to represent, and on Ω the parent's high-order modes over that cell
#     are reproduced exactly by the finer level.
#
# Letting a void cell block coverage therefore leaves a parent mode active that the finer level already duplicates on Ω,
# and the assembled operator acquires an exact null mode — a function with ‖v_h‖_{L²(Ω)} = 0, living entirely in the
# fictitious part. This mirrors `_has_overlay_constraint` in `dofs.jl`, which asks the same material question for the
# artificial overlay trace condition. With no physical domain every inactive cell holds material and the original
# active-cells-only rule is recovered without classifying anything.
#
# For the masked case we do NOT scan all of k's cells: `_cells_under_box` resolves the axis-aligned block by binary
# search.
function _covered_by_level(box::AxisBox{D,T}, k::Level{D,T}, tol::GeometryTolerance{T}, fold::_Fold{D,T}) where {D,T}
    is_inside(box, k.mesh.domain, tol) || _material_inside(fold, box, k.mesh.domain, tol) || return false
    k.mask === nothing && return true
    cells = _cells_under_box(k.mesh, box, tol)

    # Pass 1 — one centre sample per inactive cell. A void cell has *no* point in Ω, so a centre inside Ω rules that
    # verdict out, and this is the overwhelmingly common case: a coarse cell abutting a sparsely-populated overlay meets
    # that overlay's user-masked (material-carrying) cells, not its folded ones. Without a physical domain every
    # inactive cell holds material, and the first one decides.
    for kc in cells
        is_active(k.mask, kc) && continue                       # an inactive overlapped cell
        fold.physical === nothing && return false
        _inside(fold.physical.geometry, center(cell_box(k.mesh, kc))) && return false
    end

    # Pass 2 — the exact verdict, for the blocks pass 1 could not settle. Splitting the passes matters because
    # `classify_cell` is not a lookup: the Lipschitz certificate cannot certify a cell as outside Ω once the cell's
    # half-diagonal exceeds the feature it sits in, and the classifier then walks its subdivision budget to the bottom.
    # Interleaved, a block whose leading cells are void would pay that walk for each of them before reaching the
    # material cell that decides the answer; here the cheap pass reaches it first, for at most one classification's
    # worth of samples.
    for kc in cells
        is_active(k.mask, kc) && continue
        _is_void(fold, k, kc) || return false
    end
    return true
end

# ── The burial test's geometric questions ─────────────────────────────────────

# True iff `k` reproduces on Ω the function `level` assembles from the contiguous cell block `cells` — the support of
# one of `level`'s dofs. Nesting and the span-containment question ("does k's basis carry this shape at all?") belong to
# the caller; what is settled here is geometric: does what `k` is *allowed* to represent still contain the function,
# once the cells holding no material have been accounted for on both levels.
#
# Three conditions, and each is a separate failure mode:
#
#   1. Every *active* cell of the support is covered by `k` alone. `level` integrates there, so `k` must reach there —
#      and `_covered_by_level` already applies the material rule to `k`'s own mask and to the cell's own share of Ω,
#      which is all of the cell `k` has to reach.
#   2. Every inactive cell of the support holds no material. An inactive cell generates nothing, so what `level`
#      assembles is the function truncated to its active cells. Over a void cell — a folded one, or a user mask entirely
#      outside Ω — that truncation is invisible on Ω, since nothing is integrated outside Ω, and the two functions agree
#      where it matters. Over a cell that holds material it is not: material there is carried by `level` alone, the
#      truncated function has a jump `k` does not reproduce, and the dedup must not fire.
#   3. No face of `k`'s box that cuts the interior of the support carries a trace condition. `k` is clamped to zero on
#      its artificial boundary only where the box-face rule says so (`_clamps_box_face`): on an active boundary cell
#      whose face holds material. A face position whose cell is inactive, or whose face lies outside Ω, emits nothing.
#      A face strictly inside the support hull is a face the function is generally non-zero on, so one clamped position
#      of `k` along it puts the function outside `k`'s constrained span. Faces at or outside the hull are harmless: the
#      function vanishes there. A mesh-box face lying on ∂Ω_global needs no exemption here even though it too emits
#      nothing — it can never cut the hull, because the hull lies inside the level's own domain, which lies inside the
#      global one. The rule is exactly the face rule of a cover that clamps (integrated Legendre, a B-spline below
#      `:maximal`); a cover that selects by support, which reaches this test only above an integrated-Legendre level,
#      asks the material question one cover cell past the face instead, and the two disagree only where material lies
#      within that cell without touching the face.
#
# Condition 3 is vacuous whenever the whole support lies inside `k`'s box, as every support cell does that condition 1
# found inside it. What makes the rule bite is material `level` has but `k` does not need to reach: a void support cell,
# or the fictitious part of a cut one, sticking out past `k`'s box. There the face may cut the hull, and the function is
# reproduced only if nothing clamps `k` along it.
#
# Condition 1 reads the oracle's coverage memo (`_covered_by!`), which `build_coverage` filled with the same
# `_covered_by_level` answers.
function _reproduced_on_domain(level::Level{D,T}, k::Level{D,T}, cells::CartesianIndices{D}, tol::GeometryTolerance{T},
                               fold::_Fold{D,T}) where {D,T}
    for ci in cells
        if is_active(level.mask, ci)
            _covered_by!(fold, level, ci, k, tol) || return false
        else
            _is_void(fold, level, ci) || return false
        end
    end
    hull = AxisBox{D,T}(cell_box(level.mesh, first(cells)).lower, cell_box(level.mesh, last(cells)).upper)
    for d in 1:D
        faces = ((1, k.mesh.domain.lower[d], :lower), (k.mesh.cells[d], k.mesh.domain.upper[d], :upper))
        for (j, x, side) in faces                               # k's two box faces on axis d
            hull.lower[d] + tol.contain < x < hull.upper[d] - tol.contain || continue
            perp = _cells_under_box(k.mesh, hull, tol)
            layer = CartesianIndices(ntuple(e -> e == d ? (j:j) : perp.indices[e], D))
            any(kc -> _clamps_box_face(fold, k, kc, d, side), layer) && return false
        end
    end
    return true
end

# True iff `inner`'s mesh is nested inside `outer`'s over their overlap: every `inner` node coordinate lying within
# `outer`'s domain also appears among `outer`'s node coordinates, per axis. That puts every breakpoint of `inner`'s
# multilinear vertex functions on an `outer` cell boundary, so each is multilinear on each `outer` cell — the geometric
# half of the linear-dedup condition. The other half belongs to `outer`'s basis, which must also carry the vertex
# function's kink; `_coverage_constraints!` (`dofs.jl`) tests both before deduping, the basis half as `_reproduces`. The
# B-spline family reuses the same geometric half — nesting is what puts each of its interior knots on an `outer` cell
# boundary — and pairs it with its own span-containment answer.
function _nested_over(inner::Level{D,T}, outer::Level{D,T}, tol::GeometryTolerance{T}) where {D,T}
    for d in 1:D
        lo, hi = outer.mesh.domain.lower[d], outer.mesh.domain.upper[d]
        for x in inner.mesh.axes[d]
            (lo - tol.contain <= x <= hi + tol.contain) || continue
            any(y -> abs(x - y) <= tol.merge, outer.mesh.axes[d]) || return false
        end
    end
    return true
end
