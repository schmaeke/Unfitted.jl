# Per-level coverage of the superposition stack, and the material oracle that coverage
# and the dof layer consult about inactive cells.
#
# The pruning rule sheds a level's high-order modes wherever a finer level
# fully covers it, keeping only its linear skeleton:
#
#   a level-j cell is *covered* iff its box lies inside the covering region of a
#   *single* level above it (higher id) — that level's domain, minus the cells it
#   masks off that still carry material.
#
# The rule is per-covering-level on purpose: a cell covered only by the *union* of
# two abutting overlays counts as uncovered. See `build_coverage`.
#
# Coverage is mask-aware, so a deactivated overlay does not cover — which is what keeps
# "a fully-deactivated overlay behaves like no overlay" true under covered-mode pruning. The
# one inactive cell that still covers is one that holds no material at all: every cell the
# fictitious fold switched off, and a user-masked cell lying entirely outside Ω. See the
# comment on `_covered_by_level`.
#
# A cell without material is exempt on *both* sides of the question, and the two
# exemptions are separate functions:
#
#   * on the covering level, an inactive cell of the cover that holds no material does
#     not stop it covering (`_covered_by_level`);
#   * on the level being pruned, an inactive *support* cell that holds no material
#     neither has to be covered nor may block the verdict, because nothing is integrated
#     there and the function the level assembles is the same on Ω with it or without it
#     (`_reproduced_on_domain`).
#
# Reading the rule with only the first exemption in mind is how a nested cover ending
# inside a folded coarse cell came to leave both copies of an exactly reproduced mode
# active; `_reproduced_on_domain` is the shared answer, which the burial driver in
# `dofs.jl` (`_coverage_constraints!`) asks for every family and every cover it judges
# geometrically.
#
# Both exemptions, and the artificial-face tests of the dof layer, ask one question of an
# inactive cell — does it hold material? — and `_Fold` below is the one value that
# answers it.

# ── The material oracle ───────────────────────────────────────────────────────

# Whether a cell holds material, answered for one dof-layout build.
#
# A `LevelMask` merges two deactivations, the caller's `active =` selection and the
# fictitious fold (`_apply_physical_fold`, `mesh.jl`). Every site of the dof layer that
# looks behind an inactive cell — coverage, the burial test's level side, the
# integrated-Legendre artificial face, the B-spline admissible region and trace faces —
# asks the same question of it: does it hold material? A cell without any carries
# nothing to represent and no trace to keep continuous; a cell with some is represented
# by a coarser level only. The answer is a fact about the geometry and the cell box —
# `classify_cell` reports the box `:fictitious` — and not about which mask switched the
# cell off: every folded cell is void, and so is a user-masked cell lying entirely
# outside Ω.
#
# Fields:
#
#   * `physical` — the space's physical domain under its CONCRETE type `P` (`Nothing`
#     without one). `Space.physical` is abstractly typed; building a `_Fold` from its
#     run-time value is the function barrier behind which every query below is a static
#     call, and with `P = Nothing` every query's `physical === nothing` test is a
#     compile-time constant, so it answers without touching the fields below.
#   * `boxes` — the box memo `classify_cell` fills, keyed by the box corners. It is the
#     caller's: `prepare` hands in the one its fold filled and its integration plan read,
#     so a cell the fold classified costs a lookup here; a standalone call passes a fresh
#     one.
#   * `cells` — per level id, a per-cell verdict table (`0x00` not asked yet, `0x01`
#     void, `0x02` material), allocated on the level's first miss. A build asks about
#     the same cell many times — per buried function, per cover, per face position — and
#     a table read is cheaper than hashing the box into `boxes` again.
#   * `covered_by` — per (level id, cover id) pair, the same kind of table for
#     `_covered_by_level` of the level's cells against the cover (`0x01` covered, `0x02`
#     not), which `build_coverage` fills and the burial test reads back; see
#     `_covered_by!`.
#
# Both tables are indexed by level id. Ids are unique within the space the fold was
# built for, the only space it is asked about, and bounded by the largest of them, which
# sizes both up front. `covered_by` is therefore a (largest id)² matrix of which a build
# uses at most the pairs above the diagonal, and in a coupled model, where
# `_prepare_spaces` numbers a later subdomain's levels after the earlier ones', it also
# spans the ids below the space's own. Its slots are unassigned references until asked
# — a pointer each, in one allocation per layout build, pruning or not — which at the
# level counts in use is a few hundred bytes to a few KB.
#
# The verdicts are exactly `classify_cell`'s: a table miss makes the same
# `classify_cell(physical, cell_box(mesh, c), boxes)` call with the same box and memo, a
# hit returns the stored result of that call, and the call is a pure function of the
# geometry and the box. Only how often a cell is asked changes. The same holds for
# `covered_by`, a pure function of the cell box, the cover, the tolerance — one per
# build — and these verdicts.
#
# A `_Fold` is built by `_dof_topology` or `build_coverage` and lives for that one call:
# no `Space`, `Model` or layout keeps it, so its tables are never shared between two
# public calls, and the box memo keeps the owner it has without it.
#
# It lives here rather than beside `_ClassifyCache` in `physical.jl` because its queries
# are about level grids (`Level`, `cell_box`), which load after `physical.jl`, and its
# users are this file, `dofs.jl` and the B-spline extension.
struct _Fold{D,T<:Real,P}
    physical::P
    boxes::_ClassifyCache{D,T}
    cells::Vector{Array{UInt8,D}}
    covered_by::Matrix{Array{UInt8,D}}
end

function _Fold(V::Space{D,T}, boxes::_ClassifyCache{D,T}=_ClassifyCache{D,T}()) where {D,T<:Real}
    n = maximum(level -> level.id, V.levels)
    return _Fold{D,T,typeof(V.physical)}(V.physical, boxes, Vector{Array{UInt8,D}}(undef, n),
                                         Matrix{Array{UInt8,D}}(undef, n, n))
end

# Whether `box`, an arbitrary axis-aligned box, holds no material: the box memo alone,
# with no table, for a box that is not a cell of one of the space's grids — a B-spline
# level's virtual cell past its own face.
function _fictitious(f::_Fold{D,T}, box::AxisBox{D,T}) where {D,T<:Real}
    f.physical === nothing && return false
    return classify_cell(f.physical, box, f.boxes) === :fictitious
end

# Whether cell `c` of `level` holds no material: its box lies entirely outside Ω. A
# fact about the geometry and the box, whatever the cell's activity. Inlined, so the
# hit path a per-cell loop runs — the B-spline admissibility walk above all — is an
# `isassigned` test, a table load and a compare; a level's table is allocated on its
# first miss, so a level nobody asks about, and every level of a space without a
# physical domain, gets none.
@inline function _is_void(f::_Fold{D,T}, level::Level{D,T}, c::CartesianIndex{D}) where {D,T<:Real}
    f.physical === nothing && return false
    isassigned(f.cells, level.id) || return _classify_void!(f, level, c)
    verdict = f.cells[level.id][c]
    verdict == 0x00 || return verdict == 0x01
    return _classify_void!(f, level, c)
end

# A table miss: classify the cell box through the box memo and record the verdict. Out
# of line, so the hit path a per-cell loop inlines stays the few loads and the compare
# `_is_void` describes.
@noinline function _classify_void!(f::_Fold{D,T}, level::Level{D,T},
                                   c::CartesianIndex{D}) where {D,T<:Real}
    isassigned(f.cells, level.id) || (f.cells[level.id] = zeros(UInt8, level.mesh.cells))
    void = classify_cell(f.physical, cell_box(level.mesh, c), f.boxes) === :fictitious
    f.cells[level.id][c] = void ? 0x01 : 0x02
    return void
end

# Whether cell `c` of `level` is inactive and holds no material: every cell the fold
# switched off, and a user-masked cell lying entirely outside Ω. An inactive cell for
# which this is false holds material that only a coarser level represents.
function _is_fictitious(f::_Fold{D,T}, level::Level{D,T}, c::CartesianIndex{D}) where {D,T<:Real}
    is_active(level.mask, c) && return false
    return _is_void(f, level, c)
end

# ── Coverage ──────────────────────────────────────────────────────────────────

"""
    Coverage{D}

Per-level covered-cell masks for a [`Space`](@ref). `covered[level_id]` is a
`BitArray{D}` over the level's cells, `true` where the cell is fully covered by the
active region of the higher-id levels. Built by [`build_coverage`](@ref) and consumed
by the pruning constraint source `_coverage_constraints!` in `dofs.jl`.

The flag is only computed where that consumer can read it — on a masked level, the
active cells and their one-cell ∞-norm halo (see `_coverage_cells`). Outside that set
the entry keeps its `false` default and carries no information; it is not a claim that
the cell is uncovered. The VTK mesh export (`_mesh_vtk_data` in `postprocessing.jl`)
also reads the flag, for *every* cell of a level, so its `covered` cell array shows
`0` on the masked level's far-away inactive cells whatever their true coverage; read
it alongside the `active` array it is written next to.
"""
struct Coverage{D}
    covered::Dict{Int,BitArray{D}}
end

# The cells of `mesh` that `box` overlaps, as an axis-aligned block. `_axis_cells`
# (`mesh.jl`) does the per-axis binary search, so the cost is O(D·log N +
# overlapping cells) rather than O(N), and the exactness convention is stated
# once, there.
#
# What belongs here is the ± tol.contain nudge. `box` comes from *another*
# level's mesh, and an `overlay` box whose axes `_mesh_axes` built from its own
# corners does not reproduce a coarser level's node coordinates bit-for-bit, so
# a face that should land on a node can land a rounding step either side of it.
# Pulling the interval inward by `tol.contain` keeps a box whose face merely
# touches a cell boundary — or misses it by a rounding step — from claiming the
# neighbouring cell. The ladder's own cross-level map passes its endpoints
# exactly for the opposite reason; see `overlapping_cells` in `ladder.jl`.
#
# The name is deliberately not `_overlapping_cells`: that was one underscore
# from `overlapping_cells`, a public function with a different contract (which
# cells of one *level* overlap a selection on another, tolerance-free).
#
# Used by `_covered_by_level` below, which asks whether every overlapped cell of
# a masked covering level is active or fictitious — the "for every cell of the
# finer level under this coarser cell" question the block resolution answers —
# and by `_reproduced_on_domain`, which asks it of a support hull that may stick
# out past the covering level's box.
function _cells_under_box(mesh::CartesianMesh{D,T}, box::AxisBox{D,T},
                          tol::GeometryTolerance{T}) where {D,T}
    return CartesianIndices(ntuple(d -> _axis_cells(mesh.axes[d], box.lower[d] + tol.contain,
                                                    box.upper[d] - tol.contain), D))
end

# True iff `box` (a coarser-level cell) lies inside the covering region of the single
# level `k`: inside k's domain, and — when k is masked — with every k-cell it overlaps
# either active or holding no material at all. The unmasked fast path is a single
# box-containment test.
#
# Why an inactive cell without material still counts as covering. A `LevelMask` merges
# two deactivation sources (`_apply_physical_fold` in `mesh.jl`): the user's `active =`
# selection, and the fictitious fold that drops cells lying entirely outside Ω. What
# decides here is not which source switched a cell off but whether the cell holds
# material, which `fold` answers (`_Fold`).
#
#   * A cell that holds material — a user mask reaching into Ω — is represented only by
#     the coarser level, so it blocks coverage and the parent keeps its high-order modes.
#   * A cell that holds none — every folded cell, and a user mask lying entirely outside
#     Ω — has nothing integrated over it, so there is nothing there for the parent to
#     represent, and on Ω the parent's high-order modes over that cell are reproduced
#     exactly by the finer level.
#
# Letting a void cell block coverage therefore leaves a parent mode active that the
# finer level already duplicates on Ω, and the assembled operator acquires an exact null
# mode — a function with ‖v_h‖_{L²(Ω)} = 0, living entirely in the fictitious part.
# This mirrors `_has_overlay_constraint` in `dofs.jl`, which asks the same material
# question for the artificial overlay trace condition. With no physical domain every
# inactive cell holds material and the original active-cells-only rule is recovered
# without classifying anything.
#
# For the masked case we do NOT scan all of k's cells: `_cells_under_box` resolves the
# axis-aligned block by binary search.
function _covered_by_level(box::AxisBox{D,T}, k::Level{D,T}, tol::GeometryTolerance{T},
                           fold::_Fold{D,T}) where {D,T}
    is_inside(box, k.mesh.domain, tol) || return false
    k.mask === nothing && return true
    cells = _cells_under_box(k.mesh, box, tol)

    # Pass 1 — one centre sample per inactive cell. A void cell has *no* point in Ω, so a
    # centre inside Ω rules that verdict out, and this is the overwhelmingly common case:
    # a coarse cell abutting a sparsely-populated overlay meets that overlay's user-masked
    # (material-carrying) cells, not its folded ones. Without a physical domain every
    # inactive cell holds material, and the first one decides.
    for kc in cells
        is_active(k.mask, kc) && continue                       # an inactive overlapped cell
        fold.physical === nothing && return false
        _inside(fold.physical.geometry, center(cell_box(k.mesh, kc))) && return false
    end

    # Pass 2 — the exact verdict, for the blocks pass 1 could not settle. Splitting the
    # passes matters because `classify_cell` is not a lookup: the Lipschitz certificate
    # cannot certify a cell as outside Ω once the cell's half-diagonal exceeds the
    # feature it sits in, and the classifier then walks its subdivision budget to the
    # bottom. Interleaved, a block whose leading cells are void would pay that walk for
    # each of them before reaching the material cell that decides the answer; here the
    # cheap pass reaches it first, for at most one classification's worth of samples.
    for kc in cells
        is_active(k.mask, kc) && continue
        _is_void(fold, k, kc) || return false
    end
    return true
end

# `_covered_by_level` of cell `c` of `level` against the single level `k`, through the
# oracle's `covered_by` memo (see `_Fold`). The answer is a pure function of the cell
# box, `k`, the tolerance and the oracle's verdicts, all fixed for the build the oracle
# serves, so a stored verdict is the one a fresh call returns. `build_coverage` fills
# the memo as it decides its flags, and the burial test's condition 1
# (`_reproduced_on_domain`) asks the same (cell, cover) pairs again once per buried
# function and cover: on an integrated-Legendre depth-5 ladder (8² p2) over an immersed
# disc the memo cut the `_covered_by_level` evaluations of one layout from 88 440 to
# 10 492, and on a base under five band-shaped overlays around a hole from 13 204 to
# 5 092.
#
# The pair's table is allocated, every cell "not asked yet", on the pair's first ask. An
# unmasked cover gets none: its answer is a single box test.
function _covered_by!(fold::_Fold{D,T}, level::Level{D,T}, c::CartesianIndex{D}, k::Level{D,T},
                      tol::GeometryTolerance{T}) where {D,T}
    k.mask === nothing && return _covered_by_level(cell_box(level.mesh, c), k, tol, fold)
    memo = fold.covered_by
    isassigned(memo, level.id, k.id) || (memo[level.id, k.id] = zeros(UInt8, level.mesh.cells))
    table = memo[level.id, k.id]
    verdict = table[c]
    verdict == 0x00 || return verdict == 0x01
    covered = _covered_by_level(cell_box(level.mesh, c), k, tol, fold)
    table[c] = covered ? 0x01 : 0x02
    return covered
end

# True iff `k` reproduces on Ω the function `level` assembles from the contiguous cell
# block `cells` — the support of one of `level`'s dofs. Nesting and the span-containment
# question ("does k's basis carry this shape at all?") belong to the caller; what is
# settled here is geometric: does what `k` is *allowed* to represent still contain the
# function, once the cells holding no material have been accounted for on both levels.
#
# Three conditions, and each is a separate failure mode:
#
#   1. Every *active* cell of the support is covered by `k` alone. `level` integrates
#      there, so `k` must reach there — and `_covered_by_level` already applies the
#      material rule to `k`'s own mask.
#   2. Every inactive cell of the support holds no material. An inactive cell generates
#      nothing, so what `level` assembles is the function truncated to its active
#      cells. Over a void cell — a folded one, or a user mask entirely outside Ω — that
#      truncation is invisible on Ω, since nothing is integrated outside Ω, and the two
#      functions agree where it matters. Over a cell that holds material it is not:
#      material there is carried by `level` alone, the truncated function has a jump
#      `k` does not reproduce, and the dedup must not fire.
#   3. No face of `k`'s box that cuts the interior of the support carries a trace
#      condition. `k` is clamped to zero on its artificial boundary — but only where
#      its boundary cells are *active*, since a face whose inactive side is void emits
#      no constraint (`_has_overlay_constraint` / `_active_cell_at_face`). A face
#      strictly inside the support hull is a face the function is generally non-zero
#      on, so one active boundary cell of `k` along it puts the function outside `k`'s
#      constrained span. Faces at or outside the hull are harmless: the function
#      vanishes there. A mesh-box face lying on ∂Ω_global needs no exemption here even
#      though it too emits nothing — it can never cut the hull, because the hull lies
#      inside the level's own domain, which lies inside the global one.
#
# Condition 3 is vacuous whenever every support cell is active: condition 1 then puts
# the whole (contiguous, hence hull-filling) block inside `k`'s domain, so no face of
# `k` is strictly inside the hull. It is the void cells that make the rule bite — a
# support whose material part `k` covers while its fictitious part sticks out past `k`'s
# box.
#
# Condition 1 reads the oracle's coverage memo (`_covered_by!`), which `build_coverage`
# filled with the same `_covered_by_level` answers.
function _reproduced_on_domain(level::Level{D,T}, k::Level{D,T}, cells::CartesianIndices{D},
                               tol::GeometryTolerance{T}, fold::_Fold{D,T}) where {D,T}
    for ci in cells
        if is_active(level.mask, ci)
            _covered_by!(fold, level, ci, k, tol) || return false
        else
            _is_void(fold, level, ci) || return false
        end
    end
    hull = AxisBox{D,T}(cell_box(level.mesh, first(cells)).lower,
                        cell_box(level.mesh, last(cells)).upper)
    for d in 1:D
        faces = ((1, k.mesh.domain.lower[d]), (k.mesh.cells[d], k.mesh.domain.upper[d]))
        for (j, x) in faces                                     # k's two box faces on axis d
            hull.lower[d] + tol.contain < x < hull.upper[d] - tol.contain || continue
            perp = _cells_under_box(k.mesh, hull, tol)
            layer = CartesianIndices(ntuple(e -> e == d ? (j:j) : perp.indices[e], D))
            any(kc -> is_active(k.mask, kc), layer) && return false
        end
    end
    return true
end

"""
    build_coverage(V::Space, tol) -> Coverage
    build_coverage(V::Space, tol, fold) -> Coverage

Compute, for every level of `V`, the cells a higher-id level's active region contains.
Containment is tested one covering level at a time (a cell counts as covered when a
single level above contains it); a cell covered only by the *union* of two abutting
overlays is treated as uncovered — a conservative, safe choice, since its high-order
modes then stay active.

`fold` is the material oracle of `V` (`_Fold`, built over a cell-classification
cache). It is consulted only for inactive overlapped cells of a masked level, to tell
one that holds no material (which covers) from one that does (which does not) — see
`_covered_by_level`. The dof layer passes the oracle it built over the cache `prepare`
already filled during the fold, so no cell box is classified twice; the two-argument
form builds one over a fresh cache for a standalone call, and nothing is classified
when `V.physical === nothing`.

Only the cells `_coverage_cells` selects are evaluated; the rest keep the `false`
default, which the pruning rule never reads. On a masked level that is what
keeps the pass proportional to the *active* region rather than to the level's grid.

Every answer computed on the way against a masked cover is kept in `fold`'s coverage
memo, which the burial test of the same build reads back; an unmasked cover's answer
is a single box test and is not stored. The memo is not keyed by the tolerance, so a
`fold` belongs to one build and one `tol`: pass a fresh oracle for another tolerance,
or the stored answers of the first are returned for the second.
"""
function build_coverage(V::Space{D,T}, tol::GeometryTolerance{T},
                        fold::_Fold{D,T}=_Fold(V)) where {D,T}
    covered = Dict{Int,BitArray{D}}()
    for level in V.levels
        cov = falses(level.mesh.cells)
        above = [k for k in V.levels if k.id > level.id]
        # Behind a barrier, like the burial test (`_coverage_constraints!`): the per-cell
        # pass is inferred and compiled when a level with something above it first
        # reaches it, so a single-level space compiles none of the coverage test —
        # measured on the first `prepare` of an immersed single-level plate, 84 ms of
        # codegen that never runs there (`_covered_by!`, `_covered_by_level`,
        # `_coverage_cells` and the loop).
        isempty(above) || Base.inferencebarrier(_cover_cells!)(cov, level, above, tol, fold)
        covered[level.id] = cov
    end
    return Coverage{D}(covered)
end

# The coverage flags of `level` against the levels `above` it, on the cells
# `_coverage_cells` selects; every per-cover answer goes into `fold`'s coverage memo.
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
# `_coverage_constraints!` (in `dofs.jl`) is the consumer this set is cut for: its
# candidate test (`_BurialMarks`) reads `cov`, on an integrated-Legendre level, exactly
# on `_incident_cells(key, n)` for the level's raw dof keys. Raw
# dofs are enumerated on *active* cells only (`_enumerate_raws`), and a key born on
# cell `c` is incident to `c` and — through its node factors — to the cells sharing a
# face, edge, or vertex with `c`. The read set is therefore the active set dilated by
# one cell in the ∞-norm, and computing `cov` anywhere else is work whose answer nobody
# looks at. The remaining `any(cov)` fast-exit in `_coverage_constraints!` is unaffected:
# it can only lose a `true` that belonged to a cell no key is incident to, in which case
# the loop it guards would have emitted nothing anyway.
#
# An unmasked level is all-active, so its dilation is the whole grid and the pass is the
# original one.
#
# On a B-spline level the same test reads `cov` over a wider stencil — a spline of
# degree `p` lives on up to `p + 1` cells per axis. On either family it reads only the
# *active* cells of the stencil, which are the seed of the dilation above, and skips the
# inactive ones outright: a fictitious support cell can lie several cells past the
# active front, where the entry is the uninformative `false` default, and
# `_reproduced_on_domain` settles it by classification instead. So it reads only
# computed entries.
#
# The dof layer is not the only reader: `_mesh_vtk_data` in `postprocessing.jl` reads
# `cov` on every cell of the level for its `covered` diagnostic array. That reader is
# not what the set is sized for, and outside it sees the `false` default rather than a
# computed verdict — a diagnostic gap, documented on `Coverage`, not a correctness one,
# since no constraint is ever derived from an entry nobody computed.
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

# True iff `inner`'s mesh is nested inside `outer`'s over their overlap: every `inner`
# node coordinate lying within `outer`'s domain also appears among `outer`'s node
# coordinates, per axis. That puts every breakpoint of `inner`'s multilinear vertex
# functions on an `outer` cell boundary, so each is multilinear on each `outer` cell —
# the geometric half of the linear-dedup condition. The other half belongs to `outer`'s
# basis, which must also carry the vertex function's kink; `_coverage_constraints!`
# (`dofs.jl`) tests both before deduping, the basis half as `_reproduces`. The B-spline
# family reuses the same geometric half — nesting is what puts each of its interior knots
# on an `outer` cell boundary — and pairs it with its own span-containment answer.
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
