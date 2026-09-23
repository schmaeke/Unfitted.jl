# Per-level coverage of the superposition stack.
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
# one deactivation that still covers is the fictitious fold, because a cell outside Ω
# carries no material at all; see the comment on `_covered_by_level`.
#
# The fold is exempt on *both* sides of the question, and the two exemptions are
# separate functions:
#
#   * on the covering level, a fictitious cell of the cover does not stop it covering
#     (`_covered_by_level`);
#   * on the level being pruned, a fictitious cell of the *support* neither has to be
#     covered nor may block the verdict, because nothing is integrated there and the
#     function the level assembles is the same on Ω with it or without it
#     (`_reproduced_on_domain`).
#
# Reading the rule with only the first exemption in mind is how a nested cover ending
# inside a folded coarse cell came to leave both copies of an exactly reproduced mode
# active; `_reproduced_on_domain` is the shared answer, used by the pruning rule in
# `dofs.jl` and by the B-spline family's own in the extension.

"""
    Coverage{D}

Per-level covered-cell masks for a [`Space`](@ref). `covered[level_id]` is a
`BitArray{D}` over the level's cells, `true` where the cell is fully covered by the
active region of the higher-id levels. Built by [`build_coverage`](@ref) and consumed
by the pruning constraint source `_coverage_constraints` in `dofs.jl`.

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

# The cells of `mesh` that `box` overlaps, as an axis-aligned block. Located by
# binary search on the sorted per-axis coordinates, so the cost is
# O(D·log N + overlapping cells) rather than O(N). The ± tol.contain nudges keep
# a box whose face merely touches a cell boundary from claiming the neighbouring
# cell.
#
# Used by `_covered_by_level` below, which asks whether every overlapped cell of
# a masked covering level is active or fictitious — the "for every cell of the
# finer level under this coarser cell" question the block resolution answers.
function _overlapping_cells(mesh::CartesianMesh{D,T}, box::AxisBox{D,T},
                            tol::GeometryTolerance{T}) where {D,T}
    ranges = ntuple(D) do d
        ax = mesh.axes[d]
        last = length(ax) - 1                                   # number of cells on axis d
        lo = clamp(searchsortedlast(ax, box.lower[d] + tol.contain), 1, last)
        hi = clamp(searchsortedlast(ax, box.upper[d] - tol.contain), 1, last)
        lo:hi
    end
    return CartesianIndices(ranges)
end

# True iff `box` (a coarser-level cell) lies inside the covering region of the single
# level `k`: inside k's domain, and — when k is masked — with every k-cell it overlaps
# either active or fully outside Ω. The unmasked fast path is a single box-containment
# test.
#
# Why an inactive cell outside Ω still counts as covering. A `LevelMask` merges two
# deactivation sources (`_apply_physical_fold` in `mesh.jl`): the user's `active =`
# selection, and the fictitious fold that drops cells lying entirely outside Ω. Here
# they mean opposite things.
#
#   * A *user-masked* cell still holds real material. Only the coarser level
#     represents it, so it blocks coverage and the parent keeps its high-order modes.
#   * A *fictitious* cell holds no material at all. Nothing is integrated over it, so
#     there is nothing there for the parent to represent, and on Ω the parent's
#     high-order modes over that cell are reproduced exactly by the finer level.
#
# Letting a fictitious cell block coverage therefore leaves a parent mode active that
# the finer level already duplicates on Ω, and the assembled operator acquires an exact
# null mode — a function with ‖v_h‖_{L²(Ω)} = 0, living entirely in the fictitious part.
# This mirrors `_internal_face_is_physical` in `dofs.jl`, which draws the same
# fold-versus-user-mask distinction for the artificial overlay trace condition. With
# `physical === nothing` there is no fictitious material and the original
# active-cells-only rule is recovered without ever touching `cache`.
#
# For the masked case we do NOT scan all of k's cells: `_overlapping_cells` resolves the
# axis-aligned block by binary search.
function _covered_by_level(box::AxisBox{D,T}, k::Level{D,T}, tol::GeometryTolerance{T}, physical,
                           cache::_ClassifyCache{D,T}) where {D,T}
    is_inside(box, k.mesh.domain, tol) || return false
    k.mask === nothing && return true
    cells = _overlapping_cells(k.mesh, box, tol)

    # Pass 1 — one centre sample per inactive cell. `:fictitious` means *no* point of the
    # cell lies in Ω, so a centre inside Ω rules that verdict out, and this is the
    # overwhelmingly common case: a coarse cell abutting a sparsely-populated overlay
    # meets that overlay's user-masked (material-carrying) cells, not its folded ones.
    for kc in cells
        is_active(k.mask, kc) && continue                       # an inactive overlapped cell
        physical === nothing && return false
        _inside(physical.geometry, center(cell_box(k.mesh, kc))) && return false
    end

    # Pass 2 — the exact verdict, for the blocks pass 1 could not settle. Splitting the
    # passes matters because `classify_cell` is not a lookup: the Lipschitz certificate
    # cannot certify a cell as outside Ω once the cell's half-diagonal exceeds the
    # feature it sits in, and the classifier then walks its subdivision budget to the
    # bottom. Interleaved, a block whose leading cells are fictitious would pay that walk
    # for each of them before reaching the material cell that decides the answer; here
    # the cheap pass reaches it first, for at most one classification's worth of samples.
    for kc in cells
        is_active(k.mask, kc) && continue
        classify_cell(physical, cell_box(k.mesh, kc), cache) === :fictitious || return false
    end
    return true
end

# True iff cell `ci` of `level` was deactivated by the fictitious fold rather than by
# the user's `active =` selection: inactive, and holding no material at all. The two
# sources are merged into one `LevelMask` by `_apply_physical_fold` (`mesh.jl`) and mean
# opposite things everywhere the dof layer looks at them; this is the one test that
# separates them again. With `physical === nothing` there is no fold, so every inactive
# cell is a user mask and `cache` is never touched.
function _is_fictitious(level::Level{D,T}, ci::CartesianIndex{D}, physical,
                        cache::_ClassifyCache{D,T}) where {D,T}
    is_active(level.mask, ci) && return false
    physical === nothing && return false
    return classify_cell(physical, cell_box(level.mesh, ci), cache) === :fictitious
end

# True iff `k` reproduces on Ω the function `level` assembles from the contiguous cell
# block `cells` — the support of one of `level`'s dofs. Nesting and the span-containment
# question ("does k's basis carry this shape at all?") belong to the caller; what is
# settled here is geometric: does what `k` is *allowed* to represent still contain the
# function, once the fold has been accounted for on both levels.
#
# Three conditions, and each is a separate failure mode:
#
#   1. Every *material* (active) cell of the support is covered by `k` alone. `level`
#      integrates there, so `k` must reach there — and `_covered_by_level` already
#      applies the fold rule to `k`'s own mask.
#   2. Every inactive cell of the support is fictitious. An inactive cell generates
#      nothing, so what `level` assembles is the function truncated to its active
#      cells. For a fold that truncation is invisible on Ω — nothing is integrated
#      outside Ω — and the two functions agree where it matters. For a *user*-masked
#      cell it is not: material there is carried by `level` alone, the truncated
#      function has a jump `k` does not reproduce, and the dedup must not fire.
#   3. No face of `k`'s box that cuts the interior of the support carries a trace
#      condition. `k` is clamped to zero on its artificial boundary — but only where
#      its boundary cells are *active*, since a fold face emits no constraint
#      (`_has_overlay_constraint` / `_active_cell_at_face`). A face strictly inside the
#      support hull is a face the function is generally non-zero on, so one active
#      boundary cell of `k` along it puts the function outside `k`'s constrained span.
#      Faces at or outside the hull are harmless: the function vanishes there.
#      A mesh-box face lying on ∂Ω_global needs no exemption here even though it too
#      emits nothing — it can never cut the hull, because the hull lies inside the
#      level's own domain, which lies inside the global one.
#
# Condition 3 is vacuous whenever every support cell is active: condition 1 then puts
# the whole (contiguous, hence hull-filling) block inside `k`'s domain, so no face of
# `k` is strictly inside the hull. It is the fold that makes the rule bite — a support
# whose material part `k` covers while its fictitious part sticks out past `k`'s box.
function _reproduced_on_domain(level::Level{D,T}, k::Level{D,T}, cells::CartesianIndices{D},
                               physical, tol::GeometryTolerance{T},
                               cache::_ClassifyCache{D,T}) where {D,T}
    for ci in cells
        if is_active(level.mask, ci)
            _covered_by_level(cell_box(level.mesh, ci), k, tol, physical, cache) || return false
        else
            _is_fictitious(level, ci, physical, cache) || return false
        end
    end
    hull = AxisBox{D,T}(cell_box(level.mesh, first(cells)).lower,
                        cell_box(level.mesh, last(cells)).upper)
    for d in 1:D
        faces = ((1, k.mesh.domain.lower[d]), (k.mesh.cells[d], k.mesh.domain.upper[d]))
        for (j, x) in faces                                     # k's two box faces on axis d
            hull.lower[d] + tol.contain < x < hull.upper[d] - tol.contain || continue
            perp = _overlapping_cells(k.mesh, hull, tol)
            layer = CartesianIndices(ntuple(e -> e == d ? (j:j) : perp.indices[e], D))
            any(kc -> is_active(k.mask, kc), layer) && return false
        end
    end
    return true
end

"""
    build_coverage(V::Space, tol) -> Coverage
    build_coverage(V::Space, tol, classify_cache) -> Coverage

Compute, for every level of `V`, the cells a higher-id level's active region contains.
Containment is tested one covering level at a time (a cell counts as covered when a
single level above contains it); a cell covered only by the *union* of two abutting
overlays is treated as uncovered — a conservative, safe choice, since its high-order
modes then stay active.

`classify_cache` is the space's cell-classification cache. It is consulted only for
inactive overlapped cells of a masked level, to tell a fictitious fold (which covers)
from a user mask (which does not) — see `_covered_by_level`. Pass the cache `prepare`
already filled during the fold so no cell box is classified twice; the default
allocates a fresh one for standalone calls, and it stays empty and unused when
`V.physical === nothing`.

Only the cells `_coverage_cells` selects are evaluated; the rest keep the `false`
default, which the pruning rule never reads. On a masked level that is what
keeps the pass proportional to the *active* region rather than to the level's grid.
"""
function build_coverage(V::Space{D,T}, tol::GeometryTolerance{T},
                        classify_cache::_ClassifyCache{D,T}=_ClassifyCache{D,T}()) where {D,T}
    covered = Dict{Int,BitArray{D}}()
    for level in V.levels
        cov = falses(level.mesh.cells)
        above = [k for k in V.levels if k.id > level.id]
        if !isempty(above)
            wanted = _coverage_cells(level)
            for ci in cell_indices(level.mesh)
                wanted[ci] || continue
                box = cell_box(level.mesh, ci)
                cov[ci] = any(k -> _covered_by_level(box, k, tol, V.physical, classify_cache),
                              above)
            end
        end
        covered[level.id] = cov
    end
    return Coverage{D}(covered)
end

# The cells of `level` whose coverage flag can ever be read, as a `BitArray{D}`.
#
# `_coverage_constraints` (in `dofs.jl`) is the consumer this set is cut for, and it
# tests `cov` exactly on `_incident_cells(key, n)` for the level's raw dof keys. Raw
# dofs are enumerated on *active* cells only (`dof_layout`, stage 1), and a key born on
# cell `c` is incident to `c` and — through its node factors — to the cells sharing a
# face, edge, or vertex with `c`. The read set is therefore the active set dilated by
# one cell in the ∞-norm, and computing `cov` anywhere else is work whose answer nobody
# looks at. The remaining `any(cov)` fast-exit in `_coverage_constraints` is unaffected:
# it can only lose a `true` that belonged to a cell no key is incident to, in which case
# the loop it guards would have emitted nothing anyway.
#
# An unmasked level is all-active, so its dilation is the whole grid and the pass is the
# original one.
#
# The B-spline family's own `_coverage_constraints` (in the extension) reads `cov` over a
# wider stencil — a spline of degree `p` lives on up to `p + 1` cells per axis. Both
# consumers read it only on the *active* cells of that stencil, which are the seed of the
# dilation above, and skip the inactive ones outright: a fictitious support cell can lie
# several cells past the active front, where the entry is the uninformative `false`
# default, and `_reproduced_on_domain` settles it by classification instead. So both read
# only computed entries.
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
# basis, which must also carry the vertex function's kink; `_coverage_constraints`
# (`dofs.jl`) tests both before deduping. The B-spline family reuses the same geometric
# half — nesting is what puts each of its interior knots on an `outer` cell boundary —
# and pairs it with its own span-containment test.
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
