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
# For the masked case we do NOT scan all of k's cells: the k-cells `box` overlaps form
# an axis-aligned block, located by binary search on k's sorted per-axis coordinates
# (`searchsortedlast`). Cost is O(D·log Nₖ + overlapping cells) rather than O(Nₖ). The
# ± tol.contain nudges keep a box whose face merely touches a cell boundary from
# claiming the neighbouring cell.
function _covered_by_level(box::AxisBox{D,T}, k::Level{D,T}, tol::GeometryTolerance{T}, physical,
                           cache::_ClassifyCache{D,T}) where {D,T}
    is_inside(box, k.mesh.domain, tol) || return false
    k.mask === nothing && return true
    ranges = ntuple(D) do d
        ax = k.mesh.axes[d]
        last = length(ax) - 1                                   # number of cells on axis d
        lo = clamp(searchsortedlast(ax, box.lower[d] + tol.contain), 1, last)
        hi = clamp(searchsortedlast(ax, box.upper[d] - tol.contain), 1, last)
        lo:hi
    end
    cells = CartesianIndices(ranges)

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
# wider stencil — a spline of degree `p` lives on up to `p + 1` cells per axis — but only
# after requiring every one of those cells to be *active* on this level, and the active
# cells are the seed of the dilation above. So it too reads only computed entries.
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
