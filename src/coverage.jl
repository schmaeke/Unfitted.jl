# Per-level coverage of the superposition stack.
#
# The order-reduction rule (see docs/design/covered-cell-deactivation.md) sheds a
# level's high-order modes wherever a finer level fully covers it, keeping only its
# linear skeleton:
#
#   a level-j cell is *covered* iff its box lies inside the region of every level
#   *above* it (higher id) that carries material the finer level represents.
#
# Coverage is mask-aware, so a deactivated overlay does not cover — which is what keeps
# "a fully-deactivated overlay behaves like no overlay" true under order reduction. The
# one deactivation that still covers is the fictitious fold, because a cell outside Ω
# carries no material at all; see the comment on `_covered_by_level`.

"""
    Coverage{D}

Per-level covered-cell masks for a [`Space`](@ref). `covered[level_id]` is a
`BitArray{D}` over the level's cells, `true` where the cell is fully covered by the
active region of the higher-id levels. Built by [`build_coverage`](@ref) and consumed
by the order-reduction constraint source `_coverage_constraints` in `dofs.jl`.
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
function _covered_by_level(box::AxisBox{D,T}, k::Level{D,T}, tol::GeometryTolerance{T},
                           physical, cache::_ClassifyCache{D,T}) where {D,T}
    is_inside(box, k.mesh.domain, tol) || return false
    k.mask === nothing && return true
    ranges = ntuple(D) do d
        ax = k.mesh.axes[d]
        last = length(ax) - 1                                   # number of cells on axis d
        lo = clamp(searchsortedlast(ax, box.lower[d] + tol.contain), 1, last)
        hi = clamp(searchsortedlast(ax, box.upper[d] - tol.contain), 1, last)
        lo:hi
    end
    for kc in CartesianIndices(ranges)
        is_active(k.mask, kc) && continue                       # an inactive overlapped cell
        physical === nothing && return false
        classify_cell(physical, cell_box(k.mesh, kc), cache) === :fictitious || return false
    end
    return true
end

"""
    build_coverage(V::Space, tol) -> Coverage
    build_coverage(V::Space, tol, classify_cache) -> Coverage

Compute, for every level of `V`, the cells covered by the union of the active cells of
all higher-id levels. Uses single-covering-level containment (a cell counts as covered
when one level above contains it); a cell covered only by the *union* of two abutting
overlays is treated as uncovered — a conservative, safe choice (its high-order stays
active) noted in the design.

`classify_cache` is the space's cell-classification cache. It is consulted only for
inactive overlapped cells of a masked level, to tell a fictitious fold (which covers)
from a user mask (which does not) — see `_covered_by_level`. Pass the cache `prepare`
already filled during the fold so no cell box is classified twice; the default
allocates a fresh one for standalone calls, and it stays empty and unused when
`V.physical === nothing`.
"""
function build_coverage(V::Space{D,T}, tol::GeometryTolerance{T},
                        classify_cache::_ClassifyCache{D,T}=_ClassifyCache{D,T}()) where {D,T}
    covered = Dict{Int,BitArray{D}}()
    for level in V.levels
        cov = falses(level.mesh.cells)
        above = [k for k in V.levels if k.id > level.id]
        if !isempty(above)
            for ci in cell_indices(level.mesh)
                box = cell_box(level.mesh, ci)
                cov[ci] = any(k -> _covered_by_level(box, k, tol, V.physical, classify_cache),
                              above)
            end
        end
        covered[level.id] = cov
    end
    return Coverage{D}(covered)
end

# True iff `inner`'s mesh is nested inside `outer`'s over their overlap: every `inner`
# node coordinate lying within `outer`'s domain also appears among `outer`'s node
# coordinates, per axis. Sufficient for `outer` to reproduce `inner`'s multilinear
# vertex functions wherever `outer` covers them — the condition under which a covered
# coarse vertex is an exact duplicate and the linear-dedup rule fires.
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
