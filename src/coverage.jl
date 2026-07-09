# Per-level coverage of the superposition stack.
#
# The order-reduction rule (see docs/design/covered-cell-deactivation.md) sheds a
# level's high-order modes wherever a finer level fully covers it, keeping only its
# linear skeleton. "Covered" is pure box geometry — the fictitious fold has already
# removed outside-Ω cells from the level masks, so coverage need not consult Ω:
#
#   a level-j cell is *covered* iff its box lies inside the union of the ACTIVE cells
#   of every level *above* it (higher id).
#
# Coverage is mask-aware, so a deactivated or folded overlay does not cover — which is
# what keeps "a fully-deactivated overlay behaves like no overlay" true under order
# reduction.

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

# True iff `box` (a coarser-level cell) lies inside the active region of the single
# level `k`: inside k's domain, and — when k is masked — with every k-cell it overlaps
# active. The unmasked fast path is a single box-containment test.
#
# For the masked case we do NOT scan all of k's cells: the k-cells `box` overlaps form
# an axis-aligned block, located by binary search on k's sorted per-axis coordinates
# (`searchsortedlast`). Cost is O(D·log Nₖ + overlapping cells) rather than O(Nₖ). The
# ± tol.contain nudges keep a box whose face merely touches a cell boundary from
# claiming the neighbouring cell.
function _covered_by_level(box::AxisBox{D,T}, k::Level{D,T},
                           tol::GeometryTolerance{T}) where {D,T}
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
        is_active(k.mask, kc) || return false                   # an inactive overlapped cell
    end
    return true
end

"""
    build_coverage(V::Space, tol) -> Coverage

Compute, for every level of `V`, the cells covered by the union of the active cells of
all higher-id levels. Uses single-covering-level containment (a cell counts as covered
when one level above contains it); a cell covered only by the *union* of two abutting
overlays is treated as uncovered — a conservative, safe choice (its high-order stays
active) noted in the design.
"""
function build_coverage(V::Space{D,T}, tol::GeometryTolerance{T}) where {D,T}
    covered = Dict{Int,BitArray{D}}()
    for level in V.levels
        cov = falses(level.mesh.cells)
        above = [k for k in V.levels if k.id > level.id]
        if !isempty(above)
            for ci in cell_indices(level.mesh)
                box = cell_box(level.mesh, ci)
                cov[ci] = any(k -> _covered_by_level(box, k, tol), above)
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
function _nested_over(inner::Level{D,T}, outer::Level{D,T},
                      tol::GeometryTolerance{T}) where {D,T}
    for d in 1:D
        lo, hi = outer.mesh.domain.lower[d], outer.mesh.domain.upper[d]
        for x in inner.mesh.axes[d]
            (lo - tol.contain <= x <= hi + tol.contain) || continue
            any(y -> abs(x - y) <= tol.merge, outer.mesh.axes[d]) || return false
        end
    end
    return true
end
