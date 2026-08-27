# Signed-distance level sets from boundary meshes (2D polygons / 3D STL).
#
# This extension turns a `BoundaryMesh` B-rep into a scalar signed-distance
# level set `sdf(x) ≤ 0` inside the solid, which the geometry-agnostic implicit
# quadrature in `Unfitted` then integrates like any other leaf — keeping the
# package's "signed scalar level set" commitment while accepting imported mesh
# geometry. It handles the two closed-boundary cases:
#
#   * `BoundaryMesh{2,T,1}` — a closed segment loop bounding a 2D region
#     (`polyline_mesh(verts; closed=true)`),
#   * `BoundaryMesh{3,T,2}` — a triangle surface bounding a 3D solid
#     (`triangle_mesh`, or an STL via [`stl_levelset`](@ref)).
#
# All of the signed-distance machinery lives here; it loads only when FileIO,
# MeshIO, and GeometryBasics are present alongside Unfitted. The two
# `BoundaryMesh` geometry helpers it reuses — `_default_normal` and
# `_cell_midpoint` — come from `src/surface.jl`.
#
# Algorithms (clean-room implementations from the cited sources)
#
#   Closest point on a simplex: projection-and-clamp for a segment; the
#   Voronoi-region method of C. Ericson, "Real-Time Collision Detection",
#   Morgan Kaufmann (2005), §5.1.5, ISBN 978-1-55860-732-3, for a triangle.
#
#   Inside/outside sign (default): the angle-weighted pseudonormal of the
#   closest feature — J. A. Bærentzen, H. Aanæs, "Signed distance computation
#   using the angle weighted pseudonormal", IEEE Trans. Vis. Comput. Graph. 11
#   (2005) 243–253, doi:10.1109/TVCG.2005.49. In 2D a boundary vertex is
#   codimension-1 (two segments meet), so the plain `n₁ + n₂` is already correct
#   for convex and reflex vertices; in 3D a vertex needs the incident-angle
#   weighting. Correct on a watertight, consistently outward-oriented mesh.
#
#   Inside/outside sign (`orientation = :winding`, opt-in): the generalized
#   winding number — A. Jacobson, L. Kavan, O. Sorkine-Hornung, "Robust
#   inside-outside segmentation using generalized winding numbers", ACM Trans.
#   Graph. 32 (2013) 33, doi:10.1145/2461912.2461916 — robust on imperfect /
#   non-watertight / globally flipped meshes, at O(cells) per query.
#
# Differentiability
#
#   `ForwardDiff.gradient(sdf, x)` returns the oriented closest-point normal:
#   the nearest-cell search runs on the primal point, the distance to the
#   selected cell is evaluated in `Dual` arithmetic, and the sign is a primal
#   ±1 (the exact eikonal unit normal off the surface and medial axis).

module UnfittedMeshIOExt

using Unfitted
using Unfitted: BoundaryMesh, leaf, triangle_mesh
using StaticArrays
using LinearAlgebra
using NearestNeighbors
using ForwardDiff
using FileIO
using MeshIO
using GeometryBasics

import Unfitted: stl_levelset, mesh_levelset

# ── Closest point on a simplex ────────────────────────────────────────────────

# Segment: project `p` onto the line and clamp to the endpoints. Branch
# selection (the clamp) is on the primal part when `p` carries `Dual`s.
function _closest_point(cell::NTuple{2,SVector{D,T}}, p) where {D,T}
    a, b = cell
    ab = b - a
    t = clamp(dot(p - a, ab) / dot(ab, ab), 0, 1)
    return a + t * ab
end

# Triangle (Ericson §5.1.5): classify `p` against the seven Voronoi regions.
function _closest_point(cell::NTuple{3,SVector{3,T}}, p) where {T}
    a, b, c = cell
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    (d1 <= 0 && d2 <= 0) && return a
    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    (d3 >= 0 && d4 <= d3) && return b
    vc = d1 * d4 - d3 * d2
    (vc <= 0 && d1 >= 0 && d3 <= 0) && return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    (d6 >= 0 && d5 <= d6) && return c
    vb = d5 * d2 - d1 * d6
    (vb <= 0 && d2 >= 0 && d6 <= 0) && return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) &&
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)
end

# ── Angle-weighted pseudonormal feature weight ────────────────────────────────

# `tol` is a length tolerance (scaled to the mesh, passed in by the functor) for
# deciding whether the closest point sits on a vertex or an edge of the cell.
#
# 2D: a boundary vertex joins two segments (codim-1), so equal weight (n₁+n₂) is
# correct for convex and reflex vertices alike — no angle weighting needed.
_feature_weight(::NTuple{2,SVector{2,T}}, cp, tol) where {T} = 1.0

# 3D: weight a triangle by the angle it subtends at `cp` — the incident angle at
# a vertex, `π` on an edge, `2π` in the face interior. This is the
# Bærentzen–Aanæs pseudonormal, correct at reflex (non-convex) vertices where
# equal weights would flip the sign.
function _feature_weight(cell::NTuple{3,SVector{3,T}}, cp, tol) where {T}
    a, b, c = cell
    norm(cp - a) <= tol && return _vertex_angle(a, b, c)
    norm(cp - b) <= tol && return _vertex_angle(b, c, a)
    norm(cp - c) <= tol && return _vertex_angle(c, a, b)
    (_on_segment(cp, a, b, tol) || _on_segment(cp, b, c, tol) || _on_segment(cp, c, a, tol)) &&
        return Float64(pi)
    return 2 * Float64(pi)
end

_vertex_angle(v, p, q) = acos(clamp(dot(normalize(p - v), normalize(q - v)), -1.0, 1.0))

function _on_segment(cp, p, q, tol)
    pq = q - p
    l2 = dot(pq, pq)
    l2 <= 0 && return false
    t = dot(cp - p, pq) / l2
    (-tol <= t <= 1 + tol) || return false
    return norm(cp - (p + clamp(t, 0.0, 1.0) * pq)) <= tol
end

# ── Mesh signed-distance functor ──────────────────────────────────────────────

# A boundary mesh prepared for signed-distance queries: the (non-degenerate)
# cells, a KD-tree over cell midpoints for nearest-cell candidate search, the
# per-cell outward unit normals, the largest cell radius *about its midpoint*
# (measured from the same point the tree is keyed on, which is what turns the
# nearest-midpoint distance into an exact nearest-cell search bound), a
# mesh-scaled length tolerance `tol` (for the closest-point-coincidence and
# vertex/edge feature tests, so the sign is scale-free), and the sign
# convention. `D` is the embedding dimension (2 or 3).
struct _MeshSDF{D,C,TK}
    cells::Vector{C}
    tree::TK
    normals::Vector{SVector{D,Float64}}
    maxcr::Float64
    tol::Float64
    winding::Bool
end

# Bounding-box diagonal of all vertices, used as the scale for the degenerate
# test on segments.
function _mesh_scale(cells)
    D = length(first(first(cells)))
    lo = SVector{D,Float64}(ntuple(_ -> Inf, D))
    hi = SVector{D,Float64}(ntuple(_ -> -Inf, D))
    for cell in cells, v in cell
        lo = min.(lo, v)
        hi = max.(hi, v)
    end
    return norm(hi - lo)
end

# Degenerate (zero-measure) cells have no defined normal and corrupt the sign.
function _is_degenerate(cell::NTuple{2,SVector{D,T}}, scale) where {D,T}
    norm(cell[2] - cell[1]) <= 1.0e-10 * scale
end
function _is_degenerate(cell::NTuple{3,SVector{3,T}}, scale) where {T}
    ab = cell[2] - cell[1]
    ac = cell[3] - cell[1]
    return norm(cross(ab, ac)) <= 1.0e-10 * norm(ab) * norm(ac)   # sin θ ≤ tol (scale-free)
end

# Prepare a `BoundaryMesh` for signed-distance queries. Vertices are converted
# to `Float64` (the sign tests and the KD-tree do not need the mesh's eltype),
# degenerate cells are dropped before anything derived from them is built —
# their normal is undefined and would corrupt the sign — and each surviving cell
# contributes its normal (the mesh's own, renormalised, if it carries one, else
# the geometric `_default_normal`) and its midpoint to the search tree. The two
# scalars are set here because both are mesh-relative: `maxcr` bounds the
# candidate search, and `tol` is the length tolerance the vertex/edge feature
# tests use, taken from the bounding-box diagonal so the sign is scale-free.
function _build_mesh_sdf(bmesh::BoundaryMesh{D,T,K}, winding::Bool) where {D,T,K}
    cells = [map(v -> SVector{D,Float64}(v), cell) for cell in bmesh.cells]
    isempty(cells) && throw(ArgumentError("mesh has no cells"))
    overrides = bmesh.normals === nothing ? nothing : [SVector{D,Float64}(n) for n in bmesh.normals]
    scale = _mesh_scale(cells)

    kept = eltype(cells)[]
    normals = SVector{D,Float64}[]
    midpoints = SVector{D,Float64}[]
    maxcr = 0.0
    for (i, cell) in pairs(cells)
        _is_degenerate(cell, scale) && continue
        push!(kept, cell)
        n = overrides === nothing ? Unfitted._default_normal(cell) : normalize(overrides[i])
        push!(normals, n)
        mid = Unfitted._cell_midpoint(cell)
        push!(midpoints, mid)
        maxcr = max(maxcr, maximum(norm(v - mid) for v in cell))
    end
    isempty(kept) && throw(ArgumentError("mesh has no non-degenerate cells"))
    tol = 1.0e-8 * scale     # scale-free length tolerance for the sign tests
    return _MeshSDF{D,eltype(kept),typeof(KDTree(midpoints))}(kept, KDTree(midpoints), normals,
                                                              maxcr, tol, winding)
end

# Generalized winding number sign (opt-in). 2D sums signed segment angles; 3D
# sums signed solid angles. `abs(w) > 0.5` ⇒ inside, tolerating a globally
# flipped orientation as well as non-watertight gaps.
function _winding_sign(m::_MeshSDF{2}, pv)
    total = 0.0
    @inbounds for cell in m.cells
        a = cell[1] - pv
        b = cell[2] - pv
        total += atan(a[1] * b[2] - a[2] * b[1], dot(a, b))
    end
    return abs(total / (2 * Float64(pi))) > 0.5 ? -1.0 : 1.0
end
function _winding_sign(m::_MeshSDF{3}, pv)
    total = 0.0
    @inbounds for cell in m.cells
        a = cell[1] - pv
        b = cell[2] - pv
        c = cell[3] - pv
        la = norm(a)
        lb = norm(b)
        lc = norm(c)
        num = dot(a, cross(b, c))
        den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb
        total += 2 * atan(num, den)
    end
    return abs(total / (4 * Float64(pi))) > 0.5 ? -1.0 : 1.0
end

# Signed distance from `p` to the mesh. Candidate cells are those whose midpoint
# lies within `nearest_midpoint_distance + maxcr` of `p` (a bound that provably
# contains the true nearest cell). The sign is the angle-weighted pseudonormal
# of the closest feature — cells whose closest point coincides in space share
# that feature and are summed with their feature weights; cells merely
# equidistant on the medial axis have a different closest point and are excluded
# — or, with `winding = true`, the generalized winding number.
function (m::_MeshSDF{D})(p) where {D}
    pv = SVector{D,Float64}(ntuple(i -> Float64(ForwardDiff.value(p[i])), D))
    _, nnd = knn(m.tree, pv, 1)
    candidates = inrange(m.tree, pv, nnd[1] + m.maxcr + 1.0e-12)
    isempty(candidates) && (candidates = knn(m.tree, pv, 1)[1])

    # Closest point on each candidate cell (primal), computed once and reused
    # for both the nearest-cell search and the pseudonormal feature gather.
    cpvs = [_closest_point(m.cells[i], pv) for i in candidates]
    best = 1
    best_dv = norm(pv - cpvs[1])
    for k in 2:length(candidates)
        d = norm(pv - cpvs[k])
        d < best_dv && (best=k; best_dv=d)
    end
    best_cpv = cpvs[best]

    if m.winding
        s = _winding_sign(m, pv)
    else
        pseudonormal = zero(SVector{D,Float64})
        for k in eachindex(candidates)
            norm(cpvs[k] - best_cpv) <= m.tol &&
                (pseudonormal += _feature_weight(m.cells[candidates[k]], cpvs[k], m.tol) *
                                 m.normals[candidates[k]])
        end
        s = dot(pv - best_cpv, pseudonormal) >= 0 ? 1.0 : -1.0
    end

    # Differentiable distance to the selected nearest cell. The gradient is the
    # oriented unit normal off the surface; exactly on the surface √(d·d) has an
    # infinite derivative (a NaN), which the implicit kernel's gradient sampling
    # tolerates via its non-finite-sample guard.
    cp = _closest_point(m.cells[candidates[best]], p)
    return s * sqrt(dot(p - cp, p - cp))
end

# ── Public constructors ───────────────────────────────────────────────────────

"""
    mesh_levelset(mesh::BoundaryMesh; lipschitz=1.0, orientation=:pseudonormal) -> LevelSet

Build a signed-distance [`LevelSet`](@ref) leaf from a closed boundary mesh —
negative inside the solid — so `physical_domain(mesh_levelset(mesh))` integrates
the enclosed region and the leaf composes with the CSG combinators
(`intersect`/`union`/`setdiff`/`complement`). Two cases are supported:

  - `BoundaryMesh{2,T,1}` — a closed segment loop bounding a 2D region (build it
    with `polyline_mesh(verts; closed=true)` or `segment_mesh`),
  - `BoundaryMesh{3,T,2}` — a triangle surface bounding a 3D solid
    (`triangle_mesh`, or [`stl_levelset`](@ref) from a file).

The mesh must be watertight and consistently outward-oriented; with the default
geometric normals that means counter-clockwise segment loops in 2D and
right-hand-rule triangle winding in 3D (the conventions of `BoundaryMesh`'s
`_default_normal`). Degenerate (zero-measure) cells are dropped.

`orientation` selects the inside/outside test:

  - `:pseudonormal` (default) — angle-weighted pseudonormal of the closest
    feature (Bærentzen & Aanæs 2005). Fast (O(candidates) per query), exact for
    a clean mesh including reflex features.
  - `:winding` — generalized winding number (Jacobson et al. 2013). Robust on
    imperfect / non-watertight / globally flipped meshes, at O(cells) per query.

`lipschitz = 1.0` is exact for a true signed distance and makes the cell
classifier's certificate tight.

!!! note
    `triangle_mesh` is exported by both Unfitted and GeometryBasics. Loading the
    extension via `using Unfitted, FileIO, MeshIO` keeps Unfitted's in scope; do
    not also `using GeometryBasics`, which would shadow it.
"""
function mesh_levelset(mesh::BoundaryMesh{D,T,K}; lipschitz::Real=1.0,
                       orientation::Symbol=:pseudonormal) where {D,T,K}
    orientation in (:pseudonormal, :winding) ||
        throw(ArgumentError("orientation must be :pseudonormal or :winding; got :$orientation"))
    (D, K) == (2, 1) ||
        (D, K) == (3, 2) ||
        throw(ArgumentError("mesh_levelset needs a closed boundary mesh: a BoundaryMesh{2,T,1} " *
                            "(2D segment loop) or BoundaryMesh{3,T,2} (3D triangle surface); " *
                            "got D=$D, K=$K"))
    return leaf(_build_mesh_sdf(mesh, orientation === :winding); lipschitz=lipschitz)
end

# Read an STL into (vertices, faces) of plain `SVector{3,Float64}` / `NTuple`.
function _load_triangles(path::AbstractString)
    mesh = FileIO.load(path)
    points = GeometryBasics.decompose(GeometryBasics.Point3{Float64}, mesh)
    triangles = GeometryBasics.decompose(GeometryBasics.TriangleFace{Int}, mesh)
    verts = SVector{3,Float64}[SVector{3,Float64}(p[1], p[2], p[3]) for p in points]
    fcs = NTuple{3,Int}[Tuple(t) for t in triangles]
    return verts, fcs
end

"""
    stl_levelset(path; lipschitz=1.0, orientation=:pseudonormal) -> LevelSet

Read an STL file (ASCII or binary) at `path` and return a signed-distance
[`LevelSet`](@ref) leaf for the meshed solid: it loads the triangles into a
`BoundaryMesh{3,T,2}` (`triangle_mesh`) and calls [`mesh_levelset`](@ref).

    using Unfitted, FileIO, MeshIO   # GeometryBasics loads transitively
    Ω = physical_domain(stl_levelset("part.stl"); subcell_length_scale=h)

`lipschitz` and `orientation` are passed through to [`mesh_levelset`](@ref):
`1.0` is the exact Lipschitz constant of a true signed distance, and
`orientation = :winding` is the robust inside/outside test for imperfect /
non-watertight meshes. Accuracy is bounded by the STL faceting: each planar
facet is integrated exactly, while facet edges are creases resolved by the
kernel's subdivision.
"""
function stl_levelset(path::AbstractString; lipschitz::Real=1.0, orientation::Symbol=:pseudonormal)
    verts, faces = _load_triangles(path)
    return mesh_levelset(triangle_mesh(verts, faces); lipschitz=lipschitz, orientation=orientation)
end

end # module
