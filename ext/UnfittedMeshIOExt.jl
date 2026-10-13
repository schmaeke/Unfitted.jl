# Signed-distance level sets from boundary meshes (2D polygons / 3D STL).
#
# This extension turns a `BoundaryMesh` B-rep into a scalar signed-distance level set `sdf(x) ≤ 0` inside the solid,
# which the geometry-agnostic implicit quadrature in `Unfitted` then integrates like any other leaf — keeping the
# package's "signed scalar level set" commitment while accepting imported mesh geometry. It handles the two
# closed-boundary cases:
#
#   * `BoundaryMesh{2,T,1}` — a closed segment loop bounding a 2D region (`polyline_mesh(verts; closed=true)`),
#   * `BoundaryMesh{3,T,2}` — a triangle surface bounding a 3D solid (`triangle_mesh`, or an STL via
#     [`stl_levelset`](@ref)).
#
# All of the signed-distance machinery lives here; it loads only when FileIO, MeshIO, and GeometryBasics are present
# alongside Unfitted. The three `BoundaryMesh` geometry helpers it reuses — `_default_normal`, `_cell_midpoint` and
# `_simplex_measure` — come from `src/surface.jl`.
#
# Algorithms (clean-room implementations from the cited sources)
#
#   Closest point on a simplex: projection-and-clamp for a segment; the Voronoi-region method of C. Ericson, "Real-Time
#   Collision Detection", Morgan Kaufmann (2005), §5.1.5, ISBN 978-1-55860-732-3, for a triangle.
#
#   Inside/outside sign (default): the angle-weighted pseudonormal of the closest feature — J. A. Bærentzen, H. Aanæs,
#   "Signed distance computation using the angle weighted pseudonormal", IEEE Trans. Vis. Comput. Graph. 11 (2005)
#   243–253, doi:10.1109/TVCG.2005.49. In 2D a boundary vertex is codimension-1 (two segments meet), so the plain
#   `n₁ + n₂` is already correct for convex and reflex vertices; in 3D a vertex needs the incident-angle weighting.
#   Correct on a watertight, consistently outward-oriented mesh — a precondition this mode verifies before answering
#   (see "Mesh validity").
#
#   Inside/outside sign (`orientation = :winding`, opt-in): the generalized winding number — A. Jacobson, L. Kavan, O.
#   Sorkine-Hornung, "Robust inside-outside segmentation using generalized winding numbers", ACM Trans. Graph. 32 (2013)
#   33, doi:10.1145/2461912.2461916 — robust on imperfect / non-watertight / globally flipped meshes, at O(cells) per
#   query.
#
# Differentiability
#
#   `ForwardDiff.gradient(sdf, x)` returns the oriented closest-point normal: the nearest-cell search runs on the primal
#   point, the distance to the selected cell is evaluated in `Dual` arithmetic, and the sign is a primal ±1 (the exact
#   eikonal unit normal off the surface and medial axis).

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

# Segment: project `p` onto the line and clamp to the endpoints. Branch selection (the clamp) is on the primal part when
# `p` carries `Dual`s.
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
    (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) && return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)
end

# ── Angle-weighted pseudonormal feature weight ────────────────────────────────

# `tol` is a length tolerance (scaled to the mesh, passed in by the functor) for deciding whether the closest point sits
# on a vertex of the cell.
#
# 2D: a boundary vertex joins two segments (codim-1), so equal weight (n₁+n₂) is correct for convex and reflex vertices
# alike — no angle weighting needed.
_feature_weight(::NTuple{2,SVector{2,T}}, cp, tol) where {T} = 1.0

# 3D: weight a triangle by the angle it subtends at `cp` when `cp` is one of its vertices — the Bærentzen–Aanæs
# pseudonormal, correct at reflex (non-convex) vertices where equal weights would flip the sign — and by `π` anywhere
# else. The paper's weights off a vertex are `π` on an edge and `2π` in the face interior, but only the vertex case
# needs them told apart: the cells whose closest points coincide all share the closest feature, so on a closed manifold
# mesh an edge gathers the two triangles meeting there, each at `π` either way, and a face-interior point gathers its
# one triangle, whose pseudonormal `π·n` has the sign of `2π·n` exactly.
function _feature_weight(cell::NTuple{3,SVector{3,T}}, cp, tol) where {T}
    a, b, c = cell
    norm(cp - a) <= tol && return _vertex_angle(a, b, c)
    norm(cp - b) <= tol && return _vertex_angle(b, c, a)
    norm(cp - c) <= tol && return _vertex_angle(c, a, b)
    return Float64(pi)
end

_vertex_angle(v, p, q) = acos(clamp(dot(normalize(p - v), normalize(q - v)), -1.0, 1.0))

# ── Mesh signed-distance functor ──────────────────────────────────────────────

# A boundary mesh prepared for signed-distance queries: the (non-degenerate) cells, a KD-tree over cell midpoints for
# nearest-cell candidate search, the per-cell outward unit normals, the largest cell radius *about its midpoint*
# (measured from the same point the tree is keyed on, which is what turns the nearest-midpoint distance into an exact
# nearest-cell search bound), a mesh-scaled length tolerance `tol` (the slack of that search bound and the
# closest-point-coincidence and vertex tests of the sign, so the query is scale-free), and the sign convention. `D` is
# the embedding dimension (2 or 3).
struct _MeshSDF{D,C,TK}
    cells::Vector{C}
    tree::TK
    normals::Vector{SVector{D,Float64}}
    maxcr::Float64
    tol::Float64
    winding::Bool
end

# Bounding-box diagonal of all vertices, the length every tolerance of the level set is a fraction of.
function _mesh_scale(cells)
    lo = hi = first(first(cells))
    for cell in cells, v in cell
        lo, hi = min.(lo, v), max.(hi, v)
    end
    return norm(hi - lo)
end

# Degenerate (zero-measure) cells have no defined normal and corrupt the sign.
_is_degenerate(cell::NTuple{2,SVector{D,T}}, scale) where {D,T} = norm(cell[2] - cell[1]) <= 1.0e-10 * scale

# A triangle is degenerate when the sine of its smallest angle is below 1e-10 — scale-free, and a property of the
# triangle rather than of how its vertices are listed. `‖ab × ac‖` is twice the area whichever vertex it is taken at,
# and the smallest angle lies between the two longest edges, so twice the area over their product `ℓ₁ℓ₂` (the largest
# of the three pairwise edge products) is that sine. Testing the angle at the first vertex alone made the verdict depend
# on the vertex order: a needle with one edge of 1e-12 was dropped when listed from its blunt corner and kept when
# listed from either end of the short edge. Since every pair of edges multiplies to at most `ℓ₁ℓ₂`, this flags every
# triangle the first-vertex test did, and the needles it missed.
function _is_degenerate(cell::NTuple{3,SVector{3,T}}, scale) where {T}
    a, b, c = cell
    ab, bc, ca = norm(b - a), norm(c - b), norm(a - c)
    return norm(cross(b - a, c - a)) <= 1.0e-10 * max(ab * bc, bc * ca, ca * ab)
end

# ── Mesh validity ─────────────────────────────────────────────────────────────

# The pseudonormal sign is *exact* on a closed, coherently outward-oriented mesh and silently wrong on anything else — a
# hole, a flipped cell or a globally reversed winding all yield a plausible-looking distance with the sign inverted over
# part of space. The precondition is therefore checked, on the cells that survive the degeneracy filter because those
# are the ones the sign field is built from. Three tests cover it:
#
#   * every codimension-1 face is traversed once in each direction (the boundary of the cell chain vanishes). A hole
#     leaves faces traversed once; a flipped cell leaves faces traversed twice the same way. The faces are paired
#     through welded vertices, so a cell two of whose corners weld together is refused before they are: see
#     `_check_watertight`.
#   * no two cells stand on the same corners. Such a pair wound oppositely passes the face test while bounding
#     nothing: see `_check_watertight`.
#   * the enclosed volume is positive. Reversing *every* cell keeps the traversals cancelling, so only this test sees a
#     globally inside-out mesh.
#
# Each is one pass over the cells, paid once when the level set is built.

# Map every vertex of every cell to the index of the first vertex within `tol` of it. Cells do not share vertex indices
# — a mesh read from an STL stores each facet's corners separately — so the face bookkeeping below needs the coincident
# corners identified first. `tol` is loose (see `_check_watertight`) because a binary STL keeps its coordinates in
# `Float32`.
function _weld(cells, tol)
    verts = [v for cell in cells for v in cell]
    tree = KDTree(verts)
    id = zeros(Int, length(verts))
    for i in eachindex(verts)
        id[i] == 0 || continue
        for j in inrange(tree, verts[i], tol)
            id[j] == 0 && (id[j] = i)
        end
    end
    K = length(first(cells))
    return [ntuple(k -> id[(c - 1) * K + k], K) for c in eachindex(cells)]
end

# Directed codimension-1 faces of one welded cell as `(key, ±1)` pairs, the key orientation-free and the sign recording
# which way the cell traverses it: the two endpoints of a segment, the three edges of a triangle. The corners of every
# cell that reaches here are distinct, because `_check_watertight` refuses the others first.
_faces(c::NTuple{2,Int}) = ((c[1], -1), (c[2], 1))
_faces(c::NTuple{3,Int}) = (_edge(c[1], c[2]), _edge(c[2], c[3]), _edge(c[3], c[1]))
_edge(i, j) = i < j ? ((i, j), 1) : ((j, i), -1)

# Faces whose traversals do not cancel — zero on a closed, coherently oriented mesh, one per boundary or per mis-wound
# face otherwise.
function _unpaired_faces(idcells)
    net = Dict(key => 0 for c in idcells for (key, _) in _faces(c))
    for c in idcells, (key, s) in _faces(c)
        net[key] += s
    end
    return count(!iszero, values(net))
end

# Vertices are welded at `1e-6 · scale`, not at the tighter `tol` of the sign tests: a binary STL stores `Float32`
# coordinates, so one corner reached through two facets can differ by an ulp — `1.2e-7 · scale` at the far end of the
# bounding box — and welding tighter than that would report a sound mesh as full of holes.
#
# A cell two of whose corners weld together is refused outright. Its faces between those corners vanish, and the rest
# cancel one another or pair with the mesh, so the face count cannot tell a sliver lying flat in the surface from one
# that dangles: a stray segment off a polygon's corner, a fin standing on an edge, a spike, a speck floating inside the
# solid. The sign, though, is answered from the cells as given, and a dangling sliver that sticks out by more than the
# sign tests' `tol` inverts it far from the mesh: a segment 5e-7 long hanging off a corner of the square [0.2, 0.8]²
# reads three of five outside points at distance 0.1 as inside. Only a cell that survives the degeneracy filter with an
# edge shorter than the weld radius is refused here, and `:winding` answers a mesh that has one.
#
# Two cells on the same welded corners are refused as well. Wound oppositely — a sheet listed in both windings, or the
# face two touching solids share when each is exported closed — they traverse each of their faces once in each
# direction and enclose no volume, so the face and volume tests both pass. Their normals cancel in the pseudonormal,
# though, and every point whose closest feature is the pair reads as outside: two unit boxes sharing the face x = 1
# answer `+0.05` at (1.05, 0.5, 0.5), half a unit inside their union. The boundary of a solid never repeats a cell, so
# no sound mesh is refused. The corners are compared sorted, so neither the order a cell lists them in nor its winding
# matters. They are sorted as a StaticArrays `SVector` and turned back into a tuple (`Tuple ∘ sort ∘ SVector`): Julia
# 1.10, the oldest release the package supports, has no `sort` of a tuple, and sorting a `Vector` instead allocates on
# the heap for every cell, where StaticArrays' `sort` of an `SVector` allocates nothing. The composition is Base's and
# StaticArrays', so the check compiles no function of this module's own.
# A shared face triangulated differently on its two sides, or two bodies that overlap, inverts the sign in the same way
# without repeating a cell; telling them apart needs a self-intersection test, which this check does not attempt, so
# `mesh_levelset`'s docstring states them as limits.
#
# The orientation test is ∮ x·n dA, which the divergence theorem makes `D` times the enclosed volume — positive outward,
# negative inward, and only its sign is read. Each planar cell contributes `measure · (midpoint · n)` exactly, x·n being
# affine over the cell and the midpoint its centroid.
function _check_watertight(cells, normals, scale)
    idcells = _weld(cells, 1.0e-6 * scale)
    shut = count(!allunique, idcells)
    shut == 0 || throw(ArgumentError("mesh has $shut cells with two corners closer than the vertex-weld radius " *
                                     "(1e-6 of the mesh size), where closedness cannot be checked. Repair it, or " *
                                     "pass orientation=:winding"))
    open_faces = _unpaired_faces(idcells)
    open_faces == 0 || throw(ArgumentError("mesh is not closed: $open_faces unpaired cell faces. Repair it, " *
                                           "or pass orientation=:winding"))
    twins = length(idcells) - length(unique(Tuple ∘ sort ∘ SVector, idcells))
    twins == 0 || throw(ArgumentError("mesh has $twins cells on the same corners as another cell (a two-sided " *
                                      "sheet, or a face shared by two solids), which bound no volume. Remove both " *
                                      "copies, or pass orientation=:winding"))
    sum(Unfitted._simplex_measure(c) * dot(Unfitted._cell_midpoint(c), n) for (c, n) in zip(cells, normals)) > 0 ||
        throw(ArgumentError("mesh is inside-out (encloses a negative volume): reverse the cell " *
                            "winding, or pass orientation=:winding"))
    return nothing
end

# Prepare a `BoundaryMesh` for signed-distance queries. Vertices are converted to `Float64` (the sign tests and the
# KD-tree do not need the mesh's eltype), degenerate cells are dropped before anything derived from them is built —
# their normal is undefined and would corrupt the sign — and each surviving cell contributes its normal (the mesh's own,
# renormalised, if it carries one, else the geometric `_default_normal`) and its midpoint to the search tree. The two
# scalars are set here because both are mesh-relative: `maxcr` bounds the candidate search, and `tol` is the length
# tolerance of that bound's slack and of the sign's coincidence and vertex tests, taken from the bounding-box diagonal
# so the query is scale-free.
function _build_mesh_sdf(bmesh::BoundaryMesh{D,T,K}, winding::Bool) where {D,T,K}
    cells = [map(v -> SVector{D,Float64}(v), cell) for cell in bmesh.cells]
    isempty(cells) && throw(ArgumentError("mesh has no cells"))
    overrides = bmesh.normals === nothing ? nothing : [SVector{D,Float64}(n) for n in bmesh.normals]
    # A zero or non-finite override normalises to NaN, and every comparison against NaN is false — the sign test would
    # then read the whole space as inside, silently, so the direction is required to exist here.
    bad = overrides === nothing ? nothing : findfirst(n -> !(0 < norm(n) < Inf), overrides)
    bad === nothing || throw(ArgumentError("mesh normal #$bad is zero or non-finite; give each cell a direction"))
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
    tol = 1.0e-8 * scale     # scale-free length tolerance for the search slack and the sign tests
    winding || _check_watertight(kept, normals, scale)
    return _MeshSDF{D,eltype(kept),typeof(KDTree(midpoints))}(kept, KDTree(midpoints), normals, maxcr, tol, winding)
end

# Generalized winding number sign (opt-in). 2D sums signed segment angles; 3D sums signed solid angles. `abs(w) > 0.5` ⇒
# inside, tolerating a globally flipped orientation as well as non-watertight gaps.
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

# Signed distance from `p` to the mesh. Candidate cells are those whose midpoint lies within
# `nearest_midpoint_distance + maxcr` of `p`, a bound that provably contains the true nearest cell, widened by the
# mesh-relative `tol` so that rounding cannot drop a cell lying exactly on it, at any scale of the mesh. The nearest
# cell is the first candidate at the least distance. The sign is the angle-weighted pseudonormal of the closest
# feature, or, with `winding = true`, the generalized winding number.
#
# The scan keeps only the best cell and its distance. The pseudonormal gather computes again the few closest points it
# needs, with the same arithmetic and so to the same bits, which spares every query a vector of them on the heap.
function (m::_MeshSDF{D})(p) where {D}
    pv = SVector{D,Float64}(ntuple(i -> Float64(ForwardDiff.value(p[i])), D))
    candidates = inrange(m.tree, pv, last(nn(m.tree, pv)) + m.maxcr + m.tol)
    best, best_d = first(candidates), Inf
    for i in candidates
        d = norm(pv - _closest_point(m.cells[i], pv))
        d < best_d && (best=i; best_d=d)
    end
    s = m.winding ? _winding_sign(m, pv) : _pseudonormal_sign(m, candidates, pv, _closest_point(m.cells[best], pv))

    # Differentiable distance to the selected nearest cell. The gradient is the oriented unit normal off the surface;
    # exactly on the surface √(d·d) has an infinite derivative (a NaN), which the implicit kernel's gradient sampling
    # tolerates via its non-finite-sample guard.
    cp = _closest_point(m.cells[best], p)
    return s * sqrt(dot(p - cp, p - cp))
end

# The sign of `pv` relative to the closest feature, whose point on the mesh is `best_cp`: the side of the angle-weighted
# pseudonormal it lies on. The cells whose closest point coincides with `best_cp` within `tol` share that feature and
# are summed with their feature weights; a cell merely equidistant, across the medial axis, has its closest point
# elsewhere and is left out.
#
# A closest point lies in its cell, so no farther than `maxcr` from the cell's midpoint, and a cell the gather sums has
# its midpoint within `maxcr + tol` of `best_cp`. That test comes first, with a slack of `2tol` that rounding cannot
# use up, so it never turns away a cell the gather sums: a closest point is recomputed only for the cells it passes,
# about four of the twenty-two candidates of a query on a 512-facet sphere rather than all of them.
function _pseudonormal_sign(m::_MeshSDF{D}, candidates, pv, best_cp) where {D}
    pseudonormal = zero(SVector{D,Float64})
    for i in candidates
        cell = m.cells[i]
        norm(Unfitted._cell_midpoint(cell) - best_cp) <= m.maxcr + 2 * m.tol || continue
        cp = _closest_point(cell, pv)
        norm(cp - best_cp) <= m.tol && (pseudonormal += _feature_weight(cell, cp, m.tol) * m.normals[i])
    end
    return dot(pv - best_cp, pseudonormal) >= 0 ? 1.0 : -1.0
end

# ── Public constructors ───────────────────────────────────────────────────────

"""
    mesh_levelset(mesh::BoundaryMesh; lipschitz=nothing, orientation=:pseudonormal) -> LevelSet

Build a signed-distance [`LevelSet`](@ref) leaf from a closed boundary mesh — negative inside the solid — so
`physical_domain(mesh_levelset(mesh))` integrates the enclosed region and the leaf composes with the CSG combinators
(`intersect`/`union`/`setdiff`/`complement`). Two cases are supported:

  - `BoundaryMesh{2,T,1}` — a closed segment loop bounding a 2D region (build it with
    `polyline_mesh(verts; closed=true)` or `segment_mesh`),
  - `BoundaryMesh{3,T,2}` — a triangle surface bounding a 3D solid (`triangle_mesh`, or [`stl_levelset`](@ref) from a
    file).

The mesh must be watertight and consistently outward-oriented; with the default geometric normals that means
counter-clockwise segment loops in 2D and right-hand-rule triangle winding in 3D (the conventions of `BoundaryMesh`'s
`_default_normal`). Degenerate (zero-measure) cells are dropped, and under `:pseudonormal` the surviving cells are then
checked to be closed (every cell face traversed once in each direction) and to enclose a positive volume — a hole, a
flipped facet or a reversed winding is rejected rather than answered with an inverted sign. Faces are paired through
corners welded within 10⁻⁶ of the bounding-box diagonal (a binary STL stores each facet's corners separately, in
`Float32`). At that scale the check cannot tell a sliver of a closed surface from a dangling one, so a cell with two
corners that close is rejected too, even in an otherwise sound mesh. So are two cells on the same welded corners — a
sheet listed in both windings, or a face shared by two touching solids — which pass both checks while bounding nothing.
The checks do not detect bodies that overlap, or that share a face triangulated differently on its two sides, nor a
separate component wound inside-out beside a larger outward one. The default mode inverts the sign near such faces, so
merge touching or overlapping bodies into the boundary of their union and orient every component outward.

`orientation` selects the inside/outside test:

  - `:pseudonormal` (default) — angle-weighted pseudonormal of the closest feature (Bærentzen & Aanæs 2005). Fast
    (O(candidates) per query), exact for a clean mesh including reflex features.
  - `:winding` — generalized winding number (Jacobson et al. 2013). Robust on imperfect / non-watertight / globally
    flipped meshes, at O(cells) per query, and the only mode that accepts a mesh failing the checks above.

`lipschitz` defaults to the exact constant of what the mode returns: `1.0` under `:pseudonormal`, where the result is a
true signed distance and the cell classifier's certificate is tight, and `Inf` under `:winding`, where a gap in the mesh
puts a jump of the sign at a positive distance from any cell and no finite constant bounds it. Pass `lipschitz = 1.0`
explicitly to keep the tight certificate when `:winding` runs on a watertight mesh.

!!! note
    `triangle_mesh` is exported by both Unfitted and GeometryBasics. Loading the extension via
    `using Unfitted, FileIO, MeshIO` keeps Unfitted's in scope; do not also `using GeometryBasics`, which would shadow
    it.
"""
function mesh_levelset(mesh::BoundaryMesh{D,T,K}; lipschitz::Union{Real,Nothing}=nothing,
                       orientation::Symbol=:pseudonormal) where {D,T,K}
    orientation in (:pseudonormal, :winding) ||
        throw(ArgumentError("orientation must be :pseudonormal or :winding; got :$orientation"))
    (D, K) == (2, 1) ||
        (D, K) == (3, 2) ||
        throw(ArgumentError("mesh_levelset needs a closed boundary mesh: a BoundaryMesh{2,T,1} " *
                            "(2D segment loop) or BoundaryMesh{3,T,2} (3D triangle surface); " *
                            "got D=$D, K=$K"))
    winding = orientation === :winding
    L = lipschitz === nothing ? (winding ? Inf : 1.0) : lipschitz
    return leaf(_build_mesh_sdf(mesh, winding); lipschitz=L)
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
    stl_levelset(path; lipschitz=nothing, orientation=:pseudonormal) -> LevelSet

Read an STL file (ASCII or binary) at `path` and return a signed-distance [`LevelSet`](@ref) leaf for the meshed solid:
it loads the triangles into a `BoundaryMesh{3,T,2}` (`triangle_mesh`) and calls [`mesh_levelset`](@ref).

    using Unfitted, FileIO, MeshIO   # GeometryBasics loads transitively
    Ω = physical_domain(stl_levelset("part.stl"); subcell_length_scale=h)

`lipschitz` and `orientation` are passed through to [`mesh_levelset`](@ref), including its watertightness and
outward-orientation checks — an STL with holes or mis-wound facets is rejected under the default `:pseudonormal`, and
`orientation = :winding` is the robust inside/outside test that accepts it. Accuracy is bounded by the STL faceting:
each planar facet is integrated exactly, while facet edges are creases resolved by the kernel's subdivision.
"""
function stl_levelset(path::AbstractString; lipschitz::Union{Real,Nothing}=nothing, orientation::Symbol=:pseudonormal)
    verts, faces = _load_triangles(path)
    isempty(faces) && throw(ArgumentError("no triangles in $path; expected an ASCII or binary STL"))
    return mesh_levelset(triangle_mesh(verts, faces); lipschitz=lipschitz, orientation=orientation)
end

end # module
