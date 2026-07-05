# User-supplied immersed surface meshes and the surface-region
# machinery that integrates against them. The user owns the geometry:
# they hand us a list of simplices (points / segments / triangles) and
# we map each cell's reference quadrature onto its physical extent,
# locate covering parent cells on every level, and emit a
# `SurfaceRegion` per cell. The strict parent-uniqueness contract
# (every region's parent set is constant within the cell) is enforced
# by an automatic subdivision pass against the cartesian level grids,
# so users never need to align their mesh with the discretization.
#
# What this file owns:
#
#   * `BoundaryMesh{D,T,K}` — the user-facing typed payload.
#   * `segment_mesh`, `polyline_mesh`, `triangle_mesh`, `points_mesh`
#     — helper constructors that accept the common input shapes
#     (vertex lists, face index tables, point clouds).
#   * Per-K simplex geometry: cell measure, default normal,
#     reference-to-physical maps, and reference quadrature rules.
#   * Automatic subdivision of user cells against every level's grid
#     (segments via per-axis parameter crossings, triangles via
#     Sutherland–Hodgman half-space clipping + fan triangulation).
#   * `SurfaceRegion{D,T}` — one per sub-cell, carrying the
#     precomputed physical-frame Gauss points / weights / normals plus
#     the constant covering-parent list.
#   * `_surface_regions_for_mesh` — the construction-time builder that
#     ties subdivision + quadrature + parent classification together.
#
# What lives elsewhere:
#
#   * `block(...; on::BoundaryMesh)` / `loadform(...; on::BoundaryMesh)`
#     — already supported by the unified `on=` kwarg in `problems.jl`.
#   * `_assemble_region!(::SurfaceRegion, …)` and the assembly
#     partitioning that routes `BoundaryMesh`-tagged forms — in
#     `src/assembly.jl`.
#   * `boundary_integral(integrand, model; on::BoundaryMesh)` — in
#     `src/postprocessing.jl`.

# ── BoundaryMesh ──────────────────────────────────────────────────────────────

"""
    BoundaryMesh{D,T,K}(cells, normals)

User-supplied immersed surface mesh: a finite list of `(K+1)`-vertex
simplices embedded in physical space `ℝᴰ`. `K` is the simplex
dimension:

  - `K = 0` — point cloud (codim-`D` evaluation). Useful for
    pinning fields at specific physical coordinates.
  - `K = 1` — line-segment mesh (codim-1 in 2D). Use for arcs or
    polylines in 2D problems.
  - `K = 2` — triangle mesh (codim-1 in 3D). Use for surfaces of
    immersed 3D bodies.

Fields:

  - `cells::Vector{NTuple{K+1,SVector{D,T}}}` — vertex tuples of every
    cell, in physical coordinates. Vertex ordering matters for the
    derived normal (K=1: counter-clockwise / right-hand rule;
    K=2: right-hand rule across the triple cross product).
  - `normals::Union{Nothing,Vector{SVector{D,T}}}` — optional per-cell
    outward unit normal override. `nothing` (default for K≥1) means
    "derive the normal from cell geometry"; an explicit list lets the
    user impose orientation conventions or normals at K=0 points where
    no geometric normal exists.

Construct via [`segment_mesh`](@ref), [`polyline_mesh`](@ref),
[`triangle_mesh`](@ref), or [`points_mesh`](@ref). Pass directly to
[`block`](@ref), [`loadform`](@ref), or [`boundary_integral`](@ref) via
`on = mesh`.

User contract: every cell carries one [`SurfaceRegion`](@ref) per
sub-cell after automatic subdivision against every level's grid; the
package therefore accepts any mesh whose cells lie inside the
discretization on the level whose mesh they overlap. Sub-cells inherit
the parent cell's normal (geometric default or user override).
"""
struct BoundaryMesh{D,T<:Real,K,C}
    cells::Vector{C}
    normals::Union{Nothing,Vector{SVector{D,T}}}

    function BoundaryMesh{D,T,K}(cells::Vector{NTuple{KP,SVector{D,T}}},
                                 normals::Union{Nothing,Vector{SVector{D,T}}}) where {D,T<:Real,K,
                                                                                      KP}
        KP == K + 1 || throw(ArgumentError("BoundaryMesh{$D,$T,$K} expects $(K+1)-vertex cells, " *
                                           "got $KP-vertex tuples"))
        0 <= K <= D ||
            throw(ArgumentError("BoundaryMesh simplex dimension K=$K must satisfy 0 ≤ K ≤ D=$D"))
        D <= 3 ||
            throw(ArgumentError("BoundaryMesh integration is supported only for D ≤ 3; got D=$D"))
        if normals !== nothing
            length(normals) == length(cells) ||
                throw(DimensionMismatch("BoundaryMesh: $(length(normals)) normals supplied for " *
                                        "$(length(cells)) cells"))
        end
        if K == 0 && normals === nothing
            throw(ArgumentError("points_mesh requires explicit normals: a point-cell carries no " *
                                "geometric direction the package could derive"))
        end
        return new{D,T,K,NTuple{KP,SVector{D,T}}}(cells, normals)
    end
end

# ── Helper constructors ──────────────────────────────────────────────────────

"""
    segment_mesh(vertex_pairs::AbstractVector{NTuple{2,SVector{D,T}}};
                 normals = nothing) -> BoundaryMesh{D,T,1}

Build a segment mesh from an explicit list of `(vertex_a, vertex_b)`
pairs. Each pair contributes one `K=1` cell. Use this when the user
already holds the segments as a pair-of-points list (e.g. from a
contour extraction). For a sequence of consecutive points, prefer
[`polyline_mesh`](@ref).
"""
function segment_mesh(vertex_pairs::AbstractVector{NTuple{2,SVector{D,T}}};
                      normals::Union{Nothing,AbstractVector{SVector{D,T}}}=nothing) where {D,T}
    cells = collect(vertex_pairs)
    normals_vec = normals === nothing ? nothing : collect(normals)
    return BoundaryMesh{D,T,1}(cells, normals_vec)
end

"""
    polyline_mesh(vertices::AbstractVector{SVector{D,T}}; closed=false,
                  normals=nothing) -> BoundaryMesh{D,T,1}

Build a segment mesh from consecutive vertices: each adjacent
`(v_i, v_{i+1})` pair becomes one segment. With `closed = true` the
final segment wraps `(v_end, v_1)` so the polyline closes into a
loop (handy for an arc traversed once around `∂Ω`).

`normals`, if supplied, is one normal per resulting segment
(`length(vertices) - 1` open, `length(vertices)` closed).
"""
function polyline_mesh(vertices::AbstractVector{SVector{D,T}}; closed::Bool=false,
                       normals::Union{Nothing,AbstractVector{SVector{D,T}}}=nothing) where {D,T}
    n = length(vertices)
    n >= 2 || throw(ArgumentError("polyline_mesh needs at least 2 vertices"))
    pairs = NTuple{2,SVector{D,T}}[]
    for i in 1:(n-1)
        push!(pairs, (vertices[i], vertices[i + 1]))
    end
    closed && push!(pairs, (vertices[n], vertices[1]))
    return segment_mesh(pairs; normals)
end

"""
    triangle_mesh(vertices::AbstractVector{SVector{D,T}},
                  faces::AbstractVector{NTuple{3,Int}};
                  normals=nothing) -> BoundaryMesh{D,T,2}

Build a triangle mesh from a vertex list and a face-index table
(`faces[i] = (a, b, c)` references one-based indices into `vertices`).
Vertex ordering follows the right-hand rule for the geometric normal
(unless `normals` is supplied to override).
"""
function triangle_mesh(vertices::AbstractVector{SVector{D,T}}, faces::AbstractVector{NTuple{3,Int}};
                       normals::Union{Nothing,AbstractVector{SVector{D,T}}}=nothing) where {D,T}
    cells = NTuple{3,SVector{D,T}}[]
    for (a, b, c) in faces
        all(1 .<= (a, b, c) .<= length(vertices)) || throw(BoundsError(vertices, (a, b, c)))
        push!(cells, (vertices[a], vertices[b], vertices[c]))
    end
    normals_vec = normals === nothing ? nothing : collect(normals)
    return BoundaryMesh{D,T,2}(cells, normals_vec)
end

"""
    points_mesh(points::AbstractVector{SVector{D,T}};
                normals::AbstractVector{SVector{D,T}}) -> BoundaryMesh{D,T,0}

Build a `K = 0` (point-cell) mesh. Each entry of `points` becomes one
cell with a single vertex; `normals` is required because a point
carries no geometric direction. The integration weight per cell is 1
— a `boundary_integral` over a `points_mesh` therefore returns a sum
of `integrand(q)` values at the requested points, with `q.normal`
exposing the user-supplied direction.
"""
function points_mesh(points::AbstractVector{SVector{D,T}};
                     normals::AbstractVector{SVector{D,T}}) where {D,T}
    cells = NTuple{1,SVector{D,T}}[]
    for p in points
        push!(cells, (p,))
    end
    return BoundaryMesh{D,T,0}(cells, collect(normals))
end

# ── Per-K simplex geometry ───────────────────────────────────────────────────
#
# For each simplex dimension K, the package owns three pieces:
#
#   * `_simplex_measure(cell)` — physical volume of the cell (1 for
#     K=0, length for K=1, area for K=2). The Jacobian baked into
#     each Q-point weight is `measure / 2^K` for K=1,2 (matching the
#     reference frame's edge length 2) and just `measure` for K=0.
#   * `_default_normal(cell)` — the geometric outward unit normal
#     derived from cell geometry. Only defined for K=1 in 2D and
#     K=2 in 3D — the cases where a unique normal exists; K=0
#     requires the user to supply one.
#   * `_simplex_reference_quadrature(order)` — `(eta, weight)` pairs
#     in the cell's reference frame, sized by the recommended
#     quadrature order from the basis layer.
#   * `_reference_to_physical_simplex(cell, eta)` — the affine map
#     from the reference cell to the physical cell at a reference
#     point `eta`.

# K = 0: point cell. The single "Q-point" is the point itself with
# weight 1. No reference frame.
_simplex_measure(cell::NTuple{1,SVector{D,T}}) where {D,T} = one(T)

# K = 1: segment cell. Length is the Euclidean norm of the edge
# vector. Reference frame is `[−1, 1]`, so the Jacobian is
# `length / 2`.
_simplex_measure(cell::NTuple{2,SVector{D,T}}) where {D,T} = norm(cell[2] - cell[1])

# K = 2: triangle cell. Area via the cross-product norm of two edges
# (in 2D the cross is a scalar; in 3D it is a vector). Reference
# frame is the standard 2-simplex with vertices `(0,0)`, `(1,0)`,
# `(0,1)` and area 1/2, so the Jacobian is `area / (1/2) = 2 area`
# for our Gauss rule below — see `_simplex_reference_quadrature`.
function _simplex_measure(cell::NTuple{3,SVector{2,T}}) where {T}
    e1 = cell[2] - cell[1]
    e2 = cell[3] - cell[1]
    # 2D cross = scalar; area = |e1 × e2| / 2.
    return abs(e1[1] * e2[2] - e1[2] * e2[1]) / 2
end

function _simplex_measure(cell::NTuple{3,SVector{3,T}}) where {T}
    return norm(cross(cell[2] - cell[1], cell[3] - cell[1])) / 2
end

# Geometric default normal. Throws for K=0 (no normal exists from
# geometry alone).
function _default_normal(cell::NTuple{1,SVector{D,T}}) where {D,T}
    throw(ArgumentError("a K=0 point cell has no geometric normal — supply normals explicitly"))
end

# K=1 in 2D: outward normal is 90° clockwise from the cell direction
# `cell[2] − cell[1]`, normalised. The clockwise convention treats
# the polyline as the oriented boundary of a region to its left.
function _default_normal(cell::NTuple{2,SVector{2,T}}) where {T}
    e = cell[2] - cell[1]
    return normalize(SVector{2,T}(e[2], -e[1]))
end

# K=1 in 3D: a segment in 3D has no canonical normal (the cross
# product needs two non-collinear vectors). Require the user to
# supply one.
function _default_normal(cell::NTuple{2,SVector{3,T}}) where {T}
    throw(ArgumentError("a K=1 segment in 3D has no canonical normal — supply normals explicitly"))
end

# K=2 in 3D: unit cross product of two edges (right-hand rule on the
# vertex ordering).
function _default_normal(cell::NTuple{3,SVector{3,T}}) where {T}
    return normalize(cross(cell[2] - cell[1], cell[3] - cell[1]))
end

# K=2 in 2D: degenerate (a triangle filling a 2D region carries the
# whole plane in its tangent space). Throws — the user should not be
# passing a K=2 mesh in 2D in the first place.
function _default_normal(cell::NTuple{3,SVector{2,T}}) where {T}
    throw(ArgumentError("a K=2 triangle in 2D fills its embedding — supply normals explicitly"))
end

# Reference quadrature rules per K. Each returns
# `Vector{Tuple{eta, w_reference}}` where `eta` is the reference
# coordinate inside the simplex and `w_reference` the reference-frame
# weight. The physical weight is `w_reference × measure / reference_area`.

# K=0: one sample, weight 1 (identity).
function _simplex_reference_quadrature(::Val{0}, ::NTuple{D,Int}, ::Type{T}) where {D,T}
    [(zero(T), one(T))]
end

# K=1: 1D Gauss-Legendre on [−1, 1]. Order is `max(recommended_quadrature_order)`
# over the covering parents (passed in as `quadrature_order`).
function _simplex_reference_quadrature(::Val{1}, quadrature_order::NTuple{D,Int},
                                       ::Type{T}) where {D,T}
    n = maximum(quadrature_order)
    points, weights = gausslegendre(n)
    return [(T(p), T(w)) for (p, w) in zip(points, weights)]
end

# K=2: Duffy-mapped tensor Gauss on the standard 2-simplex
# `{(u, v) : u, v ≥ 0, u + v ≤ 1}`. Uses an `n × n` tensor Gauss rule
# on `[0, 1]²`, mapped via `u = ξ`, `v = η(1 − ξ)` with Jacobian
# `(1 − ξ)`. Exact for polynomials of degree `2n − 1` along each axis
# of the Duffy frame, which covers polynomial integrands of total
# degree `2n − 1` over the triangle.
function _simplex_reference_quadrature(::Val{2}, quadrature_order::NTuple{D,Int},
                                       ::Type{T}) where {D,T}
    n = maximum(quadrature_order)
    pts1d, wts1d = gausslegendre(n)
    # Shift to [0, 1].
    pts01 = [T((p + 1) / 2) for p in pts1d]
    wts01 = [T(w / 2) for w in wts1d]
    samples = Tuple{NTuple{2,T},T}[]
    for i in 1:n, j in 1:n
        xi = pts01[i]
        eta = pts01[j]
        u = xi
        v = eta * (one(T) - xi)
        jacobian = one(T) - xi
        push!(samples, ((u, v), wts01[i] * wts01[j] * jacobian))
    end
    return samples
end

# Reference-area normalisers: the reference cell's volume. The Jacobian
# baked into each physical Q-point's weight is `measure / reference_area`,
# so the reference quadrature weights sum to the physical measure when
# multiplied by `measure / reference_area`.
_reference_area(::Val{0}) = 1
_reference_area(::Val{1}) = 2          # length of [-1, 1]
_reference_area(::Val{2}) = 1 // 2     # area of the standard 2-simplex

# Affine maps from the reference cell to a physical cell. K=0 collapses
# to the vertex; K=1 linearly interpolates between the two endpoints;
# K=2 is a standard barycentric interpolation.
_reference_to_physical_simplex(cell::NTuple{1,SVector{D,T}}, eta) where {D,T} = cell[1]

function _reference_to_physical_simplex(cell::NTuple{2,SVector{D,T}}, eta::T) where {D,T}
    return cell[1] + ((eta + one(T)) / 2) * (cell[2] - cell[1])
end

function _reference_to_physical_simplex(cell::NTuple{3,SVector{D,T}}, eta::NTuple{2,T}) where {D,T}
    u, v = eta
    lambda0 = one(T) - u - v
    return lambda0 * cell[1] + u * cell[2] + v * cell[3]
end

# ── Subdivision against level grids ──────────────────────────────────────────
#
# The strict parent-uniqueness contract on `BoundaryMesh` requires
# every cell to lie inside exactly one parent cell per level — i.e.,
# not straddle any grid line. Rather than ask users to pre-split their
# mesh, the package subdivides every input cell against the union of
# all level grids before building surface regions.
#
# Per simplex dimension:
#
#   * `K = 0` (points)          — trivial, no subdivision needed.
#   * `K = 1` (segments)        — per-axis parameter-value crossings,
#                                 sorted, nudged inward at each
#                                 crossing so sub-segment endpoints
#                                 fall strictly inside the cell that
#                                 contains the sub-segment's interior.
#                                 Works in any `D`.
#   * `K = 2` (triangles in 3D) — Sutherland–Hodgman clipping against
#                                 the six axis-aligned half-spaces of
#                                 every cell the triangle overlaps,
#                                 followed by fan triangulation of the
#                                 resulting convex polygon.
#
# `K = 2` in `D = 2` (triangles in 2D) and `K = 1` in `D = 1` are
# codim-0 integrations — out of scope for boundary integration; cells
# in those configurations pass through unchanged.

# Per-axis sorted unique grid coordinates across an arbitrary level
# collection. A cell that respects this union automatically respects
# every individual level grid, so the contract holds simultaneously.
# The two-space interface builder feeds the *concatenation* of both
# subdomains' levels here so a segment/triangle is split at the merged
# trace of both grids (each sub-cell then lies inside one cut cell of
# each subdomain — the non-matching segment-merge of the references).
function _grid_lines_for_levels(levels, ::Val{D}, tol::GeometryTolerance{T}) where {D,T}
    return ntuple(d -> _merged_axis_coordinates(levels, d, tol), D)
end

# Per-axis sorted unique grid coordinates across every level of `V`.
_level_grid_lines(V::Space{D,T}, tol::GeometryTolerance{T}) where {D,T} =
    _grid_lines_for_levels(V.levels, Val(D), tol)

"""
    _subdivide_segment(p1, p2, grid_lines, nudge)

Split the segment `(p1, p2)` at every grid-line crossing across every
axis. Returns a vector of `(a, b)` `SVector` pairs, each contained in
a single cell of every level (subject to the `nudge` floor). A segment
with no crossings passes through as `[(p1, p2)]`.
"""
function _subdivide_segment(p1::SVector{D,T}, p2::SVector{D,T}, grid_lines::NTuple{D},
                            nudge::T) where {D,T}
    crossings = T[]
    for d in 1:D
        denom = p2[d] - p1[d]
        iszero(denom) && continue
        for c in grid_lines[d]
            t = (c - p1[d]) / denom
            -nudge < t < 1 + nudge && push!(crossings, t)
        end
    end
    isempty(crossings) && return NTuple{2,SVector{D,T}}[(p1, p2)]
    sort!(crossings)

    segments = NTuple{2,SVector{D,T}}[]
    t_start = zero(T)
    for t in crossings
        t_end = t - nudge
        if t_end > t_start
            push!(segments, (p1 + t_start * (p2 - p1), p1 + t_end * (p2 - p1)))
        end
        t_start = max(t_start, t + nudge)
    end
    t_start < one(T) && push!(segments, (p1 + t_start * (p2 - p1), p2))
    return segments
end

# Per-axis cell-index range of the cells that the triangle's
# bounding box overlaps. `grid_lines[d]` is sorted, so
# `searchsortedlast` resolves the first and last cell indices in
# `O(log n)` per axis.
function _overlapping_cell_indices(tri::NTuple{3,SVector{3,T}}, grid_lines::NTuple{3},
                                   tol::T) where {T}
    return ntuple(3) do d
        coords = (tri[1][d], tri[2][d], tri[3][d])
        lo = min(coords...) - tol
        hi = max(coords...) + tol
        axes = grid_lines[d]
        i_lo = max(1, searchsortedlast(axes, lo))
        i_hi = min(length(axes) - 1, searchsortedlast(axes, hi))
        return i_lo:i_hi
    end
end

# Sutherland-Hodgman clip of a convex polygon (vertex list) against
# one half-space `n · x ≤ c`. The output polygon is the input
# polygon intersected with the half-space. `nudge` lets a vertex
# exactly on the plane count as inside, avoiding zero-measure
# slivers in the clipped result.
function _clip_polygon_halfspace(polygon::Vector{SVector{D,T}}, axis::Int, sign::Int, plane::T,
                                 nudge::T) where {D,T}
    isempty(polygon) && return polygon
    out = SVector{D,T}[]
    n = length(polygon)
    inside(p) = sign > 0 ? p[axis] <= plane + nudge : p[axis] >= plane - nudge
    for i in 1:n
        a = polygon[i]
        b = polygon[mod1(i + 1, n)]
        a_in = inside(a)
        b_in = inside(b)
        if a_in
            push!(out, a)
            if !b_in
                # a inside, b outside — push intersection.
                t = (plane - a[axis]) / (b[axis] - a[axis])
                push!(out, a + t * (b - a))
            end
        elseif b_in
            # a outside, b inside — push intersection then b on next iter.
            t = (plane - a[axis]) / (b[axis] - a[axis])
            push!(out, a + t * (b - a))
        end
    end
    return out
end

# Clip a convex polygon against an axis-aligned 3D box (cell). The
# box is `lower[d] ≤ x[d] ≤ upper[d]` for each axis; clipped against
# six half-spaces in turn.
function _clip_polygon_to_box(polygon::Vector{SVector{3,T}}, lower::SVector{3,T},
                              upper::SVector{3,T}, nudge::T) where {T}
    poly = polygon
    for d in 1:3
        poly = _clip_polygon_halfspace(poly, d, -1, lower[d], nudge)
        poly = _clip_polygon_halfspace(poly, d, +1, upper[d], nudge)
    end
    return poly
end

# Fan-triangulate a convex polygon from its first vertex. Triangles
# with area below `min_area` are dropped to keep the output free of
# numerical slivers from the clipping step.
function _fan_triangulate(polygon::Vector{SVector{3,T}}, min_area::T) where {T}
    triangles = NTuple{3,SVector{3,T}}[]
    length(polygon) < 3 && return triangles
    v1 = polygon[1]
    for i in 2:(length(polygon)-1)
        v2 = polygon[i]
        v3 = polygon[i + 1]
        area = norm(cross(v2 - v1, v3 - v1)) / 2
        area > min_area && push!(triangles, (v1, v2, v3))
    end
    return triangles
end

"""
    _subdivide_triangle(v1, v2, v3, grid_lines, nudge)

Split a 3D triangle against the level-grid axis-aligned planes. For
each cell the triangle's bounding box overlaps, the triangle is
clipped against that cell's six half-spaces via Sutherland-Hodgman
and the resulting convex polygon is fan-triangulated. Sub-triangles
inherit the parent triangle's plane (and therefore its normal); the
combined area of all sub-triangles equals the input triangle's area
to floating-point precision (modulo nudge-induced sliver drops).
"""
function _subdivide_triangle(v1::SVector{3,T}, v2::SVector{3,T}, v3::SVector{3,T},
                             grid_lines::NTuple{3}, nudge::T) where {T}
    tri = (v1, v2, v3)
    cell_ranges = _overlapping_cell_indices(tri, grid_lines, nudge)
    min_area = nudge * nudge
    sub_triangles = NTuple{3,SVector{3,T}}[]
    polygon0 = SVector{3,T}[v1, v2, v3]

    for ix in cell_ranges[1], iy in cell_ranges[2], iz in cell_ranges[3]
        lower = SVector(grid_lines[1][ix], grid_lines[2][iy], grid_lines[3][iz])
        upper = SVector(grid_lines[1][ix + 1], grid_lines[2][iy + 1], grid_lines[3][iz + 1])
        clipped = _clip_polygon_to_box(polygon0, lower, upper, nudge)
        length(clipped) < 3 && continue
        append!(sub_triangles, _fan_triangulate(clipped, min_area))
    end
    return sub_triangles
end

"""
    _subdivide_mesh(mesh::BoundaryMesh, V::Space, tolerance) -> BoundaryMesh

Return a new `BoundaryMesh` whose cells respect the strict parent-
uniqueness contract on `V`. Per-cell normals are inherited by every
sub-cell. No-op for `K = 0` meshes (points trivially satisfy the
contract) and for `(D, K)` combinations not covered by the segment /
triangle subdividers — the surface-region builder's midpoint
classification then handles those directly.
"""
_subdivide_mesh(mesh::BoundaryMesh{D,T,0}, ::Space{D,T}, ::GeometryTolerance{T}) where {D,T} = mesh

function _subdivide_mesh(mesh::BoundaryMesh{D,T,1}, grid_lines::NTuple{D,Vector{T}},
                         ::GeometryTolerance{T}) where {D,T}
    nudge = sqrt(eps(T))
    cells = NTuple{2,SVector{D,T}}[]
    normals = mesh.normals === nothing ? nothing : SVector{D,T}[]
    for (i, cell) in pairs(mesh.cells)
        subs = _subdivide_segment(cell[1], cell[2], grid_lines, nudge)
        append!(cells, subs)
        if normals !== nothing
            n = mesh.normals[i]
            for _ in 1:length(subs)
                push!(normals, n)
            end
        end
    end
    return BoundaryMesh{D,T,1}(cells, normals)
end

function _subdivide_mesh(mesh::BoundaryMesh{3,T,2}, grid_lines::NTuple{3,Vector{T}},
                         ::GeometryTolerance{T}) where {T}
    nudge = sqrt(eps(T))
    cells = NTuple{3,SVector{3,T}}[]
    normals = mesh.normals === nothing ? nothing : SVector{3,T}[]
    for (i, cell) in pairs(mesh.cells)
        subs = _subdivide_triangle(cell[1], cell[2], cell[3], grid_lines, nudge)
        append!(cells, subs)
        if normals !== nothing
            n = mesh.normals[i]
            for _ in 1:length(subs)
                push!(normals, n)
            end
        end
    end
    return BoundaryMesh{3,T,2}(cells, normals)
end

# `Space` wrappers: subdivide against one space's own merged grid lines.
_subdivide_mesh(mesh::BoundaryMesh{D,T,1}, V::Space{D,T}, tol::GeometryTolerance{T}) where {D,T} =
    _subdivide_mesh(mesh, _level_grid_lines(V, tol), tol)
_subdivide_mesh(mesh::BoundaryMesh{3,T,2}, V::Space{3,T}, tol::GeometryTolerance{T}) where {T} =
    _subdivide_mesh(mesh, _level_grid_lines(V, tol), tol)

# Fallback: combinations not covered above (e.g. `K = 2` in `D = 2`,
# a codim-0 mesh) pass through unchanged.
function _subdivide_mesh(mesh::BoundaryMesh{D,T,K}, ::Space{D,T},
                         ::GeometryTolerance{T}) where {D,T,K}
    return mesh
end

# ── SurfaceRegion and builder ────────────────────────────────────────────────

"""
    SurfaceRegion{D,T}(parents, points, weights, normals)

One admissible integration region on a (post-subdivision) sub-cell of a
user-supplied [`BoundaryMesh`](@ref). The covering-parent list is
constant within the region; the quadrature rule is precomputed in
physical coordinates with weights already including the cell's
measure-to-reference-area Jacobian.

Fields mirror `FacetRegion` apart from carrying a per-Q-point normal
list rather than a single facet normal — a generalisation that lets
future per-Q-point normal sources (curved meshes, user-supplied
overrides per Q-point) drop in without changing the consumer's hot
loop.
"""
struct SurfaceRegion{D,T<:Real}
    parents::Vector{FacetParent{D,T}}
    points::Vector{SVector{D,T}}
    weights::Vector{T}
    normals::Vector{SVector{D,T}}
end

# Physical quadrature for one sub-cell of a subdivided `BoundaryMesh`: the
# reference-simplex rule mapped into physical space, jacobian-scaled weights, and
# a per-point unit normal (the mesh's own normal at `cell_index` when present,
# else the geometric `_default_normal`). Shared by the single-sided surface-region
# builder and the two-sided interface-region builder — the only per-consumer
# difference is parent classification and which region struct wraps the result.
# `T` is recovered from the simplex measure and the point/normal type `P` from
# the normal, so both accepted `K` (the cell's topological dimension) branches
# stay type-stable without threading `{D,T}` in.
function _simplex_cell_quadrature(cell, cell_index::Integer, mesh_normals, reference_samples,
                                  reference_area, ::Val{K}) where {K}
    measure = _simplex_measure(cell)
    T = typeof(measure)
    jacobian = K == 0 ? one(T) : measure / convert(T, reference_area)
    normal = if mesh_normals === nothing
        _default_normal(cell)
    else
        n = mesh_normals[cell_index]
        n / sqrt(sum(x -> x * x, n))
    end
    P = typeof(normal)
    points = P[]
    weights = T[]
    normals = P[]
    for (eta, w_ref) in reference_samples
        push!(points, _reference_to_physical_simplex(cell, eta))
        push!(weights, w_ref * jacobian)
        push!(normals, normal)
    end
    return points, weights, normals
end

# Build the full list of `SurfaceRegion`s for one `BoundaryMesh`
# against a given space. Subdivides the user mesh against the cartesian
# level grids so each emitted cell lies inside one parent per level,
# then walks the resulting sub-cells to emit one `SurfaceRegion` per
# cell with its physical quadrature, per-Q-point normal, and constant
# parent set.
function _surface_regions_for_mesh(V::Space{D,T}, mesh::BoundaryMesh{D,T,K},
                                   tolerance::GeometryTolerance{T}) where {D,T,K}
    subdivided = _subdivide_mesh(mesh, V, tolerance)
    quadrature_order = _surface_quadrature_order(V)
    reference_samples = _simplex_reference_quadrature(Val(K), quadrature_order, T)
    reference_area = _reference_area(Val(K))

    regions = SurfaceRegion{D,T}[]
    for (cell_index, cell) in pairs(subdivided.cells)
        parents = _surface_cell_parents(V, subdivided, cell, cell_index, tolerance)
        isempty(parents) &&
            throw(ArgumentError("BoundaryMesh cell #$cell_index lies entirely outside the " *
                                "discretization — no covering parent found on any level"))
        points, weights, normals = _simplex_cell_quadrature(cell, cell_index, subdivided.normals,
                                                            reference_samples, reference_area, Val(K))
        push!(regions, SurfaceRegion{D,T}(parents, points, weights, normals))
    end

    return regions
end

# Choose the per-axis Gauss-Legendre point count for surface
# integration. We use the per-axis maximum of
# `recommended_quadrature_order(level.basis, level.order)` across
# every level of `V` — at least as accurate as the volume assembly
# would be on the same polynomial order. (A future per-cell variant
# would pick the maximum across only the covering parents of the
# specific cell, matching the facet path; not worth the bookkeeping
# in the MVP.)
function _surface_quadrature_order(V::Space{D,T}) where {D,T}
    base = ntuple(_ -> 1, D)
    for level in V.levels
        per_axis = recommended_quadrature_order(level.basis, level.order)
        base = ntuple(d -> max(base[d], per_axis[d]), D)
    end
    return base
end

# Look up the parent cell per level for one `BoundaryMesh` cell. The
# cell's midpoint determines the parent on every level whose mesh
# contains it; vertices are not re-verified — `_subdivide_mesh` is
# responsible for ensuring each cell lies inside one parent per level,
# and clipping-produced vertices may legitimately sit on grid lines
# (where `locate_cell` would snap them into the adjacent cell).
# Returns the list of covering parents.
function _surface_cell_parents(V::Space{D,T}, mesh::BoundaryMesh{D,T,K},
                               cell::NTuple{KP,SVector{D,T}}, cell_index::Int,
                               tolerance::GeometryTolerance{T}) where {D,T,K,KP}
    midpoint = _cell_midpoint(cell)
    parents = FacetParent{D,T}[]
    for level in V.levels
        midpoint_cell = locate_cell(level.mesh, midpoint; tol=tolerance)
        midpoint_cell === nothing && continue
        is_active(level.mask, midpoint_cell) ||
            throw(ArgumentError("BoundaryMesh cell #$cell_index midpoint $midpoint falls inside " *
                                "an inactive cell on level $(level.id)"))
        push!(parents,
              FacetParent{D,T}(level.id, midpoint_cell, cell_box(level.mesh, midpoint_cell)))
    end
    return parents
end

# Active covering parents of a physical point across every level of `V`:
# one `FacetParent` per level whose mesh contains the point in an *active*
# cell. Unlike `_surface_cell_parents` this never throws — a point outside
# `V`, or inside a level's inactive (fictitious / masked) cell, simply
# contributes no parent from that level. The two-sided interface builder
# uses it per side and skips a sub-cell whose midpoint has no active cover
# on one of the two subdomains (the interface trace there lies outside that
# subdomain's active region).
function _active_cover_parents(V::Space{D,T}, point::SVector{D,T},
                               tol::GeometryTolerance{T}) where {D,T}
    parents = FacetParent{D,T}[]
    for level in V.levels
        cell = locate_cell(level.mesh, point; tol)
        cell === nothing && continue
        is_active(level.mask, cell) || continue
        push!(parents, FacetParent{D,T}(level.id, cell, cell_box(level.mesh, cell)))
    end
    return parents
end

# Midpoint of a simplex cell (average of vertices).
function _cell_midpoint(cell::NTuple{KP,SVector{D,T}}) where {D,T,KP}
    acc = MVector{D,T}(ntuple(_ -> zero(T), D))
    for v in cell
        acc .+= v
    end
    return SVector{D,T}(acc ./ KP)
end
