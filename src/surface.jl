# User-supplied immersed surface meshes and the simplex-region machinery that integrates against them. The user owns
# the geometry: they hand us a list of simplices (points / segments / triangles) and we split each one at the cartesian
# level grids, locate the covering parent cells of every piece on every level, and map a unit-simplex rule onto it,
# emitting one `SimplexRegion` per piece. The strict parent-uniqueness contract (every region's parent set is constant
# within the cell) is enforced by that automatic subdivision, so users never need to align their mesh with the
# discretization.
#
# What this file owns:
#
#   * `BoundaryMesh{D,T,K}` — the user-facing typed payload.
#   * `segment_mesh`, `polyline_mesh`, `triangle_mesh`, `points_mesh` — helper constructors that accept the common input
#     shapes (vertex lists, face index tables, point clouds).
#   * Simplex geometry, one affine map `x = v₁ + E·λ` for every simplex dimension `K`: the measure, the default normal
#     and the unit-simplex quadrature rules.
#   * Automatic subdivision of user cells against every level's grid (segments via per-axis parameter crossings,
#     triangles via single-plane Sutherland–Hodgman splits + fan triangulation).
#   * `SimplexRegion{D,T}` — one per sub-cell: the region record of `intersections.jl` in the sub-cell's bounding-box
#     frame, carrying the mapped rule, the sub-cell's normal and the constant covering-parent list.
#   * `_simplex_regions` — the one builder that ties subdivision + quadrature + parent classification together, for an
#     immersed surface (`_surface_regions_for_mesh`, one-sided) and for the two-sided regions of a coupling interface
#     (`_interface_regions` in `coupling.jl`) alike.
#
# What lives elsewhere:
#
#   * `block(...; on::BoundaryMesh)` / `loadform(...; on::BoundaryMesh)` — already supported by the unified `on=` kwarg
#     in `problems.jl`.
#   * the region accessors of the assembly kernel (`_integrate!`) and the pass enumeration that routes
#     `BoundaryMesh`-tagged forms (`_passes`) — in `src/assembly.jl`.
#   * `boundary_integral(integrand, model; on::BoundaryMesh)` — in `src/postprocessing.jl`.

# ── BoundaryMesh ──────────────────────────────────────────────────────────────

"""
    BoundaryMesh{D,T,K}(cells, normals)

User-supplied immersed surface mesh: a finite list of `(K+1)`-vertex simplices embedded in physical space `ℝᴰ`. `K` is
the simplex dimension, `0 ≤ K ≤ min(D, 2)`:

  - `K = 0` — point cloud (codim-`D` evaluation). Useful for pinning fields at specific physical coordinates, or as the
    point interface of a 1D coupling.
  - `K = 1` — line-segment mesh: codim-1 in 2D (arcs, polylines), a curve in 3D, a codim-0 interval in 1D.
  - `K = 2` — triangle mesh: codim-1 in 3D (surfaces of immersed bodies), a codim-0 patch in 2D.

Every one of these is subdivided and integrated alike (see the user contract below). A codim-0 mesh integrates over the
area (2D) or length (1D) it covers, and like every simplex without a geometric normal it needs `normals`.

Fields:

  - `cells::Vector{NTuple{K+1,SVector{D,T}}}` — vertex tuples of every cell, in physical coordinates. Vertex ordering
    matters for the derived normal (K=1: counter-clockwise / right-hand rule; K=2: right-hand rule across the triple
    cross product).
  - `normals::Union{Nothing,Vector{SVector{D,T}}}` — optional per-cell outward unit normal override. `nothing` (default
    for K≥1) means "derive the normal from cell geometry"; an explicit list lets the user impose orientation conventions
    or normals at K=0 points where no geometric normal exists.

Construct via [`segment_mesh`](@ref), [`polyline_mesh`](@ref), [`triangle_mesh`](@ref), or [`points_mesh`](@ref). Pass
directly to [`block`](@ref), [`loadform`](@ref), or [`boundary_integral`](@ref) via `on = mesh`.

User contract: every cell is subdivided against every level's grid and each resulting sub-cell carries one
[`SimplexRegion`](@ref), so a mesh need not be aligned with the discretization. The subdivision partitions each cell
exactly — the sub-cell measures sum to the cell's up to round-off — and decides nothing by an absolute length, so it is
the same at every geometric scale. Each sub-cell must, however, land somewhere the space can integrate: its midpoint
must lie in at least one level's mesh, and the cell it lands in must be active. A sub-cell outside every level raises —
so does any cell reaching past the outermost grid lines, whose outside part becomes such a sub-cell — and one whose
midpoint falls in a cell deactivated by a [`LevelMask`](@ref) raises as well; both messages carry the sub-cell's index
in the subdivided mesh, which is not the index of the input cell it came from. Sub-cells inherit the parent cell's
normal (geometric default or user override).
"""
struct BoundaryMesh{D,T<:Real,K,C}
    cells::Vector{C}
    normals::Union{Nothing,Vector{SVector{D,T}}}

    function BoundaryMesh{D,T,K}(cells::Vector{NTuple{KP,SVector{D,T}}},
                                 normals::Union{Nothing,Vector{SVector{D,T}}}) where {D,T<:Real,K,KP}
        KP == K + 1 ||
            throw(ArgumentError("BoundaryMesh{$D,$T,$K} expects $(K+1)-vertex cells, " * "got $KP-vertex tuples"))
        0 <= K <= min(D, 2) ||
            throw(ArgumentError("BoundaryMesh simplex dimension K=$K must satisfy 0 ≤ K ≤ min(D, 2) = $(min(D, 2)): " *
                                "points, segments and triangles"))
        D <= 3 || throw(ArgumentError("BoundaryMesh integration is supported only for D ≤ 3; got D=$D"))
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

Build a segment mesh from an explicit list of `(vertex_a, vertex_b)` pairs. Each pair contributes one `K=1` cell. Use
this when the user already holds the segments as a pair-of-points list (e.g. from a contour extraction). For a sequence
of consecutive points, prefer [`polyline_mesh`](@ref).
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

Build a segment mesh from consecutive vertices: each adjacent `(v_i, v_{i+1})` pair becomes one segment. With
`closed = true` the final segment wraps `(v_end, v_1)` so the polyline closes into a loop (handy for an arc traversed
once around `∂Ω`).

`normals`, if supplied, is one normal per resulting segment (`length(vertices) - 1` open, `length(vertices)` closed).
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

Build a triangle mesh from a vertex list and a face-index table (`faces[i] = (a, b, c)` references one-based indices
into `vertices`). Vertex ordering follows the right-hand rule for the geometric normal (unless `normals` is supplied to
override).
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

Build a `K = 0` (point-cell) mesh. Each entry of `points` becomes one cell with a single vertex; `normals` is required
because a point carries no geometric direction. The integration weight per cell is 1 — a `boundary_integral` over a
`points_mesh` therefore returns a sum of `integrand(q)` values at the requested points, with `q.normal` exposing the
user-supplied direction.
"""
function points_mesh(points::AbstractVector{SVector{D,T}}; normals::AbstractVector{SVector{D,T}}) where {D,T}
    cells = NTuple{1,SVector{D,T}}[]
    for p in points
        push!(cells, (p,))
    end
    return BoundaryMesh{D,T,0}(cells, collect(normals))
end

# ── Simplex geometry ──────────────────────────────────────────────────────────
#
# A K-simplex with vertices v₁ … v_{K+1} in ℝᴰ is the image of the unit simplex Δᴷ = {λ ∈ ℝᴷ : λ ≥ 0, Σ λ ≤ 1} under
# one affine map,
#
#     x = v₁ + E·λ,    E = [v₂ − v₁  ⋯  v_{K+1} − v₁]   (D × K, the edge matrix),
#
# so a single set of formulas serves points, segments and triangles in every `D`. The map's Jacobian is the K-volume of
# the parallelotope the edges span, the norm of the wedge `e₁ ∧ ⋯ ∧ e_K`: 1 for a point, `‖e₁‖` for a segment, and for a
# triangle `√Σ_{i<j} (e₁ᵢ e₂ⱼ − e₁ⱼ e₂ᵢ)²`, the root of a sum of squared 2 × 2 minors — `|det E|` in 2D, the norm of the
# cross product in 3D. It never forms the Gram determinant `‖e₁‖²‖e₂‖² − (e₁·e₂)²`, whose cancellation would cost a
# thin triangle its digits. `Δᴷ` has measure `1/K!`, so the simplex's measure is the Jacobian over `K!`, and a
# unit-simplex rule (`_simplex_rule`) becomes the cell's rule by scaling its weights with the Jacobian.

# The edge matrix `E` of a simplex; `D × 0` for a point, whose map is the constant `v₁`.
function _edge_matrix(cell::NTuple{KP,SVector{D,T}}) where {KP,D,T}
    return SMatrix{D,KP - 1,T}(ntuple(i -> cell[(i-1)÷D+2][(i-1)%D+1] - cell[1][(i-1)%D+1], Val(D * (KP - 1))))
end

# The Jacobian `‖e₁ ∧ ⋯ ∧ e_K‖` of a simplex's map from `Δᴷ` (see above). On a 2D triangle the single minor's
# `√(x²)` is `|x|` exactly.
function _simplex_jacobian(cell::NTuple{KP,SVector{D,T}}) where {KP,D,T}
    E = _edge_matrix(cell)
    KP == 1 && return one(T)
    KP == 2 && return norm(E[:, 1])
    squares = zero(T)
    for j in 2:D, i in 1:(j-1)
        squares += abs2(E[i, 1] * E[j, 2] - E[j, 1] * E[i, 2])
    end
    return sqrt(squares)
end

# The K-dimensional measure of a simplex: 1 for a point, the length of a segment, the area of a triangle. Read by the
# MeshIO extension under this name.
_simplex_measure(cell::NTuple{KP}) where {KP} = _simplex_jacobian(cell) / factorial(KP - 1)

# The geometric unit normal of a codim-1 simplex, from its vertex order: a 2D segment's direction `v₂ − v₁` turned 90°
# clockwise (the polyline is the oriented boundary of the region to its left), a 3D triangle's right-hand cross product.
# No other simplex has one to derive — a point carries no direction, a segment in 3D has a circle of normals, a
# codim-0 simplex fills its embedding — so the user supplies its normals.
function _default_normal(cell::NTuple{2,SVector{2,T}}) where {T}
    e = cell[2] - cell[1]
    return normalize(SVector{2,T}(e[2], -e[1]))
end

_default_normal(cell::NTuple{3,SVector{3,T}}) where {T} = normalize(cross(cell[2] - cell[1], cell[3] - cell[1]))

function _default_normal(::NTuple{KP,SVector{D}}) where {KP,D}
    throw(ArgumentError("a $(KP - 1)-simplex in $(D)D has no geometric normal — supply normals explicitly"))
end

# Midpoint (vertex average) of a simplex: the point a sub-cell is located by. Read by the MeshIO extension under this
# name.
_cell_midpoint(cell::NTuple{KP,SVector{D,T}}) where {KP,D,T} = sum(cell) / KP

# A sub-cell's bounding box: the frame of its region. An axis along which the simplex is flat is degenerate in it.
_simplex_frame(cell) = (reduce((a, b) -> min.(a, b), cell), reduce((a, b) -> max.(a, b), cell))

# An `m`-points-per-axis rule on the unit simplex `Δᴷ` as `(λ points, weights)`, from the tensor Gauss rule on
# `[−1, 1]ᴷ` (`_tensor_gauss_rule`) moved to `[0, 1]ᴷ`, `s = (η + 1) / 2` with weight `w / 2ᴷ`:
#
#   - K = 0: the single point, weight 1;
#   - K = 1: `s` itself, the m-point Gauss rule on [0, 1], exact to degree 2m − 1;
#   - K = 2: the Duffy collapse `λ = (s₂, s₁(1 − s₂))` with weight `w (1 − s₂) / 4`, exact to total degree 2m − 2: a
#     polynomial of total degree q on the triangle pulls back to degree ≤ q in s₁ and ≤ q + 1 in s₂ (the collapse's
#     Jacobian `1 − s₂`), and m points per axis integrate degree 2m − 1.
#
# Built once per point count by each builder call (`_simplex_regions`), never on the hot path.
function _simplex_rule(::Val{K}, m::Int, ::Type{T}) where {K,T}
    K == 0 && return [zero(SVector{0,T})], [one(T)]
    rule = _tensor_gauss_rule(ntuple(_ -> m, Val(K)), T)
    s = [(η .+ 1) ./ 2 for η in rule.points]
    K == 1 && return s, rule.weights ./ 2
    return [SVector(t[2], t[1] * (1 - t[2])) for t in s], [w * (1 - t[2]) / 4 for (t, w) in zip(s, rule.weights)]
end

# ── Subdivision against level grids ──────────────────────────────────────────
#
# A region's parent set must be constant over it, so every simplex is split at the grid lines of every level it is
# integrated against before any region is built; a user mesh need not be aligned with the discretization. The lines are
# per axis, merged across the levels (`_merged_axis_coordinates`), so a piece that respects them lies in one cell of
# every level at once. Per simplex dimension:
#
#   * `K = 0` (points)      — nothing to split.
#   * `K = 1` (segments)    — per-axis parameter-value crossings, sorted, then cut at each one exactly. The sub-segments
#                             partition the input: an endpoint may land on a grid line, and the sub-segment's *interior*
#                             — which is what the midpoint classification sees — lies in one cell. Works in any `D`.
#   * `K = 2` (triangles)   — plane splitting: one single-plane Sutherland–Hodgman pass per grid line (plane, in 3D)
#                             the triangle crosses, then fan triangulation of each convex piece. The pieces partition
#                             the input, and the parts beyond the outermost lines are kept, so a triangle leaving the
#                             grid fails like a segment does. Works in 2D and 3D.

# The sub-cells of `mesh` split at the per-axis `lines` (`_split_cell!`), as `(; cells, normals)` with one unit normal
# per sub-cell: its input cell's, from the mesh's own override (normalised) or else the geometric `_default_normal` of
# the input cell. Taking the normal from the input cell rather than from each piece keeps a sliver from carrying a
# normal its few surviving digits define, and computes it once per input cell.
function _subdivide_mesh(mesh::BoundaryMesh{D,T}, lines::NTuple{D,Vector{T}}) where {D,T}
    cells = eltype(mesh.cells)[]
    normals = SVector{D,T}[]
    for (i, cell) in pairs(mesh.cells)
        count = length(cells)
        _split_cell!(cells, cell, lines)
        normal = mesh.normals === nothing ? _default_normal(cell) : mesh.normals[i] / sqrt(sum(abs2, mesh.normals[i]))
        for _ in (count+1):length(cells)
            push!(normals, normal)
        end
    end
    return (; cells, normals)
end

# One input cell's pieces, each inside a single cell of every level the `lines` come from, appended to `cells` (see
# above). Both subdividers are scale-free: `_subdivide_segment` reads a dimensionless crossing parameter, and
# `_subdivide_triangle` an area relative to the input's.
_split_cell!(cells, cell::NTuple{1}, _) = push!(cells, cell)
function _split_cell!(cells, (a, b)::NTuple{2,SVector{D,T}}, lines) where {D,T}
    append!(cells, _subdivide_segment(a, b, lines, sqrt(eps(T))))
end
_split_cell!(cells, (a, b, c)::NTuple{3,SVector}, lines) = append!(cells, _subdivide_triangle(a, b, c, lines))

"""
    _subdivide_segment(p1, p2, grid_lines, nudge)

Split the segment `(p1, p2)` at every grid-line crossing across every axis. Returns a vector of `(a, b)` `SVector` pairs
that **partition** the input: consecutive pairs share their endpoint exactly, the first starts at `p1` and the last ends
at `p2`, so the sub-segment lengths sum back to `‖p2 − p1‖` up to round-off. A segment with no crossings passes through
as `[(p1, p2)]`.

`nudge` is a tolerance on the *dimensionless* crossing parameter `t ∈ [0, 1]` — never on the split geometry. It decides
two things and nothing else: whether a root of `p1 + t (p2 − p1)` against a grid line counts as a crossing at all (the
root search admits `t` marginally outside `[0, 1]`, since a segment endpoint sitting on a grid line lands there), and
whether two roots name the same point (a segment through a grid *intersection* produces one root per axis, agreeing to a
few ulp). Because `t` is a ratio of coordinate differences, the tolerance is scale-free: rescaling the geometry leaves
the subdivision unchanged.
"""
function _subdivide_segment(p1::SVector{D,T}, p2::SVector{D,T}, grid_lines::NTuple{D}, nudge::T) where {D,T}
    crossings = T[]
    for d in 1:D
        denom = p2[d] - p1[d]
        iszero(denom) && continue
        for c in grid_lines[d]
            t = (c - p1[d]) / denom
            # Admit roots marginally outside [0, 1] — an endpoint on a grid line is one — then clamp, so a split point
            # is never placed off the segment.
            -nudge < t < 1 + nudge && push!(crossings, clamp(t, zero(T), one(T)))
        end
    end
    isempty(crossings) && return NTuple{2,SVector{D,T}}[(p1, p2)]
    sort!(crossings)

    # Walk the sorted roots and cut at each one exactly. `a` carries the previous cut point by value, so neighbouring
    # sub-segments share it bit-for-bit and no arc length is lost between them. Roots within `nudge` of the running cut
    # point are the same point (`continue`); roots within `nudge` of `t = 1` are the far endpoint, and since the list is
    # sorted every later root is too (`break`) — the closing `push!` then terminates the chain on the exact `p2`.
    #
    # A cut point may therefore land exactly on a grid line, where `locate_cell` cannot say which side owns it. That is
    # deliberate and matches the triangle path: parent-uniqueness is a statement about a sub-segment's *interior*, which
    # lies strictly inside one cell precisely because every crossing became a cut. The region builder classifies by the
    # midpoint (see `_simplex_regions`), so the ambiguity at the shared endpoint is never consulted.
    segments = NTuple{2,SVector{D,T}}[]
    a = p1
    t_prev = zero(T)
    for t in crossings
        t - t_prev > nudge || continue
        one(T) - t > nudge || break
        b = p1 + t * (p2 - p1)
        push!(segments, (a, b))
        a = b
        t_prev = t
    end
    push!(segments, (a, p2))
    return segments
end

"""
    _subdivide_triangle(v1, v2, v3, grid_lines) -> Vector{NTuple{3,SVector{D,T}}}

Split a triangle in 2D or 3D at the axis-aligned grid lines (planes, in 3D) `grid_lines[d]`, sorted per axis. Axis by
axis, every piece is split (`_split_polygon`) at each line strictly inside its own extent along that axis, so the work
is the number of lines the triangle actually crosses, not the number of grid cells under its bounding box. Each final
piece is a convex polygon inside one grid cell, fan-triangulated from its first vertex.

The pieces **partition** the triangle: a crossing point is shared exactly by the two sides of its split, so the
sub-triangle areas sum to the input's up to round-off. A fan triangle whose area is at most `eps(T)` times the input's
is a round-off sliver — a line passing within an ulp of a vertex — and is dropped. Every test is either exact (a line
strictly inside an extent) or relative to the input (the sliver floor), so the subdivision is the same at every
geometric scale.

Parts beyond the outermost lines are kept like any other piece. They lie in no cell, so the surface builder raises on
them and the interface builder skips them, as for a segment that leaves the grid. Sub-triangles lie in the input's plane
and are given its normal by `_subdivide_mesh`.
"""
function _subdivide_triangle(v1::SVector{D,T}, v2::SVector{D,T}, v3::SVector{D,T}, grid_lines::NTuple{D}) where {D,T}
    pieces = [SVector{D,T}[v1, v2, v3]]
    for d in 1:D
        lines = grid_lines[d]
        split = Vector{SVector{D,T}}[]
        for piece in pieces
            lo, hi = extrema(p -> p[d], piece)
            rest = piece
            for k in (searchsortedlast(lines, lo)+1):length(lines)
                lines[k] < hi || break
                below, rest = _split_polygon(rest, d, lines[k])
                push!(split, below)
            end
            push!(split, rest)
        end
        pieces = split
    end
    floor = eps(T) * _simplex_jacobian((v1, v2, v3))
    triangles = NTuple{3,SVector{D,T}}[]
    for piece in pieces, i in 2:(length(piece)-1)
        triangle = (piece[1], piece[i], piece[i + 1])
        _simplex_jacobian(triangle) > floor && push!(triangles, triangle)
    end
    return triangles
end

# Split a convex polygon at the plane `x[d] = c` through its interior into the parts below and above it: one
# Sutherland–Hodgman pass against a single plane that keeps both sides (I. E. Sutherland, G. W. Hodgman, "Reentrant
# polygon clipping", Commun. ACM 17 (1974) 32–42, doi:10.1145/360767.360802). Walking the edges in order, a vertex goes
# to the side it lies on, and to both when it lies on the plane; an edge that crosses the plane strictly puts its
# crossing point on both sides, between its endpoints. The crossing is interpolated from the edge's lower endpoint
# along `d`, so two triangles sharing an edge compute the same points on it, and then set exactly onto the plane: the
# two sides meet along it with neither a gap nor an overlap, and a piece split at `c` ends exactly at `c`. Both sides
# of a convex polygon are convex, in the input's vertex order, and have at most one vertex more than it.
function _split_polygon(polygon::Vector{SVector{D,T}}, d::Int, c::T) where {D,T}
    below = sizehint!(SVector{D,T}[], length(polygon) + 1)
    above = sizehint!(SVector{D,T}[], length(polygon) + 1)
    a = polygon[end]
    for b in polygon
        if min(a[d], b[d]) < c < max(a[d], b[d])
            p, q = a[d] < b[d] ? (a, b) : (b, a)
            x = Base.setindex(p + (c - p[d]) / (q[d] - p[d]) * (q - p), c, d)
            push!(below, x)
            push!(above, x)
        end
        b[d] <= c && push!(below, b)
        b[d] >= c && push!(above, b)
        a = b
    end
    return below, above
end

# ── SimplexRegion and builder ─────────────────────────────────────────────────

"""
    SimplexRegion{D,T} = Region{D,T,SVector{D,T},Nothing}

One admissible integration region on a (post-subdivision) sub-cell of a user-supplied [`BoundaryMesh`](@ref) — `kind`
`:surface` — or of a coupling [`Interface`](@ref) — `kind` `:interface`. A [`Region`](@ref) whose frame is the
sub-cell's bounding box: the simplex rule is mapped into that box's `[−1, 1]ᴰ` frame, `0` on any axis along which the
sub-cell is flat (a K = 0 point is flat on every axis, so its single point is the vertex itself with weight 1), and the
weights are divided by the frame Jacobian, so the region accessors return the physical points and weights.

What its fields hold on a sub-cell:

  - `points` / `weights` — a unit-simplex rule exact for every product of two of its parents' basis functions: its size
    follows the total degree such a product reaches along the sub-cell, so an oblique segment or triangle takes more
    points than an axis-aligned one (`_simplex_regions`).
  - `parents` — one parent per level whose active cell contains the sub-cell, each with its map from the frame. On an
    interface, `parents[1:split]` lie in `field_a`'s subdomain and the rest in `field_b`'s, so the two fields are
    evaluated in their own cut cells at the shared point; `fields = (field_a, field_b)` holds the coupled fields' global
    indices. A surface region is one-sided: `split = length(parents)`, `fields = (0, 0)`.
  - `normal::SVector{D,T}` — the unit normal of the input cell the sub-cell was split from: the mesh's own normal when
    given, else the geometric default (`_default_normal`), oriented from side `a` to side `b` on an interface. It is
    constant over the sub-cell, which is flat.
  - `sides` is `nothing`: an immersed cell sits in the interior of a mesh cell and has no `(axis, side)` facet identity,
    so consumers see `q.sides === nothing`.
  - `residual` is zero: a sub-cell's geometry is stated by the user and already *is* a piece of `∂Ω` (or of `Γ`), so
    there is no trimming and no fit, unlike a facet region whose part inside `Ω` the package has to find.
"""
const SimplexRegion{D,T} = Region{D,T,SVector{D,T},Nothing}

# Build the full list of surface regions for one `BoundaryMesh` against a given space (`_simplex_regions`, one-sided).
function _surface_regions_for_mesh(V::Space{D,T}, mesh::BoundaryMesh{D,T}, tol::GeometryTolerance{T}) where {D,T}
    _simplex_regions(mesh, (V,), (0, 0), tol)
end

# Every `SimplexRegion` of `mesh` integrated against `spaces`: one space for an immersed surface (`fields = (0, 0)`,
# kind `:surface`), the two subdomain spaces of a coupling interface (`fields = (field_a, field_b)`, their global field
# indices; kind `:interface`). The mesh is split at the merged grid lines of every level of every space
# (`_subdivide_mesh`), so each sub-cell lies inside one cell of every level on both sides — on an interface, one cut
# cell of one subdomain may face several of the other (the non-matching segment-merge of the references). Per sub-cell:
#
#   1. its midpoint's coverage signature per space (`_signature`, the locator of the volume partition), all levels
#      eligible;
#   2. a surface keeps `BoundaryMesh`'s strict contract and raises where a sub-cell misses every level or lands in an
#      inactive cell (`_check_surface_cover`); an interface skips a sub-cell either side does not cover — the trace
#      there lies outside that subdomain's active region;
#   3. the parents decoded from the signatures into one list, side `a` first (`_parents!`), each with its map from the
#      sub-cell's bounding-box frame (`_simplex_frame`);
#   4. the rule size from those parents, below;
#   5. the unit-simplex rule of that size (`_simplex_rule`, built once per size) mapped through `x = v₁ + E·λ` into the
#      frame, its weights scaled by the simplex Jacobian over the frame Jacobian, so the region accessors recover the
#      physical points and weights.
#
# **Rule size.** A product of two of the parents' basis functions (on an interface, one trace from each side) has
# per-axis degree `2(n_d − 1)`, where `n_d` is the parents' per-axis Gauss count (`_axis_max` of `_gauss_counts`: per
# sub-cell, so one high-p cell elsewhere raises no rule here; per side, since each side's level ids are numbered in its
# own space; the larger per axis). Along the simplex `x = v₁ + E·λ` that product is a polynomial in `λ` of TOTAL degree
# `Q = 2 Σ_d (n_d − 1)` over the axes the sub-cell extends along — a flat axis has `x_d` constant and adds nothing — so
# the rule takes `m = 1 + Σ_d (n_d − 1)` points per axis: Gauss on a segment is exact to `2m − 1 > Q`, the Duffy rule
# on a triangle to `2m − 2 = Q`. An axis-aligned segment keeps the `n_d` points of a tensor rule along its own axis; an
# oblique one needs more, because each axis contributes its own degree: a p = 2 segment across both axes of a 2D cell
# takes 5 points, a p = 2 triangle oblique to all three axes 7 × 7. A rule sized by one axis's count alone is exact
# only on the axis-aligned simplices.
function _simplex_regions(mesh::BoundaryMesh{D,T,K}, spaces::Tuple, fields::NTuple{2,Int},
                          tol::GeometryTolerance{T}) where {D,T,K}
    levels = map(V -> V.levels, spaces)
    sub = _subdivide_mesh(mesh, _merged_axis_coordinates(reduce((a, b) -> (a..., b...), levels), Val(D), tol))
    linmaps = map(side -> map(level -> LinearIndices(level.mesh.cells), side), levels)
    eligible = map(side -> map(_ -> _AllCells(), side), levels)
    kind = length(spaces) == 1 ? :surface : :interface
    rules = Dict{Int,Tuple{Vector{SVector{K,T}},Vector{T}}}()
    regions = SimplexRegion{D,T}[]
    for (index, cell) in pairs(sub.cells)
        x = _cell_midpoint(cell)
        sigs = map((side, linmap, cells) -> _signature(side, linmap, cells, x, tol), levels, linmaps, eligible)
        if length(spaces) == 1
            _check_surface_cover(levels[1], sigs[1], x, tol, index)
        elseif any(sig -> all(iszero, sig), sigs)
            continue
        end
        lower, upper = _simplex_frame(cell)
        parents = Parent{D,T}[]
        sides = map(levels, sigs) do side, sig
            start = length(parents) + 1
            _parents!(parents, side, sig, lower, upper)
            return start:length(parents)
        end
        counts = map((V, side) -> _axis_max(_gauss_counts, V, view(parents, side), lower, upper), spaces, sides)
        m = 1 + sum(n -> max(n - 1, 0), reduce((a, b) -> max.(a, b), counts))
        λ, w = get!(() -> _simplex_rule(Val(K), m, T), rules, m)
        E = _edge_matrix(cell)
        scale = _simplex_jacobian(cell) / _frame_jacobian(lower, upper)
        points = [_frame_point(lower, upper, cell[1] + E * l) for l in λ]
        push!(regions,
              Region(lower, upper, parents, kind, points, w .* scale; split=last(sides[1]), fields,
                     normal=sub.normals[index]))
    end
    return regions
end

# `BoundaryMesh`'s contract for a surface sub-cell (the `index`-th of the subdivided mesh): its midpoint `x` lies in at
# least one level's mesh, and in an active cell of every level whose mesh holds it. `sig` is `x`'s coverage signature
# over `levels`, so a zero slot is a level that misses `x` or holds it in an inactive cell, and containment tells the
# two apart.
function _check_surface_cover(levels::Tuple, sig::Tuple, x, tol, index::Int)
    map(levels, sig) do level, slot
        slot == 0 &&
            contains_point(x, level.mesh.domain, tol) &&
            throw(ArgumentError("BoundaryMesh cell #$index midpoint $x falls inside an inactive cell on level " *
                                "$(level.id)"))
    end
    all(iszero, sig) && throw(ArgumentError("BoundaryMesh cell #$index lies entirely outside the discretization — " *
                                            "no covering parent found on any level"))
    return nothing
end
