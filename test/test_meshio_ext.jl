using StaticArrays
using LinearAlgebra
using Random
# Loading FileIO + MeshIO activates `UnfittedMeshIOExt` (GeometryBasics, the
# third trigger, comes transitively as MeshIO's dependency). It is deliberately
# *not* imported here: GeometryBasics also exports `triangle_mesh`, which would
# shadow Unfitted's.
using FileIO
using MeshIO

# ── 3D cube (triangle surface) ────────────────────────────────────────────────

# A cube [lo,hi]³ as an outward-oriented triangle mesh (8 vertices, 12
# triangles). Cube faces are planar, so each cut cell sees a linear signed
# distance and the volume integrates exactly.
function _cube_mesh(lo, hi)
    v = [SVector(lo, lo, lo), SVector(hi, lo, lo), SVector(hi, hi, lo), SVector(lo, hi, lo),
         SVector(lo, lo, hi), SVector(hi, lo, hi), SVector(hi, hi, hi), SVector(lo, hi, hi)]
    quads = [(1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]
    faces = NTuple{3,Int}[]
    for (a, b, c, d) in quads
        push!(faces, (a, b, c))
        push!(faces, (a, c, d))
    end
    ctr = sum(v) / 8
    for (i, f) in enumerate(faces)
        n = cross(v[f[2]] - v[f[1]], v[f[3]] - v[f[1]])
        tc = (v[f[1]] + v[f[2]] + v[f[3]]) / 3
        dot(n, tc - ctr) < 0 && (faces[i] = (f[1], f[3], f[2]))
    end
    return v, faces
end

function _write_ascii_stl(path, v, faces)
    open(path, "w") do s
        println(s, "solid cube")
        for f in faces
            a, b, c = v[f[1]], v[f[2]], v[f[3]]
            n = normalize(cross(b - a, c - a))
            println(s, "facet normal $(n[1]) $(n[2]) $(n[3])")
            println(s, "  outer loop")
            for p in (a, b, c)
                println(s, "    vertex $(p[1]) $(p[2]) $(p[3])")
            end
            println(s, "  endloop")
            println(s, "endfacet")
        end
        println(s, "endsolid cube")
    end
    return path
end

# Analytic membership for a 3D L-shaped prism (reflex edge + vertices), where an
# equal-weight pseudonormal would mis-sign the reflex Voronoi cone.
function _inside_L3(p)
    (-1e-9 <= p[3] <= 1 + 1e-9) && (((-1e-9 <= p[1] <= 2 + 1e-9) && (-1e-9 <= p[2] <= 1 + 1e-9)) ||
                                    ((-1e-9 <= p[1] <= 1 + 1e-9) && (-1e-9 <= p[2] <= 2 + 1e-9)))
end

# L-prism as a watertight triangle mesh, auto-oriented outward via `_inside_L3`.
# Both caps are fanned from the reflex vertex (1,1) — vertex 4 below, 11 above.
# Any other hub spans the three collinear vertices at y = 1 with a single edge
# and leaves a T-junction at (1,1), i.e. a cap edge with no matching partner in
# the wall, which `mesh_levelset` rejects as an unclosed mesh.
function _lprism_mesh()
    poly = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0), (0.0, 1.0)]
    n = length(poly)
    verts = SVector{3,Float64}[]
    for (x, y) in poly
        push!(verts, SVector(x, y, 0.0))
    end
    for (x, y) in poly
        push!(verts, SVector(x, y, 1.0))
    end
    faces = NTuple{3,Int}[(4, 5, 6), (4, 6, 7), (4, 7, 1), (4, 1, 2), (4, 2, 3), (11, 12, 13),
                          (11, 13, 14), (11, 14, 8), (11, 8, 9), (11, 9, 10)]
    for i in 1:n
        j = i % n + 1
        push!(faces, (i, j, j + n))
        push!(faces, (i, j + n, i + n))
    end
    for (k, f) in enumerate(faces)
        a, b, c = verts[f[1]], verts[f[2]], verts[f[3]]
        nrm = normalize(cross(b - a, c - a))
        _inside_L3((a + b + c) / 3 + 1e-4 * nrm) && (faces[k] = (f[1], f[3], f[2]))
    end
    return verts, faces
end

# Point-to-triangle distance, independent of the extension's Voronoi-region
# routine: the in-plane projection when it lands inside the triangle, else the
# nearest point on an edge.
function _segment_distance(a, b, p)
    ab = b - a
    return norm(p - (a + clamp(dot(p - a, ab) / dot(ab, ab), 0, 1) * ab))
end
function _triangle_distance(cell, p)
    a, b, c = cell
    edges = ((a, b), (b, c), (c, a))
    nrm = normalize(cross(b - a, c - a))
    q = p - dot(p - a, nrm) * nrm
    all(dot(cross(v - u, q - u), nrm) >= 0 for (u, v) in edges) && return norm(p - q)
    return minimum(_segment_distance(u, v, p) for (u, v) in edges)
end

# ── 2D polygons (segment loops) ───────────────────────────────────────────────

# CCW square [lo,hi]² (⇒ outward normals via BoundaryMesh's `_default_normal`).
_square_poly(lo, hi) = [SVector(lo, lo), SVector(hi, lo), SVector(hi, hi), SVector(lo, hi)]

# CCW L-polygon with a reflex vertex at (1,1), and its analytic membership.
function _lpoly2d()
    [SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(2.0, 1.0), SVector(1.0, 1.0), SVector(1.0, 2.0),
     SVector(0.0, 2.0)]
end
function _inside_L2(p)
    ((-1e-9 <= p[1] <= 2 + 1e-9) && (-1e-9 <= p[2] <= 1 + 1e-9)) ||
        ((-1e-9 <= p[1] <= 1 + 1e-9) && (-1e-9 <= p[2] <= 2 + 1e-9))
end

# Volume / area of {leaf ≤ 0} ∩ box via the implicit kernel.
function _mesh_volume(geom, region; gp=4, ms=3)
    leaves = Any[l.f for l in Unfitted._leaves(geom)]
    membership = x -> Unfitted._inside(geom, x)
    pts, wts = Unfitted.implicit_volume_quadrature(leaves, membership, region; gauss_points=gp,
                                                   max_subdiv=ms)
    return sum(wts; init=0.0)
end

# ── 3D tests ──────────────────────────────────────────────────────────────────

@testset "MeshIO ext — mesh_levelset signed distance and sign (3D)" begin
    v, f = _cube_mesh(0.2, 0.8)
    ls = mesh_levelset(triangle_mesh(v, f))
    @test ls isa Unfitted.LevelSet
    sdf = Unfitted._leaves(ls)[1].f
    @test sdf(SVector(0.5, 0.5, 0.5)) ≈ -0.3 atol = 1e-12
    @test sdf(SVector(0.5, 0.5, 0.7)) ≈ -0.1 atol = 1e-12
    @test sdf(SVector(0.5, 0.5, 0.95)) ≈ 0.15 atol = 1e-12
end

@testset "MeshIO ext — AD gradient is the oriented unit normal (3D)" begin
    v, f = _cube_mesh(0.2, 0.8)
    sdf = Unfitted._leaves(mesh_levelset(triangle_mesh(v, f)))[1].f
    g = Unfitted.ForwardDiff.gradient(sdf, SVector(0.5, 0.5, 0.7))
    @test g ≈ SVector(0.0, 0.0, 1.0) atol = 1e-9
end

@testset "MeshIO ext — classify and exact cube volume (3D)" begin
    v, f = _cube_mesh(0.2, 0.8)
    dom = physical_domain(mesh_levelset(triangle_mesh(v, f)); subcell_length_scale=0.05)
    @test classify_cell(dom, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))) === :cut
    @test classify_cell(dom, box((0.4, 0.4, 0.4), (0.6, 0.6, 0.6))) === :full
    @test classify_cell(dom, box((0.85, 0.85, 0.85), (0.95, 0.95, 0.95))) === :fictitious
    @test _mesh_volume(dom.geometry, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))) ≈ 0.6^3 atol = 1e-12
end

@testset "MeshIO ext — stl_levelset round-trip" begin
    v, f = _cube_mesh(0.2, 0.8)
    path = _write_ascii_stl(tempname() * ".stl", v, f)
    dom = physical_domain(stl_levelset(path); subcell_length_scale=0.05)
    @test classify_cell(dom, box((0.4, 0.4, 0.4), (0.6, 0.6, 0.6))) === :full
    @test _mesh_volume(dom.geometry, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))) ≈ 0.6^3 atol = 1e-6
    rm(path; force=true)
end

@testset "MeshIO ext — angle-weighted pseudonormal signs a non-convex solid (3D)" begin
    verts, faces = _lprism_mesh()
    sdf = Unfitted._leaves(mesh_levelset(triangle_mesh(verts, faces)))[1].f
    rng = MersenneTwister(20260625)
    tested = 0
    bad = 0
    for _ in 1:20000
        p = SVector(-0.3 + 2.6 * rand(rng), -0.3 + 2.6 * rand(rng), -0.3 + 1.6 * rand(rng))
        d = sdf(p)
        abs(d) <= 1e-6 && continue
        tested += 1
        ((d < 0) == _inside_L3(p)) || (bad += 1)
    end
    @test tested > 5000
    @test bad == 0
end

@testset "MeshIO ext — winding sign mode" begin
    v, f = _cube_mesh(0.2, 0.8)
    sp = Unfitted._leaves(mesh_levelset(triangle_mesh(v, f); orientation=:pseudonormal))[1].f
    sw = Unfitted._leaves(mesh_levelset(triangle_mesh(v, f); orientation=:winding))[1].f
    rng = MersenneTwister(7)
    for _ in 1:2000
        p = SVector(rand(rng), rand(rng), rand(rng))
        abs(sp(p)) <= 1e-6 && continue
        @test sign(sp(p)) == sign(sw(p))
    end
    lv, lf = _lprism_mesh()
    lw = Unfitted._leaves(mesh_levelset(triangle_mesh(lv, lf); orientation=:winding))[1].f
    @test (lw(SVector(1.5, 0.5, 0.5)) < 0) == _inside_L3(SVector(1.5, 0.5, 0.5))
    @test (lw(SVector(1.5, 1.5, 0.5)) < 0) == _inside_L3(SVector(1.5, 1.5, 0.5))
    @test_throws ArgumentError mesh_levelset(triangle_mesh(v, f); orientation=:bogus)
end

@testset "MeshIO ext — degenerate triangles are dropped" begin
    v, f = _cube_mesh(0.2, 0.8)
    push!(v, SVector(0.5, 0.5, 0.5))
    nv = length(v)
    # A zero-area triangle (repeated vertex) and a collinear sliver. v[1] and
    # v[2] are (0.2,0.2,0.2) and (0.8,0.2,0.2); the midpoint is collinear.
    push!(v, SVector(0.5, 0.2, 0.2))
    bad_faces = vcat(f, [(nv, nv, nv), (1, 2, nv + 1)])
    sdf = Unfitted._leaves(mesh_levelset(triangle_mesh(v, bad_faces)))[1].f
    @test isfinite(sdf(SVector(0.5, 0.5, 0.5)))
    @test sdf(SVector(0.5, 0.5, 0.5)) ≈ -0.3 atol = 1e-12
end

@testset "MeshIO ext — an unclosed or inside-out mesh is rejected, not answered" begin
    # Each of these used to return a plausible distance with the sign inverted
    # over part of space. The pseudonormal is exact only on a closed, outward
    # mesh, so anything else must fail at construction.
    v, f = _cube_mesh(0.2, 0.8)
    @test_throws ArgumentError mesh_levelset(triangle_mesh(v, f[1:(end-2)]))     # a hole
    flipped = [i == 1 ? (f[1][1], f[1][3], f[1][2]) : f[i] for i in eachindex(f)]
    @test_throws ArgumentError mesh_levelset(triangle_mesh(v, flipped))          # one facet
    @test_throws ArgumentError mesh_levelset(triangle_mesh(v, [(a, c, b) for (a, b, c) in f]))
    @test_throws ArgumentError mesh_levelset(polyline_mesh(_square_poly(0.2, 0.8)))   # open 2D
    @test_throws ArgumentError mesh_levelset(polyline_mesh(reverse(_square_poly(0.2, 0.8));
                                                           closed=true))              # clockwise
    # A zero normal normalises to NaN, under which every sign test is false and
    # the whole space reads as inside.
    @test_throws ArgumentError mesh_levelset(triangle_mesh(v, f;
                                                           normals=[SVector(0.0, 0.0, 0.0)
                                                                    for _ in f]))
    empty_stl = tempname() * ".stl"
    write(empty_stl, "solid s\nendsolid s\n")
    @test_throws ArgumentError stl_levelset(empty_stl)
    rm(empty_stl; force=true)
end

@testset "MeshIO ext — vertex welding absorbs an STL's Float32 corners" begin
    # The closedness check pairs cell faces through welded vertices, and an STL
    # stores every facet's corners separately in Float32: one corner reached
    # through two facets can differ by an ulp. Such a mesh is sound, and welding
    # tighter than that ulp would report it as full of holes.
    v, f = _cube_mesh(200.0, 800.0)
    rng = MersenneTwister(23)
    verts = SVector{3,Float64}[]
    faces = NTuple{3,Int}[]
    for tri in f
        for k in tri
            push!(verts, map(x -> Float64(nextfloat(Float32(x), rand(rng, -1:1))), v[k]))
        end
        push!(faces, (length(verts) - 2, length(verts) - 1, length(verts)))
    end
    sdf = Unfitted._leaves(mesh_levelset(triangle_mesh(verts, faces)))[1].f
    @test sdf(SVector(500.0, 500.0, 500.0)) ≈ -300.0 rtol = 1e-6
end

@testset "MeshIO ext — winding mode takes the meshes pseudonormal rejects" begin
    # The documented recourse for an imperfect mesh: an open box still signs
    # correctly away from the hole, and its Lipschitz declaration drops to Inf
    # because the sign jumps across the gap at a positive distance from any cell.
    v, f = _cube_mesh(0.2, 0.8)
    ls = mesh_levelset(triangle_mesh(v, f[1:(end-2)]); orientation=:winding)
    @test Unfitted._leaves(ls)[1].lipschitz == Inf
    @test Unfitted._leaves(mesh_levelset(triangle_mesh(v, f)))[1].lipschitz == 1.0
    @test Unfitted._leaves(mesh_levelset(triangle_mesh(v, f); lipschitz=2.0))[1].lipschitz == 2.0
    sdf = Unfitted._leaves(ls)[1].f
    rng = MersenneTwister(31)
    for _ in 1:2000
        p = SVector(rand(rng), rand(rng), rand(rng))
        inside = all(0.25 .<= p .<= 0.75)
        (all(0.15 .<= p .<= 0.85) && !inside) && continue   # skip the band around the hole
        @test (sdf(p) < 0) == inside
    end
end

@testset "MeshIO ext — nearest-cell search is exact and the distance is 1-Lipschitz" begin
    # The KD-tree is keyed on cell midpoints, and the nearest midpoint is not
    # the nearest cell; the candidate radius `nearest + maxcr` is what makes the
    # search exact. Checked against an independent brute force over every cell —
    # and with the search exact and the mesh closed, the leaf is a true signed
    # distance, which is the `lipschitz = 1.0` the extension declares.
    verts, faces = _lprism_mesh()
    mesh = triangle_mesh(verts, faces)
    cells = [map(x -> SVector{3,Float64}(x), c) for c in mesh.cells]
    sdf = Unfitted._leaves(mesh_levelset(mesh))[1].f
    rng = MersenneTwister(17)
    for _ in 1:400
        p = SVector(-0.4 + 2.8 * rand(rng), -0.4 + 2.8 * rand(rng), -0.4 + 1.8 * rand(rng))
        @test abs(sdf(p)) ≈ minimum(_triangle_distance(c, p) for c in cells) atol = 1e-12
    end
    for _ in 1:4000
        p = SVector(-0.4 + 2.8 * rand(rng), -0.4 + 2.8 * rand(rng), -0.4 + 1.8 * rand(rng))
        q = p + SVector(ntuple(_ -> 0.02 * (rand(rng) - 0.5), 3))
        @test abs(sdf(p) - sdf(q)) <= norm(p - q) * (1 + 1e-12)
    end
end

@testset "MeshIO ext — end-to-end FCM Poisson on an STL solid" begin
    v, f = _cube_mesh(0.2, 0.8)
    dom = physical_domain(mesh_levelset(triangle_mesh(v, f)); subcell_length_scale=0.05)
    V = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(4, 4, 4), order=1, physical=dom)
    m = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    assemble!(m)
    @test isposdef(Symmetric(Matrix(m.matrix)))
    sol = solve!(m)
    @test all(isfinite, sol.coefficients)
    diag = diagnostics(m)
    @test diag.cut_region_count > 0
    @test diag.fit_failure_count == 0
    @test diag.moment_fit_residual_max < 1e-8
end

@testset "MeshIO ext — sign is scale-free" begin
    # The pseudonormal sign tolerances scale with the mesh, so the sign is
    # correct on meshes in physical units far from O(1) (µm, km, …).
    for S in (1.0e-6, 1.0e6)
        v, f = _cube_mesh(0.2 * S, 0.8 * S)
        sdf = Unfitted._leaves(mesh_levelset(triangle_mesh(v, f)))[1].f
        @test sdf(SVector(0.5, 0.5, 0.5) .* S) ≈ -0.3 * S rtol = 1e-6   # inside
        @test sdf(SVector(0.5, 0.5, 0.95) .* S) > 0                     # outside
    end
end

@testset "MeshIO ext — non-axis-aligned mesh signs correctly" begin
    # All other 3D fixtures have grid-aligned facets; a rotated cube exercises
    # tilted facets (the case the per-fiber Gauss order must handle) and a
    # non-trivial pseudonormal. Validate the sign against the analytic membership
    # obtained by rotating the query back into the cube frame.
    θ, ψ = 0.6, 0.4
    Rz = SMatrix{3,3}(cos(θ), sin(θ), 0.0, -sin(θ), cos(θ), 0.0, 0.0, 0.0, 1.0)
    Rx = SMatrix{3,3}(1.0, 0.0, 0.0, 0.0, cos(ψ), sin(ψ), 0.0, -sin(ψ), cos(ψ))
    R = Rz * Rx
    v, f = _cube_mesh(-0.3, 0.3)
    vr = [R * vi for vi in v]
    sdf = Unfitted._leaves(mesh_levelset(triangle_mesh(vr, f)))[1].f
    inside_rot(p) = all(abs.(R' * SVector{3,Float64}(p)) .<= 0.3 + 1e-12)
    rng = MersenneTwister(3)
    tested = 0
    bad = 0
    for _ in 1:10000
        p = SVector(1.2 * rand(rng) - 0.6, 1.2 * rand(rng) - 0.6, 1.2 * rand(rng) - 0.6)
        d = sdf(p)
        abs(d) <= 1e-6 && continue
        tested += 1
        ((d < 0) == inside_rot(p)) || (bad += 1)
    end
    @test tested > 3000
    @test bad == 0
end

# ── 2D tests (segment loops) ──────────────────────────────────────────────────

@testset "MeshIO ext — 2D polygon classify and exact area" begin
    poly = polyline_mesh(_square_poly(0.2, 0.8); closed=true)
    ls = mesh_levelset(poly)
    @test ls isa Unfitted.LevelSet
    sdf = Unfitted._leaves(ls)[1].f
    @test sdf(SVector(0.5, 0.5)) ≈ -0.3 atol = 1e-12     # centre, 0.3 to each edge
    @test sdf(SVector(0.5, 0.95)) ≈ 0.15 atol = 1e-12    # outside above the top edge
    dom = physical_domain(ls; subcell_length_scale=0.05)
    @test classify_cell(dom, box((0.0, 0.0), (1.0, 1.0))) === :cut
    @test classify_cell(dom, box((0.4, 0.4), (0.6, 0.6))) === :full
    @test classify_cell(dom, box((0.85, 0.85), (0.95, 0.95))) === :fictitious
    @test _mesh_volume(dom.geometry, box((0.0, 0.0), (1.0, 1.0)); gp=4, ms=4) ≈ 0.6^2 atol = 1e-9
end

@testset "MeshIO ext — 2D non-convex polygon signs correctly" begin
    # The L-polygon has a reflex vertex at (1,1). In 2D the plain n₁+n₂ vertex
    # pseudonormal is already correct, so the sign must match the analytic
    # membership everywhere.
    sdf = Unfitted._leaves(mesh_levelset(polyline_mesh(_lpoly2d(); closed=true)))[1].f
    rng = MersenneTwister(11)
    tested = 0
    bad = 0
    for _ in 1:20000
        p = SVector(-0.3 + 2.6 * rand(rng), -0.3 + 2.6 * rand(rng))
        d = sdf(p)
        abs(d) <= 1e-6 && continue
        tested += 1
        ((d < 0) == _inside_L2(p)) || (bad += 1)
    end
    @test tested > 5000
    @test bad == 0
    # Winding mode agrees, including in the reentrant notch.
    sw = Unfitted._leaves(mesh_levelset(polyline_mesh(_lpoly2d(); closed=true);
                                        orientation=:winding))[1].f
    @test (sw(SVector(1.5, 1.5)) < 0) == _inside_L2(SVector(1.5, 1.5))
    @test (sw(SVector(0.5, 0.5)) < 0) == _inside_L2(SVector(0.5, 0.5))
end

@testset "MeshIO ext — mesh_levelset rejects unsupported simplex dimensions" begin
    # K=1 segments in 3D (a curve, no enclosed region) and K=2 triangles in 2D
    # are not closed-boundary cases.
    seg3d = segment_mesh([(SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0))];
                         normals=[SVector(0.0, 1.0, 0.0)])
    @test_throws ArgumentError mesh_levelset(seg3d)
end
