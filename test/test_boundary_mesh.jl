using LinearAlgebra
using StaticArrays

# The physical weights of a surface region's rule, as the region accessors hand them to every consumer: a region stores
# its rule in its sub-cell's bounding-box frame, so its `weights` are frame weights.
surface_weights(region) = region.weights .* Unfitted._region_jacobian(region)

@testset "BoundaryMesh constructors enforce shape and dimension rules" begin
    # K=0 requires explicit normals; empty `normals` is a dimension mismatch against a non-empty `points`.
    pts = [SVector(0.0, 0.0), SVector(0.5, 0.5)]
    @test_throws DimensionMismatch points_mesh(pts; normals=SVector{2,Float64}[])
    pmesh = points_mesh(pts; normals=[SVector(1.0, 0.0), SVector(0.0, 1.0)])
    @test pmesh isa BoundaryMesh{2,Float64,0}
    @test length(pmesh.cells) == 2

    # K=1 in 2D: explicit segment list + polyline shortcut.
    pairs = [(SVector(0.0, 0.0), SVector(1.0, 0.0))]
    smesh = segment_mesh(pairs)
    @test smesh isa BoundaryMesh{2,Float64,1}
    polyline = polyline_mesh([SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0)])
    @test length(polyline.cells) == 2
    closed = polyline_mesh([SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0)]; closed=true)
    @test length(closed.cells) == 3

    # K=2 in 3D.
    vertices3d = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)]
    tmesh = triangle_mesh(vertices3d, [(1, 2, 3)])
    @test tmesh isa BoundaryMesh{3,Float64,2}

    # D ≥ 4 is rejected.
    @test_throws ArgumentError BoundaryMesh{4,Float64,1}(NTuple{2,SVector{4,Float64}}[], SVector{4,Float64}[])

    # Wrong vertex count for the declared K.
    @test_throws ArgumentError BoundaryMesh{2,Float64,1}([(SVector(0.0, 0.0),)], nothing)

    # Mismatched normals length.
    @test_throws DimensionMismatch segment_mesh([(SVector(0.0, 0.0), SVector(1.0, 0.0))];
                                                normals=[SVector(0.0, 1.0), SVector(0.0, 1.0)])

    # K=0 without normals is rejected at construction.
    @test_throws ArgumentError BoundaryMesh{2,Float64,0}([(SVector(0.0, 0.0),)], nothing)
end

@testset "BoundaryMesh simplex measure and default normal" begin
    # K=1 segment length and outward normal in 2D.
    seg = (SVector(0.0, 0.0), SVector(2.0, 0.0))
    @test Unfitted._simplex_measure(seg) ≈ 2.0
    @test Unfitted._default_normal(seg) ≈ SVector(0.0, -1.0)

    # K=2 triangle area and outward normal in 3D (right-hand rule).
    tri = (SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0))
    @test Unfitted._simplex_measure(tri) ≈ 0.5
    @test Unfitted._default_normal(tri) ≈ SVector(0.0, 0.0, 1.0)

    # K=0 point: measure 1, no default normal.
    pt = (SVector(0.3, 0.4),)
    @test Unfitted._simplex_measure(pt) == 1.0
    @test_throws ArgumentError Unfitted._default_normal(pt)

    # Nor does any other simplex that is not codim-1: a segment in 1D (codim 0; this was a MethodError) or in 3D (a
    # circle of normals), a triangle in 2D. Each raises the same ArgumentError, and so does a mesh of them integrated
    # without explicit normals.
    @test_throws ArgumentError Unfitted._default_normal((SVector(0.0), SVector(1.0)))
    @test_throws ArgumentError Unfitted._default_normal((SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0)))
    @test_throws ArgumentError Unfitted._default_normal((SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0)))
    u = field(:u, space(box((0.0,), (1.0,)); cells=4, order=1))
    interval = segment_mesh([(SVector(0.1), SVector(0.9))])
    @test_throws "no geometric normal" prepare(Problem((u,);
                                                       loads=(loadform(u, source_form(source=1.0); on=interval),)))
end

@testset "surface region builder: simple 2D segment inside one cell" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=2)
    # A short segment fully inside cell (1,1) (the cell covers x ∈ [0, 0.25], y ∈ [0, 0.25]).
    mesh = segment_mesh([(SVector(0.05, 0.10), SVector(0.20, 0.10))])
    regions = Unfitted._surface_regions_for_mesh(V, mesh, GeometryTolerance(Float64))

    @test length(regions) == 1
    region = regions[1]
    @test length(region.parents) == 1
    @test region.parents[1].level == 1
    @test region.kind === :surface
    @test sum(surface_weights(region)) ≈ 0.15 atol = 1.0e-12
    expected_normal = Unfitted._default_normal((SVector(0.05, 0.10), SVector(0.20, 0.10)))
    @test region.normal ≈ expected_normal
end

@testset "surface region builder: straddling segment is subdivided against the grid" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    # Segment crossing the y = 0.25 grid line. With subdivision the builder produces two sub-segments — one in cell (1,
    # 1), one in cell (1, 2) — instead of throwing.
    straddling = segment_mesh([(SVector(0.1, 0.20), SVector(0.1, 0.30))])
    regions = Unfitted._surface_regions_for_mesh(V, straddling, GeometryTolerance(Float64))
    @test length(regions) == 2

    # Subdivision partitions the segment, so the sub-segment quadrature weights sum to the original length at round-off
    # — there is no tolerance floor to leave room for.
    total_length = sum(sum(surface_weights(r)) for r in regions)
    @test total_length ≈ 0.1 rtol = 1.0e-14

    # The two sub-segments live on different cells of level 1.
    cells_touched = [region.parents[1].cell for region in regions]
    @test length(unique(cells_touched)) == 2
end

@testset "boundary_integral on BoundaryMesh: closed square perimeter is exact" begin
    # Pick a square whose every edge lies strictly inside one grid cell. cells=(4,4) on [0,1]² → grid lines at multiples
    # of 0.25. A 0.1 × 0.1 square inside cell (1,1) has all four edges within one cell.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=2)
    p = [SVector(0.05, 0.05), SVector(0.20, 0.05), SVector(0.20, 0.20), SVector(0.05, 0.20)]
    square = polyline_mesh(p; closed=true)
    u = field(:u, V)
    load = loadform(u, source_form(source=0.0); on=square)
    model = prepare(Problem((u,); loads=(load,)))

    perimeter = boundary_integral(q -> 1.0, model; on=square)
    @test perimeter ≈ 0.6 atol = 1.0e-12  # 4 × 0.15
end

@testset "boundary_integral on BoundaryMesh: q.normal is the cell's geometric normal" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    seg = segment_mesh([(SVector(0.05, 0.10), SVector(0.20, 0.10))])
    u = field(:u, V)
    model = prepare(Problem((u,); loads=(loadform(u, source_form(source=0.0); on=seg),)))

    normals_seen = SVector{2,Float64}[]
    boundary_integral(model; on=seg) do q
        push!(normals_seen, q.normal)
        return 0.0
    end
    @test !isempty(normals_seen)
    @test all(n ≈ SVector(0.0, -1.0) for n in normals_seen)
end

# Allocation count of one warm `boundary_integral` call, behind a function barrier so the count is the call's own.
function _warm_boundary_allocations(integrand, model, on)
    boundary_integral(integrand, model; on)
    boundary_integral(integrand, model; on)
    return @allocations boundary_integral(integrand, model; on)
end

@testset "boundary_integral allocates nothing per quadrature point" begin
    # `boundary_integral` folds over the quadrature-point walker, whose point loop runs behind a function barrier on the
    # concrete region type. A loop over the untyped region list instead dispatches, and allocates, at every point: an
    # earlier version did, and on the 16 × 16 fixture below took 6 570 allocations and 655 µs for the ring where this
    # one takes 12 and about 1 µs. A warm call's allocation count must not grow with the number of points, on an
    # immersed surface and on a physical facet target alike.
    function ring_model(c)
        V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(c, c), order=2)
        θ = range(0, 2π; length=8c + 1)[1:(end-1)]
        ring = polyline_mesh([SVector(0.5 + 0.31cos(t), 0.5 + 0.27sin(t)) for t in θ]; closed=true)
        u = field(:u, V)
        return prepare(Problem((u,); loads=(loadform(u, source_form(source=0.0); on=ring),))), ring
    end
    integrand = q -> q.normal[1] * q.x[1]
    (small, small_ring), (large, large_ring) = ring_model(4), ring_model(16)
    @test nquadpoints(large; on=large_ring) ≥ 4 * nquadpoints(small; on=small_ring)
    @test _warm_boundary_allocations(integrand, large, large_ring) -
          _warm_boundary_allocations(integrand, small, small_ring) < 10
    @test _warm_boundary_allocations(integrand, large, boundary(:all)) -
          _warm_boundary_allocations(integrand, small, boundary(:all)) < 10
end

@testset "model.regions caches one entry per BoundaryMesh" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    mesh = segment_mesh([(SVector(0.05, 0.10), SVector(0.20, 0.10))])
    u = field(:u, V)
    load = loadform(u, source_form(source=0.0); on=mesh)
    model = prepare(Problem((u,); loads=(load,)))

    # One entry, keyed on (mesh, space): the mesh half matches by object identity, the space half is the subdomain the
    # regions were cut against — here the model's only space.
    @test length(model.regions) == 1
    on, key_space = only(keys(model.regions))
    @test on === mesh
    @test key_space === model.problem.space
    @test haskey(model.regions, (mesh, model.problem.space))
    @test diagnostics(model).surface_region_count == length(model.regions[(mesh, model.problem.space)])
    @test nquadpoints(model; kind=:surface) > 0
    # With one cached mesh the aggregate and the per-`on=` count coincide; the `on=` form is the one that stays equal to
    # the `q.point` range once a second mesh is cached.
    @test nquadpoints(model; on=mesh) == nquadpoints(model; kind=:surface)
end

@testset "block(...; on=BoundaryMesh) contributes to the assembled matrix" begin
    # A surface mass operator added on a segment perturbs the volume stiffness system. Use cells=(4,4) so the segment
    # fits in one cell.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=2)
    seg = segment_mesh([(SVector(0.05, 0.50), SVector(0.20, 0.50))])  # fully in cell (1,2)
    u = field(:u, V)
    surface_block = block(u, u, mass_form(coefficient=1000.0); on=seg)

    plain = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    augmented = prepare(Problem((u,); blocks=(stiffness_block(u), surface_block), loads=(),
                                dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    assemble!(plain)
    assemble!(augmented)
    @test !(Matrix(augmented.matrix) ≈ Matrix(plain.matrix))
    @test issymmetric(augmented.matrix)
end

@testset "BoundaryMesh contribution matches an equivalent volume mass-integral" begin
    # A horizontal segment fully inside one grid cell. We verify the surface mass contribution `∫_seg φᵢ φⱼ ds` equals
    # what a brute- force 1D Gauss sum gives, by integrating
    # `(boundary_integral on the cell's segment) of the constant 1` against the assembled weight pattern. Cleaner check:
    # integrate `∫_seg 1 ds` and confirm it equals the segment length precisely.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=2)
    seg = segment_mesh([(SVector(0.10, 0.10), SVector(0.20, 0.10))])
    u = field(:u, V)
    model = prepare(Problem((u,); loads=(loadform(u, source_form(source=0.0); on=seg),)))
    @test boundary_integral(q -> 1.0, model; on=seg) ≈ 0.10 atol = 1.0e-12
end

@testset "_subdivide_segment: segment crossing multiple grid lines" begin
    # Segment from (0.05, 0.10) to (0.85, 0.40). Crosses x = 0.25, 0.5, 0.75 and y = 0.25 — four crossings, so five
    # sub-segments.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    grid_lines = Unfitted._merged_axis_coordinates(V.levels, Val(2), GeometryTolerance(Float64))
    p1 = SVector(0.05, 0.10)
    p2 = SVector(0.85, 0.40)
    subs = Unfitted._subdivide_segment(p1, p2, grid_lines, sqrt(eps(Float64)))
    @test length(subs) == 5

    # The sub-segments *partition* the input: they start at `p1`, end at `p2`, and each one begins exactly where the
    # last ended. No arc length may be discarded at a crossing, so the lengths sum back to the whole at round-off — not
    # at some tolerance floor.
    @test subs[1][1] == p1
    @test subs[end][2] == p2
    @test all(subs[i][2] == subs[i + 1][1] for i in 1:(length(subs)-1))
    original_length = sqrt(0.80^2 + 0.30^2)
    total = sum(norm(sub[2] - sub[1]) for sub in subs)
    @test total ≈ original_length rtol = 1.0e-14

    # Parent-uniqueness is a statement about each sub-segment's *interior*: the whole open sub-segment lies in one cell,
    # which is what lets the region builder attribute it to a parent by midpoint. The shared endpoints deliberately sit
    # *on* grid lines, where `locate_cell` snaps to one side and cannot classify them — exactly as for the K=2 clipping
    # path below.
    for sub in subs
        cells = [Unfitted.locate_cell(V.levels[1].mesh, sub[1] + f * (sub[2] - sub[1]))
                 for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
        @test all(c === cells[1] !== nothing for c in cells)
    end
end

@testset "_subdivide_segment: corner crossing and degenerate splits" begin
    # A segment through a grid *intersection* produces one crossing per axis at the same parameter, agreeing only to a
    # few ulp. They must collapse to a single cut — not to a zero-length sliver straddling the corner, and not to a gap.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    grid_lines = Unfitted._merged_axis_coordinates(V.levels, Val(2), GeometryTolerance(Float64))
    nudge = sqrt(eps(Float64))

    corner = Unfitted._subdivide_segment(SVector(0.05, 0.05), SVector(0.45, 0.45), grid_lines, nudge)
    @test length(corner) == 2
    @test corner[1][2] == corner[2][1]
    @test sum(norm(s[2] - s[1]) for s in corner) ≈ norm(SVector(0.40, 0.40)) rtol = 1.0e-14

    # Endpoints sitting exactly on grid lines are crossings at t = 0 and t = 1. Neither may spawn a degenerate
    # sub-segment, and neither may cost any length.
    on_lines = Unfitted._subdivide_segment(SVector(0.25, 0.25), SVector(0.75, 0.75), grid_lines, nudge)
    @test all(norm(s[2] - s[1]) > nudge for s in on_lines)
    @test on_lines[1][1] == SVector(0.25, 0.25)
    @test on_lines[end][2] == SVector(0.75, 0.75)
    @test sum(norm(s[2] - s[1]) for s in on_lines) ≈ norm(SVector(0.50, 0.50)) rtol = 1.0e-14

    # A segment lying *along* a grid line still splits across the other axis and still conserves its length exactly.
    along = Unfitted._subdivide_segment(SVector(0.05, 0.25), SVector(0.95, 0.25), grid_lines, nudge)
    @test length(along) == 4
    @test sum(norm(s[2] - s[1]) for s in along) ≈ 0.90 rtol = 1.0e-14
end

@testset "_subdivide_triangle: triangle split against the 3D grid" begin
    # A triangle in 3D crossing two grid planes. Its sub-triangles partition it — their areas sum to the original's at
    # round-off — and each lies inside one cell of the grid, vertices included.
    omega = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = space(omega; cells=(2, 2, 2), order=1)
    grid_lines = Unfitted._merged_axis_coordinates(V.levels, Val(3), GeometryTolerance(Float64))
    v1 = SVector(0.1, 0.1, 0.3)
    v2 = SVector(0.9, 0.1, 0.3)
    v3 = SVector(0.5, 0.9, 0.3)
    subs = Unfitted._subdivide_triangle(v1, v2, v3, grid_lines)

    # Original triangle area = 0.5·|e1 × e2| = 0.5·|(0.8,0,0) × (0.4,0.8,0)| = 0.5·|0.64 k̂| = 0.32.
    expected_area = 0.32
    function area(tri)
        e1 = tri[2] - tri[1]
        e2 = tri[3] - tri[1]
        c = SVector(e1[2]*e2[3]-e1[3]*e2[2], e1[3]*e2[1]-e1[1]*e2[3], e1[1]*e2[2]-e1[2]*e2[1])
        sqrt(sum(x->x*x, c)) / 2
    end
    total_area = sum(area, subs)
    @test total_area ≈ expected_area rtol = 1.0e-14
    @test length(subs) >= 3

    # The midpoint of every sub-triangle locates its cell, and the whole sub-triangle lies in that cell's box. Vertices
    # may sit on grid planes, where `locate_cell` cannot classify them uniquely, which is why the surface-region builder
    # assigns the parent by the midpoint.
    for tri in subs
        mid = (tri[1] + tri[2] + tri[3]) / 3
        cell = Unfitted.locate_cell(V.levels[1].mesh, mid)
        @test cell !== nothing
        cbox = cell_box(V.levels[1].mesh, cell)
        @test all(all(cbox.lower .<= v .<= cbox.upper) for v in tri)
    end
end

@testset "a triangle reaching past the grid raises like a segment; an interface skips the part outside" begin
    # Regression (F4). The triangle subdivider once visited only the cells between the outermost grid lines, so the
    # part of a triangle beyond them vanished silently: this surface integrated 0.12 of its 0.16 area. A segment
    # leaving the grid raises instead, and now a triangle does too, because its outside part is kept as a sub-cell no
    # level covers.
    omega = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = space(omega; cells=(4, 4, 4), order=1)
    u = field(:u, V)
    Γ = triangle_mesh([SVector(0.6, 0.3, 0.4), SVector(1.4, 0.3, 0.4), SVector(0.6, 0.7, 0.4)], [(1, 2, 3)])
    @test_throws "entirely outside" prepare(Problem((u,); loads=(loadform(u, source_form(source=0.0); on=Γ),)))
    leg = segment_mesh([(SVector(0.6, 0.3, 0.4), SVector(1.4, 0.3, 0.4))]; normals=[SVector(0.0, 1.0, 0.0)])
    @test_throws "entirely outside" prepare(Problem((u,); loads=(loadform(u, source_form(source=0.0); on=leg),)))

    # An interface integrates the trace inside both subdomains and skips the rest, whole or in part: the area left of
    # x = 1 is ∫_{0.6}^{1} (1.4 − x)/2 dx = 0.12.
    u1, u2 = field(:u1, V), field(:u2, space(omega; cells=(3, 3, 3), order=1))
    blocks = couple(u1, u2, Γ, mass_form(coefficient=1.0))
    model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2), blocks...)))
    @test boundary_integral(q -> 1.0, model; on=first(blocks).on) ≈ 0.12 rtol = 1.0e-14
end

@testset "triangle subdivision is the same at every geometric scale" begin
    # Regression (F4). The triangle subdivider compared physical coordinates against an absolute `nudge = √eps` and
    # dropped sub-triangles below an absolute area `nudge²`, so the same mesh on the same grid split differently at a
    # different scale: at 1e-6 this triangle came out as 32 sub-cells and 1.7 % too much area, at 1 and 1e6 as 33 and
    # the exact area. Every test of the split is now exact or relative to the input.
    P = [SVector(0.13, 0.27, 0.31), SVector(0.86, 0.12, 0.58), SVector(0.41, 0.93, 0.74)]
    function split(s)
        V = space(box((0.0, 0.0, 0.0), (s, s, s)); cells=(4, 4, 4), order=1)
        u = field(:u, V)
        Γ = triangle_mesh([s * p for p in P], [(1, 2, 3)])
        model = prepare(Problem((u,); loads=(loadform(u, source_form(source=0.0); on=Γ),));
                        tolerance=GeometryTolerance(; merge=sqrt(eps()) * s))
        return diagnostics(model).surface_region_count, boundary_integral(q -> 1.0, model; on=Γ) / s^2
    end
    count, area = split(1.0)
    @test area ≈ norm(cross(P[2] - P[1], P[3] - P[1])) / 2 rtol = 1.0e-14
    for s in (1.0e-6, 1.0e6)
        scaled_count, scaled_area = split(s)
        @test scaled_count == count
        @test scaled_area ≈ area rtol = 1.0e-14
    end
end

@testset "a closed surface on the grid planes is integrated once" begin
    # Regression (fixed with F4). Every face of this cube lies in an interior grid plane of the 4³ grid. The triangle
    # clipper once visited the cells under a triangle's bounding box widened by an absolute `nudge` — for a triangle in
    # a grid plane, the cells on both sides of it — and its nudge-tolerant half-space clip kept the whole triangle in
    # each, so every face was integrated twice: area 3.0 and ∯ x·n = 0.25. The plane splitter splits a piece only at
    # the lines strictly inside its extent, so an in-plane triangle stays whole and its midpoint picks one cell.
    V = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(4, 4, 4), order=1)
    a, b = 0.25, 0.75
    vertices = [SVector(x, y, z) for z in (a, b) for y in (a, b) for x in (a, b)]
    faces = [(1, 3, 2), (2, 3, 4), (5, 6, 7), (6, 8, 7), (1, 2, 5), (2, 6, 5), (3, 7, 4), (4, 7, 8), (1, 5, 3),
             (3, 5, 7), (2, 4, 6), (4, 8, 6)]                       # outward, two triangles per face
    Γ = triangle_mesh(vertices, faces)
    u = field(:u, V)
    model = prepare(Problem((u,); loads=(loadform(u, source_form(source=0.0); on=Γ),)))
    @test boundary_integral(q -> 1.0, model; on=Γ) ≈ 6 * (b - a)^2 rtol = 1.0e-14
    @test boundary_integral(q -> q.x[1] * q.normal[1], model; on=Γ) ≈ (b - a)^3 rtol = 1.0e-14   # the volume
end

@testset "BoundaryMesh: subdivision preserves per-cell normals" begin
    # User-supplied normal on a single segment that gets split by the subdivision into multiple sub-segments. Every
    # sub-segment must inherit the original normal.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    mesh = segment_mesh([(SVector(0.1, 0.1), SVector(0.6, 0.2))]; normals=[SVector(0.0, 1.0)])
    sub = Unfitted._subdivide_mesh(mesh,
                                   Unfitted._merged_axis_coordinates(V.levels, Val(2), GeometryTolerance(Float64)))
    @test length(sub.cells) > 1
    @test all(n == SVector(0.0, 1.0) for n in sub.normals)
    # The regions carry it, and so does every sub-cell of a mesh without an override: the input cell's geometric normal.
    tol = GeometryTolerance(Float64)
    @test all(r.normal == SVector(0.0, 1.0) for r in Unfitted._surface_regions_for_mesh(V, mesh, tol))
    plain = Unfitted._surface_regions_for_mesh(V, segment_mesh([(SVector(0.1, 0.1), SVector(0.6, 0.2))]), tol)
    @test length(plain) > 1
    @test all(r.normal == Unfitted._default_normal((SVector(0.1, 0.1), SVector(0.6, 0.2))) for r in plain)
end

@testset "BoundaryMesh in 3D: triangle area integrates correctly" begin
    omega = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = space(omega; cells=(4, 4, 4), order=1)
    # Triangle whose vertices all lie inside cell (1,1,1).
    v1 = SVector(0.05, 0.05, 0.10)
    v2 = SVector(0.20, 0.05, 0.10)
    v3 = SVector(0.05, 0.20, 0.10)
    mesh = triangle_mesh([v1, v2, v3], [(1, 2, 3)])
    u = field(:u, V)
    model = prepare(Problem((u,); loads=(loadform(u, source_form(source=0.0); on=mesh),)))
    expected_area = 0.5 * 0.15 * 0.15
    @test boundary_integral(q -> 1.0, model; on=mesh) ≈ expected_area atol = 1.0e-12
end

@testset "the surface rule is sized from the sub-cell's own parents" begin
    # The rule a sub-cell uses comes from the cells covering it. A high-order level elsewhere must not raise the point
    # count of a segment it does not cover — which is what the level-wide maximum did, and the reason the
    # immersed-surface path was the one quadrature consumer that a per-cell order made more expensive rather than less.
    omega = box((0.0, 0.0), (1.0, 1.0))
    tol = GeometryTolerance(Float64)
    V = overlay(space(omega; cells=(4, 4), order=1), box((0.5, 0.5), (1.0, 1.0)); cells=(2, 2), order=3)

    near = Unfitted._surface_regions_for_mesh(V, segment_mesh([(SVector(0.05, 0.10), SVector(0.20, 0.10))]), tol)
    @test length(only(near).parents) == 1                 # the base level alone
    @test length(only(near).points) == 2                  # p = 1 → 2 Gauss points

    far = Unfitted._surface_regions_for_mesh(V, segment_mesh([(SVector(0.55, 0.60), SVector(0.70, 0.60))]), tol)
    @test length(only(far).parents) == 2
    @test length(only(far).points) == 4                   # max p = 3 → 4 Gauss points

    # The same question on a level carrying a per-cell order, where there is no per-level order to read even in
    # principle.
    orders = fill(1, 4, 4)
    orders[4, 4] = 4
    W = space(omega; cells=(4, 4), order=orders)
    quiet = Unfitted._surface_regions_for_mesh(W, segment_mesh([(SVector(0.05, 0.10), SVector(0.20, 0.10))]), tol)
    @test length(only(quiet).points) == 2
    loud = Unfitted._surface_regions_for_mesh(W, segment_mesh([(SVector(0.80, 0.85), SVector(0.95, 0.85))]), tol)
    @test length(only(loud).points) == 5                  # p = 4 → 5 Gauss points

    # An oblique segment crosses both axes, and each adds its own degree: 1 + p + p points, from the same parents.
    oblique(V, a, b) = length(only(Unfitted._surface_regions_for_mesh(V, segment_mesh([(a, b)]), tol)).points)
    @test oblique(V, SVector(0.05, 0.10), SVector(0.20, 0.15)) == 3     # base p = 1
    @test oblique(V, SVector(0.55, 0.60), SVector(0.70, 0.65)) == 7     # max p = 3
    @test oblique(W, SVector(0.80, 0.80), SVector(0.95, 0.85)) == 9     # p = 4

    # Under per-axis orders an axis-aligned segment takes its own axis's count, not the largest one: along x the trace
    # product of an order-(1, 3) cell has degree 2, which 2 Gauss points integrate exactly; along y it has degree 6.
    A = space(omega; cells=(4, 4), order=(1, 3))
    @test oblique(A, SVector(0.05, 0.10), SVector(0.20, 0.10)) == 2     # along x: p = 1
    @test oblique(A, SVector(0.10, 0.05), SVector(0.10, 0.20)) == 4     # along y: p = 3
    @test oblique(A, SVector(0.05, 0.10), SVector(0.20, 0.15)) == 5     # oblique: 1 + 1 + 3
end

# The surface mass and stiffness matrices of `Γ` on the one-cell space `V` (p = 2), and the same matrices from an
# independent reference: the basis functions evaluated one at a time through the public point evaluation
# (`value` / `field_gradient` of the unit coefficient vectors) and integrated with a high-order rule `(points, weights)`
# that the caller lays on `Γ` itself. On one cell the integrands are polynomials, so the reference is exact.
function _surface_operators_against_reference(V, Γ, points, weights)
    u = field(:u, V)
    model = prepare(Problem((u,); blocks=(block(u, u, mass_form(); on=Γ),)))
    M = Matrix(assemble_matrix(model, block(u, u, mass_form(); on=Γ); threaded=false))
    K = Matrix(assemble_matrix(model, block(u, u, stiffness_form(); on=Γ); threaded=false))
    n = active_unknowns(model)
    basis = [solution(model, [Float64(i == j) for j in 1:n]) for i in 1:n]
    φ = [value(b, model, x) for b in basis, x in points]
    ∇φ = [field_gradient(b, model, x) for b in basis, x in points]
    M_ref = [sum(weights[k] * φ[i, k] * φ[j, k] for k in eachindex(weights)) for i in 1:n, j in 1:n]
    K_ref = [sum(weights[k] * dot(∇φ[i, k], ∇φ[j, k]) for k in eachindex(weights)) for i in 1:n, j in 1:n]
    return maximum(abs, M - M_ref) / maximum(abs, M_ref), maximum(abs, K - K_ref) / maximum(abs, K_ref),
           nquadpoints(model; on=Γ)
end

@testset "simplex rules integrate trace products exactly on oblique segments and triangles" begin
    # Regression (F2). Surface and interface rules took their point count from the parents' per-axis Gauss count
    # alone — `p + 1` points on a segment, a (p + 1) × (p + 1) Duffy rule on a triangle — which is exact only where the
    # simplex runs along a coordinate axis or plane. Along an oblique simplex a product of two basis functions of
    # per-axis degree p has total degree 2p *per axis it crosses*, so main's rules were off by 2e-3 to 4e-3 of the
    # largest entry here. The rule now takes `1 + Σ_d p_d` points per axis, summed over the axes the simplex extends
    # along.
    ξ, ω = Unfitted.gausslegendre(16)
    s, w = (ξ .+ 1) ./ 2, ω ./ 2                         # 16-point Gauss on [0, 1], exact to degree 31

    # A 2D segment oblique to both axes: 1 + 2 + 2 = 5 points (a horizontal one keeps 3).
    a, b = SVector(0.12, 0.31), SVector(0.83, 0.67)
    seg = segment_mesh([(a, b)])
    V2 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=2)
    mass_err, stiffness_err, count = _surface_operators_against_reference(V2, seg, [a + t * (b - a) for t in s],
                                                                          w .* norm(b - a))
    @test mass_err < 1.0e-13
    @test stiffness_err < 1.0e-13
    @test count == 5

    # A 3D triangle oblique to all three axes: (1 + 2 + 2 + 2)² = 49 points. The reference is a 16 × 16 Duffy rule,
    # written out here rather than taken from the package.
    v1, v2, v3 = SVector(0.12, 0.21, 0.33), SVector(0.81, 0.17, 0.52), SVector(0.37, 0.88, 0.71)
    tri = triangle_mesh([v1, v2, v3], [(1, 2, 3)])
    jacobian = norm(cross(v2 - v1, v3 - v1))
    points = [v1 + s[j] * (v2 - v1) + s[i] * (1 - s[j]) * (v3 - v1) for i in eachindex(s) for j in eachindex(s)]
    weights = [w[i] * w[j] * (1 - s[j]) * jacobian for i in eachindex(s) for j in eachindex(s)]
    V3 = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(1, 1, 1), order=2)
    mass_err, stiffness_err, count = _surface_operators_against_reference(V3, tri, points, weights)
    @test mass_err < 1.0e-13
    @test stiffness_err < 1.0e-13
    @test count == 49

    # A triangle in a coordinate plane extends along two axes: (1 + 2 + 2)² = 25 points.
    flat = triangle_mesh([SVector(0.12, 0.21, 0.4), SVector(0.81, 0.17, 0.4), SVector(0.37, 0.88, 0.4)], [(1, 2, 3)])
    u = field(:u, V3)
    @test nquadpoints(prepare(Problem((u,); blocks=(block(u, u, mass_form(); on=flat),))); on=flat) == 25

    # Under per-axis orders the count follows the axes the simplex extends along, so an axis-aligned segment may take
    # fewer points than the cell's largest per-axis count: along x an order-(1, 3) cell's trace products have degree 2,
    # which 2 points integrate exactly, mass and stiffness alike.
    a, b = SVector(0.12, 0.31), SVector(0.83, 0.31)
    Va = space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=(1, 3))
    mass_err, stiffness_err, count = _surface_operators_against_reference(Va, segment_mesh([(a, b)]),
                                                                          [a + t * (b - a) for t in s],
                                                                          w .* norm(b - a))
    @test mass_err < 1.0e-13
    @test stiffness_err < 1.0e-13
    @test count == 2
end

@testset "a 2D triangle mesh is split at the grid: its surface mass is the volume mass" begin
    # Regression (F3b). A K = 2 mesh in 2D passed through the subdivision unsplit, so each triangle was integrated as
    # one region with the parents of its midpoint's cells: across every other cell it evaluated their basis functions'
    # polynomial extensions. Two triangles tiling the domain must reproduce the volume mass matrix — on an overlay too,
    # where each sub-triangle carries the parents of both levels.
    V = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=2), box((0.25, 0.25), (0.75, 0.75));
                cells=(2, 2), order=3)
    u = field(:u, V)
    corners = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)]
    tiles = triangle_mesh(corners, [(1, 2, 3), (1, 3, 4)]; normals=[SVector(1.0, 0.0), SVector(1.0, 0.0)])
    model = prepare(Problem((u,); blocks=(mass_block(u), block(u, u, mass_form(); on=tiles))))
    volume = assemble_matrix(model, mass_block(u); threaded=false)
    surface = assemble_matrix(model, block(u, u, mass_form(); on=tiles); threaded=false)
    @test maximum(abs, surface - volume) / maximum(abs, volume) < 1.0e-13
    @test boundary_integral(q -> 1.0, model; on=tiles) ≈ 1.0 rtol = 1.0e-14
end

@testset "BoundaryMesh accepts points, segments and triangles only" begin
    # Regression (F3). `0 ≤ K ≤ D` let a tetrahedron mesh through, which then failed far downstream; no subdivision or
    # rule exists past triangles, so the constructor says so.
    tet = (SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))
    @test_throws "min(D, 2)" BoundaryMesh{3,Float64,3}([tet], [SVector(1.0, 0.0, 0.0)])
    triangle = (SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0))
    @test BoundaryMesh{2,Float64,2}([triangle], nothing) isa BoundaryMesh
    @test BoundaryMesh{1,Float64,1}([(SVector(0.0), SVector(1.0))], nothing) isa BoundaryMesh
end
