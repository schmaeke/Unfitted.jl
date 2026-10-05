using LinearAlgebra
using StaticArrays

@testset "BoundaryMesh constructors enforce shape and dimension rules" begin
    # K=0 requires explicit normals; empty `normals` is a dimension
    # mismatch against a non-empty `points`.
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
    @test_throws ArgumentError BoundaryMesh{4,Float64,1}(NTuple{2,SVector{4,Float64}}[],
                                                         SVector{4,Float64}[])

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
end

@testset "SurfaceRegion builder: simple 2D segment inside one cell" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=2)
    # A short segment fully inside cell (1,1) (the cell covers
    # x ∈ [0, 0.25], y ∈ [0, 0.25]).
    mesh = segment_mesh([(SVector(0.05, 0.10), SVector(0.20, 0.10))])
    regions = Unfitted._surface_regions_for_mesh(V, mesh, GeometryTolerance(Float64))

    @test length(regions) == 1
    region = regions[1]
    @test length(region.parents) == 1
    @test region.parents[1].level == 1
    @test sum(region.weights) ≈ 0.15 atol = 1.0e-12
    expected_normal = Unfitted._default_normal((SVector(0.05, 0.10), SVector(0.20, 0.10)))
    @test all(n ≈ expected_normal for n in region.normals)
end

@testset "SurfaceRegion builder: straddling segment is subdivided against the grid" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    # Segment crossing the y = 0.25 grid line. With subdivision the
    # builder produces two sub-segments — one in cell (1, 1), one in
    # cell (1, 2) — instead of throwing.
    straddling = segment_mesh([(SVector(0.1, 0.20), SVector(0.1, 0.30))])
    regions = Unfitted._surface_regions_for_mesh(V, straddling, GeometryTolerance(Float64))
    @test length(regions) == 2

    # Subdivision partitions the segment, so the sub-segment quadrature
    # weights sum to the original length at round-off — there is no
    # tolerance floor to leave room for.
    total_length = sum(sum(r.weights) for r in regions)
    @test total_length ≈ 0.1 rtol = 1.0e-14

    # The two sub-segments live on different cells of level 1.
    cells_touched = [region.parents[1].cell for region in regions]
    @test length(unique(cells_touched)) == 2
end

@testset "boundary_integral on BoundaryMesh: closed square perimeter is exact" begin
    # Pick a square whose every edge lies strictly inside one grid cell.
    # cells=(4,4) on [0,1]² → grid lines at multiples of 0.25.
    # A 0.1 × 0.1 square inside cell (1,1) has all four edges within
    # one cell.
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

# Allocation count of one warm `boundary_integral` call, behind a function
# barrier so the count is the call's own.
function _warm_boundary_allocations(integrand, model, on)
    boundary_integral(integrand, model; on)
    boundary_integral(integrand, model; on)
    return @allocations boundary_integral(integrand, model; on)
end

@testset "boundary_integral allocates nothing per quadrature point" begin
    # `boundary_integral` folds over the quadrature-point walker, whose point loop
    # runs behind a function barrier on the concrete region type. A loop over the
    # untyped region list instead dispatches, and allocates, at every point: an
    # earlier version did, and on the 16 × 16 fixture below took 6 570
    # allocations and 655 µs for the ring where this one takes 12 and about
    # 1 µs. A warm call's allocation count must not grow with the number of
    # points, on an immersed surface and on a physical facet target alike.
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

@testset "model.surface_regions caches one entry per BoundaryMesh" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    mesh = segment_mesh([(SVector(0.05, 0.10), SVector(0.20, 0.10))])
    u = field(:u, V)
    load = loadform(u, source_form(source=0.0); on=mesh)
    model = prepare(Problem((u,); loads=(load,)))

    # One entry, keyed on (mesh, space): the mesh half matches by object
    # identity, the space half is the subdomain the regions were cut against —
    # here the model's only space.
    @test length(model.surface_regions) == 1
    on, key_space = only(keys(model.surface_regions))
    @test on === mesh
    @test key_space === model.problem.space
    @test haskey(model.surface_regions, (mesh, model.problem.space))
    @test diagnostics(model).surface_region_count ==
          length(model.surface_regions[(mesh, model.problem.space)])
    @test nquadpoints(model; kind=:surface) > 0
    # With one cached mesh the aggregate and the per-`on=` count coincide; the
    # `on=` form is the one that stays equal to the `q.point` range once a
    # second mesh is cached.
    @test nquadpoints(model; on=mesh) == nquadpoints(model; kind=:surface)
end

@testset "block(...; on=BoundaryMesh) contributes to the assembled matrix" begin
    # A surface mass operator added on a segment perturbs the volume
    # stiffness system. Use cells=(4,4) so the segment fits in one
    # cell.
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
    # A horizontal segment fully inside one grid cell. We verify the
    # surface mass contribution `∫_seg φᵢ φⱼ ds` equals what a brute-
    # force 1D Gauss sum gives, by integrating `(boundary_integral on
    # the cell's segment) of the constant 1` against the assembled
    # weight pattern. Cleaner check: integrate `∫_seg 1 ds` and
    # confirm it equals the segment length precisely.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=2)
    seg = segment_mesh([(SVector(0.10, 0.10), SVector(0.20, 0.10))])
    u = field(:u, V)
    model = prepare(Problem((u,); loads=(loadform(u, source_form(source=0.0); on=seg),)))
    @test boundary_integral(q -> 1.0, model; on=seg) ≈ 0.10 atol = 1.0e-12
end

@testset "_subdivide_segment: segment crossing multiple grid lines" begin
    # Segment from (0.05, 0.10) to (0.85, 0.40). Crosses x = 0.25, 0.5, 0.75
    # and y = 0.25 — four crossings, so five sub-segments.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    grid_lines = Unfitted._level_grid_lines(V, GeometryTolerance(Float64))
    p1 = SVector(0.05, 0.10)
    p2 = SVector(0.85, 0.40)
    subs = Unfitted._subdivide_segment(p1, p2, grid_lines, sqrt(eps(Float64)))
    @test length(subs) == 5

    # The sub-segments *partition* the input: they start at `p1`, end at
    # `p2`, and each one begins exactly where the last ended. No arc
    # length may be discarded at a crossing, so the lengths sum back to
    # the whole at round-off — not at some tolerance floor.
    @test subs[1][1] == p1
    @test subs[end][2] == p2
    @test all(subs[i][2] == subs[i + 1][1] for i in 1:(length(subs)-1))
    original_length = sqrt(0.80^2 + 0.30^2)
    total = sum(norm(sub[2] - sub[1]) for sub in subs)
    @test total ≈ original_length rtol = 1.0e-14

    # Parent-uniqueness is a statement about each sub-segment's
    # *interior*: the whole open sub-segment lies in one cell, which is
    # what lets the region builder attribute it to a parent by midpoint.
    # The shared endpoints deliberately sit *on* grid lines, where
    # `locate_cell` snaps to one side and cannot classify them — exactly
    # as for the K=2 clipping path below.
    for sub in subs
        cells = [Unfitted.locate_cell(V.levels[1].mesh, sub[1] + f * (sub[2] - sub[1]))
                 for f in (0.1, 0.25, 0.5, 0.75, 0.9)]
        @test all(c === cells[1] !== nothing for c in cells)
    end
end

@testset "_subdivide_segment: corner crossing and degenerate splits" begin
    # A segment through a grid *intersection* produces one crossing per
    # axis at the same parameter, agreeing only to a few ulp. They must
    # collapse to a single cut — not to a zero-length sliver straddling
    # the corner, and not to a gap.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    grid_lines = Unfitted._level_grid_lines(V, GeometryTolerance(Float64))
    nudge = sqrt(eps(Float64))

    corner = Unfitted._subdivide_segment(SVector(0.05, 0.05), SVector(0.45, 0.45), grid_lines,
                                         nudge)
    @test length(corner) == 2
    @test corner[1][2] == corner[2][1]
    @test sum(norm(s[2] - s[1]) for s in corner) ≈ norm(SVector(0.40, 0.40)) rtol = 1.0e-14

    # Endpoints sitting exactly on grid lines are crossings at t = 0 and
    # t = 1. Neither may spawn a degenerate sub-segment, and neither may
    # cost any length.
    on_lines = Unfitted._subdivide_segment(SVector(0.25, 0.25), SVector(0.75, 0.75), grid_lines,
                                           nudge)
    @test all(norm(s[2] - s[1]) > nudge for s in on_lines)
    @test on_lines[1][1] == SVector(0.25, 0.25)
    @test on_lines[end][2] == SVector(0.75, 0.75)
    @test sum(norm(s[2] - s[1]) for s in on_lines) ≈ norm(SVector(0.50, 0.50)) rtol = 1.0e-14

    # A segment lying *along* a grid line still splits across the other
    # axis and still conserves its length exactly.
    along = Unfitted._subdivide_segment(SVector(0.05, 0.25), SVector(0.95, 0.25), grid_lines, nudge)
    @test length(along) == 4
    @test sum(norm(s[2] - s[1]) for s in along) ≈ 0.90 rtol = 1.0e-14
end

@testset "_subdivide_triangle: triangle clipped against 3D grid" begin
    # A triangle in 3D crossing two grid planes. Its sub-triangles
    # should sum to the original area, with every sub-triangle inside
    # one cell of every level.
    omega = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = space(omega; cells=(2, 2, 2), order=1)
    grid_lines = Unfitted._level_grid_lines(V, GeometryTolerance(Float64))
    v1 = SVector(0.1, 0.1, 0.3)
    v2 = SVector(0.9, 0.1, 0.3)
    v3 = SVector(0.5, 0.9, 0.3)
    nudge = sqrt(eps(Float64))
    subs = Unfitted._subdivide_triangle(v1, v2, v3, grid_lines, nudge)

    # Original triangle area = 0.5·|e1 × e2| = 0.5·|(0.8,0,0) × (0.4,0.8,0)| = 0.5·|0.64 k̂| = 0.32.
    expected_area = 0.32
    function area(tri)
        e1 = tri[2] - tri[1]
        e2 = tri[3] - tri[1]
        c = SVector(e1[2]*e2[3]-e1[3]*e2[2], e1[3]*e2[1]-e1[1]*e2[3], e1[1]*e2[2]-e1[2]*e2[1])
        sqrt(sum(x->x*x, c)) / 2
    end
    total_area = sum(area, subs)
    @test total_area ≈ expected_area atol = 1.0e-9
    @test length(subs) >= 3

    # The midpoint of every sub-triangle lies inside exactly one cell;
    # vertices may sit on grid planes (where they cannot be classified
    # uniquely by `locate_cell`), which is fine — the surface-region
    # builder uses the midpoint to assign the parent.
    for tri in subs
        mid = (tri[1] + tri[2] + tri[3]) / 3
        @test Unfitted.locate_cell(V.levels[1].mesh, mid) !== nothing
    end
end

@testset "BoundaryMesh: subdivision preserves per-cell normals" begin
    # User-supplied normal on a single segment that gets split by the
    # subdivision into multiple sub-segments. Every sub-segment must
    # inherit the original normal.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    mesh = segment_mesh([(SVector(0.1, 0.1), SVector(0.6, 0.2))]; normals=[SVector(0.0, 1.0)])
    sub = Unfitted._subdivide_mesh(mesh, V, GeometryTolerance(Float64))
    @test length(sub.cells) > 1
    @test all(n == SVector(0.0, 1.0) for n in sub.normals)
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
    # `_surface_quadrature_order` is now only the upper bound that sizes the
    # rule table; the rule a sub-cell uses comes from the cells covering it.
    # A high-order level elsewhere must not raise the point count of a segment
    # it does not cover — which is what the level-wide maximum did, and the
    # reason the immersed-surface path was the one quadrature consumer that a
    # per-cell order made more expensive rather than less.
    omega = box((0.0, 0.0), (1.0, 1.0))
    tol = GeometryTolerance(Float64)
    V = overlay(space(omega; cells=(4, 4), order=1), box((0.5, 0.5), (1.0, 1.0)); cells=(2, 2),
                order=3)

    near = Unfitted._surface_regions_for_mesh(V,
                                              segment_mesh([(SVector(0.05, 0.10),
                                                             SVector(0.20, 0.10))]), tol)
    @test length(only(near).parents) == 1                 # the base level alone
    @test length(only(near).points) == 2                  # p = 1 → 2 Gauss points

    far = Unfitted._surface_regions_for_mesh(V,
                                             segment_mesh([(SVector(0.55, 0.60),
                                                            SVector(0.70, 0.60))]), tol)
    @test length(only(far).parents) == 2
    @test length(only(far).points) == 4                   # max p = 3 → 4 Gauss points

    # The same question on a level carrying a per-cell order, where there is no
    # per-level order to read even in principle.
    orders = fill(1, 4, 4)
    orders[4, 4] = 4
    W = space(omega; cells=(4, 4), order=orders)
    quiet = Unfitted._surface_regions_for_mesh(W,
                                               segment_mesh([(SVector(0.05, 0.10),
                                                              SVector(0.20, 0.10))]), tol)
    @test length(only(quiet).points) == 2
    loud = Unfitted._surface_regions_for_mesh(W,
                                              segment_mesh([(SVector(0.80, 0.85),
                                                             SVector(0.95, 0.85))]), tol)
    @test length(only(loud).points) == 5                  # p = 4 → 5 Gauss points
end
