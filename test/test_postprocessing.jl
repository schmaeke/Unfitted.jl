using StaticArrays: SMatrix, SVector

@testset "1D solution evaluation and L2 error" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=1, order=2)
    model = prepare(poisson(V; source=2.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    solution = solve!(model)
    exact = x -> x[1] * (1 - x[1])

    @test value(solution, model, (0.3,)) ≈ exact((0.3,)) atol = 1.0e-12
    @test field_gradient(solution, model, (0.3,))[1] ≈ 0.4 atol = 1.0e-12
    @test l2_error(solution, model, exact; norm=:absolute) < 1.0e-12
    @test l2_error(solution, model, exact) < 1.0e-12
end

@testset "overlay zero extension and superposition evaluation" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=1, order=1)
    V = overlay(V, box((0.25,), (0.75,)); cells=1, order=2)
    model = prepare(poisson(V; source=0.0))

    coefficients = zeros(Unfitted.active_unknowns(model.dofs))
    base_dofs = Unfitted.active_cell_dofs(model.dofs, 1, CartesianIndex(1))
    overlay_dofs = Unfitted.active_cell_dofs(model.dofs, 2, CartesianIndex(1))
    coefficients[base_dofs[1]] = 1.0
    coefficients[base_dofs[2]] = 2.0
    coefficients[overlay_dofs[3]] = 10.0

    base_coefficients = copy(coefficients)
    base_coefficients[overlay_dofs[3]] = 0.0
    solution = Solution(coefficients, model.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))
    base_solution = Solution(base_coefficients, model.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test value(solution, model, (0.1,)) ≈ value(base_solution, model, (0.1,))

    overlay_delta = 10.0 * Unfitted.integrated_legendre_value(2, 0.0)
    @test value(solution, model, (0.5,)) - value(base_solution, model, (0.5,)) ≈ overlay_delta
    @test field_gradient(solution, model, (0.1,)) ≈ field_gradient(base_solution, model, (0.1,))
end

@testset "D-generic constant-field evaluation smoke test" begin
    omega = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = space(omega; cells=(1, 1, 1), order=1)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test value(solution, model, (0.2, 0.3, 0.4)) ≈ 1.0
    @test all(abs.(field_gradient(solution, model, (0.2, 0.3, 0.4))) .< 1.0e-12)
    @test l2_error(solution, model, x -> 1.0; norm=:absolute) < 1.0e-12
end

@testset "l2_error pins the integration measure absolutely" begin
    # ‖c‖_{L²(Ω)} = |c|·√|Ω| in closed form, so a constant field measured against a zero reference fixes the quadrature
    # measure by its absolute value. Every other l2_error assertion in the suite is `< tiny` or a ratio, and no positive
    # rescaling of the measure — a dropped 2⁻ᴰ box Jacobian, say — can violate one of those. The 2D/3D pair pins the
    # exponent of such a factor, not merely its presence. Order 1 is a partition of unity, so a uniform coefficient
    # vector is exactly the constant field, and `value` re-checks that at an off-grid point.

    # 2D: |Ω| = 2·2 = 4, c = 3 ⇒ ‖c‖ = 3·√4 = 6.
    omega = box((0.0, 0.0), (2.0, 2.0))
    model = prepare(poisson(space(omega; cells=(2, 3), order=1); source=0.0))
    solution = Solution(fill(3.0, active_unknowns(model)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test value(solution, model, (0.7, 1.3)) ≈ 3.0
    @test l2_error(solution, model, x -> 0.0; norm=:absolute) ≈ 6.0 rtol = 1.0e-14
    # A zero reference has no norm to divide by, so the relative form documents a fall back to the absolute one.
    @test l2_error(solution, model, x -> 0.0) ≈ 6.0 rtol = 1.0e-14

    # 3D: |Ω| = 3·3·1 = 9, c = 5 ⇒ ‖c‖ = 5·√9 = 15.
    cube = box((0.0, 0.0, 0.0), (3.0, 3.0, 1.0))
    cube_model = prepare(poisson(space(cube; cells=(2, 1, 2), order=1); source=0.0))
    cube_solution = Solution(fill(5.0, active_unknowns(cube_model)), cube_model.version,
                             Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test value(cube_solution, cube_model, (0.7, 1.3, 0.4)) ≈ 5.0
    @test l2_error(cube_solution, cube_model, x -> 0.0; norm=:absolute) ≈ 15.0 rtol = 1.0e-14
end

@testset "solution/model mismatch checks" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=1, order=1)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version + 1,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test_throws ArgumentError value(solution, model, (0.5,))
    @test_throws ArgumentError l2_error(solution, model, x -> 0.0)
    @test_throws ArgumentError value(Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                                              Unfitted.SolverDiagnostics(:manual, 0.0, true)), model, (1.5,))
end

@testset "a vector field evaluates every component at once, exactly as one at a time" begin
    # `value` and `field_gradient` evaluate a vector field's basis once per level for all its components, so the whole
    # vector must equal the per-component calls exactly. The points cover the overlay and the base level alone, inside
    # and outside the disk, each in a cut cell or a full one; the coefficients make every component and dof distinct.
    disk = physical_domain(x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.35; lipschitz=1.0, subcell_length_scale=0.125,
                           max_depth=3)
    base = space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=2, physical=disk)
    V = overlay(base, box((0.25, 0.25), (0.75, 0.75)); cells=(2, 2), order=3)
    u = field(:u, V; components=2)
    model = prepare(poisson(u; source=SVector(1.0, 2.0)))
    solution = Solution([sin(3.0 * i) for i in 1:active_unknowns(model)], model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    for x in ((0.1, 0.15), (0.5, 0.5), (0.4, 0.62), (0.83, 0.5), SVector(0.3, 0.7))
        v = @inferred value(solution, model, u, x)
        g = @inferred field_gradient(solution, model, u, x)
        @test v isa SVector{2,Float64} && g isa SVector{2,SVector{2,Float64}}
        @test v == SVector(value(solution, model, u, x, 1), value(solution, model, u, x, 2))
        @test g == SVector(field_gradient(solution, model, u, x, 1), field_gradient(solution, model, u, x, 2))
        # `component = nothing` is the default spelled out, with and without the field.
        @test value(solution, model, u, x, nothing) == value(solution, model, x, nothing) == v
        @test field_gradient(solution, model, u, x, nothing) == field_gradient(solution, model, x, nothing) == g
    end

    # A component outside `1:2`, and a field of the model's name with another component count, are refused.
    @test_throws ArgumentError value(solution, model, u, (0.5, 0.5), 3)
    @test_throws ArgumentError field_gradient(solution, model, (0.5, 0.5), 0)
    @test_throws ArgumentError value(solution, model, field(:u, V), (0.5, 0.5))
end

@testset "active_unknowns(model) public accessor" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=2)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=3)
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    @test active_unknowns(model) isa Int
    @test active_unknowns(model) == Unfitted.active_unknowns(model.dofs)
    @test active_unknowns(model) == diagnostics(model).active_unknowns
end

@testset "cell iteration helpers (cell_indices, cell_box, center)" begin
    domain = box((-1.0, -1.0), (1.0, 1.0))
    m = Unfitted.mesh(domain; cells=(4, 2))

    @test cell_indices(m) == CartesianIndices((4, 2))

    # Each cell box's center sits on the midpoint of its axis-aligned grid interval; collecting them should reproduce
    # the tensor product of per-axis interval midpoints (−0.75, −0.25, 0.25, 0.75) × (−0.5, 0.5).
    centers = [center(cell_box(m, ci)) for ci in cell_indices(m)]
    @test centers[CartesianIndex(1, 1)] ≈ [-0.75, -0.5]
    @test centers[CartesianIndex(4, 2)] ≈ [0.75, 0.5]

    # cell_box agrees with the mesh's per-axis boundary coordinates.
    cb = cell_box(m, CartesianIndex(2, 1))
    @test cb.lower ≈ [-0.5, -1.0]
    @test cb.upper ≈ [0.0, 0.0]

    @test_throws BoundsError cell_box(m, CartesianIndex(5, 1))
end

@testset "boundary_integral: perimeter and face length on unit cube" begin
    # 2D unit square: ∫_∂Ω 1 ds == 4, per-face == 1, corner point == 1.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(3, 3), order=2)
    model = prepare(poisson(V; source=0.0))

    @test boundary_integral(q -> 1.0, model; on=boundary(:all)) ≈ 4.0
    @test boundary_integral(q -> 1.0, model; on=boundary(axis=1, side=:lower)) ≈ 1.0
    @test boundary_integral(q -> 1.0, model; on=boundary(axis=2, side=:upper)) ≈ 1.0
    # codim-2 point
    @test boundary_integral(q -> 1.0, model; on=boundary((axis=1, side=:lower), (axis=2, side=:lower))) ≈ 1.0

    # 3D unit cube: ∫_∂Ω 1 ds == 6 (six unit faces), per-face == 1.
    cube = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    Vc = space(cube; cells=(2, 2, 2), order=1)
    cube_model = prepare(poisson(Vc; source=0.0))

    @test boundary_integral(q -> 1.0, cube_model; on=boundary(:all)) ≈ 6.0
    @test boundary_integral(q -> 1.0, cube_model; on=boundary(axis=3, side=:upper)) ≈ 1.0
end

@testset "boundary_integral: analytic trace on a solved Laplace problem" begin
    # Solve −Δu = 0 on (0,1)² with the manufactured u(x,y) = x y. The space resolves this exactly at p = 1, so the trace
    # on y = 1 is u(x,1) = x and ∫₀¹ x dx = 1/2, while the gradient on x = 1 is ∇u = (y, x) so ∫₀¹ ∂u/∂n ds = ∫₀¹ y dy =
    # 1/2.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=2)
    exact_value(x) = x[1] * x[2]
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(exact_value; on=boundary(:all))]))
    sol = solve!(model)

    top_trace = boundary_integral(q -> value(sol, model, q.x), model; on=boundary(axis=2, side=:upper))
    @test top_trace ≈ 0.5 atol = 1.0e-10

    right_flux = boundary_integral(model; on=boundary(axis=1, side=:upper)) do q
        return field_gradient(sol, model, q.x)[1]
    end
    @test right_flux ≈ 0.5 atol = 1.0e-10
end

@testset "boundary_integral: overlay coverage on a shared face" begin
    # An overlay whose own face coincides with the physical face should contribute to the boundary integral; the
    # integral of `1` over that face must still equal its physical length (the helper does not double-count overlapping
    # parents — it sums Jacobian × Gauss weights once per admissible region).
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    V = overlay(V, box((0.0, 0.0), (0.5, 0.5)); cells=(1, 1), order=2)
    model = prepare(poisson(V; source=0.0))

    @test boundary_integral(q -> 1.0, model; on=boundary(axis=2, side=:lower)) ≈ 1.0
    @test boundary_integral(q -> 1.0, model; on=boundary(:all)) ≈ 4.0
end

@testset "boundary_integral: q.normal carries the outward facet normal" begin
    # Codim-1 faces: outward normal should be ±eₐ matching the side.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    model = prepare(poisson(V; source=0.0))

    # Capture the normal observed at one quadrature point per face.
    function first_normal(selector)
        captured = Ref{Any}(nothing)
        boundary_integral(model; on=selector) do q
            captured[] === nothing && (captured[] = q.normal)
            return 0.0
        end
        return captured[]
    end

    @test first_normal(boundary(axis=1, side=:lower)) ≈ [-1.0, 0.0]
    @test first_normal(boundary(axis=1, side=:upper)) ≈ [1.0, 0.0]
    @test first_normal(boundary(axis=2, side=:lower)) ≈ [0.0, -1.0]
    @test first_normal(boundary(axis=2, side=:upper)) ≈ [0.0, 1.0]

    # Codim-2 corner (vertex in 2D): unit average of the constrained face normals; well-defined and finite.
    n_corner = first_normal(boundary((axis=1, side=:lower), (axis=2, side=:lower)))
    @test n_corner ≈ [-1.0, -1.0] / sqrt(2.0)
end

@testset "boundary_integral on a coupled model needs a field and honours it" begin
    # Two disjoint unit squares. Each has perimeter 4, so ∂Ω of the pair is 8. Without `field=`, `boundary_integral`
    # used to resolve `boundary(:all)` to whichever subdomain a form referenced first and return 4.0 — a plausible
    # number for the wrong domain, with nothing to signal the omission.
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(3, 3), order=1)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2)),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=:u1),
                                       dirichlet(0.0; on=boundary(:all), field=:u2)]))

    # Naming the subdomain gives that subdomain's boundary — including the second one, whose facets the shared cache
    # entry never held.
    @test boundary_integral(q -> 1.0, model; on=boundary(:all), field=:u1) ≈ 4.0
    @test boundary_integral(q -> 1.0, model; on=boundary(:all), field=:u2) ≈ 4.0

    # A face of each: subdomain 2's upper-x face is at x = 3, not x = 1.
    right1 = boundary_integral(model; on=boundary(axis=1, side=:upper), field=:u1) do q
        q.x[1]
    end
    right2 = boundary_integral(model; on=boundary(axis=1, side=:upper), field=:u2) do q
        q.x[1]
    end
    @test right1 ≈ 1.0
    @test right2 ≈ 3.0

    # Omitting `field` raises rather than silently answering for one subdomain.
    @test_throws ArgumentError boundary_integral(q -> 1.0, model; on=boundary(:all))
    err = try
        boundary_integral(q -> 1.0, model; on=boundary(:all))
    catch e
        e
    end
    @test occursin("field=", err.msg)
    @test occursin(":u1", err.msg) && occursin(":u2", err.msg)

    # An unknown field name is still an error, not a silent fallback.
    @test_throws ArgumentError boundary_integral(q -> 1.0, model; on=boundary(:all), field=:nope)

    # Single-domain models are unaffected: `field` is optional and, when given, agrees with the default exactly.
    single = prepare(poisson(V1; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test boundary_integral(q -> 1.0, single; on=boundary(:all)) ==
          boundary_integral(q -> 1.0, single; on=boundary(:all), field=:u)
end

@testset "l2_error takes a field and measures that subdomain only" begin
    # Two disjoint squares carrying different L² projections. Without a field argument `l2_error` could only reach
    # `_default_field`, which raises on any multi-field problem — so the package's own coupled workflow had no L² error
    # report at all.
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(3, 3), order=1)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    # Both data are bilinear, hence exactly representable in the order-1 tensor space, so each projection reproduces its
    # own datum to round-off.
    f1 = x -> x[1]
    f2 = x -> 2 * x[2]
    model = prepare(Problem((u1, u2); blocks=(mass_block(u1), mass_block(u2)),
                            loads=(source_load(u1; source=f1), source_load(u2; source=f2))))
    sol = solve!(model)

    @test l2_error(sol, model, u1, f1; norm=:absolute) < 1.0e-10
    @test l2_error(sol, model, u2, f2; norm=:absolute) < 1.0e-10

    # Swapping the datum is a large error, so the field argument really selects which subdomain is integrated rather
    # than being ignored.
    @test l2_error(sol, model, u1, f2; norm=:absolute) > 0.1
    @test l2_error(sol, model, u2, f1; norm=:absolute) > 0.1

    # Omitting the field still raises on a multi-field model, and an unknown field name is an error rather than a silent
    # fallback.
    @test_throws ArgumentError l2_error(sol, model, f1)
    @test_throws ArgumentError l2_error(sol, model, field(:nope, V1), f1)
    @test_throws ArgumentError l2_error(sol, model, u1, f1; norm=:bogus)

    # Single-domain models are unaffected: the field is optional and, when given, produces the identical number.
    V = space(box((0.0,), (1.0,)); cells=1, order=2)
    single = prepare(poisson(V; source=2.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    u = single.problem.fields[1]
    ss = solve!(single)
    exact = x -> x[1] * (1 - x[1])
    @test l2_error(ss, single, exact) === l2_error(ss, single, u, exact)
    @test l2_error(ss, single, exact; norm=:absolute) === l2_error(ss, single, u, exact; norm=:absolute)
end

@testset "model.regions caches one entry per Dirichlet selector" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)

    # No Dirichlet → empty cache, zero facet-region count.
    plain = prepare(poisson(V; source=0.0))
    @test isempty(plain.regions)
    @test diagnostics(plain).facet_region_count == 0
    @test nquadpoints(plain; kind=:facet) == 0
    @test nquadpoints(plain) == nquadpoints(plain; kind=:volume)  # default unchanged

    # boundary(:all) → one cached selector. Region count and q-point count match what boundary_integral sees on the same
    # model.
    with_dirichlet = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test length(with_dirichlet.regions) == 1
    selector_count = sum(length, values(with_dirichlet.regions))
    @test diagnostics(with_dirichlet).facet_region_count == selector_count
    @test nquadpoints(with_dirichlet; kind=:facet) > 0

    # Distinct selectors are cached separately.
    multi = prepare(poisson(V; source=0.0,
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                                       dirichlet(0.0; on=boundary(axis=2, side=:upper))]))
    @test length(multi.regions) == 2

    # nquadpoints rejects unknown kinds with a clear message.
    @test_throws ArgumentError nquadpoints(plain; kind=:nope)
end

@testset "the facet regions refresh on move! / activate! / deactivate!" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=1)
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    before_count = diagnostics(model).facet_region_count
    move!(model; level=2, to=box((0.3, 0.3), (0.8, 0.8)))
    # The overlay box did not coincide with the physical boundary before or after the move, so the cached facet region
    # list must still be populated and consistent with the post-move space.
    @test !isempty(model.regions)
    @test diagnostics(model).facet_region_count == before_count
    # The facet-region cache contents must now reflect the new space: at least one parent in each region's list must be
    # from level 1 (the base — overlays inside Ω cannot touch ∂Ω).
    for (_, list) in model.regions
        for region in list
            @test all(parent.level == 1 for parent in region.parents)
        end
    end
end

@testset "boundary_integral: empty selection raises" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    # Mask out both cells of the bottom row of the base level (those whose y-index is 1) so no covering parent remains
    # on the y = 0 face. The base is the only level that touches the lower face, so the selector finds no admissible
    # region and must raise.
    mask = trues(2, 2)
    mask[:, 1] .= false   # cell (·, y=1) → inactive on the lower row
    V = space(omega; cells=(2, 2), order=1, active=BitArray(mask))
    model = prepare(poisson(V; source=0.0))

    @test_throws ArgumentError boundary_integral(q -> 1.0, model; on=boundary(axis=2, side=:lower))

    # The upper face is still fully covered, so a valid selector on the same model still works — confirming the throw
    # above is from missing facet coverage, not from a wrong-axis check upstream.
    @test boundary_integral(q -> 1.0, model; on=boundary(axis=2, side=:upper)) ≈ 1.0
end

# ── Sampling records ──────────────────────────────────────────────────────────
#
# What every post-processing consumer reads: the records `src/postprocessing.jl` samples (`Unfitted.SampleGrid`, whose
# contract heads that file), pinned as records rather than through a file a consumer wrote — which cells are sampled,
# which arrays they carry and in which element types, and what the sampling refuses. A consumer then pins only its own
# mapping of the records: `test_vtk.jl` the VTK writer's, reusing the fixtures defined here.

# The array `name` of a record, whether a point or a cell array.
_record_array(g, name) = last(only(filter(entry -> first(entry) == name, [g.point_data; g.cell_data])))

# The extent along axis `d` of every cell of a grid of cubes, read off the cell's own points: one column of 2ᴷ corner
# coordinates per cell. The `cut_depth` refinement is uniform per tree level, so a cell's width identifies the level it
# came from.
function _cell_extents(g::Unfitted.SampleGrid{D,T,K}, d::Int=1) where {D,T,K}
    corners = reshape([x[d] for x in g.points], 2^K, :)
    return vec(maximum(corners; dims=1) - minimum(corners; dims=1))
end

# The cells of a 2D grid that straddle ∂Ω, judged by the sign of the `level_set` array at their corners — exactly the
# test ParaView's Clip applies when it decides whether to cut a cell.
function _mixed_cells(g)
    phi = _record_array(g, "level_set")
    return [c for c in 1:Unfitted._cell_count(g) if !allequal(phi[k] <= 0 for k in (4c-3):(4c))]
end

# A record's geometry and arrays: the whole of what a consumer reads from it.
_record_contents(g) = (g.points, g.point_data, g.cell_data)

# A disk cutting every cell of a 2×2 mesh: four regions, all `:cut_fitted`.
function _cut_depth_disk_model()
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    omega = box((0.0, 0.0), (1.0, 1.0))
    physical = physical_domain(phi_disk; lipschitz=1.0, alpha=1.0, subcell_length_scale=0.125, max_depth=3)
    V = space(omega; cells=(2, 2), order=1, physical=physical)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))
    return phi_disk, model, solution
end

# A mass model on `V` and a solution with distinct coefficients, so that no two dofs carry the same value.
function _pin_model(V)
    model = prepare(mass(V))
    return (; V, model, sol=solution(model, [sin(i) for i in 1:active_unknowns(model)]))
end

# Two cells of per-cell anisotropic orders (1, 2) and (3, 1), with an order-(1, 3) overlay over the right one, so the
# right region has two parents and `:degree` takes the per-axis maximum (3, 3) over them.
_pin_2d = _pin_model(overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 1), order=reshape([(1, 2), (3, 1)], 2, 1)),
                             box((0.5, 0.0), (1.0, 1.0)); cells=1, order=(1, 3)))

@testset "records: a sampled field's data grid" begin
    phi_disk, model, sol = _cut_depth_disk_model()
    width = (u, c, x, xi) -> c.cell.upper[1] - c.cell.lower[1]
    (; field, data) = only(Unfitted._sample_fields(sol, model; subdivisions=2, cut_depth=1, cell_data=(width=width,)))
    n = Unfitted._cell_count(data)
    @test n > 16                                             # four regions, a 2 × 2 lattice each, and the cut leaves
    @test length(data.points) == n << 2
    # Every cell's corners are the face-cyclic permutation (1, 2, 4, 3) of the `_box_vertices` of the box they span.
    @test all(1:n) do c
        corners = data.points[(4c-3):(4c)]
        return corners == collect(Unfitted._box_vertices(corners[1], corners[3]))[[1, 2, 4, 3]]
    end
    @test first.(data.point_data) == ["uh", "level_set"]
    @test first.(data.cell_data) == ["region_id", "cover_count", "width"]
    @test all(length(last(entry)) == length(data.points) for entry in data.point_data)
    @test all(length(last(entry)) == n for entry in data.cell_data)
    @test _record_array(data, "level_set") == [levelset_value(field.space.physical, x) for x in data.points]
    @test issorted(_record_array(data, "region_id"))
    @test _record_array(data, "width") == [data.points[4c - 2][1] - data.points[4c - 3][1] for c in 1:n]
end

@testset "records: a level's mesh" begin
    # One record per level, filed under the field whose space it belongs to, covering every cell of the level with the
    # eight integer cell arrays and no point array.
    _, model, _ = _cut_depth_disk_model()
    level = only(Unfitted._sample_levels(model))
    @test (level.field.name, level.level, level.role) == (only(model.problem.fields).name, 1, :base)
    @test Unfitted._cell_count(level.grid) == 4
    @test isempty(level.grid.point_data)
    @test first.(level.grid.cell_data) ==
          ["level_id", "role_id", "order_max", "cell_id", "active", "covered", "active_dofs", "reduced_dofs"]
    @test all(entry -> last(entry) isa Vector{Int} && length(last(entry)) == 4, level.grid.cell_data)
end

@testset "records: fields sharing a space carry its level meshes once" begin
    # Two fields on one space, the Dirichlet condition on `a` alone: the space's one level is meshed once, under the
    # first sampled field, and its per-cell dof counts are that field's — on `a` the boundary vertices are constrained
    # and each cell keeps the one interior vertex, on `b` every vertex is active.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    a = field(:a, V)
    b = field(:b, V)
    model = prepare(Problem((a, b); blocks=(mass_block(a), mass_block(b)),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=:a)]))
    sol = solution(model, [sin(i) for i in 1:active_unknowns(model)])
    records = Unfitted._sample_fields(sol, model; subdivisions=:none, point_data=NamedTuple())
    @test [r.field.name for r in records] == [:a, :b]
    @test all(r -> isempty(r.data.point_data), records)
    meshes = Unfitted._sample_levels(model)
    @test [(m.field.name, m.level) for m in meshes] == [(:a, 1)]
    @test _record_array(only(meshes).grid, "active_dofs") == fill(1, 4)
    picked = only(Unfitted._sample_levels(model; field=b))
    @test (picked.field.name, picked.level) == (:b, 1)
    @test _record_array(picked.grid, "active_dofs") == fill(4, 4)
end

@testset "records: the quadrature cloud weighs each signature's regions" begin
    # One vertex cloud per sorted covering-level signature, in sorted order. Its weights are the physical quadrature
    # weights, so they sum to the measure of that signature's regions — the left region of the two-cell mesh is covered
    # by the base level alone, the right one by the base and the overlay.
    (; model) = _pin_2d
    cloud = Unfitted._sample_quadrature(model)
    @test [entry.signature for entry in cloud] == [[1], [1, 2]]
    regions = Unfitted.integration_plans(model)[1].regions
    for (; signature, grid) in cloud
        @test Unfitted._cell_count(grid) == length(grid.points)
        @test first.(grid.point_data) == ["weight"] && isempty(grid.cell_data)
        measure = sum(Unfitted._region_measure(r) for r in regions if sort([p.level for p in r.parents]) == signature)
        @test sum(last(only(grid.point_data))) ≈ measure rtol = 1.0e-13
    end
    @test sum(entry -> length(entry.grid.points), cloud) == nquadpoints(model)
end

@testset "records: a discretisation is sampled without a solution" begin
    # An unsolved model has its regions, their sample cells, its level meshes and its quadrature cloud; only the values
    # of a solution are missing. Sampled without one, the data grid is the solved one less the default `uh` entry, a
    # callback that reads only its arguments still runs, and one that evaluates the solution is refused naming the
    # remedy rather than handed zeros.
    _, model, sol = _cut_depth_disk_model()
    keywords = (subdivisions=2, cut_depth=1)
    solved = only(Unfitted._sample_fields(sol, model; keywords...)).data
    bare = only(Unfitted._sample_fields(nothing, model; keywords...)).data
    @test first.(bare.point_data) == ["level_set"]
    @test _record_contents(bare) == (solved.points, solved.point_data[2:end], solved.cell_data)
    plain = only(Unfitted._sample_fields(nothing, model; point_data=(x1=(u, c, x, xi) -> x[1],))).data
    @test _record_array(plain, "x1") == first.(plain.points)
    err = try
        Unfitted._sample_fields(nothing, model; point_data=(uh=(u, c, x, xi) -> u(c, xi),))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("pass the solution", err === nothing ? "" : sprint(showerror, err))
    @test [(m.level, m.role) for m in Unfitted._sample_levels(model)] == [(1, :base)]
end

@testset "a sampling callback reads only fields of the sampled field's own subdomain" begin
    # The sample accessor evaluates fields on the parents of the region being sampled, which are cells of the sampled
    # field's own subdomain. A field of the other subdomain of a coupled model owns none of them, so asking for it is a
    # usage error with a pointer to the point evaluation, not a wrong number or a crash deep in the dof layer.
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(2, 2), order=1)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2)),
                            loads=(source_load(u1; source=1.0), source_load(u2; source=1.0)),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=:u1),
                                       dirichlet(0.0; on=boundary(:all), field=:u2)]))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))
    foreign = (other=(u, c, x, xi) -> u(c, xi, u2),)
    err = try
        Unfitted._sample_fields(solution, model; field=:u1, subdivisions=:none, point_data=foreign)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    message = sprint(showerror, err)
    @test occursin("value(solution, model, u2, x)", message)
    # The refusal names the sampled field's subdomain, not a consumer of the records or its blocks.
    @test occursin("sampled field's own subdomain", message) && !occursin("write_vtk", message)
end

@testset "a sampling callback reads another field of the same subdomain" begin
    # Two fields on one space: sampling `a`, a callback may evaluate `b` at the sample, and gets the value the point
    # evaluation gives at the same `x`.
    V = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=2), box((0.0, 0.0), (0.5, 0.5)); cells=(2, 2),
                order=3)
    a = field(:a, V)
    b = field(:b, V)
    model = prepare(Problem((a, b); blocks=(stiffness_block(a), mass_block(b)), loads=(source_load(a; source=1.0),),
                            dirichlet=[dirichlet(x -> 1.0 + x[1]; on=boundary(:all), field=:a)]))
    n = Unfitted.active_unknowns(model.dofs)
    solution = Solution([sin(3i) for i in 1:n], model.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))
    point_data = (own=(u, c, x, xi) -> u(c, xi) - value(solution, model, a, x),
                  other=(u, c, x, xi) -> u(c, xi, b) - value(solution, model, b, x))
    data = only(Unfitted._sample_fields(solution, model; field=:a, subdivisions=2, point_data)).data
    @test length(data.points) > 50
    @test maximum(abs, _record_array(data, "own")) ≤ 1.0e-12
    @test maximum(abs, _record_array(data, "other")) ≤ 1.0e-12
end

@testset "a sampling callback reads a field only with the model's component count" begin
    # The accessor reads a field's component count off the `Field` it is given. A `Field` that shares a model field's
    # name but not its count is not that field: read with its own count it would return part of the model's field, or
    # that field followed by the next field's values. It is refused, as `value` refuses it; the model's own fields read
    # their values, a vector field's included.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=1)
    a = field(:a, V)
    w = field(:w, V; components=2)
    model = prepare(Problem((a, w); blocks=(mass_block(a), mass_block(w))))
    n = Unfitted.active_unknowns(model.dofs)
    solution = Solution([10.0i for i in 1:n], model.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))
    for impostor in (field(:a, V; components=2), field(:w, V))
        err = try
            Unfitted._sample_fields(solution, model; field=:a, subdivisions=:none,
                                    point_data=(read=(u, c, x, xi) -> u(c, xi, impostor),))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("field $(impostor.name) has", sprint(showerror, err))
        @test_throws ArgumentError value(solution, model, impostor, (0.5, 0.5))
    end
    point_data = (a=(u, c, x, xi) -> maximum(abs, u(c, xi, a) - value(solution, model, a, x)),
                  w=(u, c, x, xi) -> maximum(abs, u(c, xi, w) - value(solution, model, w, x)))
    data = only(Unfitted._sample_fields(solution, model; field=:a, subdivisions=:none, point_data)).data
    @test length(data.points) == 4
    @test maximum(_record_array(data, "a")) ≤ 1.0e-12
    @test maximum(_record_array(data, "w")) ≤ 1.0e-12
end

@testset "a level mesh holds every cell of its level, active or not" begin
    # A 3×3 overlay with only one active cell: its mesh holds all nine cells, told apart by the `active` cell array — 1
    # for the active cell, 0 for the eight masked ones — and the single base cell is a mesh of its own. Nothing lies
    # above the overlay, so none of its cells is covered.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    active = falses(3, 3)
    active[2, 2] = true
    V = overlay(V, box((0.1, 0.1), (0.9, 0.9)); cells=(3, 3), order=1, active=active)
    model = prepare(poisson(V; source=0.0))

    base, top = Unfitted._sample_levels(model)
    @test (base.level, base.role, Unfitted._cell_count(base.grid)) == (1, :base, 1)
    @test (top.level, top.role, Unfitted._cell_count(top.grid)) == (2, :overlay, 9)
    @test _record_array(base.grid, "active") == [1]
    @test _record_array(top.grid, "active") == Int.(vec(active))
    @test _record_array(top.grid, "covered") == zeros(Int, 9)
end

@testset "a level mesh reports coverage and reduction per cell" begin
    # A 4×4 p=2 base with an aligned p=3 overlay over the middle 2×2 block covers four base cells and sheds high-order
    # modes (plus one deduped vertex) there; the base is unmasked, so every one of its cells is active.
    V = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2), box((0.25, 0.25), (0.75, 0.75));
                cells=(4, 4), order=3)
    model = prepare(mass(V))
    base = first(Unfitted._sample_levels(model))
    @test base.role === :base
    covered = _record_array(base.grid, "covered")
    reduced = _record_array(base.grid, "reduced_dofs")
    active = _record_array(base.grid, "active")

    @test length(covered) == 16                    # base is 4×4
    @test sum(covered) == 4                         # the middle 2×2 block is covered
    @test all(active .== 1)                         # unmasked base → every cell active
    @test all(reduced[covered .== 0] .== 0)         # reduction only under coverage
    @test sum(reduced[covered .== 1]) > 0           # covered cells shed high-order modes
end

@testset "the data grid carries level_set with a PhysicalDomain" begin
    # A circular cut of the unit square: cells crossing r = 0.4 are cut, cells fully inside are full, cells fully
    # outside (with α = 1 to keep them in the dof layout) are fictitious. The `level_set` array is `φ(x)` at every
    # point, so a contour at level 0 reproduces ∂Ω.
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    omega = box((0.0, 0.0), (1.0, 1.0))
    physical = physical_domain(phi_disk; lipschitz=1.0, alpha=1.0, subcell_length_scale=0.125, max_depth=3)
    V = space(omega; cells=(2, 2), order=1, physical=physical)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    data = only(Unfitted._sample_fields(solution, model; subdivisions=:none)).data
    @test first.(data.point_data) == ["uh", "level_set"]
    @test _record_array(data, "level_set") ≈ phi_disk.(data.points) atol = 1.0e-14
end

@testset "the data grid carries no level_set without a PhysicalDomain" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=1)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    data = only(Unfitted._sample_fields(solution, model; subdivisions=:none)).data
    @test first.(data.point_data) == ["uh"]
end

@testset "a user-supplied level_set entry replaces the built-in one" begin
    # A `level_set` entry in `point_data` wins: the built-in is not added, and the entry's array — a constant that
    # cannot be φ — is the one the record carries, in the entry's place.
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    omega = box((0.0, 0.0), (1.0, 1.0))
    physical = physical_domain(phi_disk; lipschitz=1.0, alpha=1.0, subcell_length_scale=0.25, max_depth=2)
    V = space(omega; cells=(1, 1), order=1, physical=physical)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    point_data = (level_set=(u, c, x, xi) -> 42.0, uh=(u, c, x, xi) -> u(c, xi))
    data = only(Unfitted._sample_fields(solution, model; subdivisions=:none, point_data)).data
    @test first.(data.point_data) == ["level_set", "uh"]
    @test all(==(42.0), _record_array(data, "level_set"))
end

@testset "cut_depth bisects only the sub-cells ∂Ω passes through" begin
    # Four `:cut_fitted` regions of width 0.5. Each level of `cut_depth` bisects only the marked boxes, so the cell
    # count grows far slower than the 4ᵏ of a uniform 2D refinement (4, 16, 64, 256) once boxes start falling away from
    # the boundary.
    _, model, solution = _cut_depth_disk_model()
    for (depth, cells) in ((0, 4), (1, 16), (2, 52), (3, 160))
        data = only(Unfitted._sample_fields(solution, model; subdivisions=:none, cut_depth=depth)).data
        widths = _cell_extents(data)
        @test length(widths) == cells

        # The crack invariant, stated executably: every cell a clip will cut sits at the finest level, so two of them
        # meeting at a face carry identical corner coordinates and their clip facets agree exactly. Coarser cells
        # survive only where φ has one sign, and those contribute no cut surface at all — which is why no 2:1 balance
        # pass is needed.
        finest = 0.5 / 2^depth
        mixed = _mixed_cells(data)
        @test !isempty(mixed)
        @test all(w -> w ≈ finest, widths[mixed])
        # Every width is a power-of-two fraction of the 0.5 base box, and the finest level is always reached. The
        # coarsest need not survive: this disk crosses all four regions, so at depth ≥ 1 every base box is marked and
        # nothing stays at 0.5.
        @test all(w -> any(l -> w ≈ 0.5 / 2^l, 0:depth), widths)
        @test minimum(widths) ≈ finest
    end
end

@testset "cut_depth keeps level_set exact on the refined path" begin
    # The refinement must not interpolate φ: every point, including the ones the tree introduced, carries
    # `levelset_value` evaluated there.
    _, model, solution = _cut_depth_disk_model()
    (; field, data) = only(Unfitted._sample_fields(solution, model; subdivisions=:none, cut_depth=2))
    @test _record_array(data, "level_set") == [levelset_value(field.space.physical, x) for x in data.points]
end

@testset "cut_depth hands cell callbacks the refined leaf" begin
    # `context.cell` must be the box actually emitted, not the `subdivisions` box it was carved from — the subtlest
    # plumbing error the refinement can introduce, and invisible in the geometry.
    _, model, solution = _cut_depth_disk_model()
    data = only(Unfitted._sample_fields(solution, model; subdivisions=:none, cut_depth=2,
                                        cell_data=(w=(u, c, x, xi) -> c.cell.upper[1] - c.cell.lower[1],))).data
    w = _record_array(data, "w")
    @test sort(unique(round.(w; digits=12))) == [0.125, 0.25]
    @test w ≈ _cell_extents(data)
end

@testset "cut_depth spares full and fictitious regions" begin
    # A disk small enough to leave whole cells inside and whole cells outside it. `keep_fictitious` holds the outside
    # cells in the layout — α alone does not — so all three region kinds are present at once and only the cut ones may
    # be refined.
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3
    omega = box((0.0, 0.0), (1.0, 1.0))
    physical = physical_domain(phi_disk; lipschitz=1.0, alpha=1.0, keep_fictitious=true, subcell_length_scale=1 / 32,
                               max_depth=4)
    V = space(omega; cells=(8, 8), order=1, physical=physical)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    plan = Unfitted.integration_plans(model)[1]
    kinds = [r.kind for r in plan.regions]
    @test :full in kinds
    @test :fictitious_alpha in kinds
    uncut = Set(i for (i, k) in pairs(kinds) if k === :full || k === :fictitious_alpha)
    @test !isempty(uncut)

    data = only(Unfitted._sample_fields(solution, model; subdivisions=:none, cut_depth=2)).data
    widths = _cell_extents(data)
    region_ids = _record_array(data, "region_id")

    # Every cell belonging to an uncut region is still a whole base box, and each such region contributes exactly the
    # one cell it did before: 44 uncut regions, 44 unrefined cells, every one of them 1/8 wide.
    @test length(uncut) == 44
    @test count(r -> r in uncut, region_ids) == 44
    @test all(widths[cell] ≈ 0.125 for (cell, region) in pairs(region_ids) if region in uncut)
    @test length(widths) > length(uncut)
end

@testset "cut_depth is inert without a PhysicalDomain and at :none" begin
    # The non-immersed path must not pay for, or notice, the keyword, and `:none` must reproduce the unrefined sampling
    # exactly.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(1.0; on=boundary(:all))]))
    solution = solve!(model)
    plain(depth) = only(Unfitted._sample_fields(solution, model; subdivisions=2, cut_depth=depth)).data
    @test allequal(_record_contents(plain(depth)) for depth in (:none, 0, 3))

    _, fcm_model, fcm_solution = _cut_depth_disk_model()
    fcm(depth) = only(Unfitted._sample_fields(fcm_solution, fcm_model; subdivisions=:none, cut_depth=depth)).data
    @test _record_contents(fcm(:none)) == _record_contents(fcm(0))
end

@testset "cut_depth rejects values outside its budget" begin
    # The budget is resolved before anything is sampled. That a consumer's refused call leaves no output behind is the
    # consumer's to pin, as `test_vtk.jl` pins it for the VTK writer.
    _, model, solution = _cut_depth_disk_model()
    for bad in (-1, 9, :deep, 2.5)
        @test_throws ArgumentError Unfitted._sample_fields(solution, model; cut_depth=bad, subdivisions=:none)
    end
end

@testset "pin: :degree subdivides each region at its parents' per-axis maximum order" begin
    # The left region (one parent, order (1, 2)) is cut 1 × 2; the right one (orders (3, 1) and (1, 3)) 3 × 3.
    (; model, sol) = _pin_2d
    data = only(Unfitted._sample_fields(sol, model)).data
    covers = _record_array(data, "cover_count")
    @test Unfitted._cell_count(data) == 11
    @test count(==(1), covers) == 2
    @test count(==(2), covers) == 9
end

@testset "pin: subdivisions as an NTuple sets the per-axis counts" begin
    (; model, sol) = _pin_2d
    data = only(Unfitted._sample_fields(sol, model; subdivisions=(2, 3))).data
    @test Unfitted._cell_count(data) == 12             # two regions, 2 × 3 each
    @test all(==(0.25), _cell_extents(data, 1))
    @test all(h -> h ≈ 1 / 3, _cell_extents(data, 2))
end

@testset "pin: point_data and cell_data as a Dict or a tuple of Pairs" begin
    # The three spellings of the entries sample the same arrays; only their order in the record is the container's.
    (; model, sol) = _pin_2d
    uh = (u, c, x, xi) -> u(c, xi)
    tilt = (u, c, x, xi) -> x[1] - 2 * x[2]
    width = (u, c, x, xi) -> c.cell.upper[1] - c.cell.lower[1]
    spellings = (named=((uh=uh, tilt=tilt), (width=width,)),
                 dict=(Dict("uh" => uh, :tilt => tilt), Dict(:width => width)),
                 pairs=((:uh => uh, "tilt" => tilt), ("width" => width,)))
    data = map(((point_data, cell_data),) -> only(Unfitted._sample_fields(sol, model; point_data, cell_data)).data,
               spellings)
    for label in (:dict, :pairs), name in ("uh", "tilt", "width")
        @test _record_array(data[label], name) == _record_array(data.named, name)
    end
end

@testset "pin: the element type each callback value shape is sampled in" begin
    # A record holds a value unpadded and in the type it was computed in, `Bool` included: a tuple becomes the `SVector`
    # of its components, promoted to one type, and the values of one entry are promoted over every sample. What a
    # format cannot store, its consumer maps.
    (; model, sol) = _pin_2d
    shapes = (pair=(u, c, x, xi) -> (x[1], x[2]), mixed_tuple=(u, c, x, xi) -> (1, x[1], 2),
              vector=(u, c, x, xi) -> SVector(x[1], x[2], 1.0), int_vector=(u, c, x, xi) -> SVector(1, 2, c.region_id),
              mixed=(u, c, x, xi) -> c.region_id == 1 ? 1 : 0.5, int=(u, c, x, xi) -> c.region_id,
              single=(u, c, x, xi) -> Float32(x[1]), flag=(u, c, x, xi) -> x[1] > 0.5)
    sampled = (pair=SVector{2,Float64}, mixed_tuple=SVector{3,Float64}, vector=SVector{3,Float64},
               int_vector=SVector{3,Int}, mixed=Float64, int=Int, single=Float32, flag=Bool)
    data = only(Unfitted._sample_fields(sol, model; subdivisions=:none, point_data=shapes)).data
    for (name, type) in pairs(sampled)
        @test eltype(_record_array(data, String(name))) === type
    end
    @test _record_array(data, "pair") == data.points
    @test _record_array(data, "mixed") == [fill(1.0, 4); fill(0.5, 4)]
end

@testset "SVector callback values are promoted over every sample" begin
    # The values of one entry share one element type, promoted over every sample as numbers and tuples already were.
    # Converting every SVector to the first sample's type refused a fractional value after an integer first sample
    # (`InexactError`) and rounded every `Float64` value after a `Float32` one. Region 1 is walked first here, so its
    # value is the first sample's.
    (; model, sol) = _pin_2d
    first_int = (u, c, x, xi) -> c.region_id == 1 ? SVector(1, 2) : SVector(x[1], x[2] / 3)
    first_single = (u, c, x, xi) -> c.region_id == 1 ? SVector(0.5f0, 0.25f0) : SVector(x[1] / 3, x[2] / 3)
    for (f, region_one, other) in
        ((first_single, (0.5, 0.25), x -> (x[1] / 3, x[2] / 3)), (first_int, (1.0, 2.0), x -> (x[1], x[2] / 3)))
        data = only(Unfitted._sample_fields(sol, model; subdivisions=:none, point_data=(v=f,))).data
        v = _record_array(data, "v")
        @test eltype(v) === SVector{2,Float64}
        # Every value exactly what the callback returned.
        regions = _record_array(data, "region_id")
        @test v == [SVector(regions[(k - 1) ÷ 4 + 1] == 1 ? region_one : other(x)) for (k, x) in pairs(data.points)]
    end
end

@testset "a callback value of no accepted shape is refused naming its entry" begin
    # Each of these is refused while it is sampled, naming the entry and what a callback may return, before any
    # consumer sees it. A real value is not refused whatever its type: a `Rational` is sampled as a `Rational`, and
    # whether a format can store it is its consumer's question.
    (; model, sol) = _pin_2d
    refused = [(:point_data, "vector", (u, c, x, xi) -> [x[1], x[2], 0.0]),          # a Vector, not an SVector
               (:point_data, "matrix", (u, c, x, xi) -> SMatrix{2,2}(x[1], 0.0, 0.0, x[2])),
               (:point_data, "text", (u, c, x, xi) -> "text"),
               (:point_data, "complex", (u, c, x, xi) -> complex(x[1], x[2])),
               (:cell_data, "shapes", (u, c, x, xi) -> c.region_id == 1 ? x[1] : (x[1], x[2])),
               (:cell_data, "lengths", (u, c, x, xi) -> c.region_id == 1 ? SVector(x[1]) : SVector(x[1], x[2]))]
    for (kind, name, f) in refused
        entries = NamedTuple{(Symbol(name),)}((f,))
        keywords = kind === :point_data ? (; point_data=entries) : (; cell_data=entries)
        err = try
            Unfitted._sample_fields(sol, model; subdivisions=:none, keywords...)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        message = err === nothing ? "" : sprint(showerror, err)
        @test occursin("$kind entry \"$name\"", message)
        @test occursin("a real number, a tuple of reals or an SVector of reals", message)
    end
    third = (third=(u, c, x, xi) -> 1 // 3,)
    @test eltype(_record_array(only(Unfitted._sample_fields(sol, model; point_data=third)).data, "third")) ===
          Rational{Int}
end
