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
    base_solution = Solution(base_coefficients, model.version,
                             Unfitted.SolverDiagnostics(:manual, 0.0, true))

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
    # ‖c‖_{L²(Ω)} = |c|·√|Ω| in closed form, so a constant field measured
    # against a zero reference fixes the quadrature measure by its absolute
    # value. Every other l2_error assertion in the suite is `< tiny` or a
    # ratio, and no positive rescaling of the measure — a dropped 2⁻ᴰ box
    # Jacobian, say — can violate one of those. The 2D/3D pair pins the
    # exponent of such a factor, not merely its presence. Order 1 is a
    # partition of unity, so a uniform coefficient vector is exactly the
    # constant field, and `value` re-checks that at an off-grid point.

    # 2D: |Ω| = 2·2 = 4, c = 3 ⇒ ‖c‖ = 3·√4 = 6.
    omega = box((0.0, 0.0), (2.0, 2.0))
    model = prepare(poisson(space(omega; cells=(2, 3), order=1); source=0.0))
    solution = Solution(fill(3.0, active_unknowns(model)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test value(solution, model, (0.7, 1.3)) ≈ 3.0
    @test l2_error(solution, model, x -> 0.0; norm=:absolute) ≈ 6.0 rtol = 1.0e-14
    # A zero reference has no norm to divide by, so the relative form
    # documents a fall back to the absolute one.
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
    @test_throws ArgumentError value(Solution(ones(Unfitted.active_unknowns(model.dofs)),
                                              model.version,
                                              Unfitted.SolverDiagnostics(:manual, 0.0, true)),
                                     model, (1.5,))
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
    m = mesh(domain; cells=(4, 2))

    @test cell_indices(m) == CartesianIndices((4, 2))

    # Each cell box's center sits on the midpoint of its axis-aligned grid
    # interval; collecting them should reproduce the tensor product of
    # per-axis interval midpoints (−0.75, −0.25, 0.25, 0.75) × (−0.5, 0.5).
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
    @test boundary_integral(q -> 1.0, model;
                            on=boundary((axis=1, side=:lower), (axis=2, side=:lower))) ≈ 1.0  # codim-2 point

    # 3D unit cube: ∫_∂Ω 1 ds == 6 (six unit faces), per-face == 1.
    cube = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    Vc = space(cube; cells=(2, 2, 2), order=1)
    cube_model = prepare(poisson(Vc; source=0.0))

    @test boundary_integral(q -> 1.0, cube_model; on=boundary(:all)) ≈ 6.0
    @test boundary_integral(q -> 1.0, cube_model; on=boundary(axis=3, side=:upper)) ≈ 1.0
end

@testset "boundary_integral: analytic trace on a solved Laplace problem" begin
    # Solve −Δu = 0 on (0,1)² with the manufactured u(x,y) = x y. The
    # space resolves this exactly at p = 1, so the trace on y = 1 is
    # u(x,1) = x and ∫₀¹ x dx = 1/2, while the gradient on x = 1 is
    # ∇u = (y, x) so ∫₀¹ ∂u/∂n ds = ∫₀¹ y dy = 1/2.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=2)
    exact_value(x) = x[1] * x[2]
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(exact_value; on=boundary(:all))]))
    sol = solve!(model)

    top_trace = boundary_integral(q -> value(sol, model, q.x), model;
                                  on=boundary(axis=2, side=:upper))
    @test top_trace ≈ 0.5 atol = 1.0e-10

    right_flux = boundary_integral(model; on=boundary(axis=1, side=:upper)) do q
        return field_gradient(sol, model, q.x)[1]
    end
    @test right_flux ≈ 0.5 atol = 1.0e-10
end

@testset "boundary_integral: overlay coverage on a shared face" begin
    # An overlay whose own face coincides with the physical face should
    # contribute to the boundary integral; the integral of `1` over that
    # face must still equal its physical length (the helper does not
    # double-count overlapping parents — it sums Jacobian × Gauss
    # weights once per admissible region).
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

    # Codim-2 corner (vertex in 2D): unit average of the constrained
    # face normals; well-defined and finite.
    n_corner = first_normal(boundary((axis=1, side=:lower), (axis=2, side=:lower)))
    @test n_corner ≈ [-1.0, -1.0] / sqrt(2.0)
end

@testset "boundary_integral on a coupled model needs a field and honours it" begin
    # Two disjoint unit squares. Each has perimeter 4, so ∂Ω of the pair is 8.
    # Without `field=`, `boundary_integral` used to resolve `boundary(:all)` to
    # whichever subdomain a form referenced first and return 4.0 — a plausible
    # number for the wrong domain, with nothing to signal the omission.
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(3, 3), order=1)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2)),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=:u1),
                                       dirichlet(0.0; on=boundary(:all), field=:u2)]))

    # Naming the subdomain gives that subdomain's boundary — including the
    # second one, whose facets the shared cache entry never held.
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
    @test_throws ArgumentError boundary_integral(q -> 1.0, model; on=boundary(:all),
                                                 field=:nope)

    # Single-domain models are unaffected: `field` is optional and, when given,
    # agrees with the default exactly.
    single = prepare(poisson(V1; source=0.0,
                             dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test boundary_integral(q -> 1.0, single; on=boundary(:all)) ==
          boundary_integral(q -> 1.0, single; on=boundary(:all), field=:u)
end

@testset "model.facet_regions caches one entry per Dirichlet selector" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)

    # No Dirichlet → empty cache, zero facet-region count.
    plain = prepare(poisson(V; source=0.0))
    @test isempty(plain.facet_regions)
    @test diagnostics(plain).facet_region_count == 0
    @test nquadpoints(plain; kind=:facet) == 0
    @test nquadpoints(plain) == nquadpoints(plain; kind=:volume)  # default unchanged

    # boundary(:all) → one cached selector. Region count and q-point
    # count match what boundary_integral sees on the same model.
    with_dirichlet = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test length(with_dirichlet.facet_regions) == 1
    selector_count = sum(length, values(with_dirichlet.facet_regions))
    @test diagnostics(with_dirichlet).facet_region_count == selector_count
    @test nquadpoints(with_dirichlet; kind=:facet) > 0

    # Distinct selectors are cached separately.
    multi = prepare(poisson(V; source=0.0,
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                                       dirichlet(0.0; on=boundary(axis=2, side=:upper))]))
    @test length(multi.facet_regions) == 2

    # nquadpoints rejects unknown kinds with a clear message.
    @test_throws ArgumentError nquadpoints(plain; kind=:nope)
end

@testset "facet_regions refresh on move! / activate! / deactivate!" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=1)
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    before_count = diagnostics(model).facet_region_count
    move!(model; level=2, to=box((0.3, 0.3), (0.8, 0.8)))
    # The overlay box did not coincide with the physical boundary
    # before or after the move, so the cached facet region list must
    # still be populated and consistent with the post-move space.
    @test !isempty(model.facet_regions)
    @test diagnostics(model).facet_region_count == before_count
    # The facet-region cache contents must now reflect the new space:
    # at least one parent in each region's list must be from level 1
    # (the base — overlays inside Ω cannot touch ∂Ω).
    for (_, list) in model.facet_regions
        for region in list
            @test all(parent.level == 1 for parent in region.parents)
        end
    end
end

@testset "boundary_integral: empty selection raises" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    # Mask out both cells of the bottom row of the base level (those
    # whose y-index is 1) so no covering parent remains on the y = 0
    # face. The base is the only level that touches the lower face, so
    # the selector finds no admissible region and must raise.
    mask = trues(2, 2)
    mask[:, 1] .= false   # cell (·, y=1) → inactive on the lower row
    V = space(omega; cells=(2, 2), order=1, active=BitArray(mask))
    model = prepare(poisson(V; source=0.0))

    @test_throws ArgumentError boundary_integral(q -> 1.0, model; on=boundary(axis=2, side=:lower))

    # The upper face is still fully covered, so a valid selector on
    # the same model still works — confirming the throw above is from
    # missing facet coverage, not from a wrong-axis check upstream.
    @test boundary_integral(q -> 1.0, model; on=boundary(axis=2, side=:upper)) ≈ 1.0
end
