@testset "diagnostics report for manufactured 1D problem" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=2)
    exact = x -> 1 + x[1] + x[1] * (1 - x[1])
    model = prepare(poisson(V; source=x -> 2.0,
                            dirichlet=[dirichlet(x -> 1 + x[1]; on=boundary(:all))]))
    solution = solve!(model)
    report = diagnostics(model, solution; exact)

    @test report.dimension == 1
    @test report.active_unknowns == 3
    @test report.raw_dofs == 5
    @test report.integration_regions == 2
    @test report.small_overlap_count == 0
    @test isempty(report.small_overlaps)
    @test report.min_integration_volume ≈ 0.5
    @test report.min_relative_integration_volume ≈ 0.5
    @test report.symmetry_residual ≈ 0.0 atol = 1.0e-13
    @test report.condition_estimate >= 1.0
    @test report.solver == :direct
    @test report.residual_norm < 1.0e-12
    @test report.l2_error < 1.0e-12
    @test length(report.levels) == 1
    @test report.levels[1].basis == :integrated_legendre
end

@testset "moving overlay transfer regression" begin
    omega = box((0.0,), (1.0,))
    source_space = space(omega; cells=1, order=1)
    source_space = overlay(source_space, box((0.2,), (0.6,)); cells=1, order=2)
    source_model = prepare(poisson(source_space; source=x -> 0.0))
    target_model = moved(source_model; level=2, to=box((0.35,), (0.85,)))

    coefficients = zeros(Unfitted.active_unknowns(source_model.dofs))
    base_dofs = Unfitted.active_cell_dofs(source_model.dofs, 1, CartesianIndex(1))
    coefficients[base_dofs[1]] = 2.0
    coefficients[base_dofs[2]] = 2.0
    source_solution = Solution(coefficients, source_model.version,
                               Unfitted.SolverDiagnostics(:manual, 0.0, true))
    target_solution = transfer!(source_solution, source_model, target_model)

    @test source_model.version == 1
    @test target_model.version == 1
    @test target_solution.model_version == target_model.version
    @test l2_error(target_solution, target_model, x -> 2.0; norm=:absolute) < 1.0e-12
end

@testset "small overlap diagnostics expose conditioning risk" begin
    tolerance = GeometryTolerance(Float64; merge=1.0e-14, contain=1.0e-14, small_volume=1.0e-3)
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=1, order=2)
    V = overlay(V, box((0.9999,), (1.0,)); cells=1, order=2)
    model = prepare(poisson(V; source=x -> 0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]);
                    tolerance)

    assemble!(model)
    diag = diagnostics(model)

    @test diag.small_overlap_count == 1
    @test length(diag.small_overlaps) == diag.small_overlap_count
    overlap = only(diag.small_overlaps)
    @test overlap.region == 2
    @test overlap.volume ≈ 1.0e-4 atol = 1.0e-12
    @test overlap.relative_volume ≈ 1.0e-4 atol = 1.0e-12
    @test overlap.cover_count == 2
    @test diag.min_integration_volume ≈ 1.0e-4 atol = 1.0e-12
    @test diag.min_relative_integration_volume ≈ 1.0e-4 atol = 1.0e-12
    @test isfinite(diag.condition_estimate)
    @test diag.condition_estimate > 1.0

    move!(model; level=2, to=box((0.2,), (0.4,)))
    moved_diag = diagnostics(model)
    @test moved_diag.small_overlap_count == 0
    @test isempty(moved_diag.small_overlaps)
    @test moved_diag.min_integration_volume > tolerance.small_volume
end

@testset "mutators preserve prepare-time integration-plan options" begin
    # Reproducibility: move! must rebuild the plan with the criterion (and
    # tolerance) captured at prepare, not silently revert to integration_plan
    # defaults. Reaching one configuration two ways — a direct prepare, and a
    # prepare-then-move! — must yield the same integration-region structure.
    # With the non-default :all_levels criterion only the base/overlay overlap
    # is integrated, so a dropped criterion changes the region count.
    omega = box((0.0,), (1.0,))
    base = space(omega; cells=2, order=1)
    bc = [dirichlet(0.0; on=boundary(:all))]

    V_start = overlay(base, box((0.2,), (0.4,)); cells=1, order=1)
    V_end = overlay(base, box((0.5,), (0.7,)); cells=1, order=1)

    direct = prepare(stiffness(V_end; dirichlet=bc); criterion=:all_levels)
    mover = prepare(stiffness(V_start; dirichlet=bc); criterion=:all_levels)
    move!(mover; level=2, to=box((0.5,), (0.7,)))

    @test mover.plan_options == direct.plan_options
    @test diagnostics(mover).integration_regions == diagnostics(direct).integration_regions
    @test diagnostics(mover).active_unknowns == diagnostics(direct).active_unknowns
end

@testset "diagnostics rejects stale solutions" begin
    omega = box((0.0,), (1.0,))
    model = prepare(poisson(space(omega; cells=1, order=1); source=x -> 0.0))
    solution = Solution(zeros(Unfitted.active_unknowns(model.dofs)), model.version + 1,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test_throws ArgumentError diagnostics(model, solution)
end
