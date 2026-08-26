@testset "public API scaffold" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=2)
    V2 = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(2, 2), order=3)
    V3 = moved_space(V2; level=2, to=box((0.2, 0.2), (0.6, 0.6)))
    model = prepare(poisson(V2; source=1.0))

    @test Unfitted.level_count(V) == 1
    @test Unfitted.level_count(V2) == 2
    @test V2.levels[2].role == :overlay
    @test V2.levels[2].order == (3, 3)
    @test V3.levels[2].mesh.domain == box((0.2, 0.2), (0.6, 0.6))
    @test diagnostics(model).dimension == 2
    @test diagnostics(model).integration_regions == 16

    old_plan = Unfitted.integration_plan(model)
    move!(model; level=2, to=box((0.2, 0.2), (0.6, 0.6)))
    @test model.version == 2
    @test Unfitted.integration_plan(model) !== old_plan
    @test model.matrix === nothing
    @test diagnostics(model).integration_regions == 35   # merged (was 49 unmerged)
end

@testset "custom solver hook and compact display" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    called = Ref(false)
    solution = solve!(model; method=:external_direct, linear_solver=(A, b) -> begin
                          called[] = true
                          A \ b
                      end)

    @test called[]
    @test solution.coefficients[1] ≈ 0.125
    @test solution.diagnostics.method == :external_direct
    @test solution.diagnostics.residual_norm < 1.0e-12
    @test diagnostics(model).solver == :external_direct
    @test_throws ArgumentError solve!(model; method=:iterative_without_hook)
    @test_throws DimensionMismatch solve!(model; linear_solver=(A, b) -> zeros(0))

    shown = (sprint(show, V.levels[1]), sprint(show, V), sprint(show, model),
             sprint(show, diagnostics(model)), sprint(show, solution),
             sprint(show, solution.diagnostics))

    @test occursin("Level", shown[1])
    @test occursin("Space", shown[2])
    @test occursin("Model", shown[3])
    @test occursin("AssemblyDiagnostics", shown[4])
    @test occursin("Solution", shown[5])
    @test occursin("SolverDiagnostics", shown[6])
end

@testset "transfer requires source and target models" begin
    omega = box((0.0,), (1.0,))
    model = prepare(poisson(space(omega; cells=1, order=1); source=0.0))
    solution = Solution(zeros(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test_throws ArgumentError transfer(solution, model)
end
