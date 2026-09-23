@testset "diagnostics report for manufactured 1D problem" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=2)
    exact = x -> 1 + x[1] + x[1] * (1 - x[1])
    model = prepare(poisson(V; source=2.0, dirichlet=[dirichlet(x -> 1 + x[1]; on=boundary(:all))]))
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
    source_model = prepare(poisson(source_space; source=0.0))
    target_model = moved(source_model; level=2, to=box((0.35,), (0.85,)))

    coefficients = zeros(Unfitted.active_unknowns(source_model.dofs))
    base_dofs = Unfitted.active_cell_dofs(source_model.dofs, 1, CartesianIndex(1))
    coefficients[base_dofs[1]] = 2.0
    coefficients[base_dofs[2]] = 2.0
    source_solution = Solution(coefficients, source_model.version,
                               Unfitted.SolverDiagnostics(:manual, 0.0, true))
    target_solution = transfer(source_solution, source_model, target_model)

    # Two freshly prepared models of *different* discretisations. Their pins
    # differ, which is what makes the transfer above mandatory rather than
    # optional — see the two testsets below.
    @test source_model.version != target_model.version
    @test target_solution.model_version == target_model.version
    @test l2_error(target_solution, target_model, x -> 2.0; norm=:absolute) < 1.0e-12
end

@testset "a solution does not cross between two models of the same size" begin
    # The failure this pins. Both models are freshly prepared, both carry the
    # same number of active unknowns, and their level-2 masks are structurally
    # different — an ordinary coincidence on a ladder, where many different
    # masks give the same count. While every `prepare` stamped the literal 1
    # into `model.version` the guard compared 1 == 1 and then only a length, so
    # a solution from one was accepted on the other in silence: an `l2_error` of
    # 3.2615e-3 was reported where the truth was 3.2095e-3, and a complete
    # `estimate` was computed from the wrong coefficient vector.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = ladder(omega; cells=4, depth=1, order=2)
    function prepared(cells)
        mask = falses(8, 8)
        for c in cells
            mask[c] = true
        end
        return prepare(poisson(adapt(V, 2 => mask); source=1.0,
                               dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    end
    a = prepared((CartesianIndex(3, 3), CartesianIndex(3, 4)))
    b = prepared((CartesianIndex(6, 6), CartesianIndex(6, 7)))
    ua = solve!(a)
    solve!(b)

    @test active_unknowns(a) == active_unknowns(b)      # the coincidence
    @test a.version != b.version                        # ... which the pin sees through
    @test_throws ArgumentError l2_error(ua, b, x -> 0.0)
    @test_throws ArgumentError value(ua, b, (0.5, 0.5))
    @test_throws ArgumentError estimate(b, ua)
    @test_throws ArgumentError Unfitted._checked_quadfield(QuadField{Float64}(a), b)
    # and the control: on its own model everything still works
    @test value(ua, a, (0.5, 0.5)) isa Float64
end

@testset "a problem rebuilt at an unchanged discretisation keeps its state" begin
    # The shape `examples/reproductions/traveling_laser_2d` runs: a fresh
    # `Problem` every step — new `Field` object, new source closure, new
    # `dirichlet` spec — prepared on a space built afresh from the same
    # description, with the state carried forward and no transfer. This is why
    # the pin digests structure only. Folding in anything that carries object
    # identity (a closure, a spec, a `Field`, a `PhysicalDomain`) would break
    # this loop exactly as badly as a bare counter breaks the testset above,
    # only from the other side.
    omega = box((0.0,), (1.0,))
    described() = overlay(space(omega; cells=4, order=2), box((0.25,), (0.75,)); cells=2, order=3)
    step() = prepare(poisson(described(); source=x -> 1.0 + x[1],
                             dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    m1 = step()
    u = solve!(m1)
    m2 = step()

    @test m2 !== m1 && m2.problem !== m1.problem       # genuinely rebuilt
    @test m2.version == m1.version                     # ... and the same discretisation
    @test l2_error(u, m2, x -> 0.0) == l2_error(u, m1, x -> 0.0)
    @test value(u, m2, (0.5,)) == value(u, m1, (0.5,))
end

@testset "small overlap diagnostics expose conditioning risk" begin
    tolerance = GeometryTolerance(Float64; merge=1.0e-14, contain=1.0e-14, small_volume=1.0e-3)
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=1, order=2)
    V = overlay(V, box((0.9999,), (1.0,)); cells=1, order=2)
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]);
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
    model = prepare(poisson(space(omega; cells=1, order=1); source=0.0))
    solution = Solution(zeros(Unfitted.active_unknowns(model.dofs)), model.version + 1,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test_throws ArgumentError diagnostics(model, solution)
end

@testset "the discretisation pin separates a pruned model from its unpruned twin" begin
    # Leaf semantics change which raws survive, so the two spaces carry different
    # active dof numberings and a solution from one must not be accepted on the
    # other. The pin has to say so even though the geometry is identical — it is
    # the only thing that differs between these two models.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    V = overlay(space(Ω; cells=8, order=3), box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3)
    pruned = prepare(mass(V))
    unpruned = prepare(mass(V); prune=false)
    @test diagnostics(unpruned).active_unknowns > diagnostics(pruned).active_unknowns
    @test pruned.version != unpruned.version

    # And the converse: where nothing is covered the two layouts are the same
    # layout, so the pin agreeing is correct rather than a miss. It agrees because
    # the flag itself is not digested — the numbering is, and here it is identical
    # down to the `(raw, component) -> active id` map.
    flat = space(Ω; cells=8, order=3)
    @test prepare(mass(flat)).version == prepare(mass(flat); prune=false).version
end
