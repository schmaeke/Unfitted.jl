using LinearAlgebra
using StaticArrays

@testset "projection scaffold" begin
    @test isdefined(Unfitted, :transfer)
end

@testset "L2 transfer preserves an exactly represented polynomial" begin
    omega = box((0.0,), (1.0,))
    source_model = prepare(poisson(space(omega; cells=1, order=2); source=2.0,
                                   dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    source_solution = solve!(source_model)
    target_model = prepare(poisson(space(omega; cells=2, order=2); source=0.0,
                                   dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    target_solution = transfer(source_solution, source_model, target_model)
    target_mass = assemble_matrix(target_model, mass_block(first(target_model.problem.fields)))
    reused_solution = transfer(source_solution, source_model, target_model;
                               via=L2Projection(target_mass; factor=factorize(target_mass)))
    exact = x -> x[1] * (1 - x[1])

    @test target_solution.model_version == target_model.version
    @test target_solution.diagnostics.method == :l2_projection
    @test target_solution.diagnostics.residual_norm < 1.0e-12
    @test reused_solution.coefficients ≈ target_solution.coefficients
    @test l2_error(target_solution, target_model, exact; norm=:absolute) < 1.0e-12
    @test value(target_solution, target_model, (0.0,)) ≈ 0.0 atol = 1.0e-12
    @test value(target_solution, target_model, (1.0,)) ≈ 0.0 atol = 1.0e-12
end

@testset "L2 transfer applies the Dirichlet lift; cached matrix rejects non-homogeneous targets" begin
    omega = box((0.0,), (1.0,))
    exact = x -> 1 + x[1]      # harmonic and linear; representable in both order-2 spaces

    # Source carries u = 1 + x exactly (non-homogeneous Dirichlet, Laplace).
    source_model = prepare(poisson(space(omega; cells=1, order=2); source=0.0,
                                   dirichlet=[dirichlet(exact; on=boundary(:all))]))
    source_solution = solve!(source_model)

    # The default L2 path must assemble the Dirichlet lift, so the projection
    # reproduces u = 1 + x exactly — including the free interior dof at x = 0.5.
    # Without the lift the interior coefficient is wrong (the bug this guards).
    target_model = prepare(poisson(space(omega; cells=2, order=2); source=0.0,
                                   dirichlet=[dirichlet(exact; on=boundary(:all))]))
    target_solution = transfer(source_solution, source_model, target_model)
    @test l2_error(target_solution, target_model, exact; norm=:absolute) < 1.0e-10
    @test value(target_solution, target_model, (0.5,)) ≈ 1.5 atol = 1.0e-10

    # The cached-matrix path omits the lift, so it must reject this target
    # rather than silently return wrong interior coefficients.
    target_mass = assemble_matrix(target_model, mass_block(first(target_model.problem.fields)))
    @test_throws ArgumentError transfer(source_solution, source_model, target_model;
                                        via=L2Projection(target_mass))
end

@testset "L2 transfer between shifted overlays preserves constants" begin
    omega = box((0.0,), (1.0,))
    source_space = space(omega; cells=1, order=1)
    source_space = overlay(source_space, box((0.25,), (0.75,)); cells=1, order=2)
    target_space = moved_space(source_space; level=2, to=box((0.1,), (0.6,)))
    source_model = prepare(poisson(source_space; source=0.0))
    target_model = prepare(poisson(target_space; source=0.0))

    coefficients = zeros(Unfitted.active_unknowns(source_model.dofs))
    base_dofs = Unfitted.active_cell_dofs(source_model.dofs, 1, CartesianIndex(1))
    coefficients[base_dofs[1]] = 1.0
    coefficients[base_dofs[2]] = 1.0
    source_solution = Solution(coefficients, source_model.version,
                               Unfitted.SolverDiagnostics(:manual, 0.0, true))

    target_solution = transfer(source_solution, source_model, target_model)

    @test l2_error(target_solution, target_model, x -> 1.0; norm=:absolute) < 1.0e-12
    @test value(target_solution, target_model, (0.05,)) ≈ 1.0 atol = 1.0e-12
    @test value(target_solution, target_model, (0.5,)) ≈ 1.0 atol = 1.0e-12
end

@testset "L2 transfer preserves vector constants" begin
    omega = box((0.0,), (1.0,))
    source_field = field(:u, space(omega; cells=1, order=1); components=2)
    target_field = field(:u, space(omega; cells=2, order=1); components=2)
    source_model = prepare(poisson(source_field; source=SVector(0.0, 0.0)))
    target_model = prepare(poisson(target_field; source=SVector(0.0, 0.0)))

    coefficients = zeros(Unfitted.active_unknowns(source_model.dofs))
    for component in 1:2
        dofs = Unfitted.active_cell_dofs(source_model.dofs, 1, CartesianIndex(1), component)
        coefficients[dofs] .= component == 1 ? 2.0 : -1.0
    end
    source_solution = Solution(coefficients, source_model.version,
                               Unfitted.SolverDiagnostics(:manual, 0.0, true))

    target_solution = transfer(source_solution, source_model, target_model)
    exact = x -> SVector(2.0, -1.0)

    @test l2_error(target_solution, target_model, exact; norm=:absolute) < 1.0e-12
    @test value(target_solution, target_model, (0.3,)) ≈ exact((0.3,))
end

@testset "L2 transfer preserves same-space multi-field constants" begin
    omega = box((0.0,), (1.0,))
    source_space = space(omega; cells=1, order=1)
    target_space = space(omega; cells=2, order=1)
    source_u = field(:u, source_space)
    source_c = field(:c, source_space)
    target_u = field(:u, target_space)
    target_c = field(:c, target_space)
    zero_form = WeakForm(bilinear=(q, trial) -> 0.0, linear=q -> 0.0, symmetric=true)
    source_model = prepare(Problem((source_u, source_c);
                                   blocks=(block(source_u, source_u, zero_form),
                                           block(source_c, source_c, zero_form)),
                                   loads=(loadform(source_u, zero_form),
                                          loadform(source_c, zero_form))))
    target_model = prepare(Problem((target_u, target_c);
                                   blocks=(block(target_u, target_u, zero_form),
                                           block(target_c, target_c, zero_form)),
                                   loads=(loadform(target_u, zero_form),
                                          loadform(target_c, zero_form))))

    coefficients = zeros(Unfitted.active_unknowns(source_model.dofs))
    coefficients[Unfitted.active_cell_dofs(source_model.dofs, :u, 1, CartesianIndex(1))] .= 2.0
    coefficients[Unfitted.active_cell_dofs(source_model.dofs, :c, 1, CartesianIndex(1))] .= -1.0
    source_solution = Solution(coefficients, source_model.version,
                               Unfitted.SolverDiagnostics(:manual, 0.0, true))

    target_solution = transfer(source_solution, source_model, target_model)
    @test value(target_solution, target_model, target_u, (0.3,)) ≈ 2.0
    @test value(target_solution, target_model, target_c, (0.7,)) ≈ -1.0
end

@testset "L2 transfer preserves target physical constraints" begin
    omega = box((0.0,), (1.0,))
    source_model = prepare(poisson(space(omega; cells=1, order=1); source=0.0))
    source_coefficients = zeros(Unfitted.active_unknowns(source_model.dofs))
    source_dofs = Unfitted.active_cell_dofs(source_model.dofs, 1, CartesianIndex(1))
    source_coefficients[source_dofs[1]] = 1.0
    source_coefficients[source_dofs[2]] = 2.0
    source_solution = Solution(source_coefficients, source_model.version,
                               Unfitted.SolverDiagnostics(:manual, 0.0, true))

    target_model = prepare(poisson(space(omega; cells=2, order=2); source=0.0,
                                   dirichlet=[dirichlet(x -> 1 + x[1]; on=boundary(:all))]))
    target_solution = transfer(source_solution, source_model, target_model)

    @test value(target_solution, target_model, (0.0,)) ≈ 1.0 atol = 1.0e-12
    @test value(target_solution, target_model, (1.0,)) ≈ 2.0 atol = 1.0e-12
    @test l2_error(target_solution, target_model, x -> 1 + x[1]; norm=:absolute) < 1.0e-12
end

@testset "L2 transfer respects source overlay zero extension" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=1, order=1)
    V = overlay(V, box((0.25,), (0.75,)); cells=1, order=2)
    source_model = prepare(poisson(V; source=0.0))
    target_model = prepare(poisson(V; source=0.0))
    coefficients = zeros(Unfitted.active_unknowns(source_model.dofs))
    overlay_dofs = Unfitted.active_cell_dofs(source_model.dofs, 2, CartesianIndex(1))
    coefficients[overlay_dofs[3]] = 1.0
    source_solution = Solution(coefficients, source_model.version,
                               Unfitted.SolverDiagnostics(:manual, 0.0, true))

    target_solution = transfer(source_solution, source_model, target_model)

    @test value(source_solution, source_model, (0.1,)) ≈ 0.0 atol = 1.0e-12
    @test value(target_solution, target_model, (0.1,)) ≈ 0.0 atol = 1.0e-12
    @test l2_error(target_solution, target_model, x -> value(source_solution, source_model, x);
                   norm=:absolute) < 1.0e-12

    target_overlay_raw = Unfitted.cell_dofs(target_model.dofs, 2, CartesianIndex(1))
    constrained = [raw
                   for raw in target_overlay_raw
                   if Unfitted.constraint_kind(target_model.dofs, raw) == :overlay]
    @test all(iszero(Unfitted.constrained_value(target_model.dofs, raw)) for raw in constrained)
end

@testset "L2 transfer rejects incompatible models" begin
    source_model = prepare(poisson(space(box((0.0,), (1.0,)); cells=1, order=1); source=0.0))
    target_model = prepare(poisson(space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=1);
                                   source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(source_model.dofs)), source_model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test_throws ArgumentError transfer(solution, source_model, target_model)
end

@testset "Rewire backend reproduces source pointwise on monotone activation" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    full_mask = trues(2, 2)
    half_mask = BitArray([true false; true false])

    V_small = space(omega; cells=(2, 2), order=1, active=half_mask)
    V_big = space(omega; cells=(2, 2), order=1, active=full_mask)

    source_model = prepare(poisson(V_small; source=0.0))
    target_model = prepare(poisson(V_big; source=0.0))

    src_coeffs = zeros(Unfitted.active_unknowns(source_model.dofs))
    cell_dofs = Unfitted.active_cell_dofs(source_model.dofs, 1, CartesianIndex(1, 1))
    for (i, dof) in pairs(cell_dofs)
        dof == 0 && continue
        src_coeffs[dof] = Float64(i)
    end
    source_solution = Solution(src_coeffs, source_model.version,
                               Unfitted.SolverDiagnostics(:manual, 0.0, true))

    rewired = transfer(source_solution, source_model, target_model; via=Rewire())
    l2_target = transfer(source_solution, source_model, target_model)

    for xy in [(0.1, 0.1), (0.25, 0.25), (0.4, 0.7), (0.0, 0.5), (0.5, 0.0)]
        @test value(rewired, target_model, xy) ≈ value(source_solution, source_model, xy) atol = 1.0e-12
        @test value(rewired, target_model, xy) ≈ value(l2_target, target_model, xy) atol = 1.0e-10
    end
    @test rewired.diagnostics.method === :rewire
end

@testset "Rewire is exact between two strip-mask overlays (monotone growth)" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    small_mask = falses(4, 4)
    for i in 2:3, j in 2:3
        small_mask[i, j] = true
    end
    big_mask = falses(4, 4)
    for i in 1:4, j in 2:3
        big_mask[i, j] = true
    end

    V_small = space(omega; cells=(2, 2), order=2)
    V_small = overlay(V_small, box((0.2, 0.2), (0.8, 0.8)); cells=(4, 4), order=2,
                      active=small_mask)
    V_big = space(omega; cells=(2, 2), order=2)
    V_big = overlay(V_big, box((0.2, 0.2), (0.8, 0.8)); cells=(4, 4), order=2, active=big_mask)

    source_model = prepare(poisson(V_small; source=1.0,
                                   dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    source_solution = solve!(source_model)
    target_model = prepare(poisson(V_big; source=1.0,
                                   dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    rewired = transfer(source_solution, source_model, target_model; via=Rewire())

    # On the original active region (small mask coverage), rewire reproduces source pointwise.
    for x in [SVector(0.5, 0.5), SVector(0.4, 0.5), SVector(0.6, 0.5), SVector(0.5, 0.4)]
        @test value(rewired, target_model, x) ≈ value(source_solution, source_model, x) atol = 1.0e-12
    end
end

@testset "Rewire backend with strict=false skips missing source dofs" begin
    omega = box((0.0,), (1.0,))
    V_small = space(omega; cells=2, order=1)
    V_smaller = space(omega; cells=1, order=1)

    # Source has more cells than target — strict mode should throw, lax mode should drop.
    source_model = prepare(poisson(V_small; source=0.0))
    target_model = prepare(poisson(V_smaller; source=0.0))

    src_coeffs = ones(Unfitted.active_unknowns(source_model.dofs))
    sol = Solution(src_coeffs, source_model.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test_throws ArgumentError transfer(sol, source_model, target_model; via=Rewire(; strict=true))
    rewired = transfer(sol, source_model, target_model; via=Rewire(; strict=false))
    @test rewired.diagnostics.method === :rewire
end

@testset "L2 transfer onto an FCM (physical_domain) target is rejected" begin
    # The default path takes the target mass from the FCM-aware standard
    # assembler (restricted to Ω) but the source-driven rhs over full mesh boxes,
    # so an immersed target pairs an Ω-mass with a full-box rhs and must error
    # rather than silently mis-project. (FCM-aware transfer is a tracked
    # follow-up.) The guard fires before the source coefficients are read, so a
    # trivial source solution suffices.
    omega = box((-1.0, -1.0), (1.0, 1.0))
    disk = physical_domain(x -> sqrt(x[1]^2 + x[2]^2) - 0.7; lipschitz=1.0,
                           subcell_length_scale=0.5)
    source_model = prepare(poisson(space(omega; cells=(2, 2), order=1); source=0.0))
    source_solution = Solution(zeros(Unfitted.active_unknowns(source_model.dofs)),
                               source_model.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))
    target_model = prepare(poisson(space(omega; cells=(2, 2), order=1, physical=disk); source=0.0))
    @test_throws ArgumentError transfer(source_solution, source_model, target_model)
end
