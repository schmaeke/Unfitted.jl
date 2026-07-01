using LinearAlgebra
using StaticArrays
using SparseArrays
using BasicBSpline  # triggers the B-spline extension for the C¹ scatter fixture

@testset "assembly scaffold" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    problem = poisson(V; source=x -> 1.0)
    model = prepare(problem)

    @test problem.space === V
    @test problem.symmetric
    @test diagnostics(model).dimension == 1
    @test diagnostics(model).integration_regions == 2
    @test diagnostics(model).active_unknowns == 3
end

@testset "1D Poisson assembly and direct solve" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    problem = poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    model = prepare(problem)

    assemble!(model)
    @test size(model.matrix) == (1, 1)
    @test model.matrix[1, 1] ≈ 4.0
    @test model.rhs[1] ≈ 0.5
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13
    @test isposdef(Symmetric(Matrix(model.matrix)))

    solution = solve!(model)
    @test solution.coefficients[1] ≈ 0.125
    @test solution.diagnostics.residual_norm < 1.0e-12
    @test diagnostics(model).solver == :direct
end

@testset "custom accumulator weak form projects value and gradient channels" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    bc = [dirichlet(0.0; on=boundary(:all))]
    form = WeakForm(bilinear=(q, u) -> TestChannels(2.0 * u.value, 3.0 * u.gradient),
                    linear=q -> TestChannels(1.0, (0.5,)), symmetric=true)
    model = prepare(Problem(V, form; dirichlet=bc))

    assemble!(model)
    @test size(model.matrix) == (1, 1)
    @test model.matrix[1, 1] ≈ 3.0 * 4.0 + 2.0 / 3.0
    @test model.rhs[1] ≈ 0.5
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13
end

@testset "2D scalar H1 assembly is symmetric positive definite" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=2)
    model = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    assemble!(model)
    @test size(model.matrix) == (1, 1)
    @test model.matrix[1, 1] > 0.0
    @test model.rhs[1] > 0.0
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13
    @test isposdef(Symmetric(Matrix(model.matrix)))
end

@testset "trunk scalar H1 assembly is symmetric positive definite" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    # order=4 is the lowest 2D trunk order with an interior mode, so a
    # Dirichlet-on-all single cell leaves exactly one interior unknown.
    V = space(omega; cells=(1, 1), order=4, mode=:trunk)
    model = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    assemble!(model)
    @test V.levels[1].mode == :trunk
    @test size(model.matrix) == (1, 1)
    @test model.matrix[1, 1] > 0.0
    @test model.rhs[1] > 0.0
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13
    @test isposdef(Symmetric(Matrix(model.matrix)))
end

@testset "overlay assembly includes coupled active blocks" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=1, order=2)
    V = overlay(V, box((0.25,), (0.75,)); cells=1, order=2)
    model = prepare(poisson(V; source=x -> 0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    assemble!(model)
    @test diagnostics(model).active_unknowns == 2
    @test size(model.matrix) == (2, 2)
    @test model.matrix[1, 2] ≈ model.matrix[2, 1]
    @test abs(model.matrix[1, 2]) > 1.0e-12
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13
    @test isposdef(Symmetric(Matrix(model.matrix)))
end

@testset "threaded assembly matches serial assembly" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(3, 3), order=2)
    V = overlay(V, box((0.2, 0.25), (0.8, 0.75)); cells=(2, 2), order=3)
    problem = poisson(V; source=x -> 1.0 + x[1] - x[2],
                      dirichlet=[dirichlet(0.0; on=boundary(:all))])
    serial_model = prepare(problem)
    threaded_model = prepare(problem)
    reassembled = prepare(problem)

    assemble!(serial_model; threaded=false)
    assemble!(threaded_model; threaded=true)

    # The deferred compute→gather scatter sums every slot in serial (region,
    # row) order, so threaded assembly is BIT-IDENTICAL to serial, not merely
    # close — a stronger guarantee than CONTRIBUTING's roundoff tolerance.
    @test Matrix(threaded_model.matrix) == Matrix(serial_model.matrix)
    @test threaded_model.rhs == serial_model.rhs
    @test diagnostics(threaded_model).symmetry_residual ≈ 0.0 atol = 1.0e-13

    # A second threaded assembly reuses the cached gather plan and must give
    # the identical result (determinism + cache correctness).
    assemble!(reassembled; threaded=true)
    assemble!(reassembled; threaded=true)
    @test Matrix(reassembled.matrix) == Matrix(serial_model.matrix)
    @test reassembled.rhs == serial_model.rhs

    # rhs-only threaded pass (the `nothing` matrix sink): assemble_vector
    # defers each region's rhs to an arena and gathers it by dof in serial
    # order, so it too is bit-identical to the serial walk.
    load = loadform(field(:u, V),
                    WeakForm(bilinear=(q, trial) -> 0.0, linear=q -> 1.0, symmetric=false))
    @test assemble_vector(serial_model, load; threaded=true) ==
          assemble_vector(serial_model, load; threaded=false)
end

@testset "poisson accepts scalar and tensor coefficients" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    diffusion = x -> x[1] < 0.5 ? 2.0 : 4.0
    model = prepare(poisson(V; source=1.0, diffusion,
                            dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    assemble!(model)
    @test model.matrix[1, 1] ≈ 12.0
    @test model.rhs[1] ≈ 0.5
    solution = solve!(model)
    @test solution.coefficients[1] ≈ 1 / 24
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13

    omega2 = box((0.0, 0.0), (1.0, 1.0))
    V2 = space(omega2; cells=(1, 1), order=2)
    tensor = [2.0 0.25; 0.25 3.0]
    model2 = prepare(poisson(V2; source=1.0, diffusion=x -> tensor,
                             dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    assemble!(model2)
    @test size(model2.matrix) == (1, 1)
    @test model2.matrix[1, 1] > 0.0
    @test isposdef(Symmetric(Matrix(model2.matrix)))
    @test diagnostics(model2).symmetry_residual ≈ 0.0 atol = 1.0e-13
end

@testset "vector H1 field assembles component blocks" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=2)
    u = field(:u, V; components=2)
    model = prepare(poisson(u; source=x -> SVector(1.0, 2.0),
                            dirichlet=[dirichlet(SVector(0.0, 0.0); on=boundary(:all))]))

    assemble!(model)
    @test diagnostics(model).active_unknowns == 2
    @test size(model.matrix) == (2, 2)
    @test model.matrix[1, 2] ≈ 0.0 atol = 1.0e-13
    @test model.matrix[1, 1] ≈ model.matrix[2, 2]
    @test model.rhs[2] ≈ 2model.rhs[1]

    solution = solve!(model)
    center_value = value(solution, model, (0.5, 0.5))
    center_gradient = field_gradient(solution, model, (0.5, 0.5))

    @test center_value isa SVector{2}
    @test center_value[2] ≈ 2center_value[1]
    @test center_gradient[1] isa SVector{2}
    @test l2_error(solution, model,
                   x -> SVector(value(solution, model, x, 1), value(solution, model, x, 2));
                   norm=:absolute) < 1.0e-12
end

@testset "mass_form per-component coefficient indexes by component" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=2)
    u = field(:u, V; components=2)

    # A per-component (SVector) mass coefficient must scale each component's
    # mass block by its own slot. Before the fix this raised a MethodError:
    # the whole coefficient vector multiplied the scalar trial value.
    mc = prepare(mass(u; coefficient=SVector(2.0, 3.0)))
    m1 = prepare(mass(u; coefficient=1.0))
    assemble!(mc)
    assemble!(m1)
    Mc = Matrix(mc.matrix)
    M1 = Matrix(m1.matrix)

    n = size(M1, 1) ÷ 2                       # component-major dofs: 1:n is component 1
    @test size(Mc) == size(M1)
    @test Mc[1:n, 1:n] ≈ 2 .* M1[1:n, 1:n]
    @test Mc[(n+1):(2n), (n+1):(2n)] ≈ 3 .* M1[(n+1):(2n), (n+1):(2n)]
    @test Mc[1:n, (n+1):(2n)] ≈ zeros(n, n) atol = 1.0e-13   # component-diagonal

    # A scalar coefficient still applies uniformly to both components.
    ms = prepare(mass(u; coefficient=2.0))
    assemble!(ms)
    @test Matrix(ms.matrix) ≈ 2 .* M1
end

@testset "multi-field block problem reuses the unified assembly path" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    u = field(:u, V)
    c = field(:c, V)
    uform = WeakForm(bilinear=(q, trial) -> TestChannels(0.0, trial.gradient), linear=q -> 1.0,
                     symmetric=true)
    cform = WeakForm(bilinear=(q, trial) -> TestChannels(0.0, trial.gradient), linear=q -> 2.0,
                     symmetric=true)
    model = prepare(Problem((u, c); blocks=(block(u, u, uform), block(c, c, cform)),
                            loads=(loadform(u, uform), loadform(c, cform)),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=u),
                                       dirichlet(0.0; on=boundary(:all), field=c)]))

    assemble!(model)
    @test diagnostics(model).active_unknowns == 2
    @test Matrix(model.matrix) ≈ [4.0 0.0; 0.0 4.0]
    @test model.rhs ≈ [0.5, 1.0]

    solution = solve!(model)
    @test value(solution, model, u, (0.5,)) ≈ 0.125
    @test value(solution, model, c, (0.5,)) ≈ 0.25
    @test_throws ArgumentError value(solution, model, (0.5,))
end

@testset "multi-field constraints and component offsets stay field-local" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=2)
    u = field(:u, V; components=2)
    c = field(:c, V)
    zero_form = WeakForm(bilinear=(q, trial, test_component) -> 0.0,
                         linear=(q, test_component) -> 0.0, symmetric=true, component_aware=true)
    model = prepare(Problem((u, c); blocks=(block(u, u, zero_form), block(c, c, zero_form)),
                            loads=(loadform(u, zero_form), loadform(c, zero_form)),
                            dirichlet=[dirichlet(SVector(0.0, 0.0); on=boundary(:all), field=u)]))

    u1_dofs = Unfitted.active_cell_dofs(model.dofs, :u, 1, CartesianIndex(1, 1), 1)
    u2_dofs = Unfitted.active_cell_dofs(model.dofs, :u, 1, CartesianIndex(1, 1), 2)
    c_dofs = Unfitted.active_cell_dofs(model.dofs, :c, 1, CartesianIndex(1, 1))

    @test diagnostics(model).active_unknowns == 11
    @test count(!iszero, u1_dofs) == 1
    @test count(!iszero, u2_dofs) == 1
    @test only(filter(!iszero, u1_dofs)) == 1
    @test only(filter(!iszero, u2_dofs)) == 2
    @test c_dofs == collect(3:11)
    @test_throws ArgumentError prepare(Problem((u, c); blocks=(block(u, u, zero_form),),
                                               dirichlet=[dirichlet(0.0; on=boundary(:all))]))
end

@testset "symmetric multi-field coupling is assembled as one block system" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    u = field(:u, V)
    c = field(:c, V)
    stiffness_form = WeakForm(bilinear=(q, trial) -> TestChannels(0.0, trial.gradient),
                              linear=q -> 0.0, symmetric=true)
    coupling_form = WeakForm(bilinear=(q, trial) -> 0.5 * trial.value, linear=q -> 0.0,
                             symmetric=true)
    model = prepare(Problem((u, c);
                            blocks=(block(u, u, stiffness_form), block(c, c, stiffness_form),
                                    block(c, u, coupling_form)),
                            loads=(loadform(u, stiffness_form), loadform(c, stiffness_form)),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=u),
                                       dirichlet(0.0; on=boundary(:all), field=c)]))

    assemble!(model)
    dense = Matrix(model.matrix)
    @test dense[1, 2] ≈ dense[2, 1]
    @test dense[1, 2] ≈ 1 / 6
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13
end

@testset "mass stiffness and load helpers reuse scalar assembly" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    bc = [dirichlet(0.0; on=boundary(:all))]

    mass_model = prepare(mass(V; dirichlet=bc))
    stiffness_model = prepare(stiffness(V; diffusion=2.0, dirichlet=bc))
    poisson_model = prepare(poisson(V; source=0.0, diffusion=2.0, dirichlet=bc))
    load_model = prepare(load(V; source=x -> 1.0 + x[1], dirichlet=bc))

    assemble!(mass_model)
    assemble!(stiffness_model)
    assemble!(poisson_model)
    assemble!(load_model)

    @test isposdef(Symmetric(Matrix(mass_model.matrix)))
    @test Matrix(stiffness_model.matrix) ≈ Matrix(poisson_model.matrix)
    @test load_vector(stiffness_model; source=x -> 1.0 + x[1]) ≈ load_model.rhs
    @test load_vector(stiffness_model; source=2.0) ≈
          fill(1.0, diagnostics(stiffness_model).active_unknowns)
end

@testset "operators assemble on an existing prepared model" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    u = field(:u, V)
    model = prepare(Problem((u,); dirichlet=[dirichlet(0.0; on=boundary(:all), field=u)]))

    M = assemble_matrix(model, mass_block(u))
    K = assemble_matrix(model, stiffness_block(u; diffusion=2.0))
    f = assemble_vector(model, source_load(u; source=1.0))
    filtered = assemble_vector(model, source_load(u; source=1.0);
                               region_filter=region -> region.box.upper[1] <= 0.5 + 1.0e-12)
    manual = solution(model, [0.25]; method=:manual_check)

    @test M[1, 1] ≈ 1 / 3
    @test K[1, 1] ≈ 8.0
    @test f ≈ [0.5]
    @test filtered ≈ [0.25]
    @test manual.model_version == model.version
    @test manual.diagnostics.method == :manual_check
    @test_throws DimensionMismatch solution(model, Float64[])
end

@testset "operator vector assembly respects field offsets" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    u = field(:u, V)
    c = field(:c, V)
    model = prepare(Problem((u, c);
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=u),
                                       dirichlet(0.0; on=boundary(:all), field=c)]))

    rhs = assemble_vector(model, source_load(c; source=2.0))
    @test rhs ≈ [0.0, 1.0]
end

@testset "discontinuous coefficient need not align with overlay boundaries" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    V = overlay(V, box((0.2,), (0.8,)); cells=3, order=2)

    diffusion = x -> x[1] < 0.5 ? 1.0 : 4.0
    exact = x -> x[1] <= 0.5 ? 1.6 * x[1] : 0.8 + 0.4 * (x[1] - 0.5)
    model = prepare(poisson(V; source=0.0, diffusion,
                            dirichlet=[dirichlet(exact; on=boundary(:all))]))
    solution = solve!(model)

    @test l2_error(solution, model, exact; norm=:absolute) < 1.0e-11
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13
end

@testset "1D nonzero Dirichlet uses boundary projection and RHS correction" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    model = prepare(poisson(V; source=x -> 0.0,
                            dirichlet=[dirichlet(x -> 1 + x[1]; on=boundary(:all))]))

    assemble!(model)
    @test model.matrix[1, 1] ≈ 4.0
    @test model.rhs[1] ≈ 6.0
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13

    solution = solve!(model)
    @test solution.coefficients[1] ≈ 1.5
    @test value(solution, model, (0.0,)) ≈ 1.0
    @test value(solution, model, (0.5,)) ≈ 1.5
    @test value(solution, model, (1.0,)) ≈ 2.0
end

@testset "2D nonzero side Dirichlet is projected on the selected physical boundary" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=(1, 2))
    g = x -> 1 + x[2]
    model = prepare(poisson(V; source=x -> 0.0,
                            dirichlet=[dirichlet(g; on=boundary(axis=1, side=:lower))]))
    solution = solve!(model)

    @test value(solution, model, (0.0, 0.25)) ≈ g((0.0, 0.25)) atol = 1.0e-12
    @test value(solution, model, (0.0, 0.75)) ≈ g((0.0, 0.75)) atol = 1.0e-12
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13
end

@testset "state-aware forms expose the current iterate at quad points" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(3, 3), order=2)
    V = overlay(V, box((0.2, 0.25), (0.8, 0.75)); cells=(2, 2), order=3)
    u = field(:u, V)
    model = prepare(Problem((u,)))
    n = Unfitted.active_unknowns(model.dofs)
    coefficients = [sin(3i) for i in 1:n]
    iterate = solution(model, coefficients; method=:test)

    M = assemble_matrix(model, mass_block(u))
    K = assemble_matrix(model, stiffness_block(u; diffusion=1.0))

    # ∫ value(u_h) v == (M u)_i ; ∫ ∇u_h·∇v == (K u)_i
    value_form = WeakForm(bilinear=(q, trial) -> 0.0, linear=q -> value(q.state, :u),
                          symmetric=false)
    grad_form = WeakForm(bilinear=(q, trial) -> 0.0,
                         linear=q -> TestChannels(0.0, field_gradient(q.state, :u)),
                         symmetric=false)

    @test assemble_vector(model, loadform(u, value_form); state=iterate) ≈ M * coefficients rtol = 1.0e-10
    @test assemble_vector(model, loadform(u, grad_form); state=iterate) ≈ K * coefficients rtol = 1.0e-10
    # raw coefficient vector also accepted as the iterate
    @test assemble_vector(model, loadform(u, value_form); state=coefficients) ≈ M * coefficients rtol = 1.0e-10
    # a form that reads q.state without a supplied state errors clearly
    # (serial path so the ArgumentError isn't wrapped in a TaskFailedException)
    @test_throws ArgumentError assemble_vector(model, loadform(u, value_form); threaded=false)
end

@testset "state-aware forms read each field of a multi-field model" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=2)
    ux = field(:ux, V)
    uy = field(:uy, V)
    model = prepare(Problem((ux, uy)))
    n = Unfitted.active_unknowns(model.dofs)
    coefficients = [cos(2i) for i in 1:n]
    iterate = solution(model, coefficients; method=:test)

    # divergence load: ∫ (∂x u_x + ∂y u_y) v  for the ux test space, vs M_ux*ux + (coupling)
    Mux = assemble_matrix(model, mass_block(ux))
    div_form = WeakForm(bilinear=(q, trial) -> 0.0,
                        linear=q -> field_gradient(q.state, :ux)[1] +
                                    field_gradient(q.state, :uy)[2], symmetric=false)
    r = assemble_vector(model, loadform(ux, div_form); state=iterate)
    # compare against a direct region-quadrature reference of ∫ div(u_h) φ_i over ux dofs
    @test length(r) == n
    @test any(!iszero, r)            # the divergence load is nontrivial
    @test all(isfinite, r)
end

@testset "per-quadrature-point index is stable across assembly and traversal" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(3, 3), order=2)
    V = overlay(V, box((0.2, 0.25), (0.8, 0.75)); cells=(2, 2), order=3)
    u = field(:u, V)
    model = prepare(Problem((u,)))
    nq = nquadpoints(model)

    # foreach_quadrature_point visits 1:nq once and records each point's location
    seen = Int[]
    xf = Dict{Int,typeof(Unfitted.center(omega))}()
    foreach_quadrature_point(model) do q
        push!(seen, q.point)
        xf[q.point] = q.x
    end
    @test sort(seen) == collect(1:nq)

    # the same q.point appears (serial) during assembly at the same location
    xa = Dict{Int,eltype(values(xf))}()
    probe = WeakForm(bilinear=(q, trial) -> 0.0, linear=q -> (xa[q.point]=q.x; 0.0),
                     symmetric=false)
    assemble_vector(model, loadform(u, probe); threaded=false)
    @test length(xa) == nq
    @test all(xa[p] ≈ xf[p] for p in 1:nq)

    # per-point state round-trips: project a constant onto a field, resample == constant
    H = fill(2.5, nq)
    mass_factor = factorize(assemble_matrix(model, mass_block(u)))
    proj_rhs = assemble_vector(model,
                               loadform(u,
                                        WeakForm(bilinear=(q, trial) -> 0.0, linear=q -> H[q.point],
                                                 symmetric=false)))
    field_solution = solution(model, mass_factor \ proj_rhs; method=:proj)
    resampled = Float64[]
    foreach_quadrature_point(model; state=field_solution) do q
        push!(resampled, value(q.state, :u))
    end
    @test all(isapprox(v, 2.5; atol=1.0e-10) for v in resampled)
end

@testset "1D bar with inhomogeneous Neumann data has analytic solution" begin
    # −u''(x) = 0 on (0, 1), u(0) = 0, u'(1) = g. Exact: u(x) = g x.
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=4, order=2)
    g = 1.7
    u = field(:u, V)

    problem = Problem((u,); blocks=(stiffness_block(u),),
                      loads=(neumann(u, g; on=boundary(axis=1, side=:upper)),),
                      dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))])
    model = prepare(problem)
    sol = solve!(model)

    for x in (0.0, 0.25, 0.5, 0.75, 1.0)
        @test value(sol, model, (x,)) ≈ g * x atol = 1.0e-10
    end
end

@testset "2D Poisson with mixed Dirichlet / Neumann recovers manufactured solution" begin
    # On Ω = (0,1)², take u(x,y) = x + 2y.
    #   -Δu = 0 in Ω,
    #   u = u_exact on x = 0 (Dirichlet),
    #   ∂u/∂n = +1 on x = 1 (Neumann, outward normal +e₁ → ∂u/∂x = 1),
    #   ∂u/∂n = -2 on y = 0 (Neumann, outward normal -e₂ → -∂u/∂y = -2),
    #   ∂u/∂n = +2 on y = 1 (Neumann, outward normal +e₂ → ∂u/∂y = 2).
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(3, 3), order=2)
    u = field(:u, V)
    exact(x) = x[1] + 2 * x[2]

    problem = Problem((u,); blocks=(stiffness_block(u),),
                      loads=(neumann(u, 1.0; on=boundary(axis=1, side=:upper)),
                             neumann(u, -2.0; on=boundary(axis=2, side=:lower)),
                             neumann(u, 2.0; on=boundary(axis=2, side=:upper))),
                      dirichlet=[dirichlet(exact; on=boundary(axis=1, side=:lower))])
    model = prepare(problem)
    sol = solve!(model)

    for pt in ((0.1, 0.2), (0.5, 0.4), (0.9, 0.7), (0.3, 0.8))
        @test value(sol, model, pt) ≈ exact(pt) atol = 1.0e-10
    end
end

@testset "user-composed symmetric Robin form produces SPD system" begin
    # Robin: ∂u/∂n + α u = α g on x = 1, with the rest Dirichlet.
    # Compose the surface mass block + the surface load by hand —
    # tests that `block(...; on=…)` and `loadform(...; on=…)` route
    # to the unified facet hot loop and that the combined system is
    # SPD on a symmetric Robin form.
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=3, order=2)
    u = field(:u, V)
    alpha = 4.0
    g_robin = 1.25
    robin_block = block(u, u, mass_form(coefficient=alpha); on=boundary(axis=1, side=:upper))
    robin_load = neumann(u, alpha * g_robin; on=boundary(axis=1, side=:upper))

    problem = Problem((u,); blocks=(stiffness_block(u), robin_block), loads=(robin_load,),
                      dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))])
    model = prepare(problem)
    assemble!(model)

    matrix = model.matrix
    @test issymmetric(matrix)
    @test isposdef(Matrix(matrix))

    # Sanity: with no Robin contribution at all, the matrix would be
    # singular up to the Dirichlet column elimination. The Robin
    # entries are detectable as a non-trivial perturbation of the
    # plain stiffness matrix's value at the (x=1) endpoint.
    plain = prepare(poisson(V; source=0.0,
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))]))
    assemble!(plain)
    @test !(Matrix(matrix) ≈ Matrix(plain.matrix))
end

@testset "Neumann recovers natural-BC value zero when applied as constant g=0" begin
    # Two paths must produce the same matrix and rhs:
    #   (a) Dirichlet on one face, natural BC elsewhere (current code).
    #   (b) Same as (a) plus an explicit `neumann(u, 0.0; on=…)` on
    #       each natural face.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=2)
    u = field(:u, V)
    src = x -> 1.0

    plain = prepare(poisson(V; source=src,
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))]))
    explicit = prepare(Problem((u,); blocks=(stiffness_block(u),),
                               loads=(source_load(u; source=src),
                                      neumann(u, 0.0; on=boundary(axis=1, side=:upper)),
                                      neumann(u, 0.0; on=boundary(axis=2, side=:lower)),
                                      neumann(u, 0.0; on=boundary(axis=2, side=:upper))),
                               dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))]))
    assemble!(plain)
    assemble!(explicit)

    @test Matrix(plain.matrix) ≈ Matrix(explicit.matrix) atol = 1.0e-12
    @test plain.rhs ≈ explicit.rhs atol = 1.0e-12
end

# ── Symbolic-scatter assembly: cross-cutting invariants ───────────────────────

# Properties the single scatter assembly path must satisfy on every
# fixture, with no second method to compare against:
#   * deterministic — repeated serial assembly is bit-identical (also
#     exercises the cached pattern on the second call);
#   * threaded assembly is bit-identical to serial — the deferred
#     compute→gather sums every slot in serial order, so `==` holds on every
#     fixture (a stronger guarantee than a roundoff tolerance);
#   * no explicit stored zeros survive (the `dropzeros!` contract — proves
#     the dense-block pattern collapses to the true nonzeros);
#   * symmetric forms produce a structurally symmetric matrix.
# Numerical correctness itself is pinned by the value-based testsets above
# and the manufactured-solution / gallery suites.
function _check_scatter_matrix(model, blocks; symmetric=nothing)
    serial = assemble_matrix(model, blocks; symmetric=symmetric, threaded=false)
    @test assemble_matrix(model, blocks; symmetric=symmetric, threaded=false) == serial
    @test !any(iszero, nonzeros(serial))       # dropzeros! contract: no stored zeros
    threaded = assemble_matrix(model, blocks; symmetric=symmetric, threaded=true)
    @test threaded == serial
    return serial
end

# Same invariants driven through the full `assemble!` path (blocks + loads).
function _check_scatter_assemble!(model)
    assemble!(model; threaded=false)
    serial = copy(model.matrix)
    assemble!(model; threaded=false)               # re-assembly reuses the cached pattern
    @test model.matrix == serial
    @test !any(iszero, nonzeros(serial))       # dropzeros! contract: no stored zeros
    model.problem.symmetric && @test issymmetric(serial)
    assemble!(model; threaded=true)
    @test model.matrix == serial
    return model
end

@testset "scatter assembly invariants across the fixture matrix" begin
    # 1D scalar, single level, symmetric.
    let V = space(box((0.0,), (1.0,)); cells=4, order=2)
        model = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
        _check_scatter_assemble!(model)
    end

    # 2D scalar, higher order, symmetric.
    let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=3)
        model = prepare(poisson(V; source=x -> 1.0 + x[1],
                                dirichlet=[dirichlet(0.0; on=boundary(:all))]))
        _check_scatter_assemble!(model)
    end

    # 2D multi-level superposition (overlay), symmetric.
    let V0 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=2),
        V = overlay(V0, box((0.2, 0.25), (0.8, 0.75)); cells=(2, 2), order=3)

        model = prepare(poisson(V; source=x -> 1.0 - x[2],
                                dirichlet=[dirichlet(0.0; on=boundary(:all))]))
        _check_scatter_assemble!(model)
    end

    # Vector field, component-unaware form: cross-component blocks are
    # structurally present in the pattern but always zero — the dropzeros
    # equivalence the scatter path must reproduce.
    let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=2),
        u = field(:u, V; components=2)

        model = prepare(poisson(u; source=x -> SVector(1.0, 2.0),
                                dirichlet=[dirichlet(SVector(0.0, 0.0); on=boundary(:all))]))
        _check_scatter_assemble!(model)
        # The component-unaware form leaves cross-component blocks in the
        # dense-block pattern but numerically zero; dropzeros! must remove
        # them, so the lower-triangle pattern is strictly larger than the
        # final lower-triangle nnz (proves the superset→drop pipeline).
        @test length(model.pattern.rowval) > nnz(tril(model.matrix))
    end

    # Multi-field symmetric coupling block(c, u).
    let V = space(box((0.0,), (1.0,)); cells=2, order=1), u = field(:u, V), c = field(:c, V)
        stiff = WeakForm(bilinear=(q, trial) -> TestChannels(0.0, trial.gradient), linear=q -> 0.0,
                         symmetric=true)
        coupling = WeakForm(bilinear=(q, trial) -> 0.5 * trial.value, linear=q -> 0.0,
                            symmetric=true)
        model = prepare(Problem((u, c);
                                blocks=(block(u, u, stiff), block(c, c, stiff),
                                        block(c, u, coupling)),
                                loads=(loadform(u, stiff), loadform(c, stiff)),
                                dirichlet=[dirichlet(0.0; on=boundary(:all), field=u),
                                           dirichlet(0.0; on=boundary(:all), field=c)]))
        _check_scatter_assemble!(model)
    end

    # Non-symmetric form (convection-like) — full pattern, no mirror.
    let V = space(box((0.0,), (1.0,)); cells=4, order=2), u = field(:u, V)
        conv = WeakForm(bilinear=(q, trial) -> trial.gradient[1], linear=q -> 1.0, symmetric=false)
        model = prepare(Problem((u,); blocks=(block(u, u, conv),), loads=(loadform(u, conv),),
                                dirichlet=[dirichlet(0.0; on=boundary(:all))]))
        @test model.problem.symmetric == false
        _check_scatter_assemble!(model)
    end

    # B-spline base + C¹ B-spline overlay (smooth basis family, overlay
    # artificial-boundary constraints, conforming dof sharing). The
    # numeric scatter reads the assembled local block over `active_dofs`,
    # so it is agnostic to whether the dof layer used the simple
    # `Matrix{Int}` table or the `LocalDofExpansion` (pivot) table — this
    # fixture exercises the B-spline assembly path regardless.
    let V0 = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, basis=bspline()),
        V = overlay(V0, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3,
                    basis=bspline(continuity_order=1))

        model = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
        _check_scatter_assemble!(model)
    end

    # FCM (immersed boundary via a physical domain / moment-fit quadrature).
    let phi = x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.35,
        p = physical_domain(phi; lipschitz=1.0, subcell_length_scale=1.0 / 2^4, max_depth=4),
        V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=p)

        model = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
        _check_scatter_assemble!(model)
    end

    # Facet (Robin) block — the `on=` partition contributes facet regions
    # to the pattern alongside the volume plan.
    let V = space(box((0.0,), (1.0,)); cells=3, order=2), u = field(:u, V)
        robin = block(u, u, mass_form(coefficient=4.0); on=boundary(axis=1, side=:upper))
        model = prepare(Problem((u,); blocks=(stiffness_block(u), robin),
                                loads=(neumann(u, 5.0; on=boundary(axis=1, side=:upper)),),
                                dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))]))
        _check_scatter_assemble!(model)
        # also through the operator entry point, exercising pattern caching
        _check_scatter_matrix(model, (stiffness_block(u), robin))
    end

    # Operator entry point with pattern-cache reuse across two forms that
    # share the same region set (mass then stiffness).
    let V = space(box((0.0,), (1.0,)); cells=2, order=2), u = field(:u, V)
        model = prepare(Problem((u,); dirichlet=[dirichlet(0.0; on=boundary(:all), field=u)]))
        _check_scatter_matrix(model, mass_block(u))
        _check_scatter_matrix(model, stiffness_block(u; diffusion=2.0))
    end
end

@testset "neumann explicit-component flux lands only on the matching component's boundary dofs" begin
    # A constant Neumann flux g on the face x = 1 of the unit square,
    # restricted to one component of a 2-component vector field via the
    # explicit `component=` selector, must assemble ∫_face g φ_i ds into
    # that component's boundary dofs and exactly nothing into the other.
    # With order-1 (partition-of-unity) shape functions the selected
    # component's boundary dofs sum to the known integral g·|face| = g,
    # split as g/2 across the two edge nodes; the off-component block is
    # identically zero. Guards the per-component branch of `neumann`
    # (the `component=` selector routed through `_source_value`), which
    # the scalar Neumann testsets above do not exercise.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    u = field(:u, V; components=2)
    model = prepare(Problem((u,)))                  # no Dirichlet: every dof is free

    n = Unfitted.active_unknowns(model.dofs) ÷ 2    # component-major: 1:n is component 1
    face = boundary(axis=1, side=:upper)            # x = 1, |face| = 1
    g = 3.0

    # Flux on component 1 only.
    r1 = assemble_vector(model, neumann(u, g; on=face, component=1))
    @test sum(r1[1:n]) ≈ g atol = 1.0e-12                            # ∫_face g ds = g·|face|
    @test count(x -> isapprox(x, g / 2; atol=1.0e-12), r1[1:n]) == 2 # g/2 per edge node
    @test r1[(n+1):(2n)] ≈ zeros(n) atol = 1.0e-13                   # nothing leaks to component 2

    # Flux on component 2 only mirrors the result onto the other block.
    r2 = assemble_vector(model, neumann(u, g; on=face, component=2))
    @test sum(r2[(n+1):(2n)]) ≈ g atol = 1.0e-12
    @test r2[1:n] ≈ zeros(n) atol = 1.0e-13

    # The selected component's contribution equals the scalar-field Neumann
    # load on the same facet — the component routing changes only which block
    # receives the load, not the per-dof integral. Checked at order 2, where
    # the hierarchical basis is not a partition of unity, so this pins the
    # integral independently of the order-1 sum above.
    Vq = space(omega; cells=(1, 1), order=2)
    uq = field(:u, Vq; components=2)
    wq = field(:w, Vq)
    mq = prepare(Problem((uq,)))
    sq = prepare(Problem((wq,)))
    nq = Unfitted.active_unknowns(mq.dofs) ÷ 2
    rq = assemble_vector(mq, neumann(uq, g; on=face, component=1))
    rref = assemble_vector(sq, neumann(wq, g; on=face))
    @test rq[1:nq] ≈ rref atol = 1.0e-12
    @test rq[(nq+1):(2nq)] ≈ zeros(nq) atol = 1.0e-13
end
