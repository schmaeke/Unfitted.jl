using LinearAlgebra
using StaticArrays
using SparseArrays
using BasicBSpline  # triggers the B-spline extension for the C¹ scatter fixture

@testset "assembly scaffold" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    problem = poisson(V; source=1.0)
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
    problem = poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
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

@testset "a WeakForm declares only the sides it has" begin
    # ∫_Ω ∇v ⋅ ∇u dx with no load, and ∫_Ω 1 ⋅ v dx with no operator: each side
    # stands alone, and neither writes the zero stub the constructor used to
    # demand.
    bilinear_only = WeakForm(bilinear=(q, trial) -> TestChannels(0.0, trial.gradient),
                             symmetric=true)
    @test bilinear_only.linear === nothing
    @test bilinear_only.bilinear !== nothing

    linear_only = WeakForm(linear=q -> 1.0)
    @test linear_only.bilinear === nothing
    @test linear_only.linear !== nothing

    # Neither side is no contribution at all; the metadata flags carry no
    # integrand and do not stand in for one.
    @test_throws ArgumentError WeakForm()
    @test_throws ArgumentError WeakForm(symmetric=true, component_aware=true)

    # The shape the space-time example uses: the operator arrives as a block and
    # the load as its own `loadform`. The reference values are the 1D Poisson
    # ones from the testset above, because this is that problem spelled in two
    # halves.
    V = space(box((0.0,), (1.0,)); cells=2, order=1)
    u = field(:u, V)
    model = prepare(Problem((u,); blocks=(block(u, u, bilinear_only),),
                            loads=(loadform(u, linear_only),),
                            dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test model.problem.symmetric

    assemble!(model)
    @test size(model.matrix) == (1, 1)
    @test model.matrix[1, 1] ≈ 4.0
    @test model.rhs[1] ≈ 0.5

    solution = solve!(model)
    @test solution.coefficients[1] ≈ 0.125
    @test solution.diagnostics.residual_norm < 1.0e-12
end

@testset "the single-field shorthand attaches the sides the form declares" begin
    V = space(box((0.0,), (1.0,)); cells=2, order=1)
    bilinear_only = WeakForm(bilinear=(q, trial) -> TestChannels(0.0, trial.gradient),
                             symmetric=true)
    linear_only = WeakForm(linear=q -> 1.0)

    # Laplace with boundary data: an operator, no load, and a solution driven
    # entirely by the inhomogeneous Dirichlet datum, which used to need a dead
    # `linear = q -> 0.0` to express. The rhs is the Dirichlet column
    # elimination alone, so u ≡ 2 and rhs = 4 × 2.
    model = prepare(Problem(V, bilinear_only; dirichlet=[dirichlet(2.0; on=boundary(:all))]))
    @test length(model.problem.blocks) == 1
    @test isempty(model.problem.loads)

    assemble!(model)
    @test model.matrix[1, 1] ≈ 4.0
    @test model.rhs[1] ≈ 8.0

    solution = solve!(model)
    @test solution.coefficients[1] ≈ 2.0
    @test value(solution, model, (0.5,)) ≈ 2.0
    @test value(solution, model, (0.25,)) ≈ 2.0

    # The other side is treated the same way: a linear-only form yields a
    # load-only problem. Whether that problem is solvable is the caller's
    # business; what matters here is that neither side is invented.
    load_only = prepare(Problem(V, linear_only))
    @test isempty(load_only.problem.blocks)
    @test length(load_only.problem.loads) == 1

    # A *direct* attachment on the side a form does not declare is still an
    # error and not a silent zero — the `WeakForm` docstring says so, and this
    # keeps it honest. Serial assembly so the `MethodError` arrives unwrapped
    # rather than inside a `CompositeException` from a worker task.
    u = field(:u, V)
    wrong_block = prepare(Problem((u,); blocks=(block(u, u, linear_only),),
                                  dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test_throws MethodError assemble!(wrong_block; threaded=false)

    wrong_load = prepare(Problem((u,); blocks=(block(u, u, bilinear_only),),
                                 loads=(loadform(u, bilinear_only),),
                                 dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test_throws MethodError assemble!(wrong_load; threaded=false)
end

@testset "2D scalar H1 assembly is symmetric positive definite" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=2)
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

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
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

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
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    assemble!(model)
    @test diagnostics(model).active_unknowns == 2
    @test size(model.matrix) == (2, 2)
    @test model.matrix[1, 2] ≈ model.matrix[2, 1]
    @test abs(model.matrix[1, 2]) > 1.0e-12
    @test diagnostics(model).symmetry_residual ≈ 0.0 atol = 1.0e-13
    @test isposdef(Symmetric(Matrix(model.matrix)))
end

# Allocation count of one warm serial assembly call, behind a function barrier so
# the count is the call's own.
function _warm_allocations(assemble, model, args...)
    assemble(model, args...; threaded=false)
    assemble(model, args...; threaded=false)
    return @allocations assemble(model, args...; threaded=false)
end

@testset "assembly allocates nothing per region or quadrature point" begin
    # The kernel's per-region and per-point work must not allocate once warm, so
    # a call's allocation count cannot grow with the mesh. The likeliest way to
    # break that is the basis bank: the hot loop reads `ws.bank.bases[level]`
    # once per parent per quadrature point, and an abstract bank turns the read
    # into a dynamic dispatch that boxes the basis call's arguments, so a
    # homogeneous space has to give the bank a concrete eltype. Measured on this
    # fixture, the per-region dof tables of the earlier kernel took `assemble!`
    # from 1 005 allocations at 8² cells to 3 779 at 16², and `assemble_matrix`
    # from 332 at 4² to 3 775 at 16²; on the mixed-family fixture below, a boxed
    # basis call grew the count from 1 724 at 4² to 26 510 at 16². The kernel
    # that allocates nothing grows by at most a handful.
    #
    # `assemble!` is compared between 8² and 16², because below 256 unknowns it
    # also computes the condition estimate, whose allocations would hide growth
    # at 4²; `assemble_matrix` has no such step and is compared from 4².
    omega = box((0.0, 0.0), (1.0, 1.0))
    patch = box((0.25, 0.25), (0.75, 0.75))
    homogeneous(c) = prepare(mass(overlay(space(omega; cells=(c, c), order=2), patch; cells=c ÷ 2,
                                          order=3)))
    small, medium, large = homogeneous(4), homogeneous(8), homogeneous(16)
    @test eltype(Unfitted._assembly_workspace(small).bank.bases) === IntegratedLegendre
    @test _warm_allocations(assemble!, large) - _warm_allocations(assemble!, medium) < 20
    mass_u(model) = mass_block(only(model.problem.fields))
    @test _warm_allocations(assemble_matrix, large, mass_u(large)) -
          _warm_allocations(assemble_matrix, small, mass_u(small)) < 20

    # A space that genuinely mixes families widens the bank back to a common
    # supertype, pays the dynamic basis call, and must still assemble.
    mixed = prepare(mass(overlay(space(omega; cells=(4, 4), order=2), patch; cells=2, order=3,
                                 basis=bspline())))
    @test !isconcretetype(eltype(Unfitted._assembly_workspace(mixed).bank.bases))
    assemble!(mixed)
    @test all(isfinite, mixed.matrix.nzval)
    @test issymmetric(mixed.matrix)
end

@testset "symmetric mirror reproduces A + Aᵀ − diag(A) to the bit" begin
    # `_matrix_from_pattern` expands a symmetric form's lower triangle in a single
    # pass rather than materialising `A + Aᵀ − diag(A)`. The two spellings must
    # agree exactly — same stored pattern, same bits — including on the explicit
    # zeros Dirichlet column elimination leaves behind and on columns whose
    # diagonal slot the pattern never allocated.
    n = 6
    rows = [1, 3, 6, 2, 5, 4, 5, 6, 6]
    cols = [1, 1, 1, 2, 2, 3, 4, 5, 6]
    vals = [2.0, -1.5, 0.0, 4.0, 3.25, -7.0, 0.5, 1.0, -0.0]
    lower = sparse(rows, cols, vals, n, n)
    @test nnz(lower) == length(vals)  # `sparse` kept the explicit zeros

    reference = dropzeros!(lower + lower' - spdiagm(0 => diag(lower)))
    pattern = Unfitted.AssemblyPattern(n, lower.colptr, lower.rowval, true)
    mirrored = Unfitted._matrix_from_pattern(pattern, copy(lower.nzval))

    @test mirrored.colptr == reference.colptr
    @test mirrored.rowval == reference.rowval
    @test all(map(isequal, mirrored.nzval, reference.nzval))
    @test issymmetric(mirrored)
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

    # A second threaded assembly reuses the cached region dofs and pattern and
    # must give the identical result (determinism + cache correctness).
    assemble!(reassembled; threaded=true)
    assemble!(reassembled; threaded=true)
    @test Matrix(reassembled.matrix) == Matrix(serial_model.matrix)
    @test reassembled.rhs == serial_model.rhs

    # rhs-only threaded pass: assemble_vector defers each region's rhs to an
    # arena and gathers it by dof in serial order, so it too is bit-identical
    # to the serial walk.
    load = loadform(field(:u, V),
                    WeakForm(bilinear=(q, trial) -> 0.0, linear=q -> 1.0, symmetric=false))
    @test assemble_vector(serial_model, load; threaded=true) ==
          assemble_vector(serial_model, load; threaded=false)
end

@testset "a region filter assembles the same vector threaded and serial" begin
    # `region_filter` lets a compactly supported load skip the regions outside its
    # support. The threaded driver has to honour it exactly as the serial walk
    # does: a rejected region contributes nothing, and the regions that remain are
    # summed in the same order, so the two vectors agree to the bit. The filtered
    # and unfiltered threaded calls alternate on one model, so scratch that a call
    # reuses from the previous one can never carry a rejected region's stale
    # values into the result.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2)
    V = overlay(V, box((0.2, 0.25), (0.8, 0.75)); cells=(2, 2), order=3)
    u = field(:u, V)
    model = prepare(Problem((u,); dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    load = source_load(u; source=1.0)
    left = region -> region.box.upper[1] <= 0.5 + 1.0e-12
    right = region -> region.box.upper[1] > 0.5 + 1.0e-12
    nowhere = region -> false

    full = assemble_vector(model, load; threaded=false)
    lo = assemble_vector(model, load; region_filter=left, threaded=false)
    hi = assemble_vector(model, load; region_filter=right, threaded=false)
    none = zeros(length(full))
    # The filters really select. Each half misses part of the load, and the two
    # halves partition the regions, so they add up to the full vector up to the
    # order of summation.
    @test lo != full && hi != full
    @test lo + hi ≈ full rtol = 1.0e-13
    @test assemble_vector(model, load; region_filter=nowhere, threaded=false) == none

    for _ in 1:2
        @test assemble_vector(model, load; threaded=true) == full
        @test assemble_vector(model, load; region_filter=left, threaded=true) == lo
        @test assemble_vector(model, load; region_filter=right, threaded=true) == hi
        @test assemble_vector(model, load; region_filter=nowhere, threaded=true) == none
    end
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
    model = prepare(poisson(u; source=SVector(1.0, 2.0),
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
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(x -> 1 + x[1]; on=boundary(:all))]))

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
    model = prepare(poisson(V; source=0.0,
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

@testset "a state-reading tangent is the derivative of its residual" begin
    # A Newton step for −∇·((1 + u²) ∇u) = 1 assembles the residual
    #
    #     Rᵢ(c) = ∫ (1 + u_h²) ∇u_h · ∇φᵢ − φᵢ
    #
    # with `assemble_vector(…; state = c)`, and its tangent
    #
    #     Jᵢⱼ = ∂Rᵢ/∂cⱼ = ∫ (1 + u_h²) ∇φⱼ · ∇φᵢ + 2 u_h φⱼ ∇u_h · ∇φᵢ
    #
    # with `assemble_matrix(…; state = c)`, both reading the iterate u_h through
    # `q.state`. The tangent has to be the derivative of the residual, which a
    # central difference checks column by column: its truncation error is O(h²),
    # and it measured 1.9e-11 relative here. The nonzero Dirichlet datum puts a
    # lift into u_h, so the state's constrained values are read as well, and the
    # 2 u_h φⱼ term makes the tangent unsymmetric, so the full pattern is used.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=2)
    V = overlay(V, box((0.2, 0.25), (0.8, 0.75)); cells=(2, 2), order=3)
    u = field(:u, V)
    model = prepare(Problem((u,); dirichlet=[dirichlet(x -> x[1] - 0.5 * x[2]; on=boundary(:all))]))
    residual = loadform(u,
                        WeakForm(linear=q -> (uk=value(q.state, u); gk=field_gradient(q.state, u);
                                              TestChannels(-1.0, (1 + uk^2) * gk))))
    tangent = block(u, u,
                    WeakForm(bilinear=(q, trial) -> (uk=value(q.state, u);
                                                     gk=field_gradient(q.state, u);
                                                     TestChannels(0.0,
                                                                  (1 + uk^2) * trial.gradient +
                                                                  2uk * trial.value * gk))))
    n = Unfitted.active_unknowns(model.dofs)
    c = [0.5 * sin(3i) for i in 1:n]

    J = assemble_matrix(model, tangent; state=c, threaded=false)
    h = 1.0e-5
    e = zeros(n)
    finite_difference = zeros(n, n)
    for j in 1:n
        e[j] = h
        finite_difference[:, j] = (assemble_vector(model, residual; state=c + e, threaded=false) -
                                   assemble_vector(model, residual; state=c - e, threaded=false)) /
                                  2h
        e[j] = 0.0
    end
    @test !issymmetric(J)
    @test Matrix(J) ≈ finite_difference rtol = 1.0e-6

    # Threaded is bit-identical to serial with a state too, and a `Solution`
    # carrying the same coefficients is the same iterate as the raw vector.
    @test assemble_matrix(model, tangent; state=c, threaded=true) == J
    @test assemble_vector(model, residual; state=c, threaded=true) ==
          assemble_vector(model, residual; state=c, threaded=false)
    @test assemble_matrix(model, tangent; state=solution(model, c; method=:test), threaded=false) ==
          J
end

@testset "q.point is numbered per region list; nquadpoints(; on=) is its size" begin
    # `q.point` restarts at 1 for every `on=` region list, while the aggregate
    # `kind=:facet` counter sums across every cached list. A per-point array
    # sized by the aggregate and indexed by `q.point` would therefore leave the
    # tail untouched and alias one boundary's points onto another's;
    # `nquadpoints(model; on=…)` is the size that matches what a form sees.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    u = field(:u, V)
    lower = boundary(axis=1, side=:lower)
    upper = boundary(axis=1, side=:upper)

    seen_lo = Int[]
    seen_hi = Int[]
    probe(sink) = WeakForm(bilinear=(q, trial, c) -> 0.0,
                           linear=(q, c) -> (push!(sink, q.point); 0.0), symmetric=true,
                           component_aware=true)
    lo = loadform(u, probe(seen_lo); on=lower)
    hi = loadform(u, probe(seen_hi); on=upper)
    model = prepare(Problem((u,); blocks=(stiffness_block(u),), loads=(lo, hi)))
    assemble_vector(model, (lo, hi); threaded=false)

    n_lo = nquadpoints(model; on=lower)
    n_hi = nquadpoints(model; on=upper)
    @test n_lo > 0 && n_hi > 0
    @test sort(unique(seen_lo)) == collect(1:n_lo)
    @test sort(unique(seen_hi)) == collect(1:n_hi)

    # The aggregate counts both lists, so it is strictly larger than either
    # form's `q.point` range — the mismatch this keyword exists to close.
    @test nquadpoints(model; kind=:facet) == n_lo + n_hi
    @test n_lo < nquadpoints(model; kind=:facet)

    # The volume default is untouched by the new keywords.
    @test nquadpoints(model) == nquadpoints(model; kind=:volume)
end

@testset "nquadpoints(; on=) resolves the subdomain like boundary_integral" begin
    # Two disjoint squares with different cell counts, so each subdomain's face
    # carries a different number of quadrature points and a wrong resolution is
    # visible in the count alone.
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(3, 3), order=2)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2)),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=:u1),
                                       dirichlet(0.0; on=boundary(:all), field=:u2)]))

    all_faces = boundary(:all)
    n1 = nquadpoints(model; on=all_faces, field=:u1)
    n2 = nquadpoints(model; on=all_faces, field=:u2)
    @test n1 > 0 && n2 > 0
    @test n1 != n2                                       # distinct discretisations
    @test nquadpoints(model; kind=:facet) == n1 + n2     # the aggregate is the sum

    # Omitting `field` raises rather than answering for one subdomain, exactly
    # as `boundary_integral` does.
    @test_throws ArgumentError nquadpoints(model; on=all_faces)
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

@testset "a state of any eltype is read at its own precision" begin
    # `q.state` evaluates in the coefficients' number type promoted with the
    # model's, so a `BigFloat` iterate yields `BigFloat` values and gradients
    # instead of being rounded to `Float64` on the way in. Both walks sum the same
    # terms, so they agree to Float64 roundoff: measured 4.4e-16 on values of size
    # ≈ 2 and 4.2e-15 on gradients. The nonzero Dirichlet datum makes points near
    # the boundary also read constrained (`Float64`) values next to the active
    # ones.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=2)
    V = overlay(V, box((0.2, 0.25), (0.8, 0.75)); cells=(2, 2), order=3)
    u = field(:u, V)
    model = prepare(Problem((u,); dirichlet=[dirichlet(x -> 1.0 + x[1]; on=boundary(:all))]))
    c = [sin(3i) for i in 1:Unfitted.active_unknowns(model.dofs)]
    function walk(state)
        vals, grads = Any[], Any[]
        foreach_quadrature_point(model; state) do q
            push!(vals, value(q.state, u))
            push!(grads, field_gradient(q.state, :u))
            return nothing
        end
        return vals, grads
    end
    v64, g64 = walk(c)
    vbig, gbig = walk(BigFloat.(c))

    @test length(vbig) == length(v64) == nquadpoints(model)
    @test all(v -> v isa Float64, v64) && all(g -> g isa SVector{2,Float64}, g64)
    @test all(v -> v isa BigFloat, vbig)
    @test all(g -> g isa SVector{2,BigFloat}, gbig)
    @test all(isapprox.(vbig, v64; rtol=1.0e-14, atol=1.0e-14))
    @test all(isapprox.(gbig, g64; rtol=1.0e-14, atol=1.0e-14))

    # The checks above cannot see a `BigFloat` state rounded to `Float64` and
    # promoted back, because `BigFloat.(c)` is exactly representable in Float64.
    # A perturbation far below Float64 resolution can: the walk is affine in the
    # coefficients, so its response to `c + ε` must be ε times the response to
    # the all-ones direction, which a rounded state would lose entirely.
    ε = big(2.0)^-70
    vε, gε = walk(BigFloat.(c) .+ ε)
    v1, g1 = walk(ones(length(c)))
    v0, g0 = walk(zeros(length(c)))
    @test all(isapprox.((vε .- vbig) ./ ε, v1 .- v0; rtol=1.0e-10, atol=1.0e-10))
    @test all(isapprox.((gε .- gbig) ./ ε, g1 .- g0; rtol=1.0e-10, atol=1.0e-10))
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
    src = 1.0

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
        model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
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

        model = prepare(poisson(u; source=SVector(1.0, 2.0),
                                dirichlet=[dirichlet(SVector(0.0, 0.0); on=boundary(:all))]))
        _check_scatter_assemble!(model)
        # The component-unaware form leaves cross-component blocks in the
        # dense-block pattern but numerically zero; dropzeros! must remove
        # them, so the lower-triangle pattern is strictly larger than the
        # final lower-triangle nnz (proves the superset→drop pipeline), and the
        # matrix stores no entry coupling the two components.
        @test length(only(model.assembly.patterns).second.rowval) > nnz(tril(model.matrix))
        layout = Unfitted._field_layout(model.dofs, :u)
        ids(c) = filter(!iszero, layout.dofs.active_component[:, c]) .+ layout.offset
        @test !isempty(ids(1)) && !isempty(ids(2))
        @test nnz(model.matrix[ids(1), ids(2)]) == 0
        @test nnz(model.matrix[ids(2), ids(1)]) == 0
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
    # numeric scatter reads the assembled local block over the region's
    # sorted active dofs, so it is agnostic to whether the dof layer
    # produced linear-constraint pivots — this fixture exercises the
    # B-spline assembly path regardless.
    let V0 = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, basis=bspline()),
        V = overlay(V0, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3,
                    basis=bspline(continuity=1))

        model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
        _check_scatter_assemble!(model)
    end

    # FCM (immersed boundary via a physical domain / moment-fit quadrature).
    let phi = x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.35,
        p = physical_domain(phi; lipschitz=1.0, subcell_length_scale=1.0 / 2^4, max_depth=4),
        V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=p)

        model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
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

@testset "pivot assembly with nonzero Dirichlet data: threaded == serial, lift consistent" begin
    # The C¹ fixture above never reaches a linear-constraint pivot: an overlay's
    # own box faces are clamped ends, whose constraints resolve to strong
    # eliminations. A *masked* level below maximal continuity is the shape that
    # does. Its active/inactive faces emit multi-raw trace constraints, which the
    # resolver turns into pivots u_p = Σₖ wₖ u_{oₖ}, and assembly distributes
    # every pivot over its branches in both the test rows and the trial columns.
    # Nonzero Dirichlet data on the box faces makes some branches land on a raw
    # that carries a nonzero value, so the Dirichlet lift runs through the
    # expansion as well.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    mask = trues(6, 6)
    mask[4:6, 4:6] .= false                          # L-shaped active region
    V = space(Ω; cells=6, order=3, basis=bspline(; continuity=1), active=mask)
    u = field(:u, V)
    load = source_load(u; source=1.0)
    model = prepare(Problem((u,); blocks=(stiffness_block(u),), loads=(load,),
                            dirichlet=[dirichlet(x -> 1.0 + x[1] * x[2]; on=boundary(:all))]))

    # The fixture reaches the path under test: some pivot has a branch onto a raw
    # with a nonzero Dirichlet value.
    @test Unfitted.has_linear_constraints(model.dofs)
    layout = Unfitted._field_layout(model.dofs, :u).dofs
    @test any(eachindex(layout.raw_expansion)) do raw
        expansion = layout.raw_expansion[raw]
        return !isempty(expansion) &&
               expansion != [(raw, 1.0)] &&
               any(((other, _),) -> Unfitted.constrained_value(layout, other) != 0, expansion)
    end

    # Threaded assembly is bit-identical to serial on the pivot path too, and
    # stays so when a second threaded call reuses what the first one cached.
    assemble!(model; threaded=false)
    A, b = copy(model.matrix), copy(model.rhs)
    for _ in 1:2
        assemble!(model; threaded=true)
        @test model.matrix == A
        @test model.rhs == b
    end

    # The unsymmetric branch of the condensation folds every row and column in
    # full, and its threaded arena stores whole columns. On this symmetric
    # operator it must reproduce the symmetric branch up to summation order, and
    # threaded must still equal serial to the bit.
    unsymmetric = assemble_matrix(model, stiffness_block(u); symmetric=false, threaded=false)
    @test unsymmetric ≈ A rtol = 1.0e-14
    @test assemble_matrix(model, stiffness_block(u); symmetric=false, threaded=true) == unsymmetric

    # Anchors that do not depend on how the code expands a pivot. The lift check
    # below compares assembly against the state reconstruction, and both read a
    # pivot's branches from the dof layer, so a defect in that shared rule would
    # move both sides alike and pass. These three numbers were measured on the
    # assembler that distributed every emission through the expansions before it
    # condensed per region, with w = sin(3i). Rounding moves them by about 1e-16;
    # dropping or misweighting the Dirichlet branch of one pivot moves them by
    # 2.5–40 %.
    w = [sin(3i) for i in eachindex(b)]
    @test dot(w, b) ≈ -1.932281255352028 rtol = 1.0e-12
    @test sum(b) ≈ 26.955956630714347 rtol = 1.0e-12
    @test dot(w, A \ b) ≈ -0.18106058297953753 rtol = 1.0e-12

    # The lift agrees with the reconstruction. For any active coefficient vector
    # x, u_h = Σⱼ xⱼ φⱼ + u_g, where u_g is the Dirichlet lift that `dof_value`
    # reconstructs, pivot expansions included. Since b = F − a(u_g, ·),
    #
    #     a(u_h, φᵢ) = (A x)ᵢ + Fᵢ − bᵢ .
    #
    # The left side goes through the state reconstruction, the right side through
    # assembly's emission and lift, so they agree only if both expand every pivot
    # the same way. Measured at 3.2e-16 relative; leaving the pivots' Dirichlet
    # branches out of the reconstruction alone moves it to 4.4e-2.
    F = assemble_vector(model, load; threaded=false)
    @test norm(F - b) > norm(F)                     # the lift dominates the load here
    x = [sin(3i) for i in eachindex(b)]
    energy = WeakForm(linear=q -> TestChannels(0.0, field_gradient(q.state, u)))
    reconstructed = assemble_vector(model, loadform(u, energy); state=x, threaded=false)
    @test reconstructed ≈ A * x + F - b rtol = 1.0e-12
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

# ── Many distinct form types in one assembly ──────────────────────────────────

# Bytes allocated by one serial assembly call after a warm-up, measured behind a
# function barrier so the figure is the assembly's own rather than the boxing of
# test-scope locals.
function _serial_assembly_bytes(assemble, model, forms)
    assemble(model, forms; threaded=false)
    return @allocated assemble(model, forms; threaded=false)
end

@testset "four distinct form types assemble without dynamic dispatch" begin
    # Every `WeakForm` closure is its own type, so a problem's blocks and loads
    # are heterogeneous tuples, and inference unions at most three element types.
    # A fourth used to widen the hot loop's per-form variable to the abstract
    # `BlockForm` / `LoadForm`, so every channel evaluation dispatched dynamically
    # and allocated: 19× the bytes of assembling the same four blocks one at a
    # time here, 7.5 GB on a space-time Navier–Stokes Jacobian. At the default
    # optimisation level that path also rounded differently from the inlined one.
    #
    # Four scalar fields and four distinct (test, trial) pairs, so every matrix
    # entry and every rhs row is written by exactly one form. The combined
    # assembly must then equal the sum of the one-form assemblies to the bit, and
    # since it pays the region setup once where the separate calls pay it four
    # times, it can never legitimately allocate more than they do together.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2)
    a, b, c, d = field(:a, V), field(:b, V), field(:c, V), field(:d, V)
    model = prepare(Problem((a, b, c, d)))
    forms = (WeakForm(bilinear=(q, trial) -> TestChannels(0.0, trial.gradient), linear=q -> 1.0),
             WeakForm(bilinear=(q, trial) -> trial.gradient[1], linear=q -> q.x[1]),
             WeakForm(bilinear=(q, trial) -> -trial.gradient[2],
                      linear=q -> TestChannels(0.0, q.x)),
             WeakForm(bilinear=(q, trial) -> trial.value, linear=q -> q.x[2]^2))
    blocks = (block(a, a, forms[1]), block(b, c, forms[2]), block(c, b, forms[3]),
              block(d, d, forms[4]))
    loads = (loadform(a, forms[1]), loadform(b, forms[2]), loadform(c, forms[3]),
             loadform(d, forms[4]))

    for (assemble, combined_forms) in ((assemble_matrix, blocks), (assemble_vector, loads))
        combined = assemble(model, combined_forms; threaded=false)
        @test combined == sum(form -> assemble(model, (form,); threaded=false), combined_forms)
        @test assemble(model, combined_forms; threaded=true) == combined

        separate_bytes = sum(form -> _serial_assembly_bytes(assemble, model, (form,)),
                             combined_forms)
        @test _serial_assembly_bytes(assemble, model, combined_forms) ≤ separate_bytes
    end
end

# ── Concurrent assembly calls on one model ────────────────────────────────────

# One task per operator index; each task assembles `rounds` operators, cycling
# through the list from a staggered start, so at any moment the concurrent calls
# ask for different operators, and therefore for different cached sparsity
# patterns. Returns `(operator index, matrix)` pairs.
function _concurrent_assembly(model, operators; threaded::Bool, rounds::Int=8)
    tasks = map(eachindex(operators)) do k
        return Threads.@spawn [let i = mod1(k + r, length(operators))
                                   (i, assemble_matrix(model, operators[i]; threaded))
                               end
                               for r in 1:rounds]
    end
    return reduce(vcat, fetch.(tasks))
end

@testset "concurrent serial assembly on one model matches sequential assembly" begin
    # Several tasks may assemble operators of one prepared model at the same time,
    # for instance a time stepper building its mass and stiffness matrices in
    # parallel. No call may disturb another: whatever a call caches on the model
    # must be read-only to the others or private to the call. The four operators
    # cover two region sets, the volume plan and a facet target the problem
    # prepared, so the concurrent calls disagree about the pattern they need.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2)
    V = overlay(V, box((0.2, 0.25), (0.8, 0.75)); cells=(2, 2), order=3)
    u = field(:u, V)
    robin = block(u, u, mass_form(coefficient=4.0); on=boundary(axis=1, side=:upper))
    model = prepare(Problem((u,); blocks=(stiffness_block(u), robin),
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))]))
    operators = ((mass_block(u),), (stiffness_block(u),), (stiffness_block(u), robin), (robin,))
    reference = [assemble_matrix(model, operator; threaded=false) for operator in operators]

    results = _concurrent_assembly(model, operators; threaded=false)
    @test length(results) == 8 * length(operators)
    @test all(A == reference[i] for (i, A) in results)
end

@testset "concurrent threaded assembly on one model matches sequential assembly" begin
    # The threaded variant of the test above. Each threaded call parks its
    # regions' local systems in an arena before summing them, and that scratch
    # is pooled on the model between calls, so two calls running at once must
    # never be handed the same arena or the same workspace. The earlier
    # assembler pooled its arenas without handing them out one call at a time,
    # and one call's regions then overwrote another's slices between its two
    # phases: under this schedule 6–8 of 128 matrices came out wrong at 6
    # threads, and 16 of 128 at 1 thread, where the tasks interleave at every
    # `@sync`.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2)
    V = overlay(V, box((0.2, 0.25), (0.8, 0.75)); cells=(2, 2), order=3)
    u = field(:u, V)
    robin = block(u, u, mass_form(coefficient=4.0); on=boundary(axis=1, side=:upper))
    model = prepare(Problem((u,); blocks=(stiffness_block(u), robin),
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))]))
    operators = ((mass_block(u),), (stiffness_block(u),), (stiffness_block(u), robin), (robin,))
    reference = [assemble_matrix(model, operator; threaded=false) for operator in operators]

    results = _concurrent_assembly(model, operators; threaded=true, rounds=32)
    @test length(results) == 32 * length(operators)
    @test count(((i, A),) -> A != reference[i], results) == 0
end

# ── Assembly cache ────────────────────────────────────────────────────────────

@testset "threaded one-shot on= assembly survives garbage collection" begin
    # A block whose `on=` target the problem never names has no prepared region
    # list; the first call resolves one. Whatever a threaded call derives from
    # that list and keeps — its regions' dofs, the arena layout, the pattern —
    # must stay attached to that list and to nothing else. Keyed by the list's
    # `objectid`, as it once was, it was not: a list dropped after its call and
    # collected left its address free, a later call's list could be allocated
    # there, and that list was then assembled with the other target's layout.
    # Here 10–12 of 200 matrices came out wrong that way, at 1 thread as at 6.
    #
    # Between calls the loop collects the young generation and allocates a
    # varying number of small arrays, as any program does between two
    # assemblies; the varying offset is what lands a fresh list on a dead
    # list's address.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(6, 6), order=2)
    u = field(:u, V)
    model = prepare(Problem((u,); blocks=(stiffness_block(u),),
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))]))
    lo = block(u, u, mass_form(coefficient=1.0); on=boundary(axis=2, side=:lower))
    hi = block(u, u, mass_form(coefficient=3.0); on=boundary(axis=2, side=:upper))
    reference = assemble_matrix(model, (lo, hi); threaded=false)
    @test nnz(reference) > 0

    wrong = 0
    for call in 1:200
        GC.gc(false)
        ballast = [Int[] for _ in 1:mod(97*call, 257)]
        wrong += assemble_matrix(model, (lo, hi); threaded=true) != reference
        empty!(ballast)
    end
    @test wrong == 0
end

@testset "the assembly cache stays bounded" begin
    # Region lists of targets `prepare` resolved are kept for the model's
    # lifetime. Those of targets it never saw, and the sparsity patterns, are
    # kept in small most-recently-used caches (8 lists, 4 patterns), so a loop
    # over ever new one-shot targets cannot grow the model without bound, and a
    # one-shot entry falling out never takes a prepared list with it.
    V = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(2, 2, 2), order=1)
    u = field(:u, V)
    prepared = boundary(axis=1, side=:lower)
    model = prepare(Problem((u,); blocks=(stiffness_block(u),),
                            loads=(neumann(u, 1.0; on=prepared),),
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:upper))]))
    assemble!(model)
    space_ = model.problem.space
    @test haskey(model.assembly.lists, (nothing, space_))
    @test haskey(model.assembly.lists, (prepared, space_))

    # 20 distinct selectors, none of them named by the problem: the 12 edges and
    # the 8 corners of the cube.
    side(axis, upper) = (axis=axis, side=upper ? :upper : :lower)
    edges = [boundary(side(a, s), side(b, t)) for (a, b) in ((1, 2), (1, 3), (2, 3))
             for s in (false, true) for t in (false, true)]
    corners = [boundary(side(1, s), side(2, t), side(3, r)) for s in (false, true)
               for t in (false, true) for r in (false, true)]
    selectors = vcat(edges, corners)
    @test length(unique(selectors)) == 20
    @test !any(s -> isequal(s, prepared), selectors)

    form = mass_form(coefficient=2.0)
    for selector in selectors
        A = assemble_matrix(model, block(u, u, form; on=selector); threaded=true)
        @test A == assemble_matrix(model, block(u, u, form; on=selector); threaded=false)
        b = assemble_vector(model, neumann(u, 1.0; on=selector); threaded=true)
        @test b == assemble_vector(model, neumann(u, 1.0; on=selector); threaded=false)
    end
    @test length(model.assembly.oneshot) <= 8
    @test length(model.assembly.patterns) <= 4
    @test length(model.assembly.lists) == 2
    @test haskey(model.assembly.lists, (nothing, space_))
    @test haskey(model.assembly.lists, (prepared, space_))

    # Every entry still assembles what it did before: the problem's own system,
    # whose pattern the loop above evicted, is rebuilt bit for bit.
    A, b = copy(model.matrix), copy(model.rhs)
    assemble!(model)
    @test model.matrix == A
    @test model.rhs == b
end

# A fresh model of one order-3 field on `cells × cells` square cells, and three
# mass blocks on facet targets of growing size — one side, three sides, the
# whole boundary — so a call over them runs three matrix passes, each with more
# regions than the one before.
function _growing_facet_passes(cells::Int)
    u = field(:u, space(box((0.0, 0.0), (1.0, 1.0)); cells=(cells, cells), order=3))
    form = mass_form(coefficient=1.0)
    blocks = (block(u, u, form; on=boundary(axis=1, side=:lower)),
              block(u, u, form; on=boundary(:all; except=(axis=2, side=:upper))),
              block(u, u, form; on=boundary(:all)))
    return prepare(Problem((u,))), blocks
end

# Bytes allocated by the first `assemble_matrix` call on a fresh such model, and
# the model, so the caller can read what the call left pooled on it.
function _cold_matrix_bytes(cells::Int, threaded::Bool)
    model, blocks = _growing_facet_passes(cells)
    return (@allocated assemble_matrix(model, blocks; threaded)), model
end

@testset "a threaded call allocates one arena, sized for its largest pass" begin
    # The threaded driver parks every region's local system in a packed arena
    # before summing it (`Σ n(n+1)/2` entries on a symmetric pass), and every
    # pass of a call writes a prefix of the same arena pair. Sizing that pair
    # once, for the largest pass, before the first pass runs is what keeps a
    # threaded call within one arena of the serial call. Grown pass by pass
    # instead, the arena was reallocated at every pass larger than all before it,
    # each outgrown buffer was dead allocation, and `resize!` rounded the last one
    # up past the size it needed: on this fixture that waste came to 444 KB
    # beside a 233 KB arena, at every thread count.
    #
    # The bound: a cold threaded call allocates at most what the cold serial call
    # does, plus the arena pair it leaves pooled, plus per-task scratch — a
    # workspace for every task but the first, and per task a few kilobytes for
    # its buffers to grow to the region size and for its task objects. That last
    # part measured 15 KB at 1 thread and 46 KB at 6; 16 KB per task plus 16 KB
    # bounds it.
    for threaded in (false, true)
        _cold_matrix_bytes(2, threaded)                 # compile both drivers first
    end
    serial, _ = _cold_matrix_bytes(48, false)
    threaded, model = _cold_matrix_bytes(48, true)
    cache = model.assembly
    arena = sizeof(cache.arena) + sizeof(cache.rhs_arena)
    @test arena > 0
    Unfitted._assembly_workspace(model)
    workspace = @allocated Unfitted._assembly_workspace(model)
    tasks = Threads.nthreads()
    @test threaded - serial ≤ arena + (tasks - 1) * workspace + 16_384 * (tasks + 1)
end

@testset "an L2 transfer keeps the target's own sparsity pattern" begin
    # The default L² transfer assembles the target mass through the target's
    # own assembly cache. It must add its pattern beside the problem's rather
    # than replace it, so the next solve on the target rebuilds nothing; on a
    # symmetric volume-only problem the two are one and the same pattern.
    omega = box((0.0, 0.0), (1.0, 1.0))
    source = prepare(poisson(space(omega; cells=(3, 3), order=2); source=1.0,
                             dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    solution = solve!(source)

    # A Robin face makes the problem's matrix passes differ from the mass's.
    V = space(omega; cells=(4, 4), order=2)
    u = field(:u, V)
    robin = block(u, u, mass_form(coefficient=2.0); on=boundary(axis=1, side=:upper))
    target = prepare(Problem((u,); blocks=(stiffness_block(u), robin),
                             loads=(source_load(u; source=1.0),),
                             dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))]))
    solve!(target)
    own = only(target.assembly.patterns).second
    transfer(solution, source, target)
    @test length(target.assembly.patterns) == 2
    @test any(entry -> entry.second === own, target.assembly.patterns)
    assemble!(target)
    @test first(target.assembly.patterns).second === own

    volume_only = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    solve!(volume_only)
    own = only(volume_only.assembly.patterns).second
    transfer(solution, source, volume_only)
    @test only(volume_only.assembly.patterns).second === own
end
