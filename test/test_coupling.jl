# Multi-domain coupling. Phase 0 covers the *product space*: a single
# `Problem` whose fields live over independent `Space`s (own grid, basis,
# order, physical domain). With no interface term the coupled system must be
# exactly the direct sum of the per-subdomain systems — the invariant that
# proves the dof offsets, per-space integration plans, and level-id
# namespacing compose without cross-talk. Interface coupling itself (the
# `couple` verb and the two-sided interface region) is exercised separately.

using Unfitted
using Unfitted: integration_plan, integration_plans, problem_spaces
using LinearAlgebra
using SparseArrays
using StaticArrays
using Test

@testset "product space of disjoint domains == block diagonal" begin
    # Two disjoint unit squares, deliberately different resolutions, orders,
    # and even source terms, so nothing about the two discretisations lines up.
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=2)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=3, order=3)
    src1(x) = 1.0
    src2(x) = 2.0
    bc(name) = [dirichlet(0.0; on=boundary(:all), field=name)]

    # Standalone single-space references.
    m1 = prepare(poisson(field(:u1, V1); source=src1, dirichlet=bc(:u1)))
    m2 = prepare(poisson(field(:u2, V2); source=src2, dirichlet=bc(:u2)))
    assemble!(m1)
    assemble!(m2)
    A1, b1 = m1.matrix, m1.rhs
    A2, b2 = m2.matrix, m2.rhs
    n1, n2 = size(A1, 1), size(A2, 1)

    # One coupled Problem over both spaces, no interface term.
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    prob = Problem((u1, u2);
                   blocks=(stiffness_block(u1), stiffness_block(u2)),
                   loads=(source_load(u1; source=src1), source_load(u2; source=src2)),
                   dirichlet=[bc(:u1); bc(:u2)])
    model = prepare(prob)

    # The model tracks two independent subdomain discretisations.
    @test length(problem_spaces(model.problem)) == 2
    @test length(integration_plans(model)) == 2
    @test Unfitted.active_unknowns(model) == n1 + n2

    assemble!(model)
    A, b = model.matrix, model.rhs

    @test size(A) == (n1 + n2, n1 + n2)
    @test A[1:n1, 1:n1] ≈ A1
    @test A[(n1 + 1):end, (n1 + 1):end] ≈ A2
    @test iszero(A[1:n1, (n1 + 1):end])          # no cross coupling without an interface
    @test iszero(A[(n1 + 1):end, 1:n1])
    @test b[1:n1] ≈ b1
    @test b[(n1 + 1):end] ≈ b2

    # The full solve is the concatenation of the two independent solves.
    s = solve!(model)
    s1 = solve!(m1)
    s2 = solve!(m2)
    @test s.coefficients[1:n1] ≈ s1.coefficients
    @test s.coefficients[(n1 + 1):end] ≈ s2.coefficients
end

@testset "fields sharing one space keep the single-domain path" begin
    # A two-field problem over ONE space must still behave as before: one
    # integration plan, one level-id block, and the existing same-space
    # coupling machinery unchanged.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=2)
    a = field(:a, V)
    b = field(:b, V)
    prob = Problem((a, b);
                   blocks=(stiffness_block(a), mass_block(b)),
                   loads=(source_load(a; source=x -> 1.0),),
                   dirichlet=[dirichlet(0.0; on=boundary(:all), field=:a)])
    model = prepare(prob)
    @test length(problem_spaces(model.problem)) == 1
    @test length(integration_plans(model)) == 1
    assemble!(model)
    @test model.matrix !== nothing
end

@testset "positional mutators are guarded on multi-domain models" begin
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=3, order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=3, order=1)
    prob = Problem((field(:u1, V1), field(:u2, V2));
                   blocks=(stiffness_block(field(:u1, V1)), stiffness_block(field(:u2, V2))),
                   dirichlet=[dirichlet(0.0; on=boundary(:all), field=:u1),
                              dirichlet(0.0; on=boundary(:all), field=:u2)])
    model = prepare(prob)
    @test_throws ArgumentError move!(model; level=1, to=box((0.1, 0.1), (0.9, 0.9)))
    @test_throws ArgumentError activate!(model; level=1, cells=[CartesianIndex(1, 1)])
end

@testset "per-point iteration is guarded on multi-domain models" begin
    # foreach_quadrature_point / QuadField walk a single subdomain plan with a
    # single-space workspace, so a coupled model would silently visit only the
    # first subdomain — guarded rather than wrong.
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=3, order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=3, order=1)
    model = prepare(Problem((field(:u1, V1), field(:u2, V2));
                            blocks=(stiffness_block(field(:u1, V1)),
                                    stiffness_block(field(:u2, V2)))))
    @test_throws ArgumentError foreach_quadrature_point(q -> nothing, model)
    @test_throws ArgumentError QuadField{Float64}(model)
end

@testset "sharing one single-sided on= mesh across subdomains is rejected" begin
    # A single BoundaryMesh object used as the `on=` target for weak forms on two
    # different subdomains would assemble only the first subdomain's contribution
    # (the pass serves one space). That silent drop is rejected at assembly.
    V1 = space(box((0.0, 0.0), (1.0, 0.5)); cells=(4, 2), order=2)
    V2 = space(box((0.0, 0.5), (1.0, 1.0)); cells=(4, 2), order=2)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    Γ = polyline_mesh([SVector(0.0, 0.5), SVector(1.0, 0.5)])
    model = prepare(Problem((u1, u2);
                            blocks=(stiffness_block(u1), stiffness_block(u2),
                                    block(u1, u1, mass_form(coefficient=1.0); on=Γ),
                                    block(u2, u2, mass_form(coefficient=1.0); on=Γ))))
    @test_throws ArgumentError assemble!(model)
    # Distinct mesh objects (one per subdomain) are the correct, accepted form.
    Γ1 = polyline_mesh([SVector(0.0, 0.5), SVector(1.0, 0.5)])
    Γ2 = polyline_mesh([SVector(0.0, 0.5), SVector(1.0, 0.5)])
    ok = prepare(Problem((u1, u2);
                         blocks=(stiffness_block(u1), stiffness_block(u2),
                                 block(u1, u1, mass_form(coefficient=1.0); on=Γ1),
                                 block(u2, u2, mass_form(coefficient=1.0); on=Γ2))))
    assemble!(ok)
    @test ok.matrix !== nothing
end

# ── Interface coupling: the two-sided region + couple verb ────────────────────

@testset "interface integral matches the trusted surface path (same space)" begin
    # Both coupled fields over ONE space is a degenerate but exact reference:
    # side-A and side-B parents are identical, so `couple`'s diagonal block must
    # equal the surface-mass block the existing `on=BoundaryMesh` path assembles.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=2)
    a = field(:a, V)
    b = field(:b, V)
    Γ = polyline_mesh([SVector(0.0, 0.5), SVector(1.0, 0.5)])
    β = 3.0
    model = prepare(Problem((a, b); blocks=(stiffness_block(a), stiffness_block(b))))

    iface = interface(a, b, Γ)
    K_iface = assemble_matrix(model, block(a, a, mass_form(coefficient=β); on=iface))
    K_surf = assemble_matrix(model, block(a, a, mass_form(coefficient=β); on=Γ))
    cb = couple(a, b, Γ, mass_form(coefficient=β))
    K_ab = assemble_matrix(model, cb[3]; symmetric=false)   # block(a, b, -form; on=iface)

    n = Unfitted.active_unknowns(model.dofs)
    na = n ÷ 2
    @test length(cb) == 4
    @test K_iface ≈ K_surf                                   # diagonal block == surface mass
    @test K_ab[1:na, (na + 1):n] ≈ -K_surf[1:na, 1:na]       # off-diagonal == negated mass
    @test iszero(K_ab[1:na, 1:na])
end

# Two stacked subdomains coupled by a penalty interface at y = 0.5, with
# Dirichlet data `g` on every OUTER face (the interface faces are left free).
function _solve_stacked(β, g; nx1=4, ny1=2, nx2=4, ny2=2, sym=false)
    V1 = space(box((0.0, 0.0), (1.0, 0.5)); cells=(nx1, ny1), order=2)
    V2 = space(box((0.0, 0.5), (1.0, 1.0)); cells=(nx2, ny2), order=2)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    Γ = polyline_mesh([SVector(0.0, 0.5), SVector(1.0, 0.5)])
    src(x) = 2 * (x[1] - x[1]^2 + x[2] - x[2]^2)             # −Δ[x(1-x)y(1-y)]
    cpl = couple(u1, u2, Γ, mass_form(coefficient=β))
    prob = Problem((u1, u2);
                   blocks=(stiffness_block(u1), stiffness_block(u2), cpl...),
                   loads=(source_load(u1; source=src), source_load(u2; source=src)),
                   dirichlet=[dirichlet(g; on=boundary(axis=2, side=:lower), field=:u1),
                              dirichlet(g; on=boundary(axis=1, side=:lower), field=:u1),
                              dirichlet(g; on=boundary(axis=1, side=:upper), field=:u1),
                              dirichlet(g; on=boundary(axis=2, side=:upper), field=:u2),
                              dirichlet(g; on=boundary(axis=1, side=:lower), field=:u2),
                              dirichlet(g; on=boundary(axis=1, side=:upper), field=:u2)],
                   symmetric=sym)
    model = prepare(prob)
    return model, solve!(model), u1, u2
end

const _P1 = [SVector(0.3, 0.2), SVector(0.5, 0.1), SVector(0.7, 0.4)]  # interior of Ω₁
const _P2 = [SVector(0.3, 0.8), SVector(0.5, 0.9), SVector(0.7, 0.6)]  # interior of Ω₂
function _maxerr(exact, model, sol, u1, u2)
    e1 = maximum(abs(value(sol, model, u1, p) - exact(p)) for p in _P1)
    e2 = maximum(abs(value(sol, model, u2, p) - exact(p)) for p in _P2)
    return max(e1, e2)
end

@testset "penalty coupling reproduces the exact zero-flux solution" begin
    # u = x(1-x)y(1-y) has ∂u/∂y = 0 on y = 0.5, so the penalty coupling is
    # flux-consistent and reproduces u to machine precision for ANY β, on both
    # matching and non-matching grids — a strong patch test of the two-sided
    # non-matching evaluation.
    u0(x) = x[1] * (1 - x[1]) * x[2] * (1 - x[2])
    @test _maxerr(u0, _solve_stacked(1.0e3, u0)...) < 1.0e-10
    @test _maxerr(u0, _solve_stacked(1.0e3, u0; nx1=5, ny1=3, nx2=7, ny2=4)...) < 1.0e-10
end

@testset "penalty coupling converges to exact for nonzero interface flux" begin
    # Add a linear y term: ∂u/∂y = 1 on the interface, so the penalty method now
    # carries a genuine O(1/β) consistency error that shrinks as β grows.
    uy(x) = x[1] * (1 - x[1]) * x[2] * (1 - x[2]) + x[2]
    e1 = _maxerr(uy, _solve_stacked(1.0e1, uy)...)
    e3 = _maxerr(uy, _solve_stacked(1.0e3, uy)...)
    e5 = _maxerr(uy, _solve_stacked(1.0e5, uy)...)
    @test e3 < e1
    @test e5 < e3
    @test e5 < 1.0e-3
end

@testset "symmetric fast path agrees with full assembly for coupling" begin
    uy(x) = x[1] * (1 - x[1]) * x[2] * (1 - x[2]) + x[2]
    ma, sa, u1a, u2a = _solve_stacked(1.0e4, uy; sym=false)
    mb, sb, u1b, u2b = _solve_stacked(1.0e4, uy; sym=true)
    @test maximum(abs(value(sa, ma, u1a, p) - value(sb, mb, u1b, p)) for p in _P1) < 1.0e-9
    @test maximum(abs(value(sa, ma, u2a, p) - value(sb, mb, u2b, p)) for p in _P2) < 1.0e-9
end

@testset "interface regions are visible in diagnostics" begin
    # A coupled model's interface regions must be inspectable, like facet /
    # surface regions: counted in diagnostics and summable via nquadpoints.
    model, = _solve_stacked(1.0e3, x -> 0.0)
    @test diagnostics(model).interface_region_count > 0
    @test nquadpoints(model; kind=:interface) > 0
    @test nquadpoints(model; kind=:interface) ==
          sum(length(r.weights) for list in values(model.interface_regions) for r in list)
    @test_throws ArgumentError nquadpoints(model; kind=:nonsense)
end

# Symmetric weighted-Nitsche coupling via a two-sided InterfaceForm (κ=1, w=½).
# The interface normal must point a→b; the segment is reversed to give (0,+1).
function _solve_nitsche(γ, exact; nx1=4, ny1=2, nx2=4, ny2=2, sym=true)
    V1 = space(box((0.0, 0.0), (1.0, 0.5)); cells=(nx1, ny1), order=2)
    V2 = space(box((0.0, 0.5), (1.0, 1.0)); cells=(nx2, ny2), order=2)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    src(x) = 2 * (x[1] - x[1]^2 + x[2] - x[2]^2)
    Γ = polyline_mesh([SVector(1.0, 0.5), SVector(0.0, 0.5)])   # normal (0,+1) = a→b
    β = γ / (0.5 / max(ny1, ny2))
    nitsche = InterfaceForm() do q, sides, trial, _tc
        n = q.normal
        sv, su = jump_sign(sides.test), jump_sign(sides.trial)
        TestChannels(-0.5 * sv * dot(trial.gradient, n) + β * sv * su * trial.value,
                     -0.5 * su * trial.value .* n)
    end
    prob = Problem((u1, u2);
                   blocks=(stiffness_block(u1), stiffness_block(u2), couple(u1, u2, Γ, nitsche)...),
                   loads=(source_load(u1; source=src), source_load(u2; source=src)),
                   dirichlet=[dirichlet(exact; on=boundary(axis=2, side=:lower), field=:u1),
                              dirichlet(exact; on=boundary(axis=1, side=:lower), field=:u1),
                              dirichlet(exact; on=boundary(axis=1, side=:upper), field=:u1),
                              dirichlet(exact; on=boundary(axis=2, side=:upper), field=:u2),
                              dirichlet(exact; on=boundary(axis=1, side=:lower), field=:u2),
                              dirichlet(exact; on=boundary(axis=1, side=:upper), field=:u2)],
                   symmetric=sym)
    model = prepare(prob)
    return model, solve!(model), u1, u2
end

@testset "two-sided InterfaceForm gives consistent Nitsche coupling" begin
    # ∂u/∂y = 1 at the interface (nonzero flux): a CONSISTENT coupling reproduces
    # it to machine precision for any β above coercivity — penalty cannot.
    uy(x) = x[1] * (1 - x[1]) * x[2] * (1 - x[2]) + x[2]
    for γ in (10.0, 1.0e3)
        @test _maxerr(uy, _solve_nitsche(γ, uy)...) < 1.0e-9
    end
    # non-matching grids: still exact (mesh-relationship-independent consistency).
    @test _maxerr(uy, _solve_nitsche(1.0e2, uy; nx1=5, ny1=3, nx2=7, ny2=4)...) < 1.0e-8
end

@testset "component-aware InterfaceForm couples a 2-component field" begin
    # Two stacked 2-component subdomains coupled by a diagonal (component-wise)
    # jump penalty ∫ β Σᵢ [uᵢ][vᵢ], written as a component-aware InterfaceForm
    # that reads the trailing test-component argument. The manufactured
    # displacement u = (φ, 2φ), φ = x(1-x)y(1-y), vanishes on every outer face
    # (homogeneous Dirichlet) and has ∂φ/∂y = 0 on the interface, so the penalty
    # is flux-consistent and reproduces u to machine precision — exercising the
    # `test_component` threading the coupling sugar now forwards.
    φ(x) = x[1] * (1 - x[1]) * x[2] * (1 - x[2])
    uex(x) = SVector(φ(x), 2φ(x))
    s(x) = 2 * (x[1] - x[1]^2 + x[2] - x[2]^2)   # −Δφ
    V1 = space(box((0.0, 0.0), (1.0, 0.5)); cells=(4, 2), order=2)
    V2 = space(box((0.0, 0.5), (1.0, 1.0)); cells=(4, 2), order=2)
    u1 = field(:u1, V1; components=2)
    u2 = field(:u2, V2; components=2)
    Γ = polyline_mesh([SVector(0.0, 0.5), SVector(1.0, 0.5)])
    β = 1.0e3
    vpen = InterfaceForm() do q, sides, trial, tc
        # diagonal-in-components: contribute to test row `tc` only from the
        # matching trial component, so ⟦uᵢ⟧ couples to ⟦vᵢ⟧ alone.
        s = jump_sign(sides.test) * jump_sign(sides.trial)
        TestChannels(trial.component == tc ? β * s * trial.value : 0.0, SVector(0.0, 0.0))
    end
    zero_bc(name, ax, sd) = dirichlet(SVector(0.0, 0.0); on=boundary(axis=ax, side=sd), field=name)
    model = prepare(Problem((u1, u2);
                            blocks=(stiffness_block(u1), stiffness_block(u2),
                                    couple(u1, u2, Γ, vpen)...),
                            loads=(source_load(u1; source=x -> SVector(s(x), 2s(x))),
                                   source_load(u2; source=x -> SVector(s(x), 2s(x)))),
                            dirichlet=[zero_bc(:u1, 2, :lower), zero_bc(:u1, 1, :lower),
                                       zero_bc(:u1, 1, :upper), zero_bc(:u2, 2, :upper),
                                       zero_bc(:u2, 1, :lower), zero_bc(:u2, 1, :upper)]))
    # Correctness is checked on a SERIAL assembly so the manufactured-solution
    # test is deterministic in every CI cell. The shipped interface pass runs
    # threaded, but a Julia `--code-coverage` + multithreading codegen artifact can
    # nondeterministically corrupt this two-sided *vector* block under coverage
    # with ≥2 threads (see the `couple` docstring) — exactly the coverage CI cell.
    # Pinning correctness on the serial assembly keeps it green while still
    # exercising the full coupling mechanics.
    assemble!(model; threaded=false)
    sol = solve!(model)
    e1 = maximum(norm(value(sol, model, u1, p) - uex(p)) for p in _P1)
    e2 = maximum(norm(value(sol, model, u2, p) - uex(p)) for p in _P2)
    @test max(e1, e2) < 1.0e-9

    # Where threaded interface assembly is reliable — no coverage instrumentation,
    # or a single thread — verify it reproduces the serial interface block exactly.
    if Base.JLOptions().code_coverage == 0 || Threads.nthreads() == 1
        serial_matrix = copy(model.matrix)
        assemble!(model; threaded=true)
        @test model.matrix ≈ serial_matrix
    end
end
