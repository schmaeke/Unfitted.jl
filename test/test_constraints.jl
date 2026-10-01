using StaticArrays
using LinearAlgebra

@testset "constraints scaffold" begin
    bc = dirichlet(0.0; on=boundary(:all))
    side = boundary(axis=2, side=:upper)

    @test bc.value == 0.0
    @test bc.boundary.selector == :all
    @test bc.component === nothing
    @test side.selector == :sides
    @test side.sides == [(2, :upper)]
    @test dirichlet(0.1; on=side, component=2).component == 2
end

@testset "BoundarySelector has value semantics" begin
    # A selector is a *description* of a facet, not an object with an identity:
    # two separately-constructed selectors naming the same facet must be equal,
    # hash equally, and be one key in a Dict. Without this, every Dict keyed on
    # a selector is identity-keyed, and the identity-based and value-based
    # halves of facet-region resolution disagree about which selectors are the
    # same one (which is how a coupled model silently drops a Neumann load).
    a = boundary(axis=1, side=:upper)
    b = boundary(axis=1, side=:upper)
    @test a !== b                       # genuinely distinct objects
    @test a == b
    @test isequal(a, b)
    @test hash(a) == hash(b)
    @test length(Dict(a => 1, b => 2)) == 1

    # ...and selectors naming different facets stay distinct.
    @test boundary(axis=1, side=:upper) != boundary(axis=1, side=:lower)
    @test boundary(axis=1, side=:upper) != boundary(axis=2, side=:upper)
    @test boundary(:all) != boundary(axis=1, side=:upper)
    @test length(Dict(boundary(:all) => 1, boundary(axis=1, side=:lower) => 2)) == 2

    # Multi-side selectors compare element-wise, and order is part of the value.
    corner = boundary((axis=1, side=:lower), (axis=2, side=:lower))
    @test corner == boundary((axis=1, side=:lower), (axis=2, side=:lower))
    @test hash(corner) == hash(boundary((axis=1, side=:lower), (axis=2, side=:lower)))
    @test corner != boundary((axis=1, side=:lower))

    # `sides` is a Vector, so hashing must be by value, not by the vector's
    # identity — a fresh selector must find a cache entry a prepared model made.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    model = prepare(poisson(V; source=0.0,
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:upper))]))
    @test haskey(model.facet_regions, (boundary(axis=1, side=:upper), model.problem.space))
end

@testset "per-component Dirichlet frees the other components" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    u = field(:u, V; components=2)
    bottom = boundary(axis=2, side=:lower)
    top = boundary(axis=2, side=:upper)

    # only u_y constrained on top (u_x free); both fixed on the bottom
    per_component = prepare(poisson(u; source=SVector(0.0, 0.0),
                                    dirichlet=[dirichlet(SVector(0.0, 0.0); on=bottom, field=u),
                                               dirichlet(0.1; on=top, field=u, component=2)]))
    all_components = prepare(poisson(u; source=SVector(0.0, 0.0),
                                     dirichlet=[dirichlet(SVector(0.0, 0.0); on=bottom, field=u),
                                                dirichlet(SVector(0.0, 0.1); on=top, field=u)]))

    # constraining one component leaves the top u_x dofs active
    @test Unfitted.active_unknowns(per_component.dofs) >
          Unfitted.active_unknowns(all_components.dofs)

    solution = solve!(per_component)
    @test isposdef(Symmetric(Matrix(per_component.matrix)))
    @test value(solution, per_component, (0.5, 1.0), 2) ≈ 0.1 atol = 1.0e-10   # u_y prescribed on top
    @test value(solution, per_component, (0.5, 0.5), 2) ≈ 0.05 atol = 1.0e-10  # linear in y
    @test value(solution, per_component, (0.5, 0.0), 1) ≈ 0.0 atol = 1.0e-10   # u_x fixed on bottom
end

@testset "1D conforming dof sharing and physical constraints" begin
    V = space(box((0.0,), (1.0,)); cells=2, order=2)
    layout = Unfitted.dof_layout(V; dirichlet=[dirichlet(0.0; on=boundary(:all))])

    cell1 = Unfitted.cell_dofs(layout, 1, CartesianIndex(1))
    cell2 = Unfitted.cell_dofs(layout, 1, CartesianIndex(2))

    @test Unfitted.raw_dof_count(layout) == 5
    @test Unfitted.active_unknowns(layout) == 3
    @test cell1[2] == cell2[1]
    @test Unfitted.constraint_kind(layout, cell1[1]) == :dirichlet
    @test Unfitted.constraint_kind(layout, cell2[2]) == :dirichlet
    @test Unfitted.constraint_kind(layout, cell1[3]) == :free
end

@testset "boundary L2 projection assigns constrained values" begin
    V = space(box((0.0,), (1.0,)); cells=2, order=1)
    layout = Unfitted.dof_layout(V; dirichlet=[dirichlet(x -> 1 + x[1]; on=boundary(:all))])
    cell1 = Unfitted.cell_dofs(layout, 1, CartesianIndex(1))
    cell2 = Unfitted.cell_dofs(layout, 1, CartesianIndex(2))

    @test Unfitted.constrained_value(layout, cell1[1]) ≈ 1.0
    @test Unfitted.constrained_value(layout, cell2[2]) ≈ 2.0
    @test Unfitted.constrained_value(layout, cell1[2]) ≈ 0.0
end

@testset "2D tensor-product dof layout" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 1), order=2)
    layout = Unfitted.dof_layout(V; dirichlet=[dirichlet(0.0; on=boundary(:all))])
    left = Unfitted.cell_dofs(layout, 1, CartesianIndex(1, 1))
    right = Unfitted.cell_dofs(layout, 1, CartesianIndex(2, 1))

    @test Unfitted.raw_dof_count(layout) == 15
    @test Unfitted.active_unknowns(layout) == 3
    @test left[2] == right[1]
    @test left[5] == right[4]
    @test left[8] == right[7]
    @test count(!iszero, Unfitted.active_cell_dofs(layout, 1, CartesianIndex(1, 1))) == 2
    @test count(!iszero, Unfitted.active_cell_dofs(layout, 1, CartesianIndex(2, 1))) == 2
end

@testset "trunk dof layout stays basis-local" begin
    # order=4 is the lowest 2D trunk order with an interior mode (the
    # single (2,2) bubble), so a Dirichlet-on-all single cell leaves
    # exactly one interior unknown.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=4, mode=:trunk)
    layout = Unfitted.dof_layout(V; dirichlet=[dirichlet(0.0; on=boundary(:all))])
    active = Unfitted.active_cell_dofs(layout, 1, CartesianIndex(1, 1))

    @test V.levels[1].mode == :trunk
    @test Unfitted.raw_dof_count(layout) == 17
    @test Unfitted.active_unknowns(layout) == 1
    @test count(!iszero, active) == 1
    @test_throws ArgumentError space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=(2, 3),
                                     mode=:trunk)

    V2 = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1))
    overlay_active = Unfitted.active_cell_dofs(Unfitted.dof_layout(V2), 2, CartesianIndex(1, 1))
    @test V2.levels[2].mode == :trunk
    @test count(!iszero, overlay_active) == 1
end

@testset "overlay artificial constraints stay separate from physical boundaries" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=1)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=2)
    layout = Unfitted.dof_layout(V)
    overlay_active = Unfitted.active_cell_dofs(layout, 2, CartesianIndex(1, 1))
    overlay_raw = Unfitted.cell_dofs(layout, 2, CartesianIndex(1, 1))

    @test Unfitted.raw_dof_count(layout) == 13
    @test Unfitted.active_unknowns(layout) == 5
    @test count(!iszero, overlay_active) == 1
    @test Unfitted.constraint_kind(layout, overlay_raw[end]) == :free
    @test all(Unfitted.constraint_kind(layout, raw) == :overlay for raw in overlay_raw[1:(end-1)])

    physical_left = dirichlet(0.0; on=boundary(axis=1, side=:lower))
    constrained = Unfitted.dof_layout(V; dirichlet=[physical_left])
    overlay_raw = Unfitted.cell_dofs(constrained, 2, CartesianIndex(1, 1))
    @test all(Unfitted.constraint_kind(constrained, raw) != :dirichlet for raw in overlay_raw)
end

@testset "artificial overlay constraints keep homogeneous values under physical BCs" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=1)
    V = overlay(V, box((0.0, 0.25), (0.5, 0.75)); cells=(1, 1), order=2)
    layout = Unfitted.dof_layout(V; dirichlet=[dirichlet(1.0; on=boundary(axis=1, side=:lower))])
    overlay_raw = Unfitted.cell_dofs(layout, 2, CartesianIndex(1, 1))
    artificial = [raw
                  for raw in overlay_raw
                  if Unfitted.constraint_kind(layout, raw) in (:overlay, :mixed)]

    @test !isempty(artificial)
    @test all(iszero(Unfitted.constrained_value(layout, raw)) for raw in artificial)
end

@testset "D-generic dof layout smoke cases" begin
    V3 = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(1, 1, 1), order=1)
    layout3 = Unfitted.dof_layout(V3)
    layout3_dirichlet = Unfitted.dof_layout(V3; dirichlet=[dirichlet(0.0; on=boundary(:all))])

    @test Unfitted.raw_dof_count(layout3) == 8
    @test Unfitted.active_unknowns(layout3) == 8
    @test Unfitted.active_unknowns(layout3_dirichlet) == 0

    V4 = space(box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)); cells=(1, 1, 1, 1), order=1)
    V4 = overlay(V4, box((0.25, 0.25, 0.25, 0.25), (0.75, 0.75, 0.75, 0.75)); cells=(1, 1, 1, 1),
                 order=1)
    layout4 = Unfitted.dof_layout(V4)

    @test Unfitted.raw_dof_count(layout4) == 32
    @test Unfitted.active_unknowns(layout4) == 16
    @test all(iszero, Unfitted.active_cell_dofs(layout4, 2, CartesianIndex(1, 1, 1, 1)))
end

@testset "codim-D point dirichlet pins a single vertex dof" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=2)
    corner = boundary((axis=1, side=:lower), (axis=2, side=:lower))

    @test corner.selector == :sides
    @test corner.sides == [(1, :lower), (2, :lower)]

    layout = Unfitted.dof_layout(V; dirichlet=[dirichlet(0.7; on=corner)])
    # Exactly one dof is constrained, and it is the vertex at (0, 0).
    matched = [(raw, c)
               for raw in 1:Unfitted.raw_dof_count(layout), c in 1:1
               if layout.physical_dirichlet[raw, c]]
    @test length(matched) == 1
    raw_corner = matched[1][1]
    @test Unfitted.constrained_value(layout, raw_corner) ≈ 0.7
end

@testset "codim-D point dirichlet captures x-dependent values" begin
    omega = box((0.0, 0.0), (2.0, 3.0))
    V = space(omega; cells=(2, 2), order=2)
    pin = boundary((axis=1, side=:upper), (axis=2, side=:upper))   # (2, 3)
    layout = Unfitted.dof_layout(V; dirichlet=[dirichlet(x -> x[1] + 10x[2]; on=pin)])

    # The vertex mode at the top-right corner gets constrained value g(2,3) = 32.
    matched = [(raw, c)
               for raw in 1:Unfitted.raw_dof_count(layout), c in 1:1
               if layout.physical_dirichlet[raw, c]]
    @test length(matched) == 1
    @test Unfitted.constrained_value(layout, matched[1][1]) ≈ 32.0
end

@testset "codim-D point dirichlet pins a 3D vertex" begin
    omega = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = space(omega; cells=(1, 1, 1), order=2)
    corner = boundary((axis=1, side=:lower), (axis=2, side=:lower), (axis=3, side=:lower))
    layout = Unfitted.dof_layout(V; dirichlet=[dirichlet(2.5; on=corner)])

    matched = [raw for raw in 1:Unfitted.raw_dof_count(layout) if layout.physical_dirichlet[raw, 1]]
    @test length(matched) == 1
    @test Unfitted.constrained_value(layout, matched[1]) ≈ 2.5
end

@testset "boundary() rejects malformed inputs" begin
    @test_throws ArgumentError boundary((axis=1, side=:lower), (axis=1, side=:upper))
    @test_throws ArgumentError boundary((axis=2, side=:left))
end

@testset "point pin removes rigid-body translation on a roller setup" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    u = field(:u, V; components=2)
    bottom = boundary(axis=2, side=:lower)
    top = boundary(axis=2, side=:upper)
    pin = boundary((axis=1, side=:lower), (axis=2, side=:lower))

    # Bottom and top constrain only u_2 — without the pin the model is singular
    # (u_1 has a rigid-body translation mode). The codim-D pin at the bottom-left
    # corner kills u_1 at that vertex, restoring uniqueness.
    model = prepare(poisson(u; source=SVector(0.0, 0.0),
                            dirichlet=[dirichlet(0.0; on=bottom, field=u, component=2),
                                       dirichlet(0.1; on=top, field=u, component=2),
                                       dirichlet(0.0; on=pin, field=u, component=1)]))
    sol = solve!(model)
    @test isposdef(Symmetric(Matrix(model.matrix)))
    @test value(sol, model, (0.0, 0.0), 1) ≈ 0.0 atol = 1.0e-10
    @test value(sol, model, (0.5, 1.0), 2) ≈ 0.1 atol = 1.0e-10
end

@testset "same-facet multi-component Dirichlet is not halved" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    u = field(:u, V; components=2)
    bottom = boundary(axis=2, side=:lower)
    top = boundary(axis=2, side=:upper)
    # u_1 and u_2 are BOTH prescribed on the same top edge (two conditions, one
    # facet). A component-blind projection mass double-counts the top-edge dofs
    # and halves both projected values; the per-component projection returns them
    # exactly.
    model = prepare(poisson(u; source=SVector(0.0, 0.0),
                            dirichlet=[dirichlet(SVector(0.0, 0.0); on=bottom, field=u),
                                       dirichlet(0.3; on=top, field=u, component=1),
                                       dirichlet(0.1; on=top, field=u, component=2)]))
    sol = solve!(model)
    @test value(sol, model, (0.5, 1.0), 1) ≈ 0.3 atol = 1.0e-10
    @test value(sol, model, (0.5, 1.0), 2) ≈ 0.1 atol = 1.0e-10
end

@testset "cross-component corner is not contaminated" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    u = field(:u, V; components=2)
    bottom = boundary(axis=2, side=:lower)
    top = boundary(axis=2, side=:upper)
    left = boundary(axis=1, side=:lower)
    right = boundary(axis=1, side=:upper)
    # Laterals constrain only u_1; the top prescribes u_2 = 0.1. The top corners
    # are shared, so a component-blind mass lets the lateral u_1 rows pull the
    # corner u_2 below 0.1. Per-component projection keeps u_2 = 0.1 exactly
    # along the whole top edge, corners included.
    model = prepare(poisson(u; source=SVector(0.0, 0.0),
                            dirichlet=[dirichlet(SVector(0.0, 0.0); on=bottom, field=u),
                                       dirichlet(0.0; on=left, field=u, component=1),
                                       dirichlet(0.0; on=right, field=u, component=1),
                                       dirichlet(SVector(0.0, 0.1); on=top, field=u)]))
    sol = solve!(model)
    @test value(sol, model, (0.5, 1.0), 2) ≈ 0.1 atol = 1.0e-10
    @test value(sol, model, (1.0, 1.0), 2) ≈ 0.1 atol = 1.0e-8
end

# The two per-component projection bugs above, driven through the load-stepping
# path instead of through `prepare`. `update_dirichlet!` reuses the cached
# `DirichletProjection` from the second call on, so a cache that lost the
# per-component split — one shared mass over the union of the constrained dofs
# — would halve the same-facet values and drag the shared corner's u_2 toward
# zero exactly as a component-blind rebuild does, but only from the second load
# step onward, where the `prepare`-time tests above cannot see it.
@testset "cached Dirichlet projection keeps the per-component split" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=1)
    u = field(:u, V; components=2)
    bottom = boundary(axis=2, side=:lower)
    top = boundary(axis=2, side=:upper)
    left = boundary(axis=1, side=:lower)
    right = boundary(axis=1, side=:upper)

    # Two conditions sharing the top edge, one per component: a component-blind
    # mass double-counts the shared trace dofs and halves both values.
    halving(g) = [dirichlet(SVector(0.0, 0.0); on=bottom, field=u),
                  dirichlet(3g; on=top, field=u, component=1),
                  dirichlet(g; on=top, field=u, component=2)]
    model = prepare(poisson(u; source=SVector(0.0, 0.0), dirichlet=halving(0.1)))
    for g in (0.2, 0.3, -0.15)
        update_dirichlet!(model, halving(g))
        sol = solve!(model)
        @test value(sol, model, (0.5, 1.0), 1) ≈ 3g atol = 1.0e-10
        @test value(sol, model, (0.5, 1.0), 2) ≈ g atol = 1.0e-10
    end

    # Laterals constrain only u_1 and share the top corners with a vector
    # condition on the top: mass from the u_1-only conditions must not reach the
    # corner's u_2 row, which has no matching right-hand side there.
    corner(g) = [dirichlet(SVector(0.0, 0.0); on=bottom, field=u),
                 dirichlet(0.0; on=left, field=u, component=1),
                 dirichlet(0.0; on=right, field=u, component=1),
                 dirichlet(SVector(0.0, g); on=top, field=u)]
    model = prepare(poisson(u; source=SVector(0.0, 0.0), dirichlet=corner(0.1)))
    for g in (0.2, 0.3, -0.15)
        update_dirichlet!(model, corner(g))
        sol = solve!(model)
        @test value(sol, model, (0.5, 1.0), 2) ≈ g atol = 1.0e-10
        @test value(sol, model, (1.0, 1.0), 2) ≈ g atol = 1.0e-8
    end
end

# The cache's exactness contract: a reused `DirichletProjection` must reproduce
# a full rebuild *bit for bit*, not merely to within roundoff, so that a
# load-stepping run is reproducible against one that re-prepares every step.
@testset "reused Dirichlet projection is bit-identical to a fresh one" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=2)
    u = field(:u, V; components=2)
    build(g) = poisson(u; source=SVector(0.0, 0.0),
                       dirichlet=[dirichlet(SVector(0.0, 0.0); on=boundary(axis=2, side=:lower),
                                            field=u),
                                  dirichlet(x -> 0.3 + g * x[1]; on=boundary(axis=2, side=:upper),
                                            field=u, component=1),
                                  dirichlet(g; on=boundary(axis=1, side=:upper), field=u,
                                            component=2)])

    model = prepare(build(0.1))
    @test isempty(model.dirichlet_projections)   # `prepare` projects without caching

    update_dirichlet!(model, build(0.2).dirichlet)
    cached = model.dirichlet_projections[:u]
    for g in (0.4, -0.25)
        update_dirichlet!(model, build(g).dirichlet)
        # Same object, so every later step skips the boundary walk entirely.
        @test model.dirichlet_projections[:u] === cached
        reference = prepare(build(g))
        @test Unfitted.dof_layout(model).fields[1].dofs.constrained_values ==
              Unfitted.dof_layout(reference).fields[1].dofs.constrained_values
    end

    # A mask change rebuilds the dof layout, so the cache must be dropped with
    # it rather than reused against a layout it no longer describes.
    deactivate!(model; level=1, cells=[CartesianIndex(1, 1)])
    @test isempty(model.dirichlet_projections)
end

# The load-stepping driver pattern: a single `prepare(problem)` followed
# by `update_dirichlet!` between steps. The converged solution must
# match what a fresh `prepare(problem)` with the new Dirichlet datum
# produces, and the model.version must stay unchanged.
@testset "update_dirichlet! matches fresh prepare for value-only changes" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=4, order=2)

    function build(rhs_value)
        return poisson(V; source=0.0,
                       dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                                  dirichlet(rhs_value; on=boundary(axis=1, side=:upper))])
    end

    base = prepare(build(0.0))
    initial_version = base.version

    # A value-only update keeps the cached assembly pattern. The structural
    # check pins the constrained-dof set, so `active_unknowns` and every
    # region's active-dof list — everything the pattern indexes — are fixed,
    # and the next assembly must reuse the same object rather than rebuild it.
    solve!(base)
    pattern = base.pattern
    @test pattern !== nothing

    for new_value in (0.25, 0.5, -0.1)
        update_dirichlet!(base,
                          [dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                           dirichlet(new_value; on=boundary(axis=1, side=:upper))])
        @test base.version == initial_version
        @test base.matrix === nothing
        @test base.rhs === nothing
        @test base.pattern === pattern
        s = solve!(base)
        @test base.pattern === pattern

        reference = solve!(prepare(build(new_value)))
        @test value(s, base, (0.5,)) ≈ value(reference, prepare(build(new_value)), (0.5,)) atol = 1.0e-10
        @test value(s, base, (1.0,)) ≈ new_value atol = 1.0e-10
        @test value(s, base, (0.0,)) ≈ 0.0 atol = 1.0e-10
    end
end

# Same driver pattern, but on a COUPLED multi-space model: a Dirichlet on u2's
# own top face lives on V2's space, which `model.problem.space` (V1) does not
# cover. `update_dirichlet!` must re-project each field against its own space.
@testset "update_dirichlet! matches fresh prepare for a coupled multi-space model" begin
    V1 = space(box((0.0, 0.0), (1.0, 0.5)); cells=(2, 1), order=1)
    V2 = space(box((0.0, 0.5), (1.0, 1.0)); cells=(2, 1), order=1)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    Γ = polyline_mesh([SVector(1.0, 0.5), SVector(0.0, 0.5)])

    dir(g) = [dirichlet(0.0; on=boundary(axis=2, side=:lower), field=u1),
              dirichlet(g; on=boundary(axis=2, side=:upper), field=u2)]
    build(g) = Problem((u1, u2);
                       blocks=(stiffness_block(u1), stiffness_block(u2),
                               couple(u1, u2, Γ, mass_form(coefficient=100.0))...),
                       dirichlet=dir(g))

    base = prepare(build(0.0))
    initial_version = base.version
    for g in (0.5, 1.0)
        update_dirichlet!(base, dir(g))
        @test base.version == initial_version
        s = solve!(base)
        reference = solve!(prepare(build(g)))
        @test value(s, base, u2, (0.5, 1.0)) ≈ value(reference, prepare(build(g)), u2, (0.5, 1.0)) atol = 1.0e-10
        @test value(s, base, u2, (0.5, 1.0)) ≈ g atol = 1.0e-8
    end
end

# Structural checks: anything beyond a value-only change must throw.
@testset "update_dirichlet! rejects structural changes" begin
    V = space(box((0.0,), (1.0,)); cells=4, order=2)
    p = poisson(V; source=0.0,
                dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                           dirichlet(0.3; on=boundary(axis=1, side=:upper))])
    model = prepare(p)

    # Different condition count.
    @test_throws ArgumentError update_dirichlet!(model,
                                                 [dirichlet(0.0; on=boundary(axis=1, side=:lower))])

    # Different boundary selector.
    @test_throws ArgumentError update_dirichlet!(model,
                                                 [dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                                                  dirichlet(0.3; on=boundary(axis=1, side=:lower))])
end

# Function-valued Dirichlet must also refresh correctly. A constant
# callback `x -> c` must give the same answer as the scalar `c`.
@testset "update_dirichlet! handles function-valued data" begin
    V = space(box((0.0,), (1.0,)); cells=4, order=2)
    model = prepare(poisson(V; source=0.0,
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                                       dirichlet(x -> 0.0; on=boundary(axis=1, side=:upper))]))
    update_dirichlet!(model,
                      [dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                       dirichlet(x -> 0.4; on=boundary(axis=1, side=:upper))])
    s = solve!(model)
    @test value(s, model, (1.0,)) ≈ 0.4 atol = 1.0e-10
    @test value(s, model, (0.5,)) ≈ 0.2 atol = 1.0e-10
end

# Direct, basis-family-free unit test of the cascade resolver
# `_resolve_constraints!`. Outside this test the resolver is exercised only
# transitively through the B-spline Cᵐ overlay path, so this guards the
# Gauss-elimination + back-substitution machinery on its own, with no
# B-spline dependency. The hand-built system is engineered so that:
#   * a later constraint pivots a raw that an EARLIER pivot's expansion
#     still mentions (constraint B pivots raw 2, which constraint A's pivot
#     expansion references), forcing step-5 back-substitution into the
#     earlier pivot; and
#   * a still-later constraint's raw list references a raw that is itself an
#     earlier pivot (constraint C references raw 3), forcing step-1 forward
#     substitution of the already-resolved expansion.
# All coefficients are dyadic so the resolved (raw, weight) redirects are
# exactly representable in Float64 and can be checked with `==`.
@testset "linear-constraint resolver back-substitutes into earlier pivots" begin
    nraw = 5
    # A:  u₁ +  u₂ + 2u₃ = 0  ⇒ pivot raw 3 = −½u₁ − ½u₂ (mentions raw 2).
    # B: 2u₂ +  u₄      = 0  ⇒ pivot raw 2 = −½u₄; back-substituted into raw 3.
    # C:  u₃ + 2u₅      = 0  ⇒ raw 3 already pivoted (forward substitution);
    #                          pivot raw 5 in terms of the free raws {1, 4}.
    constraints = [Unfitted.LinearConstraint{Float64}([1, 2, 3], [1.0, 1.0, 2.0]),
                   Unfitted.LinearConstraint{Float64}([2, 4], [2.0, 1.0]),
                   Unfitted.LinearConstraint{Float64}([3, 5], [1.0, 2.0])]

    raw_expansion = Vector{Vector{Tuple{Int,Float64}}}(undef, nraw)
    Unfitted._resolve_constraints!(raw_expansion, constraints, nraw)

    # Free raws keep the identity expansion; every pivot is redistributed
    # onto the free raws {1, 4} only — the resolver's "free raws only"
    # invariant after back-substitution.
    @test raw_expansion[1] == [(1, 1.0)]
    @test raw_expansion[4] == [(4, 1.0)]
    @test raw_expansion[2] == [(4, -0.5)]
    @test raw_expansion[3] == [(1, -0.5), (4, 0.25)]
    @test raw_expansion[5] == [(1, 0.25), (4, -0.125)]

    # `has_linear_constraints` mirrors the predicate `dof_layout` derives
    # from the resolved table: true iff some raw redirects onto *other* raws
    # (neither the identity `[(raw, 1)]` nor the empty strong-elimination `[]`).
    has_linear_constraints = any(pairs(raw_expansion)) do (raw, e)
        !(isempty(e) || (length(e) == 1 && e[1] == (raw, 1.0)))
    end
    @test has_linear_constraints
end

@testset "elimination provenance credits the overlay, not covered-mode pruning" begin
    # A raw on an overlay's artificial boundary Γ_o that is *also* buried under a
    # finer level collects two constraints: the overlay trace condition and the
    # pruning one. The overlay constraint is queued first and is what
    # actually eliminates the raw, so `elimination_source` must read `:overlay`;
    # crediting covered-mode pruning would make `reduced_mode_counts` (and the
    # `reduced_dofs` cell array `write_vtk` exports) over-report the saving.
    #
    # The stack: a base level, an overlay at order 2, and a nested finer level
    # over the *same* box, so every cell of the middle overlay is covered and its
    # boundary node modes are buried. Mixed node/span keys — a node factor on the
    # Γ_o face, a bubble factor along the other axis — are the ones that collide.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=2)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=2, order=2)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=2)

    layout = Unfitted.dof_layout(V)
    tol = layout.tolerance
    cache = Unfitted._ClassifyCache{2,Float64}()
    coverage = Unfitted.build_coverage(V, tol, cache)
    middle = V.levels[2]
    level_keys = [key => raw for (raw, key) in pairs(layout.raw_keys) if key.level == middle.id]
    candidates = Unfitted._coverage_constraints(middle, V, coverage, tol, level_keys, cache)

    on_gamma_o(raw) = Unfitted._has_overlay_constraint(layout.raw_keys[raw], middle, V.physical,
                                                       V.domain, tol, cache)
    collisions = count(c -> on_gamma_o(first(c).raws[1]), candidates)
    # Non-vacuity: this stack really does produce raws carrying both constraints.
    @test collisions > 0

    mislabelled = count(candidates) do (c, _)
        raw = c.raws[1]
        on_gamma_o(raw) && layout.elimination_source[raw] !== :overlay
    end
    @test mislabelled == 0

    # Every candidate that is *not* on Γ_o keeps its pruning provenance,
    # so the fix removes only the mislabelled ones.
    reduced = count(r -> layout.elimination_source[r] === :coverage ||
                         layout.elimination_source[r] === :dedup,
                    (c.raws[1] for (c, _) in candidates))
    @test reduced == length(candidates) - collisions
end

@testset "a dormant level does not change the base space's Dirichlet lift" begin
    # Every level of a `ladder` spans the whole domain, so every level's mesh
    # faces coincide with the physical boundary. Before the boundary partition
    # was projected onto the active cells, an overlay carrying no active cell
    # still sliced every physical face at its own resolution — and the
    # composite rule the right-hand side `∫_∂Ω g φ` is integrated with went
    # with it. A declared-but-unpopulated ladder therefore lifted
    # non-polynomial data differently from its own base space, which is a
    # dependence on geometry that carries no unknowns.
    g(x) = sinpi(x[1]) * exp(x[2])
    dom = box((0.0, 0.0), (1.0, 1.0))
    solved = map((space(dom; cells=(4, 4), order=2),
                  ladder(dom; cells=(4, 4), order=2, depth=3, splits=2))) do V
        model = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(g; on=boundary(:all))]))
        (model, solve!(model))
    end
    # 4 faces × 4 base cells. The dormant levels at 8², 16² and 32² cells add
    # nothing, where before they took this to 4 × 32.
    @test diagnostics(solved[1][1]).facet_region_count == 16
    @test diagnostics(solved[2][1]).facet_region_count == 16
    @test solved[1][2].coefficients == solved[2][2].coefficients
end

@testset "merged facet regions still integrate the boundary exactly" begin
    # The greedy merge fuses candidate sub-rectangles that share a parent set,
    # so one region can span several candidates — but never two cells of one
    # parent level, which is what keeps a product of boundary traces
    # polynomial on it. Three things must hold: the regions still partition
    # each facet, a datum the space reproduces still comes back exactly, and
    # the parents decoded from a merged region's signature are the parents at
    # its own midpoint.
    tol = GeometryTolerance(Float64)
    dom = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = ladder(dom; cells=2, order=2, depth=2, splits=2)
    V = adapt(V, 2 => [CartesianIndex(1, 1, 1)], 3 => [CartesianIndex(1, 1, 1)])

    for axis in 1:3, side in (:lower, :upper)
        regions = Unfitted._boundary_facet_regions(V, [(axis, side)], tol)
        @test sum(sum(r.weights) for r in regions) ≈ 1.0 rtol = 1.0e-14   # the face's area
        for region in regions
            @test !isempty(region.parents)
            @test allunique(p.level for p in region.parents)
            for p in region.parents
                level = Unfitted._level_by_id(V, p.level)
                @test Unfitted.is_active(level.mask, p.cell)
                @test Unfitted._cell_on_side(level, p.cell, axis, side)
                # Every Q-point lies inside every parent's own cell: the
                # contract the merge has to preserve, and the reason the rule
                # below is still exact on a fused region.
                for x in region.points
                    @test Unfitted.contains_point(x, p.parent_box, tol)
                end
            end
        end
    end

    # A codim-2 edge (one free axis) and the codim-3 vertex (none) go through
    # the same merge with `F = 1` and `F = 0`.
    edge = Unfitted._boundary_facet_regions(V, [(1, :lower), (2, :lower)], tol)
    @test sum(sum(r.weights) for r in edge) ≈ 1.0 rtol = 1.0e-14
    g(x) = 1 + x[1] + 2x[2] - x[3] + x[1] * x[2] - x[2] * x[3]
    @test sum(sum(w * g(x) for (x, w) in zip(r.points, r.weights)) for r in edge) ≈ 0.5 rtol = 1.0e-14
    vertex = Unfitted._boundary_facet_regions(V, [(1, :lower), (2, :lower), (3, :lower)], tol)
    @test length(vertex) == 1
    @test only(vertex).weights == [1.0]
    @test only(vertex).points == [Unfitted.SVector(0.0, 0.0, 0.0)]

    # The datum above is in the space, so the solved field reproduces it.
    model = prepare(poisson(V; source=x -> 0.0, dirichlet=[dirichlet(g; on=boundary(:all))]))
    @test l2_error(solve!(model), model, g) < 1.0e-12
end

@testset "the facet merge fires where the partition is finer than the parents" begin
    # On a 3D face the two free axes make the projected candidate grid finer
    # than the pattern of parent cells: an overlay live on a corner block cuts
    # both free axes, and the candidates outside the block share their base
    # cell and their empty overlay signature. Those are what the merge fuses,
    # and the regions it emits still tile the face.
    tol = GeometryTolerance(Float64)
    dom = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = ladder(dom; cells=4, order=2, depth=1, splits=2)
    V = adapt(V, 2 => CartesianIndices((1:2, 1:2, 1:2)))
    sides = [(3, :lower)]
    merged = Unfitted._merged_axis_coordinates(V.levels, Val(3), tol,
                                               level -> Unfitted._side_cells(level, sides))
    candidates = prod(length(Unfitted._intervals_from_coordinates(merged[d], tol)) for d in 1:2)
    regions = Unfitted._boundary_facet_regions(V, sides, tol)
    @test candidates == 25                       # 5 base lines + 2 overlay lines per free axis
    @test length(regions) < candidates
    @test sum(sum(r.weights) for r in regions) ≈ 1.0 rtol = 1.0e-14
end

@testset "facet regions are memoised per face, not per selector" begin
    # `_resolve_facet_regions` keys the cache it *retains* on (selector, space),
    # but resolution itself runs through a per-face memo. The two are not the
    # same granularity: `boundary(:all)` and `boundary(axis=1, side=:lower)` are
    # not value-equal, yet the second names one of the four faces the first
    # already covers, and keying the resolution on the selector resolved that
    # shared face twice. Per-face memoisation makes the sharing deliberate. It
    # must be invisible in the regions: the same faces, in the same order, with
    # the same numbers.
    tol = GeometryTolerance(Float64)
    dom = box((0.0, 0.0), (1.0, 1.0))
    V = ladder(dom; cells=2, order=2, depth=1, splits=2)
    V = adapt(V, 2 => [CartesianIndex(1, 1)])

    facets = Unfitted.FacetResolver{2,Float64}(tol)
    whole = Unfitted._facet_regions_for_selector(V, boundary(:all), facets)
    single = Unfitted._facet_regions_for_selector(V, boundary(axis=1, side=:lower), facets)
    # Four entries, one per face of the space — not the five face resolutions
    # the two selectors would have asked for one at a time.
    @test length(facets.faces) == 4

    # Every memoised face carries exactly what an independent resolution of that
    # face builds: same region count, same order, same quadrature, same parents.
    for sides in Unfitted._facets(boundary(:all), Val(2))
        reference = Unfitted._boundary_facet_regions(V, sides, tol)
        cached = facets.faces[(V, sides)]
        @test length(cached) == length(reference)
        for (c, r) in zip(cached, reference)
            @test c.sides == r.sides
            @test c.lower == r.lower
            @test c.upper == r.upper
            @test c.kind == r.kind
            @test c.points == r.points
            @test c.weights == r.weights
            @test c.normal == r.normal
            @test [(p.level, p.cell, p.parent_box) for p in c.parents] ==
                  [(p.level, p.cell, p.parent_box) for p in r.parents]
        end
    end

    # The second selector got the memo's regions themselves, not copies of them —
    # which is the storage half of the saving — and `boundary(:all)`'s union
    # still opens with that same face, since `_facets` lists it first.
    shared = facets.faces[(V, [(1, :lower)])]
    @test length(single) == length(shared)
    @test all(a === b for (a, b) in zip(single, shared))
    @test all(a === b for (a, b) in zip(view(whole, 1:length(shared)), shared))

    # Through the public API: a problem naming two overlapping selectors over one
    # space keeps both cache entries, keeps the region count they sum to, and
    # still reproduces a datum the trace space contains.
    g(x) = 1 + x[1] + 2x[2] + x[1] * x[2]
    both = prepare(poisson(V; source=x -> 0.0,
                           dirichlet=[dirichlet(g; on=boundary(:all)),
                                      dirichlet(g; on=boundary(axis=1, side=:lower))]))
    @test length(both.facet_regions) == 2
    @test diagnostics(both).facet_region_count == length(whole) + length(single)
    @test l2_error(solve!(both), both, g) < 1.0e-12
end

@testset "one FacetResolver resolves each face of a model exactly once" begin
    # The memo above used to be a throw-away of the per-selector sweep, and the L²
    # Dirichlet projection had none at all — it called `_boundary_facet_regions`
    # itself, so a face carrying a nonzero datum was resolved twice per `prepare`
    # and again on every `update_dirichlet!`. One `FacetResolver` per model is now
    # the only route to a face, and the assertions below are on object IDENTITY
    # rather than on a count, because a count is precisely what a second,
    # equal-looking build would also satisfy.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=2)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=2)
    face = boundary(axis=1, side=:lower)
    # Two overlapping selectors — the second names a face the first already covers
    # — and a NONZERO datum, which is what makes the projection walk the boundary
    # at all: `_needs_dirichlet_projection` skips a homogeneous one, so under a zero
    # datum one of the two consumers would be silent and the test vacuous.
    g(x) = 1 + x[1] + 2x[2]
    model = prepare(poisson(V; source=0.0,
                            dirichlet=[dirichlet(g; on=boundary(:all)), dirichlet(g; on=face)]))
    resolver = model.facet_resolver

    # The four faces of the space, resolved once each — against the five
    # selector × face pairs the two selectors name between them, and the nine
    # resolutions the two consumers would have asked for separately.
    @test resolver.tolerance == model.dofs.tolerance
    @test Set(keys(resolver.faces)) ==
          Set((model.problem.space, sides) for sides in Unfitted._facets(boundary(:all), Val(2)))

    # Every region in every per-selector cache entry IS one of the resolver's,
    # position for position — not a copy of one carrying the same numbers.
    for ((selector, each_space), list) in model.facet_regions
        shared = reduce(vcat,
                        (resolver.faces[(each_space, sides)]
                         for sides in Unfitted._facets(selector, Val(2))))
        @test length(list) == length(shared)
        @test all(a === b for (a, b) in zip(list, shared))
    end

    # A load step resolves nothing. `update_dirichlet!` re-projects through the
    # model's resolver, so it walks the entries that are already there — the same
    # list objects, not equal rebuilds of them. This is the cost consolidation
    # buys, and the reason to do it before a facet rule becomes anything more
    # expensive than a tensor product.
    before = copy(resolver.faces)
    update_dirichlet!(model, [dirichlet(g; on=boundary(:all)), dirichlet(g; on=face)])
    @test Set(keys(resolver.faces)) == Set(keys(before))
    @test all(resolver.faces[key] === before[key] for key in keys(before))

    # And the projection resolves through *this* resolver, which the assertions so
    # far cannot see: `_resolve_facet_regions` runs after `system_layout` at
    # `prepare` and names the Dirichlet selectors among its own sites, so the four
    # keys above would be here either way. Emptying the memo first makes the
    # question order-sensitive. Dropping the cached `DirichletProjection` forces
    # the next load step to rebuild it, hence to walk the boundary again, and the
    # faces it resolves on that walk are the only thing that can refill the memo —
    # so a projection holding a resolver of its own leaves it empty.
    delete!(model.dirichlet_projections, only(model.dofs.fields).name)
    empty!(resolver.faces)
    update_dirichlet!(model, [dirichlet(g; on=boundary(:all)), dirichlet(g; on=face)])
    @test Set(keys(resolver.faces)) == Set(keys(before))

    # The same hand-off on the `prepare` side, which runs one level further down:
    # `system_layout` forwards the resolver through `dof_layout` into the
    # projection, so a resolver handed to the layout *alone* comes back carrying
    # the conditions' faces. That is the half of the thread the step above cannot
    # reach, and the order is the whole point again — at `prepare` this walk
    # happens first, which is why `_resolve_facet_regions` afterwards resolves
    # nothing and the model ends up at four.
    fresh = Unfitted.FacetResolver{2,Float64}(model.dofs.tolerance)
    Unfitted.system_layout(model.problem; tolerance=model.dofs.tolerance, facets=fresh)
    @test Set(keys(fresh.faces)) == Set(keys(before))

    # A resolver at another tolerance is not interchangeable: its entries were
    # merged against that tolerance, so the faces it holds are not the partition
    # this layout's constrained dofs were detected on, and projecting through it
    # would fit `g` over a boundary the operator does not integrate — the one
    # failure mode sharing the resolution is meant to rule out. It raises, and
    # raises before touching the values it was going to overwrite.
    layout = only(model.dofs.fields).dofs
    projected = copy(layout.constrained_values)
    @test any(!iszero, projected)
    coarse = Unfitted.FacetResolver{2,Float64}(GeometryTolerance(Float64; merge=1.0e-3))
    @test_throws ArgumentError Unfitted._project_dirichlet_values!(layout, model.problem.space,
                                                                   coarse, model.problem.dirichlet)
    @test layout.constrained_values == projected

    # `nothing` is the standalone spelling — `dof_layout(V; dirichlet)` with no
    # model behind it — and projects through a private resolver at the layout's own
    # tolerance. Bit-identical to the shared route, which is what makes the
    # resolver a saving rather than a semantic change.
    @test Unfitted._project_dirichlet_values!(layout, model.problem.space, nothing,
                                              model.problem.dirichlet) isa
          Unfitted.DirichletProjection{2,Float64}
    @test layout.constrained_values == projected

    # `move!` replaces the resolver along with the dof layout, so the memo tracks
    # the live discretisation: every key holds the space the model has now, and the
    # faces resolved on the superseded space are gone rather than carried forward
    # as ballast no lookup can reach.
    retired = model.problem.space
    move!(model; level=2, to=box((0.3, 0.3), (0.8, 0.8)))
    @test model.problem.space !== retired
    @test !isempty(model.facet_resolver.faces)
    @test all(key[1] === model.problem.space for key in keys(model.facet_resolver.faces))
end

@testset "cut_facet_region_count reports the facet regions whose face leaves Ω" begin
    # A facet region is the whole grid-aligned face of its parent cells, and the
    # rule on it covers the part of that face inside Ω. `cut_facet_region_count` is
    # how many of a model's facet regions needed that trimming, which is also how
    # much of its boundary integration rests on a fitted rule rather than on an
    # exact tensor product. Ω = { x₂ ≤ x₁ + 0.55 } ∩ (0, 1)² puts a plane through
    # the x₁ = 0 face: physical for x₂ ≤ 0.55, fictitious above it.
    plane = leaf(x -> x[2] - x[1] - 0.55; lipschitz=sqrt(2.0))
    cut_domain = physical_domain(plane; subcell_length_scale=0.0625)
    Vcut = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=cut_domain)
    face = boundary(axis=1, side=:lower)
    cut_model = prepare(poisson(Vcut; source=0.0, dirichlet=[dirichlet(0.0; on=face)]))

    # Four regions on that face, one per cell along it. On the face φ = x₂ − 0.55,
    # so the region over x₂ ∈ [0.5, 0.75] is *crossed* by ∂Ω and the one over
    # x₂ ∈ [0.75, 1] lies wholly outside Ω. Both count: what the number reports is
    # regions that integrate non-physical area, and lying entirely outside is the
    # extreme of that, not an exception to it. The lower two faces stay inside Ω.
    @test diagnostics(cut_model).facet_region_count == 4
    @test diagnostics(cut_model).cut_facet_region_count == 2

    # The count is a count of `FacetRegion.kind`, so pin the kinds themselves. The
    # verdict is `classify_cell` on each region's own face box against the level
    # set restricted to the x₁ = 0 slice, which is why the third region reads a
    # `:cut_*` kind and the fourth `:fictitious` rather than both reading the same
    # thing; `:cut_fitted` rather than `:cut_fallback` because the restricted leaf
    # is affine, so the moment fit on it is machine-exact.
    cut_regions = cut_model.facet_regions[(face, cut_model.problem.space)]
    @test [r.lower[2] for r in cut_regions] == [0.0, 0.25, 0.5, 0.75]
    @test [r.kind for r in cut_regions] == [:full, :full, :cut_fitted, :fictitious]

    # The number the count is about. The face's full geometric measure is 1.0 and
    # its physical measure is exactly 0.55 — the two `:full` regions contribute
    # 0.5, the `:cut_fitted` one the 0.05 of [0.5, 0.75] below x₂ = 0.55, and the
    # `:fictitious` one nothing at all. Machine-exact, because φ restricted to the
    # face is affine. This assertion read 1.0 before facet rules were trimmed and
    # is the single deliberate pin of that defect's disappearance.
    @test boundary_integral(q -> 1.0, cut_model; on=face) ≈ 0.55 rtol = 1.0e-14

    # Three facet diagnostics come off those same kinds. The fit is exact, so no
    # failure and no fallback; the residual is a real NNLS residual on the cut
    # region and nothing more. `min_relative_facet_measure` is the smallest
    # integrated-to-geometric measure ratio over the regions, which the wholly
    # fictitious face drives to exactly zero at α = 0 — the conditioning warning
    # a trimmed Dirichlet condition needs, since such a region's trace dofs are
    # constrained with no measure to fit them on.
    @test diagnostics(cut_model).facet_fit_failure_count == 0
    @test diagnostics(cut_model).facet_cut_fallback_count == 0
    @test diagnostics(cut_model).facet_moment_fit_residual_max < 1.0e-14
    @test diagnostics(cut_model).min_relative_facet_measure == 0.0

    # The case that separates the *face* test from a test on the parent cells, and
    # the reason the count is not simply "any parent is cut". A circle of radius
    # 0.1 about (0.2, 0.5) cuts two cells that sit on the x₁ = 0 face, but it stops
    # at x₁ = 0.1 and never reaches the face, so every integrated face lies inside
    # Ω and every facet integral is exactly right. Counting cut cells would report
    # those two regions; counting cut faces reports none.
    offset = leaf(x -> 0.1 - sqrt((x[1] - 0.2)^2 + (x[2] - 0.5)^2); lipschitz=1.0)
    interior = physical_domain(offset; subcell_length_scale=0.0625)
    Vhole = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=interior)
    hole_model = prepare(poisson(Vhole; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test diagnostics(hole_model).cut_region_count > 0
    @test diagnostics(hole_model).facet_region_count == 16
    @test diagnostics(hole_model).cut_facet_region_count == 0
    @test diagnostics(hole_model).min_relative_facet_measure == 1.0
    # Every face lies inside Ω, so every rule is the untrimmed tensor product and
    # the boundary measure is the box perimeter to the bit — a cut cell behind a
    # face costs the face nothing.
    @test boundary_integral(q -> 1.0, hole_model; on=boundary(:all)) ≈ 4.0 rtol = 1.0e-14
    # ...and the cells really are cut, so this is a live discrimination and not a
    # geometry in which the two tests trivially agree.
    hole_regions = hole_model.facet_regions[(boundary(:all), hole_model.problem.space)]
    @test all(r -> r.kind === :full, hole_regions)
    @test count(r -> any(p -> Unfitted.classify_cell(interior, p.parent_box) === :cut, r.parents),
                hole_regions) == 2

    # No geometry at all: nothing can leave Ω, and the classification says so
    # without a level set to ask — `_classify_facet` dispatches on the absent
    # `PhysicalDomain` rather than testing for it.
    Vplain = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2)
    plain = prepare(poisson(Vplain; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test diagnostics(plain).facet_region_count == 16
    @test diagnostics(plain).cut_facet_region_count == 0
    @test all(r -> r.kind === :full, plain.facet_regions[(boundary(:all), plain.problem.space)])

    # `keep_fictitious = true` keeps whole fictitious cells in the layout, so it can
    # only add regions, never retract a verdict: the count is a property of the
    # geometry and the face, not of the fold. On this geometry no cell on the face
    # is wholly fictitious, so both numbers are unchanged.
    kept = physical_domain(plane; alpha=1.0e-3, keep_fictitious=true, subcell_length_scale=0.0625)
    Vkept = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=kept)
    kept_model = prepare(poisson(Vkept; source=0.0, dirichlet=[dirichlet(0.0; on=face)]))
    @test diagnostics(kept_model).facet_region_count == 4
    @test diagnostics(kept_model).cut_facet_region_count == 2

    # α is honoured on facets exactly as it is on cells, and this is the package's
    # one exercise of that path. Writing the α-FCM integrand as
    #     ∫_face α(x) f = ∫_{face∩Ω} f + α ∫_{face∖Ω} f
    #                   = (1 − α) ∫_{face∩Ω} f + α ∫_face f,
    # the `:cut` region's rule is the concatenation `(1−α)·fit ∪ α·tensor` and the
    # fictitious region's is `α·tensor` — the same two rules the volume dispatch
    # builds. So the face's unit integral is `0.55 + α·0.45` exactly: the physical
    # 0.55, plus α times the 0.45 of face that is not in Ω. Every term is affine or
    # a whole cell, so this is machine-exact and not a band.
    @test [r.kind for r in kept_model.facet_regions[(face, kept_model.problem.space)]] ==
          [:full, :full, :cut_fitted, :fictitious_alpha]
    @test boundary_integral(q -> 1.0, kept_model; on=face) ≈ 0.55 + 1.0e-3 * 0.45 rtol = 1.0e-14
    # And the relative measure now reports α on the fictitious face rather than
    # zero: its dofs carry α-stabilised mass instead of none.
    @test diagnostics(kept_model).min_relative_facet_measure ≈ 1.0e-3 rtol = 1.0e-14
end

@testset "a facet region's kind is the verdict on its own face, in every codimension" begin
    # Every codimension goes through the same two steps — restrict the level set to
    # the facet's affine slice, then classify the region's own face box against it —
    # and the only thing that changes is how many axes the box has: a codim-1 face
    # reaches an `AxisBox{D-1}`, a codim-2 edge an `AxisBox{1}`, and the codim-D
    # vertex has no box at all, `AxisBox` being non-degenerate by construction, so it
    # is decided by membership at the one point of the slice.
    #
    # `FacetRegion.kind` names the *rule* the region ended up with, so it refines
    # that verdict — a cut face reads `:cut_fitted`, `:cut_fallback` or `:cut_failed`
    # according to how the moment fit on the slice went, a fictitious one
    # `:fictitious_alpha` under α > 0. This testset is about the verdict, so the
    # verdict is read back out of the kind through a closed table: an unexpected kind
    # raises a `KeyError` here rather than quietly passing as something else.
    #
    # Ω = (0,1)³ ∖ {‖x‖ ≤ 0.3} puts the hole on a corner, so each of the three
    # lower faces meets a quarter disc of radius 0.3 and the lower edges meet it as
    # an interval. On a face box whose corners all have non-negative coordinates the
    # nearest and farthest points of the box are its own corners, which gives an
    # independent analytic verdict to compare every kind against.
    tol = GeometryTolerance(Float64)
    ball = leaf(x -> 0.3 - sqrt(x[1]^2 + x[2]^2 + x[3]^2); lipschitz=1.0)
    p = physical_domain(ball; subcell_length_scale=0.0625)
    V = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(4, 4, 4), order=2, physical=p)
    verdicts = Dict(:full => :full, :fictitious => :fictitious, :fictitious_alpha => :fictitious,
                    :cut_fitted => :cut, :cut_fallback => :cut, :cut_failed => :cut,
                    :cut_alpha_failed => :cut)
    verdict(region) = verdicts[region.kind]
    expected(region) =
        let near = norm(region.lower), far = norm(region.upper)
            far <= 0.3 ? :fictitious : (near >= 0.3 ? :full : :cut)
        end

    for axis in 1:3
        regions = Unfitted._boundary_facet_regions(V, [(axis, :lower)], tol)
        @test length(regions) == 16
        @test all(r -> verdict(r) === expected(r), regions)
        @test count(r -> verdict(r) === :cut, regions) == 3
        @test count(r -> verdict(r) === :fictitious, regions) == 0
        # The upper faces are a whole unit away from the hole, so nothing on them
        # can leave Ω — the discrimination is live, not a geometry in which every
        # face answers the same.
        @test all(r -> r.kind === :full, Unfitted._boundary_facet_regions(V, [(axis, :upper)], tol))
    end

    # Codim-2: the ball covers the edge out to x₃ = 0.3, so the first region lies
    # wholly outside Ω and the second is crossed.
    edge = Unfitted._boundary_facet_regions(V, [(1, :lower), (2, :lower)], tol)
    @test [verdict(r) for r in edge] == [:fictitious, :cut, :full, :full]
    @test all(r -> verdict(r) === expected(r), edge)

    # Codim-3: the origin is the centre of the ball, hence outside Ω.
    vertex = Unfitted._boundary_facet_regions(V, [(1, :lower), (2, :lower), (3, :lower)], tol)
    @test only(vertex).kind === :fictitious
    far_vertex = Unfitted._boundary_facet_regions(V, [(1, :upper), (2, :upper), (3, :upper)], tol)
    @test only(far_vertex).kind === :full

    # The ball is symmetric in all three axes, which leaves the *coordinate* half of
    # the slice untested in both of its directions. `_facet_box` selects the free axes
    # in ascending order while `_facet_pins` hands `_restrict_domain` the pinned ones
    # in descending order, and the two have to agree on which slice coordinate is
    # which original axis; on a geometry symmetric in the axes that survive, any
    # permutation of the slice's coordinate slots is invisible. Measured on the ball
    # above: reversing the free-axis slots moves 0 of the 16 verdicts on each of its
    # lower faces, so nothing here is protected by the assertions above.
    #
    # Ω = {2x₁ + 3x₂ + x₃ > 0.9} carries a distinct coefficient on every axis, so no
    # permutation survives it, and being affine it makes the classifier exact: an
    # affine function attains its extremes over a box at corners, so `:cut` is
    # precisely "the corners disagree" and `expected_tilt` below is an oracle rather
    # than a recorded run. No cell corner lies on the plane (`2i + 3j + k = 3.6` has
    # no integer solution on this mesh), so no verdict sits on a tie, and no cell is
    # wholly fictitious, so the mask retracts nothing and every face keeps all 16
    # regions — the geometry under test is the slice and nothing else.
    tilt = physical_domain(leaf(x -> 0.9 - 2x[1] - 3x[2] - x[3]; lipschitz=sqrt(14.0));
                           subcell_length_scale=0.0625)
    Vtilt = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(4, 4, 4), order=2, physical=tilt)
    expected_tilt(region) =
        let lo = 2region.lower[1] + 3region.lower[2] + region.lower[3],
            hi = 2region.upper[1] + 3region.upper[2] + region.upper[3]

            lo > 0.9 ? :full : (hi <= 0.9 ? :fictitious : :cut)
        end

    for axis in 1:3, side in (:lower, :upper)
        regions = Unfitted._boundary_facet_regions(Vtilt, [(axis, side)], tol)
        @test length(regions) == 16
        @test all(r -> verdict(r) === expected_tilt(r), regions)
    end

    # The x₂ face is the only codim-1 face of a 3-D box whose free axes interleave
    # with its pinned one (1 and 3 around 2), hence the only one on which swapping the
    # slice's coordinate slots shows at all: it would classify against x₁ + 2x₃ and
    # move 4 of these 16 verdicts. All three verdicts occur on it, so the oracle is
    # discriminating here and not a face on which every region answers the same.
    tilted_face = Unfitted._boundary_facet_regions(Vtilt, [(2, :lower)], tol)
    @test map(k -> count(r -> verdict(r) === k, tilted_face), (:full, :cut, :fictitious)) ==
          (10, 5, 1)

    # The pinned axes interleave in turn whenever the lower-numbered one is not the
    # last — axes 1 and 2 of a 3-D box, whose free axis is 3. Consuming the pins
    # ascending would drop axis 1 first and leave the pin named "2" addressing x₃, the
    # edge's own free axis, pinning it to the axis-2 coordinate and reading
    # `[:fictitious, :cut, :full, :full]` along the edge instead. (A pin pair whose
    # higher axis is the last one cannot go wrong silently: the stale index addresses
    # a slot the shortened coordinate vector does not have, and the restriction
    # throws.)
    tilted_edge = Unfitted._boundary_facet_regions(Vtilt, [(1, :lower), (2, :lower)], tol)
    @test [verdict(r) for r in tilted_edge] == [:fictitious, :fictitious, :fictitious, :cut]
    @test all(r -> verdict(r) === expected_tilt(r), tilted_edge)
end

@testset "a trimmed facet rule integrates the physical part of its face" begin
    # The anchors for facet trimming, each against an analytic value and never
    # against a recorded run of itself. `_boundary_facet_regions` is read directly
    # where only the rule is under test: it is the one function both the operator
    # and the Dirichlet projection resolve through, so its weight sum IS the measure
    # every consumer of that face integrates over.
    tol = GeometryTolerance(Float64)
    measure(V, sides) = sum(sum(r.weights) for r in Unfitted._boundary_facet_regions(V, sides, tol))

    # 1. A symmetry plane through a hole — the shape `kirsch_plate_2d` ships, which
    #    is the reason this defect was worth fixing rather than documenting. Ω is the
    #    quarter plate (0,4)² outside the unit circle at the origin, 8×8 cells. On
    #    the x₁ = 0 face φ reduces to 1 − x₂, so the face is physical for x₂ ≥ 1 and
    #    its measure is exactly 3.0 of a geometric 4.0. The cell [0, 0.5]² is wholly
    #    fictitious, and on a prepared model the fold has already dropped it, which
    #    is why the untrimmed integral read 3.5 rather than 4.0: a face loses whole
    #    cells through the mask and the rest through the rule. Read here before the
    #    fold, so both losses are visible at once.
    #
    #    The hole is *tangent* to the face — it touches it at the single point
    #    x₂ = 1 — so the region [0.5, 1] classifies `:cut` while `Ω ∩ face` on it has
    #    measure zero, and the moment fit on the slice comes back `:empty`. That is
    #    `:cut_failed`, an empty rule, and the face's 3.0 is then the six `:full`
    #    regions alone: machine-exact without a fit having to be. Pinning the kinds
    #    is what keeps this a tangency test rather than a measure test that happens
    #    to pass.
    plate = physical_domain(leaf(x -> 1.0 - sqrt(x[1]^2 + x[2]^2); lipschitz=1.0);
                            subcell_length_scale=0.5 / 2^4, max_depth=4)
    Vplate = space(box((0.0, 0.0), (4.0, 4.0)); cells=(8, 8), order=2, physical=plate)
    for axis in 1:2
        @test measure(Vplate, [(axis, :lower)]) ≈ 3.0 rtol = 1.0e-14
        @test measure(Vplate, [(axis, :upper)]) ≈ 4.0 rtol = 1.0e-14
        @test [r.kind for r in Unfitted._boundary_facet_regions(Vplate, [(axis, :lower)], tol)] ==
              [:fictitious, :cut_failed, :full, :full, :full, :full, :full, :full]
    end

    # 2. Two subdomains partition the face they share, which nothing else in the
    #    suite checks and which is the one place a sign or side error in the
    #    restriction would survive every other assertion here: slicing the wrong
    #    way keeps each half a plausible number while their sum stops being 1.
    #    `interface_coupling_2d`'s geometry — the seam x₁ = 0.53 crossing two
    #    non-matching meshes whose boxes overlap, Ω₁ = {x₁ ≤ 0.53}, Ω₂ = {x₁ ≥ 0.53}.
    #    Ω₁'s bottom face spans six of its seven cells (the seventh, [0.6, 0.7], is
    #    wholly fictitious) and so is 0.6 untrimmed; Ω₂'s spans five of six and is
    #    0.483333. Trimmed they are 0.53 and 0.47, and the two halves of y = 0 meet
    #    at the seam with no gap and no overlap.
    seam_x = 0.53
    left = physical_domain(leaf(x -> x[1] - seam_x; lipschitz=1.0); subcell_length_scale=0.1 / 16)
    right = physical_domain(leaf(x -> seam_x - x[1]; lipschitz=1.0); subcell_length_scale=0.1 / 16)
    V₁ = space(box((0.0, 0.0), (0.70, 1.0)); cells=(7, 6), order=2, physical=left)
    V₂ = space(box((0.42, 0.0), (1.0, 1.0)); cells=(6, 8), order=2, physical=right)
    measure₁ = measure(V₁, [(2, :lower)])
    measure₂ = measure(V₂, [(2, :lower)])
    @test measure₁ ≈ seam_x rtol = 1.0e-14
    @test measure₂ ≈ 1 - seam_x rtol = 1.0e-14
    @test measure₁ + measure₂ ≈ 1.0 rtol = 1.0e-14

    # 3. A curved cut face, where the rule is accurate rather than exact. The
    #    space-time cavity's t = 0 slice is the disc of radius r₀ = 0.1373 about
    #    (0.4967, 0.5013) inside the central cell of a 3×3 grid, so the face measure
    #    is the analytic 1 − π r₀². The moment fit on the slice reaches it to 1.9e-5,
    #    which is the honest ceiling on a curved cut face and the reason no docstring
    #    here promises exactness.
    #
    #    And it is insensitive to the geometric-robustness knobs: `_choose_axis`
    #    declares the sliced sub-boxes graph-like and stops bisecting, so driving
    #    `subcell_length_scale` from 1/12 to 1/192 (effective depth 2 → 6) or capping
    #    `max_depth` at 2 reproduces the value to the last bit. The only knob is the
    #    moment order, which `_facet_moment_order` takes from the parent cells.
    r₀, ṙ = 0.1373, 0.22
    centre = SVector(0.4967, 0.5013)
    cavity = leaf(x -> (r₀ + ṙ * x[3]) - sqrt((x[1] - centre[1])^2 + (x[2] - centre[2])^2);
                  lipschitz=sqrt(1 + ṙ^2))
    slab(scale, depth) =
        let dom = physical_domain(cavity; alpha=1.0e-6, subcell_length_scale=scale, max_depth=depth)
            space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(3, 3, 3), order=(2, 2, 3),
                  physical=dom)
        end
    initial = measure(slab(1 / 12, 8), [(3, :lower)])
    @test initial ≈ 1 - π * r₀^2 atol = 3.0e-5
    @test abs(initial - (1 - π * r₀^2)) > 1.0e-6          # accurate, not exact
    @test measure(slab(1 / 192, 8), [(3, :lower)]) == initial
    @test measure(slab(1 / 12, 2), [(3, :lower)]) == initial
    # The t = T face is cut by the grown cavity too, against its own analytic value
    # 1 − π r(T)², so the slab is not a geometry in which only one face can see the
    # restriction. It comes out an order of magnitude *better* than t = 0 (3.9e-7
    # against 1.9e-5) because the larger disc spans nine cells instead of one, so
    # each cut region sees a shallower arc — which is the convergence this error
    # does have. The uncut side faces stay exactly 1.0: the trimming touches the
    # faces the level set reaches and no others.
    @test measure(slab(1 / 12, 8), [(3, :upper)]) ≈ 1 - π * (r₀ + ṙ)^2 atol = 1.0e-6
    @test measure(slab(1 / 12, 8), [(1, :lower)]) ≈ 1.0 rtol = 1.0e-14
    @test measure(slab(1 / 12, 8), [(2, :upper)]) ≈ 1.0 rtol = 1.0e-14
end

@testset "the trimmed faces and the immersed surface close ∂Ω" begin
    # The anchor that reaches a curved cut in a form no single rule can satisfy on
    # its own. Trimming makes ∂Ω's two halves — the grid-aligned faces and the
    # user's `BoundaryMesh` — a partition of ∂Ω for the first time, so the
    # divergence theorem applies to their union: ∮_∂Ω n dS = 0 identically, and
    # ∮_∂Ω x·n dS = D·|Ω| ties the facet rule, the surface rule and the volume rule
    # into one number. Untrimmed, the grid-aligned half double-counts the hole's
    # footprint and the first residual is (−0.3, −0.3) — O(1), a direct measurement
    # of the defect rather than a tolerance question.
    #
    # Ω = (1,2)² ∖ {‖x − c‖ ≤ 0.3} with c = (1,1), on 4×4 cells. The hole sits on a
    # corner so its boundary is a quarter arc, and the box is offset from the origin
    # so that x·n does not vanish on the two trimmed faces — centred at the origin
    # it would, and the second identity would be blind to the trimming it is meant
    # to check.
    R, segments = 0.3, 64
    c = SVector(1.0, 1.0)
    hole = physical_domain(leaf(x -> R - norm(x - c); lipschitz=1.0); subcell_length_scale=0.0625)
    V = space(box((1.0, 1.0), (2.0, 2.0)); cells=(4, 4), order=2, physical=hole)
    # Traversed (0, R) → (R, 0) about `c`: a 2-D segment's normal is 90° clockwise
    # from its direction, which points at the centre here, i.e. out of Ω.
    arc = polyline_mesh([c + R * SVector(sin(t), cos(t))
                         for t in range(0, π / 2; length=segments + 1)])
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    # Both trimmed faces, machine-exact: on the x₁ = 1 face the leaf restricts to
    # R − |x₂ − c₂|, affine on each region, and the cut sits inside one cell.
    @test boundary_integral(q -> 1.0, model; on=boundary(axis=1, side=:lower)) ≈ 0.7 rtol = 1.0e-14
    @test boundary_integral(q -> 1.0, model; on=boundary(axis=2, side=:lower)) ≈ 0.7 rtol = 1.0e-14
    @test diagnostics(model).cut_facet_region_count == 4
    @test diagnostics(model).facet_fit_failure_count == 0
    # Four of the sixteen regions are trimmed, two per cut face: the one inside the
    # hole and the one the arc crosses. The relative measure reports the worst of
    # them, which is the wholly fictitious one at α = 0, hence exactly zero.
    @test [r.kind for r in model.facet_regions[(boundary(:all), model.problem.space)][1:4]] ==
          [:fictitious, :cut_fitted, :full, :full]
    @test diagnostics(model).min_relative_facet_measure == 0.0

    # ∮ n dS = 0. The surface half contributes rot(b − a) exactly, whatever the
    # subdivision, so the whole residual is the facet rule's — which is why this is
    # the form that pins the trimming and not a property of the polyline.
    faces = boundary_integral(q -> q.normal, model; on=boundary(:all))
    surface = boundary_integral(q -> q.normal, model; on=arc)
    @test norm(faces + surface) < 1.0e-13
    @test norm(surface) > 0.4                      # the surface half is not empty
    # ...and untrimmed it would not close. The four faces at full measure sum to
    # zero on any box, so the residual was exactly −surface, of norm R√2 = 0.424:
    # the whole arc was surplus, and that is the O(1) defect this closes.
    @test norm(surface) ≈ norm(SVector(R, R)) rtol = 1.0e-14

    # ∮ x·n dS = D·|Ω|, against the volume rule on the same geometry — the stronger
    # form, which ties the facet rule, the surface rule and the volume rule into one
    # number. The residual decomposes exactly, with nothing left over for the facet
    # rule: the 64-segment polyline's chord-versus-arc deficit
    # 2·N·(R²/2)(θ − sin θ) with θ = π/2N, plus twice the volume rule's own error on
    # the quarter disc. A trimming error would land here and nowhere else, so the
    # identity below is what makes this a test of three rules agreeing rather than a
    # tolerance on any one of them.
    moment = boundary_integral(q -> dot(q.x, q.normal), model; on=boundary(:all)) +
             boundary_integral(q -> dot(q.x, q.normal), model; on=arc)
    volume = Ref(0.0)
    foreach_quadrature_point(model) do q
        volume[] += q.weight
    end
    area = 1 - π * R^2 / 4
    θ = π / (2 * segments)
    deficit = 2 * segments * (R^2 / 2) * (θ - sin(θ))
    @test volume[] ≈ area rtol = 1.0e-9
    @test abs(moment - 2 * volume[]) < 2.0e-5
    @test (moment - 2 * volume[]) - deficit ≈ -2 * (volume[] - area) atol = 1.0e-14
end

@testset "boundary(:all; except=…) is the union of the faces that remain" begin
    # The gap `except` closes: `:sides` gives the *intersection* of its faces, so
    # before this the union of a face *subset* had no spelling at all and every
    # caller enumerated the faces by hand. What the selector must deliver is not
    # an approximation of that list but the list itself — same constrained dofs,
    # same projected values, same solution — in every dimension.
    g(x) = 1 + x[1] + 2x[2]

    for (D, cells) in ((2, (3, 2)), (3, (2, 3, 2)))
        domain = box(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D))
        V = space(domain; cells=cells, order=2)
        all_faces = [(axis, side) for axis in 1:D for side in (:lower, :upper)]

        for dropped in all_faces
            kept = filter(!=(dropped), all_faces)
            selector = boundary(:all; except=(axis=dropped[1], side=dropped[2]))

            # The facet decomposition every consumer reads is `boundary(:all)`'s
            # list minus the one entry, in that order.
            @test Unfitted._facets(selector, Val(D)) == [[face] for face in kept]

            one = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(g; on=selector)]))
            many = prepare(poisson(V; source=0.0,
                                   dirichlet=[dirichlet(g; on=boundary(axis=a, side=s))
                                              for (a, s) in kept]))
            one_dofs, many_dofs = only(one.dofs.fields).dofs, only(many.dofs.fields).dofs
            @test one_dofs.physical_dirichlet == many_dofs.physical_dirichlet
            @test one_dofs.constrained_values == many_dofs.constrained_values
            # End to end, not merely the same dof count: one solved system
            # against the other, coefficient for coefficient.
            @test solve!(one).coefficients == solve!(many).coefficients
        end

        # And it solves the mixed problem it describes. Dropping a face of the
        # last axis and taking a datum independent of that coordinate makes the
        # natural zero-flux condition on the dropped face hold for the datum's
        # own harmonic extension, so `h` is the exact solution here and the
        # selector reproduces it.
        h(x) = 1 + x[1]
        open_top = boundary(:all; except=(axis=D, side=:upper))
        mixed = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(h; on=open_top)]))
        @test l2_error(solve!(mixed), mixed, h) < 1.0e-12

        # The whole-axis spelling drops both faces of the axis, and the facet
        # count confirms it against the 2D−2 that remain.
        whole_axis = boundary(:all; except=(axis=D,))
        @test length(Unfitted._facets(whole_axis, Val(D))) == 2D - 2
        @test Unfitted._facets(whole_axis, Val(D)) == [[face] for face in all_faces if face[1] != D]
    end
end

@testset "boundary(:all; except=…) parses an exclusion list permissively" begin
    # An omitted `side` means both faces of the axis, and it canonicalises to
    # exactly that pair — so the short spelling and the long one are one selector
    # *value*, hence one facet-region cache entry instead of two resolutions of
    # the same faces. Order in the list is not part of the value either: it is a
    # set of faces to remove.
    @test boundary(:all; except=(axis=2,)) ==
          boundary(:all; except=((axis=2, side=:lower), (axis=2, side=:upper)))
    @test hash(boundary(:all; except=(axis=2,))) ==
          hash(boundary(:all; except=((axis=2, side=:upper), (axis=2, side=:lower))))
    scrambled = [(axis=3, side=:upper), (axis=3, side=:lower), (axis=1, side=:lower)]
    @test boundary(:all; except=((axis=1, side=:lower), (axis=3,))) ==
          boundary(:all; except=scrambled)

    # A repeated axis is meaningless for an intersection and `boundary(pairs...)`
    # rejects it. For an exclusion list it is ordinary and must construct; so must
    # a face named twice over.
    @test_throws ArgumentError boundary((axis=3, side=:lower), (axis=3, side=:upper))
    @test boundary(:all; except=((axis=3, side=:lower), (axis=3, side=:upper))).sides ==
          [(3, :lower), (3, :upper)]
    @test boundary(:all; except=((axis=1,), (axis=1, side=:lower))).sides ==
          [(1, :lower), (1, :upper)]
    @test boundary(:all; except=(axis=1, side=:lower)).selector === :except

    # `except` removes faces from the whole boundary, so it means nothing on any
    # other selector, and the error has to say that rather than complain about
    # the faces.
    @test_throws ArgumentError boundary(:sides; except=(axis=1, side=:lower))
    @test_throws ArgumentError boundary(:nonsense; except=(axis=1,))

    # Malformed entries. The unknown-key rejection is what keeps the optional
    # `side` safe: `sides=:lower` would otherwise read as `(axis=1,)` and quietly
    # exclude one face too many.
    @test_throws ArgumentError boundary(:all; except=(axis=1, side=:left))
    @test_throws ArgumentError boundary(:all; except=(side=:lower,))
    @test_throws ArgumentError boundary(:all; except=(axis=1, sides=:lower))
    @test_throws ArgumentError boundary(:all; except=(axis=0,))
    @test_throws ArgumentError boundary(:all; except=(1, :lower))

    # Excluding every face selects nothing, which is never what a caller meant.
    # Like the out-of-bounds axis check below it needs `D`, so it fires when the
    # selector meets a space — the same moment `boundary(axis=4, side=:lower)`
    # fails in 2D.
    everything = boundary(:all; except=((axis=1,), (axis=2,)))
    @test_throws ArgumentError Unfitted._facets(everything, Val(2))
    @test_throws ArgumentError Unfitted._facets(boundary(:all; except=(axis=3,)), Val(2))
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    @test_throws ArgumentError prepare(poisson(V; source=0.0,
                                               dirichlet=[dirichlet(0.0; on=everything)]))
    # ...but in 3D that same list keeps the two faces of axis 3.
    @test length(Unfitted._facets(everything, Val(3))) == 2
end

@testset "boundary(:all) is unchanged by the except keyword" begin
    # `except` is a pure addition: with no exclusions the method returns the
    # selector the one-line method always returned, by value and by hash, so a
    # freshly built `boundary(:all)` still finds the facet-region cache entry a
    # prepared model made for it.
    everywhere = boundary(:all)
    @test everywhere == Unfitted.BoundarySelector(:all, Tuple{Int,Symbol}[])
    @test hash(everywhere) == hash(Unfitted.BoundarySelector(:all, Tuple{Int,Symbol}[]))
    @test everywhere == boundary(:all; except=())
    @test hash(everywhere) == hash(boundary(:all; except=()))
    @test everywhere != boundary(:all; except=(axis=1, side=:lower))
    @test length(Dict(everywhere => 1, boundary(:all; except=(axis=1, side=:lower)) => 2)) == 2

    # An unsupported symbol with no `except` still constructs and still fails
    # exactly where it used to — at prepare time, not at construction.
    @test boundary(:nonsense).selector === :nonsense
    @test isempty(boundary(:nonsense).sides)
    @test_throws ArgumentError Unfitted._facets(boundary(:nonsense), Val(2))

    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test haskey(model.facet_regions, (boundary(:all), model.problem.space))
end

@testset "except excludes closed faces, so the seam dof stays constrained" begin
    # Exclusion is by closed face: the kept faces come with their closure, so a
    # dof where a kept face meets an excluded one is still constrained, because
    # the kept face's datum extends to its own edge. That is exactly what makes
    # the dof set equal the hand-written union's, and it is the one thing about
    # `except` that can surprise — it is not "∂Ω minus a face" by measure.
    #
    # Q1 on a 2×2 grid has 9 nodes, 8 of them on ∂Ω. Dropping the x-upper face
    # leaves 7 constrained, not 5: only its *open* middle node comes free, while
    # its two corner nodes are shared with the kept y-faces and stay pinned.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    selector = boundary(:all; except=(axis=1, side=:upper))
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(1.0; on=selector)]))
    dofs = only(model.dofs.fields).dofs
    @test count(dofs.physical_dirichlet) == 7

    # Named directly: the corner dof is on an excluded face and a kept one at
    # once, and the selector matches it.
    folded = model.problem.space
    level, domain, tol = folded.levels[1], folded.domain, dofs.tolerance
    matches(key, on) = Unfitted._selector_matches(key, level, domain, on, tol)
    corner = only(key
                  for key in dofs.raw_keys
                  if matches(key, boundary((axis=1, side=:upper), (axis=2, side=:lower))))
    @test matches(corner, selector)
    # ...while the mid-face dof of the excluded face, which no kept face touches,
    # does not.
    @test count(key -> matches(key, boundary(axis=1, side=:upper)) && !matches(key, selector),
                dofs.raw_keys) == 1
end

# Grid-aligned facet integration has two consumers that must see the same facet:
# `_facet_regions_for_selector` (src/model.jl), which assembly and
# `boundary_integral` read through, and `_sample_dirichlet_facet!`
# (src/dirichlet.jl), which builds the L² boundary-trace projection. Both resolve
# through the model's one `FacetResolver`, so today they read the same region
# objects; before that they called `_boundary_facet_regions` separately and agreed
# only because it is a pure function of `(V, sides, tolerance)`. Nothing else in
# the suite would notice if either side stopped reading the shared resolution —
# which is a live risk again the moment one of them is given a rule the other is
# not.
#
# What this guards is the trimming of a cut face's rule to its physical part —
# the regions `cut_facet_region_count` reports. See the testset
# "cut_facet_region_count reports the facet regions whose face leaves Ω" above,
# and `_facet_region_stats` in src/model.jl. That trimming has to land on BOTH
# paths at once. On the operator path alone, the Dirichlet datum would be fitted
# over the whole face while the operator integrates only the physical part: a
# clean residual, a converging solve, and boundary data fitted over area that is
# not in Ω. On the projection path alone, the mirror image. Neither shows up in a
# norm.
#
# This is a guard, not a snapshot: it asserts that the two sides read the SAME
# thing, never what that thing is. Ω = { x₂ ≤ x₁ + 0.55 } ∩ (0,1)² reduces on
# the x₁ = 0 face to φ = x₂ − 0.55, so of that face's full measure 1.0 exactly
# 0.55 is physical. Both sides read 0.55 — and read 1.0 before the trimming
# landed, with every assertion below holding either way. None of them needed
# editing when the value moved; the single deliberate pin of it lives in the
# neighbouring testset named above, and stays the only one.
@testset "the Dirichlet projection and the operator integrate the same facet" begin
    plane = leaf(x -> x[2] - x[1] - 0.55; lipschitz=sqrt(2.0))
    cut_domain = physical_domain(plane; subcell_length_scale=0.0625)
    Vcut = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=cut_domain)
    face = boundary(axis=1, side=:lower)
    # A NONZERO datum. `_needs_dirichlet_projection` skips the boundary walk
    # outright for a homogeneous one, so a zero datum never reaches the
    # projection path and this comparison would have nothing to read on one side.
    g(x) = 1 + x[1] + 2x[2]
    model = prepare(poisson(Vcut; source=0.0, dirichlet=[dirichlet(g; on=face)]))

    # The geometry, pinned against the level set itself rather than against any
    # quadrature: the face straddles ∂Ω and crosses it at x₂ = 0.55, which is
    # where its physical measure 0.55 comes from. A live discrimination, then —
    # not a face that happens to lie inside Ω, for which the two paths would
    # agree with nothing to disagree about.
    @test levelset_value(cut_domain, SVector(0.0, 0.0)) < 0        # inside Ω
    @test levelset_value(cut_domain, SVector(0.0, 1.0)) > 0        # outside Ω
    @test levelset_value(cut_domain, SVector(0.0, 0.55)) ≈ 0 atol = 1.0e-15

    # The projection side. `prepare` projects without caching, so the quadrature
    # `_sample_dirichlet_facet!` actually walked is published by the first
    # `update_dirichlet!`, as a cached `DirichletProjection`. The datum is
    # unchanged, so this re-projects exactly the values `prepare` already fitted.
    update_dirichlet!(model, [dirichlet(g; on=face)])
    projection = model.dirichlet_projections[:u]
    @test projection.facet_count == 1
    trace = only(projection.samples)

    # The operator side, read independently and through the public consumer:
    # `boundary_integral` resolves the face through the model's per-face memo —
    # the same regions a Neumann, Robin or Nitsche term `on = face` assembles
    # over. Nothing here is derived from `trace`.
    operator_points = SVector{2,Float64}[]
    operator_weights = Float64[]
    operator_measure = boundary_integral(model; on=face) do q
        push!(operator_points, q.x)
        push!(operator_weights, q.weight)
        return 1.0
    end

    # 1. Measure agreement: the face the projection fits `g` over carries the
    #    measure the operator integrates.
    @test sum(trace.weights) ≈ operator_measure rtol = 1.0e-14

    # 2. Point-set agreement, as multisets — the same rule, not merely the same
    #    total, so a divergence that preserves the measure is caught too. Region
    #    order differs between the paths (the projection walks one face, while
    #    `boundary_integral` walks the resolved union), hence the sort.
    order(x) = (x[1], x[2])
    @test length(trace.points) == length(operator_points)
    @test sort(trace.points; by=order) == sort(operator_points; by=order)
    @test sort(trace.weights) ≈ sort(operator_weights) rtol = 1.0e-14

    # ...and the projection is not vacuous: the face carries unknowns and the
    # datum lands on them, so a divergence between the two faces would move real
    # numbers rather than agreeing emptily.
    @test !isempty(projection.unknowns[1])
    @test count(!iszero, only(model.dofs.fields).dofs.constrained_values) > 0
end
