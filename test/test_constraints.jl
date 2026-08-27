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
    @test haskey(model.facet_regions,
                 (boundary(axis=1, side=:upper), model.problem.space))
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
