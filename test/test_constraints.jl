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

@testset "per-component Dirichlet frees the other components" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    u = field(:u, V; components=2)
    bottom = boundary(axis=2, side=:lower)
    top = boundary(axis=2, side=:upper)

    # only u_y constrained on top (u_x free); both fixed on the bottom
    per_component = prepare(poisson(u; source=x -> SVector(0.0, 0.0),
                                    dirichlet=[dirichlet(SVector(0.0, 0.0); on=bottom, field=u),
                                               dirichlet(0.1; on=top, field=u, component=2)]))
    all_components = prepare(poisson(u; source=x -> SVector(0.0, 0.0),
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

@testset "total-degree dof layout stays basis-local" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=3, mode=:total_degree)
    layout = Unfitted.dof_layout(V; dirichlet=[dirichlet(0.0; on=boundary(:all))])
    active = Unfitted.active_cell_dofs(layout, 1, CartesianIndex(1, 1))

    @test V.levels[1].mode == :total_degree
    @test Unfitted.raw_dof_count(layout) == 13
    @test Unfitted.active_unknowns(layout) == 1
    @test count(!iszero, active) == 1
    @test_throws ArgumentError space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=(2, 3),
                                     mode=:total_degree)

    V2 = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1))
    overlay_active = Unfitted.active_cell_dofs(Unfitted.dof_layout(V2), 2, CartesianIndex(1, 1))
    @test V2.levels[2].mode == :total_degree
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
    model = prepare(poisson(u; source=x -> SVector(0.0, 0.0),
                            dirichlet=[dirichlet(0.0; on=bottom, field=u, component=2),
                                       dirichlet(0.1; on=top, field=u, component=2),
                                       dirichlet(0.0; on=pin, field=u, component=1)]))
    sol = solve!(model)
    @test isposdef(Symmetric(Matrix(model.matrix)))
    @test value(sol, model, (0.0, 0.0), 1) ≈ 0.0 atol = 1.0e-10
    @test value(sol, model, (0.5, 1.0), 2) ≈ 0.1 atol = 1.0e-10
end

# The load-stepping driver pattern: a single `prepare(problem)` followed
# by `update_dirichlet!` between steps. The converged solution must
# match what a fresh `prepare(problem)` with the new Dirichlet datum
# produces, and the model.version must stay unchanged.
@testset "update_dirichlet! matches fresh prepare for value-only changes" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=4, order=2)

    function build(rhs_value)
        return poisson(V; source=x -> 0.0,
                       dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                                  dirichlet(rhs_value; on=boundary(axis=1, side=:upper))])
    end

    base = prepare(build(0.0))
    initial_version = base.version

    for new_value in (0.25, 0.5, -0.1)
        update_dirichlet!(base,
                          [dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                           dirichlet(new_value; on=boundary(axis=1, side=:upper))])
        @test base.version == initial_version
        @test base.matrix === nothing
        @test base.rhs === nothing
        s = solve!(base)

        reference = solve!(prepare(build(new_value)))
        @test value(s, base, (0.5,)) ≈ value(reference, prepare(build(new_value)), (0.5,)) atol = 1.0e-10
        @test value(s, base, (1.0,)) ≈ new_value atol = 1.0e-10
        @test value(s, base, (0.0,)) ≈ 0.0 atol = 1.0e-10
    end
end

# Structural checks: anything beyond a value-only change must throw.
@testset "update_dirichlet! rejects structural changes" begin
    V = space(box((0.0,), (1.0,)); cells=4, order=2)
    p = poisson(V; source=x -> 0.0,
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
    model = prepare(poisson(V; source=x -> 0.0,
                            dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                                       dirichlet(x -> 0.0; on=boundary(axis=1, side=:upper))]))
    update_dirichlet!(model,
                      [dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                       dirichlet(x -> 0.4; on=boundary(axis=1, side=:upper))])
    s = solve!(model)
    @test value(s, model, (1.0,)) ≈ 0.4 atol = 1.0e-10
    @test value(s, model, (0.5,)) ≈ 0.2 atol = 1.0e-10
end
