@testset "intersections scaffold" begin
    @test isdefined(Unfitted, :VolumeRegion)
    @test isdefined(Unfitted, :IntegrationPlan)
end

function assert_parent_maps(V, plan)
    for region in plan.regions
        for parent in region.parents
            parent_box = Unfitted.cell_box(V.levels[parent.level].mesh, parent.cell)
            @test parent.parent_box == parent_box
            lower = Unfitted.reference_to_physical(parent_box, parent.local_box.lower)
            upper = Unfitted.reference_to_physical(parent_box, parent.local_box.upper)

            @test all(isapprox(lower[i], region.box.lower[i]; atol=1.0e-12)
                      for i in eachindex(lower))
            @test all(isapprox(upper[i], region.box.upper[i]; atol=1.0e-12)
                      for i in eachindex(upper))
        end
        @test !isempty(region.quadrature.points)
        @test length(region.quadrature.points) == length(region.quadrature.weights)
    end
    # merged regions must still partition the domain exactly (no lost coverage)
    @test sum(Unfitted.volume(region.box) for region in plan.regions) ≈ Unfitted.volume(V.domain)
end

@testset "integration regions" begin
    V1 = space(box((0.0,), (1.0,)); cells=2, order=1)
    plan1 = Unfitted.integration_plan(V1)
    @test length(plan1.regions) == 2
    @test all(length(region.parents) == 1 for region in plan1.regions)
    assert_parent_maps(V1, plan1)

    V1o = overlay(V1, box((0.25,), (0.75,)); cells=2, order=1)
    plan1o = Unfitted.integration_plan(V1o)
    @test length(plan1o.regions) == 4
    @test sort(length.(getfield.(plan1o.regions, :parents))) == [1, 1, 2, 2]
    assert_parent_maps(V1o, plan1o)

    V2 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    V2 = overlay(V2, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=1)
    plan2 = Unfitted.integration_plan(V2)
    @test length(plan2.regions) == 12   # same-coverage boxes merged (was 16 unmerged)
    @test count(region -> length(region.parents) == 2, plan2.regions) == 4
    assert_parent_maps(V2, plan2)

    V3 = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(1, 1, 1), order=1)
    V3 = overlay(V3, box((0.25, 0.25, 0.25), (0.75, 0.75, 0.75)); cells=(1, 1, 1), order=1)
    plan3 = Unfitted.integration_plan(V3)
    @test length(plan3.regions) == 7    # merged (was 27 unmerged)
    @test count(region -> length(region.parents) == 2, plan3.regions) == 1
    assert_parent_maps(V3, plan3)

    V4 = space(box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)); cells=(1, 1, 1, 1), order=1)
    V4 = overlay(V4, box((0.25, 0.25, 0.25, 0.25), (0.75, 0.75, 0.75, 0.75)); cells=(1, 1, 1, 1),
                 order=1)
    plan4 = Unfitted.integration_plan(V4)
    @test length(plan4.regions) == 9    # merged (was 81 unmerged)
    @test count(region -> length(region.parents) == 2, plan4.regions) == 1
    assert_parent_maps(V4, plan4)
end

# `tolerance.merge` is an absolute length. A mesh whose spacing on some axis
# falls at or below it would have that axis's own element boundaries collapsed
# by `merge_coordinates`, leaving integration regions that straddle two cells —
# silently, since nothing downstream can see it. `integration_plan` rejects the
# combination instead. The threshold is `tolerance.merge` itself, so geometry
# that merely sits close to it keeps working, and small geometry stays fully
# supported as soon as the tolerance is scaled to match it.
@testset "integration regions merge tolerance vs mesh spacing" begin
    merge_tol = sqrt(eps(Float64))

    # h > merge on both axes: unaffected, one region per cell.
    Vok = space(box((0.0, 0.0), (1.0, 1.0e-6)); cells=(2, 67), order=1)
    @test 1.0e-6 / 67 > merge_tol
    @test length(Unfitted.integration_plan(Vok).regions) == 2 * 67

    # h ≤ merge on axis 2: rejected, and the message has to be actionable.
    Vbad = space(box((0.0, 0.0), (1.0, 1.0e-7)); cells=(4, 8), order=1)
    err = try
        Unfitted.integration_plan(Vbad)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("axis 2", err.msg)
    @test occursin(repr(1.0e-7 / 8), err.msg)          # the offending spacing h
    @test occursin(repr(merge_tol), err.msg)           # the tolerance it violates
    @test occursin("GeometryTolerance", err.msg)       # the remedy

    # Same geometry with a tolerance scaled to it: fully supported, and every
    # cell gets its own region back.
    scaled = GeometryTolerance(; merge=sqrt(eps(Float64)) * 1.0e-7)
    @test length(Unfitted.integration_plan(Vbad; tolerance=scaled).regions) == 4 * 8

    # A too-fine overlay is caught on the overlay level, named by its own id.
    Vfine = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1),
                    box((0.4, 0.4), (0.4 + 1.0e-8, 0.6)); cells=(2, 2), order=1)
    err2 = try
        Unfitted.integration_plan(Vfine)
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("level 2", err2.msg)
    @test occursin("axis 1", err2.msg)
end
