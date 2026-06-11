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
