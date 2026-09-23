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

# Twin of `_merged_boxes` that builds its candidate grid from every level's
# *full* node grid — the partition exactly as it was before the candidate
# coordinates were projected onto the active cells. Everything downstream is
# the package's own code (`_coverage_signature`, `_extend_axis`,
# `_emit_merged_box!`), so a difference between the two can only come from the
# coordinates.
function full_grid_boxes(levels::Tuple, ::Val{D}, tol::GeometryTolerance{T}) where {D,T}
    intervals = ntuple(D) do d
        coords = T[]
        for level in levels
            append!(coords, Unfitted.boundary_coordinates(level.mesh)[d])
        end
        Unfitted._intervals_from_coordinates(Unfitted.merge_coordinates(coords, tol), tol)
    end
    ranges = ntuple(d -> length(intervals[d]), D)
    boxes = Unfitted.AxisBox{D,T}[]
    any(==(0), ranges) && return boxes
    linmaps = map(level -> LinearIndices(level.mesh.cells), levels)
    sigs = map(CartesianIndices(ranges)) do index
        mid = Unfitted.SVector{D,T}(ntuple(d -> (intervals[d][index.I[d]][1] +
                                                 intervals[d][index.I[d]][2]) / 2, D))
        Unfitted._coverage_signature(levels, linmaps, mid, tol)
    end
    visited = falses(ranges)
    for start in CartesianIndices(ranges)
        visited[start] && continue
        hi = start.I
        for d in 1:D
            hi = Unfitted._extend_axis(sigs, visited, ranges, start.I, hi, sigs[start], d)
        end
        Unfitted._emit_merged_box!(boxes, visited, intervals, start.I, hi)
    end
    return boxes
end

# The boxes of a partition that any level actually covers — the only ones any
# consumer of `_merged_boxes` keeps.
function covered_boxes(levels::Tuple, boxes, tol)
    return filter(b -> !isempty(Unfitted._parents_covering(levels, b, tol)), boxes)
end

@testset "the active-cell projection leaves the box partition unchanged" begin
    # A candidate coordinate the projection drops is the face of no active cell
    # on any level, so every level's coverage signature is constant across it
    # and the greedy merge fuses the two slabs it separates anyway. The emitted
    # box list must therefore come out identical — box for box, in order, and
    # bit for bit — to the one the full node grids produce. `AxisBox` is an
    # isbits struct, so `==` on these vectors *is* the bitwise comparison.
    #
    # The comparison is on the *covered* boxes. Where no level is live at all —
    # outside a masked base level, say — the projected partition stops at the
    # outermost live face instead of tiling the dead remainder, and the boxes
    # that differ are exactly the parentless ones every consumer drops. On a
    # `ladder`, whose base level is never masked, even the unfiltered lists
    # agree, and that is asserted separately below.
    tol = GeometryTolerance(Float64)

    # 1D: an overlay live on one of its cells.
    V1 = ladder(box((0.0,), (1.0,)); cells=4, order=2, depth=2, splits=2)
    V1 = adapt(V1, 3 => [CartesianIndex(5), CartesianIndex(6)])
    @test Unfitted._merged_boxes(V1.levels, Val(1), tol) == full_grid_boxes(V1.levels, Val(1), tol)

    # 2D and 3D ladders with a corner-anchored chain: every level spans the
    # whole domain, so this is the configuration the projection exists for.
    for (D, base, depth) in ((2, 6, 3), (3, 3, 2))
        dom = box(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D))
        V = ladder(dom; cells=base, order=2, depth=depth, splits=2)
        for k in 2:(depth+1)
            m = falses(V.levels[k].mesh.cells)
            m[CartesianIndices(ntuple(_ -> 1:2, D))] .= true
            V = adapt(V, k => m)
        end
        @test Unfitted._merged_boxes(V.levels, Val(D), tol) ==
              full_grid_boxes(V.levels, Val(D), tol)
        # Non-vacuity: the projection really did shrink the candidate grid.
        @test length(Unfitted._merged_axis_coordinates(V.levels, Val(D), tol)[1]) <
              length(Unfitted.merge_coordinates(vcat((Unfitted.boundary_coordinates(l.mesh)[1]
                                                      for l in V.levels)...), tol))
    end

    # A masked, *non-nested* overlay over a masked base level: nothing here
    # divides anything else evenly, so the merged coordinate set is not a
    # subset relation between grids.
    base_mask = falses(5, 5)
    base_mask[1:4, 2:5] .= true
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(5, 5), order=2, active=base_mask)
    over_mask = falses(3, 3)
    over_mask[1:2, 2:3] .= true
    V = overlay(V, box((0.13, 0.21), (0.79, 0.86)); cells=(3, 3), order=2, active=over_mask)
    @test covered_boxes(V.levels, Unfitted._merged_boxes(V.levels, Val(2), tol), tol) ==
          covered_boxes(V.levels, full_grid_boxes(V.levels, Val(2), tol), tol)
    # Non-vacuity: the unfiltered lists really do differ here, so the filter is
    # carrying the claim rather than hiding a failure.
    @test Unfitted._merged_boxes(V.levels, Val(2), tol) != full_grid_boxes(V.levels, Val(2), tol)
end

@testset "a dropped coordinate can move a canonical one by at most tol.merge" begin
    # `merge_coordinates` keeps the *first* representative of every group
    # within `tol.merge`. A dead line that sat inside that window, below a live
    # line of another level, was that group's representative, so dropping it
    # promotes the survivor and the canonical coordinate moves — by at most
    # `tol.merge`, never more. That is the whole gap between "bit-identical"
    # (a nested stack, where coincident nodes are equal to the last bit) and
    # "identical up to `tol.merge`" (anything else), and it is stated that way
    # in `_live_axis_coordinates` because of this case.
    tol = GeometryTolerance(Float64)
    eps_shift = 2.0e-9
    @test eps_shift < tol.merge
    V = space(box((0.0,), (1.0,)); cells=2, order=1)
    V = overlay(V, box((0.0,), (1.0 - eps_shift,)); cells=2, order=1, active=[CartesianIndex(1)])

    projected = Unfitted._merged_boxes(V.levels, Val(1), tol)
    full = full_grid_boxes(V.levels, Val(1), tol)
    @test length(projected) == length(full)
    @test projected != full                                  # the caveat is real
    for (p, f) in zip(projected, full)
        @test maximum(abs, p.lower - f.lower) <= tol.merge
        @test maximum(abs, p.upper - f.upper) <= tol.merge
    end
end
