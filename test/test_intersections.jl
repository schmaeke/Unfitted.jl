@testset "intersections scaffold" begin
    @test isdefined(Unfitted, :VolumeRegion)
    @test isdefined(Unfitted, :IntegrationPlan)
end

# Every parent's frame map sends the region's frame corners `p = ∓1` — through `ξ = ξ0 ∓ dξ` in the cell's reference
# frame — back onto the region's own physical corners.
function assert_parent_maps(V, plan)
    for region in plan.regions
        for parent in region.parents
            parent_box = Unfitted.cell_box(V.levels[parent.level].mesh, parent.cell)
            @test parent.parent_box == parent_box
            lower = Unfitted.reference_to_physical(parent_box, parent.ξ0 - parent.dξ)
            upper = Unfitted.reference_to_physical(parent_box, parent.ξ0 + parent.dξ)

            @test all(isapprox(lower[i], region.lower[i]; atol=1.0e-12) for i in eachindex(lower))
            @test all(isapprox(upper[i], region.upper[i]; atol=1.0e-12) for i in eachindex(upper))
        end
        @test !isempty(region.points)
        @test length(region.points) == length(region.weights)
    end
    # merged regions must still partition the domain exactly (no lost coverage)
    @test sum(Unfitted.volume(Unfitted._region_box(region)) for region in plan.regions) ≈ Unfitted.volume(V.domain)
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
    V4 = overlay(V4, box((0.25, 0.25, 0.25, 0.25), (0.75, 0.75, 0.75, 0.75)); cells=(1, 1, 1, 1), order=1)
    plan4 = Unfitted.integration_plan(V4)
    @test length(plan4.regions) == 9    # merged (was 81 unmerged)
    @test count(region -> length(region.parents) == 2, plan4.regions) == 1
    assert_parent_maps(V4, plan4)
end

@testset "the integration-region criterion: both symbols and a callable" begin
    # `:any_parent` keeps every region some level covers, `:all_levels` only the overlap of every level, and a callable
    # receives each region's parent list. `parents -> length(parents) ≥ 2` spells `:all_levels` on a two-level overlay,
    # so the two must agree region for region — directly, through `prepare`, and after `move!` replays the captured
    # plan options.
    base = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2)
    V = overlay(base, box((0.2, 0.3), (0.6, 0.7)); cells=(2, 2), order=2)
    both = parents -> length(parents) >= 2
    boxes(plan) = [Unfitted._region_box(region) for region in plan.regions]

    any_plan = Unfitted.integration_plan(V)
    all_plan = Unfitted.integration_plan(V; criterion=:all_levels)
    fn_plan = Unfitted.integration_plan(V; criterion=both)
    @test sum(Unfitted.volume(b) for b in boxes(any_plan)) ≈ 1.0
    @test sum(Unfitted.volume(b) for b in boxes(all_plan)) ≈ 0.4 * 0.4
    @test all(region -> length(region.parents) == 2, all_plan.regions)
    @test length(all_plan.regions) < length(any_plan.regions)
    @test boxes(fn_plan) == boxes(all_plan)
    @test [r.parents for r in fn_plan.regions] == [r.parents for r in all_plan.regions]
    @test_throws ArgumentError Unfitted.integration_plan(V; criterion=:no_such_rule)

    bc = [dirichlet(0.0; on=boundary(:all))]
    to = box((0.35, 0.1), (0.75, 0.5))
    V_end = overlay(base, to; cells=(2, 2), order=2)
    model_boxes(model) = boxes(only(model.space_plans))
    direct = prepare(stiffness(V_end; dirichlet=bc); criterion=:all_levels)
    mover = prepare(stiffness(V; dirichlet=bc); criterion=both)
    @test model_boxes(mover) == boxes(all_plan)
    move!(mover; level=2, to)
    @test mover.plan_options.criterion === both
    @test model_boxes(mover) == model_boxes(direct)
    @test diagnostics(mover).integration_regions == diagnostics(direct).integration_regions
end

@testset "a callable criterion never sees a region no level covers" begin
    # Regression. Where no level is active — the base cell (2, 2) and the overlay cell over it are both masked off — the
    # partition still forms a candidate box, which carries no parent. `:any_parent` and `:all_levels` reject it by
    # definition, but a callable criterion used to be handed the empty parent list, and one returning `true` kept the
    # region and crashed sizing its rule (`maximum` over no parents). Parentless boxes are now dropped before any
    # criterion runs, so `_ -> true` is `:any_parent`.
    base_mask = trues(3, 3)
    base_mask[2, 2] = false
    over_mask = trues(2, 2)
    over_mask[1, 1] = false
    V = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=1, active=base_mask),
                box((1 / 3, 1 / 3), (1.0, 1.0)); cells=(2, 2), order=1, active=over_mask)
    saw_empty = Ref(false)
    keep_all = parents -> (isempty(parents) && (saw_empty[] = true); true)
    plan = Unfitted.integration_plan(V; criterion=keep_all)
    @test !saw_empty[]
    reference = Unfitted.integration_plan(V)
    @test [Unfitted._region_box(r) for r in plan.regions] == [Unfitted._region_box(r) for r in reference.regions]
    @test [r.parents for r in plan.regions] == [r.parents for r in reference.regions]
    @test sum(Unfitted.volume(Unfitted._region_box(r)) for r in plan.regions) ≈ 1 - 1 / 9
end

# `tolerance.merge` is an absolute length. A mesh whose spacing on some axis falls at or below it would have that axis's
# own element boundaries collapsed by `merge_coordinates`, leaving integration regions that straddle two cells —
# silently, since nothing downstream can see it. `integration_plan` rejects the combination instead. The threshold is
# `tolerance.merge` itself, so geometry that merely sits close to it keeps working, and small geometry stays fully
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

    # Same geometry with a tolerance scaled to it: fully supported, and every cell gets its own region back.
    scaled = GeometryTolerance(; merge=sqrt(eps(Float64)) * 1.0e-7)
    @test length(Unfitted.integration_plan(Vbad; tolerance=scaled).regions) == 4 * 8

    # A too-fine overlay is caught on the overlay level, named by its own id.
    Vfine = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1), box((0.4, 0.4), (0.4 + 1.0e-8, 0.6));
                    cells=(2, 2), order=1)
    err2 = try
        Unfitted.integration_plan(Vfine)
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("level 2", err2.msg)
    @test occursin("axis 1", err2.msg)
end

# Twin of the partition engine (`_partition`, every axis free and every cell eligible) that builds its candidate grid
# from every level's *full* node grid — the partition exactly as it was before the candidate coordinates were projected
# onto the active cells. Everything downstream is the package's own code (`_signature`, `_extend_axis`), so a difference
# between the two can only come from the coordinates. Like the engine it seeds no box at a candidate no level covers;
# `uncovered = true` emits those too, which is what a partition without that rule would hand its consumers.
function full_grid_boxes(levels::Tuple, ::Val{D}, tol::GeometryTolerance{T}; uncovered=false) where {D,T}
    intervals = ntuple(D) do d
        coords = T[]
        for level in levels
            append!(coords, Unfitted.boundary_coordinates(level.mesh)[d])
        end
        Unfitted._intervals_from_coordinates(Unfitted.merge_coordinates(coords, tol), tol)
    end
    ranges = ntuple(d -> length(intervals[d]), D)
    boxes = Tuple{Unfitted.SVector{D,T},Unfitted.SVector{D,T}}[]
    any(==(0), ranges) && return boxes
    linmaps = map(level -> LinearIndices(level.mesh.cells), levels)
    cells = map(_ -> Unfitted._AllCells(), levels)
    sigs = map(CartesianIndices(ranges)) do index
        mid = Unfitted.SVector{D,T}(ntuple(d -> (intervals[d][index.I[d]][1] + intervals[d][index.I[d]][2]) / 2, D))
        Unfitted._signature(levels, linmaps, cells, mid, tol)
    end
    visited = falses(ranges)
    for start in CartesianIndices(ranges)
        visited[start] && continue
        uncovered || !all(iszero, sigs[start]) || continue
        hi = start.I
        for d in 1:D
            hi = Unfitted._extend_axis(sigs, visited, ranges, start.I, hi, sigs[start], d)
        end
        top = hi
        visited[CartesianIndices(ntuple(e -> start.I[e]:top[e], D))] .= true
        push!(boxes,
              (Unfitted.SVector{D,T}(ntuple(d -> intervals[d][start.I[d]][1], D)),
               Unfitted.SVector{D,T}(ntuple(d -> intervals[d][top[d]][2], D))))
    end
    return boxes
end

# The corners of the engine's volume partition of `levels`, in emission order.
function partition_boxes(levels::Tuple, ::Val{D}, tol::GeometryTolerance{T}) where {D,T}
    boxes = Tuple{Unfitted.SVector{D,T},Unfitted.SVector{D,T}}[]
    Unfitted._partition((lower, upper, _) -> push!(boxes, (lower, upper)), levels, ntuple(identity, Val(D)),
                        zero(Unfitted.SVector{D,T}), tol)
    return boxes
end

@testset "the active-cell projection leaves the box partition unchanged" begin
    # A candidate coordinate the projection drops is the face of no active cell on any level, so every level's coverage
    # signature is constant across it and the greedy merge fuses the two slabs it separates anyway. The emitted box list
    # must therefore come out identical — box for box, in order, and bit for bit — to the one the full node grids
    # produce. `isequal` compares the corner coordinates bit for bit (it tells −0.0 from 0.0, which `==` does not).
    #
    # Neither side seeds a box where no level is live at all — outside a masked base level, say — where the projected
    # partition stops at the outermost live face instead of tiling the dead remainder; everywhere else the two lists
    # must agree exactly.
    tol = GeometryTolerance(Float64)

    # 1D: an overlay live on one of its cells.
    V1 = ladder(box((0.0,), (1.0,)); cells=4, order=2, depth=2, splits=2)
    V1 = adapt(V1, 3 => [CartesianIndex(5), CartesianIndex(6)])
    @test isequal(partition_boxes(V1.levels, Val(1), tol), full_grid_boxes(V1.levels, Val(1), tol))

    # 2D and 3D ladders with a corner-anchored chain: every level spans the whole domain, so this is the configuration
    # the projection exists for.
    for (D, base, depth) in ((2, 6, 3), (3, 3, 2))
        dom = box(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D))
        V = ladder(dom; cells=base, order=2, depth=depth, splits=2)
        for k in 2:(depth+1)
            m = falses(V.levels[k].mesh.cells)
            m[CartesianIndices(ntuple(_ -> 1:2, D))] .= true
            V = adapt(V, k => m)
        end
        @test isequal(partition_boxes(V.levels, Val(D), tol), full_grid_boxes(V.levels, Val(D), tol))
        # Non-vacuity: the projection really did shrink the candidate grid.
        @test length(Unfitted._merged_axis_coordinates(V.levels, Val(D), tol)[1]) <
              length(Unfitted.merge_coordinates(vcat((Unfitted.boundary_coordinates(l.mesh)[1] for l in V.levels)...),
                                                tol))
    end

    # A masked, *non-nested* overlay over a masked base level: nothing here divides anything else evenly, so the merged
    # coordinate set is not a subset relation between grids.
    base_mask = falses(5, 5)
    base_mask[1:4, 2:5] .= true
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(5, 5), order=2, active=base_mask)
    over_mask = falses(3, 3)
    over_mask[1:2, 2:3] .= true
    V = overlay(V, box((0.13, 0.21), (0.79, 0.86)); cells=(3, 3), order=2, active=over_mask)
    @test isequal(partition_boxes(V.levels, Val(2), tol), full_grid_boxes(V.levels, Val(2), tol))
    # Non-vacuity: the full grid really does carry boxes no level covers here — the ones the rule of seeding no such box
    # keeps out of both lists — so the masked case exercises it.
    @test length(full_grid_boxes(V.levels, Val(2), tol; uncovered=true)) >
          length(partition_boxes(V.levels, Val(2), tol))
    # And skipping them changes no other box: a covered seed only ever absorbs candidates of its own signature, never an
    # uncovered one, so the partition that does emit the uncovered boxes, filtered down to the covered ones, is the
    # engine's list box for box, in order. Coverage is read the engine's way, from the signature at the box midpoint.
    linmaps = map(level -> LinearIndices(level.mesh.cells), V.levels)
    cells = map(_ -> Unfitted._AllCells(), V.levels)
    covered((lower, upper)) = !all(iszero, Unfitted._signature(V.levels, linmaps, cells, (lower + upper) / 2, tol))
    @test isequal(filter(covered, full_grid_boxes(V.levels, Val(2), tol; uncovered=true)),
                  partition_boxes(V.levels, Val(2), tol))
end

@testset "a dropped coordinate can move a canonical one by at most tol.merge" begin
    # `merge_coordinates` keeps the *first* representative of every group within `tol.merge`. A dead line that sat
    # inside that window, below a live line of another level, was that group's representative, so dropping it promotes
    # the survivor and the canonical coordinate moves — by at most `tol.merge`, never more. That is the whole gap
    # between "bit-identical" (a nested stack, where coincident nodes are equal to the last bit) and "identical up to
    # `tol.merge`" (anything else), and it is stated that way in `_live_axis_coordinates` because of this case.
    tol = GeometryTolerance(Float64)
    eps_shift = 2.0e-9
    @test eps_shift < tol.merge
    V = space(box((0.0,), (1.0,)); cells=2, order=1)
    V = overlay(V, box((0.0,), (1.0 - eps_shift,)); cells=2, order=1, active=[CartesianIndex(1)])

    projected = partition_boxes(V.levels, Val(1), tol)
    full = full_grid_boxes(V.levels, Val(1), tol)
    @test length(projected) == length(full)
    @test projected != full                                  # the caveat is real
    for ((plower, pupper), (flower, fupper)) in zip(projected, full)
        @test maximum(abs, plower - flower) <= tol.merge
        @test maximum(abs, pupper - fupper) <= tol.merge
    end
end
