# Tests for nested refinement ladders: declaration, per-level activation, and
# cross-level cell mapping.
#
# The property the ladder exists to guarantee is that covered-mode pruning is
# *lossless* — that the reduced stack spans exactly what the unreduced one
# spans. Rank alone cannot see the failure: off the nested manifold the reduced
# operator is still full rank at a healthy condition number, which is what makes
# the failure silent in the first place. Every losslessness check below therefore
# compares against the same geometry with leaf semantics switched off,
# and asserts `size(A_reduced, 1) == rank(A_unreduced)`.

using BasicBSpline
using LinearAlgebra

const LADDER_OMEGA = box((0.0, 0.0), (1.0, 1.0))
ladder_bc() = [dirichlet(0.0; on=boundary(:all))]

# Gram matrix of the active basis: rank equals the number of active dofs iff the
# retained functions are linearly independent. `prune = false` is `prepare`'s
# documented diagnostic — the same geometry with leaf semantics switched off,
# which is what the reduced space is supposed to span, and what this file
# measures every losslessness claim against.
function gram(V; prune::Bool=true)
    m = prepare(mass(V); prune=prune)
    assemble!(m)
    return Matrix(m.matrix)
end

# `true` iff reduction removed redundancy and nothing else.
lossless(V) = size(gram(V), 1) == rank(gram(V; prune=false))

@testset "ladder declares a nested, inert stack" begin
    V = ladder(LADDER_OMEGA; cells=8, order=3, depth=3, splits=2)
    @test length(V.levels) == 4
    @test [l.mesh.cells for l in V.levels] == [(8, 8), (16, 16), (32, 32), (64, 64)]
    @test is_nested(V)
    @test all(k -> !any(active_cells(V; level=k)), 2:4)
    # inert: the declared stack carries the base level's unknowns and no more
    base = prepare(poisson(space(LADDER_OMEGA; cells=8, order=3); source=1.0,
                           dirichlet=ladder_bc()))
    full = prepare(poisson(V; source=1.0, dirichlet=ladder_bc()))
    @test diagnostics(full).active_unknowns == diagnostics(base).active_unknowns
end

@testset "splits: a Tuple is per axis, a Vector is per level" begin
    a = ladder(LADDER_OMEGA; cells=6, order=2, depth=2, splits=(3, 1))
    b = ladder(LADDER_OMEGA; cells=6, order=2, depth=2, splits=[(2, 2), (3, 1)])
    @test [l.mesh.cells for l in a.levels] == [(6, 6), (18, 6), (54, 6)]
    @test [l.mesh.cells for l in b.levels] == [(6, 6), (12, 12), (36, 12)]
    @test is_nested(a) && is_nested(b)
    @test_throws ArgumentError ladder(LADDER_OMEGA; cells=6, depth=3, splits=[2, 2])
end

@testset "covered-mode pruning on a ladder is lossless" begin
    # The feature, in one assertion. A ladder refined over a block of base cells
    # sheds modes; the reduced space must still span the unreduced one.
    for depth in 1:3
        V = ladder(LADDER_OMEGA; cells=4, order=3, depth=depth, splits=2)
        spec = [k => overlapping_cells(V, [CartesianIndex(1, 1), CartesianIndex(2, 2)]; from=1,
                                       to=k) for k in 2:(depth+1)]
        W = adapt(V, spec...)
        @test is_nested(W)
        @test rank(gram(W)) == size(gram(W), 1)          # nothing left dependent
        @test lossless(W)                                 # and nothing real removed
    end
end

@testset "a non-nested stack is NOT lossless — the failure the ladder prevents" begin
    # 3 overlay cells over a 2-base-cell span is not an integer multiple, so the
    # levels do not nest and `:coverage` removes modes nothing reproduces.
    V = overlay(space(LADDER_OMEGA; cells=4, order=3), box((0.25, 0.25), (0.75, 0.75)); cells=3)
    @test !is_nested(V)
    @test size(gram(V), 1) < rank(gram(V; prune=false))
    # the same geometry at an integer multiple nests, and is lossless
    W = overlay(space(LADDER_OMEGA; cells=4, order=3), box((0.25, 0.25), (0.75, 0.75)); cells=4)
    @test is_nested(W)
    @test lossless(W)
end

@testset "a mode covered by two different levels is retained" begin
    # Split cover: two levels each cover part of a mode's stencil, so neither
    # reproduces it. `:coverage` must not fire on the union.
    V = ladder(LADDER_OMEGA; cells=4, order=3, depth=2, splits=2)
    left = falses(8, 8)
    left[1:4, 1:4] .= true                                 # level 2 over base (1:2, 1:2)
    right = falses(16, 16)
    right[9:16, 1:8] .= true                               # level 3 over base (3:4, 1:2)
    W = adapt(V, 2 => left, 3 => right)
    @test lossless(W)
end

@testset "a lower-order cover still sheds — that is the point of the rule" begin
    # Covered-mode pruning is not a claim that the cover reproduces what it removes;
    # that is the dedup half's job. It is an opt-in trade: over a region a finer
    # level resolves, the coarse cell's high-order modes buy almost nothing in L²
    # and carry the oscillation. A high-order base under a low-order fine overlay
    # is the configuration the rule exists for, so it must still fire there.
    coarse_cover = overlay(space(LADDER_OMEGA; cells=4, order=3), LADDER_OMEGA; cells=8, order=1)
    @test is_nested(coarse_cover)
    @test diagnostics(prepare(mass(coarse_cover))).reduced_mode_counts[1] > 0
    equal_order = overlay(space(LADDER_OMEGA; cells=4, order=3), LADDER_OMEGA; cells=8, order=3)
    @test diagnostics(prepare(mass(equal_order))).reduced_mode_counts[1] > 0
    # At equal order the cover does reproduce what it displaces, so the reduction
    # is lossless. At lower order it is not, and that is the documented trade —
    # `prepare(mass(V); prune=false)` is how this file switches it off.
    @test lossless(equal_order)
end

@testset "activation is unconstrained between levels" begin
    # None of these is required to be hierarchy-consistent, and all are
    # admissible. Rank is asserted; conditioning is not, because a mask boundary
    # that cuts across the level below legitimately degrades it.
    V = ladder(LADDER_OMEGA; cells=8, order=3, depth=3, splits=2)
    skip = falses(64, 64)
    skip[29:36, 29:36] .= true                             # level 4 alone, 2 and 3 dormant
    disjoint = falses(64, 64)
    disjoint[10:14, 10:14] .= true
    disjoint[50:54, 50:54] .= true
    for (name, spec) in ("level skip" => (4 => skip), "disjoint patches" => (4 => disjoint))
        W = adapt(V, spec)
        A = gram(W)
        @test rank(A) == size(A, 1)
        @test !any(active_cells(W; level=2))
    end
end

@testset "overlapping_cells maps both directions" begin
    V = ladder(LADDER_OMEGA; cells=8, order=2, depth=2, splits=2)
    down = overlapping_cells(V, [CartesianIndex(3, 3)]; from=1, to=3)
    @test count(down) == 16 && all(down[9:12, 9:12])
    @test count(overlapping_cells(V, down; from=3, to=1)) == 1
    part = falses(32, 32)
    part[9, 9] = true                                      # one child marks its parent
    @test count(overlapping_cells(V, part; from=3, to=1)) == 1
    @test overlapping_cells(V, down; from=3, to=3) == down
end

@testset "overlapping_cells on an anisotropic ladder" begin
    V = ladder(LADDER_OMEGA; cells=6, order=2, depth=2, splits=[(2, 2), (3, 1)])
    m2 = falses(12, 12)
    m2[3, 5] = true                                        # level 3 is (36, 12)
    m3 = overlapping_cells(V, m2; from=2, to=3)
    @test count(m3) == 3 && all(m3[7:9, 5])
end

@testset "overlapping_cells respects a level that does not span the domain" begin
    # A sub-box overlay: cells of the base outside the overlay must map to
    # nothing, not be credited to the overlay's edge cell.
    V = overlay(space(LADDER_OMEGA; cells=8, order=3), box((0.25, 0.25), (0.75, 0.75)); cells=8)
    corner = falses(8, 8)
    corner[1, 1] = true                                    # base cell [0, 0.125]², outside
    @test !any(overlapping_cells(V, corner; from=1, to=2))
    inner = falses(8, 8)
    inner[3, 3] = true                                     # base cell [0.25, 0.375]², inside
    @test count(overlapping_cells(V, inner; from=1, to=2)) == 4
    one_overlay = falses(8, 8)
    one_overlay[1, 1] = true                               # overlay cell [0.25, 0.3125]²
    up = overlapping_cells(V, one_overlay; from=2, to=1)
    @test count(up) == 1 && up[3, 3]
end

@testset "cell mapping is exact away from the unit scale" begin
    # The mapping must not depend on an absolute tolerance: a domain far from the
    # origin has ulps larger than one, and a domain of small extent has cells
    # smaller than the geometry tolerance.
    for dom in (box((1e10, 1e10), (1e10 + 1.0, 1e10 + 1.0)), box((0.0, 0.0), (1e-7, 1e-7)),
                box((-3.0, -7.0), (-1.0, -5.0)))
        V = ladder(dom; cells=4, order=2, depth=2, splits=2,
                   tolerance=GeometryTolerance(Float64; merge=1e-30))
        d = zeros(Int, 4, 4)
        d[3, 3] = 2
        W = adapt(V, d)
        @test count(active_cells(W; level=2)) == 4
        @test count(active_cells(W; level=3)) == 16
        @test is_nested(W; tolerance=GeometryTolerance(Float64; merge=1e-30))
    end
end

@testset "adapt: pair form sets, leaves others alone, does not alias" begin
    V = ladder(LADDER_OMEGA; cells=8, order=2, depth=2, splits=2)
    m = falses(32, 32)
    m[5:9, 5:9] .= true
    W = adapt(V, 3 => m)
    @test active_cells(W; level=3) == m
    @test !any(active_cells(W; level=2))
    @test !any(active_cells(V; level=3))                   # the source is untouched
    m[1, 1] = true                                          # and the caller's array is not shared
    @test !active_cells(W; level=3)[1, 1]
end

@testset "adapt: rejected specs" begin
    V = ladder(LADDER_OMEGA; cells=4, order=2, depth=2, splits=2)
    @test_throws ArgumentError adapt(V, fill(3, 4, 4))                  # depth out of range
    @test_throws ArgumentError adapt(V, fill(-1, 4, 4))
    @test_throws DimensionMismatch adapt(V, zeros(Int, 5, 5))
    @test_throws ArgumentError adapt(V, 9 => trues(4, 4))               # no such level
    @test_throws ArgumentError adapt(V, 2 => trues(8, 8), 2 => trues(8, 8))
    @test_throws DimensionMismatch adapt(V, 2 => trues(4, 4))           # wrong shape
    @test_throws ArgumentError adapt(V, trues(4, 4))                    # Bool is not a depth map
end

@testset "documented shapes: base active=, selection shapes, array over pair, vacuous nesting" begin
    # Four shapes the docstrings promise and nothing exercised. The base-level
    # `active =` one is why `depth_masks` names only the overlay levels: a base
    # cell at depth 0 is one no overlay refines, not one the base drops, and an
    # array form that spoke for the base at all would activate the very cells the
    # caller deactivated.
    m = trues(4, 4)
    m[1, 1] = false
    V = ladder(LADDER_OMEGA; cells=4, order=2, depth=1, splits=2, active=m)
    @test active_cells(V; level=1) == m                     # `active =` reaches the base
    @test active_cells(adapt(V, zeros(Int, 4, 4)); level=1) == m       # array form leaves it alone
    @test active_cells(adapt(V, 2 => trues(8, 8)); level=1) == m       # so does the pair form

    L = ladder(LADDER_OMEGA; cells=4, order=2, depth=1, splits=2)
    listed = overlapping_cells(L, [CartesianIndex(2, 2)]; from=1, to=2)
    @test count(listed) == 4
    @test overlapping_cells(L, (_, ci) -> ci == CartesianIndex(2, 2); from=1, to=2) == listed
    sel = falses(4, 4)
    sel[2, 2] = true
    @test overlapping_cells(L, Unfitted.LevelMask{2}(sel); from=1, to=2) == listed
    @test all(overlapping_cells(L, nothing; from=1, to=2))  # `nothing` is every cell

    # The array form overwrites an overlay mask rather than merging with it.
    P = adapt(L, 2 => trues(8, 8))
    @test !any(active_cells(adapt(P, zeros(Int, 4, 4)); level=2))

    # `is_nested` passes vacuously where the finer level's box contains no node
    # of the level below, which its docstring states and no test asserted.
    @test is_nested(overlay(space(LADDER_OMEGA; cells=4, order=2), box((0.3, 0.3), (0.45, 0.45));
                            cells=3))
end

@testset "the per-cell block and the whole-level map are the same rule" begin
    # `overlapping_cells` answers for a selection by walking the destination
    # level; `_cell_block` answers for one cell by binary search, which is what
    # the adaptivity verbs ask and what makes a mark cost microseconds instead of
    # milliseconds. The two must agree everywhere, in both directions, including
    # on a stack that is neither nested nor aligned nor isotropic — and including
    # the empty answer a level whose box misses the cell must give.
    V = overlay(overlay(space(LADDER_OMEGA; cells=4, order=2), box((0.3, 0.1), (0.8, 0.9));
                        cells=3), box((0.15, 0.55), (0.65, 0.95)); cells=5)
    @test !is_nested(V)
    agrees = true
    empties = 0
    for from in 1:length(V.levels), to in 1:length(V.levels)
        from == to && continue
        for c in cell_indices(V; level=from)
            block = Unfitted._cell_block(V, from, to, c)
            isempty(block) && (empties += 1)
            got = falses(V.levels[to].mesh.cells)
            got[block] .= true
            agrees &= (got == overlapping_cells(V, [c]; from=from, to=to))
        end
    end
    @test agrees
    @test empties > 0                                       # the sub-box cases really occur
end

@testset "depth_masks and grading" begin
    V = ladder(LADDER_OMEGA; cells=8, order=2, depth=3, splits=2)
    d = zeros(Int, 8, 8)
    d[4, 4] = 3
    @test first.(Unfitted.depth_masks(V, d)) == 2:4        # the base is never named
    plain = adapt(V, d)
    @test count(active_cells(plain; level=4)) == 8^2       # one base cell, 8x8 children
    graded = adapt(V, d; grade=1)
    @test count(active_cells(graded; level=4)) == 8^2      # grading never lowers a request
    @test count(active_cells(graded; level=2)) > count(active_cells(plain; level=2))
    # the fixed point does not depend on where the feature sits
    near, far = zeros(Int, 8, 8), zeros(Int, 8, 8)
    near[1, 1] = 3
    far[8, 8] = 3
    @test count(active_cells(adapt(V, near; grade=1); level=2)) ==
          count(active_cells(adapt(V, far; grade=1); level=2))
    @test_throws ArgumentError adapt(V, d; grade=-1)
end

@testset "depth_masks pairs splat into adapt, and an all-active mask is no mask" begin
    # `depth_masks` is the internal behind `adapt`'s array form and its result is
    # documented as splatting straight into the pair form. That identity is the
    # whole contract, so assert it rather than the shape.
    V = ladder(LADDER_OMEGA; cells=4, order=2, depth=2, splits=2)
    d = zeros(Int, 4, 4)
    d[2, 3] = 2
    by_array = adapt(V, d)
    by_pairs = adapt(V, Unfitted.depth_masks(V, d)...)
    @test all(k -> active_cells(by_array; level=k) == active_cells(by_pairs; level=k), 1:3)

    # A ladder with no overlay names no level, and the array form is then the
    # identity rather than an error.
    flat = ladder(LADDER_OMEGA; cells=4, order=2, depth=0)
    @test isempty(Unfitted.depth_masks(flat, zeros(Int, 4, 4)))
    @test active_cells(adapt(flat, zeros(Int, 4, 4)); level=1) == trues(4, 4)

    # An all-active selection is stored as the no-mask default, whichever route
    # it arrives by, so the coverage and dof fast paths survive a round trip.
    @test space(LADDER_OMEGA; cells=4, order=2, active=trues(4, 4)).levels[1].mask === nothing
    @test adapt(V, 2 => trues(8, 8)).levels[2].mask === nothing
    @test adapt(V, 2 => cell_indices(V; level=2)).levels[2].mask === nothing
    full = fill(2, 4, 4)
    @test adapt(V, full).levels[3].mask === nothing
end

@testset "is_nested is asserted false where it should be" begin
    base = space(LADDER_OMEGA; cells=4, order=2)
    # corners off the parent's node coordinates
    @test !is_nested(overlay(base, box((0.3, 0.3), (0.8, 0.8)); cells=4))
    # corners aligned but the cell count is not an integer multiple of the span
    @test !is_nested(overlay(base, box((0.25, 0.25), (0.75, 0.75)); cells=3))
    @test is_nested(overlay(base, box((0.25, 0.25), (0.75, 0.75)); cells=4))
end

@testset "ladder rejects a depth it cannot represent" begin
    @test_throws ArgumentError ladder(LADDER_OMEGA; cells=4, depth=-1)
    # spacing below the merge tolerance is caught at construction, not at prepare
    @test_throws ArgumentError ladder(box((0.0,), (1.0,)); cells=1, order=2, depth=30, splits=2)
    # so is a level too large to allocate
    @test_throws ArgumentError ladder(LADDER_OMEGA; cells=8, order=2, depth=20, splits=2)
end

@testset "adapted leaves the source intact and keeps the Space type" begin
    V = ladder(LADDER_OMEGA; cells=8, order=2, depth=2, splits=2)
    m = falses(32, 32)
    m[9:24, 9:24] .= true
    model = prepare(poisson(adapt(V, 3 => m); source=1.0, dirichlet=ladder_bc()))
    solve!(model)
    before = diagnostics(model).active_unknowns
    m2 = copy(m)
    m2[5:28, 5:28] .= true
    target = adapted(model, 3 => m2)
    @test diagnostics(model).active_unknowns == before
    @test diagnostics(target).active_unknowns > before
    @test typeof(target.problem.space) === typeof(model.problem.space)
    @test is_nested(target.problem.space)
end

@testset "L2Projection is the transfer for a refine step" begin
    V = ladder(LADDER_OMEGA; cells=8, order=3, depth=2, splits=2)
    m2 = falses(16, 16)
    m2[5:12, 5:12] .= true
    model = prepare(poisson(adapt(V, 2 => m2); source=1.0, dirichlet=ladder_bc()))
    u = solve!(model)
    m3 = falses(32, 32)
    m3[13:20, 13:20] .= true
    target = adapted(model, 2 => m2, 3 => m3)
    # Refining deepens coverage, so the parent's modes are eliminated in the
    # target: the target basis is not a superset of the source's and `Rewire`
    # has no counterpart to copy into.
    @test_throws ArgumentError transfer(u, model, target; via=Rewire())
    moved_u = transfer(u, model, target; via=L2Projection())
    pts = [(x, y) for x in 0.05:0.1:0.95 for y in 0.05:0.1:0.95]
    @test maximum(abs(value(moved_u, target, p) - value(u, model, p)) for p in pts) < 1e-11
end

@testset "adapted round-trips the pre-fold mask on an immersed model" begin
    disc = physical_domain(x -> sqrt(x[1]^2 + x[2]^2) - 0.7; subcell_length_scale=0.25)
    V = ladder(box((-1.0, -1.0), (1.0, 1.0)); cells=8, order=2, depth=2, splits=2, physical=disc)
    model = prepare(mass(adapt(V, 3 => trues(32, 32))))
    before = diagnostics(model)
    # Reading the pre-fold mask and writing it back must change nothing. Reading
    # the *effective* mask would record the fictitious fold as user intent.
    same = adapted(model, 3 => active_cells(model; level=3, effective=false))
    @test diagnostics(same).active_unknowns == before.active_unknowns
    @test diagnostics(same).inactive_cell_counts == before.inactive_cell_counts
    @test active_cells(same; level=3, effective=false) ==
          active_cells(model; level=3, effective=false)
    # and the fold is genuinely applied, not skipped
    @test any(>(0), before.inactive_cell_counts)
    @test diagnostics(model, solve!(model)).fit_failure_count == 0
end

# A cut-cell rule that counts the times it is asked for one. `cut_quadrature` is
# the documented extension point (`physical_domain`), it sits behind exactly the
# per-`(box, moment order)` memoisation the moment fit does, and the plan builds
# it serially, so a plain `Ref` counter sees every fit and nothing else. The rule
# returned is the region box's own tensor Gauss — the cheapest thing that
# honours the contract, which is all a counting test needs.
function counting_cut_rule(fits::Ref{Int})
    return function (physical, region_box::Unfitted.AxisBox{D,T},
                     moment_order::NTuple{D,Int}) where {D,T}
        fits[] += 1
        rule = Unfitted._tensor_gauss_rule(moment_order .+ 1, T)
        jacobian = Unfitted.volume(region_box) / T(2^D)
        points = [Unfitted.reference_to_physical(region_box, p) for p in rule.points]
        return points, rule.weights .* jacobian, zero(T), :custom
    end
end

@testset "derived models reuse the source's cut rules without evicting them" begin
    # Cache *identity* is the wrong pin: a fork that shares its source's cache
    # evicts the source's rules when it builds its own plan, which is the bug
    # this asserts the absence of. What is observable — and what the reuse
    # exists for — is that no derivation refits a rule the source already holds.
    fits = Ref(0)
    disc = physical_domain(x -> sqrt(x[1]^2 + x[2]^2) - 0.7; subcell_length_scale=0.25,
                           cut_quadrature=counting_cut_rule(fits))
    V = ladder(box((-1.0, -1.0), (1.0, 1.0)); cells=8, order=2, depth=1, splits=2, physical=disc)
    model = prepare(mass(adapt(V, 2 => trues(16, 16))))
    @test fits[] > 0

    fits[] = 0
    target = adapted(model, 2 => trues(16, 16))
    @test fits[] == 0
    # Its own cache, holding everything the source's does: both models are live
    # from here, so neither may reduce the other's.
    @test only(target.moment_fit_caches) !== only(model.moment_fit_caches)
    @test length(only(target.moment_fit_caches)) == length(only(model.moment_fit_caches))

    # The alternation an hp loop runs. `estimate` builds an order-elevated twin,
    # which needs a *second* moment order at the very same region boxes; the step
    # that follows needs the first one back. Evicting a cache to exactly one
    # plan's keys makes the two orders evict each other, so both refit every
    # cycle — the cost the reuse was introduced to avoid.
    u = solve!(model)
    fits[] = 0
    estimate(model, u)
    @test fits[] > 0                                   # the elevated order is new
    fits[] = 0
    adapted(model, 2 => trues(16, 16))
    @test fits[] == 0                                  # ... and the step after it is warm
    fits[] = 0
    estimate(model, u)
    @test fits[] == 0                                  # ... as is the next estimate
end

@testset "moved reuses the source's cut rules" begin
    # `moved` is the third non-mutating derivation and goes through the same
    # builder as `adapted` and `elevated`, so it reuses rules on the same terms:
    # the boxes the move leaves bit-identical cost nothing, and the source keeps
    # every rule it had. Ω = {x ≤ 0.55} cuts the x ∈ [0.5, 0.75] column of the
    # base, which the one-cell overlay never touches at either position.
    fits = Ref(0)
    half = physical_domain(x -> x[1] - 0.55; lipschitz=1.0, subcell_length_scale=1.0e-3,
                           max_depth=3, cut_quadrature=counting_cut_rule(fits))
    V = overlay(space(LADDER_OMEGA; cells=(4, 4), order=1, physical=half),
                box((0.0, 0.0), (0.25, 0.25)); cells=(1, 1), order=1)
    model = prepare(mass(V; coefficient=1.0))
    before = length(only(model.moment_fit_caches))
    @test before > 0

    fits[] = 0
    target = moved(model; level=2, to=box((0.25, 0.0), (0.5, 0.25)))
    @test fits[] == 0
    @test only(target.moment_fit_caches) !== only(model.moment_fit_caches)
    @test length(only(model.moment_fit_caches)) == before
end

@testset "ladder composes with an immersed domain" begin
    disc = physical_domain(x -> sqrt(x[1]^2 + x[2]^2) - 0.7; subcell_length_scale=0.25)
    V = adapt(ladder(box((-1.0, -1.0), (1.0, 1.0)); cells=8, order=2, depth=2, splits=2,
                     physical=disc), 3 => trues(32, 32))
    @test is_nested(V)
    model = prepare(mass(V))
    report = diagnostics(model, solve!(model))
    @test report.fit_failure_count == 0
    @test report.symmetry_residual == 0
    @test all(l -> l.nested, report.levels)
end

@testset "ladder composes with B-splines" begin
    # The B-spline family reaches covered-mode pruning through its own
    # `_coverage_constraints`, which dedups rather than sheds bubbles; a nested
    # stack is non-singular only because that dedup fires. `lossless` compares
    # against the unreduced twin, which for this family is rank-deficient by
    # exactly the deduped modes, so it is the right check here too.
    V = ladder(LADDER_OMEGA; cells=8, order=3, depth=2, splits=2, basis=bspline())
    W = adapt(V, 2 => trues(16, 16))
    @test is_nested(W)
    A = gram(W)
    @test rank(A) == size(A, 1)
    @test lossless(W)
    @test count(active_cells(W; level=2)) == 16 * 16
end

@testset "ladder in 1D and 3D" begin
    V1 = ladder(box((0.0,), (1.0,)); cells=8, order=2, depth=3, splits=2)
    m = falses(64)
    m[20:30] .= true
    W1 = adapt(V1, 4 => m)
    @test is_nested(W1) && count(active_cells(W1; level=4)) == 11
    @test count(overlapping_cells(V1, m; from=4, to=1)) == 2
    @test rank(gram(W1)) == size(gram(W1), 1)

    V3 = ladder(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=4, order=2, depth=2, splits=(2, 1, 2))
    @test [l.mesh.cells for l in V3.levels] == [(4, 4, 4), (8, 4, 8), (16, 4, 16)]
    m3 = falses(16, 4, 16)
    m3[7:10, 2:3, 7:10] .= true
    W3 = adapt(V3, 3 => m3)
    @test is_nested(W3)
    @test lossless(W3)
end
