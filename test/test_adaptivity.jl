using BasicBSpline
using Unfitted
using Unfitted: _dorfler

# The adaptivity verbs. The assertion that actually validates an error indicator
# is its EFFECTIVITY — η against the error it estimates — and its drift over a
# refinement sequence, so that is what this file leads with. A test that only
# checked η > 0 would pass for an indicator that points in the wrong direction,
# which is how a gradient-recovery indicator survived long enough to be measured
# at 52.57× drift and rejected.

const _AD_Ω = box((0.0, 0.0), (1.0, 1.0))
_ad_exact(x) = sinpi(x[1]) * sinpi(x[2])
_ad_grad(x) = (pi * cospi(x[1]) * sinpi(x[2]), pi * sinpi(x[1]) * cospi(x[2]))
function _ad_problem(V)
    poisson(V; source=x -> 2pi^2 * _ad_exact(x),
            dirichlet=[dirichlet(_ad_exact; on=boundary(:all))])
end

# H¹ seminorm of the error on a lattice independent of the space being measured:
# the model's own quadrature is sized from that space, so using it to compare two
# spaces flatters the finer one — measured at 22.5% on an adapted stack.
function _ad_energy_error(sol, model)
    acc = 0.0
    n, q = 24, 3
    ξ = (-0.7745966692414834, 0.0, 0.7745966692414834)
    w = (0.5555555555555556, 0.8888888888888888, 0.5555555555555556)
    h = 1 / n
    for i in 1:n, j in 1:n, a in 1:q, b in 1:q
        x = ((i - 1) * h + 0.5h * (1 + ξ[a]), (j - 1) * h + 0.5h * (1 + ξ[b]))
        g = field_gradient(sol, model, x)
        acc += sum(abs2, g .- _ad_grad(x)) * (0.5h * w[a]) * (0.5h * w[b])
    end
    return sqrt(acc)
end

@testset "estimate: effectivity and its drift over a refinement sequence" begin
    effectivities = Float64[]
    for n in (4, 8, 16)
        V = space(_AD_Ω; cells=n, order=2)
        model = prepare(_ad_problem(V))
        u = solve!(model)
        est = estimate(model, u)
        truth = _ad_energy_error(u, model)
        push!(effectivities, est.total / truth)
        # The indicator must fall with the error it estimates, not merely exist.
        @test est.total > 0
        @test 0.3 < est.total / truth < 3.0
        # Galerkin orthogonality makes the residual vanish on V in exact
        # arithmetic; a consistency that stops being small says the enrichment is
        # no longer the dominant missing content.
        @test est.consistency < 1.0e-2
        @test est.reference > 0
    end
    # Drift is the property that separates a usable indicator from a plausible
    # one: over this 16× range in unknowns it must stay bounded.
    @test maximum(effectivities) / minimum(effectivities) < 1.5
end

@testset "estimate: what it refuses" begin
    V = space(_AD_Ω; cells=4, order=2)
    model = prepare(_ad_problem(V))
    # Not assembled: there is no solution to inject and no reference energy.
    @test_throws ArgumentError estimate(model,
                                        Solution(zeros(active_unknowns(model)), model.version,
                                                 Unfitted.SolverDiagnostics(:manual, 0.0, true)))
    u = solve!(model)
    @test_throws ArgumentError estimate(model, u; enrichment=0)
end

@testset "refine: the h-step and the p-step, on the right levels" begin
    V = ladder(_AD_Ω; cells=4, order=2, depth=2)
    cell = CartesianIndex(2, 2)

    # The h-step activates the level BELOW, over the cells covering the marked
    # one, and hands them their parent's order UNCHANGED. Pure h: the resolution
    # moves and the order does not. Coupling the two is what made order a
    # function of depth and left the base mesh permanently at order 1.
    W = refine(V; h=[(1, cell)])
    @test cell_orders(W; level=1) == cell_orders(V; level=1)
    parent_order = cell_orders(V; level=1)[cell]
    children = overlapping_cells(V, [cell]; from=1, to=2)
    child_orders = cell_orders(W; level=2)
    for c in CartesianIndices(children)
        children[c] || continue
        @test child_orders[c] == parent_order
        @test active_cells(W; level=2)[c]
    end
    @test count(active_cells(W; level=2)) > count(active_cells(V; level=2))

    # The p-step raises the marked cell's OWN order and activates nothing.
    P = refine(V; p=[(1, cell)])
    @test cell_orders(P; level=1)[cell] == parent_order .+ 1
    @test active_cells(P; level=2) == active_cells(V; level=2)
    @test all(cell_orders(P; level=1)[c] == cell_orders(V; level=1)[c]
              for c in cell_indices(V; level=1) if c != cell)

    # Nothing already live is lost: `adapt` SETS a mask, so an application that
    # forgets to keep what is live silently coarsens the rest of the domain.
    W2 = refine(W; h=[(1, CartesianIndex(3, 3))])
    @test count(active_cells(W2; level=2)) > count(active_cells(W; level=2))
    @test all(active_cells(W2; level=2)[c]
              for c in CartesianIndices(active_cells(W; level=2)) if active_cells(W; level=2)[c])

    # On the finest level there is nothing left to activate, so a cell sent to h
    # takes the order step instead.
    fine = length(V.levels)
    X = refine(V; h=[(fine, CartesianIndex(1, 1))])
    @test cell_orders(X; level=fine)[CartesianIndex(1, 1)] ==
          cell_orders(V; level=fine)[CartesianIndex(1, 1)] .+ 1
    @test active_cells(X; level=fine) == active_cells(V; level=fine)

    # `pmax` caps the order so a loop cannot run away.
    Y = refine(V; p=[(1, cell)], pmax=2)
    @test cell_orders(Y; level=1)[cell] == (2, 2)

    @test_throws ArgumentError refine(V; h=[(99, CartesianIndex(1, 1))])
    @test_throws ArgumentError refine(V; p=[(99, CartesianIndex(1, 1))])
end

@testset "the h-versus-p guards: what never reaches the decision" begin
    V = ladder(_AD_Ω; cells=4, order=2, depth=2)
    nlevels = length(V.levels)
    marked = [(1, c) for c in cell_indices(V; level=1)]
    flat(V) = Unfitted.ErrorEstimate{Float64,2}([ones(l.mesh.cells) for l in V.levels], 1.0, 1.0,
                                                0.0)
    est = flat(V)

    # With no history, every marked leaf takes p.
    h, p = Unfitted._partition(V, est, marked, 8)
    @test isempty(h) && length(p) == length(marked)

    # At `pmax` there is no order left, so the same cells take h. Routing them to
    # p is a step that changes nothing, and a loop made only of those does not
    # terminate — measured, a 20-cycle run froze at 494 unknowns with its base
    # mesh pinned at order 8 and its last three cycles identical.
    h2, p2 = Unfitted._partition(V, est, marked, 2)      # the space is order 2
    @test isempty(p2) && length(h2) == length(marked)

    # A cell a finer level has taken over takes h however much order it has left:
    # leaf semantics shed a covered cell's bubbles, so the increment would vanish.
    W = refine(V; h=[(1, CartesianIndex(2, 2))])
    h3, p3 = Unfitted._partition(W, flat(W), [(1, CartesianIndex(2, 2))], 8)
    @test h3 == [(1, CartesianIndex(2, 2))] && isempty(p3)

    # On the finest level there is nothing to activate, so the cell takes p and
    # `refine` turns that into the order step.
    h4, p4 = Unfitted._partition(V, est, [(nlevels, CartesianIndex(1, 1))], 8)
    @test isempty(h4) && p4 == [(nlevels, CartesianIndex(1, 1))]

    @test_throws ArgumentError Unfitted._partition(V, est, [(99, CartesianIndex(1, 1))], 8)
end

@testset "the h-versus-p decision: a cell keeps p only if it earned it" begin
    # Melenk & Wohlmuth's prediction, which is the whole rule: a p-step by one
    # degree is predicted to multiply the indicator by γ_p = √0.4 ≈ 0.632. A cell
    # that beat that is behaving smoothly and takes p again; one that fell short
    # takes h. Both cells here are given the identical geometry and the identical
    # step, so the only thing separating the verdicts is what the step bought.
    V = ladder(_AD_Ω; cells=4, order=2, depth=2)
    good, bad = CartesianIndex(2, 2), CartesianIndex(3, 3)
    W = refine(V; p=[(1, good), (1, bad)])
    @test cell_orders(W; level=1)[good] == (3, 3)      # the p-step landed

    before = [zeros(l.mesh.cells) for l in V.levels]
    before[1][good] = before[1][bad] = 1.0
    est_prev = Unfitted.ErrorEstimate{Float64,2}(before, 1.0, 1.0, 0.0)

    after = [zeros(l.mesh.cells) for l in W.levels]
    after[1][good] = 0.5                               # beat γ_p — smooth
    after[1][bad] = 0.9                                # fell short — not smooth
    est_now = Unfitted.ErrorEstimate{Float64,2}(after, 1.0, 1.0, 0.0)

    h, p = Unfitted._partition(W, est_now, [(1, good), (1, bad)], 8; previous=(V, est_prev))
    @test p == [(1, good)]
    @test h == [(1, bad)]

    # Without the history there is nothing to have earned, so both take p. That
    # is the first-cycle convention, and it is why `previous` is not optional in
    # spirit even though it is in signature.
    h0, p0 = Unfitted._partition(W, est_now, [(1, good), (1, bad)], 8)
    @test isempty(h0) && length(p0) == 2
end

@testset "coarsen: the exact inverse of the h-step" begin
    # The round trip is the whole contract. A transient loop refines and releases
    # thousands of times, so anything that does not close exactly accumulates.
    V = ladder(_AD_Ω; cells=4, order=2, depth=2)
    cell = CartesianIndex(2, 2)
    same(a, b) = all(active_cells(a; level=k) == active_cells(b; level=k) &&
                         cell_orders(a; level=k) == cell_orders(b; level=k)
                     for k in 1:length(a.levels))

    W = refine(V; h=[(1, cell)])
    @test !same(W, V)
    @test same(coarsen(W; h=[(1, cell)]), V)

    # An order the children earned while live must NOT survive the release: the
    # next h-step onto that region would inherit it. Measured before this was
    # fixed, one h-step onto a released order-6 cover returned 169 unknowns where
    # a cold h-step gives 57.
    children = overlapping_cells(V, [cell]; from=1, to=2)
    raised = refine(W; p=[(2, c) for c in CartesianIndices(children) if children[c]])
    @test maximum(first, cell_orders(raised; level=2)) > 2
    @test same(coarsen(raised; h=[(1, cell)]), V)
    @test same(refine(coarsen(raised; h=[(1, cell)]); h=[(1, cell)]), W)
end

@testset "coarsen: the p-step, and what it refuses" begin
    V = ladder(_AD_Ω; cells=4, order=3, depth=2)
    cell = CartesianIndex(2, 2)

    lowered = coarsen(V; p=[(1, cell)])
    @test cell_orders(lowered; level=1)[cell] == (2, 2)
    @test all(cell_orders(lowered; level=1)[c] == (3, 3)
              for c in cell_indices(V; level=1) if c != cell)
    @test cell_orders(coarsen(V; p=[(1, cell)], pmin=3); level=1)[cell] == (3, 3)

    # Lowering a COVERED cell's order is invisible until the cover lifts, and no
    # diagnostic catches it — measured, 173.5× the energy error with the residual
    # norm and the estimator's consistency flag unchanged to the last digit. It is
    # refused rather than reported.
    covered = refine(V; h=[(1, cell)])
    @test_throws ArgumentError coarsen(covered; p=[(1, cell)])

    # The finest level covers nothing, so there is nothing to release there.
    @test_throws ArgumentError coarsen(V; h=[(length(V.levels), CartesianIndex(1, 1))])
    @test_throws ArgumentError coarsen(V; h=[(99, CartesianIndex(1, 1))])

    # Releasing a cover whose children are themselves covered leaves the deeper
    # level holding the region: a no-op, not a hole.
    deep = refine(refine(V; h=[(1, cell)]);
                  h=[(2, c)
                     for c in CartesianIndices(overlapping_cells(V, [cell]; from=1, to=2))
                     if overlapping_cells(V, [cell]; from=1, to=2)[c]])
    @test active_cells(coarsen(deep; h=[(1, cell)]); level=2) == active_cells(deep; level=2)
end

@testset "the prediction reads a released order as evidence" begin
    # A p-released cell used to fall through the equality branch and take p again
    # on sight, which is the oscillation the release was trying to avoid. Melenk
    # & Wohlmuth's coarsening row makes the exponent negative, so the cell is
    # predicted to get WORSE by the reciprocal of what the degree was worth.
    V = ladder(_AD_Ω; cells=4, order=3, depth=2)
    cell = CartesianIndex(2, 2)
    W = coarsen(V; p=[(1, cell)])

    before = [zeros(l.mesh.cells) for l in V.levels]
    before[1][cell] = 1.0
    est_prev = Unfitted.ErrorEstimate{Float64,2}(before, 1.0, 1.0, 0.0)
    caches = (Dict{Int,Any}(), Dict{Int,Any}())
    predicted = Unfitted._predicted(W, (V, est_prev), 1, cell, cell_orders(W; level=1)[cell],
                                    caches, Unfitted._GAMMA_P)
    @test predicted > 1.0                       # the error is expected to grow
    @test predicted ≈ 1.0 / Unfitted._GAMMA_P    # by exactly one degree's worth
end

@testset "refine: Dörfler marking is monotone in theta" begin
    V = ladder(_AD_Ω; cells=8, order=2, depth=1)
    model = prepare(_ad_problem(V))
    est = estimate(model, solve!(model))
    counts = [length(_dorfler(est, θ)) for θ in (0.1, 0.3, 0.5, 0.9, 1.0)]
    @test issorted(counts)
    @test counts[1] >= 1
    # θ = 1 takes every cell that carries any indicator at all.
    @test counts[end] == count(a -> a > 0, reduce(vcat, vec.(est.cells)))
    @test_throws ArgumentError refine(V, est; theta=0.0)
    @test_throws ArgumentError refine(V, est; theta=1.5)
end

@testset "Dörfler completes the group its cut lands in" begin
    # Cells related by a symmetry of the problem carry indicators equal only to
    # round-off — assembly and the solve are not associative, so they are never
    # bit-identical — and a strict minimal prefix hands the choice between them
    # to the last bits. Measured on the high-contrast disc, that split the four
    # equal-indicator cells at the diagonals two-and-two, and the loop refined
    # only the upper-left and lower-right quadrants for ten cycles while the
    # marked counts came out odd (11, 5, 7) on an orbit structure that cannot
    # produce an odd count.
    V = ladder(_AD_Ω; cells=4, order=2, depth=1)
    coarse, fine = V.levels[1].mesh.cells, V.levels[2].mesh.cells
    ind = zeros(coarse)
    tie = [CartesianIndex(1, 1), CartesianIndex(4, 1), CartesianIndex(1, 4), CartesianIndex(4, 4)]
    for (t, c) in pairs(tie)
        ind[c] = 1.0 + (t - 1) * 1.0e-15        # equal to round-off, not bit-equal
    end
    ind[CartesianIndex(2, 2)] = 0.1             # genuinely smaller
    est = Unfitted.ErrorEstimate{Float64,2}([ind, zeros(fine)], sqrt(sum(ind .^ 2)), 1.0, 0.0)

    # θ this small needs one cell to reach the bulk, so the minimal prefix cuts
    # inside the group of four; all four must come back, not the one the sort
    # happened to put first.
    marked = Unfitted._dorfler(est, 0.2)
    @test length(marked) == 4
    @test Set(c for (_, c) in marked) == Set(tie)

    # Completing a tie must not sweep in a cell that merely sorts next: 0.1 is a
    # different indicator, not the same one seen through round-off.
    @test !((1, CartesianIndex(2, 2)) in marked)
    @test length(Unfitted._dorfler(est, 1.0)) == 5
end

@testset "the loop reduces the error and the estimate" begin
    V = ladder(_AD_Ω; cells=4, order=2, depth=3)
    model = prepare(_ad_problem(V))
    u = solve!(model)
    first_est = estimate(model, u)
    first_err = _ad_energy_error(u, model)
    unknowns = [active_unknowns(model)]
    local est = first_est
    for _ in 1:4
        model = adapted(model, refine(model.prefold_space, est; theta=0.5))
        u = solve!(model)
        est = estimate(model, u)
        push!(unknowns, active_unknowns(model))
    end
    @test est.total < first_est.total
    @test _ad_energy_error(u, model) < first_err
    @test issorted(unknowns)                       # refinement only ever adds
    # And it beats uniform refinement at a comparable unknown count, which is the
    # only reason to run an adaptive loop at all.
    uniform = prepare(_ad_problem(space(_AD_Ω; cells=8, order=2)))
    @test active_unknowns(uniform) >= active_unknowns(model) ||
          _ad_energy_error(u, model) < _ad_energy_error(solve!(uniform), uniform)
end

@testset "adaptivity is dimension generic" begin
    Ω3 = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    exact3(x) = sinpi(x[1]) * sinpi(x[2]) * sinpi(x[3])
    problem3(W) = poisson(W; source=x -> 3pi^2 * exact3(x),
                          dirichlet=[dirichlet(exact3; on=boundary(:all))])
    V = ladder(Ω3; cells=3, order=2, depth=1)
    model = prepare(problem3(V))
    u = solve!(model)
    est = estimate(model, u)
    @test est.total > 0
    @test est.consistency < 1.0e-2
    W = refine(V, est; theta=0.5)
    @test active_unknowns(prepare(problem3(W))) > active_unknowns(model)
end

@testset "estimate refuses a family whose keys are not order stable" begin
    # Raising a B-spline's degree rewrites the knot vector, so injecting the
    # solution into the enriched space would copy coefficients onto different
    # functions. `Rewire` now catches that; `estimate` says so up front.
    V = space(_AD_Ω; cells=4, order=2, basis=bspline())
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    u = solve!(model)
    @test_throws ArgumentError estimate(model, u)
end

@testset "refine: the p-step actually survives" begin
    # A p-step is only ever handed a LEAF, and that is what makes it real: the
    # same increment on a covered cell is a no-op, because leaf semantics shed a
    # covered cell's bubble modes. Measured before that was understood, 356 cells
    # raised while covered kept 576 active modes between them and zero bubbles.
    V = ladder(_AD_Ω; cells=4, order=1, depth=2)
    W = refine(V; p=[(1, CartesianIndex(2, 2)), (1, CartesianIndex(3, 3))])
    model = prepare(_ad_problem(W))
    bubbles = 0
    for k in 1:length(W.levels), c in cell_indices(W; level=k)
        Unfitted.is_active(W.levels[k].mask, c) || continue
        ids = Unfitted.cell_basis_indices(W.levels[k], c)
        for (id, a) in zip(ids, Unfitted.active_cell_dofs(model.dofs, k, c))
            a == 0 && continue
            any(i -> i >= 2, Tuple(id)) && (bubbles += 1)
        end
    end
    @test bubbles > 0
end

@testset "the loop lets base cells gain order" begin
    # The defect this scheme exists to fix. With order coupled to depth a base
    # cell could only gain order by being h-refined, so a 20-cycle run left all
    # 36 base cells at order 1 — while deal.II's step-27, on the same problem,
    # ended with none of its cells there. On an analytic solution every marked
    # cell is smooth, so the base mesh must climb in p and must not be forced to
    # buy resolution it does not need.
    V = ladder(_AD_Ω; cells=4, order=1, depth=3)
    model = prepare(_ad_problem(V))
    for _ in 1:5
        u = solve!(model)
        model = adapted(model, refine(model.prefold_space, estimate(model, u); theta=0.5))
    end
    @test any(o -> o[1] > 1, cell_orders(model.prefold_space; level=1))
end
