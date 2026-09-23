using BasicBSpline
using StaticArrays
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
        # Dimension first, like every other parameterised type in the package.
        @test est isa Unfitted.ErrorEstimate{2,Float64}
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

@testset "estimate charges every component of a vector field" begin
    # The attribution used to read `active_cell_dofs` at its default component,
    # so on a vector-valued field it measured component 1 and nothing else. A
    # two-component problem whose datum sits on component 2 then came back with
    # `total = 0` and `consistency = 0` — a pair that reads as converged and
    # stops a Dörfler loop on its first cycle, against a `reference` that says
    # the solution is not zero at all.
    V = space(_AD_Ω; cells=4, order=2)
    homogeneous = [dirichlet(0.0; on=boundary(:all))]
    scalar = prepare(poisson(V; source=1.0, dirichlet=homogeneous))
    twin = estimate(scalar, solve!(scalar))
    @test twin.total > 0

    # `poisson` is diagonal in components, so a two-component field carrying the
    # scalar problem's source on component 2 alone IS the scalar problem, moved.
    u = field(:u, V; components=2)
    zero2 = [dirichlet(SVector(0.0, 0.0); on=boundary(:all))]
    second = prepare(poisson(u; source=SVector(0.0, 1.0), dirichlet=zero2))
    second_est = estimate(second, solve!(second))
    @test second_est.total ≈ twin.total
    @test second_est.reference ≈ twin.reference

    # Both components loaded is two independent copies of it, so the squared
    # indicators add and η grows by √2 — per cell as well as in the total.
    both = prepare(poisson(u; source=SVector(1.0, 1.0), dirichlet=zero2))
    both_est = estimate(both, solve!(both))
    @test both_est.total ≈ sqrt(2) * twin.total
    @test all(both_est.cells[k] ≈ sqrt(2) .* twin.cells[k] for k in 1:length(twin.cells))
end

@testset "estimate refuses a form it cannot measure" begin
    # The diagonal Bank–Weiser indicator reads A⁺_jj as the energy of complement
    # mode j, which is a coercive form's property. On an indefinite one the
    # diagonal can be negative, and dropping that mode from the sum returns a
    # smaller, entirely plausible η for a form the indicator has no claim on.
    # Helmholtz well above the first resonance is the clean case.
    V = space(_AD_Ω; cells=4, order=2)
    helmholtz = WeakForm(bilinear=(q, trial) -> TestChannels(-2000.0 * trial.value, trial.gradient),
                         linear=q -> 1.0, symmetric=true)
    model = prepare(Problem(V, helmholtz; dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    u = solve!(model)
    @test_throws ArgumentError estimate(model, u)

    # The same form at a wave number the mesh keeps coercive is estimated
    # normally, so the guard is about the operator and not about the syntax.
    mild = WeakForm(bilinear=(q, trial) -> TestChannels(-1.0 * trial.value, trial.gradient),
                    linear=q -> 1.0, symmetric=true)
    tame = prepare(Problem(V, mild; dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test estimate(tame, solve!(tame)).total > 0
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
    flat(V) = Unfitted.ErrorEstimate{2,Float64}([ones(l.mesh.cells) for l in V.levels], 1.0, 1.0,
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

    # A cell a finer level has taken over is never sent to p — leaf semantics
    # shed a covered cell's bubbles, so the increment would vanish — and it takes
    # h for as long as h still has something to activate.
    part = falses(V.levels[2].mesh.cells)
    part[CartesianIndex(3, 3)] = true                 # one of cell (2, 2)'s four children
    Wpart = adapt(V, 2 => part)
    h3, p3 = Unfitted._partition(Wpart, flat(Wpart), [(1, CartesianIndex(2, 2))], 8)
    @test h3 == [(1, CartesianIndex(2, 2))] && isempty(p3)

    # Once the cover is complete neither step is on offer: p is shed and h sets
    # no mask bit. The mark is dropped rather than spent on a step that changes
    # nothing — the children carry the error from here on — and `refine` says so
    # by returning its argument.
    W = refine(V; h=[(1, CartesianIndex(2, 2))])
    h5, p5 = Unfitted._partition(W, flat(W), [(1, CartesianIndex(2, 2))], 8)
    @test isempty(h5) && isempty(p5)
    @test refine(W; h=h5, p=p5) === W

    # On the finest level there is nothing to activate, so the cell takes p and
    # `refine` turns that into the order step.
    h4, p4 = Unfitted._partition(V, est, [(nlevels, CartesianIndex(1, 1))], 8)
    @test isempty(h4) && p4 == [(nlevels, CartesianIndex(1, 1))]

    # A finest-level cell that is ALSO at `pmax` has neither step, so it is
    # dropped instead of being parked in `p`, where it would make every cycle
    # rebuild an identical space for ever.
    h6, p6 = Unfitted._partition(V, est, [(nlevels, CartesianIndex(1, 1))], 2)
    @test isempty(h6) && isempty(p6)

    @test_throws ArgumentError Unfitted._partition(V, est, [(99, CartesianIndex(1, 1))], 8)
    @test_throws ArgumentError Unfitted._partition(V, est, [(1, CartesianIndex(9, 9))], 8)
end

@testset "the leaf test reaches every finer level, not only the one below" begin
    # `ladder` admits a stack that is live on level k+2 while k+1 is dormant, and
    # the dof layer's covering rule tests every finer level in turn. A leaf test
    # that looked one level down called such a cell a leaf and sent it to p — an
    # order step the dof walk then shed again, every cycle, for as long as the
    # skip lasted.
    V = ladder(_AD_Ω; cells=4, order=2, depth=2)
    cell = CartesianIndex(2, 2)
    flat(W) = Unfitted.ErrorEstimate{2,Float64}([ones(l.mesh.cells) for l in W.levels], 1.0, 1.0,
                                                0.0)
    grandchildren = overlapping_cells(V, [cell]; from=1, to=3)
    S = adapt(V, 3 => grandchildren)                   # level 3 live, level 2 empty
    @test !any(active_cells(S; level=2))
    @test !Unfitted._is_leaf(S, 1, cell)
    # Nor is there an h-step: every level-2 cell under this one is already held
    # from below, so activating it is dof-inert too.
    @test !Unfitted._has_h_step(S, 1, cell)
    h, p = Unfitted._partition(S, flat(S), [(1, cell)], 8)
    @test isempty(h) && isempty(p)
    @test_throws ArgumentError coarsen(S; p=[(1, cell)])

    # Inert is measured, not assumed: 97 active unknowns before either explicit
    # step and 97 after, which is what makes routing the mark anywhere a waste.
    before = active_unknowns(prepare(_ad_problem(S)))
    @test active_unknowns(prepare(_ad_problem(refine(S; p=[(1, cell)])))) == before
    @test active_unknowns(prepare(_ad_problem(refine(S; h=[(1, cell)])))) == before

    # Where the skip holds only part of the cell, both steps buy something: the
    # cell is not a leaf, h still has a dormant child to wake, and it takes h.
    corner = first(c for c in CartesianIndices(grandchildren) if grandchildren[c])
    part = falses(V.levels[3].mesh.cells)
    part[corner] = true
    P = adapt(V, 3 => part)
    @test !Unfitted._is_leaf(P, 1, cell) && Unfitted._has_h_step(P, 1, cell)
    hp, pp = Unfitted._partition(P, flat(P), [(1, cell)], 8)
    @test hp == [(1, cell)] && isempty(pp)
    partial = active_unknowns(prepare(_ad_problem(P)))
    @test active_unknowns(prepare(_ad_problem(refine(P; p=[(1, cell)])))) > partial
    @test active_unknowns(prepare(_ad_problem(refine(P; h=[(1, cell)])))) > partial

    # `coarsen`'s release rule reads the stack the same way. Here level 4 holds
    # one level-2 child while level 3 is empty, so releasing level 1's cover must
    # leave that child alone and release only its siblings.
    Vd = ladder(_AD_Ω; cells=4, order=2, depth=3)
    Wd = refine(Vd; h=[(1, cell)])
    child = CartesianIndex(3, 3)
    held = adapt(Wd, 4 => overlapping_cells(Vd, [child]; from=2, to=4))
    @test !any(active_cells(held; level=3))
    after = active_cells(coarsen(held; h=[(1, cell)]); level=2)
    @test after[child]                                 # still held, two levels down
    @test !after[CartesianIndex(4, 4)]                 # its siblings are released
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
    est_prev = Unfitted.ErrorEstimate{2,Float64}(before, 1.0, 1.0, 0.0)

    after = [zeros(l.mesh.cells) for l in W.levels]
    after[1][good] = 0.5                               # beat γ_p — smooth
    after[1][bad] = 0.9                                # fell short — not smooth
    est_now = Unfitted.ErrorEstimate{2,Float64}(after, 1.0, 1.0, 0.0)

    h, p = Unfitted._partition(W, est_now, [(1, good), (1, bad)], 8; previous=(V, est_prev))
    @test p == [(1, good)]
    @test h == [(1, bad)]

    # Without the history there is nothing to have earned, so both take p. That
    # is the first-cycle convention, and it is why `previous` is not optional in
    # spirit even though it is in signature.
    h0, p0 = Unfitted._partition(W, est_now, [(1, good), (1, bad)], 8)
    @test isempty(h0) && length(p0) == 2
end

@testset "the h-versus-p decision: what an h-step's child was promised" begin
    # The other half of the rule, and the half with the geometry in it. A child
    # of an h-step inherits an equal share of its parent's indicator, reduced by
    # the h-rate of the energy norm at the child's own order. Depth 2 is
    # load-bearing: on a depth-1 ladder the children sit on the finest level and
    # are routed to p before the prediction is ever consulted.
    V = ladder(_AD_Ω; cells=4, order=2, depth=2)
    cell = CartesianIndex(2, 2)
    W = refine(V; h=[(1, cell)])
    kids = [c
            for c in CartesianIndices(overlapping_cells(V, [cell]; from=1, to=2))
            if overlapping_cells(V, [cell]; from=1, to=2)[c]]
    @test length(kids) == 4

    before = [zeros(l.mesh.cells) for l in V.levels]
    before[1][cell] = 1.0
    est_prev = Unfitted.ErrorEstimate{2,Float64}(before, 1.0, 1.0, 0.0)
    pred = Unfitted._predicted(W, (V, est_prev), 2, kids[1])
    # η · γ_h · ρ^{p/D} / √n with ρ = 1/2^D the volume ratio of a bisected child,
    # p = 2 the inherited order and n = 2^D children: 2 · 0.25 / 2 = 0.25 in 2D.
    @test pred ≈ Unfitted._GAMMA_H * 0.5^2 / sqrt(2^2)
    @test pred ≈ 0.25

    after = [zeros(l.mesh.cells) for l in W.levels]
    after[2][kids[1]] = 1.2 * pred                    # fell short — not smooth
    after[2][kids[2]] = 0.4 * pred                    # beat it — smooth
    est_now = Unfitted.ErrorEstimate{2,Float64}(after, 1.0, 1.0, 0.0)
    h, p = Unfitted._partition(W, est_now, [(2, kids[1]), (2, kids[2])], 8; previous=(V, est_prev))
    @test h == [(2, kids[1])]
    @test p == [(2, kids[2])]
end

@testset "the prediction off the ladder: sub-box overlays and anisotropic splits" begin
    # The child count and the h-rate are both read from the geometry, because
    # neither is implied by the level's cell count off a ladder.

    # A sub-box overlay at the SAME cell count as the base: the ratio of the two
    # levels' cell counts is 1 where each parent has 2^D children, which made
    # every child look √(2^D) better than it was and take p on sight.
    inner = box((0.25, 0.25), (0.75, 0.75))
    U = overlay(overlay(space(_AD_Ω; cells=4, order=2), inner; cells=4, active=falses(4, 4)), inner;
                cells=8, active=falses(8, 8))
    cell = CartesianIndex(2, 2)                       # [0.25, 0.5]², inside the overlay
    UW = refine(U; h=[(1, cell)])
    kids = [c
            for c in CartesianIndices(overlapping_cells(U, [cell]; from=1, to=2))
            if overlapping_cells(U, [cell]; from=1, to=2)[c]]
    @test length(kids) == 4                           # four, though 4×4 over 4×4
    before = [zeros(l.mesh.cells) for l in U.levels]
    before[1][cell] = 1.0
    est_prev = Unfitted.ErrorEstimate{2,Float64}(before, 1.0, 1.0, 0.0)
    @test Unfitted._predicted(UW, (U, est_prev), 2, kids[1]) ≈ 0.25

    # An anisotropic split: level 2 halves axis 1 and leaves axis 2 alone, so the
    # parent has two children and the volume ratio is ½, not ¼. Judging it by the
    # bisection rate 0.5^p would call a merely anisotropic child non-smooth.
    S = ladder(_AD_Ω; cells=4, order=2, depth=2, splits=[(2, 1), (1, 2)])
    @test [l.mesh.cells for l in S.levels] == [(4, 4), (8, 4), (8, 8)]
    SW = refine(S; h=[(1, cell)])
    skids = [c
             for c in CartesianIndices(overlapping_cells(S, [cell]; from=1, to=2))
             if overlapping_cells(S, [cell]; from=1, to=2)[c]]
    @test length(skids) == 2
    sbefore = [zeros(l.mesh.cells) for l in S.levels]
    sbefore[1][cell] = 1.0
    sprev = Unfitted.ErrorEstimate{2,Float64}(sbefore, 1.0, 1.0, 0.0)
    # γ_h · (½)^{p/D} / √2 = 2 · 2^{−1} / √2 = 2^{−1/2}, twice the 0.5^p answer.
    @test Unfitted._predicted(SW, (S, sprev), 2, skids[1]) ≈ sqrt(0.5)
end

@testset "refine: a cell the level below does not reach" begin
    # A sub-box overlay reaches only part of the base level, so an h-mark outside
    # it has nothing to activate — exactly the position of a cell on the finest
    # level, and handled by the same rule rather than by silently doing nothing.
    inner = box((0.25, 0.25), (0.75, 0.75))
    U = overlay(space(_AD_Ω; cells=8, order=2), inner; cells=8, active=falses(8, 8))
    corner = CartesianIndex(8, 8)                     # [0.875, 1]², outside the overlay
    @test !any(overlapping_cells(U, [corner]; from=1, to=2))
    @test !Unfitted._has_h_step(U, 1, corner)

    flat = Unfitted.ErrorEstimate{2,Float64}([ones(l.mesh.cells) for l in U.levels], 1.0, 1.0, 0.0)
    h, p = Unfitted._partition(U, flat, [(1, corner)], 8)
    @test isempty(h) && p == [(1, corner)]
    @test cell_orders(refine(U; h=[(1, corner)]); level=1)[corner] == (3, 3)

    # And at `pmax` neither step is left, so the mark is dropped instead of being
    # re-issued every cycle against a cover that does not exist.
    h2, p2 = Unfitted._partition(U, flat, [(1, corner)], 2)
    @test isempty(h2) && isempty(p2)
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
    est_prev = Unfitted.ErrorEstimate{2,Float64}(before, 1.0, 1.0, 0.0)
    predicted = Unfitted._predicted(W, (V, est_prev), 1, cell)
    @test predicted > 1.0                       # the error is expected to grow
    @test predicted ≈ 1.0 / Unfitted._GAMMA_P    # by exactly one degree's worth
end

@testset "the prediction reads a released cover as evidence" begin
    # The h twin of the row above. An h-release leaves the parent active at an
    # unchanged order, so without a row of its own it falls through to "no
    # evidence" and buys p on sight — the same oscillation the p-release row was
    # added to stop. The row is the exact inverse of the h-refinement row: a
    # cover whose children met their prediction predicts the parent's own
    # indicator straight back.
    V = ladder(_AD_Ω; cells=4, order=2, depth=2)
    cell = CartesianIndex(2, 2)
    W = refine(V; h=[(1, cell)])
    R = coarsen(W; h=[(1, cell)])
    @test !any(active_cells(R; level=2))

    children = overlapping_cells(V, [cell]; from=1, to=2)
    before = [zeros(l.mesh.cells) for l in W.levels]
    before[2][children] .= 1.0                  # η = 1 on each of the four children
    est_prev = Unfitted.ErrorEstimate{2,Float64}(before, 2.0, 1.0, 0.0)
    predicted = Unfitted._predicted(R, (W, est_prev), 1, cell)
    # √(Σ η²) / (γ_h ρ^{p/D}) = 2 / (2 · 0.25) = 4, and the parent's own previous
    # indicator is not consulted: the evidence is the children's.
    @test predicted ≈ 4.0
end

@testset "refine and coarsen: pmax and pmin bound, they do not move" begin
    # A cap below a cell's order is "no more p here", not "lower it": `refine`
    # was asked to enrich the cell, and lowering it would be the opposite. The
    # floor reads the same way. Both used to move the order to the bound, which
    # the tests could not see because they set the bound at the current order.
    V = ladder(_AD_Ω; cells=4, order=4, depth=1)
    cell = CartesianIndex(2, 2)
    same(a, b) = all(active_cells(a; level=k) == active_cells(b; level=k) &&
                         cell_orders(a; level=k) == cell_orders(b; level=k)
                     for k in 1:length(a.levels))

    @test cell_orders(refine(V; p=[(1, cell)], pmax=2); level=1)[cell] == (4, 4)
    @test refine(V; p=[(1, cell)], pmax=2) === V

    # The children of an h-step take the parent's order UNCAPPED, because
    # `coarsen` restores exactly that value and a cap here would break the round
    # trip on any stack whose order already exceeds `pmax`.
    H = refine(V; h=[(1, cell)], pmax=2)
    children = overlapping_cells(V, [cell]; from=1, to=2)
    @test all(cell_orders(H; level=2)[c] == (4, 4)
              for c in CartesianIndices(children) if children[c])
    @test same(coarsen(H; h=[(1, cell)]), V)

    @test cell_orders(coarsen(V; p=[(1, cell)], pmin=6); level=1)[cell] == (4, 4)
    @test coarsen(V; p=[(1, cell)], pmin=6) === V

    # Per axis, not per cell: an axis below the bound still moves.
    A = elevate(V, 1 => [cell => (2, 9)])
    @test cell_orders(refine(A; p=[(1, cell)], pmax=8); level=1)[cell] == (3, 9)
    @test cell_orders(coarsen(A; p=[(1, cell)], pmin=3); level=1)[cell] == (2, 8)
end

@testset "refine and coarsen return their argument when nothing moved" begin
    # Identity is the exhaustion signal. Without it a saturated cycle returns a
    # fresh but identical `Space`, the caller rebuilds the model, and the loop
    # runs to its step limit with nothing changing.
    V = ladder(_AD_Ω; cells=4, order=2, depth=1)
    cell = CartesianIndex(2, 2)
    zero_est = Unfitted.ErrorEstimate{2,Float64}([zeros(l.mesh.cells) for l in V.levels], 0.0, 1.0,
                                                 0.0)
    @test isempty(_dorfler(zero_est, 0.5))
    @test refine(V, zero_est) === V
    @test refine(V; h=(), p=()) === V
    @test coarsen(V; h=(), p=()) === V

    # A p-step at `pmax`, an h-step on the finest level at `pmax`, and a release
    # of a cover that was never there all change nothing.
    @test refine(V; p=[(1, cell)], pmax=2) === V
    @test refine(V; h=[(2, CartesianIndex(3, 3))], pmax=2) === V
    @test coarsen(V; h=[(1, cell)]) === V
    @test coarsen(V; p=[(1, cell)], pmin=2) === V

    # And anything that does move returns a new space, so the signal is exact.
    @test refine(V; p=[(1, cell)]) !== V
    @test refine(V; h=[(1, cell)]) !== V
end

@testset "refine and coarsen: what they refuse" begin
    V = ladder(_AD_Ω; cells=4, order=4, depth=1)
    cell = CartesianIndex(2, 2)
    other = ladder(_AD_Ω; cells=8, order=2, depth=1)
    flat(W) = Unfitted.ErrorEstimate{2,Float64}([ones(l.mesh.cells) for l in W.levels], 1.0, 1.0,
                                                0.0)

    # A cell index out of range is the caller's mistake, named as such, rather
    # than a `BoundsError` from inside an array they never handed over.
    @test_throws ArgumentError refine(V; p=[(1, CartesianIndex(9, 9))])
    @test_throws ArgumentError refine(V; h=[(1, CartesianIndex(9, 9))])
    @test_throws ArgumentError coarsen(V; p=[(1, CartesianIndex(9, 9))])
    @test_throws ArgumentError coarsen(V; h=[(1, CartesianIndex(9, 9))])
    @test_throws ArgumentError refine(V; p=[(1, cell)], pmax=0)
    @test_throws ArgumentError coarsen(V; p=[(1, cell)], pmin=0)

    # An estimate taken on another space is indexed by that space's cells. Where
    # the shapes disagree it used to surface as a `BoundsError` from inside the
    # prediction; where they happen to agree it would silently score the wrong
    # cell, so both arguments are checked against the space they are read with.
    @test_throws DimensionMismatch refine(V, flat(other))
    @test_throws DimensionMismatch Unfitted._partition(V, flat(V), [(1, cell)], 8;
                                                       previous=(other, flat(other)))
    truncated = Unfitted.ErrorEstimate{2,Float64}([ones(4, 4)], 1.0, 1.0, 0.0)
    @test_throws DimensionMismatch refine(V, truncated)
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
    est = Unfitted.ErrorEstimate{2,Float64}([ind, zeros(fine)], sqrt(sum(ind .^ 2)), 1.0, 0.0)

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

@testset "adaptivity is dimension generic, with the history threaded" begin
    # One cycle per dimension through the PUBLIC keyword path, `refine(V, est;
    # previous = …)`, which nothing else in this file exercises: the synthetic
    # tests above call `_partition` directly, and the two end-to-end loops omit
    # `previous` and so only ever see the first-cycle "p everywhere" convention.
    #
    # The manufactured solution must not be a polynomial the space reproduces: a
    # constant source at order 2 in 1D has an exact quadratic solution, and the
    # indicator comes back at 1e-17 with nothing to mark. `h` is deliberately not
    # asserted non-empty — on a smooth sine every cell is entitled to p, and on
    # this fixture none of them fell short.
    for D in (1, 2, 3)
        Ω = box(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D))
        exact(x) = prod(sinpi, x)
        problem(W) = poisson(W; source=x -> D * pi^2 * exact(x),
                             dirichlet=[dirichlet(exact; on=boundary(:all))])
        V = ladder(Ω; cells=(D == 3 ? 3 : 4), order=2, depth=2)
        model = prepare(problem(V))
        est = estimate(model, solve!(model))
        @test est.total > 0
        @test est.consistency < 1.0e-2

        marked = _dorfler(est, 0.5)
        h, p = Unfitted._partition(V, est, marked, 8)
        @test length(h) + length(p) == length(marked)    # a fresh ladder drops nothing

        W = refine(V, est; theta=0.5)
        model2 = prepare(problem(W))
        est2 = estimate(model2, solve!(model2))
        @test active_unknowns(model2) > active_unknowns(model)
        @test est2.total < est.total

        X = refine(W, est2; theta=0.5, previous=(V, est))
        model3 = prepare(problem(X))
        est3 = estimate(model3, solve!(model3))
        @test active_unknowns(model3) > active_unknowns(model2)
        @test est3.total < est2.total

        # And the explicit verbs close on themselves in every dimension.
        cell = first(cell_indices(V; level=1))
        same(a, b) = all(active_cells(a; level=k) == active_cells(b; level=k) &&
                             cell_orders(a; level=k) == cell_orders(b; level=k)
                         for k in 1:length(a.levels))
        @test same(coarsen(refine(V; h=[(1, cell)]); h=[(1, cell)]), V)
    end
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
