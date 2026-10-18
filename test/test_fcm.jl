using StaticArrays
using LinearAlgebra

# Every built-in compression, for the properties the pipeline promises whichever one a domain carries. The NNLS target
# is tightened so that its retry, and not its acceptance band, decides the residuals asserted below.
const _FCM_COMPRESSIONS = (Caratheodory(), NonnegativeFit(; target_residual=1.0e-10))

@testset "FCM — MomentBasis is the tensor Legendre basis of its box" begin
    # On the box [0, 2] × [1, 3], ξ = (x₁ − 1, x₂ − 2), and up to order (2, 1) the basis is L_a(ξ₁) L_b(ξ₂) with L₀ = 1,
    # L₁ = ξ, L₂ = (3ξ² − 1) / 2. The functions are enumerated column-major over α, the constant first.
    region = box((0.0, 1.0), (2.0, 3.0))
    basis = MomentBasis(region, (2, 1))
    @test length(basis) == 6
    L = (ξ -> 1.0, ξ -> ξ, ξ -> (3ξ^2 - 1) / 2)
    points = [SVector(0.3, 2.9), SVector(1.7, 1.2), SVector(2.0, 1.0)]
    α = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
    expected = [L[a+1](x[1] - 1) * L[b+1](x[2] - 2) for (a, b) in α, x in points]
    @test basis(points) ≈ expected rtol = 1.0e-14

    # The moments of a rule are A w, without forming A — and those of a tensor Gauss rule are the exact integrals:
    # the box's area for the constant, zero for every other function, by orthogonality.
    weights = [0.25, 1.5, 0.125]
    @test basis(points, weights) ≈ basis(points) * weights rtol = 1.0e-14
    x, w = Unfitted.gausslegendre(3)
    gauss = [SVector(1 + x[i], 2 + x[j]) for i in 1:3 for j in 1:3]
    @test basis(gauss, [w[i] * w[j] for i in 1:3 for j in 1:3]) ≈ [4.0, 0, 0, 0, 0, 0] atol = 1.0e-14

    # Float32 stays Float32.
    basis32 = MomentBasis(box((0.0f0,), (1.0f0,)), (3,))
    @test eltype(basis32([SVector(0.5f0)])) === Float32
end

@testset "FCM — NNLS wrapper" begin
    # Small well-posed NNLS where the unconstrained solution is non-negative.
    A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    b = A * [0.5, 0.25]
    x, r = Unfitted.nnls(A, b)
    @test x ≈ [0.5, 0.25] atol = 1e-12
    @test r < 1e-12
end

@testset "FCM — a compressed rule reproduces the cut rule's moments (1D)" begin
    # The compression's target is the cut rule's own moments, here machine-exact because the cut is linear. Reference
    # moments come from an independent kernel rule at a higher Gauss count.
    region = box((0.0,), (1.0,))
    moment_order = (4,)
    ref = Unfitted.implicit_volume_quadrature(x -> x[1] - 0.7, region; gauss_points=8)
    moments = MomentBasis(region, moment_order)(ref...)
    @test moments[1] ≈ 0.7 atol = 1e-14
    for compression in _FCM_COMPRESSIONS
        p = physical_domain(x -> x[1] - 0.7; lipschitz=1.0, subcell_length_scale=1.0 / 2^8, max_depth=8, compression)
        points, weights, residual, status = Unfitted.moment_fit_rule(p, region, moment_order)
        @test status === :fitted
        @test residual < 1e-10
        @test MomentBasis(region, moment_order)(points, weights) ≈ moments atol = 1e-10
    end
end

@testset "FCM — a compressed rule is non-negative and carries at most nbasis points" begin
    # A disk, a tilted line and a polytope corner, at moment order 3: every compression returns non-negative weights at
    # no more than nbasis = ∏(order + 1) points, reproducing the moments to machine precision, and the measure it
    # integrates is the exact cut area wherever the boundary is linear.
    region = box((0.0, 0.0), (1.0, 1.0))
    order = (3, 3)
    # The disk's area is curved and left unchecked; the tilted line's is ∫₀^0.5 (1 − 2y) dy, the corner's 0.7 · 0.6.
    cases = ((x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3, nothing), (x -> x[1] + 2 * x[2] - 1.0, 0.25),
             (intersect(leaf(x -> x[1] - 0.7), leaf(x -> x[2] - 0.6)), 0.7 * 0.6))
    for (geometry, area) in cases, compression in _FCM_COMPRESSIONS
        lipschitz = geometry isa Unfitted.LevelSet ? Inf : 3.0
        p = physical_domain(geometry; lipschitz, subcell_length_scale=1.0 / 2^5, max_depth=5, compression)
        pts, ws, res, status = Unfitted.moment_fit_rule(p, region, order)
        @test status === :fitted
        @test res < 1.0e-9
        @test all(>(0), ws)
        @test length(pts) <= prod(order .+ 1)
        area === nothing || @test sum(ws) ≈ area atol = 1e-12
    end
end

@testset "FCM — a full cell reproduces tensor Gauss" begin
    # Ω = everything (φ ≡ −1): the kernel returns the full tensor rule, so the compressed moments match plain Gauss and
    # the weights sum to the cell volume.
    region = box((0.0, 0.0), (1.0, 1.0))
    for compression in _FCM_COMPRESSIONS
        p = physical_domain(x -> -1.0; lipschitz=1.0, subcell_length_scale=1.0, compression)
        pts, ws, res = Unfitted.moment_fit_rule(p, region, (3, 3))
        @test all(w -> w >= 0, ws)
        @test res < 1e-10
        @test sum(ws) ≈ 1.0 atol = 1e-10
    end
end

@testset "FCM — a kinked single level set still compresses to a valid rule" begin
    # A `max` of two lines is not smooth, so the kernel force-reduces on the kink; the cut rule is still a correct
    # non-negative rule, and its compression lands inside the failure threshold.
    region = box((0.0, 0.0), (1.0, 1.0))
    for compression in _FCM_COMPRESSIONS
        p = physical_domain(x -> max(x[1] - 0.7, x[2] - 0.6); lipschitz=2.0, subcell_length_scale=1.0 / 2^5,
                            max_depth=5, compression)
        _, ws, res, status = Unfitted.moment_fit_rule(p, region, (3, 3))
        @test status === :fitted
        @test all(w -> w >= 0, ws)
        @test res < Unfitted._FIT_FAILURE_RESIDUAL * sum(ws)   # within the threshold relative to ‖m‖ ≥ m₀ = Σw
    end
end

@testset "FCM — tilted/corner cuts: the default Gauss order gives exact moments" begin
    # A corner-beveling linear cut compounds the per-fiber polynomial degree across reductions, so the default per-fiber
    # Gauss count must scale with the dimension (not a constant offset from the order). Compare the compressed rule's
    # moments (sourced at the default `_implicit_gauss_points`) against an independent high-order reference; a too-low
    # default shows up as ~1e-5 here.
    for (region, phi, order) in ((box((0.0, 0.0), (1.0, 1.0)), x -> x[1] + x[2] - 1.0, (5, 5)),
                                 (box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), x -> x[1] + x[2] + x[3] - 1.0, (3, 3, 3)),
                                 (box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), x -> x[1] + x[2] + x[3] - 1.0, (4, 4, 4))),
        compression in _FCM_COMPRESSIONS

        ref_rule = Unfitted.implicit_volume_quadrature(phi, region; gauss_points=22)
        ref = MomentBasis(region, order)(ref_rule...)
        p = physical_domain(phi; lipschitz=2.0, subcell_length_scale=0.5, compression)
        pts, ws, res = Unfitted.moment_fit_rule(p, region, order)
        @test res < 1.0e-9
        @test MomentBasis(region, order)(pts, ws) ≈ ref atol = 1.0e-9
    end
end

@testset "FCM — a sub-cell hole is not swallowed by the cut rule" begin
    # Regression. The cut rule runs on the *region* box — a whole mesh cell — which is routinely coarser than the
    # geometry. A hole strictly inside the cell that misses its centre and every corner used to prune out of the
    # implicit kernel, so the cell came back integrated as solid (weights summing to the full cell volume) with a
    # machine-zero fit residual: a silent, one-sided over-integration that no diagnostic reported. `implicit_rule`
    # forwards each leaf's Lipschitz constant, which certifies the sign instead of sampling it.
    region = box((0.0, 0.0), (1.0, 1.0))
    order = (3, 3)
    phi = x -> 0.16 - hypot(x[1] - 0.25, x[2] - 0.25)      # Ω = outside the disc
    p = physical_domain(leaf(phi; lipschitz=1.0); subcell_length_scale=0.1)
    @test classify_cell(p, region) === :cut
    _, ws, res = Unfitted.moment_fit_rule(p, region, order)
    @test res < 1.0e-9
    @test all(>=(0), ws)
    @test sum(ws) ≈ 1 - π * 0.16^2 atol = 1e-3

    # The same geometry at the `leaf` default `lipschitz = Inf` keeps the documented uncertified behaviour: the hole
    # stays invisible to the kernel.
    q = physical_domain(leaf(phi); subcell_length_scale=0.1)
    @test classify_cell(q, region) === :cut
    _, wq, _ = Unfitted.moment_fit_rule(q, region, order)
    @test sum(wq) ≈ 1.0 atol = 1e-10
end

@testset "FCM — without a compression, a cut region carries the cut rule itself" begin
    # `compression = nothing` hands the cut rule through untouched — the same points and weights `implicit_rule`
    # returns, with a zero residual — so a caller can integrate cut cells uncompressed, or compare a compression against
    # its own input.
    region = box((0.0, 0.0), (1.0, 1.0))
    order = (4, 4)
    p = physical_domain(x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.3; lipschitz=1.0, subcell_length_scale=0.1,
                        compression=nothing)
    pts, ws, res, status = Unfitted.moment_fit_rule(p, region, order)
    @test (pts, ws) == implicit_rule(p, region, order)
    @test res == 0
    @test status === :fitted
    @test length(ws) > prod(order .+ 1)
end

@testset "FCM — the candidate cap spends its whole budget, without an integer stride" begin
    # Regression, two defects in one line. `_cap_candidates` used to draw `points[1:cld(n, cap):end]`, which
    #
    #   1. kept `cld(n, stride)` points rather than `cap`, because rounding the stride *up* overshoots the reduction —
    #      at `n = 200`, `cap = 150` the stride is 2 and only 100 candidates survive, a third of the budget the solve is
    #      already sized for thrown away; and
    #   2. selected on a fixed period, which aliases against the fiber-block length of the volume rule
    #      (`implicit_volume_quadrature` emits points in blocks of `gauss_points`), reaching only `gauss_points / gcd`
    #      of the per-fiber node positions and drawing a degenerate cloud.
    #
    # Even fractional spacing fixes both. The properties asserted here are the ones the fit depends on: the full budget
    # is spent, no candidate is drawn twice, the ends of the rule are both represented, and a rule already under the cap
    # is handed back untouched.
    order = (4, 4)
    nbasis = prod(order .+ 1)
    budget = Unfitted._candidate_budget(1)
    cap = min(budget * nbasis, max(6 * nbasis, cld(Unfitted._MAX_FIT_MATRIX_ENTRIES, nbasis)))
    pts = [SVector(Float64(i), 0.0) for i in 1:200]
    @test length(pts) > cap          # the premise of the test: the cap must bite

    capped = Unfitted._cap_candidates(pts, nbasis, budget)
    # The whole budget, not `cld(n, stride)` of it. The old stride returned 100.
    @test length(capped) == cap
    # Distinct candidates: a duplicated column is a rank-deficient one.
    @test length(unique(capped)) == cap
    # Monotone, and spanning the rule rather than a prefix of it.
    @test issorted([p[1] for p in capped])
    @test first(capped) == first(pts)
    @test last(capped) == last(pts)
    # No fixed period: the gaps between selected indices must take more than one value, which is exactly what an integer
    # stride cannot do and what the fiber-block aliasing needs.
    gaps = diff([Int(p[1]) for p in capped])
    @test length(unique(gaps)) > 1

    # A rule already inside the cap is returned as-is, identically.
    small = [SVector(Float64(i), 0.0) for i in 1:(cap-1)]
    @test Unfitted._cap_candidates(small, nbasis, budget) === small
end

@testset "FCM — the NNLS retry rescues a many-feature cut cell from the same rule" begin
    # Four holes in one cell at moment order 8. The first attempt's `6 · nbasis` candidates do not span a cone that
    # holds the moments, so the best non-negative fit over them misses by far more than the failure threshold. The retry
    # draws a denser cloud from the *same* cut rule, so the rule it returns is still a subset of that rule's points and
    # reproduces that rule's moments to machine precision.
    region = box((0.0, 0.0), (1.0, 1.0))
    order = (8, 8)
    radius = 0.1
    holes = ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7))
    geom = complement(union((leaf(x -> hypot(x[1] - c[1], x[2] - c[2]) - radius; lipschitz=1.0) for c in holes)...))
    fit = NonnegativeFit(; target_residual=1.0e-10)
    p = physical_domain(geom; subcell_length_scale=0.1, compression=fit)
    @test classify_cell(p, region) === :cut

    rule = implicit_rule(p, region, order)
    basis = MomentBasis(region, order)
    moments = basis(rule...)
    starved = Unfitted._cap_candidates(rule[1], length(basis), Unfitted._candidate_budget(1))
    _, starved_residual = Unfitted.nnls(basis(starved), moments)
    @test starved_residual > Unfitted._FIT_FAILURE_RESIDUAL * norm(moments)
    # The retry genuinely enlarges the cloud.
    @test length(Unfitted._cap_candidates(rule[1], length(basis), Unfitted._candidate_budget(2))) > 4 * length(starved)

    # The fit hands back its whole candidate cloud, the weights off its active set exactly zero.
    pts, ws = fit(rule..., moments, basis)
    @test norm(basis(pts, ws) - moments) < 1.0e-10 * norm(moments)
    @test issubset(pts, rule[1])
    @test all(>=(0), ws)
    @test count(>(0), ws) <= length(basis)

    pts, ws, res, status = Unfitted.moment_fit_rule(p, region, order)
    @test status === :fitted
    @test res < 1.0e-10
    @test sum(ws) ≈ 1 - 4 * π * radius^2 atol = 1.0e-5
end

@testset "FCM — a failed compression falls back to the cut rule" begin
    # A compressed rule that misses the failure threshold is not used, whatever compression produced it: the region
    # takes the cut rule itself — correct and non-negative by construction, merely uncompressed — rather than an empty
    # rule, which would silently drop the cell's whole contribution. The compression below keeps only the measure, a
    # one-point rule whose higher moments are far off, so the verdict is a fallback at every scale: the threshold is
    # relative to ‖m‖.
    order = (8, 8)
    measure_only(points, weights, moments, basis) = ([sum(points .* weights) / sum(weights)], [moments[1]])
    for L in (1.0, 2000.0, 1.0e-3)
        radius = 0.1L
        holes = ((0.3L, 0.3L), (0.7L, 0.3L), (0.3L, 0.7L), (0.7L, 0.7L))
        geom = complement(union((leaf(x -> hypot(x[1] - c[1], x[2] - c[2]) - radius; lipschitz=1.0) for c in holes)...))
        region = box((0.0, 0.0), (L, L))
        p = physical_domain(geom; subcell_length_scale=0.1L, compression=measure_only)
        pts, ws, res, status = Unfitted.moment_fit_rule(p, region, order)
        @test status === :fallback
        @test (pts, ws) == implicit_rule(p, region, order)
        # The fallback rule IS the moments' source, so its own moments are the ‖m‖ the verdict was taken against.
        @test res > Unfitted._FIT_FAILURE_RESIDUAL * norm(MomentBasis(region, order)(pts, ws))
        # Non-negativity survives the fallback: every Saye weight is a product of a positive Gauss weight, a positive
        # fiber half-length and a positive base weight (`_emit_fiber!` in `src/implicit.jl`).
        @test all(>=(0), ws)
        # The cell still integrates correctly — that is the whole point — but at 20×+ the points a compression would
        # have used. That cost is what `AssemblyDiagnostics.cut_fallback_points` reports.
        @test sum(ws) ≈ L^2 * (1 - 4π * 0.1^2) rtol = 1.0e-4
        @test length(ws) > 20 * prod(order .+ 1)
    end
end

@testset "FCM — whether a fit falls back does not depend on the cell's size" begin
    # Regression. The fit-failure threshold used to be absolute while the NNLS target was relative, so the verdict was a
    # statement about the cell's extent rather than about the fit. A Float32 fit settles near 1e-7 relative —
    # `eps(Float32)` is 1.2e-7 — so it misses a 1e-9 target on every attempt and lands in the band between the target
    # and the failure threshold, where the compressed fit is the rule to use. With an absolute 1e-2 the same cell was
    # `:fitted` at L = 1 (residual ≈ 1e-7 · 0.87) and `:fallback` at L = 2000 (≈ 1e-7 · 3.5e6 = 0.35): identical
    # arithmetic, opposite verdicts, and 3072 raw points instead of 25.
    order = (4, 4)
    fit_at(L, target) =
        let radius = 0.1f0 * L,
            holes = ((0.3f0 * L, 0.3f0 * L), (0.7f0 * L, 0.3f0 * L), (0.3f0 * L, 0.7f0 * L), (0.7f0 * L, 0.7f0 * L)),
            geom = complement(union((leaf(x -> sqrt((x[1] - c[1])^2 + (x[2] - c[2])^2) - radius; lipschitz=1.0)
                                     for c in holes)...))

            p = physical_domain(geom; subcell_length_scale=L / 10, max_depth=8,
                                compression=NonnegativeFit(; target_residual=target))
            Unfitted.moment_fit_rule(p, box((0.0f0, 0.0f0), (L, L)), order)
        end
    for L in (1.0f0, 2000.0f0)
        _, ws, res, status = fit_at(L, 1.0e-9)
        @test status === :fitted
        @test length(ws) <= prod(order .+ 1)
        @test sum(ws) ≈ L^2 * (1 - 4π * 0.1f0^2) rtol = 1.0e-4
        # Short of the target, inside the threshold: the residual is Float32's, relative to the cell's own moments.
        @test 1.0e-9 < res / sum(ws) < Unfitted._FIT_FAILURE_RESIDUAL
    end

    # A tolerance the precision can reach is met at any size, and the returned residual is still the absolute one: far
    # above the 1e-6 tolerance at L = 2000, far below it relative to the cut volume.
    _, ws32, res32, status32 = fit_at(2000.0f0, 1.0e-6)
    @test status32 === :fitted
    @test res32 > 1.0e-6
    @test res32 / sum(ws32) < 1.0e-6
end

@testset "FCM — exact moments are octree-depth-independent (3D)" begin
    # The implicit kernel ignores `subcell_length_scale` on smooth cells: a 3D cut cell yields identical exact moments
    # whether the octree would be shallow or deep, and the cut rule stays O(nbasis) rather than 8^depth — the core fix
    # for the 3D out-of-memory blow-up.
    region = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    order = (3, 3, 3)
    phi = x -> x[1] + 0.5 * x[2] + 0.25 * x[3] - 0.9
    coarse = physical_domain(phi; lipschitz=2.0, subcell_length_scale=0.5)
    fine = physical_domain(phi; lipschitz=2.0, subcell_length_scale=1.0 / 2^6, max_depth=8)

    pc, wc, rc = Unfitted.moment_fit_rule(coarse, region, order)
    pf, wf, rf = Unfitted.moment_fit_rule(fine, region, order)
    @test rc < 1e-9 && rf < 1e-9
    basis = MomentBasis(region, order)
    mc = basis(pc, wc)
    @test mc ≈ basis(pf, wf) atol = 1e-10   # depth-independent
    # …and both equal an independent high-order reference (not just each other).
    ref_rule = Unfitted.implicit_volume_quadrature(phi, region; gauss_points=22)
    @test mc ≈ basis(ref_rule...) atol = 1e-9

    nbasis = prod(order .+ 1)
    @test length(implicit_rule(coarse, region, order)[1]) <= 12 * nbasis
end

@testset "FCM — the weight cutoff is relative to the cut volume" begin
    # Regression. The truncation cutoff used to be compared against the compressed weights as an absolute number. A
    # quadrature weight carries the units of a volume, so the same geometry at a domain length scale of 1e-5 has 2D
    # weights of ~1e-12: the truncation then discarded most of the rule (7 of its 9 points on the cell below) while the
    # region was still reported `:fitted` with a machine-zero residual — a residual that was, at the time, captured
    # before the truncation ran and so described a rule nobody received. The measured symptom was a disc integrating 17
    # % of its area.
    #
    # The rule at scale S must be the rule at scale 1, with the points mapped by S and the weights by S^D.
    order = (2, 2)
    for compression in _FCM_COMPRESSIONS
        rule_at = S -> begin
            phi = x -> hypot(x[1] - 0.5S, x[2] - 0.5S) - 0.3S
            p = physical_domain(phi; lipschitz=1.0, subcell_length_scale=S / 32, max_depth=5, compression)
            cell = box((0.25S, 0.25S), (0.5S, 0.5S))          # one cut cell of a 4×4 mesh
            pts, ws, res, status = Unfitted.moment_fit_rule(p, cell, order)
            return cell, pts, ws, res, status
        end
        S = 1.0e-5
        cell1, pts1, ws1, res1, status1 = rule_at(1.0)
        cellS, ptsS, wsS, resS, statusS = rule_at(S)

        @test status1 === :fitted
        @test statusS === :fitted
        @test length(wsS) == length(ws1)
        @test sum(wsS) ≈ sum(ws1) * S^2 rtol = 1.0e-10
        # The whole rule as an integrator, not just its measure: every tensor Legendre moment in the cell's own
        # reference frame scales by S^D and nothing else.
        moments1, momentsS = MomentBasis(cell1, order)(pts1, ws1), MomentBasis(cellS, order)(ptsS, wsS)
        @test momentsS ≈ moments1 .* S^2 rtol = 1.0e-10
        # The reported residual describes the returned rule, and is machine-level relative to the cut volume at both
        # scales — the only scale-free reading.
        @test res1 <= 1.0e-12 * sum(ws1)
        @test resS <= 1.0e-12 * sum(wsS)
    end
end

@testset "FCM — Carathéodory pruning keeps a subset of the cut rule with its moments" begin
    # The pruned rule is a subset of the cut rule's points with non-negative weights, at most nbasis of them positive,
    # reproducing the moments to round-off — on a cell small enough to prune directly (100 cut-rule points against
    # 2 · 81) and on two that are first reduced by recombination (1728 points against 2 · 25, and 512 against 2 · 125 in
    # 3D).
    disc = leaf(x -> hypot(x[1] - 0.0123, x[2] + 0.0345) - 0.6789; lipschitz=1.0)
    holes = complement(union((leaf(x -> hypot(x[1] - c[1], x[2] - c[2]) - 0.1; lipschitz=1.0)
                              for c in ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)))...))
    ball = leaf(x -> norm(x - SVector(0.0123, -0.0345, 0.0217)) - 0.6789; lipschitz=1.0)
    cells = ((disc, box((0.5, 0.25), (0.75, 0.5)), (8, 8)), (holes, box((0.0, 0.0), (1.0, 1.0)), (4, 4)),
             (ball, box((0.5, 0.25, 0.25), (0.75, 0.5, 0.5)), (4, 4, 4)))
    for (geometry, region, order) in cells
        p = physical_domain(geometry; subcell_length_scale=0.1)
        rule = implicit_rule(p, region, order)
        basis = MomentBasis(region, order)
        moments = basis(rule...)
        pts, ws = Caratheodory()(rule..., moments, basis)
        @test issubset(pts, rule[1])
        @test all(>=(0), ws)
        @test count(>(0), ws) <= length(basis) < length(rule[1])
        @test norm(basis(pts, ws) - moments) < 1.0e-13 * norm(moments)
    end

    # Where the moment matrix is numerically rank-deficient — a sliver of a disc at moment order 12 — the pruning stops
    # at the rank, below nbasis, which is the point count the assembly then pays for.
    p = physical_domain(disc; subcell_length_scale=0.1)
    pts, ws, res, status = Unfitted.moment_fit_rule(p, box((0.45, 0.45), (0.7, 0.7)), (12, 12))
    @test status === :fitted
    @test length(pts) < 13^2
    @test res < 1.0e-12 * sum(ws)

    # Float32 carries through, to its own round-off.
    p32 = physical_domain(leaf(x -> hypot(x[1] - 0.5f0, x[2] - 0.5f0) - 0.3f0; lipschitz=1.0);
                          subcell_length_scale=0.1f0)
    pts32, ws32, res32, status32 = Unfitted.moment_fit_rule(p32, box((0.5f0, 0.5f0), (1.0f0, 1.0f0)), (4, 4))
    @test status32 === :fitted
    @test eltype(ws32) === Float32 && eltype(eltype(pts32)) === Float32
    @test res32 < 1.0f-5 * sum(ws32)
end

@testset "FCM — `_prune!` is Carathéodory's theorem" begin
    # A positive combination of 40 columns in ℝ⁶ keeps its value with at most 6 positive coefficients, and with at most
    # 4 when the columns span only 4 dimensions (two rows repeat others).
    x = range(0.05, 0.95; length=40)
    S = [xj^(a - 1) for a in 1:6, xj in x]
    for (M, rank) in ((S, 6), (S[[1, 2, 3, 4, 2, 3], :], 4))
        λ = 1 .+ sin.(1:40) .^ 2
        target = M * λ
        Unfitted._prune!(λ, M)
        @test all(>=(0), λ)
        @test count(>(0), λ) <= rank
        @test M * λ ≈ target rtol = 1.0e-13
    end

    # Coefficients spread over thirty orders of magnitude, on a rank-deficient matrix whose columns are monomials of
    # points crowded towards one end: the pruning still stops at the rank, keeps the value to round-off of its largest
    # term, and drives no coefficient negative.
    x = range(0, 1; length=60) .^ 3
    M = [xj^(a - 1) for a in (1, 2, 3, 4, 5, 6, 7, 8, 2, 5), xj in x]
    λ = 10.0 .^ (30 .* sin.(1:60) .^ 2 .- 30)
    target = M * λ
    Unfitted._prune!(λ, M)
    @test all(>=(0), λ)
    @test count(>(0), λ) <= 8
    @test norm(M * λ - target) <= 1.0e-11 * norm(target)

    # A pass that drifts is undone. The basis below is exact in its first column and pushed off the null space by 1e-6
    # in the others, as eliminations that compounded round-off would leave it — which on clustered monomials they do
    # (`_PRUNE_DRIFT`), though on no moment matrix measured — so walking it through moves `A μ` far past the guard. The
    # pass is replaced by the walk along the first column alone: one coefficient fewer, and `A μ` kept to round-off.
    x = range(0.05, 0.95; length=12)
    A = [xj^(a - 1) for a in 1:4, xj in x]
    V = nullspace(A)
    V[:, 2:end] .+= 1.0e-6 .* sin.(1:12)
    μ = 1 .+ cos.(1:12) .^ 2
    target = A * μ
    walked = Unfitted._walk!(copy(μ), copy(V))
    @test norm(A * walked - target) > 1.0e3 * Unfitted._PRUNE_DRIFT * norm(target)     # the premise: a drifting pass
    Unfitted._prune_pass!(μ, A, V)
    @test all(>=(0), μ)
    @test count(>(0), μ) == 11
    @test norm(A * μ - target) <= 1.0e-14 * norm(target)
end

@testset "FCM — a cut rule or compression with a negative weight is refused" begin
    # A signed rule breaks the α blend, the Dirichlet trace solve's test for unsupported dofs and every squared norm,
    # without a trace: the pipeline refuses it from either stage, and names the stage.
    region = box((0.0, 0.0), (1.0, 1.0))
    disc = x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.3
    function signed_rule(physical, box, order)
        pts, ws = implicit_rule(physical, box, order)
        return pts, [-ws[1]; ws[2:end]]
    end
    signed_compression(points, weights, moments, basis) = (points[1:2], [-moments[1], 2moments[1]])
    for (cut_rule, compression, stage) in
        ((signed_rule, Caratheodory(), "cut rule"), (implicit_rule, signed_compression, "compression"))
        p = physical_domain(disc; lipschitz=1.0, subcell_length_scale=0.1, cut_rule, compression)
        @test_throws ArgumentError Unfitted.moment_fit_rule(p, region, (2, 2))
        @test_throws "the $stage returned the weight" Unfitted.moment_fit_rule(p, region, (2, 2))
    end

    # Round-off below zero is not a sign: a compression may leave such weights where it means zero, and the pipeline
    # truncates them with the rest.
    function roundoff(points, weights, moments, basis)
        pts, ws = Caratheodory()(points, weights, moments, basis)
        return [pts; pts[1:1]], [ws; -eps() * moments[1]]
    end
    p = physical_domain(disc; lipschitz=1.0, subcell_length_scale=0.1, compression=roundoff)
    pts, ws, res, status = Unfitted.moment_fit_rule(p, region, (2, 2))
    @test status === :fitted
    @test all(>(0), ws)
end

@testset "FCM — a cut rule of zero weights carries no measure" begin
    # A cut rule whose weights are all zero integrates nothing, however many points it lists: the region is `:empty`,
    # like one with no points, and the compression is never consulted.
    zero_rule(physical, box, order) = ([box.lower, box.upper], [0.0, 0.0])
    never(args...) = error("the compression must not run on a rule without measure")
    p = physical_domain(x -> x[1] - 0.5; lipschitz=1.0, subcell_length_scale=0.1, cut_rule=zero_rule, compression=never)
    @test Unfitted.moment_fit_rule(p, box((0.0, 0.0), (1.0, 1.0)), (2, 2)) ==
          (SVector{2,Float64}[], Float64[], 0.0, :empty)
end

@testset "FCM — `_nullspace` is a basis of the null space at the numerical rank" begin
    # Wide, the shape `_prune!` factors; with repeated rows; zero; tall; and in Float32. The rank is clear in each, so
    # the basis has as many columns as LinearAlgebra's SVD-based `nullspace` finds, they are independent, and the matrix
    # annihilates them to round-off.
    x = range(-1, 1; length=30)
    V = [xj^(a - 1) for a in 1:8, xj in x]
    for A in (V, V[[1, 2, 3, 2, 1, 4], :], zeros(5, 9), V[:, 1:6], Float32.(V))
        N = Unfitted._nullspace(A)
        @test eltype(N) === eltype(A)
        @test size(N) == size(nullspace(A))
        @test rank(N) == size(N, 2)
        @test norm(A * N) <= 10 * eps(eltype(A)) * norm(A) * norm(N)
    end
end

@testset "FCM — NonnegativeFit refuses a non-positive target" begin
    @test NonnegativeFit().target_residual == 1.0e-6
    @test_throws ArgumentError NonnegativeFit(; target_residual=0.0)
    @test_throws ArgumentError NonnegativeFit(; target_residual=-1.0e-6)
end
