using StaticArrays
using LinearAlgebra

# Tensor Legendre moments mₐ = Σ_q w_q ψₐ(ξ(x_q)) of a quadrature rule, used to
# check that a moment-fit rule reproduces a reference set of moments.
function _legendre_moments(points, weights, region, order)
    idx = Unfitted._moment_basis_indices(order)
    factors = Unfitted._legendre_factor_buffers(order, Float64)
    m = zeros(length(idx))
    for j in eachindex(weights)
        Unfitted._fill_legendre_factors!(factors, points[j], region.lower, region.upper, order)
        for (a, α) in pairs(idx)
            m[a] += weights[j] * Unfitted._tensor_legendre_value(factors, α)
        end
    end
    return m
end

@testset "FCM — NNLS wrapper" begin
    # Small well-posed NNLS where the unconstrained solution is non-negative.
    A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    b = A * [0.5, 0.25]
    x, r = Unfitted.nnls(A, b)
    @test x ≈ [0.5, 0.25] atol = 1e-12
    @test r < 1e-12
end

@testset "FCM — moment_fit reproduces exact implicit-kernel moments (1D)" begin
    # On a smooth cut cell `moment_fit_rule` sources its moments from the exact
    # Saye implicit kernel, so the fitted rule reproduces the *exact* tensor
    # Legendre moments. Reference moments come from the kernel volume rule,
    # which is machine-exact for this linear cut.
    p = physical_domain(x -> x[1] - 0.7; lipschitz=1.0, subcell_length_scale=1.0 / 2^8, max_depth=8)
    region = box((0.0,), (1.0,))
    moment_order = (4,)

    ref = Unfitted.implicit_volume_quadrature(x -> x[1] - 0.7, region; gauss_points=8)
    moments = _legendre_moments(ref[1], ref[2], region, moment_order)
    points, weights, residual = Unfitted.moment_fit_rule(p, region, moment_order)
    @test residual < 1e-10
    @test _legendre_moments(points, weights, region, moment_order) ≈ moments atol = 1e-10
    @test moments[1] ≈ 0.7 atol = 1e-14
end

@testset "FCM — single NNLS solve gives an O(nbasis) non-negative rule" begin
    # A 2D disk cut cell: the single Lawson–Hanson solve yields at most
    # nbasis = prod(order .+ 1) non-negative weights, reproducing the exact
    # moments. No point-elimination loop is needed.
    p = physical_domain(x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3; lipschitz=1.0,
                        subcell_length_scale=1.0 / 2^5, max_depth=5)
    region = box((0.0, 0.0), (1.0, 1.0))
    moment_order = (3, 3)
    nbasis = prod(moment_order .+ 1)

    pts, ws, res = Unfitted.moment_fit_rule(p, region, moment_order; target_residual=1.0e-10)
    @test res < 1.0e-9
    @test all(w -> w > 0, ws)
    @test length(pts) <= nbasis
end

@testset "FCM — full cell reproduces tensor Gauss" begin
    # Ω = everything (φ ≡ −1): the kernel returns the full tensor rule, so the
    # fitted moments match plain Gauss and the weights sum to the cell volume.
    p = physical_domain(x -> -1.0; lipschitz=1.0, subcell_length_scale=1.0)
    region = box((0.0, 0.0), (1.0, 1.0))
    pts, ws, res = Unfitted.moment_fit_rule(p, region, (3, 3))
    @test all(w -> w >= 0, ws)
    @test res < 1e-10
    @test sum(ws) ≈ 1.0 atol = 1e-10
end

@testset "FCM — exact moments through the pipeline (2D linear cut)" begin
    # A 2D tilted linear cut: the constant moment equals the exact cut area to
    # machine precision. Ω = {x + 2y ≤ 1} ∩ [0,1]² ⇒ area ∫₀^0.5 (1−2y) dy = 0.25.
    region = box((0.0, 0.0), (1.0, 1.0))
    p = physical_domain(x -> x[1] + 2 * x[2] - 1.0; lipschitz=3.0, subcell_length_scale=0.1)
    pts, ws, res = Unfitted.moment_fit_rule(p, region, (3, 3); target_residual=1.0e-10)
    @test res < 1.0e-9
    @test all(>=(0), ws)
    @test sum(ws) ≈ 0.25 atol = 1e-12
end

@testset "FCM — CSG corner integrates exactly through the pipeline" begin
    # Ω = {x ≤ 0.7} ∩ {y ≤ 0.6} is the box [0,0.7]×[0,0.6]: the multi-level-set
    # kernel makes the polytope corner exact, so the constant moment is the
    # exact corner area.
    region = box((0.0, 0.0), (1.0, 1.0))
    geom = intersect(leaf(x -> x[1] - 0.7), leaf(x -> x[2] - 0.6))
    p = physical_domain(geom; subcell_length_scale=0.1)
    pts, ws, res = Unfitted.moment_fit_rule(p, region, (3, 3); target_residual=1.0e-10)
    @test res < 1.0e-9
    @test all(>=(0), ws)
    @test sum(ws) ≈ 0.7 * 0.6 atol = 1e-12
end

@testset "FCM — moment_fit_rule returns non-negative weights" begin
    # Non-negativity is the whole point of NNLS — verify on smooth and kinked
    # geometries. Smooth cuts reach machine-level residual; a kinked single-leaf
    # `max` level set force-reduces and still yields a valid non-negative rule.
    region = box((0.0, 0.0), (1.0, 1.0))
    order = (3, 3)
    for phi in (x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3, x -> x[1] - 0.4)
        p = physical_domain(phi; lipschitz=1.0, subcell_length_scale=1.0 / 2^5, max_depth=5)
        _, ws, res = Unfitted.moment_fit_rule(p, region, order)
        @test all(w -> w >= 0, ws)
        @test res < 1.0e-8
    end
    pk = physical_domain(x -> max(x[1] - 0.7, x[2] - 0.6); lipschitz=2.0,
                         subcell_length_scale=1.0 / 2^5, max_depth=5)
    _, wsk, resk = Unfitted.moment_fit_rule(pk, region, order)
    @test all(w -> w >= 0, wsk)
    @test resk < Unfitted._FIT_FAILURE_RESIDUAL
end

@testset "FCM — tilted/corner cuts: default Gauss order gives exact moments" begin
    # A corner-beveling linear cut compounds the per-fiber polynomial degree
    # across reductions, so the default per-fiber Gauss count must scale with the
    # dimension (not a constant offset from the order). Compare moment_fit_rule's
    # moments (which use the default `_implicit_gauss_points`) against an
    # independent high-order reference; a too-low default shows up as ~1e-5 here.
    for (region, phi, order) in ((box((0.0, 0.0), (1.0, 1.0)), x -> x[1] + x[2] - 1.0, (5, 5)),
                                 (box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), x -> x[1] + x[2] + x[3] - 1.0, (3, 3, 3)),
                                 (box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), x -> x[1] + x[2] + x[3] - 1.0, (4, 4, 4)))
        ref_rule = Unfitted.implicit_volume_quadrature(phi, region; gauss_points=22)
        ref = _legendre_moments(ref_rule[1], ref_rule[2], region, order)
        p = physical_domain(phi; lipschitz=2.0, subcell_length_scale=0.5)
        pts, ws, res = Unfitted.moment_fit_rule(p, region, order; target_residual=1.0e-10)
        @test res < 1.0e-9
        @test _legendre_moments(pts, ws, region, order) ≈ ref atol = 1.0e-9
    end
end

@testset "FCM — a sub-cell hole is not swallowed by the cut-cell rule" begin
    # Regression. `moment_fit_rule` runs on the *region* box — a whole mesh cell
    # — which is routinely coarser than the geometry. A hole strictly inside the
    # cell that misses its centre and every corner used to prune out of the
    # implicit kernel, so the cell came back integrated as solid (weights summing
    # to the full cell volume) with a machine-zero fit residual: a silent,
    # one-sided over-integration that no diagnostic reported. Forwarding each
    # leaf's Lipschitz constant certifies the sign instead of sampling it.
    region = box((0.0, 0.0), (1.0, 1.0))
    order = (3, 3)
    phi = x -> 0.16 - hypot(x[1] - 0.25, x[2] - 0.25)      # Ω = outside the disc
    p = physical_domain(leaf(phi; lipschitz=1.0); subcell_length_scale=0.1)
    @test classify_cell(p, region) === :cut
    _, ws, res = Unfitted.moment_fit_rule(p, region, order; target_residual=1.0e-10)
    @test res < 1.0e-9
    @test all(>=(0), ws)
    @test sum(ws) ≈ 1 - π * 0.16^2 atol = 1e-3

    # The same geometry at the `leaf` default `lipschitz = Inf` keeps the
    # documented uncertified behaviour: the hole stays invisible to the kernel.
    q = physical_domain(leaf(phi); subcell_length_scale=0.1)
    @test classify_cell(q, region) === :cut
    _, wq, _ = Unfitted.moment_fit_rule(q, region, order; target_residual=1.0e-10)
    @test sum(wq) ≈ 1.0 atol = 1e-10
end

@testset "FCM — the candidate-cloud retry rescues a many-feature cut cell" begin
    # Regression. `moment_fit_rule` retries a failed fit with a *denser candidate
    # cloud*, but the cap in `_cap_candidates` used to be a fixed `6 · nbasis`,
    # so every attempt drew a cloud of the same size and the retry never
    # delivered what it documented. On the cell below — four holes in one cell,
    # fitted at moment order 8 — the three attempts used to bottom out at a
    # residual of 3e-3, thirteen orders short of what the same moments admit; on
    # a cell with more holes still they stayed above `_FIT_FAILURE_RESIDUAL`
    # altogether and the region was tagged `:cut_failed` and dropped.
    region = box((0.0, 0.0), (1.0, 1.0))
    order = (8, 8)
    radius = 0.1
    holes = ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7))
    geom = complement(union((leaf(x -> hypot(x[1] - c[1], x[2] - c[2]) - radius; lipschitz=1.0)
                             for c in holes)...))
    p = physical_domain(geom; subcell_length_scale=0.1)
    @test classify_cell(p, region) === :cut

    # The first attempt in isolation: the same volume rule and (exact) moments
    # the real loop builds, capped at the first attempt's `6 · nbasis` budget.
    leaves = Unfitted._leaves(geom)
    rule = Unfitted.implicit_volume_quadrature(Any[l.f for l in leaves],
                                               x -> Unfitted._inside(geom, x), region;
                                               gauss_points=Unfitted._implicit_gauss_points(order),
                                               max_subdiv=Unfitted._effective_subcell_depth(p,
                                                                                            region),
                                               lipschitz=Float64[l.lipschitz for l in leaves])
    moments = Unfitted._moments_from_rule(rule[1], rule[2], region, order)
    starved = Unfitted._cap_candidates(rule[1], order, Unfitted._candidate_budget(1))
    _, starved_residual = Unfitted._solve_moment_fit(moments, starved, region, order)
    @test starved_residual > Unfitted._FIT_FAILURE_RESIDUAL
    # The retry must genuinely enlarge the cloud — the property the fixed cap
    # silently removed.
    denser = Unfitted._cap_candidates(rule[1], order, Unfitted._candidate_budget(2))
    @test length(denser) > 4 * length(starved)

    # …and with it the fit reaches machine precision on the same moments.
    pts, ws, res, status = Unfitted.moment_fit_rule(p, region, order; target_residual=1.0e-10)
    @test status === :fitted
    @test res < 1.0e-10
    @test all(>=(0), ws)
    @test length(pts) <= prod(order .+ 1)
    @test sum(ws) ≈ 1 - 4 * π * radius^2 atol = 1.0e-5
end

@testset "FCM — a failed fit falls back to the raw volume rule" begin
    # A fit that misses `_FIT_FAILURE_RESIDUAL` no longer yields an empty rule —
    # which silently dropped the cell's whole contribution — but the Saye volume
    # rule the moments were summed from: correct and non-negative by
    # construction, merely uncompressed.
    #
    # Driving the branch takes some doing, because the candidate-cloud retry
    # above fits essentially every reachable cut cell to machine precision. The
    # lever used here is that `_FIT_FAILURE_RESIDUAL` bounds an *absolute* L²
    # residual, which therefore scales with the cell's measure: a Float32 fit
    # settles around 1e-7 *relative*, so on a cell 2000 units wide it lands well
    # above 1e-2 absolute. The same geometry in Float64 fits and needs no
    # fallback, which is asserted below as the control.
    L = 2000.0f0
    radius = 200.0f0
    holes = ((0.3f0 * L, 0.3f0 * L), (0.7f0 * L, 0.3f0 * L), (0.3f0 * L, 0.7f0 * L),
             (0.7f0 * L, 0.7f0 * L))
    order = (4, 4)
    exact = L^2 - 4 * π * radius^2
    geom32 = complement(union((leaf(x -> sqrt((x[1] - c[1])^2 + (x[2] - c[2])^2) - radius;
                                    lipschitz=1.0) for c in holes)...))
    p32 = physical_domain(geom32; subcell_length_scale=L / 10, max_depth=8)
    region32 = box((0.0f0, 0.0f0), (L, L))

    pts, ws, res, status = Unfitted.moment_fit_rule(p32, region32, order; target_residual=1.0e-6)
    @test status === :fallback
    @test res > Unfitted._FIT_FAILURE_RESIDUAL
    # Non-negativity survives the fallback: every Saye weight is a product of a
    # positive Gauss weight, a positive fiber half-length and a positive base
    # weight (`_emit_fiber!` in `src/implicit.jl`).
    @test all(>=(0), ws)
    # The cell still integrates correctly — that is the whole point — but at
    # 20×+ the points a successful fit would have used. That cost is what
    # `AssemblyDiagnostics.cut_fallback_points` reports.
    @test sum(ws) ≈ exact rtol = 1.0e-4
    @test length(ws) > 20 * prod(order .+ 1)

    # Control: identical geometry in Float64 fits, so no fallback and an
    # O(nbasis) rule.
    geom64 = complement(union((leaf(x -> hypot(x[1] - Float64(c[1]), x[2] - Float64(c[2])) -
                                         Float64(radius); lipschitz=1.0) for c in holes)...))
    p64 = physical_domain(geom64; subcell_length_scale=Float64(L) / 10, max_depth=8)
    region64 = box((0.0, 0.0), (Float64(L), Float64(L)))
    pts64, ws64, res64, status64 = Unfitted.moment_fit_rule(p64, region64, order;
                                                            target_residual=1.0e-6)
    @test status64 === :fitted
    @test res64 < Unfitted._FIT_FAILURE_RESIDUAL
    @test length(ws64) <= prod(order .+ 1)
    @test sum(ws64) ≈ exact rtol = 1.0e-4
end

@testset "FCM — exact moments are octree-depth-independent (3D)" begin
    # The implicit kernel ignores `subcell_length_scale` on smooth cells: a 3D
    # cut cell yields identical exact moments whether the octree would be shallow
    # or deep, and the candidate cloud stays O(nbasis) rather than 8^depth — the
    # core fix for the 3D out-of-memory blow-up.
    region = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    order = (3, 3, 3)
    phi = x -> x[1] + 0.5 * x[2] + 0.25 * x[3] - 0.9
    coarse = physical_domain(phi; lipschitz=2.0, subcell_length_scale=0.5)
    fine = physical_domain(phi; lipschitz=2.0, subcell_length_scale=1.0 / 2^6, max_depth=8)

    pc, wc, rc = Unfitted.moment_fit_rule(coarse, region, order)
    pf, wf, rf = Unfitted.moment_fit_rule(fine, region, order)
    @test rc < 1e-9 && rf < 1e-9
    mc = _legendre_moments(pc, wc, region, order)
    @test mc ≈ _legendre_moments(pf, wf, region, order) atol = 1e-10   # depth-independent
    # …and both equal an independent high-order reference (not just each other).
    ref_rule = Unfitted.implicit_volume_quadrature(phi, region; gauss_points=22)
    @test mc ≈ _legendre_moments(ref_rule[1], ref_rule[2], region, order) atol = 1e-9

    nbasis = prod(order .+ 1)
    pts, _ = Unfitted.implicit_volume_quadrature(phi, region;
                                                 gauss_points=Unfitted._implicit_gauss_points(order))
    @test length(pts) <= 12 * nbasis
end
