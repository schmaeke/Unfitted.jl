using StaticArrays
using LinearAlgebra

@testset "FCM — NNLS wrapper" begin
    # Small well-posed NNLS where the unconstrained solution is non-negative.
    A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    b = A * [0.5, 0.25]
    x, r = Unfitted.nnls!(A, b)
    @test x ≈ [0.5, 0.25] atol = 1e-12
    @test r < 1e-12
end

@testset "FCM — octree leaves on a trivial Ω = whole box" begin
    # phi ≡ -1 ⇒ Ω is everything. The certificate at depth 0 fires → single :full leaf.
    p = physical_domain(x -> -1.0; lipschitz=1.0, subcell_length_scale=1.0)
    leaves = Tuple{AxisBox{2,Float64},Symbol}[]
    Unfitted.foreach_octree_leaf(p, box((0.0, 0.0), (1.0, 1.0))) do leaf, kind
        push!(leaves, (leaf, kind))
    end
    @test length(leaves) == 1
    @test leaves[1][2] === :full
    @test leaves[1][1] == box((0.0, 0.0), (1.0, 1.0))
end

@testset "FCM — octree leaves on a trivial Ω = ∅" begin
    # phi ≡ +1 ⇒ Ω is empty. Single :fictitious leaf at depth 0.
    p = physical_domain(x -> 1.0; lipschitz=1.0, subcell_length_scale=1.0)
    leaves = Tuple{AxisBox{2,Float64},Symbol}[]
    Unfitted.foreach_octree_leaf(p, box((0.0, 0.0), (1.0, 1.0))) do leaf, kind
        push!(leaves, (leaf, kind))
    end
    @test length(leaves) == 1
    @test leaves[1][2] === :fictitious
end

@testset "FCM — octree leaves enumerate :full and :cut for a real cut" begin
    # Ω = left half. Octree subdivides; some leaves :full, some :cut at max depth.
    p = physical_domain(x -> x[1] - 0.5; lipschitz=1.0, subcell_length_scale=0.125, max_depth=3)
    full_count = Ref(0)
    cut_count = Ref(0)
    fict_count = Ref(0)
    Unfitted.foreach_octree_leaf(p, box((0.0, 0.0), (1.0, 1.0))) do _, kind
        kind === :full && (full_count[] += 1)
        kind === :cut && (cut_count[] += 1)
        kind === :fictitious && (fict_count[] += 1)
    end
    @test full_count[] > 0
    @test fict_count[] > 0
    # At a coarse depth, some leaves straddle x=0.5 → :cut.
    @test cut_count[] > 0
end

@testset "FCM — moment integration on trivial Ω matches analytic" begin
    # Ω = whole box [0, 1]. Moments of 1, x, x² match analytic integrals.
    p = physical_domain(x -> -1.0; lipschitz=1.0, subcell_length_scale=1.0)
    moments = Unfitted.compute_region_moments(p, box((0.0,), (1.0,)), (4,))
    # The tensor Legendre basis on [0, 1]:
    #   ψ_0(x) = L_0((2x-1)) = 1
    #   ψ_1(x) = L_1((2x-1)) = 2x - 1
    # Integrals over [0, 1]:
    #   ∫_0^1 1 dx = 1
    #   ∫_0^1 (2x - 1) dx = 0  (Legendre orthogonality)
    @test moments[1] ≈ 1.0 atol = 1e-12
    @test abs(moments[2]) < 1e-12
end

@testset "FCM — candidate seeding respects Ω" begin
    p = physical_domain(x -> x[1] - 0.5; lipschitz=1.0, subcell_length_scale=0.125, max_depth=3)
    pts = Unfitted.seed_candidate_points(p, box((0.0,), (1.0,)), (4,))
    @test !isempty(pts)
    @test all(pt[1] <= 0.5 + 1e-12 for pt in pts)  # all seeded inside Ω

    # Ω = ∅ → no candidates.
    p_empty = physical_domain(x -> 1.0; lipschitz=1.0, subcell_length_scale=1.0)
    @test isempty(Unfitted.seed_candidate_points(p_empty, box((0.0,), (1.0,)), (4,)))
end

@testset "FCM — single-shot moment_fit matches its own moments" begin
    # The fitted rule integrates each tensor Legendre basis function to
    # exactly the moments computed by `compute_region_moments` (within NNLS
    # tolerance). Note: the absolute integration accuracy of the moments
    # themselves is bounded by `subcell_length_scale` (stair-step on :cut leaves).
    p = physical_domain(x -> x[1] - 0.7; lipschitz=1.0, subcell_length_scale=1.0 / 2^8, max_depth=8)
    region = box((0.0,), (1.0,))
    moment_order = (4,)

    moments = Unfitted.compute_region_moments(p, region, moment_order)
    points, weights, residual = Unfitted.moment_fit_rule(p, region, moment_order)
    @test residual < 1e-10

    # Reconstruct each moment from the fitted rule; should match the input moments.
    indices = Unfitted._moment_basis_indices(moment_order)
    factors = Unfitted._legendre_factor_buffers(moment_order, Float64)
    for (a, idx) in pairs(indices)
        est = 0.0
        for j in eachindex(points)
            Unfitted._fill_legendre_factors!(factors, points[j], region.lower, region.upper,
                                             moment_order)
            est += weights[j] * Unfitted._tensor_legendre_value(factors, idx)
        end
        @test est ≈ moments[a] atol = 1e-10
    end
end

@testset "FCM — moment integration converges with subcell refinement" begin
    # The stair-step approximation gives O(L) error on cut leaves, where L is
    # the resolved leaf size set by `subcell_length_scale`.
    # Σ-of-weights = ∫_Ω 1 dx → here = 0.7 for the left-of-0.7 cut.
    region = box((0.0,), (1.0,))
    moment_order = (4,)
    truth = 0.7
    errors = Float64[]
    for depth in (4, 6, 8, 10)
        p = physical_domain(x -> x[1] - 0.7; lipschitz=1.0, subcell_length_scale=1.0 / 2^depth,
                            max_depth=depth)
        m = Unfitted.compute_region_moments(p, region, moment_order)
        push!(errors, abs(m[1] - truth))
    end
    # Each refinement should at least halve the error (in practice ~4×).
    @test all(errors[i+1] < errors[i] for i in 1:(length(errors)-1))
    @test errors[end] < 1e-3
end

@testset "FCM — single-shot fit reproduces full-cell tensor Gauss" begin
    # Ω = whole region → the moments equal those of plain tensor Gauss, and
    # the fit weights are non-negative with residual ~0. We don't expect the
    # fit to reproduce the standard Gauss rule pointwise (NNLS prefers a
    # sparser support), but Σw and ∫ψ_α reconstruction must match.
    p = physical_domain(x -> -1.0; lipschitz=1.0, subcell_length_scale=1.0)
    region = box((0.0, 0.0), (1.0, 1.0))
    moment_order = (3, 3)

    points, weights, residual = Unfitted.moment_fit_rule(p, region, moment_order; elimination=false)
    @test all(w -> w >= 0, weights)
    @test residual < 1e-10
    @test sum(weights) ≈ 1.0 atol = 1e-10
end

# --- Slice 5b: point elimination + outer retry ---

@testset "FCM — elimination prunes to min_points at loose tolerance" begin
    # 2D disk-in-square cut. At a loose target residual the elimination loop
    # should drop points down to the algorithmic floor `prod(moment_order)`.
    p = physical_domain(x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3; lipschitz=1.0,
                        subcell_length_scale=1.0 / 2^5, max_depth=5)
    region = box((0.0, 0.0), (1.0, 1.0))
    moment_order = (3, 3)
    min_points = prod(moment_order)

    pts, ws, res = Unfitted.moment_fit_rule(p, region, moment_order; target_residual=1.0e-2)
    @test length(pts) == min_points
    @test res < 1.0e-2
    @test all(w -> w > 0, ws)
end

@testset "FCM — elimination keeps tight rule at tight tolerance" begin
    # Same setup with target_residual at machine tolerance: elimination
    # cannot reduce below the basis dimension without violating the target.
    p = physical_domain(x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3; lipschitz=1.0,
                        subcell_length_scale=1.0 / 2^5, max_depth=5)
    region = box((0.0, 0.0), (1.0, 1.0))
    moment_order = (3, 3)
    nbasis = prod(moment_order .+ 1)

    pts, ws, res = Unfitted.moment_fit_rule(p, region, moment_order; target_residual=1.0e-10)
    @test length(pts) <= nbasis
    @test length(pts) >= prod(moment_order)
    @test res < 1.0e-10
end

@testset "FCM — elimination reproduces moments to NNLS tolerance" begin
    # Whatever point count the elimination chooses, the fitted rule must
    # reproduce the moments to within target_residual.
    p = physical_domain(x -> x[1] - 0.6; lipschitz=1.0, subcell_length_scale=1.0 / 2^6, max_depth=6)
    region = box((0.0,), (1.0,))
    moment_order = (4,)

    moments = Unfitted.compute_region_moments(p, region, moment_order)
    pts, ws, res = Unfitted.moment_fit_rule(p, region, moment_order; target_residual=1.0e-10)
    @test res < 1.0e-10

    indices = Unfitted._moment_basis_indices(moment_order)
    factors = Unfitted._legendre_factor_buffers(moment_order, Float64)
    for (a, idx) in pairs(indices)
        est = 0.0
        for j in eachindex(pts)
            Unfitted._fill_legendre_factors!(factors, pts[j], region.lower, region.upper,
                                             moment_order)
            est += ws[j] * Unfitted._tensor_legendre_value(factors, idx)
        end
        @test est ≈ moments[a] atol = 1.0e-9
    end
end

@testset "FCM — outer retry terminates within max_outer" begin
    # The retry loop must always terminate within `max_outer` attempts and
    # return a non-catastrophic rule (residual < `_FIT_FAILURE_RESIDUAL`).
    # Whether retry actually triggers depends on geometry — both early-exit
    # and exhaustion are valid outcomes for this contract.
    p = physical_domain(x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3; lipschitz=1.0,
                        subcell_length_scale=0.0625, max_depth=4)
    region = box((0.0, 0.0), (1.0, 1.0))
    moment_order = (3, 3)
    max_outer = 4

    pts, ws, res, _, outer_iter = Unfitted._moment_fit_with_retry(p, region, moment_order;
                                                                  target_residual=1.0e-10,
                                                                  max_outer=max_outer,
                                                                  gauss_per_axis=(2, 2),)
    @test 1 <= outer_iter <= max_outer
    @test !isempty(pts)
    @test res < Unfitted._FIT_FAILURE_RESIDUAL
    @test all(w -> w > 0, ws)
end

@testset "FCM — outer retry uses fewer attempts on a well-resolved cut" begin
    # With a sufficiently small subcell_length_scale the first outer attempt succeeds.
    p = physical_domain(x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3; lipschitz=1.0,
                        subcell_length_scale=1.0 / 2^6, max_depth=6)
    region = box((0.0, 0.0), (1.0, 1.0))
    moment_order = (3, 3)

    _, _, res, _, outer_iter = Unfitted._moment_fit_with_retry(p, region, moment_order;
                                                               target_residual=1.0e-10, max_outer=4,
                                                               gauss_per_axis=ntuple(d -> 2 *
                                                                                          moment_order[d] +
                                                                                          1, 2),)
    @test outer_iter == 1
    @test res < 1.0e-10
end

@testset "FCM — moment_fit_rule returns positive weights" begin
    # Non-negativity is the whole point of NNLS — verify it on a few different
    # cut configurations.
    region = box((0.0, 0.0), (1.0, 1.0))
    moment_order = (3, 3)
    for phi in (x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3, x -> x[1] - 0.4,
                x -> max(x[1] - 0.7, x[2] - 0.6))   # L-shaped Ω
        p = physical_domain(phi; lipschitz=2.0, subcell_length_scale=1.0 / 2^5, max_depth=5)
        pts, ws, res = Unfitted.moment_fit_rule(p, region, moment_order)
        @test all(w -> w >= 0, ws)
        @test res < 1.0e-8
    end
end

@testset "FCM — elimination beats single-shot on point count at loose tolerance" begin
    # In 2D with a non-trivial Ω and a loose target, elimination should
    # produce a strictly smaller (or equal) rule than single-shot.
    p = physical_domain(x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3; lipschitz=1.0,
                        subcell_length_scale=0.0625, max_depth=4)
    region = box((0.0, 0.0), (1.0, 1.0))
    moment_order = (3, 3)

    pts_single, _, _ = Unfitted.moment_fit_rule(p, region, moment_order; elimination=false)
    pts_elim, _, _ = Unfitted.moment_fit_rule(p, region, moment_order; elimination=true,
                                              target_residual=1.0e-3)
    @test length(pts_elim) <= length(pts_single)
end
