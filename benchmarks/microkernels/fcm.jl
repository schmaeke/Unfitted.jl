using BenchmarkTools
using StaticArrays
using Unfitted: AxisBox, physical_domain
using Unfitted: moment_fit_rule, compute_region_moments, seed_candidate_points
using Unfitted: _eliminate_points, _moment_fit_with_retry, _default_moment_gauss

# Focused per-cut-cell benchmarks for the FCM moment-fit pipeline.
# Today's `build/integration_plan.jl` and `scenarios/fcm_disk_small.jl`
# benches bundle the moment fit together with octree walks, region
# merging, and Dirichlet projection — useful for end-to-end signals but
# too coarse to attribute a regression to the moment-fit kernel itself.
# A focused microkernel surface here makes `moment_fit_rule` and its
# inner pieces (`compute_region_moments`, `seed_candidate_points`,
# `_eliminate_points`) first-class regression targets.
#
# The geometry is a disk of radius `0.4` centred at `(0.5, 0.5)`; the
# benchmark cell `(0.6, 0.4)–(0.8, 0.6)` straddles the boundary
# (φ < 0 in the lower-left corner, φ > 0 in the upper-right) and so
# classifies as `:cut` at every `subcell_depth ≥ 0`. The full pipeline
# (NNMF moment integration → candidate seeding → elimination loop) runs
# on every call.
group = SUITE["microkernels"]["fcm"] = BenchmarkGroup()

let
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    cell = AxisBox(SVector(0.6, 0.4), SVector(0.8, 0.6))

    # `moment_fit_rule` and its sub-kernels at three (moment_order,
    # subcell_depth) points that bracket the regimes the package is
    # typically used at:
    #
    #   * `(4, 3)`  — low-order, shallow octree. Reference floor; the
    #                 NNLS matrix is `25 × ~100` so this is the cheapest
    #                 useful FCM call.
    #   * `(8, 4)`  — order-4 problem with `moment_order_factor = 2`
    #                 (the annular plate's regime). NNLS A grows to
    #                 `81 × ~500`; this is the bench whose per-call cost
    #                 went from seconds to milliseconds in the recent
    #                 fix and which would surface a regression most
    #                 visibly.
    #   * `(12, 4)` — stress test at very high moment order
    #                 (e.g. `order = 6, factor = 2`). nbasis grows to
    #                 169; useful for tracking how the kernels scale
    #                 with basis size rather than only the typical
    #                 working point.
    #
    # `target_residual = 1e-6` matches the package default and the
    # achievable stair-step floor at depth 4 — i.e. attempt 1 lands
    # close to the target and the early-accept path of
    # `_moment_fit_with_retry` is the steady-state hot path.
    for (mo_p, depth) in ((4, 3), (8, 4), (12, 4))
        moment_order = (mo_p, mo_p)
        physical = physical_domain(phi_disk; lipschitz=1.0, subcell_depth=depth)

        tag = "mo=$mo_p depth=$depth"

        # `moment_fit_rule` — the public driver. Composes
        # `compute_region_moments`, `seed_candidate_points`, the
        # elimination loop, and the retry outer loop. Single number
        # that pins down end-to-end per-cell FCM cost.
        group["moment_fit_rule $tag"] = @benchmarkable moment_fit_rule($physical, $cell,
                                                                       $moment_order;
                                                                       target_residual=1.0e-6)

        # Inner pieces. These let a regression attribute its cost
        # increase to either the moment integral, the candidate
        # seeding, or the NNLS elimination — instead of a single
        # opaque jump in `moment_fit_rule`.
        gauss = _default_moment_gauss(moment_order)
        group["compute_region_moments $tag"] = @benchmarkable compute_region_moments($physical,
                                                                                     $cell,
                                                                                     $moment_order;
                                                                                     gauss_per_axis=$gauss)
        group["seed_candidate_points $tag"] = @benchmarkable seed_candidate_points($physical, $cell,
                                                                                   $moment_order;
                                                                                   gauss_per_axis=$gauss)

        # `_eliminate_points` measured with pre-computed moments and
        # candidates so the cost is the inner-loop NNLS only (no
        # octree walks, no candidate seeding). This is the variable
        # whose growth dominates regressions: A is `nbasis × ncand`
        # and grows quadratically in `gauss_per_axis`.
        moments = compute_region_moments(physical, cell, moment_order; gauss_per_axis=gauss)
        candidates = seed_candidate_points(physical, cell, moment_order; gauss_per_axis=gauss)
        group["eliminate_points $tag"] = @benchmarkable _eliminate_points($moments, $candidates,
                                                                          $cell, $moment_order;
                                                                          target_residual=1.0e-6)
    end
end

# Retry-stagnation regression bench. Pre-fix `_moment_fit_with_retry`
# burned through all 3 retry attempts whenever
# `target_residual < integrator_floor`, with each attempt doubling
# per-axis Gauss density (4× candidates in 2D, ~16× NNLS cost). At
# `subcell_depth = 4` the stair-step floor sits near `1e-6`, so
# `target_residual = 1e-8` is well below what the integrator can
# deliver — yet every cut cell paid for the full retry budget.
#
# Post-fix the stagnation guard exits after the first attempt that
# fails to halve the previous residual, and the early-accept path
# returns directly when the first attempt is within a constant of the
# target. This bench is the regression trip-wire: an implementation
# that drops either guard would inflate the timing here by `~30×` and
# the allocations by `~10×` without changing any correctness signal.
let
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    cell = AxisBox(SVector(0.6, 0.4), SVector(0.8, 0.6))
    moment_order = (8, 8)
    physical = physical_domain(phi_disk; lipschitz=1.0, subcell_depth=4)
    gauss = _default_moment_gauss(moment_order)

    retry_group = group["retry_stagnation"] = BenchmarkGroup()

    # Target tighter than the floor. Pre-fix this hammered three
    # full retries per call; post-fix it accepts attempt 1 (or
    # stagnates after attempt 2 at most).
    retry_group["target=1e-8"] = @benchmarkable _moment_fit_with_retry($physical, $cell,
                                                                       $moment_order;
                                                                       target_residual=1.0e-8,
                                                                       max_outer=3,
                                                                       gauss_per_axis=$gauss)

    # Target above the floor, baseline for comparison. The first
    # attempt should always meet target and no retry should fire.
    retry_group["target=1e-3"] = @benchmarkable _moment_fit_with_retry($physical, $cell,
                                                                       $moment_order;
                                                                       target_residual=1.0e-3,
                                                                       max_outer=3,
                                                                       gauss_per_axis=$gauss)
end
