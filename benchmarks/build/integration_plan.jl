using BenchmarkTools
using Unfitted

group = SUITE["build"]["integration_plan"] = BenchmarkGroup()

# Base-only mesh (no merge work to speak of; reference floor).
let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(32, 32), order=2)
    group["D=2 32x32 base"] = @benchmarkable Unfitted.integration_plan($V)
end

# Base + one overlay — exercises the per-axis coordinate merge and the
# coverage-signature greedy box merge.
let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(32, 32), order=2)
    V = overlay(V, box((0.2, 0.2), (0.8, 0.8)); cells=(17, 17), order=3)
    group["D=2 32x32 +1 overlay"] = @benchmarkable Unfitted.integration_plan($V)
end

# Base + two overlays — same configuration as the existing benchmarks/intersections.jl
# script, so numbers are comparable across the transition.
let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(32, 32), order=2)
    V = overlay(V, box((0.2, 0.2), (0.8, 0.8)); cells=(17, 17), order=3)
    V = overlay(V, box((0.47, 0.35), (0.86, 0.74)); cells=(11, 9), order=4)
    group["D=2 32x32 +2 overlays"] = @benchmarkable Unfitted.integration_plan($V)
end

# D=3 smoke point: the merge step's CartesianIndices walk has a 16^3 candidate set.
let V = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(16, 16, 16), order=2)
    group["D=3 16^3 base"] = @benchmarkable Unfitted.integration_plan($V)
end

# FCM dispatch + NNMF cache path on a disk-shaped Ω. Cut regions go through
# `_cached_moment_fit!`, so the bench measures the realistic moment-fit cost.
let
    R = 0.7
    phi(x) = sqrt(x[1]^2 + x[2]^2) - R
    disk = physical_domain(phi; lipschitz=1.0, subcell_depth=4)
    V = space(box((-1.0, -1.0), (1.0, 1.0)); cells=(16, 16), order=2, physical=disk)
    group["D=2 16x16 FCM disk"] = @benchmarkable Unfitted.integration_plan($V)
end
