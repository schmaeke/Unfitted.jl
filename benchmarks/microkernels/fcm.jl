using BenchmarkTools
using StaticArrays
using Unfitted: AxisBox, physical_domain, leaf
using Unfitted: moment_fit_rule, implicit_volume_quadrature
# `intersect`/`setdiff` are Base set ops; Unfitted adds LevelSet methods.

# Focused per-cut-cell benchmarks for the FCM moment-fit pipeline. The
# end-to-end scenario benches (`scenarios/fcm_disk_small.jl`) bundle the moment
# fit with octree classification, region merging, and Dirichlet projection;
# these microkernels isolate `moment_fit_rule` (the public per-cell driver) and
# the Saye implicit volume rule it is built on, so a regression can be
# attributed to the moment kernel itself.
#
# The pipeline is a single integrator: `moment_fit_rule` builds the exact
# implicit volume rule, recovers the tensor Legendre moments, and runs one NNLS
# solve. Cost and point count scale with the moment basis, not with octree
# depth (the old stair-step integrator grew like 8^depth in 3D).
group = SUITE["microkernels"]["fcm"] = BenchmarkGroup()

# 2D smooth disk cut cell straddling the boundary.
let
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    cell = AxisBox(SVector(0.6, 0.4), SVector(0.8, 0.6))
    for mo_p in (4, 8, 12)
        moment_order = (mo_p, mo_p)
        q = mo_p + 2
        tag = "mo=$mo_p"
        physical = physical_domain(phi_disk; lipschitz=1.0, subcell_length_scale=0.05)
        group["moment_fit_rule $tag"] = @benchmarkable moment_fit_rule($physical, $cell,
                                                                       $moment_order;
                                                                       target_residual=1.0e-6)
        group["implicit_volume_quadrature $tag"] = @benchmarkable implicit_volume_quadrature($phi_disk,
                                                                                             $cell;
                                                                                             gauss_points=($q))
    end
end

# 3D smooth sphere cut cell — the regime that used to OOM with the octree
# integrator. The inner corner is inside (dist ≈ 0.21), the outer corner
# outside (dist ≈ 0.45), so the cell is `:cut`.
let
    phi_sphere(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2 + (x[3] - 0.5)^2) - 0.4
    cell = AxisBox(SVector(0.7, 0.45, 0.45), SVector(0.9, 0.65, 0.65))
    physical = physical_domain(phi_sphere; lipschitz=1.0, subcell_length_scale=0.05)
    fcm3d = group["implicit_3d"] = BenchmarkGroup()
    for mo_p in (4, 6)
        moment_order = (mo_p, mo_p, mo_p)
        q = mo_p + 2
        tag = "mo=$mo_p"
        fcm3d["moment_fit_rule $tag"] = @benchmarkable moment_fit_rule($physical, $cell,
                                                                       $moment_order;
                                                                       target_residual=1.0e-6)
        fcm3d["implicit_volume_quadrature $tag"] = @benchmarkable implicit_volume_quadrature($phi_sphere,
                                                                                             $cell;
                                                                                             gauss_points=($q))
    end
end

# CSG annulus (difference of two disks): a cut cell across the inner rim. The
# multi-level-set kernel keeps both circle boundaries smooth, so this is the
# high-order CSG path rather than a kinked `max` level set.
let
    annulus = setdiff(leaf(x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.45; lipschitz=1.0),
                      leaf(x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.2; lipschitz=1.0))
    physical = physical_domain(annulus; subcell_length_scale=0.05)
    cell = AxisBox(SVector(0.6, 0.6), SVector(0.8, 0.8))
    group["moment_fit_rule csg_annulus mo=8"] = @benchmarkable moment_fit_rule($physical, $cell,
                                                                               (8, 8);
                                                                               target_residual=1.0e-6)
end
