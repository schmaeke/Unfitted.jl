using BenchmarkTools
using Unfitted

# High-moment-order companion to `scenarios/fcm_disk_small.jl`: same
# circular FCM geometry, same `subcell_depth`, but `order = 4` instead
# of `2`. With the default `moment_order_factor = 2`, this drives the
# moment-fit basis order to `8` per axis (cardinality `81` in 2D) — the
# regime where the NNLS cost, the retry-mechanism behaviour, and the
# per-leaf Gauss density meaningfully diverge from what the order-2
# bench exposes.
#
# Specifically:
#
#   * NNLS A matrix grows quadratically in `gauss_per_axis × ncand`, so
#     any regression that reintroduces the old `gauss_per_axis =
#     moment_order + 1` default (instead of the
#     `ceil((moment_order + 1) / 2)` minimum) shows up here as a
#     `4ˣ`-ish per-region cost jump.
#   * The integrator's stair-step floor sits near the default
#     `target_residual = 1e-6`, so any regression in
#     `_moment_fit_with_retry`'s early-accept / stagnation guards
#     reintroduces the multi-attempt retry blow-up.
#   * At order 4 the moment basis is `81` in 2D, vs. `9` at order 2 —
#     the per-region NNLS cost is roughly an order of magnitude higher,
#     so this bench is sensitive to NNLS-related regressions that the
#     order-2 bench's smaller A matrix swallows.
#
# The 8×8 grid mirrors `fcm_disk_small.jl` so per-cell numbers are
# directly comparable across orders.
group = SUITE["scenarios"]["fcm_disk_high_order"] = BenchmarkGroup()

let
    R = 0.7
    phi(x) = sqrt(x[1]^2 + x[2]^2) - R
    function rhs(x)
        s = (x[1]^2 + x[2]^2) / R^2
        return (8 - 16s) / R^2 + (1 - s)^2
    end

    omega = box((-1.0, -1.0), (1.0, 1.0))
    disk = physical_domain(phi; lipschitz=1.0, subcell_depth=4)

    V = space(omega; cells=(8, 8), order=4, physical=disk)
    u = field(:u, V)
    problem = Problem((u,);
                      blocks=(stiffness_block(u; diffusion=1.0), mass_block(u; coefficient=1.0)),
                      loads=(source_load(u; source=rhs),),
                      dirichlet=[dirichlet(0.0; on=boundary(:all))])

    model = prepare(problem)
    assemble!(model)

    group["prepare"] = @benchmarkable prepare($problem)
    group["assemble"] = @benchmarkable assemble!($model)
    group["solve"] = @benchmarkable solve!($model)
end
