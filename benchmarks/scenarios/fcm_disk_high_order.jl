using BenchmarkTools
using Unfitted

# High-moment-order companion to `scenarios/fcm_disk_small.jl`: same
# circular FCM geometry, same `subcell_length_scale`, but `order = 4` instead
# of `2`. With the default `moment_order_factor = 2`, this drives the
# moment-fit basis order to `8` per axis (cardinality `81` in 2D) — the
# regime where the per-cut-cell cost diverges most from the order-2 bench.
#
# Specifically:
#
#   * The exact implicit kernel builds an O(nbasis) candidate cloud and the
#     moment fit is a single NNLS solve on the `81 × ncand` design matrix; this
#     bench is the trip-wire for any regression that reintroduces an iterative
#     point-elimination or accuracy-driven retry around that solve.
#   * The kernel's volume-rule construction scales with the per-fiber Gauss
#     count (`max(moment_order) + 2`); a regression inflating it shows up here.
#   * At order 4 the moment basis is `81` in 2D, vs. `9` at order 2 — the
#     per-region NNLS cost is roughly an order of magnitude higher, so this
#     bench is sensitive to NNLS-related regressions the order-2 bench swallows.
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
    # Base cell = 2/8 = 0.25; leaf scale = cell/16 reproduces depth=4.
    disk = physical_domain(phi; lipschitz=1.0, subcell_length_scale=0.25 / 16, max_depth=4)

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
