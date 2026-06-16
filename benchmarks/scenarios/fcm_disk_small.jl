using BenchmarkTools
using Unfitted

# Mirrors examples/fcm_disk_helmholtz_2d.jl at a smaller mesh. The `prepare`
# bench is the most useful FCM regression signal here: it triggers the
# cell-level classification fold and the NNMF moment-fit for every cut region.
group = SUITE["scenarios"]["fcm_disk_small"] = BenchmarkGroup()

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

    V = space(omega; cells=(8, 8), order=2, physical=disk)
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
