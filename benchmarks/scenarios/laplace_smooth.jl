using BenchmarkTools
using Unfitted

# Mirrors examples/laplace_unit_square_smooth.jl. The nonzero Dirichlet edge
# exercises `_project_dirichlet_values!` inside `prepare`, so the prepare bench
# also covers the boundary L² projection (dense cholesky).
group = SUITE["scenarios"]["laplace_smooth"] = BenchmarkGroup()

let
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(12, 12), order=4)
    exact(x) = sin(pi * x[1]) * sinh(pi * x[2]) / sinh(pi)
    problem = poisson(V; source=0.0,
                      dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                                 dirichlet(0.0; on=boundary(axis=1, side=:upper)),
                                 dirichlet(0.0; on=boundary(axis=2, side=:lower)),
                                 dirichlet(exact; on=boundary(axis=2, side=:upper))])
    model = prepare(problem)
    assemble!(model)  # warm matrix so `solve` benches only the linear solve.

    group["prepare"] = @benchmarkable prepare($problem)
    group["assemble"] = @benchmarkable assemble!($model)
    group["solve"] = @benchmarkable solve!($model)
end
