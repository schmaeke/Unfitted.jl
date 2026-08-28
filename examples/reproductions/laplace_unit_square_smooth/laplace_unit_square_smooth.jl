#=
Smooth Laplace Problem On The Unit Square
=========================================

A baseline verification problem: the smooth Laplace test introduced by

    G. J. Wagner, W. K. Liu, "Application of essential boundary conditions
    in mesh-free methods: a corrected collocation method", Int. J. Numer.
    Methods Engrg. 47(8) (2000) 1367–1379.

We solve the homogeneous Laplace equation

    -Δu = 0                 in Ω = (0, 1)²,
      u = 0                 on x = 0, x = 1, and y = 0,
      u = sin(πx)           on y = 1.

The exact solution is

    u(x, y) = sin(πx) sinh(πy) / sinh(π).

This script is *not* a refinement stress test — there is nothing
unresolved or singular about the manufactured solution. Its purpose is to
exercise the public workflow end-to-end on a problem where every
intermediate quantity has a known answer:

  1. build a `Space` on the unit square;
  2. assemble the H¹ stiffness form via [`poisson`](@ref);
  3. attach nonzero Dirichlet data on the top edge through
     [`dirichlet`](@ref) — the package projects them onto the boundary
     trace space before strong elimination, so the constrained values
     reflect the correct trace of `u(x, y = 1) = sin(πx)`;
  4. assemble and solve with the default direct sparse solver;
  5. report the relative L² error and write a ParaView bundle with the
     current solution, the exact solution, the pointwise error, and the
     gradient field.

Increase `cells` and / or `order` to watch the error decrease at the
optimal `p`-rate; this example deliberately picks `cells = (12, 12)`
and `order = 4` so a single run on a laptop gives an error well below
`1e-8` without the script taking noticeable time.
=#

using Unfitted

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

# Physical domain Ω = (0, 1)² and a single base level at p = 4. No
# overlays — the manufactured solution is smooth, so refinement adds
# no useful information here.
omega = box((0.0, 0.0), (1.0, 1.0))
V = space(omega; cells=(12, 12), order=4)

# Manufactured exact solution. Sampled both as Dirichlet data on the
# top edge and as the reference for the relative L² error computation.
exact(x) = sin(pi * x[1]) * sinh(pi * x[2]) / sinh(pi)

# Build the Poisson problem with zero right-hand side and Dirichlet
# data on all four edges. The three "u = 0" edges use the scalar
# shortcut; the top edge passes the callback so the package projects
# `sin(πx)` onto the boundary trace space before eliminating.
problem = poisson(V; source=0.0,
                  dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                             dirichlet(0.0; on=boundary(axis=1, side=:upper)),
                             dirichlet(0.0; on=boundary(axis=2, side=:lower)),
                             dirichlet(exact; on=boundary(axis=2, side=:upper))])

# Prepare the model (integration plan + dof layout + diagnostics) and
# run the default direct sparse solve. `diagnostics(model, solution;
# exact)` computes the relative L² error of the solution against the
# manufactured field on the model's own integration plan.
model = prepare(problem)
solution = solve!(model)
report = diagnostics(model, solution; exact)

# Emit a ParaView bundle next to this script. The point-data callbacks
# expose the computed solution, the exact solution, the pointwise
# error, and the physical gradient as separate fields so the bundle
# is self-contained for visual inspection.
out = joinpath(@__DIR__, "output", "laplace_unit_square_smooth")
write_vtk(out, solution, model;
          point_data=(uh=(u, c, x, xi) -> u(c, xi), exact=(u, c, x, xi) -> exact(x),
                      error=(u, c, x, xi) -> u(c, xi) - exact(x),
                      grad_uh=(u, c, x, xi) -> field_gradient(solution, model, x)))

# Final report.
print_run_report("Smooth Laplace unit square", report;
                 parameters=(:cells => V.levels[1].mesh.cells, :order => V.levels[1].order),
                 output=out)
