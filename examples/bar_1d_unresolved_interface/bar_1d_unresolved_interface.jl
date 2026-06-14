#=
One-Dimensional Bar With An Unresolved Material Interface
=========================================================

A variation of §5.1 of the UMLHP preprint (arXiv:2604.25797), where the
canonical "1D bar with an unresolved feature" is exercised as a sine
volume load on a rod whose Young's modulus jumps at a location that is
not aligned with any mesh or overlay integration boundary.

We solve the 1D elastic-bar equilibrium

    -(E(x) u′(x))′ = sin(ω x)      in Ω = (0, 1),
              u(0) = 0,
           E(1) u′(1) = 0,

with the piecewise-constant Young's modulus

    E(x) = E₁  for x ∈ [0, a),
           E₂  for x ∈ [a, 1],

and `a = 2/3`. The interface `x = a` is intentionally *unresolved* by
both the mesh and the overlay: neither the base nor the overlay
integration regions split at `a`, so the package must integrate a
basis-function product across a coefficient jump. This deliberately
mirrors the case where the interface is known only through coefficient
evaluations, moves in time, or is otherwise unavailable as an exact
geometric partition.

The exact solution follows from the flux `q(x) = E(x) u′(x)`:

    q(x) = ∫ₓ¹ sin(ω s) ds = (cos(ω x) − cos(ω)) / ω,
    u(x) = ∫₀ˣ q(s) / E(s) ds.

Therefore `u` is continuous, `q` is continuous, and `u′` jumps across
the material interface. The example compares a *base-only*
approximation against a *local-superposition* overlay around the
unresolved interface so the reader can see the H¹-error improvement the
overlay buys.
=#

using Unfitted

include(joinpath(@__DIR__, "..", "reporting.jl"))

# Physical and load parameters. `E1` and `E2` differ by a factor of 7 to
# make the jump in `u′` visible in the VTK output. `omega_load = 8.0`
# puts a handful of oscillations of the source across the rod so the
# error map shows local structure rather than a uniform bias.
omega = box((0.0,), (1.0,))
interface = 2 / 3
omega_load = 8.0
E1 = 1.0
E2 = 7.0

# Piecewise-constant Young's modulus and the sinusoidal source. Both are
# plain Julia callbacks on the physical coordinate; the package's
# `_as_coefficient` machinery wraps them as `FunctionCoefficient`s
# internally.
youngs_modulus(x) = x[1] < interface ? E1 : E2
source(x) = sin(omega_load * x[1])

# Closed-form flux and the antiderivative of the source. Pulled out so
# `exact(x)` can be assembled as a piecewise definition without
# re-deriving the integrals at every evaluation point.
flux(x) = (cos(omega_load * x) - cos(omega_load)) / omega_load
primitive(x) = sin(omega_load * x) / omega_load^2 - x * cos(omega_load) / omega_load

# Exact `u`, piecewise across the interface. Used both as Dirichlet
# data on `x = 0` (it is zero there, so no projection) and as the
# reference for the relative L² error.
function exact(x)
    s = x[1]
    if s <= interface
        return primitive(s) / E1
    end
    return primitive(interface) / E1 + (primitive(s) - primitive(interface)) / E2
end

# Build the Poisson problem on a given space and solve. Returns the
# model, the solution, and a diagnostics record with the relative L²
# error. The natural BC on `x = 1` is the do-nothing variational
# condition — no Dirichlet entry needed.
function solve_bar(V)
    problem = poisson(V; source, diffusion=youngs_modulus,
                      dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))])
    model = prepare(problem)
    solution = solve!(model)
    return model, solution, diagnostics(model, solution; exact)
end

# Two discretizations:
#   * base-only — 5 cells at p = 4 over the whole rod;
#   * overlay   — same base plus a 4-cell, p = 6 overlay around the
#                 interface. The overlay box [0.51, 0.81] straddles the
#                 interface at `x = 2/3` so the local refinement bites
#                 exactly where `u′` is discontinuous.
base_space = space(omega; cells=5, order=4)
overlay_space = overlay(base_space, box((0.51,), (0.81,)); cells=4, order=6)

base_model, base_solution, base_report = solve_bar(base_space)
overlay_model, overlay_solution, overlay_report = solve_bar(overlay_space)

# VTK bundle for the overlay run only — the base run's main interest is
# its higher L² error, which the diagnostics block reports. The overlay
# bundle carries the computed solution, the exact field, the pointwise
# error, the strain `u′`, and the per-point Young's modulus so the
# coefficient jump is visible alongside the discontinuous strain.
out = joinpath(@__DIR__, "output", "bar_1d_unresolved_interface")
write_vtk(out, overlay_solution, overlay_model; subdivisions=8,
          point_data=(uh=(u, c, x, xi) -> u(c, xi), exact=(u, c, x, xi) -> exact(x),
                      error=(u, c, x, xi) -> u(c, xi) - exact(x),
                      strain=(u, c, x, xi) -> field_gradient(overlay_solution, overlay_model, x)[1],
                      youngs_modulus=(u, c, x, xi) -> youngs_modulus(x)),)

# Two reports, side by side, so the reader sees the L²-error gap that
# the overlay buys at the cost of a few extra dofs.
println("1D bar with unresolved material interface")
print_run_report("Base discretization", base_report;
                 parameters=(:cells => base_space.levels[1].mesh.cells,
                             :order => base_space.levels[1].order, :interface => interface))
print_run_report("Overlay discretization", overlay_report;
                 parameters=(:base_cells => overlay_space.levels[1].mesh.cells,
                             :overlay_cells => overlay_space.levels[2].mesh.cells,
                             :overlay_domain => overlay_space.levels[2].mesh.domain,
                             :interface => interface), output=out)
