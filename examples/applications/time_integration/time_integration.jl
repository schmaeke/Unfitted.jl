#=
Transient Heat Conduction Driven By OrdinaryDiffEq.jl
=====================================================

Unfitted.jl discretises space. It does not integrate in time, and it does
not want to: Julia already has a mature ODE stack. This example shows how
the two meet — and the meeting point is deliberately boring, which is the
result worth demonstrating.

Semi-discretising

    ∂ₜu − Δu = f    in Ω = (0, 1)²,    u = 0 on ∂Ω,

with a fixed finite-element space in space and nothing at all in time
leaves a linear ODE in the coefficient vector,

    M u̇(t) + K u(t) = f,

where `M` is the mass matrix, `K` the stiffness matrix, and `f` the load
vector — all three assembled once, over the *active* (Dirichlet-eliminated)
dofs, and all three constant because the geometry and the space do not
change during the run. That is exactly the shape `OrdinaryDiffEq.jl` calls
a mass-matrix `ODEProblem`, so the handover is three lines:

    M = assemble_matrix(model, mass_block(u))
    K = assemble_matrix(model, stiffness_block(u))
    f = assemble_vector(model, source_load(u; source=…))

and after that the integrator owns the time axis. It picks the step size,
it decides when to re-factorise, it reports its own statistics. Nothing in
this file re-enters the package once the operators exist: there is no
re-`prepare`, no reassembly, no per-step bookkeeping of any kind. The two
integrators run below are ordinary algorithm choices — swapping one for
another is a one-word edit — and both are adaptive, so the step counts in
the table are the integrator's own decisions, not a schedule imposed here.

A caveat on scope: this is the *fixed-configuration* transient. If the
space itself changes during the run — an overlay that moves with a feature,
cells activated and deactivated as the solution evolves — the operators are
no longer constant and the state has to be projected onto the new space
between intervals, which is the package's own `move!` / `transfer`
workflow rather than an ODE-solver concern.

Dependency note
---------------

`OrdinaryDiffEq` is a heavy dependency — well over a hundred packages, and
a couple of minutes of first-time precompilation. It lives in *this
example's* `Project.toml` and nowhere else. `Unfitted` itself does not
depend on it, does not have an extension for it, and does not need one:
the coupling is three sparse arrays and a callback.

Manufactured solution
---------------------

With `S(x) = sin(πx₁) sin(πx₂)` — which vanishes on ∂Ω and satisfies
`−ΔS = 2π² S` — take

    u(x, t) = (1 + e^{−2π² t}) S(x),    f(x) = 2π² S(x).

Substituting confirms `∂ₜu − Δu = f`: the decaying part is an eigenmode
and cancels itself, leaving the steady part to balance the source. So the
initial state is `2 S`, the load is constant in time, and the solution
relaxes to the steady state `S` — a transient with something to integrate
rather than a pure decay to zero.

The initial coefficient vector is the L² projection of `u(·, 0)` onto the
discrete space, which is one solve with the mass matrix that has already
been assembled:

    u₀ = M \\ assemble_vector(model, source_load(u; source = u(·,0)))

Interpolation would have been the wrong move here for the same reason it
is the wrong move after moving an overlay: the basis is hierarchical and
not interpolatory, so its coefficients are not point values.

What the printed metrics mean
-----------------------------

`relative L2 error` is the error of the final state at `t = T` against
`u(·, T)`, measured in the same L² norm and on the same quadrature the
solver integrated on. It is a *spatial* error: the integrator tolerances
below are tight enough that the time discretisation contributes nothing
visible, which is exactly why the two integrators in the table agree to
several digits while taking different numbers of steps. Tightening
`reltol` will not improve it; refining the mesh or raising `order` will.

The table also prints each integrator's accepted and rejected step counts
and its right-hand-side evaluation count. Those numbers are the evidence
that the integrator, not this script, is driving: the two methods disagree
by a factor of several on how many steps the same problem needs, neither
count is a round number, and one of them rejects steps — all of which is
the error controller reacting to the fast initial transient.

Finally, `symmetry (M) / (K)` report `‖A − Aᵀ‖_∞` for the two operators.
Both must be exactly zero: the forms are symmetric and the assembler
preserves that to the bit, which is what lets the integrator's implicit
stages use a symmetric factorisation.
=#

using Unfitted
using OrdinaryDiffEq
using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

# ── Space discretisation ────────────────────────────────────────────────────

const cells = (8, 8)
const order = 2
const t_final = 0.2

S(x) = sin(π * x[1]) * sin(π * x[2])
source(x) = 2 * π^2 * S(x)
initial(x) = 2 * S(x)
exact(x, t) = (1 + exp(-2 * π^2 * t)) * S(x)

V = space(box((0.0, 0.0), (1.0, 1.0)); cells=cells, order=order)
u = field(:u, V)

# The problem carries the *spatial* operator and the boundary conditions. It
# is never solved as a steady problem here; `prepare` is called for the dof
# layout, the constraints, and the integration plan, which every operator
# below is then assembled against.
problem = Problem((u,); blocks=(stiffness_block(u),),
                  loads=(source_load(u; source=source),),
                  dirichlet=[dirichlet(0.0; on=boundary(:all))])
model = prepare(problem)

# ── The three operators, assembled once ─────────────────────────────────────
#
# Each of these reuses the model's dof layout, constraints, and integration
# plan; none of them touches `model.matrix` or `model.rhs`. Homogeneous
# Dirichlet conditions are eliminated, so the matrices are already the reduced
# operators over the active unknowns and no constraint bookkeeping survives
# into the time loop.

M = assemble_matrix(model, mass_block(u))
K = assemble_matrix(model, stiffness_block(u))
f = assemble_vector(model, source_load(u; source=source))

# L² projection of the initial condition. `source_load` with the initial
# datum in the source slot assembles ∫_Ω u(·,0) Nᵢ dx, and the mass solve
# turns those moments into coefficients.
u₀ = M \ assemble_vector(model, source_load(u; source=initial))

# ── Handover to OrdinaryDiffEq ──────────────────────────────────────────────
#
# `M u̇ = f − K u` is a mass-matrix ODE: the mass matrix goes on the
# `ODEFunction`, the right-hand side is the residual of the spatial operator.
# The Jacobian ∂(f − Ku)/∂u is `−K`, constant — handing it over explicitly
# (rather than letting the integrator difference it) is the whole benefit of
# having assembled the operator in the first place, and `jac_prototype` gives
# the integrator the sparsity pattern to factorise.

rhs!(du, v, p, t) = (mul!(du, K, v); du .= f .- du)
# Constant Jacobian, so the integrator asks for it only a handful of times
# per run — the allocation in `-K` is not on any hot path.
jacobian!(J, v, p, t) = (J .= -K)

odefunction = ODEFunction(rhs!; mass_matrix=M, jac=jacobian!, jac_prototype=-K)
ode = ODEProblem(odefunction, u₀, (0.0, t_final))

# Two adaptive stiff integrators of quite different construction — a
# Rosenbrock method and a variable-order BDF method. Both handle a mass
# matrix; neither needs anything from Unfitted beyond the three arrays above.
const integrators = (("Rodas5P", Rodas5P()), ("FBDF", FBDF()))

# Tolerances are deliberately far tighter than the spatial accuracy, so that
# what the final error measures is the mesh and not the time stepping.
const reltol = 1.0e-10
const abstol = 1.0e-12

# One integration run. The returned coefficient vector is handed back to the
# package as a `Solution` — the public entry point for coefficients computed
# outside it — after which every post-processing path (L² error, field
# evaluation, VTK, diagnostics) works exactly as it would after `solve!`.
function integrate(algorithm)
    ode_solution = solve(ode, algorithm; reltol=reltol, abstol=abstol)
    final = solution(model, ode_solution.u[end]; method=:ordinarydiffeq)
    return (; ode_solution, final, l2=l2_error(final, model, x -> exact(x, t_final)))
end

results = [integrate(algorithm) for (_, algorithm) in integrators]
primary_name, _ = first(integrators)
primary = first(results)

# ── Report ──────────────────────────────────────────────────────────────────

println("Transient heat conduction, t = 0 → ", t_final, " — spatial operators fixed, ",
        "time stepping by OrdinaryDiffEq.jl")
println("  ", rpad("integrator", 14), rpad("accepted", 10), rpad("rejected", 10),
        rpad("rhs evals", 12), "final rel. L² error")
for ((name, _), result) in zip(integrators, results)
    stats = result.ode_solution.stats
    println("  ", rpad(name, 14), rpad(stats.naccept, 10), rpad(stats.nreject, 10),
            rpad(stats.nf, 12), round(result.l2; sigdigits=6))
end
println()
println("  operators assembled once, then reused for every step of both runs:")
println("    M: ", size(M, 1), "×", size(M, 2), ", ", nnz(M), " nonzeros")
println("    K: ", size(K, 1), "×", size(K, 2), ", ", nnz(K), " nonzeros")
println("    f: ", length(f), " entries")
println("  symmetry (M) / (K): ", norm(M - M', Inf), " / ", norm(K - K', Inf))
println()

out = joinpath(@__DIR__, "output", "time_integration")
write_vtk(out, primary.final, model;
          point_data=(u=(uh, c, x, xi) -> uh(c, xi),
                      exact=(uh, c, x, xi) -> exact(x, t_final)))

print_run_report("Transient heat conduction — $(primary_name) at t = $(t_final)",
                 diagnostics(model, primary.final; exact=x -> exact(x, t_final)); output=out,
                 parameters=(:cells => cells, :order => order, :t_final => t_final,
                             :integrator => primary_name, :reltol => reltol, :abstol => abstol))
