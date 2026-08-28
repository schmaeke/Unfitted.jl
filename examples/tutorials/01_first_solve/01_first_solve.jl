#=
Tutorial 1 — A First Solve
==========================

**What you will learn.** The complete workflow of the package, in the
order you will always use it: describe the domain, build a
discretisation on it, declare the unknown, state the problem, prepare
it, solve it, and look at the answer. Seven calls, nothing else. No
overlays, no immersed geometry, no custom weak forms — those are
Tutorials 2, 3, and 4.

**What you should already know.** What a finite element method is, and
what "weak form", "Dirichlet boundary condition" and "L² error" mean.
Nothing about this package.

**Why one dimension.** Everything below is written on the interval
`Ω = (0, 1)` because that is the smallest complete problem there is: a
reader can hold the whole thing in their head, and the printed numbers
can be checked against a formula by hand.

One dimension is *not* a special case in the library. The core is
written for arbitrary spatial dimension `D ≥ 1` — there are no separate
1-D, 2-D and 3-D code paths — so the seven calls below are literally
the same calls you would make in 2-D or 3-D. Only their arguments grow
a coordinate. The last section of this script proves that by rerunning
the identical workflow on the unit square, changing nothing but the
`box` corners, the `cells` count, and the manufactured data.

**The problem.** A manufactured solution: we *choose* the answer first,
then differentiate it to find the source term that produces it. This is
the standard way to verify a solver, because it gives an exact
reference to measure against.

    u(x)  = sin(π x)                        the solution we want,
    −u″   = π² sin(π x) =: f(x)             the source that produces it,
    u(0)  = u(1) = 0                        homogeneous Dirichlet data.

So we solve the weak form

    ∫_Ω u′ v′ dx = ∫_Ω f v dx    for all admissible v,

and compare the computed `u_h` against the known `u`.
=#

using Unfitted

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

# ── The manufactured solution and its source term ────────────────────────────

exact(x) = sin(pi * x[1])
source(x) = pi^2 * exact(x)          # f = −u″ = π² u for this particular u

# Note that `x` is indexed rather than used as a bare number. Every
# user callback in this package receives a `D`-dimensional point, so
# `x[1]` in 1-D is the same code that reads `x[1]`, `x[2]` in 2-D. The
# habit costs nothing and keeps your callbacks dimension-agnostic.

# ── Step 1: the domain ───────────────────────────────────────────────────────
#
# `box` takes the lower and upper corners as tuples of length `D`. One
# entry per tuple means `D = 1`, so this is the interval (0, 1).

omega = box((0.0,), (1.0,))

# ── Step 2: the discretisation ───────────────────────────────────────────────
#
# `space` puts a Cartesian mesh on the domain and hangs a basis family
# on it. `cells` is the mesh resolution and `order` the polynomial
# degree; the default basis family is a hierarchical integrated
# Legendre basis, which is why nothing here names one. Eight cells at
# order 3 is far more than a sine needs — the point is to see the error
# be small, not to see it be interesting.

V = space(omega; cells=8, order=3)

# ── Step 3: the unknown ──────────────────────────────────────────────────────
#
# A `Field` names an unknown living on a space. The name `:u` is what
# appears in diagnostics and in the VTK output; for a scalar unknown
# there is nothing else to say about it.

u = field(:u, V)

# ── Step 4: the problem ──────────────────────────────────────────────────────
#
# `poisson` is the ready-made Poisson/Laplace problem: it builds the
# stiffness bilinear form `∫ ∇v · ∇u`, the load `∫ f v`, and attaches
# the boundary conditions. `boundary(:all)` selects the whole physical
# boundary `∂Ω`, which in 1-D is the two end points.
#
# `dirichlet(0.0; …)` passes the datum as a bare constant rather than
# as the closure `x -> 0.0`. Prefer constants wherever the value is
# one: a closure is a distinct type, and each distinct type forces the
# assembly kernel to recompile for it.

problem = poisson(u; source=source, dirichlet=[dirichlet(0.0; on=boundary(:all))])

# ── Step 5: prepare ──────────────────────────────────────────────────────────
#
# `prepare` does everything that depends on geometry but not on the
# right-hand side: it builds the integration plan (the admissible boxes
# quadrature runs on), enumerates the degrees of freedom, works out
# which of them the boundary conditions eliminate, and records the
# diagnostics. It is the expensive step, and it is deliberately
# separate so that a time loop or a parameter sweep can pay for it once.

model = prepare(problem)

# ── Step 6: solve ────────────────────────────────────────────────────────────
#
# `solve!` assembles the matrix and right-hand side into the model and
# runs the default direct sparse solve, returning a `Solution` — the
# coefficient vector plus the solver's own diagnostics.

solution = solve!(model)

# ── Step 7: look at the answer ───────────────────────────────────────────────
#
# `value(solution, model, u, x)` evaluates the superposed solution at a
# physical point. Points are tuples of length `D`, hence the trailing
# comma in `(0.2,)` — that is Julia's syntax for a one-element tuple,
# not a typo.
#
# The sample points deliberately avoid the mesh nodes at multiples of
# 1/8. In one dimension the Galerkin solution of this problem *interpolates*
# the exact solution at the mesh nodes — a well-known accident of 1-D
# Poisson, and no measure of how good the approximation is in between.
# Sampling off the nodes reports the honest pointwise error.

println("Tutorial 1 — pointwise values")
for x1 in (0.2, 0.45, 0.7)
    point = (x1,)
    computed = value(solution, model, u, point)
    println("    u_h(", x1, ") = ", computed, "   exact = ", exact(point), "   difference = ",
            abs(computed - exact(point)))
end
println()

# `diagnostics(model, solution; exact)` bundles the reproducibility
# record — dimension, per-level metadata, unknown counts, integration
# regions, symmetry and conditioning checks, solver residual — and,
# when handed the analytic solution, the relative L² error
#
#     ‖u_h − u‖_L²(Ω) / ‖u‖_L²(Ω),
#
# integrated on the same quadrature the solver used. That is the number
# to watch: it should fall as you raise `cells` or `order`.

report = diagnostics(model, solution; exact)
print_run_report("Tutorial 1 — Poisson on the interval (0, 1)", report;
                 parameters=(:cells => V.levels[1].mesh.cells, :order => V.levels[1].order))

# ── The same workflow in two dimensions ──────────────────────────────────────
#
# Nothing above was 1-D-specific except the data. Give `box` two-entry
# corners and `cells` a per-axis count, hand it the 2-D manufactured
# pair `u = sin(πx) sin(πy)`, `f = 2π² u`, and the same calls in the same
# order solve the same problem one dimension up. The same holds in 3-D
# and beyond; only the cost of tensor-product quadrature grows.

exact_2d(x) = sin(pi * x[1]) * sin(pi * x[2])
source_2d(x) = 2 * pi^2 * exact_2d(x)

V_2d = space(box((0.0, 0.0), (1.0, 1.0)); cells=(8, 8), order=3)
u_2d = field(:u, V_2d)
model_2d = prepare(poisson(u_2d; source=source_2d, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
solution_2d = solve!(model_2d)

println()
println("Tutorial 1 — the same script one dimension up")
println("    u_h(0.5, 0.5) = ", value(solution_2d, model_2d, u_2d, (0.5, 0.5)), "   exact = ",
        exact_2d((0.5, 0.5)))
println()
print_run_report("Tutorial 1 — Poisson on the unit square (0, 1)²",
                 diagnostics(model_2d, solution_2d; exact=exact_2d);
                 parameters=(:cells => V_2d.levels[1].mesh.cells, :order => V_2d.levels[1].order))
