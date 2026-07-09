#=
Steep tanh Layer Along a Curved Front (order-reduction benchmark)
=================================================================

A manufactured-solution benchmark for the order-reduction feature (covered
coarse cells shed their high-order modes; see
`docs/design/covered-cell-deactivation.md`). The exact solution is a very steep
hyperbolic-tangent layer riding a sinusoidal front on the bi-unit square
`Ω = (−1, 1)²`:

    u(x, y) = tanh(ψ · (y − ½ sin(π x))),      ψ = 100,

so the solution is essentially ±1 away from the front `y = ½ sin(π x)` and swings
across it over a width ~1/ψ = 0.01. The Poisson source `f = −Δu` is taken from
the exact solution by automatic differentiation, so it stays consistent if the
front or ψ is changed. Exact Dirichlet data is imposed on all four faces.

Refinement is a *stack* of overlays whose height is set by the base order `p`:
`p − 1` overlays, overlay i at order `p − i` on a `2ⁱ`-finer mesh, active only
within a *tighter* band of the front. Level by level the mesh refines, the order
drops by one, and the active region shrinks onto the sharp gradient — a curved,
hp-graded refinement built from per-cell activation rather than hand-placed
boxes, stepping the order all the way down to 1.

Mask-aware coverage then reduces each coarser level under the active region of
the next, so the effective order descends toward the front (p → p−1 → … → 1). The
script solves once and reports the relative L² error, the raw-vs-active dof
counts, and the per-level reduced-mode counts, which show the reduction cascading
down the stack.

Note: the base mesh must be fine enough that its cells fit inside the active
bands where the front is locally flat — too coarse a base (cells wider than the
front curves across them) leaves nothing fully covered and nothing to reduce.
=#

using Unfitted
using ForwardDiff: hessian
using LinearAlgebra: tr

include(joinpath(@__DIR__, "..", "reporting.jl"))

# ── Problem: steep tanh layer along y = ½ sin(π x) on the bi-unit square ───────

const PSI = 100.0
const OMEGA = box((-1.0, -1.0), (1.0, 1.0))

front(x1) = 0.5 * sin(pi * x1)
exact(x) = tanh(PSI * (x[2] - front(x[1])))

# f = −Δu, by AD so it tracks any change to `exact` (tr(Hessian) = Laplacian).
source(x) = -tr(hessian(exact, x))

# ── Discretisation: coarse high-p base + a stack of masked, hp-graded overlays ─

# Choose the base order `p`; the stack is `p − 1` overlays, overlay i at order
# `p − i`, mesh `BASE_CELLS · 2ⁱ`, active within `BAND0 / 2^(i−1)` of the front —
# so the order steps down to 1 as the mesh refines onto the gradient. `BASE_CELLS`
# must be fine enough for the base cells to be covered (see the header note).
const BASE_CELLS = (8, 8)
const BASE_ORDER = 5
const N_OVERLAYS = BASE_ORDER - 1
const BAND0 = 0.25

# Activate a cell iff the front passes within `band` of its centre. `active`
# predicates receive the cell box and its Cartesian index.
on_front(band) =
    (b, _) -> abs((b.lower[2] + b.upper[2]) / 2 - front((b.lower[1] + b.upper[1]) / 2)) < band

function layer_space()
    V = space(OMEGA; cells=BASE_CELLS, order=BASE_ORDER)
    for i in 1:N_OVERLAYS
        V = overlay(V, OMEGA; cells=BASE_CELLS .* 2^i, order=BASE_ORDER - i,
                    active=on_front(BAND0 / 2^(i - 1)))
    end
    return V
end

# ── Solve and report (order reduction is on by default) ────────────────────────

bc = dirichlet(exact; on=boundary(:all))
model = prepare(poisson(layer_space(); source, dirichlet=[bc]))
solution = solve!(model)
report = diagnostics(model, solution; exact)

out = joinpath(@__DIR__, "output", "tanh_layer_2d")
print_run_report("2D steep tanh layer along a curved front  (orders $(BASE_ORDER)→1)", report;
                 parameters=(:psi => PSI, :base_cells => BASE_CELLS, :base_order => BASE_ORDER,
                             :n_overlays => N_OVERLAYS, :band0 => BAND0), output=out)
println("  order reduction : ", sum(report.reduced_mode_counts), " modes removed across the stack",
        "  (", report.raw_dofs, " raw dofs → ", report.active_unknowns, " active)")

# ── VTK: the per-cell coverage / reduction mesh data ───────────────────────────

write_vtk(out, solution, model; subdivisions=2,
          point_data=(uh=(u, c, x, xi) -> u(c, xi), exact=(u, c, x, xi) -> exact(x),
                      error=(u, c, x, xi) -> u(c, xi) - exact(x)),)
