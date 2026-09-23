#=
Automated hp Adaptivity On A Curved Interior Layer
==================================================

The benchmark the adaptive verbs of `src/adaptivity.jl` were developed and
measured against. It is not (yet) the reproduction of a published figure:
nothing in the UMLHP preprint corresponds to it. It lives here because it is
the executable form of the package's *adaptivity* contract in the same sense
that the other scripts in this directory are the executable form of its
approximation contract — a fixed problem, a fixed protocol and a number that
must not move.

**The problem.** A hyperbolic-tangent layer whose crest follows a sine, on
the unit square:

    u(x, y) = tanh( λ (y − s(x)) ),      s(x) = ½ + ¼ sin(2πx),      λ = 30
    −Δu = f  in Ω = (0, 1)²,             u = exact  on ∂Ω.

The manufactured source follows from ∂φ tanh = 1 − u² with φ = λ(y − s):

    u_x  = −λ s′ (1 − u²)                u_xx = −λ s″ (1 − u²) − 2λ² (s′)² u (1 − u²)
    u_y  =  λ    (1 − u²)                u_yy = −2λ² u (1 − u²)
    f    = −Δu = (1 − u²) [ λ s″ + 2λ² u (1 + (s′)²) ]

This is the hardest of the three shapes an adaptive scheme meets. It is not a
point singularity, which a graded mesh anchored at a known corner handles
without any decision at all, and it is not a smooth bump, which pure
p-refinement handles. It is an *analytic* feature concentrated on a curve: it
wants a great deal of resolution somewhere, nothing anywhere else, and the
somewhere is not axis aligned. Starting from a **linear** 6×6 base is
deliberate — p = 1 is the least informative state a loop can start from.

**The loop.** Three verbs and nothing else; the caller owns the iteration, as
the caller owns the solve loop:

    est = estimate(model, u)                       # Bank–Weiser indicator
    V   = refine(V, est; theta, previous)          # Dörfler, then h or p per cell
    model = adapted(model, V)                      # one rebuild

`refine` decides h against p from the cell's own refinement history, by
Melenk & Wohlmuth's predicted error reduction: a cell that met the reduction
its last step was entitled to expect takes p again, one that fell short takes
h. That is why `previous` is threaded through the loop — it carries the space
and estimate of the preceding cycle, from which the rule reads what was done
and what it bought.

**The metric, and why it is not the one `diagnostics` reports.** The error
below is a relative energy error measured on a FIXED background lattice,
independent of the space being measured. A model's own quadrature is sized
from its own space, so using it to compare two spaces flatters the finer one —
measured, by 15–25 % on uniform spaces and by ~22 % on an adapted stack. The
`relative L2 error` printed in the closing report is the model-plan quantity,
kept because it is what every other script here prints and what the example
suite regression-checks; it is not the number this benchmark is about.

**What it is measured against.** deal.II 9.5.1's step-27 hp strategy, on the
identical problem from the identical 6×6 degree-1 start, with θ = 0.5 Dörfler
marking on the squared indicator and p ≤ 8 on both sides, scored on the same
lattice with the same analytic denominator (both codes compute |u|_H¹ =
9.452408264813 independently and agree). Geometric mean over 2 000–51 179
unknowns, relative energy error:

    this scheme vs deal.II step-27                 1.6× better
    this scheme vs order-tied-to-depth refinement  6.8× better

Convergence is exponential in the energy norm. In relative L² the same runs
show a staircase, which is a property of that norm on this problem — the L²
error is dominated by a small neighbourhood of the crest and only moves when
refinement reaches it — and not of the method: the energy error and the
estimator both fall by a near-constant factor of 1.45 per cycle.

Environment knobs, so the run can be shortened without editing the script:
`TANH_CYCLES`, `TANH_LAMBDA`, `TANH_DEPTH`, `TANH_WRITE_OUTPUT`,
`TANH_TABLE_ENERGY`. The last one gates the per-cycle `rel_energy` column
only — the closing headline below the loop is always computed — because
scoring every cycle on the 96² lattice costs more than every solve in the run
put together. Measured at -O0, four threads, `TANH_WRITE_OUTPUT = false`:
35.7 s wall (51.2 s user) for the whole script with the column on, 15.7 s
(28.1 s user) with it off, for a bit-identical answer. The column is on by
default, because the fall of the energy error per cycle is what the study is
for; the example smoke suite turns it off and bands the closing headline
instead.
=#

using Unfitted

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

const CYCLES = env("TANH_CYCLES", 22)
const LAMBDA = env("TANH_LAMBDA", 30.0)
const DEPTH = env("TANH_DEPTH", 5)
const WRITE_OUTPUT = env("TANH_WRITE_OUTPUT", true)
const TABLE_ENERGY = env("TANH_TABLE_ENERGY", true)

const THETA = 0.5
const PMAX = 8
const BASE_CELLS = 6

const omega = box((0.0, 0.0), (1.0, 1.0))

front(x) = 0.5 + 0.25 * sinpi(2x)
front_d(x) = 0.5pi * cospi(2x)                       # s′
front_dd(x) = -pi^2 * sinpi(2x)                      # s″

exact(x) = tanh(LAMBDA * (x[2] - front(x[1])))

function exact_gradient(x)
    w = 1 - exact(x)^2
    return (-LAMBDA * front_d(x[1]) * w, LAMBDA * w)
end

function source(x)
    u = exact(x)
    w = 1 - u * u
    return w * (LAMBDA * front_dd(x[1]) + 2 * LAMBDA^2 * u * (1 + front_d(x[1])^2))
end

problem(V) = poisson(V; source=source, dirichlet=[dirichlet(exact; on=boundary(:all))])

# ── The ruler ─────────────────────────────────────────────────────────────────
#
# A fixed background lattice of `panels²` panels, each carrying a 5-point Gauss
# rule per axis. It depends on neither the mesh nor the polynomial degree, which
# is the whole point: the same nodes and weights score every space in the run and
# the uniform spaces it is compared against. 96 panels is converged for λ = 30 —
# doubling it moves the reported error in the fifth digit.
const GAUSS5 = ((-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831,
                 0.9061798459386640),
                (0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665,
                 0.2369268850561891))

function lattice_energy_error(solution, model; panels::Int=96)
    ξ, w = GAUSS5
    h = 1 / panels
    err = 0.0
    ref = 0.0
    for i in 0:(panels-1), a in 1:5
        x1 = (i + 0.5 * (1 + ξ[a])) * h
        wx = 0.5 * w[a] * h
        for j in 0:(panels-1), b in 1:5
            x2 = (j + 0.5 * (1 + ξ[b])) * h
            weight = wx * 0.5 * w[b] * h
            g = field_gradient(solution, model, (x1, x2))
            e = exact_gradient((x1, x2))
            err += weight * ((g[1] - e[1])^2 + (g[2] - e[2])^2)
            ref += weight * (e[1]^2 + e[2]^2)
        end
    end
    return sqrt(err / ref)
end

# ── The run ───────────────────────────────────────────────────────────────────

V = ladder(omega; cells=BASE_CELLS, order=1, depth=DEPTH)
model = prepare(problem(V))
solution = solve!(model)

previous = nothing

println("  columns: cycle, unknowns, rel_energy, eta, consistency, n_h, n_p, ",
        "max order per level")
for cycle in 1:CYCLES
    estimate_ = estimate(model, solution)
    # The ruler is independent of the space, so it can be read every cycle — but
    # it is also the most expensive thing in the loop (see `TANH_TABLE_ENERGY`
    # in the header). With the column off the run still prints the closing
    # headline, which is the number this benchmark is about.
    energy = TABLE_ENERGY ? lattice_energy_error(solution, model) : nothing
    orders = [maximum(first, cell_orders(model.prefold_space; level=k))
              for k in 1:length(model.prefold_space.levels)]

    # `refine(V, est; theta, previous)` is exactly these two calls followed by
    # the application step. They are made separately here so the h/p decision is
    # visible in the table rather than inferred from the mesh afterwards.
    marked = mark_cells(estimate_; theta=THETA)
    h_cells, p_cells = decide(model.prefold_space, estimate_, marked; pmax=PMAX, previous=previous)

    println("  ", lpad(cycle, 3), ", ", lpad(active_unknowns(model), 7), ", ",
            rpad(energy === nothing ? "-" : sig(energy), 10), ", ", rpad(sig(estimate_.total), 10),
            ", ", rpad(sig(estimate_.consistency, 3), 9), ", ", lpad(length(h_cells), 4), ", ",
            lpad(length(p_cells), 4), ", ", orders)

    global previous = (model.prefold_space, estimate_)
    cycle == CYCLES && break
    refined = refine(model.prefold_space; h=h_cells, p=p_cells, pmax=PMAX)
    global model = adapted(model, refined)
    global solution = solve!(model)
end

report = diagnostics(model, solution; exact=exact)

out = nothing
if WRITE_OUTPUT
    out = joinpath(@__DIR__, "output", "adaptive_tanh_layer_2d")
    write_vtk(out, solution, model;
              point_data=(uh=(u, c, x, xi) -> u(c, xi), exact=(u, c, x, xi) -> exact(x),
                          error=(u, c, x, xi) -> u(c, xi) - exact(x)))
end

println()
println("relative energy error (fixed 96² lattice): ",
        sig(lattice_energy_error(solution, model), 6))

print_run_report("2D adaptive hp on a curved interior layer", report;
                 parameters=(:lambda => LAMBDA, :cycles => CYCLES, :base_cells => BASE_CELLS,
                             :depth => DEPTH, :theta => THETA, :pmax => PMAX), output=out)
