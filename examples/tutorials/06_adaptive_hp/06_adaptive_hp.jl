#=
Tutorial 6 — Adaptive hp: letting the solver choose where to refine
====================================================================

**What you will learn.** Every tutorial so far has placed refinement by hand:
Tutorial 2 put one overlay where the peak is, Tutorial 5 declared a ladder and
switched its cells on with a depth map you wrote. This one hands that decision to
the solver. Two verbs do it — `estimate` says where the error is, `refine` spends
unknowns there — and the loop around them is eight lines.

**What you should already know.** Tutorial 5: `ladder`, `adapt`, and what a
covered cell contributes (nothing — it is the parent of a leaf).

**The problem.** A solution with two features that want opposite treatments:

    u(x) = |x − a|^α  +  exp(−|x − c|² / 2σ²),   α = 2.5, a = (0.3, 0.3),
                                                 σ = 0.08, c = (0.7, 0.7)

The first term is not analytic at `a`: raising the order there buys an algebraic
rate where a smooth solution would give an exponential one, so past some order
the cheaper purchase is smaller cells. The second is analytic but sharp, so it is
cheap in order and expensive in cells. A method that only refines h wastes
unknowns on the bump; one that only raises p pays the algebraic rate at the
corner forever.

**How the loop chooses h or p.** `refine` does not read a smoothness indicator,
and it does not give a marked cell both steps: it gives it exactly one, chosen by
asking a cheaper question than smoothness. Every cycle predicts what a marked
cell's indicator ought to become if the solution there is as smooth as the step
just taken assumed — Melenk & Wohlmuth's predicted error reduction, in the form
deal.II's `hp::Refinement::predict_error` ships — and the next cycle compares.
A cell that met its prediction takes p again; one that fell short takes h.

The comparison needs the cycle before it, which is exactly what `previous`
carries: the space that was refined, and the estimate that was read on it. On the
first cycle there is no history and every marked cell takes p. Omit `previous`
and *every* cycle is a first cycle — the loop degenerates to pure p and the
tutorial silently stops being about hp at all.
=#

using Unfitted

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

const Ω = box((0.0, 0.0), (1.0, 1.0))
const ALPHA, SING = 2.5, (0.3, 0.3)
const WIDTH, BUMP = 0.08, (0.7, 0.7)

radius(x) = sqrt((x[1] - SING[1])^2 + (x[2] - SING[2])^2)
bump_sq(x) = (x[1] - BUMP[1])^2 + (x[2] - BUMP[2])^2
bump(x) = exp(-bump_sq(x) / (2 * WIDTH^2))
exact(x) = radius(x)^ALPHA + bump(x)

# −Δu, from the analytic Hessian of both terms.
function source(x)
    r = radius(x)
    singular = ALPHA^2 * r^(ALPHA - 2)
    smooth = (bump_sq(x) / WIDTH^4 - 2 / WIDTH^2) * bump(x)
    return -(singular + smooth)
end

problem(V) = poisson(V; source=source, dirichlet=[dirichlet(exact; on=boundary(:all))])

#=
── The loop ─────────────────────────────────────────────────────────────────

A `ladder` declares the levels the loop may use. It arrives inert — every
overlay is declared with no active cell — so it costs the base level's unknowns
until the loop switches cells on. `depth` is the budget: the loop can refine a
region four times and no further. It is a ceiling, not a target; this problem
reaches its tolerance having spent one level of it (see "What to try").

`estimate` returns a per-cell indicator plus the three scalars a stopping test
needs. The test below is relative — `total` against `reference`, the energy of
the active unknowns in the problem's own form — so it means the same thing for an
operator that is not the Laplacian, and it needs no knowledge of the exact
solution. It compares like with like on one problem; `reference` does not include
the energy of a nonzero Dirichlet lift, so it is not a quantity to carry across
problems with different boundary data.

`refine` marks by Dörfler on that indicator and applies one step to each marked
cell. `adapted` prepares the result in one rebuild rather than two.
=#

V = ladder(Ω; cells=8, order=3, depth=4)
model = prepare(problem(V))
u = solve!(model)

# The evidence one cycle leaves the next. `refine` reads the h-versus-p decision
# out of it, so it has to be the space and estimate of the cycle just finished —
# hence the assignment before `model` is replaced.
previous = nothing

# Plain `println` formatting, as the other example scripts do: an example's
# environment carries `Unfitted` and nothing else.
sig(x, n=4) = string(round(x; sigdigits=n))

println("step  unknowns   η           η/‖u‖_a     consistency  L² error")
for step in 1:20
    est = estimate(model, u)
    println("  ", lpad(step, 2), "  ", lpad(active_unknowns(model), 7), "  ",
            rpad(sig(est.total), 11), " ", rpad(sig(est.total / est.reference), 11), " ",
            rpad(sig(est.consistency), 11), "  ", sig(l2_error(u, model, exact)))
    est.total <= 1.0e-4 * est.reference && (println("  ── tolerance met ──"); break)

    refined = refine(model.prefold_space, est; theta=0.5, previous=previous)
    refined === model.prefold_space && (println("  ── ladder exhausted ──"); break)
    global previous = (model.prefold_space, est)
    global model = adapted(model, refined)
    global u = solve!(model)
end

#=
── Reading the output ───────────────────────────────────────────────────────

The run takes sixteen cycles to reach the tolerance, and it is worth reading as
two halves, because the interesting thing about it is that the second half is
not a continuation of the first.

Cycles 1–11 are pure p: three to eleven base cells per cycle, no h at all. The
prediction is being met everywhere the marking reaches, so the rule keeps buying
degrees, and it works — by cycle 10 η has fallen from 1.269e-1 to 1.020e-3 and
the L² error from 2.796e-3 to 3.645e-5. Then the p-steps stop paying. Between cycles
10 and 14 the L² error sits at 3.6–5.0e-5 and drifts the *wrong* way while
η/‖u‖_a falls a further 2.8× and `consistency` climbs from 1.4e-1 to 5.5e-1. The
estimate has begun to under-read, and the consistency flag is what says so.

The h-steps are what end it. The first lands at cycle 12, on four base cells out
on the bump's skirt; cycle 13 takes one more just past the bump's centre; and
cycle 15 marks three, among them `(3, 3)` — the cell holding the singularity,
reached for the first time in the run. The L² error then falls from 1.746e-5 to
6.817e-6 and `consistency` drops back to 6.6e-2: the estimate becomes
trustworthy again at the same moment the error moves. Eight h-marks in the whole
run against seventy-three p-marks, waking 32 of the first overlay's 256 cells.

Three columns are worth watching, and only one of them is the error.

`η/‖u‖_a` is the stopping quantity. It is scale free, so the tolerance does not
have to be rescaled when the solution is, and unlike the L² error it does not
require an exact solution — which is the whole point of an estimator. It is a
comparison within one problem, not across two with different boundary data, for
the reason given above.

Note that η itself is not monotone: it rises at cycle 13, from 5.302e-4 to
9.316e-4, because an h-step wakes cells that carried no approximation before and
therefore no indicator. A jump in η after an h-step is the estimator seeing
error that was previously invisible to it, not error being created.

`consistency` is the reliability flag, and it is free: Galerkin orthogonality
makes the residual vanish on the current space in exact arithmetic, so what is
left is data oscillation and the solve. It is not a saturation check — read
[`ErrorEstimate`](@ref) for what it can and cannot tell you — but a climb means
the load is under-integrated where η is smallest, and on this run the climb and
the stall in the L² error arrive together and leave together.

The L² column is here because this tutorial knows the exact solution. A real run
does not, and does not need to.

── What to try ──────────────────────────────────────────────────────────────

Set `theta = 0.9` and watch the loop take bigger steps and fewer of them; set it
to `0.2` for the opposite. The scheme was chosen so this knob does not need
refitting per problem — over a factor-three sweep the unknown count moves by at
most 1.46× — which is why it is the only constant in the loop.

Replace `ladder(...; depth = 4)` with `depth = 1` and nothing changes: the run
is identical row for row and settles at the same 1943 unknowns, because only the
first overlay is ever used. `depth` is headroom, not a plan. The "ladder
exhausted" branch — `refine` returns the space it was handed when no marked cell
has a step left — is the guard for the run where the headroom does run out, and
on this problem the tolerance is met long before it can fire.

Hand `refine` your own marked set — any iterable of `(level, cell)` pairs — to
drive the same application step from an indicator of your own.
=#

est = estimate(model, u)
print_run_report("Tutorial 6 — adaptive hp, final state", diagnostics(model, u; exact=exact);
                 parameters=(:base_cells => 8, :depth => 4, :theta => 0.5,
                             :eta_over_reference => est.total / est.reference))
