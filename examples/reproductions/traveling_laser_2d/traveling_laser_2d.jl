#=
Traveling Laser: Automated hp Adaptivity On A Moving Feature
============================================================

A small Gaussian heat source orbits a circle inside a plate, leaving a
comet-tail of hot material behind it. The discretisation follows the source
automatically: the same three verbs the steady benchmarks use, plus the one a
transient needs.

    ∂ₜT − κ ΔT = q(x, t)   in Ω = (−6, 6)²,   T = 0 on ∂Ω,   T(·, 0) = 0,

    q(x, t) = P/(2πσ²) · exp( −‖x − p(t)‖² / (2σ²) ),   p(t) on the circle r = 5/2,

with κ = 1/20, σ = 1/10, P = 1, one revolution over t ∈ [0, 4]. The domain is
large enough that T is zero on ∂Ω to machine precision, so the homogeneous
Dirichlet condition reproduces an infinite plate.

**The reference is exact.** The infinite-plate solution is the Duhamel
convolution of the Gaussian source with the heat kernel. Two Gaussians convolve
to a Gaussian of summed variance, so the double integral collapses to a single
history integral,

    T(x, t) = ∫₀ᵗ P/(2π(σ² + 2κ(t−s))) · exp( −‖x − p(s)‖²/(2(σ² + 2κ(t−s))) ) ds,

evaluated by composite Simpson at a step resolving the moving peak (width ≈
σ/‖ṗ‖). There is no manufactured source and no discretisation in the reference.

**The step is one declared form, and that is what makes the loop possible.**
The θ-method system for Tⁿ⁺¹ is

    [ (1/Δt) M + θκ K ] Tⁿ⁺¹ = (1/Δt) M Tⁿ − (1−θ)κ K Tⁿ + θ qⁿ⁺¹ + (1−θ) qⁿ,

which is an *elliptic* problem, so it is declared as a single `WeakForm` rather
than assembled by hand: the bilinear part carries `(1/Δt)` on the value channel
and `θκ` on the gradient channel, and the linear part carries the history on
BOTH channels — `Tⁿ/Δt + q_θ` against `v`, and `−(1−θ)κ ∇Tⁿ` against `∇v`.

That last point is the one to keep. Putting the history in the *declared load*
rather than adding it to the assembled vector is what lets [`estimate`](@ref)
work at all here: the estimator re-assembles the load on its own order-elevated
space, so it sees the exact residual of the step actually being solved. A
hand-assembled right-hand side would leave it estimating a different problem.
It is a requirement of the estimator, not a matter of taste, so the load must
stay declared if this example is refactored.

**Seeding is physics, not tuning.** The Bank–Weiser indicator needs the source
to be resolved before it means anything: σ = 1/10 against a base cell of 3 is a
spike thirty times smaller than a cell, and the base and order-elevated
quadratures then disagree wildly. Measured, the estimator's own consistency flag
reads 12.4 on the unrefined base mesh, 0.060 once the finest cell is 0.375, and
0.0039 at 0.094 — reliable only in the last of those. So the cells under the
source are activated geometrically from the known path. That is exact a priori
knowledge, not a fitted constant; everything behind the source — the trail,
which is the part nobody knows in advance — is found by the indicator.

**What this replaces, and why it is not a score.** An earlier, unpublished
iteration of this study shaped the trail by hand, with fitted constants per
overlay level: threshold fractions on T and on ∇T relative to their running
maxima, an activation radius, and a dilation count. Tuning them reached **2510
active unknowns at 0.57 %** relative L². That pair is quoted from that
iteration, is **not reproducible from this repository** — the script it came
from is not in the history — and its time discretisation is unrecorded. The
last of those is what stops it being a comparison, in either direction.

This script's own headline used to be 0.452 % at a step of 1/40, and that
number measured the θ-method rather than the mesh. Measured on the published
configuration at that step: taking the overlay depth from 5 to 6 — halving the
finest cell to 0.047 and lifting the time-mean unknown count from 7027 to
13718 — leaves the settled error at 0.4520 %, which is where depth 5 already
puts it, unchanged to four significant figures. Thirding the step to 1/120
instead drops it to 0.05246 %, a factor 8.62 against the 9 a second-order
method predicts. At least 88 % of the old headline was Crank–Nicolson. If the
baseline's 0.57 % carries a comparable time fraction, its spatial error is
below every point measured here; if it carries none, it is above most of them.
The sign of that comparison is unknown, not merely its size, and two numbers —
one a time-error floor, the other of unrecorded time content — do not subtract.
Read 2510 / 0.57 % as the target that motivated this study, not as a score
either scheme wins, and read `TL_DT` below as the reason everything after this
paragraph is about the mesh.

**Where the two defaults come from.** There are no per-level constants here:
the whole adjustable surface is the Dörfler fraction `TL_THETA_MARK` and the
release fraction `TL_RELEASE`. They are not derived either. They were chosen
once, by the sweep below, scored against the exact solution — an oracle no real
problem hands you — so "this script has no fitted constants" would be a claim
it cannot support. What the sweep found is what makes the pair defensible: not
a point, but a ridge. Time-mean active unknowns and time-mean relative L² over
the second half of the revolution, at the published configuration and the
default step:

                    release 0.01     release 0.02     release 0.03
    marking 0.3     4931 / 0.210 %   3696 / 0.870 %   4070 / 0.234 %
    marking 0.4     5394 / 0.149 %   4589 / 0.161 %   4244 / 0.359 %
    marking 0.5     5881 / 0.0696 %  4851 / 0.0742 %  4292 / 0.0915 %
    marking 0.6     6658 / 0.0594 %  5001 / 0.161 %   4469 / 0.171 %

A marking fraction of 0.5 is the best of the four at release 0.02 and at 0.03,
and second at 0.01, where 0.6 buys 1.17× less error for 13 % more cost. It
holds three of the six points on the Pareto front — the whole middle of it —
and all six dominated settings carry a marking fraction other than 0.5. Down
that column the release fraction moves the mean error by a factor 1.31, against
4.15, 2.41 and 2.88 down the other three, so inside it the choice of release is
close to a coin flip weighted by cost. That insensitivity is the property being
leaned on, and it is a robustness statement rather than a fit: what carries to
another problem is "mark about half of the estimated error", not two calibrated
decimals. The contrast with the scheme this replaces is therefore ten constants
fitted per overlay level against two global ones — not ten against none.

At the defaults the run settles at **5460 unknowns at 0.0609 %** and averages
**4851 unknowns at 0.0742 %** over the second half. The second pair is the one
to read: the active count is a sawtooth spanning 3770 to 6094 there, and the
settled value is one sample of it.

The two fractions are not interchangeable, and the sweep says which one decides:

  - *the marking fraction does.* Marking 0.4 instead of 0.6 at a fixed release
    of 0.01 saves 19 % of the mean cost and pays 2.51× the mean error for it.
    On the endpoint alone the same pair reads 8 % cheaper for 1.51× the error,
    which is one reason the mean is the metric here — and at the old step it
    read as free, because the time-error floor hid the difference entirely.
  - *releasing more eagerly still buys cost back, and the price is now
    legible.* `TL_RELEASE = 0.05` reaches 3378 mean unknowns at 1.45 % and
    `0.08` reaches 2694 at 2.49 %, but neither settles: they end the run at
    2.89× and 5.68× their own best error after t = 1, with the error swinging
    by factors of 4.96 and 11.06 across the second half. Eleven of the twelve
    settings above end within 1.07× of their own best; the twelfth, marking 0.6
    at release 0.03, ends at 2.85×.

**What this reproduction can answer.** Not whether automated hp is competitive
with hand-tuned overlays: the baseline's time content is unrecorded, so that
comparison is unsigned rather than merely uncontrolled. Driving the loop down
to the baseline's cost does not settle it either — marking 0.6 at release 0.05
reaches 3308 mean unknowns at 0.569 %, the baseline's quoted error at 1.3× its
unknown count, but ends 5.86× above its own best. What the sweep can answer is
what an untuned loop costs. Two fractions with no per-level fitting span a
factor 1.80 in mean cost and 14.6 in mean error, so declining to tune is not
free. Ranked by the loop's own indicator η — its mean over the second half,
which needs no exact solution — the twelve settings come out very nearly in the
order the exact error puts them, with Spearman ρ = 0.93 and the best four in
identical order, so the operating point is reachable from what the method has
on a real problem even though an oracle is what found it. And the cheap end of
the range is not a cheaper answer but an unconverged one.

Environment knobs, so the run can be shortened or re-pointed without editing the
script. The default is in brackets:

  - `TL_T_MAX` (4) — end time. One revolution; the revolution period itself is
    a constant below and does not move with it.
  - `TL_DT` (1/120) — the largest step the θ-method may take. The step is also
    clamped onto every mesh-update boundary, so the step actually taken is
    `min(TL_DT, TL_UPDATE)`, and that — not `TL_DT` — is what the report prints.
    At the default `TL_DT` binds, three steps to a mesh update. Raise it above
    `TL_UPDATE` and the cadence becomes the step again, which is the regime the
    Crank–Nicolson floor above was measured in.
  - `TL_THETA` (1/2) — the θ of the θ-method. 1/2 is Crank–Nicolson.
  - `TL_KAPPA` (1/20) and `TL_SIGMA` (1/10) — diffusivity and source width.
  - `TL_DEPTH` (5) — overlay levels under the 4×4 base.
  - `TL_THETA_MARK` (0.5) — the Dörfler fraction the marking uses.
  - `TL_RELEASE` (0.02) — release a parent whose live cover is everywhere below
    this fraction of the largest indicator.
  - `TL_UPDATE` (1/40) — how often the mesh is adapted.
  - `TL_WRITE_OUTPUT` (true) — write the ParaView series and the CSV.
  - `TL_VTK_FRAMES` (80) — roughly how many frames follow the initial
    condition. It sets a cadence over steps rather than a second time grid, so
    it cannot move a step; the cadence is a whole number of them, so the count
    is exact only where it divides the step count, as the default does.
=#

using Unfitted

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

# ── Configuration ─────────────────────────────────────────────────────────────

const T_MAX = env("TL_T_MAX", 4.0)
const DT_MAX = env("TL_DT", 1 / 120)
const THETA = env("TL_THETA", 0.5)                   # 1/2 = Crank–Nicolson
const KAPPA = env("TL_KAPPA", 0.05)
const SIGMA = env("TL_SIGMA", 0.1)
const DEPTH = env("TL_DEPTH", 5)
const THETA_MARK = env("TL_THETA_MARK", 0.5)         # Dörfler
const RELEASE = env("TL_RELEASE", 0.02)              # release below this × max η
const UPDATE_EVERY = env("TL_UPDATE", 1 / 40)        # mesh update interval
const WRITE_OUTPUT = env("TL_WRITE_OUTPUT", true)
const VTK_FRAMES = env("TL_VTK_FRAMES", 80)

# The θ-method's step is clamped onto every mesh-update boundary as well as by
# `TL_DT`, so `STEP` — not `TL_DT` — is the largest step the run actually takes.
# The two coincide only while `TL_DT` binds below the adaptation cadence, which
# at the defaults it does; the report prints `STEP`, so the parameter block
# stays true for a `TL_DT` above the cadence as well.
const STEP = min(DT_MAX, UPDATE_EVERY)

# ── Constants ─────────────────────────────────────────────────────────────────
#
# `T_REVOLUTION` is named rather than folded into `PHI_DOT` because `2π / 4.0`
# reads like `2π / T_MAX` and is not: the orbit is a property of the problem,
# so a shortened run has to stay a genuine prefix of the published one. Tidying
# it to `T_MAX` would silently change the physics of every shortened run.
#
# `COLUMNS` is the single source for the printed table header, the CSV header
# and the field names of a recorded row, so those three cannot drift apart.
const OMEGA = box((-6.0, -6.0), (6.0, 6.0))
const BASE_CELLS = 4
const P_BASE = 4
const PMAX = 6
const P_LASER = 1.0
const R_PATH = 2.5
const PHI_0 = -pi / 2
const T_REVOLUTION = 4.0
const PHI_DOT = 2pi / T_REVOLUTION
const SEED_RADIUS = 4 * SIGMA                        # a priori: the source support
const COLUMNS = (:time, :unknowns, :rel_l2_error, :eta, :consistency, :n_h, :n_p, :n_released)

# ── The source and the exact infinite-plate solution ──────────────────────────
#
# One Gaussian, written once and read three times: as the instantaneous source
# at variance σ², as the retarded kernel of the history integral at variance
# σ² + 2κ(t−s), and — through `distance` — as the ball the seeding and the
# release rule measure against.
#
# `@inline` is load-bearing on the four the history integral runs through, and
# not decoration. Splitting the kernel into named pieces puts an `exp` behind
# two more call frames, which is enough for the cost heuristic to decline to
# inline any of them, and the reference integral is the innermost loop in the
# script: measured over 4000 evaluation points at t = 4, 107.4 ms without the
# annotations against 67.1 ms with them, for bit-identical values.
@inline sqdistance(p, c) = (p[1] - c[1])^2 + (p[2] - c[2])^2
distance(p, c) = sqrt(sqdistance(p, c))
@inline gaussian(x, c, var) = (P_LASER / (2pi * var)) * exp(-sqdistance(x, c) / (2 * var))

@inline source_center(t) = (R_PATH * cos(PHI_0 + t * PHI_DOT), R_PATH * sin(PHI_0 + t * PHI_DOT))
function heat_source(t)
    c = source_center(t)
    return x -> gaussian(x, c, SIGMA^2)
end

# The Duhamel integrand at history time `s`: the same Gaussian, centred where
# the source was at `s` and spread by the heat kernel it has diffused through
# since. Two Gaussians convolve to a Gaussian of summed variance, which is what
# collapses the double integral to this single history integral.
@inline retarded(x, t, s) = gaussian(x, source_center(s), SIGMA^2 + 2 * KAPPA * (t - s))

# Composite Simpson over [0, t] on an even number of intervals sized to resolve
# the moving peak. The source traverses its own width σ in σ/‖ṗ‖ and the rule
# puts ten intervals in that time, so the node count grows with t: at t = 4 it
# is n = 2⌈t/(0.2σ/‖ṗ‖)⌉ = 1572 intervals, 1573 kernel evaluations for a single
# evaluation point. That is why the reference is read per update in the table
# rather than sampled into the frames.
function exact_temperature(t)
    t <= 0 && return x -> 0.0
    n = max(2, 2 * ceil(Int, t / (0.2 * SIGMA / (R_PATH * PHI_DOT))))
    h = t / n
    return function (x)
        acc = retarded(x, t, 0.0) + retarded(x, t, t)
        for i in 1:(n-1)
            acc += (isodd(i) ? 4.0 : 2.0) * retarded(x, t, i * h)
        end
        return acc * h / 3
    end
end

# ── The θ-step as one declared form ───────────────────────────────────────────
#
# `previous` is the `(solution, model)` pair of the step just taken, or
# `nothing` at t = 0. Both load channels are populated, so Crank–Nicolson is
# declared rather than hand-assembled and `estimate` re-derives it on the
# enriched space by itself.

# The declared load at one quadrature point: the θ-weighted source `f` on the
# value channel, plus the history Tⁿ on both channels. Two methods rather than a
# branch inside one, because the history arrives type-erased (see below):
# dispatching on it once per point is a function barrier, and the arithmetic
# behind it is concrete again. At t = 0 there is no history, and the bare scalar
# is the documented shorthand for "value channel only, zero gradient channel".
step_load(::Nothing, x, dt, f) = f
function step_load(p, x, dt, f)
    return TestChannels(value(p..., x) / dt + f, (-(1 - THETA) * KAPPA) .* field_gradient(p..., x))
end

# The callback reaches `previous` through a `Ref{Any}`, which is a type eraser
# and not mutable state: it is written once, at construction, and only read
# afterwards, so the load `estimate` re-assembles is the one the solve used.
# Closing over the pair directly would instead put modelⁿ's concrete type inside
# modelⁿ⁺¹'s, and that type name doubles per step — measured here at 730, 2244,
# 5272, 11328, 23440, 47664 and 96112 characters over seven successive steps,
# with the inference bill following it: an eight-step run on a depth-3 ladder
# took 15.3 s that way against 11.4 s this way, at −O0 and at the pre-retune
# defaults. The update cycle cuts the chain by rebuilding from `nothing`, so
# what compounds within one cycle is `TL_UPDATE / TL_DT` steps and no more —
# three of them at the defaults, where `TL_DT` now binds below the cadence. The
# erasure is therefore load-bearing at the shipped settings and not only for a
# knob a reader might turn.
function step_model(V, previous, dt, t0, t1)
    q0, q1 = heat_source(t0), heat_source(t1)
    history = Ref{Any}(previous)
    form = WeakForm(; symmetric=true,
                    bilinear=(q, trial) -> TestChannels(trial.value / dt,
                                                        (THETA * KAPPA) .* trial.gradient),
                    linear=q -> step_load(history[], q.x, dt,
                                          THETA * q1(q.x) + (1 - THETA) * q0(q.x)))
    T = field(:temperature, V)
    return prepare(Problem(T, form; dirichlet=[dirichlet(0.0; on=boundary(:all), field=T)]))
end

# ── Seeding: the cells under the source, from the known path ──────────────────
#
# Not an indicator and not tuned. The source position is given, its support is
# 4σ, and a cell inside that support must be live before any indicator computed
# on this mesh means anything.
#
# The masks are read off `V` before any of them is replaced, so the predicates
# form the union of what was live with what the ball covers rather than a
# running union — and every overlay level is named in one `adapt`, which is one
# integration-plan and dof-layout rebuild instead of one per level. The base
# level is never seeded: it is live everywhere already.
function seed(V, t)
    c = source_center(t)
    live = [active_cells(V; level=k) for k in 1:length(V.levels)]
    seeded(k) = (b, ci) -> live[k][ci] || distance(center(b), c) <= SEED_RADIUS
    return adapt(V, (k => seeded(k) for k in 2:length(V.levels))...)
end

# ── Release: the cooled tail ──────────────────────────────────────────────────
#
# `coarsen` takes explicit marks, so the rule lives here rather than in the
# package. A parent is released when every cell of its cover is cold — below
# `RELEASE` of the largest indicator — and the source is not sitting on it.
#
# `V` must be the space `est` was taken on. An `ErrorEstimate` is indexed by the
# cells of that space and is zero wherever it carried no approximation, so a cell
# that became live afterwards reads as cold here no matter how much error sits on
# it. That is why the caller marks before it refines.
#
# The cover is read upward and set-wise: `overlapping_cells` answers geometric
# overlap, which is symmetric, so a parent overlapping a live child is exactly a
# parent the live set covers. Two calls per level — one for the live cover, one
# for the live-and-hot cover — therefore answer both clauses for every parent at
# once, where asking each parent what it covers costs a whole destination mask
# per parent. The public verb's own docstring works this direction; per mark it
# is the wrong tool and per level it is the right one.
function release_marks(V, est, t)
    c = source_center(t)
    marks = Tuple{Int,CartesianIndex{2}}[]
    peak = maximum(maximum, est.cells)
    peak > 0 || return marks
    for k in (length(V.levels)-1):-1:1
        live = active_cells(V; level=k + 1)
        any(live) || continue
        covered = overlapping_cells(V, live; from=k + 1, to=k)
        warm = overlapping_cells(V, live .& (est.cells[k+1] .> RELEASE * peak); from=k + 1, to=k)
        for ci in cell_indices(V; level=k)
            (covered[ci] && !warm[ci]) || continue
            distance(center(cell_box(V, ci; level=k)), c) > 2 * SEED_RADIUS && push!(marks, (k, ci))
        end
    end
    return marks
end

# ── Frames ────────────────────────────────────────────────────────────────────
#
# Each frame is a full bundle: the FE temperature and the instantaneous source,
# plus one wireframe per overlay level, so the mesh following the laser is
# visible alongside the field it is following. The exact solution is deliberately
# not sampled into them — see `exact_temperature` for what one evaluation costs —
# and the error against it is reported per update in the table and the CSV.
#
# The schedule is a cadence over *steps*, so output policy is not an input to
# the discretisation: asking for a different number of pictures cannot move a
# step, which is exactly what it used to do. The step count comes from the same
# three constants the clamp does, so the default 80 frames over the published
# run's 480 steps is every sixth step, and a count the steps do not divide
# rounds to the nearest whole cadence rather than bending the time grid to fit.
const TOTAL_STEPS = ceil(Int, T_MAX / UPDATE_EVERY) * ceil(Int, UPDATE_EVERY / DT_MAX)
const FRAME_EVERY = max(1, round(Int, TOTAL_STEPS / VTK_FRAMES))

function write_frame(series, t, u, model)
    series === nothing && return nothing
    q = heat_source(t)
    return write_vtk(series, t, u, model; compress=true, subdivisions=20,
                     point_data=(temperature=(f, c, x, xi) -> f(c, xi),
                                 source=(f, c, x, xi) -> q(x)))
end

# ── The run ───────────────────────────────────────────────────────────────────
#
# One θ-step per iteration, and every `UPDATE_EVERY` an adaptation cycle in this
# order, which is the part that is easy to get wrong:
#
#   1. estimate on the model just solved, before anything is refined;
#   2. mark (`refine(V, est; …)` is these two calls plus the application step;
#      they are separate because the release rule needs `h_cells`);
#   3. decide h against p from the previous cycle's (space, estimate);
#   4. release, computed on the estimate's own space and filtered against
#      `h_cells` — a parent that also takes an h-step this cycle is dropped,
#      because `coarsen` would deactivate the cover `refine` has just woken.
#      The hazard is large: evaluated on the refined space, the release rule
#      hits 361 of the 510 h-steps the full study takes, 71 % of them. The two
#      guards turn out to be redundant, and both are kept deliberately —
#      measured on the shortened run, moving the rule to the refined space with
#      the filter in place is bit-identical to the shipped loop, and so is
#      dropping the filter with the rule left on the estimate's own space.
#      Dropping both settles at 1008 unknowns and 1.783 % against 1156 and
#      1.914 %;
#   5. refine, then coarsen, then seed the next interval's source cells;
#   6. record and print the error of the step, on the old model at the old time;
#   7. rebuild on the new space and carry the state across by L² projection.
#
# Step 7's rebuild passes `nothing` for the history — the type-chain cut
# `step_model` documents — and the projected solution is bound back into
# `history` there, because the next step reads its state through that pair.
function run_study(series)
    V = seed(ladder(OMEGA; cells=BASE_CELLS, order=P_BASE, depth=DEPTH, mode=:trunk), 0.0)
    model = step_model(V, nothing, DT_MAX, 0.0, DT_MAX)
    # Frame 0 is the initial condition, which is exactly zero.
    u = solution(model, zeros(active_unknowns(model)))
    write_frame(series, 0.0, u, model)

    rows, history, previous = NamedTuple[], nothing, nothing
    t, next_update, step = 0.0, UPDATE_EVERY, 0
    println("  columns: ", join(COLUMNS, ", "))
    while t < T_MAX - 1.0e-12
        dt = min(DT_MAX, T_MAX - t, next_update - t)
        model = step_model(V, history, dt, t, t + dt)
        u = solve!(model)
        t, step = t + dt, step + 1
        history = (u, model)
        step % FRAME_EVERY == 0 && write_frame(series, t, u, model)
        (t >= next_update - 1.0e-12 && t < T_MAX - 1.0e-12) || continue

        est = estimate(model, u)
        marked = mark_cells(est; theta=THETA_MARK)
        h_cells, p_cells = decide(V, est, marked; pmax=PMAX, previous=previous)
        released = filter(mark -> mark ∉ h_cells, release_marks(V, est, t + UPDATE_EVERY))
        W = refine(V; h=h_cells, p=p_cells, pmax=PMAX)
        isempty(released) || (W = coarsen(W; h=released))
        W = seed(W, t + UPDATE_EVERY)

        row = NamedTuple{COLUMNS}((t, active_unknowns(model),
                                   l2_error(u, model, exact_temperature(t)), est.total,
                                   est.consistency, length(h_cells), length(p_cells),
                                   length(released)))
        push!(rows, row)
        println("  ", rpad(sig(row.time, 3), 6), ", ", lpad(row.unknowns, 6), ", ",
                rpad(sig(row.rel_l2_error), 10), ", ", rpad(sig(row.eta), 10), ", ",
                rpad(sig(row.consistency, 3), 9), ", ", lpad(row.n_h, 4), ", ", lpad(row.n_p, 4),
                ", ", lpad(row.n_released, 4))

        new_model = step_model(W, nothing, DT_MAX, t, t + DT_MAX)
        u = transfer(u, model, new_model; via=L2Projection())
        previous, V, model = (V, est), W, new_model
        history = (u, model)
        next_update += UPDATE_EVERY
    end
    return (; model, u, rows)
end

# ── The closing pair, and why there are two of them ───────────────────────────
#
# Neither the active-unknown count nor the reported error settles. Both are
# sawtooths at the adaptation cadence: the mesh grows while the source moves
# into ground the trail has not reached, and drops when the cooled trail behind
# it is released; the error follows the mesh. The value at t = T_MAX is one
# sample of that oscillation and not a converged state, and which sample it is
# depends on where the last release happened to fall. Measured over the twelve
# (θ_mark, release) settings tabulated in the header, the endpoint understates
# the time-mean error at eleven of them and overstates the mean cost at ten, so
# quoting it alone flatters both axes at once. It is kept because it is the one
# number a reader can check against `diagnostics`, and the mean is printed
# beside it because it is the one that describes the run.
#
# The mean is taken over the second half, which at the published configuration
# is the second half of one revolution: the first half still carries the
# transient of a trail that does not yet exist. Rows are recorded one per
# adaptation cycle at a fixed cadence, so their arithmetic mean is the time
# mean, and the min–max band beside it is the swing that the mean hides. The
# cutoff carries the same 1.0e-12 slack the stepping loop uses, because `t` is
# accumulated by addition and the cycle that lands on T_MAX/2 lands a few ulp
# below it; without the slack that cycle drops out and the mean moves in the
# third digit.
function print_second_half(rows)
    half = filter(row -> row.time >= T_MAX / 2 - 1.0e-12, rows)
    isempty(half) && return nothing
    u, e = [row.unknowns for row in half], [row.rel_l2_error for row in half]
    println("mean over t ≥ ", sig(T_MAX / 2, 3), ": ", round(Int, sum(u) / length(u)),
            " unknowns (", minimum(u), "–", maximum(u), ") at relative L2 ",
            sig(sum(e) / length(e), 4), " (", sig(minimum(e), 4), "–", sig(maximum(e), 4), ")")
    return nothing
end

# ── Report and output ─────────────────────────────────────────────────────────

series = WRITE_OUTPUT ? vtk_series(joinpath(@__DIR__, "output", "traveling_laser_2d")) : nothing
model, u, rows = run_study(series)
report = diagnostics(model, u; exact=exact_temperature(T_MAX))

println("\nsettled: ", active_unknowns(model), " unknowns at relative L2 ", sig(report.l2_error, 4))
print_second_half(rows)

if WRITE_OUTPUT
    open(joinpath(@__DIR__, "output", "history.csv"), "w") do io
        println(io, join(COLUMNS, ","))
        foreach(row -> println(io, join((row[c] for c in COLUMNS), ",")), rows)
    end
    println("VTK series: ", close(series), "  (", series.count[], " frames)")
end

print_run_report("2D traveling laser, automated hp", report;
                 parameters=(:t_max => T_MAX, :dt => STEP, :update => UPDATE_EVERY, :theta => THETA,
                             :kappa => KAPPA, :sigma => SIGMA, :depth => DEPTH,
                             :theta_mark => THETA_MARK, :release => RELEASE), output=nothing)
