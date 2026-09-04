#=
Tutorial 2 — Overlays: Local Refinement by Superposition
========================================================

**What you will learn.** The idea the package exists for. Instead of
splitting cells and stitching the pieces together with hanging-node
constraints, you lay an independent finer mesh *on top of* a region of
interest and add its contribution to the coarse one. You will build
such an overlay, watch the error fall, switch individual overlay cells
on and off, and read the diagnostic that reports how many redundant
modes the coarse level shed underneath.

**What you should already know.** Tutorial 1 — the `box → space →
field → poisson → prepare → solve! → diagnostics` workflow is used
here without further comment.

**The superposition model.** The discrete solution is a sum over mesh
levels,

    u_h(x) = Σₖ u_h⁽ᵏ⁾(x),

each overlay contribution `u_h⁽ᵏ⁾` extended by zero outside its own
overlay domain `Ω⁽ᵏ⁾`. Levels are never topologically merged. What makes
the sum well posed instead of merely redundant is that every overlay is
constrained to vanish on its *artificial* boundary

    Γ_o⁽ᵏ⁾ = ∂Ω⁽ᵏ⁾ \ ∂Ω,

the part of the overlay's edge that is not the physical boundary. So the
coarse level alone carries the solution outside the overlay, and the fine
level adds detail strictly inside it, fading to zero at its own rim. All
of that is automatic: `overlay` is one call, and everything below is one
coupled Galerkin system over both levels, never two separate solves.

**The problem.** A manufactured solution with a single localised
feature — a narrow Gaussian peak at the centre of the unit box:

    u(x)  = exp(−α ‖x − c‖²),         α = 250,  c = (½, …, ½),
    −Δu   = (2Dα − 4α² ‖x − c‖²) u  =: f(x),

with the exact trace imposed as Dirichlet data on `∂Ω`. The peak is
about `1/√α ≈ 0.06` wide — narrower than a cell of the coarse mesh
below, so that mesh resolves it badly. Which is exactly the situation
an overlay is for.
=#

using Unfitted

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

# ── Problem data, written for any dimension ──────────────────────────────────
#
# Nothing in this script is 2-D except the value of `D`. The domain
# corners are built with `ntuple`, and `cells = 8` / `order = 3` are
# scalars that `space` replicates across the axes, so raising `D` to 3
# runs the identical study one dimension up (at tensor-product cost).

const D = 2
const ALPHA = 250.0

# ‖x − c‖² about the centre c = (½, …, ½). Written with a broadcast and
# a reduction rather than `x[1]`, `x[2]` so the dimension never appears.
squared_radius(x) = sum(abs2, x .- 0.5)

exact(x) = exp(-ALPHA * squared_radius(x))
source(x) = (2 * D * ALPHA - 4 * ALPHA^2 * squared_radius(x)) * exact(x)

omega = box(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D))
refined_region = box(ntuple(_ -> 0.25, D), ntuple(_ -> 0.75, D))

# ── One run, five calls ──────────────────────────────────────────────────────
#
# Every configuration below differs only in its `Space`, so the rest of
# the workflow is factored into this helper. Its body is the same five
# calls as Tutorial 1, in the same order, with nothing hidden.

function solve_and_report(V, title; parameters=())
    u = field(:u, V)
    problem = poisson(u; source=source, dirichlet=[dirichlet(exact; on=boundary(:all))])
    model = prepare(problem)
    solution = solve!(model)
    report = diagnostics(model, solution; exact)
    print_run_report(title, report; parameters)
    println()
    return model, solution, report
end

# ── Run 1: the coarse base alone ─────────────────────────────────────────────
#
# Eight cells per axis at order 3. The peak spans roughly one cell, so
# this is the "resolved badly" baseline.

V_base = space(omega; cells=8, order=3)
_, _, report_base = solve_and_report(V_base, "Run 1 — base level only")

# ── Run 2: add an overlay over the peak ──────────────────────────────────────
#
# `overlay(V, subdomain; cells, order)` returns a *new* `Space` with one
# more level. The sub-box need not align with anything — overlay
# placement is independent of the coarse mesh — but here it does align,
# which matters later for covered-mode pruning. Eight cells across a box of
# half the domain's width is twice the base resolution, and `order` is
# inherited from the base level, so this is pure h-refinement.

V_overlay = overlay(V_base, refined_region; cells=8)
_, _, report_overlay = solve_and_report(V_overlay, "Run 2 — base + full overlay on the peak")

println("Error, base only        : ", report_base.l2_error)
println("Error, with the overlay : ", report_overlay.l2_error)
println("Improvement factor      : ", report_base.l2_error / report_overlay.l2_error)
println()

#=
── Covered-mode pruning: what `reduced_mode_counts` is telling you ───────────

Look at `reduced mode counts` in Run 2's report: one entry per level.
The base level's entry is large; the overlay's is zero, because nothing
covers the finest level and so it has nothing to shed.

Why a covered coarse cell can shed modes. Inside the overlay's domain
the answer is carried by the *sum* of the two levels, and the finer
level — same order, half the cell size — already resolves everything the
coarse cell's high-order "bubble" functions were there to resolve. Those
bubbles then buy no accuracy while still costing unknowns and, because
they nearly duplicate what the overlay provides, conditioning. So the
package eliminates them.

Two guards keep the elimination from going too far:

  * Elimination is *per mode*, not per cell, and a mode is shed only when
    every cell it touches is covered. A function straddling the rim of the
    covered region survives, because outside the rim it is the only thing
    carrying the solution.

  * The coarse level's linear skeleton stays. That skeleton is what
    continues the solution across the overlay's artificial boundary, where
    the overlay itself is constrained to vanish. A buried *linear* mode is
    dropped only in the stricter case where the covering level is nested
    over this one and therefore reproduces it exactly — there the two are
    genuinely linearly dependent, not merely similar.

`prune_covered = false` switches the rule off. Note that it belongs to the
level that *sheds* the modes — the base — so it is `space` that has to be
told, not `overlay`. Run 2b does exactly that, so the mechanism is
measured here rather than merely described.
=#

# ── Run 2b: the same stack with covered-mode pruning switched off ─────────────────

V_unreduced = overlay(space(omega; cells=8, order=3, prune_covered=false), refined_region; cells=8)
_, _, report_unreduced = solve_and_report(V_unreduced,
                                          "Run 2b — Run 2 with covered-mode pruning switched off";
                                          parameters=(:prune_covered => false,))

println("Run 2  (reduction on) : ", report_overlay.active_unknowns, " unknowns, error ",
        report_overlay.l2_error)
println("Run 2b (reduction off): ", report_unreduced.active_unknowns, " unknowns, error ",
        report_unreduced.l2_error)
println()

# ── Run 3: selective activation at construction time ─────────────────────────
#
# The feature is round; the overlay box is square. Every overlay cell in
# the corners of that box is paying for resolution where the solution is
# already flat. `active = …` marks individual cells of a level as
# participating or not: inactive cells are dropped from the dof
# enumeration entirely, and the face between an active and an inactive
# cell becomes part of that level's artificial boundary — so the overlay
# still fades to zero at the edge of the active patch.
#
# The mask is given here as a predicate over `(cell_box, cell_index)`.
# It may also be a Boolean array shaped like the cell grid, or an
# iterable of the `CartesianIndex`es to keep.

const PATCH_RADIUS = 0.22
near_peak(cell_box, cell_index) = squared_radius(center(cell_box)) < PATCH_RADIUS^2

V_masked = overlay(V_base, refined_region; cells=8, active=near_peak)
model, _, report_masked = solve_and_report(V_masked,
                                           "Run 3 — overlay masked to a patch around the peak";
                                           parameters=(:patch_radius => PATCH_RADIUS,))

println("Unknowns, full overlay  : ", report_overlay.active_unknowns, "   error ",
        report_overlay.l2_error)
println("Unknowns, masked overlay: ", report_masked.active_unknowns, "   error ",
        report_masked.l2_error)
println("Active overlay cells    : ", count(active_cells(model; level=2)), " of ",
        length(active_cells(model; level=2)))
println()

# ── Run 4: switching cells on and off after the model is prepared ────────────
#
# `activate!` and `deactivate!` do the same thing to a model that is
# already prepared. They take the same selector shapes as `active =`, and
# they select cells to flip rather than replacing the whole mask. Both
# invalidate everything that depends on the mask — integration plan, dof
# layout, cached operators — and bump the model's version, so an
# outstanding `Solution` from before the change refuses to be reused
# instead of silently reporting stale values. Call `solve!` again.
#
# This is how a transient analysis follows a moving feature without
# rebuilding the model from scratch: light up the cells the feature is
# entering, and switch off the ones it has left.

activate!(model; level=2, cells=trues(V_masked.levels[2].mesh.cells))
solution_grown = solve!(model)
report_grown = diagnostics(model, solution_grown; exact)

println("After activate! (all overlay cells on):")
println("    active overlay cells : ", count(active_cells(model; level=2)))
println("    active unknowns      : ", report_grown.active_unknowns, "   (Run 2 had ",
        report_overlay.active_unknowns, ")")
println("    relative L² error    : ", report_grown.l2_error, "   (Run 2 had ",
        report_overlay.l2_error, ")")

# And back again: deactivating everything *not* near the peak restores
# Run 3's discretisation, unknown for unknown.
deactivate!(model; level=2, cells=(cell_box, i) -> !near_peak(cell_box, i))
solution_shrunk = solve!(model)
report_shrunk = diagnostics(model, solution_shrunk; exact)

println("After deactivate! (patch only again):")
println("    active overlay cells : ", count(active_cells(model; level=2)))
println("    active unknowns      : ", report_shrunk.active_unknowns, "   (Run 3 had ",
        report_masked.active_unknowns, ")")
println("    relative L² error    : ", report_shrunk.l2_error, "   (Run 3 had ",
        report_masked.l2_error, ")")
println()

# ── What to check when this script runs ──────────────────────────────────────
#
# Run 2's error is more than an order of magnitude below Run 1's. Run 2b
# carries the modes that Run 2 shed and lands on the same error to a
# dozen digits, so those unknowns really were redundant. Run 3 reaches
# Run 2's accuracy with only half the overlay cells, because the ones it
# drops sat where the solution is already flat — that is what selective
# activation buys. And Run 4's mutated model reproduces Runs 2 and 3 to
# the last digit, which is the evidence that `activate!` / `deactivate!`
# really rebuild everything they claim to.

print_run_report("Tutorial 2 — summary of the final (patch-only) configuration", report_shrunk;
                 parameters=(:dimension => D, :base_cells => V_base.levels[1].mesh.cells,
                             :overlay_cells => V_masked.levels[2].mesh.cells,
                             :order => nominal_order(V_base.levels[1]), :alpha => ALPHA,
                             :patch_radius => PATCH_RADIUS))
