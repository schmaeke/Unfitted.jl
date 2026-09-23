#=
Tutorial 5 — Ladders: Declaring a Nested Stack and Refining Per Cell
====================================================================

**What you will learn.** Tutorial 2 built one overlay by hand and
remarked, in passing, that its box "does align, which matters later for
covered-mode pruning". This is later. You will see what misalignment actually
costs — an accuracy ceiling that arrives with no error, no residual and
no rank deficiency — then declare a whole stack with `ladder`, which
cannot misalign, and switch its cells on per level with `adapt`.

**What you should already know.** Tutorial 2: `overlay`, activation
masks, and what `reduced mode counts` means.

**The rule.** A level's modes are shed where a finer level covers them.
That elimination is exact — it removes redundancy and nothing else —
only when the two meshes are *nested*: the finer level's node
coordinates must reproduce the coarser one's wherever they overlap. Two
arithmetic conditions are enough,

    (i)  every corner of the finer box is a node coordinate of the coarser
         mesh, and
    (ii) the finer cell count is an integer multiple, per axis, of the
         coarser cells the box spans.

Miss either and the same elimination removes modes that nothing replaces.
Nothing downstream notices: the operator stays full rank at a healthy
condition number and the residual is still machine precision. The only
symptom is that the answer stops improving. `is_nested` is the check, and
`ladder` is the constructor that makes the check unnecessary.

**The problem.** Tutorial 2's, unchanged — a narrow Gaussian peak in the
middle of the unit box, badly resolved by the coarse mesh:

    u(x)  = exp(−α ‖x − c‖²),         α = 250,  c = (½, …, ½),
    −Δu   = (2Dα − 4α² ‖x − c‖²) u  =: f(x).
=#

using LinearAlgebra: rank
using Unfitted

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

const D = 2
const ALPHA = 250.0

squared_radius(x) = sum(abs2, x .- 0.5)
exact(x) = exp(-ALPHA * squared_radius(x))
source(x) = (2 * D * ALPHA - 4 * ALPHA^2 * squared_radius(x)) * exact(x)

omega = box(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D))
refined_region = box(ntuple(_ -> 0.25, D), ntuple(_ -> 0.75, D))

# Every configuration below differs only in its `Space`, so the five calls of
# Tutorial 1 are factored out once more. Only the two runs worth a full report
# print one; the rest are read for their two numbers.
function run_case(V)
    model = prepare(poisson(V; source=source, dirichlet=[dirichlet(exact; on=boundary(:all))]))
    return diagnostics(model, solve!(model); exact=exact)
end

#=
── Part 1: a finer overlay that gives a worse answer ────────────────────────

The refined region spans four cells of the `cells = 8` base mesh per axis, so
an overlay nests when its own cell count is a multiple of four. Below, the
same box is meshed at 8, 9, …, 16 cells and the resulting error is printed
against the overlay's cell size.

Read the 9-cell row against the 8-cell one. It is a finer mesh, it carries
more unknowns, and it is less accurate. That is not noise and it is not a
subtlety of the peak: 9/4 is not a whole number, so the overlay's nodes fall
between the base's, and the elimination that runs by default underneath
throws away modes the overlay cannot replace.
=#

println("=== Part 1 — the same box, meshed six ways ===\n")

println("cells   h_overlay   nested   unknowns   relative L² error")
for overlay_cells in (8, 9, 10, 11, 12, 16)
    V = overlay(space(omega; cells=8, order=3), refined_region; cells=overlay_cells)
    report = run_case(V)
    println(rpad(overlay_cells, 8), rpad(round(0.5 / overlay_cells; sigdigits=3), 12),
            rpad(is_nested(V), 9), rpad(report.active_unknowns, 11), report.l2_error)
end
println()

#=
── Part 1b: what the elimination actually removed ───────────────────────────

The mass matrix is the Gram matrix of the active basis, so its rank counts the
functions that are genuinely independent. Build each stack twice — once as it
ships, once as the **unreduced twin** `prepare` returns under `prune = false`,
which keeps every covered mode — and compare the size of the reduced space with
the *rank* of the unreduced one. Equal means the elimination removed redundancy
and nothing else. Smaller means it removed something real. The twin exists for
exactly this comparison; its own operator is not one to solve with, and the
`prepare` docstring says why.

This is the only place the difference is visible. Both stacks below assemble
to a full-rank operator at a healthy condition number and solve to machine
precision; nothing in the report distinguishes them.
=#

println("=== Part 1b — was the elimination free? ===\n")

function gram_size_and_rank(V; prune=true)
    (m=prepare(mass(V); prune=prune); assemble!(m); A=Matrix(m.matrix); (size(A, 1), rank(A)))
end

for overlay_cells in (8, 9)
    reduced, _ = gram_size_and_rank(overlay(space(omega; cells=8, order=3), refined_region;
                                            cells=overlay_cells))
    _, unreduced_rank = gram_size_and_rank(overlay(space(omega; cells=8, order=3), refined_region;
                                                   cells=overlay_cells); prune=false)
    println("Overlay of ", overlay_cells, " cells: reduced space spans ", reduced,
            ", the unreduced one spans ", unreduced_rank,
            reduced == unreduced_rank ? "  — free" :
            "  — $(unreduced_rank - reduced) dimensions destroyed")
end
println()

#=
── Part 2: declaring the stack instead of placing it ────────────────────────

`ladder` builds a base plus `depth` overlays, each spanning the whole domain
at `splits` times the previous level's resolution. Conditions (i) and (ii)
hold for every pair by construction, so `is_nested` is true and stays true
however the levels are later switched on.

The stack arrives *inert*: every overlay is declared with no active cell, so
it carries the unknowns of the base level alone until you say otherwise. That
is the point of declaring rather than growing — the levels exist, and adding
resolution later costs an assembly, not a rebuild of the whole space.
=#

println("=== Part 2 — a declared ladder ===\n")

V_ladder = ladder(omega; cells=8, order=3, depth=2, splits=2)
println("Levels and their meshes  : ", [l.mesh.cells for l in V_ladder.levels])
println("Nested by construction   : ", is_nested(V_ladder))
println("Active cells per overlay : ", [count(active_cells(V_ladder; level=k)) for k in 2:3])
println()

# `adapt` replaces the activation of the levels you name. A per-base-cell depth
# map is the shorthand for the structured case: depth 2 over the middle four
# cells of the base mesh puts both overlays there.
depths = zeros(Int, 8, 8)
depths[3:6, 3:6] .= 2
report_block = run_case(adapt(V_ladder, depths))
print_run_report("Ladder, both levels over the peak", report_block;
                 parameters=(:depth_map => "2 over the middle 4×4 base cells",))

#=
── Part 3: refining per cell, and where to stop ─────────────────────────────

Nothing requires a level to be active only where the level below it is, or its
cells to tile whole parent cells. Superposition has no hanging nodes and no
2:1 balance to maintain, so the finest level can be switched on alone, over
exactly the cells the feature touches, with the level between it and the base
left dormant.

What *does* constrain the mask is the artificial boundary. An overlay is
clamped to zero on the rim of its active region, so that rim has to lie where
the correction it carries is already negligible. Below, the same level is
switched on over four discs of growing radius and the last column says how
large the exact solution still is where the clamp falls. Tightening the mask
past the point where that value matters does not save work — it buys a wrong
answer.
=#

println("\n=== Part 3 — the finest level alone, per cell ===\n")

# `cell_box` and `cell_indices` read a level without reaching inside it.
function cells_within(V, level, radius_squared)
    return [ci
            for ci in cell_indices(V; level=level)
            if squared_radius(center(cell_box(V, ci; level=level))) < radius_squared]
end

println("mask r² <   cells   unknowns      error    u at the mask rim")
report_sparse = nothing
for r2 in (0.02, 0.05, 0.10)
    cells = cells_within(V_ladder, 3, r2)
    report = run_case(adapt(V_ladder, 3 => cells))
    r2 == 0.05 && (global report_sparse = report)
    println("   ", rpad(r2, 8), rpad(length(cells), 8), rpad(report.active_unknowns, 11),
            rpad(round(report.l2_error; sigdigits=4), 12), exp(-ALPHA * r2))
end
println()
println("Both levels, block  : ", report_block.active_unknowns, " unknowns, error ",
        report_block.l2_error)
println("Level 3 only, r² < 0.05: ", report_sparse.active_unknowns, " unknowns, error ",
        report_sparse.l2_error)
println()

#=
── Part 4: marking on one level, refining on another ────────────────────────

An error indicator lives on the cells of whichever level carries the solution
there. `overlapping_cells` carries a marking from that level to the one you
want to refine — downward it names the children of the marked cells, upward
the cells that contain them.

`adapted` then builds the refined model without disturbing the old one, which
is what lets a solution be carried across. Use `L2Projection` for that, in
both directions: refining deepens coverage, so the parent's modes are
eliminated in the target and the dof-matching `Rewire` backend has no
counterpart to copy them into.
=#

println("=== Part 4 — mark on the base, refine on the overlay ===\n")

coarse = prepare(poisson(V_ladder; source=source, dirichlet=[dirichlet(exact; on=boundary(:all))]))
u_coarse = solve!(coarse)

# A crude indicator: the gradient magnitude at each base cell's centre. Enough
# to find the peak; a real one would come from a residual or from the decay of
# the cell's modal coefficients.
function cell_indicator(u, model, V, ci)
    g = field_gradient(u, model, Tuple(center(cell_box(V, ci; level=1))))
    return sqrt(sum(abs2, g))
end

indicator = map(ci -> cell_indicator(u_coarse, coarse, V_ladder, ci),
                cell_indices(V_ladder; level=1))
threshold = 0.5 * maximum(indicator)
marked = [ci for ci in cell_indices(V_ladder; level=1) if indicator[ci] >= threshold]

target = adapted(coarse, 2 => overlapping_cells(V_ladder, marked; from=1, to=2))
transfer(u_coarse, coarse, target; via=L2Projection())
report_marked = diagnostics(target, solve!(target); exact=exact)

println("Marked base cells        : ", length(marked), " of ", 8^D)
println("Unknowns, coarse → refined: ", diagnostics(coarse).active_unknowns, " → ",
        report_marked.active_unknowns)
println("Error,    coarse → refined: ", diagnostics(coarse, u_coarse; exact=exact).l2_error, " → ",
        report_marked.l2_error)
println()

#=
── What to check when this script runs ──────────────────────────────────────

Part 1 is the whole argument. The nested rows — 8, 12, 16 cells — fall
monotonically as the overlay is refined. The rows between them do not: at 9
cells the answer is worse than at 8 despite a finer mesh and more unknowns.
Part 1b says why. On the nested stack the reduced space spans exactly what the
unreduced one spans, so the elimination cost nothing; on the 9-cell stack it
spans about a hundred dimensions fewer, and those are gone. `is_nested` is the
only thing that separates the two before you measure an error, which is why
`ladder` guarantees it rather than leaving it to arithmetic done by hand.

Part 3's first row is the trap: at `r² < 0.02` the mask rim sits where the
solution is still 7e-3, the overlay is clamped to zero there, and the error is
an order of magnitude worse than the rows below it. From `r² < 0.05` onward the
rim is in the noise and the error settles at the same 2.2e-4 the block
refinement reaches — with 1897 unknowns against 2617, and the level between
never switched on at all. Widening further only adds unknowns.

Part 4 refines where a coarse indicator points, and the error falls. The
`transfer` call is a no-op for a static solve like this one; it is there
because it is the step a transient run cannot skip.
=#

print_run_report("Tutorial 5 — summary of the per-cell refined ladder", report_sparse;
                 parameters=(:base_cells => V_ladder.levels[1].mesh.cells,
                             :depth => length(V_ladder.levels) - 1, :splits => 2,
                             :mask => "level 3 where r² < 0.05, level 2 dormant"))
