#=
Tutorial 4 — A Second Basis Family: B-splines
=============================================

**What you will learn.** How to swap the basis family, what B-splines
buy you over the default, the one thing about nested B-spline levels
that would break without help from the library, and how the family
behaves on an immersed domain.

**What you should already know.** Tutorials 1 to 3.

**Why another family.** The default integrated Legendre basis is `C⁰`
across cell boundaries: values match, slopes need not. An open-knot
B-spline basis of degree `p` with simple interior knots is `C^(p−1)`
there — a cubic B-spline space is `C²` across every cell boundary of
its level. That extra smoothness is worth having when the solution is
smooth (fewer unknowns for the same accuracy, and derivative fields
that are continuous without post-processing) and worth avoiding when it
is not (a kink in the solution is expensive for a basis that refuses to
have one). Choosing the family is a modelling decision, and the package
treats it as one: geometry, assembly, constraints, projection and
solvers are family-agnostic, so the only thing that changes below is
one keyword.

**Loading the family.** `bspline()` lives in a package *extension*. The
name is always importable from `Unfitted`, but the methods behind it
only exist once its trigger package is loaded:

    using Unfitted
    using BasicBSpline        # ← without this line, bspline() is a MethodError

That is Julia's weak-dependency mechanism, not a quirk of this package:
`BasicBSpline` is declared under `[weakdeps]`, so installing Unfitted
does not drag it in, and the extension module activates the moment both
packages are present. Note that nothing in the script below ever calls
`BasicBSpline` directly — the `using` line exists purely to switch the
extension on. Your own project needs `BasicBSpline` in its
`Project.toml`, exactly as this tutorial's does.

**The two problems.**

  1. `−Δu = 1` on `Ω = (0, 1)²` with `u = 0` on `∂Ω`, refined by a
     *nested* overlay. The classical series solution gives
     `u(½, ½) = 0.07367135…`, which is the number to check against.

  2. The same equation on a perforated square — `(0, 1)²` with a disk of
     radius ¼ removed — solved by the finite cell method of Tutorial 3.
     Here the check is agreement with the integrated Legendre family on
     the identical mesh and geometry.
=#

using Unfitted
using BasicBSpline
using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

const CENTRE_REFERENCE = 0.07367135326539033      # u(½, ½) from the Fourier series

omega = box((0.0, 0.0), (1.0, 1.0))
homogeneous = [dirichlet(0.0; on=boundary(:all))]

# One run, five calls, exactly as in the earlier tutorials.
function solve_and_report(V, title; parameters=())
    u = field(:u, V)
    model = prepare(poisson(u; source=1.0, dirichlet=homogeneous))
    solution = solve!(model)
    report = diagnostics(model, solution)
    print_run_report(title, report; parameters)
    println()
    return model, solution, u, report
end

# ── Part 1a: a plain B-spline discretisation ─────────────────────────────────
#
# `basis = bspline()` is the whole change. `order = 3` is the B-spline
# degree, the same keyword that means polynomial order for integrated
# Legendre. Note the unknown count: an 8-cell degree-3 B-spline axis
# carries `cells + p = 11` functions, against `cells·p + 1 = 25` for
# integrated Legendre — that is the continuity showing up as a smaller
# space.

V_base = space(omega; cells=8, order=3, basis=bspline())
_, _, _, report_base = solve_and_report(V_base, "Part 1a — B-spline base level, 8 cells, degree 3")

# ── Part 1b: a nested overlay ────────────────────────────────────────────────
#
# The overlay covers [¼, ¾]² with 4 cells per axis. Its cell size is
# exactly the base's, and its edges fall on base cell boundaries, so the
# overlay mesh is *nested* in the base mesh. Equal cell size is the
# simplest nested case and isolates the effect this section is about; a
# genuinely finer nested overlay is treated in exactly the same way.
#
# Nesting is the natural thing to reach for, and for this family it is
# also a trap. A B-spline overlay of the same degree, on a mesh whose
# cell boundaries include the base's, reproduces *exactly* some of the
# base functions buried beneath it. The superposition then carries the
# same function twice, and a matrix with two identical columns is exactly
# singular — no amount of quadrature care will save it.
#
# `reduce_order`, on by default for both `space` and `overlay`, removes
# the duplicate. Tutorial 2 showed the integrated Legendre version of the
# same keyword, where the modes eliminated are only *nearly* redundant, so
# switching the rule off can in principle buy accuracy (in that tutorial's
# aligned setup it bought none). Here the overlay reproduces the
# eliminated function *exactly*, so the discrete space is literally
# unchanged and the elimination is free.
#
# `reduced mode counts` in the report below is `[1, 0]`: one function
# dropped from the base, none from the overlay. That is the whole
# mechanism, visible as a number.

V_nested = overlay(V_base, box((0.25, 0.25), (0.75, 0.75)); cells=4)
model_nested, solution_nested, u_nested, report_nested = solve_and_report(V_nested,
                                                                          "Part 1b — a nested B-spline overlay on [¼, ¾]²")

# ── Part 1c: what the deduplication is worth ─────────────────────────────────
#
# Turn it off and look at the condition number. `reduce_order` belongs to
# the level that *sheds* the mode — here the base — so it is `space`, not
# `overlay`, that has to be told.
#
# The reported `condition estimate` is a genuine `cond(A)` rather than an
# estimate; the model prints it for systems small enough to densify
# (a few hundred unknowns), and `NaN` above that.

V_undeduped = overlay(space(omega; cells=8, order=3, basis=bspline(), reduce_order=false),
                      box((0.25, 0.25), (0.75, 0.75)); cells=4)
_, _, _, report_undeduped = solve_and_report(V_undeduped,
                                             "Part 1c — the same stack, deduplication switched off";
                                             parameters=(:reduce_order => false,))

println("Nested B-spline stack, deduplicated : unknowns ", report_nested.active_unknowns, ", cond ",
        report_nested.condition_estimate)
println("Nested B-spline stack, as built     : unknowns ", report_undeduped.active_unknowns,
        ", cond ", report_undeduped.condition_estimate)
println("u_h(0.5, 0.5) = ", value(solution_nested, model_nested, u_nested, (0.5, 0.5)),
        "   reference = ", CENTRE_REFERENCE)
println()

# The undeduplicated operator is numerically singular: its condition
# number lands around 1e17, well past the 1/ε ≈ 4.5e15 at which double
# precision has nothing left to give. The direct solver still returns a
# plausible-looking answer, which is precisely what makes this failure
# mode dangerous — nothing announces itself. One redundant unknown costs
# roughly fifteen orders of magnitude of conditioning.

# ── Part 2: B-splines on an immersed domain ──────────────────────────────────
#
# Ω is the unit square with a disk of radius ¼ punched out of the middle,
# so `φ(x) = ¼ − ‖x − c‖ ≤ 0` describes it with a single leaf. The outer
# boundary is grid-aligned and takes an ordinary strong Dirichlet
# condition; the rim of the hole gets no condition at all, which in the
# weak form means the natural one, `∂u/∂n = 0`.
#
# There is no verification formula for this domain, so the check is a
# cross-family one: solve the identical problem with the default
# integrated Legendre basis on the same mesh and the same geometry, and
# compare the two solutions pointwise. Two different bases, two different
# unknown counts, one answer.

perforated = physical_domain(leaf(x -> 0.25 - norm(x .- 0.5); lipschitz=1.0);
                             subcell_length_scale=1 / 128, max_depth=6)

V_immersed = space(omega; cells=8, order=3, basis=bspline(), physical=perforated)
model_bs, solution_bs, u_bs, report_bs = solve_and_report(V_immersed,
                                                          "Part 2 — B-splines on a perforated square (FCM)")

V_reference = space(omega; cells=8, order=3, physical=perforated)
model_il, solution_il, u_il, report_il = solve_and_report(V_reference,
                                                          "Part 2 — the same problem in integrated Legendre")

println("Cross-family agreement on the perforated square:")
gaps = Float64[]
for x in ((0.1, 0.1), (0.5, 0.1), (0.9, 0.5), (0.5, 0.85))
    a = value(solution_bs, model_bs, u_bs, x)
    b = value(solution_il, model_il, u_il, x)
    push!(gaps, abs(a - b))
    println("    x = ", x, "   B-spline ", a, "   Legendre ", b, "   difference ", abs(a - b))
end
println("Largest difference       : ", maximum(gaps))
println("Unknowns, B-spline       : ", report_bs.active_unknowns)
println("Unknowns, Legendre       : ", report_il.active_unknowns)
println("Cut regions / fit failures: ", report_bs.cut_region_count, " / ",
        report_bs.fit_failure_count)
println()

# ── What to check when this script runs ──────────────────────────────────────
#
# Part 1: `reduced mode counts` is `[1, 0]`, the deduplicated condition
# number is a few hundred while the undeduplicated one is around 1e17,
# and `u_h(½, ½)` agrees with the series reference to about five decimals.
#
# Part 2: the cut-region count is nonzero with no fit failures, and the
# two families agree to a few times 1e-5 — the size of their own
# discretisation error, which is all one can ask of two different spaces
# on the same mesh.

print_run_report("Tutorial 4 — summary of the nested B-spline stack", report_nested;
                 parameters=(:base_cells => V_base.levels[1].mesh.cells,
                             :overlay_cells => V_nested.levels[2].mesh.cells,
                             :degree => V_base.levels[1].order, :basis => "bspline",
                             :centre_value_reference => CENTRE_REFERENCE))
