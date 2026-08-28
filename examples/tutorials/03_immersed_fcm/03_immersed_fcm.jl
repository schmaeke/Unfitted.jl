#=
Tutorial 3 — The Finite Cell Method: Geometry That Does Not Fit the Mesh
=======================================================================

**What you will learn.** How to solve on a domain `Ω` that the Cartesian
mesh knows nothing about. You will describe `Ω` as a level set, build it
from two smooth pieces with a CSG combinator, hand it to `space`, read
the three diagnostics that say whether the cut-cell quadrature did its
job, and impose a Dirichlet condition on the immersed boundary itself —
which cannot be done by eliminating degrees of freedom, because no
degree of freedom lives there.

**What you should already know.** Tutorials 1 and 2. Some acquaintance
with fictitious-domain / immersed-boundary ideas helps but is not
assumed.

**The idea.** In the finite cell method the mesh is a plain Cartesian
grid over a bounding box, and the physical domain is carved out of it by
a scalar level set:

    Ω = { x : φ(x) ≤ 0 }.

Each cell of the grid is then one of three things: entirely inside `Ω`
("full", integrated by ordinary tensor Gauss), entirely outside
("fictitious", dropped from the system), or straddling `∂Ω` ("cut").
Cut cells are the whole difficulty: a Gauss rule for the *cell* is
wrong, because the integrand must be integrated over `Ω ∩ cell` only.
This package computes the exact moments of that region with Saye's
dimension-reduction implicit quadrature and then fits a small
*non-negative* rule to them, so a cut cell ends up with a handful of
points and positive weights, just like an uncut one.

    R. I. Saye, *High-order quadrature methods for implicitly defined
    surfaces and volumes in hyperrectangles*, SIAM J. Sci. Comput. 37
    (2015) A993–A1019. doi:10.1137/140966290

    B. Müller, F. Kummer, M. Oberlack, *Highly accurate surface and
    volume integration on implicit domains by means of moment-fitting*,
    Int. J. Numer. Methods Engng. 96 (2013) 512–528.
    doi:10.1002/nme.4569

**The problem.** An annulus `Ω = { r_i ≤ ‖x‖ ≤ r_o }`, `r_i = ¼`,
`r_o = ¾`, embedded in the square `[−0.8, 0.8]²`, with

    −Δu = 4    in Ω,        u = 0   on both circles.

In polar coordinates the solution is radial and known exactly:

    u(r) = −r² + A ln r + B,    A = (r_o² − r_i²) / ln(r_o/r_i),
                                B = r_i² − A ln r_i,

which is what the reported relative L² error is measured against.
Neither circle is anywhere near the grid lines, which is the point.

This script is genuinely two-dimensional, and only one thing makes it so:
the immersed boundary is described by a mesh of line segments. In 3-D you
would hand the same `on =` keyword a triangle surface instead. The level
set, the domain, the quadrature and the weak form are all dimension-generic.
=#

using Unfitted
using LinearAlgebra: dot, norm
using StaticArrays: SVector

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

# ── The exact solution ───────────────────────────────────────────────────────

const R_INNER = 0.25
const R_OUTER = 0.75
const COEFF_A = (R_OUTER^2 - R_INNER^2) / log(R_OUTER / R_INNER)
const COEFF_B = R_INNER^2 - COEFF_A * log(R_INNER)

exact(x) =
    let r = norm(x)
        -r^2 + COEFF_A * log(r) + COEFF_B
    end

# ── Step 1: describe Ω as a CSG level set ────────────────────────────────────
#
# The annulus is the intersection of "inside the outer circle" and
# "outside the inner circle". Each is a `leaf`: a smooth scalar function
# that is negative where its own half of the condition holds.
#
# It is tempting to collapse the two into the single level set
# `max(r_i − ‖x‖, ‖x‖ − r_o)`. Do not. That function has a kink along
# the mid-radius circle, and the implicit-quadrature kernel derives its
# accuracy from the level set being smooth. Keeping the leaves separate
# and combining them with `intersect` lets every boundary piece stay
# smooth, so both rims are integrated at high order. `union`, `setdiff`
# and `complement` combine leaves the same way.
#
# `lipschitz = 1.0` declares that each leaf is a true signed-distance
# function, which it is here; that lets the classifier certify whole
# cells cheaply. Leave it at the default `Inf` if you are unsure — the
# classifier then falls back to sampling and subdivision, which is
# slower but never wrong.
#
# A level set must accept `ForwardDiff.Dual` numbers, because the implicit
# quadrature differentiates it to find the boundary. Write it out of
# ordinary arithmetic, as below, and that requirement takes care of itself.

radius(x) = sqrt(sum(abs2, x))

annulus_geometry = intersect(leaf(x -> radius(x) - R_OUTER; lipschitz=1.0),
                             leaf(x -> R_INNER - radius(x); lipschitz=1.0))

# ── Step 2: turn the geometry into a PhysicalDomain ──────────────────────────
#
# `subcell_length_scale` is a geometric robustness knob, not an accuracy
# knob. It sets how finely the classifier may subdivide a cell before
# deciding in/out, and how finely the quadrature kernel may subdivide a
# cut cell whose level set turns around inside it. Smooth, well-resolved
# cut cells like ours get exact moments without any subdivision at all,
# so the value below only matters for thin or nearly tangent features.
# `max_depth` caps the recursion.

const CELLS = 12
const EMBEDDING_HALFWIDTH = 0.8
const H = 2 * EMBEDDING_HALFWIDTH / CELLS          # base cell size

omega = box(ntuple(_ -> -EMBEDDING_HALFWIDTH, 2), ntuple(_ -> EMBEDDING_HALFWIDTH, 2))
annulus = physical_domain(annulus_geometry; subcell_length_scale=H / 16, max_depth=4)

# ── Step 3: hand the domain to the space ─────────────────────────────────────
#
# This is the whole integration change: one keyword. Everything
# downstream — dof enumeration, assembly, the L² error — now runs over
# `Ω` instead of over the box. Cells lying entirely outside `Ω` are
# dropped from the dof layout, which is why the reported unknown count
# is smaller than a `12 × 12` order-3 grid would suggest.

V = space(omega; cells=CELLS, order=3, physical=annulus)
u = field(:u, V)

# ── Step 4: describe the immersed boundary as a mesh ─────────────────────────
#
# A `BoundaryMesh` is a list of simplices in physical space — segments in
# 2-D, triangles in 3-D — that need not line up with the grid at all; the
# package subdivides each segment against every level's cells before
# integrating. Here two closed polylines approximate the two circles.
#
# Orientation sets the normal. For a 2-D segment the derived normal is
# 90° clockwise from the direction of travel, i.e. it points out of the
# region lying to your *left* as you walk the polyline. So traverse the
# outer circle counter-clockwise and the inner circle clockwise: in both
# cases the annulus is on the left and the normal points out of `Ω`.
#
# The polyline is a chord approximation of a circle, so its segment
# length is an accuracy parameter of its own. Sizing the segments by
# target arc length rather than by a fixed count keeps both rims equally
# well resolved even though they have different radii.

const SEGMENT_LENGTH = 0.004
segment_count(r) = ceil(Int, 2π * r / SEGMENT_LENGTH)

function circle_mesh(r, orientation)
    n = segment_count(r)
    vertices = [SVector(r * cos(orientation * t), r * sin(orientation * t))
                for t in range(0.0, 2π; length=n + 1)[1:n]]
    return polyline_mesh(vertices; closed=true)
end

outer_rim = circle_mesh(R_OUTER, +1)      # counter-clockwise: Ω on the left
inner_rim = circle_mesh(R_INNER, -1)      # clockwise: Ω on the left

# ── Step 5: impose u = 0 on the immersed boundary, weakly ────────────────────
#
# `dirichlet(...; on=boundary(...))` cannot help here. That machinery
# eliminates the degrees of freedom whose trace lives on a face of the
# mesh — and `∂Ω` slices through the interiors of cells, so there are no
# such degrees of freedom. The standard remedy is Nitsche's method,
# which puts the condition into the weak form instead:
#
#     a_N(u, v) = ∫_Γ [ −(∇u·n) v − u (∇v·n) + (β/h) u v ] ds.
#
# The first term is the consistency term: it is exactly what integration
# by parts leaves on Γ, so the *exact* solution still satisfies the weak
# form. The second symmetrises the operator, which keeps the system SPD
# and the direct solver happy. The third is a penalty that makes the
# whole thing coercive; `β` must exceed a constant times `p²` for that,
# and `h` is the cell size, so the penalty scales like a discrete inverse
# estimate rather than being a free fudge factor. With homogeneous data
# `u = 0` there is no right-hand-side counterpart, so the form below is
# bilinear only.
#
# A form callback returns `TestChannels(value_coefficient,
# gradient_coefficient)`: the coefficients that multiply the test
# function's value and its gradient at this quadrature point. Reading the
# formula above, `−(∇u·n) + (β/h) u` multiplies `v`, and `−u n`
# multiplies `∇v` — that is the whole translation.

const BETA = 100.0

function nitsche_bilinear(q, trial)
    n = q.normal                                    # outward unit normal at q.x
    return TestChannels(-dot(trial.gradient, n) + (BETA / H) * trial.value, -trial.value * n)
end

nitsche = WeakForm(bilinear=nitsche_bilinear, linear=q -> 0.0, symmetric=true)

# ── Step 6: assemble the problem and solve ───────────────────────────────────
#
# Three bilinear contributions — the volume stiffness plus one Nitsche
# block per rim — and one volume load. `on = <a BoundaryMesh>` is what
# moves a block from the volume onto an immersed surface; the same
# keyword takes a `boundary(...)` selector for grid-aligned faces, or an
# `interface(...)` tag for a coupled multi-domain problem.

problem = Problem((u,);
                  blocks=(block(u, u, stiffness_form()), block(u, u, nitsche; on=outer_rim),
                          block(u, u, nitsche; on=inner_rim)), loads=(source_load(u; source=4.0),))

model = prepare(problem)
solution = solve!(model)
report = diagnostics(model, solution; exact)

# ── Step 7: read the FCM diagnostics ─────────────────────────────────────────

println("Tutorial 3 — pointwise values along a radius")
for r in (0.3, 0.5, 0.7)
    x = SVector(r, 0.0)
    println("    u_h(", r, ", 0) = ", value(solution, model, u, x), "   exact = ", exact(x))
end
println()

#=
Three numbers in the report below are specific to the immersed path.

  * `cut region count` — how many integration regions the level set
    classified as cut, and therefore got a moment-fit rule instead of
    tensor Gauss. Zero would mean the geometry never actually crossed a
    cell, i.e. that you are not testing what you think you are.

  * `fit failure count` — cut regions whose non-negative fit missed the
    package's internal failure threshold. Nothing is silently dropped
    when that happens: such a region normally falls back to the raw
    implicit-quadrature rule the moments were derived from, which is
    correct and non-negative but carries 50 to 200 times the points.
    (`diagnostics` reports that subset separately as
    `cut_fallback_count`, which this printout does not show.) Either
    way, a nonzero count says those cells are too coarse for the
    geometry passing through them — refine there rather than loosening a
    tolerance. It is 0 here.

  * `moment-fit residual (max)` — the largest L² residual of any fit,
    i.e. how well the fitted rule reproduces the exact moments. Values
    around 1e-17 mean the rules are exact to roundoff. Anything
    approaching the `target_residual` of `physical_domain` (default
    1e-6) means the fits are only just succeeding.

Also worth a glance: `inactive cell counts` includes the cells the
level set folded away as fictitious, and `small overlaps` counts
integration regions so tiny that they are a conditioning hazard — this
package reports them rather than silently discarding them.
=#

print_run_report("Tutorial 3 — Poisson on an immersed annulus (FCM + Nitsche)", report;
                 parameters=(:r_inner => R_INNER, :r_outer => R_OUTER,
                             :embedding_box => "[-0.8, 0.8]²", :cells => V.levels[1].mesh.cells,
                             :order => V.levels[1].order,
                             :subcell_length_scale => annulus.subcell_length_scale,
                             :nitsche_beta => BETA, :nitsche_h => H,
                             :outer_rim_segments => length(outer_rim.cells),
                             :inner_rim_segments => length(inner_rim.cells)))
