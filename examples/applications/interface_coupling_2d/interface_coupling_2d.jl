#=
Heat Conduction Across A Bonded Bi-Material Joint
=================================================

Two materials with different conductivities, bonded along a straight seam,
each meshed on its own grid — and neither grid resolves the seam.

This is the smallest honest demonstration of the package's multi-domain
coupling. The physical problem is steady heat conduction on the unit
square, split by the vertical line `x = s` into

    Ω₁ = { x < s },   conductivity κ₁ = 1     (the "insulator")
    Ω₂ = { x > s },   conductivity κ₂ = 10    (the "conductor")

with `−∇·(κ ∇u) = f` on each side and the two physical bonding conditions
at the seam `Γ = { x = s }`:

    ⟦u⟧ = 0                (perfect thermal contact: temperature is continuous)
    ⟦κ ∇u·n⟧ = 0           (no heat is stored in the seam: flux is continuous)

Temperature is continuous but its *slope* is not: the flux condition forces
`κ₁ ∂ₓu₁ = κ₂ ∂ₓu₂`, so the field has a kink at the seam whose severity is
the conductivity ratio. That kink is why a single grid straddling the seam
does badly, and why the seam is worth modelling explicitly.

Why two spaces instead of one
-----------------------------

Each subdomain gets its *own* [`space`](@ref) — its own box, its own cell
count, its own polynomial order, its own level-set fold — and the two are
tied together only by an integral over `Γ`. Nothing is merged, and the two
grids do not have to line up with each other or with `Γ`; here they
deliberately do neither. That is the modelling freedom on offer: mesh each
material for its own physics, then bond them.

The bonding term is a symmetric weighted Nitsche coupling, written out in
this file rather than supplied by the package. Unfitted contributes the
two-sided interface integration — one quadrature point on `Γ` evaluated
simultaneously in the cut cell of `Ω₁` that contains it and in the cut cell
of `Ω₂` that contains it — and `couple` expands the user's form into the
four field blocks `Kₐₐ, K_ab, K_ba, K_bb` with the `+ − − +` jump signs.
The constitutive choice (Nitsche versus penalty versus a cohesive law with
a finite bond conductance) stays with the user.

Nitsche is used here rather than a bare penalty because it is *consistent*:
the exact solution satisfies the discrete interface equations, so the
coupling adds no spurious flux jump and the error below is a pure
discretisation error. A pure penalty would converge only as the penalty
grows, which conflates two effects and makes the printed number hard to
read.

Manufactured solution
---------------------

Take a globally smooth potential `q(x, y) = (x − s) sin(πy)` and define

    u(x, y) = q(x, y) / κᵢ    on Ωᵢ.

Then `κᵢ ∇uᵢ = ∇q` on both sides, so the flux is continuous across `Γ`
automatically, and `q(s, y) = 0`, so the temperature is continuous there
too — both bonding conditions hold by construction. The source is the same
expression on either side,

    f = −∇·(κ ∇u) = −Δq = π² (x − s) sin(πy),

and the exact trace is imposed as Dirichlet data on the three outer faces
of each subdomain (on `y = 0` and `y = 1` that trace is identically zero,
since `sin(πy)` vanishes; on `x = 0` and `x = 1` it is not).

The solution is not a polynomial, so the printed error is a genuine
discretisation error rather than a patch test that any consistent method
passes trivially.

What the printed metric means
-----------------------------

The report prints the relative L² error of each subdomain's field against
its own exact solution, integrated over that subdomain only (an error over
the union would double-count the seam). The headline "relative L2 error"
is the worse of the two. At the size below it lands near `10⁻⁵`, which is
what a p = 3 discretisation of a smooth field on ~7 cells per direction
should give; a value orders of magnitude larger means the coupling is not
doing its job — the usual cause being a Nitsche penalty `β` too small for
the cut-cell sliver, which makes the bilinear form lose coercivity.

Also worth reading: `symmetry residual` must be `0` (the coupling declares
itself symmetric, so the four blocks must actually mirror), and `cut region
count` must be nonzero (if it is zero, the seam accidentally fell on a grid
line and the example is not testing what it claims to).

Reference for the weighted-Nitsche interface form: C. Annavarapu, M. Hautefeuille,
J. E. Dolbow, *A robust Nitsche's formulation for interface problems*, Comput.
Methods Appl. Mech. Engrg. **225–228** (2012) 44–54.
[doi:10.1016/j.cma.2012.03.008](https://doi.org/10.1016/j.cma.2012.03.008).
=#

using Unfitted
using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

# ── Physical parameters ─────────────────────────────────────────────────────

const κ₁ = 1.0        # conductivity of the left material
const κ₂ = 10.0       # conductivity of the right material
const s = 0.53        # seam position, deliberately off every grid line below

# ── Manufactured solution ───────────────────────────────────────────────────
#
# `potential` is the globally smooth `q`; `u = q/κᵢ` per side makes both
# bonding conditions hold exactly (see the header). The source is `−Δq`,
# identical on both sides because the κ's cancel.

potential(x) = (x[1] - s) * sin(π * x[2])
exact₁(x) = potential(x) / κ₁
exact₂(x) = potential(x) / κ₂
source(x) = π^2 * (x[1] - s) * sin(π * x[2])

# ── Two independent discretisations ─────────────────────────────────────────
#
# Each subdomain lives on its own box, and the boxes overlap in a band around
# the seam so that both grids genuinely have cut cells there. Cells entirely
# on the wrong side of the seam are fictitious and are dropped from the dof
# layout; cells straddling it are cut and are integrated by the finite-cell
# moment fit. The boxes, the cell counts and the seam position are chosen so
# that `x = s` is a grid line of neither mesh, and so that the two meshes have
# no grid line in common in either direction. That is the general unfitted
# case: the seam runs through the interior of a cell on both sides, and the
# two sides disagree about where the cells are.
#
# The level sets are half-planes, `φ₁(x) = x − s` for Ω₁ = {φ₁ ≤ 0} and
# `φ₂(x) = s − x` for Ω₂. Both are exact signed distances, hence
# `lipschitz = 1.0`.

const order = 3
const x_end₁ = 0.70                          # Ω₁'s box reaches past the seam …
const x_start₂ = 0.42                        # … and Ω₂'s box starts before it
const box₁ = box((0.0, 0.0), (x_end₁, 1.0))
const box₂ = box((x_start₂, 0.0), (1.0, 1.0))
const cells₁ = (7, 6)                        # non-matching: 6 cells in y …
const cells₂ = (6, 8)                        # … against 8 on the other side

# Smallest cell width across the seam; it sets the Nitsche scaling below.
const h = min(x_end₁ / cells₁[1], (1.0 - x_start₂) / cells₂[1])

left = physical_domain(leaf(x -> x[1] - s; lipschitz=1.0); subcell_length_scale=h / 16)
right = physical_domain(leaf(x -> s - x[1]; lipschitz=1.0); subcell_length_scale=h / 16)

V₁ = space(box₁; cells=cells₁, order=order, physical=left)
V₂ = space(box₂; cells=cells₂, order=order, physical=right)
u₁ = field(:u₁, V₁)
u₂ = field(:u₂, V₂)

# ── The seam ────────────────────────────────────────────────────────────────
#
# A single straight segment is enough: assembly subdivides it against the
# merged grid lines of *both* subdomains, so each resulting piece lies in one
# cut cell per side. Orientation matters and is the caller's responsibility —
# the package derives the normal of a 2-D segment as 90° clockwise from its
# direction, so traversing (s, 0) → (s, 1) gives n = (1, 0), pointing out of
# Ω₁ and into Ω₂. That is the `a → b` direction the coupling form below
# assumes when it reads `q.normal`; reversing the two vertices would silently
# produce an inconsistent system.

seam = polyline_mesh([SVector(s, 0.0), SVector(s, 1.0)])

# ── The bonding law: symmetric weighted Nitsche ─────────────────────────────
#
# The interface term is
#
#     ∫_Γ ( −⟨κ ∇u·n⟩_w ⟦v⟧ − ⟦u⟧ ⟨κ ∇v·n⟩_w + β ⟦u⟧⟦v⟧ ) dΓ,
#
# with ⟦·⟧ = (·)₁ − (·)₂ the jump and ⟨·⟩_w = w₁(·)₁ + w₂(·)₂ a weighted
# average. The first term is the consistency term (it restores the flux the
# jump condition removed), the second is its adjoint (it is what makes the
# assembled system symmetric), and the third is the stabilisation that
# restores coercivity.
#
# The weights are conductivity-based, w₁ = κ₂/(κ₁+κ₂) and w₂ = κ₁/(κ₁+κ₂), so
# that w₁κ₁ + w₂κ₂ is the *harmonic* mean of the two conductivities. That is
# the choice that keeps the required β independent of the conductivity ratio;
# with the plain arithmetic average, β would have to grow with κ₂/κ₁.
#
# β must dominate an inverse estimate on the cut cells, so it scales like
# κ p²/h. The factor γ below has to absorb the fact that a cut cell carries
# only a fraction of its volume: the smaller that fraction, the larger the
# constant in the inverse estimate. γ = 20 is comfortable here; the honest way
# to see whether it is large enough is to check that the reported error is at
# the discretisation level and does not improve when γ is raised.

const w₁ = κ₂ / (κ₁ + κ₂)
const w₂ = κ₁ / (κ₁ + κ₂)
const γ = 20.0
const β = γ * (2 * κ₁ * κ₂ / (κ₁ + κ₂)) * order^2 / h

# `sides` names which subdomain the test and the trial function belong to, and
# the kernel is called once per (test, trial) side pair. `onside` picks the
# per-side material, `jump_sign` gives the ±1 that side contributes to a jump.
# The field is scalar, so the test-component argument is unused.
#
# `symmetric = true` is not decoration: the default is `false` precisely so an
# asymmetric law is never silently mirrored, and a symmetric form that forgets
# to declare itself is assembled at twice the cost.
bond = InterfaceForm(; symmetric=true) do q, sides, trial, _component
    n = q.normal
    trial_side = onside(sides.trial, (κ=κ₁, w=w₁), (κ=κ₂, w=w₂))
    test_side = onside(sides.test, (κ=κ₁, w=w₁), (κ=κ₂, w=w₂))
    su, sv = jump_sign(sides.trial), jump_sign(sides.test)
    # Coefficient on the test *value*: consistency term + penalty term.
    value = -sv * trial_side.w * trial_side.κ * dot(trial.gradient, n) + sv * su * β * trial.value
    # Coefficient on the test *gradient*: the adjoint of the consistency term.
    gradient = -test_side.w * test_side.κ * su * trial.value .* n
    TestChannels(value, gradient)
end

# ── Assemble and solve one coupled system ───────────────────────────────────
#
# `couple` returns the four blocks of the jump expansion; they go into the
# problem alongside each subdomain's own bulk stiffness and source. The result
# is one linear system over both dof blocks — never two systems iterated
# against each other.

bcs = [dirichlet(exact₁; on=boundary(axis=1, side=:lower), field=u₁),
       dirichlet(exact₁; on=boundary(axis=2, side=:lower), field=u₁),
       dirichlet(exact₁; on=boundary(axis=2, side=:upper), field=u₁),
       dirichlet(exact₂; on=boundary(axis=1, side=:upper), field=u₂),
       dirichlet(exact₂; on=boundary(axis=2, side=:lower), field=u₂),
       dirichlet(exact₂; on=boundary(axis=2, side=:upper), field=u₂)]

blocks = (stiffness_block(u₁; diffusion=κ₁), stiffness_block(u₂; diffusion=κ₂),
          couple(u₁, u₂, seam, bond)...)
loads = (source_load(u₁; source=source), source_load(u₂; source=source))

problem = Problem((u₁, u₂); blocks=blocks, loads=loads, dirichlet=bcs, symmetric=true)
model = prepare(problem)
sol = solve!(model)

# ── Accuracy ────────────────────────────────────────────────────────────────
#
# `l2_error` is measured per field, over that field's own subdomain and with
# that subdomain's own integration plan — the same regions assembly used, so
# the error is consistent with what was actually integrated.

error₁ = l2_error(sol, model, u₁, exact₁)
error₂ = l2_error(sol, model, u₂, exact₂)

# ── Output ──────────────────────────────────────────────────────────────────

out = joinpath(@__DIR__, "output", "interface_coupling_2d")
write_vtk(out, sol, model; point_data=(u=(uh, c, x, xi) -> uh(c, xi),))

# The headline number is the worse of the two subdomain errors; the shared
# reporter reads it out of the report's `l2_error` slot.
report = merge(diagnostics(model, sol), (; l2_error=max(error₁, error₂)))

print_run_report("Bonded bi-material joint — Nitsche interface coupling", report;
                 parameters=(:seam_position => s, :conductivities => (κ₁, κ₂), :cells_Ω₁ => cells₁,
                             :cells_Ω₂ => cells₂, :order => order, :nitsche_factor => γ,
                             :nitsche_penalty => β), output=out)
println("  interface regions: ", diagnostics(model).interface_region_count)
println("  per-subdomain error (Ω₁ | Ω₂): ", error₁, " | ", error₂)
