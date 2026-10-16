#=
Plate With A Narrow Slit — A Displacement That Has To Jump
==========================================================

A square plate in tension carries a slit much narrower than the cells of its grid. Under the load the slit opens, so
the displacement jumps across it — and a plain finite-cell discretisation cannot let it.

The plate is the unit square minus a rounded slit 0.031 wide and 0.562 long: the rectangle `0.459 ≤ x ≤ 0.490`,
`|y − ½| ≤ 0.2655`, closed by two half-discs of radius 0.0155. It is linear elastic in plane stress (`E = 1000`,
`ν = 0.3`), held by rollers (`u_x = 0` on `x = 0`, `u_y = 0` on `y = 0` and `y = 1`) and pulled by the traction
`t = (100, 0)` on `x = 1`. The slit is traction free, which costs nothing: a cut cell only integrates its material.

**Why plain finite cells fail.** On a `5 × 5` grid the slit lies inside three cells of the middle column. Each of them
carries one polynomial over its whole box, so the two faces of the slit, 0.031 apart, see the same smooth function and
cannot move apart. The plate comes out far too stiff, and raising the order hardly helps.

**The strip.** One overlay over the three cut cells, `(0.4, 0.6) × (0.2, 0.8)`, with ten columns of cells 0.02 wide.
Its column `0.46 ≤ x ≤ 0.48` lies inside the slit, so along the slit's straight part the fictitious fold switches that
column off and cuts the overlay in two: its functions on the left of the slit and on the right share no cell and can
take different values on the two faces. In the overlay's top and bottom rows the same column holds the material just
beyond the slit's ends and joins the two halves there, as the material does. The covered base cells keep only their
linear skeleton, which the overlay corrects on each side on its own.

**What is printed.** For `p = 1 … 8` the relative energy-norm error `e = √((U_ref − U_h) / U_ref)` of both
discretisations, where `U_h = ½ fᵀu_h` is the discrete strain energy, and the slit's opening at mid-height (the jump of
`u_x` across it at `y = ½`). Plain finite cells stay near `e ≈ 0.61` and open the slit by less than a tenth of the
reference; the strip reaches `e ≈ 0.04` and opens it to within 0.3 % at `p = 8`.

**Reference.** No closed form is known. `U_ref = 7.607295680241` and the opening 0.137905 were computed with a `ladder`
over the same grid, refined around the slit and towards its caps (down to `h = 0.0031`) and raised to `p = 8`; the
energy converges geometrically to about 1e-11, and plain finite cells on a uniform `80 × 80` grid agree to 5e-9.

The configuration is the eccentric-slit benchmark of

    J. Parvizian, "Discontinuous Finite Cell Method for problems of solid mechanics", manuscript, Figure 18,

which separates the two sides of the slit by splitting the cut cells; here an ordinary overlay and the fold do it.
=#

using Unfitted
using WriteVTK              # loads UnfittedWriteVTKExt, the ParaView writer
using LinearAlgebra
using Tensors

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

const U_ref = 7.607295680241        # reference strain energy
const opening_ref = 0.137905        # reference slit opening at y = ½

# ── Geometry ────────────────────────────────────────────────────────────────

# The slit is a rectangle and two caps; the plate is its complement. Each leaf is a signed distance (Lipschitz 1), and
# keeping them separate keeps every piece of the boundary smooth for the cut-cell quadrature.
plane(f) = leaf(f; lipschitz=1.0)
rectangle = intersect(plane(x -> 0.459 - x[1]), plane(x -> x[1] - 0.490), plane(x -> 0.2345 - x[2]),
                      plane(x -> x[2] - 0.7655))
cap(y) = plane(x -> hypot(x[1] - 0.4745, x[2] - y) - 0.0155)
slit = union(rectangle, cap(0.2345), cap(0.7655))

plate = physical_domain(complement(slit); subcell_length_scale=1 / 160, max_depth=6)
omega = box((0.0, 0.0), (1.0, 1.0))

# ── Plane-stress elasticity ─────────────────────────────────────────────────

# Hooke's law σ = λ tr(ε) 𝑰 + 2μ ε, with the plane-stress λ = Eν/(1 − ν²) in place of the three-dimensional one.
const E, ν = 1000.0, 0.3
const λ, μ = E * ν / (1 - ν^2), E / (2 * (1 + ν))
const 𝑰 = one(SymmetricTensor{2,2})
σ(ε) = λ * tr(ε) * 𝑰 + 2μ * ε
elasticity(q, trial, c) = TestChannels(0.0, σ(symmetric_gradient(trial)), c)

# Rollers on three sides, the traction on x = 1; the discrete strain energy is ½ fᵀu_h.
function solve_plate(V)
    u = field(:u, V; components=2)
    stiffness = block(u, u, WeakForm(bilinear=elasticity, symmetric=true, component_aware=true))
    traction = neumann(u, 100.0; on=boundary(axis=1, side=:upper), component=1)
    rollers = [dirichlet(0.0; on=boundary(axis=1, side=:lower), component=1),
               dirichlet(0.0; on=boundary(axis=2, side=:lower), component=2),
               dirichlet(0.0; on=boundary(axis=2, side=:upper), component=2)]
    model = prepare(Problem((u,); blocks=(stiffness,), loads=(traction,), dirichlet=rollers))
    solution = solve!(model)
    return (; u, model, solution, energy=dot(model.rhs, solution.coefficients) / 2)
end

energy_error(run) = sqrt(max(U_ref - run.energy, 0.0) / U_ref)
unknowns(run) = length(run.solution.coefficients)

# The jump of u_x across the slit at y = ½, sampled just outside its two faces.
ux(run, x) = value(run.solution, run.model, (x, 0.5))[1]
opening(run) = ux(run, 0.490 + 1e-9) - ux(run, 0.459 - 1e-9)

# ── The two discretisations ─────────────────────────────────────────────────

fcm(p) = space(omega; cells=5, order=p, physical=plate)

# Ten columns of 0.02 over the three cut cells: the one between x = 0.46 and x = 0.48 lies inside the slit.
with_strip(p) = overlay(fcm(p), box((0.4, 0.2), (0.6, 0.8)); cells=(10, 3), order=p)

# ── p-refinement ────────────────────────────────────────────────────────────

const max_order = env("SLIT_MAX_ORDER", 8)

println("Plate with a narrow slit — p-refinement on the 5 × 5 grid")
println("  ", rpad("p", 4), rpad("FCM unknowns", 14), rpad("error", 10), rpad("opening", 12), "│ ",
        rpad("strip unknowns", 16), rpad("error", 10), "opening")
runs = map(1:max_order) do p
    plain, strip = solve_plate(fcm(p)), solve_plate(with_strip(p))
    println("  ", rpad(p, 4), rpad(unknowns(plain), 14), rpad(sig(energy_error(plain)), 10),
            rpad(sig(opening(plain)), 12), "│ ", rpad(unknowns(strip), 16), rpad(sig(energy_error(strip)), 10),
            sig(opening(strip)))
    (plain, strip)
end
plain, strip = last(runs)
println()
println("energy error, FCM: ", energy_error(plain))
println("energy error, strip: ", energy_error(strip))
println("slit opening, FCM: ", opening(plain))
println("slit opening, strip: ", opening(strip))
println("slit opening, reference: ", opening_ref)
println()

# ── Output ──────────────────────────────────────────────────────────────────

# Both displacement fields, side by side in ParaView: one opens the slit, the other barely does.
out = joinpath(@__DIR__, "output", "slit_plate_2d")
for (path, run) in ((out, strip), (out * "_fcm", plain))
    u = run.u
    write_vtk(path, run.solution, run.model; point_data=(uh=(u, c, x, xi) -> u(c, xi),))
end

print_run_report("Plate with a narrow slit — 5 × 5 grid with the strip, order $max_order",
                 diagnostics(strip.model, strip.solution);
                 parameters=(:cells => (5, 5), :strip => "(0.4, 0.6) × (0.2, 0.8), cells (10, 3)", :order => max_order,
                             :E => E, :nu => ν, :U_ref => U_ref), output=out)
