#=
Bi-Material Inclusion Corner — Multi-Domain Coupling With The Finite Cell Method
===============================================================================

Reproduction of the geometry of Section 4.2 of

    Y. Elhaddad, N. Zander, T. Bog, L. Kudela, S. Kollmannsberger, J. Kirschke,
    T. Baum, M. Ruess, E. Rank, *Multi-level hp-finite cell method for embedded
    interface problems with application in biomechanics*, Int. J. Numer. Methods
    Biomed. Eng. (2017).

The steady heat-conduction (Poisson) problem on the unit disc `Ω = {‖x‖ ≤ 1}`,
which is split by a 90° sector into two materials:

  * inclusion  `Ω₂` — the first-quadrant sector `{x ≥ 0, y ≥ 0} ∩ disc`,   κ₂ = 10,
  * matrix     `Ω₁` — the remaining 270°,                                   κ₁ = 1,

each governed by `−κᵢ ∇²θ = 1` with `θ = 0` on the outer arc. The straight
material interface `Γ₁₂` (the two radial edges) carries a sharp corner at the
disc centre, inducing a vertex singularity with leading exponent λ₁ ≈ 0.7317;
two further weak singularities sit where the interface meets the outer boundary.

This is the *native multi-domain* demonstration: `Ω₁` and `Ω₂` are discretised on
two INDEPENDENT finite-cell grids (own space, own level-set fold, own dof block),
coupled across the immersed interface by a user-supplied WEIGHTED-NITSCHE term
built with `couple(θ₁, θ₂, Γ, InterfaceForm(...))`, and each subdomain's curved
outer boundary carries a Nitsche weak Dirichlet `block(θᵢ, θᵢ, WeakForm(...);
on=arcᵢ)`. Following the library's contract, NONE of these constitutive terms
live in the package — they are ordinary forms composed here; the package
supplies only the two-sided interface integration (the `InterfaceForm`
mechanism, with `q.normal` and both sides' gradients) and the single-sided
immersed-boundary integration. The coupling is CONSISTENT, so — unlike a pure
penalty — it carries no spurious interface flux jump; the same `InterfaceForm`
is what a cohesive (CZM) traction would later be written against.

Robustness note. The reference's 135° orientation puts the interface edges on
the grid diagonals — genuinely non-grid-aligned. Diagonal cuts leave cut cells
whose interior (bubble) modes are supported only in the fictitious part, which
makes the strict-cut (`α = 0`) stiffness singular. The α-FCM cut-cell rule
(`α_fict > 0`) enriches those cut cells with the α-scaled full-cell quadrature
so every mode is well-posed — the standard FCM remedy, and why the FCM
literature runs with a small nonzero α. The box uses an ODD cell count so any
axis-aligned edges also fall on cell centres.

Accuracy note. The reference strain energy is `U_ex = 1.0168443145 × 10⁻¹`. At
the moderate uniform resolution here the script reports a relative energy error
well under a percent; α-FCM also removes the refinement blow-up the strict cut
would otherwise show. The multi-domain coupling itself is consistent and exact
(machine-precision Nitsche patch test, `test/test_coupling.jl`).

The VTK output below is for visual inspection of the two coupled temperature
fields and the weak material-interface kink between them.
=#

using Unfitted
using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "..", "reporting.jl"))

# ── Parameters ──────────────────────────────────────────────────────────────
const κ₁ = 1.0                       # matrix conductivity
const κ₂ = 10.0                      # inclusion conductivity
const R = 1.0                        # disc radius
const U_ex = 1.0168443145e-1         # reference strain energy (Elhaddad §4.2)

# Resolution is env-overridable so the CI smoke test runs a small, fast case.
# `cells_per_axis` is forced ODD so the interface edges land on cell centres.
const cells_per_axis = let n = parse(Int, get(ENV, "BIC_CELLS", "15"))
    isodd(n) ? n : n + 1
end
const order = parse(Int, get(ENV, "BIC_ORDER", "3"))
const β_factor = 5.0                 # Nitsche factor γ in β = γ · ½(κ₁+κ₂) · p²/h
const write_output = lowercase(get(ENV, "BIC_WRITE_OUTPUT", "true")) in ("1", "true", "yes")

# ── Geometry: unit disc split by a 90° sector (CSG level sets) ──────────────
# The sector is rotated so its bisector points at 180° (edges at 135° and 225°),
# matching the reference figure. Rotating the two half-plane leaves into a frame
# turned by `sector_rotation` places the wedge; the edges are then diagonal —
# genuinely non-grid-aligned (the general unfitted case).
# Sector orientation, degrees — the reference figure's 135°, which puts the
# interface edges on the grid diagonals (genuinely non-grid-aligned, the general
# unfitted case). Diagonal cuts leave cut cells whose interior modes are
# supported only in the fictitious part; the α-FCM stabilisation below (nonzero
# `α_fict`) is what keeps the stiffness well-posed there.
const sector_rotation = deg2rad(parse(Float64, get(ENV, "BIC_ROTATION", "135.0")))
const _cs = cos(sector_rotation)
const _sn = sin(sector_rotation)
rotate_in(x) = SVector(_cs * x[1] + _sn * x[2], -_sn * x[1] + _cs * x[2])   # R(−φ)·x
r_norm(x) = sqrt(x[1]^2 + x[2]^2)
disc = leaf(x -> r_norm(x) - R; lipschitz=1.0)
half_1 = leaf(x -> -rotate_in(x)[1]; lipschitz=1.0)    # {x′ ≥ 0} in the rotated frame
half_2 = leaf(x -> -rotate_in(x)[2]; lipschitz=1.0)    # {y′ ≥ 0}
corner = intersect(half_1, half_2)                     # rotated first quadrant
inclusion_geometry = intersect(disc, corner)           # Ω₂ = sector
matrix_geometry = setdiff(disc, corner)                # Ω₁ = disc ∖ sector

# ── Two independent finite-cell discretisations ────────────────────────────
const box_half = 1.2
const h = 2 * box_half / cells_per_axis
const omega = box((-box_half, -box_half), (box_half, box_half))
subcell = h / 16

# α-FCM fictitious-region weight (Elhaddad uses 10⁻⁹). The cut-cell quadrature
# adds the α-scaled full-cell rule to the physical moment-fit, so cut-cell dofs
# whose basis support lies in the fictitious corner still get a well-posed
# contribution — essential for the non-grid-aligned interface here. `0` is the
# strict cut (fully-outside cells dropped), robust only for grid-aligned cuts.
const α_fict = parse(Float64, get(ENV, "BIC_ALPHA", "1.0e-9"))
matrix_domain = physical_domain(matrix_geometry; alpha=α_fict, subcell_length_scale=subcell,
                                max_depth=4)
inclusion_domain = physical_domain(inclusion_geometry; alpha=α_fict, subcell_length_scale=subcell,
                                   max_depth=4)

V₁ = space(omega; cells=(cells_per_axis, cells_per_axis), order=order, physical=matrix_domain)
V₂ = space(omega; cells=(cells_per_axis, cells_per_axis), order=order, physical=inclusion_domain)
θ₁ = field(:θ₁, V₁)
θ₂ = field(:θ₂, V₂)

# ── Immersed interface and outer-boundary meshes ────────────────────────────
edge_dir(θ) = SVector(cos(θ), sin(θ))
# Interface Γ₁₂: the two rotated radial edges (arc → origin → arc). Traversed so
# the polyline's default normal points from the matrix (Ω₁ = side a) into the
# inclusion (Ω₂ = side b) — the a→b orientation the coupling form assumes.
interface_mesh = polyline_mesh([R .* edge_dir(sector_rotation), SVector(0.0, 0.0),
                                R .* edge_dir(sector_rotation + π / 2)])
# Outer arcs, traversed CCW so their default normal points radially outward.
arc(a, b, n) = polyline_mesh([R .* edge_dir(t) for t in range(a, b; length=n)])
inclusion_arc = arc(sector_rotation, sector_rotation + π / 2, 60)
matrix_arc = arc(sector_rotation + π / 2, sector_rotation + 2π, 180)

# ── Weighted-Nitsche coupling + Nitsche weak Dirichlet (user constitutive) ──
# Consistent (unlike penalty) so no spurious interface flux jump pollutes the
# energy. Flux weights are conductivity-weighted (w₁+w₂=1, favouring the stiffer
# side); β is the Nitsche stabilisation, scaled ∝ κ·p²/h.
const w₁ = κ₂ / (κ₁ + κ₂)
const w₂ = κ₁ / (κ₁ + κ₂)
const β = β_factor * 0.5 * (κ₁ + κ₂) * order^2 / h

# Interface: t̂ = ⟨κ∇θ·n⟩_w − β⟦θ⟧; symmetric adjoint; four field blocks. Per-side
# conductivity/weight is bundled and picked with `onside`; jump signs come from
# `jump_sign`. The field is scalar, so the test-component argument is unused.
nitsche_coupling = InterfaceForm(; symmetric=true) do q, sides, trial, _tc
    n = q.normal
    u = onside(sides.trial, (κ=κ₁, w=w₁), (κ=κ₂, w=w₂))    # trial-side material
    v = onside(sides.test, (κ=κ₁, w=w₁), (κ=κ₂, w=w₂))     # test-side material
    su, sv = jump_sign(sides.trial), jump_sign(sides.test)
    TestChannels(-sv * u.w * u.κ * dot(trial.gradient, n) + sv * su * β * trial.value,
                 -v.w * v.κ * su * trial.value .* n)
end

# Single-sided Nitsche weak Dirichlet (θ = 0) on a subdomain's outer arc.
function nitsche_dirichlet(κ)
    WeakForm(bilinear=(q, trial) -> begin
                 n = q.normal
                 TestChannels(-κ * dot(trial.gradient, n) + β * trial.value, -κ * trial.value .* n)
             end, linear=(q) -> 0.0, symmetric=true)
end

blocks = (stiffness_block(θ₁; diffusion=κ₁), stiffness_block(θ₂; diffusion=κ₂),
          couple(θ₁, θ₂, interface_mesh, nitsche_coupling)...,
          block(θ₁, θ₁, nitsche_dirichlet(κ₁); on=matrix_arc),
          block(θ₂, θ₂, nitsche_dirichlet(κ₂); on=inclusion_arc))
loads = (source_load(θ₁; source=1.0), source_load(θ₂; source=1.0))

problem = Problem((θ₁, θ₂); blocks=blocks, loads=loads, symmetric=true)
model = prepare(problem)
sol = solve!(model)

# ── Strain energy U = ½ Σᵢ κᵢ ∫_Ωᵢ ‖∇θᵢ‖² (bulk stiffness only) ─────────────
bulk = assemble_matrix(model,
                       (stiffness_block(θ₁; diffusion=κ₁), stiffness_block(θ₂; diffusion=κ₂)))
U_h = 0.5 * dot(sol.coefficients, bulk * sol.coefficients)
energy_error = abs(U_h - U_ex) / U_ex

# ── Output ──────────────────────────────────────────────────────────────────
# A coupled model exports natively: `write_vtk` writes one block per field into a
# single multiblock file, each block grouping the field's own data grid (with its
# own level set) and its subdomain mesh wireframe as sibling leaves —
# `θ₁ → {data_1, level_1_base}`, `θ₂ → {data_2, level_2_base}`. Leaf names are
# ASCII (field index / level id) so ParaView's Extract Block isolates each one
# even though the field names are Unicode. Opening `bimaterial_corner.vtm` shows
# the two coupled temperature fields as independently-toggled, self-contained units.
out = joinpath(@__DIR__, "output")
if write_output
    write_vtk(joinpath(out, "bimaterial_corner"), sol, model;
              point_data=(θ=(u, c, x, xi) -> u(c, xi),))
    write_quadrature_vtm(joinpath(out, "bimaterial_quadrature"), model)
end

# ── Report ──────────────────────────────────────────────────────────────────
diag = diagnostics(model)
println("\n", "="^72)
println(" Bi-material inclusion corner (Elhaddad §4.2) — native multi-domain coupling")
println("="^72)
println(rpad(" cells / order", 34), cells_per_axis, "×", cells_per_axis, "  /  order ", order)
println(rpad(" conductivities κ₁, κ₂", 34), κ₁, ", ", κ₂)
println(rpad(" penalty factor γ (β=γκ/h)", 34), β_factor)
println(rpad(" active unknowns (θ₁ + θ₂)", 34), active_unknowns(model))
println(rpad(" integration regions", 34), diag.integration_regions)
println(rpad(" cut regions / small overlaps", 34), diag.cut_region_count, " / ",
        diag.small_overlap_count)
println(rpad(" moment-fit residual (max)", 34), round(diag.moment_fit_residual_max, sigdigits=3))
println("-"^72)
println(rpad(" strain energy  U_h", 34), round(U_h, digits=7))
println(rpad(" reference      U_ex", 34), U_ex)
println(rpad(" relative energy error", 34), round(100 * energy_error, digits=3), " %")
println("="^72)
if write_output
    println(" VTK written to ", out, "/  (open bimaterial_corner.vtm — both coupled fields)")
end
