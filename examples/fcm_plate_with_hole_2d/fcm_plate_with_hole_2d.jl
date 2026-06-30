#=
Infinite Plate With A Circular Hole (Kirsch) — FCM p-Refinement Study
====================================================================

The classic Kirsch benchmark: an infinite plate in plane stress with a
centred circular hole of radius `a`, loaded by a uniform far-field
traction `T` in the x-direction. By the two symmetry planes of the
problem we model only the upper-right quarter,

    Ω = { x ∈ (0, L)² : ‖x‖ ≥ a },     a = 1,   L = 4,   T = 10,

embedded in the cartesian extension box `(0, L)²`. The material lives
*outside* the hole, so the immersed domain is the single smooth level
set

    Ω = { φ ≤ 0 },     φ(x) = a − ‖x‖,

i.e. one `leaf(x -> a − hypot(x...))`. The bulk is integrated by the
finite-cell method (level-set classification + exact implicit-quadrature
moment fit on the cut cells along the quarter arc).

This benchmark is deliberately chosen to isolate the *integration*
pipeline: unlike the annular-plate example it needs **no weak boundary
conditions**. The hole arc `‖x‖ = a` is a free surface, which the FCM
delivers for free (cut cells contribute only the material side, so the
immersed boundary is naturally traction-free). The two cartesian
symmetry faces and the two far-field faces are grid-aligned, so every
boundary condition is imposed in the standard strong / Neumann way:

  * **Left face `x = 0`** — symmetry plane ⇒ strong `u_x = 0`
    (component-1 Dirichlet); the tangential traction vanishes naturally.
  * **Bottom face `y = 0`** — symmetry plane ⇒ strong `u_y = 0`
    (component-2 Dirichlet).
  * **Right face `x = L`** and **top face `y = L`** — the exact Kirsch
    traction `t = σ·n` applied as an inhomogeneous Neumann load.

The exact (infinite-plate) stress field, in polar coordinates centred on
the hole with `c₂ = cos 2θ`, `s₂ = sin 2θ`, is

    σ_rr = T/2 (1 − a²/r²) + T/2 (1 − 4a²/r² + 3a⁴/r⁴) c₂,
    σ_θθ = T/2 (1 + a²/r²) − T/2 (1 + 3a⁴/r⁴) c₂,
    σ_rθ = −T/2 (1 + 2a²/r² − 3a⁴/r⁴) s₂,

and the exact displacement field (plane stress, Kolosov constant
`κ = (3−ν)/(1+ν)`, shear modulus `μ = E / 2(1+ν)`) is

    u_x = T a/8μ [ (r/a)(κ+1) cosθ + 2a/r ((1+κ) cosθ + cos 3θ)
                                    − 2a³/r³ cos 3θ ],
    u_y = T a/8μ [ (r/a)(κ−3) sinθ + 2a/r ((1−κ) sinθ + sin 3θ)
                                    − 2a³/r³ sin 3θ ].

This field has `u_x = 0` on `x = 0` and `u_y = 0` on `y = 0`, so the
strong symmetry constraints are consistent with the analytic solution,
and `u → (T/E) x`, `−(νT/E) y` as `r → ∞` (the uniaxial far field).

Because Ω is smooth, the displacement is analytic on Ω, and the
exact-moment kernel integrates the curved cut cells to high order, the
discrete solution converges towards the analytic field under pure global
`p`-refinement on a *fixed* mesh. The loop below sweeps the polynomial
order and prints the relative L² displacement error; on the default
`8 × 8` mesh it falls cleanly by roughly an order of magnitude per degree
— spectral `p`-convergence down to `~4·10⁻⁸` at `p = 7` — which
demonstrates the integration pipeline is on rate against a curved
immersed boundary. (A coarser mesh such as `4 × 4` stalls earlier: the
corner cell is then almost entirely the hole, leaving a thin
high-gradient material sliver that one polynomial patch cannot resolve.)

Reference: G. Kirsch, *Die Theorie der Elastizität und die Bedürfnisse
der Festigkeitslehre*, Z. Ver. Dtsch. Ing. 42 (1898) 797–807. The
displacement form is the standard infinite-plate solution used as an
isogeometric / finite-cell verification benchmark.
=#

using Unfitted
using LinearAlgebra
using StaticArrays
using Tensors

include(joinpath(@__DIR__, "..", "reporting.jl"))

# ── Material, geometry, and loading ─────────────────────────────────────────

const E = 1.0e3         # Young's modulus
const ν = 0.3           # Poisson's ratio (plane stress)
const a = 1.0           # hole radius
const L = 4.0           # quarter-domain edge length
const T = 10.0          # far-field traction Tₓ

const μ = E / (2 * (1 + ν))             # shear modulus
const λ_ps = E * ν / (1 - ν^2)          # plane-stress Lamé-like coefficient
const κ = (3 - ν) / (1 + ν)             # Kolosov constant (plane stress)

# Immersed domain Ω = { φ ≤ 0 } with φ = a − ‖x‖: the material is everything
# *outside* the hole. One smooth circle leaf — the implicit-quadrature kernel
# integrates the curved cut cells along the quarter arc at high order.
r_norm(x) = sqrt(x[1]^2 + x[2]^2)
plate_geometry = leaf(x -> a - r_norm(x); lipschitz=1.0)

# ── Exact Kirsch fields (stress for tractions, displacement for L² error) ───

# Polar Kirsch stresses (σ_rr, σ_θθ, σ_rθ) plus the polar angle θ. Valid for
# r ≥ a; the L² error integrates over Ω only, so r ≥ a always.
function kirsch_polar_stress(x)
    r = r_norm(x)
    θ = atan(x[2], x[1])
    a2 = (a / r)^2
    a4 = a2^2
    c2 = cos(2θ)
    s2 = sin(2θ)
    σrr = T / 2 * (1 - a2) + T / 2 * (1 - 4 * a2 + 3 * a4) * c2
    σθθ = T / 2 * (1 + a2) - T / 2 * (1 + 3 * a4) * c2
    σrθ = -T / 2 * (1 + 2 * a2 - 3 * a4) * s2
    return σrr, σθθ, σrθ, θ
end

# Cartesian Cauchy stress (σ_xx, σ_yy, σ_xy), obtained by rotating the polar
# tensor by the local angle θ.
function kirsch_stress(x)
    σrr, σθθ, σrθ, θ = kirsch_polar_stress(x)
    c = cos(θ)
    s = sin(θ)
    σxx = σrr * c^2 + σθθ * s^2 - 2 * σrθ * s * c
    σyy = σrr * s^2 + σθθ * c^2 + 2 * σrθ * s * c
    σxy = (σrr - σθθ) * s * c + σrθ * (c^2 - s^2)
    return SVector(σxx, σyy, σxy)
end

# Exact traction t = σ·n on the far-field faces. Right face: n = (1, 0) ⇒
# t = (σxx, σxy). Top face: n = (0, 1) ⇒ t = (σxy, σyy).
traction_right(x) =
    let s = kirsch_stress(x)
        SVector(s[1], s[3])
    end
traction_top(x) =
    let s = kirsch_stress(x)
        SVector(s[3], s[2])
    end

# Exact plane-stress displacement field (the reference for the L² error and the
# VTK overlay).
function exact(x)
    r = r_norm(x)
    θ = atan(x[2], x[1])
    c = cos(θ)
    s = sin(θ)
    c3 = cos(3θ)
    s3 = sin(3θ)
    pref = T * a / (8 * μ)
    ux = pref * ((r / a) * (κ + 1) * c + (2 * a / r) * ((1 + κ) * c + c3) - (2 * a^3 / r^3) * c3)
    uy = pref * ((r / a) * (κ - 3) * s + (2 * a / r) * ((1 - κ) * s + s3) - (2 * a^3 / r^3) * s3)
    return SVector(ux, uy)
end

# ── Plane-stress elasticity bulk form ───────────────────────────────────────
#
# 4th-order isotropic plane-stress tensor ℂ[i,j,k,l] = λ* δ_ij δ_kl
# + μ (δ_ik δ_jl + δ_il δ_jk), with (λ*, μ) = (Eν/(1−ν²), E/2(1+ν)); the
# double contraction σ = ℂ ⊡ ε then reproduces σ_xx = E/(1−ν²)(ε_xx + ν ε_yy),
# σ_yy = E/(1−ν²)(ε_yy + ν ε_xx), σ_xy = E/(1+ν) ε_xy.
const ℂ = SymmetricTensor{4,2,Float64}((i, j, k, l) -> λ_ps * (i == j) * (k == l) +
                                                       μ *
                                                       ((i == k) * (j == l) + (i == l) * (j == k)))

# Bulk bilinear σ(u) : ε(v). The `c`-th channel of the contracted stress is its
# `c`-th row (σ symmetric ⇒ row = column).
function elasticity_bilinear(q, trial, c)
    σ = ℂ ⊡ symmetric_gradient(trial)
    return TestChannels(0.0, Vec{2,Float64}((σ[c, 1], σ[c, 2])))
end

# ── Discretization and a single p-refinement run ────────────────────────────

const cells_per_axis = 8
const orders = 1:7

omega = box((0.0, 0.0), (L, L))

# One end-to-end solve at polynomial `order`. The geometry, BCs, and exact
# reference are identical across orders; only the trial/test degree changes,
# so the L² error reflects pure p-refinement.
function run_order(order)
    cell_size = L / cells_per_axis
    plate = physical_domain(plate_geometry; subcell_length_scale=cell_size / 2^4, max_depth=4)
    V = space(omega; cells=(cells_per_axis, cells_per_axis), order=order, physical=plate)
    u = field(:u, V; components=2)

    elasticity = block(u, u,
                       WeakForm(bilinear=elasticity_bilinear, linear=(q, c) -> 0.0, symmetric=true,
                                component_aware=true))

    # Far-field Neumann tractions on the grid-aligned right and top faces.
    load_right = neumann(u, traction_right; on=boundary(axis=1, side=:upper))
    load_top = neumann(u, traction_top; on=boundary(axis=2, side=:upper))

    # Strong symmetry: u_x = 0 on the left face, u_y = 0 on the bottom face.
    bcs = [dirichlet(0.0; on=boundary(axis=1, side=:lower), component=1),
           dirichlet(0.0; on=boundary(axis=2, side=:lower), component=2)]

    problem = Problem((u,); blocks=(elasticity,), loads=(load_right, load_top), dirichlet=bcs)
    model = prepare(problem)
    solution = solve!(model)
    report = diagnostics(model, solution; exact)
    return (; u, V, plate, model, solution, report)
end

# ── p-Refinement sweep ──────────────────────────────────────────────────────

results = map(run_order, orders)

println("Kirsch plate with circular hole — p-refinement (cells = ",
        (cells_per_axis, cells_per_axis), ")")
println("  ", rpad("order", 8), rpad("unknowns", 12), rpad("rel. L² error", 16),
        "moment-fit residual")
for (order, res) in zip(orders, results)
    rep = res.report
    resid = hasproperty(rep, :moment_fit_residual_max) ? rep.moment_fit_residual_max : NaN
    println("  ", rpad(order, 8), rpad(rep.active_unknowns, 12),
            rpad(string(round(rep.l2_error; sigdigits=4)), 16), round(resid; sigdigits=3))
end
println()

# ── Output: detailed report + VTK at the finest order ───────────────────────

finest = last(results)
u = finest.u

out = joinpath(@__DIR__, "output", "fcm_plate_with_hole_2d")
write_vtk(out, finest.solution, finest.model;
          point_data=(uh=(u, c, x, xi) -> u(c, xi), exact=(u, c, x, xi) -> exact(x),
                      displacement_error=(u, c, x, xi) -> norm(u(c, xi) - exact(x))),)
write_quadrature_vtm(out * "_quadrature", finest.model)

print_run_report("FCM plate with circular hole (Kirsch) — order $(last(orders))", finest.report;
                 parameters=(:cells => finest.V.levels[1].mesh.cells,
                             :order => finest.V.levels[1].order, :hole_radius => a,
                             :domain_edge => L, :far_field_traction => T, :E => E, :nu => ν,
                             :geometry => "Ω = { ‖x‖ ≥ a } (single circle leaf)",
                             :subcell_length_scale => finest.plate.subcell_length_scale,
                             :max_depth => finest.plate.max_depth), output=out)
