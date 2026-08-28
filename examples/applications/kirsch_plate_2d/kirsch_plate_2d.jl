#=
Plate With A Circular Hole — Verification Against The Kirsch Solution
====================================================================

A stress-analysis problem you would recognise from a first course in
elasticity, solved on a grid that ignores the hole entirely.

An infinite plate in plane stress carries a centred circular hole of
radius `a` and is pulled by a uniform far-field traction `T` along `x`.
The problem has two symmetry planes, so only the upper-right quarter is
modelled,

    Ω = { x ∈ (0, L)² : ‖x‖ ≥ a },     a = 1,   L = 4,   T = 10.

The interesting modelling decision is that the mesh is a plain `8 × 8`
Cartesian grid over the square `(0, L)²` — it knows nothing about the
hole. The hole enters as a level set instead: the material is the region
where `φ(x) = a − ‖x‖` is non-positive, one `leaf(x -> a - ‖x‖)`, and the
finite-cell machinery classifies each cell as inside, outside, or cut and
builds an exact moment-fit quadrature rule on the cut ones. Nothing about
the grid has to change if the hole moves or changes radius, which is the
practical reason to work this way.

Boundary conditions come out of the geometry split:

  * The hole arc `‖x‖ = a` is **traction free**, and that costs nothing:
    a cut cell only integrates the material side, so a free immersed
    surface is what you get by not writing anything down.
  * `x = 0` and `y = 0` are symmetry planes ⇒ strong `u_x = 0` and
    `u_y = 0` respectively. Both faces are grid-aligned, so these are
    ordinary component-wise Dirichlet conditions.
  * `x = L` and `y = L` carry the exact Kirsch traction `t = σ·n` as an
    inhomogeneous Neumann load — the far field, truncated at a finite
    distance.

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

That field has `u_x = 0` on `x = 0` and `u_y = 0` on `y = 0`, so the
symmetry constraints are consistent with it, and `u → ((T/E) x, −(νT/E) y)`
as `r → ∞`.

**What the printed metric means.** The script sweeps the polynomial order
`p = 1 … 7` on the *same fixed* `8 × 8` mesh and prints the relative L²
displacement error against the analytic field, plus the largest moment-fit
residual over all cut cells. Ω is smooth, so the exact solution is analytic
on it and the error should fall by roughly an order of magnitude per degree
— spectral p-convergence, reaching about `3·10⁻⁸` at `p = 7`. A run that
stalls one or two decades early is the signature of under-resolved cut-cell
integration, not of the elasticity: watch the moment-fit residual column,
which stays near machine precision when the geometry is being integrated
properly. (A coarser mesh such as `4 × 4` genuinely does stall: the corner
cell is then mostly hole, leaving a thin high-gradient material sliver that
one polynomial patch cannot resolve.)

Reference: G. Kirsch, *Die Theorie der Elastizität und die Bedürfnisse
der Festigkeitslehre*, Z. Ver. Dtsch. Ing. **42** (1898) 797–807. The
displacement form above is the standard infinite-plate solution used as an
isogeometric / finite-cell verification benchmark.
=#

using Unfitted
using LinearAlgebra
using StaticArrays
using Tensors

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

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

# Bulk bilinear σ(u) : ε(v), written in tensor notation throughout.
# `symmetric_gradient(trial)` is the strain ε of the trial basis function
# currently being assembled, so `ℂ ⊡ ε` is an ordinary constitutive
# evaluation — swap ℂ for any other law and nothing else here changes.
#
# Assembly contracts the returned gradient coefficient against the gradient
# of the *scalar* test function of component `c`, i.e. against `∇v = e_c ⊗ ∇N`,
# and `σ : ∇v = Σⱼ σ[c, j] ∂ⱼN`. The three-argument `TestChannels` constructor
# takes the whole stress tensor and the test component and picks row `c`
# itself, so the row extraction never has to be spelled out by hand — and the
# code stays dimension-agnostic instead of naming the two 2-D slots.
elasticity_bilinear(q, trial, c) = TestChannels(0.0, ℂ ⊡ symmetric_gradient(trial), c)

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

println("Plate with a circular hole — p-refinement on a fixed grid (cells = ",
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

out = joinpath(@__DIR__, "output", "kirsch_plate_2d")
write_vtk(out, finest.solution, finest.model;
          point_data=(uh=(u, c, x, xi) -> u(c, xi), exact=(u, c, x, xi) -> exact(x),
                      displacement_error=(u, c, x, xi) -> norm(u(c, xi) - exact(x))))
write_quadrature_vtm(out * "_quadrature", finest.model)

print_run_report("Plate with a circular hole (Kirsch) — order $(last(orders))", finest.report;
                 parameters=(:cells => finest.V.levels[1].mesh.cells,
                             :order => finest.V.levels[1].order, :hole_radius => a,
                             :domain_edge => L, :far_field_traction => T, :E => E, :nu => ν,
                             :geometry => "Ω = { ‖x‖ ≥ a } (single circle leaf)",
                             :subcell_length_scale => finest.plate.subcell_length_scale,
                             :max_depth => finest.plate.max_depth), output=out)
