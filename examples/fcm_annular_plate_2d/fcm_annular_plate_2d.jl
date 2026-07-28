#=
Plane-Stress Annular Plate With FCM And Weak Boundary Conditions
================================================================

Reproduction of section 4.2 of

    M. Ruess, D. Schillinger, Y. Bazilevs, V. Varduhn, E. Rank,
    *Weakly Enforced Essential Boundary Conditions for NURBS-embedded
    and trimmed NURBS geometries on the basis of the Finite Cell
    Method*, Int. J. Numer. Methods Engng. 95 (2013) 811–846.

The benchmark is a plane-stress annular plate `Ω = { 0.25 ≤ ‖x‖ ≤ 1 }`
embedded in the cartesian extension domain `[−1, 1]²`. The bulk is
integrated by the finite-cell method (level-set classification + NNMF
moment-fit on cut cells); the immersed boundaries `∂Ω` are described
by two user-supplied closed polylines (outer arc and inner arc) that
the package automatically subdivides against the level grids.

Material and analytical reference (linear elasticity, plane stress):

    E = 1,    ν = 0,    r_i = 0.25,    r_o = 1.

A radial body load and an inner-radius radial traction conspire to
produce the analytic axisymmetric solution

    u_r(r) = −r · ln(r) / (2 ln 2),

with stresses

    σ_rr(r) = −(ln r + 1) / (2 ln 2),
    σ_θθ(r) = −ln(r) / (2 ln 2),

and body force

    b(x) = x / (‖x‖² · ln 2)        (Cartesian, derived from
                                     equilibrium in polar form).

Boundary conditions imposed weakly:

  * **Outer arc Γ_o** (`r = 1`): homogeneous Dirichlet `u = 0`
    enforced by the symmetric Nitsche formulation for vector
    elasticity:

        N(u, v) = ∫_Γ_o [−(σ(u)·n)·v − u·(σ(v)·n) + (β/h) u·v] dS.

  * **Inner arc Γ_i** (`r = 0.25`): prescribed radial traction
    `t = −σ_rr(r_i) · r̂` (pointing toward the centre) as a Neumann
    load `∫_Γ_i t·v dS`.

`u_r(r_o) = 0` by construction, so the Nitsche RHS vanishes.

At `cells = (8, 8)`, `order = 4`, `subcell_length_scale ≈ 0.0156` (≈ `cell/16`) this script
reports relative L² error around `6 · 10⁻⁴` against the analytic
displacement — well below the 1% mark Ruess targets in the energy
norm at the same `8 × 8` discretization. The L² convergence of the
displacement field is one order faster than the energy-norm rate, so
matching `e_E ≈ 0.5%` corresponds to `e_L² ≈ 10⁻⁴`, which we hit.

The example validates that the immersed-boundary integration
machinery (auto-subdivision of the user polylines against the
cartesian grid + Nitsche + Neumann composition) is on rate against a
non-trivial benchmark.
=#

using Unfitted
using LinearAlgebra
using StaticArrays
using Tensors

include(joinpath(@__DIR__, "..", "reporting.jl"))

# ── Material, geometry, and analytic reference ──────────────────────────────

const E = 1.0
const r_inner = 0.25
const r_outer = 1.0

# Annulus `Ω = { r_i ≤ ‖x‖ ≤ r_o }` as a CSG intersection of two smooth disk
# level sets, `‖x‖ − r_o ≤ 0` and `r_i − ‖x‖ ≤ 0`. Carrying the two circle
# boundaries separately — rather than collapsing them into a single
# `max(r_i − ‖x‖, ‖x‖ − r_o)`, whose mid-radius crease has a discontinuous
# gradient — lets the implicit-quadrature kernel integrate both boundaries at
# high order (machine-accurate moments on the smooth rim cut cells).
r_norm(x) = sqrt(x[1]^2 + x[2]^2)
annulus_geometry = intersect(leaf(x -> r_norm(x) - r_outer; lipschitz=1.0),
                             leaf(x -> r_inner - r_norm(x); lipschitz=1.0))

# Analytic displacement: u(x) = u_r(r) · r̂ = -ln(r²)·x / (4 ln 2).
exact(x) =
    let r2 = x[1]^2 + x[2]^2
        -log(r2) * SVector(x[1], x[2]) / (4 * log(2))
    end

# Inner-radius radial stress σ_rr(r_i). The traction on the inner arc is
# `t = -σ_rr(r_i) · r̂` (the outward normal of the annulus on the inner
# arc points toward the centre, so the radial component of the
# traction is the *negative* of the radial stress).
const sigma_rr_inner = -(log(r_inner) + 1) / (2 * log(2))

# Cartesian body force `b(x) = x / (r² · ln 2)`, derived from the
# polar equilibrium `σ_rr' + (σ_rr − σ_θθ)/r + b_r = 0` evaluated at
# the analytic stress field.
body_force(x) =
    let r2 = x[1]^2 + x[2]^2
        SVector(x[1], x[2]) / (r2 * log(2))
    end

# ── Discretization ──────────────────────────────────────────────────────────

const cells_per_axis = 8
const order = 4
const box_extent = 1.0
const h_nitsche = 2 * box_extent / cells_per_axis    # base cell size

omega = box((-box_extent, -box_extent), (box_extent, box_extent))
# `subcell_length_scale` now drives only the cut/full/fictitious classifier and
# the kernel's subdivision budget on any non-graph-like cell; the smooth rim cut
# cells get exact, depth-independent moments regardless. Base cell size = 0.25,
# octree leaves of ~0.0156 (cell / 16).
annulus = physical_domain(annulus_geometry;
                          subcell_length_scale=2 * box_extent / cells_per_axis / 2^4, max_depth=4)
V = space(omega; cells=(cells_per_axis, cells_per_axis), order=order, physical=annulus)
u = field(:u, V; components=2)

# ── User-supplied immersed boundary meshes ──────────────────────────────────
#
# Outer arc `r = r_o`: traversed CCW so the polyline's default normal
# (90° clockwise from the segment direction) points outward — away
# from the centre, which is the outward normal of the annulus on the
# outer edge.
#
# Inner arc `r = r_i`: traversed CW so the default normal points
# toward the centre — the annulus' outward normal on the inner edge.
#
# The arcs do *not* need to align with the cartesian grid; the package
# subdivides each segment against every level's grid automatically
# before building surface regions.

const n_outer = 400
const n_inner = 200

arc_ccw(r, n) = [SVector(r * cos(t), r * sin(t)) for t in range(0.0, 2π; length=n + 1)[1:n]]
arc_cw(r, n) = [SVector(r * cos(-t), r * sin(-t)) for t in range(0.0, 2π; length=n + 1)[1:n]]

const outer_arc = polyline_mesh(arc_ccw(r_outer, n_outer); closed=true)
const inner_arc = polyline_mesh(arc_cw(r_inner, n_inner); closed=true)

# ── Bulk weak form: 2D linear elasticity, plane stress, ν = 0 ──────────────
#
# 4th-order plane-stress elasticity tensor. For ν = 0 the isotropic
# split reduces to (λ, μ) = (0, E/2):
#
#     ℂ[i,j,k,l] = (E/2) (δ_ik δ_jl + δ_il δ_jk),
#
# giving σ_11 = E ε_11, σ_22 = E ε_22, σ_12 = E ε_12 (tensor εxy, so
# the standard `σ = ℂ ⊡ ε` evaluates the engineering-shear factor
# automatically through the double contraction).
const ℂ = SymmetricTensor{4,2,Float64}((i, j, k, l) -> (E / 2) *
                                                       ((i == k) * (j == l) + (i == l) * (j == k)))

# Bulk bilinear: σ(u) : ε(v). The `c`-th channel of the contracted
# stress is its `c`-th row, picked out as the gradient coefficient of
# `TestChannels` (σ symmetric ⇒ row = column).
function elasticity_bilinear(q, trial, c)
    σ = ℂ ⊡ symmetric_gradient(trial)
    return TestChannels(0.0, Vec{2,Float64}((σ[c, 1], σ[c, 2])))
end

const elasticity_block = block(u, u,
                               WeakForm(bilinear=elasticity_bilinear, linear=(q, c) -> 0.0,
                                        symmetric=true, component_aware=true))

# ── Body force load ─────────────────────────────────────────────────────────
#
# Component-aware: returns the `c`-th component of the body force at
# the quadrature point.

const body_load = loadform(u,
                           WeakForm(bilinear=(q, trial, c) -> 0.0,
                                    linear=(q, c) -> body_force(q.x)[c], symmetric=true,
                                    component_aware=true))

# ── Inner arc: Neumann traction ─────────────────────────────────────────────
#
# `t(x) = -σ_rr(r_i) · x / r_i` — radial direction, magnitude
# `σ_rr(r_i)`, pointing toward the centre.

const inner_traction = loadform(u,
                                WeakForm(bilinear=(q, trial, c) -> 0.0,
                                         linear=(q, c) -> -sigma_rr_inner * q.x[c] / r_inner,
                                         symmetric=true, component_aware=true); on=inner_arc)

# ── Outer arc: weak Dirichlet via symmetric Nitsche ─────────────────────────
#
# Bilinear-only (homogeneous Dirichlet `u = 0` → no RHS contribution).
# For trial component `a`, test component `c`, normal `n`, the three
# Nitsche pieces are:
#
#   1. `-(σ(u_h)·n)_c · v_c`           → value channel `-T_{a,c}`
#   2. `-u_h · (σ(v_h)·n)`             → gradient channel
#                                        `-trial.value · M_a`
#   3. `(β/h) δ_{a,c} · ψ_a · v_c`     → value channel `+(β/h)·trial.value`
#                                        on diagonal pairs.
#
# The trial traction comes from `σ(u_trial) · n` with `σ = ℂ ⊡ ε(u_trial)`;
# the test-side gradient coefficient `M_a` is the `c`-th column of
# `ℂ ⊡ sym(e_a ⊗ n)`, derived via the major + minor symmetries of ℂ.

function nitsche_bilinear(q, trial, c)
    a = trial.component
    n = Vec{2,Float64}((q.normal[1], q.normal[2]))
    ψ = trial.value

    σ_trial = ℂ ⊡ symmetric_gradient(trial)
    T_uc = (σ_trial ⋅ n)[c]                            # (σ(u_trial)·n)_c

    e_a = Vec{2,Float64}(ntuple(i -> Float64(i == a), 2))
    σ_an = ℂ ⊡ symmetric(e_a ⊗ n)                       # 2-tensor; M_a is its c-th column
    M_a = Vec{2,Float64}((σ_an[1, c], σ_an[2, c]))

    penalty = a == c ? (nitsche_beta / h_nitsche) * ψ : 0.0
    return TestChannels(-T_uc + penalty, -ψ * M_a)
end

const nitsche_beta = 100.0

const nitsche_block = block(u, u,
                            WeakForm(bilinear=nitsche_bilinear, linear=(q, c) -> 0.0,
                                     symmetric=true, component_aware=true); on=outer_arc)

# ── Assemble and solve ──────────────────────────────────────────────────────

const problem = Problem((u,); blocks=(elasticity_block, nitsche_block),
                        loads=(body_load, inner_traction))
model = prepare(problem)
solution = solve!(model)
report = diagnostics(model, solution; exact)

# ── Output ─────────────────────────────────────────────────────────────────

out = joinpath(@__DIR__, "output", "fcm_annular_plate_2d")
write_vtk(out, solution, model;
          point_data=(uh=(u, c, x, xi) -> u(c, xi), exact=(u, c, x, xi) -> exact(x),
                      displacement_error=(u, c, x, xi) -> norm(u(c, xi) - exact(x))))

write_quadrature_vtm(out * "_quadrature", model)

print_run_report("FCM annular plate (Ruess 2013, §4.2)", report;
                 parameters=(:cells => V.levels[1].mesh.cells, :order => V.levels[1].order,
                             :r_inner => r_inner, :r_outer => r_outer,
                             :subcell_length_scale => annulus.subcell_length_scale,
                             :max_depth => annulus.max_depth, :geometry => "annulus = disk ∩ disk",
                             :E => E, :nu => 0.0, :nitsche_h => h_nitsche,
                             :nitsche_beta => nitsche_beta,
                             :outer_arc_segments => length(outer_arc.cells),
                             :inner_arc_segments => length(inner_arc.cells),
                             :sigma_rr_inner => sigma_rr_inner), output=out)
