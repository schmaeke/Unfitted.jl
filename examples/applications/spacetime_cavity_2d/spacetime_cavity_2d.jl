#=
A Cavity That Grows — Transient Thermo-Elasticity On A Moving Domain
===================================================================

A domain that changes shape over time, solved as **one static problem** on a
mesh that never moves: no time loop, no remeshing, no state transfer between
steps, no mesh motion.

The trick is to treat time as the last coordinate of the mesh. A plane problem
on a moving domain `Ω(t) ⊂ ℝ²` becomes a *fixed* problem on the space-time
cylinder

    Q = { (x₁, x₂, t) : (x₁, x₂) ∈ Ω(t), 0 < t < T } ⊂ ℝ³,

and a shape that evolves is then nothing but a static three-dimensional
geometry. That is exactly what this package's immersed machinery already
integrates, so the moving boundary costs one level set and no new code: here a
circular cavity of radius `r(t) = r₀ + ṙ t` grows inside a square plate, and Ω
is the plate minus the cavity,

    φ(x, t) = r(t) − ‖xₛ − c‖ ≤ 0,     xₛ = (x₁, x₂),

whose zero set is a truncated cone in (x₁, x₂, t) — one `leaf`, sampled by the
finite-cell moment fit like any other cut geometry. Because φ varies in `t` as
well as in `x`, its Lipschitz constant is the **space-time** one,
`|∇φ| = √(1 + ṙ²)`, not `1`; certifying it as `1` would let the classifier prove
cells empty that are not, silently deleting cut cells while every diagnostic
still reads healthy.

Why the cavity *grows*. With the trace at `t = 0` eliminated from trial and test
space, the space-time form `b(u,v) = ∫_Q (∂ₜu) v + κ ∇ₓu·∇ₓv dQ` obeys

    b(u, u) = ½‖u(T)‖²_Ω(T) + κ‖∇ₓu‖²_L²(Q) − ½∫₀ᵀ ∮_∂Ω(t) u² vₙ dσ dt,

where `vₙ` is the outward normal speed of the moving boundary. An *advancing*
material boundary makes the last term negative and costs the form its
coercivity; a *receding* one adds to it. A cavity that grows is a material
domain that shrinks, so this is the well-behaved direction — and it is also the
recognisable engineering problem: a corrosion pit or dissolution cavity opening
up inside a loaded, cooling plate.

Two fields share the one space-time mesh:

  * `θ` — temperature, genuinely coupled in time by `∂ₜθ − κ Δₓθ = f`. This is
    the field that earns the space-time mesh: stepping it through a *changing*
    domain would need the state projected onto a new space at every step, which
    is the cost this formulation removes outright.
  * `u` — displacement, one component per *spatial* axis and hence two on a
    three-dimensional mesh, in quasi-static equilibrium `∇ₓ·σ = 0` with a thermal
    eigenstrain, `σ = 2μ εₓ(u) + λ tr εₓ(u) I − β_T θ I`: a 2 × 2 stress on a
    3-axis mesh. The coupling runs one way, `θ → u`, so the block system is
    triangular and the temperature is unaffected by the stress it causes.

Neither field needs anything written down on the cavity wall, which is the quiet
advantage of an immersed formulation: a cut cell integrates only the material
side, so *not* writing a condition leaves the wall insulated (`∇ₓθ·n = 0`) and
traction free (`σ·n = 0`) — precisely the physics of an open cavity. The plate's
four edges are faces of the background box and therefore genuine material
boundary, which is where both fields take their data. Nothing is imposed at
`t = T`: the final time is an outflow face and must stay free.

**The manufactured solution.** With `s = ‖xₛ − c‖²/r(t)²` and `w(s) = s − s²/2`,

    θ(x, t) = e^{−t/τ} w(s),

and `f` is whatever `∂ₜθ − κ Δₓθ` makes it, written out below. Two properties
earn this particular field its place. `w′(1) = 0`, so `∇ₓθ` vanishes on `s = 1`
and the insulated condition holds **exactly on the moving wall** — the geometry
is verified, not merely integrated. And `θ` is rational in `t` through `r(t)`
and multiplied by an exponential, so it lies in no tensor-product polynomial
space at any order: the error measured below is a real discretisation error and
not an interpolation that happened to be exact.

**Order.** The order is `(2, 2, 3)` — deliberately odd in time. Measured
separately on a problem with identically zero spatial error, this formulation
gains an order only at odd degree in time: `p_t = 1, 2, 3, 4` converge at rates
`2.07`, `2.01`, `3.98`, `4.00`, so `p_t = 2` buys nothing over `p_t = 1` while
`p_t = 3` is very nearly free. Order 2 in space is close to a ceiling rather than
a free choice: the cut-cell moment fit is exact for the *tensor* moment basis, so
it solves for `∏(2p+1)` weights per cut cell — 125, 343, 729 in three dimensions
at `p = 2, 3, 4` — and its cost grows faster than the cube of that. Measured on a
`(8, 8, 8)` grid over this geometry, `prepare` takes 2.1 s at `p = 2`, 48 s at
`p = 3` and 1091 s at `p = 4`. The mesh size is almost free by comparison, and so
is `subcell_length_scale` (2.18 / 2.05 / 2.05 s across three halvings), because
the exact Lipschitz certificate decides most cells without subdividing at all.

**What the printed metric means.** `relative L2 error` is `θ` against the field
above, integrated over the whole space-time cylinder — one number covering every
instant at once. `fit failure count` is the geometry assertion and should be `0`:
it counts cut cells whose material part is empty, which happens when the cone
grazes a grid line, and is why the cavity centre is placed deliberately off every
grid line. It does *not* mean the moment fit struggled — `moment-fit residual
(max)` is the number that says that.

`θ` is the verified field here and `u` is the coupled response, reported through
the VTK bundle rather than against a closed form, because on this geometry there
is not one to compare against. The displacement that *would* have an exact form
is free thermal expansion, `u = α_T θ (xₛ − c)`, which makes `σ` vanish
identically; but that requires `α_T θ I` to be a compatible strain, hence `θ`
harmonic in space, and a harmonic field with zero normal derivative on a circle
is constant. So an exactly-verifiable `u` and a spatially varying `θ` cannot be
had in the same manufactured solution, and the elastic block's own verification
against the free-expansion field belongs in the test suite rather than here.

**Known limits, stated rather than discovered.** This is a single time slab with
a continuous-in-time Galerkin discretisation and α-FCM in place of a ghost
penalty; it has no inf-sup theory on cut space-time cells, and the coercivity
identity above fails for any advancing material boundary. Long horizons want the
time axis split into slabs coupled across `t = const` seams — the construction
`applications/interface_coupling_2d` already performs across a spatial seam. A
body that *appears* after `t = 0` is genuinely underdetermined and is not this
example's problem to solve. One limit is particular to `θ`'s nonzero Dirichlet
data: facet integration is grid-aligned and never trimmed by the level set, so
where the cone crosses the `t = 0` face the initial temperature is fitted over
the cavity's footprint as well as over the material part. That footprint is the
cone's base disc, `π r₀² ≈ 0.06` of the unit face, so the fitted initial
condition is a least-squares compromise over an area some six percent larger
than the physical one.

Unlike most of this example suite, the cost here is the cut-cell moment fit
rather than compilation, so this is one of the few cases that runs *faster* at
`-O2` than at `-O0`. `SC_CELLS` sets the cells per axis.

Reference for the finite-cell moment fit and the immersed machinery used here:
see `src/fcm.jl` and `src/implicit.jl`. The space-time treatment of moving
domains by an unfitted discretisation follows the idea of, among others,
C. Lehrenfeld and M. Olshanskii, *An Eulerian finite element method for PDEs in
time-dependent domains*, ESAIM Math. Model. Numer. Anal. **53** (2019) 585–614,
doi:10.1051/m2an/2018068 — which stabilises differently (discontinuous in time,
with a ghost penalty) and is the natural next step from this example.
=#

using Unfitted
using StaticArrays
using Tensors

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

# ── Geometry, material, and the manufactured data ───────────────────────────

const L = 1.0                                   # plate edge length
const T_end = 1.0                               # time horizon
const r₀, ṙ = 0.1373, 0.22                      # cavity radius r(t) = r₀ + ṙ t
const centre = SVector(0.4967, 0.5013)          # off every grid line, on purpose
const κ, τ = 0.2, 0.35                          # diffusivity, decay time
const E, ν, α_T = 1.0, 0.3, 1.0e-2              # Young, Poisson, thermal expansion

const μ = E / (2 * (1 + ν))                     # shear modulus
const λ = E * ν / ((1 + ν) * (1 - 2ν))          # Lamé λ (plane strain)
const β_T = 2 * (μ + λ) * α_T                   # thermal stress coefficient

radius(t) = r₀ + ṙ * t
amplitude(t) = exp(-t / τ)

# Offset from the cavity axis and its squared length. The geometry is spelled out
# exactly once, here; keeping the squared form is what leaves the manufactured
# field smooth — no square root, hence no kink on the axis.
offset(x) = SVector(x[1] - centre[1], x[2] - centre[2])
ρ²(x) = sum(abs2, offset(x))
scaled(x) = ρ²(x) / radius(x[3])^2          # `s`, so `s = 1` is the moving wall

# Ω = { φ ≤ 0 } is the plate *minus* the cavity. One smooth leaf; the constant is
# the space-time one, |∇φ| = √(1 + ṙ²), and it is exact.
cavity = leaf(x -> radius(x[3]) - sqrt(ρ²(x)); lipschitz=sqrt(1 + ṙ^2))

# `w′(1) = 0` is what makes the insulated wall hold exactly on the moving
# surface, and it is the only property of `w` this example needs.
profile(s) = s - s^2 / 2

# The temperature and its source, from `∂ₜθ = A(−w/τ − 2ṙ s w′/r)` and
# `Δₓθ = 4A(1 − 2s)/r²` — both elementary, hence written out rather than
# differentiated numerically.
temperature(x) = amplitude(x[3]) * profile(scaled(x))
function heat_source(x)
    s, r, A = scaled(x), radius(x[3]), amplitude(x[3])
    return A * (-profile(s) / τ - 2ṙ * s * (1 - s) / r) - 4κ * A * (1 - 2s) / r^2
end

# ── Weak forms ──────────────────────────────────────────────────────────────
#
# Time is the last mesh axis, so a space-time gradient carries ∇ₓ in its first two
# slots and ∂ₜ in its last: `g[end]` is the time derivative, `g[spatial]` the
# spatial gradient — two entries long rather than three with the time slot zeroed,
# because its *length* is what says "spatial". Assembly zero-extends a short
# gradient coefficient into the axes the field carries no component for, where the
# flux genuinely is zero; the `TestChannels` docstring states that convention.

const spatial = SVector(1, 2)

# ∫_Q (∂ₜθ) v + κ ∇ₓθ·∇ₓv dQ. The time derivative lands on the test *value*
# channel and the spatial gradient on the test *gradient* channel, which is the
# whole of what makes this a space-time operator. It is not symmetric, and
# `symmetric = false` is the default precisely so an undeclared form keeps both
# triangles: declaring symmetry here would silently discard the ∂ₜ coupling.
heat_bilinear(q, trial) = TestChannels(trial.gradient[end], κ * trial.gradient[spatial])

heat = WeakForm(bilinear=heat_bilinear)

# The 4th-order isotropic plane-strain tensor ℂ[i,j,k,l] = λ δ_ij δ_kl
# + μ (δ_ik δ_jl + δ_il δ_jk) over the two *spatial* axes — the third mesh axis is
# time and carries no stress — so that σ = ℂ ⊡ εₓ is 2μ εₓ + λ tr(εₓ) I with the
# Lamé pair fixed above; I₂ is the identity the thermal stress −β_T θ I is written
# against. Both are `const`, as are the fields `θ` and `u` below, because the
# callbacks capture them and run once per test component per trial dof per
# quadrature point, where a non-`const` global is read as `Any` and allocates.
isotropic(i, j, k, l) = λ * (i == j) * (k == l) + μ * ((i == k) * (j == l) + (i == l) * (j == k))

const ℂ = SymmetricTensor{4,2,Float64}(isotropic)
const I₂ = one(SymmetricTensor{2,2,Float64})

# ∫_Q σ(u) : εₓ(v) dQ with σ = ℂ ⊡ εₓ(u), an ordinary constitutive evaluation, so
# swapping ℂ is the only edit another law needs. The three-argument `TestChannels`
# takes the whole stress and picks row `r` itself, since assembly contracts against
# the gradient of the scalar test function of component `r` and
# `σ : ∇v = Σⱼ σ[r, j] ∂ⱼN`. The *two*-argument `symmetric_gradient(trial, u)` is
# the one to call: `u` has one component per spatial axis while the mesh has a time
# axis too, so the strain is 2 × 2 on a 3-axis mesh — the one-argument form, sized
# by the mesh, adds precisely what its docstring describes.
elasticity_bilinear(q, trial, r) = TestChannels(0.0, ℂ ⊡ symmetric_gradient(trial, u), r)

elasticity = WeakForm(bilinear=elasticity_bilinear, component_aware=true)

# −∫_Q β_T θ tr εₓ(v) dQ, written as row `r` of the thermal stress −β_T θ I₂ so
# that it reaches the displacement rows through the same contraction as the
# elastic block. The eigenstrain is a stress contribution and not a load: it sits
# inside the one stress σ = 2μ εₓ(u) + λ tr εₓ(u) I − β_T θ I, and a load
# integrand would need θ *known*, which it is not — θ is an unknown of this same
# system, which is why the term is an off-diagonal matrix block. Its trial field
# is `θ`, whose only component is `1`, while the test runs over both displacement
# components, so no component-matching guard is wanted.
eigenstrain_bilinear(q, trial, r) = TestChannels(0.0, -β_T * trial.value * I₂, r)

eigenstrain = WeakForm(bilinear=eigenstrain_bilinear, component_aware=true)

# ── One solve ───────────────────────────────────────────────────────────────

const cells_per_axis = env("SC_CELLS", 6)
const order = (2, 2, 3)

domain = physical_domain(cavity; alpha=1.0e-6, subcell_length_scale=L / (4 * cells_per_axis))
V = space(box((0.0, 0.0, 0.0), (L, L, T_end)); cells=ntuple(_ -> cells_per_axis, 3), order=order,
          physical=domain)
const θ = field(:θ, V)
const u = field(:u, V; components=2)

# Each field's data is one selector. `θ` carries the manufactured field on every
# face but the last-time one — the four plate edges and the initial state at
# `t = 0` take the *same* datum, so they are one condition — while `u` is clamped
# on the plate edges alone and, being quasi-static, wants no condition in time at
# all: its only deficiency is the ordinary spatial rigid-body one, which the
# clamped edges already remove. `t = T` is left free in both because the final
# time is an outflow face, and constraining it would turn a transient into a
# two-point boundary-value problem in time. The exclusion is *closed*, which is
# what makes "free at `t = T`" a statement about the face and not about its rim:
# the ring where the plate edges meet `t = T` lies on a kept face and stays
# constrained — rightly, since a plate edge holds its datum at every instant
# including the last.
bcs = [dirichlet(temperature; on=boundary(:all; except=(axis=3, side=:upper)), field=θ),
       dirichlet(0.0; on=boundary(:all; except=(axis=3,)), field=u)]

# `block` carries the bilinear side only, so the manufactured source is attached
# as its own load. The displacement rows take no load: everything that drives
# them arrives through the `(u, θ)` block.
problem = Problem((θ, u);
                  blocks=(block(θ, θ, heat), block(u, u, elasticity), block(u, θ, eigenstrain)),
                  loads=(source_load(θ; source=heat_source),), dirichlet=bcs)

model = prepare(problem)
solution = solve!(model)
report = merge(diagnostics(model, solution), (; l2_error=l2_error(solution, model, θ, temperature)))

# ── Output ──────────────────────────────────────────────────────────────────
#
# The bundle is an ordinary three-dimensional unstructured grid in (x₁, x₂, t),
# so a ParaView Slice at constant `t` recovers the plane solution at that instant
# and sweeping the slice animates the transient — from a single solve. The
# `level_set` array the writer attaches automatically carries φ, whose zero
# isosurface is the cavity's space-time cone.

out = joinpath(@__DIR__, "output", "spacetime_cavity_2d")
write_vtk(out, solution, model)
write_quadrature_vtm(out * "_quadrature", model)

print_run_report("Growing cavity in a plate — space-time thermo-elasticity", report;
                 parameters=(:cells => V.levels[1].mesh.cells, :order => nominal_order(V.levels[1]),
                             :cavity_radius => (r₀, radius(T_end)), :growth_rate => ṙ,
                             :lipschitz => sqrt(1 + ṙ^2), :diffusivity => κ, :E => E, :nu => ν,
                             :thermal_expansion => α_T,
                             :geometry => "Ω = plate ∖ { ‖xₛ − c‖ ≤ r(t) }, one cone leaf",
                             :alpha => domain.alpha,
                             :subcell_length_scale => domain.subcell_length_scale), output=out)
