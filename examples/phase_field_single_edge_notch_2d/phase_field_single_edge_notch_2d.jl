#=
Phase-field single-edge-notched tension test from Section 5.1 of
C. Miehe, M. Hofacker, F. Welschinger,
"A phase field model for rate-independent crack propagation: Robust
algorithmic implementation based on operator splits",
Comput. Methods Appl. Mech. Engrg. 199 (45-48) (2010) 2765-2778
(doi:10.1016/j.cma.2010.04.011).

The specimen is Ω = (0, 1)² with initial notch
Γ₀ = {(x, 0.5) : 0 ≤ x ≤ 0.5}. The model is

    ψ(ε, d) = ((1 - d)² + k) ψ₀⁺(ε) + ψ₀⁻(ε),
    H(x, t) = maxₛ≤ₜ ψ₀⁺(ε(x, s)),
    (Gᶜ/l + 2H + η/τ)d - Gᶜl Δd = 2H + (η/τ)dₙ,
    div σ(u, d) = 0.

Refinement strategy: a **monotonically growing strip overlay** along the
crack-path band y ∈ 0.5 ± 4ℓ. Both the displacement field `u` and the damage
field `d` live on the coupled `base + overlay` space. A `LevelMask` activates
only the strip cells we currently need; the mask starts as a bootstrap halo
around the notch and grows whenever the damage indicator `d ≥ mask_threshold`
fires at a cell centre, dilated by `mask_dilation` cells to give the crack
room to propagate before the next update.

When the mask grows, state transfer follows established FCM-remeshing
practice (Sartorti & Düster, Comput Mech 77 (2026), doi:10.1007/s00466-024-02486-0):

- `u` and `d` carry over by **raw-key rewiring**. Because the new active basis
  is a strict superset of the old one (cells only activate, never deactivate),
  copying old coefficients to their matching dofs in the new layout and
  zero-initialising the genuinely new ones reconstructs the partial-field
  representation `u_total = u_1 + u_2 + ...` of Sartorti & Düster *exactly* —
  no L2 projection, no inverse mapping.
- Per-quadrature-point history `H` transfers by **RBF + P0**: an inverse
  multiquadric basis with a constant polynomial extension fits old q-point H
  values, evaluated at every new q-point. Recommended in Sartorti & Düster
  (Sec. 4) as the most robust point-based transfer in their hyperelastic FCM
  remeshing study. Max-clamped against `initial_history(x)` and the current
  `positive_energy(strain)` to keep H monotone.
- An **equilibrium step** (one `solve_phase` + one `solve_displacement` at the
  current applied displacement, no advance) absorbs the small residual the
  RBF transfer introduces — also standard remeshing practice (Sartorti &
  Düster, Sec. 3.3.1).

Example:
    SHP_PHASE_FINAL_DISPLACEMENT=0.0050 \
    julia --project=. examples/phase_field_single_edge_notch_2d.jl
=#

using Unfitted
using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "..", "reporting.jl"))

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
function env_bool(name, default)
    lowercase(get(ENV, name, default ? "true" : "false")) in ("1", "true", "yes")
end

const cells_per_axis = env_int("SHP_PHASE_CELLS", 12)
const order = env_int("SHP_PHASE_ORDER", 3)
const target_displacement = env_float("SHP_PHASE_FINAL_DISPLACEMENT", 7.5e-3)
const write_output = env_bool("SHP_PHASE_WRITE_OUTPUT", true)
const output_root = get(ENV, "SHP_PHASE_OUTPUT_ROOT",
                        joinpath(@__DIR__, "output", "phase_field_single_edge_notch_2d"))

# Material (Miehe et al. 2010, Table 1) and viscous regularization.
const lambda = 121.15
const mu = 80.77
const Gc = 2.7e-3
const ell = 0.0150
const eta_viscosity = 1.0e-6
const residual_stiffness = 1.0e-6
const time_step = 1.0

# Geometry: specimen, notch, and the static band overlay covering the crack path.
const omega = box((0.0, 0.0), (1.0, 1.0))
const notch_tip = SVector(0.5, 0.5)
const notch_left = SVector(0.0, 0.5)
const strip_half_height = 4ell
const strip_cells_x = 100
const strip_cells_y = 16
const strip_order = 2
const strip_domain = box((0.0, 0.5 - strip_half_height), (1.0, 0.5 + strip_half_height))
const strip_mesh_template = mesh(strip_domain; cells=(strip_cells_x, strip_cells_y))
const notch_history_factor = 100.0
const notch_history_width_factor = 0.25

# Growing-mask knobs.
# `mask_threshold`: damage value at which a strip cell is permanently triggered.
# Set well above the residual base-overlay smear (~0.045) so background damage
# never grows the mask, only the actual crack tip does.
# `mask_dilation`: Chebyshev halo around the triggered set so each accepted load
# step has free cells ahead of the tip to propagate into before the next update.
# `bootstrap_halo`: geometric radius around the notch segment that seeds the
# initial triggered set.
const mask_threshold = 0.5
const mask_dilation = 2
const bootstrap_halo = 2.0 * ell

# RBF + P0 history transfer follows Sartorti & Düster 2024, Sec. 3.3.1 — see
# the library's `RBFP0` for details. Default neighbour count = 10.

# Adaptive load stepping and Newton tolerances.
const max_newton_iterations = 24
const newton_rtol = 1.0e-8
const newton_atol = 1.0e-10
const initial_displacement_step = 1.0e-5
const min_displacement_step = 1.0e-7
const max_displacement_step = 5.0e-4
const max_phase_increment = 0.03
const easy_phase_increment = 0.005
const max_force_drop = 0.10
const step_growth_factor = 1.25
const step_shrink_factor = 0.5
const max_accepted_steps = 1000

# VTK export cadence (load-delta-driven).
const export_force_delta = 0.02

# ── Material, energy split, and constitutive law ─────────────────────────────
#
# Implements the tension/compression split of Miehe et al. 2010 (Sec. 3) for
# the strain-energy density: only the *tensile* part of the elastic energy
# drives damage, so a crack opens under tension but is allowed to close
# without further degradation under compression.
#
# Voigt notation throughout the mechanics: strain is stored as
#
#     ε = (εxx, εyy, γxy)ᵀ,
#
# with the engineering shear convention `γxy = 2 εxy`. The split returns
# per-component arrays in the *tensor* convention (third component
# `εxy = γxy/2`); the inner products in `positive_energy` and `split_stress`
# pick up the corresponding factor of 2 so the energy is consistent.
#
# Splitting strategy:
#
#   1. Spectral decomposition of the 2D strain into principal strains
#      `e₁ ≥ e₂` and the projection coefficients onto the principal frame.
#   2. ε⁺ = ⟨e₁⟩₊ P₁ + ⟨e₂⟩₊ P₂,   ε⁻ = ⟨e₁⟩₋ P₁ + ⟨e₂⟩₋ P₂,
#      with `⟨·⟩₊ = max(·, 0)` and `⟨·⟩₋ = min(·, 0)`.
#   3. ψ⁺(ε) = ½ λ ⟨tr ε⟩₊² + μ (ε⁺·ε⁺),     ψ⁻(ε) = ½ λ ⟨tr ε⟩₋² + μ (ε⁻·ε⁻),
#      σ⁺(ε) = λ ⟨tr ε⟩₊ I + 2μ ε⁺,         σ⁻(ε) = λ ⟨tr ε⟩₋ I + 2μ ε⁻.
#   4. Damage `d ∈ [0, 1]` degrades the tensile branch only:
#
#          σ(ε, d) = ((1 − d)² + k) σ⁺(ε) + σ⁻(ε),
#
#      where `k = residual_stiffness` is the small ersatz stiffness that
#      keeps the system non-singular at fully damaged points.

"""
    Material(lambda, mu, residual_stiffness)

Linear-elastic material parameters in the Miehe phase-field model.
`lambda` and `mu` are the Lamé constants; `residual_stiffness` is the
floor on the degradation function `g(d) = (1 − d)² + k` that keeps the
mechanical system non-singular at fully damaged points (`d = 1`).
"""
struct Material
    lambda::Float64
    mu::Float64
    residual_stiffness::Float64
end

# Voigt-strain (or Voigt-stress) static vector and stress-tangent static
# matrix. Aliased so the per-quadrature-point cache stays type-stable.
const StrainStress = SVector{3,Float64}
const StressTangent = SMatrix{3,3,Float64,9}

# History and per-step material state are stored as plain vectors indexed by
# the stable quadrature-point id `q.point` (see
# `foreach_quadrature_point` / `nquadpoints` in the package). Mechanics and
# phase models share the same space `V`, so their `q.point` indexing is
# identical and the same vectors work for both.

# Distance from a physical point to the closest point on the initial notch
# segment Γ₀ = {(x, 0.5) : 0 ≤ x ≤ 0.5}. Used by `initial_history` (seed the
# damage bump at the notch tip) and by `bootstrap_triggered_mask` (seed the
# initial active strip cells).
function distance_to_notch(x)
    projection = clamp((x[1] - notch_left[1]) / (notch_tip[1] - notch_left[1]), 0.0, 1.0)
    closest = notch_left + projection * (notch_tip - notch_left)
    return norm(SVector(x[1], x[2]) - closest)
end

# Smooth exponential decay around the notch segment, used only for the
# VTK `notch` channel so a viewer can visualise the crack-path band.
reference_notch_profile(x) = exp(-distance_to_notch(x) / ell)

# Initial history-field seed: a Gaussian bump in `ψ⁺` centred on the notch.
# Pre-loading `H` above the regularisation floor `Gc / ℓ` along the notch
# means the first phase solve already sees a damaged region there, so the
# bootstrap solve produces the diffuse-notch field without needing a
# resolved geometric crack. Restricted to `x ≤ notch_tip + 2ℓ` to avoid
# polluting the far field with seed energy.
function initial_history(x)
    x[1] <= notch_tip[1] + 2ell || return 0.0
    width = notch_history_width_factor * ell
    return notch_history_factor * Gc / ell * exp(-(distance_to_notch(x) / width)^2)
end

# 2D spectral decomposition of a Voigt strain `ε = (εxx, εyy, γxy)`. The
# principal strains are
#
#     e₁ = mean + r,   e₂ = mean − r,
#     mean = (εxx + εyy) / 2,   diff = (εxx − εyy) / 2,
#     εxy  = γxy / 2,           r = √(diff² + εxy²),
#
# and the projection coefficients onto the principal-direction outer
# products `P_α = p_α p_αᵀ` come from
#
#     p₁₁ = ½ + diff / (2r),   p₂₂ = ½ − diff / (2r),   p₁₂ = εxy / (2r).
#
# Returning `ε⁺` and `ε⁻` with components in *tensor* shear convention
# (third component is `εxy`, not `γxy`) — `positive_energy` and
# `split_stress` apply the right factor of 2 to compensate.
#
# The branch `r ≤ 1e-14` handles the spherical case where the principal
# frame is undefined: both principal strains equal the mean and the
# off-diagonal vanishes.
function positive_negative_strain(strain)
    exx, eyy, gamma = strain
    exy = 0.5gamma
    mean = 0.5 * (exx + eyy)
    diff = 0.5 * (exx - eyy)
    radius = hypot(diff, exy)

    if radius <= 1.0e-14
        positive = max(mean, 0.0)
        negative = min(mean, 0.0)
        return SVector(positive, positive, 0.0), SVector(negative, negative, 0.0)
    end

    e1 = mean + radius
    e2 = mean - radius
    p11 = 0.5 + diff / (2radius)
    p22 = 0.5 - diff / (2radius)
    p12 = exy / (2radius)

    ep1 = max(e1, 0.0)
    ep2 = max(e2, 0.0)
    em1 = min(e1, 0.0)
    em2 = min(e2, 0.0)
    positive = SVector(ep1 * p11 + ep2 * p22, ep1 * p22 + ep2 * p11, ep1 * p12 - ep2 * p12)
    negative = SVector(em1 * p11 + em2 * p22, em1 * p22 + em2 * p11, em1 * p12 - em2 * p12)
    return positive, negative
end

# Tensile elastic-energy density
#
#     ψ⁺(ε) = ½ λ ⟨tr ε⟩₊² + μ (ε⁺ · ε⁺).
#
# The `2 eps_plus[3]^2` is the Voigt-product factor: with `eps_plus[3]`
# carrying the tensor shear `εxy⁺`, the inner product expands to
# `εxx² + εyy² + 2 εxy²` (the off-diagonal contribution is counted twice
# because the strain tensor is symmetric).
#
# `ψ⁺` is the only energy density driving the history field `H` and
# therefore the only one driving damage growth in the phase equation.
function positive_energy(strain, material::Material)
    eps_plus, _ = positive_negative_strain(strain)
    trp = max(strain[1] + strain[2], 0.0)
    return 0.5 * material.lambda * trp^2 +
           material.mu * (eps_plus[1]^2 + eps_plus[2]^2 + 2eps_plus[3]^2)
end

# Cauchy stress under the Miehe tension/compression split:
#
#     σ(ε, d) = g(d) σ⁺(ε) + σ⁻(ε),
#     σ⁺ = λ ⟨tr ε⟩₊ I + 2μ ε⁺,    σ⁻ = λ ⟨tr ε⟩₋ I + 2μ ε⁻,
#     g(d) = (1 − d)² + k.
#
# `phase` is clamped to `[0, 1]` so an over-damaged Newton iterate
# cannot blow up the degradation factor. The third Voigt component of
# `positive` / `negative` is the *engineering* shear `γxy = 2 εxy` —
# the spectral decomposition's `2μ εxy⁺` and `2μ εxy⁻` already include
# the factor of 2 needed to convert the tensor `εxy` it stores back to
# the engineering convention the assembly path expects.
function split_stress(strain, phase, material::Material)
    eps_plus, eps_minus = positive_negative_strain(strain)
    g = (1.0 - clamp(phase, 0.0, 1.0))^2 + material.residual_stiffness
    trp = max(strain[1] + strain[2], 0.0)
    trm = min(strain[1] + strain[2], 0.0)

    positive = SVector(material.lambda * trp + 2material.mu * eps_plus[1],
                       material.lambda * trp + 2material.mu * eps_plus[2],
                       2material.mu * eps_plus[3])
    negative = SVector(material.lambda * trm + 2material.mu * eps_minus[1],
                       material.lambda * trm + 2material.mu * eps_minus[2],
                       2material.mu * eps_minus[3])
    return g * positive + negative
end

# Numerical stress-tangent `∂σ/∂ε`. Central differences in each Voigt
# direction. We pay the 6 extra stress evaluations per point and avoid
# the analytic tangent because the spectral split's projection
# coefficients have a `1 / r` singularity when the principal frame
# becomes degenerate (`r → 0`); the finite-difference path is finite-
# valued through that region thanks to the `r ≤ 1e-14` branch in
# `positive_negative_strain`. Step size scales with `‖ε‖` to keep
# the relative perturbation roughly constant.
function stress_tangent(strain, phase, material::Material)
    h = 1.0e-7 * max(1.0, norm(strain))
    columns = ntuple(3) do j
        direction = SVector(ntuple(i -> i == j ? h : 0.0, 3))
        (split_stress(strain + direction, phase, material) -
         split_stress(strain - direction, phase, material)) / (2h)
    end
    return SMatrix{3,3}(hcat(columns...))
end

# Strain at an arbitrary physical point from a vector displacement solution.
function displacement_strain(solution, model, u, x)
    g1 = gradient(solution, model, u, x, 1)
    g2 = gradient(solution, model, u, x, 2)
    return SVector(g1[1], g2[2], g1[2] + g2[1])
end

# Strain at a quadrature point from the current iterate exposed as `q.state`.
function strain_state(state)
    SVector(gradient(state, :u, 1)[1], gradient(state, :u, 2)[2],
            gradient(state, :u, 1)[2] + gradient(state, :u, 2)[1])
end

# Damage values frozen at every quadrature point of `model` (uses `q.point`).
function frozen_damage(model, damage_solution)
    d = zeros(nquadpoints(model))
    foreach_quadrature_point(model; state=damage_solution) do q
        d[q.point] = clamp(value(q.state, :d), 0.0, 1.0)
    end
    return d
end

seed_history(model) = QuadField{Float64}(model; init=q -> initial_history(q.x))

function update_history!(history, displacement, material)
    foreach_quadrature_point(displacement.model; state=displacement.solution) do q
        history[q.point] = max(history[q.point], positive_energy(strain_state(q.state), material),
                               initial_history(q.x))
    end
    return history
end

function phase_increment(previous, current, phase_state)
    old = frozen_damage(phase_state.model, previous)
    increment = 0.0
    foreach_quadrature_point(phase_state.model; state=current) do q
        increment = max(increment, abs(clamp(value(q.state, :d), 0.0, 1.0) - old[q.point]))
    end
    return increment
end

# ── Space and state builders ──────────────────────────────────────────────────
#
# `build_space(strip_mask)` composes the model's superposition space:
# the `(cells_per_axis × cells_per_axis)` base level at total-degree `p`
# plus the strip overlay carrying the current activation mask. Both `u`
# and `d` live on this same space, so a single rebuild refreshes the
# basis for both fields. Total-degree mode is chosen so the overlay
# does not balloon the dof count at high `p`.
function build_space(strip_mask)
    V = space(omega; cells=(cells_per_axis, cells_per_axis), order=order, mode=:total_degree)
    return overlay(V, strip_domain; cells=(strip_cells_x, strip_cells_y), order=strip_order,
                   active=strip_mask)
end

# ── Growing strip mask ───────────────────────────────────────────────────────
#
# Three pieces:
#
#   * `dilate_mask(mask, n)` — Chebyshev-distance dilation by `n` cells.
#     Each iteration ORs in neighbouring `true` cells across all
#     8-connected (in 2D) directions. Used to add a halo around the
#     triggered set so the next load step has free cells ahead of the
#     crack tip to propagate into.
#   * `bootstrap_triggered_mask()` — initial triggered set: every strip
#     cell whose centre falls inside `bootstrap_halo` of the notch.
#   * `update_triggered_mask(previous, …)` — read the current damage
#     field at every strip cell centre; cells whose damage exceeds
#     `mask_threshold` join the triggered set. The set is monotone (a
#     cell, once triggered, never untriggers) so the basis only grows.

function dilate_mask(mask::BitArray{D}, n::Integer) where {D}
    n <= 0 && return copy(mask)
    out = copy(mask)
    for _ in 1:n
        next = copy(out)
        for ci in CartesianIndices(out)
            out[ci] && continue
            for offset in CartesianIndices(ntuple(_ -> -1:1, D))
                nbr = CartesianIndex(ci.I .+ offset.I)
                if checkbounds(Bool, out, nbr) && out[nbr]
                    next[ci] = true
                    break
                end
            end
        end
        out = next
    end
    return out
end

function bootstrap_triggered_mask()
    mask = falses(strip_mesh_template.cells)
    for ci in cell_indices(strip_mesh_template)
        c = center(cell_box(strip_mesh_template, ci))
        distance_to_notch(c) <= bootstrap_halo && (mask[ci] = true)
    end
    return mask
end

function update_triggered_mask(previous, damage_solution, phase_state)
    new_triggered = copy(previous)
    for ci in cell_indices(strip_mesh_template)
        new_triggered[ci] && continue
        c = center(cell_box(strip_mesh_template, ci))
        d = clamp(value(damage_solution, phase_state.model, phase_state.damage, c), 0.0, 1.0)
        d >= mask_threshold && (new_triggered[ci] = true)
    end
    return new_triggered
end

function build_phase_state(V)
    damage = field(:d, V)
    model = prepare(Problem((damage,)))
    mass_matrix = assemble_matrix(model, mass_block(damage))
    mass_factor = factorize(mass_matrix)
    coefficients = zeros(active_unknowns(model))
    return (; model, damage, coefficients, mass_matrix, mass_factor)
end

function build_displacement_model(V, applied_displacement)
    u = field(:u, V; components=2)
    # Symmetric SENT setup: u_2 prescribed on the top and bottom edges, with a
    # codim-D pin at the bottom-left corner removing the u_1 rigid-body mode.
    # Allowing u_1 to relax on both edges keeps the elastic field symmetric
    # about y = 0.5, so the phase-field crack stays exactly on the midline.
    problem = Problem((u,);
                      dirichlet=[dirichlet(0.0; on=boundary(axis=2, side=:lower), field=u,
                                           component=2),
                                 dirichlet(applied_displacement; on=boundary(axis=2, side=:upper),
                                           field=u, component=2),
                                 dirichlet(0.0;
                                           on=boundary((axis=1, side=:lower),
                                                       (axis=2, side=:lower)), field=u,
                                           component=1)], symmetric=false)
    return (; model=prepare(problem), u)
end

# Phase-field weak form (Miehe et al. 2010 with viscous regularisation):
#
#     a(d, v) = ∫_Ω (Gc/ℓ + 2 H + η/τ) d·v dx  +  ∫_Ω Gc ℓ ∇v·∇d dx,
#     ℓ(v)   = ∫_Ω (2 H + (η/τ) dₙ)·v dx,
#
# where `H = max_{s ≤ t} ψ⁺(ε(x, s))` is the history field enforcing
# damage irreversibility, `η` is the viscosity, `τ = time_step`, and
# `dₙ` is the damage from the previous load step. The `Gc ℓ ∇v·∇d` term
# is the standard Γ-convergence regularisation that sets the diffuse
# crack bandwidth; the `(Gc/ℓ + …) d·v` term provides the local
# restoring force.
#
# The form is built per-call because `history` and `previous_damage`
# change between load steps. They are read out of the per-q-point
# vectors via `q.point`, the stable global quadrature-point index.
function phase_form(history, previous_damage)
    return WeakForm(bilinear=(q, trial) -> TestChannels((Gc / ell +
                                                         2history[q.point] +
                                                         eta_viscosity / time_step) * trial.value,
                                                        Gc * ell * trial.gradient),
                    linear=q -> 2history[q.point] +
                                eta_viscosity / time_step * previous_damage[q.point],
                    symmetric=true)
end

# Bootstrap phase form: same structure as `phase_form` but `H` is read
# directly from `initial_history(q.x)` instead of from a per-q-point
# vector, and there is no viscous regularisation term. Used once at
# `initial_state` to produce the diffuse-notch damage field from which
# the first proper `solve_phase` proceeds.
function initial_phase_form()
    return WeakForm(bilinear=(q, trial) -> begin
                        H = initial_history(q.x)
                        TestChannels((Gc / ell + 2H) * trial.value, Gc * ell * trial.gradient)
                    end, linear=q -> 2initial_history(q.x), symmetric=true)
end

# Solve the regularised phase-field problem at fixed history. Symmetric
# positive-definite, so a direct solve is enough. The matrix and rhs are
# rebuilt every call because `history` and `previous_damage` change
# between load steps; we keep the mass matrix factor on `phase_state`
# for the L²-projection of `H` onto the damage space (used by
# `history_solution`), not for this solve.
function solve_phase(phase_state, history)
    previous = solution(phase_state.model, phase_state.coefficients; method=:previous_phase)
    previous_damage = frozen_damage(phase_state.model, previous)
    form = phase_form(history, previous_damage)
    matrix = assemble_matrix(phase_state.model, block(phase_state.damage, phase_state.damage, form);
                             threaded=false)
    rhs = assemble_vector(phase_state.model, loadform(phase_state.damage, form); threaded=false)
    coefficients = matrix \ rhs
    residual = norm(matrix * coefficients - rhs)
    return coefficients, solution(phase_state.model, coefficients; method=:phase_direct, residual)
end

# ── History transfer on mask growth ──────────────────────────────────────────
#
# The library provides two transfer primitives this example combines:
#
#   * `transfer!(sol, old_model, new_model; backend=Rewire())` — exact
#     raw-key rewiring of `u` and `d`. Because the new active basis is a
#     strict superset of the old one (cells only activate, never
#     deactivate), copying old active coefficients to their matching
#     dofs in the new layout and zero-initialising the genuinely new
#     ones reproduces the source field pointwise. No L² projection, no
#     inverse mapping, no smoothing.
#   * `transfer(qfield, old_model, new_model, RBFP0())` — RBF + P0
#     transfer of the per-quadrature-point history field `H` (Sartorti
#     & Düster 2024 Sec. 3.3.1). The new q-point cloud does not have
#     any dof structure, so a point-based interpolant is the right
#     tool.
#
# Wrap the RBF transfer with the phase-field-specific max-floor so `H`
# stays monotone:
#
#     H_new(q) = max( RBF(H_old)(q),  initial_history(q.x),
#                     ψ⁺(ε(q), material) ).
#
# Monotonicity matters: damage is irreversible, so the history must
# never decrease across a transfer. The `ψ⁺(ε)` floor catches points
# where the current strain alone justifies more damage than the
# interpolated history.
function rbf_transfer_history(old_state, new_phase_state, new_displacement, material)
    transferred = transfer(old_state.history, old_state.phase_state.model, new_phase_state.model,
                           RBFP0())
    foreach_quadrature_point(new_phase_state.model) do q
        strain = displacement_strain(new_displacement.solution, new_displacement.model,
                                     new_displacement.u, q.x)
        transferred[q.point] = max(transferred[q.point], initial_history(q.x),
                                   positive_energy(strain, material), 0.0)
    end
    return transferred
end

# ── Mechanics: stress/tangent cache, Voigt helpers, Newton solve ─────────────
#
# The mechanical equilibrium `div σ(u, d) = 0` is solved by Newton's
# method at every load step. Each iteration:
#
#   1. wraps the current coefficient guess as a `solution(model, …)`
#      so the assembly callbacks can read the current strain from
#      `q.state` via `strain_state(q.state)`;
#   2. caches the Cauchy stress and the stress-tangent at every
#      quadrature point of the current iterate
#      (`build_mechanics_cache`) — both depend on the current strain
#      and the *frozen* damage from the most recent phase solve, so
#      they cannot be reused across iterations;
#   3. assembles the residual `r = ∫_Ω σ : ∇v dx` as a load-form
#      contribution — the cached stress is a constant pointwise field,
#      so the linear callback returns it directly via
#      `stress_test_channels`;
#   4. assembles the tangent `K_T = ∫_Ω ∇v : C : ∇u dx` as a block,
#      where `C = ∂σ/∂ε` is the per-q-point tangent from the cache;
#   5. solves `K_T Δu = -r`, updates coefficients, checks the
#      residual and step-norm against the rtol/atol pair.
#
# The form is `component_aware = true` and `symmetric = false`: the
# Miehe energy split breaks the major symmetry of the tangent in
# general, so we cannot rely on the symmetric-assembly fast path.

# Precompute the Cauchy stress and the stress-tangent at every
# quadrature point of `model` from the current displacement iterate and
# the (frozen) phase field. Stored as `Vector{SVector{3}}` and
# `Vector{SMatrix{3,3}}` indexed by `q.point` so the assembly loop
# below reads them in constant time per quadrature point.
function build_mechanics_cache(model, iterate, phase_qp, material)
    n = nquadpoints(model)
    stresses = Vector{StrainStress}(undef, n)
    tangents = Vector{StressTangent}(undef, n)
    foreach_quadrature_point(model; state=iterate) do q
        strain = strain_state(q.state)
        phase = phase_qp[q.point]
        stresses[q.point] = split_stress(strain, phase, material)
        tangents[q.point] = stress_tangent(strain, phase, material)
    end
    return (; stresses, tangents)
end

# Voigt strain produced by the `component`-th displacement direction's basis gradient.
function trial_strain(grad, component)
    component == 1 ? SVector(grad[1], 0.0, grad[2]) : SVector(0.0, grad[2], grad[1])
end

# Test channel pairing a Voigt stress with the `component`-th test direction.
function stress_test_channels(stress, component)
    component == 1 ? TestChannels(0.0, SVector(stress[1], stress[3])) :
    TestChannels(0.0, SVector(stress[3], stress[2]))
end

function solve_displacement(V, applied_displacement, previous_coefficients, damage_solution,
                            phase_state, material)
    mech = build_displacement_model(V, applied_displacement)
    nactive = active_unknowns(mech.model)
    coefficients = length(previous_coefficients) == nactive ? copy(previous_coefficients) :
                   zeros(nactive)
    phase_qp = frozen_damage(phase_state.model, damage_solution)
    residual_norm = Inf
    iterations = 0
    converged = false

    for iteration in 1:max_newton_iterations
        iterations = iteration
        iterate = solution(mech.model, coefficients; method=:newton_state)
        cache = build_mechanics_cache(mech.model, iterate, phase_qp, material)

        load_form = WeakForm(bilinear=(q, trial, c) -> 0.0,
                             linear=(q, c) -> stress_test_channels(cache.stresses[q.point], c),
                             symmetric=false, component_aware=true)
        residual = assemble_vector(mech.model, (loadform(mech.u, load_form),); threaded=false)
        residual_norm = norm(residual)
        if residual_norm <= max(newton_atol, newton_rtol * max(1.0, norm(coefficients)))
            converged = true
            break
        end

        block_form = WeakForm(bilinear=(q, trial, c) -> stress_test_channels(cache.tangents[q.point] *
                                                                             trial_strain(trial.gradient,
                                                                                          trial.component),
                                                                             c),
                              linear=(q, c) -> 0.0, symmetric=false, component_aware=true)
        tangent = assemble_matrix(mech.model, (block(mech.u, mech.u, block_form),); symmetric=false,
                                  threaded=false)
        delta = tangent \ (-residual)
        coefficients .+= delta
        if norm(delta) <= newton_rtol * max(1.0, norm(coefficients))
            converged = true
            break
        end
    end

    return (; mech..., coefficients, iterations,
            solution=solution(mech.model, coefficients; method=:mechanics_newton,
                              residual=residual_norm, converged))
end

# ── Reaction force on the top boundary ───────────────────────────────────────
#
# Post-process the reaction force on `y = 1` as a public-API
# [`boundary_integral`](@ref) of the vertical Cauchy stress over the
# top facet. The helper iterates the admissible boundary regions on
# the selected face — accounting for every level whose own face
# coincides with `y = 1` (base + any overlay touching the top edge) —
# and applies a Gauss rule whose order is the per-axis maximum of
# `recommended_quadrature_order(level.basis, level.order)` over the
# region's covering parents, so reaction integration is at least as
# accurate as the volume assembly on the same polynomial order.

function vertical_reaction(displacement, damage_solution, phase_state, material)
    return boundary_integral(displacement.model; on=boundary(axis=2, side=:upper)) do q
        strain = displacement_strain(displacement.solution, displacement.model, displacement.u, q.x)
        phase = clamp(value(damage_solution, phase_state.model, phase_state.damage, q.x), 0.0, 1.0)
        return split_stress(strain, phase, material)[2]
    end
end

# ── State transfer on mask growth ────────────────────────────────────────────
#
# When the strip mask grows, the active basis changes and every piece
# of state needs to migrate to the new space:
#
#   1. `transfer!(damage,        …, backend=Rewire())`  — exact dof copy.
#   2. `transfer!(displacement,  …, backend=Rewire())`  — exact dof copy.
#   3. `rbf_transfer_history(…)` — RBF + P0 transfer of `H`, monotone-
#      max-clamped against `initial_history` and the current `ψ⁺(ε)`.
#   4. **Equilibrium step.** Re-solve the phase and the displacement at
#      the same applied displacement to absorb the small residual the
#      RBF transfer leaves behind. The next load step then proceeds
#      from a converged state. This step is standard FCM-remeshing
#      practice (Sartorti & Düster 2024, Sec. 3.3.1).

function transfer_state_to_space(state, V_new, material)
    new_phase_state = build_phase_state(V_new)
    rewired_damage = transfer!(state.damage_solution, state.phase_state.model,
                               new_phase_state.model; backend=Rewire())
    new_phase_state = (; new_phase_state..., coefficients=copy(rewired_damage.coefficients))

    new_displacement_model = build_displacement_model(V_new, state.applied)
    rewired_u = transfer!(state.displacement.solution, state.displacement.model,
                          new_displacement_model.model; backend=Rewire())
    new_displacement = (; new_displacement_model..., coefficients=copy(rewired_u.coefficients),
                        iterations=0, solution=rewired_u)

    new_history = rbf_transfer_history(state, new_phase_state, new_displacement, material)

    # Equilibrium step (Sartorti & Düster 2024, Sec. 3.3.1): re-solve the phase
    # and displacement at the same applied displacement to absorb the small
    # residual the RBF transfer leaves behind. The next load step then proceeds
    # from a converged state.
    settled_phase, damage_solution = solve_phase(new_phase_state, new_history)
    new_phase_state = (; new_phase_state..., coefficients=settled_phase)
    settled_displacement = solve_displacement(V_new, state.applied, new_displacement.coefficients,
                                              damage_solution, new_phase_state, material)

    return (; state..., history=new_history, phase_state=new_phase_state,
            damage_solution=damage_solution, displacement=settled_displacement,
            displacement_coefficients=settled_displacement.coefficients)
end

# Decide whether the strip mask needs to grow and, if so, rebuild the
# space and migrate the state. Three early-out branches before the
# (expensive) `transfer_state_to_space`:
#
#   1. no cells crossed `mask_threshold` since the last update — the
#      triggered set is unchanged, return immediately;
#   2. cells crossed but the dilated halo around them is the same set
#      we already had — the strip mask is unchanged, only book-keeping
#      changes;
#   3. otherwise: rebuild the space and run the full transfer pipeline.
function maybe_grow_strip_mask(V, state, material, triggered_mask, strip_mask)
    new_triggered = update_triggered_mask(triggered_mask, state.damage_solution, state.phase_state)
    new_triggered == triggered_mask && return V, state, false, triggered_mask, strip_mask
    new_strip_mask = dilate_mask(new_triggered, mask_dilation)
    new_strip_mask == strip_mask && return V, state, false, new_triggered, new_strip_mask
    V_new = build_space(new_strip_mask)
    new_state = transfer_state_to_space(state, V_new, material)
    return V_new, new_state, true, new_triggered, new_strip_mask
end

# ── Initial state and load step ──────────────────────────────────────────────
#
# `initial_state(V)` produces the at-rest state (zero displacement,
# Gaussian-seed history, bootstrap damage). `solve_load_step(V, …, du)`
# advances the load by `du`: update history, solve phase, solve
# displacement, compute reaction, return a tentative next state and the
# diagnostics the adaptive stepping policy needs.

function initial_state(V)
    phase_state = build_phase_state(V)
    # Initial-from-notch phase solve provides the `previous_damage` field that the
    # viscous regularization in the first `solve_phase` will compare against.
    form = initial_phase_form()
    matrix = assemble_matrix(phase_state.model, block(phase_state.damage, phase_state.damage, form))
    rhs = assemble_vector(phase_state.model, loadform(phase_state.damage, form))
    phase_state = (; phase_state..., coefficients=matrix \ rhs)
    history = seed_history(phase_state.model)
    phase_coefficients, damage_solution = solve_phase(phase_state, history)
    phase_state = (; phase_state..., coefficients=phase_coefficients)
    displacement_model = build_displacement_model(V, 0.0)
    displacement_coefficients = zeros(active_unknowns(displacement_model.model))
    displacement = (; displacement_model..., coefficients=displacement_coefficients,
                    solution=solution(displacement_model.model, displacement_coefficients;
                                      method=:initial_u))
    return (; applied=0.0, load=0.0, history, phase_state, damage_solution, displacement,
            displacement_coefficients)
end

function solve_load_step(V, material, state, du)
    trial_history = copy(state.history)
    update_history!(trial_history, state.displacement, material)

    phase_state = state.phase_state
    previous_damage = solution(phase_state.model, phase_state.coefficients; method=:previous_phase)
    phase_coefficients, damage_solution = solve_phase(phase_state, trial_history)
    phase_state = (; phase_state..., coefficients=phase_coefficients)
    delta_d = phase_increment(previous_damage, damage_solution, phase_state)

    displacement = solve_displacement(V, state.applied + du, state.displacement_coefficients,
                                      damage_solution, phase_state, material)
    load_value = vertical_reaction(displacement, damage_solution, phase_state, material)

    next_state = (; applied=state.applied + du, load=load_value, history=trial_history, phase_state,
                  damage_solution, displacement,
                  displacement_coefficients=displacement.coefficients)
    return (; state=next_state, du, phase_increment=delta_d)
end

# ── Adaptive load stepping ───────────────────────────────────────────────────
#
# The applied-displacement increment `du` is grown or shrunk based on
# the per-step phase increment so the load step never advances the
# damage too far in one go. Three policy primitives:
#
#   * `clamp_du(du)`     — clamp to `[min_displacement_step,
#                          max_displacement_step]`.
#   * `next_du(du, Δd)`  — grow `du` by `step_growth_factor` when the
#                          phase change `Δd` is small, shrink it when
#                          `Δd` approaches `max_phase_increment`,
#                          otherwise keep it.
#   * `reject_reason(…)` — enumerate the reasons a candidate step is
#                          unacceptable. Currently: mechanics did not
#                          converge, the load value is non-finite,
#                          the phase increment is non-finite or too
#                          large, or the reaction force dropped by
#                          more than `max_force_drop` of its previous
#                          value (indicating an unstable load step).

clamp_du(du) = clamp(du, min_displacement_step, max_displacement_step)

function next_du(du, phase_change)
    phase_change < easy_phase_increment && return clamp_du(step_growth_factor * du)
    phase_change > 0.75max_phase_increment && return clamp_du(0.75du)
    return clamp_du(du)
end

function reject_reason(candidate, previous_load)
    s = candidate.state
    s.displacement.solution.diagnostics.converged || return :mechanics
    isfinite(s.load) || return :load
    isfinite(candidate.phase_increment) || return :phase_increment
    isfinite(s.damage_solution.diagnostics.residual_norm) || return :phase_residual
    at_min = candidate.du <= min_displacement_step * (1.0 + sqrt(eps()))
    at_min && return nothing
    candidate.phase_increment > max_phase_increment && return :phase_increment
    drop = abs(previous_load) > 1.0e-12 ? max(0.0, previous_load - s.load) / abs(previous_load) :
           0.0
    drop > max_force_drop && return :force_drop
    return nothing
end

# ── VTK output ───────────────────────────────────────────────────────────────
#
# The history field `H` lives on quadrature points, not on the damage
# basis. To visualise it alongside `u` and `d`, project it onto the
# damage field by solving the target-side mass system once per snapshot.

# L²-project the per-quadrature-point history onto the damage field so
# it can be exported alongside the damage solution. Reuses
# `phase_state.mass_factor` from `build_phase_state` so the mass matrix
# is only factorised once per phase model.
function history_solution(phase_state, history)
    rhs = assemble_vector(phase_state.model,
                          loadform(phase_state.damage,
                                   WeakForm(bilinear=(q, trial) -> 0.0,
                                            linear=q -> history[q.point], symmetric=false));
                          threaded=false)
    coefficients = phase_state.mass_factor \ rhs
    residual = norm(phase_state.mass_matrix * coefficients - rhs)
    return solution(phase_state.model, coefficients; method=:history_projection, residual)
end

function write_snapshot(output_root, snapshot_index, state, material)
    mkpath(output_root)
    path = joinpath(output_root, "solution_$(lpad(string(snapshot_index), 4, '0'))")
    displacement = state.displacement
    phase_model = state.phase_state.model
    damage = state.phase_state.damage
    history_field = history_solution(state.phase_state, state.history)
    write_vtk(path, state.damage_solution, phase_model; subdivisions=3,
              point_data=(displacement=(u, c, x, xi) -> value(displacement.solution,
                                                              displacement.model, displacement.u, x),
                          phase=(u, c, x, xi) -> clamp(u(c, xi), 0.0, 1.0),
                          history=(u, c, x, xi) -> max(value(history_field, phase_model, damage, x),
                                                       positive_energy(displacement_strain(displacement.solution,
                                                                                           displacement.model,
                                                                                           displacement.u,
                                                                                           x),
                                                                       material)),
                          notch=(u, c, x, xi) -> reference_notch_profile(x)),)
    return path
end

# ── Main loop ────────────────────────────────────────────────────────────────
#
# Outer driver. Bootstrap a strip mask + space + state, then advance
# the applied displacement adaptively until the target is reached or
# `max_accepted_steps` runs out. The order of operations inside the
# loop is:
#
#   1. Pick `du` (current candidate increment).
#   2. Try `solve_load_step`. Any exception that is not "we're already
#      at the minimum step" gets caught, reported, and the step shrinks.
#   3. Check `reject_reason`. A rejected candidate triggers a shrink
#      and a retry; an accepted candidate is committed.
#   4. After accepting: try to grow the strip mask using the new damage
#      field; if it grows, `transfer_state_to_space` migrates the state
#      and the reaction rule rebuilds for the new space.
#   5. Optional VTK snapshot when the reaction force has moved by at
#      least `export_force_delta` since the last export.
#   6. Pick the next `du` from `next_du(du, Δd)` and loop.

function main()
    triggered_mask = bootstrap_triggered_mask()
    strip_mask = dilate_mask(triggered_mask, mask_dilation)
    mask_active_initial = count(strip_mask)
    V = build_space(strip_mask)
    material = Material(lambda, mu, residual_stiffness)
    state = initial_state(V)
    snapshot_index = 0
    last_exported_load = 0.0
    last_vtk = nothing
    du = clamp_du(initial_displacement_step)
    accepted_steps = 0
    attempted_steps = 0
    rejected_steps = 0
    mask_growths = 0

    if write_output
        last_vtk = write_snapshot(output_root, snapshot_index, state, material)
        snapshot_index += 1
    end

    while state.applied < target_displacement - 10eps(max(1.0, target_displacement)) &&
        accepted_steps < max_accepted_steps
        du = min(clamp_du(du), target_displacement - state.applied)
        attempted_steps += 1

        candidate = try
            solve_load_step(V, material, state, du)
        catch err
            du <= min_displacement_step * (1.0 + sqrt(eps())) && rethrow()
            rejected_steps += 1
            du = clamp_du(step_shrink_factor * du)
            println("reject step=", accepted_steps + 1, " reason=exception(", typeof(err),
                    ") next_du=", du)
            continue
        end

        reason = reject_reason(candidate, state.load)
        if reason !== nothing
            shrunken = clamp_du(step_shrink_factor * candidate.du)
            shrunken < candidate.du ||
                error("adaptive load step failed at minimum du=$(candidate.du), reason=$reason")
            rejected_steps += 1
            du = shrunken
            println("reject step=", accepted_steps + 1, " u=", candidate.state.applied, " du=",
                    candidate.du, " reason=", reason, " Δd=", candidate.phase_increment, " F=",
                    candidate.state.load, " next_du=", du)
            continue
        end

        accepted_steps += 1
        state = candidate.state

        can_continue = state.applied < target_displacement - 10eps(max(1.0, target_displacement)) &&
                       accepted_steps < max_accepted_steps
        mask_grew = false
        if can_continue
            V, state, mask_grew, triggered_mask, strip_mask = maybe_grow_strip_mask(V, state,
                                                                                    material,
                                                                                    triggered_mask,
                                                                                    strip_mask)
        end
        if mask_grew
            mask_growths += 1
        end

        at_end = state.applied >= target_displacement - 10eps(max(1.0, target_displacement)) ||
                 accepted_steps >= max_accepted_steps
        if write_output && (abs(state.load - last_exported_load) >= export_force_delta || at_end)
            last_vtk = write_snapshot(output_root, snapshot_index, state, material)
            snapshot_index += 1
            last_exported_load = state.load
        end

        println("step=", accepted_steps, " u=", state.applied, " du=", candidate.du, " F=",
                state.load, " Δd=", candidate.phase_increment, " strip=", count(strip_mask),
                mask_grew ? " grew" : "", " it_u=", state.displacement.iterations, " |r_d|=",
                state.damage_solution.diagnostics.residual_norm, " |r_u|=",
                state.displacement.solution.diagnostics.residual_norm)

        du = min(next_du(candidate.du, candidate.phase_increment),
                 target_displacement - state.applied)
    end

    state.applied < target_displacement - 10eps(max(1.0, target_displacement)) &&
        @warn "adaptive load stepping stopped before the target displacement" applied=state.applied target_displacement accepted_steps max_accepted_steps

    report = diagnostics(state.displacement.model, state.displacement.solution)
    print_run_report("Phase-field single-edge-notched tension baseline", report;
                     parameters=(:domain => omega,
                                 :base => (; cells=(cells_per_axis, cells_per_axis), order),
                                 :strip =>
                                     (; cells=(strip_cells_x, strip_cells_y), order=strip_order,
                                      half_height=strip_half_height,
                                      mask=(; active_initial=mask_active_initial,
                                            active_final=count(strip_mask),
                                            total=prod(strip_mesh_template.cells),
                                            growths=mask_growths)),
                                 :material => (; lambda, mu, Gc, ell, eta_viscosity),
                                 :stepping => (; accepted_steps, attempted_steps, rejected_steps),
                                 :final => (; displacement=state.applied, load=state.load)),
                     output=last_vtk)
end

main()
