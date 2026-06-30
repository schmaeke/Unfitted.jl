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
using Tensors

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

# Growing-mask knobs. `mask_threshold` is set well above the residual
# base-overlay smear (~0.045) so background damage never grows the
# mask, only the actual crack tip does. `mask_dilation` is the
# Chebyshev halo around the triggered set, sized so each accepted load
# step has free cells ahead of the tip to propagate into.
const mask_threshold = 0.5
const mask_dilation = 2
const bootstrap_halo = 2.0 * ell

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
# Miehe tension/compression split (Sec. 3): only the *tensile* part of
# the elastic energy drives damage, so a crack opens under tension but
# closes without further degradation under compression. With strain
# and stress as `SymmetricTensor{2,2,Float64}`, the off-diagonal
# `ε[1,2] = εxy` enters double contractions correctly without any
# factor-of-two Voigt bookkeeping.

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

# Symmetric 2D strain/stress and minor-symmetric 4th-order tangent — the
# per-quadrature-point cache types. Per-q-point fields (history, frozen
# damage, …) are stored as plain `Vector`s indexed by `q.point`; the
# mechanics and phase models share the same space `V`, so they share
# this indexing.
const StrainStress = SymmetricTensor{2,2,Float64,3}
const StressTangent = SymmetricTensor{4,2,Float64,9}

# Distance from `x` to the closest point on the initial notch segment
# Γ₀ = {(x, 0.5) : 0 ≤ x ≤ 0.5}.
function distance_to_notch(x)
    projection = clamp((x[1] - notch_left[1]) / (notch_tip[1] - notch_left[1]), 0.0, 1.0)
    closest = notch_left + projection * (notch_tip - notch_left)
    return norm(SVector(x[1], x[2]) - closest)
end

# Smooth exponential decay around the notch for visualising the
# crack-path band in the VTK `notch` channel.
reference_notch_profile(x) = exp(-distance_to_notch(x) / ell)

# Initial history-field seed: a Gaussian bump in `ψ⁺` along the notch,
# tall enough to push `H` above the regularisation floor `Gc / ℓ` so
# the bootstrap phase solve produces the diffuse-notch field without
# needing a resolved geometric crack. Truncated outside `x ≤ notch_tip
# + 2ℓ` so the far field stays clean.
function initial_history(x)
    x[1] <= notch_tip[1] + 2ell || return 0.0
    width = notch_history_width_factor * ell
    return notch_history_factor * Gc / ell * exp(-(distance_to_notch(x) / width)^2)
end

# Tension/compression split of a 2D symmetric strain via the
# eigenvector-free identity
#
#     ε⁺ = (ε + |ε|) / 2,    ε⁻ = ε − ε⁺,
#
# with the matrix absolute value `|ε|` evaluated in closed form by the
# Cayley-Hamilton theorem: for 2D symmetric `ε`,
# `ε² = tr(ε) ε − det(ε) I` and `|ε|² = ε²`, giving
#
#     |ε| = (tr(ε) ε + 2 ⟨−det(ε)⟩₊ I) / √(tr(ε²) + 2 |det(ε)| + δ).
#
# `tr(ε)`, `det(ε)`, and `tr(ε²)` are polynomial in the components, so
# automatic differentiation through this formula has no `1 / (e₁ − e₂)`
# blow-up at a degenerate principal frame — that singularity was an
# artifact of the spectral formula, not a property of ε⁺ itself. The
# regularising offset `δ = eps(Float64)` removes the unique remaining
# 0 / 0 at exactly `ε = 0`; its effect on the result at any non-zero
# strain is below floating-point precision, and at `ε = 0` it returns
# `ε⁺ = 0` as required.
function positive_negative_strain(strain::SymmetricTensor{2,2})
    detε = det(strain)
    abs_strain = (tr(strain) * strain + 2 * max(-detε, 0.0) * one(strain)) /
                 sqrt(tr(strain ⋅ strain) + 2 * abs(detε) + eps(Float64))
    eps_plus = symmetric((strain + abs_strain) / 2)
    return eps_plus, strain - eps_plus
end

# Tensile elastic-energy density `ψ⁺(ε) = ½ λ ⟨tr ε⟩₊² + μ (ε⁺ ⊡ ε⁺)`.
# Drives the history field `H` and so is the sole source of damage
# growth in the phase equation.
function positive_energy(strain::SymmetricTensor{2,2}, material::Material)
    eps_plus, _ = positive_negative_strain(strain)
    trp = max(tr(strain), 0.0)
    return 0.5 * material.lambda * trp^2 + material.mu * (eps_plus ⊡ eps_plus)
end

# Cauchy stress under the Miehe split:
#
#     σ(ε, d) = g(d) σ⁺(ε) + σ⁻(ε),
#     σ⁺ = λ ⟨tr ε⟩₊ I + 2μ ε⁺,    σ⁻ = λ ⟨tr ε⟩₋ I + 2μ ε⁻,
#     g(d) = (1 − d)² + k.
#
# `phase` is clamped to `[0, 1]` so an over-damaged Newton iterate
# cannot blow up the degradation factor.
function split_stress(strain::SymmetricTensor{2,2}, phase, material::Material)
    eps_plus, eps_minus = positive_negative_strain(strain)
    g = (1.0 - clamp(phase, 0.0, 1.0))^2 + material.residual_stiffness
    trp = max(tr(strain), 0.0)
    trm = min(tr(strain), 0.0)
    I2 = one(strain)
    positive = material.lambda * trp * I2 + 2 * material.mu * eps_plus
    negative = material.lambda * trm * I2 + 2 * material.mu * eps_minus
    return g * positive + negative
end

# Cauchy stress and tangent `(C, σ)` at one strain via automatic
# differentiation through `split_stress`. AD-clean at every strain
# configuration thanks to the Cayley-Hamilton form of
# `positive_negative_strain`.
function stress_and_tangent(strain::SymmetricTensor{2,2}, phase, material::Material)
    return Tensors.gradient(ε -> split_stress(ε, phase, material), strain, :all)
end

# Strain `ε(u) = sym(∇u)` at an arbitrary physical point from a
# displacement solution.
function displacement_strain(solution, model, u, x)
    return symmetric(Tensor{2,2}((i, j) -> field_gradient(solution, model, u, x, i)[j]))
end

# Strain at a quadrature point from the current iterate `q.state`.
strain_state(state) = symmetric(gradient_tensor(state, :u, Val(2)))

# Damage `d ∈ [0, 1]` at a single physical point.
function damage_at(damage_solution, phase_state, x)
    clamp(value(damage_solution, phase_state.model, phase_state.damage, x), 0.0, 1.0)
end

# Per-quadrature-point damage vector, indexed by `q.point`.
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
# Both `u` and `d` live on the same superposition space (a base level
# plus the strip overlay carrying the current activation mask), so a
# single rebuild refreshes the basis for both fields. Total-degree mode
# keeps the dof count manageable at high `p`.
function build_space(strip_mask)
    V = space(omega; cells=(cells_per_axis, cells_per_axis), order=order, mode=:trunk)
    return overlay(V, strip_domain; cells=(strip_cells_x, strip_cells_y), order=strip_order,
                   active=strip_mask)
end

# ── Growing strip mask ───────────────────────────────────────────────────────
#
# The triggered set is monotone: a cell, once triggered, never
# untriggers. `bootstrap_triggered_mask` seeds it from a geometric halo
# around the notch; `update_triggered_mask` adds cells whose damage has
# crossed `mask_threshold`; `dilate_mask` then expands the set by a
# Chebyshev halo so the next load step has free cells ahead of the
# crack tip to propagate into.

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
        damage_at(damage_solution, phase_state, c) >= mask_threshold && (new_triggered[ci] = true)
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
    # Symmetric SENT setup: u₂ prescribed on top and bottom edges; a
    # codim-D pin at the bottom-left corner removes the u₁ rigid-body
    # mode while leaving u₁ free on the edges, so the elastic field
    # stays symmetric about y = 0.5 and the crack stays on the midline.
    bottom = boundary(axis=2, side=:lower)
    top = boundary(axis=2, side=:upper)
    pin = boundary((axis=1, side=:lower), (axis=2, side=:lower))
    problem = Problem((u,);
                      dirichlet=[dirichlet(0.0; on=bottom, field=u, component=2),
                                 dirichlet(applied_displacement; on=top, field=u, component=2),
                                 dirichlet(0.0; on=pin, field=u, component=1)], symmetric=false)
    return (; model=prepare(problem), u)
end

# Phase-field weak form (Miehe et al. 2010 with viscous regularisation):
#
#     a(d, v) = ∫_Ω (Gc/ℓ + 2 H + η/τ) d·v dx  +  ∫_Ω Gc ℓ ∇v·∇d dx,
#     ℓ(v)   = ∫_Ω (2 H + (η/τ) dₙ)·v dx,
#
# with `H = max_{s ≤ t} ψ⁺(ε(x, s))` the irreversible history,
# `η` the viscosity, `τ = time_step`, and `dₙ` the damage from the
# previous load step. The `Gc ℓ ∇v·∇d` term is the Γ-convergence
# regularisation setting the diffuse crack bandwidth; the (Gc/ℓ + …)
# `d·v` term is the local restoring force.
#
# `history_at(q)` and `previous_damage_at(q)` factor out where the per-
# q-point data comes from. The bootstrap solve at `initial_state`
# evaluates `H` from `initial_history(q.x)` directly and disables
# viscous regularisation (no `dₙ` to compare against yet); subsequent
# solves read both from per-q-point vectors via the stable index
# `q.point`.
function phase_form(history_at, previous_damage_at; viscous=true)
    η_over_τ = viscous ? eta_viscosity / time_step : 0.0
    return WeakForm(bilinear=(q, trial) -> TestChannels((Gc / ell + 2 * history_at(q) + η_over_τ) *
                                                        trial.value, Gc * ell * trial.gradient),
                    linear=q -> 2 * history_at(q) + η_over_τ * previous_damage_at(q),
                    symmetric=true)
end

# Solve the regularised phase-field problem at fixed history. SPD, so
# a direct solve is enough. Matrix and rhs are rebuilt every call
# because `history` and `previous_damage` change between load steps;
# `phase_state.mass_factor` is reused only by `history_solution` for
# the L²-projection of `H` onto the damage space.
function solve_phase(phase_state, history)
    previous = solution(phase_state.model, phase_state.coefficients; method=:previous_phase)
    previous_damage = frozen_damage(phase_state.model, previous)
    form = phase_form(q -> history[q.point], q -> previous_damage[q.point])
    matrix = assemble_matrix(phase_state.model, block(phase_state.damage, phase_state.damage, form);
                             threaded=false)
    rhs = assemble_vector(phase_state.model, loadform(phase_state.damage, form); threaded=false)
    coefficients = matrix \ rhs
    residual = norm(matrix * coefficients - rhs)
    return coefficients, solution(phase_state.model, coefficients; method=:phase_direct, residual)
end

# ── History transfer on mask growth ──────────────────────────────────────────
#
# `u` and `d` migrate by exact raw-key rewiring (`transfer(...,
# via=Rewire())`); because the new active basis is a strict
# superset of the old one, copying coefficients dof-by-dof reproduces
# the source field pointwise — no L² projection, no inverse map.
#
# Per-q-point history `H` migrates by RBF + P0 (Sartorti & Düster 2024
# Sec. 3.3.1), with a monotone max-floor against `initial_history(x)`
# and the current `ψ⁺(ε(x))` so damage irreversibility is preserved
# across the transfer.
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

# ── Mechanics: stress/tangent cache, channel bridge, Newton solve ────────────
#
# Newton's method on `div σ(u, d) = 0` at every load step. The
# per-iteration recipe: cache `σ` and `C = ∂σ/∂ε` at every q-point from
# the current iterate, assemble `r = ∫ σ : ∇v` as a load-form
# contribution and `K_T = ∫ ∇v : C : ∇u` as a block, solve `K_T Δu =
# −r`. Forms are `component_aware = true, symmetric = false`: the Miehe
# split breaks the major symmetry of the tangent in general, so the
# symmetric-assembly fast path is unavailable.

# Cache Cauchy stress and tangent at every q-point of the current
# iterate. Both depend on the current strain and the *frozen* damage
# from the most recent phase solve, so the cache is rebuilt per Newton
# iteration.
function build_mechanics_cache(model, iterate, phase_qp, material)
    n = nquadpoints(model)
    stresses = Vector{StrainStress}(undef, n)
    tangents = Vector{StressTangent}(undef, n)
    foreach_quadrature_point(model; state=iterate) do q
        strain = strain_state(q.state)
        phase = phase_qp[q.point]
        C, σ = stress_and_tangent(strain, phase, material)
        stresses[q.point] = σ
        tangents[q.point] = C
    end
    return (; stresses, tangents)
end

# Channel for a symmetric Cauchy stress paired with the `component`-th
# test direction. The bilinear contribution `σ : ε_test` collapses, by
# `σ`'s symmetry and the rank-1 form `ε(v) = ½(eₖ ⊗ ∇N + ∇N ⊗ eₖ)`, to
# `(σ ⋅ eₖ) · ∇N` — i.e. the gradient channel of `TestChannels` is the
# `k`-th row of `σ`.
function stress_test_channels(stress::SymmetricTensor{2,2,Float64}, component::Int)
    return TestChannels(0.0, Vec{2,Float64}((stress[component, 1], stress[component, 2])))
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

        # Residual: ∫ σ : ∇v dx as a load form with cached σ.
        residual_form = WeakForm(bilinear=(q, trial, c) -> 0.0,
                                 linear=(q, c) -> stress_test_channels(cache.stresses[q.point], c),
                                 symmetric=false, component_aware=true)
        residual = assemble_vector(mech.model, (loadform(mech.u, residual_form),); threaded=false)
        residual_norm = norm(residual)
        if residual_norm <= max(newton_atol, newton_rtol * max(1.0, norm(coefficients)))
            converged = true
            break
        end

        # Tangent: ∫ ∇v : C : ∇u dx. The bilinear callback contracts
        # the cached C with the rank-1 symmetric trial gradient and
        # passes the result paired with the test component `c`.
        tangent_form = WeakForm(bilinear=(q, trial, c) -> stress_test_channels(cache.tangents[q.point] ⊡
                                                                               symmetric_gradient(trial),
                                                                               c),
                                linear=(q, c) -> 0.0, symmetric=false, component_aware=true)
        tangent = assemble_matrix(mech.model, (block(mech.u, mech.u, tangent_form),);
                                  symmetric=false, threaded=false)
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
# `boundary_integral` of the vertical Cauchy stress over `y = 1`. The
# helper covers every level whose own face coincides with the selected
# boundary (base + any overlay touching the top edge) and integrates
# at the same polynomial order as the volume assembly.

function vertical_reaction(displacement, damage_solution, phase_state, material)
    return boundary_integral(displacement.model; on=boundary(axis=2, side=:upper)) do q
        strain = displacement_strain(displacement.solution, displacement.model, displacement.u, q.x)
        phase = damage_at(damage_solution, phase_state, q.x)
        return split_stress(strain, phase, material)[2, 2]
    end
end

# ── State transfer on mask growth ────────────────────────────────────────────
#
# When the strip mask grows, the active basis changes and every piece
# of state migrates to the new space: `u` and `d` by exact dof
# rewiring, `H` by RBF + P0. A final equilibrium re-solve at the same
# applied displacement absorbs the small residual the RBF transfer
# leaves behind, so the next load step starts from a converged state
# (Sartorti & Düster 2024, Sec. 3.3.1).

function transfer_state_to_space(state, V_new, material)
    new_phase_state = build_phase_state(V_new)
    rewired_damage = transfer(state.damage_solution, state.phase_state.model, new_phase_state.model;
                              via=Rewire())
    new_phase_state = (; new_phase_state..., coefficients=copy(rewired_damage.coefficients))

    new_displacement_model = build_displacement_model(V_new, state.applied)
    rewired_u = transfer(state.displacement.solution, state.displacement.model,
                         new_displacement_model.model; via=Rewire())
    new_displacement = (; new_displacement_model..., coefficients=copy(rewired_u.coefficients),
                        iterations=0, solution=rewired_u)

    new_history = rbf_transfer_history(state, new_phase_state, new_displacement, material)

    settled_phase, damage_solution = solve_phase(new_phase_state, new_history)
    new_phase_state = (; new_phase_state..., coefficients=settled_phase)
    settled_displacement = solve_displacement(V_new, state.applied, new_displacement.coefficients,
                                              damage_solution, new_phase_state, material)

    return (; state..., history=new_history, phase_state=new_phase_state,
            damage_solution=damage_solution, displacement=settled_displacement,
            displacement_coefficients=settled_displacement.coefficients)
end

# Three early-out levels before the expensive transfer: (1) no cells
# crossed `mask_threshold`, (2) the dilated halo is unchanged anyway,
# (3) rebuild and migrate.
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

function initial_state(V)
    phase_state = build_phase_state(V)
    # Bootstrap solve from `initial_history` (no viscous term, no `dₙ`
    # to compare against yet). Produces the diffuse-notch damage field
    # that the first proper `solve_phase` will use as `previous_damage`.
    form = phase_form(q -> initial_history(q.x), _ -> 0.0; viscous=false)
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
# `du` is grown or shrunk by the per-step phase change so the load
# never advances the damage too far in one increment. `next_du` is the
# accept-side policy; `reject_reason` is the diagnostic the outer loop
# uses to decide whether a candidate must be retried with a smaller du
# (mechanics non-convergence, non-finite load or phase increment,
# phase increment exceeding `max_phase_increment`, or a reaction-force
# drop exceeding `max_force_drop` of the previous value).

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
# The history field `H` lives on quadrature points. To visualise it
# alongside `u` and `d`, it is L²-projected onto the damage field once
# per snapshot, reusing `phase_state.mass_factor` so the mass matrix
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
# Bootstrap a strip mask + space + state, then advance the applied
# displacement adaptively until the target is reached or
# `max_accepted_steps` runs out. Per iteration: try `solve_load_step`
# at the current `du`; shrink and retry on exception or
# `reject_reason`; on acceptance, grow the strip mask if damage has
# crossed `mask_threshold`, optionally write a VTK snapshot when the
# reaction force has moved by `export_force_delta`, and pick the next
# `du` from `next_du`.

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
