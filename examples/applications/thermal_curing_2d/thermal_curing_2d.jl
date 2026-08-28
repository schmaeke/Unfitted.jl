#=
Thermal Curing Of A Thermoset, On A Cure Front That Refines Itself
==================================================================

A resin is heated by a short pulse from an embedded heater. Wherever the
temperature climbs above a cure temperature the resin cross-links, and the
cross-linking is **irreversible**: once cured, always cured, even after the
part has cooled back towards ambient. Cured resin conducts heat better than
uncured resin, so the material the cure front leaves behind changes the
operator that drives the front. The model is

    ∂ₜu − ∇·(κ(χ) ∇u) = Q(x, t)    in Ω = (0, 1)²,   u = 0 on ∂Ω,  u(·, 0) = 0,
    ∂ₜχ = A (1 − χ) g(u),                            χ(·, 0) = 0,
    κ(χ) = κ_uncured + (κ_cured − κ_uncured) χ,

with `u` the temperature above ambient, `χ ∈ [0, 1]` the cure fraction, `A`
the cure rate, and `g` a ramp that is `0` below the cure temperature and `1` a
little above it — the onset of cure kinetics, which is narrow but not a step.
`Q` is a disk-shaped heater that runs for the first `pulse_end` of the
transient and is then switched off, so the part heats, cures, and cools again
within one run.

Three features of the package carry this problem, and none of them is
decoration.

**1. `QuadField` — material state that is not a finite element unknown.**
χ lives at the quadrature points. It is P0 per point and discontinuous —
nothing in the weak form differentiates it, and there is no reason for it to
be continuous — so it has no degrees of freedom and cannot be a `Solution`.
What makes it a genuine *state* rather than a quantity one could recompute is
the irreversibility. Integrating the rate law gives

    χ(x, t) = 1 − exp( −A ∫₀ᵗ g(u(x, s)) ds ),

which depends on the entire history of `u` at `x` and not on its current
value. The run below prints the peak temperature at the final time next to the
cure temperature: the part has cooled well below the cure temperature
everywhere, and χ is nevertheless close to 1 over a substantial area.
Recomputing χ from the final temperature field would return zero. That is why the state has to be stored,
and why it has to survive a mesh change.

**2. Picard, not Newton.** κ depends on χ, χ depends on u, so the implicit
step is nonlinear:

    (M/Δt + K(χⁿ⁺¹)) uⁿ⁺¹ = M uⁿ/Δt + f(tₙ₊₁),    χⁿ⁺¹ = update(χⁿ, uⁿ⁺¹).

The fixed-point iteration below freezes χ, solves the linear system,
recomputes χ from the new temperature, and repeats until the temperature
increment stops moving. A Newton method would need the derivative of the
composition κ ∘ χ ∘ u — a second per-quadrature-point state, ∂χ/∂u, carried
and updated alongside χ, plus the tangent block that contracts it against
∇u ⋅ ∇v. That is roughly twice this file for no pedagogical gain: the physical
coupling here is weak (κ varies by a factor of three across a step in which χ
moves by at most `A Δt`), so Picard contracts linearly at a rate near 0.2.
The per-step iteration counts and the first step's full residual history are
printed, so the reader can watch it contract rather than take it on trust. The
counts drop to two once the heater is off and the part has cooled through the
cure onset: with the gate shut χ stops moving, the second iterate reproduces the
first exactly, and the step terminates at once.

**3. Activation-driven refinement, and the transfer that makes it legal.**
The interesting physics is a thin band: the cells that are *partly* cured.
Everything behind the front sits at χ = 1 and everything ahead of it at χ = 0,
and neither needs resolution. So an overlay spanning Ω is activated only on
cells holding a partially cured quadrature point, grown by one cell in every
direction so the front cannot outrun the refined region within a step. As the
front advances the active set changes, and with it the integration regions and
the quadrature-point cloud — the script prints the point count on either side
of every rebuild, and they differ every time.

A changed quadrature-point cloud means χ, which is indexed by point, is
meaningless on the new discretisation. It is moved with

    transfer(χ, old_model, new_model; via = RBFP0())

— an inverse-multiquadric RBF interpolation with a constant polynomial
extension, exact for constant fields by construction and approximately
linear-exact, following

    R. Sartorti, A. Düster, "Data transfer within a finite cell remeshing
    approach applied to large deformation problems", Comput. Mech. 77 (2026),
    doi:10.1007/s00466-024-02486-0,

whose use case is precisely this one: history data surviving a remesh. It is an
interpolation and not a monotone projection, so it over- and undershoots where
the front is steepest — by more than a tenth of the state's whole range on the
sharpest step here. The run prints the largest excursion outside [0, 1] it
produced, and both consumers of χ clamp, because a cure fraction outside [0, 1]
is not a state this material has.

Note the **two-model** shape of that call, which is not an accident.
`activate!` / `deactivate!` mutate a model in place and bump its version, and a
`QuadField` is pinned to the version it was built against — so mutating the
model first and transferring afterwards is rejected with an `ArgumentError`
rather than silently reading a stale point cloud. The way to move state is
therefore to build a *new* model for the new active set with
`space(...; active = …)`, transfer off the old model while it is still valid,
and then let it go. That is what `simulate` does below, and it is why
`transfer` takes both models.

**What the run prints, and what to look for.** The transient is run twice,
identically, except that the second run throws χ away at every rebuild and
starts again from zero on the new mesh — which is what a code without a
history transfer effectively does. Both runs move the *temperature* across by
an L² projection either way, so the only difference between them is the
material state. The headline numbers are the total cured area

    cured area = ∫_Ω χ dx  ≈  Σ_q w_q χ_q

and the reach of the front, the furthest quadrature point from the heater with
χ > ½. With the transfer the cured region grows until the heater switches off
and then holds. Without it, the run ends with **no cured material at all** —
cured area and front radius both exactly zero — and that total loss is not an
accident of where the last rebuild fell. The refinement is driven by χ itself,
so a rebuild happens on every step in which the cure state moves, and the last
one of all happens when the partially-cured band finally empties. A scheme that
adapts to its own history state therefore erases that history at precisely the
steps where it matters. The two runs also end at different peak temperatures,
because κ(χ) is wrong in the run that lost χ: this is not a slightly less
accurate answer, it is a different physical problem. That is what `RBFP0` is
for.
=#

using Unfitted
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

# ── Problem data ─────────────────────────────────────────────────────────────

const omega = box((0.0, 0.0), (1.0, 1.0))
const base_cells, base_order = 8, 2
const front_cells, front_order = (16, 16), 2
const kappa_uncured, kappa_cured = 0.2, 0.6
const cure_temperature, cure_width, cure_rate = 1.2, 0.3, 12.0
# "Partly cured", for refinement purposes. The rate law approaches χ = 1
# geometrically and does not reach it in a finite number of steps, so the band is
# opened slightly at both ends; without that the fully cured core would stay
# flagged for ever, and the refined region would grow instead of following the
# front.
const cure_band = (0.02, 0.98)
const heater_center, heater_radius, heater_power, pulse_end = (0.3, 0.5), 0.12, 60.0, 0.35
const time_step, time_steps = 0.05, 10
const picard_tol, picard_max = 1.0e-9, 20

# Both χ consumers clamp, because `transfer` interpolates and the RBF
# interpolant is not monotone: carrying χ across the sharp cure front
# undershoots noticeably where the front is steepest (the run prints the largest
# excursion it saw). A conductivity below κ_uncured is not a material this
# problem contains, and a cure fraction outside [0, 1] is not a state it has.
kappa(chi) = kappa_uncured + (kappa_cured - kappa_uncured) * clamp(chi, 0.0, 1.0)
gate(u) = clamp((u - cure_temperature) / cure_width, 0.0, 1.0)

# One explicit step of the cure law. The increment is non-negative and is added
# to χⁿ rather than recomputed from u, so χ is monotone in time by construction.
# That monotonicity is the irreversibility, and it is what makes χ history.
advance(chi, u) = clamp(chi + time_step * cure_rate * (1 - chi) * gate(u), 0.0, 1.0)

# The heater as a `source` coefficient at time `t`. One closure shape for the
# whole run, so the assembly kernel specialises once rather than once per step.
function heater(t)
    x -> t <= pulse_end && sum(abs2, x .- heater_center) < heater_radius^2 ? heater_power : 0.0
end

# ── The discretisation, rebuilt per active set ───────────────────────────────

# Base level over all of Ω plus one overlay, also over Ω, carrying `mask` as
# its per-cell activation. Returns the prepared model, its temperature field
# and the mass matrix, which is wanted both by the time step and by the L²
# transfer of the temperature.
function build(mask)
    V = overlay(space(omega; cells=base_cells, order=base_order), omega; cells=front_cells,
                order=front_order, active=mask)
    u = field(:u, V)
    model = prepare(Problem((u,); dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    return model, u, assemble_matrix(model, mass_block(u))
end

# The overlay's cell grid is uniform over the unit square, so the cell holding
# a point is one floor per axis. This is what lifts the per-quadrature-point
# state up to the per-cell granularity that activation works at.
cell_of(x) = CartesianIndex(clamp.(1 .+ floor.(Int, Tuple(x) .* front_cells), 1, front_cells))

# Cells holding a partially cured quadrature point, grown by one cell in every
# direction. The growth is the safety margin: the front advances within a step,
# and a cell refined only once the front has crossed it is refined too late.
function front_mask(chi, model)
    partial = falses(front_cells)
    foreach_quadrature_point(model) do q
        first(cure_band) < chi[q.point] < last(cure_band) && (partial[cell_of(q.x)] = true)
    end
    cells, halo = CartesianIndices(partial), oneunit(CartesianIndex(front_cells))
    return [any(@view partial[max(first(cells), c-halo):min(last(cells), c+halo)]) for c in cells]
end

# ── The two state updates ────────────────────────────────────────────────────

# The conduction form ∫_Ω κ(χ) ∇v ⋅ ∇u dx. This is why χ is a `QuadField` and
# not a coefficient callback `κ(x)`: the value is attached to the quadrature
# point through `q.point`, not to the physical position.
conduction(chi) = WeakForm(; linear=q -> 0.0, symmetric=true,
                           bilinear=(q, trial) -> TestChannels(0.0,
                                                               kappa(chi[q.point]) * trial.gradient))

# `chi_old` advanced by one cure step against a temperature iterate.
function cured(chi_old, state, model)
    chi = copy(chi_old)
    foreach_quadrature_point(model; state=state) do q
        chi[q.point] = advance(chi_old[q.point], value(q.state, :u))
    end
    return chi
end

# One implicit-Euler step, solved by fixed-point iteration on the pair (u, χ):
# freeze χ, solve (M/Δt + K(χ)) u = M uⁿ/Δt + f, re-cure, repeat. The reported
# residual is the relative change in the temperature coefficients, which is the
# quantity the iteration contracts.
function picard(model, u, mass, chi_old, previous, t)
    load = mass * previous / time_step + assemble_vector(model, source_load(u; source=heater(t)))
    coefficients, chi, residuals = previous, chi_old, Float64[]
    for _ in 1:picard_max
        next = (mass / time_step + assemble_matrix(model, block(u, u, conduction(chi)))) \ load
        push!(residuals, norm(next - coefficients) / max(norm(next), eps()))
        coefficients = next
        chi = cured(chi_old, solution(model, coefficients; method=:picard), model)
        last(residuals) < picard_tol && break
    end
    return coefficients, chi, residuals
end

# ── The transient ────────────────────────────────────────────────────────────

# Three runs share this function. `carry_state = false` is the control that
# throws χ away at each rebuild, and shows the transfer *matters*. `adapt =
# false` holds the overlay permanently active, so the discretisation never
# changes, no transfer ever happens, and the answer owes nothing to `RBFP0` —
# that is the reference the carried run has to reproduce, and it is what shows
# the transfer is *right*. Proving a feature matters and proving it is correct
# are different claims and need different controls.
function simulate(; carry_state::Bool, adapt::Bool=true)
    mask = adapt ? falses(front_cells) : trues(front_cells)
    model, u, mass = build(mask)
    chi = QuadField{Float64}(model)
    coefficients = zeros(size(mass, 1))
    iterations, history, point_counts, excursion = Int[], Float64[], Tuple{Int,Int}[], 0.0

    for step in 1:time_steps
        coefficients, chi, residuals = picard(model, u, mass, chi, coefficients, step * time_step)
        push!(iterations, length(residuals))
        step == 1 && (history = residuals)

        # Refinement follows the state: re-derive the active set from χ, and
        # rebuild only when it has actually moved.
        target_mask = adapt ? front_mask(chi, model) : mask
        target_mask == mask && continue
        mask = target_mask
        target, u_target, target_mass = build(mask)
        push!(point_counts, (nquadpoints(model), nquadpoints(target)))

        # Both states cross to the new model while the old one is still valid:
        # the temperature by an L² projection, the material state — or not — by
        # the RBF scheme.
        moved_state = transfer(solution(model, coefficients; method=:implicit_euler), model, target;
                               via=L2Projection(target_mass))
        chi = carry_state ? transfer(chi, model, target; via=RBFP0()) : QuadField{Float64}(target)
        excursion = max(excursion, maximum(max(-chi[i], chi[i] - 1) for i in 1:length(chi)))
        coefficients, model, u, mass = moved_state.coefficients, target, u_target, target_mass
    end

    # ∫_Ω χ dx, the reach of the front and the peak temperature, all read
    # straight off the final quadrature-point cloud.
    final = solution(model, coefficients; method=:picard)
    area, radius, peak = 0.0, 0.0, 0.0
    foreach_quadrature_point(model; state=final) do q
        area += q.weight * chi[q.point]
        peak = max(peak, value(q.state, :u))
        chi[q.point] > 0.5 && (radius = max(radius, norm(q.x .- heater_center)))
    end
    return (; model, final, area, radius, peak, iterations, history, point_counts, excursion)
end

carried = simulate(; carry_state=true)
dropped = simulate(; carry_state=false)
reference = simulate(; carry_state=true, adapt=false)

# ── Report ───────────────────────────────────────────────────────────────────

println("Thermal curing of a thermoset — the cure fraction rides along with the mesh")
println("  Picard iterations per step    : ", carried.iterations)
println("  residual history, first step  : ", round.(carried.history; sigdigits=3))
println("  quadrature points per rebuild : ",
        join(("$before→$after" for (before, after) in carried.point_counts), ", "),
        all(pair -> first(pair) != last(pair), carried.point_counts) ? "  (all changed)" :
        "  (SOME UNCHANGED — the transfer would be a no-op there)")
println("  largest χ excursion outside [0, 1] introduced by the RBF transfer: ",
        round(carried.excursion; sigdigits=3))
println()
println("  ", rpad("run", 30), rpad("rebuilds", 10), rpad("cured area", 14),
        rpad("front radius", 14), "peak temperature (cure onset ", cure_temperature, ")")
for (label, run) in (("χ transferred (RBFP0)", carried), ("χ dropped at every rebuild", dropped),
                     ("reference, no remeshing", reference))
    println("  ", rpad(label, 30), rpad(length(run.point_counts), 10),
            rpad(round(run.area; sigdigits=6), 14), rpad(round(run.radius; sigdigits=6), 14),
            round(run.peak; sigdigits=4))
end
println("  carried vs reference (the accuracy claim): ",
        round(100 * abs(carried.area - reference.area) / reference.area; sigdigits=3), " %")
println("  cured area lost without the transfer: ",
        round(100 * (1 - dropped.area / carried.area); sigdigits=4), " %")
println()

out = joinpath(@__DIR__, "output", "thermal_curing_2d")
write_vtk(out, carried.final, carried.model; point_data=(temperature=(uh, c, x, xi) -> uh(c, xi),))
print_run_report("Thermal curing — final state at t = $(round(time_steps * time_step; digits=6))",
                 diagnostics(carried.model, carried.final); output=out,
                 parameters=(:base => (cells=base_cells, order=base_order),
                             :front_overlay => (cells=front_cells, order=front_order),
                             :conductivity => (kappa_uncured, kappa_cured),
                             :cure => (cure_temperature, cure_width, cure_rate),
                             :time => (time_step, time_steps, pulse_end),
                             :cured_area => (transferred=carried.area, dropped=dropped.area)))
