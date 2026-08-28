#=
Traveling Heat Source On A Moving Overlay
=========================================

The moving-feature experiment of §5.4 of

    J. N. Schmäke and M. Ruess, "Unfitted Multi-Level hp Refinement for
    Localized and Moving Solution Features", arXiv:2604.25797.
    https://arxiv.org/abs/2604.25797

A compact heat source travels once around a circular path inside a square
plate and the discretisation follows it. The transient problem is

    ∂ₜT − κ ΔT = Q(x, t)      in Ω = (−5, 5)²,
             T = 0            on ∂Ω,
             T = 0            at t = 0,

with conductivity κ = 1 and Q a disk of radius 0.1 and intensity 10 whose
centre runs around the circle of radius 2.5 at constant angular velocity,
completing one revolution in `revolution_time`.

Three levels carry the solution, and each one exercises a different part of
the moving-overlay workflow:

  1. a **base** level over all of Ω. It is deliberately coarse; its job is to
     make the superposition u_h = Σₖ u_h^(k) complete wherever the overlays
     happen not to be.
  2. a **wake** overlay, also spanning Ω, but with only the cells around the
     source and its cooling trail switched on. Cells are switched with
     `activate!` and `deactivate!`, so the refined region follows the feature
     while the level's geometry never changes.
  3. a **tip** overlay: a small box that is *translated* onto the source at
     every step with `moved`, the non-destructive form of `move!`.

Because both the geometry (the tip box) and the active set (the wake mask)
change at every step, the space of step n is not the space of step n−1 and a
coefficient vector cannot simply be carried over. The state is moved by an
L² projection onto the new space,

    find Tₙ ∈ Vₙ  such that  ∫_Ω Tₙ v dx = ∫_Ω Tₙ₋₁ v dx   ∀ v ∈ Vₙ,

which is the variational transfer the method requires; pointwise
interpolation between two superposition spaces is not admissible, because a
point value of u_h is a sum of overlay contributions that the target space
generally cannot reproduce mode for mode.

**The metric.** Each transfer solves a sparse mass system `M c = b`, so the
script prints one residual ‖M c − b‖₂ per step under `projection_residuals`.
A direct solve of a well-conditioned mass matrix leaves machine noise, so
every entry should read 0.0 or thereabouts; an entry that climbs means a
transfer failed and nothing computed after it is the solution of anything.
The script prints no error against an exact solution — there is none for
this problem — so the transfer residuals, together with a clean run to the
final time, are what tell the reader the run succeeded.

**How this differs from the paper.** The workflow is the paper's; the
configuration is smaller, and the numbers below should not be read as the
published ones. §5.4 grades several static overlay levels over the
trajectory with per-level refinement indicators, and drives the transient on
a finer time grid than the mesh-update grid. This script keeps one wake
level and one tip level, and takes exactly one implicit-Euler step per
mesh update, because the point being made here is the moving overlay and the
state transfer, not the indicator hierarchy. Nothing about the transfer
changes with those choices — it is the same L² projection between the same
two kinds of space.

Two optional size knobs, both read once at load time: `THS_T_MAX` shortens
the transient without changing the source's speed (the default is one full
revolution; the smoke test uses 0.05, which is a couple of steps), and
`THS_WRITE_OUTPUT=false` suppresses the final ParaView bundle.
=#

using Unfitted
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_bool(name, default) = lowercase(get(ENV, name, string(default))) in ("1", "true", "yes")

# ── Discretisation ────────────────────────────────────────────────────────────

omega = box((-5.0, -5.0), (5.0, 5.0))
kappa = 1.0
base_cells = (4, 4)
base_order = 4

# The wake overlay spans Ω on a fine grid but keeps only a handful of cells
# active. `wake_radius` is the distance from the source centre inside which
# cells are switched on; it covers the source disk plus roughly two cells of
# halo, comfortably more than the √(2 κ Δt) ≈ 0.22 that diffusion can travel in
# one step, so the leading edge of the hot spot never runs off the refined
# region. `wake_cool_fraction` is the fraction of the current peak temperature
# below which a cell behind the source is switched off again — that is what
# makes the refined region a trailing wake rather than a growing annulus.
const wake_level = 2
wake_cells = (15, 15)
wake_order = 4
wake_radius = 1.5
wake_cool_fraction = 0.1

# The tip overlay is the moving one: a small box centred on the source, fine
# enough that the source disk spans several cells (0.7 / 12 ≈ 0.058 against a
# disk radius of 0.1) at a modest polynomial order.
const tip_level = 3
tip_halfwidth = 0.35
tip_cells = (12, 12)
tip_order = 2

# ── Source and time integration ───────────────────────────────────────────────

# The angular velocity is fixed by `revolution_time`, not by `t_max`, so
# shortening the run truncates the transient instead of speeding the source up.
revolution_time = 4.0
t_max = env_float("THS_T_MAX", revolution_time)
time_step = 1.0 / 40.0
write_output = env_bool("THS_WRITE_OUTPUT", true)

source_path_radius = 2.5
source_radius = 0.1
intensity = 10.0

function source_center(t)
    angle = -pi / 2 + 2pi * t / revolution_time
    return (source_path_radius * cos(angle), source_path_radius * sin(angle))
end

# Q is a sharp indicator, so the heat actually delivered to the discrete
# system is whatever the tensor Gauss rules of the regions under the disk see.
# Resolving that disk is the tip overlay's whole job; on the base level alone
# the delivered power would depend visibly on where the source happens to sit
# relative to the cell boundaries.
heat_source(c) = x -> sum(abs2, x .- c) <= source_radius^2 ? intensity : 0.0

# Two axis-aligned boxes overlap exactly when they overlap on every axis. The
# source is a small disk, so its load integral vanishes identically on nearly
# every integration region; `region_filter` skips those regions instead of
# evaluating a zero integrand over each of their quadrature points.
boxes_overlap(a, b) = all(a.lower .<= b.upper) && all(b.lower .<= a.upper)

function source_vector(model, temperature, t)
    c = source_center(t)
    support = box(c; halfwidth=source_radius)
    return assemble_vector(model, source_load(temperature; source=heat_source(c));
                           region_filter=region -> boxes_overlap(region.box, support))
end

# ── Wake-overlay cell selection ───────────────────────────────────────────────

# The wake overlay's own cell grid. It is the grid `overlay(...; cells=…)`
# builds over the same box, so a cell index here is the cell index the
# activation mutators expect, and the centres never move.
const wake_centers = let wake_mesh = mesh(omega; cells=wake_cells)
    [center(cell_box(wake_mesh, ci)) for ci in cell_indices(wake_mesh)]
end

# Cells the source occupies or is about to occupy.
near_source(c) = [norm(x .- c) <= wake_radius for x in wake_centers]

# Cells that have cooled below `wake_cool_fraction` of the current peak.
# Sampling at the cell centre is enough: the outcome is a refinement decision,
# not a reported quantity. At t = 0 the field is identically zero, the peak is
# zero, and the strict comparison selects nothing — so the initial wake is left
# alone rather than switched off wholesale.
function cooled_cells(state, model)
    temperatures = [value(state, model, x) for x in wake_centers]
    return temperatures .< wake_cool_fraction * maximum(temperatures)
end

# ── Transient ─────────────────────────────────────────────────────────────────

function main()
    start_center = source_center(0.0)

    V = space(omega; cells=base_cells, order=base_order, mode=:trunk)
    V = overlay(V, omega; cells=wake_cells, order=wake_order, active=near_source(start_center),
                mode=:trunk)
    V = overlay(V, box(start_center; halfwidth=tip_halfwidth); cells=tip_cells, order=tip_order,
                mode=:trunk)

    temperature = field(:temperature, V)
    model = prepare(Problem((temperature,);
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=temperature)]))
    mass = assemble_matrix(model, mass_block(temperature))
    stiffness = assemble_matrix(model, stiffness_block(temperature; diffusion=kappa))
    coefficients = zeros(size(mass, 1))

    steps = max(1, round(Int, t_max / time_step))
    residuals = Float64[]

    for step in 1:steps
        t = step * time_step
        center_now = source_center(t)
        previous = solution(model, coefficients; method=:implicit_euler)

        # Translate the tip overlay onto the new source position. `moved` is
        # the non-destructive sibling of `move!`: it returns a fresh model and
        # leaves this one valid, which is exactly what lets `previous` still be
        # projected off it below.
        target = moved(model; level=tip_level, to=box(center_now; halfwidth=tip_halfwidth))

        # Follow the source on the wake overlay. A cell the source is entering
        # is never switched off, so the two selections are disjoint: the
        # `deactivate!` cannot undo the `activate!`, and `active` — read once,
        # before either — still answers correctly for the second test. Both
        # mutators rebuild the dof layout, so a call that would flip no cell at
        # all is skipped rather than paid for.
        entering = near_source(center_now)
        cooling = cooled_cells(previous, model) .& .!entering
        active = active_cells(target; level=wake_level)
        any(cooling .& active) && deactivate!(target; level=wake_level, cells=cooling)
        any(entering .& .!active) && activate!(target; level=wake_level, cells=entering)

        # Re-assemble on the new configuration, then carry the state across.
        # Handing the freshly assembled mass to `L2Projection` reuses it rather
        # than letting the backend assemble a second copy of the same matrix.
        mass = assemble_matrix(target, mass_block(temperature))
        stiffness = assemble_matrix(target, stiffness_block(temperature; diffusion=kappa))
        projected = transfer(previous, model, target; via=L2Projection(mass))
        push!(residuals, projected.diagnostics.residual_norm)
        model = target

        # One implicit-Euler step on the new space:
        # (M/Δt + K) Tₙ = M Tₙ₋₁ / Δt + f(tₙ).
        rhs = mass * projected.coefficients / time_step + source_vector(model, temperature, t)
        coefficients = (mass / time_step + stiffness) \ rhs
    end

    # ── Report ────────────────────────────────────────────────────────────────

    t_end = steps * time_step
    final = solution(model, coefficients; method=:implicit_euler)
    report = diagnostics(model, final)

    out = joinpath(@__DIR__, "output", "traveling_heat_source_2d")
    write_output && write_vtk(out, final, model; subdivisions=4,
                              point_data=(temperature=(u, c, x, xi) -> u(c, xi),
                                          source=(u, c, x, xi) -> heat_source(source_center(t_end))(x)))

    wake_active = count(active_cells(model; level=wake_level))
    print_run_report("Traveling heat source on a moving overlay", report;
                     parameters=(:domain => omega, :conductivity => kappa,
                                 :base => (cells=base_cells, order=base_order),
                                 :wake => (cells=wake_cells, order=wake_order, radius=wake_radius,
                                           cool_fraction=wake_cool_fraction),
                                 :tip =>
                                     (halfwidth=tip_halfwidth, cells=tip_cells, order=tip_order),
                                 :revolution_time => revolution_time,
                                 :time_interval => (0.0, t_end), :time_step => time_step,
                                 :time_steps => steps, :final_source_center => source_center(t_end),
                                 :wake_active_cells => "$(wake_active) / $(prod(wake_cells))",
                                 :transfer_residual_max => maximum(residuals),
                                 :projection_residuals => residuals),
                     output=write_output ? out : nothing)
end

main()
