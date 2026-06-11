#=
Traveling Heat Source With Hierarchical Adaptive Overlays
=========================================================

The traveling-heat-source experiment of §5.4 of the UMLHP preprint
(arXiv:2604.25797): the transient heat equation

    ∂ₜ T − κ Δ T = Q(x, t)       in Ω,
                T = 0            on ∂Ω,

driven by a compact heat source moving on a circular path. This
script demonstrates the moving-overlay / mesh-update workflow of the
unfitted multi-level hp framework end-to-end, plus the *adaptive*
extension that composes both of the package's overlay mechanisms in
a single model:

  1. **N static "trajectory" overlays** — a tuple of overlay levels
     that all share the same axis-aligned box (the trajectory box,
     covering the source's circular path). Each entry of
     `static_overlay_specs` carries its own cell count, polynomial
     order, indicator thresholds (`alpha_T`, `alpha_g`), source-disk
     safety radius, and dilation halo. Outer (coarser) entries use
     looser thresholds; inner (finer) entries use tighter thresholds
     so each finer level activates a strict subset of the cells the
     coarser one keeps. The script is generic in the number of
     static levels: change the tuple, the rest follows.

  2. **One moving "tip" overlay** for peak resolution at the source.
     The square overlay box translates with the source via the
     standard `move!`-style rebuild, and the active region is a
     *disk* — the square box carries a precomputed circular cell
     mask, so the effective refinement region is translation-
     invariantly circular.

At every `mesh_update_frequency` interval the per-level masks are
recomputed from the current solution, the new state is built
(per-level static masks + new tip position + the same circular tip
mask), and the prior solution is transferred forward by L²
projection using a cached target mass matrix and factorisation.

The semidiscrete system on a fixed configuration is

    M Ṫ + K T = f(t),

stepped by a θ-method (implicit Euler at `theta = 1`). The
cell-activation feedback loop is the part this example exists to
demonstrate; the time-stepping and VTK snapshot machinery are
unchanged from the original moving-overlay version in the paper.
=#

using Unfitted
using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "..", "reporting.jl"))

# ---------------------------------------------------------------------
# Domain and base discretization
L = 10.0
N = 4
p = 4
omega = box((-0.5L, -0.5L), (0.5L, 0.5L))
kappa = 1.0

# ---------------------------------------------------------------------
# Trajectory overlays — shared static box, N hierarchical resolution
# levels. Each entry is independent: outer (looser) and inner (tighter)
# levels can co-exist anywhere their indicators fire. Add or remove
# entries to change the number of static overlay levels.
trajectory_box = box((-5.0, -5.0), (5.0, 5.0))

static_overlay_specs = ((cells=(15, 15), order=4, alpha_T=0.10, alpha_g=0.05,
                         source_active_radius=1.2, dilation=2),
                        (cells=(40, 40), order=3, alpha_T=0.25, alpha_g=0.10,
                         source_active_radius=0.7, dilation=2))

# ---------------------------------------------------------------------
# Tip overlay — small square box that moves with the source. The
# *active mask* is a disk of radius `tip_radius`, so the effective
# refinement region is circular while the underlying overlay geometry
# stays axis-aligned (the only kind the package supports).
tip_box_halfwidth = 0.35
tip_radius = 0.30        # disk radius of the active mask
tip_cells = (24, 24)
tip_order = 2
tip_dilation = 1

# ---------------------------------------------------------------------
# Time-stepping and IO
t_max = 4.0
max_time_step = 1.0 / 30.0
mesh_update_frequency = 1.0 / 40.0
export_frequency = 1.0 / 30.0
theta = 1.0
visualization_subdivisions = 4
write_transient_snapshots = true
time_tolerance = 1.0e-12

# ---------------------------------------------------------------------
# Source path and intensity
phi_0 = -pi / 2
phi_dot = 2pi / t_max
source_path_radius = 2.5
source_radius = 0.1
intensity = 10.0

function source_center(t)
    angle = phi_0 + t * phi_dot
    return (source_path_radius * cos(angle), source_path_radius * sin(angle))
end

function heat_source(center)
    return x -> ((x[1] - center[1])^2 + (x[2] - center[2])^2 <= source_radius^2) ? intensity : 0.0
end

heat_source_box(center) = box(center; halfwidth=source_radius)

# ---------------------------------------------------------------------
# Mask construction
#
# `dilate_mask(mask, n)` grows the active region by `n` cells in the
# Chebyshev metric — gives diffusion a halo to spread into during the
# next interval so the leading edge doesn't fall off the active set.
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

# ---------------------------------------------------------------------
# Per-static-overlay caches
#
# The static overlays all share `trajectory_box` but have different cell
# counts; cache one CartesianMesh per spec so the indicator + bootstrap
# passes don't rebuild them on every mesh-update interval.
const static_overlay_meshes = Tuple(mesh(trajectory_box; cells=s.cells)
                                    for s in static_overlay_specs)

# Distance from a static-overlay cell's centre to the source.
function _cell_source_distance(mesh_, ci::CartesianIndex{2}, src_center)
    c = center(cell_box(mesh_, ci))
    return sqrt((c[1] - src_center[1])^2 + (c[2] - src_center[2])^2)
end

# Initial mask for ONE static overlay used at t = 0 before any solution
# exists: just the spec's source-disk minimum plus its dilation halo.
function bootstrap_mask(spec, mesh_, src_center)
    mask = falses(mesh_.cells)
    for ci in cell_indices(mesh_)
        if _cell_source_distance(mesh_, ci, src_center) <= spec.source_active_radius
            mask[ci] = true
        end
    end
    return dilate_mask(mask, spec.dilation)
end

function bootstrap_masks(src_center)
    [bootstrap_mask(s, m, src_center)
     for (s, m) in zip(static_overlay_specs, static_overlay_meshes)]
end

# Solution-driven mask for ONE static overlay: indicator (T or ‖∇T‖
# above a per-level fraction of the running max) OR per-level source-
# disk safety net, then per-level dilation halo. The sampling is at
# the cell centre — one `value` + one `gradient` evaluation per cell of
# the per-level static mesh, via the public superposition-aware
# accessors.
function indicator_mask(spec, mesh_, sol, model, src_center)
    Ts = Array{Float64}(undef, mesh_.cells)
    Gs = Array{Float64}(undef, mesh_.cells)
    for ci in cell_indices(mesh_)
        c = center(cell_box(mesh_, ci))
        Ts[ci] = value(sol, model, c)
        Gs[ci] = norm(gradient(sol, model, c))
    end

    T_max = maximum(Ts)
    G_max = maximum(Gs)
    tau_T = T_max > 0 ? spec.alpha_T * T_max : 0.0
    tau_g = G_max > 0 ? spec.alpha_g * G_max : 0.0

    mask = falses(mesh_.cells)
    for ci in cell_indices(mesh_)
        if Ts[ci] > tau_T ||
           Gs[ci] > tau_g ||
           _cell_source_distance(mesh_, ci, src_center) <= spec.source_active_radius
            mask[ci] = true
        end
    end
    return dilate_mask(mask, spec.dilation)
end

function indicator_masks(sol, model, src_center)
    [indicator_mask(s, m, sol, model, src_center)
     for (s, m) in zip(static_overlay_specs, static_overlay_meshes)]
end

# ---------------------------------------------------------------------
# Tip overlay disk mask (translation-invariant)
#
# The tip overlay's square box moves with the source, but the local cell
# grid inside the box has the same coordinates relative to its centre
# every time. So the disk-shaped active mask is the same BitArray on
# every move — compute it once.
const tip_disk_mask = let
    tip_box_template = box((0.0, 0.0); halfwidth=tip_box_halfwidth)
    tip_mesh_template = mesh(tip_box_template; cells=tip_cells)
    mask = falses(tip_cells)
    for ci in cell_indices(tip_mesh_template)
        c = center(cell_box(tip_mesh_template, ci))
        sqrt(c[1]^2 + c[2]^2) <= tip_radius && (mask[ci] = true)
    end
    dilate_mask(mask, tip_dilation)
end

# ---------------------------------------------------------------------
# Space + model construction
#
# `build_space(center, static_masks)` composes the hierarchy:
# (1) the base mesh,
# (2) N static overlays sharing `trajectory_box`, each carrying its mask,
# (3) the moving tip overlay with its precomputed circular mask.
function build_space(center, static_masks)
    V = space(omega; cells=(N, N), order=p, mode=:total_degree)
    for (spec, mask) in zip(static_overlay_specs, static_masks)
        V = overlay(V, trajectory_box; cells=spec.cells, order=spec.order, active=mask,
                    mode=:total_degree)
    end
    V = overlay(V, box(center; halfwidth=tip_box_halfwidth); cells=tip_cells, order=tip_order,
                active=tip_disk_mask, mode=:total_degree)
    return V
end

function build_heat_state(center, static_masks)
    V = build_space(center, static_masks)
    temperature = field(:temperature, V)
    model = prepare(Problem((temperature,);
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=temperature)]))

    mass_matrix = assemble_matrix(model, mass_block(temperature))
    stiffness_matrix = assemble_matrix(model, stiffness_block(temperature; diffusion=kappa))
    return (; model, field=temperature, mass=mass_matrix, stiffness=stiffness_matrix,
            mass_factor=factorize(mass_matrix), lhs_factors=Dict{Float64,Any}())
end

# ---------------------------------------------------------------------
# Time integration
function load_vector_at(state, t)
    center = source_center(t)
    support = heat_source_box(center)
    return assemble_vector(state.model, source_load(state.field; source=heat_source(center));
                           region_filter=region -> Unfitted.box_intersection(region.box,
                                                                             support) !== nothing)
end

function lhs_factor(state, dt)
    key = round(dt; digits=14)
    return get!(state.lhs_factors, key) do
        factorize((1 / dt) * state.mass + theta * state.stiffness)
    end
end

function theta_step(state, coefficients, t0, t1)
    dt = t1 - t0
    rhs = load_vector_at(state, t1)
    rhs .*= theta

    mass_term = similar(coefficients)
    mul!(mass_term, state.mass, coefficients)
    rhs .+= mass_term ./ dt

    if theta != 1.0
        stiffness_term = similar(coefficients)
        mul!(stiffness_term, state.stiffness, coefficients)
        rhs .-= (1 - theta) .* stiffness_term
        rhs .+= (1 - theta) .* load_vector_at(state, t0)
    end

    return lhs_factor(state, dt) \ rhs
end

# ---------------------------------------------------------------------
# Time grids and IO
function mesh_update_times()
    update_count = round(Int, t_max / mesh_update_frequency)
    times = [i * mesh_update_frequency for i in 0:update_count]
    times[end] = t_max
    return times
end

function export_times()
    export_count = round(Int, t_max / export_frequency)
    times = [i * export_frequency for i in 0:export_count]
    times[end] = t_max
    return times
end

function _push_time!(times, t)
    any(s -> isapprox(s, t; atol=time_tolerance, rtol=0.0), times) || push!(times, t)
    return times
end

function interval_step_times(t0, t1, snapshot_times)
    times = Float64[]
    for t in snapshot_times
        t > t0 + time_tolerance && t <= t1 + time_tolerance && _push_time!(times, min(t, t1))
    end

    t = t0 + max_time_step
    while t < t1 - time_tolerance
        _push_time!(times, t)
        t += max_time_step
    end

    _push_time!(times, t1)
    sort!(times)
    return times
end

function write_snapshot(output_root, index, state, coefficients, t)
    snapshot = solution(state.model, coefficients; method=:theta)
    center = source_center(t)
    path = joinpath(output_root, "solution_$(lpad(string(index), 4, '0'))")
    write_vtk(path, snapshot, state.model; subdivisions=visualization_subdivisions,
              point_data=(temperature=(u, c, x, xi) -> u(c, xi),
                          temperature_gradient=(u, c, x, xi) -> gradient(snapshot, state.model, x),
                          source=(u, c, x, xi) -> heat_source(center)(x)),)
    return snapshot
end

# ---------------------------------------------------------------------
# Reporting helpers
#
# Active counts are tracked per static overlay level so the reader can
# see how each resolution layer's mask evolves independently.
sample_series(series, max_samples=20) = begin
    stride = max(1, length(series) ÷ max_samples)
    s = series[1:stride:end]
    last(series) == last(s) || push!(s, last(series))
    s
end

# ---------------------------------------------------------------------
# Main loop
function main()
    update_times = mesh_update_times()
    snapshot_times = write_transient_snapshots ? export_times() : [t_max]
    nstatic = length(static_overlay_specs)
    output_root = joinpath(@__DIR__, "output",
                           "traveling_heat_source_2d_adaptive_$(nstatic)static_p$(p)")

    initial_center = source_center(first(update_times))
    initial_masks = bootstrap_masks(initial_center)
    state = build_heat_state(initial_center, initial_masks)
    coefficients = zeros(size(state.stiffness, 1))

    projection_residuals = Float64[]
    # active_counts_per_level[i][k] = active cells of static level i after step k.
    active_counts_per_level = [Int[count(m)] for m in initial_masks]
    tip_active_count = count(tip_disk_mask)
    time_steps = 0
    snapshot_index = 0

    if write_transient_snapshots && !isempty(snapshot_times) && iszero(first(snapshot_times))
        write_snapshot(output_root, snapshot_index, state, coefficients, first(snapshot_times))
        snapshot_index += 1
    end

    for i in 1:(length(update_times)-1)
        t0 = update_times[i]
        t1 = update_times[i + 1]
        for t in interval_step_times(t0, t1, snapshot_times)
            coefficients = theta_step(state, coefficients, t0, t)
            time_steps += 1
            if any(s -> isapprox(s, t; atol=time_tolerance, rtol=0.0), snapshot_times)
                write_snapshot(output_root, snapshot_index, state, coefficients, t)
                snapshot_index += 1
            end
            t0 = t
        end

        if i < length(update_times) - 1
            # Mesh update: rebuild the state at the new source position with a
            # solution-driven mask **per static overlay level**, then
            # variationally transfer the prior state forward. We use the
            # *build-from-scratch* pattern (build_heat_state + transfer!)
            # rather than the in-place `activate!`/`deactivate!` mutators:
            # every mesh update also moves the tip overlay and forces a fresh
            # `prepare`/mass + stiffness assembly + factorization, so the
            # mutators wouldn't save any work on this hot path.
            old_model = state.model
            old_solution = solution(old_model, coefficients; method=:theta)
            new_center = source_center(t1)
            new_masks = indicator_masks(old_solution, old_model, new_center)
            state = build_heat_state(new_center, new_masks)
            projected = transfer!(old_solution, old_model, state.model;
                                  backend=L2Projection(state.mass; factor=state.mass_factor))
            coefficients = copy(projected.coefficients)
            push!(projection_residuals, projected.diagnostics.residual_norm)
            for (i_level, mask) in enumerate(new_masks)
                push!(active_counts_per_level[i_level], count(mask))
            end
        end
    end

    final_solution = solution(state.model, coefficients; method=:theta)
    report = diagnostics(state.model, final_solution)
    center = source_center(last(update_times))

    # Per-level summaries: range across the run, sampled trajectory, final
    # active count out of total cells per level.
    level_total = [prod(s.cells) for s in static_overlay_specs]
    level_range = [(minimum(c), maximum(c)) for c in active_counts_per_level]
    level_sampled = [sample_series(c) for c in active_counts_per_level]
    level_final = [count(active_cells(state.model; level=1 + i)) for i in 1:nstatic]

    print_run_report("Traveling heat-source with hierarchical adaptive overlays", report;
                     parameters=(:domain => omega, :base_cells => (N, N), :base_order => p,
                                 :trajectory_box => trajectory_box,
                                 :static_overlay_count => nstatic,
                                 :static_overlay_specs => static_overlay_specs,
                                 :tip_box_halfwidth => tip_box_halfwidth, :tip_radius => tip_radius,
                                 :tip_cells => tip_cells, :tip_order => tip_order,
                                 :tip_active_cells => "$(tip_active_count) / $(prod(tip_cells))",
                                 :time_interval => (first(update_times), last(update_times)),
                                 :mesh_update_frequency => mesh_update_frequency,
                                 :mesh_update_intervals => length(update_times) - 1,
                                 :time_integrator => :theta_method, :theta => theta,
                                 :max_time_step => max_time_step, :time_steps => time_steps,
                                 :visualization_subdivisions => visualization_subdivisions,
                                 :export_frequency => export_frequency,
                                 :write_transient_snapshots => write_transient_snapshots,
                                 :exported_snapshots => snapshot_index, :output_root => output_root,
                                 :projection_residuals => projection_residuals,
                                 :static_level_active_ranges =>
                                     ["level $i: $(r[1])..$(r[2]) of $(level_total[i])"
                                      for (i, r) in enumerate(level_range)],
                                 :static_level_active_sampled =>
                                     ["level $i: $(s)" for (i, s) in enumerate(level_sampled)],
                                 :static_level_active_final =>
                                     ["level $i: $(level_final[i]) / $(level_total[i])"
                                      for i in 1:nstatic], :final_source_center => center))
end

main()
