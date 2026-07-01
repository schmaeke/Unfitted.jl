# Solution transfer between [`Model`](@ref)s. Two backends are shipped:
#
#   * [`L2Projection`](@ref) — variational L² projection, the default.
#     Solves the target-side L² problem
#
#         find u_T ∈ V_T such that ∫_Ω u_T v dx = ∫_Ω u_S v dx ∀ v ∈ V_T,
#
#     using a partition of `Ω` that respects both the source and the
#     target meshes (so basis-function products are smooth on every
#     integration region). This produces the orthogonal projection of
#     the source field onto the target's active space, which is the
#     "best" transfer in the L² sense and the correct variational move
#     when the target's basis cannot represent the source exactly.
#   * [`Rewire`](@ref) — direct raw-key coefficient matching. Each
#     target dof whose `TensorDofKey` matches an active source dof
#     inherits the source coefficient; genuinely new dofs and freshly-
#     uncovered constraints start at zero. Lossless when the target's
#     active basis contains the source's — typically after a monotone
#     `activate!` — and avoids the projection cost / smoothing in that
#     case.

# ── Transfer regions ──────────────────────────────────────────────────────────

"""
    TransferRegion{D,T}(box, target_parents, source_parents, quadrature)

One admissible integration region for a source-to-target L²
projection. Mirrors [`VolumeRegion`](@ref) but carries *two*
parent lists — one per side of the projection — so the assembly hot
loop can evaluate the source field and the target trace
simultaneously without re-walking the mesh.

Fields:

  - `box::AxisBox{D,T}` — the region's physical-frame extent. Built
    from the *union* of source and target mesh boundaries so the
    integrand is smooth across the box.
  - `target_parents` — active target levels whose cell contains the
    box midpoint, with the box in each parent's reference frame.
  - `source_parents` — same, for the source. May be empty when the box
    lies in part of `Ω` the source did not cover (e.g. after an
    overlay activation extended the target).
  - `quadrature::TensorQuadrature{D,T}` — shared cached tensor Gauss
    rule, sized by the per-axis maximum recommended quadrature order
    across both sides.
"""
struct TransferRegion{D,T<:Real}
    box::AxisBox{D,T}
    target_parents::Vector{ParentRef{D,T}}
    source_parents::Vector{ParentRef{D,T}}
    quadrature::TensorQuadrature{D,T}
end

# Sanity-check the source / target compatibility for a transfer:
# matching physical domain, matching field count, matching per-field
# component counts. Lets backends fail early with a clear message
# instead of producing wrong numbers downstream.
function _assert_transfer_compatible(source_model::Model{D,T}, target_model::Model{D,T}) where {D,T}
    source_model.problem.space.domain == target_model.problem.space.domain ||
        throw(ArgumentError("source and target models must have the same physical domain"))
    length(source_model.dofs.fields) == length(target_model.dofs.fields) ||
        throw(ArgumentError("source and target models must have the same fields"))
    for source_field in source_model.dofs.fields
        target_field = _field_layout(target_model.dofs, source_field.name)
        source_field.components == target_field.components ||
            throw(ArgumentError("source and target field $(source_field.name) have different component counts"))
    end
    return nothing
end

# Per-axis quadrature counts for a transfer region: take the per-axis
# maximum recommended order across the *union* of source and target
# parents, so the rule integrates source × target basis products exactly
# regardless of which side carries the higher-order basis.
#
# When `source_parents` is empty (no source coverage at this region),
# the source-side count drops to 1 — the rhs contribution there is zero
# anyway, so any rule will do, and the cheap `1` keeps cache pressure
# low.
function _transfer_quadrature_counts(source_model::Model{D}, target_model::Model{D}, target_parents,
                                     source_parents) where {D}
    target_counts = _parent_quadrature_counts(Val(D), target_parents,
                                              id -> _level_by_id(target_model.problem.space, id))
    source_counts = isempty(source_parents) ? ntuple(_ -> 1, D) :
                    _parent_quadrature_counts(Val(D), source_parents,
                                              id -> _level_by_id(source_model.problem.space, id))
    return ntuple(d -> max(target_counts[d], source_counts[d]), D)
end

function _transfer_quadrature(source_model::Model{D,T}, target_model::Model{D,T}, target_parents,
                              source_parents, cache) where {D,T}
    counts = _transfer_quadrature_counts(source_model, target_model, target_parents, source_parents)
    return _cached_tensor_quadrature!(cache, counts, T)
end

# Build the [`TransferRegion`](@ref) list. Reuses the volume
# admissible-box partition from `intersections.jl`: feed both sides'
# levels to `_merged_boxes` so the resulting boxes respect every mesh
# boundary on both sides. Drop boxes the target does not cover (no
# rhs contribution and no place to deposit the result).
function _transfer_regions(source_model::Model{D,T}, target_model::Model{D,T};
                           tolerance=target_model.dofs.tolerance) where {D,T}
    levels = (source_model.problem.space.levels..., target_model.problem.space.levels...)
    regions = TransferRegion{D,T}[]
    quadrature_cache = Dict{NTuple{D,Int},TensorQuadrature{D,T}}()

    for box in _merged_boxes(levels, Val(D), tolerance)
        target_parents = _parents_covering(target_model.problem.space.levels, box, tolerance)
        isempty(target_parents) && continue
        source_parents = _parents_covering(source_model.problem.space.levels, box, tolerance)
        quadrature = _transfer_quadrature(source_model, target_model, target_parents,
                                          source_parents, quadrature_cache)
        push!(regions, TransferRegion{D,T}(box, target_parents, source_parents, quadrature))
    end

    return regions
end

# ── Transfer workspace ───────────────────────────────────────────────────────

"""
    TransferWorkspace{D,T}

Per-region scratch for the L² projection source-driven rhs pass. Mirrors
[`AssemblyWorkspace`](@ref) but is value-only (no gradient banks):
projection integrals contract basis values against basis values, never
gradients.

Source and target levels live in independent id ranges so the workspace
carries one bank of per-level value buffers per side. Each region
update writes into the side's `values[parent.level]` buffer; since
`_parents_covering` returns at most one parent per level, the in-place
update never collides within a region.

The trailing `active_dofs` / `local_by_global` / `local_rhs` fields are
the per-region rhs scatter scratch — resized and reset in place every
region, never reallocated. The target mass matrix and its Dirichlet lift
are built by the standard assembler (see [`transfer`](@ref)), so this
workspace carries no local-matrix bank.
"""
struct TransferWorkspace{D,T}
    source_bases::Vector{BasisFamily}
    source_local_ids::Vector{Vector{CartesianIndex{D}}}
    source_orders::Vector{NTuple{D,Int}}
    source_values::Vector{Vector{T}}
    source_val1d::Vector{NTuple{D,Vector{T}}}
    target_bases::Vector{BasisFamily}
    target_local_ids::Vector{Vector{CartesianIndex{D}}}
    target_orders::Vector{NTuple{D,Int}}
    target_values::Vector{Vector{T}}
    target_val1d::Vector{NTuple{D,Vector{T}}}
    active_dofs::Vector{Int}
    local_by_global::Dict{Int,Int}
    local_rhs::Vector{T}
end

# One bank of per-level value buffers per side. The allocation is shared
# with the standard assembler through `_level_value_buffers` (defined in
# assembly.jl); the transfer's value-only integrals reuse exactly the
# `bases` / `local_ids` / `orders` / `values` / `val1d` banks and skip
# the assembler's gradient banks.
function _transfer_workspace(source_model::Model{D,T}, target_model::Model{D,T}) where {D,T}
    s_bs, s_ids, s_ord, s_val, s_v1 = _level_value_buffers(source_model.problem.space.levels,
                                                           Val(D), T)
    t_bs, t_ids, t_ord, t_val, t_v1 = _level_value_buffers(target_model.problem.space.levels,
                                                           Val(D), T)
    return TransferWorkspace{D,T}(s_bs, s_ids, s_ord, s_val, s_v1, t_bs, t_ids, t_ord, t_val, t_v1,
                                  Int[], Dict{Int,Int}(), T[])
end

# Slim per-parent record aliasing the workspace value buffer (no
# allocation; the basis values for the parent's level live in
# `values[parent.level]`). Same `NamedTuple` shape downstream consumers
# (`_field_value`, `_local_parent_dofs!`) already expect. Pass the side's
# `ws.target_values` or `ws.source_values` directly.
function _transfer_data(layout::FieldLayout, parent::ParentRef, values::Vector{Vector{T}}) where {T}
    lvl = parent.level
    return (; level=lvl, raw_dofs=cell_dofs(layout.dofs, lvl, parent.cell), values=values[lvl])
end

# Refresh the workspace value buffers at one region-reference point
# `eta`. Each parent's `values` buffer aliases the level slot
# (`ws.{source,target}_values[parent.level]`); since
# `_parents_covering` returns at most one parent per level on each
# side, the in-place update never collides within a region.
function _update_transfer_basis!(ws::TransferWorkspace{D,T}, region::TransferRegion{D,T},
                                 eta::SVector{D,T}) where {D,T}
    for p in region.target_parents
        xi = reference_to_physical(p.local_box, eta)
        _tensor_values!(ws.target_bases[p.level], ws.target_values[p.level],
                        ws.target_local_ids[p.level], ws.target_orders[p.level], xi,
                        ws.target_val1d[p.level], p.cell)
    end
    for p in region.source_parents
        xi = reference_to_physical(p.local_box, eta)
        _tensor_values!(ws.source_bases[p.level], ws.source_values[p.level],
                        ws.source_local_ids[p.level], ws.source_orders[p.level], xi,
                        ws.source_val1d[p.level], p.cell)
    end
    return nothing
end

# ── L² projection source-driven rhs ──────────────────────────────────────────

# Build the per-parent target dof-table for one (transfer region, field),
# resetting the workspace's `active_dofs` / `local_by_global`. The simple
# `Matrix{Int}` table is used for layouts without non-trivial linear
# constraints (the common case) and the `LocalDofExpansion` table
# otherwise; the matching `_emit_load!` overload (in `assembly.jl`) fires
# per quadrature point. Uses the same dof-table representations as the
# standard assembler, so the local→global scatter of the transfer rhs
# lands on exactly the active slots the standard mass assembly enumerates.
function _transfer_local_dofs!(ws::TransferWorkspace, target_layout, target_data)
    empty!(ws.active_dofs)
    empty!(ws.local_by_global)
    if target_layout.dofs.has_linear_constraints
        return [_local_parent_dofs!(ws.active_dofs, ws.local_by_global, d, target_layout)
                for d in target_data]
    else
        return [_local_parent_dofs_simple!(ws.active_dofs, ws.local_by_global, d, target_layout)
                for d in target_data]
    end
end

# Accumulate one transfer region's source-driven rhs contribution. The
# target mass matrix `M_T` and its Dirichlet column-elimination lift
# `−M_ac·c_c` are assembled separately by the standard assembler over the
# target's own integration regions (see [`transfer`](@ref)), so this pass
# integrates only the linear load
#
#     b_i += ∫_box u_S(x) · φ_iᵀ dx,
#
# where `u_S` is the source field reconstructed from its coefficients on
# the region's source parents and `φᵀ` are the target traces. A region the
# source does not cover contributes nothing (`u_S ≡ 0` there), so it is
# skipped wholesale.
#
# Loop structure (mirrors the load half of `_accumulate_qpoint_generic!`
# in `assembly.jl`):
#
#   1. Build per-parent records on both sides and the per-region target
#      dof-table; resize / reset the local rhs.
#   2. For each quadrature point: update basis values, then for each
#      component reconstruct `u_S(x)` and emit it against every target test
#      dof via `_emit_load!` (which fans the contribution through the dof
#      table's active branches).
#   3. Scatter the local rhs into the global rhs through `ws.active_dofs`.
function _assemble_transfer_rhs_region!(ws::TransferWorkspace{D,T}, rhs::Vector{T},
                                        source_coefficients, source_model::Model{D,T},
                                        target_model::Model{D,T},
                                        region::TransferRegion{D,T}) where {D,T}
    isempty(region.source_parents) && return nothing
    quadrature = region.quadrature
    jacobian = volume(region.box) / convert(T, 2^D)

    for target_layout in target_model.dofs.fields
        source_layout = _field_layout(source_model.dofs, target_layout.name)
        target_data = [_transfer_data(target_layout, p, ws.target_values)
                       for p in region.target_parents]
        source_data = [_transfer_data(source_layout, p, ws.source_values)
                       for p in region.source_parents]

        # Per-parent target dof-table (and `ws.active_dofs`) for this field.
        local_by_parent = _transfer_local_dofs!(ws, target_layout, target_data)
        n = length(ws.active_dofs)
        resize!(ws.local_rhs, n)
        fill!(ws.local_rhs, zero(T))

        for (eta, weight) in zip(quadrature.points, quadrature.weights)
            qweight = weight * jacobian
            _update_transfer_basis!(ws, region, eta)
            for component in 1:target_layout.components
                # Reconstruct u_S(x) and integrate it against the target
                # traces — a linear load.
                source_value = _field_value(source_data, source_layout, source_coefficients,
                                            component)
                for (test_data, table) in zip(target_data, local_by_parent)
                    for a in eachindex(test_data.raw_dofs)
                        contribution = qweight * source_value * test_data.values[a]
                        _emit_load!(ws.local_rhs, table, a, component, contribution)
                    end
                end
            end
        end

        # Scatter the local rhs into the global rhs (no matrix block).
        for (local_row, row) in pairs(ws.active_dofs)
            rhs[row] += ws.local_rhs[local_row]
        end
    end

    return nothing
end

# ── Public API and backends ──────────────────────────────────────────────────

"""
    abstract type TransferBackend end

Strategy object selecting how [`transfer`](@ref) moves a
[`Solution`](@ref) between two `Model`s. The default
[`L2Projection`](@ref) is general-purpose but introduces smoothing at
sharp features; [`Rewire`](@ref) is lossless when the target's active
basis contains the source's.
"""
abstract type TransferBackend end

"""
    L2Projection()
    L2Projection(matrix; factor=nothing)

Variational L² projection backend. Builds the target-side mass system

    Σ_T M_T c_T = b_T,   M_T,ij = ∫_Ω φ_iᵀ φ_jᵀ dx,   b_T,i = ∫_Ω u_S(x) φ_iᵀ dx,

on the union partition of source and target meshes, and solves it
sparse-direct for the target coefficients. This produces the L²-best
target approximation of `u_S` in the target's active basis.

The single-argument form `L2Projection(matrix)` reuses a precomputed
mass matrix — typically built once via
`assemble_matrix(target_model, mass_block(target_field))` and held
across many transfers. Passing `factor=lu(matrix)` (or any object
supporting `factor \\ rhs`) additionally skips the factorisation.

Both shortcuts are useful when the same target shows up repeatedly —
e.g. a moving-overlay transfer where only the source side changes
between time steps and the target's mass matrix can be cached across
them. Supplying a `factor` without a `matrix` is rejected: the matrix
is also needed to compute the solver residual reported in the returned
solution's diagnostics.

The cached-matrix shortcut requires the target to be homogeneously
constrained: a cached mass omits the Dirichlet column-elimination lift,
so a target with non-homogeneous physical Dirichlet data is rejected
(use the default `L2Projection()` there, which assembles the lift).
"""
struct L2Projection{M,F} <: TransferBackend
    matrix::M
    factor::F
end
function L2Projection(matrix=nothing; factor=nothing)
    matrix === nothing &&
        factor !== nothing &&
        throw(ArgumentError("L2Projection: cannot supply a factor without its matrix"))
    return L2Projection{typeof(matrix),typeof(factor)}(matrix, factor)
end

"""
    Rewire(; strict=true)

Raw-key coefficient mapping backend. For each (raw, component) of every
source field:

  1. Look up the source's [`TensorDofKey`](@ref) in the target field's
     `raw_keys` table.
  2. If a target raw matches and is active, copy the source's active
     coefficient into the target's active slot.
  3. If the target raw is missing (the source key has no counterpart)
     or is constrained on the target, either raise (strict mode) or
     skip silently (`strict = false`).

Reconstructs the source field pointwise when the target's active basis
*contains* the source's — typically after a monotone
[`activate!`](@ref) or an h-/p- refinement that adds new dofs without
removing any old ones. The interpolation is exact in that case
(coefficients are not modified), and the smoothing of the L² backend
is avoided.

`strict = true` (default) is the safe choice: any key mismatch is a
programming error and should surface at the transfer call rather than
in mysterious downstream numbers. Pass `strict = false` for
non-monotone transfers where a few dropped coefficients are expected
and acceptable.
"""
struct Rewire <: TransferBackend
    strict::Bool
end
Rewire(; strict::Bool=true) = Rewire(strict)

"""
    transfer(source_solution, source_model, target_model;
             via=L2Projection(), tolerance=target_model.dofs.tolerance) -> Solution

Move a [`Solution`](@ref) from `source_model` onto `target_model` using the
strategy `via`. Returns a fresh `Solution` pinned to `target_model.version`.

`via` defaults to [`L2Projection`](@ref); pass `Rewire()` for the lossless
raw-key path or a precomputed `L2Projection(matrix; factor)` to reuse a cached
target mass. The same verb `transfer` moves a [`QuadField`](@ref) when given
one — see that method for the quadrature-point schemes.

`tolerance` controls the geometric merge tolerance used when building the
source-target union partition; it defaults to the target dof layout's stored
tolerance and is ignored by strategies that do not walk the geometry (currently
[`Rewire`](@ref)).
"""
function transfer(source_solution::Solution, source_model::Model{D,T}, target_model::Model{D,T};
                  via::TransferBackend=L2Projection(),
                  tolerance=target_model.dofs.tolerance) where {D,T}
    return _transfer!(source_solution, source_model, target_model, via, tolerance)
end

# True when any field of `layout` carries a nonzero constrained value, i.e.
# non-homogeneous physical Dirichlet data. Artificial overlay constraints are
# always homogeneous (value 0), so this flags exactly the case where the
# Dirichlet column-elimination lift `−M_ac·c_c` is nonzero — the term the
# cached-matrix transfer path omits.
function _has_nonhomogeneous_constraints(layout::SystemLayout)
    return any(field -> any(!iszero, field.dofs.constrained_values), layout.fields)
end

# `L2Projection` implementation. The target-side mass matrix `M_T` and its
# Dirichlet column-elimination lift `−M_ac·c_c` are assembled by the
# STANDARD assembler over the target's own integration regions: under exact
# quadrature `∫_Ω φ_iᵀ φ_jᵀ` (target traces only) is independent of the
# partition, so the source/target union partition is needed only for the
# source-driven rhs `∫_Ω u_S(x) φ_iᵀ`, whose integrand mixes the two meshes.
# That source rhs is accumulated over the union partition, added onto the
# lift, and the system is solved sparse-direct.
#
# The cached-matrix backend (`M ≠ Nothing`) reuses `backend.matrix` and
# omits the lift, which vanishes only for a homogeneously-constrained
# target — so a non-homogeneous target is rejected there (the default
# `L2Projection()` re-assembles the mass and applies the lift correctly).
# The branch on `M === Nothing` tests a type parameter, so it folds at
# compile time; a supplied `backend.factor` short-circuits the
# factorisation.
function _transfer!(source_solution::Solution, source_model::Model{D,T}, target_model::Model{D,T},
                    backend::L2Projection{M,F}, tolerance) where {D,T,M,F}
    _assert_transfer_compatible(source_model, target_model)
    if M !== Nothing && _has_nonhomogeneous_constraints(target_model.dofs)
        throw(ArgumentError("L2Projection(matrix): the cached mass omits the Dirichlet lift, so " *
                            "it cannot transfer onto a target with non-homogeneous Dirichlet " *
                            "data. Use the default L2Projection() (no cached matrix) here."))
    end
    # The default path takes the target mass from the standard assembler, which
    # restricts to the immersed Ω via the FCM cut rules, while the source-driven
    # rhs below is assembled over the full mesh boxes of the union partition
    # (FCM-blind). For a target carrying a `physical_domain` those two are
    # inconsistent, so reject it rather than return a silently wrong projection.
    # Making the transfer union partition FCM-aware (both mass and rhs restricted
    # to Ω) is a separate follow-up.
    if target_model.problem.space.physical !== nothing
        throw(ArgumentError("L2 transfer onto a target with an immersed physical_domain is not yet " *
                            "supported: the target mass restricts to Ω but the source-driven rhs " *
                            "integrates over full mesh boxes, so the two are inconsistent."))
    end
    source_coefficients = _checked_coefficients(source_solution, source_model)

    nactive = active_unknowns(target_model.dofs)
    if nactive == 0
        return Solution(T[], target_model.version, SolverDiagnostics(:l2_projection, 0.0, true))
    end

    # Target mass + Dirichlet lift. Default path: assemble both from the
    # standard assembler over the target's own regions in one pass (the
    # bilinear pass shifts constrained trial columns onto the returned rhs,
    # which is the lift since there is no source load). Cached path: reuse
    # `backend.matrix` with a zero lift (homogeneous target, guarded above).
    if M === Nothing
        mass_blocks = map(mass_block, target_model.problem.fields)
        # Build the mass pattern WITHOUT clobbering `target_model.pattern`:
        # that cache is keyed to the model's own problem, and a transfer must
        # not evict it. `_assembly_region_lists` + `build_assembly_pattern`
        # yield a fresh pattern; `_assembly_pattern!` would mutate the cache.
        region_lists, key = _assembly_region_lists(target_model, mass_blocks)
        pattern = build_assembly_pattern(target_model, region_lists, true, key)
        sink = ScatterSink(zeros(T, length(pattern.rowval)), pattern)
        threaded = Threads.nthreads() > 1
        _, rhs = _assemble_partitioned!(sink, target_model, mass_blocks, (), nactive, true, nothing,
                                        nothing, threaded)
        mass = _matrix_from_pattern(pattern, sink.nzval)
    else
        mass = backend.matrix
        rhs = zeros(T, nactive)
    end

    # Source-driven rhs over the source/target union partition, accumulated
    # onto the lift already in `rhs`.
    regions = _transfer_regions(source_model, target_model; tolerance=tolerance)
    ws = _transfer_workspace(source_model, target_model)
    for region in regions
        _assemble_transfer_rhs_region!(ws, rhs, source_coefficients, source_model, target_model,
                                       region)
    end

    coefficients = F === Nothing ? mass \ rhs : backend.factor \ rhs
    residual = norm(mass * coefficients - rhs)
    return Solution(coefficients, target_model.version,
                    SolverDiagnostics(:l2_projection, Float64(residual), true))
end

# `Rewire` implementation: walk every active source dof, find its
# target counterpart by `TensorDofKey`, and copy the coefficient.
# Missing target counterparts and freshly-constrained target dofs are
# treated as errors under `strict = true` and silently skipped
# otherwise.
function _transfer!(source_solution::Solution, source_model::Model{D,T}, target_model::Model{D,T},
                    backend::Rewire, _tolerance) where {D,T}
    # Rewire walks raw `TensorDofKey`s, never the geometry — the
    # geometric merge tolerance accepted by `transfer` has no role
    # here and is ignored on purpose.
    _assert_transfer_compatible(source_model, target_model)
    source_coefficients = _checked_coefficients(source_solution, source_model)

    source_layout = source_model.dofs
    target_layout = target_model.dofs
    nactive = active_unknowns(target_layout)
    coefficients = zeros(T, nactive)

    for source_field in source_layout.fields
        target_field_idx = get(target_layout.by_name, source_field.name, 0)
        target_field_idx == 0 && continue
        target_field = target_layout.fields[target_field_idx]

        # Build a target-side key → raw lookup once per field. The
        # mapping is small (one entry per target raw key) and is reused
        # across every source raw key of the field.
        target_raw_by_key = Dict(key => raw for (raw, key) in pairs(target_field.dofs.raw_keys))

        for source_raw in eachindex(source_field.dofs.raw_keys)
            source_key = source_field.dofs.raw_keys[source_raw]
            target_raw = get(target_raw_by_key, source_key, 0)
            for c in 1:source_field.components
                source_active = source_field.dofs.active_component[source_raw, c]
                source_active == 0 && continue
                if target_raw == 0
                    backend.strict &&
                        throw(ArgumentError("Rewire: source active dof at key $(source_key) component $c has no target counterpart; pass `Rewire(strict=false)` to skip"))
                    continue
                end
                target_active = target_field.dofs.active_component[target_raw, c]
                if target_active == 0
                    backend.strict &&
                        throw(ArgumentError("Rewire: source active dof at key $(source_key) component $c is constrained in the target"))
                    continue
                end
                coefficients[target_field.offset + target_active] = source_coefficients[source_field.offset + source_active]
            end
        end
    end

    return Solution(coefficients, target_model.version, SolverDiagnostics(:rewire, 0.0, true))
end

# Friendlier error-message dispatches for the most common user
# mistakes: mismatched dimension / scalar type (the model parameters
# disagree, so the typed `transfer` above does not match) and missing
# target model (positional shorthand the API does not support).
function transfer(::Solution, ::Model, ::Model; kwargs...)
    throw(ArgumentError("source and target models must have the same dimension and scalar type for transfer"))
end

function transfer(::Solution, ::Model; kwargs...)
    throw(ArgumentError("transfer requires source and target models; call transfer(solution, source_model, target_model)"))
end
