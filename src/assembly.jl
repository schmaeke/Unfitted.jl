# Coupled-Galerkin assembly kernel for one or more fields over a
# superposition `Space`. This file owns the hot loop and the public
# assembly API; the user-facing types (weak forms, fields, problems)
# live in `problems.jl` and the prepared-problem lifecycle (`Model`,
# `prepare`, `move!`, …) lives in `model.jl`.
#
# The strategy is the classical sparse-COO assembly path: walk every
# admissible integration region, evaluate basis values and physical
# gradients per parent level, and accumulate the user's weak form into
# a thread-local local matrix / rhs that gets emitted as
# `(row, col, val)` triplets at the end of the region.
#
# Two contracts shape the API:
#
#   * **Channel calculus.** User forms produce `TestChannels` (value
#     coefficient + gradient coefficient) given pointwise trial data as
#     `TrialChannels`. Assembly contracts them against test basis values
#     and gradients via `_test_contribution`. The same calculus drives
#     both bilinear contributions (`a(u,v)`) and linear contributions
#     (`ℓ(v)`).
#   * **Lifecycle.** The `Model` carrying the integration plan, dof
#     layout, diagnostics, and (lazily) the assembled matrix and right-
#     hand side is built and mutated in `model.jl`. `assemble!`
#     populates the cached matrix / rhs; mutators clear them and bump
#     `model.version` so any outstanding `Solution` is detected as
#     stale on reuse.
#
# Constrained dofs (physical Dirichlet, artificial overlay, mixed) are
# eliminated column-wise during assembly: a constrained trial column's
# stiffness contribution is moved to the right-hand side, weighted by
# the dof's stored constrained value (zero for overlay/homogeneous,
# projected datum for nonzero Dirichlet).

# ── Field evaluation and FormState ────────────────────────────────────────────

# Per-parent record carrying everything `_field_value`/`_field_gradient`
# need to reconstruct a field at a quadrature point: the parent's basis
# family, order, local-id list, gradient scaling factor, per-axis 1D
# factor buffers, and the slot for the evaluated values and gradients.
# Allocates fresh buffers; for hot-loop use go through the workspace-
# backed `_parent_dof_data` from the assembly section instead.
function _parent_basis_data(V::Space{D,T}, layout::DofLayout{D,T}, parent::ParentRef{D,T};
                            gradients::Bool=true) where {D,T}
    level = _level_by_id(V, parent.level)
    order = level.order
    local_ids = local_basis_indices(level.basis, order, level.mode)
    values = Vector{T}(undef, length(local_ids))
    gradient_values = gradients ? Vector{SVector{D,T}}(undef, length(values)) : SVector{D,T}[]
    raw_dofs = cell_dofs(layout, parent.level, parent.cell)
    gradient_scale = SVector{D,T}(2 .* inv.(edge_lengths(parent.parent_box)))
    val1d = _factor_buffers(order, T)
    der1d = gradients ? _factor_buffers(order, T) : ntuple(_ -> Vector{T}(undef, 0), D)
    return (; parent, basis=level.basis, order, local_ids, gradient_scale, values,
            gradients=gradient_values, raw_dofs, val1d, der1d)
end

# Evaluate the value-only basis tables at a region-reference point `eta`
# into the parent record's value buffer. Used by the non-hot-loop
# evaluation paths (FormState value access, `foreach_quadrature_point`
# without state, `l2_error`).
function _update_parent_basis_values!(data, eta::SVector{D,T}) where {D,T}
    xi = reference_to_physical(data.parent.local_box, eta)
    _tensor_values!(data.values, data.local_ids, data.order, xi, data.val1d)
    return data
end

# Reconstruct a scalar field value over one parent. Sums
# `Σ_i dof_value(...) × basis_value_i` over the parent's raw dofs;
# constrained dofs are resolved through `dof_value`, so the
# reconstruction includes their pinned values.
function _field_value(data, layout::Union{DofLayout{D,T},FieldLayout{D,T}}, coefficients,
                      component::Integer=1) where {D,T}
    R = promote_type(T, eltype(coefficients))
    result = zero(R)
    for i in eachindex(data.raw_dofs)
        result += dof_value(layout, coefficients, data.raw_dofs[i], component) * data.values[i]
    end
    return result
end

# Sum the parent contributions of `_field_value` over every parent
# covering the integration region — the actual superposed value.
function _field_value(parent_data::AbstractVector, layout::Union{DofLayout{D,T},FieldLayout{D,T}},
                      coefficients, component::Integer=1) where {D,T}
    R = promote_type(T, eltype(coefficients))
    result = zero(R)
    for data in parent_data
        result += _field_value(data, layout, coefficients, component)
    end
    return result
end

# Gradient analogues of `_field_value`. Returns an `SVector{D}` per
# parent, then sums to get the superposed gradient.
function _field_gradient(data, layout::Union{DofLayout{D,T},FieldLayout{D,T}}, coefficients,
                         component::Integer=1) where {D,T}
    R = promote_type(T, eltype(coefficients))
    result = zero(SVector{D,R})
    for i in eachindex(data.raw_dofs)
        result += dof_value(layout, coefficients, data.raw_dofs[i], component) * data.gradients[i]
    end
    return result
end

function _field_gradient(parent_data::AbstractVector,
                         layout::Union{DofLayout{D,T},FieldLayout{D,T}}, coefficients,
                         component::Integer=1) where {D,T}
    R = promote_type(T, eltype(coefficients))
    result = zero(SVector{D,R})
    for data in parent_data
        result += _field_gradient(data, layout, coefficients, component)
    end
    return result
end

"""
    FormState

The current solution iterate at a quadrature point, exposed to weak-form
callbacks as `q.state` when assembling with `state=`. Query it with
`value(q.state, field[, component])` and `field_gradient(q.state, field[,
component])`, where `field` is a field name `Symbol` or a
[`Field`](@ref). Evaluation is lazy and reuses the basis data already
computed for the integration region, so a callback can read `q.state`
multiple times without recomputing anything.

`FormState` is what makes the assembly path Newton-linearisation
friendly: a `bilinear` callback that wants the current iterate
`uₖ(x)` for building a tangent operator simply asks `q.state` for the
value or gradient.

`field_gradient` (rather than the natural shorter `gradient`) avoids a
name collision with `Tensors.gradient`, the automatic-differentiation
entry point of the Tensors.jl package, so the two can be loaded
together unrestricted; `value` does not collide and keeps its short
name.
"""
struct FormState{FD,L<:SystemLayout,C}
    field_data::FD
    layout::L
    coefficients::C
end

# Resolve a field name to its `fields` index inside the `SystemLayout`.
# Used by the public `value` / `field_gradient` on `FormState`.
function _state_field_index(state::FormState, name::Symbol)
    index = get(state.layout.by_name, name, 0)
    index == 0 && throw(ArgumentError("unknown field $name"))
    return index
end

"""
    value(state::FormState, name_or_field[, component=1])

Read the named field's value at the current quadrature point. `name` is
either a `Symbol` matching a field name in the model or a [`Field`](@ref)
object. For multi-component fields, pass `component`.
"""
function value(state::FormState, name::Symbol, component::Integer=1)
    index = _state_field_index(state, name)
    return _field_value(state.field_data[index], state.layout.fields[index], state.coefficients,
                        component)
end

"""
    field_gradient(state::FormState, name_or_field[, component=1])

Read the named field's physical gradient at the current quadrature
point. Same field/component semantics as [`value`](@ref). The leading
`field_` is there to avoid colliding with `Tensors.gradient` (the
automatic-differentiation entry point) when both packages are loaded
together.
"""
function field_gradient(state::FormState, name::Symbol, component::Integer=1)
    index = _state_field_index(state, name)
    return _field_gradient(state.field_data[index], state.layout.fields[index], state.coefficients,
                           component)
end

value(state::FormState, field::Field, component::Integer=1) = value(state, field.name, component)
function field_gradient(state::FormState, field::Field, component::Integer=1)
    field_gradient(state, field.name, component)
end

# Tiny helper raising the "no state attached" error from a single
# location. Reaching it indicates a callback dereferenced `q.state`
# without the caller passing `state =` to the assembly entry point.
function _no_form_state()
    throw(ArgumentError("q.state is unavailable; assemble with `state = current_iterate`"))
end
value(::Nothing, args...) = _no_form_state()
field_gradient(::Nothing, args...) = _no_form_state()

# ── Assembly workspace and hot loop ───────────────────────────────────────────

# Build a sparse matrix from COO triplets. Symmetric forms are assembled
# in the lower triangle; mirroring as `A + Aᵀ − diag(A)` yields the
# symmetric matrix exactly (IEEE addition is commutative, so the result
# is symmetric to the bit). `dropzeros!` collapses any explicit zeros
# produced by Dirichlet column elimination.
function _sparse_from_triplets(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{T}, n::Int,
                               symmetric::Bool) where {T}
    matrix = sparse(rows, cols, vals, n, n)
    symmetric && (matrix = matrix + matrix' - spdiagm(0 => diag(matrix)))
    dropzeros!(matrix)
    return matrix
end

# Build the cell-local-to-region-active dof index table for one parent
# and its field. For each `(raw_dof, component)`:
#
#   * map the raw dof to its global active id via the field layout;
#   * if it is constrained (global id == 0), record `0`;
#   * otherwise register the global id in `local_by_global` (a fresh
#     local index per *distinct* global id) and record the local index.
#
# The shared `local_by_global` cache ensures that a dof shared between
# two parents in the region gets the same local row/column on both —
# i.e. their stiffness contributions land at the same local matrix
# position before being emitted as triplets. Used inside
# `_assemble_region!` per region.
function _local_parent_dofs!(active_dofs::Vector{Int}, local_by_global::Dict{Int,Int}, data,
                             layout::FieldLayout)
    local_dofs = Matrix{Int}(undef, length(data.raw_dofs), layout.components)
    for i in eachindex(data.raw_dofs), component in 1:layout.components
        dof = _field_component_dof(layout, data.raw_dofs[i], component)
        local_dofs[i, component] = dof == 0 ? 0 : get!(local_by_global, dof) do
            push!(active_dofs, dof)
            length(active_dofs)
        end
    end
    return local_dofs
end

# One-shot variant for the off-assembly paths (boundary projection):
# builds the same `(active_dofs, local_by_parent)` pair from scratch
# without consuming a shared workspace.
function _local_active_dof_table(parent_data, layout::FieldLayout)
    active_dofs = Int[]
    local_by_global = Dict{Int,Int}()
    local_by_parent = [_local_parent_dofs!(active_dofs, local_by_global, data, layout)
                       for data in parent_data]
    return active_dofs, local_by_parent
end

# Reusable "no contribution" channel — the all-zero `TestChannels` that
# the bilinear path falls back on when a component-unaware form is
# asked for an off-diagonal (trial_component ≠ test_component) entry.
function _zero_test_channels(::Val{D}, ::Type{T}) where {D,T}
    TestChannels(zero(T), _zero_gradient(Val(D), T))
end

# Evaluate the linear callback at a quadrature point and coerce its
# return to `TestChannels{D,T}`. Component-aware forms get
# `(q, test_component)`; the rest get `(q,)` and apply to every test
# component uniformly.
function _linear_channels(form::WeakForm, q, test_component::Int, ::Val{D}, ::Type{T}) where {D,T}
    value = form.component_aware ? form.linear(q, test_component) : form.linear(q)
    return _as_test_channels(value, Val(D), T)
end

# Evaluate the bilinear callback. Component-aware forms receive the full
# `(q, trial, test_component)` triple. Component-unaware forms receive
# `(q, trial)` and apply only when `trial.component == test_component`
# (the implicit diagonal-in-components pattern) — for off-diagonal
# components the helper returns a zero channel to keep the assembly
# loops branch-free.
function _bilinear_channels(form::WeakForm, q, trial::TrialChannels{D,T},
                            test_component::Int) where {D,T}
    if form.component_aware
        return _as_test_channels(form.bilinear(q, trial, test_component), Val(D), T)
    end

    trial.component == test_component || return _zero_test_channels(Val(D), T)
    return _as_test_channels(form.bilinear(q, trial), Val(D), T)
end

# Emit a region's local matrix and rhs to the global COO triplet and rhs
# arrays. The rhs is scattered into the global rhs by row; the matrix
# is emitted column-major so the resulting sparse construction is more
# cache-friendly. Zero entries (e.g. from never-touched constrained
# columns) are skipped, both to keep the triplet list short and so
# `dropzeros!` does not have to do the work later.
function _emit_local_system!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{T}, rhs::Vector{T},
                             active_dofs::AbstractVector{Int}, local_matrix::AbstractMatrix{T},
                             local_rhs::AbstractVector{T}) where {T}
    for (local_row, row) in pairs(active_dofs)
        rhs[row] += local_rhs[local_row]
    end

    if !isempty(local_matrix)
        for local_col in axes(local_matrix, 2)
            col = active_dofs[local_col]
            for local_row in axes(local_matrix, 1)
                entry = local_matrix[local_row, local_col]
                iszero(entry) && continue
                push!(rows, active_dofs[local_row])
                push!(cols, col)
                push!(vals, entry)
            end
        end
    end

    return nothing
end

# Resolve a field name to its index inside a `SystemLayout`. Used by
# every block / load loop to translate field references into the
# layout's field-major data structures.
function _field_index(layout::SystemLayout, name::Symbol)
    index = get(layout.by_name, name, 0)
    index == 0 && throw(ArgumentError("unknown field $name"))
    return index
end

# One-shot per-parent record collector. Used by paths that allocate
# fresh basis buffers (boundary projection, single-shot evaluation);
# the hot-loop path uses `_parent_dof_data` against a workspace instead.
function _field_parent_data(V, layout::FieldLayout, parents; gradients::Bool=true)
    [_parent_basis_data(V, layout.dofs, parent; gradients) for parent in parents]
end

"""
    AssemblyWorkspace{D,T}

Thread-local assembly scratch. Carries every buffer the hot loop needs:

  - `local_ids[i]` — tensor-product multi-indices of level `i`'s basis.
  - `orders[i]`    — polynomial order tuple of level `i`'s basis.
  - `values[i]`, `gradients[i]` — per-level basis value / gradient
    buffers. Indexed by the level id; reused across every region a
    thread processes. A region carries at most one parent per level, so
    the level-indexed buffers never collide within a region.
  - `val1d[i]`, `der1d[i]` — per-axis 1D factor buffers for the basis
    evaluator.
  - `active_dofs`, `local_by_global` — per-region local-to-global dof
    table (rebuilt per region; the dict is `empty!`d, not reallocated).
  - `local_matrix`, `local_rhs` — flat scratch buffers reshaped to
    `n×n` and `n` per region.

Constructed by [`_assembly_workspace`](@ref) once per thread (or once
per assembly call in the serial path).
"""
struct AssemblyWorkspace{D,T}
    local_ids::Vector{Vector{CartesianIndex{D}}}
    orders::Vector{NTuple{D,Int}}
    values::Vector{Vector{T}}
    gradients::Vector{Vector{SVector{D,T}}}
    val1d::Vector{NTuple{D,Vector{T}}}
    der1d::Vector{NTuple{D,Vector{T}}}
    active_dofs::Vector{Int}
    local_by_global::Dict{Int,Int}
    local_matrix::Vector{T}
    local_rhs::Vector{T}
end

# Build a fresh workspace for `model`. Sizes every per-level buffer to
# the level's basis count and per-axis order. Called once per thread in
# the threaded path and once per assembly call in the serial path.
function _assembly_workspace(model::Model{D,T}) where {D,T}
    levels = model.problem.space.levels
    nlev = length(levels)
    local_ids = Vector{Vector{CartesianIndex{D}}}(undef, nlev)
    orders = Vector{NTuple{D,Int}}(undef, nlev)
    values = Vector{Vector{T}}(undef, nlev)
    gradients = Vector{Vector{SVector{D,T}}}(undef, nlev)
    val1d = Vector{NTuple{D,Vector{T}}}(undef, nlev)
    der1d = Vector{NTuple{D,Vector{T}}}(undef, nlev)
    for level in levels
        i = level.id
        ids = local_basis_indices(level.basis, level.order, level.mode)
        local_ids[i] = ids
        orders[i] = level.order
        values[i] = Vector{T}(undef, length(ids))
        gradients[i] = Vector{SVector{D,T}}(undef, length(ids))
        val1d[i] = _factor_buffers(level.order, T)
        der1d[i] = _factor_buffers(level.order, T)
    end
    return AssemblyWorkspace{D,T}(local_ids, orders, values, gradients, val1d, der1d, Int[],
                                  Dict{Int,Int}(), T[], T[])
end

# Slim per-parent record: basis values / gradients alias the level's
# workspace buffers (no allocation); only the field-specific raw dof
# ids are materialised per parent (and even those are a view into the
# pre-built `cell_dofs_by_level` table on the dof layout). Accepts
# both `ParentRef` (volume) and `FacetParent` (facet) — the body only
# touches `parent.level` and `parent.cell`, which both carry.
function _parent_dof_data(ws::AssemblyWorkspace{D,T}, layout::FieldLayout{D,T},
                          parent::Union{ParentRef{D,T},FacetParent{D,T}}) where {D,T}
    lvl = parent.level
    raw_dofs = cell_dofs(layout.dofs, lvl, parent.cell)
    return (; level=lvl, raw_dofs, values=ws.values[lvl], gradients=ws.gradients[lvl])
end

# Evaluate basis values and physical gradients once per level present in
# the region (parents are unique per level), writing into the workspace
# buffers. The physical-gradient scale factor is the standard
# axis-aligned chain rule `scale[d] = 2 / edge_lengths(parent_box)[d]`
# from `physical_basis_gradients!` in `basis.jl`.
function _update_region_basis!(ws::AssemblyWorkspace{D,T}, region::VolumeRegion{D,T},
                               eta::SVector{D,T}) where {D,T}
    for parent in region.parents
        lvl = parent.level
        xi = reference_to_physical(parent.local_box, eta)
        scale = SVector{D,T}(2 .* inv.(edge_lengths(parent.parent_box)))
        _tensor_values_grads!(ws.values[lvl], ws.gradients[lvl], ws.local_ids[lvl], ws.orders[lvl],
                              xi, scale, ws.val1d[lvl], ws.der1d[lvl])
    end
    return nothing
end

# Build the per-field local-to-global dof table for this region against
# the workspace's `active_dofs` / `local_by_global`. Both scratch
# structures are cleared first so the result is per-region; the rebuilt
# table is `local_by_field[field_index][parent_index]`, the matrix of
# local row/column indices for that parent and field.
function _local_active_dof_table!(ws::AssemblyWorkspace, field_data, layout::SystemLayout)
    empty!(ws.active_dofs)
    empty!(ws.local_by_global)
    return [[_local_parent_dofs!(ws.active_dofs, ws.local_by_global, data, field_layout)
             for data in field_data[field_index]]
            for (field_index, field_layout) in pairs(layout.fields)]
end

"""
    _assemble_region!(ws, rows, cols, vals, rhs, model, region, blocks, loads,
                      symmetric, state_coefficients=nothing, point_offset=0)

Assemble every weak-form contribution at every quadrature point of one
integration region. This is the assembly hot loop: a single call walks
the region's quadrature points and updates the COO triplet buffers and
right-hand-side vector in place.

Structure of the body, in order:

  1. **Region setup.** Compute the reference-to-physical Jacobian
     `vol(box) / 2ᴰ`, build per-field parent records (aliasing the
     workspace buffers), and the per-region local-to-global dof table.
     Optionally build a [`FormState`](@ref) so callbacks can read the
     current iterate.
  2. **Local-system setup.** Resize the workspace's flat
     `local_matrix` / `local_rhs` buffers to `n×n` and `n` for the
     number of distinct active dofs touching this region; clear them.
  3. **Quadrature loop.** At each quadrature point:
       a. Compose `q = (; x, weight, point, state)` and evaluate every
          parent's basis values / physical gradients into the workspace.
       b. **Loads:** for every `LoadForm`, evaluate the linear channels
          for every test component and accumulate the test-contribution
          weighted by `qweight` into the local rhs.
       c. **Blocks:** for every `BlockForm`, iterate trial components,
          trial parents, trial dofs. Build `trial::TrialChannels` for
          each, then iterate test components, evaluate the bilinear
          channels, and iterate test parents and dofs accumulating
          local-matrix entries.
       d. **Dirichlet elimination.** A constrained trial column
          (`col == 0`) shifts its stiffness contribution onto the rhs,
          weighted by the dof's stored `constrained_value`.
       e. **Symmetry.** When the form is symmetric, only emit
          `row ≥ col` entries; `_sparse_from_triplets` mirrors at the
          end.
  4. **Emit.** Flush the local matrix and rhs to the global triplet
     buffers and rhs via [`_emit_local_system!`](@ref).

`point_offset` is the global quad-point offset of this region in the
plan, so `q.point = point_offset + local_qp` is the stable index used
by [`foreach_quadrature_point`](@ref) and per-point history data.
"""
function _assemble_region!(ws::AssemblyWorkspace{D,T}, rows::Vector{Int}, cols::Vector{Int},
                           vals::Vector{T}, rhs::Vector{T}, model::Model{D,T},
                           region::VolumeRegion{D,T}, blocks, loads, symmetric::Bool,
                           state_coefficients=nothing, point_offset::Int=0) where {D,T}
    # Region setup: Jacobian, per-field parent data, region-local dof
    # table, optional state, local matrix / rhs buffers.
    quadrature = region.quadrature
    jacobian = volume(region.box) / convert(T, 2^D)
    field_data, local_by_field, state, local_matrix, local_rhs = _region_workspace_setup!(ws, model,
                                                                                          region.parents,
                                                                                          state_coefficients,
                                                                                          !isempty(blocks))

    # Quadrature loop.
    for (local_qp, (eta, weight)) in enumerate(zip(quadrature.points, quadrature.weights))
        qweight = weight * jacobian
        x = reference_to_physical(region.box, eta)
        q = (; x, weight=qweight, point=point_offset + local_qp, state, normal=nothing,
             sides=nothing)
        _update_region_basis!(ws, region, eta)
        _accumulate_qpoint!(local_matrix, local_rhs, q, qweight, field_data, local_by_field, blocks,
                            loads, symmetric, model, Val(D), T)
    end

    # Emit the local system to the global COO triplets / rhs.
    _emit_local_system!(rows, cols, vals, rhs, ws.active_dofs, local_matrix, local_rhs)
    return nothing
end

"""
    _assemble_region!(ws, rows, cols, vals, rhs, model, region::FacetRegion,
                      blocks, loads, symmetric, state_coefficients=nothing,
                      point_offset=0)

Facet analogue of the volume hot loop. Identical inner work
(`_accumulate_qpoint!`) — the only differences are:

  * the quadrature is precomputed in physical coordinates and the
    weight is already physical-frame (Gauss × Jacobian),
  * each parent's reference point `xi` is obtained by mapping the
    physical Q-point back into the parent cell via
    `physical_to_reference(parent.parent_box, x)`,
  * the `q` tuple carries `q.normal` (the facet's outward unit normal)
    and `q.sides` (the codim-`K` facet identifier) so user forms can
    express Neumann / Robin / Nitsche contributions naturally.

Every active local basis mode of every parent participates — `is_facet_basis`
is **not** applied here. Modes with zero *value* on the facet (e.g. integrated
Legendre bubbles along a constrained axis) can still carry nonzero
*gradient*, which Nitsche-style forms rely on.
"""
function _assemble_region!(ws::AssemblyWorkspace{D,T}, rows::Vector{Int}, cols::Vector{Int},
                           vals::Vector{T}, rhs::Vector{T}, model::Model{D,T},
                           region::FacetRegion{D,T}, blocks, loads, symmetric::Bool,
                           state_coefficients=nothing, point_offset::Int=0) where {D,T}
    field_data, local_by_field, state, local_matrix, local_rhs = _region_workspace_setup!(ws, model,
                                                                                          region.parents,
                                                                                          state_coefficients,
                                                                                          !isempty(blocks))

    for (local_qp, x) in pairs(region.points)
        qweight = region.weights[local_qp]
        q = (; x, weight=qweight, point=point_offset + local_qp, state, normal=region.normal,
             sides=region.sides)
        _update_physical_basis!(ws, region.parents, x)
        _accumulate_qpoint!(local_matrix, local_rhs, q, qweight, field_data, local_by_field, blocks,
                            loads, symmetric, model, Val(D), T)
    end

    _emit_local_system!(rows, cols, vals, rhs, ws.active_dofs, local_matrix, local_rhs)
    return nothing
end

# Refresh basis values and physical gradients per parent at a physical
# quadrature point `x`. Mirrors `_update_region_basis!` for volume
# regions but maps `x` through each parent's own `parent_box` instead
# of through a region-local frame (facet / surface regions have no
# single reference frame shared by every parent — the constrained
# coordinate is fixed but free-axis coordinates run across multiple
# cells per level). Used by `_assemble_region!(::FacetRegion, …)` and
# `_assemble_region!(::SurfaceRegion, …)`.
function _update_physical_basis!(ws::AssemblyWorkspace{D,T}, parents, x::SVector{D,T}) where {D,T}
    for parent in parents
        lvl = parent.level
        xi = physical_to_reference(parent.parent_box, x)
        scale = SVector{D,T}(2 .* inv.(edge_lengths(parent.parent_box)))
        _tensor_values_grads!(ws.values[lvl], ws.gradients[lvl], ws.local_ids[lvl], ws.orders[lvl],
                              xi, scale, ws.val1d[lvl], ws.der1d[lvl])
    end
    return nothing
end

"""
    _assemble_region!(ws, rows, cols, vals, rhs, model, region::SurfaceRegion,
                      blocks, loads, symmetric, state_coefficients=nothing,
                      point_offset=0)

Surface (immersed-boundary) analogue of the volume and facet hot
loops. The user-supplied [`BoundaryMesh`](@ref) cell maps onto one
`SurfaceRegion`; the parent set is constant within the region by the
strict construction-time check. Per quadrature point:

  * basis values + physical gradients refresh per parent via
    `physical_to_reference(parent.parent_box, x)`,
  * `q.normal` carries the region's per-Q-point unit normal (constant
    per region in the MVP; per-Q-point in future for curved meshes),
  * `q.sides` is `nothing` — immersed surfaces have no facet
    `(axis, side)` identifier (the user owns the geometry).

The basis evaluation visits every active local mode; nothing is
filtered by facet-incidence.
"""
function _assemble_region!(ws::AssemblyWorkspace{D,T}, rows::Vector{Int}, cols::Vector{Int},
                           vals::Vector{T}, rhs::Vector{T}, model::Model{D,T},
                           region::SurfaceRegion{D,T}, blocks, loads, symmetric::Bool,
                           state_coefficients=nothing, point_offset::Int=0) where {D,T}
    field_data, local_by_field, state, local_matrix, local_rhs = _region_workspace_setup!(ws, model,
                                                                                          region.parents,
                                                                                          state_coefficients,
                                                                                          !isempty(blocks))

    for (local_qp, x) in pairs(region.points)
        qweight = region.weights[local_qp]
        q = (; x, weight=qweight, point=point_offset + local_qp, state,
             normal=region.normals[local_qp], sides=nothing)
        _update_physical_basis!(ws, region.parents, x)
        _accumulate_qpoint!(local_matrix, local_rhs, q, qweight, field_data, local_by_field, blocks,
                            loads, symmetric, model, Val(D), T)
    end

    _emit_local_system!(rows, cols, vals, rhs, ws.active_dofs, local_matrix, local_rhs)
    return nothing
end

# Shared workspace setup for both volume and facet `_assemble_region!`
# methods: build the per-field parent records (aliasing the workspace
# buffers), build the per-region local-to-global dof table, attach the
# optional `FormState`, and resize/clear the local matrix and rhs
# buffers. Returns the assembled view objects the hot loop walks.
function _region_workspace_setup!(ws::AssemblyWorkspace{D,T}, model::Model{D,T}, parents,
                                  state_coefficients, have_blocks::Bool) where {D,T}
    field_data = [[_parent_dof_data(ws, layout, parent) for parent in parents]
                  for layout in model.dofs.fields]
    local_by_field = _local_active_dof_table!(ws, field_data, model.dofs)
    state = state_coefficients === nothing ? nothing :
            FormState(field_data, model.dofs, state_coefficients)

    n = length(ws.active_dofs)
    nn = have_blocks ? n * n : 0
    resize!(ws.local_matrix, nn)
    fill!(ws.local_matrix, zero(T))
    resize!(ws.local_rhs, n)
    fill!(ws.local_rhs, zero(T))
    local_matrix = reshape(view(ws.local_matrix, 1:nn), have_blocks ? n : 0, have_blocks ? n : 0)
    local_rhs = ws.local_rhs
    return field_data, local_by_field, state, local_matrix, local_rhs
end

# Accumulate every weak-form contribution at one quadrature point into
# the per-region local matrix and rhs. Walks loads first (linear
# channels into the rhs), then blocks (bilinear channels into the
# matrix, with Dirichlet-column elimination moving constrained columns
# onto the rhs). Symmetric blocks emit only the lower-triangular
# entries; `_sparse_from_triplets` mirrors at the end.
#
# Extracted from the two `_assemble_region!` overloads so the volume
# and facet hot loops share their inner work — every kind-specific
# detail is contained in the caller (Q-point source, basis refresh,
# `q` tuple shape).
function _accumulate_qpoint!(local_matrix, local_rhs, q, qweight::T, field_data, local_by_field,
                             blocks::B, loads::L, symmetric::Bool, model::Model{D,T}, ::Val{D},
                             ::Type{T}) where {B,L,D,T}
    # Loads: linear channels per test parent / dof into local rhs.
    for load in loads
        test_index = _field_index(model.dofs, load.test_name)
        test_layout = model.dofs.fields[test_index]
        test_data = field_data[test_index]
        local_rows_by_parent = local_by_field[test_index]

        for test_component in 1:test_layout.components
            linear_channels = _linear_channels(load.form, q, test_component, Val(D), T)

            for (data, local_rows) in zip(test_data, local_rows_by_parent)
                for a in eachindex(data.raw_dofs)
                    local_row = local_rows[a, test_component]
                    local_row == 0 && continue
                    local_rhs[local_row] += qweight *
                                            _test_contribution(linear_channels, data.values[a],
                                                               data.gradients[a])
                end
            end
        end
    end

    # Blocks: bilinear channels per (trial × test) pair into local
    # matrix; Dirichlet-column elimination redirects constrained-trial
    # contributions onto the rhs.
    for block in blocks
        test_index = _field_index(model.dofs, block.test_name)
        trial_index = _field_index(model.dofs, block.trial_name)
        test_layout = model.dofs.fields[test_index]
        trial_layout = model.dofs.fields[trial_index]

        for trial_component in 1:trial_layout.components
            for (trial_data, local_cols) in
                zip(field_data[trial_index], local_by_field[trial_index])
                for b in eachindex(trial_data.raw_dofs)
                    col = _field_component_dof(trial_layout, trial_data.raw_dofs[b],
                                               trial_component)
                    trial = TrialChannels(trial_component, trial_data.values[b],
                                          trial_data.gradients[b])

                    for test_component in 1:test_layout.components
                        channels = _bilinear_channels(block.form, q, trial, test_component)
                        for (test_data, local_rows) in
                            zip(field_data[test_index], local_by_field[test_index])
                            for a in eachindex(test_data.raw_dofs)
                                row = _field_component_dof(test_layout, test_data.raw_dofs[a],
                                                           test_component)
                                row == 0 && continue
                                symmetric && col != 0 && row < col && continue
                                entry = qweight * _test_contribution(channels, test_data.values[a],
                                                                     test_data.gradients[a])
                                local_row = local_rows[a, test_component]
                                if col == 0
                                    local_rhs[local_row] -= entry *
                                                            constrained_value(trial_layout.dofs,
                                                                              trial_data.raw_dofs[b],
                                                                              trial_component)
                                else
                                    local_matrix[local_row, local_cols[b, trial_component]] += entry
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return nothing
end

# Upper bound on the number of matrix triplets emitted by assembly,
# used to size the COO buffers via `sizehint!`. Correctness-neutral
# (sizehint only reserves capacity), and intentionally an over-count:
# constrained dofs are ignored and the full-block / lower-triangular
# square is used. The overshoot is harmless and keeps the buffer growth
# from showing up in profiles.
function _triplet_estimate(model::Model, plan::IntegrationPlan, symmetric::Bool)
    ncomp = sum(field.components for field in model.dofs.fields)
    total = 0
    for region in plan.regions
        base = 0
        for parent in region.parents
            level = _level_by_id(model.problem.space, parent.level)
            base += local_basis_count(level.basis, level.order, level.mode)
        end
        s = base * ncomp
        total += symmetric ? (s * (s + 1)) ÷ 2 : s * s
    end
    return total
end

# Total quad-point count across all regions of the plan. Used by
# `nquadpoints`.
function _quadrature_count(plan::IntegrationPlan)
    sum(length(r.quadrature.weights) for r in plan.regions; init=0)
end

# Per-region quadrature-point counts, used to compute the
# `point_offset` each region sees. Defined for both volume and facet
# regions so the serial/threaded drivers stay region-kind-agnostic.
_region_qpoint_count(region::VolumeRegion) = length(region.quadrature.weights)
_region_qpoint_count(region::FacetRegion) = length(region.weights)
_region_qpoint_count(region::SurfaceRegion) = length(region.weights)

# Cumulative offsets of the global `q.point` index over a region list.
# Volume and facet kinds maintain independent counters: callers pass
# the kind-specific region list and receive offsets stable across
# threaded vs. serial assembly.
function _region_qpoint_offsets(regions)
    offsets = Vector{Int}(undef, length(regions))
    acc = 0
    for (i, region) in enumerate(regions)
        offsets[i] = acc
        acc += _region_qpoint_count(region)
    end
    return offsets
end

# Append COO triplets and rhs entries from one assembly walk onto an
# already-initialised set of buffers. Used by `assemble!` /
# `assemble_matrix` / `assemble_vector` to accumulate contributions
# from multiple region lists (volume + per-selector facets) into the
# same destination matrices and rhs.
function _append_assembly!(rows, cols, vals, rhs, partial)
    append!(rows, partial.rows)
    append!(cols, partial.cols)
    append!(vals, partial.vals)
    rhs .+= partial.rhs
    return rows, cols, vals, rhs
end

# Serial assembly driver: a single workspace and a single COO/rhs
# buffer set walked across every region in order. `region_filter` lets
# load assemblies skip regions that do not intersect the load's
# support. The `regions` argument is any iterable of `VolumeRegion`s or
# `FacetRegion`s — Julia dispatches the right `_assemble_region!`
# method automatically.
function _assemble_system_serial(model::Model{D,T}, regions, nactive::Int, symmetric::Bool, blocks,
                                 loads, region_filter, state_coefficients,
                                 size_hint::Int=0) where {D,T}
    rows = Int[]
    cols = Int[]
    vals = T[]
    rhs = zeros(T, nactive)
    ws = _assembly_workspace(model)
    offsets = _region_qpoint_offsets(regions)
    if size_hint > 0
        sizehint!(rows, size_hint)
        sizehint!(cols, size_hint)
        sizehint!(vals, size_hint)
    end

    for (region_index, region) in enumerate(regions)
        region_filter === nothing || region_filter(region) || continue
        _assemble_region!(ws, rows, cols, vals, rhs, model, region, blocks, loads, symmetric,
                          state_coefficients, offsets[region_index])
    end

    return (; rows, cols, vals, rhs)
end

# Threaded assembly driver: spawn `Threads.nthreads()` tasks, each with
# its own workspace and COO / rhs buffers. Distribute regions in a
# striped pattern (`region_index in task_id:task_count:nregions`) so
# every task processes a uniform random sample of the list. After all
# tasks complete, reduce the per-task buffers into the final
# COO / rhs vectors.
#
# Region-kind-agnostic: the `regions` iterable can be a volume plan's
# region vector or a facet selector's region list. The striped
# distribution is simple and balances well when region costs are
# roughly uniform.
function _assemble_system_threaded(model::Model{D,T}, regions, nactive::Int, symmetric::Bool,
                                   blocks, loads, region_filter, state_coefficients,
                                   size_hint::Int=0) where {D,T}
    task_count = Threads.nthreads()
    task_size_hint = size_hint > 0 ? cld(size_hint, task_count) : 0
    offsets = _region_qpoint_offsets(regions)
    n_regions = length(regions)
    tasks = map(1:task_count) do task_id
        Threads.@spawn begin
            local_rows = Int[]
            local_cols = Int[]
            local_vals = T[]
            local_rhs = zeros(T, nactive)
            ws = _assembly_workspace(model)
            if task_size_hint > 0
                sizehint!(local_rows, task_size_hint)
                sizehint!(local_cols, task_size_hint)
                sizehint!(local_vals, task_size_hint)
            end

            for region_index in task_id:task_count:n_regions
                region = regions[region_index]
                region_filter === nothing || region_filter(region) || continue
                _assemble_region!(ws, local_rows, local_cols, local_vals, local_rhs, model, region,
                                  blocks, loads, symmetric, state_coefficients,
                                  offsets[region_index])
            end

            return (; rows=local_rows, cols=local_cols, vals=local_vals, rhs=local_rhs)
        end
    end
    results = fetch.(tasks)
    entry_count = sum(result -> length(result.rows), results)
    rows = Int[]
    cols = Int[]
    vals = T[]
    sizehint!(rows, entry_count)
    sizehint!(cols, entry_count)
    sizehint!(vals, entry_count)
    rhs = zeros(T, nactive)

    for result in results
        append!(rows, result.rows)
        append!(cols, result.cols)
        append!(vals, result.vals)
        rhs .+= result.rhs
    end

    return (; rows, cols, vals, rhs)
end

# Partition `blocks` and `loads` by their `on` tag. Returns the
# volume-tagged forms (with `on === nothing`) and a `Dict` of
# `on => (blocks, loads)` grouping every non-volume form by its tag,
# so one assembly pass per unique `on` value handles every
# contribution carrying that tag. Iteration order of the dict is
# insertion order under Julia's Dict (stable per-run), which is
# enough for deterministic threaded-vs-serial behaviour modulo the
# existing roundoff contract.
function _partition_forms_by_on(blocks, loads)
    volume_blocks = filter(b -> b.on === nothing, blocks)
    volume_loads = filter(l -> l.on === nothing, loads)
    partitions = Dict{Any,Tuple{Vector{Any},Vector{Any}}}()
    for b in blocks
        b.on === nothing && continue
        bs, _ls = get!(() -> (Any[], Any[]), partitions, b.on)
        push!(bs, b)
    end
    for l in loads
        l.on === nothing && continue
        _bs, ls = get!(() -> (Any[], Any[]), partitions, l.on)
        push!(ls, l)
    end
    return volume_blocks, volume_loads, partitions
end

# ── Public assembly API ───────────────────────────────────────────────────────

# Normalise a single-form / form-tuple argument to a tuple. Lets the
# user pass `assemble_matrix(model, block)` or
# `assemble_matrix(model, (block₁, block₂))` interchangeably.
_form_tuple(forms::Tuple) = forms
_form_tuple(form) = (form,)

"""
    assemble_matrix(model, block_or_blocks; symmetric=nothing,
                    threaded=Threads.nthreads() > 1, state=nothing) -> SparseMatrixCSC

Assemble one or more bilinear block contributions on the already
prepared `model`, reusing its dof layout, constraints, and integration
plan. `block_or_blocks` is either a single [`BlockForm`](@ref) or a
tuple of them.

  - `symmetric` — assemble symmetrically when truthy. Defaults to
    "every block reports `form.symmetric == true`".
  - `threaded` — drive assembly through `_assemble_system_threaded`.
    Default is true when `Threads.nthreads() > 1`.
  - `state` — pass a [`Solution`](@ref) or active coefficient vector
    to expose the current iterate to the forms as `q.state` (used to
    build Newton tangents).

Returns the sparse global matrix. Does not touch `model.matrix` or
`model.rhs`; for in-place assembly that updates the cached operators
use [`assemble!`](@ref).
"""
function assemble_matrix(model::Model{D,T}, blocks; symmetric=nothing,
                         threaded::Bool=Threads.nthreads() > 1, state=nothing) where {D,T}
    block_tuple = _form_tuple(blocks)
    symmetric_value = symmetric === nothing ? all(block -> block.form.symmetric, block_tuple) :
                      Bool(symmetric)
    nactive = active_unknowns(model.dofs)
    coeffs = _iterate_coefficients(state, model)
    rows, cols, vals, _ = _assemble_partitioned(model, block_tuple, (), nactive, symmetric_value,
                                                nothing, coeffs, threaded)
    return _sparse_from_triplets(rows, cols, vals, nactive, symmetric_value)
end

"""
    assemble_vector(model, load_or_loads; threaded=…, region_filter=nothing,
                    state=nothing) -> Vector

Assemble one or more right-hand-side contributions on the already
prepared `model`, reusing its dof layout, constraints, and integration
plan. `load_or_loads` is either a single [`LoadForm`](@ref) or a tuple
of them.

`region_filter` is an optional predicate on integration regions used by
compactly-supported loads (e.g. a moving point source supported only on
a small portion of the mesh) to skip irrelevant regions cheaply.

`state` exposes the current iterate as `q.state` for Newton-residual
loads (see [`assemble_matrix`](@ref)).

Returns the assembled right-hand-side vector.
"""
function assemble_vector(model::Model{D,T}, loads; threaded::Bool=Threads.nthreads() > 1,
                         region_filter=nothing, state=nothing) where {D,T}
    load_tuple = _form_tuple(loads)
    nactive = active_unknowns(model.dofs)
    coeffs = _iterate_coefficients(state, model)
    _, _, _, rhs = _assemble_partitioned(model, (), load_tuple, nactive, false, region_filter,
                                         coeffs, threaded)
    return rhs
end

"""
    assemble!(model; threaded=Threads.nthreads() > 1) -> Model

Assemble the global matrix and right-hand side for `model`'s problem
in place. Updates `model.matrix`, `model.rhs`, and
`model.diagnostics` (symmetry residual and condition estimate filled
in). Returns `model` for chaining.

The current implementation supports strong Dirichlet elimination
(through the dof layer) and dof-wise homogeneous overlay constraints.
"""
function assemble!(model::Model{D,T}; threaded::Bool=Threads.nthreads() > 1) where {D,T}
    plan = integration_plan(model)
    nactive = active_unknowns(model.dofs)
    symmetric = model.problem.symmetric
    rows, cols, vals, rhs = _assemble_partitioned(model, model.problem.blocks, model.problem.loads,
                                                  nactive, symmetric, nothing, nothing, threaded)

    matrix = _sparse_from_triplets(rows, cols, vals, nactive, symmetric)
    # Symmetric forms are assembled lower-triangular and mirrored as
    # `matrix + matrix' - diag(matrix)` in `_sparse_from_triplets`. IEEE
    # addition is commutative, so the result is symmetric to the bit
    # and the residual is exactly zero.
    symmetry_residual = symmetric ? 0.0 : NaN
    condition_estimate = _condition_estimate(matrix)
    model.integration = plan
    model.matrix = matrix
    model.rhs = rhs
    diag = model.diagnostics
    diag.active_unknowns = nactive
    diag.symmetry_residual = symmetry_residual
    diag.condition_estimate = condition_estimate
    _set_plan_stats!(diag, plan)
    return model
end

# Partition `blocks` and `loads` by their `on` tag, run one assembly
# pass per non-empty partition, and accumulate the COO triplets and
# rhs entries into shared buffers. Volume contributions (`on === nothing`)
# walk the model's `integration_plan`; facet contributions
# (`on::BoundarySelector`) walk the cached `model.facet_regions[on]`
# region list. Volume size hints come from `_triplet_estimate`; facet
# partitions sizehint after a quick traversal so the COO buffers grow
# at most once.
function _assemble_partitioned(model::Model{D,T}, blocks, loads, nactive::Int, symmetric::Bool,
                               region_filter, state_coefficients, threaded::Bool) where {D,T}
    volume_blocks, volume_loads, partitions = _partition_forms_by_on(blocks, loads)

    rows = Int[]
    cols = Int[]
    vals = T[]
    rhs = zeros(T, nactive)

    # Volume contributions — the dominant path. Reuses the cached
    # integration plan and the triplet-count estimator for COO
    # sizing.
    if !isempty(volume_blocks) || !isempty(volume_loads)
        plan = integration_plan(model)
        size_hint = isempty(volume_blocks) ? 0 : _triplet_estimate(model, plan, symmetric)
        _accumulate_pass!(rows, cols, vals, rhs, model, plan.regions, nactive, symmetric,
                          Tuple(volume_blocks), Tuple(volume_loads), region_filter,
                          state_coefficients, threaded, size_hint)
    end

    # Non-volume contributions — one assembly pass per unique `on=`
    # value. `region_filter` is a volume-only convenience and is not
    # forwarded to the facet / surface passes.
    for (selector, (sel_blocks, sel_loads)) in partitions
        regions = _resolve_on_regions(model, selector)
        isempty(regions) && continue
        _accumulate_pass!(rows, cols, vals, rhs, model, regions, nactive, symmetric,
                          Tuple(sel_blocks), Tuple(sel_loads), nothing, state_coefficients,
                          threaded, 0)
    end

    return rows, cols, vals, rhs
end

# Drive one assembly pass over a single region list (serial or
# threaded based on `threaded`) and accumulate the resulting COO
# triplets and rhs contributions onto the shared buffers. Used by
# `_assemble_partitioned` for both the volume pass and every facet /
# surface partition.
function _accumulate_pass!(rows, cols, vals, rhs, model, regions, nactive, symmetric, blocks, loads,
                           region_filter, state_coefficients, threaded::Bool, size_hint::Int)
    partial = if threaded
        _assemble_system_threaded(model, regions, nactive, symmetric, blocks, loads, region_filter,
                                  state_coefficients, size_hint)
    else
        _assemble_system_serial(model, regions, nactive, symmetric, blocks, loads, region_filter,
                                state_coefficients, size_hint)
    end
    return _append_assembly!(rows, cols, vals, rhs, partial)
end

# Resolve a non-`nothing` `on=` value to its region list. Reads from
# the per-kind cache on `model` when the entry exists (every selector
# referenced by `prepare(problem)` is pre-resolved); falls back to a
# fresh build for one-shot calls like
# `assemble_matrix(model, block_with_unseen_on=…)`.
function _resolve_on_regions(model::Model{D,T}, selector::BoundarySelector) where {D,T}
    return get(() -> _facet_regions_for_selector(model.problem.space, selector,
                                                 model.dofs.tolerance), model.facet_regions,
               selector)
end

function _resolve_on_regions(model::Model{D,T}, mesh::BoundaryMesh{D,T}) where {D,T}
    return get(() -> _surface_regions_for_mesh(model.problem.space, mesh, model.dofs.tolerance),
               model.surface_regions, mesh)
end

"""
    nquadpoints(model::Model; kind::Symbol = :volume) -> Int

Number of quadrature points across all regions of the requested `kind`
on `model`. Use it to size a per-quadrature-point state vector indexed
by `q.point` (see [`foreach_quadrature_point`](@ref)) — e.g. history
variables for inelastic constitutive laws.

`kind` is one of:

  - `:volume` (default) — the assembly quadrature points (the count
    matches the volume `q.point` enumeration used by every existing
    [`foreach_quadrature_point`](@ref) caller).
  - `:facet` — total facet quadrature points across every cached
    [`FacetRegion`](@ref) on the model.

Counters are **per kind**: a `q.point` index inside a volume form is
not interchangeable with a `q.point` index inside a future facet
form. Each form's per-point state array should be sized by
`nquadpoints(model; kind=...)` for its own kind.
"""
function nquadpoints(model::Model; kind::Symbol=:volume)
    kind === :volume && return _quadrature_count(integration_plan(model))
    kind === :facet && return _facet_quadpoint_count(model)
    kind === :surface && return _surface_quadpoint_count(model)
    throw(ArgumentError("nquadpoints kind must be :volume, :facet, or :surface, got $kind"))
end

# Sum of physical-frame Q-points across every cached facet region. The
# Q-point list is precomputed on the `FacetRegion` so the sum is one
# arithmetic op per region. Used by the `:facet` branch of
# `nquadpoints`.
function _facet_quadpoint_count(model::Model)
    total = 0
    for (_, list) in model.facet_regions
        for region in list
            total += length(region.weights)
        end
    end
    return total
end

# Sum of physical-frame Q-points across every cached surface region.
# Used by the `:surface` branch of `nquadpoints`.
function _surface_quadpoint_count(model::Model)
    total = 0
    for (_, list) in model.surface_regions
        for region in list
            total += length(region.weights)
        end
    end
    return total
end

"""
    foreach_quadrature_point(f, model; state=nothing)

Call `f(q)` at every assembly quadrature point of `model`. The
quadrature-point payload is

    q = (; x, weight, point, state)

where

  - `q.x` — physical coordinate of the quadrature point,
  - `q.weight` — `weight × jacobian`, matching what forms see during
    assembly,
  - `q.point` — stable global index in `1:nquadpoints(model)`. The
    indices match the indices forms see during assembly, so `q.point`
    is the natural key for per-point internal state (history
    variables, phase-field damage, plastic strain, …).
  - `q.state` — `nothing` unless a `state` (a [`Solution`](@ref) or
    raw active coefficient vector) was passed, in which case it is a
    [`FormState`](@ref) for `value(q.state, field)` and
    `field_gradient(q.state, field)`.

Iteration order matches the serial assembly path; threaded assembly
sees the same `q.point` indices but visits them in a different order.
"""
function foreach_quadrature_point(f, model::Model{D,T}; state=nothing) where {D,T}
    plan = integration_plan(model)
    coefficients = _iterate_coefficients(state, model)
    offsets = _region_qpoint_offsets(plan.regions)
    ws = _assembly_workspace(model)
    for (region_index, region) in enumerate(plan.regions)
        offset = offsets[region_index]
        jacobian = volume(region.box) / convert(T, 2^D)
        st = coefficients === nothing ? nothing :
             FormState([[_parent_dof_data(ws, layout, parent) for parent in region.parents]
                        for layout in model.dofs.fields], model.dofs, coefficients)
        for (local_qp, (eta, weight)) in
            enumerate(zip(region.quadrature.points, region.quadrature.weights))
            coefficients === nothing || _update_region_basis!(ws, region, eta)
            x = reference_to_physical(region.box, eta)
            f((; x, weight=weight * jacobian, point=offset + local_qp, state=st))
        end
    end
    return nothing
end
