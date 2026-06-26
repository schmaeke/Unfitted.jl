# Coupled-Galerkin assembly kernel for one or more fields over a
# superposition `Space`. This file owns the hot loop and the public
# assembly API; the user-facing types (weak forms, fields, problems)
# live in `problems.jl` and the prepared-problem lifecycle (`Model`,
# `prepare`, `move!`, …) lives in `model.jl`.
#
# The strategy is symbolic + numeric (Gustavson) sparse assembly. A
# symbolic pass builds the CSC sparsity pattern once (cached on the
# `Model`); the numeric pass walks every admissible integration region,
# evaluates basis values and physical gradients per parent level,
# accumulates the user's weak form into a thread-local local matrix /
# rhs, and scatter-adds that block into the prebuilt `nzval`. A dof pair
# appearing in many regions therefore accumulates with `+=` into one slot
# rather than inflating a triplet array.
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
    _tensor_values!(data.basis, data.values, data.local_ids, data.order, xi, data.val1d,
                    data.parent.cell)
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

# ── Symbolic assembly pattern (Gustavson) ─────────────────────────────────────

# The `AssemblyPattern` struct itself lives in `model.jl` (it is cached
# model state, and `model.jl` is included before this file); the builder
# and the numeric scatter that consume it live here.

# Refill `ws.active_dofs` for one region with no quadrature or kernel
# work — runs the same `_local_active_dof_table!` construction the
# numeric path uses, so the symbolic pattern is guaranteed to enumerate
# exactly the dof pairs assembly emits (active branches only; Dirichlet
# and strongly-eliminated raws resolve to global id 0 and never enter
# `active_dofs`). Returns the workspace's `active_dofs` vector, valid
# until the next call.
# (`ws` is an `AssemblyWorkspace`, defined further down this file.)
function _region_active_dofs!(ws, model::Model, parents)
    field_data = [[_parent_dof_data(ws, layout, parent) for parent in parents]
                  for layout in model.dofs.fields]
    _local_active_dof_table!(ws, field_data, model.dofs)
    return ws.active_dofs
end

# Build the CSC sparsity pattern as the union over every region of the
# dense coupling block on that region's `active_dofs` (lower triangle
# when `symmetric`, i.e. global row ≥ col).
#
# `feed!` is a function that, given a `visit` callback, calls
# `visit(active::Vector{Int})` once per dense block to include — one per
# region for volume/facet assembly, one per (region, field) for the L²
# transfer mass. The `active` vector may alias workspace storage; the
# builder copies it.
#
# Gustavson's column dedup uses an `O(nactive)` `marker` array — but the
# trick is correct only when each output column is processed contiguously
# (`marker[row] == col` must persist across all of that column's
# contributions). The blocks are not column-ordered, so we first build a
# dof→blocks inverted index and then sweep one column at a time. The
# index plus the stored per-block active sets cost `O(Σ nactive)` (one
# entry per block/dof incidence) — far below the `O(Σ nactive²)`
# dense-block over-count; the final pattern is the only nnz-sized
# allocation.
#
# Two passes over the columns: the first counts entries per column to
# build `colptr`; the second fills `rowval` and sorts each column's rows
# (CSC requires ascending row indices, which the numeric scatter's
# `searchsortedfirst` also relies on). `marker` need not be reset between
# columns within a pass — the column id is strictly increasing, so a
# stale `marker[row]` from an earlier column never equals the current
# one — but it is cleared between the two passes.
function _gustavson_pattern(feed!, n::Int, symmetric::Bool, key::UInt)
    block_actives = Vector{Vector{Int}}()
    col_blocks = [Int[] for _ in 1:n]
    feed!() do active
        push!(block_actives, copy(active))
        b = length(block_actives)
        @inbounds for j in active
            push!(col_blocks[j], b)
        end
        return nothing
    end

    marker = zeros(Int, n)
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for j in 1:n
        cnt = 0
        for b in col_blocks[j], i in block_actives[b]
            (symmetric && i < j) && continue
            if marker[i] != j
                marker[i] = j
                cnt += 1
            end
        end
        colptr[j + 1] = colptr[j] + cnt
    end

    rowval = Vector{Int}(undef, colptr[n + 1] - 1)
    fill!(marker, 0)
    @inbounds for j in 1:n
        pos = colptr[j]
        for b in col_blocks[j], i in block_actives[b]
            (symmetric && i < j) && continue
            if marker[i] != j
                marker[i] = j
                rowval[pos] = i
                pos += 1
            end
        end
        sort!(view(rowval, colptr[j]:(colptr[j + 1] - 1)))
    end

    return AssemblyPattern(n, colptr, rowval, symmetric, key)
end

# Assembly pattern over a model's volume / facet region lists: feed one
# dense block per region from its (all-field) `active_dofs`.
function build_assembly_pattern(model::Model{D,T}, region_lists, symmetric::Bool,
                                key::UInt) where {D,T}
    ws = _assembly_workspace(model)
    return _gustavson_pattern(active_unknowns(model.dofs), symmetric, key) do visit
        for regions in region_lists, region in regions
            visit(_region_active_dofs!(ws, model, region.parents))
        end
    end
end

# Ordered region lists a matrix assembly over `blocks` walks — the volume
# integration plan (when any block is volume-tagged) followed by each
# unique `on=` selector's region list, in the same order
# `_assemble_partitioned` visits them. Returns the lists plus a hash
# signature of that set, used as the pattern cache key (loads never
# contribute matrix entries, so they are ignored here).
function _assembly_region_lists(model::Model, blocks)
    volume_blocks, _, partitions = _partition_forms_by_on(blocks, ())
    lists = Any[]
    key = hash(:assembly_pattern)
    if !isempty(volume_blocks)
        push!(lists, integration_plan(model).regions)
        key = hash(:volume, key)
    end
    for (selector, _) in partitions
        regions = _resolve_on_regions(model, selector)
        isempty(regions) && continue
        push!(lists, regions)
        key = hash(selector, key)
    end
    return lists, key
end

# Return a CSC pattern matching `blocks` and `symmetric`, reusing the one
# cached on `model` when the region-set key and symmetry agree (so a
# Newton loop's repeated assembly hits the cache and only the numeric
# scatter re-runs). Rebuilds and re-caches otherwise.
function _assembly_pattern!(model::Model, blocks, symmetric::Bool)
    region_lists, key = _assembly_region_lists(model, blocks)
    cached = model.pattern
    if cached !== nothing && cached.symmetric == symmetric && cached.key == key
        return cached
    end
    pattern = build_assembly_pattern(model, region_lists, symmetric, key)
    model.pattern = pattern
    return pattern
end

# Assemble the final sparse matrix from a filled `nzval` buffer and the
# cached pattern. `colptr`/`rowval` are copied so the cached pattern is
# never mutated by `dropzeros!`; `nzval` is taken by reference (safe
# because the caller's `ScatterSink` is single-use). Symmetric forms are
# scattered in the
# lower triangle and mirrored here as `A + Aᵀ − diag`; `dropzeros!`
# collapses the explicit zeros left by Dirichlet column elimination and
# any structurally-present-but-untouched pattern slots.
function _matrix_from_pattern(pattern::AssemblyPattern, nzval::Vector{T}) where {T}
    matrix = SparseMatrixCSC(pattern.n, pattern.n, copy(pattern.colptr), copy(pattern.rowval), nzval)
    pattern.symmetric && (matrix = matrix + matrix' - spdiagm(0 => diag(matrix)))
    dropzeros!(matrix)
    return matrix
end

# ── Assembly workspace and hot loop ───────────────────────────────────────────

# Per-parent, per-region dof distribution table built by
# `_local_parent_dofs!`. Routes each cell-local basis function (and
# component) through the dof layer's `raw_expansion` into two flat
# storage layouts — one for *active* branches (entries that land in
# the local matrix) and one for *Dirichlet* branches (entries that
# move to the rhs via column elimination). The flat layout keeps the
# per-parent allocation count at a fixed `8` (four flat vectors plus
# four small offset/count matrices) regardless of `nbasis × ncomp`,
# instead of the `O(nbasis × ncomp)` per-(i, c) `Vector` allocations a
# `Matrix{Vector{…}}` shape would imply.
#
# For each `(i, c)`, the active branches are stored at
# `active_pos_flat[active_offset[i, c]:(… + active_count[i, c] - 1)]`
# with corresponding weights in `active_w_flat[…]`. Dirichlet branches
# follow the same convention with `dir_*` arrays.
#
# In the integrated-Legendre regime — where every constrained raw is
# either strongly eliminated or physically Dirichlet, and every free
# raw has identity expansion — branch counts are 0 or 1, the flat
# vectors total at most `nbasis × ncomp` entries, and the hot loop
# does a single indexed read per (a, c) (matching today's per-emission
# cost). Pivots with multi-element expansions (`m + 1` for B-spline
# C^m boundaries) widen each pivot's branch count proportionally.
struct LocalDofExpansion{T<:Real}
    active_pos_flat::Vector{Int}
    active_w_flat::Vector{T}
    active_offset::Matrix{Int}
    active_count::Matrix{Int}
    dir_raw_flat::Vector{Int}
    dir_w_flat::Vector{T}
    dir_offset::Matrix{Int}
    dir_count::Matrix{Int}
end

# Build the per-parent dof distribution table. For each `(raw, component)`
# pair, walks the dof layer's `raw_expansion[raw]` list — typically
# `[(raw, 1)]` for free raws, `[]` for strongly eliminated, and the
# resolved pivot expression for linear-constraint pivots — and routes
# each `(other_raw, weight)` entry to either the active branch
# (`other_raw`'s component-active id is nonzero) or the Dirichlet
# branch (`other_raw` is physically Dirichlet on this component). The
# branches are appended to the flat per-table vectors; the
# `active_offset`/`active_count` (and analogous `dir_*`) matrices
# record where each `(i, c)`'s entries live.
#
# The shared `local_by_global` cache ensures that a dof shared between
# two parents in the region gets the same local row/column on both —
# i.e. their stiffness contributions land at the same local matrix
# position before being scattered into the matrix. Used inside
# `_assemble_region!` per region.
function _local_parent_dofs!(active_dofs::Vector{Int}, local_by_global::Dict{Int,Int}, data,
                             layout::FieldLayout{D,T}) where {D,T}
    nbasis = length(data.raw_dofs)
    ncomp = layout.components
    expansion = layout.dofs.raw_expansion
    dirichlet = layout.dofs.physical_dirichlet
    # Pre-size to `nbasis · ncomp` — the count for the common
    # length-1-expansion case (free raws and physical-Dirichlet raws).
    # Pivoted raws (rare) may push past this capacity and trigger
    # one Vector growth at most.
    total = nbasis * ncomp
    active_pos_flat = sizehint!(Int[], total)
    active_w_flat = sizehint!(T[], total)
    active_offset = Matrix{Int}(undef, nbasis, ncomp)
    active_count = Matrix{Int}(undef, nbasis, ncomp)
    dir_raw_flat = Int[]
    dir_w_flat = T[]
    dir_offset = Matrix{Int}(undef, nbasis, ncomp)
    dir_count = Matrix{Int}(undef, nbasis, ncomp)

    for i in 1:nbasis, component in 1:ncomp
        raw = data.raw_dofs[i]
        active_offset[i, component] = length(active_pos_flat) + 1
        dir_offset[i, component] = length(dir_raw_flat) + 1
        a_n = 0
        d_n = 0
        for (other, weight) in expansion[raw]
            global_dof = _field_component_dof(layout, other, component)
            if global_dof == 0
                if dirichlet[other, component]
                    push!(dir_raw_flat, other)
                    push!(dir_w_flat, weight)
                    d_n += 1
                end
            else
                local_pos = get!(local_by_global, global_dof) do
                    push!(active_dofs, global_dof)
                    length(active_dofs)
                end
                push!(active_pos_flat, local_pos)
                push!(active_w_flat, weight)
                a_n += 1
            end
        end
        active_count[i, component] = a_n
        dir_count[i, component] = d_n
    end
    return LocalDofExpansion{T}(active_pos_flat, active_w_flat, active_offset, active_count,
                                dir_raw_flat, dir_w_flat, dir_offset, dir_count)
end

# Lightweight per-parent dof table for the common case — every raw maps
# to a single active local index (or to nothing when it is constrained).
# Used when the layout has no non-trivial linear constraints
# (`!has_linear_constraints`), i.e. the integrated Legendre family and
# the C⁰ B-spline mesh-edge path, where `raw_expansion` is always the
# identity or empty. For each `(raw, component)`:
#
#   * map the raw to its global active id via the field layout;
#   * if it is constrained (global id == 0), record `0`;
#   * otherwise register the global id in `local_by_global` (a fresh
#     local index per *distinct* global id) and record the local index.
#
# Returning a plain `Matrix{Int}` keeps the per-parent allocation at one
# array and lets the matching `_accumulate_qpoint!` overload emit each
# matrix entry with a single indexed read — the pre-constraint-primitive
# cost. The shared `local_by_global` cache ensures a dof shared between
# two parents gets the same local row/column on both.
function _local_parent_dofs_simple!(active_dofs::Vector{Int}, local_by_global::Dict{Int,Int}, data,
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

# ── Scatter sink (matrix destination for one assembly pass) ───────────────────

# Where a region's local matrix block is deposited. The right-hand side
# always accumulates into a dense vector; only the matrix destination
# varies behind the sink, so the three `_assemble_region!` overloads and
# the whole hot loop stay agnostic to it. Two cases:
#
#   * `ScatterSink` — `+=` into the pre-built CSC `nzval` slot located via
#     the cached pattern. The matrix path proper.
#   * `nothing` — the rhs-only sink used by `assemble_vector`, where no
#     block is present and the local matrix is empty.
struct ScatterSink{T}
    nzval::Vector{T}
    pattern::AssemblyPattern
end

# A fresh, empty accumulator with the same destination shape — used by
# the threaded driver to give each task its own buffer before reducing.
_empty_like(::Nothing) = nothing
_empty_like(sink::ScatterSink{T}) where {T} = ScatterSink(zeros(T, length(sink.nzval)), sink.pattern)

# Reduce a per-task sink into the shared one by summing the `nzval`
# accumulators. The threaded path is held to a tolerance, not bit
# identity, so the summation order is unconstrained.
_merge_sink!(::Nothing, ::Nothing) = nothing
function _merge_sink!(dst::ScatterSink, src::ScatterSink)
    dst.nzval .+= src.nzval
    return dst
end

# Raised when the numeric scatter targets a `(row, col)` the prebuilt
# pattern does not contain — impossible unless the symbolic and numeric
# passes disagree about a region's active dofs. Erroring here turns a
# latent silent corruption (writing into an adjacent column's slot) into
# an immediate, locatable failure. Kept `@noinline` so the check stays
# off the hot path's instruction stream.
@noinline function _scatter_pattern_miss(row::Int, col::Int)
    error("assembly scatter: entry ($row, $col) is absent from the cached sparsity pattern; " *
          "the symbolic and numeric passes disagree on the active dofs")
end

# Scatter a region's dense local block into the matrix, column-major.
# Zero entries are skipped (so they never enter the accumulation and
# `dropzeros!` has less to do); the rest locate `(row, col)` in column
# `col`'s sorted row range via `searchsortedfirst` and `+=` into that
# `nzval` slot. A miss (the located slot does not hold `row`) means the
# pattern and the numeric pass disagree and raises rather than corrupting
# a neighbouring slot. The `nothing` sink (rhs-only assembly) is a no-op.
# The serial walk visits regions and entries in a fixed order, so repeated
# serial assembly is deterministic to the bit.
_emit_matrix!(::Nothing, active_dofs, local_matrix) = nothing

function _emit_matrix!(sink::ScatterSink{T}, active_dofs::AbstractVector{Int},
                       local_matrix::AbstractMatrix{T}) where {T}
    isempty(local_matrix) && return nothing
    pattern = sink.pattern
    @inbounds for local_col in axes(local_matrix, 2)
        col = active_dofs[local_col]
        lo = pattern.colptr[col]
        hi = pattern.colptr[col + 1] - 1
        for local_row in axes(local_matrix, 1)
            entry = local_matrix[local_row, local_col]
            iszero(entry) && continue
            row = active_dofs[local_row]
            slot = searchsortedfirst(pattern.rowval, row, lo, hi, Base.Order.Forward)
            (slot <= hi && pattern.rowval[slot] == row) || _scatter_pattern_miss(row, col)
            sink.nzval[slot] += entry
        end
    end
    return nothing
end

# Emit a region's local system: scatter the rhs by row (identical for
# every sink) then deposit the matrix block into the sink.
function _emit_local_system!(sink, rhs::Vector{T}, active_dofs::AbstractVector{Int},
                             local_matrix::AbstractMatrix{T},
                             local_rhs::AbstractVector{T}) where {T}
    for (local_row, row) in pairs(active_dofs)
        rhs[row] += local_rhs[local_row]
    end
    _emit_matrix!(sink, active_dofs, local_matrix)
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

  - `bases[i]`     — basis family of level `i`. Looked up by level id so
    the hot-loop `_fill_factor_tables!` dispatch resolves through one
    abstract-container access per parent per quadrature point. The
    container is `Vector{BasisFamily}` (abstract eltype) because a
    superposition can mix families across levels; the JIT specializes
    the callee per resolved type.
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
    bases::Vector{BasisFamily}
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
    bases = Vector{BasisFamily}(undef, nlev)
    local_ids = Vector{Vector{CartesianIndex{D}}}(undef, nlev)
    orders = Vector{NTuple{D,Int}}(undef, nlev)
    values = Vector{Vector{T}}(undef, nlev)
    gradients = Vector{Vector{SVector{D,T}}}(undef, nlev)
    val1d = Vector{NTuple{D,Vector{T}}}(undef, nlev)
    der1d = Vector{NTuple{D,Vector{T}}}(undef, nlev)
    for level in levels
        i = level.id
        ids = local_basis_indices(level.basis, level.order, level.mode)
        bases[i] = level.basis
        local_ids[i] = ids
        orders[i] = level.order
        values[i] = Vector{T}(undef, length(ids))
        gradients[i] = Vector{SVector{D,T}}(undef, length(ids))
        val1d[i] = _factor_buffers(level.order, T)
        der1d[i] = _factor_buffers(level.order, T)
    end
    return AssemblyWorkspace{D,T}(bases, local_ids, orders, values, gradients, val1d, der1d, Int[],
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
        _tensor_values_grads!(ws.bases[lvl], ws.values[lvl], ws.gradients[lvl], ws.local_ids[lvl],
                              ws.orders[lvl], xi, scale, ws.val1d[lvl], ws.der1d[lvl], parent.cell)
    end
    return nothing
end

# Build the per-field local-to-global dof table for this region against
# the workspace's `active_dofs` / `local_by_global`. Both scratch
# structures are cleared first so the result is per-region; the rebuilt
# table is `local_by_field[field_index][parent_index]`.
#
# Two representations, chosen once per region by the layout's
# `has_linear_constraints` flag:
#
#   * no linear constraints (the common case) → `Matrix{Int}` per parent
#     (single active-or-constrained local index), matched by the
#     lightweight `_accumulate_qpoint!` overload;
#   * linear constraints present → `LocalDofExpansion` per parent, matched
#     by the expansion-distributing overload.
#
# The branch makes the return type a small union; the per-quadrature-point
# `_accumulate_qpoint!` call then resolves through one dispatch per region
# rather than paying the general machinery's indirection on every emission.
function _local_active_dof_table!(ws::AssemblyWorkspace, field_data, layout::SystemLayout)
    empty!(ws.active_dofs)
    empty!(ws.local_by_global)
    if has_linear_constraints(layout)
        return [[_local_parent_dofs!(ws.active_dofs, ws.local_by_global, data, field_layout)
                 for data in field_data[field_index]]
                for (field_index, field_layout) in pairs(layout.fields)]
    else
        return [[_local_parent_dofs_simple!(ws.active_dofs, ws.local_by_global, data, field_layout)
                 for data in field_data[field_index]]
                for (field_index, field_layout) in pairs(layout.fields)]
    end
end

"""
    _assemble_region!(ws, sink, rhs, model, region, blocks, loads,
                      symmetric, state_coefficients=nothing, point_offset=0)

Assemble every weak-form contribution at every quadrature point of one
integration region. This is the assembly hot loop: a single call walks
the region's quadrature points and updates the matrix `sink` and
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
          `row ≥ col` entries; `_matrix_from_pattern` mirrors at the
          end.
  4. **Emit.** Flush the local matrix and rhs to the matrix `sink` and
     global rhs via [`_emit_local_system!`](@ref).

`point_offset` is the global quad-point offset of this region in the
plan, so `q.point = point_offset + local_qp` is the stable index used
by [`foreach_quadrature_point`](@ref) and per-point history data.
"""
function _assemble_region!(ws::AssemblyWorkspace{D,T}, sink, rhs::Vector{T}, model::Model{D,T},
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
        _accumulate_qpoint!(local_matrix, local_rhs, q, qweight, field_data, local_by_field,
                            ws.active_dofs, blocks, loads, symmetric, model, Val(D), T)
    end

    # Scatter the local system into the matrix sink / global rhs.
    _emit_local_system!(sink, rhs, ws.active_dofs, local_matrix, local_rhs)
    return nothing
end

"""
    _assemble_region!(ws, sink, rhs, model, region::FacetRegion,
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
function _assemble_region!(ws::AssemblyWorkspace{D,T}, sink, rhs::Vector{T}, model::Model{D,T},
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
        _accumulate_qpoint!(local_matrix, local_rhs, q, qweight, field_data, local_by_field,
                            ws.active_dofs, blocks, loads, symmetric, model, Val(D), T)
    end

    _emit_local_system!(sink, rhs, ws.active_dofs, local_matrix, local_rhs)
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
        _tensor_values_grads!(ws.bases[lvl], ws.values[lvl], ws.gradients[lvl], ws.local_ids[lvl],
                              ws.orders[lvl], xi, scale, ws.val1d[lvl], ws.der1d[lvl], parent.cell)
    end
    return nothing
end

"""
    _assemble_region!(ws, sink, rhs, model, region::SurfaceRegion,
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
function _assemble_region!(ws::AssemblyWorkspace{D,T}, sink, rhs::Vector{T}, model::Model{D,T},
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
        _accumulate_qpoint!(local_matrix, local_rhs, q, qweight, field_data, local_by_field,
                            ws.active_dofs, blocks, loads, symmetric, model, Val(D), T)
    end

    _emit_local_system!(sink, rhs, ws.active_dofs, local_matrix, local_rhs)
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
# entries; `_matrix_from_pattern` mirrors at the end.
#
# Extracted from the two `_assemble_region!` overloads so the volume
# and facet hot loops share their inner work — every kind-specific
# detail is contained in the caller (Q-point source, basis refresh,
# `q` tuple shape).
#
# Two overloads, selected by the per-region dof-table representation
# built in `_local_active_dof_table!`:
#
#   * `LocalDofExpansion` tables → the general linear-constraint path,
#     distributing each emission through the test and trial expansions
#     (B-spline masks / `continuity_order ≥ 1`);
#   * `Matrix{Int}` tables → the lightweight path for layouts with no
#     non-trivial linear constraints (integrated Legendre and the C⁰
#     B-spline mesh-edge path), where every raw has a single
#     active-or-constrained target.
#
# Both overloads share one outer loop, `_accumulate_qpoint_generic!`;
# only the per-emission kernels (`_emit_load!` / `_emit_block!`) differ,
# and they dispatch statically on the concrete table type. These two thin
# overloads are the per-quadrature-point dispatch barrier: `local_by_field`
# is a small union at the call site, so resolving it here (once per point)
# lets the generic body specialise on the concrete table type and inline
# the right kernel — no per-emission dynamic dispatch.
function _accumulate_qpoint!(local_matrix, local_rhs, q, qweight::T, field_data,
                             local_by_field::AbstractVector{<:AbstractVector{<:LocalDofExpansion}},
                             active_dofs::Vector{Int}, blocks::B, loads::L, symmetric::Bool,
                             model::Model{D,T}, ::Val{D}, ::Type{T}) where {B,L,D,T}
    return _accumulate_qpoint_generic!(local_matrix, local_rhs, q, qweight, field_data,
                                       local_by_field, active_dofs, blocks, loads, symmetric, model,
                                       Val(D), T)
end

function _accumulate_qpoint!(local_matrix, local_rhs, q, qweight::T, field_data,
                             local_by_field::AbstractVector{<:AbstractVector{Matrix{Int}}},
                             active_dofs::Vector{Int}, blocks::B, loads::L, symmetric::Bool,
                             model::Model{D,T}, ::Val{D}, ::Type{T}) where {B,L,D,T}
    return _accumulate_qpoint_generic!(local_matrix, local_rhs, q, qweight, field_data,
                                       local_by_field, active_dofs, blocks, loads, symmetric, model,
                                       Val(D), T)
end

# Shared quadrature-point accumulation: loads (linear channels → rhs)
# then blocks (bilinear channels → matrix, with Dirichlet-column
# elimination → rhs). Symmetric blocks emit only the lower triangle;
# `_matrix_from_pattern` mirrors at the end. Walks the standard
# field × component × parent × dof nest and defers every per-emission
# decision to `_emit_load!` / `_emit_block!`, which the typed overloads
# above specialise to the concrete `local_by_field` table type.
function _accumulate_qpoint_generic!(local_matrix, local_rhs, q, qweight::T, field_data,
                                     local_by_field, active_dofs::Vector{Int}, blocks::B, loads::L,
                                     symmetric::Bool, model::Model{D,T}, ::Val{D},
                                     ::Type{T}) where {B,L,D,T}
    for load in loads
        test_index = _field_index(model.dofs, load.test_name)
        test_layout = model.dofs.fields[test_index]
        test_data = field_data[test_index]
        test_tables = local_by_field[test_index]
        for test_component in 1:test_layout.components
            linear_channels = _linear_channels(load.form, q, test_component, Val(D), T)
            for (data, table) in zip(test_data, test_tables)
                for a in eachindex(data.raw_dofs)
                    contribution = qweight * _test_contribution(linear_channels, data.values[a],
                                                                data.gradients[a])
                    _emit_load!(local_rhs, table, a, test_component, contribution)
                end
            end
        end
    end

    for block in blocks
        test_index = _field_index(model.dofs, block.test_name)
        trial_index = _field_index(model.dofs, block.trial_name)
        test_layout = model.dofs.fields[test_index]
        trial_layout = model.dofs.fields[trial_index]
        for trial_component in 1:trial_layout.components
            for (trial_data, trial_table) in
                zip(field_data[trial_index], local_by_field[trial_index])
                for b in eachindex(trial_data.raw_dofs)
                    trial = TrialChannels(trial_component, trial_data.values[b],
                                          trial_data.gradients[b])
                    # Trial-side data invariant across test dofs, hoisted
                    # once per trial dof: the active-branch offsets for an
                    # expansion table, or the global/local column for a
                    # matrix table.
                    tctx = _trial_block_context(trial_table, trial_data, trial_layout, b,
                                                trial_component)
                    for test_component in 1:test_layout.components
                        channels = _bilinear_channels(block.form, q, trial, test_component)
                        for (test_data, test_table) in
                            zip(field_data[test_index], local_by_field[test_index])
                            for a in eachindex(test_data.raw_dofs)
                                entry = qweight * _test_contribution(channels, test_data.values[a],
                                                                     test_data.gradients[a])
                                _emit_block!(local_matrix, local_rhs, test_table, test_data,
                                             test_layout, tctx, a, test_component, entry, symmetric,
                                             active_dofs)
                            end
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# ── Per-emission kernels (one pair each for the two table types) ──────────────

# Distribute one test dof's linear (load) contribution into the rhs.
# Expansion table: fan the contribution out across the dof's active
# branches. Matrix table: a single active row (skip if constrained).
@inline function _emit_load!(local_rhs, table::LocalDofExpansion, a::Int, c::Int, contribution)
    base = table.active_offset[a, c] - 1
    @inbounds for k in 1:table.active_count[a, c]
        local_rhs[table.active_pos_flat[base + k]] += table.active_w_flat[base + k] * contribution
    end
    return nothing
end

@inline function _emit_load!(local_rhs, table::Matrix{Int}, a::Int, c::Int, contribution)
    local_row = table[a, c]
    local_row == 0 && return nothing
    local_rhs[local_row] += contribution
    return nothing
end

# Per-trial-dof context, hoisted out of the test-dof loop. Expansion
# table: the trial dof's active and Dirichlet branch slices. Matrix
# table: the trial dof's global active id (`col`, 0 if constrained) and
# its local column. Both carry `trial_layout.dofs` (for `constrained_value`)
# and the trial component. Returned as a `NamedTuple` so it inlines with
# no allocation; the two `_emit_block!` kernels read the fields each needs.
@inline function _trial_block_context(trial_table::LocalDofExpansion, trial_data, trial_layout,
                                      b::Int, trc::Int)
    return (base=trial_table.active_offset[b, trc] - 1, n=trial_table.active_count[b, trc],
            dbase=trial_table.dir_offset[b, trc] - 1, dn=trial_table.dir_count[b, trc],
            posflat=trial_table.active_pos_flat, wflat=trial_table.active_w_flat,
            dwflat=trial_table.dir_w_flat, drawflat=trial_table.dir_raw_flat,
            dofs=trial_layout.dofs, trc=trc)
end

@inline function _trial_block_context(trial_table::Matrix{Int}, trial_data, trial_layout, b::Int,
                                      trc::Int)
    return (col=_field_component_dof(trial_layout, trial_data.raw_dofs[b], trc),
            localcol=trial_table[b, trc], raw=trial_data.raw_dofs[b], dofs=trial_layout.dofs,
            trc=trc)
end

# Emit one (test a, trial b) bilinear contribution; `tctx` is the trial
# dof's hoisted context. Expansion table: distribute `entry` through the
# test × trial active branches into the matrix and the trial Dirichlet
# branches into the rhs (column elimination). The symmetric lower-triangle
# skip compares the *global* active ids of the two branches, since a
# pivot's branches can land at different ids.
@inline function _emit_block!(local_matrix, local_rhs, test_table::LocalDofExpansion, test_data,
                              test_layout, tctx, a::Int, tc::Int, entry, symmetric::Bool,
                              active_dofs::Vector{Int})
    test_base = test_table.active_offset[a, tc] - 1
    test_n = test_table.active_count[a, tc]
    @inbounds for kr in 1:test_n
        test_local = test_table.active_pos_flat[test_base + kr]
        tw = test_table.active_w_flat[test_base + kr]
        for kc in 1:tctx.n
            trial_local = tctx.posflat[tctx.base + kc]
            symmetric && active_dofs[test_local] < active_dofs[trial_local] && continue
            local_matrix[test_local, trial_local] += tw * tctx.wflat[tctx.base + kc] * entry
        end
        for kd in 1:tctx.dn
            local_rhs[test_local] -= tw *
                                     tctx.dwflat[tctx.dbase + kd] *
                                     entry *
                                     constrained_value(tctx.dofs, tctx.drawflat[tctx.dbase + kd],
                                                       tctx.trc)
        end
    end
    return nothing
end

# Matrix-table variant: a single (row, col) target. Skip a constrained
# test row; redirect a constrained trial column to the rhs via Dirichlet
# elimination (`constrained_value` is zero for homogeneous overlay
# constraints, so those columns vanish). `local_row == 0` ⇔ the test
# global id is 0, so it doubles as the constrained-test check.
@inline function _emit_block!(local_matrix, local_rhs, test_table::Matrix{Int}, test_data,
                              test_layout, tctx, a::Int, tc::Int, entry, symmetric::Bool,
                              active_dofs::Vector{Int})
    local_row = test_table[a, tc]
    local_row == 0 && return nothing
    row = _field_component_dof(test_layout, test_data.raw_dofs[a], tc)
    symmetric && tctx.col != 0 && row < tctx.col && return nothing
    if tctx.col == 0
        local_rhs[local_row] -= entry * constrained_value(tctx.dofs, tctx.raw, tctx.trc)
    else
        local_matrix[local_row, tctx.localcol] += entry
    end
    return nothing
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

# Serial assembly driver: one workspace and the shared `sink`/`rhs`
# walked across every region in order. `region_filter` lets load
# assemblies skip regions that do not intersect the load's support. The
# `regions` argument is any iterable of `VolumeRegion`s or
# `FacetRegion`s — Julia dispatches the right `_assemble_region!` method
# automatically. The serial walk scatters into the sink in a fixed region
# order, so repeated serial assembly is deterministic to the bit.
function _assemble_system_serial!(sink, rhs::Vector{T}, model::Model{D,T}, regions,
                                  symmetric::Bool, blocks, loads, region_filter,
                                  state_coefficients) where {D,T}
    ws = _assembly_workspace(model)
    offsets = _region_qpoint_offsets(regions)
    for (region_index, region) in enumerate(regions)
        region_filter === nothing || region_filter(region) || continue
        _assemble_region!(ws, sink, rhs, model, region, blocks, loads, symmetric,
                          state_coefficients, offsets[region_index])
    end
    return nothing
end

# Threaded assembly driver: spawn `Threads.nthreads()` tasks, each with
# its own workspace and a fresh per-task sink / rhs (`_empty_like`).
# Distribute regions in a striped pattern
# (`region_index in task_id:task_count:nregions`) so every task
# processes a uniform sample of the list. After all tasks complete,
# reduce the per-task accumulators into the shared sink / rhs by summing
# the `nzval` buffers (`_merge_sink!`). The reduction order is
# deterministic but the floating-point sum order differs from the serial
# walk, so the threaded result matches serial to a tolerance, not
# bit-for-bit (documented in CONTRIBUTING's threading rule).
#
# Memory: each task holds its own full-length `nzval` accumulator, so the
# threaded peak is `(nthreads + 1) × nnz` of scatter buffers versus the
# serial path's `1 × nnz`. This is a bounded multiplier (not the COO
# over-count this assembly replaced), but on memory-bound large-3D runs
# capping the thread count trades parallelism for footprint. A lock-free
# colour-partitioned scatter into one shared `nzval` would restore the
# `1×` peak at much higher complexity and is intentionally not done here.
#
# Region-kind-agnostic: the `regions` iterable can be a volume plan's
# region vector or a facet selector's region list.
function _assemble_system_threaded!(sink, rhs::Vector{T}, model::Model{D,T}, regions,
                                    symmetric::Bool, blocks, loads, region_filter,
                                    state_coefficients) where {D,T}
    task_count = Threads.nthreads()
    offsets = _region_qpoint_offsets(regions)
    n_regions = length(regions)
    nactive = length(rhs)
    tasks = map(1:task_count) do task_id
        Threads.@spawn begin
            local_sink = _empty_like(sink)
            local_rhs = zeros(T, nactive)
            ws = _assembly_workspace(model)
            for region_index in task_id:task_count:n_regions
                region = regions[region_index]
                region_filter === nothing || region_filter(region) || continue
                _assemble_region!(ws, local_sink, local_rhs, model, region, blocks, loads,
                                  symmetric, state_coefficients, offsets[region_index])
            end
            return local_sink, local_rhs
        end
    end
    for (local_sink, local_rhs) in fetch.(tasks)
        _merge_sink!(sink, local_sink)
        rhs .+= local_rhs
    end
    return nothing
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
  - `threaded` — drive assembly through `_assemble_system_threaded!`.
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
    pattern = _assembly_pattern!(model, block_tuple, symmetric_value)
    sink = ScatterSink(zeros(T, length(pattern.rowval)), pattern)
    _assemble_partitioned!(sink, model, block_tuple, (), nactive, symmetric_value, nothing, coeffs,
                           threaded)
    return _matrix_from_pattern(pattern, sink.nzval)
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
    _, rhs = _assemble_partitioned!(nothing, model, (), load_tuple, nactive, false, region_filter,
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
    pattern = _assembly_pattern!(model, model.problem.blocks, symmetric)
    sink = ScatterSink(zeros(T, length(pattern.rowval)), pattern)
    _, rhs = _assemble_partitioned!(sink, model, model.problem.blocks, model.problem.loads, nactive,
                                    symmetric, nothing, nothing, threaded)

    matrix = _matrix_from_pattern(pattern, sink.nzval)
    # Symmetric forms are assembled lower-triangular and mirrored as
    # `matrix + matrix' - diag(matrix)` in `_matrix_from_pattern`. IEEE
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
# pass per non-empty partition, scattering matrix entries into the shared
# `sink` and accumulating rhs entries into the shared rhs. Volume
# contributions (`on === nothing`) walk the model's `integration_plan`;
# facet / surface contributions (`on::BoundarySelector` /
# `on::BoundaryMesh`) walk the cached region list for that selector.
function _assemble_partitioned!(sink, model::Model{D,T}, blocks, loads, nactive::Int,
                                symmetric::Bool, region_filter, state_coefficients,
                                threaded::Bool) where {D,T}
    volume_blocks, volume_loads, partitions = _partition_forms_by_on(blocks, loads)
    rhs = zeros(T, nactive)

    # Volume contributions — the dominant path. Reuses the cached
    # integration plan.
    if !isempty(volume_blocks) || !isempty(volume_loads)
        plan = integration_plan(model)
        _run_pass!(sink, rhs, model, plan.regions, symmetric, Tuple(volume_blocks),
                   Tuple(volume_loads), region_filter, state_coefficients, threaded)
    end

    # Non-volume contributions — one assembly pass per unique `on=`
    # value. `region_filter` is a volume-only convenience and is not
    # forwarded to the facet / surface passes.
    for (selector, (sel_blocks, sel_loads)) in partitions
        regions = _resolve_on_regions(model, selector)
        isempty(regions) && continue
        _run_pass!(sink, rhs, model, regions, symmetric, Tuple(sel_blocks), Tuple(sel_loads),
                   nothing, state_coefficients, threaded)
    end

    return sink, rhs
end

# Drive one assembly pass over a single region list (serial or threaded
# based on `threaded`), scattering matrix entries into the shared `sink`
# and accumulating rhs contributions into the shared `rhs`. Used by
# `_assemble_partitioned!` for both the volume pass and every facet /
# surface partition.
function _run_pass!(sink, rhs, model, regions, symmetric, blocks, loads, region_filter,
                    state_coefficients, threaded::Bool)
    if threaded
        _assemble_system_threaded!(sink, rhs, model, regions, symmetric, blocks, loads,
                                   region_filter, state_coefficients)
    else
        _assemble_system_serial!(sink, rhs, model, regions, symmetric, blocks, loads,
                                 region_filter, state_coefficients)
    end
    return nothing
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
