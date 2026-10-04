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
# accumulates the user's weak form into a per-task local matrix / rhs
# over the region's sorted local dof numbering, and scatter-adds that
# block into the prebuilt `nzval`. A dof pair appearing in many regions
# therefore accumulates with `+=` into one slot rather than inflating a
# triplet array.
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
# projected datum for nonzero Dirichlet). A linear-constraint pivot of a
# basis family (a B-spline mask below maximal continuity) keeps its own
# local row and column while a region is integrated, and is condensed
# onto the active dofs of its expansion once per region.

# ── Field evaluation ──────────────────────────────────────────────────────────

# Per-parent record carrying everything `_field_value`/`_field_gradient`
# need to reconstruct a field at a quadrature point: the parent's basis
# family, order, local-id list, gradient scaling factor, per-axis 1D
# factor buffers, and the slot for the evaluated values and gradients.
# Allocates fresh buffers, so it serves the one-shot evaluation paths of
# `postprocessing.jl`; assembly and the quadrature-point walks evaluate
# into a workspace's `BasisBank` instead.
function _parent_basis_data(V::Space{D,T}, layout::DofLayout{D,T}, parent::ParentRef{D,T};
                            gradients::Bool=true) where {D,T}
    level = _level_by_id(V, parent.level)
    # The 1D factor buffers are sized at the level's nominal order; the local id
    # list is the parent cell's own minimum-rule set, which is shorter wherever a
    # per-cell order puts that cell below the nominal maximum.
    order = nominal_order(level)
    local_ids = cell_basis_indices(level, parent.cell)
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
# evaluation paths in `postprocessing.jl` (`l2_error`, `write_vtk`).
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

# ── Symbolic assembly pattern (Gustavson) ─────────────────────────────────────

# The `AssemblyPattern` struct itself lives in `model.jl` (it is cached
# model state, and `model.jl` is included before this file); the builder
# and the numeric scatter that consume it live here.

# Fill `ws.dofs` with one region's sorted active global dof ids, with no
# quadrature or kernel work. It runs the same `_frame!` and `_slots!` the
# numeric pass runs, so the symbolic pattern enumerates exactly the dof pairs
# assembly emits: active ids and the active branches of every pivot, never a
# constrained id. Returns `ws.dofs`, valid until the workspace's next region.
# (`ws` is an `AssemblyWorkspace`, defined further down this file.)
function _region_active_dofs!(ws, model::Model, region)
    _slots!(_frame!(ws, region))
    return ws.dofs
end

# Build the CSC sparsity pattern as the union over every region of the
# dense coupling block on that region's `active_dofs` (lower triangle
# when `symmetric`, i.e. global row ≥ col).
#
# `feed!` is a function that, given a `visit` callback, calls
# `visit(active::Vector{Int})` once per dense block to include. The sole
# caller is `build_assembly_pattern`, which visits one block per region of
# every pass a matrix assembly walks — the volume plans plus each `on=`
# region list. The `active` vector may alias workspace storage; the builder
# copies it.
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
        sort!(view(rowval, colptr[j]:(colptr[j+1]-1)))
    end

    return AssemblyPattern(n, colptr, rowval, symmetric, key)
end

# Ordered volume region lists for an assembly: each distinct subdomain
# integration plan's `regions` (one list for single-domain). Ownership is
# intrinsic to each region + field layout (`region_parents`), so no per-pass
# `served` mask is needed — a field only receives parents on the levels its
# subdomain owns. Object identity of each `plan.regions` is preserved so the
# `objectid`-keyed gather / arena caches stay valid. Shared by the numeric
# volume assembly (`_assemble_partitioned!`) and the symbolic pattern build so
# both enumerate the same regions in the same order.
_volume_passes(model::Model) = Any[plan.regions for plan in integration_plans(model)]

# Assembly pattern over a model's volume/on region-list passes: feed one dense
# block per region from the active dofs its owning fields emit there.
function build_assembly_pattern(model::Model{D,T}, passes, symmetric::Bool, key::UInt) where {D,T}
    ws = _assembly_workspace(model)
    return _gustavson_pattern(active_unknowns(model.dofs), symmetric, key) do visit
        for regions in passes, region in regions
            visit(_region_active_dofs!(ws, model, region))
        end
    end
end

# Ordered region lists a matrix assembly over `blocks` walks — the volume
# integration plan (when any block is volume-tagged) followed by each
# unique `on=` selector's region list. Returns the lists plus a hash
# signature of that set, used as the pattern cache key (loads never
# contribute matrix entries, so they are ignored here). The visiting order
# is `_assemble_partitioned!`'s.
function _assembly_region_lists(model::Model, blocks)
    volume_blocks, _, partitions = _partition_forms_by_on(blocks, ())
    passes = Any[]
    key = hash(:assembly_pattern)
    if !isempty(volume_blocks)
        for pass in _volume_passes(model)
            push!(passes, pass)
        end
        key = hash(:volume, key)
    end
    for (on, (on_blocks, _)) in partitions
        for (space, _, _) in _partition_space(model, on, on_blocks, ())
            regions = _resolve_on_regions(model, on, space)
            isempty(regions) && continue
            push!(passes, regions)
            key = hash((on, space), key)
        end
    end
    return passes, key
end

# Return a CSC pattern matching `blocks` and `symmetric`, reusing the one
# cached on `model` when the region-set key and symmetry agree (so a
# Newton loop's repeated assembly hits the cache and only the numeric
# scatter re-runs). Rebuilds and re-caches otherwise.
function _assembly_pattern!(model::Model, blocks, symmetric::Bool)
    passes, key = _assembly_region_lists(model, blocks)
    cached = model.pattern
    if cached !== nothing && cached.symmetric == symmetric && cached.key == key
        return cached
    end
    pattern = build_assembly_pattern(model, passes, symmetric, key)
    model.pattern = pattern
    return pattern
end

# Mirror a symmetric form's lower-triangular CSC arrays — `rowval` holds
# only rows `i ≥ j`, which is what `_gustavson_pattern` emits under
# `symmetric` — into the full symmetric matrix, in one pass.
#
# The obvious spelling, `A + Aᵀ − diag(A)`, materialises five full-size
# temporaries: the transpose, the sum, the dense diagonal, the sparse
# diagonal, and the difference. That is invisible in a wall-clock profile
# but dominates the allocation of a Newton or transient loop reassembling
# the same pattern over and over. The pass below allocates only what it
# returns.
#
# Column `j` of the result holds, in this order, the mirrors of the
# strictly-lower entries of row `j` (one per entry `(j, j′)` of `A` with
# `j′ < j`) and then column `j`'s own entries, rows `i ≥ j`, which `A`
# already stores ascending. Sweeping the source columns in increasing `j`
# and appending each entry `(i, j)` to column `j` and — when `i > j` —
# its mirror to column `i` produces exactly that order with no sort:
# every mirror written into column `i` comes from a source column `j < i`,
# so all of them land before the sweep reaches column `i`'s own entries,
# and they arrive with strictly increasing row index `j`.
#
# Values are copied, never combined, so the result matches the
# `A + Aᵀ − diag(A)` spelling to the bit: an off-diagonal entry is a single
# term either way, and the diagonal's `2d − d` is exact in IEEE arithmetic.
# The two differ only in which numerically-zero slots they store — the sum
# also materialises every diagonal slot and both triangles' explicit zeros —
# and `_matrix_from_pattern` runs `dropzeros!` over either, which removes
# exactly those.
function _mirror_lower(n::Int, colptr::Vector{Int}, rowval::Vector{Int}, nzval::Vector{T}) where {T}
    # Count first: every source entry occupies a slot in its own column, and a
    # strictly-lower one occupies a second slot in the column it mirrors into.
    # Counts land in `full_colptr[j + 1]`; the prefix sum turns them into offsets.
    full_colptr = zeros(Int, n + 1)
    @inbounds for j in 1:n, k in colptr[j]:(colptr[j+1]-1)
        full_colptr[j + 1] += 1
        rowval[k] > j && (full_colptr[rowval[k] + 1] += 1)
    end
    full_colptr[1] = 1
    @inbounds for j in 1:n
        full_colptr[j + 1] += full_colptr[j]
    end

    pos = full_colptr[1:n]  # per-column write cursor
    full_rowval = Vector{Int}(undef, full_colptr[n + 1] - 1)
    full_nzval = Vector{T}(undef, length(full_rowval))
    @inbounds for j in 1:n, k in colptr[j]:(colptr[j+1]-1)
        i = rowval[k]
        v = nzval[k]
        full_rowval[pos[j]] = i
        full_nzval[pos[j]] = v
        pos[j] += 1
        if i > j
            full_rowval[pos[i]] = j
            full_nzval[pos[i]] = v
            pos[i] += 1
        end
    end
    return SparseMatrixCSC(n, n, full_colptr, full_rowval, full_nzval)
end

# Assemble the final sparse matrix from a filled `nzval` buffer and the
# cached pattern. An unsymmetric form copies `colptr`/`rowval` so the
# cached pattern is never mutated by `dropzeros!` and takes `nzval` by
# reference (safe because the caller's `ScatterSink` is single-use); a
# symmetric one was scattered in the lower triangle only, and
# `_mirror_lower` builds fresh arrays for the full matrix. `dropzeros!`
# then collapses the explicit zeros left by Dirichlet column elimination
# and any structurally-present-but-untouched pattern slots.
function _matrix_from_pattern(pattern::AssemblyPattern, nzval::Vector{T}) where {T}
    matrix = pattern.symmetric ? _mirror_lower(pattern.n, pattern.colptr, pattern.rowval, nzval) :
             SparseMatrixCSC(pattern.n, pattern.n, copy(pattern.colptr), copy(pattern.rowval),
                             nzval)
    dropzeros!(matrix)
    return matrix
end

# ── Scatter sink (matrix destination for one assembly pass) ───────────────────

# Where a region's local matrix block is deposited. The right-hand side
# always accumulates into a dense vector; only the matrix destination
# varies behind the sink, so the single `_assemble_region!` body and the
# whole hot loop stay agnostic to it. Three cases:
#
#   * `ScatterSink` — `+=` into the pre-built CSC `nzval` slot located via
#     the cached pattern. The serial matrix path proper.
#   * `ArenaSink` (defined with the threaded scatter further down) — store
#     the block into the region's own disjoint arena slice, deferring the
#     accumulation to the phase-2 gather.
#   * `nothing` — the rhs-only sink used by `assemble_vector`, where no
#     block is present and the local matrix is empty.
struct ScatterSink{T}
    nzval::Vector{T}
    pattern::AssemblyPattern
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
# Zero entries are skipped; the rest locate `(row, col)` in column `col`'s
# sorted row range via `searchsortedfirst` and `+=` into that `nzval` slot.
#
# The skip is load-bearing, not just a saving. Under `symmetric` the pattern
# stores rows `i ≥ j` only, and the kernel never writes an upper-triangle slot
# of the local block — so those entries are still exactly the zeros
# `_integrate!` filled, and skipping them is what keeps the lookup below inside
# the column's stored row range. (It also keeps
# structural zeros out of the accumulation, giving `dropzeros!` less to do.)
#
# A miss — the located slot does not hold `row` — means the pattern and the
# numeric pass disagree, and raises rather than corrupting a neighbouring
# slot. The `nothing` sink (rhs-only assembly) is a no-op.
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

# One-shot per-parent record collector. Used by the evaluation paths of
# `postprocessing.jl`, which allocate fresh basis buffers; assembly and the
# quadrature-point walks evaluate into a workspace's `BasisBank` instead.
function _field_parent_data(V, layout::FieldLayout, parents; gradients::Bool=true)
    [_parent_basis_data(V, layout.dofs, parent; gradients) for parent in parents]
end

# ── Basis bank and assembly workspace ─────────────────────────────────────────

# Per global level id, everything the kernel needs to evaluate that level's basis
# at a point: the family, the per-cell index sets, the nominal order, and the
# value, gradient and 1D factor buffers the evaluation writes into. The buffers
# are indexed by level id, which `prepare` made contiguous and disjoint across
# subdomains, and reused across every region a task processes; a region carries
# at most one parent per level, so they never collide within a region.
#
# Struct of arrays on purpose. Only `bases` may be abstract — on a space that
# mixes families, or B-splines of different degrees — and such a space then pays
# one dynamic basis call per parent per point, which it must. Every value and
# gradient read in the emission loop stays concrete either way. A per-level
# record bundling the same buffers would itself be abstract on a mixed space and
# make every one of those reads dynamic: measured 4.5× slower on a mixed-family
# overlay.
#
#   - `bases[i]` is narrowed by `map(identity, …)` to the eltype the space's
#     families share, `IntegratedLegendre` on the default space. An abstract
#     bank turns the per-point basis call into a dynamic dispatch that boxes its
#     `SVector` / `NTuple` / `CartesianIndex` arguments: measured at 448 B per
#     call, and 94 % of `assemble_matrix`'s allocations.
#   - `modes[i]` is the level's own minimum-rule table (`level.modes`), read by
#     parent cell as `sets[kind[cell]]`. It is the same object
#     `_build_cell_dofs!` walked to produce `cell_dofs`, taken off the level
#     rather than re-derived, because the positional pairing between a cell's
#     raw dofs and its basis values is the only link between the dof layer and
#     the basis layer, and nothing checks it.
#   - `values[i]` and `gradients[i]` are sized at the level's nominal (maximum)
#     order, so a cell at a lower per-cell order fills a prefix. See
#     `_tensor_values_grads!` in `basis.jl` for why that is transparent and why
#     the buffers are not keyed by order value.
struct BasisBank{D,T,B<:BasisFamily}
    bases::Vector{B}
    modes::Vector{CellModes{D}}
    orders::Vector{NTuple{D,Int}}
    values::Vector{Vector{T}}
    gradients::Vector{Vector{SVector{D,T}}}
    val1d::Vector{NTuple{D,Vector{T}}}
    der1d::Vector{NTuple{D,Vector{T}}}
end

# Build the bank over a problem's subdomain spaces without a type-unstable level
# concatenation. `prepare` reindexed every subdomain space into a disjoint,
# contiguous block of global level ids, so the union spans `1:N` with
# `N = Σ level_count(V)`. The buffers are allocated once at size `N` and filled
# per space through `_fill_bank!`; the only dynamic dispatch is the loop over
# the abstractly typed `spaces`, once per subdomain.
function BasisBank(spaces, ::Val{D}, ::Type{T}) where {D,T}
    n = 0
    for V in spaces
        n += level_count(V)
    end
    bases = Vector{BasisFamily}(undef, n)
    modes = Vector{CellModes{D}}(undef, n)
    orders = Vector{NTuple{D,Int}}(undef, n)
    values = Vector{Vector{T}}(undef, n)
    gradients = Vector{Vector{SVector{D,T}}}(undef, n)
    val1d = Vector{NTuple{D,Vector{T}}}(undef, n)
    der1d = Vector{NTuple{D,Vector{T}}}(undef, n)
    for V in spaces
        _fill_bank!(bases, modes, orders, values, gradients, val1d, der1d, V.levels, T)
    end
    narrow = map(identity, bases)
    return BasisBank{D,T,eltype(narrow)}(narrow, modes, orders, values, gradients, val1d, der1d)
end

# Function barrier: fill the id-indexed bank from one space's concretely typed
# level `Tuple`, so every per-level field read and the `local_basis_indices` /
# `_factor_buffers` calls dispatch statically.
function _fill_bank!(bases, modes, orders, values, gradients, val1d, der1d, levels::Tuple,
                     ::Type{T}) where {T}
    for level in levels
        i, p = level.id, nominal_order(level)
        nbasis = length(local_basis_indices(level.basis, p, level.mode))
        bases[i] = level.basis
        modes[i] = _cell_locals(level)
        orders[i] = p
        values[i] = Vector{T}(undef, nbasis)
        gradients[i] = Vector{eltype(eltype(gradients))}(undef, nbasis)
        val1d[i] = _factor_buffers(p, T)
        der1d[i] = _factor_buffers(p, T)
    end
    return nothing
end

"""
    AssemblyWorkspace{D,T,B}

Per-task assembly scratch, for one integration region at a time. A workspace
belongs to one task for the duration of a call and is never shared; every
per-region field is reset before it is reused.

  - `layout` — the model's [`SystemLayout`](@ref). The kernel reads the dof
    layer through it and never sees `Model{D,T,P}`, so it compiles once per
    workspace type, not once per problem type.
  - `bank` — the basis buffers by level id (`BasisBank`).
  - `parents`, `fptr` — the region frame `_frame!` builds: one
    `(level, slot offset, raw dofs)` record per (field, covering parent),
    field-major, so field `f` owns `parents[fptr[f]:fptr[f+1]-1]`. The raw dof
    vector is the layout's own `cell_dofs` entry, aliased, never copied.
  - `slots` — the slot table. Slot `s₀ + (c − 1)·length(raw) + i` of a parent
    record (component `c`, cell-local basis function `i`) holds the local index
    `1:n` of an active dof, `0` for a constrained one (Dirichlet or strongly
    eliminated), or `n + k` for the `k`-th linear-constraint pivot.
  - `dofs` — the global active ids of local indices `1:n`, ascending.
  - `pivots` — the `(field, raw, component)` of pivot slot `n + k`.
  - `K`, `b` — the local system, `m × m` column-major and length `m`, with
    `m = n + length(pivots)`. A pivot's row and column are condensed onto its
    branches before the system leaves the workspace.
"""
struct AssemblyWorkspace{D,T,B<:BasisFamily}
    layout::SystemLayout{D,T}
    bank::BasisBank{D,T,B}
    parents::Vector{Tuple{Int,Int,Vector{Int}}}
    fptr::Vector{Int}
    slots::Vector{Int}
    dofs::Vector{Int}
    pivots::Vector{NTuple{3,Int}}
    K::Vector{T}
    b::Vector{T}
end

# Build a workspace from the dof layout and the subdomain spaces rather than
# from the `Model`, so that nothing in it is specialised on the problem type.
function AssemblyWorkspace(layout::SystemLayout{D,T}, spaces) where {D,T}
    return AssemblyWorkspace(layout, BasisBank(spaces, Val(D), T), Tuple{Int,Int,Vector{Int}}[],
                             zeros(Int, length(layout.fields) + 1), Int[], Int[], NTuple{3,Int}[],
                             T[], T[])
end

# A fresh workspace for `model`. Deliberately unspecialised: it runs once per
# pass or per task, never per region, and specialising it on `Model{D,T,P}`
# would only compile it again for every problem type.
function _assembly_workspace(@nospecialize(model::Model))
    return AssemblyWorkspace(model.dofs, problem_spaces(model.problem))
end

# ── Region frame and slot table ───────────────────────────────────────────────

# Parents of `region` on which field `fl` (global index `f`) is evaluated, or
# `nothing` when the field does not live there. Subdomain ownership is
# intrinsic: a single-space region hands its parents only to a field whose
# level-id block contains the region's level, so a region built on one
# subdomain's plan is owned exactly by the fields on that subdomain. On a
# single-domain model every field's block covers every level and every field
# gets the parents. `nothing` rather than an empty vector keeps the frame
# allocation-free.
function region_parents(region, ::Int, fl::FieldLayout)
    (isempty(region.parents) || first(region.parents).level in fl.level_ids) ? region.parents :
    nothing
end

# Two-sided: each coupled field is evaluated on its own subdomain's parents and
# every other field on none, so an interface pass emits into exactly the four
# field blocks Kₐₐ, K_ab, K_ba, K_bb. Keyed on the field *index* (the interface
# stores its two sides' global field indices), not on the level block.
function region_parents(region::InterfaceRegion, f::Int, ::FieldLayout)
    f == region.field_a ? region.parents_a : f == region.field_b ? region.parents_b : nothing
end

# Build the region frame: for every field and every parent the field owns in
# `region`, record `(level, slot offset, raw dofs)` and push one provisional
# slot per (component, basis function), component-major per parent, holding the
# global active id (0 when constrained). This, with `_slots!`, is the one place
# a region's dof list is derived: the symbolic pattern, the threaded arena
# layout and the numeric pass all go through it, so they cannot disagree.
#
# The guard compares a cell's raw dof list with the bank, which is sized once
# from the level's nominal order — the `>=` relaxation `_tensor_values_grads!`
# relies on. `_build_cell_dofs!` and the bank both read `level.modes`, so the
# list and the cell's own index set cannot disagree in length; what can fail is
# a cell needing more basis values than its level's bank holds. One integer
# compare per (region, parent, field), outside the quadrature loop.
function _frame!(ws::AssemblyWorkspace, region)
    empty!(ws.parents)
    empty!(ws.slots)
    for (f, fl) in pairs(ws.layout.fields)
        ws.fptr[f] = length(ws.parents) + 1
        parents = region_parents(region, f, fl)
        parents === nothing && continue
        for p in parents
            raw = cell_dofs(fl.dofs, p.level, p.cell)
            length(raw) <= length(ws.bank.values[p.level]) ||
                throw(DimensionMismatch("cell $(p.cell) of level $(p.level) has $(length(raw)) raw " *
                                        "dofs but the workspace bank holds " *
                                        "$(length(ws.bank.values[p.level])) basis values"))
            push!(ws.parents, (p.level, length(ws.slots), raw))
            for c in 1:fl.components, r in raw
                push!(ws.slots, _field_component_dof(fl, r, c))
            end
        end
    end
    ws.fptr[end] = length(ws.parents) + 1
    return ws
end

# Number the region's dofs locally and give every slot its role. Returns `n`,
# the number of active local indices.
#
#   1. Collect the active global ids of the frame.
#   2. On a field with linear constraints, turn every constrained slot whose
#      (raw, component) is a pivot into pivot slot `k`, and collect the active
#      ids of its branches: a pivot widens the region's active set to its
#      expansion. The rule is the dof layer's `_fold_pivot`, the same one
#      `dof_value` reconstructs a pivot from.
#   3. Sort and deduplicate the ids. Every global id then has exactly one local
#      index, and local order is global order. The second property is what lets
#      the kernel apply a symmetric form's "global row ≥ col" test to local
#      indices before it contracts, and the scatter merge a region column into a
#      sorted pattern column.
#   4. Rewrite each slot: active id `g` ↦ its local index, `0` ↦ `0`, pivot
#      `k` ↦ `n + k`.
#
# The renumbering is bit-neutral: every local sum still receives the same terms
# in the same order. `QuickSort` sorts in place; the default algorithm allocates
# a radix scratch buffer beyond about 40 entries, which is once per region in
# 3D.
function _slots!(ws::AssemblyWorkspace)
    empty!(ws.dofs)
    empty!(ws.pivots)
    for g in ws.slots
        g > 0 && push!(ws.dofs, g)
    end
    for (f, fl) in pairs(ws.layout.fields)
        fl.dofs.has_linear_constraints || continue
        id = (o, c) -> _field_component_dof(fl, o, c)
        for p in ws.fptr[f]:(ws.fptr[f+1]-1)
            _, s0, raw = ws.parents[p]
            for c in 1:fl.components, i in eachindex(raw)
                k = s0 + (c - 1) * length(raw) + i
                ws.slots[k] == 0 || continue
                _, pivot = _fold_pivot(_push_active_branch, ws.dofs, fl.dofs, id, raw[i], c)
                pivot || continue
                push!(ws.pivots, (f, raw[i], c))
                ws.slots[k] = -length(ws.pivots)
            end
        end
    end
    unique!(sort!(ws.dofs; alg=QuickSort))
    n = length(ws.dofs)
    for k in eachindex(ws.slots)
        g = ws.slots[k]
        ws.slots[k] = g > 0 ? searchsortedfirst(ws.dofs, g) : g < 0 ? n - g : 0
    end
    return n
end

# The `_fold_pivot` step of `_slots!`: collect a branch's active id.
_push_active_branch(dofs, g, _, _) = (g > 0 && push!(dofs, g); dofs)

# ── Per-region-kind quadrature accessors ──────────────────────────────────────
#
# Volume, facet, surface and interface regions share one kernel, `_integrate!`
# below. Only these small accessors differ per region kind, and they dispatch
# statically on the region type.

# Reference→physical Jacobian for the region's weights: `vol(box)/2ᴰ` for a
# volume region (whose weights are reference-frame), `one(T)` for facet, surface,
# and interface regions (whose weights are already physical-frame).
_region_jacobian(region::VolumeRegion{D,T}) where {D,T} = volume(region.box) / convert(T, 2^D)
function _region_jacobian(::Union{FacetRegion{D,T},SurfaceRegion{D,T},InterfaceRegion{D,T}}) where {D,
                                                                                                    T}
    one(T)
end

# Quadrature point `k` of `region` as `(x, weight, ref)`: the physical point, the
# physical weight, and the reference each parent maps to its own frame
# (`_parent_xi`). A volume region iterates reference points `η` of its box, so
# `ref = η` and every parent maps it through its `local_box`. Facet, surface and
# interface regions carry physical points and weights, and each parent maps the
# physical point through its own `parent_box`: a facet region has no single
# reference frame shared by every parent, since its free-axis coordinates run
# across several cells per level.
@inline function _region_point(region::VolumeRegion, k::Int, jacobian)
    η = region.quadrature.points[k]
    return reference_to_physical(region.box, η), region.quadrature.weights[k] * jacobian, η
end
@inline function _region_point(region::Union{FacetRegion,SurfaceRegion,InterfaceRegion}, k::Int, _)
    x = region.points[k]
    return x, region.weights[k], x
end

_parent_xi(p::ParentRef, η) = reference_to_physical(p.local_box, η)
_parent_xi(p::FacetParent, x) = physical_to_reference(p.parent_box, x)

# The parent lists refreshed at every point: a region's one list, or both sides
# of an interface, whose parents live in the two subdomains' disjoint level
# banks, so the kernel reads field `a` from `a`'s cut cell and field `b` from
# `b`'s at the same physical point.
_parent_lists(region) = (region.parents,)
_parent_lists(region::InterfaceRegion) = (region.parents_a, region.parents_b)

# `q`-tuple extras the form callbacks read: the outward unit normal at the
# point (`nothing` for a volume region; the constant face normal for a facet;
# the per-point normal for an immersed surface or an interface, oriented from
# side `a` to side `b` there) and the codim-`K` facet identifier `q.sides` (only
# a facet carries one).
_region_normal(::VolumeRegion, ::Int) = nothing
_region_normal(region::FacetRegion, ::Int) = region.normal
function _region_normal(region::Union{SurfaceRegion,InterfaceRegion}, local_qp::Int)
    region.normals[local_qp]
end
_region_sides(::Union{VolumeRegion,SurfaceRegion,InterfaceRegion}) = nothing
_region_sides(region::FacetRegion) = region.sides

# Evaluate every parent's basis at the current point into the bank: values and
# physical gradients (`Val(true)`), or values only (`Val(false)`, for the L²
# transfer rhs, whose integrand never reads a gradient). The gradient scale is
# the axis-aligned chain rule `2 / edge_lengths(parent_box)` of
# `physical_basis_gradients` in `basis.jl`. One basis evaluation per parent per
# point; a region's parents have distinct levels, so their bank slots never
# collide.
@noinline function _refresh!(ws::AssemblyWorkspace{D,T}, region, ref, ::Val{G}) where {D,T,G}
    bank = ws.bank
    for parents in _parent_lists(region), p in parents
        l = p.level
        ids = bank.modes[l].sets[bank.modes[l].kind[p.cell]]
        xi = _parent_xi(p, ref)
        if G
            scale = SVector{D,T}(2 .* inv.(edge_lengths(p.parent_box)))
            _tensor_values_grads!(bank.bases[l], bank.values[l], bank.gradients[l], ids,
                                  bank.orders[l], xi, scale, bank.val1d[l], bank.der1d[l], p.cell)
        else
            _tensor_values!(bank.bases[l], bank.values[l], ids, bank.orders[l], xi, bank.val1d[l],
                            p.cell)
        end
    end
    return nothing
end

# ── FormState ─────────────────────────────────────────────────────────────────

"""
    FormState

The current solution iterate at a quadrature point, exposed to weak-form
callbacks as `q.state` when assembling with `state=`. Query it with
`value(q.state, field[, component])` and `field_gradient(q.state, field[,
component])`, where `field` is a field name `Symbol` or a
[`Field`](@ref).

Evaluation is eager. Every field's value and physical gradient, per
component, is computed once per quadrature point from dof values read once
per integration region, so a read is an index into a buffer and a callback
can read `q.state` as often as it likes. The values are those of the current
point and are valid during the callback only: the same object is refilled at
the next point, so keep the numbers, not the `FormState`.

The values carry the iterate's number type promoted with the model's: a
`BigFloat` iterate on a `Float64` model yields `BigFloat` values and
gradients. A component outside the field's range raises a `BoundsError`, and
an unknown field name an `ArgumentError`.

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
struct FormState{D,R,L<:SystemLayout}
    # The model's layout and the iterate, converted once per call to
    # `Vector{R}`, `R = promote_type(T, eltype(state))`; shared read-only.
    layout::L
    coefficients::Vector{R}
    # `dof_value` of every slot of the current region (`_state_region!`).
    slot_values::Vector{R}
    # Field `f`, component `c` at the current point lives at `offsets[f] + c`
    # (`_state_point!`).
    values::Vector{R}
    gradients::Vector{SVector{D,R}}
    offsets::Vector{Int}
end

# A state with private per-point buffers over the shared coefficient vector:
# one per task, since the buffers are rewritten at every point.
function FormState(layout::SystemLayout{D}, coefficients::Vector{R}) where {D,R}
    offsets = zeros(Int, length(layout.fields) + 1)
    for (f, fl) in pairs(layout.fields)
        offsets[f+1] = offsets[f] + fl.components
    end
    return FormState{D,R,typeof(layout)}(layout, coefficients, R[], zeros(R, offsets[end]),
                                         zeros(SVector{D,R}, offsets[end]), offsets)
end

# Buffer index of field `name`, component `component`, checking both.
function _state_index(state::FormState, name::Symbol, component::Integer)
    f = get(state.layout.by_name, name, 0)
    f == 0 && throw(ArgumentError("unknown field $name"))
    1 <= component <= state.layout.fields[f].components ||
        throw(BoundsError(state, (name, component)))
    return state.offsets[f] + component
end

"""
    value(state::FormState, name_or_field[, component=1])

Read the named field's value at the current quadrature point. `name` is
either a `Symbol` matching a field name in the model or a [`Field`](@ref)
object. For multi-component fields, pass `component`.
"""
function value(state::FormState, name::Symbol, component::Integer=1)
    return state.values[_state_index(state, name, component)]
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
    return state.gradients[_state_index(state, name, component)]
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

# Read the dof value of every slot of the region `ws` was framed on, once per
# region: `dof_value` resolves a constrained slot to its stored value and a
# pivot to its expansion, so the per-point sums below need no branch. This
# replaces a `dof_value` call per dof per read, which made a Newton tangent
# reading `q.state` cost O(n²) state work per point instead of O(n).
function _state_region!(state::FormState, ws::AssemblyWorkspace)
    resize!(state.slot_values, length(ws.slots))
    for (f, fl) in pairs(ws.layout.fields), p in ws.fptr[f]:(ws.fptr[f+1]-1)
        _, s0, raw = ws.parents[p]
        for c in 1:fl.components, i in eachindex(raw)
            state.slot_values[s0+(c-1)*length(raw)+i] = dof_value(fl, state.coefficients, raw[i], c)
        end
    end
    return state
end

# Evaluate every field's components at the current point from the refreshed
# bank: per parent the partial sum `Σᵢ uᵢ φᵢ` (and `Σᵢ uᵢ ∇φᵢ` when `G`), then the
# sum over parents. That is the association the field-evaluation helpers above
# use, so a state reads the same bits as `value(solution, model, x)`'s
# reconstruction of the same sum. Returns `state`, which becomes `q.state`.
function _state_point!(state::FormState{D,R}, ws::AssemblyWorkspace, ::Val{G}) where {D,R,G}
    bank = ws.bank
    for (f, fl) in pairs(ws.layout.fields), c in 1:fl.components
        v, g = zero(R), zero(SVector{D,R})
        for p in ws.fptr[f]:(ws.fptr[f+1]-1)
            l, s0, raw = ws.parents[p]
            s = s0 + (c - 1) * length(raw)
            pv, pg = zero(R), zero(SVector{D,R})
            @inbounds for i in eachindex(raw)
                u = state.slot_values[s+i]
                pv += u * bank.values[l][i]
                G && (pg += u * bank.gradients[l][i])
            end
            v += pv
            G && (g += pg)
        end
        state.values[state.offsets[f]+c] = v
        state.gradients[state.offsets[f]+c] = g
    end
    return state
end

# The scalar type `T` of a layout.
_scalar(::SystemLayout{D,T}) where {D,T} = T

# Check a `state` argument (`nothing`, a pin-checked `Solution`, or a
# length-checked coefficient vector, see `_iterate_coefficients`) and convert it
# once to `Vector{R}` with `R = promote_type(T, eltype(state))`, so the eltype
# stays generic while the state's container type stops being a specialisation
# axis of the kernel. A `Vector{R}` passes through without a copy.
function _state_vector(@nospecialize(state), @nospecialize(model::Model))
    coefficients = _iterate_coefficients(state, model)
    coefficients === nothing && return nothing
    R = promote_type(_scalar(model.dofs), eltype(coefficients))
    return coefficients isa Vector{R} ? coefficients : convert(Vector{R}, coefficients)
end

# ── The kernel ────────────────────────────────────────────────────────────────

# Apply `f(form, args...)` to every form of the tuple `forms`, first to last,
# unrolled at compile time by recursion on `first` / `Base.tail`. The kernel
# below walks its `blocks` and `loads` through this rather than through
# `for form in forms`.
#
# Why not a `for` loop: every `WeakForm` closure is its own type, so a problem's
# `blocks` and `loads` are heterogeneous tuples. A `for` loop reads a tuple's
# elements through a runtime index, and inference types the loop variable as the
# union of the element types — but only up to three members (the compiler's
# `MAX_TYPEUNION_LENGTH`). A fourth distinct form type widens the loop variable
# to the abstract `BlockForm` / `LoadForm`, and every callback call, and
# everything downstream of its result, then dispatches dynamically and
# allocates, once per trial dof × component × quadrature point. Measured on
# Julia 1.12.5, serial: one scalar field on a 16 × 16 order-2 mesh assembled
# three distinct blocks in 1.7 ms and 1.0 MB, and four in 65 ms and 420 MB; a
# space-time Navier–Stokes Jacobian with the four blocks (u,u), (u,p), (p,u),
# (p,p) took 1.19 s and 7.5 GB, where the same four blocks assembled one at a
# time took 0.029 s. The recursion hands every call its element's concrete
# type, so the cost no longer depends on how many distinct forms a problem
# carries.
#
# Why the per-form work is a named function taking explicit arguments, and not
# a `do` block: a closure capturing the locals of the accumulation is too large
# to inline, so every one of those reads goes through the closure object, and
# that alone measured about 4 % slower on single-form Poisson assembly. `args`
# is one concrete `Tuple` rather than a vararg, because Julia may decline to
# specialise on a vararg that is only passed through.
@inline _foreach_form(f::F, ::Tuple{}, args::Tuple) where {F} = nothing
@inline function _foreach_form(f::F, forms::Tuple, args::Tuple) where {F}
    f(first(forms), args...)
    return _foreach_form(f, Base.tail(forms), args)
end

# Integrate every form of `pass` (its `blocks` and `loads` tuples) over one
# integration region into the workspace's local system, and return `n`, the
# region's number of active local indices. `ws.dofs` then names the global id of
# each local index, `ws.b[1:n]` is the region's rhs and, for a pass with blocks,
# `ws.K` holds its matrix as an `m × m` column-major array whose leading `n × n`
# block is the active part (`m = n + length(ws.pivots)`). This is the assembly
# hot loop.
#
# One method serves all four region kinds; only the accessors above differ:
#
#   * a **volume** region iterates reference-frame points and scales each weight
#     by `vol(box)/2ᴰ`; `q.normal` and `q.sides` are `nothing`;
#   * a **facet** region iterates precomputed physical points and carries the
#     face's outward normal `q.normal` and codim-`K` identifier `q.sides`, so
#     forms can express Neumann, Robin and Nitsche contributions;
#   * a **surface** (immersed-boundary) region is a facet with a per-point
#     `q.normal` and no `q.sides`;
#   * an **interface** region is two-sided: it refreshes both subdomains' bases
#     at the shared point and carries the per-point normal oriented from side
#     `a` to side `b`.
#
# Every active local basis function of every parent participates. In
# particular `is_facet_basis` is not applied on the physical-frame kinds: a
# function with zero *value* on a face can carry a nonzero *gradient*, which
# Nitsche-style forms rely on.
#
# Per point the kernel refreshes the bases, evaluates the state when there is
# one, and composes `q = (; x, weight, point, state, normal, sides)`, with
# `point = offset + k` the stable index `nquadpoints` sizes and the walkers
# share. Loads are evaluated first, then blocks, each through `_emit!`.
# Constrained trial columns move to the rhs as they are emitted; pivot rows and
# columns are kept whole and condensed once after the last point.
#
# `@noinline` is the per-region function barrier the drivers rely on: there is
# one compiled instance per (pass type, region kind, state type), and the
# contraction's code generation, which decides how `_test_contribution`'s
# `dot` rounds, is fixed inside it whichever driver calls it.
@noinline function _integrate!(ws::AssemblyWorkspace{D,T}, pass, region, offset::Int, state,
                               sym::Bool) where {D,T}
    _frame!(ws, region)
    n = _slots!(ws)
    m = n + length(ws.pivots)
    state === nothing || _state_region!(state, ws)
    fill!(resize!(ws.K, isempty(pass.blocks) ? 0 : m * m), zero(T))
    fill!(resize!(ws.b, m), zero(T))
    jacobian = _region_jacobian(region)
    for k in 1:_region_qpoint_count(region)
        x, w, ref = _region_point(region, k, jacobian)
        _refresh!(ws, region, ref, Val(true))
        st = state === nothing ? nothing : _state_point!(state, ws, Val(true))
        q = (; x, weight=w, point=offset + k, state=st, normal=_region_normal(region, k),
             sides=_region_sides(region))
        _foreach_form(_load!, pass.loads, (ws, q, n, m, sym))
        _foreach_form(_block!, pass.blocks, (ws, q, n, m, sym))
    end
    isempty(ws.pivots) || _condense!(ws, n, m, sym, !isempty(pass.blocks))
    return n
end

# Contract one test channel `ch` against every test function of field `f`,
# component `tc`, and emit the weighted result `e = w · ch(φ)` by the roles of
# the test row and the trial column `col`:
#
#   * a constrained test row (slot 0) is dropped;
#   * `col > 0`, an active or pivot trial column, accumulates `K[row, col] += e`;
#   * `col == 0`, a constrained trial column with value `g`, moves to the rhs as
#     `b[row] −= e·g` (column elimination). A load is emitted as this case with
#     `g = −1`, since `b − e·(−1)` is exactly `b + e` in IEEE arithmetic.
#
# Under a symmetric form an active row above an active column is skipped before
# the contraction, which halves it: sorted local numbering makes "local row <
# col" the same test as "global row < col", and the mirror rebuilds that half.
# Pivot rows and columns (`> n`) are never skipped; `_condense!` keeps the
# triangle when it folds them.
#
# The bank is read through `ws` inside the loop on purpose. Hoisting
# `ws.bank.values[l]` above it made 3D assembly 1.6× slower (8³ cells, order 2:
# 19.7 against 12.2 ms), so re-measure 3D before changing this loop's shape.
@inline function _emit!(ws::AssemblyWorkspace, ch, f::Int, tc::Int, w, col::Int, g, n::Int, m::Int,
                        sym::Bool)
    for p in ws.fptr[f]:(ws.fptr[f+1]-1)
        l, s0, raw = ws.parents[p]
        s = s0 + (tc - 1) * length(raw)
        @inbounds for a in eachindex(raw)
            row = ws.slots[s+a]
            (row == 0 || (sym && row < col <= n)) && continue
            e = w * _test_contribution(ch, ws.bank.values[l][a], ws.bank.gradients[l][a])
            if col == 0
                ws.b[row] -= e * g
            else
                ws.K[(col-1)*m+row] += e
            end
        end
    end
    return nothing
end

# One load at one quadrature point: its linear channels, per test component,
# emitted into the rhs. A field that owns no parent in this region returns
# before the callback runs (a foreign subdomain on a coupled model).
# Component-aware forms are called with `(q, test_component)`, the rest with
# `(q,)` and apply to every component alike.
function _load!(ld::LoadForm, ws::AssemblyWorkspace{D,T}, q, n::Int, m::Int, sym::Bool) where {D,T}
    f = _field_index(ws.layout, ld.test_name)
    ws.fptr[f] == ws.fptr[f+1] && return nothing
    for tc in 1:ws.layout.fields[f].components
        ch = _as_test_channels(ld.form.component_aware ? ld.form.linear(q, tc) : ld.form.linear(q),
                               Val(D), T)
        _emit!(ws, ch, f, tc, q.weight, 0, -one(T), n, m, sym)
    end
    return nothing
end

# One block at one quadrature point: the trial component → trial parent → trial
# function → test component nest, each channel emitted by `_emit!` over the test
# parents. The trial function's slot is read once, with its constrained value
# `g` when the column is eliminated. Component-aware forms see every
# (trial, test) component pair; component-unaware forms apply on the diagonal
# `test_component == trial_component` only and are called with `(q, trial)`. A
# block whose test field owns no parent in this region returns before the
# callback runs.
function _block!(bl::BlockForm, ws::AssemblyWorkspace{D,T}, q, n::Int, m::Int,
                 sym::Bool) where {D,T}
    tf, uf = _field_index(ws.layout, bl.test_name), _field_index(ws.layout, bl.trial_name)
    ws.fptr[tf] == ws.fptr[tf+1] && return nothing
    C, ul, aware = ws.layout.fields[tf].components, ws.layout.fields[uf], bl.form.component_aware
    for trc in 1:ul.components, pu in ws.fptr[uf]:(ws.fptr[uf+1]-1)
        l, s0, raw = ws.parents[pu]
        for j in eachindex(raw)
            col = ws.slots[s0+(trc-1)*length(raw)+j]
            g = col == 0 ? constrained_value(ul.dofs, raw[j], trc) : zero(T)
            trial = TrialChannels(trc, ws.bank.values[l][j], ws.bank.gradients[l][j])
            for tc in (aware ? (1:C) : (trc:min(trc, C)))
                ch = _as_test_channels(aware ? bl.form.bilinear(q, trial, tc) :
                                       bl.form.bilinear(q, trial), Val(D), T)
                _emit!(ws, ch, tf, tc, q.weight, col, g, n, m, sym)
            end
        end
    end
    return nothing
end

# Fold the region's linear-constraint pivots onto their branches, once after the
# quadrature loop. With `P` the `m × n` expansion that is the identity on the
# active indices and maps pivot `n + k` onto its active branches `(x, w)`, this
# is
#
#     K ← Pᵀ K P,    b ← Pᵀ b − lift,
#
# restricted to the active block, where `lift` carries each pivot column's
# Dirichlet branches `(w, v)`. Every expansion has depth one, so one pass over
# the rows and one over the columns is exact:
#
#   * row step, pivot test row `v` and active branch `(x, w)`:
#     `b[x] += w·b[v]` and `K[x, y] += w·K[v, y]` for every column `y`, keeping
#     `x ≥ y` for an active `y` under a symmetric form. A test row's Dirichlet
#     branch is dropped, as a constrained test row is;
#   * column step, pivot trial column `u`: an active branch `(y, w)` adds
#     `w·K[x, u]` to `K[x, y]` for `x ∈ (sym ? y : 1):n`; a Dirichlet branch
#     with value `v ≠ 0` lifts `b[x] −= (w·v)·K[x, u]` for every `x ∈ 1:n`.
#
# The branches come from the dof layer's `_fold_pivot`, which `dof_value` reads
# too, so assembly and reconstruction cannot expand a pivot differently. It
# differs from distributing every emission through the expansions only in
# summation order (≤ 1e-16 relative on the gate's pivot scenarios). It handles a
# branch shared by several slots, a pivot appearing in several parents, and the
# lift through a pivot, with no branching in the emission loop.
function _condense!(ws::AssemblyWorkspace, n::Int, m::Int, sym::Bool, matrix::Bool)
    K, b = ws.K, ws.b
    for k in eachindex(ws.pivots)
        v = n + k
        f, raw, c = ws.pivots[k]
        fl = ws.layout.fields[f]
        id = (o, cc) -> _field_component_dof(fl, o, cc)
        _fold_pivot(nothing, fl.dofs, id, raw, c) do _, g, w, _
            g == 0 && return nothing
            x = searchsortedfirst(ws.dofs, g)
            b[x] += w * b[v]
            matrix || return nothing
            for y in 1:m
                (sym && y <= n && x < y) || (K[(y-1)*m+x] += w * K[(y-1)*m+v])
            end
            return nothing
        end
    end
    matrix || return nothing
    for k in eachindex(ws.pivots)
        u = n + k
        f, raw, c = ws.pivots[k]
        fl = ws.layout.fields[f]
        id = (o, cc) -> _field_component_dof(fl, o, cc)
        _fold_pivot(nothing, fl.dofs, id, raw, c) do _, g, w, gv
            if g == 0
                iszero(gv) && return nothing
                for x in 1:n
                    b[x] -= w * gv * K[(u-1)*m+x]
                end
            else
                y = searchsortedfirst(ws.dofs, g)
                for x in (sym ? y : 1):n
                    K[(y-1)*m+x] += w * K[(u-1)*m+x]
                end
            end
            return nothing
        end
    end
    return nothing
end

# Bridge from the kernel to the sink-based drivers below: integrate one region,
# then hand its active block, rhs and dof list to `_emit_local_system!`.
function _assemble_region!(ws::AssemblyWorkspace, sink, rhs, region, blocks, loads, symmetric::Bool,
                           state, point_offset::Int)
    n = _integrate!(ws, (; blocks, loads), region, point_offset, state, symmetric)
    m = isempty(blocks) ? 0 : n + length(ws.pivots)
    k = min(n, m)
    local_matrix = view(reshape(view(ws.K, 1:(m*m)), m, m), 1:k, 1:k)
    _emit_local_system!(sink, rhs, ws.dofs, local_matrix, ws.b)
    return nothing
end

# Total quad-point count across all regions of the plan. Used by
# `nquadpoints`.
function _quadrature_count(plan::IntegrationPlan)
    sum(length(r.quadrature.weights) for r in plan.regions; init=0)
end

# Per-region quadrature-point counts, used to compute the `point_offset` each
# region sees. Defined for every region kind so the serial/threaded drivers stay
# region-kind-agnostic; the physical-frame kinds (facet / surface / interface)
# share one method as their weights vector already holds one entry per point.
_region_qpoint_count(region::VolumeRegion) = length(region.quadrature.weights)
function _region_qpoint_count(region::Union{FacetRegion,SurfaceRegion,InterfaceRegion})
    length(region.weights)
end

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
# `regions` argument is any iterable of one region kind — volume, facet,
# surface, or interface — which the single `_assemble_region!` method
# handles through its per-kind accessors. The serial walk scatters into the
# sink in a fixed region order, so repeated serial assembly is deterministic
# to the bit.
function _assemble_system_serial!(sink, rhs::Vector{T}, model::Model{D,T}, regions, symmetric::Bool,
                                  blocks, loads, region_filter, state_coefficients) where {D,T}
    state = state_coefficients === nothing ? nothing : FormState(model.dofs, state_coefficients)
    _serial_regions!(_assembly_workspace(model), sink, rhs, regions, symmetric, blocks, loads,
                     region_filter, state, _region_qpoint_offsets(regions))
    return nothing
end

# The serial region loop, behind a function barrier on the workspace: the
# workspace's basis-family parameter is a runtime property of the space, so the
# loop is compiled for the concrete workspace type and calls the kernel
# statically, with no dispatch or boxing per region.
function _serial_regions!(ws::AssemblyWorkspace, sink, rhs, regions, symmetric::Bool, blocks, loads,
                          region_filter, state, offsets)
    for (region_index, region) in enumerate(regions)
        region_filter === nothing || region_filter(region) || continue
        _assemble_region!(ws, sink, rhs, region, blocks, loads, symmetric, state,
                          offsets[region_index])
    end
    return nothing
end

# ── Deferred compute→gather threaded scatter ──────────────────────────────────
#
# The threaded matrix assembly decouples the region-parallel numeric work from
# the slot accumulation so both phases run barrier-free:
#
#   Phase 1 — every region computes its dense local block + local rhs and
#     stores them into its OWN disjoint slice of a flat arena (region-indexed).
#     No two regions write the same memory, so this is embarrassingly parallel
#     (dynamically load-balanced), with no colouring and no per-colour barrier.
#   Phase 2 — a `GatherPlan` (built once, symbolically, and cached on the
#     pattern) sums the arena into `nzval` by a disjoint column partition and
#     into `rhs` by a disjoint dof partition. Each output slot has a single
#     writer, so again no atomics and no barrier between regions.
#
# Peak matrix memory is `1 × nnz` plus the flat arena (Σ local-block entries,
# independent of thread count). Because every slot's contributions are summed
# in a fixed (region, row) order — the SAME order the serial walk uses — the
# threaded result is not just deterministic run-to-run but BIT-IDENTICAL to
# serial assembly, which removes the accumulation-order roundoff sensitivity a
# colour- or atomic-ordered scatter would leave on ill-conditioned systems.
# Works for any element type (plain `+=`). The symbolic plan is the only
# expensive part (a `searchsortedfirst` per contribution); caching it makes
# repeated assembly against one structure — Newton tangents, load steps, a
# transient march — pay it once. It is *not* carried across a structural
# change: the cache lives on `model.pattern`, which `move!` / `activate!` /
# `deactivate!` clear, so a moved overlay rebuilds the plan along with the
# pattern.

# Cached symbolic layout for one region list against one pattern. Matrix side:
# `region_arena[r]..region_arena[r+1]-1` is region r's arena slice; the column
# buckets (`ccolptr`, `cslot`, `csrc`) list, per column, the (nzval slot,
# arena index) contributions in serial order. RHS side mirrors it by dof
# (`region_rhs`, `drowptr`, `dsrc`). Matrix arrays are empty for the rhs-only
# (`nothing` sink) pass.
struct GatherPlan
    arena_len::Int
    region_arena::Vector{Int}
    ccolptr::Vector{Int}
    cslot::Vector{Int}
    csrc::Vector{Int}
    rhs_len::Int
    region_rhs::Vector{Int}
    drowptr::Vector{Int}
    dsrc::Vector{Int}
end

# Phase-1 sink: deposit a region's local block + rhs into its arena slices
# instead of scattering. `pos` / `rhs_pos` are set to the region's offsets
# before each `_assemble_region!` and advance as entries are written, in the
# exact structural order `_build_gather` enumerated (column-major, lower
# triangle row≥col when symmetric).
mutable struct ArenaSink{T}
    arena::Vector{T}
    symmetric::Bool
    pos::Int
    rhs_pos::Int
end

function _emit_matrix!(sink::ArenaSink{T}, active_dofs::AbstractVector{Int},
                       local_matrix::AbstractMatrix{T}) where {T}
    isempty(local_matrix) && return nothing
    sym = sink.symmetric
    arena = sink.arena
    p = sink.pos
    @inbounds for lc in axes(local_matrix, 2)
        col = active_dofs[lc]
        for lr in axes(local_matrix, 1)
            (sym && active_dofs[lr] < col) && continue
            arena[p] = local_matrix[lr, lc]
            p += 1
        end
    end
    sink.pos = p
    return nothing
end

# Store `local_rhs` into the region's rhs-arena slice (gathered by dof in
# phase 2) rather than scattering it into a shared vector, then defer the
# matrix block to `_emit_matrix!`.
function _emit_local_system!(sink::ArenaSink{T}, rhs_arena::Vector{T},
                             active_dofs::AbstractVector{Int}, local_matrix::AbstractMatrix{T},
                             local_rhs::AbstractVector{T}) where {T}
    p = sink.rhs_pos
    @inbounds for lr in eachindex(active_dofs)
        rhs_arena[p] = local_rhs[lr]
        p += 1
    end
    sink.rhs_pos = p
    _emit_matrix!(sink, active_dofs, local_matrix)
    return nothing
end

# Build the symbolic gather plan for `regions` against `pattern` (or `nothing`
# for the rhs-only pass). Walks every region once, resolving each structural
# entry's `nzval` slot via `searchsortedfirst` (identical lookup to the serial
# `_emit_matrix!`) and bucketing contributions by column (matrix) and dof
# (rhs) in region order, so the phase-2 gather reproduces the serial sum.
function _build_gather(model::Model, regions, pattern, symmetric::Bool, nactive::Int)
    ws = _assembly_workspace(model)
    nreg = length(regions)
    has_matrix = pattern !== nothing
    n = has_matrix ? pattern.n : 0
    region_arena = Vector{Int}(undef, nreg + 1)
    region_rhs = Vector{Int}(undef, nreg + 1)
    region_arena[1] = 1
    region_rhs[1] = 1
    bslot = [Int[] for _ in 1:n]
    bsrc = [Int[] for _ in 1:n]
    ddst = [Int[] for _ in 1:nactive]
    for (r, region) in enumerate(regions)
        ad = copy(_region_active_dofs!(ws, model, region))
        ri = region_rhs[r]
        @inbounds for lr in eachindex(ad)
            push!(ddst[ad[lr]], ri)
            ri += 1
        end
        region_rhs[r+1] = ri
        ai = region_arena[r]
        if has_matrix
            colptr = pattern.colptr
            rowval = pattern.rowval
            # Enumerated in byte-for-byte lockstep with `_emit_matrix!(::ArenaSink)`
            # (column-major, lower triangle when symmetric): the arena index `ai`
            # advances here exactly as that sink's write cursor `p` does at
            # assembly time, so `arena[csrc[k]]` is the value for slot `cslot[k]`
            # — the source of the bit-identical-to-serial guarantee. The miss
            # check mirrors the serial `_emit_matrix!(::ScatterSink)`: a
            # symbolic/numeric dof disagreement raises here rather than pushing an
            # out-of-range slot that would later corrupt a neighbouring column and
            # break phase 2's column-disjointness.
            @inbounds for lc in eachindex(ad)
                col = ad[lc]
                lo = colptr[col]
                hi = colptr[col+1] - 1
                for lr in eachindex(ad)
                    row = ad[lr]
                    (symmetric && row < col) && continue
                    slot = searchsortedfirst(rowval, row, lo, hi, Base.Order.Forward)
                    (slot <= hi && rowval[slot] == row) || _scatter_pattern_miss(row, col)
                    push!(bslot[col], slot)
                    push!(bsrc[col], ai)
                    ai += 1
                end
            end
        end
        region_arena[r+1] = ai
    end
    ccolptr = _csr_ptr(bslot)                    # bslot / bsrc are pushed in lockstep,
    cslot = _csr_flatten(bslot, ccolptr)         # so they share one pointer array
    csrc = _csr_flatten(bsrc, ccolptr)
    drowptr = _csr_ptr(ddst)
    dsrc = _csr_flatten(ddst, drowptr)
    return GatherPlan(region_arena[nreg+1] - 1, region_arena, ccolptr, cslot, csrc,
                      region_rhs[nreg+1] - 1, region_rhs, drowptr, dsrc)
end

# CSR pointer array (length n+1) from a vector of buckets: `ptr[j]..ptr[j+1]-1`
# is bucket j's slice in the flattened array.
function _csr_ptr(buckets)
    ptr = Vector{Int}(undef, length(buckets) + 1)
    ptr[1] = 1
    @inbounds for j in eachindex(buckets)
        ptr[j+1] = ptr[j] + length(buckets[j])
    end
    return ptr
end

# Flatten a bucket vector into a dense array laid out by the CSR `ptr`. Bucket
# vectors pushed in lockstep (a slot and its arena index) reuse one `ptr`.
function _csr_flatten(buckets, ptr)
    out = Vector{Int}(undef, ptr[end] - 1)
    @inbounds for j in eachindex(buckets)
        off = ptr[j] - 1
        b = buckets[j]
        for k in eachindex(b)
            out[off+k] = b[k]
        end
    end
    return out
end

# Return the cached `GatherPlan` for `regions`, building and memoising it on
# the pattern on first use (keyed by the region list's identity). The rhs-only
# pass has no pattern to cache on, so it always rebuilds — assemble_vector is
# rare and its plan skips the matrix `searchsortedfirst`.
function _gather_plan!(pattern::AssemblyPattern, model, regions, symmetric::Bool, nactive::Int)
    key = objectid(regions)
    cached = get(pattern.gather_cache, key, nothing)
    cached === nothing || return cached::GatherPlan
    plan = _build_gather(model, regions, pattern, symmetric, nactive)
    pattern.gather_cache[key] = plan
    return plan
end

# Pooled phase-1 arena buffer, reused across repeated assembly. Kept in the
# pattern's cache (alongside its `GatherPlan`) so it is invalidated with the
# structure; the trailing `::Vector{T}` is the function barrier that recovers
# the concrete type from the `Any`-valued cache. Only used on the fully
# overwritten block path (see the caller), so a stale buffer is never read.
function _gather_arena!(pattern::AssemblyPattern, key::UInt, ::Type{T}, len::Int) where {T}
    return get!(() -> Vector{T}(undef, len), pattern.gather_cache, key)::Vector{T}
end

# Partition `1:m` into `task_count` contiguous ranges of roughly equal total
# contribution count (`ptr` is a CSR offset array of length `m+1`). Contiguous
# by construction, so the ranges own disjoint output slots.
function _balanced_ranges(ptr::Vector{Int}, m::Int, total::Int, task_count::Int)
    ranges = Vector{UnitRange{Int}}(undef, task_count)
    i = 1
    for t in 1:task_count
        lo = i
        cut = div(t * total, task_count)
        while i <= m && (ptr[i+1] - 1) < cut
            i += 1
        end
        hi = min(i, m)
        ranges[t] = lo:hi
        i = hi + 1
    end
    return ranges
end

# Phase 2 (matrix): sum the arena into `nzval` by a disjoint column partition.
function _gather_matrix!(nzval::Vector{T}, plan::GatherPlan, arena::Vector{T},
                         task_count::Int) where {T}
    ccolptr = plan.ccolptr
    cslot = plan.cslot
    csrc = plan.csrc
    n = length(ccolptr) - 1
    total = ccolptr[n+1] - 1
    ranges = _balanced_ranges(ccolptr, n, total, task_count)
    @sync for rg in ranges
        Threads.@spawn @inbounds for j in rg, k in ccolptr[j]:(ccolptr[j+1]-1)
            nzval[cslot[k]] += arena[csrc[k]]
        end
    end
    return nothing
end

# Phase 2 (rhs): sum the rhs-arena into `rhs` by a disjoint dof partition.
function _gather_rhs!(rhs::Vector{T}, plan::GatherPlan, rhs_arena::Vector{T},
                      task_count::Int) where {T}
    drowptr = plan.drowptr
    dsrc = plan.dsrc
    nactive = length(drowptr) - 1
    total = drowptr[nactive+1] - 1
    ranges = _balanced_ranges(drowptr, nactive, total, task_count)
    @sync for rg in ranges
        Threads.@spawn @inbounds for d in rg, k in drowptr[d]:(drowptr[d+1]-1)
            rhs[d] += rhs_arena[dsrc[k]]
        end
    end
    return nothing
end

# Threaded assembly driver (deferred compute→gather; see the block comment
# above `GatherPlan`). Phase 1 dynamically load-balances the regions across
# `nthreads` tasks — each grabs the next region via an atomic counter and
# deposits its local block / rhs into that region's disjoint arena slice
# (`ArenaSink`), so there is no write contention and no barrier. Phase 2 sums
# the arenas into the shared `sink.nzval` / `rhs` by disjoint column / dof
# partitions. The symbolic `GatherPlan` is cached on the pattern.
#
# Region-kind-agnostic: `regions` may be a volume plan's region vector or a
# facet selector's region list. The rhs-only pass (`sink === nothing`, from
# `assemble_vector`) has no pattern; its plan carries only the rhs layout and
# phase 2 gathers just the rhs.
function _assemble_system_threaded!(sink, rhs::Vector{T}, model::Model{D,T}, regions,
                                    symmetric::Bool, blocks, loads, region_filter,
                                    state_coefficients) where {D,T}
    task_count = Threads.nthreads()
    offsets = _region_qpoint_offsets(regions)
    nactive = length(rhs)
    # A pass with no bilinear blocks (e.g. a load-only problem, or `assemble_vector`
    # with its `nothing` sink) has no matrix to gather — its pattern is empty, so
    # build the plan on the rhs layout alone and skip the matrix phase.
    pattern = sink === nothing ? nothing : sink.pattern
    has_matrix = pattern !== nothing && !isempty(blocks)
    plan = has_matrix ? _gather_plan!(pattern, model, regions, symmetric, nactive) :
           _build_gather(model, regions, nothing, symmetric, nactive)

    # On the block path with no `region_filter` every arena slot is overwritten
    # in phase 1, so the buffers can be POOLED on the pattern — repeated assembly
    # (Newton tangents, transient steps) then reuses them instead of allocating
    # ~`nnz` of arena per call, and the pool is dropped for free when the pattern
    # rebuilds. `region_filter` (compact loads) skips regions, leaving their
    # slices unwritten, so that path allocates fresh and zeroes; the rhs-only
    # pass has no pattern to pool on.
    if has_matrix && region_filter === nothing
        rid = objectid(regions)
        arena = _gather_arena!(pattern, hash(:arena, rid), T, plan.arena_len)
        rhs_arena = _gather_arena!(pattern, hash(:rhs_arena, rid), T, plan.rhs_len)
    else
        arena = Vector{T}(undef, plan.arena_len)
        rhs_arena = Vector{T}(undef, plan.rhs_len)
        if region_filter !== nothing
            fill!(arena, zero(T))
            fill!(rhs_arena, zero(T))
        end
    end

    next = Threads.Atomic{Int}(0)
    @sync for _ in 1:task_count
        Threads.@spawn begin
            state = state_coefficients === nothing ? nothing :
                    FormState(model.dofs, state_coefficients)
            _arena_regions!(_assembly_workspace(model), ArenaSink{T}(arena, symmetric, 1, 1),
                            rhs_arena, regions, plan, next, symmetric, blocks, loads, region_filter,
                            state, offsets)
        end
    end

    has_matrix && _gather_matrix!(sink.nzval, plan, arena, task_count)
    _gather_rhs!(rhs, plan, rhs_arena, task_count)
    return nothing
end

# One phase-1 task: take regions from the shared atomic counter until none is
# left and deposit each into its arena slices. Behind a function barrier on the
# workspace for the reason given at `_serial_regions!`.
function _arena_regions!(ws::AssemblyWorkspace, asink::ArenaSink, rhs_arena, regions,
                         plan::GatherPlan, next::Threads.Atomic{Int}, symmetric::Bool, blocks,
                         loads, region_filter, state, offsets)
    while true
        region_index = Threads.atomic_add!(next, 1) + 1
        region_index > length(regions) && break
        region = regions[region_index]
        region_filter === nothing || region_filter(region) || continue
        asink.pos = plan.region_arena[region_index]
        asink.rhs_pos = plan.region_rhs[region_index]
        _assemble_region!(ws, asink, rhs_arena, region, blocks, loads, symmetric, state,
                          offsets[region_index])
    end
    return nothing
end

# Locate the `(blocks, loads)` slot of one `on` tag inside the partition
# list, appending a fresh slot when the tag has not been seen. Matching is
# `isequal`, exactly the equality a `Dict` key lookup applied; appending is
# what keeps the list in first-appearance order.
function _partition_slot!(partitions, on)
    for (tag, slot) in partitions
        isequal(tag, on) && return slot
    end
    slot = (Any[], Any[])
    push!(partitions, on => slot)
    return slot
end

# Partition `blocks` and `loads` by their `on` tag. Returns the
# volume-tagged forms (with `on === nothing`) and one
# `on => (blocks, loads)` pair per non-volume tag, grouping every
# contribution carrying that tag so a single assembly pass handles them
# all. The partitions are a `Vector` of pairs in first-appearance order
# rather than a `Dict`: assembly scatters every pass into the same
# accumulators, so the pass order decides the summation order and must be
# a property of the forms the caller passed, not of a hash table's slot
# layout. A problem carries a handful of distinct `on` targets, so the
# linear `isequal` scan that replaces the hash lookup is not a cost.
function _partition_forms_by_on(blocks, loads)
    volume_blocks = filter(b -> b.on === nothing, blocks)
    volume_loads = filter(l -> l.on === nothing, loads)
    partitions = Pair{Any,Tuple{Vector{Any},Vector{Any}}}[]
    for b in blocks
        b.on === nothing || push!(_partition_slot!(partitions, b.on)[1], b)
    end
    for l in loads
        l.on === nothing || push!(_partition_slot!(partitions, l.on)[2], l)
    end
    return volume_blocks, volume_loads, partitions
end

# Split one single-sided `on`-partition by subdomain space, returning
# `(space, blocks, loads)` for each space that has forms in it, in
# `problem_spaces` order. The space of a form is that of its test field (whose
# name is always a problem field, even for a one-shot
# `assemble_matrix(model, block; on=…)` whose mesh the model never cached), so
# the spaces reached here are always a subset of `problem_spaces(model.problem)`
# and iterating that already-deduplicated list is enough.
#
# A single-sided region list is built against one space and its parents live on
# that space's level block, so only that space's fields evaluate on it
# (`region_parents`). One `on=` target named by fields on *different* subdomains
# therefore needs one region list — and one pass — per subdomain: a face named
# `boundary(axis=1, side=:upper)` on two subdomains is two different sets of
# facets, and each subdomain's pass writes only rows its own fields own, so the
# passes are independent and their order is immaterial.
#
# A space with no forms in the partition is dropped rather than emitted with
# empty lists: an empty pass still contributes its region list's dof blocks to
# the sparsity pattern (`_assembly_region_lists`), which would add structural
# zeros for a subdomain that never integrates there.
#
# `Interface` partitions are two-sided and stay one pass with `space === nothing`
# — their `InterfaceRegion` carries both sides' parents itself, and the four
# blocks a `couple` call emits alternate test fields between the two subdomains,
# so splitting them by test space would tear one coupling into two passes.
function _partition_space(model::Model, on, blocks, loads)
    on isa Interface && return [(nothing, blocks, loads)]
    # Single-domain is the dominant path and has nothing to split: every test
    # field is on the one representative space, so the partition passes through
    # whole.
    _is_multidomain(model.problem) || return [(model.problem.space, blocks, loads)]

    form_space(form) = _field_space(model.problem, form.test_name)
    # Untyped: `blocks` / `loads` arrive as `Vector`s here and as `()` from the
    # pattern walk, and `filter` preserves each container type.
    passes = Any[]
    for space in problem_spaces(model.problem)
        space_blocks = filter(b -> form_space(b) === space, blocks)
        space_loads = filter(l -> form_space(l) === space, loads)
        isempty(space_blocks) && isempty(space_loads) && continue
        push!(passes, (space, space_blocks, space_loads))
    end
    return passes
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

Returns the sparse global matrix over the *active* dofs: constrained
columns are eliminated, so the matrix already matches the reduced system.
The right-hand-side correction that elimination produces (`−A_c g` for a
nonzero Dirichlet datum `g`) is computed and then discarded, because this
entry point returns a matrix only — call [`assemble!`](@ref) when the
problem has nonzero Dirichlet data and you need a consistent pair.

Does not touch `model.matrix` or `model.rhs`; for in-place assembly that
updates the cached operators use [`assemble!`](@ref).
"""
function assemble_matrix(model::Model{D,T}, blocks; symmetric=nothing,
                         threaded::Bool=Threads.nthreads() > 1, state=nothing) where {D,T}
    block_tuple = _form_tuple(blocks)
    symmetric_value = symmetric === nothing ? all(block -> block.form.symmetric, block_tuple) :
                      Bool(symmetric)
    nactive = active_unknowns(model.dofs)
    coeffs = _state_vector(state, model)
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

Returns the assembled right-hand-side vector over the active dofs. It
carries the load integrals only: no bilinear block participates in this
call, so there is no Dirichlet column elimination and therefore no `−A_c g`
correction for nonzero Dirichlet data — [`assemble!`](@ref) is the entry
point that produces a matrix and a right-hand side consistent with each
other.
"""
function assemble_vector(model::Model{D,T}, loads; threaded::Bool=Threads.nthreads() > 1,
                         region_filter=nothing, state=nothing) where {D,T}
    load_tuple = _form_tuple(loads)
    nactive = active_unknowns(model.dofs)
    coeffs = _state_vector(state, model)
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

Constraints are honoured through the dof layer: strong Dirichlet
elimination (a constrained trial column moves to the right-hand side,
weighted by its stored value), dof-wise homogeneous overlay elimination, and
— when the basis family supplies them — general homogeneous linear
constraints, whose pivots are distributed over their expansions during the
scatter.
"""
function assemble!(model::Model{D,T}; threaded::Bool=Threads.nthreads() > 1) where {D,T}
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
    scaled_condition_estimate = _scaled_condition_estimate(matrix)
    model.matrix = matrix
    model.rhs = rhs
    diag = model.diagnostics
    diag.active_unknowns = nactive
    diag.symmetry_residual = symmetry_residual
    diag.condition_estimate = condition_estimate
    diag.scaled_condition_estimate = scaled_condition_estimate
    _set_plan_stats_multi!(diag, integration_plans(model))
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

    # Volume contributions — the dominant path. One pass per distinct subdomain
    # integration plan; each region is owned by the fields on its subdomain
    # intrinsically (`region_parents`), so single-domain collapses to one pass
    # over the shared plan with every field participating.
    if !isempty(volume_blocks) || !isempty(volume_loads)
        for regions in _volume_passes(model)
            _run_pass!(sink, rhs, model, regions, symmetric, Tuple(volume_blocks),
                       Tuple(volume_loads), region_filter, state_coefficients, threaded)
        end
    end

    # Non-volume contributions — one assembly pass per unique `on=` value *per
    # subdomain space naming it* (`_partition_space`), so a target named by two
    # subdomains contributes on both instead of only the first. `region_filter`
    # is a volume-only convenience and is not forwarded to the facet / surface
    # passes.
    for (on, (on_blocks, on_loads)) in partitions,
        (space, sel_blocks, sel_loads) in _partition_space(model, on, on_blocks, on_loads)

        regions = _resolve_on_regions(model, on, space)
        isempty(regions) && continue
        # Every pass — volume, facet, surface, and two-sided interface — runs
        # threaded. One unexplained observation stands against that: a two-sided
        # `InterfaceRegion` pass for a vector coupling, under `--code-coverage`
        # at ≥2 threads, was nondeterministically corrupted on both x86 and ARM.
        # Never seen outside coverage, not reproduced since across ~30,000
        # bit-exact comparisons, mechanism not established (see the `couple`
        # docstring); the parallel path ships and `threaded=false` is the
        # escape hatch.
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
        _assemble_system_serial!(sink, rhs, model, regions, symmetric, blocks, loads, region_filter,
                                 state_coefficients)
    end
    return nothing
end

# Resolve a non-`nothing` `on=` value **on a named subdomain space** to its
# region list. Reads from the per-kind cache on `model` when the entry exists
# (every (target, space) site referenced by `prepare(problem)` is pre-resolved);
# falls back to a fresh build against `space` for one-shot calls like
# `assemble_matrix(model, block_with_unseen_on=…)`. "Fresh" is per *selector*,
# not per face: the miss path resolves through the model's `FacetResolver`, so an
# unseen selector built from faces `prepare` already resolved reuses them, and a
# genuinely new face is resolved once however many such calls name it.
#
# `space` is half of the cache key, not merely a fallback for the miss path: a
# hit is by construction a region list already built against that same space, so
# a caller naming subdomain 2 can never be handed subdomain 1's facets. It
# defaults to the representative space, which is the one space on a single-domain
# model; field-agnostic callers on a coupled model (`boundary_integral`) resolve
# the space themselves rather than take the default.
function _resolve_on_regions(model::Model{D,T}, selector::BoundarySelector,
                             space=model.problem.space) where {D,T}
    return get(() -> _facet_regions_for_selector(space, selector, model.facet_resolver),
               model.facet_regions, (selector, space))
end

function _resolve_on_regions(model::Model{D,T}, mesh::BoundaryMesh{D,T},
                             space=model.problem.space) where {D,T}
    return get(() -> _surface_regions_for_mesh(space, mesh, model.dofs.tolerance),
               model.surface_regions, (mesh, space))
end

# Resolve the two-sided integration regions for an interface coupling tag from
# the per-model cache (pre-resolved at `prepare` for every referenced
# interface); build on demand for a one-shot `assemble_matrix(model, block;
# on=iface)` with an unseen interface.
function _resolve_on_regions(model::Model, iface::Interface, _space=nothing)
    return get(() -> _interface_regions_for_model(model, iface), model.interface_regions, iface)
end

# The subdomain space a single-sided `on=` region lookup acts on. A named
# `field` picks its space; without one, a single-domain model has exactly one
# answer and a multi-domain model has none, so it raises rather than pick — see
# the `boundary_integral` docstring for why neither silent default is
# defensible. Ignored for two-sided `Interface` targets, which carry both sides.
function _on_regions_space(model::Model, field::Union{Nothing,Symbol})
    field === nothing || return _field_space(model.problem, field)
    length(problem_spaces(model.problem)) == 1 && return model.problem.space
    names = join((":" * String(f.name) for f in model.problem.fields), ", ")
    throw(ArgumentError("field argument is required for multi-domain models; pass field=… " *
                        "(one of $names)"))
end

# Resolve an interface tag's field indices and subdomain spaces from the model,
# then build its regions. Field indices come from the dof layout (global field
# order); spaces come from the effective problem's fields.
function _interface_regions_for_model(model::Model, iface::Interface)
    field_a = _field_index(model.dofs, iface.field_a)
    field_b = _field_index(model.dofs, iface.field_b)
    space_a = _field_space(model.problem, iface.field_a)
    space_b = _field_space(model.problem, iface.field_b)
    return _interface_regions(iface, space_a, space_b, field_a, field_b, model.dofs.tolerance)
end

"""
    nquadpoints(model::Model; kind::Symbol = :volume) -> Int
    nquadpoints(model::Model; on, field = nothing) -> Int

Number of quadrature points on `model`.

The `on=` form returns the size of the **one** region list a form tagged
with that `on=` value integrates over, which is exactly the range of the
`q.point` index that form sees: `1:nquadpoints(model; on=…)`. It takes
precedence over `kind`, which the two forms never need together. `on` is
a [`BoundarySelector`](@ref) or a [`BoundaryMesh`](@ref), where `field`
names the subdomain on a coupled model exactly as it does for
[`boundary_integral`](@ref), or an [`Interface`](@ref), which spans both
its subdomains and ignores `field` (this case is
[`interface_quadrature_count`](@ref)). This is the form that sizes
per-quadrature-point state — history variables for an inelastic law, a
cohesive `κ` along an interface.

The `kind=` form returns the **aggregate** over every cached region list
of that kind, a structural count for diagnostics:

  - `:volume` (default) — the points of every subdomain integration plan.
  - `:facet` — every cached [`FacetRegion`](@ref). On an immersed space this
    is not a function of the mesh and the orders alone: a region on a face
    `∂Ω` crosses carries the moment-fit rule on the facet's affine slice, not
    the tensor product `_facet_quadrature_counts` sizes, so its point count is
    whatever that rule came out at — at most the moment basis size on a
    `:cut_fitted` region, which at a low order is *more* points than the
    untrimmed rule rather than fewer (measured 3 → 5 on a cut face of an
    order-2 space), the raw Saye rule on a `:cut_fallback` one, and none at all
    for a face wholly outside `Ω` under strict `α = 0`. `α > 0` appends the
    full-face tensor rule to each of those. The count therefore moves with the
    geometry, and `cut_facet_region_count` says how many regions are in that
    regime.
  - `:surface` — every cached immersed [`SurfaceRegion`](@ref)
    ([`BoundaryMesh`](@ref) integration).
  - `:interface` — every cached multi-domain [`InterfaceRegion`](@ref).

`q.point` is numbered **per region list**, not per kind: it restarts at 1
for each distinct `on=` value and, on a coupled model, for each subdomain
naming that value. So the aggregate exceeds the `q.point` range whenever
the model caches more than one list of that kind, and indexing a
`kind=`-sized array by `q.point` would alias one list onto another. Size
per-point state with `on=`; the `kind=:volume` default is safe only
because a single-domain model has exactly one plan, and
[`foreach_quadrature_point`](@ref) (with `QuadField{T}(model; init)`,
which inherits its restriction) rejects a coupled model for that reason.
"""
function nquadpoints(model::Model; kind::Symbol=:volume, on=nothing,
                     field::Union{Nothing,Symbol}=nothing)
    on isa Interface && return interface_quadrature_count(model, on)
    on === nothing || return sum(_region_qpoint_count,
               _resolve_on_regions(model, on, _on_regions_space(model, field)); init=0)
    kind === :volume && return sum(_quadrature_count(p) for p in integration_plans(model); init=0)
    kind === :facet && return _cached_quadpoint_count(model.facet_regions)
    kind === :surface && return _cached_quadpoint_count(model.surface_regions)
    kind === :interface && return _cached_quadpoint_count(model.interface_regions)
    throw(ArgumentError("nquadpoints kind must be :volume, :facet, :surface, or :interface, " *
                        "got $kind"))
end

# Sum of physical-frame Q-points across every region list in one of the
# model's region caches (`facet_regions`, `surface_regions`,
# `interface_regions`). Each region carries its Q-point list precomputed, so
# the sum is one `length` per region. Backs the aggregate `kind=` branches of
# `nquadpoints`.
function _cached_quadpoint_count(cache)
    total = 0
    for (_, list) in cache, region in list
        total += _region_qpoint_count(region)
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

Single-domain only: on a coupled (multi-subdomain) model this throws,
since a single global per-point ordering across subdomains is not yet
defined. `QuadField{T}(model; init)` inherits the same restriction.
"""
function foreach_quadrature_point(f, model::Model{D,T}; state=nothing) where {D,T}
    # Walks a single subdomain plan with a single-space workspace, so a coupled
    # model would silently visit only the first subdomain (and evaluate `state`
    # on the wrong parents). Per-point iteration across coupled subdomains — one
    # global quadrature-point ordering shared with assembly — is deferred.
    _assert_single_domain(model, "foreach_quadrature_point")
    regions = integration_plan(model).regions
    return _walk(f, _volume_payload, _assembly_workspace(model), regions,
                 _region_qpoint_offsets(regions), _walk_state(state, model))
end

# The `q` payloads of the two walkers: the volume walker's `(x, weight, point,
# state)` and the interface walker's, which adds the interface normal.
_volume_payload(region, k, x, weight, point, state) = (; x, weight, point, state)
function _interface_payload(region, k, x, weight, point, state)
    return (; x, weight, point, state, normal=_region_normal(region, k))
end

# A walker's `FormState`, or `nothing` without a state.
function _walk_state(@nospecialize(state), @nospecialize(model::Model))
    coefficients = _state_vector(state, model)
    return coefficients === nothing ? nothing : FormState(model.dofs, coefficients)
end

# Call `f(q)` at every quadrature point of `regions`, in the serial assembly
# order, with the point's coordinates and weight computed exactly as the kernel
# computes them and `q.point = offsets[r] + k` its numbering. With a state, each
# region's dof values are read once and every field is evaluated at every point
# through the same `_frame!`, `_refresh!` and `_state_point!` the kernel uses;
# without one, no basis is evaluated at all. Behind a function barrier on the
# workspace, so the per-point calls are static.
function _walk(f, payload, ws::AssemblyWorkspace, regions, offsets, state)
    for (r, region) in enumerate(regions)
        state === nothing || _state_region!(state, _frame!(ws, region))
        jacobian = _region_jacobian(region)
        for k in 1:_region_qpoint_count(region)
            x, w, ref = _region_point(region, k, jacobian)
            st = state === nothing ? nothing :
                 (_refresh!(ws, region, ref, Val(true)); _state_point!(state, ws, Val(true)))
            f(payload(region, k, x, w, offsets[r] + k, st))
        end
    end
    return nothing
end

"""
    interface_quadrature_count(model, iface::Interface) -> Int

Number of quadrature points of the two-sided interface `iface` (built with
[`interface`](@ref)) on the prepared `model`. This is the size of a per-point
history vector keyed by the `q.point` index that
[`foreach_interface_quadrature_point`](@ref) and the coupling forms expose.
"""
function interface_quadrature_count(model::Model, iface::Interface)
    regions = _resolve_on_regions(model, iface)
    return sum(_region_qpoint_count, regions; init=0)
end

"""
    foreach_interface_quadrature_point(f, model, iface::Interface; state=nothing)

Call `f(q)` at every quadrature point of the two-sided interface `iface` (built
with [`interface`](@ref)). The payload is

    q = (; x, weight, point, normal, state)

where

  - `q.x` — physical coordinate of the interface quadrature point,
  - `q.weight` — the surface quadrature weight (arc length in 2-D, area in 3-D),
  - `q.point` — stable index in `1:interface_quadrature_count(model, iface)`,
    matching the index the coupling forms see during assembly, so it is the
    natural key for per-interface-point history (an irreversible cohesive
    `κ`, a friction state, …),
  - `q.normal` — the interface unit normal, oriented from side `a` toward side
    `b` (the two fields passed to [`couple`](@ref), in order) — a convention
    the interface mesh carries and the caller owns, see [`Interface`](@ref),
  - `q.state` — `nothing` unless a `state` (a [`Solution`](@ref) or active
    coefficient vector) was passed, in which case it is a two-field
    [`FormState`](@ref) exposing **both** coupled fields via
    `value(q.state, field)` / `field_gradient(q.state, field)` (each evaluated
    in its own subdomain's covering cut cell at the shared point `q.x`).

Unlike [`foreach_quadrature_point`](@ref) (single subdomain, volume points),
this walks the two-sided interface regions of a coupled model, so it is the
companion iterator for reading and committing interface state around a
nonlinear solve. Dimension-generic (2-D polyline / 3-D triangle interface);
iteration order matches the serial interface-assembly pass.
"""
function foreach_interface_quadrature_point(f, model::Model{D,T}, iface::Interface;
                                            state=nothing) where {D,T}
    regions = _resolve_on_regions(model, iface)
    # The frame and the refresh cover *both* sides' parents, so the state
    # evaluates each coupled field in its own subdomain's cut cell.
    return _walk(f, _interface_payload, _assembly_workspace(model), regions,
                 _region_qpoint_offsets(regions), _walk_state(state, model))
end
