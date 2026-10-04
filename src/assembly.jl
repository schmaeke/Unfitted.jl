# Coupled-Galerkin assembly kernel for one or more fields over a
# superposition `Space`. This file owns the hot loop and the public
# assembly API; the user-facing types (weak forms, fields, problems)
# live in `problems.jl` and the prepared-problem lifecycle (`Model`,
# `prepare`, `move!`, …) lives in `model.jl`.
#
# The strategy is symbolic + numeric sparse assembly. An assembly call
# first lists its passes (`_passes`): one integration region list each,
# with the forms integrated over it. The symbolic data — every region's
# sorted active dof list and the CSC sparsity pattern over them — is
# derived once per structure and cached on the `Model`
# (`model.assembly`). The numeric pass walks every region, evaluates
# basis values and physical gradients per parent level, accumulates the
# user's weak form into a per-task local matrix / rhs over the region's
# sorted local dof numbering, and merges each local column into its
# pattern column of the prebuilt `nzval`. A dof pair appearing in many
# regions therefore accumulates with `+=` into one slot rather than
# inflating a triplet array. The threaded driver runs the same kernel,
# parks each region's result in a private arena slice, and sums the
# arena by owned dof ranges through the same column merge, so its result
# is bit-identical to the serial walk.
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

# Gradient analogue of the one-parent `_field_value`: an `SVector{D}` per
# parent. Its callers sum over parents themselves.
function _field_gradient(data, layout::Union{DofLayout{D,T},FieldLayout{D,T}}, coefficients,
                         component::Integer=1) where {D,T}
    R = promote_type(T, eltype(coefficients))
    result = zero(SVector{D,R})
    for i in eachindex(data.raw_dofs)
        result += dof_value(layout, coefficients, data.raw_dofs[i], component) * data.gradients[i]
    end
    return result
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

# ── Region lists and passes ───────────────────────────────────────────────────
#
# The cache types (`RegionList`, `RegionDofs`, `AssemblyPattern`,
# `AssemblyCache`) live in `model.jl`, because the `Model` holds them and that
# file is included first; the code that fills and reads them lives here.
#
# Everything from here to the end of the symbolic section is an assembly call's
# cold path: it runs once per call or once per pass, never per region. Whatever
# takes the `Model` or the form tuples is deliberately unspecialised
# (`@nospecialize`), so it compiles once per process rather than once per form
# tuple or problem type; the form types and `Model{D,T,P}` reach only the
# per-pass drivers and the kernel. On a prototype of this structure, measured
# per new form type at `-O0` against the fully specialised assembler it
# replaced, a new block's first assembly compiled in 0.28× the time and the same
# block on a second problem type in 0.03×; the price is a few microseconds of
# dynamic dispatch per call.

# A region list with its `q.point` offsets. Its active dofs are derived later,
# on first need (`_dofs!`).
function RegionList(regions::Vector{R}) where {R}
    offsets = Vector{Int}(undef, length(regions) + 1)
    offsets[1] = 0
    for (r, region) in pairs(regions)
        offsets[r+1] = offsets[r] + _region_qpoint_count(region)
    end
    return RegionList{R}(regions, offsets, nothing)
end

# One pass of one assembly call: a region list, its `RegionKey`, and the blocks
# and loads integrated over it as concretely typed tuples. The per-pass drivers
# and the kernel are compiled per `Pass` type — per form tuple and region kind —
# and nothing else in an assembly call is.
struct Pass{R,B<:Tuple,L<:Tuple}
    key::RegionKey
    list::RegionList{R}
    blocks::B
    loads::L
end

# The regions of pass target `target` on `space`, and whether they are the
# model's own prepare-time list (`true`) or were resolved just now for a target
# `prepare` never saw (`false`). The miss paths are the builders `prepare` runs.
# "Fresh" is per target, not per face: a selector resolves through the model's
# `FacetResolver`, so a face `prepare` already resolved is reused, and a new face
# is resolved once however many selectors name it.
function _regions(@nospecialize(model::Model), ::Nothing, @nospecialize(space))
    plan = findfirst(s -> s === space, problem_spaces(model.problem))
    return model.space_plans[plan].regions, true
end
function _regions(@nospecialize(model::Model), selector::BoundarySelector, @nospecialize(space))
    cached = get(model.facet_regions, (selector, space), nothing)
    cached === nothing || return cached, true
    return _facet_regions_for_selector(space, selector, model.facet_resolver), false
end
function _regions(@nospecialize(model::Model), mesh::BoundaryMesh, @nospecialize(space))
    cached = get(model.surface_regions, (mesh, space), nothing)
    cached === nothing || return cached, true
    return _surface_regions_for_mesh(space, mesh, model.dofs.tolerance), false
end
# An interface's field indices come from the dof layout (global field order),
# its two spaces from the effective problem's fields.
function _regions(@nospecialize(model::Model), iface::Interface, _)
    cached = get(model.interface_regions, iface, nothing)
    cached === nothing || return cached, true
    field_a = _field_index(model.dofs, iface.field_a)
    field_b = _field_index(model.dofs, iface.field_b)
    return _interface_regions(iface, _field_space(model.problem, iface.field_a),
                              _field_space(model.problem, iface.field_b), field_a, field_b,
                              model.dofs.tolerance), false
end

# The `RegionList` of pass target `target` on `space` (`RegionKey`
# `(target, space)`; `space` is `nothing` for an `Interface`). Every consumer
# resolves through here — the assembly passes, `nquadpoints`, the
# quadrature-point walkers and `boundary_integral` — so all of them see one
# list, one set of offsets and so one `q.point` numbering. A list `prepare`
# resolved is kept for the model's lifetime; a target it did not see goes to
# the bounded one-shot cache, so a hot one is resolved once rather than on every
# call. Lookup and build run under the cache lock: a one-shot facet miss also
# extends the model's `FacetResolver` memo, which concurrent calls must not do
# at the same time. `Base.@lock` rather than `lock(f, l)`, whose closure would
# be one more type to compile for every caller.
function _region_list(@nospecialize(model::Model), @nospecialize(target), @nospecialize(space))
    cache = model.assembly
    key = (target, space)
    Base.@lock cache.lock begin
        list = get(cache.lists, key, nothing)
        list === nothing || return list
        list = _lru_get!(cache.oneshot, key)
        list === nothing || return list
        regions, owned = _regions(model, target, space)
        list = RegionList(regions)
        owned ? (cache.lists[key] = list) : _lru_push!(cache.oneshot, key => list, 8)
        return list
    end
end

# A small most-recently-used-first cache kept in a `Vector` of `key => value`
# pairs. `_lru_get!` returns the value stored under `key` and moves its entry to
# the front, or returns `nothing`; `_lru_push!` adds an entry at the front and
# drops the least recently used one beyond `capacity`. Keys compare by
# `isequal`, as a `Dict`'s would. A linear search is the right tool at these
# sizes (at most 8 entries), and the bound is what keeps one-shot targets and
# alternating operators from growing the cache without limit.
function _lru_get!(lru::Vector, @nospecialize(key))
    i = findfirst(entry -> isequal(first(entry), key), lru)
    i === nothing && return nothing
    i > 1 && pushfirst!(lru, popat!(lru, i))
    return last(first(lru))
end
function _lru_push!(lru::Vector, entry::Pair, capacity::Int)
    pushfirst!(lru, entry)
    length(lru) > capacity && pop!(lru)
    return last(entry)
end

# The subdomain space a single-sided `on=` target resolves on, for the callers
# that name a target rather than a form (`nquadpoints`, `boundary_integral`). A
# named `field` picks its space; without one, a single-domain model has exactly
# one answer and a multi-domain model has none, so it raises rather than pick —
# see the `boundary_integral` docstring for why neither silent default is
# defensible. An `Interface` spans both its subdomains and resolves on
# `nothing`.
function _target_space(@nospecialize(model::Model), @nospecialize(on), field::Union{Nothing,Symbol})
    on isa Interface && return nothing
    field === nothing || return _field_space(model.problem, field)
    length(problem_spaces(model.problem)) == 1 && return model.problem.space
    names = join((":" * String(f.name) for f in model.problem.fields), ", ")
    throw(ArgumentError("field argument is required for multi-domain models; pass field=… " *
                        "(one of $names)"))
end

# The passes of one assembly call over `blocks` and `loads`, in the order they
# are summed. Every pass accumulates into the same matrix and rhs, so this order
# is part of the result, and it is a property of the caller's form list alone:
#
#   1. the volume target first, then every `on=` target in its first appearance
#      in `(blocks..., loads...)`, matched by `isequal` — value-equal selectors
#      are one target, meshes and interfaces match by identity;
#   2. each target split per subdomain space, in `problem_spaces` order, a pass
#      taking the forms whose test field lives on that space. A region list is
#      built on one space's level block and only that space's fields evaluate on
#      it (`region_parents`), so a target named on two subdomains is two
#      independent passes, and a volume form runs on its own subdomain's plan
#      only;
#   3. except an `Interface`, which is one pass: its regions carry both sides'
#      parents, and the four blocks of a `couple` alternate their test field
#      between the two subdomains.
#
# A space with no form on a target gets no pass, and neither does a target whose
# list is empty. The sparsity pattern is built from the passes that carry a
# block, so the symbolic and the numeric side enumerate the same lists by
# construction.
function _passes(@nospecialize(model::Model), @nospecialize(blocks::Tuple),
                 @nospecialize(loads::Tuple))
    forms = (blocks..., loads...)
    targets = Any[]
    any(form -> form.on === nothing, forms) && push!(targets, nothing)
    for form in forms
        form.on === nothing || any(t -> isequal(t, form.on), targets) || push!(targets, form.on)
    end
    passes = Any[]
    for target in targets,
        space in (target isa Interface ? Any[nothing] : problem_spaces(model.problem))

        mine = form -> isequal(form.on, target) &&
                       (space === nothing || _field_space(model.problem, form.test_name) === space)
        target_blocks, target_loads = filter(mine, blocks), filter(mine, loads)
        isempty(target_blocks) && isempty(target_loads) && continue
        list = _region_list(model, target, space)
        isempty(list.regions) ||
            push!(passes, Pass((target, space), list, target_blocks, target_loads))
    end
    return passes
end

# ── Symbolic data: region dofs and the sparsity pattern ───────────────────────

# The active dofs of every region of `list` (see `RegionDofs`), derived once per
# list and kept on it. It runs the numeric pass's own `_frame!` and `_slots!` on
# the workspace `ws`, so the pattern and the threaded arena layout describe
# exactly the dofs the kernel emits: active ids and the active branches of every
# pivot, never a constrained id. Called under the cache lock
# (`_assembly_pattern!`, `_dofs_all!`), which is what makes the one-time fill
# safe when calls run concurrently.
function _dofs!(list::RegionList, ws)
    list.dofs === nothing || return list.dofs
    ptr = Vector{Int}(undef, length(list.regions) + 1)
    ptr[1] = 1
    val = Int[]
    for (r, region) in pairs(list.regions)
        _slots!(_frame!(ws, region))
        append!(val, ws.dofs)
        ptr[r+1] = length(val) + 1
    end
    return list.dofs = RegionDofs(ptr, val)
end

# The region dofs of every pass of a threaded call, whose arena layout is read
# from them (`_threaded!`), under one lock for the whole call.
function _dofs_all!(@nospecialize(cache::AssemblyCache), passes::Vector{Any}, @nospecialize(ws))
    Base.@lock cache.lock begin
        for pass in passes
            _dofs!(pass.list, ws)
        end
    end
    return nothing
end

# The CSC sparsity pattern over `n` active dofs of the dense blocks on every
# region of `sets` (the matrix passes' region dofs): column `j` holds every row
# `i` that shares a region with `j` — only `i ≥ j` when `symmetric` — ascending.
#
# Gustavson's column dedup marks each row with the current column in an `O(n)`
# `marker` array, which is correct only when each output column is processed
# contiguously. So the regions are first inverted into a dof → regions index:
# the concatenated region dofs read as the CSC incidence dofs × regions, whose
# transpose lists, per dof, the regions containing it. That index costs
# `O(Σ nactive)` — one entry per (region, dof) incidence, far below the
# `O(Σ nactive²)` dense-block over-count — and the pattern itself is the only
# nnz-sized allocation.
#
# Two passes over the columns: the first counts each column's rows to build
# `colptr`, the second fills `rowval` and sorts each column (CSC wants ascending
# rows, and the scatter's merge relies on it). Counting first sizes `rowval`
# exactly; appending instead nearly doubled the transient (18.6 against
# 10.8 MiB on a 2D overlay fixture). `marker` need not be reset between columns
# within a pass — the column id strictly increases, so a stale `marker[i]` never
# equals the current one — but is cleared between the two passes.
function _pattern(n::Int, sets::Vector{RegionDofs}, symmetric::Bool)
    nregions = sum(set -> length(set.ptr) - 1, sets; init=0)
    ptr = Vector{Int}(undef, nregions + 1)
    val = Vector{Int}(undef, sum(set -> length(set.val), sets; init=0))
    ptr[1] = 1
    r = 0
    for set in sets
        copyto!(val, ptr[r+1], set.val, 1, length(set.val))
        for k in 1:(length(set.ptr)-1)
            r += 1
            ptr[r+1] = ptr[r] + (set.ptr[k+1] - set.ptr[k])
        end
    end
    owners = copy(transpose(SparseMatrixCSC(n, nregions, ptr, val, fill(true, length(val)))))
    regions = rowvals(owners)
    marker = zeros(Int, n)
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for j in 1:n
        entries = 0
        for t in nzrange(owners, j), k in ptr[regions[t]]:(ptr[regions[t]+1]-1)
            i = val[k]
            ((symmetric && i < j) || marker[i] == j) && continue
            marker[i] = j
            entries += 1
        end
        colptr[j+1] = colptr[j] + entries
    end
    rowval = Vector{Int}(undef, colptr[n+1] - 1)
    fill!(marker, 0)
    @inbounds for j in 1:n
        pos = colptr[j]
        for t in nzrange(owners, j), k in ptr[regions[t]]:(ptr[regions[t]+1]-1)
            i = val[k]
            ((symmetric && i < j) || marker[i] == j) && continue
            marker[i] = j
            rowval[pos] = i
            pos += 1
        end
        sort!(view(rowval, colptr[j]:(colptr[j+1]-1)))
    end
    return AssemblyPattern(n, colptr, rowval, symmetric)
end

# The sparsity pattern of the matrix passes `passes` (those carrying a block),
# from the model's pattern cache, keyed by the passes' `RegionKey`s and the
# symmetry flag; `nothing` when there is no matrix pass. Loads add no matrix
# entry and are not part of the key, so `assemble!` on a volume-only problem and
# `assemble_matrix(model, mass_block(u))` share one entry, and a caller that
# alternates operators keeps every pattern it uses (up to four) instead of
# rebuilding one per call. A miss derives each list's region dofs (`_dofs!`) on
# `ws`, the caller's checked-out workspace, under the lock.
function _assembly_pattern!(@nospecialize(model::Model), passes::Vector{Any}, symmetric::Bool,
                            @nospecialize(ws))
    isempty(passes) && return nothing
    cache = model.assembly
    key = (RegionKey[pass.key for pass in passes], symmetric)
    Base.@lock cache.lock begin
        pattern = _lru_get!(cache.patterns, key)
        pattern === nothing || return pattern
        sets = RegionDofs[_dofs!(pass.list, ws) for pass in passes]
        return _lru_push!(cache.patterns,
                          key => _pattern(active_unknowns(model.dofs), sets, symmetric), 4)
    end
end

# Mirror a symmetric form's lower-triangular CSC arrays — `rowval` holds
# only rows `i ≥ j`, which is what `_pattern` emits under
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
# reference (safe because every assembly call allocates its own); a
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
# call, never per region, and specialising it on `Model{D,T,P}` would only
# compile it again for every problem type.
function _assembly_workspace(@nospecialize(model::Model))
    return AssemblyWorkspace(model.dofs, problem_spaces(model.problem))
end

# Take one call's scratch out of the model's cache: `tasks` workspaces, idle ones
# first and fresh ones for the rest, and for a threaded call the pooled arena
# pair, leaving empty arenas behind so that a concurrent call allocates its own.
# Scratch is never shared between calls; together with the locked lookups that
# is what makes concurrent assembly on one model safe, serial or threaded. A
# workspace costs about 4 µs to build, a sizable share of a small threaded call,
# hence the pool. Returns `(workspaces, arena, rhs_arena)`, with a concretely
# typed `workspaces` vector and the arenas `nothing` for a serial call.
function _checkout!(@nospecialize(cache::AssemblyCache), @nospecialize(model::Model), tasks::Int,
                    threaded::Bool)
    idle, arena, rhs_arena = Base.@lock cache.lock begin
        taken = Any[pop!(cache.workspaces) for _ in 1:min(tasks, length(cache.workspaces))]
        pooled, pooled_rhs = cache.arena, cache.rhs_arena
        if threaded
            cache.arena, cache.rhs_arena = similar(pooled, 0), similar(pooled_rhs, 0)
        end
        (taken, threaded ? pooled : nothing, threaded ? pooled_rhs : nothing)
    end
    layout, spaces = model.dofs, problem_spaces(model.problem)
    head = isempty(idle) ? AssemblyWorkspace(layout, spaces) : idle[1]
    return _fill_checkout(head, idle, tasks, layout, spaces), arena, rhs_arena
end

# Function barrier of `_checkout!`: `head` and then the other idle workspaces,
# topped up with fresh ones to `tasks`, in a vector of `head`'s concrete type, so
# the drivers index it without dispatch.
function _fill_checkout(head::W, idle::Vector{Any}, tasks::Int, layout, spaces) where {W}
    workspaces = Vector{W}(undef, tasks)
    workspaces[1] = head
    for t in 2:tasks
        workspaces[t] = t <= length(idle) ? idle[t]::W : AssemblyWorkspace(layout, spaces)::W
    end
    return workspaces
end

# Return a call's scratch to the cache: the workspaces, keeping at most
# `Threads.nthreads()` idle ones, and the larger of the returned and the resident
# arena pair, so the pool holds the maximum over calls, never their sum. Every
# per-region field of a workspace is reset before it is used again, so scratch
# returned from a call a callback error aborted is as good as any.
function _checkin!(@nospecialize(cache::AssemblyCache), @nospecialize(workspaces),
                   @nospecialize(arena), @nospecialize(rhs_arena))
    Base.@lock cache.lock begin
        for ws in workspaces
            length(cache.workspaces) < Threads.nthreads() && push!(cache.workspaces, ws)
        end
        if arena !== nothing
            length(arena) > length(cache.arena) && (cache.arena = arena)
            length(rhs_arena) > length(cache.rhs_arena) && (cache.rhs_arena = rhs_arena)
        end
    end
    return nothing
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
# sum over parents. That is the association of the field-evaluation helpers
# above; the basis values themselves reach the two along different routes, so
# the association is all they share. Returns `state`, which becomes `q.state`.
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
@noinline function _integrate!(ws::AssemblyWorkspace{D,T}, pass::Pass, region, offset::Int, state,
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

# Number of quadrature points of a region, which sizes its `q.point` range
# (`RegionList`) and drives the kernel's point loop. The physical-frame kinds
# (facet / surface / interface) share one method, as their weights vector
# already holds one entry per point.
_region_qpoint_count(region::VolumeRegion) = length(region.quadrature.weights)
function _region_qpoint_count(region::Union{FacetRegion,SurfaceRegion,InterfaceRegion})
    length(region.weights)
end

# ── Serial driver and the column kernel ───────────────────────────────────────

# Raised when the scatter meets a `(row, col)` the cached pattern does not
# contain — impossible unless the symbolic and numeric passes disagree about a
# region's active dofs. Raising turns a latent silent corruption (writing into a
# neighbouring slot) into an immediate, locatable failure. `@noinline` keeps the
# check off the hot path's instruction stream.
@noinline function _pattern_miss(row::Int, col::Int)
    error("assembly scatter: entry ($row, $col) is absent from the cached sparsity pattern; " *
          "the symbolic and numeric passes disagree on the active dofs")
end

# Add one column of a region's local matrix into the global `nz`: local column
# `lc` of a region whose sorted global ids are `dofs`, its stored rows
# `from:length(dofs)` read consecutively from `src`, starting at `src[s]`. This is
# the one association of local entries with pattern slots. The serial scatter
# (`src` the workspace's `K`) and threaded phase 2 (`src` the region's arena
# slice) both go through it, which is half of why the two are bit-identical.
#
# The region's rows are an ascending subset of the pattern column's, so one
# forward merge finds every slot: measured 2.7× faster than a binary search per
# entry, and a galloping merge gained nothing (0.99–1.00×) even on the 4 488-entry
# columns of a deep 3D ladder. Zero entries are skipped, as the scatter always
# has: Dirichlet column elimination leaves explicit zeros that `dropzeros!` then
# has less to remove, and the skip keeps the sum in every slot — down to the sign
# of a zero — what it has always been.
@inline function _scatter_column!(nz, pattern::AssemblyPattern, dofs, src, s::Int, lc::Int,
                                  from::Int)
    col = dofs[lc]
    p, hi = pattern.colptr[col], pattern.colptr[col+1] - 1
    @inbounds for lr in from:length(dofs)
        e = src[s]
        s += 1
        iszero(e) && continue
        row = dofs[lr]
        while p <= hi && pattern.rowval[p] < row
            p += 1
        end
        (p <= hi && pattern.rowval[p] == row) || _pattern_miss(row, col)
        nz[p] += e
    end
    return nothing
end

# The serial driver of one pass: integrate every region the filter admits, in
# list order, and move each into the global system. It is specialised on the
# pass, so `_integrate!` is a static call per region, and the move is
# `_flush!`, compiled once per workspace type. A fixed region order makes
# repeated serial assembly deterministic to the bit.
function _serial!(nz, rhs, pattern, pass::Pass, ws::AssemblyWorkspace, state, region_filter,
                  symmetric::Bool)
    for (r, region) in pairs(pass.list.regions)
        region_filter === nothing || region_filter(region) || continue
        n = _integrate!(ws, pass, region, pass.list.offsets[r], state, symmetric)
        _flush!(nz, rhs, pattern, ws, n, symmetric, !isempty(pass.blocks))
    end
    return nothing
end

# Move one integrated region's local system into the global one: the rhs entry of
# every local index, in local order, then — for a pass with blocks — every local
# column through `_scatter_column!`, rows `lc:n` under a symmetric form (the
# kernel never writes above the diagonal) and `1:n` otherwise. Form-independent
# and `@noinline`, so it compiles once per workspace type, not per form tuple.
@noinline function _flush!(nz, rhs, pattern, ws::AssemblyWorkspace, n::Int, symmetric::Bool,
                           matrix::Bool)
    for k in 1:n
        rhs[ws.dofs[k]] += ws.b[k]
    end
    matrix || return nothing
    m = n + length(ws.pivots)
    for lc in 1:n
        from = symmetric ? lc : 1
        _scatter_column!(nz, pattern, ws.dofs, ws.K, (lc - 1) * m + from, lc, from)
    end
    return nothing
end

# ── Threaded driver: deferred compute → gather ────────────────────────────────
#
# The threaded driver splits a pass into two phases that share no output slot,
# so neither needs a lock, an atomic accumulation, or a barrier between regions:
#
#   Phase 1 — tasks take regions from an atomic counter (dynamic load balance),
#     integrate each with the serial driver's own `_integrate!`, and copy its
#     rhs and stored matrix columns into the region's own slice of a flat arena.
#     No two regions share a slice.
#   Phase 2 — each task owns a contiguous range of global dofs (balanced by
#     pattern entries for a matrix pass, even for an rhs-only one) and walks
#     every region in list order, adding the region's entries in its owned rows
#     of the rhs and its owned columns of the matrix through the serial
#     scatter's `_scatter_column!`.
#
# Every slot then has exactly one writer. That writer visits the regions in
# list order and receives each region's entry at most once, bit for bit the
# value the serial driver computed, through the same column merge with the same
# zero skip. So threaded assembly is BIT-IDENTICAL to serial — not merely
# deterministic — and independent of the thread count, which removes the
# accumulation-order roundoff a colour- or atomic-ordered scatter leaves on an
# ill-conditioned system.
#
# The arena layout is a closed form of the region dof counts (`_packed`),
# evaluated by writer and reader alike, so nothing is planned or cached per
# list: the region dofs (`RegionDofs`) are the whole symbolic input. The arena
# holds `Σ n(n+1)/2` entries for a symmetric pass and `Σ n²` otherwise, whatever
# the thread count, and is pooled on the model between calls. The cost of
# walking every region in every phase-2 task measured 1.5–2 ns per region per
# task: nothing on 3D fixtures, 2.4 % of the wall time at 64 tasks on a 2D
# order-2 overlay, and 14 % only for 65 000 order-1 regions at 64 tasks, where a
# per-task region index would be the fix.

# Offset of local column `lc`'s first stored entry inside a region's packed
# block, for a region with `n` local dofs: column `c` stores rows `c:n` under a
# symmetric form and `1:n` otherwise. `_packed(n, n + 1, symmetric)` is the
# block's size.
function _packed(n::Int, lc::Int, symmetric::Bool)
    return symmetric ? (lc - 1) * n - ((lc - 1) * (lc - 2)) ÷ 2 : (lc - 1) * n
end

# Raised by phase 1 when a region's freshly numbered dofs differ from its cached
# `RegionDofs` slice: the threaded counterpart of `_pattern_miss`, which turns a
# symbolic/numeric disagreement into an error instead of a corrupted arena.
@noinline function _drift(r::Int)
    error("threaded assembly: the active dofs of region $r differ from its cached symbolic " *
          "list; the symbolic and numeric passes disagree")
end

# Run one pass threaded, accumulating into `nz` (`nothing` without a matrix) and
# `rhs`. `workspaces` and `states` hold one entry per task; `arena` and
# `rhs_arena` are the call's checked-out scratch, grown here when a pass needs
# more. Form-independent: `pass` is not specialised on, so this compiles once
# per workspace and state type, and the one form-specialised step, `_phase1!`,
# is reached by a single dynamic call. `done[r]` records the regions phase 1
# integrated; a region `region_filter` rejected stays `false` and phase 2 skips
# it, so a stale arena slice is never read.
function _threaded!(nz, rhs, pattern, @nospecialize(pass::Pass), workspaces, states, region_filter,
                    symmetric::Bool, arena, rhs_arena)
    dofs = pass.list.dofs::RegionDofs
    nregions = length(dofs.ptr) - 1
    matrix = !isempty(pass.blocks)
    offsets = Vector{Int}(undef, nregions + 1)
    offsets[1] = 1
    for r in 1:nregions
        n = dofs.ptr[r+1] - dofs.ptr[r]
        offsets[r+1] = offsets[r] + (matrix ? _packed(n, n + 1, symmetric) : 0)
    end
    length(arena) < offsets[end] - 1 && resize!(arena, offsets[end] - 1)
    length(rhs_arena) < dofs.ptr[end] - 1 && resize!(rhs_arena, dofs.ptr[end] - 1)
    done = fill(false, nregions)
    _phase1!(pass, workspaces, states, dofs, arena, rhs_arena, offsets, done, region_filter,
             symmetric)
    tasks = length(workspaces)
    owned = matrix ? _balanced_ranges(pattern.colptr, tasks) : _even_ranges(length(rhs), tasks)
    target = matrix ? nz : nothing
    @sync for range in owned
        Threads.@spawn _gather!(target, rhs, pattern, dofs, arena, rhs_arena, offsets, done, range,
                                symmetric)
    end
    return nothing
end

# Phase 1 of one pass: one task per workspace, each running `_compute!` until the
# shared region counter runs out. The only per-pass-type code of the threaded
# path besides `_compute!`; an error a callback raises in a task arrives wrapped
# (`TaskFailedException` inside a `CompositeException`), as from any `@sync`.
function _phase1!(pass::Pass, workspaces, states, dofs::RegionDofs, arena, rhs_arena,
                  offsets::Vector{Int}, done::Vector{Bool}, region_filter, symmetric::Bool)
    next = Threads.Atomic{Int}(1)
    @sync for t in eachindex(workspaces)
        Threads.@spawn _compute!(workspaces[t], states === nothing ? nothing : states[t], pass,
                                 dofs, arena, rhs_arena, offsets, done, next, region_filter,
                                 symmetric)
    end
    return nothing
end

# The phase-1 task body: take the next region, integrate it with the serial
# driver's own `_integrate!`, check that its dofs are its cached ones, and copy
# its rhs to `rhs_arena[dofs.ptr[r] …]` and each stored column `lc` to
# `arena[offsets[r] + _packed(n, lc, symmetric) …]`.
function _compute!(ws::AssemblyWorkspace, state, pass::Pass, dofs::RegionDofs, arena, rhs_arena,
                   offsets::Vector{Int}, done::Vector{Bool}, next::Threads.Atomic{Int},
                   region_filter, symmetric::Bool)
    regions = pass.list.regions
    while true
        r = Threads.atomic_add!(next, 1)
        r > length(regions) && return nothing
        region = regions[r]
        region_filter === nothing || region_filter(region) || continue
        n = _integrate!(ws, pass, region, pass.list.offsets[r], state, symmetric)
        lo = dofs.ptr[r]
        (n == dofs.ptr[r+1] - lo && view(dofs.val, lo:(lo+n-1)) == ws.dofs) || _drift(r)
        copyto!(rhs_arena, lo, ws.b, 1, n)
        if !isempty(pass.blocks)
            m = n + length(ws.pivots)
            for lc in 1:n
                from = symmetric ? lc : 1
                copyto!(arena, offsets[r] + _packed(n, lc, symmetric), ws.K, (lc - 1) * m + from,
                        n - from + 1)
            end
        end
        done[r] = true
    end
end

# The phase-2 task body for the dof range `owned`: every region phase 1 computed,
# in list order, adds its rhs entries of owned dofs and, for a matrix pass, its
# owned columns. A region's dofs are sorted, so whether it touches `owned` is one
# comparison of its first and last dof, and its owned local indices are one
# `searchsorted` range.
function _gather!(nz, rhs, pattern, dofs::RegionDofs, arena, rhs_arena, offsets::Vector{Int},
                  done::Vector{Bool}, owned::UnitRange{Int}, symmetric::Bool)
    c0, c1 = first(owned), last(owned)
    for r in 1:(length(dofs.ptr)-1)
        lo, hi = dofs.ptr[r], dofs.ptr[r+1] - 1
        (done[r] && lo <= hi && dofs.val[lo] <= c1 && dofs.val[hi] >= c0) || continue
        region = view(dofs.val, lo:hi)
        columns = searchsortedfirst(region, c0):searchsortedlast(region, c1)
        for k in columns
            rhs[region[k]] += rhs_arena[lo+k-1]
        end
        nz === nothing && continue
        n = length(region)
        for lc in columns
            from = symmetric ? lc : 1
            _scatter_column!(nz, pattern, region, arena, offsets[r] + _packed(n, lc, symmetric), lc,
                             from)
        end
    end
    return nothing
end

# Partition the columns `1:length(ptr)-1` of a CSC pattern into `tasks`
# contiguous ranges of roughly equal entry count, the last one running to the
# final column. Contiguous by construction, so the ranges own disjoint slots.
function _balanced_ranges(ptr::Vector{Int}, tasks::Int)
    m, total = length(ptr) - 1, ptr[end] - 1
    ranges = Vector{UnitRange{Int}}(undef, tasks)
    i = 1
    for t in 1:tasks
        lo = i
        while i <= m && ptr[i+1] - 1 < div(t * total, tasks)
            i += 1
        end
        hi = t == tasks ? m : min(i, m)
        ranges[t] = lo:hi
        i = hi + 1
    end
    return ranges
end

# Partition `1:n` into `tasks` contiguous ranges of near-equal length.
_even_ranges(n::Int, tasks::Int) = [(div((t-1)*n, tasks)+1):div(t*n, tasks) for t in 1:tasks]

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
  - `threaded` — assemble with the threaded driver, whose result is
    bit-identical to the serial one. Default is true when
    `Threads.nthreads() > 1`.
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
    matrix, _ = _assemble(model, block_tuple, (), symmetric_value, threaded, state, nothing)
    return _or_empty(matrix, model)
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
    _, rhs = _assemble(model, (), _form_tuple(loads), false, threaded, state, region_filter)
    return rhs::Vector{T}
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
    assembled, rhs = _assemble(model, model.problem.blocks, model.problem.loads, symmetric,
                               threaded, nothing, nothing)
    matrix = _or_empty(assembled, model)
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

# A call's matrix, or the empty `n × n` matrix when no pass carried a block
# (every block's target resolved to no region, or there was no block at all).
function _or_empty(matrix, model::Model{D,T}) where {D,T}
    n = active_unknowns(model.dofs)
    return (matrix === nothing ? spzeros(T, n, n) : matrix)::SparseMatrixCSC{T,Int}
end

# The core every assembly entry point calls: assemble `blocks` and `loads` (form
# tuples) on `model` and return `(matrix, rhs)`, the matrix `nothing` when no
# pass carries a block. The rhs includes the Dirichlet lift of every block, so
# the pair is consistent. `state` is `nothing`, a `Solution` or a coefficient
# vector; `region_filter` applies to the volume passes only.
#
# Deliberately unspecialised, like the rest of the cold path: it compiles once
# per process, not once per form tuple or problem type. Its handful of dynamic
# calls cost a few microseconds per call, and the one that matters is the
# dispatch per pass into `_serial!` or `_threaded!`, where specialisation on the
# forms begins.
#
# The per-call scratch is checked out of the model's cache and returned in a
# `finally`, so an error a callback raises still hands it back; every
# per-region field of a workspace is reset before it is reused, and phase 2 only
# reads arena slices phase 1 wrote in the same pass.
function _assemble(@nospecialize(model::Model), @nospecialize(blocks::Tuple),
                   @nospecialize(loads::Tuple), symmetric::Bool, threaded::Bool,
                   @nospecialize(state), @nospecialize(region_filter))
    coefficients = _state_vector(state, model)
    passes = _passes(model, blocks, loads)
    tasks = threaded ? Threads.nthreads() : 1
    cache = model.assembly
    workspaces, arena, rhs_arena = _checkout!(cache, model, tasks, threaded)
    try
        states = coefficients === nothing ? nothing :
                 [FormState(model.dofs, coefficients) for _ in 1:tasks]
        pattern = _assembly_pattern!(model, filter(pass -> !isempty(pass.blocks), passes),
                                     symmetric, first(workspaces))
        threaded && _dofs_all!(cache, passes, first(workspaces))
        T = _scalar(model.dofs)
        nz = pattern === nothing ? nothing : zeros(T, length(pattern.rowval))
        rhs = zeros(T, active_unknowns(model.dofs))
        for pass in passes
            # `region_filter` is a volume-pass convenience (compactly supported
            # sources); the facet, surface and interface passes ignore it.
            filter_pass = pass.key[1] === nothing ? region_filter : nothing
            if threaded
                # Every pass — volume, facet, surface, and two-sided interface —
                # runs threaded. One unexplained observation stands against that:
                # a two-sided `InterfaceRegion` pass for a vector coupling, under
                # `--code-coverage` at ≥2 threads, was nondeterministically
                # corrupted on both x86 and ARM. Never seen outside coverage, not
                # reproduced since across ~30,000 bit-exact comparisons, mechanism
                # not established (see the `couple` docstring); the parallel path
                # ships and `threaded=false` is the escape hatch.
                _threaded!(nz, rhs, pattern, pass, workspaces, states, filter_pass, symmetric,
                           arena, rhs_arena)
            else
                _serial!(nz, rhs, pattern, pass, first(workspaces),
                         states === nothing ? nothing : first(states), filter_pass, symmetric)
            end
        end
        return (pattern === nothing ? nothing : _matrix_from_pattern(pattern, nz)), rhs
    finally
        _checkin!(cache, workspaces, arena, rhs_arena)
    end
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
    on === nothing || return _region_list(model, on, _target_space(model, on, field)).offsets[end]
    kind === :volume && return sum(plan -> sum(_region_qpoint_count, plan.regions; init=0),
               integration_plans(model); init=0)
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
    return _walk(f, _volume_payload, _assembly_workspace(model),
                 _region_list(model, nothing, model.problem.space), _walk_state(state, model))
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

# Call `f(q)` at every quadrature point of `list`, in the serial assembly
# order, with the point's coordinates and weight computed exactly as the kernel
# computes them and `q.point = list.offsets[r] + k` its numbering. With a state, each
# region's dof values are read once and every field is evaluated at every point
# through the same `_frame!`, `_refresh!` and `_state_point!` the kernel uses;
# without one, no basis is evaluated at all. Behind a function barrier on the
# workspace, so the per-point calls are static.
function _walk(f, payload, ws::AssemblyWorkspace, list::RegionList, state)
    for (r, region) in pairs(list.regions)
        state === nothing || _state_region!(state, _frame!(ws, region))
        jacobian = _region_jacobian(region)
        for k in 1:_region_qpoint_count(region)
            x, w, ref = _region_point(region, k, jacobian)
            st = state === nothing ? nothing :
                 (_refresh!(ws, region, ref, Val(true)); _state_point!(state, ws, Val(true)))
            f(payload(region, k, x, w, list.offsets[r] + k, st))
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
    return _region_list(model, iface, nothing).offsets[end]
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
    # The frame and the refresh cover *both* sides' parents, so the state
    # evaluates each coupled field in its own subdomain's cut cell.
    return _walk(f, _interface_payload, _assembly_workspace(model),
                 _region_list(model, iface, nothing), _walk_state(state, model))
end
