# Two concerns share this file:
#
#   * **VTK export.** [`write_vtk`](@ref) produces a ParaView multiblock
#     bundle from a `(solution, model)` pair; [`write_quadrature_vtm`](@ref)
#     produces a quadrature-point cloud bundle for inspecting where the
#     integration regions land. Both rely on `WriteVTK.jl`.
#   * **Solution evaluation.** [`value`](@ref) and [`field_gradient`](@ref)
#     evaluate the superposed solution at arbitrary physical points;
#     [`l2_error`](@ref) computes an analytic / manufactured-solution
#     error using the model's own integration plan.
#
# Both subsystems share the same overlay-zero-extension convention: a
# level whose mesh does not cover the evaluation point contributes
# zero. That mirrors the assembly contract from
# `CONTRIBUTING.md`'s "Approximation space" section.

# ── VTK helpers ───────────────────────────────────────────────────────────────

# Default point-data callback for `write_vtk`: one entry named `uh`
# that evaluates the current solution at the sample point. Callbacks
# are called as `f(u, context, x, xi)` and may produce any per-point
# scalar, vector, or tuple value; the `u` closure is the access path
# back to the solution evaluation kernel.
_default_vtk_point_data() = (uh=(u, c, x, xi) -> u(c, xi),)

# Strip a `.vtm` / `.vtu` / `.vtp` extension from `path` so callers can
# pass either `"out"` or `"out.vtm"` interchangeably. `WriteVTK` adds
# the extension automatically.
function _vtk_base_path(path::AbstractString)
    base, ext = splitext(String(path))
    return ext in (".vtm", ".vtu", ".vtp") ? base : String(path)
end

# Compose the file path for a child VTK block by suffixing the base
# name. Used by the multiblock writer to derive paths for the solution
# `.vtu` and the per-level wireframe `.vtp`s.
function _vtk_child_path(base::AbstractString, suffix::AbstractString)
    return joinpath(dirname(base), basename(base) * "_" * suffix)
end

# VTK supports cells in 1D, 2D, and 3D natively. Higher-dimensional
# spaces have no corresponding VTK cell type and are rejected at the
# top of the public entry points so the user sees the dimension
# limitation up front.
function _check_vtk_dimension(::Val{D}) where {D}
    D <= 3 || throw(ArgumentError("VTK export supports dimensions 1, 2, and 3; got D=$D"))
end

# VTK cell type for a `D`-cube. `D = 1` is a line segment, `D = 2` is
# a four-vertex quad, `D = 3` is an eight-vertex hexahedron. Dispatched
# on `Val(D)` so the value is constant-folded into the cell
# construction.
_vtk_cell_type(::Val{1}) = VTKCellTypes.VTK_LINE
_vtk_cell_type(::Val{2}) = VTKCellTypes.VTK_QUAD
_vtk_cell_type(::Val{3}) = VTKCellTypes.VTK_HEXAHEDRON

# Corner-bit pattern in VTK's canonical vertex ordering for a `D`-cube.
# For each vertex `i ∈ 1:2ᴰ`, `_vtk_corner_bits(Val(D))[i]` is a tuple
# of `D` bits selecting `box.lower[d]` (bit 0) or `box.upper[d]`
# (bit 1) on axis `d`. The ordering follows the VTK convention
# (counterclockwise around each face, lower face before upper face)
# rather than a plain Cartesian iteration so the resulting `MeshCell`s
# orient correctly in ParaView.
_vtk_corner_bits(::Val{1}) = ((0,), (1,))
_vtk_corner_bits(::Val{2}) = ((0, 0), (1, 0), (1, 1), (0, 1))
function _vtk_corner_bits(::Val{3})
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
end

# Edge connectivity for the wireframe export: each `(a, b)` is a pair
# of corner indices (1-based, into `_vtk_corner_bits`) defining one
# edge of the `D`-cube. 2 edges in 1D, 4 in 2D, 12 in 3D.
_vtk_edge_pairs(::Val{1}) = ((1, 2),)
_vtk_edge_pairs(::Val{2}) = ((1, 2), (2, 3), (3, 4), (4, 1))
function _vtk_edge_pairs(::Val{3})
    ((1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5), (1, 5), (2, 6), (3, 7), (4, 8))
end

# Pad a `D`-dimensional physical point with trailing zeros to three
# coordinates: ParaView point arrays are always 3D regardless of the
# data's actual dimension. The padded zeros are inert (ParaView
# accepts them) and let 1D / 2D problems render in the same viewer
# pipeline as 3D ones.
_vtk_point(x::SVector{D,T}) where {D,T} = SVector{3,T}(ntuple(i -> i <= D ? x[i] : zero(T), 3))

# Collect the 2ᴰ corner coordinates of `b` in VTK canonical order.
# Used both for solution cells (one cell per subbox) and for wireframe
# edges (each edge connects two corners by `_vtk_edge_pairs` index).
function _box_corners(b::AxisBox{D,T}, ::Val{D}) where {D,T}
    return ntuple(Val(2^D)) do i
        bits = _vtk_corner_bits(Val(D))[i]
        SVector{D,T}(ntuple(d -> bits[d] == 0 ? b.lower[d] : b.upper[d], D))
    end
end

# Split an axis-aligned box into a `counts[1] × counts[2] × … × counts[D]`
# uniform grid of sub-boxes. Used to refine the VTK output beyond the
# raw admissible-box partition — a single high-order region gets
# subdivided so the linear ParaView cells can resolve the higher-order
# solution faithfully.
function _subboxes(b::AxisBox{D,T}, counts::NTuple{D,Int}) where {D,T}
    boxes = AxisBox{D,T}[]
    h = edge_lengths(b)
    for index in CartesianIndices(counts)
        lower = SVector{D,T}(ntuple(d -> b.lower[d] + h[d] * (index.I[d] - 1) / counts[d], D))
        upper = SVector{D,T}(ntuple(d -> b.lower[d] + h[d] * index.I[d] / counts[d], D))
        push!(boxes, AxisBox{D,T}(lower, upper))
    end
    return boxes
end

# Per-axis subbox count for one integration region. Four user-facing
# shapes:
#
#   * `:degree`           — `max(1, polynomial_order)` per axis,
#                           computed as the per-axis maximum over the
#                           region's parents. The natural default: a
#                           degree-`p` region is split into `p`
#                           subboxes so a piecewise-linear ParaView
#                           cell can resolve the basis without
#                           visible aliasing.
#   * `:none`             — no subdivision, one cell per region.
#   * positive integer    — uniform isotropic count.
#   * `NTuple{D,Int}`     — explicit per-axis counts.
function _subdivision_counts(space::Space{D,T}, region::VolumeRegion{D,T}, subdivisions) where {D,T}
    if subdivisions === :degree
        return ntuple(D) do d
            maximum(parent -> max(1, _level_by_id(space, parent.level).order[d]), region.parents)
        end
    elseif subdivisions === :none
        return ntuple(_ -> 1, D)
    elseif subdivisions isa Integer
        subdivisions > 0 || throw(ArgumentError("subdivisions must be positive"))
        return ntuple(_ -> Int(subdivisions), D)
    elseif subdivisions isa Tuple &&
           length(subdivisions) == D &&
           all(n -> n isa Integer && n > 0, subdivisions)
        return ntuple(d -> Int(subdivisions[d]), D)
    end

    throw(ArgumentError("subdivisions must be :degree, :none, a positive integer, or an NTuple{$D,Int}"))
end

# Normalise the `point_data` / `cell_data` keyword argument to a list
# of `String => callback` pairs. Three accepted user-facing shapes:
# `NamedTuple` (the most ergonomic; `(uh = …, σ = …)`),
# `AbstractDict` (programmatic construction), and `Tuple` of `Pair`s
# (explicit construction). The fallback raises with the offending
# parameter name (`:point_data` or `:cell_data`) embedded.
function _vtk_pairs(data, name::Symbol)
    if data isa NamedTuple
        return [String(k) => v for (k, v) in pairs(data)]
    elseif data isa AbstractDict
        return [String(k) => v for (k, v) in pairs(data)]
    elseif data isa Tuple && all(item -> item isa Pair, data)
        return [String(k) => v for (k, v) in data]
    end

    throw(ArgumentError("$name must be a NamedTuple, Dict, or tuple of Pairs"))
end

# ── VTK sample construction ──────────────────────────────────────────────────

# Build the `xi` payload of a VTK sample: the sample's region-reference
# coordinate `xi.region` and, per covering parent, the sample's
# coordinate in that parent's cell reference frame. This is the
# information user callbacks need to evaluate basis functions
# manually, e.g. for a custom field reconstruction.
function _point_xi(model::Model{D,T}, region::VolumeRegion{D,T}, x::SVector{D,T}) where {D,T}
    parents = map(region.parents) do parent
        (; level=parent.level, cell=parent.cell, xi=physical_to_reference(parent.parent_box, x))
    end
    return (; region=physical_to_reference(region.box, x), parents)
end

# Build the `(context, x, xi)` triple passed to a user VTK callback.
# `context` carries the integration region's identity, the subbox
# being sampled, the covering parents, and the model version (so a
# stale solution can be detected if needed). `location` is `:point`
# for vertex samples and `:cell` for cell-center samples.
function _vtk_sample(model::Model{D,T}, region_id::Int, region::VolumeRegion{D,T},
                     box::AxisBox{D,T}, x::SVector{D,T}, location::Symbol) where {D,T}
    context = (; location, region_id, region=region.box, cell=box, parents=region.parents,
               model_version=model.version,)
    return (; context, x, xi=_point_xi(model, region, x))
end

# Evaluate one user VTK callback at one sample. The `u(context, xi[, field])`
# accessor passed to the callback is a closure that lazily evaluates
# the superposed solution at the sample point — reusing the assembly
# parent-evaluation kernels through `_superposed_value_at` — so the
# callback can compute a scalar, vector, or tuple value without ever
# touching the dof layer directly.
# `space` / `block_layout` are the subdomain space and field layout of the block
# currently being written (for a single-domain model, the one space and the
# default field). The `u(context, xi[, field])` accessor evaluates the block's
# field by default; a named `field` is resolved on the *same* subdomain's
# parents (valid for several fields sharing one space). To read a field on a
# *different* subdomain, call `value(solution, model, other, x)` with the sample
# coordinate `x` the callback also receives.
function _evaluate_vtk_function(f, solution::Solution, model::Model, space::Space,
                                block_layout::FieldLayout, sample)
    u = (context, xi, field=nothing) -> begin
        layout = field === nothing ? block_layout : _model_field_layout(model, field)
        _superposed_value_at(solution.coefficients, space, layout, context.parents, xi.region)
    end
    return f(u, sample.context, sample.x, sample.xi)
end

# Coerce a `Vector{Any}` of callback returns to a typed array suitable
# for `WriteVTK`. Three accepted patterns:
#
#   * all-`Number` → `Vector{T}` with `T = promote_type(typeof.(v))`;
#   * all-`SVector` → typed vector of the first element's `SVector`
#     type (with conversion of each entry, in case widths or eltypes
#     differ slightly);
#   * all-`Tuple` of the same length → `Vector{SVector{N,T}}` with
#     `N` inferred from the first element and `T` promoted from every
#     component of every entry.
#
# The fallback returns the input untouched. `WriteVTK` will then
# typically error or produce a mixed-type array — the user gets a
# clear signal that their callback returned inconsistent shapes.
function _vtk_data_array(values::Vector{Any})
    isempty(values) && return Float64[]
    first_value = first(values)

    if all(v -> v isa Number, values)
        # Reduce the element types pairwise rather than splatting them into
        # `promote_type`: a `Vector{Any}` of thousands of samples would splat
        # thousands of type arguments and overflow the stack, because
        # variadic `promote_type` recurses one level per argument.
        T = mapreduce(typeof, promote_type, values)
        return T[values...]
    elseif all(v -> v isa SVector, values)
        S = typeof(first_value)
        return S[convert(S, v) for v in values]
    elseif first_value isa Tuple &&
           all(v -> v isa Tuple && length(v) == length(first_value), values)
        T = reduce(promote_type, (typeof(v[i]) for v in values for i in eachindex(first_value)))
        S = SVector{length(first_value),T}
        return S[S(v) for v in values]
    end

    return values
end

# Run every `name => callback` pair against every sample in the list,
# build a typed array of the returned values, and return the
# `name => array` pairs ready for `WriteVTK` attachment.
function _evaluate_vtk_data(pairs, samples, solution::Solution, model::Model, space::Space,
                           layout::FieldLayout)
    arrays = Pair{String,Any}[]
    for (name, f) in pairs
        values = Any[_evaluate_vtk_function(f, solution, model, space, layout, sample)
                     for sample in samples]
        push!(arrays, name => _vtk_data_array(values))
    end
    return arrays
end

# ── VTK partition and wireframes ─────────────────────────────────────────────

# Build the partitioned solution dataset: walk every integration
# region of the model, subdivide each into `_subboxes`, emit one VTK
# cell per subbox (with its 2ᴰ corners as VTK points), record
# `region_id` and `cover_count` per cell, and collect point-sample
# records for the user-supplied callbacks. The result is a
# `NamedTuple` ready for `vtk_grid` consumption.
#
# When the model carries a `PhysicalDomain`, the per-vertex level-set
# values `φ(x)` are collected as a built-in `level_set` point array —
# emitted alongside `region_id` / `cover_count` so cut, full, and
# fictitious regions are always identifiable in ParaView (e.g. as a
# zero-level isocontour) without the user having to wire a `phi`
# callback themselves. A user-supplied `level_set` entry in
# `point_data` overrides the built-in.
function _partition_vtk_data(solution::Solution, model::Model{D,T}, space::Space{D,T},
                             plan::IntegrationPlan{D,T}, layout::FieldLayout{D,T}, subdivisions,
                             point_data, cell_data) where {D,T}
    point_pairs = _vtk_pairs(point_data, :point_data)
    cell_pairs = _vtk_pairs(cell_data, :cell_data)
    physical = space.physical
    auto_level_set = physical !== nothing && !any(p -> first(p) == "level_set", point_pairs)

    points = SVector{3,T}[]
    cell_type = _vtk_cell_type(Val(D))
    cell0 = MeshCell(cell_type, SVector{2^D,Int}(ntuple(identity, 2^D)))
    cells = typeof(cell0)[]
    point_samples = Any[]
    cell_samples = Any[]
    region_ids = Int[]
    cover_counts = Int[]
    level_set_values = auto_level_set ? T[] : nothing

    for (region_id, region) in pairs(plan.regions)
        counts = _subdivision_counts(space, region, subdivisions)
        for subbox in _subboxes(region.box, counts)
            first_point = length(points) + 1
            corners = _box_corners(subbox, Val(D))
            for corner in corners
                push!(points, _vtk_point(corner))
                push!(point_samples, _vtk_sample(model, region_id, region, subbox, corner, :point))
                auto_level_set && push!(level_set_values, T(levelset_value(physical, corner)))
            end

            push!(cells,
                  MeshCell(cell_type, SVector{2^D,Int}(ntuple(i -> first_point + i - 1, 2^D))))
            push!(region_ids, region_id)
            push!(cover_counts, length(region.parents))
            push!(cell_samples,
                  _vtk_sample(model, region_id, region, subbox, center(subbox), :cell))
        end
    end

    point_arrays = _evaluate_vtk_data(point_pairs, point_samples, solution, model, space, layout)
    cell_arrays = _evaluate_vtk_data(cell_pairs, cell_samples, solution, model, space, layout)
    return (; points, cells, point_arrays, cell_arrays, region_ids, cover_counts, level_set_values)
end

# The fields a VTK write covers: a single explicit `field` (a `Field` or its
# name `Symbol`) if given, otherwise every field of the model — one grid block
# each. A single-domain model has exactly one field, reproducing the historical
# single-block output.
function _vtk_fields_to_write(model::Model, field)
    field === nothing && return collect(model.problem.fields)
    field isa Field && return [field]
    for f in model.problem.fields
        f.name === field && return [f]
    end
    throw(ArgumentError("unknown field $field"))
end

# The (subdomain space, its integration plan, its dof layout) a field is written
# against. For a single-domain model this is the one space / plan / field.
function _vtk_field_context(model::Model, fld::Field)
    space = _field_space(model.problem, fld.name)
    si = findfirst(s -> s === space, problem_spaces(model.problem))::Int
    return space, integration_plans(model)[si], _model_field_layout(model, fld)
end

# Map a level's `role` to a small integer tag for the wireframe export.
# ParaView filters can colour-by-`role_id` to visually distinguish
# base levels from overlays. Unknown roles map to `-1`.
function _role_id(role::Symbol)
    role === :base && return 0
    role === :overlay && return 1
    return -1
end

# Build the wireframe VTK dataset for one level: one `PolyData.Lines`
# segment per cell edge, with per-segment scalar fields
# `level_id`, `role_id`, `order_max`, `cell_id`. Inactive cells are
# skipped — the wireframe shows the level's *active region*, not its
# full mesh.
function _wireframe_vtk_data(level::Level{D,T}) where {D,T}
    points = SVector{3,T}[]
    line0 = MeshCell(PolyData.Lines(), SVector{2,Int}(1, 2))
    lines = typeof(line0)[]
    level_ids = Int[]
    role_ids = Int[]
    order_max = Int[]
    cell_ids = Int[]
    linear = LinearIndices(level.mesh.cells)

    for cell in cell_indices(level.mesh)
        # Inactive cells contribute nothing to the dof layout or
        # integration plan, so they have no business showing up in the
        # wireframe either. `is_active(nothing, _) === true`, so
        # unmasked levels (the base and any maskless overlay) keep
        # their full grid.
        is_active(level.mask, cell) || continue
        corners = _box_corners(cell_box(level.mesh, cell), Val(D))
        for (a, b) in _vtk_edge_pairs(Val(D))
            first_point = length(points) + 1
            push!(points, _vtk_point(corners[a]))
            push!(points, _vtk_point(corners[b]))
            push!(lines, MeshCell(PolyData.Lines(), SVector{2,Int}(first_point, first_point + 1)))
            push!(level_ids, level.id)
            push!(role_ids, _role_id(level.role))
            push!(order_max, maximum(level.order))
            push!(cell_ids, linear[cell])
        end
    end

    return (; points, lines, level_ids, role_ids, order_max, cell_ids)
end

# ── Public VTK export ────────────────────────────────────────────────────────

"""
    write_vtk(path, solution, model;
              field=nothing, subdivisions=:degree, point_data, cell_data,
              wireframes=true, ascii=false, append=true, compress=false)

Write a ParaView bundle rooted at `path`. The top-level file is a `.vtm`
multiblock dataset with **one block per field**, each named after the field and
grouping that field's data with its own mesh as sibling leaf datasets so it is a
self-contained, independently-toggled unit — `<field> → { data_<i>, level_… }`:

  - `data_<i>` — the `i`-th field's partitioned solution `.vtu`, whose cells are
    that field's admissible integration regions optionally subdivided into
    smaller sub-cells (see `subdivisions`), with per-cell `region_id` /
    `cover_count`, per-point user-defined arrays, and (with a
    [`PhysicalDomain`](@ref)) the field's own `level_set`. A single-field model
    names it simply `data`;
  - `level_<id>_<role>` — optionally, one wireframe `.vtp` per level of the
    field's subdomain showing the level's active region.

A single-field model is the one-block case of this shape. Fields sharing one
space emit that space's wireframes once, under the first such field. The tree is
homogeneous (a field block holds only leaf datasets) and every leaf's
disambiguating name is **ASCII** — the field *index* `i` for `data_<i>`, the
level *id* for the wireframes — so ParaView's Extract Block (whose data-assembly
node names keep only ASCII identifier characters) resolves each leaf on its own
even when the field names themselves are Unicode (`θ₁`, `θ₂`).

Keyword arguments:

  - `field` — restrict the output to a single [`Field`](@ref) (or its
    name `Symbol`); defaults to every field of the model.
  - `subdivisions` — per-region subdivision count. `:degree`
    (default) uses each region's parent polynomial order; `:none`
    keeps one cell per region; a positive integer uses an isotropic
    count; an `NTuple{D,Int}` uses explicit per-axis counts.
  - `point_data` — `NamedTuple` / `Dict` / `Tuple` of `name => callback`
    pairs. Each callback is invoked at every vertex sample as
    `f(u, context, x, xi)` and returns a per-point scalar / vector /
    tuple value. `u(context, xi[, field])` evaluates the current
    solution at the sample. Defaults to a single `uh` entry
    that emits the current solution value.
  - `cell_data` — same shape as `point_data` but evaluated at
    per-cell sample points (the subbox center). Defaults to no
    user-defined cell data; the bundle always includes `region_id`
    and `cover_count` as built-in cell arrays.
  - `wireframes` — emit one `.vtp` wireframe per level when `true`
    (default).
  - `ascii` / `append` / `compress` — pass-through to `WriteVTK`'s
    `vtk_grid` constructor.

For models whose [`Space`](@ref) carries a [`PhysicalDomain`](@ref),
the per-vertex level-set values `φ(x)` are automatically attached as a
built-in `level_set` point array — convenient for visualising `∂Ω`
as a zero-level isocontour or for filtering cut / full / fictitious
regions in ParaView. Pass a `level_set` entry in `point_data` to
override or rename it.
"""
function write_vtk(path::AbstractString, solution::Solution, model::Model{D,T}; field=nothing,
                   subdivisions=:degree, point_data=_default_vtk_point_data(),
                   cell_data=NamedTuple(), wireframes::Bool=true, ascii::Bool=false,
                   append::Bool=true, compress=false) where {D,T}
    _check_vtk_dimension(Val(D))
    _checked_coefficients(solution, model)
    base = _vtk_base_path(path)
    mkpath(dirname(base))

    # One grid block per field, each sampled over its own subdomain space. A
    # single-domain model has one field → a single `data` grid; a coupled model
    # writes one block per subdomain into the same multiblock file, so opening it
    # shows every coupled field.
    fields = _vtk_fields_to_write(model, field)
    multi = length(fields) > 1

    # One block per field, grouping the field's `data` grid with its own subdomain
    # mesh wireframe(s) so each coupled field is a self-contained, independently-
    # toggled unit in ParaView: `<field> → { data, level_… }`. The children of a
    # field block are all *leaf* datasets — never a mix of a leaf and a sub-block,
    # which breaks ParaView's Extract-Block resolution.
    #
    # Crucially, every leaf's *disambiguating* name part is ASCII. ParaView derives
    # its Extract-Block data-assembly node names from the dataset names and keeps
    # only ASCII identifier characters, so a Unicode field name (`θ₁`, `θ₂`) or a
    # name disambiguated only by Unicode digits collapses to a single node and the
    # blocks become indistinguishable. The wireframes are keyed by ASCII level id
    # (`level_<id>_<role>`); the data leaf is likewise keyed by the ASCII field
    # index (`data_<i>`), NOT by the field name. The field name is still the (only)
    # block label, where a Unicode collision is harmless — the user extracts leaves.
    # `meshed` dedups the wireframes of a space shared by several fields onto its
    # first.
    return vtk_multiblock(base) do vtm
        meshed = Any[]
        for (i, fld) in enumerate(fields)
            space, plan, layout = _vtk_field_context(model, fld)
            data = _partition_vtk_data(solution, model, space, plan, layout, subdivisions,
                                       point_data, cell_data)
            field_block = multiblock_add_block(vtm, string(fld.name))
            data_name = multi ? "data_$i" : "data"
            vtk = vtk_grid(_vtk_child_path(base, data_name), data.points, data.cells;
                           ascii, append, compress)
            multiblock_add_block(field_block, vtk, data_name)
            for (name, arr) in data.point_arrays
                vtk[name, VTKPointData()] = arr
            end
            if data.level_set_values !== nothing
                vtk["level_set", VTKPointData()] = data.level_set_values
            end
            vtk["region_id", VTKCellData()] = data.region_ids
            vtk["cover_count", VTKCellData()] = data.cover_counts
            for (name, arr) in data.cell_arrays
                vtk[name, VTKCellData()] = arr
            end

            (wireframes && !any(s -> s === space, meshed)) || continue
            push!(meshed, space)
            for level in space.levels
                wire = _wireframe_vtk_data(level)
                label = "level_$(level.id)_$(level.role)"
                wvtk = vtk_grid(_vtk_child_path(base, "$(label)_wire"), wire.points, wire.lines;
                                ascii, append, compress)
                multiblock_add_block(field_block, wvtk, label)
                wvtk["level_id", VTKCellData()] = wire.level_ids
                wvtk["role_id", VTKCellData()] = wire.role_ids
                wvtk["order_max", VTKCellData()] = wire.order_max
                wvtk["cell_id", VTKCellData()] = wire.cell_ids
            end
        end
    end
end

"""
    write_quadrature_vtm(path, model)

Write a ParaView multiblock bundle of the model's quadrature-point
cloud. One `.vtp` block is emitted per *distinct combination of
covering parent levels*: every region in the integration plan is
keyed by `sort([parent.level for parent in region.parents])` and
regions with the same level signature land in the same block. Each
block stores the physical quadrature-point coordinates and per-point
physical weight (`weight = w × jacobian`) as point data.

Useful for inspecting where the integration regions actually land in
ParaView (the regions / cuts / α-FCM weights are not visible from the
solution VTK alone) and for diagnosing small-overlap warnings — the
small regions show up as point clusters with tiny weights.
"""
function write_quadrature_vtm(path::AbstractString, model::Model{D,T}) where {D,T}
    _check_vtk_dimension(Val(D))
    base = _vtk_base_path(path)
    mkpath(dirname(base))

    # Regions of EVERY subdomain integration plan. The reindexed level ids are
    # globally unique, so a region's covering-level signature already separates
    # subdomains into distinct blocks — a coupled model shows every subdomain's
    # quadrature cloud, not just the first.
    all_regions = VolumeRegion{D,T}[]
    for plan in integration_plans(model)
        append!(all_regions, plan.regions)
    end

    # Group regions by the sorted list of covering parent-level ids,
    # then emit one VTP block per group.
    groups = Dict{Vector{Int},Vector{Int}}()
    for (i, region) in enumerate(all_regions)
        sig = sort!([p.level for p in region.parents])
        push!(get!(groups, sig, Int[]), i)
    end

    return vtk_multiblock(base) do vtm
        for sig in sort!(collect(keys(groups)))
            region_indices = groups[sig]
            points = SVector{3,T}[]
            verts = MeshCell{PolyData.Verts,SVector{1,Int}}[]
            weights = T[]
            for i in region_indices
                region = all_regions[i]
                jacobian = volume(region.box) / convert(T, 2^D)
                for (eta, w) in zip(region.quadrature.points, region.quadrature.weights)
                    x = reference_to_physical(region.box, eta)
                    push!(points, _vtk_point(x))
                    push!(verts, MeshCell(PolyData.Verts(), SVector{1,Int}(length(points))))
                    push!(weights, w * jacobian)
                end
            end
            suffix = isempty(sig) ? "none" : join(sig, "_")
            vtk = vtk_grid(_vtk_child_path(base, "quadrature_levels_" * suffix), points, verts)
            multiblock_add_block(vtm, vtk, "levels_" * suffix)
            vtk["weight", VTKPointData()] = weights
        end
    end
end

# ── Solution evaluation ──────────────────────────────────────────────────────

# Look up the per-field layout of `field` inside the model's
# `SystemLayout`. Thin wrapper over `_field_layout` from `dofs.jl`
# that takes a `Field` value rather than a name symbol.
_model_field_layout(model::Model, field::Field) = _field_layout(model.dofs, field.name)

# Pick the implicit field of a single-field model. Multi-field models
# require an explicit field argument so the caller does not silently
# evaluate the wrong field; this raises with a clear message in that
# case.
function _default_field(model::Model)
    length(model.problem.fields) == 1 ? first(model.problem.fields) :
    throw(ArgumentError("field argument is required for multi-field models"))
end

# Convert a `PointLike` user input (NTuple or SVector) to a typed
# `SVector{D,T}` matching the model's scalar type. Used at the top of
# every public evaluation path so the downstream kernels see a single
# canonical point representation.
_point_vector(x::PointLike{D}, ::Type{T}) where {D,T} = SVector{D,T}(x)

# Reject evaluation points outside the physical domain. The check uses
# the model's own geometry tolerance, so points on the domain
# boundary (up to `tol.contain`) are accepted.
function _assert_point_in_domain(x::SVector{D,T}, model::Model{D,T}) where {D,T}
    _assert_point_in_domain(x, model.problem.space.domain, model.dofs.tolerance)
end

# Domain-based overload: a coupled model evaluates each field against its own
# subdomain's bounding box, not the representative space's.
function _assert_point_in_domain(x::SVector{D,T}, domain::AxisBox{D,T},
                                 tol::GeometryTolerance{T}) where {D,T}
    contains_point(x, domain, tol) ||
        throw(ArgumentError("evaluation point is outside the field's domain"))
end

# One level's contribution to the superposed value at physical point
# `x`. Returns zero if the level's mesh does not contain `x` (overlay
# zero-extension) or if the containing cell is masked inactive.
# Otherwise: locate the cell, map `x` to the cell's reference frame,
# evaluate basis values there, and contract with the dof coefficients
# through `dof_value` so constrained dofs contribute their pinned
# values.
function _level_value(coefficients, model::Model{D,T}, layout::FieldLayout{D,T}, level::Level{D,T},
                      x::SVector{D,T}, component::Integer=1) where {D,T}
    cell = locate_cell(level.mesh, x; tol=model.dofs.tolerance)
    cell === nothing && return zero(promote_type(T, eltype(coefficients)))
    is_active(level.mask, cell) || return zero(promote_type(T, eltype(coefficients)))

    parent_box = cell_box(level.mesh, cell)
    xi = physical_to_reference(parent_box, x)
    values = basis_values(level.basis, level.order, level.mode, xi, cell)
    raw_dofs = cell_dofs(layout.dofs, level.id, cell)
    # Reuse the assembly reconstruction kernel: the dof sum over
    # `(raw_dofs, values)` is exactly what `_field_value` computes (constrained
    # dofs resolved through `dof_value`).
    return _field_value((; raw_dofs, values), layout, coefficients, component)
end

# Gradient analogue of `_level_value`. Uses
# `physical_basis_gradients` so the chain-rule scaling `2 / h_d` for
# the axis-aligned cell is applied automatically — the gradient
# returned is already in physical coordinates.
function _level_gradient(coefficients, model::Model{D,T}, layout::FieldLayout{D,T},
                         level::Level{D,T}, x::SVector{D,T}, component::Integer=1) where {D,T}
    R = promote_type(T, eltype(coefficients))
    cell = locate_cell(level.mesh, x; tol=model.dofs.tolerance)
    cell === nothing && return SVector{D,R}(ntuple(_ -> zero(R), D))
    is_active(level.mask, cell) || return SVector{D,R}(ntuple(_ -> zero(R), D))

    parent_box = cell_box(level.mesh, cell)
    xi = physical_to_reference(parent_box, x)
    gradients = physical_basis_gradients(level.basis, level.order, level.mode, parent_box, xi, cell)
    raw_dofs = cell_dofs(layout.dofs, level.id, cell)
    # Reuse the assembly gradient-reconstruction kernel over `(raw_dofs,
    # gradients)`; the chain-rule scaling is already in `gradients`.
    return _field_gradient((; raw_dofs, gradients), layout, coefficients, component)
end

# Shared pre-flight for the public `value` / `field_gradient` paths:
# coefficient check, point coercion, in-domain check, field-layout
# lookup. Returns the triple every evaluator needs.
function _evaluation_data(solution::Solution, model::Model{D,T}, u::Field,
                          x::PointLike{D}) where {D,T}
    coefficients = _checked_coefficients(solution, model)
    point = _point_vector(x, T)
    space = _field_space(model.problem, u.name)
    _assert_point_in_domain(point, space.domain, model.dofs.tolerance)
    return coefficients, point, _model_field_layout(model, u)
end

# Bounds-check a component argument against the field's component count.
function _check_component(layout::FieldLayout, component::Integer)
    1 <= component <= layout.components || throw(ArgumentError("field component out of bounds"))
end

# Levels of the space owning `layout`'s field. For a single-domain model this
# is the one space's levels; for a coupled model it routes each field to its
# own subdomain's (reindexed) levels, so a point evaluation of field `b` never
# reaches into field `a`'s grid.
_field_levels(model::Model, layout::FieldLayout) = _field_space(model.problem, layout.name).levels

# Sum every level's `_level_value` contribution. The overlay
# zero-extension comes from `_level_value` returning zero when its
# level does not cover `point`.
function _evaluate_field_value(coefficients, model::Model{D,T}, layout::FieldLayout{D,T},
                               point::SVector{D,T}, component::Integer=1) where {D,T}
    result = zero(promote_type(T, eltype(coefficients)))
    for level in _field_levels(model, layout)
        result += _level_value(coefficients, model, layout, level, point, component)
    end
    return result
end

# Gradient counterpart of `_evaluate_field_value`.
function _evaluate_field_gradient(coefficients, model::Model{D,T}, layout::FieldLayout{D,T},
                                  point::SVector{D,T}, component::Integer=1) where {D,T}
    R = promote_type(T, eltype(coefficients))
    result = SVector{D,R}(ntuple(_ -> zero(R), D))
    for level in _field_levels(model, layout)
        result += _level_gradient(coefficients, model, layout, level, point, component)
    end
    return result
end

# Component dispatch shared by `value` and `field_gradient`. When `component
# === nothing`: return a scalar for single-component fields and an
# `SVector` across components for vector fields. When `component` is
# explicit: bounds-check and evaluate that one component.
function _field_quantity(solution::Solution, model::Model{D,T}, u::Field, x::PointLike{D},
                         component, evaluator) where {D,T}
    coefficients, point, layout = _evaluation_data(solution, model, u, x)
    if component === nothing
        layout.components == 1 && return evaluator(coefficients, model, layout, point)
        return SVector(ntuple(c -> evaluator(coefficients, model, layout, point, c),
                              layout.components))
    end

    _check_component(layout, component)
    return evaluator(coefficients, model, layout, point, component)
end

"""
    value(solution, model, x)
    value(solution, model, x, component)
    value(solution, model, u::Field, x[, component])

Evaluate the superposed solution at physical point `x`. Scalar fields
return a scalar; component fields return an `SVector` across
components unless a single `component` is requested. Overlay levels
are extended by zero — only levels whose mesh contains `x` (and
whose covering cell is active) contribute. Throws if `x` lies outside
the physical domain.

`x` accepts `NTuple{D,<:Real}` or `SVector{D,<:Real}`. Multi-field
models require the `u::Field` form so the field to evaluate is
unambiguous.
"""
function value(solution::Solution, model::Model{D,T}, x::PointLike{D}) where {D,T}
    return value(solution, model, _default_field(model), x)
end

function value(solution::Solution, model::Model{D,T}, x::PointLike{D},
               component::Integer) where {D,T}
    return value(solution, model, _default_field(model), x, component)
end

function value(solution::Solution, model::Model{D,T}, u::Field, x::PointLike{D}) where {D,T}
    _field_quantity(solution, model, u, x, nothing, _evaluate_field_value)
end

function value(solution::Solution, model::Model{D,T}, u::Field, x::PointLike{D},
               component::Integer) where {D,T}
    _field_quantity(solution, model, u, x, component, _evaluate_field_value)
end

"""
    field_gradient(solution, model, x)
    field_gradient(solution, model, x, component)
    field_gradient(solution, model, u::Field, x[, component])

Evaluate the physical gradient of the superposed solution at physical
point `x`. Scalar fields return one `SVector{D}` gradient; component
fields return an `SVector` of component gradients unless a single
`component` is requested. Overlay levels are extended by zero, same
as [`value`](@ref). The leading `field_` avoids colliding with
`Tensors.gradient` when both packages are loaded together.

The gradient is the physical gradient (chain-rule scaled by
`2 / edge_lengths(parent_box)` per axis), not the reference-frame
gradient.
"""
function field_gradient(solution::Solution, model::Model{D,T}, x::PointLike{D}) where {D,T}
    return field_gradient(solution, model, _default_field(model), x)
end

function field_gradient(solution::Solution, model::Model{D,T}, x::PointLike{D},
                        component::Integer) where {D,T}
    return field_gradient(solution, model, _default_field(model), x, component)
end

function field_gradient(solution::Solution, model::Model{D,T}, u::Field,
                        x::PointLike{D}) where {D,T}
    _field_quantity(solution, model, u, x, nothing, _evaluate_field_gradient)
end

function field_gradient(solution::Solution, model::Model{D,T}, u::Field, x::PointLike{D},
                        component::Integer) where {D,T}
    _field_quantity(solution, model, u, x, component, _evaluate_field_gradient)
end

# ── L² error and superposed evaluation ───────────────────────────────────────

# Pointwise squared magnitude that works for both scalar fields
# (`abs2(value)`) and vector / tuple / SVector fields
# (`sum(abs2, value)`). Used by `l2_error` so the error norm
# generalises to multi-component results without per-component
# dispatch at the call site.
_squared_norm(value::Number) = abs2(value)
_squared_norm(value) = sum(abs2, value)

# Superposed value over already-evaluated per-parent records. Scalar
# fields return a scalar; component fields return an `SVector` over
# the components. Reuses the assembly parent-evaluation kernels
# (`_field_value` from `assembly.jl`) so we don't re-implement the dof
# reduction.
function _superposed_value(field_data, layout::FieldLayout, coefficients)
    layout.components == 1 && return _field_value(field_data, layout, coefficients, 1)
    return SVector(ntuple(c -> _field_value(field_data, layout, coefficients, c),
                          layout.components))
end

# Evaluate the superposed value at a region-reference point `eta`,
# allocating fresh per-parent basis records and refreshing them in
# place. Used by `_evaluate_vtk_function` to expose `u(context, xi)`
# to user VTK callbacks; `l2_error` uses the same kernels but reuses
# its own per-region records to avoid re-allocating per quadrature
# point.
function _superposed_value_at(coefficients, space::Space, layout::FieldLayout, parents, eta)
    field_data = _field_parent_data(space, layout, parents; gradients=false)
    for data in field_data
        _update_parent_basis_values!(data, eta)
    end
    return _superposed_value(field_data, layout, coefficients)
end

"""
    l2_error(solution, model, exact; norm=:relative) -> Real

Compute the L² error of `solution` against an analytic / manufactured
solution `exact(x)`:

  - `norm = :relative` (default) returns
    `sqrt(∫ ‖u_h − u_exact‖² dx) / sqrt(∫ ‖u_exact‖² dx)`;
  - `norm = :absolute` returns just the numerator;
  - if `‖u_exact‖_L² = 0`, the relative form falls back to the
    absolute error so it stays meaningful for zero-reference checks
    (e.g. a fabricated test where `exact(x) ≡ 0`).

`exact(x)` may return a scalar or a component value (e.g. an
`SVector` or tuple); the `_squared_norm` helper handles both.

The integration uses the model's own integration plan — the same
admissible regions assembly walks — so the error is consistent with
what the solver actually integrated. For SPD model problems this is
the natural convergence-study metric: monotonic decrease as you
refine, with the rate matching the basis order.
"""
function l2_error(solution::Solution, model::Model{D,T}, exact; norm::Symbol=:relative) where {D,T}
    norm in (:relative, :absolute) || throw(ArgumentError("norm must be :relative or :absolute"))
    coefficients = _checked_coefficients(solution, model)
    field_layout = _model_field_layout(model, _default_field(model))
    plan = integration_plan(model)
    error_squared = zero(T)
    exact_squared = zero(T)

    for region in plan.regions
        field_data = _field_parent_data(model.problem.space, field_layout, region.parents;
                                        gradients=false)
        quadrature = region.quadrature
        jacobian = volume(region.box) / convert(T, 2^D)

        for (eta, weight) in zip(quadrature.points, quadrature.weights)
            for data in field_data
                _update_parent_basis_values!(data, eta)
            end
            exact_value = exact(reference_to_physical(region.box, eta))
            diff = _superposed_value(field_data, field_layout, coefficients) - exact_value
            qweight = weight * jacobian
            error_squared += qweight * _squared_norm(diff)
            exact_squared += qweight * _squared_norm(exact_value)
        end
    end

    error_norm = sqrt(error_squared)
    norm === :absolute && return error_norm

    exact_norm = sqrt(exact_squared)
    return iszero(exact_norm) ? error_norm : error_norm / exact_norm
end

# ── Boundary integration ─────────────────────────────────────────────────────
#
# Quadrature-only postprocessing on a portion of the boundary. The
# region list comes from `_resolve_on_regions(model, on)` so the same
# code path handles physical [`FacetRegion`](@ref)s and immersed
# [`SurfaceRegion`](@ref)s; the per-region `_boundary_q` helper builds
# the right `q` tuple for each kind. No per-Q-point geometry work
# happens in this hot path — every region carries precomputed
# physical-frame Q-points and weights.

"""
    boundary_integral(integrand, model::Model; on) -> result

Integrate a user callback over a portion of the boundary.

`on` selects the integration region:

  - `boundary(:all)`, `boundary(axis=d, side=s)`,
    `boundary((axis=…, side=…), …)` — the physical boundary `∂Ω`. The
    Gauss rule per region is the per-axis maximum
    `recommended_quadrature_order(level.basis, level.order)` over the
    region's covering parents. Overlay levels whose own mesh face
    coincides with the selected facet contribute their own segment;
    level masking is respected.
  - A [`BoundaryMesh`](@ref) — a user-supplied immersed-boundary mesh
    (segments in 2D, triangles in 3D, points in any D). The package
    walks the precomputed per-cell quadrature rules.

`integrand(q)` is called at every quadrature point with a named tuple:

  - `q.x::SVector{D,T}` — physical coordinate.
  - `q.weight::T` — physical-frame weight (Gauss × measure Jacobian).
  - `q.normal::SVector{D,T}` — outward unit normal (facet's outward
    normal for `BoundarySelector` `on`; cell-geometry / user-supplied
    normal for `BoundaryMesh`).
  - `q.sides::Union{Nothing,Vector{Tuple{Int,Symbol}}}` — the codim-`K`
    facet identifier on a `BoundarySelector` integration; `nothing` on
    a `BoundaryMesh` integration (no facet `(axis, side)` exists).

The accumulator is `result += q.weight * integrand(q)`; the result
type is whatever `q.weight * integrand(q)` yields on the first
sample. Throws `ArgumentError` if no admissible region exists on the
selected portion of the boundary.
"""
function boundary_integral(integrand, model::Model; on)
    regions = _resolve_on_regions(model, on)
    result = nothing
    samples = 0
    for region in regions
        for (qp, x) in pairs(region.points)
            q = _boundary_q(region, qp, x)
            contribution = q.weight * integrand(q)
            result = result === nothing ? contribution : result + contribution
            samples += 1
        end
    end
    samples == 0 && throw(ArgumentError("no admissible boundary regions for on=$(on); " *
                                        "check the selector or mesh and any level masks"))
    return result
end

# Per-Q-point `q` tuple for facet vs. surface regions. The shape is
# kept identical across kinds — every `q` carries `x`, `weight`,
# `normal`, and `sides`; `sides === nothing` on surface regions.
function _boundary_q(region::FacetRegion, qp::Int, x)
    (; x, weight=region.weights[qp], normal=region.normal, sides=region.sides)
end
function _boundary_q(region::SurfaceRegion, qp::Int, x)
    (; x, weight=region.weights[qp], normal=region.normals[qp], sides=nothing)
end
