# Post-processing: what every consumer of a solved model shares, in two parts.
#
#   * **Sampling.** A solution, the meshes of its levels and the quadrature-point cloud are sampled into
#     consumer-neutral records, so that every consumer reads the same geometry and the same arrays rather than
#     re-deriving them. The contract is below.
#   * **Solution evaluation.** [`value`](@ref) and [`field_gradient`](@ref) evaluate the superposed solution at
#     arbitrary physical points; [`l2_error`](@ref) computes an analytic / manufactured-solution error over the model's
#     own integration plan; [`boundary_integral`](@ref) integrates a callback over a portion of the boundary or an
#     interface.
#
# All of them share the overlay-zero-extension convention: a level whose mesh does not cover a point contributes zero
# there. That mirrors the assembly contract from `CONTRIBUTING.md`'s "The superposition model" section.
#
# The consumers are package extensions. The one shipped is the VTK writer, `ext/UnfittedWriteVTKExt.jl`, which
# `using WriteVTK` loads: [`write_vtk`](@ref) serialises the records of a `(solution, model)` pair as a ParaView
# multiblock bundle, [`vtk_series`](@ref) collects a sequence of those bundles into a `.pvd` collection, so a feature
# that moves — and the mesh that follows it — can be watched over time, and [`write_quadrature_vtm`](@ref) writes the
# quadrature-point cloud, for inspecting where the integration regions land. Everything VTK-specific — cell types,
# padding to three components, the number types VTK stores, the multiblock layout, the file names — is decided there and
# nowhere else; the entry points themselves are declared and documented in `Unfitted.jl`.
#
# ## The record contract
#
# A `SampleGrid{D,T,K}` holds cells of dimension `K` in `D` dimensions, in the number type `T`. The cell vocabulary has
# two words: `K = D` is a set of axis-aligned `D`-cubes (a line, a quad, a hexahedron), `K = 0` a vertex cloud, one
# point per cell.
#
#   * **Points.** `points::Vector{SVector{D,T}}`, unpadded. Cell `c` owns points `(c − 1)·2ᴷ + 1 : c·2ᴷ` and no point
#     is shared between cells, so the connectivity is implicit and `_cell_count(g) == length(g.points) >> K`. The 2ᴰ
#     corners of a cube come in face-cyclic order (`_box_corners`): the lower face counterclockwise, then the upper
#     face — VTK's order for a line, a quad and a hexahedron, and a GeometryBasics `QuadFace`'s, so a quad splits into
#     the triangles (1, 2, 3) and (1, 3, 4).
#   * **Arrays.** `point_data` and `cell_data` are `name => array` lists in attach order, one value per point or per
#     cell. An array is a `Vector{R}` with `R <: Real` or a `Vector{SVector{N,R}}`, unpadded and in the element type it
#     was computed in, `Bool` included: a consumer maps what its format cannot hold (the VTK writer pads a 2-vector to
#     three components and writes a `Bool` as `UInt8`).
#
# Two internal entry points produce records, on demand and without caching them or touching the file system:
#
#   * `_sample_fields(solution, model; field, subdivisions, cut_depth, point_data, cell_data, level_meshes)`, whose
#     keywords mean what `write_vtk` documents — one `(; field, data, meshes)` per sampled field, in problem order.
#     `data::SampleGrid{D,T,D}` covers the integration regions the field was solved on, each region its `subdivisions`
#     lattice with the boxes ∂Ω crosses cut-refined; its point arrays are the user's `point_data` entries in their
#     order, then `level_set` (with a `PhysicalDomain`, unless an entry already has that name); its cell arrays are
#     `region_id` and `cover_count`, then the user's `cell_data` entries. `meshes` holds one `(; level, role, grid)`
#     per level of the field's space, in `space.levels` order — empty when `level_meshes = false`, or when an earlier
#     record already carries the space — and each `grid` covers every cell of its level, active or not, with the `Int`
#     cell arrays `level_id`, `role_id`, `order_max`, `cell_id`, `active`, `covered`, `active_dofs`, `reduced_dofs`.
#   * `_sample_quadrature(model)` — one `(; signature, grid::SampleGrid{D,T,0})` per sorted covering-level signature, in
#     sorted order, with the point array `weight`.
#
# A plotting extension would read the same records. With Makie, a 2D grid `g` coloured by its first point array, and
# a quadrature-cloud entry `e` coloured by its weights, are
#
#     faces = [QuadFace(4c - 3, 4c - 2, 4c - 1, 4c) for c in 1:_cell_count(g)]
#     mesh(Point.(g.points), faces; color=last(g.point_data[1]))
#     scatter(Point.(e.grid.points); color=last(only(e.grid.point_data)))
#
# Its public functions would be declared and documented in `Unfitted.jl`, as the VTK writer's are, each a fallback
# method whose body is `_extension_missing`, so that a call made without the plotting package loaded says which package
# to load.

# ── Sample records ────────────────────────────────────────────────────────────

# One sampled dataset in a consumer-neutral form: the record every post-processing consumer reads, so that a VTK writer
# and a plotting front end share the geometry and the arrays instead of each re-deriving them. `K` is the cell
# dimension — `D` for a grid of `D`-cubes, `0` for a vertex cloud — and the connectivity is implicit: cell `c` owns
# points `(c − 1)·2ᴷ + 1 : c·2ᴷ`, in the face-cyclic order of `_box_corners`, and no point is shared between cells.
# Points are unpadded `D`-vectors and every array is unpadded with its element type as computed (`Bool` included),
# point arrays one value per point, cell arrays one per cell, each list in attach order: what a format needs on top —
# three coordinates, a known number type — its consumer adds.
struct SampleGrid{D,T,K}
    points::Vector{SVector{D,T}}
    point_data::Vector{Pair{String,AbstractVector}}
    cell_data::Vector{Pair{String,AbstractVector}}
end

# Number of cells of a record: its points come in blocks of 2ᴷ.
_cell_count(g::SampleGrid{D,T,K}) where {D,T,K} = length(g.points) >> K

# The 2ᴰ corners of `b` in face-cyclic order: the lower face counterclockwise, then the upper face the same way — the
# order of VTK's line, quad and hexahedron and of a GeometryBasics `QuadFace`, so a consumer reads a cell's points as a
# closed polygon or a hexahedron without permuting them. It is a fixed permutation of `_box_vertices`, whose axis-1
# fastest order swaps the last two corners of every face: corner `j` is vertex `xor(j − 1, ((j − 1) >> 1) & 1) + 1`,
# which is (1, 2, 4, 3) in 2D and (1, 2, 4, 3, 5, 6, 8, 7) in 3D. Every coordinate is `lower[d]` or `upper[d]` itself,
# so two boxes whose shared face has the same coordinates — neighbours in the lattice of one region, or among the leaves
# of one cut tree — get bit-identical corner points on it; across regions the lattices are computed apart and may differ
# in the last bit. Dimension-generic: which `D` a format can hold is its consumer's question.
function _box_corners(b::AxisBox{D}) where {D}
    v = _box_vertices(b)
    return ntuple(j -> v[xor(j - 1, ((j - 1) >> 1) & 1)+1], Val(2^D))
end

# ── Field sampling ────────────────────────────────────────────────────────────

# The records of a solution's data: one `(; field, data, meshes)` per sampled field, in output order — every field in
# problem order, or the one `field` names. `data` is the field's sample grid (`_sample_field`). `meshes` holds the
# `(; level, role, grid)` records of its space's levels, in `space.levels` order (`_level_grid`); it is empty with
# `level_meshes = false`, and for a space an earlier record already carries, so fields sharing a space have its meshes
# once, under the first of them, whose layout the dof counts are read from — the convention `diagnostics` follows for
# its per-level counts. That is decided here rather than by each consumer, which would otherwise have to repeat it.
#
# The one entry of the sampling layer and its one keyword surface: a writer passes every keyword it does not consume
# on to this function, and the keywords mean what `write_vtk` documents. The keywords, the solution's pin and the
# entries' containers are checked before anything is sampled, a callback's values as they are collected, and nothing
# touches the file system. One workspace and one `FormState` serve every field: the fields are sampled one after
# another, and each field's accessor frames the regions it reads on its own (`_sample_accessor`).
#
# The records are collected by an explicit loop into a vector of their concrete type, so a consumer reads them
# type-stably whatever inference made of the workspace, whose type is not inferable from the model
# (`_assembly_workspace`): a `map` over the fields gave the records the element type inferred for its closure, and an
# abstract one sent inference through the whole serialiser with the grids' dimension and cell kind unknown — measured
# on the first `write_vtk` of a two-level 2D overlay, 2721 code instances inferred and 549 methods compiled, against
# 1549 and 406 with the loop.
function _sample_fields(solution::Solution, model::Model{D,T}; field=nothing, subdivisions=:degree, cut_depth=3,
                        point_data=_default_point_data(), cell_data=NamedTuple(), level_meshes::Bool=true) where {D,T}
    depth = _cut_depth_budget(cut_depth)
    state = FormState(model.dofs, _state_vector(solution, model))
    fields = _sampled_fields(model, field)
    entries = (point=_sample_entries(point_data, :point_data), cell=_sample_entries(cell_data, :cell_data))
    rule = _subdivision_rule(subdivisions, Val(D))
    ws, meshed = _assembly_workspace(model), Space[]
    MeshRecord = @NamedTuple{level::Int, role::Symbol, grid::SampleGrid{D,T,D}}
    records = @NamedTuple{field::Field, data::SampleGrid{D,T,D}, meshes::Vector{MeshRecord}}[]
    for fld in fields
        space = _field_space(model.problem, fld.name)
        regions = _region_list(model, nothing, space).regions     # the list the solve integrated the field over
        data = _sample_field(ws, state, space, regions, fld, rule, depth, entries, model.version)
        meshes = MeshRecord[]
        if level_meshes && !any(s -> s === space, meshed)
            dofs = _field_layout(model.dofs, fld.name).dofs
            coverage = build_coverage(space, dofs.tolerance)
            for level in space.levels
                push!(meshes, (; level=level.id, level.role, grid=_level_grid(level, dofs, coverage)))
            end
        end
        push!(meshed, space)
        push!(records, (; field=fld, data, meshes))
    end
    return records
end

# The default `point_data`: one entry, `uh`, the solution value at every point.
_default_point_data() = (uh=(u, c, x, xi) -> u(c, xi),)

# Resolve the `cut_depth` keyword to a non-negative bisection budget. Checked first, so a rejected value costs no
# sampling and, like every other refusal, leaves nothing on disk.
#
# The cap is 8 rather than a larger round number because the leaf count grows as 2^((D−1)k): at D = 3 and the measured
# ~400 B per emitted cell, depth 8 is already of order 10⁷ cells and a gigabyte, which is a limit rather than a safety
# net.
function _cut_depth_budget(cut_depth)
    cut_depth === :none && return 0
    cut_depth isa Integer && 0 <= cut_depth <= 8 && return Int(cut_depth)
    throw(ArgumentError("cut_depth must be :none or an integer in 0:8; got $(_shown(cut_depth))"))
end

# `repr(x)` for a refusal message, compiled once for every `x`: unspecialised in inference as well as in code generation
# (`@nospecializeinfer`) and never inlined, so a function that can refuse a keyword does not compile the printing of
# that keyword's type on the path that accepts it. `repr` of a `Symbol` — the default `subdivisions = :degree` — goes
# through Base's expression printer, `show_unquoted`, which a first `write_vtk` otherwise inferred whether or not a
# refusal ever ran.
@noinline Base.@nospecializeinfer _shown(@nospecialize(x)) = repr(x)

# The fields a sampling covers, as the model's own `Field`s: every field of the problem in order, or the one `field`
# names — a `Field` or its name `Symbol`, matched by name, so a `Field` built before `prepare` selects the model's field
# of that name. A single-domain model has exactly one field, which is the single-block output.
function _sampled_fields(model::Model, field)
    field === nothing && return collect(model.problem.fields)
    name = field isa Field ? field.name : field
    for f in model.problem.fields
        f.name === name && return [f]
    end
    throw(ArgumentError("unknown field $name"))
end

# Normalise a `point_data` / `cell_data` keyword to its `name => callback` entries, in the container's order. Three
# accepted shapes: a `NamedTuple` (the ergonomic one, `(uh = …, σ = …)`), an `AbstractDict` (programmatic construction)
# and a `Tuple` of `Pair`s (explicit construction). The list is untyped in the callbacks on purpose: the sampler is
# then compiled once per field type rather than once per callback set, and each callback meets its own compiled loop
# through one dynamic call per entry (`_sample_array`).
function _sample_entries(data, kind::Symbol)
    data isa Union{NamedTuple,AbstractDict} ||
        data isa Tuple && all(item -> item isa Pair, data) ||
        throw(ArgumentError("$kind must be a NamedTuple, Dict, or tuple of Pairs"))
    return Pair{String,Any}[String(k) => v for (k, v) in (data isa Tuple ? data : pairs(data))]
end

# The `subdivisions` keyword as the sampler reads it, validated once, before any region is walked: `:degree`, kept and
# resolved per region from its parents' orders (`_sample_cells`), or the per-axis lattice counts — `(1, …, 1)` for
# `:none`, `(n, …, n)` for a positive integer `n`, an `NTuple{D}` of positive integers as given.
function _subdivision_rule(subdivisions, ::Val{D}) where {D}
    subdivisions === :degree && return subdivisions
    subdivisions === :none && return ntuple(_ -> 1, Val(D))
    subdivisions isa Integer && subdivisions > 0 && return ntuple(_ -> Int(subdivisions), Val(D))
    subdivisions isa NTuple{D,Integer} && all(>(0), subdivisions) && return Int.(subdivisions)
    throw(ArgumentError("subdivisions must be :degree, :none, a positive integer or an NTuple{$D} of positive " *
                        "integers; got $(_shown(subdivisions))"))
end

# The data grid of one field: its sample cells (`_sample_cells`) and their corners, the user's entries and the built-in
# arrays. Point arrays: the user's entries in their order, then `level_set` — `φ` at every point, exactly
# `levelset_value` — when the space carries a `PhysicalDomain` and no entry already has that name. Cell arrays:
# `region_id` (the cell's region in the field's region list), `cover_count` (its number of covering parents), then the
# user's entries. A function barrier typed by the space and its regions, and never specialised on the model or on the
# callback set, which arrive as the model's pin and an untyped entry list.
#
# Each entry's array is collected through `Base.inferencebarrier` (the idiom of `_coverage_constraints!` in `dofs.jl`):
# one run-time dispatch on the callback's concrete type (`_sample_array`). The entry list holds the callbacks as `Any`
# and `_sample_array` is the only method that matches, so without the barrier inference descends into it with the
# callback unknown and infers the collection loop and the value checks for an unknown value, code that never runs: on
# the first `write_vtk` of a two-level 2D overlay, 67 more code instances inferred.
function _sample_field(ws, state::FormState, space::Space{D,T}, regions::Vector{VolumeRegion{D,T}}, fld::Field, rule,
                       depth::Int, entries, version::Int) where {D,T}
    boxes, region_ids = _sample_cells(space, regions, rule, depth)
    points = Vector{SVector{D,T}}(undef, length(boxes) << D)
    for (c, b) in pairs(boxes)
        _put_corners!(points, c, b)
    end
    u = _sample_accessor(ws, state, regions, fld)
    at_points = (; location=:point, shift=D, boxes, points, region_ids, regions, version)
    at_cells = merge(at_points, (; location=:cell, shift=0))
    sample = Base.inferencebarrier(_sample_array)
    point_data = Pair{String,AbstractVector}[]
    for (name, f) in entries.point
        push!(point_data, name => sample(:point_data, name, f, u, at_points)::AbstractVector)
    end
    space.physical === nothing ||
        any(e -> first(e) == "level_set", entries.point) ||
        push!(point_data, "level_set" => _levelset_values(space.physical, points, T))
    cover_count = similar(region_ids)
    for (c, r) in pairs(region_ids)
        cover_count[c] = length(regions[r].parents)
    end
    cell_data = Pair{String,AbstractVector}["region_id" => region_ids, "cover_count" => cover_count]
    for (name, f) in entries.cell
        push!(cell_data, name => sample(:cell_data, name, f, u, at_cells)::AbstractVector)
    end
    return SampleGrid{D,T,D}(points, point_data, cell_data)
end

# `φ` at every point, in the space's number type. A barrier: the space's `PhysicalDomain` is abstractly typed, so one
# dynamic call here makes the loop concrete, where a `levelset_value` call per point would dispatch dynamically.
_levelset_values(physical::PhysicalDomain, points, ::Type{T}) where {T} = T[levelset_value(physical, x) for x in points]

# Store the corners of `b` (`_box_corners`) as the points of cell `c` of a grid of `D`-cubes: points
# `(c − 1)·2ᴰ + 1 : c·2ᴰ`. The one writer of a grid's points, for the sample cells and the level meshes alike. It
# fills a presized vector: collecting the corners through a nested generator instead compiles Base's collection of an
# iterator of unknown length for every grid type, and grows the vector as it goes.
function _put_corners!(points::Vector{SVector{D,T}}, c::Int, b::AxisBox{D,T}) where {D,T}
    for (j, x) in enumerate(_box_corners(b))
        points[((c - 1) << D)+j] = x
    end
    return points
end

# The `u(context, xi[, field])` a callback receives: the current solution at the callback's sample, through the
# assembly's own field evaluation — the one `q.state` and the quadrature-point walker read. A region's dof values are
# read into the shared `state` once (`_frame!`, `_state_region!`); each call then evaluates the basis at the sample
# (`_refresh!`, values only) and every field there (`_state_point!`), and reads one field off `state`: a scalar for a
# one-component field, an `SVector` over its components otherwise, the count read off the `Field`'s type so the result
# is inferable. That count is checked against the model's (`_checked_field_index`), as `value` checks it: read with
# another count, a `Field` that shares a model field's name would truncate that field or run on into the next field's
# values in `state`.
#
# Framing is lazy and keyed on `context.region_id`: a region is framed by the first call that reads it and again only
# after a call has read another region. The samples of an entry come region by region, so an entry that calls `u` frames
# each region once, and a write whose callbacks never call `u` frames nothing. It is also why `u` takes the context of
# the sample at hand: a context kept from another sample reads that sample's region.
#
# `default` is the field of the record being sampled. A callback may name another `field`, but only one on the same
# subdomain: the region's parents are that subdomain's cells, so another subdomain's field owns none of them in the
# frame (`ws.fptr`) and has no value there, and asking for one raises. To read a field on a *different* subdomain,
# call `value(solution, model, other, x)` with the sample coordinate `x` the callback also receives.
function _sample_accessor(ws, state::FormState, regions, default::Field)
    framed = Ref(0)
    return function (context, xi, field::Field=default)
        region = regions[context.region_id]
        if context.region_id != framed[]
            _state_region!(state, _frame!(ws, region))
            framed[] = context.region_id
        end
        f = _checked_field_index(state.layout, field)
        (ws.fptr[f] < ws.fptr[f+1] || isempty(region.parents)) ||
            throw(ArgumentError("u(context, xi, field) reads only fields of the sampled field's own subdomain; " *
                                "evaluate $(field.name) with value(solution, model, $(field.name), x)"))
        _refresh!(ws, region, xi.region, Val(false))
        _state_point!(state, ws, Val(false))
        offset, C = state.offsets[f], component_count(field)
        return C == 1 ? state.values[offset+1] : SVector(ntuple(c -> state.values[offset+c], Val(C)))
    end
end

# The `(context, x, xi)` a callback receives at sample `k` of the walk `s`: point `k` of the grid, which lies in cell
# `(k − 1) >> D + 1` (`s.shift = D`), or the center of cell `k` (`s.shift = 0`). `context` names the kind of sample, its
# region by index and by box, the emitted cell, the region's covering parents and the model's pin; `xi.region` is `x` in
# the region's reference frame, where the accessor evaluates the solution.
function _callback_args(s, k::Int)
    c = ((k - 1) >> s.shift) + 1
    r, box = s.region_ids[c], s.boxes[c]
    region, x = s.regions[r], s.shift == 0 ? center(box) : s.points[k]
    region_box = _region_box(region)
    context = (; s.location, region_id=r, region=region_box, cell=box, region.parents, model_version=s.version)
    return context, x, (; region=physical_to_reference(region_box, x))
end

# One user entry's array over a walk: the callback at every sample, each value checked as it comes (`_sample_value`),
# collected into the narrowest array that holds them. A function barrier — the one dynamic call per entry — so the loop
# runs on the callback's and the accessor's concrete types. The array takes the first value's type, which is every
# value's type for the callbacks one writes, and is then kept as it is; a value of another type hands the walk to
# `_promoted_values`, which promotes over every sample or refuses naming the entry. An empty walk gives an empty
# `Float64` array.
#
# An explicit loop, not `map`: Base's `collect` widens its element type as it goes, which compiles its collection
# machinery anew for every callback type. Here, when the callback's value type is concrete, the type test folds to true
# and the hand-over is never compiled.
function _sample_array(kind::Symbol, name::String, f::F, u::U, s) where {F,U}
    n = length(s.boxes) << s.shift
    n == 0 && return Float64[]
    sample(k) = _sample_value(kind, name, f(u, _callback_args(s, k)...))
    first_value = sample(1)
    values = Vector{typeof(first_value)}(undef, n)
    values[1] = first_value
    for k in 2:n
        v = sample(k)
        # The samples so far, this one and the rest, each callback still called once per sample and in order.
        v isa eltype(values) ||
            return _promoted_values(kind, name, Any[j < k ? values[j] : j == k ? v : sample(j) for j in 1:n])
        values[k] = v
    end
    return values
end

# The array of an entry whose values differ in type between samples: mixed number types (an `Int` at some samples, a
# `Float64` at others) and `SVector`s of mixed element types are promoted over every sample with `promote_type`, and
# anything else — shapes that differ between samples — is refused, naming the entry and the types it returned. Typed
# on the untyped value list, so it compiles once whatever the callback.
function _promoted_values(kind::Symbol, name::String, values::Vector{Any})
    S = mapreduce(typeof, promote_type, values)
    isconcretetype(S) && (S <: Real || S <: SVector && eltype(S) <: Real) && return convert(Vector{S}, values)
    throw(ArgumentError("$kind entry \"$name\" returned $(join(unique(typeof.(values)), ", ")): a callback value " *
                        "must be a real number, a tuple of reals or an SVector of reals, of one shape at every sample"))
end

# One callback value, checked and normalised: a real number is kept as it is; a tuple or an `SVector` of reals becomes
# an `SVector` with its components promoted to one type, so a tuple and the `SVector` of the same numbers give the same
# array. Anything else — a `Vector`, a matrix, a `String`, a `Complex`, `nothing` — is refused, naming the entry and
# what it returned, before anything is written.
function _sample_value(kind::Symbol, name::String, v)
    v isa Real && return v
    v isa Union{Tuple,SVector} && all(x -> x isa Real, v) && return SVector(v...)
    throw(ArgumentError("$kind entry \"$name\" returned a $(typeof(v)): a callback value must be a real number, a " *
                        "tuple of reals or an SVector of reals, of one shape at every sample"))
end

# ── Cut-cell refinement ───────────────────────────────────────────────────────
#
# ParaView's Clip reconstructs ∂Ω from the `level_set` point array by linear interpolation along each cell's edges, so
# the cut surface it draws is a facetisation at the size of the emitted cell and at nothing else — neither φ's own
# accuracy nor the solution order enters. Resolving ∂Ω therefore means emitting small cells where ∂Ω runs, and only
# there: the boundary is codimension one, so refining toward it costs O(2^((D−1)k)) against the O(2^(Dk)) of refining
# the whole region. Measured on a 3D 5×5×5 order-2 sphere, 81 regions: 5 723 cells against 41 472 at 8 sub-cells per
# axis across ∂Ω, 23 543 against 331 776 at 16, 95 891 against 2 654 208 at 32.
#
# The structure is QuESo's two-level budget (`Octree::Node::Refine(MinLevel, MaxLevel)`, M. Meßmer et al., Comput.
# Methods Appl. Mech. Engrg. 400 (2022) 115584): `subdivisions` is the floor applied to every region and sized by
# polynomial order, because its job is to resolve the *solution*; `cut_depth` is the ceiling applied only to boxes ∂Ω
# passes through, because its job is to resolve the *geometry*. The two knobs are independent and neither can express
# the other.

# The sample cells of a field, region by region: each region's `subdivisions` lattice in `CartesianIndices` order, a
# lattice box of a region ∂Ω crosses replaced in place by the leaves of its cut tree, depth first. Returns the cells and
# the index of each cell's region, which is non-decreasing and is the `region_id` array. A box of an uncut region, of a
# space without a `PhysicalDomain`, or at `depth = 0` is the lattice box itself, so those write exactly the unrefined
# lattice by construction rather than by care.
#
# `:degree` counts per axis the largest order `max(1, p)` over the region's parent CELLS — a degree-`p` region split
# into `p` sub-cells per axis, enough for ParaView's piecewise-linear cells to show the basis without visible aliasing.
# Per parent cell, not per level: a level's nominal order is the per-axis maximum over all its cells, so sizing from it
# subdivided every region at the order of the loudest cell anywhere on the level — measured on a 3D 8³ level at p = 1
# with a single p = 6 cell, 110 592 sub-cells against 512, and 221.9 MB against 2.50 MB.
function _sample_cells(space::Space{D,T}, regions::Vector{VolumeRegion{D,T}}, rule, depth::Int) where {D,T}
    physical, lipschitz = space.physical, _cut_lipschitz(space.physical, depth)
    boxes, region_ids = AxisBox{D,T}[], Int[]
    for (r, region) in pairs(regions)
        b, emitted = _region_box(region), length(boxes)
        n = rule isa Symbol ? _axis_max((level, cell) -> max.(1, cell_order(level, cell)), space, region.parents) : rule
        h, cut = edge_lengths(b), depth > 0 && physical !== nothing && region.kind in _CUT_KINDS
        for i in CartesianIndices(n)
            lattice = AxisBox{D,T}(SVector{D,T}(ntuple(d -> b.lower[d] + h[d] * (i[d] - 1) / n[d], Val(D))),
                                   SVector{D,T}(ntuple(d -> b.lower[d] + h[d] * i[d] / n[d], Val(D))))
            cut ? _cut_leaves!(boxes, physical, lattice, depth, lipschitz) : push!(boxes, lattice)
        end
        append!(region_ids, Iterators.repeated(r, length(boxes) - emitted))
    end
    return boxes, region_ids
end

# The Lipschitz constant the cut refinement certifies its boxes with (`_cut_marked`): the largest any leaf of the
# geometry declares, looked up once per field rather than per region, since it is a property of the geometry tree. A
# refining write whose geometry declares none (`leaf(f)`'s default `Inf`) falls back to corner signs, and the first
# such write of a session warns.
function _cut_lipschitz(physical, depth::Int)
    physical === nothing && return Inf
    lipschitz = _levelset_lipschitz(physical.geometry)
    if depth > 0 && !isfinite(lipschitz)
        @warn("cut_depth is refining on corner signs alone: no leaf of this PhysicalDomain " *
              "declares a Lipschitz constant, so a feature of Ω smaller than a sub-cell can be " *
              "missed and the clipped surface is not guaranteed watertight. Pass " *
              "`leaf(f; lipschitz = L)` (1.0 for a signed distance function) to certify it.", maxlog=1)
    end
    return lipschitz
end

# Collect the leaves of the cut tree rooted at `b` into `boxes`. Descend only into marked boxes, and descend every
# marked box the full `depth`, so every leaf that ∂Ω passes through comes out at the same size. That uniformity is
# load-bearing: two finest leaves sharing a face see identical corner coordinates and therefore identical φ, so their
# clip surfaces meet exactly, and every coarser leaf is unmarked and contributes no clip surface at all. A
# variable-depth rule would buy fewer cells and lose both properties. Both hold among the trees of boxes of one size —
# the lattice of one region, or of regions whose lattice boxes coincide on a shared face — and not across two cut
# regions whose lattice boxes differ: there the finest leaves differ too, and their T-junctions can crack the clip. At
# `depth = 0` nothing is marked, and a box whose bisection is degenerate (`_box_children`) is a leaf.
function _cut_leaves!(boxes::Vector{AxisBox{D,T}}, physical::PhysicalDomain, b::AxisBox{D,T}, depth::Int,
                      lipschitz::Float64) where {D,T}
    children = depth > 0 && _cut_marked(physical, b, lipschitz) ? _box_children(b) : nothing
    children === nothing && return push!(boxes, b)
    foreach(child -> _cut_leaves!(boxes, physical, child, depth - 1, lipschitz), children)
    return boxes
end

# Should this sub-box be bisected? Two regimes, and which one applies is decided by whether the geometry declares a
# Lipschitz constant.
#
#   * finite `L` — mark iff `|φ(c)| ≤ L·r` with `r` the half-diagonal, i.e. iff the package's own uniform-sign
#     certificate (`_certified_sign`, the classifier's test) leaves the box undetermined: an *unmarked* box provably
#     has one sign of φ over its whole closed box, so it contains no part of ∂Ω and emits no cut surface, and no
#     hanging node on its faces can disagree with a finer neighbour. That is what makes the output crack-free rather
#     than merely crack-free in practice. One φ evaluation per box.
#   * `L = Inf` (the default from `leaf(f)`) — no certificate exists, so fall back to the corner signs, which is the
#     same evidence ParaView's own clip works from. 2ᴰ evaluations, and a feature of Ω smaller than the box and missing
#     every corner is invisible; the remedy is to declare the constant on the leaf.
#
# Deliberately *not* used: a secant bound `max|φ(vᵢ) − φ(c)|` in place of `L·r`. It is identically zero whenever φ is
# symmetric about the box center — two parallel plates straddling it, say — and the box then goes unmarked at every
# threshold, which is exactly the configuration that tears.
function _cut_marked(physical::PhysicalDomain, b::AxisBox, lipschitz::Float64)
    φ = x -> levelset_value(physical, x)
    isfinite(lipschitz) && return _certified_sign(φ, lipschitz, b) == 0
    corners = _box_vertices(b)
    inside = φ(first(corners)) <= 0
    return any(x -> (φ(x) <= 0) != inside, Base.tail(corners))
end

# ── Level meshes ──────────────────────────────────────────────────────────────

# The record of one level's mesh: every cell of the level, active or not, in `cell_indices` order, as a `K = D` grid
# with eight integer cell arrays that make the pruning visible in ParaView:
#
#   * `level_id`, `role_id` (0 for the base level, 1 for an overlay), `order_max`, `cell_id` — structural tags;
#   * `active`       — 1 iff the cell survives the `LevelMask` (user mask + fold);
#   * `covered`      — 1 iff a finer level fully covers the cell (see `Coverage`, which computes the flag only where the
#     dof layer reads it);
#   * `active_dofs`  — basis modes on the cell with at least one active component;
#   * `reduced_dofs` — modes eliminated by covered-mode pruning (`:coverage` / `:dedup`).
#
# Rendered "Surface With Edges" this doubles as the old wireframe while carrying the per-cell data a line mesh could
# not. `dofs` is the layout the dof counts are read from: the first field's, for a space several fields share.
#
# One pass over the cells fills every presized array, rather than one comprehension per array: each comprehension
# compiled Base's collection machinery for its own generator type, which a first write paid for every level type.
function _level_grid(level::Level{D,T}, dofs::DofLayout{D}, coverage::Coverage{D}) where {D,T}
    mesh, cells = level.mesh, cell_indices(level.mesh)
    covered, linear, n = coverage.covered[level.id], LinearIndices(mesh.cells), length(cells)
    active(raw) = any(c -> dofs.active_component[raw, c] != 0, 1:dofs.components)
    reduced(raw) = dofs.elimination_source[raw] in (:coverage, :dedup)
    columns = ntuple(_ -> Vector{Int}(undef, n), Val(6))
    order_max, cell_id, active_cell, covered_cell, active_dofs, reduced_dofs = columns
    points = Vector{SVector{D,T}}(undef, n << D)
    for (k, c) in enumerate(cells)
        raws = cell_dofs(dofs, level.id, c)
        order_max[k], cell_id[k] = maximum(cell_order(level, c)), linear[c]
        active_cell[k], covered_cell[k] = Int(is_active(level.mask, c)), Int(covered[c])
        active_dofs[k], reduced_dofs[k] = count(active, raws), count(reduced, raws)
        _put_corners!(points, k, cell_box(mesh, c))
    end
    data = Pair{String,AbstractVector}["level_id" => fill(level.id, n),
                                       "role_id" => fill(level.role === :base ? 0 : 1, n), "order_max" => order_max,
                                       "cell_id" => cell_id, "active" => active_cell, "covered" => covered_cell,
                                       "active_dofs" => active_dofs, "reduced_dofs" => reduced_dofs]
    return SampleGrid{D,T,D}(points, Pair{String,AbstractVector}[], data)
end

# ── Quadrature cloud ──────────────────────────────────────────────────────────

# The model's quadrature-point cloud as records: one `(; signature, grid)` per distinct sorted list of covering parent
# levels, in sorted order, over the regions of every subdomain's integration plan. Level ids are unique across
# subdomains, so a coupled model's subdomains land in different entries and every one of them is kept. A grid is a
# vertex cloud (`K = 0`) of the physical quadrature points, each with its physical weight `weight` (the rule weight
# times the region's Jacobian, as assembly integrates with it), region by region — subdomain order, then plan order —
# and point by point. Both arrays are presized from the regions' point counts rather than grown point by point. Not
# `_quadrature_cloud` (`data.jl`), the bare coordinates in `q.point` order that the RBFP0 transfer indexes. The entries
# are collected into a vector of their concrete type by an explicit loop, as `_sample_fields` collects its records.
function _sample_quadrature(model::Model{D,T}) where {D,T}
    groups = Dict{Vector{Int},Vector{VolumeRegion{D,T}}}()
    for subdomain in model.subdomains, region in subdomain.plan.regions
        push!(get!(() -> VolumeRegion{D,T}[], groups, sort!([p.level for p in region.parents])), region)
    end
    entries = @NamedTuple{signature::Vector{Int}, grid::SampleGrid{D,T,0}}[]
    for signature in sort!(collect(keys(groups)))
        regions, n = groups[signature], 0
        for region in regions               # not `sum(f, regions)`, whose reducer chain compiles for every region type
            n += _region_qpoint_count(region)
        end
        points = Vector{SVector{D,T}}(undef, n)
        weights, k = similar(points, T), 0
        for region in regions
            jacobian = _region_jacobian(region)
            for q in 1:_region_qpoint_count(region)
                k += 1
                points[k], weights[k], _ = _region_point(region, q, jacobian)
            end
        end
        grid = SampleGrid{D,T,0}(points, Pair{String,AbstractVector}["weight" => weights],
                                 Pair{String,AbstractVector}[])
        push!(entries, (; signature, grid))
    end
    return entries
end

# ── Solution evaluation ───────────────────────────────────────────────────────
#
# The point evaluations `value` / `field_gradient` (`_level_field`) reconstruct a field one covering cell at a time and
# sum the cells. Every other evaluation — assembly's `q.state`, the quadrature-point walker, `l2_error` and the
# sampling callbacks' accessor (`_sample_accessor`) — evaluates into a workspace's `BasisBank` (`_refresh!` and
# `_state_point!` in `assembly.jl`), which keeps the same association: per-cell partial sums, then the sum over cells.

"""
    value(solution, model, x)
    value(solution, model, x, component)
    value(solution, model, u::Field, x[, component])

Evaluate the superposed solution at physical point `x`. Scalar fields return a scalar; component fields return an
`SVector` across components unless a single `component` is requested. `component = nothing`, the default, evaluates
every component; a `component` outside `1:component_count(u)` raises `ArgumentError`. Overlay levels are extended by
zero — only levels whose mesh contains `x` (and whose covering cell is active) contribute. Throws if `x` lies outside
the evaluated field's own subdomain box, at the model's containment tolerance. That box is the embedding domain, not
`Ω`: with a [`PhysicalDomain`](@ref) in play a point outside `Ω` does not raise. It evaluates the polynomial extension
inside a cut cell, and zero inside a cell the fictitious fold deactivated.

`x` accepts `NTuple{D,<:Real}` or `SVector{D,<:Real}`. Multi-field models require the `u::Field` form so the field to
evaluate is unambiguous; `u` must be the model's field of that name, with its component count.
"""
function value(solution::Solution, model::Model{D}, x::PointLike{D},
               component::Union{Nothing,Integer}=nothing) where {D}
    return _field_quantity(solution, model, _default_field(model), x, component, Val(false))
end

function value(solution::Solution, model::Model{D}, u::Field, x::PointLike{D},
               component::Union{Nothing,Integer}=nothing) where {D}
    return _field_quantity(solution, model, u, x, component, Val(false))
end

"""
    field_gradient(solution, model, x)
    field_gradient(solution, model, x, component)
    field_gradient(solution, model, u::Field, x[, component])

Evaluate the physical gradient of the superposed solution at physical point `x`. Scalar fields return one `SVector{D}`
gradient; component fields return an `SVector` of component gradients unless a single `component` is requested.
`component = nothing`, the default, evaluates every component. Overlay levels are extended by zero, and the point and
component are checked, as for [`value`](@ref). The leading `field_` avoids colliding with `Tensors.gradient` when both
packages are loaded together.

The gradient is the physical gradient (chain-rule scaled by `2 / edge_lengths(parent_box)` per axis), not the
reference-frame gradient.
"""
function field_gradient(solution::Solution, model::Model{D}, x::PointLike{D},
                        component::Union{Nothing,Integer}=nothing) where {D}
    return _field_quantity(solution, model, _default_field(model), x, component, Val(true))
end

function field_gradient(solution::Solution, model::Model{D}, u::Field, x::PointLike{D},
                        component::Union{Nothing,Integer}=nothing) where {D}
    return _field_quantity(solution, model, u, x, component, Val(true))
end

# The one body of `value` (`G = false`) and `field_gradient` (`G = true`). The checks run in a fixed order — the
# solution's pin and length, the point's conversion to the model's number type, the field's space, the point inside
# that space's box, the field's component count, the component's range — so a call wrong in two ways always names the
# same fault. The box is the field's own subdomain's, never subdomain 1's: on a coupled model each field is bounded by
# its own. The component count is checked (`_checked_field_index`) because it is read from `u`'s type rather than from
# the layout: that keeps a vector result inferable, and a `Field` that shares a name with the model's but not its count
# is not the model's field.
#
# The components evaluated, `comps`, are static: the requested one, or `1` for a scalar field, or the tuple `(1, …, C)`,
# so every level evaluates its basis once for all of them (`_level_field`). The level sum starts from zero, as the
# per-component sum always has, so the two agree bit for bit, the sign of an exact zero included: a sum started from
# the first level would keep a −0.0 that the zero start turns into +0.0. A tuple of components is returned as an
# `SVector`.
function _field_quantity(solution::Solution, model::Model{D,T}, u::Field{<:Any,<:Any,C}, x::PointLike{D},
                         component::Union{Nothing,Integer}, ::Val{G}) where {D,T,C,G}
    coefficients = _checked_coefficients(solution, model)
    point = SVector{D,T}(x)
    space = _field_space(model.problem, u.name)
    contains_point(point, space.domain, model.dofs.tolerance) ||
        throw(ArgumentError("evaluation point is outside the field's domain"))
    layout = model.dofs.fields[_checked_field_index(model.dofs, u)]
    _check_component(component, u)
    comps = component !== nothing ? component : C == 1 ? 1 : ntuple(identity, Val(C))
    R = promote_type(T, eltype(coefficients))
    acc = map(_ -> G ? zero(SVector{D,R}) : zero(R), comps)
    for level in space.levels
        acc = map(+, acc, _level_field(coefficients, model, layout, level, point, comps, Val(G)))
    end
    return comps isa Tuple ? SVector(acc) : acc
end

# The index in `layout` of the model's field `u` names, refusing a `u` whose component count is not that field's. Every
# evaluation of a given `Field` — `value`, `field_gradient` and the sampling callbacks' `u` — reads its component count
# `C` off `u`'s type, which keeps the result inferable; the layout holds the model's own count. A `Field` that shares a
# model field's name but not its count is another field, and read with its own `C` it would return part of the model's
# field, or that field followed by the next field's values; the check costs one integer comparison.
function _checked_field_index(layout::SystemLayout, u::Field)
    f = _field_index(layout, u.name)
    components = layout.fields[f].components
    components == component_count(u) ||
        throw(ArgumentError("field $(u.name) has $components components in this model, not $(component_count(u))"))
    return f
end

# Pick the implicit field of a single-field model. Multi-field models require an explicit field argument so the caller
# does not silently evaluate the wrong field; this raises with a clear message in that case.
function _default_field(model::Model)
    length(model.problem.fields) == 1 ? first(model.problem.fields) :
    throw(ArgumentError("field argument is required for multi-field models"))
end

# One level's contribution to the field of `layout` at physical point `x`, for each of `components` (an `Int` gives one
# component's value or gradient, a tuple one per component): its value (`G = false`) or its physical gradient
# (`G = true`). Zero if the level's mesh does not contain `x` (overlay zero-extension) or if the containing cell is
# masked inactive. Otherwise: locate the cell, map `x` to the cell's reference frame, evaluate the basis values there —
# or the physical basis gradients, which `physical_basis_gradients` scales by the chain rule `2 / h_d` of the
# axis-aligned cell — once for all the components, and contract them with each component's dof values
# (`_parent_field`).
function _level_field(coefficients, model::Model{D,T}, layout::FieldLayout{D,T}, level::Level{D,T}, x::SVector{D,T},
                      components::Union{Integer,Tuple}, ::Val{G}) where {D,T,G}
    R = promote_type(T, eltype(coefficients))
    cell = locate_cell(level.mesh, x; tol=model.dofs.tolerance)
    (cell === nothing || !is_active(level.mask, cell)) && return map(_ -> G ? zero(SVector{D,R}) : zero(R), components)
    parent_box = cell_box(level.mesh, cell)
    xi = physical_to_reference(parent_box, x)
    φ = G ? physical_basis_gradients(level, cell, parent_box, xi) : basis_values(level, cell, xi)
    raw_dofs = cell_dofs(layout.dofs, level.id, cell)
    return map(c -> _parent_field(layout, coefficients, raw_dofs, φ, c), components)
end

# The field on one parent cell: `Σᵢ u(rawᵢ, component)·φᵢ` over the cell's raw dofs, with `φ` the cell's basis values
# (the field's value) or physical basis gradients (its gradient). `dof_value` reads `u`, so a constrained dof
# contributes its pinned value and a linear-constraint pivot its expansion. The number type is the coefficients'
# promoted with the model's.
function _parent_field(layout::FieldLayout{D,T}, coefficients, raw_dofs, φ, component::Integer) where {D,T}
    result = zero(promote_type(T, eltype(coefficients))) * zero(eltype(φ))
    for i in eachindex(raw_dofs)
        result += dof_value(layout, coefficients, raw_dofs[i], component) * φ[i]
    end
    return result
end

# ── L² error ──────────────────────────────────────────────────────────────────

"""
    l2_error(solution, model, exact; norm=:relative) -> Real
    l2_error(solution, model, u::Field, exact; norm=:relative) -> Real

Compute the L² error of `solution` against an analytic / manufactured solution `exact(x)`:

  - `norm = :relative` (default) returns `sqrt(∫ ‖u_h − u_exact‖² dx) / sqrt(∫ ‖u_exact‖² dx)`;
  - `norm = :absolute` returns just the numerator;
  - if `‖u_exact‖_L² = 0`, the relative form falls back to the absolute error so it stays meaningful for zero-reference
    checks (e.g. a fabricated test where `exact(x) ≡ 0`).

`exact(x)` returns a scalar for a scalar field, or an indexable value with one entry per component (an `SVector`, a
tuple or a vector) for a vector field; the pointwise error is summed component by component,
`‖u_h − u_exact‖² = Σ_c (u_h[c] − u_exact[c])²`.

`u` names the field to measure and is matched by name: the model's field of that name is measured, with the component
count of the model's layout, where [`value`](@ref) and [`field_gradient`](@ref) also refuse a `u` whose own count
differs. The error is integrated over that field's own subdomain — its space, its integration plan, its dof layout — so
on a coupled model it is each subdomain's own error, and there is no single number for the union: an interface shared
by two subdomains would be integrated twice. Omitting `u` picks the implied field of a single-field model and raises on
a multi-field one, which is why the multi-field and coupled workflows had no L² error report before this overload
existed.

The integration uses the model's own integration plan — the same admissible regions assembly walks — so the error is
consistent with what the solver actually integrated. For SPD model problems this is the natural convergence-study
metric: monotonic decrease as you refine, with the rate matching the basis order.
"""
function l2_error(solution::Solution, model::Model, exact; norm::Symbol=:relative)
    return l2_error(solution, model, _default_field(model), exact; norm)
end

function l2_error(solution::Solution, model::Model{D,T}, u::Field, exact; norm::Symbol=:relative) where {D,T}
    norm in (:relative, :absolute) || throw(ArgumentError("norm must be :relative or :absolute"))
    state = FormState(model.dofs, _state_vector(solution, model))
    list = _region_list(model, nothing, _field_space(model.problem, u.name))
    f = _field_index(model.dofs, u.name)
    offset, C = state.offsets[f], model.dofs.fields[f].components
    # A walk with the field's dof values, as `foreach_quadrature_point` runs one, but of values only (`Val(false)`): the
    # integrand reads no gradient, so the walk skips the gradient tables and reads the same values. The squared error is
    # summed component by component, which is `_squared_norm(u_h − exact)` term for term and keeps the fold type-stable
    # whatever the component count.
    ws = _assembly_workspace(model)
    error_squared, exact_squared = _walk((zero(T), zero(T)), ws, list, state, Val(false)) do (err, ref), q
        exact_value = exact(q.x)
        squared = sum(c -> abs2(q.state.values[offset+c] - exact_value[c]), 1:C)
        return err + q.weight * squared, ref + q.weight * _squared_norm(exact_value)
    end

    error_norm = sqrt(error_squared)
    norm === :absolute && return error_norm

    exact_norm = sqrt(exact_squared)
    return iszero(exact_norm) ? error_norm : error_norm / exact_norm
end

# The squared magnitude `‖u_exact‖²` of an exact value, scalar (`abs2`) or component-indexable (`sum(abs2, …)`): the
# integrand of `l2_error`'s reference norm. The error itself is summed component by component in `l2_error`.
_squared_norm(value::Number) = abs2(value)
_squared_norm(value) = sum(abs2, value)

# ── Boundary integration ──────────────────────────────────────────────────────
#
# Quadrature-only postprocessing on a portion of the boundary. It is a sum over the quadrature-point walker (`_walk` in
# `assembly.jl`), so it integrates over the region list assembly resolves for the same `on=` target, visits the points
# in the serial assembly order and hands the integrand the payload a form sees. Physical [`FacetRegion`](@ref)s,
# immersed-surface and two-sided interface [`SimplexRegion`](@ref)s take one path, the region accessors', which map
# each region's frame rule to physical points and weights.

"""
    boundary_integral(integrand, model::Model; on, field=nothing) -> result

Integrate a user callback over a portion of the boundary or over a coupling interface.

`on` selects the integration region:

  - `boundary(:all)`, `boundary(axis=d, side=s)`, `boundary((axis=…, side=…), …)` — the physical boundary `∂Ω`. The
    Gauss rule per region is the per-axis maximum `recommended_quadrature_order(level.basis, cell_order(level, cell))`
    over the region's covering parent cells. Overlay levels whose own mesh face coincides with the selected facet
    contribute their own segment; level masking is respected. On an immersed [`PhysicalDomain`](@ref) the integral runs
    over `face ∩ Ω` (see [`boundary`](@ref)): at the default `α = 0`,
    `boundary_integral(q -> 1.0, model; on=boundary(axis=d, side=s))` returns the measure of the face's physical part
    and `integrand` is called only at points inside `Ω`, while `α > 0` blends the full face back in and forfeits both.
    The `cut_facet_region_count` field of [`AssemblyDiagnostics`](@ref) counts how many of the integrated regions needed
    trimming, and `facet_moment_fit_residual_max` how well their fits went — exact where every restricted leaf is affine
    (planar), corners strictly inside the face included, and on one-dimensional facets, accurate rather than exact on a
    curved cut, as [`boundary`](@ref) details. A selector every one of whose faces lies outside `Ω` carries no
    quadrature at all under `α = 0` and raises, exactly as a selector no active cell reaches does.
  - A [`BoundaryMesh`](@ref) — a user-supplied immersed mesh of points, segments or triangles, of any simplex dimension
    `0 ≤ K ≤ min(D, 2)`: a curve or a surface, a point set, or a codim-0 patch (see [`BoundaryMesh`](@ref)). The
    package walks the precomputed per-cell quadrature rules, which integrate any product of two basis functions of the
    covering cells exactly on every sub-cell, oblique or axis-aligned.
  - An [`Interface`](@ref) — the two-sided interface of a coupled model, which spans both its subdomains and ignores
    `field`.

The first two cases are the two halves of `∂Ω` and they are disjoint, which is worth stating as an identity: for an
immersed model whose `BoundaryMesh` carries the whole immersed part of `∂Ω`, the grid-aligned integrals summed over
`boundary(:all)` plus the mesh integral is an integral over the closed surface `∂Ω`, so `∮ n dS = 0` and
`∮ x·n dS = D·|Ω|` hold up to the quadrature error of the two rules. `test_constraints.jl` pins both.

`integrand(q)` is called at every quadrature point with the payload a form tagged with the same `on=` sees during
assembly, and that [`foreach_quadrature_point`](@ref) passes:

  - `q.x::SVector{D,T}` — physical coordinate.
  - `q.weight::T` — physical-frame weight (Gauss × measure Jacobian).
  - `q.point::Int` — the point's index in `1:nquadpoints(model; on, field)`.
  - `q.state` — always `nothing`; the integrand reads a solution through [`value`](@ref) at `q.x`.
  - `q.normal::SVector{D,T}` — outward unit normal (facet's outward normal for `BoundarySelector` `on`; cell-geometry /
    user-supplied normal for `BoundaryMesh`; the normal from side `a` toward side `b` on an `Interface`).
  - `q.sides::Union{Nothing,Vector{Tuple{Int,Symbol}}}` — the codim-`K` facet identifier on a `BoundarySelector`
    integration; `nothing` on a `BoundaryMesh` or `Interface` integration (no facet `(axis, side)` exists).

`field` names the subdomain to integrate over on a multi-domain (coupled) model. Each subdomain carries its own
discretisation and its own boundary, so `on` alone does not identify a set of facets there:
`boundary(axis=1, side=:upper)` is a face of every subdomain that has one. Naming a field selects that field's space and
integrates over its boundary. On a single-domain model `field` is unnecessary and defaults to the only space.

Omitting `field` on a multi-domain model raises. The alternative defaults are both wrong rather than merely
inconvenient: returning one subdomain's answer is silent and arbitrary (it depends on which form happens to reference
the selector first), and summing the subdomains does not give `∂Ω` either — an interface shared by two subdomains lies
in both `∂Ω₁` and `∂Ω₂` while belonging to neither's physical boundary, so the sum double-counts what it should omit.
This matches the package's existing policy for entry points whose meaning needs a subdomain ([`move!`](@ref),
[`activate!`](@ref), [`foreach_quadrature_point`](@ref)).

The accumulator is `result += q.weight * integrand(q)`; the result type is whatever `q.weight * integrand(q)` yields on
the first sample. Throws `ArgumentError` if no admissible region exists on the selected portion of the boundary.
"""
function boundary_integral(integrand, model::Model; on::Union{BoundarySelector,BoundaryMesh,Interface},
                           field::Union{Nothing,Symbol}=nothing)
    list = _region_list(model, on, _target_space(model, on, field))
    # A third cause joins "no such facet" and "every cell masked out" now that a facet rule is trimmed: a selector all
    # of whose faces lie outside Ω resolves to regions that carry no quadrature. Naming it here is the difference
    # between a user checking their selector and a user checking their geometry.
    _point_count(list) == 0 && throw(ArgumentError("no admissible boundary quadrature for on=$(on); check the " *
                                                   "selector or mesh, any level masks, and whether the selected " *
                                                   "faces lie outside Ω"))
    return _walk(nothing, nothing, list, nothing) do result, q
        contribution = q.weight * integrand(q)
        return result === nothing ? contribution : result + contribution
    end
end
