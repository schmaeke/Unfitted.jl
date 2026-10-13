# Post-processing: what a consumer of a solved model needs, in three parts.
#
#   * **Sampling.** A solution, the meshes of its levels and the quadrature-point cloud are sampled into
#     consumer-neutral records, so that every consumer — the VTK writer below, a plotting front end — reads the same
#     geometry and the same arrays rather than re-deriving them. The contract is below.
#   * **VTK export.** [`write_vtk`](@ref) writes the records of a `(solution, model)` pair as a ParaView multiblock
#     bundle; [`vtk_series`](@ref) collects a sequence of those bundles into a ParaView `.pvd` collection, so a feature
#     that moves — and the mesh that follows it — can be watched over time; [`write_quadrature_vtm`](@ref) writes the
#     quadrature-point cloud, for inspecting where the integration regions land. All three serialise records with
#     `WriteVTK.jl`, and everything VTK-specific — cell types, padding to three components, the number types VTK
#     stores, the multiblock layout, the file names — is decided there and nowhere else.
#   * **Solution evaluation.** [`value`](@ref) and [`field_gradient`](@ref) evaluate the superposed solution at
#     arbitrary physical points; [`l2_error`](@ref) computes an analytic / manufactured-solution error over the model's
#     own integration plan; [`boundary_integral`](@ref) integrates a callback over a portion of the boundary or an
#     interface.
#
# All of them share the overlay-zero-extension convention: a level whose mesh does not cover a point contributes zero
# there. That mirrors the assembly contract from `CONTRIBUTING.md`'s "The superposition model" section.
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
# A plotting extension would read the same records. With Makie, a 2D grid is
# `mesh(Point.(g.points), [QuadFace((4c − 3):(4c)) for c in 1:_cell_count(g)]; color = <a point array>)` and the
# quadrature cloud `scatter(Point.(entry.grid.points); color = <its weight array>)`, and the extension's public
# functions would be declared in the core, as every package extension's are in `Unfitted.jl`.

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
# so the corners of neighbouring cells that share a face are bit for bit the same points. Dimension-generic: which `D`
# a format can hold is its consumer's question.
function _box_corners(b::AxisBox{D}) where {D}
    v = _box_vertices(b)
    return ntuple(j -> v[xor(j - 1, ((j - 1) >> 1) & 1)+1], Val(2^D))
end

# ── Public VTK export ─────────────────────────────────────────────────────────

"""
    write_vtk(path, solution, model;
              field=nothing, subdivisions=:degree, cut_depth=3, point_data, cell_data,
              level_meshes=true, ascii=false, append=true, compress=true) -> Vector{String}

Write a ParaView bundle rooted at `path`. The top-level file is a `.vtm` multiblock dataset with **one block per
field**, each named after the field and grouping that field's data with its own mesh as sibling leaf datasets so it is a
self-contained, independently-toggled unit — `<field> → { data_<i>, level_… }`:

  - `data_<i>` — the `i`-th field's partitioned solution `.vtu`, whose cells are that field's admissible integration
    regions optionally subdivided into smaller sub-cells (see `subdivisions`), with per-cell `region_id` /
    `cover_count`, per-point user-defined arrays, and (with a [`PhysicalDomain`](@ref)) the field's own `level_set`. A
    write of one field — a single-field model, or the one field picked with `field` — names it simply `data`;
  - `level_<id>_<role>` — optionally, one solid-cell `.vtu` per level of the field's subdomain: every mesh cell with
    per-cell `active`, `covered`, `active_dofs`, `reduced_dofs` (plus `level_id` / `role_id` / `order_max` / `cell_id`),
    so ParaView can colour the mesh by coverage and covered-mode pruning.

A single-field model is the one-block case of this shape. Fields sharing one space emit that space's meshes once, under
the first such field, and count their `active_dofs` / `reduced_dofs` on that field's dof layout — the convention
[`diagnostics`](@ref) follows for its per-level counts — so with several fields over one space the counts reflect the
first field's Dirichlet conditions alone. The tree is homogeneous (a field block holds only leaf datasets) and every
leaf's disambiguating name is **ASCII** — the field *index* `i` for `data_<i>`, the level *id* for the meshes — so
ParaView's Extract Block (whose data-assembly node names keep only ASCII identifier characters) resolves each leaf on
its own even when the field names themselves are Unicode (`θ₁`, `θ₂`).

Keyword arguments:

  - `field` — restrict the output to a single [`Field`](@ref) (or its name `Symbol`); defaults to every field of the
    model.
  - `subdivisions` — per-region subdivision count, sized to resolve the *solution*. `:degree` (default) uses each
    region's parent polynomial order; `:none` keeps one cell per region; a positive integer uses an isotropic count; an
    `NTuple{D,Int}` uses explicit per-axis counts.
  - `cut_depth` — extra bisections applied to the sub-cells that `∂Ω` passes through, sized to resolve the *geometry*.
    `3` by default, `:none` to disable, or an integer in `0:8`. Only regions crossed by the boundary are refined, and a
    model without a [`PhysicalDomain`](@ref) is untouched, so the cost is that of the boundary rather than of the
    volume: in 3D the leaf count grows as `8ᵏ` under `subdivisions` but only as `4ᵏ` under `cut_depth`. See "Resolving
    the immersed boundary" below.
  - `point_data` — `NamedTuple` / `Dict` / `Tuple` of `name => callback` pairs. Each callback is invoked at every vertex
    sample as `f(u, context, x, xi)` and returns a real number, a tuple of real numbers or an `SVector` of real
    numbers, of one shape at every sample. A tuple is written as the `SVector` of its components. The values of one
    entry are promoted to a common element type, which must be one VTK stores — `Float32`, `Float64` or an integer of
    at most 64 bits — or `Bool`, which VTK has no type for and is written as `UInt8` 0 / 1. A two-component value is
    written as a three-component VTK vector with a trailing zero. Any other value (a `Vector`, a matrix, a `String`, a
    `Complex`, a `Rational`, or shapes that differ between samples) is refused with an `ArgumentError` naming the
    entry, and nothing is written. Each callback is called exactly once per sample, but in an unspecified order — across
    the samples and across the entries — so a callback should compute its value from its arguments and not count or
    record its calls. `u(context, xi[, field])` evaluates the current solution at the sample, given the `context` and
    `xi` the callback received: the block's own field by default, or another `field` on the same subdomain. A field on
    another subdomain of a coupled model raises an `ArgumentError`; evaluate it with
    [`value`](@ref)`(solution, model, field, x)` instead. `x` is the physical coordinate and `xi.region` the same point
    in the region's reference frame; `context` carries `location` (`:point`, or `:cell` for a `cell_data` sample),
    `region_id`, the `region` box, the emitted `cell` box, the covering `parents` — each with the `parent_box` a
    callback needs to map `x` into a cell's own frame — and the `model_version` the samples were taken from. Defaults to
    a single `uh` entry that emits the current solution value.
  - `cell_data` — same shape as `point_data` but evaluated at per-cell sample points (the subbox center). Defaults to no
    user-defined cell data; the bundle always includes `region_id` and `cover_count` as built-in cell arrays.
  - `level_meshes` — emit one solid-cell `.vtu` per level when `true` (default), each cell carrying per-cell `active` /
    `covered` / `active_dofs` / `reduced_dofs` data.
  - `ascii` / `append` / `compress` — pass-through to `WriteVTK`'s `vtk_grid` constructor, with its own defaults:
    binary, appended and **compressed**. Compression is the cheapest lever this writer has — measured on a 3D 5×5×5
    order-2 sphere at `subdivisions = 10`, it takes the bundle from 31.60 MB to 3.81 MB, an 8.3× saving for 1.7× the
    write time — and it costs the reader nothing, since ParaView decompresses transparently. Turn it off together with
    `ascii = true, append = false` when a test or a diff needs the arrays as parsable text.

For models whose [`Space`](@ref) carries a [`PhysicalDomain`](@ref), the per-vertex level-set values `φ(x)` are
automatically attached as a built-in `level_set` point array — convenient for visualising `∂Ω` as a zero-level
isocontour or for filtering cut / full / fictitious regions in ParaView. Pass a `level_set` entry in `point_data` to
override or rename it.

Returns the path of every file written: the `.vtm` first, then the datasets in block order. The bundle is built in
memory and written, its directory included, only once it is complete: a call that fails — on a keyword it refuses, or in
a callback that throws — leaves nothing on disk.

## Resolving the immersed boundary

ParaView reconstructs `∂Ω` from the `level_set` array by interpolating it linearly along each cell's edges, so the
surface a Clip draws is a facetisation at the size of the emitted cell — neither `φ`'s own accuracy nor the solution
order improves it. Resolving `∂Ω` therefore means emitting small cells where it runs, which is what `cut_depth` does and
what `subdivisions` cannot: raising `subdivisions` refines every region uniformly, at `2^(D·k)` cells, while `cut_depth`
refines only the sub-cells the boundary crosses, at `2^((D−1)·k)`. Measured on a 3D 5×5×5 order-2 sphere with 81
regions: 5 723 cells against 41 472 at 8 sub-cells per axis across `∂Ω`, and 95 891 against 2 654 208 at 32.

A sub-cell is refined when `|φ(c)| ≤ L·r`, with `c` its center, `r` its half-diagonal, and `L` the largest Lipschitz
constant declared by any leaf of the geometry. That is the exact negation of the uniform-sign certificate the cell
classifier uses, so an unrefined sub-cell provably holds one sign of `φ` throughout, contributes no cut surface, and
cannot disagree with a finer neighbour across a shared face. Every refined sub-cell bottoms out at the same depth, so
the cut cells of one region carry identical corner coordinates where they meet and their facets join exactly: the
clipped surface is watertight within a region, and across regions whose finest cells coincide on their shared face. Two
adjacent cut regions whose finest cells differ in size — regions of different extent, or subdivided at different counts
— meet in T-junctions instead, where the facets of the two sides need not join, and the clipped surface can show a
crack there about as wide as its own facetisation error.

When **no** leaf declares a constant — `leaf(f)` defaults to `lipschitz = Inf` — no such certificate exists and
refinement falls back to the corner signs of `φ`, the same evidence ParaView's own clip works from. A feature of `Ω`
smaller than a sub-cell and missing every corner is then invisible, and the first such write of a session warns so;
`leaf(f; lipschitz = L)` (`1.0` for a signed distance function) restores the guarantee.

Two limits are worth stating plainly. The clip surface is piecewise planar per cell at every depth, so `cut_depth` buys
a finer staircase rather than a smooth surface. And at a CSG crease — an edge or corner where two leaves meet —
`levelset_value` is only `C⁰`, so the clip converges there at first order in the sub-cell size against second order on a
smooth face; an `∩`-corner stays comparatively more faceted at any given depth.

To write a *sequence* of these bundles as one animation — a moving overlay, an adaptive ladder, a time-dependent
solution — open a [`vtk_series`](@ref) and call this function with it in place of a path.
"""
function write_vtk(path::AbstractString, solution::Solution, model::Model{D}; ascii::Bool=false, append::Bool=true,
                   compress=true, kwargs...) where {D}
    _check_vtk_dimension(Val(D))
    base = _vtk_base_path(path)
    # Everything is sampled, every keyword and callback value checked, before a file is opened: a call refused or a
    # callback that throws leaves nothing. Not WriteVTK's do-block form either, which saves the multiblock in a
    # `finally`; the datasets stay in memory until `vtk_save`, and the output directory is made just before it.
    vtm = _vtk_multiblock(base, _sample_fields(solution, model; kwargs...); ascii, append, compress)
    mkpath(dirname(base))
    return vtk_save(vtm)
end

"""
    VTKSeries

Handle for a ParaView collection (`.pvd`) being written frame by frame. Opened by [`vtk_series`](@ref), extended by
`write_vtk(series, time, solution, model)`, and finished by `close(series)`. It carries the open collection, the
directory its frame bundles are written into, and the number of frames written so far.
"""
struct VTKSeries
    collection::WriteVTK.CollectionFile
    frames::String
    count::Base.RefValue{Int}
end

"""
    vtk_series(path; frames=nothing) -> VTKSeries

Open a ParaView collection at `path` for a sequence of solution bundles — the transient sibling of [`write_vtk`](@ref).
Write one frame per call, then `close` the series to write the collection file:

```julia
series = vtk_series("output/laser")
for step in 1:nsteps
    t, u, model = advance!(…)
    write_vtk(series, t, u, model)
end
close(series)                                  # → "output/laser.pvd"
```

Opening `laser.pvd` in ParaView plays the sequence: the time slider carries the times the frames were written at, and
the animation shows the solution *and* the mesh that produced it.

  - `path` — the collection file, with or without the `.pvd` extension.
  - `frames` — the directory the frame bundles go into. Defaults to `<path>_frames`, a sibling of the collection. The
    collection references each frame by a path relative to itself, so collection and frames move together as one
    directory tree.

Four properties are worth stating explicitly, because they are what makes this usable for a moving feature rather than
merely for a fixed mesh in time:

  - **Every frame is a full bundle** — the data grid *and* the per-level meshes — written under its own base name
    `frame_<n>`, with `n` counted from one and zero-padded so a directory listing sorts. On a moving overlay or an
    adaptive ladder the mesh differs from frame to frame, and that is usually the thing one wants to see; a series that
    emitted the mesh once could not show it.
  - **Keywords are per frame.** Every keyword of [`write_vtk`](@ref) is accepted on each call, with the same meaning and
    the same default, because both the sampled data and the model change as a run proceeds.
  - **The collection records each frame's own physical time**, exactly as it is given and in the order given, so a run
    with an adaptive Δt is never forced onto a uniform frame grid.
  - **Nothing is deleted.** A re-run that writes fewer frames than the previous one leaves the surplus frame files in
    place, unreferenced by the new collection — invisible in ParaView, visible in a directory listing. Clearing the
    frames directory first, if that matters, is the caller's decision; a library that removed files a user might want is
    not one.

The series deliberately has no schedule of its own — no `every`, no `times`. Which steps of a time loop deserve a
picture is a property of the loop, not of the writer, and a frame costs what a `write_vtk` call costs.
"""
function vtk_series(path::AbstractString; frames=nothing)
    base = _vtk_base_path(path)
    directory = frames === nothing ? base * "_frames" : String(frames)
    mkpath(dirname(base))
    mkpath(directory)
    return VTKSeries(paraview_collection(base * ".pvd"), directory, Ref(0))
end

"""
    write_vtk(series::VTKSeries, time, solution, model; kwargs...) -> String

Write one frame of a [`vtk_series`](@ref) and register it in the collection at physical `time`. Returns the path of the
frame's `.vtm`.

The frame is exactly the bundle `write_vtk(path, solution, model; …)` writes, under a name the series chooses
(`frame_<n>` in the series' frames directory), and every keyword of that method is accepted here unchanged — see
[`vtk_series`](@ref) for what the series does and does not do with the frames it is handed. `time` is recorded in the
collection and is otherwise unused: it need not be a step index, a multiple of anything, or evenly spaced. A frame that
fails writes nothing and does not use up a frame number. A series that has been closed takes no more frames: a frame
written to it raises an `ArgumentError`, before any sampling.
"""
function write_vtk(series::VTKSeries, time::Real, solution::Solution, model::Model{D}; ascii::Bool=false,
                   append::Bool=true, compress=true, kwargs...) where {D}
    # `close` has written the collection file and freed its XML document. Without this check the frame would be sampled
    # in full and then fail inside the XML library (`XMLNoRootError`) when the collection is asked to record it.
    isopen(series.collection) ||
        throw(ArgumentError("cannot write a frame to a closed series: close(series) already wrote its collection"))
    _check_vtk_dimension(Val(D))
    n = series.count[] + 1
    base = joinpath(series.frames, "frame_" * lpad(n, 4, '0'))
    # The serialiser of a single bundle, so a frame is byte for byte what `write_vtk(path, …)` writes.
    vtm = _vtk_multiblock(base, _sample_fields(solution, model; kwargs...); ascii, append, compress)
    # Hands the *open* multiblock over: the collection closes it — which is what writes the frame's files — and records
    # its path, relative to the `.pvd`, against `time`. The counter advances only once that has succeeded, so a rejected
    # frame does not burn a frame number; and, as for a single bundle, the directory is made only now, so it writes
    # nothing either.
    mkpath(dirname(base))
    collection_add_timestep(series.collection, vtm, time)
    series.count[] = n
    return vtm.path
end

"""
    close(series::VTKSeries) -> String

Write the ParaView collection file of `series` and return its path. The frame bundles are already on disk — each is
written as it is handed over — so this writes only the `.pvd` index over them. Calling it a second time writes nothing
and returns the same path; a frame written after it is refused.
"""
function Base.close(series::VTKSeries)
    vtk_save(series.collection)
    return series.collection.path
end

"""
    write_quadrature_vtm(path, model) -> Vector{String}

Write a ParaView multiblock bundle of the model's quadrature-point cloud. One `.vtp` block is emitted per *distinct
combination of covering parent levels*: every region of every subdomain integration plan is keyed by
`sort([parent.level for parent in region.parents])` and regions with the same level signature land in the same block.
Level ids are unique across subdomains, so a coupled model's subdomains land in different blocks and all of them are
written. Each block stores the physical quadrature-point coordinates and per-point physical weight
(`weight = w × jacobian`) as point data.

Useful for inspecting where the integration regions actually land in ParaView (the regions / cuts / α-FCM weights are
not visible from the solution VTK alone) and for diagnosing small-overlap warnings — the small regions show up as point
clusters with tiny weights. Returns the path of every file written, the `.vtm` first.
"""
function write_quadrature_vtm(path::AbstractString, model::Model{D}) where {D}
    _check_vtk_dimension(Val(D))
    base = _vtk_base_path(path)
    # As for `write_vtk`: the cloud is in memory and every dataset added before the directory is made and the bundle
    # saved, so a call that fails leaves nothing on disk.
    vtm = vtk_multiblock(base * ".vtm")
    for entry in _sample_quadrature(model)
        suffix = isempty(entry.signature) ? "none" : join(entry.signature, "_")
        _add_dataset!(vtm, base, "levels_" * suffix, "quadrature_levels_" * suffix, entry.grid)
    end
    mkpath(dirname(base))
    return vtk_save(vtm)
end

# ── VTK serialisation ─────────────────────────────────────────────────────────

# The multiblock of one bundle, built from the sampled records and still open: nothing is on disk until the caller
# saves it, or hands it to a series' collection, which saves it. One field block per record, named after its field and
# holding that field's data grid and the level meshes the record carries as sibling leaf datasets, so a coupled field
# is a self-contained, independently toggled unit in ParaView: `<field> → { data, level_… }`.
#
# The children of a field block are all leaves — never a mix of a leaf and a sub-block, which breaks ParaView's
# Extract-Block resolution — and every leaf's disambiguating name part is ASCII. ParaView derives its data-assembly
# node names from the dataset names and keeps only ASCII identifier characters, so a Unicode field name (`θ₁`, `θ₂`), or
# a name disambiguated only by Unicode digits, collapses to one node and the blocks become indistinguishable. The data
# leaf is keyed by the ASCII record index (`data_<i>`, plain `data` when there is one record), NOT by the field name,
# and the meshes by the ASCII level id (`level_<id>_<role>`). The field name is still the block label, where a Unicode
# collision is harmless: the user extracts leaves.
function _vtk_multiblock(base::AbstractString, records; ascii, append, compress)
    vtm = vtk_multiblock(base * ".vtm")
    for (i, record) in enumerate(records)
        block = multiblock_add_block(vtm, string(record.field.name))
        leaf = length(records) == 1 ? "data" : "data_$i"
        _add_dataset!(block, base, leaf, leaf, record.data; ascii, append, compress)
        for mesh in record.meshes
            label = "level_$(mesh.level)_$(mesh.role)"
            _add_dataset!(block, base, label, label * "_mesh", mesh.grid; ascii, append, compress)
        end
    end
    return vtm
end

# Write one `SampleGrid` as a dataset leaf of the multiblock `parent`: the file `<base>_<suffix>.vtu` (`.vtp` for a
# vertex cloud) under the block name `name`, with the record's point arrays and then its cell arrays attached in list
# order. One serialiser for every dataset a bundle holds — the data grids, the level meshes, the quadrature clouds —
# so the cell type, the padding and the attach order are decided in one place. The attach order is the order of the
# bytes in an appended file, which is why it is the record's list order and nothing else.
function _add_dataset!(parent, base::AbstractString, name::AbstractString, suffix::AbstractString, g::SampleGrid{D,T,K};
                       kwargs...) where {D,T,K}
    path = joinpath(dirname(base), basename(base) * "_" * suffix) * (K == 0 ? ".vtp" : ".vtu")
    vtk = vtk_grid(path, _vtk_points(g), _vtk_cells(g); kwargs...)
    multiblock_add_block(parent, vtk, name)
    for (key, array) in g.point_data
        vtk[key, VTKPointData()] = _vtk_array(array, key, :point_data)
    end
    for (key, array) in g.cell_data
        vtk[key, VTKCellData()] = _vtk_array(array, key, :cell_data)
    end
    return vtk
end

# A record's points as VTK stores them: always three coordinates, a 1D or 2D point padded with zeros (ParaView accepts
# the inert zeros, so 1D and 2D grids render in the same pipeline as 3D ones). One dense padded copy, deliberately not
# `reinterpret` or WriteVTK's own padding of a `D × n` matrix: that path writes a tuple of views and fills one scalar
# at a time, which changes both the bytes of a compressed file and the write time.
_vtk_points(g::SampleGrid{3}) = g.points
function _vtk_points(g::SampleGrid{D,T}) where {D,T}
    return [SVector{3,T}(ntuple(i -> i <= D ? x[i] : zero(T), Val(3))) for x in g.points]
end

# The record's implicit connectivity spelled as WriteVTK cells: cell `c` owns points `(c − 1)·2ᴷ + 1 : c·2ᴷ`, in the
# face-cyclic order of `_box_corners`, which is VTK's own order for a line, a quad and a hexahedron; a vertex cloud
# (`K = 0`) is one poly-data vertex per point.
_vtk_cells(g::SampleGrid{D,T,0}) where {D,T} = [MeshCell(PolyData.Verts(), SVector{1,Int}(c)) for c in 1:_cell_count(g)]
function _vtk_cells(g::SampleGrid{D,T,K}) where {D,T,K}
    type = (VTKCellTypes.VTK_LINE, VTKCellTypes.VTK_QUAD, VTKCellTypes.VTK_HEXAHEDRON)[K]
    return [MeshCell(type, SVector{1 << K,Int}(ntuple(j -> ((c - 1) << K) + j, Val(1 << K)))) for c in 1:_cell_count(g)]
end

# One record array as VTK stores it. The record keeps a value as computed; VTK gives an array a meaning by its
# component count and has no boolean or rational type, so three things happen here and nowhere else:
#
#   * a two-component vector is widened to three with a trailing zero — a 2-component array is stored but is not a
#     vector to `vtkDataSetAttributes::SetVectors`, so ParaView's Glyph, Warp By Vector and Stream Tracer could not
#     read it, and the padding matches the one `_vtk_points` gives the geometry it is attached to;
#   * `Bool` is written as `UInt8` 0 / 1, which keeps a flag or a mask readable in ParaView;
#   * an element type VTK has no type for (a `Rational`, a `BigFloat`, an `Int128`) is refused, naming the entry —
#     WriteVTK would refuse it as `data type not supported by VTK`, naming neither the entry nor the remedy.
function _vtk_array(array::AbstractVector{V}, name::AbstractString, kind::Symbol) where {V}
    R = _vtk_number(eltype(V))
    R === nothing &&
        throw(ArgumentError("$kind entry \"$name\" returned $(eltype(V)) values, which VTK cannot store: " *
                            "write_vtk accepts a real number, a tuple of reals or an SVector of " *
                            "reals, with values promoting to Float32, Float64, Bool or an integer of " *
                            "at most 64 bits"))
    V <: SVector{2} && return [SVector{3,R}(v[1], v[2], zero(R)) for v in array]
    R === eltype(V) && return array
    return V <: SVector ? [SVector{length(V),R}(v) for v in array] : convert(Vector{R}, array)
end

# The element types a VTK data array can hold, as `WriteVTK` writes them: the machine floats and the integers of up to
# 64 bits.
const _VTK_NUMBERS = (Float32, Float64, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)

# The element type a record array is written in, given the type `T` its values promote to: `T` itself when VTK stores
# it, `UInt8` for `Bool` — VTK has no boolean type and `WriteVTK` refuses one, while 0 / 1 in a byte keeps a flag or a
# mask readable in ParaView — and `nothing` for any other type, which `_vtk_array` refuses.
_vtk_number(T::Type) = T === Bool ? UInt8 : T in _VTK_NUMBERS ? T : nothing

# Strip a `.vtm` / `.vtu` / `.vtp` / `.pvd` extension from `path` so callers can pass either `"out"` or `"out.vtm"`
# interchangeably. Every file name the writers hand to `WriteVTK` then carries its own extension again: given a name
# without one, `WriteVTK` appends it, but warns whenever the name has any other extension — and a dotted base name such
# as `out_t=0.5` has one (`.5`), so every file of a time-stamped bundle used to warn.
function _vtk_base_path(path::AbstractString)
    base, ext = splitext(String(path))
    return ext in (".vtm", ".vtu", ".vtp", ".pvd") ? base : String(path)
end

# VTK supports cells in 1D, 2D, and 3D natively. Higher-dimensional spaces have no corresponding VTK cell type and are
# rejected at the top of the public entry points so the user sees the dimension limitation up front.
function _check_vtk_dimension(::Val{D}) where {D}
    D <= 3 || throw(ArgumentError("VTK export supports dimensions 1, 2, and 3; got D=$D"))
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
function _sample_fields(solution::Solution, model::Model{D,T}; field=nothing, subdivisions=:degree, cut_depth=3,
                        point_data=_default_point_data(), cell_data=NamedTuple(), level_meshes::Bool=true) where {D,T}
    depth = _cut_depth_budget(cut_depth)
    state = FormState(model.dofs, _state_vector(solution, model))
    fields = _sampled_fields(model, field)
    entries = (point=_sample_entries(point_data, :point_data), cell=_sample_entries(cell_data, :cell_data))
    rule = _subdivision_rule(subdivisions, Val(D))
    ws, meshed = _assembly_workspace(model), Space[]
    return map(fields) do fld
        space = _field_space(model.problem, fld.name)
        regions = _region_list(model, nothing, space).regions     # the list the solve integrated the field over
        data = _sample_field(ws, state, space, regions, fld, rule, depth, entries, model.version)
        meshes = @NamedTuple{level::Int, role::Symbol, grid::SampleGrid{D,T,D}}[]
        if level_meshes && !any(s -> s === space, meshed)
            dofs = _field_layout(model.dofs, fld.name).dofs
            coverage = build_coverage(space, dofs.tolerance)
            for level in space.levels
                push!(meshes, (; level=level.id, level.role, grid=_level_grid(level, dofs, coverage)))
            end
        end
        push!(meshed, space)
        return (; field=fld, data, meshes)
    end
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
    throw(ArgumentError("cut_depth must be :none or an integer in 0:8; got $(repr(cut_depth))"))
end

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
                        "integers; got $(repr(subdivisions))"))
end

# The data grid of one field: its sample cells (`_sample_cells`) and their corners, the user's entries and the built-in
# arrays. Point arrays: the user's entries in their order, then `level_set` — `φ` at every point, exactly
# `levelset_value` — when the space carries a `PhysicalDomain` and no entry already has that name. Cell arrays:
# `region_id` (the cell's region in the field's region list), `cover_count` (its number of covering parents), then the
# user's entries. A function barrier typed by the space and its regions, and never specialised on the model or on the
# callback set, which arrive as the model's pin and an untyped entry list.
function _sample_field(ws, state::FormState, space::Space{D,T}, regions::Vector{VolumeRegion{D,T}}, fld::Field, rule,
                       depth::Int, entries, version::Int) where {D,T}
    boxes, region_ids = _sample_cells(space, regions, rule, depth)
    points = SVector{D,T}[x for b in boxes for x in _box_corners(b)]
    u = _sample_accessor(ws, state, regions, fld)
    at_points = (; location=:point, shift=D, boxes, points, region_ids, regions, version)
    at_cells = merge(at_points, (; location=:cell, shift=0))
    point_data = Pair{String,AbstractVector}[name => _sample_array(:point_data, name, f, u, at_points)
                                             for (name, f) in entries.point]
    space.physical === nothing ||
        any(e -> first(e) == "level_set", entries.point) ||
        push!(point_data, "level_set" => _levelset_values(space.physical, points, T))
    cell_data = Pair{String,AbstractVector}["region_id" => region_ids,
                                            "cover_count" => [length(regions[r].parents) for r in region_ids]]
    for (name, f) in entries.cell
        push!(cell_data, name => _sample_array(:cell_data, name, f, u, at_cells))
    end
    return SampleGrid{D,T,D}(points, point_data, cell_data)
end

# `φ` at every point, in the space's number type. A barrier: the space's `PhysicalDomain` is abstractly typed, so one
# dynamic call here makes the loop concrete, where a `levelset_value` call per point would dispatch dynamically.
_levelset_values(physical::PhysicalDomain, points, ::Type{T}) where {T} = T[levelset_value(physical, x) for x in points]

# The `u(context, xi[, field])` a callback receives: the current solution at the callback's sample, through the
# assembly's own field evaluation — the one `q.state` and the quadrature-point walker read. A region's dof values are
# read into the shared `state` once (`_frame!`, `_state_region!`); each call then evaluates the basis at the sample
# (`_refresh!`, values only) and every field there (`_state_point!`), and reads one field off `state`: a scalar for a
# one-component field, an `SVector` over its components otherwise, the count read off the `Field`'s type so the result
# is inferable.
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
        f = _field_index(state.layout, field.name)
        (ws.fptr[f] < ws.fptr[f+1] || isempty(region.parents)) ||
            throw(ArgumentError("write_vtk: u(context, xi, field) reads only fields of the block's " *
                                "own subdomain; evaluate $(field.name) with value(solution, " *
                                "model, $(field.name), x)"))
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
# runs on the callback's and the accessor's concrete types. A concrete element type is kept as it is; mixed number
# types (an `Int` at some samples, a `Float64` at others) and `SVector`s of mixed element types are promoted over every
# sample with `promote_type`, and anything else, shapes that differ between samples, is refused naming the entry. An
# empty walk gives an empty `Float64` array.
function _sample_array(kind::Symbol, name::String, f::F, u::U, s) where {F,U}
    n = length(s.boxes) << s.shift
    n == 0 && return Float64[]
    values = map(k -> _sample_value(kind, name, f(u, _callback_args(s, k)...)), 1:n)
    isconcretetype(eltype(values)) && return values
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
function _level_grid(level::Level{D,T}, dofs::DofLayout{D}, coverage::Coverage{D}) where {D,T}
    mesh, cells = level.mesh, vec(cell_indices(level.mesh))
    covered, linear = coverage.covered[level.id], LinearIndices(mesh.cells)
    active(raw) = any(c -> dofs.active_component[raw, c] != 0, 1:dofs.components)
    reduced(raw) = dofs.elimination_source[raw] in (:coverage, :dedup)
    data = Pair{String,AbstractVector}["level_id" => fill(level.id, length(cells)),
                                       "role_id" => fill(level.role === :base ? 0 : 1, length(cells)),
                                       "order_max" => [maximum(cell_order(level, c)) for c in cells],
                                       "cell_id" => [linear[c] for c in cells],
                                       "active" => [Int(is_active(level.mask, c)) for c in cells],
                                       "covered" => [Int(covered[c]) for c in cells],
                                       "active_dofs" => [count(active, cell_dofs(dofs, level.id, c)) for c in cells],
                                       "reduced_dofs" => [count(reduced, cell_dofs(dofs, level.id, c)) for c in cells]]
    points = SVector{D,T}[x for c in cells for x in _box_corners(cell_box(mesh, c))]
    return SampleGrid{D,T,D}(points, Pair{String,AbstractVector}[], data)
end

# ── Quadrature cloud ──────────────────────────────────────────────────────────

# The model's quadrature-point cloud as records: one `(; signature, grid)` per distinct sorted list of covering parent
# levels, in sorted order, over the regions of every subdomain's integration plan. Level ids are unique across
# subdomains, so a coupled model's subdomains land in different entries and every one of them is kept. A grid is a
# vertex cloud (`K = 0`) of the physical quadrature points, each with its physical weight `weight` (the rule weight
# times the region's Jacobian, as assembly integrates with it), region by region — subdomain order, then plan order —
# and point by point. Both arrays are presized from the regions' point counts rather than grown point by point. Not
# `_quadrature_cloud` (`data.jl`), the bare coordinates in `q.point` order that the RBFP0 transfer indexes.
function _sample_quadrature(model::Model{D,T}) where {D,T}
    groups = Dict{Vector{Int},Vector{VolumeRegion{D,T}}}()
    for subdomain in model.subdomains, region in subdomain.plan.regions
        push!(get!(() -> VolumeRegion{D,T}[], groups, sort!([p.level for p in region.parents])), region)
    end
    return map(sort!(collect(keys(groups)))) do signature
        regions = groups[signature]
        points = Vector{SVector{D,T}}(undef, sum(_region_qpoint_count, regions))
        weights, k = similar(points, T), 0
        for region in regions
            jacobian = _region_jacobian(region)
            for q in 1:_region_qpoint_count(region)
                k += 1
                points[k], weights[k], _ = _region_point(region, q, jacobian)
            end
        end
        return (; signature, grid=SampleGrid{D,T,0}(points, ["weight" => weights], Pair{String,AbstractVector}[]))
    end
end

# ── Solution evaluation ───────────────────────────────────────────────────────
#
# The point evaluations `value` / `field_gradient` (`_level_field`) reconstruct a field one covering cell at a time and
# sum the cells. Every other evaluation — assembly's `q.state`, the quadrature-point walker, `l2_error` and
# `write_vtk`'s sample accessor — evaluates into a workspace's `BasisBank` (`_refresh!` and `_state_point!` in
# `assembly.jl`), which keeps the same association: per-cell partial sums, then the sum over cells.

"""
    value(solution, model, x)
    value(solution, model, x, component)
    value(solution, model, u::Field, x[, component])

Evaluate the superposed solution at physical point `x`. Scalar fields return a scalar; component fields return an
`SVector` across components unless a single `component` is requested. Overlay levels are extended by zero — only levels
whose mesh contains `x` (and whose covering cell is active) contribute. Throws if `x` lies outside the evaluated field's
own subdomain box, at the model's containment tolerance. That box is the embedding domain, not `Ω`: with a
[`PhysicalDomain`](@ref) in play a point outside `Ω` does not raise. It evaluates the polynomial extension inside a cut
cell, and zero inside a cell the fictitious fold deactivated.

`x` accepts `NTuple{D,<:Real}` or `SVector{D,<:Real}`. Multi-field models require the `u::Field` form so the field to
evaluate is unambiguous.
"""
function value(solution::Solution, model::Model{D,T}, x::PointLike{D}) where {D,T}
    return value(solution, model, _default_field(model), x)
end

function value(solution::Solution, model::Model{D,T}, x::PointLike{D}, component::Integer) where {D,T}
    return value(solution, model, _default_field(model), x, component)
end

function value(solution::Solution, model::Model{D,T}, u::Field, x::PointLike{D}) where {D,T}
    _field_quantity(solution, model, u, x, nothing, Val(false))
end

function value(solution::Solution, model::Model{D,T}, u::Field, x::PointLike{D}, component::Integer) where {D,T}
    _field_quantity(solution, model, u, x, component, Val(false))
end

"""
    field_gradient(solution, model, x)
    field_gradient(solution, model, x, component)
    field_gradient(solution, model, u::Field, x[, component])

Evaluate the physical gradient of the superposed solution at physical point `x`. Scalar fields return one `SVector{D}`
gradient; component fields return an `SVector` of component gradients unless a single `component` is requested. Overlay
levels are extended by zero, same as [`value`](@ref). The leading `field_` avoids colliding with `Tensors.gradient` when
both packages are loaded together.

The gradient is the physical gradient (chain-rule scaled by `2 / edge_lengths(parent_box)` per axis), not the
reference-frame gradient.
"""
function field_gradient(solution::Solution, model::Model{D,T}, x::PointLike{D}) where {D,T}
    return field_gradient(solution, model, _default_field(model), x)
end

function field_gradient(solution::Solution, model::Model{D,T}, x::PointLike{D}, component::Integer) where {D,T}
    return field_gradient(solution, model, _default_field(model), x, component)
end

function field_gradient(solution::Solution, model::Model{D,T}, u::Field, x::PointLike{D}) where {D,T}
    _field_quantity(solution, model, u, x, nothing, Val(true))
end

function field_gradient(solution::Solution, model::Model{D,T}, u::Field, x::PointLike{D},
                        component::Integer) where {D,T}
    _field_quantity(solution, model, u, x, component, Val(true))
end

# Component dispatch shared by `value` and `field_gradient`. When `component === nothing`: return a scalar for
# single-component fields and an `SVector` across components for vector fields. When `component` is explicit:
# bounds-check and evaluate that one component.
function _field_quantity(solution::Solution, model::Model{D,T}, u::Field, x::PointLike{D}, component,
                         gradient::Val) where {D,T}
    coefficients, point, layout = _evaluation_data(solution, model, u, x)
    if component === nothing
        layout.components == 1 && return _evaluate_field(coefficients, model, layout, point, 1, gradient)
        return SVector(ntuple(c -> _evaluate_field(coefficients, model, layout, point, c, gradient), layout.components))
    end

    _check_component(layout, component)
    return _evaluate_field(coefficients, model, layout, point, component, gradient)
end

# Pick the implicit field of a single-field model. Multi-field models require an explicit field argument so the caller
# does not silently evaluate the wrong field; this raises with a clear message in that case.
function _default_field(model::Model)
    length(model.problem.fields) == 1 ? first(model.problem.fields) :
    throw(ArgumentError("field argument is required for multi-field models"))
end

# Shared pre-flight for the public `value` / `field_gradient` paths: coefficient check, point coercion, in-domain check,
# field-layout lookup. Returns the triple every evaluator needs.
function _evaluation_data(solution::Solution, model::Model{D,T}, u::Field, x::PointLike{D}) where {D,T}
    coefficients = _checked_coefficients(solution, model)
    point = _point_vector(x, T)
    space = _field_space(model.problem, u.name)
    _assert_point_in_domain(point, space.domain, model.dofs.tolerance)
    return coefficients, point, _field_layout(model.dofs, u.name)
end

# Convert a `PointLike` user input (NTuple or SVector) to a typed `SVector{D,T}` matching the model's scalar type. Used
# at the top of every public evaluation path so the downstream kernels see a single canonical point representation.
_point_vector(x::PointLike{D}, ::Type{T}) where {D,T} = SVector{D,T}(x)

# Reject evaluation points outside a field's own subdomain box, at the model's geometry tolerance so points on the
# boundary (up to `tol.contain`) are accepted. The domain is always passed in by the caller, never taken from the
# problem's representative space: on a coupled model each field is bounded by its own subdomain, not by subdomain 1's.
function _assert_point_in_domain(x::SVector{D,T}, domain::AxisBox{D,T}, tol::GeometryTolerance{T}) where {D,T}
    contains_point(x, domain, tol) || throw(ArgumentError("evaluation point is outside the field's domain"))
end

# Bounds-check a component argument against the field's component count.
function _check_component(layout::FieldLayout, component::Integer)
    1 <= component <= layout.components || throw(ArgumentError("field component out of bounds"))
end

# Sum every level's `_level_field` contribution, the value or the gradient by `G`. The overlay zero-extension comes from
# `_level_field` returning zero when its level does not cover `point`.
function _evaluate_field(coefficients, model::Model{D,T}, layout::FieldLayout{D,T}, point::SVector{D,T},
                         component::Integer, ::Val{G}) where {D,T,G}
    R = promote_type(T, eltype(coefficients))
    result = G ? zero(SVector{D,R}) : zero(R)
    for level in _field_levels(model, layout)
        result += _level_field(coefficients, model, layout, level, point, component, Val(G))
    end
    return result
end

# Levels of the space owning `layout`'s field. For a single-domain model this is the one space's levels; for a coupled
# model it routes each field to its own subdomain's (reindexed) levels, so a point evaluation of field `b` never reaches
# into field `a`'s grid.
_field_levels(model::Model, layout::FieldLayout) = _field_space(model.problem, layout.name).levels

# One level's contribution to the field of `layout` at physical point `x`: its value (`G = false`) or its physical
# gradient (`G = true`). Zero if the level's mesh does not contain `x` (overlay zero-extension) or if the containing
# cell is masked inactive. Otherwise: locate the cell, map `x` to the cell's reference frame, evaluate the basis values
# there — or the physical basis gradients, which `physical_basis_gradients` scales by the chain rule `2 / h_d` of the
# axis-aligned cell — and contract them with the dof values through `dof_value`, so constrained dofs contribute their
# pinned values.
function _level_field(coefficients, model::Model{D,T}, layout::FieldLayout{D,T}, level::Level{D,T}, x::SVector{D,T},
                      component::Integer, ::Val{G}) where {D,T,G}
    R = promote_type(T, eltype(coefficients))
    cell = locate_cell(level.mesh, x; tol=model.dofs.tolerance)
    (cell === nothing || !is_active(level.mask, cell)) && return G ? zero(SVector{D,R}) : zero(R)
    parent_box = cell_box(level.mesh, cell)
    xi = physical_to_reference(parent_box, x)
    φ = G ? physical_basis_gradients(level, cell, parent_box, xi) : basis_values(level, cell, xi)
    return _parent_field(layout, coefficients, cell_dofs(layout.dofs, level.id, cell), φ, component)
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

`u` names the field to measure, exactly as for [`value`](@ref) and [`field_gradient`](@ref). The error is integrated
over that field's own subdomain — its space, its integration plan, its dof layout — so on a coupled model it is each
subdomain's own error, and there is no single number for the union: an interface shared by two subdomains would be
integrated twice. Omitting `u` picks the implied field of a single-field model and raises on a multi-field one, which is
why the multi-field and coupled workflows had no L² error report before this overload existed.

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
    # A walk with the field's dof values, as `foreach_quadrature_point` runs one. The squared error is summed component
    # by component, which is `_squared_norm(u_h − exact)` term for term and keeps the fold type-stable whatever the
    # component count.
    ws = _assembly_workspace(model)
    error_squared, exact_squared = _walk((zero(T), zero(T)), ws, list, state) do (err, ref), q
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
