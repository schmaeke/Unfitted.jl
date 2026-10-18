"""
    UnfittedWriteVTKExt

Package extension that writes Unfitted's post-processing output as ParaView files through
[`WriteVTK.jl`](https://github.com/JuliaVTK/WriteVTK.jl). Loading `WriteVTK` next to `Unfitted` installs the methods
below; without it the entry points, which `src/stubs.jl` declares and documents, raise an error that says to run
`using WriteVTK`.

The extension samples nothing itself. `src/postprocessing.jl` samples a solution, the meshes of its levels and the
quadrature-point cloud into consumer-neutral records — `Unfitted.SampleGrid`, whose contract heads that file — and
this extension serialises them. Everything VTK-specific is decided here and nowhere else:

  - the cell types: a grid of `D`-cubes as VTK lines, quads or hexahedra, a vertex cloud as poly-data vertices;
  - the padding: points and two-component vectors widened to three components with zeros;
  - the number types VTK stores: `Bool` written as `UInt8`, and a type VTK has no array for refused naming its entry;
  - the multiblock layout of a bundle, and the file names of a bundle, a series frame and the quadrature cloud.

Provided pieces:

  - `Unfitted.write_vtk(path, solution, model; …)` — one bundle;
  - `Unfitted.vtk_series(path; frames)`, `Unfitted.write_vtk(series, time, solution, model; …)` and
    `close(series)` — a `.pvd` time series of bundles, through the concrete series handle `Series <: VTKSeries`;
  - `Unfitted.write_quadrature_vtm(path, model)` — the quadrature-point cloud.

Every dataset is built in memory and the files are written only once the bundle is complete, so a call that fails — on
a keyword it refuses, or in a callback that throws — leaves nothing on disk.
"""
module UnfittedWriteVTKExt

using Unfitted
using Unfitted: SampleGrid, Solution, Model, VTKSeries, _sample_fields, _sample_levels, _sample_quadrature, _cell_count
using StaticArrays: SVector
using WriteVTK

# ── Series handle ─────────────────────────────────────────────────────────────

# The concrete `VTKSeries` that `vtk_series` returns: the open ParaView collection, the directory its frame bundles are
# written into, and the number of frames written so far, which names the next frame. It lives here rather than in the
# core because its collection is a WriteVTK type; the core knows only the abstract handle. The field names are kept
# from when the type was the core's own, so a script reading `series.count[]` keeps working.
struct Series <: VTKSeries
    collection::WriteVTK.CollectionFile
    frames::String
    count::Base.RefValue{Int}
end

# ── Public VTK export ─────────────────────────────────────────────────────────
#
# The methods behind the entry points declared and documented in `src/stubs.jl`. Each samples the records first
# (`_sample_fields` and `_sample_levels`, or `_sample_quadrature`), checking every keyword and callback value, builds
# the whole multiblock in memory, and only then makes the output directory and writes the files.

function Unfitted.write_vtk(path::AbstractString, solution::Solution, model::Model; kwargs...)
    base = _vtk_base_path(path)
    # Everything is sampled, every keyword and callback value checked, before a file is opened: a call refused or a
    # callback that throws leaves nothing. Not WriteVTK's do-block form either, which saves the multiblock in a
    # `finally`; the datasets stay in memory until `vtk_save`, and the output directory is made just before it.
    vtm = _vtk_multiblock(base, solution, model; kwargs...)
    mkpath(dirname(base))
    return vtk_save(vtm)
end

function Unfitted.vtk_series(path::AbstractString; frames=nothing)
    base = _vtk_base_path(path)
    directory = frames === nothing ? base * "_frames" : String(frames)
    mkpath(dirname(base))
    mkpath(directory)
    return Series(paraview_collection(base * ".pvd"), directory, Ref(0))
end

function Unfitted.write_vtk(series::Series, time::Real, solution::Solution, model::Model; kwargs...)
    # `close` has written the collection file and freed its XML document. Without this check the frame would be sampled
    # in full and then fail inside the XML library (`XMLNoRootError`) when the collection is asked to record it.
    isopen(series.collection) ||
        throw(ArgumentError("cannot write a frame to a closed series: close(series) already wrote its collection"))
    n = series.count[] + 1
    base = joinpath(series.frames, "frame_" * lpad(n, 4, '0'))
    # The builder of a single bundle, so a frame is byte for byte what `write_vtk(path, …)` writes.
    vtm = _vtk_multiblock(base, solution, model; kwargs...)
    # Hands the *open* multiblock over: the collection closes it — which is what writes the frame's files — and records
    # its path, relative to the `.pvd`, against `time`. The counter advances only once that has succeeded, so a rejected
    # frame does not burn a frame number; and, as for a single bundle, the directory is made only now, so it writes
    # nothing either.
    mkpath(dirname(base))
    collection_add_timestep(series.collection, vtm, time)
    series.count[] = n
    return vtm.path
end

function Base.close(series::Series)
    vtk_save(series.collection)
    return series.collection.path
end

function Unfitted.write_quadrature_vtm(path::AbstractString, model::Model{D}) where {D}
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

# Strip a `.vtm` / `.vtu` / `.vtp` / `.pvd` extension from `path` so callers can pass either `"out"` or `"out.vtm"`
# interchangeably. Every file name the writers hand to `WriteVTK` then carries its own extension again: given a name
# without one, `WriteVTK` appends it, but warns whenever the name has any other extension — and a dotted base name such
# as `out_t=0.5` has one (`.5`), so every file of a time-stamped bundle used to warn.
function _vtk_base_path(path::AbstractString)
    base, ext = splitext(String(path))
    return ext in (".vtm", ".vtu", ".vtp", ".pvd") ? base : String(path)
end

# The multiblock of one bundle rooted at `base`, sampled and built and still open: nothing is on disk until the caller
# saves it, or hands it to a series' collection, which saves it. The one keyword surface of both `write_vtk` methods:
# it takes `ascii`, `append` and `compress` for `vtk_grid` and `level_meshes` for the shape of the bundle, and passes
# `field`, which selects the level meshes too, and every other keyword to `_sample_fields`, after the dimension check,
# so a bundle and a series frame are refused, sampled and serialised alike.
#
# One field block per data record, named after its field and holding that field's data grid and the level meshes
# `_sample_levels` files under it as sibling leaf datasets, so a coupled field is a self-contained, independently
# toggled unit in ParaView: `<field> → { data, level_… }`. The children of a field block are all leaves — never a mix of
# a leaf and a sub-block, which breaks ParaView's Extract-Block resolution — and every leaf's disambiguating name part
# is ASCII. ParaView derives its data-assembly node names from the dataset names and keeps only ASCII identifier
# characters, so a Unicode field name (`θ₁`, `θ₂`), or a name disambiguated only by Unicode digits, collapses to one
# node and the blocks become indistinguishable. The data leaf is keyed by the ASCII record index (`data_<i>`, plain
# `data` when there is one record), NOT by the field name, and the meshes by the ASCII level id (`level_<id>_<role>`).
# The field name is still the block label, where a Unicode collision is harmless: the user extracts leaves.
function _vtk_multiblock(base::AbstractString, solution::Solution, model::Model{D}; ascii::Bool=false,
                         append::Bool=true, compress=true, level_meshes::Bool=true, field=nothing, kwargs...) where {D}
    _check_vtk_dimension(Val(D))
    records = _sample_fields(solution, model; field, kwargs...)
    meshes = level_meshes ? _sample_levels(model; field) : nothing
    vtm = vtk_multiblock(base * ".vtm")
    for (i, record) in enumerate(records)
        block = multiblock_add_block(vtm, string(record.field.name))
        leaf = length(records) == 1 ? "data" : "data_$i"
        _add_dataset!(block, base, leaf, leaf, record.data; ascii, append, compress)
        meshes === nothing && continue
        for mesh in meshes
            mesh.field.name === record.field.name || continue
            label = "level_$(mesh.level)_$(mesh.role)"
            _add_dataset!(block, base, label, label * "_mesh", mesh.grid; ascii, append, compress)
        end
    end
    return vtm
end

# VTK supports cells in 1D, 2D, and 3D natively. Higher-dimensional spaces have no corresponding VTK cell type and are
# rejected at the top of the public entry points so the user sees the dimension limitation up front.
function _check_vtk_dimension(::Val{D}) where {D}
    D <= 3 || throw(ArgumentError("VTK export supports dimensions 1, 2, and 3; got D=$D"))
end

# Write one `SampleGrid` as a dataset leaf of the multiblock `parent`: the file `<base>_<suffix>.vtu` (`.vtp` for a
# vertex cloud) under the block name `name`, with the record's point arrays and then its cell arrays attached in list
# order. One serialiser for every dataset a bundle holds — the data grids, the level meshes, the quadrature clouds —
# so the cell type, the padding and the attach order are decided in one place. The attach order is the order of the
# bytes in an appended file, which is why it is the record's list order and nothing else.
#
# A record's arrays come as `AbstractVector`s of several element types, so each is attached through
# `Base.inferencebarrier` (the idiom of `_coverage_constraints!` in `dofs.jl`): the call is resolved at run time on the
# array's concrete type. Without it, inference descends into the one method `_vtk_attach!` with the element type
# unknown and infers its conversions and WriteVTK's array writer for an abstract array, code that never runs: on the
# first `write_vtk` of a two-level 2D overlay, 232 more code instances inferred.
function _add_dataset!(parent, base::AbstractString, name::AbstractString, suffix::AbstractString, g::SampleGrid{D,T,K};
                       kwargs...) where {D,T,K}
    path = joinpath(dirname(base), basename(base) * "_" * suffix) * (K == 0 ? ".vtp" : ".vtu")
    vtk = vtk_grid(path, _vtk_points(g), _vtk_cells(g); kwargs...)
    multiblock_add_block(parent, vtk, name)
    attach! = Base.inferencebarrier(_vtk_attach!)
    for (key, array) in g.point_data
        attach!(vtk, VTKPointData(), key, array, :point_data)
    end
    for (key, array) in g.cell_data
        attach!(vtk, VTKCellData(), key, array, :cell_data)
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

# Attach one record array to the dataset `vtk` at `location` (`VTKPointData()` or `VTKCellData()`), as VTK stores it
# (`_vtk_array`). Specialised on the array's concrete type, behind `_add_dataset!`'s barrier.
function _vtk_attach!(vtk, location, name::String, array::AbstractVector, kind::Symbol)
    (vtk[name, location] = _vtk_array(array, name, kind))
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
#
# `R` is the element type written: the array's own, or `UInt8` for `Bool`.
function _vtk_array(array::AbstractVector{V}, name::AbstractString, kind::Symbol) where {V}
    R = eltype(V) === Bool ? UInt8 : eltype(V)
    R in _VTK_NUMBERS ||
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

end
