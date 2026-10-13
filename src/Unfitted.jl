"""
    Unfitted

Julia implementation of unfitted multi-level hp refinement on axis-aligned Cartesian grids.

The approximation is built as a superposition of overlay contributions, one per mesh level:

    u_h(x) = Σₖ u_h^(k)(x)

Each overlay contribution `u_h^(k)` is extended by zero outside its overlay domain `Ω^(k)`. Overlay levels are
independent — they are *not* topologically merged with their parents, do not introduce hanging-node constraints, and
impose homogeneous boundary conditions on their artificial overlay boundary `Γ_o^(k) = ∂Ω^(k) \\ ∂Ω`. Integration on the
resulting non-matching meshes uses admissible C∞ boxes built from the union of per-axis element-boundary coordinates of
all participating levels.

The package supports finite-cell-method (FCM) immersed integration through a [`PhysicalDomain`](@ref): the geometry of
`Ω` is a CSG combination of smooth level-set leaves (`leaf`, `intersect`, `union`, `setdiff`, `complement`; a single
callable `φ` with `Ω = {φ ≤ 0}` is the degenerate case). Cut cells use a non-negative moment-fit quadrature rule whose
moments come from Saye's dimension-reduction implicit quadrature on the level set (R. I. Saye, SIAM J. Sci. Comput.
**37** (2015) A993; J. Comput. Phys. **448** (2022) 110720), exact and octree-depth-independent on smooth cut cells. See
the top of `src/fcm.jl` and `src/implicit.jl`.

Imported boundary-mesh geometry is supported as a signed-distance level set through a package extension: with `FileIO`
and `MeshIO` loaded (which pull in `GeometryBasics`, the extension's third trigger), `mesh_levelset(mesh)` turns a
closed `BoundaryMesh` — a 2D segment loop or a 3D triangle surface — into a [`LevelSet`](@ref) leaf, and
`stl_levelset("part.stl")` reads an STL into one. Both compose with the CSG combinators. See `ext/UnfittedMeshIOExt.jl`.

ParaView output is a package extension too: with `WriteVTK` loaded, [`write_vtk`](@ref) writes a solution as a
multiblock bundle, [`vtk_series`](@ref) collects such bundles into a time series, and [`write_quadrature_vtm`](@ref)
writes the quadrature-point cloud. The extension serialises the consumer-neutral records `src/postprocessing.jl`
samples, and a call made without WriteVTK loaded raises an error saying to run `using WriteVTK`. See
`ext/UnfittedWriteVTKExt.jl`.

Reference for the unfitted multi-level hp method:

> J. N. Schmäke and M. Ruess, *Unfitted Multi-Level hp Refinement for Localized and Moving Solution Features*,
> arXiv:2604.25797.

# A small first example

```julia
using Unfitted

# Build a base discretization on the unit square, add one centered overlay
# at higher polynomial order, set up a Poisson problem with homogeneous
# Dirichlet data on every physical face, and solve.
V        = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=2)
V        = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3)
problem  = poisson(V; source=x -> 1.0,
                       dirichlet=[dirichlet(0.0; on=boundary(:all))])
model    = prepare(problem)
solution = solve!(model)
```

`CONTRIBUTING.md` documents the full set of mathematical contracts, public-API design rules, code style, and developer
workflow.
"""
module Unfitted

using LinearAlgebra
using SparseArrays
using StaticArrays
using FastGaussQuadrature
using ForwardDiff
using NearestNeighbors
using NonNegLeastSquares

export AxisBox, GeometryTolerance, PhysicalDomain, physical_domain, classify_cell, LevelSet, leaf, complement,
       levelset_value, stl_levelset, mesh_levelset, CartesianMesh, Space, IntegratedLegendre, bspline, Field, BlockForm,
       LoadForm, TrialChannels, TestChannels, WeakForm, Problem, Model, Solution, box, mesh, space, overlay, field,
       block, loadform, boundary, dirichlet, update_dirichlet!, neumann, couple, interface, Interface, InterfaceForm,
       onside, jump_sign, BoundaryMesh, segment_mesh, polyline_mesh, triangle_mesh, points_mesh, poisson, mass,
       mass_form, mass_block, stiffness, stiffness_form, stiffness_block, load, source_form, source_load, load_vector,
       prepare, assemble, assemble_matrix, assemble_vector, assemble!, foreach_quadrature_point, nquadpoints, solve!,
       solution, move!, moved, moved_space, activate!, deactivate!, active_cells, support_extension, dilate,
       cell_orders, elevate, elevated, cell_order, nominal_order, ladder, adapt, adapted, is_nested, overlapping_cells,
       level_count, transfer, L2Projection, Rewire, estimate, mark_cells, decide, refine, coarsen, QuadField, RBFP0,
       write_vtk, vtk_series, VTKSeries, write_quadrature_vtm, l2_error, boundary_integral, value, field_gradient,
       diagnostics, active_unknowns, cell_indices, cell_box, center, value_vec, gradient_tensor, symmetric_gradient

# ── Extension stubs ───────────────────────────────────────────────────────────
#
# These names are populated by `ext/UnfittedTensorsExt.jl` when Tensors.jl is loaded. They are declared here, not in the
# extension module, so that any downstream code can write
# `using Unfitted: symmetric_gradient, value_vec, gradient_tensor` unconditionally; loading Tensors.jl alongside
# Unfitted installs the concrete methods on these stubs. Calling any of them without a Tensors-compatible signature
# raises the usual `MethodError`, which is the intended behaviour: the names are only meaningful once tensor algebra is
# in scope.
#
# `symmetric_gradient` is declared here, not extended from Tensors.jl, because Tensors.jl does not export a function
# under that name; the closest cousins it ships are the AD entry points `gradient`, `hessian`, and `divergence`.
function symmetric_gradient end
function value_vec end
function gradient_tensor end

# `bspline(; continuity = :maximal)` is the public factory for the open-knot tensor B-spline basis family. It returns a
# deferred specification that the `instantiate_basis` hook turns into a concrete `BSplineFamily` once the level's mesh
# is known (the per-axis degree comes from the `order` keyword on `space` / `overlay`, matching the integrated Legendre
# convention). The struct and all methods live in `ext/UnfittedBasicBSplineExt.jl` and are installed when
# BasicBSpline.jl is loaded alongside Unfitted. The stub here lets downstream code write `using Unfitted: bspline`
# unconditionally; calling `bspline(...)` without BasicBSpline loaded raises the usual `MethodError`, which is the
# intended behaviour.
function bspline end

# `mesh_levelset(mesh::BoundaryMesh)` and `stl_levelset(path)` build a signed- distance [`LevelSet`](@ref) leaf from a
# closed boundary mesh — a 2D segment loop (`BoundaryMesh{2,T,1}`) or 3D triangle surface (`BoundaryMesh{3,T,2}`);
# `stl_levelset` reads an STL file into the latter. Both return a `leaf` whose level set is negative inside the solid,
# suitable for `physical_domain` and CSG composition, and carrying the exact Lipschitz constant of the mesh's
# inside/outside test (`1.0` by default, `Inf` under `orientation = :winding`, whose sign can jump). The signed-distance
# kernel and the STL loader live in `ext/UnfittedMeshIOExt.jl` and are installed when FileIO, MeshIO, and GeometryBasics
# are loaded alongside Unfitted. The stubs here let downstream code write `using Unfitted: mesh_levelset`
# unconditionally; calling them without those packages loaded raises the usual `MethodError`.
function stl_levelset end
function mesh_levelset end

# ── VTK output: the WriteVTK extension ────────────────────────────────────────
#
# ParaView output is the package extension `ext/UnfittedWriteVTKExt.jl`, installed when WriteVTK.jl is loaded alongside
# Unfitted. It serialises the consumer-neutral records `src/postprocessing.jl` samples and decides everything
# VTK-specific there: cell types, padding, number types, the multiblock layout and the file names. The public names are
# declared and documented here rather than in the extension, so that `?write_vtk` answers in a bare `using Unfitted`
# session (an extension's docstrings land in the extension module's own docs, not this one's) and downstream code can
# name them unconditionally.
#
# Each entry point is one fallback method over every argument list, strictly less specific than every method the
# extension installs, and its body is `_extension_missing`: without the extension the call raises an error that says to
# load WriteVTK, before anything touches the file system, and with it a call no method accepts raises the ordinary
# `MethodError`. That error bears the one trace the fallback leaves: since the fallback matches every call, Julia prints
# it among the closest candidates and adds that the error was thrown manually, as if the method were deliberately left
# unimplemented. The core defines no other method of these functions: one more specific than an extension method in
# some argument would take that method's calls, or make them ambiguous. The series handle is the abstract `VTKSeries`,
# because its concrete type holds a WriteVTK collection and is therefore the extension's; the two series operations are
# documented below on their abstract signatures, which no method of the core implements.

"""
    VTKSeries

Handle for a ParaView collection (`.pvd`) being written frame by frame. Opened by [`vtk_series`](@ref), extended by
`write_vtk(series, time, solution, model)`, and finished by `close(series)`; a series comes only from `vtk_series`.

`VTKSeries` is abstract. The concrete handle, which holds the open collection, the directory its frame bundles are
written into and the number of frames written so far, is defined by the WriteVTK package extension that
`using WriteVTK` loads; dispatch on, and test against, `VTKSeries`.
"""
abstract type VTKSeries end

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

Requires `using WriteVTK`, which loads the package extension the VTK writers live in. A call made without it raises an
error that says so, and writes nothing.

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
write_vtk(args...; kwargs...) = _extension_missing(write_vtk, args, "WriteVTK")

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
write_vtk(series::VTKSeries, time, solution, model)

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

Requires `using WriteVTK`, like [`write_vtk`](@ref).
"""
vtk_series(args...; kwargs...) = _extension_missing(vtk_series, args, "WriteVTK")

"""
    close(series::VTKSeries) -> String

Write the ParaView collection file of `series` and return its path. The frame bundles are already on disk — each is
written as it is handed over — so this writes only the `.pvd` index over them. Calling it a second time writes nothing
and returns the same path; a frame written after it is refused.
"""
Base.close(series::VTKSeries)

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

Requires `using WriteVTK`, like [`write_vtk`](@ref).
"""
write_quadrature_vtm(args...; kwargs...) = _extension_missing(write_quadrature_vtm, args, "WriteVTK")

# The body of an extension stub's fallback method `f(args...; kwargs...)`: what a call no method of `f` accepts raises.
# While the package extension `Unfitted<package>Ext` — the name every extension of this package carries, after the
# package that loads it — is not loaded, the error names that package, which is the remedy, and it is raised before the
# call has touched anything. Once the extension is loaded the call simply matched none of its methods, and the ordinary
# `MethodError` says so (without the keyword arguments, which it does not carry) rather than asking the user to load
# what is loaded. Generic in the package, so a further extension's stubs reuse it.
function _extension_missing(f, args, package::String)
    Base.get_extension(@__MODULE__, Symbol(:Unfitted, package, :Ext)) === nothing &&
        error("$(nameof(f)) is provided by Unfitted's $package extension: run `using $package` to load it")
    throw(MethodError(f, args))
end

include("geometry.jl")
include("physical.jl")
include("implicit.jl")
include("basis.jl")
include("fcm.jl")
include("mesh.jl")
include("intersections.jl")
include("coverage.jl")
include("ladder.jl")
# Two references of the dof layer run against the include order and resolve at call time rather than at include time.
# `dofs.jl` carries integrated Legendre's `_cell_modes`, a family method of a hook whose generic method is in `mesh.jl`
# and which `Level`'s constructor calls, so a name defined below is reached from above — the same shape an extension
# takes. And the L² Dirichlet projection in `dirichlet.jl` accumulates through the channel helpers of `problems.jl`
# (`_as_test_channels`, `_test_value_contribution`), which load after it. Otherwise the dof layer depends only on what
# loads above it: the four steps a layout is built in live in `dofs.jl` and `dirichlet.jl`, and `model.jl`, below both,
# composes them.
include("dofs.jl")
include("dirichlet.jl")
include("surface.jl")
include("problems.jl")
include("coupling.jl")
include("model.jl")
include("assembly.jl")
include("solvers.jl")
include("projection.jl")
include("adaptivity.jl")
include("data.jl")
include("postprocessing.jl")
include("api.jl")

end
