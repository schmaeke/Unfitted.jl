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
`stl_levelset("part.stl")` reads an STL into one. Both compose with the CSG combinators, and a call made without MeshIO
loaded raises an error saying to run `using MeshIO`. See `ext/UnfittedMeshIOExt.jl`.

ParaView output is a package extension too: with `WriteVTK` loaded, [`write_vtk`](@ref) writes a solution as a
multiblock bundle, [`vtk_series`](@ref) collects such bundles into a time series, and [`write_quadrature_vtm`](@ref)
writes the quadrature-point cloud. The extension serialises the consumer-neutral records `src/postprocessing.jl`
samples, and a call made without WriteVTK loaded raises an error saying to run `using WriteVTK`. See
`ext/UnfittedWriteVTKExt.jl`.

Plots of one- and two-dimensional models are a Makie extension: with a Makie backend loaded (`using CairoMakie`, say),
[`levelplot`](@ref) draws the meshes of a model's levels coloured by order, activity, coverage or an error estimate,
and [`quadratureplot`](@ref) draws its quadrature points. Neither needs a solution. See `ext/UnfittedMakieExt.jl`.

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
       mass_form, mass_block, stiffness, stiffness_form, stiffness_block, source_form, source_load, load_vector,
       prepare, assemble, assemble_matrix, assemble_vector, assemble!, foreach_quadrature_point, nquadpoints, solve!,
       solution, move!, moved, moved_space, activate!, deactivate!, active_cells, support_extension, dilate,
       cell_orders, elevate, elevated, cell_order, nominal_order, ladder, adapt, adapted, is_nested, overlapping_cells,
       level_count, transfer, L2Projection, Rewire, estimate, mark_cells, decide, refine, coarsen, QuadField, RBFP0,
       write_vtk, vtk_series, VTKSeries, write_quadrature_vtm, levelplot, levelplot!, quadratureplot, quadratureplot!,
       l2_error, boundary_integral, value, field_gradient, diagnostics, active_unknowns, cell_indices, cell_box, center,
       value_vec, gradient_tensor, symmetric_gradient

include("geometry.jl")
include("physical.jl")
include("implicit.jl")
include("basis.jl")
include("fcm.jl")
include("mesh.jl")
include("intersections.jl")
include("coverage.jl")
include("ladder.jl")
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

# The public entry points of the package extensions, declared and documented in the core so that `?name` answers
# without the extension. Last, because some of their documented signatures name types the files above define.
include("stubs.jl")

# The one thing loading the package does: register the hint that names the package a methodless extension entry point
# needs (`_extension_hint` in `stubs.jl`). An error hint belongs to the running session, not to the precompiled module,
# so it is registered when the module is loaded.
function __init__()
    Base.Experimental.register_error_hint(_extension_hint, MethodError)
    return nothing
end

end
