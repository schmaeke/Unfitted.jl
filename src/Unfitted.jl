"""
    Unfitted

Julia implementation of unfitted multi-level hp refinement on axis-aligned
Cartesian grids.

The approximation is built as a superposition of overlay contributions, one
per mesh level:

    u_h(x) = Σₖ u_h^(k)(x)

Each overlay contribution `u_h^(k)` is extended by zero outside its overlay
domain `Ω^(k)`. Overlay levels are independent — they are *not* topologically
merged with their parents, do not introduce hanging-node constraints, and
impose homogeneous boundary conditions on their artificial overlay boundary
`Γ_o^(k) = ∂Ω^(k) \\ ∂Ω`. Integration on the resulting non-matching meshes
uses admissible C∞ boxes built from the union of per-axis element-boundary
coordinates of all participating levels.

The package supports finite-cell-method (FCM) immersed integration through a
[`PhysicalDomain`](@ref): the geometry of `Ω` is a CSG combination of smooth
level-set leaves (`leaf`, `intersect`, `union`, `setdiff`, `complement`; a
single callable `φ` with `Ω = {φ ≤ 0}` is the degenerate case). Cut cells use a
non-negative moment-fit quadrature rule whose moments come from Saye's
dimension-reduction implicit quadrature on the level set (R. I. Saye, SIAM J.
Sci. Comput. **37** (2015) A993; J. Comput. Phys. **448** (2022) 110720),
exact and octree-depth-independent on smooth cut cells. See the top of
`src/fcm.jl` and `src/implicit.jl`.

Imported boundary-mesh geometry is supported as a signed-distance level set
through a package extension: with `FileIO` and `MeshIO` loaded (which pull in
`GeometryBasics`, the extension's third trigger),  `mesh_levelset(mesh)` turns a
closed `BoundaryMesh` — a 2D segment loop or a 3D triangle surface — into a
[`LevelSet`](@ref) leaf, and `stl_levelset("part.stl")` reads an STL into one.
Both compose with the CSG combinators. See `ext/UnfittedMeshIOExt.jl`.

Reference for the unfitted multi-level hp method:

> J. N. Schmäke and M. Ruess, *Unfitted Multi-Level hp Refinement for
> Localized and Moving Solution Features*, arXiv:2604.25797.

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

`CONTRIBUTING.md` documents the full set of mathematical contracts,
public-API design rules, code style, and developer workflow.
"""
module Unfitted

using LinearAlgebra
using SparseArrays
using StaticArrays
using FastGaussQuadrature
using ForwardDiff
using NearestNeighbors
using NonNegLeastSquares
using WriteVTK

export AxisBox, GeometryTolerance, PhysicalDomain, physical_domain, classify_cell, LevelSet, leaf,
       complement, levelset_value, stl_levelset, mesh_levelset, CartesianMesh, Space,
       IntegratedLegendre, bspline, Field, BlockForm, LoadForm, TrialChannels, TestChannels,
       WeakForm, Problem, Model, Solution, box, mesh, space, overlay, field, block, loadform,
       boundary, dirichlet, update_dirichlet!, neumann, couple, interface, Interface, InterfaceForm,
       onside, jump_sign, BoundaryMesh, segment_mesh, polyline_mesh, triangle_mesh, points_mesh,
       poisson, mass, mass_form, mass_block, stiffness, stiffness_form, stiffness_block, load,
       source_form, source_load, load_vector, prepare, assemble_matrix, assemble_vector, assemble!,
       foreach_quadrature_point, foreach_interface_quadrature_point, interface_quadrature_count,
       nquadpoints, solve!, solution, move!, moved, moved_space, activate!, deactivate!,
       active_cells, cell_orders, elevate, elevated, cell_order, ladder, adapt, adapted,
       depth_masks, is_nested, overlapping_cells, transfer, L2Projection, Rewire, QuadField, RBFP0,
       write_vtk, write_quadrature_vtm, l2_error, boundary_integral, value, field_gradient,
       diagnostics, active_unknowns, cell_indices, cell_box, center, value_vec, gradient_tensor,
       symmetric_gradient

# ── Extension stubs ───────────────────────────────────────────────────────────
#
# These names are populated by `ext/UnfittedTensorsExt.jl` when Tensors.jl is
# loaded. They are declared here, not in the extension module, so that any
# downstream code can write `using Unfitted: symmetric_gradient, value_vec,
# gradient_tensor` unconditionally; loading Tensors.jl alongside Unfitted
# installs the concrete methods on these stubs. Calling any of them without
# a Tensors-compatible signature raises the usual `MethodError`, which is
# the intended behaviour: the names are only meaningful once tensor algebra
# is in scope.
#
# `symmetric_gradient` is declared here, not extended from Tensors.jl,
# because Tensors.jl does not export a function under that name; the
# closest cousins it ships are the AD entry points `gradient`,
# `hessian`, and `divergence`.
function symmetric_gradient end
function value_vec end
function gradient_tensor end

# `bspline(; continuity_order=0)` is the public factory for the
# open-knot tensor B-spline basis family. It returns a deferred
# specification that the `instantiate_basis` hook turns into a concrete
# `BSplineFamily` once the level's mesh is known (the per-axis degree
# comes from the `order` keyword on `space` / `overlay`, matching the
# integrated Legendre convention). The struct and all methods live in
# `ext/UnfittedBasicBSplineExt.jl` and are installed when BasicBSpline.jl
# is loaded alongside Unfitted. The stub here lets downstream code write
# `using Unfitted: bspline` unconditionally; calling `bspline(...)`
# without BasicBSpline loaded raises the usual `MethodError`, which is
# the intended behaviour.
function bspline end

# `mesh_levelset(mesh::BoundaryMesh)` and `stl_levelset(path)` build a signed-
# distance [`LevelSet`](@ref) leaf from a closed boundary mesh — a 2D segment
# loop (`BoundaryMesh{2,T,1}`) or 3D triangle surface (`BoundaryMesh{3,T,2}`);
# `stl_levelset` reads an STL file into the latter. Both return a `leaf` whose
# level set is negative inside the solid, suitable for `physical_domain` and
# CSG composition, and carrying the exact Lipschitz constant of the mesh's
# inside/outside test (`1.0` by default, `Inf` under `orientation = :winding`,
# whose sign can jump). The signed-distance kernel and the STL
# loader live in `ext/UnfittedMeshIOExt.jl` and are installed when FileIO,
# MeshIO, and GeometryBasics are loaded alongside Unfitted. The stubs here let
# downstream code write `using Unfitted: mesh_levelset` unconditionally; calling
# them without those packages loaded raises the usual `MethodError`.
function stl_levelset end
function mesh_levelset end

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
include("data.jl")
include("postprocessing.jl")
include("api.jl")

end
