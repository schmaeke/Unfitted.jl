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
level-set [`PhysicalDomain`](@ref): the geometry of `Ω` is described as
`Ω = { x : φ(x) ≤ 0 }`, and cut cells use a non-negative moment-fit
quadrature rule. The moment-fit pipeline is ported from QuESo (M. Meßmer
et al., *Efficient CAD-integrated isogeometric analysis of trimmed solids*,
Comput. Methods Appl. Mech. Engrg. **400** (2022) 115584,
doi:10.1016/j.cma.2022.115584). The deliberate deviations from upstream are
documented at the top of `src/fcm.jl`.

Reference for the unfitted multi-level hp method:

> J. N. Schmäke and M. Ruess, *Unfitted multi-level hp refinement on
> Cartesian grids*, arXiv:2604.25797.

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
using NearestNeighbors
using NonNegLeastSquares
using WriteVTK

export AxisBox, GeometryTolerance, PhysicalDomain, physical_domain, classify_cell, CartesianMesh,
       Level, Space, IntegratedLegendre, Field, BlockForm, LoadForm, TrialChannels, TestChannels,
       WeakForm, Problem, Model, Solution, box, mesh, space, overlay, field, block, loadform,
       boundary, dirichlet, update_dirichlet!, neumann, BoundaryMesh, segment_mesh, polyline_mesh,
       triangle_mesh, points_mesh, poisson, mass, mass_form, mass_block, stiffness, stiffness_form,
       stiffness_block, load, source_form, source_load, load_vector, prepare, assemble_matrix,
       assemble_vector, assemble!, foreach_quadrature_point, nquadpoints, solve!, solution, move!,
       moved, moved_space, activate!, deactivate!, active_cells, transfer!, TransferBackend,
       L2Projection, Rewire, QuadField, QuadTransferScheme, RBFP0, transfer, write_vtk,
       write_quadrature_vtm, l2_error, boundary_integral, value, field_gradient, diagnostics,
       active_unknowns, cell_indices, cell_box, center, value_vec, gradient_tensor,
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

include("geometry.jl")
include("physical.jl")
include("basis.jl")
include("fcm.jl")
include("mesh.jl")
include("intersections.jl")
include("dofs.jl")
include("dirichlet.jl")
include("surface.jl")
include("problems.jl")
include("model.jl")
include("assembly.jl")
include("solvers.jl")
include("projection.jl")
include("data.jl")
include("postprocessing.jl")
include("api.jl")

end
