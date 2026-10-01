#=
Heat Conduction In An L-Bracket Described By A Surface Mesh
===========================================================

The CAD route into the package: geometry that arrives as a triangulated
boundary surface rather than as a formula.

`mesh_levelset` turns a closed `BoundaryMesh` — a 2-D segment loop or a
3-D triangle surface — into a signed-distance level set, negative inside
the solid. From there it is an ordinary [`leaf`](@ref): it goes into
`physical_domain`, it composes with the CSG combinators
(`intersect` / `union` / `setdiff` / `complement`), and the finite-cell
machinery classifies and integrates cut cells exactly as it does for an
analytic level set. Nothing downstream of the geometry knows the
difference.

The part that reads the mesh lives in a package extension, so it loads
only when `FileIO` and `MeshIO` are present alongside `Unfitted` — hence
the two `using` lines below and the two extra entries in this example's
`Project.toml`.

What an imported surface can and cannot do
------------------------------------------

There are two distinct jobs a boundary surface can do here, and only one
of them works from a file today:

  * **As geometry.** `mesh_levelset(mesh)` — or `stl_levelset("part.stl")`,
    which reads an ASCII or binary STL and calls it — returns a
    `LevelSet`. That is everything `physical_domain` needs: cell
    classification, cut-cell moment-fit quadrature, volume integration
    over `Ω`. This route works straight from disk.

  * **As an integration surface.** Passing a `BoundaryMesh` as `on = mesh`
    to a Neumann load or to a weak (Nitsche) Dirichlet form integrates
    over the immersed boundary itself. That needs the `BoundaryMesh`
    *object*, and there is no public STL → `BoundaryMesh` reader:
    `stl_levelset` returns a `LevelSet` and keeps the triangles to itself.

So an STL currently gives you the shape but not a handle on its surface.
If you need both, build the mesh yourself with `triangle_mesh` (3-D) or
`polyline_mesh` (2-D) — as this example does — and pass that one object to
`mesh_levelset` *and* to `on =`. This example only exercises the geometry
job: its lateral surface is insulated, which the finite-cell method gives
for free, and its end caps deliberately sit outside the computational box,
so there would be nothing there to integrate over anyway.

Why an L-prism and not a cube
-----------------------------

The inside/outside sign of a mesh signed-distance function is the part
that is easy to get wrong, and a convex solid never exposes it: for a
cube, every plausible sign rule agrees. This example therefore uses a
**non-convex** solid — an L-shaped cross-section extruded along z — whose
inner corner contributes a *reflex* edge running up the notch.

For a query point sitting in the notch, the closest feature of the
surface is that reflex edge, or one of the two vertices at its ends. A
naive rule that averages the incident face normals with equal weight can
report the wrong side there. `mesh_levelset` uses the angle-weighted
pseudonormal instead — each incident triangle is weighted by the angle it
subtends at the closest point — which is exact at reflex features:

> J. A. Bærentzen, H. Aanæs, *Signed distance computation using the angle
> weighted pseudonormal*, IEEE Trans. Vis. Comput. Graph. **11** (2005)
> 243–253. [doi:10.1109/TVCG.2005.49](https://doi.org/10.1109/TVCG.2005.49)

That rule is exact only on a **watertight, consistently outward-oriented**
mesh, so `mesh_levelset` checks both preconditions before it answers: every
triangle edge must be traversed once in each direction, and the enclosed
volume must come out positive. A surface with a hole, a flipped facet, or a
globally reversed winding is rejected with an error rather than silently
answered with an inverted sign — which is the failure mode you would
otherwise spend an afternoon on. (For meshes that genuinely cannot be
repaired, `orientation = :winding` swaps in the generalized winding number,
which tolerates gaps at a higher cost per query.)

The problem
-----------

Steady heat conduction in the bracket with a distributed source, both
end faces held at zero temperature and the lateral surface insulated:

    −Δu = f    in Ω,    u = 0 on the two z faces,    ∇u·n = 0 on the sides.

The insulated lateral surface is free: a cut cell integrates only the
material side, so a zero-flux immersed boundary is what you get by not
writing anything down. The two end faces are grid-aligned box faces —
the prism deliberately runs past the box in z, so the box truncates it
there — and carry ordinary strong Dirichlet conditions.

With `f(x) = (π/c)² sin(π x₃ / c)` and `c` the box height, the exact
solution is `u(x) = sin(π x₃ / c)`: it is independent of x₁ and x₂, so it
satisfies the lateral zero-flux condition for *any* cross-section, and it
vanishes on both end faces.

What the printed metrics mean
-----------------------------

Two numbers, and they check different things.

  * **relative L2 error** — against `sin(π x₃ / c)`. This says the solve
    is right. Its size is governed *purely* by the z resolution: the
    bilinear form on a prism is separable and the constant function in
    (x₁, x₂) is in the space, so the discrete solution is exactly the 1-D
    Galerkin solution in z, and the number below is the error of three
    p = 2 elements across half a period of a sine — about `6·10⁻³`.
    Raising `order` or the z cell count moves it; nothing about the
    geometry does. In particular it says almost nothing about the
    geometry: the exact solution does not vary in x₁ or x₂, so it would
    look just as good if the level set had the sign backwards and the
    code had solved on the complement of the bracket.

  * **volume error** — the finite-cell quadrature summed over `Ω`,
    compared against the analytic volume of the L-prism. *This* is the
    geometry metric. It exercises the classifier, the cut-cell moment fit,
    and above all the pseudonormal sign at the reflex edge; a flipped
    sign or a mis-integrated notch shows up here immediately. It should
    sit at roughly `10⁻¹⁵` relative — the cut geometry here is planar, and
    the implicit-quadrature kernel integrates planar cuts exactly.

Supporting evidence in the report: `fit failure count` must be `0`,
`moment-fit residual (max)` must be near machine precision, and `symmetry
residual` must be `0` (the operator is SPD, so the direct solve is a
Cholesky-grade problem).
=#

using Unfitted
using FileIO                # together these two load the
using MeshIO                # UnfittedMeshIOExt extension
using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "..", "..", "reporting.jl"))

# ── The bracket ─────────────────────────────────────────────────────────────

const arm = 0.4             # width of each arm of the L
const span = 1.0            # outer extent of the L in x₁ and x₂
const z_lo = -0.2           # the prism runs past the computational box in z,
const z_hi = 0.8            # so the box truncates it and owns the end faces

# The L-shaped cross-section, listed counter-clockwise so the extruded side
# walls come out outward-facing under the right-hand rule. Vertex 4 is the
# reflex corner — interior angle 270° — and the whole point of the example.
const profile = [SVector(0.0, 0.0), SVector(span, 0.0), SVector(span, arm), SVector(arm, arm),
                 SVector(arm, span), SVector(0.0, span)]
const reflex = 4

# Analytic area of the L-shaped cross-section; the volume reference below is
# this times the modelled height.
const cross_section = span * arm + arm * (span - arm)

# Extrude a counter-clockwise, simply-connected polygon into a closed triangle
# surface. Three details matter, and all three are watertightness details:
#
#   * The end caps are triangulated by *fanning from the reflex vertex*. A fan
#     from the wrong vertex cuts across the notch and leaves triangles outside
#     the solid — vertex 3 here is such a vertex — whereas every diagonal drawn
#     from this L's reflex corner stays inside it. That is a property of having
#     exactly one reflex corner; a polygon with several needs a real
#     ear-clipping or constrained-Delaunay triangulation.
#
#   * Caps and walls share the *same* vertex indices, so no triangle edge is
#     left with a dangling neighbour. Splitting the L into two rectangles
#     instead would put a T-junction on the seam between them, and the
#     watertightness check would — correctly — reject the result.
#
#   * The two caps are wound oppositely (the bottom fan reverses the top one)
#     so both normals point away from the solid.
function extrude(profile, reflex, z_lo, z_hi)
    n = length(profile)
    vertices = [SVector(p[1], p[2], z) for z in (z_lo, z_hi) for p in profile]
    bottom(i) = i                     # vertex indices of the z_lo copy
    top(i) = n + i                    # … and of the z_hi copy
    faces = NTuple{3,Int}[]

    for k in 1:(n-2)                  # caps: n−2 triangles each
        i, j = mod1(reflex + k, n), mod1(reflex + k + 1, n)
        push!(faces, (top(reflex), top(i), top(j)))       # outward = +z
        push!(faces, (bottom(reflex), bottom(j), bottom(i)))   # outward = −z
    end

    for i in 1:n                      # side walls: one quad per polygon edge
        j = mod1(i + 1, n)
        push!(faces, (bottom(i), bottom(j), top(j)))
        push!(faces, (bottom(i), top(j), top(i)))
    end

    return triangle_mesh(vertices, faces)
end

surface = extrude(profile, reflex, z_lo, z_hi)

# The signed-distance leaf. `lipschitz = 1.0` is the default here and is
# exact: a true signed distance has ‖∇φ‖ = 1, which lets the cell classifier
# use its cheap uniform-sign certificate instead of falling back to sampling.
bracket = mesh_levelset(surface)

# ── Discretisation ──────────────────────────────────────────────────────────
#
# The box is deliberately larger than the bracket in x₁ and x₂ (so the lateral
# surface is genuinely immersed, with no grid line coinciding with it) and
# exactly spans the modelled height in x₃ (so the two end faces are grid
# aligned and can carry strong Dirichlet conditions).

const margin = 0.15
const height = 0.6
const cells = (5, 5, 3)
const order = 2

omega = box((-margin, -margin, 0.0), (span + margin, span + margin, height))
const cell_size = minimum((span + 2margin, span + 2margin, height) ./ cells)

# `subcell_length_scale` is the geometry-robustness knob: it sets the target
# box size for the binary subdivision that classifies cells and that resolves
# cut cells the implicit kernel cannot treat as a graph — here, the cells the
# reflex edge passes through. It is not an accuracy grid; smooth planar cuts
# are integrated exactly and never subdivided.
solid = physical_domain(bracket; subcell_length_scale=cell_size / 8, max_depth=4)

V = space(omega; cells=cells, order=order, physical=solid)

# ── Problem ─────────────────────────────────────────────────────────────────

exact(x) = sin(π * x[3] / height)
source(x) = (π / height)^2 * sin(π * x[3] / height)

problem = poisson(V; source=source,
                  dirichlet=[dirichlet(0.0; on=boundary(axis=3, side=:lower)),
                             dirichlet(0.0; on=boundary(axis=3, side=:upper))])
model = prepare(problem)
sol = solve!(model)

# ── Geometry check: does the quadrature reproduce the analytic volume? ──────
#
# `foreach_quadrature_point` walks exactly the points assembly integrated on,
# with physical-frame weights, so summing the weights measures |Ω| as the
# solver saw it — cut cells, moment fit, reflex notch and all.

quadrature_volume = Ref(0.0)
foreach_quadrature_point(model) do q
    quadrature_volume[] += q.weight
    return nothing
end
exact_volume = cross_section * height
volume_error = abs(quadrature_volume[] - exact_volume) / exact_volume

# ── Output ──────────────────────────────────────────────────────────────────

# `subdivisions` sizes the sub-cells that resolve the *solution*, so the
# order-2 default of two per axis is right here. The boundary is a separate
# question: ParaView reconstructs it from the `level_set` array by interpolating
# linearly inside each cell, so a sharp picture of the bracket needs small cells
# where ∂Ω runs — and only there. `cut_depth` bisects exactly those, three times
# over, giving sixteen sub-cells per axis across the surface for a fraction of
# what the same resolution would cost applied to every region. The notch's
# reflex edge is a crease of the mesh level set, where the collapsed φ is only
# C⁰ and the clip converges at first order, so the edge stays comparatively
# faceted however deep this goes.
out = joinpath(@__DIR__, "output", "imported_geometry_3d")
write_vtk(out, sol, model;
          point_data=(u=(uh, c, x, xi) -> uh(c, xi), exact=(uh, c, x, xi) -> exact(x)), cut_depth=3)
write_quadrature_vtm(out * "_quadrature", model)

print_run_report("L-bracket from a triangle surface — imported-geometry level set",
                 diagnostics(model, sol; exact); output=out,
                 parameters=(:surface_triangles => length(surface.cells), :cells => cells,
                             :order => order,
                             :cross_section => "L-profile, arm $(arm) of span $(span)",
                             :subcell_length_scale => solid.subcell_length_scale,
                             :max_depth => solid.max_depth))
println("  quadrature volume: ", quadrature_volume[])
println("  analytic volume: ", exact_volume)
println("  volume error: ", volume_error)
