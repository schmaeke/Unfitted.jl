#=
Singular Square With Local Superposition Refinement
===================================================

The 2D corner-singularity benchmark of

    P. Kopp, E. Rank, V. M. Calo, S. Kollmannsberger, "Efficient
    multi-level hp-finite elements in arbitrary dimensions", Comput.
    Methods Appl. Mech. Engrg. 401 (2022) 115575,
    doi:10.1016/j.cma.2022.115575,

solved with the unfitted multi-level hp method of §5.2 of the UMLHP
preprint (arXiv:2604.25797). The full UMLHP study sweeps polynomial
degrees and overlay depths; this script is the compact executable
instance with a single fixed `(no_refinements, max_order)` choice that
exercises the geometric overlay nesting end-to-end.

Let Ω = (0, 1)² and `r = √(x² + y²)`. The exact solution is

    u(x, y) = √r,

a typical algebraic corner singularity. From `Δ√r = (2d − 3) / 4 · r^(−3/2)`
in spatial dimension `d`, for `d = 2`,

    −Δ √r = (3 − 2 · 2) / 4 · r^(−3/2) = −¼ r^(−3/2).

The singularity sits at the physical corner `(0, 0)`. We impose the
exact Dirichlet data on `x = 1` and `y = 1`; on `x = 0` and `y = 0` the
exact normal flux is zero away from the corner, so those sides are
left as natural (do-nothing) boundaries.

The overlay strategy is the standard geometric nesting from the UMLHP
paper: `n` levels of nested boxes anchored at the corner,

    Ωᵢ = (0, overlay_ratioⁱ)²,        i = 1, …, n,

with a polynomial order that *descends* from the base inward
(`max_order − i` at level `i`). The artificial overlay constraints
fire only on overlay boundaries that do not coincide with the physical
domain boundary — they pin the overlay correction to zero on its
artificial faces but leave the physical-boundary faces free.
=#

using Unfitted

include(joinpath(@__DIR__, "..", "reporting.jl"))

# Domain Ω = (0, 1)² and overlay-hierarchy hyperparameters. With
# `no_refinements = 8` and `overlay_ratio = 0.2` the smallest overlay
# is a square of side 0.2⁸ ≈ 2.56e-6 anchored at the corner — small
# enough to resolve the singularity to single precision while keeping
# the total dof count modest.
omega = box((0.0, 0.0), (1.0, 1.0))
no_refinements = 8
overlay_ratio = 0.2
base_cells = (1, 1)
overlay_cells = (1, 1)
max_order = no_refinements + 1

# Closed-form singular solution and the matching source. `radius` is
# pulled out because the source needs `r^(−3/2)` and the L² error
# computation reads `u(r)` separately.
radius(x) = sqrt(x[1]^2 + x[2]^2)
exact(x) = sqrt(radius(x))
source(x) = -0.25 * radius(x)^(-1.5)

# Build the nested-overlay space. Level 1 (base) covers Ω at
# `max_order = n + 1`; each overlay covers a smaller box anchored at the
# corner with a *lower* polynomial order, so the inner overlays add
# their own local correction without ballooning the dof count.
function singular_refinement_space(omega; no_refinements, overlay_ratio, base_cells, overlay_cells,
                                   max_order)
    V = space(omega; cells=base_cells, order=max_order)

    for i in 1:no_refinements
        side_length = overlay_ratio^i
        level_order = max(1, max_order - i)
        V = overlay(V, box((0.0, 0.0), (side_length, side_length)); cells=overlay_cells,
                    order=level_order)
    end

    return V
end

V = singular_refinement_space(omega; no_refinements, overlay_ratio, base_cells, overlay_cells,
                              max_order)

# Poisson problem with Dirichlet data only on the two non-corner edges.
# The corner sides `x = 0` and `y = 0` are left natural — the symmetry
# of the singular solution makes their normal flux zero away from the
# corner.
problem = poisson(V; source,
                  dirichlet=[dirichlet(exact; on=boundary(axis=1, side=:upper)),
                             dirichlet(exact; on=boundary(axis=2, side=:upper))])

model = prepare(problem)
solution = solve!(model)
report = diagnostics(model, solution; exact)

# VTK export at a coarse subdivision: the singular layer near the
# corner dominates the visualisation, so `subdivisions = 4` resolves
# the smallest overlay box visibly without ballooning the cell count.
out = joinpath(@__DIR__, "output", "singular_square_2d")
write_vtk(out, solution, model; subdivisions=4,
          point_data=(uh=(u, c, x, xi) -> u(c, xi), exact=(u, c, x, xi) -> exact(x),
                      error=(u, c, x, xi) -> u(c, xi) - exact(x),
                      grad_uh=(u, c, x, xi) -> field_gradient(solution, model, x)))

print_run_report("2D singular square", report;
                 parameters=(:no_refinements => no_refinements, :overlay_ratio => overlay_ratio,
                             :base_cells => base_cells, :overlay_cells => overlay_cells,
                             :max_order => max_order), output=out)
