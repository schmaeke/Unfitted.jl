#=
Microbenchmark guarding the Dirichlet-projection refactor.

The Dirichlet projection in `_project_dirichlet_values!` runs on every
`prepare(problem)` that carries nonzero Dirichlet data. PR1 of the
boundary-integration redesign moved its facet-region builder behind a
precomputed-Q-points API (`FacetRegion` now carries the physical Gauss
rule baked-in instead of recomputing per quadrature point). This
benchmark exists so a future change to the projection path can be
checked against the same setup we used to validate PR1's refactor.

The scenario is the smooth Laplace problem from
`examples/laplace_unit_square_smooth.jl` at a moderately fine
resolution: nonzero Dirichlet data on one edge (callback `sin(πx)`),
which forces the projection to assemble a real mass-matrix solve over
the constrained-dof subspace. Zero-data edges skip the projection
entirely and are therefore irrelevant to this measurement.
=#

using Unfitted

include(joinpath(@__DIR__, "common.jl"))

omega = box((0.0, 0.0), (1.0, 1.0))
V = space(omega; cells=(24, 24), order=4)

exact(x) = sin(pi * x[1]) * sinh(pi * x[2]) / sinh(pi)

problem = poisson(V; source=0.0,
                  dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                             dirichlet(0.0; on=boundary(axis=1, side=:upper)),
                             dirichlet(0.0; on=boundary(axis=2, side=:lower)),
                             dirichlet(exact; on=boundary(axis=2, side=:upper))])

# Run `prepare(problem)` end-to-end; the Dirichlet projection is the
# dominant cost on a single-level smooth Laplace problem at this
# resolution and order.
stats = best_timed(; samples=5) do
    prepare(problem)
end

model = stats.value
report = diagnostics(model)

println("Dirichlet projection benchmark")
println("  dimension: ", length(omega.lower))
println("  cells: ", model.problem.space.levels[1].mesh.cells)
println("  order: ", model.problem.space.levels[1].order)
println("  active unknowns: ", report.active_unknowns)
println("  facet regions: ", report.facet_region_count)
println("  facet quadrature points: ", nquadpoints(model; kind=:facet))
print_timed("prepare(problem) with nonzero Dirichlet", stats)
