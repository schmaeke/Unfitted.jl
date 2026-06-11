using Unfitted
using SparseArrays

include(joinpath(@__DIR__, "common.jl"))

omega = box((0.0, 0.0), (1.0, 1.0))
V = space(omega; cells=(24, 24), order=3)
V = overlay(V, box((0.32, 0.28), (0.72, 0.68)); cells=(9, 9), order=4)

source(x) = sin(pi * x[1]) * sin(pi * x[2])
problem = poisson(V; source, dirichlet=[dirichlet(0.0; on=boundary(:all))])
model = prepare(problem)

assembly_stats = best_timed(; samples=3) do
    assemble!(model)
end
solve_stats = best_timed(; samples=3) do
    solve!(model)
end
solution = solve_stats.value

diag = diagnostics(model, solution)
nnz_matrix = nnz(model.matrix)

println("Assembly benchmark")
println("  Julia threads: ", Threads.nthreads())
println("  dimension: ", diag.dimension)
println("  levels: ", length(diag.levels))
println("  active unknowns: ", diag.active_unknowns)
println("  integration regions: ", diag.integration_regions)
println("  matrix nonzeros: ", nnz_matrix)
print_timed("assembly", assembly_stats)
print_timed("direct solve", solve_stats)
println("  residual norm: ", diag.residual_norm)
