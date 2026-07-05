using Unfitted

include(joinpath(@__DIR__, "common.jl"))

omega = box((0.0, 0.0), (1.0, 1.0))
source_space = space(omega; cells=(18, 18), order=3)
source_space = overlay(source_space, box((0.18, 0.22), (0.62, 0.66)); cells=(8, 8), order=4)
target_space = Unfitted.moved_space(source_space; level=2, to=box((0.31, 0.26), (0.75, 0.70)))

source_model = prepare(poisson(source_space; source=0.0))
target_model = prepare(poisson(target_space; source=0.0))
source_coefficients = [sin(0.01 * i) for i in 1:Unfitted.active_unknowns(source_model.dofs)]
source_solution = Solution(source_coefficients, source_model.version,
                           Unfitted.SolverDiagnostics(:synthetic, 0.0, true))

stats = best_timed(; samples=3) do
    transfer(source_solution, source_model, target_model)
end
target_solution = stats.value
report = diagnostics(target_model, target_solution)

println("Transfer benchmark")
println("  dimension: ", report.dimension)
println("  source levels: ", length(source_model.problem.space.levels))
println("  target levels: ", length(target_model.problem.space.levels))
println("  target active unknowns: ", report.active_unknowns)
println("  target integration regions: ", report.integration_regions)
print_timed("transfer", stats)
println("  projection residual norm: ", report.residual_norm)
