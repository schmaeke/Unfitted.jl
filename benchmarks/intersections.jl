using Unfitted

include(joinpath(@__DIR__, "common.jl"))

omega = box((0.0, 0.0), (1.0, 1.0))
V = space(omega; cells=(32, 32), order=2)
V = overlay(V, box((0.2, 0.2), (0.8, 0.8)); cells=(17, 17), order=3)
V = overlay(V, box((0.47, 0.35), (0.86, 0.74)); cells=(11, 9), order=4)

stats = best_timed(; samples=5) do
    Unfitted.integration_plan(V)
end
plan = stats.value

println("Intersection benchmark")
println("  dimension: ", length(omega.lower))
println("  levels: ", length(V.levels))
println("  integration regions: ", length(plan.regions))
println("  small-overlap regions: ", plan.small_overlap_count)
println("  min integration volume: ", plan.min_volume)
print_timed("integration_plan", stats)
