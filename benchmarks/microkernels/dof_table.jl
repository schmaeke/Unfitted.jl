# Microbenchmark of the per-region dof table in assembly.
#
# Assembly numbers each integration region's dofs once (`_frame!`, `_slots!` in
# `src/assembly.jl`): every (parent, component, basis function) slot holds the
# local index of an active dof, `0` for a constrained one, or a pivot slot for
# a linear-constraint pivot, whose row and column are condensed onto the
# pivot's expansion once per region (`_condense!`, `K ← PᵀKP`,
# `b ← Pᵀb − lift`). One table serves every basis family, so the two cases
# below run the same code and differ only in whether a pivot ever appears:
#
#   * integrated Legendre — 2D Poisson on a base mesh with a centered overlay,
#     order 3 / 4, the shape of `benchmarks/assembly.jl`. No pivots: the
#     regression case for the common path, which must not pay for the pivot
#     machinery it never uses;
#   * clamped C¹ B-splines on a masked level — a cubic `continuity = 1` level
#     with an L-shaped mask, whose inactive faces emit multi-raw trace
#     constraints, so a share of the regions carries pivots and runs
#     `_condense!`.
#
# Run from the repository root:
#     julia --project=benchmarks benchmarks/microkernels/dof_table.jl
using Unfitted
using BasicBSpline   # loads the B-spline extension behind `bspline`

include(joinpath(@__DIR__, "..", "common.jl"))

legendre = let omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(24, 24), order=3)
    V = overlay(V, box((0.32, 0.28), (0.72, 0.68)); cells=(9, 9), order=4)
    prepare(poisson(V; source=x -> sin(pi * x[1]) * sin(pi * x[2]),
                    dirichlet=[dirichlet(0.0; on=boundary(:all))]))
end

clamped = let mask = trues(24, 24)
    mask[13:24, 13:24] .= false   # L-shaped active region
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=24, order=3, basis=bspline(; continuity=1),
              active=mask)
    prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
end

println("dof-table assembly microbenchmark")
println("  Julia threads: ", Threads.nthreads())
for (label, model) in (("integrated Legendre", legendre), ("clamped C¹ B-spline", clamped))
    # Assembly caches its symbolic data on the first call, so `best_timed`
    # measures the repeated numeric pass — the Newton / transient hot path the
    # dof table sits on.
    stats = best_timed(; samples=9) do
        assemble!(model)
    end
    println("  ", label, ": ", Unfitted.active_unknowns(model.dofs), " active unknowns, ",
            diagnostics(model).integration_regions, " integration regions, linear constraints: ",
            Unfitted.has_linear_constraints(model.dofs))
    print_timed("assemble! ($label)", stats)
end
