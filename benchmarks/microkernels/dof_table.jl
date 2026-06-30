# Microbenchmark gating the dual local-dof-table representation in assembly.
#
# Assembly carries two per-parent dof-table representations
# (`src/assembly.jl`): a lightweight `Matrix{Int}` fast path for layouts
# without linear constraints (the integrated Legendre family and C⁰ B-splines)
# and a `LocalDofExpansion` for layouts that do (C^m B-splines). Collapsing the
# two onto the single `LocalDofExpansion` path removes ~6 paired kernels, but
# would route the common integrated-Legendre assembly through a per-emission
# `(raw, weight)` redirect instead of a direct index. This benchmark measures
# `assemble!` on a representative integrated-Legendre problem (which uses the
# `Matrix{Int}` path) so the collapse decision is data-driven: flip the
# `_local_active_dof_table!` branch in `src/assembly.jl` to force the
# `LocalDofExpansion` path and re-run to read the regression.
using Unfitted

include(joinpath(@__DIR__, "..", "common.jl"))

# Representative integrated-Legendre assembly: 2D Poisson, base + centered
# overlay, order 3 / 4 — the same shape as `benchmarks/assembly.jl`.
omega = box((0.0, 0.0), (1.0, 1.0))
V = space(omega; cells=(24, 24), order=3)
V = overlay(V, box((0.32, 0.28), (0.72, 0.68)); cells=(9, 9), order=4)
problem = poisson(V; source=x -> sin(pi * x[1]) * sin(pi * x[2]),
                  dirichlet=[dirichlet(0.0; on=boundary(:all))])
model = prepare(problem)

# The pattern is cached on the first call, so `best_timed` measures the repeated
# numeric scatter — the Newton/transient hot path the dof-table sits on.
stats = best_timed(; samples=9) do
    assemble!(model)
end

println("dof-table assembly microbenchmark")
println("  Julia threads: ", Threads.nthreads())
println("  active unknowns: ", Unfitted.active_unknowns(model.dofs))
println("  integration regions: ", diagnostics(model).integration_regions)
print_timed("assemble!", stats)
