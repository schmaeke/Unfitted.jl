#=
Small-Overlap Conditioning Sweep
================================

A reproduction of the small-overlap conditioning study from §5.3 of
the UMLHP preprint (arXiv:2604.25797): construct a base discretization
on the square `Ω = (0, 3)²` and place a single overlay whose lower-left
corner sits at `(δ, δ)`. As `δ → 0` the overlay nearly aligns with the
base element boundaries, producing very thin integration boxes that
expose the conditioning hazards of unfitted methods.

The model problem is

    −Δu = 1               in Ω = (0, 3)²,
      u = 0               on x = 3 and y = 3,

with the remaining physical boundaries left natural. The overlay
covers

    Ωᵒ(δ) = (δ, δ) × (2 + δ, 2 + δ),

so it is always a 2 × 2 square aligned with the base 3 × 3 grid up to
the `δ` shift.

For each combination of polynomial order `p` and `δ = 2⁻ᵏ` the script
reports:

  * `active_unknowns` — the size of the assembled linear system,
  * `integration_regions` — the count of admissible boxes,
  * `small_overlap_count` — boxes below `small_volume_threshold`,
  * `min_volume` and `first_small_volume` — the worst-overlap volumes,
  * `condition_estimate` — `cond(A)` of the dense matrix where
    `active_unknowns ≤ 256` (see `_condition_estimate` in
    `src/model.jl`); `NaN` for larger systems,
  * `residual_norm` — the post-solve residual `‖A x − b‖₂`.

Sweep results print as a single comma-separated row per `(p, k)` pair
so the output can be redirected straight to a CSV file for plotting:

    julia --project=. examples/conditioning_small_overlap.jl > sweep.csv
=#

using Unfitted

include(joinpath(@__DIR__, "..", "reporting.jl"))

# Physical domain and small-volume tolerance. Set the threshold tight
# enough that the conditioning-relevant tiny regions surface as
# diagnostics, but loose enough that the well-formed regions stay
# below the warning floor.
omega = box((0.0, 0.0), (3.0, 3.0))
tolerance = GeometryTolerance(Float64; small_volume=1.0e-3)

# Sweep header. The comma-separated columns are emitted verbatim by the
# inner loop so the output can be redirected to a CSV file.
println("Small-overlap conditioning sweep")
print_parameter_block((:dimension => 2, :domain => omega,
                       :small_volume_threshold => tolerance.small_volume))
println("  columns: p, k, δ, active_unknowns, regions, small_regions, ",
        "min_volume, first_small_volume, cond_estimate, residual_norm")

# Sweep over polynomial orders `p ∈ 1:4` and shifts `δ = 2⁻ᵏ` for
# `k ∈ 1:8`. Each iteration is an independent prepare/solve so the
# loop can be parallelised trivially (it is not, here, to keep the
# sweep deterministic for regression-comparison purposes).
for p in 1:4
    for k in 1:8
        delta = 2.0^(-k)
        # Build the (base + overlay) space at the current shift, then
        # solve the unit-source Poisson problem.
        V = space(omega; cells=(3, 3), order=p)
        V = overlay(V, box((delta, delta), (2.0 + delta, 2.0 + delta)); cells=(2, 2), order=p)
        problem = poisson(V; source=1.0,
                          dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:upper)),
                                     dirichlet(0.0; on=boundary(axis=2, side=:upper))])

        model = prepare(problem; tolerance)
        assemble!(model)
        solution = solve!(model)
        report = diagnostics(model, solution)
        first_small_volume = isempty(report.small_overlaps) ? NaN :
                             first(report.small_overlaps).volume

        # One CSV row per (p, k). `condition_estimate` is `NaN` for
        # `active_unknowns > 256` — that is intentional, the
        # estimator densifies the matrix and is only cheap up to a
        # few hundred unknowns.
        println("  ", p, ", ", k, ", ", delta, ", ", report.active_unknowns, ", ",
                report.integration_regions, ", ", report.small_overlap_count, ", ",
                report.min_integration_volume, ", ", first_small_volume, ", ",
                report.condition_estimate, ", ", report.residual_norm)
    end
end
