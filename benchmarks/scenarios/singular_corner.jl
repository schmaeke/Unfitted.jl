using BenchmarkTools
using Unfitted

# Smaller version of examples/singular_square_2d.jl (4 refinements instead of 8
# to keep bench wall time reasonable). Nested overlays stress `_merged_boxes`
# and the per-region triplet emit.
group = SUITE["scenarios"]["singular_corner"] = BenchmarkGroup()

let
    omega = box((0.0, 0.0), (1.0, 1.0))
    no_refinements = 4
    overlay_ratio = 0.2
    base_cells = (1, 1)
    overlay_cells = (1, 1)
    max_order = no_refinements + 1

    radius(x) = sqrt(x[1]^2 + x[2]^2)
    exact(x) = sqrt(radius(x))
    source(x) = -0.25 * radius(x)^(-1.5)

    V = space(omega; cells=base_cells, order=max_order)
    for i in 1:no_refinements
        side = overlay_ratio^i
        V = overlay(V, box((0.0, 0.0), (side, side)); cells=overlay_cells,
                    order=max(1, max_order - i))
    end

    problem = poisson(V; source,
                      dirichlet=[dirichlet(exact; on=boundary(axis=1, side=:upper)),
                                 dirichlet(exact; on=boundary(axis=2, side=:upper))])
    model = prepare(problem)
    assemble!(model)

    group["prepare"] = @benchmarkable prepare($problem)
    group["assemble"] = @benchmarkable assemble!($model)
    group["solve"] = @benchmarkable solve!($model)
end
