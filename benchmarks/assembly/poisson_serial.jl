using BenchmarkTools
using Unfitted

group = SUITE["assembly"]["poisson_serial"] = BenchmarkGroup()

# Build a Poisson model with optional single overlay. Model construction stays
# outside the timing — only `assemble!` is measured.
function _build_poisson_model(domain, cells, order; overlay_box=nothing, overlay_cells=nothing,
                              overlay_order=nothing)
    V = space(domain; cells, order)
    if overlay_box !== nothing
        V = overlay(V, overlay_box; cells=overlay_cells, order=overlay_order)
    end
    source(x) = sum(sin, x)
    problem = poisson(V; source, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    return prepare(problem)
end

let model = _build_poisson_model(box((0.0, 0.0), (1.0, 1.0)), (16, 16), 2)
    group["D=2 16x16 p=2"] = @benchmarkable assemble!($model; threaded=false)
end

let model = _build_poisson_model(box((0.0, 0.0), (1.0, 1.0)), (24, 24), 3;
                                 overlay_box=box((0.32, 0.28), (0.72, 0.68)), overlay_cells=(9, 9),
                                 overlay_order=4)
    group["D=2 24x24 p=3 +overlay"] = @benchmarkable assemble!($model; threaded=false)
end

let model = _build_poisson_model(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (8, 8, 8), 2)
    group["D=3 8^3 p=2"] = @benchmarkable assemble!($model; threaded=false)
end
