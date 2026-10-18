# The Makie extension's tests: what `fieldplot`, `levelplot` and `quadratureplot` make of the records
# `src/postprocessing.jl` samples, asserted on the plot objects Makie builds — positions, faces, colours, edges, the
# axis each plot asks for — with no backend loaded and nothing rendered. The records themselves are pinned in
# `test_postprocessing.jl`, whose `_record_array` the tests below read them with; `runtests.jl` includes that file
# first, and this one does not stand alone.
#
# `import`, not `using`: every Makie name below is qualified, which keeps Makie's many exports out of the test files
# included after this one.
import Makie        # loads UnfittedMakieExt, which provides fieldplot, levelplot and quadratureplot
using StaticArrays: SVector

const _MakieExt = Base.get_extension(Unfitted, :UnfittedMakieExt)
const _GB = Makie.GeometryBasics

# The children of a plot by their role: the one that carries the colours, which a `Colorbar` reads, and the one that
# draws ∂Ω. A field and a level plot draw their colours first and ∂Ω last; a quadrature plot draws ∂Ω first, behind its
# points.
_coloured(plt) = plt isa _MakieExt.QuadraturePlot ? only(c for c in plt.plots if c isa Makie.Scatter) : first(plt.plots)
_outline(plt) = plt isa _MakieExt.QuadraturePlot ? first(plt.plots) : last(plt.plots)

# The exception `f()` throws, or `nothing` when it returns.
function _thrown(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

# The faces a 2D level plot must draw, stated independently of the extension's code: every drawn cell of every drawn
# record, in record order, its four corners at the height `z = k − n − 1` of the `k`-th of `n` drawn records, and its
# value repeated over them.
function _expected_2d(records, values; inactive=false)
    n = length(records)
    positions, colors = Makie.Point3{Float64}[], Float64[]
    for (k, (record, v)) in enumerate(zip(records, values)), c in 1:Unfitted._cell_count(record.grid)
        inactive || _record_array(record.grid, "active")[c] == 1 || continue
        append!(positions, [Makie.Point3{Float64}(x[1], x[2], k - n - 1) for x in record.grid.points[(4c-3):(4c)]])
        append!(colors, fill(v[c], 4))
    end
    return positions, colors
end

# The edges of a record's drawn cells, each once, as segments from the lower to the upper end at the height `z`: what a
# 2D level plot draws of a level over which no later drawn level has a cell.
function _drawn_cell_edges(record, z; inactive=false)
    segments = Set{NTuple{2,Makie.Point3{Float64}}}()
    for c in 1:Unfitted._cell_count(record.grid)
        inactive || _record_array(record.grid, "active")[c] == 1 || continue
        ll, lr, ur, ul = record.grid.points[(4c-3):(4c)]
        for (a, b) in ((ll, lr), (ul, ur), (ll, ul), (lr, ur))
            push!(segments, (Makie.Point3{Float64}(a..., z), Makie.Point3{Float64}(b..., z)))
        end
    end
    return segments
end

# A level plot's edges, the input of its edge child, as segments: consecutive pairs of its points.
_segments(points) = [(points[2i-1], points[2i]) for i in 1:(length(points)÷2)]

# The values a level plot colours a record's cells with, by one of its integer arrays.
_by_array(records, name) = [Float64.(_record_array(record.grid, name)) for record in records]

# 2D: a 2×2 order-2 base under a 2×2 order-3 overlay of its lower-left quarter whose upper-right cell is switched off,
# with homogeneous Dirichlet data on the whole boundary, so that the per-cell dof counts differ from cell to cell.
_makie_mask = trues(2, 2)
_makie_mask[2, 2] = false
_makie_2d = prepare(poisson(overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=2, order=2), box((0.0, 0.0), (0.5, 0.5));
                                    cells=2, order=3, active=_makie_mask); source=1.0,
                            dirichlet=[dirichlet(0.0; on=boundary(:all))]))
# 1D: four order-1 cells under a two-cell order-2 overlay of the left half.
_makie_1d = prepare(mass(overlay(space(box((0.0,), (1.0,)); cells=4, order=1), box((0.0,), (0.5,)); cells=2, order=2)))

# The solution of `_makie_2d`, and a two-component field on a square of order-2 cells with its solution.
_makie_2d_solution = solve!(_makie_2d)
_makie_vector = prepare(poisson(field(:u, space(box((0.0, 0.0), (1.0, 1.0)); cells=2, order=2); components=2);
                                source=SVector(1.0, -2.0), dirichlet=[dirichlet(SVector(0.0, 0.0); on=boundary(:all))]))
_makie_vector_solution = solve!(_makie_vector)

@testset "the Makie extension provides the plot entry points" begin
    # The entry points are methodless in the core and recipes of the extension. Once the extension is loaded, an
    # argument that is not a model is refused by Makie's own conversion, and the hint that asks for a backend stays
    # silent; the behaviour without the extension is pinned in a fresh process by test_api.jl.
    @test _MakieExt !== nothing
    for f in (fieldplot, levelplot, quadratureplot)
        err = _thrown(() -> f(1))
        @test err isa ArgumentError
        @test occursin("Conversion failed", sprint(showerror, err))
    end
    @test !occursin("Makie backend", sprint(showerror, MethodError(levelplot, (1,))))
end

@testset "no export of Unfitted is a different export of Makie" begin
    # A session that plots loads both packages, and a name both export for different things can be called by neither's
    # bare name.
    @test isempty([n for n in intersect(names(Unfitted), names(Makie)) if getfield(Unfitted, n) !== getfield(Makie, n)])
end

@testset "levelplot draws every active cell of every level in 2D, stacked in level order" begin
    records = Unfitted._sample_levels(_makie_2d)
    fig, ax, plt = levelplot(_makie_2d)
    @test ax isa Makie.Axis
    @test ax.aspect[] isa Makie.DataAspect
    cells, edges = plt.plots
    @test cells isa Makie.Mesh && edges isa Makie.LineSegments

    # The recipe's primitives, exactly: 4 base cells and the 3 active overlay cells, coloured by their largest order.
    positions, colors = _expected_2d(records, _by_array(records, "order_max"))
    @test length(positions) == 4 * (4 + 3)
    @test _GB.coordinates(plt.cell_geometry[]) == positions
    @test _GB.faces(plt.cell_geometry[]) == [_GB.QuadFace{Int}(4i - 3, 4i - 2, 4i - 1, 4i) for i in 1:7]
    @test plt.cell_color[] == colors
    @test unique(colors) == [2.0, 3.0]

    # Each edge of a level's drawn cells once, at the level's height plus ½. Every base edge runs along a line of the
    # overlay or outside it, so no overlay cell lies over one and each is drawn whole: 12 of the base, 10 of the
    # overlay's three drawn cells.
    segments = _segments(plt.cell_edges[])
    @test allunique(segments) && length(segments) == 12 + 10
    @test Set(segments) == union(_drawn_cell_edges(records[1], -1.5), _drawn_cell_edges(records[2], -0.5))

    # What Makie made of them: each quad split as the record contract says, (1, 2, 3) and (1, 3, 4), the colours and
    # edges handed to the children, the edges in the stroke colour.
    split = ((_GB.GLTriangleFace(4i - 3, 4i - 2, 4i - 1), _GB.GLTriangleFace(4i - 3, 4i - 1, 4i)) for i in 1:7)
    @test _GB.faces(cells[1][]) == collect(Iterators.flatten(split))
    @test cells.color[] == Float32.(colors)
    @test edges[1][] == plt.cell_edges[]
    @test edges.color[] == Makie.to_color(:black)
end

@testset "levelplot colours by any cell array, draws inactive cells on request and picks levels" begin
    records = Unfitted._sample_levels(_makie_2d)
    for name in ("level_id", "role_id", "cell_id", "active", "covered", "active_dofs", "reduced_dofs")
        plt = levelplot(_makie_2d; by=Symbol(name)).plot
        @test plt.cell_color[] == _expected_2d(records, _by_array(records, name))[2]
    end

    # The switched-off overlay cell, drawn on request: every cell of both levels.
    plt = levelplot(_makie_2d; inactive=true, by=:active).plot
    positions, colors = _expected_2d(records, _by_array(records, "active"); inactive=true)
    @test _GB.coordinates(plt.cell_geometry[]) == positions
    @test count(==(0.0), colors) == 4

    # One level alone sits where the last of the drawn levels does; asking for every level is the default.
    plt = levelplot(_makie_2d; levels=2).plot
    @test _GB.coordinates(plt.cell_geometry[]) == _expected_2d(records[2:2], _by_array(records[2:2], "order_max"))[1]
    @test all(x -> x[3] == -1, _GB.coordinates(plt.cell_geometry[]))
    for every in ([1, 2], (2, 1), Set([1, 2]), 1:2)                # any collection of ids, drawn in level order
        @test levelplot(_makie_2d; levels=every).plot.cell_geometry[] == levelplot(_makie_2d).plot.cell_geometry[]
    end

    # An attribute changed on a drawn plot redraws it. Another `by` recolours the cells without meshing the levels or
    # cutting their edges again.
    plt.levels = 1
    @test length(_GB.faces(plt.cell_geometry[])) == 4
    meshes, geometry, edges = plt.level_records[], plt.cell_geometry[], plt.cell_edges[]
    plt.by = :level_id
    @test plt.cell_color[] == fill(1.0, 16)
    @test plt.level_records[] === meshes && plt.cell_geometry[] === geometry && plt.cell_edges[] === edges

    # Refusals name what was wrong and what would be right. Makie reports an error raised while a plot computes its
    # primitives wrapped in its own exception, whose message ends with the error's.
    for (keywords, says) in (((; by=:nope), "names no cell array"), ((; by=1.5), "by must name a cell array"),
                             ((; levels=7), "its levels are 1, 2"), ((; levels=:all), "levels must be nothing"))
        err = _thrown(() -> levelplot(_makie_2d; keywords...))
        @test err !== nothing && occursin(says, sprint(showerror, err))
    end
end

@testset "levelplot draws an edge only where no later level's drawn cell lies over it" begin
    # A 2×2 base under one overlay cell over its middle, [¼, ¾]², which both interior base edges cross. CairoMakie
    # paints the edge child after every face, so a base edge drawn whole would cut the overlay cell into four: each is
    # drawn only outside it, in two pieces, and every edge once — 12 base segments, 4 of the overlay.
    P(x, y, z) = Makie.Point3{Float64}(x, y, z)
    along_x(y, a, b, z) = (P(a, y, z), P(b, y, z))
    along_y(x, a, b, z) = (P(x, a, z), P(x, b, z))
    halves = ((0.0, 0.5), (0.5, 1.0))
    base(z) = Set([[along_x(y, a, b, z) for y in (0.0, 0.5, 1.0) for (a, b) in halves];
                   [along_y(x, a, b, z) for x in (0.0, 0.5, 1.0) for (a, b) in halves]])
    crossing = Set([along_x(0.5, a, b, -1.5) for (a, b) in halves] ∪ [along_y(0.5, a, b, -1.5) for (a, b) in halves])
    outside = Set([along_x(0.5, 0.0, 0.25, -1.5), along_x(0.5, 0.75, 1.0, -1.5), along_y(0.5, 0.0, 0.25, -1.5),
                   along_y(0.5, 0.75, 1.0, -1.5)])
    middle = Set([along_x(0.25, 0.25, 0.75, -0.5), along_x(0.75, 0.25, 0.75, -0.5), along_y(0.25, 0.25, 0.75, -0.5),
                  along_y(0.75, 0.25, 0.75, -0.5)])
    cut = union(setdiff(base(-1.5), crossing), outside, middle)
    covering(active) = prepare(mass(overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=2, order=1),
                                            box((0.25, 0.25), (0.75, 0.75)); cells=1, order=2, active)))

    segments = _segments(levelplot(covering(trues(1, 1))).plot.cell_edges[])
    @test allunique(segments) && Set(segments) == cut
    # An overlay cell that is not drawn hides nothing; drawn on request, it hides what it lies over.
    hidden = covering(falses(1, 1))
    segments = _segments(levelplot(hidden).plot.cell_edges[])
    @test allunique(segments) && Set(segments) == base(-1.5)
    @test Set(_segments(levelplot(hidden; inactive=true).plot.cell_edges[])) == cut
    # The base alone draws every one of its edges whole.
    @test Set(_segments(levelplot(covering(trues(1, 1)); levels=1).plot.cell_edges[])) == base(-0.5)
end

@testset "levelplot's edges cross no later drawn cell and cover the rest of every edge" begin
    # An overlay whose lines fall between the base's, with one of its cells switched off. No base segment passes
    # strictly inside a drawn overlay cell, and every point of a base cell's edge that lies outside the drawn overlay
    # cells lies on a base segment: the cut removes exactly the covered parts.
    mask = trues(2, 2)
    mask[1, 2] = false
    model = prepare(mass(overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=3, order=1), box((0.2, 0.1), (0.9, 0.8));
                                 cells=2, order=1, active=mask)))
    base, top = Unfitted._sample_levels(model)
    boxes = [(top.grid.points[4c-3], top.grid.points[4c-1]) for c in 1:4 if mask[c]]
    covered(x) = any(((lo, hi),) -> lo[1] < x[1] < hi[1] && lo[2] < x[2] < hi[2], boxes)
    # Sample points off the overlay's lines: where two drawn overlay cells meet, an edge has a single point inside
    # neither of them and on no piece, which no plot can show.
    along(a, b) = (a + (b - a) * s for s in ((1:99) .+ 0.137) ./ 100)
    on(x, (a, b)) = all(d -> min(a[d], b[d]) - 1.0e-12 <= x[d] <= max(a[d], b[d]) + 1.0e-12, 1:2)

    segments = _segments(levelplot(model).plot.cell_edges[])
    @test allunique(segments)
    drawn = filter(s -> s[1][3] == -1.5, segments)
    @test !any(s -> any(covered, along(s...)), drawn)
    edges = NTuple{2,SVector{2,Float64}}[]
    for c in 1:Unfitted._cell_count(base.grid)
        ll, lr, ur, ul = base.grid.points[(4c-3):(4c)]
        append!(edges, ((ll, lr), (ul, ur), (ll, ul), (lr, ur)))
    end
    @test all(x -> covered(x) || any(s -> on(x, s), drawn), (x for (a, b) in edges for x in along(a, b)))
    @test length(drawn) > 12                       # some base edges came out in pieces
end

@testset "levelplot colours by an error estimate" begin
    # The indicator of each drawn cell, read off the estimate's array of its level by the cell's linear index; an
    # estimate of another discretisation is refused rather than read out of order.
    records = Unfitted._sample_levels(_makie_2d)
    est = estimate(_makie_2d, solve!(_makie_2d))
    plt = levelplot(_makie_2d; by=est).plot
    η = [Float64[est.cells[k][i] for i in _record_array(record.grid, "cell_id")] for (k, record) in pairs(records)]
    @test plt.cell_color[] == _expected_2d(records, η)[2]
    @test maximum(plt.cell_color[]) > 0

    other = prepare(poisson(space(box((0.0, 0.0), (1.0, 1.0)); cells=3, order=2); source=1.0,
                            dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    err = _thrown(() -> levelplot(_makie_2d; by=estimate(other, solve!(other))))
    @test err !== nothing && occursin("ErrorEstimate holds levels of (3, 3) cells", sprint(showerror, err))
end

@testset "levelplot draws a 1D model as a ladder of its levels" begin
    # Every active cell a segment at the height of its level id, coloured at both ends, with a tick across each end.
    records = Unfitted._sample_levels(_makie_1d)
    fig, ax, plt = levelplot(_makie_1d)
    @test ax isa Makie.Axis
    @test ax.yticks[] == [1, 2]
    @test ax.ylabel[] == "level"
    # The axis labels every level of the model and shows the labels in view, which follow the levels drawn, now and
    # after a change: a plot of level 2 alone shows its one row's label, and that plot drawn whole shows both.
    fig, ax, picked = levelplot(_makie_1d; levels=2)
    @test ax.yticks[] == [1, 2]
    Makie.update_state_before_display!(fig)
    @test ax.yaxis.tickvalues[] == [2]
    picked.levels = nothing
    Makie.update_state_before_display!(fig)
    @test ax.yaxis.tickvalues[] == [1, 2]
    segments, ticks = Makie.Point2{Float64}[], Makie.Point2{Float64}[]
    colors = Float64[]
    for record in records, c in 1:Unfitted._cell_count(record.grid)
        a, b, y = record.grid.points[2c-1][1], record.grid.points[2c][1], record.level
        append!(segments, [Makie.Point2(a, y), Makie.Point2(b, y)])
        append!(colors, fill(_record_array(record.grid, "order_max")[c], 2))
        for x in (a, b)
            append!(ticks, [Makie.Point2(x, y - 0.2), Makie.Point2(x, y + 0.2)])
        end
    end
    @test plt.cell_geometry[] == segments
    @test plt.cell_color[] == colors
    @test plt.cell_edges[] == ticks
    @test length(segments) == 2 * (4 + 2)
    # The cells and the ticks, then ∂Ω, which a model without a PhysicalDomain does not draw (one NaN, see below).
    cells, ticks_plot, outline = plt.plots
    @test cells isa Makie.LineSegments && ticks_plot isa Makie.LineSegments
    @test outline isa Makie.VLines && isnan(only(outline[1][]))
end

@testset "levelplot draws every subdomain's levels or one field's, and follows its model" begin
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(1, 1), order=1)
    u1, u2 = field(:u1, V1), field(:u2, V2)
    coupled = prepare(Problem((u1, u2); blocks=(mass_block(u1), mass_block(u2))))
    @test length(_GB.faces(levelplot(coupled).plot.cell_geometry[])) == 4 + 1
    picked = levelplot(coupled; field=:u2).plot
    @test _GB.coordinates(picked.cell_geometry[]) ==
          [Makie.Point3{Float64}(x..., -1) for x in ((2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0))]

    # Drawn from an `Observable`, the plot follows the model it holds: here a model whose overlay moved.
    observed = Makie.Observable(_makie_1d)
    plt = levelplot(observed).plot
    moved = moved_space(_makie_1d.problem.space; level=2, to=box((0.5,), (1.0,)))
    observed[] = prepare(mass(moved))
    @test extrema(p[1] for p in plt.cell_geometry[] if p[2] == 2) == (0.5, 1.0)
end

@testset "quadratureplot draws every quadrature point, coloured by weight or cover" begin
    # 2D: every point of every signature at its position; 1D: at its position against its weight.
    cloud = Unfitted._sample_quadrature(_makie_2d)
    fig, ax, plt = quadratureplot(_makie_2d)
    @test ax isa Makie.Axis && ax.aspect[] isa Makie.DataAspect
    outline, points = plt.plots                         # ∂Ω, not drawn here, then the cloud, in front of it
    @test points isa Makie.Scatter && isempty(outline[1][])
    @test plt.point_positions[] == [Makie.Point2{Float64}(x...) for record in cloud for x in record.grid.points]
    @test plt.point_color[] == reduce(vcat, [last(only(record.grid.point_data)) for record in cloud])
    @test length(plt.point_positions[]) == nquadpoints(_makie_2d)
    positions = plt.point_positions[]
    plt.by = :cover_count
    @test plt.point_color[] == reduce(vcat, [fill(length(e.signature), length(e.grid.points)) for e in cloud])
    @test plt.point_positions[] === positions                   # recoloured, not sampled again
    @test sort(unique(plt.point_color[])) == [1.0, 2.0]

    fig, ax, plt = quadratureplot(_makie_1d)
    @test ax.ylabel[] == "weight"
    cloud = Unfitted._sample_quadrature(_makie_1d)
    @test plt.point_positions[] == [Makie.Point2{Float64}(x[1], w) for e in cloud
                                    for (x, w) in zip(e.grid.points, last(only(e.grid.point_data)))]

    err = _thrown(() -> quadratureplot(_makie_2d; by=:nope))
    @test err !== nothing && occursin("by must be :weight or :cover_count", sprint(showerror, err))
end

@testset "a Colorbar reads its colour mapping off either plot" begin
    # Categorical (the default `by` of a level plot, and `:cover_count`) and continuous (the weights) alike.
    cases = ((levelplot, _makie_2d, :order_max), (levelplot, _makie_1d, :order_max),
             (quadratureplot, _makie_2d, :weight), (quadratureplot, _makie_2d, :cover_count),
             (fieldplot, _makie_2d, :cover_count), (fieldplot, _makie_2d, :region_id))
    for (f, model, by) in cases
        fig, ax, plt = f(model; by)
        @test Makie.Colorbar(fig[1, 2], plt) isa Makie.Colorbar
    end
    fig, ax, plt = fieldplot(_makie_2d_solution, _makie_2d)
    @test Makie.Colorbar(fig[1, 2], plt) isa Makie.Colorbar
end

@testset "the quantities with a few discrete values are coloured as categories" begin
    # One colour per distinct value, and a colour bar with a block and a tick per value, for a cell's order, level,
    # role, activity and coverage and a point's cover count; the dof counts, the cell index, an estimate and the weights
    # keep a continuous map. `categorical` decides for any `by`, and the map wrapped is the plot's own `colormap`.
    mapping(plt) = _coloured(plt).color_mapping_type[]
    for by in (:order_max, :level_id, :role_id, :active, :covered)
        @test mapping(levelplot(_makie_2d; by).plot) == Makie.categorical
    end
    for by in (:cell_id, :active_dofs, :reduced_dofs, estimate(_makie_2d, solve!(_makie_2d)))
        @test mapping(levelplot(_makie_2d; by).plot) == Makie.continuous
    end
    @test mapping(levelplot(_makie_1d).plot) == Makie.categorical
    @test mapping(quadratureplot(_makie_2d; by=:cover_count).plot) == Makie.categorical
    @test mapping(quadratureplot(_makie_2d).plot) == Makie.continuous
    @test mapping(fieldplot(_makie_2d).plot) == Makie.categorical                  # `automatic` is :cover_count
    @test mapping(fieldplot(_makie_2d; by=:region_id).plot) == Makie.continuous
    @test mapping(fieldplot(_makie_2d_solution, _makie_2d).plot) == Makie.continuous   # `automatic` is :uh
    @test mapping(levelplot(_makie_2d; categorical=false).plot) == Makie.continuous
    @test mapping(levelplot(_makie_2d; by=:cell_id, categorical=true).plot) == Makie.categorical
    @test first(levelplot(_makie_2d; colormap=:inferno).plot.plots).colormap[] == Makie.Categorical(:inferno)
    categories = Makie.Categorical([:gray, :tomato])
    @test first(levelplot(_makie_2d; by=:covered, colormap=categories).plot.plots).colormap[] === categories

    # Settled as the plot is created: Makie cannot switch a drawn plot between categorical and continuous colours, so
    # a drawn categorical plot keeps its categories, and colours by another quantity or map within them.
    plt = levelplot(_makie_2d).plot
    plt.by = :cell_id
    plt.colormap = :inferno
    @test mapping(plt) == Makie.categorical
    @test first(plt.plots).colormap[] == Makie.Categorical(:inferno)
    @test first(plt.plots).scaled_color[] == Float32.(plt.cell_color[])

    err = _thrown(() -> levelplot(_makie_2d; categorical=:yes))
    @test err isa ArgumentError && occursin("categorical must be true, false or automatic", sprint(showerror, err))
end

@testset "the plots refuse a model of three dimensions, naming the VTK writer" begin
    m3 = prepare(mass(space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=1, order=1)))
    for (f, writer) in ((fieldplot, "write_vtk"), (levelplot, "write_vtk"), (quadratureplot, "write_quadrature_vtm"))
        err = _thrown(() -> f(m3))
        @test err isa ArgumentError
        @test occursin("3-dimensional", sprint(showerror, err)) && occursin("`$writer`", sprint(showerror, err))
    end
end

# ── fieldplot ─────────────────────────────────────────────────────────────────

@testset "fieldplot draws a 2D solution on its sample cells" begin
    # Every sample cell of every region, at `:degree`, its corners the record's own at z = −1, each cell the two
    # triangles (1, 2, 3) and (1, 3, 4) Makie splits a quad into, coloured by the solution's value at each corner; the
    # region edges are computed only while they are drawn.
    grid = only(Unfitted._sample_fields(_makie_2d_solution, _makie_2d)).grid
    fig, ax, plt = fieldplot(_makie_2d_solution, _makie_2d)
    @test ax isa Makie.Axis && ax.aspect[] isa Makie.DataAspect
    surface, edges = plt.plots
    @test surface isa Makie.Mesh && edges isa Makie.LineSegments
    n = Unfitted._cell_count(grid)
    @test _GB.coordinates(plt.sample_geometry[]) == [Makie.Point3{Float64}(x..., -1) for x in grid.points]
    @test _GB.faces(plt.sample_geometry[]) ==
          [_GB.TriangleFace{Int}(t) for c in 1:n for t in ((4c - 3, 4c - 2, 4c - 1), (4c - 3, 4c - 1, 4c))]
    @test plt.sample_color[] == _record_array(grid, "uh")
    @test surface.color[] == Float32.(plt.sample_color[])
    @test edges.linewidth[] == 0 && isempty(plt.sample_edges[])    # none drawn, nor computed, by default
    @test fieldplot(_makie_2d_solution, _makie_2d; strokewidth=2).plot.plots[2].linewidth[] == 2
    plt.strokewidth = 1
    drawn = plt.sample_edges[]
    plt.strokewidth = 2                                            # another width redraws the same edges
    @test !isempty(drawn) && plt.sample_edges[] === drawn
end

@testset "fieldplot draws each region's edges, which cross no cell" begin
    # Against the integration plan's own region boxes, of a model whose regions are sampled at different counts and
    # whose cut cells are refined, unclipped so that every region is drawn whole: every edge lies on a side of a
    # region's box, half a unit in front of the surface, and inside no region; and the edges cover every side of every
    # region's box.
    disk = physical_domain(x -> sqrt((x[1] - 0.45)^2 + (x[2] - 0.5)^2) - 0.3; lipschitz=1.0, subcell_length_scale=0.1)
    model = prepare(mass(overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=3, order=1, physical=disk),
                                 box((0.2, 0.3), (0.7, 0.8)); cells=2, order=2)))
    boxes = [Unfitted._region_box(region) for region in Unfitted.integration_plans(model)[1].regions]
    edges = _segments(fieldplot(model; strokewidth=1, clip=false).plot.sample_edges[])
    @test all(s -> s[1][3] == s[2][3] == -0.5, edges)
    flat(x) = SVector(x[1], x[2])
    within(t, lo, hi) = lo <= t <= hi
    on_side(s, b) = any(1:2) do d                                 # the edge runs along axis 3 − d at a side of `b`
        a, z, o = flat(s[1]), flat(s[2]), 3 - d
        a[d] == z[d] &&
            a[d] in (b.lower[d], b.upper[d]) &&
            within(a[o], b.lower[o], b.upper[o]) &&
            within(z[o], b.lower[o], b.upper[o])
    end
    inside(x, b) = all(d -> b.lower[d] < x[d] < b.upper[d], 1:2)
    @test all(s -> any(b -> on_side(s, b), boxes), edges)
    @test !any(s -> any(b -> inside((flat(s[1]) + flat(s[2])) / 2, b), boxes), edges)
    covered(x) = any(s -> all(d -> min(s[1][d], s[2][d]) <= x[d] <= max(s[1][d], s[2][d]), 1:2), edges)
    sides(b) = ((b.lower, SVector(b.upper[1], b.lower[2])), (SVector(b.lower[1], b.upper[2]), b.upper),
                (b.lower, SVector(b.lower[1], b.upper[2])), (SVector(b.upper[1], b.lower[2]), b.upper))
    @test all(b -> all(((a, z),) -> all(t -> covered(a + (z - a) * t), (0.13, 0.5, 0.87)), sides(b)), boxes)
end

@testset "fieldplot colours by a built-in array, a callback, a component or a magnitude" begin
    # Each `by` is the record's one entry or a built-in array, read back point by point; a cell array colours its
    # cell's four corners alike.
    grid = only(Unfitted._sample_fields(_makie_2d_solution, _makie_2d; point_data=NamedTuple())).grid
    for name in ("cover_count", "region_id")
        plt = fieldplot(_makie_2d_solution, _makie_2d; by=Symbol(name)).plot
        @test plt.sample_color[] == repeat(Float64.(_record_array(grid, name)); inner=4)
    end
    tilt = (u, c, x, xi) -> x[1] - 2x[2]
    @test fieldplot(_makie_2d_solution, _makie_2d; by=tilt).plot.sample_color[] == [x[1] - 2x[2] for x in grid.points]

    # A vector value: its magnitude by default, or one component.
    values = _record_array(only(Unfitted._sample_fields(_makie_vector_solution, _makie_vector)).grid, "uh")
    @test fieldplot(_makie_vector_solution, _makie_vector).plot.sample_color[] == [sqrt(sum(abs2, v)) for v in values]
    for c in 1:2
        @test fieldplot(_makie_vector_solution, _makie_vector; component=c).plot.sample_color[] ==
              [v[c] for v in values]
    end

    # The level set of an immersed model, at every point, unclipped.
    disk = physical_domain(x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3; lipschitz=1.0, alpha=1.0,
                           subcell_length_scale=0.125, max_depth=2)
    immersed = prepare(mass(space(box((0.0, 0.0), (1.0, 1.0)); cells=2, order=1, physical=disk)))
    plt = fieldplot(immersed; by=:level_set, clip=false).plot
    level = only(Unfitted._sample_fields(nothing, immersed)).grid
    @test plt.sample_color[] == _record_array(level, "level_set")
    @test _GB.coordinates(plt.sample_geometry[]) == [Makie.Point3{Float64}(x..., -1) for x in level.points]
end

@testset "fieldplot refuses what it cannot draw, naming the problem" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=1, order=1)
    a, b = field(:a, V), field(:b, V)
    shared = prepare(Problem((a, b); blocks=(mass_block(a), mass_block(b))))
    shared_solution = solution(shared, ones(active_unknowns(shared)))
    cases = ((() -> fieldplot(_makie_2d; by=:uh), "by = :uh plots the solution"),
             (() -> fieldplot(_makie_2d_solution, _makie_2d; by=:level_set), "carries no PhysicalDomain"),
             (() -> fieldplot(_makie_2d_solution, _makie_2d; by=:nope), "names no array a field plot shows"),
             (() -> fieldplot(_makie_2d_solution, _makie_2d; by=2), "by must name an array"),
             (() -> fieldplot(_makie_2d_solution, _makie_2d; component=1), "this value is a scalar"),
             (() -> fieldplot(_makie_vector_solution, _makie_vector; component=3), "no component of a 2-component"),
             (() -> fieldplot(_makie_2d; by=(u, c, x, xi) -> u(c, xi)), "pass the solution"),
             (() -> fieldplot(_makie_2d_solution, _makie_2d; by=(u, c, x, xi) -> "text"), "callback \"by\" returned"),
             (() -> fieldplot(shared_solution, shared), "a and b share one: pick one with `field`"))
    for (call, says) in cases
        err = _thrown(call)
        @test err !== nothing && occursin(says, sprint(showerror, err))
    end
    @test fieldplot(shared_solution, shared; field=:b).plot.sample_color[] == fill(1.0, 4)
end

@testset "fieldplot without a solution draws the regions by their cover" begin
    # The discretisation alone: `by = automatic` is `:cover_count`, read off the records sampled without a solution.
    grid = only(Unfitted._sample_fields(nothing, _makie_2d)).grid
    plt = fieldplot(_makie_2d).plot
    @test plt.solution[] === nothing
    @test _GB.coordinates(plt.sample_geometry[]) == [Makie.Point3{Float64}(x..., -1) for x in grid.points]
    @test plt.sample_color[] == repeat(Float64.(_record_array(grid, "cover_count")); inner=4)
    @test sort(unique(plt.sample_color[])) == [1.0, 2.0]
end

@testset "fieldplot warps a 2D surface to its value in an Axis3" begin
    # Sampled at 8 sub-cells per region by default, where a flat plot samples at `:degree`: a lit surface shows facets
    # and the cracks between regions sampled on different lattices, which 8 closes. A `subdivisions` given wins.
    warped(grid) = [Makie.Point3{Float64}(x..., v) for (x, v) in zip(grid.points, _record_array(grid, "uh"))]
    grid = only(Unfitted._sample_fields(_makie_2d_solution, _makie_2d; subdivisions=8)).grid
    fig, ax, plt = fieldplot(_makie_2d_solution, _makie_2d; warp=true, strokewidth=1)
    @test ax isa Makie.Axis3
    @test ax.aspect[] == (1.0, 1.0, 2 / 3) && ax.zlabel[] == "uh"
    @test _GB.coordinates(plt.sample_geometry[]) == warped(grid)
    @test isempty(plt.sample_edges[])                                # a warped surface draws no edges
    coarse = only(Unfitted._sample_fields(_makie_2d_solution, _makie_2d; subdivisions=:degree)).grid
    plt = fieldplot(_makie_2d_solution, _makie_2d; warp=true, subdivisions=:degree).plot
    @test _GB.coordinates(plt.sample_geometry[]) == warped(coarse)
    @test length(coarse.points) < length(grid.points)
    _, ax, _ = fieldplot(_makie_vector_solution, _makie_vector; warp=true, component=2)
    @test ax.zlabel[] == "uh[2]"

    # `warp` is settled with the axis, when the plot is created: a later value changes nothing drawn, either way.
    flat = fieldplot(_makie_2d_solution, _makie_2d).plot
    flat.warp = true
    @test all(x -> x[3] == -1, _GB.coordinates(flat.sample_geometry[]))
    plt.warp = false
    @test _GB.coordinates(plt.sample_geometry[]) == warped(coarse)
end

@testset "fieldplot draws a 1D graph, one segment per sample cell" begin
    # 32 sample cells per region by default, each a segment between its two ends, so a jump between cells stays one.
    m1 = prepare(poisson(_makie_1d.problem.space; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    s1 = solve!(m1)
    grid = only(Unfitted._sample_fields(s1, m1; subdivisions=32)).grid
    fig, ax, plt = fieldplot(s1, m1)
    @test ax isa Makie.Axis && ax.xlabel[] == "x" && ax.ylabel[] == "uh"
    @test plt.sample_geometry[] ==
          [Makie.Point2{Float64}(x[1], v) for (x, v) in zip(grid.points, _record_array(grid, "uh"))]
    @test length(plt.sample_geometry[]) == 2 * 32 * length(unique(_record_array(grid, "region_id")))
    graph, outline = plt.plots                                        # the graph, then ∂Ω, not drawn here
    @test graph isa Makie.LineSegments && graph[1][] == plt.sample_geometry[]
    @test outline isa Makie.VLines && isnan(only(outline[1][]))
    @test fieldplot(s1, m1; color=:red, linewidth=3).plot.plots[1].color[] == Makie.to_color(:red)
    @test length(fieldplot(s1, m1; subdivisions=:degree).plot.sample_geometry[]) < length(plt.sample_geometry[])

    # A callback that steps between regions draws the step: the two segments that meet at x = ½ end at different
    # heights.
    step = (u, c, x, xi) -> c.region.lower[1] >= 0.5 ? 1.0 : 0.0
    ends = filter(p -> p[1] == 0.5, fieldplot(s1, m1; by=step).plot.sample_geometry[])
    @test sort([p[2] for p in ends]) == [0.0, 1.0]
end

@testset "fieldplot follows its solution and model" begin
    # An `Observable` solution on a fixed model redraws with the new one; `Makie.update!` replaces both at once.
    observed = Makie.Observable(_makie_2d_solution)
    plt = fieldplot(observed, _makie_2d).plot
    doubled = solution(_makie_2d, 2 .* _makie_2d_solution.coefficients)
    observed[] = doubled
    @test plt.sample_color[] == _record_array(only(Unfitted._sample_fields(doubled, _makie_2d)).grid, "uh")
    # A model of the plot's own type — the same problem with its overlay moved — and a solution on it, at once.
    V = moved_space(_makie_2d.problem.space; level=2, to=box((0.5, 0.5), (1.0, 1.0)))
    moved = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    moved_solution = solve!(moved)
    Makie.update!(plt, moved_solution, moved)
    @test plt.sample_color[] == _record_array(only(Unfitted._sample_fields(moved_solution, moved)).grid, "uh")
end

@testset "fieldplot draws the fields of a coupled model side by side" begin
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(1, 1), order=1)
    u1, u2 = field(:u1, V1), field(:u2, V2)
    coupled = prepare(Problem((u1, u2); blocks=(mass_block(u1), mass_block(u2))))
    s = solution(coupled, [Float64(i) for i in 1:active_unknowns(coupled)])
    grids = [r.grid for r in Unfitted._sample_fields(s, coupled)]
    plt = fieldplot(s, coupled).plot
    @test _GB.coordinates(plt.sample_geometry[]) ==
          [Makie.Point3{Float64}(x..., -1) for grid in grids for x in grid.points]
    @test plt.sample_color[] == reduce(vcat, [_record_array(grid, "uh") for grid in grids])
    @test length(_GB.coordinates(fieldplot(s, coupled; field=:u2).plot.sample_geometry[])) == 4

    # Warped, the domains [0, 1] × [0, 3] and [2, 3] × [0, 1] are drawn to scale in the box that holds both, [0, 3]²,
    # whose extent is taken axis by axis — not from the corner that sorts last, (3, 1), which flattened the y axis.
    tall = space(box((0.0, 0.0), (1.0, 3.0)); cells=1, order=1)
    wide = space(box((2.0, 0.0), (3.0, 1.0)); cells=1, order=1)
    t, w = field(:t, tall), field(:w, wide)
    _, ax, _ = fieldplot(prepare(Problem((t, w); blocks=(mass_block(t), mass_block(w)))); warp=true)
    @test ax.aspect[] == (1.0, 1.0, 2 / 3)
end

# ── The ! forms ───────────────────────────────────────────────────────────────

@testset "the ! forms draw into one axis, behind what is drawn after them" begin
    # Each recipe draws into an axis made beforehand and returns its plot, in the order drawn. The field and the level
    # plot draw behind z = 0 — the surface at −1, the two levels at −2 and −1, their edges half a unit in front — so
    # that the quadrature cloud, drawn last, lies at z = 0 in front of both, as any plot drawn afterwards does.
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])
    field = fieldplot!(ax, _makie_2d_solution, _makie_2d; strokewidth=1)
    levels = levelplot!(ax, _makie_2d)
    cloud = quadratureplot!(ax, _makie_2d)
    @test field isa _MakieExt.FieldPlot && levels isa _MakieExt.LevelPlot && cloud isa _MakieExt.QuadraturePlot
    @test ax.scene.plots == [field, levels, cloud]
    heights(points) = sort!(unique(p[3] for p in points))
    @test heights(_GB.coordinates(field.sample_geometry[])) == [-1] && heights(field.sample_edges[]) == [-0.5]
    @test heights(_GB.coordinates(levels.cell_geometry[])) == [-2, -1]
    @test heights(levels.cell_edges[]) == [-1.5, -0.5]
    @test eltype(cloud.point_positions[]) == Makie.Point2{Float64}               # in the plane z = 0
    # Without an axis, a `!` form draws into the current one; `fieldplot!` without a solution draws the regions.
    regions = fieldplot!(_makie_2d)
    @test regions.solution[] === nothing && last(ax.scene.plots) === regions
end

# ── Clipping and ∂Ω ───────────────────────────────────────────────────────────

# The area of a 2D mesh, from its triangles' `x` and `y`.
function _mesh_area(mesh)
    x = _GB.coordinates(mesh)
    return sum(_GB.faces(mesh); init=0.0) do (a, b, c)
        return abs((x[b][1] - x[a][1]) * (x[c][2] - x[a][2]) - (x[c][1] - x[a][1]) * (x[b][2] - x[a][2])) / 2
    end
end

# The total length of line segments given as consecutive pairs of points, in `x` and `y`.
_segments_length(points) = sum(((a, b),) -> sqrt((b[1] - a[1])^2 + (b[2] - a[2])^2), _segments(points); init=0.0)

# A planar level set, φ = x − 0.3, on a 4×4 order-2 square and on four order-1 cells of the unit interval: its linear
# interpolant on the sample cells is φ itself, so a clip at it is exact.
_makie_plane = prepare(mass(space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=2,
                                  physical=physical_domain(leaf(x -> x[1] - 0.3; lipschitz=1.0);
                                                           subcell_length_scale=0.1))))
_makie_plane_1d = prepare(mass(space(box((0.0,), (1.0,)); cells=4, order=1,
                                     physical=physical_domain(leaf(x -> x[1] - 0.3; lipschitz=1.0);
                                                              subcell_length_scale=0.1))))

@testset "fieldplot clips a 2D field to Ω, exactly at a planar level set" begin
    # Ω is x ≤ 0.3: the clipped mesh covers 0.3 of the unit square, keeps no point where φ > 0, puts every point it adds
    # on x = 0.3, and colours it with the linear field's own value there.
    linear = (u, c, x, xi) -> 2x[1] + 3x[2]
    plt = fieldplot(_makie_plane; by=linear).plot
    mesh = plt.sample_geometry[]
    points = _GB.coordinates(mesh)
    @test _mesh_area(mesh) ≈ 0.3 atol = 1.0e-14
    @test all(p -> p[1] - 0.3 <= 1.0e-15, points)
    grid = only(Unfitted._sample_fields(nothing, _makie_plane; point_data=(by=linear,))).grid
    sampled = Set(Makie.Point3{Float64}(x..., -1) for x in grid.points)
    added = filter(p -> !(p in sampled), points)
    @test !isempty(added) && all(p -> abs(p[1] - 0.3) <= 1.0e-15, added)
    @test maximum(abs(c - (2p[1] + 3p[2])) for (p, c) in zip(points, plt.sample_color[])) <= 1.0e-14

    # `clip = false` draws the sample cells whole, the corners in the record's order; `automatic` clips a field whose
    # space carries a PhysicalDomain, and draws one without whole.
    whole = fieldplot(_makie_plane; by=linear, clip=false).plot
    @test _GB.coordinates(whole.sample_geometry[]) == [Makie.Point3{Float64}(x..., -1) for x in grid.points]
    @test _mesh_area(whole.sample_geometry[]) ≈ 0.5                # the cells the fictitious fold kept
    @test fieldplot(_makie_plane; by=linear, clip=true).plot.sample_geometry[] == mesh
    @test fieldplot(_makie_2d).plot.sample_geometry[] == fieldplot(_makie_2d; clip=false).plot.sample_geometry[]

    # Toggling `clip` or `boundary` on a drawn plot redraws it from the samples it took, without sampling again, and
    # redraws only what it changes.
    samples = plt.samples[]
    plt.clip = false
    surface = plt.sample_geometry[]
    @test surface == whole.sample_geometry[] && plt.samples[] === samples
    plt.boundary = true
    @test !isempty(plt.sample_boundary[]) && plt.samples[] === samples && plt.sample_geometry[] === surface

    # A model on the same levels without a PhysicalDomain is of the same type, and the plot draws it whole: the samples
    # are of one type whether or not they carry a level set.
    observed = Makie.Observable(_makie_plane)
    plt = fieldplot(observed).plot
    @test _mesh_area(plt.sample_geometry[]) ≈ 0.3 atol = 1.0e-14
    observed[] = prepare(mass(space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=2)))
    @test _mesh_area(plt.sample_geometry[]) ≈ 1
end

@testset "fieldplot's clipped area and ∂Ω converge at second order in cut_depth" begin
    # A disk of radius r, whose boundary the clip and the contour facet at the size of the cells `cut_depth` refines
    # it to: each bisection quarters the area's error and the contour length's.
    r = 0.35
    disk = physical_domain(leaf(x -> hypot(x[1] - 0.5, x[2] - 0.5) - r; lipschitz=1.0); subcell_length_scale=0.05)
    model = prepare(mass(space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=1, physical=disk)))
    area, length = Float64[], Float64[]
    for depth in 1:5
        plt = fieldplot(model; cut_depth=depth, boundary=true).plot
        push!(area, abs(_mesh_area(plt.sample_geometry[]) - π * r^2))
        push!(length, abs(_segments_length(plt.sample_boundary[]) - 2π * r))
    end
    @test all(ratio -> 3.5 <= ratio <= 4.5, area[1:(end-1)] ./ area[2:end])
    @test all(ratio -> 3.5 <= ratio <= 4.5, length[1:(end-1)] ./ length[2:end])
    @test last(area) < 1.0e-4 && last(length) < 1.0e-4
end

@testset "fieldplot clips region edges, a warped surface and a 1D graph alike" begin
    linear = (u, c, x, xi) -> 2x[1] + 3x[2]
    # The region edges keep the parts in Ω, cut at x = 0.3.
    edges = fieldplot(_makie_plane; by=linear, strokewidth=1).plot.sample_edges[]
    @test all(p -> p[1] - 0.3 <= 1.0e-15, edges)
    @test any(p -> abs(p[1] - 0.3) <= 1.0e-15, edges)
    @test length(edges) < length(fieldplot(_makie_plane; by=linear, strokewidth=1, clip=false).plot.sample_edges[])

    # A warped surface is clipped the same way, every point, the clip points included, at the value's height.
    points = _GB.coordinates(fieldplot(_makie_plane; by=linear, warp=true).plot.sample_geometry[])
    @test all(p -> p[1] - 0.3 <= 1.0e-15, points)
    @test maximum(p -> abs(p[3] - (2p[1] + 3p[2])), points) <= 1.0e-14

    # A 1D graph ends at the root, with the value interpolated there.
    graph = fieldplot(_makie_plane_1d; by=(u, c, x, xi) -> 1 + 2x[1]).plot.sample_geometry[]
    @test maximum(p -> p[1], graph) ≈ 0.3 atol = 1.0e-15
    @test all(p -> abs(p[2] - (1 + 2p[1])) <= 1.0e-14, graph)
    @test length(graph) < length(fieldplot(_makie_plane_1d; clip=false).plot.sample_geometry[])
end

@testset "boundary draws ∂Ω on every plot, and needs a PhysicalDomain" begin
    # 2D: ∂Ω is the line x = 0.3, of length 1, drawn at z = −¼ by every plot, a field plot on its own sample cells and
    # the others on the sampler's defaults; on a warped field plot at the value's height. A field and a level plot draw
    # it last, in front of every cell and edge; a quadrature plot first, behind its points, which are what it shows. 1D:
    # one vertical line, at the root.
    linear = (u, c, x, xi) -> 2x[1] + 3x[2]
    @test first(quadratureplot(_makie_plane; boundary=true).plot.plots) isa Makie.LineSegments
    for plt in (fieldplot(_makie_plane; boundary=true).plot, levelplot(_makie_plane; boundary=true).plot,
                quadratureplot(_makie_plane; boundary=true).plot)
        outline = _outline(plt)
        @test outline isa Makie.LineSegments && outline.color[] == Makie.to_color(:black)
        points = outline[1][]
        @test !isempty(points) && all(p -> abs(p[1] - 0.3) <= 1.0e-15 && p[3] == -0.25, points)
        @test _segments_length(points) ≈ 1
    end
    warped = fieldplot(_makie_plane; by=linear, warp=true, boundary=true).plot.sample_boundary[]
    @test maximum(p -> abs(p[3] - (2p[1] + 3p[2])), warped) <= 1.0e-14
    for plt in (fieldplot(_makie_plane_1d; boundary=true).plot, levelplot(_makie_plane_1d; boundary=true).plot,
                quadratureplot(_makie_plane_1d; boundary=true).plot)
        @test _outline(plt) isa Makie.VLines
        @test only(_outline(plt)[1][]) ≈ 0.3 atol = 1.0e-15
    end
    @test isempty(fieldplot(_makie_plane).plot.sample_boundary[])        # off by default
    # Two fields on the immersed space: its ∂Ω is sampled and drawn once.
    a, b = field(:a, _makie_plane.problem.space), field(:b, _makie_plane.problem.space)
    two = prepare(Problem((a, b); blocks=(mass_block(a), mass_block(b))))
    @test _segments_length(_outline(levelplot(two; boundary=true).plot)[1][]) ≈ 1
    @test last(levelplot(_makie_plane; boundary=true, boundarycolor=:red, boundarywidth=3).plot.plots).linewidth[] == 3

    # A plane through sample points, φ = x − ½ on cells with corners on x = ½, where every sample edge along it bounds
    # a triangle on either side — on the left the full cells' and on the right the cut cells' refined to `cut_depth` —
    # and is drawn once, by the triangle in Ω.
    left = physical_domain(leaf(x -> x[1] - 0.5; lipschitz=1.0); subcell_length_scale=0.1)
    half = prepare(mass(space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=1, physical=left)))
    plt = fieldplot(half; boundary=true).plot
    @test _segments_length(plt.sample_boundary[]) ≈ 1 && all(p -> p[1] == 0.5, plt.sample_boundary[])
    @test _mesh_area(plt.sample_geometry[]) ≈ 0.5
    # The cut cells right of it touch Ω along that edge alone and add no point to the mesh: every point is a triangle's.
    @test length(Set(k for face in _GB.faces(plt.sample_geometry[]) for k in face)) ==
          length(_GB.coordinates(plt.sample_geometry[]))

    # Makie's VLines fails on no value at all when an axis computes its limits, which it does for every child of a
    # plot, so a 1D plot without a root draws its ∂Ω from one NaN, which neither draws nor moves the limits.
    for call in (fieldplot, levelplot, quadratureplot)
        _, ax, plt = call(_makie_1d)
        @test isnan(only(_outline(plt)[1][]))
        Makie.reset_limits!(ax)
        @test all(isfinite, ax.finallimits[].origin) && all(isfinite, ax.finallimits[].widths)
    end
    plt = fieldplot(_makie_1d).plot
    @test Makie.boundingbox(plt) == Makie.boundingbox(first(plt.plots))       # the graph's own

    # A plot of no field with a PhysicalDomain has no ∂Ω to draw, nor a clip to make.
    cases = ((() -> fieldplot(_makie_2d; boundary=true), "fieldplot draws ∂Ω"),
             (() -> levelplot(_makie_2d; boundary=true), "levelplot draws ∂Ω"),
             (() -> quadratureplot(_makie_2d; boundary=true), "quadratureplot draws ∂Ω"),
             (() -> fieldplot(_makie_2d; clip=true), "clip = true cuts the plot at ∂Ω"),
             (() -> fieldplot(_makie_plane; clip=:yes), "clip must be true, false or automatic"))
    for (call, says) in cases
        err = _thrown(call)
        @test err !== nothing && occursin(says, sprint(showerror, err))
    end
end
