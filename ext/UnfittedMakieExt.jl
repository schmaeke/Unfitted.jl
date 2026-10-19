"""
    UnfittedMakieExt

Package extension that plots Unfitted's discretisations with [Makie](https://docs.makie.org). Loading Makie — through
any of its backends, CairoMakie, GLMakie or WGLMakie — next to `Unfitted` installs the recipes below; without it the
entry points, which `src/stubs.jl` declares and documents, raise a `MethodError` whose hint says to load a backend.

The extension samples nothing itself. `src/postprocessing.jl` samples a solution, the meshes of the levels and the
quadrature-point cloud into consumer-neutral records — `Unfitted.SampleGrid`, whose contract heads that file — and this
extension draws them. Everything Makie-specific is decided here and nowhere else:

  - the primitives: a 2D field mesh as a `GeometryBasics.Mesh` of triangles, a 2D level mesh as one of quads, a 1D
    field graph, a 1D level mesh, the cell edges and a 2D `∂Ω` as line segments, a 1D `∂Ω` as vertical lines, the
    quadrature cloud as a scatter;
  - the colour of a cell: its record value repeated over the cell's own corners, which no other cell shares, so that
    the colour is flat and exact rather than interpolated across a neighbour;
  - the cut of a field plot at `∂Ω`, and `∂Ω` itself: the zero set of the level set's linear interpolant on each
    sample triangle, or each 1D sample cell. A triangle is cut by one pass, against the half-plane where the interpolant
    is ≤ 0, of the polygon clipper of I. E. Sutherland, G. W. Hodgman, "Reentrant polygon clipping", Commun. ACM 17
    (1974) 32–42, doi:10.1145/360767.360802 — a clean-room implementation, as in `src/surface.jl` (`NOTICE.md`);
  - the stacking of the levels and of `∂Ω` in front of them, and the axis each plot asks for.

Provided pieces:

  - `Unfitted.fieldplot(solution, model; …)`, `Unfitted.fieldplot(model; …)` and `Unfitted.fieldplot!(…)` — the recipe
    `FieldPlot`;
  - `Unfitted.levelplot(model; …)` and `Unfitted.levelplot!(…)` — the recipe `LevelPlot`;
  - `Unfitted.quadratureplot(model; …)` and `Unfitted.quadratureplot!(…)` — the recipe `QuadraturePlot`.

`fieldplot` draws a solution, or without one the model's integration regions; the other two take the model as their
one argument and need no solution: they draw the discretisation.
"""
module UnfittedMakieExt

using Unfitted: Field, Model, Solution, ErrorEstimate, SampleGrid, _sample_fields, _sample_levels, _sample_quadrature,
                _sampled_fields, _fields_by_space, _default_point_data, _cell_count
import Unfitted: fieldplot, fieldplot!, levelplot, levelplot!, quadratureplot, quadratureplot!
import Makie
using Makie: GeometryBasics, Point2, Point3
using StaticArrays: SVector

# ── Recipes ───────────────────────────────────────────────────────────────────
#
# The plot functions are Unfitted's methodless `levelplot` and `quadratureplot` (and their `!` forms), imported above,
# so `@recipe` adds its methods to them rather than defining functions of its own. The model argument is named
# `discretisation` because every Makie plot already carries an attribute `model`, its transformation matrix, and an
# argument of the same name would shadow it. Typing it as a `Model` makes Makie refuse any other argument with its own
# conversion error, which names the types it expected.

"""
The Makie recipe behind Unfitted's [`levelplot`](@ref), whose documentation precedes this one: what follows is Makie's
reference of the plot's attributes.
"""
Makie.@recipe LevelPlot (discretisation::Model,) begin
    "What colours a cell: the name of a level-mesh cell array, or an `ErrorEstimate` of the model."
    by = :order_max
    "The levels to draw, by id: `nothing` for every level, one id, or a collection of ids."
    levels = nothing
    "Whether to draw the cells that a level's activation mask or the fictitious fold switched off."
    inactive = false
    "The field whose space is drawn, a `Field` or its name; `nothing` for the space of every field."
    field = nothing
    "The colour of the cell edges, and in 1D of the ticks at the cell ends."
    strokecolor = :black
    "The width in pixels of the cell edges and ticks; `0` draws none."
    strokewidth = 0.5
    "In 1D, the width in pixels of the segments that carry the colour."
    linewidth = 6
    "Whether to draw ∂Ω, the zero contour of the level set's linear interpolant, which needs a `PhysicalDomain`."
    boundary = false
    "The colour of ∂Ω."
    boundarycolor = :black
    "The width of ∂Ω, in pixels."
    boundarywidth = 1.5
    """
    Whether the colours are categories, one colour and one colour-bar block per value: `true`, `false`, or `automatic`
    for the quantities with a few discrete values — `:order_max`, `:level_id`, `:role_id`, `:active` and `:covered` —
    and not for the dof counts, the cell index or an error estimate. Settled when the plot is created.
    """
    categorical = Makie.automatic
    Makie.mixin_colormap_attributes()...
    Makie.mixin_generic_plot_attributes()...
end

"""
The Makie recipe behind Unfitted's [`quadratureplot`](@ref), whose documentation precedes this one: what follows is
Makie's reference of the plot's attributes.
"""
Makie.@recipe QuadraturePlot (discretisation::Model,) begin
    "What colours a point: `:weight`, its physical weight, or `:cover_count`, the number of levels covering it."
    by = :weight
    "The marker of a point."
    marker = :circle
    "The size of a point's marker, in pixels."
    markersize = 6
    "Whether to draw ∂Ω, the zero contour of the level set's linear interpolant, which needs a `PhysicalDomain`."
    boundary = false
    "The colour of ∂Ω."
    boundarycolor = :black
    "The width of ∂Ω, in pixels."
    boundarywidth = 1.5
    "Whether the colours are categories: `true`, `false`, or `automatic` for `:cover_count` alone. Settled at creation."
    categorical = Makie.automatic
    Makie.mixin_colormap_attributes()...
    Makie.mixin_generic_plot_attributes()...
end

"""
The Makie recipe behind Unfitted's [`fieldplot`](@ref), whose documentation precedes this one: what follows is Makie's
reference of the plot's attributes.
"""
Makie.@recipe FieldPlot (solution::Union{Nothing,Solution}, discretisation::Model) begin
    """
    What is plotted: a built-in array — `:uh`, `:level_set`, `:region_id`, `:cover_count` — or a sampling callback
    `(u, context, x, xi) -> value`; `automatic` plots `:uh` with a solution and `:cover_count` without.
    """
    by = Makie.automatic
    "For a vector value, `nothing` for its magnitude or the index of the component to plot; for a scalar, `nothing`."
    component = nothing
    "In 2D, whether to lift the surface to the plotted value, in an `Axis3`. Settled when the plot is created."
    warp = false
    "The field to plot, a `Field` or its name; `nothing` for every field, each on its own subdomain."
    field = nothing
    "The sample cells per region: `automatic` — `:degree` flat in 2D, 8 warped, 32 in 1D — or any sampler value."
    subdivisions = Makie.automatic
    "The extra bisections of the sample cells that ∂Ω passes through: `:none` or an integer in `0:8`."
    cut_depth = 3
    "In 1D, the colour of the graph."
    color = @inherit linecolor
    "In 1D, the width of the graph in pixels."
    linewidth = @inherit linewidth
    "In 2D, the colour of the region edges."
    strokecolor = :black
    "In 2D, the width in pixels of the region edges; `0` draws none."
    strokewidth = 0
    """
    Whether to cut the drawing at ∂Ω and draw only Ω = {φ ≤ 0}: `true`, `false` for the polynomial extension outside Ω
    as well, or `automatic` for every field whose space carries a `PhysicalDomain`.
    """
    clip = Makie.automatic
    "Whether to draw ∂Ω, the zero contour of the level set's linear interpolant, which needs a `PhysicalDomain`."
    boundary = false
    "The colour of ∂Ω."
    boundarycolor = :black
    "The width of ∂Ω, in pixels."
    boundarywidth = 1.5
    """
    Whether the colours are categories, one colour and one colour-bar block per value: `true`, `false`, or `automatic`
    for `:cover_count` alone. Settled when the plot is created.
    """
    categorical = Makie.automatic
    Makie.mixin_colormap_attributes()...
    Makie.mixin_generic_plot_attributes()...
end

# The form without a solution, which draws the discretisation's regions: the recipe's first argument is then `nothing`,
# so both forms are one plot type, and an `Observable` of either argument updates it.
Makie.convert_arguments(::Type{<:FieldPlot}, model::Model) = (nothing, model)

# Each recipe samples its model in a computation of its own and derives each primitive from the samples in another,
# which reads only the attributes that primitive shows, so that a change of an attribute — another `by`, other
# `levels` — recomputes what it changes and samples nothing again, while a new model in an `Observable` samples anew.
# The children read the colour mapping from the recipe's own attributes, the colour map through `_colormap_node!`. The
# dimension check comes first, before anything is sampled.
#
# A level plot meshes its levels from the model and the field; picks the drawn cells from `levels` and `inactive`, and
# draws them and their edges, which in 2D are cut where a later level hides them; and colours them from `by`. Another
# `by` recolours the cells without meshing — building the coverage — or cutting the edges again.
function Makie.plot!(p::LevelPlot{<:Tuple{<:Model{D}}}) where {D}
    _check_dimension(Val(D), :levelplot, :write_vtk)
    map!((model, field) -> _sample_levels(model; field), p.attributes, [:discretisation, :field], :level_records)
    map!(_drawn_levels, p.attributes, [:level_records, :levels, :inactive], :drawn_levels)
    map!(drawn -> _level_primitives(Val(D), drawn), p.attributes, :drawn_levels, [:cell_geometry, :cell_edges])
    map!(_cell_colors, p.attributes, [:discretisation, :level_records, :drawn_levels, :by], :cell_color)
    _draw_cells!(p, Val(D), _colormap_node!(p, :cell_colormap))
    Makie.linesegments!(p, Makie.shared_attributes(p, Makie.LineSegments), p.cell_edges; color=p.strokecolor,
                        linewidth=p.strokewidth)
    map!(p.attributes, [:discretisation, :field, :boundary], :model_boundary) do model, field, boundary
        return _model_boundary(Val(D), model, field, boundary, :levelplot)
    end
    _draw_boundary!(p, :model_boundary, Val(D))
    return _refuse_now!(p, :cell_color)
end

# Makie computes a child's geometry as it creates the child but its colours only when it first draws it, so a `by` the
# plot cannot colour by would be refused by whatever draws the plot rather than by the call that passed it. The plot
# reads its colours `name` once as it is created, which refuses it there, as every other attribute is refused.
function _refuse_now!(p, name::Symbol)
    getproperty(p, name)[]
    return p
end

# The coloured cells, the first child of a level plot: a mesh of quads in 2D, segments in 1D, coloured through
# `colormap`, the node `_colormap_node!` chose.
function _draw_cells!(p::LevelPlot, ::Val{2}, colormap)
    return Makie.mesh!(p, Makie.shared_attributes(p, Makie.Mesh), p.cell_geometry; color=p.cell_color, colormap,
                       shading=false)
end
function _draw_cells!(p::LevelPlot, ::Val{1}, colormap)
    return Makie.linesegments!(p, Makie.shared_attributes(p, Makie.LineSegments), p.cell_geometry; color=p.cell_color,
                               colormap)
end

function Makie.plot!(p::FieldPlot{<:Tuple{<:Union{Nothing,Solution},<:Model{D}}}) where {D}
    _check_dimension(Val(D), :fieldplot, :write_vtk)
    # `warp` chose the axis the plot is drawn in, so it is read once, here, and a later value is ignored, as a later
    # `categorical` is.
    warp = p.warp[]
    # The sampling, the costly part, is a computation of its own, and each child draws from the samples in one of its
    # own, which reads only the attributes that child shows: toggling `clip` or `boundary` redraws without sampling
    # again, and redraws only the children it changes.
    inputs = [:solution, :discretisation, :by, :component, :field, :subdivisions, :cut_depth]
    map!(p.attributes, inputs, :samples) do solution, model, by, component, field, subdivisions, cut_depth
        return _field_samples(solution, model, by, component, field, subdivisions, cut_depth, warp)
    end
    _draw_field!(p, Val(D), warp)
    map!(p.attributes, [:samples, :boundary], :sample_boundary) do samples, boundary
        return _field_boundary(Val(D), samples, boundary, warp)
    end
    _draw_boundary!(p, :sample_boundary, Val(D))
    return p
end

# The children of a field plot: in 2D the coloured surface, first, and the region edges, which a warped surface does not
# draw; in 1D the graph, in one colour.
function _draw_field!(p::FieldPlot, ::Val{2}, warp::Bool)
    map!(p.attributes, [:samples, :clip], [:sample_geometry, :sample_color]) do samples, clip
        return _surface(samples, clip, warp)
    end
    # The region edges are computed only while they are drawn: `draws_edges` changes only when the width crosses zero,
    # so a change of width alone recomputes nothing.
    map!(width -> !warp && width > 0, p.attributes, :strokewidth, :draws_edges)
    map!(p.attributes, [:samples, :clip, :draws_edges], :sample_edges) do samples, clip, draws_edges
        return draws_edges ? _region_edges(samples, clip) : Point3{Float64}[]
    end
    colormap = _colormap_node!(p, :sample_colormap, _resolved_by(p.by[], p.solution[]))
    # A warped surface is lit, so that its shape reads apart from its colour; a flat one is not, so that its colours are
    # the colour map's own.
    Makie.mesh!(p, Makie.shared_attributes(p, Makie.Mesh), p.sample_geometry; color=p.sample_color, colormap,
                shading=warp)
    return Makie.linesegments!(p, Makie.shared_attributes(p, Makie.LineSegments), p.sample_edges; color=p.strokecolor,
                               linewidth=p.strokewidth)
end
function _draw_field!(p::FieldPlot, ::Val{1}, _)
    map!(_graph, p.attributes, [:samples, :clip], :sample_geometry)
    return Makie.linesegments!(p, Makie.shared_attributes(p, Makie.LineSegments), p.sample_geometry)
end

# The child that draws ∂Ω, the plot's attribute `name`, last, so that CairoMakie, painting children in order, puts it
# over everything the plot drew before: in 2D the contour as line segments, in 1D a vertical line at each root.
#
# Makie's `VLines` takes the extrema of its values for the axis limits, which fails on no value at all — and an axis
# computes the limits of every child of a plot, the invisible ones included — so a 1D plot without a root, which is
# every one without `boundary`, hands it one NaN instead, which it neither draws nor counts in the limits.
function _draw_boundary!(p, name::Symbol, ::Val{2})
    return Makie.linesegments!(p, Makie.shared_attributes(p, Makie.LineSegments), getproperty(p, name);
                               color=p.boundarycolor, linewidth=p.boundarywidth)
end
function _draw_boundary!(p, name::Symbol, ::Val{1})
    map!(roots -> isempty(roots) ? [NaN] : roots, p.attributes, name, :boundary_lines)
    return Makie.vlines!(p, Makie.shared_attributes(p, Makie.VLines), p.boundary_lines; color=p.boundarycolor,
                         linewidth=p.boundarywidth)
end

# The points are what a quadrature plot shows, and `∂Ω` is drawn for reference behind them in every backend: its child
# comes first, which a backend that paints in child order (CairoMakie) paints first, and lies at `_BOUNDARY_Z`, behind
# the points at `z = 0`, which a backend that tests depth (GLMakie) draws behind them.
function Makie.plot!(p::QuadraturePlot{<:Tuple{<:Model{D}}}) where {D}
    _check_dimension(Val(D), :quadratureplot, :write_quadrature_vtm)
    map!(p.attributes, [:discretisation, :boundary], :model_boundary) do model, boundary
        return _model_boundary(Val(D), model, nothing, boundary, :quadratureplot)
    end
    _draw_boundary!(p, :model_boundary, Val(D))
    map!(_sample_quadrature, p.attributes, :discretisation, :quadrature_records)
    map!(records -> _cloud_positions(Val(D), records), p.attributes, :quadrature_records, :point_positions)
    map!(_cloud_colors, p.attributes, [:quadrature_records, :by], :point_color)
    Makie.scatter!(p, Makie.shared_attributes(p, Makie.Scatter), p.point_positions; color=p.point_color,
                   colormap=_colormap_node!(p, :point_colormap))
    return _refuse_now!(p, :point_color)
end

# The colour map node a plot's coloured child reads: the plot's own `colormap`, or, for a categorical plot, that map
# wrapped in `Makie.Categorical` — one colour per distinct value, and a colour bar with one block and one tick per
# value — in the node `name`. Wrapping the plot's map rather than replacing it keeps the theme's colour map and one the
# user passes; a map that is categorical already is kept as it is.
#
# The choice is settled once, from `categorical` and `by` as the plot is created: Makie colours a categorical and a
# continuous map through different computations, and neither a drawn plot nor a `Colorbar` reading it can switch from
# one to the other. A later `by` or `colormap` is drawn within the choice made; a later `categorical` changes nothing.
# `by` is the quantity the plot shows, which a recipe whose `by` defaults to `automatic` resolves first.
function _colormap_node!(p, name::Symbol, by=p.by[])
    _is_categorical(p.categorical[], by) || return p.colormap
    map!(map -> map isa Makie.Categorical ? map : Makie.Categorical(map), p.attributes, :colormap, name)
    return getproperty(p, name)
end

# Whether a plot is categorical: as `categorical` says, and for `automatic` when `by` names one of the quantities
# `_CATEGORICAL_QUANTITIES` lists.
function _is_categorical(categorical, by)
    categorical === Makie.automatic && return by isa Symbol && by in _CATEGORICAL_QUANTITIES
    categorical isa Bool && return categorical
    throw(ArgumentError("categorical must be true, false or automatic; got $(repr(categorical))"))
end

# The quantities that take a few discrete values, which a categorical colour map shows a block each: a cell's order,
# level, role, activity and coverage, and the number of levels over a quadrature point. The dof counts and the cell
# index take too many values for a block each to read, and an error estimate and the weights vary continuously.
const _CATEGORICAL_QUANTITIES = (:order_max, :level_id, :role_id, :active, :covered, :cover_count)

# Plots draw one- and two-dimensional models. A three-dimensional one has no view a plot could show without slicing or
# clipping it, which ParaView does on the VTK writer's output, so the refusal names the writer.
function _check_dimension(::Val{D}, plot::Symbol, writer::Symbol) where {D}
    D <= 2 || throw(ArgumentError("$plot draws one- and two-dimensional models, and this one is $D-dimensional: " *
                                  "write it with `$writer` and look at it in ParaView"))
    return nothing
end

# ── Axes and colour bars ──────────────────────────────────────────────────────

# The recipes' plots of a `D`-dimensional model, which is a field plot's second argument and the others' only one. Makie
# asks a plot for its axis once its arguments are converted, so these hooks take the plot alone and read the rest off
# it.
const _PlotOf{D} = Union{LevelPlot{<:Tuple{<:Model{D}}},QuadraturePlot{<:Tuple{<:Model{D}}},
                         FieldPlot{<:Tuple{Any,<:Model{D}}}}

# Every plot draws in a 2D axis but a warped 2D field plot, which goes into an `Axis3`: that gives the height an axis
# and ticks of its own, where an `LScene` would show the surface without any. The axis follows the model's dimension
# and the plot's `warp`, which is why `warp` is settled when the plot is created. In 2D a model is its own plane, drawn
# to scale; in 1D the second axis carries the level (`levelplot`), the weight (`quadratureplot`) or the plotted value
# (`fieldplot`). A warped field plot has the domain to scale in the horizontal and the plotted value on the vertical
# axis.
Makie.preferred_axis_type(p::FieldPlot{<:Tuple{Any,<:Model{2}}}) = p.warp[] ? Makie.Axis3 : Makie.Axis
Makie.preferred_axis_type(::Union{LevelPlot,QuadraturePlot,FieldPlot}) = Makie.Axis
Makie.preferred_axis_attributes(::Type{Makie.Axis}, ::_PlotOf{2}) = (; aspect=Makie.DataAspect())
function Makie.preferred_axis_attributes(::Type{Makie.Axis}, p::LevelPlot{<:Tuple{<:Model{1}}})
    return (; ylabel="level", yticks=_level_rows(p.discretisation[]))
end
Makie.preferred_axis_attributes(::Type{Makie.Axis}, ::QuadraturePlot{<:Tuple{<:Model{1}}}) = (; ylabel="weight")
function Makie.preferred_axis_attributes(::Type{Makie.Axis}, p::FieldPlot{<:Tuple{Any,<:Model{1}}})
    return (; xlabel="x", ylabel=_field_label(p))
end
function Makie.preferred_axis_attributes(::Type{Makie.Axis3}, p::FieldPlot{<:Tuple{Any,<:Model{2}}})
    return (; aspect=_warped_aspect(p.discretisation[], p.field[]), zlabel=_field_label(p))
end

# The rows a 1D level plot's axis labels: every level of the model, by id. The axis is made once, with the plot, and
# Makie shows the ticks that lie in view, so labelling every row the model has keeps the rows labelled whatever
# `levels` and `field` pick, then or later. Read off the spaces rather than off the meshes, so that the axis samples
# nothing.
_level_rows(model::Model) = sort!(unique!([level.id for fld in model.problem.fields for level in fld.space.levels]))

# The name of the quantity a field plot shows, for its axis: the array `by` names, `value` for a callback, and the
# component picked, if one is.
function _field_label(p::FieldPlot)
    by, component = _resolved_by(p.by[], p.solution[]), p.component[]
    name = by isa Symbol ? String(by) : "value"
    return component isa Integer ? "$name[$component]" : name
end

# The aspect of a warped plot's `Axis3`: the box that holds the drawn fields' domains to scale in `x` and `y`, the
# height two thirds of its larger extent, as `Axis3`'s own default has it for a square. The box is taken axis by axis:
# `maximum` and `minimum` over the corners would compare them as vectors, lexicographically.
function _warped_aspect(model::Model, field)
    domains = [fld.space.domain for fld in _sampled_fields(model, field)]
    lower = reduce((a, b) -> min.(a, b), (d.lower for d in domains))
    upper = reduce((a, b) -> max.(a, b), (d.upper for d in domains))
    widths = upper - lower
    return (widths[1] / maximum(widths), widths[2] / maximum(widths), 2 / 3)
end

# A `Colorbar` reads the colour mapping off the child that carries the colours: the first child of a level or a field
# plot, whose edges and `∂Ω` are drawn in one solid colour and have none, and the scatter of a quadrature plot, which
# draws its `∂Ω` first so that the points lie in front of it.
Makie.extract_colormap(p::Union{LevelPlot,FieldPlot}) = Makie.extract_colormap(first(p.plots))
# A 1D field plot draws its value as the height of a graph in one colour: it has no colour map to read, and a
# `Colorbar` of it is refused, by name, rather than with Makie's report of the colours of its several children.
function Makie.extract_colormap(::FieldPlot{<:Tuple{Any,<:Model{1}}})
    throw(ArgumentError("a 1D field plot draws its value as the height of a graph in one colour, and has no colour " *
                        "map for a Colorbar to read"))
end
Makie.extract_colormap(p::QuadraturePlot) = Makie.extract_colormap(only(c for c in p.plots if c isa Makie.Scatter))

# ── Record arrays ─────────────────────────────────────────────────────────────

# The array `name` among a record's `arrays` — its `point_data` or its `cell_data` — or `nothing` when it has none of
# that name. A record holds its arrays as `AbstractVector`s, so a loop that reads one in its own body runs on an
# abstract type and dispatches at every element: an array found here is handed, as an argument, to a function that
# loops over it — a broadcast, `map`, or a function of its own — behind which the loop is compiled for the array's
# concrete type. Read in place, the plotted values of 490 000 sample points took 45 ms; through a barrier, under 0.3 ms.
function _record_array(arrays::Vector{Pair{String,AbstractVector}}, name::String)
    for (key, array) in arrays
        key == name && return array
    end
    return nothing
end

# ── Level meshes ──────────────────────────────────────────────────────────────

# The levels a level plot draws, of its mesh `records`, in level order, each as `(; id, grid, keep)`: its id, its mesh
# record, and which of its cells are drawn — all of them with `inactive`, the active ones otherwise. `levels` picks
# levels by id; the order they are drawn in is the levels' own, which is what the stacking in 2D relies on. Collected
# into a vector of a type fixed by the records', whatever is picked, as a computed value of a plot must be.
function _drawn_levels(records::Vector{@NamedTuple{field::Field,level::Int,role::Symbol,grid::G}}, levels,
                       inactive::Bool) where {G}
    ids = _picked_levels(levels, [record.level for record in records])
    drawn = @NamedTuple{id::Int, grid::G, keep::BitVector}[]
    for record in records
        record.level in ids || continue
        keep = inactive ? trues(_cell_count(record.grid)) : _cell_array(record.grid, :active) .== 1
        push!(drawn, (; id=record.level, record.grid, keep))
    end
    return drawn
end

# The colour of every corner of every drawn cell, in the order the cells are drawn (`_level_primitives`): the value
# that colours the cell (`_cell_values`), repeated over its 2ᴰ corners. The values are read for every level of the
# records, so that an estimate is checked against the whole discretisation however few of its levels are drawn.
function _cell_colors(model::Model{D}, records, drawn, by) where {D}
    values = Dict(zip((record.level for record in records), _cell_values(model, records, by)))
    return Float64[values[level.id][c] for level in drawn for c in findall(level.keep) for _ in 1:(1<<D)]
end

# The ids of the levels `levels` picks among the `available` ones, in their order: every one for `nothing`, or those
# it names — one id, or any collection of ids — each of which must be available.
function _picked_levels(levels, available::Vector{Int})
    levels === nothing && return available
    ids = levels isa Integer ? (levels,) : levels
    applicable(iterate, ids) && all(id -> id isa Integer, ids) ||
        throw(ArgumentError("levels must be nothing, a level id or a collection of level ids; got $(repr(levels))"))
    all(in(available), ids) ||
        throw(ArgumentError("levels = $(repr(levels)) names a level the drawn space does not have: its levels are " *
                            join(available, ", ")))
    return filter(in(ids), available)
end

# The value that colours each cell of each level, in the records' order: one of the record's integer cell arrays,
# named by a `Symbol`, or the cell's indicator `η_K` in an error estimate. `estimate` stores `η_K` in one array per
# level, in `space.levels` order and of the level's shape, which is the order and the length of the records of a
# single-field model; the shapes are checked against the levels themselves, so an estimate of another discretisation is
# refused rather than read out of order. A cell's `cell_id` is its linear index, so it indexes its level's array.
_cell_values(::Model, records, by::Symbol) = [Float64.(_cell_array(record.grid, by)) for record in records]
function _cell_values(model::Model, records, estimate::ErrorEstimate)
    shape = Dict(level.id => level.mesh.cells for fld in model.problem.fields for level in fld.space.levels)
    shapes = [shape[record.level] for record in records]
    length(estimate.cells) == length(records) && all(k -> size(estimate.cells[k]) == shapes[k], eachindex(records)) ||
        throw(ArgumentError("by: this ErrorEstimate holds levels of $(join(size.(estimate.cells), ", ")) cells, and " *
                            "the drawn levels have $(join(shapes, ", ")); pass the estimate `estimate(model, " *
                            "solution)` returned for this model"))
    return [Float64.(estimate.cells[k][_cell_array(records[k].grid, :cell_id)]) for k in eachindex(records)]
end
function _cell_values(::Model, records, by)
    throw(ArgumentError("by must name a cell array of the level meshes, as a Symbol, or be an ErrorEstimate; got " *
                        repr(by)))
end

# The cell array `name` of a level mesh record, refused with the names it has when it has none of that name.
function _cell_array(grid::SampleGrid, name::Symbol)
    array = _record_array(grid.cell_data, String(name))
    array === nothing || return array
    names = join((":" * key for key in first.(grid.cell_data)), ", ")
    throw(ArgumentError("by = :$name names no cell array of the level meshes; pass one of $names, or an ErrorEstimate"))
end

# The 2D drawing of the levels: one mesh of every drawn cell's quad, whose vertex colours are each cell's value repeated
# over its four corners (`_cell_colors`), and the cell edges as line segments.
#
# The faces are stacked in z, in level order: the `k`-th of `n` drawn levels lies at `z = k − n − 1`, and the mesh
# holds its faces in that order, so a later level lies in front of an earlier one in every backend — one that tests
# depth (GLMakie) draws what is nearer in z in front, and one that paints a 2D mesh's faces in their order (CairoMakie)
# paints the later level's last — and every level lies behind `z = 0`, where a plot drawn into the same axis
# afterwards lands.
#
# The edges cannot rely on that: they are one child plot drawn after the whole mesh, which a backend that paints its
# children in order puts in front of every face. So they are cut geometrically instead: each level draws only the
# parts of its edges that no drawn cell of a later level lies over (`_visible_pieces`), and each edge once
# (`_level_edges`). The edges lie `_EDGE_LIFT` in front of their level for the backends that do test depth.
function _level_primitives(::Val{2}, drawn)
    positions, edges = Point3{Float64}[], Point3{Float64}[]
    faces = GeometryBasics.QuadFace{Int}[]
    lattices = map(_drawn_lattice, drawn)
    for (k, level) in pairs(drawn)
        z = k - length(drawn) - 1
        for c in findall(level.keep)
            corners = level.grid.points[(4c-3):(4c)]
            offset = length(positions)
            append!(positions, (Point3{Float64}(x[1], x[2], z) for x in corners))
            push!(faces, GeometryBasics.QuadFace{Int}(offset + 1, offset + 2, offset + 3, offset + 4))
        end
        later = lattices[(k+1):end]
        for (axis, at, a, b) in _level_edges(lattices[k]), (s, t) in _visible_pieces([(a, b)], axis, at, later)
            ends = axis == 1 ? ((s, at), (t, at)) : ((at, s), (at, t))
            append!(edges, (Point3{Float64}(x..., z + _EDGE_LIFT) for x in ends))
        end
    end
    return GeometryBasics.Mesh(positions, faces), edges
end

# The edges of a level's drawn cells, each once, as `(axis, at, a, b)`: the edge runs along `axis` from `a` to `b` at
# the coordinate `at` on the other axis. Read off the level's lattice (`_drawn_lattice`), whose every edge is one of a
# drawn cell exactly when a cell on either side of it is drawn: each edge comes out once, in lattice order.
function _level_edges(lattice)
    (xs, ys), keep = lattice.breaks, lattice.keep
    drawn(i, j) = checkbounds(Bool, keep, i, j) && keep[i, j]
    edges = Tuple{Int,eltype(xs),eltype(xs),eltype(xs)}[]
    for j in eachindex(ys), i in 1:(length(xs)-1)
        (drawn(i, j - 1) || drawn(i, j)) && push!(edges, (1, ys[j], xs[i], xs[i+1]))
    end
    for i in eachindex(xs), j in 1:(length(ys)-1)
        (drawn(i - 1, j) || drawn(i, j)) && push!(edges, (2, xs[i], ys[j], ys[j+1]))
    end
    return edges
end

# A drawn level as the hiding test reads it: the breaks of its lattice along each axis, and which of its cells are
# drawn, indexed by cell. The records carry every cell of a level, active or not, in `cell_indices` order, so the
# sorted distinct corner coordinates are the breaks — bit-identical where two cells share a face — and the drawn mask
# reshapes onto the lattice they span.
function _drawn_lattice(level)
    breaks = ntuple(d -> sort!(unique!([x[d] for x in level.grid.points])), Val(2))
    return (; breaks, keep=reshape(level.keep, map(b -> length(b) - 1, breaks)))
end

# The parts of the pieces of an edge — `(s, t)` intervals along `axis` at `at` — that no drawn cell of the `later`
# lattices lies over, merged where they meet. A lattice hides a part that lies strictly inside one of its drawn cells:
# each piece is split at the lattice's breaks along `axis`, so that every part lies within one of its cells along the
# edge or outside all of them, and the part's midpoint decides by index lookup. A part on one of the lattice's cell
# boundaries — `at` on a break of the other axis — is inside none of its cells, which are open, so an edge that
# coincides with a later level's edge is kept.
function _visible_pieces(pieces, axis::Int, at, later)
    for lattice in later
        cuts, parts = lattice.breaks[axis], empty(pieces)
        for (s, t) in pieces
            inner = cuts[(searchsortedlast(cuts, s)+1):(searchsortedfirst(cuts, t)-1)]
            for (u, v) in zip([s; inner], [inner; t])
                m = (u + v) / 2
                _drawn_cell_at(lattice, axis == 1 ? (m, at) : (at, m)) || push!(parts, (u, v))
            end
        end
        pieces = parts
    end
    merged = empty(pieces)
    for (u, v) in pieces
        !isempty(merged) && last(merged)[2] == u ? (merged[end] = (first(merged[end]), v)) : push!(merged, (u, v))
    end
    return merged
end

# Whether the point `x` lies strictly inside a drawn cell of `lattice`: along every axis strictly between two
# consecutive breaks, in a cell the mask keeps. A point on a break lies on a cell boundary and inside no cell.
function _drawn_cell_at(lattice, x)
    index = ntuple(Val(2)) do d
        cuts = lattice.breaks[d]
        i = searchsortedlast(cuts, x[d])
        return 1 <= i < length(cuts) && cuts[i] < x[d] ? i : 0
    end
    return all(>(0), index) && lattice.keep[index...]
end

# The 1D drawing of the levels: every drawn cell a segment at the height of its level id, coloured by its value at both
# ends (`_cell_colors`), and a tick across each end of it, `_TICK_HALF_HEIGHT` levels above and below.
function _level_primitives(::Val{1}, drawn)
    segments, ticks = Point2{Float64}[], Point2{Float64}[]
    for level in drawn, c in findall(level.keep)
        a, b, y = level.grid.points[2c-1][1], level.grid.points[2c][1], level.id
        push!(segments, Point2{Float64}(a, y), Point2{Float64}(b, y))
        for x in (a, b)
            push!(ticks, Point2{Float64}(x, y - _TICK_HALF_HEIGHT), Point2{Float64}(x, y + _TICK_HALF_HEIGHT))
        end
    end
    return segments, ticks
end

# The half height of a 1D level plot's cell-end ticks, in levels: well inside the unit spacing of the rows.
const _TICK_HALF_HEIGHT = 0.2

# ── Fields ────────────────────────────────────────────────────────────────────

# The quantity a field plot shows: `by` as given, or for `automatic` the solution's value with a solution and the number
# of levels covering each region without one.
_resolved_by(by, solution) = by === Makie.automatic ? (solution === nothing ? :cover_count : :uh) : by

# The samples a field plot draws, one per sampled field, as `(; name, grid, values, φ)`: the field's name, its record's
# sample grid, the plotted value at every point of it, and the level set there — the record's `level_set` array, which
# the sampler attaches for a field whose space carries a `PhysicalDomain` and for no other — or `nothing`. `by` is
# handed to the sampler as the record's one `point_data` entry, or as none for a built-in array, so the sampler checks,
# evaluates and refuses it as it does `write_vtk`'s entries; the value is read back off the record and reduced to one
# number per point. `subdivisions = automatic` resolves by what the plot draws (`_automatic_subdivisions`).
#
# Collected into a vector of the samples' declared type, which is a model's whatever its fields and whether or not its
# spaces are immersed: Makie keeps the type of a computed value's first result, and refuses a later one of another type.
function _field_samples(solution, model::Model{D,T}, by, component, field, subdivisions, cut_depth, warp) where {D,T}
    request = _field_request(_resolved_by(by, solution), solution)
    _check_one_field_per_space(model, field)
    subdivisions = subdivisions === Makie.automatic ? _automatic_subdivisions(Val(D), warp) : subdivisions
    samples = @NamedTuple{name::Symbol, grid::SampleGrid{D,T,D}, values::Vector{Float64},
                          φ::Union{Nothing,Vector{Float64}}}[]
    for (; field, grid) in _sample_fields(solution, model; field, subdivisions, cut_depth, request.point_data)
        push!(samples, (; field.name, grid, values=_plotted_values(grid, request, component), φ=_level_set(grid)))
    end
    return samples
end

# The level set a sample's drawing is cut at, or `nothing` to draw it whole: `clip = automatic` cuts every sample whose
# field's space carries a `PhysicalDomain`, `true` every sample, refusing one without, and `false` none.
function _clip_at(sample, clip)
    clip === Makie.automatic && return sample.φ
    clip isa Bool || throw(ArgumentError("clip must be true, false or automatic; got $(repr(clip))"))
    if clip && sample.φ === nothing
        throw(ArgumentError("clip = true cuts the plot at ∂Ω, and the space of the field $(sample.name) carries no " *
                            "PhysicalDomain: pass clip = false, or automatic to clip only the fields that have one"))
    end
    return clip ? sample.φ : nothing
end

# ∂Ω as a field plot draws it, with `boundary`, from its samples' level sets (`_boundary`): in 2D at the height of the
# plotted value on a warped surface, and in front of the flat one. Refused when no sample has a level set to draw.
function _field_boundary(::Val{D}, samples, boundary::Bool, warp::Bool) where {D}
    boundary || return _boundary(Val(D), ())
    any(sample -> sample.φ !== nothing, samples) || throw(_no_boundary_error(:fieldplot))
    return _boundary(Val(D), ((s.grid, s.φ, warp ? s.values : _BOUNDARY_Z) for s in samples if s.φ !== nothing))
end

# A record's `level_set` point array, as floating-point numbers, or `nothing` for a record without one.
function _level_set(grid::SampleGrid)
    φ = _record_array(grid.point_data, "level_set")
    return φ === nothing ? nothing : Vector{Float64}(φ)
end

# The refusal of `boundary = true` on a plot of no field with a `PhysicalDomain`, whose boundary is the box's own.
function _no_boundary_error(plot::Symbol)
    return ArgumentError("$plot draws ∂Ω with boundary = true, and no drawn field's space carries a PhysicalDomain: " *
                         "there is no immersed boundary to draw")
end

# The sample cells per region a field plot takes by default, sized by what each drawing shows.
#
#   * A flat 2D plot shows the value as colour alone, and `:degree` — the sampler's own default, `p` sub-cells per axis
#     for a degree-`p` region — resolves it as ParaView's view of the same records does.
#   * A warped 2D plot shows the value as height, lit, and two things `:degree` leaves visible there: facets, and the
#     cracks where regions sampled on different lattices meet, each side of their shared edge linear between its own
#     points. The crack's width falls as the square of the sample cell's size — measured on a bump of height 1.7 under
#     an order-3 overlay over an order-2 base: 0.024 at `:degree`, 0.0068 at 4 sub-cells per region, 0.0017 at 8,
#     0.00042 at 16 — and 8 is where neither shows in a render any more.
#   * A 1D graph drawn over `p` straight segments shows the corners of a degree-`p` polynomial at any `p ≥ 2`; 32 draw
#     an order-8 cell as a smooth curve, and a 1D record of that many points per region costs next to nothing.
_automatic_subdivisions(::Val{2}, warp::Bool) = warp ? 8 : :degree
_automatic_subdivisions(::Val{1}, warp::Bool) = 32

# What the sampler is asked for to show `by`, as `(; name, location, point_data)`: the record array that holds it,
# whether a point or a cell array, and the `point_data` entries that produce it. The built-in arrays are the sampler's
# own — `level_set` with a `PhysicalDomain`, `region_id` and `cover_count` always — and need no entry; `:uh` is the
# sampler's default entry, which reads a solution; a callback is the record's one entry, named `by`.
function _field_request(by::Symbol, solution)
    if by === :uh
        message = "by = :uh plots the solution, and this plot has none: pass it as `fieldplot(solution, model)`, " *
                  "or plot the regions by :cover_count or :region_id"
        solution === nothing && throw(ArgumentError(message))
        return (; name="uh", location=:point, point_data=_default_point_data(solution))
    end
    by === :level_set && return (; name="level_set", location=:point, point_data=NamedTuple())
    if by in (:region_id, :cover_count)
        return (; name=String(by), location=:cell, point_data=NamedTuple())
    end
    throw(ArgumentError("by = :$by names no array a field plot shows; pass :uh, :level_set, :region_id, " *
                        ":cover_count, or a callback (u, context, x, xi) -> value"))
end
_field_request(by::Function, _) = (; name="by", location=:point, point_data=(by=by,))
function _field_request(by, _)
    throw(ArgumentError("by must name an array, as a Symbol, or be a callback (u, context, x, xi) -> value; got " *
                        repr(by)))
end

# A field plot draws one field per space: two fields on one space cover the same cells, and drawn together one would
# hide the other. Fields on different subdomains, those of a coupled model, are drawn side by side.
function _check_one_field_per_space(model::Model, field)
    for fields in _fields_by_space(model, field)
        length(fields) == 1 ||
            throw(ArgumentError("fieldplot draws one field per space, and $(fields[1].name) and $(fields[2].name) " *
                                "share one: pick one with `field`"))
    end
    return nothing
end

# The plotted value at every point of a sample grid: the requested array reduced to one number per value
# (`_plotted_value`), a cell array's value repeated over the cell's own corners, so that the cell is coloured flat. The
# one array a request can find missing is `level_set`, which the sampler attaches only for a `PhysicalDomain`.
function _plotted_values(grid::SampleGrid{D,T,K}, request, component) where {D,T,K}
    array = _record_array(request.location === :point ? grid.point_data : grid.cell_data, request.name)
    array === nothing &&
        throw(ArgumentError("by = :level_set plots the level set φ, and this model's space carries no " *
                            "PhysicalDomain"))
    values = map(v -> _plotted_value(v, component), array)
    return request.location === :point ? values : repeat(values; inner=1 << K)
end

# One sampled value as the number plotted: a real number as it is; a vector's magnitude, or its component `component`,
# which must be one of its components. A scalar has no component to pick.
_plotted_value(v::Real, ::Nothing) = Float64(v)
_plotted_value(v::SVector, ::Nothing) = Float64(sqrt(sum(abs2, v)))
function _plotted_value(v::SVector{N}, component::Integer) where {N}
    1 <= component <= N || throw(ArgumentError("component = $component picks no component of a $N-component value"))
    return Float64(v[component])
end
function _plotted_value(::Real, component)
    throw(ArgumentError("component = $(repr(component)) picks a component of a vector value, and this value is a " *
                        "scalar: pass component = nothing"))
end
function _plotted_value(::SVector, component)
    throw(ArgumentError("component must be nothing or the index of a component; got $(repr(component))"))
end

# The height a flat field plot draws its surface at, behind `z = 0` as a level plot's levels are, so that a plot drawn
# into the same axis afterwards lies in front of it.
const _FIELD_Z = -1.0

# How far in front of its flat surface, or of its level, a plot draws the edges on it, for the backends that test
# depth: half the spacing of a level plot's levels, so that the edges of a level lie between it and the next.
const _EDGE_LIFT = 0.5

# The height every flat plot draws ∂Ω at: in front of a field's surface and edges and of every level of a level plot,
# and behind `z = 0`, where a plot drawn into the same axis afterwards lands.
const _BOUNDARY_Z = -0.25

# The 2D surface of a field: the triangles of its samples' cells (`_surface!`), each sample cut where `clip` says
# (`_clip_at`), coloured by the plotted value at each corner, at the height of the value with `warp` and flat at
# `_FIELD_Z` without. A record's cells own their corners, so a value that jumps between cells, or between regions, is
# drawn as a jump rather than smeared across it.
function _surface(samples, clip, warp::Bool)
    positions, colors, faces = Point3{Float64}[], Float64[], GeometryBasics.TriangleFace{Int}[]
    for sample in samples
        _surface!(positions, colors, faces, sample.grid, sample.values, _clip_at(sample, clip), warp)
    end
    return GeometryBasics.Mesh(positions, faces), colors
end

# The triangles of a record's sample cells, appended to a mesh's `positions`, `colors` and `faces`. Each cell is split
# as Makie splits a quad, into (1, 2, 3) and (1, 3, 4) (`_cell_triangles`), so that a clipped and an unclipped plot
# share one face type and colour alike. Makie colours a triangle linearly between its corners, where ParaView
# interpolates a quad bilinearly, so the colour bends along the cell's diagonal; a contrasting colour map can show it as
# streaks on cells too coarse for the field, which `subdivisions` resolves. A fan of four triangles around a centre
# sampled exactly was measured as the alternative and rejected: its four gradients meet at the centre, which reads as a
# star at every cell, under the default colour map as well.
#
# Cut at a level set `φ` — uncut when `φ = nothing` — a record keeps of each triangle the part where the linear
# interpolant of φ at its corners is ≤ 0: the whole triangle, nothing, or a polygon of three or four vertices cut along
# the interpolant's zero line, which is fanned from its first vertex (one pass of the clipper of I. E. Sutherland and
# G. W. Hodgman, "Reentrant polygon clipping", Commun. ACM 17 (1974) 32–42, doi:10.1145/360767.360802, against one
# half-plane). A clip vertex takes the value and the height interpolated linearly along its edge, which is what the
# renderer interpolates across the triangle, so a clipped plot draws inside Ω exactly what an unclipped one does. A
# corner joins the mesh when a triangle first uses it, so a cell outside Ω adds nothing — not even to the axis
# limits.
function _surface!(positions, colors, faces, grid::SampleGrid{2}, v, φ, warp::Bool)
    index = zeros(Int, length(grid.points))
    function corner(k)
        if index[k] == 0
            push!(positions, Point3{Float64}(grid.points[k]..., warp ? v[k] : _FIELD_Z))
            push!(colors, v[k])
            index[k] = length(positions)
        end
        return index[k]
    end
    function cut(a, b)
        i, j, t = _crossing(grid.points, φ, a, b)
        x, value = _lerp(grid.points, i, j, t), _lerp(v, i, j, t)
        push!(positions, Point3{Float64}(x..., warp ? value : _FIELD_Z))
        push!(colors, value)
        return length(positions)
    end
    for c in 1:_cell_count(grid), triangle in _cell_triangles(c)
        if φ === nothing || all(k -> φ[k] <= 0, triangle)
            push!(faces, GeometryBasics.TriangleFace{Int}(map(corner, triangle)...))
            continue
        end
        any(k -> φ[k] < 0, triangle) || continue                   # φ ≥ 0 throughout: Ω meets it in no area
        polygon = Int[]
        for (a, b) in _triangle_edges(triangle)
            φ[a] <= 0 && push!(polygon, corner(a))
            _crosses(φ[a], φ[b]) && push!(polygon, cut(a, b))
        end
        for j in 2:(length(polygon)-1)
            push!(faces, GeometryBasics.TriangleFace{Int}(polygon[1], polygon[j], polygon[j+1]))
        end
    end
    return faces
end

# The two triangles of sample cell `c`, as indices of its points, which come in the face-cyclic order of the record:
# (1, 2, 3) and (1, 3, 4), Makie's own split of a quad.
_cell_triangles(c::Int) = ((4c - 3, 4c - 2, 4c - 1), (4c - 3, 4c - 1, 4c))

# The edges of a triangle, each from one vertex to the next.
_triangle_edges((a, b, c)) = ((a, b), (b, c), (c, a))

# Whether the linear interpolant of φ between two values vanishes strictly between them: one negative, one positive. A
# value of exactly zero is a vertex of the zero set, which its own vertex accounts for.
_crosses(φa, φb) = φa < 0 < φb || φb < 0 < φa

# Where φ's linear interpolant along the segment between points `a` and `b` of `points` vanishes, as `(i, j, t)`: the
# point `_lerp(points, i, j, t)`. The ends are taken in a fixed order, the lexicographically lower point first, so that
# two cells meeting along an edge, whose points there are bit-identical, find bit-identical crossings on it, and a
# clipped surface and a contour close up between them.
function _crossing(points, φ, a::Int, b::Int)
    i, j = Tuple(points[a]) <= Tuple(points[b]) ? (a, b) : (b, a)
    return i, j, φ[i] / (φ[i] - φ[j])
end
_lerp(values, i::Int, j::Int, t) = values[i] + t * (values[j] - values[i])

# The part of the segment between points `a` and `b` of `points` that a drawing cut at the level set `φ` keeps, as its
# two ends, or `nothing`: all of it uncut (`φ = nothing`) or with φ ≤ 0 at both ends; none of it with φ > 0 at either
# end and no crossing; and otherwise the part between the end where φ < 0 and the crossing.
function _clipped_segment(points, φ, a::Int, b::Int)
    (φ === nothing || φ[a] <= 0 && φ[b] <= 0) && return (points[a], points[b])
    _crosses(φ[a], φ[b]) || return nothing
    x = _lerp(points, _crossing(points, φ, a, b)...)
    return φ[a] < 0 ? (points[a], x) : (x, points[b])
end

# ∂Ω on a record's sample cells, appended to `segments`: in each triangle of each cell (`_cell_triangles`, the split
# the surface is drawn with), the segment along which φ's linear interpolant vanishes — between the two crossings on its
# edges, or a vertex where φ is zero and the crossing on the opposite edge, or the triangle's edge whose two vertices
# both have φ = 0. That edge bounds a second triangle, across it, which would draw it again; it is drawn by the triangle
# whose third vertex has φ < 0, the one in Ω, so a plane through sample points is drawn once along it. A triangle the
# zero set only touches, at one vertex, has no segment. `heights` gives the segment its height: one `z` for a flat plot,
# or a point array, interpolated as the surface is, for a warped one.
function _contour!(segments, grid::SampleGrid{2}, φ, heights)
    height(k) = heights isa Real ? heights : heights[k]
    for c in 1:_cell_count(grid), triangle in _cell_triangles(c)
        ends = Point3{Float64}[]
        for (a, b) in _triangle_edges(triangle)
            φ[a] == 0 && push!(ends, Point3{Float64}(grid.points[a]..., height(a)))
            if _crosses(φ[a], φ[b])
                i, j, t = _crossing(grid.points, φ, a, b)
                z = heights isa Real ? heights : _lerp(heights, i, j, t)
                push!(ends, Point3{Float64}(_lerp(grid.points, i, j, t)..., z))
            end
        end
        length(ends) == 2 || continue
        count(k -> φ[k] == 0, triangle) < 2 || any(k -> φ[k] < 0, triangle) || continue
        append!(segments, ends)
    end
    return segments
end

# The roots of φ's linear interpolant on a 1D record's sample cells, appended to `roots`: the crossing inside each cell,
# and each point where φ is zero.
function _roots!(roots, grid::SampleGrid{1}, φ)
    for c in 1:_cell_count(grid)
        a, b = 2c - 1, 2c
        φ[a] == 0 && push!(roots, grid.points[a][1])
        φ[b] == 0 && push!(roots, grid.points[b][1])
        _crosses(φ[a], φ[b]) && push!(roots, _lerp(grid.points, _crossing(grid.points, φ, a, b)...)[1])
    end
    return roots
end

# ∂Ω on sample grids, from the level set `φ` at each grid's points, as `(grid, φ, heights)` triples: in 2D the segments
# of its contour (`_contour!`) at `heights`, in 1D its roots (`_roots!`), in order and each once.
function _boundary(::Val{2}, pieces)
    contour = Point3{Float64}[]
    for (grid, φ, heights) in pieces
        _contour!(contour, grid, φ, heights)
    end
    return contour
end
function _boundary(::Val{1}, pieces)
    roots = Float64[]
    for (grid, φ, _) in pieces
        _roots!(roots, grid, φ)
    end
    return sort!(unique!(roots))
end

# The ∂Ω a level or a quadrature plot draws, sampled for the purpose: nothing without `boundary`; with it, ∂Ω on the
# sample cells of each drawn space that carries a `PhysicalDomain`, sampled once through its first field
# (`_fields_by_space`) at the sampler's defaults — `:degree`, `cut_depth = 3` — which a model needs no solution for,
# and in 2D at `_BOUNDARY_Z`.
function _model_boundary(::Val{D}, model::Model, field, boundary::Bool, plot::Symbol) where {D}
    boundary || return _boundary(Val(D), ())
    immersed = [first(fields) for fields in _fields_by_space(model, field) if first(fields).space.physical !== nothing]
    isempty(immersed) && throw(_no_boundary_error(plot))
    grids = (only(_sample_fields(nothing, model; field=fld)).grid for fld in immersed)
    return _boundary(Val(D), ((grid, _level_set(grid), _BOUNDARY_Z) for grid in grids))
end

# The edges of the regions of a field's samples, as segments `_EDGE_LIFT` in front of the flat surface, each sample cut
# where `clip` says (`_clip_at`).
function _region_edges(samples, clip)
    edges = Point3{Float64}[]
    for sample in samples
        _region_edges!(edges, sample.grid, _clip_at(sample, clip))
    end
    return edges
end

# The edges of the regions of a sample grid, appended to `edges`: of every sample cell, the edges that lie on the
# boundary of its region's box, and of those only what a drawing cut at the level set `φ` keeps (`_clipped_segment`). A
# record lists a region's cells together (its `region_id` array does not decrease), so each region is one run of cells
# and its box is the hull of their corners; a cell's corner coordinates are those of its lattice box, so an edge on the
# region's box compares equal to it. Regions tile a field's subdomain without overlapping, so a region edge crosses no
# cell, which is what lets the edges be drawn after the surface, in CairoMakie's painting order, without covering any
# of it.
function _region_edges!(edges, grid::SampleGrid{2}, φ)
    return _region_edges!(edges, grid.points, _record_array(grid.cell_data, "region_id"), φ)
end
function _region_edges!(edges, points, regions::AbstractVector, φ)            # behind the barrier on `regions`' type
    z, run = _FIELD_Z + _EDGE_LIFT, 1
    while run <= length(regions)
        cells = run:(something(findnext(!=(regions[run]), regions, run), length(regions)+1)-1)
        corners = points[(4first(cells)-3):(4last(cells))]
        lo, hi = reduce((a, b) -> min.(a, b), corners), reduce((a, b) -> max.(a, b), corners)
        for c in cells
            ll, lr, ur, ul = 4c - 3, 4c - 2, 4c - 1, 4c
            sides = ((ll, lr, points[ll][2] == lo[2]), (ul, ur, points[ul][2] == hi[2]),
                     (ll, ul, points[ll][1] == lo[1]), (lr, ur, points[lr][1] == hi[1]))
            for (a, b, on) in sides
                piece = on ? _clipped_segment(points, φ, a, b) : nothing
                piece === nothing || push!(edges, Point3{Float64}(piece[1]..., z), Point3{Float64}(piece[2]..., z))
            end
        end
        run = last(cells) + 1
    end
    return edges
end

# The 1D drawing of a field: its graph over every sample cell, a segment between the values at the cell's two ends —
# the cell's own two points, so a value that jumps between cells is drawn as a jump — and of a sample cut where `clip`
# says (`_clip_at`) only the part where φ's linear interpolant is ≤ 0, cut at its root with the value interpolated
# there. The graph takes the plot's one `color`; the value it would colour by is the height already.
function _graph(samples, clip)
    segments = Point2{Float64}[]
    for sample in samples
        (; grid, values) = sample
        φ, graph = _clip_at(sample, clip), [SVector(x[1], values[k]) for (k, x) in pairs(grid.points)]
        for c in 1:_cell_count(grid)
            piece = _clipped_segment(graph, φ, 2c - 1, 2c)
            piece === nothing || push!(segments, Point2{Float64}(piece[1]...), Point2{Float64}(piece[2]...))
        end
    end
    return segments
end

# ── Quadrature cloud ──────────────────────────────────────────────────────────

# The points of a quadrature plot: every point of every covering-level signature, in the records' order, at its
# position in 2D and at its position against its weight in 1D.
function _cloud_positions(::Val{D}, records) where {D}
    positions = Point2{Float64}[]
    for (; grid) in records
        _cloud_positions!(positions, Val(D), grid.points, _record_array(grid.point_data, "weight"))
    end
    return positions
end

# The value that colours each point of a quadrature plot, in the order of `_cloud_positions`: its weight, or the number
# of levels in its signature, which is the number of levels covering its region.
function _cloud_colors(records, by)
    by in (:weight, :cover_count) || throw(ArgumentError("by must be :weight or :cover_count; got $(repr(by))"))
    colors = Float64[]
    for (; signature, grid) in records
        weights = _record_array(grid.point_data, "weight")
        append!(colors, by === :weight ? weights : Iterators.repeated(length(signature), length(weights)))
    end
    return colors
end

# The points of one cloud record, appended to `positions`: in 2D at their positions, in 1D at their positions against
# their weights. Behind the barrier on the weights' type (`_record_array`).
function _cloud_positions!(positions, ::Val{D}, points, weights::AbstractVector) where {D}
    for (x, w) in zip(points, weights)
        push!(positions, D == 1 ? Point2{Float64}(x[1], w) : Point2{Float64}(x[1], x[2]))
    end
    return positions
end

end
