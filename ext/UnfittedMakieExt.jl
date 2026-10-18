"""
    UnfittedMakieExt

Package extension that plots Unfitted's discretisations with [Makie](https://docs.makie.org). Loading Makie — through
any of its backends, CairoMakie, GLMakie or WGLMakie — next to `Unfitted` installs the recipes below; without it the
entry points, which `src/stubs.jl` declares and documents, raise a `MethodError` whose hint says to load a backend.

The extension samples nothing itself. `src/postprocessing.jl` samples the meshes of the levels and the quadrature-point
cloud into consumer-neutral records — `Unfitted.SampleGrid`, whose contract heads that file — and this extension draws
them. Everything Makie-specific is decided here and nowhere else:

  - the primitives: a 2D level mesh as a `GeometryBasics.Mesh` of quads, a 1D level mesh and the cell edges as line
    segments, the quadrature cloud as a scatter;
  - the colour of a cell: its record value repeated over the cell's own corners, which no other cell shares, so that
    the colour is flat and exact rather than interpolated across a neighbour;
  - the stacking of the levels, and the axis each plot asks for.

Provided pieces:

  - `Unfitted.levelplot(model; …)` and `Unfitted.levelplot!(…)` — the recipe `LevelPlot`;
  - `Unfitted.quadratureplot(model; …)` and `Unfitted.quadratureplot!(…)` — the recipe `QuadraturePlot`.

Both recipes take the model as their one argument and need no solution: they draw the discretisation.
"""
module UnfittedMakieExt

using Unfitted
using Unfitted: Model, ErrorEstimate, SampleGrid, _sample_levels, _sample_quadrature, _cell_count
import Unfitted: levelplot, levelplot!, quadratureplot, quadratureplot!
import Makie
using Makie: GeometryBasics, Point2, Point3

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
    "Whether the colours are categories: `true`, `false`, or `automatic` for `:cover_count` alone. Settled at creation."
    categorical = Makie.automatic
    Makie.mixin_colormap_attributes()...
    Makie.mixin_generic_plot_attributes()...
end

# Each recipe derives its primitives from its inputs in one computation, so a change of the model or of an attribute —
# a new model in an `Observable`, another `by` — recomputes them together, and draws them as children that read the
# colour mapping from the recipe's own attributes, the colour map through `_effective_colormap`. The dimension check
# comes first, before anything is sampled.
function Makie.plot!(p::LevelPlot{<:Tuple{<:Model{D}}}) where {D}
    _check_dimension(Val(D), :levelplot, :write_vtk)
    inputs = [:discretisation, :field, :by, :levels, :inactive]
    map!(p.attributes, inputs, [:cell_geometry, :cell_color, :cell_edges]) do model, field, by, levels, inactive
        return _level_primitives(Val(D), _drawn_levels(model, field, by, levels, inactive))
    end
    _draw_cells!(p, Val(D), _colormap_node!(p, :cell_colormap))
    Makie.linesegments!(p, Makie.shared_attributes(p, Makie.LineSegments), p.cell_edges; color=p.strokecolor,
                        linewidth=p.strokewidth)
    return p
end

# The coloured cells, the first child of a level plot: a mesh of quads in 2D, segments in 1D, coloured through
# `colormap`, the node `_colormap_node!` chose.
function _draw_cells!(p::LevelPlot, ::Val{2}, colormap)
    return Makie.mesh!(p, Makie.shared_attributes(p, Makie.Mesh), p.cell_geometry; color=p.cell_color, colormap,
                       shading=Makie.NoShading)
end
function _draw_cells!(p::LevelPlot, ::Val{1}, colormap)
    return Makie.linesegments!(p, Makie.shared_attributes(p, Makie.LineSegments), p.cell_geometry; color=p.cell_color,
                               colormap)
end

function Makie.plot!(p::QuadraturePlot{<:Tuple{<:Model{D}}}) where {D}
    _check_dimension(Val(D), :quadratureplot, :write_quadrature_vtm)
    map!(p.attributes, [:discretisation, :by], [:point_positions, :point_color]) do model, by
        return _quadrature_primitives(Val(D), _sample_quadrature(model), by)
    end
    Makie.scatter!(p, Makie.shared_attributes(p, Makie.Scatter), p.point_positions; color=p.point_color,
                   colormap=_colormap_node!(p, :point_colormap))
    return p
end

# The colour map node a plot's coloured child reads: the plot's own `colormap`, or, for a categorical plot, that map
# wrapped in `Makie.Categorical` — one colour per distinct value, and a colour bar with one block and one tick per
# value — in the node `name`. Wrapping the plot's map rather than replacing it keeps the theme's colour map and one the
# user passes; a map that is categorical already is kept as it is.
#
# The choice is settled once, from `categorical` and `by` as the plot is created: Makie colours a categorical and a
# continuous map through different computations, and neither a drawn plot nor a `Colorbar` reading it can switch from
# one to the other. A later `by` or `colormap` is drawn within the choice made; a later `categorical` changes nothing.
function _colormap_node!(p, name::Symbol)
    _is_categorical(p.categorical[], p.by[]) || return p.colormap
    map!(map -> map isa Makie.Categorical ? map : Makie.Categorical(map), p.attributes, :colormap, name)
    return getproperty(p, name)
end

# Whether a plot is categorical: as `categorical` says, and for `automatic` when `by` names one of the quantities
# `_DISCRETE` lists.
function _is_categorical(categorical, by)
    categorical === Makie.automatic && return by isa Symbol && by in _DISCRETE
    categorical isa Bool && return categorical
    throw(ArgumentError("categorical must be true, false or automatic; got $(repr(categorical))"))
end

# The quantities that take a few discrete values, which a categorical colour map shows a block each: a cell's order,
# level, role, activity and coverage, and the number of levels over a quadrature point. The dof counts and the cell
# index take too many values for a block each to read, and an error estimate and the weights vary continuously.
const _DISCRETE = (:order_max, :level_id, :role_id, :active, :covered, :cover_count)

# Plots draw one- and two-dimensional models. A three-dimensional one has no view a plot could show without slicing or
# clipping it, which ParaView does on the VTK writer's output, so the refusal names the writer.
function _check_dimension(::Val{D}, plot::Symbol, writer::Symbol) where {D}
    D <= 2 || throw(ArgumentError("$plot draws one- and two-dimensional models, and this one is $D-dimensional: " *
                                  "write it with `$writer` and look at it in ParaView"))
    return nothing
end

# ── Axes and colour bars ──────────────────────────────────────────────────────

# A plot of a model draws in a 2D axis in either dimension: in 2D a model is its own plane, and in 1D the second axis
# carries the level (`levelplot`) or the weight (`quadratureplot`). A 2D domain is drawn to scale.
Makie.preferred_axis_type(::Union{LevelPlot,QuadraturePlot}) = Makie.Axis
function Makie.preferred_axis_attributes(::Type{Makie.Axis}, ::Union{LevelPlot,QuadraturePlot}, ::Model{2})
    return (; aspect=Makie.DataAspect())
end
function Makie.preferred_axis_attributes(::Type{Makie.Axis}, plot::LevelPlot, model::Model{1})
    return (; ylabel="level", yticks=_drawn_level_ids(model, plot.field[], plot.levels[]))
end
Makie.preferred_axis_attributes(::Type{Makie.Axis}, ::QuadraturePlot, ::Model{1}) = (; ylabel="weight")

# The ids of the levels a 1D level plot draws, which are its rows: the levels of the drawn fields' spaces, as
# `_sample_levels` meshes them, picked by `levels`. Read off the spaces rather than off the meshes, so that the axis
# samples nothing; a `levels` the plot cannot use is left to the plot's own refusal.
function _drawn_level_ids(model::Model, field, levels)
    ids = Int[]
    for fld in Unfitted._sampled_fields(model, field), level in fld.space.levels
        level.id in ids || push!(ids, level.id)
    end
    picked = levels isa Integer ? (levels,) : levels
    picked isa Union{Tuple,AbstractVector} && filter!(in(picked), ids)
    return sort!(ids)
end

# A `Colorbar` reads the colour mapping off the child that carries the colours, which each recipe draws first; the
# edges of a level plot are drawn in one solid colour and have none.
Makie.extract_colormap(p::Union{LevelPlot,QuadraturePlot}) = Makie.extract_colormap(first(p.plots))

# ── Level meshes ──────────────────────────────────────────────────────────────

# The levels a level plot draws, in level order, each as `(; level, grid, values, keep)`: its id, its mesh record, the
# value that colours each of its cells (`by`), and which of its cells are drawn — all of them with `inactive`, the
# active ones otherwise. `levels` picks levels by id; the order they are drawn in is the levels' own, which is what
# the stacking in 2D relies on.
function _drawn_levels(model::Model{D,T}, field, by, levels, inactive::Bool) where {D,T}
    records = _sample_levels(model; field)
    values = _cell_values(model, records, by)
    ids = _drawn_ids(records, levels)
    drawn = @NamedTuple{level::Int, grid::SampleGrid{D,T,D}, values::Vector{Float64}, keep::BitVector}[]
    for (record, v) in zip(records, values)
        record.level in ids || continue
        keep = inactive ? trues(length(v)) : _cell_array(record.grid, :active) .== 1
        push!(drawn, (; record.level, record.grid, values=v, keep))
    end
    return drawn
end

# The ids of the levels to draw: every level's, or those `levels` names, each of which must be a level of the records.
_drawn_ids(records, ::Nothing) = [record.level for record in records]
function _drawn_ids(records, levels)
    ids = levels isa Integer ? (levels,) : levels
    available = [record.level for record in records]
    ids isa Union{Tuple,AbstractVector} && all(id -> id isa Integer, ids) ||
        throw(ArgumentError("levels must be nothing, a level id or a collection of level ids; got $(repr(levels))"))
    all(in(available), ids) ||
        throw(ArgumentError("levels = $(repr(levels)) names a level the drawn space does not have: its levels are " *
                            join(available, ", ")))
    return ids
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
    return [Float64[estimate.cells[k][i] for i in _cell_array(records[k].grid, :cell_id)] for k in eachindex(records)]
end
function _cell_values(::Model, records, by)
    throw(ArgumentError("by must name a cell array of the level meshes, as a Symbol, or be an ErrorEstimate; got " *
                        repr(by)))
end

# The cell array `name` of a level mesh record, refused with the names it has when it has none of that name.
function _cell_array(grid::SampleGrid, name::Symbol)
    for (key, array) in grid.cell_data
        key == String(name) && return array
    end
    names = join((":" * key for key in first.(grid.cell_data)), ", ")
    throw(ArgumentError("by = :$name names no cell array of the level meshes; pass one of $names, or an ErrorEstimate"))
end

# The 2D drawing of the levels: one mesh of every drawn cell's quad, its vertex colours each cell's value repeated over
# the cell's four corners, and the cell edges as line segments.
#
# The faces are stacked in z, in level order: the `k`-th of `n` drawn levels lies at `z = k − n − 1`, so a later level
# lies in front of an earlier one — a 2D axis draws what is nearer in z in front, and CairoMakie depth-sorts the
# triangles of a mesh — and every level lies behind `z = 0`, where a plot drawn into the same axis afterwards lands.
#
# The edges cannot rely on that: they are one child plot drawn after the whole mesh, which a backend that paints its
# children in order (CairoMakie) puts in front of every face. So they are cut geometrically instead: each level draws
# only the parts of its edges that no drawn cell of a later level lies over (`_visible_pieces`), and each edge once
# (`_level_edges`). The edges keep a level's height plus `½` for the backends that do test depth.
function _level_primitives(::Val{2}, drawn)
    positions, colors, edges = Point3{Float64}[], Float64[], Point3{Float64}[]
    faces = GeometryBasics.QuadFace{Int}[]
    lattices = map(_drawn_lattice, drawn)
    for (k, level) in pairs(drawn)
        z = k - length(drawn) - 1
        for c in findall(level.keep)
            corners = level.grid.points[(4c-3):(4c)]
            offset = length(positions)
            append!(positions, (Point3{Float64}(x[1], x[2], z) for x in corners))
            push!(faces, GeometryBasics.QuadFace{Int}(offset + 1, offset + 2, offset + 3, offset + 4))
            append!(colors, Iterators.repeated(level.values[c], 4))
        end
        later = lattices[(k+1):end]
        for (axis, at, a, b) in _level_edges(level), (s, t) in _visible_pieces([(a, b)], axis, at, later)
            ends = axis == 1 ? ((s, at), (t, at)) : ((at, s), (at, t))
            append!(edges, (Point3{Float64}(x..., z + 0.5) for x in ends))
        end
    end
    return GeometryBasics.Mesh(positions, faces), colors, edges
end

# The edges of a level's drawn cells, each once, as `(axis, at, a, b)`: the edge runs along `axis` from `a` to `b` at
# the coordinate `at` on the other axis. A quad's corners come in the record's face-cyclic order — lower left, lower
# right, upper right, upper left — so its edges join consecutive corners. Every corner coordinate is a `lower` or
# `upper` of a box of the level's own lattice, so two cells that share an edge give it bit-identical coordinates, and
# the set of these tuples holds a shared edge once. Sorted, so that the edges come out in an order of their own rather
# than the set's.
function _level_edges(level)
    T = eltype(eltype(level.grid.points))
    edges = Set{Tuple{Int,T,T,T}}()
    for c in findall(level.keep)
        ll, lr, ur, ul = level.grid.points[(4c-3):(4c)]
        push!(edges, (1, ll[2], ll[1], lr[1]), (1, ul[2], ul[1], ur[1]), (2, ll[1], ll[2], ul[2]),
              (2, lr[1], lr[2], ur[2]))
    end
    return sort!(collect(edges))
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
# ends, and a tick across each end of it, `_TICK` levels above and below.
function _level_primitives(::Val{1}, drawn)
    segments, colors, ticks = Point2{Float64}[], Float64[], Point2{Float64}[]
    for level in drawn, c in findall(level.keep)
        a, b, y = level.grid.points[2c-1][1], level.grid.points[2c][1], level.level
        push!(segments, Point2{Float64}(a, y), Point2{Float64}(b, y))
        append!(colors, (level.values[c], level.values[c]))
        for x in (a, b)
            push!(ticks, Point2{Float64}(x, y - _TICK), Point2{Float64}(x, y + _TICK))
        end
    end
    return segments, colors, ticks
end

# The half height of a 1D level plot's cell-end ticks, in levels: well inside the unit spacing of the rows.
const _TICK = 0.2

# ── Quadrature cloud ──────────────────────────────────────────────────────────

# The points of a quadrature plot and the value that colours each: every point of every covering-level signature, in
# the records' order, at its position in 2D and at its position against its weight in 1D, coloured by its weight or by
# the number of levels in its signature, which is the number of levels covering its region.
function _quadrature_primitives(::Val{D}, entries, by) where {D}
    by in (:weight, :cover_count) || throw(ArgumentError("by must be :weight or :cover_count; got $(repr(by))"))
    positions, colors = Point2{Float64}[], Float64[]
    for (; signature, grid) in entries
        weights = last(only(grid.point_data))
        for (x, w) in zip(grid.points, weights)
            push!(positions, D == 1 ? Point2{Float64}(x[1], w) : Point2{Float64}(x[1], x[2]))
            push!(colors, by === :weight ? w : length(signature))
        end
    end
    return positions, colors
end

end
