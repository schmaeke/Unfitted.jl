# ── Types ─────────────────────────────────────────────────────────────────────

"""
    Parent{D,T}(level, cell, parent_box, ξ0, dξ)
    Parent(level::Level, cell, lower, upper)

One parent cell of an integration [`Region`](@ref): a cell whose basis functions are evaluated on the region, with the
affine map from the region's frame into the cell's own reference frame, so the assembly hot loop re-derives neither.

  - `level::Int` — id of the parent level.
  - `cell::CartesianIndex{D}` — parent cell index inside that level's mesh.
  - `parent_box::AxisBox{D,T}` — the parent cell's physical box. Read for the per-axis gradient scale
    `2 / edge_lengths` that pulls basis gradients back to physical coordinates, and by the `write_vtk` callbacks
    (`context.parents[i].parent_box`).
  - `ξ0::SVector{D,T}`, `dξ::SVector{D,T}` — the map from a region-frame point `p ∈ [−1, 1]ᴰ` to the cell's reference
    coordinate, `ξ = ξ0 + dξ ∘ p` (`∘` the per-axis product). On an axis where the region is degenerate
    (`lower[d] == upper[d]`), `dξ = 0` and `ξ0` is the region's own coordinate there.

The four-argument constructor builds the map from the region's corners `lower` / `upper`: with `l` and `u` the corners
in the cell's reference frame ([`physical_to_reference`](@ref)), `ξ0 = (l + u) / 2` and `dξ = (u − l) / 2`. Halving is
exact in floating point, so `ξ0 + dξ ∘ p` is bit for bit [`reference_to_physical`](@ref) of the box `l`–`u` at `p` —
the parent coordinate a volume region has always produced.
"""
struct Parent{D,T<:Real}
    level::Int
    cell::CartesianIndex{D}
    parent_box::AxisBox{D,T}
    ξ0::SVector{D,T}
    dξ::SVector{D,T}
end

function Parent(level, cell::CartesianIndex{D}, lower::SVector{D,T}, upper::SVector{D,T}) where {D,T}
    parent_box = cell_box(level.mesh, cell)
    l, u = physical_to_reference(parent_box, lower), physical_to_reference(parent_box, upper)
    return Parent{D,T}(level.id, cell, parent_box, (l + u) / 2, (u - l) / 2)
end

"""
    Region{D,T,N,S}(lower, upper, parents, split, fields, kind, points, weights, normal, sides, residual)
    Region(lower, upper, parents, kind, points, weights; split, fields, normal, sides, residual)

One admissible integration region: a piece of the domain, of a boundary facet or of an immersed simplex that lies
inside a single cell of every parent level, so that any product of the parents' basis functions is smooth (`C^∞`) on it
and the region's rule integrates it to the rule's own exactness. Every region the assembly kernel walks is one of
these, and the aliases name its three shapes:

  - `VolumeRegion{D,T} = Region{D,T,Nothing,Nothing}` — a box of an [`IntegrationPlan`](@ref), and either side of a
    box of the L² transfer;
  - [`FacetRegion`](@ref)`{D,T}` — a piece of a grid-aligned boundary facet: `normal` is the facet's outward unit normal
    and `sides` its codim-`K` identifier;
  - [`SimplexRegion`](@ref)`{D,T}` — a sub-cell of an immersed [`BoundaryMesh`](@ref) or of a coupling interface:
    `normal` is the sub-cell's unit normal and `sides` is `nothing`.

**The frame.** A region's `lower` / `upper` corners are both its extent and its frame: a frame point `p ∈ [−1, 1]ᴰ` is
the physical point

    x = (lower + upper) / 2 + (upper − lower) ∘ p / 2,

the expression of [`reference_to_physical`](@ref). An axis may be degenerate (`lower[d] == upper[d]`: a facet's pinned
axis, an axis along which a simplex is flat); a rule then has `p[d] = 0` there and `x[d]` is the region's coordinate.
`points` / `weights` are the rule in that frame, and the physical weight of point `k` is `weights[k] · J` with
`J = ∏ (upper[d] − lower[d]) / 2` over the non-degenerate axes — `vol / 2ᴰ` on a volume box, 1 on a point.

Fields:

  - `lower::SVector{D,T}`, `upper::SVector{D,T}` — the region's corners. A volume region's are its box (`_region_box`
    builds the [`AxisBox`](@ref)); a facet region's are the facet's coordinate on every pinned axis and its merged
    interval on every free one; a simplex region's are its sub-cell's bounding box.
  - `parents::Vector{Parent{D,T}}` — the cells evaluated on the region, at most one per level and side, each with its
    map from the region's frame ([`Parent`](@ref)). The set is constant over the region.
  - `split::Int` — `parents[1:split]` are side `a` of an interface region and the rest side `b`; `length(parents)` on
    every one-sided region.
  - `fields::NTuple{2,Int}` — `(0, 0)` on a one-sided region, which belongs to the fields whose level-id block holds its
    parents' levels; `(field_a, field_b)`, the coupled fields' global indices, on an interface region.
  - `kind::Symbol` — which rule the region carries, a closed vocabulary (`CONTRIBUTING.md`, "Region quadrature kinds"):
    `:full` (tensor Gauss over the whole region: it lies in Ω, or no [`PhysicalDomain`](@ref) is attached),
    `:fictitious` (a facet region outside Ω under `α = 0`, kept with an empty rule), `:fictitious_alpha` (outside Ω
    under `α > 0`: the α-scaled tensor rule), `:cut_fitted` (∂Ω crosses it: the moment-fit rule, `src/fcm.jl`),
    `:cut_fallback` (the fit failed, so the raw Saye volume rule the moments were summed from — correct and
    non-negative, but with 50–200× the points), `:cut_custom` (the domain's own `cut_quadrature` supplied the rule),
    `:cut_failed` (`Ω ∩ region` carries no rule at all, under `α = 0`: an empty rule) and `:cut_alpha_failed` (the same
    under `α > 0`: the α-scaled tensor rule alone, so the cell's dofs stay α-stabilised), and on simplices `:surface`
    and `:interface`. Under `α > 0` every other cut kind carries `(1 − α)·rule ∪ α·tensor`, rule first.
  - `points::Vector{SVector{D,T}}`, `weights::Vector{T}` — the rule in the region's frame (see above). The `:full` and
    the α-scaled kinds alias one cached array per distinct rule, so they cost no allocation per region.
  - `normal::N` — `nothing` on a volume region, else the region's constant unit normal.
  - `sides::S` — `nothing`, or a facet region's `Vector{Tuple{Int,Symbol}}` identifier.
  - `residual::T` — the moment-fit residual on a `:cut_*` kind, zero otherwise.

The assembly kernel reads a region only through its accessors (`_region_point`, `_refresh!`, `_parent_range`, …,
`assembly.jl`), one method each, so every kind runs the same compiled loop up to the `normal` / `sides` types.
"""
struct Region{D,T<:Real,N,S}
    lower::SVector{D,T}
    upper::SVector{D,T}
    parents::Vector{Parent{D,T}}
    split::Int
    fields::NTuple{2,Int}
    kind::Symbol
    points::Vector{SVector{D,T}}
    weights::Vector{T}
    normal::N
    sides::S
    residual::T
end

function Region(lower::SVector{D,T}, upper::SVector{D,T}, parents::Vector{Parent{D,T}}, kind::Symbol, points, weights;
                split::Int=length(parents), fields::NTuple{2,Int}=(0, 0), normal=nothing, sides=nothing,
                residual=zero(T)) where {D,T}
    return Region{D,T,typeof(normal),typeof(sides)}(lower, upper, parents, split, fields, kind, points, weights, normal,
                                                    sides, residual)
end

const VolumeRegion{D,T} = Region{D,T,Nothing,Nothing}

# A volume region's box. A region stores its corners rather than an `AxisBox` because a frame may be degenerate on some
# axis, which `AxisBox` rejects; the volume consumers that want a box — the post-processing sampler and the plan's
# volume bookkeeping — build it here.
_region_box(region::Region) = AxisBox(region.lower, region.upper)

# The Jacobian of the frame `lower`–`upper`: `∏ (upper[d] − lower[d]) / 2` over the axes it extends along, a left fold
# of exact halvings, so on a box it is bit for bit `vol / 2ᴰ`, the factor `reference_to_physical` implies, and on a
# point it is 1.
function _frame_jacobian(lower::SVector{D,T}, upper::SVector{D,T}) where {D,T}
    half = (upper - lower) / 2
    jacobian = one(T)
    for d in 1:D
        iszero(half[d]) || (jacobian *= half[d])
    end
    return jacobian
end

# The measure of a region's own extent: the product of its extents over the axes it extends along — a volume region's
# box volume, a facet region's face measure (physical part and fictitious part together), 1 on a point. Compared with
# the region's physical weight sum it gives the fraction of the region its rule integrates.
function _region_measure(region::Region{D,T}) where {D,T}
    extent = region.upper - region.lower
    return prod(d -> iszero(extent[d]) ? one(T) : extent[d], 1:D)
end

# The frame coordinate of a physical point `x` of the frame `lower`–`upper`: the inverse of the frame map,
# `physical_to_reference` on every axis the frame extends along and `0` on a degenerate one.
function _frame_point(lower::SVector{D,T}, upper::SVector{D,T}, x::SVector{D,T}) where {D,T}
    return SVector{D,T}(ntuple(d -> lower[d] == upper[d] ? zero(T) :
                                    (2 * x[d] - lower[d] - upper[d]) / (upper[d] - lower[d]), Val(D)))
end

"""
    SmallOverlap{T}(region, volume, relative_volume, cover_count)

Diagnostic record for an integration region whose physical volume falls below the configured small-overlap threshold
(`GeometryTolerance.small_volume`). Conditioning of unfitted multi-level discretizations is sensitive to slivers of this
kind, so the package records them rather than dropping or rescaling them silently. Fields:

  - `region::Int` — the index of the region inside `IntegrationPlan.regions`.
  - `volume::T` — the region's physical volume.
  - `relative_volume::T` — `volume / volume(V.domain)`. Zero when the overall physical domain is degenerate.
  - `cover_count::Int` — number of parent levels covering the region. Higher cover counts on a tiny region are a
    stronger conditioning warning, since the assembly stiffness contribution involves more basis-function pairs.
"""
struct SmallOverlap{T<:Real}
    region::Int
    volume::T
    relative_volume::T
    cover_count::Int
end

function Base.show(io::IO, overlap::SmallOverlap)
    print(io, "SmallOverlap(region=", overlap.region, ", volume=", overlap.volume, ", relative=",
          overlap.relative_volume, ", cover=", overlap.cover_count, ")")
end

"""
    IntegrationPlan{D,T}

Result of [`integration_plan`](@ref): the full set of admissible integration regions for a [`Space`](@ref), plus the
diagnostics that the assembly path forwards to the user.

Fields:

  - `regions::Vector{VolumeRegion{D,T}}` — every region the assembly will see, in construction order (see
    [`Region`](@ref): each carries its box corners, its parents, its `kind` and its rule).
  - `small_overlaps::Vector{SmallOverlap{T}}` — diagnostic records for every region whose volume fell below the
    `small_volume` of the `tolerance` the plan was built with; [`AssemblyDiagnostics`](@ref) reports their number as
    `small_overlap_count`.
  - `min_volume::T` — the smallest region volume in `regions`, or zero if `regions` is empty.
  - `min_relative_volume::T` — `min_volume / volume(V.domain)`.
  - `moment_fit_residual_max::Float64` — the largest NNMF moment-fit residual observed across all `:cut_fitted`,
    `:cut_fallback`, `:cut_failed`, and `:cut_alpha_failed` regions. A `:cut_custom` region contributes whatever its
    `cut_quadrature` rule reports as a residual, so a rule with no such notion should report zero rather than leave the
    package's own accuracy statistic reading a foreign number.
"""
struct IntegrationPlan{D,T<:Real}
    regions::Vector{VolumeRegion{D,T}}
    small_overlaps::Vector{SmallOverlap{T}}
    min_volume::T
    min_relative_volume::T
    moment_fit_residual_max::Float64
end

# ── Plan construction ─────────────────────────────────────────────────────────

"""
    integration_plan(V::Space; tolerance=GeometryTolerance(T),
                               criterion=:any_parent,
                               classify_cache=_ClassifyCache{D,T}(),
                               moment_fit_cache=_MomentFitCache{D,T}()) -> IntegrationPlan

Build the [`IntegrationPlan`](@ref) for a [`Space`](@ref). The plan carries every admissible integration region the
assembly path will iterate over, together with the small-overlap / cut-region / moment-fit diagnostics.

The construction follows the recipe in `CONTRIBUTING.md`'s "Integration regions" section:

  0. Per level, per axis: check that the mesh spacing exceeds `tolerance.merge`. An axis at or below it would have its
     own cell boundaries collapsed by step 1, silently, so this combination is rejected with an `ArgumentError` naming
     the axis, its spacing, and the tolerance.
  1. Per axis: collect every level's element-boundary coordinates that bound an active cell of that level — all of them
     on an unmasked level — and canonicalise them via `merge_coordinates` so two meshes' near-coincident coordinates
     collapse to a single shared boundary. A coordinate no level keeps is the face of no active cell, so every level's
     coverage signature is constant across it and the partition is the one the full grids would have produced: the
     candidate grid follows the active region rather than the finest participating mesh.
  2. Form the Cartesian product of the resulting intervals to get candidate boxes, take each candidate's coverage
     signature — per level, the active cell containing its midpoint — and greedily merge axis-adjacent candidates that
     share it (every cell inside the merged box lies inside the same cell of every active level). A box no level covers
     is no region of the plan.
  3. Decode each box's parents from its signature, building each parent's map from the box's frame into its own
     reference frame ([`Parent`](@ref)), and check the `criterion`.
  4. Dispatch on the `PhysicalDomain` classification to pick the region quadrature: tensor Gauss for `:full`, α-scaled
     tensor Gauss for `:fictitious + alpha > 0`, and on `:cut` the NNMF moment-fit — or the domain's `cut_quadrature`
     callable, when it supplies one, in the fit's place. Drop the region entirely under strict-α (`alpha = 0` and
     `:fictitious`).
  5. Record a [`SmallOverlap`](@ref) for every region whose physical volume falls below `tolerance.small_volume`, and
     track the maximum NNMF residual observed across cut regions.

Keyword arguments:

  - `tolerance` — geometry tolerance applied at every step (`merge_coordinates`, the signatures' midpoint containment,
    small-volume reporting). Its `merge` field is an absolute length, so a mesh finer than it on any axis is rejected
    rather than silently decimated (step 0). Small geometry is fully supported; it needs a `merge` below its own
    spacing.
  - `criterion` — `:any_parent` (default), `:all_levels`, or a function `parents -> Bool`. See `_criterion_satisfied`
    for the semantics. The predicate form is an escape hatch for a coverage rule the two symbols do not express; it is
    reached through `prepare(problem; criterion = …)` and, like every plan option, survives `move!` and `activate!`
    because [`Model`](@ref) replays its captured `plan_options`. It is only ever called with a non-empty parent list:
    a box no level covers is no region under any criterion.
  - `classify_cache` — optional classification cache shared with the cell-level fictitious fold. Pass the same cache
    through `_apply_physical_fold` so the per-cell verdicts computed there can be reused at the region level.
  - `moment_fit_cache` — optional cut-region rule cache, keyed by region bounds and moment order. Pass the same cache
    across successive plans for one space — [`Model`](@ref) does, through every `move!` and `activate!` — so a cut
    region the mutation left unchanged reuses its rule instead of being refitted. On return the cache is restricted to
    the boxes *this* plan cuts: a box that has left the plan is dropped with all of its rules, so a `move!` sweep cannot
    grow the cache. A box that stays live keeps one entry per distinct moment order fitted there, so an order-elevated
    twin of the same space and this plan do not evict each other — and, where the order at a live box keeps rising (an
    hp loop), entries accumulate with the cycles rather than staying at one plan's worth. The accumulation is bounded
    above by `moment_order_factor × pmax` per axis. On a 3D sphere with 32 cut regions, four hp cycles took the cache
    from 0.035 MiB to 0.61 MiB and the next plan build from 11.9 s (no reuse) to 0.011 s.

The function is called by `prepare`, `move!`, and `_update_mask!`, all in `src/model.jl`, which own the plan a
[`Model`](@ref) carries; `assembly.jl` and `postprocessing.jl` only read that cached plan back — the volume walks
through `_region_list`, the quadrature-point cloud through each subdomain's `plan.regions` — and user code reaches it
through the [`integration_plan`](@ref) / [`integration_plans`](@ref) accessors. The projection path in `projection.jl`
builds no plan at all: it reuses this file's admissible-box partition engine (`_partition`) directly and keeps every box
the target covers, without a `criterion`.
"""
function integration_plan(V::Space{D,T}; tolerance=GeometryTolerance(T), criterion=:any_parent,
                          classify_cache=_ClassifyCache{D,T}(), moment_fit_cache=_MomentFitCache{D,T}()) where {D,T}
    # Reject a merge tolerance that cannot resolve the meshes it is about to canonicalise, before a single region is
    # built. See `_check_axis_spacing`.
    _check_axis_spacing(V.levels, Val(D), tolerance)

    # Parents and the keep-criterion first, then one record per kept box with its sizing and classification, before
    # touching the (potentially expensive) rules; then every cut rule up front (`_prefit!`), so the rule loop walks the
    # records in partition order on nothing but cache hits. `classify_cache` is shared with `_apply_physical_fold` in
    # `mesh.jl`, so a box that coincides with a previously classified cell box (the common no-overlay case) reuses the
    # fold's verdict instead of re-walking the octree.
    physical = V.physical
    records = _RegionRecord{D,T}[]
    _partition(V.levels, ntuple(identity, Val(D)), zero(SVector{D,T}), tolerance) do lower, upper, sig
        parents = _parents(V.levels, sig, lower, upper)
        _criterion_satisfied(parents, length(V.levels), criterion) || return nothing
        verdict = physical === nothing ? :full : classify_cell(physical, AxisBox{D,T}(lower, upper), classify_cache)
        push!(records, _region_record(V, lower, upper, parents, verdict))
        return nothing
    end
    _prefit!(moment_fit_cache, physical, records)

    alpha = physical === nothing ? zero(T) : physical.alpha
    custom = physical !== nothing && physical.cut_quadrature !== nothing
    domain_volume = volume(V.domain)
    regions, small_overlaps, residual_max = _plan_regions(records, moment_fit_cache, alpha, custom, tolerance,
                                                          domain_volume)
    min_volume = isempty(regions) ? zero(T) : minimum(volume(_region_box(region)) for region in regions)
    min_relative_volume = iszero(domain_volume) ? zero(T) : min_volume / domain_volume
    return IntegrationPlan{D,T}(regions, small_overlaps, min_volume, min_relative_volume, residual_max)
end

# ── Merge tolerance vs. mesh spacing ──────────────────────────────────────────
#
# `tolerance.merge` is an absolute length, and `merge_coordinates` collapses per-axis element boundaries that separate
# by no more than it. Between levels that is exactly the intent: two meshes' near-coincident boundaries should become
# one shared coordinate rather than a sliver. Within a single level it is silent destruction. An axis whose own cell
# spacing `h` satisfies `h ≤ tolerance.merge` has its boundary list decimated — genuinely distinct cell faces collapse —
# and every surviving integration region then straddles two cells of that level. That breaks the
# one-cell-per-axis-per-level contract the whole region partition rests on, and nothing downstream can detect it: the
# integrand is no longer smooth on the region, yet the tensor Gauss rule is still applied to it. The observed outcomes
# are a converged, entirely wrong solution for a span-mode B-spline basis, and an unattributed `SingularException` for
# integrated Legendre.
#
# The check is the compatibility condition between the two lengths the user controls — mesh spacing and merge tolerance
# — not a floor on either one. Geometry at h = 1e-9 stays legal; it needs a `tolerance.merge` below it, in the same
# spirit as the region-scaled `atol` in `implicit.jl` and the diagonal-scaled sign tolerance in the MeshIO extension.
#
# The threshold is `tolerance.merge` itself and not a multiple of it: it mirrors `merge_coordinates`'s own `> tol.merge`
# keep-test, so exactly the configurations that lose a coordinate are rejected, and configurations that merely sit close
# to the tolerance keep working.
function _check_axis_spacing(levels, ::Val{D}, tol::GeometryTolerance{T}) where {D,T}
    for level in levels, d in 1:D
        coords = boundary_coordinates(level.mesh)[d]
        spacing = minimum(coords[i + 1] - coords[i] for i in 1:(length(coords)-1))
        spacing > tol.merge && continue
        # Report the scale-aware value the default would have had on this axis, √eps(T) · extent, so the remedy is a
        # number to paste.
        suggested = sqrt(eps(T)) * (coords[end] - coords[begin])
        throw(ArgumentError("level $(level.id) has cell spacing h = $spacing on axis $d, at or " *
                            "below the coordinate merge tolerance $(tol.merge); pass a smaller " *
                            "merge tolerance, e.g. `tolerance=GeometryTolerance(; " *
                            "merge=$suggested)`"))
    end
    return nothing
end

# ── Admissible box partition ──────────────────────────────────────────────────

# The cells one level may parent in a partition — its slot of `_partition`'s `cells` tuple: every cell of the level
# (`_AllCells`: the volume plan, the transfer and the immersed simplices), the cells with a face on a boundary facet
# (`_side_cells` in `dirichlet.jl`), or none at all (an empty range: a level whose mesh does not reach that facet).
# `_AllCells` is a singleton so that the membership tests below fold to `true` at compile time wherever every cell is
# eligible.
struct _AllCells end
Base.in(::CartesianIndex, ::_AllCells) = true
Base.isempty(::_AllCells) = false
_cell_range(level, ::_AllCells) = CartesianIndices(level.mesh.cells)
_cell_range(_, cells::CartesianIndices) = cells

# The admissible-box partition engine: the boxes of `levels` spanned on the `free` axes through the point `anchor`,
# whose other coordinates are held fixed, each handed to `emit(lower, upper, sig)` in emission order, `sig` the box's
# coverage signature. Every consumer turns a box into its own record on the spot, so the engine materialises no box list
# of its own. It is the box-partition half of the recipe documented in `CONTRIBUTING.md`'s "Integration regions"
# section, and every partition the package builds runs through it:
#
#   - the volume plan (`integration_plan`): every axis free, every cell eligible;
#   - a boundary facet (`_boundary_facet_regions` in `dirichlet.jl`): the facet's free axes, the anchor carrying the
#     facet's own coordinates on its pinned ones, and per level only the cells with a face on the facet;
#   - the L² transfer (`_transfer_regions` in `projection.jl`): the source and target levels in one tuple.
#
# The steps:
#
#   1. Collect, per axis, every level's element-boundary coordinates that bound an active eligible cell of that level
#      (`_merged_axis_coordinates` ⇒ `_live_axis_coordinates` ⇒ `merge_coordinates`), and form the free axes'
#      non-degenerate intervals (`_intervals_from_coordinates`). A coordinate no level keeps separates two slabs of
#      identical coverage signature, which step 4 would have merged again, so dropping it costs no region and no
#      exactness.
#   2. Form the Cartesian product of those intervals, giving the *candidate* box index grid `CartesianIndices(ranges)`.
#   3. For each candidate, compute the coverage signature at its midpoint (`_signature`). On a fixed axis the midpoint
#      is `(v + v) / 2 = v`, the anchor's own coordinate.
#   4. Greedy axis-by-axis merging: starting from an unvisited candidate, extend the box along each free axis in turn as
#      long as the next "slab" of candidates is (a) unvisited and (b) shares the signature of the starting candidate.
#      Mark every merged candidate visited so it is not emitted again.
#
# The greedy merge keeps the number of regions close to the number of genuinely distinct cell-coverage patterns rather
# than producing one region per candidate. Without the merge, a small overlay's per-axis coordinates would slice the
# entire physical domain along the overlay axis, splitting far-away base cells into long chains of
# trivially-equal-signature sub-boxes for no integration benefit.
#
# Mathematical correctness: every candidate inside a merged box has the same coverage signature, so the box lies inside
# one cell of every contributing level. The integrand of any basis-function product on the box is then a smooth
# polynomial in the parent's reference frame, and the per-region Gauss rule remains exact. The greedy order of axes can
# change the *shape* of the resulting boxes (the same volume can be packed many ways) but never their *total* volume or
# their numerical contribution.
#
# A seed no level covers — an all-zero signature, where every level is inactive or ineligible — is skipped rather than
# emitted: it is no region for any user of the engine, which has nothing to integrate, constrain or transfer there.
# Skipping it changes nothing else, since a seed with a non-zero signature only ever absorbs candidates of its own
# signature: every other box, its extent and its place in the order are the ones the full partition would emit.
#
# Steps 3–4 are split across the small helpers below rather than written as one flat loop. That split is load-bearing,
# not style. The greedy merge's running upper corner `hi` is reassigned on every extension *and* read inside `ntuple`
# closures; written in one frame, Julia heap-boxes it as a `Core.Box`, rebuilds the box on each extension and turns
# every read into a dynamic dispatch. In that form this single variable accounted for 95% of `integration_plan`'s
# allocations, on a path `prepare`, `move!` and `activate!` all run. The helpers restore the separation the compiler
# needs: every frame that *closes over* `hi` receives it as an argument and never writes it, and the frames that write
# it (`_extend_axis`, and `_partition`, which turns it into ranges with `map(:, …)` rather than a closure) never close
# over it. Do not fold them back inline.

function _partition(emit, levels::Tuple, free::NTuple{F,Int}, anchor::SVector{D,T}, tol::GeometryTolerance{T},
                    cells::Tuple=map(_ -> _AllCells(), levels)) where {F,D,T}
    coords = _merged_axis_coordinates(levels, Val(D), tol, cells)
    intervals = ntuple(j -> _intervals_from_coordinates(coords[free[j]], tol), Val(F))
    ranges = ntuple(j -> length(intervals[j]), Val(F))
    any(==(0), ranges) && return nothing

    # The per-level linear cell index maps once, and the coverage signature once per candidate: both are reused inside
    # the greedy merge, where signature equality is the hot inner check.
    linmaps = map(level -> LinearIndices(level.mesh.cells), levels)
    slot = ntuple(d -> something(findfirst(==(d), free), 0), Val(D))
    sigs = map(CartesianIndices(ranges)) do index
        lower, upper = _corners(anchor, slot, intervals, index.I, index.I)
        return _signature(levels, linmaps, cells, (lower + upper) / 2, tol)
    end

    visited = falses(ranges)
    for start in CartesianIndices(ranges)
        visited[start] && continue
        sig = sigs[start]
        all(iszero, sig) && continue
        # Greedy axis-by-axis extension: extend along free axis `j` as far as the next slab of candidates matches the
        # seed signature, then repeat for the next axis. The merge never backtracks.
        hi = start.I
        for j in 1:F
            hi = _extend_axis(sigs, visited, ranges, start.I, hi, sig, j)
        end
        for c in CartesianIndices(map(:, start.I, hi))
            visited[c] = true
        end
        lower, upper = _corners(anchor, slot, intervals, start.I, hi)
        emit(lower, upper, sig)
    end
    return nothing
end

# Per-axis merged element-boundary coordinates: gather each axis's live element-boundary coordinates from every level
# (`_live_axis_coordinates`, over that level's slot of `cells`) and canonicalise them with `merge_coordinates`, so two
# meshes' nearly-coincident boundaries collapse to a single shared coordinate. Shared by the partition engine
# (`_partition`) and the lines immersed simplices are split at (`_simplex_regions` in surface.jl).
function _merged_axis_coordinates(levels, ::Val{D}, tol::GeometryTolerance{T},
                                  cells=map(_ -> _AllCells(), levels)) where {D,T}
    per_level = map((level, eligible) -> _live_axis_coordinates(level, Val(D), eligible), levels, cells)
    return ntuple(D) do d
        merged = T[]
        for live in per_level
            append!(merged, live[d])
        end
        merge_coordinates(merged, tol)
    end
end

# Per-axis live element-boundary coordinates of one level: for each axis, the coordinates that bound at least one
# *active* cell of `level` among `cells` (see `_AllCells`). This is the projection of the level's live region onto the
# axes, and it replaces the level's whole node grid in every partition the package builds. One pass over the mask marks
# all `D` axes, so a plan pays for a level's mask once, not once per axis. A level with no eligible cell contributes no
# coordinate at all.
#
# *Why the partition does not change.* A coordinate is dropped only when it separates two slabs of cells, neither of
# which is live on this level. It is then the face of no live cell here, and the level's coverage signature
# (`_signature`) is the same on both sides of it: either the line is not a node of this mesh, so both sides sit inside
# one cell, or it is a node whose two neighbouring cells are both dead, so both sides read `0`. A line the *merged* set
# loses is one that no level keeps, so every level's signature is constant across it and the greedy merge in
# `_partition` fuses the two slabs it separates anyway. The emitted box list comes out the same, in the same order —
# and the candidate grid now follows the active region instead of the finest participating mesh, which is where the
# cost of a deep `ladder` sat. Where no level is live at all the projection stops at the outermost live face instead of
# tiling the dead remainder, which changes no emitted box: `_partition` emits no box that no level covers.
#
# The remaining caveat is canonicalisation, not geometry: `merge_coordinates` keeps the *first* representative of a
# group within `tol.merge`. On a nested stack the shared nodes are bit-identical (`_mesh_axes` divides equal rationals),
# so dropping a dead line cannot change which value represents a group and the plan is bit-identical. On a non-nested
# overlay a dead line sitting within `tol.merge` below a live line of another level *was* that group's representative,
# so the canonical coordinate can move by at most `tol.merge`.
#
# An unmasked level answers from its axes directly, without touching a mask it does not have.
function _live_axis_coordinates(level, ::Val{D}, cells) where {D}
    nodes = boundary_coordinates(level.mesh)
    scan = _cell_range(level, cells)
    isempty(scan) && return ntuple(d -> nodes[d][1:0], D)
    mask = level.mask
    lo, hi = first(scan).I, last(scan).I
    mask === nothing && return ntuple(d -> nodes[d][lo[d]:(hi[d]+1)], D)
    live = ntuple(d -> falses(length(nodes[d])), D)
    # `findall` on a `BitVector` walks 64 cells per machine word, so this costs the mask's size in *words* plus its
    # active cells — not its cell count. It runs on the flattened mask rather than the `D`-dimensional one on purpose:
    # `findall` returns `Int` for a vector and `CartesianIndex{D}` for `D > 1`, and one `D`-generic loop body cannot
    # consume both. `vec` on a `BitArray` is a reshape, so nothing is copied.
    cartesian = CartesianIndices(mask)
    for i in findall(vec(mask))
        c = cartesian[i]
        c in cells || continue
        for d in 1:D
            live[d][c.I[d]] = true
            live[d][c.I[d] + 1] = true
        end
    end
    return ntuple(d -> nodes[d][live[d]], D)
end

# Form non-degenerate intervals between successive entries of a sorted, canonicalised coordinate list. Pairs closer than
# `tol.merge` are dropped — `merge_coordinates` should already have collapsed them, but the check here is a safety net.
function _intervals_from_coordinates(coords::Vector{T}, tol::GeometryTolerance{T}) where {T}
    intervals = Tuple{T,T}[]
    for i in 1:(length(coords)-1)
        lower = coords[i]
        upper = coords[i + 1]
        if upper - lower > tol.merge
            push!(intervals, (lower, upper))
        end
    end
    return intervals
end

# Physical corners of the candidate index range `lo:hi`: the interval ends on the free axes, the anchor's coordinate on
# the others. `slot[d]` is axis `d`'s position among the free axes, `0` on a fixed axis.
function _corners(anchor::SVector{D,T}, slot::NTuple{D,Int}, intervals, lo, hi) where {D,T}
    lower = SVector{D,T}(ntuple(d -> slot[d] == 0 ? anchor[d] : intervals[slot[d]][lo[slot[d]]][1], Val(D)))
    upper = SVector{D,T}(ntuple(d -> slot[d] == 0 ? anchor[d] : intervals[slot[d]][hi[slot[d]]][2], Val(D)))
    return lower, upper
end

# Coverage signature at a point: per level, the linear index of the cell containing `x`, with `0` standing for "off this
# level's mesh", "in an inactive cell" or "in a cell the level may not parent here" (`cells`) — all three mean the level
# contributes nothing at `x`. Two boxes share a signature iff they lie inside the same cell of every contributing level,
# so the smooth-integrand contract (one cell per axis per level) is preserved when merging axis-adjacent
# equal-signature boxes, and the signature alone determines a box's parents (`_parents`).
#
# The inactive-cell zero is an invariant the assembly hot path relies on, not only a filter: every region parent is
# decoded from a signature, so every parent is an active cell of its level, and `_refresh!` (`assembly.jl`) reads a
# parent's basis set without consulting the mask — on a uniform level, the level's one set for any cell. A parent built
# from any other source would have to keep that property.
#
# The signature is an `NTuple{N,Int}` built by `map` over the level tuple rather than a `Vector{Int}`, so the equality
# comparison in the merge stays on the stack and every level's lookup is statically dispatched. A `Vector{Int}` per
# candidate box was once the largest single allocation source in `integration_plan`.
function _signature(levels::Tuple, linmaps::Tuple, cells::Tuple, x::SVector{D,T}, tol::GeometryTolerance{T}) where {D,T}
    return map(levels, linmaps, cells) do level, linmap, eligible
        isempty(eligible) && return 0
        cell = locate_cell(level.mesh, x; tol)
        return cell === nothing || !(cell in eligible) || !is_active(level.mask, cell) ? 0 : linmap[cell]
    end
end

# Extend `hi` along axis `d` for as long as successive slabs keep matching.
function _extend_axis(sigs, visited, ranges::NTuple{D,Int}, start::NTuple{D,Int}, hi::NTuple{D,Int}, sig0,
                      d::Int) where {D}
    while hi[d] < ranges[d]
        next = hi[d] + 1
        _slab_matches(sigs, visited, start, hi, sig0, d, next) || break
        hi = Base.setindex(hi, next, d)
    end
    return hi
end

# Does the candidate slab one step past `hi` along axis `d` join the box seeded at `start`? It does when every candidate
# in the slab is unvisited and carries the seed signature `sig0`.
function _slab_matches(sigs, visited, start::NTuple{D,Int}, hi::NTuple{D,Int}, sig0, d::Int, next::Int) where {D}
    slab = CartesianIndices(ntuple(e -> e == d ? (next:next) : (start[e]:hi[e]), D))
    return all(c -> !visited[c] && sigs[c] == sig0, slab)
end

# ── Region parents ────────────────────────────────────────────────────────────

# The parents a box's coverage signature stands for, in level order: per non-zero slot, the level's cell with that
# linear index and the map from the region's frame `lower`–`upper` into the cell's reference frame (`Parent`). Every
# candidate the partition merged into the box shares the signature, so the whole box lies in each of these cells — the
# contract the smooth-integrand argument needs — and decoding the signature is all it takes to find them.
#
# A `for` loop that pushes, not a `map` over the levels: a `map` yields a tuple of `Union{Nothing,Parent}` that is
# boxed per level, which once made a 3D overlay plan 1.8× slower to build. The list is sized to the signature's
# non-zero slots before the first push, so each region allocates its parent storage once instead of growing it.
#
# `_parents!` appends to an existing list instead: a two-sided interface sub-cell decodes both sides' signatures into
# one parent list, side `a` first (`_simplex_regions` in `surface.jl`).
function _parents(levels::Tuple, sig::Tuple, lower::SVector{D,T}, upper::SVector{D,T}) where {D,T}
    return _parents!(sizehint!(Parent{D,T}[], count(!iszero, sig)), levels, sig, lower, upper)
end

function _parents!(parents::Vector{Parent{D,T}}, levels::Tuple, sig::Tuple, lower::SVector{D,T},
                   upper::SVector{D,T}) where {D,T}
    for (level, index) in zip(levels, sig)
        index == 0 && continue
        push!(parents, Parent(level, CartesianIndices(level.mesh.cells)[index], lower, upper))
    end
    return parents
end

# Keep-or-drop predicate for an admissible region. Three accepted shapes:
#
#   - `:any_parent`  — keep the region if any active level covers it
#                      (the common assembly criterion: every region with
#                      at least one contributing basis function counts).
#   - `:all_levels`  — keep the region only if every level covers it, i.e.
#                      only where the full superposition overlaps. No
#                      package code selects it; it is reached through
#                      `prepare(problem; criterion = :all_levels)`.
#   - `criterion isa Function` — caller-supplied predicate, called as
#                      `criterion(parents)` and returning `Bool`. The
#                      escape hatch for a coverage rule the two symbols
#                      do not express; no in-tree caller uses it, and it
#                      reaches the plan through `prepare(problem;
#                      criterion = …)`.
#
# `parents` is never empty here: `_partition` emits no box that no level covers, because such a box has no basis
# function to integrate and no parent order to size a rule from. Handing one to a callable criterion used to let one
# that answers `true` keep the box and then crash building its rule.
function _criterion_satisfied(parents, level_count::Int, criterion)
    if criterion === :any_parent
        return !isempty(parents)
    elseif criterion === :all_levels
        return length(parents) == level_count
    elseif criterion isa Function
        return criterion(parents)
    end

    throw(ArgumentError("unknown integration-region criterion $criterion"))
end

# ── Region records and rule sizing ────────────────────────────────────────────

# One kept region of a volume plan or a boundary facet before its rule is built: its corners and parents, and what
# sizing and classification decided about it, computed once per region (`_region_record`) and read by both the fits and
# the rule loop. `order` is the moment-fit order on a `:cut` verdict and repeats `counts` otherwise, where nothing reads
# it.
const _RegionRecord{D,T} = @NamedTuple{lower::SVector{D,T}, upper::SVector{D,T}, parents::Vector{Parent{D,T}},
                                       verdict::Symbol, counts::NTuple{D,Int}, order::NTuple{D,Int}}

# The record of one kept region from its classifier `verdict`: the tensor counts from its parents (`0` on an axis the
# region does not extend along) and, on a `:cut` verdict, the moment-fit order. The facet slice of a `PhysicalDomain`
# carries the domain's own `moment_order_factor`, so reading it off `V` sizes a cut face and the cells behind it alike.
function _region_record(V::Space{D,T}, lower::SVector{D,T}, upper::SVector{D,T}, parents::Vector{Parent{D,T}},
                        verdict::Symbol) where {D,T}
    counts = _axis_max(_gauss_counts, V, parents, lower, upper)
    order = verdict === :cut ? _axis_max(_moment_orders(V.physical.moment_order_factor), V, parents, lower, upper) :
            counts
    return _RegionRecord{D,T}((lower, upper, parents, verdict, counts, order))
end

# The per-axis maximum of `f(level, cell)` over a region's parent cells: the one helper that sizes every rule a region
# carries. `f` returns a per-axis tuple for one parent cell — its tensor Gauss counts (`_gauss_counts`), its moment-fit
# order (`_moment_orders`), its post-processing sample lattice (`_sample_cells`) — and the maximum over the parents
# makes the rule exact for every parent level's basis at once. `parents` is any list of items carrying a `level` id of
# `V` and a `cell`.
#
# The value is read per parent *cell*, not per level. Under one order per level the two are the same value and the rule
# is unchanged; under a per-cell order they differ, and reading the cell is both cheaper and still exact — the minimum
# rule only ever *removes* modes from a cell, so no function on that parent exceeds its own cell's per-axis order.
# Sizing from the level's nominal maximum instead would make one high-p cell raise the rule on every region the level
# touches.
#
# `lower` / `upper`, when given, are the region's corners: an axis on which they coincide — a boundary facet's pinned
# axis — carries no extent to integrate along and gets 0.
#
# `f` runs once per parent and the maximum is taken tuple-wise (`max.`), not once per parent *and* axis: what `f` costs
# is resolving the parent's level and reading its cell's order, which is the same work for every axis. Sizing is the
# largest single cost of a partition-only plan — on the deep ladder of the build benchmark (6 levels, 4272 regions,
# 21 008 parents) it was a third of `integration_plan` while `f` ran per axis. An empty `parents` still raises, as
# `maximum` did: a region without a parent has no rule to size.
#
# `f::F` is a type parameter on purpose: `f` is only passed into a closure here, never called directly, and Julia does
# not specialise a method on a function argument it merely passes along — the helper then ran on `f::Function` and
# dispatched dynamically for every parent, boxing each per-cell tuple it returned.
function _axis_max(f::F, V::Space{D}, parents, lower=nothing, upper=nothing) where {F,D}
    counts = mapreduce(parent -> f(_level_by_id(V, parent.level), parent.cell), (a, b) -> max.(a, b), parents)
    lower === nothing && return counts
    return ntuple(d -> lower[d] == upper[d] ? 0 : counts[d], Val(D))
end

# Per-axis tensor Gauss counts that integrate products of one cell's basis functions exactly.
_gauss_counts(level, cell) = recommended_quadrature_order(level.basis, cell_order(level, cell))

# Per-axis moment-fit basis order of one parent cell: `moment_order_factor` times its `cell_order`, so `_axis_max` takes
# it over the region's parent CELLS — not over their levels, so one high-p cell elsewhere on a level does not raise the
# moment basis here.
#
# At the default `moment_order_factor = 2` the basis spans degree `2p`, so trial × test products integrate exactly and
# cut regions reach the exactness `:full` regions already get from tensor Gauss. That costs `∏(2p+1)` weights against
# the `∏(p+1)` a factor of 1 would need — 4.6×–5.8× the points in 3D at `p = 2…4`, and roughly the cube of that in fit
# time: over the 146 cut cells of the space-time cavity geometry the fit runs 1.1 s at 125 weights and ≈ 500 s at 729.
#
# A factor of 1 is nevertheless not a drop-in saving, and the condition is sharp: a stiffness integrand is degree
# `2(p−1)`, so exactness needs `factor ≥ 2 − 2/p`, which 1 meets only to `p = 2`. Measured on the order-1-to-7 Kirsch
# plate, factor 1 stops converging after `p = 2` and by `p = 7` is three times worse than first order while the fit
# residual stays at ~1e-16. `physical_domain`'s docstring carries that table.
#
# The convention is per axis, so a boundary facet reads it on its free axes alone (`_facet_regions` in `dirichlet.jl`):
# reading it from here rather than restating it is what keeps a cut face and the cells behind it integrating products of
# traces to the same degree.
_moment_orders(factor::Int) = (level, cell) -> factor .* cell_order(level, cell)

# ── Cut-region fits ───────────────────────────────────────────────────────────

# Moment-fit rule cache, keyed by the canonicalised region bounds and the moment basis order, each value one cut
# region's rule `(ref_points, ref_weights, residual, status)` as `_fit_rule` returns it.
#
# The cache belongs to one *space*, not to one plan. Within a single plan no two regions share a key — `_partition`
# emits pairwise-disjoint boxes — so `_prefit!` fits each key it does not already hold exactly once up front and the
# region loop reads back nothing but hits. Its value is across plan rebuilds — a `move!` leaves the great majority of
# cut boxes bit-identical, and refitting one costs hundreds of milliseconds in 3D — so [`Model`](@ref) owns one per
# distinct participating space and threads it back in through every rebuild. That scope also keeps the key honest: a
# space's `PhysicalDomain` is immutable and is carried unchanged through every level rebuild (`_relevel`), so one cache
# sees exactly one `physical` for its whole life and the key need not include it. `_prefit!` bounds the cache to the
# current plan's regions, so a long `move!` sweep holds one plan's rules, not the sweep's.
const _MomentFitKey{D,T} = Tuple{SVector{D,T},SVector{D,T},NTuple{D,Int}}
const _MomentFitValue{D,T} = Tuple{Vector{SVector{D,T}},Vector{T},T,Symbol}
const _MomentFitCache{D,T} = Dict{_MomentFitKey{D,T},_MomentFitValue{D,T}}

# Fit every cut region's rule before the rule loop consumes any of them, in parallel (`_fit_rules`), and reduce the
# cache to exactly this plan's regions. Classification ran first and serially, in the records (it is memoised, and a
# `Dict` is not thread-safe); the fits fill the cache afterwards, serially; the rule loop then runs in region order on
# nothing but cache hits, so region order, `residual_max` and the small-overlap bookkeeping are the serial path's, bit
# for bit. A custom `cut_quadrature` is fitted serially (`_fit_rules`) and still gets the dedup and the cross-plan
# reuse.
#
# `live` is the plan's whole cut-key set, not just its misses, because restricting the cache afterwards is what keeps a
# cache threaded through a long `move!` sweep from holding the sweep's worth of rules. On exit the cache holds only the
# *boxes* this plan cuts — but at every moment order fitted at one of them, which is where the eviction rule differs
# from the lookup key. See the `filter!` below for why the two are not the same rule, and what the difference costs.
function _prefit!(cache::_MomentFitCache{D,T}, physical, records::Vector{_RegionRecord{D,T}}) where {D,T}
    physical === nothing && return cache
    live = Set{_MomentFitKey{D,T}}()
    for record in records
        record.verdict === :cut && push!(live, (record.lower, record.upper, record.order))
    end
    pending = _MomentFitKey{D,T}[key for key in live if !haskey(cache, key)]
    fitted = _fit_rules(physical, pending, physical.cut_quadrature)
    for i in eachindex(pending)
        cache[pending[i]] = fitted[i]
    end
    # Evict by box, not by full key. The bound the eviction exists for is the geometric one — a `move!` sweep must not
    # leave the cache holding every box the overlay ever passed over — and a box that has left the plan takes all of its
    # rules with it either way, so keying the eviction on the box alone costs that bound nothing: a move changes which
    # boxes are cut, never the moment order at one of them.
    #
    # What it buys is the case full-key eviction gets wrong. An order-elevated twin of the same space — `estimate`'s
    # `V⁺`, `adaptivity.jl` — has exactly this plan's boxes at a higher moment order, and under full-key eviction the
    # two orders evict each other, so every `estimate` / `adapted` alternation refits both from scratch. Measured on a
    # 3D sphere probe (base 4³, order 2, depth 1, 80 cut regions, 4 threads, -O0): a repeated `estimate` cost 38.7 s and
    # the step after it 1.9 s; evicting by box they cost 0.006 s and 0.03 s, with the elevated order fitted once.
    #
    # What it costs, stated plainly because it is not free: a box that stays live keeps one entry per distinct moment
    # order ever fitted there. Under a `move!` sweep that is flat, since orders do not change there. Under an hp loop it
    # is not: `refine` raises the order at a live cut cell, the box does not leave the plan, and the entries at it
    # accumulate *per cycle*. Bounded above — the key's order is `moment_order_factor × cell_order` and `refine` caps
    # `cell_order` at `pmax` — but not by one plan's worth.
    #
    # The trade, measured on one configuration: four `estimate` → `refine` → `adapted` cycles on a 3D sphere (ladder
    # base 4³, order 1, depth 1, 32 cut boxes) grew the cache from 32 entries / 0.035 MiB to 96 entries over the same 32
    # boxes / 0.61 MiB, and the next step's plan build cost 0.011 s against 11.9 s with no reuse. An h-step resets the
    # accumulation where it lands, because the boxes it splits do leave the plan.
    live_boxes = Set((key[1], key[2]) for key in live)
    filter!(entry -> (entry.first[1], entry.first[2]) in live_boxes, cache)
    return cache
end

# The rules of a list of cut keys `(lower, upper, order)`, in key order: `_fit_rule` with `rule` on every key, on every
# thread.
#
# At the default `moment_order_factor = 2` the fits *are* the plan: 99.9% of a 3D build, tens of seconds over a few
# dozen cut cells. They are also independent of one another — `_fit_rule` reads the immutable `PhysicalDomain` and the
# box and touches nothing else — so each writes its own slot of a pre-sized vector, with no `Dict` in reach.
#
# A user `cut_quadrature` is fitted serially instead. It is caller code under no thread-safety contract — the package's
# own test rule counts its calls through a shared `Ref` — so the package does not unilaterally run it on N threads.
function _fit_rules(physical::PhysicalDomain, keys::Vector{_MomentFitKey{D,T}}, rule) where {D,T}
    fitted = Vector{_MomentFitValue{D,T}}(undef, length(keys))
    fit(key) = _fit_rule(physical, key[1], key[2], key[3], rule)
    if rule !== nothing
        map!(fit, fitted, keys)
    else
        Threads.@threads for i in eachindex(keys)
            fitted[i] = fit(keys[i])
        end
    end
    return fitted
end

# One cut region's rule on the box `lower`–`upper`, in the box's reference frame — the moment fit (`moment_fit_rule`, at
# the domain's own `target_residual`) when `rule` is `nothing`, else `rule(physical, box, order)` — mapped as follows:
#
#   - `ref_points` are the rule's physical points mapped into the box's `[−1, 1]ᴰ` reference frame via
#     `physical_to_reference`;
#   - `ref_weights` are the rule's physical weights divided by the box Jacobian `vol(box) / 2ᴰ`, so the assembly hot
#     loop's `weight * jacobian` step recovers the original physical weight (the same convention used for `:full`
#     regions);
#   - `residual` is the moment-fit L² residual reported by `fcm.jl`;
#   - `status` is `:fitted`, `:fallback`, or `:empty` — see [`moment_fit_rule`](@ref). The mapping above is
#     rule-agnostic, so a `:fallback` rule (the raw Saye volume rule) is mapped and consumed exactly like a fitted one.
#
# `physical.cut_quadrature` is the user extension point: a non-`nothing` callable supplies the cut-cell rule in place of
# the moment fit, in the same 4-tuple shape and the same *physical*-frame convention. Placing it here puts a custom rule
# under the same per-box memoisation, the same reference-frame mapping, and the same α-FCM blend as the fit, so swapping
# the rule changes exactly one thing. `rule` is the caller's, not read off the domain here, because a caller may decline
# the domain's rule: the volume plan passes `physical.cut_quadrature`, and a boundary facet passes `nothing`, since it
# never consults it (see `_facet_regions` in `dirichlet.jl`). Passed as a value, its type decides the branch at compile
# time, so a space with a custom rule compiles no moment fit for its cells, and a facet compiles no call to the rule.
#
# A pure function of its arguments, which is what lets `_fit_rules` run it on any thread.
function _fit_rule(physical::PhysicalDomain, lower::SVector{D,T}, upper::SVector{D,T}, order::NTuple{D,Int},
                   rule) where {D,T}
    box = AxisBox{D,T}(lower, upper)
    target_residual = physical.target_residual
    points, weights, residual, status = rule === nothing ? moment_fit_rule(physical, box, order; target_residual) :
                                        rule(physical, box, order)
    jacobian = volume(box) / convert(T, 2^D)
    return ([physical_to_reference(box, p) for p in points], T[w / jacobian for w in weights], T(residual), status)
end

# ── Region rules ──────────────────────────────────────────────────────────────

# The rule loop of `integration_plan`: one `VolumeRegion` per record, in record order, through `_trimmed_rule` — except
# a fictitious box under strict α = 0, which has nothing to integrate and is dropped — with the small-overlap records
# and the largest cut residual. Small-overlap diagnostics are accumulated rather than acted on: the package exposes the
# warning, the caller decides whether to stabilise. A function barrier on `alpha`, a field of the abstractly typed
# `Space.physical`, so the per-region rule dispatch below is static. The region list is reserved at the record count,
# its bound, so it is filled without regrowing.
function _plan_regions(records::Vector{_RegionRecord{D,T}}, fits::_MomentFitCache{D,T}, alpha, custom::Bool,
                       tolerance::GeometryTolerance{T}, domain_volume::T) where {D,T}
    regions = sizehint!(VolumeRegion{D,T}[], length(records))
    small_overlaps = SmallOverlap{T}[]
    tensor_cache = _TensorRuleCache{D,T}()
    alpha_cache = Dict{NTuple{D,Int},Vector{T}}()
    residual_max = 0.0
    for record in records
        record.verdict === :fictitious && iszero(alpha) && continue
        fit = record.verdict === :cut ? fits[(record.lower, record.upper, record.order)] : nothing
        kind, points, weights, residual = _trimmed_rule(record.verdict, record.counts, tensor_cache, alpha_cache, alpha,
                                                        fit, custom)
        residual > residual_max && (residual_max = Float64(residual))
        region = Region(record.lower, record.upper, record.parents, kind, points, weights; residual)
        region_volume = volume(_region_box(region))
        push!(regions, region)
        if region_volume <= tolerance.small_volume
            relative_volume = iszero(domain_volume) ? zero(T) : region_volume / domain_volume
            push!(small_overlaps,
                  SmallOverlap{T}(length(regions), region_volume, relative_volume, length(record.parents)))
        end
    end
    return regions, small_overlaps, residual_max
end

# Tensor Gauss rule cache, keyed by per-axis count tuple. Built lazily and shared across every `:full` region that needs
# the same per-axis count. `:full` regions then alias the cached `points`/`weights` arrays directly, so they cost only
# the assembly-loop iteration time — no allocation per region.
const _TensorRuleCache{D,T} = Dict{NTuple{D,Int},TensorQuadrature{D,T}}

function _cached_tensor_quadrature!(cache::_TensorRuleCache{D,T}, counts::NTuple{D,Int}, ::Type{T}) where {D,T}
    return get!(() -> _tensor_gauss_rule(counts, T), cache, counts)
end

# The rule one admissible region carries, from the classifier's `verdict` on it: the single home of the dispatch the
# volume plan and the boundary facets share, of the shared base rules, of the α-FCM blend, and of the closed `kind`
# vocabulary (`CONTRIBUTING.md`, "Region quadrature kinds"). Returns `(kind, points, weights, residual)`:
#
#   - `:full`                         → `:full`, the base tensor rule, ALIASED
#   - `:fictitious`, `α == 0`         → `:fictitious`, the empty rule (the volume plan
#                                       drops such a region before asking; a facet
#                                       keeps it, see `_facet_regions`)
#   - `:fictitious`, `α > 0`          → `:fictitious_alpha`, the α-scaled base rule
#   - `:cut`, a fitted rule           → `:cut_fitted`
#     (α > 0 appends the α-scaled base rule for stabilisation)
#   - `:cut`, the domain's own rule   → `:cut_custom` (`custom`: the domain's
#     `cut_quadrature` supplied it; α blends it exactly as it blends a fit)
#   - `:cut`, a failed fit            → `:cut_fallback`, the raw Saye volume rule
#     (correct but uncompressed; α > 0 appends the α-scaled base rule as above)
#   - `:cut`, empty Ω ∩ box, `α == 0` → `:cut_failed`, the empty rule (zero contribution)
#   - `:cut`, empty Ω ∩ box, `α > 0`  → `:cut_alpha_failed`, the α-scaled base rule alone
#     (nonzero: the physical sliver is negligible, but the cell's dofs stay α-stabilised)
#
# A `:fallback` rule is the Saye volume rule itself, so it is a valid non-negative physical rule and is consumed exactly
# like a fit — only with 50–200× the points. The distinct kind is what makes that cost visible in the diagnostics
# instead of hiding it in `:cut_fitted`. A custom rule owns the region outright, so it gets the single kind
# `:cut_custom` whatever status symbol it returns — the diagnostic vocabulary stays closed rather than growing one kind
# per user rule. `:empty` is the one status a custom rule shares with the fit, and it still routes to `:cut_failed` /
# `:cut_alpha_failed`.
#
# α-FCM on a cut region stabilises the fictitious part by *adding* the α-scaled base rule to the physical rule. Writing
# the α-FCM integrand as
#     ∫_box α(x) f = ∫_phys f + α∫_fict f = (1−α)∫_phys f + α∫_box f,
# the rule is the concatenation `(1−α)·fit ∪ α·tensor`, fit first. Both summands are exact (the fit integrates the
# physical part, tensor Gauss the whole box), so the total is exact for the α-FCM integrand; unlike the strict
# physical-only rule it gives *every* mode — including the interior bubbles whose support lies entirely in the
# fictitious corner of a thin cut — a nonzero, well-posed contribution. When Ω ∩ box carries no rule at all (a
# degenerate sliver, whose physical contribution is negligible by construction — the domain volume is captured by the
# well-fitted cells), the α·tensor part is kept alone so the cell's dofs are α-stabilised rather than left singular.
#
# The base rule is the region's tensor Gauss rule at its per-axis `counts` (0 on an axis the region does not extend
# along), read from the caller's cache `rules`, and its α-scaled weights from `alpha_rules`, both keyed by the counts:
# every `:full` region of one order aliases one rule, and every α-scaled kind one weight vector, so neither costs an
# allocation per region. The α-scaled weights are looked up only when `α > 0` and the verdict is not `:full`, and the
# counts alone key them because the caller's cache belongs to one `α`: one plan's (one `PhysicalDomain`), or one face's
# (a resolver serves several spaces, which may carry different `α`). `fit` is the cut rule `(points, weights, residual,
# status)` in the region's frame, read only on a `:cut` verdict. The residual is the fit's on every `:cut` kind and zero
# otherwise.
function _trimmed_rule(verdict::Symbol, counts::NTuple{D,Int}, rules::_TensorRuleCache{D,T},
                       alpha_rules::Dict{NTuple{D,Int},Vector{T}}, alpha, fit, custom::Bool) where {D,T}
    base = _cached_tensor_quadrature!(rules, counts, T)
    verdict === :full && return :full, base.points, base.weights, zero(T)
    scaled = iszero(alpha) ? base.weights : get!(() -> base.weights .* alpha, alpha_rules, counts)
    if verdict === :fictitious
        iszero(alpha) && return :fictitious, SVector{D,T}[], T[], zero(T)
        return :fictitious_alpha, base.points, scaled, zero(T)
    end
    points, weights, residual, status = fit
    if status === :empty
        iszero(alpha) && return :cut_failed, SVector{D,T}[], T[], residual
        return :cut_alpha_failed, base.points, scaled, residual
    end
    kind = custom ? :cut_custom : status === :fallback ? :cut_fallback : :cut_fitted
    iszero(alpha) && return kind, points, weights, residual
    return kind, vcat(points, base.points), vcat(weights .* (one(T) - alpha), scaled), residual
end

# The kinds `_trimmed_rule` gives a region `∂Ω` crosses, whatever rule it ended up with, and the subset on which the
# moment fit failed (a custom rule never fails a fit: no fit ran). The diagnostics count both (`_kind_stats` in
# `model.jl`), and the post-processing sampler cut-refines exactly the first (`_sample_cells`).
const _CUT_KINDS = (:cut_fitted, :cut_custom, :cut_fallback, :cut_failed, :cut_alpha_failed)
const _FAILED_KINDS = (:cut_fallback, :cut_failed, :cut_alpha_failed)
