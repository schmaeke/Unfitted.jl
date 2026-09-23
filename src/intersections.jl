# ── Types ─────────────────────────────────────────────────────────────────────

"""
    ParentRef{D,T}(level, cell, local_box, parent_box)

Reference to one parent cell of a [`VolumeRegion`](@ref). Carries
both the parent's identity and the integration box's geometry in the
parent's reference frame, so the assembly hot loop never re-derives
either:

  - `level::Int` — id of the parent level.
  - `cell::CartesianIndex{D}` — parent cell index inside that level's mesh.
  - `local_box::AxisBox{D,T}` — the integration region's box mapped into
    the parent cell's reference frame `[−1, 1]ᴰ`, via
    [`physical_to_reference`](@ref). Used to map region quadrature points
    into the parent's reference frame for basis evaluation.
  - `parent_box::AxisBox{D,T}` — the parent cell's physical box. Used to
    compute the per-axis gradient scaling factor `2 / edge_lengths`
    needed when pulling basis gradients back to physical coordinates.
"""
struct ParentRef{D,T<:Real}
    level::Int
    cell::CartesianIndex{D}
    local_box::AxisBox{D,T}
    parent_box::AxisBox{D,T}
end

"""
    VolumeRegion{D,T}(box, parents, quadrature)

One admissible integration region — an axis-aligned box that lies inside
a single cell of every covering parent level. `box` is the region's
physical-frame extent, `parents` is the list of parents covering it
(typically one per active level), and `quadrature` is the
[`RegionQuadrature`](@ref) the assembly hot loop iterates.

Because every parent cell sees `box` as a smooth (C∞) sub-rectangle of
its own reference frame, the integrand of any basis-function product is
smooth throughout `box`, and the tensor Gauss rule integrates it exactly
to the recommended order. Cut regions replace the tensor Gauss rule with
a moment-fit rule from `src/fcm.jl`; fictitious-α regions reuse the
tensor Gauss points with α-scaled weights.
"""
struct VolumeRegion{D,T<:Real}
    box::AxisBox{D,T}
    parents::Vector{ParentRef{D,T}}
    quadrature::RegionQuadrature{D,T}
end

"""
    SmallOverlap{T}(region, volume, relative_volume, cover_count)

Diagnostic record for an integration region whose physical volume falls
below the configured small-overlap threshold
(`GeometryTolerance.small_volume`). Conditioning of unfitted multi-level
discretizations is sensitive to slivers of this kind, so the package
records them rather than dropping or rescaling them silently. Fields:

  - `region::Int` — the index of the region inside
    `IntegrationPlan.regions`.
  - `volume::T` — the region's physical volume.
  - `relative_volume::T` — `volume / volume(V.domain)`. Zero when the
    overall physical domain is degenerate.
  - `cover_count::Int` — number of parent levels covering the region.
    Higher cover counts on a tiny region are a stronger conditioning
    warning, since the assembly stiffness contribution involves more
    basis-function pairs.
"""
struct SmallOverlap{T<:Real}
    region::Int
    volume::T
    relative_volume::T
    cover_count::Int
end

"""
    IntegrationPlan{D,T}

Result of [`integration_plan`](@ref): the full set of admissible
integration regions for a [`Space`](@ref), plus the diagnostics that the
assembly path forwards to the user.

Fields:

  - `regions::Vector{VolumeRegion{D,T}}` — every region the assembly
    will see, in construction order.
  - `tolerance::GeometryTolerance{T}` — the tolerance used to build this
    plan, retained for traceability.
  - `small_overlap_count::Int` — `length(small_overlaps)`.
  - `small_overlaps::Vector{SmallOverlap{T}}` — diagnostic records for
    every region whose volume fell below `tolerance.small_volume`.
  - `min_volume::T` — the smallest region volume in `regions`, or zero
    if `regions` is empty.
  - `min_relative_volume::T` — `min_volume / volume(V.domain)`.
  - `moment_fit_residual_max::Float64` — the largest NNMF moment-fit
    residual observed across all `:cut_fitted`, `:cut_fallback`,
    `:cut_failed`, and `:cut_alpha_failed` regions. A `:cut_custom` region
    contributes whatever its `cut_quadrature` rule reports as a residual,
    so a rule with no such notion should report zero rather than leave the
    package's own accuracy statistic reading a foreign number.
"""
struct IntegrationPlan{D,T<:Real}
    regions::Vector{VolumeRegion{D,T}}
    tolerance::GeometryTolerance{T}
    small_overlap_count::Int
    small_overlaps::Vector{SmallOverlap{T}}
    min_volume::T
    min_relative_volume::T
    moment_fit_residual_max::Float64
end

function Base.show(io::IO, overlap::SmallOverlap)
    print(io, "SmallOverlap(region=", overlap.region, ", volume=", overlap.volume, ", relative=",
          overlap.relative_volume, ", cover=", overlap.cover_count, ")")
end

# ── Region quadrature dispatch and caching ────────────────────────────────────

# Per-axis maximum recommended quadrature order across a region's parents.
# `level_lookup(id)` resolves a parent's level id to its `Level`. Taking
# the per-axis maximum guarantees that the rule is exact for every parent
# level's basis simultaneously.
#
# The order is read per parent *cell*, not per level. Under one order per level
# the two are the same value and the rule is unchanged; under a per-cell order
# they differ, and reading the cell is both cheaper and still exact — the
# minimum rule only ever *removes* modes from a cell, so no function on that
# parent exceeds its own cell's per-axis order. Sizing from the level's nominal
# maximum instead would make one high-p cell raise the rule on every region the
# level touches.
function _parent_quadrature_counts(::Val{D}, parents, level_lookup) where {D}
    return ntuple(D) do d
        maximum(parents) do parent
            level = level_lookup(parent.level)
            recommended_quadrature_order(level.basis, cell_order(level, parent.cell))[d]
        end
    end
end

# Tensor Gauss rule cache, keyed by per-axis count tuple. Built lazily and
# shared across every `:full` region that needs the same per-axis count.
# `:full` regions then alias the cached `points`/`weights` arrays directly
# in their `RegionQuadrature`, so they cost only the assembly-loop
# iteration time — no allocation per region.
function _cached_tensor_quadrature!(cache::Dict{NTuple{D,Int},TensorQuadrature{D,T}},
                                    counts::NTuple{D,Int}, ::Type{T}) where {D,T}
    return get!(cache, counts) do
        _tensor_gauss_rule(counts, T)
    end
end

# α-scaled weights cache for `:fictitious_alpha` regions, keyed by
# per-axis quadrature counts. The PhysicalDomain (and so `alpha`) is
# constant for the duration of one `integration_plan` call, so `alpha`
# need not be part of the cache key.
function _cached_alpha_weights!(cache::Dict{NTuple{D,Int},Vector{T}}, counts::NTuple{D,Int},
                                base_weights::Vector{T}, alpha::T) where {D,T}
    return get!(cache, counts) do
        base_weights .* alpha
    end
end

# Pick the moment-fit basis order for a cut region. With the default
# `moment_order_factor = 2` on `V.physical`, the basis spans degree twice the
# per-axis maximum of `cell_order` over the region's parent CELLS — not over
# their levels, so one high-p cell elsewhere on a level does not raise the
# moment basis here — matching what tensor Gauss integrates exactly on `:full`
# regions and so bringing cut-region integrand exactness to the same level as
# the rest of assembly. Users whose integrand is only degree-`p` (linear forms
# / sources) can set the factor to 1, halving the basis at the cost of giving
# up bilinear-form exactness.
function _moment_order_for_region(V::Space{D}, parents) where {D}
    factor = V.physical.moment_order_factor
    return ntuple(D) do d
        factor * maximum(p -> cell_order(_level_by_id(V, p.level), p.cell)[d], parents)
    end
end

# Moment-fit rule cache, keyed by the canonicalised region bounds and the
# moment basis order. Returns `(ref_points, ref_weights, residual, status)`:
#
#   - `ref_points` are the moment-fit physical points mapped into the
#     region's `[−1, 1]ᴰ` reference frame via `physical_to_reference`;
#   - `ref_weights` are the moment-fit physical weights divided by the
#     standard region Jacobian `vol(box) / 2ᴰ`, so the assembly hot
#     loop's `weight * jacobian` step recovers the original physical
#     weight (the same convention used for `:full` regions);
#   - `residual` is the moment-fit L² residual reported by `fcm.jl`;
#   - `status` is `:fitted`, `:fallback`, or `:empty` — see
#     [`moment_fit_rule`](@ref). The mapping above is rule-agnostic, so a
#     `:fallback` rule (the raw Saye volume rule) is cached and consumed
#     exactly like a fitted one.
#
# `physical.cut_quadrature` is the user extension point: a non-`nothing`
# callable supplies the cut-cell rule in place of the moment fit, in the same
# 4-tuple shape and the same *physical*-frame convention. Placing it here —
# rather than in `_build_region_quadrature` — puts a custom rule under the same
# per-box memoisation, the same reference-frame mapping, and the same α-FCM
# blend as the fit, so swapping the rule changes exactly one thing. The field is
# concretely typed, so `Q === Nothing` folds the branch away and the default
# path is the moment fit and nothing else.
#
# The cache belongs to one *space*, not to one plan. Within a single plan no
# two regions share a key — `_merged_boxes` emits pairwise-disjoint boxes — so
# `_prefit_cut_rules!` fits each key it does not already hold exactly once up
# front and the region loop below reads back nothing but hits. Its value is across plan rebuilds — a
# `move!` leaves the great majority of cut boxes bit-identical, and refitting
# one costs hundreds of milliseconds in 3D — so [`Model`](@ref) owns one per
# distinct participating space and threads it back in through every rebuild. That
# scope also keeps the key honest: a space's `PhysicalDomain` is immutable
# and is carried unchanged through `moved_space` / `_remasked_space`, so one
# cache sees exactly one `physical` for its whole life and the key need not
# include it. `_prefit_cut_rules!` bounds the cache to the current plan's
# regions, so a long `move!` sweep holds one plan's rules, not the sweep's.
const _MomentFitKey{D,T} = Tuple{SVector{D,T},SVector{D,T},NTuple{D,Int}}
const _MomentFitValue{D,T} = Tuple{Vector{SVector{D,T}},Vector{T},T,Symbol}
const _MomentFitCache{D,T} = Dict{_MomentFitKey{D,T},_MomentFitValue{D,T}}

# The rule itself, off the cache: the moment fit (or the user's
# `cut_quadrature`) followed by the reference-frame mapping described above.
# Split out of `_cached_moment_fit!` so `_prefit_cut_rules!` can run the very
# same conversion from another thread, with no `Dict` in reach.
function _fit_moment_rule(physical::PhysicalDomain, box::AxisBox{D,T},
                          moment_order::NTuple{D,Int}) where {D,T}
    rule = physical.cut_quadrature
    phys_pts, phys_ws, residual, status = rule === nothing ?
                                          moment_fit_rule(physical, box, moment_order;
                                                          target_residual=physical.target_residual) :
                                          rule(physical, box, moment_order)
    jac = volume(box) / convert(T, 2^D)
    ref_pts = [physical_to_reference(box, p) for p in phys_pts]
    ref_ws = T[w / jac for w in phys_ws]
    return (ref_pts, ref_ws, T(residual), status)
end

function _cached_moment_fit!(cache::_MomentFitCache{D,T}, physical::PhysicalDomain,
                             box::AxisBox{D,T}, moment_order::NTuple{D,Int}) where {D,T}
    return get!(() -> _fit_moment_rule(physical, box, moment_order), cache,
                (box.lower, box.upper, moment_order))
end

# Build the region quadrature for one admissible box, dispatching on the
# `PhysicalDomain` classification:
#
#   - no `PhysicalDomain` attached      → `:full`            tensor Gauss
#   - `:full`                           → `:full`            tensor Gauss
#   - `:fictitious` + `alpha == 0`      → drop region (return `nothing`)
#   - `:fictitious` + `alpha > 0`       → `:fictitious_alpha` α-scaled tensor Gauss
#   - `:cut` with successful moment fit → `:cut_fitted`       moment-fit rule
#     (α > 0 appends the α-scaled full-cell tensor rule for stabilisation)
#   - `:cut` + `physical.cut_quadrature` → `:cut_custom`       the user's rule
#     (in place of the moment fit; α blends it exactly as it blends a fit)
#   - `:cut` with failed moment fit     → `:cut_fallback`     raw Saye volume rule
#     (correct but uncompressed; α > 0 appends the α-scaled tensor rule as above)
#   - `:cut`, empty Ω ∩ box, α == 0     → `:cut_failed`       empty rule (zero contribution)
#   - `:cut`, empty Ω ∩ box, α > 0      → `:cut_alpha_failed` α-scaled tensor rule only
#     (nonzero: the physical sliver is negligible, but the cell's dofs stay α-stabilised)
#
# Returns `(RegionQuadrature, residual)` for kept regions and
# `(nothing, 0)` when the region is dropped. The `residual` field is the
# NNMF moment-fit residual on cut regions and zero otherwise.
#
# `classify_cache` is shared with `_apply_physical_fold` in `mesh.jl`, so
# a region whose box coincides with a previously-classified cell box (the
# common no-overlay case) reuses the classification verdict instead of
# re-walking the octree.
function _build_region_quadrature(V::Space{D,T}, box::AxisBox{D,T}, parents,
                                  tensor_cache::Dict{NTuple{D,Int},TensorQuadrature{D,T}},
                                  alpha_cache::Dict{NTuple{D,Int},Vector{T}},
                                  moment_fit_cache::_MomentFitCache{D,T},
                                  classify_cache::_ClassifyCache{D,T}) where {D,T}
    counts = _parent_quadrature_counts(Val(D), parents, id -> _level_by_id(V, id))
    base_rule = _cached_tensor_quadrature!(tensor_cache, counts, T)

    physical = V.physical
    if physical === nothing
        return RegionQuadrature{D,T}(:full, base_rule.points, base_rule.weights), zero(T)
    end

    state = classify_cell(physical, box, classify_cache)
    if state === :full
        return RegionQuadrature{D,T}(:full, base_rule.points, base_rule.weights), zero(T)
    elseif state === :fictitious
        iszero(physical.alpha) && return nothing, zero(T)
        scaled = _cached_alpha_weights!(alpha_cache, counts, base_rule.weights, physical.alpha)
        return RegionQuadrature{D,T}(:fictitious_alpha, base_rule.points, scaled), zero(T)
    else  # `:cut` — moment-fit pipeline from `src/fcm.jl`.
        moment_order = _moment_order_for_region(V, parents)
        ref_pts, ref_ws, residual, status = _cached_moment_fit!(moment_fit_cache, physical, box,
                                                                moment_order)
        # A `:fallback` rule is the Saye volume rule itself, so it is a valid
        # non-negative physical rule and is consumed exactly like a fit — only
        # with 50–200× the points. The distinct kind is what makes that cost
        # visible in the diagnostics instead of hiding it in `:cut_fitted`.
        #
        # A custom `cut_quadrature` rule owns the region outright, so it gets the
        # single kind `:cut_custom` whatever status symbol it returns — the
        # diagnostic vocabulary stays closed rather than growing one kind per
        # user rule. `:empty` is the one status a custom rule shares with the
        # fit, and it still routes to `:cut_failed` / `:cut_alpha_failed` below.
        kind = physical.cut_quadrature !== nothing ? :cut_custom :
               status === :fallback ? :cut_fallback : :cut_fitted
        if iszero(physical.alpha)
            status === :empty &&
                return RegionQuadrature{D,T}(:cut_failed, SVector{D,T}[], T[]), residual
            return RegionQuadrature{D,T}(kind, ref_pts, ref_ws), residual
        end

        # α-FCM on a cut cell: stabilise the fictitious part by *adding* the
        # α-scaled full-cell tensor rule to the physical moment-fit rule. Writing
        # the α-FCM integrand as
        #     ∫_cell α(x) f = ∫_phys f + α∫_fict f = (1−α)∫_phys f + α∫_cell f,
        # the rule is the concatenation `(1−α)·moment-fit ∪ α·tensor`. Both
        # summands are exact (moment-fit integrates the physical part, tensor
        # Gauss the whole cell), so the total is exact for the α-FCM integrand;
        # unlike the strict physical-only rule it gives *every* mode — including
        # the interior bubbles whose support lies entirely in the fictitious
        # corner of a thin cut — a nonzero, well-posed contribution. When Ω ∩ box
        # carries no volume rule at all (a degenerate sliver, whose physical
        # contribution is negligible by construction — the domain volume is
        # captured by the well-fitted cells), keep the α·tensor part alone so the
        # cell's dofs are α-stabilised rather than left singular.
        alpha_weights = base_rule.weights .* physical.alpha
        if status === :empty
            return RegionQuadrature{D,T}(:cut_alpha_failed, base_rule.points, alpha_weights),
                   residual
        end
        points = vcat(ref_pts, base_rule.points)
        weights = vcat(ref_ws .* (one(T) - physical.alpha), alpha_weights)
        return RegionQuadrature{D,T}(kind, points, weights), residual
    end
end

# ── Admissible box partition ──────────────────────────────────────────────────

# Form non-degenerate intervals between successive entries of a sorted,
# canonicalised coordinate list. Pairs closer than `tol.merge` are
# dropped — `merge_coordinates` should already have collapsed them, but
# the check here is a safety net.
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

# The whole cell grid of a level: the default scan range below, and the only
# one the volume and surface partitions ever want.
_all_level_cells(level) = CartesianIndices(level.mesh.cells)

# Per-axis live element-boundary coordinates of one level: for each axis, the
# coordinates that bound at least one *active* cell of `level` whose index lies
# in `cells`. This is the projection of the level's live region onto the axes,
# and it replaces the level's whole node grid in every partition the package
# builds. One pass over the mask marks all `D` axes, so a plan pays for a
# level's mask once, not once per axis.
#
# *Why the partition does not change.* A coordinate is dropped only when it
# separates two slabs of cells, neither of which is live on this level. It is
# then the face of no live cell here, and the level's coverage signature
# (`_coverage_signature`) is the same on both sides of it: either the line is
# not a node of this mesh, so both sides sit inside one cell, or it is a node
# whose two neighbouring cells are both dead, so both sides read `0`. A line the
# *merged* set loses is one that no level keeps, so every level's signature is
# constant across it and the greedy merge in `_merged_boxes` fuses the two slabs
# it separates anyway. The emitted box list comes out the same, in the same
# order — and the candidate grid now follows the active region instead of the
# finest participating mesh, which is where the cost of a deep `ladder` sat.
#
# One class of box does change: a region *no* level covers, which arises only
# where no level is live at all. The partition then stops at the outermost live
# face instead of tiling the dead remainder. Such a box has no parent, and every
# consumer of the partition drops it before anything else — `:any_parent` and
# `:all_levels` by definition, `projection.jl` for want of a target parent, and
# a caller's own predicate because a parentless region has no quadrature order
# to build a rule from in the first place.
#
# The remaining caveat is canonicalisation, not geometry: `merge_coordinates` keeps
# the *first* representative of a group within `tol.merge`. On a nested stack
# the shared nodes are bit-identical (`_mesh_axes` divides equal rationals), so
# dropping a dead line cannot change which value represents a group and the plan
# is bit-identical. On a non-nested overlay a dead line sitting within
# `tol.merge` below a live line of another level *was* that group's
# representative, so the canonical coordinate can move by at most `tol.merge`.
#
# `cells` exists for the boundary-facet partition in `dirichlet.jl`, which can
# only be parented by cells with a face on the facet; every other caller takes
# the default. An unmasked level answers from its axes directly, without
# touching a mask it does not have.
function _live_axis_coordinates(level, ::Val{D}, cells::CartesianIndices{D}) where {D}
    nodes = boundary_coordinates(level.mesh)
    mask = level.mask
    lo, hi = first(cells).I, last(cells).I
    mask === nothing && return ntuple(d -> nodes[d][lo[d]:(hi[d]+1)], D)
    live = ntuple(d -> falses(length(nodes[d])), D)
    # `findall` on a `BitVector` walks 64 cells per machine word, so this costs
    # the mask's size in *words* plus its active cells — not its cell count.
    # It runs on the flattened mask rather than the `D`-dimensional one on
    # purpose: `findall` returns `Int` for a vector and `CartesianIndex{D}` for
    # `D > 1`, and one `D`-generic loop body cannot consume both. `vec` on a
    # `BitArray` is a reshape, so nothing is copied.
    cartesian = CartesianIndices(mask.on)
    for i in findall(vec(mask.on))
        c = cartesian[i]
        c in cells || continue
        for d in 1:D
            live[d][c.I[d]] = true
            live[d][c.I[d] + 1] = true
        end
    end
    return ntuple(d -> nodes[d][live[d]], D)
end

# Per-axis merged element-boundary coordinates: gather each axis's live
# element-boundary coordinates from every level (`_live_axis_coordinates`) and
# canonicalise them with `merge_coordinates`, so two meshes' nearly-coincident
# boundaries collapse to a single shared coordinate. Shared by the
# integration-region partition (`_axis_intervals`), the physical-facet partition
# (`_boundary_facet_regions` in dirichlet.jl) and the surface grid lines
# (`_grid_lines_for_levels` in surface.jl).
function _merged_axis_coordinates(levels, ::Val{D}, tol::GeometryTolerance{T},
                                  cells=_all_level_cells) where {D,T}
    per_level = map(level -> _live_axis_coordinates(level, Val(D), cells(level)), levels)
    return ntuple(D) do d
        merged = T[]
        for live in per_level
            append!(merged, live[d])
        end
        merge_coordinates(merged, tol)
    end
end

# Per-axis intervals on which the admissible-box partition is built: turn
# each axis's merged element-boundary coordinates
# (`_merged_axis_coordinates`) into non-degenerate intervals.
function _axis_intervals(levels::Tuple, ::Val{D}, tol::GeometryTolerance{T}) where {D,T}
    coords = _merged_axis_coordinates(levels, Val(D), tol)
    return ntuple(d -> _intervals_from_coordinates(coords[d], tol), D)
end

# Coverage signature at a point: the linear cell index per level, with `0`
# standing for "off this level's mesh" or "in an inactive cell" — both
# cases are treated identically because both mean the level contributes
# nothing here. Two boxes share a coverage signature iff they lie inside
# the same cell of every active level, so the smooth-integrand contract
# (one cell per axis per level) is preserved when merging axis-adjacent
# equal-signature boxes.
#
# The signature is returned as an `NTuple{N,Int}` rather than a
# `Vector{Int}` so the equality comparison in `_merged_boxes` stays on
# the stack. The previous `Vector{Int}` allocation per candidate box was
# the largest single allocation source in `integration_plan`.
function _coverage_signature(levels::Tuple, linmaps::Tuple, point::SVector{D,T},
                             tol::GeometryTolerance{T}) where {D,T}
    return map(levels, linmaps) do level, linmap
        cell = locate_cell(level.mesh, point; tol)
        cell === nothing || !is_active(level.mask, cell) ? 0 : linmap[cell]
    end
end

# Build the admissible integration boxes for `levels`. This is the box-partition
# half of the recipe documented in `CONTRIBUTING.md`'s "Integration regions"
# section — coordinates through greedy merge; `integration_plan` runs the parent
# resolution, the keep-criterion and the quadrature dispatch over the boxes
# emitted here:
#
#   1. Collect, per axis, every participating mesh's element-boundary
#      coordinates that bound an active cell of that mesh — all of them on an
#      unmasked level (`_axis_intervals` ⇒ `_merged_axis_coordinates` ⇒
#      `_live_axis_coordinates` ⇒ `merge_coordinates` ⇒
#      `_intervals_from_coordinates`). A coordinate no level keeps separates two
#      slabs of identical coverage signature, which step 4 would have merged
#      again, so dropping it costs no region and no exactness.
#   2. Form the Cartesian product of those intervals, giving the
#      *candidate* box index grid `CartesianIndices(ranges)`.
#   3. For each candidate, compute the per-level coverage signature at
#      the box midpoint (`_coverage_signature`).
#   4. Greedy axis-by-axis merging: starting from an unvisited candidate
#      box, extend the box along each axis in turn as long as the next
#      "slab" of candidates is (a) unvisited and (b) shares the signature
#      of the starting box. Mark every visited candidate so it is not
#      emitted again.
#
# The greedy merge keeps the number of regions close to the number of
# genuinely distinct cell-coverage patterns rather than producing one
# region per candidate. Without the merge, a small overlay's per-axis
# coordinates would slice the entire physical domain along the
# overlay axis, splitting far-away base cells into long chains of
# trivially-equal-signature sub-boxes for no integration benefit.
#
# Mathematical correctness: every cell inside a merged box has the same
# coverage signature, so the box lies inside one cell of every covering
# level. The integrand of any basis-function product on the box is then a
# smooth polynomial in the parent's reference frame, and the per-region
# Gauss rule remains exact. The greedy order of axes can change the
# *shape* of the resulting boxes (the same volume can be packed many
# ways) but never their *total* volume or their numerical contribution.
#
# Steps 3–4 are split across the three small helpers below rather than
# written as one flat loop. That split is load-bearing, not style. The
# greedy merge's running upper corner `hi` is reassigned on every
# extension *and* read inside `ntuple` closures; written in one frame,
# Julia heap-boxes it as a `Core.Box`, rebuilds the box on each extension
# and turns every read into a dynamic dispatch. In that form this single
# variable accounted for 95% of `integration_plan`'s allocations, on a
# path `prepare`, `move!` and `activate!` all run. The helpers restore the
# separation the compiler needs: every frame that *closes over* `hi`
# receives it as an argument and never writes it, and the one frame that
# writes it (`_extend_axis`) closes over nothing. Do not fold them back
# inline.

# Does the candidate slab one step past `hi` along axis `d` join the box
# seeded at `start`? It does when every candidate in the slab is unvisited
# and carries the seed signature `sig0`.
function _slab_matches(sigs, visited, start::NTuple{D,Int}, hi::NTuple{D,Int}, sig0, d::Int,
                       next::Int) where {D}
    slab = CartesianIndices(ntuple(e -> e == d ? (next:next) : (start[e]:hi[e]), D))
    return all(c -> !visited[c] && sigs[c] == sig0, slab)
end

# Extend `hi` along axis `d` for as long as successive slabs keep matching.
function _extend_axis(sigs, visited, ranges::NTuple{D,Int}, start::NTuple{D,Int}, hi::NTuple{D,Int},
                      sig0, d::Int) where {D}
    while hi[d] < ranges[d]
        next = hi[d] + 1
        _slab_matches(sigs, visited, start, hi, sig0, d, next) || break
        hi = Base.setindex(hi, next, d)
    end
    return hi
end

# Mark every candidate of the merged box `start:hi` visited and push the
# box's physical extent.
function _emit_merged_box!(boxes::Vector{AxisBox{D,T}}, visited, intervals, start::NTuple{D,Int},
                           hi::NTuple{D,Int}) where {D,T}
    for c in CartesianIndices(ntuple(e -> start[e]:hi[e], D))
        visited[c] = true
    end
    lower = SVector{D,T}(ntuple(d -> intervals[d][start[d]][1], D))
    upper = SVector{D,T}(ntuple(d -> intervals[d][hi[d]][2], D))
    push!(boxes, AxisBox{D,T}(lower, upper))
    return boxes
end

function _merged_boxes(levels, ::Val{D}, tol::GeometryTolerance{T}) where {D,T}
    intervals = _axis_intervals(levels, Val(D), tol)
    ranges = ntuple(d -> length(intervals[d]), D)
    boxes = AxisBox{D,T}[]
    any(==(0), ranges) && return boxes

    # Compute the per-level linear cell index map once and the
    # per-candidate coverage signature once. Both are reused inside the
    # greedy merge loop, where signature equality is the hot inner check.
    linmaps = map(level -> LinearIndices(level.mesh.cells), levels)
    sigs = map(CartesianIndices(ranges)) do index
        mid = SVector{D,T}(ntuple(d -> (intervals[d][index.I[d]][1] + intervals[d][index.I[d]][2]) /
                                       2, D))
        _coverage_signature(levels, linmaps, mid, tol)
    end

    visited = falses(ranges)
    for start in CartesianIndices(ranges)
        visited[start] && continue
        # Greedy axis-by-axis extension: extend along axis `d` as far as
        # the next slab of candidates matches the seed signature, then
        # repeat for the next axis. The merge never backtracks.
        hi = start.I
        for d in 1:D
            hi = _extend_axis(sigs, visited, ranges, start.I, hi, sigs[start], d)
        end
        _emit_merged_box!(boxes, visited, intervals, start.I, hi)
    end
    return boxes
end

# ── Merge tolerance vs. mesh spacing ──────────────────────────────────────────
#
# `tolerance.merge` is an absolute length, and `merge_coordinates` collapses
# per-axis element boundaries that separate by no more than it. Between
# levels that is exactly the intent: two meshes' near-coincident boundaries
# should become one shared coordinate rather than a sliver. Within a single
# level it is silent destruction. An axis whose own cell spacing `h`
# satisfies `h ≤ tolerance.merge` has its boundary list decimated —
# genuinely distinct cell faces collapse — and every surviving integration
# region then straddles two cells of that level. That breaks the
# one-cell-per-axis-per-level contract the whole region partition rests on,
# and nothing downstream can detect it: the integrand is no longer smooth on
# the region, yet the tensor Gauss rule is still applied to it. The observed
# outcomes are a converged, entirely wrong solution for a span-mode B-spline
# basis, and an unattributed `SingularException` for integrated Legendre.
#
# The check is the compatibility condition between the two lengths the user
# controls — mesh spacing and merge tolerance — not a floor on either one.
# Geometry at h = 1e-9 stays legal; it needs a `tolerance.merge` below it,
# in the same spirit as the region-scaled `atol` in `implicit.jl` and the
# diagonal-scaled sign tolerance in the MeshIO extension.
#
# The threshold is `tolerance.merge` itself and not a multiple of it: it
# mirrors `merge_coordinates`'s own `> tol.merge` keep-test, so exactly the
# configurations that lose a coordinate are rejected, and configurations
# that merely sit close to the tolerance keep working.
function _check_axis_spacing(levels, ::Val{D}, tol::GeometryTolerance{T}) where {D,T}
    for level in levels, d in 1:D
        coords = boundary_coordinates(level.mesh)[d]
        spacing = minimum(coords[i + 1] - coords[i] for i in 1:(length(coords)-1))
        spacing > tol.merge && continue
        # Report the scale-aware value the default would have had on this
        # axis, √eps(T) · extent, so the remedy is a number to paste.
        suggested = sqrt(eps(T)) * (coords[end] - coords[begin])
        throw(ArgumentError("level $(level.id) has cell spacing h = $spacing on axis $d, at or " *
                            "below the coordinate merge tolerance $(tol.merge); pass a smaller " *
                            "merge tolerance, e.g. `tolerance=GeometryTolerance(; " *
                            "merge=$suggested)`"))
    end
    return nothing
end

# ── Region parent coverage ────────────────────────────────────────────────────

# For an admissible region `box`, find every parent level whose active
# cell contains the box midpoint and record the box's geometry in that
# parent's reference frame. The midpoint is a deterministic representative
# of the box's interior because the box was built as a sub-rectangle of
# the cell along every axis — every interior point of the box belongs to
# the same parent cell, so midpoint containment matches "the whole box
# lies in this cell". Inactive cells (per the level's `LevelMask`) do not
# contribute.
function _parents_covering(levels, box::AxisBox{D,T}, tol::GeometryTolerance{T}) where {D,T}
    midpoint = center(box)
    parents = ParentRef{D,T}[]
    for level in levels
        cell = locate_cell(level.mesh, midpoint; tol)
        cell === nothing && continue
        is_active(level.mask, cell) || continue
        parent_box = cell_box(level.mesh, cell)
        local_box = AxisBox{D,T}(physical_to_reference(parent_box, box.lower),
                                 physical_to_reference(parent_box, box.upper))
        push!(parents, ParentRef{D,T}(level.id, cell, local_box, parent_box))
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

# ── Plan construction ─────────────────────────────────────────────────────────

# Fit every cut region's moment rule before the region loop consumes any of
# them, in parallel, and reduce the cache to exactly this plan's regions.
#
# At the default `moment_order_factor = 2` the fits *are* the plan: 99.9% of a
# 3D build, tens of seconds over a few dozen cut cells. They are also
# independent of one another — `moment_fit_rule` reads the immutable
# `PhysicalDomain` and the region box and touches nothing else — so the only
# obstacle to running them in parallel is the two `Dict`s, and neither is in
# reach of the parallel loop: classification runs first and serially (it is
# memoised, and a `Dict` is not thread-safe), each fit writes its own slot of a
# pre-sized vector, and the cache is filled afterwards, serially. The region
# loop then runs exactly as before, in region order, on nothing but cache hits,
# so region order, `residual_max` and the small-overlap bookkeeping are the
# serial path's, bit for bit.
#
# A user `cut_quadrature` is fitted serially instead. It is caller code under
# no thread-safety contract — the package's own test rule counts its calls
# through a shared `Ref` — so the package does not unilaterally run it on N
# threads. It still gets the dedup and the cross-plan reuse.
#
# `live` is the plan's whole cut-key set, not just its misses, because
# restricting the cache afterwards is what keeps a cache threaded through a long
# `move!` sweep from holding the sweep's worth of rules. On exit the cache holds
# only the *boxes* this plan cuts — but at every moment order fitted at one of
# them, which is where the eviction rule differs from the lookup key. See the
# `filter!` below for why the two are not the same rule, and what the difference
# costs.
function _prefit_cut_rules!(cache::_MomentFitCache{D,T}, V::Space{D,T}, admissible,
                            classify_cache::_ClassifyCache{D,T}) where {D,T}
    physical = V.physical
    physical === nothing && return cache
    live = Set{_MomentFitKey{D,T}}()
    for (box, parents) in admissible
        classify_cell(physical, box, classify_cache) === :cut || continue
        push!(live, (box.lower, box.upper, _moment_order_for_region(V, parents)))
    end
    pending = _MomentFitKey{D,T}[key for key in live if !haskey(cache, key)]
    fitted = Vector{_MomentFitValue{D,T}}(undef, length(pending))
    fit(key) = _fit_moment_rule(physical, AxisBox{D,T}(key[1], key[2]), key[3])
    if physical.cut_quadrature === nothing
        Threads.@threads for i in eachindex(pending)
            fitted[i] = fit(pending[i])
        end
    else
        map!(fit, fitted, pending)
    end
    for i in eachindex(pending)
        cache[pending[i]] = fitted[i]
    end
    # Evict by box, not by full key. The bound the eviction exists for is the
    # geometric one — a `move!` sweep must not leave the cache holding every box
    # the overlay ever passed over — and a box that has left the plan takes all
    # of its rules with it either way, so keying the eviction on the box alone
    # costs that bound nothing: a move changes which boxes are cut, never the
    # moment order at one of them.
    #
    # What it buys is the case full-key eviction gets wrong. An order-elevated
    # twin of the same space — `estimate`'s `V⁺`, `adaptivity.jl` — has exactly
    # this plan's boxes at a higher moment order, and under full-key eviction
    # the two orders evict each other, so every `estimate` / `adapted`
    # alternation refits both from scratch. Measured on a 3D sphere probe (base
    # 4³, order 2, depth 1, 80 cut regions, 4 threads, -O0): a repeated
    # `estimate` cost 38.7 s and the step after it 1.9 s; evicting by box they
    # cost 0.006 s and 0.03 s, with the elevated order fitted once.
    #
    # What it costs, stated plainly because it is not free: a box that stays
    # live keeps one entry per distinct moment order ever fitted there. Under a
    # `move!` sweep that is flat, since orders do not change there. Under an hp
    # loop it is not: `refine` raises the order at a live cut cell, the box does
    # not leave the plan, and the entries at it accumulate *per cycle*. Bounded
    # above — the key's order is `moment_order_factor × cell_order` and `refine`
    # caps `cell_order` at `pmax` — but not by one plan's worth.
    #
    # The trade, measured on one configuration: four `estimate` → `refine` →
    # `adapted` cycles on a 3D sphere (ladder base 4³, order 1, depth 1, 32 cut
    # boxes) grew the cache from 32 entries / 0.035 MiB to 96 entries over the
    # same 32 boxes / 0.61 MiB, and the next step's plan build cost 0.011 s
    # against 11.9 s with no reuse. An h-step resets the accumulation where it
    # lands, because the boxes it splits do leave the plan.
    live_boxes = Set((key[1], key[2]) for key in live)
    filter!(entry -> (entry.first[1], entry.first[2]) in live_boxes, cache)
    return cache
end

"""
    integration_plan(V::Space; tolerance=GeometryTolerance(T),
                               criterion=:any_parent,
                               classify_cache=_ClassifyCache{D,T}(),
                               moment_fit_cache=_MomentFitCache{D,T}()) -> IntegrationPlan

Build the [`IntegrationPlan`](@ref) for a [`Space`](@ref). The plan carries
every admissible integration region the assembly path will iterate over,
together with the small-overlap / cut-region / moment-fit diagnostics.

The construction follows the recipe in `CONTRIBUTING.md`'s "Integration
regions" section:

  0. Per level, per axis: check that the mesh spacing exceeds
     `tolerance.merge`. An axis at or below it would have its own cell
     boundaries collapsed by step 1, silently, so this combination is
     rejected with an `ArgumentError` naming the axis, its spacing, and
     the tolerance.
  1. Per axis: collect every level's element-boundary coordinates that
     bound an active cell of that level — all of them on an unmasked
     level — and canonicalise them via `merge_coordinates` so two
     meshes' near-coincident coordinates collapse to a single shared
     boundary. A coordinate no level keeps is the face of no active
     cell, so every level's coverage signature is constant across it and
     the partition is the one the full grids would have produced: the
     candidate grid follows the active region rather than the finest
     participating mesh.
  2. Form the Cartesian product of the resulting intervals to get
     candidate boxes; greedily merge axis-adjacent candidates that share
     a coverage signature (every cell inside the merged box lies inside
     the same cell of every active level).
  3. For each kept candidate, identify the covering parent cells via
     midpoint containment, build the box's geometry in each parent's
     reference frame, and check the `criterion`.
  4. Dispatch on the `PhysicalDomain` classification to pick the region
     quadrature: tensor Gauss for `:full`, α-scaled tensor Gauss for
     `:fictitious + alpha > 0`, and on `:cut` the NNMF moment-fit — or
     the domain's `cut_quadrature` callable, when it supplies one, in the
     fit's place. Drop the region entirely under strict-α (`alpha = 0`
     and `:fictitious`).
  5. Record a [`SmallOverlap`](@ref) for every region whose physical
     volume falls below `tolerance.small_volume`, and track the maximum
     NNMF residual observed across cut regions.

Keyword arguments:

  - `tolerance` — geometry tolerance applied at every step
    (`merge_coordinates`, midpoint containment, small-volume reporting).
    Its `merge` field is an absolute length, so a mesh finer than it on
    any axis is rejected rather than silently decimated (step 0). Small
    geometry is fully supported; it needs a `merge` below its own spacing.
  - `criterion` — `:any_parent` (default), `:all_levels`, or a function
    `parents -> Bool`. See `_criterion_satisfied` for the semantics. The
    predicate form is an escape hatch for a coverage rule the two symbols do
    not express; it is reached through `prepare(problem; criterion = …)` and,
    like every plan option, survives `move!` and `activate!` because
    [`Model`](@ref) replays its captured `plan_options`.
  - `classify_cache` — optional classification cache shared with the
    cell-level fictitious fold. Pass the same cache through
    `_apply_physical_fold` so the per-cell verdicts computed there can
    be reused at the region level.
  - `moment_fit_cache` — optional cut-region rule cache, keyed by region
    bounds and moment order. Pass the same cache across successive plans
    for one space — [`Model`](@ref) does, through every `move!` and
    `activate!` — so a cut region the mutation left unchanged reuses its
    rule instead of being refitted. On return the cache is restricted to the
    boxes *this* plan cuts: a box that has left the plan is dropped with all
    of its rules, so a `move!` sweep cannot grow the cache. A box that stays
    live keeps one entry per distinct moment order fitted there, so an
    order-elevated twin of the same space and this plan do not evict each
    other — and, where the order at a live box keeps rising (an hp loop),
    entries accumulate with the cycles rather than staying at one plan's
    worth. The accumulation is bounded above by `moment_order_factor × pmax`
    per axis. On a 3D sphere with 32 cut regions, four hp cycles took the
    cache from 0.035 MiB to 0.61 MiB and the next plan build from 11.9 s
    (no reuse) to 0.011 s.

The function is called by `prepare`, `move!`, and `_update_mask!`, all in
`src/model.jl`, which own the plan a [`Model`](@ref) carries; `assembly.jl` and
`postprocessing.jl` only read that cached plan back through the
[`integration_plan`](@ref) / [`integration_plans`](@ref) accessors. The
projection path in `projection.jl` builds no plan at all: it reuses this file's
admissible-box partition (`_merged_boxes`, `_parents_covering`) directly and
keeps every box the target covers, without a `criterion`.
"""
function integration_plan(V::Space{D,T}; tolerance=GeometryTolerance(T), criterion=:any_parent,
                          classify_cache=_ClassifyCache{D,T}(),
                          moment_fit_cache=_MomentFitCache{D,T}()) where {D,T}
    # Reject a merge tolerance that cannot resolve the meshes it is about to
    # canonicalise, before a single region is built. See `_check_axis_spacing`.
    _check_axis_spacing(V.levels, Val(D), tolerance)

    regions = VolumeRegion{D,T}[]
    small_overlaps = SmallOverlap{T}[]
    tensor_cache = Dict{NTuple{D,Int},TensorQuadrature{D,T}}()
    alpha_cache = Dict{NTuple{D,Int},Vector{T}}()
    domain_volume = volume(V.domain)
    residual_max = 0.0

    # Find covering parents and apply the keep-criterion before touching the
    # (potentially expensive) quadrature dispatch, then fit every cut region's
    # rule up front. Both passes leave the region loop below alone: it still
    # walks the admissible regions in construction order, and every moment fit
    # it asks for is a cache hit. See `_prefit_cut_rules!`.
    admissible = Tuple{AxisBox{D,T},Vector{ParentRef{D,T}}}[]
    for box in _merged_boxes(V.levels, Val(D), tolerance)
        parents = _parents_covering(V.levels, box, tolerance)
        _criterion_satisfied(parents, length(V.levels), criterion) &&
            push!(admissible, (box, parents))
    end
    _prefit_cut_rules!(moment_fit_cache, V, admissible, classify_cache)

    for (box, parents) in admissible
        # Dispatch on PhysicalDomain classification; a `nothing`
        # quadrature signals a strict-α fictitious drop.
        quadrature, residual = _build_region_quadrature(V, box, parents, tensor_cache, alpha_cache,
                                                        moment_fit_cache, classify_cache)
        quadrature === nothing && continue
        residual > residual_max && (residual_max = Float64(residual))

        # Record the region. Small-overlap diagnostics are accumulated
        # rather than acted on: the package exposes the warning, the
        # caller decides whether to stabilise.
        region_volume = volume(box)
        push!(regions, VolumeRegion{D,T}(box, parents, quadrature))
        if region_volume <= tolerance.small_volume
            relative_volume = iszero(domain_volume) ? zero(T) : region_volume / domain_volume
            push!(small_overlaps,
                  SmallOverlap{T}(length(regions), region_volume, relative_volume, length(parents)))
        end
    end

    min_volume = isempty(regions) ? zero(T) : minimum(volume(region.box) for region in regions)
    min_relative_volume = iszero(domain_volume) ? zero(T) : min_volume / domain_volume
    return IntegrationPlan{D,T}(regions, tolerance, length(small_overlaps), small_overlaps,
                                min_volume, min_relative_volume, residual_max)
end
