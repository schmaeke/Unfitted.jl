# ── Types ─────────────────────────────────────────────────────────────────────

"""
    ParentRef{D,T}(level, cell, local_box, parent_box)

Reference to one parent cell of an [`VolumeRegion`](@ref). Carries
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
function _parent_quadrature_counts(::Val{D}, parents, level_lookup) where {D}
    return ntuple(D) do d
        maximum(parents) do parent
            level = level_lookup(parent.level)
            recommended_quadrature_order(level.basis, level.order)[d]
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
# `moment_order_factor = 2` on `V.physical`, the basis spans degree
# `2 × max(parent.order)` per axis — matching what tensor Gauss
# integrates exactly on `:full` regions and so bringing cut-region
# integrand exactness to the same level as the rest of assembly. Users
# whose integrand is only degree-`p` (linear forms / sources) can set the
# factor to 1, halving the basis at the cost of giving up bilinear-form
# exactness.
function _moment_order_for_region(V::Space{D}, parents) where {D}
    factor = V.physical.moment_order_factor
    return ntuple(D) do d
        factor * maximum(p -> _level_by_id(V, p.level).order[d], parents)
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
# The cache is per-plan, where `V.physical` is fixed; the cache key need
# not include it.
const _MomentFitKey{D,T} = Tuple{SVector{D,T},SVector{D,T},NTuple{D,Int}}
const _MomentFitValue{D,T} = Tuple{Vector{SVector{D,T}},Vector{T},T,Symbol}

function _cached_moment_fit!(cache::Dict{_MomentFitKey{D,T},_MomentFitValue{D,T}},
                             physical::PhysicalDomain, box::AxisBox{D,T},
                             moment_order::NTuple{D,Int}) where {D,T}
    key = (box.lower, box.upper, moment_order)
    return get!(cache, key) do
        rule = physical.cut_quadrature
        phys_pts, phys_ws, residual, status = rule === nothing ?
                                              moment_fit_rule(physical, box, moment_order;
                                                              target_residual=physical.target_residual) :
                                              rule(physical, box, moment_order)
        jac = volume(box) / convert(T, 2^D)
        ref_pts = [physical_to_reference(box, p) for p in phys_pts]
        ref_ws = T[w / jac for w in phys_ws]
        (ref_pts, ref_ws, T(residual), status)
    end
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
                                  moment_fit_cache::Dict{_MomentFitKey{D,T},_MomentFitValue{D,T}},
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

# Per-axis merged element-boundary coordinates: gather axis `d`'s
# element-boundary coordinates from every level's mesh and canonicalise
# them via `merge_coordinates`, so two meshes' nearly-coincident
# boundaries collapse to a single shared coordinate. Shared by the
# integration-region partition (`_axis_intervals`), the physical-facet
# partition (`_boundary_facet_regions` in dirichlet.jl), and the surface
# grid lines (`_level_grid_lines` in surface.jl).
function _merged_axis_coordinates(levels, d::Int, tol::GeometryTolerance{T}) where {T}
    coords = T[]
    for level in levels
        append!(coords, boundary_coordinates(level.mesh)[d])
    end
    return merge_coordinates(coords, tol)
end

# Per-axis intervals on which the admissible-box partition is built: turn
# each axis's merged element-boundary coordinates
# (`_merged_axis_coordinates`) into non-degenerate intervals.
function _axis_intervals(levels::Tuple, ::Val{D}, tol::GeometryTolerance{T}) where {D,T}
    return ntuple(D) do d
        _intervals_from_coordinates(_merged_axis_coordinates(levels, d, tol), tol)
    end
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

# Build the admissible integration boxes for `levels`. The construction
# follows the seven-step recipe documented in `CONTRIBUTING.md`'s
# "Integration regions" section:
#
#   1. Collect every participating mesh's element-boundary coordinates,
#      per axis (`_axis_intervals` ⇒ `merge_coordinates` ⇒
#      `_intervals_from_coordinates`).
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
        sig0 = sigs[start]
        hi = start
        # Greedy axis-by-axis extension: extend along axis `d` as far as
        # the next slab of candidates matches the seed signature, then
        # repeat for the next axis. The merge never backtracks.
        for d in 1:D
            while hi.I[d] < ranges[d]
                next = hi.I[d] + 1
                slab = CartesianIndices(ntuple(e -> e == d ? (next:next) : (start.I[e]:hi.I[e]), D))
                all(c -> !visited[c] && sigs[c] == sig0, slab) || break
                hi = CartesianIndex(ntuple(e -> e == d ? next : hi.I[e], D))
            end
        end
        for c in CartesianIndices(ntuple(e -> start.I[e]:hi.I[e], D))
            visited[c] = true
        end
        lower = SVector{D,T}(ntuple(d -> intervals[d][start.I[d]][1], D))
        upper = SVector{D,T}(ntuple(d -> intervals[d][hi.I[d]][2], D))
        push!(boxes, AxisBox{D,T}(lower, upper))
    end
    return boxes
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
#   - `:all_levels`  — keep the region only if every level covers it
#                      (the strict criterion used by full-superposition
#                      diagnostics).
#   - `criterion isa Function` — caller-supplied predicate, called as
#                      `criterion(parents)` and returning `Bool`. Used
#                      by projection to encode source / target coverage
#                      requirements.
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

"""
    integration_plan(V::Space; tolerance=GeometryTolerance(T),
                               criterion=:any_parent,
                               classify_cache=_ClassifyCache{D,T}()) -> IntegrationPlan

Build the [`IntegrationPlan`](@ref) for a [`Space`](@ref). The plan carries
every admissible integration region the assembly path will iterate over,
together with the small-overlap / cut-region / moment-fit diagnostics.

The construction follows the recipe in `CONTRIBUTING.md`'s "Integration
regions" section:

  1. Per axis: collect every active level's element-boundary coordinates
     and canonicalise them via `merge_coordinates` so two meshes'
     near-coincident coordinates collapse to a single shared boundary.
  2. Form the Cartesian product of the resulting intervals to get
     candidate boxes; greedily merge axis-adjacent candidates that share
     a coverage signature (every cell inside the merged box lies inside
     the same cell of every active level).
  3. For each kept candidate, identify the covering parent cells via
     midpoint containment, build the box's geometry in each parent's
     reference frame, and check the `criterion`.
  4. Dispatch on the `PhysicalDomain` classification to pick the region
     quadrature: tensor Gauss for `:full`, α-scaled tensor Gauss for
     `:fictitious + alpha > 0`, NNMF moment-fit for `:cut`. Drop the
     region entirely under strict-α (`alpha = 0` and `:fictitious`).
  5. Record a [`SmallOverlap`](@ref) for every region whose physical
     volume falls below `tolerance.small_volume`, and track the maximum
     NNMF residual observed across cut regions.

Keyword arguments:

  - `tolerance` — geometry tolerance applied at every step
    (`merge_coordinates`, midpoint containment, small-volume reporting).
  - `criterion` — `:any_parent` (default), `:all_levels`, or a function
    `parents -> Bool`. See `_criterion_satisfied` for the semantics.
  - `classify_cache` — optional classification cache shared with the
    cell-level fictitious fold. Pass the same cache through
    `_apply_physical_fold` so the per-cell verdicts computed there can
    be reused at the region level.

The function is called by `prepare`, `move!`, and `_update_mask!` in
`assembly.jl`, and once more by the projection plumbing in
`projection.jl` (with a wider `criterion`).
"""
function integration_plan(V::Space{D,T}; tolerance=GeometryTolerance(T), criterion=:any_parent,
                          classify_cache=_ClassifyCache{D,T}()) where {D,T}
    regions = VolumeRegion{D,T}[]
    small_overlaps = SmallOverlap{T}[]
    tensor_cache = Dict{NTuple{D,Int},TensorQuadrature{D,T}}()
    alpha_cache = Dict{NTuple{D,Int},Vector{T}}()
    moment_fit_cache = Dict{_MomentFitKey{D,T},_MomentFitValue{D,T}}()
    domain_volume = volume(V.domain)
    residual_max = 0.0

    for box in _merged_boxes(V.levels, Val(D), tolerance)
        # Find covering parents and apply the keep-criterion before
        # touching the (potentially expensive) quadrature dispatch.
        parents = _parents_covering(V.levels, box, tolerance)
        _criterion_satisfied(parents, length(V.levels), criterion) || continue

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
