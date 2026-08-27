# Prepared-problem lifecycle: the `Model` state container, its
# `AssemblyDiagnostics` record, the `prepare` constructor that turns a
# `Problem` into a model (folding the optional `PhysicalDomain`,
# building the integration plan, the dof layout, and the diagnostics),
# and the in-place mutators (`move!`, `activate!`, `deactivate!`) that
# bump the model's version counter and refresh the cached state.
#
# No assembly happens here — the global matrix and rhs are populated by
# `assemble!` in `assembly.jl`. Loaded between `problems.jl` (whose
# `Problem` this file owns the prepared form of) and `assembly.jl`
# (whose `assemble!` mutates `model.matrix` / `model.rhs`).
#
# Stale-solution detection contract: every mutator bumps
# `model.version`. A [`Solution`](@ref) carries the version it was
# computed against; reusing a solution after a version bump is caught
# by `_checked_coefficients` in `solvers.jl`.

# ── Assembly diagnostics ──────────────────────────────────────────────────────

"""
    AssemblyDiagnostics

Mutable diagnostics record carried on every [`Model`](@ref) and exposed
through [`diagnostics`](@ref). Updated in place by `prepare`, `move!`,
`activate!` / `deactivate!`, and `assemble!`. Every field is `Float64`-
or `Int`-typed for stability across scalar types.

Fields:

  - `dimension::Int` — spatial dimension `D`.
  - `active_unknowns::Int` — number of active (post-constraint) dofs.
  - `integration_regions::Int` — number of admissible integration
    regions in the model's plan.
  - `small_overlap_count::Int` and `small_overlaps::Vector` — number
    and per-region records of integration regions below
    `tolerance.small_volume`.
  - `min_integration_volume::Float64` — smallest region volume.
  - `min_relative_integration_volume::Float64` — same, divided by
    `volume(V.domain)`.
  - `symmetry_residual::Float64` — `‖A − Aᵀ‖_F` for the assembled
    matrix; `NaN` until `assemble!` runs, and `0.0` for symmetric forms
    (assembled lower-triangular and mirrored — bit-exact).
  - `condition_estimate::Float64` — `cond(Matrix(A))` for systems with
    `n ≤ 256`; `NaN` otherwise (computing the condition number of a
    large matrix is too expensive for a default diagnostic).
  - `solver::Symbol` — solver tag recorded by [`solve!`](@ref).
  - `inactive_cell_counts::Vector{Int}` — per-level count of cells
    deactivated by `LevelMask`. Includes both user-provided masks and
    the strict-α fictitious fold from `physical.jl`.
  - `reduced_mode_counts::Vector{Int}` — per-level count of high-order
    and dedup modes eliminated by order reduction in covered regions
    (`reduce_order`), in the same field-then-level order as
    `inactive_cell_counts`.
  - `cut_region_count::Int`, `fit_failure_count::Int`,
    `moment_fit_residual_max::Float64` — FCM moment-fit statistics from
    the integration plan. `fit_failure_count` counts every cut region
    whose fit missed `_FIT_FAILURE_RESIDUAL`, whatever rule the region
    ended up carrying.
  - `cut_fallback_count::Int`, `cut_fallback_points::Int` — the subset of
    those failures that fell back to the raw Saye volume rule
    (`:cut_fallback`), and the total number of quadrature points those
    regions carry. A successful fit yields ≈ nbasis points, so the second
    number is the price of the safety net; a nonzero count means the mesh
    is too coarse for the geometry in those cells and should be refined.
    See [`moment_fit_rule`](@ref).
  - `facet_region_count::Int` — total number of [`FacetRegion`](@ref)s
    cached on the model, summed across every cached selector
    (Dirichlet conditions and any `block`/`loadform` carrying
    `on::BoundarySelector`). Zero for problems with no physical-
    boundary integration.
  - `surface_region_count::Int` — total number of
    [`SurfaceRegion`](@ref)s cached on the model, summed across every
    cached [`BoundaryMesh`](@ref). Zero for problems with no
    immersed-boundary integration.
  - `interface_region_count::Int` — total number of
    [`InterfaceRegion`](@ref)s cached on the model, summed across every
    cached coupling [`Interface`](@ref). Zero for problems with no
    multi-domain interface coupling.
"""
mutable struct AssemblyDiagnostics
    dimension::Int
    active_unknowns::Int
    integration_regions::Int
    small_overlap_count::Int
    small_overlaps::Vector{SmallOverlap{Float64}}
    min_integration_volume::Float64
    min_relative_integration_volume::Float64
    symmetry_residual::Float64
    condition_estimate::Float64
    solver::Symbol
    inactive_cell_counts::Vector{Int}
    reduced_mode_counts::Vector{Int}
    cut_region_count::Int
    fit_failure_count::Int
    moment_fit_residual_max::Float64
    cut_fallback_count::Int
    cut_fallback_points::Int
    facet_region_count::Int
    surface_region_count::Int
    interface_region_count::Int
end

# Promote the small-overlap records of an integration plan to `Float64`
# scalars so they fit the diagnostics' fixed-precision storage. The
# integration plan carries them in the model's scalar type `T`; widening
# to `Float64` here keeps `AssemblyDiagnostics` `T`-agnostic.
function _float_small_overlaps(records)
    return [SmallOverlap{Float64}(record.region, Float64(record.volume),
                                  Float64(record.relative_volume), record.cover_count)
            for record in records]
end

function AssemblyDiagnostics(; dimension=0, active_unknowns=0, integration_regions=0,
                             small_overlap_count=0, min_integration_volume=NaN,
                             min_relative_integration_volume=NaN, symmetry_residual=NaN,
                             condition_estimate=NaN, solver=:none,
                             small_overlaps=SmallOverlap{Float64}[], inactive_cell_counts=Int[],
                             reduced_mode_counts=Int[], cut_region_count=0, fit_failure_count=0,
                             moment_fit_residual_max=0.0, cut_fallback_count=0,
                             cut_fallback_points=0, facet_region_count=0, surface_region_count=0,
                             interface_region_count=0)
    return AssemblyDiagnostics(Int(dimension), Int(active_unknowns), Int(integration_regions),
                               Int(small_overlap_count), _float_small_overlaps(small_overlaps),
                               Float64(min_integration_volume),
                               Float64(min_relative_integration_volume), Float64(symmetry_residual),
                               Float64(condition_estimate), Symbol(solver),
                               Int[inactive_cell_counts...], Int[reduced_mode_counts...],
                               Int(cut_region_count), Int(fit_failure_count),
                               Float64(moment_fit_residual_max), Int(cut_fallback_count),
                               Int(cut_fallback_points), Int(facet_region_count),
                               Int(surface_region_count), Int(interface_region_count))
end

# Per-level count of cells deactivated by a `LevelMask` (mask and
# fictitious fold combined), across every distinct subdomain space in
# `problem_spaces` order — the same order the reindexed global level ids
# run. One entry per level, `0` for an unmasked level. Reported by
# `diagnostics` so the user can see how many cells each level dropped.
function _inactive_cell_counts(spaces)
    return Int[level.mask === nothing ? 0 : count(!, level.mask.on)
               for V in spaces for level in V.levels]
end

# Per-level count of raws eliminated by order reduction (`:coverage` or `:dedup`),
# across every field of the system layout, in field-then-level order — mirroring
# `_inactive_cell_counts` so the two vectors line up entry-for-entry.
function _reduced_mode_counts(layout::SystemLayout)
    counts = Int[]
    for field in layout.fields
        dofs = field.dofs
        per_level = Dict{Int,Int}()
        for raw in eachindex(dofs.raw_keys)
            src = dofs.elimination_source[raw]
            (src === :coverage || src === :dedup) || continue
            lvl = dofs.raw_keys[raw].level
            per_level[lvl] = get(per_level, lvl, 0) + 1
        end
        for lvl in field.level_ids
            push!(counts, get(per_level, lvl, 0))
        end
    end
    return counts
end

# Scan the integration plan for NNMF-fit statistics, returning
# `(cut, failed, fallback, fallback_points)`. `:cut_fitted` are successful
# fits; `:cut_fallback` (the raw Saye volume rule), `:cut_failed` (empty
# Ω ∩ box at α = 0) and `:cut_alpha_failed` (the same at α > 0) all failed
# to fit. Every kind counts as a "cut" region, and every non-fitted kind
# counts as a fit failure — so `fit_failure_count` keeps meaning "the
# moment fit did not reach `_FIT_FAILURE_RESIDUAL`" regardless of which
# rule the region ended up carrying.
#
# `:cut_custom` — a region integrated by the domain's own `cut_quadrature`
# rule instead of the fit (see `PhysicalDomain`) — is a cut region but never
# a fit failure: no fit ran, so there is nothing for the residual-based
# counter to report. A custom rule that wants to be counted as a failure
# says so by returning `status === :empty`, which lands the region in
# `:cut_failed` / `:cut_alpha_failed` instead.
#
# `fallback_points` sums the quadrature points of the `:cut_fallback`
# regions. That is the assembly cost of the safety net, and the number a
# caller should watch: it is bounded only by the volume rule's own point
# count, not by nbasis.
function _cut_region_stats(plan::IntegrationPlan)
    cut = 0
    failed = 0
    fallback = 0
    fallback_points = 0
    for region in plan.regions
        kind = region.quadrature.kind
        if kind === :cut_fitted || kind === :cut_custom
            cut += 1
        elseif kind === :cut_fallback
            cut += 1
            failed += 1
            fallback += 1
            fallback_points += length(region.quadrature.weights)
        elseif kind === :cut_failed || kind === :cut_alpha_failed
            cut += 1
            failed += 1
        end
    end
    return cut, failed, fallback, fallback_points
end

# Fold every plan-level statistic into a diagnostics record in place, and
# return it so call sites can chain. Shared by `prepare`, `move!`,
# `_update_mask!`, and `assemble!`.
#
# `plans` holds one integration plan per subdomain and must be non-empty.
# The aggregation is the identity on a single-domain problem: region / cut /
# small-overlap counts sum, minimum volumes take the global minimum, the
# moment-fit residual takes the global maximum, and the small-overlap records
# are concatenated so every subdomain's offending regions are reported.
function _set_plan_stats_multi!(diag::AssemblyDiagnostics, plans)
    diag.integration_regions = sum(length(p.regions) for p in plans; init=0)
    diag.small_overlap_count = sum(p.small_overlap_count for p in plans; init=0)
    diag.small_overlaps = reduce(vcat, (_float_small_overlaps(p.small_overlaps) for p in plans);
                                 init=SmallOverlap{Float64}[])
    diag.min_integration_volume = minimum(Float64(p.min_volume) for p in plans)
    diag.min_relative_integration_volume = minimum(Float64(p.min_relative_volume) for p in plans)
    cut = 0
    failed = 0
    fallback = 0
    fallback_points = 0
    for p in plans
        c, f, b, bp = _cut_region_stats(p)
        cut += c
        failed += f
        fallback += b
        fallback_points += bp
    end
    diag.cut_region_count = cut
    diag.fit_failure_count = failed
    diag.cut_fallback_count = fallback
    diag.cut_fallback_points = fallback_points
    diag.moment_fit_residual_max = maximum(p.moment_fit_residual_max for p in plans; init=0.0)
    return diag
end

function Base.show(io::IO, diagnostics::AssemblyDiagnostics)
    print(io, "AssemblyDiagnostics(D=", diagnostics.dimension, ", active=",
          diagnostics.active_unknowns, ", regions=", diagnostics.integration_regions, ", small=",
          diagnostics.small_overlap_count, ", solver=:", diagnostics.solver, ")")
end

# ── Model and lifecycle ───────────────────────────────────────────────────────

# Cached CSC sparsity pattern for a model's matrix assembly. Built once
# per `model.version` (keyed additionally by the `on`-tag region set and
# the symmetry flag) by `build_assembly_pattern` in `assembly.jl`; the
# numeric scatter pass fills a matching `nzval` buffer slot-for-slot, so
# a dof pair appearing in many integration boxes accumulates with `+=`
# into one slot — peak memory is the final nnz, not the per-box
# dense-block over-count.
#
# `colptr`/`rowval` are value-type-independent, so one pattern serves
# mass / stiffness / Newton-tangent and every re-assembly. For symmetric
# forms the pattern is the lower triangle (global row ≥ col), mirrored as
# `A + Aᵀ − diag` at the end. `key` is a hash of the ordered region-list
# set the pattern was built over, so a request touching a different set
# of `on=` selectors rebuilds rather than reusing a stale pattern.
# Defined here (rather than in `assembly.jl`) because it is cached on the
# `Model` and this file is included first.
#
# `gather_cache` memoises the threaded scatter's per-region-list `GatherPlan`
# (the deferred compute→gather layout built by `assembly.jl`), keyed by the
# region list's `objectid`. It lives on the pattern so it is invalidated for
# free: a structural change drops `model.pattern`, and the fresh pattern
# starts with an empty cache. Typed `Any` because `GatherPlan` is defined in
# `assembly.jl` (included after this file); retrieval goes through a function
# barrier, so the hot loop stays type-stable.
struct AssemblyPattern
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    symmetric::Bool
    key::UInt
    gather_cache::Dict{UInt,Any}
end
function AssemblyPattern(n, colptr, rowval, symmetric, key)
    AssemblyPattern(n, colptr, rowval, symmetric, key, Dict{UInt,Any}())
end

# ── Single-sided region-cache keys ────────────────────────────────────────────
#
# A single-sided region list (`FacetRegion`s for a `BoundarySelector`,
# `SurfaceRegion`s for a `BoundaryMesh`) is built against exactly *one*
# subdomain space: its parents live on that space's level-id block, so only
# that space's fields ever evaluate on it (`region_parents`). On a coupled
# problem the same `on=` target may legitimately be named by fields on
# different subdomains — `boundary(axis=1, side=:upper)` is a face of every
# subdomain that has one — and those are different sets of facets. The cache
# key is therefore the *pair* `(on, space)`, not the target alone.
#
# A plain `Tuple` already gives each half the equality it deserves, because
# `Tuple` composes its elements' `isequal` / `hash` elementwise:
#
#   * a `BoundarySelector` is a *description*, not an object with an identity,
#     and carries value `==` / `hash` (see `dirichlet.jl`), so two value-equal
#     selectors are one key;
#   * a `Space` — like a `BoundaryMesh` or an `Interface` — defines neither, so
#     Base's fallbacks (`===`, `objectid`) apply and each object is its own key.
#
# Identity is the right notion for the space half: `problem_spaces` already
# deduplicates spaces by `===`, so "a distinct discretisation" is package-wide
# synonymous with "a distinct `Space` object".
const RegionKey = Tuple{Any,Any}

"""
    Model{D,T,P}

Prepared problem, ready to assemble and solve. Mutable so that
[`move!`](@ref), [`activate!`](@ref), [`deactivate!`](@ref), and
[`assemble!`](@ref) can update its state without forcing the caller to
re-thread a fresh value through. Fields:

  - `problem::P` — the *effective* [`Problem`](@ref): the one every
    consumer reads, already folded against a `PhysicalDomain` (cells
    outside Ω deactivated via the strict-α path in `mesh.jl`'s
    `_apply_physical_fold`) and reindexed into disjoint level-id blocks.
  - `prefold_space::Space{D,T}` — the *pre-fold* space: the level masks
    exactly as the caller described them, before `_apply_physical_fold`
    intersected them with the geometry. It is the only record of what the
    user asked for, because the fold is not invertible — once it has run, a
    level's `mask` no longer distinguishes a cell the caller excluded from a
    cell the classifier found outside Ω. [`move!`](@ref) and [`moved`](@ref)
    relocate the overlay on *this* space and fold the result from scratch, so
    a move is exactly a [`prepare`](@ref) at the new box. Only the geometry
    is kept here: the forms, field channels and Dirichlet data a rebuild
    needs are read back from `problem`, which carries them unchanged through
    the fold. [`activate!`](@ref) / [`deactivate!`](@ref) record their cell
    flips here as well as on `problem`, since a flip is a caller choice the
    next `move!` must reproduce.
  - `version::Int` — bump counter for stale-solution detection. Every
    in-place mutation that invalidates the assembled state bumps this;
    a [`Solution`](@ref) carrying an older version raises on reuse.
  - `space_plans::Vector{IntegrationPlan{D,T}}` — cached integration
    plan per *distinct* participating discretisation, in the
    first-appearance order of [`problem_spaces`](@ref)`(problem)`. A
    single-domain problem carries one plan; a multi-domain (coupled)
    problem carries one per subdomain space, each built against that
    space's own physical fold. Populated by `prepare` and refreshed by
    every mutator. The field-to-plan routing is intrinsic: each field owns
    the regions on its space's level-id block (see `FieldLayout.level_ids`
    and `region_parents`), so no per-plan routing table is stored.
  - `moment_fit_caches::Vector{_MomentFitCache{D,T}}` — one cut-region
    moment-fit rule cache per distinct participating space, aligned with
    `space_plans`. Every mutator threads it back into `integration_plan`,
    so a cut region a `move!` or a mask flip leaves bit-identical reuses
    its rule instead of being refitted — by far the dominant cost of a 3D
    plan rebuild. Each cache is scoped to one space and therefore to one
    immutable [`PhysicalDomain`](@ref), so there is nothing to invalidate;
    `integration_plan` reduces it to the new plan's own regions on the way
    out, so it does not grow with the length of a `move!` sweep.
  - `dofs::SystemLayout{D,T}` — per-field dof layout and active
    enumeration.
  - `matrix::Union{Nothing,SparseMatrixCSC{T,Int}}` — assembled global
    matrix. `nothing` until [`assemble!`](@ref) populates it; cleared
    by every mutator.
  - `rhs::Union{Nothing,Vector{T}}` — assembled right-hand side. Same
    invalidation contract as `matrix`.
  - `facet_regions::Dict{RegionKey,Vector{FacetRegion{D,T}}}` — cache of
    physical-boundary [`FacetRegion`](@ref)s, keyed by
    the `RegionKey` pair `(selector, space)`. One entry per
    `BoundarySelector` **per subdomain space** that names it, whether
    through a Dirichlet condition or a `block`/`loadform` carrying
    `on::BoundarySelector`. The space is part of the key because a
    region list lives on one space's level-id block: two subdomains
    naming a value-equal selector own different facets and must not
    share an entry. Populated at `prepare` and refreshed by every
    mutator.
  - `surface_regions::Dict{RegionKey,Vector{SurfaceRegion{D,T}}}` — cache
    of immersed-boundary [`SurfaceRegion`](@ref)s, keyed the same way by
    `(mesh, space)`. One entry per [`BoundaryMesh`](@ref) per
    subdomain space naming it through a `block`/`loadform` carrying
    `on::BoundaryMesh`. The mesh half of the key matches by object
    identity (the `BoundaryMesh` value carries no canonical hash;
    identity match prevents two structurally-equal but distinct meshes
    from accidentally sharing a cache entry).
  - `interface_regions::IdDict{Any,Vector{InterfaceRegion{D,T}}}` —
    per-interface cache of two-sided [`InterfaceRegion`](@ref)s. One
    entry per [`Interface`](@ref) referenced by a coupling `block`
    (`couple`'s four blocks share one `Interface` object). Keyed by
    object identity, as for `surface_regions`.
  - `dirichlet_projections::Dict{Symbol,DirichletProjection{D,T}}` —
    per-field cache of the datum-independent half of the L² Dirichlet
    projection (see [`DirichletProjection`](@ref)). Empty at `prepare`
    and filled by the first [`update_dirichlet!`](@ref) on each field,
    which is the only path that re-projects an existing dof layout;
    every mutator rebuilds the layout and empties the cache with it.
  - `diagnostics::AssemblyDiagnostics` — diagnostics record, updated
    in place by every lifecycle event.
  - `pattern::Union{Nothing,AssemblyPattern}` — cached CSC sparsity
    pattern for matrix assembly. `nothing` until the first
    matrix assembly builds it; cleared by every mutator (same
    invalidation contract as `matrix`/`rhs`). Lets a Newton loop reuse
    the pattern and re-run only the numeric scatter.
  - `plan_options::NamedTuple` — the integration-plan keyword options
    (`tolerance`, `criterion`, …) captured at `prepare`. Every mutator
    (`move!`, `activate!`, `deactivate!`) rebuilds the plan with these,
    so a mutation reproduces the prepared plan rather than silently
    reverting to `integration_plan`'s defaults.
"""
mutable struct Model{D,T,P}
    problem::P
    prefold_space::Space{D,T}
    version::Int
    space_plans::Vector{IntegrationPlan{D,T}}
    moment_fit_caches::Vector{_MomentFitCache{D,T}}
    dofs::SystemLayout{D,T}
    matrix::Union{Nothing,SparseMatrixCSC{T,Int}}
    rhs::Union{Nothing,Vector{T}}
    facet_regions::Dict{RegionKey,Vector{FacetRegion{D,T}}}
    surface_regions::Dict{RegionKey,Vector{SurfaceRegion{D,T}}}
    interface_regions::IdDict{Any,Vector{InterfaceRegion{D,T}}}
    dirichlet_projections::Dict{Symbol,DirichletProjection{D,T}}
    diagnostics::AssemblyDiagnostics
    pattern::Union{Nothing,AssemblyPattern}
    plan_options::NamedTuple
end

# Pick out the Dirichlet conditions that apply to the field called
# `name` from a problem's `dirichlet` list: every unscoped condition
# (`condition.field === nothing`) applies to the only field of a
# single-field problem and raises on multi-field problems; named
# conditions match by name.
function _dirichlet_for_field(problem::Problem, name::Symbol)
    scoped = Any[]
    for condition in problem.dirichlet
        if condition.field === nothing
            length(problem.fields) == 1 ||
                throw(ArgumentError("multi-field Dirichlet conditions must specify a field"))
            push!(scoped, condition)
        elseif condition.field === name
            push!(scoped, condition)
        end
    end
    return scoped
end

# All levels across every distinct participating space, concatenated in
# `problem_spaces` order. Because `prepare` reindexes each space's level ids
# into a disjoint block, the concatenation has globally-unique, contiguous
# ids `1:N` — exactly what the assembly workspace's flat level-id banks and
# each field's dof layout agree on. For a single-domain problem this is just
# the one space's levels.
function _problem_levels(problem::Problem)
    levels = Any[]
    for V in problem_spaces(problem)
        append!(levels, V.levels)
    end
    return levels
end

# Fold each distinct space against its own `PhysicalDomain` and reindex its
# level ids into a disjoint global block, then rebuild the fields over the
# prepared spaces. Returns the effective problem, the prepared distinct
# spaces (in `problem_spaces` order), and the per-space classification caches
# (each threaded into its own space's integration plan). This is the
# multi-domain generalisation of `_apply_physical_fold_to_problem`: it folds
# and namespaces every subdomain independently so their cut-cell plans, dof
# blocks, and workspace banks never collide.
function _prepare_spaces(problem::Problem{D,T}) where {D,T}
    orig = problem_spaces(problem)
    prepared = Vector{Space}(undef, length(orig))
    caches = Vector{_ClassifyCache{D,T}}(undef, length(orig))
    remap = IdDict{Space,Space}()
    offset = 0
    for (i, V) in pairs(orig)
        cache = _ClassifyCache{D,T}()
        reindexed = _reindex_space_levels(_apply_physical_fold(V, cache), offset)
        prepared[i] = reindexed
        caches[i] = cache
        remap[V] = reindexed
        offset += level_count(reindexed)
    end
    new_fields = map(f -> field(f.name, remap[f.space]; components=component_count(f)),
                     problem.fields)
    effective = Problem(new_fields; blocks=problem.blocks, loads=problem.loads,
                        dirichlet=problem.dirichlet, symmetric=problem.symmetric)
    return effective, prepared, caches
end

"""
    system_layout(problem::Problem; tolerance=GeometryTolerance(T)) -> SystemLayout

Build the per-field [`DofLayout`](@ref)s for every field of `problem`
and assemble them into a [`SystemLayout`](@ref). Each field's active
dofs are enumerated independently and then offset into a global active
block, so fields occupy disjoint ranges of the global enumeration.
Fields over different spaces (multi-domain coupling) each build their
layout from their own [`Space`](@ref); the disjoint offsets make the
global enumeration a product space `V₁ × … × Vₙ`.

Called by [`prepare`](@ref) and the in-place mutators. End users do
not usually call this directly.
"""
function system_layout(problem::Problem{D,T}; tolerance=GeometryTolerance(T),
                       classify_caches=IdDict{Space,_ClassifyCache{D,T}}()) where {D,T}
    layouts = FieldLayout{D,T}[]
    by_name = Dict{Symbol,Int}()
    offset = 0
    for field in problem.fields
        # Reuse the field's space fold cache when `prepare` / `move!` provide it,
        # so the constraint pass never re-classifies fold-boundary cells.
        cache = get(() -> _ClassifyCache{D,T}(), classify_caches, field.space)
        layout = dof_layout(field.space; dirichlet=_dirichlet_for_field(problem, field.name),
                            tolerance, components=component_count(field), classify_cache=cache)
        # The field's (reindexed, contiguous) level-id block — how assembly
        # routes each region to its owning subdomain field without a `served` mask.
        level_ids = extrema(l.id for l in field.space.levels)
        push!(layouts,
              FieldLayout{D,T}(field.name, component_count(field), layout, offset,
                               level_ids[1]:level_ids[2]))
        by_name[field.name] = length(layouts)
        offset += active_unknowns(layout)
    end
    return SystemLayout{D,T}(layouts, by_name, offset, tolerance)
end

# Map each prepared subdomain space to its fold classification cache, for
# `system_layout` to reuse. Built from the aligned `spaces`/`caches` vectors
# `_prepare_spaces` returns (identity keys — each field's space is one of these).
function _caches_by_space(spaces, caches::Vector{_ClassifyCache{D,T}}) where {D,T}
    by_space = IdDict{Space,_ClassifyCache{D,T}}()
    for i in eachindex(spaces)
        by_space[spaces[i]] = caches[i]
    end
    return by_space
end

"""
    prepare(problem::Problem; tolerance=…, criterion=…) -> Model

Build a fresh [`Model`](@ref) for `problem`. Folds an optional
[`PhysicalDomain`](@ref) into the per-level cell masks (strict-α path),
constructs the [`IntegrationPlan`](@ref) sharing the cell-classification
cache, builds the per-field [`DofLayout`](@ref)s, fills the diagnostics
record, and returns the assembled state with `matrix = rhs = nothing`
(call [`assemble!`](@ref) or one of the public assembly wrappers next).

Forwarded keyword arguments go to `integration_plan`; see its docstring
for the full list.
"""
function prepare(problem::Problem{D,T}; kwargs...) where {D,T}
    effective_problem, spaces, caches = _prepare_spaces(problem)
    # Capture the user's integration-plan options (tolerance, criterion, …) so
    # the in-place mutators can reproduce this exact plan instead of reverting
    # to `integration_plan`'s defaults. The classification caches are
    # geometry-derived, not user options, so they are rebuilt per mutation.
    plan_options = (; kwargs...)
    tolerance = get(plan_options, :tolerance, GeometryTolerance(T))
    # One integration plan per distinct subdomain space, each sharing that
    # space's own cell-classification cache. The moment-fit caches, unlike the
    # classification ones, outlive this call: they are the model's, and every
    # mutator hands them back so an unchanged cut region is not refitted.
    fit_caches = _MomentFitCache{D,T}[_MomentFitCache{D,T}() for _ in eachindex(spaces)]
    space_plans = IntegrationPlan{D,T}[integration_plan(spaces[i]; plan_options...,
                                                        classify_cache=caches[i],
                                                        moment_fit_cache=fit_caches[i])
                                       for i in eachindex(spaces)]
    layout = system_layout(effective_problem; tolerance,
                           classify_caches=_caches_by_space(spaces, caches))
    facet_regions = _resolve_facet_regions(effective_problem, tolerance)
    surface_regions = _resolve_surface_regions(effective_problem, tolerance)
    interface_regions = _resolve_interface_regions(effective_problem, layout, tolerance)
    diag = AssemblyDiagnostics(dimension=D, active_unknowns=active_unknowns(layout),
                               inactive_cell_counts=_inactive_cell_counts(spaces),
                               reduced_mode_counts=_reduced_mode_counts(layout),
                               facet_region_count=_region_count(facet_regions),
                               surface_region_count=_region_count(surface_regions),
                               interface_region_count=_region_count(interface_regions))
    _set_plan_stats_multi!(diag, space_plans)
    return Model{D,T,typeof(effective_problem)}(effective_problem, problem.space, 1, space_plans,
                                                fit_caches,
                                                layout, nothing, nothing, facet_regions,
                                                surface_regions, interface_regions,
                                                Dict{Symbol,DirichletProjection{D,T}}(), diag,
                                                nothing, plan_options)
end

# Build the facet-region cache for a problem from every *site* that references
# a `BoundarySelector`: Dirichlet conditions and any block/load carrying
# `on::BoundarySelector`. Each site contributes the pair (selector, the space of
# the field naming it), and each distinct pair is resolved through
# `_facet_regions_for_selector` and stored, so two subdomains naming a
# value-equal selector get two entries, their own facets each. Pairs without any
# admissible regions (e.g. an overlay-only level mask emptying a face) end up
# with an empty list, not absent — callers can distinguish "selector with no
# regions" from "selector never referenced".
function _resolve_facet_regions(problem::Problem{D,T}, tolerance::GeometryTolerance{T}) where {D,T}
    regions = Dict{RegionKey,Vector{FacetRegion{D,T}}}()
    for (selector, space) in _referenced_facet_sites(problem)
        key = (selector, space)
        haskey(regions, key) && continue
        regions[key] = _facet_regions_for_selector(space, selector, tolerance)
    end
    return regions
end

# Yields every (selector, space) *site* referenced in `problem`: one entry per
# Dirichlet condition and per block/load carrying `on::BoundarySelector`, paired
# with the space of the field that names it. A condition with `field === nothing`
# (only legal on a single-field problem) and a form on a single-domain problem
# both land on the one representative space, which is that problem's only one.
function _referenced_facet_sites(problem::Problem)
    return Iterators.flatten((((c.boundary, _condition_space(problem, c))
                               for c in problem.dirichlet),
                              ((b.on, _field_space(problem, b.test_name))
                               for b in problem.blocks if b.on isa BoundarySelector),
                              ((l.on, _field_space(problem, l.test_name))
                               for l in problem.loads if l.on isa BoundarySelector)))
end

# The subdomain space a Dirichlet condition constrains. `field === nothing` is
# only reachable on a single-field problem (the multi-field constructors demand
# a name), where the representative space *is* that field's space.
function _condition_space(problem::Problem, condition)
    condition.field === nothing ? problem.space : _field_space(problem, condition.field)
end

# Build the surface-region cache for a problem from every (mesh, space) site
# referenced by a block or load's `on` tag, keyed like the facet cache — the
# mesh half of the key by object identity (a `BoundaryMesh` value carries no
# canonical hash, and identity match prevents two structurally-equal but
# distinct meshes from accidentally sharing an entry), the space half likewise.
# One mesh named by fields on two subdomains yields two entries, each cut
# against its own grid; a genuine *two-sided* coupling across a mesh is an
# [`Interface`](@ref), which carries both sides in one region.
function _resolve_surface_regions(problem::Problem{D,T},
                                  tolerance::GeometryTolerance{T}) where {D,T}
    regions = Dict{RegionKey,Vector{SurfaceRegion{D,T}}}()
    for (mesh, space) in _referenced_boundary_mesh_sites(problem)
        key = (mesh, space)
        haskey(regions, key) && continue
        regions[key] = _surface_regions_for_mesh(space, mesh, tolerance)
    end
    return regions
end

function _referenced_boundary_mesh_sites(problem::Problem)
    return Iterators.flatten((((b.on, _field_space(problem, b.test_name))
                               for b in problem.blocks if b.on isa BoundaryMesh),
                              ((l.on, _field_space(problem, l.test_name))
                               for l in problem.loads if l.on isa BoundaryMesh)))
end

# Build the interface-region cache for a problem from every `Interface`
# referenced by a coupling block's `on` tag (the four blocks a `couple` call
# emits share one `Interface` object → one cache entry). Each interface's field
# indices come from the dof `layout` (global field order) and its two subdomain
# spaces from the effective problem's fields; the two-sided regions are built by
# subdividing the interface mesh against the merged trace of both grids. Keyed by
# `IdDict` (object identity), as for the surface cache.
function _resolve_interface_regions(problem::Problem{D,T}, layout::SystemLayout{D,T},
                                    tolerance::GeometryTolerance{T}) where {D,T}
    regions = IdDict{Any,Vector{InterfaceRegion{D,T}}}()
    for iface in _referenced_interfaces(problem)
        haskey(regions, iface) && continue
        field_a = layout.by_name[iface.field_a]
        field_b = layout.by_name[iface.field_b]
        space_a = _field_space(problem, iface.field_a)
        space_b = _field_space(problem, iface.field_b)
        regions[iface] = _interface_regions(iface, space_a, space_b, field_a, field_b, tolerance)
    end
    return regions
end

function _referenced_interfaces(problem::Problem)
    return Iterators.flatten(((b.on for b in problem.blocks if b.on isa Interface),
                              (l.on for l in problem.loads if l.on isa Interface)))
end

# Total number of cached regions across a per-selector / per-mesh / per-interface
# region cache. Sums the list lengths of the facet, surface, or interface caches;
# used for the diagnostics `facet_region_count` / `surface_region_count` /
# `interface_region_count` fields.
_region_count(regions::AbstractDict) = sum(length, values(regions); init=0)

# Resolve one selector into its full list of `FacetRegion`s (the union
# of regions across every facet the selector covers). Used by
# `_resolve_facet_regions` and reused by consumers that need to look up
# a selector that was not pre-resolved at `prepare` time.
function _facet_regions_for_selector(V::Space{D,T}, selector::BoundarySelector,
                                     tolerance::GeometryTolerance{T}) where {D,T}
    regions = FacetRegion{D,T}[]
    for sides in _facets(selector, Val(D))
        append!(regions, _boundary_facet_regions(V, sides, tolerance))
    end
    return regions
end

# Rebuild `problem` over a new (moved / remasked) space, reusing the field
# channels, forms, and Dirichlet data. Shared rebuild rule behind `_moved_problem`
# and `_remasked_problem` (the geometry-fold rebuild goes through
# `_prepare_spaces`, which also reindexes level ids).
function _problem_with_space(problem::Problem, new_space::Space)
    new_fields = map(f -> field(f.name, new_space; components=component_count(f)), problem.fields)
    return Problem(new_fields; blocks=problem.blocks, loads=problem.loads,
                   dirichlet=problem.dirichlet, symmetric=problem.symmetric)
end

function Base.show(io::IO, model::Model{D}) where {D}
    print(io, "Model(D=", D, ", version=", model.version, ", active=", active_unknowns(model.dofs),
          ", regions=", diagnostics(model).integration_regions, ", assembled=",
          model.matrix !== nothing, ")")
end

# Condition-number estimator for small assembled matrices. Returns `NaN`
# above `max_size` since `cond(Matrix(sparse))` densifies the matrix and
# is only viable up to a few hundred unknowns. The 256 cap keeps the
# diagnostic free for unit tests and small examples while never
# silently dominating real-problem assembly cost.
function _condition_estimate(matrix::SparseMatrixCSC; max_size::Int=256)
    n = size(matrix, 1)
    n == 0 && return NaN
    n > max_size && return NaN
    return Float64(cond(Matrix(matrix)))
end

"""
    integration_plan(model::Model) -> IntegrationPlan

The **first** subdomain's cached integration plan (`first(model.space_plans)`).
Plans are built by [`prepare`](@ref) and refreshed by [`move!`](@ref) /
[`activate!`](@ref) / [`deactivate!`](@ref), so once a model is prepared every
assembly call uses the same plan the dof layout was built against. For a
single-domain model this is *the* plan; a coupled model holds one plan per
subdomain — use [`integration_plans`](@ref) to reach all of them.
"""
integration_plan(model::Model) = first(model.space_plans)

"""
    integration_plans(model::Model) -> Vector{IntegrationPlan}

Every subdomain's cached [`IntegrationPlan`](@ref), in [`problem_spaces`](@ref)
order. The assembly volume pass iterates these; each region is owned by the
fields on its subdomain intrinsically (`region_parents` via
`FieldLayout.level_ids`). For a single-domain model this is a one-element vector
whose only element is what [`integration_plan`](@ref) returns.
"""
integration_plans(model::Model) = model.space_plans

"""
    dof_layout(model::Model) -> SystemLayout

Cached per-field dof layout, including constraint matrices and the
active-enumeration table.
"""
dof_layout(model::Model) = model.dofs

"""
    active_unknowns(model::Model) -> Int

Number of active (post-constraint) dofs on `model` — the size of the
global system the solver sees, and the length of any coefficient
vector consumed by [`solution`](@ref), [`assemble!`](@ref), or an
external time integrator. Delegates to
`active_unknowns(model.dofs::SystemLayout)`.
"""
active_unknowns(model::Model) = active_unknowns(model.dofs)

# ── Model mutation ────────────────────────────────────────────────────────────

# Guard the positional-level mutators (`move!`, `activate!`, `deactivate!`)
# against multi-domain models. Those mutators address a level by its position
# in the single `model.problem.space`, which is ambiguous once a problem spans
# several subdomain spaces; supporting per-subdomain mutation is deferred.
# A single-domain model passes through silently.
function _assert_single_domain(model::Model, op::AbstractString)
    length(problem_spaces(model.problem)) == 1 ||
        throw(ArgumentError("$op is not yet supported for multi-domain (coupled) models"))
    return nothing
end

# Rebuild the model's problem with overlay `level` moved to box `to`, reusing
# the existing forms and boundary data on the moved space. Shared by `move!`
# and `moved`, both of which then hand the result to the same
# fold → reindex → plan path `prepare` runs.
#
# The geometry comes from `model.prefold_space`, never from the effective
# `model.problem.space`. The effective masks already carry the fictitious fold
# computed at the *old* overlay position, and `_apply_physical_fold` intersects
# (`active .& .!fictitious`), so folding them a second time can only ever remove
# cells: a cell dropped because the old box put it outside Ω would stay dropped
# after a move that puts it wholly inside. Starting from the pre-fold masks makes
# `move!` reproduce `prepare` at the new box exactly, which is the contract both
# mutators advertise.
#
# Everything that is *not* geometry — field names and component counts, blocks,
# loads, Dirichlet data — is read from `model.problem`, which carries all of it
# unchanged through the fold and is kept current by `update_dirichlet!`.
#
# Both callers are single-domain (`_assert_single_domain`), so `prefold_space`
# is *the* space and `_problem_with_space` re-homes every field onto the moved
# copy without ambiguity.
function _moved_problem(model::Model{D,T}, level::Integer, to::AxisBox{D,T}, tolerance) where {D,T}
    return _problem_with_space(model.problem,
                               moved_space(model.prefold_space; level, to, tolerance))
end

# Shared invalidation tail for `move!` / `_update_mask!`. Clears the
# cached matrix / rhs, rebuilds the facet-region cache against the
# (already-updated) space, builds a fresh diagnostics record, and folds
# the current integration plan's stats in. Assumes `model.problem`,
# `model.dofs`, and `model.space_plans` already reflect the new state.
#
# The Dirichlet projection cache is dropped rather than rebuilt: its unknown
# sets, facet regions and mass all belong to the dof layout the caller has
# just replaced, and the next `update_dirichlet!` rebuilds it against the new
# one.
function _invalidate_assembly!(model::Model{D,T},
                               tolerance::GeometryTolerance{T}=GeometryTolerance(T)) where {D,T}
    model.matrix = nothing
    model.rhs = nothing
    model.pattern = nothing
    empty!(model.dirichlet_projections)
    model.facet_regions = _resolve_facet_regions(model.problem, tolerance)
    model.surface_regions = _resolve_surface_regions(model.problem, tolerance)
    model.interface_regions = _resolve_interface_regions(model.problem, model.dofs, tolerance)
    diag = AssemblyDiagnostics(dimension=D, active_unknowns=active_unknowns(model.dofs),
                               inactive_cell_counts=_inactive_cell_counts(problem_spaces(model.problem)),
                               reduced_mode_counts=_reduced_mode_counts(model.dofs),
                               facet_region_count=_region_count(model.facet_regions),
                               surface_region_count=_region_count(model.surface_regions),
                               interface_region_count=_region_count(model.interface_regions))
    _set_plan_stats_multi!(diag, model.space_plans)
    model.diagnostics = diag
    return model
end

"""
    move!(model; level, to) -> Model

Move overlay `level` of a prepared `model` to box `to` in place,
rebuilding its integration plan (reusing the `tolerance`/`criterion`
captured at [`prepare`](@ref)) and dof layout, and invalidating any
assembled operators. The base level (level 1) cannot be moved.

`move!` overwrites the old state; to transfer a solution onto the
moved configuration, use [`moved`](@ref) to build a separate target
model and then [`transfer`](@ref).

Invalidation contract: bumps `model.version`, clears `model.matrix`
and `model.rhs`, rebuilds the integration plan and dof layout,
refreshes diagnostics. An outstanding [`Solution`](@ref) raises on
reuse.
"""
function move!(model::Model{D,T}; level::Integer, to::AxisBox{D,T}) where {D,T}
    _assert_single_domain(model, "move!")
    opts = model.plan_options
    tolerance = get(opts, :tolerance, GeometryTolerance(T))
    moved_p = _moved_problem(model, level, to, tolerance)
    # Same fold → reindex → per-space-plan path as `prepare` (one space here,
    # since `move!` is single-domain), so the folded plan and dof layout are
    # rebuilt identically instead of hand-rolled.
    effective_problem, spaces, caches = _prepare_spaces(moved_p)
    model.problem = effective_problem
    model.prefold_space = moved_p.space
    model.version += 1
    fit_caches = model.moment_fit_caches
    model.space_plans = IntegrationPlan{D,T}[integration_plan(spaces[i]; opts...,
                                                              classify_cache=caches[i],
                                                              moment_fit_cache=fit_caches[i])
                                             for i in eachindex(spaces)]
    model.dofs = system_layout(effective_problem; tolerance,
                               classify_caches=_caches_by_space(spaces, caches))
    return _invalidate_assembly!(model, tolerance)
end

"""
    moved(model; level, to) -> Model

Return a new prepared [`Model`](@ref) with overlay `level` moved to box
`to`, reusing the source problem's forms and boundary data. Unlike
[`move!`](@ref), the source model is left intact, so it can serve as
the source of a [`transfer`](@ref):

    target = moved(model; level=2, to=box((0.1,), (0.6,)))
    target_solution = transfer(solution, model, target)

Like [`move!`](@ref), `moved` addresses the level by its position in the
model's single space and is therefore restricted to single-domain models;
on a coupled model it throws `ArgumentError`.
"""
function moved(model::Model{D,T}; level::Integer, to::AxisBox{D,T}) where {D,T}
    # Same restriction as `move!`, and for the same reason: `level` indexes one
    # space's level tuple, which is ambiguous once a problem spans several
    # subdomain spaces. `_problem_with_space` re-homes *every* field onto the
    # moved space, so on a coupled model there is no answer to give rather than
    # a wrong one — hence a raise, not a pick.
    _assert_single_domain(model, "moved")
    opts = model.plan_options
    tolerance = get(opts, :tolerance, GeometryTolerance(T))
    return prepare(_moved_problem(model, level, to, tolerance); opts...)
end

# Rebuild `problem` with one level's mask replaced. Symmetric to
# `_moved_problem` but swaps the mask instead of the mesh.
function _remasked_problem(problem::Problem, level_index::Integer, mask)
    return _problem_with_space(problem, _remasked_space(problem.space, level_index, mask))
end

# Activate / deactivate `cells` on `level_index` and rebuild the
# model's reusable state. Same invalidation contract as `move!`:
# bumps `model.version`, rebuilds the integration plan, dof layout,
# and diagnostics, and clears any assembled matrix / rhs. Any
# outstanding `Solution` becomes stale.
function _update_mask!(model::Model{D,T}, level_index::Integer, cells, value::Bool) where {D,T}
    _assert_single_domain(model, "activate! / deactivate!")
    1 <= level_index <= length(model.problem.space.levels) ||
        throw(ArgumentError("level index $level_index out of bounds"))
    opts = model.plan_options
    tolerance = get(opts, :tolerance, GeometryTolerance(T))
    old_level = model.problem.space.levels[level_index]
    new_mask = _apply_mask_update(old_level.mask, old_level.mesh, cells, value)
    model.problem = _remasked_problem(model.problem, level_index, new_mask)
    # Record the same flip on the pre-fold space, applied to the *user* mask
    # rather than to the effective one. Two masks, two meanings, updated
    # independently: `problem` keeps the documented "operates on the effective
    # mask" semantics (activating a fictitious cell overrides the geometry here
    # and now), while `prefold_space` keeps a record of the caller's own choices
    # for the next `move!` to fold afresh. Copying the effective mask across
    # instead would launder fold-derived deactivations into user intent, and
    # leaving the pre-fold mask untouched would make the next `move!` discard
    # the caller's flips — including flips on levels the move does not touch.
    prefold_level = model.prefold_space.levels[level_index]
    model.prefold_space = _remasked_space(model.prefold_space, level_index,
                                          _apply_mask_update(prefold_level.mask,
                                                             prefold_level.mesh, cells, value))
    model.version += 1
    model.space_plans = [integration_plan(model.problem.space; opts...,
                                          moment_fit_cache=model.moment_fit_caches[1])]
    model.dofs = system_layout(model.problem; tolerance)
    return _invalidate_assembly!(model, tolerance)
end

"""
    activate!(model; level, cells) -> Model

Mark `cells` on `level` as active in place. `cells` is a selection of
cells, not a whole mask, and accepts the selector shapes the `active=`
keyword of [`space`](@ref) documents: an iterable of `CartesianIndex{D}`,
a predicate `(cell_box, cell_index) -> Bool`, or an `AbstractArray{Bool,D}`
matching the level's cell grid. Currently-active cells in the selection are
unchanged.

Rebuilds the integration plan, dof layout, and diagnostics; clears any
assembled matrix / right-hand side; bumps `model.version` so an
existing [`Solution`](@ref) raises on reuse. Mirrors the
[`move!`](@ref) invalidation contract.

When the model's space has a `physical_domain`, the mutator operates
on the *effective* mask (which already folds in the geometric
`:fictitious` classification). Activating a cell that the level set
classifies as fictitious therefore overrides the geometry until the
next [`move!`](@ref) or fresh [`prepare`](@ref).
"""
function activate!(model::Model{D,T}; level::Integer, cells) where {D,T}
    return _update_mask!(model, level, cells, true)
end

"""
    deactivate!(model; level, cells) -> Model

Mark `cells` on `level` as inactive in place. See [`activate!`](@ref)
for the accepted shapes of `cells`, the invalidation contract, and the
interaction with a model's `physical_domain`.
"""
function deactivate!(model::Model{D,T}; level::Integer, cells) where {D,T}
    return _update_mask!(model, level, cells, false)
end

"""
    active_cells(model; level) -> BitArray{D}

Return a `BitArray{D}` indicating which cells of `level` are active.
Returns a copy so caller mutations do not leak into the model. Levels
without a mask return an all-true array.
"""
function active_cells(model::Model; level::Integer)
    levels = model.problem.space.levels
    1 <= level <= length(levels) || throw(ArgumentError("level index $level out of bounds"))
    lvl = levels[level]
    lvl.mask === nothing && return trues(lvl.mesh.cells)
    return copy(lvl.mask.on)
end

# ── Public: refresh Dirichlet values without rebuilding the model ────────────

"""
    update_dirichlet!(model::Model, dirichlet) -> Model

Refresh the *values* of a prepared model's Dirichlet conditions in
place, reusing the existing integration plan, dof enumeration,
facet/surface region caches, and immersed-boundary moment-fit cache.

The argument `dirichlet` is a vector or tuple of
[`DirichletCondition`](@ref)s that must structurally match the
conditions the model was prepared with: the same number of
conditions in the same order, each with identical `boundary`,
`field`, and `component`. Only the `value` of each condition is
allowed to differ. Any structural mismatch throws `ArgumentError`
and asks the caller to rebuild through [`prepare`](@ref) instead.

The canonical use is the inner loop of a load-stepping driver:

```julia
problem = Problem((u,); dirichlet=[dirichlet(0.0; on=top, field=u),
                                   dirichlet(0.0; on=bottom, field=u)])
model = prepare(problem)
for step in 1:n_steps
    u_top = step * Δu
    update_dirichlet!(model, [dirichlet(u_top; on=top, field=u),
                              dirichlet(0.0;   on=bottom, field=u)])
    solution = solve!(model)  # or run a Newton loop here
end
```

For the same problem, calling `update_dirichlet!` is typically
orders of magnitude cheaper than re-running `prepare(problem)`
because the per-step rebuild of the integration plan and the
NNMF moment-fit cache is skipped — only the right-hand side of the
boundary-trace projection on the constrained-dof subspace is
reassembled and back-substituted.

# Semantics

  - The constrained-dof set is *not* rediscovered. It depends only on
    the boundary selector + field topology, both of which the
    structural check pins.
  - `layout.constrained_values` is refilled by solving `M c = b` for the
    new `b_i = ∫ g(x) φ_i dS`. The boundary mass `M`, and the sampled
    basis traces that `b` is accumulated from, depend on the mesh
    alone, so the first call caches them on the model as a
    [`DirichletProjection`](@ref) and every later call reuses them —
    exactly, not merely to within roundoff. The facet regions are
    therefore walked and the mass factorised once per mesh, not once
    per increment.
  - `model.matrix` and `model.rhs` are cleared so the next
    [`assemble!`](@ref) (or `assemble_vector` / `assemble_matrix`
    call) picks up the new constrained data through the standard
    column-elimination path.
  - `model.version` is *not* bumped: existing [`Solution`](@ref) and
    [`QuadField`](@ref) handles stay valid against the updated
    layout.
  - The model's stored `problem` is replaced with a fresh `Problem`
    carrying the new Dirichlet list, so `model.problem.dirichlet`
    reflects the current state for inspection and diagnostics.

# Errors

Throws `ArgumentError` if `dirichlet` is structurally incompatible
with `model.problem.dirichlet` (different length, or any condition
moves boundary / field / component). The error message points the
caller at [`prepare`](@ref).
"""
function update_dirichlet!(model::Model{D,T}, dirichlet) where {D,T}
    new_dirichlet = collect(dirichlet)
    _check_dirichlet_update_compatibility(model.problem.dirichlet, new_dirichlet)

    # Replace `model.problem` (immutable) with a fresh copy carrying the
    # new dirichlet list. The inner constructor sidesteps the validation
    # walk in `Problem(fields; …)` — fields, blocks, loads, and the
    # symmetric flag were already checked when the model was prepared.
    p = model.problem
    model.problem = typeof(p)(p.space, p.fields, p.blocks, p.loads, new_dirichlet, p.symmetric)

    # Re-project values into each field's existing DofLayout. The
    # constrained-dof set is unchanged, so `_project_dirichlet_values!`
    # only refills `layout.constrained_values` — it does not touch the
    # raw-dof index, the active enumeration, or any of the boolean
    # constraint masks. Each field is projected against its OWN space (a
    # coupled multi-domain problem gives every field an independent space);
    # `model.problem.space` is only the first field's and does not cover the
    # others' cells — mirror the per-field routing `prepare` uses.
    #
    # The structural check above pins every quantity the field's
    # `DirichletProjection` depends on, so the cached one — if this is not the
    # first update on the field — is handed straight back in and only the
    # right-hand side is rebuilt.
    for field_layout in model.dofs.fields
        field_dirichlet = _dirichlet_for_field(model.problem, field_layout.name)
        field_space = _field_space(model.problem, field_layout.name)
        cached = get(model.dirichlet_projections, field_layout.name, nothing)
        model.dirichlet_projections[field_layout.name] = _project_dirichlet_values!(field_layout.dofs,
                                                                                    field_space,
                                                                                    field_dirichlet,
                                                                                    cached)
    end

    # Clear cached operators. The RHS depends on the constrained values
    # via column elimination, so the next assembly must rebuild it. The
    # matrix would technically still be valid (Dirichlet only moves
    # entries into the RHS at assembly time), but dropping it too means
    # the next `solve!` cannot silently use a stale operator if the
    # caller also changes blocks or loads between steps.
    #
    # `model.pattern` is deliberately retained: `update_dirichlet!` only
    # rewrites constrained *values*, leaving the constrained-dof set — and
    # therefore `active_unknowns` and every region's `active_dofs` — fixed,
    # so the cached sparsity pattern stays structurally valid and the next
    # assembly reuses it (the version is not bumped either).
    model.matrix = nothing
    model.rhs = nothing

    return model
end

# ── Diagnostics reporting ─────────────────────────────────────────────────────

"""
    diagnostics(model::Model) -> AssemblyDiagnostics
    diagnostics(model::Model, solution; exact=nothing) -> NamedTuple

The one-argument form returns the model's cached `AssemblyDiagnostics`
record. The two-argument form composes a reproducibility-report
`NamedTuple` that bundles the diagnostics with per-level metadata,
solver residual from the solution, and (optionally) the relative L²
error against `exact(x)`. The result is the natural artefact to
include in PR descriptions, paper figures, and regression tests.
"""
diagnostics(model::Model) = model.diagnostics

# Compact per-level reproducibility report used inside
# `diagnostics(model, solution)`: enough information to know which
# basis, order, mode, cell count, and domain a level had at solve time.
function _level_report(level)
    return (; id=level.id, role=level.role, cells=level.mesh.cells, order=level.order,
            mode=level.mode, basis=basis_name(level.basis), domain=level.mesh.domain,)
end

function diagnostics(model::Model, solution; exact=nothing)
    _checked_coefficients(solution, model)
    diag = diagnostics(model)
    error = exact === nothing ? nothing : l2_error(solution, model, exact)
    return (; dimension=diag.dimension, levels=map(_level_report, _problem_levels(model.problem)),
            active_unknowns=diag.active_unknowns, raw_dofs=raw_dof_count(model.dofs),
            integration_regions=diag.integration_regions,
            facet_region_count=diag.facet_region_count,
            surface_region_count=diag.surface_region_count,
            interface_region_count=diag.interface_region_count,
            small_overlap_count=diag.small_overlap_count, small_overlaps=diag.small_overlaps,
            min_integration_volume=diag.min_integration_volume,
            min_relative_integration_volume=diag.min_relative_integration_volume,
            inactive_cell_counts=diag.inactive_cell_counts,
            reduced_mode_counts=diag.reduced_mode_counts, cut_region_count=diag.cut_region_count,
            fit_failure_count=diag.fit_failure_count,
            moment_fit_residual_max=diag.moment_fit_residual_max,
            cut_fallback_count=diag.cut_fallback_count,
            cut_fallback_points=diag.cut_fallback_points,
            symmetry_residual=diag.symmetry_residual, condition_estimate=diag.condition_estimate,
            solver=diag.solver, residual_norm=solution.diagnostics.residual_norm, l2_error=error,)
end
