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
  - `cut_region_count::Int`, `fit_failure_count::Int`,
    `moment_fit_residual_max::Float64` — FCM moment-fit statistics from
    the integration plan.
  - `facet_region_count::Int` — total number of [`FacetRegion`](@ref)s
    cached on the model, summed across every cached selector
    (Dirichlet conditions and any `block`/`loadform` carrying
    `on::BoundarySelector`). Zero for problems with no physical-
    boundary integration.
  - `surface_region_count::Int` — total number of
    [`SurfaceRegion`](@ref)s cached on the model, summed across every
    cached [`BoundaryMesh`](@ref). Zero for problems with no
    immersed-boundary integration.
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
    cut_region_count::Int
    fit_failure_count::Int
    moment_fit_residual_max::Float64
    facet_region_count::Int
    surface_region_count::Int
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

# Copy the plan-derived region statistics into a diagnostics record in
# place. Used by `_set_plan_stats!` (every lifecycle event) and by
# `assemble!`.
function _set_integration_stats!(diag::AssemblyDiagnostics, plan::IntegrationPlan)
    diag.integration_regions = length(plan.regions)
    diag.small_overlap_count = plan.small_overlap_count
    diag.small_overlaps = _float_small_overlaps(plan.small_overlaps)
    diag.min_integration_volume = Float64(plan.min_volume)
    diag.min_relative_integration_volume = Float64(plan.min_relative_volume)
    return diag
end

function AssemblyDiagnostics(; dimension=0, active_unknowns=0, integration_regions=0,
                             small_overlap_count=0, min_integration_volume=NaN,
                             min_relative_integration_volume=NaN, symmetry_residual=NaN,
                             condition_estimate=NaN, solver=:none,
                             small_overlaps=SmallOverlap{Float64}[],
                             inactive_cell_counts=Int[], cut_region_count=0, fit_failure_count=0,
                             moment_fit_residual_max=0.0, facet_region_count=0,
                             surface_region_count=0)
    return AssemblyDiagnostics(Int(dimension), Int(active_unknowns), Int(integration_regions),
                               Int(small_overlap_count), _float_small_overlaps(small_overlaps),
                               Float64(min_integration_volume),
                               Float64(min_relative_integration_volume), Float64(symmetry_residual),
                               Float64(condition_estimate), Symbol(solver),
                               Int[inactive_cell_counts...],
                               Int(cut_region_count), Int(fit_failure_count),
                               Float64(moment_fit_residual_max), Int(facet_region_count),
                               Int(surface_region_count))
end

# Per-level count of cells deactivated by a `LevelMask`. Returns one
# entry per level (0 for unmasked levels), in level order. Used during
# diagnostics construction so the user can see how many cells each
# level dropped (mask + fictitious-fold combined).
function _inactive_cell_counts(V::Space)
    return [level.mask === nothing ? 0 : count(!, level.mask.on) for level in V.levels]
end

# Scan the integration plan for NNMF-fit statistics. `:cut_fitted` are
# successful fits; `:cut_failed` is a fit that hit the catastrophic
# residual threshold and contributes zero quadrature weight. Both count
# as "cut" regions for diagnostics; only the latter is a fit failure.
function _cut_region_stats(plan::IntegrationPlan)
    cut = 0
    failed = 0
    for region in plan.regions
        kind = region.quadrature.kind
        if kind === :cut_fitted
            cut += 1
        elseif kind === :cut_failed
            cut += 1
            failed += 1
        end
    end
    return cut, failed
end

# Convenience helper shared by `prepare`, `move!`, `_update_mask!`, and
# `assemble!`: fold every plan-level statistic into the diagnostics
# record at once. Returns the diagnostics so call sites can chain.
function _set_plan_stats!(diag::AssemblyDiagnostics, plan::IntegrationPlan)
    _set_integration_stats!(diag, plan)
    diag.cut_region_count, diag.fit_failure_count = _cut_region_stats(plan)
    diag.moment_fit_residual_max = plan.moment_fit_residual_max
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
struct AssemblyPattern
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    symmetric::Bool
    key::UInt
end

"""
    Model{D,T,P}

Prepared problem, ready to assemble and solve. Mutable so that
[`move!`](@ref), [`activate!`](@ref), [`deactivate!`](@ref), and
[`assemble!`](@ref) can update its state without forcing the caller to
re-thread a fresh value through. Fields:

  - `problem::P` — the underlying [`Problem`](@ref), possibly already
    folded against a `PhysicalDomain` (cells outside Ω deactivated via
    the strict-α path in `mesh.jl`'s `_apply_physical_fold`).
  - `version::Int` — bump counter for stale-solution detection. Every
    in-place mutation that invalidates the assembled state bumps this;
    a [`Solution`](@ref) carrying an older version raises on reuse.
  - `integration::Union{Nothing,IntegrationPlan{D,T}}` — cached
    integration plan. `nothing` only between construction and the first
    integration-plan request; populated by `prepare` and refreshed by
    every mutator.
  - `dofs::SystemLayout{D,T}` — per-field dof layout and active
    enumeration.
  - `matrix::Union{Nothing,SparseMatrixCSC{T,Int}}` — assembled global
    matrix. `nothing` until [`assemble!`](@ref) populates it; cleared
    by every mutator.
  - `rhs::Union{Nothing,Vector{T}}` — assembled right-hand side. Same
    invalidation contract as `matrix`.
  - `facet_regions::Dict{BoundarySelector,Vector{FacetRegion{D,T}}}` —
    per-selector cache of physical-boundary [`FacetRegion`](@ref)s. One
    entry per `BoundarySelector` referenced by a Dirichlet condition
    or by any `block`/`loadform` carrying `on::BoundarySelector`.
    Populated at `prepare` and refreshed by every mutator.
  - `surface_regions::IdDict{Any,Vector{SurfaceRegion{D,T}}}` — per-mesh
    cache of immersed-boundary [`SurfaceRegion`](@ref)s. One entry per
    [`BoundaryMesh`](@ref) referenced by a `block`/`loadform` carrying
    `on::BoundaryMesh`. Keyed by object identity (the `BoundaryMesh`
    value carries no canonical hash; identity match prevents two
    structurally-equal but distinct meshes from accidentally sharing
    a cache entry).
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
    version::Int
    integration::Union{Nothing,IntegrationPlan{D,T}}
    dofs::SystemLayout{D,T}
    matrix::Union{Nothing,SparseMatrixCSC{T,Int}}
    rhs::Union{Nothing,Vector{T}}
    facet_regions::Dict{BoundarySelector,Vector{FacetRegion{D,T}}}
    surface_regions::IdDict{Any,Vector{SurfaceRegion{D,T}}}
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

"""
    system_layout(problem::Problem; tolerance=GeometryTolerance(T)) -> SystemLayout

Build the per-field [`DofLayout`](@ref)s for every field of `problem`
and assemble them into a [`SystemLayout`](@ref). Each field's active
dofs are enumerated independently and then offset into a global active
block, so fields occupy disjoint ranges of the global enumeration.

Called by [`prepare`](@ref) and the in-place mutators. End users do
not usually call this directly.
"""
function system_layout(problem::Problem{D,T}; tolerance=GeometryTolerance(T)) where {D,T}
    layouts = FieldLayout{D,T}[]
    by_name = Dict{Symbol,Int}()
    offset = 0
    for field in problem.fields
        layout = dof_layout(field.space; dirichlet=_dirichlet_for_field(problem, field.name),
                            tolerance, components=component_count(field))
        push!(layouts, FieldLayout{D,T}(field.name, component_count(field), layout, offset))
        by_name[field.name] = length(layouts)
        offset += active_unknowns(layout)
    end
    return SystemLayout{D,T}(layouts, by_name, offset, tolerance)
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
    effective_problem, classify_cache = _apply_physical_fold_to_problem(problem)
    # Capture the user's integration-plan options (tolerance, criterion, …) so
    # the in-place mutators can reproduce this exact plan instead of reverting
    # to `integration_plan`'s defaults. `classify_cache` is geometry-derived,
    # not a user option, so it is rebuilt per mutation rather than stored here.
    plan_options = (; kwargs...)
    plan = integration_plan(effective_problem.space; plan_options..., classify_cache=classify_cache)
    tolerance = get(plan_options, :tolerance, GeometryTolerance(T))
    layout = system_layout(effective_problem; tolerance)
    facet_regions = _resolve_facet_regions(effective_problem, tolerance)
    surface_regions = _resolve_surface_regions(effective_problem, tolerance)
    diag = AssemblyDiagnostics(dimension=D, active_unknowns=active_unknowns(layout),
                               inactive_cell_counts=_inactive_cell_counts(effective_problem.space),
                               facet_region_count=_facet_region_count(facet_regions),
                               surface_region_count=_surface_region_count(surface_regions))
    _set_plan_stats!(diag, plan)
    return Model{D,T,typeof(effective_problem)}(effective_problem, 1, plan, layout, nothing,
                                                nothing, facet_regions, surface_regions, diag,
                                                nothing, plan_options)
end

# Build the facet-region cache for a problem from every
# `BoundarySelector` it references: Dirichlet conditions and any
# block/load carrying `on::BoundarySelector`. Each unique selector is
# resolved through `_facet_regions_for_selector` and stored in the
# returned dict. Selectors without any admissible regions (e.g. an
# overlay-only level mask emptying a face) end up with an empty list,
# not absent — callers can distinguish "selector with no regions" from
# "selector never referenced".
function _resolve_facet_regions(problem::Problem{D,T}, tolerance::GeometryTolerance{T}) where {D,T}
    regions = Dict{BoundarySelector,Vector{FacetRegion{D,T}}}()
    for selector in _referenced_facet_selectors(problem)
        haskey(regions, selector) && continue
        regions[selector] = _facet_regions_for_selector(problem.space, selector, tolerance)
    end
    return regions
end

# Yields every `BoundarySelector` referenced anywhere in `problem`:
# Dirichlet conditions and the `on` tags of blocks and loads.
function _referenced_facet_selectors(problem::Problem)
    return Iterators.flatten(((c.boundary for c in problem.dirichlet),
                              (b.on for b in problem.blocks if b.on isa BoundarySelector),
                              (l.on for l in problem.loads if l.on isa BoundarySelector)))
end

# Build the surface-region cache for a problem from every
# `BoundaryMesh` referenced by a block or load's `on` tag. Keyed by
# `IdDict` (object identity) — see the `Model` docstring for why.
function _resolve_surface_regions(problem::Problem{D,T},
                                  tolerance::GeometryTolerance{T}) where {D,T}
    regions = IdDict{Any,Vector{SurfaceRegion{D,T}}}()
    for mesh in _referenced_boundary_meshes(problem)
        haskey(regions, mesh) && continue
        regions[mesh] = _surface_regions_for_mesh(problem.space, mesh, tolerance)
    end
    return regions
end

function _referenced_boundary_meshes(problem::Problem)
    return Iterators.flatten(((b.on for b in problem.blocks if b.on isa BoundaryMesh),
                              (l.on for l in problem.loads if l.on isa BoundaryMesh)))
end

# Sum of region counts across every cached `BoundaryMesh` for the
# diagnostics `surface_region_count` field.
function _surface_region_count(regions::IdDict)
    total = 0
    for (_, list) in regions
        total += length(list)
    end
    return total
end

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

# Sum of facet-region list lengths across every cached selector. Used
# by the diagnostics' `facet_region_count` field.
function _facet_region_count(regions::Dict)
    total = 0
    for (_, list) in regions
        total += length(list)
    end
    return total
end

# Rebuild `problem` over a new (folded / moved / remasked) space, reusing
# the field channels, forms, and Dirichlet data. Shared rebuild rule
# behind `_apply_physical_fold_to_problem`, `_moved_problem`, and
# `_remasked_problem`.
function _problem_with_space(problem::Problem, new_space::Space)
    new_fields = map(f -> field(f.name, new_space; components=component_count(f)), problem.fields)
    return Problem(new_fields; blocks=problem.blocks, loads=problem.loads,
                   dirichlet=problem.dirichlet, symmetric=problem.symmetric)
end

# Apply the PhysicalDomain fold (cells outside Ω → inactive in the level
# mask) to `problem`'s space and return `(new_problem, classify_cache)`.
# The cache carries the per-box cell classifications computed by the
# fold; it is threaded into `integration_plan` so the region-level
# dispatcher reuses them instead of re-classifying the same boxes. A
# `nothing` physical domain or an empty fictitious set is a no-op fast
# path (returns the original problem and an empty cache).
function _apply_physical_fold_to_problem(problem::Problem{D,T}) where {D,T}
    cache = _ClassifyCache{D,T}()
    new_space = _apply_physical_fold(problem.space, cache)
    new_space === problem.space && return problem, cache
    return _problem_with_space(problem, new_space), cache
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

Cached integration plan held on `model`. Built by [`prepare`](@ref) and
refreshed by [`move!`](@ref) / [`activate!`](@ref) / [`deactivate!`](@ref),
so once a model is prepared every assembly call uses the same plan that
the dof layout was built against. Falls back to a fresh
`integration_plan(model.problem.space)` only if `model.integration` has
never been populated (which the lifecycle path keeps from happening in
normal use).
"""
function integration_plan(model::Model)
    model.integration === nothing ? integration_plan(model.problem.space) : model.integration
end

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

# Rebuild the model's problem with overlay `level` moved to box `to`,
# reusing the existing forms and boundary data on the moved space.
function _moved_problem(model::Model{D,T}, level::Integer, to::AxisBox{D,T}, tolerance) where {D,T}
    return _problem_with_space(model.problem,
                               moved_space(model.problem.space; level, to, tolerance))
end

# Shared invalidation tail for `move!` / `_update_mask!`. Clears the
# cached matrix / rhs, rebuilds the facet-region cache against the
# (already-updated) space, builds a fresh diagnostics record, and folds
# the current integration plan's stats in. Assumes `model.problem`,
# `model.dofs`, and `model.integration` already reflect the new state.
function _invalidate_assembly!(model::Model{D,T},
                               tolerance::GeometryTolerance{T}=GeometryTolerance(T)) where {D,T}
    model.matrix = nothing
    model.rhs = nothing
    model.pattern = nothing
    model.facet_regions = _resolve_facet_regions(model.problem, tolerance)
    model.surface_regions = _resolve_surface_regions(model.problem, tolerance)
    diag = AssemblyDiagnostics(dimension=D, active_unknowns=active_unknowns(model.dofs),
                               inactive_cell_counts=_inactive_cell_counts(model.problem.space),
                               facet_region_count=_facet_region_count(model.facet_regions),
                               surface_region_count=_surface_region_count(model.surface_regions))
    _set_plan_stats!(diag, model.integration)
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
model and then [`transfer!`](@ref).

Invalidation contract: bumps `model.version`, clears `model.matrix`
and `model.rhs`, rebuilds the integration plan and dof layout,
refreshes diagnostics. An outstanding [`Solution`](@ref) raises on
reuse.
"""
function move!(model::Model{D,T}; level::Integer, to::AxisBox{D,T}) where {D,T}
    opts = model.plan_options
    tolerance = get(opts, :tolerance, GeometryTolerance(T))
    moved_p = _moved_problem(model, level, to, tolerance)
    model.problem, classify_cache = _apply_physical_fold_to_problem(moved_p)
    model.version += 1
    model.integration = integration_plan(model.problem.space; opts...,
                                         classify_cache=classify_cache)
    model.dofs = system_layout(model.problem; tolerance)
    return _invalidate_assembly!(model, tolerance)
end

"""
    moved(model; level, to) -> Model

Return a new prepared [`Model`](@ref) with overlay `level` moved to box
`to`, reusing the source problem's forms and boundary data. Unlike
[`move!`](@ref), the source model is left intact, so it can serve as
the source of a [`transfer!`](@ref):

    target = moved(model; level=2, to=box((0.1,), (0.6,)))
    target_solution = transfer!(solution, model, target)
"""
function moved(model::Model{D,T}; level::Integer, to::AxisBox{D,T}) where {D,T}
    opts = model.plan_options
    tolerance = get(opts, :tolerance, GeometryTolerance(T))
    return prepare(_moved_problem(model, level, to, tolerance); opts...)
end

# Rebuild the model's problem with one level's mask replaced. Symmetric
# to `_moved_problem` but swaps the mask instead of the mesh.
function _remasked_problem(model::Model, level_index::Integer, mask)
    return _problem_with_space(model.problem,
                               _remasked_space(model.problem.space, level_index, mask))
end

# Activate / deactivate `cells` on `level_index` and rebuild the
# model's reusable state. Same invalidation contract as `move!`:
# bumps `model.version`, rebuilds the integration plan, dof layout,
# and diagnostics, and clears any assembled matrix / rhs. Any
# outstanding `Solution` becomes stale.
function _update_mask!(model::Model{D,T}, level_index::Integer, cells, value::Bool) where {D,T}
    1 <= level_index <= length(model.problem.space.levels) ||
        throw(ArgumentError("level index $level_index out of bounds"))
    opts = model.plan_options
    tolerance = get(opts, :tolerance, GeometryTolerance(T))
    old_level = model.problem.space.levels[level_index]
    new_mask = _apply_mask_update(old_level.mask, old_level.mesh, cells, value)
    model.problem = _remasked_problem(model, level_index, new_mask)
    model.version += 1
    model.integration = integration_plan(model.problem.space; opts...)
    model.dofs = system_layout(model.problem; tolerance)
    return _invalidate_assembly!(model, tolerance)
end

"""
    activate!(model; level, cells) -> Model

Mark `cells` on `level` as active in place. `cells` accepts the same
shapes as the `active=` kwarg on [`overlay`](@ref): an iterable of
`CartesianIndex{D}`, a predicate `(cell_box, cell_index) -> Bool`, or
an `AbstractArray{Bool,D}` matching the level's cell grid. Currently-
active cells in the selection are unchanged.

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
NNMF moment-fit cache is skipped — only the boundary-trace mass
matrix on the constrained-dof subspace is reassembled and resolved.

# Semantics

  - The constrained-dof set is *not* rediscovered. It depends only on
    the boundary selector + field topology, both of which the
    structural check pins.
  - `layout.constrained_values` is refilled by reassembling the
    boundary mass matrix `M` and solving `M c = b` for the new
    `b_i = ∫ g(x) φ_i dS`. Existing basis-trace evaluation is reused
    indirectly through the boundary region cache.
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
    # constraint masks.
    for field_layout in model.dofs.fields
        field_dirichlet = _dirichlet_for_field(model.problem, field_layout.name)
        _project_dirichlet_values!(field_layout.dofs, model.problem.space, field_dirichlet)
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
    return (; dimension=diag.dimension, levels=map(_level_report, model.problem.space.levels),
            active_unknowns=diag.active_unknowns, raw_dofs=raw_dof_count(model.dofs),
            integration_regions=diag.integration_regions,
            facet_region_count=diag.facet_region_count,
            surface_region_count=diag.surface_region_count,
            small_overlap_count=diag.small_overlap_count, small_overlaps=diag.small_overlaps,
            min_integration_volume=diag.min_integration_volume,
            min_relative_integration_volume=diag.min_relative_integration_volume,
            inactive_cell_counts=diag.inactive_cell_counts, cut_region_count=diag.cut_region_count,
            fit_failure_count=diag.fit_failure_count,
            moment_fit_residual_max=diag.moment_fit_residual_max,
            symmetry_residual=diag.symmetry_residual, condition_estimate=diag.condition_estimate,
            solver=diag.solver, residual_norm=solution.diagnostics.residual_norm, l2_error=error,)
end
