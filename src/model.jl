# Prepared-problem lifecycle: the `Model` state container, its `AssemblyDiagnostics` record, the `prepare` constructor
# that turns a `Problem` into a model (folding the optional `PhysicalDomain`, building the integration plan, the dof
# layout, and the diagnostics), and the in-place mutators (`move!`, `activate!`, `deactivate!`) that bump the model's
# version counter and refresh the cached state. Also here: `moved` (the non-destructive `move!`), the `active_cells`
# query, the `diagnostics` reports, and `update_dirichlet!` — which mutates a prepared model but is *not* one of the
# "mutators" the rest of this file means by that word, because it rewrites constrained values without changing the
# structure: the version, the integration plan, the dof layout, and the sparsity pattern all survive it.
#
# No assembly happens here — the global matrix and rhs are populated by `assemble!` in `assembly.jl`. Loaded between
# `problems.jl` (whose `Problem` this file owns the prepared form of) and `assembly.jl` (whose `assemble!` mutates
# `model.matrix` / `model.rhs`).
#
# Stale-solution detection contract: `model.version` is a pin on the active dof numbering, seeded by `prepare` from a
# structural digest of the discretisation (`_discretisation_pin`) and moved on by each of `move!`, `activate!` and
# `deactivate!`. A [`Solution`](@ref) carries the pin it was computed against; reusing it against a mutated model, or
# against a model of a different discretisation, is caught by `_checked_coefficients` in `solvers.jl`.

# ── Assembly diagnostics ──────────────────────────────────────────────────────

"""
    AssemblyDiagnostics

Mutable diagnostics record carried on every [`Model`](@ref) and exposed through [`diagnostics`](@ref).
[`assemble!`](@ref) and [`solve!`](@ref) update the model's record in place; `prepare` and the version-bumping mutators
(`move!`, `activate!` / `deactivate!`) install a *fresh* record instead, so a reference held across one of those keeps
reporting the old state — re-read it through [`diagnostics`](@ref). Every field is `Float64`- or `Int`-typed for
stability across scalar types.

Fields, in declaration order, which is also the order [`diagnostics`](@ref)`(model, solution)` reports them in:

  - `dimension::Int` — spatial dimension `D`.
  - `active_unknowns::Int` — number of active (post-constraint) dofs.
  - `integration_regions::Int` — number of admissible integration regions in the model's integration plans, summed over
    the subdomains of a coupled model.
  - `facet_region_count::Int` — total number of [`FacetRegion`](@ref)s cached on the model, summed across every cached
    selector (Dirichlet conditions and any `block`/`loadform` carrying `on::BoundarySelector`). Zero for problems with
    no physical- boundary integration. Because the sum runs over selector × face pairs and not over distinct regions,
    two selectors that name the same face each count it, so this number **grows** when the same boundary is spelled with
    fewer, broader selectors even though the resolved regions and the integration work are unchanged — the regions
    themselves are shared through the model's [`FacetResolver`](@ref) either way. The space-time cavity example is the
    instance: at `SC_CELLS = 3` it constrains five distinct faces carrying nine regions each, and spelling them one face
    at a time made `θ`'s and `u`'s conditions on a shared plate edge *the same selector value*, which the cache keys
    together — five entries, 5 × 9 = 45. One `boundary(:all; except=…)` per field gives each field its own selector, so
    the four faces both name are counted twice: nine selector × face pairs, 9 × 9 = 81. The faces, the regions and the
    quadrature are identical in both spellings, so read a rise here as a change of spelling until the region geometry
    itself has been checked.
  - `cut_facet_region_count::Int` — how many of those regions have part of their **own face** outside `Ω`, counted the
    same way (so it is directly comparable with `facet_region_count`). A facet region is the whole grid-aligned face of
    its parent cells, and its rule covers the part of that face inside `Ω`: this count is how much of the boundary
    integration — the L² Dirichlet mass and right-hand side, a Neumann / Robin / Nitsche term placed with
    `on = boundary(…)`, [`boundary_integral`](@ref) — rests on a trimmed rule rather than on an exact tensor product. On
    a face `∂Ω` crosses transversally that rule is accurate, not exact (see `facet_moment_fit_residual_max`), so a
    nonzero count is where to look when a boundary quantity is less accurate than the cells behind it.

    The verdict is on the face, not on the parent cells: a cut *cell* whose face lies wholly inside `Ω` needs no
    trimming and is not counted. A region whose face lies wholly *outside* `Ω` is counted — that is the extreme case
    rather than an exception, and it is what a symmetry plane through a hole produces; such a region keeps its place in
    the list, with an empty rule, because its parents still carry constrained trace dofs. It is the `kind` field of each
    [`FacetRegion`](@ref), built from [`classify_cell`](@ref) applied to the region's own face box against the level set
    restricted to the facet's affine slice, and the count is of the regions whose kind is not `:full`. Being a
    classification rather than a sampling of the rule, it carries the classifier's own resolution rather than the rule's
    point spacing; the `kind` field of [`FacetRegion`](@ref) states what that resolution does and does not see. Zero
    when the space carries no [`PhysicalDomain`](@ref), and zero on an immersed space all of whose integrated faces lie
    inside `Ω`.
  - `facet_fit_failure_count::Int`, `facet_cut_fallback_count::Int`, `facet_moment_fit_residual_max::Float64` — the
    facet analogues of `fit_failure_count` / `cut_fallback_count` / `moment_fit_residual_max`, over the cut *faces*
    rather than the cut cells, and counted the same way `cut_facet_region_count` is. A facet cut region's rule is the
    moment fit on the facet's affine slice of `Ω`, so it has the same three outcomes a cell's does: a fit
    (`:cut_fitted`), the raw Saye rule on the slice when the residual misses the failure threshold (`:cut_fallback`),
    and no rule at all where `Ω ∩ face` carries none (`:cut_failed` / `:cut_alpha_failed`). The residual bounds the
    fit's own compression error — how well the returned rule reproduces the moments the quadrature kernel handed it —
    and is zero on a model with no cut face. It does **not** bound the kernel's error on those moments, a separate and
    silent ceiling: the moments are exact where every leaf of the level set restricted to the face is affine (planar),
    corners where several meet strictly inside the face box included, and on one-dimensional facets, and accurate
    rather than exact on a curved cut, as [`boundary`](@ref)'s docstring details.

    Read a nonzero `facet_fit_failure_count` with `facet_moment_fit_residual_max` beside it, because the two name
    different situations and only one is a warning. A *tangency* — `∂Ω` touching a face without crossing it — classifies
    `:cut` while `Ω ∩ face` has measure zero, so the fit returns nothing, the region's rule is correctly empty, and the
    residual stays at `0.0`. That is `kirsch_plate_2d`: both of its counted regions are tangencies,
    `facet_fit_failure_count` reads 2, and each symmetry face still integrates to its exact 3.0 from the `:full` regions
    alone. A fit that genuinely struggled shows up as a nonzero residual or as `facet_cut_fallback_count`.
  - `min_relative_facet_measure::Float64` — the smallest ratio of a facet region's integrated measure (its weight sum)
    to the full geometric measure of its own face, over every resolved facet region. `1.0` when nothing is trimmed,
    which includes a model with no facet regions at all. It is a fraction of **one face**, so it is not comparable with
    `min_relative_integration_volume`, which is one region's volume as a fraction of the whole domain and therefore
    falls with the cell count rather than with a cut; the two answer different questions and have no common scale.

    `0.0` wants reading beside `facet_fit_failure_count`, for the same reason that field carries a tangency caveat. Two
    situations report it: a face lying wholly outside `Ω` under strict `α = 0`, which is the one worth looking at, and a
    zero-measure `:cut_failed` region, where `∂Ω` is merely tangent to the face and the `:full` regions beside it
    already integrate it exactly. `kirsch_plate_2d` is the in-tree instance of the second — its two symmetry faces
    report `0.0` here and `2` in `facet_fit_failure_count` while each still integrates to its exact `3.0`.

    Either way it is the conditioning warning to read before a trimmed boundary condition: trimming is a small-cut
    generator for the Dirichlet trace mass in exactly the way a thin cut cell is one for the stiffness matrix — measured
    on a single-plane cut through one face, the trace mass goes from condition 40 at an untrimmed face to 2.8e5 at a
    ratio of 5e-2 and 9.6e13 at 1e-3. The package reports the ratio and leaves the stabilisation to the caller, `α` on
    the domain being the knob — `CONTRIBUTING.md` asks for conditioning behaviour to be visible before stabilisation is
    added, not for a default stabiliser.
  - `dirichlet_trace_factors::Vector{Symbol}` — which branch the L² Dirichlet trace solve `M c = b` took, one entry per
    component of each field, concatenated in field-declaration order. Every field contributes exactly its own
    `components` entries, so a position names a `(field, component)` pair without a lookup. `:cholesky` where that
    component's boundary mass `∫ φᵢ φⱼ ds` was positive definite, `:pseudoinverse` where `cholesky` reported it
    indefinite, and `:none` where the component had no constrained dof to solve for — which is also what a model
    prepared with homogeneous data reads, constant-zero data (a zero number, or a tuple or array whose every entry is
    zero) skipping the projection at [`prepare`](@ref) altogether.

    `:pseudoinverse` is the entry worth reading: it is the only externally visible trace of a boundary trace space that
    has gone singular, and on an immersed space its commonest cause is the next two fields. It is a definiteness verdict
    and **not** a conditioning one, so it says nothing about a mass that is merely near-singular —
    `min_relative_facet_measure` is the field for that, and a `:cholesky` beside a tiny ratio there is the combination
    to distrust.
  - `unsupported_dirichlet_dof_count::Int`, `unsupported_dirichlet_dofs::Vector{UnsupportedDirichletDof}` — the dofs a
    Dirichlet condition constrains with no measure anywhere on their facet support, and their count (`length` of the
    list, carried as a scalar so a report prints it without walking the list, exactly as `small_overlap_count` is).
    Trimming is what makes the configuration reachable: which dofs a condition constrains is decided by the grid-aligned
    face test alone, so a dof whose every facet region is trimmed away is still constrained, while the projection has
    nothing left to fit it on and the pseudoinverse pins it to zero. Zero on a non-immersed space and on every immersed
    space whose integrated faces stay inside `Ω`. Zero, too, on any model prepared with homogeneous data, whatever its
    geometry: that model has run no trace solve, so the list is empty, as `dirichlet_trace_factors` reads `:none`, until
    the first [`update_dirichlet!`](@ref) runs one and establishes both.

    What such a condition *ought* to mean is an open semantic question and the package does not answer it here; the
    record exists so the answer it does give is not silent. [`UnsupportedDirichletDof`](@ref) has the measured instance
    and the alternative.
  - `surface_region_count::Int` — total number of immersed-surface [`SimplexRegion`](@ref)s cached on the model, summed
    across every cached [`BoundaryMesh`](@ref). Zero for problems with no immersed-boundary integration.
  - `interface_region_count::Int` — total number of interface [`SimplexRegion`](@ref)s cached on the model, summed
    across every cached coupling [`Interface`](@ref). Zero for problems with no multi-domain interface coupling.
  - `small_overlap_count::Int` and `small_overlaps::Vector` — number and per-region records of integration regions below
    `tolerance.small_volume`.
  - `min_integration_volume::Float64` — smallest region volume.
  - `min_relative_integration_volume::Float64` — same, divided by `volume(V.domain)`.
  - `inactive_cell_counts::Vector{Int}` — count of cells deactivated by `LevelMask`, one entry per level of each
    distinct participating space, in `problem_spaces` order. Includes both user-provided masks and the fictitious fold
    `mesh.jl`'s `_apply_physical_fold` applies whenever the domain's `keep_fictitious` is `false` (the default) — that
    fold runs at any `alpha`, not only in the strict `alpha = 0` case.
  - `reduced_mode_counts::Vector{Int}` — count of high-order and dedup modes eliminated by covered-mode pruning in
    covered regions (leaf semantics; see [`prepare`](@ref)'s `prune`), one entry per level of each *field*, concatenated
    in field-declaration order. Note the different granularity from `inactive_cell_counts`, which is per *space*: the
    two vectors line up entry-for-entry only when every space carries exactly one field (the common case). Two fields
    over one space give this vector twice the length of that one.
  - `cut_region_count::Int`, `fit_failure_count::Int`, `moment_fit_residual_max::Float64` — FCM cut-cell statistics
    from the integration plan. `fit_failure_count` counts every cut region whose compressed rule missed the failure
    threshold `_FIT_FAILURE_RESIDUAL · ‖m‖` (relative to the norm of the cell's moment vector, so the verdict does not
    depend on the cell's size) or held no points, whatever rule the region ended up carrying.
  - `cut_fallback_count::Int`, `cut_fallback_points::Int` — the subset of those failures that fell back to the
    uncompressed cut rule (`:cut_fallback`), and the total number of quadrature points those regions carry. A compressed
    rule carries at most as many points as its moment basis has functions, so the second number is the price of the
    safety net; a nonzero count means the mesh is too coarse for the geometry in those cells and should be refined. See
    [`cut_cell_rule`](@ref).
  - `symmetry_residual::Float64` — the reported value of `‖A − Aᵀ‖_F` for the assembled matrix. It is asserted, never
    measured: `assemble!` writes exactly `0.0` when the problem declares itself symmetric — such a form is scattered
    into the lower triangle only and mirrored entry-for-entry by `_mirror_lower`, so `A = Aᵀ` holds to the bit by
    construction — and `NaN` otherwise. `NaN` therefore means "no symmetry claim", both before `assemble!` has run and
    for a form the problem did not declare symmetric; it is never a measurement of asymmetry.
  - `condition_estimate::Float64` — `cond(Matrix(A))` for systems with `1 ≤ n ≤ 256`; `NaN` otherwise (computing the
    condition number of a large matrix is too expensive for a default diagnostic, and an empty system has none).
  - `scaled_condition_estimate::Float64` — the same quantity for the symmetrically diagonal-scaled operator `D⁻¹ A D⁻¹`,
    `D = diag(√|A_ii|)`, which is what a Jacobi-preconditioned solver actually sees. Same size cap, and `NaN` where any
    diagonal entry is zero.

    Reporting both is what makes an **immersed** system readable, because on a cut cell the two answer different
    questions. An unscaled `cond` there is dominated by the spread of the diagonal — a basis function whose support is
    almost entirely fictitious carries an almost-zero diagonal entry — and that spread is exactly what a diagonal
    preconditioner removes. Whether anything is left after it is removed is a property of the *basis*, and the two
    shipped families differ on it sharply. Measured on a disc of radius `r` removed from the unit square, `−Δu = 1`,
    degree 2 on an 8² grid — a size at which both families stay under the cap above, so both numbers are the ones this
    diagnostic prints:

    ```text
    r        family       unknowns    cond       scaled cond
    0.25     Legendre      216        1.36e6     4.22e4
    0.25     B-spline       64        5.99e2     6.04
    0.30     Legendre      192        1.06e5     5.58e3
    0.30     B-spline       64        1.26e5     7.17
    ```

    The gap widens with the degree, and past degree 2 only the B-spline half is reportable here: at degree 3 on the same
    grid the spline system has 81 unknowns and scales to 39.9 (`r = 0.25`) and 45.1 (`r = 0.30`) from 1.48e5 and 3.63e8,
    while the integrated-Legendre system has 504 and is `NaN` by the cap. Computed directly, past the cap, a degree-3
    16² pair gives 2.2e10 for integrated Legendre against 4.4e1 for the splines.

    The B-spline rows are the maximal-continuity family: on a cut cell holding a background vertex only one spline is
    supported there, so near-dependence survives only on vertex-free slivers, while a `C⁰` basis of degree ≥ 2 produces
    it on every small cut cell. That difference is the reason an immersed spline system is solvable with a diagonal
    preconditioner and an immersed high-order `C⁰` system is not, and it is invisible in `condition_estimate` alone.
  - `solver::Symbol` — solver tag recorded by [`solve!`](@ref).
"""
Base.@kwdef mutable struct AssemblyDiagnostics
    dimension::Int = 0
    active_unknowns::Int = 0
    integration_regions::Int = 0
    facet_region_count::Int = 0
    cut_facet_region_count::Int = 0
    facet_fit_failure_count::Int = 0
    facet_cut_fallback_count::Int = 0
    facet_moment_fit_residual_max::Float64 = 0.0
    min_relative_facet_measure::Float64 = 1.0
    dirichlet_trace_factors::Vector{Symbol} = Symbol[]
    # The count is `length(unsupported_dirichlet_dofs)`, kept as a field for the report: `prepare` and
    # `update_dirichlet!`, the list's only writers, set the two together.
    unsupported_dirichlet_dof_count::Int = 0
    unsupported_dirichlet_dofs::Vector{UnsupportedDirichletDof} = UnsupportedDirichletDof[]
    surface_region_count::Int = 0
    interface_region_count::Int = 0
    small_overlap_count::Int = 0
    small_overlaps::Vector{SmallOverlap{Float64}} = SmallOverlap{Float64}[]
    min_integration_volume::Float64 = NaN
    min_relative_integration_volume::Float64 = NaN
    inactive_cell_counts::Vector{Int} = Int[]
    reduced_mode_counts::Vector{Int} = Int[]
    cut_region_count::Int = 0
    fit_failure_count::Int = 0
    moment_fit_residual_max::Float64 = 0.0
    cut_fallback_count::Int = 0
    cut_fallback_points::Int = 0
    symmetry_residual::Float64 = NaN
    condition_estimate::Float64 = NaN
    scaled_condition_estimate::Float64 = NaN
    solver::Symbol = :none
end

function Base.show(io::IO, diagnostics::AssemblyDiagnostics)
    print(io, "AssemblyDiagnostics(D=", diagnostics.dimension, ", active=", diagnostics.active_unknowns, ", regions=",
          diagnostics.integration_regions, ", small=", diagnostics.small_overlap_count, ", solver=:",
          diagnostics.solver, ")")
end

# ── Region-cache keys ─────────────────────────────────────────────────────────
#
# A single-sided region list (`FacetRegion`s for a `BoundarySelector`, `SimplexRegion`s for a `BoundaryMesh`) is built
# against exactly *one* subdomain space: its parents live on that space's level-id block, so only that space's fields
# ever evaluate on it (`_parent_range`). On a coupled problem the same `on=` target may legitimately be named by fields
# on different subdomains — `boundary(axis=1, side=:upper)` is a face of every subdomain that has one — and those are
# different sets of facets. The cache key is therefore the *pair* `(on, space)`, not the target alone. A two-sided
# interface list carries the parents of both its subdomains, so its key is `(iface, nothing)`, and the two kinds share
# one cache (`Model.regions`) without ever sharing a key.
#
# A plain `Tuple` already gives each half the equality it deserves, because `Tuple` composes its elements' `isequal` /
# `hash` elementwise:
#
#   * a `BoundarySelector` is a *description*, not an object with an identity, and carries value `==` / `hash` (see
#     `dirichlet.jl`), so two value-equal selectors are one key;
#   * a `Space` — like a `BoundaryMesh` or an `Interface` — defines neither, so Base's fallbacks (`===`, `objectid`)
#     apply and each object is its own key.
#
# Identity is the right notion for the space half: `problem_spaces` already deduplicates spaces by `===`, so "a distinct
# discretisation" is package-wide synonymous with "a distinct `Space` object".
const RegionKey = Tuple{Any,Any}

# ── Assembly cache ────────────────────────────────────────────────────────────
#
# Everything assembly derives from a model's structure and reuses across calls lives in one object, `model.assembly`.
# The types are defined here rather than in `assembly.jl` because the `Model` holds them and this file is included
# first; the code that fills and reads them lives in `assembly.jl`.
#
# A `RegionKey` names every integration pass an assembly call can run:
#
#     volume plan of `space`         (nothing, space)
#     facet list of a selector       (selector, space)
#     surface list of a mesh         (mesh, space)
#     two-sided interface            (iface, nothing)
#
# Selectors compare by value, spaces, meshes and interfaces by identity (see above), and every key holds a reference to
# its objects, so no two distinct lists can ever share a key. That rules out by construction the failure of a cache
# keyed by `objectid`: once a transient region list is collected, a later list can be allocated at the same address and
# inherit an entry that describes a different list.

# The sorted active global dof ids of every region of one list: region `r` owns `val[ptr[r]:ptr[r+1]-1]`, ascending.
# Read as a matrix it is the dofs × regions CSC incidence the sparsity pattern is built from, and it fixes each region's
# slice of the threaded arena, so it is derived once per list, through the same `_frame!` and `_slots!` the numeric pass
# runs.
struct RegionDofs
    ptr::Vector{Int}
    val::Vector{Int}
end

# One integration region list as assembly sees it: the regions, the `q.point` offset of each (region `r`'s points are
# `offsets[r] .+ (1:npoints)`), and the regions' active dofs. `offsets[1]` is the model's point base
# (`AssemblyCache.point_base`): 0 on every model a caller holds, so the points number from 1 and `offsets[end]` is the
# count, but not on `estimate`'s enriched model, so the count is read as `_point_count`. All three are derived once,
# under the cache lock, when the list is first resolved, and never change afterwards.
struct RegionList{R}
    regions::Vector{R}
    offsets::Vector{Int}
    dofs::RegionDofs
end

# CSC sparsity pattern of one operator over the active dofs: the union over every region of every matrix pass of the
# dense block on the region's active dofs. A symmetric operator stores the lower triangle (global row ≥ col) only and is
# mirrored when the matrix is built. `colptr` and `rowval` do not depend on the scalar type or on any value, so one
# pattern serves every operator over the same passes: mass, stiffness, a Newton tangent at any iterate.
struct AssemblyPattern
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    symmetric::Bool
end

# The model's assembly cache. Every field is guarded by `lock`, which an assembly call takes a fixed handful of times
# (list and pattern lookup, scratch checkout and return) and never inside the quadrature loop or the scatter.
#
#   - `lists` — the region lists of every volume plan and of every `on=` target `prepare` resolved, as `key => list`
#     pairs searched linearly. Bounded by those, so never evicted.
#   - `oneshot` — the lists of `on=` targets `prepare` did not see (a block built after the fact, `boundary_integral` on
#     a new selector), most recently used first, at most 8. A hot one-shot target is resolved once.
#   - `patterns` — sparsity patterns keyed by the matrix passes' region lists and the symmetry flag, most recently used
#     first, at most 4, so operators that alternate (mass and stiffness of a time stepper, the problem's own operator
#     and a transfer's mass) keep their patterns.
#   - `arena` — the threaded scratch of the last threaded call, handed to one call at a time; a concurrent call that
#     finds it taken allocates its own, and the larger one is kept when it comes back.
#   - `workspaces` — idle `AssemblyWorkspace`s, at most `Threads.nthreads()`. Typed `Any` because the workspace type is
#     defined in `assembly.jl`; the drivers recover it behind a function barrier.
#   - `point_base` — the number every region list's `q.point`s count up from, fixed when the model is prepared. It is 0
#     on every model but one, so the points of a list number `1:n`. The exception is the throw-away enriched model
#     `estimate` prepares, which re-evaluates the caller's own forms on a cloud they were not written for: it numbers
#     from `_FOREIGN_POINTS` (`data.jl`), far below 1, so a form reading per-point state by `q.point` there fails at the
#     read instead of landing on another model's point. Carrying the base on the model rather than in a process-wide
#     flag is what keeps every other model, task and thread out of it.
#
# Nothing here depends on constrained values, so `update_dirichlet!` keeps the cache, and `_remodel!` replaces it whole
# when the structure changes.
mutable struct AssemblyCache{T}
    const lock::ReentrantLock
    const lists::Vector{Pair{RegionKey,RegionList}}
    const oneshot::Vector{Pair{RegionKey,RegionList}}
    const patterns::Vector{Pair{Tuple{Vector{RegionList},Bool},AssemblyPattern}}
    arena::Vector{T}
    const workspaces::Vector{Any}
    const point_base::Int
end
function AssemblyCache{T}(point_base::Int=0) where {T}
    return AssemblyCache{T}(ReentrantLock(), Pair{RegionKey,RegionList}[], Pair{RegionKey,RegionList}[],
                            Pair{Tuple{Vector{RegionList},Bool},AssemblyPattern}[], T[], Any[], point_base)
end

# ── Model and `prepare` ───────────────────────────────────────────────────────

# One prepared subdomain of a model: its space, its integration plan and the moment-fit cache the plan was built on
# (see `Model.subdomains`). The three are made together by `prepare`, one per distinct space, and every per-subdomain
# reader wants them together, so they are one record rather than parallel vectors kept aligned by position.
struct _Subdomain{D,T}
    space::Space{D,T}
    plan::IntegrationPlan{D,T}
    fits::_MomentFitCache{D,T}
end

"""
    Model{D,T,P}

Prepared problem, ready to assemble and solve. Mutable so that [`move!`](@ref), [`activate!`](@ref),
[`deactivate!`](@ref), and [`assemble!`](@ref) can update its state without forcing the caller to re-thread a fresh
value through. Fields:

  - `problem::P` — the *effective* [`Problem`](@ref): the one every consumer reads, already folded against a
    `PhysicalDomain` (cells classified fully outside Ω deactivated by `mesh.jl`'s `_apply_physical_fold`, which runs at
    any `alpha` unless the domain sets `keep_fictitious`) and reindexed into disjoint level-id blocks.
  - `prefold_space::Space{D,T}` — the *pre-fold* space: the level masks exactly as the caller described them, before
    `_apply_physical_fold` intersected them with the geometry. It is the only record of what the user asked for, because
    the fold is not invertible — once it has run, a level's `mask` no longer distinguishes a cell the caller excluded
    from a cell the classifier found outside Ω. [`move!`](@ref) and [`moved`](@ref) relocate the overlay on *this* space
    and fold the result from scratch, so a move is exactly a [`prepare`](@ref) at the new box. Only the geometry is kept
    here: the forms, field channels and Dirichlet data a rebuild needs are read back from `problem`, which carries them
    unchanged through the fold. [`activate!`](@ref) / [`deactivate!`](@ref) record their cell flips here as well as on
    `problem`, since a flip is a caller choice the next `move!` must reproduce.
  - `version::Int` — the discretisation pin: the token a [`Solution`](@ref) or [`QuadField`](@ref) carries so a consumer
    can ask whether it is still a coefficient vector of *this* model's active dof numbering. It is **not** a count of
    anything. [`prepare`](@ref) seeds it from a structural digest of the discretisation, so two models prepared from the
    same discretisation carry the same pin and a state vector moves freely between them — which is what a transient loop
    that rebuilds its problem every step needs — while two different discretisations carry different pins and a
    carry-over raises. Every in-place mutation that invalidates the assembled state then moves the pin on as well, so a
    solution taken before a [`move!`](@ref) or a mask flip raises as it always has. See `_discretisation_pin` for
    exactly what the digest reads and, just as importantly, what it must not.
  - `subdomains::Vector{_Subdomain{D,T}}` — one prepared record per *distinct* participating discretisation, in the
    first-appearance order of `problem_spaces(problem)`: one on a single-domain problem, one per subdomain space on a
    multi-domain (coupled) one. Populated by `prepare` and refreshed by every mutator. A record holds what is built
    together for one subdomain and read together:

      - `space` — the subdomain's space as prepared: folded against its own physical domain and reindexed into its own
        block of level ids. It is the very object the effective `problem`'s fields over the subdomain hold (`===`), so
        the per-subdomain consumers — the assembly passes, workspace and volume lookup, the reproducibility report —
        read it here instead of recomputing `problem_spaces`.
      - `plan` — the subdomain's cached [`IntegrationPlan`](@ref), built against that space's own physical fold. The
        field-to-plan routing is intrinsic: each field owns the regions on its space's level-id block (see
        `FieldLayout.level_ids` and `_parent_range`), so no per-plan routing table is stored.
      - `fits` — the subdomain's cut-region moment-fit rule cache, which `plan` was built on. Every mutator threads it
        back into `integration_plan`, so a cut region a `move!` or a mask flip leaves bit-identical reuses its rule
        instead of being refitted — by far the dominant cost of a 3D plan rebuild. A cache is keyed by
        `(region box, moment order)` and omits the geometry, on the reasoning that one cache sees exactly one immutable
        [`PhysicalDomain`](@ref) for its whole life; that is what the guards on [`adapted`](@ref)`(model, space)`
        enforce, and it is why there is nothing here to invalidate.

    Lineage, because a cache outlives the model that created it. A mutator (`move!`, `activate!`, `deactivate!`) keeps
    the model's own cache objects, since it overwrites the model rather than forking it. A non-mutating derivation
    (`adapted`, `elevated`, `moved`) gives the new model a *copy*, seeded warm from the source: source and target are
    then both live, and a shared cache would have each one's plan build evict the other's rules.

    Size, because the eviction is by region *box* rather than by the full `(box, moment order)` key. `integration_plan`
    drops every box the plan it has just built does not cut, so a `move!` sweep costs one plan's worth of rules and not
    the sweep's. A box that stays live keeps one entry per distinct moment order fitted there, which is what lets a base
    model and the order-elevated twin [`estimate`](@ref) builds coexist instead of evicting each other — at the price
    that where the order at a live box keeps rising, as in an hp loop, the entries at it grow with the number of cycles
    rather than holding at one plan's worth. That growth is bounded above — the moment order is
    `moment_order_factor × cell_order` and [`refine`](@ref) caps the cell order at `pmax` — and it is the smaller half
    of the trade: on a 3D sphere with 32 cut regions, four hp cycles took the cache from 0.035 MiB to 0.61 MiB and the
    next plan build from 11.9 s with no reuse to 0.011 s.
  - `dofs::SystemLayout{D,T}` — per-field dof layout and active enumeration.
  - `matrix::Union{Nothing,SparseMatrixCSC{T,Int}}` — assembled global matrix. `nothing` until [`assemble!`](@ref)
    populates it; cleared by every mutator.
  - `rhs::Union{Nothing,Vector{T}}` — assembled right-hand side. Same invalidation contract as `matrix`.
  - `facet_resolver::FacetResolver{D,T}` — the model's single route from a boundary *face* to its
    [`FacetRegion`](@ref)s, and the memo of every face already resolved (see [`FacetResolver`](@ref)). It sits one level
    below `regions`: that cache keys whole `BoundarySelector`s, which are unions of faces, while this memo keys the
    faces themselves, so a face two selectors both name is resolved once and shared. Every consumer resolves through it
    — the per-selector entries of `regions` below, the one-shot lookup for a selector no `prepare` saw, and the L²
    Dirichlet projection, which the dof layer reaches through the resolver `prepare` hands `system_layout`. Replaced
    wholesale by every mutator, which is also what retires the entries of the spaces the mutation left behind; a
    non-mutating derivation (`adapted`, `elevated`, `moved`) gets its own fresh one rather than a copy, since the keys
    are spaces and a derivation has new ones to resolve anyway.
  - `regions::Dict{RegionKey,Vector}` — the region list of every `on=` target the problem names, resolved at `prepare`
    and refreshed by every mutator. Each entry is keyed by the `RegionKey` its assembly pass carries:

      - `(selector, space)` — the physical-boundary [`FacetRegion`](@ref)s of a [`BoundarySelector`](@ref), one entry
        per selector **per subdomain space** that names it, whether through a Dirichlet condition or a
        `block`/`loadform` carrying `on::BoundarySelector`. The space is part of the key because a region list lives on
        one space's level-id block: two subdomains naming a value-equal selector own different facets and must not
        share an entry. A selector whose faces hold no admissible region (an overlay-only mask emptying a face, say)
        keeps an empty list, so "named, and empty" stays distinguishable from "never named".
      - `(mesh, space)` — the immersed-boundary [`SimplexRegion`](@ref)s of a [`BoundaryMesh`](@ref), one entry per
        mesh per subdomain space naming it through a `block`/`loadform` carrying `on::BoundaryMesh`.
      - `(iface, nothing)` — the two-sided [`SimplexRegion`](@ref)s of a coupling [`Interface`](@ref), one entry per
        interface a coupling `block` names (`couple`'s four blocks share one `Interface` object). There is no space half
        because the regions carry the parents of both subdomains.

    A selector matches by value; a mesh, an interface and a space match by object identity, so two structurally equal
    but distinct meshes never share an entry. A value is a `Vector{FacetRegion{D,T}}` under a selector and a
    `Vector{SimplexRegion{D,T}}` otherwise, and the per-kind counts of [`diagnostics`](@ref) and
    [`nquadpoints`](@ref)`(model; kind)` select their entries by the type of the key's head.
  - `dirichlet_projections::Dict{Symbol,DirichletProjection{D,T}}` — per-field cache of the datum-independent half of
    the L² Dirichlet projection (see [`DirichletProjection`](@ref)). Empty at `prepare` and filled by the first
    [`update_dirichlet!`](@ref) on each field, which is the only path that re-projects an existing dof layout; every
    mutator rebuilds the layout and empties the cache with it.
  - `diagnostics::AssemblyDiagnostics` — diagnostics record. Mutated in place by [`assemble!`](@ref) and
    [`solve!`](@ref); replaced outright by `prepare` and by every version-bumping mutator (see
    [`AssemblyDiagnostics`](@ref)).
  - `assembly::AssemblyCache{T}` — what assembly derives from the structure and reuses across calls: the region list of
    every pass with its `q.point` offsets and its regions' active dofs (unbounded for the volume plans and the `on=`
    targets `prepare` resolved; the 8 most recent for targets it did not), the CSC sparsity patterns of the 4 most
    recently assembled pass sets, and pooled per-call scratch (workspaces, the threaded arena). Every version-bumping
    mutator replaces it, since those change the structure it describes. [`update_dirichlet!`](@ref) is the exception: it
    clears `matrix`/`rhs` but *keeps* the cache, because rewriting constrained values leaves the constrained-dof set —
    and so every region's active dofs — untouched. Lets a Newton or load-stepping loop reuse the pattern and re-run only
    the numeric scatter. The cache is locked, so several tasks may assemble on one model at the same time, serial or
    threaded; mutating the model while an assembly runs is not supported.
  - `plan_options::NamedTuple` — the integration-plan keyword options (`tolerance`, `criterion`, …) captured at
    `prepare`. Every mutator (`move!`, `activate!`, `deactivate!`) rebuilds the plan with these, so a mutation
    reproduces the prepared plan rather than silently reverting to `integration_plan`'s defaults.
"""
mutable struct Model{D,T,P}
    problem::P
    prefold_space::Space{D,T}
    version::Int
    subdomains::Vector{_Subdomain{D,T}}
    dofs::SystemLayout{D,T}
    matrix::Union{Nothing,SparseMatrixCSC{T,Int}}
    rhs::Union{Nothing,Vector{T}}
    facet_resolver::FacetResolver{D,T}
    regions::Dict{RegionKey,Vector}
    dirichlet_projections::Dict{Symbol,DirichletProjection{D,T}}
    diagnostics::AssemblyDiagnostics
    assembly::AssemblyCache{T}
    plan_options::NamedTuple
end

"""
    prepare(problem::Problem; prune=true, tolerance=…, criterion=…) -> Model

Build a fresh [`Model`](@ref) for `problem`. Folds an optional [`PhysicalDomain`](@ref) into the per-level cell masks
(dropping cells classified fully outside Ω, unless the domain sets `keep_fictitious`), constructs the
[`IntegrationPlan`](@ref) sharing the cell-classification cache, builds the per-field [`DofLayout`](@ref)s, fills the
diagnostics record, and returns the assembled state with `matrix = rhs = nothing` (call [`assemble!`](@ref) or one of
the public assembly wrappers next).

On a multi-domain (coupled) problem every distinct participating [`Space`](@ref) is folded, reindexed into its own
level-id block, and given its own integration plan and moment-fit cache.

`prune` is leaf semantics, on by default and applied by the layout build (see [`dof_layout`](@ref)) to every field of
the problem: a cell carries basis functions only where no finer level has taken its region over (see [`space`](@ref) for
the rule).

`prune = false` builds the **unreduced twin**, and it is a diagnostic rather than a discretisation to solve with. The
twin keeps every covered mode, so on a nested stack the operator it produces is *exactly singular* — a retained coarse
mode and the cover's reproduction of it are the same function, and the difference is a null direction. Measured on the
unit square: 121 null directions on an 8-cell p = 3 base under a nested 4-cell overlay, and 794 on a depth-2 ladder with
every level live.

It exists to be measured against. What covered-mode pruning claims is that it removes redundancy and nothing else, and
comparing the reduced space's size with the *rank* of its unreduced twin — `test/test_ladder.jl` spells it out as
`size(gram(V), 1) == rank(gram(V; prune = false))` — is the only instrument that can see a wrong elimination: the
reduced operator stays full rank at a healthy condition number whether or not the elimination was correct. Off the
nested manifold the twin is merely larger — 233 unknowns against 209 on a 4-cell p = 3 base under a non-nesting 3-cell
overlay — and where nothing is covered the two layouts are byte-identical, down to the `(raw, component) -> active id`
map, so the two models share a pin and a solution crosses between them.

The flag is carried on the model, so a model prepared unpruned stays unpruned through [`move!`](@ref),
[`activate!`](@ref) and every derivation.

The remaining keyword arguments are forwarded to `integration_plan`; see its docstring for the full list.
"""
prepare(problem::Problem{D,T}; kwargs...) where {D,T} = _prepared_model(problem, (; kwargs...), nothing)

# The body of [`prepare`](@ref), with the moment-fit caches optionally supplied: `reuse` is `nothing` for fresh ones,
# which is what `prepare` starts with, or the vector a derivation of an existing model hands in through `_derived`,
# which is where the choice between sharing and copying is made and explained. `point_base` is the assembly cache's
# (see `AssemblyCache`): 0 everywhere but in `estimate`'s enriched model.
function _prepared_model(problem::Problem{D,T}, plan_options::NamedTuple,
                         reuse::Union{Nothing,Vector{_MomentFitCache{D,T}}}; point_base::Int=0) where {D,T}
    effective_problem, spaces, caches = _prepare_spaces(problem)
    # `plan_options` captures the user's integration-plan options (tolerance, criterion, …) so the in-place mutators can
    # reproduce this exact plan instead of reverting to `integration_plan`'s defaults. The classification caches are
    # geometry-derived, not user options, so they are rebuilt here.
    tolerance = get(plan_options, :tolerance, GeometryTolerance(T))
    # `prune` rides in the same bag although it is a dof-layer option and not a plan one, because the bag is what every
    # derivation replays: a model prepared unpruned must stay unpruned through `move!`, `activate!` and `adapted`, and
    # carrying it anywhere else would mean a second channel that can disagree with this one. It is split off here rather
    # than forwarded, since `integration_plan` would reject the keyword.
    prune = get(plan_options, :prune, true)
    prune_exempt = get(plan_options, :prune_exempt, ())
    integration_options = Base.structdiff(plan_options, NamedTuple{(:prune, :prune_exempt)})
    # One prepared subdomain per distinct space: the space, its integration plan sharing the space's own
    # cell-classification cache, and its moment-fit cache. The moment-fit caches, unlike the classification ones,
    # outlive this call: they are the model's, and every mutator hands them back so an unchanged cut region is not
    # refitted. A handed-in vector is aligned with `problem_spaces`, so a length mismatch is a caller bug and is raised
    # as one. Falling back to fresh caches here instead would turn it into a silent full refit — the exact cost the
    # reuse exists to avoid, and invisible from the outside.
    fits = if reuse === nothing
        _MomentFitCache{D,T}[_MomentFitCache{D,T}() for _ in eachindex(spaces)]
    else
        length(reuse) == length(spaces) ||
            throw(ArgumentError("moment-fit caches are aligned with the problem's distinct spaces; " *
                                "got $(length(reuse)) for $(length(spaces))"))
        reuse
    end
    subdomains = _Subdomain{D,T}[_Subdomain{D,T}(spaces[i],
                                                 integration_plan(spaces[i]; integration_options...,
                                                                  classify_cache=caches[i], moment_fit_cache=fits[i]),
                                                 fits[i]) for i in eachindex(spaces)]
    # The facet resolver is built before the dof layout, not after the region cache, because the layout's Dirichlet
    # projection resolves faces too: it is the first consumer, and handing it the model's resolver is what makes the
    # projection and the operator integrate the same facet by construction rather than by two call sites agreeing.
    #
    # The projections themselves are not kept (`_system_layout`): their trace-solve verdicts come back beside the layout
    # and go into the diagnostics below, and `dirichlet_projections` starts empty, for `update_dirichlet!` to fill.
    facet_resolver = FacetResolver{D,T}(tolerance)
    # Each prepared subdomain space mapped to its fold classification cache, for the layout to reuse: `spaces` and
    # `caches` are aligned (`_prepare_spaces`), and the keys are identities — each field's space is one of these.
    classify_caches = IdDict{Space,_ClassifyCache{D,T}}(zip(spaces, caches))
    layout, trace_factors, unsupported = _system_layout(effective_problem; tolerance, prune, prune_exempt,
                                                        classify_caches, facets=facet_resolver)
    # The region list of every `on=` target the problem names, each resolved once however often it is named. The facet
    # lists come after the layout, so every face a nonzero Dirichlet datum projected onto is read from the resolver's
    # memo here rather than resolved again.
    regions = Dict{RegionKey,Vector}()
    for key in _referenced_targets(effective_problem)
        haskey(regions, key) || (regions[key] = _resolve_target(key..., effective_problem, layout, facet_resolver))
    end
    facets = _kind_stats_of(_cached_lists(regions, BoundarySelector))
    diag = AssemblyDiagnostics(; dimension=D, active_unknowns=active_unknowns(layout),
                               facet_region_count=_region_count(regions, BoundarySelector),
                               cut_facet_region_count=facets.trimmed, facet_fit_failure_count=facets.failed,
                               facet_cut_fallback_count=facets.fallbacks,
                               facet_moment_fit_residual_max=facets.residual_max,
                               min_relative_facet_measure=facets.min_relative_measure,
                               dirichlet_trace_factors=trace_factors,
                               unsupported_dirichlet_dof_count=length(unsupported),
                               unsupported_dirichlet_dofs=unsupported,
                               surface_region_count=_region_count(regions, BoundaryMesh),
                               interface_region_count=_region_count(regions, Interface),
                               _plan_stats(s.plan for s in subdomains)...,
                               inactive_cell_counts=_inactive_cell_counts(spaces),
                               reduced_mode_counts=_reduced_mode_counts(layout))
    return Model{D,T,typeof(effective_problem)}(effective_problem, problem.space, _discretisation_pin(spaces, layout),
                                                subdomains, layout, nothing, nothing, facet_resolver, regions,
                                                Dict{Symbol,DirichletProjection{D,T}}(), diag,
                                                AssemblyCache{T}(point_base), plan_options)
end

# Fold each distinct space against its own `PhysicalDomain` and reindex its level ids into a disjoint global block, then
# rebuild the fields over the prepared spaces. Returns the effective problem, the prepared distinct spaces (in
# `problem_spaces` order), and the per-space classification caches (each threaded into its own space's integration
# plan). Folding and namespacing every subdomain independently is what keeps their cut-cell plans, dof blocks, and
# workspace banks from colliding: `_reindex_space_levels` gives each prepared space its own contiguous level-id block,
# and the fields are re-homed onto the prepared copies so no consumer ever sees the pre-fold spaces the caller passed
# in.
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
    return _rehome(problem, V -> remap[V]), prepared, caches
end

# `problem` with every field re-homed onto `space_of(field.space)`: the same names, components, forms, Dirichlet data
# and symmetry flag over other spaces. `_prepare_spaces` re-homes onto the prepared spaces, and `_derived` rebuilds a
# model's problem over a derived (adapted / elevated / moved / remasked / enriched) space, which `_prepare_spaces` then
# folds and reindexes like any other.
function _rehome(problem::Problem, space_of)
    fields = map(f -> field(f.name, space_of(f.space); components=component_count(f)), problem.fields)
    return Problem(fields; blocks=problem.blocks, loads=problem.loads, dirichlet=problem.dirichlet,
                   symmetric=problem.symmetric)
end

# ── Dof layouts ───────────────────────────────────────────────────────────────

"""
    dof_layout(V::Space; dirichlet=[], tolerance=GeometryTolerance(T), components=1,
               prune=true, prune_exempt=(), classify_cache=_ClassifyCache{D,T}()) -> DofLayout

Construct the basis-aware dof layout of one field over a superposition [`Space`](@ref). The construction proceeds in
four steps:

  1. **Topology.** Walk every level's cells in order and build each active cell's raw dof id vector through
     `TensorDofKey` lookup: adjacent cells share endpoint nodes, bubble (span) modes are cell-local, inactive cells get
     an empty vector, and each level's raws come out as one contiguous range (`level_raws`). Then record the artificial
     eliminations in an [`EliminationLedger`](@ref), one level at a time: the family-dispatched
     [`_overlay_constraints!`](@ref), then, under leaf semantics, [`_coverage_constraints!`](@ref) against the masks
     [`build_coverage`](@ref) computes, then the family's `_resolve_rows!` on any linear-constraint rows the two
     recorded. A raw is credited to the first source that eliminates it, so a raw the artificial boundary eliminates
     keeps `:overlay` when covered-mode pruning names it too. The ledger's sources become `elimination_source` and its
     pivot table `pivots`, once every pivot is checked to be expressed over raws that stay free. This step reads the
     space alone.
  2. **Physical-Dirichlet detection**, per component: one pass per level over the level's raws, matching the physical
     faces each raw's function lies on against every condition's facets as face bitmasks.
  3. **Numbering**: active ids component-major (component 1 first, then component 2, …), skipping every (raw, component)
     entry that is either physically Dirichlet-constrained or artificially eliminated.
  4. **Projection** of nonzero Dirichlet data onto the boundary trace space, into `constrained_values` (see
     [`DirichletProjection`](@ref)). Skipped for homogeneous data, whose constrained values are zero already.

Keyword arguments:

  - `dirichlet` — iterable of [`DirichletCondition`](@ref)s, every one applied to this layout whatever field it names. A
    constant tuple or array datum must carry `components` entries, as [`dirichlet`](@ref) documents; any other length
    raises `ArgumentError` before anything is built. A condition's `component` is not range-checked here, there being no
    field to check it against: one outside `1:components` constrains nothing.
  - `tolerance` — `GeometryTolerance` used by boundary detection, and the tolerance of the private `FacetResolver` the
    projection resolves faces through. A [`Model`](@ref) shares one resolver between its projections and its assembly
    path instead, through [`prepare`](@ref).
  - `components` — scalar channels per field (≥ 1).
  - `prune` — leaf semantics, on by default. `false` retains every covered mode and is the unreduced twin the reduction
    is measured against; see [`prepare`](@ref), which is where a caller reaches it.
  - `prune_exempt` — level ids this pass leaves unpruned while every other level is pruned normally. It exists for
    [`estimate`](@ref)'s enriched twin, where a co-located same-degree cover would deduplicate exactly the keys the
    solution is about to be injected onto; turning leaf semantics off *globally* there instead was measured to inflate
    the indicator on a covered cell of an unrelated level by 4.1×, because that level's order-elevated covered modes
    then survive and land in the complement. Empty by default, and a no-op when `prune = false`.
  - `classify_cache` — the space's cell-classification cache (shared with the `PhysicalDomain` fold). The
    fictitious-fold constraint predicate and the coverage rule both reuse it instead of re-classifying fold-boundary
    cells; defaults to a fresh empty cache for standalone calls.

The integrated Legendre family eliminates strongly and records no rows, so its layout has no pivot and assembly sees
only active and constrained dofs, and so does the B-spline family at maximal continuity. Below maximal continuity a
B-spline level records multi-raw rows encoding the C^m trace-vanishing condition on artificial boundaries; assembly
gives each pivot they resolve to a local row and column of its own and condenses them onto the pivot's expansion once
per integration region.

The projection's own verdicts — which branch each component's trace solve took, and which dofs it left undetermined —
are not kept: they reach a user through [`diagnostics`](@ref) on a prepared model.
"""
function dof_layout(V::Space{D,T}; dirichlet=[], tolerance=GeometryTolerance(T), components::Integer=1,
                    prune::Bool=true, prune_exempt=(),
                    classify_cache::_ClassifyCache{D,T}=_ClassifyCache{D,T}()) where {D,T}
    components > 0 || throw(ArgumentError("dof layout components must be positive"))
    # A plain loop, not `foreach` with a closure: over the abstractly typed condition vector a mixed list of data gives,
    # each call of the closure is dispatched at run time with the closure boxed, one 16-byte allocation per condition.
    for condition in dirichlet
        _check_dirichlet_datum(condition, components, condition.field)
    end
    t = _dof_topology(V; tolerance, prune, prune_exempt, classify_cache)
    return first(_field_dofs(t, V, dirichlet, components, FacetResolver{D,T}(tolerance)))
end

# One field over an already built topology: steps 2 to 4 of the layout (see `dofs.jl`). Detection, then numbering, then
# — for nonzero data only — the projection, which is returned beside the layout so the caller can read its trace-solve
# verdicts (`_append_trace_stats!`) and then let it go. Homogeneous data return `nothing` in its place: their
# constrained values are the zeros the numbering already wrote, so no boundary is walked and no mass factorised.
function _field_dofs(t::_DofTopology{D,T}, V::Space{D,T}, conditions, components::Integer,
                     facets::FacetResolver{D,T}) where {D,T}
    dofs = DofLayout(t, _physical_dirichlet(t, V, conditions, components))
    _needs_dirichlet_projection(conditions) || return dofs, nothing
    return dofs, _project_dirichlet_values!(dofs, V, facets, conditions)
end

"""
    system_layout(problem::Problem{D,T}; tolerance=GeometryTolerance(T), prune=true,
                  prune_exempt=(), classify_caches=IdDict{Space,_ClassifyCache{D,T}}(),
                  facets=FacetResolver{D,T}(tolerance)) -> SystemLayout

Build the per-field [`DofLayout`](@ref)s for every field of `problem` and assemble them into a [`SystemLayout`](@ref).
Each field's active dofs are enumerated independently and then offset into a global active block, so fields occupy
disjoint ranges of the global enumeration. Fields over different spaces (multi-domain coupling) each build their layout
from their own [`Space`](@ref); the disjoint offsets make the global enumeration a product space `V₁ × … × Vₙ`.

Fields over one space (the same object) share that space's dof topology: its raw dofs, cell → raw tables, level ranges,
artificial eliminations and pivots are built once, and every such field's [`DofLayout`](@ref) holds the same arrays
(`===`). Each field still has its own physical-Dirichlet flags, active numbering and constrained values, so two fields
over one space are numbered, constrained and updated independently.

`classify_caches` maps a [`Space`](@ref) to the cell-classification cache already filled by that space's fictitious
fold, so the constraint pass reuses those verdicts instead of re-classifying fold-boundary cells; the empty default just
classifies on demand.

`prune` is leaf semantics, applied to every field alike — it describes the discretisation, not one field of it.
[`prepare`](@ref) is where a caller sets it. `prune_exempt` is the per-level escape hatch; see [`dof_layout`](@ref) for
both.

`facets` is the `FacetResolver` every field's Dirichlet projection resolves its boundary faces through, so all of them
share one memo: it is keyed by `(space, face)`, so fields over different spaces cannot collide, and fields over one
space resolve each face once between them. [`prepare`](@ref) hands in the model's own resolver, which is what makes the
projection read the very faces the assembly path and [`boundary_integral`](@ref) read. The default is a fresh resolver
at `tolerance`, which is what a standalone call with no model behind it wants; a resolver handed in must carry
`tolerance` too.

The trace solves' verdicts are not part of the result — `prepare` reads them into [`diagnostics`](@ref) from the
projections as they are built, and then drops the projections.

Called by [`prepare`](@ref) and the in-place mutators. End users do not usually call this directly.
"""
system_layout(problem::Problem; kwargs...) = first(_system_layout(problem; kwargs...))

# `system_layout`, plus the Dirichlet trace-solve verdicts of the projections it built:
# `(layout, factors, unsupported)`, with `factors` per field per component and `unsupported` per field, component and
# raw, in that order — which is what `diagnostics` reports and what the diagnostics record holds. Each
# projection is read (`_append_trace_stats!`) as soon as its field is laid out and then dropped, so the layout is built
# with at most one projection alive, and the model `prepare` returns keeps none of them: `model.dirichlet_projections`
# is filled by `update_dirichlet!` alone.
#
# The topology (`_dof_topology`) is built once per distinct space — keyed by object identity, as `problem_spaces` keys
# it — and every field over that space numbers itself over the one result. A topology reads only the space and this
# call's `tolerance`, `prune` and `prune_exempt`, so building it again for a second field would repeat the whole cell
# walk, the coverage masks and every level's eliminations to produce equal arrays. The fields share them instead: their
# layouts hold the same space-block arrays (`===`), which `DofLayout` documents as never mutated, and only detection,
# numbering and projection run per field.
function _system_layout(problem::Problem{D,T}; tolerance=GeometryTolerance(T), prune::Bool=true, prune_exempt=(),
                        classify_caches=IdDict{Space,_ClassifyCache{D,T}}(),
                        facets::FacetResolver{D,T}=FacetResolver{D,T}(tolerance)) where {D,T}
    layouts = FieldLayout{D,T}[]
    by_name = Dict{Symbol,Int}()
    factors, unsupported = Symbol[], UnsupportedDirichletDof[]
    topologies = IdDict{Space,_DofTopology{D,T}}()
    offset = 0
    for field in problem.fields
        t = get!(topologies, field.space) do
            # Reuse the space's fold cache when `prepare` / `move!` provide it, so the constraint pass never
            # re-classifies fold-boundary cells.
            cache = get(() -> _ClassifyCache{D,T}(), classify_caches, field.space)
            _dof_topology(field.space; tolerance, prune, prune_exempt, classify_cache=cache)
        end
        components = component_count(field)
        dofs, projection = _field_dofs(t, field.space, _dirichlet_for_field(problem, field.name), components, facets)
        _append_trace_stats!(factors, unsupported, field.name, projection, components)
        # The field's (reindexed, contiguous) level-id block — how assembly routes each region to its owning subdomain
        # field without a `served` mask.
        level_ids = extrema(l.id for l in field.space.levels)
        push!(layouts, FieldLayout{D,T}(field.name, components, dofs, offset, level_ids[1]:level_ids[2]))
        by_name[field.name] = length(layouts)
        offset += active_unknowns(dofs)
    end
    return SystemLayout{D,T}(layouts, by_name, offset, tolerance), factors, unsupported
end

# Pick out the Dirichlet conditions that apply to the field called `name` from a problem's `dirichlet` list: every
# unscoped condition (`condition.field === nothing`) applies to the only field of a single-field problem and raises on
# multi-field problems; named conditions match by name. An `Any[]`, which the projection reads as such.
#
# A plain loop rather than a filtered comprehension: over the abstractly typed condition vector the comprehension's
# filter and generator are built and boxed at run time, two allocations more per field on every `update_dirichlet!`
# (51 against 49 on a cached load increment of a 2D overlay model).
function _dirichlet_for_field(problem::Problem, name::Symbol)
    scoped = Any[]
    for condition in problem.dirichlet
        if condition.field === nothing
            length(problem.fields) == 1 || throw(ArgumentError("multi-field Dirichlet conditions must specify a field"))
            push!(scoped, condition)
        elseif condition.field === name
            push!(scoped, condition)
        end
    end
    return scoped
end

# Append one field's L² Dirichlet trace-solve verdicts to `factors` and `unsupported`, in component order: the branch
# each component's solve took (`_trace_branch` of its factor) and an `UnsupportedDirichletDof` record for each dof that
# solve left undetermined. `projection` is the one the field's data were just projected through, or `nothing` for a
# field whose data are homogeneous — `prepare` skips the projection for those, so every component reads `:none` and
# nothing is unsupported.
#
# Read off the projection because that is where the verdicts live and nowhere else: the dof layout keeps no copy of
# them, and `prepare` drops each projection it builds as soon as this has read it, so the model it returns holds none.
# `update_dirichlet!` reads the ones it caches the same way. The unsupported dofs are self-describing records rather
# than a flattened vector: the list is empty on every untrimmed boundary, so there is no position worth counting from.
function _append_trace_stats!(factors::Vector{Symbol}, unsupported::Vector{UnsupportedDirichletDof}, name::Symbol,
                              projection, components::Integer)
    for c in 1:components
        push!(factors, projection === nothing ? :none : _trace_branch(projection.factors[c]))
        projection === nothing && continue
        append!(unsupported, (UnsupportedDirichletDof(name, c, raw) for raw in projection.unsupported[c]))
    end
    return factors, unsupported
end

# ── Region caches ─────────────────────────────────────────────────────────────

# Every `on=` target `prepare` resolves a region list for, as the `RegionKey` its assembly pass will look up: each
# Dirichlet condition's selector on the space it constrains, then each block's and load's `on` — a `BoundarySelector`
# or a `BoundaryMesh` on the space of the form's test field, an `Interface` on `nothing`, since its regions span both
# its subdomains. A target named several times is yielded as often; `prepare` resolves it once. A form with
# `on === nothing` integrates over its subdomain's plan, which `prepare` builds per subdomain rather than per form, and
# any other `on` is left to assembly, which raises on it. A condition with `field === nothing` (only legal on a
# single-field problem) lands on that problem's one space.
function _referenced_targets(problem::Problem)
    return Iterators.flatten((((c.boundary, c.field === nothing ? problem.space : _field_space(problem, c.field))
                               for c in problem.dirichlet),
                              ((f.on, f.on isa Interface ? nothing : _field_space(problem, f.test_name))
                               for f in (problem.blocks..., problem.loads...)
                               if f.on isa Union{BoundarySelector,BoundaryMesh,Interface})))
end

# The regions of one `on=` target on space `V`: how `prepare` fills `Model.regions`, and how assembly resolves a target
# `prepare` never saw (`_regions` in `assembly.jl`), so the two cannot build a target differently.
#
#   - A selector is the union of its faces' `FacetRegion`s (`_facet_regions_for_selector`), resolved through the model's
#     `FacetResolver`: a face another selector or the Dirichlet projection already resolved is shared, not rebuilt.
#   - A mesh is cut against `V`'s grid at the layout's tolerance. One mesh named by fields on two subdomains is two
#     targets, each cut against its own grid; a genuine *two-sided* coupling across a mesh is an `Interface`.
#   - An interface is cut against the merged grid lines of both its subdomains (`_interface_regions`), with the coupled
#     fields' indices from the layout, which is why it reads the problem and the layout and ignores `V`.
_resolve_target(selector::BoundarySelector, V, _, _, facets) = _facet_regions_for_selector(V, selector, facets)
_resolve_target(mesh::BoundaryMesh, V, _, layout, _) = _surface_regions_for_mesh(V, mesh, layout.tolerance)
_resolve_target(iface::Interface, _, problem, layout, _) = _interface_regions(iface, problem, layout)

# Resolve one selector into its full list of `FacetRegion`s (the union of regions across every facet the selector
# covers): the facet case of `_resolve_target`.
#
# The union is assembled face by face through `facets` (see [`FacetResolver`](@ref)), so a face already resolved on `V`
# — by this selector or by any other selector sharing it — contributes the regions it resolved to rather than a second
# copy. `_boundary_facet_regions` is a pure function of `(V, sides, tolerance)`, so the memo returns exactly what a
# fresh resolution would: the same regions, in the same order, region objects shared instead of duplicated.
function _facet_regions_for_selector(V::Space{D,T}, selector::BoundarySelector, facets::FacetResolver{D,T}) where {D,T}
    regions = FacetRegion{D,T}[]
    for sides in _facets(selector, Val(D))
        append!(regions, _resolve_face(facets, V, sides))
    end
    return regions
end

# The cached region lists of one kind of target, `K` the type of their key's head: `BoundarySelector` for the facet
# lists, `BoundaryMesh` for the surface lists, `Interface` for the interface lists. The per-kind diagnostics and
# `nquadpoints(model; kind)` read the cache through this, and `_region_count` sums the lists' lengths, so a region two
# targets share is counted once per target.
_cached_lists(regions::AbstractDict, ::Type{K}) where {K} = (list for (key, list) in regions if first(key) isa K)
_region_count(regions::AbstractDict, ::Type{K}) where {K} = sum(length, _cached_lists(regions, K); init=0)

# ── Diagnostics `prepare` records ─────────────────────────────────────────────

# `_kind_stats` over several region lists: the statistics of their concatenation, read one list at a time. Each list is
# its own call, a function barrier, so the lists of the model's region cache — whose values are typed only `Vector` —
# are each walked with their concrete region type rather than dispatching on every region.
function _kind_stats_of(lists)
    foldl((stats, regions) -> _kind_stats(regions, stats)::typeof(_NO_KIND_STATS), lists; init=_NO_KIND_STATS)
end

# Everything the diagnostics report about the rules of a region list, read off the regions' `kind`s and returned as
# `(cut, trimmed, failed, fallbacks, fallback_points, residual_max, min_relative_measure)`. The plan statistics and the
# facet statistics both compose from it; the two count "cut" differently, and both counts are here:
#
#   - `cut` — the regions `∂Ω` crosses (`_CUT_KINDS`), the volume plan's `cut_region_count`. `:cut_fitted` carry a
#     compressed cut rule; `:cut_fallback` (the cut rule itself), `:cut_failed` (empty Ω ∩ box at α = 0) and
#     `:cut_alpha_failed` (the same at α > 0) all failed to compress, and are `failed` (`_FAILED_KINDS`) too — so
#     `fit_failure_count` keeps meaning "the compression did not succeed" whichever rule the region ended up carrying.
#   - `trimmed` — every region that is not `:full`, the facet statistics' `cut_facet_region_count`: a face part of whose
#     area falls outside Ω, `:fictitious*` faces included.
#   - `fallbacks` and `fallback_points` — the `:cut_fallback` regions and their quadrature points. That is the assembly
#     cost of the safety net, and the number a caller should watch: it is bounded only by the cut rule's own point
#     count, not by nbasis.
#   - `residual_max` and `min_relative_measure` — over the `trimmed` regions, the largest moment-fit residual and the
#     smallest fraction of a region's own measure its rule integrates (`_region_measure`). A list that trims nothing
#     reports 0 and exactly 1 rather than a Gauss weight sum's last bit.
#
# Reading the kind rather than re-deriving a test from the rule is what makes the facet counts as sharp as the
# classifier behind the kind and no sharper — `FacetRegion`'s docstring states what that sharpness does and does not
# see — and independent of the rule itself, which is no longer a plain tensor product exactly on the faces it reports.
#
# Why the facet count is worth reporting at all (`trimmed`, summed over every cached selector, so a region two selectors
# share is counted once per selector, exactly as `facet_region_count` is). The grid-aligned facet *partition* is
# level-set-blind: `_boundary_facet_regions` selects its contributing levels with `_level_side_is_physical` — pure
# mesh-edge versus domain-edge geometry — and partitions the face over `_side_cells`. A `FacetRegion` is therefore the
# *whole* grid-aligned face of its parent cells, and where that face leaves Ω the rule on it is the moment fit on the
# facet's affine slice of Ω rather than an exact tensor product. Every consumer of a facet region inherits that: the L²
# Dirichlet mass and right-hand side, a Neumann / Robin / Nitsche term placed with `on = boundary(…)`, and
# `boundary_integral`. That fit is machine-exact where every restricted leaf is affine (planar) and accurate rather than
# exact on a curved cut (`boundary`'s docstring states the condition), so the count says how much of a model's boundary
# integration rests on a fitted rule, and `residual_max` says how well those fits reproduced the moments they were
# given, not how right those moments were.
#
# The classification is of the region's own *face*, not of its parent cells, and the distinction is not academic: a cut
# cell can have a face lying wholly inside Ω, and no trimming is needed there. Classifying the cells would make the
# number a loose upper bound — on the space-time cavity example at `SC_CELLS = 3` it reads 61 of 81 against a true 1,
# because the growing cavity cuts nearly every cell while touching only the `t = 0` and `t = T` faces.
#
# `stats` is where the counts start from: the statistics of the lists already read, so that `_kind_stats_of` can carry
# them from one list into the next.
function _kind_stats(regions, stats=_NO_KIND_STATS)
    (; cut, trimmed, failed, fallbacks, fallback_points, residual_max, min_relative_measure) = stats
    for region in regions
        kind = region.kind
        kind === :full && continue
        trimmed += 1
        cut += kind in _CUT_KINDS
        failed += kind in _FAILED_KINDS
        if kind === :cut_fallback
            fallbacks += 1
            fallback_points += length(region.weights)
        end
        residual_max = max(residual_max, Float64(region.residual))
        relative = _region_jacobian(region) * sum(region.weights) / _region_measure(region)
        min_relative_measure = min(min_relative_measure, Float64(relative))
    end
    return (; cut, trimmed, failed, fallbacks, fallback_points, residual_max, min_relative_measure)
end
const _NO_KIND_STATS = (; cut=0, trimmed=0, failed=0, fallbacks=0, fallback_points=0, residual_max=0.0,
                        min_relative_measure=1.0)

# The integration plans' half of the diagnostics, over every subdomain's plan, under the record's own field names. The
# aggregation is the identity on a single-domain problem: region and small-overlap counts sum, the minimum volumes take
# the global minimum, the moment-fit residual the global maximum of each plan's own (NaN-ignoring) one, and the cut
# statistics are `_kind_stats` over every plan's regions. The small-overlap records are concatenated, so every
# subdomain's offending regions are reported, and widened to `Float64`: a plan carries them in the model's scalar type
# `T`, and the record is `T`-agnostic. `plans` must be non-empty.
function _plan_stats(plans)
    kinds = _kind_stats_of(p.regions for p in plans)
    small = SmallOverlap{Float64}[SmallOverlap{Float64}(r.region, Float64(r.volume), Float64(r.relative_volume),
                                                        r.cover_count) for p in plans for r in p.small_overlaps]
    return (; integration_regions=sum(p -> length(p.regions), plans), small_overlap_count=length(small),
            small_overlaps=small, min_integration_volume=minimum(p -> Float64(p.min_volume), plans),
            min_relative_integration_volume=minimum(p -> Float64(p.min_relative_volume), plans),
            cut_region_count=kinds.cut, fit_failure_count=kinds.failed,
            moment_fit_residual_max=maximum(p -> p.moment_fit_residual_max, plans; init=0.0),
            cut_fallback_count=kinds.fallbacks, cut_fallback_points=kinds.fallback_points)
end

# Per-level count of cells deactivated by a `LevelMask` (mask and fictitious fold combined), across every distinct
# subdomain space in `problem_spaces` order — the same order the reindexed global level ids run. One entry per level,
# `0` for an unmasked level. Reported by `diagnostics` so the user can see how many cells each level dropped.
function _inactive_cell_counts(spaces)
    return Int[level.mask === nothing ? 0 : count(!, level.mask) for V in spaces for level in V.levels]
end

# Per-level count of raws eliminated by covered-mode pruning (`:coverage` or `:dedup`), one entry per id of each field's
# level block, concatenated in field order. That is the per-*field* granularity the `reduced_mode_counts` docstring
# states, not the per-space one `_inactive_cell_counts` walks, so the two vectors line up entry-for-entry only when each
# space carries one field. Pruning reads the space alone — never a field's components or Dirichlet data — so two fields
# over one space report the same counts, once each.
#
# Each level's raws are one range of the layout (`level_raws`), so a level is counted from its own elimination sources
# without a pass over every raw.
function _reduced_mode_counts(layout::SystemLayout)
    return Int[count(s -> s === :coverage || s === :dedup, view(f.dofs.elimination_source, f.dofs.level_raws[id]))
               for f in layout.fields for id in f.level_ids]
end

# ── Discretisation pin ────────────────────────────────────────────────────────
#
# `Model.version` is the token a `Solution` or `QuadField` carries so a consumer can ask "are you still a coefficient
# vector of this model's active numbering". A bare counter answers that only for an in-place mutation of one model.
# Every `prepare` used to hand back the literal `1`, so two *independently* prepared models were indistinguishable — and
# independently prepared models are the normal case, not a corner: `adapted`, `elevated`, `moved`, `estimate`'s twin and
# every rebuild inside an adaptive or transient loop produce them. A solution from one was then accepted on the other
# whenever the active-unknown counts happened to agree, which on a ladder is ordinary, because many structurally
# different masks give the same count. Measured: an `l2_error` of 3.2615e-3 reported where the truth was 3.2095e-3, and
# a complete `estimate` computed from the wrong coefficient vector, neither of them raising.
#
# So the token is SEEDED from a structural digest of the discretisation and still moved on in place. Two preparations of
# the same discretisation get the same seed, which is what a transient loop that re-prepares at an unchanged space and
# carries its state forward needs; two different discretisations get different seeds, so the carry-over raises; an
# in-place mutation still moves the token, so every raise the counter produced before is still produced.
#
# WHAT GOES IN — everything the active dof numbering is a function of:
#
#   * per level, as one tuple: id, role, mode, basis family name, the mesh's corner bits and cell counts, the activation
#     mask's bits, and the order field's palette, class map and nominal. An absent mask or class map — every cell
#     active, one order for every cell — is digested as `-1`, where an array's digest starts with its shape, so a
#     uniform unmasked level costs no per-cell rounds however many cells it has;
#   * per field: name, component count, offset, raw and active dof counts, and the `(raw, component) -> active id` map
#     itself — which is precisely the numbering a coefficient vector indexes into, and the only thing that separates two
#     models differing solely in *which* dofs a Dirichlet condition constrains rather than how many;
#   * the system's active-unknown total.
#
# Leaf semantics (`prepare`'s `prune`) are deliberately NOT digested as a flag, although they change which raws survive.
# They reach the digest through the layout instead, and that is the sharper reading of the same question: where the rule
# eliminates anything, the active counts and the `(raw, component) -> active id` map below already differ, so the two
# models separate; where it eliminates nothing the two layouts are byte-identical — same raws, same active count, same
# numbering — and a coefficient vector really is interchangeable between them. Folding the flag in on top would refuse
# that carry-over for a difference no consumer can observe.
#
# WHAT STAYS OUT, and this boundary is the load-bearing half: anything carrying object identity. Forms, closures,
# `Field` objects, `dirichlet` specs and their data, the `PhysicalDomain`. A problem rebuilt at an unchanged
# discretisation constructs all of them afresh every step, so digesting any of them would break the same legitimate loop
# a bare counter breaks, only from the other side. The geometry is not thereby ignored: its effect on the numbering is
# the fictitious fold, and the folded masks are digested above. A datum change goes through `update_dirichlet!`, which
# rewrites values over an unchanged numbering and deliberately does not move the pin.
#
# FNV-1a over the integers themselves rather than `Base.hash`, for the reason `test/test_graded_golden.jl` gives for its
# own digest: `Base.hash` carries no stability guarantee across Julia versions, and this value is displayed, goes into
# the reproducibility report (`diagnostics(model, solution)`, below) and may be quoted back in a bug report.
const _PIN_PRIME = 0x00000100000001b3
const _PIN_BASIS = 0xcbf29ce484222325

# `x % UInt64` is modular and total, so a `UInt16` order class, a `Bool` and a negative `Int` all fold without a range
# check.
function _pin(h::UInt64, x::Integer)
    v = x % UInt64
    for shift in 0:8:56
        h = (h ⊻ ((v >> shift) & 0xff)) * _PIN_PRIME
    end
    return h
end
_pin(h::UInt64, x::AbstractFloat) = _pin(h, reinterpret(UInt64, Float64(x)))
_pin(h::UInt64, s::Symbol) = foldl((a, c) -> _pin(a, UInt32(c)), String(s); init=_pin(h, 0x5f))
_pin(h::UInt64, t::Tuple) = foldl(_pin, t; init=_pin(h, length(t)))
# A `BitArray` folds through its packed chunks rather than bit by bit: the same value, 64× fewer rounds, and the size is
# mixed in so two masks that differ only in shape cannot collide on a shared chunk pattern.
_pin(h::UInt64, b::BitArray) = foldl(_pin, b.chunks; init=_pin(h, size(b)))
_pin(h::UInt64, a::AbstractArray) = foldl(_pin, a; init=_pin(h, size(a)))

# The digest itself. `spaces` are the *effective* (post-fold, re-indexed) subdomain spaces and `layout` the dof layout
# built from them, so this reads the discretisation as every consumer sees it rather than as the caller wrote it.
# Returned as a non-negative `Int` so the pin prints and compares like the counter it replaces, and so the `+= 1` a
# mutator applies cannot wrap it into the sign bit.
function _discretisation_pin(spaces, layout::SystemLayout)
    h = _pin(_PIN_BASIS, length(spaces))
    for V in spaces
        h = _pin(h, level_count(V))
        for level in V.levels
            o = level.orders
            h = _pin(h,
                     (level.id, level.role, _mode(level.basis), basis_name(level.basis), level.mesh.domain.lower,
                      level.mesh.domain.upper, level.mesh.cells, something(level.mask, -1), o.palette,
                      something(o.class, -1), o.nominal))
        end
    end
    for f in layout.fields
        h = _pin(h, f.name)
        h = _pin(h, f.components)
        h = _pin(h, f.offset)
        h = _pin(h, length(f.dofs.raw_keys))
        h = _pin(h, f.dofs.active_count)
        h = _pin(h, f.dofs.active_component)
    end
    return Int(_pin(h, layout.active_unknowns) & 0x7fff_ffff_ffff_ffff)
end

# ── Model display, conditioning estimates and accessors ───────────────────────

function Base.show(io::IO, model::Model{D}) where {D}
    print(io, "Model(D=", D, ", pin=0x", string(model.version; base=16), ", active=", active_unknowns(model.dofs),
          ", regions=", diagnostics(model).integration_regions, ", assembled=", model.matrix !== nothing, ")")
end

# Condition-number estimates of a small assembled matrix, `(plain, scaled)`: `cond(A)`, and the same on the
# symmetrically diagonal-scaled operator `D⁻¹ A D⁻¹` with `D = diag(√|A_ii|)`. Both are `NaN` for an empty matrix and
# above `max_size`, since `cond` needs the dense matrix and is only viable up to a few hundred unknowns: the 256 cap
# keeps the diagnostic free for unit tests and small examples while never silently dominating real-problem assembly
# cost. The matrix is densified once for both (`cond` copies its argument, so the scaling below reads the original).
#
# The scaled operator is what a Jacobi-preconditioned solver sees, and the half of the conditioning question that is
# about the basis rather than about the scale spread of the diagonal. On an immersed system those two halves are far
# apart and the distinction is not academic: a basis function whose support is almost entirely fictitious carries an
# almost-zero diagonal, which inflates `cond(A)` by the spread of that diagonal alone. Scaling removes exactly that, and
# what remains measures whether the surviving functions are near-dependent on Ω. `AssemblyDiagnostics` documents the
# measured gap between the two shipped families.
#
# It is also the *accurate* computation of the two on such a system. The unscaled matrix's smallest eigenvalue can sit
# below `λ_max · eps`, where a dense eigensolver returns roundoff — including a negative value on an operator that is
# positive definite — while the scaled matrix resolves the same mode comfortably. A scaled `NaN` beside a finite plain
# estimate means a zero diagonal entry, which is a dof carrying no energy at all rather than a conditioning verdict.
function _condition_estimates(matrix::SparseMatrixCSC; max_size::Int=256)
    n = size(matrix, 1)
    (n == 0 || n > max_size) && return NaN, NaN
    dense = Matrix(matrix)
    plain = Float64(cond(dense))
    scale = [sqrt(abs(dense[i, i])) for i in 1:n]
    any(iszero, scale) && return plain, NaN
    for j in 1:n, i in 1:n
        dense[i, j] /= scale[i] * scale[j]
    end
    return plain, Float64(cond(dense))
end

"""
    integration_plan(model::Model) -> IntegrationPlan

The **first** subdomain's cached integration plan (`first(model.subdomains).plan`). Plans are built by
[`prepare`](@ref) and refreshed by [`move!`](@ref) / [`activate!`](@ref) / [`deactivate!`](@ref), so once a model is
prepared every assembly call uses the same plan the dof layout was built against. For a single-domain model this is
*the* plan; a coupled model holds one plan per subdomain — use [`integration_plans`](@ref) to reach all of them.
"""
integration_plan(model::Model) = first(model.subdomains).plan

"""
    integration_plans(model::Model) -> Vector{IntegrationPlan}

Every subdomain's cached [`IntegrationPlan`](@ref), in `problem_spaces` order, read off `model.subdomains` into a fresh
vector (the plans in it are the model's own). The assembly volume pass integrates over these; each region is owned by
the fields on its subdomain intrinsically (`_parent_range` via `FieldLayout.level_ids`). For a single-domain model this
is a one-element vector whose only element is what [`integration_plan`](@ref) returns.
"""
integration_plans(model::Model) = [s.plan for s in model.subdomains]

"""
    dof_layout(model::Model) -> SystemLayout

Cached per-field dof layout, including constraint matrices and the active-enumeration table.
"""
dof_layout(model::Model) = model.dofs

"""
    active_unknowns(model::Model) -> Int

Number of active (post-constraint) dofs on `model` — the size of the global system the solver sees, and the length of
any coefficient vector consumed by [`solution`](@ref), [`assemble!`](@ref), or an external time integrator. Delegates to
`active_unknowns(model.dofs::SystemLayout)`.
"""
active_unknowns(model::Model) = active_unknowns(model.dofs)

# ── Model mutation ────────────────────────────────────────────────────────────

"""
    move!(model; level, to) -> Model

Move overlay `level` of a prepared `model` to box `to` in place, rebuilding its integration plan (reusing the
`tolerance`/`criterion` captured at [`prepare`](@ref)) and dof layout, and invalidating any assembled operators. The
base level (level 1) cannot be moved.

`move!` overwrites the old state; to transfer a solution onto the moved configuration, use [`moved`](@ref) to build a
separate target model and then [`transfer`](@ref).

Invalidation contract — the one the sibling mutators [`activate!`](@ref) and [`deactivate!`](@ref) mirror. `move!`

  - bumps `model.version`, so an outstanding [`Solution`](@ref) raises on reuse;
  - clears `model.matrix` and `model.rhs`, and replaces the assembly cache `model.assembly` (region lists, sparsity
    patterns, pooled scratch);
  - empties the Dirichlet projection cache, whose unknown sets and boundary mass belong to the dof layout being
    replaced;
  - rebuilds `model.prefold_space` at the new box, the subdomain record (the folded space and its integration plan),
    the dof layout, and the region cache `model.regions`;
  - installs a fresh [`AssemblyDiagnostics`](@ref) record (a reference taken before the move keeps reporting the old
    state).

The cut-cell moment-fit cache is deliberately *not* invalidated: it is keyed by region bounds and moment order, so a cut
region the move leaves unchanged reuses its rule instead of being refitted. A mutator keeps the model's own cache
objects — it overwrites the model rather than forking it, so no sibling model can lose rules to it — and the plan build
bounds them to the boxes the new configuration cuts, so a sweep of many moves costs one plan's rules and not the
sweep's.
"""
function move!(model::Model{D,T}; level::Integer, to::AxisBox{D,T}) where {D,T}
    _assert_single_domain(model, "move!")
    return _remodel!(model, _moved_prefold_space(model, level, to))
end

# Guard the positional-level mutators (`move!`, `activate!`, `deactivate!`) against multi-domain models. Those mutators
# address a level by its position in the single `model.problem.space`, which is ambiguous once a problem spans several
# subdomain spaces; supporting per-subdomain mutation is deferred. A single-domain model passes through silently. The
# derivations, the estimator, `L2Projection` and `QuadField`, which read that one space as the whole discretisation,
# refuse through the same guard.
#
# Single-domain means one prepared subdomain: every field lives on one space, compared by identity as `problem_spaces`
# deduplicates spaces. Read off `model.subdomains` rather than recomputed from the fields, because the guard runs on
# every `L2Projection` transfer, twice, and on every `QuadField`, where building the space vector allocated per call.
function _assert_single_domain(model::Model, op::AbstractString)
    length(model.subdomains) == 1 || throw(ArgumentError("$op is not yet supported for multi-domain (coupled) models"))
    return nothing
end

# Rebuild every piece of reusable state `model` carries from a derived pre-fold space, in place. This is the whole body
# of the in-place mutators: `move!`, `activate!` and `deactivate!` differ only in how they derive `space` from
# `model.prefold_space`, and everything after that is the rebuild `prepare` runs, so it is written once here rather than
# three times.
#
# The geometry comes from `model.prefold_space`, never from the effective `model.problem.space`. The effective masks
# already carry the fictitious fold, and `_apply_physical_fold` intersects (`active .& .!fictitious`), so folding them a
# second time can only ever remove cells: a cell dropped because the old overlay box put it outside Ω would stay dropped
# after a move that puts it wholly inside. Deriving from the pre-fold space makes a mutation reproduce `prepare` at the
# new configuration exactly, which is the contract all three mutators advertise. Everything that is *not* geometry —
# field names and component counts, blocks, loads, Dirichlet data — is read back from `model.problem`, which carries it
# unchanged through the fold and is kept current by `update_dirichlet!`. Every caller is single-domain
# (`_assert_single_domain`), so `prefold_space` is *the* space and `_rehome` re-homes every field onto the derived copy.
#
# The rebuilt model is then copied onto `model` field by field, by name, so the rule is "everything a `prepare` builds
# is replaced" and a field added to `Model` later is replaced with the rest rather than forgotten. A field that must
# survive a rebuild has to be excluded by name, as `version` is: the pin moves on from the model's own value, so an
# outstanding `Solution` raises. Everything else is what the rebuild should leave:
#
#   - `matrix` and `rhs` are `nothing` on the fresh model, so the assembled state is cleared;
#   - `plan_options` is the same object, and the subdomains' moment-fit caches are the model's own, passed in uncopied
#     (`_derived` gives the reason);
#   - the Dirichlet projection cache is a fresh empty one: its unknown sets, facet regions and mass all belong to the
#     dof layout being replaced, and the next `update_dirichlet!` rebuilds it against the new one;
#   - the facet resolver is replaced for the same reason and with a second effect: its keys hold spaces by identity, so
#     taking the fresh model's resolver retires every entry resolved on the space this rebuild has just superseded,
#     instead of carrying them forward as ballast no lookup can reach.
#
# `setproperty!` converts as a field assignment does, so a rebuilt problem of another concrete type raises exactly as
# `model.problem = …` would.
function _remodel!(model::Model{D,T}, space::Space{D,T}) where {D,T}
    fresh = _derived(model, space; caches=_fits(model))
    for name in fieldnames(Model)
        name === :version || setproperty!(model, name, getfield(fresh, name))
    end
    model.version += 1
    return model
end

# A model of `model`'s problem prepared over `space`: the one builder behind every model derived from another — the
# non-mutating derivations, the in-place mutators (`_remodel!`) and `estimate`'s enriched model. The forms, field
# channels and Dirichlet data are `model`'s (`_rehome`); the plan options are too, unless `options` says otherwise, as
# `estimate`'s does when it exempts a level from pruning.
#
# `caches` has no default, so every caller states what happens to `model`'s moment-fit caches. Every derivation hands
# them on rather than starting empty, for the reason `move!` gives for not invalidating them: they are keyed by region
# bounds and moment order, and a mask edit, an order edit or a move changes which cells are active and never the
# geometry, so every fitted cut-cell rule is still valid. Rebuilding them instead refits every cut region on every
# step, which on an immersed model is the dominant cost of the step. Whether the new model *shares* them or gets a copy
# is the policy each caller states:
#
#   - the non-mutating derivations ([`adapted`](@ref), and through it [`elevated`](@ref) and [`moved`](@ref)) pass a
#     copy, because source and target are then both live and each builds a plan against its own cut regions. Sharing
#     would make the fork's plan build evict the rules the source's own plan is still using — an `O(#cut regions)` copy
#     against a full refit;
#   - the in-place mutators pass the model's own caches through `_remodel!`, since they overwrite the model rather than
#     fork it, so there is no sibling to protect and the model keeps one cache object for its whole life;
#   - `estimate` (`adaptivity.jl`) shares, so the elevated-order rules its throw-away enriched model fits land on the
#     source model — except where a live refinement twin changes the partition, so that the enriched plan would evict
#     every rule the source holds; there it passes a copy and merges the rules fitted on it back into the source's.
#
# `point_base` numbers the new model's quadrature points (`AssemblyCache`). Only `estimate` moves it off 0.
function _derived(model::Model{D,T}, space::Space{D,T}; options::NamedTuple=model.plan_options,
                  caches::Vector{_MomentFitCache{D,T}}, point_base::Int=0) where {D,T}
    return _prepared_model(_rehome(model.problem, _ -> space), options, caches; point_base)
end

# The moment-fit caches of `model`'s subdomains, in order: what a derivation hands `_derived`, which states whether the
# new model shares them or gets copies. The vector is fresh; the caches in it are the model's own objects.
_fits(model::Model) = [s.fits for s in model.subdomains]

# The moved pre-fold space `move!` installs and `moved` prepares. Split out so the two spell the move once, and so
# `moved` reads as what it is: the same derivation as `move!`, prepared into a second model instead of installed.
function _moved_prefold_space(model::Model{D,T}, level::Integer, to::AxisBox{D,T}) where {D,T}
    tolerance = get(model.plan_options, :tolerance, GeometryTolerance(T))
    return moved_space(model.prefold_space; level, to, tolerance)
end

"""
    moved(model; level, to) -> Model

Return a new prepared [`Model`](@ref) with overlay `level` moved to box `to`, reusing the source problem's forms and
boundary data. Unlike [`move!`](@ref), the source model is left intact, so it can serve as the source of a
[`transfer`](@ref):

    target = moved(model; level=2, to=box((0.1,), (0.6,)))
    target_solution = transfer(solution, model, target)

Like [`move!`](@ref), `moved` addresses the level by its position in the model's single space and is therefore
restricted to single-domain models; on a coupled model it throws `ArgumentError`.

It derives the moved space and hands it to [`adapted`](@ref)`(model, space)`, which is the one builder every derived
model goes through, so `moved` reuses the source's fitted cut-cell rules on the same terms an adaptive step does: a cut
region the move leaves bit-identical is not refitted. The target gets its own copy of the caches, so the source keeps
every rule it has.
"""
function moved(model::Model{D,T}; level::Integer, to::AxisBox{D,T}) where {D,T}
    # Same restriction as `move!`, and for the same reason: `level` indexes one space's level tuple, which is ambiguous
    # once a problem spans several subdomain spaces. `_rehome` re-homes *every* field onto the moved space, so on a
    # coupled model there is no answer to give rather than a wrong one — hence a raise, not a pick. It is asserted
    # here, before `adapted` would assert it under its own name, so the message says `moved`.
    _assert_single_domain(model, "moved")
    return adapted(model, _moved_prefold_space(model, level, to))
end

"""
    activate!(model; level, cells) -> Model

Mark `cells` on `level` as active in place. `cells` is a selection of cells, not a whole mask, and accepts every shape
the `active=` keyword of [`space`](@ref) documents: an iterable of `CartesianIndex{D}`, a predicate
`(cell_box, cell_index) -> Bool`, an `AbstractArray{Bool,D}` or a [`LevelMask`](@ref) matching the level's cell grid
(`true` selects the cell), or `nothing`, which selects every cell. Currently-active cells in the selection are
unchanged, and cells outside it keep their state.

Bumps `model.version`, so an existing [`Solution`](@ref) raises on reuse, and runs the same invalidation tail as
[`move!`](@ref) — see that docstring for exactly what is cleared, rebuilt, and deliberately kept. The one difference is
the pre-fold record: `move!` rebuilds it at the new box, while this records the cell flip on it.

Like [`move!`](@ref), it addresses the level by position in the model's single space and therefore raises
`ArgumentError` on a multi-domain (coupled) model.

The flip is recorded on the *pre-fold* mask and the result is folded against the geometry afresh, so a mutated model is
exactly the [`prepare`](@ref) of the mask the caller has built up. On a space carrying a [`physical_domain`](@ref),
activating a cell the level set classifies as fictitious is therefore a no-op on the effective space: the request is
kept (`active_cells(model; level, effective = false)` shows it, and a later [`move!`](@ref) that brings the cell inside
Ω honours it) but the geometry is not overridden. An override would reconstruct exactly the configuration
[`physical_domain`](@ref) refuses to build in the first place — a fully fictitious cell kept active under the strict-cut
path receives no quadrature and makes the system singular, which is why `keep_fictitious = true` requires `alpha > 0` —
and it would survive only until the next re-fold, since nothing a rebuild reads records it.
"""
activate!(model::Model{D,T}; level::Integer, cells) where {D,T} = _update_mask!(model, level, cells, true)

"""
    deactivate!(model; level, cells) -> Model

Mark `cells` on `level` as inactive in place. See [`activate!`](@ref) for the accepted shapes of `cells`, the
invalidation contract, and the interaction with a model's `physical_domain`. `cells = nothing` selects every cell, so it
leaves the level dormant.
"""
deactivate!(model::Model{D,T}; level::Integer, cells) where {D,T} = _update_mask!(model, level, cells, false)

# Activate / deactivate `cells` on `level_index` and rebuild the model's reusable state. The flip is recorded on the
# *pre-fold* mask — the caller's own record of what was asked for — and `_remodel!` folds the result against the
# geometry afresh, exactly as `prepare` would at that mask. So on a model carrying a `PhysicalDomain`, activating a cell
# the level set classifies as fictitious is a no-op on the effective space and stays visible through
# `active_cells(model; level, effective = false)`, instead of overriding the geometry until the next `move!` re-folded
# it away. Any outstanding `Solution` becomes stale.
#
# The update is word-wise on the level's active bits, against the selection `_selection` reads from every shape
# `active =` takes. The result goes through `_collapse_mask`, so the no-mask fast path survives bouncing in and out of
# the mask, and the selection is read even when it changes nothing, so a malformed one raises whatever the level's
# current state.
function _update_mask!(model::Model{D,T}, level_index::Integer, cells, value::Bool) where {D,T}
    _assert_single_domain(model, "activate! / deactivate!")
    V = model.prefold_space
    k = _check_level(V, level_index)
    on = active_cells(V; level=k)
    selected = _selection(cells, V.levels[k].mesh)
    value ? (on .|= selected) : (on .&= .!selected)
    return _remodel!(model, _relevel(V, k; mask=_collapse_mask(on)))
end

"""
    adapted(model::Model, level => cells, ...) -> Model
    adapted(model::Model, depths::AbstractArray{<:Integer,D}; grade=0) -> Model

A fresh model whose activation masks are [`adapt`](@ref)ed as the spec says, leaving `model` untouched so a solution can
be carried across with [`transfer`](@ref). The non-mutating counterpart of [`activate!`](@ref) and
[`deactivate!`](@ref), standing to them as [`moved`](@ref) does to [`move!`](@ref) — and the reason it exists at all:
both mutators overwrite the model in place, so neither can produce the source/target pair a transfer needs.

Setting several levels in one call is one rebuild, where the same edit through the mutators is one integration plan and
one dof layout per level touched. The tolerance and criterion captured at [`prepare`](@ref) are reused, as are the
moment-fit caches: an adaptive step changes which cells are active and no geometry, so every fitted cut-cell rule is
still valid.

The spec is applied to the model's *pre-fold* space, which is the record of what the caller asked for; [`prepare`](@ref)
then folds the result against the geometry afresh. Read masks back with `active_cells(model; level, effective=false)`
for the same reason — an effective mask fed back in would record the fictitious fold as the caller's own exclusion.

Single-domain only, as [`moved`](@ref) is: a level index has no meaning once a problem spans several subdomain spaces.

# A transient step

```julia
m = active_cells(model; level=4, effective=false)
m[marked] .= true
target = adapted(model, 4 => m)
u = transfer(u, model, target; via=L2Projection())
model = target
```

`L2Projection` is the transfer for an adaptive step in *both* directions, not only when coarsening. Refining deepens
coverage, so covered-mode pruning eliminates the *parent's* modes in the target: the target's active basis is not a
superset of the source's even though the target's span contains the source's, and [`Rewire`](@ref) — which matches dofs
— has nothing to copy into. It raises in strict mode and silently drops those coefficients otherwise.

The transfer is exact to roundoff for integrated Legendre and for maximum-regularity B-splines, because reduction
removes exactly the redundancy and the source field is still in the target's span. It is not exact for
reduced-continuity B-splines (`bspline(; continuity = 0)` or `1` at degree 3), where a refine step was measured to lose
about 2e-5 per step — an amount comparable with the discretisation error, so it accumulates over a transient. That
behaviour is a property of the family's own reduction rule rather than of this call.

On a model carrying a [`PhysicalDomain`](@ref) there is currently no working transfer at all: `L2Projection` rejects an
immersed target and `Rewire` has no counterpart for the eliminated modes. Adaptive stepping on an immersed model
therefore has to re-solve rather than carry state forward.
"""
function adapted(model::Model{D,T}, spec...; kwargs...) where {D,T}
    _assert_single_domain(model, "adapted")
    return adapted(model, adapt(model.prefold_space, spec...; kwargs...))
end

"""
    adapted(model::Model, space::Space) -> Model

Prepare `space` as a new [`Model`](@ref) carrying `model`'s problem, tolerance and plan options. This is the one-rebuild
form of an hp step: compose the h- and the p-half at the `Space` level and hand the result here, rather than paying a
full `prepare` for the intermediate.

It is also the single builder every non-mutating derivation of a model goes through — the spec forms above,
[`elevated`](@ref) and [`moved`](@ref) each derive a space from `model.prefold_space` and call this — so they all get
the same guards, the same captured plan options and the same cache lineage, and the package has one path from a model to
its successor rather than one per verb.

`space` must be derived from `model.prefold_space` — by [`adapt`](@ref), [`elevate`](@ref), [`moved_space`](@ref), or
any composition of them. The moment-fit caches are reused, and their keys are `(box, moment_order)` with the geometry
deliberately left out, on the reasoning that one cache sees exactly one [`PhysicalDomain`](@ref) for its whole life. A
space carrying a different domain or a different fold would break that, so it is refused rather than silently fitted
with the wrong cut rules. The new model gets its own *copy* of the caches, seeded with everything the source has fitted:
both models are live afterwards, and a shared cache would have each one's plan build evict the other's rules.
"""
function adapted(model::Model{D,T}, space::Space{D,T}) where {D,T}
    _assert_single_domain(model, "adapted")
    prefold = model.prefold_space
    space.physical === prefold.physical ||
        throw(ArgumentError("adapted(model, space) reuses this model's moment-fit caches, whose keys omit the " *
                            "geometry because one cache sees a single `physical` for its whole life; the space " *
                            "handed in carries a different one. Derive it from `model.prefold_space` with " *
                            "`adapt` / `elevate`."))
    space.domain == prefold.domain ||
        throw(ArgumentError("adapted(model, space) expects a space over this model's domain $(prefold.domain); " *
                            "got $(space.domain)."))
    # Seeded warm from the source, but the target's own object: both models are live from here on, and each builds a
    # plan that bounds its cache to its own cut boxes. Sharing would make the fork evict the rules the source's plan is
    # still using — `_derived` has the lineage rule in full.
    return _derived(model, space; caches=copy.(_fits(model)))
end

"""
    elevated(model::Model, level => order, ...) -> Model

The [`elevate`](@ref) counterpart of [`adapted`](@ref): a new [`Model`](@ref) whose named levels carry the given
polynomial orders, prepared with the same problem, tolerance and plan options as `model`. `order` takes every shape
`elevate` accepts.

The spec is applied to the model's *pre-fold* space for the same reason [`adapted`](@ref)'s is, and the result is handed
to [`adapted`](@ref)`(model, space)`, so the moment-fit caches are reused on the same terms: a p-adaptive step changes
which basis functions exist and no geometry, so every fitted cut-cell rule is still valid.

Single-domain only, as [`adapted`](@ref) is. An hp step wants both halves at once, and going through this call and
[`adapted`](@ref) in turn builds and throws away a complete intermediate model. Compose at the `Space` level instead —
where the compose itself costs microseconds — and prepare once:

```julia
target = adapted(model, elevate(adapt(model.prefold_space, 4 => m), 4 => p))
```
"""
function elevated(model::Model{D,T}, spec...) where {D,T}
    _assert_single_domain(model, "elevated")
    return adapted(model, elevate(model.prefold_space, spec...))
end

# ── Reading a model's levels ──────────────────────────────────────────────────

"""
    cell_orders(model; level) -> Array{NTuple{D,Int},D}

Per-axis polynomial order of every cell of `level`, as a fresh array. `level` is a position in the model's space, as for
[`activate!`](@ref). A level whose order is uniform returns that order repeated over the cell grid.

Reads the model's *pre-fold* space, so the array can be edited and handed back to [`elevated`](@ref) without laundering
anything into it — unlike a mask, an order field is never touched by the fictitious fold, so the two spaces agree and
the distinction `active_cells`'s `effective` keyword draws does not arise here.

Unlike the mutators this reads the problem's representative space rather than raising on a coupled model, so on a
multi-domain model it reports the first subdomain.
"""
cell_orders(model::Model; level::Integer) = cell_orders(model.prefold_space; level=level)

"""
    active_cells(model; level, effective=true) -> BitArray{D}

Return a `BitArray{D}` indicating which cells of `level` are active. `level` is a position in the model's space, as for
[`activate!`](@ref). Returns a copy so caller mutations do not leak into the model. Levels without a mask return an
all-true array.

With `effective = true` (the default) the mask returned is the effective one: on a space carrying a
[`PhysicalDomain`](@ref) it already has the fictitious fold applied, so a cell can read inactive because the caller
excluded it or because the level set put it outside Ω. Pass `effective = false` to read the *pre-fold* mask instead —
the caller's own selection, with no geometry folded in.

The distinction matters whenever a mask is read back and written again, which is what an adaptive step does. Feeding an
effective mask to [`adapt`](@ref) or [`adapted`](@ref) launders the fold into user intent: the cells the geometry
switched off are recorded as cells the caller excluded, and the next fold cannot undo that. Round-tripping through
`effective = false` is an identity; through `effective = true` it is not.

Unlike the mutators this reads the problem's representative space (the first field's) rather than raising on a coupled
model, so on a multi-domain model it reports the first subdomain.
"""
function active_cells(model::Model; level::Integer, effective::Bool=true)
    return active_cells(effective ? model.problem.space : model.prefold_space; level)
end

"""
    support_extension(model::Model, cells; level) -> BitArray{D}

The `Model` spelling of [`support_extension`](@ref)`(::Space, cells; level)`, for a marking loop that has a prepared
model rather than a bare space. The dilation depends only on the level's mesh and basis family, both of which a fold
leaves alone, so this reads the pre-fold space and the answer is the one to hand back to [`adapt`](@ref) or
[`activate!`](@ref) — neither of which may be fed a laundered fold (see [`active_cells`](@ref)).
"""
support_extension(model::Model, cells; level::Integer) = support_extension(model.prefold_space, cells; level)

"""
    dilate(model::Model, cells; level, by) -> BitArray{D}

The `Model` spelling of [`dilate`](@ref)`(::Space, cells; level, by)`. Reads the pre-fold space for the same reason
[`support_extension`](@ref) does: the result is going back to [`adapt`](@ref) or [`activate!`](@ref), and neither may be
fed a mask the fictitious fold has already written over.
"""
dilate(model::Model, cells; level::Integer, by) = dilate(model.prefold_space, cells; level=level, by=by)

# ── Public: refresh Dirichlet values without rebuilding the model ─────────────

"""
    update_dirichlet!(model::Model, dirichlet) -> Model

Refresh the *values* of a prepared model's Dirichlet conditions in place, reusing the existing integration plan, dof
enumeration, region cache, and immersed-boundary moment-fit cache.

The argument `dirichlet` is a vector or tuple of [`DirichletCondition`](@ref)s that must structurally match the
conditions the model was prepared with: the same number of conditions in the same order, each with identical `boundary`,
`field`, and `component`. Only the `value` of each condition is allowed to differ. Any structural mismatch throws
`ArgumentError` and asks the caller to rebuild through [`prepare`](@ref) instead.

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

For the same problem, calling `update_dirichlet!` is typically orders of magnitude cheaper than re-running
`prepare(problem)` because the per-step rebuild of the integration plan and the moment-fit cache is skipped — only
the right-hand side of the boundary-trace projection on the constrained-dof subspace is reassembled and
back-substituted.

# Semantics

  - The constrained-dof set is *not* rediscovered. It depends only on the boundary selector + field topology, both of
    which the structural check pins.
  - `layout.constrained_values` is refilled by solving `M c = b` for the new `b_i = ∫ g(x) φ_i dS`. The boundary mass
    `M`, and the sampled basis traces that `b` is accumulated from, depend on the mesh alone, so the first call caches
    them on the model as a [`DirichletProjection`](@ref) and every later call reuses them — exactly, not merely to
    within roundoff. The facet regions are therefore walked and the mass factorised once per mesh, not once per
    increment; and the regions themselves come from the model's `FacetResolver`, so even that first walk resolves no
    face [`prepare`](@ref) has not already resolved.
  - The `dirichlet_trace_factors` and `unsupported_dirichlet_dof*` entries of [`diagnostics`](@ref) are refreshed. They
    too describe the boundary mass, so an increment restates them — except the *first* one on a model prepared with
    homogeneous data, which runs the trace solve for the first time and is where they are established.
  - `model.matrix` and `model.rhs` are cleared so the next [`assemble!`](@ref) (or `assemble_vector` / `assemble_matrix`
    call) picks up the new constrained data through the standard column-elimination path.
  - `model.version` is *not* bumped: existing [`Solution`](@ref) and [`QuadField`](@ref) handles stay valid against the
    updated layout.
  - The model's stored `problem` is replaced with a fresh `Problem` carrying the new Dirichlet list, so
    `model.problem.dirichlet` reflects the current state for inspection and diagnostics.

# Errors

Throws `ArgumentError` if `dirichlet` is structurally incompatible with `model.problem.dirichlet` (different length, or
any condition moves boundary / field / component). The error message points the caller at [`prepare`](@ref). Also throws
`ArgumentError`, naming the field, if a constant tuple or array datum does not carry one entry per component of its
field (see [`dirichlet`](@ref)). Either check runs before anything is touched, so a rejected call leaves the model as it
was.
"""
function update_dirichlet!(model::Model{D,T}, dirichlet) where {D,T}
    new_dirichlet = collect(dirichlet)
    _check_dirichlet_update_compatibility(model.problem.dirichlet, new_dirichlet)
    # The one part of `Problem`'s check of its conditions the structural check does not cover: a value's shape. Each new
    # condition belongs to the field its old counterpart did (`field` is pinned), so its component count is read off
    # that field's layout. The layout's field vector is concretely typed, so this costs no allocation, where indexing
    # the problem's field tuple at run time did: with three conditions on a scalar and a 2-component field, that added
    # six allocations (3.3 kB) to an increment; this loop takes 0.12 µs of its 17 µs (Apple M2 Pro).
    for condition in new_dirichlet
        owner = condition.field === nothing ? only(model.dofs.fields) : _field_layout(model.dofs, condition.field)
        _check_dirichlet_datum(condition, owner.components, owner.name)
    end

    # Replace `model.problem` (immutable) with a fresh copy carrying the new dirichlet list. The inner constructor
    # sidesteps the validation walk in `Problem(fields; …)` — fields, blocks, loads, and the symmetric flag were already
    # checked when the model was prepared, and the conditions have just been checked above.
    p = model.problem
    model.problem = typeof(p)(p.space, p.fields, p.blocks, p.loads, new_dirichlet, p.symmetric)

    # Re-project values into each field's existing DofLayout. The constrained-dof set is unchanged, so
    # `_project_dirichlet_values!` only refills `layout.constrained_values` — it does not touch the raw-dof index, the
    # active enumeration, or any of the boolean constraint masks. Each field is projected against its OWN space (a
    # coupled multi-domain problem gives every field an independent space); `model.problem.space` is only the first
    # field's and does not cover the others' cells — mirror the per-field routing `prepare` uses.
    #
    # The structural check above pins every quantity the field's `DirichletProjection` depends on, so the cached one —
    # if this is not the first update on the field — is handed straight back in and only the right-hand side is rebuilt.
    #
    # Each projection's trace-solve verdicts are read off it as it comes back, in field then component order, exactly as
    # `prepare` reads the ones it builds.
    trace_factors, unsupported = Symbol[], UnsupportedDirichletDof[]
    for field_layout in model.dofs.fields
        name = field_layout.name
        cached = get(model.dirichlet_projections, name, nothing)
        projection = _project_dirichlet_values!(field_layout.dofs, _field_space(model.problem, name),
                                                model.facet_resolver, _dirichlet_for_field(model.problem, name), cached)
        model.dirichlet_projections[name] = projection
        _append_trace_stats!(trace_factors, unsupported, name, projection, field_layout.components)
    end
    # A model prepared with homogeneous data has run no trace solve, so this is where its first increment establishes
    # the branch and the unsupported dofs. Later increments restate the same two answers: both describe the boundary
    # mass, which the structural check above pins.
    diag = model.diagnostics
    diag.dirichlet_trace_factors, diag.unsupported_dirichlet_dofs = trace_factors, unsupported
    diag.unsupported_dirichlet_dof_count = length(unsupported)

    # Clear cached operators. The RHS depends on the constrained values via column elimination, so the next assembly
    # must rebuild it. The matrix would technically still be valid (Dirichlet only moves entries into the RHS at
    # assembly time), but dropping it too means the next `solve!` cannot silently use a stale operator if the caller
    # also changes blocks or loads between steps.
    #
    # `model.assembly` is deliberately retained: `update_dirichlet!` only rewrites constrained *values*, leaving the
    # constrained-dof set — and therefore `active_unknowns` and every region's active dofs — fixed, so the cached region
    # lists and sparsity patterns stay structurally valid and the next assembly reuses them (the version is not bumped
    # either). Nothing in the cache holds a constrained value: assembly reads the elimination values and a pivot's
    # Dirichlet branches from the dof layout, which this function has just rewritten in place, at numeric time.
    model.matrix = nothing
    model.rhs = nothing

    return model
end

# ── Diagnostics reporting ─────────────────────────────────────────────────────

"""
    diagnostics(model::Model) -> AssemblyDiagnostics
    diagnostics(model::Model, solution; exact=nothing) -> NamedTuple

The one-argument form returns the model's cached `AssemblyDiagnostics` record. The two-argument form composes a
reproducibility-report `NamedTuple` that bundles the diagnostics with per-level metadata, solver residual from the
solution, and (optionally) the relative L² error against `exact(x)`. The result is the natural artefact to include in PR
descriptions, paper figures, and regression tests.

The report's keys are the record's fields in their declaration order (see [`AssemblyDiagnostics`](@ref)), with `levels`
after `dimension` and `raw_dofs` — every field's raw dof slots, counted before any elimination — after
`active_unknowns`, followed by `residual_norm` (the solution's) and `l2_error` (`nothing` without `exact`).

Each `levels` entry carries `id`, `role`, `cells`, `order` (the level's nominal per-axis maximum), `order_palette` (the
distinct per-cell orders, a one-element vector on a uniform level), `mode`, `basis`, `domain`, `nested` — whether every
higher level's nodes coincide with this level's where the two overlap — and `raw_functions` and `active_functions`.
`nested` is the geometric half of what makes leaf semantics lossless, and it is reported per level because
[`move!`](@ref) can void it silently; see [`is_nested`](@ref).

`raw_functions` counts the basis functions the level enumerated on its active cells, and `active_functions` those of
them that survived every elimination: the artificial overlay boundary, leaf-semantics pruning and physical Dirichlet
conditions alike. Both count functions, not unknowns; a function of a field with `k` components carries `k` unknowns. A
level is counted once however many fields share its space, and both numbers are read from the first field over that
space, so with several fields over one space `active_functions` reflects the Dirichlet conditions of the first of them.
A dormant level, with no active cell, reads 0 / 0; a level that enumerated functions and kept none reads `n` / 0 and
contributes nothing to the system. On a maximal-continuity spline level the usual cause is a region `p` cells wide or
less, which holds no function whose whole support fits inside it: the level costs a mesh and buys nothing.
"""
diagnostics(model::Model) = model.diagnostics

function diagnostics(model::Model{D,T}, solution; exact=nothing) where {D,T}
    _checked_coefficients(solution, model)
    diag = diagnostics(model)
    error = exact === nothing ? nothing : l2_error(solution, model, exact)
    tol = get(model.plan_options, :tolerance, GeometryTolerance(T))
    # Level reports for every subdomain space, in subdomain order, which is global level-id order. Built per space
    # because nesting is a relation between the levels of one space; a coupled problem's spaces are independent stacks.
    levels = Any[_level_report(level, s.space, tol, model.dofs) for s in model.subdomains for level in s.space.levels]
    # The record's fields in declaration order, `levels` and `raw_dofs` placed after the two they describe: `merge`
    # keeps the first argument's keys where they are and appends the rest in order.
    record = NamedTuple{fieldnames(AssemblyDiagnostics)}(ntuple(i -> getfield(diag, i),
                                                                Val(fieldcount(AssemblyDiagnostics))))
    return merge((; dimension=diag.dimension, levels, active_unknowns=diag.active_unknowns,
                  raw_dofs=raw_dof_count(model.dofs)), record,
                 (; residual_norm=solution.diagnostics.residual_norm, l2_error=error))
end

# Compact per-level reproducibility report used inside `diagnostics(model, solution)`: enough to know which basis,
# order, mode, cell count and domain a level had at solve time, plus whether it was nested under.
#
# `nested` is the geometric half of the condition leaf semantics want — every higher level's node coordinates agree with
# this level's where they overlap. It is reported per level rather than per space so a violation can be localised, and
# it is worth reporting at all because it is otherwise invisible: `move!` can void it on a stack that was nested when it
# was built, and the resulting loss shows up as neither a residual nor a rank deficiency. See [`is_nested`](@ref).
#
# `raw_functions` and `active_functions` count the functions the level *enumerated* and those that *survived*. Both, and
# not only the second, because the two answer different questions and the pair is what separates them:
#
#   raw = 0, active = 0     the level is dormant — no active cell, so nothing
#                           was enumerated. A ladder's unwoken rungs look like
#                           this and it is not a problem.
#   raw > 0, active = 0     the level enumerated functions and nothing of it
#                           reaches the system. For a maximal-continuity spline
#                           level the usual cause is the thin case: a region `p`
#                           cells wide or less holds no function whose whole
#                           support fits inside it, so the level costs a mesh and
#                           buys nothing.
#
# `active` is "survived every elimination", not "survived the artificial boundary": it is read off `active_component`,
# which is zero for an artificial-boundary constraint, for leaf-semantics pruning, and for a physical Dirichlet
# condition alike. A one-cell order-1 level with Dirichlet data on all four of its faces therefore reports 4 / 0
# although it has no artificial boundary at all, and an ordinary 4×4 order-2 Poisson level reports 49 / 81 with the 32
# missing functions Dirichlet. The signature narrows the cause; it does not name it.
#
# Counted per *function* rather than per unknown, because that is the quantity the basis families' dof ledgers are
# stated in — `∏_d (n_d − p_d)` for a selected spline level, say. On a field with `components > 1` a function carries
# one unknown per component, so the entries sum to `active_unknowns` only on scalar fields.
#
# Both are read from the first field whose level-id block holds the level, so a level is counted once however many
# fields share its space: summing over fields reported every level doubled on a two-field problem, while a two-component
# vector field over the same space — the same dof count, spelled differently — reported it once. Fields over ONE space
# share its one dof topology (`_system_layout`), so `raw_functions` is the same whichever of them is read;
# `active_functions` also reflects the Dirichlet conditions of the field it is read from, the first one over the space.
# Fields over *different* spaces hold disjoint level-id blocks (`_reindex_space_levels`), so the first field holding the
# id is a field over the level's own space. A level's functions are one range of the layout (`level_raws`), so it is
# counted from that range alone; a dormant level's range is empty and reads 0 / 0.
function _level_report(level, V::Space, tol::GeometryTolerance, layout::SystemLayout)
    nested = all(k -> k.id <= level.id || _nested_over(level, k, tol), V.levels)
    dofs = layout.fields[findfirst(f -> level.id in f.level_ids, layout.fields)].dofs
    raws = dofs.level_raws[level.id]
    # `order` stays the level's nominal (maximum) per-axis order, so an existing reader keeps reading the same field
    # with the same meaning. `order_palette` carries the distinct per-cell orders — a one-element vector on a uniform
    # level — so a graded run is reproducible from the report without the entry's type varying from level to level.
    return (; id=level.id, role=level.role, cells=level.mesh.cells, order=nominal_order(level),
            order_palette=copy(level.orders.palette), mode=_mode(level.basis), basis=basis_name(level.basis),
            domain=level.mesh.domain, nested=nested, raw_functions=length(raws),
            active_functions=count(r -> any(!iszero, view(dofs.active_component, r, :)), raws),)
end
