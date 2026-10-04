# Solution transfer between [`Model`](@ref)s. Two backends are shipped:
#
#   * [`L2Projection`](@ref) — variational L² projection, the default.
#     Solves the target-side L² problem
#
#         find u_T ∈ V_T such that ∫_Ω u_T v dx = ∫_Ω u_S v dx ∀ v ∈ V_T,
#
#     using a partition of `Ω` that respects both the source and the
#     target meshes (so basis-function products are smooth on every
#     integration region). This produces the orthogonal projection of
#     the source field onto the target's active space, which is the
#     "best" transfer in the L² sense and the correct variational move
#     when the target's basis cannot represent the source exactly.
#   * [`Rewire`](@ref) — direct raw-key coefficient matching. Each
#     target dof whose `TensorDofKey` matches an active source dof
#     inherits the source coefficient; genuinely new dofs and freshly-
#     uncovered constraints start at zero. Lossless when the target's
#     active basis contains the source's — typically after a monotone
#     `activate!` — and avoids the projection cost / smoothing in that
#     case.

# ── Transfer regions ──────────────────────────────────────────────────────────

"""
    TransferRegion{D,T}(box, target_parents, source_parents, quadrature)

One admissible integration region for a source-to-target L²
projection. Mirrors [`VolumeRegion`](@ref) but carries *two*
parent lists — one per side of the projection — so the assembly hot
loop can evaluate the source field and the target trace
simultaneously without re-walking the mesh.

Fields:

  - `box::AxisBox{D,T}` — the region's physical-frame extent. Built
    from the *union* of source and target mesh boundaries so the
    integrand is smooth across the box.
  - `target_parents` — active target levels whose cell contains the
    box midpoint, with the box in each parent's reference frame.
  - `source_parents` — same, for the source. May be empty when the box
    lies in part of `Ω` the source did not cover (e.g. after an
    overlay activation extended the target).
  - `quadrature::TensorQuadrature{D,T}` — shared cached tensor Gauss
    rule, sized by the per-axis maximum recommended quadrature order
    across both sides.
"""
struct TransferRegion{D,T<:Real}
    box::AxisBox{D,T}
    target_parents::Vector{ParentRef{D,T}}
    source_parents::Vector{ParentRef{D,T}}
    quadrature::TensorQuadrature{D,T}
end

# Sanity-check the source / target compatibility for a transfer: the same
# space bounding box (`Space.domain`; the immersed `Space.physical` is not
# compared here), the same number of fields, and the same field names with
# the same per-field component counts. The name check is implicit in
# `_field_layout`, which raises on a source field the target does not carry.
# Lets backends fail early with a clear message instead of producing wrong
# numbers downstream. On a coupled model `problem.space` is the
# *representative* (first field's) space, so only that subdomain's box is
# compared.
function _assert_transfer_compatible(source_model::Model{D,T}, target_model::Model{D,T}) where {D,T}
    source_model.problem.space.domain == target_model.problem.space.domain ||
        throw(ArgumentError("source and target models must have the same physical domain"))
    length(source_model.dofs.fields) == length(target_model.dofs.fields) ||
        throw(ArgumentError("source and target models must have the same fields"))
    for source_field in source_model.dofs.fields
        target_field = _field_layout(target_model.dofs, source_field.name)
        source_field.components == target_field.components ||
            throw(ArgumentError("source and target field $(source_field.name) have different component counts"))
    end
    return nothing
end

# Per-axis quadrature counts for a transfer region: take the per-axis
# maximum recommended order across the *union* of source and target
# parents, so the rule integrates source × target basis products exactly
# regardless of which side carries the higher-order basis.
#
# When `source_parents` is empty (no source coverage at this region),
# the source-side count drops to 1 — the rhs contribution there is zero
# anyway, so any rule will do, and the cheap `1` keeps cache pressure
# low.
function _transfer_quadrature_counts(source_model::Model{D}, target_model::Model{D}, target_parents,
                                     source_parents) where {D}
    target_counts = _parent_quadrature_counts(Val(D), target_parents,
                                              id -> _level_by_id(target_model.problem.space, id))
    source_counts = isempty(source_parents) ? ntuple(_ -> 1, D) :
                    _parent_quadrature_counts(Val(D), source_parents,
                                              id -> _level_by_id(source_model.problem.space, id))
    return ntuple(d -> max(target_counts[d], source_counts[d]), D)
end

function _transfer_quadrature(source_model::Model{D,T}, target_model::Model{D,T}, target_parents,
                              source_parents, cache) where {D,T}
    counts = _transfer_quadrature_counts(source_model, target_model, target_parents, source_parents)
    return _cached_tensor_quadrature!(cache, counts, T)
end

# Build the [`TransferRegion`](@ref) list. Reuses the volume
# admissible-box partition from `intersections.jl`: feed both sides'
# levels to `_merged_boxes` so the resulting boxes respect every mesh
# boundary on both sides. Drop boxes the target does not cover (no
# rhs contribution and no place to deposit the result).
function _transfer_regions(source_model::Model{D,T}, target_model::Model{D,T};
                           tolerance=target_model.dofs.tolerance) where {D,T}
    levels = (source_model.problem.space.levels..., target_model.problem.space.levels...)
    regions = TransferRegion{D,T}[]
    quadrature_cache = Dict{NTuple{D,Int},TensorQuadrature{D,T}}()

    for box in _merged_boxes(levels, Val(D), tolerance)
        target_parents = _parents_covering(target_model.problem.space.levels, box, tolerance)
        isempty(target_parents) && continue
        source_parents = _parents_covering(source_model.problem.space.levels, box, tolerance)
        quadrature = _transfer_quadrature(source_model, target_model, target_parents,
                                          source_parents, quadrature_cache)
        push!(regions, TransferRegion{D,T}(box, target_parents, source_parents, quadrature))
    end

    return regions
end

# ── L² projection source-driven rhs ──────────────────────────────────────────

# One side of a [`TransferRegion`](@ref) as the assembly kernel's helpers see a
# region: its parent list. The target and the source sides each get one, so
# `_frame!`, `_refresh!` and the state evaluation run on them unchanged, with
# `region_parents` and `_parent_lists` taking their single-sided defaults.
struct _TransferView{D,T}
    parents::Vector{ParentRef{D,T}}
end

# Accumulate the source-driven rhs of the L² transfer over the union partition
# `regions`, onto `rhs`. The target mass matrix `M_T` and its Dirichlet
# column-elimination lift `−M_ac·c_c` are assembled separately, by the standard
# assembler over the target's own integration regions (see
# [`L2Projection`](@ref)), so this pass integrates only the linear load
#
#     b_i += ∫_box u_S(x) · φ_iᵀ dx,
#
# where `u_S` is the source field reconstructed from its coefficients on the
# region's source parents and `φᵀ` are the target traces.
#
# It runs on two assembly workspaces, because the two models' level ids
# overlap and so cannot share a bank: the target one numbers the region's
# target dofs (`_frame!` and `_slots!`, one table for every basis family) and
# collects the local rhs, and the source one evaluates `u_S` as a `FormState`
# over the source coefficients. Both refresh values only, since the integrand
# never reads a gradient.
function _transfer_rhs!(rhs, source_coefficients, source_model::Model, target_model::Model, regions)
    target = _assembly_workspace(target_model)
    source = _assembly_workspace(source_model)
    state = FormState(source_model.dofs, _state_vector(source_coefficients, source_model))
    return _transfer_rhs_regions!(rhs, target, source, state, regions)
end

# The region loop of `_transfer_rhs!`, behind a function barrier on the two
# workspaces. Per region the source dof values are read once; per point and per
# target field component, `u_S` of the same-named source field is emitted
# against every target test function as a load, `(qweight·u_S)·φ`, accumulated
# per row in point order. A pivot target is condensed onto its branches before
# the local rhs is added to the global one. A region the source does not cover
# contributes nothing (`u_S ≡ 0` there) and is skipped.
function _transfer_rhs_regions!(rhs::Vector{T}, target::AssemblyWorkspace{D,T},
                                source::AssemblyWorkspace, state::FormState, regions) where {D,T}
    # Target field `f`'s component `c` reads the source field of the same name.
    source_field = [_field_index(source.layout, fl.name) for fl in target.layout.fields]
    for region in regions
        isempty(region.source_parents) && continue
        target_view = _TransferView(region.target_parents)
        source_view = _TransferView(region.source_parents)
        n = _slots!(_frame!(target, target_view))
        m = n + length(target.pivots)
        _state_region!(state, _frame!(source, source_view))
        fill!(resize!(target.b, m), zero(T))
        jacobian = volume(region.box) / convert(T, 2^D)
        for (η, weight) in zip(region.quadrature.points, region.quadrature.weights)
            qweight = weight * jacobian
            _refresh!(target, target_view, η, Val(false))
            _refresh!(source, source_view, η, Val(false))
            _state_point!(state, source, Val(false))
            for (f, fl) in pairs(target.layout.fields), c in 1:fl.components
                u = state.values[state.offsets[source_field[f]]+c]
                _emit!(target, qweight * u, f, c, one(T), 0, -one(T), n, m, false)
            end
        end
        isempty(target.pivots) || _condense!(target, n, m, false, false)
        for k in 1:n
            rhs[target.dofs[k]] += target.b[k]
        end
    end
    return nothing
end

# ── Public API and backends ──────────────────────────────────────────────────

"""
    abstract type TransferBackend end

Strategy object selecting how [`transfer`](@ref) moves a
[`Solution`](@ref) between two `Model`s. The default
[`L2Projection`](@ref) is general-purpose but introduces smoothing at
sharp features; [`Rewire`](@ref) is lossless when the target's active
basis contains the source's.
"""
abstract type TransferBackend end

"""
    L2Projection()
    L2Projection(matrix; factor=nothing)

Variational L² projection backend. Builds the target-side mass system

    M_T c_T = b_T,   M_T,ij = ∫_Ω φ_iᵀ φ_jᵀ dx,   b_T,i = ∫_Ω u_S(x) φ_iᵀ dx,

and solves it sparse-direct for the target coefficients. This produces the
L²-best target approximation of `u_S` in the target's active basis.

`M_T` and its Dirichlet column-elimination lift `−M_ac·c_c` come from the
standard assembler over the *target's own* integration regions: under exact
quadrature `∫_Ω φ_iᵀ φ_jᵀ` does not depend on the partition. Only the
source-driven rhs `b_T`, whose integrand mixes the two meshes, is
accumulated over a union partition that respects both.

The single-argument form `L2Projection(matrix)` reuses a precomputed
mass matrix — typically built once via
`assemble_matrix(target_model, mass_block(target_field))` and held
across many transfers. Passing `factor=lu(matrix)` (or any object
supporting `factor \\ rhs`) additionally skips the factorisation.

Both shortcuts are useful when the same target shows up repeatedly —
e.g. a moving-overlay transfer where only the source side changes
between time steps and the target's mass matrix can be cached across
them. Supplying a `factor` without a `matrix` is rejected: the matrix
is also needed to compute the solver residual reported in the returned
solution's diagnostics.

The cached-matrix shortcut requires the target to be homogeneously
constrained: a cached mass omits the Dirichlet column-elimination lift,
so a target with non-homogeneous physical Dirichlet data is rejected
(use the default `L2Projection()` there, which assembles the lift).

A target whose [`Space`](@ref) carries a [`PhysicalDomain`](@ref) is
rejected with an `ArgumentError` as well: the target mass restricts to `Ω`
through the cut-cell rules while the source-driven rhs integrates over the
full mesh boxes of the union partition, so the two halves of the system
would disagree about the domain. Making the union partition FCM-aware is
separate work.

This backend is single-domain only, and nothing checks it: the union
partition is built from `problem.space`, the representative (first field's)
space, while the rhs pass walks every field, so on a coupled model the
fields of the other subdomains are integrated against subdomain 1's cells
and the run fails inside the dof lookup rather than with a diagnostic. Use
[`Rewire`](@ref) there. Several fields sharing *one* space are fine.
"""
struct L2Projection{M,F} <: TransferBackend
    matrix::M
    factor::F
end
function L2Projection(matrix=nothing; factor=nothing)
    matrix === nothing &&
        factor !== nothing &&
        throw(ArgumentError("L2Projection: cannot supply a factor without its matrix"))
    return L2Projection{typeof(matrix),typeof(factor)}(matrix, factor)
end

"""
    Rewire(; strict=true)

Raw-key coefficient mapping backend. For each (raw, component) of every
source field:

  1. Look up the source's `TensorDofKey` in the target field's
     `raw_keys` table.
  2. If a target raw matches and is active, copy the source's active
     coefficient into the target's active slot.
  3. If the target raw is missing (the source key has no counterpart)
     or is constrained on the target, either raise (strict mode) or
     skip silently (`strict = false`).

Reconstructs the source field pointwise when the target's active basis
*contains* the source's. The coefficients are not modified, so the
reconstruction is exact, and the smoothing of the L² backend is avoided.

A moved overlay ([`moved`](@ref) / [`move!`](@ref)) does not satisfy
containment either, and in the way that is hardest to notice: the keys survive
the move unchanged but name functions at the new position, so every lookup
succeeds and the coefficients land on the wrong basis. This backend raises on a
level whose mesh has moved rather than translate the field.

Containment is a stronger condition than "the target space is larger",
and under leaf semantics a refinement does not satisfy
it: activating a finer level buries the parent's high-order modes, so
covered-mode pruning eliminates them in the target and the target's basis
trades parent modes for child modes rather than extending. The target's
*span* still contains the source's — that is what makes the elimination
lossless — but this backend matches dofs, not spans, and there is no
counterpart to copy the eliminated coefficients into. It raises in strict
mode; in non-strict mode it drops them, which silently discards their
whole contribution to the field. Use [`L2Projection`](@ref) for an
adaptive step in either direction, and reserve this backend for a target
built by adding dofs without eliminating any.

`strict = true` (default) is the safe choice: any key mismatch is a
programming error and should surface at the transfer call rather than
in mysterious downstream numbers. Pass `strict = false` only when the
dropped coefficients are genuinely negligible — it is not a fallback for
a target this backend cannot express, and it does not warn.
"""
struct Rewire <: TransferBackend
    strict::Bool
end
Rewire(; strict::Bool=true) = Rewire(strict)

"""
    transfer(source_solution, source_model, target_model;
             via=L2Projection(), tolerance=target_model.dofs.tolerance) -> Solution

Move a [`Solution`](@ref) from `source_model` onto `target_model` using the
strategy `via`. Returns a fresh `Solution` pinned to `target_model.version`.

`via` defaults to [`L2Projection`](@ref); pass `Rewire()` for the lossless
raw-key path or a precomputed `L2Projection(matrix; factor)` to reuse a cached
target mass. The same verb `transfer` moves a [`QuadField`](@ref) when given
one — see that method for the quadrature-point schemes.

`tolerance` controls the geometric merge tolerance used when building the
source-target union partition; it defaults to the target dof layout's stored
tolerance and is ignored by strategies that do not walk the geometry (currently
[`Rewire`](@ref)).

The two backends do not cover the same cases. [`L2Projection`](@ref) walks
the geometry of both models, rejects a target carrying a
[`PhysicalDomain`](@ref), and is single-domain only. [`Rewire`](@ref) reads
only raw dof keys, so it also serves coupled models, but it reproduces the
source exactly only when the target's active basis contains it. Either way
the two models must agree on the space bounding box, the field names, and
the per-field component counts.
"""
function transfer(source_solution::Solution, source_model::Model{D,T}, target_model::Model{D,T};
                  via::TransferBackend=L2Projection(),
                  tolerance=target_model.dofs.tolerance) where {D,T}
    return _transfer!(source_solution, source_model, target_model, via, tolerance)
end

# True when any field of `layout` carries a nonzero constrained value, i.e.
# non-homogeneous physical Dirichlet data. Artificial overlay constraints are
# always homogeneous (value 0), so this flags exactly the case where the
# Dirichlet column-elimination lift `−M_ac·c_c` is nonzero — the term the
# cached-matrix transfer path omits.
function _has_nonhomogeneous_constraints(layout::SystemLayout)
    return any(field -> any(!iszero, field.dofs.constrained_values), layout.fields)
end

# `L2Projection` implementation. The target-side mass matrix `M_T` and its
# Dirichlet column-elimination lift `−M_ac·c_c` are assembled by the
# STANDARD assembler over the target's own integration regions: under exact
# quadrature `∫_Ω φ_iᵀ φ_jᵀ` (target traces only) is independent of the
# partition, so the source/target union partition is needed only for the
# source-driven rhs `∫_Ω u_S(x) φ_iᵀ`, whose integrand mixes the two meshes.
# That source rhs is accumulated over the union partition, added onto the
# lift, and the system is solved sparse-direct.
#
# The cached-matrix backend (`M ≠ Nothing`) reuses `backend.matrix` and
# omits the lift, which vanishes only for a homogeneously-constrained
# target — so a non-homogeneous target is rejected there (the default
# `L2Projection()` re-assembles the mass and applies the lift correctly).
# The branch on `M === Nothing` tests a type parameter, so it folds at
# compile time; a supplied `backend.factor` short-circuits the
# factorisation.
function _transfer!(source_solution::Solution, source_model::Model{D,T}, target_model::Model{D,T},
                    backend::L2Projection{M,F}, tolerance) where {D,T,M,F}
    _assert_transfer_compatible(source_model, target_model)
    if M !== Nothing && _has_nonhomogeneous_constraints(target_model.dofs)
        throw(ArgumentError("L2Projection(matrix): the cached mass omits the Dirichlet lift, so " *
                            "it cannot transfer onto a target with non-homogeneous Dirichlet " *
                            "data. Use the default L2Projection() (no cached matrix) here."))
    end
    # The default path takes the target mass from the standard assembler, which
    # restricts to the immersed Ω via the FCM cut rules, while the source-driven
    # rhs below is assembled over the full mesh boxes of the union partition
    # (FCM-blind). For a target carrying a `physical_domain` those two are
    # inconsistent, so reject it rather than return a silently wrong projection.
    # Making the transfer union partition FCM-aware (both mass and rhs restricted
    # to Ω) is a separate follow-up.
    if target_model.problem.space.physical !== nothing
        throw(ArgumentError("L2 transfer onto a target with an immersed physical_domain is not yet " *
                            "supported: the target mass restricts to Ω but the source-driven rhs " *
                            "integrates over full mesh boxes, so the two are inconsistent."))
    end
    source_coefficients = _checked_coefficients(source_solution, source_model)

    nactive = active_unknowns(target_model.dofs)
    if nactive == 0
        return Solution(T[], target_model.version, SolverDiagnostics(:l2_projection, 0.0, true))
    end

    # Target mass + Dirichlet lift. Default path: assemble both from the
    # standard assembler over the target's own regions in one pass (the
    # bilinear pass shifts constrained trial columns onto the returned rhs,
    # which is the lift since there is no source load). Cached path: reuse
    # `backend.matrix` with a zero lift (homogeneous target, guarded above).
    if M === Nothing
        # The mass pattern joins the target's pattern cache, which keeps the
        # four most recently used ones, beside the problem's own pattern rather
        # than in place of it; on a symmetric volume-only problem the problem's
        # operator and the mass integrate over the same pass, and the two share
        # one pattern.
        mass, rhs = assemble(target_model, map(mass_block, target_model.problem.fields), ())
    else
        mass = backend.matrix
        rhs = zeros(T, nactive)
    end

    # Source-driven rhs over the source/target union partition, accumulated
    # onto the lift already in `rhs`.
    regions = _transfer_regions(source_model, target_model; tolerance=tolerance)
    _transfer_rhs!(rhs, source_coefficients, source_model, target_model, regions)

    coefficients = F === Nothing ? mass \ rhs : backend.factor \ rhs
    residual = norm(mass * coefficients - rhs)
    return Solution(coefficients, target_model.version,
                    SolverDiagnostics(:l2_projection, Float64(residual), true))
end

# `Rewire` matches source to target by `TensorDofKey` equality, which is only
# sound while the same key names the same function on both sides. That is a
# property of the basis family, and the two shipped families differ on it.
#
# An integrated-Legendre key indexes a mode by its DEGREE, and mode 3 is the same
# function at order 4 and at order 7, so a pure order elevation rewires exactly —
# measured, order 3 → 4 on 8² transfers with 0.000e+00 error. A B-spline key
# indexes a global 1D function over the whole axis (`_AXIS_BSPLINE`), and raising
# the degree rewrites the knot vector, so index *i* at degree 3 and index *i* at
# degree 4 are simply different functions. The keys still compare equal, the
# lookup still succeeds, and the coefficients land on the wrong basis: measured,
# the same order 3 → 4 transfer "succeeded" with no raise and a relative error of
# 1.80e-1. `strict = true` did not catch it, because it tests key PRESENCE while
# the docstring promises span containment.
#
# The test is family-generic rather than a B-spline special case: the degree lives
# in `BSplineFamily`'s type (its spaces are `BSplineSpace{p}`), while
# `IntegratedLegendre` is one type for every order. So comparing the instantiated
# basis TYPE admits exactly the rewires that are sound and rejects the rest, and a
# future family inherits the right behaviour by construction. The mesh is
# compared for the same reason: a key indexes into a mesh, and two meshes of
# different size *or position* do not share an indexing. Position is the half a
# cell-count test alone misses — `moved` and `move!` keep every key and move the
# function it names, so a displaced overlay matches key for key and lands the
# source's coefficients on functions somewhere else. Measured with the guard
# stubbed out, on a 2D Poisson base 8² p=3 carrying a 4×4 overlay displaced by
# half a fine cell: `strict = true` raised nothing and the transferred field
# deviated by up to 4.3e-2 from the source, against a solution whose own
# magnitude is 7.4e-2 — a 58% error, silently.
#
# `domain` and `cells` settle it between them: a `CartesianMesh`'s `axes` are
# derived from exactly those two by `_mesh_axes` (mesh.jl), and `AxisBox`
# equality is corner-wise and bit-exact for meshes built that way.
function _assert_keys_comparable(source_model::Model, target_model::Model)
    src, tgt = source_model.problem.space, target_model.problem.space
    for k in 1:min(length(src.levels), length(tgt.levels))
        a, b = src.levels[k], tgt.levels[k]
        typeof(a.basis) === typeof(b.basis) ||
            throw(ArgumentError("Rewire: level $k carries $(basis_name(a.basis)) at order " *
                                "$(nominal_order(a)) on the source and $(basis_name(b.basis)) at order " *
                                "$(nominal_order(b)) on the target, and those name different function sets, so a " *
                                "dof key does not name the same function on both sides — raising a B-spline degree " *
                                "rewrites the knot vector and does exactly this. Use `L2Projection()` instead."))
        (a.mesh.cells == b.mesh.cells && a.mesh.domain == b.mesh.domain) ||
            throw(ArgumentError("Rewire: level $k is meshed as $(a.mesh.cells) cells over $(a.mesh.domain) on the " *
                                "source and $(b.mesh.cells) cells over $(b.mesh.domain) on the target; a dof key " *
                                "indexes into a mesh, so the same key names a function of a different size or at a " *
                                "different position on the two sides and they are not comparable. Use " *
                                "`L2Projection()` instead."))
    end
    return nothing
end

# `Rewire` implementation: walk every active source dof, find its
# target counterpart by `TensorDofKey`, and copy the coefficient.
# Missing target counterparts and freshly-constrained target dofs are
# treated as errors under `strict = true` and silently skipped
# otherwise.
function _transfer!(source_solution::Solution, source_model::Model{D,T}, target_model::Model{D,T},
                    backend::Rewire, _tolerance) where {D,T}
    # Rewire walks raw `TensorDofKey`s, never the geometry — the
    # geometric merge tolerance accepted by `transfer` has no role
    # here and is ignored on purpose.
    _assert_transfer_compatible(source_model, target_model)
    _assert_keys_comparable(source_model, target_model)
    source_coefficients = _checked_coefficients(source_solution, source_model)

    source_layout = source_model.dofs
    target_layout = target_model.dofs
    nactive = active_unknowns(target_layout)
    coefficients = zeros(T, nactive)

    for source_field in source_layout.fields
        target_field_idx = get(target_layout.by_name, source_field.name, 0)
        target_field_idx == 0 && continue
        target_field = target_layout.fields[target_field_idx]

        # Build a target-side key → raw lookup once per field. The
        # mapping is small (one entry per target raw key) and is reused
        # across every source raw key of the field.
        target_raw_by_key = Dict(key => raw for (raw, key) in pairs(target_field.dofs.raw_keys))

        for source_raw in eachindex(source_field.dofs.raw_keys)
            source_key = source_field.dofs.raw_keys[source_raw]
            target_raw = get(target_raw_by_key, source_key, 0)
            for c in 1:source_field.components
                source_active = source_field.dofs.active_component[source_raw, c]
                source_active == 0 && continue
                if target_raw == 0
                    backend.strict &&
                        throw(ArgumentError("Rewire: source active dof at key $(source_key) component $c has no target counterpart; pass `Rewire(strict=false)` to skip"))
                    continue
                end
                target_active = target_field.dofs.active_component[target_raw, c]
                if target_active == 0
                    backend.strict &&
                        throw(ArgumentError("Rewire: source active dof at key $(source_key) component $c is constrained in the target"))
                    continue
                end
                coefficients[target_field.offset + target_active] = source_coefficients[source_field.offset + source_active]
            end
        end
    end

    return Solution(coefficients, target_model.version, SolverDiagnostics(:rewire, 0.0, true))
end

# Friendlier error-message dispatches for the most common user
# mistakes: mismatched dimension / scalar type (the model parameters
# disagree, so the typed `transfer` above does not match) and missing
# target model (positional shorthand the API does not support).
function transfer(::Solution, ::Model, ::Model; kwargs...)
    throw(ArgumentError("source and target models must have the same dimension and scalar type for transfer"))
end

function transfer(::Solution, ::Model; kwargs...)
    throw(ArgumentError("transfer requires source and target models; call transfer(solution, source_model, target_model)"))
end
