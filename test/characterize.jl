# ── Characterization harness — a REVIEW INSTRUMENT, not a test ────────────────
#
# This script is deliberately NOT included from `test/runtests.jl`. It asserts
# nothing. It prints a deterministic, diffable report of golden values over a
# matrix of configurations, so that a batch of correctness fixes can be judged
# by what it moves and what it leaves alone:
#
#     julia --project=. test/characterize.jl > before.txt
#     …apply one fix…
#     julia --project=. test/characterize.jl > after.txt
#     diff before.txt after.txt
#
# WHAT THIS INSTRUMENT DOES NOT COVER. Every `order =` argument below is a
# scalar, so no case here builds a level with a PER-CELL polynomial order. The
# report was measured byte-identical against five deliberately broken minimum
# rules — the rule deleted outright, a per-axis test in place of set membership,
# inactive cells minimised over, a maximum rule, and a truncated incidence walk —
# so a matching hash says nothing at all about the graded kernel. That kernel has
# its own instrument: `test/test_graded_golden.jl`, which locks the per-cell mode
# sets directly and moves under all five of those controls.
#
# A DELIBERATE MOVE, so a diff against an older `before.txt` is not misread.
# The B-spline family's default changed from `continuity_order = 0` to
# `continuity = :maximal`, and with it the mechanism: a level now *selects* the
# functions whose whole support lies inside its active region instead of clamping
# and constraining its artificial faces. Every `bspline` case below therefore
# moves — cases 3, 8, 9, 23 and 30 — and the move is the feature, not collateral
# damage. Their `MUST-NOT-CHANGE` marking is about correctness fixes and still
# means what it says for any *later* change; re-baseline against a run made on
# this commit or newer.
#
# Every case carries one of two markings:
#
#   MUST-NOT-CHANGE     the configuration is *currently correct*. A fix that
#                       moves any number here has collateral damage, even when
#                       the targeted case starts working. Bit-identical or bust.
#
#   EXPECTED-TO-CHANGE  the configuration is *currently wrong* in the way the
#                       named finding documents. The values below record the
#                       wrong behaviour so the fix can be seen to land. Each
#                       such case names its finding and, in `wrong_because`,
#                       what specifically is wrong about the numbers printed.
#
# Determinism rules obeyed here, because a diff is the whole product:
#   * fixed case order, fixed key order, no timings, no addresses, no paths;
#   * `Dict` / `IdDict` iteration order is never observable — the two
#     identity-keyed region caches are read through the objects that key them,
#     and every derived multiset is sorted before printing;
#   * floating-point values print through `repr`, i.e. shortest-round-trip full
#     precision, so a change in the last bit shows up as a diff.
#
# Run it twice; the output must be byte-identical. Compare only runs made with
# the same Julia version, the same command-line flags and the same thread count:
# the report is exact to the last bit, and optimisation level or thread count can
# legitimately reorder a floating-point reduction. The header echoes version and
# thread count so a mismatched pair is visible in the diff itself.
#
# Cost is one-time JIT compilation of the configuration matrix, not arithmetic:
# the numerical work is well under a second, the process is ~75 s at the default
# `-O2` and ~55 s at `-O1` (verified byte-identical to `-O2` here). Each level
# is deterministic within itself, but `-O0` does NOT agree with `-O2` — it
# reorders reductions — so a report is only comparable against one produced at
# the same level. The reference baseline is generated at `-O0`, which is also the
# level `CONTRIBUTING.md` documents for the test suite; keep it there. Nothing is
# checked in — the report is a hand-run before/after pair, as the recipe above
# says, so a baseline is only ever as good as the commit it was taken on. Take a
# fresh one whenever a case is added, because an older baseline has no lines to
# diff the new case against.
#
# Findings this instrument is aimed at:
#   #1  facet/surface region resolution is not subdomain-aware
#   #2  `_NNLS_WEIGHT_TOL` is an absolute cutoff on a cut-volume-scaled weight
#   #3  `tolerance.merge` is an absolute length compared against mesh spacing
#   #4  the fictitious fold is applied to the already-folded mask
#   #5  `moved` has no single-domain guard
#   #8  `physical_domain(tree; lipschitz=L)` discards L
#   #9  `_subdivide_segment` discards 2·nudge of arc length per grid crossing
#   #T4 grid-aligned facet integration is not level-set-aware (FIXED; cases 32-33
#       now pin the fix rather than the defect)
#
# WHY #T4 NEEDED TWO NEW CASES, and what the 31 before them do NOT say about it.
# Grid-aligned facet integration used to take no `PhysicalDomain`: a `FacetRegion`
# is the whole face of its parent cells, and the rule on it covered all of that
# face, so where `∂Ω` crossed it the rule spent weight on area Ω does not
# contain. Cases 32 and 33 are the only cases that can see it, and the gap they
# closed was not an oversight but a property of the
# matrix: the two ingredients #T4 needs never co-occurred. The immersed cases
# (12, 13, 22, 28) carry no boundary selector at all, so they resolve no facet
# region whatsoever; the facet cases (14, 15, 16, 21) are not immersed, so no face
# of theirs can leave Ω. Nor are cases 17, 26 and 27, the ones that look immersed:
# their curved geometry is a user-supplied `BoundaryMesh` or interface, which is
# cut against the grid already and is out of #T4's scope on principle — the
# package must decide which part of a face it inherited from the mesh lies in Ω,
# but must not second-guess geometry the user stated. Every
# `cut_facet_region_count` those cases print is therefore 0,
# and the landed fix left all 31 of them byte-identical, sha256 of the first 2029
# lines 23a0cc2410cd9cc930f29332d7bd905f257c129ec3c5830a0b0a3243c627a811. Read
# their stability under a facet-integration change as an artifact of the matrix,
# not as evidence about the change.

# ── Environment ───────────────────────────────────────────────────────────────
#
# The B-spline cases need the BasicBSpline extension, which the package's own
# `--project=.` environment cannot load (it is a weak dependency). If the active
# project cannot see BasicBSpline, fall back to a dedicated, persistent
# environment in the depot that devs this repository and adds it. Building that
# environment happens at most once, and every Pkg message is muted so it can
# never reach the report.

import Pkg

if Base.identify_package("BasicBSpline") === nothing
    let env = joinpath(first(DEPOT_PATH), "environments", "unfitted-characterize"),
        repo = normpath(joinpath(@__DIR__, ".."))

        Pkg.activate(env; io=devnull)
        # The environment is persistent and shared across checkouts, so a stale
        # one may still `dev` a DIFFERENT tree than the one being measured — the
        # main checkout, say, while this script runs from a git worktree. That
        # failure is silent and total: the report then describes source the run
        # never touched, and a fix under test shows up as byte-identical. Re-`dev`
        # whenever the recorded path is not this repository.
        needs_dev = !isfile(joinpath(env, "Project.toml"))
        if !needs_dev
            manifest = joinpath(env, "Manifest.toml")
            entry = isfile(manifest) ? get(Pkg.TOML.parsefile(manifest), "deps", nothing) : nothing
            recorded = entry === nothing ? nothing :
                       get(first(get(entry, "Unfitted", [Dict()])), "path", nothing)
            needs_dev = recorded === nothing || normpath(recorded) != normpath(repo)
        end
        if needs_dev
            Pkg.develop(; path=repo, io=devnull)
            Pkg.add("BasicBSpline"; io=devnull)
        end
    end
end

# Precompile with output muted. Otherwise the first run after any `src/` edit
# interleaves Julia's precompilation chatter into the report, which reads as a
# spurious diff in whichever case happened to be printing at the time.
Pkg.precompile(; io=devnull)

using Unfitted
using BasicBSpline
using LinearAlgebra
using SparseArrays

using Unfitted: SVector, SMatrix, active_unknowns, raw_dof_count, volume, problem_spaces,
                moment_fit_rule, _diffusion_flux

# ── Formatting ────────────────────────────────────────────────────────────────
#
# One key per line, fixed column, value through `fmtval`. Floats print at full
# precision on purpose: this instrument exists to see the last bit move.

const KEYPAD = 34

fmtval(x::Float64) = repr(x)
fmtval(x::Float32) = repr(x)
fmtval(x::Bool) = x ? "true" : "false"
fmtval(x::Integer) = string(x)
fmtval(x::Symbol) = ":" * String(x)
fmtval(x::AbstractString) = String(x)
fmtval(::Nothing) = "n/a"
fmtval(x::CartesianIndex) = "CI" * fmtval(Tuple(x))
fmtval(x::Tuple) = "(" * join(map(fmtval, x), ", ") * ")"
fmtval(x::AbstractVector) = "[" * join(map(fmtval, x), ", ") * "]"

emit(key::AbstractString, value) = println("  ", rpad(key, KEYPAD), " = ", fmtval(value))

# Same column, no indent: the environment lines above the first case. They are
# not case values, and a diff in them explains a diff in everything below.
emit_env(key::AbstractString, value) = println(rpad(key, KEYPAD + 2), " = ", value)

# Compact, order-preserving rendering of a boolean cell mask: column-major
# scan, one character per cell. Deterministic and short enough to diff by eye.
maskstr(a::AbstractArray{Bool}) = String([b ? '1' : '0' for b in vec(a)])

# Point label for a sample-value key, e.g. "(0.5, 0.5)". Coordinates are picked
# to have exact short decimal forms, so the key text stays stable.
pointlabel(p) = "(" * join((repr(Float64(c)) for c in p), ", ") * ")"

const CASE_MARKS = String[]

function case(id::Integer, label::AbstractString, mark::AbstractString; note=nothing,
              wrong_because=nothing)
    push!(CASE_MARKS, mark)
    println()
    println("CASE ", lpad(id, 2, '0'), "  ", rpad(label, 38), "  ", mark)
    note === nothing || println("  note: ", note)
    wrong_because === nothing || println("  wrong_because: ", wrong_because)
    return nothing
end

# ── Metric kernels ────────────────────────────────────────────────────────────

# Checksum of a sparse matrix that is sensitive to value *and* position: the
# index-weighted sum moves if any stored value changes or if the same values
# are stored in a different order, the row/colptr checksums move if the sparsity
# pattern shifts without changing the values, and the extremal entries pin the
# two ends of the value range together with where they sit.
function emit_matrix(prefix::AbstractString, A)
    if A === nothing
        emit(prefix * ".state", "unassembled")
        return nothing
    end
    nz = nonzeros(A)
    rows = rowvals(A)
    n = length(nz)
    emit(prefix * ".size", (size(A, 1), size(A, 2)))
    emit(prefix * ".nnz", n)
    if n == 0
        emit(prefix * ".empty", true)
        return nothing
    end
    weighted = 0.0
    total = 0.0
    absolute = 0.0
    rowsum = 0
    @inbounds for i in 1:n
        weighted += nz[i] * i
        total += nz[i]
        absolute += abs(nz[i])
        rowsum += rows[i] * i
    end
    colsum = 0
    @inbounds for j in eachindex(A.colptr)
        colsum += A.colptr[j] * j
    end
    imin = argmin(nz)
    imax = argmax(nz)
    emit(prefix * ".index_weighted", weighted)
    emit(prefix * ".sum", total)
    emit(prefix * ".abs_sum", absolute)
    emit(prefix * ".min", nz[imin])
    emit(prefix * ".min_at", imin)
    emit(prefix * ".max", nz[imax])
    emit(prefix * ".max_at", imax)
    emit(prefix * ".row_checksum", rowsum)
    emit(prefix * ".colptr_checksum", colsum)
    return nothing
end

# Same discipline for a dense vector: position-sensitive checksum, magnitude,
# and the two extremes with their indices.
function emit_vector(prefix::AbstractString, b)
    if b === nothing
        emit(prefix * ".state", "unassembled")
        return nothing
    end
    n = length(b)
    emit(prefix * ".length", n)
    if n == 0
        emit(prefix * ".empty", true)
        return nothing
    end
    weighted = 0.0
    total = 0.0
    absolute = 0.0
    @inbounds for i in 1:n
        weighted += b[i] * i
        total += b[i]
        absolute += abs(b[i])
    end
    imin = argmin(b)
    imax = argmax(b)
    emit(prefix * ".index_weighted", weighted)
    emit(prefix * ".sum", total)
    emit(prefix * ".abs_sum", absolute)
    emit(prefix * ".min", b[imin])
    emit(prefix * ".min_at", imin)
    emit(prefix * ".max", b[imax])
    emit(prefix * ".max_at", imax)
    return nothing
end

# Sorted multiset of region-kind tags across every subdomain plan, e.g.
# "cut_fitted=12 full=4". Sorted so the string never depends on Dict order.
function region_kinds(model)
    counts = Dict{Symbol,Int}()
    for plan in model.space_plans, region in plan.regions
        counts[region.quadrature.kind] = get(counts, region.quadrature.kind, 0) + 1
    end
    return join((string(k, "=", counts[k]) for k in sort!(collect(keys(counts)))), " ")
end

# Sorted multiset of per-region cover counts (how many levels cover a region),
# e.g. "1=12 2=4". A change here means the multi-level intersection moved.
function cover_histogram(model)
    counts = Dict{Int,Int}()
    for plan in model.space_plans, region in plan.regions
        c = length(region.parents)
        counts[c] = get(counts, c, 0) + 1
    end
    return join((string(k, "=", counts[k]) for k in sort!(collect(keys(counts)))), " ")
end

# ∫_Ω 1 dx as the model's own quadrature sees it: every region's reference-frame
# weights times its box jacobian, exactly the `l2_error` convention. This is the
# single most sensitive scalar in the report — it moves if a cut rule empties
# (#2), if the grid is decimated (#3), or if a region box changes at all.
function domain_measure(model)
    D = diagnostics(model).dimension
    jacobian_scale = 1 / 2.0^D
    total = 0.0
    for plan in model.space_plans, region in plan.regions
        w = 0.0
        for weight in region.quadrature.weights
            w += weight
        end
        total += volume(region.box) * jacobian_scale * w
    end
    return total
end

# Number of regions carrying no quadrature point at all. A nonzero count is the
# silent-zero-stiffness condition described in the `PhysicalDomain` docstring.
function empty_region_count(model)
    n = 0
    for plan in model.space_plans, region in plan.regions
        isempty(region.quadrature.weights) && (n += 1)
    end
    return n
end

# The whole diagnostics record, in a fixed order, plus the plan-derived
# quantities the record does not carry.
function emit_model(model)
    d = diagnostics(model)
    emit("dimension", d.dimension)
    emit("active_unknowns", active_unknowns(model))
    emit("raw_dofs", raw_dof_count(model.dofs))
    emit("subdomain_spaces", length(problem_spaces(model.problem)))
    emit("integration_regions", d.integration_regions)
    emit("region_kinds", region_kinds(model))
    emit("region_cover_counts", cover_histogram(model))
    emit("empty_regions", empty_region_count(model))
    emit("domain_measure", domain_measure(model))
    emit("min_integration_volume", d.min_integration_volume)
    emit("min_relative_integration_volume", d.min_relative_integration_volume)
    emit("small_overlap_count", d.small_overlap_count)
    emit("inactive_cell_counts", d.inactive_cell_counts)
    emit("reduced_mode_counts", d.reduced_mode_counts)
    emit("cut_region_count", d.cut_region_count)
    emit("fit_failure_count", d.fit_failure_count)
    emit("cut_fallback_count", d.cut_fallback_count)
    emit("cut_fallback_points", d.cut_fallback_points)
    emit("moment_fit_residual_max", d.moment_fit_residual_max)
    emit("facet_region_count", d.facet_region_count)
    emit("cut_facet_region_count", d.cut_facet_region_count)
    emit("surface_region_count", d.surface_region_count)
    emit("interface_region_count", d.interface_region_count)
    emit("nquadpoints.volume", nquadpoints(model; kind=:volume))
    emit("nquadpoints.facet", nquadpoints(model; kind=:facet))
    emit("nquadpoints.surface", nquadpoints(model; kind=:surface))
    emit("nquadpoints.interface", nquadpoints(model; kind=:interface))
    emit("symmetry_residual", d.symmetry_residual)
    emit("condition_estimate", d.condition_estimate)
    emit("solver", d.solver)
    emit_matrix("matrix", model.matrix)
    emit_vector("rhs", model.rhs)
    return nothing
end

# `prefix` namespaces every key, so a case reporting more than one solved model
# keeps its key set unambiguous. The default is empty and prints exactly the
# unprefixed keys.
function emit_solution(model, solution; exact=nothing, points=(), fields=(), prefix="")
    emit(prefix * "solver.method", solution.diagnostics.method)
    emit(prefix * "solver.residual_norm", solution.diagnostics.residual_norm)
    emit(prefix * "solver.converged", solution.diagnostics.converged)
    emit_vector(prefix * "coefficients", solution.coefficients)
    if exact !== nothing
        emit(prefix * "l2_error.absolute", l2_error(solution, model, exact; norm=:absolute))
        emit(prefix * "l2_error.relative", l2_error(solution, model, exact))
    end
    for p in points
        emit(prefix * "u" * pointlabel(p), value(solution, model, p))
        exact === nothing || emit(prefix * "u_exact" * pointlabel(p), Float64(exact(p)))
    end
    for (u, ps) in fields
        for p in ps
            emit(prefix * String(u.name) * pointlabel(p), value(solution, model, u, p))
        end
    end
    return nothing
end

# The L² Dirichlet trace projection's own record, for the cases whose subject is
# the facet rule that projection is built on: how many of the model's constrained
# values came out nonzero, which branch the per-component trace solve took, and
# which dofs it had no measure to fit. The last two come off the diagnostics
# record, where the package reports them itself — this used to re-apply the
# condition list through `update_dirichlet!` purely to get at the discarded
# `DirichletProjection` and read its factor types, which also dropped the cached
# operators and so had to run after `emit_model`. Neither constraint survives.
#
# `only` on the field layouts keeps the first key inside the determinism rules
# rather than beside them: a single-field model carries one entry, so no iteration
# order is observable, and `only` throws rather than printing an arbitrary one if
# a later case is multi-field. The diagnostics quantities need no such care —
# both are ordered by the layout's own field and component order.
function emit_dirichlet_projection(model)
    d = diagnostics(model)
    emit("dirichlet.nonzero_value_count",
         count(!iszero, only(model.dofs.fields).dofs.constrained_values))
    emit("dirichlet.trace_factors", d.dirichlet_trace_factors)
    emit("dirichlet.unsupported_dof_count", d.unsupported_dirichlet_dof_count)
    emit("dirichlet.unsupported_dofs",
         [(r.field, r.component, r.raw) for r in d.unsupported_dirichlet_dofs])
    return nothing
end

# The facet rule's own statistics, for the two cases whose subject is that rule.
# Emitted here rather than from `emit_model` on purpose: every case would then
# grow four lines, and the 31 cases that came before #T4 have to stay byte-
# identical for the diff to mean anything. The first three are the facet
# analogues of the cut-cell fit statistics and say how the moment fit on each
# cut face's affine slice went; the fourth is the smallest integrated-to-
# geometric measure ratio over the regions, which is the conditioning warning a
# trimmed Dirichlet condition needs and reads exactly 0 where a face lies wholly
# outside Ω at α = 0.
function emit_facet_rule(model)
    d = diagnostics(model)
    emit("facet_fit_failure_count", d.facet_fit_failure_count)
    emit("facet_cut_fallback_count", d.facet_cut_fallback_count)
    emit("facet_moment_fit_residual_max", d.facet_moment_fit_residual_max)
    emit("min_relative_facet_measure", d.min_relative_facet_measure)
    return nothing
end

# Run `f`, reporting either its value or the type of the exception it raised.
# Used wherever a fix is expected to convert silent-wrong into a loud error, so
# the report keeps its shape across that transition.
attempt(label::AbstractString, f) =
    try
        v = f()
        emit(label, "ok")
        return v
    catch e
        emit(label, "threw:" * String(nameof(typeof(e))))
        return nothing
    end

# ── Shared fixtures ───────────────────────────────────────────────────────────

# Manufactured solutions that the shipped orders represent exactly, so the
# printed `l2_error` sits at round-off and any real change is unmissable.
u1d(x) = x[1] * (1 - x[1])
f1d(x) = 2.0
u2d(x) = x[1] * (1 - x[1]) * x[2] * (1 - x[2])
f2d(x) = 2 * (x[1] - x[1]^2 + x[2] - x[2]^2)
u3d(x) = x[1] * (1 - x[1]) * x[2] * (1 - x[2]) * x[3] * (1 - x[3])
function f3d(x)
    a = x[1] - x[1]^2
    b = x[2] - x[2]^2
    c = x[3] - x[3]^2
    return 2 * (b * c + a * c + a * b)
end

const P1 = ((0.25,), (0.5,), (0.7,))
const P2 = ((0.25, 0.25), (0.5, 0.5), (0.7, 0.3))
const P3 = ((0.25, 0.25, 0.25), (0.5, 0.5, 0.5), (0.7, 0.3, 0.6))

zero_dirichlet() = [dirichlet(0.0; on=boundary(:all))]

# One prepared + solved Poisson model on a space, reported in full.
function poisson_case(V, source, exact, points)
    model = prepare(poisson(V; source, dirichlet=zero_dirichlet()))
    solution = solve!(model)
    emit_model(model)
    emit_solution(model, solution; exact, points)
    return model, solution
end

# One prepared + solved model whose only block is `stiffness_block(u; diffusion)`,
# driven by `source` under zero Dirichlet data on ∂Ω. Split out so the several
# spellings of `diffusion` reach the assembler through one identical path and a
# difference between them can only come from `_diffusion_flux`.
function diffusion_case(V, diffusion, source)
    u = field(:u, V)
    model = prepare(Problem((u,); blocks=(stiffness_block(u; diffusion),),
                            loads=(source_load(u; source),), dirichlet=zero_dirichlet()))
    return model, solve!(model)
end

# ── Report ────────────────────────────────────────────────────────────────────

function main()
    println("# Unfitted.jl characterization report")
    println("# review instrument for batch 1; deterministic, diffable, assertion-free")
    println("# markings: MUST-NOT-CHANGE = currently correct, must stay bit-identical")
    println("#           EXPECTED-TO-CHANGE = currently wrong, records the documented bug")
    emit_env("env.julia_version", VERSION)
    emit_env("env.nthreads", Threads.nthreads())

    # ── 01 ── D = 1, integrated Legendre, base level only.
    case(1, "d1-legendre-p3", "MUST-NOT-CHANGE")
    poisson_case(space(box((0.0,), (1.0,)); cells=8, order=3), f1d, u1d, P1)

    # ── 02 ── D = 1 with one overlay: the multi-level superposition path.
    case(2, "d1-legendre-overlay", "MUST-NOT-CHANGE")
    let V = space(box((0.0,), (1.0,)); cells=8, order=3)
        V = overlay(V, box((0.25,), (0.75,)); cells=4, order=4)
        poisson_case(V, f1d, u1d, P1)
    end

    # ── 03 ── D = 1, the second shipped basis family.
    case(3, "d1-bspline-p3", "MUST-NOT-CHANGE")
    poisson_case(space(box((0.0,), (1.0,)); cells=8, order=3, basis=bspline()), f1d, u1d, P1)

    # ── 04 ── D = 2, base level only.
    case(4, "d2-legendre-p2", "MUST-NOT-CHANGE")
    poisson_case(space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2), f2d, u2d, P2)

    # ── 05 ── D = 2 with one overlay at higher order.
    case(5, "d2-legendre-overlay", "MUST-NOT-CHANGE")
    let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2)
        V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(2, 2), order=3)
        poisson_case(V, f2d, u2d, P2)
    end

    # ── 06 ── D = 2, overlay carrying a per-cell mask from construction.
    case(6, "d2-overlay-masked", "MUST-NOT-CHANGE")
    let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2)
        V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(2, 2), order=3,
                    active=[CartesianIndex(1, 1), CartesianIndex(2, 1), CartesianIndex(1, 2)])
        model, _ = poisson_case(V, f2d, u2d, P2)
        emit("active_cells.level2", maskstr(active_cells(model; level=2)))
    end

    # ── 07 ── D = 2, mask mutated in place after `prepare` (no physical domain,
    #          so the fictitious fold is the identity and #4 cannot reach it).
    case(7, "d2-mask-mutated", "MUST-NOT-CHANGE")
    let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2)
        V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(2, 2), order=3)
        model = prepare(poisson(V; source=f2d, dirichlet=zero_dirichlet()))
        deactivate!(model; level=2, cells=[CartesianIndex(2, 2)])
        emit("after_deactivate.active", active_unknowns(model))
        emit("after_deactivate.mask", maskstr(active_cells(model; level=2)))
        activate!(model; level=2, cells=[CartesianIndex(2, 2)])
        emit("after_activate.active", active_unknowns(model))
        emit("after_activate.mask", maskstr(active_cells(model; level=2)))
        deactivate!(model; level=2, cells=[CartesianIndex(1, 2)])
        solution = solve!(model)
        emit_model(model)
        emit_solution(model, solution; exact=u2d, points=P2)
    end

    # ── 08 ── D = 2, B-splines.
    case(8, "d2-bspline-p3", "MUST-NOT-CHANGE")
    poisson_case(space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=3, basis=bspline()), f2d,
                 u2d, P2)

    # ── 09 ── D = 2, B-splines with an overlay: the family's constraint path.
    case(9, "d2-bspline-overlay", "MUST-NOT-CHANGE")
    let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=3, basis=bspline())
        V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(2, 2), order=3, basis=bspline())
        poisson_case(V, f2d, u2d, P2)
    end

    # ── 10 ── D = 3, base level only.
    case(10, "d3-legendre-p2", "MUST-NOT-CHANGE")
    poisson_case(space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(2, 2, 2), order=2), f3d, u3d,
                 P3)

    # ── 11 ── D = 3 with an overlay.
    case(11, "d3-legendre-overlay", "MUST-NOT-CHANGE")
    let V = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(2, 2, 2), order=2)
        V = overlay(V, box((0.25, 0.25, 0.25), (0.75, 0.75, 0.75)); cells=(2, 2, 2), order=2)
        poisson_case(V, f3d, u3d, P3)
    end

    # ── 12 ── FCM: an immersed disc, strict cut path (alpha = 0). Well posed
    #          without boundary data because the form carries a mass block.
    case(12, "d2-fcm-disc-alpha0", "MUST-NOT-CHANGE",
         note="moment_fit_residual_max is expected to move under fix #2, which recomputes " *
              "the residual after weight truncation; the weights themselves must not.")
    let phi = x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.3,
        p = physical_domain(phi; lipschitz=1.0, subcell_length_scale=1 / 32, max_depth=4)

        V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=p)
        u = field(:u, V)
        model = prepare(Problem((u,); blocks=(stiffness_block(u), mass_block(u)),
                                loads=(source_load(u; source=1.0),)))
        solution = solve!(model)
        emit_model(model)
        emit("exact_area", Float64(pi * 0.3^2))
        emit_solution(model, solution; points=((0.5, 0.5), (0.4, 0.6)))
    end

    # ── 13 ── FCM with α-blended fictitious quadrature.
    case(13, "d2-fcm-disc-alpha", "MUST-NOT-CHANGE",
         note="see case 12 on moment_fit_residual_max under fix #2.")
    let phi = x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.3,
        p = physical_domain(phi; lipschitz=1.0, alpha=0.05, subcell_length_scale=1 / 32,
                            max_depth=4)

        V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=p)
        u = field(:u, V)
        model = prepare(Problem((u,); blocks=(stiffness_block(u), mass_block(u)),
                                loads=(source_load(u; source=1.0),)))
        solution = solve!(model)
        emit_model(model)
        emit_solution(model, solution; points=((0.5, 0.5), (0.4, 0.6)))
    end

    # ── 14 ── Neumann data on a boundary selector, single domain. This is the
    #          healthy half of #1: one subdomain, so selector resolution has
    #          nothing to get wrong, and the fix must leave it alone.
    case(14, "d2-neumann-facet", "MUST-NOT-CHANGE")
    let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2),
        u = field(:u, V),
        exact = x -> x[1]

        model = prepare(Problem((u,); blocks=(stiffness_block(u),),
                                loads=(neumann(u, 1.0; on=boundary(axis=1, side=:upper)),),
                                dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower))]))
        solution = solve!(model)
        emit_model(model)
        emit("facet_measure.x_upper",
             boundary_integral(q -> 1.0, model; on=boundary(axis=1, side=:upper)))
        emit("facet_flux.x_upper",
             boundary_integral(q -> q.normal[1], model; on=boundary(axis=1, side=:upper)))
        emit_solution(model, solution; exact, points=P2)
    end

    # ── 15 ── Multi-domain product space, no interface: the block-diagonal
    #          invariant. Insensitive to #9 (no surface geometry at all).
    case(15, "d2-coupled-product", "MUST-NOT-CHANGE",
         note="facet_region_count and nquadpoints.facet are structural probes of the cache " *
              "#1 re-keyed: before that fix the two value-equal `boundary(:all)` conditions " *
              "both resolved to subdomain 1's 3x3 geometry, so the counts read 2x12 = 24 and " *
              "2x36 = 72 rather than 12+8 and 36+32. They now read 20 and 68 — V1 is order 2 " *
              "(3 points per facet) but V2 is order 3 (4 points), so 12x3 + 8x4 = 68. Nothing " *
              "else in this case may move.")
    let V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=2),
        V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(2, 2), order=3),
        u1 = field(:u1, V1),
        u2 = field(:u2, V2)

        model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2)),
                                loads=(source_load(u1; source=1.0), source_load(u2; source=2.0)),
                                dirichlet=[dirichlet(0.0; on=boundary(:all), field=:u1),
                                           dirichlet(0.0; on=boundary(:all), field=:u2)]))
        solution = solve!(model)
        emit_model(model)
        emit_solution(model, solution;
                      fields=((u1, ((0.5, 0.5), (0.25, 0.75))), (u2, ((2.5, 0.5), (2.25, 0.75)))))
    end

    # ── 16 ── Nonzero Dirichlet data on two value-equal `boundary(:all)`
    #          selectors, one per subdomain. Unlike the Neumann shape of case
    #          21 this one is *correct* today — the Dirichlet projection builds
    #          its own facet regions rather than reading the shared cache — so
    #          it is the case that would notice if #1's re-keying reached the
    #          projection as well.
    case(16, "d2-coupled-dirichlet-lift", "MUST-NOT-CHANGE",
         note="facet_cache_keys (2), facet_region_count (16) and nquadpoints.facet (48) are " *
              "structural probes of the same cache: before #1 both selectors resolved to " *
              "subdomain 1's 2x2 geometry, so the counts read 2x8 = 16 and 2x24 = 48 rather " *
              "than 8+12 and 24+36. They now read 20 and 60 (both subdomains order 2). Every " *
              "other value here — above all the four sample values and max_error_vs_exact — " *
              "must stay bit-identical.")
    let V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=2),
        V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(3, 3), order=2),
        u1 = field(:u1, V1),
        u2 = field(:u2, V2),
        # Harmonic and linear, so each subdomain solution is the Dirichlet lift
        # alone and any error in the lift shows at every sample point.
        g = x -> 1 + x[1]

        model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2)),
                                dirichlet=[dirichlet(g; on=boundary(:all), field=:u1),
                                           dirichlet(g; on=boundary(:all), field=:u2)]))
        solution = solve!(model)
        samples = ((u1, ((0.5, 0.5), (0.25, 0.5))), (u2, ((2.5, 0.5), (2.25, 0.5))))
        emit("facet_cache_keys", length(model.facet_regions))
        emit_model(model)
        emit_solution(model, solution; fields=samples)
        worst = 0.0
        for (u, ps) in samples, p in ps
            worst = max(worst, abs(value(solution, model, u, p) - g(p)))
        end
        emit("max_error_vs_exact", worst)
    end

    # ── 17 ── Immersed surface integration whose segments lie strictly inside
    #          single cells: no grid crossing, so `_subdivide_segment` returns
    #          the segment untouched and #9 cannot reach this case. It is the
    #          control against which case 26's deficit is read.
    case(17, "d2-surface-in-cell", "MUST-NOT-CHANGE")
    let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2),
        square = polyline_mesh([SVector(0.03, 0.03), SVector(0.10, 0.03), SVector(0.10, 0.10),
                                SVector(0.03, 0.10)]; closed=true),
        u = field(:u, V)

        model = prepare(Problem((u,);
                                blocks=(stiffness_block(u),
                                        block(u, u, mass_form(coefficient=2.0); on=square)),
                                loads=(source_load(u; source=1.0),),
                                dirichlet=[dirichlet(0.0; on=boundary(:all))]))
        solution = solve!(model)
        emit_model(model)
        emit("surface_measure", boundary_integral(q -> 1.0, model; on=square))
        emit("surface_measure_exact", 4 * 0.07)
        emit("surface_measure_deficit", 4 * 0.07 - boundary_integral(q -> 1.0, model; on=square))
        emit_solution(model, solution; points=P2)
    end

    # ── 18 ── `L2Projection` transfer between two independent models.
    case(18, "d2-transfer-l2", "MUST-NOT-CHANGE")
    let source_space = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=2),
        target_space = space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=3)

        source_model = prepare(poisson(source_space; source=f2d, dirichlet=zero_dirichlet()))
        source_solution = solve!(source_model)
        target_model = prepare(poisson(target_space; source=f2d, dirichlet=zero_dirichlet()))
        transferred = transfer(source_solution, source_model, target_model)
        emit("source.active_unknowns", active_unknowns(source_model))
        emit("target.active_unknowns", active_unknowns(target_model))
        emit("transfer.method", transferred.diagnostics.method)
        emit("transfer.residual_norm", transferred.diagnostics.residual_norm)
        emit("transfer.converged", transferred.diagnostics.converged)
        emit_vector("transfer.coefficients", transferred.coefficients)
        emit("transfer.l2_error.absolute", l2_error(transferred, target_model, u2d; norm=:absolute))
        emit("transfer.l2_error.relative", l2_error(transferred, target_model, u2d))
        for p in P2
            emit("transfer.u" * pointlabel(p), value(transferred, target_model, p))
        end
        # The cached-matrix path over the same target must agree exactly.
        target_mass = assemble_matrix(target_model, mass_block(first(target_model.problem.fields)))
        reused = transfer(source_solution, source_model, target_model;
                          via=L2Projection(target_mass; factor=factorize(target_mass)))
        emit_vector("transfer.cached.coefficients", reused.coefficients)
        emit_matrix("transfer.target_mass", target_mass)
    end

    # ── 19 ── `move!` / `moved` / `transfer` on a model with no physical
    #          domain: the configuration where #4 has nothing to fold.
    case(19, "d1-move-overlay", "MUST-NOT-CHANGE")
    let build = ob -> begin
            V = space(box((0.0,), (1.0,)); cells=4, order=2)
            V = overlay(V, ob; cells=2, order=3)
            prepare(poisson(V; source=f1d, dirichlet=zero_dirichlet()))
        end
        model = build(box((0.25,), (0.75,)))
        solution = solve!(model)
        emit("start.active_unknowns", active_unknowns(model))
        emit("start.integration_regions", diagnostics(model).integration_regions)
        target = moved(model; level=2, to=box((0.1,), (0.6,)))
        transferred = transfer(solution, model, target)
        emit("moved.active_unknowns", active_unknowns(target))
        emit("moved.integration_regions", diagnostics(target).integration_regions)
        emit_vector("moved.transfer.coefficients", transferred.coefficients)
        emit("moved.transfer.l2_error", l2_error(transferred, target, u1d; norm=:absolute))
        move!(model; level=2, to=box((0.1,), (0.6,)))
        moved_solution = solve!(model)
        emit("move!.matches_moved.active", active_unknowns(model) == active_unknowns(target))
        emit("move!.matches_moved.regions",
             diagnostics(model).integration_regions == diagnostics(target).integration_regions)
        direct = build(box((0.1,), (0.6,)))
        emit("move!.matches_direct.active", active_unknowns(model) == active_unknowns(direct))
        emit_model(model)
        emit_solution(model, moved_solution; exact=u1d, points=P1)
    end

    # ── 20 ── D = 2 anisotropic mesh well clear of the merge tolerance: the
    #          healthy neighbour of cases 23/24, so #3's guard can be shown not
    #          to fire on a legitimately stretched mesh.
    case(20, "d2-anisotropic-healthy", "MUST-NOT-CHANGE")
    let V = space(box((0.0, 0.0), (1.0, 1.0e-3)); cells=(4, 4), order=2),
        exact = x -> x[1] * (1 - x[1])

        model = prepare(poisson(V; source=f1d, dirichlet=zero_dirichlet()))
        solution = solve!(model)
        emit("h_axis2", 1.0e-3 / 4)
        emit("tolerance.merge", sqrt(eps(Float64)))
        emit_model(model)
        emit_solution(model, solution; exact, points=((0.25, 0.0005), (0.5, 0.0005)))
    end

    # ── 21 ── #1. Two subdomains, one Neumann load each, on two *value-equal*
    #          boundary selectors. `_facet_selector_space` resolves both to the
    #          first referencing form's space, so subdomain 2's load is built
    #          against subdomain 1's geometry and silently vanishes.
    case(21, "x-coupled-value-equal-neumann", "EXPECTED-TO-CHANGE (#1)",
         wrong_because="u2(3.0, 0.5) is 0.0 against an exact 1.0, and rhs.block2.abs_sum is " *
                       "exactly 0.0: subdomain 2's Neumann load never lands. After the fix " *
                       "both subdomains must match their exact solution.")
    let V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=2),
        V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(2, 2), order=2),
        u1 = field(:u1, V1),
        u2 = field(:u2, V2)

        # u = x − x₀ on each subdomain: Δu = 0, u = 0 on the lower-x face,
        # ∂u/∂n = 1 on the upper-x face. The two selectors are separately
        # constructed and value-equal — the shape the finding names.
        model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2)),
                                loads=(neumann(u1, 1.0; on=boundary(axis=1, side=:upper)),
                                       neumann(u2, 1.0; on=boundary(axis=1, side=:upper))),
                                dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower),
                                                     field=:u1),
                                           dirichlet(0.0; on=boundary(axis=1, side=:lower),
                                                     field=:u2)]))
        solution = solve!(model)
        emit("facet_cache_keys", length(model.facet_regions))
        emit_model(model)
        n = active_unknowns(model)
        half = n ÷ 2
        emit_vector("rhs.block1", model.rhs[1:half])
        emit_vector("rhs.block2", model.rhs[(half+1):end])
        emit_solution(model, solution;
                      fields=((u1, ((1.0, 0.5), (0.5, 0.5))), (u2, ((3.0, 0.5), (2.5, 0.5)))))
        emit("u1_exact(1.0, 0.5)", 1.0)
        emit("u2_exact(3.0, 0.5)", 1.0)
    end

    # ── 22 ── #2. The same disc geometry as case 12, scaled to S = 1e-5, where
    #          a correct cut weight is ~1e-11 and `_NNLS_WEIGHT_TOL = 1e-12`
    #          truncates most of the rule to nothing.
    case(22, "x-fcm-cut-scale-1e-5", "EXPECTED-TO-CHANGE (#2)",
         wrong_because="empty_regions > 0 and domain_measure is a small fraction of exact_area, " *
                       "while every region is still tagged :cut_fitted and " *
                       "moment_fit_residual_max reads ~1e-27 because it is captured before " *
                       "truncation. matrix_rank < active_unknowns and matrix_isposdef is false.")
    let S = 1.0e-5,
        phi = x -> hypot(x[1] - 0.5S, x[2] - 0.5S) - 0.3S,
        p = physical_domain(phi; lipschitz=1.0, subcell_length_scale=S / 32, max_depth=5)

        V = space(box((0.0, 0.0), (S, S)); cells=(4, 4), order=2, physical=p)
        u = field(:u, V)
        model = prepare(Problem((u,); blocks=(stiffness_block(u), mass_block(u)),
                                loads=(source_load(u; source=1.0),)))
        assemble!(model)
        emit("length_scale", S)
        emit("exact_area", Float64(pi * (0.3S)^2))
        emit("nnls_weight_tol", 1.0e-12)
        emit_model(model)
        dense = Matrix(model.matrix)
        emit("matrix_rank", rank(dense))
        emit("matrix_isposdef", isposdef(dense))
        emit("measure_over_exact", domain_measure(model) / (pi * (0.3S)^2))
    end

    # ── 23 ── #3. One very thin axis: h = 1.25e-8 sits below
    #          `tolerance.merge = sqrt(eps) = 1.49e-8`, so `merge_coordinates`
    #          decimates the axis. B-splines swallow it and return a converged,
    #          entirely wrong solution.
    case(23, "x-thin-axis-bspline", "EXPECTED-TO-CHANGE (#3)",
         wrong_because="h_axis2 (1.25e-8) is below tolerance.merge (1.49e-8), so the y grid is " *
                       "decimated: integration_regions is 16 where the 4x8 mesh has 32 cells. " *
                       "The solve reports converged = true with a tiny residual and " *
                       "l2_error.relative ≈ 1.0 — the solution is ≈ 0 everywhere. After the " *
                       "fix this must raise, naming h, the tolerance, and the axis.")
    let H = 1.0e-7, exact = x -> x[1] * (1 - x[1])
        emit("h_axis2", H / 8)
        emit("tolerance.merge", sqrt(eps(Float64)))
        emit("mesh_cells", (4, 8))
        model = attempt("prepare",
                        () -> begin
                            V = space(box((0.0, 0.0), (1.0, H)); cells=(4, 8), order=3,
                                      basis=bspline())
                            prepare(poisson(V; source=f1d, dirichlet=zero_dirichlet()))
                        end)
        if model !== nothing
            solution = attempt("solve", () -> solve!(model))
            emit_model(model)
            if solution !== nothing
                emit_solution(model, solution; exact, points=((0.5, H / 2), (0.25, H / 2)))
            end
        end
    end

    # ── 24 ── #3, integrated Legendre on the same mesh: loud rather than silent
    #          today, but loud in the wrong way (a bare `SingularException` from
    #          the solver, with nothing naming the decimated axis).
    case(24, "x-thin-axis-legendre", "EXPECTED-TO-CHANGE (#3)",
         wrong_because="the same decimated grid (16 regions for 32 cells) reaches the solver, " *
                       "which dies with an unattributed SingularException. After the fix the " *
                       "failure must come from the geometry layer and name the axis.")
    let H = 1.0e-7, exact = x -> x[1] * (1 - x[1])
        emit("h_axis2", H / 8)
        emit("tolerance.merge", sqrt(eps(Float64)))
        model = attempt("prepare",
                        () -> begin
                            V = space(box((0.0, 0.0), (1.0, H)); cells=(4, 8), order=2)
                            prepare(poisson(V; source=f1d, dirichlet=zero_dirichlet()))
                        end)
        if model !== nothing
            attempt("assemble", () -> assemble!(model))
            emit_model(model)
            solution = attempt("solve", () -> solve!(model))
            solution === nothing || emit_solution(model, solution; exact, points=((0.5, H / 2),))
        end
    end

    # ── 25 ── #8. `lipschitz=` on a `LevelSet` argument is silently discarded,
    #          so the certified and uncertified paths integrate differently
    #          while both report a 3e-16 fit residual and a `:cut` label.
    case(25, "x-physical-domain-leaf-lipschitz", "EXPECTED-TO-CHANGE (#8)",
         wrong_because="kwarg_on_tree.sum_w equals uncertified.sum_w (1.0) rather than " *
                       "callable.sum_w (0.9195): the hole is invisible, an 8.0% " *
                       "over-integration of the cell. lipschitz = 0.0 is also accepted on the " *
                       "tree path where the callable path throws. After the fix the discarding " *
                       "combination must be rejected outright.")
    let region = box((0.0, 0.0), (1.0, 1.0)),
        order = (3, 3),
        phi = x -> 0.16 - hypot(x[1] - 0.25, x[2] - 0.25)   # Ω = outside the disc

        exact_area = 1 - pi * 0.16^2

        report = (tag, domain) -> begin
            emit(tag * ".classification", classify_cell(domain, region))
            _, ws, res = moment_fit_rule(domain, region, order; target_residual=1.0e-10)
            total = 0.0
            for w in ws
                total += w
            end
            emit(tag * ".points", length(ws))
            emit(tag * ".sum_w", total)
            emit(tag * ".residual", res)
            emit(tag * ".error_vs_exact", total - exact_area)
        end
        emit("exact_area", exact_area)
        # The discarding combination is what #8 rejects, so this one is built
        # under `attempt`: before the fix it constructs and reports the wrong
        # integral below, after the fix it throws and there is nothing to report.
        # Every other configuration here is legal on both sides of the fix.
        kwarg_on_tree = attempt("kwarg_on_tree.construct",
                                () -> physical_domain(leaf(phi); lipschitz=1.0,
                                                      subcell_length_scale=0.1))
        kwarg_on_tree === nothing || report("kwarg_on_tree", kwarg_on_tree)
        report("leaf_carries_L",
               physical_domain(leaf(phi; lipschitz=1.0); subcell_length_scale=0.1))
        report("callable", physical_domain(phi; lipschitz=1.0, subcell_length_scale=0.1))
        report("uncertified", physical_domain(leaf(phi); subcell_length_scale=0.1))
        attempt("tree_lipschitz_0",
                () -> physical_domain(leaf(phi); lipschitz=0.0, subcell_length_scale=0.1))
        attempt("callable_lipschitz_0",
                () -> physical_domain(phi; lipschitz=0.0, subcell_length_scale=0.1))
    end

    # ── 26 ── #9. A closed 24-gon inscribed in a circle of radius 0.3, whose
    #          perimeter is known exactly, integrated on an 8x8 grid it crosses
    #          many times. Every crossing discards 2·nudge of arc length.
    case(26, "x-surface-circle-perimeter", "EXPECTED-TO-CHANGE (#9)",
         wrong_because="surface_measure falls short of the analytic polygon perimeter by " *
                       "~4.7e-8 absolute / 2.5e-8 relative — the 2·nudge-per-crossing loss. " *
                       "After the fix the deficit must sit at round-off, as it already does " *
                       "for the crossing-free control in case 17.")
    let N = 24, R = 0.3, C = 0.5
        pts = [SVector(C + R * cos(2pi * (k - 1) / N), C + R * sin(2pi * (k - 1) / N)) for k in 1:N]
        Γ = polyline_mesh(pts; closed=true)
        exact_perimeter = N * 2R * sin(pi / N)
        V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(8, 8), order=2)
        u = field(:u, V)
        model = prepare(Problem((u,);
                                blocks=(stiffness_block(u),
                                        block(u, u, mass_form(coefficient=1.0); on=Γ)),
                                loads=(source_load(u; source=1.0),),
                                dirichlet=[dirichlet(0.0; on=boundary(:all))]))
        solution = solve!(model)
        measure = boundary_integral(q -> 1.0, model; on=Γ)
        emit("polygon_segments", N)
        emit("surface_measure", measure)
        emit("surface_measure_exact", exact_perimeter)
        emit("surface_measure_deficit", exact_perimeter - measure)
        emit("surface_measure_rel_deficit", (exact_perimeter - measure) / exact_perimeter)
        emit("nudge_2x", 2 * sqrt(eps(Float64)))
        emit("normal_integral_x", boundary_integral(q -> q.normal[1], model; on=Γ))
        emit("normal_integral_y", boundary_integral(q -> q.normal[2], model; on=Γ))
        emit_model(model)
        emit_solution(model, solution; points=P2)
    end

    # ── 27 ── A coupled model with a penalty interface. Correct today, and the
    #          patch test below reproduces the exact zero-flux solution — but
    #          its interface segments cross the grid, so #9's fix moves these
    #          numbers in their last ~8 digits. Nothing else may move them.
    case(27, "d2-coupled-interface", "EXPECTED-TO-CHANGE (#9 collateral)",
         wrong_because="nothing here is wrong today: max_error_vs_exact is at round-off. This " *
                       "case is marked EXPECTED-TO-CHANGE only because the interface polyline " *
                       "crosses grid lines, so fix #9 necessarily shifts the interface " *
                       "quadrature — interface_measure_deficit (1.8e-7 of a unit-length " *
                       "interface) is exactly what #9 hands back. Any change before #9, or a " *
                       "change larger than ~1e-6 relative after it, is a defect.")
    let V1 = space(box((0.0, 0.0), (1.0, 0.5)); cells=(4, 2), order=2),
        V2 = space(box((0.0, 0.5), (1.0, 1.0)); cells=(3, 2), order=2),
        u1 = field(:u1, V1),
        u2 = field(:u2, V2),
        Γ = polyline_mesh([SVector(0.0, 0.5), SVector(1.0, 0.5)]),
        β = 1.0e3,
        exact = x -> x[1] * (1 - x[1]) * x[2] * (1 - x[2])

        cpl = couple(u1, u2, Γ, mass_form(coefficient=β))
        model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2), cpl...),
                                loads=(source_load(u1; source=f2d), source_load(u2; source=f2d)),
                                dirichlet=[dirichlet(0.0; on=boundary(axis=2, side=:lower),
                                                     field=:u1),
                                           dirichlet(0.0; on=boundary(axis=1, side=:lower),
                                                     field=:u1),
                                           dirichlet(0.0; on=boundary(axis=1, side=:upper),
                                                     field=:u1),
                                           dirichlet(0.0; on=boundary(axis=2, side=:upper),
                                                     field=:u2),
                                           dirichlet(0.0; on=boundary(axis=1, side=:lower),
                                                     field=:u2),
                                           dirichlet(0.0; on=boundary(axis=1, side=:upper),
                                                     field=:u2)]))
        solution = solve!(model)
        # Read the interface-region cache through the very object that keys it:
        # the cache is an `IdDict`, so iterating it would expose identity-hash
        # order to the report. `prepare` pre-resolves every referenced interface,
        # so a `KeyError` here would itself be a finding.
        regions = model.interface_regions[cpl[1].on]
        measure = 0.0
        for region in regions, w in region.weights
            measure += w
        end
        emit("interface_measure", measure)
        emit("interface_measure_exact", 1.0)
        emit("interface_measure_deficit", 1.0 - measure)
        emit("interface_regions", length(regions))
        emit_model(model)
        p1 = ((0.3, 0.2), (0.5, 0.25))
        p2 = ((0.3, 0.8), (0.5, 0.75))
        emit_solution(model, solution; fields=((u1, p1), (u2, p2)))
        worst = 0.0
        for (u, ps) in ((u1, p1), (u2, p2)), p in ps
            worst = max(worst, abs(value(solution, model, u, p) - exact(p)))
        end
        emit("max_error_vs_exact", worst)
    end

    # ── 28 ── #4. `move!` on a model with a physical domain: the overlay's
    #          folded mask is re-folded instead of the user's, so cells that the
    #          geometry frees at the new position stay dropped.
    case(28, "x-move-with-physical-domain", "EXPECTED-TO-CHANGE (#4)",
         wrong_because="the fictitious fold is applied to the ALREADY-FOLDED mask, so a " *
                       "move can only ever shrink the active set. The overlay starts wholly " *
                       "outside Ω (all four cells folded away) and moves wholly inside it, " *
                       "where every cell should be active: `move!` keeps inactive=[8, 4] and " *
                       "32 unknowns over 8 regions, while a direct prepare at the same box " *
                       "gives inactive=[8, 0], 41 unknowns and 16 regions. After the fix " *
                       "moved and direct must agree in every printed value. (The geometry " *
                       "here is deliberate: an earlier version of this case moved between " *
                       "two boxes that both overlap Ω, where the folded mask happens to be " *
                       "correct at both ends, and it showed nothing.)")
    let phi = x -> hypot(x[1] - 0.25, x[2] - 0.25) - 0.28,
        p = physical_domain(phi; lipschitz=1.0, subcell_length_scale=1 / 32, max_depth=4),
        build = ob -> begin
            V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=p)
            V = overlay(V, ob; cells=(2, 2), order=2)
            u = field(:u, V)
            prepare(Problem((u,); blocks=(stiffness_block(u), mass_block(u)),
                            loads=(source_load(u; source=1.0),)))
        end,
        start_box = box((0.55, 0.55), (0.95, 0.95)),
        to_box = box((0.05, 0.05), (0.45, 0.45))

        model = build(start_box)
        emit("start.active_unknowns", active_unknowns(model))
        emit("start.mask.level2", maskstr(active_cells(model; level=2)))
        emit("start.inactive_cell_counts", diagnostics(model).inactive_cell_counts)
        move!(model; level=2, to=to_box)
        moved_solution = solve!(model)
        emit("moved.mask.level2", maskstr(active_cells(model; level=2)))
        emit_model(model)
        emit_solution(model, moved_solution; points=((0.5, 0.5), (0.4, 0.6)))

        direct = build(to_box)
        direct_solution = solve!(direct)
        emit("direct.mask.level2", maskstr(active_cells(direct; level=2)))
        emit("direct.active_unknowns", active_unknowns(direct))
        emit("direct.integration_regions", diagnostics(direct).integration_regions)
        emit("direct.inactive_cell_counts", diagnostics(direct).inactive_cell_counts)
        emit("direct.reduced_mode_counts", diagnostics(direct).reduced_mode_counts)
        emit("direct.domain_measure", domain_measure(direct))
        emit_matrix("direct.matrix", direct.matrix)
        emit_vector("direct.coefficients", direct_solution.coefficients)
        emit("match.active_unknowns", active_unknowns(model) == active_unknowns(direct))
        emit("match.mask",
             maskstr(active_cells(model; level=2)) == maskstr(active_cells(direct; level=2)))
        emit("match.inactive_cell_counts",
             diagnostics(model).inactive_cell_counts == diagnostics(direct).inactive_cell_counts)
        emit("match.reduced_mode_counts",
             diagnostics(model).reduced_mode_counts == diagnostics(direct).reduced_mode_counts)
        emit("match.regions",
             diagnostics(model).integration_regions == diagnostics(direct).integration_regions)
        emit("match.domain_measure", domain_measure(model) == domain_measure(direct))
        emit("match.matrix_nnz", nnz(model.matrix) == nnz(direct.matrix))
        emit("match.matrix_values", nonzeros(model.matrix) == nonzeros(direct.matrix))
        emit("match.coefficients", moved_solution.coefficients == direct_solution.coefficients)
    end

    # ── 29 ── #5. `moved` on a coupled model has no single-domain guard: every
    #          field is re-homed onto subdomain 1's space, and the result solves.
    case(29, "x-moved-on-coupled-model", "EXPECTED-TO-CHANGE (#5)",
         wrong_because="moved returns a model with subdomain_spaces = 1 and field :u2 living " *
                       "on [0,1]^2 instead of [2,3]x[0,1], and solve! accepts it. After the " *
                       "fix `moved` must throw ArgumentError, as `move!` already does.")
    let V1 = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=2),
                     box((0.25, 0.25), (0.75, 0.75)); cells=(2, 2), order=2),
        V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(2, 2), order=2),
        u1 = field(:u1, V1),
        u2 = field(:u2, V2)

        model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2)),
                                loads=(source_load(u1; source=1.0), source_load(u2; source=1.0)),
                                dirichlet=[dirichlet(0.0; on=boundary(:all), field=:u1),
                                           dirichlet(0.0; on=boundary(:all), field=:u2)]))
        emit("source.subdomain_spaces", length(problem_spaces(model.problem)))
        emit("source.active_unknowns", active_unknowns(model))
        target = attempt("moved", () -> moved(model; level=2, to=box((0.1, 0.1), (0.6, 0.6))))
        if target !== nothing
            emit("moved.subdomain_spaces", length(problem_spaces(target.problem)))
            emit("moved.active_unknowns", active_unknowns(target))
            for f in target.problem.fields
                emit("moved.field." * String(f.name) * ".domain",
                     (Tuple(f.space.domain.lower)..., Tuple(f.space.domain.upper)...))
            end
            solution = attempt("moved.solve", () -> solve!(target))
            if solution !== nothing
                emit_model(target)
                emit_vector("moved.coefficients", solution.coefficients)
            end
        end
    end

    # ── 30 ── D = 2, integrated Legendre base under a masked B-spline overlay. The
    #          one family pairing no other case reaches: case 09 is spline over spline,
    #          where the base takes the no-op fallback and covered-mode pruning never runs.
    #          Appended after the EXPECTED-TO-CHANGE block rather than filed with the
    #          other MUST-NOT-CHANGE cases so the 29 cases above keep their numbering.
    case(30, "d2-legendre-bspline-masked", "MUST-NOT-CHANGE",
         note="the overlay is a nested 2x2 block of base cells, so the base vertex at " *
              "(0.375, 0.375) is buried and mesh nesting alone would dedup it. It must " *
              "survive: a maximal-smoothness spline is C^(p-1) at a simple interior knot " *
              "and has no kink to replace that hat with. Pins the survival — " *
              "reduced_mode_counts = [8, 0], the buried high-order modes and nothing " *
              "else. Deduping the vertex prints [9, 0], one fewer active unknown, and a " *
              "space that cannot reproduce a constant.")
    let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(8, 8), order=2), mask = falses(4, 4)
        mask[1:2, 1:2] .= true
        V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(4, 4), order=3, basis=bspline(),
                    active=mask)
        model, _ = poisson_case(V, f2d, u2d, P2)
        emit("active_cells.level2", maskstr(active_cells(model; level=2)))
    end

    # ── 31 ── D = 2 under an anisotropic diffusion tensor. Every other case in
    #          this report leaves `diffusion` at its default `1`, i.e. the scalar
    #          branch of `_diffusion_flux`; without this case the `AbstractMatrix`
    #          branch — how `A ∇u` is contracted — is unpinned.
    case(31, "d2-diffusion-tensor", "MUST-NOT-CHANGE",
         note="pins the diffusion-tensor contraction ⟨∇v, A ∇u⟩ for the three spellings of " *
              "`diffusion` the docstring offers: a dense `Matrix{Float64}`, the same tensor " *
              "as a static `SMatrix`, and an isotropic scalar. The solved model's A is " *
              "symmetric, which `stiffness_form`'s `symmetric = true` requires, so it cannot " *
              "see the contraction's orientation; the `flux.*` keys close that by contracting " *
              "a deliberately NON-symmetric tensor directly, where A and Aᵀ differ. " *
              "`dense_equals_static` pins that the two matrix spellings are one code path: " *
              "`_diffusion_flux` copies any `AbstractMatrix` into an `SMatrix` before the " *
              "product, so both assemble and solve to the same bits at every optimisation " *
              "level. Contracting the dense spelling through the generic dense product " *
              "instead prints false above -O0 and still true at the -O0 this baseline is " *
              "generated at, so that key guards the invariant only above -O0.")
    let V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2),
        dense = [2.0 0.75; 0.75 3.0],
        static = SMatrix{2,2}(dense),
        # −∇·(A ∇u) for u = u2d and this constant symmetric A. u2d is degree 2
        # per axis, so `order = 2` reproduces it and l2_error sits at round-off.
        f_tensor = x -> (2 * dense[1, 1] * x[2] * (1 - x[2]) + 2 * dense[2, 2] * x[1] * (1 - x[1]) -
                         (dense[1, 2] + dense[2, 1]) * (1 - 2 * x[1]) * (1 - 2 * x[2]))

        emit("tensor", (dense[1, 1], dense[1, 2], dense[2, 1], dense[2, 2]))

        # The contraction on its own, against a non-symmetric tensor. The
        # solved model below carries a symmetric A, where swapping A for Aᵀ is
        # invisible; here it moves these numbers, as does landing on the wrong
        # branch of `_diffusion_flux`. A change of rounding alone does not move
        # them at the -O0 this baseline is generated at.
        let skew = [2.0 0.75; 0.25 3.0], g = SVector(0.3, -0.7)
            emit("flux.dense", Tuple(_diffusion_flux(skew, g)))
            emit("flux.static", Tuple(_diffusion_flux(SMatrix{2,2}(skew), g)))
            emit("flux.scalar", Tuple(_diffusion_flux(2.5, g)))
            emit("flux.uniform_scaling", Tuple(_diffusion_flux(2.5I, g)))
            attempt("flux.wrong_size", () -> _diffusion_flux([1.0 0.0 0.0; 0.0 1.0 0.0], g))
        end

        model, solution = diffusion_case(V, dense, f_tensor)
        emit_model(model)
        emit_solution(model, solution; exact=u2d, points=P2)

        static_model, static_solution = diffusion_case(V, static, f_tensor)
        emit_matrix("static.matrix", static_model.matrix)
        emit_solution(static_model, static_solution; exact=u2d, points=P2, prefix="static.")
        emit("dense_equals_static.matrix", model.matrix == static_model.matrix)
        emit("dense_equals_static.coefficients",
             solution.coefficients == static_solution.coefficients)

        # The isotropic branch at a non-unit coefficient, which no other case
        # reaches: 2.5 f2d is the source that keeps u2d exact for diffusion 2.5.
        scalar_model, scalar_solution = diffusion_case(V, 2.5, x -> 2.5 * f2d(x))
        emit_matrix("scalar.matrix", scalar_model.matrix)
        emit_solution(scalar_model, scalar_solution; exact=u2d, points=P2, prefix="scalar.")
    end

    # ── 32 ── #T4. A cut face carrying a NONZERO Dirichlet datum — the one shape
    #          neither this report nor any shipped application had. On the x₁ = 0
    #          face the leaf x₂ − x₁ − 0.55 restricts to φ = x₂ − 0.55, so of that
    #          face's unit measure exactly 0.55 is physical: the crossing sits at
    #          x₂ = 0.55, inside the third of the face's four regions, which the
    #          plane cuts while the fourth lies wholly above it. The datum is
    #          nonzero because `_needs_dirichlet_projection` skips the boundary
    #          walk outright for a constant-zero one, and a skipped walk would
    #          leave the trace projection out of the report altogether.
    case(32, "x-facet-cut-nonzero-datum", "MUST-NOT-CHANGE",
         note="pins the trimmed rule on a cut face carrying a nonzero datum, which is the " *
              "shape no shipped application has. facet_measure.x_lower reads 0.55 and " *
              "facet_flux.x_lower -0.55, both machine-exact — the restricted leaf is linear " *
              "on the face — against 1.0 and -1.0 before the rule was trimmed, when 45% of " *
              "the weight sat on area Ω does not contain. The flux matters on its own: " *
              "nothing else in the repository measures one over a cut face, and a reaction " *
              "force or a heat flow *is* that integral rather than merely being fitted over " *
              "it. nquadpoints.facet 12 → 11, the fitted rule on the cut region being " *
              "smaller than the tensor product it replaced and the wholly fictitious region " *
              "carrying none. dirichlet.trace_factors [:cholesky] → [:pseudoinverse], " *
              "unsupported_dof_count 0 → 2 and nonzero_value_count 9 → 7 are a CONSEQUENCE and " *
              "not part of the fix: the two dofs supported only on the fourth region, whose " *
              "face lies wholly outside Ω, are constrained with no measure to fit them on, so " *
              "the trace solve falls back to a pseudoinverse that pins them to their " *
              "minimum-norm value. Case 33 names that hazard.")
    let plane = leaf(x -> x[2] - x[1] - 0.55; lipschitz=sqrt(2.0)),
        p = physical_domain(plane; subcell_length_scale=0.0625),
        face = boundary(axis=1, side=:lower),
        g = x -> 1 + x[1] + 2x[2]

        V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=p)
        model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(g; on=face)]))
        solution = solve!(model)
        emit_model(model)
        emit_facet_rule(model)
        emit("facet_measure.x_lower", boundary_integral(q -> 1.0, model; on=face))
        emit("facet_measure_exact.x_lower", 0.55)
        # The trimmed FLUX, which nothing else in the repository measures — no
        # shipped example puts a Neumann or Robin term on a face that ∂Ω cuts. The
        # outward normal here is the constant −e₁, so this integral is exactly
        # minus the measure above and moves with it; a flux is the consumer a wrong
        # face measure corrupts most directly, since a reaction force or a heat
        # flow *is* this integral rather than merely being fitted over it.
        emit("facet_flux.x_lower", boundary_integral(q -> q.normal[1], model; on=face))
        emit("facet_flux_exact.x_lower", -0.55)
        # No manufactured solution: g is harmonic, but the immersed cut carries the
        # natural condition rather than g, so the solution is not g anywhere. These
        # are diffable sample values, not an accuracy claim.
        emit_solution(model, solution; points=P2)
        emit_dirichlet_projection(model)
    end

    # ── 33 ── #T4 at its sharpest: a face lying WHOLLY outside Ω on a cell that
    #          is still active. Ω = (0,1)² ∖ {‖x‖ ≤ 0.3} puts the hole on the
    #          corner, so the corner cell [0, 0.25]² is cut — its far corner sits
    #          at r = 0.354 > 0.3 — and is therefore active and parents a region
    #          on each of its two boundary faces, while both of those faces lie
    #          at r ≤ 0.25 < 0.3, entirely in the hole. The whole-cell fictitious
    #          fold cannot reach them: the cell is not fictitious, only its faces
    #          are, which is precisely the configuration a face can only lose by
    #          being trimmed within one cell.
    case(33, "x-facet-wholly-fictitious", "MUST-NOT-CHANGE",
         note="the sharpest form of the same thing, and the one that names the hazard trimming " *
              "leaves open. The two faces the disc reaches read 0.7, machine-exact — it " *
              "restricts to a linear φ on them — and the other two keep reading 1.0; all " *
              "four read 1.0 before the rule was trimmed. The corner cell classifies :cut, " *
              "hence stays active and parents a region on each of its two faces, while a " *
              "positive corner_cell.face_phi_at_far_end says both of those faces lie entirely " *
              "outside Ω (φ = 0.3 − x₂ is monotone along the face, so its far end is the " *
              "least-fictitious point there is) — the configuration no whole-cell fold can " *
              "reach. Three dofs lose their rows of the trace mass entirely — the node at " *
              "the origin and each face's own bubble over that cell — so " *
              "dirichlet.trace_factors moves [:cholesky] → [:pseudoinverse] and " *
              "dirichlet.unsupported_dof_count 0 → 3, while dirichlet.nonzero_value_count " *
              "only goes 32 → 31: the pseudoinverse's minimum-norm value is zero to the " *
              "roundoff of its SVD, so two of the three stay ~1e-16 and a scan of the values " *
              "cannot see them. That outcome is reported here, not chosen: a dof can carry a " *
              "Dirichlet condition with no measure anywhere on its facet support while its " *
              "volume support still reaches into Ω, and what such a condition ought to mean " *
              "is an open semantic question rather than a defect in the rule.")
    let disc = leaf(x -> 0.3 - hypot(x[1], x[2]); lipschitz=1.0),
        p = physical_domain(disc; subcell_length_scale=0.0625),
        # −Δ sin(x₁ + 2x₂) = 5 sin(x₁ + 2x₂), so g is the manufactured solution of
        # the box problem. It is NOT the solution here: the immersed hole carries
        # the natural condition, which g does not satisfy, so l2_error.* reads a
        # few percent and is a diffable scalar rather than an accuracy claim. It
        # earns its place by being global — it moves if the trace fit moves at all.
        g = x -> sin(x[1] + 2x[2])

        V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2, physical=p)
        model = prepare(poisson(V; source=x -> 5 * sin(x[1] + 2x[2]),
                                dirichlet=[dirichlet(g; on=boundary(:all))]))
        solution = solve!(model)
        emit_model(model)
        emit_facet_rule(model)
        measure = face -> boundary_integral(q -> 1.0, model; on=face)
        emit("facet_measure.x_lower", measure(boundary(axis=1, side=:lower)))
        emit("facet_measure.x_upper", measure(boundary(axis=1, side=:upper)))
        emit("facet_measure.y_lower", measure(boundary(axis=2, side=:lower)))
        emit("facet_measure.y_upper", measure(boundary(axis=2, side=:upper)))
        # Only the two faces the disc reaches are wrong; the upper two already read
        # their physical measure of 1.0, and that they keep reading it is half the
        # point of printing all four.
        emit("facet_measure_exact.x_lower", 1 - 0.3)
        emit("facet_measure_exact.y_lower", 1 - 0.3)
        # The hazard, pinned so it cannot quietly stop being the hazard: were the
        # corner cell ever to classify fictitious, the whole-cell fold would drop
        # its faces and the `pinv` transition above would not be reachable here.
        # The two faces are mirror images through x₁ ↔ x₂ and `hypot` is symmetric,
        # so one φ reading covers both.
        emit("corner_cell.classification",
             classify_cell(p, cell_box(V, CartesianIndex(1, 1); level=1)))
        emit("corner_cell.active", active_cells(model; level=1)[1, 1])
        emit("corner_cell.face_phi_at_far_end", levelset_value(p, SVector(0.0, 0.25)))
        emit_solution(model, solution; exact=g, points=P2)
        emit_dirichlet_projection(model)
    end

    # ── Summary ───────────────────────────────────────────────────────────────
    println()
    println("SUMMARY")
    emit("cases", length(CASE_MARKS))
    emit("must_not_change", count(m -> startswith(m, "MUST-NOT-CHANGE"), CASE_MARKS))
    emit("expected_to_change", count(m -> startswith(m, "EXPECTED-TO-CHANGE"), CASE_MARKS))
    return nothing
end

main()
