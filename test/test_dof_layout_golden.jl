using Unfitted
using BasicBSpline   # switches on the B-spline extension for the clamped and maximal cases
using StaticArrays: SVector
using Unfitted: cell_dofs, constraint_kind, is_active

# Golden values for the dof layout of seven small configurations.
#
# WHY THIS FILE EXISTS. The dof layer — raw enumeration, the artificial boundary, leaf-semantics pruning,
# physical-Dirichlet detection, the active numbering and the linear-constraint pivots — is the object a rewrite of its
# data model must keep exactly. The suite's other tests check properties of it (a count, a symmetry, an error), and an
# aggregate such as an unknown count or an error norm can absorb a renumbered, relabelled or re-pivoted dof and stay
# plausible. This instrument sees the layout itself, integer by integer.
#
# WHAT IT LOCKS. Per configuration, through the public accessors and the one pivot rule assembly and `dof_value` share,
# and through nothing that names how a layout is stored: every field's name, component count, offset and raw count;
# every raw key's integers; every active cell's raw ids (`cell_dofs`), in local-basis order; the active id of every
# (raw, component); the `constraint_kind` of every (raw, component); every linear-constraint pivot's branches and
# weights, read through `_fold_pivot` with the identity id map; and the model's `reduced_mode_counts` and active
# unknowns. A rewrite that keeps the layout keeps this digest whatever it stores.
#
# WHAT IT LEAVES OUT. Constrained values: they come out of a LAPACK factorisation, which is not bit-reproducible across
# machines. Pivot weights are quantised to `round(Int64, w · 2³²)` for the same reason — a clamped trace row's weights
# are floating-point quotients — while still separating any two weights that differ in a meaningful digit.
#
# HOW IT LOCKS IT. A 64-bit FNV-1a checksum over the integers themselves, as in `test_graded_golden.jl`, not through
# `Base.hash`, whose value carries no stability guarantee across Julia versions. The helpers carry a `_dl_` prefix so
# the single-process `Pkg.test` sees no redefinition of that file's `_mix`.
#
# WHEN IT FAILS. Each configuration is its own testset, and its named assertions (unknowns, pivots, reduced modes) run
# before its digest, so a failure names the configuration and usually the quantity that moved. A moved digest with
# unchanged named assertions means a key, a cell table, the numbering or a classification moved. The values were written
# at the start of the dof-layout rewrite and are not values to update casually: a change here is either the defect being
# hunted or a deliberate change of the discrete space, and the commit that makes it says which.

const _DL_FNV_PRIME = 0x00000100000001b3
const _DL_FNV_BASIS = 0xcbf29ce484222325

# One FNV-1a step per byte of the integer's two's-complement 64-bit image.
function _dl_mix(h::UInt64, x::Integer)
    v = UInt64(unsigned(Int64(x)) & typemax(UInt64))
    for shift in 0:8:56
        h = (h ⊻ ((v >> shift) & 0xff)) * _DL_FNV_PRIME
    end
    return h
end

# `constraint_kind` as an integer. An unknown kind raises rather than hashing to an arbitrary code, so a new
# classification cannot slip through as a digest change.
const _DL_KINDS = (:free, :dirichlet, :overlay, :coverage, :dedup, :mixed)
function _dl_kind_code(kind::Symbol)
    code = findfirst(==(kind), _DL_KINDS)
    code === nothing && throw(ArgumentError("unexpected constraint kind $kind"))
    return code
end

# The pivot of `raw`, if it is one, as `[(branch raw, weight)]` in expansion order. The identity id map makes
# `_fold_pivot` hand back raw ids rather than active ids, so the read is independent of the numbering and never touches
# a constrained value.
function _dl_pivot(dofs, raw::Integer)
    branches, pivot = Unfitted._fold_pivot((acc, g, w, v) -> push!(acc, (g, w)), Tuple{Int,Float64}[], dofs,
                                           (o, c) -> o, raw, 1)
    return pivot ? branches : nothing
end

# The digest of one prepared model, and the quantities its named assertions read. Fields are visited in the system's
# order and each field's levels in its space's order; `spaces` maps a field name to the space the field was prepared
# over, whose level masks say which cells are active.
function _dl_digest(model)
    h = _DL_FNV_BASIS
    pivots = 0
    spaces = Dict(f.name => f.space for f in model.problem.fields)
    for f in model.dofs.fields
        dofs = f.dofs
        nraw = length(dofs.raw_keys)
        for b in codeunits(String(f.name))
            h = _dl_mix(h, b)
        end
        h = _dl_mix(_dl_mix(_dl_mix(h, f.components), f.offset), nraw)
        for key in dofs.raw_keys
            h = _dl_mix(h, key.level)
            for a in key.axes
                h = _dl_mix(_dl_mix(_dl_mix(h, a.kind), a.index), a.mode)
            end
        end
        for level in spaces[f.name].levels
            h = _dl_mix(h, level.id)
            for (n, ci) in enumerate(cell_indices(level.mesh))
                is_active(level.mask, ci) || continue
                raws = cell_dofs(dofs, level.id, ci)
                h = _dl_mix(_dl_mix(h, n), length(raws))
                for raw in raws
                    h = _dl_mix(h, raw)
                end
            end
        end
        for c in 1:f.components, raw in 1:nraw
            h = _dl_mix(h, dofs.active_component[raw, c])
            h = _dl_mix(h, _dl_kind_code(constraint_kind(dofs, raw, c)))
        end
        for raw in 1:nraw
            branches = _dl_pivot(dofs, raw)
            branches === nothing && continue
            pivots += 1
            h = _dl_mix(_dl_mix(h, raw), length(branches))
            for (other, w) in branches
                h = _dl_mix(_dl_mix(h, other), round(Int64, w * 2^32))
            end
        end
    end
    reduced = diagnostics(model).reduced_mode_counts
    h = _dl_mix(h, length(reduced))
    for r in reduced
        h = _dl_mix(h, r)
    end
    h = _dl_mix(h, active_unknowns(model))
    return (; unknowns=active_unknowns(model), pivots, reduced=collect(Int, reduced), digest=h)
end

# A Poisson problem (stiffness plus a unit source) over `V` with the given Dirichlet list.
function _dl_problem(V, bcs; components::Integer=1)
    u = field(:u, V; components=components)
    src = components == 1 ? 1.0 : SVector(ntuple(_ -> 1.0, components))
    return Problem((u,); blocks=(stiffness_block(u),), loads=(source_load(u; source=src),), dirichlet=bcs)
end

# The seven configurations: each a different way for the layout to be non-trivial.
const _DL_U2 = box((0.0, 0.0), (1.0, 1.0))
function _dl_configurations()
    # 1. IL overlay with nonzero data on all faces but one. The overlay sits on the corner where a Dirichlet face meets
    #    the excepted one, so detection must keep the excepted face natural on both levels.
    c1 = _dl_problem(overlay(space(_DL_U2; cells=6, order=3), box((0.0, 2 / 3), (1 / 3, 1.0)); cells=4, order=2),
                     [dirichlet(x -> 1.0 + x[1] * x[2]; on=boundary(:all; except=(axis=2, side=:upper)))])
    # 2. An adapted IL ladder: a corner refined to the bottom rung, graded.
    depths = zeros(Int, 4, 4)
    depths[1, 1] = 3
    c2 = _dl_problem(adapt(ladder(_DL_U2; cells=4, order=2, depth=3), depths; grade=1),
                     [dirichlet(0.0; on=boundary(:all))])
    # 3. A 3D IL overlay refining one corner cell of the base.
    c3 = _dl_problem(overlay(space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=3, order=2),
                             box((0.0, 0.0, 0.0), (1 / 3, 1 / 3, 1 / 3)); cells=2, order=2),
                     [dirichlet(0.0; on=boundary(:all))])
    # 4. A 2-component IL field: an `:except` condition on both components and a component-2 condition on the excepted
    #    face, over an overlay on that face.
    c4 = _dl_problem(overlay(space(_DL_U2; cells=4, order=2), box((0.5, 0.25), (1.0, 0.75)); cells=2, order=3),
                     [dirichlet(SVector(0.0, 0.0); on=boundary(:all; except=(axis=1, side=:upper))),
                      dirichlet(x -> 1.0 + x[2]; on=boundary(axis=1, side=:upper), component=2)]; components=2)
    # 5. A clamped C¹ B-spline level under an L mask: the masked faces' trace rows leave linear-constraint pivots.
    mask = trues(8, 8)
    mask[5:8, 5:8] .= false
    c5 = _dl_problem(space(_DL_U2; cells=8, order=3, basis=bspline(; continuity=1), active=mask),
                     [dirichlet(0.0; on=boundary(:all))])
    # 6. A maximal B-spline base under a nested maximal cover: support selection on the overlay, the B-spline dedup on
    #    the base.
    c6 = _dl_problem(overlay(space(_DL_U2; cells=8, order=3, basis=bspline()), box((0.25, 0.25), (0.75, 0.75)); cells=8,
                             order=3, basis=bspline()), [dirichlet(0.0; on=boundary(:all))])
    # 7. A 1D IL overlay with nonzero data at both ends.
    c7 = _dl_problem(overlay(space(box((0.0,), (1.0,)); cells=8, order=3), box((0.25,), (0.5,)); cells=4, order=2),
                     [dirichlet(x -> 1.0 + x[1]; on=boundary(:all))])
    return ["IL 2D overlay, nonzero :except" => c1, "IL 2D adapted ladder" => c2, "IL 3D overlay" => c3,
            "IL 2-component field, :except + component" => c4, "clamped C1 B-spline, L mask" => c5,
            "maximal B-spline, nested cover" => c6, "IL 1D overlay, nonzero" => c7]
end

# Written at the round-2 baseline (`ff390ba`): unknowns, pivots, reduced modes, digest.
const _DL_GOLDEN = [(332, 0, [36, 0], 0xf0ccf0662e2096cf),         # 1. IL 2D overlay
                    (497, 0, [36, 64, 64, 0], 0x2ff3a8c13faa4987), # 2. IL 2D ladder
                    (151, 0, [8, 0], 0x7b635d24165ea243),          # 3. IL 3D overlay
                    (139, 0, [12, 0], 0xb4b9d4a456bc9be7),         # 4. 2-component field
                    (52, 24, [0], 0x99f646a1bd4bee52),             # 5. clamped, L mask
                    (105, 0, [1, 0], 0xe65f9b923b3ecffc),          # 6. maximal, nested cover
                    (25, 0, [5, 0], 0x7f0cec4d9dd4a035)]           # 7. IL 1D overlay

@testset "dof layout golden: seven configurations are locked" begin
    configurations = _dl_configurations()
    @test length(configurations) == length(_DL_GOLDEN)
    for ((name, problem), (unknowns, pivots, reduced, digest)) in zip(configurations, _DL_GOLDEN)
        @testset "$name" begin
            got = _dl_digest(prepare(problem))
            @test got.unknowns == unknowns
            @test got.pivots == pivots
            @test got.reduced == reduced
            @test got.digest == digest
        end
    end
end
