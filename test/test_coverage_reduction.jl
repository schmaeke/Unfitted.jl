# Regression tests for covered-mode pruning in covered regions (the coverage constraint
# source). These promote the design spike's configurations to guarded unit tests.
#
# The checks are non-invasive: the mass matrix is the Gram matrix of the active basis,
# so its rank equals the number of active dofs iff the retained functions are linearly
# independent, and the L² projection of a low-order polynomial reproduces it iff the
# space is complete to that order.

using Unfitted
using Unfitted: _field_layout, active_unknowns, constraint_kind
using LinearAlgebra
using StaticArrays

const _OMEGA_CR = box((0.0, 0.0), (1.0, 1.0))

# Assemble the mass (Gram) matrix and the field's dof layout for a space.
# `prune = false` is `prepare`'s documented diagnostic: the same geometry with
# leaf semantics switched off, which is the unreduced twin every losslessness
# assertion below is made against.
function _gram(V; prune::Bool=true)
    model = prepare(mass(V); prune=prune)
    assemble!(model)
    return model, _field_layout(model.dofs, :u).dofs
end

# L² projection residual ‖f − Πf‖ of `f` onto the assembled space, using a
# pseudoinverse so it stays defined when the full (unreduced) Gram is rank-deficient.
function _proj_residual(model, f, exact_sq)
    M = Symmetric(Matrix(model.matrix))
    b = load_vector(model; source=f)
    c = pinv(M) * b
    return sqrt(max(exact_sq - dot(b, c), 0.0))
end

_one(_) = 1.0
_x1(x) = x[1]
# Exact ∫ over the unit square.
const _INT_ONE = 1.0
const _INT_X1SQ = 1 / 3

# Reduced 2×2 block: base 4×4 p=2, overlay covering the middle 2×2 base cells at p=3,
# aligned (overlay nodes ⊇ base nodes), so the buried centre vertex is deduped.
function _nested_block()
    return overlay(space(_OMEGA_CR; cells=(4, 4), order=2), box((0.25, 0.25), (0.75, 0.75));
                   cells=(4, 4), order=3)
end

@testset "leaf semantics switched off is a no-op" begin
    _, lf = _gram(_nested_block(); prune=false)
    @test all(s -> s in (:free, :overlay), lf.elimination_source)
    @test count(==(:coverage), lf.elimination_source) == 0
    @test count(==(:dedup), lf.elimination_source) == 0
end

@testset "covered-mode pruning: nested block sheds high-order, stays complete and full rank" begin
    mf, lf = _gram(_nested_block(); prune=false)
    mr, lr = _gram(_nested_block())

    # Fewer active dofs, and the drop is exactly the covered high-order plus the one
    # deduped centre vertex.
    @test active_unknowns(lr) < active_unknowns(lf)
    @test count(==(:coverage), lr.elimination_source) == 8
    @test count(==(:dedup), lr.elimination_source) == 1
    @test active_unknowns(lf) - active_unknowns(lr) == 9

    # Reduced space: linearly independent (Gram full rank + SPD → no dead dof) and
    # complete to first order (constant + linear reproduced exactly).
    Mr = Symmetric(Matrix(mr.matrix))
    @test rank(Mr) == active_unknowns(lr)
    @test isposdef(Mr)
    @test _proj_residual(mr, _one, _INT_ONE) < 1e-6
    @test _proj_residual(mr, _x1, _INT_X1SQ) < 1e-6
end

@testset "diagnostics report per-level reduced-mode counts" begin
    m = prepare(mass(_nested_block()))
    counts = diagnostics(m).reduced_mode_counts
    @test length(counts) == 2          # base + overlay
    @test counts[1] == 9               # base (level 1): 8 coverage + 1 dedup
    @test counts[2] == 0               # overlay (level 2, top of stack): untouched
end

@testset "coincident same-order cell: rank deficiency removed" begin
    # Overlay is one p=2 cell coincident with the interior base cell (2,2). Its only
    # surviving mode duplicates the base cell's bubble → the full space is rank
    # deficient by one; covered-mode pruning drops that buried bubble.
    coincident = overlay(space(_OMEGA_CR; cells=(4, 4), order=2), box((0.25, 0.25), (0.5, 0.5));
                         cells=(1, 1), order=2)

    mf, lf = _gram(coincident; prune=false)
    mr, lr = _gram(coincident)
    @test rank(Symmetric(Matrix(mf.matrix))) == active_unknowns(lf) - 1   # deficient
    @test rank(Symmetric(Matrix(mr.matrix))) == active_unknowns(lr)       # repaired
    @test _proj_residual(mr, _one, _INT_ONE) < 1e-6
end

@testset "non-aligned overlay: linear skeleton retained, completeness preserved" begin
    # 3×3 overlay over the middle 2×2 base cells is not nested: the buried centre
    # vertex is NOT reproducible, so it must stay (no dedup) and the constant survives.
    V = overlay(space(_OMEGA_CR; cells=(4, 4), order=2), box((0.25, 0.25), (0.75, 0.75));
                cells=(3, 3), order=3)
    m, l = _gram(V)
    @test count(==(:coverage), l.elimination_source) > 0     # high-order still shed
    @test count(==(:dedup), l.elimination_source) == 0       # vertex kept
    M = Symmetric(Matrix(m.matrix))
    @test rank(M) == active_unknowns(l)
    @test _proj_residual(m, _one, _INT_ONE) < 1e-6
    @test _proj_residual(m, _x1, _INT_X1SQ) < 1e-6
end

@testset "constraint_kind reports coverage and dedup sources" begin
    _, l = _gram(_nested_block())
    kinds = [constraint_kind(l, raw) for raw in eachindex(l.raw_keys)]
    @test :coverage in kinds
    @test :dedup in kinds
    # A high-order (bubble) mode buried in the block is :coverage; the shared centre
    # vertex is :dedup. Spot-check the mode type of each reported source.
    for raw in eachindex(l.raw_keys)
        k = constraint_kind(l, raw)
        axes = l.raw_keys[raw].axes
        k == :coverage && @test any(a -> a.kind == Unfitted._AXIS_SPAN, axes)
        k == :dedup && @test all(a -> a.kind == Unfitted._AXIS_NODE, axes)
    end
end

@testset "recursion: three-level stack reduces every covered level" begin
    V = space(_OMEGA_CR; cells=(4, 4), order=3)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(4, 4), order=3)
    V = overlay(V, box((0.375, 0.375), (0.625, 0.625)); cells=(2, 2), order=3)
    m, l = _gram(V)

    # Both the base (level 1) and the middle overlay (level 2) get coverage
    # eliminations; the top level (3) has nothing above it and is untouched.
    lvl_of(raw) = l.raw_keys[raw].level
    reduced_levels = Set(lvl_of(raw)
                         for raw in eachindex(l.raw_keys)
                         if l.elimination_source[raw] in (:coverage, :dedup))
    @test 1 in reduced_levels
    @test 2 in reduced_levels
    @test !(3 in reduced_levels)

    M = Symmetric(Matrix(m.matrix))
    @test rank(M) == active_unknowns(l)
    @test _proj_residual(m, _one, _INT_ONE) < 1e-6
    @test _proj_residual(m, _x1, _INT_X1SQ) < 1e-6
end

@testset "reduced Poisson solve stays accurate for a smooth solution" begin
    # u = sin(πx)sin(πy); the p=3 overlay resolves the middle, so dropping the covered
    # base high-order must not spoil the smooth solve.
    src = x -> 2pi^2 * sin(pi * x[1]) * sin(pi * x[2])
    exact = x -> sin(pi * x[1]) * sin(pi * x[2])
    bc = dirichlet(0.0; on=boundary(:all))
    Vr = overlay(space(_OMEGA_CR; cells=(8, 8), order=3), box((0.3, 0.3), (0.7, 0.7)); cells=(4, 4),
                 order=3)
    m = prepare(poisson(Vr; source=src, dirichlet=[bc]))
    assemble!(m)
    sol = solve!(m)
    @test isposdef(Symmetric(Matrix(m.matrix)))
    @test l2_error(sol, m, exact; norm=:relative) < 5e-2
end

@testset "covered-mode pruning composes with an immersed physical domain" begin
    # A covered region that is also cut by ∂Ω: the fold removes fictitious cells, cut
    # cells keep moment-fit quadrature, and covered-mode pruning drops only buried
    # high-order. The reduced immersed solve must remain well posed AND complete.
    disk = physical_domain(x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.35; lipschitz=1.0,
                           subcell_length_scale=0.125, max_depth=4)
    V = overlay(space(_OMEGA_CR; cells=(4, 4), order=2, physical=disk),
                box((0.25, 0.25), (0.75, 0.75)); cells=(4, 4), order=3)
    m, l = _gram(V)
    @test count(==(:coverage), l.elimination_source) > 0
    @test isposdef(Symmetric(Matrix(m.matrix)))

    # Quadrature-consistent completeness over Ω: L²-project f ≡ 1 onto the reduced,
    # immersed space and check it reconstructs the constant pointwise inside Ω — the
    # covered cut cells must still carry the partition of unity over Ω ∩ cell.
    c = pinv(Symmetric(Matrix(m.matrix))) * load_vector(m; source=_one)
    proj = Solution(c, m.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))
    for x in (SVector(0.5, 0.5), SVector(0.42, 0.55), SVector(0.58, 0.46))
        @test value(proj, m, x) ≈ 1.0 atol = 1e-6
    end
end

@testset "fictitious fold counts as coverage: no null mode from a folded fine cell" begin
    # Ω is the unit square minus a disk, so the fold deactivates every overlay cell
    # that falls entirely inside the hole. Such a cell carries no material, so it must
    # still count as *covering*: were it to block coverage, the base cells straddling
    # ∂Ω would keep high-order modes the overlay already reproduces exactly on Ω, and
    # the Gram would acquire exact null modes — functions with ‖v_h‖_{L²(Ω)} = 0,
    # supported entirely in the fictitious part. A user mask is the opposite case and
    # still blocks coverage; the two are told apart by `classify_cell` in
    # `_covered_by_level`.
    #
    # The overlay is deliberately wider than the hole, so every base cell the fold
    # touches lies strictly inside it and the check is not entangled with the
    # artificial overlay boundary.
    hole = physical_domain(x -> 0.25 - sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2); lipschitz=1.0,
                           subcell_length_scale=0.015625, max_depth=6)
    V = overlay(space(_OMEGA_CR; cells=(8, 8), order=2, physical=hole),
                box((0.125, 0.125), (0.875, 0.875)); cells=(12, 12), order=2)
    m, l = _gram(V)
    @test diagnostics(m).inactive_cell_counts[2] > 0    # the fold really folds something
    @test count(==(:coverage), l.elimination_source) > 0

    # No exact kernel: the smallest eigenvalue is a genuine small-cut-cell mode, orders
    # of magnitude above the roundoff floor, and the Cholesky factorization succeeds.
    M = Symmetric(Matrix(m.matrix))
    lam = eigvals(M)
    @test count(<=(1e-13 * maximum(lam)), lam) == 0
    @test minimum(lam) > 0
    @test isposdef(M)

    # Still complete over Ω: the reduced space reproduces the constant pointwise.
    c = pinv(M) * load_vector(m; source=_one)
    proj = Solution(c, m.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))
    for x in (SVector(0.2, 0.2), SVector(0.5, 0.85), SVector(0.8, 0.5))
        @test value(proj, m, x) ≈ 1.0 atol = 1e-8
    end
end

@testset "fictitious fold covers a coarse cell whose own centre lies in Ω" begin
    # The hole is small enough to sit strictly inside one base cell, so the base cell
    # carries material at its own centre while the overlay block underneath it contains
    # cells the fold dropped whole. That separation is what the previous test does not
    # reach — there the hole is wide enough that every base cell with a folded overlay
    # cell beneath it has its own centre inside the hole too.
    #
    # `_covered_by_level` decides the case on the *overlay* cells it overlaps, never on
    # the coarse box: its fast rejection samples the centre of each inactive overlay
    # cell, because `:fictitious` means no point of that cell lies in Ω. Reading the
    # coarse cell's centre instead would let a material coarse centre veto coverage
    # while a folded overlay cell sits underneath, and the base cell would keep the
    # high-order modes the overlay already reproduces on Ω — the null mode again.
    hole = physical_domain(x -> 0.06 - sqrt((x[1] - 0.32)^2 + (x[2] - 0.32)^2); lipschitz=1.0,
                           subcell_length_scale=0.0078125, max_depth=6)
    V = overlay(space(_OMEGA_CR; cells=(4, 4), order=2, physical=hole),
                box((0.25, 0.25), (0.75, 0.75)); cells=(16, 16), order=2)
    m, l = _gram(V)
    @test diagnostics(m).inactive_cell_counts[2] > 0     # the fold really folds something

    # Base cell (2, 2) is the one holding the hole. Its centre is in Ω, its overlay
    # block is not entirely active — and it is covered all the same.
    base, fine = Unfitted._problem_levels(m.problem)[1:2]
    cell = CartesianIndex(2, 2)
    @test Unfitted._inside(m.problem.fields[1].space.physical.geometry,
                           Unfitted.center(Unfitted.cell_box(base.mesh, cell)))
    @test Unfitted._covered_by_level(Unfitted.cell_box(base.mesh, cell), fine,
                                     GeometryTolerance(Float64),
                                     Unfitted._Fold(m.problem.fields[1].space))
    @test count(==(:coverage), l.elimination_source) == 8

    # …so no exact kernel, exactly as in the wide-hole case above.
    M = Symmetric(Matrix(m.matrix))
    lam = eigvals(M)
    @test count(<=(1e-13 * maximum(lam)), lam) == 0
    @test minimum(lam) > 0
    @test isposdef(M)
end

# The material test written out, as the dof layer states it independently of the
# oracle under test: a cell is void when the classifier, asked without a memo, reports
# its box entirely outside Ω.
function _void_by_classifier(physical, level, c)
    physical !== nothing && classify_cell(physical, cell_box(level.mesh, c)) === :fictitious
end

# Bytes the material oracle's queries allocate over every cell of `level`, measured
# inside a function so that only the calls themselves are counted.
function _oracle_query_bytes(fold, level, probe)
    bytes = @allocated Unfitted._is_void(fold, probe)
    for c in cell_indices(level.mesh)
        bytes += @allocated Unfitted._is_void(fold, level, c)
        bytes += @allocated Unfitted._is_fictitious(fold, level, c)
    end
    return bytes
end

@testset "the material oracle answers the classifier's verdict on every cell" begin
    # Every site of the dof layer that asks whether an inactive cell holds material —
    # coverage, the burial test's level side, the artificial-face tests, the B-spline
    # admissible region — asks one `_Fold` per layout build, which memoises
    # `classify_cell` per level in a verdict table over the caller's box memo. The
    # question is geometric: a cell is void when its box lies entirely outside Ω,
    # whichever mask switched it off. So the oracle must give the classifier's verdict
    # on every cell — a folded one, a user mask entirely outside Ω, a user mask over
    # material, an active one — on the first ask (the table miss, through the memo)
    # and on every later one (the hit), and under `keep_fictitious`, where no fold runs
    # and every inactive cell is the user's.
    disc(; kw...) = physical_domain(x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.3; lipschitz=1.0,
                                    subcell_length_scale=1 / 64, max_depth=4, kw...)
    # Each level masks one cell entirely outside the disc and one entirely inside it.
    basemask = trues(8, 8)
    basemask[1, 1] = basemask[4, 4] = false
    overlaymask = trues(8, 16)
    overlaymask[1, 16] = overlaymask[7, 8] = false
    masked_stack(physical) = overlay(space(_OMEGA_CR; cells=8, order=2, physical=physical,
                                           active=basemask), box((0.0, 0.0), (0.5, 1.0));
                                     cells=(8, 16), order=2, active=overlaymask)
    void_masked = ((1, CartesianIndex(1, 1)), (2, CartesianIndex(1, 16)))
    material_masked = ((1, CartesianIndex(4, 4)), (2, CartesianIndex(7, 8)))
    probes = (box((1.0, 0.0), (1.125, 0.125)), box((0.45, 0.45), (0.55, 0.55)),
              box((0.75, 0.4), (0.85, 0.6)))
    for physical in (disc(), disc(; alpha=1e-3, keep_fictitious=true))
        # The prepared space and its warm memo, exactly as the layout build gets them.
        _, spaces, caches = Unfitted._prepare_spaces(mass(masked_stack(physical)))
        V, cache = only(spaces), only(caches)
        expected = Dict(level.id => [_void_by_classifier(physical, level, c)
                                     for c in cell_indices(level.mesh)] for level in V.levels)
        inactive(level) = [!Unfitted.is_active(level.mask, c) for c in cell_indices(level.mesh)]
        # Non-vacuity: the fold switched cells off (unless kept), and both kinds of user
        # mask are there, each inactive and of the kind it was placed to be.
        folded = sum(level -> count(inactive(level)), V.levels) - 4
        @test physical.keep_fictitious ? folded == 0 : folded > 0
        for (id, c) in void_masked
            @test inactive(V.levels[id])[c] && expected[id][c]
        end
        for (id, c) in material_masked
            @test inactive(V.levels[id])[c] && !expected[id][c]
        end

        # The void query first, on a fresh oracle, so it takes every miss; then it and
        # the inactive-and-void query again, all hits.
        fold = Unfitted._Fold(V, cache)
        @test fold isa Unfitted._Fold{2,Float64,typeof(V.physical)}
        @test !any(id -> isassigned(fold.cells, id), eachindex(fold.cells))
        for pass in 1:2, level in V.levels
            cells = cell_indices(level.mesh)
            @test [Unfitted._is_void(fold, level, c) for c in cells] == expected[level.id]
            @test [Unfitted._is_fictitious(fold, level, c) for c in cells] ==
                  inactive(level) .& expected[level.id]
        end
        # Every verdict the tables hold is the one the box memo holds for that box.
        for level in V.levels
            memo = [cache[(b.lower, b.upper)]
                    for b in (cell_box(level.mesh, c) for c in cell_indices(level.mesh))]
            @test fold.cells[level.id] == [m === :fictitious ? 0x01 : 0x02 for m in memo]
        end
        for b in probes
            @test Unfitted._is_void(fold, b) == (classify_cell(physical, b) === :fictitious)
        end
    end

    # Without a physical domain every cell holds material: each query is a constant,
    # nothing is classified, no table is allocated and no query allocates.
    V = masked_stack(nothing)
    fold = Unfitted._Fold(V)
    @test fold isa Unfitted._Fold{2,Float64,Nothing}
    for level in V.levels
        cells = cell_indices(level.mesh)
        @test !any(c -> Unfitted._is_void(fold, level, c) || Unfitted._is_fictitious(fold, level, c),
                   cells)
        _oracle_query_bytes(fold, level, probes[1])                 # compile once
        @test _oracle_query_bytes(fold, level, probes[1]) == 0
    end
    @test !any(b -> Unfitted._is_void(fold, b), probes)
    Unfitted.build_coverage(V, GeometryTolerance(Float64), fold)
    @test !any(id -> isassigned(fold.cells, id), eachindex(fold.cells)) && isempty(fold.boxes)
end

@testset "the coverage memo answers what a fresh coverage test does" begin
    # `build_coverage` keeps every per-cover answer it computes in the material oracle's
    # memo (`_Fold.covered_by`), and the burial test reads them back through
    # `_covered_by!` instead of asking
    # `_covered_by_level` again. On an immersed three-level stack — a middle overlay the
    # fold and a user mask over material both switch cells off, under an unmasked inner
    # one — every flag, every stored answer and every answer asked through the memo,
    # first as a miss and then as a hit, must be the one `_covered_by_level` computes on
    # a fresh oracle with no memo at all.
    disc = physical_domain(x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.3; lipschitz=1.0,
                           subcell_length_scale=1 / 64, max_depth=4)
    # The user hole lies inside the disc, under base cells 5…6 along each axis.
    hole = trues(12, 12)
    hole[8:9, 8:9] .= false
    stack = overlay(overlay(space(_OMEGA_CR; cells=8, order=2, physical=disc),
                            box((0.125, 0.125), (0.875, 0.875)); cells=12, order=2, active=hole),
                    box((0.375, 0.375), (0.625, 0.625)); cells=4, order=2)
    _, spaces, caches = Unfitted._prepare_spaces(mass(stack))
    V, cache = only(spaces), only(caches)
    tol = GeometryTolerance(Float64)
    fold = Unfitted._Fold(V, cache)
    coverage = Unfitted.build_coverage(V, tol, fold)
    reference = Unfitted._Fold(V)
    fresh(level, c, k) = Unfitted._covered_by_level(cell_box(level.mesh, c), k, tol, reference)
    middle, inner = V.levels[2], V.levels[3]
    # Non-vacuity: the middle cover is masked (by the fold and by the user), the inner
    # one is not.
    @test middle.mask !== nothing && count(!, middle.mask.on) > 4
    @test inner.mask === nothing
    stored = UInt8[]
    for level in V.levels
        cells = cell_indices(level.mesh)
        above = [k for k in V.levels if k.id > level.id]
        wanted = Unfitted._coverage_cells(level)
        @test coverage.covered[level.id] ==
              [!isempty(above) && wanted[c] && any(k -> fresh(level, c, k), above) for c in cells]
        for k in above
            # A masked cover's answers are stored as they were computed, and nowhere else;
            # an unmasked one's are never stored.
            if k.mask === nothing
                @test !isassigned(fold.covered_by, level.id, k.id)
            elseif isassigned(fold.covered_by, level.id, k.id)
                table = fold.covered_by[level.id, k.id]
                @test all(c -> table[c] == 0x00 || (table[c] == 0x01) == fresh(level, c, k), cells)
                append!(stored, table)
            end
            for pass in 1:2
                @test [Unfitted._covered_by!(fold, level, c, k, tol) for c in cells] ==
                      [fresh(level, c, k) for c in cells]
            end
        end
    end
    # Non-vacuity: `build_coverage` stored both verdicts, and left cells unasked.
    @test 0x00 in stored && 0x01 in stored && 0x02 in stored
end

# `_nested_over`'s definition written out in the test, independently of the code under
# test: a scan of every node of the outer axis.
function _nested_by_scan(inner, outer, tol)
    for d in eachindex(inner.mesh.axes)
        lo, hi = outer.mesh.domain.lower[d], outer.mesh.domain.upper[d]
        for x in inner.mesh.axes[d]
            (lo - tol.contain <= x <= hi + tol.contain) || continue
            any(y -> abs(x - y) <= tol.merge, outer.mesh.axes[d]) || return false
        end
    end
    return true
end

@testset "nesting is decided at the tolerance edges as a scan of the axis decides it" begin
    # `_nested_over` asks, of every inner node inside the outer box widened by
    # `tol.contain`, whether some outer node lies within `tol.merge` of it. The probes
    # sit exactly at the tolerance edges — `merge` either side of an outer node,
    # `contain` past either end of the outer box — and one rounding step either side of
    # each edge. All are exact binary fractions, so the edges are hit exactly.
    tol = GeometryTolerance{Float64}(2.0^-6, 2.0^-4, 2.0^-6)
    m, c = tol.merge, tol.contain
    outer = space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=1).levels[1]
    axis = outer.mesh.axes[1]
    @test axis == [0.0, 0.25, 0.5, 0.75, 1.0]
    probes = Float64[]
    for y in axis, δ in (0.0, m, -m), step in (identity, prevfloat, nextfloat)
        push!(probes, step(y + δ))
    end
    for edge in (-c, 1 + c), step in (identity, prevfloat, nextfloat)
        push!(probes, step(edge))
    end
    append!(probes, (-0.0, 0.125, 0.3, -1.0, 2.0))

    # Each probe as the one interior node of a one-cell inner level, whose two
    # axis-1 nodes are the probe and an end of the outer box (a one-cell axis is exactly
    # its two corners).
    probe_level(x) = space(x < 1 ? box((x, 0.0), (1.0, 1.0)) : box((0.0, 0.0), (x, 1.0)); cells=1,
                           order=1).levels[1]
    nested = [Unfitted._nested_over(probe_level(x), outer, tol) for x in probes]
    @test nested == [_nested_by_scan(probe_level(x), outer, tol) for x in probes]
    @test any(nested) && !all(nested)
    # Both answers occur at the `merge` edge itself and one rounding step past it.
    @test Unfitted._nested_over(probe_level(0.25 + m), outer, tol)
    @test Unfitted._nested_over(probe_level(0.25 - m), outer, tol)
    @test !Unfitted._nested_over(probe_level(nextfloat(0.25 + m)), outer, tol)
    @test !Unfitted._nested_over(probe_level(prevfloat(0.25 - m)), outer, tol)
    @test !Unfitted._nested_over(probe_level(-c), outer, tol)            # inside the widened box
    @test Unfitted._nested_over(probe_level(prevfloat(-c)), outer, tol)  # outside it: not asked
end

# Ω = {x₁ ≤ 0.9} inside [-1, 2]², so a 3×3 base folds its whole right column away as
# fictitious and the fold face sits at x = 1. The cover is a sub-box overlay, never a
# `ladder`: a ladder spans the level below, `is_inside` passes, and the fold is settled
# by the mask branch of `_covered_by_level` — the configuration below never arises.
_HALF_PLANE_CR = box((-1.0, -1.0), (2.0, 2.0))
function _half_plane_cr()
    physical_domain(x -> x[1] - 0.9; lipschitz=1.0, subcell_length_scale=0.125, max_depth=6)
end
function _fold_cover(xhi, p)
    base = space(_HALF_PLANE_CR; cells=3, order=p, physical=_half_plane_cr())
    return overlay(base, box((0.0, -1.0), (xhi, 2.0)); cells=(round(Int, 2xhi), 6), order=p)
end

@testset "a cover ending inside a folded coarse cell is still a cover" begin
    # The two extents are the two sides of the rule, and both are asserted.
    #
    #   xhi = 1.5 — the cover's face lands *inside* the folded base cell [1, 2]. That
    #     cell is fictitious, so the base's modes on the x = 1 fold face stay free
    #     (`_has_overlay_constraint`); on the cover the same face is interior, so
    #     the cover's copies are free too; and under nesting the cover reproduces them
    #     exactly on Ω. Both copies used to survive — the burial test asked
    #     `is_inside(cell, cover.domain)` of the fictitious cell, which a cover ending
    #     inside it fails — and the Gram matrix picked up 10 exact null modes at p = 3
    #     (4 vertex hats plus 3 × 2 face modes on x = 1) and 4 at p = 1.
    #   xhi = 1.0 — the cover stops *at* the fold face. Its own boundary cells there are
    #     cut cells, hence active, hence clamped: the cover's trace vanishes at x = 1
    #     and it cannot carry the base's hat. The mode must be retained, and is.
    #
    # `rank == active_unknowns` is the discriminating assertion. The pointwise
    # reproduction below is the completeness guard only and cannot stand in for it: the
    # null modes live entirely in the fictitious part, a pseudo-inverse projection never
    # touches them, and every reproduction assertion here passes on the singular
    # operator too. What it does guard is the other direction — that the repair prunes
    # nothing the space needs.
    for (p, xhi, dedup, unknowns) in
        ((3, 1.5, 4, 154), (3, 1.0, 0, 145), (1, 1.5, 4, 22), (1, 1.0, 0, 19))
        m, l = _gram(_fold_cover(xhi, p))
        mu, _ = _gram(_fold_cover(xhi, p); prune=false)
        M = Symmetric(Matrix(m.matrix))
        @test count(==(:dedup), l.elimination_source) == dedup
        @test active_unknowns(l) == unknowns
        @test rank(M) == active_unknowns(l)
        @test isposdef(M)
        # And nothing real was deleted: the cover is nested and same-order, so the
        # pruned space must still span the unpruned one (the pattern at "a mode covered
        # by two different levels is retained").
        @test active_unknowns(l) == rank(Matrix(mu.matrix))

        # Still complete to order p over Ω.
        c = pinv(M) * load_vector(m; source=x -> x[1]^p)
        proj = Solution(c, m.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))
        for x in (SVector(-0.5, 0.5), SVector(0.5, -0.3), SVector(0.8, 1.2))
            @test value(proj, m, x) ≈ x[1]^p atol = 1e-8
        end
    end
end

@testset "covered-mode pruning in 3D (edge, face, and interior modes)" begin
    # Aligned same-order nested overlay over the middle 2×2×2 base block: the buried
    # centre vertex is deduped and the eight covered cells shed their edge/face/interior
    # modes. Exercises the 3-axis incidence and coverage that no 2D config reaches.
    omega3 = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = overlay(space(omega3; cells=(4, 4, 4), order=2),
                box((0.25, 0.25, 0.25), (0.75, 0.75, 0.75)); cells=(4, 4, 4), order=2)
    m, l = _gram(V)
    @test count(==(:coverage), l.elimination_source) > 0   # buried edge/face/interior modes
    @test count(==(:dedup), l.elimination_source) >= 1     # the buried centre vertex
    M = Symmetric(Matrix(m.matrix))
    @test rank(M) == active_unknowns(l)
    @test isposdef(M)
    @test _proj_residual(m, _one, _INT_ONE) < 1e-6         # ∫1 over the unit cube = 1
    @test _proj_residual(m, _x1, _INT_X1SQ) < 1e-6         # ∫x₁² over the unit cube = 1/3
end

@testset "seam of abutting overlays: straddled base cell keeps its high-order modes" begin
    # Overlay A covers base column 2 fully and column 3 up to x=0.625; overlay B covers
    # the rest of column 3. The column-3 cells are covered only by the A ∪ B union — the
    # seam at x=0.625 falls inside them — so build_coverage (single-covering-level)
    # leaves them uncovered and their high-order survives, while column-2 (covered by A
    # alone) is reduced. The overlay meshes are non-aligned with the base so no
    # coincident-vertex redundancy clouds the seam behaviour under test.
    V = space(_OMEGA_CR; cells=(4, 4), order=2)
    V = overlay(V, box((0.25, 0.25), (0.625, 0.75)); cells=(5, 3), order=3)
    V = overlay(V, box((0.625, 0.25), (0.75, 0.75)); cells=(2, 3), order=3)
    m, l = _gram(V)
    @test count(==(:coverage), l.elimination_source) > 0     # column-2 cells reduced by A

    # A base span-mode incident to the straddled cell (3,2) is NOT coverage-eliminated.
    straddled = [raw
                 for raw in eachindex(l.raw_keys)
                 if l.raw_keys[raw].level == 1 &&
                        any(a -> a.kind == Unfitted._AXIS_SPAN, l.raw_keys[raw].axes) &&
                        CartesianIndex(3, 2) in Unfitted._incident_cells(l.raw_keys[raw], (4, 4))]
    @test !isempty(straddled)
    @test all(raw -> l.elimination_source[raw] != :coverage, straddled)

    M = Symmetric(Matrix(m.matrix))
    @test rank(M) == active_unknowns(l)
    @test isposdef(M)
    @test _proj_residual(m, _one, _INT_ONE) < 1e-6
    @test _proj_residual(m, _x1, _INT_X1SQ) < 1e-6
end

# Split cover: two overlays each take half of a base mode's incidence stencil, so
# the mode is buried under their *union* but under neither one alone. Each overlay
# is clamped to zero on the face they share, so at that seam nothing carries what
# the elimination would remove — the mode has to be retained.
#
# `cov[ci]` only records that *some* higher level covers cell `ci`, which is why
# this needs the same single-covering-level test the dedup half already applies.
function _split_cover()
    V = space(_OMEGA_CR; cells=(4, 4), order=3)
    V = overlay(V, box((0.0, 0.0), (0.5, 0.5)); cells=(4, 4), order=3)
    return overlay(V, box((0.5, 0.0), (1.0, 0.5)); cells=(4, 4), order=3)
end

@testset "a mode covered by two different levels is retained" begin
    model, _ = _gram(_split_cover())
    reduced = size(Matrix(model.matrix), 1)
    unreduced, _ = _gram(_split_cover(); prune=false)
    # The reduced space must still span the unreduced one: pruning may remove
    # redundancy and nothing else. Before the single-covering-level test this
    # dropped modes on the seam between the two overlays.
    @test reduced == rank(Matrix(unreduced.matrix))
end
