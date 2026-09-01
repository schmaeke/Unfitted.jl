# Regression tests for order reduction in covered regions (the coverage constraint
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
function _gram(V)
    model = prepare(mass(V))
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
function _nested_block(ro)
    overlay(space(_OMEGA_CR; cells=(4, 4), order=2, reduce_order=ro),
            box((0.25, 0.25), (0.75, 0.75)); cells=(4, 4), order=3)
end

@testset "reduce_order=false is a no-op" begin
    _, lf = _gram(_nested_block(false))
    @test all(s -> s in (:free, :overlay), lf.elimination_source)
    @test count(==(:coverage), lf.elimination_source) == 0
    @test count(==(:dedup), lf.elimination_source) == 0
end

@testset "order reduction: nested block sheds high-order, stays complete and full rank" begin
    mf, lf = _gram(_nested_block(false))
    mr, lr = _gram(_nested_block(true))

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
    m = prepare(mass(_nested_block(true)))
    counts = diagnostics(m).reduced_mode_counts
    @test length(counts) == 2          # base + overlay
    @test counts[1] == 9               # base (level 1): 8 coverage + 1 dedup
    @test counts[2] == 0               # overlay (level 2, top of stack): untouched
end

@testset "coincident same-order cell: rank deficiency removed" begin
    # Overlay is one p=2 cell coincident with the interior base cell (2,2). Its only
    # surviving mode duplicates the base cell's bubble → the full space is rank
    # deficient by one; order reduction drops that buried bubble.
    full = overlay(space(_OMEGA_CR; cells=(4, 4), order=2, reduce_order=false),
                   box((0.25, 0.25), (0.5, 0.5)); cells=(1, 1), order=2)
    red = overlay(space(_OMEGA_CR; cells=(4, 4), order=2, reduce_order=true),
                  box((0.25, 0.25), (0.5, 0.5)); cells=(1, 1), order=2)

    mf, lf = _gram(full)
    mr, lr = _gram(red)
    @test rank(Symmetric(Matrix(mf.matrix))) == active_unknowns(lf) - 1   # deficient
    @test rank(Symmetric(Matrix(mr.matrix))) == active_unknowns(lr)       # repaired
    @test _proj_residual(mr, _one, _INT_ONE) < 1e-6
end

@testset "non-aligned overlay: linear skeleton retained, completeness preserved" begin
    # 3×3 overlay over the middle 2×2 base cells is not nested: the buried centre
    # vertex is NOT reproducible, so it must stay (no dedup) and the constant survives.
    V = overlay(space(_OMEGA_CR; cells=(4, 4), order=2, reduce_order=true),
                box((0.25, 0.25), (0.75, 0.75)); cells=(3, 3), order=3)
    m, l = _gram(V)
    @test count(==(:coverage), l.elimination_source) > 0     # high-order still shed
    @test count(==(:dedup), l.elimination_source) == 0       # vertex kept
    M = Symmetric(Matrix(m.matrix))
    @test rank(M) == active_unknowns(l)
    @test _proj_residual(m, _one, _INT_ONE) < 1e-6
    @test _proj_residual(m, _x1, _INT_X1SQ) < 1e-6
end

@testset "constraint_kind reports coverage and dedup sources" begin
    _, l = _gram(_nested_block(true))
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
    V = space(_OMEGA_CR; cells=(4, 4), order=3, reduce_order=true)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(4, 4), order=3, reduce_order=true)
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
    Vr = overlay(space(_OMEGA_CR; cells=(8, 8), order=3, reduce_order=true),
                 box((0.3, 0.3), (0.7, 0.7)); cells=(4, 4), order=3)
    m = prepare(poisson(Vr; source=src, dirichlet=[bc]))
    assemble!(m)
    sol = solve!(m)
    @test isposdef(Symmetric(Matrix(m.matrix)))
    @test l2_error(sol, m, exact; norm=:relative) < 5e-2
end

@testset "order reduction composes with an immersed physical domain" begin
    # A covered region that is also cut by ∂Ω: the fold removes fictitious cells, cut
    # cells keep moment-fit quadrature, and order reduction drops only buried
    # high-order. The reduced immersed solve must remain well posed AND complete.
    disk = physical_domain(x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.35; lipschitz=1.0,
                           subcell_length_scale=0.125, max_depth=4)
    V = overlay(space(_OMEGA_CR; cells=(4, 4), order=2, reduce_order=true, physical=disk),
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
    V = overlay(space(_OMEGA_CR; cells=(8, 8), order=2, reduce_order=true, physical=hole),
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
    V = overlay(space(_OMEGA_CR; cells=(4, 4), order=2, reduce_order=true, physical=hole),
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
                                     GeometryTolerance(Float64), m.problem.fields[1].space.physical,
                                     Unfitted._ClassifyCache{2,Float64}())
    @test count(==(:coverage), l.elimination_source) == 8

    # …so no exact kernel, exactly as in the wide-hole case above.
    M = Symmetric(Matrix(m.matrix))
    lam = eigvals(M)
    @test count(<=(1e-13 * maximum(lam)), lam) == 0
    @test minimum(lam) > 0
    @test isposdef(M)
end

@testset "order reduction in 3D (edge, face, and interior modes)" begin
    # Aligned same-order nested overlay over the middle 2×2×2 base block: the buried
    # centre vertex is deduped and the eight covered cells shed their edge/face/interior
    # modes. Exercises the 3-axis incidence and coverage that no 2D config reaches.
    omega3 = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = overlay(space(omega3; cells=(4, 4, 4), order=2, reduce_order=true),
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
    V = space(_OMEGA_CR; cells=(4, 4), order=2, reduce_order=true)
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
function _split_cover(ro)
    V = space(_OMEGA_CR; cells=(4, 4), order=3, reduce_order=ro)
    V = overlay(V, box((0.0, 0.0), (0.5, 0.5)); cells=(4, 4), order=3, reduce_order=ro)
    return overlay(V, box((0.5, 0.0), (1.0, 0.5)); cells=(4, 4), order=3, reduce_order=ro)
end

@testset "a mode covered by two different levels is retained" begin
    model, _ = _gram(_split_cover(true))
    reduced = size(Matrix(model.matrix), 1)
    unreduced, _ = _gram(_split_cover(false))
    # The reduced space must still span the unreduced one: pruning may remove
    # redundancy and nothing else. Before the single-covering-level test this
    # dropped modes on the seam between the two overlays.
    @test reduced == rank(Matrix(unreduced.matrix))
end
