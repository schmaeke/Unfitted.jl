using BasicBSpline
using StaticArrays
using LinearAlgebra: Symmetric, eigvals, isposdef, norm, pinv, rank

# Smoke tests for the BasicBSpline extension. Mirror the integrated
# Legendre test surface (`test_basis.jl`) where the interface is shared,
# and add B-spline-specific tests for the dof-key sharing and mask
# handling.
#
# Several testsets below reuse `_gram` / `_proj_residual` and the `_one` / `_x1` /
# `_INT_*` constants from `test_coverage_reduction.jl`, which `runtests.jl`
# includes first; this file does not stand alone.

@testset "BSpline extension: a per-cell order is refused, not ignored" begin
    # Per-cell polynomial order is a basis-family capability. A B-spline degree
    # is a type parameter of the whole-axis knot vector, the clamped end
    # multiplicity is p + 1, the 1D dimension is `cells + p`, and a function's
    # support is p + 1 cells wide — so no set of functions belongs to one cell
    # for a per-cell order to name, and there is no shared entity for the
    # minimum rule to act on. The request must raise rather than be silently
    # dropped.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    @test_throws ArgumentError space(Ω; cells=4, order=[2 + (i + j) % 2 for i in 1:4, j in 1:4],
                                     basis=bspline())
    V = space(Ω; cells=4, order=2, basis=bspline())
    @test_throws ArgumentError overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=2,
                                       order=[2 3; 3 2], basis=bspline())
    # The trait triggers on non-uniformity, not on the shape: a per-cell field
    # that happens to be flat is a perfectly well defined B-spline level.
    W = space(Ω; cells=4, order=fill(3, 4, 4), basis=bspline())
    @test length(W.levels[1].orders.palette) == 1
    @test nominal_order(W.levels[1]) == (3, 3)
    @test Unfitted.dof_layout(W).active_count ==
          Unfitted.dof_layout(space(Ω; cells=4, order=3, basis=bspline())).active_count
    # And `elevate` refuses through the same trait.
    @test_throws ArgumentError elevate(V, 1 => [1 2 3 1; 1 1 1 1; 1 1 1 1; 1 1 1 1])
end

@testset "BSpline extension: basis interface" begin
    fam_marker = bspline()
    @test fam_marker isa Unfitted.BasisFamily

    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=3, basis=bspline())
    fam = V.levels[1].basis
    @test Unfitted.basis_name(fam) == :bspline
    @test Unfitted.local_basis_count(fam, (3, 3)) == 16
    @test Unfitted.recommended_quadrature_order(fam, (2, 3)) == (3, 4)
    # `local_basis_indices` / `local_basis_count` come from the `::BasisFamily`
    # tensor defaults: an open-knot span carries the full ∏_d {0, …, p_d} set,
    # so the family adds nothing of its own here.
    @test Unfitted.local_basis_indices(fam, (1, 2))[1] == CartesianIndex(0, 0)
    @test Unfitted.local_basis_indices(fam, (1, 2)) ==
          Unfitted.local_basis_indices(fam, (1, 2), :tensor)
    @test Unfitted.local_basis_count(fam, (3, 3), :tensor) == 16
    # `dim` per axis = cells + p (open knot, no junctions, m=0).
    @test BasicBSpline.dim.(fam.spaces) == (7, 7)

    # The family declares only `:tensor`, so the core mode check — not a
    # hand-written guard in the extension — rejects integrated Legendre's
    # `:trunk` for it, at `space` time.
    @test Unfitted._supported_modes(fam) == (:tensor,)
    @test_throws ArgumentError space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=3, basis=bspline(),
                                     mode=:trunk)
    @test_throws ArgumentError Unfitted.local_basis_indices(fam, (3, 3), :trunk)
end

@testset "BSpline extension: continuity validation" begin
    # Negative continuity is rejected at factory time, and so is a symbol that is
    # not `:maximal`.
    @test_throws ArgumentError bspline(continuity=-1)
    @test_throws ArgumentError bspline(continuity=:smooth)
    # continuity > p − 1 is rejected at level construction.
    @test_throws ArgumentError space(box((0.0,), (1.0,)); cells=4, order=2,
                                     basis=bspline(continuity=2))
    # Too-thin level for a *clamped* continuity is rejected: eliminating m + 1
    # functions per side must leave something to pivot onto.
    @test_throws ArgumentError space(box((0.0,), (1.0,)); cells=1, order=3,
                                     basis=bspline(continuity=1))
    # `:maximal` carries no thickness requirement, because it clamps nothing. A
    # level too thin to hold a whole support contributes no functions, which is a
    # statement about the spline space (minimal support needs p + 1 cells) rather
    # than a misuse — an adaptive loop that wakes a small cluster must reach it.
    Vthin = space(box((0.0,), (1.0,)); cells=1, order=2, basis=bspline())
    @test Vthin.levels[1].basis isa Unfitted.BasisFamily
    # Valid C¹ family on a degree-3 mesh.
    V = space(box((0.0,), (1.0,)); cells=8, order=3, basis=bspline(continuity=1))
    @test V.levels[1].basis isa Unfitted.BasisFamily
    # An integer request that already equals p − 1 on every axis is `:maximal` by
    # another name: the same space, reached by selection rather than by a
    # constraint cascade. The family normalises the one spelling into the other,
    # so the two levels are indistinguishable afterwards.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    B = box((0.25, 0.25), (0.75, 0.75))
    max_family = space(Ω; cells=6, order=3, basis=bspline()).levels[1].basis
    int_family = space(Ω; cells=6, order=3, basis=bspline(continuity=2)).levels[1].basis
    @test max_family.continuity == int_family.continuity
    free(b) = Unfitted.active_unknowns(prepare(mass(overlay(space(Ω; cells=16, order=3,
                                                                  basis=bspline()), B; cells=8,
                                                            order=3, basis=b))).dofs)
    @test free(bspline()) == free(bspline(continuity=2))
end

@testset "BSpline extension: support selection — the dof ledger is the closed form" begin
    # The two halves of the selection rule, as counts. A level whose faces are all
    # PHYSICAL keeps every function it has — nothing is artificial, so nothing has
    # to vanish — and a level whose faces are all ARTIFICIAL keeps exactly the ones
    # whose whole `(p + 1)`-cell support fits inside it.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    free(V) = Unfitted.active_unknowns(prepare(mass(V)).dofs)
    for p in 1:4, n in (4, 8)
        @test free(space(Ω; cells=n, order=p, basis=bspline())) == (n + p)^2
    end
    # An overlay on a base fine enough that nothing is nested, so the count below is
    # the overlay's own contribution and not a dedup in disguise.
    for p in 1:4, n in (2, 3, 5, 7, 11)
        base = space(Ω; cells=16, order=p, basis=bspline())
        V = overlay(base, box((0.25, 0.25), (0.75, 0.75)); cells=n, order=p, basis=bspline())
        @test free(V) - free(base) == max(0, n - p)^2
    end
    # Arbitrary mask geometries are a per-function containment test, so they neither
    # need the mask to be separable nor produce a single linear-constraint pivot —
    # which is what keeps assembly on its cheap single-target emission path and makes
    # the reconstruction defect `dof_value` used to carry unreachable here.
    for p in 2:3
        for mask in (let m = trues(12, 12)
                         m[7:12, 7:12] .= false
                         m
                     end,                                        # L-shape
                     let m = trues(12, 12)
                         m[4:8, 4:8] .= false
                         m
                     end,                                        # hole
                     BitArray([i + j <= 12 for i in 1:12, j in 1:12]))   # staircase
            V = space(Ω; cells=12, order=p, basis=bspline(), active=mask)
            model = prepare(mass(V))
            @test !Unfitted.has_linear_constraints(model.dofs)
            A = Symmetric(Matrix(assemble_matrix(model, mass_block(field(:u, V)))))
            @test rank(A) == Unfitted.active_unknowns(model.dofs)
            @test isposdef(A)
        end
    end
end

@testset "BSpline extension: the superposition is smooth, not merely each level" begin
    # The property the family exists for, measured where it can fail: across the
    # ARTIFICIAL boundary of an overlay, which is the one place a superposition can
    # lose the smoothness its levels have. Probe the jump in ∂u/∂x₁ across the
    # overlay face at x₁ = ¼ as the probe closes in. The measured quantity is
    # `|∂₁u(¼ − ε) − ∂₁u(¼ + ε)|`, which for a C¹ field is `2ε·|∂₁₁u| + O(ε²)` and
    # therefore falls by ten for every decade in ε; for a merely C⁰ field it tends
    # to the jump itself and stops falling.
    # Degree 2, and the last decade before round-off: the smooth term `2ε·∂₁₁u`
    # dominates a genuine jump until ε is small enough, so a ratio read at
    # ε = 1e-3 says nothing about either case. Measured across the four decades,
    # `:maximal` falls by ten at every one of them while `continuity = 0` flattens
    # out — the jump itself — between the third and the fourth.
    source(x) = 2.0 * pi^2 * sin(pi * x[1]) * sin(pi * x[2])
    function decade_ratio(basis)
        V = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=2, basis=basis)
        V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=8, order=2, basis=basis)
        model = prepare(poisson(V; source=source, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
        solution = solve!(model)
        jump(ε) = maximum(abs(field_gradient(solution, model, (0.25 - ε, y))[1] -
                              field_gradient(solution, model, (0.25 + ε, y))[1])
                          for y in (0.35, 0.5, 0.65))
        return jump(1.0e-4) / jump(1.0e-5)
    end
    # `:maximal` is C¹ or better across the face, so the ratio is ten.
    @test decade_ratio(bspline()) ≈ 10.0 rtol = 1.0e-3
    # `continuity = 0` keeps the overlay's boundary functions, so the normal
    # derivative genuinely jumps and the ratio collapses towards one.
    @test decade_ratio(bspline(; continuity=0)) < 1.5
    # The default family is C⁰ across the same face, for the same reason.
    @test decade_ratio(IntegratedLegendre()) < 1.5
end

@testset "support_extension dilates by the family's support radius" begin
    Ω = box((0.0, 0.0), (1.0, 1.0))
    marked = [CartesianIndex(8, 8), CartesianIndex(8, 9), CartesianIndex(9, 8),
              CartesianIndex(9, 9)]
    # Integrated Legendre's modes are cell-local or shared across the face they sit
    # on, so its radius is zero and the dilation is the identity.
    V = space(Ω; cells=16, order=3)
    @test support_extension(V, marked; level=1) ==
          Unfitted._normalize_mask(marked, V.levels[1].mesh).on
    # A degree-p B-spline reaches p cells, so a 2×2 block becomes (2 + 2p)².
    for p in 1:3
        W = space(Ω; cells=16, order=p, basis=bspline())
        extended = support_extension(W, marked; level=1)
        @test count(extended) == (2 + 2p)^2
        @test all(extended[c] for c in marked)
    end
    # And the dilation is what turns a masked region that carries no unknowns into
    # one that does. A 2×2 active block is two cells wide, below the `p + 1` cells a
    # degree-2 support needs, so such a level carries nothing at all; the extension
    # is 6×6 and carries `(6 − 2)² = 16`.
    free(cells) = Unfitted.active_unknowns(prepare(mass(space(Ω; cells=16, order=2, basis=bspline(),
                                                              active=cells))).dofs)
    @test free(marked) == 0
    @test free(support_extension(space(Ω; cells=16, order=2, basis=bspline()), marked; level=1)) ==
          16
    # The dilation clips to the level's own grid rather than widening its box.
    edge = space(Ω; cells=8, order=3, basis=bspline())
    @test size(support_extension(edge, [CartesianIndex(1, 1)]; level=1)) == (8, 8)
end

@testset "dilate: the geometric primitive support_extension is built on" begin
    Ω = box((0.0, 0.0), (1.0, 1.0))
    V = space(Ω; cells=16, order=3, basis=bspline())
    one_cell = [CartesianIndex(8, 8)]

    # A box, not a cross: one cell grows to ∏_d (2·by[d] + 1).
    for by in (0, 1, 2, 3)
        @test count(dilate(V, one_cell; level=1, by=by)) == (2by + 1)^2
    end
    # Per-axis widths, and the array form agrees with the Space form.
    @test count(dilate(V, one_cell; level=1, by=(1, 3))) == 3 * 7
    seed = falses(16, 16)
    seed[8, 8] = true
    @test dilate(seed, (1, 3)) == dilate(V, one_cell; level=1, by=(1, 3))
    # `dilate` accepts a plain `Array{Bool}` as well as a `BitArray`.
    @test dilate(Array(seed), 2) == dilate(seed, 2)

    # Monotone in the width: dilating by `a` then `b` is dilating once by `a + b`.
    @test dilate(dilate(seed, 2), 3) == dilate(seed, 5)
    # And `by = 0` is the identity.
    @test dilate(seed, 0) == seed

    # Clipped to the grid, never widening the box.
    corner = falses(16, 16)
    corner[1, 1] = true
    @test count(dilate(corner, 3)) == 4 * 4
    @test size(dilate(corner, 3)) == (16, 16)

    # `support_extension` IS `dilate` at the family's own radius — asserted rather
    # than assumed, because the two are separate public names.
    marked = [CartesianIndex(8, 8), CartesianIndex(9, 9)]
    for p in 1:3
        W = space(Ω; cells=16, order=p, basis=bspline())
        @test support_extension(W, marked; level=1) == dilate(W, marked; level=1, by=p)
    end
    # Integrated Legendre reaches no further than its own cell, so the two differ:
    # its support extension is the identity while `dilate` still grows what it is
    # told to.
    L = space(Ω; cells=16, order=3)
    @test support_extension(L, marked; level=1) == dilate(L, marked; level=1, by=0)
    @test count(dilate(L, marked; level=1, by=2)) > count(support_extension(L, marked; level=1))

    # Refusals, so a wrong `by` is loud rather than silently clamped.
    @test_throws ArgumentError dilate(V, one_cell; level=1, by=-1)
    @test_throws ArgumentError dilate(V, one_cell; level=1, by=(1, -1))
    @test_throws ArgumentError dilate(V, one_cell; level=1, by=1.5)
    # The `Model` spelling reads the pre-fold space, so it agrees with the space's.
    model = prepare(mass(V))
    @test dilate(model, one_cell; level=1, by=2) == dilate(V, one_cell; level=1, by=2)
end

@testset "diagnostics: a level reports what it enumerated and what survived" begin
    # The pair is what separates the two ways of contributing nothing, which is
    # the question a too-thin spline overlay raises and which no single number
    # answers. A dormant level enumerated nothing; a too-thin one enumerated
    # functions and lost every one of them to its artificial boundary.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    BC = [dirichlet(0.0; on=boundary(:all))]
    report_of(V) =
        let model = prepare(poisson(V; source=1.0, dirichlet=BC))
            diagnostics(model, solve!(model))
        end
    levels_of(V) = report_of(V).levels

    base = space(Ω; cells=16, order=3, basis=bspline())
    thin = overlay(base, box((0.25, 0.25), (0.75, 0.75)); cells=3, order=3, basis=bspline())
    thick = overlay(base, box((0.25, 0.25), (0.75, 0.75)); cells=8, order=3, basis=bspline())
    dormant = overlay(base, box((0.25, 0.25), (0.75, 0.75)); cells=8, order=3, basis=bspline(),
                      active=CartesianIndex{2}[])

    # Too thin: it enumerated (3 + 3)² = 36 functions and kept none.
    thin_level = levels_of(thin)[2]
    @test thin_level.raw_functions == 36
    @test thin_level.active_functions == 0
    # Dormant: nothing enumerated at all, which is not the same thing.
    dormant_level = levels_of(dormant)[2]
    @test dormant_level.raw_functions == 0
    @test dormant_level.active_functions == 0
    # Thick enough: the closed form, `(n − p)² = 25`.
    thick_level = levels_of(thick)[2]
    @test thick_level.active_functions == 25
    @test thick_level.raw_functions == (8 + 3)^2

    # On a scalar field a function carries one unknown, so the counts sum to the
    # total. Read from ONE report: a `Solution` is pinned to the model that
    # produced it, so pairing one model's diagnostics with another model's
    # solution is not merely wasteful, it is rejected.
    thick_report = report_of(thick)
    report = thick_report.levels
    @test sum(l -> l.active_functions, report) == thick_report.active_unknowns
    # A vector field carries one unknown per component per function, so the
    # functions stay the same and the unknowns scale — which is why the report
    # counts functions.
    u = field(:u, thick; components=2)
    vec_model = prepare(poisson(u; source=SVector(1.0, 1.0),
                                dirichlet=[dirichlet(SVector(0.0, 0.0); on=boundary(:all))]))
    vec_report = diagnostics(vec_model, solve!(vec_model))
    @test [l.active_functions for l in vec_report.levels] == [l.active_functions for l in report]
    @test vec_report.active_unknowns == 2 * sum(l -> l.active_functions, report)
end

@testset "BSpline extension: immersed physical domain — fold exempt, user mask not" begin
    # The family carries an immersed `PhysicalDomain`. The one thing that takes is
    # the fold exemption in `_active_cell_at_face`: a face whose inactive side is a
    # fully-fictitious cell carries no trace to vanish on, so it emits no
    # constraint, while a user-masked face still does. Without the exemption the
    # cut cells around the hole would lose their boundary modes.
    omega = box((0.0, 0.0), (1.0, 1.0))
    hole = physical_domain(x -> 0.25 - norm(x .- 0.5); lipschitz=1.0, subcell_length_scale=1 / 64,
                           max_depth=6)
    V = space(omega; cells=8, order=2, basis=bspline(), physical=hole)
    l = Unfitted._field_layout(prepare(mass(V)).dofs, :u).dofs
    # dim = cells + p = 10 per axis. Every function touches an active cell, and the
    # fold contributes no constraint, so every raw survives.
    @test length(l.raw_keys) == 100
    @test count(==(:overlay), l.elimination_source) == 0

    # Deactivate a corner block by hand on the same level: that face is a real
    # artificial boundary and must still be constrained.
    mask = trues(8, 8)
    mask[7:8, 7:8] .= false
    Vm = space(omega; cells=8, order=2, basis=bspline(), physical=hole, active=mask)
    lm = Unfitted._field_layout(prepare(mass(Vm)).dofs, :u).dofs
    @test count(==(:overlay), lm.elimination_source) > 0
end

@testset "BSpline extension: an immersed spline system is Jacobi-conditionable" begin
    # The cut-cell conditioning hazard, and the half of it that is about the basis
    # rather than about the diagonal. `cond(A)` on an immersed system is dominated
    # by the spread of the diagonal — a function whose support is almost entirely
    # fictitious carries an almost-zero entry — and a diagonal preconditioner
    # removes exactly that. What survives is near-dependence among the functions
    # that remain, and the two families differ on it by ten orders of magnitude:
    # on a cut cell holding a background vertex only one maximal-continuity spline
    # is supported there, while a C⁰ basis of degree ≥ 2 produces the dependence
    # on every small cut cell.
    #
    #   F. de Prenter, C. V. Verhoosel, G. J. van Zwieten, E. H. van Brummelen,
    #   *Condition number analysis and preconditioning of the finite cell method*,
    #   Comput. Methods Appl. Mech. Engrg. 316 (2017) 297–327,
    #   doi:10.1016/j.cma.2016.07.025.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    scaled(basis, p, n, r) =
        let hole = physical_domain(leaf(x -> r - norm(x .- 0.5); lipschitz=1.0);
                                   subcell_length_scale=1 / 128, max_depth=6),
            V = space(Ω; cells=n, order=p, basis=basis, physical=hole),
            model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

            diagnostics(model, solve!(model)).scaled_condition_estimate
        end
    for r in (0.25, 0.3)
        # 43.8 and 43.6 measured; the band is generous because the number is a
        # property of the geometry as well as of the basis.
        @test scaled(bspline(), 3, 8, r) < 1.0e3
    end
    # The default family on the same geometry, at a size the estimator still
    # reports on: orders of magnitude worse, and the gap is the point.
    @test scaled(IntegratedLegendre(), 2, 8, 0.25) > 1.0e3
    @test scaled(bspline(), 2, 8, 0.25) < 1.0e2
    # Both estimates exist and are finite on a system with no cut cells at all,
    # and there they agree to a small factor, because the diagonal is nearly
    # uniform and there is almost nothing for the scaling to remove. Note the
    # bound is two-sided: diagonal scaling is only optimal up to a factor of the
    # maximum row count (A. van der Sluis, Numer. Math. 14 (1969) 14–23,
    # doi:10.1007/BF02165096), so it may raise the condition number slightly —
    # measured 4.090 against 4.004 here — and an assertion that it never does
    # would be wrong rather than merely tight.
    plain = prepare(poisson(space(Ω; cells=4, order=2, basis=bspline()); source=1.0,
                            dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    d = diagnostics(plain, solve!(plain))
    @test isfinite(d.condition_estimate) && isfinite(d.scaled_condition_estimate)
    @test 0.5 < d.scaled_condition_estimate / d.condition_estimate < 2.0
    # Above the estimator's size cap both are `NaN`, which is a "not computed"
    # rather than a verdict — the same contract `condition_estimate` has always
    # had, and the reason the cross-family comparison above runs at a size where
    # the integrated-Legendre system still fits under it.
    big = prepare(poisson(space(Ω; cells=8, order=3); source=1.0,
                          dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    dbig = diagnostics(big, solve!(big))
    @test isnan(dbig.condition_estimate) && isnan(dbig.scaled_condition_estimate)
end

@testset "BSpline extension: finite-cell solve on a perforated plate" begin
    # End to end through the FCM path: α-fold, moment-fit cut-cell quadrature and
    # the B-spline trace constraints together. The operator must stay positive
    # definite, the space must still reproduce a constant *over Ω* (the covered cut
    # cells carry the partition of unity there), and the solution must agree with
    # the integrated-Legendre reference on the same mesh.
    omega = box((0.0, 0.0), (1.0, 1.0))
    hole = physical_domain(x -> 0.25 - norm(x .- 0.5); lipschitz=1.0, subcell_length_scale=1 / 64,
                           max_depth=6)
    V = space(omega; cells=8, order=3, basis=bspline(), physical=hole)
    m, _ = _gram(V)
    M = Symmetric(Matrix(m.matrix))
    @test isposdef(M)
    c = pinv(M) * load_vector(m; source=_one)
    proj = Solution(c, m.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))
    for x in (SVector(0.15, 0.15), SVector(0.5, 0.9), SVector(0.85, 0.5))
        @test value(proj, m, x) ≈ 1.0 atol = 1e-8
    end

    bc = [dirichlet(0.0; on=boundary(:all))]
    ub = field(:u, V)
    mb = prepare(poisson(V; source=1.0, dirichlet=bc))
    sb = solve!(mb)
    Vil = space(omega; cells=8, order=3, physical=hole)
    uil = field(:u, Vil)
    mil = prepare(poisson(Vil; source=1.0, dirichlet=bc))
    sil = solve!(mil)
    for x in ((0.15, 0.15), (0.5, 0.85))
        @test isapprox(value(sb, mb, ub, x), value(sil, mil, uil, x), atol=1e-4)
    end

    # A B-spline overlay on the same immersed base. The overlay's *own* fold empties
    # the cells inside the hole, so both branches of `_active_cell_at_face` — the
    # artificial mesh edge and the fold-exempt internal face — run on one level.
    # `cells=5` does not divide the base's, so no dedup clouds the check.
    mo, _ = _gram(overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=5, order=3))
    @test isposdef(Symmetric(Matrix(mo.matrix)))

    # The FCM plumbing must be a no-op when nothing is fictitious: with φ ≡ −1 the
    # answer is the plain B-spline run's, to roundoff. (Not asserted bit for bit —
    # the moment-fit rule reproduces tensor Gauss exactly but need not order its
    # points the same way, and the reduction is not reassociable.)
    full = physical_domain(x -> -1.0; lipschitz=1.0, subcell_length_scale=1.0)
    Vf = space(omega; cells=8, order=3, basis=bspline(), physical=full)
    uf = field(:u, Vf)
    mf = prepare(poisson(Vf; source=1.0, dirichlet=bc))
    sf = solve!(mf)
    Vp = space(omega; cells=8, order=3, basis=bspline())
    up = field(:u, Vp)
    mp = prepare(poisson(Vp; source=1.0, dirichlet=bc))
    sp = solve!(mp)
    @test value(sf, mf, uf, (0.5, 0.5)) ≈ value(sp, mp, up, (0.5, 0.5)) atol = 1e-12
end

@testset "BSpline extension: 1D values match BasicBSpline directly" begin
    # Construct a degree-3 family on [0, 1] with 4 cells. Evaluate the
    # per-axis 1D values at a midpoint of cell 2 and compare to a
    # direct BasicBSpline call.
    V = space(box((0.0,), (1.0,)); cells=4, order=3, basis=bspline())
    fam = V.levels[1].basis
    cell = CartesianIndex(2)
    val1d = (Vector{Float64}(undef, 4),)  # p + 1 = 4
    ξ = SVector(0.5)
    Unfitted._fill_factor_tables!(fam, val1d, (3,), ξ, cell)
    # Direct reference: cell 2 spans t ∈ [0.25, 0.5], midpoint at ξ = 0.5
    # maps to t = 0.25 + (0.5 + 1) / 2 * 0.25 = 0.25 + 0.1875 = 0.4375.
    P = fam.spaces[1]
    expected = bsplinebasisall(P, 2, 0.4375)
    for i in 1:4
        @test val1d[1][i] ≈ expected[i]
    end

    # The generic `::BasisFamily` evaluator must carry `cell` through to the
    # span lookup — in 1D its output is exactly the per-axis table, so it can be
    # compared to BasicBSpline directly, and evaluating the same ξ on cell 1
    # must give a different span and different values.
    on_cell2 = Unfitted.basis_values(fam, (3,), :tensor, ξ, cell)
    on_cell1 = Unfitted.basis_values(fam, (3,), :tensor, ξ, CartesianIndex(1))
    @test on_cell2 ≈ collect(expected)
    @test on_cell1 != on_cell2
end

@testset "BSpline extension: dof-key sharing across cells" begin
    # Two adjacent cells of the same level should share p = degree
    # raw dofs along the shared axis.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=3, basis=bspline())
    problem = poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    model = prepare(problem)
    field_layout = model.dofs.fields[1]
    cell1 = Unfitted.cell_dofs(field_layout.dofs, 1, CartesianIndex(1, 1))
    cell2 = Unfitted.cell_dofs(field_layout.dofs, 1, CartesianIndex(2, 1))
    # Adjacent cells along axis 1 share p+1 functions along axis 1; with
    # the 2D tensor product, the number of shared raw dofs equals
    # (p + 1) on the shared axis × (p + 1) on the perpendicular axis −
    # the (p + 1) × 1 strip that's purely cell-1, etc. Simpler check:
    # the set intersection has length p * (p + 1) = 12 for p = 3.
    @test length(intersect(Set(cell1), Set(cell2))) == 12
end

@testset "BSpline extension: 2D Poisson matches integrated Legendre" begin
    f(x) = 1.0
    # B-spline run.
    Vbs = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, basis=bspline())
    u_bs = field(:u, Vbs)
    p_bs = poisson(Vbs; source=f, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    m_bs = prepare(p_bs)
    s_bs = solve!(m_bs)
    ucenter_bs = value(s_bs, m_bs, u_bs, (0.5, 0.5))
    # Integrated Legendre reference run.
    Vil = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3)
    u_il = field(:u, Vil)
    p_il = poisson(Vil; source=f, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    m_il = prepare(p_il)
    s_il = solve!(m_il)
    ucenter_il = value(s_il, m_il, u_il, (0.5, 0.5))
    # Both must approximate the analytic ≈ 0.07367 to ~4 digits at
    # this resolution and agree with each other to ~3.
    @test isapprox(ucenter_bs, 0.07367, atol=1.0e-3)
    @test isapprox(ucenter_il, 0.07367, atol=1.0e-3)
    @test isapprox(ucenter_bs, ucenter_il, atol=1.0e-3)
end

@testset "BSpline extension: 1D bar with homogeneous Dirichlet" begin
    # u''(x) = 1, u(0) = u(1) = 0, exact: u(x) = x(1-x)/2.
    V = space(box((0.0,), (1.0,)); cells=8, order=3, basis=bspline())
    u = field(:u, V)
    problem = poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    model = prepare(problem)
    solution = solve!(model)
    for x in (0.25, 0.5, 0.75)
        expected = x * (1 - x) / 2
        @test isapprox(value(solution, model, u, (x,)), expected, atol=1.0e-6)
    end
end

@testset "BSpline extension: arbitrary mask geometries accepted" begin
    domain2d = box((0.0, 0.0), (1.0, 1.0))
    # Separable: lower-left 2×2 quadrant of a 4×4 mesh.
    mask_sep = falses(4, 4)
    mask_sep[1:2, 1:2] .= true
    V_sep = space(domain2d; cells=4, order=3, basis=bspline(), active=mask_sep)
    @test V_sep.levels[1].basis isa Unfitted.BasisFamily

    # Disjoint rectangles separated by inactive strip.
    mask_disj = falses(5, 5)
    mask_disj[1:2, 1:2] .= true
    mask_disj[4:5, 4:5] .= true
    V_disj = space(domain2d; cells=5, order=2, basis=bspline(), active=mask_disj)
    @test V_disj.levels[1].basis isa Unfitted.BasisFamily

    # L-shape (non-separable): linear constraints handle it.
    mask_L = falses(3, 3)
    mask_L[1:3, 1] .= true
    mask_L[1, 2:3] .= true
    V_L = space(domain2d; cells=3, order=2, basis=bspline(), active=mask_L)
    @test V_L.levels[1].basis isa Unfitted.BasisFamily

    # Centered hole inside an otherwise active patch.
    mask_hole = trues(4, 4)
    mask_hole[2:3, 2:3] .= false
    V_hole = space(domain2d; cells=4, order=3, basis=bspline(), active=mask_hole)
    @test V_hole.levels[1].basis isa Unfitted.BasisFamily
end

@testset "Linear-constraint resolver: cascade through chained pivots" begin
    # Hand-crafted constraint system that requires the back-substitution
    # branch of `_resolve_constraints!`. Three raws, two constraints:
    #
    #     C1:  u₁ − u₂ = 0       greedy pivots u₁, expansion = [(2, 1)]
    #     C2:  2 u₂ − u₃ = 0     greedy pivots u₂, expansion = [(3, 0.5)]
    #
    # Back-substitution must rewrite u₁'s expansion in terms of u₃ once
    # u₂ becomes a pivot: u₁ = u₂ = 0.5 u₃. Without the back-substitute
    # step u₁ would still mention the now-pivoted u₂ and the assembly
    # would chase a stale pointer.
    constraints = [Unfitted.LinearConstraint{Float64}([1, 2], [1.0, -1.0]),
                   Unfitted.LinearConstraint{Float64}([2, 3], [2.0, -1.0])]
    expansion = Vector{Vector{Tuple{Int,Float64}}}(undef, 3)
    Unfitted._resolve_constraints!(expansion, constraints, 3)
    @test expansion[1] == [(3, 0.5)]
    @test expansion[2] == [(3, 0.5)]
    @test expansion[3] == [(3, 1.0)]
end

@testset "Linear-constraint resolver: a constraint that collapsed to 0 = 0 eliminates nothing" begin
    # `_combine_terms` applies its drop tolerance to a one-term list too. It has to:
    # the B-spline trace generator emits the same constraint once per perpendicular
    # cell touching a face, and after the first emission pivots its raw away, a
    # re-emission can be left with a single term whose coefficient is exactly zero.
    # `0 · u = 0` constrains nothing; pivoting on it would strongly eliminate u.
    @test isempty(Unfitted._combine_terms([(1, 0.0)]))
    @test Unfitted._combine_terms([(1, 2.0)]) == [(1, 2.0)]
    constraints = [Unfitted.LinearConstraint{Float64}([1, 2], [1.0, 0.0]),
                   Unfitted.LinearConstraint{Float64}([1, 2], [1.0, 0.0])]
    expansion = Vector{Vector{Tuple{Int,Float64}}}(undef, 2)
    Unfitted._resolve_constraints!(expansion, constraints, 2)
    @test expansion[1] == []            # the constraint that says something
    @test expansion[2] == [(2, 1.0)]    # its duplicate leaves u₂ free

    # The configuration that reaches it end to end. A *masked* level below maximal
    # continuity is the only shape that still emits multi-raw trace constraints: an
    # overlay's own box faces are clamped ends, whose trace matrix is triangular and
    # therefore resolves to strong eliminations, and a maximal-continuity level emits
    # no linear constraint at all. Here the same perpendicular index is constrained
    # from several mask cells along the face, so the re-emissions are exactly the
    # collapsed constraints the unit test above pins.
    free_on_level(V, level) =
        let l = Unfitted._field_layout(prepare(mass(V)).dofs, :u).dofs
            count(i -> l.raw_keys[i].level == level && l.elimination_source[i] === :free,
                  eachindex(l.raw_keys))
        end
    Ω = box((0.0, 0.0), (1.0, 1.0))
    mask = trues(8, 8)
    mask[5:8, 5:8] .= false
    Vmask = space(Ω; cells=8, order=3, basis=bspline(; continuity=1), active=mask)
    layout = Unfitted._field_layout(prepare(mass(Vmask)).dofs, :u).dofs
    @test layout.has_linear_constraints
    @test count(==(:overlay), layout.elimination_source) > 0
    # And the two shapes that no longer reach it, kept because their dof ledgers are
    # the closed forms the family's docstring states: `n − p` per axis at maximal
    # continuity, for a level whose faces are all artificial.
    B = box((0.25, 0.25), (0.75, 0.75))
    @test free_on_level(overlay(space(Ω; cells=8, order=1, basis=bspline()), B; cells=4, order=1),
                        2) == 9
    @test free_on_level(overlay(space(Ω; cells=8, order=3, basis=bspline()), B; cells=8, order=3),
                        2) == 25
end

@testset "Linear-constraint resolver: a pivot contributes its expansion to the field" begin
    # `dof_value` used to read `constrained_value` for every eliminated raw, which
    # is zero for a linear-constraint pivot. Assembly distributes through
    # `raw_expansion` when it emits, so the *solve* was right and only the
    # *reconstruction* dropped the pivot's contribution — silently, in `value`,
    # `field_gradient`, `l2_error`, `write_vtk`, and L2Projection's source read.
    #
    # The witness is an identity that needs both paths: for a mass operator M and
    # the active coefficient vector c of an L² projection,
    #
    #     cᵀ M c = ∫_Ω u_h²
    #
    # exactly, because both sides are the same quadratic form. The left side goes
    # through assembly, the right through `dof_value`. Measured before the fix:
    # 4.9e-2 relative on the C¹ masked level below, 5.8e-3 at C⁰, and 4.4e-7 on the
    # base-plus-masked-overlay stack — small enough to sit under a hand-written
    # tolerance, which is how it survived.
    f(x) = sin(2.3 * x[1]) * cos(1.7 * x[2]) + 0.4 * x[1] * x[2]
    function projection_gap(V)
        u = field(:u, V)
        model = prepare(Problem((u,); blocks=(mass_block(u),), loads=(source_load(u; source=f),)))
        sol = solve!(model)
        c = sol.coefficients
        quadratic = c' * model.matrix * c
        integral = 0.0
        foreach_quadrature_point(model; state=sol) do q
            v = value(q.state, u)
            integral += q.weight * v * v
            return nothing
        end
        return abs(quadratic - integral) / abs(quadratic),
               Unfitted.has_linear_constraints(model.dofs)
    end

    Ω = box((0.0, 0.0), (1.0, 1.0))
    mask = trues(6, 6)
    mask[4:6, 4:6] .= false                          # L-shaped active region
    for m in 0:1                                     # m < p − 1 ⇒ genuine pivots
        gap, pivots = projection_gap(space(Ω; cells=6, order=3, active=mask,
                                           basis=bspline(; continuity=m)))
        @test pivots
        @test gap < 1e-13
    end
    # Two controls, both of which were already correct and must stay so: a family
    # that produces no pivots at all, and the *maximal*-continuity B-spline level,
    # whose mask constraints resolve to strong eliminations rather than pivots.
    for V in (space(Ω; cells=6, order=3, active=mask),
              space(Ω; cells=6, order=3, active=mask, basis=bspline(; continuity=2)))
        gap, pivots = projection_gap(V)
        @test !pivots
        @test gap < 1e-13
    end
end

@testset "BSpline extension: C¹ overlay strictly inside the base assembles" begin
    # B-spline base + C¹ B-spline overlay whose four corners are *inside*
    # the base's mesh, so each corner raw participates in artificial-
    # boundary constraints from *both* axes (each at derivative orders
    # k = 0 and k = 1). The greedy cascade in `_resolve_constraints!`
    # must produce a non-singular system — coverage for the
    # back-substitution branch Cᵐ overlay corners reach.
    #
    # This overlay also *nests* (four cells over half of eight divide the base's),
    # so the base function buried under it is deduped. That is a separate mechanism
    # from the corner cascade, and it is the one that decides the conditioning here:
    # without it the two levels carry the same function twice and the operator is
    # exactly singular. The nesting testset below isolates it; what this one asserts
    # is that the corner cascade on top of it still leaves a system a direct solve
    # handles cleanly (cond ≈ 3e2, not the ≈ 1e17 of the undeduped stack).
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, basis=bspline())
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3, basis=bspline(continuity=1))
    u = field(:u, V)
    problem = poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    model = prepare(problem)
    solution = solve!(model)
    @test active_unknowns(model.dofs) > 0
    @test diagnostics(model, solution).residual_norm < 1.0e-10
    # u(0.5, 0.5) for −Δu = 1 on the unit square with u = 0 on ∂Ω.
    @test isapprox(value(solution, model, u, (0.5, 0.5)), 0.07367, atol=1.0e-4)
end

@testset "BSpline extension: a nested overlay deduplicates what it reproduces" begin
    # An overlay of the same degree whose cell boundaries include the base's
    # reproduces — exactly — every base function buried underneath it, so the two
    # levels carry the same function twice and the superposition is singular until
    # one copy goes. Leaf semantics (on by default) remove it.
    #
    # Which functions qualify is decidable from the knot vectors alone: with uniform
    # simple interior knots, base function `i` lives on cells `max(1, i − p) …
    # min(n, i)`, and it is buried iff that range sits inside the overlay's. Over
    # base cells 3…6 (the overlay box [0.25, 0.75] on 8 cells) that leaves
    # `{i : i − p ≥ 3, i ≤ 6}` — `4 − p` functions per axis, `(4 − p)^D` in D
    # dimensions. The dedup count below is that number, not an observation.
    function nested(p)
        return overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=p, basis=bspline()),
                       box((0.25, 0.25), (0.75, 0.75)); cells=4, order=p)
    end
    for p in 1:3
        m, l = _gram(nested(p))
        @test count(==(:dedup), l.elimination_source) == (4 - p)^2
        # B-splines have no bubble/skeleton split, so the dedup is all of order
        # reduction for this family — nothing is shed for accuracy.
        @test count(==(:coverage), l.elimination_source) == 0
        # Repaired, and still complete: the removed function is reproduced by the
        # overlay, so the span is unchanged and the projection is unaffected.
        M = Symmetric(Matrix(m.matrix))
        @test rank(M) == active_unknowns(l)
        @test isposdef(M)
        @test _proj_residual(m, _one, _INT_ONE) < 1e-6
        @test _proj_residual(m, _x1, _INT_X1SQ) < 1e-6
    end

    # Opting out keeps the duplicate and the operator is exactly singular. That is
    # the documented trade — `bspline`'s docstring names it, and names the way out
    # (break the nesting) — and it is the same one integrated Legendre makes.
    mf, lf = _gram(nested(3); prune=false)
    @test count(==(:dedup), lf.elimination_source) == 0
    @test rank(Symmetric(Matrix(mf.matrix))) == active_unknowns(lf) - 1
end

@testset "BSpline extension: a fictitious fold does not disable the dedup" begin
    # The same half-plane fold as "a cover ending inside a folded coarse cell is still a
    # cover" in test_coverage_reduction.jl, under a nested B-spline stack. Ω = {x₁ ≤ 0.9}
    # in [-1, 2]², base 6×6 at degree 2, cover of quarter-size cells.
    #
    # The B-spline gap was wider than the integrated-Legendre one and did not need the
    # cover to stop anywhere in particular: the burial test demanded every support cell
    # be *active*, and a degree-2 function spans three cells per axis, so one folded cell
    # anywhere under the support disabled the dedup for every function reaching it. The
    # cover reaching the domain face (xhi = 2.0) failed exactly like the cover ending
    # inside the fold (1.5, 1.25) — 16 exact null modes, `dedup == 0`, in all three.
    #
    # On Ω the truncated and untruncated base functions coincide (nothing is integrated
    # over a fictitious cell), the cover's own functions on the fold face are free
    # (`_active_cell_at_face` exempts it), and a nested same-degree cover reproduces the
    # base function exactly — so all 16 must go, and the space must be unchanged.
    Ωfold = box((-1.0, -1.0), (2.0, 2.0))
    half() = physical_domain(x -> x[1] - 0.9; lipschitz=1.0, subcell_length_scale=0.125,
                             max_depth=6)
    function fold_stack(xhi)
        base = space(Ωfold; cells=6, order=2, basis=bspline(), physical=half())
        return overlay(base, box((0.0, -1.0), (xhi, 2.0)); cells=(round(Int, 4xhi), 12), order=2)
    end
    # Where the cover's face falls no longer changes the answer, and that is the whole
    # point of the burial test being exact. `xhi = 1.0` is the configuration that used
    # to be pinned `@test_broken`: the cover's face lands on a knot of the level below
    # *inside* the band of cut cells, which under the clamped mechanism let the cover
    # reproduce a *combination* of two buried functions without reproducing either, and
    # left eight exact null modes no per-function dedup could see. Support selection
    # removes the functions that caused it — the cover is not clamped at that face, it
    # simply stops — and the subdivision walk then answers containment exactly.
    for xhi in (2.0, 1.75, 1.5, 1.25, 1.0)
        m, l = _gram(fold_stack(xhi))
        mu, _ = _gram(fold_stack(xhi); prune=false)
        M = Symmetric(Matrix(m.matrix))
        @test count(==(:dedup), l.elimination_source) == 16
        @test active_unknowns(l) == 88
        @test rank(M) == active_unknowns(l)                     # nothing left over
        @test isposdef(M)
        @test active_unknowns(l) == rank(Matrix(mu.matrix))     # nothing real deleted
    end

    # The fold is not the only thing the cover face may land in. Here Ω = {x₁ ≤ 0.6}, so
    # the cover ending at x = 1.0 ends two cells deep in fictitious material. The
    # subdivision walk reaches past the cover's own box there and finds the fine
    # functions it needs, because a function whose support sticks out into fictitious
    # material is itself admissible — nothing is integrated beyond the face, so the
    # truncation is invisible on Ω. A rule that stopped at the cover's box would
    # deduplicate nothing here and leave all 16 duplicates.
    deep = overlay(space(Ωfold; cells=6, order=2, basis=bspline(),
                         physical=physical_domain(x -> x[1] - 0.6; lipschitz=1.0,
                                                  subcell_length_scale=0.125, max_depth=6)),
                   box((0.0, -1.0), (1.0, 2.0)); cells=(4, 12), order=2)
    md, ld = _gram(deep)
    @test count(==(:dedup), ld.elimination_source) == 16
    @test active_unknowns(ld) == 74
    @test rank(Symmetric(Matrix(md.matrix))) == active_unknowns(ld)

    # And the cut line may fall *beyond* the cover's face rather than before it, which
    # is the case that needs the dedup to reach past the box in the other direction:
    # Ω = {x₁ ≤ 1.1} with the cover ending at x = 1.25 deduplicates 24, and a rule
    # keyed on whole cell boxes rejects them all because the base cell straddling
    # x = 1.1 is not contained in the cover's box even though its material part is.
    beyond = overlay(space(Ωfold; cells=6, order=2, basis=bspline(),
                           physical=physical_domain(x -> x[1] - 1.1; lipschitz=1.0,
                                                    subcell_length_scale=0.125, max_depth=6)),
                     box((0.0, -1.0), (1.25, 2.0)); cells=(5, 12), order=2)
    mb, lb = _gram(beyond)
    @test count(==(:dedup), lb.elimination_source) == 24
    @test rank(Symmetric(Matrix(mb.matrix))) == active_unknowns(lb)
    @test active_unknowns(lb) == rank(Matrix(_gram(beyond; prune=false)[1].matrix))
end

@testset "BSpline extension: a degree-1 cover is the hat space and deduplicates hats" begin
    # The reverse direction of "the dedup fires only where the span really contains"
    # below — an integrated-Legendre base under a B-spline cover, rather than a B-spline
    # base under an integrated-Legendre cover. A degree-1 open-knot B-spline on a nested mesh *is* the Q1 hat
    # basis, so it reproduces the buried base hat and the two copies are linearly
    # dependent; at degree ≥ 2 the cover is C^(p−1) at a simple interior knot and cannot
    # carry the kink, so the hat must stay. The dedup gate used to test the cover's
    # family (`k.basis isa IntegratedLegendre`) rather than ask it, so the degree-1 cover
    # was refused the dedup and the stack carried one exact null mode.
    base() = space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=2)
    B = box((0.25, 0.25), (0.75, 0.75))
    # The degree-2 cover keeps `(4 − 2)² = 4` functions under support selection, where
    # the clamped mechanism kept `(4 + 2 − 2)² = 16`; the degree-1 cover is unaffected,
    # because `p − 1 = 0` makes the two mechanisms the same rule.
    for (deg, dedup, unknowns) in ((1, 1, 81), (2, 0, 77))
        V = overlay(base(), B; cells=4, order=deg, basis=bspline())
        @test Unfitted._spans_hats(V.levels[2].basis) == (deg == 1)
        m, l = _gram(V)
        @test count(==(:dedup), l.elimination_source) == dedup
        @test count(==(:coverage), l.elimination_source) == 8
        @test active_unknowns(l) == unknowns
        @test rank(Symmetric(Matrix(m.matrix))) == active_unknowns(l)
        @test isposdef(Symmetric(Matrix(m.matrix)))
    end

    # And the degree-1 cover gives the same space an integrated-Legendre p = 1 cover
    # does: same active count, and the Gram spectra agree to roundoff.
    mb, lb = _gram(overlay(base(), B; cells=4, order=1, basis=bspline()))
    mi, li = _gram(overlay(base(), B; cells=4, order=1))
    @test active_unknowns(lb) == active_unknowns(li)
    @test eigvals(Symmetric(Matrix(mb.matrix))) ≈ eigvals(Symmetric(Matrix(mi.matrix)))
end

@testset "BSpline extension: the dedup fires only where the span really contains" begin
    base() = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, basis=bspline())
    B = box((0.25, 0.25), (0.75, 0.75))
    # Non-nested cells, a coarser degree, a finer degree: in each case the covering
    # span misses the base function — the finer degree because a maximal-regularity
    # spline of degree p + 1 is C^p at a simple knot, too smooth to carry the kink
    # of a degree-p one. Nothing is deduped, and nothing is rank deficient either.
    for over in (overlay(base(), B; cells=5, order=3), overlay(base(), B; cells=4, order=2),
                 overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=2, basis=bspline()), B; cells=4,
                         order=3))
        m, l = _gram(over)
        @test count(==(:dedup), l.elimination_source) == 0
        @test rank(Symmetric(Matrix(m.matrix))) == active_unknowns(l)
    end

    # The covering *span* decides, not the covering family: a `:tensor` integrated
    # Legendre overlay of order ≥ p contains every C⁰ tensor polynomial of that
    # degree on its cells, and a maximal-regularity B-spline of degree p is one.
    # Below p it is not.
    for (q, expected) in ((2, 0), (3, 1), (4, 1))
        m, l = _gram(overlay(base(), B; cells=4, order=q, basis=Unfitted.IntegratedLegendre()))
        @test count(==(:dedup), l.elimination_source) == expected
        @test rank(Symmetric(Matrix(m.matrix))) == active_unknowns(l)
    end
end

@testset "BSpline extension: nested dedup in 3D" begin
    # The support rule and the coverage walk are D-generic; 3D is where a wrong
    # `ntuple` shows up. Degree 1 on 8 base cells buries `{i : i − 1 ≥ 3, i ≤ 6}` =
    # 3 functions per axis, so 27 of them in 3D.
    omega3 = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    V = overlay(space(omega3; cells=8, order=1, basis=bspline()),
                box((0.25, 0.25, 0.25), (0.75, 0.75, 0.75)); cells=4, order=1)
    m, l = _gram(V)
    @test count(==(:dedup), l.elimination_source) == 27
    @test isposdef(Symmetric(Matrix(m.matrix)))
end

@testset "BSpline extension: C^m overlay reduces dof count for smooth problems" begin
    # 2D smooth Laplace with nonzero Dirichlet on the top edge:
    #   exact u(x,y) = sin(πx) sinh(πy) / sinh(π).
    # Base + overlay; C^m overlay constraints should reduce active dofs
    # while keeping the L² error competitive for this smooth solution.
    smooth(x) = sin(pi * x[1]) * sinh(pi * x[2]) / sinh(pi)
    function build(overlay_basis, base_continuity)
        V = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3,
                  basis=bspline(continuity=base_continuity))
        V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=8, order=3, basis=overlay_basis)
        u = field(:u, V)
        problem = poisson(V; source=0.0,
                          dirichlet=[dirichlet(0.0; on=boundary(axis=1, side=:lower)),
                                     dirichlet(0.0; on=boundary(axis=1, side=:upper)),
                                     dirichlet(0.0; on=boundary(axis=2, side=:lower)),
                                     dirichlet(smooth; on=boundary(axis=2, side=:upper))])
        model = prepare(problem)
        solution = solve!(model)
        return (active=active_unknowns(model.dofs),
                err=diagnostics(model, solution; exact=smooth).l2_error)
    end
    c0 = build(bspline(; continuity=0), 0)
    c1 = build(bspline(; continuity=1), 0)
    c2 = build(bspline(), 0)                      # `:maximal` is C² at degree 3
    # Smoothness across the overlay's artificial boundary is bought with unknowns,
    # and the ledger is exact: per axis the overlay keeps `n + p − 2(m + 1)`
    # functions at a requested `m`, and `n − p` at `:maximal`. With n = 8 and p = 3
    # that is 9, 7 and 5 per axis, so the counts fall strictly as m rises.
    @test c1.active < c0.active
    @test c2.active < c1.active
    # All variants reach comparable L² error on the smooth problem (well below
    # 1e-3), which is the point: on a solution this smooth the extra functions the
    # lower continuities buy are not what the accuracy rests on.
    @test c0.err < 1.0e-3
    @test c1.err < 1.0e-3
    @test c2.err < 1.0e-3
end

@testset "BSpline extension: L-shape overlay mask stays independent and complete" begin
    # The L-shape mask is non-separable, so its artificial boundary is carried by
    # linear constraints rather than by dof-wise elimination. The mass matrix is the
    # Gram matrix of the surviving basis, so it is full rank iff those constraints
    # eliminate nothing they should not, and the space reproduces 1 and x₁ exactly iff
    # the base's linear skeleton survives under the covered region. The L is one cell
    # thick, so no base vertex is buried and the linear dedup never fires.
    Vbase = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=2)
    mask_L = falses(4, 4)
    mask_L[1:4, 1] .= true
    mask_L[1, 2:4] .= true
    Vover = overlay(Vbase, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3, basis=bspline(),
                    active=mask_L)
    m, l = _gram(Vover)
    @test rank(Symmetric(Matrix(m.matrix))) == active_unknowns(l)
    @test _proj_residual(m, _one, _INT_ONE) < 1e-6
    @test _proj_residual(m, _x1, _INT_X1SQ) < 1e-6
end

@testset "BSpline extension: block overlay mask keeps the vertex the spline cannot replace" begin
    # The same Gram check on a separable 2×2 block mask, whose active region is a nested
    # patch of four base cells, burying the base vertex at (0.375, 0.375). Its hat is
    # kinked there, and the covering level is a maximal-smoothness spline whose span has
    # no kink at a simple interior knot — so the linear dedup must not fire, however
    # perfectly the two meshes nest. Covered-mode pruning still sheds the buried high-order
    # modes; the surviving linear skeleton is what keeps the space complete to first
    # order. Deduping the vertex costs ‖1 − Π1‖ ≈ 6e-3, 0.44 pointwise.
    Vbase = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=2)
    overlay_mask = falses(4, 4)
    overlay_mask[1:2, 1:2] .= true
    Vover = overlay(Vbase, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3, basis=bspline(),
                    active=overlay_mask)
    m, l = _gram(Vover)
    @test count(==(:dedup), l.elimination_source) == 0
    @test rank(Symmetric(Matrix(m.matrix))) == active_unknowns(l)
    @test _proj_residual(m, _one, _INT_ONE) < 1e-6
    @test _proj_residual(m, _x1, _INT_X1SQ) < 1e-6
end

@testset "BSpline extension: moved overlay rebuilds knot vectors" begin
    # Moving a B-spline overlay must rebuild its per-axis knot vectors
    # for the overlay's new cell coordinates — reusing the stale family
    # would leave `axis_coords` / `cell_to_span` pointing at the old
    # position and corrupt every basis evaluation on the moved level.
    # Regression for the `instantiate_basis` hook on `moved_space`.
    V = space(box((0.0,), (1.0,)); cells=8, order=3)
    V = overlay(V, box((0.2,), (0.6,)); cells=4, order=3, basis=bspline())
    fam_before = V.levels[2].basis
    @test fam_before isa Unfitted.BasisFamily
    @test fam_before.axis_coords[1][1] ≈ 0.2
    @test fam_before.axis_coords[1][end] ≈ 0.6

    V2 = moved_space(V; level=2, to=box((0.3,), (0.7,)))
    fam_after = V2.levels[2].basis
    @test fam_after isa typeof(fam_before)
    @test fam_after.axis_coords[1][1] ≈ 0.3
    @test fam_after.axis_coords[1][end] ≈ 0.7
    # Cell count and degree are unchanged, so dim = cells + p still holds.
    @test BasicBSpline.dim(fam_after.spaces[1]) == 4 + 3

    # End-to-end: the moved configuration still assembles and solves.
    u = field(:u, V2)
    problem = poisson(V2; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    model = prepare(problem)
    solution = solve!(model)
    @test isfinite(value(solution, model, u, (0.5,)))
end
