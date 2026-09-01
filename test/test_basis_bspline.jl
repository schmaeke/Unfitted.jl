using BasicBSpline
using StaticArrays
using LinearAlgebra: Symmetric, isposdef, norm, pinv, rank

# Smoke tests for the BasicBSpline extension. Mirror the integrated
# Legendre test surface (`test_basis.jl`) where the interface is shared,
# and add B-spline-specific tests for the dof-key sharing and mask
# handling.
#
# Several testsets below reuse `_gram` / `_proj_residual` and the `_one` / `_x1` /
# `_INT_*` constants from `test_coverage_reduction.jl`, which `runtests.jl`
# includes first; this file does not stand alone.

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

@testset "BSpline extension: continuity_order validation" begin
    # Negative continuity_order is rejected at factory time.
    @test_throws ArgumentError bspline(continuity_order=-1)
    # continuity_order > p - 1 is rejected at level construction.
    @test_throws ArgumentError space(box((0.0,), (1.0,)); cells=4, order=2,
                                     basis=bspline(continuity_order=2))
    # Too-thin level for the requested continuity is rejected.
    @test_throws ArgumentError space(box((0.0,), (1.0,)); cells=1, order=2,
                                     basis=bspline(continuity_order=1))
    # Valid C¹ family on a degree-2 mesh.
    V = space(box((0.0,), (1.0,)); cells=8, order=3, basis=bspline(continuity_order=1))
    @test V.levels[1].basis isa Unfitted.BasisFamily
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

    # The two configurations that reach it end to end. Degree 1 puts exactly two
    # terms in each trace constraint, one of them zero at a clamped end, so a 2D
    # overlay must keep `dim − 2 = 3` functions per axis, 9 in all — not 1.
    free_on_overlay(V) =
        let l = Unfitted._field_layout(prepare(mass(V)).dofs, :u).dofs
            count(i -> l.raw_keys[i].level == 2 && l.elimination_source[i] === :free,
                  eachindex(l.raw_keys))
        end
    Ω = box((0.0, 0.0), (1.0, 1.0))
    B = box((0.25, 0.25), (0.75, 0.75))
    @test free_on_overlay(overlay(space(Ω; cells=8, order=1, basis=bspline()), B; cells=4, order=1)) ==
          9
    # `continuity_order = p − 1` is the other one: its top-order trace constraint
    # leaves a single zero term behind. dim = 11, m + 1 = 3 eliminated per side ⇒ 5
    # per axis, 25 in all.
    @test free_on_overlay(overlay(space(Ω; cells=8, order=3, basis=bspline()), B; cells=8, order=3,
                                  basis=bspline(continuity_order=2))) == 25
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
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3,
                basis=bspline(continuity_order=1))
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
    # one copy goes. `prune_covered` (`true` by default) removes it.
    #
    # Which functions qualify is decidable from the knot vectors alone: with uniform
    # simple interior knots, base function `i` lives on cells `max(1, i − p) …
    # min(n, i)`, and it is buried iff that range sits inside the overlay's. Over
    # base cells 3…6 (the overlay box [0.25, 0.75] on 8 cells) that leaves
    # `{i : i − p ≥ 3, i ≤ 6}` — `4 − p` functions per axis, `(4 − p)^D` in D
    # dimensions. The dedup count below is that number, not an observation.
    nested(ro, p) = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=p, basis=bspline(),
                                  prune_covered=ro), box((0.25, 0.25), (0.75, 0.75)); cells=4,
                            order=p)
    for p in 1:3
        m, l = _gram(nested(true, p))
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
    mf, lf = _gram(nested(false, 3))
    @test count(==(:dedup), lf.elimination_source) == 0
    @test rank(Symmetric(Matrix(mf.matrix))) == active_unknowns(lf) - 1
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
                  basis=bspline(continuity_order=base_continuity))
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
    c0 = build(bspline(), 0)
    c1 = build(bspline(continuity_order=1), 0)
    c2 = build(bspline(continuity_order=2), 1)
    # Higher continuity strictly reduces active dof count.
    @test c1.active < c0.active
    @test c2.active < c1.active
    # All variants reach comparable L² error on the smooth problem
    # (well below 1e-3).
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
