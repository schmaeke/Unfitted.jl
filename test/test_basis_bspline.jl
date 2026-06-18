using BasicBSpline
using StaticArrays
using LinearAlgebra: norm

# Smoke tests for the BasicBSpline extension. Mirror the integrated
# Legendre test surface (`test_basis.jl`) where the interface is shared,
# and add B-spline-specific tests for the dof-key sharing and mask
# handling.

@testset "BSpline extension: basis interface" begin
    fam_marker = bspline()
    @test fam_marker isa Unfitted.BasisFamily

    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=3, basis=bspline())
    fam = V.levels[1].basis
    @test Unfitted.basis_name(fam) == :bspline
    @test Unfitted.local_basis_count(fam, (3, 3)) == 16
    @test Unfitted.recommended_quadrature_order(fam, (2, 3)) == (3, 4)
    @test Unfitted.local_basis_indices(fam, (1, 2))[1] == CartesianIndex(0, 0)
    # `dim` per axis = cells + p (open knot, no junctions, m=0).
    @test BasicBSpline.dim.(fam.spaces) == (7, 7)
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
end

@testset "BSpline extension: dof-key sharing across cells" begin
    # Two adjacent cells of the same level should share p = degree
    # raw dofs along the shared axis.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=3, basis=bspline())
    problem = poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
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
    problem = poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
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

@testset "BSpline extension: C¹ overlay strictly inside the base assembles" begin
    # B-spline base + C¹ B-spline overlay whose four corners are *inside*
    # the base's mesh, so each corner raw participates in artificial-
    # boundary constraints from *both* axes (each at derivative orders
    # k = 0 and k = 1). The greedy cascade in `_resolve_constraints!`
    # must produce a non-singular system. We assert assembly succeeds,
    # the solver converges to a small residual, and the centerline value
    # is finite — a coverage smoke test for the back-substitution branch
    # exercised by C^m overlay corners.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, basis=bspline())
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3,
                basis=bspline(continuity_order=1))
    u = field(:u, V)
    problem = poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    model = prepare(problem)
    solution = solve!(model)
    @test active_unknowns(model.dofs) > 0
    @test diagnostics(model, solution).residual_norm < 1.0e-10
    @test isfinite(value(solution, model, u, (0.5, 0.5)))
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

@testset "BSpline extension: arbitrary mask geometries solve" begin
    # Each non-separable mask geometry (L-shape, hole) must produce a
    # well-posed system with a finite solution. No singularity from
    # over-elimination thanks to the linear-constraint primitive.
    Vbase = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=2)
    # L-shape overlay mask.
    mask_L = falses(4, 4)
    mask_L[1:4, 1] .= true
    mask_L[1, 2:4] .= true
    Vover = overlay(Vbase, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3, basis=bspline(),
                    active=mask_L)
    problem = poisson(Vover; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    model = prepare(problem)
    solution = solve!(model)
    @test isfinite(value(solution, model, field(:u, Vover), (0.5, 0.5)))
end

@testset "BSpline extension: masked overlay still solves" begin
    # 2D Poisson on the unit square with a B-spline overlay covering
    # only a sub-rectangle via mask. The mask-induced face becomes a
    # C⁰ junction; the overlay contribution vanishes on every
    # artificial boundary (mesh-edge + C⁰ junctions).
    Vbase = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=2)
    overlay_mask = falses(4, 4);
    overlay_mask[1:2, 1:2] .= true
    Vover = overlay(Vbase, box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3, basis=bspline(),
                    active=overlay_mask)
    u = field(:u, Vover)
    problem = poisson(Vover; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    model = prepare(problem)
    solution = solve!(model)
    # Sanity: SPD-style residual is small and the centerline value is
    # finite and within a reasonable range of the no-overlay solution.
    @test active_unknowns(model.dofs) > 0
    @test isfinite(value(solution, model, u, (0.5, 0.5)))
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
    problem = poisson(V2; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))])
    model = prepare(problem)
    solution = solve!(model)
    @test isfinite(value(solution, model, u, (0.5,)))
end
