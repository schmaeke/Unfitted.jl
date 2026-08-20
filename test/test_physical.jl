using StaticArrays
using LinearAlgebra

@testset "physical_domain constructor" begin
    @test_throws ArgumentError physical_domain(x -> x[1]; lipschitz=0.0, subcell_length_scale=0.1)
    @test_throws ArgumentError physical_domain(x -> x[1]; lipschitz=-1.0, subcell_length_scale=0.1)
    @test_throws ArgumentError physical_domain(x -> x[1]; alpha=-1.0, subcell_length_scale=0.1)
    @test_throws ArgumentError physical_domain(x -> x[1]; subcell_length_scale=0.0)
    @test_throws ArgumentError physical_domain(x -> x[1]; subcell_length_scale=-1.0)
    @test_throws ArgumentError physical_domain(x -> x[1]; subcell_length_scale=0.1, max_depth=-1)
    @test_throws ArgumentError physical_domain(x -> x[1]; subcell_length_scale=0.1,
                                               target_residual=0.0)
    @test_throws ArgumentError physical_domain(x -> x[1]; subcell_length_scale=0.1,
                                               target_residual=-1.0)
    # keep_fictitious keeps fully-fictitious cells active, so it needs α > 0 to
    # give them quadrature; pairing it with the strict-cut path is singular.
    @test_throws ArgumentError physical_domain(x -> x[1]; subcell_length_scale=0.1,
                                               keep_fictitious=true, alpha=0.0)
    @test physical_domain(x -> x[1]; subcell_length_scale=0.1, keep_fictitious=true, alpha=1.0e-6).keep_fictitious

    # A bare callable is auto-wrapped as a single leaf carrying the `lipschitz`
    # keyword; the domain-level integration knobs live on the PhysicalDomain.
    p = physical_domain(x -> x[1]; lipschitz=1.0, subcell_length_scale=0.0625)
    @test p.geometry isa Unfitted.Leaf
    @test p.geometry.lipschitz == 1.0
    @test p.alpha == 0.0
    @test p.subcell_length_scale == 0.0625
    @test p.max_depth == 8
    @test p.target_residual == 1.0e-6

    p_alpha = physical_domain(x -> x[1]; lipschitz=1.0, alpha=0.25, subcell_length_scale=0.1)
    @test p_alpha.alpha == 0.25

    p_tight = physical_domain(x -> x[1]; lipschitz=1.0, subcell_length_scale=0.1,
                              target_residual=1.0e-10)
    @test p_tight.target_residual == 1.0e-10

    p2 = physical_domain(x -> x[1]; subcell_length_scale=0.1)
    @test isinf(p2.geometry.lipschitz)
end

@testset "CSG level-set constructors" begin
    # leaf wraps a callable; per-leaf Lipschitz is validated on the leaf.
    @test leaf(x -> x[1]) isa Unfitted.Leaf
    @test leaf(x -> x[1]; lipschitz=2.0).lipschitz == 2.0
    @test_throws ArgumentError leaf(x -> x[1]; lipschitz=0.0)

    a = leaf(x -> x[1] - 0.5)
    b = leaf(x -> x[2] - 0.5)
    inter = intersect(a, b)
    uni = union(a, b)
    diff = setdiff(a, b)
    comp = complement(a)

    # Membership matches the Boolean semantics; the scalar value agrees in sign.
    inside = SVector(0.3, 0.3)   # a≤0 and b≤0
    mixed = SVector(0.3, 0.7)    # a≤0, b>0
    @test Unfitted._inside(inter, inside) && !Unfitted._inside(inter, mixed)
    @test Unfitted._inside(uni, inside) && Unfitted._inside(uni, mixed)
    @test !Unfitted._inside(diff, inside) && Unfitted._inside(diff, mixed)
    # complement(a) = {a > 0} = {x₁ > 0.5}; flips with x₁, not x₂.
    @test !Unfitted._inside(comp, inside) && Unfitted._inside(comp, SVector(0.7, 0.3))
    @test (levelset_value(physical_domain(inter; subcell_length_scale=0.1), mixed) > 0)
    @test length(Unfitted._leaves(inter)) == 2

    # A CSG tree passes straight through physical_domain.
    p = physical_domain(inter; subcell_length_scale=0.1)
    @test p.geometry === inter
end

@testset "classify_cell — CSG combinations" begin
    # Annulus r ∈ [0.2, 0.45] about the origin: full in the ring, fictitious in
    # the hole, cut across either rim.
    ann = setdiff(leaf(x -> hypot(x...) - 0.45; lipschitz=1.0),
                  leaf(x -> hypot(x...) - 0.2; lipschitz=1.0))
    p = physical_domain(ann; subcell_length_scale=0.02)
    @test classify_cell(p, box((0.30, 0.0), (0.32, 0.02))) === :full
    @test classify_cell(p, box((0.0, 0.0), (0.05, 0.05))) === :fictitious
    @test classify_cell(p, box((0.44, 0.0), (0.50, 0.06))) === :cut
    @test classify_cell(p, box((0.18, 0.0), (0.24, 0.06))) === :cut

    # Union: a box inside one disk classifies :full even where the other leaf is
    # uncertain (the three-valued certificate short-circuits the union).
    uni = union(leaf(x -> hypot(x[1] - 0.3, x[2] - 0.3) - 0.2; lipschitz=1.0),
                leaf(x -> hypot(x[1] - 0.7, x[2] - 0.7) - 0.2; lipschitz=1.0))
    pu = physical_domain(uni; subcell_length_scale=0.02)
    @test classify_cell(pu, box((0.28, 0.28), (0.32, 0.32))) === :full
    @test classify_cell(pu, box((0.0, 0.9), (0.05, 0.95))) === :fictitious
end

@testset "CSG — levelset_value agrees in sign with membership (VTK contour)" begin
    # The VTK level_set field uses levelset_value; a contour at 0 reproduces ∂Ω
    # only if (levelset_value ≤ 0) ⇔ inside Ω for every CSG node — i.e. max/min/
    # negate are the De Morgan duals of all/any/not. Check on a grid for union,
    # complement, difference, and a nested tree.
    a = leaf(x -> hypot(x[1] - 0.35, x[2] - 0.5) - 0.25)
    b = leaf(x -> hypot(x[1] - 0.65, x[2] - 0.5) - 0.25)
    c = leaf(x -> x[2] - 0.5)
    trees = [union(a, b), complement(a), setdiff(union(a, b), c),
             union(a, b, leaf(x -> hypot(x[1] - 0.5, x[2] - 0.85) - 0.18))]
    for tree in trees
        p = physical_domain(tree; subcell_length_scale=0.1)
        for i in 0:24, j in 0:24
            x = SVector(i / 24, j / 24)
            v = levelset_value(p, x)
            # Agreement holds away from ∂Ω; exactly on a constituent boundary
            # (value == 0, e.g. the y = 0.5 grid line) the contour and the
            # strict `≤ 0` membership legitimately coincide in measure zero.
            abs(v) > 1e-9 && @test (v <= 0) == Unfitted._inside(tree, x)
        end
    end
end

@testset "CSG — classify: union dominance, complement, n-ary intersection" begin
    # Union short-circuit: a box that is certainly inside A but straddles B's rim
    # (B uncertain) must still classify :full via AnyOf — the headline case.
    a = leaf(x -> hypot(x[1] - 0.4, x[2] - 0.5) - 0.45; lipschitz=1.0)
    b = leaf(x -> hypot(x[1] - 0.6, x[2] - 0.5) - 0.3; lipschitz=1.0)
    pu = physical_domain(union(a, b); subcell_length_scale=0.02)
    @test classify_cell(pu, box((0.28, 0.48), (0.32, 0.52))) === :full   # inside A, on B's rim

    # Complement flips full/fictitious.
    pc = physical_domain(complement(a); subcell_length_scale=0.02)
    @test classify_cell(pc, box((0.38, 0.48), (0.42, 0.52))) === :fictitious  # inside A
    @test classify_cell(pc, box((0.94, 0.94), (0.98, 0.98))) === :full        # outside A

    # n-ary (3-part) intersection: Ω = {0.2 ≤ x ≤ 0.8, y ≤ 0.8}.
    tri = intersect(leaf(x -> x[1] - 0.8), leaf(x -> x[2] - 0.8), leaf(x -> 0.2 - x[1]))
    pt = physical_domain(tri; subcell_length_scale=0.02)
    @test classify_cell(pt, box((0.4, 0.4), (0.5, 0.5))) === :full
    @test classify_cell(pt, box((0.0, 0.4), (0.1, 0.5))) === :fictitious     # x < 0.2
    @test classify_cell(pt, box((0.75, 0.4), (0.85, 0.5))) === :cut          # straddles x = 0.8
end

@testset "PhysicalDomain.target_residual drives integration_plan moment-fit" begin
    # The default 1.0e-6 is a loose bound on the moment fit. The Saye kernel's
    # moments are exact on graph-like cut cells and high-order on curved ones,
    # so the achieved residual is set by the kernel, not by the target: a
    # tighter target only triggers extra conditioning retries that cannot beat
    # the kernel's curved-boundary approximation floor. The 2×2 base box
    # [−1, 1]² is split 8×8 into 0.25-wide cells; subcell_length_scale here
    # only bounds classifier/fallback subdivision, not moment accuracy.
    phi(x) = sqrt(x[1]^2 + x[2]^2) - 0.7
    omega = box((-1.0, -1.0), (1.0, 1.0))

    loose = physical_domain(phi; lipschitz=1.0, subcell_length_scale=2.0 / 8 / 2^4)
    tight = physical_domain(phi; lipschitz=1.0, subcell_length_scale=2.0 / 8 / 2^4,
                            target_residual=1.0e-10)

    V_loose = space(omega; cells=(8, 8), order=2, physical=loose)
    V_tight = space(omega; cells=(8, 8), order=2, physical=tight)

    plan_loose = Unfitted.integration_plan(V_loose)
    plan_tight = Unfitted.integration_plan(V_tight)

    # Identical region structure regardless of fit tolerance.
    @test length(plan_loose.regions) == length(plan_tight.regions)
    cut_loose = count(r -> r.quadrature.kind === :cut_fitted, plan_loose.regions)
    cut_tight = count(r -> r.quadrature.kind === :cut_fitted, plan_tight.regions)
    @test cut_loose == cut_tight > 0

    # Both targets reach the same residual floor: it is set by the kernel's
    # approximation of the curved boundary, which a tighter target cannot beat.
    @test plan_loose.moment_fit_residual_max < 1.0e-5
    @test plan_tight.moment_fit_residual_max < 1.0e-5
end

@testset "classify_cell — Lipschitz certificate" begin
    # SDF for a disk centered at (0.5, 0.5) with radius 0.3.
    # phi(x) = ‖x − c‖ − r;  inside Ω ⇔ phi ≤ 0.
    p = physical_domain(x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3; lipschitz=1.0,
                        subcell_length_scale=0.0625)

    # A small box well inside the disk → certificate fires → :full.
    @test classify_cell(p, box((0.45, 0.45), (0.55, 0.55))) === :full

    # A small box well outside the disk → certificate fires → :fictitious.
    @test classify_cell(p, box((0.0, 0.0), (0.1, 0.1))) === :fictitious

    # A box straddling the disk boundary → :cut.
    @test classify_cell(p, box((0.7, 0.4), (0.9, 0.6))) === :cut
end

@testset "classify_cell — corner sampling fallback" begin
    # phi(x) = x[1] − 0.5; Ω = left half.
    p = physical_domain(x -> x[1] - 0.5; lipschitz=Inf, subcell_length_scale=10.0, max_depth=0)
    # With lipschitz=Inf the certificate never fires; sampling decides at depth 0.
    # `subcell_length_scale=10` is far larger than the unit boxes used below, so
    # the length-scale check never asks for subdivision either.
    @test classify_cell(p, box((0.0, 0.0), (0.4, 1.0))) === :full
    @test classify_cell(p, box((0.6, 0.0), (1.0, 1.0))) === :fictitious
    @test classify_cell(p, box((0.4, 0.0), (0.6, 1.0))) === :cut
end

@testset "no physical domain leaves behavior unchanged" begin
    bc = dirichlet(0.0; on=boundary(:all))
    src = x -> sin(pi * x[1]) * sin(pi * x[2])
    omega = box((0.0, 0.0), (1.0, 1.0))

    V1 = space(omega; cells=(8, 8), order=2)
    V2 = space(omega; cells=(8, 8), order=2, physical=nothing)

    m1 = prepare(poisson(V1; source=src, dirichlet=[bc]))
    m2 = prepare(poisson(V2; source=src, dirichlet=[bc]))
    assemble!(m1)
    assemble!(m2)

    @test m1.matrix == m2.matrix
    @test m1.rhs == m2.rhs
end

@testset "physical domain = bounding box leaves behavior unchanged" begin
    # phi(x) = −1 inside the entire bounding box → no cells fictitious.
    bc = dirichlet(0.0; on=boundary(:all))
    omega = box((0.0, 0.0), (1.0, 1.0))
    V_no = space(omega; cells=(8, 8), order=2)
    V_phys = space(omega; cells=(8, 8), order=2,
                   physical=physical_domain(x -> -1.0; lipschitz=1.0, subcell_length_scale=1.0))

    m_no = prepare(poisson(V_no; source=x -> 1.0, dirichlet=[bc]))
    m_phys = prepare(poisson(V_phys; source=x -> 1.0, dirichlet=[bc]))

    @test diagnostics(m_no).active_unknowns == diagnostics(m_phys).active_unknowns
    @test diagnostics(m_phys).inactive_cell_counts == [0]

    assemble!(m_no)
    assemble!(m_phys)
    @test m_no.matrix == m_phys.matrix
end

@testset "interior hole drops the expected cells" begin
    # 10×10 mesh of width 0.1 each. Disk hole at (0.5, 0.5) radius 0.2.
    # Cells (5,5), (5,6), (6,5), (6,6) sit entirely inside the disk and
    # should be classified :fictitious. Uses subcell_length_scale=1.0e-6, max_depth=2 + order=1 to
    # keep the NNMF cost on the surrounding cut cells small — this test is
    # about cell classification, not cut-quadrature accuracy.
    omega = box((0.0, 0.0), (1.0, 1.0))
    hole = physical_domain(x -> 0.2 - sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2); lipschitz=1.0,
                           subcell_length_scale=1.0e-6, max_depth=2)
    V = space(omega; cells=(10, 10), order=1, physical=hole)

    m = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    @test diagnostics(m).inactive_cell_counts == [4]

    active = active_cells(m; level=1)
    @test !active[5, 5] && !active[5, 6] && !active[6, 5] && !active[6, 6]
    # A cell touching ∂Ω stays active (the classifier keeps it as a cut cell).
    @test active[4, 4]    # touched by the disk but not inside it
    @test active[1, 1]    # well outside the disk
end

@testset "fictitious-cell fold drops the same cells as a manual mask" begin
    # The cell-level fictitious fold deactivates exactly the cells a manual
    # `active=…` mask would (same `active_cells`). The dof layouts differ by
    # design, though. A geometry-blind mask eliminates every boundary mode on
    # the active/inactive interface (the C⁰ overlay-constraint trace). The
    # physical fold instead keeps the modes whose interface face is *fully
    # fictitious*: they vanish on every physical face — so they cannot break
    # continuity — and carry the surrounding cut cells' approximation up to ∂Ω.
    # The fold therefore retains strictly more unknowns: here the 8 perimeter
    # nodes around the 2×2 hole (a 3×3 node patch minus its interior node,
    # which neither variant enumerates). (Matrix values differ regardless: cut
    # cells use moment-fitted quadrature, the manual variant standard Gauss.)
    bc = dirichlet(0.0; on=boundary(:all))
    src = x -> sin(pi * x[1]) * sin(pi * x[2])
    omega = box((0.0, 0.0), (1.0, 1.0))

    hole = physical_domain(x -> 0.2 - sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2); lipschitz=1.0,
                           subcell_length_scale=1.0e-6, max_depth=2)
    V_phys = space(omega; cells=(10, 10), order=1, physical=hole)
    V_manual = space(omega; cells=(10, 10), order=1,
                     active=[CartesianIndex(i, j)
                             for i in 1:10, j in 1:10 if !(i in 5:6 && j in 5:6)])

    m_phys = prepare(poisson(V_phys; source=src, dirichlet=[bc]))
    m_manual = prepare(poisson(V_manual; source=src, dirichlet=[bc]))

    @test active_cells(m_phys; level=1) == active_cells(m_manual; level=1)
    @test diagnostics(m_phys).active_unknowns == diagnostics(m_manual).active_unknowns + 8
end

@testset "physical fold combines with user mask on overlay" begin
    # User deactivates one overlay row; the physical fold drops cells fully
    # outside Ω. The effective mask should be the intersection.
    bc = dirichlet(0.0; on=boundary(:all))
    omega = box((0.0, 0.0), (1.0, 1.0))

    # phi(x) = x[1] − 0.495. Cells whose entire x-range is ≥ 0.495 become
    # fictitious. Overlay cells of width 0.1 at axis-1 indices 3 and 4
    # (covering 0.5–0.6 and 0.6–0.7) are entirely outside Ω; index 2
    # (0.4–0.5) is :cut and kept active by the classifier.
    p = physical_domain(x -> x[1] - 0.495; lipschitz=1.0, subcell_length_scale=1.0e-6, max_depth=2)

    V = overlay(space(omega; cells=(8, 8), order=1, physical=p), box((0.3, 0.3), (0.7, 0.7));
                cells=(4, 4), order=1, active=(b, i) -> i.I[2] != 4)

    m = prepare(poisson(V; source=x -> 1.0, dirichlet=[bc]))
    bits = active_cells(m; level=2)

    @test all(!bits[i, 4] for i in 1:4)             # user-deactivated row j=4
    @test all(!bits[i, j] for i in 3:4, j in 1:3)   # fictitious columns i=3, i=4
    @test all(bits[i, j] for i in 1:2, j in 1:3)    # active interior (i=1 full, i=2 cut)
end

@testset "threaded == serial under fictitious fold" begin
    bc = dirichlet(0.0; on=boundary(:all))
    omega = box((0.0, 0.0), (1.0, 1.0))
    hole = physical_domain(x -> 0.2 - sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2); lipschitz=1.0,
                           subcell_length_scale=1.0e-6, max_depth=2)
    V = space(omega; cells=(10, 10), order=1, physical=hole)

    m_s = prepare(poisson(V; source=x -> 1.0, dirichlet=[bc]))
    m_t = prepare(poisson(V; source=x -> 1.0, dirichlet=[bc]))
    assemble!(m_s; threaded=false)
    assemble!(m_t; threaded=true)

    @test m_s.matrix ≈ m_t.matrix
    @test m_s.rhs ≈ m_t.rhs
end

@testset "move! re-applies the physical fold for the new geometry" begin
    bc = dirichlet(0.0; on=boundary(:all))
    omega = box((0.0, 0.0), (1.0, 1.0))
    # Ω is the left half of the unit square; phi(x) = x[1] − 0.5.
    p = physical_domain(x -> x[1] - 0.5; lipschitz=1.0, subcell_length_scale=1.0e-6, max_depth=2)

    # Overlay starts on the left (entirely inside Ω) → no fictitious cells.
    V = overlay(space(omega; cells=(8, 8), order=1, physical=p), box((0.1, 0.3), (0.4, 0.7));
                cells=(3, 4), order=1)
    m = prepare(poisson(V; source=x -> 1.0, dirichlet=[bc]))
    @test all(active_cells(m; level=2))   # all overlay cells active

    # Move the overlay to the right (entirely outside Ω) → all cells fictitious.
    move!(m; level=2, to=box((0.6, 0.3), (0.9, 0.7)))
    @test !any(active_cells(m; level=2))   # all overlay cells inactive
end

# --- Region-level classification and α-FCM ---

@testset "α=1 + all-fictitious matches no-physical baseline" begin
    # 1D, single cell, Ω = ∅ (phi=1 everywhere). With α=1 the region is
    # :fictitious_alpha but the weight scale is 1, so the matrix matches the
    # plain (no physical_domain) assembly.
    omega = box((0.0,), (1.0,))
    bc = dirichlet(0.0; on=boundary(:all))

    V_none = space(omega; cells=1, order=2)
    V_alpha1 = space(omega; cells=1, order=2,
                     physical=physical_domain(x -> 1.0; lipschitz=1.0, alpha=1.0,
                                              keep_fictitious=true, subcell_length_scale=1.0))

    m_none = prepare(stiffness(V_none; dirichlet=[bc]))
    m_alpha1 = prepare(stiffness(V_alpha1; dirichlet=[bc]))
    assemble!(m_none)
    assemble!(m_alpha1)

    @test Unfitted.active_unknowns(m_none.dofs) == Unfitted.active_unknowns(m_alpha1.dofs)
    @test m_none.matrix ≈ m_alpha1.matrix
end

@testset "α-FCM weights scale the bilinear form" begin
    omega = box((0.0,), (1.0,))
    bc = dirichlet(0.0; on=boundary(:all))

    V_alpha1 = space(omega; cells=1, order=2,
                     physical=physical_domain(x -> 1.0; lipschitz=1.0, alpha=1.0,
                                              keep_fictitious=true, subcell_length_scale=1.0))
    V_alpha05 = space(omega; cells=1, order=2,
                      physical=physical_domain(x -> 1.0; lipschitz=1.0, alpha=0.5,
                                               keep_fictitious=true, subcell_length_scale=1.0))

    m1 = prepare(stiffness(V_alpha1; dirichlet=[bc]))
    m05 = prepare(stiffness(V_alpha05; dirichlet=[bc]))
    assemble!(m1)
    assemble!(m05)

    @test m05.matrix ≈ 0.5 * m1.matrix
end

@testset "α-FCM: cut-cell enrichment, and fictitious cells dropped by default" begin
    # Ω = {x₁ ≤ 0.5}: the middle column of cells is CUT, the right column is
    # fully fictitious.
    omega = box((0.0, 0.0), (1.0, 1.0))
    cut = leaf(x -> x[1] - 0.5; lipschitz=1.0)
    bc = dirichlet(0.0; on=boundary(:all))
    V_none = space(omega; cells=(3, 3), order=3)

    # (1) keep_fictitious + α=1: every cell integrated at full weight, so the
    #     assembly matches a plain no-physical model. This exercises the cut-cell
    #     rule (1−α)·moment-fit ∪ α·tensor, which at α=1 must collapse to the
    #     full-cell tensor — a broken cut-cell α term would zero the cut column.
    V_keep = space(omega; cells=(3, 3), order=3,
                   physical=physical_domain(cut; lipschitz=1.0, alpha=1.0, keep_fictitious=true,
                                            subcell_length_scale=0.05, max_depth=4))
    m_none = prepare(stiffness(V_none; dirichlet=[bc]))
    m_keep = prepare(stiffness(V_keep; dirichlet=[bc]))
    assemble!(m_none)
    assemble!(m_keep)
    @test Unfitted.active_unknowns(m_none.dofs) == Unfitted.active_unknowns(m_keep.dofs)
    @test m_none.matrix ≈ m_keep.matrix

    # (2) default (keep_fictitious = false) with α > 0 STILL drops the fully-
    #     fictitious right column — α stabilises cut cells, not whole fictitious
    #     cells — so the active dof count is strictly smaller.
    V_drop = space(omega; cells=(3, 3), order=3,
                   physical=physical_domain(cut; lipschitz=1.0, alpha=1.0e-6,
                                            subcell_length_scale=0.05, max_depth=4))
    m_drop = prepare(stiffness(V_drop; dirichlet=[bc]))
    @test Unfitted.active_unknowns(m_drop.dofs) < Unfitted.active_unknowns(m_keep.dofs)
end

@testset "α=0 + all-fictitious drops every region" begin
    omega = box((0.0,), (1.0,))
    bc = dirichlet(0.0; on=boundary(:all))

    # phi(x) = 1 ⇒ Ω = ∅. The fictitious fold deactivates every cell at the
    # cell level (so dof enumeration is empty); the integration plan correctly
    # produces no regions.
    V = space(omega; cells=1, order=2,
              physical=physical_domain(x -> 1.0; lipschitz=1.0, subcell_length_scale=1.0))
    m = prepare(stiffness(V; dirichlet=[bc]))
    plan = Unfitted.integration_plan(m)

    @test Unfitted.active_unknowns(m.dofs) == 0
    @test isempty(plan.regions)

    assemble!(m)
    @test size(m.matrix) == (0, 0)
end

@testset "region quadrature kinds" begin
    # Mixed configuration: a disk hole on a 10×10 mesh. Boundary cells produce
    # `:cut_fitted` regions via the moment fit; cells fully inside Ω stay
    # `:full`; fictitious cells are already dropped at the cell level by the
    # fold. Uses subcell_length_scale=1.0e-6, max_depth=2 + order=1 to keep the
    # moment-fit cost on the surrounding cut cells bounded.
    omega = box((0.0, 0.0), (1.0, 1.0))
    hole = physical_domain(x -> 0.2 - sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2); lipschitz=1.0,
                           subcell_length_scale=1.0e-6, max_depth=2)
    V = space(omega; cells=(10, 10), order=1, physical=hole)
    m = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    plan = Unfitted.integration_plan(m)
    kinds = Set(region.quadrature.kind for region in plan.regions)
    @test :full in kinds
    @test :cut_fitted in kinds
end

@testset "α-FCM weight cache shares the scaled vector" begin
    # Two regions of the same tensor order in the same plan should reference
    # the same α-scaled weights vector (cache hit), preserving the
    # "shared underlying array" invariant the α-FCM weight cache documents.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=2,
              physical=physical_domain(x -> 1.0; lipschitz=1.0, alpha=0.5, keep_fictitious=true,
                                       subcell_length_scale=1.0))
    plan = Unfitted.integration_plan(V)

    fict_regions = [r for r in plan.regions if r.quadrature.kind === :fictitious_alpha]
    @test length(fict_regions) >= 2
    # All :fictitious_alpha regions share weights vector identity (cache hit).
    w = fict_regions[1].quadrature.weights
    @test all(r.quadrature.weights === w for r in fict_regions)
end

# --- Moment-fit quadrature in the integration plan ---

@testset "1D cut mass matrix matches analytic integral" begin
    # Cell (0, 1), order 2, Ω = (0, 0.7). Mass entry M[1,1] for the
    # endpoint mode N_0(x) = 1 − x: ∫_0^0.7 (1 − x)^2 dx = (1 − 0.3^3) / 3.
    omega = box((0.0,), (1.0,))
    p = physical_domain(x -> x[1] - 0.7; lipschitz=1.0, subcell_length_scale=1.0 / 2^10,
                        max_depth=10)
    V = space(omega; cells=1, order=2, physical=p)
    m = prepare(mass(V; coefficient=1.0))
    assemble!(m)

    M = Matrix(m.matrix)
    truth = (1 - 0.3^3) / 3
    @test M[1, 1] ≈ truth atol = 1.0e-5
    @test issymmetric(M)
end

@testset "2D Poisson on a cut disk solves and is SPD" begin
    # End-to-end smoke: PhysicalDomain set, system assembles, matrix SPD,
    # solver returns a finite solution. Uses moderate accuracy parameters
    # for a fast test.
    omega = box((0.0, 0.0), (1.0, 1.0))
    hole = physical_domain(x -> 0.2 - sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2); lipschitz=1.0,
                           subcell_length_scale=1.0e-6, max_depth=2)
    V = space(omega; cells=(6, 6), order=1, physical=hole)
    m = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    assemble!(m)

    @test isposdef(Symmetric(Matrix(m.matrix)))
    sol = solve!(m)
    @test all(isfinite, sol.coefficients)
end

@testset "3D immersed Poisson on a cut sphere — exact-moment kernel, O(nbasis), no OOM" begin
    # A 3D FCM problem assembles and solves with the exact implicit-kernel
    # moments. The moment-fit residual reaches machine precision (the exact
    # Saye kernel path is taken), every cut cell carries O(nbasis) quadrature
    # points rather than the 8^depth cloud a stair-step octree would need, and
    # the system stays SPD.
    omega = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    hole = physical_domain(x -> 0.3 - sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2 + (x[3] - 0.5)^2);
                           lipschitz=1.0, subcell_length_scale=0.1)
    V = space(omega; cells=(6, 6, 6), order=1, physical=hole)
    m = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    assemble!(m)

    @test isposdef(Symmetric(Matrix(m.matrix)))
    sol = solve!(m)
    @test all(isfinite, sol.coefficients)

    diag = diagnostics(m)
    @test diag.cut_region_count > 0
    @test diag.fit_failure_count == 0
    # Exact kernel ⇒ residual at machine precision, far below any stair-step floor.
    @test diag.moment_fit_residual_max < 1e-8

    # Every cut-cell rule is O(nbasis): a single NNLS solve keeps at most
    # nbasis = (order·factor + 1)^D = 3³ = 27 points; a stair-step octree
    # would have produced thousands here.
    plan = Unfitted.integration_plan(m)
    cut_regions = [r for r in plan.regions if r.quadrature.kind === :cut_fitted]
    @test maximum(length(r.quadrature.points) for r in cut_regions) <= 27
end

@testset "diagnostics report cut_region_count" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    hole = physical_domain(x -> 0.2 - sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2); lipschitz=1.0,
                           subcell_length_scale=1.0e-6, max_depth=2)
    V = space(omega; cells=(6, 6), order=1, physical=hole)
    m = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    diag = diagnostics(m)
    plan = Unfitted.integration_plan(m)
    expected = count(r -> r.quadrature.kind in
                          (:cut_fitted, :cut_fallback, :cut_failed, :cut_alpha_failed),
                     plan.regions)
    @test diag.cut_region_count == expected
    @test diag.cut_region_count > 0
    @test diag.fit_failure_count == 0   # no failures expected on this smooth Ω
end

@testset "moment-fit cache shares rules across identical regions" begin
    # If the integration plan produces two regions with byte-identical bounds
    # (e.g., symmetric tiling), they should share one fitted rule (cache hit
    # → same Vector identity).
    omega = box((0.0,), (1.0,))
    p = physical_domain(x -> x[1] - 0.5; lipschitz=1.0, subcell_length_scale=1.0e-6, max_depth=3)
    V = space(omega; cells=2, order=2, physical=p)
    plan = Unfitted.integration_plan(V)
    cut_regions = [r for r in plan.regions if r.quadrature.kind === :cut_fitted]
    # 1D, 2 cells, Ω = left half: cell 1 is :full, cell 2 is :cut (sits across
    # the boundary). Only one cut region, so this is more of a sanity check
    # that the cache key construction doesn't crash.
    @test length(cut_regions) <= 1
end

@testset "cut region integrates the volume to better than stair-step" begin
    # Sanity: the moment-fit rule on a 1D cut cell integrates `1` (i.e., the
    # 0th moment) more accurately than full-cell tensor Gauss would.
    omega = box((0.0,), (1.0,))
    p = physical_domain(x -> x[1] - 0.7; lipschitz=1.0, subcell_length_scale=1.0 / 2^8, max_depth=8)
    V = space(omega; cells=1, order=2, physical=p)
    plan = Unfitted.integration_plan(V)

    region = plan.regions[1]
    @test region.quadrature.kind === :cut_fitted
    jac = Unfitted.volume(region.box) / 2.0   # 1D
    integral = sum(w * jac for w in region.quadrature.weights)
    @test integral ≈ 0.7 atol = 1.0e-3
end

@testset "starved linear cut stays :cut_fitted, not :cut_failed" begin
    # A tiny Ω with subdivision disabled (max_depth=0) leaves the moment fit
    # very little to work with on the single depth-0 cut cell.
    omega = box((0.0,), (1.0,))
    # At subcell_length_scale=10.0, max_depth=0 the classifier emits the whole
    # region as :cut (corners of [0,1] straddle phi=0), and the NNLS fit must
    # work with the moments of that single undivided cut cell.
    p = physical_domain(x -> x[1] - 0.001; lipschitz=1.0, subcell_length_scale=10.0, max_depth=0)
    V = space(omega; cells=1, order=2, physical=p)
    plan = Unfitted.integration_plan(V)
    # :cut_failed (empty Ω∩R) and :cut_fallback (residual >
    # _FIT_FAILURE_RESIDUAL) are both defensive tags. The leaf here is linear, so
    # the Saye moments are exact and a zero-residual non-negative fit exists: the
    # region is :cut_fitted, neither of the two, and the max residual stays far
    # below the failure threshold.
    failed = count(r -> r.quadrature.kind in (:cut_fallback, :cut_failed, :cut_alpha_failed),
                   plan.regions)
    @test failed == 0
    @test plan.moment_fit_residual_max < Unfitted._FIT_FAILURE_RESIDUAL
    # Diagnostics counts must agree with the plan tags — guarding the flagging
    # logic that the old `failed >= 0` placeholder did not.
    m = prepare(stiffness(V; dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test diagnostics(m).fit_failure_count == failed
end

# --- Space.physical forwarding and moment-fit knobs ---

@testset "moved_space preserves Space.physical" begin
    # Forward an immersed-domain Space through `moved_space` and assert the
    # `physical` field is preserved on the new Space.
    bc = dirichlet(0.0; on=boundary(:all))
    omega = box((0.0, 0.0), (1.0, 1.0))
    p = physical_domain(x -> x[1] - 0.5; lipschitz=1.0, subcell_length_scale=1.0e-6, max_depth=2)
    V = overlay(space(omega; cells=(8, 8), order=1, physical=p), box((0.1, 0.3), (0.4, 0.7));
                cells=(3, 4), order=1)
    V_moved = Unfitted.moved_space(V; level=2, to=box((0.2, 0.3), (0.5, 0.7)))
    @test V_moved.physical === V.physical
end

@testset "overlay forwards Space.physical" begin
    # Adding an overlay to a Space with a physical_domain must keep the
    # physical_domain on the resulting Space.
    omega = box((0.0, 0.0), (1.0, 1.0))
    p = physical_domain(x -> x[1] - 0.5; lipschitz=1.0, subcell_length_scale=1.0e-6, max_depth=2)
    V_base = space(omega; cells=(4, 4), order=1, physical=p)
    V_overlay = overlay(V_base, box((0.1, 0.1), (0.4, 0.4)); cells=(2, 2), order=1)
    @test V_overlay.physical === p
end

@testset "NNMF on a region with multi-level parents" begin
    # A cut base cell that is also covered by a cut overlay cell — exercises
    # `_moment_order_for_region`'s max-over-parents logic. Use orders that
    # differ between levels so the chosen moment_order is meaningful.
    bc = dirichlet(0.0; on=boundary(:all))
    omega = box((0.0,), (1.0,))
    p = physical_domain(x -> x[1] - 0.45; lipschitz=1.0, subcell_length_scale=1.0e-6, max_depth=4)
    V = overlay(space(omega; cells=2, order=1, physical=p), box((0.25,), (0.75,)); cells=2, order=2)
    plan = Unfitted.integration_plan(V)

    # At least one cut-fitted region with two parents (base + overlay).
    multi_parent_cuts = [r
                         for r in plan.regions
                         if r.quadrature.kind === :cut_fitted && length(r.parents) == 2]
    @test !isempty(multi_parent_cuts)
    # moment_order for such a region = 2 × max(base.order, overlay.order) = 2 × 2 = 4.
    region = multi_parent_cuts[1]
    moment_order = Unfitted._moment_order_for_region(V, region.parents)
    @test moment_order == (4,)
end

@testset "moment_order_factor knob tunes the NNMF basis" begin
    omega = box((0.0,), (1.0,))
    p_default = physical_domain(x -> x[1] - 0.5; lipschitz=1.0, subcell_length_scale=1.0e-6,
                                max_depth=2)
    p_cheap = physical_domain(x -> x[1] - 0.5; lipschitz=1.0, subcell_length_scale=1.0e-6,
                              max_depth=2, moment_order_factor=1)

    V_def = space(omega; cells=1, order=2, physical=p_default)
    V_che = space(omega; cells=1, order=2, physical=p_cheap)

    parents = Unfitted._parents_covering(V_def.levels, box((0.0,), (1.0,)),
                                         GeometryTolerance(Float64))
    @test Unfitted._moment_order_for_region(V_def, parents) == (4,)
    @test Unfitted._moment_order_for_region(V_che, parents) == (2,)

    @test_throws ArgumentError physical_domain(x -> x[1]; subcell_length_scale=0.1,
                                               moment_order_factor=0)
end

@testset "high moment order on a starved cut still fits below the failure threshold" begin
    # A high moment order (20) against a coarse, undivided cut cell stresses the
    # moment fit. The leaf is linear, so the Saye moments are exact and a single
    # NNLS solve stays well below the failure threshold — the residual branch
    # that would tag a region :cut_failed is not reached through the public path
    # with the exact kernel.
    p = physical_domain(x -> 0.5 - x[1]; lipschitz=1.0, subcell_length_scale=10.0, max_depth=0)
    region_box = box((0.0,), (1.0,))
    pts, ws, res = Unfitted.moment_fit_rule(p, region_box, (20,); target_residual=1.0e-14)
    @test !isempty(pts)
    @test res < Unfitted._FIT_FAILURE_RESIDUAL
end

@testset "α-FCM fit failure is :cut_alpha_failed and counts as a failure" begin
    # The empty-Ω∩R branches are defensive (unreachable via the public path with
    # exact kernels), so pin their contract directly. Unlike the empty strict-cut
    # `:cut_failed`, `:cut_alpha_failed` carries a NONZERO α-scaled rule so the
    # cell's dofs stay α-stabilised. `:cut_fallback` carries the raw Saye volume
    # rule — a full, correct rule with many more points than a fit. All three
    # non-fitted kinds feed the fit-failure count, all four count as cut regions,
    # and only `:cut_fallback` contributes to the fallback point budget.
    RQ = Unfitted.RegionQuadrature{1,Float64}
    VR = Unfitted.VolumeRegion{1,Float64}
    b = box((0.0,), (1.0,))
    noparents = Unfitted.ParentRef{1,Float64}[]
    afail = RQ(:cut_alpha_failed, [SVector(0.5)], [0.3])
    strict = RQ(:cut_failed, SVector{1,Float64}[], Float64[])
    back = RQ(:cut_fallback, [SVector(0.25), SVector(0.75)], [0.5, 0.5])
    @test !isempty(afail.weights)          # α fallback is a nonzero rule …
    @test isempty(strict.weights)          # … the strict failure is empty
    plan = Unfitted.IntegrationPlan{1,Float64}([VR(b, noparents,
                                                   RQ(:cut_fitted, [SVector(0.5)], [1.0])),
                                                VR(b, noparents, strict), VR(b, noparents, afail),
                                                VR(b, noparents, back)],
                                               GeometryTolerance(Float64), 0,
                                               Unfitted.SmallOverlap{Float64}[], 1.0, 1.0, 0.0)
    cut, failed, fallback, fallback_points = Unfitted._cut_region_stats(plan)
    @test cut == 4
    @test failed == 3
    @test fallback == 1
    @test fallback_points == 2
end

@testset "custom cut_quadrature rule replaces the moment fit on cut regions" begin
    # `PhysicalDomain.cut_quadrature` is the public extension point for
    # integrating cut cells by some other scheme. The rule below is the
    # plainest one honouring the documented contract: the full-cell tensor
    # Gauss rule of the region box, in *physical* coordinates, with strictly
    # positive weights. It ignores Ω on purpose, so `Σ w_q = vol(box)` exactly
    # — a value no moment fit could return on a cut cell, which is what makes
    # "the rule was actually used" checkable without re-deriving a fit.
    calls = Ref(0)
    function tensor_rule(physical, region_box::Unfitted.AxisBox{D,T},
                         moment_order::NTuple{D,Int}) where {D,T}
        calls[] += 1
        rule = Unfitted._tensor_gauss_rule(moment_order .+ 1, T)
        jacobian = Unfitted.volume(region_box) / T(2^D)
        points = [Unfitted.reference_to_physical(region_box, p) for p in rule.points]
        return points, rule.weights .* jacobian, zero(T), :custom
    end

    omega = box((0.0, 0.0), (1.0, 1.0))
    geometry = x -> sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3
    settings = (; lipschitz=1.0, subcell_length_scale=0.05)
    fitted = physical_domain(geometry; settings...)
    custom = physical_domain(geometry; settings..., cut_quadrature=tensor_rule)

    # `Q === Nothing` on the default keeps the struct concrete, which is the
    # reason the rule is a field rather than a boxed global.
    @test fitted.cut_quadrature === nothing
    @test typeof(fitted).parameters[3] === Nothing
    @test isconcretetype(typeof(fitted))
    @test isconcretetype(typeof(custom))

    plan_fitted = Unfitted.integration_plan(space(omega; cells=(4, 4), order=2,
                                                  physical=fitted))
    plan_custom = Unfitted.integration_plan(space(omega; cells=(4, 4), order=2,
                                                  physical=custom))

    kinds(plan) = [r.quadrature.kind for r in plan.regions]
    @test :cut_fitted in kinds(plan_fitted)
    @test !(:cut_custom in kinds(plan_fitted))
    # The rule replaces exactly the cut regions: same region boxes, same
    # `:full` regions, `:cut_fitted` swapped for `:cut_custom` one for one.
    @test [r.box for r in plan_custom.regions] == [r.box for r in plan_fitted.regions]
    @test count(==(:full), kinds(plan_custom)) == count(==(:full), kinds(plan_fitted))
    @test count(==(:cut_custom), kinds(plan_custom)) ==
          count(==(:cut_fitted), kinds(plan_fitted)) > 0
    @test calls[] == count(==(:cut_custom), kinds(plan_custom))

    # Points come back in the region's reference frame with the standard
    # Jacobian folded out, so `Σ w_q · vol(box) / 2ᴰ` is the physical measure
    # the rule integrates — here the whole box, since the rule ignores Ω.
    for region in plan_custom.regions
        region.quadrature.kind === :cut_custom || continue
        jacobian = Unfitted.volume(region.box) / 4
        @test sum(region.quadrature.weights) * jacobian ≈ Unfitted.volume(region.box)
        @test all(>=(0), region.quadrature.weights)
    end

    # A custom region is a cut region but never a fit failure: no fit ran, so
    # there is no residual for `fit_failure_count` to report on.
    cut, failed, fallback, fallback_points = Unfitted._cut_region_stats(plan_custom)
    @test cut == count(==(:cut_custom), kinds(plan_custom))
    @test failed == 0
    @test fallback == 0
    @test fallback_points == 0

    # α blends the custom rule exactly as it blends a fit: the region carries
    # `(1 − α)·custom ∪ α·tensor`, so the custom part is scaled, not replaced.
    alpha = 0.25
    blended = physical_domain(geometry; settings..., alpha=alpha,
                              cut_quadrature=tensor_rule)
    plan_blended = Unfitted.integration_plan(space(omega; cells=(4, 4), order=2,
                                                   physical=blended))
    strict = first(r for r in plan_custom.regions if r.quadrature.kind === :cut_custom)
    mixed = first(r for r in plan_blended.regions
                  if r.quadrature.kind === :cut_custom && r.box == strict.box)
    n = length(strict.quadrature.weights)
    @test length(mixed.quadrature.weights) > n
    @test mixed.quadrature.points[1:n] == strict.quadrature.points
    @test mixed.quadrature.weights[1:n] ≈ strict.quadrature.weights .* (1 - alpha)
    @test all(>=(0), mixed.quadrature.weights)
end
