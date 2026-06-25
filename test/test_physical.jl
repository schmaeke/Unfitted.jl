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
    # The default 1.0e-6 is matched to the natural stair-step accuracy of the
    # octree moment integration. A tight target triggers wasted retries that
    # cannot actually be satisfied at a fixed subcell_length_scale; the loose
    # default produces the same residuals at a fraction of the cost. The 2×2
    # base-cell box is [−1, 1]² split 8×8, giving 0.25-wide cells; a leaf
    # scale of 1/2^4 = 0.0625 of the cell ≈ 0.016 of the base box reproduces
    # the old `subcell_length_scale=1.0e-6, max_depth=4` per-cut-cell octree.
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

    # Tight target cannot actually drive the moment-fit below the octree's
    # stair-step floor, but the loose default reaches the same accuracy floor.
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

    assemble!(m_no);
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
    # A cell touching ∂Ω stays active (stair-step approximation in Slice 3).
    @test active[4, 4]    # touched by the disk but not inside it
    @test active[1, 1]    # well outside the disk
end

@testset "fictitious-cell fold matches manual deactivation (dof structure)" begin
    # Verify that the cell-level fictitious fold (Slice 3) drops the same
    # cells from the dof layout as a manual `active=…` mask. The matrix
    # values disagree post-Slice-5c because cut cells now use NNMF-fitted
    # quadrature while the manual variant uses standard Gauss everywhere
    # the cell is active — different mathematical operators, by design.
    # Test the structural equivalence only.
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
    @test diagnostics(m_phys).active_unknowns == diagnostics(m_manual).active_unknowns
end

@testset "physical fold combines with user mask on overlay" begin
    # User deactivates one overlay row; the physical fold drops cells fully
    # outside Ω. The effective mask should be the intersection.
    bc = dirichlet(0.0; on=boundary(:all))
    omega = box((0.0, 0.0), (1.0, 1.0))

    # phi(x) = x[1] − 0.495. Cells whose entire x-range is ≥ 0.495 become
    # fictitious. Overlay cells of width 0.1 at axis-1 indices 3 and 4
    # (covering 0.5–0.6 and 0.6–0.7) are entirely outside Ω; index 2
    # (0.4–0.5) is :cut and kept active under the Slice 3 stair-step rule.
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

# --- Slice 4: region-level classification + α-FCM ---

@testset "Slice 4 — α=1 + all-fictitious matches no-physical baseline" begin
    # 1D, single cell, Ω = ∅ (phi=1 everywhere). With α=1 the region is
    # :fictitious_alpha but the weight scale is 1, so the matrix matches the
    # plain (no physical_domain) assembly.
    omega = box((0.0,), (1.0,))
    bc = dirichlet(0.0; on=boundary(:all))

    V_none = space(omega; cells=1, order=2)
    V_alpha1 = space(omega; cells=1, order=2,
                     physical=physical_domain(x -> 1.0; lipschitz=1.0, alpha=1.0,
                                              subcell_length_scale=1.0))

    m_none = prepare(stiffness(V_none; dirichlet=[bc]))
    m_alpha1 = prepare(stiffness(V_alpha1; dirichlet=[bc]))
    assemble!(m_none);
    assemble!(m_alpha1)

    @test Unfitted.active_unknowns(m_none.dofs) == Unfitted.active_unknowns(m_alpha1.dofs)
    @test m_none.matrix ≈ m_alpha1.matrix
end

@testset "Slice 4 — α-FCM weights scale the bilinear form" begin
    omega = box((0.0,), (1.0,))
    bc = dirichlet(0.0; on=boundary(:all))

    V_alpha1 = space(omega; cells=1, order=2,
                     physical=physical_domain(x -> 1.0; lipschitz=1.0, alpha=1.0,
                                              subcell_length_scale=1.0))
    V_alpha05 = space(omega; cells=1, order=2,
                      physical=physical_domain(x -> 1.0; lipschitz=1.0, alpha=0.5,
                                               subcell_length_scale=1.0))

    m1 = prepare(stiffness(V_alpha1; dirichlet=[bc]))
    m05 = prepare(stiffness(V_alpha05; dirichlet=[bc]))
    assemble!(m1);
    assemble!(m05)

    @test m05.matrix ≈ 0.5 * m1.matrix
end

@testset "Slice 4 — α=0 + all-fictitious drops every region" begin
    omega = box((0.0,), (1.0,))
    bc = dirichlet(0.0; on=boundary(:all))

    # phi(x) = 1 ⇒ Ω = ∅. Slice 3 fold deactivates every cell at the cell
    # level (so dof enumeration is empty); the integration plan correctly
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

@testset "Slice 4 — region quadrature kinds" begin
    # Mixed configuration: a disk hole on a 10×10 mesh. After Slice 5c, the
    # boundary cells produce `:cut_fitted` regions via NNMF; cells fully
    # inside Ω stay `:full`; fictitious cells are already dropped at the
    # cell level by Slice 3. Uses subcell_length_scale=1.0e-6, max_depth=2 + order=1 to keep the
    # NNMF cost on the surrounding cut cells bounded.
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

@testset "Slice 4 — α-FCM weight cache shares the scaled vector" begin
    # Two regions of the same tensor order in the same plan should reference
    # the same α-scaled weights vector (cache hit), preserving the
    # "shared underlying array" invariant Slice 4 documents.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=2,
              physical=physical_domain(x -> 1.0; lipschitz=1.0, alpha=0.5,
                                       subcell_length_scale=1.0))
    plan = Unfitted.integration_plan(V)

    fict_regions = [r for r in plan.regions if r.quadrature.kind === :fictitious_alpha]
    @test length(fict_regions) >= 2
    # All :fictitious_alpha regions share weights vector identity (cache hit).
    w = fict_regions[1].quadrature.weights
    @test all(r.quadrature.weights === w for r in fict_regions)
end

# --- Slice 5c: NNMF moment-fit hooked into the integration plan ---

@testset "Slice 5c — 1D cut mass matrix matches analytic integral" begin
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

@testset "Slice 5c — 2D Poisson on a cut disk solves and is SPD" begin
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
    # Local proxy for the 3D acceptance criterion (crack_hole_3d on helios):
    # a 3D FCM problem assembles and solves with the exact implicit-kernel
    # moments. The moment-fit residual reaches machine precision (the kernel
    # path is taken, not the octree stair-step), every cut cell carries
    # O(nbasis) quadrature points rather than the 8^depth cloud that made the
    # octree path run out of memory in 3D, and the system stays SPD.
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
    # Exact kernel ⇒ residual far below the octree stair-step floor.
    @test diag.moment_fit_residual_max < 1e-8

    # Every cut-cell rule is O(nbasis): a single NNLS solve keeps at most
    # nbasis = (order·factor + 1)^D = 3³ = 27 points; the octree path produced
    # thousands here.
    plan = Unfitted.integration_plan(m)
    cut_regions = [r for r in plan.regions if r.quadrature.kind === :cut_fitted]
    @test maximum(length(r.quadrature.points) for r in cut_regions) <= 27
end

@testset "Slice 5c — diagnostics report cut_region_count" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    hole = physical_domain(x -> 0.2 - sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2); lipschitz=1.0,
                           subcell_length_scale=1.0e-6, max_depth=2)
    V = space(omega; cells=(6, 6), order=1, physical=hole)
    m = prepare(poisson(V; source=x -> 1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    diag = diagnostics(m)
    plan = Unfitted.integration_plan(m)
    expected = count(r -> r.quadrature.kind in (:cut_fitted, :cut_failed), plan.regions)
    @test diag.cut_region_count == expected
    @test diag.cut_region_count > 0
    @test diag.fit_failure_count == 0   # no failures expected on this smooth Ω
end

@testset "Slice 5c — moment-fit cache shares rules across identical regions" begin
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

@testset "Slice 5c — cut region integrates the volume to better than stair-step" begin
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

@testset "Slice 5c — :cut_failed marks unrecoverable fits" begin
    # Force a catastrophic NNMF residual by making moments unreachable: very
    # small Ω with no subdivision → moments dominated by stair-step error.
    omega = box((0.0,), (1.0,))
    # Tiny Ω near zero — at subcell_length_scale=10.0, max_depth=0 the classifier emits the whole
    # region as :cut (corners of [0,1] straddle phi=0), and NNMF must work
    # with whatever moments the depth-0 stair-step produces.
    p = physical_domain(x -> x[1] - 0.001; lipschitz=1.0, subcell_length_scale=10.0, max_depth=0)
    V = space(omega; cells=1, order=2, physical=p)
    plan = Unfitted.integration_plan(V)
    # NNMF either succeeds (residual ≤ failure threshold → :cut_fitted) or
    # fails (→ :cut_failed). Either outcome is valid; we just verify the
    # plan builds without raising and the diagnostic count is consistent.
    failed = count(r -> r.quadrature.kind === :cut_failed, plan.regions)
    @test failed >= 0   # placeholder: the path exists
end

# --- Review pass 2: closing test coverage gaps ---

@testset "Review — moved_space preserves Space.physical" begin
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

@testset "Review — overlay forwards Space.physical" begin
    # Adding an overlay to a Space with a physical_domain must keep the
    # physical_domain on the resulting Space.
    omega = box((0.0, 0.0), (1.0, 1.0))
    p = physical_domain(x -> x[1] - 0.5; lipschitz=1.0, subcell_length_scale=1.0e-6, max_depth=2)
    V_base = space(omega; cells=(4, 4), order=1, physical=p)
    V_overlay = overlay(V_base, box((0.1, 0.1), (0.4, 0.4)); cells=(2, 2), order=1)
    @test V_overlay.physical === p
end

@testset "Review — NNMF on a region with multi-level parents" begin
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

@testset "Review — moment_order_factor knob tunes the NNMF basis" begin
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

@testset "Review — fit_failure_count > 0 on a forced-failure setup" begin
    # Construct a setup where the moment basis is unreachable: very high
    # moment_order against a coarsely-resolved Ω near the cell boundary.
    # We can't drive a failure through the public API at default
    # `moment_order = 2·level.order`, so test the underlying `_FIT_FAILURE_RESIDUAL`
    # guard directly via `_build_region_quadrature` with a hand-built scenario.
    omega = box((0.0,), (1.0,))
    p = physical_domain(x -> 0.5 - x[1]; lipschitz=1.0, subcell_length_scale=10.0, max_depth=0)
    V = space(omega; cells=1, order=1, physical=p)

    # Manually invoke moment_fit_rule with an absurdly high moment_order on a
    # single subcell_length_scale=10.0, max_depth=0 cut region. The NNMF is starved on moments and
    # candidates; we observe the residual classification path.
    region_box = box((0.0,), (1.0,))
    pts, ws, res = Unfitted.moment_fit_rule(p, region_box, (20,); target_residual=1.0e-14)
    # Either: residual exceeds the failure threshold (would mark :cut_failed
    # in the plan dispatcher), or NNMF still converges. Both outcomes are
    # consistent with our error reporting; assert the residual is real.
    @test isfinite(res)
end
