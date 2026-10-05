using StaticArrays
using LinearAlgebra
using SparseArrays

@testset "mask normalization" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    mesh = Unfitted.CartesianMesh(omega; cells=(4, 4))

    @test Unfitted._normalize_mask(nothing, mesh) === nothing

    bits = trues(4, 4)
    bits[1, 1] = false
    mask = Unfitted._normalize_mask(bits, mesh)
    @test mask isa Unfitted.LevelMask{2}
    @test !mask.on[1, 1]
    @test mask.on[2, 2]

    # Pre-built LevelMask passes through as a copy (caller mutations of the
    # original BitArray must not leak into the model).
    passed = Unfitted._normalize_mask(mask, mesh)
    @test passed isa Unfitted.LevelMask{2}
    @test passed.on == mask.on
    @test passed.on !== mask.on

    pred_mask = Unfitted._normalize_mask((b, i) -> b.lower[1] < 0.5, mesh)
    @test pred_mask isa Unfitted.LevelMask{2}
    @test pred_mask.on[1, 1] && pred_mask.on[1, 2]
    @test !pred_mask.on[3, 1] && !pred_mask.on[3, 2]

    iter_mask = Unfitted._normalize_mask([CartesianIndex(1, 1), CartesianIndex(2, 2)], mesh)
    @test iter_mask.on[1, 1] && iter_mask.on[2, 2]
    @test !iter_mask.on[1, 2] && !iter_mask.on[2, 1]

    @test_throws DimensionMismatch Unfitted._normalize_mask(trues(3, 3), mesh)
    @test_throws DimensionMismatch Unfitted._normalize_mask(Unfitted.LevelMask{2}(trues(3, 3)),
                                                            mesh)
    @test_throws ArgumentError Unfitted._normalize_mask([CartesianIndex(5, 5)], mesh)
    @test_throws ArgumentError Unfitted._normalize_mask([(1, 1)], mesh)
end

@testset "is_active predicate" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    mesh = Unfitted.CartesianMesh(omega; cells=(2, 2))
    mask = Unfitted._normalize_mask([CartesianIndex(1, 1)], mesh)

    @test Unfitted.is_active(nothing, CartesianIndex(1, 1))
    @test Unfitted.is_active(mask, CartesianIndex(1, 1))
    @test !Unfitted.is_active(mask, CartesianIndex(2, 2))
end

# Standard "safe" overlay configuration used across tests: base order=2 with an
# overlay box misaligned with the base mesh and overlay order=3, avoiding the
# rank-deficient case of a same-order overlay whose cells exactly coincide with
# base cells.
const _OMEGA = box((0.0, 0.0), (1.0, 1.0))
function _safe_overlay(; overlay_cells=(4, 4), active=nothing)
    overlay(space(_OMEGA; cells=(8, 8), order=2), box((0.3, 0.3), (0.7, 0.7)); cells=overlay_cells,
            order=3, active=active)
end

@testset "no-mask path identical to today" begin
    bc = dirichlet(0.0; on=boundary(:all))
    src = x -> sin(pi * x[1]) * sin(pi * x[2])

    V1 = _safe_overlay()
    V2 = _safe_overlay(active=nothing)

    m1 = prepare(poisson(V1; source=src, dirichlet=[bc]))
    m2 = prepare(poisson(V2; source=src, dirichlet=[bc]))
    assemble!(m1)
    assemble!(m2)
    sol1 = solve!(m1)
    sol2 = solve!(m2)

    @test Unfitted.active_unknowns(m1.dofs) == Unfitted.active_unknowns(m2.dofs)
    @test sol1.coefficients == sol2.coefficients
end

@testset "fully-deactivated overlay equals no overlay" begin
    bc = dirichlet(0.0; on=boundary(:all))
    src = x -> sin(pi * x[1]) * sin(pi * x[2])

    V_off = _safe_overlay(active=falses(4, 4))
    V_no = space(_OMEGA; cells=(8, 8), order=2)

    m_off = prepare(poisson(V_off; source=src, dirichlet=[bc]))
    m_no = prepare(poisson(V_no; source=src, dirichlet=[bc]))
    assemble!(m_off)
    assemble!(m_no)
    sol_off = solve!(m_off)
    sol_no = solve!(m_no)

    @test Unfitted.active_unknowns(m_off.dofs) == Unfitted.active_unknowns(m_no.dofs)
    @test sol_off.coefficients ≈ sol_no.coefficients atol = 1e-12
end

@testset "half-mask matches geometrically smaller overlay" begin
    # Partial: overlay (0.3,0.7)x(0.3,0.7) with cells=(4,4), order=3; cells with
    # axis-1 index <= 2 active -> active region is (0.3,0.5)x(0.3,0.7) with 2x4
    # cells of width 0.1.
    # Reference: overlay (0.3,0.5)x(0.3,0.7) with cells=(2,4), order=3 of width 0.1.
    # Both overlays are misaligned with the (1/8-cell-width) base mesh.
    bc = dirichlet(0.0; on=boundary(:all))
    src = x -> sin(pi * x[1]) * sin(pi * x[2])

    active_cells = [CartesianIndex(i, j) for i in 1:2, j in 1:4]
    V_partial = overlay(space(_OMEGA; cells=(8, 8), order=2), box((0.3, 0.3), (0.7, 0.7));
                        cells=(4, 4), order=3, active=active_cells)
    V_small = overlay(space(_OMEGA; cells=(8, 8), order=2), box((0.3, 0.3), (0.5, 0.7));
                      cells=(2, 4), order=3)

    m_partial = prepare(poisson(V_partial; source=src, dirichlet=[bc]))
    m_small = prepare(poisson(V_small; source=src, dirichlet=[bc]))

    @test Unfitted.active_unknowns(m_partial.dofs) == Unfitted.active_unknowns(m_small.dofs)

    assemble!(m_partial)
    assemble!(m_small)
    sol_partial = solve!(m_partial)
    sol_small = solve!(m_small)

    @test l2_error(sol_partial, m_partial, x -> value(sol_small, m_small, x); norm=:absolute) <
          1e-10
end

@testset "interior active/inactive face introduces artificial constraint" begin
    # Overlay 3x3 cells (axis indices 1..3) over (0.3,0.7)x(0.3,0.7); deactivate
    # cell (1,1). The NODE-NODE dof at overlay axis indices (2,2) lies on the
    # interior face between active and inactive cells -> artificial constraint.
    # The NODE-NODE dof at (3,2) is fully enclosed by active cells -> free.
    bc = dirichlet(0.0; on=boundary(:all))
    inactive_one = setdiff(vec([CartesianIndex(i, j) for i in 1:3, j in 1:3]),
                           [CartesianIndex(1, 1)])
    V = overlay(space(_OMEGA; cells=(8, 8), order=2), box((0.3, 0.3), (0.7, 0.7)); cells=(3, 3),
                order=3, active=inactive_one)

    model = prepare(poisson(V; source=1.0, dirichlet=[bc]))
    layout = Unfitted._field_layout(model.dofs, :u).dofs
    overlay_level = V.levels[2]

    raw_22 = only(raw
                  for (raw, key) in pairs(layout.raw_keys)
                  if key.level == overlay_level.id &&
                         key.axes[1].kind == Unfitted._AXIS_NODE &&
                         key.axes[1].index == 2 &&
                         key.axes[2].kind == Unfitted._AXIS_NODE &&
                         key.axes[2].index == 2)
    @test Unfitted.constraint_kind(layout, raw_22) == :overlay

    raw_32 = only(raw
                  for (raw, key) in pairs(layout.raw_keys)
                  if key.level == overlay_level.id &&
                         key.axes[1].kind == Unfitted._AXIS_NODE &&
                         key.axes[1].index == 3 &&
                         key.axes[2].kind == Unfitted._AXIS_NODE &&
                         key.axes[2].index == 2)
    @test Unfitted.constraint_kind(layout, raw_32) == :free
end

@testset "SPD preserved on masked configuration" begin
    bc = dirichlet(0.0; on=boundary(:all))
    V = _safe_overlay(overlay_cells=(4, 4), active=(b, i) -> i.I[1] <= 2)

    model = prepare(poisson(V; source=1.0, dirichlet=[bc]))
    assemble!(model)

    M = Matrix(model.matrix)
    @test issymmetric(M)
    @test isposdef(Symmetric(M))
end

@testset "threaded == serial on masked configuration" begin
    bc = dirichlet(0.0; on=boundary(:all))
    V = _safe_overlay(overlay_cells=(4, 4), active=(b, i) -> i.I[1] <= 2 || i.I[2] <= 2)

    m_serial = prepare(poisson(V; source=1.0, dirichlet=[bc]))
    m_threaded = prepare(poisson(V; source=1.0, dirichlet=[bc]))
    assemble!(m_serial; threaded=false)
    assemble!(m_threaded; threaded=true)

    # Threaded assembly may reassociate region contributions; differences must
    # stay at roundoff scale.
    @test m_serial.matrix ≈ m_threaded.matrix
    @test m_serial.rhs ≈ m_threaded.rhs

    sol_serial = solve!(m_serial)
    sol_threaded = solve!(m_threaded)
    @test sol_serial.coefficients ≈ sol_threaded.coefficients
end

@testset "point evaluation in inactive overlay cell sees only base contribution" begin
    bc = dirichlet(0.0; on=boundary(:all))
    src = x -> sin(pi * x[1]) * sin(pi * x[2])

    # Only cell (1,1) of the 4x4 overlay is active. Point (0.6, 0.6) sits well
    # inside the inactive region -> overlay contribution must be exactly zero.
    V = overlay(space(_OMEGA; cells=(8, 8), order=2), box((0.3, 0.3), (0.7, 0.7)); cells=(4, 4),
                order=3, active=[CartesianIndex(1, 1)])
    model = prepare(poisson(V; source=src, dirichlet=[bc]))
    assemble!(model)
    sol = solve!(model)

    x_inactive = SVector(0.6, 0.6)
    overlay_level = V.levels[2]
    overlay_value = Unfitted._level_field(sol.coefficients, model,
                                          Unfitted._field_layout(model.dofs, :u), overlay_level,
                                          x_inactive, 1, Val(false))
    @test overlay_value == 0.0
end

@testset "moved_space preserves the mask" begin
    V = _safe_overlay(active=(b, i) -> i.I[1] <= 2)
    V_moved = Unfitted.moved_space(V; level=2, to=box((0.2, 0.2), (0.6, 0.6)))

    @test V_moved.levels[2].mask !== nothing
    @test V_moved.levels[2].mask.on == V.levels[2].mask.on
end

@testset "active_cells query" begin
    V_clean = _safe_overlay()
    model_clean = prepare(poisson(V_clean; source=1.0,
                                  dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test active_cells(model_clean; level=1) == trues(8, 8)
    @test active_cells(model_clean; level=2) == trues(4, 4)

    V_partial = _safe_overlay(active=(b, i) -> i.I[1] <= 2)
    model_partial = prepare(poisson(V_partial; source=1.0,
                                    dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    bits = active_cells(model_partial; level=2)
    @test bits == [i <= 2 for i in 1:4, _ in 1:4]

    # Mutating the returned BitArray must not leak into the model.
    bits[3, 3] = true
    @test active_cells(model_partial; level=2)[3, 3] == false

    @test_throws ArgumentError active_cells(model_partial; level=99)
end

@testset "activate! flips inactive cells on and invalidates the model" begin
    bc = dirichlet(0.0; on=boundary(:all))
    src = x -> sin(pi * x[1]) * sin(pi * x[2])

    V = _safe_overlay(active=(b, i) -> i.I[1] <= 2)
    model = prepare(poisson(V; source=src, dirichlet=[bc]))
    assemble!(model)
    sol = solve!(model)

    @test model.matrix !== nothing
    starting_version = model.version

    activate!(model; level=2, cells=[CartesianIndex(3, 1), CartesianIndex(3, 2)])

    @test model.version == starting_version + 1
    @test model.matrix === nothing
    @test model.rhs === nothing
    @test active_cells(model; level=2)[3, 1]
    @test active_cells(model; level=2)[3, 2]
    @test !active_cells(model; level=2)[4, 4]

    # The pre-mutation solution must now refuse to be used with this model.
    @test_throws ArgumentError Unfitted._checked_coefficients(sol, model)

    # Re-assemble and check the matrix is rebuilt cleanly.
    assemble!(model)
    sol2 = solve!(model)
    @test isposdef(Symmetric(Matrix(model.matrix)))
    @test sol2.model_version == model.version
end

@testset "deactivate! flips active cells off" begin
    bc = dirichlet(0.0; on=boundary(:all))
    V = _safe_overlay()
    model = prepare(poisson(V; source=1.0, dirichlet=[bc]))

    deactivate!(model; level=2, cells=(b, i) -> i.I[1] >= 3)

    bits = active_cells(model; level=2)
    @test count(bits) == 8         # cells with i.I[1] in {1, 2} -> 2 * 4 = 8 active
    @test all(!bits[i, j] for i in 3:4, j in 1:4)
    @test all(bits[i, j] for i in 1:2, j in 1:4)
end

@testset "move! carries the caller's mask flips across" begin
    # `move!` re-derives from the pre-fold problem, so the flips `activate!` /
    # `deactivate!` recorded have to be recorded there too — otherwise moving
    # one overlay silently reverts the caller's mask everywhere, including on
    # levels the move does not touch.
    bc = dirichlet(0.0; on=boundary(:all))
    V = overlay(overlay(space(_OMEGA; cells=(8, 8), order=2), box((0.1, 0.1), (0.5, 0.5));
                        cells=(4, 4), order=2), box((0.5, 0.5), (0.9, 0.9)); cells=(4, 4), order=2)
    model = prepare(poisson(V; source=1.0, dirichlet=[bc]))

    deactivate!(model; level=2, cells=[CartesianIndex(1, 1)])
    deactivate!(model; level=3, cells=[CartesianIndex(4, 4)])
    move!(model; level=2, to=box((0.2, 0.2), (0.6, 0.6)))

    @test !active_cells(model; level=2)[1, 1]   # flip on the moved level
    @test count(active_cells(model; level=2)) == 15
    @test !active_cells(model; level=3)[4, 4]   # flip on an untouched level
    @test count(active_cells(model; level=3)) == 15
end

@testset "activating to all-on collapses to no-mask fast path" begin
    bc = dirichlet(0.0; on=boundary(:all))
    V = _safe_overlay(active=(b, i) -> i.I[1] <= 2)
    model = prepare(poisson(V; source=1.0, dirichlet=[bc]))

    @test model.problem.space.levels[2].mask !== nothing
    activate!(model; level=2, cells=trues(4, 4))
    @test model.problem.space.levels[2].mask === nothing
    @test active_cells(model; level=2) == trues(4, 4)
end

@testset "diagnostics reports inactive cell counts per level" begin
    bc = dirichlet(0.0; on=boundary(:all))
    V = _safe_overlay(active=(b, i) -> i.I[1] <= 2)
    model = prepare(poisson(V; source=1.0, dirichlet=[bc]))

    counts = diagnostics(model).inactive_cell_counts
    @test counts == [0, 8]    # base unmasked; overlay has 8 inactive cells

    deactivate!(model; level=2, cells=[CartesianIndex(1, 1)])
    @test diagnostics(model).inactive_cell_counts == [0, 9]

    activate!(model; level=2, cells=trues(4, 4))
    @test diagnostics(model).inactive_cell_counts == [0, 0]
end

@testset "mutation matches fresh construction" begin
    bc = dirichlet(0.0; on=boundary(:all))
    src = x -> sin(pi * x[1]) * sin(pi * x[2])

    # Start from a fully-active overlay, then deactivate the right half.
    V_mut = _safe_overlay()
    m_mut = prepare(poisson(V_mut; source=src, dirichlet=[bc]))
    deactivate!(m_mut; level=2, cells=(b, i) -> i.I[1] >= 3)
    assemble!(m_mut)
    sol_mut = solve!(m_mut)

    # Same configuration built directly.
    V_fresh = _safe_overlay(active=(b, i) -> i.I[1] <= 2)
    m_fresh = prepare(poisson(V_fresh; source=src, dirichlet=[bc]))
    assemble!(m_fresh)
    sol_fresh = solve!(m_fresh)

    @test Unfitted.active_unknowns(m_mut.dofs) == Unfitted.active_unknowns(m_fresh.dofs)
    @test m_mut.matrix ≈ m_fresh.matrix
    @test m_mut.rhs ≈ m_fresh.rhs
    @test sol_mut.coefficients ≈ sol_fresh.coefficients
end

@testset "activate! accepts BitArray selector" begin
    bc = dirichlet(0.0; on=boundary(:all))
    V = _safe_overlay(active=falses(4, 4))
    model = prepare(poisson(V; source=1.0, dirichlet=[bc]))

    select = falses(4, 4)
    select[1, 1] = true
    select[2, 3] = true
    activate!(model; level=2, cells=select)

    bits = active_cells(model; level=2)
    @test bits[1, 1] && bits[2, 3]
    @test count(bits) == 2
end

@testset "activate! rejects out-of-bounds level" begin
    V = _safe_overlay()
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    @test_throws ArgumentError activate!(model; level=0, cells=[CartesianIndex(1, 1)])
    @test_throws ArgumentError activate!(model; level=5, cells=[CartesianIndex(1, 1)])
end

@testset "activating a fold-deactivated cell does not override the geometry" begin
    # The fictitious fold and the user's `active =` selection are merged into one
    # `LevelMask`, and only one of them is the caller's to set. A cell the
    # classifier found entirely outside Ω carries no material, has no quadrature
    # rule and contributes nothing, so `activate!` on it is a no-op on the
    # effective space — while staying visible in the pre-fold record, which is
    # what a later `move!` or a geometry change re-derives from.
    disc = physical_domain(x -> sqrt(x[1]^2 + x[2]^2) - 0.4; lipschitz=1.0,
                           subcell_length_scale=0.05)
    V = space(box((-1.0, -1.0), (1.0, 1.0)); cells=(4, 4), order=1, physical=disc)
    model = prepare(mass(V; coefficient=1.0))
    outside = findfirst(!, active_cells(model; level=1))
    @test outside !== nothing

    activate!(model; level=1, cells=[outside])
    @test !active_cells(model; level=1)[outside]                  # the geometry wins
    @test active_cells(model; level=1, effective=false)[outside]   # the request is kept
    @test diagnostics(model, solve!(model)).fit_failure_count == 0
end
