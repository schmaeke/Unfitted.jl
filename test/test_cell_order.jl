using StaticArrays
using LinearAlgebra: norm

using Unfitted: dof_layout, cell_dofs, cell_basis_indices, basis_values, _level_by_id, _index_admissible,
                local_basis_indices, _tensor_values!, _factor_buffers, _entity_carries, _incident_cells, TensorDofKey,
                _axis_dof_key, _AXIS_SPAN, _supports_cell_order, IntegratedLegendre, cell_order

# Per-cell polynomial order: the minimum rule, and the invariants that keep a mixed-order space usable.
#
# The acceptance test here is the *per-basis-function one-sided trace jump*, and it is the only assertion in this file
# that can catch a broken minimum rule. Nothing structural can: a non-conforming mixed-order space assembles, solves,
# reports `symmetry_residual = 0`, and — measured below — matches the conforming space's L² projection error and
# convergence rate to three digits, because it is a strictly *larger* space and L² projection never feels H¹ conformity.
# What it loses is the Galerkin solution, which the last testset pins down.

# Maximum one-sided trace jump over every raw dof of `V`, across the internal face between cells `a` and `b` of level
# `lid`, which are adjacent along axis `d`.
#
# The two cells' shared face is ξ_d = +1 in `a`'s reference frame and ξ_d = −1 in `b`'s, so this evaluates the *same
# physical point* from both sides with no ε offset — a jump of exactly zero is attainable, and roundoff is the only
# floor. Basis functions are compared one at a time rather than through a solution: a jump in the solution decays under
# h-refinement even on a broken space, while the per-basis-function jump is h-independent and is what conformity means.
function trace_jump(V, lid, a, b, d; n=13)
    layout = dof_layout(V)
    level = _level_by_id(V, lid)
    D = length(a.I)
    ra = cell_dofs(layout, lid, a)
    rb = cell_dofs(layout, lid, b)
    ts = range(-1.0, 1.0; length=n)
    tangential = D == 1 ? [()] : vec([Tuple(g) for g in Iterators.product(ntuple(_ -> ts, D - 1)...)])
    worst = 0.0
    for g in tangential
        face(sign) = SVector{D,Float64}(ntuple(k -> k == d ? sign : g[k < d ? k : k - 1], D))
        va = basis_values(level, a, face(1.0))
        vb = basis_values(level, b, face(-1.0))
        for r in union(ra, rb)
            ia = findfirst(==(r), ra)
            ib = findfirst(==(r), rb)
            xa = ia === nothing ? 0.0 : va[ia]
            xb = ib === nothing ? 0.0 : vb[ib]
            worst = max(worst, abs(xa - xb))
        end
    end
    return worst
end

# Every internal face of a single-level space, worst jump over all of them.
function worst_trace_jump(V; level::Int=1, n::Int=9)
    lvl = _level_by_id(V, level)
    D = length(lvl.mesh.cells)
    worst = 0.0
    for d in 1:D, a in cell_indices(V; level=level)
        a.I[d] < lvl.mesh.cells[d] || continue
        b = CartesianIndex(ntuple(k -> k == d ? a.I[k] + 1 : a.I[k], D))
        worst = max(worst, trace_jump(V, level, a, b, d; n=n))
    end
    return worst
end

const _CO_2D = box((0.0, 0.0), (1.0, 1.0))
const _CO_3D = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

@testset "per-cell order: uniform levels are unchanged" begin
    # A per-cell field that happens to be flat must collapse to the uniform representation, so an existing user pays
    # nothing — one palette entry and no per-cell order or mode array, same dof counts, same reduction counts, same
    # solution.
    for (Ω, cells, p) in ((_CO_2D, 6, 3), (_CO_3D, 3, 2))
        D = length(Ω.lower)
        plain = space(Ω; cells=cells, order=p)
        dense = space(Ω; cells=cells, order=fill(p, ntuple(_ -> cells, D)))
        @test length(plain.levels[1].orders.palette) == 1
        @test length(dense.levels[1].orders.palette) == 1  # collapsed, not merely equal
        @test plain.levels[1].orders.class === nothing && plain.levels[1].modes.kind === nothing
        @test dense.levels[1].orders.class === nothing && dense.levels[1].modes.kind === nothing
        @test nominal_order(dense; level=1) == nominal_order(plain; level=1)
        @test dof_layout(dense).active_count == dof_layout(plain).active_count
        @test cell_orders(plain; level=1) == fill(ntuple(_ -> p, D), ntuple(_ -> cells, D))
    end

    # A predicate that returns one order everywhere collapses too.
    V = space(_CO_2D; cells=4, order=(_box, _cell) -> 2)
    @test length(V.levels[1].orders.palette) == 1
    @test V.levels[1].orders.class === nothing && V.levels[1].modes.kind === nothing
    @test nominal_order(V; level=1) == (2, 2)

    # And a stacked, pruned space keeps its reduction verdicts.
    ref = overlay(space(_CO_2D; cells=4, order=3), box((0.25, 0.25), (0.75, 0.75)); cells=4, order=3)
    per = overlay(space(_CO_2D; cells=4, order=fill(3, 4, 4)), box((0.25, 0.25), (0.75, 0.75)); cells=4,
                  order=fill(3, 4, 4))
    lr = dof_layout(ref)
    lp = dof_layout(per)
    @test length(lr.raw_keys) == length(lp.raw_keys)
    @test lr.active_count == lp.active_count
    @test lr.elimination_source == lp.elimination_source
end

@testset "per-cell order: C⁰ conformity across an order jump" begin
    # Two cells at different orders. Without the minimum rule the high-order cell generates shared-entity keys the
    # low-order cell does not, and each of them is nonzero on the face they share: the measured jump is sup|N̂₃| =
    # 0.3036 in 2D and 3D `:tensor`, and 0.1859 on the 3D `:trunk` face key a per-axis (rather than set-membership) rule
    # would also admit. With the rule the jump is exactly zero — not small, zero, because both sides evaluate the same
    # product of the same 1D factors.
    V = space(_CO_2D; cells=(2, 1), order=reshape([3, 2], 2, 1))
    @test length(dof_layout(V).raw_keys) == 21          # 22 without the rule
    @test trace_jump(V, 1, CartesianIndex(1, 1), CartesianIndex(2, 1), 1) == 0.0

    # Anisotropy is not compromised: only a key's SPAN (tangential) axes are capped, and the face-normal orders
    # constrain nothing on that face.
    V = space(_CO_2D; cells=(2, 1), order=reshape([(5, 2), (2, 5)], 2, 1))
    @test trace_jump(V, 1, CartesianIndex(1, 1), CartesianIndex(2, 1), 1) == 0.0

    V = space(_CO_3D; cells=(2, 1, 1), order=reshape([3, 2], 2, 1, 1))
    @test length(dof_layout(V).raw_keys) == 75          # 82 without the rule
    @test trace_jump(V, 1, CartesianIndex(1, 1, 1), CartesianIndex(2, 1, 1), 1) == 0.0

    # `:trunk` in 3D is where a per-axis mode comparison fails and the set test does not: a face key with bubble modes
    # (3, 2) has trunk degree 5, so the order-4 cell never generates it even though 3 ≤ 4 and 2 ≤ 4.
    V = space(_CO_3D; cells=(2, 1, 1), order=reshape([5, 4], 2, 1, 1), mode=:trunk)
    @test length(dof_layout(V).raw_keys) == 101         # 103 under a per-axis rule
    @test trace_jump(V, 1, CartesianIndex(1, 1, 1), CartesianIndex(2, 1, 1), 1) == 0.0

    # Graded fields in both dimensions, every internal face.
    @test worst_trace_jump(space(_CO_2D; cells=4, order=[2 + (i + j) % 4 for i in 1:4, j in 1:4])) == 0.0
    @test worst_trace_jump(space(_CO_3D; cells=2, order=[2 + (i + 2j + 3k) % 4 for i in 1:2, j in 1:2, k in 1:2])) ==
          0.0
    # 1D needs no rule at all — every shared key carries mode 0 — but must still come out conforming.
    @test worst_trace_jump(space(box((0.0,), (1.0,)); cells=(3,), order=reshape([2, 5, 3], 3))) == 0.0
end

@testset "per-cell order: the minimum rule's shape" begin
    # A shared entity is capped at the minimum over its incident cells, and only on the axes it spans. A=(5,2) beside
    # B=(2,5) across a face ⟂ axis 1 must drop exactly the keys whose SPAN axis is 2 with mode > min(2, 5) = 2 — the
    # axis-1 orders 5 and 2 feed cell-local bubbles and constrain nothing there.
    V = space(_CO_2D; cells=(2, 1), order=reshape([(5, 2), (2, 5)], 2, 1))
    lvl = V.levels[1]
    free = Set(cell_basis_indices(lvl, CartesianIndex(1, 1)))
    full = Set(local_basis_indices(lvl.basis, (5, 2)))
    dropped = setdiff(full, free)
    @test length(dropped) == 0                     # cell A is already at 2 on axis 2
    freeB = Set(cell_basis_indices(lvl, CartesianIndex(2, 1)))
    fullB = Set(local_basis_indices(lvl.basis, (2, 5)))
    droppedB = setdiff(fullB, freeB)
    @test length(droppedB) == 3
    @test all(id -> id.I[1] == 0 && id.I[2] in 3:5, droppedB)

    # The kept set is in general not a tensor product of per-axis ranges, which is why it has to be stored per cell
    # rather than derived from an order.
    V = space(_CO_2D; cells=(2, 1), order=reshape([6, 4], 2, 1))
    kept = cell_basis_indices(V.levels[1], CartesianIndex(1, 1))
    @test length(kept) == 47                       # 49 minus (0,5) and (0,6)
    @test !any(o -> length(kept) == prod(o .+ 1), ((6, 6), (6, 4), (4, 6), (4, 4)))

    # The rule is a function of the KEY, so both incident cells agree: every raw is generated either by every active
    # incident cell or by none.
    layout = dof_layout(space(_CO_2D; cells=4, order=[2 + (i + j) % 4 for i in 1:4, j in 1:4]))
    lvl = space(_CO_2D; cells=4, order=[2 + (i + j) % 4 for i in 1:4, j in 1:4]).levels[1]
    n = lvl.mesh.cells
    for (raw, key) in pairs(layout.raw_keys)
        generated = [ci for ci in _incident_cells(key, n) if raw in cell_dofs(layout, 1, ci)]
        @test length(generated) == length(collect(_incident_cells(key, n)))
    end
end

@testset "per-cell order: inactive cells do not lower a shared entity" begin
    # An inactive cell generates nothing, so it cannot disagree across a face and must not enter the minimum. Letting it
    # in would silently delete live dofs on a fictitious fold, where `_has_overlay_constraint` deliberately keeps the
    # boundary modes free.
    orders = fill(4, 3, 3)
    orders[2, 2] = 1
    active = trues(3, 3)
    active[2, 2] = false
    aware = space(_CO_2D; cells=3, order=orders, active=active)
    # The same space with the low-order cell simply absent from the field: the active-aware rule must agree with it,
    # because the inactive cell is invisible.
    equivalent = space(_CO_2D; cells=3, order=fill(4, 3, 3), active=active)
    @test length(dof_layout(aware).raw_keys) == length(dof_layout(equivalent).raw_keys)
    @test dof_layout(aware).active_count == dof_layout(equivalent).active_count
    # And a mask edit rebuilds the table: reactivating the centre cell must make its order-1 field bite again.
    reactivated = space(_CO_2D; cells=3, order=orders)
    @test length(dof_layout(reactivated).raw_keys) < length(dof_layout(equivalent).raw_keys)
end

@testset "per-cell order: validation" begin
    # `order ≥ 1` is a per-cell requirement once the order is per cell: the key collapses endpoint modes 0 and 1 into
    # one NODE factor, so no filter over keys can notice a cell missing one of its two endpoint modes.
    @test_throws ArgumentError space(_CO_2D; cells=2, order=[1 1; 1 0])
    # `:trunk` needs isotropy per cell, not per level.
    @test_throws ArgumentError space(_CO_2D; cells=2, order=reshape([(2, 2), (3, 2), (2, 2), (2, 2)], 2, 2),
                                     mode=:trunk)
    # Shape and type errors are actionable.
    @test_throws DimensionMismatch space(_CO_2D; cells=2, order=fill(2, 3, 3))
    @test_throws ArgumentError space(_CO_2D; cells=2, order="two")
    @test_throws ArgumentError space(_CO_2D; cells=2, order=fill(0, 2, 2))
    # A palette of 300 distinct orders is a p-field, not a mistake: the class index is `UInt16`, so the cap sits three
    # orders of magnitude above what an anisotropic estimator produces and is no longer reachable by a test that builds
    # a level. The rejection itself is still there, just out of reach.
    wide = space(box((0.0,), (1.0,)); cells=(300,), order=reshape(collect(1:300), 300))
    @test length(wide.levels[1].orders.palette) == 300
    @test wide.levels[1].orders.class isa Array{UInt16,1}     # a graded level carries its class map
end

@testset "per-cell order: the index-set membership predicate" begin
    # `_index_admissible` is the O(D) form of "is this multi-index in `local_basis_indices(basis, order)`", and the
    # minimum rule asks it at an order no cell carries. The two forms must not drift apart, in either mode.
    for D in 1:3, mode in (:tensor, :trunk), p in 1:5
        fam = Unfitted._with_mode(IntegratedLegendre(), mode)
        order = ntuple(_ -> p, D)
        want = Set(local_basis_indices(fam, order))
        for id in CartesianIndices(ntuple(_ -> 0:6, D))
            @test _index_admissible(fam, order, id) == (id in want)
        end
    end
    # Anisotropic, `:tensor` only (`:trunk` requires isotropy).
    fam = IntegratedLegendre()
    for order in ((2, 5), (5, 2), (1, 4))
        want = Set(local_basis_indices(fam, order))
        for id in CartesianIndices((0:6, 0:6))
            @test _index_admissible(fam, order, id) == (id in want)
        end
    end
end

@testset "per-cell order: the basis value buffer guard" begin
    # The tensor kernel's guard was relaxed from `length(values) == length(indices)` to `>=` so a bank sized for the
    # level's longest minimum-rule set can serve a cell whose set is shorter. The surviving half of the guard — a buffer
    # too SHORT for the index list — is what stands between a length bug and a silently wrong answer, so it must still
    # fire.
    fam = IntegratedLegendre()
    ids = local_basis_indices(fam, (2, 2))
    ξ = SVector(0.1, -0.2)
    cell = CartesianIndex(1, 1)
    @test_throws DimensionMismatch _tensor_values!(fam, zeros(length(ids) - 1), ids, (2, 2), ξ,
                                                   _factor_buffers((2, 2), Float64), cell)
    long = zeros(length(ids) + 7)
    _tensor_values!(fam, long, ids, (2, 2), ξ, _factor_buffers((2, 2), Float64), cell)
    exact = zeros(length(ids))
    _tensor_values!(fam, exact, ids, (2, 2), ξ, _factor_buffers((2, 2), Float64), cell)
    @test long[1:length(ids)] == exact

    # The same two halves for the gradient buffer the kernel's `grad = (gradients, scale, der1d)` argument carries: a
    # short one is refused, a long one is filled in a prefix, and the values the gradient call writes are the
    # values-only call's.
    scale = SVector(2.0, 4.0)
    grad(n) = (zeros(SVector{2,Float64}, n), scale, _factor_buffers((2, 2), Float64))
    @test_throws DimensionMismatch _tensor_values!(fam, zeros(length(ids)), ids, (2, 2), ξ,
                                                   _factor_buffers((2, 2), Float64), cell, grad(length(ids) - 1))
    long_values, long_grad = zeros(length(ids) + 7), grad(length(ids) + 7)
    _tensor_values!(fam, long_values, ids, (2, 2), ξ, _factor_buffers((2, 2), Float64), cell, long_grad)
    exact_values, exact_grad = zeros(length(ids)), grad(length(ids))
    _tensor_values!(fam, exact_values, ids, (2, 2), ξ, _factor_buffers((2, 2), Float64), cell, exact_grad)
    @test long_grad[1][1:length(ids)] == exact_grad[1]
    @test long_values[1:length(ids)] == exact_values == exact
end

@testset "per-cell order: elevate / cell_orders round trip" begin
    V = space(_CO_2D; cells=4, order=2)
    p = cell_orders(V; level=1)
    p[CartesianIndex(1, 1)] = (5, 5)
    p[CartesianIndex(4, 4)] = (4, 4)
    W = elevate(V, 1 => p)
    # The order lives in a field, not a type parameter, so an order edit cannot recompile the assembly pipeline.
    @test typeof(W) === typeof(V)
    @test cell_orders(W; level=1) == p
    @test nominal_order(W; level=1) == (5, 5)                  # nominal = per-axis maximum
    @test nominal_order(elevate(W, 1 => cell_orders(W; level=1)); level=1) == (5, 5)
    # Round-tripping back to a flat field collapses the representation again.
    @test length(elevate(W, 1 => 2).levels[1].orders.palette) == 1

    # The pair form raises named cells and leaves every other cell alone.
    Z = elevate(V, 1 => [CartesianIndex(2, 3) => 4])
    q = cell_orders(Z; level=1)
    @test q[CartesianIndex(2, 3)] == (4, 4)
    @test q[CartesianIndex(1, 1)] == (2, 2)
    @test_throws ArgumentError elevate(V, 1 => [CartesianIndex(9, 9) => 4])
    @test_throws ArgumentError elevate(V, 1 => 2, 1 => 3)

    # Overlays take the same shapes, resolved against their own cell grid.
    S = overlay(space(_CO_2D; cells=4, order=2), box((0.25, 0.25), (0.75, 0.75)); cells=2, order=[3 4; 4 3])
    @test cell_orders(S; level=2) == [(3, 3) (4, 4); (4, 4) (3, 3)]
    @test nominal_order(S; level=2) == (4, 4)
    @test worst_trace_jump(S; level=2) == 0.0
end

@testset "per-cell order: elevate takes a pair list in every documented container" begin
    # The pair form is documented as a vector, tuple, dictionary or generator of `cell => order` pairs. Each must mean
    # exactly what the vector means, and go through the same entry checks.
    V = space(_CO_2D; cells=4, order=2)
    a, b = CartesianIndex(1, 1), CartesianIndex(3, 2)
    want = cell_orders(elevate(V, 1 => [a => 3, b => (4, 3)]); level=1)
    @test want[a] == (3, 3) && want[b] == (4, 3) && want[CartesianIndex(2, 2)] == (2, 2)
    for spec in ((a => 3, b => (4, 3)), Dict(a => 3, b => (4, 3)), (c => o for (c, o) in ((a, 3), (b, (4, 3)))))
        @test cell_orders(elevate(V, 1 => spec); level=1) == want
    end
    @test_throws ArgumentError elevate(V, 1 => (CartesianIndex(9, 9) => 3,))
    @test_throws ArgumentError elevate(V, 1 => Dict((1, 1) => 3))
    @test_throws ArgumentError elevate(V, 1 => (c => 0 for c in (a,)))
end

@testset "per-cell order: cell_basis_indices returns the level's shared, read-only set" begin
    # The documented contract: a lookup into the level's own table on every level, so every active cell of a uniform
    # level gets the SAME vector (callers must not mutate it), equal to the index set at the level's order, and an
    # inactive cell gets the empty set.
    V = space(_CO_2D; cells=2, order=2, active=[CartesianIndex(1, 1), CartesianIndex(2, 2), CartesianIndex(1, 2)])
    level = V.levels[1]
    shared = cell_basis_indices(level, CartesianIndex(1, 1))
    @test cell_basis_indices(level, CartesianIndex(2, 2)) === shared
    @test cell_basis_indices(level, CartesianIndex(1, 2)) === shared
    @test shared == local_basis_indices(level.basis, (2, 2))
    @test isempty(cell_basis_indices(level, CartesianIndex(2, 1)))
    # And a p-step on integrated Legendre keeps the space's type, uniform or graded.
    @test typeof(elevate(V, 1 => 3)) === typeof(V)
    @test typeof(elevate(V, 1 => [CartesianIndex(1, 1) => 4])) === typeof(V)
end

@testset "per-cell order: through the model" begin
    Ω = _CO_2D
    orders = [2 + (i + j) % 3 for i in 1:8, j in 1:8]
    V = space(Ω; cells=8, order=orders)
    u = field(:u, V)
    model = prepare(poisson(V; source=(x) -> 2π^2 * sinpi(x[1]) * sinpi(x[2]),
                            dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    sol = solve!(model)
    d = diagnostics(model, sol)
    @test d.symmetry_residual == 0.0
    @test length(d.levels[1].order_palette) == 3
    @test d.levels[1].order == (4, 4)          # the report keeps its `order` key
    exact(x) = sinpi(x[1]) * sinpi(x[2])
    @test l2_error(sol, model, exact) < 5e-4
    # Point evaluation must read the same per-cell index set the dof walk used — a level-wide list would pair basis
    # values with the wrong raws, silently.
    @test abs(value(sol, model, u, (0.5, 0.5)) - 1.0) < 5e-3

    # `elevated` mirrors `adapted`, and a p-step keeps the model's type.
    raised = elevated(model, 1 => fill(4, 8, 8))
    @test active_unknowns(raised) > active_unknowns(model)
    @test cell_orders(raised; level=1) == fill((4, 4), 8, 8)
end

@testset "per-cell order: L² projection rate, and what it cannot see" begin
    # The brief's second acceptance measurement. It passes — the conforming graded space keeps its rate — but the honest
    # record is that it has no discriminating power: measured against a deliberately non-conforming twin the rates and
    # errors agree to three digits, because the broken space is strictly LARGER and an L² projection never feels H¹
    # conformity. Keep it as a regression on the space's approximation power, not as a conformity test.
    f(x) = sinpi(2 * x[1]) * exp(x[2])
    errs = Float64[]
    for n in (4, 8, 16)
        orders = [2 + (i + j) % 3 for i in 1:n, j in 1:n]
        V = space(_CO_2D; cells=n, order=orders)
        u = field(:u, V)
        model = prepare(Problem((u,); blocks=(mass_block(u),), loads=(source_load(u; source=f),)))
        sol = solve!(model)
        push!(errs, l2_error(sol, model, f))
    end
    rates = [log2(errs[i] / errs[i + 1]) for i in 1:2]
    @test all(>(1.7), rates)
    @test errs[end] < 1e-3
end

@testset "per-cell order: every level-rebuild path carries the field" begin
    # `Level` is rebuilt verbatim at six sites. Four keep the cell grid and must carry the order field; the two that
    # change the mask must additionally rebuild the minimum-rule table, because the rule minimises over the active
    # incident cells only.
    orders = [2 + (i + j) % 3 for i in 1:4, j in 1:4]
    V = overlay(space(_CO_2D; cells=4, order=2), box((0.2, 0.2), (0.8, 0.8)); cells=4, order=orders)
    @test cell_orders(V; level=2) == [(o, o) for o in orders]

    # moved_space — the cell grid is unchanged, only its coordinates.
    W = moved_space(V; level=2, to=box((0.1, 0.1), (0.7, 0.7)))
    @test cell_orders(W; level=2) == cell_orders(V; level=2)
    @test worst_trace_jump(W; level=2) == 0.0

    # activate! / deactivate! rebuild the level's mask through the one level builder.
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    before = active_unknowns(model)
    deactivate!(model; level=2, cells=[CartesianIndex(1, 1)])
    @test active_unknowns(model) < before
    @test cell_orders(model; level=2) == [(o, o) for o in orders]
    activate!(model; level=2, cells=[CartesianIndex(1, 1)])
    @test active_unknowns(model) == before

    # move! rebuilds the space through `moved_space`.
    move!(model; level=2, to=box((0.15, 0.15), (0.75, 0.75)))
    @test cell_orders(model; level=2) == [(o, o) for o in orders]
    @test solve!(model) isa Solution
end

@testset "per-cell order: transfer, Dirichlet data and the order_max mesh array" begin
    Ω = _CO_2D
    lo = space(Ω; cells=4, order=[2 + (i + j) % 2 for i in 1:4, j in 1:4])
    hi = elevate(lo, 1 => [3 + (i + j) % 2 for i in 1:4, j in 1:4])
    exact(x) = 1 + x[1] + 2x[2]

    # Nonzero physical Dirichlet data through a per-cell-order boundary trace. The trace pairs `local_basis_indices`
    # with `cell_dofs` positionally, so a level-wide index list would attach the datum to the wrong raws — silently.
    src = prepare(poisson(lo; source=0.0, dirichlet=[dirichlet(exact; on=boundary(:all))]))
    u = solve!(src)
    @test l2_error(u, src, exact) < 1e-12

    # L² transfer between two per-cell-order spaces. The transfer workspace has the same level-keyed banks the assembler
    # does and needs the same table.
    tgt = prepare(poisson(hi; source=0.0, dirichlet=[dirichlet(exact; on=boundary(:all))]))
    moved = transfer(u, src, tgt; via=L2Projection())
    @test l2_error(moved, tgt, exact) < 1e-10

    # The `order_max` cell array of the level's mesh is the p-field visualisation for free: each cell's own order, in
    # `cell_indices` order.
    (; grid) = only(only(Unfitted._sample_fields(u, src; point_data=NamedTuple())).meshes)
    @test Dict(grid.cell_data)["order_max"] == vec([2 + (i + j) % 2 for i in 1:4, j in 1:4])
end

@testset "per-cell order: immersed physical domain" begin
    # The fictitious fold rebuilds the level's mask, so the minimum-rule table must be rebuilt with it — and the
    # moment-fit order is now taken per parent cell, so a low-p cut cell no longer pays the level maximum's fit.
    Ω = box((-1.0, -1.0), (1.0, 1.0))
    disc = physical_domain(x -> x[1]^2 + x[2]^2 - 0.64; lipschitz=2.0, alpha=1e-8, subcell_length_scale=0.1)
    orders = [x1 <= 4 ? 2 : 4 for x1 in 1:8, _ in 1:8]
    V = space(Ω; cells=8, order=orders, physical=disc)
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    sol = solve!(model)
    d = diagnostics(model, sol)
    @test d.cut_region_count > 0
    @test d.fit_failure_count == 0
    @test d.symmetry_residual == 0.0
    @test active_unknowns(model) > 0
    # The fold deactivated cells; the surviving space is still C⁰ across every internal face between two active cells.
    lvl = model.problem.space.levels[1]
    for a in cell_indices(model.problem.space; level=1), d0 in 1:2
        a.I[d0] < 8 || continue
        b = CartesianIndex(ntuple(k -> k == d0 ? a.I[k] + 1 : a.I[k], 2))
        (Unfitted.is_active(lvl.mask, a) && Unfitted.is_active(lvl.mask, b)) || continue
        @test trace_jump(model.problem.space, 1, a, b, d0; n=5) == 0.0
    end
end

@testset "per-cell order: the mode table cannot go stale" begin
    # `CellModes` is derived by `Level`'s constructor from the orders, mesh, mode and mask. It used to be an argument,
    # which meant a caller could pair a table built against one mask with a different one; the failure was silent,
    # because the dof layout is built *from* the table and went stale with it. A witness field recorded which mask the
    # table came from so the constructor could check. There is nothing left to check: the constructor takes no table.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    field = [2 + (i + j) % 3 for i in 1:4, j in 1:4]
    V = space(Ω; cells=(4, 4), order=field)
    lvl = V.levels[1]

    @test !hasproperty(lvl, :order)          # the nominal summary is a function now
    @test nominal_order(lvl) == (4, 4)
    @test !hasfield(typeof(lvl.orders), :mask)     # the witness is gone with its cause
    @test !hasfield(typeof(lvl.orders), :locals)   # what the basis emits is not what was asked for

    # The derivation tracks the mask on every path that changes it, and the mode sets it produces agree with a level
    # built at that mask from scratch.
    kept = [ci for ci in cell_indices(V; level=1) if ci != CartesianIndex(2, 2)]
    V2 = adapt(V, 1 => kept)
    fresh = space(Ω; cells=(4, 4), order=field, active=[ci != CartesianIndex(2, 2) for ci in CartesianIndices((4, 4))])
    for ci in cell_indices(V2; level=1)
        @test cell_basis_indices(V2.levels[1], ci) == cell_basis_indices(fresh.levels[1], ci)
    end
    @test active_unknowns(prepare(mass(V2))) < active_unknowns(prepare(mass(V)))

    # A uniform level is a palette of one with no per-cell order or mode array: its readers answer from the one entry.
    uniform = space(Ω; cells=(4, 4), order=3)
    @test length(uniform.levels[1].orders.palette) == 1
    @test uniform.levels[1].orders.class === nothing && uniform.levels[1].modes.kind === nothing
    # The `Space` forms are the public spelling: space plus a `level` keyword, like every other accessor, with no reach
    # into `V.levels`.
    @test cell_order(uniform, CartesianIndex(2, 2); level=1) == (3, 3)
    @test nominal_order(uniform; level=1) == (3, 3)
    @test cell_order(uniform.levels[1], CartesianIndex(2, 2)) == (3, 3)
    # `show` summarises a per-cell field by its palette size. A uniform level's palette has one entry and says nothing,
    # so the suffix belongs only on a level whose order actually varies.
    @test !occursin("cell_orders=", sprint(show, uniform.levels[1]))
    @test occursin("cell_orders=3", sprint(show, space(Ω; cells=(4, 4), order=field).levels[1]))
    @test_throws ArgumentError cell_order(uniform, CartesianIndex(2, 2); level=2)
    @test_throws ArgumentError nominal_order(uniform; level=0)
    @test level_count(uniform) == 1

    # The fictitious fold re-masks underneath the user; it is the case that bit.
    disc = physical_domain(x -> sum(abs2, x .- 0.5) - 0.09; lipschitz=2.0, alpha=1e-8, subcell_length_scale=0.03)
    Vf = space(Ω; cells=(4, 4), order=field, physical=disc)
    @test active_unknowns(prepare(mass(Vf))) > 0
end

@testset "per-cell order: the palette admits an anisotropic p-field" begin
    # An anisotropic estimator is the natural one for a tensor-product family, and it reaches far more than 255 distinct
    # orders on a modest 3D grid — an 8³ grid admits 512. The palette index is wide enough that the cap is not a shape a
    # real p-field runs into.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    anisotropic = [(1 + (i - 1) % 16, 1 + (j - 1) % 16) for i in 1:16, j in 1:16]
    V = space(Ω; cells=(16, 16), order=anisotropic)
    @test V.levels[1].orders.class isa Array{UInt16,2}
    @test length(V.levels[1].orders.palette) == 256          # would have thrown at 255
    @test cell_orders(V; level=1)[CartesianIndex(3, 5)] == (3, 5)
    @test active_unknowns(prepare(mass(V))) > 0
end

@testset "per-cell order: an h- and a p-step in one rebuild" begin
    # An hp driver takes both halves every step. Going through `adapted` and `elevated` in turn builds and discards a
    # complete intermediate model; the Space-level compose costs microseconds, so the one-rebuild form is the one a loop
    # should use.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    V = ladder(Ω; cells=(8, 8), order=2, depth=1, splits=2)
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    mask = [ci for ci in cell_indices(V; level=2) if sum(abs2, center(cell_box(V, ci; level=2)) .- 0.5) < 0.08]
    orders = fill(2, 8, 8)
    orders[3:6, 3:6] .= 4

    stepwise = elevated(adapted(model, 2 => mask), 1 => orders)
    composed = adapted(model, elevate(adapt(V, 2 => mask), 1 => orders))
    @test active_unknowns(composed) == active_unknowns(stepwise)
    @test cell_orders(composed; level=1) == cell_orders(stepwise; level=1)
    @test active_cells(composed; level=2) == active_cells(stepwise; level=2)
    @test norm(solve!(composed).coefficients - solve!(stepwise).coefficients) < 1e-12

    # The caches this reuses are keyed without the geometry, so a space that is not this model's own is refused rather
    # than fitted with the wrong rules.
    @test_throws ArgumentError adapted(model, space(box((0.0, 0.0), (2.0, 2.0)); cells=(4, 4), order=2))
end

@testset "per-cell order: the intersection rule is the minimum rule" begin
    # The kernel asks each incident cell whether it generates the mode, instead of forming the componentwise minimum and
    # asking there. The two agree only because `_index_admissible` is monotone non-decreasing in the order — that is a
    # property of the FAMILY, not of the dof layer, so assert it directly. A family that broke it would get a silently
    # non-conforming space.
    for mode in (:tensor, :trunk), D in (2, 3)
        basis = Unfitted._with_mode(IntegratedLegendre(), mode)
        @test _supports_cell_order(basis)
        for lo in 1:4
            los = ntuple(_ -> lo, D)
            ids = local_basis_indices(basis, los)
            for hi in lo:5, d in 1:D
                his = ntuple(e -> e == d ? hi : lo, D)
                mode === :trunk && his != ntuple(_ -> hi, D) && continue   # trunk wants isotropy
                # Raising one axis may only ADD modes, never remove one.
                @test all(id -> _index_admissible(basis, his, id), ids)
            end
        end
    end

    # And the equivalence itself, cell for cell, against the componentwise minimum the rule used to form explicitly.
    minimum_rule(orders, mask, key, n) = begin
        pmin = nothing
        for ci in _incident_cells(key, n)
            Unfitted.is_active(mask, ci) || continue
            o = orders[ci]
            pmin = pmin === nothing ? o : ntuple(d -> min(pmin[d], o[d]), length(o))
        end
        pmin
    end
    Ω = box((0.0, 0.0), (1.0, 1.0))
    for field in ([1 + (i + j) % 4 for i in 1:6, j in 1:6], [(1 + i % 3, 1 + j % 4) for i in 1:6, j in 1:6]),
        msk in (nothing, [!(i == 3 && j == 3) for i in 1:6, j in 1:6])

        V = space(Ω,; cells=(6, 6), order=field, active=msk)
        lvl = V.levels[1]
        orders = [cell_order(lvl, ci) for ci in cell_indices(V; level=1)]
        n = lvl.mesh.cells
        for cell in cell_indices(V; level=1)
            Unfitted.is_active(lvl.mask, cell) || continue
            for id in local_basis_indices(lvl.basis, cell_order(lvl, cell))
                key = TensorDofKey{2}(0, ntuple(d -> _axis_dof_key(cell.I[d], id.I[d]), 2))
                pmin = minimum_rule(orders, lvl.mask, key, n)
                classical = pmin === nothing || _index_admissible(lvl.basis, pmin, id)
                # The dense `orders` matrix is the reference rule's input; the kernel reads the palette record the level
                # actually carries.
                @test _entity_carries(lvl.basis, lvl.orders, lvl.mask, key, n, id) == classical
            end
        end
    end
end

# A family that declares the per-cell-order capability without supplying the `_cell_modes` method that states its
# minimum rule. Declared here rather than in `src/` because the point is exactly that no shipped family is in this
# state.
struct _OptInNoRule <: Unfitted.BasisFamily end
Unfitted.basis_name(::_OptInNoRule) = :opt_in_no_rule
Unfitted._supports_cell_order(::_OptInNoRule) = true

@testset "opting into per-cell order without a minimum rule is refused, not ignored" begin
    # `_supports_cell_order` and the `_cell_modes` override used to be independent, so a family could pass
    # `_build_cell_orders`, build a graded level, and assemble a non-conforming space with no error anywhere: the
    # generic fallback gives every active cell `palette[1]`'s index set while `cell_order` reports the field that was
    # asked for. The generic method now refuses a real palette, so the two cannot be separated.
    Ω = box((0.0, 0.0), (1.0, 1.0))
    mesh = Unfitted.CartesianMesh(Ω; cells=(2, 2))
    family = _OptInNoRule()
    graded = Unfitted._build_cell_orders(family, [(1, 1), (2, 2), (1, 1), (2, 2)], mesh)
    @test length(graded.palette) == 2
    @test_throws ArgumentError Unfitted._cell_modes(family, graded, mesh, nothing)
    @test_throws ArgumentError space(Ω; cells=(2, 2), order=[1 2; 1 2], basis=family)

    # A one-entry palette is the uniform case and still goes through: `Level`'s constructor answers it for every family,
    # with the family's index set at the one order shared by every active cell.
    flat = Unfitted._build_cell_orders(family, (2, 2), mesh)
    @test flat.class === nothing
    level = space(Ω; cells=(2, 2), order=2, basis=family, active=[CartesianIndex(1, 1), CartesianIndex(2, 2)]).levels[1]
    modes = level.modes
    @test length(modes.sets) == 2                        # the empty set plus the shared one
    @test modes.kind === nothing                         # no per-cell map: every active cell has the shared set
    @test modes.sets[2] == local_basis_indices(family, (2, 2))
    @test cell_basis_indices(level, CartesianIndex(1, 1)) === modes.sets[2]
    @test isempty(cell_basis_indices(level, CartesianIndex(1, 2)))   # the mask, not a table, says which cells generate
end

@testset "per-cell order: a uniform level stores no per-cell arrays" begin
    # A per-cell order class and mode kind on a uniform level would be constants stored once per cell — four bytes on
    # every cell of most levels, and tens of millions of cells on a deep ladder. The uniform level stores neither, so
    # its record is the same size at any cell count, while a graded level carries both; the readers answer the same
    # either way, and `cell_orders` round-trips through `elevate` in both representations.
    for active in (nothing, [!(i == 2 && j == 3) for i in 1:4, j in 1:4])
        V = space(_CO_2D; cells=4, order=3, active=active)
        lvl = V.levels[1]
        @test lvl.orders.class === nothing && lvl.modes.kind === nothing
        @test lvl.orders.palette == [(3, 3)] && length(lvl.modes.sets) == 2
        for c in cell_indices(V; level=1)
            @test cell_order(lvl, c) == (3, 3)
            on = Unfitted.is_active(lvl.mask, c)
            @test cell_basis_indices(lvl, c) === (on ? lvl.modes.sets[2] : lvl.modes.sets[1])
            on && @test Unfitted._active_set(lvl.modes, c) === cell_basis_indices(lvl, c)
        end
        @test cell_orders(V; level=1) == fill((3, 3), 4, 4)
        back = elevate(V, 1 => cell_orders(V; level=1))
        @test back.levels[1].orders.class === nothing && back.levels[1].modes.kind === nothing
        @test cell_orders(back; level=1) == cell_orders(V; level=1)
        @test all(c -> cell_basis_indices(back.levels[1], c) == cell_basis_indices(lvl, c), cell_indices(V; level=1))
    end
    small, large = space(_CO_2D; cells=4, order=3).levels[1], space(_CO_2D; cells=64, order=3).levels[1]
    @test Base.summarysize(large.orders) == Base.summarysize(small.orders)
    @test Base.summarysize(large.modes) == Base.summarysize(small.modes)

    field = [(2 + (i + j) % 2, 2 + i % 3) for i in 1:4, j in 1:4]
    graded = space(_CO_2D; cells=4, order=field)
    lvl = graded.levels[1]
    @test lvl.orders.class isa Array{UInt16,2} && size(lvl.orders.class) == (4, 4)
    @test lvl.modes.kind isa Array{UInt16,2} && size(lvl.modes.kind) == (4, 4)
    @test cell_orders(graded; level=1) == field
    @test all(c -> cell_order(lvl, c) == field[c], cell_indices(graded; level=1))
    @test cell_orders(elevate(graded, 1 => cell_orders(graded; level=1)); level=1) == field
    # Elevating a graded level to one order drops both arrays again, and a pair edit of a uniform one builds them.
    flat = elevate(graded, 1 => 3).levels[1]
    @test flat.orders.class === nothing && flat.modes.kind === nothing
    raised = elevate(space(_CO_2D; cells=4, order=3), 1 => CartesianIndex(2, 2) => 4).levels[1]
    @test raised.orders.class isa Array{UInt16,2} && raised.modes.kind isa Array{UInt16,2}
end

@testset "per-cell order: the per-cell readers refuse a cell off the grid" begin
    # A uniform level has no per-cell array whose indexing would raise, so the readers check the cell against the grid
    # themselves; a cell off the grid must not quietly read as the level's one order or set. The refusal stays the
    # `BoundsError` an array read raised before.
    Ω = _CO_2D
    for V in (space(Ω; cells=(4, 3), order=2), space(Ω; cells=(4, 3), order=2, active=(b, c) -> c[1] <= 2),
              space(Ω; cells=(4, 3), order=[1 + (i + j) % 3 for i in 1:4, j in 1:3]))
        lvl = V.levels[1]
        for c in (CartesianIndex(5, 1), CartesianIndex(1, 4), CartesianIndex(0, 2))
            @test_throws BoundsError cell_order(lvl, c)
            @test_throws BoundsError cell_order(V, c; level=1)
            @test_throws BoundsError cell_basis_indices(lvl, c)
            @test_throws BoundsError basis_values(lvl, c, (0.0, 0.0))
            @test_throws BoundsError Unfitted.physical_basis_gradients(lvl, c, box((0.0, 0.0), (1.0, 1.0)), (0.0, 0.0))
        end
    end
end
