using StaticArrays: SVector

@testset "basis scaffold" begin
    basis = IntegratedLegendre()

    @test Unfitted.basis_name(basis) == :integrated_legendre
    @test Unfitted.recommended_quadrature_order(basis, (2, 3)) == (3, 4)
    @test Unfitted.local_basis_count(basis, (2, 3)) == 12
end

@testset "local_basis_count is the size of the index set it counts" begin
    # The count must refuse what the set refuses. Integrated Legendre needs order ≥ 1 on every axis (order 0 leaves one
    # endpoint mode, not a partition of unity); the count used to answer `prod(order .+ 1)` there regardless.
    basis = IntegratedLegendre()
    @test_throws ArgumentError Unfitted.local_basis_count(basis, (0, 0))
    @test_throws ArgumentError Unfitted.local_basis_count(basis, (2, 0))
    @test_throws ArgumentError Unfitted.local_basis_count(Unfitted._with_mode(basis, :trunk), (0, 0))
    for order in ((1,), (2, 3), (1, 2, 3), (4, 4, 4))
        @test Unfitted.local_basis_count(basis, order) ==
              length(Unfitted.local_basis_indices(basis, order)) ==
              prod(order .+ 1)
    end
end

@testset "integrated Legendre 1D" begin
    for ξ in (-1.0, -0.3, 0.2, 1.0)
        @test Unfitted.integrated_legendre_value(0, ξ) ≈ (1.0 - ξ) / 2
        @test Unfitted.integrated_legendre_value(1, ξ) ≈ (1.0 + ξ) / 2
        @test Unfitted.integrated_legendre_derivative(0, ξ) ≈ -0.5
        @test Unfitted.integrated_legendre_derivative(1, ξ) ≈ 0.5
    end

    for i in 2:8
        @test abs(Unfitted.integrated_legendre_value(i, -1.0)) < 1.0e-13
        @test abs(Unfitted.integrated_legendre_value(i, 1.0)) < 1.0e-13

        ξ = -0.7 + 1.4 * (i - 2) / 6
        h = 1.0e-6
        fd = (Unfitted.integrated_legendre_value(i, ξ + h) - Unfitted.integrated_legendre_value(i, ξ - h)) / (2h)
        @test Unfitted.integrated_legendre_derivative(i, ξ) ≈ fd rtol = 1.0e-7 atol = 1.0e-8
    end
end

@testset "the hot factor tables are the closed forms, bit for bit" begin
    # `_fill_factor_tables!` fills every mode's value and derivative from one Legendre recurrence and one square root
    # per mode; `integrated_legendre_value` / `_derivative` evaluate each mode from its own closed form and are the
    # independent reference. They must agree exactly: the recurrence is the same expression, and the derivative's
    # constant √((2m − 1)/2) is the value's √(4m − 2) halved, which is exact under a correctly rounded square root
    # because both scalings are powers of two. The values-only call (`der1d === nothing`) must fill the same values.
    basis = IntegratedLegendre()
    p = 10
    for T in (Float64, Float32, BigFloat)
        @test all(m -> sqrt(T(4m - 2)) / 2 == sqrt(T(2m - 1) / 2), 2:64)
        for x in T.((-1, -0.7, -0.25, 0, 0.3, 0.85, 1))
            val1d, der1d, values_only = (Unfitted._factor_buffers((p,), T) for _ in 1:3)
            Unfitted._fill_factor_tables!(basis, val1d, der1d, (p,), SVector{1,T}(x), CartesianIndex(1))
            Unfitted._fill_factor_tables!(basis, values_only, nothing, (p,), SVector{1,T}(x), CartesianIndex(1))
            @test eltype(val1d[1]) === eltype(der1d[1]) === T
            @test val1d[1] == [Unfitted.integrated_legendre_value(m, x) for m in 0:p]
            @test der1d[1] == [Unfitted.integrated_legendre_derivative(m, x) for m in 0:p]
            @test values_only[1] == val1d[1]
        end
    end

    # Each axis reads its own coordinate and fills exactly its own order's modes.
    ξ = SVector(0.35, -0.6, 0.9)
    order = (4, 10, 2)
    val1d, der1d = Unfitted._factor_buffers(order, Float64), Unfitted._factor_buffers(order, Float64)
    Unfitted._fill_factor_tables!(basis, val1d, der1d, order, ξ, CartesianIndex(1, 1, 1))
    for d in 1:3
        @test val1d[d] == [Unfitted.integrated_legendre_value(m, ξ[d]) for m in 0:order[d]]
        @test der1d[d] == [Unfitted.integrated_legendre_derivative(m, ξ[d]) for m in 0:order[d]]
    end
end

# The per-cell evaluators are keyed by a level and a cell. Reference-cell evaluation is a one-cell level over
# [−1, 1]ᴰ: its only cell is the reference cell, so the chain-rule factor `2 / h_d` is 1 and its physical gradients are
# the bare reference gradients ∂N_α/∂ξ.
function _reference_level(order::NTuple{D,Int}; mode::Symbol=:tensor) where {D}
    return space(box(ntuple(_ -> -1.0, D), ntuple(_ -> 1.0, D)); cells=1, order, mode).levels[1]
end
_reference_cell(::NTuple{D,Int}) where {D} = CartesianIndex(ntuple(_ -> 1, D))

@testset "tensor basis values and ordering" begin
    basis = IntegratedLegendre()

    @test Unfitted.local_basis_indices(basis, (1, 2)) ==
          [CartesianIndex(0, 0), CartesianIndex(1, 0), CartesianIndex(0, 1), CartesianIndex(1, 1), CartesianIndex(0, 2),
           CartesianIndex(1, 2)]

    # The level-keyed evaluator on the reference cell, 2D to 4D: the counts and the partition of unity.
    values = Unfitted.basis_values(_reference_level((1, 1)), _reference_cell((1, 1)), (0.25, -0.5))
    @test length(values) == 4
    @test sum(values) ≈ 1.0

    values3 = Unfitted.basis_values(_reference_level((1, 2, 1)), _reference_cell((1, 2, 1)), (0.0, 0.25, -0.25))
    @test length(values3) == 12

    values4 = Unfitted.basis_values(_reference_level((1, 1, 1, 1)), _reference_cell((1, 1, 1, 1)),
                                    (0.0, 0.1, -0.2, 0.3))
    @test length(values4) == 16
    @test sum(values4) ≈ 1.0
end

@testset "the per-cell evaluators are keyed by a level, not by an order" begin
    # The order-keyed `basis_values(basis, order, mode, xi, cell)` / `physical_basis_gradients(basis, order, mode, box,
    # xi, cell)` are gone. They took any order and any cell and handed both to the family's kernel, which on a B-spline
    # family reads the knot vector at the span the cell names without a bounds check, and their index set was the
    # order's rather than the cell's. The level-keyed pair reads both from the level.
    @test !hasmethod(Unfitted.basis_values,
                     Tuple{Unfitted.BasisFamily,NTuple{1,Int},Symbol,NTuple{1,Float64},CartesianIndex{1}})
    @test !hasmethod(Unfitted.physical_basis_gradients,
                     Tuple{Unfitted.BasisFamily,NTuple{1,Int},Symbol,AxisBox{1,Float64},NTuple{1,Float64},
                           CartesianIndex{1}})
end

@testset "trunk basis filtering" begin
    basis = IntegratedLegendre()
    # The mode is folded into the family: `_with_mode` is what `space`, `overlay` and `ladder` apply to their `mode`
    # keyword, and every index-set question then takes the family alone.
    trunk = Unfitted._with_mode(basis, :trunk)

    # 2D counts: trunk trims only the interior, so p=2 (no interior modes yet) matches tensor minus the single (2,2)
    # interior mode.
    @test Unfitted.local_basis_count(trunk, (2, 2)) == 8
    @test Unfitted.local_basis_count(trunk, (3, 3)) == 12
    # 3D is where trunk pays off: the interior face/volume modes are the bulk of the tensor count at moderate order.
    @test Unfitted.local_basis_count(trunk, (2, 2, 2)) == 20
    @test Unfitted.local_basis_count(trunk, (3, 3, 3)) == 32
    @test Unfitted.local_basis_count(trunk, (4, 4, 4)) == 50

    # Edge modes survive up to degree p; interior modes only while their bubble degrees sum to ≤ p, so (2,2) is dropped
    # at p=3 and reappears at p=4.
    @test CartesianIndex(2, 2) ∉ Unfitted.local_basis_indices(trunk, (2, 2))
    @test CartesianIndex(3, 1) ∈ Unfitted.local_basis_indices(trunk, (3, 3))
    @test CartesianIndex(2, 2) ∉ Unfitted.local_basis_indices(trunk, (3, 3))
    @test CartesianIndex(2, 2) ∈ Unfitted.local_basis_indices(trunk, (4, 4))
    @test_throws ArgumentError Unfitted.local_basis_indices(trunk, (2, 3))

    values = Unfitted.basis_values(_reference_level((1, 1); mode=:trunk), _reference_cell((1, 1)), (0.2, -0.3))
    @test length(values) == 4
    @test sum(values) ≈ 1.0

    # Mode validation is family-aware: integrated Legendre is the family that carries `:trunk` as well as `:tensor`, and
    # a mode name no family defines is rejected before any family is asked. The B-spline suite covers the other branch —
    # a defined mode a family does not carry. The keyword always wins, so a trunk family asked for `:tensor` gets
    # `:tensor`, and `_mode` reads the mode back.
    @test all(m -> Unfitted._mode(Unfitted._with_mode(basis, m)) === m, (:tensor, :trunk))
    @test Unfitted._mode(Unfitted._with_mode(trunk, :tensor)) === :tensor
    @test Unfitted._mode(basis) === :tensor
    @test_throws ArgumentError Unfitted._with_mode(basis, :serendipity)
    # The tensor family keeps its one public spelling as its display; the trunk one is marked.
    @test sprint(show, basis) == "IntegratedLegendre()"
    @test sprint(show, trunk) != sprint(show, basis)

    lower = Unfitted.boundary_basis_indices(trunk, (2, 2); axis=1, side=:lower)
    @test length(lower) == 3
    @test all(id.I[1] == 0 for id in lower)
end

@testset "tensor basis gradients" begin
    # On a one-cell level over [−1, 1]ᴰ the cell *is* the reference cell, so its chain-rule factor `2 / h_d` is 1 and
    # the physical gradient the evaluator returns is the bare reference gradient ∂N_α/∂ξ.
    level1 = _reference_level((2,))
    reference1 = box((-1.0,), (1.0,))
    @test cell_box(level1.mesh, CartesianIndex(1)) == reference1
    gradients1 = Unfitted.physical_basis_gradients(level1, CartesianIndex(1), reference1, (0.25,))
    @test length(gradients1) == 3
    @test gradients1[1][1] ≈ -0.5
    @test gradients1[2][1] ≈ 0.5

    order = (2, 2)
    ξ = (0.15, -0.35)
    at = CartesianIndex(1, 1)
    level2 = _reference_level(order)
    reference2 = box((-1.0, -1.0), (1.0, 1.0))
    gradients = Unfitted.physical_basis_gradients(level2, at, reference2, ξ)
    h = 1.0e-6

    values_plus = Unfitted.basis_values(level2, at, (ξ[1] + h, ξ[2]))
    values_minus = Unfitted.basis_values(level2, at, (ξ[1] - h, ξ[2]))
    for i in eachindex(gradients)
        @test gradients[i][1] ≈ (values_plus[i] - values_minus[i]) / (2h) rtol = 1.0e-7 atol = 1.0e-8
    end

    # The chain rule, on a real one-cell level of edge lengths (2, 4) and its own cell box.
    cell = box((2.0, -1.0), (4.0, 3.0))
    physical_level = space(cell; cells=1, order).levels[1]
    physical = Unfitted.physical_basis_gradients(physical_level, at, cell_box(physical_level.mesh, at), ξ)
    @test physical[1] ≈ gradients[1] .* (2 ./ Unfitted.edge_lengths(cell))

    level3 = _reference_level((1, 1, 1))
    gradients3 = Unfitted.physical_basis_gradients(level3, CartesianIndex(1, 1, 1),
                                                   box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), (0.2, -0.1, 0.4))
    @test length(gradients3) == 8
    @test length(gradients3[1]) == 3
end

@testset "basis boundary metadata and quadrature" begin
    basis = IntegratedLegendre()

    lower = Unfitted.boundary_basis_indices(basis, (2, 3); axis=1, side=:lower)
    upper = Unfitted.boundary_basis_indices(basis, (2, 3); axis=2, side=:upper)

    @test all(id.I[1] == 0 for id in lower)
    @test length(lower) == 4
    @test all(id.I[2] == 1 for id in upper)
    @test length(upper) == 3
    @test Unfitted.is_boundary_basis(basis, CartesianIndex(1, 1, 0), 1, :upper)
    @test !Unfitted.is_boundary_basis(basis, CartesianIndex(2, 1, 0), 1, :upper)

    q = Unfitted.gauss_rule(basis, (2, 3), Float64)
    @test length(q.points) == 12
    @test length(q.weights) == 12
    @test sum(q.weights) ≈ 4.0
    @test all(all(-1.0 < x < 1.0 for x in point) for point in q.points)
end
