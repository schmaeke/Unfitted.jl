@testset "basis scaffold" begin
    basis = IntegratedLegendre()

    @test Unfitted.basis_name(basis) == :integrated_legendre
    @test Unfitted.recommended_quadrature_order(basis, (2, 3)) == (3, 4)
    @test Unfitted.local_basis_count(basis, (2, 3)) == 12
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
        fd = (Unfitted.integrated_legendre_value(i, ξ + h) -
              Unfitted.integrated_legendre_value(i, ξ - h)) / (2h)
        @test Unfitted.integrated_legendre_derivative(i, ξ) ≈ fd rtol = 1.0e-7 atol = 1.0e-8
    end
end

@testset "tensor basis values and ordering" begin
    basis = IntegratedLegendre()

    @test Unfitted.local_basis_indices(basis, (1, 2)) ==
          [CartesianIndex(0, 0), CartesianIndex(1, 0), CartesianIndex(0, 1), CartesianIndex(1, 1),
           CartesianIndex(0, 2), CartesianIndex(1, 2)]

    # `basis_values` carries the parent cell in its signature for every family;
    # integrated Legendre ignores it, so any in-bounds index does here.
    values = Unfitted.basis_values(basis, (1, 1), :tensor, (0.25, -0.5), CartesianIndex(1, 1))
    @test length(values) == 4
    @test sum(values) ≈ 1.0

    values3 = Unfitted.basis_values(basis, (1, 2, 1), :tensor, (0.0, 0.25, -0.25),
                                    CartesianIndex(1, 1, 1))
    @test length(values3) == 12

    values4 = Unfitted.basis_values(basis, (1, 1, 1, 1), :tensor, (0.0, 0.1, -0.2, 0.3),
                                    CartesianIndex(1, 1, 1, 1))
    @test length(values4) == 16
    @test sum(values4) ≈ 1.0
end

@testset "trunk basis filtering" begin
    basis = IntegratedLegendre()

    # 2D counts: trunk trims only the interior, so p=2 (no interior modes
    # yet) matches tensor minus the single (2,2) interior mode.
    @test Unfitted.local_basis_count(basis, (2, 2), :trunk) == 8
    @test Unfitted.local_basis_count(basis, (3, 3), :trunk) == 12
    # 3D is where trunk pays off: the interior face/volume modes are the
    # bulk of the tensor count at moderate order.
    @test Unfitted.local_basis_count(basis, (2, 2, 2), :trunk) == 20
    @test Unfitted.local_basis_count(basis, (3, 3, 3), :trunk) == 32
    @test Unfitted.local_basis_count(basis, (4, 4, 4), :trunk) == 50

    # Edge modes survive up to degree p; interior modes only while their
    # bubble degrees sum to ≤ p, so (2,2) is dropped at p=3 and reappears
    # at p=4.
    @test CartesianIndex(2, 2) ∉ Unfitted.local_basis_indices(basis, (2, 2), :trunk)
    @test CartesianIndex(3, 1) ∈ Unfitted.local_basis_indices(basis, (3, 3), :trunk)
    @test CartesianIndex(2, 2) ∉ Unfitted.local_basis_indices(basis, (3, 3), :trunk)
    @test CartesianIndex(2, 2) ∈ Unfitted.local_basis_indices(basis, (4, 4), :trunk)
    @test_throws ArgumentError Unfitted.local_basis_indices(basis, (2, 3), :trunk)

    values = Unfitted.basis_values(basis, (1, 1), :trunk, (0.2, -0.3), CartesianIndex(1, 1))
    @test length(values) == 4
    @test sum(values) ≈ 1.0

    # Mode validation is family-aware: integrated Legendre is the family that
    # declares `:trunk`, and a mode name no family defines is rejected by the
    # family-blind gate the family-aware form delegates to. The B-spline suite
    # covers the other branch — a defined mode a family does not carry.
    @test Unfitted._supported_modes(basis) == (:tensor, :trunk)
    @test Unfitted._check_basis_mode(basis, :trunk, (2, 2)) === :trunk
    @test_throws ArgumentError Unfitted._check_basis_mode(basis, :serendipity, (2, 2))

    lower = Unfitted.boundary_basis_indices(basis, (2, 2); axis=1, side=:lower, mode=:trunk)
    @test length(lower) == 3
    @test all(id.I[1] == 0 for id in lower)
end

@testset "tensor basis gradients" begin
    basis = IntegratedLegendre()
    # A cell whose edge length is 2 in every axis *is* the reference cell, so
    # its chain-rule factor `2 / h_d` is 1 and the physical gradient the single
    # entry point returns is the bare reference gradient ∂N_α/∂ξ.
    reference1 = box((-1.0,), (1.0,))
    gradients1 = Unfitted.physical_basis_gradients(basis, (2,), :tensor, reference1, (0.25,),
                                                   CartesianIndex(1))
    @test length(gradients1) == 3
    @test gradients1[1][1] ≈ -0.5
    @test gradients1[2][1] ≈ 0.5

    order = (2, 2)
    ξ = (0.15, -0.35)
    at = CartesianIndex(1, 1)
    reference2 = box((-1.0, -1.0), (1.0, 1.0))
    gradients = Unfitted.physical_basis_gradients(basis, order, :tensor, reference2, ξ, at)
    h = 1.0e-6

    values_plus = Unfitted.basis_values(basis, order, :tensor, (ξ[1] + h, ξ[2]), at)
    values_minus = Unfitted.basis_values(basis, order, :tensor, (ξ[1] - h, ξ[2]), at)
    for i in eachindex(gradients)
        @test gradients[i][1] ≈ (values_plus[i] - values_minus[i]) / (2h) rtol = 1.0e-7 atol = 1.0e-8
    end

    cell = box((2.0, -1.0), (4.0, 3.0))
    physical = Unfitted.physical_basis_gradients(basis, order, :tensor, cell, ξ, at)
    @test physical[1] ≈ gradients[1] .* (2 ./ Unfitted.edge_lengths(cell))

    reference3 = box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    gradients3 = Unfitted.physical_basis_gradients(basis, (1, 1, 1), :tensor, reference3,
                                                   (0.2, -0.1, 0.4), CartesianIndex(1, 1, 1))
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
