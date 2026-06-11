@testset "geometry scaffold" begin
    b = box((0.0, 0.0), (2.0, 3.0))

    @test Tuple(b.lower) == (0.0, 0.0)
    @test Tuple(b.upper) == (2.0, 3.0)
    @test Unfitted.dimension(b) == 2
    @test Unfitted.volume(b) == 6.0
    @test Tuple(Unfitted.center(b)) == (1.0, 1.5)
    @test (1.0, 1.0) in b

    for D in 1:4
        lower = ntuple(_ -> 0.0, D)
        upper = ntuple(i -> Float64(i), D)
        point = ntuple(i -> 0.25 * i, D)
        boxD = box(lower, upper)
        xi = Unfitted.physical_to_reference(boxD, point)
        x = Unfitted.reference_to_physical(boxD, xi)

        @test all(isapprox(x[i], point[i]; atol=1.0e-14) for i in 1:D)
        @test all(-1.0 <= xi[i] <= 1.0 for i in 1:D)
    end

    a = box((0.0, 0.0), (1.0, 1.0))
    c = box((0.5, 0.25), (1.5, 0.75))
    @test Unfitted.box_intersection(a, c) == box((0.5, 0.25), (1.0, 0.75))
    @test box((0.5, 0.5); halfwidth=0.25) == box((0.25, 0.25), (0.75, 0.75))
    @test box((0.5, 0.5); halfwidth=(0.25, 0.1)) == box((0.25, 0.4), (0.75, 0.6))

    tol = GeometryTolerance(Float64; merge=1.0e-8, contain=1.0e-8, small_volume=1.0e-12)
    @test Unfitted.merge_coordinates([0.0, 1.0, 1.0 + 0.5e-8, 2.0], tol) == [0.0, 1.0, 2.0]
    @test Unfitted.is_inside(box((0.0, 0.0), (1.0, 1.0)), b)
end
