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

# Tolerant containment: `contains_point` widens every face by `tol.contain`,
# so a point just outside a face is accepted while the overshoot stays within
# the tolerance and rejected once it exceeds it. Guards the conservative
# `lower − contain ≤ x ≤ upper + contain` contract relied on by cell lookup.
@testset "geometry tolerant containment" begin
    b = box((0.0, 0.0), (1.0, 1.0))
    tol = GeometryTolerance(Float64; contain=1.0e-6)

    # interior point accepted, clearly-exterior point rejected (slack inert)
    @test Unfitted.contains_point((0.5, 0.5), b, tol)
    @test !Unfitted.contains_point((0.5, 2.0), b, tol)

    # just past the upper x-face: inside the slack, then beyond it
    @test Unfitted.contains_point((1.0 + 0.5e-6, 0.5), b, tol)
    @test !Unfitted.contains_point((1.0 + 2.0e-6, 0.5), b, tol)

    # just past the lower y-face: inside the slack, then beyond it
    @test Unfitted.contains_point((0.5, -0.5e-6), b, tol)
    @test !Unfitted.contains_point((0.5, -2.0e-6), b, tol)
end

# `box_intersection` reports an overlap only when the result has strictly
# positive measure on every axis: a genuine overlap yields the component-wise
# max-lower / min-upper box, while a disjoint pair and a face-touching
# (zero-width) pair both yield `nothing` rather than a degenerate flat box.
@testset "geometry box_intersection disjoint and touching" begin
    a = box((0.0, 0.0), (2.0, 2.0))

    @test Unfitted.box_intersection(a, box((1.0, 0.5), (3.0, 1.5))) == box((1.0, 0.5), (2.0, 1.5))
    @test Unfitted.box_intersection(a, box((3.0, 3.0), (4.0, 4.0))) === nothing
    @test Unfitted.box_intersection(a, box((2.0, 0.0), (3.0, 2.0))) === nothing
end

# The `AxisBox` constructor enforces `lower[i] < upper[i]` on every axis: an
# inverted or equal corner on any axis must raise, so degenerate (zero- or
# negative-measure) regions never reach the assembly hot loop. Covers both the
# `box` helper and the direct `AxisBox` constructor.
@testset "geometry degenerate box rejected" begin
    @test_throws ArgumentError box((1.0, 0.0), (0.0, 1.0))            # inverted axis 1
    @test_throws ArgumentError box((0.0, 0.0), (1.0, 0.0))            # equal axis 2 (zero width)
    @test_throws ArgumentError box((0.0, 0.0, 0.0), (1.0, 0.0, 1.0))  # equal inner axis (3D)
    @test_throws ArgumentError AxisBox((1.0,), (0.0,))               # inverted, 1D, direct ctor
end

# Volume, center, edge lengths, box_intersection, and merge_coordinates are all
# dimension-generic; this exercises them with exact values across D = 1, 2, 3, 4
# through NTuple constructors and a loop, guarding the package's D-generic
# contract (no per-dimension special cases). The box on each axis i is [0, i+1],
# so volume = ∏(i+1) = (D+1)! and the center is upper/2.
@testset "geometry D-generic exact values" begin
    merge_tol = GeometryTolerance(Float64; merge=1.0e-9)
    for D in 1:4
        lower = ntuple(_ -> 0.0, D)
        upper = ntuple(i -> Float64(i + 1), D)
        b = box(lower, upper)

        @test Unfitted.dimension(b) == D
        @test Unfitted.volume(b) == Float64(factorial(D + 1))
        @test Tuple(Unfitted.center(b)) == ntuple(i -> (i + 1) / 2, D)
        @test Tuple(Unfitted.edge_lengths(b)) == ntuple(i -> Float64(i + 1), D)

        # Intersect with [0.5, i+2] per axis: the overlap is exactly [0.5, i+1].
        other = box(ntuple(_ -> 0.5, D), ntuple(i -> Float64(i + 2), D))
        expected = box(ntuple(_ -> 0.5, D), ntuple(i -> Float64(i + 1), D))
        @test Unfitted.box_intersection(b, other) == expected
        @test Unfitted.volume(expected) == prod(ntuple(i -> i + 0.5, D))

        # merge_coordinates collapses each near-duplicate cluster (separated by
        # < tol.merge) to its first representative: D clusters at 1, …, D.
        coords = Float64[]
        for i in 1:D
            push!(coords, Float64(i) + 0.3e-9)
            push!(coords, Float64(i))
        end
        @test Unfitted.merge_coordinates(coords, merge_tol) == [Float64(i) for i in 1:D]
    end
end
