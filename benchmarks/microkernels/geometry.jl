using BenchmarkTools
using StaticArrays
using Unfitted
using Unfitted: merge_coordinates, box_intersection, locate_cell, CartesianMesh

group = SUITE["microkernels"]["geometry"] = BenchmarkGroup()

# merge_coordinates: three overlapping per-axis coordinate sets, like the input
# `_axis_intervals` sees from a base + two overlays.
let
    tol = GeometryTolerance(Float64)
    coords = vcat(collect(0.0:(1/32):1.0), collect(0.2:(1/17):0.8), collect(0.47:(1/11):0.86))
    group["merge_coordinates n=$(length(coords))"] = @benchmarkable merge_coordinates($coords, $tol)
end

# box_intersection: a partially overlapping D=3 pair.
let
    a = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    b = box((0.3, 0.2, 0.1), (0.9, 0.8, 0.95))
    group["box_intersection D=3"] = @benchmarkable box_intersection($a, $b)
end

# locate_cell: D=2 and D=3 binary-search path.
let
    tol = GeometryTolerance(Float64)
    mesh2 = CartesianMesh(box((0.0, 0.0), (1.0, 1.0)); cells=(32, 32))
    mesh3 = CartesianMesh(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(16, 16, 16))
    pt2 = SVector{2,Float64}(0.37, 0.59)
    pt3 = SVector{3,Float64}(0.37, 0.59, 0.23)
    group["locate_cell D=2"] = @benchmarkable locate_cell($mesh2, $pt2; tol=($tol))
    group["locate_cell D=3"] = @benchmarkable locate_cell($mesh3, $pt3; tol=($tol))
end
