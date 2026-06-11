@testset "mesh scaffold" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    m = mesh(omega; cells=(2, 4))

    @test m.cells == (2, 4)
    @test Unfitted.cell_count(m) == 8
    @test Unfitted.cell_box(m, CartesianIndex(1, 1)) == box((0.0, 0.0), (0.5, 0.25))
    @test Unfitted.locate_cell(m, (0.75, 0.75)) == CartesianIndex(2, 4)
    @test Unfitted.locate_cell(m, (1.0, 1.0)) == CartesianIndex(2, 4)
    @test length(collect(Unfitted.cell_indices(m))) == 8
    @test Unfitted.boundary_coordinates(m)[1] == [0.0, 0.5, 1.0]

    m4 = mesh(box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)); cells=(1, 2, 1, 2))
    @test Unfitted.cell_count(m4) == 4
    @test Unfitted.locate_cell(m4, (0.5, 0.75, 0.5, 0.25)) == CartesianIndex(1, 2, 1, 1)

    V = space(omega; cells=(2, 2), order=1)
    @test_throws ArgumentError overlay(V, box((-0.1, 0.0), (0.5, 0.5)); cells=(1, 1))
end
