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

@testset "cell_box refuses a cell off the grid with an error that can be printed" begin
    # The refusal is a `BoundsError`, and printing it must name the cell rather than fail inside `showerror` (a
    # hand-built `BoundsError(mesh, ::CartesianIndex)` could be thrown but not displayed). The `Space` form reaches the
    # same check.
    m = mesh(box((0.0, 0.0), (1.0, 1.0)); cells=3)
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=3)
    for refuse in (() -> cell_box(m, CartesianIndex(4, 1)), () -> cell_box(V, CartesianIndex(4, 1); level=1))
        err = try
            refuse()
            nothing
        catch e
            e
        end
        @test err isa BoundsError
        @test occursin("CartesianIndex(4, 1)", sprint(showerror, err))
    end
end

@testset "_level_by_id resolves every level and rejects an unknown id" begin
    # The lookup recurses over the level tuple (see its comment in mesh.jl); pin its contract on a three-level ladder.
    V = ladder(box((0.0, 0.0), (1.0, 1.0)); cells=4, order=2, depth=2, splits=2)
    @test length(V.levels) == 3
    for level in V.levels
        @test Unfitted._level_by_id(V, level.id) === level
    end
    @test_throws ArgumentError Unfitted._level_by_id(V, maximum(level.id for level in V.levels) + 1)
end
