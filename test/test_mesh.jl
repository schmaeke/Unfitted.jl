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

@testset "a mesh refuses integer coordinates and names the float spelling" begin
    # Interior nodes are fractions of the extent. An integer domain used to raise an `InexactError` from inside the node
    # computation (two cells) or to build a mesh no `GeometryTolerance` exists for (one cell); both spellings, and both
    # entry points, now refuse up front and say what to write instead.
    for build in (() -> space(box((0, 0), (1, 1)); cells=2), () -> mesh(box((0, 0), (1, 1)); cells=2),
                  () -> mesh(box((0, 0), (1, 1)); cells=1))
        err = try
            build()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("box((0.0, 0.0), (1.0, 1.0))", sprint(showerror, err))
    end
    # A rational domain is untouched: its nodes are exact.
    @test mesh(box((0 // 1,), (1 // 1,)); cells=3).axes[1] == [0, 1 // 3, 2 // 3, 1]
end

@testset "moved_space names an out-of-range level index" begin
    # A level past the last one is an index error, not the base level: the message must say so, as every other entry
    # point that takes a level index does. The base level itself keeps its own refusal.
    V = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=4), box((0.25, 0.25), (0.75, 0.75)); cells=2)
    to = box((0.2, 0.2), (0.7, 0.7))
    @test_throws "out of bounds" moved_space(V; level=5, to)
    @test_throws "out of bounds" moved_space(V; level=0, to)
    @test_throws "only overlay levels" moved_space(V; level=1, to)
    @test moved_space(V; level=2, to).levels[2].mesh.domain == to
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
