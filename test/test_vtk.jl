using StaticArrays
using WriteVTK      # loads UnfittedWriteVTKExt, which provides write_vtk, vtk_series and write_quadrature_vtm

# The VTK writer's tests: what it makes of the records `src/postprocessing.jl` samples — the files, their layout, the
# cell types, the number types and the padding — and what it refuses. The records themselves are pinned in
# `test_postprocessing.jl`, whose fixtures `_cut_depth_disk_model`, `_pin_model` and `_pin_2d` the tests below reuse;
# `runtests.jl` includes that file first, and this one does not stand alone.

# Parse one `<DataArray>` back out of an ASCII `.vtu`/`.vtp` by name, so the tests can assert the numbers ParaView will
# show. Requires `ascii=true, append=false` on the write; the appended-binary form carries no parsable text.
_vtu_values(xml, name) = split(match(Regex("Name=\"$name\"[^>]*>([^<]*)</DataArray>"), xml).captures[1])
_vtu_ints(xml, name) = parse.(Int, _vtu_values(xml, name))
_vtu_floats(xml, name) = parse.(Float64, _vtu_values(xml, name))

@testset "the WriteVTK extension provides the VTK entry points" begin
    # The entry points are declared in the core and implemented by the extension `using WriteVTK` loads. Once it is
    # loaded, a call none of its methods accepts is an ordinary `MethodError`, not advice to load what is loaded; the
    # behaviour without the extension is pinned in a fresh process by test_api.jl.
    @test Base.get_extension(Unfitted, :UnfittedWriteVTKExt) !== nothing
    @test_throws MethodError write_vtk(1)
    @test_throws MethodError vtk_series(1)
    @test_throws MethodError write_quadrature_vtm(1)
    series = vtk_series(joinpath(mktempdir(), "run"))
    @test series isa VTKSeries
    @test close(series) isa String
end

@testset "partitioned VTK bundle with level meshes" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=2)
    # u ≡ 1 solves −Δu = 0 with u = 1 on ∂Ω and lies in the base space, so the overlay's p = 2 modes come back at zero
    # and every exported value is 1.
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(1.0; on=boundary(:all))]))
    solution = solve!(model)

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "case"), solution, model; subdivisions=:none, ascii=true, append=false,
                      compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi), twice=(u, c, x, xi) -> 2 * u(c, xi)),
                      cell_data=(midpoint_x=(u, c, x, xi) -> x[1],))

    @test joinpath(dir, "case.vtm") in files
    @test joinpath(dir, "case_data.vtu") in files
    @test joinpath(dir, "case_level_1_base_mesh.vtu") in files
    @test joinpath(dir, "case_level_2_overlay_mesh.vtu") in files
    @test all(isfile, files)

    solution_xml = read(joinpath(dir, "case_data.vtu"), String)
    # 5 admissible regions (the overlay box plus the four base remainders), one unsubdivided VTK cell each, 4 corners
    # per cell.
    uh = _vtu_floats(solution_xml, "uh")
    @test length(uh) == 20
    @test maximum(abs, uh .- 1) < 1.0e-12
    @test maximum(abs, _vtu_floats(solution_xml, "twice") .- 2) < 1.0e-12
    @test _vtu_ints(solution_xml, "region_id") == 1:5
    @test _vtu_ints(solution_xml, "cover_count") == [1, 1, 2, 1, 1]
    @test _vtu_floats(solution_xml, "midpoint_x") ≈ [0.5, 0.125, 0.5, 0.875, 0.5]

    mesh_xml = read(joinpath(dir, "case_level_2_overlay_mesh.vtu"), String)
    @test occursin("level_id", mesh_xml)
    @test occursin("role_id", mesh_xml)
    @test occursin("order_max", mesh_xml)
    @test occursin("active", mesh_xml)
    @test occursin("covered", mesh_xml)
    @test occursin("active_dofs", mesh_xml)
    @test occursin("reduced_dofs", mesh_xml)

    bundle_xml = read(joinpath(dir, "case.vtm"), String)
    @test occursin("case_data.vtu", bundle_xml)
    @test occursin("case_level_1_base_mesh.vtu", bundle_xml)
    @test occursin("case_level_2_overlay_mesh.vtu", bundle_xml)
end

@testset "coupled multi-domain VTK bundle" begin
    # A coupled model writes one block per field (named by field), each grouping that field's `data` grid with its own
    # mesh(es) as sibling leaf datasets — a homogeneous tree whose leaf names are disambiguated by ASCII index / level
    # id (not the possibly-Unicode field name) so ParaView's Extract Block resolves each leaf uniquely — plus one
    # quadrature block per subdomain.
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(2, 2), order=1)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    model = prepare(Problem((u1, u2); blocks=(stiffness_block(u1), stiffness_block(u2)),
                            loads=(source_load(u1; source=1.0), source_load(u2; source=1.0)),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=:u1),
                                       dirichlet(0.0; on=boundary(:all), field=:u2)]))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))
    dir = mktempdir()
    files = write_vtk(joinpath(dir, "cpl"), solution, model; subdivisions=:none, ascii=true, append=false,
                      compress=false)

    # one data `.vtu` per field, ASCII-indexed so Extract Block can isolate it
    @test joinpath(dir, "cpl_data_1.vtu") in files
    @test joinpath(dir, "cpl_data_2.vtu") in files
    vtm = read(joinpath(dir, "cpl.vtm"), String)
    # per-field blocks named u1 / u2, each grouping a data + mesh leaf
    @test occursin("name=\"u1\"", vtm)
    @test occursin("name=\"u2\"", vtm)
    # leaf names are disambiguated by ASCII index / level id (never by the possibly-Unicode field name, which ParaView's
    # data assembly would collapse)
    @test occursin("name=\"data_1\"", vtm) && occursin("name=\"data_2\"", vtm)
    @test occursin("name=\"level_1_base\"", vtm) && occursin("name=\"level_2_base\"", vtm)
    @test !occursin("name=\"meshes\"", vtm)        # no grouping sub-block: level meshes are direct leaves
    # only the two field blocks are `<Block>`s (homogeneous: leaves below them)
    @test count("<Block", vtm) == 2
    # one mesh per distinct level (ids 1 and 2), no duplication
    meshes = filter(f -> occursin("_mesh.vtu", f), files)
    @test length(meshes) == 2 == length(unique(meshes))

    # quadrature: one block per subdomain (covering-level signatures "1" and "2")
    qfiles = write_quadrature_vtm(joinpath(dir, "cplq"), model)
    @test joinpath(dir, "cplq_quadrature_levels_1.vtp") in qfiles
    @test joinpath(dir, "cplq_quadrature_levels_2.vtp") in qfiles
end

@testset "two fields on one space write each level mesh once" begin
    # The mesh dedup: fields sharing a space must not emit duplicate level blocks. A two-field/one-space model has a
    # single base level → one mesh.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    a = field(:a, V)
    b = field(:b, V)
    model = prepare(Problem((a, b); blocks=(stiffness_block(a), mass_block(b)), loads=(source_load(a; source=1.0),),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=:a)]))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))
    dir = mktempdir()
    files = write_vtk(joinpath(dir, "shared"), solution, model; subdivisions=:none, ascii=true, append=false,
                      compress=false)
    meshes = filter(f -> occursin("_mesh.vtu", f), files)
    @test length(meshes) == 1
end

@testset "VTK exports vector point data" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=2)
    u = field(:u, V; components=2)
    model = prepare(poisson(u; source=SVector(0.0, 0.0)))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "vector_case"), solution, model; subdivisions=:none, ascii=true, append=false,
                      compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    @test joinpath(dir, "vector_case_data.vtu") in files
    solution_xml = read(joinpath(dir, "vector_case_data.vtu"), String)
    @test occursin("uh", solution_xml)

    # A VTK data array is a vector only at three components; a 2-component array is stored but cannot be assigned as
    # VECTORS, so ParaView's Glyph / Warp By Vector cannot read it. The 2-D field is padded with a trailing zero, the
    # same padding the WriteVTK extension's `_vtk_points` applies to the geometry it is attached to.
    uh_header = match(r"<DataArray[^>]*Name=\"uh\"[^>]*>", solution_xml)
    @test uh_header !== nothing
    @test occursin("NumberOfComponents=\"3\"", uh_header.match)
    @test occursin("NumberOfComponents=\"3\"", match(r"<DataArray[^>]*Name=\"Points\"[^>]*>", solution_xml).match)
end

@testset "VTK export rejects dimensions above 3" begin
    omega = box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    V = space(omega; cells=(1, 1, 1, 1), order=1)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test_throws ArgumentError write_vtk(joinpath(mktempdir(), "case"), solution, model)
end

@testset "VTK time series indexes one full bundle per frame" begin
    # A series is `write_vtk` in a loop plus a `.pvd` index over the frames. The overlay moves between the two frames
    # here, which is the case the feature exists for: each frame carries its own level meshes, so the animation shows
    # the mesh following the feature and not just the field on a frozen mesh.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = overlay(space(omega; cells=(2, 2), order=1), box((0.1, 0.1), (0.5, 0.5)); cells=(1, 1), order=2)
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    dir = mktempdir()
    series = vtk_series(joinpath(dir, "run"))
    first_frame = write_vtk(series, 0.0, solve!(model), model; subdivisions=:none, ascii=true, append=false,
                            compress=false)
    @test first_frame == joinpath(dir, "run_frames", "frame_0001.vtm")
    @test isfile(joinpath(dir, "run_frames", "frame_0001_level_2_overlay_mesh.vtu"))

    # Byte-identity, the contract of the body split: a frame is the ordinary bundle. Written from the same
    # `(solution, model)` before the overlay moves.
    solo = solve!(model)
    write_vtk(joinpath(dir, "solo"), solo, model; subdivisions=:none, ascii=true, append=false, compress=false)
    @test read(joinpath(dir, "run_frames", "frame_0001_data.vtu")) == read(joinpath(dir, "solo_data.vtu"))

    move!(model; level=2, to=box((0.5, 0.5), (0.9, 0.9)))
    write_vtk(series, 0.25, solve!(model), model; subdivisions=:none, ascii=true, append=false, compress=false)
    @test read(joinpath(dir, "run_frames", "frame_0002_level_2_overlay_mesh.vtu")) !=
          read(joinpath(dir, "run_frames", "frame_0001_level_2_overlay_mesh.vtu"))

    collection = close(series)
    @test collection == joinpath(dir, "run.pvd")
    sets = collect(eachmatch(r"<DataSet timestep=\"([^\"]*)\"[^>]*file=\"([^\"]*)\"", read(collection, String)))
    @test length(sets) == 2
    @test parse.(Float64, [s.captures[1] for s in sets]) == [0.0, 0.25]
    @test [s.captures[2] for s in sets] ==
          [joinpath("run_frames", "frame_0001.vtm"), joinpath("run_frames", "frame_0002.vtm")]
    @test close(series) == collection           # closing twice is harmless
end

@testset "VTK series writes its frames where it is told" begin
    # The frames directory is a keyword, the collection names each frame relative to itself, and a `.pvd` given on the
    # path is stripped rather than doubled. The time is recorded exactly as handed over — no frame grid.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=1)
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    dir = mktempdir()
    series = vtk_series(joinpath(dir, "movie.pvd"); frames=joinpath(dir, "pics"))
    write_vtk(series, 1.5, solve!(model), model; level_meshes=false)

    @test isfile(joinpath(dir, "pics", "frame_0001.vtm"))
    @test isfile(joinpath(dir, "pics", "frame_0001_data.vtu"))
    xml = read(close(series), String)
    @test occursin("file=\"" * joinpath("pics", "frame_0001.vtm") * "\"", xml)
    @test occursin("timestep=\"1.5\"", xml)
end

@testset "quadrature VTM splits regions by parent levels" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=1)
    model = prepare(poisson(V; source=0.0))

    dir = mktempdir()
    files = write_quadrature_vtm(joinpath(dir, "qp"), model)

    @test joinpath(dir, "qp.vtm") in files
    @test joinpath(dir, "qp_quadrature_levels_1.vtp") in files
    @test joinpath(dir, "qp_quadrature_levels_1_2.vtp") in files
    @test all(isfile, files)

    base_only_xml = read(joinpath(dir, "qp_quadrature_levels_1.vtp"), String)
    @test occursin("weight", base_only_xml)

    coupled_xml = read(joinpath(dir, "qp_quadrature_levels_1_2.vtp"), String)
    @test occursin("weight", coupled_xml)

    bundle_xml = read(joinpath(dir, "qp.vtm"), String)
    @test occursin("qp_quadrature_levels_1.vtp", bundle_xml)
    @test occursin("qp_quadrature_levels_1_2.vtp", bundle_xml)
end

@testset "cut_depth survives the series byte-identity contract" begin
    # A refined frame must still be the same bytes as the solo write it mirrors: the tree walk may not introduce
    # ordering that depends on anything but the geometry.
    _, model, solution = _cut_depth_disk_model()
    dir = mktempdir()
    series = vtk_series(joinpath(dir, "run"); frames=joinpath(dir, "frames"))
    write_vtk(series, 0.0, solution, model; subdivisions=:none, cut_depth=2, ascii=true, append=false, compress=false,
              point_data=(uh=(u, c, x, xi) -> u(c, xi),))
    close(series)
    write_vtk(joinpath(dir, "solo"), solution, model; subdivisions=:none, cut_depth=2, ascii=true, append=false,
              compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    @test read(joinpath(dir, "frames", "frame_0001_data.vtu")) == read(joinpath(dir, "solo_data.vtu"))
end

@testset "write_vtk compresses by default" begin
    # The default is `compress = true`, matching `WriteVTK`'s own. Pinned by what the files declare and weigh: the
    # default `.vtu` names the zlib compressor and is under half the size of the uncompressed one. That the encodings
    # decode to the same arrays is WriteVTK's to keep and is not decoded here.
    _, model, solution = _cut_depth_disk_model()
    dir = mktempdir()
    pd = (uh=(u, c, x, xi) -> u(c, xi),)

    write_vtk(joinpath(dir, "on"), solution, model; subdivisions=4, cut_depth=:none, point_data=pd)
    write_vtk(joinpath(dir, "off"), solution, model; subdivisions=4, cut_depth=:none, point_data=pd, compress=false)

    on_xml = read(joinpath(dir, "on_data.vtu"), String)
    off_xml = read(joinpath(dir, "off_data.vtu"), String)
    @test occursin("compressor=\"vtkZLibDataCompressor\"", on_xml)
    @test !occursin("compressor=", off_xml)
    @test filesize(joinpath(dir, "on_data.vtu")) < filesize(joinpath(dir, "off_data.vtu")) / 2

    # The same write as parsable text carries a `uh` and a `level_set` value at every point.
    write_vtk(joinpath(dir, "plain"), solution, model; subdivisions=4, cut_depth=:none, point_data=pd, ascii=true,
              append=false, compress=false)
    plain_xml = read(joinpath(dir, "plain_data.vtu"), String)
    @test length(_vtu_floats(plain_xml, "uh")) == length(_vtu_floats(plain_xml, "level_set"))
end

# ── Pins: the writer's output contract ────────────────────────────────────────
#
# What a rewrite of the writer must keep, pinned end to end on tiny models built once and shared: that a bundle's
# datasets are the records it was handed, serialised; the VTK cell type and corner order per dimension; the type and
# component count each callback value is written with; the files a bundle consists of and the list `write_vtk` returns;
# and the size of the quadrature cloud. What the records hold — the subdivision rules, the accepted spellings of the
# keywords, the values a callback may return — is pinned in `test_postprocessing.jl`.

# The attributes of the `<DataArray>` named `name` in a `.vtu` / `.vtp`, as a `Dict`; readable in every encoding.
function _vtu_tag(xml, name)
    tag = match(Regex("<DataArray[^>]*Name=\"$name\"[^>]*>"), xml).match
    return Dict(m.captures[1] => m.captures[2] for m in eachmatch(r"(\w+)=\"([^\"]*)\"", tag))
end

# An integer attribute of the `<Piece>` of a `.vtu` / `.vtp` (`NumberOfPoints`, `NumberOfCells`, `NumberOfVerts`, …).
_vtu_piece(xml, attribute) = parse(Int, match(Regex("<Piece[^>]*$attribute=\"(\\d+)\""), xml).captures[1])

# The keywords that make a bundle parsable text.
_pin_ascii = (ascii=true, append=false, compress=false)

# 1D: two order-1 cells. 3D: one order-1 cell of an off-origin box with unequal edges, so every corner of it has
# distinct coordinates. The 2D model, `_pin_2d` — two cells of per-cell anisotropic orders under a one-cell overlay — is
# `test_postprocessing.jl`'s, as is `_pin_model`.
_pin_1d = _pin_model(space(box((0.0,), (1.0,)); cells=2, order=1))
_pin_3d = _pin_model(space(box((0.5, -1.0, 2.0), (1.5, 0.25, 4.0)); cells=1, order=1))

@testset "pin: a bundle's datasets are its records, as VTK stores them" begin
    # The writer samples nothing itself: the data grid of a bundle is the record `_sample_fields` returns and each level
    # mesh the record `_sample_levels` returns, serialised — the points padded to three coordinates, the cells in record
    # order over the implicit connectivity, and every array, scalar or vector, holding the record's values. With the
    # record tests of `test_postprocessing.jl` this pins what a bundle shows end to end.
    _, model, solution = _cut_depth_disk_model()
    keywords = (subdivisions=2, cut_depth=2, cell_data=(region=(u, c, x, xi) -> c.region_id,),
                point_data=(uh=(u, c, x, xi) -> u(c, xi), xy=(u, c, x, xi) -> (x[1], x[2])))
    dir = mktempdir()
    write_vtk(joinpath(dir, "rec"), solution, model; keywords..., _pin_ascii...)
    records = ("rec_data.vtu" => only(Unfitted._sample_fields(solution, model; keywords...)).data,
               "rec_level_1_base_mesh.vtu" => only(Unfitted._sample_levels(model)).grid)
    for (file, grid) in records
        xml = read(joinpath(dir, file), String)
        n = length(grid.points)
        @test reshape(_vtu_floats(xml, "Points"), 3, :) == [reduce(hcat, grid.points); zeros(1, n)]
        @test _vtu_ints(xml, "connectivity") == 0:(n-1)
        @test _vtu_ints(xml, "offsets") == 4:4:n
        # `xy` is the one vector array, of two components, which VTK stores padded with a zero third.
        for (name, array) in [grid.point_data; grid.cell_data]
            stored = eltype(array) <: SVector ? vec([reduce(hcat, array); zeros(1, length(array))]) : array
            @test _vtu_floats(xml, name) == stored
        end
    end
end

@testset "pin: a 1D bundle is VTK_LINE cells of two points each" begin
    (; model, sol) = _pin_1d
    dir = mktempdir()
    write_vtk(joinpath(dir, "line"), sol, model; subdivisions=:none, _pin_ascii...)
    for file in ("line_data.vtu", "line_level_1_base_mesh.vtu")
        xml = read(joinpath(dir, file), String)
        @test _vtu_piece(xml, "NumberOfCells") == 2
        @test _vtu_piece(xml, "NumberOfPoints") == 4
        @test _vtu_ints(xml, "types") == [3, 3]                  # VTK_LINE
        @test _vtu_ints(xml, "offsets") == [2, 4]
        @test _vtu_ints(xml, "connectivity") == 0:3
        @test reshape(_vtu_floats(xml, "Points"), 3, :) == [0.0 0.5 0.5 1.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    end
end

@testset "pin: a 3D bundle is VTK_HEXAHEDRON cells with their corners in VTK order" begin
    (; V, model, sol) = _pin_3d
    dir = mktempdir()
    write_vtk(joinpath(dir, "hex"), sol, model; subdivisions=:none, _pin_ascii...)
    b = cell_box(V, CartesianIndex(1, 1, 1); level=1)
    l, u = b.lower, b.upper
    # VTK's hexahedron: the lower face counterclockwise, then the upper face in the same order.
    corners = [l[1] u[1] u[1] l[1] l[1] u[1] u[1] l[1]
               l[2] l[2] u[2] u[2] l[2] l[2] u[2] u[2]
               l[3] l[3] l[3] l[3] u[3] u[3] u[3] u[3]]
    for file in ("hex_data.vtu", "hex_level_1_base_mesh.vtu")
        xml = read(joinpath(dir, file), String)
        @test _vtu_ints(xml, "types") == [12]                    # VTK_HEXAHEDRON
        @test _vtu_ints(xml, "offsets") == [8]
        @test _vtu_ints(xml, "connectivity") == 0:7
        @test reshape(_vtu_floats(xml, "Points"), 3, :) == corners
    end
end

@testset "pin: the written type and component count of each callback value shape" begin
    (; model, sol) = _pin_2d
    dir = mktempdir()
    shapes = (pair=(u, c, x, xi) -> (x[1], x[2]),                 # a 2-tuple: padded to a 3-vector
              mixed_tuple=(u, c, x, xi) -> (1, x[1], 2),          # Int and Float64 components: promoted
              vector=(u, c, x, xi) -> SVector(x[1], x[2], 1.0), int_vector=(u, c, x, xi) -> SVector(1, 2, c.region_id),
              mixed=(u, c, x, xi) -> c.region_id == 1 ? 1 : 0.5,  # Int at some samples, Float64 at others: promoted
              int=(u, c, x, xi) -> c.region_id, single=(u, c, x, xi) -> Float32(x[1]))
    written = (pair=("Float64", "3"), mixed_tuple=("Float64", "3"), vector=("Float64", "3"), int_vector=("Int64", "3"),
               mixed=("Float64", "1"), int=("Int64", "1"), single=("Float32", "1"))
    write_vtk(joinpath(dir, "shapes"), sol, model; subdivisions=:none, level_meshes=false, point_data=shapes,
              cell_data=(cell_pair=shapes.pair,), _pin_ascii...)
    xml = read(joinpath(dir, "shapes_data.vtu"), String)
    for (name, (type, components)) in pairs(written)
        tag = _vtu_tag(xml, name)
        @test (tag["type"], tag["NumberOfComponents"]) == (type, components)
    end
    @test (_vtu_tag(xml, "cell_pair")["type"], _vtu_tag(xml, "cell_pair")["NumberOfComponents"]) == ("Float64", "3")
    points = reshape(_vtu_floats(xml, "Points"), 3, :)
    @test reshape(_vtu_floats(xml, "pair"), 3, :) == [points[1:2, :]; zeros(1, size(points, 2))]
    @test reshape(_vtu_floats(xml, "mixed_tuple"), 3, :) == [ones(1, 8); points[1:1, :]; fill(2.0, 1, 8)]
    @test _vtu_floats(xml, "mixed") == [fill(1.0, 4); fill(0.5, 4)]
    @test reshape(_vtu_floats(xml, "cell_pair"), 3, :)[3, :] == zeros(2)
end

@testset "pin: field= given as a Field writes that field alone, as `data`" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    a = field(:a, V)
    b = field(:b, V)
    model = prepare(Problem((a, b); blocks=(stiffness_block(a), mass_block(b)), loads=(source_load(a; source=1.0),),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=:a)]))
    sol = solution(model, [sin(i) for i in 1:active_unknowns(model)])
    dir = mktempdir()
    files = write_vtk(joinpath(dir, "byfield"), sol, model; field=b, subdivisions=:none, _pin_ascii...)
    @test basename.(files) == ["byfield.vtm", "byfield_data.vtu", "byfield_level_1_base_mesh.vtu"]
    vtm = read(joinpath(dir, "byfield.vtm"), String)
    @test occursin("name=\"b\"", vtm) && !occursin("name=\"a\"", vtm)
    # The default `uh` entry evaluates the written field, `b`, not the model's first one.
    xml = read(joinpath(dir, "byfield_data.vtu"), String)
    points = reshape(_vtu_floats(xml, "Points"), 3, :)
    @test _vtu_floats(xml, "uh") ≈ [value(sol, model, b, (points[1, k], points[2, k])) for k in axes(points, 2)]
    # The Field and its name write the same bytes.
    write_vtk(joinpath(dir, "byname"), sol, model; field=:b, subdivisions=:none, _pin_ascii...)
    @test read(joinpath(dir, "byfield_data.vtu")) == read(joinpath(dir, "byname_data.vtu"))
end

@testset "pin: level_meshes=false writes no level mesh" begin
    (; model, sol) = _pin_2d
    dir = mktempdir()
    files = write_vtk(joinpath(dir, "bare"), sol, model; level_meshes=false)
    @test basename.(files) == ["bare.vtm", "bare_data.vtu"]
    @test sort(readdir(dir)) == ["bare.vtm", "bare_data.vtu"]
    @test !occursin("_mesh", read(joinpath(dir, "bare.vtm"), String))
end

@testset "pin: the quadrature cloud holds one vertex per quadrature point" begin
    (; model) = _pin_2d
    dir = mktempdir()
    files = write_quadrature_vtm(joinpath(dir, "cloud"), model)
    pieces = [read(file, String) for file in files if endswith(file, ".vtp")]
    @test length(pieces) == 2                                # signatures [1] and [1, 2]
    @test sum(xml -> _vtu_piece(xml, "NumberOfVerts"), pieces) == nquadpoints(model)
    @test sum(xml -> _vtu_piece(xml, "NumberOfPoints"), pieces) == nquadpoints(model)
end

@testset "pin: write_vtk(path, …) returns every file it wrote, the .vtm first" begin
    (; model, sol) = _pin_2d
    dir = mktempdir()
    files = write_vtk(joinpath(dir, "all"), sol, model)
    @test files isa Vector{String}
    @test first(files) == joinpath(dir, "all.vtm")
    @test length(files) == 4                                 # the data grid and two level meshes
    @test sort(files) == sort(joinpath.(dir, readdir(dir)))
end

# ── Regressions: the writer's fixed bugs ──────────────────────────────────────

@testset "a refused or throwing write leaves nothing on disk" begin
    # Refusals of a keyword or of a callback's value, and an exception a callback raises — here only while the second
    # field of a coupled model is sampled, after the first field's block is complete — all happen while the bundle is
    # still in memory. The writer used to make the directory first and save whatever it had on the way out, leaving a
    # stub `.vtm` and partial grids behind.
    (; model, sol) = _pin_2d
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(1, 1), order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(1, 1), order=1)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    coupled = prepare(Problem((u1, u2); blocks=(mass_block(u1), mass_block(u2))))
    coupled_sol = solution(coupled, ones(active_unknowns(coupled)))
    cases = [(ArgumentError, model, sol, (; subdivisions=0)), (ArgumentError, model, sol, (; subdivisions=(2, 2, 2))),
             (ArgumentError, model, sol, (; subdivisions=:bad)), (ArgumentError, model, sol, (; cut_depth=9)),
             (ArgumentError, model, sol, (; field=:nope)), (ArgumentError, model, sol, (; point_data=[1])),
             (ArgumentError, model, sol, (; point_data=(text=(u, c, x, xi) -> "text",))),
             (ErrorException, coupled, coupled_sol,
              (; cell_data=(late=(u, c, x, xi) -> x[1] > 1.5 ? error("second field") : 0.0,)))]
    for (exception, m, s, keywords) in cases
        dir = mktempdir()
        @test_throws exception write_vtk(joinpath(dir, "sub", "x"), s, m; keywords...)
        @test isempty(readdir(dir))
    end
    # A series frame that fails while it is sampled writes nothing either, and does not use up its frame number.
    dir = mktempdir()
    series = vtk_series(joinpath(dir, "run"))
    @test_throws ErrorException write_vtk(series, 0.0, sol, model; point_data=(late=(u, c, x, xi) -> error("frame"),))
    @test isempty(readdir(joinpath(dir, "run_frames")))
    @test series.count[] == 0
    @test write_vtk(series, 0.0, sol, model; level_meshes=false) == joinpath(dir, "run_frames", "frame_0001.vtm")
    close(series)
end

@testset "a value VTK cannot store is refused naming its entry" begin
    # The sampling keeps a real value in the type it was computed in, so a `Rational` or a `BigFloat` reaches the
    # writer, and WriteVTK would refuse it as `data type not supported by VTK`, naming neither the entry nor what the
    # writer accepts. The writer refuses it first, naming both, and like every refusal it leaves nothing on disk. A
    # value of no accepted shape at all is the sampling's to refuse (`test_postprocessing.jl`).
    (; model, sol) = _pin_2d
    for (name, f) in (("rational", (u, c, x, xi) -> 1 // 3), ("big", (u, c, x, xi) -> big(x[1])))
        dir = mktempdir()
        err = try
            write_vtk(joinpath(dir, "x"), sol, model; subdivisions=:none, point_data=NamedTuple{(Symbol(name),)}((f,)))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        message = err === nothing ? "" : sprint(showerror, err)
        @test occursin("point_data entry \"$name\"", message)
        @test occursin("which VTK cannot store", message)
        @test isempty(readdir(dir))
    end
end

@testset "a Bool callback value is written as UInt8 0 / 1" begin
    # VTK has no boolean type, and WriteVTK refused a Bool array outright (`data type not supported by VTK: Bool`),
    # although a Bool is the natural value of a flag or a mask. It is written as UInt8 0 / 1: alone, in an SVector or in
    # a tuple, and padded to three components like any other 2-vector.
    (; model, sol) = _pin_2d
    dir = mktempdir()
    write_vtk(joinpath(dir, "flags"), sol, model; subdivisions=:none, level_meshes=false, _pin_ascii...,
              point_data=(right=(u, c, x, xi) -> x[1] > 0.5, corner=(u, c, x, xi) -> SVector(x[1] > 0.5, x[2] > 0.5),
                          pair=(u, c, x, xi) -> (x[1] > 0.5, x[2] > 0.5)),
              cell_data=(first=(u, c, x, xi) -> c.region_id == 1,))
    xml = read(joinpath(dir, "flags_data.vtu"), String)
    for (name, components) in (("right", "1"), ("corner", "3"), ("pair", "3"), ("first", "1"))
        tag = _vtu_tag(xml, name)
        @test (tag["type"], tag["NumberOfComponents"]) == ("UInt8", components)
    end
    points = reshape(_vtu_floats(xml, "Points"), 3, :)
    @test _vtu_ints(xml, "right") == Int.(points[1, :] .> 0.5)
    @test reshape(_vtu_ints(xml, "corner"), 3, :) == [Int.(points[1:2, :] .> 0.5); zeros(Int, 1, size(points, 2))]
    @test _vtu_ints(xml, "pair") == _vtu_ints(xml, "corner")
    @test _vtu_ints(xml, "first") == Int.(_vtu_ints(xml, "region_id") .== 1)
end

@testset "a frame written to a closed series is refused up front" begin
    # `close` writes the collection and frees it. A later frame used to be sampled in full and then fail inside the XML
    # library (`XMLNoRootError`); it is now refused before any sampling, writes nothing and keeps the frame count.
    (; model, sol) = _pin_2d
    dir = mktempdir()
    series = vtk_series(joinpath(dir, "run"))
    write_vtk(series, 0.0, sol, model; level_meshes=false)
    close(series)
    frames = readdir(joinpath(dir, "run_frames"))
    sampled = Ref(0)
    err = try
        write_vtk(series, 1.0, sol, model; level_meshes=false, point_data=(n=(u, c, x, xi) -> (sampled[] += 1; 0.0),))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("closed series", err === nothing ? "" : sprint(showerror, err))
    @test sampled[] == 0
    @test readdir(joinpath(dir, "run_frames")) == frames
    @test series.count[] == 1
end

@testset "a dotted base name writes the same files without warnings" begin
    # WriteVTK appends a file's extension when it is missing, but warns when the name carries any other one, and a
    # time-stamped base name such as `out_t=0.5` has one (`.5`): every file of its bundle used to warn. The writers now
    # name each file with its extension — the data grid, the level meshes, the quadrature clouds, a series frame and its
    # collection — and the names are unchanged.
    (; model, sol) = _pin_2d
    base = joinpath(mktempdir(), "out_t=0.5")
    files = @test_logs write_vtk(base, sol, model)
    @test files == base .* [".vtm", "_data.vtu", "_level_1_base_mesh.vtu", "_level_2_overlay_mesh.vtu"]
    cloud = joinpath(mktempdir(), "cloud_t=0.5")
    @test (@test_logs write_quadrature_vtm(cloud, model)) ==
          [cloud * ".vtm", cloud * "_quadrature_levels_1.vtp", cloud * "_quadrature_levels_1_2.vtp"]
    run = joinpath(mktempdir(), "run_t=0.5")
    series = @test_logs vtk_series(run)
    @test (@test_logs write_vtk(series, 0.5, sol, model; level_meshes=false)) ==
          joinpath(run * "_frames", "frame_0001.vtm")
    @test (@test_logs close(series)) == run * ".pvd"
    @test isfile(run * ".pvd")
end
