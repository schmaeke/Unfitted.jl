using StaticArrays
using WriteVTK      # loads UnfittedWriteVTKExt, which provides write_vtk, vtk_series and write_quadrature_vtm

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

@testset "a VTK callback reads only fields of the block's own subdomain" begin
    # The sample accessor evaluates fields on the parents of the region being written, which are cells of the block's
    # own subdomain. A field of the other subdomain of a coupled model owns none of them, so asking for it is a usage
    # error with a pointer to the point evaluation, not a wrong number or a crash deep in the dof layer.
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
    foreign = (other=(u, c, x, xi) -> u(c, xi, u2),)
    err = try
        write_vtk(joinpath(dir, "foreign"), solution, model; field=:u1, level_meshes=false, subdivisions=:none,
                  point_data=foreign)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("value(solution, model, u2, x)", sprint(showerror, err))
end

@testset "a VTK callback reads another field of the same subdomain" begin
    # Two fields on one space: writing `a`, a callback may evaluate `b` at the sample, and gets the value the point
    # evaluation gives at the same `x`.
    V = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(3, 3), order=2), box((0.0, 0.0), (0.5, 0.5)); cells=(2, 2),
                order=3)
    a = field(:a, V)
    b = field(:b, V)
    model = prepare(Problem((a, b); blocks=(stiffness_block(a), mass_block(b)), loads=(source_load(a; source=1.0),),
                            dirichlet=[dirichlet(x -> 1.0 + x[1]; on=boundary(:all), field=:a)]))
    n = Unfitted.active_unknowns(model.dofs)
    solution = Solution([sin(3i) for i in 1:n], model.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))
    gaps = Float64[]
    record(gap) = (push!(gaps, abs(gap)); gap)
    point_data = (own=(u, c, x, xi) -> record(u(c, xi) - value(solution, model, a, x)),
                  other=(u, c, x, xi) -> record(u(c, xi, b) - value(solution, model, b, x)))
    write_vtk(joinpath(mktempdir(), "named"), solution, model; field=:a, level_meshes=false, subdivisions=2, point_data)
    @test length(gaps) > 100
    @test maximum(gaps) ≤ 1.0e-12
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

@testset "level mesh exports all cells with an active flag" begin
    # A 3x3 overlay with only one active cell now exports ALL nine cells as solid cells, distinguished by the `active`
    # cell-data (1 for the active cell, 0 for the eight masked ones) — no longer a filtered wireframe.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    active = falses(3, 3)
    active[2, 2] = true
    V = overlay(V, box((0.1, 0.1), (0.9, 0.9)); cells=(3, 3), order=1, active=active)
    model = prepare(poisson(V; source=0.0))
    coefficients = ones(Unfitted.active_unknowns(model.dofs))
    soln = Solution(coefficients, model.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "masked"), soln, model; subdivisions=:none, ascii=true, append=false,
                      compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    overlay_file = joinpath(dir, "masked_level_2_overlay_mesh.vtu")
    @test overlay_file in files
    overlay_xml = read(overlay_file, String)
    @test occursin("NumberOfCells=\"9\"", overlay_xml)   # all cells exported, not filtered
    @test occursin("active", overlay_xml)
    @test occursin("covered", overlay_xml)

    # The single base cell is exported too, as its own mesh block.
    base_xml = read(joinpath(dir, "masked_level_1_base_mesh.vtu"), String)
    @test occursin("NumberOfCells=\"1\"", base_xml)
end

@testset "level-mesh cell data reports coverage and reduction per cell" begin
    # The solid-cell per-level mesh carries `covered` / `reduced_dofs` / `active` cell data; assert the actual values,
    # not just that the arrays are emitted. A 4×4 p=2 base with an aligned p=3 overlay over the middle 2×2 block covers
    # four base cells and sheds high-order (plus one deduped vertex) there.
    V = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2), box((0.25, 0.25), (0.75, 0.75));
                cells=(4, 4), order=3)
    model = prepare(mass(V))
    sol = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                   Unfitted.SolverDiagnostics(:manual, 0.0, true))
    dir = mktempdir()
    write_vtk(joinpath(dir, "red"), sol, model; subdivisions=:none, ascii=true, append=false)

    base = read(joinpath(dir, "red_level_1_base_mesh.vtu"), String)
    covered = _vtu_ints(base, "covered")
    reduced = _vtu_ints(base, "reduced_dofs")
    active = _vtu_ints(base, "active")

    @test length(covered) == 16                    # base is 4×4
    @test sum(covered) == 4                         # the middle 2×2 block is covered
    @test all(active .== 1)                         # unmasked base → every cell active
    @test all(reduced[covered .== 0] .== 0)         # reduction only under coverage
    @test sum(reduced[covered .== 1]) > 0           # covered cells shed high-order modes
end

@testset "VTK export rejects dimensions above 3" begin
    omega = box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    V = space(omega; cells=(1, 1, 1, 1), order=1)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test_throws ArgumentError write_vtk(joinpath(mktempdir(), "case"), solution, model)
end

@testset "VTK auto-emits level_set for models with a PhysicalDomain" begin
    # A circular cut of the unit square: cells crossing r = 0.4 are cut, cells fully inside are full, cells fully
    # outside (with α = 1 to keep them in the dof layout) are fictitious. The exported `level_set` array must equal
    # `φ(x)` at every VTK vertex so a ParaView contour at level 0 reproduces ∂Ω.
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    omega = box((0.0, 0.0), (1.0, 1.0))
    physical = physical_domain(phi_disk; lipschitz=1.0, alpha=1.0, subcell_length_scale=0.125, max_depth=3)
    V = space(omega; cells=(2, 2), order=1, physical=physical)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "disk"), solution, model; subdivisions=:none, ascii=true, append=false,
                      compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    solution_xml = read(joinpath(dir, "disk_data.vtu"), String)
    points = reshape(_vtu_floats(solution_xml, "Points"), 3, :)
    @test(_vtu_floats(solution_xml, "level_set") ≈ [phi_disk(view(points, :, i)) for i in axes(points, 2)],
          atol = 1.0e-14)
end

@testset "VTK omits level_set when no PhysicalDomain is attached" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    write_vtk(joinpath(dir, "plain"), solution, model; subdivisions=:none, ascii=true, append=false, compress=false,
              point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    solution_xml = read(joinpath(dir, "plain_data.vtu"), String)
    @test !occursin("level_set", solution_xml)
end

@testset "user-supplied level_set callback overrides the auto-emit" begin
    # A user `level_set` callback in `point_data` should win: the built-in is suppressed, the user's array (whose
    # constant value `42.0` cannot match φ) is written instead. Detecting the override via the constant payload keeps
    # the test independent of how WriteVTK serialises numeric arrays.
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    omega = box((0.0, 0.0), (1.0, 1.0))
    physical = physical_domain(phi_disk; lipschitz=1.0, alpha=1.0, subcell_length_scale=0.25, max_depth=2)
    V = space(omega; cells=(1, 1), order=1, physical=physical)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    write_vtk(joinpath(dir, "override"), solution, model; subdivisions=:none, ascii=true, append=false, compress=false,
              point_data=(uh=(u, c, x, xi) -> u(c, xi), level_set=(u, c, x, xi) -> 42.0))

    solution_xml = read(joinpath(dir, "override_data.vtu"), String)
    @test occursin("level_set", solution_xml)
    @test occursin("42", solution_xml)
    # The auto-emit would have written the φ value at e.g. (0, 0) ≈ 0.7071 − 0.4 = 0.307. The override means no such
    # value appears.
    @test !occursin("0.30710678", solution_xml)
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

# Per-cell x-extent of every quad in a parsed 2D `.vtu`, used by the `cut_depth` tests below: the refinement is uniform
# per level, so a cell's width identifies the tree level it came from.
function _vtu_quad_widths(xml)
    points = reshape(_vtu_floats(xml, "Points"), 3, :)
    connectivity = reshape(_vtu_ints(xml, "connectivity"), 4, :)
    return [begin
                xs = @view points[1, connectivity[:, c] .+ 1]
                maximum(xs) - minimum(xs)
            end
            for c in axes(connectivity, 2)]
end

# Which cells of a parsed 2D `.vtu` straddle ∂Ω, judged by the sign of the exported `level_set` at their corners —
# exactly the test ParaView's Clip applies when it decides whether to cut a cell.
function _vtu_mixed_cells(xml)
    phi = _vtu_floats(xml, "level_set")
    connectivity = reshape(_vtu_ints(xml, "connectivity"), 4, :)
    return [c for c in axes(connectivity, 2) if !allequal(phi[i + 1] <= 0 for i in connectivity[:, c])]
end

# A disk cutting every cell of a 2×2 mesh: four regions, all `:cut_fitted`.
function _cut_depth_disk_model()
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    omega = box((0.0, 0.0), (1.0, 1.0))
    physical = physical_domain(phi_disk; lipschitz=1.0, alpha=1.0, subcell_length_scale=0.125, max_depth=3)
    V = space(omega; cells=(2, 2), order=1, physical=physical)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))
    return phi_disk, model, solution
end

@testset "cut_depth bisects only the sub-cells ∂Ω passes through" begin
    # Four `:cut_fitted` regions of width 0.5. Each level of `cut_depth` bisects only the marked boxes, so the cell
    # count grows far slower than the 4ᵏ of a uniform 2D refinement (4, 16, 64, 256) once boxes start falling away from
    # the boundary.
    phi_disk, model, solution = _cut_depth_disk_model()
    dir = mktempdir()

    for (depth, cells) in ((0, 4), (1, 16), (2, 52), (3, 160))
        write_vtk(joinpath(dir, "d$(depth)"), solution, model; subdivisions=:none, cut_depth=depth, ascii=true,
                  append=false, compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))
        xml = read(joinpath(dir, "d$(depth)_data.vtu"), String)
        widths = _vtu_quad_widths(xml)
        @test length(widths) == cells

        # The crack invariant, stated executably: every cell ParaView will cut sits at the finest level, so two of them
        # meeting at a face carry identical corner coordinates and their clip facets agree exactly. Coarser cells
        # survive only where φ has one sign, and those contribute no cut surface at all — which is why no 2:1 balance
        # pass is needed.
        finest = 0.5 / 2^depth
        mixed = _vtu_mixed_cells(xml)
        @test !isempty(mixed)
        @test all(w -> w ≈ finest, widths[mixed])
        # Every width is a power-of-two fraction of the 0.5 base box, and the finest level is always reached. The
        # coarsest need not survive: this disk crosses all four regions, so at depth ≥ 1 every base box is marked and
        # nothing stays at 0.5.
        @test all(w -> any(l -> w ≈ 0.5 / 2^l, 0:depth), widths)
        @test minimum(widths) ≈ finest
    end
end

@testset "cut_depth keeps level_set exact on the refined path" begin
    # The refinement must not interpolate φ: every emitted vertex, including the ones the tree introduced, carries
    # `levelset_value` evaluated there.
    phi_disk, model, solution = _cut_depth_disk_model()
    dir = mktempdir()
    write_vtk(joinpath(dir, "exact"), solution, model; subdivisions=:none, cut_depth=2, ascii=true, append=false,
              compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    xml = read(joinpath(dir, "exact_data.vtu"), String)
    points = reshape(_vtu_floats(xml, "Points"), 3, :)
    @test _vtu_floats(xml, "level_set") ≈ [phi_disk(view(points, :, i)) for i in axes(points, 2)] atol = 1.0e-14
end

@testset "cut_depth hands cell callbacks the refined leaf" begin
    # `context.cell` must be the box actually emitted, not the `subdivisions` box it was carved from — the subtlest
    # plumbing error this change can introduce, and invisible in the geometry.
    _, model, solution = _cut_depth_disk_model()
    dir = mktempdir()
    write_vtk(joinpath(dir, "leaf"), solution, model; subdivisions=:none, cut_depth=2, ascii=true, append=false,
              compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),),
              cell_data=(w=(u, c, x, xi) -> c.cell.upper[1] - c.cell.lower[1],))

    xml = read(joinpath(dir, "leaf_data.vtu"), String)
    @test sort(unique(round.(_vtu_floats(xml, "w"); digits=12))) == [0.125, 0.25]
    @test _vtu_floats(xml, "w") ≈ _vtu_quad_widths(xml)
end

@testset "cut_depth spares full and fictitious regions" begin
    # A disk small enough to leave whole cells inside and whole cells outside it. `keep_fictitious` holds the outside
    # cells in the layout — α alone does not — so all three region kinds are present at once and only the cut ones may
    # be refined.
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.3
    omega = box((0.0, 0.0), (1.0, 1.0))
    physical = physical_domain(phi_disk; lipschitz=1.0, alpha=1.0, keep_fictitious=true, subcell_length_scale=1 / 32,
                               max_depth=4)
    V = space(omega; cells=(8, 8), order=1, physical=physical)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    plan = Unfitted.integration_plans(model)[1]
    kinds = [r.kind for r in plan.regions]
    @test :full in kinds
    @test :fictitious_alpha in kinds
    uncut = Set(i for (i, k) in pairs(kinds) if k === :full || k === :fictitious_alpha)
    @test !isempty(uncut)

    dir = mktempdir()
    write_vtk(joinpath(dir, "mixed"), solution, model; subdivisions=:none, cut_depth=2, ascii=true, append=false,
              compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))
    xml = read(joinpath(dir, "mixed_data.vtu"), String)
    widths = _vtu_quad_widths(xml)
    region_ids = _vtu_ints(xml, "region_id")

    # Every cell belonging to an uncut region is still a whole base box, and each such region contributes exactly the
    # one cell it did before: 44 uncut regions, 44 unrefined cells, every one of them 1/8 wide.
    @test length(uncut) == 44
    @test count(r -> r in uncut, region_ids) == 44
    @test all(widths[cell] ≈ 0.125 for (cell, region) in pairs(region_ids) if region in uncut)
    @test length(widths) > length(uncut)
end

@testset "cut_depth is inert without a PhysicalDomain and at :none" begin
    # The non-immersed path must not pay for, or notice, the new keyword, and `:none` must reproduce the unrefined
    # writer byte for byte.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(1.0; on=boundary(:all))]))
    solution = solve!(model)

    dir = mktempdir()
    reference = nothing
    for depth in (:none, 0, 3)
        write_vtk(joinpath(dir, "plain_$(depth)"), solution, model; subdivisions=2, cut_depth=depth, ascii=true,
                  append=false, compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))
        xml = read(joinpath(dir, "plain_$(depth)_data.vtu"), String)
        reference === nothing ? (reference = xml) : @test(xml == reference)
    end

    _, fcm_model, fcm_solution = _cut_depth_disk_model()
    for depth in (:none, 0)
        write_vtk(joinpath(dir, "fcm_$(depth)"), fcm_solution, fcm_model; subdivisions=:none, cut_depth=depth,
                  ascii=true, append=false, compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))
    end
    @test read(joinpath(dir, "fcm_none_data.vtu"), String) == read(joinpath(dir, "fcm_0_data.vtu"), String)
end

@testset "cut_depth rejects values outside its budget" begin
    # The budget is resolved before anything is sampled or a file opened, so a rejected call leaves nothing on disk —
    # the contract every refusal of the writer keeps, the dimension and coefficient checks included.
    _, model, solution = _cut_depth_disk_model()
    dir = mktempdir()
    for bad in (-1, 9, :deep, 2.5)
        @test_throws ArgumentError write_vtk(joinpath(dir, "bad"), solution, model; cut_depth=bad, subdivisions=:none)
    end
    @test isempty(readdir(dir))
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
# What a rewrite of the writer must keep, pinned end to end on three tiny models built once and shared: the VTK cell
# type and corner order per dimension, the subdivision rules, the accepted spellings of the keywords and of a callback's
# value, the files a bundle consists of and the list `write_vtk` returns, and the size of the quadrature cloud.

# The attributes of the `<DataArray>` named `name` in a `.vtu` / `.vtp`, as a `Dict`; readable in every encoding.
function _vtu_tag(xml, name)
    tag = match(Regex("<DataArray[^>]*Name=\"$name\"[^>]*>"), xml).match
    return Dict(m.captures[1] => m.captures[2] for m in eachmatch(r"(\w+)=\"([^\"]*)\"", tag))
end

# An integer attribute of the `<Piece>` of a `.vtu` / `.vtp` (`NumberOfPoints`, `NumberOfCells`, `NumberOfVerts`, …).
_vtu_piece(xml, attribute) = parse(Int, match(Regex("<Piece[^>]*$attribute=\"(\\d+)\""), xml).captures[1])

# The keywords that make a bundle parsable text.
_pin_ascii = (ascii=true, append=false, compress=false)

# A mass model on `V` and a solution with distinct coefficients, so that no two dofs carry the same value.
function _pin_model(V)
    model = prepare(mass(V))
    return (; V, model, sol=solution(model, [sin(i) for i in 1:active_unknowns(model)]))
end

# 1D: two order-1 cells. 2D: two cells of per-cell anisotropic orders (1, 2) and (3, 1), with an order-(1, 3) overlay
# over the right one, so the right region has two parents and `:degree` takes the per-axis maximum (3, 3) over them.
# 3D: one order-1 cell of an off-origin box with unequal edges, so every corner of it has distinct coordinates.
_pin_1d = _pin_model(space(box((0.0,), (1.0,)); cells=2, order=1))
_pin_2d = _pin_model(overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 1), order=reshape([(1, 2), (3, 1)], 2, 1)),
                             box((0.5, 0.0), (1.0, 1.0)); cells=1, order=(1, 3)))
_pin_3d = _pin_model(space(box((0.5, -1.0, 2.0), (1.5, 0.25, 4.0)); cells=1, order=1))

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

@testset "pin: :degree subdivides each region at its parents' per-axis maximum order" begin
    (; model, sol) = _pin_2d
    dir = mktempdir()
    write_vtk(joinpath(dir, "deg"), sol, model; level_meshes=false, _pin_ascii...)
    xml = read(joinpath(dir, "deg_data.vtu"), String)
    # The left region (one parent, order (1, 2)) is cut 1 × 2; the right one (orders (3, 1) and (1, 3)) 3 × 3.
    covers = _vtu_ints(xml, "cover_count")
    @test _vtu_piece(xml, "NumberOfCells") == 11
    @test count(==(1), covers) == 2
    @test count(==(2), covers) == 9
end

@testset "pin: subdivisions as an NTuple sets the per-axis counts" begin
    (; model, sol) = _pin_2d
    dir = mktempdir()
    write_vtk(joinpath(dir, "tup"), sol, model; subdivisions=(2, 3), level_meshes=false, _pin_ascii...)
    xml = read(joinpath(dir, "tup_data.vtu"), String)
    @test _vtu_piece(xml, "NumberOfCells") == 12             # two regions, 2 × 3 each
    points = reshape(_vtu_floats(xml, "Points"), 3, 4, :)
    @test all(c -> maximum(points[1, :, c]) - minimum(points[1, :, c]) == 0.25, axes(points, 3))
    @test all(c -> maximum(points[2, :, c]) - minimum(points[2, :, c]) ≈ 1 / 3, axes(points, 3))
end

@testset "pin: point_data and cell_data as a Dict or a tuple of Pairs" begin
    # The three spellings of the entries write the same arrays; only their order in the file is the container's.
    (; model, sol) = _pin_2d
    dir = mktempdir()
    uh = (u, c, x, xi) -> u(c, xi)
    tilt = (u, c, x, xi) -> x[1] - 2 * x[2]
    width = (u, c, x, xi) -> c.cell.upper[1] - c.cell.lower[1]
    spellings = (named=((uh=uh, tilt=tilt), (width=width,)),
                 dict=(Dict("uh" => uh, :tilt => tilt), Dict(:width => width)),
                 pairs=((:uh => uh, "tilt" => tilt), ("width" => width,)))
    xml = Dict(map(collect(pairs(spellings))) do (label, (point_data, cell_data))
                   write_vtk(joinpath(dir, "$label"), sol, model; level_meshes=false, point_data, cell_data,
                             _pin_ascii...)
                   return label => read(joinpath(dir, "$(label)_data.vtu"), String)
               end)
    for label in (:dict, :pairs), name in ("uh", "tilt", "width")
        @test _vtu_values(xml[label], name) == _vtu_values(xml[:named], name)
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
             (ArgumentError, model, sol, (; subdivisions=:bad)), (ArgumentError, model, sol, (; field=:nope)),
             (ArgumentError, model, sol, (; point_data=[1])),
             (ArgumentError, model, sol, (; point_data=(text=(u, c, x, xi) -> "text",))),
             (ErrorException, coupled, coupled_sol,
              (; cell_data=(late=(u, c, x, xi) -> x[1] > 1.5 ? error("second field") : 0.0,)))]
    for (exception, m, s, keywords) in cases
        dir = mktempdir()
        @test_throws exception write_vtk(joinpath(dir, "sub", "x"), s, m; keywords...)
        @test isempty(readdir(dir))
    end
end

@testset "SVector callback values are promoted over every sample" begin
    # The values of one entry share one element type, promoted over every sample as numbers and tuples already were.
    # Converting every SVector to the first sample's type refused a fractional value after an integer first sample
    # (`InexactError`) and rounded every `Float64` value after a `Float32` one. Region 1 is walked first here, so its
    # value is the first sample's.
    (; model, sol) = _pin_2d
    dir = mktempdir()
    first_int = (u, c, x, xi) -> c.region_id == 1 ? SVector(1, 2) : SVector(x[1], x[2] / 3)
    first_single = (u, c, x, xi) -> c.region_id == 1 ? SVector(0.5f0, 0.25f0) : SVector(x[1] / 3, x[2] / 3)
    for (name, f, region_one, other) in (("first_single", first_single, (0.5, 0.25), x -> (x[1] / 3, x[2] / 3)),
                                         ("first_int", first_int, (1.0, 2.0), x -> (x[1], x[2] / 3)))
        write_vtk(joinpath(dir, name), sol, model; subdivisions=:none, level_meshes=false, point_data=(v=f,),
                  _pin_ascii...)
        xml = read(joinpath(dir, name * "_data.vtu"), String)
        @test _vtu_tag(xml, "v")["type"] == "Float64"
        points = reshape(_vtu_floats(xml, "Points"), 3, :)
        regions = _vtu_ints(xml, "region_id")
        # Every value exactly what the callback returned, padded to three components.
        expected = [regions[(k - 1) ÷ 4 + 1] == 1 ? region_one : other(points[:, k]) for k in axes(points, 2)]
        @test reshape(_vtu_floats(xml, "v"), 3, :) == reduce(hcat, [[e..., 0.0] for e in expected])
    end
end

@testset "an unsupported callback value is refused naming its entry" begin
    # Each of these used to reach WriteVTK, which refused it as `data type not supported by VTK: Any` (or a
    # `DimensionMismatch`), naming neither the entry nor what it accepts. The refusal now names both, and like every
    # refusal it leaves nothing on disk.
    (; model, sol) = _pin_2d
    refused = [(:point_data, "vector", (u, c, x, xi) -> [x[1], x[2], 0.0]),          # a Vector, not an SVector
               (:point_data, "matrix", (u, c, x, xi) -> SMatrix{2,2}(x[1], 0.0, 0.0, x[2])),
               (:point_data, "text", (u, c, x, xi) -> "text"),
               (:point_data, "complex", (u, c, x, xi) -> complex(x[1], x[2])),
               (:point_data, "rational", (u, c, x, xi) -> 1 // 3),                     # real, but not a VTK type
               (:cell_data, "shapes", (u, c, x, xi) -> c.region_id == 1 ? x[1] : (x[1], x[2])),
               (:cell_data, "lengths", (u, c, x, xi) -> c.region_id == 1 ? SVector(x[1]) : SVector(x[1], x[2]))]
    for (kind, name, f) in refused
        dir = mktempdir()
        entries = NamedTuple{(Symbol(name),)}((f,))
        keywords = kind === :point_data ? (; point_data=entries) : (; cell_data=entries)
        err = try
            write_vtk(joinpath(dir, "x"), sol, model; subdivisions=:none, keywords...)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        message = err === nothing ? "" : sprint(showerror, err)
        @test occursin("$kind entry \"$name\"", message)
        @test occursin("a real number, a tuple of reals or an SVector of reals", message)
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
    # name each file with its extension; the names are unchanged.
    (; model, sol) = _pin_2d
    base = joinpath(mktempdir(), "out_t=0.5")
    files = @test_logs write_vtk(base, sol, model; level_meshes=false)
    @test files == [base * ".vtm", base * "_data.vtu"]
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

# ── Records: what a consumer reads ────────────────────────────────────────────
#
# The writers serialise consumer-neutral records (`Unfitted.SampleGrid`), which a plotting front end would read as they
# are. These internal tests pin the record contract itself, without decoding a file: the point and cell counts, the
# corner order, one value per point or per cell in every array, the built-in arrays and their order.

@testset "records: a sampled field's data grid and level mesh" begin
    phi_disk, model, sol = _cut_depth_disk_model()
    width = (u, c, x, xi) -> c.cell.upper[1] - c.cell.lower[1]
    records = Unfitted._sample_fields(sol, model; subdivisions=2, cut_depth=1, cell_data=(width=width,))
    (; field, data, meshes) = only(records)
    n = Unfitted._cell_count(data)
    @test n > 16                                             # four regions, a 2 × 2 lattice each, and the cut leaves
    @test length(data.points) == n << 2
    # Every cell's corners are the face-cyclic permutation (1, 2, 4, 3) of the `_box_vertices` of the box they span.
    @test all(1:n) do c
        corners = data.points[(4c-3):(4c)]
        return corners == collect(Unfitted._box_vertices(corners[1], corners[3]))[[1, 2, 4, 3]]
    end
    @test first.(data.point_data) == ["uh", "level_set"]
    @test first.(data.cell_data) == ["region_id", "cover_count", "width"]
    @test all(length(last(entry)) == length(data.points) for entry in data.point_data)
    @test all(length(last(entry)) == n for entry in data.cell_data)
    arrays = Dict(data.point_data..., data.cell_data...)
    @test arrays["level_set"] == [levelset_value(field.space.physical, x) for x in data.points]
    @test issorted(arrays["region_id"])
    @test arrays["width"] == [data.points[4c - 2][1] - data.points[4c - 3][1] for c in 1:n]

    @test [(m.level, m.role) for m in meshes] == [(1, :base)]
    mesh = only(meshes).grid
    @test Unfitted._cell_count(mesh) == 4
    @test isempty(mesh.point_data)
    @test first.(mesh.cell_data) ==
          ["level_id", "role_id", "order_max", "cell_id", "active", "covered", "active_dofs", "reduced_dofs"]
    @test all(entry -> last(entry) isa Vector{Int} && length(last(entry)) == 4, mesh.cell_data)
end

@testset "records: fields sharing a space carry its level meshes once" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    a = field(:a, V)
    b = field(:b, V)
    model = prepare(Problem((a, b); blocks=(mass_block(a), mass_block(b))))
    sol = solution(model, [sin(i) for i in 1:active_unknowns(model)])
    records = Unfitted._sample_fields(sol, model; subdivisions=:none, point_data=NamedTuple())
    @test [r.field.name for r in records] == [:a, :b]
    @test [length(r.meshes) for r in records] == [1, 0]
    @test all(r -> isempty(r.data.point_data), records)
    @test isempty(only(Unfitted._sample_fields(sol, model; field=b, level_meshes=false)).meshes)
end

@testset "records: the quadrature cloud weighs each signature's regions" begin
    # One vertex cloud per sorted covering-level signature, in sorted order. Its weights are the physical quadrature
    # weights, so they sum to the measure of that signature's regions — the left region of the two-cell mesh is covered
    # by the base level alone, the right one by the base and the overlay.
    (; model) = _pin_2d
    cloud = Unfitted._sample_quadrature(model)
    @test [entry.signature for entry in cloud] == [[1], [1, 2]]
    regions = Unfitted.integration_plans(model)[1].regions
    for (; signature, grid) in cloud
        @test Unfitted._cell_count(grid) == length(grid.points)
        @test first.(grid.point_data) == ["weight"] && isempty(grid.cell_data)
        measure = sum(Unfitted._region_measure(r) for r in regions if sort([p.level for p in r.parents]) == signature)
        @test sum(last(only(grid.point_data))) ≈ measure rtol = 1.0e-13
    end
    @test sum(entry -> length(entry.grid.points), cloud) == nquadpoints(model)
end
