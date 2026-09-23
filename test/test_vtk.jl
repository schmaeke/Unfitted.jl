using StaticArrays

# Parse one `<DataArray>` back out of an ASCII `.vtu`/`.vtp` by name, so the
# tests can assert the numbers ParaView will show. Requires `ascii=true,
# append=false` on the write; the appended-binary form carries no parsable text.
function _vtu_values(xml, name)
    split(match(Regex("Name=\"$name\"[^>]*>([^<]*)</DataArray>"), xml).captures[1])
end
_vtu_ints(xml, name) = parse.(Int, _vtu_values(xml, name))
_vtu_floats(xml, name) = parse.(Float64, _vtu_values(xml, name))

@testset "partitioned VTK bundle with level meshes" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=2)
    # u ≡ 1 solves −Δu = 0 with u = 1 on ∂Ω and lies in the base space, so the
    # overlay's p = 2 modes come back at zero and every exported value is 1.
    model = prepare(poisson(V; source=0.0, dirichlet=[dirichlet(1.0; on=boundary(:all))]))
    solution = solve!(model)

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "case"), solution, model; subdivisions=:none, ascii=true,
                      append=false, compress=false,
                      point_data=(uh=(u, c, x, xi) -> u(c, xi),
                                  twice=(u, c, x, xi) -> 2 * u(c, xi)),
                      cell_data=(midpoint_x=(u, c, x, xi) -> x[1],))

    @test joinpath(dir, "case.vtm") in files
    @test joinpath(dir, "case_data.vtu") in files
    @test joinpath(dir, "case_level_1_base_mesh.vtu") in files
    @test joinpath(dir, "case_level_2_overlay_mesh.vtu") in files
    @test all(isfile, files)

    solution_xml = read(joinpath(dir, "case_data.vtu"), String)
    # 5 admissible regions (the overlay box plus the four base remainders), one
    # unsubdivided VTK cell each, 4 corners per cell.
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
    # A coupled model writes one block per field (named by field), each grouping
    # that field's `data` grid with its own mesh(es) as sibling leaf datasets
    # — a homogeneous tree whose leaf names are disambiguated by ASCII index /
    # level id (not the possibly-Unicode field name) so ParaView's Extract Block
    # resolves each leaf uniquely — plus one quadrature block per subdomain.
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
    files = write_vtk(joinpath(dir, "cpl"), solution, model; subdivisions=:none, ascii=true,
                      append=false, compress=false)

    # one data `.vtu` per field, ASCII-indexed so Extract Block can isolate it
    @test joinpath(dir, "cpl_data_1.vtu") in files
    @test joinpath(dir, "cpl_data_2.vtu") in files
    vtm = read(joinpath(dir, "cpl.vtm"), String)
    # per-field blocks named u1 / u2, each grouping a data + mesh leaf
    @test occursin("name=\"u1\"", vtm)
    @test occursin("name=\"u2\"", vtm)
    # leaf names are disambiguated by ASCII index / level id (never by the
    # possibly-Unicode field name, which ParaView's data assembly would collapse)
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
    # The mesh dedup: fields sharing a space must not emit duplicate level
    # blocks. A two-field/one-space model has a single base level → one mesh.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    a = field(:a, V)
    b = field(:b, V)
    model = prepare(Problem((a, b); blocks=(stiffness_block(a), mass_block(b)),
                            loads=(source_load(a; source=1.0),),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=:a)]))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))
    dir = mktempdir()
    files = write_vtk(joinpath(dir, "shared"), solution, model; subdivisions=:none, ascii=true,
                      append=false, compress=false)
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
    files = write_vtk(joinpath(dir, "vector_case"), solution, model; subdivisions=:none, ascii=true,
                      append=false, compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    @test joinpath(dir, "vector_case_data.vtu") in files
    solution_xml = read(joinpath(dir, "vector_case_data.vtu"), String)
    @test occursin("uh", solution_xml)

    # A VTK data array is a vector only at three components; a 2-component array
    # is stored but cannot be assigned as VECTORS, so ParaView's Glyph / Warp By
    # Vector cannot read it. The 2-D field is padded with a trailing zero, the
    # same padding `_vtk_point` applies to the geometry it is attached to.
    uh_header = match(r"<DataArray[^>]*Name=\"uh\"[^>]*>", solution_xml)
    @test uh_header !== nothing
    @test occursin("NumberOfComponents=\"3\"", uh_header.match)
    @test occursin("NumberOfComponents=\"3\"",
                   match(r"<DataArray[^>]*Name=\"Points\"[^>]*>", solution_xml).match)
end

@testset "level mesh exports all cells with an active flag" begin
    # A 3x3 overlay with only one active cell now exports ALL nine cells as solid
    # cells, distinguished by the `active` cell-data (1 for the active cell, 0 for
    # the eight masked ones) — no longer a filtered wireframe.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    active = falses(3, 3)
    active[2, 2] = true
    V = overlay(V, box((0.1, 0.1), (0.9, 0.9)); cells=(3, 3), order=1, active=active)
    model = prepare(poisson(V; source=0.0))
    coefficients = ones(Unfitted.active_unknowns(model.dofs))
    soln = Solution(coefficients, model.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "masked"), soln, model; subdivisions=:none, ascii=true,
                      append=false, compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

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
    # The solid-cell per-level mesh carries `covered` / `reduced_dofs` / `active`
    # cell data; assert the actual values, not just that the arrays are emitted. A
    # 4×4 p=2 base with an aligned p=3 overlay over the middle 2×2 block covers four
    # base cells and sheds high-order (plus one deduped vertex) there.
    V = overlay(space(box((0.0, 0.0), (1.0, 1.0)); cells=(4, 4), order=2),
                box((0.25, 0.25), (0.75, 0.75)); cells=(4, 4), order=3)
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
    # A circular cut of the unit square: cells crossing r = 0.4 are
    # cut, cells fully inside are full, cells fully outside (with α = 1
    # to keep them in the dof layout) are fictitious. The exported
    # `level_set` array must equal `φ(x)` at every VTK vertex so a
    # ParaView contour at level 0 reproduces ∂Ω.
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    omega = box((0.0, 0.0), (1.0, 1.0))
    physical = physical_domain(phi_disk; lipschitz=1.0, alpha=1.0, subcell_length_scale=0.125,
                               max_depth=3)
    V = space(omega; cells=(2, 2), order=1, physical=physical)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "disk"), solution, model; subdivisions=:none, ascii=true,
                      append=false, compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    solution_xml = read(joinpath(dir, "disk_data.vtu"), String)
    points = reshape(_vtu_floats(solution_xml, "Points"), 3, :)
    @test _vtu_floats(solution_xml, "level_set") ≈
          [phi_disk(view(points, :, i)) for i in axes(points, 2)] atol = 1.0e-14
end

@testset "VTK omits level_set when no PhysicalDomain is attached" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    write_vtk(joinpath(dir, "plain"), solution, model; subdivisions=:none, ascii=true, append=false,
              compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    solution_xml = read(joinpath(dir, "plain_data.vtu"), String)
    @test !occursin("level_set", solution_xml)
end

@testset "user-supplied level_set callback overrides the auto-emit" begin
    # A user `level_set` callback in `point_data` should win: the
    # built-in is suppressed, the user's array (whose constant value
    # `42.0` cannot match φ) is written instead. Detecting the override
    # via the constant payload keeps the test independent of how
    # WriteVTK serialises numeric arrays.
    phi_disk(x) = sqrt((x[1] - 0.5)^2 + (x[2] - 0.5)^2) - 0.4
    omega = box((0.0, 0.0), (1.0, 1.0))
    physical = physical_domain(phi_disk; lipschitz=1.0, alpha=1.0, subcell_length_scale=0.25,
                               max_depth=2)
    V = space(omega; cells=(1, 1), order=1, physical=physical)
    model = prepare(poisson(V; source=0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    write_vtk(joinpath(dir, "override"), solution, model; subdivisions=:none, ascii=true,
              append=false, compress=false,
              point_data=(uh=(u, c, x, xi) -> u(c, xi), level_set=(u, c, x, xi) -> 42.0))

    solution_xml = read(joinpath(dir, "override_data.vtu"), String)
    @test occursin("level_set", solution_xml)
    @test occursin("42", solution_xml)
    # The auto-emit would have written the φ value at e.g. (0, 0) ≈
    # 0.7071 − 0.4 = 0.307. The override means no such value appears.
    @test !occursin("0.30710678", solution_xml)
end

@testset "VTK time series indexes one full bundle per frame" begin
    # A series is `write_vtk` in a loop plus a `.pvd` index over the frames. The
    # overlay moves between the two frames here, which is the case the feature
    # exists for: each frame carries its own level meshes, so the animation shows
    # the mesh following the feature and not just the field on a frozen mesh.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = overlay(space(omega; cells=(2, 2), order=1), box((0.1, 0.1), (0.5, 0.5)); cells=(1, 1),
                order=2)
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))

    dir = mktempdir()
    series = vtk_series(joinpath(dir, "run"))
    first_frame = write_vtk(series, 0.0, solve!(model), model; subdivisions=:none, ascii=true,
                            append=false, compress=false)
    @test first_frame == joinpath(dir, "run_frames", "frame_0001.vtm")
    @test isfile(joinpath(dir, "run_frames", "frame_0001_level_2_overlay_mesh.vtu"))

    # Byte-identity, the contract of the body split: a frame is the ordinary
    # bundle. Written from the same `(solution, model)` before the overlay moves.
    solo = solve!(model)
    write_vtk(joinpath(dir, "solo"), solo, model; subdivisions=:none, ascii=true, append=false,
              compress=false)
    @test read(joinpath(dir, "run_frames", "frame_0001_data.vtu")) ==
          read(joinpath(dir, "solo_data.vtu"))

    move!(model; level=2, to=box((0.5, 0.5), (0.9, 0.9)))
    write_vtk(series, 0.25, solve!(model), model; subdivisions=:none, ascii=true, append=false,
              compress=false)
    @test read(joinpath(dir, "run_frames", "frame_0002_level_2_overlay_mesh.vtu")) !=
          read(joinpath(dir, "run_frames", "frame_0001_level_2_overlay_mesh.vtu"))

    collection = close(series)
    @test collection == joinpath(dir, "run.pvd")
    sets = collect(eachmatch(r"<DataSet timestep=\"([^\"]*)\"[^>]*file=\"([^\"]*)\"",
                             read(collection, String)))
    @test length(sets) == 2
    @test parse.(Float64, [s.captures[1] for s in sets]) == [0.0, 0.25]
    @test [s.captures[2] for s in sets] ==
          [joinpath("run_frames", "frame_0001.vtm"), joinpath("run_frames", "frame_0002.vtm")]
    @test close(series) == collection           # closing twice is harmless
end

@testset "VTK series writes its frames where it is told" begin
    # The frames directory is a keyword, the collection names each frame
    # relative to itself, and a `.pvd` given on the path is stripped rather than
    # doubled. The time is recorded exactly as handed over — no frame grid.
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
