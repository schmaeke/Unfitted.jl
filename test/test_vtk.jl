using StaticArrays

@testset "partitioned VTK bundle with wireframes" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=2)
    model = prepare(poisson(V; source=x -> 0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "case"), solution, model; subdivisions=:none, ascii=true,
                      append=false, compress=false,
                      point_data=(uh=(u, c, x, xi) -> u(c, xi),
                                  twice=(u, c, x, xi) -> 2 * u(c, xi)),
                      cell_data=(midpoint_x=(u, c, x, xi) -> x[1],),)

    @test joinpath(dir, "case.vtm") in files
    @test joinpath(dir, "case_data.vtu") in files
    @test joinpath(dir, "case_level_1_base_wire.vtp") in files
    @test joinpath(dir, "case_level_2_overlay_wire.vtp") in files
    @test all(isfile, files)

    solution_xml = read(joinpath(dir, "case_data.vtu"), String)
    @test occursin("uh", solution_xml)
    @test occursin("twice", solution_xml)
    @test occursin("midpoint_x", solution_xml)
    @test occursin("region_id", solution_xml)
    @test occursin("cover_count", solution_xml)

    wire_xml = read(joinpath(dir, "case_level_2_overlay_wire.vtp"), String)
    @test occursin("level_id", wire_xml)
    @test occursin("role_id", wire_xml)
    @test occursin("order_max", wire_xml)

    bundle_xml = read(joinpath(dir, "case.vtm"), String)
    @test occursin("case_data.vtu", bundle_xml)
    @test occursin("case_level_1_base_wire.vtp", bundle_xml)
    @test occursin("case_level_2_overlay_wire.vtp", bundle_xml)
end

@testset "coupled multi-domain VTK bundle" begin
    # A coupled model writes one block per field (named by field), each grouping
    # that field's `data` grid with its own wireframe(s) as sibling leaf datasets
    # — a homogeneous tree whose leaf names are disambiguated by ASCII index /
    # level id (not the possibly-Unicode field name) so ParaView's Extract Block
    # resolves each leaf uniquely — plus one quadrature block per subdomain.
    V1 = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    V2 = space(box((2.0, 0.0), (3.0, 1.0)); cells=(2, 2), order=1)
    u1 = field(:u1, V1)
    u2 = field(:u2, V2)
    model = prepare(Problem((u1, u2);
                            blocks=(stiffness_block(u1), stiffness_block(u2)),
                            loads=(source_load(u1; source=x -> 1.0),
                                   source_load(u2; source=x -> 1.0)),
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
    # per-field blocks named u1 / u2, each grouping a data + wireframe leaf
    @test occursin("name=\"u1\"", vtm)
    @test occursin("name=\"u2\"", vtm)
    # leaf names are disambiguated by ASCII index / level id (never by the
    # possibly-Unicode field name, which ParaView's data assembly would collapse)
    @test occursin("name=\"data_1\"", vtm) && occursin("name=\"data_2\"", vtm)
    @test occursin("name=\"level_1_base\"", vtm) && occursin("name=\"level_2_base\"", vtm)
    @test !occursin("mesh_wireframes", vtm)        # not a separate flat wireframe block
    @test !occursin("name=\"mesh\"", vtm)          # no sub-block: wireframes are direct leaves
    # only the two field blocks are `<Block>`s (homogeneous: leaves below them)
    @test count("<Block", vtm) == 2
    # one wireframe per distinct level (ids 1 and 2), no duplication
    wires = filter(f -> occursin("_wire.vtp", f), files)
    @test length(wires) == 2 == length(unique(wires))

    # quadrature: one block per subdomain (covering-level signatures "1" and "2")
    qfiles = write_quadrature_vtm(joinpath(dir, "cplq"), model)
    @test joinpath(dir, "cplq_quadrature_levels_1.vtp") in qfiles
    @test joinpath(dir, "cplq_quadrature_levels_2.vtp") in qfiles
end

@testset "two fields on one space write each wireframe once" begin
    # The wireframe dedup: fields sharing a space must not emit duplicate level
    # blocks. A two-field/one-space model has a single base level → one wireframe.
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    a = field(:a, V)
    b = field(:b, V)
    model = prepare(Problem((a, b);
                            blocks=(stiffness_block(a), mass_block(b)),
                            loads=(source_load(a; source=x -> 1.0),),
                            dirichlet=[dirichlet(0.0; on=boundary(:all), field=:a)]))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))
    dir = mktempdir()
    files = write_vtk(joinpath(dir, "shared"), solution, model; subdivisions=:none, ascii=true,
                      append=false, compress=false)
    wires = filter(f -> occursin("_wire.vtp", f), files)
    @test length(wires) == 1
end

@testset "VTK exports vector point data" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=2)
    u = field(:u, V; components=2)
    model = prepare(poisson(u; source=x -> SVector(0.0, 0.0)))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "vector_case"), solution, model; subdivisions=:none, ascii=true,
                      append=false, compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),),)

    @test joinpath(dir, "vector_case_data.vtu") in files
    solution_xml = read(joinpath(dir, "vector_case_data.vtu"), String)
    @test occursin("uh", solution_xml)
end

@testset "wireframe filters inactive overlay cells" begin
    # A 3x3 overlay with only one cell active should produce exactly that
    # cell's edges in the overlay wireframe (1 cell × 4 quad edges = 4 lines),
    # not the full grid (9 cells × 4 = 36).
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    active = falses(3, 3)
    active[2, 2] = true
    V = overlay(V, box((0.1, 0.1), (0.9, 0.9)); cells=(3, 3), order=1, active=active)
    model = prepare(poisson(V; source=x -> 0.0))
    coefficients = ones(Unfitted.active_unknowns(model.dofs))
    soln = Solution(coefficients, model.version, Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "masked"), soln, model; subdivisions=:none, ascii=true,
                      append=false, compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    @test joinpath(dir, "masked_level_2_overlay_wire.vtp") in files

    overlay_xml = read(joinpath(dir, "masked_level_2_overlay_wire.vtp"), String)
    @test occursin("NumberOfLines=\"4\"", overlay_xml)

    # The base mesh has no mask, so its full 1x1 wireframe (4 edges) survives.
    base_xml = read(joinpath(dir, "masked_level_1_base_wire.vtp"), String)
    @test occursin("NumberOfLines=\"4\"", base_xml)
end

@testset "VTK export rejects dimensions above 3" begin
    omega = box((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    V = space(omega; cells=(1, 1, 1, 1), order=1)
    model = prepare(poisson(V; source=x -> 0.0))
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
    model = prepare(poisson(V; source=x -> 0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    files = write_vtk(joinpath(dir, "disk"), solution, model; subdivisions=:none, ascii=true,
                      append=false, compress=false, point_data=(uh=(u, c, x, xi) -> u(c, xi),))

    solution_xml = read(joinpath(dir, "disk_data.vtu"), String)
    @test occursin("level_set", solution_xml)
end

@testset "VTK omits level_set when no PhysicalDomain is attached" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(1, 1), order=1)
    model = prepare(poisson(V; source=x -> 0.0))
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
    model = prepare(poisson(V; source=x -> 0.0))
    solution = Solution(ones(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    dir = mktempdir()
    write_vtk(joinpath(dir, "override"), solution, model; subdivisions=:none, ascii=true,
              append=false, compress=false,
              point_data=(uh=(u, c, x, xi) -> u(c, xi), level_set=(u, c, x, xi) -> 42.0),)

    solution_xml = read(joinpath(dir, "override_data.vtu"), String)
    @test occursin("level_set", solution_xml)
    @test occursin("42", solution_xml)
    # The auto-emit would have written the φ value at e.g. (0, 0) ≈
    # 0.7071 − 0.4 = 0.307. The override means no such value appears.
    @test !occursin("0.30710678", solution_xml)
end

@testset "quadrature VTM splits regions by parent levels" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    V = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(1, 1), order=1)
    model = prepare(poisson(V; source=x -> 0.0))

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
