using LinearAlgebra

@testset "public API scaffold" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(4, 4), order=2)
    V2 = overlay(V, box((0.25, 0.25), (0.75, 0.75)); cells=(2, 2), order=3)
    V3 = moved_space(V2; level=2, to=box((0.2, 0.2), (0.6, 0.6)))
    model = prepare(poisson(V2; source=1.0))

    @test Unfitted.level_count(V) == 1
    @test Unfitted.level_count(V2) == 2
    @test V2.levels[2].role == :overlay
    @test nominal_order(V2.levels[2]) == (3, 3)
    @test V3.levels[2].mesh.domain == box((0.2, 0.2), (0.6, 0.6))
    @test diagnostics(model).dimension == 2
    @test diagnostics(model).integration_regions == 16

    old_plan = Unfitted.integration_plan(model)
    before = model.version
    move!(model; level=2, to=box((0.2, 0.2), (0.6, 0.6)))
    # `version` is a pin on the dof numbering, not a count: `prepare` seeds it from the discretisation, and a mutation
    # moves it on from there.
    @test model.version == before + 1
    @test Unfitted.integration_plan(model) !== old_plan
    @test model.matrix === nothing
    @test diagnostics(model).integration_regions == 35   # merged (was 49 unmerged)
end

@testset "custom solver hook and compact display" begin
    omega = box((0.0,), (1.0,))
    V = space(omega; cells=2, order=1)
    model = prepare(poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=boundary(:all))]))
    called = Ref(false)
    solution = solve!(model; method=:external_direct, linear_solver=(A, b) -> begin
                          called[] = true
                          A \ b
                      end)

    @test called[]
    @test solution.coefficients[1] ≈ 0.125
    @test solution.diagnostics.method == :external_direct
    @test solution.diagnostics.residual_norm < 1.0e-12
    @test diagnostics(model).solver == :external_direct
    @test_throws ArgumentError solve!(model; method=:iterative_without_hook)
    @test_throws DimensionMismatch solve!(model; linear_solver=(A, b) -> zeros(0))

    shown = (sprint(show, V.levels[1]), sprint(show, V), sprint(show, model), sprint(show, diagnostics(model)),
             sprint(show, solution), sprint(show, solution.diagnostics))

    @test occursin("Level", shown[1])
    @test occursin("Space", shown[2])
    @test occursin("Model", shown[3])
    @test occursin("AssemblyDiagnostics", shown[4])
    @test occursin("Solution", shown[5])
    @test occursin("SolverDiagnostics", shown[6])
end

@testset "transfer requires source and target models" begin
    omega = box((0.0,), (1.0,))
    model = prepare(poisson(space(omega; cells=1, order=1); source=0.0))
    solution = Solution(zeros(Unfitted.active_unknowns(model.dofs)), model.version,
                        Unfitted.SolverDiagnostics(:manual, 0.0, true))

    @test_throws ArgumentError transfer(solution, model)
end

# A form assembled as symmetric is built in the lower triangle and mirrored as `tril(A) + tril(A)ᵀ − diag(A)`, so an
# asymmetric form declared symmetric loses its upper triangle with nothing downstream to notice: the mirrored operator
# of an advection problem is still solvable and still reports a tiny residual. The default therefore has to be "assemble
# in full".
@testset "WeakForm defaults to asymmetric assembly" begin
    V = space(box((0.0,), (1.0,)); cells=2, order=1)
    advection(q, trial) = trial.gradient[1]      # a(u, v) = ∫ ∂ₓu · v — not symmetric
    nosource(q) = 0.0

    form = WeakForm(bilinear=advection, linear=nosource)
    @test !form.symmetric

    model = prepare(Problem(V, form))
    @test !model.problem.symmetric

    A = assemble!(model).matrix
    mirrored = assemble_matrix(model, model.problem.blocks; symmetric=true)
    @test A != A'                 # the upper triangle survives the default …
    @test mirrored == mirrored'   # … and is replaced by the mirror when declared
    @test A != mirrored
end

# An out-of-range component index matches no dof and no assembly row, so a Dirichlet condition carrying one leaves its
# boundary free and a Neumann load carrying one assembles to zero — silently in both cases. Every entry point that can
# see the field's component count rejects it instead.
@testset "component indices are range-checked" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    u = field(:u, V; components=2)
    everywhere = boundary(:all)

    @test dirichlet(0.0; on=everywhere, field=u, component=2).component == 2
    @test_throws ArgumentError dirichlet(0.0; on=everywhere, field=u, component=3)
    @test_throws ArgumentError dirichlet(0.0; on=everywhere, field=u, component=0)
    @test_throws ArgumentError neumann(u, 1.0; on=everywhere, component=3)

    # Conditions naming their field by `Symbol`, and unscoped conditions on a single-field problem, are resolved at
    # `Problem` construction instead.
    @test_throws ArgumentError poisson(u; source=0.0, dirichlet=[dirichlet(0.0; on=everywhere, field=:u, component=3)])
    @test_throws ArgumentError poisson(u; source=0.0, dirichlet=[dirichlet(0.0; on=everywhere, component=3)])

    # An unscoped condition on a multi-field problem names no field to check its component against, and `Problem`
    # refuses it as such.
    @test_throws "multi-field Dirichlet conditions must specify a field" Problem((u, field(:w, V));
                                                                                 blocks=(stiffness_block(u),),
                                                                                 dirichlet=[dirichlet(0.0;
                                                                                                      on=everywhere,
                                                                                                      component=3)])
end

# A Dirichlet condition belongs to one field of its problem: the only one when it names none, the one it names
# otherwise. `Problem` resolves that owner when it is built. An unscoped condition on a problem of several fields, and a
# name no field carries, used to construct and fail only at `prepare`, after the problem had been passed around and
# perhaps rehomed; the messages are the ones `prepare` gave.
@testset "Problem refuses a Dirichlet condition it cannot attach to a field" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    u, w = field(:u, V), field(:w, V; components=2)
    everywhere = boundary(:all)
    unscoped = [dirichlet(0.0; on=everywhere)]
    @test_throws "multi-field Dirichlet conditions must specify a field" Problem((u, w); dirichlet=unscoped)
    @test_throws "unknown field v" Problem((u, w); dirichlet=[dirichlet(0.0; on=everywhere, field=:v)])
    @test_throws "unknown field v" Problem(u, mass_form(); dirichlet=[dirichlet(0.0; on=everywhere, field=:v)])
    @test_throws "unknown field v" poisson(V; source=1.0, dirichlet=[dirichlet(0.0; on=everywhere, field=:v)])

    # The conditions every problem shape does take still build: unscoped on one field, named on several.
    @test length(Problem((u,); dirichlet=unscoped).dirichlet) == 1
    named = [dirichlet(0.0; on=everywhere, field=:u), dirichlet((1.0, 2.0); on=everywhere, field=w)]
    @test length(Problem((u, w); blocks=(mass_block(u), mass_block(w)), dirichlet=named).dirichlet) == 2
end

# Bytes one warm call of the diffusion flux allocates, measured behind a function barrier so the figure is the call's
# own; the first call compiles.
function _flux_bytes(a, ugrad)
    Unfitted._diffusion_flux(a, ugrad)
    return @allocated Unfitted._diffusion_flux(a, ugrad)
end

# `diffusion` is documented as a scalar, a `D × D` matrix, or a callback returning either, and `_diffusion_flux` also
# accepts a `UniformScaling`. `_diffusion_flux` copies every `AbstractMatrix` into an `SMatrix` before contracting it
# with the gradient, so the dense and static spellings of one tensor take one code path; the isotropic spellings must
# reproduce the tensor that represents them.
@testset "diffusion spellings agree" begin
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=2)
    u = field(:u, V)
    model = prepare(Problem((u,); blocks=(stiffness_block(u),)))
    K(a) = Matrix(assemble_matrix(model, stiffness_block(u; diffusion=a)))

    dense = [2.0 0.75; 0.75 3.0]
    static = Unfitted.SMatrix{2,2}(dense)

    # One code path, but not one compiled kernel: each spelling is its own closure type, so the assembler is compiled
    # once for each, and the optimiser decides separately in each kernel which of the static product's `muladd`s become
    # FMAs. The two therefore agree to rounding, not bit for bit. On x86 under Julia 1.13 at the default optimisation
    # level, LLVM vectorises the dense kernel's flux and leaves it unfused while it fuses the static one, and 60 of the
    # 625 entries here differ, by at most 5 ulp (relative norm 8e-17).
    @test K(dense) ≈ K(static) rtol = 1.0e-14

    # What no tolerance can see is the dense spelling drifting off the static product onto the generic one, which
    # allocates its result once per quadrature point per trial function and rounds within a few ulp of the static one.
    # The flux of either spelling allocates nothing.
    ugrad = Unfitted.SVector(0.3, -1.7)
    @test _flux_bytes(dense, ugrad) == 0
    @test _flux_bytes(static, ugrad) == 0

    isotropic = 2.5
    @test K(isotropic) ≈ K(isotropic * I)
    @test K(isotropic) ≈ K([isotropic 0.0; 0.0 isotropic])
    @test K(isotropic) ≈ K(Unfitted.SMatrix{2,2}(isotropic, 0.0, 0.0, isotropic))

    # A tensor of the wrong shape is a loud error, never a silent contraction against whatever subblock happens to fit.
    @test_throws DimensionMismatch K([1.0 0.0 0.0; 0.0 1.0 0.0])
    @test_throws DimensionMismatch K([1.0 0.0; 0.0 1.0; 0.0 0.0])
end

# A constant per-component coefficient carries one entry per component of its field, exactly as a constant Dirichlet
# datum does. A longer one used to be truncated silently and a shorter one raised a `BoundsError` from inside the
# assembly loop; a constant diffusion that is not a scalar, a `UniformScaling` or a matrix raised a `MethodError` at
# every quadrature point. Each builder that knows the field now refuses the wrong shape where it is spelled.
@testset "constant coefficients are checked against their field" begin
    SVector = Unfitted.SVector
    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    u = field(:u, V)
    w = field(:w, V; components=2)
    everywhere = boundary(:all)

    for value in ((1.0,), SVector(1.0, 2.0, 99.0), [1.0, 2.0, 3.0])
        needs = "of field w, 2, but this one has length $(length(value))"
        @test_throws needs mass_block(w; coefficient=value)
        @test_throws needs mass(w; coefficient=value)
        @test_throws needs source_load(w; source=value)
        @test_throws needs poisson(w; source=value)
        @test_throws needs neumann(w, value; on=everywhere)
        @test_throws needs neumann(w, value; on=everywhere, component=1)
    end
    @test_throws "of field u, 1, but this one has length 2" mass_block(u; coefficient=SVector(1.0, 2.0))
    @test_throws "of field u, 1, but this one has length 2" poisson(V; source=(1.0, 2.0))

    # The right length, a scalar and a callback build as before; a callback is not evaluated ahead of time.
    @test mass_block(w; coefficient=(1.0, 2.0)) isa BlockForm
    @test mass_block(w; coefficient=2.0) isa BlockForm
    @test source_load(w; source=x -> SVector(1.0, 2.0, 3.0)) isa LoadForm
    @test poisson(V; source=(1.0,)) isa Problem

    for diffusion in ([1.0, 2.0], (1.0, 2.0), SVector(1.0, 2.0), ones(2, 2, 2))
        @test_throws "a constant diffusion is a scalar, a UniformScaling or a square matrix" stiffness_form(; diffusion)
        @test_throws ArgumentError stiffness(V; diffusion)
        @test_throws ArgumentError poisson(V; source=1.0, diffusion)
    end
    for diffusion in (2.0, 3, 2.0 * I, [2.0 0.0; 0.0 3.0], x -> 2.0)
        @test stiffness_form(; diffusion) isa WeakForm
    end
end

# Every exported name should answer `?name` in a bare `using Unfitted` session. Reading the *source* is what let two of
# them regress: a docstring separated from its definition by an intervening comment block parses as a standalone string
# and never reaches `Core.@doc`, so the prose is right there in the file and `?BlockForm` still prints nothing. Only
# `Base.Docs.meta` sees the difference, so that is what this asserts.
@testset "exported names carry reachable docstrings" begin
    meta = Base.Docs.meta(Unfitted)
    undocumented = [n for n in names(Unfitted) if n !== :Unfitted && !haskey(meta, Base.Docs.Binding(Unfitted, n))]

    # The two names that regressed, pinned by name so a re-detachment fails here.
    @test :BlockForm ∉ undocumented
    @test :LoadForm ∉ undocumented

    # The `Problem` constructors' docstring, which a comment block between it and its method detached as well. The
    # struct's own docstring keeps `:Problem` out of `undocumented` either way, so the constructors' text is looked up.
    texts = [join(doc.text) for doc in values(meta[Base.Docs.Binding(Unfitted, :Problem)].docs)]
    @test any(t -> occursin("Three convenience constructors", t), texts)

    # The remaining gap is known and structural, not a detachment: `mass_form` / `mass_block` each carry one docstring
    # covering their two siblings as well, and a docstring attaches to a single definition, so the siblings have no
    # entry of their own.
    #
    # Subset, not equality: documenting one of these is an improvement and must not fail the suite, while a *new*
    # undocumented export does. The names an extension implements are not exempt: the core declares and documents each.
    grouped = [:source_form, :stiffness_form, :source_load, :stiffness_block]
    @test issubset(undocumented, grouped)

    # The VTK writers are implemented by an extension and documented here, the two series operations included: their
    # docstrings sit on abstract signatures that no method of the core implements, so pin that they reached the docs.
    @test haskey(meta[Base.Docs.Binding(Unfitted, :write_vtk)].docs, Tuple{VTKSeries,Any,Any,Any})
    @test haskey(meta[Base.Docs.Binding(Base, :close)].docs, Tuple{VTKSeries})
    # So are the tensor readers' `Val` and `Field` forms: one docstring per signature the Tensors extension implements.
    ndocs(name) = length(meta[Base.Docs.Binding(Unfitted, name)].docs)
    @test (ndocs(:symmetric_gradient), ndocs(:value_vec), ndocs(:gradient_tensor)) == (3, 2, 2)
end

# The VTK writers, the tensor-notation readers, the B-spline factory and the mesh level sets are package extensions that
# `using WriteVTK`, `using Tensors`, `using BasicBSpline` and `using MeshIO` load. Without its extension each entry
# point must say so, naming the package, before it touches anything (the file system included), and `using Unfitted`
# alone must load none of those packages, which is what keeps them out of the package's load time. Only a fresh process
# can show either: this one may have loaded them already (the extensions' test files do), and a loaded package cannot be
# unloaded. The child loads Unfitted alone and calls each entry point with placeholder arguments, which a stub never
# reads, so no model is built and the test costs one package load.
@testset "the extension entry points ask for their package until it is loaded" begin
    dir = mktempdir()
    script = """
        using Unfitted
        dir = only(ARGS)
        calls = (write_vtk => (joinpath(dir, "w", "bundle"), nothing, nothing),
                 vtk_series => (joinpath(dir, "s", "run"),),
                 write_quadrature_vtm => (joinpath(dir, "q", "cloud"), nothing),
                 symmetric_gradient => (nothing,), value_vec => (nothing, :u, Val(2)),
                 gradient_tensor => (nothing, :u, Val(2)), bspline => (),
                 mesh_levelset => (nothing,), stl_levelset => (joinpath(dir, "m", "part.stl"),))
        for (f, args) in calls
            message = try
                f(args...)
                "returned"
            catch e
                sprint(showerror, e)
            end
            println(nameof(f), " => ", replace(message, '\\n' => ' '))
        end
        packages = ("WriteVTK", "Tensors", "BasicBSpline", "MeshIO", "FileIO", "GeometryBasics")
        println("packages loaded => ", any(p -> p.name in packages, keys(Base.loaded_modules)))
        """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script $dir`
    reported = Dict{String,String}()
    for line in readlines(cmd)
        occursin(" => ", line) || continue
        name, message = split(line, " => "; limit=2)
        reported[name] = message
    end
    entry_points = ("write_vtk" => "WriteVTK", "vtk_series" => "WriteVTK", "write_quadrature_vtm" => "WriteVTK",
                    "symmetric_gradient" => "Tensors", "value_vec" => "Tensors", "gradient_tensor" => "Tensors",
                    "bspline" => "BasicBSpline", "mesh_levelset" => "MeshIO", "stl_levelset" => "MeshIO")
    for (name, package) in entry_points
        @test occursin("using $package", get(reported, name, ""))
    end
    @test get(reported, "packages loaded", "") == "false"
    @test isempty(readdir(dir))
end
