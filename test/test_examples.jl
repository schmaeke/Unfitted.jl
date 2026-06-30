# Smoke-test guard for the `examples/` scripts.
#
# The example scripts under `examples/<name>/<name>.jl` are the package's
# paper-regression reproductions — they are the executable form of the
# scientific contract (UMLHP §5.* benchmarks, the FCM moment-fit
# reproductions, the phase-field and traveling-heat-source demonstrations).
# The rest of the suite unit-tests the building blocks, but until now nothing
# ran the example scripts end to end, so a public-API change could silently
# break a paper reproduction without any test going red.
#
# This file closes that gap. For each example it spawns the script as an
# **isolated Julia subprocess** at a deliberately coarse size and asserts that
# the process exits 0, and — where the example prints a stable, clearly
# labelled metric (`relative L2 error`, `residual norm` from the shared
# `examples/reporting.jl` formatter) — that the metric is finite and within a
# loose sanity bound. Coarse ≠ accurate: the point is "ran end to end and
# produced sane numbers", not "converged to the published digits".
#
# Why subprocesses rather than `include`:
#   - Each example is a self-contained script with top-level `const`s, a
#     `main()`, and its own `using` lines; `include`-ing several of them into
#     one session would collide on those globals and leak ENV/output state
#     between examples.
#   - A hard failure in one example (an error, or even a segfault from a native
#     dependency) then fails only that example's `@test`, never the whole
#     runner.
#   - The coarse-sizing ENV knobs each example reads (e.g. `THS_T_MAX`,
#     `SHP_PHASE_*`) are passed per subprocess via `setenv`, so they cannot
#     bleed into the rest of the suite.
#
# Project selection. Every example is launched on the *active* project
# (`Base.active_project()`): under `Pkg.test` that is the instantiated test
# sandbox, which carries `Unfitted`, `StaticArrays`, `LinearAlgebra` *and*
# `Tensors` (a test/weak dependency); in a bare `julia --project=. ...`
# session it is the package root, which lacks `Tensors`. The three
# `Tensors`-using examples (phase-field, both FCM plates) are therefore gated
# on `Tensors` actually being loadable in the active project: they run under
# the full `Pkg.test` suite and are skipped (not failed) in a bare-root
# isolation run.
#
# Slow examples. The two FCM examples are correct and bounded (~15-20 s each)
# but are not "fast"; running them by default would push the added suite time
# past its budget, and the FCM integration kernel is already covered by
# `test_fcm.jl`. They are gated behind `UNFITTED_TEST_SLOW_EXAMPLES=1` so a
# nightly / pre-release run can exercise the full set while the default run
# stays lean.

@testset "examples" begin
    # Directory of the example scripts and the project the subprocesses run on.
    # `Base.active_project()` returns a path to a `Project.toml`; `--project`
    # wants the directory containing it.
    examples_dir = abspath(joinpath(@__DIR__, "..", "examples"))
    active_project = something(Base.active_project(), abspath(joinpath(@__DIR__, "..")))
    project_dir = dirname(active_project)

    # `Tensors` is a weak/test dependency: it is loadable under `Pkg.test` but
    # not from a bare `--project=.` session. `identify_package` reports whether
    # it is a direct dependency of the active project without loading it.
    tensors_available = Base.identify_package("Tensors") !== nothing

    # The two FCM examples only run when explicitly opted in.
    slow_enabled = lowercase(get(ENV, "UNFITTED_TEST_SLOW_EXAMPLES", "false")) in
                   ("1", "true", "yes")

    # Run one example script as an isolated subprocess on `project_dir`, with
    # `env` layered over the inherited environment to set coarse-size knobs.
    # `-O0` (overriding the parent's optimisation level) is deliberate: these
    # coarse runs are dominated by JIT compilation, not by the tiny numerics, so
    # unoptimised codegen lowers the wall time with no meaningful runtime cost.
    # stdout and stderr are merged into one pipe so the captured `output` both
    # carries the report lines we parse and gives a full diagnostic dump when
    # the process fails. Returns `(succeeded, combined_output)`.
    function run_example(script, env)
        cmd = setenv(`$(Base.julia_cmd()) -O0 --project=$project_dir --startup-file=no $script`,
                     merge(ENV, env))
        pipe = Pipe()
        process = run(pipeline(cmd; stdout=pipe, stderr=pipe); wait=false)
        close(pipe.in)
        output = read(pipe, String)
        wait(process)
        return success(process), output
    end

    # Every numeric value printed as "<label>: <number>" by the shared report
    # formatter, in print order. A multi-run example (the 1D bar prints a base
    # and an overlay error) yields one entry per occurrence.
    function metric_values(output, label)
        pattern = Regex(label * raw"\s*:\s*([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
        return [parse(Float64, m.captures[1]) for m in eachmatch(pattern, output)]
    end

    # A relative L² error is sane when it is finite and below one (a coarse run
    # may be far from the published digits, but a value ≥ 1 or non-finite means
    # the solve produced garbage).
    sane_error(e) = isfinite(e) && 0 ≤ e < 1

    # Metric checks, run only after a clean exit. Each asserts the example's
    # headline metric is present and sane; `check_success_only` is for the
    # examples whose output is a CSV sweep / transient log with no single
    # headline number (we still guard their exit code).
    function check_single_l2(output)
        errors = metric_values(output, "relative L2 error")
        @test length(errors) == 1
        @test sane_error(errors[1])
    end

    function check_bar(output)
        errors = metric_values(output, "relative L2 error")
        @test length(errors) == 2
        @test all(sane_error, errors)
        # The overlay around the unresolved interface must beat the base-only
        # run — that improvement is the entire point of the example.
        @test errors[2] < errors[1]
    end

    function check_phase(output)
        residuals = metric_values(output, "residual norm")
        @test !isempty(residuals)
        @test isfinite(last(residuals)) && 0 ≤ last(residuals) < 1e-3
    end

    check_success_only(output) = nothing

    # Coarse configuration for each example. `env` holds the size knobs;
    # `tensors` marks the `Tensors`-using examples; `slow` marks the opt-in
    # FCM examples; `check` runs the metric assertions on a clean exit.
    cases = (
             # Smooth manufactured Laplace, published 12×12 p=4 config (already
             # sub-second numerics); prints one relative L² error.
             (name="laplace_unit_square_smooth", env=Dict{String,String}(), tensors=false,
              slow=false, check=check_single_l2),
             # 1D bar, unresolved material interface; tiny base + overlay solves,
             # prints a base and an overlay relative L² error.
             (name="bar_1d_unresolved_interface", env=Dict{String,String}(), tensors=false,
              slow=false, check=check_bar),
             # Small-overlap conditioning sweep; 32 tiny solves, CSV output with no
             # single headline metric — guarded by exit code only.
             (name="conditioning_small_overlap", env=Dict{String,String}(), tensors=false,
              slow=false, check=check_success_only),
             # Corner-singularity nested-overlay solve at its published size (small
             # system; cost is compilation); prints one relative L² error.
             (name="singular_square_2d", env=Dict{String,String}(), tensors=false, slow=false,
              check=check_single_l2),
             # Traveling heat source: shortened transient (`THS_T_MAX=0.05` runs one
             # mesh-update + L²-transfer interval) and VTK export disabled; no single
             # headline metric — guarded by exit code only.
             (name="traveling_heat_source_2d",
              env=Dict("THS_T_MAX" => "0.05", "THS_WRITE_OUTPUT" => "false"), tensors=false,
              slow=false, check=check_success_only),
             # Phase-field SENT: coarse base (6×6, p=2), a few load steps to a tiny
             # final displacement, VTK disabled; prints the converged Newton
             # residual norm.
             (name="phase_field_single_edge_notch_2d",
              env=Dict("SHP_PHASE_CELLS" => "6", "SHP_PHASE_ORDER" => "2",
                       "SHP_PHASE_FINAL_DISPLACEMENT" => "1.0e-4",
                       "SHP_PHASE_WRITE_OUTPUT" => "false"), tensors=true, slow=false,
              check=check_phase),
             # FCM annular plate (Nitsche + Neumann on immersed arcs), published
             # 8×8 p=4 config; opt-in. Prints one relative L² error.
             (name="fcm_annular_plate_2d", env=Dict{String,String}(), tensors=true, slow=true,
              check=check_single_l2),
             # FCM Kirsch plate-with-hole p-refinement sweep, published 8×8 config;
             # opt-in. Prints the finest-order relative L² error.
             (name="fcm_plate_with_hole_2d", env=Dict{String,String}(), tensors=true, slow=true,
              check=check_single_l2))

    for case in cases
        @testset "$(case.name)" begin
            script = joinpath(examples_dir, case.name, case.name * ".jl")
            if case.slow && !slow_enabled
                @info "skipping slow example (set UNFITTED_TEST_SLOW_EXAMPLES=1 to run)" example = case.name
                @test_skip true
            elseif case.tensors && !tensors_available
                @info "skipping Tensors example (Tensors not in active project; runs under Pkg.test)" example = case.name
                @test_skip true
            else
                ok, output = run_example(script, case.env)
                ok || @error "example exited non-zero" example = case.name output
                @test ok
                ok && case.check(output)
            end
        end
    end
end
