# Smoke-test guard for the `examples/` scripts.
#
# The example scripts under `examples/<name>/<name>.jl` are the package's
# paper-regression reproductions — they are the executable form of the
# scientific contract (UMLHP §5.* benchmarks, the FCM moment-fit
# reproductions, the multi-domain coupling, the phase-field and
# traveling-heat-source demonstrations). The rest of the suite unit-tests the
# building blocks; this file is what runs the scripts end to end, so a
# public-API change cannot silently break a paper reproduction without a test
# going red.
#
# For each example it asserts that the script runs to completion and — where
# the example prints a stable, clearly labelled metric (`relative L2 error`,
# `residual norm` from the shared `examples/reporting.jl` formatter) — that the
# metric is finite and within a loose sanity bound. Coarse ≠ accurate: the
# point is "ran end to end and produced sane numbers", not "converged to the
# published digits".
#
# ── One batched subprocess, not ten ───────────────────────────────────────────
#
# Every example used to be launched in its own Julia subprocess. That was
# chosen for three genuine properties — per-case ENV isolation, no collisions
# between the scripts' top-level `const`s, and containment of a hard failure —
# but it paid the ~7.2 s package-load-and-compile floor once per example.
# Measured: the whole set of ten runs in 53.0 s inside a single subprocess,
# against 70.3 s for the six examples that a per-process run could afford. All
# three properties are preserved by other means in `run_examples_child.jl`,
# which is the batch's child half and documents each one at its head:
#
#   - collisions: each example is `Base.include`d into its own fresh `Module`,
#     so its globals are private to it;
#   - ENV: the coarse-size knobs are set immediately around one case and every
#     touched key is restored (or deleted) afterwards;
#   - containment: each case runs under its own `try`/`catch`, so an exception
#     fails only that case's `@test`. The one property that does *not* survive
#     is resilience to a hard crash: a segfault in a native dependency now ends
#     the batch, where a subprocess would have lost only one example. The
#     parent degrades as gracefully as it can — a case whose output block was
#     opened but never closed is reported as `crashed`, and every case that
#     never started is reported as missing — but those examples genuinely do
#     not run. That trade was made deliberately for the ~4× drop in the
#     example suite's compile bill.
#
# The child prints one delimited block per case; this file splits the batch's
# stdout on those delimiters and applies the same per-case checks it always
# has. `-O0` (overriding the parent's optimisation level) is deliberate: these
# coarse runs are dominated by JIT compilation, not by the tiny numerics, so
# unoptimised codegen lowers the wall time with no meaningful runtime cost.
#
# ── Project selection and the `Tensors` gate ──────────────────────────────────
#
# The batch is launched on the *active* project (`Base.active_project()`):
# under `Pkg.test` that is the instantiated test sandbox, which carries
# `Unfitted` and its dependencies (`StaticArrays`, `LinearAlgebra`,
# `ForwardDiff`) *and* `Tensors` (a test/weak dependency); in a bare
# `julia --project=. …` session it is the package root, which lacks `Tensors`.
# The three `Tensors`-using examples (phase-field, both FCM plates) are
# therefore gated on `Tensors` actually being loadable in the active project:
# they run under the full `Pkg.test` suite and are skipped (not failed) in a
# bare-root isolation run.
#
# There is no "slow example" gate. It used to hold back the two FCM plates, the
# bi-material coupling and the tanh layer behind `UNFITTED_TEST_SLOW_EXAMPLES`,
# which no CI job and no documented command ever set — so the only end-to-end
# drivers of FCM-with-Nitsche-on-immersed-arcs, multi-domain interface
# coupling, and the hp-graded order-reduction stack had no automated coverage
# at all. Their measured 9.8–15.5 s also sits inside the "fast" set's own
# 8.0–16.3 s range, so the gate was not separating what it claimed to separate.
# Batched, running all ten costs less than running six did.

@testset "examples" begin
    # The batch runs one child process on the active project.
    # `Base.active_project()` returns a path to a `Project.toml`; `--project`
    # wants the directory containing it.
    child_script = joinpath(@__DIR__, "run_examples_child.jl")
    active_project = something(Base.active_project(), abspath(joinpath(@__DIR__, "..")))
    project_dir = dirname(active_project)

    # Block delimiters. The parent owns them and passes them to the child in
    # the payload, so the two halves cannot drift apart on the protocol.
    markers = (opening="##UNFITTED-EXAMPLE-BEGIN##", closing="##UNFITTED-EXAMPLE-END##",
               done="##UNFITTED-EXAMPLES-DONE##")

    # `Tensors` is a weak/test dependency: it is loadable under `Pkg.test` but
    # not from a bare `--project=.` session. `identify_package` reports whether
    # it is a direct dependency of the active project without loading it.
    tensors_available = Base.identify_package("Tensors") !== nothing

    # Run the whole selection in one subprocess and return its captured
    # `(stdout, stderr)`. The case list travels as a Julia literal in a temp
    # file — the child `include`s it — which keeps the size knobs next to the
    # case they belong to instead of smearing them over the command line.
    # stdout and stderr are read on *separate* pipes: the delimited blocks live
    # on stdout and must not be spliced by anything the child writes to stderr,
    # while stderr is kept for the diagnostic dump when a case has no block.
    # Both pipes are drained concurrently so neither can fill and deadlock.
    function run_batch(selection)
        payload = (markers=markers, cases=[(name=case.name, env=case.env) for case in selection])
        payload_path = tempname() * ".jl"
        write(payload_path, repr(payload))

        try
            cmd = `$(Base.julia_cmd()) -O0 --project=$project_dir --startup-file=no $child_script --cases=$payload_path`
            out_pipe = Pipe()
            err_pipe = Pipe()
            process = run(pipeline(cmd; stdout=out_pipe, stderr=err_pipe); wait=false)
            close(out_pipe.in)
            close(err_pipe.in)
            out_task = @async read(out_pipe, String)
            err_task = @async read(err_pipe, String)
            wait(process)
            return fetch(out_task), fetch(err_task), success(process)
        finally
            rm(payload_path; force=true)
        end
    end

    # Split the child's stdout into one `name => (status, seconds, output)`
    # entry per case. Text outside a block (anything the child printed before
    # the first delimiter, or after the last) is discarded. A block that was
    # opened and never closed means the process died inside that case, so it is
    # recorded as `crashed` with whatever output it had produced.
    function split_blocks(text)
        blocks = Dict{String,@NamedTuple{status::String,seconds::String,output::String}}()
        name = nothing
        buffer = IOBuffer()

        for line in eachline(IOBuffer(text))
            if startswith(line, markers.opening)
                name === nothing ||
                    (blocks[name] = (status="crashed", seconds="?", output=String(take!(buffer))))
                name = strip(line[(length(markers.opening)+1):end])
                buffer = IOBuffer()
            elseif name !== nothing && startswith(line, markers.closing)
                fields = split(line)
                options = Dict(first(kv) => last(kv) for kv in split.(fields[3:end], "="; limit=2))
                blocks[name] = (status=get(options, "status", "unknown"),
                                seconds=get(options, "seconds", "?"), output=String(take!(buffer)))
                name = nothing
            elseif name !== nothing
                println(buffer, line)
            end
        end

        name === nothing ||
            (blocks[name] = (status="crashed", seconds="?", output=String(take!(buffer))))
        return blocks
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
    # headline number (we still guard that they ran to the end).
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
    # `tensors` marks the `Tensors`-using examples; `check` runs the metric
    # assertions on a clean run.
    cases = (
             # Smooth manufactured Laplace, published 12×12 p=4 config (already
             # sub-second numerics); prints one relative L² error.
             (name="laplace_unit_square_smooth", env=Dict{String,String}(), tensors=false,
              check=check_single_l2),
             # 1D bar, unresolved material interface; tiny base + overlay solves,
             # prints a base and an overlay relative L² error.
             (name="bar_1d_unresolved_interface", env=Dict{String,String}(), tensors=false,
              check=check_bar),
             # Small-overlap conditioning sweep; 32 tiny solves, CSV output with no
             # single headline metric — guarded by running to the end only.
             (name="conditioning_small_overlap", env=Dict{String,String}(), tensors=false,
              check=check_success_only),
             # Corner-singularity nested-overlay solve at its published size (small
             # system; cost is compilation); prints one relative L² error.
             (name="singular_square_2d", env=Dict{String,String}(), tensors=false,
              check=check_single_l2),
             # Traveling heat source: shortened transient (`THS_T_MAX=0.05` runs one
             # mesh-update + L²-transfer interval) and VTK export disabled; no single
             # headline metric — guarded by running to the end only.
             (name="traveling_heat_source_2d",
              env=Dict("THS_T_MAX" => "0.05", "THS_WRITE_OUTPUT" => "false"), tensors=false,
              check=check_success_only),
             # Phase-field SENT: coarse base (6×6, p=2), a few load steps to a tiny
             # final displacement, VTK disabled; prints the converged Newton
             # residual norm.
             (name="phase_field_single_edge_notch_2d",
              env=Dict("SHP_PHASE_CELLS" => "6", "SHP_PHASE_ORDER" => "2",
                       "SHP_PHASE_FINAL_DISPLACEMENT" => "1.0e-4",
                       "SHP_PHASE_WRITE_OUTPUT" => "false"), tensors=true, check=check_phase),
             # FCM annular plate (Nitsche + Neumann on immersed arcs), published
             # 8×8 p=4 config. The only end-to-end driver of weak boundary
             # conditions on immersed arcs; prints one relative L² error.
             (name="fcm_annular_plate_2d", env=Dict{String,String}(), tensors=true,
              check=check_single_l2),
             # FCM Kirsch plate-with-hole p-refinement sweep, published 8×8 config;
             # prints the finest-order relative L² error.
             (name="fcm_plate_with_hole_2d", env=Dict{String,String}(), tensors=true,
              check=check_single_l2),
             # Bi-material inclusion corner: native multi-domain coupling of two
             # immersed FCM subdomains across an immersed interface. Coarse
             # VTK-disabled config; guarded by running to the end — the coupled
             # immersed solve completing is the smoke test.
             (name="bimaterial_inclusion_corner_2d",
              env=Dict("BIC_CELLS" => "11", "BIC_WRITE_OUTPUT" => "false"), tensors=false,
              check=check_success_only),
             # Steep tanh layer on a sinusoidal front — the order-reduction benchmark. An
             # hp-graded overlay stack steps the order down to 1 and the base is reduced
             # under it; prints one relative L² error.
             (name="tanh_layer_2d", env=Dict{String,String}(), tensors=false,
              check=check_single_l2))

    # Only the cases that can actually run go into the batch; the rest are
    # reported as skips below.
    selection = [case for case in cases if !(case.tensors && !tensors_available)]
    batch_stdout, batch_stderr, batch_ok = isempty(selection) ? ("", "", true) :
                                            run_batch(selection)
    blocks = split_blocks(batch_stdout)
    batch_finished = occursin(markers.done, batch_stdout)
    # The child's exit status is a signal no per-case block can carry: a case
    # that takes the process down leaves its own block unterminated, but a child
    # that dies *between* cases, or exits non-zero after printing every block,
    # would otherwise pass silently.
    if !(batch_ok && batch_finished)
        @error "example batch did not complete cleanly" batch_ok batch_finished batch_stderr
    end
    @test batch_ok && batch_finished

    for case in cases
        @testset "$(case.name)" begin
            if case.tensors && !tensors_available
                @info "skipping Tensors example (Tensors not in active project; runs under Pkg.test)" example = case.name
                @test_skip true
            else
                block = get(blocks, case.name, nothing)
                ok = block !== nothing && block.status == "ok"
                if block === nothing
                    @error "example never ran (the batch process ended early)" example = case.name batch_finished batch_stderr
                elseif !ok
                    @error "example did not run to completion" example = case.name status = block.status seconds = block.seconds output = block.output
                end
                @test ok
                ok && case.check(block.output)
            end
        end
    end
end
