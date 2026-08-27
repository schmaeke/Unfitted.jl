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
# For each example it asserts that the script runs to completion and that the
# headline metric it prints (`relative L2 error`, `residual norm`, … from the
# shared `examples/reporting.jl` formatter, the conditioning sweep's CSV rows,
# the bi-material energy error) lands inside that case's `tol` band. Each band
# is the value measured on the configuration the case actually runs, rounded up
# by about one decade — enough headroom that no legitimate rounding difference
# can trip it, tight enough that a regression of one decade does. The bands are
# not a convergence claim: three of the ten cases run coarsened and are far from
# the published digits, but every case is pinned to what *it* produces today.
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
    # and an overlay error) yields one entry per occurrence. The separator is
    # whitespace and/or a colon so the same reader also handles the examples
    # that lay their report out as an aligned two-column table.
    function metric_values(output, label)
        pattern = Regex(label * raw"[\s:]+([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
        return [parse(Float64, m.captures[1]) for m in eachmatch(pattern, output)]
    end

    # A metric is sane when it is finite, non-negative and inside its case's
    # measured band. `NaN`/`Inf` never reach here — the number pattern above
    # does not match them, so a garbage metric fails the count assertion first.
    sane_error(value, tol) = isfinite(value) && 0 ≤ value < tol

    # Metric checks, run only after a clean exit, each taking the case's `tol`.
    # `check_one` covers every example whose report carries exactly one headline
    # number; printing it twice (or not at all) is itself a regression.
    function check_one(output, tol, label)
        values = metric_values(output, label)
        @test length(values) == 1
        @test sane_error(values[1], tol)
    end

    check_single_l2(output, tol) = check_one(output, tol, "relative L2 error")
    check_energy(output, tol) = check_one(output, tol, "relative energy error")
    check_phase(output, tol) = check_one(output, tol, "residual norm")

    function check_bar(output, tol)
        errors = metric_values(output, "relative L2 error")
        @test length(errors) == 2
        @test all(e -> sane_error(e, tol), errors)
        # The overlay around the unresolved interface must beat the base-only
        # run — that improvement is the entire point of the example.
        @test errors[2] < errors[1]
    end

    # The conditioning sweep's report is a CSV body, one row per (p, k); rows
    # are the only lines that open with an integer followed by a comma. Columns
    # are read by position, so the sweep's own header is pinned first — a
    # reordered or inserted column then fails here instead of silently moving
    # the band onto a different quantity. `cond(A)` grows with both p and 1/δ —
    # that growth is what §5.3 measures — so its band is per-order; the residual
    # band is flat, since every one of the 32 rows is a direct solve.
    function check_conditioning(output, tol)
        @test occursin("columns: p, k, δ, active_unknowns, regions, small_regions, min_volume, " *
                       "first_small_volume, cond_estimate, residual_norm", output)
        rows = [split(line, ", ")
                for line in eachline(IOBuffer(output)) if occursin(r"^\s*\d+, ", line)]
        @test length(rows) == 4 * 8
        @test all(row -> sane_error(parse(Float64, row[9]), tol.condition[parse(Int, row[1])]),
                  rows)
        @test all(row -> sane_error(parse(Float64, row[10]), tol.residual), rows)
    end

    # The transient prints no error against an exact solution; the numbers it
    # does produce are the residuals of the L²-transfers that carry the state
    # across each mesh update, printed as one Julia vector. A direct mass solve
    # leaves machine noise, so anything above the band means a transfer failed.
    function check_transfer(output, tol)
        printed = match(r"projection_residuals: \[([^\]]*)\]", output)
        residuals = printed === nothing ? Float64[] :
                    parse.(Float64, split(printed.captures[1], ", "; keepempty=false))
        @test !isempty(residuals)
        @test all(r -> sane_error(r, tol), residuals)
    end

    # Configuration for each example. `env` holds the size knobs; `tensors`
    # marks the `Tensors`-using examples; `check` runs the metric assertions on
    # a clean run and `tol` is the band it enforces. Every `tol` is the measured
    # value on this configuration with about a decade of headroom — the comment
    # on each case records what was measured, so a band that has drifted away
    # from reality is visible without re-running the example.
    cases = (
             # Smooth manufactured Laplace, published 12×12 p=4 config (already
             # sub-second numerics); prints one relative L² error, 2.17e-8.
             (name="laplace_unit_square_smooth", env=Dict{String,String}(), tensors=false,
              check=check_single_l2, tol=1.0e-6),
             # 1D bar, unresolved material interface; tiny base + overlay solves,
             # prints a base and an overlay relative L² error, 3.18e-2 and 1.35e-2.
             # The band is set by the larger, base-only error.
             (name="bar_1d_unresolved_interface", env=Dict{String,String}(), tensors=false,
              check=check_bar, tol=1.0e-1),
             # Small-overlap conditioning sweep; 32 tiny solves, CSV output. Largest
             # `cond(A)` per order (always at δ = 2⁻⁸): 3.65e2, 1.50e3, 6.30e5,
             # 3.65e7; largest residual over all 32 rows 4.15e-15.
             (name="conditioning_small_overlap", env=Dict{String,String}(), tensors=false,
              check=check_conditioning,
              tol=(condition=(4.0e3, 2.0e4, 7.0e6, 4.0e8), residual=1.0e-10)),
             # Corner-singularity nested-overlay solve at its published size (small
             # system; cost is compilation); prints one relative L² error, 4.34e-5.
             (name="singular_square_2d", env=Dict{String,String}(), tensors=false,
              check=check_single_l2, tol=1.0e-3),
             # Traveling heat source: shortened transient (`THS_T_MAX=0.05` runs one
             # mesh-update + L²-transfer interval) and VTK export disabled. No error
             # against an exact solution; the transfer residual measures 0.0.
             (name="traveling_heat_source_2d",
              env=Dict("THS_T_MAX" => "0.05", "THS_WRITE_OUTPUT" => "false"), tensors=false,
              check=check_transfer, tol=1.0e-10),
             # Phase-field SENT: coarse base (6×6, p=2), a few load steps to a tiny
             # final displacement, VTK disabled; prints the converged Newton residual
             # norm, 3.45e-9. Individual load steps reach 4.4e-8, so the band sits a
             # decade above those rather than above the final value.
             (name="phase_field_single_edge_notch_2d",
              env=Dict("SHP_PHASE_CELLS" => "6", "SHP_PHASE_ORDER" => "2",
                       "SHP_PHASE_FINAL_DISPLACEMENT" => "1.0e-4",
                       "SHP_PHASE_WRITE_OUTPUT" => "false"), tensors=true, check=check_phase,
              tol=1.0e-6),
             # FCM annular plate (Nitsche + Neumann on immersed arcs), published
             # 8×8 p=4 config. The only end-to-end driver of weak boundary
             # conditions on immersed arcs; prints one relative L² error, 3.50e-4.
             (name="fcm_annular_plate_2d", env=Dict{String,String}(), tensors=true,
              check=check_single_l2, tol=1.0e-2),
             # FCM Kirsch plate-with-hole p-refinement sweep, published 8×8 config;
             # prints the finest-order relative L² error, 2.93e-8.
             (name="fcm_plate_with_hole_2d", env=Dict{String,String}(), tensors=true,
              check=check_single_l2, tol=1.0e-6),
             # Bi-material inclusion corner: native multi-domain coupling of two
             # immersed FCM subdomains across an immersed interface. Coarse
             # VTK-disabled config; prints the relative energy error against
             # Elhaddad's reference as a percentage, 0.278 % — so the band is in
             # percent too.
             (name="bimaterial_inclusion_corner_2d",
              env=Dict("BIC_CELLS" => "11", "BIC_WRITE_OUTPUT" => "false"), tensors=false,
              check=check_energy, tol=3.0),
             # Steep tanh layer on a sinusoidal front — the order-reduction benchmark. An
             # hp-graded overlay stack steps the order down to 1 and the base is reduced
             # under it; prints one relative L² error, 1.76e-2.
             (name="tanh_layer_2d", env=Dict{String,String}(), tensors=false,
              check=check_single_l2, tol=1.0e-1))

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
                ok && case.check(block.output, case.tol)
            end
        end
    end
end
