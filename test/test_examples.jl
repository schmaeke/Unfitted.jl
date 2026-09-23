# Smoke-test guard for the `examples/` scripts.
#
# The example scripts live at `examples/<tier>/<name>/<name>.jl` in three
# tiers, and the tiers are why this file matters for more than regression
# pinning:
#
#   - `tutorials/` are the documentation a newcomer reads first. A tutorial
#     that no longer runs, or whose printed numbers no longer match the prose
#     telling the reader what to look for, is worse than no tutorial.
#   - `applications/` are the recognisable problems, and the only place the
#     package is exercised together with third-party packages.
#   - `reproductions/` are the executable form of the scientific contract
#     (the UMLHP §5.* benchmarks, the FCM moment-fit reproductions, and the
#     adaptivity benchmark scored against deal.II step-27).
#
# The rest of the suite unit-tests the building blocks; this file is what runs
# the scripts end to end, so a public-API change cannot silently break a
# tutorial or a paper reproduction without a test going red.
#
# For each example it asserts that the script runs to completion and that the
# headline numbers it prints (`relative L2 error`, `condition estimate`, … from
# the shared `examples/reporting.jl` formatter, plus each example's own extra
# lines) land inside that case's `tol` band. Each band is the value measured on
# the configuration the case actually runs, rounded up by about one decade —
# enough headroom that no legitimate rounding difference can trip it, tight
# enough that a regression of one decade does. The bands are not a convergence
# claim: several cases run coarsened and are far from the published digits, but
# every case is pinned to what *it* produces today, and the comment on each case
# records the measurement, so a band that has drifted away from reality is
# visible without re-running the example.
#
# Where an example's *prose* makes a claim, the claim is asserted rather than
# the number alone. Tutorial 2 tells the reader the overlay beats the base level
# by an order of magnitude, that switching covered-mode pruning off changes the
# answer in no digit that matters, and that `activate!`/`deactivate!` reproduce
# the configuration they rebuild — all three are checked below. Tutorial 4 tells
# the reader that one duplicated B-spline mode costs fourteen orders of
# magnitude of conditioning, so both ends of that span are checked. A tutorial whose text
# and output disagree fails here.
#
# ── One batched subprocess, not one per example ───────────────────────────────
#
# Every example used to be launched in its own Julia subprocess. That was
# chosen for three genuine properties — per-case ENV isolation, no collisions
# between the scripts' top-level `const`s, and containment of a hard failure —
# but it paid the ~7.2 s package-load-and-compile floor once per example.
# Measured on the previous example set: the whole batch ran in 53.0 s inside a
# single subprocess, against 70.3 s for the six examples a per-process run could
# afford. All three properties are preserved by other means in
# `run_examples_child.jl`, which is the batch's child half and documents each
# one at its head:
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
# ── Project selection and the `requires` gate ─────────────────────────────────
#
# The batch is launched on the *active* project (`Base.active_project()`).
# Under `Pkg.test` that is the instantiated test sandbox, which carries
# `Unfitted` and its dependencies (`StaticArrays`, `LinearAlgebra`,
# `ForwardDiff`) *and* every weak dependency listed in the package's `test`
# target — `BasicBSpline`, `FileIO`, `GeometryBasics`, `MeshIO`, `Tensors`. In
# a bare `julia --project=. …` session the active project is the package root,
# which carries none of those, because they are `[weakdeps]` rather than
# `[deps]`.
#
# Each case therefore declares in `requires` the packages it needs beyond
# `Unfitted` itself, and a case whose requirements the active project cannot
# resolve is *skipped* rather than failed. `Base.identify_package` answers that
# question without loading anything. Four cases carry a requirement:
# `tutorials/04_bspline` (`BasicBSpline`), `applications/kirsch_plate_2d`
# (`Tensors`), `applications/imported_geometry_3d` (`FileIO` + `GeometryBasics`
# + `MeshIO`, the three triggers of `UnfittedMeshIOExt`), and
# `applications/time_integration` (`OrdinaryDiffEq`).
#
# `OrdinaryDiffEq` is the one that behaves differently, and the difference is
# worth stating plainly rather than hiding behind the gate. It is not a
# dependency of the package, not a weak dependency, and not in the `test`
# target — deliberately: it is a hundred-package tree that would dominate the
# suite's install and precompile time, and the example exists precisely to show
# that Unfitted needs no extension to work with it. So that case is skipped
# under `Pkg.test` as well as in an isolation run: it has **no automated
# coverage here**, and the only way to run it is on its own project,
#
#     julia --project=examples/applications/time_integration \
#         examples/applications/time_integration/time_integration.jl
#
# which is what the skip message prints. Giving it a band below anyway would be
# dishonest — an unreachable assertion is not a test — so its `tol` is recorded
# for whoever wires up a CI job that does run it, and nothing more.
#
# There is no "slow example" gate. It used to hold several cases behind
# `UNFITTED_TEST_SLOW_EXAMPLES`, which no CI job and no documented command ever
# set — so the only end-to-end drivers of FCM-with-Nitsche-on-immersed-arcs and
# multi-domain interface coupling had no automated coverage at all. Batched,
# running everything costs less than running a subset did.

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

    # Can the active project resolve this package name? `identify_package`
    # answers without loading it, which is what lets a weak dependency be
    # detected rather than attempted.
    package_available(name) = Base.identify_package(name) !== nothing
    missing_packages(case) = filter(!package_available, collect(case.requires))

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
    # formatter, in print order. A multi-run example (Tutorial 1 prints a 1-D
    # and a 2-D error; Tutorial 2 prints five) yields one entry per occurrence.
    # The separator is whitespace and/or a colon so the same reader also handles
    # the examples that lay their report out as an aligned two-column table.
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
        @test all(value -> sane_error(value, tol), values)
    end

    check_single_l2(output, tol) = check_one(output, tol, "relative L2 error")

    # A report line that must carry one *exact* value — a count of cut regions
    # or failed fits, a symmetry residual that has to be zero to the bit. These
    # are structural facts about the run, not converged quantities, so a band
    # would only hide a change in what the example exercises.
    function check_exact(output, label, expected)
        values = metric_values(output, label)
        @test length(values) == 1
        @test all(==(expected), values)
    end

    # Tutorial 1 runs the identical seven-call workflow twice, on the interval
    # and then on the square, so it prints two errors. Checking both is what
    # pins the tutorial's central claim — that nothing in the workflow is
    # dimension-specific — rather than only that the 1-D run happened.
    function check_first_solve(output, tol)
        errors = metric_values(output, "relative L2 error")
        @test length(errors) == 2
        @test all(error -> sane_error(error, tol), errors)
    end

    # Tutorial 2 prints five reports — Run 1 (base only), Run 2 (full overlay),
    # Run 2b (covered-mode pruning off), Run 3 (masked overlay), and a summary that
    # reprints Run 4's numbers, which are Run 3's discretisation rebuilt by
    # `deactivate!`. Each assertion below corresponds to a sentence the tutorial
    # tells the reader to verify from the output.
    function check_overlays(output, tol)
        errors = metric_values(output, "relative L2 error")
        @test length(errors) == 5
        @test all(error -> sane_error(error, tol.coarsest), errors)
        # "Run 2's error is more than an order of magnitude below Run 1's" —
        # the entire point of adding an overlay. Measured factor: 27.3.
        @test errors[2] < errors[1] / 10
        # "Run 2b carries the modes that Run 2 shed and lands on the same error
        # to a dozen digits, so those unknowns really were redundant." If order
        # reduction ever started costing accuracy, it would show here first.
        @test isapprox(errors[3], errors[2]; rtol=1.0e-8)
        # "Run 4's mutated model reproduces Run 3 to the last digit" — the
        # evidence that `activate!`/`deactivate!` really rebuild the dof layout,
        # the integration plan and the cached operators.
        @test isapprox(errors[5], errors[4]; rtol=1.0e-8)
        @test sane_error(errors[5], tol.headline)
        # The number the tutorial's pruning section tells the reader to
        # look for: the base level sheds 121 buried high-order modes under the
        # full overlay, the overlay itself sheds none.
        @test occursin("reduced mode counts: [121, 0]", output)
    end

    # Tutorial 5 is about a structural property, not an error band: the claim is
    # that a *finer* overlay can be *less* accurate when its mesh does not nest,
    # and that the reason is an elimination removing modes nothing replaces. Both
    # halves are asserted, because either one alone is a coincidence away from
    # passing — the anomaly could be noise, and the dimension count could be an
    # arithmetic artefact with no consequence for the answer.
    function check_ladders(output, tol)
        # Part 1's table, read by row: cells, h, nested, unknowns, error.
        rows = [(cells=parse(Int, m.captures[1]), nested=m.captures[2] == "true",
                 unknowns=parse(Int, m.captures[3]), error=parse(Float64, m.captures[4]))
                for m in eachmatch(r"^(\d+)\s+[\d.e+-]+\s+(true|false)\s+(\d+)\s+([\d.e+-]+)\s*$"m,
                                   output)]
        @test length(rows) == 6
        if length(rows) == 6
            by_cells = Dict(r.cells => r for r in rows)
            @test all(sane_error(r.error, tol.coarsest) for r in rows)
            # The nesting verdict is arithmetic: a multiple of the four base
            # cells the box spans nests, and nothing else does.
            @test [r.cells for r in rows if r.nested] == [8, 12, 16]
            # The anomaly the tutorial is built around. A 9-cell overlay is finer
            # than an 8-cell one and carries more unknowns; measured, it is 1.4x
            # less accurate. If this ever stops holding the tutorial's whole
            # argument has gone with it.
            @test by_cells[9].unknowns > by_cells[8].unknowns
            @test by_cells[9].error > by_cells[8].error
            # while the nested subsequence converges as it should
            @test by_cells[16].error < by_cells[12].error < by_cells[8].error
        end

        # Part 1b, the mechanism. The nested stack's reduced space spans exactly
        # what the unreduced one spans; the misaligned stack's does not, and the
        # gap is around a hundred dimensions (measured 108).
        @test occursin(r"Overlay of 8 cells:.*— free", output)
        destroyed = match(r"Overlay of 9 cells:.*— (\d+) dimensions destroyed", output)
        @test destroyed !== nothing
        destroyed === nothing || @test parse(Int, destroyed.captures[1]) > 50

        # A ladder is nested by construction and reports it on every level.
        @test occursin("Nested by construction   : true", output)
        @test !occursin("nested=false", output)

        # Part 3: the mask has to leave the overlay's artificial boundary where
        # the correction is already small. Too tight (r² < 0.02) and the answer
        # is an order worse than the block refinement; wide enough (r² < 0.05)
        # and it matches it with a quarter fewer unknowns.
        errors = metric_values(output, "relative L2 error")
        @test !isempty(errors)
        block = match(r"Both levels, block\s+: (\d+) unknowns, error ([\d.e+-]+)", output)
        sparse = match(r"Level 3 only, r² < 0\.05: (\d+) unknowns, error ([\d.e+-]+)", output)
        @test block !== nothing && sparse !== nothing
        if block !== nothing && sparse !== nothing
            @test isapprox(parse(Float64, sparse.captures[2]), parse(Float64, block.captures[2]);
                           rtol=tol.sparse_vs_block)
            @test parse(Int, sparse.captures[1]) < parse(Int, block.captures[1])
        end
    end

    # Tutorial 6 is the only example that drives `estimate`/`refine` as a loop,
    # and its prose tells the reader what the table must show. Each assertion
    # below is one of those sentences:
    #
    #   - the loop terminates on its own tolerance, not on the step cap;
    #   - `active unknowns` only ever grows, and η falls by orders over the run
    #     — though not monotonically, for the reason recorded at the assertion;
    #   - `consistency` ends below its own peak — the tutorial's claim is not
    #     that it stays small (it does not on this fixture; it climbs while the
    #     h-steps are still pending) but that it comes back down once they land;
    #   - both steps are used. That one is read from the closing report's
    #     `inactive cell counts`: the h-step is the only thing that can wake a
    #     cell of an overlay level, so an overlay whose inactive count is below
    #     its cell count is the evidence that the h/p decision fired at all. A
    #     loop that had degenerated to pure p — which is what omitting
    #     `previous` produces — leaves every overlay fully inactive and fails
    #     here.
    function check_adaptive_hp(output, tol)
        rows = [(step=parse(Int, m.captures[1]), unknowns=parse(Int, m.captures[2]),
                 eta=parse(Float64, m.captures[3]), ratio=parse(Float64, m.captures[4]),
                 consistency=parse(Float64, m.captures[5]), l2=parse(Float64, m.captures[6]))
                for m in
                    eachmatch(r"^\s+(\d+)\s+(\d+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*$"m,
                              output)]
        @test length(rows) ≥ tol.steps
        @test occursin("── tolerance met ──", output)
        if !isempty(rows)
            @test all(diff([row.unknowns for row in rows]) .> 0)
            # η is NOT monotone and the tutorial says why: an h-step wakes cells
            # that carried no indicator before, so the cycle after one reads
            # higher (measured, 5.302e-4 → 9.316e-4 at cycle 13). What has to
            # hold is the fall over the run — measured 640×, banded at 100×.
            @test first(rows).eta > tol.eta_fall * last(rows).eta
            @test last(rows).ratio ≤ tol.eta_ratio
            @test maximum(row.consistency for row in rows) < tol.consistency
            @test last(rows).consistency < maximum(row.consistency for row in rows)
        end

        inactive = match(r"inactive cell counts: \[([\d, ]+)\]", output)
        @test inactive !== nothing
        if inactive !== nothing
            counts = parse.(Int, split(inactive.captures[1], ", "))
            # `ladder(Ω; cells = 8, depth = 4)` doubles each level, so level 2
            # carries 16² cells. Fewer than that inactive means the h-step ran.
            @test length(counts) == 5
            @test counts[2] < 256
        end
        check_one(output, tol.eta_ratio, "eta_over_reference")
        check_single_l2(output, tol.l2)
    end

    # Tutorial 3 is the finite-cell tutorial, so the cut-cell pipeline is the
    # thing under test, not just the error. A cut-region count of zero would
    # mean the annulus had stopped intersecting the grid and the example was
    # quietly demonstrating nothing; a nonzero fit-failure count would mean the
    # moment fit fell back to a raw volume rule, which the tutorial's own text
    # tells the reader must not happen here.
    function check_immersed_fcm(output, tol)
        check_single_l2(output, tol)
        check_exact(output, "cut region count", 56)
        check_exact(output, "fit failure count", 0)
    end

    # Tutorial 4 never calls `diagnostics` with `exact =`, so it prints no
    # "relative L2 error" line at all — its verification is a reference value
    # and a cross-family comparison instead, and requiring an L² band here would
    # simply fail. What it does print, in this order, is a `condition estimate`
    # for Part 1a (plain base), Part 1b (nested overlay, deduplicated), Part 1c
    # (the same stack with deduplication off) and then Part 2; the first three
    # positions are what the section is about, so they are read by position.
    function check_bspline(output, tol)
        # One duplicated base mode shed under the nested overlay: the mechanism
        # the whole of Part 1 exists to show, visible as a number.
        @test occursin("reduced mode counts: [1, 0]", output)

        conditions = metric_values(output, "condition estimate")
        @test length(conditions) ≥ 3
        if length(conditions) ≥ 3
            # "One redundant unknown costs roughly fourteen orders of
            # magnitude of conditioning." Both ends are asserted: the
            # deduplicated nested stack stays a few hundred (measured 687.1),
            # the undeduplicated one is numerically singular (measured
            # 3.43e16). The upper end is a floor rather than a band on purpose:
            # once an operator is singular to working precision its reported
            # condition is 1/eps times an accident, and pinning it two-sidedly
            # would pin the accident.
            @test sane_error(conditions[2], tol.deduplicated)
            @test isfinite(conditions[3]) && conditions[3] > tol.singular
        end

        # Part 1's accuracy check: u_h(½, ½) against the Fourier-series value
        # 0.07367135326539033 for −Δu = 1 on the unit square.
        centre = match(r"u_h\(0\.5, 0\.5\) = ([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)", output)
        @test centre !== nothing
        if centre !== nothing
            reference = 0.07367135326539033
            @test sane_error(abs(parse(Float64, centre.captures[1]) - reference) / reference,
                             tol.centre)
        end

        # Part 2's check: the same immersed problem in two different families on
        # the same mesh, compared pointwise. Agreement at the size of their own
        # discretisation error is all one can ask of two different spaces.
        check_one(output, tol.cross_family, "Largest difference")

        # And the immersed half must actually be immersed, with every cut cell
        # fitted rather than fallen back on.
        cut = match(r"Cut regions / fit failures: (\d+) / (\d+)", output)
        @test cut !== nothing
        if cut !== nothing
            @test parse(Int, cut.captures[1]) > 0
            @test parse(Int, cut.captures[2]) == 0
        end
    end

    # The bi-material joint. Beyond the error, two structural facts: the problem
    # declares `symmetric = true`, so the four coupling blocks must mirror each
    # other to the bit; and the seam is deliberately placed off the grid lines,
    # so a cut-region count of zero would mean it had landed on one and the
    # example had stopped testing what it claims to.
    function check_interface_coupling(output, tol)
        check_single_l2(output, tol)
        check_exact(output, "symmetry residual", 0.0)
        regions = metric_values(output, "cut region count")
        @test length(regions) == 1
        @test all(>(0), regions)
    end

    # The imported-geometry bracket prints two numbers that check different
    # things, and the second is the one that exercises the import. The L² error
    # is governed purely by the z resolution — the exact solution does not vary
    # in x₁ or x₂, so it would look just as good with the level-set sign
    # inverted. The volume error compares the finite-cell quadrature summed over
    # Ω against the analytic volume of the L-prism, and *that* is what a flipped
    # pseudonormal at the reflex edge, or a mis-integrated notch, breaks.
    function check_imported_geometry(output, tol)
        check_single_l2(output, tol.l2)
        check_one(output, tol.volume, "volume error")
        check_exact(output, "fit failure count", 0)
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

    # The curing application prints no error against an exact solution — the
    # problem has none. What it prints instead is a *contrast*: the same
    # transient run twice, once carrying the per-quadrature-point cure state
    # across every activation-driven rebuild and once dropping it. Each
    # assertion below is a sentence of the example's prose.
    function check_curing(output, tol)
        # "the script prints the point count on either side of every rebuild,
        # and they differ every time" — if one rebuild left the quadrature cloud
        # alone, the transfer would be demonstrating nothing there. Guarded by
        # the rebuild count below, since an empty list also reports "all
        # changed".
        @test occursin("(all changed)", output)
        rebuilds = match(r"χ transferred \(RBFP0\)\s+(\d+)\s", output)
        @test rebuilds !== nothing
        if rebuilds !== nothing
            @test parse(Int, rebuilds.captures[1]) ≥ tol.rebuilds
        end

        # "Picard contracts linearly at a rate near 0.2" — the fixed point is
        # reached on every step rather than the iteration cap being hit.
        counts = match(r"Picard iterations per step\s*:\s*\[([0-9, ]+)\]", output)
        @test counts !== nothing
        if counts !== nothing
            @test maximum(parse.(Int, split(counts.captures[1], ", "))) ≤ tol.picard
        end

        # The accuracy claim, and the reason the reference run exists: the
        # carried answer must reproduce the run that never remeshed, not merely
        # differ from the run that threw the state away. Contrast proves the
        # transfer matters; only this proves it is right.
        accuracy = match(r"carried vs reference[^:]*:\s*([0-9.eE+-]+)\s*%", output)
        @test accuracy !== nothing
        if accuracy !== nothing
            @test parse(Float64, accuracy.captures[1]) ≤ tol.accuracy_percent
        end

        # "the RBF interpolant is not monotone … it overshoots a little where
        # the front is steepest". A little is the claim; this is the bound.
        excursion = metric_values(output,
                                  raw"excursion outside \[0, 1\] introduced by the " *
                                  "RBF transfer")
        @test length(excursion) == 1
        @test all(value -> sane_error(value, tol.excursion), excursion)

        # The headline. `cured area = ∫_Ω χ dx` has to be a substantial region
        # when χ rides along, and a small fraction of it when χ is dropped —
        # "the physics comes out wrong, not merely less accurate".
        areas = match(r"cured_area: \(transferred = ([-+0-9.eE]+), dropped = ([-+0-9.eE]+)\)",
                      output)
        @test areas !== nothing
        if areas !== nothing
            transferred, dropped = parse.(Float64, areas.captures)
            @test tol.cured_area[1] < transferred < tol.cured_area[2]
            @test 0 ≤ dropped < transferred / tol.contrast
        end
    end

    # Where the traveling laser writes its ParaView series and its CSV. The
    # series writer deletes nothing, by design, so the directory is cleared
    # before the batch runs — otherwise a stale collection from an earlier run
    # at a different size would satisfy `check_laser_output` on its own.
    LASER_OUTPUT = joinpath(@__DIR__, "..", "examples", "reproductions", "traveling_laser_2d",
                            "output")

    # The write path — the ParaView series and the CSV — has no other automated
    # coverage anywhere in the suite, and it is the half of this example a
    # reader actually looks at. Three claims are asserted: the collection
    # indexes exactly the frames that were written, at the times they were
    # written at and by a path relative to itself; every frame is a full
    # bundle, a data grid plus one mesh per level, which is what makes the mesh
    # following the source visible at all; and the CSV carries the table's own
    # columns with one row per update.
    function check_laser_output(output, rows, tol)
        collection = joinpath(LASER_OUTPUT, "traveling_laser_2d.pvd")
        @test isfile(collection)
        isfile(collection) || return nothing
        datasets = [(time=parse(Float64, m.captures[1]), file=String(m.captures[2]))
                    for m in eachmatch(r"<DataSet timestep=\"([^\"]+)\"[^>]*file=\"([^\"]+)\"",
                                       read(collection, String))]
        @test length(datasets) == tol.frames
        @test all(d -> isfile(joinpath(LASER_OUTPUT, d.file)), datasets)
        if !isempty(datasets)
            # Frame one is the initial condition, which is exactly t = 0, and
            # the collection records each frame's own physical time in order.
            @test first(datasets).time == 0.0
            @test issorted([d.time for d in datasets])
        end
        # `TL_DEPTH = 4` is a five-level ladder, so a full bundle is five level
        # meshes and one data grid.
        frames = readdir(joinpath(LASER_OUTPUT, "traveling_laser_2d_frames"))
        @test count(endswith(".vtu"), frames) == 6 * tol.frames
        @test occursin(Regex("VTK series: .*traveling_laser_2d\\.pvd  \\($(tol.frames) frames\\)"),
                       output)

        csv = readlines(joinpath(LASER_OUTPUT, "history.csv"))
        @test !isempty(csv)
        isempty(csv) && return nothing
        @test first(csv) == "time,unknowns,rel_l2_error,eta,consistency,n_h,n_p,n_released"
        @test length(csv) == length(rows) + 1
    end

    # The traveling laser is the only example that runs `refine` and `coarsen`
    # against each other, so the L² band alone would pass a loop that had
    # stopped doing either. The update table is read instead: every column that
    # records a decision has to be non-empty over the run, and the settled
    # unknown count has to land in a band, because a loop that refined without
    # releasing (or released without refining) still produces a plausible error
    # while costing or resolving the wrong amount.
    #
    # It is also the only case that writes output, and it does so at a frame
    # count that does not divide its step count. That combination is the
    # regression this case exists to hold: the script used to clamp every time
    # step onto the frame grid, so asking for a different number of pictures
    # changed how the PDE was integrated. Hence the two-sided band on the
    # settled error. Measured on this exact configuration: 1686 unknowns at
    # 0.017389 as it stands, against 1632 at 0.014433 with the frame clamp back
    # in place — a 17 % move in the reported error, bought with nothing but a
    # request for three pictures instead of eighty, and one a band open at the
    # bottom waves straight through. Every band below is the one measured with
    # output off, for the same reason.
    function check_laser(output, tol)
        rows = [(unknowns=parse(Int, m.captures[2]), n_h=parse(Int, m.captures[6]),
                 n_p=parse(Int, m.captures[7]), n_rel=parse(Int, m.captures[8]))
                for m in
                    eachmatch(r"^\s*([\d.]+)\s*,\s*(\d+),\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*,\s*(\d+),\s*(\d+),\s*(\d+)\s*$"m,
                              output)]
        @test length(rows) ≥ tol.updates
        @test sum(row -> row.n_rel, rows; init=0) > 0
        @test sum(row -> row.n_h, rows; init=0) > 0
        @test sum(row -> row.n_p, rows; init=0) > 0
        settled = match(r"settled: (\d+) unknowns at relative L2 ([\d.e+-]+)", output)
        @test settled !== nothing
        if settled !== nothing
            @test tol.unknowns[1] < parse(Int, settled.captures[1]) < tol.unknowns[2]
            @test tol.settled[1] < parse(Float64, settled.captures[2]) < tol.settled[2]
        end
        check_single_l2(output, tol.l2)
        check_laser_output(output, rows, tol)
    end

    # The tanh layer's headline is the relative ENERGY error on a lattice that
    # does not depend on the space being measured — the script's own prose says
    # so in as many words — so that is what is banded here, alongside the L²
    # error every other case reports. The rest of the check is structural: the
    # study's claim is that the loop spends h AND p and keeps the order under
    # `PMAX`, and none of that is visible in an error band. The energy column of
    # the table is switched off in this case's `env` and prints `-`, which the
    # row pattern accepts and nothing below reads.
    function check_tanh(output, tol)
        rows = [(unknowns=parse(Int, m.captures[2]), n_h=parse(Int, m.captures[6]),
                 n_p=parse(Int, m.captures[7]), orders=m.captures[8])
                for m in
                    eachmatch(r"^\s*(\d+),\s*(\d+),\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*,\s*(\d+),\s*(\d+),\s*\[([\d, ]+)\]"m,
                              output)]
        @test length(rows) == tol.cycles
        @test sum(row -> row.n_h, rows; init=0) > 0
        @test sum(row -> row.n_p, rows; init=0) > 0
        if !isempty(rows)
            @test all(≤(tol.pmax), parse.(Int, split(last(rows).orders, ", ")))
            @test tol.unknowns[1] < last(rows).unknowns < tol.unknowns[2]
        end
        check_one(output, tol.energy, raw"relative energy error \(fixed 96² lattice\)")
        check_single_l2(output, tol.l2)
    end

    # Configuration for each example. `name` is the script's `<tier>/<name>`
    # path under `examples/`; `env` holds the size knobs; `requires` names the
    # packages beyond `Unfitted` that the case needs in the active project;
    # `check` runs the metric assertions on a clean run and `tol` is the band it
    # enforces.
    cases = (
             # ── Tutorials ───────────────────────────────────────────────────
             # The seven-call workflow, run once on the interval and once,
             # unchanged, on the square. Two relative L² errors: 6.32e-6 (1-D)
             # and 8.91e-6 (2-D).
             (name="tutorials/01_first_solve", env=Dict{String,String}(), requires=(),
              check=check_first_solve, tol=1.0e-4),
             # Overlays, masking and the activation contract. Five relative L²
             # errors: 4.30e-2 (base only) then 1.576e-3 four times over, as
             # the overlay configurations are varied around a fixed answer. The
             # `coarsest` band covers the base-only run, `headline` the final
             # patch-only configuration.
             (name="tutorials/02_overlays", env=Dict{String,String}(), requires=(),
              check=check_overlays, tol=(coarsest=5.0e-1, headline=1.0e-2)),
             # Finite cell method on an annulus with Nitsche on both immersed
             # rims; prints one relative L² error, 1.21e-4, over 56 cut regions
             # with no failed fits.
             (name="tutorials/03_immersed_fcm", env=Dict{String,String}(), requires=(),
              check=check_immersed_fcm, tol=1.0e-3),
             # The B-spline family. No L² error at all (see `check_bspline`);
             # measured instead: 105 unknowns at cond 687.1 deduplicated
             # against 106 at 3.43e16 undeduplicated, u_h(½,½) within 4.67e-5
             # relative of the series value, and 1.15e-5 largest cross-family
             # pointwise gap. Part 1c builds its stack through
             # `prepare(…; prune = false)`; measured bit-for-bit identical to
             # the internal it replaced.
             (name="tutorials/04_bspline", env=Dict{String,String}(), requires=("BasicBSpline",),
              check=check_bspline,
              tol=(deduplicated=1.0e4, singular=1.0e10, centre=1.0e-3, cross_family=1.0e-4)),

             # Nested ladders. No single headline error: the claim is that a
             # finer non-nested overlay is *less* accurate (9 cells against 8)
             # and that the cause is 108 destroyed dimensions, both asserted in
             # `check_ladders`. `coarsest` bands Part 1's whole table, whose
             # worst row is 2.26e-3.
             (name="tutorials/05_ladders", env=Dict{String,String}(), requires=(),
              check=check_ladders, tol=(coarsest=5.0e-2, sparse_vs_block=1.0e-4)),

             # The estimate/refine loop driven to its own tolerance. No size
             # knob: the case is 12.2 s standalone at -O0, and shortening it
             # would cut the half of the run the tutorial is about — the first
             # h-step lands at cycle 12 and the singular cell is not reached
             # until cycle 15. Measured: 16 cycles to `η ≤ 1e-4 · ‖u‖_a`,
             # 529 → 1943 unknowns, η 1.269e-1 → 1.983e-4 (640×), η/‖u‖_a
             # 8.260e-5, relative L² 6.817e-6, `consistency` peaking at 0.5453
             # (cycle 14) and ending at 0.0656, 224 of the first overlay's 256
             # cells still inactive. The `consistency` band is 1.0 rather than
             # a decade above the peak: a decade is meaningless on a ratio, and
             # 1.0 is the value with a meaning — the residual `V` leaves on its
             # own dofs exceeding the one it leaves on the enrichment.
             (name="tutorials/06_adaptive_hp", env=Dict{String,String}(), requires=(),
              check=check_adaptive_hp,
              tol=(l2=1.0e-4, eta_ratio=1.0e-3, eta_fall=100.0, consistency=1.0, steps=10)),

             # ── Applications ────────────────────────────────────────────────
             # Kirsch plate-with-hole p-refinement sweep on a fixed 8×8 grid;
             # prints the finest-order relative L² error, 2.96e-8. The weak form
             # is written in `Tensors.jl` notation, hence the requirement.
             (name="applications/kirsch_plate_2d", env=Dict{String,String}(), requires=("Tensors",),
              check=check_single_l2, tol=1.0e-6),
             # Bonded bi-material joint across a seam that misses the grid
             # lines; prints the worse of the two subdomain errors, 1.82e-5,
             # over 14 cut regions with a bit-exact symmetric operator.
             (name="applications/interface_coupling_2d", env=Dict{String,String}(), requires=(),
              check=check_interface_coupling, tol=1.0e-3),
             # L-bracket whose geometry arrives as a triangle surface mesh.
             # Relative L² error 6.12e-3 (pure z-resolution error), volume error
             # 5.35e-15 — the latter is the one that exercises the import.
             (name="applications/imported_geometry_3d", env=Dict{String,String}(),
              requires=("FileIO", "GeometryBasics", "MeshIO"), check=check_imported_geometry,
              tol=(l2=1.0e-1, volume=1.0e-12)),
             # Transient heat conduction with the time axis owned by
             # `OrdinaryDiffEq`, which is not a dependency of this package and
             # is not in its test target — so this case is *always* skipped
             # here and has no automated coverage. Band recorded for whoever
             # wires up a job that runs it on the example's own project:
             # measured relative L² error 4.10e-4 at t = 0.2, identical to six
             # digits under both integrators.
             (name="applications/time_integration", env=Dict{String,String}(),
              requires=("OrdinaryDiffEq",), check=check_single_l2, tol=1.0e-2),
             # Irreversible thermal curing on a cure front that drives its own
             # overlay activation: the only example carrying `QuadField`
             # per-quadrature-point state and `RBFP0` history transfer through a
             # physical loop. Measured: 6 rebuilds, each moving the quadrature
             # cloud (576 → 1224 points); Picard converges in at most 13
             # iterations per step; the RBF transfer undershoots χ by at most
             # 0.156; cured area 0.1226 carrying χ against exactly 0.0 dropping
             # it. The contrast factor is the point of the example, so it is
             # asserted at ten rather than at the ∞ the run actually delivers.
             (name="applications/thermal_curing_2d", env=Dict{String,String}(), requires=(),
              check=check_curing,
              tol=(rebuilds=3, picard=16, excursion=5.0e-1, accuracy_percent=3.0,
                   cured_area=(1.0e-2, 1.0), contrast=10)),

             # ── Reproductions ───────────────────────────────────────────────
             # Smooth manufactured Laplace, published 12×12 p=4 config (already
             # sub-second numerics); prints one relative L² error, 2.17e-8.
             (name="reproductions/laplace_unit_square_smooth", env=Dict{String,String}(),
              requires=(), check=check_single_l2, tol=1.0e-6),
             # Corner-singularity nested-overlay solve at its published size
             # (small system; cost is compilation); one relative L² error,
             # 4.34e-5.
             (name="reproductions/singular_square_2d", env=Dict{String,String}(), requires=(),
              check=check_single_l2, tol=1.0e-3),
             # Small-overlap conditioning sweep; 32 tiny solves, CSV output.
             # Largest `cond(A)` per order (always at δ = 2⁻⁸): 3.65e2, 1.50e3,
             # 6.30e5, 3.65e7; largest residual over all 32 rows 4.15e-15.
             (name="reproductions/conditioning_small_overlap", env=Dict{String,String}(),
              requires=(), check=check_conditioning,
              tol=(condition=(4.0e3, 2.0e4, 7.0e6, 4.0e8), residual=1.0e-10)),
             # Traveling laser: shortened transient (`TL_T_MAX = 0.8`, a fifth
             # of one revolution, `TL_DEPTH = 4`). Measured on that
             # configuration: 31 updates, Σn_h = 32, Σn_p = 119,
             # Σn_released = 36, settling at 1686 unknowns and relative L²
             # 1.7389e-2, 29.0 s wall standalone at four threads. The unknown
             # band is deliberately narrow (1.4× either side) rather than a
             # decade, because the one regression it exists to catch has a
             # known signature: computing the release marks on the refined
             # space instead of the estimate's own space cancels most of the
             # h-steps and settles at 1100 instead, which a wider band would
             # wave through.
             #
             # This is the only case that leaves VTK on, and the only automated
             # coverage the series writer and the CSV have. `TL_VTK_FRAMES = 3`
             # buys that for about 15 s and ~5 MB of frames, at a frame count
             # deliberately coprime with the 32 steps — see `check_laser` for
             # why a misaligned frame count is the interesting one. Every band
             # below was measured with output off, so a frame schedule that
             # touched the discretisation would fail here rather than quietly
             # re-baseline.
             #
             # The full study (t = 4, depth 5) settles at 9346 unknowns and
             # 0.452 % in 13.1 minutes. CONTRIBUTING's rule is that a
             # reproduction is not coarsened for speed, and this case breaks it
             # knowingly: thirteen minutes is more than the rest of the example
             # suite put together. What the shortened run gives up is the
             # settled state of the trail, not any branch of the loop — every
             # verb, including `coarsen`, fires inside the first fifth of a
             # revolution.
             (name="reproductions/traveling_laser_2d",
              env=Dict("TL_T_MAX" => "0.8", "TL_DEPTH" => "4", "TL_WRITE_OUTPUT" => "true",
                       "TL_VTK_FRAMES" => "3"), requires=(), check=check_laser,
              tol=(l2=1.0e-1, updates=20, unknowns=(1200, 2600), settled=(0.0170, 0.0178),
                   frames=3)),

             # Automated hp on a curved interior layer: 22 adaptive cycles from
             # a linear 6×6 base at the published size, VTK off, and the
             # per-cycle lattice column off (`TANH_TABLE_ENERGY = false`).
             # Measured standalone at -O0: 35.7 s wall with that column on,
             # 15.7 s with it off, for a bit-identical answer — 23 evaluations
             # of a 96² × 25-point ruler cost more than every solve in the run
             # put together. The single closing evaluation stays, because it is
             # the number the benchmark is about, and it is banded here.
             # Measured: 3739 unknowns, relative energy error 2.4774e-3,
             # relative L² 3.5277e-4, Σn_h = 60, Σn_p = 296, final max order
             # per level [3, 5, 7, 6, 1, 1].
             (name="reproductions/adaptive_tanh_layer_2d",
              env=Dict("TANH_WRITE_OUTPUT" => "false", "TANH_TABLE_ENERGY" => "false"), requires=(),
              check=check_tanh,
              tol=(l2=5.0e-3, energy=2.0e-2, pmax=8, cycles=22, unknowns=(1900, 7500))))

    # Only the cases whose requirements the active project can resolve go into
    # the batch; the rest are reported as skips below.
    selection = [case for case in cases if isempty(missing_packages(case))]
    rm(LASER_OUTPUT; force=true, recursive=true)
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
            unavailable = missing_packages(case)
            if !isempty(unavailable)
                @info "skipping example (the active project cannot resolve its dependencies); " *
                      "run it on its own project with " *
                      "`julia --project=examples/$(case.name) examples/$(case.name)/" *
                      "$(basename(case.name)).jl`" example = case.name missing = unavailable
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
