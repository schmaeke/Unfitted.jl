# Batched runner for the `examples/` scripts — the child half of the example
# smoke test driven by `test_examples.jl`.
#
# The parent used to launch every example in its own Julia subprocess. That
# bought three properties for free, all of which this file has to provide by
# other means because the whole set now shares one process:
#
#   1. **No global collisions.** Every example is a self-contained script with
#      top-level `const`s (`E`, `order`, `omega`, …), its own `using` lines and
#      sometimes a `main()`; `include`ing several of them into one namespace
#      would redefine each other's globals. Each example is therefore included
#      into a *private, freshly created* `Module`, so its globals are its own
#      and two examples can disagree about what `const order` means. A fresh
#      `Module` has no `include` of its own, so we install one that closes over
#      the module before including the script — without it the example's own
#      `include(joinpath(@__DIR__, "..", "reporting.jl"))` cannot resolve.
#   2. **No ENV bleed.** The coarse-size knobs (`THS_T_MAX`, `SHP_PHASE_*`,
#      `BIC_*`) are read by the scripts at include time from `ENV`. They are set
#      immediately around one case and every touched key is restored afterwards
#      — deleted if it was previously unset — so no knob survives into the next
#      case or into the rest of the suite.
#   3. **Containment.** A subprocess confined *any* hard failure to one case.
#      Here each case runs under its own `try`/`catch`, so an exception (or a
#      failed `using` of an unavailable package) is recorded as that case's
#      failure and the batch continues with the next one. The honest limit of
#      the trade: a **segfault in a native dependency now takes down the whole
#      batch**, where the old design would have lost only one case. The parent
#      degrades gracefully — every case whose block never closed is reported as
#      crashed — but the remaining examples genuinely do not get to run. That is
#      the price paid for running ten examples in one 7.2 s compilation instead
#      of ten, and it is paid deliberately.
#
# One further property is genuinely lost rather than reproduced: the examples
# now share a process, so a script that mutated process-global state (an RNG
# seed, the BLAS thread count, the working directory) would be visible to every
# case after it. None of the ten does today — the ENV knobs above are the only
# global any of them touches — but an example that needs true isolation is a
# reason to give that case its own subprocess again, not to assume the batch
# will contain it.
#
# Everything an example writes to `stdout` *and* `stderr` is captured per case
# (via a file-backed `redirect_stdio`, which also catches output from native
# libraries) and replayed on this process's real stdout between two delimiter
# lines, so the parent can split the batch output into one clean block per case
# and hand each block to that case's existing metric checks. Capturing rather
# than streaming is what keeps the blocks contiguous: warnings and `@info` from
# an example would otherwise arrive on a second stream and interleave with the
# delimiters at arbitrary byte offsets.
#
# Usage — the parent drives the first form; the second is for running examples
# by hand (`--project=.` works for every example that does not need `Tensors`):
#
#     julia -O0 --project=<env> test/run_examples_child.jl --cases=<payload.jl>
#     julia -O0 --project=. test/run_examples_child.jl laplace_unit_square_smooth
#
# `<payload.jl>` is a file that evaluates to a `NamedTuple`
#
#     (markers = (opening = "…", closing = "…", done = "…"),
#      cases   = [(name = "…", env = Dict("KNOB" => "value")), …])
#
# The parent owns the delimiter strings and passes them down, so parent and
# child cannot drift apart on the protocol. The defaults below are used only by
# the standalone form.

const EXAMPLES_DIRECTORY = abspath(joinpath(@__DIR__, "..", "examples"))

const DEFAULT_MARKERS = (opening="##UNFITTED-EXAMPLE-BEGIN##", closing="##UNFITTED-EXAMPLE-END##",
                         done="##UNFITTED-EXAMPLES-DONE##")

# Path of one example script: `examples/<name>/<name>.jl`.
example_script(name) = joinpath(EXAMPLES_DIRECTORY, name, name * ".jl")

# Turn the command line into the `(markers, cases)` payload. `--cases=<file>`
# supplies both (the parent's form); bare arguments are example names run with
# no size knobs and the default delimiters (the interactive form). Mixing the
# two would leave it ambiguous which case list wins, so it is rejected.
function parse_arguments(arguments)
    payload = nothing
    names = String[]

    for argument in arguments
        if startswith(argument, "--cases=")
            payload === nothing || error("run_examples_child.jl: repeated --cases=")
            payload = include(abspath(argument[(length("--cases=")+1):end]))
        elseif startswith(argument, "--")
            error("run_examples_child.jl: unknown option $argument")
        else
            push!(names, argument)
        end
    end

    if payload !== nothing
        isempty(names) ||
            error("run_examples_child.jl: pass either --cases= or example names, not both")
        return payload
    end

    isempty(names) && error("run_examples_child.jl: no examples given")
    return (markers=DEFAULT_MARKERS,
            cases=[(name=name, env=Dict{String,String}()) for name in names])
end

# Run `body` with `env` layered over the process environment and restore every
# key it touched afterwards, whether `body` returns or throws. Keys that were
# unset before are deleted again rather than left behind as empty strings —
# several examples branch on `haskey`-style `get(ENV, key, default)` defaults.
function with_environment(body, env)
    saved = Dict{String,Union{String,Nothing}}(key => get(ENV, key, nothing) for key in keys(env))
    try
        for (key, value) in env
            ENV[key] = value
        end
        return body()
    finally
        for (key, value) in saved
            value === nothing ? delete!(ENV, key) : (ENV[key] = value)
        end
    end
end

# Include one example script into a private module with its output captured,
# and report `(status, seconds, output)`. `status` is `"ok"` when the script ran
# to its end and `"failed"` when it threw; a thrown error is appended to the
# captured text (message plus backtrace) so the parent's diagnostic dump shows
# what a subprocess's merged stderr used to show.
function run_example(name)
    sandbox = Module(Symbol("Example_", name))
    Core.eval(sandbox, :(include(path) = Base.include(@__MODULE__, path)))

    log = tempname()
    status = "ok"
    elapsed = 0.0

    # Announce the capture file on the real stdout before redirecting. A case
    # that takes the whole process down (a segfault in a native dependency —
    # the one containment property the old per-example subprocess had and this
    # design gives up) never reaches the `read` below, so the parent's only
    # route to its output is this path. The file survives precisely because the
    # `finally` never runs in that case.
    println("  capture: ", log)
    flush(stdout)

    try
        # `flush` before the redirect so nothing still buffered on the real
        # stdout can be swept into the capture file.
        flush(stdout)
        flush(stderr)

        open(log, "w") do io
            redirect_stdio(; stdout=io, stderr=io) do
                started = time()
                try
                    Base.include(sandbox, example_script(name))
                catch error
                    status = "failed"
                    println(io)
                    println(io, "example threw ", typeof(error), ":")
                    showerror(io, error, catch_backtrace())
                    println(io)
                end
                elapsed = time() - started
                flush(io)
                return nothing
            end
            return nothing
        end

        return (status=status, seconds=elapsed, output=read(log, String))
    finally
        # Keep the capture file when the case failed, so a CI log can point at
        # it; a clean case has already had its output folded into the block.
        status == "ok" && rm(log; force=true)
    end
end

# Run the whole batch, one delimited block per case. The opening delimiter is
# printed (and flushed) *before* the case starts, so a case that kills the
# process leaves an unterminated block the parent can attribute to exactly that
# example instead of silently dropping every remaining case.
function run_batch(arguments)
    payload = parse_arguments(arguments)
    markers = payload.markers

    for case in payload.cases
        println(markers.opening, " ", case.name)
        flush(stdout)

        result = try
            with_environment(() -> run_example(case.name), case.env)
        catch error
            # Reached only if the capture machinery itself fails (an
            # unwritable temp directory, a broken redirect) or the environment
            # cannot be restored — anything the script throws, a missing script
            # included, is already caught inside `run_example`.
            (status="failed", seconds=0.0, output=sprint(showerror, error, catch_backtrace()))
        end

        print(result.output)
        endswith(result.output, "\n") || println()
        println(markers.closing, " ", case.name, " status=", result.status, " seconds=",
                round(result.seconds; digits=1))
        flush(stdout)
    end

    println(markers.done, " ", length(payload.cases))
    flush(stdout)
    return nothing
end

run_batch(ARGS)
