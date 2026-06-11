#!/usr/bin/env julia
# Driver for the Unfitted BenchmarkTools suite. Activates the bench env, pins
# OpenBLAS to one thread (so `Threads.@spawn` assembly workers do not contend
# with BLAS), loads `suite.jl`, tunes once with a cached parameter file, runs
# the suite, and dumps a timestamped JSON to `benchmarks/output/`.
#
# Filter to a sub-tree with `UNFITTEDHP_BENCH_FILTER`, e.g.:
#     UNFITTEDHP_BENCH_FILTER=scenarios/fcm_disk_small \
#     julia --project=benchmarks benchmarks/runbenchmarks.jl

using Pkg
Pkg.activate(@__DIR__)

try
    Pkg.instantiate()
catch err
    @error """
    The benchmark environment could not be instantiated. Unfitted must be
    dev'd into this env once. From the repository root, run:

        julia --project=benchmarks -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
    """ exception=(err, catch_backtrace())
    rethrow()
end

using BenchmarkTools
using Dates
using LinearAlgebra
using Printf
using Statistics
using Unfitted

BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "suite.jl"))

const OUTPUT_DIR = joinpath(@__DIR__, "output")
const PARAMS_PATH = joinpath(OUTPUT_DIR, "params.json")
isdir(OUTPUT_DIR) || mkpath(OUTPUT_DIR)

println("=== Unfitted benchmarks ===")
println("Julia version:  ", VERSION)
println("Julia threads:  ", Threads.nthreads())
println("BLAS threads:   ", BLAS.get_num_threads())
println("Groups:         ", join(sort(collect(keys(SUITE))), ", "))
println()

function _resolve_filter(suite, filter_path)
    isempty(filter_path) && return suite
    target = suite
    for k in split(filter_path, "/")
        target = target[String(k)]
    end
    return target
end

filter_path = get(ENV, "UNFITTEDHP_BENCH_FILTER", "")
isempty(filter_path) || println("Filter:         ", filter_path)
run_target = _resolve_filter(SUITE, filter_path)

if isfile(PARAMS_PATH)
    println("Loading cached parameters from $PARAMS_PATH")
    loadparams!(SUITE, BenchmarkTools.load(PARAMS_PATH)[1], :evals, :samples)
else
    println("Tuning suite (cached at $PARAMS_PATH; delete to retune)...")
    tune!(SUITE)
    BenchmarkTools.save(PARAMS_PATH, params(SUITE))
end

println("\nRunning suite...")
results = run(run_target; verbose=true)

commit = try
    cmd = pipeline(`git -C $(dirname(@__DIR__)) rev-parse --short HEAD`; stderr=devnull)
    strip(read(cmd, String))
catch
    "unknown"
end
timestamp = Dates.format(now(), "yyyymmdd-HHMMSS")
result_path = joinpath(OUTPUT_DIR, "$(timestamp)-$(commit).json")
BenchmarkTools.save(result_path, results)
println("\nSaved results: $result_path")

println("\nSummary (median time, allocated memory, allocation count):")
println("-"^88)
for (path, trial) in BenchmarkTools.leaves(results)
    name = join(path, " / ")
    m = median(trial)
    @printf "  %-62s %10s  %10s  allocs=%d\n" name BenchmarkTools.prettytime(m.time) BenchmarkTools.prettymemory(m.memory) m.allocs
end
println("-"^88)
