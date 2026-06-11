# Top-level BenchmarkTools suite for Unfitted. Defines `SUITE::BenchmarkGroup`
# and is included by `runbenchmarks.jl` (and by any future PkgBenchmark entry
# point). Each included case file mutates `SUITE` directly under its own group.
#
# Usage:
#     julia --project=benchmarks benchmarks/runbenchmarks.jl
#
# First-time setup (once per checkout):
#     julia --project=benchmarks -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'

using BenchmarkTools

const SUITE = BenchmarkGroup()
SUITE["microkernels"] = BenchmarkGroup()
SUITE["build"] = BenchmarkGroup()
SUITE["assembly"] = BenchmarkGroup()
SUITE["scenarios"] = BenchmarkGroup()

include(joinpath(@__DIR__, "microkernels", "basis.jl"))
include(joinpath(@__DIR__, "microkernels", "geometry.jl"))
include(joinpath(@__DIR__, "microkernels", "fcm.jl"))
include(joinpath(@__DIR__, "build", "integration_plan.jl"))
include(joinpath(@__DIR__, "assembly", "poisson_serial.jl"))
include(joinpath(@__DIR__, "scenarios", "laplace_smooth.jl"))
include(joinpath(@__DIR__, "scenarios", "singular_corner.jl"))
include(joinpath(@__DIR__, "scenarios", "heat_overlay_step.jl"))
include(joinpath(@__DIR__, "scenarios", "fcm_disk_small.jl"))
include(joinpath(@__DIR__, "scenarios", "fcm_disk_high_order.jl"))
