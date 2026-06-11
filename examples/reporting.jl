# Shared reporting helper used by every example in this directory.
#
# `print_run_report(title, report; parameters, output)` formats the
# diagnostics `NamedTuple` returned by `diagnostics(model, solution)`
# into a compact human-readable report. The print blocks are split out
# so individual examples can call them independently for partial
# reports (e.g. inside a sweep loop).
#
# Optional report fields are silently skipped. The script does not
# pull in any package functionality beyond what `Unfitted.jl` exports,
# so it can be `include`d from any example without circular imports.

# Print the `(name, value)` pairs that the example author wants to
# record at the top of the report — typical entries are cell counts,
# polynomial orders, and the value of any sweep parameter that
# distinguishes one run from another.
function print_parameter_block(parameters)
    isempty(parameters) && return nothing
    println("  parameters:")
    for parameter in parameters
        println("    ", first(parameter), ": ", last(parameter))
    end
    return nothing
end

# Print the per-level metadata block of the diagnostics report: one
# line per level showing id, role, cell counts, polynomial order, basis
# mode, and basis-family name.
function print_level_block(report)
    hasproperty(report, :levels) || return nothing
    println("  levels: ", length(report.levels))
    for level in report.levels
        println("    ", level.id, " ", level.role, " cells=", level.cells, " order=", level.order,
                " mode=", level.mode, " basis=", level.basis)
    end
    return nothing
end

# Print the small-overlap diagnostic block: total count, followed by
# the first `limit` records (and a "... N more" line if the list was
# longer). Silently skipped when the report has no small overlaps.
function print_small_overlap_block(report; limit=4)
    count = hasproperty(report, :small_overlap_count) ? report.small_overlap_count : 0
    println("  small overlaps: ", count)
    count == 0 && return nothing

    records = report.small_overlaps
    for record in Iterators.take(records, limit)
        print("    ")
        show(stdout, record)
        println()
    end
    length(records) > limit && println("    ... ", length(records) - limit, " more")
    return nothing
end

# Print one optional `(label, value)` line if the report carries the
# field, the value is meaningful, and the value is not a missing-data
# sentinel (`nothing`, `:none`, `NaN` for floats).
function print_optional_report_value(label, report, name)
    hasproperty(report, name) || return nothing
    value = getproperty(report, name)
    value === nothing && return nothing
    value === :none && return nothing
    value isa AbstractFloat && isnan(value) && return nothing
    println("  ", label, ": ", value)
    return nothing
end

# Top-level entry: print a full report for one run. `title` is the
# run's heading line, `report` is the `NamedTuple` returned by
# `diagnostics(model, solution; exact = …)`, `parameters` is an
# iterable of `(name, value)` pairs added to the report header, and
# `output` (when non-`nothing`) is the VTK bundle path so the report
# tells the reader where the visualisation landed.
function print_run_report(title, report; parameters=(), output=nothing)
    println(title)
    print_parameter_block(parameters)
    print_optional_report_value("dimension", report, :dimension)
    print_level_block(report)
    print_optional_report_value("active unknowns", report, :active_unknowns)
    print_optional_report_value("raw dofs", report, :raw_dofs)
    print_optional_report_value("integration regions", report, :integration_regions)
    print_optional_report_value("facet regions", report, :facet_region_count)
    print_optional_report_value("surface regions", report, :surface_region_count)
    print_small_overlap_block(report)
    print_optional_report_value("min integration volume", report, :min_integration_volume)
    print_optional_report_value("min relative integration volume", report,
                                :min_relative_integration_volume)
    print_optional_report_value("inactive cell counts", report, :inactive_cell_counts)
    print_optional_report_value("cut region count", report, :cut_region_count)
    print_optional_report_value("fit failure count", report, :fit_failure_count)
    print_optional_report_value("moment-fit residual (max)", report, :moment_fit_residual_max)
    print_optional_report_value("symmetry residual", report, :symmetry_residual)
    print_optional_report_value("condition estimate", report, :condition_estimate)
    print_optional_report_value("solver", report, :solver)
    print_optional_report_value("residual norm", report, :residual_norm)
    print_optional_report_value("relative L2 error", report, :l2_error)
    output === nothing || println("  VTK bundle: ", output, ".vtm")
    return nothing
end
