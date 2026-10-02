# Shared reporting helper used by every example in this directory.
#
# `print_run_report(title, report; parameters, output)` formats the
# diagnostics `NamedTuple` returned by `diagnostics(model, solution)`
# into a compact human-readable report. The print blocks are split out
# so individual examples can call them independently for partial
# reports (e.g. inside a sweep loop). `env` and `sig` below are the two
# helpers every example was otherwise redefining for itself.
#
# Optional report fields are silently skipped. The script does not
# pull in any package functionality beyond what `Unfitted.jl` exports,
# so it can be `include`d from any example without circular imports.
#
# None of this is package API, and none of it may become package API:
# a number formatter is a dumping ground, and reading process
# environment inside the package would be exactly the hidden global
# state `CONTRIBUTING.md`'s public-API design goals rule out.

# Read one environment knob, typed by the default it is given, so that
# an example writes the value it means once and never repeats a
# `parse`. A `Bool` knob compares against the literal `"true"`, which
# is the spelling every example documents; every other `Number` is
# parsed at the default's own type, so an `Int` knob cannot silently
# arrive as a float. `Bool` needs its own method because `Bool <:
# Number` and `parse(Bool, "true")` would otherwise decide the
# question with different spelling rules.
env(name, default::AbstractString) = get(ENV, name, default)
env(name, default::Bool) = get(ENV, name, string(default)) == "true"
env(name, default::T) where {T<:Number} = parse(T, get(ENV, name, string(default)))

# Round to `n` significant digits and render with plain `string`, which
# is what the per-cycle tables in the examples are built from: the
# example environments carry `Unfitted` and nothing else, so the tables
# are laid out with `rpad`/`lpad` rather than a format string, and a
# value of predictable width matters there more than its last digit.
# Closing headline numbers are printed unrounded instead.
sig(x, n=4) = string(round(x; sigdigits=n))

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
# mode, basis-family name, and — where the level carries more than one
# per-cell order — how graded it is.
function print_level_block(report)
    hasproperty(report, :levels) || return nothing
    println("  levels: ", length(report.levels))
    for level in report.levels
        # `nested` is the geometric condition covered-mode pruning wants of the levels
        # above this one. It is reported because losing it is otherwise invisible:
        # the operator stays full rank with a clean residual while the space it
        # spans has quietly shrunk.
        print("    ", level.id, " ", level.role, " cells=", level.cells, " order=", level.order,
              " mode=", level.mode, " basis=", level.basis)
        hasproperty(level, :nested) && print(" nested=", level.nested)
        # `order` is the level's NOMINAL order — the per-axis maximum over its
        # cells — and stays that, so a reader who has always read this field keeps
        # reading the same quantity. On a level an adaptive p-step has graded,
        # that one number is exactly what hides the result: a level carrying
        # orders 2 through 8 prints the same `order=(8, 8)` as a uniform one. The
        # palette says how many distinct per-cell orders the level actually holds
        # and the span of their maxima, which is what makes a graded run
        # reproducible from its own report. Printed only when there is more than
        # one entry, so a uniform level reads exactly as it did before, and
        # written from the palette itself so it is dimension-generic.
        if hasproperty(level, :order_palette) && length(level.order_palette) > 1
            low, high = extrema(maximum, level.order_palette)
            print(" orders=", length(level.order_palette), " in ", low, ":", high)
        end
        # What the level actually contributes. `active` alone would leave the two
        # ways of contributing nothing indistinguishable: a dormant level has
        # enumerated nothing (`raw = 0`), while a level whose artificial boundary
        # took every function it had has `raw > 0` and `active = 0` — the shape a
        # spline overlay too thin to hold a support comes out as.
        if hasproperty(level, :active_functions)
            print(" functions=", level.active_functions, "/", level.raw_functions)
        end
        println()
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
    # How many of those regions have part of their own face outside Ω — the ones
    # whose rule is trimmed to `face ∩ Ω` instead of covering the whole face.
    # Labelled so it
    # does not contain "facet regions" as a substring — the same scraping hazard
    # the "diagonal-scaled condition" line below is named around: the example
    # smoke suite collects metrics by label with a substring regex, so the
    # obvious "cut facet regions" would silently double every reading of the
    # line above it.
    print_optional_report_value("regions on cut faces", report, :cut_facet_region_count)
    print_optional_report_value("surface regions", report, :surface_region_count)
    print_small_overlap_block(report)
    print_optional_report_value("min integration volume", report, :min_integration_volume)
    print_optional_report_value("min relative integration volume", report,
                                :min_relative_integration_volume)
    print_optional_report_value("inactive cell counts", report, :inactive_cell_counts)
    print_optional_report_value("reduced mode counts", report, :reduced_mode_counts)
    print_optional_report_value("cut region count", report, :cut_region_count)
    print_optional_report_value("fit failure count", report, :fit_failure_count)
    print_optional_report_value("moment-fit residual (max)", report, :moment_fit_residual_max)
    print_optional_report_value("symmetry residual", report, :symmetry_residual)
    print_optional_report_value("condition estimate", report, :condition_estimate)
    # Labelled so it does not contain "condition estimate" as a substring: the
    # example smoke suite scrapes metrics by label with a substring regex, and a
    # second line matching the first's label would silently double its readings.
    print_optional_report_value("diagonal-scaled condition", report, :scaled_condition_estimate)
    print_optional_report_value("solver", report, :solver)
    print_optional_report_value("residual norm", report, :residual_norm)
    print_optional_report_value("relative L2 error", report, :l2_error)
    output === nothing || println("  VTK bundle: ", output, ".vtm")
    return nothing
end
