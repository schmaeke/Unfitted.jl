#!/usr/bin/env julia

# Pre-commit entry point: format every `.jl` file under the project root
# (or verify formatting in `--check` mode) and print a code-statistics
# summary. Discovery is a filesystem walk, not a git query, so untracked
# and git-ignored sources are included unless their directory is listed in
# `SKIP_DIRECTORIES`.
#
# Usage:
#   julia precommit.jl          # format in place + print stats
#   julia precommit.jl --check  # verify formatting (non-zero exit if any
#                            # file would change) + print stats
#   julia precommit.jl --stats  # stats only; JuliaFormatter is not required
#
# Formatting honours `.JuliaFormatter.toml` at the project root via
# JuliaFormatter's path-based config lookup. The check mode stages each
# file inside the project root for that reason, keeping the `.jl`
# extension — `JuliaFormatter.format` dispatches on it and silently does
# nothing for a path it does not recognise as Julia source, so staging
# without it makes the check report every file as clean.
# `SKIP_DIRECTORIES` excludes generated and output trees, and `.claude`,
# which holds git worktrees during agent-assisted work.

using Printf

const PROJECT_ROOT = @__DIR__
const SKIP_DIRECTORIES = Set([".git", ".claude", "old", "tmp", "output", "refs"])
const _FORMATTER_ERROR = Ref{Any}(nothing)
const _FORMATTER_AVAILABLE = try
    @eval using JuliaFormatter
    true
catch error
    _FORMATTER_ERROR[] = error
    false
end

# ── CLI ────────────────────────────────────────────────────────────────────────

function _parse_mode(arguments)
    isempty(arguments) && return :format
    if length(arguments) == 1
        arg = arguments[1]
        arg == "--check" && return :check
        arg == "--stats" && return :stats
    end
    throw(ArgumentError("usage: julia precommit.jl [--check | --stats]"))
end

# ── File discovery ────────────────────────────────────────────────────────────

function _project_files(root::AbstractString)
    files = String[]

    for (current_root, dirs, names) in walkdir(root)
        filter!(dir -> !(dir in SKIP_DIRECTORIES), dirs)

        for name in sort!(collect(names))
            endswith(name, ".jl") || continue
            push!(files, joinpath(current_root, name))
        end
    end

    return sort!(files)
end

# ── Formatting ────────────────────────────────────────────────────────────────

function _check_formatted(path::AbstractString)
    original = read(path, String)
    # Stage the file inside the project root so JuliaFormatter discovers
    # `.JuliaFormatter.toml` via its path-based config lookup. The staged copy
    # must keep the `.jl` extension: `JuliaFormatter.format` dispatches on it
    # and silently returns without formatting anything for a path it does not
    # recognise as Julia source, which would make this function report every
    # file as clean no matter what it contains.
    temp_dir = mktempdir(PROJECT_ROOT)
    try
        temp_path = joinpath(temp_dir, basename(path))
        write(temp_path, original)
        JuliaFormatter.format(temp_path; overwrite=true, verbose=false)
        return original == read(temp_path, String)
    finally
        rm(temp_dir; force=true, recursive=true)
    end
end

function _report_failure(stage::AbstractString, path::AbstractString, err)
    println(stderr, stage, " failed for ", relpath(path, PROJECT_ROOT), ":")
    showerror(stderr, err)
    println(stderr)
end

# ── Code statistics ───────────────────────────────────────────────────────────

"""
    LineStats(code, comment, docstring, blank)

Per-file line tallies produced by [`_file_stats`](@ref). The fields are:

  - `code` — non-blank executable source lines. A code line with a trailing
    `# note` is counted here only, not double-counted as a comment.
  - `comment` — single-line `#` comments and every line inside a
    `#= … =#` block (including the opening and closing fences).
  - `docstring` — every line of a `\"\"\"…\"\"\"` block that opens at the
    start of a logical line (the canonical docstring placement), including
    the opening and closing fences.
  - `blank` — empty or whitespace-only lines.

Their sum equals the file's physical line count.
"""
struct LineStats
    code::Int
    comment::Int
    docstring::Int
    blank::Int
end

LineStats() = LineStats(0, 0, 0, 0)

function Base.:+(a::LineStats, b::LineStats)
    LineStats(a.code + b.code, a.comment + b.comment, a.docstring + b.docstring, a.blank + b.blank)
end

"""
    _file_stats(path) -> LineStats

Walk a `.jl` file with a small line-state machine and classify each line as
code, comment, docstring, or blank. See [`LineStats`](@ref) for what counts
where and the precise treatment of edge cases.

The classifier identifies a `\"\"\"…\"\"\"` block as a docstring only when
its opening fence is the first non-whitespace token on its line (the
canonical Julia convention). Triple-quoted string literals embedded inside
an expression such as `@error \"\"\"…\"\"\"` are counted as code.
"""
function _file_stats(path::AbstractString)
    code = comment = docstring = blank = 0
    state = :normal  # :normal | :block_comment | :triple_doc | :triple_code

    for raw in eachline(path)
        stripped = lstrip(raw)

        if state === :block_comment
            comment += 1
            occursin("=#", raw) && (state = :normal)
        elseif state === :triple_doc
            docstring += 1
            occursin("\"\"\"", raw) && (state = :normal)
        elseif state === :triple_code
            code += 1
            occursin("\"\"\"", raw) && (state = :normal)
        elseif isempty(stripped)
            blank += 1
        elseif startswith(stripped, "#=")
            comment += 1
            # `#= … =#` may close on the same line; otherwise enter block-comment state.
            tail = SubString(stripped, ncodeunits("#=") + 1)
            occursin("=#", tail) || (state = :block_comment)
        elseif startswith(stripped, "#")
            comment += 1
        elseif startswith(stripped, "\"\"\"")
            docstring += 1
            tail = SubString(stripped, ncodeunits("\"\"\"") + 1)
            occursin("\"\"\"", tail) || (state = :triple_doc)
        else
            code += 1
            # A `"""` opened mid-expression bleeds into the next line as code.
            isodd(count("\"\"\"", raw)) && (state = :triple_code)
        end
    end

    return LineStats(code, comment, docstring, blank)
end

# ── Reporting ─────────────────────────────────────────────────────────────────

const _STATS_BUCKETS = ("src", "ext", "test", "examples", "benchmarks")
const _STATS_RULE = repeat('─', 84)
const _STATS_ROW_FORMAT = Printf.Format("  %-58s %7s %7s %7s\n")
const _STATS_DATA_FORMAT = Printf.Format("  %-58s %7d %7d %7d\n")

function _bucket_for(rel::AbstractString)
    parts = splitpath(rel)
    isempty(parts) && return "other"
    return parts[1] in _STATS_BUCKETS ? parts[1] : "other"
end

"""
    _print_stats(files)

Print a per-file table grouped by top-level directory bucket (`src/`,
`test/`, `examples/`, `benchmarks/`, then `other`), followed by per-bucket
subtotals and an overall total. The output is fixed-width and intended to
remain readable under any CI log viewer.
"""
function _print_stats(files)
    isempty(files) && return

    println()
    println("Code statistics")
    println(_STATS_RULE)
    Printf.format(stdout, _STATS_ROW_FORMAT, "File", "SLOC", "Cmnt", "Docs")
    println(_STATS_RULE)

    aggregates = Dict{String,LineStats}()
    total = LineStats()
    last_bucket = ""

    for path in files
        rel = relpath(path, PROJECT_ROOT)
        bucket = _bucket_for(rel)
        if bucket != last_bucket && !isempty(last_bucket)
            println()
        end
        last_bucket = bucket
        stats = _file_stats(path)
        Printf.format(stdout, _STATS_DATA_FORMAT, rel, stats.code, stats.comment, stats.docstring)
        aggregates[bucket] = get(aggregates, bucket, LineStats()) + stats
        total += stats
    end

    println(_STATS_RULE)
    Printf.format(stdout, _STATS_ROW_FORMAT, "Subtotals", "SLOC", "Cmnt", "Docs")
    println(_STATS_RULE)
    for bucket in (_STATS_BUCKETS..., "other")
        haskey(aggregates, bucket) || continue
        a = aggregates[bucket]
        Printf.format(stdout, _STATS_DATA_FORMAT, bucket * "/", a.code, a.comment, a.docstring)
    end
    println(_STATS_RULE)
    Printf.format(stdout, _STATS_DATA_FORMAT, "Total", total.code, total.comment, total.docstring)
    println()

    return nothing
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    mode = _parse_mode(ARGS)
    needs_formatter = mode in (:format, :check)

    if needs_formatter && !_FORMATTER_AVAILABLE
        println(stderr, "JuliaFormatter is not available.")
        println(stderr, "Install it in your default Julia environment or this project, then rerun:")
        println(stderr, "  julia -e 'import Pkg; Pkg.add(\"JuliaFormatter\")'")
        println(stderr, "Original error: ", sprint(showerror, _FORMATTER_ERROR[]))
        return 1
    end

    files = _project_files(PROJECT_ROOT)

    if mode === :stats
        _print_stats(files)
        return 0
    end

    isempty(files) && return 0

    errors = 0
    unformatted = String[]
    changed = 0

    if mode === :check
        for path in files
            try
                _check_formatted(path) || push!(unformatted, path)
            catch err
                _report_failure("Check", path, err)
                errors += 1
            end
        end
    else  # :format
        for path in files
            try
                original = read(path, String)
                JuliaFormatter.format(path; overwrite=true, verbose=false)
                read(path, String) == original || (changed += 1)
            catch err
                _report_failure("Format", path, err)
                errors += 1
            end
        end
    end

    _print_stats(files)

    if mode === :check
        if !isempty(unformatted)
            println(stderr, "The following files are not correctly formatted:")
            for path in unformatted
                println(stderr, "  ", relpath(path, PROJECT_ROOT))
            end
            println(stderr, "Run `julia precommit.jl` to format the project.")
        end

        if errors > 0
            println(stderr, errors, " file(s) raised errors during check.")
        end

        (isempty(unformatted) && errors == 0) || return 1
        println("Formatting check passed.")
        return 0
    end

    processed = length(files) - errors
    println("Formatted $processed file(s); $changed changed, $(processed - changed) already clean.")
    return errors == 0 ? 0 : 1
end

exit(main())
