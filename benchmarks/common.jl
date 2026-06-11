function best_timed(f; samples=3)
    f()
    best = nothing

    for _ in 1:samples
        GC.gc()
        stats = @timed f()
        if best === nothing || stats.time < best.time
            best = stats
        end
    end

    return best
end

function print_timed(label, stats)
    println("  ", label, " seconds: ", stats.time)
    println("  ", label, " allocated bytes: ", stats.bytes)
    return stats
end
