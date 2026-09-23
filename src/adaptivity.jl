# ── Automated hp adaptivity ───────────────────────────────────────────────────
#
# Three verbs over one loop: estimate where the error is, mark the cells worth
# spending on, refine them. Everything else the loop needs — rebuilding the
# model, carrying the solution across — is already in the package, so this file
# adds no driver and owns no state. The user keeps the loop, as they keep the
# solve loop.
#
# THE RULE. Dörfler on the Bank–Weiser indicator picks the cells worth spending
# on; each marked cell is then sent to h or to p by whether its LAST refinement
# delivered the error reduction that step was entitled to expect. A cell that met
# its prediction is behaving smoothly and takes p again; one that fell short is
# not, and takes h. The predictions are Melenk & Wohlmuth's (Adv. Comput. Math.
# 15, 2001) — see the note above `_partition`.
#
# WHY NOT A SMOOTHNESS INDICATOR. Mitchell & McClain (ACM TOMS 41(1), 2014)
# compared thirteen hp strategies over twenty problems and rate Legendre
# coefficient decay the best general-purpose choice — "COEFDECAY appears to be
# the best choice as a general strategy across all categories of problems". It
# was implemented here, exactly as the literature specifies (a true Legendre
# spectrum obtained by an exact change of basis from the integrated-Legendre
# coefficients, bands grouped by ‖k‖₁, p-refinement where σ > 1 under
# |a_k| ~ c exp(−σ‖k‖₁)), and it was removed, because it does not survive contact
# with THIS loop's operating point.
#
# The reason is structural rather than a bug, and it is worth recording so nobody
# reimplements it. σ > 1 is an asymptotic statement about the tail of an
# expansion, and Mavriplis derived it for spectral elements at p ≈ 8–16. A cell
# at order 1–3 has two to four bands, and the slope through them measures the
# ratio of the linear coefficient to the cell mean — a fact about the local
# amplitude of the solution, not about its analyticity. This loop starts every
# cell at order 1 and keeps most of them low, so the indicator is read exactly
# where it has nothing to say: measured on the SMOOTH λ = 5 tanh front, where
# almost every cell should take p, only 45% of leaves cleared σ > 1, the median σ
# was 0.87 and the lower quartile 0. Prediction has no such floor. It works from
# the first refinement at any order, because it asks what the last step actually
# bought rather than what the coefficients look like.
#
# WHAT IT IS WORTH, against deal.II's step-27 on the same tanh front, at matched
# unknowns and on the same mesh-independent lattice measure, over three regimes
# of the layer steepness λ:
#
#              this rule        step-27        no decision (p until pmax)
#     λ =   5   3.2e-07         6e-07           8.7e-09
#     λ =  30   2.0e-05         1.2e-05         —
#     λ = 120   3.3e-04         8.2e-04         4.2e-03
#
# So: better than the reference where the solution is hard, within 1.7× where it
# is middling, and beaten only by having no decision at all on the easiest case,
# which is the one where a decision has nothing to earn. The decision-free rule
# is the trap worth naming — it is optimal on a smooth solution and it is NOT a
# decision, because it discovers that a cell wanted h only after paying for every
# degree up to the cap; measured against step-27 it won by 23× at λ = 5 and lost
# by 3.6× and 6.1× at λ = 30 and 120, the penalty growing with exactly the
# non-smoothness a decision exists to detect.
#
# The two steps stay INDEPENDENT, which is the one thing that looks redundant and
# is not. It is tempting to fold them together — let the ladder carry an
# increasing order schedule, so deepening the stack over a cell is simultaneously
# an h-step and a p-step, and the whole application collapses to incrementing one
# integer. Measured, that costs a consistent ~1.5×: on the two-feature fixture at
# matched error, coupled reaches 11.63× against uniform where independent reaches
# 17.22×. The reason is that coupling cannot express *pure* p — reaching order 7
# needs four levels of depth, dragging 16× finer cells along with it. It is the
# same failure the inert p-step produced: an order schedule welded to depth
# cannot put a high degree on a large cell. Keeping the steps apart is what lets
# the rule above spend p first and h only when p has run out.

"""
    ErrorEstimate{T,D}

What [`estimate`](@ref) returns: a Bank–Weiser error indicator per cell, with the
scalars a stopping test needs.

  - `cells::Vector{Array{T,D}}` — `cells[k][c]` is `η_K` for cell `c` of level
    `k`, and zero on cells that carry no approximation there.
  - `total::T` — `√Σ_K η²_K`, the global indicator.
  - `reference::T` — `√a(u_h, u_h)` in the problem's own bilinear form, so
    `total / reference` is a scale-free quantity a tolerance can be compared
    against. The problem supplies the norm, so this stays meaningful when the
    operator is not the Laplacian.
  - `consistency::T` — `‖R|_V‖ / ‖R|_W‖`. Galerkin orthogonality makes the
    numerator zero in exact arithmetic, so a value that stops being small says
    the enrichment is no longer the dominant missing content and the indicator
    has entered the regime where it under-reads. It is free — the residual is
    already computed — and it is the only reliability signal available without
    an exact solution.

Read the fields; you never construct one. Same posture as
[`AssemblyDiagnostics`](@ref).
"""
struct ErrorEstimate{T,D}
    cells::Vector{Array{T,D}}
    total::T
    reference::T
    consistency::T
end

function Base.show(io::IO, e::ErrorEstimate)
    print(io, "ErrorEstimate(total=", e.total, ", reference=", e.reference, ", total/reference=",
          e.reference > 0 ? e.total / e.reference : NaN, ", consistency=", e.consistency, ")")
end

# ── Leaf test ─────────────────────────────────────────────────────────────────
#
# A cell is a leaf when no finer level has taken any of it over. The distinction
# is the whole reason the p-step needs guarding: leaf semantics shed a covered
# cell's bubble modes, so raising the order of a non-leaf cell adds nothing that
# survives the dof walk. `refine` sends such a cell to h instead.
#
# "Any active child" rather than "fully covered" is deliberately the conservative
# side of the question: a partly covered cell does keep its bubbles, but it is
# also a cell the stack is already in the middle of taking over, and spending a
# p-step there is at best premature.
function _is_leaf(V::Space{D}, k::Integer, cell::CartesianIndex{D}, below) where {D}
    below === nothing && return true
    children = overlapping_cells(V, [cell]; from=k, to=k + 1)
    for child in CartesianIndices(children)
        children[child] && below[child] && return false
    end
    return true
end

# The active mask of the level below `k`, or `nothing` when there is no level
# below or nothing is live on it — the case that lets `_is_leaf` answer without
# touching the geometry, which is the common one on the finest active level.
function _below(V::Space{D}, k::Integer) where {D}
    k == length(V.levels) && return nothing
    mask = active_cells(V; level=k + 1)
    return any(mask) ? mask : nothing
end

# `V` with every cell's order raised by `inc`. A level whose order is uniform is
# raised uniformly rather than through a per-cell field: it keeps the palette at
# one entry (so the enriched level pays nothing for machinery it does not use),
# and it is the only form a B-spline level accepts, since a per-cell degree has
# no meaning for a whole-axis knot vector.
function _enriched_space(V::Space{D,T}, inc::Int) where {D,T}
    specs = Pair{Int,Any}[]
    for k in 1:length(V.levels)
        level = V.levels[k]
        raised = if length(level.orders.palette) == 1
            nominal_order(level) .+ inc
        else
            map(o -> o .+ inc, cell_orders(V; level=k))
        end
        push!(specs, k => raised)
    end
    return elevate(V, specs...)
end

"""
    estimate(model::Model, solution::Solution; enrichment = 1) -> ErrorEstimate

Bank–Weiser error indicator: build the order-elevated space `V⁺`, inject
`solution` into it, and read the residual in the directions `V` cannot represent.

    η²_K = Σ_{j ∈ W(K)} R_j² / A⁺_jj ,      R = b⁺ − A⁺ u⁺ ,      W = V⁺ ⊖ V

`W` is a set difference on raw dof keys, not a comparison of mode indices: at an
anisotropic order the two disagree, and the index form selects 5 of the 11
complement modes at order `(3, 5)`. Each complement mode is shared equally among
the cells it is incident to, so no cell is charged for a neighbour's error; the
sharing is always well defined because a mode is only shed when *every* incident
cell is covered.

`enrichment` is the order increment defining `V⁺`. One is the usual choice and
the only one measured here.

The indicator's quality is what justifies its cost: effectivity drifts 1.23× over
a 64× range in unknowns, against 52.57× for a gradient-recovery indicator on the
same problems.

The cost is real and worth stating precisely, because it is the loop's largest
single term. Measured on a depth-2 ladder at 24² base cells, one `estimate` is
32 ms against a 3.7 ms solve and a 32 ms rebuild — about 47% of an adaptive step
— and splits as `assemble!(V⁺)` 20.1 ms, `prepare(V⁺)` 8.1 ms, the injection
1.3 ms and the attribution 2.5 ms. The enriched matrix is formed in full and then
read for a diagonal and one matrix–vector product, so a matrix-free path would
remove most of the 63% that assembly costs and none of the 25% that `prepare`
does. That is a capability for `assembly.jl` to grow on its own terms, not
something this file should reach around.

Single-field and single-domain. Integrated Legendre only: raising a B-spline's
degree rewrites the knot vector, so the injection has no meaning (see
[`Rewire`](@ref)), and this raises rather than returning a number that looks
plausible.

See [`refine`](@ref) for the rest of the loop.
"""
function estimate(model::Model{D,T}, solution::Solution; enrichment::Integer=1) where {D,T}
    _assert_single_domain(model, "estimate")
    enrichment >= 1 || throw(ArgumentError("enrichment must be at least 1; got $enrichment"))
    V = model.prefold_space
    for (k, level) in pairs(V.levels)
        level.basis isa IntegratedLegendre ||
            throw(ArgumentError("estimate: level $k carries $(basis_name(level.basis)), and the Bank–Weiser " *
                                "indicator injects the solution into an order-elevated space. Raising that " *
                                "family's degree rewrites its knot vector, so the injection would copy " *
                                "coefficients onto different functions. Only integrated Legendre is supported."))
    end
    model.matrix === nothing &&
        throw(ArgumentError("estimate: the model has not been assembled; call `solve!` or " *
                            "`assemble!` before estimating"))

    plus = _prepared_model(_problem_with_space(model.problem, _enriched_space(V, Int(enrichment))),
                           model.plan_options, model.moment_fit_caches)
    assemble!(plus)
    injected = transfer(solution, model, plus; via=Rewire())
    residual = plus.rhs .- plus.matrix * injected.coefficients

    base_keys = Set(_only_field_layout(model.dofs).dofs.raw_keys)
    plus_field = _only_field_layout(plus.dofs)
    plus_keys = plus_field.dofs.raw_keys
    enriched = plus.problem.space

    # A dof is incident to several cells; its contribution is split equally among
    # them, so a cell is never charged for a neighbour's error.
    incidence = Dict{Int,Int}()
    for k in 1:length(enriched.levels), cell in cell_indices(enriched; level=k)
        for raw in cell_dofs(plus.dofs, k, cell)
            incidence[raw] = get(incidence, raw, 0) + 1
        end
    end

    cells = [zeros(T, level.mesh.cells) for level in enriched.levels]
    counted = Set{Int}()
    residual_V = residual_W = zero(T)
    for k in 1:length(enriched.levels), cell in cell_indices(enriched; level=k)
        raws = cell_dofs(plus.dofs, k, cell)
        actives = active_cell_dofs(plus.dofs, k, cell)
        for (raw, active) in zip(raws, actives)
            active == 0 && continue                       # constrained away
            in_W = !(plus_keys[raw] in base_keys)
            if !(raw in counted)
                push!(counted, raw)
                in_W ? (residual_W += residual[active]^2) : (residual_V += residual[active]^2)
            end
            in_W || continue
            d = plus.matrix[active, active]
            d > 0 || continue
            cells[k][cell] += residual[active]^2 / d / incidence[raw]
        end
    end
    for a in cells
        a .= sqrt.(a)
    end

    coefficients = _checked_coefficients(solution, model)
    reference = sqrt(max(zero(T), dot(coefficients, model.matrix * coefficients)))
    total = sqrt(sum(sum(a .^ 2) for a in cells))
    consistency = sqrt(residual_V / max(residual_W, eps(T)))
    return ErrorEstimate{T,D}(cells, total, reference, consistency)
end

# Dörfler marking: the smallest set of cells carrying `theta` of the total
# squared indicator, ordered by decreasing η_K.
#
# `theta` is the loop's only free constant, and the scheme was chosen so that it
# does not have to be refitted per problem: over a factor-three sweep the
# resulting unknown count moves by at most 1.46×, against 1.86× for a smoothness
# threshold and 1.36× for a benefit/cost rule whose informative window is
# narrower still.
function _dorfler(estimate::ErrorEstimate{T,D}, theta::Real) where {T,D}
    0 < theta <= 1 || throw(ArgumentError("theta must lie in (0, 1]; got $theta"))
    marked = Tuple{Int,CartesianIndex{D}}[]
    for (k, indicators) in pairs(estimate.cells), cell in CartesianIndices(indicators)
        indicators[cell] > 0 && push!(marked, (k, cell))
    end
    isempty(marked) && return marked
    sort!(marked; by=kc -> -estimate.cells[kc[1]][kc[2]])
    goal = theta * estimate.total^2
    accumulated = zero(T)
    taken = length(marked)
    for (i, (k, cell)) in pairs(marked)
        accumulated += estimate.cells[k][cell]^2
        if accumulated >= goal
            taken = i
            break
        end
    end

    # Extend through the group the cut lands in. Dörfler asks for a MINIMAL set
    # reaching `theta`, and any minimal set satisfies it, so this is a free
    # choice — but resolving it by strict comparison hands the decision to
    # round-off, and on a symmetric problem that is the whole decision. Cells
    # related by a symmetry of the problem carry indicators equal only to about
    # 1e-15 relative, never bit-identical, because assembly and the solve are not
    # associative. The sort then orders them arbitrarily and the prefix keeps
    # some of each orbit: measured on a disc with 4-fold symmetry, the four
    # equal-indicator cells at the diagonals split two-and-two, and the mesh
    # refined only the upper-left and lower-right quadrants for ten cycles. The
    # marked counts gave it away — 11, 5 and 7 cells out of an orbit structure
    # that cannot produce an odd count.
    #
    # Completing the group marks slightly more than `theta`, which Dörfler
    # permits (it asks for at least `theta`), and it buys back the symmetry of
    # the problem exactly.
    last = estimate.cells[marked[taken][1]][marked[taken][2]]
    while taken < length(marked)
        next = estimate.cells[marked[taken + 1][1]][marked[taken + 1][2]]
        isapprox(next, last; rtol=sqrt(eps(T))) || break
        taken += 1
    end
    return marked[1:taken]
end

# ── The h-versus-p decision ───────────────────────────────────────────────────
#
# Melenk & Wohlmuth's predicted error reduction (Adv. Comput. Math. 15, 2001),
# in the form deal.II implements it. Each cycle predicts what a cell's indicator
# ought to become if the solution there is as smooth as the step just taken
# assumed; the next cycle compares. A cell that met its prediction is behaving
# smoothly and takes p again; one that fell short is not, and takes h.
#
#     no step taken   η_pred = ∞      (see below; Melenk & Wohlmuth write γ_n η_K)
#     p-step by Δp    η_pred = η_K γ_p^{Δp}
#     h-step          η_pred(K_c)² = n⁻¹ (η_{K_p} γ_h 0.5^{p} γ_p^{Δp})² per child
#
#     take p if η_K < η_pred, else take h
#
# with γ_p = √0.4 and γ_h = 2, the values Melenk & Wohlmuth give and deal.II
# ships. The `0.5^p` is the h-rate of the energy norm, so under inherited
# order the child prediction is η_parent·2^{1−p} in 2D: the theoretical factor
# 2^{−p} with γ_h as the tolerance on it.
#
# Three kinds of cell never reach the comparison:
#
#   - one on the finest level has no level left to activate, so h is not on offer
#     and it takes p;
#   - one that is not a leaf, or one already at `pmax` in every axis, takes h.
#     The first would shed the very modes a p-step gives it; the second has no
#     order left to buy, and routing it to p is what turns a stalled loop into an
#     endless one — measured, a 20-cycle run froze at 494 unknowns with its base
#     mesh pinned at order 8 and its last three cycles identical;
#   - one with NO EVIDENCE takes p: the first cycle, a predecessor that carried no
#     error, or — the common case — a cell that was not marked last cycle and so
#     had no step applied to it. That last exclusion is not a detail. Melenk &
#     Wohlmuth's table has a γ_n row for cells left alone, and deal.II ships
#     γ_n = 1, i.e. "expect the indicator to stand still"; but in THIS loop every
#     marked cell receives a step, so that row can only ever fire for a cell about
#     which the previous cycle learned nothing. Testing it anyway turns the rule
#     into "did your error happen to tick down", which is a coin flip, and the
#     coin decides most of the decisions: measured on the smooth λ = 5 front it
#     split 127 h against 85 p on a solution that wants p almost everywhere, and
#     cost four orders of magnitude against taking p unconditionally. No evidence
#     is not evidence of roughness, and p is the cheap mistake — a wasted degree
#     costs one cycle, a wasted subdivision costs 2^D cells that never go away.
const _GAMMA_P = sqrt(0.4)
const _GAMMA_H = 2.0

# The indicator this cell was predicted to have by now, read off what happened to
# it between `previous` and `V`. Nothing is stored between cycles: the previous
# space and its estimate say what was done, because a p-step shows up as a raised
# order and an h-step as a cell that was not active before.
function _predicted(V::Space{D,T}, previous, k::Integer, cell::CartesianIndex{D}, q_now, caches,
                    gamma_p::Real) where {D,T}
    Vprev, eprev = previous
    k <= length(Vprev.levels) || return T(Inf)
    prev_active, prev_orders = caches
    active = get!(() -> active_cells(Vprev; level=k), prev_active, k)
    q = minimum(q_now)

    if !active[cell]
        # Newly activated, so it is the child of an h-step one level up.
        k == 1 && return T(Inf)
        parents = overlapping_cells(Vprev, [cell]; from=k, to=k - 1)
        coarse = get!(() -> cell_orders(Vprev; level=k - 1), prev_orders, k - 1)
        η = zero(T)
        parent_order = 0
        for par in CartesianIndices(parents)
            parents[par] || continue
            if eprev.cells[k - 1][par] > η
                η = eprev.cells[k - 1][par]
                parent_order = minimum(coarse[par])
            end
        end
        η > 0 || return T(Inf)
        children = prod(cld.(V.levels[k].mesh.cells, V.levels[k - 1].mesh.cells))
        return η * T(_GAMMA_H) * T(0.5)^q * T(gamma_p)^(q - parent_order) /
               sqrt(T(max(children, 1)))
    end

    η = eprev.cells[k][cell]
    η > 0 || return T(Inf)
    q_prev = minimum(get!(() -> cell_orders(Vprev; level=k), prev_orders, k)[cell])
    # Equal orders mean nothing was applied here, so there is no evidence and the
    # caller takes p. A LOWERED order is evidence, and of the opposite sign: the
    # exponent goes negative, gamma_p^(negative) exceeds one, and the cell is
    # predicted to get WORSE by the reciprocal of what the degree was worth. That
    # is Melenk & Wohlmuth's coarsening row, and without it a p-released cell
    # falls through the equality branch and takes p again on sight — which is the
    # oscillation the release was trying to avoid.
    q == q_prev && return T(Inf)
    return η * T(gamma_p)^(q - q_prev)
end

function _partition(V::Space{D,T}, estimate::ErrorEstimate{T,D}, marked, pmax::Integer;
                    previous=nothing, gamma_p::Real=_GAMMA_P) where {D,T}
    nlevels = length(V.levels)
    h = Tuple{Int,CartesianIndex{D}}[]
    p = Tuple{Int,CartesianIndex{D}}[]
    below = Dict{Int,Any}()
    orders = Dict{Int,Any}()
    caches = (Dict{Int,Any}(), Dict{Int,Any}())
    order_of(k) = get!(() -> cell_orders(V; level=k), orders, k)

    for (k, cell) in marked
        1 <= k <= nlevels ||
            throw(ArgumentError("refine: marked level $k is outside the space's 1:$nlevels"))
        if k == nlevels
            push!(p, (k, cell))              # `refine` turns this into the order step
            continue
        end
        saturated = all(>=(Int(pmax)), order_of(k)[cell])
        if saturated || !_is_leaf(V, k, cell, get!(() -> _below(V, k), below, k))
            push!(h, (k, cell))
            continue
        end
        take_p = previous === nothing ||
                 estimate.cells[k][cell] <
                 _predicted(V, previous, k, cell, order_of(k)[cell], caches, gamma_p)
        take_p ? push!(p, (k, cell)) : push!(h, (k, cell))
    end
    return h, p
end

"""
    refine(V::Space, estimate::ErrorEstimate; theta = 0.5, pmax = 8, previous = nothing) -> Space
    refine(V::Space; h = (), p = (), pmax = 8) -> Space

Refine `V` and return the refined space; prepare it in one rebuild with
`adapted(model, refine(model.prefold_space, est))`.

The first form is the loop: mark by Dörfler on `estimate` — the smallest set of
cells carrying `theta` of the total squared indicator — then decide per marked
cell whether to spend on h or on p, and apply. The second form takes the two sets
explicitly as iterables of `(level, cell)` pairs, so an indicator of your own
plugs into the same application step.

The two steps, and what each one means here:

  - the **h-step** activates the cells of the level *below* the marked one that
    cover it, mapped through [`overlapping_cells`](@ref) — refining "this cell"
    means activating its cover one level down — and gives them their parent's
    order unchanged. It is a pure h-step: the resolution changes and the order
    does not. On the finest level there is nothing left to activate, so a cell
    marked for h there takes the order step instead;
  - the **p-step** raises the marked cell's own order by one, capped at `pmax` so
    a loop cannot run away, and activates nothing. It is only ever handed a leaf,
    which is what makes it survive: leaf semantics shed a covered cell's bubble
    modes, so the same increment on a covered cell would be a no-op. Measured
    before that was understood, 356 cells raised while covered retained 576
    active modes between them and **zero** bubbles.

Existing activation is preserved rather than replaced. [`adapt`](@ref) *sets* a
level's mask, so an application that forgets to keep what is already live
silently un-refines the rest of the domain; this keeps it.

Which step a marked cell takes is decided from its own refinement history. Pass
`previous = (V_last, estimate_last)` — the space and estimate from the cycle
before — and each marked cell takes p when its indicator met the reduction its
last step predicted, h when it fell short. Omit `previous` and every marked cell
takes p, falling back to h only where p is unavailable; that is the right
behaviour for the first cycle and the wrong one thereafter. See the note above
`_partition` in `src/adaptivity.jl` for the prediction and why it is preferred
here to a smoothness indicator.

See [`estimate`](@ref) for the rest of the loop.
"""
function refine(V::Space{D,T}, estimate::ErrorEstimate{T,D}; theta::Real=0.5, pmax::Integer=8,
                previous=nothing, gamma_p::Real=_GAMMA_P) where {D,T}
    h, p = _partition(V, estimate, _dorfler(estimate, theta), pmax; previous=previous,
                      gamma_p=gamma_p)
    return refine(V; h=h, p=p, pmax=pmax)
end

function refine(V::Space{D,T}; h=(), p=(), pmax::Integer=8) where {D,T}
    nlevels = length(V.levels)
    pmax >= 1 || throw(ArgumentError("pmax must be at least 1; got $pmax"))
    check(k) = 1 <= k <= nlevels ||
               throw(ArgumentError("refine: marked level $k is outside the space's 1:$nlevels"))

    # One field and one mask per level, built against `V` and merged, because a
    # level can be written from both sides: it holds the p-step of its own cells
    # and the inherited order of the children some coarser cell just activated.
    # `elevate` rejects a level named twice, so merging is not optional.
    masks = Dict{Int,Any}()
    fields = Dict{Int,Any}()
    base = Dict{Int,Any}()
    mask_of(k) = get!(() -> active_cells(V; level=k), masks, k)
    field_of(k) = get!(() -> cell_orders(V; level=k), fields, k)
    base_of(k) = get!(() -> cell_orders(V; level=k), base, k)

    for (k, cell) in p
        check(k)
        field = field_of(k)
        field[cell] = min.(field[cell] .+ 1, Int(pmax))
    end

    for (k, cell) in h
        check(k)
        if k == nlevels
            field = field_of(k)
            field[cell] = min.(field[cell] .+ 1, Int(pmax))
            continue
        end
        children = overlapping_cells(V, [cell]; from=k, to=k + 1)
        target = min.(base_of(k)[cell], Int(pmax))
        mask = mask_of(k + 1)
        field = field_of(k + 1)
        for child in CartesianIndices(children)
            children[child] || continue
            # A child that is not live carries whatever order it held in a
            # previous life, and `coarsen` clears that on release — but only for
            # the covers it releases. Taking the maximum against a stale entry
            # would resurrect it: measured, one h-step onto a released order-6
            # cover returned 169 unknowns where a cold h-step gives 57. The
            # maximum is still right for a child that IS live, whose order was
            # earned and may exceed this parent's.
            field[child] = mask[child] ? max.(field[child], target) : target
            mask[child] = true
        end
    end

    refined = isempty(masks) ? V : adapt(V, (k => masks[k] for k in sort!(collect(keys(masks))))...)
    isempty(fields) && return refined
    return elevate(refined, (k => fields[k] for k in sort!(collect(keys(fields))))...)
end

"""
    coarsen(V::Space; h = (), p = (), pmin = 1) -> Space

Release refinement. The mirror of [`refine`](@ref)'s explicit form, and the verb
a transient needs: on a moving feature, refinement left behind is pure cost.

  - an **h-mark** `(level, cell)` names a *parent*, and deactivates the cells of
    the level below that cover it — the exact inverse of `refine`'s h-step. The
    released children also have their order set back to the parent's, which is
    exactly what `refine`'s h-step gives them — so the two verbs invert each
    other on the whole order field, not merely on its observable part;
  - a **p-mark** `(level, cell)` lowers that cell's own order by one, floored at
    `pmin`.

Releasing a cover whose children are themselves covered is a no-op rather than
an error: the region is carried further down and the deeper level is what holds
it, so releasing this level alone would leave a hole. Release the stack from the
top down and each call is well defined.

Lowering the order of a cell a finer level has taken over is **rejected**. Under
leaf semantics a covered cell has already shed its high-order modes, so the
change is invisible while the cover is live and appears without warning when it
lifts — measured, 173.5× the energy error, with `diagnostics`, `isposdef`, the
residual norm and `estimate`'s consistency flag all unchanged to the last digit.
There is no diagnostic that catches it, so it is refused here instead.

There is deliberately no `coarsen(V, estimate)` form. Which cells to release is
not settled: measured on a translating layer, a monotone loop reached a lower
error at fewer unknowns than every release rule tried, because in a diffusive
problem the heat stays and there is little to release. Release earns its place on
unknowns rather than on error — a monotone loop carries 2.6× the from-scratch
optimum after six layer-widths of travel — so the caller marks, and the
package does not pretend to know the rule.
"""
function coarsen(V::Space{D,T}; h=(), p=(), pmin::Integer=1) where {D,T}
    nlevels = length(V.levels)
    pmin >= 1 || throw(ArgumentError("pmin must be at least 1; got $pmin"))
    check(k) = 1 <= k <= nlevels ||
               throw(ArgumentError("coarsen: marked level $k is outside the space's 1:$nlevels"))

    masks = Dict{Int,Any}()
    fields = Dict{Int,Any}()
    mask_of(k) = get!(() -> active_cells(V; level=k), masks, k)
    field_of(k) = get!(() -> cell_orders(V; level=k), fields, k)

    for (k, cell) in p
        check(k)
        _is_leaf(V, k, cell, _below(V, k)) ||
            throw(ArgumentError("coarsen: cell $cell of level $k is covered by a finer level, " *
                                "and lowering a covered cell's order is not observable until the " *
                                "cover lifts. Release the cover first."))
        field = field_of(k)
        field[cell] = max.(field[cell] .- 1, Int(pmin))
    end

    for (k, cell) in h
        check(k)
        k < nlevels ||
            throw(ArgumentError("coarsen: level $k is the finest, so it covers nothing to release"))
        children = overlapping_cells(V, [cell]; from=k, to=k + 1)
        mask = mask_of(k + 1)
        field = field_of(k + 1)
        # The released children take the PARENT's order, which is exactly what
        # `refine`'s h-step gives them. An inactive cell's order is unobservable,
        # so this is not required for correctness — but it makes the two verbs
        # inverses of the whole order field rather than only of its observable
        # part, which is what lets a transient loop refine and release the same
        # region thousands of times without drift. Note `nominal_order` is NOT the
        # value to use: it is a maximum kept for buffer sizing, and on a level
        # that has been elevated it has already moved.
        parent_order = field_of(k)[cell]
        deeper = k + 1 < nlevels ? _below(V, k + 1) : nothing
        for child in CartesianIndices(children)
            children[child] || continue
            mask[child] || continue
            # A child that is itself covered is not this level's to release.
            _is_leaf(V, k + 1, child, deeper) || continue
            mask[child] = false
            field[child] = parent_order
        end
    end

    released = isempty(masks) ? V :
               adapt(V, (k => masks[k] for k in sort!(collect(keys(masks))))...)
    isempty(fields) && return released
    return elevate(released, (k => fields[k] for k in sort!(collect(keys(fields))))...)
end
