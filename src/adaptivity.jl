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
# not, and takes h. The predictions are Melenk & Wohlmuth's (2001, see
# References) — see the note above `decide`.
#
# WHY NOT A SMOOTHNESS INDICATOR. Mitchell & McClain (2014, see References)
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
# expansion, and Mavriplis (1994, see References) derived it for spectral
# elements at p ≈ 8–16. A cell at order 1–3 has two to four bands, and the slope
# through them measures the ratio of the linear coefficient to the cell mean — a
# fact about the local amplitude of the solution, not about its analyticity. This
# loop starts every cell at order 1 and keeps most of them low, so the indicator
# is read exactly where it has nothing to say: measured on the SMOOTH λ = 5 tanh
# front, where almost every cell should take p, only 45% of leaves cleared σ > 1,
# the median σ was 0.87 and the lower quartile 0. Prediction has no such floor.
# It works from the first refinement at any order, because it asks what the last
# step actually bought rather than what the coefficients look like.
#
# WHAT IT IS WORTH, against step-27 of deal.II 9.5 (Arndt et al. 2023, see
# References) on the same tanh front, at matched unknowns and on the same
# mesh-independent lattice measure, over three regimes of the layer steepness λ.
# step-27 is the right comparison precisely because it is the other family: its
# h/p decision is a Fourier-coefficient-decay smoothness estimate
# (`SmoothnessEstimator::Fourier::coefficient_decay` feeding
# `hp::Refinement::choose_p_over_h`), so these rows measure prediction against a
# smoothness indicator on the same problem and the same marking.
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
# integer. Measured, that costs a consistent ~1.5×: on a two-feature problem —
# u = |x − (0.3, 0.3)|^1.5 plus a Gaussian bump of width 0.08 at (0.7, 0.7) on
# the unit square, against uniform order 4 — coupling reaches a relative energy
# error of 2.6e-4 at 5239 unknowns, an 11.63× saving, where the independent
# increments reach it at 3521, a 17.22× saving. Both figures are one point of a
# curve and only comparable at a stated error: the same independent loop reads
# 4.41× at 1e-3. The reason for the gap is that coupling cannot express *pure* p
# — reaching order 7 needs four levels of depth, dragging 16× finer cells along
# with it. It is the same failure the inert p-step produced: an order schedule
# welded to depth cannot put a high degree on a large cell. Keeping the steps
# apart is what lets the rule above spend p first and h only when p has run out.
#
# ── References ────────────────────────────────────────────────────────────────
#
# The four published algorithms this file implements, and the reference
# implementation its numbers are measured against.
#
#   R. E. Bank, A. Weiser, "Some a posteriori error estimators for elliptic
#     partial differential equations", Math. Comp. 44 (1985) 283–301,
#     doi:10.1090/S0025-5718-1985-0777265-X.
#     The error indicator: the residual of the computed solution read in an
#     enriched space, taken through the diagonal of the enriched operator.
#     `estimate` computes it.
#
#   W. Dörfler, "A convergent adaptive algorithm for Poisson's equation", SIAM
#     J. Numer. Anal. 33 (1996) 1106–1124, doi:10.1137/0733054.
#     The bulk-chasing marking rule — the smallest set of cells carrying a
#     fixed fraction θ of the total squared indicator. `mark_cells` implements it.
#
#   J. M. Melenk, B. I. Wohlmuth, "On residual-based a posteriori error
#     estimation in hp-FEM", Adv. Comput. Math. 15 (2001) 311–331,
#     doi:10.1023/A:1014268310921.
#     The predicted error reduction that decides h against p, and the constants
#     γ_p² = 0.4, γ_h² = 4, γ_n² = 1. The note above `decide` states the
#     prediction in this package's form.
#
#   W. F. Mitchell, M. A. McClain, "A comparison of hp-adaptive strategies for
#     elliptic partial differential equations", ACM Trans. Math. Softw. 41
#     (2014) 1, 1–39, doi:10.1145/2629459.
#     The thirteen-strategy comparison quoted under WHY NOT A SMOOTHNESS
#     INDICATOR.
#
#   C. Mavriplis, "Adaptive mesh strategies for the spectral element method",
#     Comput. Methods Appl. Mech. Engrg. 116 (1994) 77–86,
#     doi:10.1016/S0045-7825(94)80010-3.
#     The σ > 1 coefficient-decay criterion measured and rejected above.
#
#   D. Arndt, W. Bangerth, M. Bergbauer, M. Feder, M. Fehling, J. Heinz,
#     T. Heister, L. Heltai, M. Kronbichler, M. Maier, P. Munch, J.-P. Pelteret,
#     B. Turcksin, D. Wells, S. Zampini, "The deal.II library, version 9.5",
#     J. Numer. Math. 31 (2023) 231–246, doi:10.1515/jnma-2023-0089.
#     Tutorial step-27 is the comparison target above.
#     `hp::Refinement::predict_error` (`include/deal.II/hp/refinement.h`) is the
#     form of Melenk & Wohlmuth's prediction this file follows, and ships their
#     constants unsquared as γ_p = √0.4, γ_h = 2, γ_n = 1.

"""
    ErrorEstimate{D,T}

What [`estimate`](@ref) returns: a Bank–Weiser error indicator per cell, with the
scalars a stopping test needs.

  - `cells::Vector{Array{T,D}}` — `cells[k][c]` is `η_K` for cell `c` of level
    `k`, and zero on cells that carry no approximation there.
  - `total::T` — `√Σ_K η²_K`, the global indicator.
  - `reference::T` — `√(cᵀ A c)` over the *active* unknowns, in the problem's own
    bilinear form: the energy of the part of `u_h` the unknowns determine. Under
    homogeneous Dirichlet data that is `√a(u_h, u_h)`; with nonzero data it is
    not, because the constrained columns are folded into the right-hand side and
    the lift's own energy is never formed, so what is measured is the interior
    part of the solution and the quantity is not invariant under adding a
    constant to the Dirichlet data. Measured on an 8² order-2 Poisson problem
    with `u = sin πx sin πy`: adding `C` to the boundary data leaves `total` and
    the discrete energy error bit-identical at 1.8257e-2 while `reference` runs
    2.2214, 6.8407 and 53.137 at `C` = 0, 1 and 10, moving `total / reference` by
    24× with nothing about the approximation changed. It is still scale-free and
    still what a tolerance should be compared against on one problem; it is not a
    quantity to compare across different boundary data. The problem supplies the
    norm, so this stays meaningful when the operator is not the Laplacian. A
    `reference` of exactly zero means the discrete energy came out below
    round-off, and `show` prints the ratio as `NaN` rather than dividing.
  - `consistency::T` — `‖R|_V‖ / ‖R|_W‖`, the residual the injected solution
    leaves on `V`'s own dofs against what it leaves on the enrichment. It is NOT
    a saturation check. `V`'s functions and integration regions are identical
    inside `V⁺`, so the numerator can only differ from the converged residual
    through what the order change moved: the quadrature the load and the
    coefficients are integrated with, the Dirichlet trace projected at order
    `q + 1` instead of `q`, and the solver residual. It is therefore a free
    data-oscillation and solve check, and it says nothing about whether `W`
    captures the missing content: a solution whose error happens to be
    orthogonal to the enrichment gives a small `consistency` and a small `η` at
    once. Measured on a 4² order-2 space, changing nothing but the data — the
    non-polynomial `sin πx sin πy` replaced by `x² + y²`, which both rules
    integrate exactly and whose trace `V` reproduces — drops it from 7.1e-4 to
    5.2e-7. A value that climbs means the data is under-integrated where `η` is
    smallest.

    For an actual saturation signal there is one the loop already has, under two
    conditions: on a refine-only sequence with homogeneous Dirichlet data,
    Galerkin orthogonality makes `est_new.reference² − est_prev.reference²` the
    true reduction in squared energy error, so a previous `total²` well below
    that gain says the previous estimate under-read. Both conditions are load
    bearing — a [`coarsen`](@ref) breaks the nesting, and a nonzero lift breaks
    the identification of `reference` with `√a(u_h, u_h)` above.

Read the fields; you never construct one. Same posture as
[`AssemblyDiagnostics`](@ref).
"""
struct ErrorEstimate{D,T}
    cells::Vector{Array{T,D}}
    total::T
    reference::T
    consistency::T
end

function Base.show(io::IO, e::ErrorEstimate)
    print(io, "ErrorEstimate(total=", e.total, ", reference=", e.reference, ", total/reference=",
          e.reference > 0 ? e.total / e.reference : NaN, ", consistency=", e.consistency, ")")
end

# ── Reading the stack under one cell ──────────────────────────────────────────
#
# Both questions below are asked once per marked cell and are answered from the
# per-axis block `_cell_block` (`src/ladder.jl`) resolves by binary search, never
# from `overlapping_cells`, which builds a mask over the whole target level. On a
# depth-3 3D ladder that is 0.29 µs against 7.65 ms per mark, and a cycle marks
# hundreds of them.
#
# A cell is a leaf when no finer level has taken any of it over. The distinction
# is the whole reason the p-step needs guarding: leaf semantics shed a covered
# cell's bubble modes, so raising the order of a non-leaf cell adds nothing that
# survives the dof walk. `refine` sends such a cell to h instead.
#
# EVERY finer level is asked, not only the one immediately below. The dof layer's
# covering rule tests each higher-id level in turn (`build_coverage` in
# `src/coverage.jl`), so on a stack where level k+2 is live over a cell while k+1
# is dormant — a structure `ladder` admits by design and one `adapt` call builds
# — the cell is covered and its bubbles are shed. Asking k+1 alone called such a
# cell a leaf, sent it to p, and the order step it was given then vanished in the
# dof walk: a mark spent on nothing, every cycle, for as long as the skip lasted.
#
# "Any active cell of a finer level overlaps this one" rather than "fully
# covered" is deliberately the conservative side of the question: a partly
# covered cell does keep its bubbles, but it is also a cell the stack is already
# in the middle of taking over, and spending a p-step there is at best premature.

# Which levels hold an active cell at all, as one `Bool` per level. A verb builds
# this once and both tests below skip a dormant level without touching the
# geometry — which on a declared-but-empty ladder is every level under the mark,
# and is the common case. Measured on a depth-3 3D ladder with empty overlays,
# the leaf test on a base cell falls from 6.19 µs to 0.05 µs; its worst case —
# a live 64³ level that does not reach the marked cell, so the whole block is
# read and none of it is live — is 4.5 µs.
_live_levels(V::Space) = [level.mask === nothing || any(level.mask.on) for level in V.levels]

function _is_leaf(V::Space{D}, k::Integer, cell::CartesianIndex{D}, live=_live_levels(V)) where {D}
    for j in (k+1):length(V.levels)
        live[j] || continue
        mask = V.levels[j].mask
        for c in _cell_block(V, k, j, cell)
            is_active(mask, c) && return false
        end
    end
    return true
end

# Whether an h-step on this cell would buy anything: level k+1 must exist, its
# box must reach the cell, and the block must hold a cell that is both dormant
# and a leaf.
#
# The three conditions are what makes the h-step total on an arbitrary stack, and
# each rules out a mark that would otherwise be re-issued every cycle for
# nothing. A level below that does not reach the cell — a sub-box overlay, or two
# overlays over different regions — offers no h-step, exactly as the finest level
# offers none, so both fall under one rule rather than a level-count test. A
# block already fully live offers none either: the step would set no mask bit. And
# a dormant cell that a *deeper* level has already taken over is dof-inert to
# activate, because the same covering rule that sheds the marked cell's bubbles
# sheds its would-be child's: measured on a depth-2 ladder with level 3 live over
# a base cell and level 2 empty, activating level 2 under it moved the active
# unknowns by exactly zero, 97 before and 97 after.
function _has_h_step(V::Space{D}, k::Integer, cell::CartesianIndex{D},
                     live=_live_levels(V)) where {D}
    k < length(V.levels) || return false
    mask = V.levels[k + 1].mask
    for c in _cell_block(V, k, k + 1, cell)
        is_active(mask, c) && continue
        _is_leaf(V, k + 1, c, live) && return true
    end
    return false
end

# The cells of one level that carry an approximation. `cell_dofs` answers with an
# empty vector on a dormant cell, so restricting the estimator's two sweeps to
# these changes no result — only how much of a level has to be walked to find
# that out. That is the point on a ladder, where a level holds N^D cells however
# few are switched on: a depth-5 3D chain over a 4³ base declares 2 097 152 of
# them on its finest level. The mask scan is one pass over a `BitArray`, cheap
# against the per-cell dof lookup and the vector it allocates per component.
function _live_cells(level::Level)
    level.mask === nothing ? cell_indices(level.mesh) :
    CartesianIndices(level.mask.on)[level.mask.on]
end

# `V` with every cell's order raised by `inc`. A uniform level is raised as a
# tuple rather than through a per-cell field, and the reason is cost rather than
# meaning: `_normalize_order` collapses an all-equal field back to exactly this
# tuple, so the per-cell path arrives at the same level having materialised three
# N^D arrays on the way (the field `cell_orders` builds, the raised copy, and the
# palette scan that collapses it). Measured on a uniform 512² level, 7.9 ms and
# 13.0 MiB against 1.3 ms and 1.0 MiB for the tuple, for an identical result.
#
# The generator's element type is `Pair{Int,Union{NTuple{D,Int},Array{...,D}}}`,
# which is what `elevate(V, pairs::Pair{<:Integer}...)` already dispatches on, so
# no `Pair{Int,Any}` accumulator is needed to hold the two shapes together.
function _enriched_space(V::Space{D,T}, inc::Int) where {D,T}
    raised(k) = (o=V.levels[k].orders;
                 length(o.palette) == 1 ? o.nominal .+ inc :
                 map(p -> p .+ inc, cell_orders(V; level=k)))
    return elevate(V, (k => raised(k) for k in 1:length(V.levels))...)
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
cell is covered. Every component of the field is charged to its own cell: on a
field declared with `components = n` the sum above runs over all `n` components
of each complement mode, so `η_K` measures the whole vector-valued residual.

`enrichment` is the order increment defining `V⁺`. One is the usual choice and
the only one measured here.

The indicator's quality is what justifies its cost: effectivity drifts 1.23× over
a 64× range in unknowns, against 52.57× for a gradient-recovery indicator on the
same problems.

The cost is worth stating, because `estimate` is the loop's largest single term
and stays so at every order. Kernel work per cell is modes² × quadrature points
and each factor carries a `(p+1)^D`, so raising every cell by one degree
multiplies the assembly by about `((p+2)/(p+1))^(3D)` — in 2D, 11.4 / 5.6 / 3.8
at `p` = 1 / 2 / 3. Measured on the ladder of
`examples/reproductions/adaptive_tanh_layer_2d` (6² base, depth 5, a nested band
of active cells along the front; four threads, `-O0`, two runs), `assemble!(V⁺)`
against `assemble!(V)` comes out at 7.5× / 4.9–5.4× / 3.5×: the prediction from
`p` = 2 up, and short of it at `p` = 1, where per-region work that does not scale
with the mode count is still a large share of an 18 ms assembly. `estimate` is
then 78–82% of `prepare + assemble + solve + estimate` at all three orders, worst
at the low orders the loop starts from. Inside it the split MOVES with the
order — `assemble!(V⁺)` runs from about half of `estimate` at `p` = 1 to about
84% at `p` = 3, and `prepare(V⁺)` from about a fifth down to 3% — so no fixed
split is worth quoting. The injection and the residual matvec are nowhere in it,
at 1.3–7.6 ms and 0.5–4.4 ms across the same three orders.

None of those shares is all of the assembly. `plus` is a fresh model on every
call, so it also pays the symbolic pattern and the threaded gather plan in full
and amortises them over a single assembly — the opposite of the Newton or
transient loop those caches were built for. Symbolic against numeric inside
`assemble!(V⁺)`, same ladder and four threads: 63 / 135 / 439 ms against
103 / 556 / 1796 ms at `-O0`, and 78 / 120 / 253 ms against 22 / 74 / 249 ms at
the default `-O2`, at `p` = 1 / 2 / 3. So read `-O0` figures as `-O0` figures:
`estimate` is 3–5× faster at `-O2` and `assemble!(V)` 6–7×, because the kernel is
the part that shrinks — at `-O2` the symbolic half is the larger of the two at
`p` = 1 and 2 and level with the numeric at `p` = 3.

A matrix-free path is not the lever: it removes the sparse insertion and none of
the kernel work. Neither — measured — is a row and column restriction of the
kernel, which is the obvious reading of the paragraph above. Assembling only what
the indicator reads (`R_W = b⁺_W − A⁺[W,V] u_V` and `diag A⁺[W,W]`, as a
selection of matrix entries applied to the pattern, the gather plan and the
emission loop) halves the entries stored and reproduces `cells`, `total` and
`reference` to the bit — and is worth 1.15× at `-O0` and nothing at `-O2`
(0.96–1.05× over the three orders). The contraction count falls by roughly four
and the time does not: deciding that a (test, trial) pair is unwanted costs about
what contracting it costs, and the branch that decides it is what stops the
contraction vectorising.

What would pay is a cheaper question rather than a cheaper answer to this one.
Both quantities the indicator reads are right-hand-side shaped — `R` is
`ℓ(v) − a(u⁺, v)` integrated against the injected state, and `diag A⁺` is the
bilinear form evaluated at `a == b` alone — and neither needs a sparse pattern or
a gather plan. An rhs pass over `V⁺` that reads the state at every point costs
9.7 ms at `p` = 3 against the block pass's 249 ms. The price is that `R_W` would
then come out of quadrature instead of out of `b⁺ − A⁺u⁺`: the two agree to
3e-14 relative (measured on `‖R|_V‖ / ‖R|_W‖` over the three orders), so every
`η_K` would move in its last digits. That is why this file still forms the
operator, and it is the trade to make deliberately rather than by accident.

What the indicator assumes is a **coercive** bilinear form. The diagonal
Bank–Weiser indicator reads `A⁺_jj` as the energy of complement mode `j`, so it
needs `A⁺_jj > 0` for every active complement mode and `a(u_h, u_h) > 0`;
symmetry is not required, so convection–diffusion is fine and Helmholtz above the
first resonance is not. A non-positive complement diagonal raises rather than
being skipped: skipping it drops that direction from `η` silently, which is
exactly the regime where the indicator would otherwise report a plausible number
for a form it cannot measure. A `reference` that comes out at zero is *not*
refused, because on a deep, ill-conditioned stack `cᵀ A c` can legitimately lose
every digit; it is reported as zero and [`ErrorEstimate`](@ref)'s `show` prints
the ratio as `NaN`.

Forms must not be keyed by quadrature-point index, and this **refuses** one
that is. `estimate` evaluates the problem's own blocks and loads a second time,
on `V⁺`'s integration plan, whose quadrature clouds and `q.point` numbering
differ from the base model's — `V⁺` has more points per cell, and they sit at
different places. A form that reads per-point state through `q.point` (a
[`QuadField`](@ref), the documented mechanism for material history) would
therefore index an array built for `model` with `plus`'s indices: out of range
where the base plan is the smaller of the two, and at the wrong points where it
is not, which is the reading that produces a plausible number rather than an
error. A [`QuadField`](@ref) captured by any of the problem's forms is found
before the enriched model is built and raises. Where a model carries per-point
state, compute the indicator on a model whose forms read that state through the
physical point `q.x` instead.

What it refuses: single-field and single-domain, and every basis family that
does not carry a per-cell order (`_supports_cell_order`). The injection copies
coefficients by dof key and needs the same key to name the same function at both
orders; a B-spline's degree cannot be raised without rewriting its knot vector,
so the injection has no meaning there (see [`Rewire`](@ref)) and this raises
rather than returning a number that looks plausible. It also refuses a model that
has not been assembled, and an `enrichment` below one.

See [`refine`](@ref) for the rest of the loop.
"""
function estimate(model::Model{D,T}, solution::Solution; enrichment::Integer=1) where {D,T}
    _assert_single_domain(model, "estimate")
    enrichment >= 1 || throw(ArgumentError("enrichment must be at least 1; got $enrichment"))
    V = model.prefold_space
    for (k, level) in pairs(V.levels)
        # The trait, not the family. What the injection needs is that raising the
        # degree leaves every existing function's dof key naming the same
        # function, and `_supports_cell_order` is exactly the property that
        # states it — the knot-vector reasoning behind the B-spline answer lives
        # with the trait in `mesh.jl`. `Rewire`'s `_assert_keys_comparable` is
        # the backstop, but it fires only after the enriched model has been
        # prepared and assembled, which is the expensive half of this call.
        _supports_cell_order(level.basis) ||
            throw(ArgumentError("estimate: level $k carries $(basis_name(level.basis)), whose degree " *
                                "cannot be raised without renaming its functions, so injecting the " *
                                "solution into an order-elevated space has no meaning (see `Rewire`). " *
                                "Only families declaring `_supports_cell_order` are supported."))
    end
    model.matrix === nothing &&
        throw(ArgumentError("estimate: the model has not been assembled; call `solve!` or " *
                            "`assemble!` before estimating"))
    # Before the expensive half: the enriched twin below re-runs THESE forms on
    # a different quadrature cloud, and a form keyed by `q.point` reads the
    # wrong points there without raising. `data.jl` carries the search and the
    # reasoning; it runs once per call and never per point.
    _assert_no_point_keyed_state(model.problem, "estimate")

    # The caches are handed over UNcopied, which is the opposite of what every
    # other derivation does (`adapted` gives its target its own copy so a fork
    # cannot evict a sibling's rules). It is deliberate here: `plus` is thrown
    # away at the end of this call, and the (p+1)-order cut rules it fits are
    # wanted on `model`, where the next `adapted` copies them forward and the
    # next `estimate` finds them. `integration_plan` evicts by region box, so
    # the two moment orders coexist at the same boxes instead of evicting each
    # other — see `_prefit_cut_rules!` in `intersections.jl`.
    plus = _prepared_model(_problem_with_space(model.problem, _enriched_space(V, Int(enrichment))),
                           model.plan_options, model.moment_fit_caches)
    _on_foreign_cloud(plus.version) do
        assemble!(plus)
    end
    injected = transfer(solution, model, plus; via=Rewire())
    residual = plus.rhs .- plus.matrix * injected.coefficients

    base_keys = Set(_only_field_layout(model.dofs).dofs.raw_keys)
    plus_field = _only_field_layout(plus.dofs)
    plus_keys = plus_field.dofs.raw_keys
    enriched = plus.problem.space

    # Both per-raw questions, answered once each rather than once per incidence.
    # `W` membership is a hash of a `TensorDofKey`, and a raw dof is incident to
    # 2^D cells and repeated over every component, so asking it inside the sweep
    # below pays for the same lookup up to 2^D·n_components times. The diagonal
    # is extracted in one pass for the same reason — `plus.matrix[j, j]` is a
    # binary search down a sparse column.
    #
    # Together with the live-cell restriction below, that is the whole of what
    # separates this pass from one keyed by `Dict`/`Set` over every declared
    # cell: measured on the tanh ladder (6² base, depth 5, four threads, `-O0`),
    # the two sweeps fall from 9.3 / 13.4 / 19.4 ms to 3.2 / 6.5 / 12.8 ms at
    # `p` = 1 / 2 / 3, on identical output. That is a low single-digit percentage
    # of `estimate`, which the cost note above explains: the enriched assembly is
    # the term that matters, and this is what keeps the component loop from
    # multiplying a term that does not.
    in_W = BitVector(!(key in base_keys) for key in plus_keys)
    diagonal = diag(plus.matrix)

    # A dof is incident to several cells; its contribution is split equally among
    # them, so a cell is never charged for a neighbour's error. Raw ids are
    # contiguous, so this is a dense vector rather than a `Dict`.
    incidence = zeros(Int, length(plus_keys))
    for k in 1:length(enriched.levels), cell in _live_cells(enriched.levels[k])
        for raw in cell_dofs(plus.dofs, k, cell)
            incidence[raw] += 1
        end
    end

    # The attribution sweep. `counted` is indexed by ACTIVE id, not by raw:
    # active ids are unique across components, so a vector-valued field's
    # `(raw, component)` slots are each counted once into the V/W split while
    # every one of them is charged to its cell.
    cells = [zeros(T, level.mesh.cells) for level in enriched.levels]
    counted = falses(length(residual))
    residual_V = residual_W = zero(T)
    for k in 1:length(enriched.levels), cell in _live_cells(enriched.levels[k])
        raws = cell_dofs(plus.dofs, k, cell)
        for component in 1:plus_field.components
            for (raw, active) in zip(raws, active_cell_dofs(plus.dofs, k, cell, component))
                active == 0 && continue                   # constrained away
                if !counted[active]
                    counted[active] = true
                    in_W[raw] ? (residual_W += residual[active]^2) :
                    (residual_V += residual[active]^2)
                end
                in_W[raw] || continue
                d = diagonal[active]
                d > 0 ||
                    throw(ArgumentError("estimate: the enriched diagonal of a complement mode on " *
                                        "level $k, cell $cell, component $component is $d. The " *
                                        "Bank–Weiser indicator reads that diagonal as the mode's " *
                                        "energy, so it needs a coercive form; on this one the " *
                                        "indicator has no meaning."))
                cells[k][cell] += residual[active]^2 / d / incidence[raw]
            end
        end
    end
    for a in cells
        a .= sqrt.(a)
    end

    coefficients = _checked_coefficients(solution, model)
    reference = sqrt(max(zero(T), dot(coefficients, model.matrix * coefficients)))
    total = sqrt(sum(sum(a .^ 2) for a in cells))
    consistency = sqrt(residual_V / max(residual_W, eps(T)))
    return ErrorEstimate{D,T}(cells, total, reference, consistency)
end

"""
    mark_cells(estimate::ErrorEstimate; theta = 0.5) -> Vector{Tuple{Int,CartesianIndex{D}}}

Mark the cells carrying the bulk of the estimated error: the smallest set whose
squared indicators sum to at least `theta` of the total, taken in order of
decreasing η_K. This is Dörfler's bulk criterion (1996, see the References at
the top of `src/adaptivity.jl`), and it is the first half of the adaptive loop —
[`refine`](@ref)`(V, est; theta, pmax, previous)` is exactly

    h, p = decide(V, est, mark_cells(est; theta); pmax, previous)
    refine(V; h, p, pmax)

so a loop that wants the marking or the h/p split in its own hands reaches
them here rather than reimplementing either.

Returns `(level, cell)` pairs — the shape [`decide`](@ref) takes and the same
one `refine`'s explicit `h = ` / `p = ` form accepts. A cell whose indicator is
zero is never marked: it carries no approximation of its own, because it is
inactive or because a finer level has taken its region over.

`theta` must lie in `(0, 1]`; at `theta = 1` every cell with a positive
indicator is marked. It is the loop's only free constant, and the criterion was
chosen so it does not have to be refitted per problem: over a factor-three
sweep the resulting unknown count moves by at most 1.46×, against 1.86× for a
smoothness threshold and 1.36× for a benefit/cost rule whose informative window
is narrower still.

The marked set is extended through the group of equal indicators the `theta`
cut lands in. Dörfler asks for a set reaching *at least* `theta`, so this is
permitted, and it is what keeps a symmetric problem's refinement symmetric:
indicators related by a symmetry agree to about 1e-15 and never bit-identically,
so a strict prefix hands the decision to round-off and refines part of each
symmetry orbit.

The verb is `mark_cells` rather than `mark` because `Base` exports `mark` for
`IO` streams, and a second exported `mark` would make every unqualified use of
the name ambiguous for anyone writing `using Unfitted`.
"""
function mark_cells(estimate::ErrorEstimate{D,T}; theta::Real=0.5) where {D,T}
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
# Melenk & Wohlmuth's predicted error reduction (2001, see References), in the
# form deal.II's `hp::Refinement::predict_error` implements it. Each cycle
# predicts what a cell's indicator ought to become if the solution there is as
# smooth as the step just taken assumed; the next cycle compares. A cell that met
# its prediction is behaving smoothly and takes p again; one that fell short is
# not, and takes h.
#
#     no step taken     η_pred = ∞     (see below; Melenk & Wohlmuth write γ_n η_K)
#     p-step by Δp      η_pred = η_K γ_p^{Δp}
#     h-step            η_pred(K_c)² = n⁻¹ (η_{K_p} γ_h ρ^{p/D} γ_p^{Δp})²  per child
#     h-release         η_pred(K_p)² = Σ_c (η_{K_c} γ_p^{Δp})² / (γ_h ρ^{p/D})²
#
#     take p if η_K < η_pred, else take h
#
# with γ_p = √0.4 and γ_h = 2. Melenk & Wohlmuth state the constants for SQUARED
# indicators, γ_p² = 0.4 and γ_h² = 4; these are their square roots, which is
# what deal.II's `predict_error` ships and what the unsquared η here needs. Both
# are fixed rather than exposed: they are the literature's calibration of the
# prediction, not a knob the loop tunes, and `theta` is where a caller who wants
# one should reach.
#
# The p row runs in both directions: a *lowered* order makes Δp negative,
# γ_p^{Δp} exceeds one, and the cell is predicted to get worse by the reciprocal
# of what the degree was worth — which is what stops a `coarsen` p-release from
# being undone on sight. The two h rows are reciprocal in the same way.
#
# ρ is the VOLUME RATIO of the child cell to its parent, so ρ^{p/D} is the h-rate
# of the energy norm at the mesh size the two levels actually differ by, and `p`
# is the child's order AFTER the step. Melenk & Wohlmuth and deal.II write 0.5^p
# because their refinement bisects every axis, which this one does not have to:
# `ladder(...; splits=…)` takes any factor per axis and per level and `overlay`
# places any resolution at all. ρ^{p/D} reproduces 0.5^p on isotropic bisection,
# gives 0.25^p at `splits = 4`, and 2^{−p/2} on a (2, 1) split — where the
# unsplit axis contributes an unreduced error term and a child judged by 0.5^p
# would look like a failure of smoothness rather than of the split. Reading an
# anisotropic split through one volume-based mesh size is a judgement: the
# per-axis error terms do not reduce uniformly, and no scalar rate can say so.
# Under inherited order the children's ℓ²-aggregate prediction is
# η_parent·γ_h·ρ^{p/D}, i.e. η_parent·2^{1−p} on isotropic bisection in any
# dimension, and each of the n children carries an equal 1/√n share of it. The
# h-release row inverts that aggregate exactly, so releasing a cover whose
# children met their prediction predicts the parent's own indicator back.
#
# n is the number of children OF THE CHOSEN PARENT, counted from the geometry
# rather than from the ratio of the two levels' cell counts. The ratio is the
# child count only when both levels span the same box at an integer multiple —
# on a `ladder`. On a sub-box overlay at the same cell count it is 1 where each
# parent has 2^D children, and every child then looks √(2^D) better than it is.
# Where several parents overlap the child, the one with the largest η is taken:
# the child inherits the error of the cell that was actually worth refining, and
# on a nested stack there is only ever one. On a non-nested overlay, splitting a
# parent's error equally among cells that only partly overlap it is a heuristic
# and is stated as one.
#
# Three kinds of cell never reach the comparison:
#
#   - one with no h-step on offer takes p. That is the finest level, a level
#     below whose box does not reach the cell, and a cover that is already fully
#     live: in all three the step would activate nothing;
#   - one that is not a leaf, or one already at `pmax` in every axis, takes h.
#     The first would shed the very modes a p-step gives it; the second has no
#     order left to buy, and routing it to p is what turns a stalled loop into an
#     endless one — measured, a 20-cycle run froze at 494 unknowns with its base
#     mesh pinned at order 8 and its last three cycles identical. A cell with
#     NEITHER step on offer is dropped from the marking altogether, so that
#     `refine` can report exhaustion by returning its argument unchanged;
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

# The evidence one cycle leaves the next: the space that was refined, and the
# estimate that was read on it. The two travel together because neither means
# anything alone — a cell's history is the difference between the two spaces,
# scored by the indicator the earlier one carried.
const _History{D,T} = Tuple{Space{D,T},ErrorEstimate{D,T}}

# The h-rate ρ^{q/D} between two cells of consecutive levels, from the boxes
# themselves rather than from an assumed bisection: ρ is the volume ratio, so
# ρ^{1/D} is the mesh size the h^p law of the energy norm refers to. See the
# note above for why a scalar rate on an anisotropic split is a judgement.
function _h_rate(fine::AxisBox, coarse::AxisBox, q::Int, ::Val{D}) where {D}
    return (volume(fine) / volume(coarse))^(q / D)
end

# The indicator this cell was predicted to have by now, read off what happened to
# it between `previous` and `V`. Nothing is stored between cycles: the previous
# space and its estimate say what was done, because a p-step shows up as a raised
# order, an h-step as a cell that was not active before, and an h-release as a
# cover that was live before and is not now.
#
# Every question here is per cell and has an O(1) accessor: `cell_order` reads
# one palette entry, `is_active` one mask bit, `_cell_block` resolves the stack
# by binary search. None of them is memoised, and none should be — a cache keyed
# by level materialises `cell_orders(…; level=k)`, a dense `Array{NTuple{D,Int},D}`
# over the whole level, to answer a question about the handful of cells a cycle
# marks. On the tanh ladder's finest level — 192² cells — that is a measured
# 589 872 bytes per copy to read a few dozen entries.
function _predicted(V::Space{D,T}, previous::_History{D,T}, k::Integer,
                    cell::CartesianIndex{D}) where {D,T}
    Vprev, eprev = previous
    k <= length(Vprev.levels) || return T(Inf)
    q = minimum(cell_order(V.levels[k], cell))

    if !is_active(Vprev.levels[k].mask, cell)
        # Newly activated, so it is the child of an h-step one level up. The
        # parent is the overlapping cell of level k−1 that carried the most
        # error; on a nested stack there is exactly one.
        k == 1 && return T(Inf)
        η = zero(T)
        parent = nothing
        for par in _cell_block(Vprev, k, k - 1, cell)
            eprev.cells[k - 1][par] > η || continue
            η, parent = eprev.cells[k - 1][par], par
        end
        parent === nothing && return T(Inf)              # no parent carried any error
        n = length(_cell_block(Vprev, k - 1, k, parent))
        rate = _h_rate(cell_box(V, cell; level=k), cell_box(Vprev, parent; level=k - 1), q, Val(D))
        return η *
               T(_GAMMA_H) *
               T(rate) *
               T(_GAMMA_P)^(q - minimum(cell_order(Vprev.levels[k - 1], parent))) /
               sqrt(T(max(n, 1)))
    end

    # The h-coarsening row, which is read before the cell's own history because
    # its evidence is the children's error, not the parent's. `decide`
    # reaches this only for a leaf of `V`, so a cover that was live in `Vprev`
    # and is not now is exactly an h-release, and the cell is predicted to get
    # worse by the reciprocal of what the subdivision was worth — the children's
    # error, ℓ²-combined and scaled to this cell's current order, undone by the
    # rate the h-step was credited with. Without it a released parent falls
    # through the equality test below, takes the no-evidence row and buys p on
    # sight, which is the oscillation the release was trying to avoid; the
    # p-release row exists for the same reason.
    if k < length(V.levels) && k < length(Vprev.levels)
        block = _cell_block(Vprev, k, k + 1, cell)
        released = zero(T)
        for ch in block
            (is_active(Vprev.levels[k + 1].mask, ch) && !is_active(V.levels[k + 1].mask, ch)) ||
                continue
            released += (eprev.cells[k + 1][ch] *
                         T(_GAMMA_P)^(q - minimum(cell_order(Vprev.levels[k + 1], ch))))^2
        end
        if released > 0
            rate = _h_rate(cell_box(Vprev, first(block); level=k + 1),
                           cell_box(Vprev, cell; level=k), q, Val(D))
            return sqrt(released) / (T(_GAMMA_H) * T(rate))
        end
    end

    # Equal orders mean nothing was applied here, so there is no evidence and the
    # caller takes p. A LOWERED order is evidence, and of the opposite sign: the
    # exponent goes negative, γ_p^(negative) exceeds one, and the cell is
    # predicted to get WORSE by the reciprocal of what the degree was worth. That
    # is Melenk & Wohlmuth's coarsening row for p, and without it a p-released
    # cell falls through the equality branch and takes p again on sight.
    η = eprev.cells[k][cell]
    η > 0 || return T(Inf)
    q_prev = minimum(cell_order(Vprev.levels[k], cell))
    q == q_prev && return T(Inf)
    return η * T(_GAMMA_P)^(q - q_prev)
end

# A `(level, cell)` mark against the space it names. Both halves are checked
# here, because the alternative is a `BoundsError` raised from inside the
# prediction or the order field, naming an array the caller never handed over.
function _check_mark(V::Space{D}, k::Integer, cell::CartesianIndex{D},
                     verb::AbstractString) where {D}
    k = _check_level(V, k, "$verb: marked level")
    checkbounds(Bool, CartesianIndices(V.levels[k].mesh.cells), cell) ||
        throw(ArgumentError("$verb: marked cell $cell is outside level $k's cell grid " *
                            "$(V.levels[k].mesh.cells)"))
    return k
end

# An `ErrorEstimate` is indexed by the cell indices of the space it is read
# against, from the first line of `decide` onward. A mismatched one is
# otherwise a `BoundsError` from deep inside the prediction or — where two
# levels happen to agree in shape — a silently misattributed score, so both
# arguments are checked once, here, against the space each belongs to.
function _check_estimate(V::Space{D,T}, est::ErrorEstimate{D,T}, name::AbstractString) where {D,T}
    shapes = [level.mesh.cells for level in V.levels]
    map(size, est.cells) == shapes ||
        throw(DimensionMismatch("$name holds indicators shaped $(map(size, est.cells)), but this " *
                                "space's levels have cells $shapes"))
    return est
end

"""
    decide(V::Space, estimate::ErrorEstimate, marked; pmax = 8, previous = nothing) -> (h, p)

Decide, for each marked cell, whether to spend an h-step or a p-step on it, and
return the two sets as vectors of `(level, cell)` pairs — the second half of the
adaptive loop, and exactly what [`refine`](@ref)`(V; h, p, pmax)` applies. `marked`
is what [`mark_cells`](@ref) returns, or any iterable of `(level, cell)` pairs of
your own.

A marked cell takes **p** when its indicator met the reduction its last step
predicted and **h** when it fell short, which is why `previous = (V_last,
est_last)` — the space and estimate of the cycle before — is what makes the
decision a decision. Without it every cell takes p, falling back to h only where
p is unavailable: right for the first cycle and wrong thereafter. The prediction
is Melenk & Wohlmuth's, with their γ_p and γ_h; the long note above `_GAMMA_P`
in `src/adaptivity.jl` gives it in full, including why an h-released cover and a
lowered order are read as evidence of the opposite sign.

Three kinds of marked cell never reach the comparison: one with no h-step
available takes p; one that is not a leaf, or that is already at `pmax` on every
axis, takes h; and one with *neither* step available is dropped from both sets,
which is what lets `refine` report exhaustion by returning its argument
unchanged. `pmax` bounds the p climb and never lowers an order.

Both `estimate` and `previous`'s estimate are checked against the space they are
read on, since an indicator is indexed by that space's cells: a shape mismatch
raises `DimensionMismatch` rather than misattributing a score, and a marked
level or cell outside `V` raises `ArgumentError`.
"""
function decide(V::Space{D,T}, estimate::ErrorEstimate{D,T}, marked; pmax::Integer=8,
                previous::Union{Nothing,_History{D,T}}=nothing) where {D,T}
    _check_estimate(V, estimate, "estimate")
    h = Tuple{Int,CartesianIndex{D}}[]
    p = Tuple{Int,CartesianIndex{D}}[]
    live = _live_levels(V)
    if previous !== nothing
        Vprev, eprev = previous
        _check_estimate(Vprev, eprev, "previous")
        all(k -> V.levels[k].mesh.cells == Vprev.levels[k].mesh.cells,
            1:min(length(V.levels), length(Vprev.levels))) ||
            throw(DimensionMismatch("previous: a shared level's cell grid differs from this " *
                                    "space's, so a cell's history cannot be read at its own index"))
    end

    for (k, cell) in marked
        _check_mark(V, k, cell, "refine")
        # What is on offer here, then which of the two to spend. A cell with
        # neither step available is dropped rather than parked in `p`: `refine`
        # would return a new but identical space, and a loop made of those does
        # not terminate.
        h_offered = _has_h_step(V, k, cell, live)
        p_offered = !all(>=(Int(pmax)), cell_order(V.levels[k], cell)) && _is_leaf(V, k, cell, live)
        h_offered || p_offered || continue
        take_p = p_offered && (!h_offered ||
                               previous === nothing ||
                               estimate.cells[k][cell] < _predicted(V, previous, k, cell))
        push!(take_p ? p : h, (k, cell))
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
cell whether to spend on h or on p, and apply. Those two halves are
[`mark_cells`](@ref) and [`decide`](@ref), and this form is their composition,

    refine(V, est; theta, pmax, previous) ==
        refine(V; decide(V, est, mark_cells(est; theta); pmax, previous)..., pmax)

so a driver that needs the marked set or the h/p split itself — to report them,
or to keep [`coarsen`](@ref) from releasing what this cycle just refined — calls
the two directly and hands the result to the second form rather than
reimplementing either. The second form takes the two sets explicitly as
iterables of `(level, cell)` pairs, so an indicator of your own plugs into the
same application step.

The two steps, and what each one means here:

  - the **h-step** activates the cells of level `k + 1` that cover the marked
    cell of level `k`, mapped through the same geometric overlap
    [`overlapping_cells`](@ref) reports — refining "this cell" means activating
    its cover one level down — and gives the cells it wakes their parent's order
    unchanged. A cell that is already live keeps the order it earned, so this is
    a pure h-step: the resolution changes and the order does not. The target is
    always level `k + 1`; where that level has nothing under the cell — the
    finest level, or a level whose box does not reach it, as on a sub-box overlay
    — there is nothing to activate and the cell takes the order step instead;
  - the **p-step** raises the marked cell's own order by one and activates
    nothing. `pmax` bounds the climb so a loop cannot run away: it raises only
    axes below `pmax` and leaves a cell already at or above it alone, so passing
    a `pmax` under a cell's current order means "no more p here" rather than
    "lower it". The step is only ever handed a leaf by the loop form, which is
    what makes it survive: leaf semantics shed a covered cell's bubble modes, so
    the same increment on a covered cell would be a no-op. Measured before that
    was understood, 356 cells raised while covered retained 576 active modes
    between them and **zero** bubbles.

Existing activation is preserved rather than replaced. [`adapt`](@ref) *sets* a
level's mask, so an application that forgets to keep what is already live
silently un-refines the rest of the domain; this keeps it.

On an immersed model — one whose space carries a [`PhysicalDomain`](@ref) —
there is no state transfer across an h-step: [`L2Projection`](@ref) refuses a
target carrying a physical domain, and [`Rewire`](@ref) refuses a target whose
active basis does not contain the source's, which is what leaf semantics make of
a refinement. An adaptive loop on an immersed model therefore re-solves on the
refined space rather than carrying its iterate across, and `estimate` needs that
fresh solve before it can be called again.

`refine` returns `V` **itself** when nothing changed — every marked cell was
already at `pmax`, or on the finest level, or its cover was already live — so
`refined === V` is the loop's exhaustion signal and is worth testing before
paying for another rebuild. Any change at all returns a new `Space`.

What the h-step presumes about the stack. It is a refinement rather than a
deletion only because leaf semantics shed the parent's bubble modes and the
activated children replace them, and they replace them only where level `k + 1`
*nests* over the marked cell. A [`ladder`](@ref) guarantees that by
construction; [`move!`](@ref) can void it and nothing re-checks, so
[`is_nested`](@ref) and `diagnostics(…).levels[k].nested` are the witnesses on a
hand-built stack. Neither form checks it here, because the geometric test has
false alarms — it reports `false` on a pair that sheds nothing — and refusing a
mark on one would block a legitimate refinement. On a non-nested pair the
overlap rule still decides which cells move: a child straddling two parents is
activated for whichever of them is marked, and released when either is.

Which step a marked cell takes is decided from its own refinement history. Pass
`previous = (V_last, estimate_last)` — the space and estimate from the cycle
before — and each marked cell takes p when its indicator met the reduction its
last step predicted, h when it fell short. The two spaces must agree in cell
counts on the levels they share, since a cell's history is read at its own
index; a ladder may grow levels between cycles. Omit `previous` and every marked
cell takes p, falling back to h only where p is unavailable; that is the right
behaviour for the first cycle and the wrong one thereafter. See the note above
[`decide`](@ref) for the prediction and why it is preferred
here to a smoothness indicator. The prediction's own constants — Melenk &
Wohlmuth's γ_p and γ_h — are fixed rather than exposed, because they are the
literature's calibration of the rule and not a knob to fit per problem; `theta`
is the loop's only tunable.

See [`estimate`](@ref) for the rest of the loop.
"""
function refine(V::Space{D,T}, estimate::ErrorEstimate{D,T}; theta::Real=0.5, pmax::Integer=8,
                previous::Union{Nothing,_History{D,T}}=nothing) where {D,T}
    h, p = decide(V, estimate, mark_cells(estimate; theta=theta); pmax=pmax, previous=previous)
    return refine(V; h=h, p=p, pmax=pmax)
end

function refine(V::Space{D,T}; h=(), p=(), pmax::Integer=8) where {D,T}
    nlevels = length(V.levels)
    pmax >= 1 || throw(ArgumentError("pmax must be at least 1; got $pmax"))

    # One field and one mask per level, built against `V` and merged, because a
    # level can be written from both sides: it holds the p-step of its own cells
    # and the inherited order of the children some coarser cell just activated.
    # `elevate` rejects a level named twice, so merging is not optional.
    masks = Dict{Int,Any}()
    fields = Dict{Int,Any}()
    mask_of(k) = get!(() -> active_cells(V; level=k), masks, k)
    field_of(k) = get!(() -> cell_orders(V; level=k), fields, k)
    # Raise only the axes below the cap. A `pmax` under a cell's current order is
    # a "no more p here" instruction, not a request to lower an order `refine`
    # was asked to enrich, and the loop's saturation test reads it the same way.
    raised(o) = min.(o .+ 1, max.(o, Int(pmax)))

    for (k, cell) in p
        _check_mark(V, k, cell, "refine")
        field = field_of(k)
        field[cell] = raised(field[cell])
    end

    for (k, cell) in h
        _check_mark(V, k, cell, "refine")
        # Level k+1 is the target, and where it has no cell under this one — the
        # finest level, or a level whose box does not reach the cell — there is
        # nothing to activate and the order step is what is left to spend. This
        # is where the substitution happens; `_has_h_step` is where the loop form
        # DECIDES that h is unavailable, and it sends such a cell to `p` so this
        # branch is unreachable from `refine(V, estimate)`. The two are not a
        # duplicated rule: a decision that h is unavailable and an application
        # that converts an impossible h are different jobs, and the explicit form
        # — which takes the caller's own sets — needs the second on its own.
        block = k < nlevels ? _cell_block(V, k, k + 1, cell) : nothing
        if block === nothing || isempty(block)
            field = field_of(k)
            field[cell] = raised(field[cell])
            continue
        end
        # The children take the parent's order as it stands, uncapped: `coarsen`
        # restores exactly this value on release, and capping here would break
        # that inverse on any stack whose order already exceeds `pmax`. It is read
        # from `V` rather than from `field_of(k)`, which the p-step above may
        # already have raised — an h-step hands down the order the parent had when
        # the cycle started, never one bought in the same cycle.
        target = cell_order(V.levels[k], cell)
        mask = mask_of(k + 1)
        field = field_of(k + 1)
        for child in block
            # A child that is not live carries whatever order it held in a
            # previous life, and `coarsen` clears that on release — but only for
            # the covers it releases. Writing the parent's order over it is what
            # keeps a stale entry from being resurrected: measured, one h-step
            # onto a released order-6 cover returned 169 unknowns where a cold
            # h-step gives 57. A child that IS live is left exactly as it is —
            # its order was earned on its own evidence, and silently raising it
            # would make this an order step wearing an h-step's name.
            mask[child] || (field[child] = target)
            mask[child] = true
        end
    end

    # Identity is the exhaustion signal, so a step that wrote nothing must return
    # the space it was given rather than a fresh copy of it. Each touched level is
    # compared against the one it was built from; the arrays are the ones the
    # merge already allocated, so this costs one more read of each.
    filter!(kv -> kv.second != active_cells(V; level=kv.first), masks)
    filter!(kv -> kv.second != cell_orders(V; level=kv.first), fields)
    isempty(masks) && isempty(fields) && return V

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
    `pmin`: it lowers only axes above `pmin` and leaves a cell already at or
    below it alone, so a `pmin` above a cell's current order means "no more
    release here" rather than "raise it".

Releasing a cover whose children are themselves covered is a no-op rather than
an error: the region is carried further down and the deeper level is what holds
it, so releasing this level alone would leave a hole. Release the stack from the
top down and each call is well defined. So is an h-mark on a level the one below
does not reach, which has no cover to release. As in [`refine`](@ref), a call
that changes nothing returns `V` itself.

Lowering the order of a cell a finer level has taken over is **rejected**, and
*any* finer level counts, not only the one immediately below — the dof layer's
covering rule tests them all, so a cell held by level `k + 2` while `k + 1` is
dormant is just as covered. Under leaf semantics a covered cell has already shed
its high-order modes, so the change is invisible while the cover is live and
appears without warning when it lifts — measured, 173.5× the energy error, with
`diagnostics`, `isposdef`, the residual norm and `estimate`'s consistency flag
all unchanged to the last digit. There is no diagnostic that catches it, so it is
refused here instead.

The h-mark presumes, as `refine`'s h-step does, that the level below nests over
the marked cell; on a non-nested pair a child straddling two parents is released
when either of them is marked. A release *is* evidence to the next cycle's
decision: `refine`'s prediction scores a cell whose cover was just released by
the reciprocal of what the subdivision was worth, so a released region is not
re-refined on sight.

An [`ErrorEstimate`](@ref) is indexed by the cells of the space it was taken on,
and is zero on every cell that carried no approximation there. A release rule
that reads one must therefore be evaluated on that space, *before* [`refine`](@ref)
is applied: a cell the same cycle activates scores zero in the earlier estimate
and reads as cold, so releasing on the refined space undoes the h-step that just
woke it. Exclude, too, any parent the same cycle marks for h — its cover was
woken by `refine` and this verb would put it straight back to sleep. Measured on
`examples/reproductions/traveling_laser_2d`, a driver that released on the
refined space undid 607 of the 639 h-steps it took, 95 %.

On an immersed model — one whose space carries a [`PhysicalDomain`](@ref) —
there is no state transfer across this step: [`L2Projection`](@ref) refuses a
target carrying a physical domain, and [`Rewire`](@ref) refuses a target whose
basis does not contain the source's, which releasing a cover is. A transient
that coarsens an immersed model therefore re-solves on the released space
rather than carrying its iterate across.

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

    masks = Dict{Int,Any}()
    fields = Dict{Int,Any}()
    mask_of(k) = get!(() -> active_cells(V; level=k), masks, k)
    field_of(k) = get!(() -> cell_orders(V; level=k), fields, k)
    live = _live_levels(V)
    # Lower only the axes above the floor, the mirror of `refine`'s `raised`.
    lowered(o) = max.(o .- 1, min.(o, Int(pmin)))

    for (k, cell) in p
        _check_mark(V, k, cell, "coarsen")
        _is_leaf(V, k, cell, live) ||
            throw(ArgumentError("coarsen: cell $cell of level $k is covered by a finer level, " *
                                "and lowering a covered cell's order is not observable until the " *
                                "cover lifts. Release the cover first."))
        field = field_of(k)
        field[cell] = lowered(field[cell])
    end

    for (k, cell) in h
        _check_mark(V, k, cell, "coarsen")
        k < nlevels ||
            throw(ArgumentError("coarsen: level $k is the finest, so it covers nothing to release"))
        block = _cell_block(V, k, k + 1, cell)
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
        for child in block
            mask[child] || continue
            # A child that is itself covered — by any deeper level, not only the
            # next one — is not this level's to release.
            _is_leaf(V, k + 1, child, live) || continue
            mask[child] = false
            field[child] = parent_order
        end
    end

    # Identity when nothing moved, as in `refine`: a release of a cover that was
    # not there, or a p-mark already at `pmin`, returns the space it was given.
    filter!(kv -> kv.second != active_cells(V; level=kv.first), masks)
    filter!(kv -> kv.second != cell_orders(V; level=kv.first), fields)
    isempty(masks) && isempty(fields) && return V

    released = isempty(masks) ? V :
               adapt(V, (k => masks[k] for k in sort!(collect(keys(masks))))...)
    isempty(fields) && return released
    return elevate(released, (k => fields[k] for k in sort!(collect(keys(fields))))...)
end
