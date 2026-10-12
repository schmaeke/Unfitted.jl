# ── Automated hp adaptivity ───────────────────────────────────────────────────
#
# Three verbs over one loop: estimate where the error is, mark the cells worth spending on, refine them. Everything else
# the loop needs — rebuilding the model, carrying the solution across — is already in the package, so this file adds no
# driver and owns no state. The user keeps the loop, as they keep the solve loop.
#
# THE RULE. Dörfler on the Bank–Weiser indicator picks the cells worth spending on; each marked cell is then sent to h
# or to p by whether its LAST refinement delivered the error reduction that step was entitled to expect. A cell that met
# its prediction is behaving smoothly and takes p again; one that fell short is not, and takes h. The predictions are
# Melenk & Wohlmuth's (2001, see References) — see the note above `decide`.
#
# WHY NOT A SMOOTHNESS INDICATOR. Mitchell & McClain (2014, see References) compared thirteen hp strategies over twenty
# problems and rate Legendre coefficient decay the best general-purpose choice — "COEFDECAY appears to be the best
# choice as a general strategy across all categories of problems". It was implemented here, exactly as the literature
# specifies (a true Legendre spectrum obtained by an exact change of basis from the integrated-Legendre coefficients,
# bands grouped by ‖k‖₁, p-refinement where σ > 1 under |a_k| ~ c exp(−σ‖k‖₁)), and it was removed, because it does not
# survive contact with THIS loop's operating point.
#
# The reason is structural rather than a bug, and it is worth recording so nobody reimplements it. σ > 1 is an
# asymptotic statement about the tail of an expansion, and Mavriplis (1994, see References) derived it for spectral
# elements at p ≈ 8–16. A cell at order 1–3 has two to four bands, and the slope through them measures the ratio of the
# linear coefficient to the cell mean — a fact about the local amplitude of the solution, not about its analyticity.
# This loop starts every cell at order 1 and keeps most of them low, so the indicator is read exactly where it has
# nothing to say: measured on the SMOOTH λ = 5 tanh front, where almost every cell should take p, only 45% of leaves
# cleared σ > 1, the median σ was 0.87 and the lower quartile 0. Prediction has no such floor. It works from the first
# refinement at any order, because it asks what the last step actually bought rather than what the coefficients look
# like.
#
# WHAT IT IS WORTH, against step-27 of deal.II 9.5 (Arndt et al. 2023, see References) on the same tanh front, at
# matched unknowns and on the same mesh-independent lattice measure, over three regimes of the layer steepness λ.
# step-27 is the right comparison precisely because it is the other family: its h/p decision is a
# Fourier-coefficient-decay smoothness estimate (`SmoothnessEstimator::Fourier::coefficient_decay` feeding
# `hp::Refinement::choose_p_over_h`), so these rows measure prediction against a smoothness indicator on the same
# problem and the same marking.
#
#              this rule        step-27        no decision (p until pmax)
#     λ =   5   3.2e-07         6e-07           8.7e-09
#     λ =  30   2.0e-05         1.2e-05         —
#     λ = 120   3.3e-04         8.2e-04         4.2e-03
#
# So: better than the reference where the solution is hard, within 1.7× where it is middling, and beaten only by having
# no decision at all on the easiest case, which is the one where a decision has nothing to earn. The decision-free rule
# is the trap worth naming — it is optimal on a smooth solution and it is NOT a decision, because it discovers that a
# cell wanted h only after paying for every degree up to the cap; measured against step-27 it won by 23× at λ = 5 and
# lost by 3.6× and 6.1× at λ = 30 and 120, the penalty growing with exactly the non-smoothness a decision exists to
# detect.
#
# The two steps stay INDEPENDENT, which is the one thing that looks redundant and is not. It is tempting to fold them
# together — let the ladder carry an increasing order schedule, so deepening the stack over a cell is simultaneously an
# h-step and a p-step, and the whole application collapses to incrementing one integer. Measured, that costs a
# consistent ~1.5×: on a two-feature problem — u = |x − (0.3, 0.3)|^1.5 plus a Gaussian bump of width 0.08 at (0.7, 0.7)
# on the unit square, against uniform order 4 — coupling reaches a relative energy error of 2.6e-4 at 5239 unknowns, an
# 11.63× saving, where the independent increments reach it at 3521, a 17.22× saving. Both figures are one point of a
# curve and only comparable at a stated error: the same independent loop reads 4.41× at 1e-3. The reason for the gap is
# that coupling cannot express *pure* p — reaching order 7 needs four levels of depth, dragging 16× finer cells along
# with it. It is the same failure the inert p-step produced: an order schedule welded to depth cannot put a high degree
# on a large cell. Keeping the steps apart is what lets the rule above spend p first and h only when p has run out.
#
# ── References ────────────────────────────────────────────────────────────────
#
# The four published algorithms this file implements, and the reference implementation its numbers are measured against.
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

# ── The error estimate ────────────────────────────────────────────────────────

"""
    ErrorEstimate{D,T}

What [`estimate`](@ref) returns: a Bank–Weiser error indicator per cell, with the scalars a stopping test needs.

  - `cells::Vector{Array{T,D}}` — `cells[k][c]` is `η_K` for cell `c` of level `k`, and zero on cells that carry no
    approximation there.
  - `total::T` — `√Σ_K η²_K`, the global indicator.
  - `reference::T` — `√(cᵀ A c)` over the *active* unknowns, in the problem's own bilinear form: the energy of the part
    of `u_h` the unknowns determine. Under homogeneous Dirichlet data that is `√a(u_h, u_h)`; with nonzero data it is
    not, because the constrained columns are folded into the right-hand side and the lift's own energy is never formed,
    so what is measured is the interior part of the solution and the quantity is not invariant under adding a constant
    to the Dirichlet data. Measured on an 8² order-2 Poisson problem with `u = sin πx sin πy`: adding `C` to the
    boundary data leaves `total` and the discrete energy error bit-identical at 1.8257e-2 while `reference` runs 2.2214,
    6.8407 and 53.137 at `C` = 0, 1 and 10, moving `total / reference` by 24× with nothing about the approximation
    changed. It is still scale-free and still what a tolerance should be compared against on one problem; it is not a
    quantity to compare across different boundary data. The problem supplies the norm, so this stays meaningful when the
    operator is not the Laplacian. A `reference` of exactly zero means the discrete energy came out below round-off, and
    `show` prints the ratio as `NaN` rather than dividing.
  - `consistency::T` — `‖R|_V‖ / ‖R|_W‖`, the residual the injected solution leaves on `V`'s own dofs against what it
    leaves on the enrichment. It is NOT a saturation check. `V`'s functions and integration regions are identical inside
    `V⁺`, so the numerator can only differ from the converged residual through what the order change moved: the
    quadrature the load and the coefficients are integrated with, the Dirichlet trace projected at order `q + 1` instead
    of `q`, and the solver residual. It is therefore a free data-oscillation and solve check, and it says nothing about
    whether `W` captures the missing content: a solution whose error happens to be orthogonal to the enrichment gives a
    small `consistency` and a small `η` at once. Measured on a 4² order-2 space, changing nothing but the data — the
    non-polynomial `sin πx sin πy` replaced by `x² + y²`, which both rules integrate exactly and whose trace `V`
    reproduces — drops it from 7.1e-4 to 5.2e-7. A value that climbs means the data is under-integrated where `η` is
    smallest. It is a pure ratio, so scaling the data leaves it unchanged; it is `Inf` when the residual on `W` is
    exactly zero while the one on `V` is not, and `0` when both are, as on zero data.

    For an actual saturation signal there is one the loop already has, under two conditions: on a refine-only sequence
    with homogeneous Dirichlet data, Galerkin orthogonality makes `est_new.reference² − est_prev.reference²` the true
    reduction in squared energy error, so a previous `total²` well below that gain says the previous estimate
    under-read. Both conditions are load bearing — a [`coarsen`](@ref) breaks the nesting, and a nonzero lift breaks the
    identification of `reference` with `√a(u_h, u_h)` above.

Read the fields; you never construct one. Same posture as [`AssemblyDiagnostics`](@ref).
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

"""
    estimate(model::Model, solution::Solution; enrichment = 1) -> ErrorEstimate

Bank–Weiser error indicator: build an enriched space `V⁺ ⊇ V`, inject `solution` into it, and read the residual in the
directions `V` cannot represent.

    η²_K = Σ_{j ∈ W(K)} R_j² / A⁺_jj ,      R = b⁺ − A⁺ u⁺ ,      W = V⁺ ⊖ V

**How `V⁺` is built depends on the family, and it has to.** A family carrying a per-cell order (`_supports_cell_order`,
which integrated Legendre declares) is enriched in place: every cell's order rises by `enrichment`, and the injection is
exact because a hierarchic mode keeps its identity under an order increase. A family that does not — a B-spline level,
whose degree belongs to a whole-axis knot vector — is enriched by **superposition** instead: a co-located level of the
same family, degree and active region on a mesh refined by `enrichment + 1` per axis. That renames nothing, so the
injection is again exact, and the two spline spaces nest because the refinement factor is an integer, so `V⁺ ⊇ V` holds.
The twin's per-cell indicators are charged back to the cells of the level it enriches, so the estimate a caller sees is
always indexed by `V`'s own levels and is fed straight to [`mark_cells`](@ref) and [`refine`](@ref) either way.

Only a live level is twinned: a dormant one has nothing to enrich. The level a twin enriches is built with leaf
semantics off *for that level alone*, and the reason is in `estimate`'s body: a nested same-degree cover would otherwise
deduplicate the very keys the injection looks for. Every other level keeps leaf semantics, pruned by the same rule as
in `V`, which is what keeps the indicator a function of the model rather than of which levels happen to be present. The
rule's outcome differs from `V`'s in one place: the twins sit above every level of `V`, so where a twin is live it also
covers the cells of any higher level. A level there that carries a per-cell order then sheds modes `V` keeps free, `V⁺`
no longer contains `V`, and that stack is refused (below).

`W` is a set difference on raw dof keys, not a comparison of mode indices: at an anisotropic order the two disagree, and
the index form selects 5 of the 11 complement modes at order `(3, 5)`. Each complement mode is shared equally among the
cells it is incident to, so no cell is charged for a neighbour's error; the sharing is always well defined because a
mode is only shed when *every* incident cell is covered. Every component of the field is charged to its own cell: on a
field declared with `components = n` the sum above runs over all `n` components of each complement mode, so `η_K`
measures the whole vector-valued residual.

`enrichment` is the order increment defining `V⁺`. One is the usual choice and the only one measured here.

The indicator's quality is what justifies its cost: effectivity drifts 1.23× over a 64× range in unknowns, against
52.57× for a gradient-recovery indicator on the same problems.

The cost is worth stating, because `estimate` is the loop's largest single term and stays so at every order. Kernel work
per cell is modes² × quadrature points and each factor carries a `(p+1)^D`, so raising every cell by one degree
multiplies the assembly by about `((p+2)/(p+1))^(3D)` — in 2D, 11.4 / 5.6 / 3.8 at `p` = 1 / 2 / 3. Measured on the
ladder of `examples/reproductions/adaptive_tanh_layer_2d` (6² base, depth 5, a nested band of active cells along the
front; four threads, `-O0`, two runs), `assemble!(V⁺)` against `assemble!(V)` comes out at 7.5× / 4.9–5.4× / 3.5×: the
prediction from `p` = 2 up, and short of it at `p` = 1, where per-region work that does not scale with the mode count is
still a large share of an 18 ms assembly. `estimate` is then 78–82% of `prepare + assemble + solve + estimate` at all
three orders, worst at the low orders the loop starts from. Inside it the split MOVES with the order — `assemble!(V⁺)`
runs from about half of `estimate` at `p` = 1 to about 84% at `p` = 3, and `prepare(V⁺)` from about a fifth down to 3% —
so no fixed split is worth quoting. The injection and the residual matvec are nowhere in it, at 1.3–7.6 ms and 0.5–4.4
ms across the same three orders.

None of those shares is all of the assembly. `plus` is a fresh model on every call, so it also pays the symbolic
region-dof lists and pattern in full and amortises them over a single assembly — the opposite of the Newton or transient
loop those caches were built for. Symbolic against numeric inside `assemble!(V⁺)`, same ladder and four threads: 63 /
135 / 439 ms against 103 / 556 / 1796 ms at `-O0`, and 78 / 120 / 253 ms against 22 / 74 / 249 ms at the default `-O2`,
at `p` = 1 / 2 / 3. So read `-O0` figures as `-O0` figures: `estimate` is 3–5× faster at `-O2` and `assemble!(V)` 6–7×,
because the kernel is the part that shrinks — at `-O2` the symbolic half is the larger of the two at `p` = 1 and 2 and
level with the numeric at `p` = 3.

A matrix-free path is not the lever: it removes the sparse insertion and none of the kernel work. Neither — measured —
is a row and column restriction of the kernel, which is the obvious reading of the paragraph above. Assembling only what
the indicator reads (`R_W = b⁺_W − A⁺[W,V] u_V` and `diag A⁺[W,W]`, as a selection of matrix entries applied to the
symbolic data and the emission loop) halves the entries stored and reproduces `cells`, `total` and `reference` to the
bit — and is worth 1.15× at `-O0` and nothing at `-O2` (0.96–1.05× over the three orders). The contraction count falls
by roughly four and the time does not: deciding that a (test, trial) pair is unwanted costs about what contracting it
costs, and the branch that decides it is what stops the contraction vectorising.

What would pay is a cheaper question rather than a cheaper answer to this one. Both quantities the indicator reads are
right-hand-side shaped — `R` is `ℓ(v) − a(u⁺, v)` integrated against the injected state, and `diag A⁺` is the bilinear
form evaluated at `a == b` alone — and neither needs a sparse pattern. An rhs pass over `V⁺` that reads the state at
every point costs 9.7 ms at `p` = 3 against the block pass's 249 ms. The price is that `R_W` would then come out of
quadrature instead of out of `b⁺ − A⁺u⁺`: the two agree to 3e-14 relative (measured on `‖R|_V‖ / ‖R|_W‖` over the three
orders), so every `η_K` would move in its last digits. That is why this file still forms the operator, and it is the
trade to make deliberately rather than by accident.

What the indicator assumes is a **coercive** bilinear form. The diagonal Bank–Weiser indicator reads `A⁺_jj` as the
energy of complement mode `j`, so it needs `A⁺_jj > 0` for every active complement mode and `a(u_h, u_h) > 0`; symmetry
is not required, so convection–diffusion is fine and Helmholtz above the first resonance is not. A non-positive
complement diagonal raises rather than being skipped: skipping it drops that direction from `η` silently, which is
exactly the regime where the indicator would otherwise report a plausible number for a form it cannot measure. A
`reference` that comes out at zero is *not* refused, because on a deep, ill-conditioned stack `cᵀ A c` can legitimately
lose every digit; it is reported as zero and [`ErrorEstimate`](@ref)'s `show` prints the ratio as `NaN`.

Forms must not be keyed by quadrature-point index, and this **refuses** one that is. `estimate` evaluates the problem's
own blocks and loads a second time, on `V⁺`'s integration plan, whose quadrature clouds and `q.point` numbering differ
from the base model's — `V⁺` has more points per cell, and they sit at different places. A form that reads per-point
state through `q.point` (a [`QuadField`](@ref), the documented mechanism for material history) would therefore index an
array built for `model` with `plus`'s indices: out of range where the base plan is the smaller of the two, and at the
wrong points where it is not, which is the reading that produces a plausible number rather than an error. A
[`QuadField`](@ref) captured by any of the problem's forms is found before the enriched model is built and raises. A
read the forms make without capturing the state — through a global — raises at the read instead, as does a read of
any array by `q.point`, a plain one with a `BoundsError`: `V⁺` numbers its points far below 1, so no such read can
land on a point of `model`'s cloud. That numbering is `V⁺`'s own and sets nothing process-wide, so a `q.point` read in
any other task, one in an `estimate` of another model running at the same time included, is unaffected. Where a model
carries per-point state, compute the indicator on a model whose forms read that state through the physical point `q.x`
instead.

What it refuses, before anything enriched is built and in messages that name `estimate`: a multi-domain or multi-field
problem (the one field may carry any number of components), a model that has not been assembled, a `solution` that is
not one of `model` as it stands (computed on another discretisation, or before a mutation), and an `enrichment` below
one. It does not refuse a basis family without a per-cell order — that family is enriched by superposition instead, as
described above — but it does refuse, after preparing `V⁺` and before assembling it, an enriched space that does not
contain `V`. The injection copies coefficients by dof key, so a key that is free in `V` and not in `V⁺` has nowhere to
go. That happens where a twin, which is placed above every level of `V`, covers a higher level that carries a per-cell
order: estimate a mixed stack with its B-spline levels above its integrated-Legendre ones (a Legendre base under a
B-spline overlay estimates, the reverse is refused).

See [`refine`](@ref) for the rest of the loop.
"""
function estimate(model::Model{D,T}, solution::Solution; enrichment::Integer=1) where {D,T}
    # Every refusal comes before `V⁺` exists, cheapest first, so a refused call costs neither the enriched preparation
    # nor its assembly, and names `estimate` itself. The second field and the stale solution used to be found only after
    # both: the field by the attribution sweep's single-field lookup, whose message did not name the verb, and the pin
    # by the injection, after the enriched load had been evaluated 512 and 256 times on a 4² order-2 Poisson problem.
    _assert_single_domain(model, "estimate")
    enrichment >= 1 || throw(ArgumentError("estimate: enrichment must be at least 1; got $enrichment"))
    V = model.prefold_space
    model.matrix === nothing && throw(ArgumentError("estimate: the model has not been assembled; call `solve!` or " *
                                                    "`assemble!` before estimating"))
    length(model.dofs.fields) == 1 ||
        throw(ArgumentError("estimate: the problem has $(length(model.dofs.fields)) fields; the indicator is defined " *
                            "for a single field, which may carry any number of components"))
    coefficients = try
        _checked_coefficients(solution, model)
    catch err
        err isa ArgumentError ? throw(ArgumentError("estimate: " * err.msg)) : rethrow()
    end
    # The enriched model V⁺ below re-runs THESE forms on a different quadrature cloud, where a form keyed by `q.point`
    # has no point of its own to read. V⁺'s numbering makes such a read raise at the read (below); this walk refuses a
    # captured `QuadField` before anything is built, and names it. `data.jl` carries the search and the reasoning; it
    # runs once per call and never per point.
    _assert_no_point_keyed_state(model.problem, "estimate")

    # Without a live twin, `V⁺` is built on `model`'s caches UNcopied, which is the opposite of what every other
    # derivation does (`adapted` gives its target its own copy so a fork cannot evict a sibling's rules). Sharing is
    # deliberate where `V⁺` cuts `model`'s own boxes: `plus` is thrown away at the end of this call, and the
    # (p+1)-order cut rules it fits are wanted on `model`, where the next `adapted` copies them forward and the next
    # `estimate` finds them. `integration_plan` evicts by region box, so the two moment orders coexist at the same boxes
    # instead of evicting each other — see `_prefit!` in `intersections.jl`. A live twin breaks that premise, and
    # changes the pruning too, so `_twinned_model` builds `V⁺` then. `twinned` holds the ids of the levels a twin is
    # charged to, the key `dof_layout` reads its pruning exemption by.
    #
    # `V⁺` numbers its points from `_FOREIGN_POINTS`, far below 1, so a form that reads per-point state by `q.point`
    # without capturing it — through a global, which the walk above cannot see — raises at the read rather than reading
    # `model`'s points at the twin's indices (see "Detecting point-keyed state" in `data.jl`).
    enriched_space, charge = _enriched_space(V, Int(enrichment))
    twinned = Int[V.levels[k].id for (k, factor) in charge if factor > 1]
    plus = if isempty(twinned)
        _derived(model, enriched_space; caches=_fits(model), point_base=_FOREIGN_POINTS)
    else
        _twinned_model(model, enriched_space, twinned)
    end

    # Inject before assembling. `Rewire` reads only the two dof layouts, so it needs nothing assembled, and after the
    # refusals above it has one way left to fail: a key of `V` that `V⁺` does not carry free, i.e. `V⁺ ⊉ V`. The known
    # cause is a twin covering a level it should not. A twin is appended above every level of `V`, so it covers a
    # higher level carrying a per-cell order, which in `V⁺` then sheds modes that are free in `V`. The structural fix —
    # a twin that never covers an original level — belongs to the coverage rule, so here the call is refused under its
    # own name, with the supported ordering and `Rewire`'s text, and before the enriched assembly rather than after it.
    injected = try
        transfer(solution, model, plus; via=Rewire())
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("estimate: the enriched space V⁺ does not contain V, so the solution cannot be injected " *
                            "into it. A level enriched by superposition (one without a per-cell order, such as a " *
                            "B-spline level) gets a refined twin placed above every level of V; where the twin is " *
                            "live it covers the cells of any higher level that carries a per-cell order, and that " *
                            "level then sheds modes in V⁺ that are free in V. Place such levels above the " *
                            "per-cell-order ones. The injection reported: " *
                            err.msg))
    end
    assemble!(plus)
    residual = plus.rhs .- plus.matrix * injected.coefficients

    base_keys = Set(_only_field_layout(model.dofs).dofs.raw_keys)
    plus_field = _only_field_layout(plus.dofs)
    plus_keys = plus_field.dofs.raw_keys
    enriched = plus.problem.space

    # Both per-raw questions, answered once each rather than once per incidence. `W` membership is a hash of a
    # `TensorDofKey`, and a raw dof is incident to 2^D cells and repeated over every component, so asking it inside the
    # sweep below pays for the same lookup up to 2^D·n_components times. The diagonal is extracted in one pass for the
    # same reason — `plus.matrix[j, j]` is a binary search down a sparse column.
    #
    # Together with the live-cell restriction below, that is the whole of what separates this pass from one keyed by
    # `Dict`/`Set` over every declared cell: measured on the tanh ladder (6² base, depth 5, four threads, `-O0`), the
    # two sweeps fall from 9.3 / 13.4 / 19.4 ms to 3.2 / 6.5 / 12.8 ms at `p` = 1 / 2 / 3, on identical output. That is
    # a low single-digit percentage of `estimate`, which the cost note above explains: the enriched assembly is the term
    # that matters, and this is what keeps the component loop from multiplying a term that does not.
    in_W = BitVector(!(key in base_keys) for key in plus_keys)
    diagonal = diag(plus.matrix)

    # A dof is incident to several cells; its contribution is split equally among them, so a cell is never charged for a
    # neighbour's error. Raw ids are contiguous, so this is a dense vector rather than a `Dict`.
    #
    # Both sweeps address a level's cell tables by its level *id*, as `cell_dofs` is keyed; the attribution sweep keeps
    # the position `j` for `charge`, which follows `enriched.levels`. The two coincide on a single-domain space, the
    # only kind `estimate` accepts, but only the id is the table's key.
    #
    # Each level's id and live cells are read once, into the typed vectors `ids` and `live_cells`, and the sweeps read
    # nothing else off a level. That is what keeps the per-cell loops free of dynamic dispatch. `V⁺` is built at run
    # time, so `enriched` is not concretely typed here and neither is anything read off one of its levels. A cell or an
    # id taken from there was `Any`, and every `cell_dofs`, `_coarse_cell` and `cells[k][…] +=` below then dispatched
    # at run time: about 12 allocations per live cell, 70 % of all that `estimate` allocated on a 32² order-2 Poisson
    # problem.
    ids = Int[level.id for level in enriched.levels]
    live_cells = _LiveCells{D}[_live_cells(level) for level in enriched.levels]
    incidence = zeros(Int, length(plus_keys))
    for j in eachindex(ids), cell in live_cells[j], raw in cell_dofs(plus_field.dofs, ids[j], cell)
        incidence[raw] += 1
    end

    # The attribution sweep, each level charged where `charge` says. A twin carries the complement modes of the level it
    # enriches, so its indicators belong to that level's cells, each twin cell to the coarse cell it subdivides: the
    # twins exist only inside this call, and a marked pair naming one could not be applied. A cell's terms are summed
    # locally in (component, raw) order and charged once, which is bit for bit the attribution this replaced, one dense
    # array per level of `V⁺` folded onto the parents cell by cell afterwards: the same terms in the same order, every
    # level of `V` complete before any twin is added, and a dormant twin cell, which that fold also visited, adding an
    # exact 0.0.
    #
    # `counted` is indexed by ACTIVE id, not by raw: active ids are unique across components, so a vector-valued field's
    # `(raw, component)` slots are each counted once into the V/W split while every one of them is charged to its cell.
    # A slot's active id is read straight from the field's table, so the sweep allocates nothing per (cell, component).
    cells = Array{T,D}[zeros(T, level.mesh.cells) for level in V.levels]
    counted = falses(length(residual))
    residual_V = residual_W = zero(T)
    for j in eachindex(ids)
        k, factor = charge[j]
        for cell in live_cells[j]
            η² = zero(T)
            raws = cell_dofs(plus_field.dofs, ids[j], cell)
            for component in 1:plus_field.components, raw in raws
                active = _field_component_dof(plus_field, raw, component)
                active == 0 && continue                   # constrained away
                if !counted[active]
                    counted[active] = true
                    in_W[raw] ? (residual_W += residual[active]^2) : (residual_V += residual[active]^2)
                end
                in_W[raw] || continue
                d = diagonal[active]
                d > 0 || throw(ArgumentError("estimate: the enriched diagonal of a complement mode on " *
                                             "level $j, cell $cell, component $component is $d. The " *
                                             "Bank–Weiser indicator reads that diagonal as the mode's " *
                                             "energy, so it needs a coercive form; on this one the " *
                                             "indicator has no meaning."))
                η² += residual[active]^2 / d / incidence[raw]
            end
            cells[k][_coarse_cell(cell, factor)] += η²
        end
    end
    for a in cells
        a .= sqrt.(a)
    end

    reference = sqrt(max(zero(T), dot(coefficients, model.matrix * coefficients)))
    total = sqrt(sum(sum(a .^ 2) for a in cells))
    # A pure ratio, so it is as scale-free as the residual it reads. Flooring the denominator at an absolute
    # `eps(T)`, as this once did, made it a function of the data's units: on a 4² order-2 Poisson problem it read
    # 7.06e-4 with the source scaled by 1 or 1e-4, but 3.54e-5 at 1e-8 and 3.54e-7 at 1e-10, for the same relative
    # residual. The ratio itself gives `Inf` where nothing is left on `W` while `V` carries a residual, and NaN where a
    # residual is NaN, as it is in the total. The one case it leaves undefined, nothing left on either side (zero data),
    # is perfectly consistent.
    consistency = iszero(residual_V) && iszero(residual_W) ? zero(T) : sqrt(residual_V / residual_W)
    return ErrorEstimate{D,T}(cells, total, reference, consistency)
end

# The enrichment `estimate` reads, as a record: `(W, charge)`, where `charge[j] = (k, f)` says that level `j` of `W` is
# charged to level `k` of `V`, each of its cells to the cell of `k` it lies in when `k` is refined `f`-fold per axis.
# Every level of `V` is its own `(j, 1)`, raised by `inc` in place where its family carries a per-cell order. Every live
# level whose family does not gets a refinement twin (`_refinement_twin`), appended above all of them and charged
# `(k, inc + 1)` to the level it enriches. A dormant level gets none: every cell of its twin would be dormant too, so
# the twin would carry no raw and no integration region — the partition follows live cells — and add nothing to the
# indicator but a level to walk.
#
# A level is raised by rebuilding it on its own order record with every palette entry raised by `inc`, the class map
# and the mesh unchanged. That is the record `elevate(V, k => p .+ inc)` would build from the raised per-cell field: an
# increment keeps distinct palette entries distinct and in first-occurrence order, keeps an isotropic entry isotropic
# and an order ≥ 1 at least 1, so the palette, the class map and every check the build would run come out the same —
# and no per-cell field is materialised on the way. A uniform level stays a palette of one. (`inc ≥ 1`: `estimate`
# refuses a smaller enrichment.)
#
# Each level is rebuilt directly, by `_relevel`, rather than through `elevate`. The only other thing `elevate` adds is
# the check that no level is named twice, and this loop names each level once by construction. Calling it instead meant
# splatting a generator whose length is known only at run time into a `Vararg` call, which inference resolves by
# iterating the generator abstractly, with closures of its own specialised on the generator's type and compiled on the
# spot. Measured on an adapted 8² order-2 ladder, that was 129 ms of inference and 127 ms of the closures' codegen,
# nearly a fifth of the first `estimate`'s compile time.
function _enriched_space(V::Space{D,T}, inc::Int) where {D,T}
    W, charge, live = V, [(k, 1) for k in 1:length(V.levels)], _live_levels(V)
    for k in 1:length(V.levels)
        _supports_cell_order(V.levels[k].basis) || continue
        o = V.levels[k].orders
        W = _relevel(W, k; field=CellOrders{D}(map(p -> p .+ inc, o.palette), o.class, o.nominal .+ inc))
    end
    for k in 1:length(V.levels)
        (live[k] && !_supports_cell_order(V.levels[k].basis)) || continue
        W = _refinement_twin(W, V.levels[k], inc + 1)
        push!(charge, (k, inc + 1))
    end
    return W, charge
end

# A level co-located with `level` — same box, same family, same degree, same active region — on a mesh refined by
# `factor` per axis. This is the enrichment for a family whose degree cannot be raised in place, and the reason it is a
# *superposition* rather than an elevation is the same reason `_supports_cell_order` is false for such a family: raising
# a B-spline degree rewrites the whole-axis knot vector, so every function is renamed and the injection that `estimate`
# needs has nothing to copy onto. Refining the mesh renames nothing that already exists; it only adds.
#
# `factor` is an integer, so the twin's knot lines contain the level's and the two spline spaces nest — which is what
# makes `V⁺ ⊇ V` and the injection exact. The mask is mapped child-to-parent so the twin is live exactly where the level
# is, and its own selection rule then trims it to the functions that fit, exactly as it does for any level. `repeat`
# with `inner` is that map: it writes each parent bit over its `factor`^D children, as `_coarse_cell` reads it back. The
# mode is passed as the level's own, because `overlay` applies its `mode` keyword to the family it is given and
# defaults it to the base level's mode, which an overlay's twin need not share.
function _refinement_twin(W::Space{D,T}, level::Level{D,T}, factor::Int) where {D,T}
    mask = level.mask === nothing ? nothing : repeat(level.mask; inner=ntuple(_ -> factor, D))
    return overlay(W, level.mesh.domain; cells=level.mesh.cells .* factor, order=nominal_order(level),
                   basis=level.basis, mode=_mode(level.basis), active=mask)
end

# The cell of a level that cell `c` of its `f`-fold refinement lies in, and `c` itself at `f = 1`: where `estimate`
# charges a twin's indicator.
_coarse_cell(c::CartesianIndex{D}, f::Int) where {D} = CartesianIndex(ntuple(d -> (c.I[d] - 1) ÷ f + 1, D))

# `estimate`'s enriched model when `space` holds a live refinement twin, charged to the levels whose ids are `twinned`.
# Two things differ from the plain enrichment, and both come from the twin.
#
# Pruning. A superposed twin is a *nested* level of the same degree, so leaf semantics would deduplicate the level it
# enriches — removing exactly the keys the injection is about to look for, and turning `Rewire` into a refusal. Only
# *that* level may skip pruning, and only in this throwaway model: turning leaf semantics off for the whole enriched
# model instead was measured to inflate the indicator on a covered cell of an unrelated integrated-Legendre level by
# 4.1×, because that level's order-elevated covered modes then survive, land in `W`, and are charged as error to a cell
# `decide` can spend nothing on. Exempting the enriched level costs the indicator nothing. `u⁺` is zero on every key the
# injection does not fill, so the duplicate columns contribute nothing to `R = b⁺ − A⁺u⁺`; `b⁺` and the diagonal on `W`
# are untouched; and `A⁺` being singular is harmless here, since the indicator reads its diagonal and forms one
# matrix–vector product and never factorises.
#
# The caches. A twin refines the integration partition wherever it is live, so the enriched plan cuts none of `model`'s
# boxes there, and building it on `model`'s own caches, which a plan restricts to the boxes it cuts, evicted every rule
# `model` held: on an immersed 8² order-2 B-spline ladder the next `adapted` refitted all 44 of its boxes. So the plan
# is built on a copy, and the rules fitted there are merged back into `model`'s caches, where both sets coexist as the
# two moment orders of the plain enrichment do. The twin's rules are worth keeping: on a ladder split by the twin's
# factor its mesh is the next level's, so its boxes are the ones the next h-step wakes. Over two estimate → refine →
# adapted cycles on an immersed 8² order-2 B-spline ladder of depth 2, the merge took the fits from 309 (shared) to 236,
# every `adapted` refitting nothing; a copy alone took them to 326, because the next `adapted` then refitted the twin's
# boxes. The price is that `model`'s caches carry one twin plan's boxes beyond its own until its next plan build evicts
# them.
#
# It is a function rather than a branch of `estimate` for compile time: `space` is built at run time and is not
# concretely typed there, so this call is dispatched at run time, and the copy and the merge are code-generated only
# when a twin is first estimated, never for a stack without one. Inference still reads its body, for the abstract
# signature, when `estimate` itself is first compiled, whatever the stack.
function _twinned_model(model::Model, space::Space, twinned::Vector{Int})
    caches = copy.(_fits(model))
    plus = _derived(model, space; options=merge(model.plan_options, (; prune_exempt=twinned)), caches,
                    point_base=_FOREIGN_POINTS)
    foreach(merge!, _fits(model), caches)
    return plus
end

# The cells of one level that carry an approximation. `cell_dofs` answers with an empty vector on a dormant cell, so
# restricting the estimator's two sweeps to these changes no result — only how much of a level has to be walked to find
# that out. That is the point on a ladder, where a level holds N^D cells however few are switched on: a depth-5 3D chain
# over a 4³ base declares 2 097 152 of them on its finest level. The mask scan is one pass over a `BitArray`, cheap
# against a dof lookup per declared cell: `cell_dofs` allocates nothing, since it returns the layout's own vector, but
# it is still a table read per cell, and on such a level almost every one of them comes back empty. `estimate` builds
# the list once per level and walks it in both sweeps.
#
# The list has one of two shapes, the lazy grid of an unmasked level or the vector of a masked one, and `_LiveCells`
# names the pair. `estimate` holds its lists in a vector of that `Union`, so a cell drawn from one is a concrete
# `CartesianIndex` and the per-cell loop splits on the two shapes, where the alternative, one shape for both, would
# materialise every cell of an unmasked level.
const _LiveCells{D} = Union{CartesianIndices{D,NTuple{D,Base.OneTo{Int}}},Vector{CartesianIndex{D}}}

_live_cells(level::Level) = level.mask === nothing ? cell_indices(level.mesh) : CartesianIndices(level.mask)[level.mask]

# ── Dörfler marking ───────────────────────────────────────────────────────────

"""
    mark_cells(estimate::ErrorEstimate; theta = 0.5) -> Vector{Tuple{Int,CartesianIndex{D}}}

Mark the cells carrying the bulk of the estimated error: the smallest set whose squared indicators sum to at least
`theta` of the total, taken in order of decreasing η_K. This is Dörfler's bulk criterion (1996, see the References at
the top of `src/adaptivity.jl`), and it is the first half of the adaptive loop —
[`refine`](@ref)`(V, est; theta, pmax, previous)` is exactly

    h, p = decide(V, est, mark_cells(est; theta); pmax, previous)
    refine(V; h, p, pmax)

so a loop that wants the marking or the h/p split in its own hands reaches them here rather than reimplementing either.

Returns `(level, cell)` pairs — the shape [`decide`](@ref) takes and the same one `refine`'s explicit `h = ` / `p = `
form accepts. A cell whose indicator is zero is never marked: it carries no approximation of its own, because it is
inactive or because a finer level has taken its region over.

`theta` must lie in `(0, 1]`; at `theta = 1` every cell with a positive indicator is marked, however small that
indicator is against the total. It is the loop's only free constant, and the criterion was chosen so it does not have
to be refitted per problem: over a factor-three sweep the resulting unknown count moves by at most 1.46×, against 1.86×
for a smoothness threshold and 1.36× for a benefit/cost rule whose informative window is narrower still.

The marked set is extended through the group of equal indicators the `theta` cut lands in. Dörfler asks for a set
reaching *at least* `theta`, so this is permitted, and it is what keeps a symmetric problem's refinement symmetric:
indicators related by a symmetry agree to about 1e-15 and never bit-identically, so a strict prefix hands the decision
to round-off and refines part of each symmetry orbit.

The verb is `mark_cells` rather than `mark` because `Base` exports `mark` for `IO` streams, and a second exported `mark`
would make every unqualified use of the name ambiguous for anyone writing `using Unfitted`.
"""
function mark_cells(estimate::ErrorEstimate{D,T}; theta::Real=0.5) where {D,T}
    0 < theta <= 1 || throw(ArgumentError("theta must lie in (0, 1]; got $theta"))
    # Every cell carrying an indicator, scanned level by level, then sorted by decreasing η. The sort is stable, so
    # cells of equal η keep the scan order and the marking is a function of the estimate alone. The cut is the first
    # prefix whose squared indicators reach `theta` of the total, summed left to right in that order; a sum that
    # round-off leaves short of the goal marks every cell.
    #
    # Left to right is `accumulate` over a generator, `Iterators.accumulate`'s documented fold: each partial sum is the
    # previous one plus the next term. `cumsum` is no substitute. On floating-point data it re-associates the partial
    # sums for accuracy, as v₁ + (v₂ + … + vᵢ) and pairwise in blocks on long inputs, so they differ from the running
    # sum in the last bit, and a goal that falls between the two moves the cut by a cell.
    η((k, cell)) = estimate.cells[k][cell]
    marked = sort!(Tuple{Int,CartesianIndex{D}}[(k, cell) for (k, a) in pairs(estimate.cells)
                                                for cell in CartesianIndices(a) if a[cell] > 0]; by=kc -> -η(kc))
    # At θ = 1 the set is every positive cell by definition, so the prefix is not consulted. It would lose a cell whose
    # η² lies below the round-off of the running sum (η under about 1e-8 of the total): that term adds nothing, the sum
    # can reach total² one cell early, and the group extension, read against the cut value, never reaches so small an
    # indicator either.
    (isempty(marked) || theta == 1) && return marked
    taken = something(findfirst(>=(theta * estimate.total^2), accumulate(+, η(kc)^2 for kc in marked)), length(marked))

    # Extend through the group the cut lands in. Dörfler asks for a MINIMAL set reaching `theta`, and any minimal set
    # satisfies it, so this is a free choice — but resolving it by strict comparison hands the decision to round-off,
    # and on a symmetric problem that is the whole decision. Cells related by a symmetry of the problem carry indicators
    # equal only to about 1e-15 relative, never bit-identical, because assembly and the solve are not associative. The
    # sort then orders them arbitrarily and the prefix keeps some of each orbit: measured on a disc with 4-fold
    # symmetry, the four equal-indicator cells at the diagonals split two-and-two, and the mesh refined only the
    # upper-left and lower-right quadrants for ten cycles. The marked counts gave it away — 11, 5 and 7 cells out of an
    # orbit structure that cannot produce an odd count.
    #
    # Completing the group marks slightly more than `theta`, which Dörfler permits (it asks for at least `theta`), and
    # it buys back the symmetry of the problem exactly. The group is read against the cut value itself, never against
    # the previous member, so it cannot creep down a slow ramp of indicators.
    cut = η(marked[taken])
    while taken < length(marked) && isapprox(η(marked[taken + 1]), cut; rtol=sqrt(eps(T)))
        taken += 1
    end
    return marked[1:taken]
end

# ── The h-versus-p decision ───────────────────────────────────────────────────
#
# Melenk & Wohlmuth's predicted error reduction (2001, see References), in the form deal.II's
# `hp::Refinement::predict_error` implements it. Each cycle predicts what a cell's indicator ought to become if the
# solution there is as smooth as the step just taken assumed; the next cycle compares. A cell that met its prediction is
# behaving smoothly and takes p again; one that fell short is not, and takes h.
#
#     no step taken     η_pred = ∞     (see below; Melenk & Wohlmuth write γ_n η_K)
#     p-step by Δp      η_pred = η_K γ_p^{Δp}
#     h-step            η_pred(K_c)² = n⁻¹ (η_{K_p} γ_h ρ^{p/D} γ_p^{Δp})²  per child
#     h-release         η_pred(K_p)² = Σ_c (η_{K_c} γ_p^{Δp})² / (γ_h ρ^{p/D})²
#
#     take p if η_K < η_pred, else take h
#
# with γ_p = √0.4 and γ_h = 2. Melenk & Wohlmuth state the constants for SQUARED indicators, γ_p² = 0.4 and γ_h² = 4;
# these are their square roots, which is what deal.II's `predict_error` ships and what the unsquared η here needs. Both
# are fixed rather than exposed: they are the literature's calibration of the prediction, not a knob the loop tunes, and
# `theta` is where a caller who wants one should reach.
#
# The p row runs in both directions: a *lowered* order makes Δp negative, γ_p^{Δp} exceeds one, and the cell is
# predicted to get worse by the reciprocal of what the degree was worth — which is what stops a `coarsen` p-release from
# being undone on sight. The two h rows are reciprocal in the same way.
#
# ρ is the VOLUME RATIO of the child cell to its parent, so ρ^{p/D} is the h-rate of the energy norm at the mesh size
# the two levels actually differ by, and `p` is the child's order AFTER the step. Melenk & Wohlmuth and deal.II write
# 0.5^p because their refinement bisects every axis, which this one does not have to: `ladder(...; splits=…)` takes any
# factor per axis and per level and `overlay` places any resolution at all. ρ^{p/D} reproduces 0.5^p on isotropic
# bisection, gives 0.25^p at `splits = 4`, and 2^{−p/2} on a (2, 1) split — where the unsplit axis contributes an
# unreduced error term and a child judged by 0.5^p would look like a failure of smoothness rather than of the split.
# Reading an anisotropic split through one volume-based mesh size is a judgement: the per-axis error terms do not reduce
# uniformly, and no scalar rate can say so. Under inherited order the children's ℓ²-aggregate prediction is
# η_parent·γ_h·ρ^{p/D}, i.e. η_parent·2^{1−p} on isotropic bisection in any dimension, and each of the n children
# carries an equal 1/√n share of it. The h-release row inverts that aggregate exactly, so releasing a cover whose
# children met their prediction predicts the parent's own indicator back.
#
# n is the number of children OF THE CHOSEN PARENT, counted from the geometry rather than from the ratio of the two
# levels' cell counts. The ratio is the child count only when both levels span the same box at an integer multiple — on
# a `ladder`. On a sub-box overlay at the same cell count it is 1 where each parent has 2^D children, and every child
# then looks √(2^D) better than it is. Where several parents overlap the child, the one with the largest η is taken: the
# child inherits the error of the cell that was actually worth refining, and on a nested stack there is only ever one.
# On a non-nested overlay, splitting a parent's error equally among cells that only partly overlap it is a heuristic and
# is stated as one.
#
# Three kinds of cell never reach the comparison:
#
#   - one with no h-step on offer takes p. That is the finest level, a level below whose box does not reach the cell,
#     and a cover that is already fully live: in all three the step would activate nothing;
#   - one that is not a leaf, or one already at `pmax` in every axis, takes h. The first would shed the very modes a
#     p-step gives it; the second has no order left to buy, and routing it to p is what turns a stalled loop into an
#     endless one — measured, a 20-cycle run froze at 494 unknowns with its base mesh pinned at order 8 and its last
#     three cycles identical. A cell with NEITHER step on offer is dropped from the marking altogether, so that `refine`
#     can report exhaustion by returning its argument unchanged;
#   - one with NO EVIDENCE takes p: the first cycle, a predecessor that carried no error, or — the common case — a cell
#     that was not marked last cycle and so had no step applied to it. That last exclusion is not a detail. Melenk &
#     Wohlmuth's table has a γ_n row for cells left alone, and deal.II ships γ_n = 1, i.e. "expect the indicator to
#     stand still"; but in THIS loop every marked cell receives a step, so that row can only ever fire for a cell about
#     which the previous cycle learned nothing. Testing it anyway turns the rule into "did your error happen to tick
#     down", which is a coin flip, and the coin decides most of the decisions: measured on the smooth λ = 5 front it
#     split 127 h against 85 p on a solution that wants p almost everywhere, and cost four orders of magnitude against
#     taking p unconditionally. No evidence is not evidence of roughness, and p is the cheap mistake — a wasted degree
#     costs one cycle, a wasted subdivision costs 2^D cells that never go away.
const _GAMMA_P = sqrt(0.4)
const _GAMMA_H = 2.0

# The evidence one cycle leaves the next: the space that was refined, and the estimate that was read on it. The two
# travel together because neither means anything alone — a cell's history is the difference between the two spaces,
# scored by the indicator the earlier one carried.
const _History{D,T} = Tuple{Space{D,T},ErrorEstimate{D,T}}

"""
    decide(V::Space, estimate::ErrorEstimate, marked; pmax = 8, previous = nothing) -> (h, p)

Decide, for each marked cell, whether to spend an h-step or a p-step on it, and return the two sets as vectors of
`(level, cell)` pairs — the second half of the adaptive loop, and exactly what [`refine`](@ref)`(V; h, p, pmax)`
applies. `marked` is what [`mark_cells`](@ref) returns, or any iterable of `(level, cell)` pairs of your own.

A marked cell takes **p** when its indicator met the reduction its last step predicted and **h** when it fell short,
which is why `previous = (V_last, est_last)` — the space and estimate of the cycle before — is what makes the decision a
decision. Without it every cell takes p, falling back to h only where p is unavailable: right for the first cycle and
wrong thereafter. The prediction is Melenk & Wohlmuth's, with their γ_p and γ_h; the long note above `_GAMMA_P` in
`src/adaptivity.jl` gives it in full, including why an h-released cover and a lowered order are read as evidence of the
opposite sign.

Three kinds of marked cell never reach the comparison: one with no h-step available takes p — unless its level carries
no per-cell order, where there is no p-step either and the cell is dropped rather than parked; one that is not a leaf,
or that is already at `pmax` on every axis, takes h; and one with *neither* step available is dropped from both sets,
which is what lets `refine` report exhaustion by returning its argument unchanged. `pmax` bounds the p climb and never
lowers an order.

Both `estimate` and `previous`'s estimate are checked against the space they are read on, since an indicator is indexed
by that space's cells: a shape mismatch raises `DimensionMismatch` rather than misattributing a score, and a marked
level or cell outside `V` raises `ArgumentError`, as does a `pmax` below one.
"""
function decide(V::Space{D,T}, estimate::ErrorEstimate{D,T}, marked; pmax::Integer=8,
                previous::Union{Nothing,_History{D,T}}=nothing) where {D,T}
    # The bound `refine` refuses below one is refused here too. Accepted, `pmax = 0` reads every cell as already at its
    # cap and quietly sends it to h, a split `refine` would refuse to apply under the same bound.
    pmax >= 1 || throw(ArgumentError("decide: pmax must be at least 1; got $pmax"))
    _check_estimate(V, estimate, "estimate")
    h = Tuple{Int,CartesianIndex{D}}[]
    p = Tuple{Int,CartesianIndex{D}}[]
    live = _live_levels(V)
    if previous !== nothing
        Vprev, eprev = previous
        _check_estimate(Vprev, eprev, "previous")
        all(k -> V.levels[k].mesh.cells == Vprev.levels[k].mesh.cells, 1:min(length(V.levels), length(Vprev.levels))) ||
            throw(DimensionMismatch("previous: a shared level's cell grid differs from this " *
                                    "space's, so a cell's history cannot be read at its own index"))
    end

    for (k, cell) in marked
        _check_mark(V, k, cell, "decide")
        # What is on offer here, then which of the two to spend. A cell with neither step available is dropped rather
        # than parked in `p`: `refine` would return a new but identical space, and a loop made of those does not
        # terminate.
        h_offered = _has_h_step(V, k, cell, live)
        # A p-step raises one cell's order, so it exists only where the family carries a per-cell order at all. On a
        # level that does not — a B-spline, whose degree belongs to a whole-axis knot vector — every marked cell takes
        # h, which is the adaptive loop the isogeometric literature runs. Raising such a level's degree is still
        # available, as a modelling decision rather than a per-cell one: superpose a level of higher degree over the
        # region with [`overlay`](@ref), which is linearly independent of the level below by construction, since a
        # degree-`p` spline that vanishes outside a region would have to be `C^p` there to lie in the degree-`p+1`
        # space, hence a polynomial, hence zero.
        p_offered = _supports_cell_order(V.levels[k].basis) &&
                    !all(>=(Int(pmax)), cell_order(V.levels[k], cell)) &&
                    _is_leaf(V, k, cell, live)
        h_offered || p_offered || continue
        take_p = p_offered &&
                 (!h_offered || previous === nothing || estimate.cells[k][cell] < _predicted(V, previous, k, cell))
        push!(take_p ? p : h, (k, cell))
    end
    return h, p
end

# An `ErrorEstimate` is indexed by the cell indices of the space it is read against, from the first line of `decide`
# onward. A mismatched one is otherwise a `BoundsError` from deep inside the prediction or — where two levels happen to
# agree in shape — a silently misattributed score, so both arguments are checked once, here, against the space each
# belongs to.
function _check_estimate(V::Space{D,T}, est::ErrorEstimate{D,T}, name::AbstractString) where {D,T}
    shapes = [level.mesh.cells for level in V.levels]
    map(size, est.cells) == shapes ||
        throw(DimensionMismatch("$name holds indicators shaped $(map(size, est.cells)), but this " *
                                "space's levels have cells $shapes"))
    return est
end

# A `(level, cell)` mark against the space it names. Both halves are checked here, because the alternative is a
# `BoundsError` raised from inside the prediction or the order field, naming an array the caller never handed over.
function _check_mark(V::Space{D}, k::Integer, cell::CartesianIndex{D}, verb::AbstractString) where {D}
    k = _check_level(V, k, "$verb: marked level")
    checkbounds(Bool, CartesianIndices(V.levels[k].mesh.cells), cell) ||
        throw(ArgumentError("$verb: marked cell $cell is outside level $k's cell grid " * "$(V.levels[k].mesh.cells)"))
    return k
end

# The indicator this cell was predicted to have by now, read off what happened to it between `previous` and `V`. Nothing
# is stored between cycles: the previous space and its estimate say what was done, because a p-step shows up as a raised
# order, an h-step as a cell that was not active before, and an h-release as a cover that was live before and is not
# now.
#
# Every question here is per cell and has an O(1) accessor: `cell_order` reads one palette entry, `is_active` one mask
# bit, `_cell_block` resolves the stack by binary search. None of them is memoised, and none should be — a cache keyed
# by level materialises `cell_orders(…; level=k)`, a dense `Array{NTuple{D,Int},D}` over the whole level, to answer a
# question about the handful of cells a cycle marks. On the tanh ladder's finest level — 192² cells — that is a measured
# 589 872 bytes per copy to read a few dozen entries.
function _predicted(V::Space{D,T}, previous::_History{D,T}, k::Integer, cell::CartesianIndex{D}) where {D,T}
    Vprev, eprev = previous
    k <= length(Vprev.levels) || return T(Inf)
    q = minimum(cell_order(V.levels[k], cell))

    if !is_active(Vprev.levels[k].mask, cell)
        # Newly activated, so it is the child of an h-step one level up. The parent is the overlapping cell of level k−1
        # that carried the most error; on a nested stack there is exactly one.
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
        return η * T(_GAMMA_H) * T(rate) * T(_GAMMA_P)^(q - minimum(cell_order(Vprev.levels[k - 1], parent))) /
               sqrt(T(max(n, 1)))
    end

    # The h-coarsening row, which is read before the cell's own history because its evidence is the children's error,
    # not the parent's. `decide` reaches this only for a leaf of `V`, so a cover that was live in `Vprev` and is not now
    # is exactly an h-release, and the cell is predicted to get worse by the reciprocal of what the subdivision was
    # worth — the children's error, ℓ²-combined and scaled to this cell's current order, undone by the rate the h-step
    # was credited with. Without it a released parent falls through the equality test below, takes the no-evidence row
    # and buys p on sight, which is the oscillation the release was trying to avoid; the p-release row exists for the
    # same reason.
    if k < length(V.levels) && k < length(Vprev.levels)
        block = _cell_block(Vprev, k, k + 1, cell)
        released = zero(T)
        for ch in block
            (is_active(Vprev.levels[k + 1].mask, ch) && !is_active(V.levels[k + 1].mask, ch)) || continue
            released += (eprev.cells[k + 1][ch] * T(_GAMMA_P)^(q - minimum(cell_order(Vprev.levels[k + 1], ch))))^2
        end
        if released > 0
            rate = _h_rate(cell_box(Vprev, first(block); level=k + 1), cell_box(Vprev, cell; level=k), q, Val(D))
            return sqrt(released) / (T(_GAMMA_H) * T(rate))
        end
    end

    # Equal orders mean nothing was applied here, so there is no evidence and the caller takes p. A LOWERED order is
    # evidence, and of the opposite sign: the exponent goes negative, γ_p^(negative) exceeds one, and the cell is
    # predicted to get WORSE by the reciprocal of what the degree was worth. That is Melenk & Wohlmuth's coarsening row
    # for p, and without it a p-released cell falls through the equality branch and takes p again on sight.
    η = eprev.cells[k][cell]
    η > 0 || return T(Inf)
    q_prev = minimum(cell_order(Vprev.levels[k], cell))
    q == q_prev && return T(Inf)
    return η * T(_GAMMA_P)^(q - q_prev)
end

# The h-rate ρ^{q/D} between two cells of consecutive levels, from the boxes themselves rather than from an assumed
# bisection: ρ is the volume ratio, so ρ^{1/D} is the mesh size the h^p law of the energy norm refers to. See the note
# above for why a scalar rate on an anisotropic split is a judgement.
_h_rate(fine::AxisBox, coarse::AxisBox, q::Int, ::Val{D}) where {D} = (volume(fine) / volume(coarse))^(q / D)

# ── Reading the stack under one cell ──────────────────────────────────────────
#
# Every question below is asked once per marked cell and is answered from the per-axis block `_cell_block`
# (`src/ladder.jl`) resolves by binary search, never from `overlapping_cells`, which builds a mask over the whole target
# level. On a depth-3 3D ladder that is 0.29 µs against 7.65 ms per mark, and a cycle marks hundreds of them.
#
# A cell is a leaf when no finer level has taken any of it over. The distinction is the whole reason the p-step needs
# guarding: leaf semantics shed a covered cell's bubble modes, so raising the order of a non-leaf cell adds nothing that
# survives the dof walk. `refine` sends such a cell to h instead.
#
# EVERY finer level is asked, not only the one immediately below. The dof layer's covering rule tests each higher-id
# level in turn (`build_coverage` in `src/coverage.jl`), so on a stack where level k+2 is live over a cell while k+1 is
# dormant — a structure `ladder` admits by design and one `adapt` call builds — the cell is covered and its bubbles are
# shed. Asking k+1 alone called such a cell a leaf, sent it to p, and the order step it was given then vanished in the
# dof walk: a mark spent on nothing, every cycle, for as long as the skip lasted.
#
# "Any active cell of a finer level overlaps this one" rather than "fully covered" is deliberately the conservative side
# of the question: a partly covered cell does keep its bubbles, but it is also a cell the stack is already in the middle
# of taking over, and spending a p-step there is at best premature.

# Which levels hold an active cell at all, as one `Bool` per level. A verb builds this once and both tests below skip a
# dormant level without touching the geometry — which on a declared-but-empty ladder is every level under the mark, and
# is the common case. Measured on a depth-3 3D ladder with empty overlays, the leaf test on a base cell falls from 6.19
# µs to 0.05 µs; its worst case — a live 64³ level that does not reach the marked cell, so the whole block is read and
# none of it is live — is 4.5 µs.
_live_levels(V::Space) = [level.mask === nothing || any(level.mask) for level in V.levels]

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

# The block of level k+1 under cell `c` of level k: what an h-step on (k, c) wakes, and what a release of it puts back
# to sleep. It is empty on the finest level and wherever level k+1's box does not reach the cell, and every reader takes
# an empty block to mean the same thing — there is no h-step here: `_has_h_step` offers none, `refine` turns the mark
# into the order step, and `coarsen` has nothing to release. The empty block has `_cell_block`'s own type, so the
# answer stays concrete.
function _children(V::Space{D}, k::Integer, c::CartesianIndex{D}) where {D}
    k < length(V.levels) || return CartesianIndices(ntuple(_ -> 1:0, D))
    return _cell_block(V, k, k + 1, c)
end

# Whether an h-step on this cell would buy anything: level k+1 must exist, its box must reach the cell, and the block
# must hold a cell that is both dormant and a leaf.
#
# The three conditions are what makes the h-step total on an arbitrary stack, and each rules out a mark that would
# otherwise be re-issued every cycle for nothing. A level below that does not reach the cell — a sub-box overlay, or two
# overlays over different regions — offers no h-step, exactly as the finest level offers none, so both fall under one
# rule rather than a level-count test. A block already fully live offers none either: the step would set no mask bit.
# And a dormant cell that a *deeper* level has already taken over is dof-inert to activate, because the same covering
# rule that sheds the marked cell's bubbles sheds its would-be child's: measured on a depth-2 ladder with level 3 live
# over a base cell and level 2 empty, activating level 2 under it moved the active unknowns by exactly zero, 97 before
# and 97 after.
function _has_h_step(V::Space{D}, k::Integer, cell::CartesianIndex{D}, live=_live_levels(V)) where {D}
    children = _children(V, k, cell)
    isempty(children) && return false
    mask = V.levels[k + 1].mask
    return any(c -> !is_active(mask, c) && _is_leaf(V, k + 1, c, live), children)
end

# ── Refinement and release ────────────────────────────────────────────────────

"""
    refine(V::Space, estimate::ErrorEstimate; theta = 0.5, pmax = 8, previous = nothing) -> Space
    refine(V::Space; h = (), p = (), pmax = 8) -> Space

Refine `V` and return the refined space; prepare it in one rebuild with
`adapted(model, refine(model.prefold_space, est))`.

The first form is the loop: mark by Dörfler on `estimate` — the smallest set of cells carrying `theta` of the total
squared indicator — then decide per marked cell whether to spend on h or on p, and apply. Those two halves are
[`mark_cells`](@ref) and [`decide`](@ref), and this form is their composition,

    refine(V, est; theta, pmax, previous) ==
        refine(V; decide(V, est, mark_cells(est; theta); pmax, previous)..., pmax)

so a driver that needs the marked set or the h/p split itself — to report them, or to keep [`coarsen`](@ref) from
releasing what this cycle just refined — calls the two directly and hands the result to the second form rather than
reimplementing either. The second form takes the two sets explicitly as iterables of `(level, cell)` pairs, so an
indicator of your own plugs into the same application step.

The two steps, and what each one means here:

  - the **h-step** activates the cells of level `k + 1` that cover the marked cell of level `k`, mapped through the same
    geometric overlap [`overlapping_cells`](@ref) reports — refining "this cell" means activating its cover one level
    down — and gives the cells it wakes their parent's order unchanged. A cell that is already live keeps the order it
    earned, so this is a pure h-step: the resolution changes and the order does not. The target is always level `k + 1`;
    where that level has nothing under the cell — the finest level, or a level whose box does not reach it, as on a
    sub-box overlay — there is nothing to activate and the cell takes the order step instead, or is dropped where the
    family carries no per-cell order and there is no second step to take;
  - the **p-step** raises the marked cell's own order by one and activates nothing. `pmax` bounds the climb so a loop
    cannot run away: it raises only axes below `pmax` and leaves a cell already at or above it alone, so passing a
    `pmax` under a cell's current order means "no more p here" rather than "lower it". The step is only ever handed a
    leaf by the loop form, which is what makes it survive: leaf semantics shed a covered cell's bubble modes, so the
    same increment on a covered cell would be a no-op. Measured before that was understood, 356 cells raised while
    covered retained 576 active modes between them and **zero** bubbles.

A cell takes at most one order step per call, however many marks name it: a duplicated p-mark, or a p-mark beside an
h-mark that turns into the order step, raises it by one from its order in `V`.

Existing activation is preserved rather than replaced. [`adapt`](@ref) *sets* a level's mask, so an application that
forgets to keep what is already live silently un-refines the rest of the domain; this keeps it.

On an immersed model — one whose space carries a [`PhysicalDomain`](@ref) — there is no state transfer across an h-step:
[`L2Projection`](@ref) refuses a target carrying a physical domain, and [`Rewire`](@ref) refuses a target whose active
basis does not contain the source's, which is what leaf semantics make of a refinement. An adaptive loop on an immersed
model therefore re-solves on the refined space rather than carrying its iterate across, and `estimate` needs that fresh
solve before it can be called again.

`refine` returns `V` **itself** when nothing changed — every marked cell was already at `pmax`, or on the finest level,
or its cover was already live — so `refined === V` is the loop's exhaustion signal and is worth testing before paying
for another rebuild. Any change at all returns a new `Space`.

What the h-step presumes about the stack. It is a refinement rather than a deletion only because leaf semantics shed the
parent's bubble modes and the activated children replace them, and they replace them only where level `k + 1` *nests*
over the marked cell. A [`ladder`](@ref) guarantees that by construction; [`move!`](@ref) can void it and nothing
re-checks, so [`is_nested`](@ref) and `diagnostics(…).levels[k].nested` are the witnesses on a hand-built stack. Neither
form checks it here, because the geometric test has false alarms — it reports `false` on a pair that sheds nothing — and
refusing a mark on one would block a legitimate refinement. On a non-nested pair the overlap rule still decides which
cells move: a child straddling two parents is activated for whichever of them is marked, and released when either is.

Which step a marked cell takes is decided from its own refinement history. Pass `previous = (V_last, estimate_last)` —
the space and estimate from the cycle before — and each marked cell takes p when its indicator met the reduction its
last step predicted, h when it fell short. The two spaces must agree in cell counts on the levels they share, since a
cell's history is read at its own index; a ladder may grow levels between cycles. Omit `previous` and every marked cell
takes p, falling back to h only where p is unavailable; that is the right behaviour for the first cycle and the wrong
one thereafter. See the note above [`decide`](@ref) for the prediction and why it is preferred here to a smoothness
indicator. The prediction's own constants — Melenk & Wohlmuth's γ_p and γ_h — are fixed rather than exposed, because
they are the literature's calibration of the rule and not a knob to fit per problem; `theta` is the loop's only tunable.

See [`estimate`](@ref) for the rest of the loop.
"""
function refine(V::Space{D,T}, estimate::ErrorEstimate{D,T}; theta::Real=0.5, pmax::Integer=8,
                previous::Union{Nothing,_History{D,T}}=nothing) where {D,T}
    h, p = decide(V, estimate, mark_cells(estimate; theta=theta); pmax=pmax, previous=previous)
    return refine(V; h=h, p=p, pmax=pmax)
end

function refine(V::Space{D,T}; h=(), p=(), pmax::Integer=8) where {D,T}
    pmax >= 1 || throw(ArgumentError("pmax must be at least 1; got $pmax"))
    edit = _StackEdit(V)
    # Raise only the axes below the cap. A `pmax` under a cell's current order is a "no more p here" instruction, not a
    # request to lower an order `refine` was asked to enrich, and the loop's saturation test reads it the same way.
    #
    # Every order step starts from the cell's order in `V`, never from the edit, so a cell marked twice takes one step:
    # a duplicated p-mark, or a p-mark beside an h-mark that the finest level turns into the order step. Reading the
    # edit compounded them — a duplicated p-mark took an order-2 cell to 4 — against the docstring's "raises the marked
    # cell's own order by one".
    raised(o) = min.(o .+ 1, max.(o, Int(pmax)))

    for (k, cell) in p
        _check_mark(V, k, cell, "refine")
        # A level with no per-cell order has no p-step to take, and `_set_order!` drops the mark, which is what
        # `coarsen` does with the mirror case. `decide` never hands one over, so this is the explicit form's branch;
        # without the drop the mark would reach the level rebuild and raise there, a worse answer than "there was
        # nothing to spend here".
        _set_order!(edit, k, cell, raised(cell_order(V.levels[k], cell)))
    end

    for (k, cell) in h
        _check_mark(V, k, cell, "refine")
        # Level k+1 is the target, and where it has no cell under this one — the finest level, or a level whose box does
        # not reach the cell — there is nothing to activate and the order step is what is left to spend; where the
        # family carries no per-cell order there is no second step either, and `_set_order!` drops the cell rather than
        # turning it into a whole-level order change nobody asked for. This is where the substitution happens;
        # `_has_h_step` is where the loop form DECIDES that h is unavailable, and it sends such a cell to `p` so this
        # branch is unreachable from `refine(V, estimate)`. The two are not a duplicated rule: a decision that h is
        # unavailable and an application that converts an impossible h are different jobs, and the explicit form —
        # which takes the caller's own sets — needs the second on its own.
        children = _children(V, k, cell)
        if isempty(children)
            _set_order!(edit, k, cell, raised(cell_order(V.levels[k], cell)))
            continue
        end
        # The children take the parent's order as it stands, uncapped: `coarsen` restores exactly this value on release,
        # and capping here would break that inverse on any stack whose order already exceeds `pmax`. It is read from `V`
        # rather than from the edit, which the p-step above may already have raised — an h-step hands down the order the
        # parent had when the cycle started, never one bought in the same cycle.
        target = cell_order(V.levels[k], cell)
        mask = _mask!(edit, k + 1)
        for child in children
            # A child that is not live carries whatever order it held in a previous life, and `coarsen` clears that on
            # release — but only for the covers it releases. Writing the parent's order over it is what keeps a stale
            # entry from being resurrected: measured, one h-step onto a released order-6 cover returned 169 unknowns
            # where a cold h-step gives 57. A child that IS live is left exactly as it is — its order was earned on its
            # own evidence, and silently raising it would make this an order step wearing an h-step's name.
            mask[child] || _set_order!(edit, k + 1, child, target)
            mask[child] = true
        end
    end

    # The support extension. A family whose functions reach beyond the cell that generates them carries nothing on a
    # region narrower than that reach, so waking exactly the cells under a marked parent can wake *no unknowns at all* —
    # a refinement that refines nothing, and a loop that then marks the same cell again forever. Dilating the woken set
    # by the family's own support radius is what makes the step buy what it claims to; it is the support-extension
    # condition of admissible hierarchical meshes, and it is applied to the cells this step woke rather than to the
    # whole mask, so a region that was already live is left exactly as it was. `_support_radius` is zero for integrated
    # Legendre, so this loop is a no-op on the default family.
    #
    # The woken set is the bits the edited mask gained over `V`'s own, and that is exact rather than an approximation:
    # the h-step above only ever sets a bit and the p-step never touches a mask, so nothing else can differ. A level
    # without a mask in `V` is live everywhere and had nothing to wake.
    for (k, m) in pairs(edit.masks)
        old = V.levels[k].mask
        (m === nothing || old === nothing) && continue
        radius = _support_radius(V.levels[k].basis, nominal_order(V.levels[k]))
        all(iszero, radius) || (m .|= _dilate_cells(m .& .!old, radius))
    end
    return _apply(edit)
end

"""
    coarsen(V::Space; h = (), p = (), pmin = 1) -> Space

Release refinement. The mirror of [`refine`](@ref)'s explicit form, and the verb a transient needs: on a moving feature,
refinement left behind is pure cost.

  - an **h-mark** `(level, cell)` names a *parent*, and deactivates the cells of the level below that cover it — the
    exact inverse of `refine`'s h-step. The released children also have their order set back to the parent's, which is
    exactly what `refine`'s h-step gives them — so the two verbs invert each other on the whole order field, not merely
    on its observable part;
  - a **p-mark** `(level, cell)` lowers that cell's own order by one, floored at `pmin`: it lowers only axes above
    `pmin` and leaves a cell already at or below it alone, so a `pmin` above a cell's current order means "no more
    release here" rather than "raise it". As in `refine`, a cell takes at most one order step per call: a duplicated
    p-mark lowers it by one from its order in `V`.

Releasing a cover whose children are themselves covered is a no-op rather than an error: the region is carried further
down and the deeper level is what holds it, so releasing this level alone would leave a hole. On a family whose
functions reach beyond their cell (a B-spline level) that includes the support collar a held child needs, and no cell a
deeper level holds is released as part of anyone's collar. Release the stack from the top down and each call is well
defined. So is an h-mark on a level the one below does not reach, which has no cover to release. As in
[`refine`](@ref), a call that changes nothing returns `V` itself.

Lowering the order of a cell a finer level has taken over is **rejected**, and *any* finer level counts, not only the
one immediately below — the dof layer's covering rule tests them all, so a cell held by level `k + 2` while `k + 1` is
dormant is just as covered. Under leaf semantics a covered cell has already shed its high-order modes, so the change is
invisible while the cover is live and appears without warning when it lifts — measured, 173.5× the energy error, with
`diagnostics`, `isposdef`, the residual norm and `estimate`'s consistency flag all unchanged to the last digit. There is
no diagnostic that catches it, so it is refused here instead.

The h-mark presumes, as `refine`'s h-step does, that the level below nests over the marked cell; on a non-nested pair a
child straddling two parents is released when either of them is marked. A release *is* evidence to the next cycle's
decision: `refine`'s prediction scores a cell whose cover was just released by the reciprocal of what the subdivision
was worth, so a released region is not re-refined on sight.

An [`ErrorEstimate`](@ref) is indexed by the cells of the space it was taken on, and is zero on every cell that carried
no approximation there. A release rule that reads one must therefore be evaluated on that space, *before*
[`refine`](@ref) is applied: a cell the same cycle activates scores zero in the earlier estimate and reads as cold, so
releasing on the refined space undoes the h-step that just woke it. Exclude, too, any parent the same cycle marks for h
— its cover was woken by `refine` and this verb would put it straight back to sleep. Measured on
`examples/reproductions/traveling_laser_2d`, a driver missing both guards would undo 361 of the 510 h-steps it takes, 71
%. The two are independently sufficient on that driver: dropping either one alone leaves the run bit-identical, and only
removing both lets the cancellation through.

On an immersed model — one whose space carries a [`PhysicalDomain`](@ref) — there is no state transfer across this step:
[`L2Projection`](@ref) refuses a target carrying a physical domain, and [`Rewire`](@ref) refuses a target whose basis
does not contain the source's, which releasing a cover is. A transient that coarsens an immersed model therefore
re-solves on the released space rather than carrying its iterate across.

There is deliberately no `coarsen(V, estimate)` form. Which cells to release is not settled: measured on a translating
layer, a monotone loop reached a lower error at fewer unknowns than every release rule tried, because in a diffusive
problem the heat stays and there is little to release. Release earns its place on unknowns rather than on error — a
monotone loop carries 2.6× the from-scratch optimum after six layer-widths of travel — so the caller marks, and the
package does not pretend to know the rule.
"""
function coarsen(V::Space{D,T}; h=(), p=(), pmin::Integer=1) where {D,T}
    pmin >= 1 || throw(ArgumentError("pmin must be at least 1; got $pmin"))
    edit = _StackEdit(V)
    live = _live_levels(V)
    radius(k) = _support_radius(V.levels[k].basis, nominal_order(V.levels[k]))
    # The children released per level, for the support-collar mirror below: allocated when a release first reaches a
    # level whose family has a collar to release, and never on a level no release reaches.
    blocks = Vector{Union{Nothing,BitArray{D}}}(nothing, length(V.levels))
    # Lower only the axes above the floor, the mirror of `refine`'s `raised`, and from the cell's order in `V` for the
    # same reason: a duplicated p-mark lowers once.
    lowered(o) = max.(o .- 1, min.(o, Int(pmin)))

    for (k, cell) in p
        _check_mark(V, k, cell, "coarsen")
        # The mirror of `decide`'s `p_offered`: a level whose degree belongs to a whole-axis knot vector has no per-cell
        # order to lower, so there is nothing for this mark to release and it is dropped rather than turned into a
        # whole-level order change. `refine` never hands such a level a p-step, so a loop's release set cannot contain
        # one either; this is the explicit form's branch, and dropping keeps `coarsen` the inverse of `refine` on it.
        # The drop comes before the leaf test, so such a mark is dropped even on a covered cell.
        _supports_cell_order(V.levels[k].basis) || continue
        _is_leaf(V, k, cell, live) ||
            throw(ArgumentError("coarsen: cell $cell of level $k is covered by a finer level, " *
                                "and lowering a covered cell's order is not observable until the " *
                                "cover lifts. Release the cover first."))
        _set_order!(edit, k, cell, lowered(cell_order(V.levels[k], cell)))
    end

    for (k, cell) in h
        _check_mark(V, k, cell, "coarsen")
        k < length(V.levels) || throw(ArgumentError("coarsen: level $k is the finest, so it covers nothing to release"))
        # The released children take the PARENT's order, which is exactly what `refine`'s h-step gives them. An inactive
        # cell's order is unobservable, so this is not required for correctness — but it makes the two verbs inverses of
        # the whole order field rather than only of its observable part, which is what lets a transient loop refine and
        # release the same region thousands of times without drift. It is read from `V`, as `refine` reads it: a p-mark
        # in this call cannot have lowered it, since only a live child is released and a parent with a live child is no
        # leaf, so its p-mark has been refused above. Note `nominal_order` is NOT the value to use: it is a maximum kept
        # for buffer sizing, and on a level that has been elevated it has already moved.
        parent = cell_order(V.levels[k], cell)
        mask = _mask!(edit, k + 1)
        if blocks[k + 1] === nothing && !all(iszero, radius(k + 1))
            blocks[k + 1] = falses(V.levels[k + 1].mesh.cells)
        end
        block = blocks[k + 1]
        for child in _children(V, k, cell)
            # A child that is itself covered — by any deeper level, not only the next one — is not this level's to
            # release, and it stays out of the released block for the same reason: the deeper level is built on it.
            # Recorded in the block, the collar pass below would release it anyway, with its whole dilated footprint —
            # measured on an order-3 B-spline ladder of depth 2, releasing the level-1 parent of a child that level 3
            # refines left level 2 with no live cell under 64 live level-3 cells. Left out, it stays in the eroded core
            # below, so the collar keeps its dilated footprint and the held child keeps the support its own functions
            # need: the docstring's "no-op rather than a hole".
            _is_leaf(V, k + 1, child, live) || continue
            block === nothing || (block[child] = true)
            mask[child] || continue
            mask[child] = false
            _set_order!(edit, k + 1, child, parent)
        end
    end

    # The mirror of `refine`'s support extension, and what keeps this verb the exact inverse it claims to be. `refine`
    # wakes a marked parent's block *dilated* by the family's `_support_radius`; releasing only the block would leave
    # that collar live for ever — no public verb reaches it, since it lies outside every block `coarsen` looks at — and
    # a transient loop that refines and releases a travelling feature would grow its active set without bound. Measured
    # before this pass existed, on a depth-2 ladder at degree 3 with the mark walking (3,4) … (7,4), the live level-2
    # count ran 60, 56, 88, 84, 108.
    #
    # The collar is released *minus* what other still-refined parents justify: a cell of the released parent's dilated
    # footprint stays live if it also lies in the dilated footprint of a parent whose own block is still live. That is
    # exactly the set `refine` would have produced from the remaining marks, so refine-then-coarsen returns the mask it
    # started from, and a cell some other caller activated outside every footprint is never touched.
    #
    # Recover the set `refine` dilated, so the collar can be released without taking one another refined region still
    # needs. It cannot be read off the live cells exactly, and that is worth stating rather than papering over: the
    # dilation fills in neighbouring parents' blocks, so "this parent's block is live" stops meaning "this parent was
    # refined", and clipping at a box face defeats in opposite directions the two instruments that might recover it — a
    # morphological erosion recovers more than the block there, while requiring a parent's whole dilated footprint to
    # be live rejects a genuinely refined edge parent. Measured, the two agree cell for cell on an 8² ladder, so the
    # simpler one is used.
    #
    # Erosion is therefore the conservative reading: away from a box face `dilate(erode(dilate(X, r), r), r) ==
    # dilate(X, r)` recovers the blocks exactly and the release is exact; near one it recovers more, so a collar stays
    # live. Conservative is the safe direction — it can only leave refinement standing, never take away refinement
    # another mark needs — and it is what fixes the defect that matters, an active set that grew without bound as a
    # feature travelled.
    #
    # A collar cell a deeper level holds is not this release's either, whoever's footprint it lies in: the leaf test on
    # each release is the h-mark loop's own rule, applied to the collar. A level whose block stayed empty — no h-mark
    # reached it, or every child it named is held — releases nothing and is skipped.
    for (k, block) in pairs(blocks)
        (block === nothing || !any(block)) && continue
        r, mask = radius(k), _mask!(edit, k)
        keep = _dilate_cells(_erode_cells(active_cells(V; level=k), r) .& .!block, r)
        dropped = _dilate_cells(block, r)
        for c in CartesianIndices(mask)
            dropped[c] && !keep[c] && mask[c] && _is_leaf(V, k, c, live) && (mask[c] = false)
        end
    end
    return _apply(edit)
end

# ── One pending edit of the stack ─────────────────────────────────────────────
#
# What one `refine` or `coarsen` call writes, held until the call ends and applied in one pass. Both verbs write a level
# from two sides — the order steps of its own cells, and the mask bits and inherited orders of the children a coarser
# mark wakes or releases — so the writes are collected per level and each level that ends up changed is rebuilt ONCE,
# with its final mask and orders together. Composing `adapt` and `elevate` instead rebuilds a level whose mask and
# orders both moved twice, and needs dense copies of every touched level's mask and order field just to tell whether
# anything moved.
#
# The two kinds of write are stored as they are shaped. A mask write is dense — a support dilation sets a whole collar —
# so a level's mask is copied on its first write and edited in place. An order write is sparse — a handful of cells per
# mark — so orders are a per-level dictionary of the cells written, and no dense order field exists unless an edit
# survives to `_apply`.
struct _StackEdit{D,S<:Space{D}}
    space::S
    masks::Vector{Union{Nothing,BitArray{D}}}
    orders::Vector{Union{Nothing,Dict{CartesianIndex{D},NTuple{D,Int}}}}
end

function _StackEdit(V::Space{D}) where {D}
    n = length(V.levels)
    return _StackEdit{D,typeof(V)}(V, Vector{Union{Nothing,BitArray{D}}}(nothing, n),
                                   Vector{Union{Nothing,Dict{CartesianIndex{D},NTuple{D,Int}}}}(nothing, n))
end

# Level k's edited mask, copied from the space on first use.
function _mask!(e::_StackEdit, k::Integer)
    m = e.masks[k]
    return m === nothing ? (e.masks[k] = active_cells(e.space; level=k)) : m
end

# Record an order write. A family without a per-cell order has none to write — a B-spline level's degree belongs to a
# whole-axis knot vector, and no step may change it — so the write is dropped here, once, rather than guarded at every
# site that writes: the p-step, the order step an h-mark turns into, and the order a woken or released child is given.
function _set_order!(e::_StackEdit{D}, k::Integer, c::CartesianIndex{D}, o::NTuple{D,Int}) where {D}
    _supports_cell_order(e.space.levels[k].basis) || return e
    written = e.orders[k]
    written === nothing && (written = e.orders[k] = Dict{CartesianIndex{D},NTuple{D,Int}}())
    written[c] = o
    return e
end

# The edit as a space. Writes that left a level as it was are dropped first — a mask equal to the level's own, an order
# equal to the cell's own — and each level that still differs is rebuilt once. When none does the input space comes
# back ITSELF, `===`, which is the exhaustion signal `refine` documents: a call that only re-wrote what was there — a
# release of a cover that was not live, an order step at the cap, a wake of a cover already live — must not hand back a
# fresh copy, or a saturated loop rebuilds an identical model for ever.
#
# The rebuild is the one `adapt` followed by `elevate` produces, in one `_relevel` per changed level: the final mask,
# collapsed as `adapt` collapses it, and — where an order moved — the written cells normalised exactly as `elevate`'s
# pair form normalises them; an unchanged part is the level's own. Only a family with a per-cell order carries order
# writes, and integrated Legendre instantiates to itself, so building from the input level rather than from an
# intermediate re-masked copy changes nothing but the count of rebuilds.
function _apply(e::_StackEdit)
    V = out = e.space
    for k in eachindex(V.levels)
        old, m = V.levels[k], e.masks[k]
        m !== nothing && (old.mask === nothing ? all(m) : m == old.mask) && (m = nothing)
        changed = [c => o for (c, o) in something(e.orders[k], ()) if o != cell_order(old, c)]
        m === nothing && isempty(changed) && continue
        out = _relevel(out, k; mask=m === nothing ? old.mask : _collapse_mask(m),
                       field=isempty(changed) ? old.orders : _normalize_order(_pair_orders(V, k, changed), old.mesh))
    end
    return out
end
