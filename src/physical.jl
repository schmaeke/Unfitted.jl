# ── CSG level-set tree ────────────────────────────────────────────────────────
#
# An immersed domain Ω is described by a Boolean combination of smooth level sets ("leaves"), rather than a single
# scalar φ. Carrying the constituent leaves separately keeps every boundary piece smooth, so the implicit quadrature
# kernel (`src/implicit.jl`) stays high-order across creases and corners that a single `max`/`min` level set would turn
# into a non-differentiable kink (Saye's multi-component construction: R. I. Saye, "High-order quadrature on
# multi-component domains implicitly defined by multivariate polynomials", J. Comput. Phys. 448 (2022) 110720).

"""
    LevelSet

Abstract supertype of the CSG nodes describing an immersed domain Ω. The concrete nodes are a [`leaf`](@ref) (a single
smooth level set with `Ω = {f ≤ 0}`) and the Boolean combinators built by `intersect`, `union`, `setdiff`, and
[`complement`](@ref). A `LevelSet` is passed to [`physical_domain`](@ref).
"""
abstract type LevelSet end

"""
    Leaf(f, lipschitz)

A single smooth level set `f` with `Ω_leaf = { x : f(x) ≤ 0 }`. `lipschitz` is a Lipschitz constant `L` of `f`
(`|f(x) − f(y)| ≤ L‖x − y‖`); `Inf` disables the cheap sign certificate in the cell classifier and in the implicit
quadrature kernel, and forces corner sampling in both. Construct with [`leaf`](@ref).
"""
struct Leaf{F} <: LevelSet
    f::F
    lipschitz::Float64
end

# n-ary intersection (Ω = ⋂ parts; membership = inside *all* parts) and union (Ω = ⋃ parts; membership = inside *any*
# part). `Not` is set complement.
struct AllOf{C<:Tuple} <: LevelSet
    parts::C
end
struct AnyOf{C<:Tuple} <: LevelSet
    parts::C
end
struct Not{C<:LevelSet} <: LevelSet
    part::C
end

"""
    leaf(f; lipschitz = Inf) -> Leaf

Wrap a scalar level-set callback `f` (on `SVector{D,T}` coordinates) as a CSG leaf with `Ω_leaf = { x : f(x) ≤ 0 }`. `f`
must accept `ForwardDiff.Dual` arguments so the quadrature kernel can take its gradient: the finite-cell path
([`implicit_rule`](@ref)) installs no gradient override, and the `grad` keyword of
[`implicit_volume_quadrature`](@ref) is reachable only by calling that kernel directly.

`f` must also be safe to call from several threads at once — a pure function of `x` is. The package builds the
quadrature of cut cells and of cut boundary faces on every thread, and those builds evaluate the leaves concurrently; a
custom `cut_rule` or `compression` (see [`PhysicalDomain`](@ref)) makes the cell builds serial, but the face builds
always use [`implicit_rule`](@ref) and still evaluate the leaves on several threads.

`lipschitz` is the Lipschitz constant used by the sign certificate of both the cell classifier and the implicit
quadrature kernel; `1.0` for a true signed-distance leaf, `Inf` to rely on corner sampling. Supplying it is what lets a
feature smaller than a mesh cell survive: under `Inf` the kernel can prune such a leaf on the cell box and integrate the
cut cell as if it were entirely inside Ω.
"""
function leaf(f; lipschitz::Real=Inf)
    lipschitz > 0 || throw(ArgumentError("lipschitz must be positive; got $lipschitz"))
    return Leaf(f, Float64(lipschitz))
end

# Bare callables passed to a combinator are auto-wrapped as default leaves.
_as_levelset(g::LevelSet) = g
_as_levelset(f) = leaf(f)

"""
    intersect(a::LevelSet, b...) -> LevelSet
    union(a::LevelSet, b...) -> LevelSet
    setdiff(a::LevelSet, b) -> LevelSet
    complement(a) -> LevelSet

CSG combinators on level sets (the `Base` set operations are extended for the [`LevelSet`](@ref) type). `intersect`
builds `Ω = ⋂ {fᵢ ≤ 0}`, `union` builds `Ω = ⋃ {fᵢ ≤ 0}`, `setdiff(a, b) = a ∩ complement(b)` (e.g. an annulus as a disk
minus a disk), and `complement` flips inside and outside. Bare callable arguments are auto-wrapped as default
[`leaf`](@ref)s. The first argument must be a `LevelSet`, so wrap at least one operand with `leaf` (e.g.
`intersect(leaf(f), g)`).
"""
Base.intersect(a::LevelSet) = a
Base.intersect(a::LevelSet, b, rest...) = AllOf((a, _as_levelset(b), map(_as_levelset, rest)...))

"""
    union(a::LevelSet, b...) -> LevelSet

CSG union of level sets: `Ω = ⋃ {fᵢ ≤ 0}`, the region inside *any* part. Bare callable arguments are auto-wrapped as
default [`leaf`](@ref)s, so only the first argument has to be a `LevelSet` (e.g. `union(leaf(f), g)`). One of the four
combinators documented together under `intersect`; pass the result to [`physical_domain`](@ref).
"""
Base.union(a::LevelSet) = a
Base.union(a::LevelSet, b, rest...) = AnyOf((a, _as_levelset(b), map(_as_levelset, rest)...))

"""
    setdiff(a::LevelSet, b) -> LevelSet

CSG difference of level sets: `setdiff(a, b) = a ∩ complement(b)`, e.g. an annulus as a disk minus a disk. `b` may be a
bare callable, auto-wrapped as a default [`leaf`](@ref). One of the four combinators documented together under
`intersect`; pass the result to [`physical_domain`](@ref).
"""
Base.setdiff(a::LevelSet, b) = AllOf((a, Not(_as_levelset(b))))

"""
    complement(a) -> LevelSet

Set complement of a level set: inside and outside are swapped (`Ω ↦ ℝᴰ ∖ Ω`). `a` is a [`LevelSet`](@ref) or a bare
callable (auto-wrapped as a default [`leaf`](@ref)). One of the CSG combinators alongside `intersect`, `union`, and
`setdiff`; pass the result to [`physical_domain`](@ref).
"""
complement(a) = Not(_as_levelset(a))

# ── Tree evaluation ───────────────────────────────────────────────────────────

# Membership test x ∈ Ω. A leaf is inside where f ≤ 0; intersection/union/ complement combine with all/any/not.
# Short-circuiting is fine here (no index bookkeeping), unlike the classifier's three-valued walk below.
_inside(l::Leaf, x) = l.f(x) <= 0
_inside(n::AllOf, x) = all(p -> _inside(p, x), n.parts)
_inside(n::AnyOf, x) = any(p -> _inside(p, x), n.parts)
_inside(n::Not, x) = !_inside(n.part, x)

# A single scalar that is ≤ 0 exactly on Ω, reconstructed from the tree (intersection → max, union → min, complement →
# negate). This is the representation the kernel deliberately avoids for *integration* (its kinks are why we carry
# leaves separately), but it is the right thing to *display*: a contour at 0 reproduces ∂Ω. Used by the post-processing
# samples (the `level_set` array and the cut refinement in `postprocessing.jl`) and nowhere in the hot path.
_value(l::Leaf, x) = l.f(x)
_value(n::AllOf, x) = maximum(p -> _value(p, x), n.parts)
_value(n::AnyOf, x) = minimum(p -> _value(p, x), n.parts)
_value(n::Not, x) = -_value(n.part, x)

# Collect the leaves in a stable depth-first order; the quadrature kernel partitions by these and the classifier
# certifies each one's sign.
_collect_leaves!(acc, l::Leaf) = (push!(acc, l); acc)
_collect_leaves!(acc, n::AllOf) = (foreach(p -> _collect_leaves!(acc, p), n.parts); acc)
_collect_leaves!(acc, n::AnyOf) = (foreach(p -> _collect_leaves!(acc, p), n.parts); acc)
_collect_leaves!(acc, n::Not) = _collect_leaves!(acc, n.part)
_leaves(g::LevelSet) = _collect_leaves!(Leaf[], g)

# ── PhysicalDomain ────────────────────────────────────────────────────────────

"""
    PhysicalDomain{G,T,R,C}(geometry, alpha, subcell_length_scale, max_depth,
                            moment_order_factor, keep_fictitious, cut_rule, compression)

CSG level-set description of a physical domain Ω for finite-cell-style immersed integration. `Ω = { x : x ∈ geometry }`,
where `geometry` is a [`LevelSet`](@ref) tree of smooth leaves combined by `intersect`/`union`/ `setdiff`/`complement`
(a single [`leaf`](@ref) is the degenerate case, with the familiar `Ω = {φ ≤ 0}`).

Fields:

  - `geometry::G`: the CSG level-set tree. Each leaf carries its own Lipschitz constant; the tree's Boolean structure
    defines membership.
  - `alpha::T`: fictitious-region stabilization weight (α-FCM). `0` is the strict cut path; `> 0` enriches every **cut**
    cell's quadrature with the α-scaled full-cell rule, so cut-cell dofs whose basis support lies in the fictitious part
    still get a well-posed contribution (see `_trimmed_rule` in `intersections.jl`). It is honoured one
    dimension down on a boundary **face** in exactly the same way — on the level set restricted to that face's affine
    slice (`_facet_regions` in `dirichlet.jl`) — which is what lets α stabilise a thin physical sliver of face; the
    price there is that the rule then carries points outside Ω, so a datum defined only on Ω needs `alpha = 0` (see
    [`boundary`](@ref)). Fully-fictitious cells (support entirely outside Ω) are **dropped from the dof layout** unless
    `keep_fictitious` says otherwise; α alone never keeps them, because the α term stabilizes cuts, not whole fictitious
    cells.
  - `keep_fictitious::Bool`: opt into the classic α-FCM treatment where fully-fictitious cells are *kept* active with
    α-scaled full-cell quadrature instead of being dropped. `false` (default) drops them, which keeps the system lean
    and the physical domain clean for post-processing; `true` requires `alpha > 0`, and [`physical_domain`](@ref) raises
    an `ArgumentError` on the `alpha = 0` pairing rather than building a domain whose kept cells carry no quadrature.
  - `subcell_length_scale::T`: target box size, in physical units, for the binary subdivision both the classifier and
    the implicit-quadrature kernel share. A box is bisected until its largest axis extent is `≤ subcell_length_scale`
    (capped by `max_depth`); `_effective_subcell_depth` turns the two into a single depth budget. It feeds exactly two
    consumers, both subdivision budgets — not an octree moment grid:
    (1) the cut/full/fictitious classifier, which bisects only when the Lipschitz certificate and corner sampling cannot
    decide a box, so this is a geometry-*robustness* knob (resolving thin or near-tangent features); and
    (2) the subdivision of [`implicit_rule`](@ref)'s kernel, which bisects a cut box only where no axis is graph-like
    and clear of a fold (a leaf turning back inside or just beyond the box). A cut cell on which some axis is gets a
    rule of the kernel's full order at once and is never subdivided, so on those cells the accuracy does not depend on
    this scale at all. Pick it relative to the smallest geometric feature you must classify cleanly.
  - `max_depth::Int`: hard cap on the subdivision depth of both consumers above.
  - `moment_order_factor::Int`: multiplier on the order of the tensor Legendre moments a cut cell's rule reproduces,
    per axis (`factor × max(cell_order)` over the region's parent cells).

    `2` (the default) spans degree `2p`, so trial × test *products* integrate exactly — matching on cut regions what
    tensor Gauss already delivers on `:full` ones — and the compressed rule carries at most `∏(2p+1)` points.

    `1` spans degree `p` and carries at most `∏(p+1)` points, the tensor Gauss count on an uncut cell. It is the
    convention of much of the moment-fitting literature and of QuESo, whose trimmed-element rules likewise carry
    `n ≤ ∏(p+1)` points, and it is dramatically cheaper: a compression costs roughly the cube of the point count, so
    with [`NonnegativeFit`](@ref) over the 146 cut cells of the space-time cavity geometry at cells `(8,8,8)` it runs
    1.1 s at `∏(p+1) = 125` against ≈ 500 s at `∏(2p+1) = 729`.

    It is not, however, a free choice, and the condition is sharp. A stiffness integrand `∇u·∇v` has degree `2(p−1)`, so
    exactness needs the moment basis to span `2p − 2`, i.e. `factor ≥ 2 − 2/p` — which `1` satisfies only up to `p = 2`.
    Linear functionals (sources, loads) are degree `p` and are exact at `1` for any order. Measured on
    `examples/applications/kirsch_plate_2d`, a plane-stress plate swept over orders 1–7, relative L² error at factor `1`
    against factor `2`:

        order 1   2.41e-2    2.19e-2
        order 2   1.12e-3    1.11e-3
        order 3   4.48e-4    1.03e-4
        order 4   3.39e-4    9.39e-6
        order 7   7.33e-2    < 1e-6

    At factor `1` the sequence stops converging after `p = 2` and then diverges, ending three times worse than first
    order, while the moment residual holds at ~1e-16 throughout — the rule is exact for the moments it reproduces, and
    those are too few. Use `1` for low-order work or when only linear functionals are integrated over cut cells; keep
    `2` whenever high-order convergence on cut cells is the point.
  - `cut_rule::R`: the first stage of a cut cell's quadrature, which integrates `box ∩ Ω` accurately — by default
    [`implicit_rule`](@ref). See "The cut rule and its compression" below.
  - `compression::C`: the second stage, which reduces the cut rule to `∏(moment order + 1)` points or fewer that
    reproduce its moments — by default [`NonnegativeFit`](@ref) — or `nothing` to integrate cut cells with the cut rule
    itself. See below.

# The cut rule and its compression

Every cut region's rule is `moment_fit_rule(physical, box, moment_order)` ([`moment_fit_rule`](@ref)): the cut rule
integrates `box ∩ Ω` with as many points as accuracy takes, and the compression replaces them by at most `nbasis =
∏(moment_order .+ 1)` points that reproduce the cut rule's tensor Legendre moments. Either stage can be replaced by a
callable; both are called once per distinct `(region box, moment order)` pair.

The cut rule has the signature

    cut_rule(physical, box, moment_order) -> (points, weights)

and returns a rule for `box ∩ Ω` in **physical coordinates**: `points[q] ∈ box ⊂ ℝᴰ` and `weights[q]` its physical
weight, so `Σ w_q ≈ vol(box ∩ Ω)`. A rule in the box's reference frame silently integrates the wrong geometry. Its
accuracy is the accuracy of the cell: the compression reproduces its moments and can be no better than it. A cut rule
that finds nothing returns empty vectors, which routes the region to `:cut_failed` (`α = 0`) or `:cut_alpha_failed`
(`α > 0`) and reaches `diagnostics(...).fit_failure_count`. A rule that misses part of a cut volume is more dangerous
than an empty one: a dof supported only where the rule has no points gets a zero row and the system is singular at
`α = 0` — measured for a space-tree rule that keeps the tensor Gauss points of its cut leaves that fall in Ω, at depth 3
on an 8² mesh at every order from 2.

The compression has the signature

    compression(points, weights, moments, box, moment_order) -> (points, weights)

and receives the cut rule together with its moments `m_α = Σ_q w_q ψ_α(ξ(x_q))`, where `ψ_α(ξ) = ∏_d L_{α_d}(ξ_d)` is
the tensor product of Legendre polynomials (normalised to `L_n(1) = 1`) on the box's reference coordinate
`ξ ∈ [−1, 1]ᴰ`, and `α` runs over `CartesianIndices(0:p₁, …, 0:p_D)` in column-major order. It returns a rule in the
same physical frame whose moments match `m`. [`moment_fit_rule`](@ref) measures that: a rule whose residual exceeds
1e-2 `‖m‖` is discarded for the cut rule itself (region kind `:cut_fallback`). A compression is also applied on the
boundary faces of a cut region, one dimension down, so it must not assume the box's dimension.

Three properties of the pipeline are the caller's responsibility for either stage:

 1. **Weights must be non-negative.** The α-FCM blend `(1 − α)·rule ∪ α·tensor` assumes it, the Dirichlet trace solve
    reads a zero diagonal of its boundary mass as an unsupported dof only because every facet weight is `≥ 0`, and a
    negative weight can make an otherwise SPD form indefinite and a squared norm negative. A moment fit at fixed points
    — the tensor Gauss points of the box, solved in closed form — is the standard example: on 31 cut cells of discs,
    balls, polytopes and perforated cells it put negative weights on 38% of its points and 79% of its points outside Ω,
    and the L² error it integrated came out negative.
 2. **The callable must be a deterministic function of its arguments.** Rules are memoised per
    `(region box, moment order)` in a cache the [`Model`](@ref) owns per participating space and threads back through
    every plan rebuild, so a callable reading mutable state outside its arguments is captured at its first answer for a
    box and silently reused — across rebuilds, not merely within one. Put every knob in the callable itself.
 3. **A custom stage is run serially.** The package builds cut rules on every thread when both stages are its own;
    with a caller's callable in either stage it builds them on one thread, because caller code is under no
    thread-safety contract. The leaves are another matter (see [`leaf`](@ref)).

The cut rule is not consulted on a boundary face: it is a rule on a cell box in the model's own dimension, so a face
always takes [`implicit_rule`](@ref) on the level set restricted to its affine slice, followed by the domain's
compression. Both stages are stored on the domain rather than in a global, so two models in one session can use
different ones and nothing about the choice is process-wide.

Construct via [`physical_domain`](@ref).
"""
struct PhysicalDomain{G<:LevelSet,T<:Real,R,C}
    geometry::G
    alpha::T
    subcell_length_scale::T
    max_depth::Int
    moment_order_factor::Int
    keep_fictitious::Bool
    cut_rule::R
    compression::C
end

"""
    physical_domain(geometry; lipschitz=Inf, alpha=0.0, keep_fictitious=false,
                    subcell_length_scale, max_depth=8, moment_order_factor=2,
                    cut_rule=implicit_rule, compression=NonnegativeFit())

Construct a [`PhysicalDomain`](@ref). `geometry` is either a [`LevelSet`](@ref) CSG tree (built from [`leaf`](@ref) and
`intersect`/`union`/`setdiff`/ [`complement`](@ref)) or a bare scalar callable `φ`, which is auto-wrapped as a single
leaf with `Ω = {φ ≤ 0}` — preserving the single-level-set ergonomics `physical_domain(φ; lipschitz=…)`. The `lipschitz`
keyword applies only to that auto-wrapped single leaf; a CSG tree carries one constant per leaf, so set it on the
`leaf(...)` call instead. Passing a finite `lipschitz` together with a tree that has a leaf without its own constant is
**rejected**, rather than silently ignored: the certificate the caller asked for would never be installed, and the
uncertified leaf would then be integrated as if its feature were not there.

`subcell_length_scale` is the **required** classifier accuracy / subdivision scale (see the [`PhysicalDomain`](@ref)
docstring). `alpha = 0` is the strict-cut path; `alpha > 0` enables α-FCM. `keep_fictitious = true` retains
fully-fictitious cells as active dofs (the classic α-FCM fill) and therefore requires `alpha > 0` — pairing it with
`alpha = 0` leaves those cells without quadrature and is rejected. `moment_order_factor` sets the order of the moments a
cut cell's rule reproduces.

`cut_rule` and `compression` are the two stages of a cut cell's quadrature: [`implicit_rule`](@ref) integrates the cut
volume, and [`NonnegativeFit`](@ref) compresses that rule to at most `nbasis` points; `compression = nothing` keeps
the cut rule uncompressed. Either stage can be a callable of the caller's; the contracts they must honour are under
"The cut rule and its compression" in the [`PhysicalDomain`](@ref) docstring.
"""
function physical_domain(geometry; lipschitz::Real=Inf, alpha::Real=0.0, keep_fictitious::Bool=false,
                         subcell_length_scale::Real, max_depth::Integer=8, moment_order_factor::Integer=2,
                         cut_rule=implicit_rule, compression=NonnegativeFit())
    # Validated here rather than only inside `leaf`, so the constraint holds on every path: with a tree argument the
    # auto-wrap never runs, and `lipschitz` would otherwise reach no check at all.
    lipschitz > 0 || throw(ArgumentError("lipschitz must be positive; got $lipschitz"))
    alpha >= 0 || throw(ArgumentError("alpha must be ≥ 0; got $alpha"))
    !(keep_fictitious && iszero(alpha)) ||
        throw(ArgumentError("keep_fictitious=true requires alpha > 0: fully-fictitious cells " *
                            "kept active under the strict-cut (alpha=0) path receive no " *
                            "quadrature and would make the system singular"))
    subcell_length_scale > 0 || throw(ArgumentError("subcell_length_scale must be > 0; got $subcell_length_scale"))
    max_depth >= 0 || throw(ArgumentError("max_depth must be ≥ 0; got $max_depth"))
    moment_order_factor >= 1 || throw(ArgumentError("moment_order_factor must be ≥ 1; got $moment_order_factor"))
    # A bare callable becomes a single leaf, honouring the `lipschitz` keyword; an explicit CSG tree carries per-leaf
    # Lipschitz constants already, so the keyword has nowhere to go there. Only the combination that actually loses
    # information is rejected — a finite constant supplied alongside a tree with a leaf that carries none — because that
    # is the case where the caller asked for a certificate and would get none: the classifier and the quadrature kernel
    # then fall back to corner sampling, prune the uncertified leaf on any box whose corners miss its feature, and
    # integrate the cell as if the feature were not there, all while reporting `:cut` and a round-off fit residual.
    # Rejecting is preferred over applying `L` to the leaves that lack one: a single constant is not generally valid for
    # every leaf (scaling one leaf by 10 scales its constant by 10), and a constant that is too small certifies a
    # uniform sign that does not hold — the same silent over-integration, now with the package's fingerprints on it. A
    # tree whose leaves all carry their own constants is unaffected by the keyword, so passing one alongside such a tree
    # stays legal.
    if geometry isa LevelSet && !isinf(lipschitz)
        ls = _leaves(geometry)
        bare = count(l -> isinf(l.lipschitz), ls)
        bare == 0 || throw(ArgumentError("lipschitz=$lipschitz was given with a LevelSet tree, and " *
                                         "$bare of its $(length(ls)) leaves carry no constant of their " *
                                         "own; set it on the leaf instead: leaf(f; lipschitz=$lipschitz)"))
    end
    g = geometry isa LevelSet ? geometry : leaf(geometry; lipschitz=lipschitz)
    T = promote_type(typeof(float(alpha)), typeof(float(subcell_length_scale)))
    return PhysicalDomain(g, T(alpha), T(subcell_length_scale), Int(max_depth), Int(moment_order_factor),
                          keep_fictitious, cut_rule, compression)
end

"""
    levelset_value(physical, x) -> Real

Scalar reconstruction of the domain's level set at `x` that is `≤ 0` exactly on Ω (intersection → max of children, union
→ min, complement → negate, leaf → `f(x)`). Intended for visualization — a contour at level 0 reproduces ∂Ω — not for
integration, which uses the separate leaves.
"""
levelset_value(physical::PhysicalDomain, x) = _value(physical.geometry, x)

# ── Box certificates ──────────────────────────────────────────────────────────

# Number of binary-subdivision levels needed to drive `box`'s largest axis extent down to
# `physical.subcell_length_scale`, capped at `physical.max_depth`. Shared by the classifier and the quadrature kernel's
# subdivision budget so the resolution contract is uniform. Boxes already at or below the scale return 0.
function _effective_subcell_depth(physical::PhysicalDomain, box::AxisBox)
    max_extent = maximum(box.upper - box.lower)
    max_extent <= physical.subcell_length_scale && return 0
    return min(physical.max_depth, ceil(Int, log2(max_extent / physical.subcell_length_scale)))
end

# Largest Lipschitz constant declared anywhere in the tree, or `Inf` if any leaf declines to declare one.
# `levelset_value` collapses the tree with max / min / negate, and each of those is non-expansive in its arguments, so
# the collapsed scalar is itself Lipschitz with this constant. That is what the post-processing cut refinement
# (`_cut_marked`) needs: it bands on the collapsed φ rather than on the separate leaves, so it needs a constant for φ,
# not one per leaf. `Inf` propagates — one undeclared leaf leaves the whole tree uncertified, exactly as it does for
# `_tri`, and the refinement then falls back to corner signs.
_levelset_lipschitz(l::Leaf) = float(l.lipschitz)
_levelset_lipschitz(n::AllOf) = maximum(_levelset_lipschitz, n.parts)
_levelset_lipschitz(n::AnyOf) = maximum(_levelset_lipschitz, n.parts)
_levelset_lipschitz(n::Not) = _levelset_lipschitz(n.part)

# ── Cell classifier ───────────────────────────────────────────────────────────

# Three-valued (Kleene) membership certificate over the CSG tree: returns `+1` if the box is certainly inside Ω, `-1` if
# certainly outside, `0` if undetermined. Each leaf is certified by its Lipschitz bound (`_certified_sign`: |f(c)| > L·r
# ⇒ uniform sign, and inside is f < 0); the Boolean combinators propagate certainty without enumerating sign patterns —
# e.g. a union is certainly inside as soon as one part is, certainly outside only if every part is. This is what lets a
# cut cell of one component be classified `:full`/`:fictitious` when another component dominates it.
_tri(l::Leaf, box::AxisBox) = -_certified_sign(l.f, l.lipschitz, box)
function _tri(n::AllOf, box::AxisBox)
    r = 1
    for p in n.parts
        t = _tri(p, box)
        t == -1 && return -1
        t == 0 && (r = 0)
    end
    return r
end
function _tri(n::AnyOf, box::AxisBox)
    r = -1
    for p in n.parts
        t = _tri(p, box)
        t == 1 && return 1
        t == 0 && (r = 0)
    end
    return r
end
_tri(n::Not, box::AxisBox) = -_tri(n.part, box)

# Tally Ω-membership at the box center and its 2ᴰ corners. Mixed counts prove ∂Ω crosses the box; corners — rather than
# interior points — are the natural sample set for an axis-aligned box and double as the child vertices when the octree
# recurses.
function _corner_membership(geometry::LevelSet, box::AxisBox{D}) where {D}
    inside = count(x -> _inside(geometry, x), (center(box), _box_vertices(box)...))
    return inside, 2^D + 1 - inside
end

# Recursive classifier returning one of `:full`, `:cut`, `:fictitious`. Strategy, cheapest first:
#
#   1. Three-valued Lipschitz certificate on the CSG tree. A definite verdict settles `:full`/`:fictitious` with no
#      sampling.
#   2. Corner membership. Mixed inside/outside proves ∂Ω crosses the box.
#   3. Octree recursion. Signs agree but the certificate did not fire; bisect into 2ᴰ children (`_box_children`) and
#      recurse. Any `:cut` child, or any disagreement between `:full`/`:fictitious` children, makes the parent `:cut`.
#   4. Depth budget exhausted with agreeing samples, or a box too small to bisect: trust the consensus.
#
# The single-leaf case reduces exactly to the Lipschitz-then-corner strategy the package used before CSG.
function _classify_box(geometry::LevelSet, box::AxisBox, depth::Integer, max_depth::Integer)
    t = _tri(geometry, box)
    t == 1 && return :full
    t == -1 && return :fictitious

    n_inside, n_outside = _corner_membership(geometry, box)
    (n_inside > 0 && n_outside > 0) && return :cut

    children = depth < max_depth ? _box_children(box) : nothing
    if children !== nothing
        result::Union{Nothing,Symbol} = nothing
        for child in children
            state = _classify_box(geometry, child, depth + 1, max_depth)
            result === nothing || result === state || return :cut
            result = state
        end
        return result::Symbol
    end

    return n_inside > 0 ? :full : :fictitious
end

"""
    classify_cell(physical, box) -> Symbol
    classify_cell(physical, box, cache) -> Symbol

Classify an axis-aligned `box` against `physical`'s CSG level set. Returns one of `:full` (entirely inside Ω), `:cut`
(crossed by ∂Ω), or `:fictitious` (entirely outside Ω). Uses the three-valued Lipschitz certificate first and falls back
to octree-bounded corner sampling; see the comment block on `_classify_box` for the full strategy.

The 3-arg form memoizes the result by `(box.lower, box.upper)` in a caller-owned `Dict`, shared between the cell-level
fold and the per-region integration dispatch.
"""
function classify_cell(physical::PhysicalDomain, box::AxisBox)
    return _classify_box(physical.geometry, box, 0, _effective_subcell_depth(physical, box))
end

# Memoisation cache for `classify_cell`, keyed by the box corner pair.
const _ClassifyCache{D,T} = Dict{Tuple{SVector{D,T},SVector{D,T}},Symbol}

function classify_cell(physical::PhysicalDomain, box::AxisBox{D,T}, cache::_ClassifyCache{D,T}) where {D,T}
    return get!(cache, (box.lower, box.upper)) do
        _classify_box(physical.geometry, box, 0, _effective_subcell_depth(physical, box))
    end
end

# ── Affine slices ─────────────────────────────────────────────────────────────

# Restrict a CSG tree to the affine slice `{x_k = v}`, one dimension down. Each leaf's callback is restricted by the
# callback-level `_restrict` in `src/implicit.jl` — which carries the leaf's Lipschitz constant through unchanged, the
# slice being an isometric embedding of the base into ℝᴰ — and the Boolean structure is rebuilt around the restricted
# leaves.
#
# Restricting the *leaves* rather than the scalar reconstruction is the whole point. `levelset_value` collapses the tree
# to max/min, which creases ∂Ω wherever two components meet, and a crease inside the sliced box costs the quadrature
# kernel its high order exactly as that function's docstring warns: on a wedge whose two faces meet strictly inside the
# sliced box, the leaf-wise slice integrates the area to round-off at moment order 4 — the kernel puts a breakpoint
# where the two leaves cross — while the collapsed one is 5.0e-3 off, and still 1.9e-3 at order 8. `test_physical.jl`
# pins that comparison.
_restrict(l::Leaf, k::Int, v) = Leaf(_restrict(l.f, k, v), l.lipschitz)
_restrict(n::AllOf, k::Int, v) = AllOf(map(p -> _restrict(p, k, v), n.parts))
_restrict(n::AnyOf, k::Int, v) = AnyOf(map(p -> _restrict(p, k, v), n.parts))
_restrict(n::Not, k::Int, v) = Not(_restrict(n.part, k, v))

# Ω restricted to the affine slice `⋂ⱼ {x_{kⱼ} = vⱼ}` named by `pins`, as a `PhysicalDomain` of dimension
# `D − length(pins)`. The struct carries no dimension parameter, so a slice is the same domain with a restricted tree,
# and every consumer that takes its `D` from the box it is handed — `classify_cell`, `_effective_subcell_depth`,
# [`moment_fit_rule`](@ref) — works on the slice unchanged. The integration knobs come along with it, which is what
# makes a face and the cells behind it resolve the same geometry at the same accuracy.
#
# `pins` are `(axis, coordinate)` pairs in DESCENDING axis order, and the order is load-bearing: each restriction drops
# its axis from the coordinate vector, so an axis index stays the one the caller named only while no lower-numbered axis
# has been dropped before it.
#
# Built by direct struct construction rather than through `physical_domain`, so that the fields survive verbatim —
# `keep_fictitious` and the compression included — with no keyword round-trip, and so that the slice never meets that
# constructor's argument validation, which is about what a caller may *ask* for and has nothing to say about a slice of
# an already-valid domain (its uncertified-leaf rejection, in particular, would fire on a tree the caller never wrote).
# The one field that does not survive is the cut rule: a caller's is a rule on a cell box in the model's own dimension,
# so a slice integrates its faces with [`implicit_rule`](@ref), which takes its dimension from the box it is handed.
function _restrict_domain(physical::PhysicalDomain, pins)
    geometry = foldl((g, pin) -> _restrict(g, pin[1], pin[2]), pins; init=physical.geometry)
    return PhysicalDomain(geometry, physical.alpha, physical.subcell_length_scale, physical.max_depth,
                          physical.moment_order_factor, physical.keep_fictitious, implicit_rule, physical.compression)
end
