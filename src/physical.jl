# ── CSG level-set tree ────────────────────────────────────────────────────────
#
# An immersed domain Ω is described by a Boolean combination of smooth level
# sets ("leaves"), rather than a single scalar φ. Carrying the constituent
# leaves separately keeps every boundary piece smooth, so the implicit
# quadrature kernel (`src/implicit.jl`) stays high-order across
# creases and corners that a single `max`/`min` level set would turn into a
# non-differentiable kink (Saye's multi-component construction: R. I. Saye,
# "High-order quadrature on multi-component domains implicitly defined by
# multivariate polynomials", J. Comput. Phys. 448 (2022) 110720).

"""
    LevelSet

Abstract supertype of the CSG nodes describing an immersed domain Ω. The
concrete nodes are a [`leaf`](@ref) (a single smooth level set with
`Ω = {f ≤ 0}`) and the Boolean combinators built by `intersect`, `union`,
`setdiff`, and [`complement`](@ref). A `LevelSet` is passed to
[`physical_domain`](@ref).
"""
abstract type LevelSet end

"""
    Leaf(f, lipschitz)

A single smooth level set `f` with `Ω_leaf = { x : f(x) ≤ 0 }`. `lipschitz` is
a Lipschitz constant `L` of `f` (`|f(x) − f(y)| ≤ L‖x − y‖`); `Inf` disables
the cheap sign certificate in the cell classifier and in the implicit
quadrature kernel, and forces corner sampling in both. Construct with
[`leaf`](@ref).
"""
struct Leaf{F} <: LevelSet
    f::F
    lipschitz::Float64
end

# n-ary intersection (Ω = ⋂ parts; membership = inside *all* parts) and union
# (Ω = ⋃ parts; membership = inside *any* part). `Not` is set complement.
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

Wrap a scalar level-set callback `f` (on `SVector{D,T}` coordinates) as a CSG
leaf with `Ω_leaf = { x : f(x) ≤ 0 }`. `f` must accept `ForwardDiff.Dual`
arguments so the quadrature kernel can take its gradient (pass an
AD-compatible closure, or supply a finite-difference gradient downstream).
`lipschitz` is the Lipschitz constant used by the sign certificate of both the
cell classifier and the implicit quadrature kernel; `1.0` for a true
signed-distance leaf, `Inf` to rely on corner sampling. Supplying it is what
lets a feature smaller than a mesh cell survive: under `Inf` the kernel can
prune such a leaf on the cell box and integrate the cut cell as if it were
entirely inside Ω.
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

CSG combinators on level sets (the `Base` set operations are extended for the
[`LevelSet`](@ref) type). `intersect` builds `Ω = ⋂ {fᵢ ≤ 0}`, `union` builds
`Ω = ⋃ {fᵢ ≤ 0}`, `setdiff(a, b) = a ∩ complement(b)` (e.g. an annulus as a
disk minus a disk), and `complement` flips inside and outside. Bare callable
arguments are auto-wrapped as default [`leaf`](@ref)s. The first argument must
be a `LevelSet`, so wrap at least one operand with `leaf` (e.g.
`intersect(leaf(f), g)`).
"""
Base.intersect(a::LevelSet) = a
Base.intersect(a::LevelSet, b, rest...) = AllOf((a, _as_levelset(b), map(_as_levelset, rest)...))
Base.union(a::LevelSet) = a
Base.union(a::LevelSet, b, rest...) = AnyOf((a, _as_levelset(b), map(_as_levelset, rest)...))
Base.setdiff(a::LevelSet, b) = AllOf((a, Not(_as_levelset(b))))

"""
    complement(a) -> LevelSet

Set complement of a level set: inside and outside are swapped
(`Ω ↦ ℝᴰ ∖ Ω`). `a` is a [`LevelSet`](@ref) or a bare callable (auto-wrapped as
a default [`leaf`](@ref)). One of the CSG combinators alongside `intersect`,
`union`, and `setdiff`; pass the result to [`physical_domain`](@ref).
"""
complement(a) = Not(_as_levelset(a))

# ── Tree evaluation ───────────────────────────────────────────────────────────

# Membership test x ∈ Ω. A leaf is inside where f ≤ 0; intersection/union/
# complement combine with all/any/not. Short-circuiting is fine here (no
# index bookkeeping), unlike the classifier's three-valued walk below.
_inside(l::Leaf, x) = l.f(x) <= 0
_inside(n::AllOf, x) = all(p -> _inside(p, x), n.parts)
_inside(n::AnyOf, x) = any(p -> _inside(p, x), n.parts)
_inside(n::Not, x) = !_inside(n.part, x)

# A single scalar that is ≤ 0 exactly on Ω, reconstructed from the tree
# (intersection → max, union → min, complement → negate). This is the
# representation the kernel deliberately avoids for *integration* (its kinks
# are why we carry leaves separately), but it is the right thing to *display*:
# a contour at 0 reproduces ∂Ω. Used by VTK export and nowhere in the hot path.
_value(l::Leaf, x) = l.f(x)
_value(n::AllOf, x) = maximum(p -> _value(p, x), n.parts)
_value(n::AnyOf, x) = minimum(p -> _value(p, x), n.parts)
_value(n::Not, x) = -_value(n.part, x)

# Collect the leaves in a stable depth-first order; the quadrature kernel
# partitions by these and the classifier certifies each one's sign.
_collect_leaves!(acc, l::Leaf) = (push!(acc, l); acc)
_collect_leaves!(acc, n::AllOf) = (foreach(p -> _collect_leaves!(acc, p), n.parts); acc)
_collect_leaves!(acc, n::AnyOf) = (foreach(p -> _collect_leaves!(acc, p), n.parts); acc)
_collect_leaves!(acc, n::Not) = _collect_leaves!(acc, n.part)
_leaves(g::LevelSet) = _collect_leaves!(Leaf[], g)

# ── PhysicalDomain ────────────────────────────────────────────────────────────

"""
    PhysicalDomain{G,T,Q}(geometry, alpha, subcell_length_scale, max_depth,
                          moment_order_factor, target_residual, keep_fictitious,
                          cut_quadrature)

CSG level-set description of a physical domain Ω for finite-cell-style immersed
integration. `Ω = { x : x ∈ geometry }`, where `geometry` is a
[`LevelSet`](@ref) tree of smooth leaves combined by `intersect`/`union`/
`setdiff`/`complement` (a single [`leaf`](@ref) is the degenerate case, with
the familiar `Ω = {φ ≤ 0}`).

Fields:

  - `geometry::G`: the CSG level-set tree. Each leaf carries its own Lipschitz
    constant; the tree's Boolean structure defines membership.
  - `alpha::T`: fictitious-region stabilization weight (α-FCM). `0` is the
    strict cut path; `> 0` enriches every **cut** cell's quadrature with the
    α-scaled full-cell rule, so cut-cell dofs whose basis support lies in the
    fictitious part still get a well-posed contribution (see
    `_build_region_quadrature` in `intersections.jl`). Fully-fictitious cells
    (support entirely outside Ω) are **dropped from the dof layout regardless of
    α** — the α term stabilizes cuts, not whole fictitious cells.
  - `keep_fictitious::Bool`: opt into the classic α-FCM treatment where
    fully-fictitious cells are *kept* active with α-scaled full-cell quadrature
    instead of being dropped. `false` (default) drops them, which keeps the
    system lean and the physical domain clean for post-processing; `true`
    requires `alpha > 0` to be meaningful.
  - `subcell_length_scale::T`: target box size, in physical units, for the
    binary subdivision both the classifier and the moment-fit kernel share. A
    box is bisected until its largest axis extent is `≤ subcell_length_scale`
    (capped by `max_depth`); `_effective_subcell_depth` turns the two into a
    single depth budget. It feeds exactly two consumers, both subdivision
    budgets — not an octree moment grid:
    (1) the cut/full/fictitious classifier, which bisects only when the
    Lipschitz certificate and corner sampling cannot decide a box, so this is a
    geometry-*robustness* knob (resolving thin or near-tangent features); and
    (2) the implicit-quadrature kernel's fallback subdivision on *non-graph-like*
    cut cells (a leaf with a turning point / multiple roots inside the box).
    Smooth, graph-like cut cells get exact, depth-independent moments from the
    kernel (see `src/fcm.jl`) and are never subdivided, so on those cells moment
    accuracy does not depend on this scale at all. Pick it relative to the
    smallest geometric feature you must classify cleanly.
  - `max_depth::Int`: hard cap on the subdivision depth of both consumers above.
  - `moment_order_factor::Int`: multiplier on the NNMF moment-fit basis order
    per axis (`factor × max(level.order)` over the region's parents). `2`
    (default) integrates trial × test products exactly; `1` halves the basis.
  - `target_residual::T`: NNMF L² residual the moment-fit aims for in cut
    regions. Default `1e-6`; the exact kernel reaches far below it, so this
    only bounds the conditioning retry.
  - `cut_quadrature::Q`: the cut-cell quadrature rule. `nothing` (the default,
    `Q === Nothing`) is the package's own non-negative moment fit; any other
    value is a callable that replaces it on **cut** regions only. See
    "Custom cut-cell quadrature" below for the contract it must honour.

# Custom cut-cell quadrature

`cut_quadrature` is the extension point for integrating a cut cell by some
other scheme — a space-tree/octree rule, a tessellation, a rule read from a
file. It is called once per distinct `(region box, moment order)` pair, with
the signature

    (physical, box, moment_order) -> (points, weights, residual, status)

and must return the same 4-tuple [`moment_fit_rule`](@ref) returns:
`points::Vector{SVector{D,T}}`, `weights::Vector{T}`, a scalar `residual`,
and a `Symbol` status (report `residual = 0` when a rule has no notion of a fit
residual). Only `:full`, `:fictitious`, and the fictitious fold are unaffected
— every other part of the pipeline treats the returned rule exactly as it
treats a fitted one. Four properties of that pipeline are the caller's
responsibility:

 1. **Points are in physical coordinates.** `points[q] ∈ box ⊂ ℝᴰ` and
    `weights[q]` is its physical weight, so `Σ w_q ≈ vol(box ∩ Ω)` — the same
    convention [`moment_fit_rule`](@ref) returns. The per-box rule cache and
    the mapping into the region's `[−1, 1]ᴰ` reference frame downstream both
    assume it; handing back reference-frame points silently integrates the
    wrong geometry.
 2. **Weights must be non-negative.** The α-FCM blend below and the
    fictitious fold in `src/mesh.jl` both assume `w_q ≥ 0`, and a negative
    weight can make an otherwise SPD form indefinite. The moment fit
    guarantees this by construction (NNLS); a custom rule must guarantee it
    itself.
 3. **An empty rule produces silent zero stiffness.** A rule that returns no
    points for a sliver cell contributes nothing to the tangent, and *no
    diagnostic is raised*: the region is kept, tagged, and integrates to
    zero, so every dof supported only there ends up with a zero row and the
    system is singular. This is not hypothetical — it is what makes an
    aggressive space-tree configuration singular at `alpha = 0`. A rule that
    knows it found nothing should return `status === :empty`, which routes
    the region to `:cut_failed` (α = 0) or `:cut_alpha_failed` (α > 0) exactly
    as an empty moment fit would, and so at least reaches
    `diagnostics(...).fit_failure_count`.
 4. **α blending still applies**, downstream of the rule. Under `alpha > 0` a
    cut region carries `(1 − α)·(custom rule) ∪ α·(full-cell tensor rule)`,
    just as it would with the moment fit, so a custom rule inherits α-FCM
    rather than replacing it.

The rule must also be a deterministic function of its three arguments: results
are memoised per `(region box, moment order)` for the lifetime of a plan
build, so a rule reading mutable state outside its arguments is captured at
its first answer for a box and silently reused. Put every knob in the callable
itself — that is what makes the rule reproducible and thread-safe. `residual`
is reported verbatim through `diagnostics(model).moment_fit_residual_max`.

A region built from a custom rule is tagged `:cut_custom` (see
`_build_region_quadrature` in `src/intersections.jl`) — one kind for every
custom rule, whatever status symbol it returns, `:empty` excepted as in point
3 above. `_cut_region_stats` counts `:cut_custom` as a cut region and **not**
as a fit failure, since there is no fit to fail. The rule is stored on the domain
rather than in a global, so two models in one session can use different rules
and nothing about the choice is process-wide or thread-shared.

Construct via [`physical_domain`](@ref).
"""
struct PhysicalDomain{G<:LevelSet,T<:Real,Q}
    geometry::G
    alpha::T
    subcell_length_scale::T
    max_depth::Int
    moment_order_factor::Int
    target_residual::T
    keep_fictitious::Bool
    cut_quadrature::Q
end

"""
    physical_domain(geometry; lipschitz=Inf, alpha=0.0, keep_fictitious=false,
                    subcell_length_scale, max_depth=8, moment_order_factor=2,
                    target_residual=1e-6, cut_quadrature=nothing)

Construct a [`PhysicalDomain`](@ref). `geometry` is either a [`LevelSet`](@ref)
CSG tree (built from [`leaf`](@ref) and `intersect`/`union`/`setdiff`/
[`complement`](@ref)) or a bare scalar callable `φ`, which is auto-wrapped as a
single leaf with `Ω = {φ ≤ 0}` — preserving the single-level-set ergonomics
`physical_domain(φ; lipschitz=…)`. The `lipschitz` keyword applies only to that
auto-wrapped single leaf; for a CSG tree, set each leaf's Lipschitz constant on
the `leaf(...)` call instead.

`subcell_length_scale` is the **required** classifier accuracy / subdivision
scale (see the [`PhysicalDomain`](@ref) docstring). `alpha = 0` is the
strict-cut path; `alpha > 0` enables α-FCM. `keep_fictitious = true` retains
fully-fictitious cells as active dofs (the classic α-FCM fill) and therefore
requires `alpha > 0` — pairing it with `alpha = 0` leaves those cells without
quadrature and is rejected. `moment_order_factor` and `target_residual` tune
the NNMF moment fit.

`cut_quadrature = nothing` (default) integrates cut cells with the package's
non-negative moment fit. Passing a callable
`(physical, box, moment_order) -> (points, weights, residual, status)`
replaces that rule on cut regions and nothing else; the contract it must
honour — physical-frame points, non-negative weights, the silent-zero-stiffness
hazard of an empty rule, and the α blend that still applies downstream — is
spelled out under "Custom cut-cell quadrature" in the [`PhysicalDomain`](@ref)
docstring.
"""
function physical_domain(geometry; lipschitz::Real=Inf, alpha::Real=0.0,
                         keep_fictitious::Bool=false, subcell_length_scale::Real,
                         max_depth::Integer=8, moment_order_factor::Integer=2,
                         target_residual::Real=1.0e-6, cut_quadrature=nothing)
    alpha >= 0 || throw(ArgumentError("alpha must be ≥ 0; got $alpha"))
    !(keep_fictitious && iszero(alpha)) ||
        throw(ArgumentError("keep_fictitious=true requires alpha > 0: fully-fictitious cells " *
                            "kept active under the strict-cut (alpha=0) path receive no " *
                            "quadrature and would make the system singular"))
    subcell_length_scale > 0 ||
        throw(ArgumentError("subcell_length_scale must be > 0; got $subcell_length_scale"))
    max_depth >= 0 || throw(ArgumentError("max_depth must be ≥ 0; got $max_depth"))
    moment_order_factor >= 1 ||
        throw(ArgumentError("moment_order_factor must be ≥ 1; got $moment_order_factor"))
    target_residual > 0 ||
        throw(ArgumentError("target_residual must be positive; got $target_residual"))
    # A bare callable becomes a single leaf, honouring the `lipschitz` keyword;
    # an explicit CSG tree carries per-leaf Lipschitz constants already.
    g = geometry isa LevelSet ? geometry : leaf(geometry; lipschitz=lipschitz)
    T = promote_type(typeof(float(alpha)), typeof(float(subcell_length_scale)),
                     typeof(float(target_residual)))
    return PhysicalDomain{typeof(g),T,typeof(cut_quadrature)}(g, T(alpha),
                                                              T(subcell_length_scale),
                                                              Int(max_depth),
                                                              Int(moment_order_factor),
                                                              T(target_residual), keep_fictitious,
                                                              cut_quadrature)
end

"""
    levelset_value(physical, x) -> Real

Scalar reconstruction of the domain's level set at `x` that is `≤ 0` exactly on
Ω (intersection → max of children, union → min, complement → negate, leaf →
`f(x)`). Intended for visualization — a contour at level 0 reproduces ∂Ω — not
for integration, which uses the separate leaves.
"""
levelset_value(physical::PhysicalDomain, x) = _value(physical.geometry, x)

# ── Box certificates ──────────────────────────────────────────────────────────

# Number of binary-subdivision levels needed to drive `box`'s largest axis
# extent down to `physical.subcell_length_scale`, capped at `physical.max_depth`.
# Shared by the classifier and the quadrature kernel's subdivision budget so the
# resolution contract is uniform. Boxes already at or below the scale return 0.
function _effective_subcell_depth(physical::PhysicalDomain, box::AxisBox)
    max_extent = maximum(box.upper - box.lower)
    max_extent <= physical.subcell_length_scale && return 0
    return min(physical.max_depth, ceil(Int, log2(max_extent / physical.subcell_length_scale)))
end

# Half-diagonal of an axis-aligned box — the radius of the smallest enclosing
# ball. Used by the per-leaf Lipschitz certificate: for a Lipschitz `f` and any
# point `y` in the box, |f(y) − f(c)| ≤ L·r with `c` the center and `r` this
# radius, so a center sign whose magnitude exceeds `L·r` is uniform on the box.
_half_diagonal(b::AxisBox) = norm(b.upper - b.lower) / 2

# ── Cell classifier ───────────────────────────────────────────────────────────

# Three-valued (Kleene) membership certificate over the CSG tree: returns
# `+1` if the box is certainly inside Ω, `-1` if certainly outside, `0` if
# undetermined. Each leaf is certified by its Lipschitz bound (|f(c)| > L·r ⇒
# uniform sign); the Boolean combinators propagate certainty without
# enumerating sign patterns — e.g. a union is certainly inside as soon as one
# part is, certainly outside only if every part is. This is what lets a cut
# cell of one component be classified `:full`/`:fictitious` when another
# component dominates it.
function _tri(l::Leaf, box::AxisBox)
    fc = l.f(center(box))
    thr = l.lipschitz * _half_diagonal(box)
    return fc < -thr ? 1 : (fc > thr ? -1 : 0)
end
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

# Tally Ω-membership at the box center and its 2ᴰ corners. Mixed counts prove
# ∂Ω crosses the box; corners — rather than interior points — are the natural
# sample set for an axis-aligned box and double as the child vertices when the
# octree recurses.
function _corner_membership(geometry::LevelSet, box::AxisBox{D,T}) where {D,T}
    inside = _inside(geometry, center(box)) ? 1 : 0
    outside = 1 - inside
    for ci in CartesianIndices(ntuple(_ -> 0:1, D))
        x = SVector{D,T}(ntuple(d -> ci.I[d] == 0 ? box.lower[d] : box.upper[d], D))
        _inside(geometry, x) ? (inside += 1) : (outside += 1)
    end
    return inside, outside
end

# Recursive classifier returning one of `:full`, `:cut`, `:fictitious`.
# Strategy, cheapest first:
#
#   1. Three-valued Lipschitz certificate on the CSG tree. A definite verdict
#      settles `:full`/`:fictitious` with no sampling.
#   2. Corner membership. Mixed inside/outside proves ∂Ω crosses the box.
#   3. Octree recursion. Signs agree but the certificate did not fire; bisect
#      into 2ᴰ children and recurse. Any `:cut` child, or any disagreement
#      between `:full`/`:fictitious` children, makes the parent `:cut`.
#   4. Depth budget exhausted with agreeing samples: trust the consensus.
#
# The single-leaf case reduces exactly to the Lipschitz-then-corner strategy
# the package used before CSG.
function _classify_box(geometry::LevelSet, box::AxisBox{D,T}, depth::Integer,
                       max_depth::Integer) where {D,T}
    t = _tri(geometry, box)
    t == 1 && return :full
    t == -1 && return :fictitious

    n_inside, n_outside = _corner_membership(geometry, box)
    (n_inside > 0 && n_outside > 0) && return :cut

    if depth < max_depth
        c = center(box)
        result::Union{Nothing,Symbol} = nothing
        for ci in CartesianIndices(ntuple(_ -> 0:1, D))
            child_lower = SVector{D,T}(ntuple(d -> ci.I[d] == 0 ? box.lower[d] : c[d], D))
            child_upper = SVector{D,T}(ntuple(d -> ci.I[d] == 0 ? c[d] : box.upper[d], D))
            child_state = _classify_box(geometry, AxisBox{D,T}(child_lower, child_upper), depth + 1,
                                        max_depth)
            if result === nothing
                result = child_state
            elseif result !== child_state
                return :cut
            end
        end
        return result::Symbol
    end

    return n_inside > 0 ? :full : :fictitious
end

"""
    classify_cell(physical, box) -> Symbol
    classify_cell(physical, box, cache) -> Symbol

Classify an axis-aligned `box` against `physical`'s CSG level set. Returns one
of `:full` (entirely inside Ω), `:cut` (crossed by ∂Ω), or `:fictitious`
(entirely outside Ω). Uses the three-valued Lipschitz certificate first and
falls back to octree-bounded corner sampling; see the comment block on
`_classify_box` for the full strategy.

The 3-arg form memoizes the result by `(box.lower, box.upper)` in a
caller-owned `Dict`, shared between the cell-level fold and the per-region
integration dispatch.
"""
function classify_cell(physical::PhysicalDomain, box::AxisBox)
    return _classify_box(physical.geometry, box, 0, _effective_subcell_depth(physical, box))
end

# Memoisation cache for `classify_cell`, keyed by the box corner pair.
const _ClassifyCache{D,T} = Dict{Tuple{SVector{D,T},SVector{D,T}},Symbol}

function classify_cell(physical::PhysicalDomain, box::AxisBox{D,T},
                       cache::_ClassifyCache{D,T}) where {D,T}
    return get!(cache, (box.lower, box.upper)) do
        _classify_box(physical.geometry, box, 0, _effective_subcell_depth(physical, box))
    end
end
