# Implicit (level-set) quadrature by dimension reduction.
#
# Algorithm
#
#   Saye's high-order dimension-reduction quadrature for implicitly defined domains in hyperrectangles, generalised to a
#   Boolean combination of several smooth level sets ("multi-component"):
#
#     R. I. Saye, "High-order quadrature methods for implicitly defined
#     surfaces and volumes in hyperrectangles", SIAM J. Sci. Comput. 37
#     (2015) A993–A1019, doi:10.1137/140966290.
#     R. I. Saye, "High-order quadrature on multi-component domains implicitly
#     defined by multivariate polynomials", J. Comput. Phys. 448 (2022)
#     110720, doi:10.1016/j.jcp.2021.110720.
#
#   The same construction is the moment source for moment-fitting on implicit domains in
#
#     B. Müller, F. Kummer, M. Oberlack, "Highly accurate surface and volume
#     integration on implicit domains by means of moment-fitting", Int. J.
#     Numer. Methods Engng. 96 (2013) 512–528, doi:10.1002/nme.4569.
#
#   The pair leaves that make interior corners exact (`_PairLeaf` below) specialise the second paper's pairwise
#   elimination — every pair of level sets contributes the polynomial that eliminates the height variable between them
#   (their resultant) to the base problem — to leaves that are monotone along the height axis, where the eliminant is
#   one leaf evaluated at the other's fiber root.
#
#   Fiber roots are found by the Illinois variant of regula falsi,
#
#     M. Dowell, P. Jarratt, "A modified regula falsi method for computing the
#     root of an equation", BIT 11 (1971) 168–174, doi:10.1007/BF01934364.
#
#   This is a clean-room pure-Julia implementation written from the method descriptions, not a port of any specific
#   codebase.
#
# What it produces
#
#   Given a list of smooth level-set callbacks and a membership predicate over their signs (the CSG tree in
#   `src/physical.jl` supplies both), plus an axis-aligned box R, the kernel returns a quadrature rule {(xₖ, wₖ)} for
#   the cut volume Ω ∩ R. The rule is high-order accurate: exact for polynomials where every active level set is
#   graph-like over R (linear leaves ⇒ machine precision, including polytope corners), and spectrally accurate for
#   smooth curved boundaries away from turning points.
#
#   In the finite-cell moment-fit pipeline (`src/fcm.jl`) the volume rule plays two roles: the moments mₐ = ∫_{Ω∩R} ψₐ
#   dx are recovered as Σₖ wₖ ψₐ(xₖ), and the points xₖ double as the non-negative moment-fit candidate cloud.
#
# Method in one paragraph
#
#   Integration over Ω ∩ R is reduced one dimension at a time. Pick a "height" axis d* in which the active level sets
#   are graph-like (each varies monotonically, or is constant, along d* — a constant leaf, e.g. a perpendicular
#   half-space, simply partitions the base, which is what keeps polytope corners exact). The (D−1)-dimensional base
#   integral over the projection is built by the same recursion, partitioned by the leaves' restrictions to the two
#   faces x_{d*} = lo and x_{d*} = hi so the projection boundary and the change in fiber topology are resolved, and by
#   one pair leaf per two monotone leaves, whose zero set holds the projection of their crossing — where two fiber roots
#   swap order inside the box, a polytope corner for instance. At each base node a 1D Gauss rule is placed on the parts
#   of the fiber that lie in Ω, found by root-finding on the level sets and testing the membership predicate. When a box
#   is not graph-like it is bisected (a shallow octree above the kernel); at the depth floor the kernel reduces anyway —
#   the multi-root fiber scan still yields the correct region, only the base partition (and thus the order) degrades at
#   the unresolved feature. ∇φ is obtained by automatic differentiation (`ForwardDiff`); a caller whose leaves cannot
#   take `Dual` arguments overrides that with its own operator through the `grad` keyword, for which `_fd_gradient`
#   below is the central-difference choice. Nothing in the finite-cell path sets it, so leaves reaching
#   `moment_fit_rule` must be AD-capable. Pair leaves are the exception the kernel makes for itself: they are
#   differentiated by central differences (`_FDLeaf`) and never see a `Dual`.

# ── Coordinate insert / remove helpers ─────────────────────────────────────────

# Insert scalar `v` at axis `k` of an `M`-vector, yielding an `(M+1)`-vector. `v` may differ in type from `x`'s eltype —
# when automatic differentiation pushes `ForwardDiff.Dual` base coordinates through a reduced closure the spliced-in
# face value is still a plain real, so the result type is promoted.
@inline function _insert_axis(x::SVector{M,T}, k::Int, v) where {M,T}
    R = promote_type(T, typeof(v))
    return SVector{M + 1,R}(ntuple(i -> i < k ? R(x[i]) : (i == k ? R(v) : R(x[i - 1])), M + 1))
end

# Drop axis `k` from an `M`-vector, yielding an `(M−1)`-vector.
@inline function _remove_axis(x::SVector{M,T}, k::Int) where {M,T}
    return SVector{M - 1,T}(ntuple(i -> i < k ? x[i] : x[i + 1], M - 1))
end

# Project a box onto the base obtained by removing the height axis `k`.
_remove_axis(b::AxisBox{M,T}, k::Int) where {M,T} = AxisBox{M - 1,T}(_remove_axis(b.lower, k), _remove_axis(b.upper, k))

# ── Kernel context and gradient ────────────────────────────────────────────────

# Per-call configuration for the recursion: the shared 1D Gauss nodes/weights on [−1, 1], the gradient operator (AD or
# finite difference), the subdivision-depth budget, the number of scan samples used to bracket roots along a fiber, the
# region-scaled width below which a sub-interval is dropped as a numerically-coincident breakpoint, and the relative
# threshold below which a leaf's directional derivative counts as "flat" (constant) along an axis.
struct _QuadCtx{T,G}
    nodes::Vector{T}
    weights::Vector{T}
    gradient::G
    max_subdiv::Int
    root_samples::Int
    atol::T
    flat_tol::T
end

# Central-difference gradient operator for level sets through which automatic differentiation cannot be pushed. It is
# opt-in — a caller passes it as the `grad` keyword of `implicit_volume_quadrature`, and nothing falls back to it. The
# default step `h` is the cube-root-of-eps balance between truncation and round-off, scaled per axis by the coordinate
# magnitude; the kernel's own pair leaves pass a step relative to their box instead (`_gradient`).
function _fd_gradient(g, x::SVector{D,T}, h=cbrt(eps(T)) .* max.(one(T), abs.(x))) where {D,T}
    return SVector{D,T}(ntuple(D) do d
                            e = SVector{D,T}(ntuple(i -> i == d ? h[d] : zero(T), D))
                            (g(x + e) - g(x - e)) / (2h[d])
                        end)
end

# ── 1D safeguarded root finding ────────────────────────────────────────────────

# Root of `f` in a bracket `a < b` whose end values `fa`, `fb` have opposite signs, by the Illinois variant of regula
# falsi (Dowell & Jarratt 1971, see the file header). Each step evaluates the secant point of the bracket and keeps the
# sub-bracket over which the sign changes; when the same end is replaced twice in a row, the value at the end that
# stayed is halved, which removes regula falsi's one-sided stagnation and makes the convergence superlinear (order
# 3^(1/3) ≈ 1.442 at a simple root). On the kernel's smooth fibers that is several times fewer leaf evaluations per
# root than bisection.
#
# Safeguards, so the method keeps bisection's guarantees:
#
#   - the step is clamped strictly inside the bracket — a secant point that rounds onto an end moves one float inward,
#     which is also what ends the iteration in a step or two once the secant has found the root;
#   - a step is a bisection when the secant point is not a number, or when the last three steps have not halved the
#     bracket — the regime of a multiple root, where regula falsi, Illinois or not, converges only linearly;
#   - the side of `a` is fixed by its sign at entry, so a halved end value that underflows to zero cannot misread one.
#
# The bracket therefore shrinks at every evaluation, halves at least every third one, and always contains a sign
# change; the iteration stops at an exact zero or once the bracket is two adjacent floats, returning a point of it.
function _root(f, a::T, b::T, fa::T, fb::T) where {T}
    negative_a = fa < 0
    side = 0
    widths = (T(Inf), T(Inf), T(Inf))   # bracket widths before each of the last three steps
    while nextfloat(a) < b
        c = a - fa * (b - a) / (fb - fa)
        (isnan(c) || b - a > widths[1] / 2) && (c = (a + b) / 2)
        widths = (widths[2], widths[3], b - a)
        c = clamp(c, nextfloat(a), prevfloat(b))
        fc = f(c)
        fc == 0 && return c
        if (fc < 0) == negative_a
            a, fa = c, fc
            side == -1 && (fb /= 2)
            side = -1
        else
            b, fb = c, fc
            side == 1 && (fa /= 2)
            side = 1
        end
    end
    return (a + b) / 2
end

# ── Sign sampling and height-axis certificate ──────────────────────────────────

# A level-set callback carrying a Lipschitz constant `L` of `f` (`|f(x) − f(y)| ≤ L‖x − y‖`). The wrapper is itself
# callable and forwards to `f`, so every consumer in this file — `_choose_axis`, `_fiber_breaks`, `_emit_fiber!`, and
# the `ForwardDiff` gradient operator — keeps treating it as a plain callback; only `_sample_sign` dispatches on the
# wrapper to use the certificate below.
struct _LipschitzLeaf{F}
    f::F
    lipschitz::Float64
end

@inline (g::_LipschitzLeaf)(x) = g.f(x)

# Attach a Lipschitz bound to a level-set callback. `nothing` passes the callback through unchanged, which selects the
# uncertified sampling path.
_certified(f, ::Nothing) = f
function _certified(f, lipschitz::Real)
    lipschitz > 0 || throw(ArgumentError("lipschitz must be positive; got $lipschitz"))
    return _LipschitzLeaf(f, Float64(lipschitz))
end

# Sign of `g` if it is uniformly signed across the box center and its 2ᴰ corners, else `0` (a sign change was sampled).
# Used to prune leaves that do not vary on the box.
#
# This is a *heuristic*: a feature smaller than the sample spacing is invisible, and pruning every leaf makes the caller
# treat the box as uniform. The `_LipschitzLeaf` method below replaces it by a certificate wherever the caller supplied
# a Lipschitz constant.
function _sample_sign(g, U::AxisBox)
    sgn(v) = v < zero(v) ? -1 : (v > zero(v) ? 1 : 0)
    sref = sgn(g(center(U)))
    return sref != 0 && all(x -> sgn(g(x)) == sref, _box_vertices(U)) ? sref : 0
end

# Certified sign of a Lipschitz leaf on `U`, the same test the CSG cell classifier uses (`_certified_sign`: |f(c)| > L·r
# at the center `c`, with `r` the half-diagonal, proves `f` cannot change sign on `U`). Unlike point sampling this never
# prunes a leaf that does vary on the box, which is what keeps a geometric feature smaller than the sample spacing — a
# small hole well inside a cell, say — from vanishing into a silently full box rule.
#
# `L = Inf` (the default of [`leaf`](@ref)) would certify nothing and leave every leaf active; that mode falls back to
# the point-sampling heuristic above, matching the pre-certificate behavior.
function _sample_sign(g::_LipschitzLeaf, U::AxisBox)
    return isfinite(g.lipschitz) ? _certified_sign(g.f, g.lipschitz, U) : _sample_sign(g.f, U)
end

# The restriction of a level-set callback `g` to the affine slice `{x_k = v}`: the base-problem leaf of one
# dimension-reduction step, and through `src/physical.jl` the leaf of a boundary facet's slice of Ω. A callable record
# rather than a closure, so that the two face restrictions of one leaf can tell they are siblings (`_siblings`). The
# slice is an isometric embedding of the base into ℝᴰ (`‖_insert_axis(y, k, v) − _insert_axis(z, k, v)‖ = ‖y − z‖`), so
# a Lipschitz constant of `g` bounds the restriction just as tightly and is carried through unchanged.
struct _Slice{G,T}
    g::G
    k::Int
    v::T
end

(s::_Slice)(x) = s.g(_insert_axis(x, s.k, s.v))
_restrict(g, k::Int, v) = _Slice(g, k, v)
_restrict(g::_LipschitzLeaf, k::Int, v) = _LipschitzLeaf(_Slice(g.f, k, v), g.lipschitz)

# ── Pair leaves: crossings inside the box ──────────────────────────────────────

# The base problem of one reduction is partitioned by the leaves' restrictions to the faces x_k = lo and x_k = hi, which
# resolves every change of fiber topology on those faces but none strictly inside the box. Where two leaves gᵢ and gⱼ
# that are both monotone along k cross inside it, their fiber roots swap order and the fiber integral has a kink over
# the projection of {gᵢ = 0 = gⱼ}, with no base breakpoint there — so a polytope corner strictly inside a cut cell was
# integrated to the base rule's accuracy (about 1e-5 at the default orders), not exactly. Each such pair therefore adds
# one leaf to the base problem,
#
#     P(y) = gᵢ(y ⊕ₖ t̃ⱼ(y)),
#
# gᵢ evaluated at gⱼ's fiber root t̃ⱼ(y). P vanishes on the projection of the crossing; its other zeros are breakpoints
# the base already has, or harmless extra ones. For linear leaves P is the linear function that eliminates x_k between
# the two, so an interior corner becomes an exact breakpoint at every level of the reduction, and the recursion handles
# the nested pairs of 3D (Saye's pairwise elimination, see the file header).
#
# t̃ⱼ(y) is gⱼ's root on the fiber where its end values bracket one (`_root`); elsewhere it is the secant root of those
# end values — exact for a linear gⱼ, so P stays the eliminated linear function over the whole base — and the end of
# smaller |gⱼ| when that secant is not finite. Leaves are therefore also evaluated on the fiber's secant extension,
# outside the box; a `leaf` is defined on all coordinates, so that is within its contract.
#
# `gi` and `gj` are `Any`, as the kernel's leaf lists are: typed fields would nest a pair leaf's type once per level of
# the reduction (pairs of restricted pairs in 3D), and the compile work with it.
struct _PairLeaf{T}
    gi::Any
    gj::Any
    k::Int
    lo::T
    hi::T
end

function (p::_PairLeaf)(y)
    fj = t -> p.gj(_insert_axis(y, p.k, t))
    fa, fb = fj(p.lo), fj(p.hi)
    t = if (fa < 0 < fb) || (fb < 0 < fa)
        _root(fj, p.lo, p.hi, fa, fb)
    else
        s = (p.lo * fb - p.hi * fa) / (fb - fa)
        isfinite(s) ? s : (abs(fa) <= abs(fb) ? p.lo : p.hi)
    end
    return p.gi(_insert_axis(y, p.k, t))
end

# A callback the kernel differentiates by central differences at the fixed step `h` instead of through
# `ctx.gradient`: the wrapper on every pair leaf, so a pair leaf never sees a `Dual` (its root finding would have to be
# differentiated through) and never instantiates one. The step is a fixed fraction, cbrt(eps), of the fiber the pair
# leaf was built on, so the gradient does not depend on the geometry's length scale; its error, about 1e-10 relative,
# is far below the `flat_tol` (√eps) that the gradient only feeds. Pair leaves steer the axis choice and the pruning
# of base problems and are never integrated themselves. Restriction keeps the wrapper and the step.
struct _FDLeaf{F,T}
    f::F
    h::T
end

(g::_FDLeaf)(x) = g.f(x)
_restrict(g::_FDLeaf, k::Int, v) = _FDLeaf(_Slice(g.f, k, v), g.h)

# Whether two leaves are the restrictions of one leaf to the two faces of the previous reduction, which never get a
# pair leaf: where that leaf was monotone along the previous height axis its two face restrictions cannot cross (g = 0
# at both ends of one fiber contradicts monotonicity), and where it was flat they are one function, whose pair leaf
# vanishes identically and would scatter breakpoints over round-off noise — measured on the gate's napkin-ring octant,
# that made the volume rule 1.6× less accurate at every Gauss order. (A leaf that was bad along that axis, at the
# depth floor, could cross its sibling; the floor promises no order there.) `_parent` is the leaf a restriction was cut
# from, and `nothing` for a leaf that is not a restriction.
_parent(s::_Slice) = s.g
_parent(g::Union{_LipschitzLeaf,_FDLeaf}) = _parent(g.f)
_parent(_) = nothing
_siblings(a, b) = (p = _parent(a)) !== nothing && p === _parent(b)

# The gradient of one leaf at `x`: central differences for a pair leaf, the context's operator (`ForwardDiff`, or the
# caller's `grad`) for everything else.
_gradient(ctx::_QuadCtx, g, x) = ctx.gradient(g, x)
_gradient(::_QuadCtx, g::_FDLeaf, x::SVector) = _fd_gradient(g.f, x, map(_ -> g.h, x))

# Pick a height axis and report whether it is certifiably graph-like. For each axis k, every leaf is classified by
# sampling ∂g/∂x_k at the box center and 2ᴰ corners (`_axis_class`):
#
#   :monotone — consistent non-zero sign ⇒ at most one fiber root;
#   :flat     — |∂g/∂x_k| ≈ 0 at every sample ⇒ no fiber root (the leaf partitions the base instead; this is what makes
#               polytope corners exact, e.g. a half-space perpendicular to k);
#   :bad      — the sign of ∂g/∂x_k changes ⇒ possibly several fiber roots.
#
# The axis is the lexicographic maximum of (−#bad leaves, score), the first axis winning a tie, where the score is the
# smallest |∂g/∂x_k| over the monotone leaves (best conditioning) and 0 when no leaf is monotone along k (an axis along
# which nothing varies cannot root-find the interface). An axis with no bad leaf therefore always wins, and is the one
# of best conditioning; with none, the least-bad axis is the force-reduce choice at the subdivision floor. Returns
# `(k, usable, classes)`: `usable` says that no leaf is bad along `k`, and `classes` holds every leaf's class along it.
function _choose_axis(lsets, U::AxisBox{D}, ctx::_QuadCtx) where {D}
    samples = (center(U), _box_vertices(U)...)
    grads = [map(x -> _gradient(ctx, g, x), samples) for g in lsets]
    return _choose_axis(grads, ctx.flat_tol, Val(D))::Tuple{Int,Bool,Vector{Symbol}}
end

# The axis choice on the sampled gradients, behind a function barrier: `lsets` holds `Any`-typed callbacks, so the
# gradients' type is known only at run time, and the classification below then runs on concrete types. The call into
# the barrier is dynamic for the same reason, so its caller asserts the result type: without it `k` is inferred as
# `Any`, and `_implicit_quad` would dispatch `_emit_fiber!` dynamically, boxing its arguments, once per base node.
function _choose_axis(grads::Vector, flat_tol, ::Val{D}) where {D}
    k = argmax(1:D) do k
        nbad, score = 0, oftype(flat_tol, Inf)
        for gs in grads
            class, minmag = _axis_class(gs, k, flat_tol)
            nbad += class === :bad
            class === :monotone && (score = min(score, minmag))
        end
        return (-nbad, isinf(score) ? zero(score) : score)
    end
    classes = [first(_axis_class(gs, k, flat_tol)) for gs in grads]
    return k, !(:bad in classes), classes
end

# Class of one leaf along axis `k` from its gradient samples `gs`, and the smallest |∂g/∂x_k| over the samples that
# vary. A component counts as zero within `flat_tol` of the leaf's largest finite gradient norm (floored at 1, so a
# nearly constant leaf is not held to an absolute round-off scale). Non-finite samples are ignored: an SDF-like leaf
# evaluated at its own center yields ∇ = 0/0 = NaN, which must not poison the classification.
function _axis_class(gs, k::Int, flat_tol)
    tol = flat_tol * max(maximum(g -> all(isfinite, g) ? norm(g) : zero(flat_tol), gs), one(flat_tol))
    pos = any(g -> isfinite(g[k]) && g[k] > tol, gs)
    neg = any(g -> isfinite(g[k]) && g[k] < -tol, gs)
    minmag = minimum(g -> isfinite(g[k]) && abs(g[k]) > tol ? abs(g[k]) : oftype(tol, Inf), gs)
    return (pos && neg ? :bad : pos || neg ? :monotone : :flat), minmag
end

# ── Fiber rule emission ────────────────────────────────────────────────────────

# Sorted breakpoints of the fiber `t ↦ embed(t)` on `[lo, hi]`: the endpoints plus every sign change of every active
# leaf, located by scanning `ctx.root_samples` sub-intervals and solving each bracketed crossing with `_root`.
function _fiber_breaks(lsets, embed, lo::T, hi::T, ctx::_QuadCtx) where {T}
    breaks = T[lo, hi]
    M = ctx.root_samples
    for g in lsets
        tprev = lo
        fprev = g(embed(lo))
        for s in 1:M
            tcur = s == M ? hi : lo + (hi - lo) * (T(s) / M)
            fcur = g(embed(tcur))
            # A root landing exactly on a scan sample is itself the breakpoint. Any other crossing is a strict sign
            # change between neighbouring samples, read from the two signs: the product `fprev * fcur` underflows to
            # zero once both values are below about 1e-162 in magnitude, which would hide the root of a leaf that is
            # merely scaled down (Ω = {φ ≤ 0} does not depend on the scale of φ).
            if fcur == 0
                push!(breaks, tcur)
            elseif (fprev < 0 < fcur) || (fcur < 0 < fprev)
                push!(breaks, _root(t -> g(embed(t)), tprev, tcur, fprev, fcur))
            end
            tprev, fprev = tcur, fcur
        end
    end
    sort!(breaks)
    return breaks
end

# Place 1D Gauss rules on the parts of one fiber that lie in Ω and append the lifted points/weights (scaled by the base
# weight `wbase`). `embed(t)` lifts a fiber coordinate to the full D-dimensional point; `membership` is the Ω test.
function _emit_fiber!(pts::Vector{SVector{D,T}}, wts::Vector{T}, lsets, embed, lo::T, hi::T, wbase::T, membership,
                      ctx::_QuadCtx) where {D,T}
    breaks = _fiber_breaks(lsets, embed, lo, hi, ctx)
    nodes, gw = ctx.nodes, ctx.weights
    for i in 1:(length(breaks)-1)
        a, b = breaks[i], breaks[i + 1]
        b - a <= ctx.atol && continue
        mid = (a + b) / 2
        membership(embed(mid)) || continue
        half = (b - a) / 2
        for j in eachindex(nodes)
            push!(pts, embed(mid + half * nodes[j]))
            push!(wts, wbase * gw[j] * half)
        end
    end
    return nothing
end

# ── Full-box tensor rule ───────────────────────────────────────────────────────

# Tensor Gauss rule over the whole box, used when no leaf varies on the box and the box is inside Ω: the context's 1D
# rule on every axis (`_tensor_rule`), mapped in place into physical coordinates with the box Jacobian folded in.
function _full_box_rule(U::AxisBox{D,T}, ctx::_QuadCtx) where {D,T}
    rule = _tensor_rule(ntuple(_ -> (ctx.nodes, ctx.weights), Val(D)))
    map!(η -> reference_to_physical(U, η), rule.points, rule.points)
    rule.weights .*= volume(U) / convert(T, 2^D)
    return rule.points, rule.weights
end

# ── Core recursion ─────────────────────────────────────────────────────────────

# Bisect `U` into its 2ᴰ children (`_box_children`) and concatenate `recurse(child)` over them. This is the "not
# graph-like ⇒ subdivide" branch of the volume recursion; `recurse` is the per-child continuation. `nothing` when `U` is
# too small to bisect, which the caller treats as the depth floor.
function _subdivide_and_collect(recurse, U::AxisBox{D,T}) where {D,T}
    children = _box_children(U)
    children === nothing && return nothing
    pts = SVector{D,T}[]
    wts = T[]
    for child in children
        cp, cw = recurse(child)
        append!(pts, cp)
        append!(wts, cw)
    end
    return pts, wts
end

# The leaves of the base problem of one reduction along axis `k` of a box spanning `[lo, hi]` on it: every active leaf's
# restrictions to the two faces x_k = lo and x_k = hi, which resolve the projection boundary and every change of fiber
# topology on the faces, then one pair leaf per two leaves monotone along `k` that are not siblings, for the crossings
# inside the box (`_PairLeaf`, `_siblings`). `classes` are the leaves' classes along `k` (`_choose_axis`).
function _base_leaves(active, classes, k::Int, lo::T, hi::T) where {T}
    base = Any[]
    for g in active
        push!(base, _restrict(g, k, lo), _restrict(g, k, hi))
    end
    monotone = active[classes .=== :monotone]
    h = cbrt(eps(T)) * (hi - lo)
    for i in eachindex(monotone), j in (i+1):lastindex(monotone)
        _siblings(monotone[i], monotone[j]) && continue
        push!(base, _FDLeaf(_PairLeaf{T}(monotone[i], monotone[j], k, lo, hi), h))
    end
    return base
end

# Recursively integrate over Ω ∩ U. `lsets` is the list of leaf callbacks to partition by; `membership(x)` tests x ∈ Ω
# at a full-dimensional point. The original problem starts from the domain's leaves and its membership predicate; each
# reduction passes the leaves' face restrictions as a partition-only base problem (`Returns(true)` membership) and
# applies the real membership in its own fiber loop. Stages:
#
#   1. Prune leaves that do not vary on U (a partition-only optimisation; `membership` re-evaluates the full predicate,
#      so this is safe). With no varying leaf left, U is uniform: full tensor rule if inside, else empty.
#   2. Base case D = 1: emit the inside parts of the single axis.
#   3. Choose the height axis. If not certifiably graph-like and budget remains, bisect and recurse; at the floor (or
#      on a box too small to bisect), force-reduce on the least-bad axis.
#   4. Reduce: build the base problem on U with the height axis removed (every leaf's two face restrictions and one pair
#      leaf per two leaves monotone along the axis that are not siblings, partition-only) and recurse; then sweep the
#      base nodes, emitting a fiber rule at each.
function _implicit_quad(lsets, membership, U::AxisBox{D,T}, ctx::_QuadCtx, depth::Int) where {D,T}
    active = Any[]
    for g in lsets
        _sample_sign(g, U) == 0 && push!(active, g)
    end
    if isempty(active)
        return membership(center(U)) ? _full_box_rule(U, ctx) : (SVector{D,T}[], T[])
    end

    if D == 1
        pts = SVector{1,T}[]
        wts = T[]
        _emit_fiber!(pts, wts, active, t -> SVector{1,T}(t), U.lower[1], U.upper[1], one(T), membership, ctx)
        return pts, wts
    end

    k, usable, classes = _choose_axis(active, U, ctx)
    if !usable && depth < ctx.max_subdiv
        rule = _subdivide_and_collect(box -> _implicit_quad(active, membership, box, ctx, depth + 1), U)
        rule === nothing || return rule
    end

    # Reduce one dimension: partition-only base, then a fiber rule per base node.
    facelo, facehi = U.lower[k], U.upper[k]
    base = _base_leaves(active, classes, k, facelo, facehi)
    bpts, bwts = _implicit_quad(base, Returns(true), _remove_axis(U, k), ctx, 0)

    pts = SVector{D,T}[]
    wts = T[]
    for n in eachindex(bpts)
        xb, wb = bpts[n], bwts[n]
        _emit_fiber!(pts, wts, active, t -> _insert_axis(xb, k, t), facelo, facehi, wb, membership, ctx)
    end
    return pts, wts
end

# ── Public entry points ────────────────────────────────────────────────────────

# `scale` is the region's largest edge length; the breakpoint-drop tolerance `atol` is taken relative to it so the
# kernel behaves identically under a rescaling of the geometry. `flat_tol` is already relative (to the gradient
# magnitude) inside `_choose_axis`.
function _quad_ctx(::Type{T}, gauss_points::Int, grad, max_subdiv::Int, scale::Real) where {T}
    rawnodes, rawweights = gausslegendre(gauss_points)
    gradient = grad === nothing ? (g, x) -> ForwardDiff.gradient(g, x) : grad
    return _QuadCtx(T.(rawnodes), T.(rawweights), gradient, max_subdiv, max(6, 2 * gauss_points),
                    eps(T)^(3 // 4) * T(scale), sqrt(eps(T)))
end

# Pair each leaf callback with its Lipschitz bound, or pass the callbacks through untouched when the caller opted out.
# The length check is worth an explicit error: a silently truncated `zip` would certify the wrong leaves.
function _certified_leaves(leaves::AbstractVector, lipschitz)
    lipschitz === nothing && return collect(Any, leaves)
    length(lipschitz) == length(leaves) ||
        throw(DimensionMismatch("lipschitz has $(length(lipschitz)) entries but there are " *
                                "$(length(leaves)) leaves"))
    return Any[_certified(f, L) for (f, L) in zip(leaves, lipschitz)]
end

# The leaf list with every repeat dropped, first occurrences in order. A CSG tree may name one leaf twice — (a ∩ b) ∪
# (a ∩ c) for a ∩ (b ∪ c) — and `_leaves` lists it once per occurrence. A repeat adds nothing to the partition, but two
# copies of one leaf would get a pair leaf (`_base_leaves`) that vanishes identically, whose round-off scatters
# breakpoints over the base, and the copies' pair leaves with a third leaf would pair up again one reduction further
# down. Identity (`===`) is the test, so wrappers of one callback with one Lipschitz constant are the same leaf.
function _distinct(leaves::Vector{Any})
    distinct = Any[]
    for g in leaves
        any(h -> h === g, distinct) || push!(distinct, g)
    end
    return distinct
end

"""
    implicit_volume_quadrature(leaves, membership, region; gauss_points,
                               grad=nothing, max_subdiv=4, lipschitz=nothing)
        -> (points, weights)
    implicit_volume_quadrature(phi, region; gauss_points, …) -> (points, weights)

Return a quadrature rule `{(xₖ, wₖ)}` for the cut volume `Ω ∩ region`, so that

    ∫_{Ω ∩ region} f(x) dx  ≈  Σₖ wₖ f(xₖ)

with the points in physical coordinates and the box Jacobian folded into the weights. `leaves` is a vector of smooth
level-set callbacks and `membership` is a predicate `x -> Bool` testing `x ∈ Ω` (Saye's multi-component
dimension-reduction quadrature; see the file header). The single-argument form takes one callback `phi` with
`Ω = {φ ≤ 0}`.

The rule is exact for polynomials where every active leaf is graph-like over `region` (linear leaves ⇒ machine
precision, including polytope corners) and spectrally accurate for smooth curved boundaries.

Arguments:

  - `gauss_points`: number of 1D Gauss–Legendre points per inside fiber sub-interval. For a *tilted* cut the
    inside-interval endpoints are linear in the base coordinates, so each fiber reduction raises the integrated degree
    by one; integrating tensor moments of total order `p` over a beveled corner needs `q ≈ (p + D) / 2`, not `≈ p / 2`
    (the finite-cell pipeline picks this via `_implicit_gauss_points`).
  - `grad`: optional gradient operator `(g, x) -> SVector` overriding the `ForwardDiff` default; pass `_fd_gradient`
    from this file for non-AD callbacks. The finite-cell path never sets it, so this keyword is reachable only by
    calling the kernel directly.
  - `max_subdiv`: subdivision-depth budget for non-graph-like cells. Beyond it the kernel force-reduces (the region
    stays correct; order degrades only at the unresolved feature).
  - `lipschitz`: optional Lipschitz constants `Lᵢ` of the leaves (`|fᵢ(x) − fᵢ(y)| ≤ Lᵢ‖x − y‖`) — a vector matching
    `leaves` in the multi-component form, a scalar in the single-`phi` form, and `nothing` (default) to opt out. `1.0`
    is the constant of a true signed-distance function. See the sampling note below for what supplying it certifies.

Feature detection is by point sampling unless `lipschitz` is supplied: a leaf is pruned where it is uniformly signed at
the box center and corners, and fiber roots are bracketed by a bounded scan. A feature — or a pair of roots — smaller
than that sampling can be missed, and a box in which *every* leaf is wrongly pruned is integrated as if it were
uniformly inside or outside Ω, so without `lipschitz` the caller must size `region` below the smallest geometric
feature.

Passing `lipschitz` replaces the pruning heuristic by the certificate `|fᵢ(c)| > Lᵢ·r` at the box center `c` with `r`
the half-diagonal — the same test the CSG cell classifier uses (`classify_cell`). It never prunes a leaf that does vary
on the box, so a sub-`region` feature can no longer disappear; the fiber scan resolution is unaffected. The certificate
is the *only* thing that makes the kernel safe on a `region` coarser than the geometry, at the cost of keeping more
leaves active per box.
"""
function implicit_volume_quadrature(leaves::AbstractVector, membership, region::AxisBox{D,T}; gauss_points::Int,
                                    grad=nothing, max_subdiv::Int=4, lipschitz=nothing) where {D,T}
    ctx = _quad_ctx(T, gauss_points, grad, max_subdiv, maximum(region.upper - region.lower))
    return _implicit_quad(_distinct(_certified_leaves(leaves, lipschitz)), membership, region, ctx, 0)
end

function implicit_volume_quadrature(phi, region::AxisBox{D,T}; gauss_points::Int, grad=nothing, max_subdiv::Int=4,
                                    lipschitz=nothing) where {D,T}
    return implicit_volume_quadrature(Any[phi], x -> phi(x) <= 0, region; gauss_points, grad, max_subdiv,
                                      lipschitz=lipschitz === nothing ? nothing : Any[lipschitz])
end
