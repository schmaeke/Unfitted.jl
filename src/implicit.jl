# Implicit (level-set) quadrature by dimension reduction.
#
# Algorithm
#
#   Saye's high-order dimension-reduction quadrature for implicitly defined
#   domains in hyperrectangles, generalised to a Boolean combination of several
#   smooth level sets ("multi-component"):
#
#     R. I. Saye, "High-order quadrature methods for implicitly defined
#     surfaces and volumes in hyperrectangles", SIAM J. Sci. Comput. 37
#     (2015) A993–A1019, doi:10.1137/140966290.
#     R. I. Saye, "High-order quadrature on multi-component domains implicitly
#     defined by multivariate polynomials", J. Comput. Phys. 448 (2022)
#     110720, doi:10.1016/j.jcp.2021.110720.
#
#   The same construction is the moment source for moment-fitting on implicit
#   domains in
#
#     B. Müller, F. Kummer, M. Oberlack, "Highly accurate surface and volume
#     integration on implicit domains by means of moment-fitting", Int. J.
#     Numer. Methods Engng. 96 (2013) 512–528, doi:10.1002/nme.4569.
#
#   This is a clean-room pure-Julia implementation written from the method
#   description, not a port of any specific codebase.
#
# What it produces
#
#   Given a list of smooth level-set callbacks and a membership predicate over
#   their signs (the CSG tree in `src/physical.jl` supplies both), plus an
#   axis-aligned box R, the kernel returns a quadrature rule {(xₖ, wₖ)} for the
#   cut volume Ω ∩ R. The rule is high-order accurate: exact for polynomials
#   where every active level set is graph-like over R (linear leaves ⇒ machine
#   precision, including polytope corners), and spectrally accurate for smooth
#   curved boundaries away from turning points.
#
#   In the finite-cell moment-fit pipeline (`src/fcm.jl`) the volume rule plays
#   two roles: the moments mₐ = ∫_{Ω∩R} ψₐ dx are recovered as Σₖ wₖ ψₐ(xₖ),
#   and the points xₖ double as the non-negative moment-fit candidate cloud.
#
# Method in one paragraph
#
#   Integration over Ω ∩ R is reduced one dimension at a time. Pick a "height"
#   axis d* in which the active level sets are graph-like (each varies
#   monotonically, or is constant, along d* — a constant leaf, e.g. a
#   perpendicular half-space, simply partitions the base, which is what keeps
#   polytope corners exact). The (D−1)-dimensional base integral over the
#   projection is built by the same recursion, partitioned by the leaves'
#   restrictions to the two faces x_{d*} = lo and x_{d*} = hi so the projection
#   boundary and the change in fiber topology are resolved. At each base node a
#   1D Gauss rule is placed on the parts of the fiber that lie in Ω, found by
#   root-finding on the level sets and testing the membership predicate. When a
#   box is not graph-like it is bisected (a shallow octree above the kernel);
#   at the depth floor the kernel reduces anyway — the multi-root fiber scan
#   still yields the correct region, only the base partition (and thus the
#   order) degrades at the unresolved feature. ∇φ is obtained by automatic
#   differentiation (`ForwardDiff`), with a finite-difference fallback.

# ── Coordinate insert / remove helpers ─────────────────────────────────────────

# Insert scalar `v` at axis `k` of an `M`-vector, yielding an `(M+1)`-vector.
# `v` may differ in type from `x`'s eltype — when automatic differentiation
# pushes `ForwardDiff.Dual` base coordinates through a reduced closure the
# spliced-in face value is still a plain real, so the result type is promoted.
@inline function _insert_axis(x::SVector{M,T}, k::Int, v) where {M,T}
    R = promote_type(T, typeof(v))
    return SVector{M + 1,R}(ntuple(i -> i < k ? R(x[i]) : (i == k ? R(v) : R(x[i - 1])), M + 1))
end

# Drop axis `k` from an `M`-vector, yielding an `(M−1)`-vector.
@inline function _remove_axis(x::SVector{M,T}, k::Int) where {M,T}
    return SVector{M - 1,T}(ntuple(i -> i < k ? x[i] : x[i + 1], M - 1))
end

# Project a box onto the base obtained by removing the height axis `k`.
function _remove_axis(b::AxisBox{M,T}, k::Int) where {M,T}
    return AxisBox{M - 1,T}(_remove_axis(b.lower, k), _remove_axis(b.upper, k))
end

# ── Kernel context and gradient ────────────────────────────────────────────────

# Per-call configuration for the recursion: the shared 1D Gauss nodes/weights
# on [−1, 1], the gradient operator (AD or finite difference), the
# subdivision-depth budget, the number of scan samples used to bracket roots
# along a fiber, the region-scaled width below which a sub-interval is dropped as
# a numerically-coincident breakpoint, and the relative threshold below which a
# leaf's directional derivative counts as "flat" (constant) along an axis.
struct _QuadCtx{T,G}
    nodes::Vector{T}
    weights::Vector{T}
    gradient::G
    max_subdiv::Int
    root_samples::Int
    atol::T
    flat_tol::T
end

# Central-difference gradient fallback for level sets through which automatic
# differentiation cannot be pushed. Step size is the cube-root-of-eps balance
# between truncation and round-off, scaled per axis by the coordinate magnitude.
function _fd_gradient(g, x::SVector{D,T}) where {D,T}
    h0 = cbrt(eps(T))
    return SVector{D,T}(ntuple(D) do d
                            h = h0 * max(one(T), abs(x[d]))
                            e = SVector{D,T}(ntuple(i -> i == d ? h : zero(T), D))
                            (g(x + e) - g(x - e)) / (2h)
                        end)
end

# ── 1D safeguarded root finding ────────────────────────────────────────────────

# Bisection of `f` on a bracket `[a, b]` with `f(a)`, `f(b)` of opposite sign.
# Converges to the machine-representable root. Bisection (rather than
# Newton/Brent) never leaves the bracket, so a root is always returned.
function _bisect(f, a::T, b::T, fa::T, fb::T) where {T}
    lo, hi, flo = a, b, fa
    for _ in 1:80
        m = (lo + hi) / 2
        (m == lo || m == hi) && return m
        fm = f(m)
        fm == 0 && return m
        if (flo < 0) == (fm < 0)
            lo, flo = m, fm
        else
            hi = m
        end
    end
    return (lo + hi) / 2
end

# ── Sign sampling and height-axis certificate ──────────────────────────────────

# Sign of `g` if it is uniformly signed across the box center and its 2ᴰ
# corners, else `0` (a sign change was sampled). Used to prune leaves that do
# not vary on the box.
function _sample_sign(g, U::AxisBox{D,T}) where {D,T}
    vc = g(center(U))
    sref = vc < zero(vc) ? -1 : (vc > zero(vc) ? 1 : 0)
    sref == 0 && return 0
    for ci in CartesianIndices(ntuple(_ -> 0:1, D))
        x = SVector{D,T}(ntuple(a -> ci.I[a] == 0 ? U.lower[a] : U.upper[a], D))
        v = g(x)
        s = v < zero(v) ? -1 : (v > zero(v) ? 1 : 0)
        s != sref && return 0
    end
    return sref
end

# Pick a height axis and report whether it is certifiably graph-like. For each
# axis k, every leaf is classified by sampling ∂g/∂x_k at the box center and
# 2ᴰ corners:
#
#   MONOTONE — consistent non-zero sign ⇒ at most one fiber root;
#   FLAT     — |∂g/∂x_k| ≈ 0 at every sample ⇒ no fiber root (the leaf
#              partitions the base instead; this is what makes polytope corners
#              exact, e.g. a half-space perpendicular to k);
#   BAD      — the sign of ∂g/∂x_k changes ⇒ possibly several fiber roots.
#
# An axis is usable when no leaf is BAD; among usable axes pick the one
# maximising the smallest |∂g/∂x_k| over the MONOTONE leaves (best
# conditioning). Returns `(k, usable)`; when no axis is usable, `k` is the
# least-bad axis for the force-reduce path at the subdivision floor.
function _choose_axis(lsets, U::AxisBox{D,T}, ctx::_QuadCtx) where {D,T}
    samples = SVector{D,T}[center(U)]
    for ci in CartesianIndices(ntuple(_ -> 0:1, D))
        push!(samples, SVector{D,T}(ntuple(a -> ci.I[a] == 0 ? U.lower[a] : U.upper[a], D)))
    end
    grads = [[ctx.gradient(g, s) for s in samples] for g in lsets]
    # Gradient scale per leaf from the finite samples only: an SDF-like leaf
    # evaluated at its own centre yields ∇ = 0/0 = NaN, which must not poison
    # the flat/monotone/bad classification below.
    scales = map(grads) do gs
        finite = (norm(g) for g in gs if all(isfinite, g))
        m = reduce(max, finite; init=zero(T))
        m > 0 ? m : one(T)
    end

    best_k = 0
    best_score = T(-Inf)
    fallback_k = 1
    fallback_bad = typemax(Int)
    fallback_score = T(-Inf)
    for k in 1:D
        usable = true
        score = T(Inf)
        any_monotone = false
        nbad = 0
        for (gi, gs) in enumerate(grads)
            tol = ctx.flat_tol * max(scales[gi], one(T))
            pos = false
            neg = false
            minmag = T(Inf)
            for g in gs
                dk = g[k]
                isfinite(dk) || continue   # ignore non-finite gradient samples
                if dk > tol
                    pos = true
                    minmag = min(minmag, dk)
                elseif dk < -tol
                    neg = true
                    minmag = min(minmag, -dk)
                end
            end
            if pos && neg
                usable = false
                nbad += 1
            elseif pos || neg
                any_monotone = true
                score = min(score, minmag)
            end
        end
        # An axis along which no active leaf varies (all FLAT) cannot root-find
        # the interface — it scores lowest so a genuinely monotone axis wins
        # (critical for the surface rule, harmless for the volume rule).
        any_monotone || (score = zero(T))
        if usable && score > best_score
            best_score = score
            best_k = k
        end
        fscore = isfinite(score) ? score : zero(T)
        if nbad < fallback_bad || (nbad == fallback_bad && fscore > fallback_score)
            fallback_bad = nbad
            fallback_score = fscore
            fallback_k = k
        end
    end
    return best_k == 0 ? (fallback_k, false) : (best_k, true)
end

# ── Fiber rule emission ────────────────────────────────────────────────────────

# Sorted breakpoints of the fiber `t ↦ embed(t)` on `[lo, hi]`: the endpoints
# plus every sign change of every active leaf, located by scanning
# `ctx.root_samples` sub-intervals and bisecting each bracketed crossing.
function _fiber_breaks(lsets, embed, lo::T, hi::T, ctx::_QuadCtx) where {T}
    breaks = T[lo, hi]
    M = ctx.root_samples
    for g in lsets
        tprev = lo
        fprev = g(embed(lo))
        for s in 1:M
            tcur = s == M ? hi : lo + (hi - lo) * (T(s) / M)
            fcur = g(embed(tcur))
            # A root landing exactly on a scan sample makes `fprev*fcur`
            # vanish, so the strict sign-change test below would miss it; the
            # sample is itself the breakpoint.
            if fcur == 0
                push!(breaks, tcur)
            elseif fprev * fcur < 0
                push!(breaks, _bisect(t -> g(embed(t)), tprev, tcur, fprev, fcur))
            end
            tprev, fprev = tcur, fcur
        end
    end
    sort!(breaks)
    return breaks
end

# Place 1D Gauss rules on the parts of one fiber that lie in Ω and append the
# lifted points/weights (scaled by the base weight `wbase`). `embed(t)` lifts a
# fiber coordinate to the full D-dimensional point; `membership` is the Ω test.
function _emit_fiber!(pts::Vector{SVector{D,T}}, wts::Vector{T}, lsets, embed, lo::T, hi::T,
                      wbase::T, membership, ctx::_QuadCtx) where {D,T}
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

# Tensor Gauss rule over the whole box, used when no leaf varies on the box and
# the box is inside Ω. Maps directly into physical coordinates with the box
# Jacobian folded in.
function _full_box_rule(U::AxisBox{D,T}, ctx::_QuadCtx) where {D,T}
    nodes, gw = ctx.nodes, ctx.weights
    n = length(nodes)
    jac = volume(U) / convert(T, 2^D)
    pts = Vector{SVector{D,T}}(undef, n^D)
    wts = Vector{T}(undef, n^D)
    idx = 1
    for ci in CartesianIndices(ntuple(_ -> 1:n, D))
        eta = SVector{D,T}(ntuple(a -> nodes[ci.I[a]], D))
        w = one(T)
        for a in 1:D
            w *= gw[ci.I[a]]
        end
        pts[idx] = reference_to_physical(U, eta)
        wts[idx] = w * jac
        idx += 1
    end
    return pts, wts
end

# ── Core recursion ─────────────────────────────────────────────────────────────

# Bisect `U` into its 2ᴰ children and concatenate `recurse(child, depth+1)` over
# them. This is the shared "not graph-like ⇒ subdivide" branch of the volume and
# surface recursions; `recurse` is the per-child continuation.
function _subdivide_and_collect(recurse, U::AxisBox{D,T}, depth::Int) where {D,T}
    c = center(U)
    pts = SVector{D,T}[]
    wts = T[]
    for ci in CartesianIndices(ntuple(_ -> 0:1, D))
        lower = SVector{D,T}(ntuple(a -> ci.I[a] == 0 ? U.lower[a] : c[a], D))
        upper = SVector{D,T}(ntuple(a -> ci.I[a] == 0 ? c[a] : U.upper[a], D))
        cp, cw = recurse(AxisBox{D,T}(lower, upper), depth + 1)
        append!(pts, cp)
        append!(wts, cw)
    end
    return pts, wts
end

# Recursively integrate over Ω ∩ U. `lsets` is the list of leaf callbacks to
# partition by; `membership(x)` tests x ∈ Ω at a full-dimensional point. The
# original problem starts from the domain's leaves and its membership predicate;
# each reduction passes the leaves' face restrictions as a partition-only base
# problem (`Returns(true)` membership) and applies the real membership in its
# own fiber loop. Stages:
#
#   1. Prune leaves that do not vary on U (a partition-only optimisation;
#      `membership` re-evaluates the full predicate, so this is safe). With no
#      varying leaf left, U is uniform: full tensor rule if inside, else empty.
#   2. Base case D = 1: emit the inside parts of the single axis.
#   3. Choose the height axis. If not certifiably graph-like and budget remains,
#      bisect and recurse; at the floor, force-reduce on the least-bad axis.
#   4. Reduce: build the base problem on U with the height axis removed (every
#      leaf's two face restrictions, partition-only) and recurse; then sweep the
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
        _emit_fiber!(pts, wts, active, t -> SVector{1,T}(t), U.lower[1], U.upper[1], one(T),
                     membership, ctx)
        return pts, wts
    end

    k, usable = _choose_axis(active, U, ctx)
    if !usable && depth < ctx.max_subdiv
        return _subdivide_and_collect((box, d) -> _implicit_quad(active, membership, box, ctx, d),
                                      U, depth)
    end

    # Reduce one dimension: partition-only base, then a fiber rule per base node.
    Ub = _remove_axis(U, k)
    facelo, facehi = U.lower[k], U.upper[k]
    base = Any[]
    for g in active
        push!(base, x -> g(_insert_axis(x, k, facelo)))
        push!(base, x -> g(_insert_axis(x, k, facehi)))
    end
    bpts, bwts = _implicit_quad(base, Returns(true), Ub, ctx, 0)

    pts = SVector{D,T}[]
    wts = T[]
    for n in eachindex(bpts)
        xb, wb = bpts[n], bwts[n]
        _emit_fiber!(pts, wts, active, t -> _insert_axis(xb, k, t), facelo, facehi, wb, membership,
                     ctx)
    end
    return pts, wts
end

# ── Surface recursion ──────────────────────────────────────────────────────────

# Quadrature for the implicit surface {φ = 0} ∩ U of a single level set. One
# dimension is reduced as in the volume case; the base over the projection is a
# partition-only volume rule, and at each base node the single fiber root x*
# contributes the surface element with the graph Jacobian |∇φ| / |∂φ/∂x_k|.
# (The single-leaf surface rule is what the package validates against; a
# multi-component surface rule is a documented follow-up.)
function _implicit_surface(phi, U::AxisBox{D,T}, ctx::_QuadCtx, depth::Int) where {D,T}
    _sample_sign(phi, U) != 0 && return SVector{D,T}[], T[]

    k, usable = _choose_axis(Any[phi], U, ctx)
    if !usable && depth < ctx.max_subdiv
        return _subdivide_and_collect((box, d) -> _implicit_surface(phi, box, ctx, d), U, depth)
    end

    Ub = _remove_axis(U, k)
    facelo, facehi = U.lower[k], U.upper[k]
    base = Any[x -> phi(_insert_axis(x, k, facelo)), x -> phi(_insert_axis(x, k, facehi))]
    bpts, bwts = _implicit_quad(base, Returns(true), Ub, ctx, 0)

    pts = SVector{D,T}[]
    wts = T[]
    for n in eachindex(bpts)
        xb, wb = bpts[n], bwts[n]
        flo = phi(_insert_axis(xb, k, facelo))
        fhi = phi(_insert_axis(xb, k, facehi))
        flo * fhi < 0 || continue
        r = _bisect(t -> phi(_insert_axis(xb, k, t)), facelo, facehi, flo, fhi)
        xstar = _insert_axis(xb, k, r)
        grad = ctx.gradient(phi, xstar)
        push!(pts, xstar)
        push!(wts, wb * norm(grad) / abs(grad[k]))
    end
    return pts, wts
end

# ── Public entry points ────────────────────────────────────────────────────────

# `scale` is the region's largest edge length; the breakpoint-drop tolerance
# `atol` is taken relative to it so the kernel behaves identically under a
# rescaling of the geometry. `flat_tol` is already relative (to the gradient
# magnitude) inside `_choose_axis`.
function _quad_ctx(::Type{T}, gauss_points::Int, grad, max_subdiv::Int, scale::Real) where {T}
    rawnodes, rawweights = gausslegendre(gauss_points)
    gradient = grad === nothing ? (g, x) -> ForwardDiff.gradient(g, x) : grad
    return _QuadCtx(T.(rawnodes), T.(rawweights), gradient, max_subdiv, max(6, 2 * gauss_points),
                    eps(T)^(3 // 4) * T(scale), sqrt(eps(T)))
end

"""
    implicit_volume_quadrature(leaves, membership, region; gauss_points,
                               grad=nothing, max_subdiv=4) -> (points, weights)
    implicit_volume_quadrature(phi, region; gauss_points, …) -> (points, weights)

Return a quadrature rule `{(xₖ, wₖ)}` for the cut volume `Ω ∩ region`, so that

    ∫_{Ω ∩ region} f(x) dx  ≈  Σₖ wₖ f(xₖ)

with the points in physical coordinates and the box Jacobian folded into the
weights. `leaves` is a vector of smooth level-set callbacks and `membership` is
a predicate `x -> Bool` testing `x ∈ Ω` (Saye's multi-component
dimension-reduction quadrature; see the file header). The single-argument form
takes one callback `phi` with `Ω = {φ ≤ 0}`.

The rule is exact for polynomials where every active leaf is graph-like over
`region` (linear leaves ⇒ machine precision, including polytope corners) and
spectrally accurate for smooth curved boundaries.

Arguments:

  - `gauss_points`: number of 1D Gauss–Legendre points per inside fiber
    sub-interval. For a *tilted* cut the inside-interval endpoints are linear in
    the base coordinates, so each fiber reduction raises the integrated degree
    by one; integrating tensor moments of total order `p` over a beveled corner
    needs `q ≈ (p + D) / 2`, not `≈ p / 2` (the finite-cell pipeline picks this
    via `_implicit_gauss_points`).
  - `grad`: optional gradient operator `(g, x) -> SVector` overriding the
    `ForwardDiff` default; pass [`_fd_gradient`](@ref) for non-AD callbacks.
  - `max_subdiv`: subdivision-depth budget for non-graph-like cells. Beyond it
    the kernel force-reduces (the region stays correct; order degrades only at
    the unresolved feature).

Feature detection is by point sampling: a leaf is pruned where it is uniformly
signed at the box center and corners, and fiber roots are bracketed by a bounded
scan. A feature — or a pair of roots — smaller than that sampling can be missed,
so `region` should be sized below the smallest geometric feature. In the
finite-cell pipeline the cell classifier sizes cut cells by
`subcell_length_scale` and shares the same sampling, so a sub-cell feature is
consistently never routed here.
"""
function implicit_volume_quadrature(leaves::AbstractVector, membership, region::AxisBox{D,T};
                                    gauss_points::Int, grad=nothing, max_subdiv::Int=4) where {D,T}
    ctx = _quad_ctx(T, gauss_points, grad, max_subdiv, maximum(region.upper - region.lower))
    return _implicit_quad(collect(Any, leaves), membership, region, ctx, 0)
end

function implicit_volume_quadrature(phi, region::AxisBox{D,T}; gauss_points::Int, grad=nothing,
                                    max_subdiv::Int=4) where {D,T}
    return implicit_volume_quadrature(Any[phi], x -> phi(x) <= 0, region; gauss_points, grad,
                                      max_subdiv)
end

"""
    implicit_surface_quadrature(phi, region; gauss_points, grad=nothing, max_subdiv=4)
        -> (points, weights)

Return a quadrature rule for the implicit surface `∂Ω ∩ region = {φ = 0} ∩
region` of a single level set `phi`, so that
`∫_{∂Ω ∩ region} f dS ≈ Σₖ wₖ f(xₖ)`. Same dimension-reduction construction and
arguments as [`implicit_volume_quadrature`](@ref); each base node contributes
its fiber root with the height-graph surface element `|∇φ| / |∂φ/∂x_{d*}|`.
Defined for `D ≥ 2`.
"""
function implicit_surface_quadrature(phi, region::AxisBox{D,T}; gauss_points::Int, grad=nothing,
                                     max_subdiv::Int=4) where {D,T}
    D >= 2 || throw(ArgumentError("implicit_surface_quadrature requires D ≥ 2"))
    ctx = _quad_ctx(T, gauss_points, grad, max_subdiv, maximum(region.upper - region.lower))
    return _implicit_surface(phi, region, ctx, 0)
end
