"""
    PhysicalDomain{F,T}(phi, lipschitz, alpha, subcell_depth, moment_order_factor, target_residual)

Level-set description of a physical domain Ω for finite-cell-style immersed
integration. The convention is `Ω = { x : φ(x) ≤ 0 }`.

Fields:

  - `phi(x)`: scalar-valued level set on `SVector{D,T}` coordinates. A true
    signed-distance function is allowed but not required.
  - `lipschitz::T`: a Lipschitz constant `L` with
    `|φ(x) − φ(y)| ≤ L · ‖x − y‖`. `L = 1` for a true SDF; passing `Inf`
    disables the cheap Lipschitz certificate and forces classification by
    corner sampling + octree subdivision down to `subcell_depth`.
  - `alpha::T`: fictitious-region stabilization weight (α-FCM). `0` is the
    strict cut path (cells fully outside Ω are dropped from the dof
    layout); `> 0` keeps fictitious cells active with quadrature weights
    pre-multiplied by `α`.
  - `subcell_depth::Int`: maximum octree refinement depth used when corner
    sampling alone cannot certify the cell's classification.
  - `moment_order_factor::Int`: multiplier on the NNMF moment-fit basis
    order per axis. The actual moment order chosen for a cut region is
    `factor × max(level.order)` over the region's parents. `factor = 2`
    (default) integrates trial × test products exactly on the moment
    basis, matching what tensor Gauss does on `:full` regions.
    `factor = 1` halves the basis but only integrates source-style
    (degree-`p`) integrands exactly — useful when the bilinear-form
    approximation is acceptable and cut-cell speed matters.
  - `target_residual::T`: NNMF L² residual the moment-fit aims for in cut
    regions. Default `1e-6` is calibrated to the natural stair-step
    accuracy floor of the octree moment integration at the default
    `subcell_depth = 4`. QuESo's reference implementation hardcodes
    `1e-10` and its shipped examples typically run `1e-8`, but those
    rely on QuESo's B-rep-exact surface-integral moments — unreachable
    in this port's stair-step integrator (see `src/fcm.jl` for the
    geometry-kernel deviation). Tightening below `1e-6` without also
    raising `subcell_depth` triggers retries the NNLS cannot satisfy
    and allocates gigabytes for no accuracy gain. Rule of thumb: lower
    this by roughly two decades for every additional level of
    `subcell_depth` (e.g. `1e-8` at depth 6).

Construct via [`physical_domain`](@ref).
"""
struct PhysicalDomain{F,T<:Real}
    phi::F
    lipschitz::T
    alpha::T
    subcell_depth::Int
    moment_order_factor::Int
    target_residual::T
end

"""
    physical_domain(phi; lipschitz=Inf, alpha=0.0, subcell_depth=4,
                    moment_order_factor=2, target_residual=1e-6)

Construct a [`PhysicalDomain`](@ref). `phi(x)` must return a real scalar
with `phi(x) ≤ 0` inside Ω. `lipschitz` defaults to `Inf` (no certificate;
pure sampling-based classification). `alpha = 0` is the strict-cut path
(fictitious cells dropped); pass `alpha > 0` for α-FCM stabilization on
fictitious cells. `moment_order_factor` tunes the NNMF basis order; see
the [`PhysicalDomain`](@ref) docstring for the cost / accuracy trade-off.
`target_residual` is the NNMF residual the moment-fit aims for in cut
regions; the default is matched to the natural stair-step accuracy of the
octree moment integration.
"""
function physical_domain(phi; lipschitz::Real=Inf, alpha::Real=0.0, subcell_depth::Integer=4,
                         moment_order_factor::Integer=2, target_residual::Real=1.0e-6)
    lipschitz > 0 || throw(ArgumentError("lipschitz must be positive; got $lipschitz"))
    alpha >= 0 || throw(ArgumentError("alpha must be ≥ 0; got $alpha"))
    subcell_depth >= 0 || throw(ArgumentError("subcell_depth must be ≥ 0; got $subcell_depth"))
    moment_order_factor >= 1 ||
        throw(ArgumentError("moment_order_factor must be ≥ 1; got $moment_order_factor"))
    target_residual > 0 ||
        throw(ArgumentError("target_residual must be positive; got $target_residual"))
    T = promote_type(typeof(float(lipschitz)), typeof(float(alpha)), typeof(float(target_residual)))
    return PhysicalDomain{typeof(phi),T}(phi, T(lipschitz), T(alpha), Int(subcell_depth),
                                         Int(moment_order_factor), T(target_residual))
end

# Half-diagonal of an axis-aligned box — the radius of the smallest ball
# enclosing the box. Used by the Lipschitz certificate in `_classify_box`:
# for a Lipschitz `φ` and any point `y` inside the box, |φ(y) − φ(c)| ≤ L·r
# where `c` is the center and `r` is the half-diagonal returned here, so a
# sign of `φ(c)` whose magnitude exceeds `L·r` is uniform across the box.
_half_diagonal(b::AxisBox) = norm(b.upper - b.lower) / 2

# Sample φ at the 2^D corners of `box` and tally the inside/outside counts.
# The center sample `phi_c` is reused (it has already been evaluated by
# `_classify_box` for the Lipschitz certificate), so we count it explicitly
# instead of re-evaluating φ at the center. Corners — rather than interior
# points — are the natural sample set for an axis-aligned box: they are the
# vertices whose signs determine which faces ∂Ω crosses, and they double as
# the leaf-vertex set when the octree recurses one level deeper.
function _corner_signs(physical::PhysicalDomain, box::AxisBox{D,T}, phi_c) where {D,T}
    inside = (phi_c <= zero(phi_c)) ? 1 : 0
    outside = (phi_c <= zero(phi_c)) ? 0 : 1
    for ci in CartesianIndices(ntuple(_ -> 0:1, D))
        x = SVector{D,T}(ntuple(d -> ci.I[d] == 0 ? box.lower[d] : box.upper[d], D))
        phi_x = physical.phi(x)
        phi_x <= zero(phi_x) ? (inside += 1) : (outside += 1)
    end
    return inside, outside
end

# Recursive classifier returning one of `:full`, `:cut`, `:fictitious`.
#
# Strategy, in order of preference (cheapest to most expensive):
#
#   1. Lipschitz certificate. If |φ(center)| > L · r (r = half-diagonal),
#      `φ` is sign-uniform across the box and we are exactly inside (`:full`)
#      or outside (`:fictitious`) Ω. Free when a Lipschitz `L` is supplied;
#      skipped when `L = Inf`.
#   2. Corner sampling. Evaluate φ at the 2^D vertices. Mixed signs prove
#      the boundary crosses the box and we return `:cut` immediately.
#   3. Octree recursion. Signs all agree but the certificate did not fire
#      (so we cannot rule out a small cavity strictly inside the box). If
#      depth budget remains, bisect the box into 2^D equal children and
#      classify each. Any child returning `:cut` propagates; any
#      disagreement between children's `:full`/`:fictitious` verdicts also
#      means the boundary crosses the parent and we return `:cut`. If every
#      child returns the same verdict, propagate it.
#   4. Depth budget exhausted with agreeing samples. Trust the consensus
#      verdict — there is no further evidence available.
#
# The depth budget caps the recursive cost at `(2^D)^subcell_depth` leaf
# classifications per call, which is the same bound the octree moment
# integrator uses.
function _classify_box(physical::PhysicalDomain, box::AxisBox{D,T}, depth::Integer) where {D,T}
    c = center(box)
    r = _half_diagonal(box)
    phi_c = physical.phi(c)
    threshold = physical.lipschitz * r

    phi_c < -threshold && return :full
    phi_c > threshold && return :fictitious

    n_inside, n_outside = _corner_signs(physical, box, phi_c)
    (n_inside > 0 && n_outside > 0) && return :cut

    if depth < physical.subcell_depth
        # Bisection: every child shares one corner with the parent (the
        # box center) and inherits seven of its corners' coordinates from
        # the parent's `lower`/`upper`. The 2^D children are enumerated by
        # iterating `CartesianIndices(ntuple(_ -> 0:1, D))` — bit 0 in each
        # axis takes `lower`, bit 1 takes `c`.
        result::Union{Nothing,Symbol} = nothing
        for ci in CartesianIndices(ntuple(_ -> 0:1, D))
            child_lower = SVector{D,T}(ntuple(d -> ci.I[d] == 0 ? box.lower[d] : c[d], D))
            child_upper = SVector{D,T}(ntuple(d -> ci.I[d] == 0 ? c[d] : box.upper[d], D))
            child_state = _classify_box(physical, AxisBox{D,T}(child_lower, child_upper), depth + 1)
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

Classify an axis-aligned `box` against `physical`'s level set. Returns one
of `:full` (entirely inside Ω), `:cut` (crossed by ∂Ω), or `:fictitious`
(entirely outside Ω). Uses the Lipschitz certificate first and falls back
to octree-bounded corner sampling; see the comment block on
`_classify_box` for the full strategy.

The 3-arg form memoizes the result by `(box.lower, box.upper)` in a
caller-owned `Dict`. A prepared model shares one cache between the
cell-level fold (which classifies every level's cells once at construction
time) and the per-region dispatch in the integration plan — halving the
classification work in the common single-level case where the region and
cell boxes coincide.
"""
classify_cell(physical::PhysicalDomain, box::AxisBox) = _classify_box(physical, box, 0)

# Memoisation cache for `classify_cell`: keyed by the box corner pair,
# valued by the classification verdict. A single cache is threaded through
# `prepare` and `move!` so the cell-level fold and the per-region dispatch
# share the work.
const _ClassifyCache{D,T} = Dict{Tuple{SVector{D,T},SVector{D,T}},Symbol}

function classify_cell(physical::PhysicalDomain, box::AxisBox{D,T},
                       cache::_ClassifyCache{D,T}) where {D,T}
    return get!(cache, (box.lower, box.upper)) do
        _classify_box(physical, box, 0)
    end
end
