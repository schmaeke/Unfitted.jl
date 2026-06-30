# Coordinate alias used across geometry, mesh, basis, and evaluation signatures.
# `NTuple` for raw user input; `SVector` for stored coordinates and downstream
# kernels. Both round-trip cheaply via `SVector{D}(…)`.
const PointLike{D} = Union{NTuple{D,<:Real},SVector{D,<:Real}}

"""
    GeometryTolerance([T=Float64]; merge, contain, small_volume)

Coordinate tolerances driving geometry construction, intersection building,
and conditioning diagnostics. Three thresholds, all positive and expressed
in the same units as physical-domain coordinates:

  - `merge` — coordinates closer than this in a given axis are
    canonicalised to the same value. Drives [`merge_coordinates`](@ref)
    and therefore the admissible-box construction in
    `intersections.jl`, so it sets the smallest distinguishable
    mesh-coordinate increment.
  - `contain` — slack on point-in-box queries
    ([`contains_point`](@ref), [`is_inside`](@ref), `locate_cell`).
    Containment is conservative: a point counts as inside when
    `lower − contain ≤ x ≤ upper + contain` on every axis.
  - `small_volume` — reporting threshold for tiny integration regions.
    Regions with volume below this are recorded as `SmallOverlap`
    entries in the assembly diagnostics; they are *not* silently
    dropped — small-overlap conditioning is made visible instead of
    hidden.

The keyword constructor accepts any positive value per field. Defaults are
`merge = √eps(T)` and `contain = small_volume = merge`. Tighten `merge`
only when the geometry is known to need finer canonicalisation; loosen
`contain` only at integration tests that genuinely require it.
"""
struct GeometryTolerance{T<:Real}
    merge::T
    contain::T
    small_volume::T
end

function GeometryTolerance(::Type{T}=Float64; merge=sqrt(eps(T)), contain=merge,
                           small_volume=merge) where {T<:Real}
    return GeometryTolerance{T}(convert(T, merge), convert(T, contain), convert(T, small_volume))
end

"""
    merge_coordinates(values, tol::GeometryTolerance{T}) -> Vector{T}

Sort `values` and return a `Vector{T}` in which neighbouring entries
differing by at most `tol.merge` have been collapsed to their first
representative. The result is the canonical per-axis coordinate set used
to build admissible integration intervals in `intersections.jl`: two
participating meshes' element-boundary coordinates that are
indistinguishable up to `tol.merge` produce a single shared interval
rather than a sliver.

Stable, allocation-light, single-pass after the sort. `values` may be any
iterable of reals; the sorted working copy is the only allocation — the
merge dedups in place into its own prefix.
"""
function merge_coordinates(values, tol::GeometryTolerance{T}) where {T}
    sorted = sort!(T[convert(T, value) for value in values])
    isempty(sorted) && return sorted

    # Dedup ascending neighbours in place: keep `sorted[i]` only when it
    # separates from the last kept representative `sorted[n]` by more than
    # `tol.merge`. `sorted` is ascending, so the gap is already non-negative
    # and no `abs` is needed. Survivors are packed into the prefix `1:n`.
    n = 1
    @inbounds for i in 2:length(sorted)
        if sorted[i] - sorted[n] > tol.merge
            n += 1
            sorted[n] = sorted[i]
        end
    end
    return resize!(sorted, n)
end

"""
    AxisBox{D,T}(lower, upper)
    box(lower, upper)
    box(center; halfwidth)

Axis-aligned physical box in `D` dimensions, represented by its lower and
upper corner coordinates. Coordinates are stored as `SVector{D,T}` and the
constructor enforces the non-degenerate invariant `lower[i] < upper[i]`
on every axis — every region the assembly hot loop sees is a proper
`D`-cube with positive volume.

Constructors:

  - `box(lower, upper)` — corner-to-corner. Accepts `NTuple{D,<:Real}`
    or `SVector{D,<:Real}` corners; the scalar type is promoted from
    the inputs.
  - `box(center; halfwidth)` — centered form. `center` is a point;
    `halfwidth` may be a scalar (isotropic) or a `D`-tuple / `SVector`
    (anisotropic) of half-edge-lengths. Equivalent to
    `box(center − halfwidth, center + halfwidth)`.

Geometry consumers never reach into boxes through `x`/`y`/`z` field names;
they iterate over `1:D` and rely on the dimension-generic
[`dimension`](@ref), [`edge_lengths`](@ref), [`volume`](@ref),
[`center`](@ref), [`contains_point`](@ref), and
[`reference_to_physical`](@ref) / [`physical_to_reference`](@ref) maps.
"""
struct AxisBox{D,T<:Real}
    lower::SVector{D,T}
    upper::SVector{D,T}

    function AxisBox{D,T}(lower::SVector{D,T}, upper::SVector{D,T}) where {D,T<:Real}
        D >= 1 || throw(ArgumentError("AxisBox dimension must be at least 1"))
        all(upper[i] > lower[i] for i in 1:D) ||
            throw(ArgumentError("AxisBox upper bounds must exceed lower bounds"))
        return new{D,T}(lower, upper)
    end
end

AxisBox(lower::SVector{D,T}, upper::SVector{D,T}) where {D,T<:Real} = AxisBox{D,T}(lower, upper)
function AxisBox(lower::NTuple{D,T}, upper::NTuple{D,T}) where {D,T<:Real}
    AxisBox{D,T}(SVector{D,T}(lower), SVector{D,T}(upper))
end

"""
    box(lower, upper) -> AxisBox
    box(center; halfwidth) -> AxisBox

Construct an [`AxisBox`](@ref), the dimension-generic axis-aligned box used
throughout the package.

  - `box(lower, upper)` takes corner coordinates as an `NTuple{D,<:Real}` or
    `SVector{D,<:Real}`; the scalar type is promoted from the inputs.
  - `box(center; halfwidth)` takes a center point and a `halfwidth` that is
    either a scalar (isotropic) or a `D`-tuple / `SVector` (anisotropic). It
    is equivalent to `box(center .- halfwidth, center .+ halfwidth)`.

See [`AxisBox`](@ref) for the stored representation and the non-degenerate
`lower[i] < upper[i]` invariant the constructor enforces.
"""
function box(lower::NTuple{D,<:Real}, upper::NTuple{D,<:Real}) where {D}
    return box(SVector(lower), SVector(upper))
end

function box(lower::SVector{D,<:Real}, upper::SVector{D,<:Real}) where {D}
    T = promote_type(eltype(lower), eltype(upper))
    return AxisBox{D,T}(SVector{D,T}(lower), SVector{D,T}(upper))
end

function box(center::PointLike{D}; halfwidth) where {D}
    c = SVector{D}(center)
    h = halfwidth isa Real ? SVector(ntuple(_ -> halfwidth, D)) : SVector{D}(halfwidth)
    return box(c .- h, c .+ h)
end

"""
    dimension(b::AxisBox) -> Int

Spatial dimension `D` of the box, recovered from its static type so the
result is a compile-time constant whenever `D` is statically known.
"""
dimension(::AxisBox{D}) where {D} = D

"""
    edge_lengths(b::AxisBox) -> SVector{D,T}

Per-axis edge lengths, `b.upper − b.lower`. Stored as an `SVector` so
downstream Jacobian and gradient-scaling computations stay stack-allocated.
"""
edge_lengths(b::AxisBox{D,T}) where {D,T} = b.upper - b.lower

"""
    volume(b::AxisBox) -> T

Geometric measure of the box: product of the per-axis edge lengths. In
`D` dimensions this is the `D`-cube volume. Used by the reference-to-
physical Jacobian `vol(b) / 2ᴰ` and by the small-overlap diagnostics in
`intersections.jl`.
"""
volume(b::AxisBox) = prod(edge_lengths(b))

"""
    center(b::AxisBox) -> SVector{D}

Geometric center of the box, `(b.lower + b.upper) / 2`. Used to construct
midpoint coverage queries (see `_parents_covering` in `intersections.jl`)
and to build the reference-to-physical affine map below.
"""
center(b::AxisBox) = (b.lower + b.upper) / 2

# Strict containment without tolerance: a point lies in the box iff every
# coordinate satisfies `lower[i] ≤ x[i] ≤ upper[i]`. Use `contains_point`
# instead when the query is geometrically motivated and a positive
# `GeometryTolerance.contain` is in scope.
function Base.in(point::PointLike{D}, b::AxisBox{D,T}) where {D,T}
    return all(b.lower[i] <= point[i] <= b.upper[i] for i in 1:D)
end

"""
    contains_point(point, b::AxisBox, tol::GeometryTolerance) -> Bool

Tolerance-aware point-in-box query: `point` is considered inside `b` when
`b.lower[i] − tol.contain ≤ point[i] ≤ b.upper[i] + tol.contain` on every
axis. Used by cell lookup and integration-region construction to absorb
floating-point noise at element boundaries. For an exact (tolerance-free)
query use `point in b`.
"""
function contains_point(point::PointLike{D}, b::AxisBox{D,T}, tol::GeometryTolerance{T}) where {D,T}
    return all(b.lower[i] - tol.contain <= point[i] <= b.upper[i] + tol.contain for i in 1:D)
end

# Structural equality on the corner coordinates. Floating-point exactness is
# acceptable here because the boxes consumed downstream are always built
# from canonicalised grid coordinates produced by `merge_coordinates`.
Base.:(==)(a::AxisBox{D}, b::AxisBox{D}) where {D} = a.lower == b.lower && a.upper == b.upper

"""
    box_intersection(a::AxisBox{D,T}, b::AxisBox{D,T}) -> Union{AxisBox{D,T},Nothing}

Component-wise intersection of two boxes: per-axis `max(a.lower, b.lower)`
and `min(a.upper, b.upper)`. Returns the resulting box when the
intersection has strictly positive measure on every axis, and `nothing`
otherwise — degenerate (zero-volume) intersections are reported as "no
overlap" rather than as a flat box. This mirrors the constructor's
non-degeneracy invariant on `AxisBox`.
"""
function box_intersection(a::AxisBox{D,T}, b::AxisBox{D,T}) where {D,T}
    lower = max.(a.lower, b.lower)
    upper = min.(a.upper, b.upper)
    return all(upper[i] > lower[i] for i in 1:D) ? AxisBox{D,T}(lower, upper) : nothing
end

"""
    is_inside(inner::AxisBox{D,T}, outer::AxisBox{D,T}, tol=GeometryTolerance(T)) -> Bool

Tolerance-aware containment of one box inside another: every corner of
`inner` lies inside `outer` up to `tol.contain` on each axis. Used to
validate overlay placement, since the `overlay(...)` constructor requires
the overlay domain to lie inside the physical domain.
"""
function is_inside(inner::AxisBox{D,T}, outer::AxisBox{D,T},
                   tol::GeometryTolerance{T}=GeometryTolerance(T)) where {D,T}
    return all(inner.lower[i] >= outer.lower[i] - tol.contain &&
               inner.upper[i] <= outer.upper[i] + tol.contain for i in 1:D)
end

"""
    reference_to_physical(b::AxisBox{D,T}, ξ) -> SVector{D,T}

Map a reference-cube coordinate `ξ ∈ [−1, 1]ᴰ` to the corresponding
physical point in `b` via the affine map

    x = (b.lower + b.upper) / 2 + (b.upper − b.lower) / 2 ∘ ξ

with `∘` the per-axis (Hadamard) product. The constant Jacobian determinant
is `vol(b) / 2ᴰ`, which is the factor every region quadrature rule
multiplies its reference weights by in the assembly hot loop.
"""
function reference_to_physical(b::AxisBox{D,T}, xi::PointLike{D}) where {D,T}
    xi_vec = SVector{D,T}(xi)
    return (b.lower + b.upper) / 2 + (b.upper - b.lower) .* xi_vec / 2
end

"""
    physical_to_reference(b::AxisBox{D,T}, x) -> SVector{D,T}

Inverse of [`reference_to_physical`](@ref): map a physical point `x ∈ b`
back to its reference-cube coordinate `ξ ∈ [−1, 1]ᴰ`,

    ξ = (2 x − b.lower − b.upper) / (b.upper − b.lower)

with the division taken per axis. Used to rebase region boxes into parent
reference frames during integration-region construction and to evaluate
basis functions at arbitrary physical points during post-processing.
"""
function physical_to_reference(b::AxisBox{D,T}, x::PointLike{D}) where {D,T}
    x_vec = SVector{D,T}(x)
    return (2 .* x_vec - b.lower - b.upper) ./ (b.upper - b.lower)
end
