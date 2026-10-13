# Coordinate alias used across geometry, mesh, basis, and evaluation signatures. `NTuple` for raw user input; `SVector`
# for stored coordinates and downstream kernels. Both round-trip cheaply via `SVector{D}(…)`.
const PointLike{D} = Union{NTuple{D,<:Real},SVector{D,<:Real}}

"""
    GeometryTolerance([T=Float64]; merge, contain, small_volume)

Coordinate tolerances driving geometry construction, intersection building, and conditioning diagnostics. Three
thresholds, all non-negative and expressed in the same units as physical-domain coordinates:

  - `merge` — coordinates closer than this in a given axis are canonicalised to the same value. Drives
    [`merge_coordinates`](@ref) and therefore the admissible-box construction in `intersections.jl`, so it sets the
    smallest distinguishable mesh-coordinate increment. Being an absolute length, it has to be read against the geometry
    it is applied to: a mesh whose spacing on some axis is at or below `merge` would have that axis's own cell
    boundaries collapsed. [`integration_plan`](@ref) rejects that combination outright instead of integrating a
    decimated grid.
  - `contain` — slack on point-in-box queries ([`contains_point`](@ref), [`is_inside`](@ref), `locate_cell`).
    Containment is conservative: a point counts as inside when `lower − contain ≤ x ≤ upper + contain` on every axis.
  - `small_volume` — reporting threshold for tiny integration regions. Regions with volume below this are recorded as
    `SmallOverlap` entries in the assembly diagnostics; they are *not* silently dropped — small-overlap conditioning is
    made visible instead of hidden.

Each field accepts any non-negative value; zero is legal and means exact merging, exact containment and no
small-overlap report. A negative or NaN value raises `ArgumentError`: a negative threshold silently disables the test
it drives, and a NaN one makes every comparison against it false. Defaults are `merge = √eps(T)` and
`contain = small_volume = merge`. The defaults are calibrated for coordinates of order one; on geometry at a markedly
different length scale, pass `merge` scaled to that geometry — e.g. `√eps(T) · L` for a domain of extent `L`. Tighten
`merge` also when the geometry is known to need finer canonicalisation; loosen `contain` only at integration tests that
genuinely require it.
"""
struct GeometryTolerance{T<:Real}
    merge::T
    contain::T
    small_volume::T

    # Every route to a tolerance, the keyword constructor included, ends here, so this is where a value that is not a
    # tolerance is refused. Each threshold is compared with `≤` or `>`, so a negative one turns its test off — `merge =
    # -1` stops merging even exact duplicates, a negative `contain` shrinks every box — and a NaN one makes every
    # comparison false, which collapses each merged axis to its first coordinate. Neither is caught anywhere downstream
    # before it has changed a result.
    function GeometryTolerance{T}(merge, contain, small_volume) where {T<:Real}
        return new{T}(_tolerance_value(T, merge, :merge), _tolerance_value(T, contain, :contain),
                      _tolerance_value(T, small_volume, :small_volume))
    end
end

# The positional constructor Julia generates for a struct without an inner constructor, kept by hand now that there is
# one; like every other route it ends in the inner constructor's check.
function GeometryTolerance(merge::T, contain::T, small_volume::T) where {T<:Real}
    return GeometryTolerance{T}(merge, contain, small_volume)
end

# Convert one threshold to `T` and refuse it unless it is non-negative (`v >= 0` is false for NaN as well).
function _tolerance_value(::Type{T}, value, name::Symbol) where {T}
    v = convert(T, value)
    v >= 0 || throw(ArgumentError("GeometryTolerance $name must be non-negative; got $value"))
    return v
end

function GeometryTolerance(::Type{T}=Float64; merge=sqrt(eps(T)), contain=merge, small_volume=merge) where {T<:Real}
    return GeometryTolerance{T}(convert(T, merge), convert(T, contain), convert(T, small_volume))
end

"""
    merge_coordinates(values, tol::GeometryTolerance{T}) -> Vector{T}

Sort `values` and return a `Vector{T}` in which neighbouring entries differing by at most `tol.merge` have been
collapsed to their first representative. The result is the canonical per-axis coordinate set used to build admissible
integration intervals in `intersections.jl`: two participating meshes' element-boundary coordinates that are
indistinguishable up to `tol.merge` produce a single shared interval rather than a sliver.

Stable, allocation-light, single-pass after the sort. `values` may be any iterable of reals; the sorted working copy is
the only allocation — the merge dedups in place into its own prefix.
"""
function merge_coordinates(values, tol::GeometryTolerance{T}) where {T}
    sorted = sort!(T[convert(T, value) for value in values])
    isempty(sorted) && return sorted

    # Dedup ascending neighbours in place: keep `sorted[i]` only when it separates from the last kept representative
    # `sorted[n]` by more than `tol.merge`. `sorted` is ascending, so the gap is already non-negative and no `abs` is
    # needed. Survivors are packed into the prefix `1:n`.
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

Axis-aligned physical box in `D` dimensions, represented by its lower and upper corner coordinates. Coordinates are
stored as `SVector{D,T}` and the constructor enforces the non-degenerate invariant `lower[i] < upper[i]` on every axis —
every region the assembly hot loop sees is a proper `D`-cube with positive volume.

Constructors:

  - `box(lower, upper)` — corner-to-corner. Accepts `NTuple{D,<:Real}` or `SVector{D,<:Real}` corners; the scalar type
    is promoted from the inputs.
  - `box(center; halfwidth)` — centered form. `center` is a point; `halfwidth` may be a scalar (isotropic) or a
    `D`-tuple / `SVector` (anisotropic) of half-edge-lengths. Equivalent to
    `box(center − halfwidth, center + halfwidth)`.

Geometry consumers never reach into boxes through `x`/`y`/`z` field names; they iterate over `1:D` and rely on the
dimension-generic [`dimension`](@ref), [`edge_lengths`](@ref), [`volume`](@ref), [`center`](@ref),
[`contains_point`](@ref), and [`reference_to_physical`](@ref) / [`physical_to_reference`](@ref) maps.
"""
struct AxisBox{D,T<:Real}
    lower::SVector{D,T}
    upper::SVector{D,T}

    function AxisBox{D,T}(lower::SVector{D,T}, upper::SVector{D,T}) where {D,T<:Real}
        D >= 1 || throw(ArgumentError("AxisBox dimension must be at least 1"))
        all(upper[i] > lower[i] for i in 1:D) || throw(ArgumentError("AxisBox upper bounds must exceed lower bounds"))
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

Construct an [`AxisBox`](@ref), the dimension-generic axis-aligned box used throughout the package.

  - `box(lower, upper)` takes corner coordinates as an `NTuple{D,<:Real}` or `SVector{D,<:Real}`; the scalar type is
    promoted from the inputs.
  - `box(center; halfwidth)` takes a center point and a `halfwidth` that is either a scalar (isotropic) or a `D`-tuple /
    `SVector` (anisotropic). It is equivalent to `box(center .- halfwidth, center .+ halfwidth)`.

See [`AxisBox`](@ref) for the stored representation and the non-degenerate `lower[i] < upper[i]` invariant the
constructor enforces.
"""
box(lower::NTuple{D,<:Real}, upper::NTuple{D,<:Real}) where {D} = box(SVector(lower), SVector(upper))

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

Spatial dimension `D` of the box, recovered from its static type so the result is a compile-time constant whenever `D`
is statically known.
"""
dimension(::AxisBox{D}) where {D} = D

"""
    edge_lengths(b::AxisBox) -> SVector{D,T}

Per-axis edge lengths, `b.upper − b.lower`. Stored as an `SVector` so downstream Jacobian and gradient-scaling
computations stay stack-allocated.
"""
edge_lengths(b::AxisBox{D,T}) where {D,T} = b.upper - b.lower

"""
    volume(b::AxisBox) -> T

Geometric measure of the box: product of the per-axis edge lengths. In `D` dimensions this is the `D`-cube volume. Used
by the reference-to- physical Jacobian `vol(b) / 2ᴰ` and by the small-overlap diagnostics in `intersections.jl`.
"""
volume(b::AxisBox) = prod(edge_lengths(b))

"""
    center(b::AxisBox) -> SVector{D}

Geometric center of the box, `(b.lower + b.upper) / 2`. Used for midpoint queries — cell-level sign samples and
certificates in `physical.jl` and `implicit.jl`, the post-processing cell samples — and to build the
reference-to-physical affine map below.
"""
center(b::AxisBox) = (b.lower + b.upper) / 2

# The 2ᴰ vertices of the box spanned by `lower` and `upper`, vertex `i` taking `lower[d]` where bit `d − 1` of `i − 1`
# is 0 and `upper[d]` where it is 1 — axis 1 fastest, the `CartesianIndices` order. The one corner enumeration of the
# sign samplers (the kernel's leaf pruning and height-axis choice, the classifier's corner membership, the
# post-processing cut refinement) and of `_box_children` below. The post-processing records read the same corners in
# face-cyclic order (`_box_corners`, VTK's cell order), a fixed permutation of this one.
function _box_vertices(lower::SVector{D,T}, upper::SVector{D,T}) where {D,T}
    return ntuple(i -> SVector{D,T}(ntuple(d -> (i - 1) >> (d - 1) & 1 == 0 ? lower[d] : upper[d], D)), Val(2^D))
end
_box_vertices(b::AxisBox) = _box_vertices(b.lower, b.upper)

# The 2ᴰ children of one bisection of `b` on every axis, in `_box_vertices` order: child `i` spans vertex `i` of the
# lower half-box `[lower, c]` to vertex `i` of the upper one `[c, upper]`. Shared by the classifier's octree
# (`_classify_box`), the implicit kernel's subdivision and the post-processing cut refinement (`_cut_leaves!`).
#
# Returns `nothing` when the bisection would be degenerate. On a box whose extent has reached the floating-point floor
# the midpoint coincides with a bound on some axis, and `AxisBox`'s strict `lower[i] < upper[i]` invariant would reject
# the child; every caller treats `nothing` as its depth floor and stops subdividing, rather than looping on a midpoint
# that no longer moves.
function _box_children(b::AxisBox{D,T}) where {D,T}
    c = center(b)
    all(d -> b.lower[d] < c[d] < b.upper[d], 1:D) || return nothing
    return map(AxisBox{D,T}, _box_vertices(b.lower, c), _box_vertices(c, b.upper))
end

# Half-diagonal of an axis-aligned box — the radius of the smallest enclosing ball. Used by the Lipschitz certificate
# below: for a Lipschitz `f` and any point `y` in the box, |f(y) − f(c)| ≤ L·r with `c` the center and `r` this radius.
_half_diagonal(b::AxisBox) = norm(b.upper - b.lower) / 2

# Sign of `f` over the whole of `box`, certified by a Lipschitz constant `L` of `f` (`|f(x) − f(y)| ≤ L‖x − y‖`): with
# `c` the center and `r = _half_diagonal(box)`,
#
#     |f(y) − f(c)| ≤ L‖y − c‖ ≤ L·r    for every y ∈ box,
#
# so `f(c) > L·r` proves `f > 0` on the box (`+1`) and `f(c) < −L·r` proves `f < 0` (`−1`). Anything else is `0`,
# undetermined — including `L = Inf`, which certifies nothing. One evaluation of `f`. The single certificate of the
# CSG classifier (`_tri`), the implicit kernel's leaf pruning (`_sample_sign`) and the post-processing cut refinement
# (`_cut_marked`), so the three agree on which boxes a leaf provably misses.
function _certified_sign(f, L, box::AxisBox)
    fc = f(center(box))
    thr = L * _half_diagonal(box)
    return fc > thr ? 1 : (fc < -thr ? -1 : 0)
end

# Strict containment without tolerance: a point lies in the box iff every coordinate satisfies
# `lower[i] ≤ x[i] ≤ upper[i]`. Use `contains_point` instead when the query is geometrically motivated and a positive
# `GeometryTolerance.contain` is in scope.
Base.in(point::PointLike{D}, b::AxisBox{D,T}) where {D,T} = all(b.lower[i] <= point[i] <= b.upper[i] for i in 1:D)

"""
    contains_point(point, b::AxisBox, tol::GeometryTolerance) -> Bool

Tolerance-aware point-in-box query: `point` is considered inside `b` when
`b.lower[i] − tol.contain ≤ point[i] ≤ b.upper[i] + tol.contain` on every axis. Used by cell lookup and
integration-region construction to absorb floating-point noise at element boundaries. For an exact (tolerance-free)
query use `point in b`.
"""
function contains_point(point::PointLike{D}, b::AxisBox{D,T}, tol::GeometryTolerance{T}) where {D,T}
    return all(b.lower[i] - tol.contain <= point[i] <= b.upper[i] + tol.contain for i in 1:D)
end

# Structural equality on the corner coordinates. Floating-point exactness is acceptable here because every box this is
# asked about carries coordinates reproduced bit-for-bit by one deterministic construction — the same user-supplied
# corners, the mesh axes of `_mesh_axes` in `mesh.jl`, or the canonicalised region coordinates of `merge_coordinates` —
# never two independently rounded paths to the "same" corner.
Base.:(==)(a::AxisBox{D}, b::AxisBox{D}) where {D} = a.lower == b.lower && a.upper == b.upper

# `isequal` and `hash` go together, as Julia's contract `isequal(a, b) ⇒ hash(a) == hash(b)` requires. With `==` alone
# overloaded, `isequal` fell back to `==` while `hash` stayed the bitwise default, so `box((-0.0, 0), (1, 1))` and
# `box((0.0, 0), (1, 1))`, or a `Float32` and a `Float64` box with the same corners, were `isequal` with different
# hashes, and a `Set` or `Dict` of boxes could hold both or miss either. Both now follow the corners' own `isequal` and
# `hash`, the split Julia's floating-point numbers make: `-0.0 == 0.0`, but they are not `isequal`.
Base.isequal(a::AxisBox, b::AxisBox) = isequal(a.lower, b.lower) && isequal(a.upper, b.upper)
Base.hash(b::AxisBox, h::UInt) = hash(b.upper, hash(b.lower, hash(:AxisBox, h)))

"""
    box_intersection(a::AxisBox{D,T}, b::AxisBox{D,T}) -> Union{AxisBox{D,T},Nothing}

Component-wise intersection of two boxes: per-axis `max(a.lower, b.lower)` and `min(a.upper, b.upper)`. Returns the
resulting box when the intersection has strictly positive measure on every axis, and `nothing` otherwise — degenerate
(zero-volume) intersections are reported as "no overlap" rather than as a flat box. This mirrors the constructor's
non-degeneracy invariant on `AxisBox`.
"""
function box_intersection(a::AxisBox{D,T}, b::AxisBox{D,T}) where {D,T}
    lower = max.(a.lower, b.lower)
    upper = min.(a.upper, b.upper)
    return all(upper[i] > lower[i] for i in 1:D) ? AxisBox{D,T}(lower, upper) : nothing
end

"""
    is_inside(inner::AxisBox{D,T}, outer::AxisBox{D,T}, tol=GeometryTolerance(T)) -> Bool

Tolerance-aware containment of one box inside another: every corner of `inner` lies inside `outer` up to `tol.contain`
on each axis. Used to validate overlay placement, since the `overlay(...)` constructor requires the overlay domain to
lie inside the space's bounding box `V.domain` (not inside an immersed `Ω` — an overlay may cover fictitious material).
"""
function is_inside(inner::AxisBox{D,T}, outer::AxisBox{D,T}, tol::GeometryTolerance{T}=GeometryTolerance(T)) where {D,T}
    return all(inner.lower[i] >= outer.lower[i] - tol.contain && inner.upper[i] <= outer.upper[i] + tol.contain
               for i in 1:D)
end

"""
    reference_to_physical(b::AxisBox{D,T}, ξ) -> SVector{D,T}

Map a reference-cube coordinate `ξ ∈ [−1, 1]ᴰ` to the corresponding physical point in `b` via the affine map

    x = (b.lower + b.upper) / 2 + (b.upper − b.lower) / 2 ∘ ξ

with `∘` the per-axis (Hadamard) product. The constant Jacobian determinant is `vol(b) / 2ᴰ`, which is the factor every
region quadrature rule multiplies its reference weights by in the assembly hot loop.
"""
function reference_to_physical(b::AxisBox{D,T}, xi::PointLike{D}) where {D,T}
    xi_vec = SVector{D,T}(xi)
    return (b.lower + b.upper) / 2 + (b.upper - b.lower) .* xi_vec / 2
end

"""
    physical_to_reference(b::AxisBox{D,T}, x) -> SVector{D,T}

Inverse of [`reference_to_physical`](@ref): map a physical point `x ∈ b` back to its reference-cube coordinate
`ξ ∈ [−1, 1]ᴰ`,

    ξ = (2 x − b.lower − b.upper) / (b.upper − b.lower)

with the division taken per axis. Used to rebase region boxes into parent reference frames during integration-region
construction and to evaluate basis functions at arbitrary physical points during post-processing.
"""
function physical_to_reference(b::AxisBox{D,T}, x::PointLike{D}) where {D,T}
    x_vec = SVector{D,T}(x)
    return (2 .* x_vec - b.lower - b.upper) ./ (b.upper - b.lower)
end
