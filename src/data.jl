# Per-quadrature-point material data and its mesh-change transfer.
#
# `QuadField` is the natural container for per-quadrature-point internal state — phase-field damage `H`, plastic strain,
# predeformation gradient, history variables for an inelastic constitutive law — that the assembly callbacks consult
# through `q.point` indexing.
#
# Moving the underlying mesh (overlay activation/deactivation, geometry updates, h-/p- refinement) changes the
# quadrature-point cloud, so a `QuadField` belonging to the old model becomes meaningless on the new one. `transfer`
# rebuilds the field on the target model's quadrature points via a point-based interpolation scheme. [`RBFP0`](@ref) —
# an inverse-multiquadric RBF with a constant polynomial extension — is the only scheme today; `transfer`'s `via`
# keyword is typed to it. It moves one scalar per point, so a component-valued state travels as one `QuadField` per
# component.

# ── QuadField ────────────────────────────────────────────────────────────────

"""
    QuadField{T}(model; init = q -> zero(T))
    QuadField(data::AbstractVector{T}, model::Model)

`T`-valued data indexed by `q.point` over a `Model`'s quadrature points. Carries the `model.version` pin at construction
time so reuse against a mutated model, or against a model of a different discretisation, raises — the same contract
[`Solution`](@ref) uses for solution coefficients, and for the same reason: the pin is a digest of the discretisation,
so two models prepared from the same one share it and two different ones do not.

Constructors:

  - `QuadField{T}(model; init)` — allocate a fresh field and fill it by calling `init(q)` at every quadrature point. The
    default `init = q -> zero(T)` produces an all-zero field.
  - `QuadField(data, model)` — wrap a pre-built vector. The vector length must equal `nquadpoints(model)`; the data is
    copied so the caller can mutate its source vector without leaking into the `QuadField`.

Material-state history variables (phase-field `H`, plastic strain, predeformation gradient, …) are the natural use case.
Move them across model changes with [`transfer`](@ref), which reconstructs the field on the new model's quadrature
points through an [`RBFP0`](@ref) interpolation. The container places no restriction on `T`, but that transfer does: it
takes a `QuadField{T}` on a `Model{D,T}` and nothing else, so a component-valued state is carried one component per
`QuadField`.

Indexing is checked, and the check also guards a read from a cloud this field does not belong to. [`estimate`](@ref)
re-evaluates the problem's blocks and loads on an order-elevated twin with a different and larger set of points, so a
`qf[q.point]` there would read out of range or — worse — silently at the wrong point. The twin numbers its points far
below 1 instead, so such a read raises an `ArgumentError` naming the cause, on whichever task or thread makes it, and
`estimate` refuses outright when it can see the field among a form's captures. The numbering is the twin's own and
nothing is set process-wide, so ordinary assembly, other tasks and threads, and every read against the model the field
was built on are unaffected. The check is an explicit call, which a caller's `@inbounds` does not remove.

Both constructors are single-domain only and raise on a coupled model. `q.point` restarts at 1 for each subdomain, so
one flat array — sized by `nquadpoints(model)`, the sum over the subdomains, as `QuadField(data, model)` checks — would
alias the subdomains onto each other, and `QuadField{T}(model; init)`, which walks the volume through
[`foreach_quadrature_point`](@ref), has no single list to walk there either.
"""
mutable struct QuadField{T}
    data::Vector{T}
    model_version::Int
end

function QuadField(data::AbstractVector{T}, model::Model) where {T}
    _assert_single_domain(model, "QuadField")
    length(data) == nquadpoints(model) ||
        throw(DimensionMismatch("QuadField data length $(length(data)) does not match " *
                                "nquadpoints(model) = $(nquadpoints(model))"))
    return QuadField{T}(collect(data), model.version)
end

function QuadField{T}(model::Model; init=_ -> zero(T)) where {T}
    _assert_single_domain(model, "QuadField")
    n = nquadpoints(model)
    data = Vector{T}(undef, n)
    foreach_quadrature_point(model) do q
        data[q.point] = init(q)
    end
    return QuadField{T}(data, model.version)
end

# Forwarding to the underlying data vector. `setindex!` returns the `QuadField` itself so chained updates work; `copy`
# makes a defensive copy of the data and preserves the version pin.
#
# `getindex` checks by an explicit call rather than `@boundscheck`, so a caller's `@inbounds` cannot remove the check:
# it is also what refuses a `q.point` of `estimate`'s twin (`_out_of_range`, under "Detecting point-keyed state"
# below), and it is the one compare the vector's own `getindex` would make, so the guard costs a read nothing.
Base.length(qf::QuadField) = length(qf.data)
Base.eltype(::Type{QuadField{T}}) where {T} = T
Base.eltype(qf::QuadField) = eltype(typeof(qf))
@inline function Base.getindex(qf::QuadField, i::Integer)
    checkbounds(Bool, qf.data, i) || _out_of_range(qf, i)
    return @inbounds qf.data[i]
end
Base.setindex!(qf::QuadField, v, i::Integer) = (qf.data[i]=v; qf)
Base.copy(qf::QuadField{T}) where {T} = QuadField{T}(copy(qf.data), qf.model_version)

# Stale-QuadField check used by `transfer` and any consumer that reads a `QuadField` against a `Model`. Same contract as
# `_checked_coefficients` in `solvers.jl`: returns the data vector when the pin and the length agree, raises a clear
# error otherwise.
function _checked_quadfield(qf::QuadField, model::Model)
    qf.model_version == model.version ||
        throw(ArgumentError("this QuadField was built on a different discretisation, or on this model " *
                            "before it was mutated (field pin 0x$(string(qf.model_version; base=16)), " *
                            "model pin 0x$(string(model.version; base=16))); carry it across with " *
                            "`transfer`"))
    length(qf.data) == nquadpoints(model) ||
        throw(DimensionMismatch("QuadField length $(length(qf.data)) does not match " *
                                "nquadpoints(model) = $(nquadpoints(model))"))
    return qf.data
end

function Base.show(io::IO, qf::QuadField{T}) where {T}
    print(io, "QuadField{", T, "}(npoints=", length(qf.data), ", pin=0x", string(qf.model_version; base=16), ")")
end

# ── Detecting point-keyed state ──────────────────────────────────────────────
#
# `q.point` is an index into ONE model's quadrature cloud. A verb that re-evaluates forms it did not write, on a second
# cloud — `estimate`, which assembles the caller's own blocks and loads on an order-elevated twin — therefore hands a
# form built for `model` the twin's indices: out of range where the base plan is the smaller of the two, and silently at
# the wrong points where it is not. The `QuadField` pin already separates the two models, and has since the pin became a
# digest of the discretisation rather than a counter. What it could not do is reach the indexing site: `qf[q.point]` is
# an ordinary `getindex` on a vector, and the form owns the call.
#
# Two layers bring the identity there, and they cover different cases because a form can hold its state in two different
# ways.
#
# The first is this walk, run once by `estimate` before it builds anything. It looks for a `QuadField` among the form's
# own captures, which is where the documented mechanism puts it: `conduction(chi)` returning a closure over `chi` is the
# shape `examples/applications/thermal_curing_2d` uses and the one the `QuadField` docstring describes. Finding it there
# costs one reflective walk per `estimate` call, nothing per point, and lets the refusal name the field and both pins
# before the expensive half of the call.
#
# The walk is bounded rather than exhaustive: it follows struct fields and the elements of tuples, named tuples, arrays
# and dictionaries to `depth` levels, skipping arrays of bits types (coefficient vectors, masks, meshes), and returns
# the first field it finds. Missing one is a false negative and never a false alarm, which is the safe direction for a
# guard that raises. It reads only what is there: an undefined struct field is skipped, and so is an unassigned array
# slot (`_assigned`) — a scratch buffer such as `Vector{Any}(undef, n)` has one, and reading it raised an
# `UndefRefError` out of `estimate` before the walk reached anything. A dictionary is walked by its values, never looked
# up again by its keys: a key mutated after insertion no longer hashes to its slot, and a lookup by it raises.
function _captured_quadfield(x, depth::Int=6)
    x isa QuadField && return x
    depth <= 0 && return nothing
    if x isa Union{Tuple,NamedTuple,AbstractArray,AbstractDict}
        x isa AbstractArray && isbitstype(eltype(x)) && return nothing
        for v in (x isa AbstractDict ? values(x) : x isa AbstractArray ? _assigned(x) : x)
            found = _captured_quadfield(v, depth - 1)
            found === nothing || return found
        end
        return nothing
    end
    isbits(x) && return nothing
    for i in 1:nfields(x)
        isdefined(x, i) || continue
        found = _captured_quadfield(getfield(x, i), depth - 1)
        found === nothing || return found
    end
    return nothing
end

# The elements an array holds, in linear order — every array supports linear indexing — passing over each slot
# `isassigned` reports empty, which only an array of a non-bits element type can have.
_assigned(x::AbstractArray) = (x[i] for i in LinearIndices(x) if isassigned(x, i))

# Refuse `verb` on a problem whose forms carry per-quadrature-point state. The message names the field's own pin, which
# is what identifies the model its `q.point` numbering belongs to. Called by `estimate` (`adaptivity.jl`), whose name it
# takes as `verb`.
function _assert_no_point_keyed_state(problem, verb::AbstractString)
    for form in Iterators.flatten((problem.blocks, problem.loads))
        qf = _captured_quadfield(form.form)
        qf === nothing && continue
        throw(ArgumentError("$verb: a form of this problem captures a $(typeof(qf)) (pin " *
                            "0x$(string(qf.model_version; base=16))), which is indexed by " *
                            "`q.point`. This call re-evaluates the problem's own blocks and " *
                            "loads on a second integration plan, whose quadrature cloud has " *
                            "more points per cell and puts them elsewhere, so those reads " *
                            "would land out of range or — worse — on the wrong points. Read " *
                            "the state through the physical point `q.x` instead, or run this " *
                            "on a model whose forms carry none."))
    end
    return nothing
end

# The second layer is the twin's point numbering, and it exists because the first has one blind spot that reflection
# cannot close: a form written at top level reads a *global* binding rather than capturing it, so its closure has no
# fields at all and carries no evidence of the state it reads. `estimate` prepares the twin with its points numbered
# from `_FOREIGN_POINTS`, far below 1, so every `q.point` the twin hands out lies outside every array a caller can hold.
# A `QuadField` read raises the `ArgumentError` of `_out_of_range`, which names the cause; a read from a plain array
# raises Julia's own `BoundsError`. Neither can raise on anything legitimate: only the twin numbers its points there,
# and it is built fresh inside the call.
#
# The numbering is the twin's own (`AssemblyCache.point_base`), so nothing is set process-wide. A flag set around the
# twin's assembly would also stop the read, but it is seen by every task: it raises on an unrelated read in another
# task while it is set, and two estimates interleaved restore it out of order and leave it set for good. Here another
# task assembling, another `estimate` in flight and every thread of the twin's own threaded assembly see exactly what
# they would see alone, and nothing outlives the call. It also costs the accessor nothing: the bounds check is the whole
# guard, and only its failure path reads the constant.
#
# The constant has three properties, and the suite asserts each:
#
#   - every point of any cloud the twin could have, `_FOREIGN_POINTS + n` with `n` far below 2⁶⁰, is negative and below
#     `_FOREIGN_POINTS ÷ 2`, the line `_out_of_range` reads as "a point of the twin";
#   - it is far below any index written by hand, at about −2.85e18;
#   - it does not wrap. An `@inbounds v[q.point]` on a raw vector reads the address `v + (q.point − 1)·s` for elements
#     of `s` bytes, modulo 2⁶⁴, and a base whose product with `s` is ≡ 0 would put that read exactly on element `k` —
#     the silent misread the numbering is there to prevent. `typemin(Int) ÷ 2 = −2⁶²` is such a base for every `s`
#     divisible by 4. This one is the 64-bit golden-ratio constant shifted right by two, whose multiples spread evenly
#     modulo 2⁶⁴: for every `s` from 1 to 64 bytes `_FOREIGN_POINTS·s` stays at least 2⁵⁷ away from 0 modulo 2⁶⁴, so
#     such a read lands far outside the vector rather than on one of its elements.
const _FOREIGN_POINTS = -reinterpret(Int, 0x9E3779B97F4A7C15 >> 2)

# The error of an index outside a field's data, raised out of line so the accessor keeps the single compare of its
# bounds check. An index below `_FOREIGN_POINTS ÷ 2` is a `q.point` of `estimate`'s twin and is named as such; any other
# raises the `BoundsError` the data vector itself would.
@noinline function _out_of_range(qf::QuadField, i::Integer)
    i < _FOREIGN_POINTS ÷ 2 || throw(BoundsError(qf.data, i))
    throw(ArgumentError("this QuadField (pin 0x$(string(qf.model_version; base=16))) is being read at a `q.point` " *
                        "of another model's quadrature cloud: `estimate` re-evaluates the problem's forms on an " *
                        "enriched model of its own, whose points it numbers far below 1 so that such a read raises " *
                        "rather than landing on the wrong point. Read the state through the physical point `q.x`, " *
                        "or estimate on a model whose forms carry no per-point state."))
end

# ── Transfer scheme ──────────────────────────────────────────────────────────

"""
    RBFP0(; neighbors=10)

Inverse-multiquadric radial basis function interpolation with a constant polynomial extension. The interpolant at a
target point `x` is

    f̂(x) = Σ_{i=1}^{K} wᵢ · ψ(‖xᵢ − x‖ / r̄) + α,

with `xᵢ` the `K` nearest source neighbours (`K = neighbors`), `ψ(r) = 1 / √(1 + r²)` the inverse-multiquadric kernel,
`r̄` the mean pairwise neighbour distance (rescaling so the kernel argument is dimensionless), and `(w, α)` solved from
the `(K+1) × (K+1)` system

    [ A  1 ] [ w ]   [ fᵢ ]
    [ 1ᵀ 0 ] [ α ] = [ 0  ],   Aᵢⱼ = ψ(‖xᵢ − xⱼ‖ / r̄).

The polynomial extension makes the scheme exact for constant fields by construction and approximately linear-exact for
smooth fields. It is robust at element boundaries (no element-mapping inversion required) and matches Sartorti &
Düster's recommendation for point-based history-data transfer:

> R. Sartorti, A. Düster, "Data transfer within a finite cell remeshing approach applied to large deformation problems",
> Comput. Mech. 77 (2026), [doi:10.1007/s00466-024-02486-0](https://doi.org/10.1007/s00466-024-02486-0).

(See Sec. 3.3.1 of that work for an analysis of why the RBF + P0 scheme outperforms element-local interpolation in their
hyperelastic FCM remeshing study.)

Neighbour counts up to `_RBFP0_MAX_NEIGHBORS = 16` are supported; larger counts are not. Up to 13 neighbours the local
system is solved as a `StaticArrays.SMatrix`, in pure Julia and without the heap. From 14 on it is larger than the
14 × 14 up to which StaticArrays factors a static matrix itself, and it is solved as a dense `Matrix` through LAPACK.
"""
struct RBFP0
    neighbors::Int
end
RBFP0(; neighbors::Int=10) = RBFP0(neighbors)

# Maximum supported neighbour count for `RBFP0`. The local solve is a `(K+1) × (K+1)` SMatrix system; the cap keeps the
# static-array path performant and the per-target memory footprint tight.
const _RBFP0_MAX_NEIGHBORS = 16

# ── Public transfer ──────────────────────────────────────────────────────────

"""
    transfer(source::QuadField, source_model, target_model; via=RBFP0(), threaded=true) -> QuadField

Reconstruct `source` on `target_model`'s quadrature-point cloud with the [`RBFP0`](@ref) scheme `via`. Returns a fresh
[`QuadField`](@ref) bound to `target_model`. The same verb `transfer` moves a [`Solution`](@ref) when given one.

`source` must be a `QuadField{T}` over the models' own scalar type `T`: the local solve carries neighbour coordinates
and neighbour values in one type, so a component-valued state is transferred one component at a time. That one is not an
argument check — a mismatched element type surfaces as a `MethodError` from the local solve. Two `ArgumentError`s guard
the neighbour count — `1 ≤ via.neighbors ≤ 16` (the static-solve cap documented on [`RBFP0`](@ref)) and at least
`via.neighbors` quadrature points on the source model.

Algorithm: collect the source and the target quadrature-point clouds, one walk each, and build a `KDTree` over the
source cloud; for each target quadrature point find the `via.neighbors` nearest source neighbours, run the local
RBF + P0 solve on them, and evaluate the interpolant at the target point.

The outer loop over target points is embarrassingly parallel: each target's small solve is independent and runs the
same arithmetic whichever thread takes it, so the result is bit-identical regardless of thread count for a fixed
neighbour set — the dense solve from 14 neighbours on included, which agreed bit for bit at every count from 1 to 16 on
eight threads. `threaded = true` is the default; `threaded = false` runs the same loop serially, which is how the suite
asserts that the two paths agree bit for bit.
"""
function transfer(source::QuadField, source_model::Model{D,T}, target_model::Model{D,T}; via::RBFP0=RBFP0(),
                  threaded::Bool=true) where {D,T}
    source_data = _checked_quadfield(source, source_model)
    k = via.neighbors
    1 <= k <= _RBFP0_MAX_NEIGHBORS || throw(ArgumentError("RBFP0 supports 1..$_RBFP0_MAX_NEIGHBORS neighbours; got $k"))
    nsrc = length(source_data)
    nsrc >= k || throw(ArgumentError("RBFP0 needs ≥ neighbors=$k source quadrature points, model only has $nsrc"))
    cloud = _quadrature_cloud(source_model)
    targets = _quadrature_cloud(target_model)
    out = Vector{T}(undef, length(targets))
    _rbfp0_transfer!(Val(k), out, KDTree(cloud; reorder=false), cloud, source_data, targets, threaded)
    return QuadField{T}(out, target_model.version)
end

# The physical coordinates of `model`'s quadrature points in `q.point` order, from one walk: RBFP0 reads it once for the
# source cloud its KD-tree indexes and once for the target points it evaluates at. `NearestNeighbors.KDTree` builds its
# tree over exactly such a `Vector{SVector}` — handed a `D × N` matrix it first copies the columns into one — so the
# vector is passed as it is, and the tree, the neighbours found and the result are those of the matrix form.
function _quadrature_cloud(model::Model{D,T}) where {D,T}
    cloud = Vector{SVector{D,T}}(undef, nquadpoints(model))
    foreach_quadrature_point(q -> (cloud[q.point] = q.x), model)
    return cloud
end

# The per-point loop of the RBFP0 transfer, behind the one dynamic dispatch a transfer makes: `Val(K)` turns the runtime
# neighbour count into a compile-time one, so this body and the static solve below it are compiled once per `K` and run
# with no dispatch per point. Per target point it finds the `K` nearest source points, sorted by distance, and writes
# the local interpolant's value there. Each target is independent, so the threaded loop writes the same values as the
# serial one, bit for bit.
#
# What remains on the heap per point is `knn`'s two result vectors — four allocations on Julia 1.12, which counts an
# array and its memory block separately — and, from `K = 14` on, the solve's (see `_rbfp0_local`). `knn!` fills
# preallocated vectors, but it exists only from NearestNeighbors 0.4.19 on, and the package's compat admits 0.4.8.
function _rbfp0_transfer!(::Val{K}, out::Vector{T}, tree, cloud::Vector{SVector{D,T}}, data,
                          targets::Vector{SVector{D,T}}, threaded::Bool) where {K,D,T}
    function point!(i)
        nearest = first(knn(tree, targets[i], K, true))
        sources, values = ntuple(j -> cloud[nearest[j]], Val(K)), ntuple(j -> data[nearest[j]], Val(K))
        out[i] = _rbfp0_local(targets[i], sources, values)
        return nothing
    end
    if threaded
        Threads.@threads for i in eachindex(targets)
            point!(i)
        end
    else
        foreach(point!, eachindex(targets))
    end
    return out
end

# The RBF + P0 interpolant through the `K` neighbours `sources` with the values `values`, evaluated at `x`. It solves
# the bordered (K+1) × (K+1) system
#
#     [ A  1 ] [ w ]   [ fᵢ ]
#     [ 1ᵀ 0 ] [ α ] = [ 0  ],   Aᵢⱼ = ψ(‖xᵢ − xⱼ‖ / r̄),
#
# whose bottom row enforces `Σ wᵢ = 0` (the standard polynomial orthogonality condition), and returns
# `α + Σ wᵢ ψ(‖xᵢ − x‖ / r̄)`. `r̄ = mean ‖xᵢ − xⱼ‖` is the natural length scale; it is floored at `eps(T)` so a cluster
# of coincident neighbours cannot divide by zero, and falls back to `1` at `K = 1`, where there is no pair to average.
# The fallback value is arbitrary there: the `K = 1` system is `[1 1; 1 0] [w; α] = [f₁; 0]`, whose solution `w = 0`,
# `α = f₁` is nearest-neighbour extension at any scale.
#
# The matrix is filled in place as an `MMatrix` and solved as an `SMatrix`. It does not escape, so an optimised build
# keeps it off the heap, where `MMatrix(zeros(T, M, M))` used to build a heap `Matrix` first and the right-hand side a
# second heap vector at every target point: up to `K = 13` the whole solve allocates nothing. A larger system exceeds
# the 14 × 14 up to which StaticArrays factors a static matrix itself, and its `\` converts to a dense `Matrix` and
# solves that, 8 to 10 allocations per point at `K = 14` to 16. A neighbour value of another type than the coordinates'
# has no method here, which is the `MethodError` `transfer`'s docstring describes.
function _rbfp0_local(x::SVector{D,T}, sources::NTuple{K,SVector{D,T}}, values::NTuple{K,T}) where {K,D,T}
    mean_r = zero(T)
    @inbounds for i in 1:K, j in (i+1):K
        mean_r += norm(sources[i] - sources[j])
    end
    pairs = K * (K - 1) ÷ 2
    mean_r = pairs == 0 ? one(T) : max(mean_r / pairs, eps(T))
    A = zeros(MMatrix{K + 1,K + 1,T})
    @inbounds for i in 1:K
        A[i, i] = _rbf_invmq(zero(T))
        for j in (i+1):K
            A[i, j] = A[j, i] = _rbf_invmq(norm(sources[i] - sources[j]) / mean_r)
        end
        A[i, K+1] = A[K+1, i] = one(T)
    end
    sol = SMatrix(A) \ SVector(values..., zero(T))
    result = sol[K+1]
    @inbounds for i in 1:K
        result += sol[i] * _rbf_invmq(norm(sources[i] - x) / mean_r)
    end
    return result
end

# Inverse-multiquadric kernel `ψ(r) = 1 / √(1 + r²)`. Monotone decreasing, positive, bounded by 1 at `r = 0`.
_rbf_invmq(r) = inv(sqrt(1 + r * r))
