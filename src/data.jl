# Per-quadrature-point material data and its mesh-change transfer.
#
# `QuadField` is the natural container for per-quadrature-point internal
# state — phase-field damage `H`, plastic strain, predeformation
# gradient, history variables for an inelastic constitutive law — that
# the assembly callbacks consult through `q.point` indexing.
#
# Moving the underlying mesh (overlay activation/deactivation, geometry
# updates, h-/p- refinement) changes the quadrature-point cloud, so a
# `QuadField` belonging to the old model becomes meaningless on the new
# one. `transfer` rebuilds the field on the target model's quadrature
# points via a point-based interpolation scheme; the default scheme is
# [`RBFP0`](@ref), an inverse-multiquadric RBF with a constant
# polynomial extension.

# ── QuadField ────────────────────────────────────────────────────────────────

"""
    QuadField{T}(model; init = q -> zero(T))
    QuadField(data::AbstractVector{T}, model::Model)

`T`-valued data indexed by `q.point` over a `Model`'s quadrature points.
Carries the `model.version` at construction time so reuse against a
stale model raises — the same contract [`Solution`](@ref) uses for
solution coefficients.

Constructors:

  - `QuadField{T}(model; init)` — allocate a fresh field and fill it by
    calling `init(q)` at every quadrature point. The default
    `init = q -> zero(T)` produces an all-zero field.
  - `QuadField(data, model)` — wrap a pre-built vector. The vector
    length must equal `nquadpoints(model)`; the data is copied so the
    caller can mutate its source vector without leaking into the
    `QuadField`.

Material-state history variables (phase-field `H`, plastic strain,
predeformation gradient, …) are the natural use case. Move them across
model changes with [`transfer`](@ref), which reconstructs the field on the
new model's quadrature points through an [`RBFP0`](@ref) interpolation.
"""
mutable struct QuadField{T}
    data::Vector{T}
    model_version::Int
end

function QuadField(data::AbstractVector{T}, model::Model) where {T}
    length(data) == nquadpoints(model) ||
        throw(DimensionMismatch("QuadField data length $(length(data)) does not match nquadpoints(model) = $(nquadpoints(model))"))
    return QuadField{T}(collect(data), model.version)
end

function QuadField{T}(model::Model; init=_ -> zero(T)) where {T}
    n = nquadpoints(model)
    data = Vector{T}(undef, n)
    foreach_quadrature_point(model) do q
        data[q.point] = init(q)
    end
    return QuadField{T}(data, model.version)
end

# Forwarding to the underlying data vector. `setindex!` returns the
# `QuadField` itself so chained updates work; `copy` makes a defensive
# copy of the data and preserves the version pin.
Base.length(qf::QuadField) = length(qf.data)
Base.eltype(::Type{QuadField{T}}) where {T} = T
Base.eltype(qf::QuadField) = eltype(typeof(qf))
Base.getindex(qf::QuadField, i::Integer) = qf.data[i]
Base.setindex!(qf::QuadField, v, i::Integer) = (qf.data[i]=v; qf)
Base.copy(qf::QuadField{T}) where {T} = QuadField{T}(copy(qf.data), qf.model_version)

# Stale-QuadField check used by `transfer` and any consumer that reads
# a `QuadField` against a `Model`. Same contract as
# `_checked_coefficients` in `solvers.jl`: returns the data vector when
# version and length agree, raises a clear error otherwise.
function _checked_quadfield(qf::QuadField, model::Model)
    qf.model_version == model.version ||
        throw(ArgumentError("QuadField belongs to model version $(qf.model_version) but model is at version $(model.version)"))
    length(qf.data) == nquadpoints(model) ||
        throw(DimensionMismatch("QuadField length $(length(qf.data)) does not match nquadpoints(model) = $(nquadpoints(model))"))
    return qf.data
end

function Base.show(io::IO, qf::QuadField{T}) where {T}
    print(io, "QuadField{", T, "}(npoints=", length(qf.data), ", model_version=", qf.model_version,
          ")")
end

# ── Transfer scheme ──────────────────────────────────────────────────────────

"""
    RBFP0(; neighbors=10)

Inverse-multiquadric radial basis function interpolation with a
constant polynomial extension. The interpolant at a target point `x`
is

    f̂(x) = Σ_{i=1}^{K} wᵢ · ψ(‖xᵢ − x‖ / r̄) + α,

with `xᵢ` the `K` nearest source neighbours (`K = neighbors`),
`ψ(r) = 1 / √(1 + r²)` the inverse-multiquadric kernel, `r̄` the mean
pairwise neighbour distance (rescaling so the kernel argument is
dimensionless), and `(w, α)` solved from the `(K+1) × (K+1)` system

    [ A  1 ] [ w ]   [ fᵢ ]
    [ 1ᵀ 0 ] [ α ] = [ 0  ],   Aᵢⱼ = ψ(‖xᵢ − xⱼ‖ / r̄).

The polynomial extension makes the scheme exact for constant fields by
construction and approximately linear-exact for smooth fields. It is
robust at element boundaries (no element-mapping inversion required)
and matches Sartorti & Düster's recommendation for point-based
history-data transfer:

> R. Sartorti, A. Düster, "Data transfer within a finite cell remeshing
> approach applied to large deformation problems", Comput. Mech. 77 (2026),
> [doi:10.1007/s00466-024-02486-0](https://doi.org/10.1007/s00466-024-02486-0).

(See Sec. 3.3.1 of that work for an analysis of why the RBF + P0 scheme
outperforms element-local interpolation in their hyperelastic FCM
remeshing study.)

Neighbour counts up to `_RBFP0_MAX_NEIGHBORS = 16` use a
`StaticArrays.SMatrix` solve and stay BLAS-free; larger counts are not
currently supported.
"""
struct RBFP0
    neighbors::Int
end
RBFP0(; neighbors::Int=10) = RBFP0(neighbors)

# Maximum supported neighbour count for `RBFP0`. The local solve is a
# `(K+1) × (K+1)` SMatrix system; the cap keeps the static-array path
# performant and the per-target memory footprint tight.
const _RBFP0_MAX_NEIGHBORS = 16

# Inverse-multiquadric kernel `ψ(r) = 1 / √(1 + r²)`. Monotone
# decreasing, positive, bounded by 1 at `r = 0`.
_rbf_invmq(r) = inv(sqrt(1 + r * r))

# Build and solve the (K+1) × (K+1) RBF + P0 system as a static
# `SMatrix`. Returns the solution `SVector` (length `K + 1`: K kernel
# weights followed by the constant `α`) plus the `mean_r` rescaling
# used so the evaluation path can reproduce it.
#
# The system has the bordered form
#
#     [ A  1 ] [ w ]   [ fᵢ ]
#     [ 1ᵀ 0 ] [ α ] = [ 0  ],   Aᵢⱼ = ψ(‖xᵢ − xⱼ‖ / r̄),
#
# where the bottom row enforces `Σ wᵢ = 0` (the standard polynomial
# orthogonality condition). `r̄ = mean ‖xᵢ − xⱼ‖` is the natural
# length scale; we floor it at `eps(T)` to handle the degenerate
# single-cluster case.
function _rbfp0_static_solve(local_sources::NTuple{K,SVector{D,T}},
                             local_values::NTuple{K,T}) where {K,D,T}
    mean_r = zero(T)
    pairs = 0
    @inbounds for i in 1:K, j in (i+1):K
        mean_r += norm(local_sources[i] - local_sources[j])
        pairs += 1
    end
    mean_r = pairs == 0 ? one(T) : max(mean_r / pairs, eps(T))
    M = K + 1
    A = MMatrix{M,M,T,M * M}(zeros(T, M, M))
    @inbounds for i in 1:K
        A[i, i] = _rbf_invmq(zero(T))
        for j in (i+1):K
            v = _rbf_invmq(norm(local_sources[i] - local_sources[j]) / mean_r)
            A[i, j] = v
            A[j, i] = v
        end
        A[i, M] = one(T)
        A[M, i] = one(T)
    end
    rhs = MVector{M,T}(zeros(T, M))
    @inbounds for i in 1:K
        rhs[i] = local_values[i]
    end
    return SMatrix(A) \ SVector(rhs), mean_r
end

# Evaluate the RBF + P0 interpolant at `x` given the local solve
# returned by `_rbfp0_static_solve`. Same kernel `ψ(r) = 1 / √(1 + r²)`
# and the same `mean_r` rescaling; the constant `sol[M]` is the
# polynomial extension `α`.
function _rbfp0_evaluate(x::SVector{D,T}, local_sources::NTuple{K,SVector{D,T}}, sol::SVector{M,T},
                         mean_r::T) where {K,D,T,M}
    K + 1 == M || throw(ArgumentError("solution length $M does not match K + 1 = $(K + 1)"))
    result = sol[M]
    @inbounds for i in 1:K
        result += sol[i] * _rbf_invmq(norm(local_sources[i] - x) / mean_r)
    end
    return result
end

# Materialise the source quadrature-point cloud in two complementary
# layouts: a `D × N` `Matrix{T}` for `NearestNeighbors.KDTree`
# compatibility (it indexes points by column) and a
# `Vector{SVector{D,T}}` for the small RBF solve below.
function _collect_source_cloud(source_model::Model{D,T}) where {D,T}
    n = nquadpoints(source_model)
    points = Matrix{T}(undef, D, n)
    svec_points = Vector{SVector{D,T}}(undef, n)
    foreach_quadrature_point(source_model) do q
        x = q.x
        @inbounds for d in 1:D
            points[d, q.point] = x[d]
        end
        svec_points[q.point] = x
    end
    return points, svec_points
end

# ── Public transfer ──────────────────────────────────────────────────────────

"""
    transfer(source::QuadField, source_model, target_model; via=RBFP0(), threaded=true) -> QuadField

Reconstruct `source` on `target_model`'s quadrature-point cloud with the
[`RBFP0`](@ref) scheme `via`. Returns a fresh [`QuadField`](@ref) bound to
`target_model`. The same verb `transfer` moves a [`Solution`](@ref) when given
one.

Algorithm: build a `KDTree` over the source cloud; for each target
quadrature point find the `via.neighbors` nearest source neighbours,
run the local RBF + P0 solve on them, and evaluate the interpolant at
the target point.

The outer loop over target points is embarrassingly parallel: each
target's small solve is independent and uses pure-Julia
`StaticArrays` operations, so the result is bit-identical regardless
of thread count for a fixed neighbour set. `threaded = true` is the
default; pass `threaded = false` for tests that need ordered
evaluation.
"""
function transfer(source::QuadField, source_model::Model{D,T}, target_model::Model{D,T};
                  via::RBFP0=RBFP0(), threaded::Bool=true) where {D,T}
    source_data = _checked_quadfield(source, source_model)
    k = via.neighbors
    1 <= k <= _RBFP0_MAX_NEIGHBORS ||
        throw(ArgumentError("RBFP0 supports 1..$_RBFP0_MAX_NEIGHBORS neighbours; got $k"))
    nsrc = length(source_data)
    nsrc >= k ||
        throw(ArgumentError("RBFP0 needs ≥ neighbors=$k source quadrature points, model only has $nsrc"))

    # Source cloud + KD-tree built once; target points collected once.
    cloud_matrix, cloud_svecs = _collect_source_cloud(source_model)
    tree = KDTree(cloud_matrix; reorder=false)

    n_target = nquadpoints(target_model)
    target_data = Vector{T}(undef, n_target)
    target_points = Vector{SVector{D,T}}(undef, n_target)
    foreach_quadrature_point(target_model) do q
        target_points[q.point] = q.x
    end

    # Per-target work: K-nearest-neighbour query + local RBF + P0 solve.
    # Pulled into a named closure so the threaded and serial branches
    # below share the body.
    function process(i)
        x = target_points[i]
        idxs, _ = knn(tree, x, k, true)
        target_data[i] = _rbfp0_resolve(Val(k), x, idxs, cloud_svecs, source_data)
    end

    if threaded
        Threads.@threads for i in 1:n_target
            process(i)
        end
    else
        for i in 1:n_target
            process(i)
        end
    end

    return QuadField{T}(target_data, target_model.version)
end

# Materialise the `k` source neighbours' coordinates and values as
# `NTuple`s so the small RBF + P0 solve runs on `SMatrix` / `SVector`.
# The `Val(k)` closure keeps the body type-stable: every distinct
# `k ∈ 1:_RBFP0_MAX_NEIGHBORS` produces its own specialised method.
# An explicit `@generated` version would shave the dispatch overhead
# slightly but is not necessary at the current `K ≤ 16` cap.
@inline function _rbfp0_resolve(::Val{K}, x::SVector{D,T}, idxs::AbstractVector{Int},
                                cloud::Vector{SVector{D,T}},
                                source_data::AbstractVector{T}) where {K,D,T}
    local_sources = ntuple(j -> cloud[idxs[j]], Val(K))
    local_values = ntuple(j -> source_data[idxs[j]], Val(K))
    sol, mean_r = _rbfp0_static_solve(local_sources, local_values)
    return _rbfp0_evaluate(x, local_sources, sol, mean_r)
end
