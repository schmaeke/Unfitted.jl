# ── Basis family interface ────────────────────────────────────────────────────

"""
    BasisFamily

Abstract supertype for every basis family supported by the package. A basis
family answers, for each cell or span, the questions assembly and the dof
layer ask:

  - how many local basis functions live on a cell;
  - which multi-indices identify them and in what tensor-product order;
  - what their values and gradients are at a reference point, possibly
    pulled back from a physical cell;
  - which of them touch a given axis-side of the cell (used for facet
    identification by the dof layer and the Dirichlet projection);
  - what quadrature order integrates products of two basis functions
    exactly on a `:full` region.

The concrete implementation shipped here is [`IntegratedLegendre`](@ref); the
`BasicBSpline` package extension adds a second family. Adding a family means
implementing this interface plus the corresponding dof/constraint behaviour —
geometry, intersections, assembly, projection, and solvers stay untouched.

A new family `F <: BasisFamily` provides:

  - `instantiate_basis(spec_or_F, mesh, order)` — build the per-level basis;
  - `local_basis_indices(::F, order[, mode])` and `local_basis_count(::F,
    order, mode)` — the cell-local multi-indices and their count, including any
    order/mode validation;
  - `is_boundary_basis(::F, id, axis, side)` — per-axis facet incidence;
  - `basis_values` / `physical_basis_gradients` — the hot-path evaluation;
  - the dof-key and overlay-constraint hooks the dof layer needs.

It inherits the basis-agnostic `::BasisFamily` defaults — `recommended_
quadrature_order` (`order .+ 1`), the tensor `local_basis_count` (`prod(order
.+ 1)`), `is_facet_basis`, and `boundary_basis_indices` — and overrides any
that do not fit. The integrated Legendre family (this file) and the B-spline
extension are the two worked examples.
"""
abstract type BasisFamily end

"""
    IntegratedLegendre()

Default basis family: hierarchical integrated Legendre tensor-product basis
on axis-aligned Cartesian cells.

In each axis the 1D modes are

    N̂₀(ξ) = (1 − ξ) / 2
    N̂₁(ξ) = (1 + ξ) / 2
    N̂ₘ(ξ) = (Lₘ(ξ) − Lₘ₋₂(ξ)) / √(4m − 2)        for m ≥ 2

where Lₘ is the standard Legendre polynomial. Modes 0 and 1 are the
linear endpoint shape functions; modes m ≥ 2 are bubble functions that
vanish at the cell boundaries (`N̂ₘ(±1) = 0`). The `√(4m − 2)`
normalisation makes the L²(−1, 1) inner products `⟨N̂ₘ′, N̂ₙ′⟩` diagonal
for the bubble block, which keeps the 1D stiffness matrix well-scaled at
high order.

`D`-dimensional basis functions are tensor products of the 1D modes:

    N_α(ξ) = ∏_{d=1}^{D} N̂_{α_d}(ξ_d),   α = (α₁, …, α_D),   0 ≤ α_d ≤ p_d

Two index sets are available, set by the `mode` keyword on
[`space`](@ref) / [`overlay`](@ref):

  - `:tensor` — the full tensor product `∏_d {0, …, p_d}` (default).
  - `:trunk` — the Szabó–Babuška trunk (serendipity) space at isotropic
    order `p`. Keep every multi-index whose *trunk degree*
    `Σ_d t(α_d) ≤ p`, where `t(α_d) = α_d` for a bubble mode `α_d ≥ 2`
    and `t(α_d) = 0` for the two linear endpoint modes `α_d ∈ {0, 1}`.
    This retains all vertex modes and every edge mode up to degree `p`
    (so `C⁰` inter-cell conformity is preserved) while trimming the
    high-degree interior face/volume modes to `Σ` of their bubble
    degrees `≤ p`. Requires isotropic order. The saving over `:tensor`
    grows with dimension: e.g. for `p = 4` a hexahedral cell carries 50
    modes instead of 125.

All integrated-Legendre-specific dof and constraint behaviour
(endpoint-vs-bubble identification, boundary-mode dispatch, facet
detection) lives in this file and `src/dofs.jl`; the rest of the codebase
queries this family through the [`BasisFamily`](@ref) interface.
"""
struct IntegratedLegendre <: BasisFamily end

"""
    basis_name(basis) -> Symbol

Short identifier used in error messages and diagnostics output.
"""
basis_name(::IntegratedLegendre) = :integrated_legendre

# ── Quadrature rules ──────────────────────────────────────────────────────────

"""
    TensorQuadrature{D,T}(points, weights)

Tensor-product Gauss–Legendre quadrature on the reference cube `[−1, 1]ᴰ`.
Built once per `(per_axis_counts, T)` pair by [`_tensor_gauss_rule`](@ref)
and cached, so all `:full` regions of the same per-axis order share the
same `points`/`weights` arrays.
"""
struct TensorQuadrature{D,T<:Real}
    points::Vector{SVector{D,T}}
    weights::Vector{T}
end

"""
    RegionQuadrature{D,T}(kind, points, weights)

Per-region quadrature rule consumed by the assembly hot loop. `kind` is an
informational tag set by the integration-region dispatcher in
`intersections.jl`:

  - `:full` — region is entirely inside Ω (or no `PhysicalDomain` is
    attached). `points`/`weights` alias the shared cached tensor Gauss
    rule; no copy is made.
  - `:fictitious_alpha` — region is entirely outside Ω under α-FCM;
    weights are the tensor Gauss weights pre-multiplied by `α` and
    shared across regions of the same per-axis order via a per-plan
    cache.
  - `:cut_fitted` — region is crossed by ∂Ω; weights come from the NNMF
    moment-fit rule (see `src/fcm.jl`) and are unique per region. Under
    α-FCM (`α > 0`) the rule additionally carries the α-scaled full-cell
    tensor part (`(1−α)·moment-fit ∪ α·tensor`).
  - `:cut_fallback` — the moment-fit residual exceeded the catastrophic
    threshold, so the region carries the raw Saye volume rule the moments
    were summed from (see [`moment_fit_rule`](@ref)) instead of a fitted
    one: correct and non-negative, but with 50–200× the points. Under
    α-FCM the α-scaled tensor part is appended as for `:cut_fitted`.
  - `:cut_failed` — strict-cut (`α = 0`) region whose `Ω ∩ box` carries no
    volume rule at all; the region was emitted with empty
    `points`/`weights` so it contributes zero quadrature.
  - `:cut_alpha_failed` — same under α-FCM (`α > 0`); the physical part is
    empty but the region carries the α-scaled tensor rule alone so the
    cell's dofs stay α-stabilised rather than singular.

All three non-fitted kinds are reported in
`AssemblyDiagnostics.fit_failure_count`; `:cut_fallback` is additionally
counted, with its point cost, in `cut_fallback_count` and
`cut_fallback_points`.

Assembly iterates `zip(points, weights)` without dispatching on `kind`,
so the hot loop's cost is the same regardless of the quadrature source.
"""
struct RegionQuadrature{D,T<:Real}
    kind::Symbol
    points::Vector{SVector{D,T}}
    weights::Vector{T}
end

# Build a fresh tensor Gauss–Legendre rule with `counts[d]` points along
# axis `d`. The per-axis 1D rules come from FastGaussQuadrature; we then
# enumerate `CartesianIndices(counts)` to build the `prod(counts)` tensor
# points and the corresponding product weights. Always called through the
# cache in `intersections.jl`, never on the hot path.
function _tensor_gauss_rule(counts::NTuple{D,Int}, ::Type{T}) where {D,T<:Real}
    axes = ntuple(D) do d
        points, weights = gausslegendre(counts[d])
        (T.(points), T.(weights))
    end

    npoints = prod(counts)
    points = Vector{SVector{D,T}}(undef, npoints)
    weights = Vector{T}(undef, npoints)
    counter = 1

    for index in CartesianIndices(counts)
        points[counter] = SVector{D,T}(ntuple(d -> axes[d][1][index.I[d]], D))
        weights[counter] = prod(axes[d][2][index.I[d]] for d in 1:D)
        counter += 1
    end

    return TensorQuadrature{D,T}(points, weights)
end

"""
    recommended_quadrature_order(basis::IntegratedLegendre, order) -> NTuple{D,Int}

Per-axis number of Gauss–Legendre points that integrates products of two
basis functions of polynomial order `order[d]` exactly. The `::BasisFamily`
default is `order[d] + 1`: an `n`-point Gauss rule is exact for polynomials up
to degree `2n − 1`, which covers the degree `2 · order[d]` of a product of two
degree-`order[d]` modes. A family whose functions are not degree-`order[d]`
polynomials overrides this.
"""
function recommended_quadrature_order(::BasisFamily, order::NTuple{D,Int}) where {D}
    return ntuple(i -> order[i] + 1, D)
end

"""
    local_basis_count(basis, order)         -> Int
    local_basis_count(basis, order, mode)   -> Int

Number of local basis functions on a single cell. The two-argument form is the
tensor count `prod(order .+ 1)`, a `::BasisFamily` default for tensor-product
families. The three-argument form takes a basis mode and is family-specific:
for integrated Legendre `:trunk` counts the multi-indices passing the
trunk-degree filter (see [`IntegratedLegendre`](@ref)); other families may
restrict the supported modes.
"""
function local_basis_count(::BasisFamily, order::NTuple{D,Int}) where {D}
    return prod(ntuple(i -> order[i] + 1, D))
end

# Validate a basis-mode symbol against the per-axis polynomial order.
# `:tensor` accepts anisotropic orders; `:trunk` requires isotropic order,
# since the trunk-degree filter compares against a single scalar `p`.
function _check_basis_mode(mode::Symbol, order::NTuple{D,Int}) where {D}
    mode in (:tensor, :trunk) || throw(ArgumentError("basis mode must be :tensor or :trunk"))
    if mode === :trunk && any(!=(order[1]), order)
        throw(ArgumentError("mode=:trunk requires isotropic order"))
    end
    return mode
end

# ── 1D Legendre primitives ────────────────────────────────────────────────────

"""
    legendre_value(n, x) -> Real

Standard Legendre polynomial `Lₙ(x)` evaluated via the three-term
recurrence

    L₀(ξ) = 1
    L₁(ξ) = ξ
    Lₘ(ξ) = ((2m − 1) ξ Lₘ₋₁(ξ) − (m − 1) Lₘ₋₂(ξ)) / m,     m ≥ 2.

Used by the per-mode value/derivative wrappers below for testing and one-
off evaluation. The hot-path assembly route does not call this — see
[`_fill_factor_tables!`](@ref) for the fused per-axis recurrence that
serves it.
"""
function legendre_value(n::Integer, x::Real)
    n >= 0 || throw(ArgumentError("Legendre index must be nonnegative"))
    T = typeof(float(x))
    ξ = convert(T, x)

    n == 0 && return one(T)
    n == 1 && return ξ

    pnm2 = one(T)
    pnm1 = ξ
    pn = zero(T)
    for k in 2:n
        pn = ((2k - 1) * ξ * pnm1 - (k - 1) * pnm2) / k
        pnm2, pnm1 = pnm1, pn
    end

    return pn
end

"""
    integrated_legendre_value(i, ξ) -> Real

Integrated Legendre 1D basis value `N̂ᵢ(ξ)` for `i ≥ 0` on the reference
interval `[−1, 1]`:

    N̂₀(ξ) = (1 − ξ) / 2
    N̂₁(ξ) = (1 + ξ) / 2
    N̂ᵢ(ξ) = (Lᵢ(ξ) − Lᵢ₋₂(ξ)) / √(4i − 2),     i ≥ 2.

Modes 0 and 1 are the linear endpoint shape functions; modes i ≥ 2 are
bubbles that vanish at ξ = ±1.
"""
function integrated_legendre_value(i::Integer, ξ::Real)
    i >= 0 || throw(ArgumentError("basis index must be nonnegative"))
    T = typeof(float(ξ))
    x = convert(T, ξ)

    i == 0 && return (one(T) - x) / 2
    i == 1 && return (one(T) + x) / 2

    return (legendre_value(i, x) - legendre_value(i - 2, x)) / sqrt(convert(T, 4i - 2))
end

"""
    integrated_legendre_derivative(i, ξ) -> Real

Derivative `N̂ᵢ′(ξ)` of the integrated Legendre 1D basis function with
respect to the reference coordinate:

    N̂₀′(ξ) = −1/2
    N̂₁′(ξ) = +1/2
    N̂ᵢ′(ξ) = √((2i − 1) / 2) · Lᵢ₋₁(ξ),     i ≥ 2.

The closed form for `i ≥ 2` follows from differentiating
`N̂ᵢ = (Lᵢ − Lᵢ₋₂) / √(4i − 2)` and applying the standard Legendre
identity `Lₘ′ = ((2m − 1) Lₘ₋₁ + Lₘ₋₃′)`.
"""
function integrated_legendre_derivative(i::Integer, ξ::Real)
    i >= 0 || throw(ArgumentError("basis index must be nonnegative"))
    T = typeof(float(ξ))
    x = convert(T, ξ)

    i == 0 && return -one(T) / 2
    i == 1 && return one(T) / 2

    return sqrt(convert(T, 2i - 1) / 2) * legendre_value(i - 1, x)
end

# ── Local basis indexing ──────────────────────────────────────────────────────

"""
    local_basis_indices(basis::IntegratedLegendre, order)         -> Vector{CartesianIndex{D}}
    local_basis_indices(basis::IntegratedLegendre, order, mode)   -> Vector{CartesianIndex{D}}

Multi-indices `α = (α₁, …, α_D)` enumerating the local basis functions on
a cell of polynomial order `order`. Lexicographic over
`CartesianIndices(ntuple(d -> 0:order[d], D))`, axis 1 varying fastest —
this is the canonical tensor-product ordering used by every basis
consumer (assembly, projection, post-processing, the dof layer).

`mode` defaults to `:tensor` (no filter); `mode = :trunk` keeps only
indices with trunk degree `Σ_d t(α_d) ≤ p`, where `t(α_d) = α_d` for a
bubble mode (`α_d ≥ 2`) and `t(α_d) = 0` for the two linear endpoint
modes (`α_d ∈ {0, 1}`) — the Szabó–Babuška trunk space (isotropic order
`p` required, see [`IntegratedLegendre`](@ref)).
"""
function local_basis_indices(::IntegratedLegendre, order::NTuple{D,Int}) where {D}
    all(o -> o >= 1, order) ||
        throw(ArgumentError("integrated Legendre order must be at least 1 in every axis"))
    return vec(collect(CartesianIndices(ntuple(d -> 0:order[d], D))))
end

# Trunk degree of a single 1D integrated-Legendre mode index. The two
# endpoint modes 0 and 1 are the linear vertex shape functions and count
# as 0; a bubble mode m ≥ 2 is an integrated Legendre polynomial of degree
# m and counts as m. Counting the vertex factors as 0 is what lets the
# trunk filter keep every edge mode up to degree p while still trimming
# the interior — a pure polynomial-total-degree filter would instead drop
# the top edge modes and break C⁰ conformity.
_trunk_degree(mode::Integer) = mode ≥ 2 ? Int(mode) : 0

# Trunk degree of a D-dim multi-index `α` is `Σ_d t(α_d)`. This is what the
# `:trunk` mode compares against the isotropic polynomial order `p`: edge
# modes (one bubble axis) survive up to degree p, while face/volume modes
# survive only while their bubble degrees sum to ≤ p.
function _trunk_degree(id::CartesianIndex{D}) where {D}
    degree = 0
    for axis in 1:D
        degree += _trunk_degree(id.I[axis])
    end
    return degree
end

function local_basis_indices(basis::IntegratedLegendre, order::NTuple{D,Int},
                             mode::Symbol) where {D}
    _check_basis_mode(mode, order)
    indices = local_basis_indices(basis, order)
    mode === :tensor && return indices
    return [id for id in indices if _trunk_degree(id) <= order[1]]
end

function local_basis_count(basis::IntegratedLegendre, order::NTuple{D,Int}, mode::Symbol) where {D}
    mode === :tensor && return local_basis_count(basis, order)
    return length(local_basis_indices(basis, order, mode))
end

# ── Hot-path tensor-product evaluation ────────────────────────────────────────

# Promote a `PointLike` input (NTuple or SVector) to an `SVector{D,T}` whose
# scalar type is the common floating-point promotion of the inputs.
function _reference_coordinate(xi::PointLike{D}) where {D}
    T = float(promote_type(map(typeof, Tuple(xi))...))
    return SVector{D,T}(xi)
end

# Caller-owned per-axis 1D scratch buffers used by `_fill_factor_tables!`
# and the tensor-product evaluation kernels. One `Vector{T}` per axis,
# length `order[d] + 1`. The assembly workspace pre-allocates one of
# these per level so the hot loop never reallocates.
#
# The per-axis size requirement `order[d] + 1` is the count of 1D modes
# active on a single cell for *every* basis family the package currently
# supports: integrated Legendre carries one endpoint per side plus
# `order − 1` bubbles, and an open-knot B-spline of degree `p` has
# exactly `p + 1` 1D functions non-zero on any single span. So the
# buffer allocation is family-agnostic; only the per-axis evaluator
# (`_fill_factor_tables!`) dispatches.
function _factor_buffers(order::NTuple{D,Int}, ::Type{T}) where {D,T}
    ntuple(d -> Vector{T}(undef, order[d] + 1), D)
end

# Recover the per-axis maximum mode index from a list of multi-indices.
# Each axis is walked once; the total work is `O(D · #indices)`, which
# matches a fused single-pass implementation. Used to size the
# `_factor_buffers` for cases where the caller has an index list but not
# an explicit `order`.
function _indices_order(indices::AbstractVector{CartesianIndex{D}}) where {D}
    ntuple(d -> maximum(id -> id.I[d], indices), D)
end

# Fill caller-owned per-axis 1D tables for `basis` at one reference
# point: `val1d[d][i + 1] = N̂ᵢ(ξ[d])` for the `order[d] + 1` 1D modes
# active on the cell along axis `d`. The 5-arg form fills only values;
# the 6-arg form additionally fills `der1d[d][i + 1] = N̂ᵢ′(ξ[d])`. The
# specific recurrence is basis-family-specific (Legendre three-term for
# integrated Legendre, de Boor for B-splines); both share the same
# `O(D · (order + 1))` per-point cost so the tensor-product evaluators
# downstream stay basis-agnostic.
#
# `cell::CartesianIndex{D}` is the parent cell's mesh-index along each
# axis. Integrated Legendre ignores it (its 1D modes are cell-local in
# the reference frame, identical on every cell); the B-spline family in
# the extension uses it to look up the knot-vector span the parent cell
# corresponds to. Callers always pass the same `parent.cell` the rest
# of the assembly loop uses, so the integrated-Legendre overhead is
# zero and the B-spline overload gets the lookup for free.
#
# The integrated-Legendre overloads below are the in-package default;
# additional basis families plug in via further overloads. The B-spline
# overloads live in `ext/UnfittedBasicBSplineExt.jl`.
function _fill_factor_tables!(::IntegratedLegendre, val1d, order::NTuple{D,Int}, ξ::SVector{D,T},
                              ::CartesianIndex{D}) where {D,T}
    @inbounds for d in 1:D
        x = ξ[d]
        p = order[d]
        vd = val1d[d]
        vd[1] = (one(T) - x) / 2
        if p >= 1
            vd[2] = (one(T) + x) / 2
        end
        if p >= 2
            ℓm2 = one(T)
            ℓm1 = x
            for m in 2:p
                ℓm = ((2m - 1) * x * ℓm1 - (m - 1) * ℓm2) / m
                vd[m + 1] = (ℓm - ℓm2) / sqrt(T(4m - 2))
                ℓm2, ℓm1 = ℓm1, ℓm
            end
        end
    end
    return nothing
end

function _fill_factor_tables!(::IntegratedLegendre, val1d, der1d, order::NTuple{D,Int},
                              ξ::SVector{D,T}, ::CartesianIndex{D}) where {D,T}
    @inbounds for d in 1:D
        x = ξ[d]
        p = order[d]
        vd = val1d[d]
        dd = der1d[d]
        vd[1] = (one(T) - x) / 2
        dd[1] = -one(T) / 2
        if p >= 1
            vd[2] = (one(T) + x) / 2
            dd[2] = one(T) / 2
        end
        if p >= 2
            ℓm2 = one(T)
            ℓm1 = x
            for m in 2:p
                ℓm = ((2m - 1) * x * ℓm1 - (m - 1) * ℓm2) / m
                vd[m + 1] = (ℓm - ℓm2) / sqrt(T(4m - 2))
                dd[m + 1] = sqrt(T(2m - 1) / 2) * ℓm1
                ℓm2, ℓm1 = ℓm1, ℓm
            end
        end
    end
    return nothing
end

# Tensor-product values from precomputed 1D tables. Fills the per-axis
# tables via `_fill_factor_tables!(basis, …)` (the only family-specific
# step), then walks each multi-index `α` in `indices` and assembles the
# product `N_α(ξ) = ∏_d val1d[d][α_d + 1]`. The tensor-product loop is
# basis-agnostic: it only assumes the per-axis tables are indexed by the
# 1D mode number on the cell, which is the canonical convention every
# family in this package follows. The caller owns `val1d`.
function _tensor_values!(basis::BasisFamily, values::AbstractVector,
                         indices::AbstractVector{CartesianIndex{D}}, order::NTuple{D,Int},
                         ξ::SVector{D,T}, val1d, cell::CartesianIndex{D}) where {D,T}
    length(values) == length(indices) ||
        throw(DimensionMismatch("basis value buffer has wrong length"))
    _fill_factor_tables!(basis, val1d, order, ξ, cell)
    @inbounds for (a, id) in pairs(indices)
        values[a] = prod(ntuple(d -> val1d[d][id.I[d] + 1], D))
    end
    return values
end

# Tensor-product values and physically-scaled gradients from precomputed 1D
# tables. For each multi-index `α`:
#
#     N_α(ξ)     = ∏_d val1d[d][α_d + 1]
#     ∂N_α/∂x_d  = der1d[d][α_d + 1] · ∏_{j ≠ d} val1d[j][α_j + 1] · scale[d]
#
# `scale[d]` is the chain-rule factor from reference to physical
# coordinates. For the bare reference gradient pass `scale = (1, …, 1)`;
# for the physical gradient on an axis-aligned cell with edge lengths
# `h_d` pass `scale[d] = 2 / h_d` (the reference cell has edge length 2).
#
# The "product except one" trick is `O(D)` per gradient component for
# `O(D²)` per multi-index — for typical `D ≤ 4` and modest mode counts
# this is comfortably allocation-free with `_product_except` operating
# on `NTuple` scratch.
function _tensor_values_grads!(basis::BasisFamily, values::AbstractVector,
                               gradients::AbstractVector,
                               indices::AbstractVector{CartesianIndex{D}}, order::NTuple{D,Int},
                               ξ::SVector{D,T}, scale::SVector{D,T}, val1d, der1d,
                               cell::CartesianIndex{D}) where {D,T}
    length(values) == length(indices) ||
        throw(DimensionMismatch("basis value buffer has wrong length"))
    length(gradients) == length(indices) ||
        throw(DimensionMismatch("basis gradient buffer has wrong length"))
    _fill_factor_tables!(basis, val1d, der1d, order, ξ, cell)
    @inbounds for (a, id) in pairs(indices)
        vfac = ntuple(d -> val1d[d][id.I[d] + 1], D)
        values[a] = prod(vfac)
        gradients[a] = SVector{D,T}(ntuple(k -> der1d[k][id.I[k] + 1] *
                                                _product_except(vfac, k) *
                                                scale[k], D))
    end
    return values, gradients
end

# Product `∏_{d ≠ k} values[d]`. Used by the gradient kernel to compute
# the "value part" of the gradient component along axis `k` without
# recomputing the full product.
function _product_except(values::NTuple{D}, k::Integer) where {D}
    result = one(values[1])
    for d in 1:D
        d == k || (result *= values[d])
    end
    return result
end

# ── Public value and gradient evaluation ──────────────────────────────────────

# Sentinel cell index used by the in-package public-API wrappers
# (`basis_values`, `physical_basis_gradients`, …) whose call signatures
# do not carry a parent-cell argument. Integrated Legendre ignores the
# cell positional in `_fill_factor_tables!`, so any in-bounds value
# would do; we pick the lowest-corner cell deterministically. Basis
# families whose evaluator depends on the cell (e.g. B-splines) expose
# their own public-API wrappers that take a cell explicitly.
_cell_sentinel(::Val{D}) where {D} = CartesianIndex(ntuple(_ -> 1, D))

"""
    basis_values!(basis::IntegratedLegendre, values, order[, mode], xi)
    basis_values!(basis::IntegratedLegendre, values, indices,       xi)

Write the local basis values at reference point `xi` into the caller-owned
buffer `values`. `length(values)` must equal the number of basis functions
(either `local_basis_count(basis, order, mode)` or `length(indices)`).
The 4-arg form lets callers re-use a precomputed index list (the cheaper
path for the hot loop); the 5-arg form looks the indices up internally.

`mode` defaults to `:tensor`. Allocates fresh per-axis scratch buffers
internally — for hot-loop use go through `_tensor_values!` with reused
scratch instead.
"""
function basis_values!(basis::IntegratedLegendre, values::AbstractVector, order::NTuple{D,Int},
                       mode::Symbol, xi::PointLike{D}) where {D}
    indices = local_basis_indices(basis, order, mode)
    return basis_values!(basis, values, indices, xi)
end

function basis_values!(basis::IntegratedLegendre, values::AbstractVector,
                       indices::AbstractVector{CartesianIndex{D}}, xi::PointLike{D}) where {D}
    ξ = _reference_coordinate(xi)
    order = _indices_order(indices)
    return _tensor_values!(basis, values, indices, order, ξ, _factor_buffers(order, eltype(ξ)),
                           _cell_sentinel(Val(D)))
end

function basis_values!(basis::IntegratedLegendre, values::AbstractVector, order::NTuple{D,Int},
                       xi::PointLike{D}) where {D}
    basis_values!(basis, values, order, :tensor, xi)
end

"""
    basis_values(basis::IntegratedLegendre, order[, mode], xi) -> Vector

Allocating version of [`basis_values!`](@ref): returns a fresh vector
holding the local basis values at reference point `xi`. Convenient for
one-shot evaluation; the in-place form is preferred for hot loops.
"""
function basis_values(basis::IntegratedLegendre, order::NTuple{D,Int}, mode::Symbol,
                      xi::PointLike{D}) where {D}
    T = promote_type(map(typeof, Tuple(xi))...)
    values = Vector{T}(undef, local_basis_count(basis, order, mode))
    return basis_values!(basis, values, order, mode, xi)
end

function basis_values(basis::IntegratedLegendre, order::NTuple{D,Int}, xi::PointLike{D}) where {D}
    basis_values(basis, order, :tensor, xi)
end

# Cell-aware variant used by `postprocessing.jl` (where the parent cell
# index is already known) and by any caller that wants to support
# basis families whose evaluator depends on the cell. Integrated
# Legendre ignores `cell` and routes to the cell-agnostic method; the
# B-spline family overloads this signature directly in the extension.
function basis_values(basis::IntegratedLegendre, order::NTuple{D,Int}, mode::Symbol,
                      xi::PointLike{D}, ::CartesianIndex{D}) where {D}
    return basis_values(basis, order, mode, xi)
end

"""
    reference_basis_gradients!(basis::IntegratedLegendre, gradients, order, mode, xi)
    reference_basis_gradients(basis::IntegratedLegendre, order[, mode], xi) -> Vector{SVector{D}}

Local basis gradients with respect to the *reference* coordinate `ξ`.
Each entry is an `SVector{D}` whose `d`-th component is `∂N_α/∂ξ_d` at
`xi`. For the gradient with respect to the *physical* coordinate on an
axis-aligned cell, use [`physical_basis_gradients`](@ref) instead.

The in-place form writes into a caller-owned buffer; the allocating form
returns a fresh vector. Both allocate per-axis scratch buffers internally.
"""
function reference_basis_gradients!(basis::IntegratedLegendre, gradients::AbstractVector,
                                    order::NTuple{D,Int}, mode::Symbol, xi::PointLike{D}) where {D}
    indices = local_basis_indices(basis, order, mode)
    ξ = _reference_coordinate(xi)
    T = eltype(ξ)
    values = Vector{T}(undef, length(indices))
    scale = SVector{D,T}(ntuple(_ -> one(T), D))
    _tensor_values_grads!(basis, values, gradients, indices, order, ξ, scale,
                          _factor_buffers(order, T), _factor_buffers(order, T),
                          _cell_sentinel(Val(D)))
    return gradients
end

function reference_basis_gradients(basis::IntegratedLegendre, order::NTuple{D,Int}, mode::Symbol,
                                   xi::PointLike{D}) where {D}
    T = promote_type(map(typeof, Tuple(xi))...)
    gradients = Vector{SVector{D,T}}(undef, local_basis_count(basis, order, mode))
    return reference_basis_gradients!(basis, gradients, order, mode, xi)
end

function reference_basis_gradients(basis::IntegratedLegendre, order::NTuple{D,Int},
                                   xi::PointLike{D}) where {D}
    reference_basis_gradients(basis, order, :tensor, xi)
end

"""
    physical_basis_gradients!(basis::IntegratedLegendre, gradients, order, mode, cell, xi)
    physical_basis_gradients(basis::IntegratedLegendre, order[, mode], cell, xi) -> Vector{SVector{D}}

Local basis gradients with respect to the *physical* coordinate on the
axis-aligned `cell`. The chain rule for an axis-aligned cell with edge
lengths `h_d = edge_lengths(cell)[d]` gives

    ∂N_α/∂x_d = (2 / h_d) · ∂N_α/∂ξ_d,

so the physical gradient is the reference gradient scaled per axis by
`scale[d] = 2 / h_d`. `xi` is the reference coordinate ξ ∈ [−1, 1]ᴰ; map
a physical point through [`physical_to_reference`](@ref) first if needed.

The in-place form writes into a caller-owned buffer; the allocating form
returns a fresh vector. Both allocate per-axis scratch internally.
"""
function physical_basis_gradients!(basis::IntegratedLegendre, gradients::AbstractVector,
                                   order::NTuple{D,Int}, mode::Symbol, cell::AxisBox{D,T},
                                   xi::PointLike{D}) where {D,T}
    indices = local_basis_indices(basis, order, mode)
    ξ = _reference_coordinate(xi)
    R = eltype(ξ)
    values = Vector{R}(undef, length(indices))
    scale = SVector{D,R}(2 ./ edge_lengths(cell))
    _tensor_values_grads!(basis, values, gradients, indices, order, ξ, scale,
                          _factor_buffers(order, R), _factor_buffers(order, R),
                          _cell_sentinel(Val(D)))
    return gradients
end

function physical_basis_gradients(basis::IntegratedLegendre, order::NTuple{D,Int}, mode::Symbol,
                                  cell::AxisBox{D,T}, xi::PointLike{D}) where {D,T}
    gradients = Vector{SVector{D,T}}(undef, local_basis_count(basis, order, mode))
    return physical_basis_gradients!(basis, gradients, order, mode, cell, xi)
end

function physical_basis_gradients(basis::IntegratedLegendre, order::NTuple{D,Int},
                                  cell::AxisBox{D,T}, xi::PointLike{D}) where {D,T}
    physical_basis_gradients(basis, order, :tensor, cell, xi)
end

# Cell-aware variant used by `postprocessing.jl` (where the parent cell
# index is already known) and by any caller that wants to support
# basis families whose evaluator depends on the cell. Integrated
# Legendre ignores `cell_index` and routes to the cell-agnostic
# method; the B-spline family overloads this signature directly in
# the extension.
function physical_basis_gradients(basis::IntegratedLegendre, order::NTuple{D,Int}, mode::Symbol,
                                  cell_box::AxisBox{D,T}, xi::PointLike{D},
                                  ::CartesianIndex{D}) where {D,T}
    return physical_basis_gradients(basis, order, mode, cell_box, xi)
end

# ── Boundary and facet basis identification ───────────────────────────────────

# True iff the 1D integrated Legendre mode index `mode` carries support on
# the requested cell side: mode 0 lives on the lower endpoint (`ξ = −1`),
# mode 1 on the upper endpoint (`ξ = +1`), and modes m ≥ 2 (bubbles)
# vanish at both endpoints — they never sit on either side.
function boundary_mode(mode::Integer, side::Symbol)
    side === :lower && return mode == 0
    side === :upper && return mode == 1
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# True iff the tensor-product basis function with multi-index `id` carries
# support on the codim-1 face specified by `(axis, side)`. For integrated
# Legendre this reduces to checking the single 1D mode along the requested
# axis.
function is_boundary_basis(::IntegratedLegendre, id::CartesianIndex{D}, axis::Integer,
                           side::Symbol) where {D}
    1 <= axis <= D || throw(ArgumentError("axis out of bounds"))
    return boundary_mode(id.I[axis], side)
end

# Codim-K facet membership for a tensor-product multi-index: the basis function
# sits on the facet specified by `sides` (a list of `(axis, side)` pairs) iff
# its 1D mode along *every* listed axis touches the requested side. Codim 1
# reduces to a single `is_boundary_basis` check; codim K requires every listed
# axis-side match. This `::BasisFamily` default delegates the per-axis test to
# the family's `is_boundary_basis`, so it needs no family-specific override.
function is_facet_basis(basis::BasisFamily, id::CartesianIndex{D},
                        sides::AbstractVector{<:Tuple{Integer,Symbol}}) where {D}
    for (axis, side) in sides
        is_boundary_basis(basis, id, axis, side) || return false
    end
    return true
end

"""
    boundary_basis_indices(basis, order; axis, side, mode=:tensor)
        -> Vector{CartesianIndex{D}}

Multi-indices of the local basis functions whose support touches the codim-1
face at the requested `(axis, side)`. Used by the Dirichlet projection in
`src/dofs.jl` to identify which trial / test modes contribute on each physical
facet. This `::BasisFamily` default filters `local_basis_indices(basis, order,
mode)` through the family's `is_boundary_basis`, so it needs no override.
"""
function boundary_basis_indices(basis::BasisFamily, order::NTuple{D,Int}; axis::Integer,
                                side::Symbol, mode::Symbol=:tensor) where {D}
    return [id
            for id in local_basis_indices(basis, order, mode)
            if is_boundary_basis(basis, id, axis, side)]
end

# ── Public quadrature rule ────────────────────────────────────────────────────

"""
    gauss_rule(basis::IntegratedLegendre, order, [T=Float64]) -> TensorQuadrature

Tensor Gauss–Legendre rule on the reference cube `[−1, 1]ᴰ` sized by
[`recommended_quadrature_order`](@ref) for the given basis and polynomial
order. Useful for one-shot reference-cell quadrature in tests and
examples; the assembly path uses the cached `_tensor_gauss_rule` directly.
"""
function gauss_rule(basis::IntegratedLegendre, order::NTuple{D,Int},
                    ::Type{T}=Float64) where {D,T<:Real}
    _tensor_gauss_rule(recommended_quadrature_order(basis, order), T)
end
