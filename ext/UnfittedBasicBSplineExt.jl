"""
    UnfittedBasicBSplineExt

Package extension that adds an open-knot, tensor-product B-spline basis family to Unfitted on top of
[`BasicBSpline.jl`](https://github.com/hyrodium/BasicBSpline.jl). Loading `BasicBSpline` next to `Unfitted` installs the
family and the required basis-interface methods; without `BasicBSpline`, the package functions exactly as before with
the default [`IntegratedLegendre`](@ref) family.

# Mathematical setting

Per axis `d` the family carries a `BasicBSpline.BSplineSpace{p_d, T, K_d}` on an open knot vector aligned with the
level's mesh cell coordinates:

  - End knots have multiplicity `p_d + 1` (clamped open knots), so the first 1D function evaluates to 1 at the lower
    edge and the last function evaluates to 1 at the upper edge — every other 1D function vanishes at both endpoints.
  - Interior knots all have multiplicity 1 (max-regularity), giving `C^{p_d − 1}` smoothness across every cell boundary
    — uniform knots, no insertions.

Smoothness across the artificial overlay boundary (the mesh-edge faces and any mask-induced internal faces) is a
property of the *selection*, not of the knot multiplicity, and the `continuity` keyword on [`bspline`](@ref) chooses
between two mechanisms.

At `continuity = :maximal`, the default, the level keeps exactly the functions whose full `(p_d + 1)`-cell support lies
inside its admissible region — its active cells, the cells that hold no material (every cell the fictitious fold
switched off, a user-masked cell lying entirely outside Ω, and a cell past an artificial box face lying entirely outside
Ω), and everything beyond a box face that coincides with the boundary of the space's own domain — that face is not
artificial, and what lies past it is not part of the discretised domain at all. Such a function is already globally
`C^{p_d − 1}` and identically zero outside its support, so its extension by zero is `C^{p_d − 1}` for free, under any
mask geometry and with no constraint equation anywhere. This is Kraft's hierarchical-spline selection rule stated for
superposed independent grids; the full statement, its maximality proof and the citations are in [`bspline`](@ref).

At an integer `continuity = m < p_d − 1` the level is instead clamped at its artificial faces and the trace and its
derivatives `k = 0 … m` are constrained to vanish by homogeneous linear constraints, eliminating `m + 1` boundary
functions per face and leaving a strictly larger space — `∏_d (n_d + p_d − 2(m + 1))` functions against
`∏_d (n_d − p_d)` — at the price of a globally `C^m` superposition. Arbitrary mask geometries are supported either way,
because neither mechanism needs the mask to be separable.

The `D`-dimensional basis is the tensor product of these per-axis 1D spaces. Adjacent cells of the level share `p_d` of
`p_d + 1` 1D functions per axis — the shared dofs fall out of the `TensorDofKey` cache in `src/dofs.jl` once each
cell-local mode index is mapped to its global 1D function index via the `_AXIS_BSPLINE` tag.

# What this extension installs

  - `BSplineFamily{D,T,…}` (a `BasisFamily` subtype) plus a deferred `_BSplineSpec` returned by the public
    [`bspline`](@ref) factory. The deferred form lives only between `bspline(...)` and the [`instantiate_basis`](@ref)
    hook, which materialises the per-axis `BSplineSpace`s once the mesh axes are known. The knot vectors depend on the
    mesh alone — never on the mask, whose faces are handled by the selection and the constraints described below.
  - The family-specific basis-interface overloads — `basis_name` and `is_boundary_basis`. Everything else in the
    interface is inherited from the `::BasisFamily` defaults in `src/basis.jl`: the tensor `local_basis_indices` /
    `local_basis_count` (an open-knot span carries exactly the full `∏_d {0, …, p_d}` set),
    `recommended_quadrature_order`, `is_facet_basis`, `boundary_basis_indices`, and the boundary trace; the level-keyed
    point evaluators `basis_values` / `physical_basis_gradients` reach the family through the kernels below alone. The
    family declares no mode support of its own either: the `::BasisFamily` default is `:tensor` only, which is exactly
    what this family serves, so `_check_basis_mode` rejects `:trunk` for it.
  - Hot-path `_fill_factor_tables!` overloads (values, values + derivatives) using `BasicBSpline.bsplinebasisall` per
    axis after mapping the cell-local reference coordinate `ξ ∈ [−1, 1]` to the knot-vector parameter.
  - `_tensor_dof_key` and `_key_on_level_side` overloads that route through the family's own `_AXIS_BSPLINE` key tag.
  - The `_overlay_constraints!` overload, which carries both artificial-boundary mechanisms and writes them into the dof
    layer's [`EliminationLedger`](@ref). At `continuity = :maximal` it is the **support selection**: a strong
    elimination (`_eliminate!`) of every function whose `(p_d + 1)`-cell support is not contained in the level's
    admissible region, and no linear constraint at all, so the layout has no pivot, `has_linear_constraints` is `false`,
    and assembly has no pivot to look for or condense. At an integer `continuity = m < p_d − 1` it records the
    homogeneous trace-vanishing conditions (orders `k = 0 … m`) on every overlay / mask face as linear-constraint rows
    (`_constrain!`), which the `_resolve_rows!` overload below turns into pivots. Masked levels of any geometry are
    eliminable either way.
  - The burial hooks of the dof layer's one pruning driver, `_coverage_constraints!`: `_burial_cells`, `_cover_test`
    with its exact test `_reproduced_by`, `_dedup!`, and the rows of the `_reproduces` trait that have a B-spline side.
    Covered-mode pruning for this family is the dedup and nothing else: a buried function a covering level's span
    already contains exactly is eliminated, because the two are linearly dependent and the superposition is otherwise
    singular. Under selection the burial test is an *exact* containment rather than a conservative one; see
    [`bspline`](@ref) under "Nested levels". On a clamped level the dedup is a one-term row rather than a direct
    elimination, so that the row resolution decides whether a trace row already eliminated the function.
  - The `_resolve_rows!` overload, the only resolver of linear-constraint rows in the package: the dof layer stores a
    level's rows but resolves none itself, and its default raises. This family's method is the cascade Gauss elimination
    `_cascade!` — pivoting on the largest coefficient, dropping terms below a relative tolerance, back-substituting each
    new pivot into the earlier ones — which leaves every pivot a depth-one expansion over raws that stay free.
  - The material exemption, which appears once in each mechanism: an inactive cell that holds no material (every cell
    the fictitious fold switched off, and a user-masked cell lying entirely outside Ω) is admissible for the selection
    rule, and an internal face whose inactive cell holds none carries no trace to vanish on and emits no constraint,
    while an inactive cell that holds material (a user mask reaching into Ω) blocks in both. That one distinction is all
    the family needs to carry an immersed [`PhysicalDomain`](@ref), so the finite-cell workflow is open to it.

The hot loops in `src/assembly.jl` and `src/projection.jl` are unchanged — they dispatch on `level.basis` (via the
`bases` vector of the workspace's basis bank) and the B-spline overloads fire naturally.
"""
module UnfittedBasicBSplineExt

using Unfitted
# Only the names this module uses *unqualified* are imported; every method it installs on an Unfitted function is
# written `Unfitted.f(...)` at its definition, which needs no import. The list is therefore an honest measure of how far
# the extension reaches into the package.
using Unfitted: BasisFamily, IntegratedLegendre, Level, CartesianMesh, AxisDofKey, TensorDofKey, EliminationLedger,
                _LevelContext, _eliminate!, _constrain!, _pivot!, is_active, AxisBox, _check_basis_mode,
                _level_side_is_physical, _tensor_dof_key, _Fold, _is_fictitious, _is_void
using StaticArrays: SVector
using BasicBSpline: BSplineSpace, BSplineDerivativeSpace, KnotVector, bsplinebasisall, degree, dim

# ── Public factory and deferred spec ──────────────────────────────────────────

"""
    bspline(; continuity = :maximal) -> _BSplineSpec

Mark a [`space`](@ref) / [`overlay`](@ref) call as using the open-knot tensor B-spline basis family. The per-axis
polynomial degree is taken from the `order` keyword on `space` / `overlay` (matching the integrated Legendre
convention), so a typical use looks like

```julia
using Unfitted, BasicBSpline
V = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, basis=bspline())
```

# Keyword arguments

  - `continuity` — the smoothness the level's contribution carries **across its own artificial boundary**, and therefore
    the smoothness of the whole superposition there. `:maximal` (the default) is `C^{p_d − 1}` per axis, the most a
    degree-`p_d` spline has to give; an `Integer` `m` asks for `C^m` and buys unknowns with the difference. `m` may run
    from `0` (`C⁰`, which is all the integrated-Legendre family offers) up to `p_d − 1` on the lowest-degree axis, and
    an integer request equal to `p_d − 1` on every axis is `:maximal` by another name and takes the same code path.

# What a level contributes, and the one theorem behind it

Write `Â` for the level's **admissible region**: its active cells, plus every cell that holds no material — an inactive
cell lying entirely outside Ω, which every cell the fictitious fold switched off is and a user-masked cell may be, and a
cell of the grid continued past an artificial box face lying entirely outside Ω (nothing is integrated there, so a
truncation there is invisible on Ω) — plus everything beyond a box face that coincides with the boundary of the space's
own domain (that face is not artificial, so nothing has to vanish on it). At `:maximal` the level generates exactly

    W = span { B_α : supp B_α ⊆ Â }

where `supp B_α = ∏_d [α_d − p_d, α_d]` is the full `(p_d + 1)`-cell support of the tensor B-spline, counted **without
clipping** to the level's own mesh — the cells that fall outside the mesh are precisely what tells the rule that the
function reaches past the level's box.

That selection is not one option among several. A degree-`p` B-spline on simple interior knots is globally `C^{p−1}` and
identically zero outside its support, so every such `B_α` extends by zero to a globally `C^{p−1}` function, with no
constraint equation anywhere and under any mask geometry. And it is *maximal*: if `v` is a spline of the level that
vanishes outside `Â` and is `C^{p−1}` across `∂Â`, then `v ≡ 0` on every cell outside `Â`, so by the local linear
independence of the B-splines on a cell every coefficient of a function whose support meets that cell is zero — that is,
`v ∈ W`. No selection rule and no constraint system can do better; the boundary layer below is a property of the space,
not of the basis.

The rule is Kraft's selection, stated for superposed independent grids instead of a nested hierarchy:

> R. Kraft, *Adaptive and linearly independent multilevel B-splines*, in: A. Le Méhauté, C. Rabut, L. L. Schumaker
> (Eds.), Surface Fitting and Multiresolution Methods, Vanderbilt University Press (1997) 209–218.
>
> C. Giannelli, B. Jüttler, H. Speleers, *THB-splines: The truncated basis for hierarchical splines*, Comput. Aided
> Geom. Design **29** (2012) 485–498, [doi:10.1016/j.cagd.2012.03.025](https://doi.org/10.1016/j.cagd.2012.03.025),
> Definition 1.

Its per-patch, arbitrary-degree, arbitrary-smoothness form — the one this family implements — is the Patchwork B-spline
selection `K^ℓ = {j : β_j^ℓ|_{π^ℓ} ≠ 0 and β_j^ℓ|_{Γ^ℓ} = 0}`, whose constraining boundary `Γ^ℓ` is this package's
`Γ_o^(k)`:

> D. Engleitner, B. Jüttler, *Patchwork B-spline refinement*, Computer-Aided Design **90** (2017) 168–179,
> [doi:10.1016/j.cad.2017.05.021](https://doi.org/10.1016/j.cad.2017.05.021), Lemma 2 and Theorem 1.

# The dof ledger, and the boundary layer

On a level of `n_d` cells per axis whose faces are all artificial, `:maximal` leaves `∏_d max(0, n_d − p_d)` functions,
and an integer `m` leaves `∏_d max(0, n_d + p_d − 2(m + 1))`. So a level contributes **nothing at all** until its active
region is more than `p_d` cells wide in every axis, and an activation front must advance `p_d` cells before the level
carries a new unknown. That is the minimal-support theorem — a nonzero element of the degree-`p`, `C^{p−1}` spline space
supported on `k` consecutive knot intervals exists only for `k ≥ p + 1` (C. de Boor, *A Practical Guide to Splines*,
revised edition, Springer (2001), Ch. IX, [doi:10.1007/978-1-4612-6333-3](https://doi.org/10.1007/978-1-4612-6333-3)) —
and it is why a refinement region has to contain the **support extension** of the feature it resolves rather than only
the feature. [`support_extension`](@ref) computes that dilation, and passing an under-extended region is the one way to
make this family look worse than it is: measured on a tanh layer of width 1/40 with a degree-3 base of 8 cells and a
band overlay at `h = 1/128`, the relative L² error fell 0.0735 → 3.85e-4 as the band was widened from 1 to 4 base cells
per side, at constant overlay resolution.

# Nested levels

A B-spline overlay of the same degree whose cell boundaries include the base's reproduces, exactly, every base function
whose support lies inside the overlay's admissible region — so the two levels would carry the same function twice and
the superposed operator would be exactly singular. Leaf semantics eliminate the duplicate, unconditionally and with no
keyword; that costs nothing here, because the discrete space is unchanged, unlike integrated Legendre's shedding of
buried high-order modes, which trades accuracy for dofs.

Under `:maximal` the burial test is the same containment the selection rule uses, and it is **exact** rather than
conservative. Subdivision writes a buried `B_α^ℓ` as a positive combination of the fine functions whose support lies
inside `supp B_α^ℓ`; every one of those is itself admissible on `k` as soon as `supp B_α^ℓ ⊆ Â^k`, so containment is
sufficient. It is also necessary: every cell of the support carries at least one fine function with a strictly positive
subdivision coefficient, so a live cell of the support outside `Â^k` means a dropped term and no reproduction. There is
therefore no configuration in which the cover reproduces a *combination* of buried functions without reproducing each of
them — the failure that a clamped overlay admits, and the reason this family used to leave one fold configuration
exactly singular.

An integrated-Legendre cover in `:tensor` mode at order ≥ the degree reproduces the same functions and is deduped the
same way, because its span is every `C⁰` tensor polynomial of that degree on its cells. `:trunk` drops the mixed
high-order terms and does not. The converse also holds: a *degree-1* B-spline overlay is the `C⁰` multilinear hat space
of its own mesh, so it deduplicates the buried vertex functions of an integrated-Legendre level below; at degree ≥ 2 it
is `C^{p−1}` at a simple interior knot, too smooth to carry the hat's kink, and nothing is deduplicated.

At an integer `continuity` below `p − 1` the level is *clamped* at its artificial faces instead of selected, the
reproduction question stops being a containment, and the burial test falls back to the conservative predicate
`_reproduced_on_domain` uses. One fold configuration is then still left singular — when the overlay's box face falls on
a knot of the level below inside a band of cut cells, the clamped overlay reproduces a combination of two buried
functions while reproducing neither on its own, and the operator keeps one exact null mode per combination (measured at
8 on a 6×6 degree-2 base over Ω = {x₁ ≤ 0.9} in [−1, 2]² with a quarter-cell overlay ending at x = 1). Use `:maximal`,
move the overlay's face off the cut-cell band, or break the nesting.

The only stack that deliberately carries both copies is the **unreduced twin**
[`prepare`](@ref)`(problem; prune = false)` builds, and on a nested stack that twin is exactly singular by construction
— which is what it is for. It exists to be measured against, not to be solved with.

# Why a deferred spec

The actual per-axis `BSplineSpace`s depend on the level's mesh-cell coordinates, which do not exist yet when the
`basis=` value is chosen. `bspline(...)` therefore returns a deferred [`_BSplineSpec`](@ref); the concrete
[`BSplineFamily`](@ref) is built by the [`instantiate_basis`](@ref) hook in this extension once the mesh is known. A
mask changes nothing about the knot vectors — a mask face selects functions rather than inserting knots — so a level's
family is rebuilt on a move but is the same family before and after a mask update. The spec is itself a
[`BasisFamily`](@ref) subtype so the existing `space` / `overlay` code path accepts it without special casing.
"""
function Unfitted.bspline(; continuity::Union{Symbol,Integer}=:maximal)
    continuity === :maximal && return _BSplineSpec(_MAXIMAL_CONTINUITY)
    continuity isa Symbol &&
        throw(ArgumentError("bspline: continuity must be :maximal or a nonnegative integer; " * "got :$continuity"))
    continuity >= 0 || throw(ArgumentError("bspline: continuity must be nonnegative; got $continuity"))
    return _BSplineSpec(Int(continuity))
end

# Sentinel for `continuity = :maximal`, resolved per axis to `p_d − 1` once the order is known. It is stored rather than
# resolved at the factory because the degree belongs to the `order` keyword on `space` / `overlay`, which the factory
# never sees. Negative so it can never collide with a requested smoothness.
const _MAXIMAL_CONTINUITY = -1

"""
    _BSplineSpec(continuity) <: BasisFamily

Deferred B-spline family specification returned by [`bspline`](@ref). Subtypes `BasisFamily` so [`space`](@ref) accepts
it as a `basis=` value, but carries no per-axis knot vectors yet — the [`instantiate_basis`](@ref) hook in this
extension materialises a real [`BSplineFamily`](@ref) from the mesh axes. Once the level is built, the spec is gone;
every hot-path method dispatches on the concrete `BSplineFamily` type.

`continuity` is either a requested smoothness `m ≥ 0` or the [`_MAXIMAL_CONTINUITY`](@ref) sentinel. It is the request
and only the request: `instantiate_basis` validates it against the level's mesh and per-axis degree and chooses the
mechanism, and every concrete family it builds carries the same request on, so a rebuild at another order or cell count
is judged exactly as a fresh build would be.
"""
struct _BSplineSpec <: BasisFamily
    continuity::Int
end

# ── BSplineFamily struct and accessors ────────────────────────────────────────

"""
    BSplineFamily{D,T,S,DS}

Open-knot tensor-product B-spline basis family.

Fields:

  - `spaces::S` — `D`-tuple of `BasicBSpline.BSplineSpace` objects, one per axis. Built from the level's mesh axes as
    clamped open uniform knot vectors (interior knot multiplicity 1, `C^{p − 1}`).
  - `derivative_spaces::DS` — `D`-tuple of `BSplineDerivativeSpace{1, _}` objects, one per axis, precomputed for the
    derivative hot path so every quadrature point uses a stable type.
  - `axis_coords::NTuple{D,Vector{T}}` — per-axis mesh cell-corner coordinates (a copy of `mesh.axes[d]`). Stored on the
    family so the hot kernel can do the `ξ ∈ [−1, 1]` → `t` affine map without looking up the level.
  - `cell_to_span::NTuple{D,Vector{Int}}` — per-axis map from a cell index `c ∈ 1:cells[d]` to the BasicBSpline span
    index. With the uniform max-regularity knots used here it is the identity (`cell_to_span[d][c] = c`); it is kept as
    an explicit field because it also doubles as the per-cell first global 1D function index —
    `bsplinebasisall(P, span, t)` returns the functions with global indices `span, span + 1, …, span + p`.
  - `continuity::Int` — the smoothness the level was *asked* to carry across its own artificial boundary, as
    requested: an integer `0 ≤ m ≤ p_d − 1` or the [`_MAXIMAL_CONTINUITY`](@ref) sentinel for `C^{p_d − 1}` per axis.
    The mechanism is derived from it and the degrees by `_selects_by_support`, and the two mechanisms are different
    things rather than different numbers: the sentinel, or an integer equal to `p_d − 1` on every axis (the same space
    by another name), selects the functions whose full support lies inside the admissible region and eliminates the
    rest strongly, while an integer below `p_d − 1` on some axis clamps the level at its artificial faces and imposes
    trace-vanishing conditions of orders `0 … m`. See [`bspline`](@ref). The request is stored rather than the
    mechanism because a rebuild at another degree must re-derive the mechanism from what was asked: `C²` at degree 3
    selects, at degree 4 it clamps.

Constructed automatically from a [`_BSplineSpec`](@ref) by the [`instantiate_basis`](@ref) hook in this extension —
users do not build it directly.
"""
struct BSplineFamily{D,T,S<:Tuple,DS<:Tuple} <: BasisFamily
    spaces::S
    derivative_spaces::DS
    axis_coords::NTuple{D,Vector{T}}
    cell_to_span::NTuple{D,Vector{Int}}
    continuity::Int
end

# Whether the level selects by support (`:maximal`) rather than clamping and constraining: the selection sentinel, or an
# integer request equal to `p_d − 1` on every axis, which names the same space and is reached by the cheaper mechanism
# (strong eliminations instead of a constraint cascade). Derived from the stored request and the family's own degrees
# rather than stored, so it cannot disagree with them after a rebuild at another degree. Written as a predicate because
# the two mechanisms diverge in several places — the overlay constraints, the burial test, the dedup and the support
# radius — and a comparison at each would say what is stored rather than what it means.
function _selects_by_support(family::BSplineFamily)
    m = family.continuity
    return m == _MAXIMAL_CONTINUITY || all(P -> m == degree(P) - 1, family.spaces)
end

# ── Basis instantiation: materialise the deferred spec onto a mesh ────────────

# `instantiate_basis` hook for the deferred `_BSplineSpec`, and the one place a B-spline request is judged. Invoked when
# `space` / `overlay` build a level from `bspline(...)`, once the level's mesh and order are known, and — through the
# `BSplineFamily` method below — on every later rebuild of the level as well. Validates the request and materialises the
# concrete `BSplineFamily` from the mesh axes; the spec itself only ever exists between `bspline(...)` and this call, so
# every downstream method dispatches on the concrete family type.
#
# The mask is not an argument, because the family does not depend on it: arbitrary `LevelMask` geometries are handled by
# either mechanism in the dof layer — the support-selection rule is a per-function containment test and the
# trace-vanishing constraints are assembled face by face, so neither needs the mask to be separable.
#
# A request for an integer `continuity = m` that already equals `p_d − 1` on every axis takes the selection mechanism
# (`_selects_by_support`). The two name the same space — the proof is in `bspline`'s docstring — but selection reaches
# it with strong eliminations rather than a constraint cascade, so routing the integer spelling through the cheaper
# mechanism costs the caller nothing and saves the dof layer a pass. The family keeps the request as given rather than
# rewriting it to the sentinel: the equivalence holds at this degree only, and a level rebuilt at another degree must
# be judged against what was asked.
function Unfitted.instantiate_basis(spec::_BSplineSpec, mesh::CartesianMesh{D,T}, order::NTuple{D,Int},
                                    mode::Symbol) where {D,T<:Real}
    _check_basis_mode(spec, mode, order)
    all(p -> p >= 1, order) || throw(ArgumentError("bspline order (per-axis polynomial degree) must be ≥ 1"))
    m = spec.continuity
    m == _MAXIMAL_CONTINUITY && return _materialize(spec, mesh, order)
    max_m = minimum(p -> p - 1, order)
    m <= max_m || throw(ArgumentError("bspline continuity $m exceeds p_d − 1 = $max_m on the " *
                                      "lowest-degree axis; raise the order, lower the continuity, " *
                                      "or ask for :maximal"))
    all(p -> m == p - 1, order) && return _materialize(spec, mesh, order)    # selects by support
    # The clamped mechanism needs enough interior 1D dofs for the constraint system to have something left to pivot
    # onto: eliminating `m + 1` boundary functions per side must leave at least one interior dof, i.e.
    # `dim ≥ 2(m + 1) + 1`, equivalently `cells + p ≥ 2m + 3`. Selection carries no such requirement and is not checked
    # above — a level too thin to hold a whole support simply contributes nothing, which is a statement about the spline
    # space (minimal support needs `p + 1` cells) rather than a misuse, and an adaptive loop that activates a small
    # cluster must be allowed to reach it.
    for d in 1:D
        cells_plus_p = mesh.cells[d] + order[d]
        cells_plus_p >= 2m + 3 || throw(ArgumentError("bspline level too thin to support " *
                                                      "continuity $m on axis $d: cells + p = " *
                                                      "$cells_plus_p, need ≥ $(2m + 3). Increase " *
                                                      "cells, lower the continuity, or ask for " *
                                                      ":maximal, which has no thickness " *
                                                      "requirement."))
    end
    return _materialize(spec, mesh, order)
end

# Re-instantiate an already-concrete `BSplineFamily` against a (possibly new) mesh and order: every rebuild of a
# B-spline level after its first construction comes here — overlay inheritance (`overlay`'s default `basis` is the base
# level's concrete family), `elevate`, a move, a mask edit, the fictitious fold, a level-id reindex, `estimate`'s
# refinement twin and the adaptive rebuild. It is the spec method on the family's own request, so each of them re-runs
# the continuity bound, the clamped thinness check and the mechanism choice exactly as a fresh build of the same request
# would, and a rebuild that keeps the mesh and the order reproduces an equal family. A family that kept its stored
# continuity instead, with only the mode checked, let an inherited or elevated level carry a continuity its new degree
# cannot have (`C¹` at degree 1), be thinner than the clamped mechanism needs, or keep a mechanism its new degree no
# longer implies (a `C²` request inherited at degree 4 silently became `C³`).
function Unfitted.instantiate_basis(family::BSplineFamily, mesh::CartesianMesh{D,T}, order::NTuple{D,Int},
                                    mode::Symbol) where {D,T<:Real}
    return Unfitted.instantiate_basis(_BSplineSpec(family.continuity), mesh, order, mode)
end

# ── Knot-vector construction ──────────────────────────────────────────────────

# Build the concrete `BSplineFamily` from a deferred `_BSplineSpec`, the mesh, and the per-axis polynomial degree.
# Uniform knots only: the cell-to-span map is the identity (`cell_to_span[d][c] = c`), and
# `dim(spaces[d]) = cells[d] + p[d]`.
function _materialize(spec::_BSplineSpec, mesh::CartesianMesh{D,T}, p::NTuple{D,Int}) where {D,T}
    m = spec.continuity
    spaces = ntuple(D) do d
        kv = _open_knot_vector(mesh.axes[d], p[d])
        BSplineSpace{p[d]}(kv)
    end
    derivative_spaces = ntuple(d -> BSplineDerivativeSpace{1}(spaces[d]), D)
    axis_coords = ntuple(d -> copy(mesh.axes[d]), D)
    cell_to_span = ntuple(D) do d
        collect(1:mesh.cells[d])  # uniform knots: span = cell index
    end
    return BSplineFamily{D,T,typeof(spaces),typeof(derivative_spaces)}(spaces, derivative_spaces, axis_coords,
                                                                       cell_to_span, m)
end

# Build a clamped open uniform knot vector for one axis from the mesh's per-axis coordinates and the requested degree
# `p`. End knots have multiplicity `p + 1` (clamped open knots); every interior knot has multiplicity 1 (max-regularity
# `C^{p − 1}` continuity across cell boundaries). The resulting knot vector aligns each cell with one span of the
# resulting `BSplineSpace`, so the per-cell span index in the B-spline parameter coincides with the cell's mesh index.
#
# Boundary conditions on the artificial overlay boundary (mesh-edge faces and mask-induced internal faces) are imposed
# by `_overlay_constraints!` below — by support selection at maximal continuity, by trace-vanishing linear constraints
# below it — *not* via knot multiplicity. This keeps the basis globally `C^{p − 1}` inside the active region regardless
# of mask geometry.
function _open_knot_vector(axis_coords::AbstractVector{T}, p::Integer) where {T<:Real}
    knots = T[]
    sizehint!(knots, length(axis_coords) + 2 * p)
    for _ in 1:(p+1)
        push!(knots, axis_coords[1])
    end
    for i in 2:(length(axis_coords)-1)
        push!(knots, axis_coords[i])
    end
    for _ in 1:(p+1)
        push!(knots, axis_coords[end])
    end
    return KnotVector(knots)
end

# ── Basis-interface methods ───────────────────────────────────────────────────

Unfitted.basis_name(::BSplineFamily) = :bspline
# The deferred spec answers to the same name: it is the family the user asked for, and `_check_basis_mode` reports
# through `basis_name` while validating a `space(...; basis=bspline(), mode=…)` call, before the spec is materialised.
Unfitted.basis_name(::_BSplineSpec) = :bspline

# ── Boundary / facet incidence ────────────────────────────────────────────────

# True iff cell-local 1D mode index `i ∈ 0:p` corresponds to a function that is non-zero on the requested side of the
# *level's* mesh box, where the clamped end knots of multiplicity `p + 1` leave exactly one non-zero 1D function: local
# index `0` (global index 1) at the lower edge, local index `p` (global index `dim`) at the upper edge.
#
# Read it as an outer-boundary predicate, not as a general per-cell support test. In-tree it is only ever asked about a
# boundary cell's outer face — `is_facet_basis` filters the Dirichlet boundary trace in `dirichlet.jl`, whose facets lie
# on the physical boundary. At an *interior* cell face the answer would be different: with simple interior knots, `p` of
# the `p + 1` local functions are non-zero at the face (only local index `p` vanishes at the cell's lower edge, only
# local index `0` at its upper edge), so a caller that needs interior-face support must ask the knot vector, not this.
function Unfitted.is_boundary_basis(family::BSplineFamily, id::CartesianIndex{D}, axis::Integer, side::Symbol) where {D}
    1 <= axis <= D || throw(ArgumentError("axis out of bounds"))
    p = degree(family.spaces[axis])
    side === :lower && return id.I[axis] == 0
    side === :upper && return id.I[axis] == p
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# ── Hot kernels: per-axis 1D B-spline value and derivative tables ─────────────

# Affine reference-to-parameter map for axis `d` on cell `c`: ξ ∈ [−1, 1] ↦ t ∈
# [axis_coords[d][c], axis_coords[d][c + 1]]. Returns `(t, half_width)` where `half_width = (t_upper − t_lower) / 2`.
# Stored half-width is reused for the derivative-axis chain rule: a B-spline basis function `N_i(t)` satisfies
# `dN/dξ = dN/dt · half_width`, so the reference-frame derivative the kernel needs to fill into `der1d[d]` already
# includes that factor.
@inline function _axis_parameter(family::BSplineFamily{D,T}, axis::Integer, cell::Integer, ξ::Real) where {D,T}
    coords = family.axis_coords[axis]
    t_lower = coords[cell]
    t_upper = coords[cell + 1]
    half_width = (t_upper - t_lower) / 2
    t = t_lower + (ξ + one(ξ)) * half_width
    return t, half_width
end

# Per-axis B-spline values into `val1d[d]` for one quadrature point. Calls `bsplinebasisall(spaces[d], span, t)` which
# returns the `p_d + 1` values on `span` as a stack-allocated `SVector{p_d + 1}` — no allocation; the SVector entries
# are then unrolled into the per-axis buffer.
function Unfitted._fill_factor_tables!(family::BSplineFamily{D,T,S,DS}, val1d, order::NTuple{D,Int}, ξ::SVector{D,Tξ},
                                       cell::CartesianIndex{D}) where {D,T,S,DS,Tξ}
    @inbounds for d in 1:D
        c = cell.I[d]
        span = family.cell_to_span[d][c]
        t, _ = _axis_parameter(family, d, c, ξ[d])
        values = bsplinebasisall(family.spaces[d], span, t)
        vd = val1d[d]
        for i in 1:(order[d]+1)
            vd[i] = values[i]
        end
    end
    return nothing
end

# Per-axis B-spline values + reference-frame derivatives. Values from `bsplinebasisall(spaces[d], span, t)`; raw
# `t`-derivatives from `bsplinebasisall(derivative_spaces[d], span, t)` (a fully unrolled
# `BasicBSpline.BSplineDerivativeSpace{1, …}` call); reference-frame `ξ`-derivatives obtained by multiplying with
# `half_width`. The downstream tensor-product kernel multiplies by the assembly-loop `scale[d] = 2 / edge_lengths[d]`
# factor; the two compose to `dN/dx = dN/dt · (h_d / 2) · (2 / h_d) = dN/dt`, the physical derivative — consistent with
# the integrated-Legendre chain-rule convention.
function Unfitted._fill_factor_tables!(family::BSplineFamily{D,T,S,DS}, val1d, der1d, order::NTuple{D,Int},
                                       ξ::SVector{D,Tξ}, cell::CartesianIndex{D}) where {D,T,S,DS,Tξ}
    @inbounds for d in 1:D
        c = cell.I[d]
        span = family.cell_to_span[d][c]
        t, half_width = _axis_parameter(family, d, c, ξ[d])
        values = bsplinebasisall(family.spaces[d], span, t)
        derivs = bsplinebasisall(family.derivative_spaces[d], span, t)
        vd = val1d[d]
        dd = der1d[d]
        for i in 1:(order[d]+1)
            vd[i] = values[i]
            dd[i] = derivs[i] * half_width
        end
    end
    return nothing
end

# ── Dof keys and boundary association ─────────────────────────────────────────

# The `AxisDofKey.kind` tag of a B-spline factor: its `index` is a global 1D function index along the axis, and its
# `mode` is unused (0). A tag need only be unique within one family (see the tag comment in `src/dofs.jl`), so it lives
# with the family. Its value differs from the integrated-Legendre tags although nothing requires it to.
const _AXIS_BSPLINE = UInt8(2)

function Unfitted._tensor_dof_key(level::Level{D,T,<:BSplineFamily}, cell::CartesianIndex{D},
                                  local_id::CartesianIndex{D}) where {D,T}
    family = level.basis
    axes = ntuple(d -> _bspline_axis_dof_key(family, d, cell.I[d], local_id.I[d]), D)
    return TensorDofKey{D}(level.id, axes)
end

# Build the per-axis dof key for the B-spline family. A cell-local 1D mode `m ∈ 0:p_d` on cell `c` maps to the global 1D
# function index `cell_to_span[d][c] + m`; that index uniquely identifies the function across the level, so adjacent
# cells naturally reuse the dof slot for every shared function through the `TensorDofKey` cache in `src/dofs.jl`.
@inline function _bspline_axis_dof_key(family::BSplineFamily, axis::Integer, cell_axis::Integer, mode::Integer)
    return AxisDofKey(_AXIS_BSPLINE, family.cell_to_span[axis][cell_axis] + mode, 0)
end

# True iff the per-axis B-spline factor of `key` anchors at the level's lower or upper mesh edge along `axis`. Used by
# the **physical Dirichlet** detection in `dirichlet.jl::_mark_physical_dirichlet!`: only the function that is
# interpolatory at the boundary (index 1 at the lower edge, index `dim` at the upper edge for an open clamped knot
# vector) carries the Dirichlet value. Higher-derivative constraints (`continuity ≥ 1`) at the **overlay-artificial**
# boundary are imposed by the linear-constraint rows [`_overlay_constraints!`](@ref) records, not through this
# predicate.
function Unfitted._key_on_level_side(key::TensorDofKey{D}, level::Level{D,T,<:BSplineFamily}, axis::Integer,
                                     side::Symbol) where {D,T}
    axis_key = key.axes[axis]
    axis_key.kind == _AXIS_BSPLINE || return false
    family = level.basis
    n = dim(family.spaces[axis])
    side === :lower && return axis_key.index == 1
    side === :upper && return axis_key.index == n
    throw(ArgumentError("boundary side must be :lower or :upper"))
end

# ── Support selection: the admissible region and the functions inside it ──────

# The cells of a level over which its functions may live — its *admissible region* — are the active ones, plus the
# inactive ones that hold no material.
#
# A `LevelMask` merges two deactivations (`_apply_physical_fold`), and what decides here is not which one switched a
# cell off but whether the cell holds material. An inactive cell that does — a user mask reaching into Ω — carries
# material that only the coarser level represents, so a function reaching into it would be truncated at a face where it
# does not vanish, and the superposition would lose its continuity there. An inactive cell that holds none — every cell
# the fictitious fold switched off, and a user mask entirely outside Ω — has nothing integrated over it, so the
# truncated and untruncated functions agree on Ω and the function may stay. That single distinction is the whole of this
# family's finite-cell compatibility — it is what keeps a cut cell's modes alive instead of constraining them away.
#
# The region is read lazily, cell by cell, off the build's material oracle (`ctx.fold`): its verdict table for the level
# answers whether an inactive cell is void, classifying the cell on its first ask and reading the stored verdict on
# every later one. The table, rather than a fresh classification per ask, because a degree-`p` function's support meets
# `(p + 1)^D` cells and every cell is asked about by up to `(p + 1)^D` functions. Lazily, rather than as a dense
# per-level table filled up front, because most inactive cells of a level lie farther than `p` cells from every active
# one, where no function's support reaches. A dense table asked about all of them for each record — for the selection
# and again for every burial test that asks this level as a cover — classifying each cell once per build and finding it
# in the box memo every later time. Measured on an 8² base under five band-shaped overlays around a hole, that was 82 %
# of the layout and 79 824 classifications, against the 1 092 integrated Legendre's coverage asks on the same stack;
# read lazily, the B-spline layout classifies 5 177 cells.
#
# The table is the build's one table for the level, shared rather than owned here: the selection, every burial test that
# builds a record on this level as a cover, the clamped trace faces (`_active_cell_at_face`), the burial's level-side
# test and the coverage pass all ask the same oracle. A grid cell is therefore classified at most once per layout build,
# whichever of them asks first, and only if one of them asks at all; a cell the fold already classified costs a lookup
# in the box memo behind the oracle.
#
# Everything a level's admissibility question needs is gathered here once per record: the level, which of its box faces
# are physical, its uniform per-axis cell width (for the virtual cells outside the mesh), and the oracle, which answers
# for the level's inactive cells through its verdict table and for those virtual cells through its box memo. A struct
# rather than five arguments because both the selection rule and the burial test ask the same question of the same
# level, and the burial test asks it of a *second* level as well.
struct _Admissibility{D,T,S,F}
    level::Level{D,T,S}
    lower_physical::NTuple{D,Bool}
    upper_physical::NTuple{D,Bool}
    width::NTuple{D,T}
    fold::F
end

# The admissibility record of `level`, read off the hook context of the level being processed. `level` may be another
# level of the same space — the burial test builds one for each covering level — because everything it takes from `ctx`
# (the domain, the tolerance and the material oracle) is space-wide. `ctx.fold` is concretely typed, so `F` is too, and
# every admissibility query specialises on it.
#
# Nothing is classified or allocated here: the oracle allocates the level's verdict table on its first miss, so an
# all-active level (a `nothing` mask), whose cells `_cell_admissible` never asks it about, gets none.
function _admissibility(level::Level{D,T,S}, ctx::_LevelContext{D,T}) where {D,T,S}
    mesh = level.mesh
    width = ntuple(d -> (mesh.domain.upper[d] - mesh.domain.lower[d]) / mesh.cells[d], D)
    domain, tol, fold = ctx.space.domain, ctx.tolerance, ctx.fold
    lower = ntuple(d -> _level_side_is_physical(level, domain, d, :lower, tol), D)
    upper = ntuple(d -> _level_side_is_physical(level, domain, d, :upper, tol), D)
    return _Admissibility{D,T,S,typeof(fold)}(level, lower, upper, width, fold)
end

# `_overlay_constraints!` at `continuity = :maximal`: one strong elimination per function whose support is not
# admissible, and nothing else.
#
# This is the whole artificial-boundary treatment for a maximal-continuity level, and it is a *selection* rather than a
# constraint system. The functions that survive are already globally `C^{p−1}` and identically zero outside their
# support, so their extension by zero is `C^{p−1}` with no equation to impose; the ones that do not survive are removed
# outright. Three consequences follow from that and none of them is incidental:
#
#   * `has_linear_constraints` stays `false`, so assembly never searches a region for pivots and never condenses one:
#     every slot is a plain active or constrained dof.
#   * There are no pivots, so the reconstruction path cannot mis-read one. The defect that `dof_value` used to carry is
#     unreachable here by construction rather than by repair.
#   * The cascade resolver never runs on a chain, so it cannot fill in, cannot lose a term to a relative drop tolerance,
#     and cannot cost `O(n²)` in the chain length.
#
# Arbitrary mask geometries are handled without a special case, because containment is a per-function test and knows
# nothing about the shape of the region it tests against.
function _selection_constraints!(ledger::EliminationLedger, level::Level{D,T,<:BSplineFamily},
                                 ctx::_LevelContext{D,T}) where {D,T}
    admissibility = _admissibility(level, ctx)
    for raw in ctx.raws
        _support_admissible(admissibility, ctx.raw_keys[raw]) || _eliminate!(ledger, raw, :overlay)
    end
    return ledger
end

# Whether the tensor B-spline named by `key` may stay: its full, unclipped support must be admissible.
function _support_admissible(a::_Admissibility{D,T,<:BSplineFamily}, key::TensorDofKey{D}) where {D,T}
    lo, hi = _support_block(a.level.basis, ntuple(d -> key.axes[d].index, D))
    return _block_admissible(a, lo, hi)
end

# The unclipped support block of the level's 1D function indices `i`: per axis the function with global index `i_d` is
# supported on cells `i_d − p_d … i_d`.
function _support_block(family::BSplineFamily{D}, index::NTuple{D,Int}) where {D}
    return ntuple(d -> index[d] - degree(family.spaces[d]), D), index
end

# Whether every cell of the block `lo … hi` is admissible — the one predicate both the selection rule and the burial
# test are expressed over. `lo`/`hi` are unclipped: an index outside `1 … n` is exactly the signal that the block
# reaches past the level's own box.
function _block_admissible(a::_Admissibility{D}, lo::NTuple{D,Int}, hi::NTuple{D,Int}) where {D}
    for c in CartesianIndices(ntuple(d -> lo[d]:hi[d], D))
        _cell_admissible(a, c) || return false
    end
    return true
end

# Whether cell `c` — which may lie outside the level's own grid — is admissible: whether a function supported there may
# be kept without the superposition losing its smoothness on Ω.
#
# Three cases, and the third is the one that makes this family work under an immersed geometry:
#
#   1. Inside the grid: active, or inactive and holding no material. The mask answers the first half; the second is
#      asked of the build's material oracle only for an inactive cell, through the level's verdict table (see
#      `_Admissibility`).
#   2. Outside the grid across a face that coincides with the space's own boundary. There is no artificial boundary
#      there and nothing beyond it is part of the discretised domain, so the question is vacuous and the answer is yes.
#      This is what keeps the clamped, interpolatory end functions of an unmasked base level — the ones the Dirichlet
#      path needs.
#   3. Outside the grid across an *artificial* face. The function would be truncated at that face, so it may stay only
#      if the material beyond carries nothing: if the cell is fictitious, the truncated and untruncated functions agree
#      on Ω and the extension is smooth *where it is integrated*. This is the same material exemption as case 1's second
#      half, applied past the level's own box, and without it a level whose face sits in fictitious material loses the
#      functions that reach ∂Ω — the ones the cut cells there need most.
#
# `@inline` because it is the body of `_block_admissible`'s per-cell loop, which every function's selection and every
# burial test runs over `(p + 1)^D` cells or more. Reading the mask and then the verdict table made it too large for the
# compiler to inline on its own, and a call per cell cost more than laziness saves on a small stack. Measured as the
# minimum of 400 `system_layout` builds of a 16² p3 plate around a hole under a nested 24² overlay: 0.38 ms with the
# call, 0.28 ms inlined, which is what the dense table took; and on the five-overlay band stack 4.3 ms with the call
# against 3.1 ms inlined.
@inline function _cell_admissible(a::_Admissibility{D}, c::CartesianIndex{D}) where {D}
    n = a.level.mesh.cells
    inside = true
    for d in 1:D
        if c.I[d] < 1
            a.lower_physical[d] && return true
            inside = false
        elseif c.I[d] > n[d]
            a.upper_physical[d] && return true
            inside = false
        end
    end
    inside && return is_active(a.level.mask, c) || _is_void(a.fold, a.level, c)
    return _is_void(a.fold, _virtual_cell_box(a, c))
end

# The box of cell `c` of the level's grid, for a `c` that may lie outside `1 … n`. The mesh is uniform by construction
# (`_mesh_axes`), so the grid continues past the box with the same width, and that continuation is exactly what a
# function reaching past the level's own face is supported on.
function _virtual_cell_box(a::_Admissibility{D,T}, c::CartesianIndex{D}) where {D,T}
    mesh = a.level.mesh
    lower = SVector{D,T}(ntuple(d -> mesh.domain.lower[d] + (c.I[d] - 1) * a.width[d], D))
    upper = SVector{D,T}(ntuple(d -> mesh.domain.lower[d] + c.I[d] * a.width[d], D))
    return AxisBox{D,T}(lower, upper)
end

# ── The artificial-boundary hook: clamped trace rows ──────────────────────────

# The B-spline `_overlay_constraints!`. At `continuity = :maximal` it is the support selection above. Below it, it
# records the homogeneous linear constraints encoding the trace-vanishing condition `∂^k u^(level) / ∂n^k = 0` for
# `k = 0, …, m` on every artificial face of the level's active region, as rows of the ledger (`_constrain!`, credited to
# `:overlay`). That replaces the per-raw `_has_overlay_constraint` predicate the integrated-Legendre family uses with a
# multi-raw constraint generator that exploits the tensor-product structure of the B-spline basis.
#
# Mathematics. At an artificial face perpendicular to axis `d` at physical coordinate `t* = axes[d][cell]`, the trace
# condition factors:
#
#     u(t*, x') = Σ_{α_d, α'} c_α N_{α_d}(t*) N_{α'}(x') = 0
#
# By linear independence of the `N_{α'}(x')` functions on the active face region, this reduces to one constraint per
# perpendicular multi-index `α'`:
#
#     Σ_{α_d = 0..p_d} N^{(k)}_{α_d}(t*) · c_{(α_d, α')} = 0
#
# The axis-d coefficients `N^{(k)}_{α_d}(t*)` are computed once per face via
# `bsplinebasisall(BSplineDerivativeSpace{k}, span, t*)` and reused across every perpendicular multi-index. The raw
# lookup for each `(α_d, α')` resolves through the standard `TensorDofKey → raw` map, with a representative active cell
# chosen on the active side of the face. For arbitrary mask geometries the *same* α' may produce *identical* constraints
# from multiple `(perp_cell, perp_local_mode)` iterations. The redundancy is correctness-preserving: the cascade
# resolver `_cascade!` substitutes the first emission's pivot into each re-emission, which then collapses to `0 = 0` and
# eliminates nothing. Its cost is per face — a re-emission is one more row of `p_d + 1` terms to substitute and combine
# for each perpendicular cell that touches the face.
#
# `continuity = 0` (`m = 0`) generates the classical C⁰ vanishing-trace constraint — the same condition a C⁰ knot
# insertion at the face would impose, but reached through constraints, which puts no restriction on the shape of the
# mask. `m > 0` adds vanishing normal derivatives up to order `m`, the trace structure an `H^{m+1}`-conforming essential
# boundary condition needs; for the B-spline/FCM setting see
#
#     M. Ruess, D. Schillinger, Y. Bazilevs, V. Varduhn, E. Rank,
#     *Weakly Enforced Essential Boundary Conditions for NURBS-embedded and
#     trimmed NURBS geometries on the basis of the Finite Cell Method*,
#     Int. J. Numer. Methods Engng. 95 (2013) 811–846.
function Unfitted._overlay_constraints!(ledger::EliminationLedger, level::Level{D,T,<:BSplineFamily},
                                        ctx::_LevelContext{D,T}) where {D,T}
    _selects_by_support(level.basis) && return _selection_constraints!(ledger, level, ctx)
    # The clamped path does not walk `ctx.raws`: the trace generator walks the mesh face/perp grids and looks raws up
    # through `ctx.raw_by_key`, and every key it builds carries `level.id`, so its rows name only this level's raws.
    # `ctx.fold` goes to `_active_cell_at_face`, which asks it whether the inactive side of an internal face holds
    # material; with no physical domain it classifies nothing.
    family = level.basis
    m = family.continuity
    n_cells = level.mesh.cells
    domain, tol, raw_by_key = ctx.space.domain, ctx.tolerance, ctx.raw_by_key

    for d in 1:D
        p_d = degree(family.spaces[d])
        P_d = family.spaces[d]
        # Perp cell and perp local-mode grids are invariant per axis d.
        perp_cells = CartesianIndices(ntuple(e -> e == d ? (1:1) : (1:n_cells[e]), D))
        perp_modes = CartesianIndices(ntuple(e -> e == d ? (0:0) : (0:degree(family.spaces[e])), D))

        for j in 0:n_cells[d]
            # Skip physical-boundary edges — physical-Dirichlet detection (`_physical_dirichlet`) handles them, not the
            # overlay constraint.
            j == 0 && _level_side_is_physical(level, domain, d, :lower, tol) && continue
            j == n_cells[d] && _level_side_is_physical(level, domain, d, :upper, tol) && continue

            t_star = j == 0 ? family.axis_coords[d][1] :
                     j == n_cells[d] ? family.axis_coords[d][end] : family.axis_coords[d][j + 1]

            # Two possible spans (= active-cell axis-d index) per face: `j` if the lower-cell is active, `j+1` if the
            # upper is. Boundary faces have a single span (lower_span == upper_span). Precompute the `(p_d + 1)`-vectors
            # of axis-d trace values per derivative order per span. One `bsplinebasisall` call per (k, span) — at most
            # `2(m + 1)` calls per face — instead of one per (perp_cell, perp_local, k) iteration.
            lower_span = j == 0 ? 1 : j
            upper_span = j == n_cells[d] ? n_cells[d] : j + 1
            coeffs_lower, coeffs_upper = _trace_coeffs(P_d, lower_span, upper_span, t_star, Val(m))

            for perp_cell in perp_cells
                located = _active_cell_at_face(level, d, j, perp_cell, n_cells, Val(D), ctx.fold)
                located === nothing && continue
                active_cell, span = located
                coeffs_for_span = span == lower_span ? coeffs_lower : coeffs_upper

                for perp_local in perp_modes
                    for kp1 in 1:(m+1)
                        coeffs = coeffs_for_span[kp1]
                        raws = Int[]
                        cs = T[]
                        sizehint!(raws, p_d + 1)
                        sizehint!(cs, p_d + 1)
                        for i in 0:p_d
                            local_id = CartesianIndex(ntuple(e -> e == d ? i : perp_local[e], D))
                            key = _tensor_dof_key(level, active_cell, local_id)
                            raw = get(raw_by_key, key, 0)
                            raw == 0 && continue
                            push!(raws, raw)
                            push!(cs, T(coeffs[i + 1]))
                        end
                        isempty(raws) || _constrain!(ledger, raws, cs, :overlay)
                    end
                end
            end
        end
    end
    return ledger
end

# Per-face trace coefficients (axis-d B-spline values and derivatives at `t_star`) for both potential spans, indexed by
# derivative order `k+1`. Returns `(coeffs_lower, coeffs_upper)` where each is a tuple of length `m+1` holding the
# `(p_d + 1)`-element `SVector` of trace values. When the two spans coincide (boundary faces), reuses the lower-span
# vectors directly to avoid the second `bsplinebasisall` call.
function _trace_coeffs(P_d, lower_span::Int, upper_span::Int, t_star, ::Val{m}) where {m}
    coeffs_lower = ntuple(kp1 -> _trace_at(P_d, lower_span, t_star, kp1 - 1), m + 1)
    coeffs_upper = lower_span == upper_span ? coeffs_lower :
                   ntuple(kp1 -> _trace_at(P_d, upper_span, t_star, kp1 - 1), m + 1)
    return coeffs_lower, coeffs_upper
end

@inline _trace_at(P_d, span, t_star, k::Int) = k == 0 ? bsplinebasisall(P_d, span, t_star) :
                                               bsplinebasisall(BSplineDerivativeSpace{k}(P_d), span, t_star)

# Returns `(active_cell::CartesianIndex{D}, span::Int)` if the face at axis `d`, position `j`, perpendicular cell
# `perp_cell` is artificial (i.e., contributes a trace constraint), `nothing` otherwise. For mesh edges, "artificial"
# reduces to "the boundary cell is active"; for internal faces, it's "the two adjacent cells differ in activity".
#
# With one exemption, and it is what lets this family carry an immersed `PhysicalDomain`. What decides an internal face
# is whether its inactive cell holds material, not which deactivation switched it off: an inactive cell that does — a
# user mask reaching into Ω — holds material only the coarser level represents, so the face between it and the active
# side is a real artificial boundary and the trace must vanish there; one that holds none — every cell the fictitious
# fold switched off, and a user mask entirely outside Ω — leaves no trace to vanish on, and constraining it would delete
# the cut cell's boundary modes — the very over-constraint the FCM solution must not suffer. `_has_overlay_constraint`
# in `src/dofs.jl` draws the same line for the integrated-Legendre family, and both ask it of `_is_fictitious`
# (`src/coverage.jl`) on the build's material oracle `fold`. With no physical domain every inactive cell holds material
# and nothing is classified.
function _active_cell_at_face(level::Level{D}, d::Int, j::Int, perp_cell::CartesianIndex{D}, n_cells::NTuple{D,Int},
                              ::Val{D}, fold::_Fold{D}) where {D}
    if j == 0 || j == n_cells[d]
        axis_d_idx = j == 0 ? 1 : n_cells[d]
        cell = CartesianIndex(ntuple(e -> e == d ? axis_d_idx : perp_cell[e], D))
        return is_active(level.mask, cell) ? (cell, axis_d_idx) : nothing
    end
    cell_j = CartesianIndex(ntuple(e -> e == d ? j : perp_cell[e], D))
    cell_jp1 = CartesianIndex(ntuple(e -> e == d ? j + 1 : perp_cell[e], D))
    a_j = is_active(level.mask, cell_j)
    a_jp1 = is_active(level.mask, cell_jp1)
    a_j == a_jp1 && return nothing
    _is_fictitious(fold, level, a_j ? cell_jp1 : cell_j) && return nothing
    return a_j ? (cell_j, j) : (cell_jp1, j + 1)
end

# ── Covered-mode pruning: dedup of what a covering level already contains ─────

# This family's side of the dof layer's one pruning driver, `_coverage_constraints!` (`src/dofs.jl`, whose docstring
# states the hooks' contract) — the dedup half only, and it is the whole of covered-mode pruning for this family.
#
# Integrated Legendre splits into a linear skeleton plus bubble modes, so it can shed the bubbles of a covered cell on
# their own and trade accuracy for dofs. A B-spline basis has no such split: its only reducible mode is one the covering
# level reproduces *exactly*, which leaves the discrete space unchanged and is therefore free. What it buys is not dof
# count but well-posedness — the reproduced function and its copy above are linearly dependent, so leaving both active
# makes the superposed operator exactly singular. The elimination is therefore unconditional here as everywhere: the
# only stack that keeps both copies is the unreduced twin `prepare(problem; prune = false)` builds, which opts out of
# the repair along with the reduction, exactly as it does for integrated Legendre. The family sheds nothing, so it keeps
# the driver's `_sheds_any` / `_sheds` defaults, and a level that no cover above may dedup (`_reproduces`,
# `_nested_over`) is done before a candidate is looked at.
#
# This is also the coarse half of the hierarchical selection rule: what it removes is exactly `{β : supp β ⊆ Ω^{k}}`,
# the functions Kraft's construction drops from the coarse level when a finer one takes over (Giannelli, Jüttler,
# Speleers, Comput. Aided Geom. Design 29 (2012) 485–498, Definition 1(I)). The fine half is the support selection in
# `_selection_constraints!`; together they are the hierarchical spline basis, stated for superposed independent grids.
function Unfitted._burial_cells(level::Level{D,T,<:BSplineFamily}, key::TensorDofKey{D}) where {D,T<:Real}
    return _support_cells(key, level.basis, level.mesh.cells)
end

function _support_cells(key::TensorDofKey{D}, family::BSplineFamily{D}, n::NTuple{D,Int}) where {D}
    return CartesianIndices(ntuple(d -> _axis_support(key.axes[d].index, degree(family.spaces[d]), n[d]), D))
end

# The cells this level's axis-`d` 1D function with global index `i` is supported on. Cell `c` carries the global indices
# `c … c + p` (uniform knots, `cell_to_span` the identity), so function `i` lives on `max(1, i − p) … min(ncells, i)` —
# up to `p + 1` cells. The core's `_axis_incidence` cannot answer this: it is written for the integrated-Legendre node /
# span keys and reports a single cell for any other tag.
@inline _axis_support(i::Integer, p::Integer, ncells::Integer) = max(1, i-p):min(ncells, i)

# The rows of the `_reproduces` trait (`src/dofs.jl`) with a B-spline side.
#
# Above a B-spline level of per-axis degree `p`, given the meshes nest: a buried function's support lies inside `k`'s
# active region, and it vanishes there to order `p − 1` (its own end knots are simple), so `k`'s trace conditions —
# orders `0 … m ≤ p − 1` — never exclude it. What is left is a question about the two spans:
#
#   * a B-spline `k` of the *same* degree. Interior knots are simple on both sides, so `k` is `C^{p−1}` exactly where
#     the buried function kinks and, under nesting, that function is one of `k`'s own splines. A *higher* degree does
#     not help — at a shared simple knot `k` would be `C^p`, too smooth to carry the kink — and a lower one cannot carry
#     the polynomial degree.
#   * an integrated-Legendre `k` in `:tensor` mode of order ≥ `p`. Its span is every `C⁰` tensor polynomial of that
#     degree on its cells, which contains a `C^{p−1}` spline of degree `p`. `:trunk` drops the mixed high-order terms,
#     so it does not.
#
# Every other family answers `false` (the trait's default). That is the safe direction: skipping a legitimate dedup
# leaves a rank-deficient operator, whereas deduping a function nothing reproduces would delete part of the space.
function Unfitted._reproduces(k::Level{D,T,<:BSplineFamily}, level::Level{D,T,<:BSplineFamily}) where {D,T<:Real}
    return all(d -> degree(k.basis.spaces[d]) == degree(level.basis.spaces[d]), 1:D)
end

function Unfitted._reproduces(k::Level{D,T,<:IntegratedLegendre}, level::Level{D,T,<:BSplineFamily}) where {D,T<:Real}
    k.mode === :tensor || return false
    p = ntuple(d -> degree(level.basis.spaces[d]), D)
    # `nominal_order(k)` is the level's per-axis maximum, which is not enough when `k` carries a per-cell order: a
    # single cell below `p` anywhere under the buried function's support cannot reproduce it. Requiring every palette
    # entry to clear `p` is the conservative reading — it can only skip a legitimate dedup, never delete a function
    # nothing reproduces.
    return all(o -> all(d -> o[d] >= p[d], 1:D), k.orders.palette)
end

# The reverse direction: whether a B-spline `k`, standing above an integrated-Legendre level, spans that level's C⁰ hats
# and may therefore deduplicate them. At degree 1 the open knot vector is the Q1 hat basis outright — simple interior
# knots give exactly C⁰, the clamped ends give the two boundary hats — so on a nested mesh the buried hat is one of
# `k`'s own functions. Any axis at degree ≥ 2 is C^(p−1) with p − 1 ≥ 1 at a simple interior knot, too smooth to carry
# the kink, and the whole tensor product goes with it. Answering per axis rather than per level is what makes this a
# mixed-family question: a degree-1 B-spline cover over an integrated-Legendre base is the C⁰ hat space and was
# previously refused the dedup by a family type test, leaving the stack exactly singular.
function Unfitted._reproduces(k::Level{D,T,<:BSplineFamily}, ::Level{D,T,<:IntegratedLegendre}) where {D,T<:Real}
    return all(d -> degree(k.basis.spaces[d]) == 1, 1:D)
end

# One reproduction test per covering level, chosen by what that cover actually is. A cover that selects by support
# answers the question *exactly*, by walking the subdivision and asking whether every fine function in it survived its
# own selection; it therefore needs that cover's admissibility record, which the driver asks for once per cover rather
# than once per buried function. Any other cover — an integrated-Legendre overlay, or a B-spline level below maximal
# continuity — is *clamped* on its artificial boundary, so its span near that boundary is not described by any statement
# about supports, and the conservative geometric predicate `_reproduced_on_domain` is the right one: it keeps the
# driver's `nothing` default, as does every cover of an integrated-Legendre level, a degree-1 selecting one included.
function Unfitted._cover_test(::Level{D,T,<:BSplineFamily}, k::Level{D,T,<:BSplineFamily},
                              ctx::_LevelContext{D,T}) where {D,T<:Real}
    return _selects_by_support(k.basis) ? _admissibility(k, ctx) : nothing
end

# True iff the covering level `k`, whose admissibility record is `cover`, reproduces on Ω the function of `level` named
# by `key` — the burial test in the form support selection makes available (the driver's exact test, `_cover_test`), and
# the one that is **exact** rather than conservative.
#
# Subdivision is the whole argument. On nested meshes of equal degree a B-spline of the coarse level is a combination,
# with strictly positive coefficients, of exactly those fine B-splines whose support lies inside its own, and that
# representation is unique. So the coarse function is reproduced **iff every one of those fine functions is itself kept
# by `k`** — sufficiency because the combination is then available verbatim, necessity because a dropped term cannot be
# replaced: a fine-space function vanishing on Ω has no coefficient on any fine function whose support meets Ω, so the
# coefficients of the live terms are forced.
#
# Both halves are decided by one predicate, `_block_admissible` on `k`, which is the same predicate `k` used to select
# its own functions. That is what makes the test exact rather than a geometric approximation of it, and it is why the
# fictitious-fold cases need no special handling here: a fine function supported entirely in fictitious material is
# admissible on `k` for the same reason it contributes nothing, so it can neither block the dedup nor be missed by it.
#
# This is what the clamped path cannot say. There, `k` keeps functions that do not vanish on its own box face, so its
# span near that face is larger than any statement about supports describes, and it can reproduce a *combination* of two
# buried functions without reproducing either — a dependency a per-function dedup cannot see, and one exact null mode
# per combination in the assembled operator.
function Unfitted._reproduced_by(cover::_Admissibility{D,T,<:BSplineFamily}, level::Level{D,T,<:BSplineFamily},
                                 key::TensorDofKey{D}) where {D,T<:Real}
    coarse = level.basis
    index = ntuple(d -> key.axes[d].index, D)
    clo, chi = _support_block(coarse, index)
    # The support as coordinates on the coarse grid, continued past its own box the same way the grid is.
    mesh = level.mesh
    width = ntuple(d -> (mesh.domain.upper[d] - mesh.domain.lower[d]) / mesh.cells[d], D)
    lower = SVector{D,T}(ntuple(d -> mesh.domain.lower[d] + (clo[d] - 1) * width[d], D))
    upper = SVector{D,T}(ntuple(d -> mesh.domain.lower[d] + chi[d] * width[d], D))
    flo, fhi = _fine_cell_range(cover, lower, upper)
    fine = cover.level.basis
    # "Every fine function whose own support fits inside the block is admissible" is the same statement as "every cell
    # of the block is admissible", because those functions' supports tile it exactly: as `m` sweeps `flo + p … fhi`, the
    # union of `m − p … m` is `flo … fhi`, no more and no less. Asking it per function re-tests each cell up to
    # `(p + 1)^D` times for an identical answer, so it is asked once. The emptiness guard is the one case where the two
    # differ — a block too narrow to hold a single fine function reproduces nothing — and `false` is the answer there.
    any(d -> fhi[d] < flo[d] + degree(fine.spaces[d]), 1:D) && return false
    return _block_admissible(cover, flo, fhi)
end

# The cells of `k`'s grid — virtual indices included — whose closure lies inside the coordinate box `lo … hi`. Used to
# turn a buried function's support, which is stated in the *coarse* level's cells, into the fine cell block the
# subdivision lives on.
#
# The meshes nest (the driver has already asked `_nested_over`), so every coordinate involved lands on one of `k`'s knot
# lines and the division is exact up to floating point; rounding to the nearest integer is therefore the right reading
# rather than a tolerance choice, and it extends past `k`'s own box because the grid does.
function _fine_cell_range(a::_Admissibility{D,T}, lower::SVector{D,T}, upper::SVector{D,T}) where {D,T}
    origin = a.level.mesh.domain.lower
    lo = ntuple(d -> round(Int, (lower[d] - origin[d]) / a.width[d]) + 1, D)
    hi = ntuple(d -> round(Int, (upper[d] - origin[d]) / a.width[d]), D)
    return lo, hi
end

# The dedup is a strong elimination on a selecting level, but a one-term ROW on a clamped one. There a buried function
# can also be a term of a trace row, and only resolving the level's rows in order decides which of the two eliminates it
# and is credited: the trace rows come first, so a function they eliminate outright stays credited to `:overlay`, and
# its dedup row collapses to `0 = 0`. Eliminating it directly instead would remove it *before* the trace rows are
# resolved and change which raws they pivot (see `EliminationLedger`).
function Unfitted._dedup!(ledger::EliminationLedger{T}, level::Level{D,T,<:BSplineFamily}, raw::Int) where {D,T<:Real}
    _selects_by_support(level.basis) && return _eliminate!(ledger, raw, :dedup)
    return _constrain!(ledger, [raw], [one(T)], :dedup)
end

# ── Resolving a level's rows: the cascade ─────────────────────────────────────

# The dof layer's resolve hook for this family. A level clamped at an integer `continuity` records its trace rows and,
# under a covering level, its dedups as linear-constraint rows (`_constrain!`); the core stores them and calls this once
# per level, after both of the level's hooks, and only when a row is pending, so a selecting (`:maximal`) level, which
# records none, never reaches it. The cascade below lives here rather than in the core because this family is its only
# consumer: the core keeps the ledger, the pivot table and a default that raises for a family that records rows without
# resolving them.
Unfitted._resolve_rows!(ledger::EliminationLedger, ::Level{D,T,<:BSplineFamily}) where {D,T} = _cascade!(ledger)

# Combine repeated raws in a list of `(raw, coefficient)` pairs by summing the coefficients, dropping terms whose
# coefficient falls below the working tolerance. Pure-data helper consumed by the cascade resolver below; sorts in place
# for compactness.
#
# A one-term list is not a special case, and must not be short-circuited past the drop test. A multi-raw constraint
# re-emitted after its raws were pivoted away can leave exactly one surviving term whose coefficient is zero —
# `0·u = 0`, which constrains nothing — and returning that unfiltered makes the resolver pivot on it and eliminate `u`
# outright.
function _combine_terms(terms::Vector{Tuple{Int,T}}) where {T}
    isempty(terms) && return terms
    sort!(terms; by=first)
    write = 1
    @inbounds for read in 2:length(terms)
        if terms[read][1] == terms[write][1]
            terms[write] = (terms[write][1], terms[write][2] + terms[read][2])
        else
            write += 1
            terms[write] = terms[read]
        end
    end
    resize!(terms, write)
    drop_tol = sqrt(eps(T)) * maximum(t -> abs(t[2]), terms; init=one(T))
    filter!(t -> abs(t[2]) > drop_tol, terms)
    return terms
end

"""
    _cascade!(ledger::EliminationLedger) -> ledger

Resolve the current level's rows, `ledger.rows`, by cascade elimination and record every raw they eliminate with
`_pivot!`: strongly, when its expansion is empty, or as a *linear-constraint pivot* `u_raw = Σᵢ wᵢ u_{rᵢ}` over raws
that stay free.

The algorithm is the classical sparse Gauss elimination, adapted to exploit the small constraint matrix's structure:

  1. For each row, in emission order, substitute what is already known into its terms: a raw pivoted by an earlier row
     of this level contributes its expansion, a raw that is still free contributes itself, and a raw eliminated before
     the rows (by `_eliminate!`) contributes nothing.
  2. Combine like terms and drop near-zeros; skip the row entirely if it collapses to `0 = 0`. A skipped row eliminates
     nothing and is credited with nothing.
  3. Pick a pivot by largest absolute coefficient (numerical stability).
  4. Express the pivot as a linear combination of the remaining terms' raws, and credit it to the row's source.
  5. Back-substitute the pivot into every earlier pivot's expansion that mentions the new pivot raw.

Step 5 keeps the invariant: at any point after a row has been processed, *every pivot's expansion is in terms of
currently-free raws only*. Assembly emission then never needs to chase chains. The pivots are written to the ledger once
the last row is processed, in raw order; storage is per raw, so the order is immaterial.

Each row costs a substitution and a sort-and-combine over its terms after substitution, plus one rewrite per earlier
pivot that mentions the new pivot raw, found through the `mentions` index rather than by scanning every expansion (see
`_back_substitute!`).

The working tables are dense vectors over the span `lo:hi` of the raws the rows name, entry `raw - (lo - 1)` for `raw`.
That span holds every raw a table is ever indexed by — a term, a pivot and a mention all name a raw some row names —
and, since a level's rows name only its own raws, it lies inside the level. Dense level-local tables instead of hash
tables keyed by raw: the arithmetic is the same, but no new `Dict` type is compiled at the first build that resolves a
row, and the lookups are plain loads.
"""
function _cascade!(ledger::EliminationLedger{T}) where {T}
    lo, hi = typemax(Int), typemin(Int)
    for (raws, _, _) in ledger.rows, raw in raws
        lo, hi = min(lo, raw), max(hi, raw)
    end
    lo > hi && return ledger                    # no row names a raw: nothing to resolve
    offset, n = lo - 1, hi - lo + 1
    # Per raw: the source of the row that pivoted it, `:free` until one does, and its current expansion, assigned with
    # it.
    sources = fill(:free, n)
    expansions = Vector{Vector{Tuple{Int,T}}}(undef, n)
    # Per raw: the pivots whose expansion mentions it (see `_back_substitute!`).
    mentions = Vector{Vector{Int}}(undef, n)

    for (raws, coefficients, source) in ledger.rows
        # 1) Substitute already-pivoted raws; drop already-eliminated ones.
        terms = Tuple{Int,T}[]
        sizehint!(terms, length(raws))
        for (raw, coeff) in zip(raws, coefficients)
            if sources[raw - offset] !== :free
                for (other, w) in expansions[raw - offset]
                    push!(terms, (other, coeff * w))
                end
            elseif ledger.source[raw] === :free
                push!(terms, (raw, coeff))
            end
        end

        # 2) Combine like terms; trivially satisfied rows fall away.
        combined = _combine_terms(terms)
        isempty(combined) && continue

        # 3) Pick pivot by largest |coefficient| (numerical stability).
        _, pivot_pos = findmax(t -> abs(t[2]), combined)
        pivot_raw, pivot_coeff = combined[pivot_pos]

        # 4) Build the pivot expansion (in terms of remaining raws).
        new_expansion = Tuple{Int,T}[(raw, -c / pivot_coeff) for (k, (raw, c)) in pairs(combined) if k != pivot_pos]

        expansions[pivot_raw - offset] = new_expansion
        sources[pivot_raw - offset] = source
        _register_mentions!(mentions, offset, pivot_raw, new_expansion)

        # 5) Back-substitute into earlier pivots that mention this raw.
        _back_substitute!(expansions, mentions, offset, pivot_raw, new_expansion)
    end
    for (k, source) in pairs(sources)
        source === :free || _pivot!(ledger, k + offset, expansions[k], source)
    end
    return ledger
end

# Back-substitute a freshly created pivot's expansion into every other pivot's expansion that mentions the new pivot
# raw. Necessary because our greedy pivot ordering doesn't guarantee that a constraint's pivot choice is downstream of
# all its references — when a later constraint pivots a raw that an earlier pivot's expansion still mentions, the
# earlier expansion needs the substitution to stay in "free raws only" form.
#
# `mentions` lists, per raw, the pivots whose expansion currently contains it, so this visits only the expansions that
# can possibly need rewriting. It used to scan all `nraw` expansions per constraint, which made the pass `O(K · nraw)` —
# quadratic in the problem — while the comment here claimed `O(K · D · p)`, "negligible at problem scale". It was not
# negligible: on a covered stack the dof layout reached 98.7% of `prepare` and 2.8× the cost of `assemble!`, growing
# ×12.7 and ×14.8 per ×4 in unknowns. The scan's answer was almost always "nothing to do", because every integrated
# Legendre elimination was then a single-raw strong constraint whose pivot expansion is empty and which therefore
# mentions no raw at all. (Those are direct `_eliminate!` calls now and never reach the resolver.)
#
# The index is allowed to over-report: an expansion rewritten by an earlier substitution may no longer contain the raw
# that registered it, so the membership test below stays. It must never under-report, which is why an entry is added at
# *every* site that writes an expansion — here and in `_cascade!` step 4. Every holder it lists is a pivot, since only a
# pivot's expansion is ever registered, so each one has an assigned `expansions` entry.
#
# Both tables are `_cascade!`'s level-local ones: entry `raw - offset` belongs to `raw`, and an entry stays unassigned
# until something is written to it.
function _back_substitute!(expansions::Vector{Vector{Tuple{Int,T}}}, mentions::Vector{Vector{Int}}, offset::Int,
                           pivot_raw::Int, pivot_expansion::Vector{Tuple{Int,T}}) where {T}
    isassigned(mentions, pivot_raw - offset) || return expansions
    @inbounds for raw in mentions[pivot_raw - offset]
        raw == pivot_raw && continue
        expansion = expansions[raw - offset]
        # The index over-reports; this is the exact test.
        any(t -> first(t) == pivot_raw, expansion) || continue
        # Rebuild with substitution.
        new_terms = Tuple{Int,T}[]
        sizehint!(new_terms, length(expansion) + length(pivot_expansion))
        for (other, w) in expansion
            if other == pivot_raw
                for (sub_other, sub_w) in pivot_expansion
                    push!(new_terms, (sub_other, w * sub_w))
                end
            else
                push!(new_terms, (other, w))
            end
        end
        combined = _combine_terms(new_terms)
        expansions[raw - offset] = combined
        # This expansion now mentions the pivot's own references. `pivot_raw` cannot be among them — `pivot_expansion`
        # was built excluding it — so the list being iterated is never the one appended to.
        _register_mentions!(mentions, offset, raw, combined)
    end
    return expansions
end

# Record that `raw`'s expansion contains each of `terms`' raws, in the level-local `mentions` table of
# `_back_substitute!`. A raw's list is allocated by its first mention, so a level whose pivots are all strong
# eliminations allocates none.
function _register_mentions!(mentions::Vector{Vector{Int}}, offset::Int, raw::Int,
                             terms::Vector{Tuple{Int,T}}) where {T}
    for (other, _) in terms
        k = other - offset
        isassigned(mentions, k) ? push!(mentions[k], raw) : (mentions[k] = [raw])
    end
    return mentions
end

# ── Family traits ─────────────────────────────────────────────────────────────

# A degree-`p_d` B-spline spans `p_d + 1` cells along axis `d`, so a marked region has to be dilated by `p_d` cells per
# side before the level carries the functions the marking meant to wake. Read by `support_extension`; the
# `::BasisFamily` default of zero would silently leave an adaptive loop marking regions that generate no unknowns at
# all.
#
# The radius comes from the family's own degrees rather than from the `order` argument, which on this family is the same
# number — a B-spline level's degree belongs to its knot vectors, so there is one degree per axis and no per-cell field
# to disagree with.
function Unfitted._support_radius(family::BSplineFamily{D}, ::NTuple{D,Int}) where {D}
    # The radius belongs to the SELECTING mechanism. A level clamped at an integer `continuity` keeps `n + p − 2(m + 1)`
    # functions per axis and carries some on a region one cell wide, so it needs no room made for it and a nonzero
    # radius here would make `refine` wake cells nothing asked for.
    _selects_by_support(family) || return ntuple(_ -> 0, D)
    return ntuple(d -> degree(family.spaces[d]), D)
end

end # module UnfittedBasicBSplineExt
