# The public entry points of Unfitted's package extensions, declared and documented in the core.
#
# Five package extensions install methods on names declared here: Tensors.jl the tensor-notation readers
# `symmetric_gradient`, `value_vec` and `gradient_tensor`; BasicBSpline.jl the B-spline family's factory `bspline`;
# MeshIO.jl the mesh level sets `mesh_levelset` and `stl_levelset` (MeshIO loads FileIO and GeometryBasics, the other
# two triggers of its extension, so `using MeshIO` alone installs them); WriteVTK.jl the VTK writers; and Makie.jl the
# plots `levelplot` and `quadratureplot`. Every such name is declared and documented here rather than in its extension,
# so that `?name` answers in a bare `using Unfitted` session (an extension's docstrings land in the extension module's
# own docs, not this one's) and downstream code can name it unconditionally.
#
# An entry point is declared in one of two ways, decided by who owns the method that matches every call:
#
#   * **A fallback method**, for an entry point whose extension defines its methods one signature at a time. The
#     fallback is one method over every argument list, strictly less specific than every method its extension
#     installs, and its body is `_extension_missing`: without the extension the call raises an error that says which
#     package to load, before the call has touched anything, and with it a call no method accepts raises the ordinary
#     `MethodError`. That error bears the one trace the fallback leaves: since the fallback matches every call, Julia
#     prints it among the closest candidates and adds that the error was thrown manually, as if the method were
#     deliberately left unimplemented.
#   * **No method at all**, for a Makie recipe. `Makie.@recipe` defines the plot function's catch-all
#     `f(args...; kw...)` itself, in the extension, and an extension may not overwrite a method of its parent while it
#     precompiles (Julia refuses with "Method overwriting is not permitted during Module precompilation"), so the core
#     cannot hold one. The function is declared methodless and listed in `_METHODLESS_ENTRY_POINTS`: a call made
#     without the extension is then an ordinary `MethodError`, to which the error hint `_extension_hint` adds the
#     package to load. With the extension loaded, a call the recipe cannot draw is refused by Makie, naming the argument
#     types it expected.
#
# The core defines no other method of these functions: one more specific than an extension method in some argument
# would take that method's calls, or make them ambiguous. A further signature an extension implements is documented on
# its abstract signature, which no method of the core implements: the tensor readers' `Val` and `Field` forms and the
# VTK series operations below. The `Val` and `Field` forms name `TrialChannels` and `Field`, which `problems.jl`
# defines, so `Unfitted.jl` includes this file last.
#
# `symmetric_gradient` is declared here rather than extended from Tensors.jl, because Tensors.jl does not export a
# function under that name; the closest cousins it ships are the AD entry points `gradient`, `hessian` and
# `divergence`.

# ── Tensor notation: the Tensors extension ────────────────────────────────────

"""
    symmetric_gradient(trial::TrialChannels{D,T}) -> SymmetricTensor{2,D,T}

Rank-1 symmetric gradient

    ε(N(x) eₖ) = ½ (eₖ ⊗ ∇N + ∇N ⊗ eₖ),    k = trial.component,

of the single vector-field trial basis function currently being assembled, returned as a `SymmetricTensor{2,D,T}`.
Allows a bilinear callback to read

    ε_trial = symmetric_gradient(trial)
    σ_trial = ℂ ⊡ ε_trial          # any constitutive law
    return TestChannels(0.0, σ_trial, test_component)

without hand-rolling the Voigt ↔ tensor conversion or the off-diagonal factor-of-two bookkeeping. The minor symmetry of
the strain tensor is preserved by the `SymmetricTensor{2,D}` storage so contractions cost `D(D+1)/2` multiplies rather
than `D²`.

`trial.component` is the component index of the trial basis function currently being assembled, exactly as documented on
[`TrialChannels`](@ref). Reading it as the spatial index `k` of `eₖ` assumes a field with one component per spatial axis
(`C = D`), which is what a strain tensor means. Neither mismatch is detected here, and the two fail in opposite ways:

  - At `C > D` a component index above `D` cannot appear in a `SymmetricTensor{2,D}`, so the result is the zero tensor.
  - At `C < D` the result is a `D × D` tensor whose surplus rows and columns carry derivatives along axes the field has
    no component for. The space-time case is the one that bites: on a mesh whose last axis is time, a displacement field
    with one component per *spatial* axis has `C = D − 1`, and the surplus slots hold `∂ₜ`. Nothing downstream can tell
    — the operator stays symmetric and positive semi-definite, it solves, it converges under refinement — yet under an
    isotropic law it carries exactly the spurious `μ ∫ ∂ₜu · ∂ₜv` those slots describe, an identity
    `test/test_tensors_ext.jl` pins.

Pass the component count — `symmetric_gradient(trial, Val(C))` or `symmetric_gradient(trial, field)` — to get the
`C × C` strain the field actually has. The two return the same tensor and differ only in what they cost the closure that
calls them, which the `::Field` method's docstring measures.

Requires `using Tensors`, which loads the package extension that implements it; a call made without it raises an
error that says so.
"""
symmetric_gradient(args...; kwargs...) = _extension_missing(symmetric_gradient, args, "Tensors")

"""
    value_vec(state::FormState, name::Symbol, ::Val{D}) -> Vec{D,T}

Read a vector field's value at the current quadrature point as a `Vec{D,T}`. `D` must be passed via `Val(D)` (or
equivalent) so the return type is statically known. Replaces

    SVector(value(state, name, 1), …, value(state, name, D))

at the constitutive-law call site.

Requires `using Tensors`, which loads the package extension that implements it; a call made without it raises an
error that says so.
"""
value_vec(args...; kwargs...) = _extension_missing(value_vec, args, "Tensors")

"""
    gradient_tensor(state::FormState, name::Symbol, ::Val{D}) -> Tensor{2,D,T}

Read the full Jacobian `[∂uᵢ/∂xⱼ]` of a vector field at the current quadrature point as a 2nd-order tensor. The
component layout is `G[i, j] = ∂(component i)/∂xⱼ`, matching the Tensors.jl convention. Combine with `Tensors.symmetric`
to obtain the small-strain tensor

    ε(u) = symmetric(gradient_tensor(state, :u, Val(D)))

without ever materialising a Voigt 3-vector. As with [`value_vec`](@ref) the dimension must be passed via `Val(D)` for
type inference.

The single `Val` argument sizes **both** tensor indices — `Val(n)` reads `n` gradient rows and returns an `n × n` tensor
— so this is the field's Jacobian only when the field has one component per axis (`C = D`). On a field with `C ≠ D`
neither argument gives one:

  - `Val(C)` returns the `C × C` block `[∂uᵢ/∂xⱼ]`, `i, j ∈ 1:C`. At `C < D` that is the Jacobian with its columns
    **truncated** to the first `C` axes, not the field's `C × D` Jacobian.
  - `Val(D)` raises a `BoundsError` at the first component the field does not carry.

A field with `C ≠ D` has no square Jacobian, and Tensors.jl has no rectangular 2nd-order tensor to return it as, so
there is nothing for this function to widen to. Read such a field's gradient one component at a time with
`field_gradient(state, name, k)`, which returns the `k`-th row as an `SVector{D}`.

Requires `using Tensors`, which loads the package extension that implements it; a call made without it raises an
error that says so.
"""
gradient_tensor(args...; kwargs...) = _extension_missing(gradient_tensor, args, "Tensors")

# ── Tensor notation: the further signatures of the Tensors extension ──────────
#
# The `Val` and `Field` forms of the tensor-notation readers declared above, documented on abstract signatures that no
# method of the core implements.

"""
    symmetric_gradient(trial::TrialChannels{D,T}, ::Val{C}) -> SymmetricTensor{2,C,T}

Rank-1 symmetric gradient of the trial basis function currently being assembled, sized by the **component count** `C` of
the field it belongs to rather than by the mesh dimension `D`:

    ε(N(x) eₖ)ᵢⱼ = ½ (δᵢₖ ∂ⱼN + δⱼₖ ∂ᵢN),    i, j ∈ 1:C,    k = trial.component.

This is the form to use whenever a field has fewer components than the mesh has axes — a space-time mesh, where the last
axis is time and the displacement field carries one component per spatial axis, being the case that motivates it. The
one-argument [`symmetric_gradient`](@ref) sizes the strain by `D` there and silently adds the time-derivative slots to
the operator.

The result is a genuine `C × C` tensor and not a `D × D` one with the surplus slots zeroed, because a padded strain lies
about its own size: `dev(ε)` would subtract `tr(ε)/D` instead of `tr(ε)/C`, every invariant would be formed in the wrong
space, and an anisotropic `ℂ` would contract against a slot the material does not have. Returning the smaller tensor is
therefore not an optimisation but the only honest answer, and the constitutive law that consumes it is written over `C`
axes too:

    ε_trial = symmetric_gradient(trial, Val(2))   # on a 3-axis space-time mesh
    σ_trial = ℂ ⊡ ε_trial                         # ℂ::SymmetricTensor{4,2}
    return TestChannels(0.0, σ_trial, test_component)

The `C`-length gradient coefficient that comes back from the three-argument [`TestChannels`](@ref) constructor is
zero-extended to the mesh dimension by assembly, which is what makes the return path above work unchanged.

`C > D` raises a `DimensionMismatch`: a field with more components than the mesh has axes has no strain tensor, and the
missing derivatives cannot be invented.

Of the two two-argument forms this is the one with no capture cost: `Val(C)` is a singleton, whereas a captured
[`Field`](@ref) carries its whole `Space` into the closure and costs a heap box per call. Prefer it in a form written
inside a function; the `::Field` method's docstring has the measurement.

**This overload is transitional.** [`TrialChannels`](@ref) does not yet carry the component count as a type parameter;
once it does, the one-argument `symmetric_gradient(trial)` is correctly sized on its own and both two-argument forms
become redundant.
"""
symmetric_gradient(trial::TrialChannels{D,T}, ::Val{C}) where {D,T,C}

"""
    symmetric_gradient(trial::TrialChannels, field::Field) -> SymmetricTensor{2,C}

Sugar for `symmetric_gradient(trial, Val(C))` with `C` the component count of `field`, read off its type, so a weak form
written against a named [`Field`](@ref) never spells its own component count twice:

    bilinear(q, trial, c) = TestChannels(0.0, ℂ ⊡ symmetric_gradient(trial, u), c)

Identical to the `Val(C)` form in every respect, including the `C > D` `DimensionMismatch` — see
[`symmetric_gradient`](@ref) for the convention, for why the result is `C × C`, and for the note that both two-argument
forms are transitional.

How the field reaches the callback decides what it costs. The callback runs once per test component per trial dof per
quadrature point, so every figure below is per call, measured on Julia 1.12:

  - **`const` global, or `Val(C)`** — 0 B, and a concrete `TestChannels`. Bind the field `const` when the closure is
    written at top level, as an example script does, or pass `Val(C)` there instead.
  - **Non-`const` global** — read as `Any` on every call: 240 B, with the return type inferred as `Any`. The cost is the
    global read and not the field argument — a non-`const` global `ℂ` costs 192 B the same way — but this is the form
    that puts a field read inside the hot loop.
  - **Captured local** — a form written *inside a function* genuinely captures the field by value, and the field brings
    the whole [`Space`](@ref) with it: `sizeof(Field) = 248 B`, of which 240 B is the space, so the closure grows to 320
    B and the enclosing `WeakForm` to 328 B, against 72 B and 80 B for a captured `Val(C)`. Size is not the trigger,
    though; losing `isbits` is. A `Space` is not an `isbits` type, so neither is a `WeakForm` holding one, and once a
    problem's blocks stop sharing a single form type the per-quadrature-point block walk has to heap-box that form at
    every callback call — 336 B a box after GC pool rounding. On an 8-region, 216-quadrature-point space-time problem
    whose displacement block makes 216 × 108 = 23 328 such calls, that is 23 328 boxes and 7.84 MB per assembly pass,
    while the `Val(C)` capture, still `isbits`, boxes nothing at all.

The shipped `examples/applications/spacetime_cavity_2d` pays none of this: its `u` is a `const` global and its bilinear
forms are named top-level methods, so nothing is captured and each `WeakForm` is 2 B.
"""
symmetric_gradient(trial::TrialChannels, field::Field{D,T,C}) where {D,T,C}

"""
    value_vec(state::FormState, field::Field{D,T,C}) -> Vec{C,T}

Read the field's value at the current quadrature point as a `Vec{C,T}`. Equivalent to
`value_vec(state, field.name, Val(C))` (see [`value_vec`](@ref)) but infers the component count from the field's type
parameters.
"""
value_vec(state, field::Field{D,T,C}) where {D,T,C}

"""
    gradient_tensor(state::FormState, field::Field{D,T,D}) -> Tensor{2,D,T}

Read the field's physical Jacobian at the current quadrature point as a 2nd-order tensor. Defined for vector fields
whose component count equals the spatial dimension (`Field{D,T,D}`), which is the only case that has a square Jacobian
at all.

A field with `C ≠ D` components has a `C × D` Jacobian, and the `Val`-explicit overload above cannot produce it either:
its one argument sizes both tensor indices, so `Val(C)` truncates the columns to the first `C` axes and `Val(D)` raises
a `BoundsError`. Read such a field's gradient one component at a time with `field_gradient(state, name, k)` instead; the
`MethodError` this signature raises is that statement.
"""
gradient_tensor(state, field::Field{D,T,D}) where {D,T}

# ── B-spline basis: the BasicBSpline extension ────────────────────────────────

"""
    bspline(; continuity = :maximal) -> _BSplineSpec

Mark a [`space`](@ref) / [`overlay`](@ref) call as using the open-knot tensor B-spline basis family. The per-axis
polynomial degree is taken from the `order` keyword on `space` / `overlay` (matching the integrated Legendre
convention), so a typical use looks like

```julia
using Unfitted, BasicBSpline
V = space(box((0.0, 0.0), (1.0, 1.0)); cells=8, order=3, basis=bspline())
```

Requires `using BasicBSpline`, which loads the package extension that implements the family; a call made without it
raises an error that says so.

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
`_reproduced_on_domain` uses. A clamped overlay can reproduce a combination of two buried functions while reproducing
neither on its own only behind a clamped face that no material crosses, and the artificial boundary is clamped only
where its face holds material, so the configuration that used to show it — the overlay's box face on a knot of the level
below inside a band of cut cells, 8 exact null modes on a 6×6 degree-2 base over Ω = {x₁ ≤ 0.9} in [−1, 2]² with a
quarter-cell overlay ending at x = 1 — is full rank. The exception is a face that ∂Ω lies on exactly, which counts as
holding material and stays clamped.

The only stack that deliberately carries both copies is the **unreduced twin**
[`prepare`](@ref)`(problem; prune = false)` builds, and on a nested stack that twin is exactly singular by construction
— which is what it is for. It exists to be measured against, not to be solved with.

# Why a deferred spec

The actual per-axis `BSplineSpace`s depend on the level's mesh-cell coordinates, which do not exist yet when the
`basis=` value is chosen. `bspline(...)` therefore returns a deferred [`_BSplineSpec`](@ref); the concrete
[`BSplineFamily`](@ref) is built by the [`instantiate_basis`](@ref) hook in the BasicBSpline extension once the mesh is
known. A mask changes nothing about the knot vectors — a mask face selects functions rather than inserting knots — so a
level's family is rebuilt on a move but is the same family before and after a mask update. The spec is itself a
[`BasisFamily`](@ref) subtype so the existing `space` / `overlay` code path accepts it without special casing.
"""
bspline(args...; kwargs...) = _extension_missing(bspline, args, "BasicBSpline")

# ── Mesh level sets: the MeshIO extension ─────────────────────────────────────

"""
    mesh_levelset(mesh::BoundaryMesh; lipschitz=nothing, orientation=:pseudonormal) -> LevelSet

Build a signed-distance [`LevelSet`](@ref) leaf from a closed boundary mesh — negative inside the solid — so
`physical_domain(mesh_levelset(mesh))` integrates the enclosed region and the leaf composes with the CSG combinators
(`intersect`/`union`/`setdiff`/`complement`). Two cases are supported:

  - `BoundaryMesh{2,T,1}` — a closed segment loop bounding a 2D region (build it with
    `polyline_mesh(verts; closed=true)` or `segment_mesh`),
  - `BoundaryMesh{3,T,2}` — a triangle surface bounding a 3D solid (`triangle_mesh`, or [`stl_levelset`](@ref) from a
    file).

The mesh must be watertight and consistently outward-oriented; with the default geometric normals that means
counter-clockwise segment loops in 2D and right-hand-rule triangle winding in 3D (the conventions of `BoundaryMesh`'s
`_default_normal`). Degenerate (zero-measure) cells are dropped, and under `:pseudonormal` the surviving cells are then
checked to be closed (every cell face traversed once in each direction) and to enclose a positive volume — a hole, a
flipped facet or a reversed winding is rejected rather than answered with an inverted sign. Faces are paired through
corners welded within 10⁻⁶ of the bounding-box diagonal (a binary STL stores each facet's corners separately, in
`Float32`). At that scale the check cannot tell a sliver of a closed surface from a dangling one, so a cell with two
corners that close is rejected too, even in an otherwise sound mesh. So are two cells on the same welded corners — a
sheet listed in both windings, or a face shared by two touching solids — which pass both checks while bounding nothing.
The checks do not detect bodies that overlap, a two-sided sheet or a shared face triangulated differently on its two
sides, nor a separate component wound inside-out beside a larger outward one. The default mode inverts the sign near
such faces, so merge touching or overlapping bodies into the boundary of their union and orient every component
outward.

`orientation` selects the inside/outside test:

  - `:pseudonormal` (default) — angle-weighted pseudonormal of the closest feature (Bærentzen & Aanæs 2005). Fast
    (O(candidates) per query), exact for a clean mesh including reflex features, down to the length at which it tells
    features apart, 10⁻⁸ of the bounding-box diagonal: next to a ridge too sharp for it to resolve, the sign can be
    wrong at points up to some tens of such lengths from the surface, which `:winding` signs correctly.
  - `:winding` — generalized winding number (Jacobson et al. 2013). Robust on imperfect / non-watertight / globally
    flipped meshes, at O(cells) per query, and the only mode that accepts a mesh failing the checks above.

`lipschitz` defaults to the exact constant of what the mode returns: `1.0` under `:pseudonormal`, where the result is a
true signed distance and the cell classifier's certificate is tight, and `Inf` under `:winding`, where a gap in the mesh
puts a jump of the sign at a positive distance from any cell and no finite constant bounds it. Pass `lipschitz = 1.0`
explicitly to keep the tight certificate when `:winding` runs on a watertight mesh.

!!! note
    `triangle_mesh` is exported by both Unfitted and GeometryBasics. Loading the extension via
    `using Unfitted, FileIO, MeshIO` keeps Unfitted's in scope; do not also `using GeometryBasics`, which would shadow
    it.

Requires `using MeshIO`, which loads the package extension that implements it; a call made without it raises an
error that says so.
"""
mesh_levelset(args...; kwargs...) = _extension_missing(mesh_levelset, args, "MeshIO")

"""
    stl_levelset(path; lipschitz=nothing, orientation=:pseudonormal) -> LevelSet

Read an STL file (ASCII or binary) at `path` and return a signed-distance [`LevelSet`](@ref) leaf for the meshed solid:
it loads the triangles into a `BoundaryMesh{3,T,2}` (`triangle_mesh`) and calls [`mesh_levelset`](@ref).

    using Unfitted, FileIO, MeshIO   # GeometryBasics loads transitively
    Ω = physical_domain(stl_levelset("part.stl"); subcell_length_scale=h)

`lipschitz` and `orientation` are passed through to [`mesh_levelset`](@ref), including its watertightness and
outward-orientation checks — an STL with holes or mis-wound facets is rejected under the default `:pseudonormal`, and
`orientation = :winding` is the robust inside/outside test that accepts it. Accuracy is bounded by the STL faceting:
each planar facet is integrated exactly, while facet edges are creases resolved by the kernel's subdivision.

Requires `using MeshIO`, which loads the package extension that implements it; a call made without it raises an
error that says so.
"""
stl_levelset(args...; kwargs...) = _extension_missing(stl_levelset, args, "MeshIO")

# ── VTK output: the WriteVTK extension ────────────────────────────────────────
#
# ParaView output is the package extension `ext/UnfittedWriteVTKExt.jl`, installed when WriteVTK.jl is loaded alongside
# Unfitted. It serialises the consumer-neutral records `src/postprocessing.jl` samples and decides everything
# VTK-specific there: cell types, padding, number types, the multiblock layout and the file names. Its entry points
# follow the rule above. The series handle is the abstract `VTKSeries`, because its concrete type holds a WriteVTK
# collection and is therefore the extension's; the two series operations are documented below on their abstract
# signatures.

"""
    VTKSeries

Handle for a ParaView collection (`.pvd`) being written frame by frame. Opened by [`vtk_series`](@ref), extended by
`write_vtk(series, time, solution, model)`, and finished by `close(series)`; a series comes only from `vtk_series`.

`VTKSeries` is abstract. The concrete handle, which holds the open collection, the directory its frame bundles are
written into and the number of frames written so far, is defined by the WriteVTK package extension that
`using WriteVTK` loads; dispatch on, and test against, `VTKSeries`.
"""
abstract type VTKSeries end

"""
    write_vtk(path, solution, model;
              field=nothing, subdivisions=:degree, cut_depth=3, point_data, cell_data,
              level_meshes=true, ascii=false, append=true, compress=true) -> Vector{String}

Write a ParaView bundle rooted at `path`. The top-level file is a `.vtm` multiblock dataset with **one block per
field**, each named after the field and grouping that field's data with its own mesh as sibling leaf datasets so it is a
self-contained, independently-toggled unit — `<field> → { data_<i>, level_… }`:

  - `data_<i>` — the `i`-th field's partitioned solution `.vtu`, whose cells are that field's admissible integration
    regions optionally subdivided into smaller sub-cells (see `subdivisions`), with per-cell `region_id` /
    `cover_count`, per-point user-defined arrays, and (with a [`PhysicalDomain`](@ref)) the field's own `level_set`. A
    write of one field — a single-field model, or the one field picked with `field` — names it simply `data`;
  - `level_<id>_<role>` — optionally, one solid-cell `.vtu` per level of the field's subdomain: every mesh cell with
    per-cell `active`, `covered`, `active_dofs`, `reduced_dofs` (plus `level_id` / `role_id` / `order_max` / `cell_id`),
    so ParaView can colour the mesh by coverage and covered-mode pruning.

A single-field model is the one-block case of this shape. Fields sharing one space emit that space's meshes once, under
the first such field, and count their `active_dofs` / `reduced_dofs` on that field's dof layout — the convention
[`diagnostics`](@ref) follows for its per-level counts — so with several fields over one space the counts reflect the
first field's Dirichlet conditions alone. The tree is homogeneous (a field block holds only leaf datasets) and every
leaf's disambiguating name is **ASCII** — the field *index* `i` for `data_<i>`, the level *id* for the meshes — so
ParaView's Extract Block (whose data-assembly node names keep only ASCII identifier characters) resolves each leaf on
its own even when the field names themselves are Unicode (`θ₁`, `θ₂`).

$(_SAMPLING_KEYWORDS_DOC)
The writer's own keywords:

  - `level_meshes` — emit one solid-cell `.vtu` per level when `true` (default), each cell carrying per-cell `active` /
    `covered` / `active_dofs` / `reduced_dofs` data.
  - `ascii` / `append` / `compress` — pass-through to `WriteVTK`'s `vtk_grid` constructor, with its own defaults:
    binary, appended and **compressed**. Compression is the cheapest lever this writer has — measured on a 3D 5×5×5
    order-2 sphere at `subdivisions = 10`, it takes the bundle from 31.60 MB to 3.81 MB, an 8.3× saving for 1.7× the
    write time — and it costs the reader nothing, since ParaView decompresses transparently. Turn it off together with
    `ascii = true, append = false` when a test or a diff needs the arrays as parsable text.

VTK stores a value in one of its own number types, so the values of a `point_data` or `cell_data` entry must also
promote to `Float32`, `Float64` or an integer of at most 64 bits — or to `Bool`, which VTK has no type for and which is
written as `UInt8` 0 / 1. An entry whose values promote to another real type (a `Rational`, a `BigFloat`) is refused
with an `ArgumentError` naming it, and nothing is written; a `Rational` among `Float64` values promotes to `Float64` and
is accepted. A vector value is written as a VTK vector, a two-component one with a trailing zero as its third.

Returns the path of every file written: the `.vtm` first, then the datasets in block order. The bundle is built in
memory and written, its directory included, only once it is complete: a call that fails — on a keyword it refuses, or in
a callback that throws — leaves nothing on disk.

Requires `using WriteVTK`, which loads the package extension the VTK writers live in. A call made without it raises an
error that says so, and writes nothing.

$(_IMMERSED_BOUNDARY_DOC)
To write a *sequence* of these bundles as one animation — a moving overlay, an adaptive ladder, a time-dependent
solution — open a [`vtk_series`](@ref) and call this function with it in place of a path.
"""
write_vtk(args...; kwargs...) = _extension_missing(write_vtk, args, "WriteVTK")

"""
    vtk_series(path; frames=nothing) -> VTKSeries

Open a ParaView collection at `path` for a sequence of solution bundles — the transient sibling of [`write_vtk`](@ref).
Write one frame per call, then `close` the series to write the collection file:

```julia
series = vtk_series("output/laser")
for step in 1:nsteps
    t, u, model = advance!(…)
    write_vtk(series, t, u, model)
end
close(series)                                  # → "output/laser.pvd"
```

Opening `laser.pvd` in ParaView plays the sequence: the time slider carries the times the frames were written at, and
the animation shows the solution *and* the mesh that produced it.

  - `path` — the collection file, with or without the `.pvd` extension.
  - `frames` — the directory the frame bundles go into. Defaults to `<path>_frames`, a sibling of the collection. The
    collection references each frame by a path relative to itself, so collection and frames move together as one
    directory tree.

Four properties are worth stating explicitly, because they are what makes this usable for a moving feature rather than
merely for a fixed mesh in time:

  - **Every frame is a full bundle** — the data grid *and* the per-level meshes — written under its own base name
    `frame_<n>`, with `n` counted from one and zero-padded so a directory listing sorts. On a moving overlay or an
    adaptive ladder the mesh differs from frame to frame, and that is usually the thing one wants to see; a series that
    emitted the mesh once could not show it.
  - **Keywords are per frame.** Every keyword of [`write_vtk`](@ref) is accepted on each call, with the same meaning and
    the same default, because both the sampled data and the model change as a run proceeds.
  - **The collection records each frame's own physical time**, exactly as it is given and in the order given, so a run
    with an adaptive Δt is never forced onto a uniform frame grid.
  - **Nothing is deleted.** A re-run that writes fewer frames than the previous one leaves the surplus frame files in
    place, unreferenced by the new collection — invisible in ParaView, visible in a directory listing. Clearing the
    frames directory first, if that matters, is the caller's decision; a library that removed files a user might want is
    not one.

The series deliberately has no schedule of its own — no `every`, no `times`. Which steps of a time loop deserve a
picture is a property of the loop, not of the writer, and a frame costs what a `write_vtk` call costs.

Requires `using WriteVTK`, like [`write_vtk`](@ref).
"""
vtk_series(args...; kwargs...) = _extension_missing(vtk_series, args, "WriteVTK")

"""
    write_vtk(series::VTKSeries, time, solution, model; kwargs...) -> String

Write one frame of a [`vtk_series`](@ref) and register it in the collection at physical `time`. Returns the path of the
frame's `.vtm`.

The frame is exactly the bundle `write_vtk(path, solution, model; …)` writes, under a name the series chooses
(`frame_<n>` in the series' frames directory), and every keyword of that method is accepted here unchanged — see
[`vtk_series`](@ref) for what the series does and does not do with the frames it is handed. `time` is recorded in the
collection and is otherwise unused: it need not be a step index, a multiple of anything, or evenly spaced. A frame that
fails writes nothing and does not use up a frame number. A series that has been closed takes no more frames: a frame
written to it raises an `ArgumentError`, before any sampling.
"""
write_vtk(series::VTKSeries, time, solution, model)

"""
    close(series::VTKSeries) -> String

Write the ParaView collection file of `series` and return its path. The frame bundles are already on disk — each is
written as it is handed over — so this writes only the `.pvd` index over them. Calling it a second time writes nothing
and returns the same path; a frame written after it is refused.
"""
Base.close(series::VTKSeries)

"""
    write_quadrature_vtm(path, model) -> Vector{String}

Write a ParaView multiblock bundle of the model's quadrature-point cloud. One `.vtp` block is emitted per *distinct
combination of covering parent levels*: every region of every subdomain integration plan is keyed by
`sort([parent.level for parent in region.parents])` and regions with the same level signature land in the same block.
Level ids are unique across subdomains, so a coupled model's subdomains land in different blocks and all of them are
written. Each block stores the physical quadrature-point coordinates and per-point physical weight
(`weight = w × jacobian`) as point data.

Useful for inspecting where the integration regions actually land in ParaView (the regions / cuts / α-FCM weights are
not visible from the solution VTK alone) and for diagnosing small-overlap warnings — the small regions show up as point
clusters with tiny weights. Returns the path of every file written, the `.vtm` first.

Requires `using WriteVTK`, like [`write_vtk`](@ref).
"""
write_quadrature_vtm(args...; kwargs...) = _extension_missing(write_quadrature_vtm, args, "WriteVTK")

# ── Plots: the Makie extension ────────────────────────────────────────────────
#
# Plotting is the package extension `ext/UnfittedMakieExt.jl`, installed when Makie.jl is loaded alongside Unfitted —
# by any of its backends, CairoMakie, GLMakie or WGLMakie, each of which loads Makie. It draws the consumer-neutral
# records `src/postprocessing.jl` samples, which the VTK writer serialises, and decides everything Makie-specific
# there. Its entry points are Makie recipes, so they are declared methodless, as the rule at the top of this file says.

"""
    levelplot(model; by=:order_max, levels=nothing, inactive=false, field=nothing,
              strokecolor=:black, strokewidth=0.5, linewidth=6, categorical=automatic, colormap, colorrange, …)
    levelplot!([axis,] model; …)

Plot the meshes of `model`'s levels, each cell coloured flat by one per-cell quantity: the discretisation itself, which
needs no solution — which cells each level has and which of them are active, the polynomial order of every cell, where
covered-mode pruning removes modes, or where an error estimate puts the error. `levelplot` returns what every Makie
plotting function returns, the figure, the axis and the plot; `levelplot!` draws into an existing axis and returns the
plot.

The cells drawn are the active cells of every drawn level — every cell of it with `inactive = true` — in
`cell_indices` order, the cells of the level meshes [`write_vtk`](@ref) writes as `level_<id>_<role>`, each with its
edges:

  - **2D.** Each level is a mesh of quads, the levels stacked in level order, the base lowest and each later level above
    the ones before it, so that wherever a later level has a drawn cell it hides the levels below, and the plot shows
    at every point the last drawn level that has a cell there, edges included: an edge of a level is drawn only where
    no drawn cell of a later level lies over it, and an edge two cells of one level share is drawn once. The axis keeps
    the data's aspect ratio. To see one level whole, its cells under the overlays included, pass its id as `levels`.
  - **1D.** Each level is a row of segments at the height of its level id, the base at the bottom, with a tick at every
    cell end, so that the levels read as a ladder over the domain.

Keyword arguments, which are the plot's attributes:

  - `by` — what colours a cell. Either the name of one of the level meshes' integer cell arrays — `:order_max`
    (default), the largest per-axis polynomial order of the cell; `:level_id`; `:role_id`, 0 on the base level and 1
    on an overlay; `:cell_id`, the cell's linear index in its level; `:active`, 1 where the level's activation mask and
    the fictitious fold keep the cell; `:covered`, 1 where the active region of a single higher-id level holds the
    cell's material; `:active_dofs`, the cell's modes with an active component; `:reduced_dofs`, its modes that
    covered-mode pruning eliminated — or an [`ErrorEstimate`](@ref) of `model`, whose indicator `η_K` colours each cell
    `K`. An estimate must be the one [`estimate`](@ref) returned for this model's discretisation, one array per level of
    that level's shape, which is checked.
  - `levels` — the levels to draw, by id: `nothing` (default) for every level, one id, or a collection of ids.
  - `inactive` — whether to draw the cells that a level's activation mask or the fictitious fold switched off; `false`
    by default, since they carry no approximation, and drawn above a coarser level they would hide its active cells.
    `by = :active` with `inactive = true` and one level shows that level's mask.
  - `field` — the field whose space is drawn on a model with several, a [`Field`](@ref) or its name; `nothing`
    (default) draws the space of every field, each space once.
  - `strokecolor`, `strokewidth` — the colour and the width in pixels of the cell edges and, in 1D, of the ticks at the
    cell ends; `strokewidth = 0` draws neither.
  - `linewidth` — in 1D, the width in pixels of the segments that carry the colour.
  - `categorical` — whether the colours are categories: each distinct value its own colour of `colormap`, and a
    `Colorbar` one block and one tick per value. `automatic` (default) makes the quantities with a few discrete values
    categorical — `:order_max`, `:level_id`, `:role_id`, `:active` and `:covered` — and not the dof counts, the cell
    index or an error estimate; `true` or `false` decides for any `by`. It is settled when the plot is created: a drawn
    plot keeps it when `by` or `colormap` changes, since Makie cannot switch a drawn plot between categorical and
    continuous colours.
  - `colormap`, `colorrange`, `colorscale`, `lowclip`, `highclip`, `nan_color`, `alpha` — the colour mapping, as for
    every Makie plot; a `Colorbar` reads it off the plot.

The per-cell arrays mean what they mean on `write_vtk`'s level meshes, including the limit of the `covered` flag: it is
computed only where the dof layer reads it, so on a masked level an inactive cell far from the active ones shows `0`
whatever its true coverage. `:covered` and `:reduced_dofs` are read on one level at a time, `levels = id`: the cells
they mark lie under the level that covers them, which the stacked levels draw in front.

A `by`, `levels` or estimate the plot cannot use is refused with an error that names the problem; Makie raises it
wrapped in its own exception, since it arises while the plot computes what to draw, and does so again when the
attribute is changed on a drawn plot. A model of three or more dimensions is refused with an `ArgumentError`: write it
with [`write_vtk`](@ref), whose level meshes carry the same arrays, and look at it in ParaView.

Requires Makie, which each of its backends loads: `using CairoMakie` to draw into files, `GLMakie` for an interactive
window, `WGLMakie` for a browser or a notebook. Without it a call raises a `MethodError` that says so.
"""
function levelplot end

"""
    levelplot!([axis,] model; …)

Draw [`levelplot`](@ref) into `axis`, or into the current axis, and return the plot. The attributes are `levelplot`'s.
"""
function levelplot! end

"""
    quadratureplot(model; by=:weight, marker=:circle, markersize=6, categorical=automatic, colormap, colorrange, …)
    quadratureplot!([axis,] model; …)

Plot `model`'s quadrature points: every point of every subdomain's integration plan, where its integration regions put
it and what it weighs — the cloud [`write_quadrature_vtm`](@ref) writes. It shows what the level meshes cannot: the
integration regions themselves, the moment-fitted rules of the cut cells, and, by their tiny weights, the small overlaps
that condition the system badly. `quadratureplot` returns the figure, the axis and the plot; `quadratureplot!` draws
into an existing axis and returns the plot.

  - **2D.** Each point is a marker at its position. The axis keeps the data's aspect ratio.
  - **1D.** Each point is a marker at its position against its weight, so that the rule of every region reads as a
    profile.

Keyword arguments, which are the plot's attributes:

  - `by` — what colours a point: `:weight` (default), its physical weight, the rule weight times the region's Jacobian
    as assembly integrates with it; or `:cover_count`, the number of levels whose cells cover its region — 1 where the
    base level alone is active, more under an overlay.
  - `marker`, `markersize` — the marker of a point and its size in pixels.
  - `categorical` — whether the colours are categories, as for [`levelplot`](@ref): `automatic` (default) for
    `:cover_count`, not for `:weight`.
  - `colormap`, `colorrange`, `colorscale`, `lowclip`, `highclip`, `nan_color`, `alpha` — the colour mapping, as for
    every Makie plot; a `Colorbar` reads it off the plot.

Any other `by` is refused with an error that names it, wrapped by Makie as for [`levelplot`](@ref). A model of three
or more dimensions is refused with an `ArgumentError`: write its cloud with [`write_quadrature_vtm`](@ref) and look at
it in ParaView.

Requires Makie, which each of its backends loads, as for [`levelplot`](@ref).
"""
function quadratureplot end

"""
    quadratureplot!([axis,] model; …)

Draw [`quadratureplot`](@ref) into `axis`, or into the current axis, and return the plot. The attributes are
`quadratureplot`'s.
"""
function quadratureplot! end

# ── The fallback and the hint every entry point shares ────────────────────────

# The remedy for a call made without `package`'s extension, as the fallback and the hint both state it. Makie is the
# exception to `using` the package: drawing needs a backend, and every backend loads Makie, so the remedy names one.
function _how_to_load(package::String)
    package == "Makie" && return "run `using CairoMakie`, or another Makie backend, to load it"
    return "run `using $package` to load it"
end

# The body of an extension stub's fallback method `f(args...; kwargs...)`: what a call no method of `f` accepts raises.
# While the package extension `Unfitted<package>Ext` — the name every extension of this package carries, after the
# package that loads it — is not loaded, the error names that package, which is the remedy, and it is raised before the
# call has touched anything. Once the extension is loaded the call simply matched none of its methods, and the ordinary
# `MethodError` says so (without the keyword arguments, which it does not carry) rather than asking the user to load
# what is loaded. Generic in the package, so every extension's entry points share it.
function _extension_missing(f, args, package::String)
    Base.get_extension(@__MODULE__, Symbol(:Unfitted, package, :Ext)) === nothing &&
        error("$(nameof(f)) is provided by Unfitted's $package extension: $(_how_to_load(package))")
    throw(MethodError(f, args))
end

# The methodless entry points, each with the package whose extension implements it (see the top of this file). A
# recipe added to an extension is declared above and added here.
const _METHODLESS_ENTRY_POINTS = (levelplot => "Makie", levelplot! => "Makie", quadratureplot => "Makie",
                                  quadratureplot! => "Makie")

# The error hint `__init__` (in `Unfitted.jl`) registers for `MethodError`: for a call to a methodless entry point made
# while its extension is not loaded, the line that names the package to load, as `_extension_missing` names it for a
# fallback. It says nothing about any other `MethodError`, nor once the extension is loaded, when the call failed for
# another reason. A call with keyword arguments fails as a call of `Core.kwcall`, whose second argument is the function
# that was called, so that is the function looked up.
function _extension_hint(io::IO, err::MethodError, _, _)
    f = err.f === Core.kwcall && length(err.args) >= 2 ? err.args[2] : err.f
    i = findfirst(entry -> first(entry) === f, _METHODLESS_ENTRY_POINTS)
    i === nothing && return nothing
    package = last(_METHODLESS_ENTRY_POINTS[i])
    Base.get_extension(@__MODULE__, Symbol(:Unfitted, package, :Ext)) === nothing || return nothing
    print(io, "\n", nameof(f), " is provided by Unfitted's ", package, " extension: ", _how_to_load(package))
    return nothing
end
