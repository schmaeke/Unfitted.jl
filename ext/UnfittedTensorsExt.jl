"""
    UnfittedTensorsExt

Package extension that lets weak-form callbacks read trial-side data and
return test-side coefficients in [`Tensors.jl`](https://github.com/Ferrite-FEM/Tensors.jl)
notation while preserving the `SVector{D,T}`-based storage that the
assembly hot loop already uses.

The extension is purely additive — it does not change any existing
behaviour. Loading `Tensors` next to `Unfitted` installs the methods
below; without `Tensors`, the package functions exactly as before.

Provided pieces:

  - `Tensors.Vec(::TrialChannels)` — the trial-basis gradient channel as
    a `Vec{D,T}`. Both `SVector` and `Vec` wrap `NTuple{D,T}` underneath
    so the conversion is a tuple repack that the Julia compiler folds
    away.
  - `Unfitted.symmetric_gradient(::TrialChannels)` — the symmetric
    gradient `ε(N(x) eₖ)` of a single vector-field trial basis function
    as a `SymmetricTensor{2,D,T}`. Lets a vector-field bilinear callback
    read `ε(u_trial) = symmetric_gradient(trial)` and form the bilinear
    integrand `σ : ε(v)` against any user constitutive law. The
    two-argument forms `symmetric_gradient(::TrialChannels, ::Val{C})` and
    `symmetric_gradient(::TrialChannels, ::Field)` size the strain by the
    field's component count `C` instead of by the mesh dimension `D`,
    which is what a field with `C < D` components needs — a displacement
    on a space-time mesh, whose last axis is time.
  - `Unfitted.TestChannels(::Real, ::Vec)` and
    `Unfitted.TestChannels(::Real, ::SecondOrderTensor, ::Integer)` —
    return-side constructors taking either a Tensors.jl `Vec` gradient
    coefficient or the stress tensor whose row the test component
    selects. Both unpack into the existing `SVector`-typed channel
    without allocating.
  - `Unfitted.value_vec` / `Unfitted.gradient_tensor` — read the value
    and physical Jacobian of a vector field at the current quadrature
    point as a `Vec{D,T}` and a `Tensor{2,D,T}`. Replaces hand-rolled
    Voigt packing in mechanics and constitutive callbacks.

All methods are written generically in the spatial dimension `D ≥ 1`
to match the package's dimension-independence contract; tests cover at
least 2D and 3D. A field's component count `C` is an independent
parameter, not a synonym for `D`: `value_vec` and the two-argument
`symmetric_gradient` are sized by `C`, and only `gradient_tensor`, whose
square result needs the two to agree, requires `C = D`.

The Unfitted-side field-gradient reader exported by the package is
[`field_gradient`](@ref) (rather than the natural shorter `gradient`)
specifically to keep this combination unrestricted: `Tensors.gradient`
is the AD entry point of Tensors.jl, and the two would collide under
the bare name. With the longer name in place, `using Unfitted; using
Tensors` brings both into scope without ambiguity.
"""
module UnfittedTensorsExt

using Unfitted
using Unfitted: TrialChannels, TestChannels, Field
using StaticArrays: SVector
using Tensors

# ── Trial-side: TrialChannels → Tensors.jl objects ────────────────────────────

"""
    Tensors.Vec(trial::TrialChannels{D,T}) -> Vec{D,T}

Read the physical gradient channel of a trial basis function as a
`Vec{D,T}`. `SVector{D,T}` and `Vec{D,T}` are both `NTuple{D,T}`
underneath, so the conversion is a tuple repack that LLVM resolves to
an identity store; there is no allocation and no extra arithmetic.
"""
@inline Tensors.Vec(t::TrialChannels{D,T}) where {D,T} = Vec{D,T}(Tuple(t.gradient))

"""
    symmetric_gradient(trial::TrialChannels{D,T}) -> SymmetricTensor{2,D,T}

Rank-1 symmetric gradient

    ε(N(x) eₖ) = ½ (eₖ ⊗ ∇N + ∇N ⊗ eₖ),    k = trial.component,

of the single vector-field trial basis function currently being
assembled, returned as a `SymmetricTensor{2,D,T}`. Allows a bilinear
callback to read

    ε_trial = symmetric_gradient(trial)
    σ_trial = ℂ ⊡ ε_trial          # any constitutive law
    return TestChannels(0.0, σ_trial, test_component)

without hand-rolling the Voigt ↔ tensor conversion or the off-diagonal
factor-of-two bookkeeping. The minor symmetry of the strain tensor is
preserved by the `SymmetricTensor{2,D}` storage so contractions cost
`D(D+1)/2` multiplies rather than `D²`.

`trial.component` is the component index of the trial basis function
currently being assembled, exactly as documented on
[`TrialChannels`](@ref). Reading it as the spatial index `k` of `eₖ`
assumes a field with one component per spatial axis (`C = D`), which is
what a strain tensor means. Neither mismatch is detected here, and the two
fail in opposite ways:

  - At `C > D` a component index above `D` cannot appear in a
    `SymmetricTensor{2,D}`, so the result is the zero tensor.
  - At `C < D` the result is a `D × D` tensor whose surplus rows and
    columns carry derivatives along axes the field has no component for.
    The space-time case is the one that bites: on a mesh whose last axis is
    time, a displacement field with one component per *spatial* axis has
    `C = D − 1`, and the surplus slots hold `∂ₜ`. Nothing downstream can
    tell — the operator stays symmetric and positive semi-definite, it
    solves, it converges under refinement — yet under an isotropic law it
    carries exactly the spurious `μ ∫ ∂ₜu · ∂ₜv` those slots describe, an
    identity `test/test_tensors_ext.jl` pins.

Pass the component count — `symmetric_gradient(trial, Val(C))` or
`symmetric_gradient(trial, field)` — to get the `C × C` strain the field
actually has. The two return the same tensor and differ only in what they
cost the closure that calls them, which the `::Field` method's docstring
measures.
"""
@inline function Unfitted.symmetric_gradient(t::TrialChannels{D,T}) where {D,T}
    g = Vec(t)
    c = t.component
    return SymmetricTensor{2,D,T}() do i, j
        T(((i == c) * g[j] + (j == c) * g[i]) / 2)
    end
end

"""
    symmetric_gradient(trial::TrialChannels{D,T}, ::Val{C}) -> SymmetricTensor{2,C,T}

Rank-1 symmetric gradient of the trial basis function currently being
assembled, sized by the **component count** `C` of the field it belongs to
rather than by the mesh dimension `D`:

    ε(N(x) eₖ)ᵢⱼ = ½ (δᵢₖ ∂ⱼN + δⱼₖ ∂ᵢN),    i, j ∈ 1:C,    k = trial.component.

This is the form to use whenever a field has fewer components than the mesh
has axes — a space-time mesh, where the last axis is time and the
displacement field carries one component per spatial axis, being the case
that motivates it. The one-argument [`symmetric_gradient`](@ref) sizes the
strain by `D` there and silently adds the time-derivative slots to the
operator.

The result is a genuine `C × C` tensor and not a `D × D` one with the
surplus slots zeroed, because a padded strain lies about its own size:
`dev(ε)` would subtract `tr(ε)/D` instead of `tr(ε)/C`, every invariant
would be formed in the wrong space, and an anisotropic `ℂ` would contract
against a slot the material does not have. Returning the smaller tensor is
therefore not an optimisation but the only honest answer, and the
constitutive law that consumes it is written over `C` axes too:

    ε_trial = symmetric_gradient(trial, Val(2))   # on a 3-axis space-time mesh
    σ_trial = ℂ ⊡ ε_trial                         # ℂ::SymmetricTensor{4,2}
    return TestChannels(0.0, σ_trial, test_component)

The `C`-length gradient coefficient that comes back from the three-argument
[`TestChannels`](@ref) constructor is zero-extended to the mesh dimension by
assembly, which is what makes the return path above work unchanged.

`C > D` raises a `DimensionMismatch`: a field with more components than the
mesh has axes has no strain tensor, and the missing derivatives cannot be
invented.

Of the two two-argument forms this is the one with no capture cost: `Val(C)`
is a singleton, whereas a captured [`Field`](@ref) carries its whole `Space`
into the closure and costs a heap box per call. Prefer it in a form written
inside a function; the `::Field` method's docstring has the measurement.

**This overload is transitional.** [`TrialChannels`](@ref) does not yet
carry the component count as a type parameter; once it does, the
one-argument `symmetric_gradient(trial)` is correctly sized on its own and
both two-argument forms become redundant.
"""
@inline function Unfitted.symmetric_gradient(t::TrialChannels{D,T}, ::Val{C}) where {D,T,C}
    C ≤ D || throw(DimensionMismatch("a field of $C components has no strain tensor on a " *
                                     "$D-dimensional mesh"))
    g = Vec(t)
    c = t.component
    return SymmetricTensor{2,C,T}() do i, j
        T(((i == c) * g[j] + (j == c) * g[i]) / 2)
    end
end

# The field's own dimension is deliberately not tied to the trial channels'
# `D`: the mesh dimension is checked inside the `Val` form above, and tying it
# here would replace that `DimensionMismatch` with a bare `MethodError` naming
# two type parameters. `C` comes from the type rather than from
# `component_count(field)` so the returned tensor's size is known at compile
# time and the strain stays allocation-free in the assembly hot loop.
#
# The block sits above the docstring rather than between it and the definition:
# a comment there detaches the docstring silently, and the method ends up
# undocumented (verified on Julia 1.12).
"""
    symmetric_gradient(trial::TrialChannels, field::Field) -> SymmetricTensor{2,C}

Sugar for `symmetric_gradient(trial, Val(C))` with `C` the component count
of `field`, read off its type, so a weak form written against a named
[`Field`](@ref) never spells its own component count twice:

    bilinear(q, trial, c) = TestChannels(0.0, ℂ ⊡ symmetric_gradient(trial, u), c)

Identical to the `Val(C)` form in every respect, including the
`C > D` `DimensionMismatch` — see [`symmetric_gradient`](@ref) for the
convention, for why the result is `C × C`, and for the note that both
two-argument forms are transitional.

How the field reaches the callback decides what it costs. The callback runs
once per test component per trial dof per quadrature point, so every figure
below is per call, measured on Julia 1.12:

  - **`const` global, or `Val(C)`** — 0 B, and a concrete `TestChannels`.
    Bind the field `const` when the closure is written at top level, as an
    example script does, or pass `Val(C)` there instead.
  - **Non-`const` global** — read as `Any` on every call: 240 B, with the
    return type inferred as `Any`. The cost is the global read and not the
    field argument — a non-`const` global `ℂ` costs 192 B the same way — but
    this is the form that puts a field read inside the hot loop.
  - **Captured local** — a form written *inside a function* genuinely
    captures the field by value, and the field brings the whole
    [`Space`](@ref) with it: `sizeof(Field) = 248 B`, of which 240 B is the
    space, so the closure grows to 320 B and the enclosing `WeakForm` to
    328 B, against 72 B and 80 B for a captured `Val(C)`. Size is not the
    trigger, though; losing `isbits` is. A `Space` is not an `isbits` type,
    so neither is a `WeakForm` holding one, and once a problem's blocks stop
    sharing a single form type the per-quadrature-point block walk has to
    heap-box that form at every callback call — 336 B a box after GC pool
    rounding. On an 8-region, 216-quadrature-point space-time problem whose
    displacement block makes 216 × 108 = 23 328 such calls, that is 23 328
    boxes and 7.84 MB per assembly pass, while the `Val(C)` capture, still
    `isbits`, boxes nothing at all.

The shipped `examples/applications/spacetime_cavity_2d` pays none of this:
its `u` is a `const` global and its bilinear forms are named top-level
methods, so nothing is captured and each `WeakForm` is 2 B.
"""
@inline function Unfitted.symmetric_gradient(t::TrialChannels, ::Field{D,T,C}) where {D,T,C}
    return symmetric_gradient(t, Val(C))
end

# ── Test-side: TestChannels constructor accepting a Vec coefficient ───────────

"""
    TestChannels(value::Real, gradient::Vec{D,T}) -> TestChannels{D,T}

Constructor that accepts a Tensors.jl `Vec` for the gradient
coefficient, so a callback can return tensor algebra results directly:

    return TestChannels(0.0, σ ⋅ n)          # a traction, say

The conversion to the internal `SVector{D,T}` storage reads the `D`
coordinates straight into the existing `SVector` constructor and adds no
runtime cost beyond it. When the gradient coefficient is a row of a
stress tensor — the elasticity case — use the three-argument
constructor below instead of extracting the row by hand.
"""
# The coordinates are read through `getindex`, not `Tuple(g)`: `Base.Tuple`
# has no `Vec` method, so `Tuple(g)` falls through to the generic iterator
# constructor, whose result length is not statically known, and heap-allocates.
# This constructor is the return path of every tensor-notation bilinear
# callback — once per test component per trial dof per quadrature point — so it
# must stay allocation-free.
@inline function Unfitted.TestChannels(value::Real, g::Vec{D,T}) where {D,T}
    return TestChannels(value, SVector{D,T}(ntuple(i -> g[i], Val(D))))
end

"""
    TestChannels(value::Real, σ::SecondOrderTensor{D,T}, component::Integer) -> TestChannels{D,T}

Return-side constructor for a stress-like 2nd-order tensor, which is what
a constitutive law hands back. Assembly contracts the gradient channel
against the *scalar* test basis gradient of test component `c`, i.e.
against the test function `v = N eᶜ` with `∇v = eᶜ ⊗ ∇N`, so

    σ : ∇v = Σⱼ σ[c, j] ∂ⱼN = ⟨row c of σ, ∇N⟩,

and the gradient coefficient of `∫ σ : ∇v` is row `component` of `σ`:

    return TestChannels(0.0, ℂ ⊡ symmetric_gradient(trial), test_component)

Row, not column. The two agree for the symmetric `σ` of small-strain
elasticity, but they differ for a general 2nd-order tensor — a first
Piola–Kirchhoff stress, say — and the contraction asks for the row in
both cases.

`component` indexes a row of `σ`, so a test component above `D` raises a
`BoundsError`. That is the same `C = D` assumption
[`symmetric_gradient`](@ref) documents: a field carrying more components
than the space has axes has no stress tensor to take a row of.
"""
@inline function Unfitted.TestChannels(value::Real, σ::SecondOrderTensor{D,T},
                                       component::Integer) where {D,T}
    return TestChannels(value, SVector{D,T}(ntuple(j -> σ[component, j], Val(D))))
end

# ── State-side: FormState reads as tensor-valued quantities ───────────────────
#
# At a quadrature point the assembly path exposes `q.state::FormState`,
# whose `value(state, name, k)` and `field_gradient(state, name, k)` accessors
# read the current iterate's component-wise value and physical gradient.
# The wrappers below stack those scalar reads into a `Vec{D}` or a
# `Tensor{2,D}` at the call site so constitutive callbacks read in
# tensor notation. The `Val{D}` parameter is required so the return
# type is inferred at compile time and the small unrolled `ntuple`
# expansions stay allocation-free in the assembly hot loop.
#
# `FormState` evaluates every field once per point, so each accessor call
# is a field-name lookup and a buffer read; `gradient_tensor` still reads
# the `D` gradient rows once and then indexes them, rather than calling the
# accessor for each of the `D²` tensor slots.

"""
    value_vec(state::FormState, name::Symbol, ::Val{D}) -> Vec{D,T}

Read a vector field's value at the current quadrature point as a
`Vec{D,T}`. `D` must be passed via `Val(D)` (or equivalent) so the
return type is statically known. Replaces

    SVector(value(state, name, 1), …, value(state, name, D))

at the constitutive-law call site.
"""
@inline function Unfitted.value_vec(state, name::Symbol, ::Val{D}) where {D}
    return Vec{D}(ntuple(k -> Unfitted.value(state, name, k), Val(D)))
end

"""
    gradient_tensor(state::FormState, name::Symbol, ::Val{D}) -> Tensor{2,D,T}

Read the full Jacobian `[∂uᵢ/∂xⱼ]` of a vector field at the current
quadrature point as a 2nd-order tensor. The component layout is
`G[i, j] = ∂(component i)/∂xⱼ`, matching the Tensors.jl convention.
Combine with `Tensors.symmetric` to obtain the small-strain tensor

    ε(u) = symmetric(gradient_tensor(state, :u, Val(D)))

without ever materialising a Voigt 3-vector. As with [`value_vec`](@ref)
the dimension must be passed via `Val(D)` for type inference.

The single `Val` argument sizes **both** tensor indices — `Val(n)` reads `n`
gradient rows and returns an `n × n` tensor — so this is the field's
Jacobian only when the field has one component per axis (`C = D`). On a
field with `C ≠ D` neither argument gives one:

  - `Val(C)` returns the `C × C` block `[∂uᵢ/∂xⱼ]`, `i, j ∈ 1:C`. At `C < D`
    that is the Jacobian with its columns **truncated** to the first `C`
    axes, not the field's `C × D` Jacobian.
  - `Val(D)` raises a `BoundsError` at the first component the field does
    not carry.

A field with `C ≠ D` has no square Jacobian, and Tensors.jl has no
rectangular 2nd-order tensor to return it as, so there is nothing for this
function to widen to. Read such a field's gradient one component at a time
with `field_gradient(state, name, k)`, which returns the `k`-th row as an
`SVector{D}`.
"""
@inline function Unfitted.gradient_tensor(state, name::Symbol, ::Val{D}) where {D}
    rows = ntuple(i -> Unfitted.field_gradient(state, name, i), Val(D))
    return Tensor{2,D}() do i, j
        rows[i][j]
    end
end

# ── Field-typed overloads: infer dimensions from the field's type ────────────
#
# `Field{D,T,C,S}` carries the spatial dimension `D` and the component
# count `C` as type parameters. Passing the field object instead of the
# bare name lets the wrappers above infer those dimensions, dropping
# the `Val(D)` boilerplate at every call site.
#
# `value_vec` returns `Vec{C,T}` (the field has `C` components).
# `gradient_tensor` is well-defined as a square `Tensor{2,D,T}` only
# when `C = D` (the vector-field-in-physical-space case the example
# code uses); it dispatches on `Field{D,T,D}` to make the constraint
# explicit and falls through to a clear `MethodError` otherwise. There is
# no `Val`-explicit escape for `C ≠ D`: that argument sizes both tensor
# indices at once, so it can only ever return a square tensor, and a
# `C × D` Jacobian is not one. The `MethodError` is the honest answer —
# read those fields row by row with `field_gradient(state, name, k)`.

"""
    value_vec(state::FormState, field::Field{D,T,C}) -> Vec{C,T}

Read the field's value at the current quadrature point as a
`Vec{C,T}`. Equivalent to `value_vec(state, field.name, Val(C))` (see
[`value_vec`](@ref)) but infers the component count from the field's type
parameters.
"""
@inline function Unfitted.value_vec(state, field::Field{D,T,C}) where {D,T,C}
    return value_vec(state, field.name, Val(C))
end

"""
    gradient_tensor(state::FormState, field::Field{D,T,D}) -> Tensor{2,D,T}

Read the field's physical Jacobian at the current quadrature point as
a 2nd-order tensor. Defined for vector fields whose component count
equals the spatial dimension (`Field{D,T,D}`), which is the only case
that has a square Jacobian at all.

A field with `C ≠ D` components has a `C × D` Jacobian, and the
`Val`-explicit overload above cannot produce it either: its one argument
sizes both tensor indices, so `Val(C)` truncates the columns to the first
`C` axes and `Val(D)` raises a `BoundsError`. Read such a field's gradient
one component at a time with `field_gradient(state, name, k)` instead; the
`MethodError` this signature raises is that statement.
"""
@inline function Unfitted.gradient_tensor(state, field::Field{D,T,D}) where {D,T}
    return gradient_tensor(state, field.name, Val(D))
end

end # module UnfittedTensorsExt
