"""
    UnfittedTensorsExt

Package extension that lets weak-form callbacks read trial-side data and return test-side coefficients in
[`Tensors.jl`](https://github.com/Ferrite-FEM/Tensors.jl) notation while preserving the `SVector{D,T}`-based storage
that the assembly hot loop already uses.

The extension is purely additive — it does not change any existing behaviour. Loading `Tensors` next to `Unfitted`
installs the methods below, on names that `src/stubs.jl` declares and documents (the two `TestChannels` constructors
and `Vec(::TrialChannels)` are documented on `TestChannels` and `TrialChannels` in `src/problems.jl`); without
`Tensors`, a call of `symmetric_gradient`, `value_vec` or `gradient_tensor` raises an error that says to run
`using Tensors`, and the rest of the package functions exactly as before.

Provided pieces:

  - `Tensors.Vec(::TrialChannels)` — the trial-basis gradient channel as a `Vec{D,T}`. Both `SVector` and `Vec` wrap
    `NTuple{D,T}` underneath so the conversion is a tuple repack that the Julia compiler folds away.
  - `Unfitted.symmetric_gradient(::TrialChannels)` — the symmetric gradient `ε(N(x) eₖ)` of a single vector-field trial
    basis function as a `SymmetricTensor{2,D,T}`. Lets a vector-field bilinear callback read
    `ε(u_trial) = symmetric_gradient(trial)` and form the bilinear integrand `σ : ε(v)` against any user constitutive
    law. The two-argument forms `symmetric_gradient(::TrialChannels, ::Val{C})` and
    `symmetric_gradient(::TrialChannels, ::Field)` size the strain by the field's component count `C` instead of by the
    mesh dimension `D`, which is what a field with `C < D` components needs — a displacement on a space-time mesh, whose
    last axis is time.
  - `Unfitted.TestChannels(::Real, ::Vec)` and `Unfitted.TestChannels(::Real, ::SecondOrderTensor, ::Integer)` —
    return-side constructors taking either a Tensors.jl `Vec` gradient coefficient or the stress tensor whose row the
    test component selects. Both unpack into the existing `SVector`-typed channel without allocating.
  - `Unfitted.value_vec` / `Unfitted.gradient_tensor` — read the value and physical Jacobian of a vector field at the
    current quadrature point as a `Vec{D,T}` and a `Tensor{2,D,T}`. Replaces hand-rolled Voigt packing in mechanics and
    constitutive callbacks.

All methods are written generically in the spatial dimension `D ≥ 1` to match the package's dimension-independence
contract; tests cover at least 2D and 3D. A field's component count `C` is an independent parameter, not a synonym for
`D`: `value_vec` and the two-argument `symmetric_gradient` are sized by `C`, and only `gradient_tensor`, whose square
result needs the two to agree, requires `C = D`.

The Unfitted-side field-gradient reader exported by the package is [`field_gradient`](@ref) (rather than the natural
shorter `gradient`) specifically to keep this combination unrestricted: `Tensors.gradient` is the AD entry point of
Tensors.jl, and the two would collide under the bare name. With the longer name in place,
`using Unfitted; using Tensors` brings both into scope without ambiguity.
"""
module UnfittedTensorsExt

using Unfitted
using Unfitted: TrialChannels, TestChannels, Field
using StaticArrays: SVector
using Tensors

# ── Trial-side: TrialChannels → Tensors.jl objects ────────────────────────────

# `SVector{D,T}` and `Vec{D,T}` are both `NTuple{D,T}` underneath, so the conversion is a tuple repack that LLVM
# resolves to an identity store: no allocation and no extra arithmetic.
@inline Tensors.Vec(t::TrialChannels{D,T}) where {D,T} = Vec{D,T}(Tuple(t.gradient))

# The strain sized by the mesh dimension, which is the `Val` form below at `C = D`, where its `C ≤ D` check folds
# away.
@inline Unfitted.symmetric_gradient(t::TrialChannels{D}) where {D} = symmetric_gradient(t, Val(D))

# The strain sized by the field's component count `C` rather than by the mesh dimension: a genuine `C × C` tensor,
# and `C > D` has none.
@inline function Unfitted.symmetric_gradient(t::TrialChannels{D,T}, ::Val{C}) where {D,T,C}
    C ≤ D || throw(DimensionMismatch("a field of $C components has no strain tensor on a " * "$D-dimensional mesh"))
    g = Vec(t)
    c = t.component
    return SymmetricTensor{2,C,T}() do i, j
        T(((i == c) * g[j] + (j == c) * g[i]) / 2)
    end
end

# The field's own dimension is deliberately not tied to the trial channels' `D`: the mesh dimension is checked inside
# the `Val` form above, and tying it here would replace that `DimensionMismatch` with a bare `MethodError` naming two
# type parameters. `C` comes from the type rather than from `component_count(field)` so the returned tensor's size is
# known at compile time and the strain stays allocation-free in the assembly hot loop.
@inline function Unfitted.symmetric_gradient(t::TrialChannels, ::Field{D,T,C}) where {D,T,C}
    return symmetric_gradient(t, Val(C))
end

# ── Test-side: TestChannels constructor accepting a Vec coefficient ───────────

# The coordinates are read through `getindex`, not `Tuple(g)`: `Base.Tuple` has no `Vec` method, so `Tuple(g)` falls
# through to the generic iterator constructor, whose result length is not statically known, and heap-allocates. This
# constructor is the return path of every tensor-notation bilinear callback — once per test component per trial dof per
# quadrature point — so it must stay allocation-free.
@inline function Unfitted.TestChannels(value::Real, g::Vec{D,T}) where {D,T}
    return TestChannels(value, SVector{D,T}(ntuple(i -> g[i], Val(D))))
end

# Row `component` of `σ`, the gradient coefficient of `∫ σ : ∇v` for that test component, read slot by slot for the
# same reason.
@inline function Unfitted.TestChannels(value::Real, σ::SecondOrderTensor{D,T}, component::Integer) where {D,T}
    return TestChannels(value, SVector{D,T}(ntuple(j -> σ[component, j], Val(D))))
end

# ── State-side: FormState reads as tensor-valued quantities ───────────────────
#
# At a quadrature point the assembly path and `foreach_quadrature_point` expose `q.state::FormState`, whose
# `value(state, name, k)` and `field_gradient(state, name, k)` accessors read the current iterate's component-wise value
# and physical gradient.
# The wrappers below stack those scalar reads into a `Vec{D}` or a `Tensor{2,D}` at the call site so constitutive
# callbacks read in tensor notation. The `Val{D}` parameter is required so the return type is inferred at compile time
# and the small unrolled `ntuple` expansions stay allocation-free in the assembly hot loop.
#
# `FormState` evaluates every field once per point, so each accessor call is a field-name lookup and a buffer read;
# `gradient_tensor` still reads the `D` gradient rows once and then indexes them, rather than calling the accessor for
# each of the `D²` tensor slots.

@inline function Unfitted.value_vec(state, name::Symbol, ::Val{D}) where {D}
    return Vec{D}(ntuple(k -> Unfitted.value(state, name, k), Val(D)))
end

@inline function Unfitted.gradient_tensor(state, name::Symbol, ::Val{D}) where {D}
    rows = ntuple(i -> Unfitted.field_gradient(state, name, i), Val(D))
    return Tensor{2,D}() do i, j
        rows[i][j]
    end
end

# ── Field-typed overloads: infer dimensions from the field's type ────────────
#
# `Field{D,T,C,S}` carries the spatial dimension `D` and the component count `C` as type parameters. Passing the field
# object instead of the bare name lets the wrappers above infer those dimensions, dropping the `Val(D)` boilerplate at
# every call site.
#
# `value_vec` returns `Vec{C,T}` (the field has `C` components). `gradient_tensor` is well-defined as a square
# `Tensor{2,D,T}` only when `C = D` (the vector-field-in-physical-space case the example code uses); it dispatches on
# `Field{D,T,D}` to make the constraint explicit and falls through to a clear `MethodError` otherwise. There is no
# `Val`-explicit escape for `C ≠ D`: that argument sizes both tensor indices at once, so it can only ever return a
# square tensor, and a `C × D` Jacobian is not one. The `MethodError` is the honest answer — read those fields row by
# row with `field_gradient(state, name, k)`.

@inline function Unfitted.value_vec(state, field::Field{D,T,C}) where {D,T,C}
    return value_vec(state, field.name, Val(C))
end

@inline function Unfitted.gradient_tensor(state, field::Field{D,T,D}) where {D,T}
    return gradient_tensor(state, field.name, Val(D))
end

end # module UnfittedTensorsExt
