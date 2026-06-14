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
    integrand `σ : ε(v)` against any user constitutive law.
  - `Unfitted.TestChannels(::Real, ::Vec)` — return-side constructor that
    accepts a Tensors.jl `Vec` for the gradient coefficient. Again a
    zero-cost tuple repack into the existing `SVector`-typed channel.
  - `Unfitted.value_vec` / `Unfitted.gradient_tensor` — read the value
    and physical Jacobian of a vector field at the current quadrature
    point as a `Vec{D,T}` and a `Tensor{2,D,T}`. Replaces hand-rolled
    Voigt packing in mechanics and constitutive callbacks.

All methods are written generically in the spatial dimension `D ≥ 1`
to match the package's dimension-independence contract; tests cover at
least 2D and 3D.

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
    return TestChannels(0.0, σ_trial ⋅ basevec(Vec{D,T}, test_component))

without hand-rolling the Voigt ↔ tensor conversion or the off-diagonal
factor-of-two bookkeeping. The minor symmetry of the strain tensor is
preserved by the `SymmetricTensor{2,D}` storage so contractions cost
`D(D+1)/2` multiplies rather than `D²`.

`trial.component` is the spatial component index `1 ≤ k ≤ D` of the
trial basis function currently being assembled, exactly as documented
on [`TrialChannels`](@ref).
"""
@inline function Unfitted.symmetric_gradient(t::TrialChannels{D,T}) where {D,T}
    g = Vec(t)
    c = t.component
    return SymmetricTensor{2,D,T}() do i, j
        T(((i == c) * g[j] + (j == c) * g[i]) / 2)
    end
end

# ── Test-side: TestChannels constructor accepting a Vec coefficient ───────────

"""
    TestChannels(value::Real, gradient::Vec{D,T}) -> TestChannels{D,T}

Constructor that accepts a Tensors.jl `Vec` for the gradient
coefficient, so a callback can return tensor algebra results directly:

    return TestChannels(0.0, σ ⋅ basevec(Vec{D,T}, test_component))

The conversion to the internal `SVector{D,T}` storage is a tuple repack
(same `NTuple{D,T}` representation) and adds no runtime cost beyond the
existing `SVector` constructor path.
"""
@inline Unfitted.TestChannels(value::Real, g::Vec{D,T}) where {D,T} = TestChannels(value,
                                                                                   SVector{D,T}(Tuple(g)))

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
"""
@inline function Unfitted.gradient_tensor(state, name::Symbol, ::Val{D}) where {D}
    return Tensor{2,D}() do i, j
        Unfitted.field_gradient(state, name, i)[j]
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
# explicit and falls through to a clear `MethodError` otherwise. Users
# with `C ≠ D` can still call the `Val(D)`-explicit form above.

"""
    value_vec(state::FormState, field::Field{D,T,C}) -> Vec{C,T}

Read the field's value at the current quadrature point as a
`Vec{C,T}`. Equivalent to [`value_vec(state, field.name, Val(C))`](@ref)
but infers the component count from the field's type parameters.
"""
@inline function Unfitted.value_vec(state, field::Field{D,T,C}) where {D,T,C}
    return Vec{C}(ntuple(k -> Unfitted.value(state, field.name, k), Val(C)))
end

"""
    gradient_tensor(state::FormState, field::Field{D,T,D}) -> Tensor{2,D,T}

Read the field's physical Jacobian at the current quadrature point as
a 2nd-order tensor. Defined for vector fields whose component count
equals the spatial dimension (`Field{D,T,D}`); for mismatched
dimensions, use the `Val(D)`-explicit overload above and construct
the rectangular Jacobian manually.
"""
@inline function Unfitted.gradient_tensor(state, field::Field{D,T,D}) where {D,T}
    return Tensor{2,D}() do i, j
        Unfitted.field_gradient(state, field.name, i)[j]
    end
end

end # module UnfittedTensorsExt
