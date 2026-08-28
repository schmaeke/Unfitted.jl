# Weak-form algebra and problem composition for the coupled-Galerkin
# assembly path. This file owns the user-facing types that describe
# *what* is being solved — pointwise channels (`TrialChannels`,
# `TestChannels`), the weak-form callback wrapper (`WeakForm`), named
# fields (`Field`), per-block / per-load bindings (`BlockForm`,
# `LoadForm`), and the top-level `Problem` container — together with
# the channel-arithmetic helpers (`_as_test_channels`,
# `_test_contribution`, `_test_value_contribution`) that the assembly
# kernel and the Dirichlet projection both ride.
#
# No assembly, no model lifecycle — those live in `model.jl` (the
# prepared-problem state machine) and `assembly.jl` (the hot loop and
# public assemble APIs). Loaded between `dofs.jl` (whose
# `FieldLayout`/`SystemLayout` consume the field structures here) and
# `model.jl` (which holds the prepared problem).

# ── Weak forms and pointwise channels ─────────────────────────────────────────

"""
    WeakForm(; bilinear, linear, symmetric=false, component_aware=false)

Accumulator weak-form callbacks and metadata.

The `bilinear` callback represents the integrand of the bilinear form

    a(u, v) = ∫_Ω (coefficient × test_value × trial_value + … × ∇v · ∇u + …) dx

evaluated at a single quadrature point. `bilinear(q, trial)` receives the
pointwise scalar trial data as [`TrialChannels`](@ref) and returns
[`TestChannels`](@ref), whose fields are the coefficients that multiply
the corresponding *test* value and *test* physical gradient. Returning
a scalar is shorthand for "value coefficient only, no gradient
coefficient".

The `linear` callback represents the integrand of the linear form

    ℓ(v) = ∫_Ω (source × test_value + … × ∇v + …) dx,

evaluated at a single quadrature point. `linear(q)` follows the same
return convention; the assembly path then contracts the returned
`TestChannels` against the test basis value and gradient to produce one
entry of the right-hand side per active dof.

For component (vector) fields, pass `component_aware = true` and define
`bilinear(q, trial, test_component)` and `linear(q, test_component)`.
`trial.component` is the active trial component; `test_component` is the
active row component. Component-unaware forms automatically apply the
diagonal pattern `trial.component == test_component`.

`symmetric` flags whether `a(u, v) == a(v, u)` and defaults to `false`,
so an undeclared form is assembled in full. Pass `symmetric = true` for a
form known to be symmetric: assembly then builds only the lower triangle
and mirrors it as `tril(A) + tril(A)ᵀ − diag(A)`, roughly halving the
stored pattern and the scatter work. That mirror *discards* the upper
triangle, so declaring symmetry on an asymmetric form silently produces
the wrong matrix. The default therefore costs a symmetric form the caller
forgot to declare a factor of two, rather than costing an asymmetric one
its correctness.
"""
struct WeakForm{B,L}
    bilinear::B
    linear::L
    symmetric::Bool
    component_aware::Bool
end

function WeakForm(; bilinear, linear, symmetric::Bool=false, component_aware::Bool=false)
    return WeakForm{typeof(bilinear),typeof(linear)}(bilinear, linear, symmetric, component_aware)
end

"""
    TrialChannels(component, value, gradient)
    TrialChannels(value, gradient)

Pointwise scalar trial data passed to a weak-form `bilinear` callback at
a single quadrature point.

  - `component::Int` — the trial component currently being assembled. For
    scalar fields this is always `1`; for vector fields the bilinear
    block iterates over `1:C` of the trial [`Field`](@ref) and passes
    each component in turn.
  - `value::T` — the trial basis value at the quadrature point. The
    full reconstructed trial field is `Σ_b coeff_b · value_b`, but
    assembly considers one basis function at a time so the value here
    is a single basis value, not a coefficient sum.
  - `gradient::SVector{D,T}` — the physical gradient of the same basis
    value, in the physical frame (not the reference frame).

The two-argument constructor defaults `component = 1` and is the common
scalar-field shorthand.
"""
struct TrialChannels{D,T<:Real}
    component::Int
    value::T
    gradient::SVector{D,T}
end

function TrialChannels(component::Integer, value::Real, gradient::SVector{D,<:Real}) where {D}
    T = promote_type(typeof(value), eltype(gradient))
    return TrialChannels{D,T}(Int(component), convert(T, value), SVector{D,T}(gradient))
end

function TrialChannels(value::Real, gradient::SVector{D,<:Real}) where {D}
    return TrialChannels(1, value, gradient)
end

function TrialChannels(value::Real, gradient::NTuple{D,<:Real}) where {D}
    TrialChannels(value, SVector{D}(gradient))
end

"""
    TestChannels(value, gradient)

Coefficients that multiply a scalar test-function value and its physical
gradient at one quadrature point. Returned by a `WeakForm` bilinear or
linear callback. The assembly contraction is

    contribution = channels.value × test_value + ⟨channels.gradient, test_gradient⟩,

so a `mass`-style contribution sets `value = coefficient × trial.value`
and `gradient = 0`; a `stiffness`-style contribution sets `value = 0` and
`gradient = diffusion × trial.gradient`; a `source`-style linear
contribution sets `value = source` and `gradient = 0`.

Callbacks may also return a plain scalar, which is interpreted as a pure
value coefficient with a zero gradient coefficient (see
`_as_test_channels`).
"""
struct TestChannels{D,T<:Real}
    value::T
    gradient::SVector{D,T}
end

function TestChannels(value::Real, gradient::SVector{D,<:Real}) where {D}
    T = promote_type(typeof(value), eltype(gradient))
    return TestChannels{D,T}(convert(T, value), SVector{D,T}(gradient))
end

function TestChannels(value::Real, gradient::NTuple{D,<:Real}) where {D}
    TestChannels(value, SVector{D}(gradient))
end

# Zero-vector gradient in the assembly scalar type, used both as the
# default test-gradient coefficient when a callback returns a scalar and
# as an initialiser for the FormState gradient evaluation.
_zero_gradient(::Val{D}, ::Type{T}) where {D,T} = SVector{D,T}(ntuple(_ -> zero(T), Val(D)))

# Coerce a callback return to a `TestChannels{D,T}` of the assembly's
# scalar type. Three dispatched cases:
#
#   * The callback already returned a `TestChannels{D,T}` — pass through.
#   * The callback returned a `TestChannels{D, T'}` with a different
#     scalar type — convert in-place.
#   * The callback returned a scalar — wrap as a value-only channel.
_as_test_channels(channels::TestChannels{D,T}, ::Val{D}, ::Type{T}) where {D,T<:Real} = channels

function _as_test_channels(channels::TestChannels{D}, ::Val{D}, ::Type{T}) where {D,T<:Real}
    return TestChannels{D,T}(convert(T, channels.value), SVector{D,T}(channels.gradient))
end

function _as_test_channels(value::Number, ::Val{D}, ::Type{T}) where {D,T<:Real}
    return TestChannels{D,T}(convert(T, value), _zero_gradient(Val(D), T))
end

# Scalar contraction `channels.value × test_value`. Used by the Dirichlet
# projection path in `dirichlet.jl` and by the value-only branch of
# `_emit_*`.
@inline _test_value_contribution(channels::TestChannels, test_value) = channels.value * test_value

# Full contraction `channels.value × test_value + ⟨channels.gradient,
# test_gradient⟩`. The `@inline` makes sure the dot product is constant-
# folded across the (D ≤ 4) tensor dimensions.
@inline function _test_contribution(channels::TestChannels{D}, test_value, test_gradient) where {D}
    return channels.value * test_value + dot(channels.gradient, test_gradient)
end

# ── Fields, blocks, loads, problems ───────────────────────────────────────────

"""
    Field{D,T,C,S}

One named field over a [`Space`](@ref). `C` is the component count
(parametric so the type carries it) and `S` is the concrete `Space`
type. Use [`field`](@ref) to construct.
"""
struct Field{D,T,C,S<:Space{D,T}}
    name::Symbol
    space::S
end

"""
    field(name::Symbol, space::Space; components=1) -> Field

Create one scalar (`components = 1`) or vector (`components > 1`) field
over `space`. Components share the same scalar basis family, overlay
constraints, and integration regions — they only differ in their
per-component physical Dirichlet data.
"""
function field(name::Symbol, space::Space{D,T}; components::Integer=1) where {D,T}
    components > 0 || throw(ArgumentError("field components must be positive"))
    return Field{D,T,Int(components),typeof(space)}(name, space)
end

"""
    component_count(field) -> Int

Number of scalar components in `field`.
"""
component_count(::Field{D,T,C}) where {D,T,C} = C

# Reject a component index the field does not carry. An out-of-range index
# matches no component downstream, so the Dirichlet condition or Neumann load
# carrying it contributes nothing at all — a silently free boundary or an
# all-zero load rather than an error.
function _check_component(component, field::Field)
    component === nothing ||
        1 ≤ component ≤ component_count(field) ||
        throw(ArgumentError("component $component outside 1:$(component_count(field)) of $(field.name)"))
end

# Field *names*, not full `Field` objects: the assembly hot loop only
# needs the name to look the field up via `_field_index`, and erasing
# `Field{D,T,C,Space{...}}` from the struct parameters keeps the
# `Problem`/`Model` type tree from doubling the field-type nesting per
# block.
"""
    BlockForm(test_name::Symbol, trial_name::Symbol, form, on)

One bilinear block of a [`Problem`](@ref). `form.bilinear` contributes
to the matrix rows of the field named `test_name` and the columns of
the field named `trial_name`. Same-name blocks produce the diagonal
blocks; cross-name blocks produce the off-diagonal coupling. The `on`
tag selects the integration region kind:

  - `nothing` — volume integration over `Ω`.
  - `::BoundarySelector` — facet integration over the selected portion
    of the physical boundary `∂Ω` (codim-1 face, edge, vertex, or the
    whole `∂Ω`).
  - `::BoundaryMesh` — surface integration over the user-supplied
    immersed-boundary mesh.
  - `::Interface` — two-sided integration over a coupling interface
    between two independently discretised subdomains, each field
    evaluated in its own subdomain's cell at the shared quadrature
    point. Build the tag with [`interface`](@ref); [`couple`](@ref)
    builds the four blocks of a jump coupling for you.

Construct via [`block`](@ref), which accepts the [`Field`](@ref)s
directly and defaults `on = nothing`.
"""
struct BlockForm{F,O}
    test_name::Symbol
    trial_name::Symbol
    form::F
    on::O
end

# See `BlockForm` above for why the field is stored by name.
"""
    LoadForm(test_name::Symbol, form, on)

One linear (right-hand-side) contribution to a [`Problem`](@ref).
`form.linear` contributes to the rhs entries of the field named
`test_name`. The `on` tag selects the integration region kind exactly
as for [`BlockForm`](@ref).

Construct via [`loadform`](@ref), which accepts a [`Field`](@ref)
directly and defaults `on = nothing`.
"""
struct LoadForm{F,O}
    test_name::Symbol
    form::F
    on::O
end

"""
    block(test_field::Field, trial_field::Field, form; on=nothing) -> BlockForm

Wrap `form` as the bilinear contribution from `trial_field` (columns) to
`test_field` (rows) inside a multi-field [`Problem`](@ref). When `on` is
a [`BoundarySelector`](@ref), the contribution integrates over that
facet of `∂Ω` instead of the volume — the user-supplied form receives
`q.normal` and `q.sides` on every quadrature point.
"""
function block(test_field::Field, trial_field::Field, form; on=nothing)
    return BlockForm(test_field.name, trial_field.name, form, on)
end

"""
    loadform(test_field::Field, form; on=nothing) -> LoadForm

Wrap `form` as the linear (rhs) contribution to the rows of `test_field`
inside a multi-field [`Problem`](@ref). When `on` is a
[`BoundarySelector`](@ref), the contribution integrates over that facet
of `∂Ω` instead of the volume.
"""
loadform(test_field::Field, form; on=nothing) = LoadForm(test_field.name, form, on)

"""
    Problem{D,T,S,FS,B,L}

A coupled-Galerkin problem: a space, an ordered tuple of fields over
that space, the bilinear blocks and load contributions assembled into
the system, and the Dirichlet conditions imposed at the dof layer.

Fields:

  - `space::S` — a **representative** [`Space`](@ref) (the first field's),
    used only where a single space suffices: the problem's `(D, T)` and
    the single-field / single-domain consumers. The authoritative space of
    each field is `field.space`; a coupled problem carries several distinct
    subdomain spaces (see `problem_spaces`).
  - `fields::FS` — `Tuple` of [`Field`](@ref)s. Order is preserved in
    the global dof enumeration.
  - `blocks::B` — `Tuple` of [`BlockForm`](@ref)s contributing to the
    matrix.
  - `loads::L` — `Tuple` of [`LoadForm`](@ref)s contributing to the
    right-hand side.
  - `dirichlet::Vector` — every [`DirichletCondition`](@ref) the problem
    imposes. Scoped to each field by name during `system_layout`.
  - `symmetric::Bool` — whether the overall system is symmetric. Used
    to decide whether to assemble in the lower triangle and mirror.

Use the [`Problem`](@ref) constructors rather than building this
directly.
"""
struct Problem{D,T,S<:Space{D,T},FS<:Tuple,B,L}
    space::S
    fields::FS
    blocks::B
    loads::L
    dirichlet::Vector
    symmetric::Bool
end

"""
    Problem(V::Space, form; dirichlet=[])
    Problem(u::Field, form; dirichlet=[])
    Problem(fields::Tuple; blocks=(), loads=(), dirichlet=[], symmetric=nothing)

Three convenience constructors:

  - **Scalar over a space** — wraps `V` in an implicit field named `:u`,
    pairs it with itself in the bilinear `block(u, u, form)` and as the
    load test field `loadform(u, form)`. Symmetry inherited from `form`.
  - **Component-field shorthand** — same pattern but with a caller-named
    [`Field`](@ref), so vector fields and custom names work.
  - **Multi-field** — explicit `(field₁, field₂, …)` tuple plus a list
    of blocks and loads referencing those fields by name. `symmetric`
    defaults to "all blocks are symmetric"; pass `symmetric = false`
    explicitly for non-symmetric coupling.

Fields may live on **distinct** spaces: a coupled problem discretises each
subdomain on its own [`Space`](@ref) (own mesh, own level-set fold, own dof
block), and off-diagonal `block(uₐ, u_b, …)` / [`couple`](@ref) forms tie
them together. Single-space problems are the common special case.
"""
Problem(V::Space{D,T}, form; dirichlet=[]) where {D,T} = Problem(field(:u, V), form; dirichlet)

function Problem(u::Field, form; dirichlet=[])
    return Problem((u,); blocks=(block(u, u, form),), loads=(loadform(u, form),), dirichlet,
                   symmetric=form.symmetric)
end

# Collect the field names of a problem's field tuple and check for
# duplicates. Used by the multi-field `Problem` constructor.
function _field_names(fields)
    names = Symbol[]
    for field in fields
        field.name in names && throw(ArgumentError("duplicate field name $(field.name)"))
        push!(names, field.name)
    end
    return names
end

# Sanity-check the field tuple of a multi-field problem: non-empty, no
# duplicate names. Fields may live over *different* spaces (multi-domain
# coupling): each `Field` carries its own `Space`, and the dof, integration,
# and assembly layers key everything by the owning field. Returns a
# *representative* space — the first field's — used only to infer the
# problem's `(D, T)` type parameters and to serve single-field consumers that
# still reach through `problem.space`; the authoritative per-field space is
# always `field.space`. Also returns the field-name list.
function _check_problem_fields(fields::Tuple)
    isempty(fields) && throw(ArgumentError("a problem needs at least one field"))
    names = _field_names(fields)
    return first(fields).space, names
end

"""
    problem_spaces(problem::Problem) -> Vector{Space}

Distinct spaces spanned by a problem's fields, in first-appearance order,
deduplicated by object identity (fields sharing a space appear once). This
is the set of independent discretisations a multi-domain problem couples;
for a single-domain problem it is a one-element vector. Consumers that must
act per-discretisation — the physical fold, the per-space integration
plans, and the assembly workspace's level-id namespacing — iterate this.
"""
function problem_spaces(problem::Problem)
    spaces = Space[]
    for field in problem.fields
        any(s -> s === field.space, spaces) || push!(spaces, field.space)
    end
    return spaces
end

# Whether `problem` couples more than one distinct subdomain space. The
# single-space case is the dominant path and takes typed fast branches (e.g. the
# concretely-typed level tuple in `_assembly_workspace`).
_is_multidomain(problem::Problem) = length(problem_spaces(problem)) > 1

# The `Space` owning `field_name` in `problem`. Used to route a field's
# volume / boundary / interface contributions to its own discretisation.
function _field_space(problem::Problem, field_name::Symbol)
    for field in problem.fields
        field.name === field_name && return field.space
    end
    throw(ArgumentError("unknown field $field_name"))
end

# Check that every block/load form references known field names. Raises
# on the first unknown field so the user sees the offending name.
function _check_form_fields(forms, names)
    for form in forms
        form.test_name in names || throw(ArgumentError("unknown test field $(form.test_name)"))
        form isa BlockForm &&
            !(form.trial_name in names) &&
            throw(ArgumentError("unknown trial field $(form.trial_name)"))
    end
end

# Range-check each Dirichlet condition's `component` against the field it
# names, so an index no component carries is rejected here rather than
# silently leaving its boundary free. A condition left unscoped
# (`field === nothing`) on a multi-field problem names no field to check
# against; `_dirichlet_for_field` resolves those at `prepare` time.
function _check_dirichlet(dirichlet, fields::Tuple)
    for condition in dirichlet
        name = condition.field
        index = name === nothing ? (length(fields) == 1 ? 1 : nothing) :
                findfirst(f -> f.name === name, fields)
        index === nothing || _check_component(condition.component, fields[index])
    end
end

function Problem(fields::Tuple; blocks=(), loads=(), dirichlet=[], symmetric=nothing)
    space_data, names = _check_problem_fields(fields)
    block_tuple = Tuple(blocks)
    load_tuple = Tuple(loads)
    _check_form_fields(block_tuple, names)
    _check_form_fields(load_tuple, names)
    _check_dirichlet(dirichlet, fields)
    symmetric_value = symmetric === nothing ? all(block -> block.form.symmetric, block_tuple) :
                      Bool(symmetric)
    D = length(space_data.domain.lower)
    T = eltype(space_data.domain.lower)
    return Problem{D,T,typeof(space_data),typeof(fields),typeof(block_tuple),typeof(load_tuple)}(space_data,
                                                                                                 fields,
                                                                                                 block_tuple,
                                                                                                 load_tuple,
                                                                                                 collect(dirichlet),
                                                                                                 symmetric_value)
end
