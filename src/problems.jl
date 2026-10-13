# Weak-form algebra and problem composition for the coupled-Galerkin assembly path. This file owns the user-facing types
# that describe *what* is being solved — pointwise channels (`TrialChannels`, `TestChannels`), the weak-form callback
# wrapper (`WeakForm`), named fields (`Field`), per-block / per-load bindings (`BlockForm`, `LoadForm`), and the
# top-level `Problem` container — together with the channel-arithmetic helpers (`_as_test_channels`,
# `_test_contribution`) that the assembly kernel and the L² transfer ride.
#
# No assembly, no model lifecycle — those live in `model.jl` (the prepared-problem state machine) and `assembly.jl` (the
# hot loop and public assemble APIs). Loaded between `dofs.jl` (whose `FieldLayout`/`SystemLayout` consume the field
# structures here) and `model.jl` (which holds the prepared problem).

# ── Weak forms and pointwise channels ─────────────────────────────────────────

"""
    WeakForm(; bilinear=nothing, linear=nothing, symmetric=false, component_aware=false)

Accumulator weak-form callbacks and metadata.

The `bilinear` callback represents the integrand of the bilinear form

    a(u, v) = ∫_Ω (coefficient × test_value × trial_value + … × ∇v · ∇u + …) dx

evaluated at a single quadrature point. `bilinear(q, trial)` receives the pointwise scalar trial data as
[`TrialChannels`](@ref) and returns [`TestChannels`](@ref), whose fields are the coefficients that multiply the
corresponding *test* value and *test* physical gradient. Returning a scalar is shorthand for "value coefficient only, no
gradient coefficient". A gradient coefficient shorter than the mesh dimension is zero-extended into the remaining slots
— the shape a field with fewer components than the mesh has axes produces — while a longer one raises a
`DimensionMismatch`; see [`TestChannels`](@ref).

The `linear` callback represents the integrand of the linear form

    ℓ(v) = ∫_Ω (source × test_value + … × ∇v + …) dx,

evaluated at a single quadrature point. `linear(q)` follows the same return convention; the assembly path then contracts
the returned `TestChannels` against the test basis value and gradient to produce one entry of the right-hand side per
active dof.

Both callbacks default to `nothing`, so a form declares only the sides it actually has: a bilinear-only form to hand to
[`block`](@ref), a linear-only form to hand to [`loadform`](@ref). A form declaring neither side describes no
contribution at all and raises `ArgumentError` — `symmetric` and `component_aware` carry no integrand and do not rescue
it.

An absent side is *absent*, not zero: nothing substitutes a zero callback for it. Attach a one-sided form *directly*
only on the side it declares — [`block`](@ref) reads `bilinear` and [`loadform`](@ref) reads `linear`, and each raises
`ArgumentError` for a form that lacks the side it reads instead of contributing zero.

The single-field `Problem(V, form)` / `Problem(u, form)` constructors attach both sides of one form, and read the
declaration to decide which: a one-sided form there yields a problem carrying that side alone. A bilinear-only form is
the Laplace-with-boundary-data shape — an operator with no load, driven entirely by its inhomogeneous Dirichlet data —
and needs no stub to say so.

For component (vector) fields, pass `component_aware = true` and define `bilinear(q, trial, test_component)` and
`linear(q, test_component)`. `trial.component` is the active trial component; `test_component` is the active row
component. Component-unaware forms automatically apply the diagonal pattern `trial.component == test_component`.

`symmetric` flags whether `a(u, v) == a(v, u)` and defaults to `false`, so an undeclared form is assembled in full. Pass
`symmetric = true` for a form known to be symmetric: assembly then builds only the lower triangle and mirrors it as
`tril(A) + tril(A)ᵀ − diag(A)`, roughly halving the stored pattern and the scatter work. That mirror *discards* the
upper triangle, so declaring symmetry on an asymmetric form silently produces the wrong matrix. The default therefore
costs a symmetric form the caller forgot to declare a factor of two, rather than costing an asymmetric one its
correctness.

Assembly calls a form's callbacks only at the quadrature points of regions on which its test field has degrees of
freedom. On a coupled model, a volume form therefore runs over its own subdomain and nowhere else, and a callback must
not rely on being called at the points of another subdomain — to count or record them, say.
"""
struct WeakForm{B,L}
    bilinear::B
    linear::L
    symmetric::Bool
    component_aware::Bool
end

function WeakForm(; bilinear=nothing, linear=nothing, symmetric::Bool=false, component_aware::Bool=false)
    (bilinear === nothing && linear === nothing) &&
        throw(ArgumentError("a WeakForm needs a bilinear part, a linear part, or both"))
    return WeakForm{typeof(bilinear),typeof(linear)}(bilinear, linear, symmetric, component_aware)
end

"""
    TrialChannels(component, value, gradient)
    TrialChannels(value, gradient)

Pointwise scalar trial data passed to a weak-form `bilinear` callback at a single quadrature point.

  - `component::Int` — the trial component currently being assembled. For scalar fields this is always `1`; for vector
    fields the bilinear block iterates over `1:C` of the trial [`Field`](@ref) and passes each component in turn.
  - `value::T` — the trial basis value at the quadrature point. The full reconstructed trial field is
    `Σ_b coeff_b · value_b`, but assembly considers one basis function at a time so the value here is a single basis
    value, not a coefficient sum.
  - `gradient::SVector{D,T}` — the physical gradient of the same basis value, in the physical frame (not the reference
    frame).

The two-argument constructor defaults `component = 1` and is the common scalar-field shorthand.
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

TrialChannels(value::Real, gradient::SVector{D,<:Real}) where {D} = TrialChannels(1, value, gradient)

TrialChannels(value::Real, gradient::NTuple{D,<:Real}) where {D} = TrialChannels(value, SVector{D}(gradient))

"""
    TestChannels(value, gradient)

Coefficients that multiply a scalar test-function value and its physical gradient at one quadrature point. Returned by a
`WeakForm` bilinear or linear callback. The assembly contraction is

    contribution = channels.value × test_value + ⟨channels.gradient, test_gradient⟩,

so a `mass`-style contribution sets `value = coefficient × trial.value` and `gradient = 0`; a `stiffness`-style
contribution sets `value = 0` and `gradient = diffusion × trial.gradient`; a `source`-style linear contribution sets
`value = source` and `gradient = 0`.

Callbacks may also return a plain scalar, which is interpreted as a pure value coefficient with a zero gradient
coefficient.

The gradient coefficient may also be **shorter** than the mesh dimension `D`, in which case assembly zero-extends it
into the remaining slots. That is what a field with fewer components than the mesh has axes produces — a `C`-component
displacement on a `D`-axis space-time mesh has a `C × C` stress, whose rows are `C`-length coefficients — and the axes
it does not reach are the ones the field carries no component for, where the flux is genuinely zero. A coefficient
**longer** than `D` names a flux along an axis the mesh does not have and raises a `DimensionMismatch` rather than being
truncated.
"""
struct TestChannels{D,T<:Real}
    value::T
    gradient::SVector{D,T}
end

function TestChannels(value::Real, gradient::SVector{D,<:Real}) where {D}
    T = promote_type(typeof(value), eltype(gradient))
    return TestChannels{D,T}(convert(T, value), SVector{D,T}(gradient))
end

TestChannels(value::Real, gradient::NTuple{D,<:Real}) where {D} = TestChannels(value, SVector{D}(gradient))

# Coerce a callback's return to the `TestChannels{D,T}` the kernel contracts: the mesh dimension `D` and the assembly
# scalar type `T`. A plain scalar is a value-only channel with a zero gradient coefficient. A `TestChannels{K}` keeps
# its value and its gradient's `K` slots, converted to `T`, and is zero-extended into the remaining `D − K`. `K`, `D`
# and `T` are static, so the slot test folds away.
#
# A channel that already is a `TestChannels{D,T}` passes through untouched, by the first method. The general one would
# return the same values, but rebuilding the channel changes how the compiled kernel fuses its multiply-adds: measured
# without the pass-through, the stiffness matrix of a 4D Poisson problem moved by 4.9e-17 relative (enough to flip an
# exact zero out of the sparsity pattern), and a dense and a static diffusion tensor no longer assembled to the same
# bits. The pass-through keeps the kernel's code, and its results, as they were.
#
# A short gradient coefficient is legitimate, and the space-time case is where it comes from: a field of `C` components
# on a mesh of `D = C + 1` axes (the last axis time) has a `C × C` stress, and the row of it that the test component
# selects is a `C`-length gradient coefficient, while assembly contracts against the `D`-length test gradient. The
# slots the coefficient does not reach are the axes the field carries no component for, and the callback's flux along
# them really is zero, so extending by zero records what the callback said rather than patching what it forgot.
#
# The long direction is an error and not a truncation. A coefficient longer than the mesh dimension names a flux along
# an axis that does not exist; dropping the surplus slots would quietly delete a term the callback asked for, and the
# resulting operator would still be symmetric, still solve and still converge — the same silent wrongness the
# zero-extension exists to remove from the short direction.
_as_test_channels(channels::TestChannels{D,T}, ::Val{D}, ::Type{T}) where {D,T<:Real} = channels

function _as_test_channels(channels::TestChannels{K}, ::Val{D}, ::Type{T}) where {K,D,T<:Real}
    K <= D || throw(DimensionMismatch("a test gradient coefficient of length $K cannot be used on " *
                                      "a $D-dimensional mesh: it names a flux along an axis the " *
                                      "mesh does not have"))
    gradient = channels.gradient
    return TestChannels{D,T}(convert(T, channels.value),
                             SVector{D,T}(ntuple(i -> i <= K ? convert(T, gradient[i]) : zero(T), Val(D))))
end

function _as_test_channels(value::Number, ::Val{D}, ::Type{T}) where {D,T<:Real}
    return TestChannels{D,T}(convert(T, value), zero(SVector{D,T}))
end

# Value-only contraction for a scalar channel, the L² transfer rhs: it never reads the test gradient, so a value-only
# basis refresh is enough.
@inline _test_contribution(channel::Number, test_value, _) = channel * test_value

# Full contraction `channels.value × test_value + ⟨channels.gradient, test_gradient⟩`. The `@inline` makes sure the dot
# product is constant- folded across the (D ≤ 4) tensor dimensions.
@inline function _test_contribution(channels::TestChannels{D}, test_value, test_gradient) where {D}
    return channels.value * test_value + dot(channels.gradient, test_gradient)
end

# ── Fields, blocks, loads, problems ───────────────────────────────────────────

"""
    Field{D,T,C,S}

One named field over a [`Space`](@ref). `C` is the component count (parametric so the type carries it) and `S` is the
concrete `Space` type. Use [`field`](@ref) to construct.
"""
struct Field{D,T,C,S<:Space{D,T}}
    name::Symbol
    space::S
end

"""
    field(name::Symbol, space::Space; components=1) -> Field

Create one scalar (`components = 1`) or vector (`components > 1`) field over `space`. Components share the same scalar
basis family, overlay constraints, and integration regions — they only differ in their per-component physical Dirichlet
data.
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

# The field a single-field entry point (`Problem(x, form)`, `mass`, `stiffness`, `poisson`) works on: a bare `Space` is
# wrapped in the implicit scalar field `:u`, and a `Field` is taken as the caller named and sized it.
_as_field(V::Space) = field(:u, V)
_as_field(u::Field) = u

# Reject a component index the field does not carry. An out-of-range index matches no component downstream, so the
# Dirichlet condition or Neumann load carrying it contributes nothing at all — a silently free boundary or an all-zero
# load rather than an error.
function _check_component(component, field::Field)
    component === nothing ||
        1 ≤ component ≤ component_count(field) ||
        throw(ArgumentError("component $component outside 1:$(component_count(field)) of $(field.name)"))
end

# Field *names*, not full `Field` objects: the assembly hot loop only needs the name to look the field up via
# `_field_index`, and erasing `Field{D,T,C,Space{...}}` from the struct parameters keeps the `Problem`/`Model` type tree
# from doubling the field-type nesting per block.
"""
    BlockForm(test_name::Symbol, trial_name::Symbol, form, on)

One bilinear block of a [`Problem`](@ref). `form.bilinear` contributes to the matrix rows of the field named `test_name`
and the columns of the field named `trial_name`. Same-name blocks produce the diagonal blocks; cross-name blocks produce
the off-diagonal coupling. The `on` tag selects the integration region kind:

  - `nothing` — volume integration over `Ω`.
  - `::BoundarySelector` — facet integration over the selected portion of the physical boundary `∂Ω` (codim-1 face,
    edge, vertex, or the whole `∂Ω`).
  - `::BoundaryMesh` — surface integration over the user-supplied immersed-boundary mesh.
  - `::Interface` — two-sided integration over a coupling interface between two independently discretised subdomains,
    each field evaluated in its own subdomain's cell at the shared quadrature point. Build the tag with
    [`interface`](@ref); [`couple`](@ref) builds the four blocks of a jump coupling for you.

Construct via [`block`](@ref), which accepts the [`Field`](@ref)s directly and defaults `on = nothing`.
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

One linear (right-hand-side) contribution to a [`Problem`](@ref). `form.linear` contributes to the rhs entries of the
field named `test_name`. The `on` tag selects the integration region kind exactly as for [`BlockForm`](@ref).

Construct via [`loadform`](@ref), which accepts a [`Field`](@ref) directly and defaults `on = nothing`.
"""
struct LoadForm{F,O}
    test_name::Symbol
    form::F
    on::O
end

"""
    block(test_field::Field, trial_field::Field, form; on=nothing) -> BlockForm

Wrap `form` as the bilinear contribution from `trial_field` (columns) to `test_field` (rows) inside a multi-field
[`Problem`](@ref). When `on` is a [`BoundarySelector`](@ref), the contribution integrates over that facet of `∂Ω`
instead of the volume — the user-supplied form receives `q.normal` and `q.sides` on every quadrature point. A form
without a `bilinear` side raises `ArgumentError` (see [`WeakForm`](@ref)).

A block carries `form.bilinear` and nothing else. A [`WeakForm`](@ref) that also declares a `linear` part contributes
nothing to the right-hand side through `block`: assembly reads the load side only from the [`LoadForm`](@ref)s in a
problem's `loads`. The linear part of a two-sided form therefore has to be attached separately with [`loadform`](@ref) —
or, for the plain `ℓ(v) = ∫_Ω f v dx` shape, with [`source_load`](@ref). Omitting it leaves the homogeneous system,
which assembles, solves, and reports a clean residual, so the omission surfaces only as a wrong answer.

The single-field `Problem(V, form)` / `Problem(u, form)` constructors do attach both sides of a form that declares both.
That is why the habit the convenience path teaches does not carry over here: on the multi-field path the two sides are
two separate attachments.
"""
function block(test_field::Field, trial_field::Field, form; on=nothing)
    # A side the form lacks used to reach the assembly loop as an uncallable `nothing` and raise a `MethodError` there,
    # inside a `TaskFailedException` when threaded; the attachment is where the mistake is spelled, so it is refused
    # here, and in `loadform` likewise.
    form.bilinear === nothing && throw(ArgumentError("block attaches a form's bilinear side, and this form has " *
                                                     "none; attach a linear-only form with loadform"))
    return BlockForm(test_field.name, trial_field.name, form, on)
end

"""
    loadform(test_field::Field, form; on=nothing) -> LoadForm

Wrap `form` as the linear (rhs) contribution to the rows of `test_field` inside a multi-field [`Problem`](@ref). When
`on` is a [`BoundarySelector`](@ref), the contribution integrates over that facet of `∂Ω` instead of the volume. A form
without a `linear` side raises `ArgumentError` (see [`WeakForm`](@ref)).
"""
function loadform(test_field::Field, form; on=nothing)
    form.linear === nothing && throw(ArgumentError("loadform attaches a form's linear side, and this form has " *
                                                   "none; attach a bilinear-only form with block"))
    return LoadForm(test_field.name, form, on)
end

"""
    Problem{D,T,S,FS,B,L}

A coupled-Galerkin problem: a space, an ordered tuple of fields over that space, the bilinear blocks and load
contributions assembled into the system, and the Dirichlet conditions imposed at the dof layer.

Fields:

  - `space::S` — a **representative** [`Space`](@ref) (the first field's), used only where a single space suffices: the
    problem's `(D, T)` and the single-field / single-domain consumers. The authoritative space of each field is
    `field.space`; a coupled problem carries several distinct subdomain spaces (see `problem_spaces`).
  - `fields::FS` — `Tuple` of [`Field`](@ref)s. Order is preserved in the global dof enumeration.
  - `blocks::B` — `Tuple` of [`BlockForm`](@ref)s contributing to the matrix.
  - `loads::L` — `Tuple` of [`LoadForm`](@ref)s contributing to the right-hand side.
  - `dirichlet::Vector` — every [`DirichletCondition`](@ref) the problem imposes. Scoped to each field by name during
    `system_layout`.
  - `symmetric::Bool` — whether the overall system is symmetric. Used to decide whether to assemble in the lower
    triangle and mirror.

Use the [`Problem`](@ref) constructors rather than building this directly.
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

  - **Scalar over a space** — wraps `V` in an implicit field named `:u`, pairs it with itself in the bilinear
    `block(u, u, form)` and as the load test field `loadform(u, form)`, attaching each side only when `form` declares it
    (see [`WeakForm`](@ref)). Symmetry inherited from `form`.
  - **Component-field shorthand** — same pattern but with a caller-named [`Field`](@ref), so vector fields and custom
    names work.
  - **Multi-field** — explicit `(field₁, field₂, …)` tuple plus a list of blocks and loads referencing those fields by
    name. `symmetric` defaults to "all blocks are symmetric"; pass `symmetric = false` explicitly for non-symmetric
    coupling.

Fields may live on **distinct** spaces: a coupled problem discretises each subdomain on its own [`Space`](@ref) (own
mesh, own level-set fold, own dof block), and off-diagonal `block(uₐ, u_b, …)` / [`couple`](@ref) forms tie them
together. Single-space problems are the common special case.
"""
# The single-field shorthand attaches one form to both sides, so it attaches each side only where the form declares one.
# `WeakForm` required both callbacks until it learned to take `nothing`, so on every form constructible before then both
# branches are taken and this is a no-op; what it adds is the one-sided case, where a `nothing` would otherwise reach
# the assembly loop and raise. The bilinear-only direction is the useful one: an operator with no load, driven entirely
# by inhomogeneous Dirichlet data, which is Laplace with boundary data and which used to need a dead `linear = q -> 0.0`
# to say so.
function Problem(x::Union{Space,Field}, form; dirichlet=[])
    u = _as_field(x)
    blocks = form.bilinear === nothing ? () : (block(u, u, form),)
    loads = form.linear === nothing ? () : (loadform(u, form),)
    return Problem((u,); blocks, loads, dirichlet, symmetric=form.symmetric)
end

"""
    problem_spaces(problem::Problem) -> Vector{Space}

Distinct spaces spanned by a problem's fields, in first-appearance order, deduplicated by object identity (fields
sharing a space appear once). This is the set of independent discretisations a multi-domain problem couples; for a
single-domain problem it is a one-element vector. Consumers that must act per-discretisation — the physical fold, the
per-space integration plans, and the assembly workspace's level-id namespacing — iterate this.
"""
function problem_spaces(problem::Problem)
    spaces = Space[]
    for field in problem.fields
        any(s -> s === field.space, spaces) || push!(spaces, field.space)
    end
    return spaces
end

# The `Space` owning `field_name` in `problem`. Used to route a field's volume / boundary / interface contributions to
# its own discretisation.
function _field_space(problem::Problem, field_name::Symbol)
    for field in problem.fields
        field.name === field_name && return field.space
    end
    throw(ArgumentError("unknown field $field_name"))
end

# Validate a problem's parts in one pass, in the order they depend on each other: the fields (at least one, no name
# twice), then every block and load against the field names, then every Dirichlet condition against the field it
# constrains. Each refusal names the offending field, and a repeated name is reported at its first repeat.
#
# Every condition is resolved to its owner here, where the fields are at hand: an unscoped one (`field === nothing`)
# belongs to the only field of a single-field problem, a named one to the field of that name. Its `component` must be
# one the owner carries, since an index no component carries would silently leave its boundary free, and a constant
# tuple or array datum must carry one entry per component (`_check_dirichlet_datum`). An unscoped condition on a
# multi-field problem, or a name no field carries, used to construct and fail only at `prepare`, where the layout of
# each field picks out its conditions; the messages are the ones `prepare` gave. `update_dirichlet!` checks its new
# conditions' data against the model's field layouts instead (its structural check already pins `field` and
# `component`).
#
# The names are gathered into a tuple of `Symbol`s first, so the lookups run over a homogeneous tuple; the fields
# themselves are a heterogeneous one. Fields may live on different spaces (multi-domain coupling): each `Field` carries
# its own, and the dof, integration and assembly layers key everything by the owning field.
function _check_problem(fields::Tuple, blocks::Tuple, loads::Tuple, dirichlet)
    isempty(fields) && throw(ArgumentError("a problem needs at least one field"))
    names = map(f -> f.name, fields)
    for (i, name) in enumerate(names)
        findfirst(==(name), names) == i || throw(ArgumentError("duplicate field name $name"))
    end
    for form in (blocks..., loads...)
        form.test_name in names || throw(ArgumentError("unknown test field $(form.test_name)"))
        form isa BlockForm &&
            !(form.trial_name in names) &&
            throw(ArgumentError("unknown trial field $(form.trial_name)"))
    end
    for condition in dirichlet
        name = condition.field
        name === nothing &&
            length(fields) > 1 &&
            throw(ArgumentError("multi-field Dirichlet conditions must specify a field"))
        index = name === nothing ? 1 : findfirst(==(name), names)
        index === nothing && throw(ArgumentError("unknown field $name"))
        owner = fields[index]
        _check_component(condition.component, owner)
        _check_dirichlet_datum(condition, component_count(owner), owner.name)
    end
    return nothing
end

# The system's symmetry flag: `flag` when one is given, else whether every block declares its form symmetric. The one
# rule for both the `Problem` constructor and a one-shot assembly of loose blocks (`_assemble`), so the two cannot infer
# the flag differently. The blocks are not specialised on: this runs once per problem or assembly call, and the
# closure's `all` specialises on the tuple anyway.
function _symmetric_flag(@nospecialize(blocks), flag)
    return flag === nothing ? all(block -> block.form.symmetric, blocks) : Bool(flag)
end

# The representative space `V` is the first field's: it supplies the problem's `(D, T)` type parameters and serves the
# single-field consumers that reach through `problem.space`, while the authoritative space of each field is
# `field.space`. `D` and `T` appear only in the bound of `S`, so the type parameters are spelled out.
function Problem(fields::Tuple; blocks=(), loads=(), dirichlet=[], symmetric=nothing)
    block_tuple, load_tuple = Tuple(blocks), Tuple(loads)
    _check_problem(fields, block_tuple, load_tuple, dirichlet)
    V = first(fields).space
    D, T = length(V.domain.lower), eltype(V.domain.lower)
    flag = _symmetric_flag(block_tuple, symmetric)
    return Problem{D,T,typeof(V),typeof(fields),typeof(block_tuple),typeof(load_tuple)}(V, fields, block_tuple,
                                                                                        load_tuple, collect(dirichlet),
                                                                                        flag)
end
