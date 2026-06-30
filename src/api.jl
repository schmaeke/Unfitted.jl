# Thin public-API layer over the lower-level types and forms. Functions
# here exist to shorten the canonical user workflows — building a
# Poisson / mass / stiffness problem, picking a physical boundary,
# attaching Dirichlet data — without forcing the user to construct
# `WeakForm` callbacks by hand. They are deliberately compact wrappers:
# every entry point composes the geometry, basis, dof, and assembly
# layers below but adds no behaviour that those layers can't already
# express.

# ── Coefficient wrappers ──────────────────────────────────────────────────────

# `Number` (scalar), `AbstractMatrix` (D × D diffusion tensor), and
# `UniformScaling` (isotropic scalar) values share the same "fixed
# value at every point" semantics, so they are wrapped uniformly as
# `ConstantCoefficient`. Anything else is assumed to be a callable
# `value(x)` and wrapped as `FunctionCoefficient`. The two-type
# representation lets the bilinear / linear callbacks dispatch on the
# coefficient kind cheaply at the call site.
struct ConstantCoefficient{C}
    value::C
end

struct FunctionCoefficient{F}
    f::F
end

# A constant coefficient is anything that is not a callback: a scalar, a
# per-component `SVector`/tuple/array, a `D × D` diffusion tensor, or a
# `UniformScaling`. Only genuine callables `c(x)` become `FunctionCoefficient`,
# so a bare per-component value (e.g. `coefficient = SVector(2, 3)`) is resolved
# component-wise rather than mistakenly called as a function of `x`.
function _as_coefficient(value::Union{Number,AbstractArray,UniformScaling,Tuple})
    ConstantCoefficient(value)
end
_as_coefficient(value) = FunctionCoefficient(value)

# Resolve a coefficient at the physical point `x`. Constant
# coefficients ignore `x`; function coefficients evaluate `f(x)`.
_coefficient_value(c::ConstantCoefficient, x) = c.value
_coefficient_value(c::FunctionCoefficient, x) = c.f(x)

# Pick the requested component of a coefficient value. Routes through
# `_component_value` so a scalar coefficient applies to every
# component uniformly while an indexable (`SVector`, `Tuple`) coefficient
# resolves the component slot.
function _component_coefficient_value(c, x, component::Integer)
    _component_value(_coefficient_value(c, x), component)
end

# Diffusion flux `a · ∇u` for the three accepted shapes of `a`:
# scalar (isotropic), `UniformScaling` (also isotropic; supports
# `I` and `λI`), and `D × D` `AbstractMatrix` (anisotropic tensor).
# The matrix shape is checked at the call site; the scalar / uniform
# paths broadcast directly across the gradient SVector.
_diffusion_flux(a::Number, ugrad::SVector{D}) where {D} = a * ugrad
_diffusion_flux(a::UniformScaling, ugrad::SVector{D}) where {D} = a * ugrad

function _diffusion_flux(a::AbstractMatrix, ugrad::SVector{D}) where {D}
    size(a) == (D, D) ||
        throw(DimensionMismatch("diffusion tensor has size $(size(a)); expected ($D, $D)"))
    return SVector{D}(a * ugrad)
end

# Stiffness bilinear-channel kernel: returns a `TestChannels` whose
# gradient coefficient is `a · ∇u` (the diffusion flux) and whose
# value coefficient is zero. Off-diagonal-in-components contributions
# are zero (diagonal-in-components stiffness only).
function _stiffness_channels(diffusion, q, trial, test_component)
    trial.component == test_component ?
    TestChannels(0.0, _diffusion_flux(_coefficient_value(diffusion, q.x), trial.gradient)) : 0.0
end

# Source-callback kernel: resolves the source coefficient at the
# quadrature point and returns its component value. Wrapped here so
# the linear closures in `source_form` and `_poisson_form` can share
# the same pattern.
_source_value(source, q, test_component) = _component_coefficient_value(source, q.x, test_component)

# Mass bilinear-channel kernel: the component-resolved coefficient times
# the trial value, diagonal in components (off-diagonal trial/test pairs
# contribute zero). Mirrors `_stiffness_channels` / `_source_value` so all
# three canonical forms resolve coefficients the same component-aware way —
# in particular a per-component (`SVector`/tuple) mass coefficient picks the
# slot `test_component` rather than multiplying the trial value by the whole
# coefficient vector.
function _mass_value(coefficient, q, trial, test_component)
    return trial.component == test_component ?
           _component_coefficient_value(coefficient, q.x, test_component) * trial.value : 0.0
end

# ── Boundary selectors ───────────────────────────────────────────────────────

"""
    boundary(:all)
    boundary(axis=d, side=s)
    boundary((axis=d1, side=s1), (axis=d2, side=s2), ...)

Select a physical facet for boundary conditions:

  - `boundary(:all)` — the whole physical boundary `∂Ω` (union of all
    codim-1 faces).
  - `boundary(axis=d, side=s)` — one codim-1 face. `side` is `:lower`
    or `:upper`.
  - `boundary((axis=d₁, side=s₁), …)` — the intersection of `K`
    codim-1 faces, i.e. a codim-`K` facet. Two pairs in 2D pick a
    corner; two pairs in 3D pick an edge; three pairs in 3D pick a
    vertex. Each axis may appear at most once, and `side` must be
    `:lower` or `:upper`.

The dimension-independent Dirichlet projection treats every facet
through one code path: a codim-`K` facet is integrated against the
`(D − K)`-dim quadrature on the facet, which for `K = D` collapses to
a single point evaluation with unit weight.
"""
boundary(selector::Symbol) = BoundarySelector(selector, Tuple{Int,Symbol}[])

function boundary(; axis::Integer, side::Symbol)
    side in (:lower, :upper) || throw(ArgumentError("boundary side must be :lower or :upper"))
    return BoundarySelector(:sides, [(Int(axis), side)])
end

function boundary(pairs::NamedTuple...)
    isempty(pairs) && throw(ArgumentError("boundary requires at least one (axis, side) pair"))
    sides = Tuple{Int,Symbol}[]
    seen = Set{Int}()
    for p in pairs
        (haskey(p, :axis) && haskey(p, :side)) ||
            throw(ArgumentError("boundary pairs must carry :axis and :side keys"))
        axis = Int(p.axis)
        side = p.side
        side in (:lower, :upper) || throw(ArgumentError("boundary side must be :lower or :upper"))
        axis in seen && throw(ArgumentError("boundary axis $axis specified more than once"))
        push!(seen, axis)
        push!(sides, (axis, side))
    end
    return BoundarySelector(:sides, sides)
end

"""
    dirichlet(value; on, field=nothing, component=nothing) -> DirichletCondition

Create a physical Dirichlet condition.

  - `value` — Dirichlet datum. Scalar (applied everywhere on the
    selected facet), indexable component value (one entry per
    component for vector fields), or callback `value(x)` evaluated at
    physical coordinates.
  - `on::BoundarySelector` — the facet to constrain. Build via
    [`boundary`](@ref).
  - `field` — field to constrain. Pass `nothing` for single-field
    problems (the only field is constrained implicitly). Pass a
    `Symbol` or [`Field`](@ref) for multi-field problems.
  - `component` — component index for vector fields when constraining
    only one channel (e.g. a roller / symmetry boundary that pins the
    normal component). `nothing` (default) constrains every component.

Nonzero `value` data are projected onto the boundary trace space
before strong elimination so the constrained dof values reflect the
correct trace. Homogeneous (constant-zero) data skip the projection
entirely.
"""
function dirichlet(value; on::BoundarySelector, field=nothing, component=nothing)
    field_name = field isa Field ? field.name : field
    field_name === nothing ||
        field_name isa Symbol ||
        throw(ArgumentError("Dirichlet field must be a field object, Symbol, or nothing"))
    component === nothing ||
        component isa Integer ||
        throw(ArgumentError("Dirichlet component must be an integer or nothing"))
    return DirichletCondition(value, on, field_name,
                              component === nothing ? nothing : Int(component))
end

# ── Canonical weak forms ──────────────────────────────────────────────────────

"""
    mass_form(; coefficient=1)
    stiffness_form(; diffusion=1)
    source_form(; source)

Build standard accumulator [`WeakForm`](@ref)s without committing to
a full [`Problem`](@ref). Useful when several operators share one
prepared model — e.g. building a mass operator and a stiffness
operator over the same `Model` for time stepping.

  - `mass_form(; coefficient)` — `∫ c(x) · v u dx`. `coefficient` is
    a scalar, a callback `c(x)`, or a per-component indexable value.
    Diagonal in components for vector fields.
  - `stiffness_form(; diffusion)` — `∫ ⟨∇v, A(x) ∇u⟩ dx`. `diffusion`
    is a scalar (isotropic), a callback returning a scalar, a constant
    `D × D` matrix (anisotropic), or a callback returning such a
    matrix. Diagonal in components for vector fields.
  - `source_form(; source)` — `∫ source(x) · v dx`. `source` is a
    scalar, a callback `source(x)`, or a per-component value.

All three are symmetric and component-aware.
"""
function mass_form(; coefficient=1)
    coefficient_data = _as_coefficient(coefficient)
    return WeakForm(bilinear=(q, trial, test_component) -> _mass_value(coefficient_data, q, trial,
                                                                       test_component),
                    linear=(q, test_component) -> 0.0, symmetric=true, component_aware=true)
end

function stiffness_form(; diffusion=1)
    diffusion_coefficient = _as_coefficient(diffusion)
    return WeakForm(bilinear=(q, trial, test_component) -> _stiffness_channels(diffusion_coefficient,
                                                                               q, trial,
                                                                               test_component),
                    linear=(q, test_component) -> 0.0, symmetric=true, component_aware=true)
end

function source_form(; source)
    source_coefficient = _as_coefficient(source)
    return WeakForm(bilinear=(q, trial, test_component) -> 0.0,
                    linear=(q, test_component) -> _source_value(source_coefficient, q,
                                                                test_component), symmetric=true,
                    component_aware=true)
end

# Combined stiffness + source form used by `poisson`. Equivalent to
# composing `stiffness_form` and `source_form`, but built inline so
# `poisson` can wrap one `WeakForm` rather than carrying two and
# combining them at the `Problem` level.
function _poisson_form(; source, diffusion=1)
    source_coefficient = _as_coefficient(source)
    diffusion_coefficient = _as_coefficient(diffusion)
    return WeakForm(bilinear=(q, trial, test_component) -> _stiffness_channels(diffusion_coefficient,
                                                                               q, trial,
                                                                               test_component),
                    linear=(q, test_component) -> _source_value(source_coefficient, q,
                                                                test_component), symmetric=true,
                    component_aware=true)
end

# ── Blocks, loads, and problem wrappers ───────────────────────────────────────

"""
    mass_block(u; coefficient=1)         -> BlockForm
    stiffness_block(u; diffusion=1)      -> BlockForm
    source_load(u; source)               -> LoadForm

Standard block and load contributions for [`assemble_matrix`](@ref),
[`assemble_vector`](@ref), or an explicit
`Problem((fields...); blocks, loads)`. Each wraps the corresponding
canonical [`WeakForm`](@ref) in a same-field `block(u, u, form)` or
`loadform(u, form)`.
"""
mass_block(u::Field; coefficient=1) = block(u, u, mass_form(; coefficient))
stiffness_block(u::Field; diffusion=1) = block(u, u, stiffness_form(; diffusion))
source_load(u::Field; source) = loadform(u, source_form(; source))

"""
    neumann(u::Field, value; on::BoundarySelector, component=nothing) -> LoadForm

Build the standard inhomogeneous Neumann load on a portion of the
physical boundary `∂Ω`:

    ℓ(v) = ∫_{on} value · v ds.

`u` is the test field receiving the contribution. `value` accepts a
scalar, a callback `value(x)`, or a per-component indexable value
(for vector fields). `on` is a [`BoundarySelector`](@ref) selecting
the facet — build one with [`boundary`](@ref). `component` restricts
a scalar `value` to one component of a vector field; `nothing`
(default) applies it to every component. For arbitrary per-component
values pass an `SVector` / tuple directly.

For Robin, Nitsche, or other non-canonical boundary conditions,
compose [`block`](@ref) and [`loadform`](@ref) directly with
`on = …` and your own [`WeakForm`](@ref) — the package supplies the
integration primitives, not the constitutive choice.
"""
function neumann(u::Field, value; on::BoundarySelector, component=nothing)
    component === nothing && return loadform(u, source_form(; source=value); on)
    component isa Integer || throw(ArgumentError("Neumann component must be an integer or nothing"))
    selected = Int(component)

    # Single-component variant: emit a contribution only when the
    # walking test component matches the requested one.
    coefficient = _as_coefficient(value)
    return loadform(u,
                    WeakForm(bilinear=(q, trial, c) -> 0.0,
                             linear=(q, c) -> c == selected ? _source_value(coefficient, q, c) : 0.0,
                             symmetric=true, component_aware=true); on)
end

"""
    mass(V::Space; coefficient=1, dirichlet=[])      -> Problem
    mass(u::Field; coefficient=1, dirichlet=[])      -> Problem

Build the mass-form [`Problem`](@ref) over a scalar space `V` or a
component field `u`:

    a(u, v) = ∫_Ω coefficient(x) · v u dx,    ℓ(v) = 0.

`coefficient` accepts a scalar, a callback `coefficient(x)`, or a
per-component indexable value (`SVector`/tuple). For vector fields the
form is diagonal in the component index.
`dirichlet` is the list of physical Dirichlet conditions to impose;
see [`dirichlet`](@ref).
"""
function mass(V::Space{D,T}; coefficient=one(T), dirichlet=[]) where {D,T}
    return mass(field(:u, V); coefficient, dirichlet)
end

function mass(u::Field{D,T}; coefficient=one(T), dirichlet=[]) where {D,T}
    return Problem((u,); blocks=(mass_block(u; coefficient),), dirichlet)
end

"""
    stiffness(V::Space; diffusion=1, dirichlet=[])   -> Problem
    stiffness(u::Field; diffusion=1, dirichlet=[])   -> Problem

Build the H¹ stiffness-form [`Problem`](@ref) over a scalar space `V`
or a component field `u`:

    a(u, v) = ∫_Ω ⟨∇v, A(x) ∇u⟩ dx,    ℓ(v) = 0.

For vector fields the form is diagonal in the component index.
`diffusion` accepts the same shapes as [`poisson`](@ref): scalar,
callback returning a scalar, constant `D × D` matrix, or callback
returning such a matrix.
"""
function stiffness(V::Space{D,T}; diffusion=one(T), dirichlet=[]) where {D,T}
    return stiffness(field(:u, V); diffusion, dirichlet)
end

function stiffness(u::Field{D,T}; diffusion=one(T), dirichlet=[]) where {D,T}
    return Problem((u,); blocks=(stiffness_block(u; diffusion),), dirichlet)
end

"""
    load(V::Space; source, dirichlet=[])  -> Problem
    load(u::Field; source, dirichlet=[])  -> Problem

Build a load-vector [`Problem`](@ref) over a scalar space `V` or a
component field `u`:

    a(u, v) = 0,    ℓ(v) = ∫_Ω source(x) · v dx.

For vector fields `source` may return a scalar (applied to all
components) or an indexable component value (`SVector`, `Tuple`).
For repeated time-dependent loads on an already prepared model,
prefer [`load_vector`](@ref) — it reuses the model's existing
integration plan and dof layout instead of building fresh ones from
the problem.
"""
load(V::Space{D,T}; source, dirichlet=[]) where {D,T} = load(field(:u, V); source, dirichlet)

function load(u::Field{D,T}; source, dirichlet=[]) where {D,T}
    return Problem((u,); loads=(source_load(u; source),), dirichlet)
end

"""
    load_vector(model; source) -> Vector

Assemble a load vector on the active field of an already prepared
`model`, reusing its dof layout, constraints, and integration plan.
`source` may be a scalar, a per-component indexable value, or a
callback `source(x)`. Equivalent to
`assemble_vector(model, source_load(default_field, source))` but
takes the default-field shortcut.

The canonical use case is time-dependent or otherwise repeated load
assembly: build the model once via [`prepare`](@ref), then call
`load_vector` once per time step with the current source.
"""
function load_vector(model::Model{D,T}; source) where {D,T}
    return assemble_vector(model, source_load(_default_field(model); source))
end

"""
    poisson(V::Space; source, diffusion=1, dirichlet=[])  -> Problem
    poisson(u::Field; source, diffusion=1, dirichlet=[])  -> Problem

Build the H¹ Poisson / Laplace [`Problem`](@ref) over a scalar space
`V` or a component field `u`:

    a(u, v) = ∫_Ω ⟨∇v, A(x) ∇u⟩ dx,    ℓ(v) = ∫_Ω source(x) · v dx.

`source` accepts a scalar, a callback `source(x)`, or a per-component
indexable value (vector fields). `diffusion` accepts:

  - a scalar (isotropic constant diffusion),
  - a callback returning a scalar (isotropic spatially-varying
    diffusion),
  - a constant `D × D` matrix (anisotropic constant tensor),
  - a callback returning a `D × D` matrix-like object (anisotropic
    spatially-varying tensor).

Matrix diffusion is applied as `⟨∇v, A · ∇u⟩` per component (no
inter-component coupling); for cross-component coupling, build the
multi-field [`Problem`](@ref) explicitly.
"""
function poisson(V::Space{D,T}; source, diffusion=one(T), dirichlet=[]) where {D,T}
    return poisson(field(:u, V); source, diffusion, dirichlet)
end

function poisson(u::Field{D,T}; source, diffusion=one(T), dirichlet=[]) where {D,T}
    return Problem(u, _poisson_form(; source, diffusion); dirichlet)
end
