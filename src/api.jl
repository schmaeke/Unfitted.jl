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
# `I` and `λI`), and `D × D` `AbstractMatrix` (anisotropic tensor). A
# callback `a(x)` is not a fourth shape: `_coefficient_value` resolves it
# to one of these three before the flux is formed. Squareness is checked
# once at the call site (`_check_square_diffusion`) and agreement with the
# spatial dimension per point below; the scalar / uniform paths broadcast
# directly across the gradient SVector.
_diffusion_flux(a::Number, ugrad::SVector{D}) where {D} = a * ugrad
_diffusion_flux(a::UniformScaling, ugrad::SVector{D}) where {D} = a * ugrad

# A diffusion tensor must be square for every spatial dimension, so squareness
# is checkable the moment the form is built — before `D` is known and, more
# importantly, before assembly spawns tasks. Deep in a threaded assembly the
# `DimensionMismatch` below would reach the caller wrapped in a
# `TaskFailedException`, which buries the message; here it names the offending
# shape at the call site that supplied it. The per-point check remains as the
# backstop for a square tensor of the wrong size.
function _check_square_diffusion(a::AbstractMatrix)
    size(a, 1) == size(a, 2) ||
        throw(DimensionMismatch("diffusion tensor has size $(size(a)); expected a square matrix"))
    return nothing
end
_check_square_diffusion(_) = nothing

# The tensor is copied into an `SMatrix` before the product. A heap
# `AbstractMatrix` times a static vector takes the generic dense product,
# which allocates its result vector on every call — and this call sits once
# per quadrature point per trial function. The static product keeps the whole
# contraction in registers, and copying `D²` entries is cheaper than the
# allocation it removes. A tensor that already is an `SMatrix` is passed
# through untouched, so that spelling pays nothing for the conversion.
#
# Both products evaluate the same sum ∑ⱼ aᵢⱼ ∂ⱼu left to right, but the
# unrolled static row sums contract to FMA under optimisation while the
# generic ones do not, so the flux can differ from the generic product in the
# last few ulps.
function _diffusion_flux(a::AbstractMatrix, ugrad::SVector{D}) where {D}
    size(a) == (D, D) ||
        throw(DimensionMismatch("diffusion tensor has size $(size(a)); expected ($D, $D)"))
    return SMatrix{D,D}(a) * ugrad
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
    boundary(:all; except = (axis=d, side=s))
    boundary(axis=d, side=s)
    boundary((axis=d1, side=s1), (axis=d2, side=s2), ...)

Select a physical facet for boundary conditions:

  - `boundary(:all)` — the whole physical boundary `∂Ω` (union of all
    codim-1 faces).
  - `boundary(:all; except = …)` — the union of all codim-1 faces *but*
    the ones listed. An entry is `(axis=d, side=s)` for a single face or
    `(axis=d,)` for both faces of axis `d`; pass one entry, or a tuple or
    vector of them. `except` is accepted on `:all` only, and it may not
    name every face — that would select nothing.
  - `boundary(axis=d, side=s)` — one codim-1 face. `side` is `:lower`
    or `:upper`.
  - `boundary((axis=d₁, side=s₁), …)` — the intersection of `K`
    codim-1 faces, i.e. a codim-`K` facet. Two pairs in 2D pick a
    corner; two pairs in 3D pick an edge; three pairs in 3D pick a
    vertex. Each axis may appear at most once, and `side` must be
    `:lower` or `:upper`.

Several faces therefore mean two different things depending on where
they are written, and the difference is the whole reason `except` exists:
in the positional form they *intersect* — `boundary((axis=1, side=:lower),
(axis=2, side=:lower))` is the corner where two faces meet, not the pair
of them — while in `except` they are removed from a union. On a box every
face subset is the complement of another, so `except` reaches all of
them, and an axis may be named twice there (`(axis=3, side=:lower)` and
`(axis=3, side=:upper)` together) where the positional form rejects it.

The dimension-independent Dirichlet projection treats every facet
through one code path: a codim-`K` facet is integrated against the
`(D − K)`-dim quadrature on the facet, which for `K = D` collapses to
a single point evaluation with unit weight.

Exclusions are **closed-face** exclusions. The kept faces are taken with
their closure, so a dof on the seam where a kept face meets an excluded
one is *still* constrained, and `boundary(:all; except = …)` picks out
exactly the dof set the remaining faces spelled out one by one do. That
is right for a Dirichlet trace — the datum on a kept face extends to that
face's own edge, and the hand-written union behaves identically — and it
is a genuine surprise for a flux: an `except` selector is not "`∂Ω` minus
a face" in the sense of measure, it is the closed union of what remains,
and [`neumann`](@ref) integrates over the kept faces with no special
treatment of the edges they share with the excluded ones.

A selector names **grid-aligned** geometry and nothing else. Its facets
are faces of the background box, partitioned over the cells that touch
them — but what is *integrated* over them is only the part inside `Ω`.
On an immersed space the rule on a face `∂Ω` crosses is the finite-cell
moment fit applied to the level set restricted to that face's own affine
slice, so `boundary_integral(q -> 1.0; on=boundary(axis=d, side=s))`
returns the measure of `face ∩ Ω`. A cell lying *entirely* outside `Ω` is
also masked inactive and parents nothing, so a face loses whole cells
through the mask and the rest of its non-physical area through the rule.
The `cut_facet_region_count` field of [`AssemblyDiagnostics`](@ref) counts
how many resolved facet regions needed that trimming; on a straight cut the
trimmed rule is machine-exact, on a curved one accurate rather than exact,
and `facet_moment_fit_residual_max` reports how well the fits went.

What a selector does **not** name is the immersed boundary itself. `∂Ω`'s
grid-aligned part and its immersed part are disjoint and together make up
`∂Ω`, so integrate the immersed part over a [`BoundaryMesh`](@ref) and add
the two — see [`boundary_integral`](@ref), which states the closure
identity the pair satisfies.
"""
function boundary(selector::Symbol; except=())
    # The no-exclusion path is the original one-line method, and it is reached
    # before every other check in this method so that it stays exactly that: any
    # symbol still constructs the selector it always constructed, and one that
    # no consumer supports still fails where it always failed — at prepare time,
    # in `_facets` / `_selector_matches`, not here.
    isempty(except) && return BoundarySelector(selector, Tuple{Int,Symbol}[])
    selector === :all ||
        throw(ArgumentError("except= removes faces from the whole boundary, so it applies " *
                            "only to boundary(:all); got boundary(:$selector)"))
    return BoundarySelector(:except, _excluded_faces(except))
end

# Parse `boundary`'s `except` argument into the excluded codim-1 faces stored on
# an `:except` selector. One named tuple is accepted directly, so the common
# single-exclusion case needs no wrapping brackets.
#
# This is deliberately *not* `boundary(pairs...)`'s parser. That one rejects a
# repeated axis, which is right for an intersection — `(axis=3, side=:lower)`
# together with `(axis=3, side=:upper)` names an empty facet — and wrong for an
# exclusion list, where the same pair is the perfectly ordinary request to drop
# both faces of axis 3. Here a repeated axis, and even a repeated face, is
# admissible: the list is a set of faces to remove, so duplicates are dropped
# rather than diagnosed.
#
# Unknown keys *are* rejected, which `boundary(pairs...)` has no need to do. The
# reason is the optional `side`: a typo such as `(axis=1, sides=:lower)` would
# otherwise read as `(axis=1,)` and silently exclude one face too many.
#
# The result is canonicalised — deduplicated, then sorted — so that two spellings
# of the same exclusion set are one selector *value*: `(axis=2,)` and its two
# faces written out are then a single facet-region cache entry rather than two
# resolutions of the same faces, and the whole list is a set, order and all. The
# order the *kept* faces come out in is `_kept_faces`' business, not this one's.
_excluded_faces(spec::NamedTuple) = _excluded_faces((spec,))

function _excluded_faces(specs)
    faces = Tuple{Int,Symbol}[]
    for spec in specs
        spec isa NamedTuple && haskey(spec, :axis) && issubset(keys(spec), (:axis, :side)) ||
            throw(ArgumentError("boundary except entries must be named tuples (axis=d, side=s) " *
                                "or (axis=d,); got $spec"))
        axis = Int(spec.axis)
        axis >= 1 || throw(ArgumentError("boundary axis $axis must be positive"))
        if haskey(spec, :side)
            spec.side in (:lower, :upper) ||
                throw(ArgumentError("boundary side must be :lower or :upper"))
            push!(faces, (axis, spec.side))
        else
            append!(faces, ((axis, :lower), (axis, :upper)))
        end
    end
    return sort!(unique!(faces))
end

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
    An index outside `1:components` raises `ArgumentError` as soon as the
    owning field is known — here when `field` is a [`Field`](@ref),
    otherwise at [`Problem`](@ref) construction.

Nonzero `value` data are projected onto the boundary trace space
before strong elimination so the constrained dof values reflect the
correct trace. Homogeneous (constant-zero) data skip the projection
entirely.

On an immersed space, that projection is taken over `face ∩ Ω`: where `∂Ω`
cuts the selected face, the mass matrix `∫ φᵢ φⱼ ds` and the right-hand
side `∫ g φᵢ ds` are both integrated with the trimmed rule
(see [`boundary`](@ref)), so `value` is only ever evaluated at points
inside `Ω` and the fitted trace is the least-squares fit over the physical
face. A datum defined only on `Ω` — a square root, a reciprocal distance,
an interpolated field from a prior solution — is therefore usable on a cut
face.

Two consequences worth knowing. Trimming is a small-cut generator for the
trace mass exactly as a thin cut cell is one for the stiffness matrix: a
face whose physical sliver is tiny gives a near-singular mass, and
`min_relative_facet_measure` on [`AssemblyDiagnostics`](@ref) reports the
worst such ratio. And which dofs are *constrained* is a separate question,
decided per dof key by the grid-aligned face test alone, so a dof can be
constrained while the trimmed support it would be fitted on carries no
measure at all; the trace solve then falls back from a Cholesky
factorisation to a pseudoinverse and assigns it the minimum-norm value.
`α` on the [`PhysicalDomain`](@ref) is honoured on facets and is the knob
for both — it defaults to `0`, so neither is stabilised unless asked for.

A `boundary(:all; except = …)` selector constrains its kept faces *closed*
(see [`boundary`](@ref)): a dof on the edge between a kept face and an
excluded one carries this condition, because the datum on the kept face
extends to that face's own boundary. One such condition is therefore
interchangeable with the remaining faces written out one by one — same
dofs, same projected values.
"""
function dirichlet(value; on::BoundarySelector, field=nothing, component=nothing)
    field_name = field isa Field ? field.name : field
    field_name === nothing ||
        field_name isa Symbol ||
        throw(ArgumentError("Dirichlet field must be a field object, Symbol, or nothing"))
    component === nothing ||
        component isa Integer ||
        throw(ArgumentError("Dirichlet component must be an integer or nothing"))
    field isa Field && _check_component(component, field)
    return DirichletCondition(value, on, field_name,
                              component === nothing ? nothing : Int(component))
end

# ── Canonical weak forms ──────────────────────────────────────────────────────

"""
    mass_form(; coefficient=1)

Build the mass [`WeakForm`](@ref) `∫_Ω c(x) · v u dx` without committing
to a full [`Problem`](@ref). Useful when several operators share one
prepared model — e.g. building a mass operator and a stiffness operator
over the same [`Model`](@ref) for time stepping.

`coefficient` is a scalar (the default `1`), a callback `c(x)`, or an
indexable value (`SVector`, `Tuple`) carrying one entry per component.
The form is diagonal in the component index for vector fields, and is
built with `symmetric = true` and `component_aware = true`. It declares
its `bilinear` side and no `linear` one, and an absent side is absent
rather than zero (see [`WeakForm`](@ref)), so it belongs on a
[`block`](@ref).

See [`mass_block`](@ref) for the same form wrapped as a same-field
block, and [`mass`](@ref) for the whole one-field problem.
"""
function mass_form(; coefficient=1)
    coefficient_data = _as_coefficient(coefficient)
    return WeakForm(bilinear=(q, trial, test_component) -> _mass_value(coefficient_data, q, trial,
                                                                       test_component),
                    symmetric=true, component_aware=true)
end

"""
    stiffness_form(; diffusion=1)

Build the stiffness [`WeakForm`](@ref) `∫_Ω ⟨∇v, A(x) ∇u⟩ dx` without
committing to a full [`Problem`](@ref).

`diffusion` is a scalar (isotropic; the default `1`), a `UniformScaling`
(`I`, `λI` — also isotropic), a constant `D × D` matrix (anisotropic
tensor), or a callback `A(x)` returning any of those. A constant matrix
argument is checked for squareness here, when the form is built, rather
than inside the threaded assembly where the `DimensionMismatch` would
reach the caller wrapped in a `TaskFailedException`; a square tensor
whose size does not match the space's `D` is still caught at the first
quadrature point. The form is diagonal in the component index for vector
fields, and is built with `symmetric = true` and
`component_aware = true`. It declares its `bilinear` side and no `linear`
one, and an absent side is absent rather than zero (see
[`WeakForm`](@ref)), so it belongs on a [`block`](@ref).

`A` need not be full rank. Squareness is the only structural property checked
when the form is built, and agreement with the space's `D` the only one checked
per point; nothing asks `A` to be positive definite. A tensor with a zero row
and column `d` — for a diagonal tensor simply `A[d,d] = 0` — therefore names an
axis the operator does not differentiate along, so `A = κ · diag(1, 1, 0)` on a
three-axis mesh is the directional Laplacian `∫_Ω κ (∂₁v ∂₁u + ∂₂v ∂₂u) dx`.
That covers an anisotropic medium impermeable along one axis, a plane reduction
of a higher-dimensional mesh, and the spatial part of a space-time operator,
none of which need a hand-written [`WeakForm`](@ref).

What a dropped axis costs is well-posedness, not correctness. The flux along it
is genuinely zero, so this block on its own annihilates every function that
varies only along the dropped axes, and something else has to remove that null
space: a Dirichlet condition on faces orthogonal to a *differentiated* axis, or
another term in the same rows — the ∂ₜ channel of a space-time form, say. With
`A = diag(1, 0)` on a 3 × 3 unit square at order 1, constraining the two axis-1
faces leaves a full-rank 8 × 8 block while constraining the two axis-2 faces
leaves rank 6.

That `symmetric = true` assumes a **symmetric** tensor `A`, which a
diffusivity or a conductivity is; nothing checks it. A non-symmetric `A`
still reports the form as symmetric, so a [`Problem`](@ref) built from it
inherits `symmetric = true` and assembly mirrors the lower triangle
(see [`WeakForm`](@ref)) — silently discarding the true upper triangle.
Assemble a non-symmetric tensor by declaring the asymmetry explicitly:

```julia
# at problem construction …
Problem((u,); blocks=(stiffness_block(u; diffusion=A),), symmetric=false)
# … or on a one-shot assemble against an existing model
assemble_matrix(model, stiffness_block(u; diffusion=A); symmetric=false)
```

See [`stiffness_block`](@ref) for the same form wrapped as a same-field
block, and [`stiffness`](@ref) for the whole one-field problem.
"""
function stiffness_form(; diffusion=1)
    _check_square_diffusion(diffusion)
    diffusion_coefficient = _as_coefficient(diffusion)
    return WeakForm(bilinear=(q, trial, test_component) -> _stiffness_channels(diffusion_coefficient,
                                                                               q, trial,
                                                                               test_component),
                    symmetric=true, component_aware=true)
end

"""
    source_form(; source)

Build the load [`WeakForm`](@ref) `ℓ(v) = ∫_Ω source(x) · v dx` without
committing to a full [`Problem`](@ref). It declares its `linear` side and
no `bilinear` one, so it contributes to the right-hand side only — and an
absent side is absent rather than zero (see [`WeakForm`](@ref)), so it
belongs on a [`loadform`](@ref) and not on a [`block`](@ref).

`source` is a scalar, a callback `source(x)`, or an indexable value
(`SVector`, `Tuple`) carrying one entry per component. The form is built
`component_aware = true`, and `symmetric = true` — an inert flag on a
form with no bilinear part to mirror.

See [`source_load`](@ref) for the same form wrapped as a load
contribution, [`load`](@ref) for the whole one-field problem, and
[`neumann`](@ref) for the boundary-integrated counterpart.
"""
function source_form(; source)
    source_coefficient = _as_coefficient(source)
    return WeakForm(linear=(q, test_component) -> _source_value(source_coefficient, q,
                                                                test_component), symmetric=true,
                    component_aware=true)
end

# Combined stiffness + source form used by `poisson`. Equivalent to
# composing `stiffness_form` and `source_form`, but built inline so
# `poisson` can wrap one `WeakForm` rather than carrying two and
# combining them at the `Problem` level.
function _poisson_form(; source, diffusion=1)
    _check_square_diffusion(diffusion)
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
    mass_block(u::Field; coefficient=1) -> BlockForm

Wrap [`mass_form`](@ref) as the same-field bilinear block
`block(u, u, mass_form(; coefficient))`, ready for [`assemble_matrix`](@ref)
or an explicit `Problem((fields...); blocks, loads)`. `coefficient` is
forwarded unchanged; see [`mass_form`](@ref) for the shapes it accepts.
"""
mass_block(u::Field; coefficient=1) = block(u, u, mass_form(; coefficient))

"""
    stiffness_block(u::Field; diffusion=1) -> BlockForm

Wrap [`stiffness_form`](@ref) as the same-field bilinear block
`block(u, u, stiffness_form(; diffusion))`, ready for
[`assemble_matrix`](@ref) or an explicit
`Problem((fields...); blocks, loads)`. `diffusion` is forwarded
unchanged; see [`stiffness_form`](@ref) for the shapes it accepts and for
why a non-symmetric tensor needs `symmetric = false` at the assembling
call.
"""
stiffness_block(u::Field; diffusion=1) = block(u, u, stiffness_form(; diffusion))

"""
    source_load(u::Field; source) -> LoadForm

Wrap [`source_form`](@ref) as the load contribution
`loadform(u, source_form(; source))` to the rows of `u`, ready for
[`assemble_vector`](@ref) or an explicit
`Problem((fields...); blocks, loads)`. `source` is forwarded unchanged;
see [`source_form`](@ref) for the shapes it accepts. Pass `on = …` by
building the [`loadform`](@ref) directly when the load belongs on a
boundary rather than in the volume — or use [`neumann`](@ref).
"""
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
(default) applies it to every component, and an index outside
`1:components` of `u` raises `ArgumentError`. For arbitrary
per-component values pass an `SVector` / tuple directly.

For Robin, Nitsche, or other non-canonical boundary conditions,
compose [`block`](@ref) and [`loadform`](@ref) directly with
`on = …` and your own [`WeakForm`](@ref) — the package supplies the
integration primitives, not the constitutive choice.

On an immersed space the load is integrated over `face ∩ Ω` for the face
named by `on` (see [`boundary`](@ref)), so a flux placed on a face that
`∂Ω` cuts is applied to the physical part of that face and `ℓ(v)` carries
the physical measure. The same holds for any Robin or Nitsche term composed
by hand with `on = boundary(…)`. On a straight cut the rule is
machine-exact; on a curved one it is accurate and not exact, and
`cut_facet_region_count` / `facet_moment_fit_residual_max` on
[`AssemblyDiagnostics`](@ref) say how many faces are in that regime and how
well their fits went. A flux on the immersed boundary itself belongs on a
[`BoundaryMesh`](@ref): the two parts of `∂Ω` are disjoint, so the total
flux is the sum of the two integrals.

A `boundary(:all; except = …)` selector puts the flux on its kept faces
taken *closed* (see [`boundary`](@ref)): the load is the sum over those
faces, and the edges they share with the excluded ones get no special
treatment. That is the right reading for a Dirichlet trace and a
deliberate one to check for a flux, where "`∂Ω` minus a face" might be
expected to mean something about measure. It does not; nothing here is
trimmed.
"""
function neumann(u::Field, value; on::BoundarySelector, component=nothing)
    component === nothing && return loadform(u, source_form(; source=value); on)
    component isa Integer || throw(ArgumentError("Neumann component must be an integer or nothing"))
    _check_component(component, u)
    selected = Int(component)

    # Single-component variant: emit a contribution only when the
    # walking test component matches the requested one.
    coefficient = _as_coefficient(value)
    return loadform(u,
                    WeakForm(linear=(q, c) -> c == selected ? _source_value(coefficient, q, c) : 0.0,
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
`UniformScaling`, callback returning a scalar, constant `D × D` matrix,
or callback returning such a matrix — and carries the same assumption
that a matrix `A` is symmetric, as well as the same freedom to be
rank-deficient, whose zero rows drop those axes from the operator (see
[`stiffness_form`](@ref)).
`dirichlet` is the list of physical Dirichlet conditions to impose; see
[`dirichlet`](@ref).
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
`dirichlet` is the list of physical Dirichlet conditions to impose; see
[`dirichlet`](@ref). For repeated time-dependent loads on an already
prepared model, prefer [`load_vector`](@ref) — it reuses the model's
existing integration plan and dof layout instead of building fresh ones
from the problem.
"""
load(V::Space{D,T}; source, dirichlet=[]) where {D,T} = load(field(:u, V); source, dirichlet)

function load(u::Field{D,T}; source, dirichlet=[]) where {D,T}
    return Problem((u,); loads=(source_load(u; source),), dirichlet)
end

"""
    load_vector(model; source) -> Vector

Assemble a load vector on the single field of an already prepared
`model`, reusing its dof layout, constraints, and integration plan.
`source` may be a scalar, a per-component indexable value, or a
callback `source(x)`. Equivalent to
`assemble_vector(model, source_load(u; source))` for the model's only
field `u`, which it looks up so the caller need not name it. A
multi-field model has no implicit field and raises `ArgumentError`;
spell the field out with [`source_load`](@ref) and
[`assemble_vector`](@ref) there.

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
  - a `UniformScaling` — `I` or `λI`, also isotropic,
  - a callback returning a scalar (isotropic spatially-varying
    diffusion),
  - a constant `D × D` matrix (anisotropic constant tensor),
  - a callback returning a `D × D` matrix-like object (anisotropic
    spatially-varying tensor).

Matrix diffusion is applied as `⟨∇v, A · ∇u⟩` per component (no
inter-component coupling); for cross-component coupling, build the
multi-field [`Problem`](@ref) explicitly. A matrix `A` need not be full
rank — a zero row and column drops that axis from the operator, which is
how a plane reduction or a medium impermeable along one axis is spelled;
see [`stiffness_form`](@ref) for what the dropped axis costs in
well-posedness.

The returned problem declares `symmetric = true`, which assumes a
symmetric `A`; a non-symmetric tensor must be assembled through an
explicitly asymmetric [`Problem`](@ref) instead — see
[`stiffness_form`](@ref). `dirichlet` is the list of physical Dirichlet
conditions to impose; see [`dirichlet`](@ref).
"""
function poisson(V::Space{D,T}; source, diffusion=one(T), dirichlet=[]) where {D,T}
    return poisson(field(:u, V); source, diffusion, dirichlet)
end

function poisson(u::Field{D,T}; source, diffusion=one(T), dirichlet=[]) where {D,T}
    return Problem(u, _poisson_form(; source, diffusion); dirichlet)
end
