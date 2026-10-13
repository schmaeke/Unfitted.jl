# Thin public-API layer over the lower-level types and forms. Functions here exist to shorten the canonical user
# workflows — building a Poisson / mass / stiffness problem, picking a physical boundary, attaching Dirichlet data —
# without forcing the user to construct `WeakForm` callbacks by hand. They are deliberately compact wrappers: every
# entry point composes the geometry, basis, dof, and assembly layers below but adds no behaviour that those layers can't
# already express.

# ── Coefficients ──────────────────────────────────────────────────────────────

# A coefficient, a diffusion, a source and a Neumann value are data like a Dirichlet value: a constant — a scalar, a
# per-component tuple or array, a `D × D` tensor, a `UniformScaling` — or a callback `c(x)`. Each form's closure
# captures the value as the caller gave it and resolves it at every quadrature point through `_datum_at` and, per
# component, `_component_value` (both in `dirichlet.jl`, beside the Dirichlet projection that reads its data the same
# way), so the resolution dispatches on the captured type at compile time. A per-component constant is checked against
# its field by `_check_per_component` wherever a builder knows the field; `mass_form` and `source_form` know none.

# Diffusion flux `a · ∇u` for the three accepted shapes of `a`: scalar (isotropic), `UniformScaling` (also isotropic;
# supports `I` and `λI`), and `D × D` `AbstractMatrix` (anisotropic tensor). A callback `a(x)` is not a fourth shape:
# `_datum_at` resolves it to one of these three before the flux is formed. Squareness is checked once at the
# call site (`_check_square_diffusion`) and agreement with the spatial dimension per point below; the scalar / uniform
# paths broadcast directly across the gradient SVector.
_diffusion_flux(a::Number, ugrad::SVector{D}) where {D} = a * ugrad
_diffusion_flux(a::UniformScaling, ugrad::SVector{D}) where {D} = a * ugrad

# A diffusion tensor must be square for every spatial dimension, so squareness is checkable the moment the form is built
# — before `D` is known and, more importantly, before assembly spawns tasks. Deep in a threaded assembly the
# `DimensionMismatch` below would reach the caller wrapped in a `TaskFailedException`, which buries the message; here it
# names the offending shape at the call site that supplied it. The per-point check remains as the backstop for a square
# tensor of the wrong size.
#
# A constant that is none of the three shapes — a vector, a tuple, an array of higher rank — has no flux at all and used
# to raise a `MethodError` for `_diffusion_flux` at every quadrature point; it is refused here for the same reason. A
# callback is not checked: only evaluating it could say what it returns.
function _check_square_diffusion(a::AbstractMatrix)
    size(a, 1) == size(a, 2) ||
        throw(DimensionMismatch("diffusion tensor has size $(size(a)); expected a square matrix"))
    return nothing
end
function _check_square_diffusion(a::Union{Tuple,AbstractArray})
    throw(ArgumentError("a constant diffusion is a scalar, a UniformScaling or a square matrix; got a " *
                        "$(typeof(a))"))
end
_check_square_diffusion(_) = nothing

# The tensor is copied into an `SMatrix` before the product. A heap `AbstractMatrix` times a static vector takes the
# generic dense product, which allocates its result vector on every call — and this call sits once per quadrature point
# per trial function. The static product keeps the whole contraction in registers, and copying `D²` entries is cheaper
# than the allocation it removes. A tensor that already is an `SMatrix` is passed through untouched, so that spelling
# pays nothing for the conversion.
#
# Both products evaluate the same sum ∑ⱼ aᵢⱼ ∂ⱼu left to right, but the unrolled static row sums contract to FMA under
# optimisation while the generic ones do not, so the flux can differ from the generic product in the last few ulps.
function _diffusion_flux(a::AbstractMatrix, ugrad::SVector{D}) where {D}
    size(a) == (D, D) || throw(DimensionMismatch("diffusion tensor has size $(size(a)); expected ($D, $D)"))
    return SMatrix{D,D}(a) * ugrad
end

# The three per-point helpers below — `_stiffness_channels`, `_source_value` and `_mass_value` — receive the datum as
# the form's closure captured it and only pass it on to `_datum_at`; none of them calls it. Julia does not specialise a
# method on a `Function` argument that its body merely passes along, so a callback datum would run each helper as one
# `::Function` instance, and wherever the closure does not inline the helper — a matrix-valued diffusion callback in
# 3D, for one — `_datum_at` and everything after it would dispatch dynamically and allocate at every quadrature point
# and trial function. Each helper therefore takes the datum's type as a parameter `F`: that keeps a callback
# specialised now that no wrapper struct carries its type, and changes nothing for a constant, which is never a
# `Function`.

# Stiffness bilinear-channel kernel: returns a `TestChannels` whose gradient coefficient is `a · ∇u` (the diffusion
# flux) and whose value coefficient is zero. Off-diagonal-in-components contributions are zero (diagonal-in-components
# stiffness only).
function _stiffness_channels(diffusion::F, q, trial, test_component) where {F}
    trial.component == test_component ? TestChannels(0.0, _diffusion_flux(_datum_at(diffusion, q.x), trial.gradient)) :
    0.0
end

# Source-callback kernel: resolves the source at the quadrature point and returns its component value. Shared by
# `source_form`, and through it `_poisson_form`, and by `neumann`'s single-component form.
_source_value(source::F, q, test_component) where {F} = _component_value(_datum_at(source, q.x), test_component)

# Mass bilinear-channel kernel: the component-resolved coefficient times the trial value, diagonal in components
# (off-diagonal trial/test pairs contribute zero). Mirrors `_stiffness_channels` / `_source_value` so all three
# canonical forms resolve coefficients the same component-aware way — in particular a per-component (`SVector`/tuple)
# mass coefficient picks the slot `test_component` rather than multiplying the trial value by the whole coefficient
# vector.
function _mass_value(coefficient::F, q, trial, test_component) where {F}
    return trial.component == test_component ?
           _component_value(_datum_at(coefficient, q.x), test_component) * trial.value : 0.0
end

# ── Boundary selectors ───────────────────────────────────────────────────────

"""
    boundary(:all)
    boundary(:all; except = (axis=d, side=s))
    boundary(axis=d, side=s)
    boundary((axis=d1, side=s1), (axis=d2, side=s2), ...)

Select a physical facet for boundary conditions:

  - `boundary(:all)` — the whole physical boundary `∂Ω` (union of all codim-1 faces).
  - `boundary(:all; except = …)` — the union of all codim-1 faces *but* the ones listed. An entry is `(axis=d, side=s)`
    for a single face or `(axis=d,)` for both faces of axis `d`; pass one entry, or a tuple or vector of them. `except`
    is accepted on `:all` only, and it may not name every face — that would select nothing.
  - `boundary(axis=d, side=s)` — one codim-1 face. `side` is `:lower` or `:upper`.
  - `boundary((axis=d₁, side=s₁), …)` — the intersection of `K` codim-1 faces, i.e. a codim-`K` facet. Two pairs in 2D
    pick a corner; two pairs in 3D pick an edge; three pairs in 3D pick a vertex. Each pair carries the keys `axis` and
    `side` and no other, each axis may appear at most once, and `side` must be `:lower` or `:upper`.

`:all` is the only symbol a selector takes: a face is named by its `(axis, side)` pair, and any other symbol raises
`ArgumentError` here. So does an axis below 1, in every spelling; an axis beyond the dimension of the space is refused
when the selector meets the space, at [`prepare`](@ref).

Several faces therefore mean two different things depending on where they are written, and the difference is the whole
reason `except` exists: in the positional form they *intersect* —
`boundary((axis=1, side=:lower), (axis=2, side=:lower))` is the corner where two faces meet, not the pair of them —
while in `except` they are removed from a union. On a box every face subset is the complement of another, so `except`
reaches all of them, and an axis may be named twice there (`(axis=3, side=:lower)` and `(axis=3, side=:upper)` together)
where the positional form rejects it.

The dimension-independent Dirichlet projection treats every facet through one code path: a codim-`K` facet is integrated
against the `(D − K)`-dim quadrature on the facet, which for `K = D` collapses to a single point evaluation with unit
weight.

Exclusions are **closed-face** exclusions. The kept faces are taken with their closure, so a dof on the seam where a
kept face meets an excluded one is *still* constrained, and `boundary(:all; except = …)` picks out exactly the dof set
the remaining faces spelled out one by one do. That is right for a Dirichlet trace — the datum on a kept face extends to
that face's own edge, and the hand-written union behaves identically — and it is a genuine surprise for a flux: an
`except` selector is not "`∂Ω` minus a face" in the sense of measure, it is the closed union of what remains, and
[`neumann`](@ref) integrates over the kept faces with no special treatment of the edges they share with the excluded
ones.

A selector names **grid-aligned** geometry and nothing else. Its facets are faces of the background box, partitioned
over the cells that touch them — but what is *integrated* over them is only the part inside `Ω`. On an immersed space
the rule on a face `∂Ω` crosses is the finite-cell moment fit applied to the level set restricted to that face's own
affine slice, so under the default `α = 0` every quadrature point lies inside `Ω` and
`boundary_integral(q -> 1.0; on=boundary(axis=d, side=s))` returns the measure of `face ∩ Ω`. A cell lying *entirely*
outside `Ω` is also masked inactive and parents nothing, so a face loses whole cells through the mask and the rest of
its non-physical area through the rule. The `cut_facet_region_count` field of [`AssemblyDiagnostics`](@ref) counts how
many resolved facet regions needed that trimming, and `facet_moment_fit_residual_max` reports how well their fits went.

The trimmed rule is **machine-exact where every leaf of the level set restricted to the face is affine (planar)** — one
planar leaf, or several meeting at a corner *strictly inside* the face box, where the quadrature kernel places a
breakpoint (a face cut by two planes meeting inside it comes out exact to round-off) — and on every one-dimensional
facet (a face of a 2D domain, an edge of a 3D one), whose crossings the kernel's root finding resolves to round-off. A
single leaf creased by its own `max` / `min` of planes is not affine: the kernel places no breakpoint at its crease, so
build such a domain from one leaf per plane. A curved cut is accurate rather than exact: it converges in the moment
order and in nothing else, measured 3.0e-5 on a disc cutting one cell of a face, unmoved by `subcell_length_scale` or
`max_depth`. Read `facet_moment_fit_residual_max` as a bound on the fit and never on the kernel: on a curved cut the
moments the fit reproduces are themselves approximations, which the residual cannot see, and volume integration on the
same tree shares that ceiling.

`α` on the [`PhysicalDomain`](@ref) is honoured on a face exactly as it is in a cell. At `α > 0` a trimmed region's rule
is the blend `(1−α)·fit ∪ α·tensor`, from `∫_face α f = (1−α)∫_{face∩Ω} f + α∫_face f`, and a region whose face lies
wholly outside `Ω` keeps an α-scaled full-face rule in place of an empty one. So the points no longer lie inside `Ω`, a
unit integral reads `|face ∩ Ω| + α·|face ∖ Ω|`, and an integrand or datum undefined outside `Ω` is no longer usable on
a cut face. That is the trade `α` offers — a conditioning stabiliser for a thin physical sliver of face, paid for with
evaluations outside `Ω` — and it defaults to `0`, so nothing is blended unless asked for.

What a selector does **not** name is the immersed boundary itself. `∂Ω`'s grid-aligned part and its immersed part are
disjoint and together make up `∂Ω`, so integrate the immersed part over a [`BoundaryMesh`](@ref) and add the two — see
[`boundary_integral`](@ref), which states the closure identity the pair satisfies.
"""
function boundary(selector::Symbol; except=())
    # `:all` is the only symbol a selector is built from; a face is named by `(axis, side)`, never by a symbol. Any
    # other symbol used to construct a selector all the same, and none of them did what it seemed to: `:sides` with no
    # side listed is the empty facet, whose empty face mask every raw covers in the Dirichlet detection pass, so a
    # condition on it constrained every dof of the space, interior ones included; `:except` was an undocumented alias
    # of `:all`; and anything else failed only at `prepare`, far from the call that spelled it. A misplaced `except` is
    # diagnosed first, so its message, which names the keyword, is the one such a call gets.
    isempty(except) ||
        selector === :all ||
        throw(ArgumentError("except= removes faces from the whole boundary, so it applies " *
                            "only to boundary(:all); got boundary(:$selector)"))
    selector === :all ||
        throw(ArgumentError("boundary(:$selector) is not a selector: use boundary(:all) for the whole " *
                            "boundary, boundary(axis=d, side=s) for one face, or boundary((axis=d₁, " *
                            "side=s₁), …) for the intersection of several faces"))
    isempty(except) && return BoundarySelector(:all, Tuple{Int,Symbol}[])
    return BoundarySelector(:except, _excluded_faces(except))
end

# One codim-1 face spec of `boundary`, a named tuple `(axis=d, side=s)`, as `(axis, side)`; with `side_required =
# false`, also `(axis=d,)`, both faces of axis `d`, returned with `side === nothing`. The one parser of the positional
# pairs (`side_required = true`) and of the `except` entries (`false`), each context keeping its own messages.
#
# "Both faces" is the *absence* of the `side` key, never a value of it: a `side` that is present must be `:lower` or
# `:upper`, so `(axis=1, side=nothing)` is refused like any other side, in a pair and in an exclusion alike. Read by
# value, it used to be taken for `(axis=1,)`: an exclusion silently dropped both faces of the axis, and a pair raised an
# incidental `MethodError` instead of naming the side.
#
# Unknown keys are refused in both. The reason is the optional `side` of an exclusion: a typo such as
# `(axis=1, sides=:lower)` would otherwise read as `(axis=1,)` and silently exclude one face too many; a pair, whose
# `side` is required, used to accept such a key silently all the same. An axis below 1 names no axis of any space and is
# refused here; one beyond the dimension needs `D` and is refused when the selector meets a space (`_facets`). A pair
# used to pass an axis below 1 on to `prepare`, which refused it as out of bounds.
function _face(spec, side_required::Bool)
    if side_required    # a pair of `boundary(pairs...)`, a `NamedTuple` by dispatch
        haskey(spec, :axis) && haskey(spec, :side) ||
            throw(ArgumentError("boundary pairs must carry :axis and :side keys"))
        issubset(keys(spec), (:axis, :side)) ||
            throw(ArgumentError("boundary pairs take the keys :axis and :side only; got $spec"))
    else
        spec isa NamedTuple && haskey(spec, :axis) && issubset(keys(spec), (:axis, :side)) ||
            throw(ArgumentError("boundary except entries must be named tuples (axis=d, side=s) or (axis=d,); " *
                                "got $spec"))
    end
    axis = Int(spec.axis)
    axis >= 1 || throw(ArgumentError("boundary axis $axis must be positive"))
    haskey(spec, :side) || return axis, nothing
    spec.side in (:lower, :upper) || throw(ArgumentError("boundary side must be :lower or :upper"))
    return axis, spec.side
end

# `boundary`'s `except` argument as the excluded codim-1 faces an `:except` selector stores. One named tuple is accepted
# directly, so the common single exclusion needs no wrapping brackets.
#
# The entries are parsed as the positional pairs are, but the list is not an intersection. A pair list rejects a
# repeated axis, which is right for an intersection — `(axis=3, side=:lower)` together with `(axis=3, side=:upper)`
# names an empty facet — and wrong for an exclusion list, where the same two entries are the perfectly ordinary request
# to drop both faces of axis 3. Here a repeated axis, and even a repeated face, is admissible: the list is a set of
# faces to remove, so duplicates are dropped rather than diagnosed.
#
# The result is canonicalised — deduplicated, then sorted — so that two spellings of the same exclusion set are one
# selector *value*: `(axis=2,)` and its two faces written out are then a single facet-region cache entry rather than two
# resolutions of the same faces, and the whole list is a set, order and all. The order the *kept* faces come out in is
# `_kept_faces`' business, not this one's.
_excluded_faces(spec::NamedTuple) = _excluded_faces((spec,))

function _excluded_faces(specs)
    faces = Tuple{Int,Symbol}[]
    for spec in specs
        axis, side = _face(spec, false)
        side === nothing ? append!(faces, ((axis, :lower), (axis, :upper))) : push!(faces, (axis, side))
    end
    return sort!(unique!(faces))
end

boundary(; axis::Integer, side::Symbol) = boundary((; axis, side))

# A call without arguments dispatches to the keyword method above and raises there for the missing `axis`, so `pairs` is
# never empty here.
function boundary(pairs::NamedTuple...)
    sides = Tuple{Int,Symbol}[]
    for pair in pairs
        axis, side = _face(pair, true)
        any(s -> first(s) == axis, sides) && throw(ArgumentError("boundary axis $axis specified more than once"))
        push!(sides, (axis, side))
    end
    return BoundarySelector(:sides, sides)
end

"""
    dirichlet(value; on, field=nothing, component=nothing) -> DirichletCondition

Create a physical Dirichlet condition.

  - `value` — Dirichlet datum, applied everywhere on the selected facet: a scalar, which every constrained component
    takes; a tuple or array (`SVector`, `Tuple`, `Vector`) with exactly one entry per component of the field, entry `c`
    being component `c`'s value — under `component` too, which reads its own entry; or a callback `value(x)` evaluated
    at physical coordinates and returning either. A tuple or array of any other length raises `ArgumentError`, naming
    the field and the length it needs, as soon as the owning field is known — here when `field` is a [`Field`](@ref),
    otherwise at [`Problem`](@ref) construction — and in [`update_dirichlet!`](@ref) and a standalone
    [`dof_layout`](@ref) before either changes or builds anything. A callback's result is not checked ahead of time:
    only evaluating it could say what it returns.
  - `on::BoundarySelector` — the facet to constrain. Build via [`boundary`](@ref).
  - `field` — field to constrain. Pass `nothing` for single-field problems (the only field is constrained implicitly).
    Pass a `Symbol` or [`Field`](@ref) for multi-field problems: [`Problem`](@ref) construction raises
    `ArgumentError` for a condition without `field` on a problem of several fields, and for a name none of its fields
    carries.
  - `component` — component index for vector fields when constraining only one channel (e.g. a roller / symmetry
    boundary that pins the normal component). `nothing` (default) constrains every component. An index outside
    `1:components` raises `ArgumentError` as soon as the owning field is known — here when `field` is a [`Field`](@ref),
    otherwise at [`Problem`](@ref) construction.

Nonzero `value` data are projected onto the boundary trace space before strong elimination so the constrained dof values
reflect the correct trace. Homogeneous data — a zero number, or a tuple or array whose every entry is zero — skip the
projection entirely.

On an immersed space, that projection is taken over `face ∩ Ω`: where `∂Ω` cuts the selected face, the mass matrix
`∫ φᵢ φⱼ ds` and the right-hand side `∫ g φᵢ ds` are both integrated with the trimmed rule (see [`boundary`](@ref)), so
under the default `α = 0` `value` is only ever evaluated at points inside `Ω` and the fitted trace is the least-squares
fit over the physical face. A datum defined only on `Ω` — a square root, a reciprocal distance, an interpolated field
from a prior solution — is therefore usable on a cut face, and usable there **only at `α = 0`**: the α-FCM blend appends
full-face points, so such a datum raises on them rather than being quietly extended.

Two consequences worth knowing. Trimming is a small-cut generator for the trace mass exactly as a thin cut cell is one
for the stiffness matrix: a face whose physical sliver is tiny gives a near-singular mass, and
`min_relative_facet_measure` on [`AssemblyDiagnostics`](@ref) reports the worst such ratio. And which dofs are
*constrained* is a separate question, decided per dof key by the grid-aligned face test alone, so a dof can be
constrained while the trimmed support it would be fitted on carries no measure at all; the trace solve then falls back
from a Cholesky factorisation to a pseudoinverse and assigns it the minimum-norm value. That outcome is reported rather
than chosen — `unsupported_dirichlet_dof_count` counts those dofs, `unsupported_dirichlet_dofs` names them and
`dirichlet_trace_factors` says which branch each component's solve took, all on [`AssemblyDiagnostics`](@ref) — because
what a Dirichlet condition *ought* to mean on a dof whose volume support still reaches into `Ω` is an open question; see
[`UnsupportedDirichletDof`](@ref).
`α` on the [`PhysicalDomain`](@ref) is honoured on facets and is the knob for both — it defaults to `0`, so neither is
stabilised unless asked for, and raising it is what costs the `Ω`-only datum above.

A `boundary(:all; except = …)` selector constrains its kept faces *closed* (see [`boundary`](@ref)): a dof on the edge
between a kept face and an excluded one carries this condition, because the datum on the kept face extends to that
face's own boundary. One such condition is therefore interchangeable with the remaining faces written out one by one —
same dofs, same projected values.
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
    condition = DirichletCondition(value, on, field_name, component === nothing ? nothing : Int(component))
    field isa Field && _check_dirichlet_datum(condition, component_count(field), field.name)
    return condition
end

# ── Canonical weak forms ──────────────────────────────────────────────────────

"""
    mass_form(; coefficient=1)

Build the mass [`WeakForm`](@ref) `∫_Ω c(x) · v u dx` without committing to a full [`Problem`](@ref). Useful when
several operators share one prepared model — e.g. building a mass operator and a stiffness operator over the same
[`Model`](@ref) for time stepping.

`coefficient` is a scalar (the default `1`), a callback `c(x)`, or an indexable value (`SVector`, `Tuple`) carrying one
entry per component. The form is diagonal in the component index for vector fields, and is built with `symmetric = true`
and `component_aware = true`. It declares its `bilinear` side and no `linear` one, and an absent side is absent rather
than zero (see [`WeakForm`](@ref)), so it belongs on a [`block`](@ref).

See [`mass_block`](@ref) for the same form wrapped as a same-field block, and [`mass`](@ref) for the whole one-field
problem.
"""
function mass_form(; coefficient=1)
    return WeakForm(bilinear=(q, trial, test_component) -> _mass_value(coefficient, q, trial, test_component),
                    symmetric=true, component_aware=true)
end

"""
    stiffness_form(; diffusion=1)

Build the stiffness [`WeakForm`](@ref) `∫_Ω ⟨∇v, A(x) ∇u⟩ dx` without committing to a full [`Problem`](@ref).

`diffusion` is a scalar (isotropic; the default `1`), a `UniformScaling` (`I`, `λI` — also isotropic), a constant
`D × D` matrix (anisotropic tensor), or a callback `A(x)` returning any of those. A constant matrix argument is checked
for squareness here, when the form is built, rather than inside the threaded assembly where the `DimensionMismatch`
would reach the caller wrapped in a `TaskFailedException`; a square tensor whose size does not match the space's `D` is
still caught at the first quadrature point. Any other constant — a vector or a tuple, say — raises `ArgumentError` here
too. The form is diagonal in the component index for vector fields, and is built
with `symmetric = true` and `component_aware = true`. It declares its `bilinear` side and no `linear` one, and an absent
side is absent rather than zero (see [`WeakForm`](@ref)), so it belongs on a [`block`](@ref).

`A` need not be full rank. Squareness is the only structural property checked when the form is built, and agreement with
the space's `D` the only one checked per point; nothing asks `A` to be positive definite. A tensor with a zero row and
column `d` — for a diagonal tensor simply `A[d,d] = 0` — therefore names an axis the operator does not differentiate
along, so `A = κ · diag(1, 1, 0)` on a three-axis mesh is the directional Laplacian `∫_Ω κ (∂₁v ∂₁u + ∂₂v ∂₂u) dx`. That
covers an anisotropic medium impermeable along one axis, a plane reduction of a higher-dimensional mesh, and the spatial
part of a space-time operator, none of which need a hand-written [`WeakForm`](@ref).

What a dropped axis costs is well-posedness, not correctness. The flux along it is genuinely zero, so this block on its
own annihilates every function that varies only along the dropped axes, and something else has to remove that null
space: a Dirichlet condition on faces orthogonal to a *differentiated* axis, or another term in the same rows — the ∂ₜ
channel of a space-time form, say. With `A = diag(1, 0)` on a 3 × 3 unit square at order 1, constraining the two axis-1
faces leaves a full-rank 8 × 8 block while constraining the two axis-2 faces leaves rank 6.

That `symmetric = true` assumes a **symmetric** tensor `A`, which a diffusivity or a conductivity is; nothing checks it.
A non-symmetric `A` still reports the form as symmetric, so a [`Problem`](@ref) built from it inherits
`symmetric = true` and assembly mirrors the lower triangle (see [`WeakForm`](@ref)) — silently discarding the true upper
triangle. Assemble a non-symmetric tensor by declaring the asymmetry explicitly:

```julia
# at problem construction …
Problem((u,); blocks=(stiffness_block(u; diffusion=A),), symmetric=false)
# … or on a one-shot assemble against an existing model
assemble_matrix(model, stiffness_block(u; diffusion=A); symmetric=false)
```

See [`stiffness_block`](@ref) for the same form wrapped as a same-field block, and [`stiffness`](@ref) for the whole
one-field problem.
"""
function stiffness_form(; diffusion=1)
    _check_square_diffusion(diffusion)
    return WeakForm(bilinear=(q, trial, test_component) -> _stiffness_channels(diffusion, q, trial, test_component),
                    symmetric=true, component_aware=true)
end

"""
    source_form(; source)

Build the load [`WeakForm`](@ref) `ℓ(v) = ∫_Ω source(x) · v dx` without committing to a full [`Problem`](@ref). It
declares its `linear` side and no `bilinear` one, so it contributes to the right-hand side only — and an absent side is
absent rather than zero (see [`WeakForm`](@ref)), so it belongs on a [`loadform`](@ref) and not on a [`block`](@ref).

`source` is a scalar, a callback `source(x)`, or an indexable value (`SVector`, `Tuple`) carrying one entry per
component. The form is built `component_aware = true`, and `symmetric = true` — an inert flag on a form with no bilinear
part to mirror.

See [`source_load`](@ref) for the same form wrapped as a load contribution, which is also how the whole one-field load
problem is spelled, and [`neumann`](@ref) for the boundary-integrated counterpart.
"""
function source_form(; source)
    return WeakForm(linear=(q, test_component) -> _source_value(source, q, test_component), symmetric=true,
                    component_aware=true)
end

# The Poisson form, the stiffness form's bilinear side and the source form's linear side in one `WeakForm`, so that
# `poisson` attaches one form rather than combining two at the `Problem` level. Its callbacks are the very closures
# `stiffness_form` and `source_form` build — one code path, compiled once per coefficient type for all three — and
# `stiffness_form` checks the diffusion's shape.
function _poisson_form(; source, diffusion=1)
    return WeakForm(bilinear=stiffness_form(; diffusion).bilinear, linear=source_form(; source).linear, symmetric=true,
                    component_aware=true)
end

# ── Blocks, loads, and problem wrappers ───────────────────────────────────────

"""
    mass_block(u::Field; coefficient=1) -> BlockForm

Wrap [`mass_form`](@ref) as the same-field bilinear block `block(u, u, mass_form(; coefficient))`, ready for
[`assemble_matrix`](@ref) or an explicit `Problem((fields...); blocks, loads)`. `coefficient` is forwarded unchanged;
see [`mass_form`](@ref) for the shapes it accepts. A constant tuple or array must carry exactly one entry per component
of `u`, and any other length raises `ArgumentError` here, before anything is assembled.
"""
function mass_block(u::Field; coefficient=1)
    _check_per_component(coefficient, component_count(u), "mass coefficient", u.name)
    return block(u, u, mass_form(; coefficient))
end

"""
    stiffness_block(u::Field; diffusion=1) -> BlockForm

Wrap [`stiffness_form`](@ref) as the same-field bilinear block `block(u, u, stiffness_form(; diffusion))`, ready for
[`assemble_matrix`](@ref) or an explicit `Problem((fields...); blocks, loads)`. `diffusion` is forwarded unchanged; see
[`stiffness_form`](@ref) for the shapes it accepts and for why a non-symmetric tensor needs `symmetric = false` at the
assembling call.
"""
stiffness_block(u::Field; diffusion=1) = block(u, u, stiffness_form(; diffusion))

"""
    source_load(u::Field; source) -> LoadForm

Wrap [`source_form`](@ref) as the load contribution `loadform(u, source_form(; source))` to the rows of `u`, ready for
[`assemble_vector`](@ref) or an explicit `Problem((fields...); blocks, loads)`. `source` is forwarded unchanged; see
[`source_form`](@ref) for the shapes it accepts. A constant tuple or array must carry exactly one entry per component of
`u`, and any other length raises `ArgumentError` here, before anything is assembled. Pass `on = …` by building the
[`loadform`](@ref) directly when the load belongs on a boundary rather than in the volume — or use [`neumann`](@ref).

The whole one-field load problem `a(u, v) = 0`, `ℓ(v) = ∫_Ω source(x) · v dx` is one line:

```julia
Problem((u,); loads=(source_load(u; source),), dirichlet)
```

For repeated loads on an already prepared model, [`load_vector`](@ref) reuses its integration plan and dof layout.
"""
function source_load(u::Field; source)
    _check_per_component(source, component_count(u), "source", u.name)
    return loadform(u, source_form(; source))
end

"""
    neumann(u::Field, value; on::BoundarySelector, component=nothing) -> LoadForm

Build the standard inhomogeneous Neumann load on a portion of the physical boundary `∂Ω`:

    ℓ(v) = ∫_{on} value · v ds.

`u` is the test field receiving the contribution. `value` accepts a scalar, a callback `value(x)`, or a per-component
indexable value (for vector fields). `on` is a [`BoundarySelector`](@ref) selecting the facet — build one with
[`boundary`](@ref). `component` restricts a scalar `value` to one component of a vector field; `nothing` (default)
applies it to every component, and an index outside `1:components` of `u` raises `ArgumentError`. For arbitrary
per-component values pass an `SVector` / tuple directly; a constant tuple or array must carry exactly one entry per
component of `u`, and any other length raises `ArgumentError`.

For Robin, Nitsche, or other non-canonical boundary conditions, compose [`block`](@ref) and [`loadform`](@ref) directly
with `on = …` and your own [`WeakForm`](@ref) — the package supplies the integration primitives, not the constitutive
choice.

On an immersed space the load is integrated over `face ∩ Ω` for the face named by `on` (see [`boundary`](@ref)), so a
flux placed on a face that `∂Ω` cuts is applied to the physical part of that face and, at the default `α = 0`, `ℓ(v)`
carries the physical measure — `α > 0` blends the full face back in, as [`boundary`](@ref) states. The same holds for
any Robin or Nitsche term composed by hand with `on = boundary(…)`. How exact that rule is depends on the restricted
level set rather than on the dimension, and [`boundary`](@ref) gives the condition; `cut_facet_region_count` /
`facet_moment_fit_residual_max` on [`AssemblyDiagnostics`](@ref) say how many faces are in that regime and how well
their fits went. A flux on the immersed boundary itself belongs on a [`BoundaryMesh`](@ref): the two parts of `∂Ω` are
disjoint, so the total flux is the sum of the two integrals.

A `boundary(:all; except = …)` selector puts the flux on its kept faces taken *closed* (see [`boundary`](@ref)): the
load is the sum over those faces, and the edges they share with the excluded ones get no special treatment. That is the
right reading for a Dirichlet trace and a deliberate one to check for a flux, where "`∂Ω` minus a face" might be
expected to mean something about measure. It does not; nothing here is trimmed.
"""
function neumann(u::Field, value; on::BoundarySelector, component=nothing)
    _check_per_component(value, component_count(u), "Neumann value", u.name)
    component === nothing && return loadform(u, source_form(; source=value); on)
    component isa Integer || throw(ArgumentError("Neumann component must be an integer or nothing"))
    _check_component(component, u)
    selected = Int(component)

    # Single-component variant: emit a contribution only when the walking test component matches the requested one.
    return loadform(u,
                    WeakForm(linear=(q, c) -> c == selected ? _source_value(value, q, c) : 0.0, symmetric=true,
                             component_aware=true); on)
end

"""
    mass(V::Space; coefficient=1, dirichlet=[])      -> Problem
    mass(u::Field; coefficient=1, dirichlet=[])      -> Problem

Build the mass-form [`Problem`](@ref) over a scalar space `V` or a component field `u`:

    a(u, v) = ∫_Ω coefficient(x) · v u dx,    ℓ(v) = 0.

`coefficient` accepts a scalar, a callback `coefficient(x)`, or a per-component indexable value (`SVector`/tuple). For
vector fields the form is diagonal in the component index.
`dirichlet` is the list of physical Dirichlet conditions to impose; see [`dirichlet`](@ref).
"""
function mass(x::Union{Space,Field}; coefficient=1, dirichlet=[])
    u = _as_field(x)
    return Problem((u,); blocks=(mass_block(u; coefficient),), dirichlet)
end

"""
    stiffness(V::Space; diffusion=1, dirichlet=[])   -> Problem
    stiffness(u::Field; diffusion=1, dirichlet=[])   -> Problem

Build the H¹ stiffness-form [`Problem`](@ref) over a scalar space `V` or a component field `u`:

    a(u, v) = ∫_Ω ⟨∇v, A(x) ∇u⟩ dx,    ℓ(v) = 0.

For vector fields the form is diagonal in the component index. `diffusion` accepts the same shapes as [`poisson`](@ref):
scalar, `UniformScaling`, callback returning a scalar, constant `D × D` matrix, or callback returning such a matrix —
and carries the same assumption that a matrix `A` is symmetric, as well as the same freedom to be rank-deficient, whose
zero rows drop those axes from the operator (see [`stiffness_form`](@ref)).
`dirichlet` is the list of physical Dirichlet conditions to impose; see [`dirichlet`](@ref).
"""
function stiffness(x::Union{Space,Field}; diffusion=1, dirichlet=[])
    u = _as_field(x)
    return Problem((u,); blocks=(stiffness_block(u; diffusion),), dirichlet)
end

"""
    load_vector(model; source) -> Vector

Assemble a load vector on the single field of an already prepared `model`, reusing its dof layout, constraints, and
integration plan. `source` may be a scalar, a per-component indexable value, or a callback `source(x)`. Equivalent to
`assemble_vector(model, source_load(u; source))` for the model's only field `u`, which it looks up so the caller need
not name it. A multi-field model has no implicit field and raises `ArgumentError`; spell the field out with
[`source_load`](@ref) and [`assemble_vector`](@ref) there.

The canonical use case is time-dependent or otherwise repeated load assembly: build the model once via
[`prepare`](@ref), then call `load_vector` once per time step with the current source. A load problem built from scratch
is `Problem((u,); loads=(source_load(u; source),), dirichlet)`; see [`source_load`](@ref).
"""
load_vector(model::Model{D,T}; source) where {D,T} = assemble_vector(model, source_load(_default_field(model); source))

"""
    poisson(V::Space; source, diffusion=1, dirichlet=[])  -> Problem
    poisson(u::Field; source, diffusion=1, dirichlet=[])  -> Problem

Build the H¹ Poisson / Laplace [`Problem`](@ref) over a scalar space `V` or a component field `u`:

    a(u, v) = ∫_Ω ⟨∇v, A(x) ∇u⟩ dx,    ℓ(v) = ∫_Ω source(x) · v dx.

`source` accepts a scalar, a callback `source(x)`, or a per-component indexable value (vector fields); a constant
tuple or array must carry exactly one entry per component, and any other length raises `ArgumentError`. `diffusion`
accepts:

  - a scalar (isotropic constant diffusion),
  - a `UniformScaling` — `I` or `λI`, also isotropic,
  - a callback returning a scalar (isotropic spatially-varying diffusion),
  - a constant `D × D` matrix (anisotropic constant tensor),
  - a callback returning a `D × D` matrix-like object (anisotropic spatially-varying tensor).

Matrix diffusion is applied as `⟨∇v, A · ∇u⟩` per component (no inter-component coupling); for cross-component coupling,
build the multi-field [`Problem`](@ref) explicitly. A matrix `A` need not be full rank — a zero row and column drops
that axis from the operator, which is how a plane reduction or a medium impermeable along one axis is spelled; see
[`stiffness_form`](@ref) for what the dropped axis costs in well-posedness.

The returned problem declares `symmetric = true`, which assumes a symmetric `A`; a non-symmetric tensor must be
assembled through an explicitly asymmetric [`Problem`](@ref) instead — see [`stiffness_form`](@ref). `dirichlet` is the
list of physical Dirichlet conditions to impose; see [`dirichlet`](@ref).
"""
function poisson(x::Union{Space,Field}; source, diffusion=1, dirichlet=[])
    u = _as_field(x)
    _check_per_component(source, component_count(u), "source", u.name)
    return Problem(u, _poisson_form(; source, diffusion); dirichlet)
end
