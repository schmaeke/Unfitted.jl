# Multi-domain interface coupling: the machinery that lets two independently
# discretised subdomains exchange a weak coupling term across a shared
# interface `Γ`. Three pieces live here:
#
#   * `Interface` — the `on=` tag that names the two coupled fields and the
#     interface geometry (a user-supplied `BoundaryMesh`). It sits beside
#     `BoundarySelector` and `BoundaryMesh` as a third `BlockForm.on` kind and
#     is grouped into its own assembly pass by `_partition_forms_by_on`.
#   * `InterfaceRegion` — a two-sided integration region: shared physical
#     quadrature points / weights / per-point normals (like `SurfaceRegion`),
#     plus a *separate* covering-parent list per side, so field `a` is
#     evaluated in its own subdomain's cut cell and field `b` in its
#     subdomain's cut cell at the *same* physical point. The builder subdivides
#     `Γ` against the merged trace of *both* grids (the non-matching
#     segment-merge of the references), so one cut cell of one subdomain may
#     face several of the other.
#   * `couple` — the public verb. Following the library's "supply the
#     integration primitives, not the constitutive choice" rule, it does not
#     bake in any penalty / Nitsche / cohesive law: it takes the user's
#     bilinear interface form `a(trial, test)` and expands the jump coupling
#     `∫_Γ a([u], [v]) dΓ`, `[u] = uₐ − u_b`, into the four field blocks
#     `Kₐₐ, K_ab, K_ba, K_bb` with the exact `+ − − +` sign pattern. The
#     constitutive content (β, flux weights, cohesive traction) lives entirely
#     in `a`, written by the user from ordinary `WeakForm`s.
#
# The region assembly accessors (`_region_qpoint!`, `_region_normal`, …) and
# the `on`-resolution live with the other region kinds in `assembly.jl`; the
# per-model region cache lives on the `Model` in `model.jl`.

# ── Interface tag ─────────────────────────────────────────────────────────────

"""
    Interface(field_a::Symbol, field_b::Symbol, geometry)

The `on=` tag identifying a coupling interface between the fields named
`field_a` and `field_b`, integrated over `geometry` (a [`BoundaryMesh`](@ref):
a segment loop in 2D, a triangle surface in 3D). The interface trace is
foreign to both subdomains' grids; assembly evaluates each field in whichever
of *its own* cut cells contains each interface quadrature point.

Build one with [`interface`](@ref); attach coupling contributions with
[`couple`](@ref) or by passing `on = interface(...)` to [`block`](@ref).

The unit normal carried by `geometry` orients the interface from `field_a`'s
subdomain toward `field_b`'s; user forms read it as `q.normal`. **You own this
orientation.** For a pure jump penalty ([`couple`](@ref) with a [`WeakForm`](@ref))
the sign is irrelevant, but any normal-reading form (a Nitsche flux, a cohesive
traction) is only consistent when the mesh is oriented `field_a → field_b`: a
reversed polyline / flipped surface silently yields an inconsistent — possibly
non-coercive — system. Nothing checks it; order `geometry`'s vertices so its
normal points out of `field_a`'s subdomain.
"""
struct Interface{G}
    field_a::Symbol
    field_b::Symbol
    geometry::G
end

# Identity semantics: `couple` shares one `Interface` object across its four
# blocks, so `on`-partitioning and the `IdDict` region cache key on that object.
# Two independent `interface(...)` calls are distinct couplings even over the
# same mesh, and this also avoids hashing the (potentially large) geometry.
Base.hash(iface::Interface, h::UInt) = hash(objectid(iface), h)
Base.:(==)(a::Interface, b::Interface) = a === b

"""
    interface(uₐ::Field, u_b::Field, geometry::BoundaryMesh) -> Interface

Build an [`Interface`](@ref) `on=` tag coupling `uₐ` and `u_b` across
`geometry`. Pass it to [`block`](@ref) to attach an arbitrary two-sided
interface term, or use [`couple`](@ref) for the common jump coupling.
"""
interface(u_a::Field, u_b::Field, geometry::BoundaryMesh) = Interface(u_a.name, u_b.name, geometry)

# ── Interface region ──────────────────────────────────────────────────────────

"""
    InterfaceRegion{D,T}

One two-sided integration region on a sub-cell of a coupling interface. Mirrors
[`SurfaceRegion`](@ref) — precomputed physical quadrature points, physical-frame
weights, and a per-point normal — but carries a *separate* covering-parent list
for each side (`parents_a` in `field_a`'s subdomain, `parents_b` in `field_b`'s)
so the two fields are evaluated in their own cut cells at the shared point.
`field_a` / `field_b` are the global field indices the assembly hot loop keys on
through [`region_parents`](@ref).
"""
struct InterfaceRegion{D,T<:Real}
    field_a::Int
    field_b::Int
    parents_a::Vector{FacetParent{D,T}}
    parents_b::Vector{FacetParent{D,T}}
    points::Vector{SVector{D,T}}
    weights::Vector{T}
    normals::Vector{SVector{D,T}}
end

# Build every `InterfaceRegion` for one interface. The interface mesh is
# subdivided against the *merged* grid lines of both subdomains, so each
# sub-cell lies inside a single cut cell of each subdomain; the covering
# parents are then classified per side. A sub-cell whose midpoint has no active
# cover on one side (its trace lies outside that subdomain's active region) is
# skipped. `field_a` / `field_b` are the resolved global field indices.
function _interface_regions(iface::Interface, V_a::Space{D,T}, V_b::Space{D,T}, field_a::Int,
                            field_b::Int, tol::GeometryTolerance{T}) where {D,T}
    merged = _grid_lines_for_levels((V_a.levels..., V_b.levels...), Val(D), tol)
    subdivided = _subdivide_mesh(iface.geometry, merged, tol)
    qorder = ntuple(d -> max(_surface_quadrature_order(V_a)[d], _surface_quadrature_order(V_b)[d]),
                    D)
    return _emit_interface_regions(subdivided, V_a, V_b, field_a, field_b, qorder, tol)
end

function _emit_interface_regions(subdivided::BoundaryMesh{D,T,K}, V_a::Space{D,T}, V_b::Space{D,T},
                                 field_a::Int, field_b::Int, qorder::NTuple{D,Int},
                                 tol::GeometryTolerance{T}) where {D,T,K}
    reference_samples = _simplex_reference_quadrature(Val(K), qorder, T)
    reference_area = _reference_area(Val(K))
    regions = InterfaceRegion{D,T}[]
    for (cell_index, cell) in pairs(subdivided.cells)
        midpoint = _cell_midpoint(cell)
        parents_a = _active_cover_parents(V_a, midpoint, tol)
        parents_b = _active_cover_parents(V_b, midpoint, tol)
        (isempty(parents_a) || isempty(parents_b)) && continue

        points, weights, normals = _simplex_cell_quadrature(cell, cell_index, subdivided.normals,
                                                            reference_samples, reference_area,
                                                            Val(K))
        push!(regions,
              InterfaceRegion{D,T}(field_a, field_b, parents_a, parents_b, points, weights,
                                   normals))
    end
    return regions
end

# ── Interface-kernel sugar ────────────────────────────────────────────────────

"""
    onside(side, a, b)

Pick the value on `side` of a coupling interface: `a` when `side === :a`, `b`
when `side === :b`. Replaces the `side === :a ? a : b` ternaries an
[`InterfaceForm`](@ref) kernel writes for per-side material data. Bundle several
per-side quantities into a `NamedTuple` to pick them all at once, e.g.
`m = onside(sides.trial, (κ=κ₁, w=w₁), (κ=κ₂, w=w₂))` then `m.κ`, `m.w`.
"""
onside(side::Symbol, a, b) = side === :a ? a : b

"""
    jump_sign(side)

The sign `side` contributes to an interface jump `⟦·⟧ = (·)_a − (·)_b`: `+1` on
`:a`, `−1` on `:b`. In an [`InterfaceForm`](@ref) kernel the test/trial jumps are
`⟦v⟧ = jump_sign(sides.test)·v` and `⟦u⟧ = jump_sign(sides.trial)·u`.
"""
jump_sign(side::Symbol) = onside(side, 1, -1)

# ── couple ────────────────────────────────────────────────────────────────────

# Zero-gradient / value negation of a weak-form callback return: flips the sign
# of the emitted `TestChannels` (or bare scalar) so the off-diagonal blocks of a
# jump coupling carry `−a`. `_negate_channels` covers both the structured and
# the scalar-shorthand return conventions of `WeakForm`.
_negate_channels(c::TestChannels) = TestChannels(-c.value, -c.gradient)
_negate_channels(x::Number) = -x

# Wrap a bilinear form so its emitted channels are negated, preserving the
# component-aware arity and the symmetry / linear metadata.
function _negated_form(form::WeakForm)
    bilinear = form.component_aware ?
               ((q, trial, tc) -> _negate_channels(form.bilinear(q, trial, tc))) :
               ((q, trial) -> _negate_channels(form.bilinear(q, trial)))
    return WeakForm(; bilinear, linear=form.linear, symmetric=form.symmetric,
                    component_aware=form.component_aware)
end

"""
    InterfaceForm(kernel; symmetric=true)

A *two-sided* interface bilinear form — the general coupling mechanism the
library provides so a user can express any interface constitutive law
(weighted Nitsche, cohesive, …) without the library committing to one.

`kernel(q, sides, trial, test_component)` is evaluated per interface quadrature
point, per trial basis function, exactly like a component-aware [`WeakForm`](@ref)
bilinear callback, with three extra facts supplied:

  - `q.normal` — the interface unit normal, oriented from side `:a` toward side
    `:b` (the two fields passed to [`couple`](@ref), in order).
  - `sides::NamedTuple` — `(test, trial)`, each `:a` or `:b`, naming which
    subdomain the test and trial basis currently belong to. The four field
    blocks `Kₐₐ, K_ab, K_ba, K_bb` are assembled by calling `kernel` with the
    four `(test, trial)` combinations, so the callback branches on `sides` to
    place its consistency / adjoint / penalty contributions.
  - `test_component` — the vector component (`1:D`) of the test channel the
    return value lands on. A scalar law ignores it; a vector law (elasticity
    cohesion / traction) reads the trial's component from `trial` and emits the
    coupling for row `test_component`.

`trial` is the [`TrialChannels`](@ref) (value + physical gradient) of the
`sides.trial` field; the returned [`TestChannels`](@ref) (or bare scalar) are
the coefficients on the `sides.test` field's value and gradient. Reading the
weighted average flux `⟨σn⟩_w` therefore needs both sides' gradients, which is
why the mechanism is two-sided; a callback may also read `q.state` for a
damage-dependent (cohesive) traction.

`symmetric` declares whether the *assembled coupling* is symmetric (true for a
symmetric-variant Nitsche, false for a skew or cohesive one); it flows to the
block forms so the problem's symmetry is inferred correctly.

Mind the interface orientation (see [`Interface`](@ref)): `q.normal` points
`uₐ → u_b`, and you own that ordering.

Example — symmetric weighted-Nitsche coupling of a scalar diffusion field
(`κₐ, κ_b` the two conductivities, `wₐ + w_b = 1` the flux weights, `β` the
stabilisation), passed to `couple(uₐ, u_b, Γ, nitsche)`. Per-side material is
bundled and picked with [`onside`](@ref); jump signs come from [`jump_sign`](@ref).
The field is scalar, so `test_component` is unused:

```julia
nitsche = InterfaceForm() do q, sides, trial, _tc
    n = q.normal
    u = onside(sides.trial, (κ=κₐ, w=wₐ), (κ=κ_b, w=w_b))   # trial-side material
    v = onside(sides.test,  (κ=κₐ, w=wₐ), (κ=κ_b, w=w_b))   # test-side material
    su, sv = jump_sign(sides.trial), jump_sign(sides.test)  # signs of ⟦u⟧, ⟦v⟧
    value    = -sv * u.w * u.κ * dot(trial.gradient, n) + sv * su * β * trial.value
    gradient = -v.w * v.κ * su * trial.value .* n           # symmetric adjoint (θ = 1)
    TestChannels(value, gradient)
end
```
"""
struct InterfaceForm{K}
    kernel::K
    symmetric::Bool
end
InterfaceForm(kernel; symmetric::Bool=true) = InterfaceForm{typeof(kernel)}(kernel, symmetric)

"""
    couple(uₐ::Field, u_b::Field, Γ::BoundaryMesh, form::WeakForm) -> NTuple{4,BlockForm}

Expand the jump coupling `∫_Γ a([u], [v]) dΓ` — `[u] = uₐ − u_b` — between two
independently discretised fields into the four field blocks it populates, ready
to drop into a [`Problem`](@ref)'s `blocks`:

    a([u],[v]) = a(uₐ,vₐ) − a(uₐ,v_b) − a(u_b,vₐ) + a(u_b,v_b),

i.e. `+Kₐₐ, −K_ab, −K_ba, +K_bb`, all integrated over the shared interface `Γ`.

`form` is *your* bilinear interface form `a(trial, test)`, written from ordinary
[`WeakForm`](@ref)s — the library supplies the two-sided interface integration,
not the constitutive law. The canonical penalty coupling `∫_Γ β [u]·[v] dΓ` is
`couple(uₐ, u_b, Γ, mass_form(coefficient = β))`; a form reading `q.normal` and
`trial.gradient` gives a flux/Nitsche coupling (mind the interface orientation —
see [`Interface`](@ref)). Like every assembled bilinear form, the callback must
be **linear in `trial` at fixed `q.state`**; a jump-nonlinear cohesive law is
expressed by reading both fields' current iterate through `q.state` and returning
its secant/tangent coefficient (a Newton/fixed-point step), not by making the
callback itself nonlinear. There is no per-point interface history channel, so
genuinely path-dependent state must be carried by the caller.

For a bespoke interface term that is *not* a symmetric jump coupling, attach it
directly with `block(test, trial, form; on = interface(uₐ, u_b, Γ))`.

!!! note "Threaded assembly under code coverage"
    Interface coupling passes assemble in parallel, like every other pass. A known
    Julia `--code-coverage` + multithreading codegen artifact can, in that specific
    instrumented configuration, nondeterministically corrupt the assembled
    two-sided interface block — observed only for vector/multi-component couplings,
    only under coverage with `Threads.nthreads() ≥ 2`, and never in an ordinary
    (non-coverage) run. The corruption is a Julia-side issue, not a modelling one;
    it is not reproducible outside a coverage-instrumented process and does not
    affect real solves. If you must assemble a coupled system for bit-reproducible
    output *inside* a coverage-instrumented multithreaded process, assemble that
    step serially with `assemble!(model; threaded=false)`.
"""
function couple(u_a::Field, u_b::Field, geometry::BoundaryMesh, form::WeakForm)
    iface = interface(u_a, u_b, geometry)
    neg = _negated_form(form)
    return (block(u_a, u_a, form; on=iface), block(u_b, u_a, neg; on=iface),
            block(u_a, u_b, neg; on=iface), block(u_b, u_b, form; on=iface))
end

"""
    couple(uₐ::Field, u_b::Field, Γ::BoundaryMesh, form::InterfaceForm) -> NTuple{4,BlockForm}

Expand a two-sided [`InterfaceForm`](@ref) into the four field blocks it
populates. Each block wraps the user kernel with its fixed `(test, trial)` side
pair, so the kernel is evaluated with `sides = (test=…, trial=…)` on the shared
interface region (with `q.normal` oriented `uₐ → u_b` — you own that orientation,
see [`Interface`](@ref)). This is the general coupling path — weighted Nitsche,
cohesive, or any two-sided law; the jump-jump [`WeakForm`](@ref) overload above
is the special case for a pure penalty.
"""
function couple(u_a::Field, u_b::Field, geometry::BoundaryMesh, form::InterfaceForm)
    iface = interface(u_a, u_b, geometry)
    side_form(test_side, trial_side) = WeakForm(bilinear=(q, trial, tc) -> form.kernel(q,
                                                                                       (test=test_side,
                                                                                        trial=trial_side),
                                                                                       trial, tc),
                                                linear=(q, tc) -> 0.0, symmetric=form.symmetric,
                                                component_aware=true)
    return (block(u_a, u_a, side_form(:a, :a); on=iface),
            block(u_a, u_b, side_form(:a, :b); on=iface),
            block(u_b, u_a, side_form(:b, :a); on=iface),
            block(u_b, u_b, side_form(:b, :b); on=iface))
end
