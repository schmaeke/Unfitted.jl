# Coupled-Galerkin assembly kernel for one or more fields over a
# superposition `Space`. This file owns the hot loop and the public
# assembly API; the user-facing types (weak forms, fields, problems)
# live in `problems.jl` and the prepared-problem lifecycle (`Model`,
# `prepare`, `move!`, …) lives in `model.jl`.
#
# The strategy is symbolic + numeric sparse assembly. An assembly call
# first lists its passes (`_passes`): one integration region list each,
# with the forms integrated over it. The symbolic data — every region's
# sorted active dof list and the CSC sparsity pattern over them — is
# derived once per structure and cached on the `Model`
# (`model.assembly`). The numeric pass walks every region, evaluates
# basis values and physical gradients per parent level, accumulates the
# user's weak form into a per-task local matrix / rhs over the region's
# sorted local dof numbering, and merges each local column into its
# pattern column of the prebuilt `nzval`. A dof pair appearing in many
# regions therefore accumulates with `+=` into one slot rather than
# inflating a triplet array. The threaded driver runs the same kernel,
# parks each region's result in a private arena slice, and sums the
# arena by owned dof ranges through the same column merge, so its result
# is bit-identical to the serial walk.
#
# Two contracts shape the API:
#
#   * **Channel calculus.** User forms produce `TestChannels` (value
#     coefficient + gradient coefficient) given pointwise trial data as
#     `TrialChannels`. Assembly contracts them against test basis values
#     and gradients via `_test_contribution`. The same calculus drives
#     both bilinear contributions (`a(u,v)`) and linear contributions
#     (`ℓ(v)`).
#   * **Lifecycle.** The `Model` carrying the integration plan, dof
#     layout, diagnostics, and (lazily) the assembled matrix and right-
#     hand side is built and mutated in `model.jl`. `assemble!`
#     populates the cached matrix / rhs; mutators clear them and bump
#     `model.version` so any outstanding `Solution` is detected as
#     stale on reuse.
#
# Constrained dofs (physical Dirichlet, artificial overlay, mixed) are
# eliminated column-wise during assembly: a constrained trial column's
# stiffness contribution is moved to the right-hand side, weighted by
# the dof's stored constrained value (zero for overlay/homogeneous,
# projected datum for nonzero Dirichlet). A linear-constraint pivot of a
# basis family (a B-spline mask below maximal continuity) keeps its own
# local row and column while a region is integrated, and is condensed
# onto the active dofs of its expansion once per region.

# ── Data model ────────────────────────────────────────────────────────────────
#
# The types an assembly call works with. The cache types (`RegionList`,
# `RegionDofs`, `AssemblyPattern`, `AssemblyCache`) live in `model.jl`, because
# the `Model` holds them and that file is included first; the code that fills
# and reads them lives here.

# One pass of one assembly call: a region list, its `RegionKey`, and the blocks
# and loads integrated over it as concretely typed tuples. The per-pass drivers
# and the kernel are compiled per `Pass` type — per form tuple and region kind —
# and nothing else in an assembly call is.
struct Pass{R,B<:Tuple,L<:Tuple}
    key::RegionKey
    list::RegionList{R}
    blocks::B
    loads::L
end

# Per global level id, everything the kernel needs to evaluate that level's basis
# at a point: the family, the per-cell index sets, the nominal order, and the
# value, gradient and 1D factor buffers the evaluation writes into. The buffers
# are indexed by level id, which `prepare` made contiguous and disjoint across
# subdomains, and reused across every region a task processes; a region carries
# at most one parent per level, so they never collide within a region.
#
# Struct of arrays on purpose. Only `bases` may be abstract — on a space that
# mixes families, or B-splines of different degrees — and such a space then pays
# one dynamic basis call per parent per point, which it must. Every value and
# gradient read in the emission loop stays concrete either way. A per-level
# record bundling the same buffers would itself be abstract on a mixed space and
# make every one of those reads dynamic: measured 4.5× slower on a mixed-family
# overlay.
#
#   - `bases[i]` is narrowed by `map(identity, …)` to the eltype the space's
#     families share, `IntegratedLegendre` on the default space. An abstract
#     bank turns the per-point basis call into a dynamic dispatch that boxes its
#     `SVector` / `NTuple` / `CartesianIndex` arguments: measured at 448 B per
#     call, and 94 % of `assemble_matrix`'s allocations.
#   - `modes[i]` is the level's own minimum-rule table (`level.modes`), read by
#     parent cell as `sets[kind[cell]]`. It is the same object
#     `_build_cell_dofs!` walked to produce `cell_dofs`, taken off the level
#     rather than re-derived, because the positional pairing between a cell's
#     raw dofs and its basis values is the only link between the dof layer and
#     the basis layer, and nothing checks it.
#   - `values[i]` and `gradients[i]` are sized at the level's nominal (maximum)
#     order, so a cell at a lower per-cell order fills a prefix. See
#     `_tensor_values_grads!` in `basis.jl` for why that is transparent and why
#     the buffers are not keyed by order value.
struct BasisBank{D,T,B<:BasisFamily}
    bases::Vector{B}
    modes::Vector{CellModes{D}}
    orders::Vector{NTuple{D,Int}}
    values::Vector{Vector{T}}
    gradients::Vector{Vector{SVector{D,T}}}
    val1d::Vector{NTuple{D,Vector{T}}}
    der1d::Vector{NTuple{D,Vector{T}}}
end

# Build the bank over a problem's subdomain spaces without a type-unstable level
# concatenation. `prepare` reindexed every subdomain space into a disjoint,
# contiguous block of global level ids, so the union spans `1:N` with
# `N = Σ level_count(V)`. The bank is allocated once at size `N`, with an
# abstract family slot, filled per space through `_fill_bank!`, and then rebuilt
# around the narrowed `bases`; the only dynamic dispatch is the loop over the
# abstractly typed `spaces`, once per subdomain. A plain loop sums the level
# counts: `sum` over the untyped spaces cost 50–60 ms of first-call compile.
function BasisBank(spaces, ::Val{D}, ::Type{T}) where {D,T}
    n = 0
    for V in spaces
        n += level_count(V)
    end
    bank = BasisBank{D,T,BasisFamily}(Vector{BasisFamily}(undef, n), Vector{CellModes{D}}(undef, n),
                                      Vector{NTuple{D,Int}}(undef, n), Vector{Vector{T}}(undef, n),
                                      Vector{Vector{SVector{D,T}}}(undef, n),
                                      Vector{NTuple{D,Vector{T}}}(undef, n),
                                      Vector{NTuple{D,Vector{T}}}(undef, n))
    for V in spaces
        _fill_bank!(bank, V.levels)
    end
    narrow = map(identity, bank.bases)
    return BasisBank{D,T,eltype(narrow)}(narrow, bank.modes, bank.orders, bank.values,
                                         bank.gradients, bank.val1d, bank.der1d)
end

# Function barrier: fill the id-indexed bank from one space's concretely typed
# level `Tuple`, so every per-level field read and the `local_basis_indices` /
# `_factor_buffers` calls dispatch statically.
function _fill_bank!(bank::BasisBank{D,T}, levels::Tuple) where {D,T}
    for level in levels
        i, p = level.id, nominal_order(level)
        nbasis = length(local_basis_indices(level.basis, p, level.mode))
        bank.bases[i] = level.basis
        bank.modes[i] = _cell_locals(level)
        bank.orders[i] = p
        bank.values[i] = Vector{T}(undef, nbasis)
        bank.gradients[i] = Vector{SVector{D,T}}(undef, nbasis)
        bank.val1d[i] = _factor_buffers(p, T)
        bank.der1d[i] = _factor_buffers(p, T)
    end
    return nothing
end

"""
    AssemblyWorkspace{D,T,B}

Per-task assembly scratch, for one integration region at a time. A workspace
belongs to one task for the duration of a call and is never shared; every
per-region field is reset before it is reused.

  - `layout` — the model's [`SystemLayout`](@ref). The kernel reads the dof
    layer through it and never sees `Model{D,T,P}`, so it compiles once per
    workspace type, not once per problem type.
  - `bank` — the basis buffers by level id (`BasisBank`).
  - `parents`, `fptr` — the region frame `_frame!` builds: one
    `(level, slot offset, raw dofs)` record per (field, covering parent),
    field-major, so field `f` owns `parents[fptr[f]:fptr[f+1]-1]`. The raw dof
    vector is the layout's own `cell_dofs` entry, aliased, never copied.
  - `slots` — the slot table. Slot `s₀ + (c − 1)·length(raw) + i` of a parent
    record (component `c`, cell-local basis function `i`) holds the local index
    `1:n` of an active dof, `0` for a constrained one (Dirichlet or strongly
    eliminated), or `n + k` for the `k`-th linear-constraint pivot.
  - `dofs` — the global active ids of local indices `1:n`, ascending.
  - `pivots` — the `(field, raw, component)` of pivot slot `n + k`.
  - `branches` — every pivot's expansion, one `(k, x, w, v)` per branch in
    expansion order: pivot `k`, the branch's local index `x` (`0` when it is
    constrained, with constrained value `v`) and its weight `w`.
  - `K`, `b` — the local system, `m × m` column-major and length `m`, with
    `m = n + length(pivots)`. A pivot's row and column are condensed onto its
    branches before the system leaves the workspace.
"""
struct AssemblyWorkspace{D,T,B<:BasisFamily}
    layout::SystemLayout{D,T}
    bank::BasisBank{D,T,B}
    parents::Vector{Tuple{Int,Int,Vector{Int}}}
    fptr::Vector{Int}
    slots::Vector{Int}
    dofs::Vector{Int}
    pivots::Vector{NTuple{3,Int}}
    branches::Vector{Tuple{Int,Int,T,T}}
    K::Vector{T}
    b::Vector{T}
end

# Build a workspace from the dof layout and the subdomain spaces rather than
# from the `Model`, so that nothing in it is specialised on the problem type.
function AssemblyWorkspace(layout::SystemLayout{D,T}, spaces) where {D,T}
    return AssemblyWorkspace(layout, BasisBank(spaces, Val(D), T), Tuple{Int,Int,Vector{Int}}[],
                             zeros(Int, length(layout.fields) + 1), Int[], Int[], NTuple{3,Int}[],
                             Tuple{Int,Int,T,T}[], T[], T[])
end

# A fresh workspace for `model`. Deliberately unspecialised: it runs once per
# call, never per region, and specialising it on `Model{D,T,P}` would only
# compile it again for every problem type.
function _assembly_workspace(@nospecialize(model::Model))
    return AssemblyWorkspace(model.dofs, problem_spaces(model.problem))
end

"""
    FormState

The current solution iterate at a quadrature point, exposed as `q.state` to
weak-form callbacks when assembling with `state=`, and to the callback of
[`foreach_quadrature_point`](@ref) when walking with `state=`. Query it with
`value(q.state, field[, component])` and `field_gradient(q.state, field[,
component])`, where `field` is a field name `Symbol` or a
[`Field`](@ref).

Evaluation is eager. Every field's value and physical gradient, per
component, is computed once per quadrature point from dof values read once
per integration region, so a read is an index into a buffer and a callback
can read `q.state` as often as it likes. The values are those of the current
point and are valid during the callback only: the same object is refilled at
the next point, so keep the numbers, not the `FormState`.

On a coupled model a field is evaluated only where it lives: on its own
subdomain's regions, and on an interface on its own side. Every other field
reads zero there, even where it is defined at the point; read it from a
[`Solution`](@ref) with `value(solution, model, u, q.x)` instead.

The values carry the iterate's number type promoted with the model's: a
`BigFloat` iterate on a `Float64` model yields `BigFloat` values and
gradients. A component outside the field's range raises a `BoundsError`, and
an unknown field name an `ArgumentError`.

`FormState` is what makes the assembly path Newton-linearisation
friendly: a `bilinear` callback that wants the current iterate
`uₖ(x)` for building a tangent operator simply asks `q.state` for the
value or gradient.

`field_gradient` (rather than the natural shorter `gradient`) avoids a
name collision with `Tensors.gradient`, the automatic-differentiation
entry point of the Tensors.jl package, so the two can be loaded
together unrestricted; `value` does not collide and keeps its short
name.
"""
struct FormState{D,R,L<:SystemLayout}
    # The model's layout and the iterate, converted once per call to
    # `Vector{R}`, `R = promote_type(T, eltype(state))`; shared read-only.
    layout::L
    coefficients::Vector{R}
    # `dof_value` of every slot of the current region (`_state_region!`).
    slot_values::Vector{R}
    # Field `f`, component `c` at the current point lives at `offsets[f] + c`
    # (`_state_point!`).
    values::Vector{R}
    gradients::Vector{SVector{D,R}}
    offsets::Vector{Int}
end

# A state with private per-point buffers over the shared coefficient vector:
# one per task, since the buffers are rewritten at every point.
function FormState(layout::SystemLayout{D}, coefficients::Vector{R}) where {D,R}
    offsets = zeros(Int, length(layout.fields) + 1)
    for (f, fl) in pairs(layout.fields)
        offsets[f+1] = offsets[f] + fl.components
    end
    return FormState{D,R,typeof(layout)}(layout, coefficients, R[], zeros(R, offsets[end]),
                                         zeros(SVector{D,R}, offsets[end]), offsets)
end

# ── Public assembly API ───────────────────────────────────────────────────────

"""
    assemble(model; symmetric=model.problem.symmetric,
             threaded=Threads.nthreads() > 1, state=nothing,
             region_filter=nothing) -> (A, b)
    assemble(model, blocks, loads; symmetric=nothing,
             threaded=Threads.nthreads() > 1, state=nothing,
             region_filter=nothing) -> (A, b)

Assemble a Galerkin system on the prepared `model` and return the pair of its
sparse matrix `A::SparseMatrixCSC{T,Int}` and its right-hand side `b::Vector{T}`,
both over the model's `n =` [`active_unknowns`](@ref)`(model)` active dofs. Every
other assembly entry point wraps this one: [`assemble!`](@ref) stores the pair of
the first form on the model, and [`assemble_matrix`](@ref) and
[`assemble_vector`](@ref) each return one half of the second form's.

The first form assembles the model's own problem: its blocks and loads, by
default under the problem's `symmetric` flag. The second assembles any `blocks` (a
[`BlockForm`](@ref) or a tuple of them) and `loads` (a [`LoadForm`](@ref) or a
tuple of them), either of which may be the empty tuple `()`, on the same dof
layout, constraints and integration regions — the operator of a time step or of a
Newton iteration, say, without preparing a second model.

The two halves are consistent with each other. Constrained dofs are eliminated
column by column: with `a` the active and `c` the constrained dofs, and `g` the
constrained values (the projected Dirichlet datum; zero for homogeneous and
overlay constraints),

    A = K_aa,    b = F_a − K_ac g,

where `K` is the matrix of `blocks` and `F` the vector of `loads` over all dofs.
The lift `−K_ac g` is that of exactly the blocks in the call, so `loads = ()`
returns it alone. A linear-constraint pivot of a basis family (a masked B-spline
level below maximal continuity) is condensed onto the active dofs of its
expansion, in rows and columns alike. Without a block — none given, or none whose
target has a region — `A` is the empty `n × n` matrix.

  - `symmetric` — assemble the lower triangle only and mirror it, which halves the
    scatter and makes `A` symmetric to the bit; declared for an unsymmetric form, it
    produces a wrong matrix (see [`WeakForm`](@ref)). Defaults to the problem's flag
    in the first form and to "every block declares `symmetric = true`" in the
    second.
  - `threaded` — run the threaded driver, whose result is bit-identical to the
    serial one at any thread count.
  - `state` — a [`Solution`](@ref) or an active coefficient vector, exposed to every
    form as `q.state` and read through [`value`](@ref) and
    [`field_gradient`](@ref): how a Newton tangent and residual see the current
    iterate. With the default `nothing`, a form that reads `q.state` raises.
  - `region_filter` — a predicate on integration regions; a volume region it
    rejects is skipped. It serves compactly supported loads, such as a moving point
    source, whose cost should scale with their support, and applies to volume terms
    only: boundary, surface and interface terms always integrate in full.

What assembly derives from the structure — the region lists, every region's
active dofs, the sparsity patterns — is cached on the model and reused by every
later call over the same targets, through any of the entry points, until a
mutator ([`move!`](@ref), [`activate!`](@ref), [`deactivate!`](@ref)) changes the
structure; [`update_dirichlet!`](@ref) keeps it. Several tasks may assemble on one
model at the same time.
"""
function assemble(model::Model{D,T}, @nospecialize(blocks), @nospecialize(loads); symmetric=nothing,
                  threaded::Bool=Threads.nthreads() > 1, state=nothing,
                  region_filter=nothing) where {D,T}
    matrix, rhs = _assemble(model, blocks, loads, symmetric, threaded, state, region_filter)
    n = length(rhs)
    return (matrix === nothing ? spzeros(T, n, n) : matrix)::SparseMatrixCSC{T,Int}, rhs::Vector{T}
end
function assemble(model::Model; kw...)
    problem = model.problem
    return assemble(model, problem.blocks, problem.loads; symmetric=problem.symmetric, kw...)
end

"""
    assemble_matrix(model, block_or_blocks; symmetric=nothing,
                    threaded=Threads.nthreads() > 1, state=nothing,
                    region_filter=nothing) -> SparseMatrixCSC

Assemble one or more bilinear block contributions on the prepared `model`: the
matrix half of [`assemble`](@ref)`(model, block_or_blocks, (); kw...)`, whose
docstring describes the keywords. `block_or_blocks` is a single
[`BlockForm`](@ref) or a tuple of them.

Constrained columns are eliminated, so the matrix is over the active dofs and
matches the reduced system. The right-hand-side lift that the elimination
produces (`−K_ac g` for nonzero Dirichlet data `g`) goes with the vector half;
call [`assemble`](@ref) for the consistent pair.

Does not touch `model.matrix` or `model.rhs`; for in-place assembly that updates
the cached operators use [`assemble!`](@ref).
"""
function assemble_matrix(model::Model, @nospecialize(blocks); kw...)
    first(assemble(model, blocks, (); kw...))
end

"""
    assemble_vector(model, load_or_loads; threaded=Threads.nthreads() > 1,
                    region_filter=nothing, state=nothing) -> Vector

Assemble one or more right-hand-side contributions on the prepared `model`: the
vector half of [`assemble`](@ref)`(model, (), load_or_loads; threaded,
region_filter, state)`, whose docstring describes the keywords. `load_or_loads`
is a single [`LoadForm`](@ref) or a tuple of them.

The vector carries the load integrals only. No bilinear block takes part, so
there is no Dirichlet column elimination and no lift `−K_ac g` for nonzero
Dirichlet data; [`assemble`](@ref) and [`assemble!`](@ref) return a right-hand
side consistent with a matrix. `region_filter` lets a compactly supported load (a
moving point source, say) skip the volume regions outside its support cheaply.
"""
function assemble_vector(model::Model{D,T}, @nospecialize(loads);
                         threaded::Bool=Threads.nthreads() > 1, region_filter=nothing,
                         state=nothing) where {D,T}
    # Straight to the core: `assemble` would build an empty `n × n` matrix only
    # for this wrapper to discard it.
    return last(_assemble(model, (), loads, false, threaded, state, region_filter))::Vector{T}
end

"""
    assemble!(model; threaded=Threads.nthreads() > 1) -> Model

Assemble `model`'s problem in place: store the pair
[`assemble`](@ref)`(model; threaded)` as `model.matrix` and `model.rhs`, which
[`solve!`](@ref) then reuses, and fill the assembly entries of
[`diagnostics`](@ref): `symmetry_residual` — exactly `0.0` for a problem declared
symmetric, whose matrix is mirrored to the bit, and `NaN` otherwise — and the two
condition estimates (see [`AssemblyDiagnostics`](@ref)). Returns `model` for
chaining.

Constraints are honoured through the dof layer as in [`assemble`](@ref): strong
Dirichlet elimination, which moves a constrained trial column to the right-hand
side weighted by its stored value; dof-wise homogeneous overlay elimination; and,
where a basis family supplies them, linear constraints, whose pivots are
condensed onto their expansions once per integration region.
"""
function assemble!(model::Model; threaded::Bool=Threads.nthreads() > 1)
    matrix, rhs = assemble(model; threaded)
    model.matrix, model.rhs = matrix, rhs
    diag = model.diagnostics
    diag.symmetry_residual = model.problem.symmetric ? 0.0 : NaN
    diag.condition_estimate = _condition_estimate(matrix)
    diag.scaled_condition_estimate = _scaled_condition_estimate(matrix)
    return model
end

# ── One assembly call ─────────────────────────────────────────────────────────
#
# This section and the three after it (region lists, scratch checkout, sparsity
# pattern) are an assembly call's cold path: it runs once per call or once per
# pass, never per region. Whatever takes the `Model` or the form tuples is
# deliberately unspecialised (`@nospecialize`), so it compiles once per process
# rather than once per form tuple or problem type; the form types and
# `Model{D,T,P}` reach only the per-pass drivers and the kernel. On a prototype
# of this structure, measured per new form type at `-O0` against the fully
# specialised assembler it replaced, a new block's first assembly compiled in
# 0.28× the time and the same block on a second problem type in 0.03×; the price
# is a few microseconds of dynamic dispatch per call.

# The core every assembly entry point calls: assemble `blocks` and `loads` (a
# form or a tuple of forms each) on `model` and return `(matrix, rhs)`, the matrix
# `nothing` when no pass carries a block. The rhs includes the Dirichlet lift of
# every block, so the pair is consistent. `symmetric` is a `Bool`, or `nothing`
# for "every block declares symmetry"; `state` is `nothing`, a `Solution` or a
# coefficient vector; `region_filter` applies to the volume passes only.
#
# Deliberately unspecialised, like the rest of the cold path: it compiles once
# per process, not once per form tuple or problem type. Its handful of dynamic
# calls cost a few microseconds per call, and the one that matters is the
# dispatch per pass into `_serial!` or `_threaded!`, where specialisation on the
# forms begins. The public wrappers above specialise on the `Model` alone, which
# is what types their results.
#
# The per-call scratch is checked out of the model's cache and returned in a
# `finally`, so an error a callback raises still hands it back; every
# per-region field of a workspace is reset before it is reused, and phase 2 only
# reads arena slices phase 1 wrote in the same pass.
function _assemble(@nospecialize(model::Model), @nospecialize(blocks), @nospecialize(loads),
                   @nospecialize(symmetric), threaded::Bool, @nospecialize(state),
                   @nospecialize(region_filter))
    blocks, loads = _form_tuple(blocks), _form_tuple(loads)
    symmetric = (symmetric === nothing ? all(block -> block.form.symmetric, blocks) :
                 Bool(symmetric))::Bool
    coefficients = _state_vector(state, model)
    passes = _passes(model, blocks, loads)
    tasks = threaded ? Threads.nthreads() : 1
    cache = model.assembly
    workspaces, arena = _checkout!(cache, model, tasks, threaded)
    try
        states = coefficients === nothing ? nothing :
                 [FormState(model.dofs, coefficients) for _ in 1:tasks]
        pattern = _assembly_pattern!(model, filter(pass -> !isempty(pass.blocks), passes),
                                     symmetric)
        # A threaded call reads every pass's arena layout off its region dofs and
        # sizes the one arena once, for its largest pass, which every pass then
        # reuses. Grown pass by pass instead, it was reallocated, and copied, at
        # every pass larger than all before it, each outgrown buffer dead: on a
        # cold five-pass interface problem more than the final arena (64 against
        # 51 KB). A checked-out arena already that large is reused; a shorter one
        # is replaced rather than resized, since nothing in it is read again. A
        # plain loop rather than a comprehension: a closure over the untyped
        # passes would compile once per form tuple.
        layouts = Vector{Int}[]
        if threaded
            for pass in passes
                push!(layouts, _arena_offsets(pass.list.dofs, !isempty(pass.blocks), symmetric))
            end
            entries = maximum(last, layouts; init=1) - 1
            length(arena) < entries && (arena = similar(arena, entries))
        end
        T = _scalar(model.dofs)
        nz = pattern === nothing ? nothing : zeros(T, length(pattern.rowval))
        rhs = zeros(T, active_unknowns(model.dofs))
        for (p, pass) in pairs(passes)
            # `region_filter` is a volume-pass convenience (compactly supported
            # sources); the facet, surface and interface passes ignore it.
            filter_pass = pass.key[1] === nothing ? region_filter : nothing
            if threaded
                # Every pass — volume, facet, surface, and two-sided interface —
                # runs threaded. One unexplained observation stands against that:
                # a two-sided `InterfaceRegion` pass for a vector coupling, under
                # `--code-coverage` at ≥2 threads, was nondeterministically
                # corrupted on both x86 and ARM. Never seen outside coverage, not
                # reproduced since across ~30,000 bit-exact comparisons, mechanism
                # not established (see the `couple` docstring); the parallel path
                # ships and `threaded=false` is the escape hatch.
                _threaded!(nz, rhs, pattern, pass, layouts[p], workspaces, states, filter_pass,
                           symmetric, arena)
            else
                _serial!(nz, rhs, pattern, pass, first(workspaces),
                         states === nothing ? nothing : first(states), filter_pass, symmetric)
            end
        end
        return (pattern === nothing ? nothing : _matrix_from_pattern(pattern, nz)), rhs
    finally
        _checkin!(cache, workspaces, arena)
    end
end

# Normalise a single-form / form-tuple argument to a tuple. Lets the
# user pass `assemble_matrix(model, block)` or
# `assemble_matrix(model, (block₁, block₂))` interchangeably.
_form_tuple(forms::Tuple) = forms
_form_tuple(form) = (form,)

# The scalar type `T` of a layout.
_scalar(::SystemLayout{D,T}) where {D,T} = T

# The passes of one assembly call over `blocks` and `loads`, in the order they
# are summed. Every pass accumulates into the same matrix and rhs, so this order
# is part of the result, and it is a property of the caller's form list alone:
#
#   1. the volume target first, then every `on=` target in its first appearance
#      in `(blocks..., loads...)`, matched by `isequal` — value-equal selectors
#      are one target, meshes and interfaces match by identity;
#   2. each target split per subdomain space, in `problem_spaces` order, a pass
#      taking the forms whose test field lives on that space. A region list is
#      built on one space's level block and only that space's fields evaluate on
#      it (`region_parents`), so a target named on two subdomains is two
#      independent passes, and a volume form runs on its own subdomain's plan
#      only;
#   3. except an `Interface`, which is one pass: its regions carry both sides'
#      parents, and the four blocks of a `couple` alternate their test field
#      between the two subdomains.
#
# A space with no form on a target gets no pass, and neither does a target whose
# list is empty. The sparsity pattern is built from the passes that carry a
# block, so the symbolic and the numeric side enumerate the same lists by
# construction.
function _passes(@nospecialize(model::Model), @nospecialize(blocks::Tuple),
                 @nospecialize(loads::Tuple))
    forms = (blocks..., loads...)
    targets = Any[]
    any(form -> form.on === nothing, forms) && push!(targets, nothing)
    for form in forms
        form.on === nothing || any(t -> isequal(t, form.on), targets) || push!(targets, form.on)
    end
    passes = Any[]
    for target in targets,
        space in (target isa Interface ? Any[nothing] : problem_spaces(model.problem))

        mine = form -> isequal(form.on, target) &&
                       (space === nothing || _field_space(model.problem, form.test_name) === space)
        target_blocks, target_loads = filter(mine, blocks), filter(mine, loads)
        isempty(target_blocks) && isempty(target_loads) && continue
        list = _region_list(model, target, space)
        isempty(list.regions) ||
            push!(passes, Pass((target, space), list, target_blocks, target_loads))
    end
    return passes
end

# ── Region lists ──────────────────────────────────────────────────────────────

# The `RegionList` of pass target `target` on `space` (`RegionKey`
# `(target, space)`; `space` is `nothing` for an `Interface`). Every consumer
# resolves through here — the assembly passes, `nquadpoints`,
# `foreach_quadrature_point` and `boundary_integral` — so all of them see one
# list, one set of offsets and so one `q.point` numbering. A list `prepare`
# resolved is kept for the model's lifetime; a target it did not see goes to
# the bounded one-shot cache, so a hot one is resolved once rather than on every
# call. Lookup and build run under the cache lock: a one-shot facet miss also
# extends the model's `FacetResolver` memo, which concurrent calls must not do
# at the same time, and a miss derives the list's region dofs on an idle pooled
# workspace, which no other call can take while the lock is held. With none
# idle it pools a fresh one, which the call's own checkout then takes: building
# one per list instead added 11 % to a cold interface problem's allocation.
# `Base.@lock` rather than `lock(f, l)`, whose closure would be one more type to
# compile for every caller.
function _region_list(@nospecialize(model::Model), @nospecialize(target), @nospecialize(space))
    cache = model.assembly
    Base.@lock cache.lock begin
        list = _lru_get!(cache.lists, target, space)
        list === nothing || return list
        list = _lru_get!(cache.oneshot, target, space)
        list === nothing || return list
        regions, owned = _regions(model, target, space)
        isempty(cache.workspaces) && push!(cache.workspaces, _assembly_workspace(model))
        list = RegionList(regions, last(cache.workspaces))
        entry = (target, space) => list
        owned ? push!(cache.lists, entry) : _lru_push!(cache.oneshot, entry, 8)
        return list
    end
end

# A region list with its `q.point` offsets and the active dofs of every region
# (see `RegionDofs`), all derived once, when the list is resolved. The dofs come
# from the numeric pass's own `_frame!` and `_slots!`, run on the workspace `ws`,
# so the pattern and the threaded arena layout describe exactly the dofs the
# kernel emits: active ids and the active branches of every pivot, never a
# constrained id. A plain loop rather than `cumsum` over a `map`, which cost
# 30–40 ms of first-call compile per region kind.
function RegionList(regions::Vector{R}, ws) where {R}
    offsets = Vector{Int}(undef, length(regions) + 1)
    ptr = Vector{Int}(undef, length(regions) + 1)
    offsets[1], ptr[1] = 0, 1
    val = Int[]
    for (r, region) in pairs(regions)
        offsets[r+1] = offsets[r] + _region_qpoint_count(region)
        _slots!(_frame!(ws, region))
        append!(val, ws.dofs)
        ptr[r+1] = length(val) + 1
    end
    return RegionList{R}(regions, offsets, RegionDofs(ptr, val))
end

# The regions of pass target `target` on `space`, and whether they are the
# model's own prepare-time list (`true`) or were resolved just now for a target
# `prepare` never saw (`false`). The miss paths are the builders `prepare` runs.
# "Fresh" is per target, not per face: a selector resolves through the model's
# `FacetResolver`, so a face `prepare` already resolved is reused, and a new face
# is resolved once however many selectors name it.
function _regions(@nospecialize(model::Model), ::Nothing, @nospecialize(space))
    plan = findfirst(s -> s === space, problem_spaces(model.problem))
    return model.space_plans[plan].regions, true
end
function _regions(@nospecialize(model::Model), selector::BoundarySelector, @nospecialize(space))
    cached = get(model.facet_regions, (selector, space), nothing)
    cached === nothing || return cached, true
    return _facet_regions_for_selector(space, selector, model.facet_resolver), false
end
function _regions(@nospecialize(model::Model), mesh::BoundaryMesh, @nospecialize(space))
    cached = get(model.surface_regions, (mesh, space), nothing)
    cached === nothing || return cached, true
    return _surface_regions_for_mesh(space, mesh, model.dofs.tolerance), false
end
function _regions(@nospecialize(model::Model), iface::Interface, _)
    cached = get(model.interface_regions, iface, nothing)
    cached === nothing || return cached, true
    return _interface_regions(iface, model.problem, model.dofs), false
end

# A small most-recently-used-first cache kept in a `Vector` of `key => value`
# pairs, every key a 2-tuple. `_lru_get!` returns the value stored under
# `(head, tail)` and moves its entry to the front, or returns `nothing`;
# `_lru_push!` adds an entry at the front and drops the least recently used one
# beyond `capacity`. A key matches when its `tail` is the same object (a space,
# `nothing`, or the symmetry flag) and its `head` is `isequal` (a selector by
# value, everything else by identity), exactly as `isequal` on the tuple would
# decide. A linear search is the right tool at these sizes (at most 8 entries,
# or the handful of lists `prepare` resolved), and the bound is what keeps
# one-shot targets and alternating operators from growing the cache without
# limit. An index loop on the two halves, rather than a `Dict` or `findfirst`
# with a closure over a key tuple: either compiles its hashing or its closure
# once per key type, which every new space or selector type paid on its first
# call.
function _lru_get!(lru::Vector, @nospecialize(head), @nospecialize(tail))
    for i in eachindex(lru)
        stored = first(lru[i])
        (stored[2] === tail && isequal(stored[1], head)) || continue
        i > 1 && pushfirst!(lru, popat!(lru, i))
        return last(first(lru))
    end
    return nothing
end
function _lru_push!(lru::Vector, entry::Pair, capacity::Int)
    pushfirst!(lru, entry)
    length(lru) > capacity && pop!(lru)
    return last(entry)
end

# ── Scratch checkout ──────────────────────────────────────────────────────────

# Take one call's scratch out of the model's cache: `tasks` workspaces, idle ones
# first and fresh ones for the rest, and for a threaded call the pooled arena,
# leaving an empty one behind so that a concurrent call allocates its own.
# Scratch is never shared between calls; together with the locked lookups that
# is what makes concurrent assembly on one model safe, serial or threaded. A
# workspace costs about 4 µs to build, a sizable share of a small threaded call,
# hence the pool. Returns `(workspaces, arena)`, with a concretely typed
# `workspaces` vector and the arena `nothing` for a serial call.
function _checkout!(@nospecialize(cache::AssemblyCache), @nospecialize(model::Model), tasks::Int,
                    threaded::Bool)
    idle, arena = Base.@lock cache.lock begin
        taken = Any[pop!(cache.workspaces) for _ in 1:min(tasks, length(cache.workspaces))]
        pooled = cache.arena
        threaded && (cache.arena = similar(pooled, 0))
        (taken, threaded ? pooled : nothing)
    end
    head = isempty(idle) ? _assembly_workspace(model) : idle[1]
    return _fill_checkout(head, idle, tasks, model), arena
end

# Function barrier of `_checkout!`: `head` and then the other idle workspaces,
# topped up with fresh ones to `tasks`, in a vector of `head`'s concrete type, so
# the drivers index it without dispatch.
function _fill_checkout(head::W, idle::Vector{Any}, tasks::Int,
                        @nospecialize(model::Model)) where {W}
    workspaces = Vector{W}(undef, tasks)
    workspaces[1] = head
    for t in 2:tasks
        workspaces[t] = t <= length(idle) ? idle[t]::W : _assembly_workspace(model)::W
    end
    return workspaces
end

# Return a call's scratch to the cache: the workspaces, keeping at most
# `Threads.nthreads()` idle ones, and the larger of the returned and the resident
# arena, so the pool holds the maximum over calls, never their sum. Every
# per-region field of a workspace is reset before it is used again, so scratch
# returned from a call a callback error aborted is as good as any. By index
# rather than by iteration: `workspaces` is not inferred here, and each
# iteration state would be boxed.
function _checkin!(@nospecialize(cache::AssemblyCache), @nospecialize(workspaces),
                   @nospecialize(arena))
    Base.@lock cache.lock begin
        for t in 1:min(length(workspaces), Threads.nthreads()-length(cache.workspaces))
            push!(cache.workspaces, workspaces[t])
        end
        if arena !== nothing && length(arena) > length(cache.arena)
            cache.arena = arena
        end
    end
    return nothing
end

# ── Sparsity pattern ──────────────────────────────────────────────────────────

# The sparsity pattern of the matrix passes `passes` (those carrying a block),
# from the model's pattern cache, keyed by the passes' region lists and the
# symmetry flag; `nothing` when there is no matrix pass. The key holds the lists
# themselves, so it can never alias the lists of another structure; a one-shot
# list that fell out of its cache and was resolved again misses its pattern
# once. Loads add no matrix entry and are not part of the key, so `assemble!` on
# a volume-only problem and `assemble_matrix(model, mass_block(u))` share one
# entry, and a caller that alternates operators keeps every pattern it uses (up
# to four) instead of rebuilding one per call.
function _assembly_pattern!(@nospecialize(model::Model), passes::Vector{Any}, symmetric::Bool)
    isempty(passes) && return nothing
    cache = model.assembly
    lists = RegionList[pass.list for pass in passes]
    Base.@lock cache.lock begin
        pattern = _lru_get!(cache.patterns, lists, symmetric)
        pattern === nothing || return pattern
        sets = RegionDofs[list.dofs for list in lists]
        return _lru_push!(cache.patterns,
                          (lists, symmetric) =>
                              _pattern(active_unknowns(model.dofs), sets, symmetric), 4)
    end
end

# The CSC sparsity pattern over `n` active dofs of the dense blocks on every
# region of `sets` (the matrix passes' region dofs): column `j` holds every row
# `i` that shares a region with `j` — only `i ≥ j` when `symmetric` — ascending.
#
# Gustavson's column dedup marks each row with the current column in an `O(n)`
# `marker` array, which is correct only when each output column is processed
# contiguously. So the regions are first inverted into a dof → regions index:
# the concatenated region dofs read as the CSC incidence dofs × regions, whose
# transpose lists, per dof, the regions containing it. That index costs
# `O(Σ nactive)` — one entry per (region, dof) incidence, far below the
# `O(Σ nactive²)` dense-block over-count — and the pattern itself is the only
# nnz-sized allocation.
#
# Two passes over the columns: the first counts each column's rows to build
# `colptr`, the second fills `rowval` and sorts each column (CSC wants ascending
# rows, and the scatter's merge relies on it). Counting first sizes `rowval`
# exactly; appending instead nearly doubled the transient (18.6 against
# 10.8 MiB on a 2D overlay fixture). `marker` need not be reset between columns
# within a pass — the column id strictly increases, so a stale `marker[i]` never
# equals the current one — but is cleared between the two passes.
function _pattern(n::Int, sets::Vector{RegionDofs}, symmetric::Bool)
    nregions = sum(set -> length(set.ptr) - 1, sets; init=0)
    ptr = Vector{Int}(undef, nregions + 1)
    val = Vector{Int}(undef, sum(set -> length(set.val), sets; init=0))
    ptr[1] = 1
    r = 0
    for set in sets
        copyto!(val, ptr[r+1], set.val, 1, length(set.val))
        for k in 1:(length(set.ptr)-1)
            r += 1
            ptr[r+1] = ptr[r] + (set.ptr[k+1] - set.ptr[k])
        end
    end
    owners = copy(transpose(SparseMatrixCSC(n, nregions, ptr, val, fill(true, length(val)))))
    regions = rowvals(owners)
    marker = zeros(Int, n)
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for j in 1:n
        entries = 0
        for t in nzrange(owners, j), k in ptr[regions[t]]:(ptr[regions[t]+1]-1)
            i = val[k]
            ((symmetric && i < j) || marker[i] == j) && continue
            marker[i] = j
            entries += 1
        end
        colptr[j+1] = colptr[j] + entries
    end
    rowval = Vector{Int}(undef, colptr[n+1] - 1)
    fill!(marker, 0)
    @inbounds for j in 1:n
        pos = colptr[j]
        for t in nzrange(owners, j), k in ptr[regions[t]]:(ptr[regions[t]+1]-1)
            i = val[k]
            ((symmetric && i < j) || marker[i] == j) && continue
            marker[i] = j
            rowval[pos] = i
            pos += 1
        end
        sort!(view(rowval, colptr[j]:(colptr[j+1]-1)))
    end
    return AssemblyPattern(n, colptr, rowval, symmetric)
end

# Assemble the final sparse matrix from a filled `nzval` buffer and the
# cached pattern. An unsymmetric form copies `colptr`/`rowval` so the
# cached pattern is never mutated by `dropzeros!` and takes `nzval` by
# reference (safe because every assembly call allocates its own); a
# symmetric one was scattered in the lower triangle only, and
# `_mirror_lower` builds fresh arrays for the full matrix. `dropzeros!`
# then collapses the explicit zeros left by Dirichlet column elimination
# and any structurally-present-but-untouched pattern slots.
function _matrix_from_pattern(pattern::AssemblyPattern, nzval::Vector)
    n, colptr, rowval = pattern.n, pattern.colptr, pattern.rowval
    return dropzeros!(pattern.symmetric ? _mirror_lower(n, colptr, rowval, nzval) :
                      SparseMatrixCSC(n, n, copy(colptr), copy(rowval), nzval))
end

# Mirror a symmetric form's lower-triangular CSC arrays — `rowval` holds
# only rows `i ≥ j`, which is what `_pattern` emits under
# `symmetric` — into the full symmetric matrix, in one pass.
#
# The obvious spelling, `A + Aᵀ − diag(A)`, materialises five full-size
# temporaries: the transpose, the sum, the dense diagonal, the sparse
# diagonal, and the difference. That is invisible in a wall-clock profile
# but dominates the allocation of a Newton or transient loop reassembling
# the same pattern over and over. The pass below allocates only what it
# returns.
#
# Column `j` of the result holds, in this order, the mirrors of the
# strictly-lower entries of row `j` (one per entry `(j, j′)` of `A` with
# `j′ < j`) and then column `j`'s own entries, rows `i ≥ j`, which `A`
# already stores ascending. Sweeping the source columns in increasing `j`
# and appending each entry `(i, j)` to column `j` and — when `i > j` —
# its mirror to column `i` produces exactly that order with no sort:
# every mirror written into column `i` comes from a source column `j < i`,
# so all of them land before the sweep reaches column `i`'s own entries,
# and they arrive with strictly increasing row index `j`.
#
# Values are copied, never combined, so the result matches the
# `A + Aᵀ − diag(A)` spelling to the bit: an off-diagonal entry is a single
# term either way, and the diagonal's `2d − d` is exact in IEEE arithmetic.
# The two differ only in which numerically-zero slots they store — the sum
# also materialises every diagonal slot and both triangles' explicit zeros —
# and `_matrix_from_pattern` runs `dropzeros!` over either, which removes
# exactly those.
function _mirror_lower(n::Int, colptr::Vector{Int}, rowval::Vector{Int}, nzval::Vector{T}) where {T}
    # Count first: every source entry occupies a slot in its own column, and a
    # strictly-lower one occupies a second slot in the column it mirrors into.
    # Counts land in `full_colptr[j + 1]`; the prefix sum turns them into offsets.
    full_colptr = zeros(Int, n + 1)
    @inbounds for j in 1:n, k in colptr[j]:(colptr[j+1]-1)
        full_colptr[j + 1] += 1
        rowval[k] > j && (full_colptr[rowval[k] + 1] += 1)
    end
    full_colptr[1] = 1
    @inbounds for j in 1:n
        full_colptr[j + 1] += full_colptr[j]
    end

    pos = full_colptr[1:n]  # per-column write cursor
    full_rowval = Vector{Int}(undef, full_colptr[n + 1] - 1)
    full_nzval = Vector{T}(undef, length(full_rowval))
    @inbounds for j in 1:n, k in colptr[j]:(colptr[j+1]-1)
        i = rowval[k]
        v = nzval[k]
        full_rowval[pos[j]] = i
        full_nzval[pos[j]] = v
        pos[j] += 1
        if i > j
            full_rowval[pos[i]] = j
            full_nzval[pos[i]] = v
            pos[i] += 1
        end
    end
    return SparseMatrixCSC(n, n, full_colptr, full_rowval, full_nzval)
end

# ── Serial driver and the column kernel ───────────────────────────────────────

# The serial driver of one pass: integrate every region the filter admits, in
# list order, and move each into the global system. It is specialised on the
# pass, so `_integrate!` is a static call per region, and the move is
# `_flush!`, compiled once per workspace type. A fixed region order makes
# repeated serial assembly deterministic to the bit.
function _serial!(nz, rhs, pattern, pass::Pass, ws::AssemblyWorkspace, state, region_filter,
                  symmetric::Bool)
    for (r, region) in pairs(pass.list.regions)
        region_filter === nothing || region_filter(region) || continue
        n = _integrate!(ws, pass, region, pass.list.offsets[r], state, symmetric)
        _flush!(nz, rhs, pattern, ws, n, symmetric, !isempty(pass.blocks))
    end
    return nothing
end

# Move one integrated region's local system into the global one: the rhs entry of
# every local index, in local order, then — for a pass with blocks — every local
# column through `_scatter_column!`, rows `lc:n` under a symmetric form (the
# kernel never writes above the diagonal) and `1:n` otherwise. Form-independent
# and `@noinline`, so it compiles once per workspace type, not per form tuple.
@noinline function _flush!(nz, rhs, pattern, ws::AssemblyWorkspace, n::Int, symmetric::Bool,
                           matrix::Bool)
    for k in 1:n
        rhs[ws.dofs[k]] += ws.b[k]
    end
    matrix || return nothing
    m = n + length(ws.pivots)
    for lc in 1:n
        from = symmetric ? lc : 1
        _scatter_column!(nz, pattern, ws.dofs, ws.K, (lc - 1) * m + from, lc, from)
    end
    return nothing
end

# Add one column of a region's local matrix into the global `nz`: local column
# `lc` of a region whose sorted global ids are `dofs`, its stored rows
# `from:length(dofs)` read consecutively from `src`, starting at `src[s]`. This is
# the one association of local entries with pattern slots. The serial scatter
# (`src` the workspace's `K`) and threaded phase 2 (`src` the region's arena
# slice) both go through it, which is half of why the two are bit-identical.
#
# The region's rows are an ascending subset of the pattern column's, so one
# forward merge finds every slot: measured 2.7× faster than a binary search per
# entry, and a galloping merge gained nothing (0.99–1.00×) even on the 4 488-entry
# columns of a deep 3D ladder. Zero entries are skipped, as the scatter always
# has: Dirichlet column elimination leaves explicit zeros that `dropzeros!` then
# has less to remove, and the skip keeps the sum in every slot — down to the sign
# of a zero — what it has always been.
@inline function _scatter_column!(nz, pattern::AssemblyPattern, dofs, src, s::Int, lc::Int,
                                  from::Int)
    col = dofs[lc]
    p, hi = pattern.colptr[col], pattern.colptr[col+1] - 1
    @inbounds for lr in from:length(dofs)
        e = src[s]
        s += 1
        iszero(e) && continue
        row = dofs[lr]
        while p <= hi && pattern.rowval[p] < row
            p += 1
        end
        (p <= hi && pattern.rowval[p] == row) || _pattern_miss(row, col)
        nz[p] += e
    end
    return nothing
end

# Raised when the scatter meets a `(row, col)` the cached pattern does not
# contain — impossible unless the symbolic and numeric passes disagree about a
# region's active dofs. Raising turns a latent silent corruption (writing into a
# neighbouring slot) into an immediate, locatable failure. `@noinline` keeps the
# check off the hot path's instruction stream.
@noinline function _pattern_miss(row::Int, col::Int)
    error("assembly scatter: entry ($row, $col) is absent from the cached sparsity pattern; " *
          "the symbolic and numeric passes disagree on the active dofs")
end

# ── Threaded driver: deferred compute → gather ────────────────────────────────
#
# The threaded driver splits a pass into two phases that share no output slot,
# so neither needs a lock, an atomic accumulation, or a barrier between regions:
#
#   Phase 1 — tasks take regions from an atomic counter (dynamic load balance),
#     integrate each with the serial driver's own `_integrate!`, and copy its
#     rhs and stored matrix columns into the region's own slice of a flat
#     arena. No two regions share a slice.
#   Phase 2 — each task owns a contiguous range of global dofs (balanced by
#     pattern entries for a matrix pass, even for an rhs-only one) and walks
#     every region in list order, adding the region's entries in its owned rows
#     of the rhs and its owned columns of the matrix through the serial
#     scatter's `_scatter_column!`.
#
# Every slot then has exactly one writer. That writer visits the regions in
# list order and receives each region's entry at most once, bit for bit the
# value the serial driver computed, through the same column merge with the same
# zero skip. So threaded assembly is BIT-IDENTICAL to serial — not merely
# deterministic — and independent of the thread count, which removes the
# accumulation-order roundoff a colour- or atomic-ordered scatter leaves on an
# ill-conditioned system.
#
# The arena layout is a closed form of the region dof counts (`_arena_offsets`),
# evaluated once per pass and read by writer and reader alike, so nothing is
# planned or cached per list: the region dofs (`RegionDofs`) are the whole
# symbolic input. The arena holds a pass's `Σ n` rhs entries followed by its
# packed matrix columns, `Σ n(n+1)/2` entries for a symmetric pass and `Σ n²`
# otherwise, whatever the thread count; it is sized once per call for the
# call's largest pass, which every pass then reuses, and pooled on the model
# between calls. The cost of walking every region in every phase-2 task
# measured 1.5–2 ns per region per task: nothing on 3D fixtures, 2.4 % of the
# wall time at 64 tasks on a 2D order-2 overlay, and 14 % only for 65 000
# order-1 regions at 64 tasks, where a per-task region index would be the fix.
#
# Neither phase's task count enters the result, so each is sized to its work.
# A threaded call checks out one workspace per thread, pooled on the model after
# the first call, so an idle phase-1 task costs only its spawn, about half a
# kilobyte. Phase 1 runs one task per thread, because how long a region takes
# depends on the user's callbacks, which nothing here can see, but never more
# tasks than the pass has regions. Phase 2 runs one task per `_GATHER_GRAIN`
# arena entries it gathers, at most one per thread, because there the work is
# exactly that: a merge per entry, and no callback.

# Phase-2 arena entries per task. Measured at 16 threads on a 16-core machine
# against one task per thread: 1024 or 4096 took the threaded assembly of the
# gate's small problems to 0.80–0.96× and left the large ones within 1 %;
# 16384 was 5–8 % slower on mid-sized overlays.
const _GATHER_GRAIN = 4096

# The arena layout of one threaded pass over regions `dofs`, as offsets: region
# `r`'s rhs entries sit at `arena[dofs.ptr[r] …]` and, for a matrix pass, its
# packed columns at `arena[offsets[r] + _packed(n, lc, symmetric) …]`, the
# matrix block following the `Σ n` rhs entries of the whole pass, so
# `offsets[end] - 1` is the arena length the pass needs. A function barrier:
# `_assemble` reads the dofs off an untyped pass, and the per-region sum must not
# dispatch.
function _arena_offsets(dofs::RegionDofs, matrix::Bool, symmetric::Bool)
    offsets = Vector{Int}(undef, length(dofs.ptr))
    offsets[1] = dofs.ptr[end]
    for r in 1:(length(dofs.ptr)-1)
        n = dofs.ptr[r+1] - dofs.ptr[r]
        offsets[r+1] = offsets[r] + (matrix ? _packed(n, n + 1, symmetric) : 0)
    end
    return offsets
end

# Offset of local column `lc`'s first stored entry inside a region's packed
# block, for a region with `n` local dofs: column `c` stores rows `c:n` under a
# symmetric form and `1:n` otherwise. `_packed(n, n + 1, symmetric)` is the
# block's size.
function _packed(n::Int, lc::Int, symmetric::Bool)
    return symmetric ? (lc - 1) * n - ((lc - 1) * (lc - 2)) ÷ 2 : (lc - 1) * n
end

# Run one pass threaded, accumulating into `nz` (`nothing` without a matrix) and
# `rhs`. `workspaces` and `states` hold one entry per task the call may run;
# phase 1 of this pass uses as many as it has regions. `arena` is the call's
# checked-out scratch, which `_assemble` has sized for the call's largest pass,
# and `offsets` this pass's layout in it (`_arena_offsets`). Form-independent:
# neither the pass nor the per-task scratch is specialised on, so this compiles
# once per scalar type, and the one form-specialised step, `_phase1!`, is
# reached by a single dynamic call. `done[r]` records the regions phase 1
# integrated; a region `region_filter` rejected stays `false` and phase 2 skips
# it, so a stale arena slice is never read.
function _threaded!(nz, rhs, pattern, @nospecialize(pass::Pass), offsets::Vector{Int},
                    @nospecialize(workspaces), @nospecialize(states), @nospecialize(region_filter),
                    symmetric::Bool, arena)
    dofs = pass.list.dofs
    matrix = !isempty(pass.blocks)
    done = fill(false, length(dofs.ptr) - 1)
    _phase1!(pass, workspaces, states, dofs, arena, offsets, done, region_filter, symmetric)
    owners = clamp(cld(offsets[end] - 1, _GATHER_GRAIN), 1, Threads.nthreads())
    owned = _balanced_ranges(matrix ? pattern.colptr : (1:(length(rhs)+1)), owners)
    @sync for range in owned
        Threads.@spawn _gather!(matrix ? nz : nothing, rhs, pattern, dofs, arena, offsets, done,
                                range, symmetric)
    end
    return nothing
end

# Phase 1 of one pass: one task per workspace, but never more tasks than the
# pass has regions. Each task takes the next region from the shared counter,
# integrates it with the serial driver's own `_integrate!` and parks its local
# system in the arena (`_park!`), until the counter runs out. The only
# per-pass-type code of the threaded path; an error a callback raises in a task
# arrives wrapped (`TaskFailedException` inside a `CompositeException`), as from
# any `@sync`. It is allocation-free per region only while `workspaces` and
# `states` are concretely typed vectors, which `_checkout!` guarantees.
function _phase1!(pass::Pass, workspaces, states, dofs::RegionDofs, arena, offsets::Vector{Int},
                  done::Vector{Bool}, region_filter, symmetric::Bool)
    regions, next = pass.list.regions, Threads.Atomic{Int}(1)
    @sync for t in 1:min(length(workspaces), length(regions))
        Threads.@spawn begin
            ws, state = workspaces[t], states === nothing ? nothing : states[t]
            while (r = Threads.atomic_add!(next, 1)) <= length(regions)
                region_filter === nothing || region_filter(regions[r]) || continue
                n = _integrate!(ws, pass, regions[r], pass.list.offsets[r], state, symmetric)
                _park!(arena, ws, dofs, offsets, r, n, symmetric, !isempty(pass.blocks))
                done[r] = true
            end
        end
    end
    return nothing
end

# Check that region `r`'s fresh dofs are its cached ones, and copy its local
# system into its arena slices: the rhs to `arena[dofs.ptr[r] …]` and each stored
# column `lc` to `arena[offsets[r] + _packed(n, lc, symmetric) …]`.
# Form-independent and `@noinline`, like `_flush!`, so it compiles once per
# workspace type rather than once per pass type.
@noinline function _park!(arena, ws::AssemblyWorkspace, dofs::RegionDofs, offsets::Vector{Int},
                          r::Int, n::Int, symmetric::Bool, matrix::Bool)
    lo = dofs.ptr[r]
    (n == dofs.ptr[r+1] - lo && view(dofs.val, lo:(lo+n-1)) == ws.dofs) || _drift(r)
    copyto!(arena, lo, ws.b, 1, n)
    matrix || return nothing
    m = n + length(ws.pivots)
    for lc in 1:n
        from = symmetric ? lc : 1
        copyto!(arena, offsets[r] + _packed(n, lc, symmetric), ws.K, (lc - 1) * m + from,
                n - from + 1)
    end
    return nothing
end

# Raised by phase 1 when a region's freshly numbered dofs differ from its cached
# `RegionDofs` slice: the threaded counterpart of `_pattern_miss`, which turns a
# symbolic/numeric disagreement into an error instead of a corrupted arena.
@noinline function _drift(r::Int)
    error("threaded assembly: the active dofs of region $r differ from its cached symbolic " *
          "list; the symbolic and numeric passes disagree")
end

# The phase-2 task body for the dof range `owned`: every region phase 1 computed,
# in list order, adds its rhs entries of owned dofs and, for a matrix pass, its
# owned columns. A region's dofs are sorted, so whether it touches `owned` is one
# comparison of its first and last dof, and its owned local indices are one
# `searchsorted` range.
function _gather!(nz, rhs, pattern, dofs::RegionDofs, arena, offsets::Vector{Int},
                  done::Vector{Bool}, owned::UnitRange{Int}, symmetric::Bool)
    c0, c1 = first(owned), last(owned)
    for r in 1:(length(dofs.ptr)-1)
        lo, hi = dofs.ptr[r], dofs.ptr[r+1] - 1
        (done[r] && lo <= hi && dofs.val[lo] <= c1 && dofs.val[hi] >= c0) || continue
        region = view(dofs.val, lo:hi)
        columns = searchsortedfirst(region, c0):searchsortedlast(region, c1)
        for k in columns
            rhs[region[k]] += arena[lo+k-1]
        end
        nz === nothing && continue
        n = length(region)
        for lc in columns
            from = symmetric ? lc : 1
            _scatter_column!(nz, pattern, region, arena, offsets[r] + _packed(n, lc, symmetric), lc,
                             from)
        end
    end
    return nothing
end

# Partition the columns `1:length(ptr)-1` of a CSC pattern into `tasks`
# contiguous ranges of roughly equal entry count: range `t` ends at the first
# column through which the pattern holds `t/tasks` of its entries, found by
# bisection on `ptr`, and the last one runs to the final column. A range is
# empty when one column holds more than a task's share. Contiguous by
# construction, so the ranges own disjoint slots. An rhs-only pass, one entry
# per dof, passes `ptr = 1:(n+1)` and gets `n` split into near-equal lengths.
function _balanced_ranges(ptr::AbstractVector{Int}, tasks::Int)
    m, total = length(ptr) - 1, ptr[end] - 1
    ranges = Vector{UnitRange{Int}}(undef, tasks)
    lo = 1
    for t in 1:tasks
        hi = t == tasks ? m : min(searchsortedfirst(ptr, div(t * total, tasks) + 1) - 1, m)
        ranges[t] = lo:hi
        lo = hi + 1
    end
    return ranges
end

# ── The kernel ────────────────────────────────────────────────────────────────

# Integrate every form of `pass` (its `blocks` and `loads` tuples) over one
# integration region into the workspace's local system, and return `n`, the
# region's number of active local indices. `ws.dofs` then names the global id of
# each local index, `ws.b[1:n]` is the region's rhs and, for a pass with blocks,
# `ws.K` holds its matrix as an `m × m` column-major array whose leading `n × n`
# block is the active part (`m = n + length(ws.pivots)`). This is the assembly
# hot loop.
#
# One method serves all four region kinds; only the accessors below differ:
#
#   * a **volume** region iterates reference-frame points and scales each weight
#     by `vol(box)/2ᴰ`; `q.normal` and `q.sides` are `nothing`;
#   * a **facet** region iterates precomputed physical points and carries the
#     face's outward normal `q.normal` and codim-`K` identifier `q.sides`, so
#     forms can express Neumann, Robin and Nitsche contributions;
#   * a **surface** (immersed-boundary) region is a facet with a per-point
#     `q.normal` and no `q.sides`;
#   * an **interface** region is two-sided: it refreshes both subdomains' bases
#     at the shared point and carries the per-point normal oriented from side
#     `a` to side `b`.
#
# Every active local basis function of every parent participates. In
# particular `is_facet_basis` is not applied on the physical-frame kinds: a
# function with zero *value* on a face can carry a nonzero *gradient*, which
# Nitsche-style forms rely on.
#
# Per point the kernel refreshes the bases, evaluates the state when there is
# one, and composes `q` (`_payload!`), with `point = offset + k` the stable index
# `nquadpoints` sizes and `foreach_quadrature_point` shares. Loads are evaluated
# first, then blocks, each through `_emit!`.
# Constrained trial columns move to the rhs as they are emitted; pivot rows and
# columns are kept whole and condensed once after the last point.
#
# `@noinline` is the per-region function barrier the drivers rely on: there is
# one compiled instance per (pass type, region kind, state type), and the
# contraction's code generation, which decides how `_test_contribution`'s
# `dot` rounds, is fixed inside it whichever driver calls it.
@noinline function _integrate!(ws::AssemblyWorkspace{D,T}, pass::Pass, region, offset::Int, state,
                               sym::Bool) where {D,T}
    _frame!(ws, region)
    n = _slots!(ws)
    m = n + length(ws.pivots)
    state === nothing || _state_region!(state, ws)
    fill!(resize!(ws.K, isempty(pass.blocks) ? 0 : m * m), zero(T))
    fill!(resize!(ws.b, m), zero(T))
    jacobian = _region_jacobian(region)
    for k in 1:_region_qpoint_count(region)
        q = _payload!(ws, region, k, jacobian, offset + k, state)
        _foreach_form(_load!, pass.loads, (ws, q, n, m, sym))
        _foreach_form(_block!, pass.blocks, (ws, q, n, m, sym))
    end
    isempty(ws.pivots) || _condense!(ws, n, m, sym, !isempty(pass.blocks))
    return n
end

# Apply `f(form, args...)` to every form of the tuple `forms`, first to last,
# unrolled at compile time by recursion on `first` / `Base.tail`. The kernel
# above walks its `blocks` and `loads` through this rather than through
# `for form in forms`.
#
# Why not a `for` loop: every `WeakForm` closure is its own type, so a problem's
# `blocks` and `loads` are heterogeneous tuples. A `for` loop reads a tuple's
# elements through a runtime index, and inference types the loop variable as the
# union of the element types — but only up to three members (the compiler's
# `MAX_TYPEUNION_LENGTH`). A fourth distinct form type widens the loop variable
# to the abstract `BlockForm` / `LoadForm`, and every callback call, and
# everything downstream of its result, then dispatches dynamically and
# allocates, once per trial dof × component × quadrature point. Measured on
# Julia 1.12.5, serial: one scalar field on a 16 × 16 order-2 mesh assembled
# three distinct blocks in 1.7 ms and 1.0 MB, and four in 65 ms and 420 MB; a
# space-time Navier–Stokes Jacobian with the four blocks (u,u), (u,p), (p,u),
# (p,p) took 1.19 s and 7.5 GB, where the same four blocks assembled one at a
# time took 0.029 s. The recursion hands every call its element's concrete
# type, so the cost no longer depends on how many distinct forms a problem
# carries.
#
# Why the per-form work is a named function taking explicit arguments, and not
# a `do` block: a closure capturing the locals of the accumulation is too large
# to inline, so every one of those reads goes through the closure object, and
# that alone measured about 4 % slower on single-form Poisson assembly. `args`
# is one concrete `Tuple` rather than a vararg, because Julia may decline to
# specialise on a vararg that is only passed through.
@inline _foreach_form(f::F, ::Tuple{}, args::Tuple) where {F} = nothing
@inline function _foreach_form(f::F, forms::Tuple, args::Tuple) where {F}
    f(first(forms), args...)
    return _foreach_form(f, Base.tail(forms), args)
end

# One load at one quadrature point: its linear channels, per test component,
# emitted into the rhs. A field that owns no parent in this region returns
# before the callback runs (a foreign subdomain on a coupled model).
# Component-aware forms are called with `(q, test_component)`, the rest with
# `(q,)` and apply to every component alike.
function _load!(ld::LoadForm, ws::AssemblyWorkspace{D,T}, q, n::Int, m::Int, sym::Bool) where {D,T}
    f = _field_index(ws.layout, ld.test_name)
    ws.fptr[f] == ws.fptr[f+1] && return nothing
    for tc in 1:ws.layout.fields[f].components
        ch = _as_test_channels(ld.form.component_aware ? ld.form.linear(q, tc) : ld.form.linear(q),
                               Val(D), T)
        _emit!(ws, ch, f, tc, q.weight, 0, -one(T), n, m, sym)
    end
    return nothing
end

# One block at one quadrature point: the trial component → trial parent → trial
# function → test component nest, each channel emitted by `_emit!` over the test
# parents. The trial function's slot is read once, with its constrained value
# `g` when the column is eliminated. Component-aware forms see every
# (trial, test) component pair; component-unaware forms apply on the diagonal
# `test_component == trial_component` only and are called with `(q, trial)`. A
# block whose test field owns no parent in this region returns before the
# callback runs.
function _block!(bl::BlockForm, ws::AssemblyWorkspace{D,T}, q, n::Int, m::Int,
                 sym::Bool) where {D,T}
    tf, uf = _field_index(ws.layout, bl.test_name), _field_index(ws.layout, bl.trial_name)
    ws.fptr[tf] == ws.fptr[tf+1] && return nothing
    C, ul, aware = ws.layout.fields[tf].components, ws.layout.fields[uf], bl.form.component_aware
    for trc in 1:ul.components, pu in ws.fptr[uf]:(ws.fptr[uf+1]-1)
        l, s0, raw = ws.parents[pu]
        for j in eachindex(raw)
            col = ws.slots[s0+(trc-1)*length(raw)+j]
            g = col == 0 ? constrained_value(ul.dofs, raw[j], trc) : zero(T)
            trial = TrialChannels(trc, ws.bank.values[l][j], ws.bank.gradients[l][j])
            for tc in (aware ? (1:C) : (trc:min(trc, C)))
                ch = _as_test_channels(aware ? bl.form.bilinear(q, trial, tc) :
                                       bl.form.bilinear(q, trial), Val(D), T)
                _emit!(ws, ch, tf, tc, q.weight, col, g, n, m, sym)
            end
        end
    end
    return nothing
end

# Contract one test channel `ch` against every test function of field `f`,
# component `tc`, and emit the weighted result `e = w · ch(φ)` by the roles of
# the test row and the trial column `col`:
#
#   * a constrained test row (slot 0) is dropped;
#   * `col > 0`, an active or pivot trial column, accumulates `K[row, col] += e`;
#   * `col == 0`, a constrained trial column with value `g`, moves to the rhs as
#     `b[row] −= e·g` (column elimination). A load is emitted as this case with
#     `g = −1`, since `b − e·(−1)` is exactly `b + e` in IEEE arithmetic.
#
# Under a symmetric form an active row above an active column is skipped before
# the contraction, which halves it: sorted local numbering makes "local row <
# col" the same test as "global row < col", and the mirror rebuilds that half.
# Pivot rows and columns (`> n`) are never skipped; `_condense!` keeps the
# triangle when it folds them.
#
# The bank is read through `ws` inside the loop on purpose. Hoisting
# `ws.bank.values[l]` above it made 3D assembly 1.6× slower (8³ cells, order 2:
# 19.7 against 12.2 ms), so re-measure 3D before changing this loop's shape.
@inline function _emit!(ws::AssemblyWorkspace, ch, f::Int, tc::Int, w, col::Int, g, n::Int, m::Int,
                        sym::Bool)
    for p in ws.fptr[f]:(ws.fptr[f+1]-1)
        l, s0, raw = ws.parents[p]
        s = s0 + (tc - 1) * length(raw)
        @inbounds for a in eachindex(raw)
            row = ws.slots[s+a]
            (row == 0 || (sym && row < col <= n)) && continue
            e = w * _test_contribution(ch, ws.bank.values[l][a], ws.bank.gradients[l][a])
            if col == 0
                ws.b[row] -= e * g
            else
                ws.K[(col-1)*m+row] += e
            end
        end
    end
    return nothing
end

# Fold the region's linear-constraint pivots onto their branches, once after the
# quadrature loop. With `P` the `m × n` expansion that is the identity on the
# active indices and maps pivot `n + k` onto its active branches `(x, w)`, this
# is
#
#     K ← Pᵀ K P,    b ← Pᵀ b − lift,
#
# restricted to the active block, where `lift` carries each pivot column's
# Dirichlet branches `(w, v)`. Every expansion has depth one, so one pass over
# the rows and one over the columns is exact:
#
#   * row step, pivot test row `v` and active branch `(x, w)`:
#     `b[x] += w·b[v]` and `K[x, y] += w·K[v, y]` for every column `y`, keeping
#     `x ≥ y` for an active `y` under a symmetric form. A test row's Dirichlet
#     branch is dropped, as a constrained test row is;
#   * column step, pivot trial column `u`: an active branch `(y, w)` adds
#     `w·K[x, u]` to `K[x, y]` for `x ∈ (sym ? y : 1):n`; a Dirichlet branch
#     with value `v ≠ 0` lifts `b[x] −= (w·v)·K[x, u]` for every `x ∈ 1:n`.
#
# The branches are the ones `_slots!` recorded from the dof layer's
# `_fold_pivot`, which `dof_value` reads too, so assembly and reconstruction
# cannot expand a pivot differently. The result differs from distributing every
# emission through the expansions only in summation order (≤ 1e-16 relative on
# the gate's pivot scenarios). It handles a branch shared by several slots, a
# pivot appearing in several parents, and the lift through a pivot, with no
# branching in the emission loop.
function _condense!(ws::AssemblyWorkspace, n::Int, m::Int, sym::Bool, matrix::Bool)
    K, b = ws.K, ws.b
    for (k, x, w, _) in ws.branches
        x == 0 && continue
        v = n + k
        b[x] += w * b[v]
        matrix || continue
        for y in 1:m
            (sym && y <= n && x < y) || (K[(y-1)*m+x] += w * K[(y-1)*m+v])
        end
    end
    matrix || return nothing
    for (k, y, w, value) in ws.branches
        u = n + k
        if y == 0
            iszero(value) && continue
            for x in 1:n
                b[x] -= w * value * K[(u-1)*m+x]
            end
        else
            for x in (sym ? y : 1):n
                K[(y-1)*m+x] += w * K[(u-1)*m+x]
            end
        end
    end
    return nothing
end

# ── Region frame and slot table ───────────────────────────────────────────────

# Build the region frame: for every field and every parent the field owns in
# `region`, record `(level, slot offset, raw dofs)` and push one provisional
# slot per (component, basis function), component-major per parent, holding the
# global active id (0 when constrained). This, with `_slots!`, is the one place
# a region's dof list is derived: the symbolic pattern, the threaded arena
# layout and the numeric pass all go through it, so they cannot disagree.
#
# The guard compares a cell's raw dof list with the bank, which is sized once
# from the level's nominal order — the `>=` relaxation `_tensor_values_grads!`
# relies on. `_build_cell_dofs!` and the bank both read `level.modes`, so the
# list and the cell's own index set cannot disagree in length; what can fail is
# a cell needing more basis values than its level's bank holds. One integer
# compare per (region, parent, field), outside the quadrature loop.
function _frame!(ws::AssemblyWorkspace, region)
    empty!(ws.parents)
    empty!(ws.slots)
    for (f, fl) in pairs(ws.layout.fields)
        ws.fptr[f] = length(ws.parents) + 1
        parents = region_parents(region, f, fl)
        parents === nothing && continue
        for p in parents
            raw = cell_dofs(fl.dofs, p.level, p.cell)
            length(raw) <= length(ws.bank.values[p.level]) ||
                throw(DimensionMismatch("cell $(p.cell) of level $(p.level) has $(length(raw)) raw " *
                                        "dofs but the workspace bank holds " *
                                        "$(length(ws.bank.values[p.level])) basis values"))
            push!(ws.parents, (p.level, length(ws.slots), raw))
            for c in 1:fl.components, r in raw
                push!(ws.slots, _field_component_dof(fl, r, c))
            end
        end
    end
    ws.fptr[end] = length(ws.parents) + 1
    return ws
end

# Parents of `region` on which field `fl` (global index `f`) is evaluated, or
# `nothing` when the field does not live there. Subdomain ownership is
# intrinsic: a single-space region hands its parents only to a field whose
# level-id block contains the region's level, so a region built on one
# subdomain's plan is owned exactly by the fields on that subdomain. On a
# single-domain model every field's block covers every level and every field
# gets the parents. `nothing` rather than an empty vector keeps the frame
# allocation-free.
function region_parents(region, ::Int, fl::FieldLayout)
    (isempty(region.parents) || first(region.parents).level in fl.level_ids) ? region.parents :
    nothing
end

# Two-sided: each coupled field is evaluated on its own subdomain's parents and
# every other field on none, so an interface pass emits into exactly the four
# field blocks Kₐₐ, K_ab, K_ba, K_bb. Keyed on the field *index* (the interface
# stores its two sides' global field indices), not on the level block.
function region_parents(region::InterfaceRegion, f::Int, ::FieldLayout)
    f == region.field_a ? region.parents_a : f == region.field_b ? region.parents_b : nothing
end

# Number the region's dofs locally and give every slot its role. Returns `n`,
# the number of active local indices.
#
#   1. Collect the active global ids of the frame.
#   2. On a field with linear constraints, turn every constrained slot whose
#      (raw, component) is a pivot into pivot slot `k`, record its branches for
#      `_condense!`, and collect their active ids: a pivot widens the region's
#      active set to its expansion. The rule is the dof layer's `_fold_pivot`,
#      the same one `dof_value` reconstructs a pivot from.
#   3. Sort and deduplicate the ids. Every global id then has exactly one local
#      index, and local order is global order. The second property is what lets
#      the kernel apply a symmetric form's "global row ≥ col" test to local
#      indices before it contracts, and the scatter merge a region column into a
#      sorted pattern column.
#   4. Rewrite each slot: active id `g` ↦ its local index, `0` ↦ `0`, pivot
#      `k` ↦ `n + k`; and each branch's active id to its local index.
#
# The renumbering is bit-neutral: every local sum still receives the same terms
# in the same order. `QuickSort` sorts in place; the default algorithm allocates
# a radix scratch buffer beyond about 40 entries, which is once per region in
# 3D.
function _slots!(ws::AssemblyWorkspace)
    empty!(ws.dofs)
    empty!(ws.pivots)
    empty!(ws.branches)
    for g in ws.slots
        g > 0 && push!(ws.dofs, g)
    end
    for (f, fl) in pairs(ws.layout.fields)
        fl.dofs.has_linear_constraints || continue
        id = (o, c) -> _field_component_dof(fl, o, c)
        for p in ws.fptr[f]:(ws.fptr[f+1]-1)
            _, s0, raw = ws.parents[p]
            for c in 1:fl.components, i in eachindex(raw)
                k = s0 + (c - 1) * length(raw) + i
                ws.slots[k] == 0 || continue
                _, pivot = _fold_pivot(_push_branch, ws, fl.dofs, id, raw[i], c)
                pivot || continue
                push!(ws.pivots, (f, raw[i], c))
                ws.slots[k] = -length(ws.pivots)
            end
        end
    end
    unique!(sort!(ws.dofs; alg=QuickSort))
    n = length(ws.dofs)
    for k in eachindex(ws.slots)
        g = ws.slots[k]
        ws.slots[k] = g > 0 ? searchsortedfirst(ws.dofs, g) : g < 0 ? n - g : 0
    end
    for (j, (k, g, w, v)) in pairs(ws.branches)
        ws.branches[j] = (k, g > 0 ? searchsortedfirst(ws.dofs, g) : 0, w, v)
    end
    return n
end

# The `_fold_pivot` step of `_slots!`: record a branch of the pivot about to be
# numbered, and collect its active id.
function _push_branch(ws::AssemblyWorkspace, g, w, v)
    push!(ws.branches, (length(ws.pivots) + 1, g, w, v))
    g > 0 && push!(ws.dofs, g)
    return ws
end

# ── Per-region-kind quadrature accessors ──────────────────────────────────────
#
# Volume, facet, surface and interface regions share one kernel, `_integrate!`
# above. Only these small accessors differ per region kind, and they dispatch
# statically on the region type.

# The region kinds that store their quadrature points and weights in the
# physical frame, one entry per point: facets, immersed surfaces and interfaces.
# A volume region instead carries a reference rule on its box.
const _PhysicalRegion = Union{FacetRegion,SurfaceRegion,InterfaceRegion}

# Number of quadrature points of a region, which sizes its `q.point` range
# (`RegionList`) and drives the kernel's point loop.
_region_qpoint_count(region::VolumeRegion) = length(region.quadrature.weights)
_region_qpoint_count(region::_PhysicalRegion) = length(region.weights)

# Reference→physical Jacobian for the region's weights: `vol(box)/2ᴰ` for a
# volume region (whose weights are reference-frame), one for facet, surface,
# and interface regions (whose weights are already physical-frame).
_region_jacobian(region::VolumeRegion{D,T}) where {D,T} = volume(region.box) / convert(T, 2^D)
_region_jacobian(region::_PhysicalRegion) = one(eltype(region.weights))

# The per-point step shared by the kernel and the walker: the `q` payload of
# point `k` of `region`, whose index in its list is `point`,
# `(; x, weight, point, state, normal, sides)`. With a workspace `ws` the basis
# is refreshed at the point first and, with a `state`, every field is evaluated
# there; a walk without a state passes `ws = nothing` and evaluates nothing. The
# kernel hands the payload to every form callback and the walker (`_walk`) to
# its caller, so a form and a walk over the same list see the same payload at
# the same point by construction.
@inline function _payload!(ws, region, k::Int, jacobian, point::Int, state)
    x, weight, ref = _region_point(region, k, jacobian)
    ws === nothing || _refresh!(ws, region, ref, Val(true))
    state = state === nothing ? nothing : _state_point!(state, ws, Val(true))
    return (; x, weight, point, state, normal=_region_normal(region, k),
            sides=_region_sides(region))
end

# Quadrature point `k` of `region` as `(x, weight, ref)`: the physical point, the
# physical weight, and the reference each parent maps to its own frame
# (`_parent_xi`). A volume region iterates reference points `η` of its box, so
# `ref = η` and every parent maps it through its `local_box`. Facet, surface and
# interface regions carry physical points and weights, and each parent maps the
# physical point through its own `parent_box`: a facet region has no single
# reference frame shared by every parent, since its free-axis coordinates run
# across several cells per level.
@inline function _region_point(region::VolumeRegion, k::Int, jacobian)
    η = region.quadrature.points[k]
    return reference_to_physical(region.box, η), region.quadrature.weights[k] * jacobian, η
end
@inline function _region_point(region::_PhysicalRegion, k::Int, _)
    x = region.points[k]
    return x, region.weights[k], x
end

# `q`-tuple extras the form callbacks read: the outward unit normal at the
# point (`nothing` for a volume region; the constant face normal for a facet;
# the per-point normal for an immersed surface or an interface, oriented from
# side `a` to side `b` there) and the codim-`K` facet identifier `q.sides` (only
# a facet carries one).
_region_normal(::VolumeRegion, ::Int) = nothing
_region_normal(region::FacetRegion, ::Int) = region.normal
_region_normal(region::Union{SurfaceRegion,InterfaceRegion}, k::Int) = region.normals[k]
_region_sides(::Union{VolumeRegion,SurfaceRegion,InterfaceRegion}) = nothing
_region_sides(region::FacetRegion) = region.sides

# Evaluate every parent's basis at the current point into the bank: values and
# physical gradients (`Val(true)`), or values only (`Val(false)`, for the L²
# transfer rhs, whose integrand never reads a gradient). The gradient scale is
# the axis-aligned chain rule `2 / edge_lengths(parent_box)` of
# `physical_basis_gradients` in `basis.jl`. One basis evaluation per parent per
# point; a region's parents have distinct levels, so their bank slots never
# collide.
@noinline function _refresh!(ws::AssemblyWorkspace{D,T}, region, ref, ::Val{G}) where {D,T,G}
    bank = ws.bank
    for parents in _parent_lists(region), p in parents
        l = p.level
        ids = bank.modes[l].sets[bank.modes[l].kind[p.cell]]
        xi = _parent_xi(p, ref)
        if G
            scale = SVector{D,T}(2 .* inv.(edge_lengths(p.parent_box)))
            _tensor_values_grads!(bank.bases[l], bank.values[l], bank.gradients[l], ids,
                                  bank.orders[l], xi, scale, bank.val1d[l], bank.der1d[l], p.cell)
        else
            _tensor_values!(bank.bases[l], bank.values[l], ids, bank.orders[l], xi, bank.val1d[l],
                            p.cell)
        end
    end
    return nothing
end

# The parent lists refreshed at every point: a region's one list, or both sides
# of an interface, whose parents live in the two subdomains' disjoint level
# banks, so the kernel reads field `a` from `a`'s cut cell and field `b` from
# `b`'s at the same physical point.
_parent_lists(region) = (region.parents,)
_parent_lists(region::InterfaceRegion) = (region.parents_a, region.parents_b)

# A parent's own reference coordinate of the current point: a volume region's
# `η` mapped through `local_box`, the region box in the parent's `[−1, 1]ᴰ`; a
# physical point through the parent's `parent_box`.
_parent_xi(p::ParentRef, η) = reference_to_physical(p.local_box, η)
_parent_xi(p::FacetParent, x) = physical_to_reference(p.parent_box, x)

# ── FormState: evaluation and accessors ───────────────────────────────────────

# Read the dof value of every slot of the region `ws` was framed on, once per
# region: `dof_value` resolves a constrained slot to its stored value and a
# pivot to its expansion, so the per-point sums below need no branch. This
# replaces a `dof_value` call per dof per read, which made a Newton tangent
# reading `q.state` cost O(n²) state work per point instead of O(n).
function _state_region!(state::FormState, ws::AssemblyWorkspace)
    resize!(state.slot_values, length(ws.slots))
    for (f, fl) in pairs(ws.layout.fields), p in ws.fptr[f]:(ws.fptr[f+1]-1)
        _, s0, raw = ws.parents[p]
        for c in 1:fl.components, i in eachindex(raw)
            state.slot_values[s0+(c-1)*length(raw)+i] = dof_value(fl, state.coefficients, raw[i], c)
        end
    end
    return state
end

# Evaluate every field's components at the current point from the refreshed
# bank: per parent the partial sum `Σᵢ uᵢ φᵢ` (and `Σᵢ uᵢ ∇φᵢ` when `G`), then the
# sum over parents. That is also the association of the point evaluation
# (`_parent_field` in `postprocessing.jl`), whose basis values reach it along a
# different route, so the association is all the two share. Returns `state`,
# which becomes `q.state`.
function _state_point!(state::FormState{D,R}, ws::AssemblyWorkspace, ::Val{G}) where {D,R,G}
    bank = ws.bank
    for (f, fl) in pairs(ws.layout.fields), c in 1:fl.components
        v, g = zero(R), zero(SVector{D,R})
        for p in ws.fptr[f]:(ws.fptr[f+1]-1)
            l, s0, raw = ws.parents[p]
            s = s0 + (c - 1) * length(raw)
            pv, pg = zero(R), zero(SVector{D,R})
            @inbounds for i in eachindex(raw)
                u = state.slot_values[s+i]
                pv += u * bank.values[l][i]
                G && (pg += u * bank.gradients[l][i])
            end
            v += pv
            G && (g += pg)
        end
        state.values[state.offsets[f]+c] = v
        state.gradients[state.offsets[f]+c] = g
    end
    return state
end

# Buffer index of `field` (a name or a `Field`), component `component`,
# checking both.
function _state_index(state::FormState, field::Union{Symbol,Field}, component::Integer)
    name = field isa Field ? field.name : field
    f = _field_index(state.layout, name)
    1 <= component <= state.layout.fields[f].components ||
        throw(BoundsError(state, (name, component)))
    return state.offsets[f] + component
end

"""
    value(state::FormState, name_or_field[, component=1])

Read the named field's value at the current quadrature point. `name` is
either a `Symbol` matching a field name in the model or a [`Field`](@ref)
object. For multi-component fields, pass `component`.
"""
function value(state::FormState, field::Union{Symbol,Field}, component::Integer=1)
    return state.values[_state_index(state, field, component)]
end

"""
    field_gradient(state::FormState, name_or_field[, component=1])

Read the named field's physical gradient at the current quadrature
point. Same field/component semantics as [`value`](@ref). The leading
`field_` is there to avoid colliding with `Tensors.gradient` (the
automatic-differentiation entry point) when both packages are loaded
together.
"""
function field_gradient(state::FormState, field::Union{Symbol,Field}, component::Integer=1)
    return state.gradients[_state_index(state, field, component)]
end

# Tiny helper raising the "no state attached" error from a single
# location. Reaching it indicates a callback read `q.state` where it is
# `nothing`: an assembly call or a walk given no `state =`, or a
# `boundary_integral`, whose `q.state` always is.
function _no_form_state()
    throw(ArgumentError("q.state is unavailable: pass `state=` to the assembly call or walk; " *
                        "`boundary_integral` reads a solution by `value(solution, model, u, q.x)`"))
end
value(::Nothing, args...) = _no_form_state()
field_gradient(::Nothing, args...) = _no_form_state()

# ── Quadrature-point walker ───────────────────────────────────────────────────

"""
    nquadpoints(model::Model; kind::Symbol = :volume) -> Int
    nquadpoints(model::Model; on = nothing, field = nothing) -> Int

Number of quadrature points on `model`.

With `on` or `field`, the size of the **one** region list that a form tagged
with that `on=` value — and, on a coupled model, with test field `field` — is
integrated over: exactly the range `1:nquadpoints(model; on, field)` of the
`q.point` index that form sees, and that [`foreach_quadrature_point`](@ref)
visits with the same keywords. `on` is a [`BoundarySelector`](@ref) or a
[`BoundaryMesh`](@ref), whose subdomain `field` names on a coupled model exactly
as it does for [`boundary_integral`](@ref); an [`Interface`](@ref), which spans
both its subdomains and ignores `field`; or `nothing`, the volume, where `field`
alone selects its own subdomain's integration plan. `kind` is ignored in this
form. This is the form that sizes per-quadrature-point state — history
variables for an inelastic law, a cohesive `κ` along an interface.

Without `on` and `field`, the **aggregate** over every cached region list of
kind `kind`, a structural count for diagnostics:

  - `:volume` (default) — the points of every subdomain integration plan.
  - `:facet` — every cached [`FacetRegion`](@ref). On an immersed space this
    is not a function of the mesh and the orders alone: a region on a face
    `∂Ω` crosses carries the moment-fit rule on the facet's affine slice, not
    the tensor product `_facet_quadrature_counts` sizes, so its point count is
    whatever that rule came out at — at most the moment basis size on a
    `:cut_fitted` region, which at a low order is *more* points than the
    untrimmed rule rather than fewer (measured 3 → 5 on a cut face of an
    order-2 space), the raw Saye rule on a `:cut_fallback` one, and none at all
    for a face wholly outside `Ω` under strict `α = 0`. `α > 0` appends the
    full-face tensor rule to each of those. The count therefore moves with the
    geometry, and `cut_facet_region_count` says how many regions are in that
    regime.
  - `:surface` — every cached immersed [`SurfaceRegion`](@ref)
    ([`BoundaryMesh`](@ref) integration).
  - `:interface` — every cached multi-domain [`InterfaceRegion`](@ref).

`q.point` is numbered **per region list**, not per kind: it restarts at 1
for each distinct `on=` value and, on a coupled model, for each subdomain
naming that value. So the aggregate exceeds the `q.point` range whenever
the model caches more than one list of that kind, and indexing a
`kind=`-sized array by `q.point` would alias one list onto another. Size
per-point state with `on` and `field`; the `kind=:volume` default is safe
only because a single-domain model has exactly one plan, and both
[`foreach_quadrature_point`](@ref) without `field` and
`QuadField{T}(model; init)` reject a coupled model for that reason.
"""
function nquadpoints(model::Model; kind::Symbol=:volume, on=nothing,
                     field::Union{Nothing,Symbol}=nothing)::Int
    (on === nothing && field === nothing) ||
        return _region_list(model, on, _target_space(model, on, field)).offsets[end]
    # The aggregate sums over one of four iterator types, which inference does not
    # follow; the declared `::Int` is what keeps a caller that sizes a loop by the
    # count type-stable (the `RBFP0` transfer's serial loop allocated per point).
    lists = kind === :volume ? (plan.regions for plan in integration_plans(model)) :
            kind === :facet ? values(model.facet_regions) :
            kind === :surface ? values(model.surface_regions) :
            kind === :interface ? values(model.interface_regions) :
            throw(ArgumentError("nquadpoints kind must be :volume, :facet, :surface, or " *
                                ":interface, got $kind"))
    return sum(regions -> sum(_region_qpoint_count, regions; init=0), lists; init=0)
end

"""
    foreach_quadrature_point(f, model; on=nothing, field=nothing, state=nothing)

Call `f(q)` at every quadrature point of one integration region list of
`model`: the points at which a form tagged with the same `on=` — and, on a
coupled model, with test field `field` — is integrated, in the order the
serial assembly visits them. `q` is the payload such a form sees,

    q = (; x, weight, point, state, normal, sides)

where

  - `q.x` — the physical coordinate of the point;
  - `q.weight` — its physical quadrature weight: the reference weight times
    the cell Jacobian on the volume, the measure on a boundary, surface or
    interface;
  - `q.point` — its index in `1:nquadpoints(model; on, field)`, the index the
    forms see during assembly, serial or threaded, and so the natural key for
    per-point state (history variables, phase-field damage, plastic strain,
    an irreversible cohesive `κ`, …);
  - `q.state` — `nothing` unless a `state` (a [`Solution`](@ref) or an active
    coefficient vector) is passed, in which case it is a [`FormState`](@ref):
    `value(q.state, field)` and `field_gradient(q.state, field)` read the
    fields of the walked list at the point — on a volume, boundary or surface
    walk those of the walked subdomain, on an interface both coupled fields,
    each in its own cut cell at the shared point. Any other field reads zero,
    even where it is defined at `q.x`; evaluate it there with
    `value(solution, model, u, q.x)`;
  - `q.normal` — `nothing` on the volume, the outward unit normal on a
    boundary or surface, and on an interface the unit normal oriented from
    side `a` toward side `b` (the two fields passed to [`couple`](@ref), in
    order);
  - `q.sides` — the facet identifier on a [`BoundarySelector`](@ref),
    `nothing` elsewhere.

`on` selects the list as it does for a form: `nothing` (the default) the
volume, a [`BoundarySelector`](@ref) the physical boundary, a
[`BoundaryMesh`](@ref) an immersed surface, and an [`Interface`](@ref) the
two-sided interface of a coupled model. On a coupled model `field` names the
subdomain of a volume, boundary or surface walk, as for
[`boundary_integral`](@ref), and omitting it raises, because each subdomain
numbers its points separately and there is no single list to walk; an
interface spans both its subdomains and ignores `field`. Two `Interface`s
match when they name the same two fields, in the same order, over the same
[`BoundaryMesh`](@ref) object: a fresh `interface(uₐ, u_b, Γ)` over the mesh
given to [`couple`](@ref), like the coupling blocks' own `first(blocks).on`,
finds the list `prepare` resolved, while a mesh rebuilt as a new object, even
an equal one, is intersected with both grids again.

The basis is evaluated only when a `state` is given, so a walk without one
costs a pass over the stored points and weights once its list is resolved.
"""
function foreach_quadrature_point(f, @nospecialize(model::Model); on=nothing,
                                  field::Union{Nothing,Symbol}=nothing, state=nothing)
    list = _region_list(model, on, _target_space(model, on, field))
    coefficients = _state_vector(state, model)
    state = coefficients === nothing ? nothing : FormState(model.dofs, coefficients)
    workspace = state === nothing ? nothing : _assembly_workspace(model)
    return _walk((acc, q) -> (f(q); acc), nothing, workspace, list, state)
end

# The subdomain space an `on=` target resolves on, for the callers that name a
# target rather than a form (`nquadpoints`, `foreach_quadrature_point`,
# `boundary_integral`). A named `field` picks its space; without one, a
# single-domain model has exactly one answer and a multi-domain model has none,
# so it raises rather than pick — see the `boundary_integral` docstring for why
# neither silent default is defensible. An `Interface` spans both its
# subdomains and resolves on `nothing`.
function _target_space(@nospecialize(model::Model), @nospecialize(on), field::Union{Nothing,Symbol})
    on isa Interface && return nothing
    field === nothing || return _field_space(model.problem, field)
    length(problem_spaces(model.problem)) == 1 && return model.problem.space
    names = join((":" * String(f.name) for f in model.problem.fields), ", ")
    throw(ArgumentError("field argument is required for multi-domain models; pass field=… " *
                        "(one of $names)"))
end

# Fold `op(acc, q)` over every quadrature point of `list` in the serial assembly
# order, starting from `acc`, and return the result: `foreach_quadrature_point`
# discards it, `boundary_integral` sums through it. The payload is the kernel's
# own (`_payload!`), from the same `_region_point`, with `q.point =
# list.offsets[r] + k`. With a state, each region's dof values are read once and
# every field is evaluated at every point through the same `_frame!`,
# `_refresh!` and `_state_point!` the kernel uses, on the workspace `ws`;
# without one, `ws` is `nothing` and no basis is evaluated at all. Reached by a
# dynamic call, which makes it the function barrier for the region and state
# types. Its callers build `ws` fresh (`_assembly_workspace`) on purpose rather
# than check one out of the model's assembly pool: a walk needs one per call,
# which costs a few microseconds, and a checkout would need a check-in in a
# `finally` at every caller.
function _walk(op, acc, ws, list::RegionList, state)
    for (r, region) in pairs(list.regions)
        state === nothing || _state_region!(state, _frame!(ws, region))
        jacobian = _region_jacobian(region)
        for k in 1:_region_qpoint_count(region)
            acc = op(acc, _payload!(ws, region, k, jacobian, list.offsets[r] + k, state))
        end
    end
    return acc
end
