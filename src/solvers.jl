# Solution wrapping and linear-solve dispatch.
#
# The solver layer is intentionally thin: `solve!` runs Julia's direct
# sparse solve out of the box, and a user-supplied `linear_solver`
# callback drops in any iterative or preconditioned solver without
# adding a dependency to the package.

# ── Solution and diagnostics ──────────────────────────────────────────────────

"""
    SolverDiagnostics(method, residual_norm, converged)

Per-solve diagnostic record carried on every [`Solution`](@ref):

  - `method::Symbol` — solver tag. `:direct` for the default direct
    sparse solve; `:custom` for a user-supplied `linear_solver`;
    `:manual` for [`solution`](@ref) wrappers that bypass the solve
    entirely; `:l2_projection` and `:rewire` from `transfer`.
  - `residual_norm::Float64` — `‖A x − b‖₂` of the returned solution
    against the assembled system; `NaN` when the wrapper bypassed the
    solve (e.g. [`solution`](@ref) constructors fed by an external time
    integrator).
  - `converged::Bool` — whether the solve is to be trusted. [`solve!`](@ref)
    sets it to `isfinite(residual_norm)` on *both* of its paths: a
    `linear_solver` hook hands back only a coefficient vector, so it has no
    channel through which to report a convergence verdict of its own. An
    iterative solver that tracks its own criteria should record the verdict
    through [`solution`](@ref), which stores the `converged=` keyword
    verbatim.
"""
struct SolverDiagnostics
    method::Symbol
    residual_norm::Float64
    converged::Bool
end

"""
    Solution{C}

Wrapper around an active coefficient vector tied to a specific
[`Model`](@ref)'s active dof numbering. The pin it carries is the
stale-solution detection mechanism: every consumer that takes a `Solution`
calls `_checked_coefficients` to assert that the pin still matches the model
it is handed, so a `Solution` used against a numbering it was not computed
on raises at the use site instead of silently returning wrong numbers.

Two things move the pin, and both are caught. The mutators that change the
dof numbering ([`move!`](@ref), [`activate!`](@ref), [`deactivate!`](@ref))
move it on in place. And a model of a *different discretisation* carries a
different pin, because `model.version` is seeded from a structural digest of
the discretisation rather than from a counter — so a solution taken on one
[`adapted`](@ref) / [`moved`](@ref) / [`prepare`](@ref)d model is refused by
another, which is what [`transfer`](@ref) exists for. Two models prepared
from the *same* discretisation share a pin, so a transient loop that rebuilds
its problem every step and carries its state forward keeps working.

[`assemble!`](@ref) and [`update_dirichlet!`](@ref) deliberately do *not*
bump: they refill the matrix and right-hand side over an unchanged
active-dof numbering, so an outstanding `Solution` remains a valid
coefficient vector of that numbering and stays usable. Only its
`diagnostics.residual_norm` goes stale — it was measured against the system
that has since been reassembled.

Fields:

  - `coefficients::C` — active-dof coefficient vector. Stored as
    supplied (no defensive copy) so external time integrators can
    update the same buffer in place.
  - `model_version::Int` — the `model.version` pin at the time of solve.
  - `diagnostics::SolverDiagnostics` — solver record.

Use [`solution`](@ref) or [`solve!`](@ref) to construct.
"""
struct Solution{C}
    coefficients::C
    model_version::Int
    diagnostics::SolverDiagnostics
end

function Solution(coefficients::C, model_version::Integer, diagnostics::SolverDiagnostics) where {C}
    return Solution{C}(coefficients, Int(model_version), diagnostics)
end

function Base.show(io::IO, diagnostics::SolverDiagnostics)
    print(io, "SolverDiagnostics(method=:", diagnostics.method, ", residual=",
          diagnostics.residual_norm, ", converged=", diagnostics.converged, ")")
end

function Base.show(io::IO, solution::Solution)
    print(io, "Solution(pin=0x", string(solution.model_version; base=16), ", dofs=",
          length(solution.coefficients), ", method=:", solution.diagnostics.method, ", residual=",
          solution.diagnostics.residual_norm, ")")
end

"""
    solution(model::Model, coefficients; method=:manual, residual=NaN, converged=true) -> Solution

Wrap an active coefficient vector as a [`Solution`](@ref) tied to the
current model version. The coefficient vector is stored as supplied
(no defensive copy). This is the public entry point for external time
integrators or solvers that compute coefficients outside the package
and need to feed them back into the post-processing / diagnostics /
transfer paths.

`method` defaults to `:manual`; pass an explicit name when the
external code knows what it did. `residual` and `converged` end up in
the [`SolverDiagnostics`](@ref) record verbatim.
"""
function solution(model::Model, coefficients; method::Symbol=:manual, residual=NaN,
                  converged::Bool=true)
    length(coefficients) == active_unknowns(model.dofs) ||
        throw(DimensionMismatch("coefficient vector does not match the model active space"))
    diagnostics = SolverDiagnostics(method, Float64(residual), converged)
    return Solution(coefficients, model.version, diagnostics)
end

# ── Coefficient checks (cross-module) ─────────────────────────────────────────

# Stale-solution check used by every consumer that takes a `Solution`
# (transfer, evaluation, post-processing). Returns the coefficient vector when
# the pin and the length agree; raises clear errors otherwise. This is the
# single source of truth for "does this Solution still match the model it is
# being used on"; assembly, projection, and postprocessing all route through
# here. The length check stays even though the pin implies it, because it is
# the one that catches a caller-built coefficient vector of the wrong size.
function _checked_coefficients(solution::Solution, model::Model)
    solution.model_version == model.version ||
        throw(ArgumentError("this solution was computed on a different discretisation, or on this model " *
                            "before it was mutated (solution pin 0x$(string(solution.model_version; base=16)), " *
                            "model pin 0x$(string(model.version; base=16))); carry it across with `transfer`, " *
                            "or re-solve on this model"))
    length(solution.coefficients) == active_unknowns(model.dofs) ||
        throw(DimensionMismatch("solution coefficient vector does not match the model active space"))
    return solution.coefficients
end

# The active coefficient vector of the iterate passed as `state=` to `assemble`
# and its wrappers, `foreach_quadrature_point` and the L² transfer's source
# evaluation, checked and converted once. Three dispatched cases:
#
#   * `nothing`  → no state: callbacks see `q.state == nothing`.
#   * `Solution` → version- and length-checked via `_checked_coefficients`.
#   * `AbstractVector` → length-checked only; useful for embedding
#     external time integrators that own their own coefficient buffers
#     and don't bother wrapping them in a `Solution`.
#
# The result is a `Vector{R}`, `R = promote_type(T, eltype(state))`, so the
# eltype stays generic while the state's container type stops being a
# specialisation axis of the kernel; a `Vector{R}` passes through without a
# copy. Dispatched methods rather than one unspecialised body, which measured
# three more allocations per call.
_state_vector(::Nothing, ::Model) = nothing
function _state_vector(solution::Solution, model::Model)
    return _state_vector(_checked_coefficients(solution, model), model)
end
function _state_vector(coefficients::AbstractVector, model::Model)
    length(coefficients) == active_unknowns(model.dofs) ||
        throw(DimensionMismatch("state coefficient vector does not match the model active space"))
    R = promote_type(_scalar(model.dofs), eltype(coefficients))
    return coefficients isa Vector{R} ? coefficients : convert(Vector{R}, coefficients)
end

# ── Solve ─────────────────────────────────────────────────────────────────────

# Record the solver tag on the model's diagnostics. Called from
# `solve!` after the solve completes so the reproducibility report
# downstream knows which solver path produced the cached result.
function _record_solver!(model::Model, method::Symbol)
    model.diagnostics.solver = method
    return model
end

"""
    solve!(model::Model; linear_solver=nothing, method=nothing) -> Solution

Solve an assembled `model`'s linear system `A x = b`. If `model.matrix`
or `model.rhs` is `nothing`, calls [`assemble!`](@ref) first.

Two solver paths:

  - **Direct sparse solve** (default). With `linear_solver = nothing`,
    use Julia's built-in `A \\ b` and record `method = :direct`. This
    routes through SuiteSparse for sparse `A` and is the right default
    for problems up to a few hundred thousand active unknowns.
  - **Custom solver hook**. Pass `linear_solver` as a callable
    `linear_solver(A, b) -> Vector` and the active coefficient vector
    it returns is used as the solution. This is the integration point
    for iterative Krylov solvers, preconditioned solvers, or any
    domain-specific solve — none of which the package itself depends on.

`method` renames the solver tag recorded on the returned
[`SolverDiagnostics`](@ref) and on `diagnostics(model).solver`. It
defaults to `:direct` without a `linear_solver` and `:custom` with one.
It renames a solve, it does not select one: without a `linear_solver`
the only accepted value is `:direct`, and any other name raises
`ArgumentError` rather than quietly labelling the built-in direct solve
as something it is not.

Returns a fresh [`Solution`](@ref) carrying the coefficients, the
model version at solve time, and a [`SolverDiagnostics`](@ref) record
with `‖A x − b‖₂` as the residual norm and `converged` set from its
finiteness. The model's cached `matrix`/`rhs` are left untouched so a
follow-up `solve!` (e.g. with a different `linear_solver`) reuses them.
"""
function solve!(model::Model; linear_solver=nothing, method::Union{Nothing,Symbol}=nothing)
    if model.matrix === nothing || model.rhs === nothing
        assemble!(model)
    end

    solve_method = method === nothing ? (linear_solver === nothing ? :direct : :custom) : method
    if linear_solver === nothing
        solve_method === :direct ||
            throw(ArgumentError("method=:$solve_method requires a linear_solver"))
        coefficients = model.matrix \ model.rhs
    else
        coefficients = linear_solver(model.matrix, model.rhs)
        length(coefficients) == length(model.rhs) ||
            throw(DimensionMismatch("linear_solver returned $(length(coefficients)) coefficients; expected $(length(model.rhs))"))
    end

    residual = norm(model.matrix * coefficients - model.rhs)
    _record_solver!(model, solve_method)
    solver_diagnostics = SolverDiagnostics(solve_method, Float64(residual), isfinite(residual))
    return Solution(coefficients, model.version, solver_diagnostics)
end
