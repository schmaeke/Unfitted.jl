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
  - `converged::Bool` — true when the solver reports convergence (or
    the residual is finite for direct solves). Iterative solvers fill
    this from their own convergence criteria.
"""
struct SolverDiagnostics
    method::Symbol
    residual_norm::Float64
    converged::Bool
end

"""
    Solution{C}

Wrapper around an active coefficient vector tied to a specific
[`Model`](@ref) version. The version pin is the stale-solution
detection mechanism: any mutator ([`move!`](@ref), [`activate!`](@ref),
[`deactivate!`](@ref), [`assemble!`](@ref)) bumps `model.version`, and
every consumer that takes a `Solution` calls `_checked_coefficients`
to assert the versions still match. The result is that a `Solution`
built against an older model state raises at the use site instead of
silently returning wrong numbers.

Fields:

  - `coefficients::C` — active-dof coefficient vector. Stored as
    supplied (no defensive copy) so external time integrators can
    update the same buffer in place.
  - `model_version::Int` — the `model.version` at the time of solve.
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
    print(io, "Solution(version=", solution.model_version, ", dofs=", length(solution.coefficients),
          ", method=:", solution.diagnostics.method, ", residual=",
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
# (transfer, evaluation, post-processing). Returns the coefficient
# vector when version and length agree; raises clear errors otherwise.
# This is the single source of truth for "does this Solution still
# match the model that produced it"; assembly, projection, and
# postprocessing all route through here.
function _checked_coefficients(solution::Solution, model::Model)
    solution.model_version == model.version ||
        throw(ArgumentError("solution belongs to model version $(solution.model_version), but model is at version $(model.version)"))
    length(solution.coefficients) == active_unknowns(model.dofs) ||
        throw(DimensionMismatch("solution coefficient vector does not match the model active space"))
    return solution.coefficients
end

# Active coefficient vector of the iterate passed as `state=` to
# `assemble_matrix` / `assemble_vector` / `foreach_quadrature_point`.
# Three dispatched cases:
#
#   * `nothing`  → assembly runs without `q.state`, callbacks see
#                  `q.state == nothing`.
#   * `Solution` → version-checked via `_checked_coefficients`.
#   * `AbstractVector` → length-checked only; useful for embedding
#     external time integrators that own their own coefficient buffers
#     and don't bother wrapping them in a `Solution`.
_iterate_coefficients(::Nothing, model::Model) = nothing
_iterate_coefficients(solution::Solution, model::Model) = _checked_coefficients(solution, model)
function _iterate_coefficients(coefficients::AbstractVector, model::Model)
    length(coefficients) == active_unknowns(model.dofs) ||
        throw(DimensionMismatch("state coefficient vector does not match the model active space"))
    return coefficients
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

`method` overrides the recorded solver tag. Defaults to `:direct`
(no `linear_solver`) or `:custom` (with `linear_solver`).

Returns a fresh [`Solution`](@ref) carrying the coefficients, the
model version at solve time, and a [`SolverDiagnostics`](@ref) record
with `‖A x − b‖₂` as the residual norm. The model's cached
`matrix`/`rhs` are left untouched so a follow-up `solve!` (e.g. with a
different `linear_solver`) reuses them.
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
