using BasicBSpline   # switches on the B-spline extension for the clamped case
using BenchmarkTools
using Unfitted

# The build cost of the dof layer: what it costs to construct a layout, to prepare a
# model around one, and to rebuild or update a model in place.
#
# The layout is built on every `prepare` and on every structural mutation. Unlike
# assembly, nothing amortises it over a solve, so in an adaptive or moving-overlay loop
# its build cost is paid on every step. Three entry points are timed on each of four
# problems, chosen so that each stresses a different part of the layer:
#
#   * `dof_layout`, the standalone layout of one field over a space: enumeration,
#     artificial eliminations, physical-Dirichlet detection, numbering and, for
#     nonzero data, the boundary projection;
#   * `system_layout`, every field of a problem as `prepare` builds them;
#   * `prepare`, the whole model, layout included.
#
# The four problems:
#
#   * 3D 12³ p3 with boundary(:all): one large level, every boundary raw of it
#     physical Dirichlet, under zero data, so detection runs and no projection does;
#   * a 2D 32² + 32² p3 corner overlay with nonzero data on every face: two levels
#     and the boundary projection;
#   * a deep 2D ladder, 6² depth 5 p2, with one base cell refined to the bottom rung:
#     six levels and their coverage pruning;
#   * a clamped C¹ p3 B-spline, 64², under an L mask: the linear-constraint rows and
#     their pivots.
#
# Then come the verbs that rebuild or update a prepared model: `move!` and
# `update_dirichlet!` (first increment, which builds the projection, and later ones,
# which reuse it) on the overlay, `activate!` / `deactivate!` on the ladder, and
# `estimate`, which builds an enriched twin layout, on a tanh-front ladder.
#
# Only entry points that every representation of the layout keeps are called, so the
# same four problems serve as a before/after reference for changes to the dof layer.
group = SUITE["build"]["dof_layout"] = BenchmarkGroup()

let
    Ω2 = box((0.0, 0.0), (1.0, 1.0))
    Ω3 = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    # A Poisson problem (stiffness plus a unit source) over `V` with Dirichlet list `bcs`.
    function poisson_problem(V, bcs)
        u = field(:u, V)
        return Problem((u,); blocks=(stiffness_block(u),), loads=(source_load(u; source=1.0),),
                       dirichlet=bcs)
    end
    # The three layout entry points on one problem, as one subgroup.
    function layout_cases(name, V, bcs)
        problem = poisson_problem(V, bcs)
        cases = group[name] = BenchmarkGroup()
        cases["dof_layout"] = @benchmarkable Unfitted.dof_layout($V; dirichlet=($bcs))
        cases["system_layout"] = @benchmarkable Unfitted.system_layout($problem)
        cases["prepare"] = @benchmarkable prepare($problem)
        return cases, problem
    end
    # An in-place verb that switches a model between two equivalent states, so every
    # evaluation is one real rebuild from a valid model and no untimed re-prepare is
    # needed between samples.
    function toggler(on, off)
        state = Ref(false)
        return function ()
            state[] = !state[]
            return state[] ? on() : off()
        end
    end

    layout_cases("D=3 12^3 p3 :all", space(Ω3; cells=12, order=3),
                 [dirichlet(0.0; on=boundary(:all))])

    # Nonzero data, so `dof_layout`, `prepare` and `move!` all include the projection.
    data(s) = [dirichlet(x -> s * (prod(sinpi, x) + exp(x[1]) * sin(x[2]) + 1.0);
                         on=boundary(:all))]
    corner = overlay(space(Ω2; cells=32, order=3), box((0.0, 0.0), (0.5, 0.5)); cells=32, order=3)
    overlay_cases, overlay_problem = layout_cases("D=2 32^2 + 32^2 p3 overlay, nonzero", corner,
                                                  data(1.0))
    let model = prepare(overlay_problem)
        b1, b2 = box((0.0, 0.0), (0.5, 0.5)), box((0.25, 0.0), (0.75, 0.5))
        move = toggler(() -> move!(model; level=2, to=b2), () -> move!(model; level=2, to=b1))
        overlay_cases["move!"] = @benchmarkable $move()
    end
    let increment = data(2.0)
        # A fresh model per sample: `prepare` keeps no projection, so the first
        # increment always builds one.
        overlay_cases["update_dirichlet! first"] = @benchmarkable(update_dirichlet!(m, $increment),
                                                                  setup=(m = prepare($overlay_problem)),
                                                                  evals=1)
    end
    let model = prepare(overlay_problem), d1 = data(1.0), d2 = data(2.0)
        update_dirichlet!(model, d1)   # builds and caches the projection
        next = toggler(() -> update_dirichlet!(model, d2), () -> update_dirichlet!(model, d1))
        overlay_cases["update_dirichlet! next"] = @benchmarkable $next()
    end

    depths = zeros(Int, 6, 6)
    depths[3, 3] = 5
    deep = adapt(ladder(Ω2; cells=6, order=2, depth=5, splits=2), depths; grade=1)
    ladder_cases, ladder_problem = layout_cases("D=2 ladder 6^2 depth 5 p2", deep,
                                                [dirichlet(0.0; on=boundary(:all))])
    let model = prepare(ladder_problem)
        cells = [CartesianIndex(i, j) for i in 17:20 for j in 17:20]
        flip = toggler(() -> activate!(model; level=4, cells=cells),
                       () -> deactivate!(model; level=4, cells=cells))
        ladder_cases["activate!/deactivate!"] = @benchmarkable $flip()
    end

    mask = trues(64, 64)
    mask[33:64, 33:64] .= false
    layout_cases("D=2 clamped C1 p3 64^2, L mask",
                 space(Ω2; cells=64, order=3, basis=bspline(; continuity=1), active=mask),
                 [dirichlet(0.0; on=boundary(:all))])

    # A tanh front (λ = 30) along a sine, refined along the front, with its exact
    # solution as the Dirichlet datum and the matching source.
    let λ = 30.0
        exact(x) = tanh(λ * (x[2] - (0.5 + 0.25 * sinpi(2 * x[1]))))
        # `t`, not `u`: a nested function's assignment to a name the enclosing `let`
        # also binds rebinds that local, which would box the field `u` below and put
        # an allocation in every source evaluation.
        function source(x)
            t = exact(x)
            s′ = 0.5π * cospi(2 * x[1])
            s″ = -π^2 * sinpi(2 * x[1])
            return (1 - t^2) * (λ * s″ + 2 * λ^2 * t * (1 + s′^2))
        end
        front(x) = 0.5 + 0.25 * sinpi(2x)
        marks = [abs((j - 0.5) / 6 - front((i - 0.5) / 6)) < 0.2 ? 2 : 0 for i in 1:6, j in 1:6]
        u = field(:u, adapt(ladder(Ω2; cells=6, order=2, depth=3), marks; grade=1))
        model = prepare(Problem((u,); blocks=(stiffness_block(u),),
                                loads=(source_load(u; source=source),),
                                dirichlet=[dirichlet(exact; on=boundary(:all))]))
        sol = solve!(model)
        group["D=2 tanh ladder 6^2 depth 3 p2"] = BenchmarkGroup()
        group["D=2 tanh ladder 6^2 depth 3 p2"]["estimate"] = @benchmarkable estimate($model, $sol)
    end
end
