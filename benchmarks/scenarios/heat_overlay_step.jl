using BenchmarkTools
using LinearAlgebra
using Unfitted

# Reduced version of
# the traveling heat-source study this benchmark was derived from.
# Covers the per-step
# hot ops of a theta-step (mass + stiffness reassembly, compactly supported
# load) and one variational transfer between two overlay configurations,
# both with and without the cached target mass + factor used by the example.
group = SUITE["scenarios"]["heat_overlay_step"] = BenchmarkGroup()

let
    L = 10.0
    N = 7
    p = 3
    no_refinements = 2
    omega = box((-0.5L, -0.5L), (0.5L, 0.5L))

    function build_space(center)
        V = space(omega; cells=(N, N), order=p)
        for ref in 1:no_refinements
            width = L * 0.5^(ref + 1)
            V = overlay(V, box(center; halfwidth=0.5width); cells=(N, N), order=max(1, p - ref))
        end
        return V
    end

    center_src = (2.5, 0.0)
    center_tgt = (2.0, 1.5)

    V_src = build_space(center_src)
    V_tgt = build_space(center_tgt)
    temp_src = field(:temperature, V_src)
    temp_tgt = field(:temperature, V_tgt)

    problem_src = Problem((temp_src,);
                          dirichlet=[dirichlet(0.0; on=boundary(:all), field=temp_src)])
    problem_tgt = Problem((temp_tgt,);
                          dirichlet=[dirichlet(0.0; on=boundary(:all), field=temp_tgt)])
    model_src = prepare(problem_src)
    model_tgt = prepare(problem_tgt)

    mass_block_src = mass_block(temp_src)
    stiff_block_src = stiffness_block(temp_src; diffusion=1.0)
    source_fn(x) = ((x[1] - center_src[1])^2 + (x[2] - center_src[2])^2 <= 0.04) ? 10.0 : 0.0
    load_src = source_load(temp_src; source=source_fn)

    src_coeffs = [sin(0.01 * i) for i in 1:Unfitted.active_unknowns(model_src.dofs)]
    src_solution = solution(model_src, src_coeffs; method=:synthetic)

    # Cached target operators — mirror
    # the traveling heat-source study this benchmark was derived from.
    # The cached path threads the precomputed mass + factor in through the
    # `L2Projection(matrix; factor)` backend so the transfer skips both the
    # target-side mass assembly and its factorisation. The uncached bench
    # uses the default `L2Projection()` backend (no cache) so the assembly
    # and factorisation costs are visible side-by-side.
    mass_tgt = assemble_matrix(model_tgt, mass_block(temp_tgt))
    mass_tgt_factor = factorize(mass_tgt)
    cached_backend = L2Projection(mass_tgt; factor=mass_tgt_factor)

    group["assemble_mass"] = @benchmarkable assemble_matrix($model_src, $mass_block_src)
    group["assemble_stiffness"] = @benchmarkable assemble_matrix($model_src, $stiff_block_src)
    group["assemble_load"] = @benchmarkable assemble_vector($model_src, $load_src)
    group["transfer cached"] = @benchmarkable transfer($src_solution, $model_src, $model_tgt;
                                                       via=($cached_backend))
    group["transfer uncached"] = @benchmarkable transfer($src_solution, $model_src, $model_tgt)
end
