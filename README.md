# Unfitted.jl

> [!WARNING]
> **Early development.** Unfitted.jl is under active development and has not
> had a stable release yet. The public API, default values, and numerical
> conventions can change between commits without prior notice. Pin a specific
> commit if you need reproducibility, and review the diff before updating.

A compact, dimension-generic Julia implementation of **unfitted multi-level
hp refinement** on axis-aligned Cartesian grids, with **selective per-cell
activation** and **finite-cell-method (FCM) immersed-boundary integration**
via non-negative moment-fit quadrature.

## Method reference

The method is described in

> J. N. Schmäke and M. Ruess, *Unfitted multi-level hp refinement on
> Cartesian grids*, arXiv preprint
> [arXiv:2604.25797](https://arxiv.org/abs/2604.25797).

## What's in the package

- **Unfitted multi-level hp refinement** on tensor-product Cartesian
  meshes via the superposition of independently discretized levels
  (base + overlays). Unlike the *fitted* multi-level hp method
  (Zander et al., as implemented and extended by Kopp et al. CMAME
  **401** (2022) 115575), this package neither topologically merges
  overlays with their parents nor introduces hanging-node
  constraints; every overlay imposes homogeneous Dirichlet data on
  its artificial boundary `Γ_o^(k) = ∂Ω^(k) \ ∂Ω` instead. The
  default basis family is hierarchical integrated Legendre, with a
  small interface left open for alternatives such as B-splines.
- **Selective per-cell activation** with the `move!`-style invalidation
  contract; useful for transient problems where small-scale features
  evolve in time.
- **Immersed-boundary integration** through a `PhysicalDomain` carrying a
  Lipschitz level set. Cells outside `Ω` are dropped from the dof layout;
  cells crossed by `∂Ω` use a non-negative moment-fitted quadrature rule
  ported from QuESo (see [`NOTICE.md`](NOTICE.md) for upstream attribution).
- **D-generic core**: 1D, 2D, 3D, and 4D smoke-tested.

## Installation

Unfitted.jl is not yet registered. Add it directly from the repository:

```julia
import Pkg
Pkg.add(url="https://github.com/schmaeke/Unfitted.jl")
```

Julia 1.10 or newer is required.

## Quick example

Modified Helmholtz on a disk-shaped immersed domain:

```julia
using Unfitted

omega = box((-1.0, -1.0), (1.0, 1.0))
disk  = physical_domain(x -> sqrt(x[1]^2 + x[2]^2) - 0.7; lipschitz=1.0)

V = space(omega; cells=(16, 16), order=2, physical=disk)
u = field(:u, V)

problem = Problem((u,);
                  blocks    = (stiffness_block(u), mass_block(u)),
                  loads     = (source_load(u; source = x -> 1.0),),
                  dirichlet = [dirichlet(0.0; on=boundary(:all))])

model    = prepare(problem)
solution = solve!(model)
report   = diagnostics(model, solution)
```

A small tour of `examples/`. Each example lives in its own
sub-directory with a self-contained `Project.toml` (Unfitted is wired
in via `[sources]`), so example-specific dependencies stay out of the
package's own `Project.toml`. Run any example with

```bash
julia --project=examples/<name> examples/<name>/<name>.jl
```

| Sub-directory | What it demonstrates |
|---|---|
| `laplace_unit_square_smooth/` | Smooth Laplace verification — public-API walkthrough |
| `bar_1d_unresolved_interface/` | 1D elastic bar across a discontinuous coefficient |
| `singular_square_2d/` | 2D corner-singularity convergence with nested overlays |
| `conditioning_small_overlap/` | Small-overlap conditioning sweep |
| `traveling_heat_source_2d/` | Transient heat with hierarchical adaptive overlays |
| `phase_field_single_edge_notch_2d/` | Phase-field fracture, SENT specimen |
| `fcm_annular_plate_2d/` | FCM plane-stress annular plate with Nitsche + Neumann (Ruess 2013 §4.2) |

## Documentation

Source files are the primary documentation. Every public symbol carries a
docstring; non-trivial internal helpers carry leading comment blocks
explaining the math, the algorithmic choices, or both.

For project goals, mathematical contracts, code style, testing standards,
and the contributor / PR workflow, see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing

If you use Unfitted.jl in academic work, please cite the method paper:

> J. N. Schmäke and M. Ruess, *Unfitted multi-level hp refinement on
> Cartesian grids*, arXiv:2604.25797.

A machine-readable record will be provided in
[`CITATION.cff`](CITATION.cff).

## License and attribution

Unfitted.jl is distributed under the MIT License (see [`LICENSE.md`](LICENSE.md)).

The finite-cell-method machinery in `src/fcm.jl` is derived from
[QuESo](https://github.com/manuelmessmer/QuESo) (BSD-4-Clause). See
[`NOTICE.md`](NOTICE.md) for the full upstream attribution.
