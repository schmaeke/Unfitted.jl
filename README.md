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

> J. N. Schmäke and M. Ruess, *Unfitted Multi-Level hp Refinement for
> Localized and Moving Solution Features*, arXiv preprint
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
  default basis family is hierarchical integrated Legendre; an
  open-knot tensor-product B-spline family is available through the
  `BasicBSpline.jl` package extension (load `BasicBSpline` alongside
  `Unfitted` and pass `basis = bspline()`); worked through in
  [`examples/tutorials/04_bspline/`](examples/tutorials/04_bspline).
- **Selective per-cell activation** with the `move!`-style invalidation
  contract; useful for transient problems where small-scale features
  evolve in time. Overlays, masks and the invalidation contract are
  worked through in
  [`examples/tutorials/02_overlays/`](examples/tutorials/02_overlays).
- **Immersed-boundary integration** through a `PhysicalDomain` carrying a
  CSG level set of smooth leaves (`leaf`/`intersect`/`union`/`setdiff`/
  `complement`). Cells outside `Ω` are dropped from the dof layout; cells
  crossed by `∂Ω` use a non-negative moment-fitted quadrature rule whose
  moments come from Saye's exact implicit quadrature, with the moment-fit
  structure informed by QuESo (see [`NOTICE.md`](NOTICE.md) for upstream
  attribution). The level set need not be a formula: a closed segment loop
  or triangle surface becomes a signed-distance leaf through
  `mesh_levelset` / `stl_levelset` in the `FileIO` + `MeshIO` extension.
  See [`examples/tutorials/03_immersed_fcm/`](examples/tutorials/03_immersed_fcm)
  for the analytic route and
  [`examples/applications/imported_geometry_3d/`](examples/applications/imported_geometry_3d)
  for the imported one.
- **Multi-domain coupling**: fields may live on *independent* spaces —
  each subdomain with its own mesh, level-set fold and dof block — tied
  together across a shared interface mesh by `couple(uₐ, u_b, Γ, form)`.
  A `WeakForm` gives the jump coupling `∫_Γ a(⟦u⟧, ⟦v⟧) dΓ`, expanded
  into the four field blocks with the `+ − − +` sign pattern; an
  `InterfaceForm` kernel — reading `q.normal`, the current side's
  `trial` gradient (with `sides` naming which side that is), both
  coupled fields through `q.state`, and the `onside` / `jump_sign`
  helpers — expresses any two-sided law (weighted Nitsche, cohesive,
  flux-weighted); `couple` instantiates that kernel once per side pair,
  so the two-sidedness spans the four blocks. The package supplies the
  two-sided interface integration, never the constitutive choice.
- **Order reduction on covered regions** (`reduce_order`, on by
  default): every high-order mode whose entire incidence stencil is
  covered by a finer level is eliminated, leaving the linear skeleton.
  Elimination is per-mode rather than per-cell — a mode is shed only
  when every cell it touches is covered, so an edge or face mode
  straddling the boundary of the covered region survives — and a buried
  linear mode that a *nested* finer level reproduces exactly is
  deduplicated. Coverage is mask-aware — a user-deactivated overlay cell
  does not cover, so a fully deactivated overlay still behaves like no
  overlay — while a cell folded away as fictitious does cover, because
  it carries no material. Shedding high-order modes is
  integrated-Legendre only; the deduplication also runs on a B-spline
  level, where it is what keeps a nested stack non-singular rather than
  an accuracy trade — the duplicate it removes is reproduced exactly by
  the level above. `diagnostics(model, solution).reduced_mode_counts`
  reports the count per level, concatenated field-by-field.
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
disk  = physical_domain(x -> sqrt(x[1]^2 + x[2]^2) - 0.7; lipschitz=1.0,
                        subcell_length_scale=0.02)

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

`examples/` is organised in three tiers, which answer three different
questions. **Tutorials** teach the API and are meant to be read in order.
**Applications** are recognisable engineering problems solved plainly, and are
where the package is shown working alongside third-party Julia packages.
**Reproductions** are the scientific record — the benchmarks behind the method
paper, at their published configurations.

Each example lives in its own sub-directory with a self-contained
`Project.toml` (Unfitted is wired in via `[sources]`), so example-specific
dependencies stay out of the package's own `Project.toml`. Run any example
with

```bash
julia --project=examples/<tier>/<name> examples/<tier>/<name>/<name>.jl
```

`Manifest.toml` is git-ignored, so the first run of an example resolves and
installs that example's own dependencies. Most resolve in seconds;
`applications/time_integration` pulls in the `OrdinaryDiffEq.jl` tree and takes
a few minutes the first time.

### Tutorials — start here

| Sub-directory | What it teaches |
|---|---|
| `tutorials/01_first_solve/` | The whole workflow in seven calls: `box → space → field → poisson → prepare → solve! → diagnostics`, first on an interval and then, unchanged, on a square |
| `tutorials/02_overlays/` | Local refinement by superposition: adding an overlay, masking it to a patch, `activate!`/`deactivate!`, and reading `reduced_mode_counts` |
| `tutorials/03_immersed_fcm/` | Geometry the mesh knows nothing about: a CSG level set, cut-cell moment-fit quadrature, and a Nitsche condition on an immersed boundary |
| `tutorials/04_bspline/` | Swapping the basis family to B-splines, why a nested B-spline stack needs deduplication, and B-splines on an immersed domain |

### Applications — the package on real problems

| Sub-directory | What it solves |
|---|---|
| `applications/kirsch_plate_2d/` | Plane-stress plate with a circular hole, verified against the Kirsch solution; the weak form is written in `Tensors.jl` notation |
| `applications/interface_coupling_2d/` | Bonded bi-material joint: two independently meshed subdomains tied across a shared seam by a weighted Nitsche interface form |
| `applications/imported_geometry_3d/` | Heat conduction in a non-convex L-bracket whose geometry arrives as a triangle surface mesh (`FileIO` + `MeshIO`) rather than as a formula |
| `applications/time_integration/` | Transient heat conduction where Unfitted supplies `M`, `K` and `f` once and `OrdinaryDiffEq.jl` owns the time axis |
| `applications/thermal_curing_2d/` | Irreversible thermal curing of a thermoset: the cure fraction is per-quadrature-point `QuadField` state, the conductivity depends on it (Picard per step), and the overlay activates on the cure front — with `RBFP0` carrying the state across every rebuild |

### Reproductions — the paper's benchmarks

| Sub-directory | What it reproduces |
|---|---|
| `reproductions/laplace_unit_square_smooth/` | Smooth Laplace verification on the unit square |
| `reproductions/singular_square_2d/` | 2D corner-singularity convergence with nested overlays |
| `reproductions/conditioning_small_overlap/` | Small-overlap conditioning sweep, one CSV row per (order, overlap) |
| `reproductions/traveling_heat_source_2d/` | Transient heat with hierarchical adaptive overlays and L² state transfer between mesh updates |

## Documentation

Source files are the primary documentation. Every public symbol carries a
docstring; non-trivial internal helpers carry leading comment blocks
explaining the math, the algorithmic choices, or both.

For project goals, mathematical contracts, code style, testing standards,
and the contributor / PR workflow, see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing

If you use Unfitted.jl in academic work, please cite the method paper:

> J. N. Schmäke and M. Ruess, *Unfitted Multi-Level hp Refinement for
> Localized and Moving Solution Features*, arXiv:2604.25797.

A machine-readable record is available in
[`CITATION.cff`](CITATION.cff).

## License and attribution

Unfitted.jl is distributed under the MIT License (see [`LICENSE.md`](LICENSE.md)).

The non-negative moment-fit structure in `src/fcm.jl` is informed by
[QuESo](https://github.com/manuelmessmer/QuESo) (BSD-4-Clause), used as an
algorithmic reference only (no code vendored); the cut-cell moments come
from Saye's implicit quadrature. See [`NOTICE.md`](NOTICE.md) for the full
upstream attribution.
