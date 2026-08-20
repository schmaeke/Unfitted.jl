# Contributing to Unfitted.jl

Thanks for your interest in Unfitted.jl. This document is the single source
of truth for contributors — humans and AI coding agents alike. It explains
what the package is, how its pieces fit together, the mathematical and
software contracts it commits to, and how to develop, test, and submit
changes.

If you are an AI coding agent, please also read `CLAUDE.md`: it points back
to this document and adds a few agent-specific workflow rules.

## Project mission

Unfitted.jl is a compact, high-performance, science-grade Julia
implementation of unfitted multi-level hp refinement on axis-aligned
Cartesian grids. It implements the method described in

> J. N. Schmäke and M. Ruess, *Unfitted Multi-Level hp Refinement for
> Localized and Moving Solution Features*, arXiv:2604.25797.
> <https://arxiv.org/abs/2604.25797>

The implementation is written from the method specification — not ported
from any single reference codebase. Sub-systems informed by upstream
open-source projects are attributed in `NOTICE.md` and at the top of every
file that carries the borrowed design. The notable case is the finite-cell
method moment-fit quadrature in `src/fcm.jl`. Its cut-cell moments are
computed exactly by Saye's dimension-reduction implicit quadrature on a CSG
level set (`src/implicit.jl`, `src/physical.jl`); the non-negative
moment-fit idea is

> B. Müller, F. Kummer, M. Oberlack, *Highly accurate surface and volume
> integration on implicit domains by means of moment-fitting*, Int. J.
> Numer. Methods Engng. **96** (2013) 512–528.
> [doi:10.1002/nme.4569](https://doi.org/10.1002/nme.4569).

and the non-negative least-squares moment-fit *structure* follows QuESo's
`QuadratureTrimmedElement` — used as an algorithmic reference only, with no
code vendored:

> M. Meßmer, T. Teschemacher, L. F. Leidinger, R. Wüchner, K.-U.
> Bletzinger, *Efficient CAD-integrated isogeometric analysis of trimmed
> solids*, Comput. Methods Appl. Mech. Engrg. **400** (2022) 115584.
> [doi:10.1016/j.cma.2022.115584](https://doi.org/10.1016/j.cma.2022.115584).
> QuESo source: <https://github.com/manuelmessmer/QuESo> (BSD-4-Clause).

Primary goals, in priority order:

1. Correctness and reproducibility of the numerical method.
2. Dimension-independent core algorithms for arbitrary spatial dimension
   `D ≥ 1`, not only 1D and 2D.
3. Compact, comprehensible code with minimal abstractions.
4. Efficient shared-memory execution on CPUs.
5. Pure Julia project code; focused Julia packages where they remove real
   work.
6. A basis-family design with integrated Legendre basis functions as the
   default, while keeping it straightforward to add alternatives such as
   B-splines.
7. A compact, powerful, efficient public API for common modeling,
   refinement, assembly, solving, projection, and evaluation workflows
   without hiding the numerical method.
8. Clear tests and examples that make scientific regressions obvious.

## Getting started

### Prerequisites

You will need:

  - Julia 1.10 or newer (set `julia` on your `PATH`).
  - `JuliaFormatter.jl` in your default Julia environment for the
    repository's pre-commit script. Install once with
    `julia -e 'import Pkg; Pkg.add("JuliaFormatter")'`.

The repository tracks `Project.toml`. `Manifest.toml` is intentionally
git-ignored so that contributors resolve dependencies against their own
local Julia version.

### Setting up the development environment

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

### Verifying your setup

Run the test suite and the formatter / code-statistics script:

```bash
julia --project=. -e 'using Pkg; Pkg.test("Unfitted")'
julia precommit.jl --check
```

Both commands must exit zero on a clean checkout. If they do not, please
open an issue with the failing output before opening a PR.

Tests must pass under `JULIA_NUM_THREADS=1` *and* under more than one
thread; do not rely on a fixed thread count.

## Repository layout

```text
Project.toml                  package metadata + production dependencies
LICENSE.md                    MIT (Unfitted.jl)
NOTICE.md                     third-party attributions (QuESo BSD-4-Clause)
README.md                     public landing page
CLAUDE.md                     agent-instruction pointer documents
CONTRIBUTING.md               this file — contributor + agent guide
CITATION.cff                  machine-readable citation metadata
.JuliaFormatter.toml          repo-wide formatting config (yas style, indent 4)
precommit.jl                  formatter wrapper + SLOC/comment/docstring stats
src/                          library source
test/                         unit and regression tests
examples/                     runnable scripts, one sub-directory per example, each with its own Project.toml
benchmarks/                   targeted performance benchmarks (own Project.toml)
```

### Source files (`src/`)

Each file owns one well-defined concern. Cross-file dependencies are
limited to the imports made obvious by `src/Unfitted.jl`'s include order:

| File              | Responsibility                                              |
|-------------------|-------------------------------------------------------------|
| `Unfitted.jl`     | top-level module: imports, exports, include order           |
| `geometry.jl`     | D-dimensional axis-aligned boxes, coordinate maps, tolerances |
| `physical.jl`     | CSG level-set tree, `PhysicalDomain`, three-valued cell classifier |
| `implicit.jl`     | Saye dimension-reduction implicit quadrature on the level set (volume/surface rules) |
| `basis.jl`        | basis-family interface, integrated Legendre default, tensor-product evaluation |
| `fcm.jl`          | finite-cell-method cut-cell quadrature: exact moments from `implicit.jl`, non-negative (NNLS) moment fit |
| `mesh.jl`         | Cartesian mesh levels, the superposition `Space`, per-cell activation masks |
| `intersections.jl`| admissible integration regions for non-matching meshes; cut/fictitious region-quadrature dispatch |
| `dofs.jl`         | dof layout, raw/active enumeration, overlay/boundary-constraint detection |
| `dirichlet.jl`    | physical Dirichlet boundary conditions, boundary selectors, strong elimination and Nitsche enforcement |
| `surface.jl`      | immersed-boundary surface meshes (`BoundaryMesh`) and surface-region integration |
| `problems.jl`     | `Field`, weak-form channels and blocks (`BlockForm`/`LoadForm`/`WeakForm`), `Problem` |
| `model.jl`        | `Model` lifecycle: `prepare`, `move!`/`activate!`/`deactivate!`, diagnostics |
| `assembly.jl`     | coupled Galerkin assembly: channel calculus, cached symbolic-scatter (Gustavson) pattern, serial scatter + two-phase compute→gather threaded path |
| `solvers.jl`      | small solver/preconditioner wrappers + `Solution`           |
| `projection.jl`   | variational and rewire-based state transfer between models  |
| `data.jl`         | per-quadrature-point `QuadField` + RBF transfer             |
| `postprocessing.jl` | VTK export, L² error, field/gradient evaluation           |
| `api.jl`          | thin public-API wrappers (mass, stiffness, poisson, neumann, …) |

Only split a single file when a second basis family, an alternative
transfer scheme, or an honest test burden makes the split helpful.
Resist the temptation to add framework layers prophylactically.

### Tests (`test/`)

`runtests.jl` includes one `test_*.jl` per area of responsibility. Tests
are written against the public API where possible, and use exact-value
or manufactured-solution checks rather than plots. Fast unit tests live
here; expensive convergence studies belong in `examples/`.

### Examples (`examples/`)

Self-contained runnable scripts. Each example lives in its own
sub-directory `examples/<name>/` with a `<name>.jl` driver and a
`Project.toml` that wires `Unfitted` in via `[sources] = {path =
"../.."}`. Example-specific dependencies (e.g. `StaticArrays`,
`LinearAlgebra`, or third-party packages used only for an immersed
geometry demo) belong in the example's own `Project.toml` and must
never be added to the package's top-level `Project.toml`. Per-example
`Project.toml` files set `julia = "1.11"` in `[compat]` because
`[sources]` is a Julia 1.11 feature; the package itself still
supports Julia 1.10.

A shared helper `examples/reporting.jl` is `include`d by every
example via `joinpath(@__DIR__, "..", "reporting.jl")` and depends
only on what `Unfitted.jl` already exports.

Run an example with

```bash
julia --project=examples/<name> examples/<name>/<name>.jl
```

Output artifacts go under `examples/<name>/output/`, which is
git-ignored repository-wide.

Examples should demonstrate the *public* API; reach into internals
only when an example is specifically about those internals.

### Benchmarks (`benchmarks/`)

Small targeted benchmarks with their own `Project.toml`. Driven and
documented by `benchmarks/runbenchmarks.jl`.

### Developer tooling

The repository root ships `precommit.jl`, a thin wrapper around
`JuliaFormatter` that also prints per-file and per-bucket SLOC / comment /
docstring statistics. Three modes:

```bash
julia precommit.jl          # format every tracked .jl file in place + print stats
julia precommit.jl --check  # verify formatting; non-zero exit if anything would change
julia precommit.jl --stats  # stats only; does not require JuliaFormatter installed
```

CI and pre-merge checks should run `julia precommit.jl --check` and treat
a non-zero exit as a hard failure.

## Reference material

The method specification lives in the UMLHP preprint cited above. When
working on geometry, basis evaluation, integration regions, assembly,
constraints, projection, or field evaluation, the paper is the
authoritative source for the contracts the code must honor.

When porting an algorithm from an upstream open-source project, follow
this protocol:

  1. Read the upstream source carefully enough to understand the
     algorithm, not just the surface API.
  2. Write down the smaller domain-specific Julia design you intend to
     implement *before* writing code. The port should not be a
     line-by-line translation; it should be a Julia design informed by
     the algorithm.
  3. Preserve attribution at the top of the implementing file and in
     `NOTICE.md`. Include the canonical citation (authors, paper title,
     journal/volume/year, DOI, license, source URL).
  4. Never import or runtime-depend on upstream code from outside the
     Julia package ecosystem inside `src/`, `test/`, or `examples/`.

QuESo's source under `github.com/manuelmessmer/QuESo` is the algorithmic
reference for the *non-negative moment-fit structure* in `src/fcm.jl`; it
is a design reference only and no QuESo code is vendored. The deliberate
deviations from that source — a CSG level-set geometry kernel instead of a
B-rep triangulation, and *exact* moments from Saye's implicit quadrature
(`src/implicit.jl`) instead of a divergence-theorem surface integral with a
point-elimination retry — are documented at the top of `src/fcm.jl`.

## Method and architecture

The mathematical contracts below are more important than code style.
Code that violates them is broken, regardless of how clean it looks.

### The superposition model

An approximate solution `u_h(x)` is the sum of overlay contributions, one
per mesh level:

```text
u_h(x) = Σₖ u_h^(k)(x)
```

Each overlay contribution `u_h^(k)` is extended by zero outside its
overlay domain `Ω^(k)`. The base level covers the full physical domain
`Ω`; subsequent overlay levels `k > 0` cover proper sub-regions of `Ω`,
chosen to refine where the solution needs more resolution.

Overlay levels are *not* topologically merged with their parents, and the
package contains no hanging-node machinery. Instead, every overlay level
imposes homogeneous boundary conditions on its *artificial overlay
boundary*:

```text
Γ_o^(k) = ∂Ω^(k) \ ∂Ω
```

— i.e. the part of the overlay's boundary that does not coincide with the
physical boundary `∂Ω`. These artificial constraints are imposed
strongly (eliminated dofs for the integrated Legendre basis, homogeneous
linear constraints for general basis families). They are kept strictly
separate from *physical* boundary conditions on `∂Ω`: mixing the two is
explicitly disallowed.

### Dimension independence

The core library is written for arbitrary spatial dimension `D ≥ 1`.
Geometry, meshes, element lookup, basis evaluation, integration regions,
assembly, constraints, projection, and field evaluation are all
parameterized by `D`. There are no separate 1D / 2D / 3D code paths
shadowing a generic implementation.

In practice this means:

  - Prefer `NTuple{D,T}`, `SVector{D,T}`, `CartesianIndex{D}`,
    `CartesianIndices`, and per-axis tuples/vectors over named `x`, `y`,
    `z` fields.
  - A `D`-dimensional cell is a tensor-product hypercube/box: an
    interval in 1D, a quadrilateral in 2D, a hexahedron in 3D, a `D`-cube
    in general. Do not encode assumptions that cells are only intervals,
    quads, or hexes.
  - Express tensor-product loops through multi-indices, recursion,
    generated functions, or `CartesianIndices` — whichever is simplest
    and type-stable.
  - Dimension-specific examples, plotting, and visualization are
    permitted. Core algorithms must not special-case 1D, 2D, or 3D
    unless the branch is explicitly justified, tested, and still leaves
    the generic `D` path intact.

High dimensions are expensive because tensor-product quadrature grows
rapidly with `D`; correctness and generic behavior still matter.

### Basis family model

Basis families are a first-party concept of the codebase. The default is
a hierarchical integrated Legendre basis on tensor-product Cartesian
cells. Integrated Legendre assumptions live exclusively in the basis and
dof-layout layer; the rest of the codebase queries the basis through a
small interface.

A basis family must provide, directly or through a compact trait-like
interface:

  - the number and identity of local basis functions for a cell or span;
  - the active basis functions covering an element or integration box;
  - values and physical/reference gradients at quadrature points;
  - a tensor-product ordering convention;
  - support information sufficient for overlap queries and assembly;
  - boundary or entity association information needed for physical
    Dirichlet conditions and artificial overlay constraints;
  - quadrature-order recommendations when exactness depends on the
    family and polynomial degree.

The integrated Legendre implementation uses 1D hierarchical endpoint and
interior modes, then builds `D`-dimensional tensor-product bases from
those 1D factors. The mode-0 and mode-1 functions are linear "endpoint"
shape functions; modes m ≥ 2 are "bubble" functions that vanish at the
cell boundaries.

Adding a B-spline (or any other) basis should not require rewriting
geometry, intersections, assembly, projection, or solvers. It should
primarily mean adding a basis-family implementation and its dof /
constraint behavior, plus the corresponding tests.

Do not assume that all basis functions are element-local, nodal,
interpolatory at boundaries, or associated with mesh vertices. Those are
properties of *some* families, not of the framework.

### Mesh and refinement model

Meshes are axis-aligned Cartesian tensor-product meshes. Each level
carries its own mesh, basis family, polynomial/order metadata, and dofs.
Overlay levels are positioned independently of lower levels: overlay
boundaries need not coincide with lower-level element boundaries. There
is no fitted multi-level hp deactivation rule for linear independence.

Cells of a level may also be selectively activated or deactivated via a
per-cell `LevelMask`. See *Selective activation* below.

### Degree-of-freedom enumeration and constraints

Physical boundary conditions and overlay constraints are assigned before
final global dof enumeration. Constrained and active dofs are kept
explicitly distinguishable. The dof layer:

  - eliminates homogeneous overlay constraints directly when the basis
    family supplies dof-wise constraints;
  - applies nonzero physical Dirichlet data via consistent RHS updates
    when columns are eliminated;
  - if a basis family needs homogeneous linear constraints rather than
    dof-wise elimination, keeps that logic in the constraint/dof layer
    and tests it separately.

Tests must check that overlay contributions vanish on artificial overlay
boundaries and that physical boundary conditions are not accidentally
applied to artificial overlay boundaries.

### Coupling and assembly

Assembly produces *one* coupled variational problem over all active
spaces. Every overlapping active basis-function pair across all relevant
levels is included. Level-wise uncoupled subproblems are never an
acceptable substitute for the coupled Galerkin system.

Symmetric bilinear forms are preserved to roundoff. For SPD model
problems the constrained global matrix is SPD unless a known dependency
or nullspace is present. Assembly queries the basis/dof layer for active
local functions and local-to-global maps — it must not assume a fixed
formula such as `(p + 1)^D` outside integrated Legendre basis code.

### Integration regions

For non-matching axis-aligned meshes, admissible integration boxes are
constructed as follows:

  1. For each coordinate direction, collect all element-boundary
     coordinates from the participating meshes.
  2. Sort and merge coordinates closer than the geometry tolerance.
  3. Form non-degenerate intervals between adjacent coordinates.
  4. Take the Cartesian product of intervals to form candidate boxes.
  5. Use midpoint containment to identify which parent element of each
     mesh covers each candidate box.
  6. Keep boxes that satisfy the caller's criterion: usually "at least
     the required participating active fields" for coupling assembly,
     "at least one active field" for field evaluation or visualization,
     or source/target coverage for projection.
  7. For every kept box, store each contributing parent element and the
     box mapped into that element's local coordinates.

The integrand of a basis-function product is smooth (`C^∞`) inside such
an admissible box. Do not integrate a product of basis functions across
an element-boundary kink without partitioning the region first.

### Moving overlays and state transfer

When overlay positions change:

  - rebuild affected intersection regions and level couplings;
  - reuse geometry-dependent matrices only when the geometry and active
    spaces are unchanged;
  - update time-dependent load vectors independently;
  - transfer state by a variational projection (such as an `L²`
    projection or an energy projection), not by pointwise interpolation;
  - test projection on constants, low-order polynomials, and fields
    represented exactly in both spaces.

### Small overlaps and conditioning

Tiny integration boxes — those much smaller than the geometry tolerance
relative to the domain — are a known conditioning hazard for unfitted
methods. The package detects and reports them in the assembly
diagnostics instead of silently dropping or rescaling them.

  - Track condition numbers or solver iteration counts in tests and
    benchmarks for representative small-overlap cases.
  - Do not silently drop, merge, or rescale tiny overlap regions in
    production assembly unless the change is explicitly tested and
    documented as a numerical method choice.
  - Prefer making conditioning behavior visible before adding
    stabilization.

## Public API design

The public API is a first-party design concern. It is compact enough to
learn quickly but powerful enough to express the paper's workflows
without forcing users to manipulate low-level dof tables, constraint
masks, or intersection-region internals.

### Design goals

  - **Short common path.** Create a Cartesian base discretization, add
    one or more overlay levels, choose basis families and orders, impose
    physical boundary conditions, define sources / material coefficients,
    assemble, solve or step in time, evaluate fields, move overlays, and
    project state — all with a small number of clear calls.
  - **Expert escape hatches.** Users can inspect or provide quadrature
    rules, tolerances, basis families, solver/preconditioner choices,
    overlay geometry, and projection forms without rewriting assembly
    code.
  - **Methods exposed directly.** Names and objects reflect domains,
    levels, overlays, basis families, active spaces, integration
    regions, constraints, forms, projections, and solutions.
  - **Keyword constructors, small configuration structs.** Prefer
    keyword arguments and small typed configuration structs over long
    positional argument lists or untyped dictionaries.
  - **Sensible defaults.** Integrated Legendre basis, conservative
    quadrature, robust tolerances, standard sparse assembly, and a
    direct solver out of the box. Defaults are documented and easy to
    override.
  - **No hidden global state.** API calls are deterministic from their
    arguments and the project configuration.
  - **Thin wrappers.** High-level API wrappers compose lower-level
    tested kernels rather than duplicate geometry, basis, constraint, or
    assembly logic.
  - **Performance through the API.** Convenience never forces dynamic
    dispatch, type instability, avoidable heap allocation, or repeated
    preprocessing in hot loops.
  - **Inspectable objects.** A user can query active unknowns, level
    metadata, basis metadata, integration-region counts, constraints,
    solver diagnostics, residuals, and small-overlap warnings.
  - **Actionable errors.** Distinguish invalid overlay placement,
    inconsistent dimensions, unsupported basis-family constraints,
    singular systems, and small-overlap conditioning warnings — by
    error type and message.

### Common workflows the API must keep easy

  - Static Poisson / Laplace-style scalar problems on a box domain.
  - 1D bar-style problems with point loads or discontinuous
    coefficients and sources.
  - Adding centered or explicitly positioned overlay boxes with
    per-level order choices.
  - Selectively activating or deactivating individual overlay cells as
    a feature evolves in time.
  - Attaching an immersed physical domain via a level-set function and
    integrating only over `Ω`.
  - Moving an overlay and rebuilding only the affected geometry and
    coupling data.
  - Variational projection from an old space to a new space.
  - Evaluating a superposed solution at points or on a visualization
    grid.
  - Producing a minimal reproducibility report: `D`, basis family,
    orders, levels, active unknowns, integration regions, cut-region
    counts, residual and error metrics, and solver diagnostics.

Every public API addition must have:

  - a docstring with argument conventions and defaults;
  - at least one small test that uses the public call rather than
    internals;
  - a check that dimensions and basis families are handled generically
    where applicable;
  - an example or test showing the simplified workflow if it replaces
    tedious manual setup.

### Selective activation (per-cell mask)

Individual cells of a level can be marked active or inactive. This is
how transient analyses light up overlay cells only where small-scale
features evolve.

  - **Construction-time mask** via `space(...; active=…)` and
    `overlay(...; active=…)`. Accepted shapes: `nothing` (default, all
    cells active), an `AbstractArray{Bool,D}` matching the mesh's cell
    grid, an iterable of `CartesianIndex{D}` listing active cells, or a
    predicate `(cell_box, cell_index) -> Bool`.
  - **Mutation**: `activate!(model; level, cells)` and
    `deactivate!(model; level, cells)`. The mutators follow the `move!`
    invalidation contract: bump `model.version`, clear `model.matrix`
    and `model.rhs`, rebuild the integration plan and dof layout,
    refresh diagnostics. An outstanding `Solution` raises on stale
    reuse.
  - **Query**: `active_cells(model; level)` returns a copy of the
    level's `BitArray` (or an all-true array for unmasked levels).
  - **Dof layer treatment**: faces between active and inactive cells of
    the same level are artificial overlay constraints, analogous to
    mesh-box-face constraints. Span-mode dofs of inactive cells are not
    enumerated.

All masking lives at cell granularity; sub-cell activation is out of
scope and would require a different abstraction.

### Immersed boundary (finite cell method)

The package supports the finite cell method with non-negative moment-fit
quadrature on cut cells. The geometry of `Ω` is a **CSG level set**: a
Boolean combination of smooth leaves, each a scalar function `f` with
`Ω_leaf = { x : f(x) ≤ 0 }`. A single leaf is the familiar
`Ω = { x : φ(x) ≤ 0 }`; the combinators `intersect`, `union`, `setdiff`,
and `complement` build the rest (e.g. an annulus as a disk minus a disk).
Carrying the leaves separately — rather than collapsing them into one
`min`/`max` level set — keeps every boundary piece smooth, so the implicit
quadrature kernel stays high-order across creases and corners. Boolean
indicator representations are intentionally not supported.

  - **Build geometry** with [`leaf`](@ref) and the CSG combinators:
    `leaf(f; lipschitz=Inf)`, `intersect(a, b…)`, `union(a, b…)`,
    `setdiff(a, b)`, `complement(a)`. A bare callable is auto-wrapped as a
    default leaf, so the single-level-set case stays ergonomic.

  - **Public construction**:
    `physical_domain(geometry; lipschitz=Inf, alpha=0.0, subcell_length_scale,
    max_depth=8, moment_order_factor=2, target_residual=1e-6)`.

    - `geometry`: a `LevelSet` CSG tree, or a bare scalar callable `φ` on
      `SVector{D,T}` (auto-wrapped as a single leaf). `φ` need not be a
      true signed-distance function, but must accept `ForwardDiff.Dual`
      arguments so the quadrature kernel can take its gradient.
    - `lipschitz`: Lipschitz constant `L` of the auto-wrapped single leaf.
      `1.0` for a true SDF; `Inf` disables the cheap "uniform sign"
      certificate and forces classification by corner sampling and
      subdivision. For a CSG tree, set each leaf's constant on its
      `leaf(...)` call instead.
    - `alpha`: fictitious-region weight for α-FCM stabilization. `0` is
      the strict cut path; `> 0` enables α-FCM (cut cells are enriched
      with the α-scaled full-cell rule unconditionally, and fully
      fictitious cells carry α-scaled weights). Independently, fully
      fictitious *cells* are dropped from the dof layout by default and
      retained only when `keep_fictitious = true` (which then requires
      `α > 0`). The α-scaled cut-cell enrichment applies regardless of
      `keep_fictitious`, so pre-`keep_fictitious` whole-cell results are
      not recovered by setting it.
    - `subcell_length_scale` (required): target box size, in physical
      units, for the binary subdivision shared by two consumers — the
      cut/full/fictitious cell classifier, and the implicit kernel's
      fallback subdivision on *non-graph-like* cut cells (a leaf with a
      turning point inside the box). It is a geometry-robustness knob
      (resolving thin or near-tangent features), **not** a moment-accuracy
      grid: smooth, graph-like cut cells get exact, depth-independent
      moments and are never subdivided. One `PhysicalDomain` then serves
      coarse and fine levels of a superposition `Space`.
    - `max_depth` (default `8`): hard cap on the subdivision depth of both
      consumers above.
    - `moment_order_factor`: multiplier on the moment-fit basis order per
      axis. `2` (default) integrates trial × test products exactly on the
      moment basis, matching what tensor Gauss does on `:full` regions;
      `1` halves the basis but only integrates degree-`p` integrands
      exactly.
    - `target_residual`: target L² residual for the moment fit. The exact
      kernel reaches far below the `1e-6` default in a single NNLS solve,
      so this only bounds a small conditioning retry (a denser candidate
      cloud) — never moment accuracy or subdivision depth.

  - **Attach** with `space(omega; ..., physical=physical_domain(…))`.
    The default `physical=nothing` keeps the no-FCM hot path.

  - **Imported geometry**: with `FileIO` and `MeshIO` loaded,
    `mesh_levelset(mesh)` turns a closed `BoundaryMesh` (a 2D segment loop
    or a 3D triangle surface) into a signed-distance leaf, and
    `stl_levelset("part.stl")` reads an STL into one. Both compose with the
    CSG combinators (see `ext/UnfittedMeshIOExt.jl`).

  - **Region quadrature kinds** (visible via `region.quadrature.kind`):
    `:full` (tensor Gauss), `:fictitious_alpha` (α-scaled tensor Gauss),
    `:cut_fitted` (NNLS moment-fit rule, plus the α-scaled tensor part
    when `α > 0`), `:cut_fallback` (the moment-fit residual exceeded the
    failure threshold, so the region carries the raw Saye volume rule the
    moments were summed from — correct and non-negative, but with 50–200×
    the points), `:cut_failed` (strict-cut `α = 0` region whose `Ω ∩ box`
    carries no volume rule at all; region contributes zero quadrature),
    `:cut_alpha_failed` (the same under `α > 0`: the empty physical part
    is dropped but the α-scaled tensor rule is retained so the cell's
    dofs stay α-stabilised — a nonzero rule). The assembly hot loop
    is unchanged — it just iterates `zip(points, weights)`.

  - **Moment-fit defaults**: moment-fit basis order = `moment_order_factor
    × max(level.order)` per axis over the region's parents; exact tensor
    Legendre moments from the Saye volume rule; a single Lawson–Hanson
    NNLS solve (`NonNegLeastSquares.jl`) selects ≤ `nbasis` non-negative
    weights; up to 3 attempts, retrying only with a denser candidate cloud
    for NNLS conditioning (higher fiber Gauss order *and* an eight-fold
    larger candidate budget per attempt), never with more subdivision; if
    no attempt fits, the raw volume rule is used as a fallback rather than
    dropping the cell. Rules are cached by canonicalized region bounds +
    moment order.

  - **Diagnostics**: `diagnostics(model, solution).cut_region_count`,
    `fit_failure_count`, `moment_fit_residual_max`,
    `cut_fallback_count`, `cut_fallback_points`,
    `inactive_cell_counts`. A nonzero `cut_fallback_count` means those
    cells are under-resolved for their geometric complexity and should
    drive refinement; the fallback is a safety net, not a substitute for
    an adequate mesh.

### Naming

Prefer concise verbs and nouns such as `mesh`, `overlay`, `basis`,
`space`, `assemble`, `solve`, `project`, `move`, `evaluate`, and
`diagnostics` over framework-heavy terminology. The public API should
read like the method, not like a software architecture.

## Code style and documentation

The actual code is held to a strict compactness standard. The
documentation around it is intentionally generous: source files are the
documentation, and every non-trivial entity should be possible to
understand from the file alone.

### Source-code compactness

  - Keep the public API small, expressive, and stable. Add public names
    only when they simplify a real user workflow or expose an essential
    method concept.
  - Prefer simple concrete structs over abstract hierarchies.
  - Use parametric types for dimension and scalar type where it
    improves performance or clarity (e.g. `AxisBox{D,T}`).
  - Avoid designing a general-purpose FEM library. This project is
    specialized to axis-aligned unfitted multi-level hp refinement.
  - Favor short functions with explicit data flow.
  - Avoid hidden global state.
  - Avoid macros unless they remove repeated boilerplate in hot kernels
    and are tested.
  - Avoid dynamic dispatch in quadrature, mapping, and assembly hot
    loops.
  - Precompute shape values, derivatives, quadrature rules, active
    basis lists, and local-to-global maps when reused.
  - Use tolerances intentionally. Do not compare floating-point
    coordinates with `==` except for canonicalized grid coordinates or
    tests designed for exact values.
  - Keep `D` visible in types where practical. Avoid storing dimension
    only as a runtime integer in hot paths.
  - Never write separate 1D, 2D, and 3D implementations of the same
    core algorithm. Write one `D`-generic implementation and wrap it in
    dimension-specific examples if needed.

### Documentation and comments

Source files are the primary documentation. Every non-trivial entity
should be possible to understand from the file alone. Write for a
reader who knows conventional finite elements but not the advanced
techniques specific to this method.

The rules:

  - **Every public symbol** — exported function, struct, abstract type,
    or constant — gets a docstring written in full sentences. The
    docstring covers arguments, return value, defaults, conventions, and
    any non-obvious behavior. Mention the section of the paper or the
    cited algorithm when it informs the implementation.
  - **Every non-trivial internal function** gets either a docstring or
    a multi-line `#` comment block immediately above the definition.
    "Non-trivial" means anything that a new reader could not predict
    from the name and signature alone. Tiny obvious helpers
    (`_role_id(::Symbol)`, single-line predicate wrappers) do not need
    one.
  - **Explain the *why*, not the *what*.** The code already states what
    it does. Comments and docstrings exist for the reasoning: the
    invariant being preserved, the numerical-convention being honored,
    the choice between approaches, the reason a particular allocation
    pattern matters.
  - **Inline commentary in complex function bodies.** When a function
    body runs more than a handful of lines or chains together
    non-obvious steps — basis-table updates, sparse-scatter emission,
    boundary projection, retry loops — place a short comment above
    each non-obvious step explaining why it is there. Use one-line
    section headers within a long body to mark distinct stages. A
    reader should be able to follow the implementation by skimming
    the comments before reading the code.
  - **Math goes in unicode**, not LaTeX. Write `Σₖ u_h^(k)(x)`,
    `Ω = { x : φ(x) ≤ 0 }`, `‖ A x − b ‖₂`, not `\sum_k`, `\Omega`,
    `\|Ax-b\|_2`. ASCII pseudocode is the fallback when unicode does
    not render the expression cleanly.
  - **Cite sources properly.** When code implements an algorithm from a
    paper, cite the paper with authors, title, journal, year, DOI, and
    the specific section or equation. Citations to local files
    (e.g. anything in a `refs/` working directory) are not acceptable
    in the package source — the source must stand on its own.
  - **Open-source ready prose.** Write as if the file will land on
    GitHub today, because it might. Avoid implementation notes,
    apologies, references to TODOs unless they reference an open
    issue, or internal jargon.

The companion script `precommit.jl --stats` reports per-file and total
SLOC, comment, and docstring counts. SLOC restrictions apply only to
actual source code; comment and docstring volume is unrestricted.

#### Style coherence

All source files share the same documentation and comment style. New
files match existing ones; touched files are nudged toward the standard
below so the codebase converges over time.

  - **Docstring layout.** Triple-quoted. The opening lines are the call
    signatures (each indented four spaces, one per supported overload),
    followed by a blank line and prose in full sentences. Argument
    conventions, defaults, return value, and citations follow in that
    order. The `PhysicalDomain` and `physical_domain` docstrings at
    the top of `src/physical.jl` are fully worked examples.

  - **Section dividers in long files.** When a file has clearly
    separable stages or concerns (`src/fcm.jl`'s NNLS / moment / fit
    pipeline; `precommit.jl`'s CLI / discovery / formatting /
    stats / main) group them with single-line `─` dividers carrying the
    section name. The canonical form:

    ```julia
    # ── Stage name ────────────────────────────────────────────────────────────────
    ```

    Use the unicode light-horizontal character `─` (U+2500), not ASCII
    `-` or `=`. Short files with one tightly focused concept do not
    need dividers.

  - **File-top attribution blocks** for ported files are `#` comment
    blocks (not docstrings) and appear before any code. They carry:
    the algorithm name; the full upstream citation with authors, title,
    journal, year, and DOI; the upstream license; the source URL; and
    a brief list of deliberate deviations from upstream. The top of
    `src/fcm.jl` is the model.

  - **Internal-helper commentary** sits *above* the function or struct
    definition, as either a docstring or a `#` block. Trailing inline
    comments on a function header line are not used.

  - **Math and citations** inside docstrings and comments follow the
    same rules as the surrounding prose: unicode math, full citations
    with DOI, no references to local working directories.

### Coordinate conventions

The conventions used throughout the codebase, in one place so files do
not have to re-derive them:

  - **Physical coordinates** `x ∈ Ω ⊂ ℝᴰ`: the user-facing world frame.
  - **Reference cell coordinates** `ξ ∈ [−1, 1]ᴰ`: per-cell reference
    frame used by basis evaluation and quadrature.
  - **Region-local sub-box coordinates** `η ∈ [−1, 1]ᴰ`: per-region
    reference frame used by integration; each parent cell's local frame
    is recovered via `reference_to_physical(parent.local_box, η)`.
  - **Tensor-product basis ordering**: lexicographic over
    `CartesianIndices(ntuple(d -> 0:order[d], D))`, with the dimension-1
    axis varying fastest.
  - **Global dof ordering**: per-field, then component-major within a
    field. Constrained dofs are not enumerated; their value is held in
    `layout.constrained_values`.

Tests enforce these conventions. Document any change to them at the top
of the relevant source file and update the relevant tests in the same
PR.

### Dependency policy

Use Julia standard libraries first: `LinearAlgebra`, `SparseArrays`,
`SuiteSparse`, `Printf`, `Random`, `Test`. Focused dependencies are
acceptable when they remove real work:

  - fixed-size small arrays for geometry and local kernels, e.g.
    `StaticArrays`;
  - Gaussian quadrature, e.g. `FastGaussQuadrature`;
  - iterative Krylov solvers or preconditioners through a narrow
    package;
  - non-negative constrained linear-least-squares solvers, e.g.
    `NonNegLeastSquares` (used by `src/fcm.jl` for the NNMF moment-fit
    solve);
  - basis-family support packages only when a non-default basis is
    added and the dependency is smaller than implementing the needed
    operations;
  - VTK output for examples, e.g. `WriteVTK`;
  - development quality tools, e.g. `BenchmarkTools`, `JuliaFormatter`,
    `Aqua`, `JET`, `Documenter`.

Rules:

  - Ask before adding a large FEM framework, mesh generator, symbolic
    system, plotting stack, or non-Julia build step.
  - Do not add a dependency to avoid implementing a small axis-aligned
    operation that is central to this method.
  - Do not add production dependencies without a short justification in
    the PR summary.
  - `Manifest.toml` is git-ignored. Do not change that.

## Performance

### Threading model

  - Parallelize over independent integration regions, elements, or
    batches.
  - The threaded matrix assembly is a deferred compute→gather: phase 1
    computes each region's local block and rhs into its OWN disjoint
    arena slice (no shared writes, dynamically load-balanced); phase 2
    sums the arena into the sparse operator by a disjoint column
    partition and into the rhs by a disjoint dof partition. Both phases
    are barrier-free and lock-free, and the symbolic gather layout is
    cached on the pattern so repeated assembly pays for it once.
  - Do not push into a shared vector from multiple threads without a
    disjoint-output partition or explicit synchronization; prefer arena
    slices reduced by a fixed-order gather over thread-local accumulators
    (whose peak memory scales with the thread count).
  - Avoid locks and atomic *accumulation* in the hot scatter. A racy atomic
    scatter is nondeterministic and was measured to corrupt the solve of an
    ill-conditioned (cond ≈ 3e16) system; a fixed-order gather does not. (An
    atomic counter that only hands out region indices for load balancing is
    fine — it never touches the shared output.)
  - Threaded assembly is **bit-identical to serial**: every slot is
    summed in the serial (region, row) order, so results are reproducible
    run-to-run and independent of the thread count. Tests may assert
    equality, not merely a roundoff tolerance.

### Allocation guidelines

  - Keep local element kernels allocation-free after setup.
  - Use `@inbounds` only where bounds are protected by tests.
  - Use `@views` where slicing would allocate.
  - Run `@code_warntype`, `@allocated`, or `JET` on new hot paths when
    performance matters.
  - Benchmark before adding complicated optimizations.

For performance-relevant changes, run the targeted benchmarks before
and after:

```bash
julia --project=. benchmarks/intersections.jl
julia --project=. benchmarks/assembly.jl
```

## Testing

Every feature needs tests. Prefer small deterministic tests over large
demos.

### Minimum coverage

  - `AxisBox` construction, containment, intersection, volume/measure,
    and tolerance merging in 1D, 2D, 3D, and at least one 4D smoke
    case.
  - Cartesian mesh construction in 1D, 2D, 3D, plus a small 4D smoke
    test for mesh indexing and element lookup if inexpensive.
  - Parent-element lookup by midpoint.
  - Physical-to-reference and reference-to-physical maps for arbitrary
    `D`.
  - Integrated Legendre 1D basis values, endpoint behavior, gradients,
    polynomial exactness, and quadrature consistency.
  - Tensor-product basis ordering, values, and gradients in 2D and 3D,
    plus one low-order 4D smoke test.
  - Basis-family interface tests that fail if assembly or constraints
    rely on integrated-Legendre-only assumptions outside the basis
    layer.
  - Overlay boundary constraint detection.
  - Global dof enumeration after constraints.
  - Integration-region construction for non-matching mesh pairs in 1D,
    2D, and 3D, plus a small 4D smoke case.
  - Symmetry and SPD behavior for simple Poisson/Laplace and 1D bar
    systems after constraints.
  - Correct RHS handling for nonzero Dirichlet data.
  - Variational projection between two shifted overlay configurations.
  - Public API workflow tests that build and solve at least one small
    problem without manual dof enumeration, manual constraint masks,
    or manual intersection-region construction.
  - Regression examples based on the paper: 1D elastic bar with
    discontinuous strain, 2D singular corner problem, small-overlap
    conditioning smoke test, and a moving heat-source smoke or reduced
    regression test.
  - Selective activation: dof-structure equivalence between a
    `LevelMask` and a geometrically smaller overlay covering the same
    active cells; artificial-boundary constraint on internal
    active/inactive interfaces; `activate!`/`deactivate!` invalidation
    contract.
  - Immersed boundary (`PhysicalDomain`): strict-α cell-level
    fictitious fold drops the right cells; the expected
    `:full` / `:cut_fitted` / `:fictitious_alpha` / `:cut_fallback` /
    `:cut_failed` / `:cut_alpha_failed` region-kind tags appear; the
    candidate-cloud retry rescues a many-feature cut cell that the first
    attempt cannot fit, and the raw-volume-rule fallback still integrates
    a cell whose fit fails outright; α-FCM weight scaling on a
    fully fictitious cell; NNMF moment reproduction to within `target_residual`;
    end-to-end SPD on a cut-disk Poisson; 1D analytic mass-entry check
    on a cut cell.

### Testing rules

  - Do not rely on plots for correctness.
  - Include analytic or manufactured-solution checks wherever possible.
  - Separate fast unit tests from expensive convergence studies.
  - Expensive studies belong in `examples/`, `benchmarks/`, or a
    clearly marked testset that is not required for every quick edit.
  - When fixing a bug, add a regression test that fails without the
    fix.
  - When adding a new basis family, add at least one small assembly
    test, one boundary-constraint test, and one projection/evaluation
    test for that family.

### Reporting numerical results

For any model problem reported in a PR, issue, paper, or example, give
at least:

  - the public API calls or example script used to reproduce the run;
  - number of active unknowns;
  - number of integration regions;
  - basis family and order/knot metadata;
  - spatial dimension `D`;
  - matrix symmetry check for symmetric forms;
  - solver/factorization used;
  - residual norm;
  - the relevant analytic error or regression metric;
  - any small-overlap warning count.

Do not claim agreement with the paper unless the same setup, polynomial
degrees, overlay sizes, quadrature, basis family, and error definitions
are implemented — or unless the differences are clearly stated.

## Submitting changes

### Workflow

  1. Branch from `main` with a descriptive name. Short topic branches
     (`feature/...`, `fix/...`, `docs/...`) are preferred.
  2. Make one focused change per PR. Keep diffs small and reviewable.
  3. Add or update tests alongside the implementation.
  4. Run `julia --project=. -e 'using Pkg; Pkg.test("Unfitted")'` and
     `julia precommit.jl --check`. Both must pass.
  5. Open a pull request with a description that explains the *why*,
     the user-visible change (if any), and any numerical results the
     change produces.

### Pull-request expectations

A PR description should include:

  - A one-line summary suitable for the changelog.
  - A short rationale: what problem this solves, and why this approach.
  - A list of files touched and the gist of what changed.
  - Test changes: new tests added, existing tests adjusted (and why).
  - Numerical results if applicable, following the "Reporting
    numerical results" section.
  - Any open questions or follow-up tasks.

### Definition of done

A task is done only when:

  - the implementation is compact and specialized to this method;
  - relevant tests pass under at least one and many threads;
  - new behavior is covered by tests, or by a justified benchmark or
    example;
  - numerical invariants are preserved;
  - core changes remain dimension-independent unless explicitly scoped
    otherwise;
  - public API changes are documented, tested, and justified by a real
    user workflow;
  - common workflows remain accessible without manual low-level
    bookkeeping;
  - basis-family assumptions are localized to the basis and dof-layout
    layer and documented;
  - performance-sensitive changes avoid obvious allocations in hot
    loops;
  - documentation or comments explain any new convention;
  - upstream attributions in `NOTICE.md` and at the top of ported files
    remain accurate after any algorithmic edits to ported code;
  - the PR description includes verification commands and outcomes.

## Anti-patterns

These are the things we do not do, ever, without a very good reason
spelled out in the PR:

  - Build a general FEM framework. This project is specialized.
  - Let the public API become a dumping ground for internals or one-off
    helpers.
  - Force users through manual dof enumeration, constraint masking, or
    intersection bookkeeping for common workflows.
  - Hard-code the core library to 1D, 2D, or 3D.
  - Assume integrated Legendre basis behavior outside the basis and
    dof-layout layer.
  - Make B-splines or other basis families require rewriting geometry,
    intersections, assembly, projection, or solvers.
  - Hide conditioning problems by silently dropping tiny overlaps.
  - Use point interpolation for state transfer after moving overlays.
  - Mix physical and artificial boundary constraints.
  - Introduce Boolean-indicator geometry representations for the
    immersed domain — the package commits to signed scalar level sets.
  - Vendor third-party code without adding (or updating) the
    corresponding entry in `NOTICE.md` and a top-of-file attribution.
  - Add non-Julia production code.
  - Add heavy dependencies without approval.
  - Leave generated plots, scratch files, or benchmark artifacts in the
    repository unless explicitly requested.

## Questions, ideas, contributions

Feel free to open an issue for design discussion before opening a PR
for substantial changes — especially anything that touches the public
API surface, the math contracts, or the threading model.

Thank you for contributing.
