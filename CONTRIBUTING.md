# Contributing to Unfitted.jl

Thanks for your interest in Unfitted.jl. This document is the single source of truth for contributors — humans and AI
coding agents alike: what the package is, how its pieces fit together, the mathematical and software contracts it
commits to, and how to develop, test, and submit changes. AI coding agents should also read `CLAUDE.md`, which points
back here and adds a few agent-specific workflow rules.

## Contents

  - [Project goals](#project-goals)
  - [Getting started](#getting-started): [Prerequisites](#prerequisites) ·
    [Setting up the development environment](#setting-up-the-development-environment) ·
    [Verifying your setup](#verifying-your-setup)
  - [Repository layout](#repository-layout): [Source files (`src/`)](#source-files-src) ·
    [Package extensions (`ext/`)](#package-extensions-ext) · [Tests (`test/`)](#tests-test) ·
    [Examples (`examples/`)](#examples-examples) · [Benchmarks (`benchmarks/`)](#benchmarks-benchmarks) ·
    [Developer tooling](#developer-tooling)
  - [Reference material](#reference-material)
  - [Method and architecture](#method-and-architecture): [The superposition model](#the-superposition-model) ·
    [Dimension independence](#dimension-independence) · [Basis family model](#basis-family-model) ·
    [Mesh and refinement model](#mesh-and-refinement-model) ·
    [Selective activation (per-cell mask)](#selective-activation-per-cell-mask) ·
    [Per-cell polynomial order](#per-cell-polynomial-order) ·
    [Degree-of-freedom enumeration and constraints](#degree-of-freedom-enumeration-and-constraints) ·
    [Integration regions](#integration-regions) · [Coupling and assembly](#coupling-and-assembly) ·
    [Immersed boundary (finite cell method)](#immersed-boundary-finite-cell-method) ·
    [Automated hp adaptivity](#automated-hp-adaptivity) ·
    [Moving overlays and state transfer](#moving-overlays-and-state-transfer) ·
    [Small overlaps and conditioning](#small-overlaps-and-conditioning)
  - [Public API design](#public-api-design): [Design goals](#design-goals) ·
    [Common workflows the API must keep easy](#common-workflows-the-api-must-keep-easy) ·
    [Adding to the public API](#adding-to-the-public-api) · [Naming](#naming)
  - [Code style and documentation](#code-style-and-documentation): [Source-code compactness](#source-code-compactness) ·
    [Documentation and comments](#documentation-and-comments) · [Style coherence](#style-coherence) ·
    [Coordinate conventions](#coordinate-conventions)
  - [Dependencies and extensions](#dependencies-and-extensions)
  - [Performance](#performance): [Threading model](#threading-model) ·
    [Allocations and benchmarks](#allocations-and-benchmarks)
  - [Testing](#testing): [Testing rules](#testing-rules) · [Required coverage](#required-coverage) ·
    [Example smoke suite](#example-smoke-suite) · [Reporting numerical results](#reporting-numerical-results)
  - [Known gaps](#known-gaps)
  - [Submitting changes](#submitting-changes): [Workflow](#workflow) ·
    [Pull-request expectations](#pull-request-expectations) · [Definition of done](#definition-of-done)
  - [Anti-patterns](#anti-patterns)
  - [Questions, ideas, contributions](#questions-ideas-contributions)

## Project goals

Unfitted.jl is a compact, high-performance, science-grade Julia implementation of unfitted multi-level hp refinement on
axis-aligned Cartesian grids. It implements the method described in

> J. N. Schmäke and M. Ruess, *Unfitted Multi-Level hp Refinement for Localized and Moving Solution Features*,
> arXiv:2604.25797. <https://arxiv.org/abs/2604.25797>

Goals, in priority order:

  1. Correctness and reproducibility of the numerical method.
  2. Dimension-independent core algorithms for arbitrary spatial dimension `D ≥ 1`, not only 1D and 2D.
  3. Compact, comprehensible code with minimal abstractions.
  4. Efficient shared-memory execution on CPUs.
  5. Pure Julia project code; focused Julia packages where they remove real work.
  6. A basis-family design with integrated Legendre basis functions as the default, while keeping it straightforward to
     add alternatives such as B-splines.
  7. A compact, powerful, efficient public API for common modeling, refinement, assembly, solving, projection, and
     evaluation workflows without hiding the numerical method.
  8. Clear tests and examples that make scientific regressions obvious.

The implementation is written from the method specification, not ported from any single reference codebase. Sub-systems
informed by upstream open-source projects are attributed in `NOTICE.md` and at the top of every file that carries the
borrowed design; the notable case, the cut-cell quadrature, is described under
[Reference material](#reference-material).

## Getting started

### Prerequisites

  - Julia 1.10 or newer, with `julia` on your `PATH`.
  - `JuliaFormatter.jl` in your default Julia environment, for the pre-commit script, which runs without `--project`.
    Install it once with `julia -e 'import Pkg; Pkg.add("JuliaFormatter")'`.

The repository tracks `Project.toml` files but no `Manifest.toml`, which is git-ignored repository-wide, so contributors
resolve dependencies against their own Julia version.

### Setting up the development environment

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

### Verifying your setup

Run the test suite and the formatter / code-statistics script:

```bash
julia --project=. -e 'using Pkg; Pkg.test("Unfitted"; julia_args=["-O0"])'
julia precommit.jl --check
```

Both must exit zero on a clean checkout. If they do not, open an issue with the failing output before opening a PR.

  - **`-O0` is the standard setting, not a shortcut.** The suite is compile-bound — a timing run charged 265.9 s of a
    270.6 s invocation to the compiler — so switching the optimiser off roughly halves the time to a verdict while every
    assertion still runs, and still passes. CI keeps exactly one leg at Julia's default optimisation level, so the
    inlining and specialisation users get are exercised too. `Pkg.test` forces `--check-bounds=yes` on every leg, so
    what that leg varies is the optimiser, not production code generation in the strict sense.
  - **Threads.** Tests must pass under `JULIA_NUM_THREADS=1` *and* under more than one thread; do not rely on a fixed
    thread count.

## Repository layout

```text
Project.toml                  package metadata + production dependencies
LICENSE.md                    MIT (Unfitted.jl)
NOTICE.md                     third-party attributions (QuESo BSD-4-Clause) + clean-room algorithm citations
README.md                     public landing page
CLAUDE.md                     agent-instruction pointer document
CONTRIBUTING.md               this file — contributor + agent guide
CITATION.cff                  machine-readable citation metadata
.JuliaFormatter.toml          repo-wide formatting config (yas style, indent 4, margin 120)
.git-blame-ignore-revs        formatting-only commits for git blame
precommit.jl                  formatter wrapper + line-length check + SLOC/comment/docstring stats
src/                          library source
ext/                          package extensions (BasicBSpline, FileIO+MeshIO+GeometryBasics, Makie, Tensors, WriteVTK)
test/                         unit and regression tests
examples/                     runnable scripts in three tiers, one directory and Project.toml per example
benchmarks/                   targeted performance benchmarks (own Project.toml)
resources/                    logo sources (SVG, TikZ) used by README.md
.github/workflows/ci.yml      CI: Julia 1.10/latest × 1/4 threads (three legs at -O0, one at default -O2), format check
```

### Source files (`src/`)

Each file owns one well-defined concern. Cross-file dependencies are limited to the imports made obvious by
`src/Unfitted.jl`'s include order, which the table follows:

| File | Responsibility |
|---|---|
| `Unfitted.jl` | top-level module: imports, exports, include order |
| `geometry.jl` | `D`-dimensional axis-aligned boxes (`AxisBox`), coordinate maps, tolerances (`GeometryTolerance`) |
| `physical.jl` | the CSG level-set tree, `PhysicalDomain`, the three-valued cell classifier, the tree's affine slice on a facet (`_restrict_domain`) |
| `implicit.jl` | Saye's dimension-reduction implicit quadrature on the level set (volume rules) |
| `basis.jl` | the basis-family interface and capability traits; integrated Legendre, the default; the `:tensor` / `:trunk` index sets and the `_index_admissible` filter the minimum rule is expressed over; tensor-product evaluation; tensor Gauss–Legendre rules; boundary-basis identification |
| `fcm.jl` | cut-cell quadrature in two stages — the cut rule (`implicit_rule`, a wrapper over `implicit.jl`'s kernel) and its compression to a subset of its points reproducing its moments, exactly (`Caratheodory`) or to a residual (`NonnegativeFit`) — run by `cut_cell_rule` |
| `mesh.jl` | mesh levels and the superposition `Space`; activation masks (`LevelMask`, an alias of `BitArray`) and per-cell orders (`CellOrders` / `CellModes`); the `space` / `overlay` / `adapt` / `elevate` verbs; the fictitious-cell fold into the level masks; the level-keyed evaluators `basis_values` / `physical_basis_gradients` |
| `intersections.jl` | the integration-region record (`Region`, `Parent`), admissible regions for non-matching meshes, and the region-quadrature dispatch |
| `coverage.jl` | per-level covered-cell masks (`Coverage`, `build_coverage`) behind covered-mode pruning, and the material oracle (`_Fold`) the dof layer asks whether an inactive cell holds material |
| `ladder.jl` | nested refinement ladders: `ladder`, the depth-map form of `adapt` and its `depth_masks` grading, `overlapping_cells`, `is_nested` |
| `dofs.jl` | dof keys; the per-space dof topology (raw enumeration, elimination ledger, pivot table); per-field numbering and accessors; integrated Legendre's dof hooks (artificial boundary, leaf-semantics pruning, the minimum rule) |
| `dirichlet.jl` | the boundary-selector and Dirichlet-condition types, per-level physical-Dirichlet detection, codim-`K` facet regions and quadrature, the L² boundary projection of nonzero data |
| `surface.jl` | immersed-boundary surface meshes (`BoundaryMesh`) and surface-region integration |
| `problems.jl` | `Field`, weak-form channels and blocks (`BlockForm`, `LoadForm`, `WeakForm`), `Problem` |
| `coupling.jl` | multi-domain interface coupling: the `Interface` `on=` tag, two-sided interface regions, `InterfaceForm`, the four-block `couple` jump expansion |
| `model.jl` | the `Model` lifecycle — `prepare`, `move!` / `moved`, `activate!` / `deactivate!`, the non-mutating `adapted` / `elevated`, model-level `active_cells` / `cell_orders`, the discretisation pin, `dof_layout`, `update_dirichlet!` — and the `AssemblyDiagnostics` record with its conditioning estimates |
| `assembly.jl` | coupled Galerkin assembly: channel calculus, the cached symbolic-scatter (Gustavson) pattern, the serial scatter and the two-phase compute→gather threaded path |
| `solvers.jl` | `Solution` / `SolverDiagnostics`, the default direct sparse solve, the `linear_solver` hook for external Krylov or preconditioned solvers |
| `projection.jl` | state transfer between models: variational (`L2Projection`) and coefficient rewiring (`Rewire`) |
| `adaptivity.jl` | automated hp adaptivity: `estimate` (Bank–Weiser), Dörfler marking, the h-versus-p decision by Melenk–Wohlmuth predicted error reduction, `refine`, `coarsen` |
| `data.jl` | per-quadrature-point `QuadField` and its RBF transfer (`RBFP0`) |
| `postprocessing.jl` | what every post-processing consumer shares: the consumer-neutral sampling records (data grids, level meshes, the quadrature cloud), field and gradient evaluation, L² error, boundary integrals |
| `api.jl` | thin public-API wrappers: the canonical weak forms (`mass_form`, `stiffness_form`, `source_form`), the block, load and problem builders (`mass`, `stiffness`, `poisson`, `neumann`, …), the `boundary` and `dirichlet` constructors, and `_datum_at`, the one reader of coefficient data |
| `stubs.jl` | the package extensions' public entry points, declared and documented in the core (see [Dependencies and extensions](#dependencies-and-extensions)); included last |

Only split a single file when a second basis family, an alternative transfer scheme, or an honest test burden makes the
split helpful. Resist the temptation to add framework layers prophylactically.

### Package extensions (`ext/`)

Each extension loads when its trigger packages are loaded. Its public names are declared and documented in
`src/stubs.jl`, so `?name` finds them and a call without the extension names the package to load; the mechanism is under
[Dependencies and extensions](#dependencies-and-extensions).

| Extension | Triggers | Provides |
|---|---|---|
| `UnfittedBasicBSplineExt` | `BasicBSpline` | the open-knot tensor B-spline family behind `bspline()` |
| `UnfittedMeshIOExt` | `FileIO`, `GeometryBasics`, `MeshIO` | signed-distance leaves from a closed `BoundaryMesh` — a 2D segment loop or a 3D triangle surface (`mesh_levelset`) — or an STL file (`stl_levelset`) |
| `UnfittedTensorsExt` | `Tensors` | the tensor-notation readers of weak-form callbacks: `Vec(::TrialChannels)`, `symmetric_gradient`, `value_vec`, `gradient_tensor`, and two `TestChannels` constructors |
| `UnfittedWriteVTKExt` | `WriteVTK` | the VTK consumer of `postprocessing.jl`'s records: `write_vtk`, `vtk_series` (a `VTKSeries`), `write_quadrature_vtm` |
| `UnfittedMakieExt` | `Makie`, through any of its backends | the plotting consumer of the same records: the recipes `fieldplot`, `levelplot` and `quadratureplot`, for 1D and 2D models |

### Tests (`test/`)

`runtests.jl` includes one `test_*.jl` per area of responsibility, written against the public API where possible (the
rules are under [Testing](#testing)). Two scripts sit beside them: `run_examples_child.jl`, the subprocess the example
smoke suite runs its cases in, and `characterize.jl`, a review instrument that is deliberately not included — it asserts
nothing and prints a diffable golden-value report (`julia --project=. test/characterize.jl > before.txt`, apply a fix,
rerun, diff).

### Examples (`examples/`)

Self-contained runnable scripts in three tiers. The tier is part of the path, and choosing it is the first decision when
adding an example:

| Tier | Purpose | What belongs there |
|---|---|---|
| `tutorials/` | Teach the API | Numbered lessons, read in order, each building on the last. |
| `applications/` | Show the package on a recognisable problem | A problem a reader knows from a textbook or from practice, solved plainly, with no cleverness that obscures the API — and the package working alongside third-party Julia packages (`Tensors.jl`, `FileIO`/`MeshIO`, `OrdinaryDiffEq.jl`). |
| `reproductions/` | Hold the scientific record | Benchmarks whose problem, protocol and headline number are fixed and must not move. |

  - **Tutorials.** A header block states what the reader will learn and what they should already know; the comments
    teach rather than annotate; the dimension is never hardcoded unless the example is deliberately 1D (not yet met: see
    [Known gaps](#known-gaps)). Aim at roughly 60 SLOC of code — comment volume is unrestricted, and generosity is
    expected.
  - **Reproductions** are two studies of the method paper (`singular_square_2d`, one instance of the §5.2 sweep, and the
    §5.3 `conditioning_small_overlap`), a literature baseline (`laplace_unit_square_smooth`, after Wagner & Liu 2000),
    and the package's own adaptivity benchmarks, each scored against an external reference (`traveling_laser_2d` against
    an exact convolution, `adaptive_tanh_layer_2d` against deal.II 9.5.1's step-27 on a fixed lattice). The example
    smoke suite pins them, and they are not to be coarsened for speed. The one standing exception is
    `traveling_laser_2d`, which runs a fifth of its revolution in the default suite because the full study takes 6.3
    minutes at `-O0` on eight threads, more than the rest of the example suite put together; its case comment records
    the full-study numbers and what the shortened run gives up.

Every example follows the same conventions:

  - **Layout.** One sub-directory `examples/<tier>/<name>/` with a `<name>.jl` driver and a `Project.toml` that wires
    `Unfitted` in via `[sources]` as `{path = "../../.."}` — three levels up, because of the tier.
  - **Dependencies.** Example-specific packages (e.g. `StaticArrays`, `LinearAlgebra`, or a third-party package used
    only by an imported-geometry or time-integration demo) belong in the example's own `Project.toml` and must never be
    added to the package's top-level `Project.toml`. Each example sets `julia = "1.11"` in `[compat]`, because
    `[sources]` is a Julia 1.11 feature; the package itself still supports Julia 1.10.
  - **Reporting.** Every example `include`s the shared helper `examples/reporting.jl` via
    `joinpath(@__DIR__, "..", "..", "reporting.jl")` and prints its report through it — `print_run_report`, or, for the
    conditioning sweep, `print_parameter_block` followed by one CSV row per configuration — so the whole set reads the
    same way. The helper depends only on what `Unfitted.jl` exports.
  - **Running.** The first run resolves and installs the example's own dependencies, since `Manifest.toml` is
    git-ignored; output artifacts go under `examples/<tier>/<name>/output/`, which is git-ignored too:

    ```bash
    julia --project=examples/<tier>/<name> examples/<tier>/<name>/<name>.jl
    ```

  - **Public API.** Examples demonstrate the *public* API; use an internal (`_`-prefixed) name of `Unfitted` only when
    an example is specifically about those internals, which none of the current set is.
  - **Prose** is held to the standard of `src/` (see [Documentation and comments](#documentation-and-comments)): full
    sentences, unicode maths, never LaTeX, and a citation with authors, title, journal, year and DOI wherever a
    published result is involved (not yet met: see [Known gaps](#known-gaps)). An application or reproduction states the
    problem, the reference if there is one, and what the printed metric means, so that a reader can tell from the output
    alone whether the run succeeded.

### Benchmarks (`benchmarks/`)

Two kinds, sharing their own `Project.toml`: a BenchmarkTools suite (`suite.jl`, grouping `microkernels/`, `build/`,
`assembly/` and `scenarios/`), driven and documented by `benchmarks/runbenchmarks.jl`; and standalone timing scripts
(`intersections.jl`, `assembly.jl`, `transfer.jl`, `dirichlet_projection.jl`) that need nothing beyond the package. The
commands are under [Allocations and benchmarks](#allocations-and-benchmarks).

### Developer tooling

The repository root ships `precommit.jl`, a thin wrapper around `JuliaFormatter` that also prints per-file and
per-bucket SLOC / comment / docstring statistics. Three modes:

```bash
julia precommit.jl          # format every .jl file in place + check line length + print stats
julia precommit.jl --check  # verify formatting and line length; non-zero exit if either fails
julia precommit.jl --stats  # stats only; does not require JuliaFormatter installed
```

CI and pre-merge checks run `julia precommit.jl --check` and treat a non-zero exit as a hard failure.

## Reference material

The method specification lives in the UMLHP preprint cited under [Project goals](#project-goals). When working on
geometry, basis evaluation, integration regions, assembly, constraints, projection, or field evaluation, the paper is
the authoritative source for the contracts the code must honor.

When porting an algorithm from an upstream open-source project, follow this protocol:

  1. Read the upstream source carefully enough to understand the algorithm, not just the surface API.
  2. Write down the smaller domain-specific Julia design you intend to implement *before* writing code. The port should
     not be a line-by-line translation; it should be a Julia design informed by the algorithm.
  3. Preserve attribution at the top of the implementing file and in `NOTICE.md`. Include the canonical citation
     (authors, paper title, journal/volume/year, DOI, license, source URL).
  4. Never import or runtime-depend on upstream code from outside the Julia package ecosystem inside `src/`, `test/`, or
     `examples/`.

The cut-cell quadrature in `src/fcm.jl` is the notable borrowed design. A cut cell is integrated by Saye's
dimension-reduction implicit quadrature on a CSG level set (`src/implicit.jl`, `src/physical.jl`), and that rule is
compressed to one reproducing its moments — the moment-fitting idea of Müller, Kummer and Oberlack — by Carathéodory
pruning (`Caratheodory`, the default) or by the non-negative least-squares fit `NonnegativeFit`, whose structure follows
QuESo's `QuadratureTrimmedElement` (`github.com/manuelmessmer/QuESo`, BSD-4-Clause). The pipeline was first ported from
QuESo; no QuESo-derived code remains, and none is vendored. The full Müller–Kummer–Oberlack citation is in the header of
`src/fcm.jl` and the full QuESo citation in `NOTICE.md`; the header also documents the deliberate deviations from QuESo:
a CSG level-set geometry kernel instead of a B-rep triangulation, and moments from Saye's implicit quadrature (exact
where the leaves are planar, spectrally accurate on curved ones) instead of a divergence-theorem surface integral, which
drops QuESo's octree stair-step integrator, its `PointElimination` loop and its `AssembleIPs` retry.

## Method and architecture

The mathematical contracts below are more important than code style. Code that violates them is broken, regardless of
how clean it looks.

### The superposition model

The approximate solution is the sum of one contribution per mesh level, each extended by zero outside its overlay domain
`Ω^(k)`:

```text
u_h(x) = Σₖ u_h^(k)(x)
```

The base level covers the full physical domain `Ω`. Every overlay lies inside the base level's box, which it may span
entirely (every level of a `ladder` does); its active region is what refines where the solution needs more resolution.

Overlays are *not* topologically merged with their parents, and the package contains no hanging-node machinery. Instead
every overlay imposes homogeneous boundary conditions on its *artificial overlay boundary*, the part of its boundary
that does not coincide with the physical boundary `∂Ω`:

```text
Γ_o^(k) = ∂Ω^(k) \ ∂Ω
```

  - `Ω^(k)` is the overlay's *active region*: its box, less the cells a `LevelMask` switches off
    ([Selective activation](#selective-activation-per-cell-mask)), and on an immersed space
    ([Immersed boundary](#immersed-boundary-finite-cell-method)) that region intersected with `Ω`. So `Γ_o^(k)` is the
    part of the active region's boundary that meets `Ω`: a face of it lying wholly outside `Ω`, on the overlay's box or
    between two of its cells, has nothing on either side to keep continuous and carries no condition.
  - One rule decides this for every face of every family that clamps, `_on_artificial_boundary` in `src/dofs.jl`;
    maximal-continuity B-spline support selection asks the same material question of the cells across the face
    (`_cell_admissible`).
  - The constraints are imposed strongly: eliminated dofs for integrated Legendre and maximal-continuity B-splines;
    homogeneous linear constraints, resolved by the family that emits them, for B-splines below maximal continuity.
  - Artificial constraints are kept strictly separate from *physical* boundary conditions on `∂Ω`. Mixing the two is
    explicitly disallowed.

### Dimension independence

The core library is written for arbitrary spatial dimension `D ≥ 1`. Geometry, meshes, element lookup, basis evaluation,
integration regions, assembly, constraints, projection and field evaluation are all parameterized by `D`, and no
separate 1D / 2D / 3D code path shadows a generic implementation. The one dimensional limit in the core is
immersed-surface integration: a `BoundaryMesh` carries points, segments and triangles, so surface and interface terms
(an immersed Nitsche condition, a `couple` interface) exist for `D ≤ 3`.

  - Prefer `NTuple{D,T}`, `SVector{D,T}`, `CartesianIndex{D}`, `CartesianIndices`, and per-axis tuples or vectors over
    named `x`, `y`, `z` fields.
  - A `D`-dimensional cell is a tensor-product box: an interval in 1D, a quadrilateral in 2D, a hexahedron in 3D, a
    `D`-cube in general. Do not encode the assumption that cells are only intervals, quads or hexes.
  - Express tensor-product loops through multi-indices, recursion, generated functions or `CartesianIndices`, whichever
    is simplest and type-stable.
  - Dimension-specific examples, plotting and visualization are permitted. Core algorithms must not special-case 1D, 2D
    or 3D unless the branch is explicitly justified, tested, and still leaves the generic `D` path intact.

High dimensions are expensive, because tensor-product quadrature grows rapidly with `D`; correctness and generic
behavior still matter.

### Basis family model

Basis families are a first-party concept of the codebase. The default is a hierarchical integrated Legendre basis on
tensor-product Cartesian cells. Its assumptions live exclusively in the basis and dof-layout layer; the rest of the code
queries the basis through a small interface. A basis family must provide, directly or through a compact trait-like
interface:

  - the number and identity of its local functions on a cell or span, through `local_basis_indices(F, order)` in one
    form: the family value carries the basis mode. The `mode` keyword of `space` / `overlay` / `ladder` is folded into
    the family once, when a level is built, by `_with_mode(F, mode)` (the `::BasisFamily` default serves `:tensor`
    alone; integrated Legendre adds `:trunk`), so no index-set question takes a mode argument, and `_mode(F)` reads it
    back;
  - `instantiate_basis(F, mesh, order)` when the concrete family depends on the level's mesh (the B-spline knot
    vectors), run on every build and rebuild of a level;
  - the active basis functions covering an element or integration box;
  - values and physical / reference gradients at quadrature points, through the one per-family `_fill_factor_tables!`
    hook that the hot path and the level-keyed evaluators `basis_values(level, cell, ξ)` /
    `physical_basis_gradients(level, cell, box, ξ)` share;
  - a tensor-product ordering convention;
  - support information sufficient for overlap queries and assembly;
  - boundary or entity association information for physical Dirichlet conditions and artificial overlay constraints;
  - quadrature-order recommendations where exactness depends on the family and the polynomial degree;
  - the dof-layer hooks `_tensor_dof_key`, `_key_on_level_side`, `_overlay_constraints!` (whose default eliminates each
    raw the per-raw `_has_overlay_constraint` predicate accepts) and, if the family records linear-constraint rows,
    `_resolve_rows!` (see
    [Degree-of-freedom enumeration and constraints](#degree-of-freedom-enumeration-and-constraints)) — these read a
    typed per-level context and write into the elimination ledger — and, for leaf semantics, the burial predicates of
    the one pruning driver `_coverage_constraints!` (`_sheds_any`, `_burial_cells`, `_reproduces`, …).

The complete list, with its defaults — including `basis_name`, `is_boundary_basis`, and the per-cell-order hooks
`_supports_cell_order`, `_cell_modes` and `_index_admissible` — is the `BasisFamily` docstring in `src/basis.jl`.

The integrated Legendre implementation builds `D`-dimensional tensor-product bases from 1D hierarchical factors: modes 0
and 1 are linear "endpoint" shape functions, and modes `m ≥ 2` are "bubble" functions that vanish at the cell
boundaries.

Adding a B-spline (or any other) basis should not require rewriting geometry, intersections, assembly, projection or
solvers; it should primarily mean adding a basis-family implementation, its dof and constraint behavior, and the
corresponding tests. Do not assume that basis functions are element-local, nodal, interpolatory at boundaries, or
associated with mesh vertices: those are properties of *some* families, not of the framework.

### Mesh and refinement model

Meshes are axis-aligned Cartesian tensor-product meshes. Each level carries its own mesh, basis family, per-cell order
field ([Per-cell polynomial order](#per-cell-polynomial-order)) and activation mask — a `LevelMask`, a `BitArray` over
the level's cells, or `nothing` for "every cell active" ([Selective activation](#selective-activation-per-cell-mask));
the dof layout numbers each level's functions as one contiguous range. Overlay levels are positioned independently of
lower levels: their boundaries need not coincide with lower-level element boundaries, and they are never topologically
merged with their parents.

**Leaf semantics.** The package does carry a pruning rule for redundancy, on by default and with no keyword on `space`,
`overlay` or `ladder`:

  - Every high-order mode whose entire incidence stencil is covered by a finer level is eliminated, leaving the linear
    skeleton. Elimination is per mode, not per cell: a mode is shed only when *every* cell it is incident to is covered,
    so an edge or face mode straddling the boundary of the covered region survives even though its own cell is covered.
  - A buried linear mode that a *nested* finer level reproduces exactly is deduplicated.

**Coverage** is mask- and material-aware, and this is the one place the rule is stated:

  - An inactive cell blocks coverage when it holds material (a user-masked cell reaching into Ω); one that holds none —
    every folded cell, and a user-masked cell lying entirely outside Ω — does not.
  - On an immersed space a cell needs only its share of Ω covered: a cut cell whose material lies inside a finer level's
    box is covered even where its box sticks out past that level's face, because nothing is integrated over the part
    that sticks out. Asking for the whole box would leave the cell's high-order modes active next to the finer level's
    copies of them on Ω, each an exact null mode of the operator.
  - Both material questions — the part of a cut cell outside its cover, and whether a face of a level's active region,
    on its box or between two of its cells, meets Ω — are answered by the cell classifier, whose verdicts are taken on
    closed sets. A face on which `∂Ω` lies exactly therefore counts as holding material: it stays clamped, the cell
    behind it stays uncovered, and the exact null modes these rules remove elsewhere remain (7 and 20 at p = 2 and 3 on
    Ω = {x₁ ≤ ω} under an overlay ending at x₁ = ω; none with ω moved 1e-3 off the face). This is a documented
    limitation rather than a defect to repair: an immersed boundary not built from planes on the grid essentially never
    lies on a cell face. If an overlay's face must sit on such a plane, move it a fraction of a cell into the fictitious
    side.

**Per family.** Covered-mode pruning is a basis-family-specific feature run by one generic driver,
`_coverage_constraints!` in `src/dofs.jl` (the covering rule is `src/coverage.jl`), in which a family takes part through
its burial hooks:

| Family | What leaf semantics eliminate |
|---|---|
| Integrated Legendre | both: covered high-order modes, and buried linear modes a nested level reproduces |
| B-splines (`ext/UnfittedBasicBSplineExt.jl`) | the dedup only — the family has no bubble / skeleton split, so its one reducible mode is the one a covering level reproduces exactly; eliminating it is not an accuracy trade but what keeps a nested B-spline stack non-singular, since the two copies are linearly dependent |
| any other family | nothing, unless it opts in (`_sheds_any`, `_reproduces`) |

Under `continuity = :maximal` the B-spline burial test is *exact* rather than conservative: it walks the subdivision and
asks whether every fine function the buried one is a combination of survived the cover's own support selection, which is
necessary as well as sufficient because the subdivision coefficients are strictly positive and the representation is
unique.

`prepare(problem; prune = false)` is the public way to build a space's unreduced twin. It is a diagnostic — exactly
singular on a nested stack — rather than a discretisation to solve with. `diagnostics(...).reduced_mode_counts` reports
the reduced-mode count per level, concatenated field by field (one entry per level of each field, in declaration order).

### Selective activation (per-cell mask)

Individual cells of a level can be marked active or inactive; this is how transient analyses light up overlay cells only
where small-scale features evolve. All masking lives at cell granularity: sub-cell activation is out of scope and would
require a different abstraction.

| | Calls |
|---|---|
| Construction | `space(...; active=…)`, `overlay(...; active=…)`: `nothing` (default, every cell active), an `AbstractArray{Bool,D}` matching the mesh's cell grid, an iterable of `CartesianIndex{D}` listing active cells, or a predicate `(cell_box, cell_index) -> Bool` |
| Mutation | `activate!(model; level, cells)`, `deactivate!(model; level, cells)` |
| Query | `active_cells(V::Space; level)`, a copy of the level's `BitArray` (all-true on an unmasked level); `active_cells(model; level, effective = true)`, the same on a prepared model |

  - **The mutators follow the `move!` invalidation contract**: they bump `model.version`, clear `model.matrix` and
    `model.rhs`, replace the assembly cache (`model.assembly`), drop the Dirichlet projection cache, rebuild the
    integration plan, the dof layout and the region cache (`model.regions`), and refresh the diagnostics. An outstanding
    `Solution` raises on stale reuse. Like `move!`, they address a level by position in the model's single space and
    therefore raise `ArgumentError` on a multi-domain (coupled) model.
  - **The pre-fold mask is the one a caller writes to.** On a space carrying a `PhysicalDomain` each level has two
    masks: the caller's, on `model.prefold_space`, and the *effective* one the fictitious fold wrote over it. The
    mutators, `adapt`, `adapted` and `elevated` all derive from the pre-fold space, because a rebuild must re-decide the
    fold rather than inherit it. The model query defaults to `effective = true` and therefore does **not** round-trip:
    feeding its result back to `adapt` launders the fold into user intent, which the next fold cannot undo (see the
    `active_cells` docstring). Read back with `effective = false` when the mask is going to be written again.
  - **Dof layer.** A face between an active and an inactive cell of the same level is a face of the active region, so
    the rule of [The superposition model](#the-superposition-model) decides it: the face is clamped like a mesh-box face
    where it holds material itself, and left free where it holds none — next to a void cell, or through a hole between
    two cut cells (`_on_artificial_boundary`, asked per raw by `_has_overlay_constraint`). Span-mode dofs of inactive
    cells are not enumerated.

### Per-cell polynomial order

The polynomial order of a level may vary from cell to cell, so p-refinement is contained inside one mesh instead of
needing a co-located overlay per increment. It is a basis-family capability: integrated Legendre declares it through
`_supports_cell_order`, and every other family (B-splines included) refuses a non-uniform field loudly rather than
ignoring it, because a per-cell degree on a shared knot vector names no set of functions.

A family that refuses is not thereby stuck at one degree: p-refinement is then done the multi-level way, by superposing
a level of higher degree over the region that needs it. For maximal-continuity splines that superposition is linearly
independent **by construction** and needs no deduplication — a degree-`p`, `C^(p−1)` function supported inside the
region would have to be `C^p` to lie in the degree-`p+1` space there, hence a polynomial, hence zero — so the two spaces
intersect trivially. The adaptive loop follows suit ([Automated hp adaptivity](#automated-hp-adaptivity)): on such a
level every marked cell takes h, the loop the isogeometric literature runs, and `estimate` enriches by superposing a
refined twin rather than by raising the degree.

| | Calls |
|---|---|
| Construction | `space(...; order=…)`, `overlay(...; order=…)`: an integer or an `NTuple{D,Int}` (uniform), an `AbstractArray{<:Integer,D}` or `AbstractArray{<:NTuple{D,Integer},D}` shaped like the level's cell grid, or a predicate `(cell_box, cell_index) -> order`. A field whose entries are all equal collapses to the uniform representation, so the per-cell path costs a uniform level nothing |
| Rebuild | `elevate(V::Space, level => order, …)`, `elevated(model, level => order, …)`, mirroring `adapt` / `adapted`. `elevate` also accepts an iterable of `CartesianIndex{D} => order` pairs, which leaves unnamed cells alone — the shape a marking loop produces |
| Query | `cell_orders(V; level)`, `cell_orders(model; level)`: the per-axis order of every cell as a fresh array, all-equal on a uniform level |

`elevate` is a separate verb from `adapt` because a mask spec and an order spec collide on `nothing` and on a predicate,
and neither collision is detectable at dispatch. Compose them: `elevate(adapt(V, h), p)`.

  - **The minimum rule.** Where two cells of different order share a face, the shared entity carries the *minimum* of
    the two orders (the minimum rule of Szabó & Babuška 1991 and Demkowicz 2006; full citations in `src/basis.jl` and
    `src/dofs.jl`). Without it a mode generated only by the high-order side is nonzero on the shared face and the space
    is not C⁰ — and it still assembles, still solves, and still returns a plausible residual. The rule is a predicate on
    the dof *key*: the key's own multi-index must belong to the family's index set at the order of every **active** cell
    incident to it. That intersection is the classical componentwise-minimum statement, because the family's index-set
    filter `_index_admissible` is monotone non-decreasing in the order, and it is the better spelling because the
    minimum is then never formed and no question is ever asked at an order no cell carries. Three details are
    load-bearing:
      - it is keyed on the key, so both sides agree by construction;
      - it is a set membership rather than a per-axis mode comparison (the two differ in 3D under `:trunk`);
      - inactive incident cells are skipped rather than minimised over (a fictitious fold otherwise loses live dofs).
  - **One order field.** `Level.orders` is a `CellOrders` — a palette of the distinct per-axis orders plus, on a graded
    level, one `UInt16` palette index per cell — and `Level.modes` the minimum-rule table `CellModes`, which `Level`'s
    inner constructor derives from the orders, the mesh, the family (which carries the basis mode) and the mask, and
    which a caller never supplies. A uniform level stores neither per-cell array (`class = nothing`, `kind = nothing`),
    because both would be constants — 4 bytes on every cell (`Base.summarysize` of an 8² ladder of depth 7: 0.15 bytes
    per cell without them, most of it the mask's bit, against 4.15 with them; about 90 MB across the 22.4 M cells of a
    depth-9 stack), and the discretisation pin would digest the class array at 8 hash rounds per cell, where it digests
    an absent one as `-1`. The record type is still one: the uniform case is a `nothing` branch in `_order_at` (behind
    `cell_order`), in `_active_set` (behind `cell_basis_indices` and the assembly hot path) and in `cell_orders`.
    `cell_order` and `cell_basis_indices` check the cell against the grid themselves, since a uniform level has no array
    whose indexing would; `_active_set` is the unchecked hot-path read, and reads no mask because every region parent is
    an active cell by construction (`_signature`).
  - **`cell_order` versus `nominal_order`.** `cell_order(level, cell)` is the question anything that integrates,
    evaluates or subdivides must ask, and every quadrature consumer does: the region rule, the moment order, the facet
    rule, the immersed simplex rule and the post-processing sample lattice all size from the parent *cell*.
    `nominal_order(level)`, the per-axis maximum over the palette, is a sizing and default quantity: the workspace's 1D
    factor buffers, where over-sizing is safe (its value and gradient buffers are sized from the level's longest
    minimum-rule set instead), and the order a new `overlay` or a re-instantiated basis family inherits.
  - **Order ≥ 1 per cell** is enforced per palette entry at construction, and `mode = :trunk` requires isotropy per
    cell; `_build_cell_orders` checks both through `local_basis_indices`. The first cannot be expressed by the minimum
    rule: the dof key collapses the two endpoint modes into one node factor, so no filter over keys can see a missing
    endpoint mode. The second holds because the `:trunk` filter compares against one scalar `p`.
  - **The acceptance test is the per-basis-function one-sided trace jump**, asserted `== 0` (`test/test_cell_order.jl`).
    An L² projection convergence rate does *not* discriminate — a non-conforming space is strictly larger and its
    projection error is marginally lower — and the solution's own jump norm decays under h-refinement even when the
    space is broken.

### Degree-of-freedom enumeration and constraints

Physical boundary conditions and artificial constraints are assigned before the final global dof enumeration, and
constrained and active dofs are kept explicitly distinguishable. A layout is built in four steps:

  1. **Topology, once per space** — raw dofs enumerated level by level from the family's `TensorDofKey`s (each level's
     raws a contiguous range), and the artificial eliminations: the artificial overlay boundary and, under leaf
     semantics, covered-mode pruning. It depends on the space alone, so every field over one space shares it, and it is
     never mutated.
  2. **Physical Dirichlet detection, per field** — from the family's boundary association (`_key_on_level_side`) and
     each selector's facets.
  3. **Numbering, per field** — component-major active ids over the dofs neither step eliminated.
  4. **Projection, per field** — nonzero Dirichlet data are projected onto the boundary trace space and applied through
     consistent right-hand-side updates when columns are eliminated.

A basis family records its artificial constraints in an `EliminationLedger`:

  - `_eliminate!(ledger, raw, source)` — a dof-wise strong elimination; the first source to eliminate a raw is the one
    it is credited to;
  - `_constrain!(ledger, raws, coefficients, source)` — a homogeneous linear constraint `Σᵢ cᵢ u_{rᵢ} = 0` among one
    level's raws.

The core stores a level's rows but does not resolve them. A family that records rows implements `_resolve_rows!`, which
receives them in emission order after the level's hooks have run and records each raw it eliminates with `_pivot!` —
strongly, or as a depth-one pivot `u_r = Σᵢ wᵢ u_{rᵢ}` over raws that stay free; the core default raises. B-splines
below maximal continuity are the only such family today, so the cascade resolver lives in
`ext/UnfittedBasicBSplineExt.jl` and is tested in `test/test_basis_bspline.jl`. Pivots are stored once per topology, and
assembly (`_slots!`, `_condense!`) and `dof_value` read them through one rule, `_fold_pivot`.

Tests must check that overlay contributions vanish on artificial overlay boundaries and that physical boundary
conditions are not accidentally applied to artificial overlay boundaries.

### Integration regions

For non-matching axis-aligned meshes, admissible integration boxes are constructed as follows:

  1. For each coordinate direction, collect the element-boundary coordinates of the participating meshes that bound an
     *active* cell of their own level — all of them on an unmasked level. A coordinate no level keeps separates two
     slabs with the same coverage signature, which step 5 would merge again, so dropping it costs neither a region nor
     any exactness, and the candidate grid follows the live cells rather than the declared resolution.
  2. Sort, and merge coordinates closer than the geometry tolerance.
  3. Form the non-degenerate intervals between adjacent coordinates.
  4. Take the Cartesian product of the intervals as candidate boxes.
  5. Compute each candidate's per-level *coverage signature* at its midpoint — which cell of each level contains it, or
     `0` when the level's mesh does not reach the point or that cell is inactive (activity is a per-cell property
     carried by `Level.mask`, never a property of the level) — and greedily merge axis-adjacent candidates that share a
     signature. Every cell inside a merged box still lies inside a single cell of every covering level, so the `C^∞`
     contract below holds while the region count stays proportional to the number of genuinely distinct coverage
     patterns. A candidate no level covers (an all-zero signature) seeds no box: it is no region of any consumer.
  6. Decode each merged box's covering parent cells from its signature and keep the box only if it satisfies the
     caller's `criterion`, expressed over those parent *levels*: `:any_parent` (the default — at least one level covers
     the box; used by assembly, field evaluation and visualization), `:all_levels` (every level of the space covers it),
     or a caller-supplied predicate `parents -> Bool`. Any other value raises `ArgumentError`. No package code uses the
     predicate form (only the tests do): `projection.jl` builds its transfer regions from the partition engine
     (`_partition`) directly and applies its own target-coverage test rather than going through a `criterion`.
  7. Dispatch the region quadrature on the `PhysicalDomain` classification: tensor Gauss on `:full`, α-scaled tensor
     Gauss on `:fictitious` with `α > 0`, and on `:cut` the domain's cut rule under its compression (`cut_cell_rule`;
     see [Region quadrature kinds](#region-quadrature-kinds)). A `:fictitious` region under strict α (`α = 0`) is
     dropped outright.
  8. Store each contributing parent element with the affine map from the box's frame into that element's reference
     frame, record a `SmallOverlap` for every region below `tolerance.small_volume`
     ([Small overlaps and conditioning](#small-overlaps-and-conditioning)), and track the largest moment residual of a
     compressed cut rule seen.

The integrand of a basis-function product is smooth (`C^∞`) inside such an admissible box. Do not integrate a product of
basis functions across an element-boundary kink without partitioning the region first.

Every region the assembly kernel walks — a volume box, a piece of a boundary facet, a sub-cell of an immersed
`BoundaryMesh` or of a coupling interface — is one record, `Region`. Its corners `lower` / `upper` are both its extent
and its frame (degenerate on an axis it does not extend along: a facet's pinned axes, a flat axis of a simplex), its
rule lives in that frame, and each `Parent` carries the map `ξ = ξ0 + dξ ∘ η` from the frame into its cell
([Coordinate conventions](#coordinate-conventions)). Boundary facets are partitioned by the same engine restricted to
the facet's free axes, so the steps above apply to them one dimension down.

### Coupling and assembly

Assembly produces *one* coupled variational problem over all active spaces: every overlapping active basis-function pair
across all relevant levels is included. Level-wise uncoupled subproblems are never an acceptable substitute for the
coupled Galerkin system.

  - Symmetric bilinear forms are preserved to roundoff. For SPD model problems the constrained global matrix is SPD
    unless a known dependency or null space is present.
  - Assembly queries the basis / dof layer for active local functions and local-to-global maps. It must not assume a
    fixed formula such as `(p + 1)^D` outside integrated Legendre basis code.

### Immersed boundary (finite cell method)

The package supports the finite cell method with non-negative, compressed quadrature on cut cells. The geometry of `Ω`
is a **CSG level set**: a Boolean combination of smooth leaves, each a scalar function `f` with
`Ω_leaf = { x : f(x) ≤ 0 }`. A single leaf is the familiar `Ω = { x : φ(x) ≤ 0 }`; the combinators build the rest (an
annulus is a disk minus a disk). Carrying the leaves separately — rather than collapsing them into one `min`/`max` level
set — keeps every boundary piece smooth, so the implicit quadrature kernel stays high-order across creases and corners.
Boolean indicator representations are intentionally not supported.

  - **Build geometry** with `leaf(f; lipschitz=Inf)`, `intersect(a, b…)`, `union(a, b…)`, `setdiff(a, b)` and
    `complement(a)`. A bare callable is auto-wrapped as a default leaf, so the single-level-set case stays ergonomic.
    `lipschitz` is the leaf's Lipschitz constant: `1.0` for a true signed distance; `Inf` disables the cheap
    uniform-sign certificate and forces classification by corner sampling and subdivision. The `leaf` docstring states
    what a leaf must accept (`ForwardDiff.Dual` arguments, calls from several threads).
  - **Imported geometry**: with `FileIO` and `MeshIO` loaded, `mesh_levelset(mesh)` turns a closed `BoundaryMesh` (a 2D
    segment loop or a 3D triangle surface) into a signed-distance leaf, and `stl_levelset("part.stl")` reads an STL into
    one. Both compose with the CSG combinators (`ext/UnfittedMeshIOExt.jl`).
  - **Construct and attach** the domain:

    ```julia
    physical_domain(geometry; lipschitz=Inf, alpha=0.0, keep_fictitious=false,
                    subcell_length_scale, max_depth=8, moment_order_factor=2,
                    cut_rule=implicit_rule, compression=Caratheodory())
    space(omega; …, physical=physical_domain(…))
    ```

    The default `physical=nothing` keeps the no-FCM hot path. Every keyword is documented on the `physical_domain` and
    `PhysicalDomain` docstrings in `src/physical.jl`; three are easy to misread:

    - `alpha = 0` is the strict cut path and `alpha > 0` enables α-FCM (see
      [Region quadrature kinds](#region-quadrature-kinds)). Fully fictitious *cells* are dropped from the dof layout
      unless `keep_fictitious = true`, which requires `alpha > 0` — with `alpha = 0` those cells would carry no
      quadrature, so the pairing is an `ArgumentError`. α alone never keeps them, and the α-scaled cut-cell enrichment
      applies whether or not they are kept.
    - `subcell_length_scale` (required) is a geometry-robustness knob — the subdivision budget the cell classifier and
      the implicit kernel share, for thin or near-tangent features — and **not** a moment-accuracy grid: a cut cell with
      a graph-like axis clear of a fold gets the kernel's full order at once and is never subdivided. Being a physical
      length, one domain serves every level of a `Space`. `max_depth` caps both subdivisions.
    - `moment_order_factor = 2` spans degree `2p`, so trial × test products integrate exactly, matching on cut regions
      what tensor Gauss does on `:full` ones. `1` spans degree `p` with only `∏(p+1)` points — the uncut Gauss count and
      QuESo's convention — at roughly the cube less compression time, but a stiffness form is then exact only up to
      `p = 2`: exactness needs `factor ≥ 2 − 2/p`. `PhysicalDomain`'s docstring carries the measured order-1-to-7
      comparison.
  - **Two stages make a cut cell's rule**, and `cut_cell_rule` runs them. The *cut rule*,
    `(physical, box, moment_order) -> (points, weights)`, integrates `box ∩ Ω` accurately with as many points as that
    takes; `implicit_rule` (default) is Saye's implicit volume rule at a Gauss count matched to the moment order. The
    *compression*, `(points, weights, moments, basis) -> (points, weights)`, replaces it by a rule reproducing its
    moments over the tensor Legendre basis `basis::MomentBasis` of the box, in intent with at most `length(basis)`
    positive weights: `Caratheodory()` (default) prunes the cut rule to a subset of its points, exact by construction;
    `NonnegativeFit()` fits non-negative weights on candidates drawn from it; `nothing` keeps the cut rule uncompressed.
    `cut_cell_rule` drops the zero weights a compression leaves and does not check the count.
  - **The stage contract.** Both stages return *physical*-frame points and non-negative weights — a weight below zero
    beyond round-off is an `ArgumentError`, which reaches `prepare` wrapped in a `CompositeException` like any error
    raised on the threads of a plan build. Both are memoised per box and moment order, and both must be thread-safe like
    every leaf, since a plan's cut rules are built on every thread. Only the compression is used on boundary faces. The
    full contract is the `PhysicalDomain` docstring's section "The cut rule and its compression".
  - **Cut-rule defaults.** The moment order is `moment_order_factor × max(cell_order)` per axis over the region's parent
    cells, and `cut_cell_rule` sums the cut rule's tensor Legendre moments for the compression. Whatever the
    compression, a rule whose residual misses `_FIT_FAILURE_RESIDUAL · ‖m‖` (1e-2, relative because the residual carries
    a moment's units, so whether a cell falls back never depends on its size) is replaced by the cut rule itself rather
    than dropping the cell. Rules are cached by canonicalised region bounds and moment order. The algorithms —
    Carathéodory's recombination and pruning on a pivoted Householder QR written in Julia, `NonnegativeFit`'s
    Lawson–Hanson solve and its retries on denser candidate clouds — are documented in the `src/fcm.jl` header and the
    `Caratheodory` and `NonnegativeFit` docstrings.
  - **Diagnostics.** The FCM statistics — cut-region, fit-failure and fallback counts, the largest moment residual, the
    inactive-cell counts that fold in the fictitious drop, their facet analogues, and the Dirichlet trace fields — are
    documented field by field, with their caveats, on the `AssemblyDiagnostics` and `diagnostics` docstrings in
    `src/model.jl`. On an immersed system compare basis families by `scaled_condition_estimate`, the condition of
    `D⁻¹ A D⁻¹` that a Jacobi preconditioner sees. The unscaled number is dominated by the spread of the diagonal, and
    the scaled one is where the families differ: 43.8 against 2.2e10 at degree 3 on 16² with a disc removed, computed
    directly past the diagnostic's 256-unknown cap. A nonzero `cut_fallback_count` means those cells are under-resolved
    for their geometric complexity and should drive refinement: the fallback is a safety net, not a substitute for an
    adequate mesh.

#### Region quadrature kinds

Every region carries one rule, named by `region.kind` from this closed vocabulary:

| Kind | Region | Rule |
|---|---|---|
| `:full` | inside `Ω`, or no `PhysicalDomain` attached | tensor Gauss |
| `:fictitious_alpha` | outside `Ω`, under `α > 0` | α-scaled tensor Gauss |
| `:fictitious` | facet regions only: a face outside `Ω` under `α = 0` | an empty rule, kept where the volume plan drops the region |
| `:cut_fitted` | crossed by `∂Ω` | the compressed cut rule, or the cut rule itself under `compression = nothing` |
| `:cut_fallback` | crossed by `∂Ω`, compression missed the failure threshold | the cut rule the moments were summed from: correct and non-negative, but uncompressed — on the many-feature cells where a fallback is plausible, 10³–10⁵ points and a 50–200× blow-up in that cell's assembly work |
| `:cut_failed` | `Ω ∩ box` carries no cut rule at all, under `α = 0` | an empty rule: the region contributes zero quadrature |
| `:cut_alpha_failed` | the same, under `α > 0` | the α-scaled tensor rule alone, so the cell's dofs stay α-stabilised |
| `:surface`, `:interface` | simplex regions of an immersed `BoundaryMesh` or a coupling interface | the simplex's own rule |

Under `α > 0`, `:cut_fitted` and `:cut_fallback` carry the concatenation `(1 − α)·rule ∪ α·tensor`, rule first, from
`∫_cell α(x) f = (1 − α)∫_Ω f + α∫_cell f`. One compiled loop serves every kind: it walks the rule's points and weights
(`_region_point`).

**Grid-aligned boundary faces use the same vocabulary**, on `FacetRegion.kind`, because they run the same dispatch one
dimension down. A codim-`K` facet is the affine slice `⋂ⱼ {x_{kⱼ} = vⱼ}`, so `Ω` on it is the CSG tree with every leaf
restricted to that slice (`_restrict_domain`), and the region's own `(D − K)`-dimensional face box is classified and
integrated against it. `α` is honoured there too, so a trimmed face at `α > 0` carries the same blend a cut cell does —
and therefore quadrature points outside `Ω`, which is why a datum or integrand defined only on `Ω` needs `α = 0`. Two
differences are deliberate:

  - A zero-measure facet region is **kept**, with an empty rule, where a fictitious volume region is dropped: its parent
    cells still carry the trace dofs the dof layer constrains on the grid-aligned face test, which knows nothing about
    the level set.
  - The domain's `cut_rule` is **not** consulted on a facet: it is a rule on a cell box in the model's own dimension, so
    a facet cut region always takes `implicit_rule` on the restricted level set, under the domain's own compression.

### Automated hp adaptivity

Where to refine can be left to the solver. Three verbs do it, the caller owns the loop, and nothing in
`src/adaptivity.jl` holds state between cycles.

| Verb | Does |
|---|---|
| `estimate(model, solution; enrichment = 1)` | returns an `ErrorEstimate`, a Bank–Weiser indicator per level and cell |
| `refine(V, est; theta = 0.5, pmax = 8, previous = nothing)` | marks cells by Dörfler (`mark_cells`) and sends each marked cell to h or to p (`decide`); the loop form is their composition |
| `coarsen(V; h = (), p = (), pmin = 1)` | inverts both steps: releases an h-step by deactivating the cover it woke, a p-step by lowering the order, bounded below by `pmin` |

**Estimating.**

  - An `ErrorEstimate` carries the per-level, per-cell indicator `cells`, its `total`, a `reference` against which
    `total / reference` is a scale-free stopping quantity on one problem — not across different Dirichlet data, because
    a nonzero lift's own energy is never formed — and a `consistency` ratio, a free data-oscillation and solve check,
    *not* a saturation signal (its docstring says why).
  - The indicator is Bank–Weiser: build the order-elevated space `V⁺`, inject the solution into it, and read the
    residual in the directions `V` cannot represent through the diagonal of the enriched operator,

    ```text
    η²_K = Σ_{j ∈ W(K)} R_j² / A⁺_jj ,   R = b⁺ − A⁺ u⁺ ,   W = V⁺ ⊖ V
    ```

    `W` is a set difference on dof *keys*, not a comparison of mode indices (the two disagree at an anisotropic order),
    and every complement mode is shared equally among the cells incident to it. The form must be **coercive**, since the
    diagonal is read as a mode's energy.

    > R. E. Bank, A. Weiser, *Some a posteriori error estimators for elliptic partial differential equations*, Math.
    > Comp. **44** (1985) 283–301.
    > [doi:10.1090/S0025-5718-1985-0777265-X](https://doi.org/10.1090/S0025-5718-1985-0777265-X).

  - **How `V⁺` is built is a basis-family question**, answered in full on the `estimate` docstring. A family carrying a
    per-cell order (`_supports_cell_order`) is enriched in place by raising every cell's order. A family without one — a
    B-spline level, whose degree belongs to a whole-axis knot vector — is enriched by **superposition**: a co-located
    twin of the same family, degree and active region on a mesh refined by `enrichment + 1` per axis, which nests
    because the factor is an integer and whose indicators are charged back to the level it enriches. The twin sits above
    every level, so where it is live over a higher level that carries a per-cell order, that level sheds modes in `V⁺`
    that are free in `V`; then `V⁺ ⊉ V` and `estimate` refuses the stack. Put B-spline levels above integrated-Legendre
    ones.

**Refining.**

  - Dörfler marking takes the smallest set of cells carrying `theta` of the total squared indicator; `theta` is the
    loop's only tunable.

    > W. Dörfler, *A convergent adaptive algorithm for Poisson's equation*, SIAM J. Numer. Anal. **33** (1996)
    > 1106–1124. [doi:10.1137/0733054](https://doi.org/10.1137/0733054).

  - **The h-versus-p decision is predicted error reduction**, deliberately not a smoothness indicator:
    `src/adaptivity.jl`'s header records why, the comparison against deal.II's step-27, and why the two steps are kept
    independent rather than welded to depth. Each cycle predicts what a cell's indicator ought to become if the solution
    there is as smooth as the step just taken assumed; the next cycle compares, and a cell that met its prediction takes
    p again while one that fell short takes h. `previous = (V_last, est_last)` is the only state the rule needs — a
    p-step shows up as a raised order and an h-step as a cell that was not active before — so nothing is stored between
    cycles. Omit `previous` and every marked cell takes p, falling back to h only where p is unavailable: right for the
    first cycle, wrong thereafter. The constants γ_p² = 0.4 and γ_h² = 4 are the literature's calibration and are
    deliberately not exposed.

    > J. M. Melenk, B. I. Wohlmuth, *On residual-based a posteriori error estimation in hp-FEM*, Adv. Comput. Math.
    > **15** (2001) 311–331. [doi:10.1023/A:1014268310921](https://doi.org/10.1023/A:1014268310921).

  - **A family with wide support is owed two things.** The p-step is withheld on a level that carries no per-cell order,
    so every marked cell there takes h (or is dropped, if it has no h-step either); and the h-step **support-extends**
    the cells it wakes by the family's own `_support_radius`, because a region narrower than that radius carries no
    functions at all, and a refinement that refines nothing makes a loop that never terminates. `_support_radius` is
    zero for integrated Legendre, so neither costs the default family anything.

**Coarsening.** There is deliberately no `coarsen(V, estimate)` form: which cells to release is not settled by the
indicator, so the caller marks.

**Restrictions**, all checked rather than assumed:

  - a single-domain, single-field model (the field may carry any number of components);
  - an assembled model, and a solution of that model as it stands;
  - `enrichment ≥ 1`;
  - a coercive form, checked as far as the indicator reads it: a non-positive complement diagonal raises;
  - no form keyed by quadrature-point index, which the enriched model would read on a different cloud;
  - `V⁺ ⊇ V`, so a B-spline level under an integrated-Legendre level is refused.

On an immersed model the h-step has no state transfer; see
[Moving overlays and state transfer](#moving-overlays-and-state-transfer).

### Moving overlays and state transfer

When overlay positions change:

  - rebuild the integration plan and dof layout, reusing every cut-cell rule whose box the move left unchanged;
  - reuse geometry-dependent matrices only when the geometry and active spaces are unchanged;
  - update time-dependent load vectors independently;
  - transfer a field's coefficients by `L2Projection`, or by `Rewire` where the target basis contains the source's —
    never by pointwise interpolation (per-quadrature-point history data is the exception: `RBFP0` interpolates a
    `QuadField` across a rebuild by design);
  - test projection on constants, low-order polynomials, and fields represented exactly in both spaces.

On an immersed model — a space carrying a `PhysicalDomain` — there is no state transfer across an h-refinement or an
h-release (see [Known gaps](#known-gaps)). `L2Projection` refuses any target carrying a physical domain outright, and
`Rewire()` refuses a target whose active basis does not contain the source's, which is what leaf semantics make of a
refinement and what a release makes of a cover (`Rewire(strict = false)` would drop the buried coefficients instead of
refusing). An adaptive or transient loop on an immersed model therefore re-solves on the new space rather than carrying
its iterate across, and `estimate` needs that fresh solve before it can be called again. The restriction is specific to
the h-step: a pure **p**-step buries nothing, so `Rewire` accepts it — on the unit square minus a centred disc, an 8×8
order-2 base raised to order 3 on two adjacent leaf cells (216 → 223 unknowns) transfers with the evaluated field
unchanged to the last bit at four interior points. `test_adaptivity.jl` pins both refusals, so neither can decay into
silently dropped coefficients.

### Small overlaps and conditioning

Tiny integration boxes — regions whose box volume falls below `GeometryTolerance.small_volume` (default `√eps`, an
absolute volume) — are a known conditioning hazard for unfitted methods. The package detects and reports them in the
assembly diagnostics instead of silently dropping or rescaling them.

  - Track condition numbers or solver iteration counts in tests and benchmarks for representative small-overlap cases.
  - Do not silently drop, merge or rescale tiny overlap regions in production assembly unless the change is explicitly
    tested and documented as a numerical method choice.
  - Prefer making conditioning behavior visible before adding stabilization.

## Public API design

The public API is a first-party design concern. It is compact enough to learn quickly but powerful enough to express the
paper's workflows without forcing users to manipulate low-level dof tables, constraint masks, or intersection-region
internals. The contracts behind its feature-level verbs — activation, per-cell order, the immersed boundary and hp
adaptivity — are under [Method and architecture](#method-and-architecture).

### Design goals

  - **Short common path.** Create a Cartesian base discretization, add overlay levels, choose basis families and orders,
    impose physical boundary conditions, define sources and material coefficients, assemble, solve or step in time,
    evaluate fields, move overlays, and project state — all with a small number of clear calls.
  - **Expert escape hatches.** Users can inspect or provide quadrature rules, tolerances, basis families, solver hooks
    (`linear_solver`), overlay geometry, and the transfer backend (`L2Projection`, optionally with a cached mass matrix,
    or `Rewire`) without rewriting assembly code.
  - **Methods exposed directly.** Names and objects reflect domains, levels, overlays, basis families, active spaces,
    integration regions, constraints, forms, projections, and solutions.
  - **Keyword constructors, small typed configuration structs**, rather than long positional argument lists or untyped
    dictionaries.
  - **Sensible defaults.** Integrated Legendre basis, conservative quadrature, robust tolerances, standard sparse
    assembly, and a direct solver out of the box — documented and easy to override.
  - **No hidden global state.** API calls are deterministic from their arguments and the project configuration.
  - **Thin wrappers.** High-level wrappers compose lower-level tested kernels rather than duplicate geometry, basis,
    constraint, or assembly logic.
  - **Performance through the API.** Convenience never forces dynamic dispatch, type instability, avoidable heap
    allocation, or repeated preprocessing in hot loops.
  - **Inspectable objects.** A user can query active unknowns, level and basis metadata, integration-region counts,
    constraints, solver diagnostics, residuals, and small-overlap warnings.
  - **Actionable errors.** Invalid overlay placement, inconsistent dimensions, unsupported basis-family constraints,
    singular systems, and small-overlap conditioning warnings are told apart by error type and message (not yet met: see
    [Known gaps](#known-gaps)).

### Common workflows the API must keep easy

  - Static Poisson / Laplace-style scalar problems on a box domain.
  - 1D bar-style problems with point loads or discontinuous coefficients and sources.
  - Adding centered or explicitly positioned overlay boxes with per-level order choices, and grading the polynomial
    order from cell to cell within one level.
  - Declaring a nested refinement ladder up front and switching its cells on per level as the solution develops.
  - Selectively activating or deactivating individual overlay cells as a feature evolves in time, and growing a marked
    region by a margin — `dilate` for a width the caller chooses, `support_extension` for the one the basis family
    needs.
  - Handing the refinement decision to the solver: estimate, mark, refine, coarsen, with the loop itself left to the
    caller.
  - Attaching an immersed physical domain via a level-set function and integrating only over `Ω`.
  - Moving an overlay in one call, reusing the cut-cell rules of the regions it leaves unchanged.
  - Variational projection from an old space to a new space.
  - Evaluating a superposed solution at points or on a visualization grid.
  - Producing a minimal reproducibility report: `D`, basis family, orders, levels, active unknowns, integration regions,
    cut-region counts, residual and error metrics, and solver diagnostics.

### Adding to the public API

Keep the public API small, expressive, and stable: add a public name only when it simplifies a real user workflow or
exposes an essential method concept. Every addition must have:

  - a docstring with argument conventions and defaults;
  - at least one small test that uses the public call rather than internals;
  - a check that dimensions and basis families are handled generically where applicable;
  - an example or test showing the simplified workflow if it replaces tedious manual setup.

### Naming

Prefer concise verbs and nouns such as `space`, `overlay`, `ladder`, `adapt`, `refine`, `assemble`, `solve!`,
`transfer`, `value`, and `diagnostics` over framework-heavy terminology. The public API should read like the method, not
like a software architecture.

An exported name must not also be an export of a package an extension integrates: a session that loads both could call
neither by its bare name (not yet met: see [Known gaps](#known-gaps)).

## Code style and documentation

The code is held to a strict compactness standard; the documentation around it is intentionally generous.

### Source-code compactness

  - Prefer simple concrete structs over abstract hierarchies, and short functions with explicit data flow.
  - Use parametric types for dimension and scalar type where it improves performance or clarity (e.g. `AxisBox{D,T}`).
    Keep `D` visible in types where practical; avoid storing the dimension only as a runtime integer in hot paths.
  - Avoid designing a general-purpose FEM library. This project is specialized to axis-aligned unfitted multi-level hp
    refinement.
  - Avoid hidden global state.
  - Avoid macros unless they remove repeated boilerplate in hot kernels and are tested.
  - Avoid dynamic dispatch in quadrature, mapping, and assembly hot loops.
  - Precompute shape values, derivatives, quadrature rules, active basis lists, and local-to-global maps when reused.
  - Use tolerances intentionally. Do not compare floating-point coordinates with `==` except for canonicalized grid
    coordinates or tests designed for exact values.
  - Write one `D`-generic implementation of each core algorithm, never separate 1D, 2D, and 3D ones, and wrap it in
    dimension-specific examples if needed (see [Dimension independence](#dimension-independence)).

### Documentation and comments

Source files are the primary documentation: every non-trivial entity should be possible to understand from its file
alone. Write for a reader who knows conventional finite elements but not the advanced techniques specific to this
method.

  - **Every public symbol** — exported function, struct, abstract type, or constant — gets a docstring in full sentences
    covering arguments, return value, defaults, conventions, and any non-obvious behavior. It names the section of the
    paper or the cited algorithm when that informs the implementation.
  - **Every non-trivial internal function** gets a docstring or a multi-line `#` comment block immediately above the
    definition. "Non-trivial" means anything a new reader could not predict from the name and signature alone; tiny
    obvious helpers (`_squared_norm(::Number)`, single-line predicate wrappers) need none.
  - **Explain the *why*, not the *what*.** The code already states what it does. Comments and docstrings carry the
    reasoning: the invariant being preserved, the numerical convention being honored, the choice between approaches, the
    reason an allocation pattern matters.
  - **Inline commentary in complex function bodies.** When a body runs more than a handful of lines or chains
    non-obvious steps — basis-table updates, sparse-scatter emission, boundary projection, retry loops — put a short
    comment above each non-obvious step saying why it is there, and mark the distinct stages of a long body with
    one-line section headers. A reader should be able to follow the implementation by skimming the comments before
    reading the code.
  - **Math goes in unicode**, not LaTeX: `Σₖ u_h^(k)(x)`, `Ω = { x : φ(x) ≤ 0 }`, `‖ A x − b ‖₂`, not `\sum_k`,
    `\Omega`, `\|Ax-b\|_2`. ASCII pseudocode is the fallback when unicode does not render the expression cleanly.
  - **Cite sources properly.** Code implementing an algorithm from a paper cites it with authors, title, journal, year,
    DOI, and the specific section or equation. Citations to local files (anything in a `refs/` working directory, say)
    are not acceptable in the package source, which must stand on its own.
  - **Open-source ready prose.** Write as if the file will land on GitHub today, because it might: no implementation
    notes, apologies, internal jargon, or references to TODOs unless they reference an open issue.

`julia precommit.jl --stats` reports per-file and total SLOC, comment, and docstring counts. SLOC restrictions apply
only to actual source code; comment and docstring volume is unrestricted.

### Style coherence

All source files share one documentation and comment style. New files match existing ones, and touched files are nudged
toward the standard below so the codebase converges over time.

  - **Docstring layout.** Triple-quoted. The opening lines are the call signatures (each indented four spaces, one per
    supported overload), then a blank line and prose in full sentences; argument conventions, defaults, return value,
    and citations follow in that order. The `PhysicalDomain` and `physical_domain` docstrings in `src/physical.jl` are
    worked examples of the layout; `bspline` in `src/stubs.jl` also shows the return type and a citation. The one
    exception is a Makie recipe, whose docstring opens with prose because Makie appends its attribute reference to it.
  - **Section dividers in long files.** A file with clearly separable stages or concerns (`src/fcm.jl`'s moments / cut
    rule / compression pipeline; `precommit.jl`'s CLI / file discovery / formatting / line length / code statistics /
    reporting / main) groups them under single-line dividers carrying the section name, in the canonical form

    ```julia
    # ── Stage name ────────────────────────────────────────────────────────────────
    ```

    drawn with the unicode light-horizontal character `─` (U+2500), not ASCII `-` or `=`. Short files with one tightly
    focused concept need no dividers.
  - **File-top attribution blocks** for ported files are `#` comment blocks (not docstrings) placed before any code.
    They carry the algorithm name; the full upstream citation with authors, title, journal, year, and DOI; the upstream
    license; the source URL; and a brief list of deliberate deviations from upstream. The top of `src/fcm.jl` is the
    model (not yet met: see [Known gaps](#known-gaps)).
  - **Internal-helper commentary** sits *above* the function or struct definition, as a docstring or a `#` block.
    Trailing inline comments on a function header line are not used (not yet met: see [Known gaps](#known-gaps)).
  - **Math and citations** inside docstrings and comments follow the rules of the surrounding prose: unicode math, full
    citations with DOI, no references to local working directories.
  - **Line length.** No line of a `.jl` file is longer than 120 characters — code, comments and docstrings alike,
    indentation included, counted in characters rather than bytes. Prose paragraphs in comments and docstrings are
    filled to that width rather than wrapped early; preformatted material (signature lines, code blocks, tables, aligned
    columns, math displays, section dividers) keeps its own layout. `.JuliaFormatter.toml` sets `margin = 120` for code,
    and `julia precommit.jl --check` fails on any longer line, which is what catches the comments, docstrings and string
    literals the formatter does not rewrap. Split an over-long string literal into pieces joined with `*`, keeping its
    text unchanged. Markdown files are held to the same width — prose filled to 120 characters, with a table row, which
    Markdown cannot break, the one line allowed past it — though `precommit.jl` checks only `.jl` files.

### Coordinate conventions

The conventions used throughout the codebase, in one place so files do not have to re-derive them:

| Frame | Coordinates | Definition |
|---|---|---|
| Physical | `x ∈ ℝᴰ` | The user-facing world frame, in which the background box and `Ω` are given. |
| Reference cell | `ξ ∈ [−1, 1]ᴰ` | The per-cell frame of basis evaluation and quadrature. |
| Region frame | `η ∈ [−1, 1]ᴰ` | The frame of an integration region's corners, `x = (lower + upper)/2 + (upper − lower) ∘ η / 2`, with `η = 0` on an axis the region does not extend along. Each parent cell's reference coordinate is `ξ = parent.ξ0 + parent.dξ ∘ η`. |

  - **Tensor-product basis ordering**: lexicographic over `CartesianIndices(ntuple(d -> 0:order[d], D))`, with the
    dimension-1 axis varying fastest.
  - **Global dof ordering**: per field, then component-major within a field. Constrained dofs are not enumerated; their
    values live in the per-field `DofLayout.constrained_values` matrix (`(raw, component) → value`), reachable from a
    `SystemLayout` as `layout.fields[i].dofs.constrained_values` and read through `constrained_value` / `dof_value`.

Tests enforce these conventions. Document any change to them at the top of the relevant source file and update the
relevant tests in the same PR.

## Dependencies and extensions

Use Julia standard libraries first: `LinearAlgebra`, `SparseArrays` (with `SuiteSparse` reached implicitly through `\`),
and `Test` / `Random` in the test environment. Focused dependencies are acceptable when they remove real work:

  - fixed-size small arrays for geometry and local kernels, e.g. `StaticArrays`;
  - Gaussian quadrature, e.g. `FastGaussQuadrature`;
  - iterative Krylov solvers or preconditioners, through a narrow package;
  - non-negative constrained linear least squares, e.g. `NonNegLeastSquares` (behind the `NonnegativeFit` compression in
    `src/fcm.jl`);
  - forward-mode automatic differentiation for level-set gradients, e.g. `ForwardDiff` (used by `src/implicit.jl`, which
    is why every CSG leaf must accept `ForwardDiff.Dual` arguments);
  - spatial search for scattered-data transfer, e.g. `NearestNeighbors` (the KD-tree behind `RBFP0` in `src/data.jl`);
  - basis-family support packages, only when a non-default basis is added and the dependency is smaller than
    implementing the needed operations;
  - VTK output, e.g. `WriteVTK`, and plotting, e.g. `Makie`, each as a weak dependency behind its package extension — on
    Makie itself, never on one of its backends, which the user picks and which each load Makie;
  - development quality tools, e.g. `BenchmarkTools`, `JuliaFormatter`, `Aqua`, `JET`, `Documenter`.

Rules:

  - Ask before adding a large FEM framework, mesh generator, symbolic system, plotting stack, or non-Julia build step.
  - Do not add a dependency to avoid implementing a small axis-aligned operation that is central to this method.
  - Do not add a production dependency without a short justification in the PR summary.
  - Optional integrations belong in `[weakdeps]` + `[extensions]`, not in `[deps]`. The five the package ships are
    listed under [Package extensions](#package-extensions-ext).
  - `Manifest.toml` is git-ignored. Do not change that.

An extension cannot export a name, so its public names are declared in `src/stubs.jl`. A docstring that `?name` must
find in a session without the extension belongs there too: one written in the extension lands in the extension module's
own docs. Such a name is declared in one of two ways:

  - **As a rule**, one fallback method that, without the extension, raises an error naming the package to load
    (`_extension_missing`).
  - **A Makie recipe** is the exception. `Makie.@recipe` defines the function's catch-all method itself, in the
    extension, and an extension may not overwrite a method of its parent while it precompiles. So the core declares the
    function without any method and lists it in `_METHODLESS_ENTRY_POINTS`, and the error hint `_extension_hint`, which
    `__init__` registers, names a Makie backend on the `MethodError` a call without the extension raises.

A further signature an extension implements is documented on an abstract signature in the core.

## Performance

### Threading model

  - Parallelize over independent integration regions, elements, or batches.
  - Threaded assembly is a deferred compute→gather. Phase 1 computes each region's local block and rhs into its own
    disjoint arena slice (no shared writes, dynamically load-balanced). Phase 2 gives each task a contiguous range of
    dofs, walks every region in list order, and adds the region's entries in its owned rows and columns through the
    column kernel the serial scatter uses. Neither phase takes a lock or a barrier between regions; the one barrier is
    the `@sync` that ends phase 1.
  - There is no gather plan: the arena layout is a closed form of each region's dof count, and the only symbolic input,
    each region's sorted dof list, is cached on the model with the pattern, so repeated assembly pays for it once.
  - The model's assembly cache is locked, and every call checks out its own scratch (workspaces, arena), so several
    tasks may assemble on one `Model` at the same time, serial or threaded. The lock is taken a fixed handful of times
    per call, never in the quadrature loop or the scatter.
  - Do not push into a shared vector from multiple threads without a disjoint-output partition or explicit
    synchronization. Prefer arena slices reduced by a fixed-order gather over thread-local accumulators, whose peak
    memory scales with the thread count.
  - Avoid locks and atomic *accumulation* in the hot scatter. A racy atomic scatter is nondeterministic and was measured
    to corrupt the solve of an ill-conditioned system; a fixed-order gather does not. An atomic counter that only hands
    out region indices for load balancing is fine — it never touches the shared output.
  - Threaded assembly is **bit-identical to serial**: every slot is summed in the serial (region, row) order, so results
    are reproducible run to run and independent of the thread count. Tests may assert equality, not merely a roundoff
    tolerance.

### Allocations and benchmarks

  - Keep local element kernels allocation-free after setup.
  - Use `@inbounds` only where bounds are protected by tests, and `@views` where slicing would allocate.
  - Run `@code_warntype`, `@allocated`, or `JET` on new hot paths when performance matters.
  - Benchmark before adding complicated optimizations.

For performance-relevant changes, run the targeted benchmarks ([Benchmarks](#benchmarks-benchmarks)) before and after;
the BenchmarkTools suite needs Unfitted dev'd into its environment once:

```bash
julia --project=. benchmarks/intersections.jl
julia --project=. benchmarks/assembly.jl
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'   # once
julia --project=benchmarks benchmarks/runbenchmarks.jl
```

## Testing

Every feature needs tests. Prefer small deterministic tests over large demos.

### Testing rules

  - Do not rely on plots for correctness; include analytic or manufactured-solution checks wherever possible.
  - Separate fast unit tests from expensive convergence studies, which belong in `examples/`, `benchmarks/`, or a
    clearly marked testset that is not required for every quick edit.
  - When fixing a bug, add a regression test that fails without the fix.
  - When adding a new basis family, add at least one small assembly test, one boundary-constraint test, and one
    projection/evaluation test for that family.
  - Pass constant-valued inputs as bare constants, not as closures, for the coefficient-shaped keywords that accept both
    — `source`, `coefficient`, `diffusion`, Neumann values and Dirichlet data, which all pass through `_datum_at`. Write
    `source = 1.0`, never `source = x -> 1.0`. Every distinct closure is a distinct *type*, and each one forces a full
    re-specialisation of the assembly emission kernel, for an assembled result that is identical either way.

### Required coverage

What the suite must pin, by area, and the test file that pins it today. Where a row is only partly met, the shortfall is
listed under [Known gaps](#known-gaps).

| Area | Must be pinned | Where |
|---|---|---|
| Geometry | `AxisBox` construction, containment, intersection, volume/measure and tolerance merging in 1D, 2D, 3D and at least one 4D smoke case; physical-to-reference and reference-to-physical maps for arbitrary `D` | `test_geometry.jl` |
| Meshes | Cartesian mesh construction in 1D, 2D, 3D, plus a small 4D smoke test of mesh indexing and element lookup if inexpensive; parent-element lookup by midpoint | `test_mesh.jl`, `test_intersections.jl` |
| Basis | integrated Legendre 1D values, endpoint behaviour, gradients, polynomial exactness and quadrature consistency; tensor-product ordering, values and gradients in 2D and 3D, plus one low-order 4D smoke test | `test_basis.jl` |
| Basis-family interface | tests that fail if assembly or constraints rely on integrated-Legendre-only assumptions outside the basis layer | `test_basis_bspline.jl`, `test_cell_order.jl` |
| Constraints and dofs | overlay-boundary constraint detection; global dof enumeration after constraints | `test_constraints.jl` |
| Integration regions | construction for non-matching mesh pairs in 1D, 2D and 3D, plus a small 4D smoke case | `test_intersections.jl` |
| Assembly | symmetry and SPD behaviour of simple Poisson/Laplace and 1D bar systems after constraints; correct right-hand side for nonzero Dirichlet data | `test_assembly.jl` |
| Projection | variational projection between two shifted overlay configurations | `test_projection.jl` |
| Public-API workflow | build and solve at least one small problem without manual dof enumeration, manual constraint masks or manual intersection-region construction | `test_api.jl` |
| Regression examples | required by *capability*, not by file name, so that reorganising `examples/` cannot quietly drop one: a 1D problem carrying a discontinuous or under-resolved feature; a 2D singular corner problem (`reproductions/singular_square_2d`, UMLHP §5.2); a small-overlap conditioning smoke test (`reproductions/conditioning_small_overlap`, §5.3); a moving heat-source smoke or reduced regression test (`reproductions/traveling_laser_2d`, the package's own moving-source study against an exact convolution reference) | the [example smoke suite](#example-smoke-suite) |

The feature areas pin more, so each is a list.

  - **Selective activation** (`test_activation.jl`, `test_coverage_reduction.jl`, `test_basis_bspline.jl`):
    - dof-structure equivalence between a `LevelMask` and a geometrically smaller overlay covering the same active
      cells;
    - the artificial-boundary constraint on internal active/inactive interfaces, decided by the same rule as a box face:
      a mask face through a hole carries none while the stretches of it that meet `Ω` stay clamped, and a face the two
      cells' verdicts settle is never classified itself;
    - the `activate!` / `deactivate!` invalidation contract;
    - every region parent an active cell of its level, on a masked uniform overlay, an immersed level and a ladder — the
      invariant the hot path's mask-free set read rests on;
    - `dilate`'s box shape (one cell grows to `∏_d (2·by_d + 1)`), its per-axis widths, its additivity in the width (two
      dilations are one by the sum, and width 0 is the identity), its clipping at a box face, its refusals, and its
      agreement with `support_extension` at the family's own radius (`test_basis_bspline.jl`).
  - **Immersed boundary** (`test_physical.jl`, `test_fcm.jl`):
    - the strict-α cell-level fictitious fold drops the right cells;
    - the region-kind tags `:full` / `:cut_fitted` / `:fictitious_alpha` / `:cut_fallback` / `:cut_failed` /
      `:cut_alpha_failed` appear (`:cut_fallback` through `cut_cell_rule`'s status and a hand-built record: no real
      geometry is known to reach it);
    - every built-in compression reproduces the cut rule's moments with non-negative weights at ≤ `nbasis` points, at
      every length scale, and `MomentBasis` evaluates the tensor Legendre basis and sums moments that integrate it
      exactly;
    - Carathéodory pruning keeps a subset of the cut rule, directly and through recombination, stops at the numerical
      rank on a sliver, keeps the moments of a rank-deficient matrix under weights spread over thirty orders of
      magnitude, undoes a pass that drifts and keeps a sound one in Float32, and carries Float32; the pure-Julia null
      space agrees in dimension with LinearAlgebra's `nullspace` on wide, rank-deficient, zero, tall and Float32
      matrices;
    - the NNLS retry rescues a many-feature cut cell that its first attempt cannot fit, from the same cut rule, and a
      compression that misses the failure threshold, or keeps no weight at all, falls back to the cut rule;
    - α-FCM weight scaling on a fully fictitious cell, end-to-end SPD on a cut-disk Poisson problem, and a 1D analytic
      mass-entry check on a cut cell;
    - a caller's cut rule and compression replace the package's on cut regions only, compose with each other and with α,
      count their calls atomically as stages that may run on any thread must, and are rejected for a negative weight; a
      cut rule that finds nothing routes its region to `:cut_failed` / `:cut_alpha_failed`, one of zero weights being
      read as empty (`cut_cell_rule`'s `:empty` status);
    - the rule cache, keyed by `(box, moment order)` and owned by the model, is reused across rebuilds without
      rebuilding a rule — `move!`, `adapted`, `moved`, an `estimate`'s twin — and stays bounded by the current plan,
      pinned with a counting `cut_rule` (`test_physical.jl`, `test_ladder.jl`); within one plan a key never repeats,
      because `_partition` emits pairwise-disjoint boxes.
  - **Multi-domain coupling** (`test_coupling.jl`, `test_boundary_mesh.jl`): a machine-precision Nitsche interface patch
    test across two independently discretised subdomains; the `couple` four-block sign pattern; and the skip of
    interface regions where one side has no cover, exercised for geometry outside a subdomain (`test_coupling.jl`'s
    point interface, `test_boundary_mesh.jl`'s triangle reaching past the grid). The variant with the trace in an
    inactive cell of a masked subdomain must be exercised too (not yet met: see [Known gaps](#known-gaps)).
  - **Covered-mode pruning** (`test_coverage_reduction.jl`, `test_activation.jl`, `test_ladder.jl`):
    - covered high-order modes eliminated, and buried linear modes deduplicated only under a nested finer level;
      `reduced_mode_counts` matches the eliminated set;
    - the user-masked (`active =`) overlays `test_coverage_reduction.jl` builds check the material oracle and the
      coverage memo against a direct recomputation, not a pruned space — except the hole-face testset, which checks a
      masked overlay's reduced layout and its rank;
    - the fully deactivated case, indirectly, by `test_activation.jl`'s "fully-deactivated overlay equals no overlay",
      which compares active-unknown counts against an overlay-free space;
    - coverage under a ladder (`test_ladder.jl`): a split cover retained ("a mode covered by two different levels is
      retained"), a single cover shed ("a lower-order cover still sheds"), and losslessness by rank against the
      unreduced twin ("covered-mode pruning on a ladder is lossless");
    - coverage of a cut cell by its share of `Ω`, and an overlay face outside `Ω` left unclamped, against a *control* —
      the same overlay with its box extended over the cut cells, which leaves its active cells unchanged — for a folded
      and a cut layer of overlay cells at the face, at p = 1…3, on a planar and a curved boundary and in 3D: same
      unknowns, same eliminations, same Gram spectrum, full rank;
    - a face holding material staying clamped, a cell whose material crosses the face staying uncovered, the coincident
      face's remaining null modes (the limitation documented under
      [Mesh and refinement model](#mesh-and-refinement-model)), and both material queries of the oracle, branch by
      branch.
  - **Per-cell polynomial order** (`test_cell_order.jl`, `test_basis_bspline.jl`, `test_graded_golden.jl`): C⁰
    conformity across an order jump, asserted as a per-basis-function one-sided trace jump `== 0`; the minimum rule's
    shape, and its set-membership form under `:trunk` where a per-axis comparison disagrees; inactive incident cells
    never lowering a shared entity; the `elevate` / `cell_orders` round trip as an identity; the intersection form
    reproducing the componentwise-minimum form cell for cell, and the index-set filter's monotonicity in the order;
    refusal of a non-uniform field — through `space` and through `elevate` — on a family that does not declare
    `_supports_cell_order` (`test_basis_bspline.jl`); and the graded-kernel golden digest (`test_graded_golden.jl`). An
    example should teach the feature (not yet met: see [Known gaps](#known-gaps)).
  - **Nested refinement ladders** (`test_ladder.jl`): `ladder` nests by construction; losslessness witnessed by
    `size(gram(V), 1) == rank(gram(V; prune = false))`, with a non-nested negative control; `overlapping_cells` in both
    directions and under an anisotropic split; both forms of `adapt` — the `level => mask` pairs and the per-base-cell
    depth map, with grading; `adapted` keeping the source model intact and reusing the rule cache.
  - **Automated hp adaptivity** (`test_adaptivity.jl`):
    - `estimate`'s effectivity bounded over a range of unknowns, and its refusals, each before `V⁺` is built — an
      unassembled model, `enrichment = 0`, a second field, a stale solution, a form it cannot measure,
      per-quadrature-point state, and `V⁺ ⊉ V`;
    - its superposed enrichment on a family that carries no per-cell order, indexed back onto the caller's levels, and a
      whole loop driven on such a stack to a hundredfold reduction in both the error and the estimate;
    - the h- and p-steps landing on the levels and cells they claim; the guards that bypass the h/p decision (no step
      available, not a leaf, already at `pmax`); a cell keeping p only when it met its prediction;
    - `coarsen` as the exact inverse of both steps, and its refusal on a covered cell; Dörfler marking monotone in θ,
      with group completion;
    - the loop reducing both the error and the estimate, with one cycle of the full `previous`-threaded decision running
      unchanged in 1D, 2D and 3D.
  - **Package extensions**:
    - the B-spline family (`test_basis_bspline.jl`): basis values, constraints and a small assembly; its refusal of a
      per-cell order through both `space` and `elevate`; the support-selection dof ledger — per axis `n + p` with both
      faces physical and `n − p` with both artificial, `p` eliminated functions per artificial face, and the closed form
      `n + p − 2(m + 1)` at a requested continuity `m`; the superposition being smooth and not merely each level,
      asserted as the decade ratio of the normal-derivative jump across an overlay face (10 at `:maximal`, below 1.5 at
      `C⁰` and for integrated Legendre); an arbitrary mask producing no linear-constraint pivot, which spares assembly
      any pivot to condense and makes the reconstruction defect unreachable;
    - the dedup that keeps a nested B-spline stack non-singular, including the mixed-family direction where a degree-1
      cover spans an integrated-Legendre level's hats, and the fold configurations where the cover's face lands before,
      inside and beyond the cut band. The configuration with the cover's face on a knot of the level below, inside the
      band of cut cells, is pinned full rank and lossless against the unreduced twin under the *clamped* mechanism as
      well, which `bspline(; continuity = m)` for `m < p − 1` still selects: a clamped cover reproduces a *combination*
      of two buried functions without reproducing either only behind a clamped face that no material crosses, and the
      artificial-boundary rule clamps a face only where it holds material. The one configuration that still yields such
      a null mode is a face `∂Ω` lies on exactly (see [Mesh and refinement model](#mesh-and-refinement-model)). A cover
      holding no base cell whole is pinned under both mechanisms;
    - the MeshIO signed-distance leaf (`test_meshio_ext.jl`) and the Tensors notation round trip
      (`test_tensors_ext.jl`);
    - the WriteVTK writer's bundles, series and quadrature clouds as serialisations of the consumer-neutral sampling
      records (`test_vtk.jl`), and the records themselves — the sample cells and their cut refinement, the arrays and
      their element types, the refusals, a discretisation sampled without a solution — as records, with no writer loaded
      (`test_postprocessing.jl`);
    - the Makie recipes' primitives (`test_makie_ext.jl`): a field's sample cells, values, components, magnitudes,
      region edges, warped heights and 1D graph, with and without a solution, and their cut at `∂Ω`, exact at a planar
      level set and converging at second order in `cut_depth` on a disk; `∂Ω` drawn by every plot; each level mesh's
      cells, colours, edges and stacking in 2D, the ladder of levels in 1D, and the colouring by an error estimate; the
      quadrature cloud — all asserted exactly against those records on the plot objects Makie builds, with no backend
      loaded and nothing rendered;
    - in a fresh process that loads Unfitted alone (`test_api.jl`), the entry points asking for their package —
      `using WriteVTK` for the writers, a Makie backend for the plots, with keywords or without — while writing nothing
      and loading none of the extension packages.

### Example smoke suite

Every `examples/<tier>/<name>/<name>.jl` script runs to a clean exit in the default suite (`test_examples.jl`), at its
published size or at a coarser one its case comment justifies — reproductions are not coarsened for speed, with the one
exception [Examples](#examples-examples) states — with a finite, sane headline metric wherever the script prints one,
banded at the value measured on the configuration the case actually runs.

Where a tutorial's *prose* tells the reader what to look for in the output, that claim is asserted too, so a tutorial
whose text and output disagree fails here: the overlay's order-of-magnitude improvement over the base level, the
agreement between a reduced and an unreduced stack, and the conditioning one duplicated B-spline mode costs (recorded at
253 against 4.9e15, asserted as a band below 1e4 and a floor above 1e10).

The cases share one batched subprocess (`test/run_examples_child.jl`), each included into its own `Module`, because the
per-process compilation floor dominates when every script pays it separately. Each case declares the packages it needs
beyond `Unfitted` in a `requires` field, and a case the active project cannot resolve is skipped rather than failed. The
examples that need a weak dependency — the eleven that write ParaView output (`WriteVTK`), and those needing `Tensors`,
`BasicBSpline` or `FileIO` + `GeometryBasics` + `MeshIO` — run under `Pkg.test`, where all of those sit in the `test`
target, and skip in a bare `--project=.` isolation run. The exception is `applications/time_integration`, one of the
eleven, which runs nowhere (see [Known gaps](#known-gaps)).

### Reporting numerical results

For any model problem reported in a PR, issue, paper, or example, give at least:

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

Do not claim agreement with the paper unless the same setup, polynomial degrees, overlay sizes, quadrature, basis
family, and error definitions are implemented — or unless the differences are clearly stated.

## Known gaps

These are rules of this document that the code does not yet meet. Each is kept as a rule rather than weakened to match
the code; closing a gap means changing the code and deleting its entry here and its marker at the rule.

**Code and API**

  - **No state transfer across an h-step on an immersed model**.
    [Moving overlays and state transfer](#moving-overlays-and-state-transfer) states the restriction, its reason and the
    p-step that is exempt.
  - **Errors are told apart by message only** ([Design goals](#design-goals), "Actionable errors"). The package defines
    no exception types of its own — it raises Base's: mostly `ArgumentError`, with `DimensionMismatch`, `BoundsError`,
    `MethodError` and the `ErrorException` of a plain `error` call; a singular system is not detected (`solve!` reports
    `converged` from the finiteness of the residual); and small overlaps are reported in `diagnostics`, never warned.
  - **An export clashes with an extension's package** ([Naming](#naming)). `triangle_mesh` is exported by both
    `Unfitted` and GeometryBasics, a trigger package of the MeshIO extension. The documented
    `using Unfitted, FileIO, MeshIO` does not surface the clash, since GeometryBasics loads transitively, but a session
    that also runs `using GeometryBasics` must qualify `Unfitted.triangle_mesh`.

**Documentation and style**

  - **The model attribution block has incomplete citations** ([Style coherence](#style-coherence)). The file-top block
    of `src/fcm.jl`, which the rule names as the model, cites QuESo, Carathéodory, Businger–Golub and Litterer–Lyons
    without titles, and abbreviates the QuESo authors.
  - **A trailing comment on a function header** ([Style coherence](#style-coherence)). The header of `_region_edges!` in
    `ext/UnfittedMakieExt.jl` carries one.
  - **An example cites a published result without its DOI** ([Examples](#examples-examples)).
    `reproductions/laplace_unit_square_smooth` cites Wagner & Liu (2000) without one.

**Examples**

  - **Tutorials that hardcode the dimension** ([Examples](#examples-examples)). `tutorials/03_immersed_fcm`,
    `tutorials/04_bspline` and `tutorials/06_adaptive_hp` hardcode two dimensions (literal 2D boxes, `x[1]` / `x[2]`, a
    2D polyline boundary); only `02_overlays` and `05_ladders` keep the dimension one named constant.
  - **No 1D discontinuous or under-resolved regression example** ([Required coverage](#required-coverage), "Regression
    examples"). The one 1D example, `tutorials/01_first_solve`, solves a *smooth* manufactured sine on the interval: it
    exercises the 1D path end to end, and the smoke suite pins its error, but it carries no discontinuity and nothing
    the mesh fails to resolve. The nearest live coverage is a unit test rather than an example: `test_assembly.jl`'s
    "discontinuous coefficient need not align with overlay boundaries" puts a 1D jump in the coefficient inside an
    overlay cell; in 2D, `applications/interface_coupling_2d` covers a jump in material data across a seam, and
    `tutorials/02_overlays` a feature the base mesh cannot resolve. Restoring the requirement means adding a 1D
    discontinuous- or unresolved-feature run — most cheaply as a further section of `tutorials/01_first_solve`, or as a
    `reproductions/` entry if the paper's elastic bar, whose material interface falls inside a cell, is wanted back
    verbatim.
  - **No example teaches per-cell polynomial order** ([Required coverage](#required-coverage),
    [Common workflows](#common-workflows-the-api-must-keep-easy)). `elevate` and `elevated` occur nowhere under
    `examples/`, no example passes an array or a predicate to `space(...; order=)`, and the tutorial sequence goes from
    `05_ladders` to `06_adaptive_hp`, which drives the automated loop without ever showing the manual verb it is built
    on.
  - **`applications/time_integration` never runs** ([Example smoke suite](#example-smoke-suite)). It needs
    `OrdinaryDiffEq`, which is deliberately absent from the package's dependencies, weak dependencies and test target —
    it is a hundred-package tree, and the example exists precisely to show that no extension is needed for it. The case
    is therefore skipped everywhere and has no automated coverage; it must be run by hand, or by a separate CI job, on
    the example's own project.

**Tests**

  - **Basic coverage met only in part** ([Required coverage](#required-coverage), the Geometry, Meshes and Basis rows).
    Box containment is tested in 2D only; mesh construction directly in 2D and 4D, and in 1D and 3D only through
    `test_ladder.jl`'s ladders; the integrated Legendre basis has no direct 1D polynomial-exactness or
    quadrature-consistency testset (`test_projection.jl`'s exactly represented polynomial covers it indirectly); and 3D
    tensor ordering is checked by count only.
  - **No coupling test over a masked subdomain** ([Required coverage](#required-coverage), "Multi-domain coupling"). No
    coupling test masks a subdomain, so the variant where the trace lies in an inactive cell is not exercised.

## Submitting changes

### Workflow

  1. Branch from `main` with a descriptive name. Short topic branches (`feature/...`, `fix/...`, `docs/...`) are
     preferred.
  2. Make one focused change per PR. Keep diffs small and reviewable.
  3. Add or update tests alongside the implementation.
  4. Run the test suite and `julia precommit.jl --check` exactly as in [Verifying your setup](#verifying-your-setup).
     Both must pass.
  5. Open a pull request with a description that explains the *why*, the user-visible change (if any), and any numerical
     results the change produces.

### Pull-request expectations

A PR description should include:

  - a one-line summary suitable for the changelog;
  - a short rationale: what problem this solves, and why this approach;
  - a list of files touched and the gist of what changed;
  - test changes: new tests added, existing tests adjusted (and why);
  - numerical results if applicable, following [Reporting numerical results](#reporting-numerical-results);
  - any open questions or follow-up tasks.

### Definition of done

A task is done only when:

  - the implementation is compact and specialized to this method;
  - relevant tests pass under one thread and under several;
  - new behavior is covered by tests, or by a justified benchmark or example;
  - numerical invariants are preserved;
  - core changes remain dimension-independent unless explicitly scoped otherwise;
  - public API changes are documented, tested, and justified by a real user workflow;
  - common workflows remain accessible without manual low-level bookkeeping;
  - basis-family assumptions are localized to the basis and dof-layout layer and documented;
  - performance-sensitive changes avoid obvious allocations in hot loops;
  - documentation or comments explain any new convention;
  - upstream attributions in `NOTICE.md` and at the top of ported files remain accurate after any algorithmic edits to
    ported code;
  - the PR description includes verification commands and outcomes.

## Anti-patterns

These are the things we do not do, ever, without a very good reason spelled out in the PR:

  - Build a general FEM framework; this project is specialized ([Source-code compactness](#source-code-compactness)).
  - Let the public API become a dumping ground for internals or one-off helpers
    ([Adding to the public API](#adding-to-the-public-api)).
  - Force users through manual dof enumeration, constraint masking, or intersection bookkeeping for common workflows
    ([Common workflows](#common-workflows-the-api-must-keep-easy)).
  - Hard-code the core library to 1D, 2D, or 3D ([Dimension independence](#dimension-independence)).
  - Assume integrated Legendre basis behavior outside the basis and dof-layout layer
    ([Basis family model](#basis-family-model)).
  - Make B-splines or other basis families require rewriting geometry, intersections, assembly, projection, or solvers
    ([Basis family model](#basis-family-model)).
  - Hide conditioning problems by silently dropping tiny overlaps
    ([Small overlaps and conditioning](#small-overlaps-and-conditioning)).
  - Transfer a field's coefficients by point interpolation after moving overlays; per-quadrature-point history data is
    the exception, and `RBFP0` is the scheme for it
    ([Moving overlays and state transfer](#moving-overlays-and-state-transfer)).
  - Mix physical and artificial boundary constraints ([The superposition model](#the-superposition-model)).
  - Introduce Boolean-indicator geometry representations for the immersed domain; the package commits to signed scalar
    level sets ([Immersed boundary](#immersed-boundary-finite-cell-method)).
  - Vendor third-party code without adding (or updating) the corresponding entry in `NOTICE.md` and a top-of-file
    attribution ([Reference material](#reference-material)).
  - Add non-Julia production code.
  - Add heavy dependencies without approval ([Dependencies and extensions](#dependencies-and-extensions)).
  - Leave generated plots, scratch files, or benchmark artifacts in the repository unless explicitly requested.

## Questions, ideas, contributions

Feel free to open an issue for design discussion before opening a PR for substantial changes — especially anything that
touches the public API surface, the math contracts, or the threading model.

Thank you for contributing.
