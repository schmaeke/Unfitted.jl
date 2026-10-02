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
julia --project=. -e 'using Pkg; Pkg.test("Unfitted"; julia_args=["-O0"])'
julia precommit.jl --check
```

`-O0` is the standard developer setting for this suite and not a
shortcut: the suite is compile-bound rather than compute-bound — a timing
run charged 265.9 s of a 270.6 s invocation to the compiler — so
switching the optimiser off roughly halves the time to a verdict while
every assertion still runs, and still passes. Continuous integration
keeps exactly one leg at Julia's default optimisation level, so the
inlining and specialisation decisions users get are exercised too.
(`Pkg.test` forces `--check-bounds=yes` on every leg, so no leg is
production code generation in the strict sense — the variable that leg
controls is the optimiser.)

Both commands must exit zero on a clean checkout. If they do not, please
open an issue with the failing output before opening a PR.

Tests must pass under `JULIA_NUM_THREADS=1` *and* under more than one
thread; do not rely on a fixed thread count.

## Repository layout

```text
Project.toml                  package metadata + production dependencies
LICENSE.md                    MIT (Unfitted.jl)
NOTICE.md                     third-party attributions (QuESo BSD-4-Clause) + clean-room algorithm citations
README.md                     public landing page
CLAUDE.md                     agent-instruction pointer documents
CONTRIBUTING.md               this file — contributor + agent guide
CITATION.cff                  machine-readable citation metadata
.JuliaFormatter.toml          repo-wide formatting config (yas style, indent 4)
precommit.jl                  formatter wrapper + SLOC/comment/docstring stats
src/                          library source
ext/                          package extensions (BasicBSpline, FileIO+MeshIO+GeometryBasics, Tensors)
test/                         unit and regression tests
examples/                     runnable scripts in three tiers (tutorials/, applications/, reproductions/), one sub-directory per example, each with its own Project.toml
benchmarks/                   targeted performance benchmarks (own Project.toml)
.github/workflows/ci.yml      CI: tests on Julia 1.10 and latest, 1 and 4 threads (three legs at -O0, one at default -O), plus the format check
```

### Source files (`src/`)

Each file owns one well-defined concern. Cross-file dependencies are
limited to the imports made obvious by `src/Unfitted.jl`'s include order:

| File              | Responsibility                                              |
|-------------------|-------------------------------------------------------------|
| `Unfitted.jl`     | top-level module: imports, exports, include order           |
| `geometry.jl`     | D-dimensional axis-aligned boxes, coordinate maps, tolerances |
| `physical.jl`     | CSG level-set tree, `PhysicalDomain`, three-valued cell classifier |
| `implicit.jl`     | Saye dimension-reduction implicit quadrature on the level set (volume rules) |
| `basis.jl`        | basis-family interface and capability traits, integrated Legendre default, the `:tensor` / `:trunk` index sets and the `_index_admissible` filter the minimum rule is expressed over, tensor-product evaluation |
| `fcm.jl`          | finite-cell-method cut-cell quadrature: exact moments from `implicit.jl`, non-negative (NNLS) moment fit |
| `mesh.jl`         | Cartesian mesh levels, the superposition `Space`, per-cell activation masks (`LevelMask`) and per-cell polynomial order (`CellOrders` / `CellModes`); the `space` / `overlay` / `adapt` / `elevate` verbs |
| `intersections.jl`| admissible integration regions for non-matching meshes; cut/fictitious region-quadrature dispatch |
| `coverage.jl`     | per-level covered-cell masks (`Coverage`, `build_coverage`); the mask-aware, fictitious-fold-aware covering rule behind covered-mode pruning |
| `ladder.jl`       | nested refinement ladders: `ladder` declaration, the depth-map form of `adapt` and the `depth_masks` grading behind it, cross-level cell mapping (`overlapping_cells`), the `is_nested` predicate |
| `dofs.jl`         | dof layout, raw/active enumeration, overlay/boundary-constraint detection, the hp minimum rule across an order jump, and the leaf-semantics eliminations (`_coverage_constraints`) |
| `dirichlet.jl`    | physical Dirichlet conditions and boundary selectors, the per-key boundary-face detection `dof_layout` eliminates on, codim-K facet regions and quadrature, and the L² boundary projection for nonzero data |
| `surface.jl`      | immersed-boundary surface meshes (`BoundaryMesh`) and surface-region integration |
| `problems.jl`     | `Field`, weak-form channels and blocks (`BlockForm`/`LoadForm`/`WeakForm`), `Problem` |
| `coupling.jl`     | multi-domain interface coupling: the `Interface` `on=` tag, two-sided `InterfaceRegion` construction, `InterfaceForm`, and the four-block `couple` jump expansion |
| `model.jl`        | `Model` lifecycle: `prepare`, `move!`/`moved`, `activate!`/`deactivate!`, the non-mutating `adapted` / `elevated` rebuilds, `active_cells` / `cell_orders` on a model, the discretisation pin, diagnostics |
| `assembly.jl`     | coupled Galerkin assembly: channel calculus, cached symbolic-scatter (Gustavson) pattern, serial scatter + two-phase compute→gather threaded path |
| `solvers.jl`      | `Solution` / `SolverDiagnostics`, the default direct sparse solve, and the `linear_solver` hook for external Krylov or preconditioned solvers |
| `projection.jl`   | variational and rewire-based state transfer between models  |
| `adaptivity.jl`   | automated hp adaptivity: `estimate` (Bank–Weiser indicator), Dörfler marking, the h-versus-p decision by Melenk–Wohlmuth predicted error reduction, `refine` and `coarsen` |
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

Self-contained runnable scripts, organised in three tiers. The tier is
part of the path, and choosing it is the first decision when adding an
example:

| Tier | Purpose | What belongs there |
|---|---|---|
| `tutorials/` | Teach the API | Numbered, read in order, each building on the last. A header block states what the reader will learn and what they should already know; the comments teach rather than annotate; the dimension is never hardcoded unless the example is deliberately 1-D. Aim at roughly 60 SLOC of code — comment volume is unrestricted, and generosity is expected. |
| `applications/` | Show the package on a recognisable problem | A problem a reader would know from a textbook or from practice, solved plainly, with no cleverness that obscures the API. This is also where the package is shown working alongside third-party Julia packages (`Tensors.jl`, `FileIO`/`MeshIO`, `OrdinaryDiffEq.jl`). |
| `reproductions/` | Hold the scientific record | Benchmarks whose problem, protocol and headline number are fixed and must not move: the method paper's benchmarks at their published configurations, the FCM moment-fit reproductions, and the package's own benchmarks scored against an external reference (`adaptive_tanh_layer_2d` against deal.II 9.5.1's step-27 on a fixed lattice). These are pinned by the example smoke suite and are not to be coarsened for speed. The one standing exception is `traveling_laser_2d`, which runs a fifth of its revolution in the default suite because the published configuration takes six and a half minutes at `-O0` on eight threads against roughly one and a half for the whole examples testset; its case comment records the full-study numbers and what the shortened run gives up. |

Each example lives in its own sub-directory
`examples/<tier>/<name>/` with a `<name>.jl` driver and a
`Project.toml` that wires `Unfitted` in via `[sources] = {path =
"../../.."}` — three levels up, because of the tier. Example-specific
dependencies (e.g. `StaticArrays`, `LinearAlgebra`, or third-party
packages used only for an imported-geometry or time-integration demo)
belong in the example's own `Project.toml` and must never be added to
the package's top-level `Project.toml`. Per-example `Project.toml`
files set `julia = "1.11"` in `[compat]` because `[sources]` is a
Julia 1.11 feature; the package itself still supports Julia 1.10.

A shared helper `examples/reporting.jl` is `include`d by every
example via `joinpath(@__DIR__, "..", "..", "reporting.jl")` and
depends only on what `Unfitted.jl` already exports. Every example
prints its report through `print_run_report`, so the whole set reads
the same way and the smoke suite has one output format to parse.

Run an example with

```bash
julia --project=examples/<tier>/<name> examples/<tier>/<name>/<name>.jl
```

`Manifest.toml` is git-ignored repository-wide, so the first run of an
example resolves and installs that example's own dependencies.

Output artifacts go under `examples/<tier>/<name>/output/`, which is
git-ignored repository-wide.

Examples should demonstrate the *public* API; reach into
`Unfitted._internal` only when an example is specifically about those
internals, which none of the current set is.

Prose in an example is held to the same standard as prose in `src/`:
full sentences, unicode maths, never LaTeX, and a citation with
authors, title, journal, year and DOI wherever a published result is
involved. An application or reproduction must state the problem, the
reference if there is one, and what the printed metric means, so that a
reader can tell from the output alone whether the run succeeded.

### Benchmarks (`benchmarks/`)

Small targeted benchmarks with their own `Project.toml`. Driven and
documented by `benchmarks/runbenchmarks.jl`.

### Developer tooling

The repository root ships `precommit.jl`, a thin wrapper around
`JuliaFormatter` that also prints per-file and per-bucket SLOC / comment /
docstring statistics. Three modes:

```bash
julia precommit.jl          # format every .jl file in place + print stats
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
boundaries need not coincide with lower-level element boundaries, and
overlays are never topologically merged with their parents. The package
does, however, carry a pruning rule for redundancy. Under *leaf
semantics*, which are unconditional and have no keyword, every
high-order mode whose entire incidence stencil is covered by a finer
level is eliminated, leaving the linear skeleton. Elimination is
per-mode, not per-cell: a mode is shed only when *every* cell it is
incident to is covered, so an edge or face mode straddling the boundary
of the covered region survives even though its own cell is covered. A
buried linear mode that a *nested* finer level reproduces exactly is
deduplicated. Coverage is mask-aware — a user-masked cell blocks
coverage, a fictitious fold does not.

Covered-mode pruning is a basis-family-specific feature, and
`_coverage_constraints` is where a family implements it. The
`Level{D,T,<:IntegratedLegendre}` method does both eliminations above.
The B-spline extension overrides the hook too, but only for the dedup:
that family has no bubble/skeleton split, so its one reducible mode is
the one a covering level reproduces exactly — and eliminating it is not
an accuracy trade but the thing that keeps a nested B-spline stack
non-singular, since the two copies are linearly dependent. Under
`continuity = :maximal` the burial test there is *exact* rather than
conservative: it walks the subdivision and asks whether every fine
function the buried one is a combination of survived the cover's own
support selection, which is necessary as well as sufficient because the
subdivision coefficients are strictly positive and its representation is
unique. Any further
family hits the generic fallback that returns an empty constraint list,
so on such a space leaf semantics eliminate nothing. `prepare(problem;
prune = false)` is the one way to build a space's unreduced twin; it is
a diagnostic — exactly singular on a nested stack — rather than a
discretisation to solve with. See `src/coverage.jl` and
`_coverage_constraints` in `src/dofs.jl`;
`diagnostics(...).reduced_mode_counts` reports the count per level,
concatenated field-by-field (one entry per level of each field, in
declaration order).

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

  1. For each coordinate direction, collect the element-boundary
     coordinates of the participating meshes that bound an *active* cell
     of their own level — all of them on an unmasked level. A coordinate
     no level keeps separates two slabs carrying the same coverage
     signature, which step 5 would merge again, so dropping it costs
     neither a region nor any exactness, and the candidate grid then
     follows the live cells rather than the declared resolution.
  2. Sort and merge coordinates closer than the geometry tolerance.
  3. Form non-degenerate intervals between adjacent coordinates.
  4. Take the Cartesian product of intervals to form candidate boxes.
  5. Compute each candidate's per-level *coverage signature* at its
     midpoint (which cell of each level contains it, or `0` when the
     level's mesh does not reach the point or that cell is inactive —
     activity is a per-cell property carried by `Level.mask`, never a
     property of the level), then greedily merge axis-adjacent
     candidates that share a signature. Every cell inside a merged box
     still lies inside a single cell of every covering level, so the
     `C^∞` contract below is preserved while the region count stays
     proportional to the number of genuinely distinct coverage
     patterns.
  6. For every merged box, resolve the covering parent cells by midpoint
     containment and keep the box only if it satisfies the caller's
     `criterion`, expressed over those parent *levels*: `:any_parent`
     (the default — at least one level covers the box; used by assembly,
     field evaluation, and visualization), `:all_levels` (every level of
     the space covers it), or a caller-supplied predicate
     `parents -> Bool`. No in-tree caller uses the predicate form:
     `projection.jl` builds its transfer regions from `_merged_boxes` /
     `_parents_covering` directly and applies its own target-coverage
     test rather than going through a `criterion`. Any other value
     raises `ArgumentError`.
  7. Dispatch the region quadrature on the `PhysicalDomain`
     classification — tensor Gauss on `:full`, α-scaled tensor Gauss on
     `:fictitious` with `α > 0`, the moment-fit pipeline on `:cut` (or
     the domain's `cut_quadrature` callable in its place, see *Region
     quadrature kinds* below); a `:fictitious` region under strict α
     (`α = 0`) is dropped outright.
  8. Store each contributing parent element with the box mapped into
     that element's local coordinates, record a `SmallOverlap` for every
     region below `tolerance.small_volume`, and track the largest
     moment-fit residual seen.

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

**Requirement not yet met:** on an immersed model — a space carrying a
`PhysicalDomain` — there is no state transfer across an h-refinement or an
h-release. `L2Projection` refuses any target carrying a physical domain
outright, and `Rewire` refuses a target whose active basis does not contain
the source's, which is what leaf semantics make of a refinement and what a
release makes of a cover. An adaptive or transient loop on an immersed model
therefore re-solves on the new space rather than carrying its iterate across,
and `estimate` needs that fresh solve before it can be called again. The
restriction is specific to the h-step: a pure **p**-step buries nothing, so
`Rewire` accepts it — measured on the unit square minus a centred disc, an
8×8 order-2 base raised to order 3 on two leaf cells (216 → 223 unknowns)
transferred with the evaluated field unchanged to the last bit at four
interior points. `test_adaptivity.jl` pins both refusals so neither can decay
into silently dropped coefficients.

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
    per-level order choices, and grading the polynomial order from cell
    to cell within one level.
  - Declaring a nested refinement ladder up front and switching its
    cells on per level as the solution develops.
  - Selectively activating or deactivating individual overlay cells as
    a feature evolves in time, and growing a marked region by a margin —
    `dilate` for a width the caller chooses, `support_extension` for the one
    the basis family needs.
  - Handing the refinement decision to the solver: estimate, mark,
    refine, coarsen, with the loop itself left to the caller.
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
    invalidation contract: bump `model.version`, clear `model.matrix`,
    `model.rhs` and the cached assembly `pattern`, drop the Dirichlet
    projection cache, rebuild the integration plan, dof layout and the
    facet / surface / interface region caches, and refresh diagnostics.
    An outstanding `Solution` raises on stale reuse. Like `move!`, they
    address a level by position in the model's single space and
    therefore raise `ArgumentError` on a multi-domain (coupled) model.
  - **Query**: `active_cells(V::Space; level)` returns a copy of the
    level's `BitArray` (an all-true array for an unmasked level), and
    `active_cells(model; level, effective = true)` does the same on a
    prepared model.
  - **The pre-fold mask is the one a caller writes to.** On a space
    carrying a `PhysicalDomain` there are two masks per level: the one
    the caller asked for, on `model.prefold_space`, and the *effective*
    one the fictitious fold wrote over it. The mutators, `adapt`,
    `adapted` and `elevated` all derive from the pre-fold space, because
    a rebuild must re-decide the fold rather than inherit it. The model
    query defaults to `effective = true` and therefore does **not**
    round-trip: feeding its result back to `adapt` launders the fold into
    user intent, the cells the geometry switched off are recorded as
    cells the caller excluded, and the next fold cannot undo that. Read
    back with `effective = false` when the mask is going to be written
    again.
  - **Dof layer treatment**: a face between an active and an inactive
    cell of the same level is an artificial overlay constraint, analogous
    to a mesh-box-face constraint — unless every inactive cell across it
    was folded away as fictitious rather than masked by the user, in
    which case the modes stay free (`_internal_face_is_physical`). A
    fictitious fold covers; a user mask does not. Span-mode dofs of
    inactive cells are not enumerated.

All masking lives at cell granularity; sub-cell activation is out of
scope and would require a different abstraction.

### Per-cell polynomial order

The polynomial order of a level may vary from cell to cell, so p-refinement
is contained inside one mesh instead of needing a co-located overlay per
increment. It is a basis-family capability: integrated Legendre declares it
through `_supports_cell_order`, every other family (B-splines included)
refuses a non-uniform field loudly rather than ignoring it, because a
per-cell degree on a shared knot vector names no set of functions.

A family that refuses is not thereby stuck at one degree: p-refinement is
then done the multi-level way, by superposing a level of higher degree over
the region that needs it. For maximal-continuity splines that superposition
is linearly independent **by construction** and needs no deduplication — a
degree-`p`, `C^(p−1)` function supported inside the region would have to be
`C^p` to lie in the degree-`p+1` space there, hence a polynomial, hence zero
— so the two spaces intersect trivially. `decide` therefore withholds the
p-step on such a level and sends every marked cell to h, which is the
adaptive loop the isogeometric literature runs; `estimate` uses the same
superposition idea for its own enrichment.

  - **Construction-time order** via `space(...; order=…)` and
    `overlay(...; order=…)`. On top of the two uniform shapes (an integer,
    an `NTuple{D,Int}`) the keyword accepts an `AbstractArray{<:Integer,D}`
    or `AbstractArray{<:NTuple{D,Integer},D}` shaped like the level's cell
    grid, and a predicate `(cell_box, cell_index) -> order`. A field whose
    entries are all equal collapses back to the uniform representation, so
    the per-cell path costs a uniform level nothing.
  - **Rebuild**: `elevate(V::Space, level => order, …)` and
    `elevated(model, level => order, …)`, mirroring `adapt` / `adapted`.
    `elevate` additionally accepts an iterable of `CartesianIndex{D} => order`
    pairs, which leaves unnamed cells alone — the shape a marking loop
    produces. It is a separate verb from `adapt` because a mask spec and an
    order spec collide on `nothing` and on a predicate, and neither collision
    is detectable at dispatch. Compose them: `elevate(adapt(V, h), p)`.
  - **Query**: `cell_orders(V; level)` / `cell_orders(model; level)` return the
    per-axis order of every cell as a fresh array, all-equal on a uniform
    level.
  - **The minimum rule.** Where two cells of different order share a face,
    the shared entity carries the *minimum* of the two orders (the minimum
    rule of Szabó & Babuška 1991 and Demkowicz 2006; full citations in
    `src/basis.jl` and `src/dofs.jl`). Without it a mode generated by only
    the high-order side is nonzero on the shared face and the space is not
    C⁰ — and it still assembles, still solves, and still returns a plausible
    residual. The rule is a predicate on the dof *key*: the key's own
    multi-index must belong to the family's index set at the order of every
    **active** cell incident to it. That intersection is the classical
    componentwise-minimum statement, because the family's index-set filter
    (`_index_admissible`) is monotone non-decreasing in the order, and it is
    the better spelling because the minimum is then never formed and no
    question is ever asked at an order no cell carries. Three details are
    load-bearing: it must be keyed on the key (so both sides agree by
    construction), it must be a set membership rather than a per-axis mode
    comparison (they differ in 3D under `:trunk`), and inactive incident
    cells must be skipped rather than minimised over (a fictitious fold
    otherwise loses live dofs).
  - **One order field, two readers.** `Level.orders` is a `CellOrders` — a
    palette of the distinct per-axis orders plus one index per cell, so a
    uniform level is a palette of one and not a special case.
    `cell_order(level, cell)` is the question anything that integrates,
    evaluates or subdivides must ask, and every quadrature consumer does:
    the region rule, the moment-fit order, the facet rule and the VTK
    subdivision all size from the parent *cell*. `nominal_order(level)` is
    the per-axis maximum over the palette, and it is a sizing and default
    quantity — workspace banks and 1D factor buffers, where over-sizing is
    safe; `_surface_quadrature_order`, which sizes one table of rules for
    regions that no single parent cell owns and says so; and the order a
    new `overlay` or a re-instantiated basis family inherits. `Level.modes`
    is the derived minimum-rule table (`CellModes`), computed by `Level`'s
    inner constructor from the orders, the mesh, the mode and the mask, and
    never supplied by a caller.
  - **Order ≥ 1 per cell** is enforced per palette entry at construction, and
    `mode = :trunk` requires isotropy per cell. Neither can be expressed by
    the minimum rule: the dof key collapses the two endpoint modes into one
    node factor, so no filter over keys can see a missing endpoint mode.
  - **The acceptance test is the per-basis-function one-sided trace jump**,
    asserted `== 0` (`test/test_cell_order.jl`). An L² projection convergence
    rate does *not* discriminate — a non-conforming space is strictly larger
    and its projection error is marginally lower — and the solution's own jump
    norm decays under h-refinement even when the space is broken.

### Automated hp adaptivity

Where to refine can be left to the solver. Three verbs do it, the caller owns
the loop, and nothing in `src/adaptivity.jl` holds state between cycles.

  - **`estimate(model, solution; enrichment = 1)`** returns an `ErrorEstimate`:
    a per-level, per-cell indicator `cells`, a scalar `total`, a `reference`
    against which `total / reference` is a scale-free stopping quantity, and a
    `consistency` ratio — a free data-oscillation and solve check, *not* a
    saturation signal, and its docstring says why. `total / reference` is
    scale-free on one problem but is not comparable across different Dirichlet
    data, because a nonzero lift's own energy is never formed.
    The indicator is Bank–Weiser — build the order-elevated
    space `V⁺`, inject the solution into it, and read the residual in the
    directions `V` cannot represent, through the diagonal of the enriched
    operator:

    ```text
    η²_K = Σ_{j ∈ W(K)} R_j² / A⁺_jj ,   R = b⁺ − A⁺ u⁺ ,   W = V⁺ ⊖ V
    ```

    `W` is a set difference on dof *keys*, not a comparison of mode indices:
    the two disagree at an anisotropic order. Every complement mode is shared
    equally among the cells incident to it. The form must be **coercive**, since
    the diagonal is read as a mode's energy.

    **How `V⁺` is built is a basis-family question, and there are two answers.**
    A family carrying a per-cell order (`_supports_cell_order`) is enriched in
    place by raising every cell's order, which needs its dof keys to survive that
    increase naming the same functions. A family that does not — a B-spline
    level, whose degree belongs to a whole-axis knot vector — is enriched by
    **superposition**: a co-located level of the same family, degree and active
    region on a mesh refined by `enrichment + 1` per axis. Refining renames
    nothing, the two spline spaces nest because the factor is an integer, so
    `V⁺ ⊇ V` and the injection is exact. The twin's per-cell indicators are
    charged back to the cells of the level it enriches, so the estimate is always
    indexed by the caller's own levels. It is built with leaf semantics off,
    because a nested same-degree cover would otherwise deduplicate the very keys
    the injection looks for.

    > R. E. Bank, A. Weiser, *Some a posteriori error estimators for elliptic
    > partial differential equations*, Math. Comp. **44** (1985) 283–301.
    > [doi:10.1090/S0025-5718-1985-0777265-X](https://doi.org/10.1090/S0025-5718-1985-0777265-X).

  - **`refine(V, est; theta = 0.5, pmax = 8, previous = nothing)`** marks by
    Dörfler — the smallest set of cells carrying `theta` of the total squared
    indicator — and then sends each marked cell to h or to p. `theta` is the
    loop's only tunable. The two halves are public as `mark_cells` and
    `decide`, and `refine`'s loop form is their composition.

    > W. Dörfler, *A convergent adaptive algorithm for Poisson's equation*,
    > SIAM J. Numer. Anal. **33** (1996) 1106–1124.
    > [doi:10.1137/0733054](https://doi.org/10.1137/0733054).

    Two things the application step owes a family with wide support. The p-step
    is withheld on a level that carries no per-cell order, so every marked cell
    there takes h; and the h-step **support-extends** the cells it wakes, by the
    family's own `_support_radius`, because a region narrower than that radius
    carries no functions at all and a refinement that refines nothing makes a
    loop that never terminates. `_support_radius` is zero for integrated
    Legendre, so neither costs the default family anything.

  - **The h-versus-p decision is predicted error reduction**, not a smoothness
    indicator. Each cycle predicts what a cell's indicator ought to become if
    the solution there is as smooth as the step just taken assumed; the next
    cycle compares, and a cell that met its prediction takes p again while one
    that fell short takes h. `previous = (V_last, est_last)` is the only state
    the rule needs — a p-step shows up as a raised order and an h-step as a
    cell that was not active before — so nothing is stored between cycles.
    Omit `previous` and every marked cell takes p, which is right for the first
    cycle and wrong thereafter. The constants γ_p² = 0.4 and γ_h² = 4 are the
    literature's calibration and are deliberately not exposed.

    > J. M. Melenk, B. I. Wohlmuth, *On residual-based a posteriori error
    > estimation in hp-FEM*, Adv. Comput. Math. **15** (2001) 311–331.
    > [doi:10.1023/A:1014268310921](https://doi.org/10.1023/A:1014268310921).

    A coefficient-decay smoothness indicator was implemented as specified and
    removed: σ > 1 is an asymptotic statement about a spectral tail, and this
    loop reads it at orders 1–3 where it has nothing to say. `src/adaptivity.jl`'s
    header records that measurement, the comparison against deal.II's step-27,
    and why the two steps are kept independent rather than welded to depth.

  - **`coarsen(V; h, p, pmin = 1)`** is the inverse of both steps: it releases
    an h-step by deactivating the cover it woke, and a p-step by lowering the
    order, bounded below by `pmin`. There is deliberately no
    `coarsen(V, estimate)` form — which cells to release is not settled by the
    indicator — so the caller marks.

The loop's restrictions, all of them checked rather than assumed: a
single-domain model; every level on a family declaring `_supports_cell_order`,
because the decision and the enrichment both rest on a per-cell order; a
coercive form; an assembled model; `enrichment ≥ 1`; and no form keyed by
quadrature-point index, which the enriched twin would read on a different
cloud. On an immersed model the h-step has no state transfer — see *Moving
overlays and state transfer*.

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
    `physical_domain(geometry; lipschitz=Inf, alpha=0.0, keep_fictitious=false,
    subcell_length_scale, max_depth=8, moment_order_factor=2,
    target_residual=1e-6, cut_quadrature=nothing)`.

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
      the strict cut path; `> 0` enables α-FCM (every cut cell's rule
      becomes `(1 − α)·(physical rule) ∪ α·(full-cell tensor rule)`, and
      fully fictitious cells carry α-scaled weights). Independently, fully
      fictitious *cells* are dropped from the dof layout by default and
      retained only when `keep_fictitious = true` (which then requires
      `α > 0`). The α-scaled cut-cell enrichment applies regardless of
      `keep_fictitious`, so pre-`keep_fictitious` whole-cell results are
      not recovered by setting it.
    - `keep_fictitious` (default `false`): retain fully-fictitious cells
      in the dof layout with α-scaled full-cell quadrature (the classic
      α-FCM fill) instead of dropping them. Requires `alpha > 0`; pairing
      it with `alpha = 0` is rejected with an `ArgumentError`, because
      those cells would carry no quadrature at all.
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
      axis. `2` (default) spans degree `2p`, integrating trial × test
      products exactly and so matching on cut regions what tensor Gauss does
      on `:full` ones. `1` spans degree `p`, carrying only `∏(p+1)` points —
      the tensor Gauss count on an uncut cell, and QuESo's convention — at
      roughly the cube less fit time, but exact for a stiffness form only up
      to `p = 2` (the condition is `factor ≥ 2 − 2/p`). `physical_domain`'s
      docstring carries the measured order-1-to-7 comparison, where factor
      `1` stops converging after `p = 2`.
    - `target_residual`: target L² residual for the moment fit. The exact
      kernel reaches far below the `1e-6` default in a single NNLS solve,
      so this only bounds a small conditioning retry (a denser candidate
      cloud) — never moment accuracy or subdivision depth.
    - `cut_quadrature`: the cut-cell rule. `nothing` (default) is the
      non-negative moment fit; a callable
      `(physical, box, moment_order) -> (points, weights, residual, status)`
      replaces it on cut regions only, and those regions come out tagged
      `:cut_custom`. The rule must return *physical*-frame points and
      non-negative weights, gets α-blended downstream exactly as a fit
      does, and — the trap — silently contributes zero stiffness if it
      returns no points. The full contract is in the `PhysicalDomain`
      docstring under "Custom cut-cell quadrature".

  - **Attach** with `space(omega; ..., physical=physical_domain(…))`.
    The default `physical=nothing` keeps the no-FCM hot path.

  - **Imported geometry**: with `FileIO` and `MeshIO` loaded,
    `mesh_levelset(mesh)` turns a closed `BoundaryMesh` (a 2D segment loop
    or a 3D triangle surface) into a signed-distance leaf, and
    `stl_levelset("part.stl")` reads an STL into one. Both compose with the
    CSG combinators (see `ext/UnfittedMeshIOExt.jl`).

  - **Region quadrature kinds** (visible via `region.quadrature.kind`):
    `:full` (tensor Gauss), `:fictitious_alpha` (α-scaled tensor Gauss),
    `:cut_fitted` (NNLS moment-fit rule; under `α > 0` the rule is the
    concatenation `(1 − α)·moment-fit ∪ α·full-cell tensor Gauss`, from
    `∫_cell α(x) f = (1 − α)∫_Ω f + α∫_cell f`), `:cut_fallback` (the
    moment-fit residual exceeded the failure threshold, so the region
    carries the raw Saye volume rule the moments were summed from —
    correct and non-negative, but with 50–200× the points, and blended
    with the α-scaled tensor rule exactly as `:cut_fitted` is under
    `α > 0`), `:cut_failed` (strict-cut `α = 0` region whose `Ω ∩ box`
    carries no volume rule at all; region contributes zero quadrature),
    `:cut_alpha_failed` (the same under `α > 0`: the empty physical part
    is dropped but the α-scaled tensor rule is retained so the cell's
    dofs stay α-stabilised — a nonzero rule), `:cut_custom` (the rule came
    from the domain's `cut_quadrature` callable instead of the moment fit;
    counted as a cut region, never as a fit failure). The assembly hot loop
    is unchanged — it just iterates `zip(points, weights)`.

    **Grid-aligned boundary faces use the same vocabulary**, on
    `FacetRegion.kind`, because they run the same dispatch one dimension
    down: a codim-`K` facet is the affine slice `⋂ⱼ {x_{kⱼ} = vⱼ}`, so `Ω`
    on it is the CSG tree with every leaf restricted to that slice
    (`_restrict_domain`), and the region's own `(D − K)`-dimensional face box
    is classified and moment-fitted against it. `α` is honoured there too, so a
    trimmed face at `α > 0` carries the same `(1 − α)·fit ∪ α·tensor` blend a cut
    cell does — and therefore quadrature points outside `Ω`, which is why a datum
    or integrand defined only on `Ω` needs `α = 0`. Two deliberate differences.
    A zero-measure facet region is **kept**, with an empty rule, where a
    fictitious volume region is dropped — its parent cells still carry the
    trace dofs the dof layer constrains on the grid-aligned face test, which
    knows nothing about the level set. And `cut_quadrature` is **not**
    consulted on a facet: it is specified as a rule on a cell box in the
    model's own dimension, so a facet cut region always goes through
    `moment_fit_rule`. `:cut_custom` therefore never appears on a facet.

  - **Moment-fit defaults**: moment-fit basis order = `moment_order_factor
    × max(cell_order)` per axis over the region's parent cells, the factor
    defaulting to `1`; exact tensor Legendre moments from the Saye volume
    rule; a single Lawson–Hanson NNLS solve (`NonNegLeastSquares.jl`) selects
    ≤ `nbasis` non-negative weights from a candidate cloud drawn by even
    fractional spacing over the volume rule (an integer stride aliases
    against the fiber Gauss count and draws a degenerate cloud); up to 3
    attempts, retrying only with a denser cloud — a larger cloud spans a
    wider cone and so admits a non-negative exact solution the smaller one
    did not — never with more subdivision; `target_residual` bounds that
    retry and is **relative** to `‖m‖`, since the residual carries the units
    of a moment; if no attempt fits, the raw volume rule is used as a
    fallback rather than dropping the cell. Rules are cached by canonicalized
    region bounds + moment order.

  - **Diagnostics**: the FCM statistics on `diagnostics(model, solution)`
    are `cut_region_count`, `fit_failure_count`,
    `moment_fit_residual_max`, `cut_fallback_count`,
    `cut_fallback_points`, and `inactive_cell_counts` (which folds in the
    fictitious cell drop). The same report also carries
    `reduced_mode_counts` (covered-mode pruning), `dimension`,
    `integration_regions`, `facet_region_count`, `cut_facet_region_count`
    (how many of those facet regions have part of their own *face* outside
    `Ω`, hence carry a trimmed rule rather than an exact tensor product: a
    facet region is the whole grid-aligned face of its cells, and an integral
    over it — Dirichlet projection, a Neumann / Robin / Nitsche term tagged
    `on::BoundarySelector`, `boundary_integral` — runs over `face ∩ Ω` at the
    default `α = 0`, and over the α-blended face above it. The verdict is on
    the face and not on the parent cells, because a cut *cell*
    whose face lies wholly inside `Ω` needs no trimming; it is
    `FacetRegion.kind`, built from `classify_cell` on the region's own face
    box against the level set restricted to the facet's affine slice, so it
    carries the classifier's own resolution instead of the rule's: a
    fictitious sliver counts whenever `subcell_length_scale` and `max_depth`
    resolve it, and one finer than that budget is missed exactly as it is
    missed on the cell behind the face), `facet_fit_failure_count`,
    `facet_cut_fallback_count`, `facet_moment_fit_residual_max` (the facet
    analogues of the three cut-cell fit statistics, over the cut faces; the
    residual bounds the fit's compression error against the moments the
    quadrature kernel supplied and not the kernel's own error on them, and the
    condition for an exact facet rule is on the restricted level set — see
    `boundary`'s docstring), `min_relative_facet_measure` (the smallest ratio of
    a facet region's integrated measure to the full geometric measure of its own
    face — `1.0` when nothing is trimmed; `0.0` both for a face lying wholly
    outside `Ω` under strict `α = 0` and for a zero-measure `:cut_failed`
    tangency, so read it beside `facet_fit_failure_count`. A fraction of one
    face, hence not comparable with `min_relative_integration_volume`, which is
    a fraction of the whole domain. The conditioning warning for a trimmed
    Dirichlet condition: trimming is a small-cut generator for the trace mass
    exactly as a thin cut cell is one for the stiffness matrix.
    Reported and not stabilised, per "Small overlaps and conditioning" above),
    `dirichlet_trace_factors` (which branch the L² Dirichlet trace solve took on
    each component of each field — `:cholesky`, `:pseudoinverse`, or `:none`
    where the component had nothing to solve for — concatenated in
    field-declaration order),
    `unsupported_dirichlet_dof_count` / `unsupported_dirichlet_dofs` (the dofs a
    Dirichlet condition constrains with no measure anywhere on their facet
    support, which is the configuration trimming makes reachable: the constrained
    set is decided by the grid-aligned face test alone, so a dof whose every
    facet region is trimmed away is constrained with nothing left to fit it on
    and the pseudoinverse pins it to zero. Reported, not resolved — not
    constraining such a dof would move `active_unknowns` and the size of the
    system, and is a separate semantic decision),
    `surface_region_count`, `interface_region_count`, `raw_dofs`,
    `active_unknowns`, `levels`
    (one entry per level with `id`, `role`, `cells`, `order` — the nominal
    per-axis maximum — `order_palette`, the distinct per-cell orders,
    `mode`, `basis`, `domain`, `nested`, whether every higher level's
    nodes coincide with this level's where they overlap: the geometric half
    of what makes leaf semantics lossless, reported per level because
    `move!` can void it silently, and `raw_functions` / `active_functions`,
    what the level enumerated and what survived *every* elimination —
    artificial-boundary constraints, leaf pruning and physical Dirichlet data
    alike. Both, because the pair separates the two ways of contributing
    nothing: `raw = 0` is a dormant level, while `raw > 0` with `active = 0`
    is a level none of whose functions reach the system, which is the shape a
    spline overlay too thin to hold a support comes out as and one that no
    single number distinguishes),
    `small_overlap_count` / `small_overlaps`, `min_integration_volume`,
    `min_relative_integration_volume`, `symmetry_residual`,
    `condition_estimate`, `scaled_condition_estimate` (the same quantity for
    the diagonally scaled operator `D⁻¹ A D⁻¹`, which is what a Jacobi
    preconditioner sees — on an immersed system the unscaled number is
    dominated by the spread of the diagonal and the scaled one is where the
    basis families differ, measured at 43.8 against 2.2e10 at degree 3 on 16²
    with a disc removed), `solver`, `residual_norm`, and `l2_error`
    (`nothing` unless `exact=` is passed — the key is always present).
    `diagnostics(model)` alone returns the cached `AssemblyDiagnostics`
    record. A nonzero `cut_fallback_count` means those cells are
    under-resolved for their geometric complexity and should drive
    refinement; the fallback is a safety net, not a substitute for an
    adequate mesh.

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
    order. The `PhysicalDomain` and `physical_domain` docstrings in
    `src/physical.jl` are fully worked examples.

  - **Section dividers in long files.** When a file has clearly
    separable stages or concerns (`src/fcm.jl`'s NNLS / moment / fit
    pipeline; `precommit.jl`'s CLI / file discovery / formatting /
    code statistics / reporting / main) group them with single-line `─`
    dividers carrying the section name. The canonical form:

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
    field. Constrained dofs are not enumerated; their value lives in the
    per-field `DofLayout.constrained_values` matrix
    (`(raw, component) → value`), reachable from a `SystemLayout` as
    `layout.fields[i].dofs.constrained_values` and read through
    `constrained_value` / `dof_value`.

Tests enforce these conventions. Document any change to them at the top
of the relevant source file and update the relevant tests in the same
PR.

### Dependency policy

Use Julia standard libraries first: `LinearAlgebra`, `SparseArrays`
(with `SuiteSparse` reached implicitly through `\`), and `Test` /
`Random` in the test environment. Focused dependencies are acceptable
when they remove real work:

  - fixed-size small arrays for geometry and local kernels, e.g.
    `StaticArrays`;
  - Gaussian quadrature, e.g. `FastGaussQuadrature`;
  - iterative Krylov solvers or preconditioners through a narrow
    package;
  - non-negative constrained linear-least-squares solvers, e.g.
    `NonNegLeastSquares` (used by `src/fcm.jl` for the NNMF moment-fit
    solve);
  - forward-mode automatic differentiation for level-set gradients, e.g.
    `ForwardDiff` (used by `src/implicit.jl`; every CSG leaf must accept
    `ForwardDiff.Dual` arguments);
  - spatial search for scattered-data transfer, e.g. `NearestNeighbors`
    (the KD-tree behind `RBFP0` in `src/data.jl`);
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
  - Optional integrations belong in `[weakdeps]` + `[extensions]`, not in
    `[deps]`. The package currently ships three: `BasicBSpline` (B-spline
    basis family), `FileIO` + `MeshIO` + `GeometryBasics` (STL /
    boundary-mesh level sets), and `Tensors` (tensor-notation weak forms).
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
  - Regression examples based on the paper, required here by
    *capability* rather than by file name, so that reorganising
    `examples/` cannot quietly drop one: a one-dimensional problem
    carrying a discontinuous or under-resolved feature; a 2D singular
    corner problem; a small-overlap conditioning smoke test; and a
    moving heat-source smoke or reduced regression test. The last three
    are `reproductions/singular_square_2d`,
    `reproductions/conditioning_small_overlap` and
    `reproductions/traveling_laser_2d`, all pinned by the example
    smoke suite. **Requirement not yet met:** the one-dimensional slot
    is vacant. It used to be `bar_1d_unresolved_interface`, an elastic
    bar whose material interface fell inside a cell; that example was
    retired in the examples rebuild, and `tutorials/01_first_solve`,
    which inherited the 1D slot, solves a *smooth* manufactured sine on
    the interval — it exercises the 1D path end to end, and the smoke
    suite pins its error, but it carries no discontinuity and nothing
    the mesh fails to resolve. The nearest live coverage is split, and
    none of it is one-dimensional: `test_assembly.jl`'s "discontinuous
    coefficient need not align with overlay boundaries" for the
    discontinuity itself, `applications/interface_coupling_2d` for a
    jump in material data across a seam, and `tutorials/02_overlays`
    for a feature the base mesh cannot resolve. Restoring the
    requirement means adding a 1D discontinuous- or
    unresolved-feature run — most cheaply as a further section of
    `tutorials/01_first_solve`, or as a `reproductions/` entry if the
    paper's bar is wanted back verbatim.
  - Selective activation: dof-structure equivalence between a
    `LevelMask` and a geometrically smaller overlay covering the same
    active cells; artificial-boundary constraint on internal
    active/inactive interfaces; `activate!`/`deactivate!` invalidation
    contract. `dilate`'s box shape (one cell grows to `∏_d (2·by_d + 1)`),
    its per-axis widths, its monotonicity in the width, its clipping at a
    box face, its refusals, and its agreement with `support_extension` at
    the family's own radius.
  - Immersed boundary (`PhysicalDomain`): strict-α cell-level
    fictitious fold drops the right cells; the expected
    `:full` / `:cut_fitted` / `:fictitious_alpha` / `:cut_fallback` /
    `:cut_failed` / `:cut_alpha_failed` / `:cut_custom` region-kind tags
    appear; the candidate-cloud retry rescues a many-feature cut cell
    that the first attempt cannot fit, and the raw-volume-rule fallback
    still integrates a cell whose fit fails outright; α-FCM weight
    scaling on a fully fictitious cell; NNMF moment reproduction to
    within `target_residual`; end-to-end SPD on a cut-disk Poisson; 1D
    analytic mass-entry check on a cut cell; a custom `cut_quadrature`
    callable replaces the fit on cut regions only and is counted as a
    cut region, not as a fit failure. **Requirement not yet met:** no
    test drives a custom rule returning an `:empty` status, so its
    routing to `:cut_failed` / `:cut_alpha_failed` is unexercised; and
    because `_merged_boxes` emits pairwise-disjoint boxes, the
    `(box, moment order)` cache key never repeats within one plan, so
    no test distinguishes a memoised rule from an unmemoised one.
  - Multi-domain coupling: a machine-precision Nitsche interface patch
    test across two independently discretised subdomains; the `couple`
    four-block sign pattern. **Requirement not yet met:** no coupling
    test masks a subdomain, so the skip of interface regions where one
    side has no active cover is unexercised.
  - Covered-mode pruning: covered high-order modes eliminated and buried
    linear modes deduplicated only under a nested finer level;
    `reduced_mode_counts` matches the eliminated set. No test in
    `test_coverage_reduction.jl` builds a masked overlay; the fully
    deactivated case is pinned indirectly by test_activation.jl's
    "fully-deactivated overlay equals no overlay", which compares
    active-unknown counts against an overlay-free space; masked coverage
    under a ladder — split cover retained, single cover shed, lossless by
    rank against the unreduced twin — is pinned in `test_ladder.jl`
    ("covered-mode pruning on a ladder is lossless", with the rank witness).
  - Per-cell polynomial order: C⁰ conformity across an order jump, asserted
    as a per-basis-function one-sided trace jump `== 0`; the minimum rule's
    shape, and its set-membership form under `:trunk` where a per-axis
    comparison disagrees; inactive incident cells never lowering a shared
    entity; the `elevate` / `cell_orders` round trip as an identity; the
    intersection form reproducing the componentwise-minimum form cell for
    cell, and the index-set filter's monotonicity in the order; refusal of a
    non-uniform field — through `space` and through `elevate` — on a family
    that does not declare `_supports_cell_order` (`test_basis_bspline.jl`);
    and the graded-kernel golden digest (`test_graded_golden.jl`).
    **Requirement not yet met:** no example teaches the feature. `elevate`
    and `elevated` occur nowhere under `examples/`, no example passes an
    array or a predicate to `space(...; order=)`, and the tutorial sequence
    goes from `05_ladders` to `06_adaptive_hp`, which drives the automated
    loop without ever showing the manual verb it is built on. A reader who
    wants a graded space has this guide and the tests and no runnable
    demonstration.
  - Nested refinement ladders: `ladder` nests by construction; losslessness
    witnessed by `size(gram(V), 1) == rank(gram(V; prune = false))` with a
    non-nested negative control; `overlapping_cells` in both directions and
    under an anisotropic split; both forms of `adapt` — the `level => mask`
    pairs and the per-base-cell depth map, with grading; `adapted` keeping
    the source model intact and reusing the moment-fit cache.
  - Automated hp adaptivity: `estimate`'s effectivity bounded over a range of
    unknowns; its refusals — an unassembled model, `enrichment = 0`; its
    superposed enrichment on a family that carries no per-cell order, indexed
    back onto the caller's levels, and a whole loop driven on such a stack to a
    hundredfold reduction in both the error and the estimate; the h- and p-steps
    landing on the levels and cells they claim; the guards that bypass the
    h/p decision (no step available, not a leaf, already at `pmax`); a cell
    keeping p only when it met its prediction; `coarsen` as the exact inverse
    of both steps and its refusal on a covered cell; Dörfler marking monotone
    in θ with group completion; and the loop reducing both the error and the
    estimate, with one cycle of the full `previous`-threaded decision running
    unchanged in 1D, 2D and 3D.
  - Package extensions: the B-spline family's basis values, constraints
    and a small assembly; its refusal of a per-cell order through both
    `space` and `elevate`; the support-selection dof ledger — per axis
    `n + p` with both faces physical and `n − p` with both artificial, one
    eliminated function per artificial face for degree 1 and `p` for higher,
    and the closed form `n + p − 2(m + 1)` at a requested continuity `m`; the
    superposition being smooth and not merely each level, asserted as the
    decade ratio of the normal-derivative jump across an overlay face
    (10 at `:maximal`, below 1.5 at `C⁰` and for integrated Legendre); an
    arbitrary mask producing no linear-constraint pivot, which is what keeps
    the fast assembly path and makes the reconstruction defect unreachable;
    the dedup that keeps a nested B-spline stack non-singular, including the
    mixed-family direction where a degree-1 cover spans an
    integrated-Legendre level's hats, and the fold configurations where the
    cover's face lands before, inside and beyond the cut band; the MeshIO
    signed-distance leaf; the Tensors notation round-trip.
    **Requirement not yet met:** the fold configuration that support selection
    repairs is still singular under the *clamped* mechanism, which `bspline(;
    continuity = m)` for `m < p − 1` still selects. Where a clamped cover's box
    face falls on a knot of the level below inside a band of cut cells, it
    reproduces a *combination* of two buried functions without reproducing either,
    and a dedup keyed on single buried functions cannot see it: measured at 8 exact
    null modes of 104 unknowns, pinned `@test_broken` in `test_basis_bspline.jl`.
    The repair is a per-face rank repair — of the `p` buried functions straddling
    the face knot with the same perpendicular factor, keep the `m + 1` trace orders
    and strongly eliminate the rest — which changes what the dedup may eliminate
    rather than the burial test. Until it lands, use `continuity = :maximal` (the
    default, where the burial test is exact and the configuration is full rank),
    move the cover's face off the cut-cell band, or break the nesting.
  - Example smoke suite: every `examples/<tier>/<name>/<name>.jl`
    script runs to a clean exit at a coarse size in the default suite,
    with a finite, sane headline metric wherever the script prints one,
    banded at the value measured on the configuration the case actually
    runs. Where a tutorial's *prose* tells the reader what to look for
    in the output, that claim is asserted too — the overlay's
    order-of-magnitude improvement over the base level, the agreement
    between a reduced and an unreduced stack, the thirteen orders of
    magnitude of conditioning that one duplicated B-spline mode costs —
    so a tutorial whose text and output disagree fails here. The cases
    share one batched subprocess (`test/run_examples_child.jl`), each
    included into its own `Module`, because the per-process compilation
    floor dominated when every script paid it separately. Each case
    declares the packages it needs beyond `Unfitted` in a `requires`
    field, and a case the active project cannot resolve is skipped
    rather than failed: the three examples that need a weak dependency
    (`Tensors`; `BasicBSpline`; `FileIO` + `GeometryBasics` + `MeshIO`)
    run under `Pkg.test`, where all of those sit in the `test` target,
    and skip in a bare `--project=.` isolation run.
    **Requirement not yet met:** `applications/time_integration` needs
    `OrdinaryDiffEq`, which is deliberately absent from the package's
    dependencies, weak dependencies and test target — it is a
    hundred-package tree, and the example exists precisely to show that
    no extension is needed for it. That case is therefore skipped
    everywhere and has no automated coverage; it must be run by hand,
    or by a separate CI job, on the example's own project.

### Testing rules

  - Do not rely on plots for correctness.
  - Include analytic or manufactured-solution checks wherever possible.
  - Separate fast unit tests from expensive convergence studies.
  - Expensive studies belong in `examples/`, `benchmarks/`, or a
    clearly marked testset that is not required for every quick edit.
  - Pass constant-valued inputs as bare constants, not as closures, for
    the coefficient-shaped keywords that accept both — `source`,
    `coefficient`, `diffusion` and Dirichlet data, which all pass through
    `_as_coefficient`. Write `source = 1.0`, never `source = x -> 1.0`.
    Every distinct closure is a distinct *type*, and each one forces a
    full re-specialisation of the assembly emission kernel at roughly
    0.9 s of compile time, for
    an assembled result that is identical either way.
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
  4. Run
     `julia --project=. -e 'using Pkg; Pkg.test("Unfitted"; julia_args=["-O0"])'`
     and `julia precommit.jl --check`. Both must pass. The suite is
     compile-bound, so `-O0` roughly halves the wall clock without
     changing what is asserted; see "Verifying your setup".
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
