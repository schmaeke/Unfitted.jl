# Third-party attributions

Unfitted.jl includes software derived from upstream open-source projects.
Their original copyright notices and license terms are preserved below.

Ordinary Julia package dependencies fetched through the General registry
(listed in `Project.toml`) are not reproduced here — their licenses
travel with each installed package.

## Clean-room algorithm implementations

The following algorithms are implemented from their published descriptions
(clean-room — no third-party source code is vendored). They are cited here, and
in full at the top of the implementing file, for academic attribution:

  - **Implicit (level-set) quadrature** in `src/implicit.jl` —
    R. I. Saye, "High-order quadrature methods for implicitly defined surfaces
    and volumes in hyperrectangles", SIAM J. Sci. Comput. **37** (2015) A993,
    [doi:10.1137/140966290](https://doi.org/10.1137/140966290); and
    "High-order quadrature on multi-component domains…", J. Comput. Phys.
    **448** (2022) 110720,
    [doi:10.1016/j.jcp.2021.110720](https://doi.org/10.1016/j.jcp.2021.110720).
    Its use as a moment source follows B. Müller, F. Kummer, M. Oberlack,
    Int. J. Numer. Methods Engng. **96** (2013) 512,
    [doi:10.1002/nme.4569](https://doi.org/10.1002/nme.4569). The pair leaves
    that make polytope corners inside a cell exact specialise the 2022 paper's
    pairwise elimination of the height variable between level sets to leaves
    monotone along the height axis. Fiber roots are found by the Illinois
    variant of regula falsi: M. Dowell, P. Jarratt,
    "A modified regula falsi method for computing the root of an equation",
    BIT **11** (1971) 168,
    [doi:10.1007/BF01934364](https://doi.org/10.1007/BF01934364).

  - **Carathéodory pruning** in `src/fcm.jl` (`Caratheodory`) — the
    reduction of a positive measure to at most as many atoms as the moments
    it must keep, after C. Carathéodory, "Über den Variabilitätsbereich der
    Fourier'schen Konstanten von positiven harmonischen Funktionen",
    Rend. Circ. Mat. Palermo **32** (1911) 193,
    [doi:10.1007/BF03014795](https://doi.org/10.1007/BF03014795), made
    affordable by the recombination of C. Litterer, T. Lyons, "High order
    recombination and an application to cubature on Wiener space",
    Ann. Appl. Probab. **22** (2012) 1301,
    [doi:10.1214/11-AAP786](https://doi.org/10.1214/11-AAP786). Its null
    spaces come from Householder QR with column pivoting, after P. Businger,
    G. H. Golub, "Linear least squares solutions by Householder
    transformations", Numer. Math. **7** (1965) 269,
    [doi:10.1007/BF01436084](https://doi.org/10.1007/BF01436084).

  - **Polygon splitting** in `src/surface.jl` — the triangles of an immersed
    surface are split at grid planes by single-plane passes of the polygon
    clipper of I. E. Sutherland, G. W. Hodgman, "Reentrant polygon
    clipping", Commun. ACM **17** (1974) 32,
    [doi:10.1145/360767.360802](https://doi.org/10.1145/360767.360802). The
    same single pass, against the half-plane where a level set's linear
    interpolant is ≤ 0, clips the triangles of a plot's sample cells to `Ω`
    in `ext/UnfittedMakieExt.jl`.

  - **Mesh signed-distance** in `ext/UnfittedMeshIOExt.jl` — closest point on a
    triangle: C. Ericson, "Real-Time Collision Detection", Morgan Kaufmann
    (2005), ISBN 978-1-55860-732-3; angle-weighted pseudonormal sign:
    J. A. Bærentzen, H. Aanæs, IEEE Trans. Vis. Comput. Graph. **11** (2005)
    243, [doi:10.1109/TVCG.2005.49](https://doi.org/10.1109/TVCG.2005.49);
    generalized winding number: A. Jacobson, L. Kavan, O. Sorkine-Hornung,
    ACM Trans. Graph. **32** (2013) 33,
    [doi:10.1145/2461912.2461916](https://doi.org/10.1145/2461912.2461916).

## QuESo (Quadrature for Embedded Solids)

The `NonnegativeFit` compression in `src/fcm.jl` adopts the non-negative
moment-fit-via-NNLS approach to cut-cell quadrature that QuESo's
`QuadratureTrimmedElement` uses, and the package's FCM pipeline was
originally developed by porting that structure from QuESo. The current
implementation no longer contains QuESo-derived code: the distinctly
QuESo pieces — the octree stair-step moment integrator, the
`PointElimination` inner loop, and the accuracy-driven `AssembleIPs` outer
retry — have been removed, and the moment-fit-via-NNLS idea itself is the
older method of B. Müller, F. Kummer, M. Oberlack (Int. J. Numer. Methods
Engng. **96** (2013) 512, doi:10.1002/nme.4569). QuESo is acknowledged here
as the reference that informed the FCM pipeline design.

`NonnegativeFit` does run a bounded three-attempt loop of its own, which is
a different mechanism: each attempt draws a denser NNLS candidate cloud from
the same cut rule, because a subset of that rule need not admit an exact
non-negative fit of its moments. It never searches for points to eliminate
from a fitted rule, and never changes the cut rule or its moments.

The geometry and moments are computed differently from QuESo. QuESo
classifies bounding boxes against a triangulated B-rep and computes
moments from a B-rep divergence-theorem surface integral; Unfitted
classifies against a CSG level set (`PhysicalDomain` in
`src/physical.jl`) and computes the moments from Saye's
dimension-reduction implicit quadrature (`src/implicit.jl`; R.
I. Saye, SIAM J. Sci. Comput. **37** (2015) A993, doi:10.1137/140966290,
and J. Comput. Phys. **448** (2022) 110720, doi:10.1016/j.jcp.2021.110720)
— a clean-room implementation written from the cited papers and not
derived from any third-party source. The kernel gives exact,
octree-depth-independent moments on smooth (graph-like) cut cells,
machine precision for linear leaves and polytope corners.

### Upstream source

  - <https://github.com/manuelmessmer/QuESo>

### Reference

> M. Meßmer, T. Teschemacher, L. F. Leidinger, R. Wüchner,
> K.-U. Bletzinger, *Efficient CAD-integrated isogeometric analysis of
> trimmed solids*, Comput. Methods Appl. Mech. Engrg. **400** (2022)
> 115584,
> [doi:10.1016/j.cma.2022.115584](https://doi.org/10.1016/j.cma.2022.115584).

### License

QuESo is distributed under the BSD-4-Clause license:

```text
BSD 4-Clause License

Copyright (c) 2021, Manuel Messmer
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. All advertising materials mentioning features or use of this software
   must display the following acknowledgement: "This product includes
   software developed by Manuel Messmer."

4. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

This product includes software developed by Manuel Messmer.
