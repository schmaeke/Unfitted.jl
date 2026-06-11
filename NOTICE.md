# Third-party attributions

Unfitted.jl includes software derived from upstream open-source projects.
Their original copyright notices and license terms are preserved below.

Ordinary Julia package dependencies fetched through the General registry
(listed in `Project.toml`) are not reproduced here — their licenses
travel with each installed package.

## QuESo (Quadrature for Embedded Solids)

The finite-cell-method machinery in `src/fcm.jl` — specifically the
octree leaf walker, the moment-fit / NNLS pipeline, the point-
elimination inner loop, and the outer retry — is a Julia port of
QuESo's `QuadratureTrimmedElement`. The structural change is the
geometry kernel: QuESo classifies bounding boxes against a triangulated
B-rep, while Unfitted classifies against a user-supplied Lipschitz
level set (`PhysicalDomain` in `src/physical.jl`); QuESo's
surface-IP / divergence-theorem path is not ported.

As a direct consequence of that geometry-kernel swap, the moment
integration in this port is stair-step accurate (bounded by
`O(2^-subcell_depth)`) instead of B-rep-exact, so the moment-fit NNLS
residual cannot fall below the integrator's own floor. The default
`PhysicalDomain.target_residual` is therefore `1e-6` (matched to the
default `subcell_depth=4`), looser than QuESo's hardcoded `1e-10` and
their shipped examples' typical `1e-8`. Users who raise
`subcell_depth` should tighten `target_residual` accordingly.

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
