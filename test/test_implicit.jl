using StaticArrays
using LinearAlgebra

# Integrate `f` against a kernel rule `(points, weights)`.
_quad(f, rule) = sum(rule[2][i] * f(rule[1][i]) for i in eachindex(rule[2]); init=0.0)

const _IVQ = Unfitted.implicit_volume_quadrature

@testset "implicit quadrature — 1D linear cut moments are exact" begin
    # Ω = {x ≤ 0.7} on [0, 1]. Monomial moments ∫₀^0.7 xⁿ dx = 0.7ⁿ⁺¹/(n+1) are reproduced to machine precision: a
    # linear interface is graph-like, so the Saye rule is polynomial-exact.
    a = 0.7
    rule = _IVQ(x -> x[1] - a, box((0.0,), (1.0,)); gauss_points=6)
    for n in 0:5
        @test _quad(x -> x[1]^n, rule) ≈ a^(n + 1) / (n + 1) atol = 1e-13
    end
end

@testset "implicit quadrature — a fiber root is bracketed by signs, not by a product" begin
    # Scaling a leaf by a positive constant does not move Ω = {φ ≤ 0}. A scan that detects a crossing by `f(a)·f(b) < 0`
    # loses the root once the two samples are small: at a scale of 1e-170 the product, about 1e-342, underflows to zero,
    # no breakpoint is placed, and the whole fiber [0, 1] is classified by its midpoint — outside Ω, so the rule comes
    # back empty instead of integrating [0, 0.3].
    rule = _IVQ(x -> 1e-170 * (x[1] - 0.3), box((0.0,), (1.0,)); gauss_points=2)
    @test _quad(x -> 1.0, rule) ≈ 0.3 atol = 1e-15
    @test _quad(x -> x[1], rule) ≈ 0.3^2 / 2 atol = 1e-15
end

@testset "implicit quadrature — the fiber root finder keeps bisection's guarantees in fewer steps" begin
    # The Illinois regula falsi `_root` lands on the floating-point root — within two ulps, the rounding of f near it —
    # on every bracket. A simple root takes a handful of evaluations where bisection spends about 50 to reach adjacent
    # floats; at a triple root, where regula falsi of any kind converges only linearly, the bisection safeguard still
    # gets there, within three times bisection's count.
    for (f, r, simple) in ((t -> t^3 - 2, cbrt(2.0), true), (t -> exp(t) - 3, log(3.0), true),
                           (t -> sqrt((t - 0.31)^2 + 0.2^2) - 0.25, 0.46, true), (t -> 0.7 - t, 0.7, true),
                           (t -> (t - 0.3)^3, 0.3, false))
        a, b = r - 0.21, r + 0.17
        evaluations = Ref(0)
        x = Unfitted._root(t -> (evaluations[] += 1; f(t)), a, b, f(a), f(b))
        @test abs(x - r) <= 2 * eps(r)
        @test evaluations[] <= (simple ? 12 : 3 * 52)
    end
end

@testset "implicit quadrature — 2D axis-aligned cut moments are exact" begin
    # Ω = {x ≤ a} on [0, 1]², independent of y. ∫∫ xᵐ yⁿ = a^(m+1)/(m+1) · 1/(n+1).
    a = 0.4
    rule = _IVQ(x -> x[1] - a, box((0.0, 0.0), (1.0, 1.0)); gauss_points=8)
    for m in 0:4, n in 0:4
        exact = a^(m + 1) / (m + 1) * 1 / (n + 1)
        @test _quad(x -> x[1]^m * x[2]^n, rule) ≈ exact atol = 1e-12
    end
end

@testset "implicit quadrature — 2D tilted plane is exact" begin
    # Ω = {x + y ≤ 1}: the unit triangle. Area 1/2, ∫xy = 1/24, centroid moments.
    rule = _IVQ(x -> x[1] + x[2] - 1, box((0.0, 0.0), (1.0, 1.0)); gauss_points=8)
    @test _quad(x -> 1.0, rule) ≈ 0.5 atol = 1e-13
    @test _quad(x -> x[1], rule) ≈ 1 / 6 atol = 1e-13
    @test _quad(x -> x[1] * x[2], rule) ≈ 1 / 24 atol = 1e-13
    @test _quad(x -> x[1]^2, rule) ≈ 1 / 12 atol = 1e-13
end

@testset "implicit quadrature — base partition resolves clamped fibers" begin
    # Ω = {x ≤ 0.3 + y} on [0, 1]². For y > 0.7 the root leaves the fiber and the inside length saturates at 1, a kink
    # in the projected integrand. The base recursion partitions at the face restriction {x = 1} (i.e. y = 0.7), so the
    # moments stay machine-exact. Analytic:
    #   area  = ∫₀^0.7 (0.3+y) dy + ∫_0.7^1 1 dy            = 0.755
    #   ∫x²   = ∫₀^0.7 (0.3+y)³/3 dy + ∫_0.7^1 1/3 dy        = 0.1826583…
    rule = _IVQ(x -> x[1] - (0.3 + x[2]), box((0.0, 0.0), (1.0, 1.0)); gauss_points=10)
    area = 0.755
    ix2 = (1 / 12) * (1.0^4 - 0.3^4) + 0.3 * (1 / 3)
    @test _quad(x -> 1.0, rule) ≈ area atol = 1e-12
    @test _quad(x -> x[1]^2, rule) ≈ ix2 atol = 1e-12
end

@testset "implicit quadrature — 3D linear cut moments are exact" begin
    # Ω = {x ≤ a} on [0, 1]³, and the tilted simplex {x+y+z ≤ 1} (volume 1/6).
    a = 0.55
    rule = _IVQ(x -> x[1] - a, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); gauss_points=8)
    for m in 0:3, n in 0:2, p in 0:2
        exact = a^(m + 1) / (m + 1) * 1 / (n + 1) * 1 / (p + 1)
        @test _quad(x -> x[1]^m * x[2]^n * x[3]^p, rule) ≈ exact atol = 1e-12
    end
    simplex = _IVQ(x -> x[1] + x[2] + x[3] - 1, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); gauss_points=8)
    @test _quad(x -> 1.0, simplex) ≈ 1 / 6 atol = 1e-12
end

@testset "implicit quadrature — 4D linear cut volume (dimension-generic smoke)" begin
    # The recursion is written for arbitrary D; a 4D half-space cut exercises the D → D−1 reduction three times. Ω = {x₁
    # ≤ a} on [0, 1]⁴ ⇒ volume a.
    a = 0.6
    lo = SVector(0.0, 0.0, 0.0, 0.0)
    hi = SVector(1.0, 1.0, 1.0, 1.0)
    rule = _IVQ(x -> x[1] - a, AxisBox(lo, hi); gauss_points=4)
    @test _quad(x -> 1.0, rule) ≈ a atol = 1e-12
    @test _quad(x -> x[2], rule) ≈ a * 0.5 atol = 1e-12
end

@testset "implicit quadrature — tensor Legendre moment vector matches analytic" begin
    # The moment-fit pipeline needs mₐ = ∫_{Ω∩R} ψₐ dx with ψₐ the tensor Legendre basis. Build that vector from the
    # kernel rule and the existing Legendre evaluators in `fcm.jl`, then check the constant moment equals the cut area
    # and the orthogonality-driven zeros vanish, for Ω = {x ≤ 0.4}.
    region = box((0.0, 0.0), (1.0, 1.0))
    order = (3, 3)
    a = 0.4
    rule = _IVQ(x -> x[1] - a, region; gauss_points=8)
    idx = Unfitted._moment_basis_indices(order)
    factors = Unfitted._legendre_factor_buffers(order, Float64)
    moments = zeros(length(idx))
    for j in eachindex(rule[2])
        Unfitted._fill_legendre_factors!(factors, rule[1][j], region.lower, region.upper, order)
        for (k, α) in pairs(idx)
            moments[k] += rule[2][j] * Unfitted._tensor_legendre_value(factors, α)
        end
    end
    # α = (0,0): L₀⊗L₀ ≡ 1 ⇒ moment is the cut area a.
    @test moments[1] ≈ a atol = 1e-12
    # α = (0,n) for n ≥ 1: ∫₀^a dx · ∫₀¹ Lₙ(2y−1) dy = a · 0 by Legendre orthogonality to the constant.
    for n in 1:3
        k = findfirst(==(CartesianIndex(0, n)), idx)
        @test abs(moments[k]) < 1e-12
    end
end

@testset "implicit quadrature — curved interface is high-order (clean cell)" begin
    # A rim cut cell with no turning point: the arc is a single-valued graph in the max-gradient axis, so the rule is
    # spectrally accurate. Self-converge: raising the order changes the area only at round-off scale.
    phi = x -> sqrt(x[1]^2 + x[2]^2) - 1.0
    b = box((0.5, 0.2), (0.9, 0.6))
    a8 = _quad(x -> 1.0, _IVQ(phi, b; gauss_points=8))
    a16 = _quad(x -> 1.0, _IVQ(phi, b; gauss_points=16))
    @test isapprox(a8, a16; atol=1e-10)
end

@testset "implicit quadrature — disk area and sphere volume converge" begin
    # Whole disk / ball in one box (off-grid centre so subdivision isolates the interior gradient-null point). The four
    # (six) cardinal turning points cap accuracy at the Gauss order, but the area/volume is still good to ~1e-4 and
    # improves with the order.
    disk = x -> sqrt((x[1] - 0.53)^2 + (x[2] - 0.47)^2) - 0.3
    a10 = _quad(x -> 1.0, _IVQ(disk, box((0.0, 0.0), (1.0, 1.0)); gauss_points=10, max_subdiv=6))
    @test isapprox(a10, π * 0.3^2; atol=1e-4)

    ball = x -> sqrt((x[1] - 0.53)^2 + (x[2] - 0.47)^2 + (x[3] - 0.51)^2) - 0.3
    v8 = _quad(x -> 1.0, _IVQ(ball, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); gauss_points=8, max_subdiv=6))
    v12 = _quad(x -> 1.0, _IVQ(ball, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); gauss_points=12, max_subdiv=6))
    truth = 4 / 3 * π * 0.3^3
    @test isapprox(v8, truth; atol=1e-4)
    @test abs(v12 - truth) < abs(v8 - truth)   # higher order ⇒ smaller error
end

@testset "implicit quadrature — empty and full regions" begin
    # Ω = ∅ (φ ≡ 1) ⇒ no points. Ω = everything (φ ≡ −1) ⇒ full tensor rule whose weights sum to the box volume.
    empty = _IVQ(x -> 1.0, box((0.0, 0.0), (1.0, 1.0)); gauss_points=4)
    @test isempty(empty[1])
    full = _IVQ(x -> -1.0, box((0.0, 0.0), (2.0, 1.0)); gauss_points=4)
    @test sum(full[2]) ≈ 2.0 atol = 1e-13
end

# Build a kernel volume rule from a CSG level-set tree (leaves + membership). `certified = true` forwards the tree's
# per-leaf Lipschitz constants, which is what `moment_fit_rule` does in the finite-cell pipeline.
function _csg_rule(geom, region; gp, ms=5, certified=false)
    leaves = Unfitted._leaves(geom)
    lipschitz = certified ? Float64[l.lipschitz for l in leaves] : nothing
    _IVQ(Any[l.f for l in leaves], x -> Unfitted._inside(geom, x), region; gauss_points=gp, max_subdiv=ms,
         lipschitz=lipschitz)
end

@testset "implicit quadrature — force-reduce stays correct without subdivision" begin
    # A whole disk centred in the box has an interior point where ∇φ = 0 and is tangent to every axis somewhere, so no
    # axis is graph-like. With no subdivision budget the kernel force-reduces on the least-bad axis: the multi-root
    # fiber scan still yields the correct region (only the order degrades), so the area is right to a loose tolerance
    # and never throws.
    disk = x -> sqrt((x[1] - 0.53)^2 + (x[2] - 0.47)^2) - 0.3
    a0 = _quad(x -> 1.0, _IVQ(disk, box((0.0, 0.0), (1.0, 1.0)); gauss_points=6, max_subdiv=0))
    @test isfinite(a0) && a0 > 0
    @test isapprox(a0, π * 0.3^2; atol=5e-2)          # correct region, low order
    a2 = _quad(x -> 1.0, _IVQ(disk, box((0.0, 0.0), (1.0, 1.0)); gauss_points=6, max_subdiv=2))
    @test isapprox(a2, π * 0.3^2; atol=1e-3)           # subdivision recovers accuracy
end

@testset "implicit quadrature — CSG intersection (polytope corner) is exact" begin
    # Ω = {x ≤ a} ∩ {y ≤ b} is the box [0,a]×[0,b]. The perpendicular half-plane is FLAT along the reduction axis, so
    # the rule is polynomial-exact.
    a, b = 0.7, 0.6
    geom = intersect(leaf(x -> x[1] - a), leaf(x -> x[2] - b))
    rule = _csg_rule(geom, box((0.0, 0.0), (1.0, 1.0)); gp=6)
    @test _quad(x -> 1.0, rule) ≈ a * b atol = 1e-13
    @test _quad(x -> x[1] * x[2], rule) ≈ (a^2 / 2) * (b^2 / 2) atol = 1e-13
    @test _quad(x -> x[1]^2 * x[2]^2, rule) ≈ (a^3 / 3) * (b^3 / 3) atol = 1e-13
end

@testset "implicit quadrature — CSG union and difference" begin
    # Union of two overlapping half-strips: {x ≤ 0.3} ∪ {x ≥ 0.7} on [0,1]², area 0.6, exact (each leaf is graph-like in
    # x, the base is partitioned by both faces). Difference: {x ≤ 0.8} \ {x ≤ 0.3} = strip [0.3,0.8], area 0.5.
    uni = union(leaf(x -> x[1] - 0.3), leaf(x -> 0.7 - x[1]))
    @test _quad(x -> 1.0, _csg_rule(uni, box((0.0, 0.0), (1.0, 1.0)); gp=6)) ≈ 0.6 atol = 1e-12
    diff = setdiff(leaf(x -> x[1] - 0.8), leaf(x -> x[1] - 0.3))
    @test _quad(x -> 1.0, _csg_rule(diff, box((0.0, 0.0), (1.0, 1.0)); gp=6)) ≈ 0.5 atol = 1e-12
end

@testset "implicit quadrature — CSG annulus is high-order" begin
    # Annulus as a disk minus a disk (no fake mid-radius crease): quarter in [0,1]² with both circle boundaries smooth ⇒
    # high accuracy.
    ann = setdiff(leaf(x -> hypot(x...) - 0.45), leaf(x -> hypot(x...) - 0.2))
    a = _quad(x -> 1.0, _csg_rule(ann, box((0.0, 0.0), (1.0, 1.0)); gp=14, ms=6))
    @test isapprox(a, π * (0.45^2 - 0.2^2) / 4; atol=1e-4)
end

# ∫ g over the tetrahedron abcd by a collapsed (Duffy) Gauss rule of n³ points, exact to degree 2n − 3 in g.
function _tetrahedron_integral(g, a, b, c, d; n=8)
    ξ, ω = Unfitted.gausslegendre(n)
    total = 0.0
    for i in 1:n, j in 1:n, k in 1:n
        s, t, r = (ξ[i] + 1) / 2, (ξ[j] + 1) / 2, (ξ[k] + 1) / 2
        x = a + s * (b - a) + t * (1 - s) * (c - a) + r * (1 - s) * (1 - t) * (d - a)
        total += ω[i] * ω[j] * ω[k] / 8 * (1 - s)^2 * (1 - t) * g(x)
    end
    return total * abs(dot(b - a, cross(c - a, d - a)))
end

@testset "implicit quadrature — a tetrahedron's interior corners are exact in 3D" begin
    # Four oblique planes cut a tetrahedron whose vertices lie strictly inside cells of a 4³ grid, so every vertex and
    # edge is a crossing of two or three leaves inside a cut cell. Pair leaves put each crossing into the base problem
    # (a pair of pairs, one reduction further down, for a vertex), so the cell rules sum to the exact volume and moments
    # to round-off at every fiber order. Without them the volume was off by 2.3e-5 at q = 3 and 1.6e-4 at q = 5.
    v = (SVector(0.13, 0.11, 0.17), SVector(0.88, 0.21, 0.12), SVector(0.31, 0.87, 0.23), SVector(0.38, 0.34, 0.91))
    planes = map(((1, 2, 3, 4), (1, 2, 4, 3), (1, 3, 4, 2), (2, 3, 4, 1))) do (a, b, c, o)
        n = normalize(cross(v[b] - v[a], v[c] - v[a]))
        n = dot(n, v[o] - v[a]) > 0 ? -n : n                       # outward: away from the fourth vertex
        return x -> dot(n, x - v[a])
    end
    membership = x -> all(f -> f(x) <= 0, planes)
    gs = (x -> 1.0, x -> x[1], x -> x[2]^2, x -> x[1] * x[2] * x[3])
    exact = [_tetrahedron_integral(g, v...) for g in gs]
    for q in (3, 5)
        total = zeros(length(gs))
        for I in CartesianIndices((4, 4, 4))
            region = box((Tuple(I) .- 1) ./ 4, Tuple(I) ./ 4)
            rule = _IVQ(Any[planes...], membership, region; gauss_points=q, lipschitz=ones(4))
            total .+= [_quad(g, rule) for g in gs]
        end
        @test all(isapprox.(total, exact; rtol=1e-13))
    end
end

@testset "implicit quadrature — finite-difference gradient fallback" begin
    # For level sets that are not AD-differentiable the caller can pass a finite-difference gradient; on a smooth linear
    # cut it reproduces the ForwardDiff result to integration accuracy.
    a = 0.45
    ad = _IVQ(x -> x[1] - a, box((0.0, 0.0), (1.0, 1.0)); gauss_points=6)
    fd = _IVQ(x -> x[1] - a, box((0.0, 0.0), (1.0, 1.0)); gauss_points=6, grad=Unfitted._fd_gradient)
    @test _quad(x -> 1.0, fd) ≈ _quad(x -> 1.0, ad) atol = 1e-10
    @test _quad(x -> x[1] * x[2], fd) ≈ _quad(x -> x[1] * x[2], ad) atol = 1e-10
end

@testset "implicit quadrature — Float32 carries through" begin
    # The kernel is scalar-type generic; a Float32 box returns a Float32 rule.
    rule = _IVQ(x -> x[1] - 0.5f0, box((0.0f0, 0.0f0), (1.0f0, 1.0f0)); gauss_points=5)
    @test eltype(rule[2]) === Float32
    @test _quad(x -> 1.0f0, rule) ≈ 0.5f0 atol = 1.0f-5
end

# Signed distance to the disc of radius `r` centred at `c` (Lipschitz constant 1) and its negation, so `{_disc ≤ 0}` is
# the disc and `{_hole ≤ 0}` is everything outside it.
_disc(c, r) = x -> hypot(x[1] - c[1], x[2] - c[2]) - r
_hole(c, r) = x -> r - hypot(x[1] - c[1], x[2] - c[2])

@testset "implicit quadrature — a sub-region feature needs the Lipschitz certificate" begin
    # A disc of radius 0.16 centred at (0.25, 0.25) lies strictly inside [0, 1]² yet contains neither the box centre nor
    # any of its four corners. Every sample therefore reports the same sign, `_sample_sign` prunes the only leaf, and
    # with no leaf left the kernel takes the box to be uniform and returns the full tensor rule: the hole is silently
    # integrated as solid. The certificate |f(c)| > L·r bounds f over the *whole* box, so it never prunes a leaf that
    # varies on it and the hole reappears.
    region = box((0.0, 0.0), (1.0, 1.0))
    hole = _hole((0.25, 0.25), 0.16)
    @test _quad(x -> 1.0, _IVQ(hole, region; gauss_points=12)) ≈ 1.0 atol = 1e-13
    certified = _IVQ(hole, region; gauss_points=12, lipschitz=1.0)
    @test _quad(x -> 1.0, certified) ≈ 1 - π * 0.16^2 atol = 5e-5
    @test all(>(0), certified[2])
end

@testset "implicit quadrature — a feature entering the region through an edge" begin
    # The blind spot is not confined to features enclosed by the box: the disc of radius 0.25 centred at (−0.05, 0.5)
    # overlaps [0, 1]² through the x = 0 edge only, and its circular segment still misses the centre and all corners.
    # ∂f/∂x keeps one sign over the region, so the segment is graph-like in x and the certified rule is high-order.
    # Segment area with d = 0.05:
    #   A = r² arccos(d / r) − d √(r² − d²)
    region = box((0.0, 0.0), (1.0, 1.0))
    seg = _hole((-0.05, 0.5), 0.25)
    area = 0.25^2 * acos(0.05 / 0.25) - 0.05 * sqrt(0.25^2 - 0.05^2)
    @test _quad(x -> 1.0, _IVQ(seg, region; gauss_points=12)) ≈ 1.0 atol = 1e-13
    @test _quad(x -> 1.0, _IVQ(seg, region; gauss_points=12, lipschitz=1.0)) ≈ 1 - area atol = 1e-5
end

@testset "implicit quadrature — 3D sub-region feature (dimension-generic)" begin
    # The same failure in 3D: a ball of radius 0.15 at (0.3, 0.3, 0.3) misses the centre and all eight corners of
    # [0, 1]³. The recursion bisects until the ball is graph-like on the children, so the certified volume is far
    # sharper here than the 2D disc's.
    region = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ball = x -> 0.15 - sqrt((x[1] - 0.3)^2 + (x[2] - 0.3)^2 + (x[3] - 0.3)^2)
    @test _quad(x -> 1.0, _IVQ(ball, region; gauss_points=8)) ≈ 1.0 atol = 1e-13
    @test _quad(x -> 1.0, _IVQ(ball, region; gauss_points=8, lipschitz=1.0)) ≈ 1 - 4 / 3 * π * 0.15^3 atol = 1e-7
end

@testset "implicit quadrature — per-leaf Lipschitz constants on a CSG tree" begin
    # The multi-component form takes one constant per leaf in leaf order. Two non-overlapping discs, each individually
    # invisible to sampling on this box, are both recovered; a length mismatch is an error rather than a silently
    # truncated pairing.
    region = box((0.0, 0.0), (1.0, 1.0))
    geom = complement(union(leaf(_disc((0.25, 0.25), 0.12); lipschitz=1.0),
                            leaf(_disc((0.75, 0.72), 0.1); lipschitz=1.0)))
    @test _quad(x -> 1.0, _csg_rule(geom, region; gp=12)) ≈ 1.0 atol = 1e-13
    @test _quad(x -> 1.0, _csg_rule(geom, region; gp=12, certified=true)) ≈ 1 - π * (0.12^2 + 0.1^2) atol = 5e-5

    fs = Any[l.f for l in Unfitted._leaves(geom)]
    membership = x -> Unfitted._inside(geom, x)
    @test_throws DimensionMismatch _IVQ(fs, membership, region; gauss_points=6, lipschitz=[1.0])
    @test_throws ArgumentError _IVQ(fs[1], region; gauss_points=6, lipschitz=0.0)
end

@testset "implicit quadrature — the `lipschitz` default preserves the sampling path" begin
    # Backwards compatibility. Omitting the keyword, passing `nothing`, and passing `Inf` (the `leaf` default,
    # documented as the uncertified mode) must all reproduce the pre-certificate rule bit for bit — blind spot included
    # — so no existing caller changes behavior.
    region = box((0.0, 0.0), (1.0, 1.0))
    hole = _hole((0.25, 0.25), 0.16)
    plain = _IVQ(hole, region; gauss_points=10)
    @test _IVQ(hole, region; gauss_points=10, lipschitz=nothing) == plain
    @test _IVQ(hole, region; gauss_points=10, lipschitz=Inf) == plain

    # And where sampling already resolves the cut, the certificate changes nothing: it can only ever keep *more* leaves
    # active, never fewer.
    tilted = x -> x[1] + x[2] - 1
    @test _quad(x -> 1.0, _IVQ(tilted, region; gauss_points=8, lipschitz=sqrt(2))) ≈
          _quad(x -> 1.0, _IVQ(tilted, region; gauss_points=8)) atol = 1e-15
end

@testset "implicit quadrature — crossings of a 2D CSG domain inside cut cells" begin
    # Two overlapping discs minus a rotated square: the two circles cross, and the square's four corners lie, strictly
    # inside cells of an 8² grid. Summed cell by cell, the area matches the analytic value to 1e-12 (5.7e-14 measured;
    # 3.0e-5 before the crossings became base breakpoints).
    c1, r1, c2, r2 = SVector(0.4, 0.45), 0.3, SVector(0.65, 0.55), 0.25
    θ, s = π / 6, 0.08
    corners = [SVector(0.5, 0.5) + s * SVector(cos(θ) * a - sin(θ) * b, sin(θ) * a + cos(θ) * b)
               for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    sides = map(1:4) do i
        t = corners[mod1(i + 1, 4)] - corners[i]
        n = SVector(t[2], -t[1]) / norm(t)
        return leaf(x -> dot(n, x - corners[i]); lipschitz=1.0)
    end
    geom = setdiff(union(leaf(_disc(c1, r1); lipschitz=1.0), leaf(_disc(c2, r2); lipschitz=1.0)), intersect(sides...))
    d = norm(c2 - c1)
    lens = r1^2 * acos((d^2 + r1^2 - r2^2) / (2d * r1)) + r2^2 * acos((d^2 + r2^2 - r1^2) / (2d * r2)) -
           sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)) / 2
    area = π * (r1^2 + r2^2) - lens - (2s)^2
    total = sum(_quad(x -> 1.0, _csg_rule(geom, box((Tuple(I) .- 1) ./ 8, Tuple(I) ./ 8); gp=8, ms=4, certified=true))
                for I in CartesianIndices((8, 8)))
    @test total ≈ area rtol = 1e-12
end

@testset "implicit quadrature — the height-axis choice is inferred" begin
    # `_choose_axis` samples `Any`-typed leaves, so the call into its classification is dynamic, and its result type is
    # asserted for `_implicit_quad`'s sake: without that the chosen axis was inferred as `Any`, and every base node's
    # fiber emission was dispatched dynamically and boxed its arguments.
    ctx = Unfitted._quad_ctx(Float64, 4, nothing, 4, 1.0)
    region = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    @test (@inferred Unfitted._choose_axis(Any[x -> x[1] + 2x[2] - 1], region, ctx)) == (2, true, [:monotone])
end

@testset "implicit quadrature — a leaf named twice in a CSG tree is integrated once" begin
    # Ω = a ∩ (b ∪ c) spelled (a ∩ b) ∪ (a ∩ c) lists the sphere `a` twice. The kernel drops the repeat, so both
    # spellings give the same rule. Two copies of one leaf got a pair leaf that vanishes identically and scattered
    # breakpoints over its round-off: 2.6× the points in 2D (8² cells, q = 6) and 7.4× in 3D (4³, q = 5), where pairs of
    # such pairs follow.
    for D in (2, 3)
        c0 = SVector(ntuple(_ -> 0.5, D))
        a = leaf(x -> norm(x - c0) - 0.37; lipschitz=1.0)
        nb, nc = normalize(SVector(ntuple(i -> 0.3 + 0.2i, D))), normalize(SVector(ntuple(i -> 0.9 - 0.25i, D)))
        b = leaf(x -> dot(nb, x - c0) - 0.05; lipschitz=1.0)
        c = leaf(x -> dot(nc, x - c0) + 0.02; lipschitz=1.0)
        n = D == 2 ? 8 : 2
        rules(geom) = [_csg_rule(geom, box((Tuple(I) .- 1) ./ n, Tuple(I) ./ n); gp=5, ms=4, certified=true)
                       for I in CartesianIndices(ntuple(_ -> n, D))]
        @test rules(union(intersect(a, b), intersect(a, c))) == rules(intersect(a, union(b, c)))
    end
end

@testset "implicit quadrature — leaves are evaluated inside the region only" begin
    # A level set may be defined on the mesh domain alone — an image interpolant without extrapolation, a lookup into a
    # grid. Every evaluation the kernel makes stays in the closed region, a pair leaf's included, so such leaves work in
    # a corner cell of the domain where two planes cross, through the kernel and the finite-cell fit alike, and give the
    # rule of the same leaves defined everywhere. (Pair leaves evaluated their leaves at a secant root beyond the fiber
    # and at finite-difference steps beyond the region's corners, and threw here.)
    inside(x) = all(c -> 0 <= c <= 1, x)
    bounded(f) = x -> (inside(x) || throw(DomainError(x, "outside the mesh domain")); f(x))
    p1 = x -> (x[1] - 0.12) + 0.5 * (x[2] - 0.12)
    p2 = x -> -(x[1] - 0.12) + 2.0 * (x[2] - 0.12)
    corner = box((0.0, 0.0), (0.25, 0.25))
    rule(f1, f2, region) = _IVQ(Any[f1, f2], x -> f1(x) <= 0 && f2(x) <= 0, region; gauss_points=4,
                                lipschitz=[1.2, 2.3])
    @test rule(bounded(p1), bounded(p2), corner) == rule(p1, p2, corner)
    @test _quad(x -> 1.0, rule(bounded(p1), bounded(p2), corner)) ≈ 0.0144 rtol = 1e-14
    domain = physical_domain(intersect(leaf(bounded(p1); lipschitz=1.2), leaf(bounded(p2); lipschitz=2.3));
                             subcell_length_scale=0.25)
    _, weights, _, status = Unfitted.moment_fit_rule(domain, corner, (4, 4))
    @test status === :fitted
    @test sum(weights) ≈ 0.0144 rtol = 1e-13
    q1 = x -> (x[1] - 0.12) + 0.5 * (x[2] - 0.12) + 0.3 * (x[3] - 0.1)
    q2 = x -> -(x[1] - 0.12) + 2.0 * (x[2] - 0.12) + 0.2 * (x[3] - 0.1)
    corner3 = box((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    @test rule(bounded(q1), bounded(q2), corner3) == rule(q1, q2, corner3)
end

@testset "implicit quadrature — traces on opposite faces get no pair leaf" begin
    # The base problem pairs its monotone leaves for the crossings inside the box, but never two restrictions to
    # opposite faces of the previous reduction: the fiber integral is a sum of one term per root clamped to the fiber,
    # so a trace on the face lo and one on the face hi kink different terms, and their crossing needs no breakpoint. Two
    # traces on one face still pair, and so do leaves that are not traces.
    g1 = x -> x[1] + x[2] - 0.6
    g2 = x -> x[1] - x[2] + 0.1
    traces = Any[Unfitted._restrict(g, 1, v) for g in (g1, g2) for v in (0.0, 0.5)]
    base = Unfitted._base_leaves(traces, fill(:monotone, 4), 1, 0.0, 0.5)
    pairs = [l.f for l in base if l isa Unfitted._FDLeaf]
    @test length(pairs) == 2 && all(p -> Unfitted._face(p.gi) == Unfitted._face(p.gj), pairs)
    plain = Unfitted._base_leaves(Any[g1, g2], [:monotone, :monotone], 1, 0.0, 0.5)
    @test count(l -> l isa Unfitted._FDLeaf, plain) == 1

    # What that saves: cube − (sphere ∪ cylinder) around the common axis, where the cylinder lies inside the sphere and
    # its traces on one face cross the sphere's on the other in every cell. Paired, they add a breakpoint to each of
    # those cells that bounds no part of Ω (3072 kernel points instead of 2240), enough to starve the first moment-fit
    # attempt: it missed its target or fitted to only 1e-9, and a miss refits with an eight times larger candidate
    # cloud. Now each cell fits at the first attempt, to round-off.
    sphere = leaf(x -> norm(x - SVector(0.5, 0.5, 0.5)) - 0.32; lipschitz=1.0)
    cylinder = leaf(x -> hypot(x[1] - 0.5, x[2] - 0.5) - 0.14; lipschitz=1.0)
    domain = physical_domain(complement(union(sphere, cylinder)); subcell_length_scale=1 / 64, max_depth=4)
    for lower in Iterators.product((0.25, 0.5), (0.25, 0.5), (0.25, 0.5))
        _, _, residual, status = Unfitted._moment_fit_rule(domain, box(lower, lower .+ 0.25), (4, 4, 4), 1e-6, 1)
        @test status === :fitted && residual < 1e-14
    end
end
