using StaticArrays
using LinearAlgebra

# Integrate `f` against a kernel rule `(points, weights)`.
_quad(f, rule) = sum(rule[2][i] * f(rule[1][i]) for i in eachindex(rule[2]); init=0.0)

const _IVQ = Unfitted.implicit_volume_quadrature
const _ISQ = Unfitted.implicit_surface_quadrature

@testset "implicit quadrature — 1D linear cut moments are exact" begin
    # Ω = {x ≤ 0.7} on [0, 1]. Monomial moments ∫₀^0.7 xⁿ dx = 0.7ⁿ⁺¹/(n+1)
    # are reproduced to machine precision: a linear interface is graph-like,
    # so the Saye rule is polynomial-exact.
    a = 0.7
    rule = _IVQ(x -> x[1] - a, box((0.0,), (1.0,)); gauss_points=6)
    for n in 0:5
        @test _quad(x -> x[1]^n, rule) ≈ a^(n + 1) / (n + 1) atol = 1e-13
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
    # Ω = {x ≤ 0.3 + y} on [0, 1]². For y > 0.7 the root leaves the fiber and
    # the inside length saturates at 1, a kink in the projected integrand. The
    # base recursion partitions at the face restriction {x = 1} (i.e. y = 0.7),
    # so the moments stay machine-exact. Analytic:
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
    simplex = _IVQ(x -> x[1] + x[2] + x[3] - 1, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
                   gauss_points=8)
    @test _quad(x -> 1.0, simplex) ≈ 1 / 6 atol = 1e-12
end

@testset "implicit quadrature — 4D linear cut volume (dimension-generic smoke)" begin
    # The recursion is written for arbitrary D; a 4D half-space cut exercises
    # the D → D−1 reduction three times. Ω = {x₁ ≤ a} on [0, 1]⁴ ⇒ volume a.
    a = 0.6
    lo = SVector(0.0, 0.0, 0.0, 0.0)
    hi = SVector(1.0, 1.0, 1.0, 1.0)
    rule = _IVQ(x -> x[1] - a, AxisBox(lo, hi); gauss_points=4)
    @test _quad(x -> 1.0, rule) ≈ a atol = 1e-12
    @test _quad(x -> x[2], rule) ≈ a * 0.5 atol = 1e-12
end

@testset "implicit quadrature — tensor Legendre moment vector matches analytic" begin
    # The moment-fit pipeline needs mₐ = ∫_{Ω∩R} ψₐ dx with ψₐ the tensor
    # Legendre basis. Build that vector from the kernel rule and the existing
    # Legendre evaluators in `fcm.jl`, then check the constant moment equals
    # the cut area and the orthogonality-driven zeros vanish, for Ω = {x ≤ 0.4}.
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
    # α = (0,n) for n ≥ 1: ∫₀^a dx · ∫₀¹ Lₙ(2y−1) dy = a · 0 by Legendre
    # orthogonality to the constant.
    for n in 1:3
        k = findfirst(==(CartesianIndex(0, n)), idx)
        @test abs(moments[k]) < 1e-12
    end
end

@testset "implicit quadrature — curved interface is high-order (clean cell)" begin
    # A rim cut cell with no turning point: the arc is a single-valued graph in
    # the max-gradient axis, so the rule is spectrally accurate. Self-converge:
    # raising the order changes the area only at round-off scale.
    phi = x -> sqrt(x[1]^2 + x[2]^2) - 1.0
    b = box((0.5, 0.2), (0.9, 0.6))
    a8 = _quad(x -> 1.0, _IVQ(phi, b; gauss_points=8))
    a16 = _quad(x -> 1.0, _IVQ(phi, b; gauss_points=16))
    @test isapprox(a8, a16; atol=1e-10)
end

@testset "implicit quadrature — disk area and sphere volume converge" begin
    # Whole disk / ball in one box (off-grid centre so subdivision isolates the
    # interior gradient-null point). The four (six) cardinal turning points cap
    # accuracy at the Gauss order, but the area/volume is still good to ~1e-4
    # and improves with the order.
    disk = x -> sqrt((x[1] - 0.53)^2 + (x[2] - 0.47)^2) - 0.3
    a10 = _quad(x -> 1.0, _IVQ(disk, box((0.0, 0.0), (1.0, 1.0)); gauss_points=10, max_subdiv=6))
    @test isapprox(a10, π * 0.3^2; atol=1e-4)

    ball = x -> sqrt((x[1] - 0.53)^2 + (x[2] - 0.47)^2 + (x[3] - 0.51)^2) - 0.3
    v8 = _quad(x -> 1.0,
               _IVQ(ball, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); gauss_points=8, max_subdiv=6))
    v12 = _quad(x -> 1.0,
                _IVQ(ball, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); gauss_points=12, max_subdiv=6))
    truth = 4 / 3 * π * 0.3^3
    @test isapprox(v8, truth; atol=1e-4)
    @test abs(v12 - truth) < abs(v8 - truth)   # higher order ⇒ smaller error
end

@testset "implicit quadrature — surface rule" begin
    # Flat surface: {x = 0.5} ∩ [0,1]³ has area 1, reproduced exactly (the
    # height graph is constant). The full circle perimeter is limited by the
    # cardinal turning points, like the disk area, but still close.
    flat = _ISQ(x -> x[1] - 0.5, box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); gauss_points=6)
    @test _quad(x -> 1.0, flat) ≈ 1.0 atol = 1e-12

    disk = x -> sqrt((x[1] - 0.53)^2 + (x[2] - 0.47)^2) - 0.3
    perim = _quad(x -> 1.0, _ISQ(disk, box((0.0, 0.0), (1.0, 1.0)); gauss_points=12, max_subdiv=6))
    @test isapprox(perim, 2π * 0.3; atol=1e-2)

    @test_throws ArgumentError _ISQ(x -> x[1] - 0.5, box((0.0,), (1.0,)); gauss_points=4)
end

@testset "implicit quadrature — empty and full regions" begin
    # Ω = ∅ (φ ≡ 1) ⇒ no points. Ω = everything (φ ≡ −1) ⇒ full tensor rule
    # whose weights sum to the box volume.
    empty = _IVQ(x -> 1.0, box((0.0, 0.0), (1.0, 1.0)); gauss_points=4)
    @test isempty(empty[1])
    full = _IVQ(x -> -1.0, box((0.0, 0.0), (2.0, 1.0)); gauss_points=4)
    @test sum(full[2]) ≈ 2.0 atol = 1e-13
end

# Build a kernel volume rule from a CSG level-set tree (leaves + membership).
function _csg_rule(geom, region; gp, ms=5)
    _IVQ(Any[l.f for l in Unfitted._leaves(geom)], x -> Unfitted._inside(geom, x), region;
         gauss_points=gp, max_subdiv=ms)
end

@testset "implicit quadrature — force-reduce stays correct without subdivision" begin
    # A whole disk centred in the box has an interior point where ∇φ = 0 and is
    # tangent to every axis somewhere, so no axis is graph-like. With no
    # subdivision budget the kernel force-reduces on the least-bad axis: the
    # multi-root fiber scan still yields the correct region (only the order
    # degrades), so the area is right to a loose tolerance and never throws.
    disk = x -> sqrt((x[1] - 0.53)^2 + (x[2] - 0.47)^2) - 0.3
    a0 = _quad(x -> 1.0, _IVQ(disk, box((0.0, 0.0), (1.0, 1.0)); gauss_points=6, max_subdiv=0))
    @test isfinite(a0) && a0 > 0
    @test isapprox(a0, π * 0.3^2; atol=5e-2)          # correct region, low order
    a2 = _quad(x -> 1.0, _IVQ(disk, box((0.0, 0.0), (1.0, 1.0)); gauss_points=6, max_subdiv=2))
    @test isapprox(a2, π * 0.3^2; atol=1e-3)           # subdivision recovers accuracy
end

@testset "implicit quadrature — CSG intersection (polytope corner) is exact" begin
    # Ω = {x ≤ a} ∩ {y ≤ b} is the box [0,a]×[0,b]. The perpendicular half-plane
    # is FLAT along the reduction axis, so the rule is polynomial-exact.
    a, b = 0.7, 0.6
    geom = intersect(leaf(x -> x[1] - a), leaf(x -> x[2] - b))
    rule = _csg_rule(geom, box((0.0, 0.0), (1.0, 1.0)); gp=6)
    @test _quad(x -> 1.0, rule) ≈ a * b atol = 1e-13
    @test _quad(x -> x[1] * x[2], rule) ≈ (a^2 / 2) * (b^2 / 2) atol = 1e-13
    @test _quad(x -> x[1]^2 * x[2]^2, rule) ≈ (a^3 / 3) * (b^3 / 3) atol = 1e-13
end

@testset "implicit quadrature — CSG union and difference" begin
    # Union of two overlapping half-strips: {x ≤ 0.3} ∪ {x ≥ 0.7} on [0,1]²,
    # area 0.6, exact (each leaf is graph-like in x, the base is partitioned by
    # both faces). Difference: {x ≤ 0.8} \ {x ≤ 0.3} = strip [0.3,0.8], area 0.5.
    uni = union(leaf(x -> x[1] - 0.3), leaf(x -> 0.7 - x[1]))
    @test _quad(x -> 1.0, _csg_rule(uni, box((0.0, 0.0), (1.0, 1.0)); gp=6)) ≈ 0.6 atol = 1e-12
    diff = setdiff(leaf(x -> x[1] - 0.8), leaf(x -> x[1] - 0.3))
    @test _quad(x -> 1.0, _csg_rule(diff, box((0.0, 0.0), (1.0, 1.0)); gp=6)) ≈ 0.5 atol = 1e-12
end

@testset "implicit quadrature — CSG annulus is high-order" begin
    # Annulus as a disk minus a disk (no fake mid-radius crease): quarter in
    # [0,1]² with both circle boundaries smooth ⇒ high accuracy.
    ann = setdiff(leaf(x -> hypot(x...) - 0.45), leaf(x -> hypot(x...) - 0.2))
    a = _quad(x -> 1.0, _csg_rule(ann, box((0.0, 0.0), (1.0, 1.0)); gp=14, ms=6))
    @test isapprox(a, π * (0.45^2 - 0.2^2) / 4; atol=1e-4)
end

@testset "implicit quadrature — finite-difference gradient fallback" begin
    # For level sets that are not AD-differentiable the caller can pass a
    # finite-difference gradient; on a smooth linear cut it reproduces the
    # ForwardDiff result to integration accuracy.
    a = 0.45
    ad = _IVQ(x -> x[1] - a, box((0.0, 0.0), (1.0, 1.0)); gauss_points=6)
    fd = _IVQ(x -> x[1] - a, box((0.0, 0.0), (1.0, 1.0)); gauss_points=6,
              grad=Unfitted._fd_gradient)
    @test _quad(x -> 1.0, fd) ≈ _quad(x -> 1.0, ad) atol = 1e-10
    @test _quad(x -> x[1] * x[2], fd) ≈ _quad(x -> x[1] * x[2], ad) atol = 1e-10
end

@testset "implicit quadrature — Float32 carries through" begin
    # The kernel is scalar-type generic; a Float32 box returns a Float32 rule.
    rule = _IVQ(x -> x[1] - 0.5f0, box((0.0f0, 0.0f0), (1.0f0, 1.0f0)); gauss_points=5)
    @test eltype(rule[2]) === Float32
    @test _quad(x -> 1.0f0, rule) ≈ 0.5f0 atol = 1.0f-5
end
