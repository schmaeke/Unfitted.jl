using LinearAlgebra
using StaticArrays
using Tensors

# ── Trial-side: TrialChannels → Tensors.jl objects ────────────────────────────

@testset "UnfittedTensorsExt: Vec(::TrialChannels) preserves component order" begin
    # 2D: the Vec just exposes the SVector-backed gradient channel.
    t2 = TrialChannels(1, 1.0, SVector(0.7, -1.3))
    v2 = Vec(t2)
    @test v2 isa Vec{2,Float64}
    @test Tuple(v2) === Tuple(t2.gradient)

    # 3D: same contract — the wrapper is dimension-independent.
    t3 = TrialChannels(2, 1.0, SVector(0.5, -0.25, 1.5))
    v3 = Vec(t3)
    @test v3 isa Vec{3,Float64}
    @test (v3[1], v3[2], v3[3]) === (0.5, -0.25, 1.5)
end

@testset "UnfittedTensorsExt: symmetric_gradient matches the closed-form ε(N eₖ)" begin
    # 2D: each of the two component branches against the Voigt-tensor reference
    # ε = (εxx, εyy, εxy) (note: tensor shear εxy, not engineering γxy).
    for c in 1:2
        t = TrialChannels(c, 1.0, SVector(0.7, -1.3))
        ε = symmetric_gradient(t)
        @test ε isa SymmetricTensor{2,2,Float64}
        ref = c == 1 ? (εxx=0.7, εyy=0.0, εxy=-0.65) : (εxx=0.0, εyy=-1.3, εxy=0.35)
        @test ε[1, 1] ≈ ref.εxx
        @test ε[2, 2] ≈ ref.εyy
        @test ε[1, 2] ≈ ref.εxy
    end

    # 3D: enforce the minor symmetry contract componentwise against the
    # analytic expansion ½ (eₖ ⊗ ∇N + ∇N ⊗ eₖ).
    for c in 1:3
        g = SVector(0.1 * c, -0.2, 0.4)
        t = TrialChannels(c, 1.0, g)
        ε = symmetric_gradient(t)
        @test ε isa SymmetricTensor{2,3,Float64}
        for i in 1:3, j in 1:3
            ref = ((i == c) * g[j] + (j == c) * g[i]) / 2
            @test ε[i, j] ≈ ref
        end
    end
end

# ── Test-side: TestChannels(::Real, ::Vec) constructor parity ─────────────────

@testset "UnfittedTensorsExt: TestChannels(value, ::Vec) ≡ SVector path" begin
    g2 = Vec{2,Float64}((1.5, -0.25))
    @test TestChannels(0.0, g2) === TestChannels(0.0, SVector{2,Float64}(1.5, -0.25))

    g3 = Vec{3,Float64}((0.1, 0.2, 0.3))
    @test TestChannels(-1.0, g3) === TestChannels(-1.0, SVector{3,Float64}(0.1, 0.2, 0.3))
end

# ── End-to-end: anisotropic Poisson via SymmetricTensor matches SMatrix ───────

@testset "UnfittedTensorsExt: anisotropic 2D Poisson assembly matches the SMatrix baseline" begin
    # Same coefficient and forcing expressed two ways:
    #   * baseline form uses SMatrix * SVector for K∇u;
    #   * Tensors form uses SymmetricTensor ⋅ Vec.
    # The assembled matrix and rhs must agree to roundoff — proves that
    # the channel round-trip through `Vec` and `TestChannels(::Real, ::Vec)`
    # preserves the assembly contribution exactly.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=2)
    bc = [dirichlet(0.0; on=boundary(:all))]

    K_mat = SMatrix{2,2,Float64}(2.0, 0.3, 0.3, 1.0)        # (xx, yx, xy, yy)
    K_sym = SymmetricTensor{2,2,Float64}((2.0, 0.3, 1.0))    # (xx, xy, yy)
    source(x) = sin(pi * x[1]) * sin(pi * x[2])

    form_baseline = WeakForm(; bilinear=(q, u) -> TestChannels(0.0, K_mat * u.gradient),
                             linear=q -> source(q.x), symmetric=true)
    form_tensors = WeakForm(; bilinear=(q, u) -> TestChannels(0.0, K_sym ⋅ Vec(u)),
                            linear=q -> source(q.x), symmetric=true)

    m_b = prepare(Problem(V, form_baseline; dirichlet=bc))
    m_t = prepare(Problem(V, form_tensors; dirichlet=bc))
    assemble!(m_b)
    assemble!(m_t)

    @test Matrix(m_b.matrix) ≈ Matrix(m_t.matrix) atol=1.0e-13
    @test Vector(m_b.rhs) ≈ Vector(m_t.rhs) atol=1.0e-13
end

# ── State-side: value_vec / gradient_tensor match per-component accessors ─────

@testset "UnfittedTensorsExt: value_vec and gradient_tensor mirror component reads" begin
    # Run a public-API quadrature-point traversal with `state=coefficients`
    # over a small 2D vector-field model. The contract under test is that
    # the tensor wrappers' i-th coordinate equals the corresponding
    # scalar accessor at the same quadrature point — no analytic
    # solution required.
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    u = field(:u, V; components=2)
    model = prepare(Problem((u,)))

    # Deterministic non-trivial coefficient vector so both `value` and
    # `gradient` reads exercise non-zero arithmetic.
    coeffs = [1.0 + 0.1 * i for i in 1:active_unknowns(model)]

    visited = Ref(0)
    foreach_quadrature_point(model; state=coeffs) do q
        visited[] += 1
        vt = value_vec(q.state, :u, Val(2))
        gt = gradient_tensor(q.state, :u, Val(2))
        @test vt isa Vec{2,Float64}
        @test gt isa Tensor{2,2,Float64}
        @test vt[1] ≈ value(q.state, :u, 1)
        @test vt[2] ≈ value(q.state, :u, 2)
        for i in 1:2, j in 1:2
            @test gt[i, j] ≈ field_gradient(q.state, :u, i)[j]
        end
    end
    @test visited[] > 0
end

@testset "UnfittedTensorsExt: Field-typed overloads ≡ Val(D) forms" begin
    # Passing the `Field` object instead of the bare name lets the
    # extension infer the spatial dim and component count from the
    # field's parametric type, dropping `Val(D)` at every call site.
    # Equivalence to the `Val(D)` form is bit-for-bit (both stack the
    # same scalar reads into the same Vec/Tensor constructor).
    omega = box((0.0, 0.0), (1.0, 1.0))
    V = space(omega; cells=(2, 2), order=1)
    u = field(:u, V; components=2)
    model = prepare(Problem((u,)))
    coeffs = [1.0 + 0.1 * i for i in 1:active_unknowns(model)]

    visited = Ref(0)
    foreach_quadrature_point(model; state=coeffs) do q
        visited[] += 1
        @test value_vec(q.state, u) === value_vec(q.state, :u, Val(2))
        @test gradient_tensor(q.state, u) === gradient_tensor(q.state, :u, Val(2))
    end
    @test visited[] > 0
end

# ── Tension/compression strain split: spectral vs. Cayley-Hamilton ────────────
#
# The phase-field single-edge-notched example (`examples/phase_field_*`)
# uses `Tensors.gradient` of the Miehe tension/compression-split stress
# `σ(ε, d)` as the algorithmic tangent. The natural spectral expression
# of the split goes through `eigen(::SymmetricTensor{2,2})`, whose
# eigenvector derivatives carry a `1/(e₁ − e₂)` factor that diverges at
# degenerate principal frames — so AD blows up there. The example
# therefore expresses the split via the matrix absolute value
#
#     ε⁺ = (ε + |ε|) / 2,
#
# with `|ε|` written in closed form by Cayley-Hamilton in terms of
# `tr(ε)`, `det(ε)`, `tr(ε²)`, and `|det(ε)|`. The Cayley-Hamilton
# formula is mathematically identical to the spectral one but
# polynomial in the components, so its AD derivative has no
# `1/(e₁ − e₂)` blow-up.
#
# The two tests below pin the contract the example relies on:
#
#   1. The Cayley-Hamilton split reproduces the spectral split to
#      roundoff on a sweep of strain regimes — same model, only the
#      formula changes.
#   2. AD through the Cayley-Hamilton formula returns a finite tangent
#      at every probed configuration, including the previously
#      problematic exactly-degenerate frame and the zero strain.
#
# A miniature local copy of the constitutive law keeps these tests
# self-contained; the example itself is not imported as a library. The
# third testset below pins the *spectral* path's failure mode as
# documentation — if a future Tensors.jl release smooths the
# degenerate-frame derivative, that test fires and the example's
# Cayley-Hamilton rewrite could in principle be reverted.

# Strain regimes the simulation is expected to visit. Reused by the
# two CH testsets and the spectral pinning testset.
const _SPLIT_PROBE_STRAINS = ("tensile separated" =>
                                  SymmetricTensor{2,2,Float64}((1.0e-3, 2.0e-4, 5.0e-4)),
                              "compressive separated" =>
                                  SymmetricTensor{2,2,Float64}((-1.0e-3, 2.0e-4, -5.0e-4)),
                              "mixed sign at trace cusp" =>
                                  SymmetricTensor{2,2,Float64}((1.0e-3, 5.0e-4, -1.0e-3)),
                              "near-degenerate frame" =>
                                  SymmetricTensor{2,2,Float64}((5.0e-4 + 1.0e-12, 1.0e-12, 5.0e-4)),
                              "exactly spherical" =>
                                  SymmetricTensor{2,2,Float64}((5.0e-4, 0.0, 5.0e-4)),
                              "exact zero" => zero(SymmetricTensor{2,2,Float64}))

function _spectral_split(ε::SymmetricTensor{2,2})
    let e = eigen(ε)
        p1 = e.vectors[:, 1] ⊗ e.vectors[:, 1]
        p2 = e.vectors[:, 2] ⊗ e.vectors[:, 2]
        plus = max(e.values[1], 0.0) * p1 + max(e.values[2], 0.0) * p2
        (symmetric(plus), ε - symmetric(plus))
    end
end

function _ch_split(ε::SymmetricTensor{2,2})
    let detε = det(ε)
        abs_ε = (tr(ε) * ε + 2 * max(-detε, 0.0) * one(ε)) /
                sqrt(tr(ε ⋅ ε) + 2 * abs(detε) + eps(Float64))
        plus = symmetric((ε + abs_ε) / 2)
        (plus, ε - plus)
    end
end

function _split_stress(ε::SymmetricTensor{2,2}, phase, λ, μ, k, split)
    plus, minus = split(ε)
    g = (1 - clamp(phase, 0.0, 1.0))^2 + k
    trp, trm = max(tr(ε), 0.0), min(tr(ε), 0.0)
    I2 = one(ε)
    return g * (λ * trp * I2 + 2 * μ * plus) + λ * trm * I2 + 2 * μ * minus
end

@testset "Cayley-Hamilton split reproduces spectral split to roundoff" begin
    for (name, ε) in _SPLIT_PROBE_STRAINS
        ε_plus_s, _ = _spectral_split(ε)
        ε_plus_ch, _ = _ch_split(ε)
        # Absolute roundoff at the strain scale we care about — the
        # spectral and CH formulas are mathematically identical so any
        # discrepancy is float-arithmetic reordering plus the `eps`
        # regularisation in the CH discriminant. 1e-13 is generous
        # given the strain magnitudes (~1e-3) and IEEE precision.
        @test norm(ε_plus_s - ε_plus_ch) < 1.0e-13
    end
end

@testset "AD through Cayley-Hamilton stress is finite at every probed strain" begin
    λ, μ, k, phase = 121.15, 80.77, 1.0e-6, 0.2
    for (name, ε) in _SPLIT_PROBE_STRAINS
        C, σ = Tensors.gradient(s -> _split_stress(s, phase, λ, μ, k, _ch_split), ε, :all)
        @test all(isfinite, C)
        @test all(isfinite, σ)
    end
end

@testset "AD through spectral eigen is non-finite at degenerate frames (pin)" begin
    # Pins Tensors.jl's documented limitation: dual-number propagation
    # through `eigen(::SymmetricTensor{2,2})` carries a `1/(e₁ − e₂)`
    # factor that NaNs at exact eigenvalue degeneracy. If a future
    # release regularises this, the assertion fires and the example's
    # rewrite away from `eigen` could be revisited.
    λ, μ, k, phase = 121.15, 80.77, 1.0e-6, 0.2
    ε_spherical = SymmetricTensor{2,2,Float64}((5.0e-4, 0.0, 5.0e-4))
    ad = Tensors.gradient(s -> _split_stress(s, phase, λ, μ, k, _spectral_split), ε_spherical)
    @test any(!isfinite, ad)
end
