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

# ── Return side: the stress row the test-component contraction selects ────────

@testset "UnfittedTensorsExt: TestChannels(value, σ, c) is the row σ : ∇v asks for" begin
    # Assembly contracts the gradient channel against the scalar test basis
    # gradient of test component `c`, i.e. against `v = N eᶜ`. The channel
    # must therefore satisfy ⟨channel, ∇N⟩ = σ : ∇v = σ : (eᶜ ⊗ ∇N) for every
    # `c`. Checked against a deliberately NON-symmetric σ, where row and
    # column differ — for the symmetric σ of small-strain elasticity a
    # transposed extraction is invisible.
    σ2 = Tensor{2,2,Float64}((1.0, 2.0, 3.0, 4.0))          # column-major
    @test σ2[1, 2] != σ2[2, 1]
    ∇N2 = Vec{2,Float64}((0.7, -1.3))
    for c in 1:2
        channels = TestChannels(-0.5, σ2, c)
        @test channels isa TestChannels{2,Float64}
        @test channels.value == -0.5
        @test channels.gradient == SVector(σ2[c, 1], σ2[c, 2])
        e_c = basevec(Vec{2,Float64}, c)
        @test dot(channels.gradient, SVector(Tuple(∇N2))) ≈ σ2 ⊡ (e_c ⊗ ∇N2)
    end

    # 3D, and a `SymmetricTensor` argument: same contraction identity.
    σ3 = SymmetricTensor{2,3,Float64}((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    ∇N3 = Vec{3,Float64}((0.5, -0.25, 1.5))
    for c in 1:3
        channels = TestChannels(0.0, σ3, c)
        @test channels isa TestChannels{3,Float64}
        e_c = basevec(Vec{3,Float64}, c)
        @test dot(channels.gradient, SVector(Tuple(∇N3))) ≈ σ3 ⊡ (e_c ⊗ ∇N3)
    end
end

# ── State side: the wrappers against an analytic value and Jacobian ───────────

# L² projection of `f` onto the model's space, returned as a coefficient
# vector for `state =`. Exact whenever `f` lies in the space — a linear
# field at order ≥ 1 does — so the tensor wrappers can be checked against
# `f` itself rather than against the accessors they are built from.
function _project_field(model, u, f)
    gram = Matrix(assemble_matrix(model, mass_block(u)))
    rhs = Vector(assemble_vector(model, source_load(u; source=f)))
    return gram \ rhs
end

@testset "UnfittedTensorsExt: gradient_tensor is [∂uᵢ/∂xⱼ], not its transpose" begin
    # A linear field with a non-symmetric Jacobian `A`, reproduced exactly by
    # the order-1 space. `gradient_tensor` must return `A`; the transposed
    # convention differs by `|A[1,2] − A[2,1]| = 2`, so this pins the index
    # order rather than restating the implementation.
    for (cells, A, b) in (((2, 2), SMatrix{2,2,Float64}(3.0, 7.0, 5.0, 11.0), SVector(0.25, -0.5)),
                          ((1, 1, 1), SMatrix{3,3,Float64}(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0),
                           SVector(0.1, 0.2, 0.3)))
        D = length(cells)
        exact(x) = A * SVector{D,Float64}(x) + b
        V = space(box(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D)); cells=cells, order=1)
        u = field(:u, V; components=D)
        model = prepare(Problem((u,)))
        coefficients = _project_field(model, u, exact)

        worst = Ref((0.0, 0.0, 0))
        foreach_quadrature_point(model; state=coefficients) do q
            value_error, gradient_error, visited = worst[]
            v = value_vec(q.state, u)
            G = gradient_tensor(q.state, u)
            value_error = max(value_error, maximum(abs.(Tuple(v) .- Tuple(exact(q.x)))))
            for i in 1:D, j in 1:D
                gradient_error = max(gradient_error, abs(G[i, j] - A[i, j]))
            end
            worst[] = (value_error, gradient_error, visited + 1)
        end
        value_error, gradient_error, visited = worst[]
        @test visited > 0
        @test value_error < 1.0e-12
        @test gradient_error < 1.0e-11
    end
end

# ── End to end: the elasticity operator annihilates the rigid-body modes ──────

@testset "UnfittedTensorsExt: tensor-notation elasticity kills rigid-body modes" begin
    # The whole chain — `symmetric_gradient` on the trial side, a constitutive
    # double contraction, the stress-row return — assembled as a real
    # `component_aware` block. Its null space must contain the rigid-body
    # motions of plane elasticity: two translations and the infinitesimal
    # rotation. Nothing in the implementation encodes that, so it is an
    # independent check of ε(u), the contraction, and the row convention at
    # once. A transposed row extraction breaks the rotation mode.
    λ, μ = 1.0, 1.0
    ℂ = SymmetricTensor{4,2,Float64}((i, j, k, l) -> λ * (i == j) * (k == l) +
                                                     μ * ((i == k) * (j == l) + (i == l) * (j == k)))
    tensor_bilinear(q, trial, c) = TestChannels(0.0, ℂ ⊡ symmetric_gradient(trial), c)
    # The spelling the shipped examples hand-roll, kept as a second opinion on
    # the new constructor.
    function manual_bilinear(q, trial, c)
        σ = ℂ ⊡ symmetric_gradient(trial)
        return TestChannels(0.0, Vec{2,Float64}((σ[c, 1], σ[c, 2])))
    end

    V = space(box((0.0, 0.0), (1.0, 1.0)); cells=(2, 2), order=1)
    u = field(:u, V; components=2)
    model = prepare(Problem((u,)))
    form(bilinear) = WeakForm(bilinear=bilinear, linear=(q, c) -> 0.0, symmetric=true,
                              component_aware=true)
    K = Matrix(assemble_matrix(model, block(u, u, form(tensor_bilinear))))
    K_manual = Matrix(assemble_matrix(model, block(u, u, form(manual_bilinear))))
    @test K == K_manual

    rigid_modes = (x -> SVector(1.0, 0.0), x -> SVector(0.0, 1.0), x -> SVector(-x[2], x[1]))
    scale = opnorm(K)
    @test scale > 0
    for mode in rigid_modes
        coefficients = _project_field(model, u, mode)
        @test norm(K * coefficients) < 1.0e-10 * scale * max(1.0, norm(coefficients))
    end

    # A non-rigid mode is not annihilated, so the assertion above has teeth.
    stretch = _project_field(model, u, x -> SVector(x[1], 0.0))
    @test norm(K * stretch) > 1.0e-3 * scale
end

# ── Space-time shape: fewer field components than the mesh has axes ───────────

@testset "UnfittedTensorsExt: the strain is sized by the field, not by the mesh" begin
    # Three mesh axes, two field components — the space-time shape, where the
    # last axis is time and the displacement carries one component per
    # *spatial* axis. One prepared model serves every assertion below: the
    # suite is compile-bound, so a second discretisation would cost more than
    # it proves.
    λ, μ = 1.3, 0.7
    isotropic(i, j, k, l) = λ * (i == j) * (k == l) +
                            μ * ((i == k) * (j == l) + (i == l) * (j == k))
    ℂ_mesh = SymmetricTensor{4,3,Float64}(isotropic)
    ℂ_field = SymmetricTensor{4,2,Float64}(isotropic)

    V = space(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)); cells=(2, 2, 2), order=1)
    u = field(:u, V; components=2)
    @test u isa Field{3,Float64,2}
    model = prepare(Problem((u,)))
    form(bilinear) = WeakForm(bilinear=bilinear, linear=(q, c) -> 0.0, symmetric=true,
                              component_aware=true)

    # The one-argument strain is a 3×3 tensor here — the defect — while the
    # two-argument one is the 2×2 tensor the field actually has.
    mesh_bilinear(q, trial, c) = TestChannels(0.0, ℂ_mesh ⊡ symmetric_gradient(trial), c)
    field_bilinear(q, trial, c) = TestChannels(0.0, ℂ_field ⊡ symmetric_gradient(trial, u), c)
    A_mesh = Matrix(assemble_matrix(model, block(u, u, form(mesh_bilinear))))
    A_field = Matrix(assemble_matrix(model, block(u, u, form(field_bilinear))))

    # The surplus the mesh-sized strain carries, written from scratch as an
    # ordinary component-diagonal stiffness along the last axis rather than
    # derived from either form above, so the identity below is a genuine
    # cross-check and not a restatement. A component-unaware form supplies the
    # `trial.component == test_component` pattern itself.
    time_bilinear(q, trial) = TestChannels(0.0, SVector(0.0, 0.0, μ * trial.gradient[3]))
    A_time = Matrix(assemble_matrix(model,
                                    block(u, u,
                                          WeakForm(bilinear=time_bilinear, linear=q -> 0.0,
                                                   symmetric=true))))

    # The load-bearing assertion is the exact identity
    #
    #     A_mesh − A_field = μ ∫_Q ∂ₜu · ∂ₜv,
    #
    # because it states the defect in the language of the defect. A tolerance
    # on a solution cannot: the mesh-sized operator is symmetric, positive
    # semi-definite, solves to roundoff and converges under refinement, so a
    # coincidentally close answer satisfies any bound placed on it. This
    # equality holds only if the surplus is exactly the spurious
    # time-stiffness and nothing else, and it breaks the moment either strain
    # changes size.
    scale = opnorm(A_mesh)
    @test scale > 0
    @test opnorm(A_time) > 0.1 * scale               # the defect is not a rounding matter
    @test norm(A_mesh - A_field - A_time) < 1.0e-13 * scale

    # The two-argument form returns a genuine 2×2 tensor — not a 3×3 one with
    # the time slots zeroed, which would still report the wrong trace to
    # `dev` and the wrong invariants to any material law — and it agrees with
    # the mesh-sized result exactly where the two overlap.
    trial = TrialChannels(1, 1.0, SVector(0.7, -1.3, 0.4))
    ε_field = symmetric_gradient(trial, u)
    ε_mesh = symmetric_gradient(trial)
    @test ε_field isa SymmetricTensor{2,2,Float64}
    @test ε_mesh isa SymmetricTensor{2,3,Float64}
    @test ε_field === symmetric_gradient(trial, Val(2))
    for i in 1:2, j in 1:2
        @test ε_field[i, j] == ε_mesh[i, j]
    end

    # The short gradient coefficient a C×C stress produces is zero-extended
    # into the axes the field has no component for.
    @test Unfitted._as_test_channels(TestChannels(1.5, SVector(2.0, 3.0)), Val(3), Float64) ===
          TestChannels(1.5, SVector(2.0, 3.0, 0.0))

    # Both guards refuse the long direction loudly. Truncating either would be
    # a fresh instance of the silent wrongness this testset exists to pin.
    @test_throws DimensionMismatch symmetric_gradient(trial, Val(4))
    @test_throws DimensionMismatch Unfitted._as_test_channels(TestChannels(0.0,
                                                                           SVector(1.0, 2.0, 3.0)),
                                                              Val(2), Float64)

    # The ergonomic three-argument `TestChannels(value, σ, component)` return
    # path died inside assembly with a `MethodError` on `_as_test_channels`
    # whenever the stress was smaller than the mesh, which is every `C ≠ D`
    # field. Drive it through a whole solve.
    solved = prepare(Problem(u,
                             WeakForm(bilinear=field_bilinear,
                                      linear=(q, c) -> (c == 1 ? 1.0 : 0.0), symmetric=true,
                                      component_aware=true);
                             dirichlet=[dirichlet(SVector(0.0, 0.0); on=boundary(:all))]))
    assemble!(solved)
    sol = solve!(solved)
    @test active_unknowns(solved) > 0
    @test norm(solved.matrix * sol.coefficients - solved.rhs) < 1.0e-10 * norm(solved.rhs)
    @test norm(sol.coefficients) > 0

    # `gradient_tensor`'s one `Val` argument sizes both tensor indices, so a
    # `C ≠ D` field has no square Jacobian to read: `Val(C)` truncates the
    # columns to the first `C` axes and `Val(D)` runs off the components the
    # field carries. Pinned so the docstring saying exactly that stays honest.
    coefficients = [1.0 + 0.1 * i for i in 1:active_unknowns(model)]
    visited = Ref(0)
    foreach_quadrature_point(model; state=coefficients) do q
        visited[] += 1
        visited[] == 1 || return nothing
        G = gradient_tensor(q.state, :u, Val(2))
        for i in 1:2, j in 1:2
            @test G[i, j] == field_gradient(q.state, :u, i)[j]
        end
        @test_throws BoundsError gradient_tensor(q.state, :u, Val(3))
        @test_throws MethodError gradient_tensor(q.state, u)
        return nothing
    end
    @test visited[] > 0
end

# ── Hot-loop contract: the conversions must not allocate ──────────────────────

# Every conversion the assembly path uses, measured over a warmed loop inside
# a single function so that each operand is a local of statically known type —
# the context the element kernel actually provides. Measuring at global scope
# would report the boxing of the globals instead of the conversion. Each loop
# feeds `sink` so it cannot be eliminated as dead code, and `sink` is returned
# so the caller can pin that the work happened.
function _conversion_allocations()
    ℂ = SymmetricTensor{4,2,Float64}((i, j, k, l) -> (i == j) * (k == l) +
                                                     ((i == k) * (j == l) + (i == l) * (j == k)))
    trial2 = TrialChannels(1, 1.0, SVector(0.7, -1.3))
    trial3 = TrialChannels(2, 1.0, SVector(0.5, -0.25, 1.5))
    vec2 = Vec{2,Float64}((1.5, -0.25))
    vec3 = Vec{3,Float64}((0.1, 0.2, 0.3))
    stress(trial, c) = TestChannels(0.0, ℂ ⊡ symmetric_gradient(trial), c)

    sink = 0.0
    warmup = (Vec(trial2)[1], Vec(trial3)[1], symmetric_gradient(trial2)[1, 1],
              symmetric_gradient(trial3)[1, 1], TestChannels(0.0, vec2).gradient[1],
              TestChannels(0.0, vec3).gradient[1], stress(trial2, 1).gradient[1])
    sink += sum(warmup)

    vec_2d = @allocated for _ in 1:100
        sink += Vec(trial2)[1]
    end
    vec_3d = @allocated for _ in 1:100
        sink += Vec(trial3)[1]
    end
    strain_2d = @allocated for _ in 1:100
        sink += symmetric_gradient(trial2)[1, 1]
    end
    strain_3d = @allocated for _ in 1:100
        sink += symmetric_gradient(trial3)[1, 1]
    end
    channels_2d = @allocated for _ in 1:100
        sink += TestChannels(0.0, vec2).gradient[1]
    end
    channels_3d = @allocated for _ in 1:100
        sink += TestChannels(0.0, vec3).gradient[1]
    end
    stress_row = @allocated for _ in 1:100
        sink += stress(trial2, 1).gradient[1]
    end
    return (; vec_2d, vec_3d, strain_2d, strain_3d, channels_2d, channels_3d, stress_row, sink)
end

@testset "UnfittedTensorsExt: the assembly-path conversions are allocation-free" begin
    # These conversions run once per test component per trial dof per
    # quadrature point, so the package's allocation-free element-kernel
    # contract applies to them. A nonzero count means a conversion fell off
    # the inlined path — `Tuple(::Vec)` used to, at 208 B a call, on the
    # return path of every tensor-notation bilinear callback.
    allocations = _conversion_allocations()
    @test isfinite(allocations.sink)
    for name in (:vec_2d, :vec_3d, :strain_2d, :strain_3d, :channels_2d, :channels_3d, :stress_row)
        @test (name, getfield(allocations, name)) == (name, 0)
    end
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
