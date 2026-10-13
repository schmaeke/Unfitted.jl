using StaticArrays
using LinearAlgebra

@testset "QuadField construction and stale-version contract" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    model = prepare(poisson(space(omega; cells=(2, 2), order=1); source=0.0))

    qf = QuadField{Float64}(model; init=q -> q.x[1] + q.x[2])
    @test length(qf) == Unfitted.nquadpoints(model)
    @test eltype(qf) === Float64

    # Stale-version detection: bumping the model invalidates the QuadField.
    activate!(model; level=1, cells=CartesianIndex{2}[])  # no-op, but bumps version
    @test_throws ArgumentError Unfitted._checked_quadfield(qf, model)

    # Fresh QuadField on the new model is fine.
    qf2 = QuadField{Float64}(model; init=q -> 1.0)
    @test Unfitted._checked_quadfield(qf2, model) === qf2.data
    @test all(qf2.data .== 1.0)
end

@testset "QuadField indexing and copy" begin
    omega = box((0.0,), (1.0,))
    model = prepare(poisson(space(omega; cells=2, order=1); source=0.0))
    qf = QuadField{Float64}(model)
    @test all(==(0.0), qf.data)
    qf[1] = 3.5
    @test qf[1] == 3.5
    q2 = copy(qf)
    q2[1] = 1.0
    @test qf[1] == 3.5   # original untouched
end

@testset "QuadField indexing out of range, and at a point of estimate's enriched model" begin
    model = prepare(poisson(space(box((0.0,), (1.0,)); cells=2, order=1); source=0.0))
    qf = QuadField{Float64}(model; init=q -> q.x[1])
    n = length(qf)
    @test qf[n] == qf.data[n]
    @test_throws BoundsError qf[0]
    @test_throws BoundsError qf[n+1]
    # `estimate`'s enriched model numbers its points from `_FOREIGN_POINTS`, and an index there is named for what it is.
    base = Unfitted._FOREIGN_POINTS
    @test_throws "another model's quadrature cloud" qf[base+1]
    @test_throws ArgumentError qf[base+n]

    # The constant's three properties (see "Detecting point-keyed state" in `data.jl`): every point of any feasible
    # cloud lies below the line `_out_of_range` reads; it is far below any index written by hand; and for no element
    # size from 1 to 64 bytes does `base · s` come within 2⁴⁸ of a multiple of 2⁶⁴, so an `@inbounds` read of a raw
    # vector at `base + k` cannot wrap around onto element `k`.
    @test base + 2^60 < base ÷ 2 < 0
    @test base < -2^61
    @test all(1:64) do s
        r = mod(Int128(base) * s, Int128(2)^64)
        return 2^48 <= r <= Int128(2)^64 - 2^48
    end
end

@testset "RBFP0 reproduces constant fields exactly (Sartorti & Düster Tab. 3)" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    source_model = prepare(poisson(space(omega; cells=(3, 3), order=2); source=0.0))
    target_model = prepare(poisson(space(omega; cells=(5, 5), order=1); source=0.0))

    source = QuadField{Float64}(source_model; init=q -> 7.25)
    target = transfer(source, source_model, target_model; via=RBFP0(; neighbors=10))

    @test all(isapprox.(target.data, 7.25; atol=1.0e-10))
    @test target.model_version == target_model.version
end

@testset "RBFP0 reproduces linear fields exactly via the polynomial extension" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    source_model = prepare(poisson(space(omega; cells=(3, 3), order=2); source=0.0))
    target_model = prepare(poisson(space(omega; cells=(5, 5), order=1); source=0.0))

    f = x -> 1.0 + 0.5 * x[1]
    source = QuadField{Float64}(source_model; init=q -> f(q.x))
    target = transfer(source, source_model, target_model; via=RBFP0(; neighbors=12))

    Unfitted.foreach_quadrature_point(target_model) do q
        @test isapprox(target.data[q.point], f(q.x); atol=5.0e-3)
    end
end

@testset "RBFP0 threaded and serial paths are bit-identical" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    source_model = prepare(poisson(space(omega; cells=(4, 4), order=2); source=0.0))
    target_model = prepare(poisson(space(omega; cells=(6, 6), order=1); source=0.0))

    source = QuadField{Float64}(source_model; init=q -> exp(-((q.x[1] - 0.5)^2 + (q.x[2] - 0.5)^2)))

    # The default count is solved as a static matrix; 16 neighbours exceed the size StaticArrays factors itself and go
    # through its dense-matrix fallback, which must agree across threads as well.
    for k in (10, 16)
        serial = transfer(source, source_model, target_model; via=RBFP0(; neighbors=k), threaded=false)
        threaded = transfer(source, source_model, target_model; via=RBFP0(; neighbors=k), threaded=true)
        @test serial.data == threaded.data
    end
end

# Allocation count of one warm serial RBFP0 transfer onto `target_model`, behind a function barrier so the count is the
# call's own.
function _warm_rbfp0_allocations(source, source_model, target_model)
    via = RBFP0(; neighbors=10)
    transfer(source, source_model, target_model; via, threaded=false)
    return @allocations transfer(source, source_model, target_model; via, threaded=false)
end

@testset "RBFP0's neighbour search allocates nothing per target point" begin
    # Each task refills one pair of `knn!` result vectors, so the neighbour search allocates nothing per target point.
    # `knn` allocated its two result vectors at every point, four allocations each on Julia 1.12, which took this
    # fixture's count up by 3 073 from the 8² target (256 points) to the 16² one (1 024), and by 3 841 at -O0. With
    # `knn!` the count grows by 1 in an optimised build (49 → 50); at -O0, where the suite runs, the local solve's
    # static matrix stays on the heap, one allocation per point (+769). The bound, two allocations per added point,
    # separates the two at either optimisation level.
    omega = box((0.0, 0.0), (1.0, 1.0))
    source_model = prepare(poisson(space(omega; cells=(4, 4), order=2); source=0.0))
    source = QuadField{Float64}(source_model; init=q -> q.x[1] * q.x[2])
    small, large = (prepare(poisson(space(omega; cells=(c, c), order=1); source=0.0)) for c in (8, 16))
    added = nquadpoints(large) - nquadpoints(small)
    @test _warm_rbfp0_allocations(source, source_model, large) - _warm_rbfp0_allocations(source, source_model, small) <
          2 * added
end

@testset "RBFP0 rejects unsupported neighbour counts" begin
    omega = box((0.0,), (1.0,))
    model = prepare(poisson(space(omega; cells=4, order=2); source=0.0))
    qf = QuadField{Float64}(model; init=q -> 1.0)
    @test_throws ArgumentError transfer(qf, model, model; via=RBFP0(; neighbors=0))
    @test_throws ArgumentError transfer(qf, model, model; via=RBFP0(; neighbors=32))
end
