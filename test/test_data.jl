using StaticArrays
using LinearAlgebra

@testset "QuadField construction and stale-version contract" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    model = prepare(poisson(space(omega; cells=(2, 2), order=1); source=x -> 0.0))

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
    model = prepare(poisson(space(omega; cells=2, order=1); source=x -> 0.0))
    qf = QuadField{Float64}(model)
    @test all(==(0.0), qf.data)
    qf[1] = 3.5
    @test qf[1] == 3.5
    q2 = copy(qf)
    q2[1] = 1.0
    @test qf[1] == 3.5   # original untouched
end

@testset "RBFP0 reproduces constant fields exactly (Sartorti & Düster Tab. 3)" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    source_model = prepare(poisson(space(omega; cells=(3, 3), order=2); source=x -> 0.0))
    target_model = prepare(poisson(space(omega; cells=(5, 5), order=1); source=x -> 0.0))

    source = QuadField{Float64}(source_model; init=q -> 7.25)
    target = transfer(source, source_model, target_model, RBFP0(; neighbors=10))

    @test all(isapprox.(target.data, 7.25; atol=1.0e-10))
    @test target.model_version == target_model.version
end

@testset "RBFP0 reproduces linear fields exactly via the polynomial extension" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    source_model = prepare(poisson(space(omega; cells=(3, 3), order=2); source=x -> 0.0))
    target_model = prepare(poisson(space(omega; cells=(5, 5), order=1); source=x -> 0.0))

    f = x -> 1.0 + 0.5 * x[1]
    source = QuadField{Float64}(source_model; init=q -> f(q.x))
    target = transfer(source, source_model, target_model, RBFP0(; neighbors=12))

    Unfitted.foreach_quadrature_point(target_model) do q
        @test isapprox(target.data[q.point], f(q.x); atol=5.0e-3)
    end
end

@testset "RBFP0 threaded and serial paths are bit-identical" begin
    omega = box((0.0, 0.0), (1.0, 1.0))
    source_model = prepare(poisson(space(omega; cells=(4, 4), order=2); source=x -> 0.0))
    target_model = prepare(poisson(space(omega; cells=(6, 6), order=1); source=x -> 0.0))

    source = QuadField{Float64}(source_model; init=q -> exp(-((q.x[1] - 0.5)^2 + (q.x[2] - 0.5)^2)))

    serial = transfer(source, source_model, target_model, RBFP0(); threaded=false)
    threaded = transfer(source, source_model, target_model, RBFP0(); threaded=true)

    @test serial.data == threaded.data
end

@testset "RBFP0 rejects unsupported neighbour counts" begin
    omega = box((0.0,), (1.0,))
    model = prepare(poisson(space(omega; cells=4, order=2); source=x -> 0.0))
    qf = QuadField{Float64}(model; init=q -> 1.0)
    @test_throws ArgumentError transfer(qf, model, model, RBFP0(; neighbors=0))
    @test_throws ArgumentError transfer(qf, model, model, RBFP0(; neighbors=32))
end
