using Unfitted
using Test

@testset verbose=true "Unfitted" begin
    include("test_geometry.jl")
    include("test_basis.jl")
    include("test_mesh.jl")
    include("test_intersections.jl")
    include("test_constraints.jl")
    include("test_activation.jl")
    include("test_physical.jl")
    include("test_fcm.jl")
    include("test_assembly.jl")
    include("test_projection.jl")
    include("test_data.jl")
    include("test_postprocessing.jl")
    include("test_boundary_mesh.jl")
    include("test_vtk.jl")
    include("test_regressions.jl")
    include("test_api.jl")
    include("test_tensors_ext.jl")
    include("test_basis_bspline.jl")
end
