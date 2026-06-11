using BenchmarkTools
using StaticArrays
using Unfitted: IntegratedLegendre
using Unfitted: _fill_factor_tables!, _tensor_values_grads!, _factor_buffers, local_basis_indices,
                _tensor_gauss_rule, recommended_quadrature_order

group = SUITE["microkernels"]["basis"] = BenchmarkGroup()

for (D, p) in ((1, 4), (2, 2), (2, 4), (3, 2), (3, 3))
    order = ntuple(_ -> p, D)
    basis = IntegratedLegendre()
    indices = local_basis_indices(basis, order)
    val1d = _factor_buffers(order, Float64)
    der1d = _factor_buffers(order, Float64)
    xi = SVector{D,Float64}(ntuple(d -> 0.3 + 0.1d, D))
    scale = SVector{D,Float64}(ntuple(_ -> 2.0, D))
    values = Vector{Float64}(undef, length(indices))
    grads = Vector{SVector{D,Float64}}(undef, length(indices))
    qcounts = recommended_quadrature_order(basis, order)

    tag = "D=$D p=$p"
    group["fill_factor_tables $tag"] = @benchmarkable _fill_factor_tables!($val1d, $der1d, $order,
                                                                           $xi)
    group["tensor_values_grads $tag"] = @benchmarkable _tensor_values_grads!($values, $grads,
                                                                             $indices, $order, $xi,
                                                                             $scale, $val1d, $der1d)
    group["tensor_gauss_rule $tag"] = @benchmarkable _tensor_gauss_rule($qcounts, Float64)
end
