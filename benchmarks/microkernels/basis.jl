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
    # Basis-family dispatch and the owning cell index are part of the kernel
    # signature: a family with per-cell state (B-splines and their knot spans)
    # needs both. The all-ones index is what the reference-frame entry points in
    # `src/basis.jl` pass when the evaluation is cell-independent.
    cell = CartesianIndex(ntuple(_ -> 1, D))

    tag = "D=$D p=$p"
    group["fill_factor_tables $tag"] = @benchmarkable _fill_factor_tables!($basis, $val1d, $der1d,
                                                                           $order, $xi, $cell)
    group["tensor_values_grads $tag"] = @benchmarkable _tensor_values_grads!($basis, $values,
                                                                             $grads, $indices,
                                                                             $order, $xi, $scale,
                                                                             $val1d, $der1d, $cell)
    group["tensor_gauss_rule $tag"] = @benchmarkable _tensor_gauss_rule($qcounts, Float64)
end
