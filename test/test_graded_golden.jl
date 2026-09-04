using Unfitted
using Unfitted: cell_basis_indices, cell_order, dof_layout, local_basis_indices, cell_indices,
                is_active

# Golden values for a level carrying a PER-CELL polynomial order.
#
# WHY THIS FILE EXISTS. `test/characterize.jl` is the repository's golden
# instrument, and it is blind to this feature: all 46 of its `order =` arguments
# are scalars, so it never builds a graded level. Its report was measured
# byte-identical against five deliberately broken minimum rules — including the
# rule deleted outright — so "characterize is unchanged" says nothing whatever
# about per-cell order. Anything that rewrites the graded kernel needs an
# instrument that can see it, and this is that instrument.
#
# WHAT IT LOCKS. The object the minimum rule actually computes: for every cell of
# every configuration, the exact multi-index set the level generates, paired with
# the raw dof ids the layout assigns to it. Not a downstream aggregate — an error
# norm or an unknown count can absorb a wrong mode set and stay plausible.
#
# HOW IT LOCKS IT. A 64-bit FNV-1a checksum over the integers themselves, not
# over formatted text and not through `Base.hash`, whose value carries no
# stability guarantee across Julia versions. The inputs are multi-indices and dof
# ids, so the digest is exactly reproducible on any machine and any version.
#
# WHEN IT FAILS. A moved digest means the per-cell mode sets moved. That is
# either the bug you are hunting or the improvement you are making, and the named
# assertions below run first so a failure usually arrives with a diagnosis rather
# than only a number.

const _FNV_PRIME = 0x00000100000001b3
const _FNV_BASIS = 0xcbf29ce484222325

function _mix(h::UInt64, x::Integer)
    v = UInt64(unsigned(Int64(x)) & typemax(UInt64))
    for shift in 0:8:56
        h = (h ⊻ ((v >> shift) & 0xff)) * _FNV_PRIME
    end
    return h
end

# The graded fields. Each is a different way for orders to vary, because the
# minimum rule only does anything where they differ, and the shapes that break it
# are not the same: `separable` in particular varies along one axis only, so no
# cell has a lower neighbour on an axis it shares an index with — the case a
# conservative "does this cell touch a jump" filter gets wrong.
function _blocky(c, lo, hi)
    [clamp(lo + (sum(Tuple(i)) - length(c)) ÷ 8, lo, hi) for i in CartesianIndices(c)]
end
_fine(c, lo, hi) = [lo + (sum(Tuple(i)) - length(c)) % (hi - lo + 1) for i in CartesianIndices(c)]
_onehot(c, lo, hi) = (a=fill(lo, c); a[CartesianIndex(cld.(c, 2))]=hi; a)
function _band(c, lo, hi)
    mid = ntuple(d -> c[d] / 2, length(c))
    r = minimum(c) / 6
    return [sum((Tuple(i)[d] - mid[d])^2 for d in 1:length(c)) < r^2 ? hi : lo
            for i in CartesianIndices(c)]
end
function _aniso(c, lo, hi)
    D = length(c)
    return [ntuple(d -> lo + ((Tuple(i)[d] + 2 * Tuple(i)[mod1(d + 1, D)]) ÷ 3) % (hi - lo + 1), D)
            for i in CartesianIndices(c)]
end
function _separable(c, lo, hi)
    D = length(c)
    return [ntuple(d -> lo + (Tuple(i)[d] ÷ 3) % (hi - lo + 1), D) for i in CartesianIndices(c)]
end

_halfmask(c) = (m=trues(c); m[1:cld(c[1], 2), ntuple(_ -> :, length(c) - 1)...].=false; m)
function _ballmask(c)
    mid = ntuple(d -> c[d] / 2, length(c))
    r = minimum(c) / 3
    return [sum((Tuple(i)[d] - mid[d])^2 for d in 1:length(c)) > r^2 for i in CartesianIndices(c)]
end
_sparsemask(c) = (m=trues(c); for (i, ci) in enumerate(CartesianIndices(c))
                      i % 7 == 0 && (m[ci] = false)
                  end; m)

# Fold one level into the digest, and check the invariants that give a moved
# digest a name. `level` is a position, as everywhere else in the public API.
function _absorb!(state, V, tag::Int; level::Integer=1)
    lvl = V.levels[level]
    layout = dof_layout(V)
    cells = layout.cell_dofs_by_level[level]
    h = _mix(state[], tag)
    for (n, ci) in enumerate(cell_indices(V; level=level))
        ids = cell_basis_indices(lvl, ci)
        dofs = cells[ci]
        # (1) The pairing the whole dof↔basis contract rests on. Every consumer
        #     that walks these two positionally is wrong the moment it breaks.
        @test length(ids) == length(dofs)
        # (2) The minimum rule only ever REMOVES. A kept mode must be one the
        #     cell's own order generates; anything else is a mode from nowhere.
        if !isempty(ids)
            own = Set(local_basis_indices(lvl.basis, cell_order(lvl, ci), lvl.mode))
            @test all(in(own), ids)
        end
        # (3) An inactive cell generates nothing.
        is_active(lvl.mask, ci) || @test isempty(ids)
        h = _mix(h, n)
        h = _mix(h, length(ids))
        for id in ids, d in Tuple(id)
            h = _mix(h, d)
        end
        for raw in dofs
            h = _mix(h, raw)
        end
    end
    h = _mix(h, length(layout.raw_keys))
    h = _mix(h, layout.active_count)
    state[] = h
    return nothing
end

@testset "graded golden: the per-cell mode sets are locked" begin
    state = Ref(_FNV_BASIS)
    tag = Ref(0)
    next!() = (tag[] += 1)
    Ω(D) = box(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D))

    C1, C2, C3 = (32,), (16, 16), (8, 8, 8)
    for f in (_blocky, _fine, _band, _onehot, _aniso, _separable)
        # `_aniso` and `_separable` return tuples, which 1D cannot express as a
        # distinct case, and `:trunk` requires isotropy per cell.
        isotropic = f !== _aniso && f !== _separable
        isotropic && _absorb!(state, space(Ω(1); cells=C1, order=f(C1, 1, 5)), next!())
        _absorb!(state, space(Ω(2); cells=C2, order=f(C2, 1, 4)), next!())
        _absorb!(state, space(Ω(3); cells=C3, order=f(C3, 1, 4)), next!())
        if isotropic
            _absorb!(state, space(Ω(2); cells=C2, order=f(C2, 1, 4), mode=:trunk), next!())
            _absorb!(state, space(Ω(3); cells=C3, order=f(C3, 1, 5), mode=:trunk), next!())
        end
        _absorb!(state, space(Ω(2); cells=C2, order=f(C2, 1, 4), active=_halfmask(C2)), next!())
        _absorb!(state, space(Ω(2); cells=C2, order=f(C2, 1, 4), active=_sparsemask(C2)), next!())
        _absorb!(state, space(Ω(3); cells=C3, order=f(C3, 1, 4), active=_ballmask(C3)), next!())
    end

    # The rebuild path. `adapt` onto a mask must agree cell-for-cell with a fresh
    # build at that mask — this is the invariant the mask witness was added for,
    # and it is the one a derived table gets wrong.
    @testset "adapt agrees with a fresh build at the same mask" begin
        field = _blocky(C2, 1, 4)
        V = space(Ω(2); cells=C2, order=field)
        m = trues(C2)
        for k in 1:12
            m[CartesianIndex(1 + (3k) % C2[1], 1 + (5k) % C2[2])] = false
            V = adapt(V, 1 => copy(m))
            fresh = space(Ω(2); cells=C2, order=field, active=copy(m))
            for ci in cell_indices(V; level=1)
                @test cell_basis_indices(V.levels[1], ci) == cell_basis_indices(fresh.levels[1], ci)
            end
            _absorb!(state, V, next!())
        end
    end

    # Overlay stacks, and `elevate` on a level that is not the base.
    V = overlay(space(Ω(2); cells=(16, 16), order=_blocky((16, 16), 1, 4)),
                box((0.25, 0.25), (0.75, 0.75)); cells=(16, 16), order=3)
    _absorb!(state, V, next!())
    _absorb!(state, V, next!(); level=2)
    _absorb!(state, elevate(V, 2 => 6), next!(); level=2)
    _absorb!(state, elevate(V, 1 => _blocky((16, 16), 2, 5)), next!())

    # The fictitious fold, which re-masks the level underneath the user.
    disc = physical_domain(x -> sum(abs2, x .- 0.5) - 0.16; lipschitz=2.0, alpha=1e-8,
                           subcell_length_scale=0.05)
    _absorb!(state, space(Ω(2); cells=(8, 8), order=_blocky((8, 8), 1, 4), physical=disc), next!())
    _absorb!(state, space(Ω(2); cells=(8, 8), order=_aniso((8, 8), 1, 4), physical=disc), next!())
    _absorb!(state,
             space(Ω(2); cells=(8, 8), order=_blocky((8, 8), 1, 4), mode=:trunk, physical=disc),
             next!())

    # The digest. It covers every configuration above; a change here means the
    # per-cell mode sets moved, which is either the defect you are hunting or the
    # improvement you are making. It is not a value to update casually — see the
    # header, and check the named assertions above first.
    @test tag[] == 61
    @test state[] == 0xcd4398f3391102f8
end
