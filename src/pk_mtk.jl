# ModelingToolkit (v11) versions of the oral 1- and 2-compartment models, with a
# depot compartment. Used ONLY to cross-check the closed-form solutions in pk.jl
# (see test/test_pk.jl); they are not on any inference path.
#
# Amounts are tracked in the depot (Ad), central (Ac) and peripheral (Ap)
# compartments; the observed concentration is C = Ac / V (V1 for two-compartment).
# Doses are discrete events (Ad += F * amt). A dose at time 0 is folded into
# the initial condition; negative dose times are rejected (the model starts at t = 0
# with empty compartments, so earlier doses would need their history propagated).

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq

# Split doses into an initial depot load and (times, amounts) of later events.
function _split_oral_doses(doses, F)
    a0 = 0.0
    later = Tuple{Float64,Float64}[]
    for d in doses
        d.route === :oral || error("pk_mtk models oral doses only")
        d.time < 0 && throw(ArgumentError("dose time $(d.time) < 0: model starts at t = 0 with empty compartments"))
        if d.time == 0
            a0 += F * d.amt
        else
            push!(later, (d.time, F * d.amt))
        end
    end
    return a0, later
end

function _dose_events(Ad, later)
    return [[tt] => [Ad ~ Pre(Ad) + a] for (tt, a) in later]
end

"""
    mtk_onecmt(doses; name=:onecmt)

Compiled MTK system for the oral one-compartment model. Returns
`(sys, a0, later)`; use [`mtk_conc`](@ref) to evaluate it.
"""
function mtk_onecmt(doses; F = 1.0, name = :onecmt)
    a0, later = _split_oral_doses(doses, F)
    @parameters CL V ka
    @variables Ad(t) Ac(t) C(t)
    eqs = [
        D(Ad) ~ -ka * Ad,
        D(Ac) ~ ka * Ad - CL / V * Ac,
        C ~ Ac / V,
    ]
    @named raw = System(eqs, t; discrete_events = _dose_events(Ad, later))
    sys = mtkcompile(raw)
    return (; sys, a0, later)
end

"""
    mtk_twocmt(doses; F=1.0, name=:twocmt)

Compiled MTK system for the oral two-compartment model (central V1, peripheral V2).
"""
function mtk_twocmt(doses; F = 1.0, name = :twocmt)
    a0, later = _split_oral_doses(doses, F)
    @parameters CL V1 Q V2 ka
    @variables Ad(t) Ac(t) Ap(t) C(t)
    eqs = [
        D(Ad) ~ -ka * Ad,
        D(Ac) ~ ka * Ad - (CL / V1 + Q / V1) * Ac + Q / V2 * Ap,
        D(Ap) ~ Q / V1 * Ac - Q / V2 * Ap,
        C ~ Ac / V1,
    ]
    @named raw = System(eqs, t; discrete_events = _dose_events(Ad, later))
    sys = mtkcompile(raw)
    return (; sys, a0, later)
end

"""
    mtk_conc(built, pk, ts; reltol=1e-10, abstol=1e-12)

Solve the MTK model built by `mtk_onecmt` / `mtk_twocmt` for parameters taken
from `pk` (an `OneCmt` / `TwoCmt` with matching structure) and return the
concentration at the sorted times `ts`.
"""
function mtk_conc(built, pk::OneCmt, ts; reltol = 1e-10, abstol = 1e-12)
    sys = built.sys
    u0 = [sys.Ad => built.a0, sys.Ac => 0.0]
    ps = [sys.CL => pk.CL, sys.V => pk.V, sys.ka => pk.ka]
    prob = ODEProblem(sys, merge(Dict(u0), Dict(ps)), (0.0, maximum(ts)))
    sol = solve(prob, Vern9(); reltol, abstol)
    return [sol(tt; idxs = sys.C) for tt in ts]
end

function mtk_conc(built, pk::TwoCmt, ts; reltol = 1e-10, abstol = 1e-12)
    sys = built.sys
    u0 = [sys.Ad => built.a0, sys.Ac => 0.0, sys.Ap => 0.0]
    ps = [sys.CL => pk.CL, sys.V1 => pk.V1, sys.Q => pk.Q, sys.V2 => pk.V2, sys.ka => pk.ka]
    prob = ODEProblem(sys, merge(Dict(u0), Dict(ps)), (0.0, maximum(ts)))
    sol = solve(prob, Vern9(); reltol, abstol)
    return [sol(tt; idxs = sys.C) for tt in ts]
end
