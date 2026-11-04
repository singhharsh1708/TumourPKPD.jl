# ModelingToolkit (v11) version of the Simeoni tumour model, used ONLY to cross-check
# simulate_tumour (see test/test_pd_mtk.jl); not on any inference path.
#
# Two variants, both with a time-dependent drug input:
#   mtk_simeoni_pk        oral one-compartment PK (depot Ad, central Ac) and the four
#                         Simeoni states in ONE system; exposure e = (Ac/V)/cref;
#                         doses are discrete events (Ad += F*amt). A dose at t <= 0 is
#                         folded into the initial depot amount.
#   mtk_simeoni_registered  exposure supplied by a registered symbolic function
#                         `pd_exposure(t)` evaluating the closed-form `conc` of src/pk.jl
#                         (state is held in a Ref so a compiled system can be reused).

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq

# --- shared Simeoni equations -------------------------------------------------------
function _simeoni_eqs(e)
    @parameters λ0 λ1 k1 k2 ψ
    @variables x1(t) x2(t) x3(t) x4(t) w(t)
    kill = k2 * e * x1
    eqs = [
        D(x1) ~ λ0 * x1 * (1 + (λ0 * w / λ1)^ψ)^(-1 / ψ) - kill,
        D(x2) ~ kill - k1 * x2,
        D(x3) ~ k1 * (x2 - x3),
        D(x4) ~ k1 * (x3 - x4),
        w ~ x1 + x2 + x3 + x4,
    ]
    return eqs, (; λ0, λ1, k1, k2, ψ, x1, x2, x3, x4, w)
end

"""
    mtk_simeoni_pk(doses; F=1.0, name=:simeoni_pk)

Compiled system: oral 1-compartment PK + Simeoni in one ODE system.
Returns `(sys, a0, later)`.
"""
function mtk_simeoni_pk(doses; F = 1.0, name = :simeoni_pk)
    a0 = 0.0
    later = Tuple{Float64,Float64}[]
    for d in doses
        d.route === :oral || error("mtk_simeoni_pk models oral doses only")
        d.time <= 0 ? (a0 += F * d.amt) : push!(later, (d.time, F * d.amt))
    end
    @parameters CL V ka cref
    @variables Ad(t) Ac(t) e(t)
    pkeqs = [D(Ad) ~ -ka * Ad, D(Ac) ~ ka * Ad - CL / V * Ac, e ~ Ac / V / cref]
    eqs, s = _simeoni_eqs(e)
    events = [[tt] => [Ad ~ Pre(Ad) + a] for (tt, a) in later]
    @named raw = System([pkeqs; eqs], t; discrete_events = events)
    sys = mtkcompile(raw)
    return (; sys, a0, later)
end

"""
    mtk_simulate_pk(built, θ::Simeoni, pk::OneCmt, cref, tobs; reltol, abstol)

Total tumour volume at `tobs` from the combined PK + Simeoni MTK system.
"""
function mtk_simulate_pk(built, θ, pk::OneCmt, cref, tobs; reltol = 1e-10, abstol = 1e-10)
    sys = built.sys
    u0 = Dict(sys.Ad => built.a0, sys.Ac => 0.0, sys.x1 => θ.w0, sys.x2 => 0.0,
              sys.x3 => 0.0, sys.x4 => 0.0)
    ps = Dict(sys.CL => pk.CL, sys.V => pk.V, sys.ka => pk.ka, sys.cref => cref,
              sys.λ0 => θ.λ0, sys.λ1 => θ.λ1, sys.k1 => θ.k1, sys.k2 => θ.k2, sys.ψ => θ.ψ)
    prob = ODEProblem(sys, merge(u0, ps), (0.0, maximum(tobs)))
    sol = solve(prob, Vern9(); reltol, abstol)
    return [sol(tt; idxs = sys.w) for tt in tobs]
end

# --- registered closed-form exposure ------------------------------------------------
const PD_EXPOSURE = Ref{Any}(nothing)
pd_exposure(tt) = PD_EXPOSURE[](tt)
@register_symbolic pd_exposure(t)

"""
    mtk_simeoni_registered(; name=:simeoni_reg)

Compiled Simeoni system whose exposure is the registered function `pd_exposure(t)`.
Set `PD_EXPOSURE[]` to any callable (e.g. a `PKExposure`) before solving. Pass the
dose times as `tstops` when solving.
"""
function mtk_simeoni_registered(; name = :simeoni_reg)
    eqs, s = _simeoni_eqs(pd_exposure(t))
    @named raw = System(eqs, t)
    return (; sys = mtkcompile(raw))
end

function mtk_simulate_registered(built, θ, exposure, tobs; reltol = 1e-10, abstol = 1e-10)
    sys = built.sys
    PD_EXPOSURE[] = exposure
    u0 = Dict(sys.x1 => θ.w0, sys.x2 => 0.0, sys.x3 => 0.0, sys.x4 => 0.0)
    ps = Dict(sys.λ0 => θ.λ0, sys.λ1 => θ.λ1, sys.k1 => θ.k1, sys.k2 => θ.k2, sys.ψ => θ.ψ)
    tend = maximum(tobs)
    prob = ODEProblem(sys, merge(u0, ps), (0.0, tend))
    stops = filter(s -> 0 < s < tend, TumourPKPD.dose_times(exposure))
    sol = solve(prob, Vern9(); reltol, abstol, tstops = stops)
    return [sol(tt; idxs = sys.w) for tt in tobs]
end
