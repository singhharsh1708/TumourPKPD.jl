# Closed-form linear PK with dose superposition.
# Units are whatever the caller uses, as long as they are consistent
# (e.g. hours + mg/kg + L/kg -> mg/L for theophylline; days for PDXE).
# Everything is generic in the number type so ForwardDiff duals pass through.

"""
    Dose(time, amt, route)

One dose of `amt` at `time`. `route` is `:oral` or `:iv` (bolus).
"""
struct Dose
    time::Float64
    amt::Float64
    route::Symbol
end
Dose(time, amt; route::Symbol = :oral) = Dose(float(time), float(amt), route)

"""
    dosing_schedule(amt; start=0, interval=1, n=1, route=:oral)

`n` doses of `amt`, `interval` apart, starting at `start`.
"""
dosing_schedule(amt; start = 0.0, interval = 1.0, n::Integer = 1, route::Symbol = :oral) =
    [Dose(start + (i - 1) * interval, amt, route) for i in 1:n]

abstract type PKModel end

"""
    OneCmt(CL, V, ka, F)

One-compartment model, first-order absorption for oral doses. `ka` and `F`
are ignored for IV bolus doses.
"""
struct OneCmt{T<:Real} <: PKModel
    CL::T
    V::T
    ka::T
    F::T
end
OneCmt(CL, V, ka, F) = OneCmt(promote(CL, V, ka, F)...)
OneCmt(; CL, V, ka = 1.0, F = 1.0) = OneCmt(CL, V, ka, F)

"""
    TwoCmt(CL, V1, Q, V2, ka, F)

Two-compartment model (central V1, peripheral V2, intercompartmental Q).
"""
struct TwoCmt{T<:Real} <: PKModel
    CL::T
    V1::T
    Q::T
    V2::T
    ka::T
    F::T
end
TwoCmt(CL, V1, Q, V2, ka, F) = TwoCmt(promote(CL, V1, Q, V2, ka, F)...)
TwoCmt(; CL, V1, Q, V2, ka = 1.0, F = 1.0) = TwoCmt(CL, V1, Q, V2, ka, F)

# Concentration at time `τ ≥ 0` after a single unit dose.
function unit_conc(pk::OneCmt, route::Symbol, τ)
    k = pk.CL / pk.V
    if route === :iv
        return exp(-k * τ) / pk.V
    end
    ka = pk.ka
    d = ka - k
    if abs(d) < 1e-8 * k
        # ka == k limit
        return pk.F * k * τ * exp(-k * τ) / pk.V
    end
    return pk.F * ka / (pk.V * d) * (exp(-k * τ) - exp(-ka * τ))
end

function twocmt_rates(pk::TwoCmt)
    k10 = pk.CL / pk.V1
    k12 = pk.Q / pk.V1
    k21 = pk.Q / pk.V2
    s = k10 + k12 + k21
    r = sqrt(s^2 - 4 * k10 * k21)
    α = (s + r) / 2
    β = (s - r) / 2
    return α, β, k21
end

function unit_conc(pk::TwoCmt, route::Symbol, τ)
    α, β, k21 = twocmt_rates(pk)
    if route === :iv
        A = (α - k21) / (α - β)
        B = (k21 - β) / (α - β)
        return (A * exp(-α * τ) + B * exp(-β * τ)) / pk.V1
    end
    ka = pk.ka
    # Assumes ka distinct from α and β (true almost surely for sampled values).
    ca = (k21 - α) / ((ka - α) * (β - α))
    cb = (k21 - β) / ((ka - β) * (α - β))
    ck = (k21 - ka) / ((α - ka) * (β - ka))
    return pk.F * ka / pk.V1 * (ca * exp(-α * τ) + cb * exp(-β * τ) + ck * exp(-ka * τ))
end

"""
    conc(pk, doses, t)

Concentration at time `t` by superposition over all doses given at or before `t`.
"""
function conc(pk::PKModel, doses::AbstractVector{Dose}, t::Real)
    c = zero(promote_type(typeof(pk.CL), typeof(float(t))))
    for d in doses
        τ = t - d.time
        τ < 0 && continue
        c += d.amt * unit_conc(pk, d.route, τ)
    end
    return c
end
conc(pk::PKModel, doses::AbstractVector{Dose}, ts::AbstractVector) = [conc(pk, doses, t) for t in ts]
