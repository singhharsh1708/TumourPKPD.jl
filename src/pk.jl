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

# φ(x) = (1 - exp(-x)) / x, with its x -> 0 limit. Smooth in x, so gradients
# stay correct when two rate constants are (nearly) equal.
@inline function phi_rel(x)
    if abs(x) < 1e-4
        return 1 - x / 2 + x^2 / 6 - x^3 / 24
    end
    return -expm1(-x) / x
end

# ∫₀^τ ka e^{-ka s} e^{-λ (τ - s)} ds = ka τ e^{-λ τ} φ((ka - λ) τ)
@inline absorbed(ka, λ, τ) = ka * τ * exp(-λ * τ) * phi_rel((ka - λ) * τ)

# Concentration at time `τ ≥ 0` after a single unit dose.
function unit_conc(pk::OneCmt, route::Symbol, τ)
    k = pk.CL / pk.V
    route === :iv && return exp(-k * τ) / pk.V
    return pk.F * absorbed(pk.ka, k, τ) / pk.V
end

function twocmt_rates(pk::TwoCmt)
    k10 = pk.CL / pk.V1
    k12 = pk.Q / pk.V1
    k21 = pk.Q / pk.V2
    # s^2 - 4 k10 k21 written as a sum of non-negative terms (no cancellation)
    r = sqrt((k10 - k21)^2 + k12 * (k12 + 2 * (k10 + k21)))
    α = (k10 + k12 + k21 + r) / 2
    β = k10 * k21 / α          # avoids losing the slow rate to cancellation
    return α, β, k21
end

# IV unit-dose disposition: (A e^{-ατ} + B e^{-βτ}) / V1. When α ≈ β the
# system is effectively one-compartment with rate k10 (requires Q ≈ 0).
function twocmt_coefs(pk::TwoCmt)
    α, β, k21 = twocmt_rates(pk)
    if α - β <= 1e-12 * α
        return one(α), zero(α), α, β
    end
    A = (α - k21) / (α - β)
    B = (k21 - β) / (α - β)
    return A, B, α, β
end

function unit_conc(pk::TwoCmt, route::Symbol, τ)
    A, B, α, β = twocmt_coefs(pk)
    if route === :iv
        return (A * exp(-α * τ) + B * exp(-β * τ)) / pk.V1
    end
    # Oral = IV disposition convolved with first-order absorption, term by term.
    return pk.F * (A * absorbed(pk.ka, α, τ) + B * absorbed(pk.ka, β, τ)) / pk.V1
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
