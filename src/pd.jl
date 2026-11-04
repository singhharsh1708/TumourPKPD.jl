# Simeoni 2004 tumour growth inhibition model.
#   x1: proliferating cells, x2..x4: damaged cells on their way to death.
#   dx1 = λ0 x1 / (1 + (λ0 w / λ1)^ψ)^(1/ψ) - k2 e(t) x1
#   dx2 = k2 e(t) x1 - k1 x2
#   dx3 = k1 (x2 - x3)
#   dx4 = k1 (x3 - x4)
#   w = x1 + x2 + x3 + x4 (total tumour volume)
# e(t) is the drug exposure driving the kill term (a concentration, or a
# concentration normalised by a reference level; see PKExposure).

"""
    Simeoni(λ0, λ1, k1, k2, w0; ψ=20)

λ0: exponential growth rate (1/time), λ1: linear growth rate (volume/time),
k1: transit rate of damaged cells (1/time), k2: potency (1/(exposure·time)),
w0: initial tumour volume.
"""
struct Simeoni{T<:Real}
    λ0::T
    λ1::T
    k1::T
    k2::T
    w0::T
    ψ::T
end
function Simeoni(λ0, λ1, k1, k2, w0; ψ = 20.0)
    v = promote(λ0, λ1, k1, k2, w0, ψ)
    return Simeoni(v...)
end

"Exposure that is always zero (untreated arm)."
struct NoDrug end
(::NoDrug)(t) = 0.0
dose_times(::NoDrug) = Float64[]

"""
    PKExposure(pk, doses, cref)

Exposure e(t) = conc(pk, doses, t) / cref. `cref` sets the scale on which k2
is reported (e.g. Cmax of the reference regimen), so k2 stays comparable
across PK scenarios.
"""
struct PKExposure{P<:PKModel,T<:Real}
    pk::P
    doses::Vector{Dose}
    cref::T
end
PKExposure(pk, doses) = PKExposure(pk, doses, 1.0)
(e::PKExposure)(t) = conc(e.pk, e.doses, t) / e.cref
dose_times(e::PKExposure) = [d.time for d in e.doses]

# Smooth saturation factor 1 / (1 + r^ψ)^(1/ψ), computed in log space so large
# tumours do not overflow.
@inline function growth_factor(r, ψ)
    r <= 0 && return one(r)
    return exp(-log1pexp(ψ * log(r)) / ψ)
end

function simeoni_rhs!(du, u, p, t)
    θ, expo = p
    x1, x2, x3, x4 = u
    w = x1 + x2 + x3 + x4
    kill = θ.k2 * expo(t) * x1
    du[1] = θ.λ0 * x1 * growth_factor(θ.λ0 * w / θ.λ1, θ.ψ) - kill
    du[2] = kill - θ.k1 * x2
    du[3] = θ.k1 * (x2 - x3)
    du[4] = θ.k1 * (x3 - x4)
    return nothing
end

"""
    simulate_tumour(θ, exposure, tobs; solver=Tsit5(), reltol=1e-6, abstol=1e-6)

Total tumour volume w(t) at times `tobs` (sorted, ≥ 0; the model starts at
t = 0 with w = w0, all in x1). Returns `(w, ok)`; `ok` is false when the
solver failed, in which case `w` must not be used.
"""
function simulate_tumour(θ::Simeoni, exposure, tobs::AbstractVector;
                         solver = Tsit5(), reltol = 1e-6, abstol = 1e-6, maxiters = 100_000)
    T = promote_type(typeof(θ.λ0), typeof(_exposure_eltype(exposure)))
    n = length(tobs)
    n == 0 && return T[], true
    bad = (fill(T(NaN), n), false)
    issorted(tobs) && first(tobs) >= 0 || throw(ArgumentError("tobs must be sorted and ≥ 0"))
    all(x -> isfinite(x) && x > 0, (θ.λ0, θ.λ1, θ.k1, θ.w0)) && isfinite(θ.k2) && θ.k2 >= 0 || return bad
    ts = unique(float.(tobs))
    tend = last(ts)
    if tend == 0
        return fill(T(θ.w0), n), true
    end
    u0 = T[θ.w0, 0, 0, 0]
    stops = filter(s -> 0 < s < tend, dose_times(exposure))
    prob = ODEProblem{true}(simeoni_rhs!, u0, (0.0, tend), (θ, exposure))
    sol = solve(prob, solver; saveat = ts, tstops = stops, reltol, abstol, maxiters,
                save_everystep = false, dense = false, verbose = false)
    SciMLBase.successful_retcode(sol) || return bad
    length(sol.u) == length(ts) || return bad
    wu = [sum(u) for u in sol.u]
    all(isfinite, wu) || return bad
    # map back onto the requested times (handles duplicates)
    w = Vector{eltype(wu)}(undef, n)
    k = 1
    for (i, t) in enumerate(tobs)
        while ts[k] != t
            k += 1
        end
        w[i] = wu[k]
    end
    return w, true
end

_exposure_eltype(::NoDrug) = 0.0
_exposure_eltype(e::PKExposure) = e.pk.CL * e.cref
