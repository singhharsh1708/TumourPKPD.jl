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

Doses are sorted by time at construction. For `OneCmt` (oral and/or IV doses)
e(t) is evaluated from a precomputed state at the last dose time ≤ t
(two `exp` calls and a binary search per call, whatever the number of doses);
other models, and near-equal absorption/elimination rates, use a sorted loop
over the doses at or before t. Results agree with `conc(pk, doses, t) / cref`.
"""
struct PKExposure{P<:PKModel,T<:Real,C}
    pk::P
    doses::Vector{Dose}      # sorted by time
    cref::T
    times::Vector{Float64}   # distinct dose times, sorted
    cache::C                 # `nothing` or a _OneCmtState
end
function PKExposure(pk::PKModel, doses::AbstractVector{Dose}, cref::Real)
    ds = sort(collect(Dose, doses); by = d -> d.time)
    times = unique(d.time for d in ds)
    return PKExposure(pk, ds, cref, times, _exposure_cache(pk, ds, times, cref))
end
PKExposure(pk, doses) = PKExposure(pk, doses, 1.0)

# One-compartment state after each distinct dose time tᵢ: for t in [tᵢ, tᵢ₊₁),
#   e(t) = a[i] exp(-k (t - tᵢ)) + b[i] exp(-ka (t - tᵢ)),   divided by cref.
struct _OneCmtState{T}
    a::Vector{T}
    b::Vector{T}
    k::T
    ka::T
end

_exposure_cache(pk, ds, times, cref) = nothing
function _exposure_cache(pk::OneCmt, ds, times, cref)
    k = pk.CL / pk.V
    ka = pk.ka
    # 1/(ka - k) amplifies rounding when the two rates nearly coincide: use the sorted loop there
    any(d -> d.route === :oral, ds) && !(abs(ka - k) > 1e-3 * max(k, ka)) && return nothing
    T = promote_type(typeof(k), typeof(ka), typeof(pk.F), typeof(float(cref)))
    n = length(times)
    a = zeros(T, n)
    b = zeros(T, n)
    ai = zero(T); bi = zero(T)
    j = 1
    for (i, ti) in enumerate(times)
        if i > 1
            Δ = ti - times[i-1]
            ai *= exp(-k * Δ)
            bi *= exp(-ka * Δ)
        end
        while j <= length(ds) && ds[j].time == ti
            d = ds[j]
            if d.route === :iv
                ai += d.amt / pk.V
            else
                c0 = pk.F * d.amt * ka / (pk.V * (ka - k))
                ai += c0
                bi -= c0
            end
            j += 1
        end
        a[i] = ai / cref
        b[i] = bi / cref
    end
    return _OneCmtState{T}(a, b, T(k), T(ka))
end

function (e::PKExposure{P,T,Nothing})(t) where {P,T}
    c = zero(promote_type(typeof(e.pk.CL), typeof(float(t))))
    for d in e.doses
        τ = t - d.time
        τ < 0 && break
        c += d.amt * unit_conc(e.pk, d.route, τ)
    end
    return c / e.cref
end
function (e::PKExposure{P,T,<:_OneCmtState})(t) where {P,T}
    s = e.cache
    i = searchsortedlast(e.times, t)
    i == 0 && return zero(s.k) * t
    τ = t - e.times[i]
    return s.a[i] * exp(-s.k * τ) + s.b[i] * exp(-s.ka * τ)
end
dose_times(e::PKExposure) = e.times

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
