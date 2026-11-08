# Hierarchical Simeoni tumour growth inhibition model, fitted per drug with Turing/NUTS.
#
# Included from src/TumourPKPD.jl AFTER pk.jl and pd.jl (uses Dose, OneCmt, Simeoni,
# NoDrug, PKExposure, simulate_tumour, dose_times from those files).
#
# Unit = PDX model; each has a control curve (NoDrug) and a treated curve (PKExposure).
#
#   logλ0_i ~ N(μ_λ0, ω_λ0)  (sampled centred; or μ_λ0 + ω_λ0 z_λ0_i non-centred)  shared by both arms of unit i
#   logk2_i = μ_k2 + ω_k2 z_k2_i
#   logVtr (λ1 = exp(μ_λ0)·Vtr), logk1, σ  drug level / pooled
#   w0 per curve: fixed at the observed day-0 volume (default) or latent, see `latent_w0`
#   log v_ij ~ Normal(log w_i(t_ij), σ)    v > 0
#   v_ij == 0 -> left-censored: P(log w + σ ε < log LOD) = Φ((log LOD - log w)/σ)
#
# The day-0 observation is not part of the likelihood in the fixed-w0 mode (it sets w0).

using Turing
using Distributions
using ForwardDiff
using JLD2
using MCMCChains
using Random, Statistics
using LinearAlgebra: I
using LogExpFunctions: logsumexp
import Optim
const DynamicPPL = Turing.DynamicPPL
const LogDensityProblems = Turing.LogDensityProblems

export PosteriorDraws, split_modes, target_fingerprint, fingerprint_text, verify_fingerprint, frozen_pdx_split, design_units, map_modes, distinct_modes, stage_target, stage_warmup, stage_sample, stage_layout,
       stage_chain, constrain_draws, unconstrain, stage_seed, window_units, map_init, multistart_map, chain_to_parts, chain_from_parts, valid_baseline, prepare_units, cache_stats, save_fit, load_fit, fit_drug, pointwise_loglik, pointwise_index,
       grouped_loglik, HierPriors, default_priors, hier_model, prepare_data,
       population_loglik, simulate_units, reference_cref, SynthUnit, SynthCurve, marginal_unit_loglik

# ---------------------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------------------

"""
    HierPriors

All prior hyper-parameters (log scale unless stated). See `default_priors()` and the
inference report for the justification of every number.

- `μ_λ0 ~ Normal(m, s)`            log growth rate (1/day)
- `ω_λ0 ~ HalfNormal(s)`           between-PDX sd of logλ0
- `μ_k2 ~ Normal(m, s)`            log potency on cref-normalised exposure (1/day)
- `ω_k2 ~ HalfNormal(s)`
- `logVtr ~ Normal(m, s)`          log transition volume (mm^3); λ1 = exp(μ_λ0) · Vtr  (log λ1 = μ_λ0 + logVtr)
- `logk1 ~ Normal(m, s)`           log transit rate (1/day)
- `σ ~ HalfNormal(s)`              log-volume residual sd

Model v2 only (`model=:v2`):
- `lkj_eta`                        LKJ(η) prior on the 2×2 correlation of (log λ0, log k2): (ρ+1)/2 ~ Beta(η, η) (η = 2: LKJ(2))
- `ω_Vtr ~ HalfNormal(s)`          between-PDX sd of log transition volume (per-PDX Vtr_i, λ1_i = λ0_i Vtr_i); logVtr's prior then
                                   describes the population mean μ_Vtr. s = 0.5: the across-PDX sd of log max-volume of the
                                   controls is about 0.5, and the pooled-logVtr prior sd is 0.7, so ω_Vtr must stay below that
- `logρ_w ~ Normal(m, s)`          (waning=true) log waning rate of the kill term, k2 exp(-ρ t); centred on a 45-day half-life
                                   (ρ = ln2/45 = 0.0154/day), sd 0.5 (half-lives of about 27-75 days at ±1 sd)
"""
Base.@kwdef struct HierPriors
    μ_λ0::NTuple{2,Float64} = (log(0.05), 0.5)    # strict prior-predictive check (frozen training PDX, days <= 21): passes; see scripts/prior_predictive.jl
    ω_λ0::Float64 = 0.5
    μ_k2::NTuple{2,Float64} = (log(0.15), 1.3)
    ω_k2::Float64 = 1.0
    logVtr::NTuple{2,Float64} = (log(1000.0), 0.7)
    logk1::NTuple{2,Float64} = (log(0.7), 0.6)
    σ::Float64 = 0.3
    lkj_eta::Float64 = 2.0
    ω_Vtr::Float64 = 0.5
    logρ_w::NTuple{2,Float64} = (log(log(2.0) / 45.0), 0.5)
end
default_priors() = HierPriors()

# ---------------------------------------------------------------------------------------
# Data preparation (duck-typed on the data contract: unit.model, unit.control.t/.v, unit.treated.t/.v)
# ---------------------------------------------------------------------------------------

"One prepared curve: observation times > 0, log volumes, censoring flags, observed day-0 volume."
struct CurveData
    t::Vector{Float64}
    logv::Vector{Float64}      # log volume; 0.0 placeholder where censored
    cens::Vector{Bool}
    v0::Float64                # observed day-0 volume (mm^3), > 0
end

struct UnitData
    model::String
    control::CurveData
    treated::CurveData
end

_default_lod() = isdefined(@__MODULE__, :detection_limit) ? detection_limit() : 4.0

"""
    valid_baseline(curve) -> Bool

True when the curve's earliest observation is at day 0 with a finite, strictly positive volume.
A zero, missing or non-finite day-0 value is NOT replaced by anything: such units are dropped
(`prepare_units`), because a fabricated baseline would be treated as an observed one.
"""
function valid_baseline(tc)
    isempty(tc.t) && return false
    t = collect(Float64, tc.t); v = collect(Float64, tc.v)
    i = argmin(t)                      # first of the earliest observations
    return t[i] == 0.0 && isfinite(v[i]) && v[i] > 0
end

function prepare_curve(tc; lod::Real = _default_lod())
    valid_baseline(tc) || throw(ArgumentError("curve has no valid positive day-0 volume"))
    t = collect(Float64, tc.t)
    v = collect(Float64, tc.v)
    ord = sortperm(t)
    t = t[ord]; v = v[ord]
    v0 = v[findfirst(==(0.0), t)]
    keep = [t[k] > 0 && isfinite(v[k]) && v[k] >= 0 for k in eachindex(t)]
    tt = t[keep]; vv = v[keep]
    cens = vv .== 0
    logv = [c ? 0.0 : log(x) for (x, c) in zip(vv, cens)]
    return CurveData(tt, logv, cens, v0)
end

prepare_unit(u; lod = _default_lod()) =
    UnitData(String(u.model), prepare_curve(u.control; lod), prepare_curve(u.treated; lod))

"""
    prepare_units(units; lod, warn=true) -> (prepared::Vector{UnitData}, dropped::Vector{String})

Units whose control or treated curve lacks a valid day-0 volume are dropped with a logged warning.
"""
function prepare_units(units; lod = _default_lod(), warn::Bool = true)
    keep = [valid_baseline(u.control) && valid_baseline(u.treated) for u in units]
    dropped = String[String(u.model) for (u, k) in zip(units, keep) if !k]
    if warn && !isempty(dropped)
        @warn "dropping $(length(dropped)) of $(length(units)) units without a valid positive day-0 volume" models = dropped
    end
    return [prepare_unit(u; lod) for (u, k) in zip(units, keep) if k], dropped
end

# ---------------------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------------------

"""
    reference_cref(pk, doses; days=7, dt=0.01)

Cmax of the reference regimen under `pk`: maximum concentration on a fine grid over the first
`days` of dosing. Normalising exposure by it puts k2 on a scale that does not depend on the
(unverified) absolute dose.
"""
function reference_cref(pk, doses; days = 7.0, dt = 0.01)
    ds = [d for d in doses if d.time <= days]
    return maximum(conc(pk, ds, 0.0:dt:float(days)))
end

# PKExposure (src/pd.jl) has the cached one-compartment path (binary search over dose times,
# two exp per call). Measured the same speed as the FastExposure it replaced (198 vs 196 us per
# unit value+gradient), so that class was dropped.
make_exposure(pk, doses, cref) = PKExposure(pk, collect(Dose, doses), float(cref))

# ---------------------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------------------

struct SolverCfg{S}
    solver::S
    reltol::Float64
    abstol::Float64
end
SolverCfg(; solver = Tsit5(), reltol = 1e-5, abstol = 1e-5) = SolverCfg(solver, reltol, abstol)

"""
    HierData

Everything the likelihood needs besides parameters. `local_grad` switches the ForwardDiff
path to per-unit local gradients (5-7 dual partials per solve instead of the full parameter
dimension); `threaded` threads the unit loop.
"""
# One-slot cache of the last local value/gradient evaluation. ForwardDiff evaluates the model
# once per chunk of partials, always at the same primal point; with the cache the (expensive)
# per-unit ODE gradients are computed once per gradient call instead of once per chunk, so a
# small chunk size can be used for any number of units (very wide chunks do not compile for
# hundreds of dimensions).
mutable struct LocalGradCache
    x::Vector{Float64}
    vals::Vector{Float64}
    grads::Matrix{Float64}
    hits::Int
    misses::Int
end
LocalGradCache() = LocalGradCache(Float64[], Float64[], zeros(0, 0), 0, 0)

struct HierData{E,S}
    units::Vector{UnitData}
    exposure::E
    lod::Float64
    solver::SolverCfg{S}
    latent_w0::Bool
    local_grad::Bool
    threaded::Bool
    nu::Float64                      # Student-t degrees of freedom of the log-volume residuals (Inf = Normal)
    baseline_sd::Float64             # fixed sd of log w0 around log v0 (latent baselines)
    centered_lam0::Bool              # sample logλ0_i directly (centred) instead of via z (non-centred)
    k2_centered::Vector{Bool}        # per unit: sample logk2_i directly (centred) instead of via z (non-centred)
    model_v2::Bool                   # v2: correlated (log λ0, log k2), per-PDX transition volume
    waning::Bool                     # kill term k2·exp(-ρ t) with a pooled ρ
    token::Base.RefValue{Int}        # identity key for the task-local caches
    dropped::Vector{String}          # units removed for an invalid day-0 volume
end
HierData(units, expo, lod, solver, latent_w0, local_grad, threaded;
         nu = 5.0, baseline_sd = 0.1, centered_lam0 = true, k2_centered = falses(length(units)),
         model_v2 = false, waning = false, dropped = String[]) =
    HierData(units, expo, lod, solver, latent_w0, local_grad, threaded, float(nu), float(baseline_sd),
             centered_lam0, collect(Bool, k2_centered), model_v2, waning, Ref(rand(Int)), dropped)

# One cache per (HierData, task): chains running in separate tasks never evict each other's entry
# and no lock is needed. The cache lives in task_local_storage.
_local_cache(d::HierData) = get!(LocalGradCache, task_local_storage(), (:tumourpkpd_cache, d.token[]))::LocalGradCache
"(hits, misses) of the current task's local-gradient cache for `d`."
cache_stats(d::HierData) = (c = _local_cache(d); (c.hits, c.misses))

function prepare_data(units, doses, pk, cref; lod = _default_lod(), solver = SolverCfg(),
                      latent_w0 = true, local_grad = true, threaded = true, nu = 5.0, baseline_sd = 0.1,
                      centered_lam0 = true, k2_centered = nothing, model_v2 = false, waning = false)
    ud, dropped = prepare_units(units; lod)
    isempty(ud) && throw(ArgumentError("no units with valid baselines"))
    expo = make_exposure(pk, doses, cref)
    kc = k2_centered === nothing ? falses(length(ud)) : k2_centered
    length(kc) == length(ud) || throw(ArgumentError("k2_centered must have one entry per kept unit ($(length(ud)))"))
    return HierData(ud, expo, float(lod), solver, latent_w0, local_grad, threaded; nu, baseline_sd, centered_lam0,
                    k2_centered = kc, model_v2, waning, dropped)
end

@inline _lognormpdf(y, μ, σ) = -0.5 * ((y - μ) / σ)^2 - log(σ) - 0.9189385332046727

# Student-t location-scale log density / log CDF (ν = Inf -> Normal). The t log-CDF has no ForwardDiff
# rule, so it gets one: d/dx log F(x) = f(x) / F(x).
@inline function _t_logpdf(y, μ, σ, ν)
    isinf(ν) && return _lognormpdf(y, μ, σ)
    z = (y - μ) / σ
    return _tconst(ν) - 0.5 * (ν + 1) * log1p(z^2 / ν) - log(σ)
end
_tconst(ν) = logpdf(TDist(ν), 0.0)
_tlogcdf(x::Real, ν) = logcdf(TDist(ν), x)
function _tlogcdf(x::ForwardDiff.Dual{T}, ν) where {T}
    v = ForwardDiff.value(x)
    lc = logcdf(TDist(ν), v)
    slope = isfinite(lc) ? exp(logpdf(TDist(ν), v) - lc) : zero(lc)
    return ForwardDiff.Dual{T}(lc, slope * ForwardDiff.partials(x))
end
@inline _t_logcdf(x, μ, σ, ν) = isinf(ν) ? logcdf(Normal(μ, σ), x) : _tlogcdf((x - μ) / σ, ν)

# Log-likelihood of one curve. Returns -Inf if the ODE solve fails.
# `pointwise` (optional Vector) receives per-observation contributions.
# Residuals: log v ~ Student-t(ν, log w, σ); v = 0 is left-censored: P(log v_obs < log LOD).
# Predicted volume is floored at 1e-3 mm^3 (far below the 4 mm^3 LOD): eradicated tumours can give
# w ~ -2e-7 with a successful solve, and the log of a non-positive number is undefined. The floor makes
# the likelihood flat in w below 1e-3 mm^3 (a kink, but only in a region where every observation is
# censored or hugely penalised).
const W_FLOOR = 1e-3
function curve_loglik(c::CurveData, λ0, λ1, k1, k2, logw0, σ, expo, lod, cfg::SolverCfg;
                      pointwise = nothing, nu = 5.0)
    isempty(c.t) && return zero(promote_type(typeof(λ0), typeof(σ)))
    θ = Simeoni(λ0, λ1, k1, k2, exp(logw0))
    w, ok = simulate_tumour(θ, expo, c.t; solver = cfg.solver, reltol = cfg.reltol, abstol = cfg.abstol)
    T = promote_type(typeof(λ0), typeof(σ), typeof(logw0))
    ok || return T(-Inf)
    logLOD = log(lod)
    s = zero(T)
    for j in eachindex(c.t)
        lw = log(max(w[j], W_FLOOR))
        l = c.cens[j] ? _t_logcdf(logLOD, lw, σ, nu) : _t_logpdf(c.logv[j], lw, σ, nu)
        pointwise === nothing || (pointwise[j] = ForwardDiff.value(l))
        s += l
    end
    return s
end

# Control arm: no drug, k1/k2 irrelevant.
@inline control_loglik(u::UnitData, llam0, llam1, σ, lw0, d::HierData) =
    curve_loglik(u.control, exp(llam0), exp(llam1), one(llam0), zero(llam0), lw0, σ,
                 NoDrug(), d.lod, d.solver; nu = d.nu)

"""
    WaningExposure(expo, ρ)

Exposure with a pooled efficacy decay: e(t) = expo(t) · exp(-ρ t), so the kill term is k2 · exp(-ρ t) · e_PK(t) (t in days since
treatment start). Used for `waning=true`; dose stops are those of the wrapped exposure.
"""
struct WaningExposure{E,R}
    expo::E
    ρ::R
end
(w::WaningExposure)(t) = w.expo(t) * exp(-w.ρ * t)
dose_times(w::WaningExposure) = dose_times(w.expo)
_exposure_eltype(w::WaningExposure) = w.ρ * 1.0

@inline function treated_loglik(u::UnitData, llam0, llam1, lk1, lk2, σ, lw0, d::HierData; lrho = nothing)
    ex = (d.waning && lrho !== nothing) ? WaningExposure(d.exposure, exp(lrho)) : d.exposure
    return curve_loglik(u.treated, exp(llam0), exp(llam1), exp(lk1), exp(lk2), lw0, σ, ex, d.lod, d.solver; nu = d.nu)
end

# --- value + gradient by local forward-mode (no dependence on outer AD) -----------------

struct LocalTag end

@inline function _seed(x::NTuple{K,Float64}) where {K}
    return ntuple(Val(K)) do j
        ForwardDiff.Dual{LocalTag}(x[j], ForwardDiff.Partials(ntuple(i -> i == j ? 1.0 : 0.0, Val(K))))
    end
end

# Test hook: when set, every locally seeded (dual) arm evaluation reports a failed solve (value -Inf,
# zero partials), to exercise the "Float64 solve fine, dual solve failed" rejection path.
const _FAULT_DUAL = Ref(false)

@inline function _valgrad(f, x::NTuple{K,Float64}) where {K}
    r = f(_seed(x)...)
    _FAULT_DUAL[] && return (-Inf, ntuple(_ -> 0.0, Val(K)))
    r isa ForwardDiff.Dual || return (Float64(r), ntuple(_ -> 0.0, Val(K)))
    return (ForwardDiff.value(r), ForwardDiff.partials(r).values)
end

# Local parameters per unit, in this order (always 8 stored; unused ones have zero gradient):
#   1 llam0  2 lk2  3 llam1  4 lk1  5 σ  6 lw0c  7 lw0t  8 lrho (log waning rate; only with waning=true)
const NLOC = 8

"""
Value and gradient (length-8 tuple, order above) of one unit's log-likelihood at plain Float64 inputs.

The VALUE is the ordinary Float64 solve, i.e. exactly what `unit_loglik` / `pointwise_loglik` return;
only the gradient comes from the locally seeded dual solve. A failed or non-finite plain solve gives
-Inf in both paths. The single remaining asymmetry: if the plain solve succeeds but the dual solve
fails or has a non-finite derivative the state is rejected (-Inf) because no gradient exists.
(A 7-tuple without `lrho` is accepted.)
"""
function unit_valgrad(u::UnitData, p::NTuple{8,Float64}, d::HierData)
    llam0, lk2, llam1, lk1, σ, lw0c, lw0t, lrho = p
    zero8 = ntuple(_ -> 0.0, Val(8))
    wn = d.waning
    v = unit_loglik(u, llam0, lk2, llam1, lk1, σ, lw0c, lw0t, d; lrho = wn ? lrho : nothing)
    isfinite(v) || return (-Inf, zero8)
    # control arm: (llam0, llam1, σ [, lw0c])
    if d.latent_w0
        vc, gc = _valgrad((a, b, c, w) -> control_loglik(u, a, b, c, w, d), (llam0, llam1, σ, lw0c))
        g1c, g3c, g5c, g6c = gc
    else
        vc, gc = _valgrad((a, b, c) -> control_loglik(u, a, b, c, lw0c, d), (llam0, llam1, σ))
        g1c, g3c, g5c = gc; g6c = 0.0
    end
    # treated arm: (llam0, llam1, lk1, lk2, σ [, lw0t] [, lrho])
    if d.latent_w0 && wn
        vt, gt = _valgrad((a, b, c, e, f, w, r) -> treated_loglik(u, a, b, c, e, f, w, d; lrho = r),
                          (llam0, llam1, lk1, lk2, σ, lw0t, lrho))
        g1t, g3t, g4t, g2t, g5t, g7t, g8t = gt
    elseif d.latent_w0
        vt, gt = _valgrad((a, b, c, e, f, w) -> treated_loglik(u, a, b, c, e, f, w, d), (llam0, llam1, lk1, lk2, σ, lw0t))
        g1t, g3t, g4t, g2t, g5t, g7t = gt; g8t = 0.0
    elseif wn
        vt, gt = _valgrad((a, b, c, e, f, r) -> treated_loglik(u, a, b, c, e, f, lw0t, d; lrho = r),
                          (llam0, llam1, lk1, lk2, σ, lrho))
        g1t, g3t, g4t, g2t, g5t, g8t = gt; g7t = 0.0
    else
        vt, gt = _valgrad((a, b, c, e, f) -> treated_loglik(u, a, b, c, e, f, lw0t, d), (llam0, llam1, lk1, lk2, σ))
        g1t, g3t, g4t, g2t, g5t = gt; g7t = 0.0; g8t = 0.0
    end
    g = (g1c + g1t, g2t, g3c + g3t, g4t, g5c + g5t, g6c, g7t, g8t)
    # a failed dual solve returns Dual(-Inf) with zero partials: its "gradient" is meaningless, so the
    # dual arm values must be finite before the gradients are accepted
    (isfinite(vc) && isfinite(vt)) || return (-Inf, zero8)
    all(isfinite, g) || return (-Inf, zero8)
    return (v, g)
end
unit_valgrad(u::UnitData, p::NTuple{7,Float64}, d::HierData) = unit_valgrad(u, (p..., 0.0), d)

# Plain (generic number type) unit log-likelihood.
function unit_loglik(u::UnitData, llam0, lk2, llam1, lk1, σ, lw0c, lw0t, d::HierData; lrho = nothing)
    return control_loglik(u, llam0, llam1, σ, lw0c, d) +
           treated_loglik(u, llam0, llam1, lk1, lk2, σ, lw0t, d; lrho)
end

@inline function _foreach_unit(f, n, threaded)
    if threaded && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            f(i)
        end
    else
        for i in 1:n
            f(i)
        end
    end
    return nothing
end

"""
    population_loglik(d, llam0, lk2, lw0c, lw0t, llam1, lk1, σ; lrho=nothing)

Sum over units of the unit log-likelihoods. Each unit's value is computed in a pure function
into a preallocated buffer (no `~` inside threads); the caller adds the sum with
`@addlogprob!`. Buffer element type follows the parameter type. `llam1` is a scalar (pooled λ1, model v1) or a
per-unit vector (v2); `lrho` the log waning rate (waning=true).

For ForwardDiff dual inputs with `d.local_grad`, each unit's value and gradient are obtained by
a small local forward pass and the result is stitched back into a dual number through the
chain rule, so cost is linear in the number of units regardless of the outer chunk size.
"""
function population_loglik(d::HierData, llam0::AbstractVector, lk2::AbstractVector,
                           lw0c::AbstractVector, lw0t::AbstractVector,
                           llam1, lk1, σ; lrho = nothing)
    T = promote_type(eltype(llam0), eltype(lk2), eltype(lw0c), eltype(lw0t), eltype(llam1),
                     typeof(lk1), typeof(σ), lrho === nothing ? Float64 : typeof(lrho))
    N = length(d.units)
    buf = Vector{T}(undef, N)
    _foreach_unit(N, d.threaded) do i
        l1 = llam1 isa AbstractVector ? llam1[i] : llam1
        buf[i] = unit_loglik(d.units[i], llam0[i], lk2[i], l1, lk1, σ, lw0c[i], lw0t[i], d; lrho)
    end
    return sum(buf)
end

# local-gradient path for ForwardDiff duals
function population_loglik(d::HierData, llam0::AbstractVector{D}, lk2::AbstractVector{D},
                           lw0c::AbstractVector, lw0t::AbstractVector,
                           llam1, lk1, σ; lrho = nothing) where {Tg,N,D<:ForwardDiff.Dual{Tg,Float64,N}}
    d.local_grad || return invoke(population_loglik,
        Tuple{HierData,AbstractVector,AbstractVector,AbstractVector,AbstractVector,Any,Any,Any},
        d, llam0, lk2, lw0c, lw0t, llam1, lk1, σ; lrho)
    nu = length(d.units)
    val(x) = ForwardDiff.value(x)
    per_unit_l1 = llam1 isa AbstractVector
    vk1 = val(lk1); vσ = val(σ); vrho = lrho === nothing ? 0.0 : val(lrho)
    vlam1s = per_unit_l1 ? 0.0 : val(llam1)
    # packed primal point: 4 global entries, then 5 per unit (llam0, lk2, lw0c, lw0t, llam1_i)
    xv = Vector{Float64}(undef, 4 + 5nu)
    xv[1] = vk1; xv[2] = vσ; xv[3] = vrho; xv[4] = vlam1s
    for i in 1:nu
        o = 4 + 5(i - 1)
        xv[o+1] = val(llam0[i]); xv[o+2] = val(lk2[i]); xv[o+3] = val(lw0c[i]); xv[o+4] = val(lw0t[i])
        xv[o+5] = per_unit_l1 ? val(llam1[i]) : 0.0
    end
    cache = _local_cache(d)
    if cache.x == xv
        cache.hits += 1
        vals = cache.vals; grads = cache.grads
    else
        vals = Vector{Float64}(undef, nu)
        grads = Matrix{Float64}(undef, NLOC, nu)
        _foreach_unit(nu, d.threaded) do i
            o = 4 + 5(i - 1)
            l1 = per_unit_l1 ? xv[o+5] : vlam1s
            p = (xv[o+1], xv[o+2], l1, vk1, vσ, xv[o+3], xv[o+4], vrho)
            v, g = unit_valgrad(d.units[i], p, d)
            vals[i] = v
            for j in 1:NLOC
                grads[j, i] = g[j]
            end
        end
        cache.x = xv; cache.vals = vals; cache.grads = grads; cache.misses += 1
    end
    total = sum(vals)
    isfinite(total) || return D(-Inf)           # zero partials; treated as a rejected state
    P = ForwardDiff.Partials{N,Float64}
    acc = zero(P)
    g3 = 0.0; g4 = 0.0; g5 = 0.0; g8 = 0.0
    for i in 1:nu
        acc += grads[1, i] * ForwardDiff.partials(llam0[i])
        acc += grads[2, i] * ForwardDiff.partials(lk2[i])
        if d.latent_w0
            acc += grads[6, i] * ForwardDiff.partials(lw0c[i])
            acc += grads[7, i] * ForwardDiff.partials(lw0t[i])
        end
        per_unit_l1 ? (acc += grads[3, i] * ForwardDiff.partials(llam1[i])) : (g3 += grads[3, i])
        g4 += grads[4, i]; g5 += grads[5, i]; g8 += grads[8, i]
    end
    per_unit_l1 || (acc += g3 * ForwardDiff.partials(llam1))
    acc += g4 * ForwardDiff.partials(lk1)
    acc += g5 * ForwardDiff.partials(σ)
    lrho === nothing || (acc += g8 * ForwardDiff.partials(lrho))
    return D(total, acc)
end

# ---------------------------------------------------------------------------------------
# Turing model
# ---------------------------------------------------------------------------------------

# index of unit i inside its own group (centred / non-centred)
function _k2_positions(mask::AbstractVector{Bool})
    pos = zeros(Int, length(mask)); a = 0; b = 0
    for i in eachindex(mask)
        mask[i] ? (a += 1; pos[i] = a) : (b += 1; pos[i] = b)
    end
    return pos
end

@model function hier_model_v1(d::HierData, pri::HierPriors)
    N = length(d.units)
    μ_λ0 ~ Normal(pri.μ_λ0...)
    ω_λ0 ~ truncated(Normal(0.0, pri.ω_λ0); lower = 0.0)
    μ_k2 ~ Normal(pri.μ_k2...)
    ω_k2 ~ truncated(Normal(0.0, pri.ω_k2); lower = 0.0)
    logVtr ~ Normal(pri.logVtr...)
    logλ1 = μ_λ0 + logVtr                      # λ1 = exp(μ_λ0) · Vtr: transition volume Vtr = λ1/λ0
    logk1 ~ Normal(pri.logk1...)
    σ ~ truncated(Normal(0.0, pri.σ); lower = 0.0)
    # λ0 is strongly identified by each unit's control curve (and shared by its treated arm), so it is sampled
    # CENTRED: with data this informative the non-centred z_i has a posterior sd of ~0.01-0.05 against a N(0,1)
    # prior and forces NUTS step sizes of that size. k2 is only weakly identified for non-responders (upper
    # bound) and eradicated tumours (lower bound), so it stays non-centred.
    if d.centered_lam0
        llam0 ~ filldist(Normal(μ_λ0, ω_λ0), N)
    else
        z_λ0 ~ filldist(Normal(), N)
        llam0 = μ_λ0 .+ ω_λ0 .* z_λ0
    end
    nc = count(d.k2_centered)
    if nc > 0
        lk2c ~ filldist(Normal(μ_k2, ω_k2), nc)           # well-identified potencies: centred
    end
    if nc < N
        z_k2 ~ filldist(Normal(), N - nc)                 # weakly identified (non-responders, eradicated): non-centred
    end
    kpos = _k2_positions(d.k2_centered)
    lk2 = [d.k2_centered[i] ? lk2c[kpos[i]] : μ_k2 + ω_k2 * z_k2[kpos[i]] for i in 1:N]
    if d.latent_w0
        # Flat prior on log w0 plus the single day-0 observation measured with a FIXED error sd τ
        # (`baseline_sd`, ~ caliper measurement error)  =>  logw0 = log v0 + τ z (non-centred).
        z_w0c ~ filldist(Normal(), N)
        z_w0t ~ filldist(Normal(), N)
        lw0c = [log(u.control.v0) for u in d.units] .+ d.baseline_sd .* z_w0c
        lw0t = [log(u.treated.v0) for u in d.units] .+ d.baseline_sd .* z_w0t
    else
        lw0c = [log(u.control.v0) for u in d.units]
        lw0t = [log(u.treated.v0) for u in d.units]
    end
    if d.waning
        logρ_w ~ Normal(pri.logρ_w...)
        ll = population_loglik(d, llam0, lk2, lw0c, lw0t, logλ1, logk1, σ; lrho = logρ_w)
    else
        ll = population_loglik(d, llam0, lk2, lw0c, lw0t, logλ1, logk1, σ)
    end
    Turing.@addlogprob! ll
    return nothing
end

"""
    hier_model(d, priors)

The Turing model: v1 (`d.model_v2 == false`) has independent (log λ0, log k2) effects and a pooled transition volume; v2 has
correlated (log λ0, log k2) effects with an LKJ(η) prior on their 2×2 correlation and a per-PDX transition volume with partial
pooling. Both optionally have a pooled waning rate (`d.waning`).
"""
hier_model(d::HierData, pri::HierPriors) = d.model_v2 ? hier_model_v2(d, pri) : hier_model_v1(d, pri)

# ----- v2 ------------------------------------------------------------------------------------------
# Per PDX i:  (log λ0_i, log k2_i) ~ MVN((μ_λ0, μ_k2), Σ),  Σ = diag(ω) R diag(ω),  R = [1 ρ; ρ 1],  (ρ+1)/2 ~ Beta(η, η) = LKJ(η) for 2x2.
#             Cholesky factor of R = [1 0; ρ sqrt(1-ρ²)], so with z1, z2 ~ N(0,1) (non-centred) 
#                 log λ0_i = μ_λ0 + ω_λ0 z1_i,   log k2_i = μ_k2 + ω_k2 (ρ z1_i + sqrt(1-ρ²) z2_i).
#             Centred sampling of log λ0_i (as in v1) keeps this exact: log k2_i | log λ0_i ~ N(cm_i, ω_k2 sqrt(1-ρ²)),
#             cm_i = μ_k2 + ρ ω_k2 (log λ0_i - μ_λ0)/ω_λ0; units flagged in `k2_centered` sample log k2_i directly from that
#             conditional, the others through its non-centred z2_i.
#             log Vtr_i = μ_Vtr + ω_Vtr z3_i (independent of the pair),  log λ1_i = log λ0_i + log Vtr_i.
@model function hier_model_v2(d::HierData, pri::HierPriors)
    N = length(d.units)
    μ_λ0 ~ Normal(pri.μ_λ0...)
    ω_λ0 ~ truncated(Normal(0.0, pri.ω_λ0); lower = 0.0)
    μ_k2 ~ Normal(pri.μ_k2...)
    ω_k2 ~ truncated(Normal(0.0, pri.ω_k2); lower = 0.0)
    ρ_u ~ Beta(pri.lkj_eta, pri.lkj_eta)               # (ρ + 1)/2: the LKJ(η) prior on a 2x2 correlation matrix
    μ_Vtr ~ Normal(pri.logVtr...)
    ω_Vtr ~ truncated(Normal(0.0, pri.ω_Vtr); lower = 0.0)
    logk1 ~ Normal(pri.logk1...)
    σ ~ truncated(Normal(0.0, pri.σ); lower = 0.0)
    if d.waning
        logρ_w ~ Normal(pri.logρ_w...)
    end
    if d.centered_lam0
        llam0 ~ filldist(Normal(μ_λ0, ω_λ0), N)
        astd = (llam0 .- μ_λ0) ./ ω_λ0
    else
        z_λ0 ~ filldist(Normal(), N)
        llam0 = μ_λ0 .+ ω_λ0 .* z_λ0
        astd = z_λ0
    end
    ρ = 2 * ρ_u - 1
    sb = ω_k2 * sqrt(1 - ρ^2)
    cm = μ_k2 .+ (ρ * ω_k2) .* astd                    # conditional mean of log k2 given log λ0
    nc = count(d.k2_centered)
    if nc > 0
        cidx = findall(d.k2_centered)
        lk2c ~ arraydist([Normal(cm[i], sb) for i in cidx])
    end
    if nc < N
        z_k2 ~ filldist(Normal(), N - nc)
    end
    kpos = _k2_positions(d.k2_centered)
    lk2 = [d.k2_centered[i] ? lk2c[kpos[i]] : cm[i] + sb * z_k2[kpos[i]] for i in 1:N]
    z_Vtr ~ filldist(Normal(), N)
    llam1 = llam0 .+ (μ_Vtr .+ ω_Vtr .* z_Vtr)
    if d.latent_w0
        z_w0c ~ filldist(Normal(), N)
        z_w0t ~ filldist(Normal(), N)
        lw0c = [log(u.control.v0) for u in d.units] .+ d.baseline_sd .* z_w0c
        lw0t = [log(u.treated.v0) for u in d.units] .+ d.baseline_sd .* z_w0t
    else
        lw0c = [log(u.control.v0) for u in d.units]
        lw0t = [log(u.treated.v0) for u in d.units]
    end
    if d.waning
        ll = population_loglik(d, llam0, lk2, lw0c, lw0t, llam1, logk1, σ; lrho = logρ_w)
    else
        ll = population_loglik(d, llam0, lk2, lw0c, lw0t, llam1, logk1, σ)
    end
    Turing.@addlogprob! ll
    return nothing
end

# ---------------------------------------------------------------------------------------
# PosteriorDraws + persistence
# ---------------------------------------------------------------------------------------

struct PosteriorDraws
    drug::String
    models::Vector{String}
    doses::Vector{Dose}
    pk::Vector{OneCmt{Float64}}
    cref::Float64
    μ_λ0::Vector{Float64}; ω_λ0::Vector{Float64}
    μ_k2::Vector{Float64}; ω_k2::Vector{Float64}
    logλ1::Vector{Float64}; logk1::Vector{Float64}; σ::Vector{Float64}
    λ0::Matrix{Float64}
    k2::Matrix{Float64}
    w0_control::Vector{Float64}; w0_treated::Vector{Float64}   # per model: posterior median (latent) or observed
    w0_control_draws::Matrix{Float64}                           # ndraw × nmodel baselines per draw
    w0_treated_draws::Matrix{Float64}                           # (constant columns when w0 is fixed)
    diagnostics::Dict{String,Any}
end

# Fixed-baseline constructor (old 16-field form): per-draw baselines repeat the fixed values.
function PosteriorDraws(drug, models, doses, pk, cref, μ_λ0, ω_λ0, μ_k2, ω_k2, logλ1, logk1, σ,
                        λ0, k2, w0_control, w0_treated, diagnostics)
    S = size(λ0, 1)
    wc = repeat(permutedims(collect(Float64, w0_control)), S, 1)
    wt = repeat(permutedims(collect(Float64, w0_treated)), S, 1)
    return PosteriorDraws(drug, models, doses, pk, cref, μ_λ0, ω_λ0, μ_k2, ω_k2, logλ1, logk1, σ,
                          λ0, k2, w0_control, w0_treated, wc, wt, diagnostics)
end

"Save a `PosteriorDraws` to `path` (JLD2, key \"pd\")."
function save_fit(path::AbstractString, pd::PosteriorDraws)
    mkpath(dirname(abspath(path)))
    jldsave(path; pd)
    return path
end

"Load a `PosteriorDraws` written by `save_fit`."
load_fit(path::AbstractString) = JLD2.load(path, "pd")::PosteriorDraws

# ---------------------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------------------

# crude per-unit growth rate: slope of log v on t over the control curve's positive obs
function _control_slope(u::UnitData)
    c = u.control
    idx = findall(j -> !c.cens[j], eachindex(c.t))
    length(idx) < 3 && return NaN
    t = [0.0; c.t[idx]]; y = [log(c.v0); c.logv[idx]]
    tm = mean(t); ym = mean(y)
    return sum((t .- tm) .* (y .- ym)) / max(sum((t .- tm) .^ 2), eps())
end

function initial_values(d::HierData, pri::HierPriors, rng::AbstractRNG; jitter = 0.1)
    N = length(d.units)
    sl = filter(x -> isfinite(x) && x > 0.005, _control_slope.(d.units))
    μ0 = isempty(sl) ? pri.μ_λ0[1] : clamp(log(median(sl)), log(0.02), log(0.8))
    sl_all = _control_slope.(d.units)
    ll0 = [isfinite(x) && x > 0.005 ? clamp(log(x), log(0.01), log(1.0)) : μ0 for x in sl_all]
    nt = (; μ_λ0 = μ0 + jitter * randn(rng), ω_λ0 = 0.3 * exp(jitter * randn(rng)),
          μ_k2 = pri.μ_k2[1] + jitter * randn(rng), ω_k2 = 0.5 * exp(jitter * randn(rng)),
          logVtr = pri.logVtr[1] + jitter * randn(rng), logk1 = pri.logk1[1] + jitter * randn(rng),
          σ = 0.25 * exp(jitter * randn(rng)), z_k2 = jitter .* randn(rng, N))
    nt = d.centered_lam0 ? merge(nt, (; llam0 = ll0 .+ jitter .* randn(rng, N))) :
                           merge(nt, (; z_λ0 = jitter .* randn(rng, N)))
    if d.latent_w0
        nt = merge(nt, (; z_w0c = jitter .* randn(rng, N), z_w0t = jitter .* randn(rng, N)))
    end
    nt = _model_extras(nt, d, pri, rng, jitter, false)
    return apply_k2_mask(nt, d.k2_centered)
end

# v2 / waning parameters on top of the v1 start (logVtr -> μ_Vtr, + ρ_u, ω_Vtr, z_Vtr; + logρ_w)
function _model_extras(nt::NamedTuple, d::HierData, pri::HierPriors, rng::AbstractRNG, jitter, from_prior::Bool)
    N = length(d.units)
    if d.model_v2
        rest = Base.structdiff(nt, (; logVtr = nothing))
        ρu = from_prior ? rand(rng, Beta(pri.lkj_eta, pri.lkj_eta)) : clamp(0.5 + jitter * randn(rng) * 0.2, 0.05, 0.95)
        ωV = from_prior ? 0.05 + abs(pri.ω_Vtr * randn(rng)) : 0.3 * exp(jitter * randn(rng))
        zV = (from_prior ? 0.5 : jitter) .* randn(rng, N)
        nt = merge(rest, (; μ_Vtr = nt.logVtr, ω_Vtr = ωV, ρ_u = ρu, z_Vtr = zV))
    end
    if d.waning
        nt = merge(nt, (; logρ_w = pri.logρ_w[1] + (from_prior ? pri.logρ_w[2] : jitter) * randn(rng)))
    end
    return nt
end

# ---------------------------------------------------------------------------------------
# fit_drug
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# Frozen whole-PDX split shared by all drugs
# ---------------------------------------------------------------------------------------

"""
    frozen_pdx_split(drugs=list_drugs(); frac=0.2, seed=1, dir=default_cache_dir()) -> (train_ids, test_ids, info)

One stratified whole-PDX split (`split_models`, stratified by tumour type) computed on the UNION of the PDX models that have a
paired control/treated curve for any of `drugs`, so the same PDX identities are held out for every drug (controls are shared
per PDX). A drug's own holdout is `test_ids ∩ its models`. Deterministic in (frac, seed, set of models); independent of
input order. `info` records the universe for the saved metadata.
"""
function frozen_pdx_split(drugs = list_drugs(); frac::Real = 0.2, seed::Integer = 1, dir = default_cache_dir())
    uni = Dict{String,String}()
    for dr in drugs
        for u in drug_dataset(dr; dir)
            uni[String(u.model)] = String(u.tumour_type)
        end
    end
    ids = sort(collect(keys(uni)))
    train, test = split_models([(; model = m, tumour_type = uni[m]) for m in ids]; frac, seed)
    return train, test, (; drugs = collect(String.(drugs)), n_universe = length(ids), frac = float(frac), seed = Int(seed))
end

# ---------------------------------------------------------------------------------------
# Fit window
# ---------------------------------------------------------------------------------------

"Truncate a curve (any struct with model, tumour_type, treatment, t, v) to observations with t <= window."
function truncate_curve(c, window::Real)
    m = [t <= window for t in c.t]
    return typeof(c)(c.model, c.tumour_type, c.treatment, c.t[m], c.v[m])
end

"""
    window_units(units, window_days) -> units

Prespecified fit window: keep only observations with day <= `window_days` in both arms of every unit
(`window_days <= 0` or `Inf` = full follow-up, returns `units` unchanged). Apply BEFORE any late/holdout split.
"""
function window_units(units, window_days::Real)
    (window_days <= 0 || isinf(window_days)) && return units
    return [typeof(u)(u.model, u.tumour_type, truncate_curve(u.control, window_days),
                      truncate_curve(u.treated, window_days)) for u in units]
end

"""
    apply_k2_mask(nt, mask) -> NamedTuple

Re-express a fully non-centred start (`z_k2` of length N) for a model whose units in `mask` have centred potency:
`lk2c = μ_k2 + ω_k2 z_k2[mask]`, `z_k2 = z_k2[.!mask]`.
"""
function apply_k2_mask(nt::NamedTuple, mask::AbstractVector{Bool})
    any(mask) || return nt
    haskey(nt, :z_k2) || return nt
    length(nt.z_k2) == length(mask) || return nt        # already converted
    rest = Base.structdiff(nt, (; z_k2 = nothing))
    if haskey(nt, :ρ_u)          # v2: conditional of log k2 given log λ0 (see hier_model_v2)
        ρ = 2 * nt.ρ_u - 1
        astd = haskey(nt, :z_λ0) ? nt.z_λ0 : (nt.llam0 .- nt.μ_λ0) ./ nt.ω_λ0
        cmk = nt.μ_k2 .+ (ρ * nt.ω_k2) .* astd[mask]
        out = merge(rest, (; lk2c = cmk .+ (nt.ω_k2 * sqrt(1 - ρ^2)) .* nt.z_k2[mask]))
    else
        out = merge(rest, (; lk2c = nt.μ_k2 .+ nt.ω_k2 .* nt.z_k2[mask]))
    end
    all(mask) || (out = merge(out, (; z_k2 = nt.z_k2[.!mask])))
    return out
end

"""
    choose_k2_centering(model, nt; sd_threshold=0.3, chunksize=32) -> Vector{Bool}

Parameterisation choice (it does not change the target density): at a mode `nt` of the NON-centred model, the
conditional posterior sd of each z_k2_i is 1/sqrt(-∂²log p/∂z_k2_i²). Where the data pin it far tighter than its N(0,1)
prior (sd < `sd_threshold`) a non-centred z creates a very narrow, μ_k2/ω_k2-coupled direction (this is what forces NUTS
step sizes of ~0.005 on the real binimetinib data), so those units are sampled centred; weakly identified ones stay
non-centred.
"""
function choose_k2_centering(model, nt::NamedTuple; sd_threshold::Real = 0.3, chunksize = 32)
    N = length(nt.z_k2)
    adtype = AutoForwardDiff(; chunksize)
    vi = DynamicPPL.VarInfo(Random.MersenneTwister(0), model)
    vi = DynamicPPL.link(DynamicPPL.initialize_parameters!!(vi, nt, model), model)
    ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi; adtype)
    x = copy(vi[:])
    # offset of z_k2 in the flattened (linked) vector: parameters are stored in model order
    offs = 0
    for k in keys(DynamicPPL.values_as(vi, NamedTuple))
        k === :z_k2 && break
        offs += length(nt[k])
    end
    mask = falses(N)
    h = 1e-3
    for i in 1:N
        j = offs + i
        e = zeros(length(x)); e[j] = h
        gp = LogDensityProblems.logdensity_and_gradient(ldf, x .+ e)[2][j]
        gm = LogDensityProblems.logdensity_and_gradient(ldf, x .- e)[2][j]
        hii = (gp - gm) / (2h)
        sd = hii < 0 ? 1 / sqrt(-hii) : Inf
        mask[i] = sd < sd_threshold
    end
    return collect(Bool, mask)
end

# ---------------------------------------------------------------------------------------
# Multi-start MAP of the sampling target (with the unconstrained-space Jacobian, so the hierarchical
# sds are not driven to the boundary the way a Jacobian-free MAP is)
# ---------------------------------------------------------------------------------------

"""
    multistart_map(model, d, priors, rng; nstarts=5, maxiters=1000, chunksize, logger=nothing) -> Vector

Runs L-BFGS on log p(θ) (+ log-Jacobian, i.e. exactly the log density NUTS samples) from `nstarts` starting
points: start 1 is the data-driven start, the others are drawn from the prior (z effects ~ N(0, 0.5²)).
Returns one NamedTuple per start `(; nt, lp, iters, converged, start)` with `nt` the constrained parameter
values (usable as `init`). Failed starts have `lp = -Inf`.
"""
function multistart_map(model, d::HierData, priors::HierPriors, rng::AbstractRNG; nstarts::Integer = 5,
                        maxiters::Integer = 1000, chunksize = 32, logger = nothing)
    out = NamedTuple[]
    for k in 1:nstarts
        nt0 = k == 1 ? initial_values(d, priors, rng; jitter = 0.0) : prior_start(d, priors, rng)
        t0 = time()
        r = try
            find_mode(model, nt0; maxiters, chunksize)
        catch err
            (; nt = nt0, lp = -Inf, iters = 0, converged = false)
        end
        msg = "  [map] start $k/$nstarts: lp = $(round(r.lp; digits = 2)), $(r.iters) iterations, $(round(time() - t0; digits = 1)) s"
        logger === nothing ? (println(msg); flush(stdout)) : logger(msg)
        push!(out, (; r..., start = k))
    end
    return out
end

function find_mode(model, nt0; maxiters = 1000, chunksize = 32)
    adtype = AutoForwardDiff(; chunksize)
    vi = DynamicPPL.VarInfo(Random.MersenneTwister(0), model)
    vi = DynamicPPL.initialize_parameters!!(vi, nt0, model)
    vi = DynamicPPL.link(vi, model)
    ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi; adtype)
    f(x) = (lp = LogDensityProblems.logdensity(ldf, x); isfinite(lp) ? -lp : 1e30)
    function g!(G, x)
        lp, gr = LogDensityProblems.logdensity_and_gradient(ldf, x)
        G .= isfinite(lp) ? -gr : 0.0
        return nothing
    end
    res = Optim.optimize(f, g!, copy(vi[:]), Optim.LBFGS(), Optim.Options(iterations = maxiters, g_tol = 1e-3))
    x = Optim.minimizer(res)
    vi2 = DynamicPPL.invlink(DynamicPPL.unflatten(vi, x), model)
    nt = DynamicPPL.values_as(vi2, NamedTuple)
    return (; nt, lp = -Optim.minimum(res), iters = Optim.iterations(res), converged = Optim.converged(res))
end

function prior_start(d::HierData, pri::HierPriors, rng::AbstractRNG)
    N = length(d.units)
    nt = (; μ_λ0 = pri.μ_λ0[1] + pri.μ_λ0[2] * randn(rng), ω_λ0 = 0.05 + abs(pri.ω_λ0 * randn(rng)),
          μ_k2 = pri.μ_k2[1] + pri.μ_k2[2] * randn(rng), ω_k2 = 0.05 + abs(pri.ω_k2 * randn(rng)),
          logVtr = pri.logVtr[1] + pri.logVtr[2] * randn(rng), logk1 = pri.logk1[1] + pri.logk1[2] * randn(rng),
          σ = 0.05 + abs(pri.σ * randn(rng)), z_k2 = 0.5 .* randn(rng, N))
    nt = d.centered_lam0 ? merge(nt, (; llam0 = nt.μ_λ0 .+ nt.ω_λ0 .* randn(rng, N))) :
                           merge(nt, (; z_λ0 = 0.5 .* randn(rng, N)))
    d.latent_w0 && (nt = merge(nt, (; z_w0c = 0.5 .* randn(rng, N), z_w0t = 0.5 .* randn(rng, N))))
    nt = _model_extras(nt, d, pri, rng, 0.0, true)
    return apply_k2_mask(nt, d.k2_centered)
end

const _POSITIVE_PARAMS = (:ω_λ0, :ω_k2, :σ, :ω_Vtr)

"Small random perturbation of a constrained start `nt` (scales stay positive)."
function jitter_init(nt::NamedTuple, rng::AbstractRNG, s::Real)
    s == 0 && return nt
    vals = map(keys(nt)) do k
        v = nt[k]
        if k in _POSITIVE_PARAMS
            max(v, 1e-3) * exp(s * randn(rng))
        elseif k === :ρ_u
            let lg = log(clamp(v, 1e-4, 1 - 1e-4) / (1 - clamp(v, 1e-4, 1 - 1e-4))) + s * randn(rng)
                1 / (1 + exp(-lg))
            end
        elseif v isa Real
            v + s * randn(rng)
        else
            v .+ s .* randn(rng, length(v))
        end
    end
    return NamedTuple{keys(nt)}(Tuple(vals))
end

"""
    map_init(units, doses, pk, cref; kwargs...) -> (nt, lps, modes)

Multi-start MAP only (no sampling), for process-parallel fits: the parent computes this once and hands the best
mode `nt` to every chain process as `init`. Accepts the model keywords of `fit_drug` (window_days, lod, nu,
baseline_sd, latent_w0, priors, reltol, abstol) plus `nstarts`, `maxiters`, `rng`, `logger`.
"""
function map_init(units, doses, pk::OneCmt, cref::Real; kwargs...)
    r = map_modes(units, doses, pk, cref; kwargs...)
    return r.starts[1].nt, r.all_lps, r.modes, r.k2_centered
end

"""
    split_modes(modes; lp_tol=0.5, par_tol=0.1, max_gap=30) -> (; kept, excluded)

Collapse the multi-start MAP results to distinct modes (best log-density first). A start is a new mode if its log-density
differs by more than `lp_tol` from every kept mode OR its population parameters (μ_λ0, ω_λ0, μ_k2, ω_k2, logVtr, logk1, σ)
differ by more than `par_tol` in some coordinate; otherwise it is merged into the kept mode. Peak height alone does not
establish negligible posterior mass, so by default (`converged_gap = Inf`) a distinct mode that CONVERGED is always kept; only a
NON-converged start more than `max_gap` log-density units below the best is excluded. `converged_gap` (finite) also excludes
converged modes more than that far below the best (still recorded); use it when chains started in such a mode are known not to
leave it (v2 binimetinib pilot: a converged mode 90 below the best held two chains for 600 iterations at a data log-likelihood
72 lower than the others). Every excluded start is recorded as `(; lp, converged, dist, gap, k1, reason)` (gap = best lp - lp; k1 = exp of the mode's log k1)
with `dist` the largest population-parameter difference to the best mode (and failed starts with `lp = -Inf`).
"""
function split_modes(modes; lp_tol::Real = 0.5, par_tol::Real = 0.1, max_gap::Real = 30, converged_gap::Real = Inf)
    popvec(m) = Float64[getfield(m.nt, k) for k in (:μ_λ0, :ω_λ0, :μ_k2, :ω_k2, :logVtr, :μ_Vtr, :logk1, :σ, :ω_Vtr, :ρ_u, :logρ_w) if haskey(m.nt, k)]
    conv(m) = hasproperty(m, :converged) ? Bool(m.converged) : false
    k1of(m) = haskey(m.nt, :logk1) ? exp(Float64(m.nt.logk1)) : NaN
    excluded = NamedTuple[]
    good = NamedTuple[]
    for m in modes
        if isfinite(m.lp)
            push!(good, m)
        else
            push!(excluded, (; lp = m.lp, converged = false, dist = NaN, gap = NaN, k1 = k1of(m), reason = "failed start"))
        end
    end
    isempty(good) && return (; kept = NamedTuple[], excluded)
    sort!(good; by = m -> -m.lp)
    best = first(good)
    kept = NamedTuple[]
    for m in good
        if (best.lp - m.lp > max_gap && !conv(m)) || best.lp - m.lp > converged_gap
            push!(excluded, (; lp = m.lp, converged = conv(m), dist = maximum(abs.(popvec(best) .- popvec(m))),
                             gap = best.lp - m.lp, k1 = k1of(m),
                             reason = (conv(m) ? "converged but " : "not converged and ") * "$(round(best.lp - m.lp; digits = 1)) below the best (gap limit)"))
            continue
        end
        same = any(k -> abs(k.lp - m.lp) <= lp_tol && maximum(abs.(popvec(k) .- popvec(m))) <= par_tol, kept)
        same || push!(kept, m)
    end
    return (; kept, excluded)
end
distinct_modes(modes; kwargs...) = split_modes(modes; kwargs...).kept

"""
    map_modes(units, doses, pk, cref; kwargs...) -> (; starts, all_lps, modes, k2_centered, excluded)

Multi-start MAP (see `map_init` for the keywords), reduced to the distinct modes. `starts` is one entry per distinct mode, best
first, `(; nt, lp)` with `nt` already converted to the sampling coordinates (centred λ0; k2 centred where `k2_centered`, a mask
chosen at the best mode and shared by all modes). `modes` are all raw starts, `all_lps` their log-densities.
"""
function map_modes(units, doses, pk::OneCmt, cref::Real; lod::Real = _default_lod(),
                   priors::HierPriors = default_priors(), latent_w0::Bool = true, reltol = 1e-5, abstol = 1e-5,
                   window_days::Real = 42, nu::Real = 5.0, baseline_sd::Real = 0.1, nstarts::Integer = 5,
                   maxiters::Integer = 1000, rng::AbstractRNG = Random.default_rng(), logger = nothing,
                   threaded::Bool = true, centered_lam0::Bool = true, centre_k2::Bool = true,
                   centre_sd::Real = 0.3, model_v2::Bool = false, waning::Bool = false, mode_gap::Real = Inf)
    # The mode search always uses the fully NON-centred parameterisation: in the centred form the density is
    # unbounded as ω -> 0 with all effects equal to μ (a spike, not a mode of any mass). Every distinct mode is then
    # converted: λ0 centred (if `centered_lam0`) and k2 centred for the units whose potency the data pin down at the
    # best mode (`choose_k2_centering`).
    d = prepare_data(window_units(units, window_days), doses, pk, cref; lod, solver = SolverCfg(; reltol, abstol),
                     latent_w0, threaded, nu, baseline_sd, centered_lam0 = false, model_v2, waning)
    model = hier_model(d, priors)
    modes = multistart_map(model, d, priors, rng; nstarts, maxiters, logger)
    sm = split_modes(modes; converged_gap = mode_gap)
    dm = sm.kept
    isempty(dm) && throw(ErrorException("all MAP starts failed"))
    nt_best = dm[1].nt
    mask = centre_k2 ? choose_k2_centering(model, nt_best; sd_threshold = centre_sd) : falses(length(d.units))
    msg = "  [map] $(length(dm)) distinct mode(s) from $(length(modes)) starts, log-densities " *
          join(round.([m.lp for m in dm]; digits = 1), ", ") * "; k2 centred for $(count(mask)) of $(length(mask)) units (conditional sd of z_k2 < $(centre_sd))"
    logger === nothing ? (println(msg); flush(stdout)) : logger(msg)
    conv(nt) = apply_k2_mask(centered_lam0 ? to_centered(nt) : nt, mask)
    starts = [(; nt = conv(m.nt), lp = m.lp) for m in dm]
    best_nt = dm[1].nt
    excluded = [(; e..., ) for e in sm.excluded]
    return (; starts, all_lps = [m.lp for m in modes], modes, k2_centered = collect(Bool, mask), excluded)
end

"Convert a non-centred start (z_λ0) to the centred parameterisation (llam0 = μ_λ0 + ω_λ0 z_λ0)."
function to_centered(nt::NamedTuple)
    haskey(nt, :z_λ0) || return nt
    rest = Base.structdiff(nt, (; z_λ0 = nothing))
    return merge(rest, (; llam0 = nt.μ_λ0 .+ nt.ω_λ0 .* nt.z_λ0))
end

"""
    fit_drug(units, doses, pk, cref; drug="drug", nchains=4, nsamples=500, nadapt=500,
             target_accept=0.8, init=nothing, rng=Random.default_rng(), lod=detection_limit(),
             priors=default_priors(), latent_w0=true, parallel=:auto,
             local_grad=true, reltol=1e-5, abstol=1e-5, max_depth=10) -> (chain, PosteriorDraws)

`units`: vector of objects with `.model`, `.control` and `.treated` (each with `.t`, `.v`), i.e.
the data contract's `PDXUnit`. `pk::OneCmt` is held fixed. `parallel=:auto` (default) picks `:chains` (one
thread per chain, unthreaded likelihood) when there are at least as many chains as threads, `:both`
(chains in tasks, each also threading its units) for fewer chains, `:units` for a single chain or thread.
`:units` can also be forced (threaded units, chains in sequence).

`split` (a String such as "full", or a Dict such as `Dict("kind"=>"late", "late_frac"=>0.3)`; stored as a Dict in `diagnostics["split"]`, the form scripts/validation_report.jl reads), `cutoffs` (`"<model>|treated"`/`"<model>|control"` => last training day, for late holdout) and
`seed` are only recorded in `diagnostics` (keys "split", "cutoffs", "seed", plus "lod", "nchains",
"draws_per_chain", "train_models") so validation can check for leakage; the caller truncates or drops
the training data itself.

Model/fit options (phase 2b): `window_days=42` prespecified fit window (both arms truncated to day <= window before any
split; 0 = full follow-up); `nu=5` Student-t residuals (Inf = Normal); `baseline_sd=0.1` fixed sd of the latent baselines;
`map_starts=5`, `map_maxiters=1000` multi-start MAP (chains start with `init_jitter` around the best mode; all MAP
log-densities stored in `diagnostics["map_lps"]`; 0 starts = data-driven jittered start); `centered_lam0=true` and
`k2_centered` (per-unit mask, chosen from the MAP by `choose_k2_centering`) select the parameterisation of the
random effects (they do not change the target density).

`chain_start_modes` / `distinct_mode_lps` / `excluded_modes` record, for chains sampled elsewhere, which distinct MAP mode each chain started from (for `cross_mode_rhat`).

`chain_only=true` returns `(chain, nothing)` right after sampling; `precomputed_chain` (e.g. the
`chainscat` of single chains sampled in separate processes, see `chain_to_parts`) skips sampling and
builds the `PosteriorDraws` from that chain, with `precomputed_runtime` as the wall time to record.

`init`: `nothing` (data-driven start, jittered per chain), a NamedTuple, or a vector of NamedTuples.
"""
function fit_drug(units, doses, pk::OneCmt, cref::Real;
                  drug::AbstractString = "drug", nchains::Integer = 4, nsamples::Integer = 500,
                  nadapt::Integer = 500, target_accept::Real = 0.8, init = nothing,
                  rng::AbstractRNG = Random.default_rng(), lod::Real = _default_lod(),
                  priors::HierPriors = default_priors(), latent_w0::Bool = true,
                  parallel::Symbol = :auto, local_grad::Bool = true,
                  reltol::Real = 1e-5, abstol::Real = 1e-5, max_depth::Integer = 10,
                  progress::Bool = false, chunksize = nothing,
                  split = "full", cutoffs = nothing, seed = nothing, log_every::Integer = 0,
                  logger = nothing, chain_only::Bool = false, precomputed_chain = nothing,
                  precomputed_runtime = nothing, window_days::Real = 42, nu::Real = 5.0,
                  baseline_sd::Real = 0.1, map_starts::Integer = 5, map_maxiters::Integer = 1000,
                  init_jitter::Real = 0.02, map_lps = nothing, worker_info = nothing,
                  centered_lam0::Bool = true, k2_centered = nothing, chain_start_modes = nothing,
                  distinct_mode_lps = nothing, excluded_modes = nothing, model::Symbol = :v1, waning::Bool = false, mode_gap::Real = Inf)
    # :auto = :units. Chains inside one process scale badly (measured: 4 concurrent tasks ran 4x slower
    # per gradient because the ODE solves allocate heavily and share one GC); use one process per chain
    # (scripts/procchains.jl, `chain_only` + `precomputed_chain`) for chain-level parallelism.
    parallel === :auto && (parallel = :units)
    parallel in (:units, :chains, :both) || throw(ArgumentError("parallel must be :auto, :units, :chains or :both"))
    model in (:v1, :v2) || throw(ArgumentError("model must be :v1 or :v2"))
    model_v2 = model === :v2
    model_name = String(model)                      # `model` is rebound to the Turing model below
    units = window_units(units, window_days)       # prespecified fit window, applied to both arms
    threaded = parallel in (:units, :both)
    mask = k2_centered
    start_modes = Int[]; mode_lps = Float64[]
    excluded_modes === nothing && (excluded_modes = NamedTuple[])
    if init === nothing && map_starts > 0 && precomputed_chain === nothing
        mm = map_modes(units, doses, pk, cref; lod, priors, latent_w0, reltol, abstol, window_days = 0,
                       nu, baseline_sd, nstarts = map_starts, maxiters = map_maxiters, rng, logger, threaded,
                       centered_lam0, model_v2, waning, mode_gap)
        map_lps = mm.all_lps; mask = mm.k2_centered; excluded_modes = mm.excluded
        # chains start at DIFFERENT discovered modes (cycled) when there is more than one, with small jitter
        nmodes = length(mm.starts)
        start_modes = [mod1(k, nmodes) for k in 1:nchains]
        mode_lps = [m.lp for m in mm.starts]
        init = [jitter_init(mm.starts[start_modes[k]].nt, rng, init_jitter) for k in 1:nchains]
    end
    # start modes of chains sampled elsewhere (process-parallel / staged chains) are passed in for the metadata
    chain_start_modes === nothing || (start_modes = collect(Int, chain_start_modes))
    distinct_mode_lps === nothing || (mode_lps = collect(Float64, distinct_mode_lps))
    d = prepare_data(units, doses, pk, cref; lod, solver = SolverCfg(; reltol, abstol),
                     latent_w0, local_grad, threaded, nu, baseline_sd, centered_lam0, k2_centered = mask, model_v2, waning)
    N = length(d.units)
    model = hier_model(d, priors)
    dim = 7 + 2N + (latent_w0 ? 2N : 0) + (model_v2 ? 3 + N : 0) + (waning ? 1 : 0)
    cs = chunksize === nothing ? min(dim, local_grad ? 32 : 12) : chunksize
    adtype = AutoForwardDiff(; chunksize = cs)
    inits = if init === nothing
        [initial_values(d, priors, rng) for _ in 1:nchains]
    elseif init isa AbstractVector
        collect(init)
    elseif init isa NamedTuple
        [jitter_init(init, rng, init_jitter) for _ in 1:nchains]
    else
        collect(init)
    end
    precomputed_chain === nothing || (nchains = size(precomputed_chain, 3))
    sampler = NUTS(nadapt, target_accept; max_depth, adtype)
    t0 = time()
    t_last = Ref(t0)
    # optional progress log (flushed): one line every `log_every` iterations of any chain
    cb = if log_every > 0
        ctr = Threads.Atomic{Int}(0)
        cblock = ReentrantLock()                                # callbacks from MCMCThreads chains share the buffers
        total = nchains * (nsamples + nadapt)
        nsw = Float64[]; tdw = Float64[]; epw = Float64[]       # last `log_every` iterations (one chain per process)
        (rng_, model_, sampler_, sample_, state_, iteration; kwargs...) -> lock(cblock) do
            n = Threads.atomic_add!(ctr, 1) + 1
            st = hasproperty(sample_, :stat) ? sample_.stat : nothing
            push!(nsw, st !== nothing && hasproperty(st, :n_steps) ? float(st.n_steps) : NaN)
            push!(tdw, st !== nothing && hasproperty(st, :tree_depth) ? float(st.tree_depth) : NaN)
            push!(epw, st !== nothing && hasproperty(st, :step_size) ? float(st.step_size) : NaN)
            if n % log_every == 0
                phase = iteration <= nadapt ? "warmup" : "sampling"
                el = round(time() - t0; digits = 0)
                msg = "  [fit] iter $(iteration)/$(nadapt + nsamples) ($(phase)), $(n)/$(total) in total, $(el) s elapsed; " *
                      "last $(log_every): mean n_steps $(round(mean(nsw); digits = 1)), tree depth $(round(mean(tdw); digits = 2)), " *
                      "step size $(round(mean(epw); sigdigits = 3)), $(round((time() - t_last[]) / log_every; digits = 2)) s/iter"
                t_last[] = time()
                empty!(nsw); empty!(tdw); empty!(epw)
                logger === nothing ? (println(msg); flush(stdout)) : logger(msg)
            end
            nothing
        end
    else
        nothing
    end
    # AbstractMCMC only calls the callback for retained draws, so with progress logging the warmup is kept
    # during sampling (discard_adapt=false) and dropped again below.
    ckw = cb === nothing ? (;) : (; callback = cb, discard_adapt = false)
    nsteps = cb === nothing ? nsamples : nsamples + nadapt      # with discard_adapt=false the warmup is part of N
    chain = if precomputed_chain !== nothing
        precomputed_chain
    elseif nchains == 1
        sample(rng, model, sampler, nsteps; initial_params = inits[1], progress, ckw...)
    else
        alg = parallel in (:chains, :both) ? MCMCThreads() : MCMCSerial()
        sample(rng, model, sampler, alg, nsteps, nchains; initial_params = inits, progress, ckw...)
    end
    if cb !== nothing && precomputed_chain === nothing
        chain = chain[(nadapt+1):end, :, :]                      # drop the warmup that was kept for the progress log
    end
    chain_only && return chain, nothing
    runtime = precomputed_runtime === nothing ? time() - t0 : float(precomputed_runtime)
    settings = Dict{String,Any}("nchains" => nchains, "nsamples" => nsamples, "nadapt" => nadapt,
        "target_accept" => target_accept, "max_depth" => max_depth, "lod" => float(lod),
        "latent_w0" => latent_w0, "parallel" => String(parallel), "local_grad" => local_grad,
        "reltol" => float(reltol), "abstol" => float(abstol), "chunksize" => cs,
        "nthreads" => Threads.nthreads(), "n_units" => N, "nu" => float(nu), "baseline_sd" => float(baseline_sd),
        "window_days" => float(window_days), "centered_lam0" => centered_lam0, "model" => model_name, "waning" => waning, "mode_gap" => float(mode_gap), "k2_centered_units" => count(d.k2_centered), "map_starts" => map_starts, "init_jitter" => float(init_jitter),
        "priors" => Dict(string(k) => getfield(priors, k) for k in fieldnames(HierPriors)))
    settings["seed"] = seed
    worker_info === nothing || merge!(settings, Dict{String,Any}(string(k) => v for (k, v) in pairs(worker_info)))
    pd = posterior_draws(chain, d, string(drug), doses, pk, cref; runtime, settings, max_depth)
    # bookkeeping for the validation worker (leakage checks)
    pd.diagnostics["lod"] = float(lod)
    pd.diagnostics["nchains"] = nchains
    pd.diagnostics["draws_per_chain"] = size(chain, 1)          # actual, also for precomputed chains
    pd.diagnostics["fit_window_day"] = (window_days <= 0 || isinf(window_days)) ? 0.0 : float(window_days)
    pd.diagnostics["nu"] = float(nu); pd.diagnostics["baseline_sd"] = float(baseline_sd)
    pd.diagnostics["map_lps"] = map_lps === nothing ? Float64[] : collect(Float64, map_lps)
    pd.diagnostics["train_models"] = [u.model for u in d.units]
    pd.diagnostics["split"] = split isa AbstractDict ? Dict{String,Any}(string(k) => v for (k, v) in pairs(split)) :
                              Dict{String,Any}("kind" => String(split))
    pd.diagnostics["cutoffs"] = cutoffs === nothing ? Dict{String,Float64}() :
                                Dict{String,Float64}(string(k) => float(v) for (k, v) in pairs(cutoffs))
    pd.diagnostics["seed"] = seed
    pd.diagnostics["latent_w0"] = latent_w0
    pd.diagnostics["dropped_units"] = copy(d.dropped)
    pd.diagnostics["k2_centered"] = copy(d.k2_centered)                       # aligned with pd.models / train_models
    pd.diagnostics["k2_centered_models"] = [u.model for (u, c) in zip(d.units, d.k2_centered) if c]
    pd.diagnostics["chain_start_modes"] = start_modes
    pd.diagnostics["mode_lps"] = mode_lps
    pd.diagnostics["n_distinct_modes"] = length(mode_lps)
    # chains started at different discovered modes: their R-hat is the cross-mode R-hat
    pd.diagnostics["excluded_modes"] = [Dict{String,Any}("lp" => e.lp, "converged" => e.converged, "dist" => e.dist, "gap" => get(e, :gap, NaN), "k1" => get(e, :k1, NaN), "reason" => e.reason) for e in excluded_modes]
    # cross-mode R-hat only when the chains actually started from >= 2 distinct recorded modes
    length(unique(start_modes)) > 1 && (pd.diagnostics["cross_mode_rhat"] = pd.diagnostics["max_rhat"])
    # coverage of the training data, for extrapolation flags downstream
    pd.diagnostics["max_observed_day"] = maximum(max(isempty(u.control.t) ? 0.0 : u.control.t[end], isempty(u.treated.t) ? 0.0 : u.treated.t[end]) for u in d.units)
    pd.diagnostics["median_followup_day"] = median([u.treated.t[end] for u in d.units if !isempty(u.treated.t)])
    ttype = Dict{String,String}(String(u.model) => String(hasproperty(u, :tumour_type) ? u.tumour_type : "unknown") for u in units)
    pd.diagnostics["tumour_types"] = Dict{String,String}(m => get(ttype, m, "unknown") for m in pd.models)
    return chain, pd
end

"""
    chain_to_parts(chain) -> (data::Array{Float64,3}, names::Vector{Symbol}, internals::Vector{Symbol})
    chain_from_parts(data, names, internals) -> Chains

Plain-array form of an MCMCChains `Chains` (draws × params × chains), for saving single chains sampled in
separate processes and recombining them with `chainscat`.
"""
chain_to_parts(chain) = (Array{Float64}(chain.value.data), Symbol.(MCMCChains.names(chain)),
                         Symbol.(MCMCChains.names(chain, :internals)))
chain_from_parts(data, names, internals) = Chains(data, names, Dict(:internals => internals))

function chain_diagnostics(chain; max_depth::Integer = 10)
    ss = MCMCChains.summarystats(chain)
    rh = filter(isfinite, collect(ss.nt.rhat)); eb = filter(isfinite, collect(ss.nt.ess_bulk))
    et = filter(isfinite, collect(ss.nt.ess_tail))
    ni = names_internal(chain)
    nd = :numerical_error in ni ? Int(sum(chain[:numerical_error])) : 0
    td = :tree_depth in ni ? chain[:tree_depth] : nothing
    # E-BFMI per chain: sum (E_n - E_{n-1})^2 / sum (E_n - mean E)^2 (min over chains; < 0.3 flags poor momentum resampling)
    ebfmi = NaN
    if :hamiltonian_energy in ni
        E = Array(chain[:hamiltonian_energy])
        E = ndims(E) == 1 ? reshape(E, :, 1) : E
        vals = [begin e = E[:, c]; v = sum(abs2, e .- mean(e)); v > 0 ? sum(abs2, diff(e)) / v : NaN end for c in axes(E, 2)]
        vals = filter(isfinite, vals)
        isempty(vals) || (ebfmi = minimum(vals))
    end
    return Dict{String,Any}(
        "max_rhat" => isempty(rh) ? NaN : maximum(rh),          # NaN = statistic unavailable
        "min_ess_bulk" => isempty(eb) ? NaN : minimum(eb),
        "min_ess_tail" => isempty(et) ? NaN : minimum(et),
        "min_ebfmi" => ebfmi,
        "cross_mode_rhat" => NaN,                                  # set by the caller when chains start from different modes
        "n_divergences" => nd,
        "mean_acceptance" => :acceptance_rate in ni ? mean(chain[:acceptance_rate]) : NaN,
        "mean_tree_depth" => td === nothing ? NaN : mean(td),
        "max_tree_depth_hits" => td === nothing ? 0 : Int(sum(td .>= max_depth)),
        "mean_step_size" => :step_size in ni ? mean(chain[:step_size]) : NaN,
        "mean_n_steps" => :n_steps in ni ? mean(chain[:n_steps]) : NaN)
end
names_internal(chain) = Set(MCMCChains.names(chain, :internals))

function posterior_draws(chain, d::HierData, drug, doses, pk, cref; runtime, settings, max_depth = 10)
    N = length(d.units)
    g(s) = vec(Array(chain[s]))
    μ_λ0 = g(:μ_λ0); ω_λ0 = g(:ω_λ0); μ_k2 = g(:μ_k2); ω_k2 = g(:ω_k2)
    S = length(μ_λ0)
    kpos = _k2_positions(d.k2_centered)
    nlk2c = count(n -> startswith(String(n), "lk2c["), Symbol.(MCMCChains.names(chain)))
    nzk2 = count(n -> startswith(String(n), "z_k2["), Symbol.(MCMCChains.names(chain)))
    (nlk2c == count(d.k2_centered) && nzk2 == N - nlk2c) || throw(ArgumentError(
        "k2 centring mask does not match the chain: chain has $nlk2c centred / $nzk2 non-centred potency parameters, " *
        "mask has $(count(d.k2_centered)) / $(N - count(d.k2_centered)); pass the mask the chain was sampled with"))
    λ0log = if d.centered_lam0
        hcat([g(Symbol("llam0[$i]")) for i in 1:N]...)
    else
        μ_λ0 .+ ω_λ0 .* hcat([g(Symbol("z_λ0[$i]")) for i in 1:N]...)
    end
    λ0 = exp.(λ0log)
    ρcorr = Float64[]; λ1mat = Matrix{Float64}(undef, 0, 0); logλ1pool = nothing
    if d.model_v2
        # conditional of log k2 given log λ0, as sampled (see hier_model_v2)
        ρcorr = 2 .* g(:ρ_u) .- 1
        sb = ω_k2 .* sqrt.(1 .- ρcorr .^ 2)
        cm = μ_k2 .+ (ρcorr .* ω_k2) .* ((λ0log .- μ_λ0) ./ ω_λ0)
        lk2 = hcat([d.k2_centered[i] ? g(Symbol("lk2c[$(kpos[i])]")) : cm[:, i] .+ sb .* g(Symbol("z_k2[$(kpos[i])]")) for i in 1:N]...)
        μV = g(:μ_Vtr)
        λ1mat = exp.(λ0log .+ μV .+ g(:ω_Vtr) .* hcat([g(Symbol("z_Vtr[$i]")) for i in 1:N]...))
        logλ1pool = μ_λ0 .+ μV                       # population-level λ1 (median PDX): exp(μ_λ0 + μ_Vtr)
    else
        lk2 = hcat([d.k2_centered[i] ? g(Symbol("lk2c[$(kpos[i])]")) : μ_k2 .+ ω_k2 .* g(Symbol("z_k2[$(kpos[i])]")) for i in 1:N]...)
        logλ1pool = μ_λ0 .+ g(:logVtr)
    end
    k2 = exp.(lk2)
    σ = g(:σ)
    v0c = [u.control.v0 for u in d.units]; v0t = [u.treated.v0 for u in d.units]
    # per-draw baselines: latent  w0 = v0 exp(σ_s z_s);  fixed  w0 = v0 (constant columns)
    if d.latent_w0
        zc = hcat([g(Symbol("z_w0c[$i]")) for i in 1:N]...)
        zt = hcat([g(Symbol("z_w0t[$i]")) for i in 1:N]...)
        w0c_draws = exp.(log.(permutedims(v0c)) .+ d.baseline_sd .* zc)
        w0t_draws = exp.(log.(permutedims(v0t)) .+ d.baseline_sd .* zt)
        w0c = vec(median(w0c_draws; dims = 1)); w0t = vec(median(w0t_draws; dims = 1))
    else
        w0c = v0c; w0t = v0t
        w0c_draws = repeat(permutedims(v0c), S, 1); w0t_draws = repeat(permutedims(v0t), S, 1)
    end
    diag = chain_diagnostics(chain; max_depth)
    diag["runtime_s"] = runtime
    diag["settings"] = settings
    # model metadata (v2 fields live here so the PosteriorDraws layout of existing v1 fits is unchanged)
    diag["model"] = d.model_v2 ? "v2" : "v1"
    diag["waning"] = d.waning
    diag["waning_rate"] = d.waning ? exp.(g(:logρ_w)) : Float64[]          # ρ per draw (1/day), kill ∝ k2 exp(-ρ t)
    if d.model_v2
        diag["corr_lk"] = ρcorr                                             # corr(log λ0, log k2), one per draw
        diag["lambda1_draws"] = λ1mat                                       # ndraw × nmodel, λ1_i = λ0_i Vtr_i (mm^3/day)
        diag["mu_logVtr"] = g(:μ_Vtr); diag["omega_Vtr"] = g(:ω_Vtr)        # population mean / sd of log Vtr
    end
    return PosteriorDraws(drug, [u.model for u in d.units], collect(Dose, doses),
        fill(OneCmt(pk.CL, pk.V, pk.ka, pk.F), S), float(cref),
        μ_λ0, ω_λ0, μ_k2, ω_k2, logλ1pool, g(:logk1), σ, λ0, k2, w0c, w0t,
        w0c_draws, w0t_draws, diag)
end

"""
    design_units(drug, design; frac=0.2, seed=1, window=nothing, subsample=0) -> (; train, split, cutoffs, window)

Training units and saved split metadata for the production designs. `primary`: the frozen stratified 80% of PDX models
(`frozen_pdx_split`), window 42 d. `late21`: the same models, both arms truncated at day 21. `subsample` > 0 draws that many
paired PDX models first (deterministic, for smoke tests). The same function is used by every stage so that init, warmup,
sample and merge rebuild identical data.
"""
function design_units(drug::AbstractString, design::AbstractString; frac::Real = 0.2, seed::Integer = 1,
                      window = nothing, subsample::Integer = 0)
    design in ("primary", "late21") || throw(ArgumentError("design must be primary or late21"))
    units = drug_dataset(drug)
    if subsample > 0
        units = units[sort(randperm(Random.MersenneTwister(0), length(units))[1:min(subsample, length(units))])]
    end
    win = window === nothing ? (design == "late21" ? 21.0 : 42.0) : float(window)
    units = window_units(units, win)
    train_all, _, uinfo = frozen_pdx_split(; frac, seed)
    keep = Set(train_all)
    train = [u for u in units if u.model in keep]
    hold = sort([u.model for u in units if !(u.model in keep)])
    split = Dict{String,Any}("kind" => design, "model_frac" => float(frac), "seed" => Int(seed), "stratified" => "tumour_type",
                             "universe" => "union of paired PDX models over " * join(uinfo.drugs, ",") * " ($(uinfo.n_universe))",
                             "holdout_models" => hold)
    cutoffs = Dict{String,Float64}()
    if design == "late21"
        split["late_window_day"] = win
        for u in train
            cutoffs["$(u.model)|treated"] = maximum(u.treated.t)
            cutoffs["$(u.model)|control"] = maximum(u.control.t)
        end
    end
    return (; train, split, cutoffs, window = win)
end

# ---------------------------------------------------------------------------------------
# Target fingerprint (stage artifacts must belong to the same target)
# ---------------------------------------------------------------------------------------

function _fnv64(text::AbstractString, h::UInt64)
    for b in codeunits(text)
        h = xor(h, UInt64(b)); h *= 0x00000100000001b3
    end
    return h
end
"128-bit hex digest (two FNV-1a passes) of a string; deterministic across machines and Julia versions."
target_fingerprint(text::AbstractString) = string(_fnv64(text, 0xcbf29ce484222325); base = 16, pad = 16) *
                                           string(_fnv64(text, 0x84222325cbf29ce4); base = 16, pad = 16)

_fp(x::AbstractFloat) = repr(Float64(x))
_fp(x::Integer) = string(x)
_fp(x::AbstractString) = String(x)
_fp(x::Symbol) = String(x)
_fp(x::Bool) = x ? "T" : "F"
_fp(x::Tuple) = "(" * join(_fp.(x), ",") * ")"
_fp(x::AbstractVector) = "[" * join(_fp.(x), ",") * "]"
_fp(x::AbstractDict) = "{" * join(["$(k)=" * _fp(x[k]) for k in sort(collect(keys(x)); by = string)], ";") * "}"
_fp(x::Nothing) = "nothing"

"""
    fingerprint_text(; drug, design, window, models, split, priors, nu, baseline_sd, lod, doses, pk, cref,
                     k2_centered, centered_lam0, latent_w0, reltol, abstol, init_id="") -> String

Canonical description of the sampling target (and, via `init_id`, of the init artifact it came from) that is hashed by
`target_fingerprint`. Covers: drug, design, window, the training unit ids and the split (holdout ids, fractions, seed, late window),
the priors, ν, baseline sd, LOD, the dosing regimen, the PK values, cref, the solver tolerances, the k2-centring mask, the
parameterisation flags, the model version (`model`, "v1"/"v2"), `waning` and, when finite, the `mode_gap` used to choose the chain starts. No data values are included.
"""
function fingerprint_text(; drug, design, window, models, split, priors, nu, baseline_sd, lod, doses, pk, cref,
                          k2_centered, centered_lam0, latent_w0, reltol, abstol, init_id = "", model = "v1", waning = false, mode_gap = Inf)
    pri = Dict(string(k) => getfield(priors, k) for k in fieldnames(HierPriors))
    reg = [(d.time, d.amt, String(d.route)) for d in doses]
    parts = ["drug=" * _fp(drug), "design=" * _fp(design), "window=" * _fp(window), "models=" * _fp(collect(String.(models))),
                 "split=" * _fp(split), "priors=" * _fp(pri), "nu=" * _fp(nu), "bsd=" * _fp(baseline_sd), "lod=" * _fp(lod),
                 "regimen=" * _fp(reg), "pk=" * _fp([pk.CL, pk.V, pk.ka, pk.F]), "cref=" * _fp(cref),
                 "tol=" * _fp([reltol, abstol]), "k2mask=" * _fp(collect(Bool, k2_centered)), "cen=" * _fp(centered_lam0),
                 "latent=" * _fp(latent_w0), "model=" * _fp(string(model)), "waning=" * _fp(waning), "init=" * _fp(init_id)]
    # only when finite, so fingerprints of artifacts made without a gap limit (v1 default) stay valid
    isfinite(mode_gap) && push!(parts, "modegap=" * _fp(float(mode_gap)))
    return join(parts, "\n")
end

"Throw a clear error if two fingerprints differ."
function verify_fingerprint(expected::AbstractString, found; what::AbstractString = "artifact", against::AbstractString = "init.jld2")
    (found isa AbstractString && found == expected) && return true
    throw(ErrorException("target fingerprint mismatch: $what carries " * (found isa AbstractString ? "$found" : "no fingerprint") *
                         " but $against has $expected. It was produced for a different target (drug/design/window/units/priors/" *
                         "settings, regimen, mask) or another init.jld2. Delete the stale file(s) and rerun the earlier stages."))
end

# ---------------------------------------------------------------------------------------
# Staged sampling (init / warmup / sample / merge) with AdvancedHMC called directly
# ---------------------------------------------------------------------------------------
# The saved sampler state is parameters only: the unconstrained position, the adapted step size and the adapted diagonal
# metric (inverse mass). No model, data or closure is serialised; the sampling stage rebuilds the log density from the data
# and the artifact, and runs the kernel with NO adaptation.

const AHMC = Turing.Inference.AHMC

"Flat parameter layout of a model (order of the linked vector): names of the chain columns and which coordinates are log-linked."
struct StageLayout
    keys::Vector{Symbol}
    lens::Vector{Int}
    names::Vector{Symbol}
    logged::Vector{Bool}
    logit::Vector{Bool}
end

const _LOG_LINKED = (:ω_λ0, :ω_k2, :σ, :ω_Vtr)      # truncated(Normal; lower = 0) priors are linked with log
const _LOGIT_LINKED = (:ρ_u,)                         # Beta(η, η) on (0, 1) is linked with logit

function stage_layout(nt::NamedTuple)
    ks = Symbol[]; ls = Int[]; names = Symbol[]; logged = Bool[]; logit = Bool[]
    for k in keys(nt)
        v = nt[k]
        push!(ks, k)
        if v isa Real
            push!(ls, 1); push!(names, k); push!(logged, k in _LOG_LINKED); push!(logit, k in _LOGIT_LINKED)
        else
            push!(ls, length(v))
            for i in 1:length(v)
                push!(names, Symbol("$(k)[$i]")); push!(logged, false); push!(logit, false)
            end
        end
    end
    return StageLayout(ks, ls, names, logged, logit)
end

"Constrained NamedTuple -> flat unconstrained vector in the layout order (log for the scale parameters)."
function unconstrain(layout::StageLayout, nt::NamedTuple)
    out = Float64[]
    for (k, n) in zip(layout.keys, layout.lens)
        v = nt[k]
        append!(out, v isa Real ? (k in _LOG_LINKED ? log(v) : k in _LOGIT_LINKED ? log(v / (1 - v)) : float(v)) : collect(Float64, v))
    end
    return out
end

"Unconstrained draws (ndraw × dim) -> constrained values with the layout's column order."
function constrain_draws(layout::StageLayout, θ::AbstractMatrix)
    out = Matrix{Float64}(undef, size(θ)...)
    for j in axes(θ, 2)
        out[:, j] = layout.logged[j] ? exp.(θ[:, j]) : layout.logit[j] ? 1 ./ (1 .+ exp.(-θ[:, j])) : θ[:, j]
    end
    return out
end

"""
    stage_target(units, doses, pk, cref; start, kwargs...) -> (; d, model, ldf, layout, θ0, ℓπ, ∂ℓπ∂θ)

Rebuild the sampling target (data -> HierData -> Turing model -> linked LogDensityFunction) for the staged workflow.
`start` is a constrained NamedTuple in the sampling coordinates (as returned by `map_modes`); `k2_centered` the mask.
Keywords as `fit_drug`: lod, priors, latent_w0, reltol, abstol, nu, baseline_sd, centered_lam0, k2_centered, window_days,
threaded, chunksize.
"""
function stage_target(units, doses, pk::OneCmt, cref::Real; start::NamedTuple, lod::Real = _default_lod(),
                      priors::HierPriors = default_priors(), latent_w0::Bool = true, reltol = 1e-5, abstol = 1e-5,
                      nu::Real = 5.0, baseline_sd::Real = 0.1, centered_lam0::Bool = true, k2_centered = nothing,
                      window_days::Real = 42, threaded::Bool = true, chunksize = 32, model_v2::Bool = false,
                      waning::Bool = false, model_fn = nothing)
    us = window_units(units, window_days)
    d = prepare_data(us, doses, pk, cref; lod, solver = SolverCfg(; reltol, abstol), latent_w0, threaded, nu,
                     baseline_sd, centered_lam0, k2_centered, model_v2, waning)
    model = model_fn === nothing ? hier_model(d, priors) : model_fn(d, priors)      # model_fn: experimental parameterisations (scripts/p6_*)
    vi = DynamicPPL.link(DynamicPPL.initialize_parameters!!(DynamicPPL.VarInfo(Random.MersenneTwister(0), model), start, model), model)
    dim = length(vi[:])
    ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi; adtype = AutoForwardDiff(; chunksize = min(dim, chunksize)))
    # the layout must follow the model's own parameter order
    ordered = DynamicPPL.values_as(vi, NamedTuple)
    layout = stage_layout(NamedTuple{keys(ordered)}(Tuple(start[k] for k in keys(ordered))))
    θ0 = unconstrain(layout, start)
    length(θ0) == dim || throw(ErrorException("start does not match the model's parameters ($(length(θ0)) vs $dim)"))
    ℓπ = x -> LogDensityProblems.logdensity(ldf, x)
    ∂ = x -> LogDensityProblems.logdensity_and_gradient(ldf, x)
    return (; d, model, ldf, layout, θ0, ℓπ, ∂ℓπ∂θ = ∂)
end

"Metric from a saved inverse mass: a vector is the diagonal metric, a matrix the dense one."
_stage_metric(Minv::AbstractVector) = AHMC.DiagEuclideanMetric(copy(Minv))
_stage_metric(Minv::AbstractMatrix) = AHMC.DenseEuclideanMetric(copy(Minv))

# Dense metric estimated blockwise: only the covariance entries inside a group (and the diagonal) are kept, so the inverse mass is
# block-diagonal (groups of coordinates dense, groups independent). Same Stan-style shrinkage as AdvancedHMC's WelfordCov.
# `groups[j]` is the group id of coordinate j; coordinates with id 0 form one group (the population parameters).
mutable struct MaskedCov{F<:AbstractFloat} <: AHMC.Adaptation.DenseMatrixEstimator{F}
    n::Int
    n_min::Int
    μ::Vector{F}
    M::Matrix{F}
    δ::Vector{F}
    cov::Matrix{F}
    mask::BitMatrix
end
function MaskedCov(groups::AbstractVector{<:Integer}; n_min::Int = 10)
    D = length(groups)
    mask = BitMatrix([groups[i] == groups[j] for i in 1:D, j in 1:D])
    return MaskedCov{Float64}(0, n_min, zeros(D), zeros(D, D), zeros(D), Matrix{Float64}(I, D, D), mask)
end
AHMC.Adaptation.resize_adaptor!(::MaskedCov, ::Tuple{Int}) = nothing
function AHMC.Adaptation.reset!(wc::MaskedCov)
    wc.n = 0; fill!(wc.μ, 0.0); fill!(wc.M, 0.0)
    return nothing
end
function Base.push!(wc::MaskedCov, s::AbstractVector{<:AbstractFloat})
    wc.n += 1
    wc.δ .= s .- wc.μ
    wc.μ .+= wc.δ ./ wc.n
    wc.M .+= (s .- wc.μ) * wc.δ'
    return nothing
end
function AHMC.Adaptation.get_estimation(wc::MaskedCov)
    n = float(wc.n)
    @assert n >= 2 "Cannot get covariance with only one sample"
    return (n / ((n + 5) * (n - 1)) .* wc.M .+ (1e-3 * (5 / (n + 5))) .* Matrix{Float64}(I, size(wc.M)...)) .* wc.mask
end

"Group id per coordinate of the layout: 0 for population parameters, the unit index for per-unit coordinates (needs the k2 centring mask)."
function stage_unit_groups(layout::StageLayout, k2_centered::AbstractVector{Bool})
    cidx = findall(k2_centered); nidx = findall(.!k2_centered)
    g = Int[]
    for (k, n) in zip(layout.keys, layout.lens)
        if n == 1 && !(k in (:llam0, :z_λ0, :lk2c, :z_k2, :z_Vtr, :llam1, :z_w0c, :z_w0t))
            push!(g, 0)
        elseif k === :lk2c
            append!(g, cidx)
        elseif k === :z_k2
            append!(g, nidx)
        else
            append!(g, 1:n)
        end
    end
    return g
end

_stat_row(st) = (float(st.n_steps), float(st.tree_depth), float(st.step_size), float(st.acceptance_rate),
                 float(st.numerical_error), float(st.log_density), float(st.hamiltonian_energy))
const STAT_NAMES = (:n_steps, :tree_depth, :step_size, :acceptance_rate, :numerical_error, :lp, :hamiltonian_energy)

function _stage_log(logger, msg)
    logger === nothing ? (println(msg); flush(stdout)) : logger(msg)
end

"""
    stage_warmup(tgt, θ0, rng; nadapt=500, δ=0.8, max_depth=10, log_every=25, logger=nothing, label="",
                 resume=nothing, upto=nadapt) -> (; θ, ϵ, Minv, stats, time, i, adaptor, done)

NUTS with Stan-style windowed adaptation of the step size and the diagonal metric, from `θ0`, `nadapt` iterations in total.
Returns the final position, the adapted step size and the adapted inverse diagonal metric (and the per-iteration stats).

Resumable: with `upto < nadapt` only iterations `1:upto` run and the result carries `adaptor` (the Stan adaptor object: window
bookkeeping, dual-averaging and variance-estimator state, parameters only) and `i`; pass that result back as `resume=` to
continue with iterations `i+1:upto′`. `init_buffer`/`term_buffer`/`window_size` are the Stan adaptation windows (defaults 75/50/25). Only the last segment (i == nadapt) finalises the step size and metric (`done = true`).
"""
function stage_warmup(tgt, θ0::AbstractVector, rng::AbstractRNG; nadapt::Integer = 500, δ::Real = 0.8, max_depth::Integer = 10,
                      Δ_max::Real = 1000.0, log_every::Integer = 25, logger = nothing, label::AbstractString = "",
                      resume = nothing, upto::Integer = nadapt, init_buffer::Integer = 75, term_buffer::Integer = 50,
                      window_size::Integer = 25, metric_kind::Symbol = :diag, groups = nothing)
    D = length(θ0)
    if resume === nothing
        metric = metric_kind in (:dense, :block) ? AHMC.DenseEuclideanMetric(D) : AHMC.DiagEuclideanMetric(D)
        h = AHMC.Hamiltonian(metric, tgt.ℓπ, tgt.∂ℓπ∂θ)
        ϵ0 = AHMC.find_good_stepsize(rng, h, θ0)
        integ = AHMC.Leapfrog(ϵ0)
        pc = metric_kind === :nutpie ? AHMC.NutpieVar((D,)) :
             metric_kind === :block ? MaskedCov(groups === nothing ? error("metric_kind = :block needs groups") : groups) :
             AHMC.MassMatrixAdaptor(metric)
        adaptor = AHMC.StanHMCAdaptor(pc, AHMC.StepSizeAdaptor(δ, integ); init_buffer, term_buffer, window_size)
        i0 = 0
        θstart = copy(θ0)
    else
        metric = _stage_metric(resume.Minv)
        h = AHMC.Hamiltonian(metric, tgt.ℓπ, tgt.∂ℓπ∂θ)
        integ = AHMC.Leapfrog(float(resume.ϵ))
        adaptor = resume.adaptor
        i0 = resume.i
        θstart = copy(resume.θ)
    end
    κ = AHMC.HMCKernel(AHMC.Trajectory{AHMC.MultinomialTS}(integ, AHMC.GeneralisedNoUTurn(max_depth, Δ_max)))
    h, t = AHMC.sample_init(rng, h, θstart)
    nseg = upto - i0
    stats = Matrix{Float64}(undef, nseg, length(STAT_NAMES))
    t0 = time(); tl = t0
    for j in 1:nseg
        i = i0 + j
        t = AHMC.transition(rng, h, κ, t.z)
        st = AHMC.stat(t)
        h, κ, _ = AHMC.adapt!(h, κ, adaptor, i, nadapt, t.z, st.acceptance_rate)
        stats[j, :] .= _stat_row(st)
        if log_every > 0 && i % log_every == 0
            w = max(1, j - log_every + 1):j
            _stage_log(logger, "  [warmup$(label)] iter $i/$nadapt, $(round(time() - t0; digits = 0)) s elapsed; last $log_every: " *
                       "mean n_steps $(round(mean(stats[w, 1]); digits = 1)), tree depth $(round(mean(stats[w, 2]); digits = 2)), " *
                       "step size $(round(mean(stats[w, 3]); sigdigits = 3)), $(round((time() - tl) / log_every; digits = 2)) s/iter")
            tl = time()
        end
    end
    return (; θ = copy(t.z.θ), ϵ = κ.τ.integrator.ϵ, Minv = copy(h.metric.M⁻¹), stats, time = time() - t0, i = upto,
            adaptor, done = upto >= nadapt)
end

"""
    stage_sample(tgt, θ0, ϵ, Minv, rng; ndraws=250, burnin=0, max_depth=10, log_every=25, logger=nothing, label="")
        -> (; θ, stats, final, time)

NUTS with the given FIXED step size and diagonal metric: `AHMC.adapt!` is never called, so nothing is adapted. `burnin`
iterations are run and discarded first (no adaptation either). `θ` is ndraws × dim (unconstrained), `final` the last position.
"""
function stage_sample(tgt, θ0::AbstractVector, ϵ::Real, Minv::AbstractVecOrMat, rng::AbstractRNG; ndraws::Integer = 250,
                      burnin::Integer = 0, max_depth::Integer = 10, Δ_max::Real = 1000.0, log_every::Integer = 25,
                      logger = nothing, label::AbstractString = "")
    metric = _stage_metric(Minv)
    h = AHMC.Hamiltonian(metric, tgt.ℓπ, tgt.∂ℓπ∂θ)
    κ = AHMC.HMCKernel(AHMC.Trajectory{AHMC.MultinomialTS}(AHMC.Leapfrog(float(ϵ)), AHMC.GeneralisedNoUTurn(max_depth, Δ_max)))
    h, t = AHMC.sample_init(rng, h, copy(θ0))
    D = length(θ0)
    θs = Matrix{Float64}(undef, ndraws, D)
    stats = Matrix{Float64}(undef, ndraws, length(STAT_NAMES))
    t0 = time(); tl = t0
    ntot = burnin + ndraws
    sw = zeros(0, length(STAT_NAMES))
    for i in 1:ntot
        t = AHMC.transition(rng, h, κ, t.z)
        st = AHMC.stat(t)
        row = collect(_stat_row(st))
        if i > burnin
            θs[i-burnin, :] .= t.z.θ
            stats[i-burnin, :] .= row
        end
        sw = vcat(sw, row')
        if log_every > 0 && i % log_every == 0
            _stage_log(logger, "  [sample$(label)] iter $i/$ntot ($(i <= burnin ? "burn-in" : "sampling")), $(round(time() - t0; digits = 0)) s elapsed; " *
                       "last $log_every: mean n_steps $(round(mean(sw[:, 1]); digits = 1)), tree depth $(round(mean(sw[:, 2]); digits = 2)), " *
                       "step size $(round(mean(sw[:, 3]); sigdigits = 3)), $(round((time() - tl) / log_every; digits = 2)) s/iter")
            tl = time(); sw = zeros(0, length(STAT_NAMES))
        end
    end
    # no adaptation happened: the kernel and metric are exactly what was passed in
    @assert κ.τ.integrator.ϵ == float(ϵ) && h.metric.M⁻¹ == Minv
    return (; θ = θs, stats, final = copy(t.z.θ), time = time() - t0)
end

"MCMCChains `Chains` from constrained draws (ndraw × dim) and the stats matrix; internals follow Turing's names."
function stage_chain(layout::StageLayout, draws::AbstractMatrix, stats::AbstractMatrix)
    data = hcat(draws, stats)
    sn = collect(STAT_NAMES)[1:size(stats, 2)]
    nm = vcat(layout.names, sn)
    return Chains(reshape(data, size(data, 1), size(data, 2), 1), nm, Dict(:internals => sn))
end

"Seeds: warmup chain K, sample part P (parts count from 1); deterministic and distinct."
stage_seed(base::Integer, K::Integer, P::Integer = 0) = 1_000_003 * Int(base) + 10_007 * Int(K) + 101 * Int(P) + 7

# ---------------------------------------------------------------------------------------
# Pointwise log-likelihoods for PSIS-LOO
# ---------------------------------------------------------------------------------------

# accept raw units or already prepared UnitData; units with an invalid baseline are dropped (with a warning)
_prep_units(units, lod) = (!isempty(units) && first(units) isa UnitData) ? collect(UnitData, units) :
                          first(prepare_units(units; lod))

"""
    pointwise_index(units; lod=detection_limit()) -> NamedTuple(model, arm, t, censored, group)

Column description of `pointwise_loglik(pd, units)`: for each unit (in the order of `units`)
control observations (t > 0) first, then treated. `group[k]` is the unit index (PDX model).
"""
function pointwise_index(units; lod = _default_lod())
    units = _prep_units(units, lod)
    mdl = String[]; arm = Symbol[]; t = Float64[]; cn = Bool[]; grp = Int[]
    for (i, u) in enumerate(units)
        ud = u
        for (a, c) in ((:control, ud.control), (:treated, ud.treated))
            for j in eachindex(c.t)
                push!(mdl, ud.model); push!(arm, a); push!(t, c.t[j]); push!(cn, c.cens[j]); push!(grp, i)
            end
        end
    end
    return (; model = mdl, arm, t, censored = cn, group = grp)
end

"""
    pointwise_loglik(pd, units; lod=nothing, threaded=true) -> Matrix (ndraw × nobs)

Log-likelihood of every post-baseline observation under each posterior draw (conditional on
that draw's per-PDX random effects and, for latent-baseline fits, that draw's own baselines
`pd.w0_*_draws`, i.e. the "conditional" log-lik for within-tumour LOO of the model that was fitted).
Columns follow `pointwise_index(units)`; units are matched to `pd.models` by id. The day-0
observation is excluded, censored observations contribute the log-cdf term.
"""
function pointwise_loglik(pd::PosteriorDraws, units; lod = nothing, threaded = true,
                          draws = nothing)
    set = pd.diagnostics["settings"]
    lodv = lod === nothing ? get(set, "lod", _default_lod()) : float(lod)
    cfg = SolverCfg(; reltol = get(set, "reltol", 1e-5), abstol = get(set, "abstol", 1e-5))
    nu = float(get(set, "nu", Inf))                 # fits saved before the Student-t change used Normal residuals
    v2 = get(pd.diagnostics, "model", "v1") == "v2"
    wn = Bool(get(pd.diagnostics, "waning", false))
    λ1mat = v2 ? pd.diagnostics["lambda1_draws"] : nothing
    wrate = wn ? pd.diagnostics["waning_rate"] : nothing
    ud = _prep_units(units, lodv)
    col = Dict(m => i for (i, m) in enumerate(pd.models))
    idx = pointwise_index(ud)
    dr = draws === nothing ? (1:length(pd.σ)) : draws
    S = length(dr)
    out = Matrix{Float64}(undef, S, length(idx.t))
    offs = cumsum([0; [length(u.control.t) + length(u.treated.t) for u in ud]])
    expo = make_exposure(pd.pk[1], pd.doses, pd.cref)
    _foreach_unit(S, threaded) do s
        ds = dr[s]
        for (i, u) in enumerate(ud)
            m = col[u.model]
            λ0 = pd.λ0[ds, m]; k2 = pd.k2[ds, m]
            λ1 = v2 ? λ1mat[ds, m] : exp(pd.logλ1[ds]); k1 = exp(pd.logk1[ds]); σ = pd.σ[ds]
            nc = length(u.control.t)
            pwc = zeros(nc); pwt = zeros(length(u.treated.t))
            ex = pd.pk[ds] === pd.pk[1] || pd.pk[ds] == pd.pk[1] ? expo :
                 make_exposure(pd.pk[ds], pd.doses, pd.cref)
            wn && (ex = WaningExposure(ex, wrate[ds]))
            lc = curve_loglik(u.control, λ0, λ1, 1.0, 0.0, log(pd.w0_control_draws[ds, m]), σ, NoDrug(), lodv, cfg; pointwise = pwc, nu)
            lt = curve_loglik(u.treated, λ0, λ1, k1, k2, log(pd.w0_treated_draws[ds, m]), σ, ex, lodv, cfg; pointwise = pwt, nu)
            if !(isfinite(lc) && isfinite(lt))
                fill!(pwc, -Inf); fill!(pwt, -Inf)
            end
            out[s, offs[i]+1:offs[i]+nc] .= pwc
            out[s, offs[i]+nc+1:offs[i+1]] .= pwt
        end
    end
    return out
end

"""
    grouped_loglik(ll, idx) -> Matrix (ndraw × nunit)

Sum pointwise log-lik over each PDX model (one column per unit, in `units` order).
Conditional on the draw's random effects; for new-PDX LOO use `marginal_unit_loglik`.
"""
function grouped_loglik(ll::AbstractMatrix, idx)
    G = maximum(idx.group)
    out = zeros(size(ll, 1), G)
    for k in axes(ll, 2)
        @views out[:, idx.group[k]] .+= ll[:, k]
    end
    return out
end
grouped_loglik(pd::PosteriorDraws, units; kwargs...) =
    grouped_loglik(pointwise_loglik(pd, units; kwargs...), pointwise_index(units))

"""
    marginal_unit_loglik(pd, units; M=64, rng, draws=nothing, eps=nothing) -> Matrix (ndraw × nunit)

log p(y_i | population draw s) for a NEW realisation of unit i's latent effects,

    log mean_m p(y_i | λ0 = exp(μ+ω ε_m1), k2 = exp(μ+ω ε_m2), w0c = v0c exp(σ ε_m3), w0t = v0t exp(σ ε_m4)),

Monte Carlo over M fresh standard-normal vectors (common random numbers across draws). The two
baseline coordinates are only integrated for latent-baseline fits (`diagnostics["latent_w0"]`):
conditional on the day-0 observations the baseline posterior is
log w0 ~ N(log v0, τ²) with τ = `baseline_sd` (flat prior on log w0), so the day-0 observation is used once. For fixed-baseline
fits the baselines stay at the observed day-0 volumes. This is the whole-PDX likelihood block for
grouped (new-PDX) PSIS-LOO. Cost: ndraw × nunit × M × 2 ODE solves. Noisy if the likelihood is
very peaked relative to the population distribution; raise M and check the Monte Carlo error.
`eps` (M × 4) can be supplied for reproducible comparisons.
"""
function marginal_unit_loglik(pd::PosteriorDraws, units; M::Integer = 64,
                              rng::AbstractRNG = Random.default_rng(), lod = nothing,
                              threaded = true, draws = nothing, eps = nothing)
    set = pd.diagnostics["settings"]
    lodv = lod === nothing ? get(set, "lod", _default_lod()) : float(lod)
    cfg = SolverCfg(; reltol = get(set, "reltol", 1e-5), abstol = get(set, "abstol", 1e-5))
    latent = Bool(get(pd.diagnostics, "latent_w0", get(set, "latent_w0", false)))
    nu = float(get(set, "nu", Inf))
    bsd = get(set, "baseline_sd", nothing)           # legacy fits (no key): baseline sd was σ
    ud = _prep_units(units, lodv)
    dr = draws === nothing ? (1:length(pd.σ)) : draws
    S = length(dr)
    v2 = get(pd.diagnostics, "model", "v1") == "v2"
    wn = Bool(get(pd.diagnostics, "waning", false))
    ncol = v2 ? 5 : 4                                   # v2 adds a fifth latent coordinate (log Vtr_i)
    ε = eps === nothing ? randn(rng, M, ncol) : Matrix{Float64}(eps)
    size(ε, 2) >= ncol || throw(ArgumentError("eps needs at least $ncol columns for this fit"))
    M = size(ε, 1)
    expo = make_exposure(pd.pk[1], pd.doses, pd.cref)
    out = Matrix{Float64}(undef, S, length(ud))
    _foreach_unit(S, threaded) do s
        ds = dr[s]
        k1 = exp(pd.logk1[ds]); σ = pd.σ[ds]
        ρc = v2 ? pd.diagnostics["corr_lk"][ds] : 0.0
        λ1pool = exp(pd.logλ1[ds])
        tmp = Vector{Float64}(undef, M)
        for (i, u) in enumerate(ud)
            hd = HierData([u], expo, lodv, cfg, latent, false, false; nu, waning = wn)
            τ = bsd === nothing ? σ : float(bsd)
            lr = wn ? log(pd.diagnostics["waning_rate"][ds]) : nothing
            for m in 1:M
                # v1: independent effects; v2: corr(log λ0, log k2) = ρ (Cholesky of the 2x2 correlation) and λ1 = λ0 Vtr
                llam0 = pd.μ_λ0[ds] + pd.ω_λ0[ds] * ε[m, 1]
                lk2 = pd.μ_k2[ds] + pd.ω_k2[ds] * (ρc * ε[m, 1] + sqrt(1 - ρc^2) * ε[m, 2])
                llam1 = v2 ? llam0 + pd.diagnostics["mu_logVtr"][ds] + pd.diagnostics["omega_Vtr"][ds] * ε[m, 5] : log(λ1pool)
                lwc = log(u.control.v0) + (latent ? τ * ε[m, 3] : 0.0)
                lwt = log(u.treated.v0) + (latent ? τ * ε[m, 4] : 0.0)
                tmp[m] = unit_loglik(u, llam0, lk2, llam1, log(k1), σ, lwc, lwt, hd; lrho = lr)
            end
            out[s, i] = logsumexp(tmp) - log(M)
        end
    end
    return out
end

# ---------------------------------------------------------------------------------------
# Synthetic data (used by tests and scripts/synthetic_recovery.jl)
# ---------------------------------------------------------------------------------------

struct SynthCurve
    model::String
    tumour_type::String
    treatment::String
    t::Vector{Float64}
    v::Vector{Float64}
end
struct SynthUnit
    model::String
    tumour_type::String
    control::SynthCurve
    treated::SynthCurve
end

"""
    simulate_units(rng, N, doses, pk, cref; truth, w0_med=180, lod=detection_limit(), tspan=(21, 40), wmax=2000,
                   nu=5, baseline_sd=0.1)

Simulate `N` PDX units from known population parameters `truth` =
(μ_λ0, ω_λ0, μ_k2, ω_k2, logλ1, logk1, σ). Observation times ~ twice a week to a per-unit
horizon in [21, 40]; baseline (day-0) volume is itself a noisy observation of the true start
volume. Curves end at the first observation above `wmax` (ethical endpoint). Residuals are Student-t(`nu`) scaled by σ (Inf = Normal); the day-0 observation has fixed log error sd `baseline_sd`. Observations whose true-volume + noise falls below `lod` are recorded as 0 (undetectable).
Returns `(units, tru)` with `tru.λ0`, `tru.k2` the per-unit truths.
"""
function simulate_units(rng::AbstractRNG, N::Integer, doses, pk, cref; truth,
                        w0_med = 180.0, w0_sd = 0.25, lod = _default_lod(), tspan = (21.0, 40.0), wmax = 2000.0,
                        nu = 5.0, baseline_sd = 0.1)
    expo0 = make_exposure(pk, doses, cref)
    expo = haskey(truth, :ρ_w) ? WaningExposure(expo0, truth.ρ_w) : expo0       # optional waning efficacy (1/day)
    z1 = randn(rng, N); z2 = randn(rng, N)
    ρc = haskey(truth, :corr) ? truth.corr : 0.0                                   # v2 truth: corr(log λ0, log k2)
    λ0 = exp.(truth.μ_λ0 .+ truth.ω_λ0 .* z1)
    k2 = exp.(truth.μ_k2 .+ truth.ω_k2 .* (ρc .* z1 .+ sqrt(1 - ρc^2) .* z2))
    Vtr = haskey(truth, :μ_Vtr) ? exp.(truth.μ_Vtr .+ truth.ω_Vtr .* randn(rng, N)) : fill(NaN, N)   # per-PDX transition volume (v2)
    λ1v = haskey(truth, :μ_Vtr) ? λ0 .* Vtr : fill(exp(truth.logλ1), N)
    k1 = exp(truth.logk1); σ = truth.σ
    units = SynthUnit[]
    w0true = zeros(2, N)
    for i in 1:N
        tend = rand(rng, tspan[1]:tspan[2])
        ts = unique(sort([0.0; collect(3.0:3.5:tend); tend]))
        ts = unique(round.(ts; digits = 0))
        tt = ts[ts .> 0]
        out = Vector{Vector{Float64}}()
        for arm in (:control, :treated)
            w0 = w0_med * exp(w0_sd * randn(rng))
            w0true[arm === :control ? 1 : 2, i] = w0
            θ = Simeoni(λ0[i], λ1v[i], arm === :control ? 1.0 : k1, arm === :control ? 0.0 : k2[i], w0)
            ex = arm === :control ? NoDrug() : expo
            w, ok = simulate_tumour(θ, ex, tt; reltol = 1e-9, abstol = 1e-9)
            ok || error("synthetic simulation failed")
            noise() = isinf(nu) ? randn(rng) : rand(rng, TDist(nu))      # Student-t(ν) scaled by σ
            vobs = [x * exp(σ * noise()) for x in w]
            vobs = [x < lod ? 0.0 : x for x in vobs]
            vfull = [w0 * exp(baseline_sd * randn(rng)); vobs]
            # ethical endpoint: curve ends at the first observation above wmax
            kend = something(findfirst(>(wmax), vfull), length(vfull))
            push!(out, vfull[1:kend])
        end
        push!(units, SynthUnit("syn$(i)", "synthetic",
            SynthCurve("syn$(i)", "synthetic", "untreated", [0.0; tt][1:length(out[1])], out[1]),
            SynthCurve("syn$(i)", "synthetic", "drug", [0.0; tt][1:length(out[2])], out[2])))
    end
    return units, (; λ0, k2, λ1 = λ1v, Vtr, k1, σ, w0c = w0true[1, :], w0t = w0true[2, :])
end
