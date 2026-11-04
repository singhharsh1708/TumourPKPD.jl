# Hierarchical Simeoni tumour growth inhibition model, fitted per drug with Turing/NUTS.
#
# Included from src/TumourPKPD.jl AFTER pk.jl and pd.jl (uses Dose, OneCmt, Simeoni,
# NoDrug, PKExposure, simulate_tumour, dose_times from those files).
#
# Unit = PDX model; each has a control curve (NoDrug) and a treated curve (PKExposure).
#
#   logλ0_i = μ_λ0 + ω_λ0 z_λ0_i          (shared by control and treated arm of unit i)
#   logk2_i = μ_k2 + ω_k2 z_k2_i
#   logλ1, logk1, σ                        drug level / pooled
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
using LogExpFunctions: logsumexp

export PosteriorDraws, chain_to_parts, chain_from_parts, valid_baseline, prepare_units, cache_stats, save_fit, load_fit, fit_drug, pointwise_loglik, pointwise_index,
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
- `logλ1 ~ Normal(m, s)`           log linear growth rate (mm^3/day)
- `logk1 ~ Normal(m, s)`           log transit rate (1/day)
- `σ ~ HalfNormal(s)`              log-volume residual sd
"""
Base.@kwdef struct HierPriors
    μ_λ0::NTuple{2,Float64} = (log(0.12), 0.7)
    ω_λ0::Float64 = 0.5
    μ_k2::NTuple{2,Float64} = (log(0.15), 1.3)
    ω_k2::Float64 = 1.0
    logλ1::NTuple{2,Float64} = (log(300.0), 0.9)
    logk1::NTuple{2,Float64} = (log(0.7), 0.6)
    σ::Float64 = 0.3
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
    token::Base.RefValue{Int}        # identity key for the task-local caches
    dropped::Vector{String}          # units removed for an invalid day-0 volume
end
HierData(units, expo, lod, solver, latent_w0, local_grad, threaded, dropped = String[]) =
    HierData(units, expo, lod, solver, latent_w0, local_grad, threaded, Ref(rand(Int)), dropped)

# One cache per (HierData, task): chains running in separate tasks never evict each other's entry
# and no lock is needed. The cache lives in task_local_storage.
_local_cache(d::HierData) = get!(LocalGradCache, task_local_storage(), (:tumourpkpd_cache, d.token[]))::LocalGradCache
"(hits, misses) of the current task's local-gradient cache for `d`."
cache_stats(d::HierData) = (c = _local_cache(d); (c.hits, c.misses))

function prepare_data(units, doses, pk, cref; lod = _default_lod(), solver = SolverCfg(),
                      latent_w0 = true, local_grad = true, threaded = true)
    ud, dropped = prepare_units(units; lod)
    isempty(ud) && throw(ArgumentError("no units with valid baselines"))
    expo = make_exposure(pk, doses, cref)
    return HierData(ud, expo, float(lod), solver, latent_w0, local_grad, threaded, dropped)
end

@inline _lognormpdf(y, μ, σ) = -0.5 * ((y - μ) / σ)^2 - log(σ) - 0.9189385332046727

# Log-likelihood of one curve. Returns -Inf if the ODE solve fails.
# `pointwise` (optional Vector) receives per-observation contributions.
function curve_loglik(c::CurveData, λ0, λ1, k1, k2, logw0, σ, expo, lod, cfg::SolverCfg;
                      pointwise = nothing)
    isempty(c.t) && return zero(promote_type(typeof(λ0), typeof(σ)))
    θ = Simeoni(λ0, λ1, k1, k2, exp(logw0))
    w, ok = simulate_tumour(θ, expo, c.t; solver = cfg.solver, reltol = cfg.reltol, abstol = cfg.abstol)
    T = promote_type(typeof(λ0), typeof(σ), typeof(logw0))
    ok || return T(-Inf)
    logLOD = log(lod)
    s = zero(T)
    for j in eachindex(c.t)
        lw = log(max(w[j], 1e-3))   # floor far below the LOD; eradicated tumours can give w ~ -2e-7
        l = c.cens[j] ? logcdf(Normal(lw, σ), logLOD) : _lognormpdf(c.logv[j], lw, σ)
        pointwise === nothing || (pointwise[j] = ForwardDiff.value(l))
        s += l
    end
    return s
end

# Control arm: no drug, k1/k2 irrelevant.
@inline control_loglik(u::UnitData, llam0, llam1, σ, lw0, d::HierData) =
    curve_loglik(u.control, exp(llam0), exp(llam1), one(llam0), zero(llam0), lw0, σ,
                 NoDrug(), d.lod, d.solver)

@inline treated_loglik(u::UnitData, llam0, llam1, lk1, lk2, σ, lw0, d::HierData) =
    curve_loglik(u.treated, exp(llam0), exp(llam1), exp(lk1), exp(lk2), lw0, σ,
                 d.exposure, d.lod, d.solver)

# --- value + gradient by local forward-mode (no dependence on outer AD) -----------------

struct LocalTag end

@inline function _seed(x::NTuple{K,Float64}) where {K}
    return ntuple(Val(K)) do j
        ForwardDiff.Dual{LocalTag}(x[j], ForwardDiff.Partials(ntuple(i -> i == j ? 1.0 : 0.0, Val(K))))
    end
end

@inline function _valgrad(f, x::NTuple{K,Float64}) where {K}
    r = f(_seed(x)...)
    r isa ForwardDiff.Dual || return (Float64(r), ntuple(_ -> 0.0, Val(K)))
    return (ForwardDiff.value(r), ForwardDiff.partials(r).values)
end

# Local parameters per unit, in this order (always 7 stored; fixed-w0 mode leaves lw0 grads 0):
#   1 llam0  2 lk2  3 llam1  4 lk1  5 σ  6 lw0c  7 lw0t
const NLOC = 7

"""
Value and gradient (length-7 tuple, order above) of one unit's log-likelihood at plain Float64 inputs.

The VALUE is the ordinary Float64 solve, i.e. exactly what `unit_loglik` / `pointwise_loglik` return;
only the gradient comes from the locally seeded dual solve. A failed or non-finite plain solve gives
-Inf in both paths. The single remaining asymmetry: if the plain solve succeeds but the dual solve
fails or has a non-finite derivative the state is rejected (-Inf) because no gradient exists.
"""
function unit_valgrad(u::UnitData, p::NTuple{7,Float64}, d::HierData)
    llam0, lk2, llam1, lk1, σ, lw0c, lw0t = p
    zero7 = ntuple(_ -> 0.0, Val(7))
    v = unit_loglik(u, llam0, lk2, llam1, lk1, σ, lw0c, lw0t, d)
    isfinite(v) || return (-Inf, zero7)
    if d.latent_w0
        _, gc = _valgrad((a, b, c, w) -> control_loglik(u, a, b, c, w, d), (llam0, llam1, σ, lw0c))
        _, gt = _valgrad((a, b, c, e, f, w) -> treated_loglik(u, a, b, c, e, f, w, d),
                         (llam0, llam1, lk1, lk2, σ, lw0t))
        g = (gc[1] + gt[1], gt[4], gc[2] + gt[2], gt[3], gc[3] + gt[5], gc[4], gt[6])
    else
        _, gc = _valgrad((a, b, c) -> control_loglik(u, a, b, c, lw0c, d), (llam0, llam1, σ))
        _, gt = _valgrad((a, b, c, e, f) -> treated_loglik(u, a, b, c, e, f, lw0t, d),
                         (llam0, llam1, lk1, lk2, σ))
        g = (gc[1] + gt[1], gt[4], gc[2] + gt[2], gt[3], gc[3] + gt[5], 0.0, 0.0)
    end
    all(isfinite, g) || return (-Inf, zero7)
    return (v, g)
end

# Plain (generic number type) unit log-likelihood.
function unit_loglik(u::UnitData, llam0, lk2, llam1, lk1, σ, lw0c, lw0t, d::HierData)
    return control_loglik(u, llam0, llam1, σ, lw0c, d) +
           treated_loglik(u, llam0, llam1, lk1, lk2, σ, lw0t, d)
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
    population_loglik(d, llam0, lk2, lw0c, lw0t, llam1, lk1, σ)

Sum over units of the unit log-likelihoods. Each unit's value is computed in a pure function
into a preallocated buffer (no `~` inside threads); the caller adds the sum with
`@addlogprob!`. Buffer element type follows the parameter type.

For ForwardDiff dual inputs with `d.local_grad`, each unit's value and gradient are obtained by
a small local forward pass and the result is stitched back into a dual number through the
chain rule, so cost is linear in the number of units regardless of the outer chunk size.
"""
function population_loglik(d::HierData, llam0::AbstractVector, lk2::AbstractVector,
                           lw0c::AbstractVector, lw0t::AbstractVector,
                           llam1, lk1, σ)
    T = promote_type(eltype(llam0), eltype(lk2), eltype(lw0c), eltype(lw0t),
                     typeof(llam1), typeof(lk1), typeof(σ))
    N = length(d.units)
    buf = Vector{T}(undef, N)
    _foreach_unit(N, d.threaded) do i
        buf[i] = unit_loglik(d.units[i], llam0[i], lk2[i], llam1, lk1, σ, lw0c[i], lw0t[i], d)
    end
    return sum(buf)
end

# local-gradient path for ForwardDiff duals
function population_loglik(d::HierData, llam0::AbstractVector{D}, lk2::AbstractVector{D},
                           lw0c::AbstractVector, lw0t::AbstractVector,
                           llam1, lk1, σ) where {Tg,N,D<:ForwardDiff.Dual{Tg,Float64,N}}
    d.local_grad || return invoke(population_loglik,
        Tuple{HierData,AbstractVector,AbstractVector,AbstractVector,AbstractVector,Any,Any,Any},
        d, llam0, lk2, lw0c, lw0t, llam1, lk1, σ)
    nu = length(d.units)
    val(x) = ForwardDiff.value(x)
    vlam1 = val(llam1); vk1 = val(lk1); vσ = val(σ)
    xv = Vector{Float64}(undef, 3 + 4nu)
    xv[1] = vlam1; xv[2] = vk1; xv[3] = vσ
    for i in 1:nu
        xv[3+4(i-1)+1] = val(llam0[i]); xv[3+4(i-1)+2] = val(lk2[i])
        xv[3+4(i-1)+3] = val(lw0c[i]);  xv[3+4(i-1)+4] = val(lw0t[i])
    end
    cache = _local_cache(d)
    if cache.x == xv
        cache.hits += 1
        vals = cache.vals; grads = cache.grads
    else
        vals = Vector{Float64}(undef, nu)
        grads = Matrix{Float64}(undef, NLOC, nu)
        _foreach_unit(nu, d.threaded) do i
            p = (xv[3+4(i-1)+1], xv[3+4(i-1)+2], vlam1, vk1, vσ, xv[3+4(i-1)+3], xv[3+4(i-1)+4])
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
    g3 = 0.0; g4 = 0.0; g5 = 0.0
    for i in 1:nu
        acc += grads[1, i] * ForwardDiff.partials(llam0[i])
        acc += grads[2, i] * ForwardDiff.partials(lk2[i])
        if d.latent_w0
            acc += grads[6, i] * ForwardDiff.partials(lw0c[i])
            acc += grads[7, i] * ForwardDiff.partials(lw0t[i])
        end
        g3 += grads[3, i]; g4 += grads[4, i]; g5 += grads[5, i]
    end
    acc += g3 * ForwardDiff.partials(llam1)
    acc += g4 * ForwardDiff.partials(lk1)
    acc += g5 * ForwardDiff.partials(σ)
    return D(total, acc)
end

# ---------------------------------------------------------------------------------------
# Turing model
# ---------------------------------------------------------------------------------------

@model function hier_model(d::HierData, pri::HierPriors)
    N = length(d.units)
    μ_λ0 ~ Normal(pri.μ_λ0...)
    ω_λ0 ~ truncated(Normal(0.0, pri.ω_λ0); lower = 0.0)
    μ_k2 ~ Normal(pri.μ_k2...)
    ω_k2 ~ truncated(Normal(0.0, pri.ω_k2); lower = 0.0)
    logλ1 ~ Normal(pri.logλ1...)
    logk1 ~ Normal(pri.logk1...)
    σ ~ truncated(Normal(0.0, pri.σ); lower = 0.0)
    z_λ0 ~ filldist(Normal(), N)
    z_k2 ~ filldist(Normal(), N)
    llam0 = μ_λ0 .+ ω_λ0 .* z_λ0
    lk2 = μ_k2 .+ ω_k2 .* z_k2
    if d.latent_w0
        # Flat prior on log w0 plus the single day-0 observation  =>  logw0 = log v0 + σ z (non-centred).
        z_w0c ~ filldist(Normal(), N)
        z_w0t ~ filldist(Normal(), N)
        lw0c = [log(u.control.v0) for u in d.units] .+ σ .* z_w0c
        lw0t = [log(u.treated.v0) for u in d.units] .+ σ .* z_w0t
    else
        lw0c = [log(u.control.v0) for u in d.units]
        lw0t = [log(u.treated.v0) for u in d.units]
    end
    ll = population_loglik(d, llam0, lk2, lw0c, lw0t, logλ1, logk1, σ)
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
    nt = (; μ_λ0 = μ0 + jitter * randn(rng), ω_λ0 = 0.3 * exp(jitter * randn(rng)),
          μ_k2 = pri.μ_k2[1] + jitter * randn(rng), ω_k2 = 0.5 * exp(jitter * randn(rng)),
          logλ1 = pri.logλ1[1] + jitter * randn(rng), logk1 = pri.logk1[1] + jitter * randn(rng),
          σ = 0.25 * exp(jitter * randn(rng)),
          z_λ0 = jitter .* randn(rng, N), z_k2 = jitter .* randn(rng, N))
    if d.latent_w0
        nt = merge(nt, (; z_w0c = jitter .* randn(rng, N), z_w0t = jitter .* randn(rng, N)))
    end
    return nt
end

# ---------------------------------------------------------------------------------------
# fit_drug
# ---------------------------------------------------------------------------------------

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
                  precomputed_runtime = nothing)
    # :auto = :units. Chains inside one process scale badly (measured: 4 concurrent tasks ran 4x slower
    # per gradient because the ODE solves allocate heavily and share one GC); use one process per chain
    # (scripts/procchains.jl, `chain_only` + `precomputed_chain`) for chain-level parallelism.
    parallel === :auto && (parallel = :units)
    parallel in (:units, :chains, :both) || throw(ArgumentError("parallel must be :auto, :units, :chains or :both"))
    d = prepare_data(units, doses, pk, cref; lod, solver = SolverCfg(; reltol, abstol),
                     latent_w0, local_grad, threaded = parallel in (:units, :both))
    N = length(d.units)
    model = hier_model(d, priors)
    dim = 7 + 2N + (latent_w0 ? 2N : 0)
    cs = chunksize === nothing ? min(dim, local_grad ? 32 : 12) : chunksize
    adtype = AutoForwardDiff(; chunksize = cs)
    inits = if init === nothing
        [initial_values(d, priors, rng) for _ in 1:nchains]
    elseif init isa NamedTuple
        fill(init, nchains)
    else
        collect(init)
    end
    precomputed_chain === nothing || (nchains = size(precomputed_chain, 3))
    sampler = NUTS(nadapt, target_accept; max_depth, adtype)
    t0 = time()
    # optional progress log (flushed): one line every `log_every` iterations of any chain
    cb = if log_every > 0
        count = Threads.Atomic{Int}(0)
        total = nchains * (nsamples + nadapt)
        (rng_, model_, sampler_, sample_, state_, iteration; kwargs...) -> begin
            n = Threads.atomic_add!(count, 1) + 1
            if n % log_every == 0
                msg = "  [fit] $(n)/$(total) iterations, $(round(time() - t0; digits = 0)) s elapsed"
                logger === nothing ? (println(msg); flush(stdout)) : logger(msg)
            end
            nothing
        end
    else
        nothing
    end
    ckw = cb === nothing ? (;) : (; callback = cb)
    chain = if precomputed_chain !== nothing
        precomputed_chain
    elseif nchains == 1
        sample(rng, model, sampler, nsamples; initial_params = inits[1], progress, ckw...)
    else
        alg = parallel in (:chains, :both) ? MCMCThreads() : MCMCSerial()
        sample(rng, model, sampler, alg, nsamples, nchains; initial_params = inits, progress, ckw...)
    end
    chain_only && return chain, nothing
    runtime = precomputed_runtime === nothing ? time() - t0 : float(precomputed_runtime)
    settings = Dict{String,Any}("nchains" => nchains, "nsamples" => nsamples, "nadapt" => nadapt,
        "target_accept" => target_accept, "max_depth" => max_depth, "lod" => float(lod),
        "latent_w0" => latent_w0, "parallel" => String(parallel), "local_grad" => local_grad,
        "reltol" => float(reltol), "abstol" => float(abstol), "chunksize" => cs,
        "nthreads" => Threads.nthreads(), "n_units" => N,
        "priors" => Dict(string(k) => getfield(priors, k) for k in fieldnames(HierPriors)))
    settings["seed"] = seed
    pd = posterior_draws(chain, d, string(drug), doses, pk, cref; runtime, settings, max_depth)
    # bookkeeping for the validation worker (leakage checks)
    pd.diagnostics["lod"] = float(lod)
    pd.diagnostics["nchains"] = nchains
    pd.diagnostics["draws_per_chain"] = nsamples
    pd.diagnostics["train_models"] = [u.model for u in d.units]
    pd.diagnostics["split"] = split isa AbstractDict ? Dict{String,Any}(string(k) => v for (k, v) in pairs(split)) :
                              Dict{String,Any}("kind" => String(split))
    pd.diagnostics["cutoffs"] = cutoffs === nothing ? Dict{String,Float64}() :
                                Dict{String,Float64}(string(k) => float(v) for (k, v) in pairs(cutoffs))
    pd.diagnostics["seed"] = seed
    pd.diagnostics["latent_w0"] = latent_w0
    pd.diagnostics["dropped_units"] = copy(d.dropped)
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
    ni = names_internal(chain)
    nd = :numerical_error in ni ? Int(sum(chain[:numerical_error])) : 0
    td = :tree_depth in ni ? chain[:tree_depth] : nothing
    return Dict{String,Any}(
        "max_rhat" => isempty(rh) ? NaN : maximum(rh),          # NaN = statistic unavailable
        "min_ess_bulk" => isempty(eb) ? NaN : minimum(eb),
        "n_divergences" => nd,
        "mean_acceptance" => :acceptance_rate in ni ? mean(chain[:acceptance_rate]) : NaN,
        "mean_tree_depth" => td === nothing ? NaN : mean(td),
        "max_tree_depth_hits" => td === nothing ? 0 : Int(sum(td .>= max_depth)),
        "mean_step_size" => :step_size in ni ? mean(chain[:step_size]) : NaN)
end
names_internal(chain) = Set(MCMCChains.names(chain, :internals))

function posterior_draws(chain, d::HierData, drug, doses, pk, cref; runtime, settings, max_depth = 10)
    N = length(d.units)
    g(s) = vec(Array(chain[s]))
    μ_λ0 = g(:μ_λ0); ω_λ0 = g(:ω_λ0); μ_k2 = g(:μ_k2); ω_k2 = g(:ω_k2)
    S = length(μ_λ0)
    zl = hcat([g(Symbol("z_λ0[$i]")) for i in 1:N]...)
    zk = hcat([g(Symbol("z_k2[$i]")) for i in 1:N]...)
    λ0 = exp.(μ_λ0 .+ ω_λ0 .* zl)
    k2 = exp.(μ_k2 .+ ω_k2 .* zk)
    σ = g(:σ)
    v0c = [u.control.v0 for u in d.units]; v0t = [u.treated.v0 for u in d.units]
    # per-draw baselines: latent  w0 = v0 exp(σ_s z_s);  fixed  w0 = v0 (constant columns)
    if d.latent_w0
        zc = hcat([g(Symbol("z_w0c[$i]")) for i in 1:N]...)
        zt = hcat([g(Symbol("z_w0t[$i]")) for i in 1:N]...)
        w0c_draws = exp.(log.(permutedims(v0c)) .+ σ .* zc)
        w0t_draws = exp.(log.(permutedims(v0t)) .+ σ .* zt)
        w0c = vec(median(w0c_draws; dims = 1)); w0t = vec(median(w0t_draws; dims = 1))
    else
        w0c = v0c; w0t = v0t
        w0c_draws = repeat(permutedims(v0c), S, 1); w0t_draws = repeat(permutedims(v0t), S, 1)
    end
    diag = chain_diagnostics(chain; max_depth)
    diag["runtime_s"] = runtime
    diag["settings"] = settings
    return PosteriorDraws(drug, [u.model for u in d.units], collect(Dose, doses),
        fill(OneCmt(pk.CL, pk.V, pk.ka, pk.F), S), float(cref),
        μ_λ0, ω_λ0, μ_k2, ω_k2, g(:logλ1), g(:logk1), σ, λ0, k2, w0c, w0t,
        w0c_draws, w0t_draws, diag)
end

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
            λ1 = exp(pd.logλ1[ds]); k1 = exp(pd.logk1[ds]); σ = pd.σ[ds]
            nc = length(u.control.t)
            pwc = zeros(nc); pwt = zeros(length(u.treated.t))
            ex = pd.pk[ds] === pd.pk[1] || pd.pk[ds] == pd.pk[1] ? expo :
                 make_exposure(pd.pk[ds], pd.doses, pd.cref)
            lc = curve_loglik(u.control, λ0, λ1, 1.0, 0.0, log(pd.w0_control_draws[ds, m]), σ, NoDrug(), lodv, cfg; pointwise = pwc)
            lt = curve_loglik(u.treated, λ0, λ1, k1, k2, log(pd.w0_treated_draws[ds, m]), σ, ex, lodv, cfg; pointwise = pwt)
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
conditional on the draw's σ and the day-0 observations the baseline posterior is
log w0 ~ N(log v0, σ²) (flat prior on log w0), so the day-0 observation is used once. For fixed-baseline
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
    ud = _prep_units(units, lodv)
    dr = draws === nothing ? (1:length(pd.σ)) : draws
    S = length(dr)
    ε = eps === nothing ? randn(rng, M, 4) : Matrix{Float64}(eps)
    M = size(ε, 1)
    expo = make_exposure(pd.pk[1], pd.doses, pd.cref)
    out = Matrix{Float64}(undef, S, length(ud))
    _foreach_unit(S, threaded) do s
        ds = dr[s]
        λ1 = exp(pd.logλ1[ds]); k1 = exp(pd.logk1[ds]); σ = pd.σ[ds]
        tmp = Vector{Float64}(undef, M)
        for (i, u) in enumerate(ud)
            hd = HierData([u], expo, lodv, cfg, latent, false, false)
            for m in 1:M
                λ0 = exp(pd.μ_λ0[ds] + pd.ω_λ0[ds] * ε[m, 1])
                k2 = exp(pd.μ_k2[ds] + pd.ω_k2[ds] * ε[m, 2])
                lwc = log(u.control.v0) + (latent ? σ * ε[m, 3] : 0.0)
                lwt = log(u.treated.v0) + (latent ? σ * ε[m, 4] : 0.0)
                tmp[m] = unit_loglik(u, log(λ0), log(k2), log(λ1), log(k1), σ, lwc, lwt, hd)
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
    simulate_units(rng, N, doses, pk, cref; truth, w0_med=180, lod=detection_limit(), tspan=(21, 40), wmax=2000)

Simulate `N` PDX units from known population parameters `truth` =
(μ_λ0, ω_λ0, μ_k2, ω_k2, logλ1, logk1, σ). Observation times ~ twice a week to a per-unit
horizon in [21, 40]; baseline (day-0) volume is itself a noisy observation of the true start
volume. Curves end at the first observation above `wmax` (ethical endpoint). Observations whose true-volume + noise falls below `lod` are recorded as 0 (undetectable).
Returns `(units, tru)` with `tru.λ0`, `tru.k2` the per-unit truths.
"""
function simulate_units(rng::AbstractRNG, N::Integer, doses, pk, cref; truth,
                        w0_med = 180.0, w0_sd = 0.25, lod = _default_lod(), tspan = (21.0, 40.0), wmax = 2000.0)
    expo = make_exposure(pk, doses, cref)
    λ0 = exp.(truth.μ_λ0 .+ truth.ω_λ0 .* randn(rng, N))
    k2 = exp.(truth.μ_k2 .+ truth.ω_k2 .* randn(rng, N))
    λ1 = exp(truth.logλ1); k1 = exp(truth.logk1); σ = truth.σ
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
            θ = Simeoni(λ0[i], λ1, arm === :control ? 1.0 : k1, arm === :control ? 0.0 : k2[i], w0)
            ex = arm === :control ? NoDrug() : expo
            w, ok = simulate_tumour(θ, ex, tt; reltol = 1e-9, abstol = 1e-9)
            ok || error("synthetic simulation failed")
            vobs = [x * exp(σ * randn(rng)) for x in w]
            vobs = [x < lod ? 0.0 : x for x in vobs]
            vfull = [w0 * exp(σ * randn(rng)); vobs]
            # ethical endpoint: curve ends at the first observation above wmax
            kend = something(findfirst(>(wmax), vfull), length(vfull))
            push!(out, vfull[1:kend])
        end
        push!(units, SynthUnit("syn$(i)", "synthetic",
            SynthCurve("syn$(i)", "synthetic", "untreated", [0.0; tt][1:length(out[1])], out[1]),
            SynthCurve("syn$(i)", "synthetic", "drug", [0.0; tt][1:length(out[2])], out[2])))
    end
    return units, (; λ0, k2, λ1, k1, σ, w0c = w0true[1, :], w0t = w0true[2, :])
end
