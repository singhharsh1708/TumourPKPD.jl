# Computation layer of the explorer: no plotting, no Bonito. The UI (explorer.jl), the headless smoke test
# and the static screenshots all go through these functions.
#
# Exposure and doses are in the units of the fit: days, and doses as multiples of the PDXE reference
# regimen (the fit is normalised by the Cmax of that regimen, so the absolute mg/kg never matters here).

module ExplorerCore

using TumourPKPD, Statistics, Random, JLD2
using TumourPKPD: Dose

include("demo_fits.jl")

export DRUGS, SCHEDULES, NEWPDX, Params, FitHandle, load_fits, update_view, trial_view,
       build_doses, extrapolation_info, reference_doses,
       clear_caches!, TRIAL_CATS, trial_matches, trial_duration, make_demo_fit, write_demo_fits

const DRUGS = ["binimetinib", "BKM120", "BYL719"]
const SCHEDULES = ["qd", "bid", "q2d", "5-on-2-off"]
const NEWPDX = "New PDX (population)"
const NDRAWS = 200            # posterior draws used for the main plot
const TRIAL_CATS = ("mCR", "mPR", "mSD", "mPD")
const CACHE_MAX = 96

# ---------------------------------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------------------------------

"Everything the app needs about one fitted (or demo) drug."
struct FitHandle
    drug::String
    pd::PosteriorDraws
    demo::Bool
    label::String                     # short provenance label shown in the UI
    path::String
    draw_idx::Vector{Int}             # fixed subset of stored draws used for the main plot
    units::Dict{String,PDXUnit}
    model_labels::Vector{String}      # "X-1234 · NSCLC", same order as pd.models
    ref_doses::Vector{Dose}
    ref_amt::Float64
    ref_interval::Float64
    ref_route::Symbol
    ref_schedule::String              # which of SCHEDULES matches the reference regimen
    pkprior::NamedTuple
    window_models::Dict{String,Float64}  # last treated observation per PDX model (days)
    window_drug::Float64              # 90th percentile of the last treated observation (new PDX)
    eval_window::Float64              # window on which PDXE response categories are scored (days)
    observed_counts::Dict{String,Int} # published PDXE response categories for this drug
    lock::ReentrantLock
    cache::Dict{Any,Any}
    cache_order::Vector{Any}
end

"""
    Params

Everything the user controls. `model === nothing` means a new PDX (population prediction).
`mult` is the multiplier on the per-dose amount of the PDXE reference regimen.
"""
Base.@kwdef struct Params
    drug::String
    model::Union{Nothing,String} = nothing
    mult::Float64 = 1.0
    schedule::String = "qd"
    duration::Float64 = 28.0
    horizon::Float64 = 60.0
    obs_noise::Bool = true
end

function _cached!(f, h::FitHandle, key)
    hit = lock(h.lock) do
        haskey(h.cache, key) ? Some(h.cache[key]) : nothing
    end
    hit === nothing || return something(hit)
    val = f()
    lock(h.lock) do
        if !haskey(h.cache, key)
            h.cache[key] = val
            push!(h.cache_order, key)
            while length(h.cache_order) > CACHE_MAX
                delete!(h.cache, popfirst!(h.cache_order))
            end
        end
    end
    return val
end

clear_caches!(h::FitHandle) = lock(h.lock) do
    empty!(h.cache); empty!(h.cache_order)
end

# ---------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------

function _schedule_of(interval::Float64)
    interval ≈ 1.0 && return "qd"
    interval ≈ 0.5 && return "bid"
    interval ≈ 2.0 && return "q2d"
    return "qd"
end

function make_handle(drug::String, pd::PosteriorDraws, path::String)
    demo = get(pd.diagnostics, "demo", false) === true
    units = drug_dataset(drug)
    umap = Dict(u.model => u for u in units)
    tt = Dict(u.model => u.tumour_type for u in units)
    labels = [string(m, " · ", get(tt, m, "?")) for m in pd.models]
    _, pkp = regimen(drug)                      # PK prior and status strings (data/regimens.toml)
    ref = pd.doses                              # regimen the fit was normalised to
    interval = length(ref) > 1 ? ref[2].time - ref[1].time : 1.0
    lastd = [maximum(u.treated.t) for u in units]
    wm = Dict(u.model => maximum(u.treated.t) for u in units)
    wdrug = quantile(lastd, 0.9)
    evalw = round(median(lastd))
    cnt = Dict(c => 0 for c in TRIAL_CATS)
    try
        m = load_metrics()
        for (tr, c) in zip(m.treatment, m.response_category)
            tr == drug && haskey(cnt, c) && (cnt[c] += 1)
        end
    catch err
        @warn "could not read published response categories" exception = err
    end
    S = length(pd.σ)
    idx = S <= NDRAWS ? collect(1:S) : sort!(randperm(MersenneTwister(2026), S)[1:NDRAWS])
    label = demo ? "DEMO — not fitted" : "fitted posterior"
    return FitHandle(drug, pd, demo, label, path, idx, umap, labels, collect(Dose, ref), ref[1].amt,
        interval, ref[1].route, _schedule_of(interval), pkp, wm, wdrug, evalw, cnt,
        ReentrantLock(), Dict{Any,Any}(), Any[])
end

"""
    load_fits(dir; demo_dir=nothing, drugs=DRUGS) -> Dict{String,FitHandle}

Reads `dir/<drug>.jld2` for each drug. A drug without a file is taken from `demo_dir/<drug>.jld2`
(written on demand) when `demo_dir` is given, and then labelled "DEMO — not fitted"; otherwise it is skipped.
"""
function load_fits(dir::AbstractString; demo_dir = nothing, drugs = DRUGS)
    out = Dict{String,FitHandle}()
    for d in drugs
        p = joinpath(dir, "$d.jld2")
        if isfile(p)
            out[d] = make_handle(d, load_fit(p), p)
        elseif demo_dir !== nothing
            dp = joinpath(demo_dir, "$d.jld2")
            if !isfile(dp)
                @info "writing demo fit" drug = d path = dp
                save_fit(dp, make_demo_fit(d))
            end
            @warn "no fit for $d in $dir: using a DEMO fit (synthetic, not fitted)" path = dp
            out[d] = make_handle(d, load_fit(dp), dp)
        else
            @warn "no fit for $d in $dir: drug skipped"
        end
    end
    isempty(out) && error("no fits found in $dir")
    return out
end

# ---------------------------------------------------------------------------------------------------
# Regimens
# ---------------------------------------------------------------------------------------------------

"""
    build_doses(h, mult, schedule, duration) -> Vector{Dose}

`mult` × the per-dose amount of the PDXE reference regimen, on `schedule` for `duration` days
(`TumourPKPD.make_regimen`; 5-on-2-off is the union of five-day blocks, each built with `make_regimen`).
"""
function build_doses(h::FitHandle, mult::Real, schedule::AbstractString, duration::Real)
    mk(; kw...) = TumourPKPD.make_regimen(; amt_multiplier = mult, route = h.ref_route, ref_amt = h.ref_amt, kw...)
    schedule == "qd" && return mk(interval_days = 1.0, duration_days = duration)
    schedule == "bid" && return mk(interval_days = 0.5, duration_days = duration)
    schedule == "q2d" && return mk(interval_days = 2.0, duration_days = duration)
    if schedule == "5-on-2-off"
        out = Dose[]
        for w in 0.0:7.0:(duration - 1e-9)
            n = min(5.0, duration - w)
            append!(out, mk(interval_days = 1.0, duration_days = n, start = w))
        end
        return out
    end
    throw(ArgumentError("unknown schedule $schedule; use one of $SCHEDULES"))
end

"The PDXE reference regimen, truncated at `tmax` days (later doses cannot affect a shorter simulation)."
reference_doses(h::FitHandle, tmax::Real) = [d for d in h.ref_doses if d.time <= tmax]

_truncate(doses, tmax) = [d for d in doses if d.time <= tmax]

# ---------------------------------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------------------------------

"""
    extrapolation_info(h, p, doses) -> (flag, reasons)

`TumourPKPD.is_extrapolation(ref, doses; horizon, observed_max_day)`, where
`observed_max_day` is the last treated observation of the selected PDX (or the 90th percentile over the
drug's models for a new PDX). `reasons` is a vector of strings.
"""
function extrapolation_info(h::FitHandle, p::Params, doses::Vector{Dose})
    win = p.model === nothing ? h.window_drug : get(h.window_models, p.model, h.window_drug)
    flag, rs = TumourPKPD.is_extrapolation(h.ref_doses, doses; horizon = p.horizon, observed_max_day = win)
    reasons = String.(rs)
    p.model === nothing &&
        push!(reasons, "new-PDX prediction: tumour-specific growth and sensitivity are drawn from the fitted between-PDX distribution")
    return (flag, reasons)
end

# ---------------------------------------------------------------------------------------------------
# Main view: predictive bands + observed data
# ---------------------------------------------------------------------------------------------------

const LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

_seed(h, p) = hash((h.drug, p.model === nothing ? "" : p.model))

function _tgrid(horizon)
    n = clamp(round(Int, horizon) + 1, 41, 161)
    return collect(range(0.0, float(horizon); length = n))
end

function _bands(sims)
    return predictive_bands(sims; levels = LEVELS)   # ntime × 5
end

function _model_args(h, p)
    p.model === nothing ? (; newpdx = true) : (; model = p.model)
end

function _control_bands(h::FitHandle, p::Params, tg)
    key = (:control, p.model, p.horizon, p.obs_noise)
    _cached!(h, key) do
        rng = MersenneTwister(_seed(h, p))
        sims = simulate_control(h.pd; tgrid = tg, draws = h.draw_idx, obs_noise = p.obs_noise,
                                rng = rng, _model_args(h, p)...)
        _bands(sims)
    end
end

function _treated_bands(h::FitHandle, p::Params, doses, tg)
    key = (:treated, p.model, round(p.mult; digits = 4), p.schedule, p.duration, p.horizon, p.obs_noise)
    _cached!(h, key) do
        rng = MersenneTwister(_seed(h, p))
        sims = simulate_regimen(h.pd, _truncate(doses, p.horizon); tgrid = tg, draws = h.draw_idx,
                                obs_noise = p.obs_noise, rng = rng, _model_args(h, p)...)
        _bands(sims)
    end
end

"Observed PDXE curves (control and treated) of the selected model, or empty vectors for a new PDX."
function observed(h::FitHandle, model)
    model === nothing && return (; tc = Float64[], vc = Float64[], tt = Float64[], vt = Float64[])
    u = get(h.units, model, nothing)
    u === nothing && return (; tc = Float64[], vc = Float64[], tt = Float64[], vt = Float64[])
    return (; tc = u.control.t, vc = u.control.v, tt = u.treated.t, vt = u.treated.v)
end

"""
    update_view(h, p) -> NamedTuple

Everything the main plot and the banner need. Cached per (drug, model, regimen, horizon). Reports
`elapsed` (seconds) and whether the result came from the cache.
"""
function update_view(h::FitHandle, p0::Params)
    t0 = time()
    p = Params(; drug = p0.drug, model = p0.model, mult = p0.mult, schedule = p0.schedule,
               duration = min(p0.duration, p0.horizon), horizon = p0.horizon, obs_noise = p0.obs_noise)
    tg = _tgrid(p.horizon)
    doses = build_doses(h, p.mult, p.schedule, p.duration)
    nin = lock(() -> length(h.cache), h.lock)
    ctl = _control_bands(h, p, tg)
    trt = _treated_bands(h, p, doses, tg)
    flag, reasons = extrapolation_info(h, p, doses)
    win = p.model === nothing ? h.window_drug : get(h.window_models, p.model, h.window_drug)
    nout = lock(() -> length(h.cache), h.lock)
    return (; params = p, t = tg, control = ctl, treated = trt, obs = observed(h, p.model),
            window = win, extrap = flag, reasons, ndoses = length(doses),
            elapsed = time() - t0, cached = nout == nin)
end

# ---------------------------------------------------------------------------------------------------
# Virtual trial
# ---------------------------------------------------------------------------------------------------

function _trial(h::FitHandle, doses, window; quick::Bool, population::Bool, seed)
    # the population mode is documented for fewer draws (each draw already simulates n_pdx_population PDX)
    kw = population ? (; population = true, ndraws = 50) : (;)
    r = TumourPKPD.simulate_trial(h.pd, doses; quick, horizon = window, rng = MersenneTwister(seed), kw...)
    s = r.summary
    return (; responder = s.responder_rate, cats = Dict(c => getproperty(s, Symbol(c)) for c in TRIAL_CATS),
            n_pdx = r.n_pdx, ndraws = r.ndraws, horizon = r.horizon, n_na = r.n_na,
            interval_kind = String(r.interval_kind), population = r.population)
end

"Effective trial treatment duration: the chosen treatment duration capped at the trial scoring window."
trial_duration(h::FitHandle, p::Params) = min(p.duration, h.eval_window)

"""
    trial_view(h, p; quick=true, population=false) -> NamedTuple

Virtual PDX trial (`TumourPKPD.simulate_trial`) for the current regimen and the PDXE reference regimen,
cached; the reference is computed once per drug and mode. Both use the same seed (common random numbers).
The tested regimen is built from the treatment duration only, capped at the scoring window: the viewing
horizon of the main plot never changes it. Responder rates depend on follow-up length: every result
carries its `horizon`, the drug's median PDXE follow-up. `quick=true` is 50 draws × 100 PDX;
`quick=false` is 200 × 200; `population=true` (with `quick=false`) is 50 draws × 1000 PDX and the interval is
then a credible interval of the population response rate (see `interval_kind`).

Every result carries `prov`, the inputs it was computed for; use [`trial_matches`](@ref) before showing it.
`extrapolation_current` / `extrapolation_reference` are `is_extrapolation(ref, doses; horizon, observed_max_day)`
for the tested doses and the scoring horizon. Published PDXE category counts are included as context.
"""
function trial_view(h::FitHandle, p::Params; quick::Bool = true, population::Bool = false)
    t0 = time()
    w = h.eval_window
    dur = trial_duration(h, p)
    doses = build_doses(h, p.mult, p.schedule, dur)
    rdoses = reference_doses(h, w)
    same = length(rdoses) == length(doses) && all(a.time == b.time && a.amt == b.amt for (a, b) in zip(rdoses, doses))
    cur = _cached!(h, (:trial, round(p.mult; digits = 4), p.schedule, dur, quick, population)) do
        _trial(h, doses, w; quick, population, seed = 7)
    end
    ref = same ? cur : _cached!(h, (:trial_ref, quick, population)) do
        _trial(h, rdoses, w; quick, population, seed = 7)
    end
    omax = maximum(values(h.window_models))
    ext(d) = TumourPKPD.is_extrapolation(h.ref_doses, d; horizon = w, observed_max_day = omax)
    ec = ext(doses); er = ext(rdoses)
    prov = (; drug = h.drug, demo = h.demo, path = h.path, mult = round(p.mult; digits = 4), schedule = p.schedule,
            duration = dur, horizon = w, quick, population)
    return (; current = cur, reference = ref, observed = h.observed_counts, n_observed = sum(values(h.observed_counts)),
            horizon = w, quick, population, prov,
            extrapolation_current = (flag = ec[1], reasons = String.(ec[2])),
            extrapolation_reference = (flag = er[1], reasons = String.(er[2])),
            elapsed = time() - t0)
end

"True when the trial result `tv` was computed for the drug, fit, dose multiplier, schedule and treatment duration in `p`."
function trial_matches(tv, h::FitHandle, p::Params)
    tv === nothing && return false
    q = tv.prov
    return q.drug == p.drug && q.drug == h.drug && q.demo == h.demo && q.path == h.path &&
           q.mult == round(p.mult; digits = 4) && q.schedule == p.schedule && q.duration == trial_duration(h, p)
end

end # module
