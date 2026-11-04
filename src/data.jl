# PDXE data access: cleaned PCT curves, per-drug paired datasets, regimens and mouse PK priors.
# Raw data are never read here; run data/download.jl and data/clean.jl first (see data/README.md).
#
# Included by the lead into module TumourPKPD (needs `Dose` and `dosing_schedule` from pk.jl).
# Public names: TumourCurve, PDXUnit, load_pct, load_metrics, regimen, drug_dataset, default_cache_dir,
# detection_limit, list_drugs, DETECTION_LIMIT_MM3.

using CSV, DataFrames

"""
    TumourCurve(model, tumour_type, treatment, t, v)

One animal's tumour-volume curve. `t` in days from the first observation of that curve (which is
treatment start, day 0, for every curve used in `drug_dataset`), `v` in mm^3. `v == 0` means
"no measurable tumour" and is to be treated as left-censored at `detection_limit()`.
"""
struct TumourCurve
    model::String
    tumour_type::String
    treatment::String
    t::Vector{Float64}
    v::Vector{Float64}
end

"""
    PDXUnit(model, tumour_type, control, treated)

One PDX model with an untreated control curve and a treated curve (same model, same day-0 convention).
"""
struct PDXUnit
    model::String
    tumour_type::String
    control::TumourCurve
    treated::TumourCurve
end

"Directory holding the cleaned CSVs (`pct_long.csv`, `pct_metrics.csv`, `treatment_audit.csv`)."
default_cache_dir() = normpath(joinpath(@__DIR__, "..", "data", "cache"))

const _REGIMEN_FILE = normpath(joinpath(@__DIR__, "..", "data", "regimens.toml"))

_regimens_toml() = Base.parsed_toml(_REGIMEN_FILE)   # Base TOML parser: avoids adding the TOML stdlib to Project.toml; switch to TOML.parsefile if preferred

"""
    detection_limit()

Volume (mm^3) below which a recorded 0 is taken to lie. Value comes from `data/regimens.toml` `[data]`
(4 mm^3, see data/README.md for the justification). It is an assumption, not a published number.
"""
detection_limit() = float(_regimens_toml()["data"]["detection_limit_mm3"])
const DETECTION_LIMIT_MM3 = 4.0   # keep in sync with data/regimens.toml (checked in test/test_data.jl)

function _read_long(dir)
    f = joinpath(dir, "pct_long.csv")
    isfile(f) || error("$f not found. Run `julia --project=. data/download.jl` then `julia --project=. data/clean.jl`.")
    CSV.read(f, DataFrame; stringtype = String, types = Dict(:model => String, :tumour_type => String,
        :treatment => String, :arm => String, :treatment_raw => String))
end

# Curves keyed by (model, treatment, replicate), plus the flags needed for pairing.
function _curves(long::DataFrame; replicates::Symbol = :first)
    replicates in (:first, :all) || throw(ArgumentError("replicates must be :first or :all"))
    out = Vector{Tuple{TumourCurve,Bool}}()
    for g in groupby(long, [:model, :treatment, :replicate])
        replicates == :first && g.replicate[1] != 1 && continue
        c = TumourCurve(g.model[1], g.tumour_type[1], g.treatment[1], Float64.(g.day), Float64.(g.volume_mm3))
        push!(out, (c, any(g.t0_shifted)))
    end
    return out
end

"""
    load_pct(; dir=default_cache_dir(), treatments=nothing, replicates=:first, min_points=1)

All cleaned PCT curves (every arm: untreated, single agents, combinations). `treatments` optionally
restricts to a list of treatment labels. Three untreated models (X-025, X-158, X-166) contain several
control animals; `replicates=:first` keeps animal 1 only, `:all` returns every animal as its own curve
(same model id, so do not use `:all` for pairing).
"""
function load_pct(; dir = default_cache_dir(), treatments = nothing, replicates::Symbol = :first, min_points::Int = 1)
    long = _read_long(dir)
    treatments === nothing || (long = long[in.(long.treatment, Ref(Set(treatments))), :])
    cs = [c for (c, _) in _curves(long; replicates = replicates) if length(c.t) >= min_points]
    return cs
end

"Curve-metric table (response_category in mCR/mPR/mSD/mPD, best_response, best_avg_response, ...)."
function load_metrics(; dir = default_cache_dir())
    f = joinpath(dir, "pct_metrics.csv")
    isfile(f) || error("$f not found. Run data/clean.jl first.")
    CSV.read(f, DataFrame; stringtype = String)
end

"Treatments with a regimen entry in data/regimens.toml."
list_drugs() = sort(collect(keys(_regimens_toml()["drugs"])))

"""
    drug_dataset(drug; dir=default_cache_dir(), min_points=3)

One `PDXUnit` per PDX model that has BOTH an untreated and a `drug` arm. A pair is kept when both curves
have at least `min_points` observations and both start at day 0 as recorded (7 curves in the sheet start
later; they are excluded here). Models with several control animals use the first one.
"""
function drug_dataset(drug::AbstractString; dir = default_cache_dir(), min_points::Int = 3)
    long = _read_long(dir)
    long = long[(long.treatment .== drug) .| (long.treatment .== "untreated"), :]
    ctl = Dict{String,TumourCurve}()
    trt = Dict{String,TumourCurve}()
    for (c, shifted) in _curves(long; replicates = :first)
        (length(c.t) >= min_points && !shifted) || continue
        (c.treatment == "untreated" ? ctl : trt)[c.model] = c
    end
    models = sort(collect(intersect(keys(ctl), keys(trt))))
    isempty(models) && @warn "no paired models for treatment $drug"
    return [PDXUnit(m, trt[m].tumour_type, ctl[m], trt[m]) for m in models]
end

"""
    regimen(drug; duration=nothing) -> (doses::Vector{Dose}, pkprior::NamedTuple)

Dosing schedule (mg/kg, days from treatment start) and mouse PK prior for `drug` from data/regimens.toml.
`duration` (days) overrides the dosing horizon (default `duration_days` in the file; doses after a curve's
last observation do not affect it, so a long horizon is harmless).

`pkprior` fields (per-day units): `CL` L/day/kg, `V` L/kg, `ka` 1/day, `F` (fraction) medians; `CL_logsd`,
`V_logsd`, `ka_logsd` are sd of log(parameter); `F_logitsd` is sd of logit(F). Also `dose_status` and
`pk_status` strings saying which numbers are verified and which are assumptions.
"""
function regimen(drug::AbstractString; duration = nothing)
    T = _regimens_toml()
    haskey(T["drugs"], drug) || throw(ArgumentError("no regimen for $drug; available: $(list_drugs())"))
    d = T["drugs"][drug]
    dur = duration === nothing ? float(d["duration_days"]) : float(duration)
    interval = float(d["interval_days"])
    # every scheduled time in [0, dur)
    n = max(ceil(Int, dur / interval - 1e-9), 0)
    route = Symbol(d["route"])
    doses = dosing_schedule(float(d["amt_mg_kg"]); start = 0.0, interval = interval, n = n, route = route)
    p = d["pk"]
    pkprior = (CL = float(p["CL"]), V = float(p["V"]), ka = float(p["ka"]), F = float(p["F"]),
        CL_logsd = float(p["CL_logsd"]), V_logsd = float(p["V_logsd"]), ka_logsd = float(p["ka_logsd"]),
        F_logitsd = float(p["F_logitsd"]),
        dose_status = String(d["dose_status"]), pk_status = String(p["status"]))
    return doses, pkprior
end
