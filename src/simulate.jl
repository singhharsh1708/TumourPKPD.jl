# Posterior-predictive simulation of tumour trajectories from PosteriorDraws.
# Needs (from the enclosing module): Random, Statistics, PosteriorDraws (hierarchical.jl),
# PKExposure/NoDrug/Simeoni/simulate_tumour (pd.jl).

function _draw_indices(rng, nall::Int, n::Int)
    n == nall && return collect(1:nall)
    n > nall && return rand(rng, 1:nall, n)          # more requested than stored: resample
    return sort!(randperm(rng, nall)[1:n])
end

# Per-row PDX parameters (λ0, k2, λ1) of stored draw `i`: a known model `j` uses its own stored λ0/k2 and the draw's
# pooled λ1; a new PDX draws fresh effects (independent log-normal λ0, k2 from `z1`, `z2`) with the draw's pooled λ1.
# This is the single place to change when the fits gain correlated (log λ0, log k2) effects or a per-PDX transition
# volume (then use the bivariate draw and λ1 = λ0·Vtr_i).
@inline function _row_params(pd, i, j, newpdx::Bool, z1, z2)
    if newpdx
        return (exp(pd.μ_λ0[i] + pd.ω_λ0[i] * z1), exp(pd.μ_k2[i] + pd.ω_k2[i] * z2), exp(pd.logλ1[i]))
    end
    return (pd.λ0[i, j], pd.k2[i, j], exp(pd.logλ1[i]))
end

# Core: returns (y, idx, σ, λ0, k2) for the selected draws. `mkexpo(i)` builds the
# exposure object for stored draw `i`.
function _simulate(pd, mkexpo, w0, w0mat::Matrix{Float64}, tgrid, ndraws, model, newpdx, obs_noise,
                   rng, rtol, atol, draws = nothing, baseline_models = nothing)
    tg = collect(Float64, tgrid)
    (!isempty(tg) && issorted(tg) && first(tg) >= 0) ||
        throw(ArgumentError("tgrid must be non-empty, sorted and >= 0"))
    if model === nothing && !newpdx
        throw(ArgumentError("give `model=<id>` for a known PDX or `newpdx=true` for a new one"))
    elseif model !== nothing && newpdx
        throw(ArgumentError("`model` and `newpdx=true` are mutually exclusive"))
    end
    nall = length(pd.σ)
    if draws === nothing
        n = Int(ndraws)
        idx = _draw_indices(rng, nall, n)
    else
        idx = collect(Int, draws)
        all(1 .<= idx .<= nall) || throw(ArgumentError("draws must index stored posterior draws"))
        n = length(idx)
    end
    nt = length(tg)

    # All random numbers are drawn serially so results do not depend on thread count.
    zλ = newpdx ? randn(rng, n) : Float64[]
    zk = newpdx ? randn(rng, n) : Float64[]
    pool = if newpdx
        baseline_models === nothing ? collect(1:length(pd.models)) :
            [(j = findfirst(==(String(m)), pd.models); j === nothing &&
              throw(ArgumentError("baseline model $(m) not in the fit")); j) for m in baseline_models]
    else
        Int[]
    end
    newpdx && isempty(pool) && throw(ArgumentError("empty baseline model pool"))
    jpick = newpdx ? rand(rng, pool, n) : Int[]            # drawn before the noise (see simulate_regimen)
    ν = nu_from_fit(pd)
    ε = obs_noise ? (isinf(ν) ? randn(rng, n, nt) : rand(rng, TDist(ν), n, nt)) : Matrix{Float64}(undef, 0, 0)

    j = 0
    if model !== nothing
        j = findfirst(==(String(model)), pd.models)
        j === nothing && throw(ArgumentError("model $(model) not in the fit; use newpdx=true"))
    end
    w0s = if w0 !== nothing
        w0 isa Real ? fill(float(w0), n) :
            (length(w0) == n ? collect(Float64, w0) : throw(DimensionMismatch("w0 must be a scalar or length $(n)")))
    elseif newpdx
        [w0mat[idx[r], jpick[r]] for r in 1:n]
    else
        [w0mat[idx[r], j] for r in 1:n]
    end

    λ0 = Vector{Float64}(undef, n)
    k2 = Vector{Float64}(undef, n)
    λ1 = Vector{Float64}(undef, n)
    for r in 1:n
        λ0[r], k2[r], λ1[r] = _row_params(pd, idx[r], j, newpdx, newpdx ? zλ[r] : 0.0, newpdx ? zk[r] : 0.0)
    end

    y = Matrix{Float64}(undef, n, nt)
    Threads.@threads for r in 1:n
        i = idx[r]
        θ = Simeoni(λ0[r], λ1[r], exp(pd.logk1[i]), k2[r], w0s[r])
        w, ok = simulate_tumour(θ, mkexpo(i), tg; reltol = rtol, abstol = atol)
        if ok
            if obs_noise
                σ = pd.σ[i]
                for c in 1:nt
                    # baseline (t = 0) is conditioned on, so it carries no noise
                    y[r, c] = tg[c] == 0 ? w[c] : max(w[c], 0.0) * exp(σ * ε[r, c])
                end
            else
                y[r, :] .= w
            end
        else
            y[r, :] .= NaN
        end
    end
    return (; y, idx, σ = pd.σ[idx], λ0, k2, w0 = w0s)
end

"""
    simulate_regimen(pd::PosteriorDraws, doses; model=nothing, w0=nothing, tgrid,
                     ndraws=200, newpdx=false, obs_noise=true, rng=Random.default_rng(),
                     rtol=1e-6, atol=1e-6, details=false)

Posterior-predictive treated-arm tumour volumes (mm^3), a `ndraws × length(tgrid)`
matrix. Each row uses one stored posterior draw: its PK (`pd.pk[i]`), `cref`, `logλ1`,
`logk1`, `σ`, and either the PDX model's own `λ0`/`k2` (`model="X-1234"`) or fresh
random effects `exp(μ + ω z)`, `z ~ N(0,1)` (`newpdx=true`, a population prediction for
an unseen PDX). `doses` are given in the same units as the fit (mg/kg, days); exposure is
`conc/cref` so `doses` may differ from the fitted reference regimen.

Baseline `w0` (mm^3) defaults to the draw's own baseline for a known model
(`pd.w0_treated_draws[i, j]`; constant across draws for fixed-baseline fits). For `newpdx=true`
each row uses the draw-matched baseline of a training model picked at random
(`pd.w0_treated_draws[i, j_r]`, `j_r` uniform over the fit's models, or over `baseline_models`
if given), i.e. the training set's baseline distribution. A user `w0` (scalar, or a vector with
one entry per row) overrides both. `obs_noise=true` multiplies by noise
`exp(σ ε)` at every `t > 0`, `ε ~ Student-t(ν)` with `ν = nu_from_fit(pd)` (the fit's `diagnostics["nu"]`, 5 if
absent, `Inf` = Normal), i.e. the fitted observation model (the baseline is conditioned on). Rows whose ODE solve fails
are filled with NaN. With `details=true` returns `(y, idx, σ, λ0, k2, w0)` where `idx` are the
stored-draw indices used (needed for log predictive density with `obs_noise=false`).

`draws` (a vector of stored-draw indices) overrides `ndraws`/random selection: pass the same
`draws` for every PDX model so that row `s` is the same population draw everywhere (needed for
joint quantities such as the responder fraction); fresh new-PDX random effects are still drawn
per call from `rng`. Using the same `rng` seed with `obs_noise=true` and `false` gives the same
latent paths (random effects are drawn before the noise).
"""
function simulate_regimen(pd, doses::AbstractVector{Dose}; model = nothing, w0 = nothing,
                          tgrid, ndraws = 200, newpdx::Bool = false, obs_noise::Bool = true,
                          rng::AbstractRNG = Random.default_rng(), rtol = 1e-6, atol = 1e-6,
                          details::Bool = false, draws = nothing, baseline_models = nothing)
    ds = collect(Dose, doses)
    cref = pd.cref
    mk = i -> PKExposure(pd.pk[i], ds, cref)
    r = _simulate(pd, mk, w0, pd.w0_treated_draws, tgrid, ndraws, model, newpdx, obs_noise, rng, rtol, atol,
                  draws, baseline_models)
    return details ? r : r.y
end

"""
    simulate_control(pd::PosteriorDraws; model=nothing, w0=nothing, tgrid, ndraws=200,
                     newpdx=false, obs_noise=true, rng, rtol, atol, details=false)

Untreated (NoDrug) predictive trajectories; same conventions as [`simulate_regimen`](@ref).
Baselines come from `pd.w0_control_draws` as described there. `draws`, `baseline_models` as in `simulate_regimen`.
"""
function simulate_control(pd; model = nothing, w0 = nothing, tgrid, ndraws = 200,
                          newpdx::Bool = false, obs_noise::Bool = true,
                          rng::AbstractRNG = Random.default_rng(), rtol = 1e-6, atol = 1e-6,
                          details::Bool = false, draws = nothing, baseline_models = nothing)
    r = _simulate(pd, _ -> NoDrug(), w0, pd.w0_control_draws, tgrid, ndraws, model, newpdx, obs_noise, rng,
                  rtol, atol, draws, baseline_models)
    return details ? r : r.y
end

"""
    predictive_bands(sims; levels=(0.05, 0.5, 0.95))

Column-wise quantiles (ignoring NaN rows) of a `ndraw × ntime` simulation matrix.
Returns a `ntime × length(levels)` matrix.
"""
function predictive_bands(sims::AbstractMatrix; levels = (0.05, 0.5, 0.95))
    out = Matrix{Float64}(undef, size(sims, 2), length(levels))
    for c in axes(sims, 2)
        col = filter(isfinite, @view sims[:, c])
        for (k, q) in enumerate(levels)
            out[c, k] = isempty(col) ? NaN : quantile(col, q)
        end
    end
    return out
end

"""
    fit_window_day(pd) -> Union{Float64,Nothing}

The fit's data window (`diagnostics["fit_window_day"]`, e.g. 42 for the primary and 21 for the late21
design); `nothing` if not recorded or 0 (= full follow-up).
"""
function fit_window_day(pd)
    v = _fit_meta(pd, "fit_window_day", nothing)
    (v === nothing || !(v > 0) || isinf(v)) && return nothing
    return float(v)
end

"""
    latent_baseline(pd, v0, n; rng) -> Vector

`n` draws of a test PDX's latent baseline given its observed baseline volume `v0`:
`v0 * exp(τ z)`, `z ~ N(0,1)`, `τ = baseline_sd_from_fit(pd)` (0.1 for the Student-t fits). Pass as `w0=` to
condition a prediction on the observed baseline with the fitted measurement error.
"""
latent_baseline(pd, v0::Real, n::Integer; rng::AbstractRNG = Random.default_rng()) =
    v0 .* exp.(baseline_sd_from_fit(pd) .* randn(rng, n))

# ---------------------------------------------------------------------------------------------
# What-if regimens
# ---------------------------------------------------------------------------------------------

"""
    scaled_regimen(doses; multiplier=1.0) -> Vector{Dose}

Same schedule with every dose amount multiplied by `multiplier`.
"""
scaled_regimen(doses::AbstractVector{Dose}; multiplier::Real = 1.0) =
    [Dose(d.time, d.amt * multiplier, d.route) for d in doses]

"""
    make_regimen(; amt_multiplier=1.0, interval_days, n_doses=nothing, duration_days=nothing,
                   days_on=nothing, days_off=0.0, start=0.0, route=:oral, ref_amt) -> Vector{Dose}

Build a what-if schedule: dose `ref_amt * amt_multiplier` (mg/kg per administration; `ref_amt` is the
reference regimen's dose, e.g. `pd.doses[1].amt`) every `interval_days` from `start` (`0.5` = BID with
the amount given per administration). Give exactly one of `n_doses` (number of administrations) or
`duration_days` (all administrations at times `< start + duration_days`).

Dosing holidays: with `days_on` and `days_off` the schedule repeats cycles of `days_on + days_off` days
in which administrations (every `interval_days`) occur only during the first `days_on` days of a cycle;
e.g. `days_on=5, days_off=2, interval_days=1` is 5-on-2-off (Mon-Fri dosing), counted from `start`.
`days_off` alone (without `days_on`) is an error.
"""
function make_regimen(; amt_multiplier::Real = 1.0, interval_days::Real, n_doses = nothing,
                      duration_days = nothing, days_on = nothing, days_off::Real = 0.0,
                      start::Real = 0.0, route::Symbol = :oral, ref_amt::Real)
    (n_doses === nothing) == (duration_days === nothing) &&
        throw(ArgumentError("give exactly one of n_doses or duration_days"))
    interval_days > 0 || throw(ArgumentError("interval_days must be positive"))
    amt_multiplier >= 0 || throw(ArgumentError("amt_multiplier must be non-negative"))
    days_off >= 0 || throw(ArgumentError("days_off must be non-negative"))
    amt = ref_amt * amt_multiplier
    if days_on === nothing
        days_off == 0 || throw(ArgumentError("days_off needs days_on"))
        n = n_doses === nothing ? max(ceil(Int, duration_days / interval_days - 1e-9), 0) : Int(n_doses)
        return dosing_schedule(amt; start, interval = interval_days, n, route)
    end
    days_on > 0 || throw(ArgumentError("days_on must be positive"))
    cycle = days_on + days_off
    per_cycle = max(ceil(Int, days_on / interval_days - 1e-9), 1)        # administrations in the on-period
    doses = Dose[]
    c = 0
    while true
        for j in 0:per_cycle-1
            t = start + c * cycle + j * interval_days
            if n_doses !== nothing
                length(doses) >= n_doses && return doses
            else
                t < start + duration_days - 1e-9 || return doses
            end
            push!(doses, Dose(t, amt, route))
        end
        c += 1
    end
end

"""
    is_extrapolation(ref, doses; horizon=nothing, observed_max_day=nothing) -> (flag, reasons)

Is the what-if regimen `doses` outside what the data identify? PDXE tested one dose level and
schedule per drug, so the sorted dose events `(time, amount, route)` of `doses` are compared with those
of the reference regimen `ref` (the fit's regimen, `pd.doses`) over the prediction window, i.e. all
events at times `< horizon` (all events if `horizon` is not given, so pass the horizon to compare
against a long reference schedule). Anything that differs is flagged with a reason: dose amount
(multiplier), route, interval, first-dose day, omitted interior doses (holidays, skipped doses), doses
at times the reference does not have or shifted doses, and dosing that stops before the reference
does (including missed final doses). Predicting beyond the longest observed follow-up
(`horizon > observed_max_day`) is also flagged. Returns the flag and the human-readable `reasons`
(empty if none). Also accepts a `PosteriorDraws` as `ref` (uses `pd.doses`, the saved
`max_observed_day` and the saved `fit_window_day`, when the fit records them; a horizon beyond the fit window is
flagged "beyond the fitted N-day window").
"""
function is_extrapolation(ref::AbstractVector{Dose}, doses::AbstractVector{Dose};
                          horizon = nothing, observed_max_day = nothing)
    reasons = String[]
    isempty(doses) && return (true, ["no doses given (untreated), which is the control arm, not a dosing scenario"])
    note = "PDXE tested one dose level and schedule per drug"
    inwin(d) = horizon === nothing || d.time < float(horizon) - 1e-9
    R = sort([d for d in ref if inwin(d)]; by = d -> (d.time, d.amt))
    U = sort([d for d in doses if inwin(d)]; by = d -> (d.time, d.amt))
    tol = 1e-9
    day(x) = round(x, digits = 3)
    days(ts) = join(day.(first(ts, 6)), ", ") * (length(ts) > 6 ? ", ..." : "")
    if isempty(U)
        return (true, ["no doses fall inside the prediction window (before day $(horizon)): $note"])
    end
    isempty(R) && return (true, ["the reference gives no doses inside the prediction window: $note"])
    # match events by time (within tol)
    ri = 1; ui = 1
    matched = Tuple{Dose,Dose}[]; miss = Dose[]; extra = Dose[]
    while ri <= length(R) && ui <= length(U)
        dt = U[ui].time - R[ri].time
        if abs(dt) <= tol
            push!(matched, (R[ri], U[ui])); ri += 1; ui += 1
        elseif dt > 0
            push!(miss, R[ri]); ri += 1
        else
            push!(extra, U[ui]); ui += 1
        end
    end
    append!(miss, R[ri:end]); append!(extra, U[ui:end])
    if !isempty(matched)
        ratios = [u.amt / r.amt for (r, u) in matched]
        if !all(isapprox.(ratios, 1.0; rtol = 1e-9))
            m = median(ratios)
            push!(reasons, "dose amount differs from the reference (multiplier $(round(m, sigdigits = 3)), $(round(median([u.amt for (_, u) in matched]), sigdigits = 4)) vs $(round(median([r.amt for (r, _) in matched]), sigdigits = 4)) mg/kg): $note, so exposure-response at other dose levels is extrapolated")
        end
        if any(r.route != u.route for (r, u) in matched)
            push!(reasons, "route differs from the reference ($(U[1].route) vs $(R[1].route)): $note")
        end
    elseif any(d.route != R[1].route for d in U)
        push!(reasons, "route differs from the reference ($(U[1].route) vs $(R[1].route)): $note")
    end
    ut = [d.time for d in U]; rt = [d.time for d in R]
    intervals_differ = false; iu = 0.0
    if length(ut) > 1 && length(rt) > 1
        iu = median(diff(ut)); iref = median(diff(rt))
        if !isapprox(iu, iref; rtol = 1e-9)
            intervals_differ = true
            push!(reasons, "dosing interval differs from the reference ($(round(iu, sigdigits = 4)) vs $(round(iref, sigdigits = 4)) days): $note")
        end
    end
    isapprox(ut[1], rt[1]; atol = tol) ||
        push!(reasons, "first dose at day $(day(ut[1])) instead of day $(day(rt[1])): $note")
    lastu = ut[end]
    # with a different interval the next scheduled dose after the last one is not a 'missed' dose
    tail = [d for d in miss if d.time > lastu + (intervals_differ ? iu : 0.0) + tol]
    interior = [d for d in miss if d.time < lastu - tol]
    isempty(tail) ||
        push!(reasons, "dosing stops after day $(day(lastu)): $(length(tail)) reference dose(s) at day(s) $(days([d.time for d in tail])) are not given and the treatment-free period was not observed")
    if !intervals_differ
        if !isempty(extra) && !isempty(interior)
            push!(reasons, "dose times are shifted or rescheduled relative to the reference: reference day(s) $(days([d.time for d in interior])) are not dosed and $(length(extra)) dose(s) are given at day(s) $(days([d.time for d in extra])) instead")
        elseif !isempty(interior)
            push!(reasons, "$(length(interior)) interior reference dose(s) are omitted (dosing holiday or skipped dose) at day(s) $(days([d.time for d in interior]))")
        elseif !isempty(extra)
            push!(reasons, "$(length(extra)) dose(s) are given at day(s) $(days([d.time for d in extra])) where the reference has none")
        end
    end
    if horizon !== nothing && observed_max_day !== nothing && horizon > observed_max_day + 1e-9
        push!(reasons, "prediction horizon $(horizon) d exceeds the longest observed follow-up ($(observed_max_day) d)")
    end
    return (!isempty(reasons), reasons)
end

function is_extrapolation(pd, doses::AbstractVector{Dose}; horizon = nothing, observed_max_day = nothing)
    omd = observed_max_day === nothing ? _fit_meta(pd, "max_observed_day", nothing) : observed_max_day
    flag, reasons = is_extrapolation(pd.doses, doses; horizon, observed_max_day = omd)
    fw = fit_window_day(pd)
    if horizon !== nothing && fw !== nothing && horizon > fw + 1e-9
        push!(reasons, "prediction horizon $(horizon) d is beyond the fitted $(round(Int, fw))-day window (the model only saw data up to day $(round(Int, fw)))")
        flag = true
    end
    if horizon !== nothing && omd === nothing
        push!(reasons, "longest observed follow-up is not recorded in the fit (diagnostics[\"max_observed_day\"]), so the horizon cannot be checked")
        flag = true
    end
    return (flag, reasons)
end

# ---------------------------------------------------------------------------------------------
# Virtual PDX trial
# ---------------------------------------------------------------------------------------------

"""
    pdxe_like_grid(horizon) -> Vector{Float64}

Twice-weekly PDXE-like measurement days 0, 4, 7, 10, 14, ... (every 3.5 d, rounded to whole days),
keeping only visits before `horizon` and ending exactly at `horizon` (also for fractional horizons).
"""
function pdxe_like_grid(horizon::Real)
    h = float(horizon)
    g = filter(x -> x < h - 1e-9, unique(round.(0.0:3.5:h)))
    push!(g, h)
    return unique(g)
end

function _model_types(pd)
    mt = _fit_meta(pd, "tumour_types", nothing)
    mt === nothing && return nothing
    mt isa AbstractDict && return Dict(String(k) => String(v) for (k, v) in mt)
    length(mt) == length(pd.models) || return nothing
    return Dict(String(m) => String(t) for (m, t) in zip(pd.models, mt))
end

mean_finite_(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))

_cri(x, level) = (v = filter(isfinite, x); isempty(v) ? (median = NaN, lo = NaN, hi = NaN) :
    (median = median(v), lo = quantile(v, (1 - level) / 2), hi = quantile(v, 1 - (1 - level) / 2)))

"""
    simulate_trial(pd, doses; n_pdx=200, ndraws=200, population=false, n_pdx_population=1000,
                   tumour_type=nothing, horizon=nothing, tgrid=nothing, quick=false, level=0.9,
                   censor=true, rng, draws=nothing, classifier=pdxe_response_paper, rtol=1e-5, atol=1e-5)

Virtual PDX trial of the regimen `doses`. For each of `ndraws` posterior draws (a random subset of
the stored draws) it simulates `n_pdx` new PDX tumours (fresh random effects from that draw's
population, a draw-matched baseline of a randomly chosen training model, lognormal observation
noise), classifies every tumour with `classifier` (default [`pdxe_response_paper`](@ref)) on a
twice-weekly grid up to `horizon` days, and returns per-draw responder rate (mCR+mPR) and category
fractions. `ndraws` is capped at the number of stored draws.

- `horizon` defaults to `min(median_followup_day or 28, fit_window_day)`; an explicit `horizon` beyond
  the fit window (`diagnostics["fit_window_day"]`, 42 d for the primary fits) is run but flagged in
  `extrapolation` as "beyond the fitted N-day window". Responder
  rates depend on follow-up length, so compare with PDXE at a matching horizon. `tgrid` overrides
  the grid (must start at 0).
- `tumour_type` restricts the baseline pool to training PDX of that type; this needs the fit to
  carry `diagnostics["tumour_types"]` (Dict model => type, or a vector aligned with `pd.models`).
  The population `μ, ω` are not type-specific (the model has no tissue effect), so this changes only the
  baseline distribution.
- `censor=true` records simulated volumes below the fit's detection limit as 0 (undetectable),
  as in PDXE, before classification.
- `quick=true` caps `ndraws` at 50, `n_pdx` at 100 and `n_pdx_population` at 400 for interactive use.
- **What the interval means.** By default (`population=false`) each draw runs one virtual trial of
  `n_pdx` PDX, and `summary` gives the median and `level` interval of the per-draw trial fractions: a
  *posterior predictive interval for a trial of `n_pdx` PDX models*, which combines posterior
  uncertainty about the population with finite-trial sampling variation (about `sqrt(p(1-p)/n_pdx)`
  per draw, which stays even with zero posterior uncertainty). It is not a credible interval for the
  population response probability. With `population=true` each draw simulates `n_pdx_population`
  (default 1000) PDX instead, so the per-draw fraction estimates that draw's *population* response
  probability with Monte Carlo standard error `mc_se` (reported, about 1.6 points at p = 0.4); `summary`
  is then the posterior credible interval of the population responder rate and category fractions
  (still including that small simulation error, which `mc_se` quantifies). Cost scales with
  `ndraws × n_pdx_population`: use fewer draws (e.g. 50) for the population mode.
- The ODE solves of all `ndraws × n_pdx` tumours are threaded; random numbers are drawn serially, so
  results do not depend on the thread count.

Returns `(responder_rate, fractions, categories, summary, extrapolation, tgrid, idx, n_na, mc_se,
interval_kind, population, ...)`; `interval_kind` is a sentence describing what `summary`'s interval is
(show it next to the numbers). Details:
`responder_rate` (`ndraws`) and `fractions` (`ndraws × 4`, columns `RESPONSE_CATEGORIES`) are over
non-NA tumours of each draw (failed solves); `categories` is `n_pdx × ndraws`; `summary` has
`(median, lo, hi)` of the responder rate and of each category fraction; `extrapolation` is
`is_extrapolation(pd, doses; horizon)`, which a consumer must show with the result.
"""
function simulate_trial(pd, doses::AbstractVector{Dose}; n_pdx::Integer = 200, ndraws::Integer = 200,
                        population::Bool = false, n_pdx_population::Integer = 1000,
                        tumour_type = nothing, horizon = nothing, tgrid = nothing, quick::Bool = false,
                        level::Real = 0.9, censor::Bool = true, rng::AbstractRNG = Random.default_rng(),
                        draws = nothing, classifier = pdxe_response_paper, rtol = 1e-5, atol = 1e-5)
    if quick
        ndraws = min(ndraws, 50); n_pdx = min(n_pdx, 100); n_pdx_population = min(n_pdx_population, 400)
    end
    population && (n_pdx = n_pdx_population)
    fw = fit_window_day(pd)
    hz = if horizon === nothing
        h0 = float(_fit_meta(pd, "median_followup_day", 28.0))
        fw === nothing ? h0 : min(h0, fw)
    else
        float(horizon)          # an explicit horizon beyond the fit window is allowed but flagged
    end
    tg = tgrid === nothing ? pdxe_like_grid(hz) : collect(Float64, tgrid)
    (first(tg) == 0 && issorted(tg)) || throw(ArgumentError("tgrid must be sorted and start at day 0"))
    hz = last(tg)
    bm = nothing
    if tumour_type !== nothing
        mt = _model_types(pd)
        mt === nothing && throw(ArgumentError("this fit does not record tumour types (diagnostics[\"tumour_types\"])"))
        bm = [m for m in pd.models if get(mt, m, nothing) == String(tumour_type)]
        isempty(bm) && throw(ArgumentError("no training PDX of type $(tumour_type); have $(sort(unique(values(mt))))"))
    end
    # draws are shared by every tumour of a draw, so build each draw's exposure once; doses after
    # the horizon cannot affect the trajectory, so drop them
    ds = [d for d in doses if d.time <= hz]
    nall = length(pd.σ)
    idx = draws === nothing ? _draw_indices(rng, nall, min(Int(ndraws), nall)) : collect(Int, draws)
    nd = length(idx)
    expo = Dict(i => PKExposure(pd.pk[i], ds, pd.cref) for i in unique(idx))
    rows = repeat(idx, inner = n_pdx)                       # row r*(n_pdx) block = draw r
    r = _simulate(pd, i -> expo[i], nothing, pd.w0_treated_draws, tg, length(rows), nothing, true, true,
                  rng, rtol, atol, rows, bm)
    y = r.y
    if censor
        lod = detection_limit_from_fit(pd)
        @inbounds for k in eachindex(y)
            y[k] < lod && (y[k] = 0.0)
        end
        y[:, 1] .= r.w0          # baseline is not censored
    end
    cats = Vector{String}(undef, size(y, 1))
    Threads.@threads for k in axes(y, 1)
        row = @view y[k, :]
        cats[k] = all(isfinite, row) ? classifier(tg, row).category : "NA"
    end
    C = reshape(cats, n_pdx, nd)
    fr = zeros(nd, 4); rate = fill(NaN, nd); mcse = fill(NaN, nd); nna = 0
    for d in 1:nd
        col = @view C[:, d]
        ok = count(!=("NA"), col)
        nna += n_pdx - ok
        ok == 0 && (fr[d, :] .= NaN; continue)
        for (c, name) in enumerate(RESPONSE_CATEGORIES)
            fr[d, c] = count(==(name), col) / ok
        end
        rate[d] = (fr[d, 1] + fr[d, 2])
        mcse[d] = sqrt(rate[d] * (1 - rate[d]) / ok)
    end
    summ = (responder_rate = _cri(rate, level),
            mCR = _cri(fr[:, 1], level), mPR = _cri(fr[:, 2], level),
            mSD = _cri(fr[:, 3], level), mPD = _cri(fr[:, 4], level), level = level)
    ext = is_extrapolation(pd, doses; horizon = hz)
    kind = population ?
        "posterior credible interval for the population response probability (per-draw estimate from $(n_pdx) PDX, Monte Carlo SE about $(round(100 * mean_finite_(mcse), digits = 1)) points)" :
        "posterior predictive interval for a trial of $(n_pdx) PDX models (posterior uncertainty plus finite-trial sampling)"
    return (responder_rate = rate, fractions = fr, categories = C, summary = summ, extrapolation = ext,
            tgrid = tg, idx = idx, n_na = nna, n_pdx = Int(n_pdx), ndraws = nd, horizon = hz,
            tumour_type = tumour_type, population = population, mc_se = mcse, interval_kind = kind)
end

"""
    compare_trials(pd, regimens::AbstractDict; seed=1, baseline=nothing, kwargs...)

Run [`simulate_trial`](@ref) for each named regimen (`name => doses`) with common random numbers
(same stored draws, same new-PDX random effects, baselines and noise for every regimen), so
differences between regimens are paired and low-noise. Returns `(results, table, contrasts, contrast_kind)`:
`results` is `name => trial`, `table` a vector of rows `(name, responder_median, lo, hi, mCR, mPR, mSD, mPD
medians, extrapolation, reasons)` in sorted-name order, `contrasts` is `name => (median, lo, hi)` of the
per-draw responder-rate difference to `baseline` (a key of `regimens`), if given. `table` intervals and
`contrasts` carry the meaning of the trial mode: with the default `population=false` they are posterior
predictive intervals for trials of `n_pdx` PDX (paired contrasts keep the residual finite-trial variation
even with common random numbers); pass `population=true` (forwarded to `simulate_trial`) for credible
intervals of the population response-probability difference. `contrast_kind` states which.
"""
function compare_trials(pd, regimens::AbstractDict; seed::Integer = 1, baseline = nothing, kwargs...)
    names = sort(collect(keys(regimens)); by = string)
    res = Dict{Any,Any}()
    base_draws = nothing
    for nm in names
        tr = simulate_trial(pd, regimens[nm]; rng = MersenneTwister(seed), kwargs...)
        res[nm] = tr
    end
    table = [(name = nm, responder_median = res[nm].summary.responder_rate.median,
              responder_lo = res[nm].summary.responder_rate.lo, responder_hi = res[nm].summary.responder_rate.hi,
              mCR = res[nm].summary.mCR.median, mPR = res[nm].summary.mPR.median,
              mSD = res[nm].summary.mSD.median, mPD = res[nm].summary.mPD.median,
              extrapolation = res[nm].extrapolation[1], reasons = res[nm].extrapolation[2]) for nm in names]
    contrasts = Dict{Any,Any}()
    if baseline !== nothing
        haskey(res, baseline) || throw(ArgumentError("baseline $(baseline) not in regimens"))
        for nm in names
            nm == baseline && continue
            kwlevel = get(kwargs, :level, 0.9)
            contrasts[nm] = _cri(res[nm].responder_rate .- res[baseline].responder_rate, kwlevel)
        end
    end
    kind = isempty(res) ? "" : first(values(res)).population ?
        "posterior credible interval of the difference in population response probability (paired draws)" :
        "posterior predictive interval of the difference in responder fraction between trials of $(first(values(res)).n_pdx) PDX models (paired draws and random numbers)"
    return (results = res, table = table, contrasts = contrasts, contrast_kind = kind)
end
