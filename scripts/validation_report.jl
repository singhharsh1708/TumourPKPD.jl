# Validation report: held-out PDX predictive checks, late-window forecast, response categories,
# convergence tables, PSIS-LOO.
#
#   julia --project=TumourPKPD.jl -t 3 scripts/validation_report.jl <drug> [--marginal-loo] [--no-loo] [--ndraws=200]
#                                      [--resp-draws=1000] [--out=DIR]
#   julia --project=TumourPKPD.jl -t 3 scripts/validation_report.jl <drug> --fits-dir=DIR --tag=v2 [--suffix=_v2]
#                                      [--suffix-v1=] [--fits-dir-v1=fits] [--out=DIR]
#        model-comparison mode: verifies and scores the fits DIR/<drug>_primary<suffix>.jld2 and DIR/<drug>_late21<suffix>.jld2
#        (v2; e.g. --suffix=_v2 or --suffix=_v2_waning) against the v1 fits (fits/<drug>_primary.jld2 ...), writes
#        docs/validation_<drug>_<tag>.md and docs/compare_<drug>.md (v1 vs v2 key metrics). Also works without --fits-dir for a
#        single report: `<drug> --suffix=_v2 --tag=v2`.
#   julia --project=TumourPKPD.jl -t 2 scripts/validation_report.jl --selftest
#
# Reads (under fits/, each a `save_fit` JLD2 holding a PosteriorDraws):
#   <drug>_primary.jld2  frozen stratified 80% of PDX, days <= 42  -> held-out PDX prediction (scored on days <= 42),
#                                                                  PSIS-LOO on its training PDX, convergence
#   <drug>_late21.jld2   same 80%, days <= 21                      -> forecast of days 22-42 of the training PDX
# Held-out PDX = frozen_pdx_split(; frac, seed) (union over drugs, stratified by tumour type) intersected with
# the drug's models; frac/seed come from the fit's diagnostics["split"]. The checks below error out on any
# mismatch between the fit's saved metadata and these designs. Missing fits are reported as "not available".
# Writes docs/validation_<drug>.md and docs/fig_<drug>_*.png.
#
# --selftest builds synthetic PDX units (Student-t noise) and an "oracle" posterior (true parameters, no parameter
# uncertainty) to exercise the whole pipeline; not a scientific result.

using Random, Statistics, Printf, Dates
using TumourPKPD
const M = TumourPKPD
const ROOT = normpath(joinpath(@__DIR__, ".."))
const PRIMARY_WINDOW = 42.0         # primary fits saw days <= 42 only; held-out PDX are scored on days <= 42
const LATE_WINDOW = 21.0            # late21 fits saw days <= 21; forecast days 22-42

function ensure_loaded()
    isdefined(M, :PosteriorDraws) || Base.include(M, joinpath(ROOT, "src", "hierarchical.jl"))
    if !isdefined(M, :simulate_regimen)
        @eval M import ParetoSmooth
        Base.include(M, joinpath(ROOT, "src", "simulate.jl"))
        Base.include(M, joinpath(ROOT, "src", "validate.jl"))
    end
end

# ---------------------------------------------------------------------------------------
# Fit metadata checks (errors on mismatch)
# ---------------------------------------------------------------------------------------
#
# Metadata read from PosteriorDraws.diagnostics (or diagnostics["settings"]):
#   "train_models" :: Vector{String}   PDX ids the fit was trained on (must equal pd.models)
#   "split"        :: Dict             "model_frac", "seed", "holdout_models", "late_window_day"
#   "cutoffs"      :: Dict{String,Real} late21 fit: last observed day <= 21 per training curve,
#                                      keys "<model>|treated" / "<model>|control"
#   "fit_window_day", "nu", "baseline_sd", "lod", "nchains", ...
# Absent keys are listed in the document as "NOT verified"; present keys that disagree are errors.

meta(pd, k, default = nothing) = M._fit_meta(pd, k, default)

function check_drug(pd, drug)
    pd.drug == drug || error("fit is for drug $(repr(pd.drug)), expected $(repr(drug))")
end

function check_window(pd, expected, label)
    v = meta(pd, "fit_window_day")
    v === nothing && return ["$(label): fit_window_day"]
    isapprox(v, expected) || error("$(label) fit has fit_window_day=$(v), expected $(expected)")
    return String[]
end

"Frozen split: held-out ids = frozen_pdx_split ∩ the drug's models; training = the rest of the drug's models."
function expected_split(drug_models; frac, seed, test_all)
    test = sort(intersect(test_all, drug_models))
    return sort(setdiff(drug_models, test)), test
end

function verify_common(pd, drug, train_ids, test_ids; model_frac, seed, label)
    check_drug(pd, drug)
    leak = intersect(test_ids, pd.models)
    isempty(leak) || error("$(label): held-out PDX models are in the fit's training models: $(first(leak, 5))")
    extra = setdiff(pd.models, train_ids)
    isempty(extra) || error("$(label): fit contains models outside the frozen training set (frac=$(model_frac), seed=$(seed)): $(first(extra, 5))")
    unverified = String[]
    tm = meta(pd, "train_models")
    if tm === nothing
        push!(unverified, "$(label): train_models")
    else
        sort(String.(tm)) == sort(pd.models) || error("$(label): saved train_models differ from the fit's models")
        isempty(intersect(test_ids, String.(tm))) || error("$(label): held-out PDX models are in saved train_models")
    end
    sp = meta(pd, "split")
    if sp isa AbstractDict
        haskey(sp, "model_frac") && !isapprox(sp["model_frac"], model_frac) &&
            error("$(label): fit used model_frac=$(sp["model_frac"]), report uses $(model_frac)")
        haskey(sp, "seed") && sp["seed"] != seed && error("$(label): fit used split seed=$(sp["seed"]), report uses $(seed)")
        if haskey(sp, "holdout_models")
            sort(String.(sp["holdout_models"])) == sort(test_ids) ||
                error("$(label): saved split[\"holdout_models\"] differ from the frozen split intersected with the drug's models")
        else
            push!(unverified, "$(label): split[holdout_models]")
        end
        (haskey(sp, "model_frac") && haskey(sp, "seed")) || push!(unverified, "$(label): split(model_frac, seed)")
    else
        push!(unverified, "$(label): split")
    end
    return unverified
end

verify_primary_fit(pd, drug, train_ids, test_ids; model_frac, seed) =
    vcat(verify_common(pd, drug, train_ids, test_ids; model_frac, seed, label = "primary"),
         check_window(pd, PRIMARY_WINDOW, "primary"))

"Last observed day <= `window` of a curve (the late21 cutoff)."
last_obs_le(c, window) = (ts = [t for t in c.t if t <= window]; isempty(ts) ? NaN : maximum(ts))

function verify_late21_fit(pd, drug, units, train_ids, test_ids; model_frac, seed)
    unv = verify_common(pd, drug, train_ids, test_ids; model_frac, seed, label = "late21")
    append!(unv, check_window(pd, LATE_WINDOW, "late21"))
    sp = meta(pd, "split")
    if sp isa AbstractDict && haskey(sp, "late_window_day")
        isapprox(sp["late_window_day"], LATE_WINDOW) || error("late21: split late_window_day=$(sp["late_window_day"]), expected $(LATE_WINDOW)")
    else
        push!(unv, "late21: split[late_window_day]")
    end
    co = meta(pd, "cutoffs")
    if co isa AbstractDict
        for u in units
            u.model in pd.models || continue
            for (arm, c) in ((:treated, u.treated), (:control, u.control))
                exp_t = last_obs_le(c, LATE_WINDOW)
                got = get(co, "$(u.model)|$(arm)", nothing)
                got === nothing && error("late21: no saved cutoff for $(u.model)|$(arm)")
                abs(got - exp_t) < 1e-9 || error("late21: cutoff mismatch for $(u.model)|$(arm): fit $(got), expected last observed day <= $(LATE_WINDOW) = $(exp_t)")
            end
        end
    else
        push!(unv, "late21: cutoffs")
    end
    return unv
end

# ---------------------------------------------------------------------------------------
# Prediction records
# ---------------------------------------------------------------------------------------

"One predicted curve: observations, noisy draws, latent (noise-free) draws, per-draw σ, observed baseline v0."
struct CurveRecord
    model::String
    tumour_type::String
    arm::Symbol
    t::Vector{Float64}
    y::Vector{Float64}
    sims::Matrix{Float64}
    latent::Matrix{Float64}
    σ::Vector{Float64}
    v0::Float64
end

model_seed(seed, model) = Int(hash((seed, String(model))) % 2^31)

# Noisy and latent paths use the same rng seed (the new-PDX random effects are drawn before the observation noise) and
# the same baselines, so both share random effects; the seed depends on the PDX only, so control and treated share
# random effects. `draws` are common.
function predict_curve(pd, model, tumour_type, arm::Symbol, t, y, v0; newpdx, draws, seed)
    simf = arm === :control ? (; kw...) -> M.simulate_control(pd; kw...) :
                              (; kw...) -> M.simulate_regimen(pd, pd.doses; kw...)
    # Held-out (new) PDX: condition on the observed baseline with the fitted measurement error. Known PDX (late21
    # forecast): keep the stored draw-matched baselines, which are linked to that draw's growth/potency parameters
    # and carry what the fit learned from days <= 21.
    kw = if newpdx
        w0 = M.latent_baseline(pd, v0, length(draws); rng = MersenneTwister(model_seed(seed, "$(model)|$(arm)|w0")))
        (; tgrid = t, w0, draws, details = true, newpdx = true)
    else
        (; tgrid = t, draws, details = true, model = model)
    end
    noisy = simf(; kw..., obs_noise = true, rng = MersenneTwister(model_seed(seed, model)))
    lat = simf(; kw..., obs_noise = false, rng = MersenneTwister(model_seed(seed, model)))
    return CurveRecord(model, tumour_type, arm, collect(Float64, t), collect(Float64, y),
                       noisy.y, lat.y, lat.σ, float(v0))
end

function baseline_volume(c, fallback)
    i = findfirst(==(0.0), c.t)
    (i !== nothing && c.v[i] > 0) ? c.v[i] : fallback
end

common_draws(pd, ndraws, seed) = M._draw_indices(MersenneTwister(seed), length(pd.σ), ndraws)

"Predictions for the `units` over times chosen by `select(curve) -> mask of times to score`."
function predict_units(pd, units; newpdx, select, draws, seed = 1, arms = (:treated, :control))
    recs = CurveRecord[]
    med_t = median(pd.w0_treated); med_c = median(pd.w0_control)
    for u in units
        for arm in arms
            c = arm === :treated ? u.treated : u.control
            mask = select(c)
            any(mask) || continue
            v0 = baseline_volume(c, arm === :treated ? med_t : med_c)
            push!(recs, predict_curve(pd, u.model, u.tumour_type, arm, c.t[mask], c.v[mask], v0; newpdx, draws, seed))
        end
    end
    return recs
end

# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------

mean_finite(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))

function curve_metrics(recs::Vector{CurveRecord}; level = 0.9, lod, nu, nboot = 500, seed = 1)
    isempty(recs) && return nothing
    pos(r) = r.y .> 0
    y = reduce(vcat, [r.y[pos(r)] for r in recs])
    S = reduce(hcat, [r.sims[:, pos(r)] for r in recs])
    pdx = reduce(vcat, [fill(r.model, count(pos(r))) for r in recs])      # tumour = PDX identity
    cg = M.coverage_by_group(y, S, pdx; level)
    fail = M.ensemble_failures(S)
    wlog = M.interval_width(S; level, transform = log)
    lpd = reduce(vcat, [M.log_predictive_density(r.y, r.latent, r.σ; limit = lod, nu) for r in recs])
    crps = M.crps(y, S)
    ncens = sum(count(<=(0), r.y) for r in recs)
    cov_boot = M.cluster_bootstrap(idx -> M.coverage(y[idx], S[:, idx]; level), pdx;
                                   nboot, rng = MersenneTwister(seed))
    return (n_points = length(y), n_censored = ncens, n_curves = length(recs),
            n_pdx = length(unique(r.model for r in recs if any(>(0), r.y))), level, lod, nu,
            coverage_pooled = cg.pooled, coverage_tumour_avg = cg.tumour_avg,
            coverage_pdx_ci = (quantile(cov_boot, 0.025), quantile(cov_boot, 0.975)),
            width_log_mean = mean_finite(wlog), n_width_nan = count(isnan, wlog),
            lpd_mean_finite = mean_finite(lpd), lpd_mean_all = (v = filter(!isnan, lpd); isempty(v) ? NaN : mean(v)),
            lpd_n = count(!isnan, lpd), n_lpd_neginf = count(==(-Inf), lpd), n_lpd_nan = count(isnan, lpd),
            crps_mean = mean_finite(crps), n_crps_nan = count(isnan, crps),
            n_failed_draw_rows = fail.n_failed_rows, n_empty_ensembles = fail.n_empty_columns,
            calib = [(l, M.coverage(y, S; level = l)) for l in 0.1:0.1:0.9])
end

"""
Coverage of the central `level` interval for observations `ys` with ragged ensembles `cols` (each observation's own
finite draws). An empty ensemble counts as a miss; nothing else is discarded.
"""
function coverage_ragged(ys, cols; level = 0.9)
    isempty(ys) && return NaN
    α = (1 - level) / 2
    hit = 0
    for (y, c) in zip(ys, cols)
        isempty(c) && continue
        hit += quantile(c, α) <= y <= quantile(c, 1 - α)
    end
    return hit / length(ys)
end

"""
Tail calibration: fraction of observations below the predictive `α` quantile and above the `1-α` quantile (each from
its own finite ensemble; nominal α each; empty ensembles are skipped here and counted by the coverage). Returns
`(below, above, n)`.
"""
function ragged_tails(ys, cols; α = 0.05)
    lo = 0; hi = 0; n = 0
    for (y, c) in zip(ys, cols)
        isempty(c) && continue
        n += 1
        lo += y < quantile(c, α); hi += y > quantile(c, 1 - α)
    end
    return n == 0 ? (below = NaN, above = NaN, n = 0) : (below = lo / n, above = hi / n, n = n)
end

"""
Day bins merged so that each bin aggregates at least `min_pdx` distinct PDX models for the arm (greedy from the
left; a remainder joins the last bin). Returns the merged edges or `nothing` if even all bins together have fewer
than `min_pdx` PDX (the arm is then suppressed). `valid(r, k)` selects which observations count as contributors
(default all; the residual check counts only positive observations with a non-empty positive ensemble).
"""
function merged_edges(recs::Vector{CurveRecord}, arm, edges; min_pdx, valid = (r, k) -> true)
    nb = length(edges) - 1
    sets = [Set{String}() for _ in 1:nb]
    for r in recs
        r.arm === arm || continue
        for k in eachindex(r.t)
            valid(r, k) || continue
            b = findfirst(i -> edges[i] < r.t[k] <= edges[i+1], 1:nb)
            b === nothing || push!(sets[b], r.model)
        end
    end
    groups = Tuple{Int,Int}[]; start = 1; cur = Set{String}()
    for b in 1:nb
        union!(cur, sets[b])
        if length(cur) >= min_pdx
            push!(groups, (start, b)); start = b + 1; cur = Set{String}()
        end
    end
    isempty(groups) && return nothing
    start <= nb && (groups[end] = (groups[end][1], nb))
    return vcat([edges[g[1]] for g in groups], edges[end])
end

"""
Binned predictive check of log(V/V0) for held-out curves, per arm. Both observed and simulated volumes are
censored at the LOD (an undetectable observation, or a simulated volume below the LOD, counts as `lod`), V0 is the
curve's observed baseline. Per day bin: observed vs pooled predictive 5/50/95% quantiles of log(V/V0), the
fraction below the LOD (observed vs predicted), the 90% coverage of the observed positive volumes (each from its own
finite ensemble), the tail calibration (fraction of observations below the predictive 5% / above the 95% quantile,
nominal 5% each) and `n_pdx`, the number of distinct PDX models in the bin. With `min_pdx` the bins are merged
until each has at least that many PDX models and arms that never reach it are suppressed (public outputs).
"""
function binned_check(recs::Vector{CurveRecord}; lod, edges, min_pdx = nothing)
    out = NamedTuple[]
    for arm in (:treated, :control)
        ed = min_pdx === nothing ? edges : merged_edges(recs, arm, edges; min_pdx)
        ed === nothing && continue
        for b in 1:length(ed)-1
            lo, hi = ed[b], ed[b+1]
            obs = Float64[]; obsbelow = 0; n = 0; pred = Float64[]; predbelow = 0; npred = 0
            yc = Float64[]; cols = Vector{Vector{Float64}}(); pdx = Set{String}(); pdxcov = Set{String}()
            for r in recs
                r.arm === arm || continue
                for k in eachindex(r.t)
                    (lo < r.t[k] <= hi) || continue
                    n += 1; push!(pdx, r.model)
                    below = r.y[k] < lod
                    obsbelow += below
                    push!(obs, log(max(r.y[k], lod) / r.v0))
                    col = filter(isfinite, @view r.sims[:, k])
                    predbelow += count(<(lod), col); npred += length(col)
                    append!(pred, log.(max.(col, lod) ./ r.v0))
                    if r.y[k] > 0
                        push!(yc, r.y[k]); push!(cols, col); push!(pdxcov, r.model)
                    end
                end
            end
            n == 0 && continue
            q(x) = isempty(x) ? (NaN, NaN, NaN) : (quantile(x, 0.05), quantile(x, 0.5), quantile(x, 0.95))
            # coverage and tails use the positive observations only: suppress them when fewer than min_pdx PDX contribute
            enough = min_pdx === nothing || length(pdxcov) >= min_pdx
            tl = enough ? ragged_tails(yc, cols) : (below = NaN, above = NaN, n = 0)
            push!(out, (arm = arm, lo = lo, hi = hi, n = n, n_pdx = length(pdx), n_pdx_cov = length(pdxcov), obs_q = q(obs), pred_q = q(pred),
                        obs_below = obsbelow / n, pred_below = npred == 0 ? NaN : predbelow / npred,
                        coverage90 = enough ? coverage_ragged(yc, cols; level = 0.9) : NaN, tail_below5 = tl.below, tail_above95 = tl.above))
        end
    end
    return out
end

"""
Within-PDX residual check. For every observed positive volume the signed log residual is
`log y - median_s log(simulated y)` (the median of its own predictive ensemble). Per arm and (merged) day bin the residuals
are first reduced to one median per PDX, then summarised across PDX (each PDX weighted equally, so a changing set of
PDX with follow-up cannot masquerade as a trend). `median` / `ci` (PDX bootstrap 90%) use all PDX in the bin;
`panel_median` uses only PDX observed in every bin of that arm (a balanced panel, so any change between bins is within-PDX);
it is `NaN` unless the panel has at least `min_pdx` PDX. Positive values mean the observed volumes are larger than predicted.
"""
function residual_check(recs::Vector{CurveRecord}; edges, min_pdx = nothing, nboot = 500, seed = 1)
    out = NamedTuple[]
    rng = MersenneTwister(seed)
    for arm in (:treated, :control)
        valid = (r, k) -> r.y[k] > 0 && any(x -> isfinite(x) && x > 0, @view r.sims[:, k])
        ed = min_pdx === nothing ? edges : merged_edges(recs, arm, edges; min_pdx, valid)
        ed === nothing && continue
        nb = length(ed) - 1
        per = [Dict{String,Vector{Float64}}() for _ in 1:nb]
        for r in recs
            r.arm === arm || continue
            for k in eachindex(r.t)
                r.y[k] > 0 || continue
                b = findfirst(i -> ed[i] < r.t[k] <= ed[i+1], 1:nb)
                b === nothing && continue
                col = filter(x -> isfinite(x) && x > 0, @view r.sims[:, k])
                isempty(col) && continue
                push!(get!(per[b], r.model, Float64[]), log(r.y[k]) - median(log.(col)))
            end
        end
        pm = [Dict(m => median(v) for (m, v) in per[b]) for b in 1:nb]
        panel = isempty(pm) ? String[] : collect(intersect(Set.(keys.(pm))...))
        for b in 1:nb
            vals = collect(values(pm[b])); isempty(vals) && continue
            boot = [median(rand(rng, vals, length(vals))) for _ in 1:nboot]
            pv = [pm[b][m] for m in panel]
            enough = length(panel) >= (min_pdx === nothing ? 1 : min_pdx)
            push!(out, (arm = arm, lo = ed[b], hi = ed[b+1], n_pdx = length(vals), median = median(vals),
                        ci = (quantile(boot, 0.05), quantile(boot, 0.95)),
                        n_panel = length(panel), panel_median = enough ? median(pv) : NaN))
        end
    end
    return out
end

"Pooled tail calibration over all points of the records of one arm: `(below5, above95, n)`."
function pooled_tails(recs::Vector{CurveRecord}, arm)
    ys = Float64[]; cols = Vector{Vector{Float64}}()
    for r in recs
        r.arm === arm || continue
        for k in eachindex(r.t)
            r.y[k] > 0 || continue
            push!(ys, r.y[k]); push!(cols, filter(isfinite, @view r.sims[:, k]))
        end
    end
    t = ragged_tails(ys, cols)
    return (below5 = t.below, above95 = t.above, n = t.n,
            n_pdx = length(unique(r.model for r in recs if r.arm === arm && any(>(0), r.y))))
end

"Observed curve cut to `t <= window`."
cut_curve(c, window) = (m = c.t .<= window; (t = c.t[m], v = c.v[m]))

"""
Predicted response categories of one unit's treated arm, one per draw, on the curve cut to days <= window. Same
conventions for every mode: simulated volumes below the LOD are recorded as 0 and the day-0 volume is set to the observed
`v0`, so the classification divides by the same quantity as the observed category. `newpdx=true`: held-out PDX, fresh
random effects and the observed baseline with the fitted measurement error; `newpdx=false`: a training PDX with its own
(known) random effects and stored baselines. `noise=true` is the *recorded* response (observation noise as in the data);
`noise=false` is the *latent* response of the noise-free trajectory. Paired: both use the same random effects.
"""
function response_cats(pd, u, t, v0; draws, newpdx, noise, lod, seed, classifier)
    rngs = MersenneTwister(model_seed(seed, u.model))
    sims = if newpdx
        w0 = M.latent_baseline(pd, v0, length(draws); rng = MersenneTwister(model_seed(seed, "$(u.model)|resp")))
        M.simulate_regimen(pd, pd.doses; newpdx = true, tgrid = t, w0, draws, obs_noise = noise, rng = rngs)
    else
        M.simulate_regimen(pd, pd.doses; model = u.model, tgrid = t, draws, obs_noise = noise, rng = rngs)
    end
    sims[:, 2:end] .= ifelse.(sims[:, 2:end] .< lod, 0.0, sims[:, 2:end])
    sims[:, 1] .= v0
    return M.response_categories(t, sims; classifier)
end

"Brier score of probabilities `p` for Bool outcomes `y`."
brier(p, y) = mean((p .- y) .^ 2)

"""
Plain-language reading of an AUROC for new-PDX response prediction. New PDX get exchangeable random effects and no
covariates, so P(responder) differs between held-out PDX only through baseline volume, measurement schedule and Monte
Carlo variation; a CI that includes 0.5 is reported as no demonstrated discrimination.
"""
function auroc_statement(auc, ci, npos, nneg)
    (npos == 0 || nneg == 0 || isnan(auc)) &&
        return "AUROC is undefined: the held-out set has $(npos == 0 ? "no responders" : "no non-responders")."
    npos == 1 && return "No demonstrated responder discrimination for new PDX (exchangeable random effects, no covariates). " *
        "Only one held-out responder: the AUROC is just the rank of that single PDX, and the cluster bootstrap conditional on one positive PDX is not informative (its interval is not evidence of discrimination)."
    msg = if isnan(ci[1]) || ci[1] <= 0.5 <= ci[2]
        "No demonstrated responder discrimination for new PDX (exchangeable random effects, no covariates): the AUROC interval includes 0.5."
    elseif ci[1] > 0.5
        "The AUROC interval excludes 0.5, but predictions are population-level (exchangeable random effects, no covariates): any discrimination comes from baseline volume and measurement schedule, not from PDX-specific biology."
    else
        "The AUROC interval lies below 0.5: predictions rank held-out PDX worse than chance."
    end
    npos < 5 && (msg *= " Only $(npos) held-out responders: the AUROC interval is wide and fragile.")
    return msg
end

"""
Response metrics for held-out PDX on the curves cut to days <= `window` (the primary fit window), with `ndraws` common
posterior draws (row s = one population draw across models). The *recorded response* (noisy categories) is the main
prediction; the *latent response* (noise-free trajectories, every draw classified) is a separately labelled sensitivity
line. Observed and predicted categories use the paper variant of the PDXE rules. AUROC and its bootstrap CI both use
P(responder) of the recorded response; Kendall τ_b uses the expected ordinal score; benchmarks are the constant-risk
predictor (AUROC 0.5, Brier of the mean predicted risk) and the prevalence oracle. With `train_units` the predicted
responder fraction of the training PDX (known random effects, in-sample) is reported next to the held-out one.
Responder-fraction intervals are posterior predictive (they include finite-cohort variability).
"""
function response_metrics(pd, units; draws, window = PRIMARY_WINDOW, lod, seed = 7, obs_cat = nothing,
                          classifier = M.pdxe_response_paper, train_units = nothing)
    t0 = time()
    names = String[]; ocat = String[]; opub = String[]; pscore = Float64[]; pres = Float64[]; modal = String[]
    pres_lat = Float64[]; pscore_lat = Float64[]
    R = Vector{Vector{Union{Missing,Bool}}}(); Rl = Vector{Vector{Union{Missing,Bool}}}()
    asresp(cats) = [x == "NA" ? missing : M.is_responder(x) for x in cats]
    for u in units
        c = cut_curve(u.treated, window)
        (length(c.t) >= 2 && issorted(c.t) && c.t[1] == 0) || continue
        oc = classifier(c.t, c.v).category
        oc == "NA" && continue
        v0 = baseline_volume(u.treated, median(pd.w0_treated))
        cats = response_cats(pd, u, c.t, v0; draws, newpdx = true, noise = true, lod, seed, classifier)
        latc = response_cats(pd, u, c.t, v0; draws, newpdx = true, noise = false, lod, seed, classifier)
        pr = M.category_probs(cats)
        push!(names, u.model); push!(ocat, oc)
        push!(pscore, M.expected_score(cats)); push!(pres, M.p_responder(cats))
        push!(pscore_lat, M.expected_score(latc)); push!(pres_lat, M.p_responder(latc))
        push!(modal, M.RESPONSE_CATEGORIES[argmax([pr.mCR, pr.mPR, pr.mSD, pr.mPD])])
        push!(R, asresp(cats)); push!(Rl, asresp(latc))
        push!(opub, obs_cat === nothing ? "NA" : get(obs_cat, u.model, "NA"))
    end
    isempty(names) && return nothing
    ag = M.ordinal_agreement(pscore, ocat; responder_scores = pres)
    ag_lat = M.ordinal_agreement(pscore_lat, ocat; responder_scores = pres_lat)
    pub = findall(!=("NA"), opub)
    ag_pub = isempty(pub) ? nothing : M.ordinal_agreement(pscore[pub], opub[pub]; responder_scores = pres[pub])
    rf = M.responder_fraction(Matrix{Union{Missing,Bool}}(reduce(hcat, R)))
    rf_lat = M.responder_fraction(Matrix{Union{Missing,Bool}}(reduce(hcat, Rl)))
    yresp = M.is_responder.(ocat)
    obs_rf = mean(yresp)
    cats4 = collect(M.RESPONSE_CATEGORIES)
    conf = [count((ocat .== a) .& (modal .== b)) for a in cats4, b in cats4]      # rows observed, cols predicted modal
    boot = M.cluster_bootstrap(names; nboot = 1000, rng = MersenneTwister(seed)) do idx
        M.auroc(pres[idx], M.is_responder.(ocat[idx]))
    end
    bf = filter(isfinite, boot)
    auc_ci = isempty(bf) ? (NaN, NaN) : (quantile(bf, 0.025), quantile(bf, 0.975))
    npos = count(yresp); nneg = length(yresp) - npos
    pbar = mean(pres)
    bench = (brier_model = brier(pres, yresp), brier_constant = brier(fill(pbar, length(pres)), yresp),
             brier_prevalence_oracle = obs_rf * (1 - obs_rf), constant_risk = pbar, auroc_constant = 0.5)
    # training PDX: known random effects, in-sample
    ins = nothing
    if train_units !== nothing
        oy = Bool[]; Rt = Vector{Vector{Union{Missing,Bool}}}(); Rtl = Vector{Vector{Union{Missing,Bool}}}()
        for u in train_units
            c = cut_curve(u.treated, window)
            (length(c.t) >= 2 && issorted(c.t) && c.t[1] == 0) || continue
            oc = classifier(c.t, c.v).category
            oc == "NA" && continue
            v0 = baseline_volume(u.treated, median(pd.w0_treated))
            push!(oy, M.is_responder(oc))
            push!(Rt, asresp(response_cats(pd, u, c.t, v0; draws, newpdx = false, noise = true, lod, seed, classifier)))
            push!(Rtl, asresp(response_cats(pd, u, c.t, v0; draws, newpdx = false, noise = false, lod, seed, classifier)))
        end
        isempty(oy) || (ins = (n_pdx = length(oy), obs_frac = mean(oy),
                               noisy = M.responder_fraction(Matrix{Union{Missing,Bool}}(reduce(hcat, Rt))),
                               latent = M.responder_fraction(Matrix{Union{Missing,Bool}}(reduce(hcat, Rtl)))))
    end
    return (names = names, obs_cat = ocat, obs_published = opub, pred_modal = modal,
            n_pdx = length(names), n_draws = length(draws), runtime_s = time() - t0, n_pos = npos, n_neg = nneg,
            pred_score = pscore, p_resp = pres, p_resp_latent = pres_lat, agreement = ag, agreement_latent = ag_lat,
            agreement_vs_published = ag_pub, auroc_ci = auc_ci, auroc_boot_dropped = 1 - length(bf) / length(boot),
            auroc_statement = auroc_statement(ag.auroc, auc_ci, npos, nneg), benchmark = bench,
            pred_resp_frac = rf, pred_resp_frac_latent = rf_lat, obs_resp_frac = obs_rf, insample = ins,
            confusion = conf, modal_agreement = mean(modal .== ocat), window = window,
            cat_counts = [count(==(c), ocat) for c in cats4])
end

# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------

"""
Figures. PUBLIC (in `outdir`): `fig_<drug>_logratio.png`, `fig_<drug>_calibration.png`, `fig_<drug>_responsecal.png`,
all aggregated over at least `min_pdx` PDX models. LOCAL (in `outdir/local`, never published): `fig_<drug>_bands.png`
(individual curves with observed points) and `fig_<drug>_response.png` (per-PDX observed category vs predicted
P(responder)), plus `response_<drug>.csv`. Returns `(public, local)` path vectors.
"""
function make_figures(drug, outdir, recs, cm, rm, bins = NamedTuple[]; min_pdx = M.PUBLIC_MIN_PDX)
    pub = String[]; loc = String[]
    mkpath(outdir); ldir = joinpath(outdir, "local"); mkpath(ldir)
    (isempty(recs) && rm === nothing) && return (pub, loc)
    using_cairo()
    CM = Base.invokelatest(() -> getfield(Main, :CairoMakie))
    isempty(recs) || Base.invokelatest(figure_bands, CM, drug, ldir, recs, loc)
    isempty(bins) || Base.invokelatest(figure_logratio, CM, drug, outdir, bins, pub)
    (cm === nothing || cm.n_pdx < min_pdx) || Base.invokelatest(figure_calibration, CM, drug, outdir, cm, pub)
    if rm !== nothing
        Base.invokelatest(figure_response, CM, drug, ldir, rm, loc)
        open(joinpath(ldir, "response_$(drug).csv"), "w") do io
            println(io, "model,observed_category,published_category,modal_predicted,p_responder,expected_score")
            for i in eachindex(rm.names)
                println(io, join([rm.names[i], rm.obs_cat[i], rm.obs_published[i], rm.pred_modal[i], rm.p_resp[i], rm.pred_score[i]], ","))
            end
        end
        push!(loc, joinpath(ldir, "response_$(drug).csv"))
        rm.n_pdx >= min_pdx && Base.invokelatest(figure_response_cal, CM, drug, outdir, rm, pub; min_pdx)
    end
    return (pub, loc)
end
using_cairo() = isdefined(Main, :CairoMakie) || Core.eval(Main, :(using CairoMakie))

function figure_bands(CM, drug, outdir, recs, paths)
    sel = recs[1:min(12, length(recs))]
    fig = CM.Figure(size = (1100, 220 * ceil(Int, length(sel) / 4)))
    for (k, r) in enumerate(sel)
        ax = CM.Axis(fig[div(k - 1, 4) + 1, mod1(k, 4)], title = "$(r.model) $(r.arm)",
                     xlabel = "day", ylabel = "volume (mm³)", yscale = log10)
        b = M.predictive_bands(r.sims; levels = (0.05, 0.5, 0.95))
        ok = all(>(0), b[:, 1])
        CM.band!(ax, r.t, max.(b[:, 1], 1e-1), b[:, 3], color = (:steelblue, 0.3))
        CM.lines!(ax, r.t, max.(b[:, 2], 1e-1), color = :steelblue)
        CM.scatter!(ax, r.t, max.(r.y, 1e-1), color = :black, markersize = 6)
    end
    p = joinpath(outdir, "fig_$(drug)_bands.png"); CM.save(p, fig); push!(paths, p)
end

function figure_logratio(CM, drug, outdir, bins, paths)
    fig = CM.Figure(size = (1200, 380))
    for (j, arm) in enumerate((:treated, :control))
        ax = CM.Axis(fig[1, j], title = "held-out PDX, $(arm) arm", xlabel = "day", ylabel = "log(V/V0), censored at LOD")
        bb = [b for b in bins if b.arm === arm]
        isempty(bb) && continue
        x = [(b.lo + b.hi) / 2 for b in bb]
        CM.band!(ax, x, [b.pred_q[1] for b in bb], [b.pred_q[3] for b in bb], color = (:steelblue, 0.3), label = "predicted 5-95%")
        CM.lines!(ax, x, [b.pred_q[2] for b in bb], color = :steelblue, label = "predicted median")
        CM.rangebars!(ax, x, [b.obs_q[1] for b in bb], [b.obs_q[3] for b in bb], color = :black, whiskerwidth = 8)
        CM.scatter!(ax, x, [b.obs_q[2] for b in bb], color = :black, label = "observed median, 5-95%")
        j == 1 && CM.axislegend(ax, position = :lb)
    end
    ax3 = CM.Axis(fig[1, 3], title = "fraction below LOD", xlabel = "day", ylabel = "fraction", limits = (nothing, nothing, 0, 1))
    for (arm, col) in ((:treated, :firebrick), (:control, :gray40))
        bb = [b for b in bins if b.arm === arm]
        isempty(bb) && continue
        x = [(b.lo + b.hi) / 2 for b in bb]
        CM.lines!(ax3, x, [b.pred_below for b in bb], color = col, label = "$(arm) predicted")
        CM.scatter!(ax3, x, [b.obs_below for b in bb], color = col, marker = :diamond, label = "$(arm) observed")
    end
    CM.axislegend(ax3, position = :lt)
    p = joinpath(outdir, "fig_$(drug)_logratio.png"); CM.save(p, fig); push!(paths, p)
end

function figure_calibration(CM, drug, outdir, cm, paths)
    fig = CM.Figure(size = (420, 420))
    ax = CM.Axis(fig[1, 1], xlabel = "nominal level", ylabel = "empirical coverage",
                 title = "Held-out calibration ($(drug))", limits = (0, 1, 0, 1))
    CM.lines!(ax, [0, 1], [0, 1], color = :gray, linestyle = :dash)
    CM.scatterlines!(ax, first.(cm.calib), last.(cm.calib), color = :steelblue)
    p = joinpath(outdir, "fig_$(drug)_calibration.png"); CM.save(p, fig); push!(paths, p)
end

"Aggregated: predicted P(responder) groups of at least `min_pdx` held-out PDX vs observed responder fraction."
function figure_response_cal(CM, drug, outdir, rm, paths; min_pdx)
    n = length(rm.p_resp); k = min(5, n ÷ min_pdx)
    k >= 2 || return
    ord = sortperm(rm.p_resp)
    chunks = [ord[round(Int, (i - 1) * n / k) + 1:round(Int, i * n / k)] for i in 1:k]
    all(length(c) >= min_pdx for c in chunks) || return
    xp = [mean(rm.p_resp[c]) for c in chunks]; yo = [mean(M.is_responder.(rm.obs_cat[c])) for c in chunks]
    fig = CM.Figure(size = (440, 420))
    ax = CM.Axis(fig[1, 1], xlabel = "mean predicted P(responder)", ylabel = "observed responder fraction",
                 title = "Responder calibration, $(drug) (groups of >= $(min_pdx) held-out PDX)", titlesize = 12, limits = (0, 1, 0, 1))
    CM.lines!(ax, [0, 1], [0, 1], color = :gray, linestyle = :dash)
    CM.scatterlines!(ax, xp, yo, color = :steelblue)
    p = joinpath(outdir, "fig_$(drug)_responsecal.png"); CM.save(p, fig); push!(paths, p)
end

function figure_response(CM, drug, outdir, rm, paths)
    fig = CM.Figure(size = (520, 420))
    ax = CM.Axis(fig[1, 1], xlabel = "observed category", ylabel = "predicted P(responder)",
                 xticks = (1:4, collect(M.RESPONSE_CATEGORIES)), title = "New-PDX response prediction ($(drug))")
    x = [findfirst(==(c), M.RESPONSE_CATEGORIES) for c in rm.obs_cat]
    CM.scatter!(ax, x .+ 0.15 .* (rand(MersenneTwister(1), length(x)) .- 0.5), rm.p_resp, color = (:steelblue, 0.7))
    p = joinpath(outdir, "fig_$(drug)_response.png"); CM.save(p, fig); push!(paths, p)
end

# ---------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------

f3(x) = @sprintf("%.3f", x)
pct(x) = @sprintf("%.1f%%", 100x)
fmt(x) = x === nothing ? "not recorded" : x isa AbstractFloat ? @sprintf("%.4g", x) : string(x)

function metrics_table(io, name, cm)
    cm === nothing && return println(io, "$name: not available (no fit file / no held-out points).\n")
    println(io, "**$name**: $(cm.n_points) scored points ($(cm.n_censored) undetectable, scored as left-censored at ",
            "LOD $(cm.lod) mm³ in the LPD only), $(cm.n_curves) curves, $(cm.n_pdx) PDX models; nominal level $(pct(cm.level)); ",
            "observation model Student-t, ν = $(cm.nu).\n")
    println(io, "| metric | value |\n|---|---|")
    println(io, "| coverage, pooled | $(pct(cm.coverage_pooled)) (PDX-cluster bootstrap 95% CI $(pct(cm.coverage_pdx_ci[1]))–$(pct(cm.coverage_pdx_ci[2]))) |")
    println(io, "| coverage, PDX-averaged (each PDX weighted equally) | $(pct(cm.coverage_tumour_avg)) |")
    println(io, "| mean interval width (log-volume) | $(f3(cm.width_log_mean)) (NaN for $(cm.n_width_nan) points) |")
    println(io, "| mean log predictive density, finite scores (volume scale) | $(f3(cm.lpd_mean_finite)) over $(cm.lpd_n - cm.n_lpd_neginf) points |")
    println(io, "| LPD scores equal to -Inf (no valid draw), kept | $(cm.n_lpd_neginf); mean over all scored points incl. these: $(cm.lpd_mean_all) |")
    println(io, "| mean CRPS (mm³) | $(f3(cm.crps_mean)) (NaN for $(cm.n_crps_nan) points) |")
    println(io, "| failed solves | $(cm.n_failed_draw_rows) draw rows with non-finite values; $(cm.n_empty_ensembles) empty ensembles (counted as coverage misses) |\n")
end

function binned_table(io, bins)
    isempty(bins) && return println(io, "No binned check available.\n")
    println(io, "Pooled over PDX × posterior draws; both observed and simulated volumes censored at the LOD; log(V/V0) with V0 the observed baseline. ",
            "Quantiles are 5% / 50% / 95%. Tail calibration: fraction of observed positive volumes below the predictive 5% quantile / above the 95% ",
            "quantile (nominal 5% / 5%). Below the 5% quantile = tumours smaller than predicted (they shrink more or grow less than the predictive lower tail allows); excess regrowth shows in the upper tail (above the 95% quantile).\n")
    println(io, "| arm | days | n obs (PDX) | observed log(V/V0) | predicted log(V/V0) | observed < LOD | predicted < LOD | 90% coverage | below 5% / above 95% |\n|---|---|---|---|---|---|---|---|---|")
    for b in bins
        q(t) = @sprintf("%.2f / %.2f / %.2f", t...)
        tails = isnan(b.tail_below5) ? "n/a" : "$(pct(b.tail_below5)) / $(pct(b.tail_above95))"
        println(io, "| $(b.arm) | $(round(Int, b.lo))–$(round(Int, b.hi)) | $(b.n) ($(b.n_pdx)) | $(q(b.obs_q)) | $(q(b.pred_q)) | $(pct(b.obs_below)) | $(pct(b.pred_below)) | $(isnan(b.coverage90) ? "n/a" : pct(b.coverage90)) | $tails |")
    end
    println(io)
end

function residual_table(io, rows)
    isempty(rows) && return println(io, "No residual check available.\n")
    println(io, "Median signed log residual, `log y - median log(predictive)` per observation, reduced to one median per PDX in the bin, then the median across PDX ",
            "(each PDX equal weight; 90% PDX-bootstrap interval). The balanced-panel column uses only PDX observed in every bin of the arm, so changes between bins ",
            "are within-PDX rather than a changing mix of PDX. Positive = observed larger than predicted.\n")
    println(io, "| arm | days | PDX | median residual (90% interval) | balanced panel (PDX) median residual |\n|---|---|---|---|---|")
    for r in rows
        pan = isnan(r.panel_median) ? "n/a" : @sprintf("%+.3f (%d)", r.panel_median, r.n_panel)
        println(io, "| $(r.arm) | $(round(Int, r.lo))–$(round(Int, r.hi)) | $(r.n_pdx) | $(@sprintf("%+.3f (%+.3f, %+.3f)", r.median, r.ci...)) | $pan |")
    end
    println(io)
end

function convergence_md(io, label, pd)
    pd === nothing && return println(io, "**$label**: fit not available.\n")
    c = M.convergence_summary(pd)
    println(io, "**$label** ($(c.nchains) chains, $(c.n_draws) stored draws, $(fmt(c.draws_per_chain)) per chain)\n")
    println(io, "| item | recorded by the fit |\n|---|---|")
    for (nm, v) in (("max R-hat (all parameters)", c.max_rhat), ("min bulk ESS", c.min_ess_bulk), ("min tail ESS", c.min_ess_tail),
                    ("divergent transitions", c.n_divergences), ("E-BFMI (min over chains)", c.ebfmi),
                    ("tree-depth cap hits", c.max_tree_depth_hits), ("mean tree depth", c.mean_tree_depth),
                    ("mean acceptance", c.mean_acceptance), ("mean step size", c.mean_step_size), ("mean leapfrog steps", c.mean_n_steps),
                    ("cross-mode R-hat", c.cross_mode_rhat), ("MAP log-densities of the starts", c.map_lps))
        println(io, "| $nm | $(fmt(v)) |")
    end
    for (k, v) in sort(collect(c.extra); by = first)
        println(io, "| $k | $(fmt(v)) |")
    end
    println(io)
    if c.recomputed === nothing
        println(io, "Hyperparameter R-hat/ESS could not be recomputed (draw count not divisible by nchains).\n")
    else
        r = c.recomputed
        println(io, "Recomputed from the stored draws of the hyperparameters (chain layout from `nchains`):\n")
        println(io, "| parameter | R-hat | bulk ESS | tail ESS |\n|---|---|---|---|")
        for (i, nm) in enumerate(r.names)
            println(io, "| $nm | $(@sprintf("%.3f", r.rhat[i])) | $(@sprintf("%.0f", r.ess_bulk[i])) | $(@sprintf("%.0f", r.ess_tail[i])) |")
        end
        println(io)
    end
    flags = String[]
    c.max_rhat !== nothing && c.max_rhat > 1.01 && push!(flags, "max R-hat $(fmt(c.max_rhat)) > 1.01")
    c.n_divergences !== nothing && c.n_divergences > 0 && push!(flags, "$(c.n_divergences) divergent transitions")
    c.min_ess_bulk !== nothing && c.min_ess_bulk < 100 * c.nchains && push!(flags, "min bulk ESS $(fmt(c.min_ess_bulk)) < 100 per chain")
    c.ebfmi !== nothing && minimum(c.ebfmi) < 0.3 && push!(flags, "E-BFMI $(fmt(c.ebfmi)) < 0.3")
    c.cross_mode_rhat !== nothing && c.cross_mode_rhat > 1.01 && push!(flags, "cross-mode R-hat $(fmt(c.cross_mode_rhat)) > 1.01")
    c.recomputed !== nothing && any(x -> isfinite(x) && x > 1.01, c.recomputed.rhat) && push!(flags, "recomputed hyperparameter R-hat > 1.01")
    println(io, isempty(flags) ? "No convergence flags.\n" : "**Convergence flags: " * join(flags, "; ") * "**\n")
end

"""
Bins to hand to the public guard: all bins (their `n_pdx`) plus, for bins that show a coverage / tail value, a pseudo-bin
carrying the number of PDX that actually contribute to those statistics (positive observations only).
"""
guard_bins(bins) = vcat(bins, [(arm = b.arm, lo = b.lo, hi = b.hi, n_pdx = b.n_pdx_cov) for b in bins
                               if hasproperty(b, :n_pdx_cov) && !isnan(b.coverage90)])

"""
Writes the PUBLIC report `outdir/validation_<drug>.md` (aggregated outputs only: every bin and summary aggregates
at least `min_pdx` PDX models, no PDX ids, no individual observations). Summaries with fewer PDX are suppressed
and said so in the text; `assert_public_output` then refuses to write if any bin/summary under `min_pdx`, any id in
`ids` or any individual-level figure is still present.
"""
function write_report(drug, outdir; primary = nothing, late = nothing, rm = nothing, bins_primary = NamedTuple[],
                      bins_late = NamedTuple[], loo_c = nothing, loo_m = nothing, pd_primary = nothing, pd_late = nothing,
                      resid_primary = NamedTuple[], resid_late = NamedTuple[], tag = "",
                      figs = String[], notes = String[], settings = "", unverified = String[],
                      ids = String[], min_pdx = M.PUBLIC_MIN_PDX)
    mkpath(outdir)
    name = isempty(tag) ? drug : "$(drug)_$(tag)"
    path = joinpath(outdir, "validation_$(name).md")
    notes = copy(notes)
    sup(x, what) = (x !== nothing && x.n_pdx < min_pdx) ? (push!(notes, "$(what) suppressed: it aggregates only $(x.n_pdx) PDX models (< $(min_pdx))."); nothing) : x
    primary = sup(primary, "New-PDX summary"); late = sup(late, "Late-21 summary"); rm = sup(rm, "Response summary")
    if rm !== nothing && rm.insample !== nothing && rm.insample.n_pdx < min_pdx
        push!(notes, "In-sample (training PDX) response summary suppressed: it aggregates only $(rm.insample.n_pdx) PDX models (< $(min_pdx)).")
        rm = merge(rm, (insample = nothing,))
    end
    text = let io = IOBuffer()
        println(io, "# Validation: $(name)\n")
        println(io, "Generated $(Dates.format(now(), "yyyy-mm-dd HH:MM")). $(settings)\n")
        println(io, "Design: the **primary** fit (frozen stratified 80% of PDX, days ≤ $(round(Int, PRIMARY_WINDOW)) only) predicts the held-out 20% PDX; ",
                "held-out PDX are therefore scored on days ≤ $(round(Int, PRIMARY_WINDOW)) only, and the observed response categories below are recomputed on ",
                "that window. The **late21** fit (same 80%, days ≤ $(round(Int, LATE_WINDOW))) forecasts days $(round(Int, LATE_WINDOW) + 1)–$(round(Int, PRIMARY_WINDOW)) of its training PDX. ",
                "Observation model: Student-t on log volume, censored at the detection limit. ",
                "This report is aggregated: every bin and summary pools at least $(min_pdx) PDX models, and there are no individual PDX ids or observations ",
                "(the licensed PDXE data are not redistributed); individual-curve figures are written to the local output directory only.\n")
        for n in notes; println(io, "> ", n); end
        isempty(unverified) || println(io, "> Fit metadata not recorded, so these were NOT verified: ", join(unverified, ", "), ".")
        println(io, "\n## Held-out PDX predictive check (primary fit; new PDX, fresh random effects, days ≤ $(round(Int, PRIMARY_WINDOW)))\n")
        metrics_table(io, "New-PDX prediction", primary)
        println(io, "### log(V/V0) by day, with noise and censoring\n")
        binned_table(io, bins_primary)
        isempty(resid_primary) || (println(io, "### Residuals by day, held-out PDX\n"); residual_table(io, resid_primary))
        println(io, "### Response categories on held-out PDX (PDXE rules, Gao et al. 2015)\n")
        if rm === nothing
            println(io, "Not available.\n")
        else
            println(io, "Observed and predicted categories use `pdxe_response_paper` (windows t > 10, short curves on all points) on the days ≤ $(round(Int, rm.window)) record; ",
                    "simulated volumes below the LOD are recorded as 0 like the data and the day-0 volume is the observed one. This rule reproduces the published BestAvgResponse in 98.4% of PDXE curves but its BestResponse is ",
                    "systematically lower than the published one in ~40% of curves (published mPD scored mSD by the rule): 4-way agreement with the published labels is ~91%, ",
                    "responder vs non-responder ~99.9%.\n")
            println(io, "**Recorded response** = categories of simulated noisy observations (the prediction of what would be recorded). **Latent response** = categories of the ",
                    "noise-free trajectories, every draw classified; it is a sensitivity line that isolates the contribution of observation noise (same random effects, same LOD recording and baseline convention), ",
                    "not a prediction of recorded categories. $(rm.n_draws) common posterior draws; response section runtime $(round(rm.runtime_s; digits = 1)) s.\n")
            println(io, "Observed category counts (mCR/mPR/mSD/mPD): ", join(rm.cat_counts, "/"), " over ", length(rm.names), " held-out PDX ($(rm.n_pos) responder$(rm.n_pos == 1 ? "" : "s")).\n")
            println(io, "Responder fractions (all intervals are **posterior predictive, including finite-cohort variability**, 90%):\n")
            println(io, "| | held-out PDX (new, n=$(rm.n_pdx)) | training PDX (known random effects, in-sample$(rm.insample === nothing ? "" : ", n=$(rm.insample.n_pdx)")) |\n|---|---|---|")
            ins = rm.insample
            fr(x) = "$(pct(x.mean)) ($(pct(x.lo))–$(pct(x.hi)))"
            println(io, "| observed responder fraction | $(pct(rm.obs_resp_frac)) | $(ins === nothing ? "n/a" : pct(ins.obs_frac)) |")
            println(io, "| predicted, recorded response (noisy) | $(fr(rm.pred_resp_frac)) | $(ins === nothing ? "n/a" : fr(ins.noisy)) |")
            println(io, "| predicted, latent response (noise-free) | $(fr(rm.pred_resp_frac_latent)) | $(ins === nothing ? "n/a" : fr(ins.latent)) |\n")
            println(io, "$(rm.pred_resp_frac.n_missing) failed model-draws excluded (held-out, recorded).\n")
            println(io, "| metric | value |\n|---|---|")
            println(io, "| Kendall τ_b (predicted expected ordinal score vs observed ordinal) | $(f3(rm.agreement.tau_b)) (latent: $(f3(rm.agreement_latent.tau_b))) |")
            println(io, "| AUROC responder (mCR/mPR) vs rest, ranked by predicted P(responder) | $(f3(rm.agreement.auroc)) (same score; cluster-bootstrap 95% CI $(f3(rm.auroc_ci[1]))–$(f3(rm.auroc_ci[2])); $(pct(rm.auroc_boot_dropped)) of replicates had no responder or no non-responder and were dropped; latent: $(f3(rm.agreement_latent.auroc))) |")
            println(io, "| constant-risk benchmark | AUROC 0.5 by construction. Brier score: model $(f3(rm.benchmark.brier_model)); constant risk at the mean predicted P(responder) = $(f3(rm.benchmark.constant_risk)): $(f3(rm.benchmark.brier_constant)); prevalence-oracle reference $(f3(rm.benchmark.brier_prevalence_oracle)) |")
            println(io, "| modal predicted category = observed category | $(pct(rm.modal_agreement)) |")
            rm.agreement_vs_published === nothing || println(io,
                "| vs the published (full follow-up) categories: τ_b / AUROC | $(f3(rm.agreement_vs_published.tau_b)) / $(f3(rm.agreement_vs_published.auroc)) (n=$(rm.agreement_vs_published.n)) |")
            println(io)
            println(io, "**", rm.auroc_statement, "**\n")
            println(io, "Confusion matrix (rows observed, columns modal predicted category):\n")
            println(io, "| | ", join(M.RESPONSE_CATEGORIES, " | "), " |\n|---|---|---|---|---|")
            for (i, c) in enumerate(M.RESPONSE_CATEGORIES)
                println(io, "| $c | ", join(rm.confusion[i, :], " | "), " |")
            end
            println(io)
        end
        println(io, "## Forecast of days $(round(Int, LATE_WINDOW) + 1)–$(round(Int, PRIMARY_WINDOW)) from the late21 fit (known tumours)\n")
        metrics_table(io, "Late-21 forecast", late)
        isempty(bins_late) || (println(io, "### log(V/V0) by day, late-21 forecast\n"); binned_table(io, bins_late))
        isempty(resid_late) || (println(io, "### Within-PDX residuals by day, late-21 forecast\n"); residual_table(io, resid_late))
        println(io, "## Convergence\n")
        convergence_md(io, "Primary fit", pd_primary)
        convergence_md(io, "Late-21 fit", pd_late)
        println(io, "## PSIS-LOO (primary fit, training PDX, days ≤ $(round(Int, PRIMARY_WINDOW)))\n")
        for (nm, l) in (("Conditional, one unit per PDX model (random effects fixed at their draws)", loo_c),
                        ("Marginal, one unit per PDX model (random effects integrated; new-PDX LOO)", loo_m))
            if l === nothing
                println(io, "$nm: not computed.\n")
            else
                println(io, "**$nm**: elpd = $(@sprintf("%.1f", l.elpd)) ± $(@sprintf("%.1f", l.se)), p_eff = $(@sprintf("%.1f", l.p_eff)), ",
                        "n = $(l.n); Pareto-k ≤0.5 / 0.5–0.7 / 0.7–1 / >1: ",
                        "$(l.pareto_k_counts.good)/$(l.pareto_k_counts.ok)/$(l.pareto_k_counts.bad)/$(l.pareto_k_counts.very_bad). ",
                        l.pareto_k_counts.bad + l.pareto_k_counts.very_bad > 0 ?
                            "Units with k > 0.7 need exact refits before trusting elpd.\n" : "All k ≤ 0.7.\n")
            end
        end
        if !isempty(figs)
            println(io, "## Figures\n")
            for f in figs; println(io, "![](", basename(f), ")"); end
        end
        String(take!(io))
    end
    summaries = [x for x in (primary, late, rm) if x !== nothing]
    rm !== nothing && rm.insample !== nothing && push!(summaries, rm.insample)
    loo_n = [(n_pdx = l.n,) for l in (loo_c, loo_m) if l !== nothing]
    M.assert_public_output(; bins = vcat(guard_bins(bins_primary), guard_bins(bins_late), resid_primary, resid_late), summaries = vcat(summaries, loo_n), text, ids, figures = figs, min_pdx)
    write(path, text)
    return path
end


# ---------------------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------------------

"Repo-relative form of a path for public outputs (never an absolute local path: outside the repo only the base name)."
function public_path(p)
    a = abspath(p); r = relpath(a, ROOT)
    return startswith(r, "..") ? basename(a) : r
end

load_if(path) = isfile(path) ? M.load_fit(path) : nothing

"Path of a merged fit: `<dir>/<drug>_<design><suffix>.jld2`, e.g. suffix `_v2` or `_v2_waning`."
fit_path(dir, drug, design, suffix = "") = joinpath(dir, "$(drug)_$(design)$(suffix).jld2")

primary_select(c) = (c.t .> 0) .& (c.t .<= PRIMARY_WINDOW)
late_select(c) = (c.t .> LATE_WINDOW) .& (c.t .<= PRIMARY_WINDOW)
const EDGES_PRIMARY = [0.0, 7, 14, 21, 28, 35, 42]
const EDGES_LATE = [21.0, 28, 35, 42]
const PUBMIN = M.PUBLIC_MIN_PDX

"""
Everything computed from the (already verified) fits: held-out PDX predictive check and response metrics from the
primary fit, late21 forecast metrics, binned and residual checks (merged to >= PUBMIN PDX per bin). `counts = (primary, late, resp)`
are the actual numbers of predictive draws (curve checks of the primary and late21 fits, response section); draw indices
are `randperm(MersenneTwister(seed), S)[1:n]` of each fit's own `S` stored draws (`seed + 1` for the response section),
so the plan is "same count, same seed rule" and the counts are reported.
"""
function evaluate_fits(units, pd_primary, pd_late, train_ids, test_ids; counts, seed = 1, obs_cat = nothing)
    o = Dict{Symbol,Any}(:recs_new => CurveRecord[], :recs_late => CurveRecord[], :cm_new => nothing, :cm_late => nothing,
                         :rm => nothing, :bins_new => NamedTuple[], :bins_late => NamedTuple[],
                         :resid_new => NamedTuple[], :resid_late => NamedTuple[], :counts => counts)
    if pd_primary !== nothing
        test_units = [u for u in units if u.model in test_ids]
        lod = M.detection_limit_from_fit(pd_primary); nu = M.nu_from_fit(pd_primary)
        draws = common_draws(pd_primary, counts.primary, seed)
        o[:recs_new] = predict_units(pd_primary, test_units; newpdx = true, select = primary_select, draws, seed)
        o[:cm_new] = curve_metrics(o[:recs_new]; lod, nu)
        tp = pooled_tails(o[:recs_new], :treated); cp = pooled_tails(o[:recs_new], :control)
        o[:cm_new] = merge(o[:cm_new], (tails_treated = tp, tails_control = cp))
        o[:bins_new] = binned_check(o[:recs_new]; lod, edges = EDGES_PRIMARY, min_pdx = PUBMIN)
        o[:resid_new] = residual_check(o[:recs_new]; edges = EDGES_PRIMARY, min_pdx = PUBMIN)
        rdraws = common_draws(pd_primary, counts.resp, seed + 1)
        o[:rm] = response_metrics(pd_primary, test_units; draws = rdraws, lod, seed, obs_cat,
                                  train_units = [u for u in units if u.model in pd_primary.models])
    end
    if pd_late !== nothing
        lod = M.detection_limit_from_fit(pd_late); nu = M.nu_from_fit(pd_late)
        draws = common_draws(pd_late, counts.late, seed)
        o[:recs_late] = predict_units(pd_late, [u for u in units if u.model in pd_late.models]; newpdx = false,
                                      select = late_select, draws, seed)
        o[:cm_late] = curve_metrics(o[:recs_late]; lod, nu)
        o[:cm_late] = merge(o[:cm_late], (tails_treated = pooled_tails(o[:recs_late], :treated), tails_control = pooled_tails(o[:recs_late], :control)))
        o[:bins_late] = binned_check(o[:recs_late]; lod, edges = EDGES_LATE, min_pdx = PUBMIN)
        o[:resid_late] = residual_check(o[:recs_late]; edges = EDGES_LATE, min_pdx = PUBMIN)
    end
    return (; (k => v for (k, v) in o)...)
end

"""
Key metrics of an evaluation for the v1 vs v2 comparison table: `(name, value, n_pdx)` where `n_pdx` is the number of PDX
models that actually contribute to the value (`nothing` for items not derived from PDX data). Metrics with fewer than
PUBMIN contributing PDX are suppressed (left out).
"""
function key_metrics(ev, pd_primary, pd_late)
    km = NamedTuple[]
    add(k, v, n) = (n === nothing || n >= PUBMIN) && push!(km, (name = k, value = v, n_pdx = n))
    cm = ev.cm_new; rm = ev.rm; cl = ev.cm_late
    if cm !== nothing
        add("held-out 90% coverage", cm.coverage_pooled, cm.n_pdx); add("held-out mean log predictive density", cm.lpd_mean_finite, cm.n_pdx)
        add("held-out mean CRPS (mm³)", cm.crps_mean, cm.n_pdx)
        add("held-out treated below-5% / above-95% tails", "$(pct(cm.tails_treated.below5)) / $(pct(cm.tails_treated.above95))", cm.tails_treated.n_pdx)
    end
    if rm !== nothing
        add("observed responder fraction (held-out)", rm.obs_resp_frac, rm.n_pdx)
        add("predicted responder fraction, recorded (held-out)", rm.pred_resp_frac.mean, rm.n_pdx)
        add("predicted responder fraction, latent (held-out)", rm.pred_resp_frac_latent.mean, rm.n_pdx)
        add("Kendall τ_b (held-out)", rm.agreement.tau_b, rm.n_pdx); add("AUROC (held-out)", rm.agreement.auroc, rm.n_pdx)
        add("Brier skill vs constant risk (held-out)", 1 - rm.benchmark.brier_model / rm.benchmark.brier_constant, rm.n_pdx)
        if rm.insample !== nothing
            add("observed responder fraction (training)", rm.insample.obs_frac, rm.insample.n_pdx)
            add("predicted responder fraction, recorded (training, in-sample)", rm.insample.noisy.mean, rm.insample.n_pdx)
        end
    end
    if cl !== nothing
        add("late21 90% coverage", cl.coverage_pooled, cl.n_pdx); add("late21 mean log predictive density", cl.lpd_mean_finite, cl.n_pdx)
        add("late21 mean CRPS (mm³)", cl.crps_mean, cl.n_pdx)
        add("late21 treated below-5% / above-95% tails", "$(pct(cl.tails_treated.below5)) / $(pct(cl.tails_treated.above95))", cl.tails_treated.n_pdx)
        rr = [r for r in ev.resid_late if r.arm === :treated]
        isempty(rr) || add("late21 treated median residual, last day bin", last(rr).median, last(rr).n_pdx)
    end
    for (lab, pd) in (("primary", pd_primary), ("late21", pd_late))
        pd === nothing && continue
        c = M.convergence_summary(pd)
        c.max_rhat === nothing || add("$(lab) max R-hat", c.max_rhat, nothing)
        c.min_ess_bulk === nothing || add("$(lab) min bulk ESS", c.min_ess_bulk, nothing)
        c.cross_mode_rhat === nothing || add("$(lab) cross-mode R-hat", c.cross_mode_rhat, nothing)
    end
    ev.counts === nothing || (add("predictive draws, held-out curve check", ev.counts.primary, nothing);
                              add("predictive draws, late21 forecast", ev.counts.late, nothing);
                              add("predictive draws, response section", ev.counts.resp, nothing))
    return km
end

fmtv(x) = x isa AbstractFloat ? @sprintf("%.4g", x) : string(x)

"""
`docs/compare_<drug>.md`: v1 vs v2 key metrics (public, aggregated). A metric appears only if it is present in both
evaluations (each already suppressed below PUBMIN contributing PDX); the contributor counts of the shown metrics, and
the bins of both evaluations (`bins1`, `bins2`), go through the public guard. `plan` documents the shared split and the
sampling plan.
"""
function write_compare(drug, outdir, km1, km2; tag1, tag2, dir1, dir2, unv1, unv2, ids, plan = nothing, bins1 = NamedTuple[], bins2 = NamedTuple[])
    d1 = Dict(m.name => m for m in km1); d2 = Dict(m.name => m for m in km2)
    shown = [m for m in km1 if haskey(d2, m.name)]
    io = IOBuffer()
    println(io, "# Model comparison: $(drug), $(tag1) vs $(tag2)\n")
    println(io, "Generated $(Dates.format(now(), "yyyy-mm-dd HH:MM")). $(tag1): `$(public_path(dir1))`; $(tag2): `$(public_path(dir2))`. Both fit sets passed the frozen-split, window, ",
            "cutoff and held-out-leakage checks against one shared split (the script errors otherwise) and are scored by the same code. ",
            "Aggregated metrics only: each pools at least $(PUBMIN) contributing PDX models, and metrics with fewer are left out.\n")
    if plan !== nothing
        println(io, "Shared split: model fraction $(plan.model_frac), seed $(plan.seed), $(plan.n_holdout) held-out PDX for this drug (identical for both fit sets; ",
                "$(plan.declared ? "declared in all fits' metadata" : "not declared in the fits' metadata, defaults used and the held-out/training sets verified against them")). ",
                "Sampling plan: each fit's predictive draws are `randperm(MersenneTwister(seed), S)[1:n]` over its own stored draws with the same seed rule and the same counts ",
                "n = $(plan.counts.primary) (held-out curve check), $(plan.counts.late) (late21 forecast), $(plan.counts.resp) (response section; seed + 1); ",
                "stored draws $(tag1): primary $(plan.S1.primary), late21 $(plan.S1.late); $(tag2): primary $(plan.S2.primary), late21 $(plan.S2.late).\n")
    end
    isempty(unv1) || println(io, "> $(tag1): metadata not recorded, not verified: ", join(unv1, ", "), ".")
    isempty(unv2) || println(io, "> $(tag2): metadata not recorded, not verified: ", join(unv2, ", "), ".")
    println(io, "\n| metric | $(tag1) | $(tag2) | $(tag2) − $(tag1) |\n|---|---|---|---|")
    for m in shown
        a = m.value; b = d2[m.name].value
        delta = (a isa Real && b isa Real) ? @sprintf("%+.4g", b - a) : ""
        println(io, "| $(m.name) | $(fmtv(a)) | $(fmtv(b)) | $delta |")
    end
    println(io, "\nCoverage closer to 0.90 is better; log predictive density higher is better; CRPS lower is better; tails nominal 5% / 5% ",
            "(below the 5% quantile = smaller than predicted, above the 95% quantile = larger than predicted, i.e. excess regrowth). ",
            "Convergence rows show whether each fit set is fit for comparison: do not interpret metric differences for a fit with R-hat > 1.01 as model differences.\n")
    text = String(take!(io))
    contrib = [(n_pdx = min(m.n_pdx, d2[m.name].n_pdx),) for m in shown if m.n_pdx !== nothing && d2[m.name].n_pdx !== nothing]
    M.assert_public_output(; text, ids, summaries = contrib, bins = vcat(guard_bins(bins1), guard_bins(bins2)), figures = String[])
    path = joinpath(outdir, "compare_$(drug).md")
    write(path, text)
    return path
end

"""
`run_real(drug; fitdir, tag, ...)`: verify the fits in `fitdir` against the frozen designs and (unless `write=false`)
write `docs/validation_<drug>[_<tag>].md` (+ figures; local figures under docs/local). `split = (model_frac, seed)` overrides
the split declared in the fits (comparison mode passes one shared split, and every fit is verified against it); `counts =
(primary, late, resp)` fixes the numbers of predictive draws (default: the requested counts capped at the stored draws);
`fits = (primary, late)` passes preloaded fits. The evaluation's bins always go through the public bin guard, also with
`write=false`. Returns the evaluation and the verification outcome. `loo=false` skips PSIS-LOO.
"""
function run_real(drug; ndraws = 200, nresp = 1000, seed = 1, marginal_loo = false, loo = true, write = true, tag = "",
                  fitdir = joinpath(ROOT, "fits"), outdir = joinpath(ROOT, "docs"), split = nothing, counts = nothing, fits = nothing,
                  suffix = "")
    isdefined(M, :drug_dataset) || error("data.jl (drug_dataset, load_metrics) is not included in TumourPKPD")
    units = M.drug_dataset(drug)
    drug_models = sort([u.model for u in units])
    obs_cat = try
        mt = M.load_metrics()
        d = Dict{String,String}()
        for r in eachrow(mt)
            string(r.treatment) == drug && (d[string(r.model)] = string(r.response_category))
        end
        d
    catch
        nothing
    end
    notes = String[]; unverified = String[]
    pd_primary, pd_late = fits === nothing ?
        (load_if(fit_path(fitdir, drug, "primary", suffix)), load_if(fit_path(fitdir, drug, "late21", suffix))) : fits
    # split settings: the shared split if given, else the fit's declared frac/seed; the frozen split is recomputed and the
    # fits are verified against it
    sp(pd) = pd === nothing ? nothing : meta(pd, "split")
    getsp(k, default) = (for pd in (pd_primary, pd_late); d = sp(pd); d isa AbstractDict && haskey(d, k) && return d[k]; end; default)
    model_frac = split === nothing ? float(getsp("model_frac", 0.2)) : float(split.model_frac)
    split_seed = split === nothing ? Int(getsp("seed", 1)) : Int(split.seed)
    train_all, test_all, _ = M.frozen_pdx_split(; frac = model_frac, seed = split_seed)
    train_ids, test_ids = expected_split(drug_models; frac = model_frac, seed = split_seed, test_all)
    pd_primary === nothing && push!(notes, "No $(public_path(fitdir))/$(drug)_primary$(suffix).jld2: held-out PDX prediction, response metrics and LOO skipped.")
    pd_primary === nothing || append!(unverified, verify_primary_fit(pd_primary, drug, train_ids, test_ids; model_frac, seed = split_seed))
    pd_late === nothing && push!(notes, "No $(public_path(fitdir))/$(drug)_late21$(suffix).jld2: late-21 forecast skipped.")
    pd_late === nothing || append!(unverified, verify_late21_fit(pd_late, drug, units, train_ids, test_ids; model_frac, seed = split_seed))
    Sp = pd_primary === nothing ? 0 : length(pd_primary.σ); Sl = pd_late === nothing ? 0 : length(pd_late.σ)
    counts === nothing && (counts = (primary = min(ndraws, Sp), late = min(ndraws, Sl), resp = min(nresp, Sp)))
    (counts.primary <= Sp && counts.resp <= Sp && counts.late <= Sl) || error("requested draw counts exceed the stored draws of $(fitdir)")
    ev = evaluate_fits(units, pd_primary, pd_late, train_ids, test_ids; counts, seed, obs_cat)
    # public bin guard for every evaluation (also when no report is written)
    M.assert_public_output(; bins = vcat(guard_bins(ev.bins_new), guard_bins(ev.bins_late), ev.resid_new, ev.resid_late))
    loo_c = loo_m = nothing
    if loo && pd_primary !== nothing
        ci = M.chain_index_from_fit(pd_primary)
        fit_units = M.window_units([u for u in units if u.model in pd_primary.models], PRIMARY_WINDOW)
        loo_c = M.loo(M.grouped_loglik(pd_primary, fit_units); chain_index = ci)
        marginal_loo && (loo_m = M.loo(M.marginal_unit_loglik(pd_primary, fit_units); chain_index = ci))
    end
    path = nothing
    if write
        name = isempty(tag) ? drug : "$(drug)_$(tag)"
        figs, local_figs = make_figures(name, outdir, ev.recs_new, ev.cm_new, ev.rm, ev.bins_new)
        mdl = pd_primary !== nothing ? pd_primary : pd_late
        modelinfo = mdl === nothing ? "" : " Model: $(meta(mdl, "model", "v1"))$(Bool(meta(mdl, "waning", false)) ? " with waning efficacy" : "")."
        settings = "Fits: `$(public_path(fitdir))` (files `*$(suffix).jld2`).$(modelinfo) Frozen split frac $(model_frac), seed $(split_seed): $(length(test_ids)) held-out of $(length(drug_models)) paired PDX for this drug. Predictive draws used (of $(Sp) / $(Sl) stored in the primary / late21 fit): $(counts.primary) for the held-out curve check, $(counts.late) for the late21 forecast, $(counts.resp) for the response section."
        path = write_report(drug, outdir; primary = ev.cm_new, late = ev.cm_late, rm = ev.rm, bins_primary = ev.bins_new, bins_late = ev.bins_late,
                            resid_primary = ev.resid_new, resid_late = ev.resid_late, tag,
                            loo_c, loo_m, pd_primary, pd_late, figs, notes, settings, unverified, ids = [u.model for u in units])
    end
    return (; ev, pd_primary, pd_late, unverified, path, ids = [u.model for u in units], n_holdout = length(test_ids),
            model_frac, seed = split_seed, stored = (primary = Sp, late = Sl))
end

"""
One shared split from labelled fits `[(label, pd), ...]`: every fit's declared `(model_frac, seed, holdout_models)` (when
recorded) must be identical (error otherwise). Returns `((model_frac, seed), declared::Bool)`; without any declaration the
defaults (0.2, 1) are used and each fit is then verified against them by `run_real`.
"""
function shared_split(labelled)
    decl = Tuple[]
    for (lab, pd) in labelled
        pd === nothing && continue
        sp = meta(pd, "split")
        sp isa AbstractDict && haskey(sp, "model_frac") && haskey(sp, "seed") &&
            push!(decl, (lab, float(sp["model_frac"]), Int(sp["seed"]), haskey(sp, "holdout_models") ? sort(String.(sp["holdout_models"])) : nothing))
    end
    for d in decl[2:end]
        (d[2] == decl[1][2] && d[3] == decl[1][3]) ||
            error("the fit sets use different splits: $(decl[1][1]) has (frac, seed) = ($(decl[1][2]), $(decl[1][3])) but $(d[1]) has ($(d[2]), $(d[3])); a comparison needs one shared split")
        (d[4] === nothing || decl[1][4] === nothing || d[4] == decl[1][4]) ||
            error("the fit sets declare different held-out PDX lists ($(decl[1][1]) vs $(d[1]))")
    end
    return isempty(decl) ? ((model_frac = 0.2, seed = 1), false) : ((model_frac = decl[1][2], seed = decl[1][3]), true)
end

"Same draw counts for both evaluations: the minimum of the requested and of every stored length (primary: curve check and response, late21)."
function shared_counts(fits1, fits2; ndraws, nresp)
    S(pd) = pd === nothing ? typemax(Int) : length(pd.σ)
    return (primary = min(ndraws, S(fits1[1]), S(fits2[1])), late = min(ndraws, S(fits1[2]), S(fits2[2])),
            resp = min(nresp, S(fits1[1]), S(fits2[1])))
end

"""
Comparison mode. ONE shared split is established first: every fit's declared `(model_frac, seed, holdout_models)` (when
recorded) must be identical across both fit sets (error otherwise); that split, recomputed from the frozen design, is then
passed to both evaluations, where both fit sets are verified against it. The draw counts are the minimum of the requested
and of every stored length, the same for both evaluations; the plan and the counts are reported in `compare_<drug>.md`.
"""
function run_compare(drug; tag, fitdir2, fitdir1 = joinpath(ROOT, "fits"), outdir = joinpath(ROOT, "docs"), loo = true,
                     ndraws = 200, nresp = 1000, suffix = "", suffix1 = "", kwargs...)
    fits1 = (load_if(fit_path(fitdir1, drug, "primary", suffix1)), load_if(fit_path(fitdir1, drug, "late21", suffix1)))
    fits2 = (load_if(fit_path(fitdir2, drug, "primary", suffix)), load_if(fit_path(fitdir2, drug, "late21", suffix)))
    (fits1[1] === nothing && fits1[2] === nothing) && error("no v1 fits found in $(fitdir1) (suffix \"$(suffix1)\")")
    (fits2[1] === nothing && fits2[2] === nothing) && error("no $(tag) fits found in $(fitdir2) (suffix \"$(suffix)\")")
    split, declared = shared_split([("v1 primary", fits1[1]), ("v1 late21", fits1[2]), ("$(tag) primary", fits2[1]), ("$(tag) late21", fits2[2])])
    counts = shared_counts(fits1, fits2; ndraws, nresp)
    common = (; counts, split, outdir, ndraws, nresp, kwargs...)
    r2 = run_real(drug; fitdir = fitdir2, tag, loo, write = true, fits = fits2, suffix, common...)
    r1 = run_real(drug; fitdir = fitdir1, tag = "", loo = false, write = false, fits = fits1, suffix = suffix1, common...)
    km1 = key_metrics(r1.ev, r1.pd_primary, r1.pd_late); km2 = key_metrics(r2.ev, r2.pd_primary, r2.pd_late)
    plan = (model_frac = split.model_frac, seed = split.seed, n_holdout = r2.n_holdout, declared = declared, counts,
            S1 = r1.stored, S2 = r2.stored)
    cp = write_compare(drug, outdir, km1, km2; tag1 = "v1", tag2 = tag, dir1 = fitdir1, dir2 = fitdir2,
                       unv1 = r1.unverified, unv2 = r2.unverified, ids = r2.ids, plan,
                       bins1 = vcat(r1.ev.bins_new, r1.ev.bins_late), bins2 = vcat(r2.ev.bins_new, r2.ev.bins_late))
    return (; v2_report = r2.path, compare = cp)
end

function run_selftest(; outdir = mktempdir())
    ensure_loaded()
    rng = MersenneTwister(11)
    doses = dosing_schedule(10.0; start = 0.0, interval = 1.0, n = 45)
    pk = OneCmt(CL = 20.0, V = 5.0, ka = 8.0, F = 0.8)
    cref = maximum(conc(pk, doses, 0:0.1:21))
    truth = (μ_λ0 = log(0.12), ω_λ0 = 0.3, μ_k2 = log(0.15), ω_k2 = 0.8, logλ1 = log(8.0), logk1 = log(0.4), σ = 0.15)
    units, tru = M.simulate_units(rng, 80, doses, pk, cref; truth, lod = 1.0, tspan = (30.0, 44.0))   # Student-t(5) noise
    S = 200; model_frac = 0.25; seed = 3; drug = "selftest"
    ids = [u.model for u in units]
    train_ids, test_ids = M.split_models(units; frac = model_frac, seed)
    tr_idx = [i for (i, m) in enumerate(ids) if m in train_ids]
    jit(x) = x .+ 0.01 .* randn(rng, S)                                  # a little spread so R-hat/ESS are defined
    function mk(sel_idx, window; extra = Dict{String,Any}())
        models = ids[sel_idx]
        diag = Dict{String,Any}("settings" => Dict{String,Any}("nchains" => 4, "lod" => 1.0, "nu" => 5.0),
                                "nu" => 5.0, "baseline_sd" => 0.1, "lod" => 1.0, "nchains" => 4, "fit_window_day" => window,
                                "max_observed_day" => 44.0, "median_followup_day" => 28.0, "draws_per_chain" => S ÷ 4,
                                "max_rhat" => 1.003, "min_ess_bulk" => 800.0, "n_divergences" => 0, "max_tree_depth_hits" => 0)
        merge!(diag, extra)
        # oracle baselines: each unit's observed day-0 volume with the fixed measurement error (draw-matched, as in the fits)
        v0c = [baseline_volume(units[i].control, 180.0) for i in sel_idx]; v0t = [baseline_volume(units[i].treated, 180.0) for i in sel_idx]
        wc = v0c' .* exp.(0.1 .* randn(rng, S, length(sel_idx))); wt = v0t' .* exp.(0.1 .* randn(rng, S, length(sel_idx)))
        M.PosteriorDraws(drug, models, doses, fill(pk, S), cref,
            jit(fill(truth.μ_λ0, S)), fill(truth.ω_λ0, S), jit(fill(truth.μ_k2, S)), fill(truth.ω_k2, S),
            jit(fill(truth.logλ1, S)), jit(fill(truth.logk1, S)), fill(truth.σ, S),
            repeat(tru.λ0[sel_idx]', S), repeat(tru.k2[sel_idx]', S),
            vec(median(wc; dims = 1)), vec(median(wt; dims = 1)), wc, wt, diag)
    end
    split_meta = Dict{String,Any}("model_frac" => model_frac, "seed" => seed, "holdout_models" => test_ids, "late_window_day" => LATE_WINDOW)
    # late21 design: cutoffs = last observed day <= 21, training PDX only
    cutoffs = Dict{String,Float64}()
    for u in units, (arm, c) in ((:treated, u.treated), (:control, u.control))
        u.model in train_ids && (cutoffs["$(u.model)|$(arm)"] = last_obs_le(c, LATE_WINDOW))
    end
    pd_late = mk(tr_idx, LATE_WINDOW; extra = Dict("split" => split_meta, "cutoffs" => cutoffs, "train_models" => train_ids))
    pd_primary = mk(tr_idx, PRIMARY_WINDOW; extra = Dict("split" => split_meta, "train_models" => train_ids))
    unv = vcat(verify_primary_fit(pd_primary, drug, train_ids, test_ids; model_frac, seed),
               verify_late21_fit(pd_late, drug, units, train_ids, test_ids; model_frac, seed))
    # evaluation of both fits (same code path as the real data)
    lod = M.detection_limit_from_fit(pd_late)
    counts = (primary = S, late = S, resp = S)
    ev = evaluate_fits(units, pd_primary, pd_late, train_ids, test_ids; counts, seed)
    recs_new, recs_late, cm_new, cm_late, rm = ev.recs_new, ev.recs_late, ev.cm_new, ev.cm_late, ev.rm
    bins_new, bins_late = ev.bins_new, ev.bins_late
    train_units = [u for u in units if u.model in train_ids]
    # the checks must reject a leaking fit, a wrong holdout list, a wrong window, a wrong split setting and a wrong cutoff
    errs = Dict{String,Any}()
    tryerr(f) = try; f(); nothing; catch e; e; end
    leak = mk(1:length(ids), PRIMARY_WINDOW; extra = Dict("train_models" => ids))
    errs["leak"] = tryerr(() -> verify_primary_fit(leak, drug, train_ids, test_ids; model_frac, seed))
    badsplit = mk(tr_idx, PRIMARY_WINDOW; extra = Dict("split" => merge(split_meta, Dict{String,Any}("holdout_models" => test_ids[1:end-1]))))
    errs["holdout list"] = tryerr(() -> verify_primary_fit(badsplit, drug, train_ids, test_ids; model_frac, seed))
    errs["window"] = tryerr(() -> verify_primary_fit(pd_late, drug, train_ids, test_ids; model_frac, seed))
    errs["frac"] = tryerr(() -> verify_primary_fit(pd_primary, drug, train_ids, test_ids; model_frac = 0.2, seed))
    bad_co = merge(cutoffs, Dict(first(keys(cutoffs)) => -1.0))
    errs["cutoff"] = tryerr(() -> verify_late21_fit(mk(tr_idx, LATE_WINDOW; extra = Dict("cutoffs" => bad_co, "split" => split_meta)),
                                                      drug, units, train_ids, test_ids; model_frac, seed))
    # the public-output guard must refuse small bins, PDX ids and individual-level figures
    small = binned_check(recs_new[1:6]; lod, edges = EDGES_PRIMARY)                 # no merging: a handful of PDX per bin
    errs["small bins"] = tryerr(() -> write_report("selftest_bad", mktempdir(); bins_primary = small, ids))
    errs["ids in text"] = tryerr(() -> write_report("selftest_bad", mktempdir(); notes = ["see PDX $(first(ids))"], ids))
    errs["individual figure"] = tryerr(() -> write_report("selftest_bad", mktempdir(); figs = ["docs/fig_x_bands.png"], ids))
    errs["absolute path"] = tryerr(() -> write_report("selftest_bad", mktempdir(); notes = ["fits in /Users/someone/fits"], ids))
    @assert public_path(joinpath(ROOT, "fits")) == "fits" && public_path("/private/tmp/elsewhere/fits") == "fits" "public_path must be repo-relative"
    @assert all(b -> b.n_pdx >= M.PUBLIC_MIN_PDX, vcat(bins_new, bins_late, ev.resid_new, ev.resid_late)) "merged bins must aggregate >= $(M.PUBLIC_MIN_PDX) PDX"
    # new response section content
    @assert rm.insample !== nothing && rm.insample.n_pdx >= M.PUBLIC_MIN_PDX && rm.n_draws == 200 "in-sample response / draw count"
    @assert isfinite(rm.benchmark.brier_model) && rm.benchmark.auroc_constant == 0.5
    @assert occursin("No demonstrated", auroc_statement(0.6, (0.4, 0.8), 3, 20)) && occursin("Only one held-out responder", auroc_statement(0.6, (0.4, 0.8), 1, 20))
    @assert occursin("excludes 0.5", auroc_statement(0.8, (0.6, 0.9), 8, 20)) && occursin("undefined", auroc_statement(NaN, (NaN, NaN), 0, 20))
    @assert !isempty(ev.resid_late) && all(r -> r.n_pdx >= M.PUBLIC_MIN_PDX, ev.resid_late) && all(b -> 0 <= b.tail_below5 <= 1, bins_new)
    # a v2 (+ waning) fit set under the `_v2_waning` suffix: file naming, verification and the whole evaluation path
    N = length(tr_idx)
    v2extra = Dict("model" => "v2", "corr_lk" => zeros(S), "mu_logVtr" => fill(truth.logλ1 - truth.μ_λ0, S), "omega_Vtr" => fill(0.3, S),
                   "lambda1_draws" => fill(exp(truth.logλ1), S, N), "waning" => true, "waning_rate" => fill(1e-4, S))
    v2dir = mktempdir()
    pd_p2 = mk(tr_idx, PRIMARY_WINDOW; extra = merge(v2extra, Dict("split" => split_meta, "train_models" => train_ids)))
    pd_l2 = mk(tr_idx, LATE_WINDOW; extra = merge(v2extra, Dict("split" => split_meta, "cutoffs" => cutoffs, "train_models" => train_ids)))
    M.save_fit(fit_path(v2dir, drug, "primary", "_v2_waning"), pd_p2); M.save_fit(fit_path(v2dir, drug, "late21", "_v2_waning"), pd_l2)
    @assert basename(fit_path(v2dir, drug, "primary", "_v2_waning")) == "selftest_primary_v2_waning.jld2"
    l2 = (load_if(fit_path(v2dir, drug, "primary", "_v2_waning")), load_if(fit_path(v2dir, drug, "late21", "_v2_waning")))
    @assert l2[1] !== nothing && l2[2] !== nothing && meta(l2[1], "model") == "v2" && Bool(meta(l2[1], "waning"))
    unv2 = vcat(verify_primary_fit(l2[1], drug, train_ids, test_ids; model_frac, seed), verify_late21_fit(l2[2], drug, units, train_ids, test_ids; model_frac, seed))
    @assert shared_split([("v1", pd_primary), ("v2", l2[1])]) == ((model_frac = model_frac, seed = seed), true)
    ev2 = evaluate_fits(units, l2[1], l2[2], train_ids, test_ids; counts, seed)
    @assert 0.75 < ev2.cm_late.coverage_pooled <= 1 && ev2.cm_new.n_pdx >= M.PUBLIC_MIN_PDX && ev2.rm.n_pdx >= M.PUBLIC_MIN_PDX "v2 evaluation"
    println("v2+waning oracle set (suffix _v2_waning): late21 coverage ", f3(ev2.cm_late.coverage_pooled), ", held-out coverage ", f3(ev2.cm_new.coverage_pooled),
            "; convergence rows: ", length(M.convergence_summary(l2[1]).recomputed.names))
    # compare mode: v1 vs v2 table passes the public guard and carries both columns
    km = key_metrics(ev, pd_primary, pd_late)
    plan = (model_frac = model_frac, seed = seed, n_holdout = length(test_ids), declared = true, counts,
            S1 = (primary = S, late = S), S2 = (primary = S, late = S))
    cpath = write_compare(drug, outdir, km, km; tag1 = "v1", tag2 = "v2", dir1 = "fits", dir2 = "fits_v2", unv1 = unv, unv2 = unv, ids, plan,
                          bins1 = vcat(bins_new, bins_late), bins2 = vcat(bins_new, bins_late))
    ctext = read(cpath, String)
    # contributor counts: undersized metrics are suppressed by key_metrics and refused by the guard if they get through
    small_ev = merge(ev, (cm_late = merge(ev.cm_late, (n_pdx = 3,)),))
    km_small = key_metrics(small_ev, pd_primary, pd_late)
    @assert !any(m -> startswith(m.name, "late21 90% coverage"), km_small) && any(m -> m.name == "held-out 90% coverage", km_small) "undersized metric must be suppressed"
    bad_km = vcat(km, [(name = "tiny metric", value = 1.0, n_pdx = 4)])
    errs["small metric in compare"] = tryerr(() -> write_compare(drug, mktempdir(), bad_km, bad_km; tag1 = "v1", tag2 = "v2", dir1 = "a", dir2 = "b", unv1 = String[], unv2 = String[], ids, plan))
    bad_bins = [(arm = :treated, lo = 21.0, hi = 28.0, n_pdx = 3)]
    errs["small bin in compare (v1 evaluation)"] = tryerr(() -> write_compare(drug, mktempdir(), km, km; tag1 = "v1", tag2 = "v2", dir1 = "a", dir2 = "b", unv1 = String[], unv2 = String[], ids, plan, bins1 = bad_bins))
    # shared split and sampling plan
    f1 = (pd_primary, pd_late)
    pd_other = mk(tr_idx, PRIMARY_WINDOW; extra = Dict("split" => merge(split_meta, Dict{String,Any}("seed" => seed + 1)), "train_models" => train_ids))
    pd_other_list = mk(tr_idx, PRIMARY_WINDOW; extra = Dict("split" => merge(split_meta, Dict{String,Any}("holdout_models" => test_ids[1:end-1])), "train_models" => train_ids))
    @assert shared_split([("a", pd_primary), ("b", pd_late)]) == ((model_frac = model_frac, seed = seed), true)
    errs["different split seed"] = tryerr(() -> shared_split([("v1", pd_primary), ("v2", pd_other)]))
    errs["different holdout list"] = tryerr(() -> shared_split([("v1", pd_primary), ("v2", pd_other_list)]))
    small_pd = mk(tr_idx, PRIMARY_WINDOW; extra = Dict("split" => split_meta))
    cut = M.PosteriorDraws(small_pd.drug, small_pd.models, small_pd.doses, small_pd.pk[1:120], small_pd.cref, small_pd.μ_λ0[1:120], small_pd.ω_λ0[1:120],
                           small_pd.μ_k2[1:120], small_pd.ω_k2[1:120], small_pd.logλ1[1:120], small_pd.logk1[1:120], small_pd.σ[1:120],
                           small_pd.λ0[1:120, :], small_pd.k2[1:120, :], small_pd.w0_control, small_pd.w0_treated,
                           small_pd.w0_control_draws[1:120, :], small_pd.w0_treated_draws[1:120, :], small_pd.diagnostics)
    @assert shared_counts((pd_primary, pd_late), (cut, pd_late); ndraws = 150, nresp = 1000) == (primary = 120, late = 150, resp = 120) "draw counts must be the minimum over both fit sets"
    @assert occursin("| v1 | v2 |", ctext) && occursin("held-out 90% coverage", ctext) && !any(id -> occursin(id, ctext), ids)
    @assert all(e -> e isa ErrorException, values(errs)) "verification failed to reject: $(join([k for (k, e) in errs if !(e isa ErrorException)], ", "))"
    # PSIS-LOO on the training PDX, conditional (windowed curves)
    fit_units = M.window_units(train_units, PRIMARY_WINDOW)
    loo_c = M.loo(M.grouped_loglik(pd_primary, fit_units); chain_index = M.chain_index_from_fit(pd_primary))
    figs, local_figs = make_figures("selftest", outdir, recs_new, cm_new, rm, bins_new)
    p = write_report("selftest", outdir; primary = cm_new, late = cm_late, rm, bins_primary = bins_new, bins_late,
                     resid_primary = ev.resid_new, resid_late = ev.resid_late,
                     loo_c, pd_primary, pd_late, figs, unverified = unv, ids,
                     notes = ["SELFTEST with an oracle posterior (true parameters, Student-t noise); pipeline check only."])
    println("selftest report: ", p)
    println("late21 forecast coverage (nominal 0.9): ", f3(cm_late.coverage_pooled), "  PDX-avg ", f3(cm_late.coverage_tumour_avg),
            "   held-out new-PDX coverage: ", f3(cm_new.coverage_pooled))
    println("new-PDX response: tau_b=", f3(rm.agreement.tau_b), " AUROC=", f3(rm.agreement.auroc), " modal agreement=", f3(rm.modal_agreement), " n=", rm.agreement.n,
            "; responder fraction obs ", pct(rm.obs_resp_frac), " pred recorded ", pct(rm.pred_resp_frac.mean), " latent ", pct(rm.pred_resp_frac_latent.mean),
            "; in-sample obs ", pct(rm.insample.obs_frac), " pred ", pct(rm.insample.noisy.mean), " latent ", pct(rm.insample.latent.mean))
    println("treated tails (below 5% / above 95%): new-PDX ", pct(cm_new.tails_treated.below5), " / ", pct(cm_new.tails_treated.above95),
            "; late21 ", pct(cm_late.tails_treated.below5), " / ", pct(cm_late.tails_treated.above95), "; response runtime ", round(rm.runtime_s; digits = 1), " s")
    println("verification rejected: ", join(["$k ($(first(e.msg, 36)))" for (k, e) in errs], " | "))
    println("public figures: ", join(basename.(figs), ", "), "  local: ", join(basename.(local_figs), ", "))
    return (; cm_late, cm_new, rm, path = p, compare = cpath)
end


function parse_args(args)
    pos = [a for a in args if !startswith(a, "--")]
    kw = Dict{String,String}()
    for a in args
        startswith(a, "--") && occursin("=", a) && (kv = split(a[3:end], "="; limit = 2); kw[kv[1]] = kv[2])
    end
    return pos, kw
end

if abspath(PROGRAM_FILE) == @__FILE__
    ensure_loaded()
    args = ARGS
    pos, kw = parse_args(args)
    if "--selftest" in args || isempty(args)
        run_selftest(outdir = get(ENV, "VALIDATE_SELFTEST_DIR", mktempdir()))
    else
        common = (; ndraws = parse(Int, get(kw, "ndraws", "200")), nresp = parse(Int, get(kw, "resp-draws", "1000")),
                  marginal_loo = "--marginal-loo" in args, loo = !("--no-loo" in args),
                  outdir = abspath(get(kw, "out", joinpath(ROOT, "docs"))))
        if haskey(kw, "fits-dir")
            tag = get(kw, "tag", "v2")
            r = run_compare(pos[1]; tag, fitdir2 = abspath(kw["fits-dir"]), suffix = get(kw, "suffix", ""),
                            suffix1 = get(kw, "suffix-v1", ""), fitdir1 = abspath(get(kw, "fits-dir-v1", joinpath(ROOT, "fits"))), common...)
            println("report: ", r.v2_report, "\ncomparison: ", r.compare)
        else
            r = run_real(pos[1]; tag = get(kw, "tag", ""), suffix = get(kw, "suffix", ""), common...)
            println("report: ", r.path)
        end
    end
end
