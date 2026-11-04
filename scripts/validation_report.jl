# Validation report: held-out PDX predictive checks, late-window forecast, response categories,
# convergence tables, PSIS-LOO.
#
#   julia --project=TumourPKPD.jl -t 2 scripts/validation_report.jl <drug> [--marginal-loo]
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
            n_pdx = length(unique(r.model for r in recs)), level, lod, nu,
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
Day bins merged so that each bin aggregates at least `min_pdx` distinct PDX models for the arm (greedy from the
left; a remainder joins the last bin). Returns the merged edges or `nothing` if even all bins together have fewer
than `min_pdx` PDX (the arm is then suppressed).
"""
function merged_edges(recs::Vector{CurveRecord}, arm, edges; min_pdx)
    nb = length(edges) - 1
    sets = [Set{String}() for _ in 1:nb]
    for r in recs
        r.arm === arm || continue
        for t in r.t
            b = findfirst(i -> edges[i] < t <= edges[i+1], 1:nb)
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
finite ensemble) and `n_pdx`, the number of distinct PDX models in the bin. With `min_pdx` the bins are merged
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
            yc = Float64[]; cols = Vector{Vector{Float64}}(); pdx = Set{String}()
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
                        push!(yc, r.y[k]); push!(cols, col)
                    end
                end
            end
            n == 0 && continue
            q(x) = isempty(x) ? (NaN, NaN, NaN) : (quantile(x, 0.05), quantile(x, 0.5), quantile(x, 0.95))
            push!(out, (arm = arm, lo = lo, hi = hi, n = n, n_pdx = length(pdx), obs_q = q(obs), pred_q = q(pred),
                        obs_below = obsbelow / n, pred_below = npred == 0 ? NaN : predbelow / npred,
                        coverage90 = coverage_ragged(yc, cols; level = 0.9)))
        end
    end
    return out
end

"Observed curve cut to `t <= window`."
cut_curve(c, window) = (m = c.t .<= window; (t = c.t[m], v = c.v[m]))

"""
Response metrics for held-out PDX, on the curves cut to days <= `window` (the primary fit window) for both the observed
classification and the predictions. Every model is simulated with the same `draws` (row s = one population draw across
models) and the same detection-limit recording as the data (simulated volumes below the LOD recorded as 0). Predicted
categories and the truth use the paper variant of the PDXE rules; the published category (`obs_cat`, model => category,
computed by Novartis on the full follow-up) is reported separately. AUROC and its bootstrap CI use P(responder);
Kendall τ_b uses the expected ordinal score.
"""
function response_metrics(pd, units; draws, window = PRIMARY_WINDOW, lod, seed = 7, obs_cat = nothing,
                          classifier = M.pdxe_response_paper)
    names = String[]; ocat = String[]; opub = String[]; pscore = Float64[]; pres = Float64[]; modal = String[]
    R = Vector{Vector{Union{Missing,Bool}}}()
    for u in units
        c = cut_curve(u.treated, window)
        (length(c.t) >= 2 && issorted(c.t) && c.t[1] == 0) || continue
        oc = classifier(c.t, c.v).category
        oc == "NA" && continue
        v0 = baseline_volume(u.treated, median(pd.w0_treated))
        w0 = M.latent_baseline(pd, v0, length(draws); rng = MersenneTwister(model_seed(seed, "$(u.model)|resp")))
        sims = M.simulate_regimen(pd, pd.doses; newpdx = true, tgrid = c.t, w0, draws,
                                  rng = MersenneTwister(model_seed(seed, u.model)))
        sims[:, 2:end] .= ifelse.(sims[:, 2:end] .< lod, 0.0, sims[:, 2:end])
        sims[:, 1] .= v0          # classify relative to the observed day-0 volume, like the observed category
        cats = M.response_categories(c.t, sims; classifier)
        pr = M.category_probs(cats)
        push!(names, u.model); push!(ocat, oc)
        push!(pscore, M.expected_score(cats)); push!(pres, M.p_responder(cats))
        push!(modal, M.RESPONSE_CATEGORIES[argmax([pr.mCR, pr.mPR, pr.mSD, pr.mPD])])
        push!(R, [x == "NA" ? missing : M.is_responder(x) for x in cats])
        push!(opub, obs_cat === nothing ? "NA" : get(obs_cat, u.model, "NA"))
    end
    isempty(names) && return nothing
    ag = M.ordinal_agreement(pscore, ocat; responder_scores = pres)
    pub = findall(!=("NA"), opub)
    ag_pub = isempty(pub) ? nothing : M.ordinal_agreement(pscore[pub], opub[pub]; responder_scores = pres[pub])
    Rm = Matrix{Union{Missing,Bool}}(reduce(hcat, R))              # ndraw × nmodel, same draw per row
    rf = M.responder_fraction(Rm)
    obs_rf = mean(M.is_responder.(ocat))
    cats4 = collect(M.RESPONSE_CATEGORIES)
    conf = [count((ocat .== a) .& (modal .== b)) for a in cats4, b in cats4]      # rows observed, cols predicted modal
    boot = M.cluster_bootstrap(names; nboot = 500, rng = MersenneTwister(seed)) do idx
        M.auroc(pres[idx], M.is_responder.(ocat[idx]))
    end
    bf = filter(isfinite, boot)
    auc_ci = isempty(bf) ? (NaN, NaN) : (quantile(bf, 0.025), quantile(bf, 0.975))
    return (names = names, obs_cat = ocat, obs_published = opub, pred_modal = modal,
            n_pdx = length(names), pred_score = pscore, p_resp = pres, agreement = ag, agreement_vs_published = ag_pub,
            auroc_ci = auc_ci, pred_resp_frac = rf, obs_resp_frac = obs_rf, confusion = conf,
            modal_agreement = mean(modal .== ocat), window = window,
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
            "Quantiles are 5% / 50% / 95%.\n")
    println(io, "| arm | days | n obs | observed log(V/V0) | predicted log(V/V0) | observed < LOD | predicted < LOD | 90% coverage |\n|---|---|---|---|---|---|---|---|")
    for b in bins
        q(t) = @sprintf("%.2f / %.2f / %.2f", t...)
        println(io, "| $(b.arm) | $(round(Int, b.lo))–$(round(Int, b.hi)) | $(b.n) | $(q(b.obs_q)) | $(q(b.pred_q)) | $(pct(b.obs_below)) | $(pct(b.pred_below)) | $(isnan(b.coverage90) ? "n/a" : pct(b.coverage90)) |")
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
Writes the PUBLIC report `outdir/validation_<drug>.md` (aggregated outputs only: every bin and summary aggregates
at least `min_pdx` PDX models, no PDX ids, no individual observations). Summaries with fewer PDX are suppressed
and said so in the text; `assert_public_output` then refuses to write if any bin/summary under `min_pdx`, any id in
`ids` or any individual-level figure is still present.
"""
function write_report(drug, outdir; primary = nothing, late = nothing, rm = nothing, bins_primary = NamedTuple[],
                      bins_late = NamedTuple[], loo_c = nothing, loo_m = nothing, pd_primary = nothing, pd_late = nothing,
                      figs = String[], notes = String[], settings = "", unverified = String[],
                      ids = String[], min_pdx = M.PUBLIC_MIN_PDX)
    mkpath(outdir)
    path = joinpath(outdir, "validation_$(drug).md")
    notes = copy(notes)
    sup(x, what) = (x !== nothing && x.n_pdx < min_pdx) ? (push!(notes, "$(what) suppressed: it aggregates only $(x.n_pdx) PDX models (< $(min_pdx))."); nothing) : x
    primary = sup(primary, "New-PDX summary"); late = sup(late, "Late-21 summary"); rm = sup(rm, "Response summary")
    text = let io = IOBuffer()
        println(io, "# Validation: $(drug)\n")
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
        println(io, "### Response categories on held-out PDX (PDXE rules, Gao et al. 2015)\n")
        if rm === nothing
            println(io, "Not available.\n")
        else
            println(io, "Observed and predicted categories use `pdxe_response_paper` (windows t > 10, short curves on all points) on the days ≤ $(round(Int, rm.window)) record; ",
                    "simulated volumes below the LOD are recorded as 0 like the data. This rule reproduces the published BestAvgResponse in 98.4% of PDXE curves but its BestResponse is ",
                    "systematically lower than the published one in ~40% of curves (published mPD scored mSD by the rule): 4-way agreement with the published labels is ~91%, ",
                    "responder vs non-responder ~99.9%.\n")
            println(io, "Observed category counts (mCR/mPR/mSD/mPD): ", join(rm.cat_counts, "/"), " over ", length(rm.names), " held-out PDX.\n")
            println(io, "| metric | value |\n|---|---|")
            println(io, "| Kendall τ_b (predicted expected ordinal score vs observed ordinal) | $(f3(rm.agreement.tau_b)) |")
            println(io, "| AUROC responder (mCR/mPR) vs rest, ranked by predicted P(responder) | $(f3(rm.agreement.auroc)) (same score; cluster-bootstrap 95% CI $(f3(rm.auroc_ci[1]))–$(f3(rm.auroc_ci[2]))) |")
            println(io, "| modal predicted category = observed category | $(pct(rm.modal_agreement)) |")
            rm.agreement_vs_published === nothing || println(io,
                "| vs the published (full follow-up) categories: τ_b / AUROC | $(f3(rm.agreement_vs_published.tau_b)) / $(f3(rm.agreement_vs_published.auroc)) (n=$(rm.agreement_vs_published.n)) |")
            println(io, "| predicted responder fraction (common posterior draws across models) | $(pct(rm.pred_resp_frac.mean)) ($(pct(rm.pred_resp_frac.lo))–$(pct(rm.pred_resp_frac.hi))); $(rm.pred_resp_frac.n_missing) failed model-draws excluded |")
            println(io, "| observed responder fraction | $(pct(rm.obs_resp_frac)) |\n")
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
    loo_n = [(n_pdx = l.n,) for l in (loo_c, loo_m) if l !== nothing]
    M.assert_public_output(; bins = vcat(bins_primary, bins_late), summaries = vcat(summaries, loo_n), text, ids, figures = figs, min_pdx)
    write(path, text)
    return path
end


# ---------------------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------------------

load_if(path) = isfile(path) ? M.load_fit(path) : nothing

primary_select(c) = (c.t .> 0) .& (c.t .<= PRIMARY_WINDOW)
late_select(c) = (c.t .> LATE_WINDOW) .& (c.t .<= PRIMARY_WINDOW)
const EDGES_PRIMARY = [0.0, 7, 14, 21, 28, 35, 42]
const EDGES_LATE = [21.0, 28, 35, 42]

function run_real(drug; ndraws = 200, seed = 1, marginal_loo = false,
                  fitdir = joinpath(ROOT, "fits"), outdir = joinpath(ROOT, "docs"))
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
    pd_primary = load_if(joinpath(fitdir, "$(drug)_primary.jld2"))
    pd_late = load_if(joinpath(fitdir, "$(drug)_late21.jld2"))
    # split settings come from the fit (frac, seed); the frozen split is recomputed and compared
    sp(pd) = pd === nothing ? nothing : meta(pd, "split")
    getsp(k, default) = (for pd in (pd_primary, pd_late); d = sp(pd); d isa AbstractDict && haskey(d, k) && return d[k]; end; default)
    model_frac = float(getsp("model_frac", 0.2)); split_seed = Int(getsp("seed", 1))
    train_all, test_all, _ = M.frozen_pdx_split(; frac = model_frac, seed = split_seed)
    train_ids, test_ids = expected_split(drug_models; frac = model_frac, seed = split_seed, test_all)
    recs_new = CurveRecord[]; recs_late = CurveRecord[]
    cm_new = cm_late = rm = nothing; bins_new = NamedTuple[]; bins_late = NamedTuple[]
    if pd_primary === nothing
        push!(notes, "No fits/$(drug)_primary.jld2: held-out PDX prediction, response metrics and LOO skipped.")
    else
        append!(unverified, verify_primary_fit(pd_primary, drug, train_ids, test_ids; model_frac, seed = split_seed))
        test_units = [u for u in units if u.model in test_ids]
        draws = common_draws(pd_primary, ndraws, seed)
        lod = M.detection_limit_from_fit(pd_primary); nu = M.nu_from_fit(pd_primary)
        recs_new = predict_units(pd_primary, test_units; newpdx = true, select = primary_select, draws, seed)
        cm_new = curve_metrics(recs_new; lod, nu)
        bins_new = binned_check(recs_new; lod, edges = EDGES_PRIMARY, min_pdx = M.PUBLIC_MIN_PDX)
        rm = response_metrics(pd_primary, test_units; draws, lod, seed, obs_cat)
    end
    if pd_late === nothing
        push!(notes, "No fits/$(drug)_late21.jld2: late-21 forecast skipped.")
    else
        append!(unverified, verify_late21_fit(pd_late, drug, units, train_ids, test_ids; model_frac, seed = split_seed))
        draws = common_draws(pd_late, ndraws, seed)
        lod = M.detection_limit_from_fit(pd_late); nu = M.nu_from_fit(pd_late)
        recs_late = predict_units(pd_late, [u for u in units if u.model in pd_late.models]; newpdx = false,
                                  select = late_select, draws, seed)
        cm_late = curve_metrics(recs_late; lod, nu)
        bins_late = binned_check(recs_late; lod, edges = EDGES_LATE, min_pdx = M.PUBLIC_MIN_PDX)
    end
    # PSIS-LOO on the primary fit's training PDX, on the windowed curves the fit saw
    loo_c = loo_m = nothing
    if pd_primary !== nothing
        ci = M.chain_index_from_fit(pd_primary)
        fit_units = M.window_units([u for u in units if u.model in pd_primary.models], PRIMARY_WINDOW)
        loo_c = M.loo(M.grouped_loglik(pd_primary, fit_units); chain_index = ci)
        marginal_loo && (loo_m = M.loo(M.marginal_unit_loglik(pd_primary, fit_units); chain_index = ci))
    end
    figs, local_figs = make_figures(drug, outdir, recs_new, cm_new, rm, bins_new)
    settings = "Frozen split frac $(model_frac), seed $(split_seed): $(length(test_ids)) held-out of $(length(drug_models)) paired PDX for this drug; $(ndraws) common predictive draws."
    return write_report(drug, outdir; primary = cm_new, late = cm_late, rm, bins_primary = bins_new, bins_late,
                        loo_c, loo_m, pd_primary, pd_late, figs, notes, settings, unverified,
                        ids = [u.model for u in units])
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
    # known-tumour forecast of days 22-42 (training PDX, oracle per-unit parameters)
    lod = M.detection_limit_from_fit(pd_late); nu = M.nu_from_fit(pd_late)
    draws = common_draws(pd_late, S, seed)
    train_units = [u for u in units if u.model in train_ids]
    recs_late = predict_units(pd_late, train_units; newpdx = false, select = late_select, draws, seed)
    cm_late = curve_metrics(recs_late; lod, nu)
    bins_late = binned_check(recs_late; lod, edges = EDGES_LATE, min_pdx = M.PUBLIC_MIN_PDX)
    # held-out PDX, new-PDX prediction from the primary fit
    tu = [u for u in units if u.model in test_ids]
    draws = common_draws(pd_primary, S, seed)
    recs_new = predict_units(pd_primary, tu; newpdx = true, select = primary_select, draws, seed)
    cm_new = curve_metrics(recs_new; lod, nu)
    bins_new = binned_check(recs_new; lod, edges = EDGES_PRIMARY, min_pdx = M.PUBLIC_MIN_PDX)
    rm = response_metrics(pd_primary, tu; draws, lod, seed)
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
    pub_bins = binned_check(recs_new; lod, edges = EDGES_PRIMARY, min_pdx = M.PUBLIC_MIN_PDX)
    @assert all(b -> b.n_pdx >= M.PUBLIC_MIN_PDX, vcat(pub_bins, bins_late)) "merged bins must aggregate >= $(M.PUBLIC_MIN_PDX) PDX"
    @assert all(e -> e isa ErrorException, values(errs)) "verification failed to reject: $(join([k for (k, e) in errs if !(e isa ErrorException)], ", "))"
    # PSIS-LOO on the training PDX, conditional (windowed curves)
    fit_units = M.window_units(train_units, PRIMARY_WINDOW)
    loo_c = M.loo(M.grouped_loglik(pd_primary, fit_units); chain_index = M.chain_index_from_fit(pd_primary))
    figs, local_figs = make_figures("selftest", outdir, recs_new, cm_new, rm, bins_new)
    p = write_report("selftest", outdir; primary = cm_new, late = cm_late, rm, bins_primary = bins_new, bins_late,
                     loo_c, pd_primary, pd_late, figs, unverified = unv, ids,
                     notes = ["SELFTEST with an oracle posterior (true parameters, Student-t noise); pipeline check only."])
    println("selftest report: ", p)
    println("late21 forecast coverage (nominal 0.9): ", f3(cm_late.coverage_pooled), "  PDX-avg ", f3(cm_late.coverage_tumour_avg),
            "   held-out new-PDX coverage: ", f3(cm_new.coverage_pooled))
    println("new-PDX response: tau_b=", f3(rm.agreement.tau_b), " AUROC=", f3(rm.agreement.auroc), " modal agreement=", f3(rm.modal_agreement), " n=", rm.agreement.n)
    println("verification rejected: ", join(["$k ($(first(e.msg, 36)))" for (k, e) in errs], " | "))
    println("public figures: ", join(basename.(figs), ", "), "  local: ", join(basename.(local_figs), ", "))
    return (; cm_late, cm_new, rm, path = p)
end

if abspath(PROGRAM_FILE) == @__FILE__
    ensure_loaded()
    args = ARGS
    if "--selftest" in args || isempty(args)
        run_selftest(outdir = get(ENV, "VALIDATE_SELFTEST_DIR", mktempdir()))
    else
        run_real(args[1]; marginal_loo = "--marginal-loo" in args)
    end
end
