# Validation report: held-out predictive checks, response-category ranking, PSIS-LOO.
#
#   julia --project=TumourPKPD.jl -t 2 scripts/validation_report.jl <drug> [options]
#   julia --project=TumourPKPD.jl -t 2 scripts/validation_report.jl --selftest
#
# Reads (all under fits/, each a `save_fit` JLD2 holding a PosteriorDraws):
#   <drug>.jld2          fit on all PDX models, all points   -> PSIS-LOO, descriptive in-sample checks
#   <drug>_late.jld2     fit on early points only (split_late of every curve, frac)   -> known-tumour forecast
#   <drug>_holdout.jld2  fit without the held-out PDX models (split_models, frac/seed) -> new-PDX prediction
# Missing fits are reported as "not available" and skipped. The split settings must match the
# ones used when the hold-out fits were made (defaults: late frac 0.3; models frac 0.2, seed 1).
# Writes docs/validation_<drug>.md and docs/fig_<drug>_*.png.
#
# --selftest builds synthetic PDX units and an "oracle" posterior (true parameters, no
# parameter uncertainty) to exercise the whole pipeline; not a scientific result.

using Random, Statistics, Printf, Dates
using TumourPKPD
const M = TumourPKPD
const ROOT = normpath(joinpath(@__DIR__, ".."))

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
# Metadata the inference side saves in PosteriorDraws.diagnostics (or diagnostics["settings"]):
#   "train_models" :: Vector{String}   PDX ids the fit was trained on (hold-out fit; must equal pd.models)
#   "split"        :: Dict             "late_frac" (late fit) and/or "model_frac", "seed" (hold-out fit)
#   "cutoffs"      :: Dict{String,Real} last training time per curve, key "<model>|treated" / "<model>|control"
#                                      (or just "<model>"); late fit only
#   "nchains", "lod"                   as written by fit_drug
# Absent keys are reported in the document as "unverified"; present keys that disagree are errors.

meta(pd, k, default = nothing) = M._fit_meta(pd, k, default)

function check_drug(pd, drug)
    pd.drug == drug || error("fit is for drug $(repr(pd.drug)), expected $(repr(drug))")
end

function verify_holdout_fit(pd, drug, test_ids, expected_train; model_frac, seed)
    check_drug(pd, drug)
    leak = intersect(test_ids, pd.models)
    isempty(leak) || error("held-out PDX models are in the fit's training models: $(first(leak, 5))")
    extra = setdiff(pd.models, expected_train)
    isempty(extra) || error("fit contains models outside split_models' training set (frac=$(model_frac), seed=$(seed)): $(first(extra, 5))")
    unverified = String[]
    tm = meta(pd, "train_models")
    if tm === nothing
        push!(unverified, "train_models")
    else
        sort(String.(tm)) == sort(pd.models) || error("saved train_models differ from the fit's models")
        isempty(intersect(test_ids, String.(tm))) || error("held-out PDX models are in saved train_models")
    end
    sp = meta(pd, "split")
    if sp isa AbstractDict
        haskey(sp, "model_frac") && !isapprox(sp["model_frac"], model_frac) &&
            error("fit used model_frac=$(sp["model_frac"]), report uses $(model_frac)")
        haskey(sp, "seed") && sp["seed"] != seed && error("fit used split seed=$(sp["seed"]), report uses $(seed)")
        (haskey(sp, "model_frac") && haskey(sp, "seed")) || push!(unverified, "split(model_frac, seed)")
    else
        push!(unverified, "split")
    end
    return unverified
end

last_train_time(c, late_frac) = begin
    ord = sortperm(c.t)
    tr, _ = M.split_late_indices(length(ord); frac = late_frac)
    isempty(tr) ? NaN : c.t[ord[tr[end]]]
end

function verify_late_fit(pd, drug, units, late_frac)
    check_drug(pd, drug)
    unverified = String[]
    sp = meta(pd, "split")
    if sp isa AbstractDict && haskey(sp, "late_frac")
        isapprox(sp["late_frac"], late_frac) || error("fit used late_frac=$(sp["late_frac"]), report uses $(late_frac)")
    else
        push!(unverified, "split(late_frac)")
    end
    co = meta(pd, "cutoffs")
    if co isa AbstractDict
        for u in units
            u.model in pd.models || continue
            for (arm, c) in ((:treated, u.treated), (:control, u.control))
                exp_t = last_train_time(c, late_frac)
                got = get(co, "$(u.model)|$(arm)", get(co, u.model, nothing))
                got === nothing && error("no saved cutoff for $(u.model)|$(arm)")
                abs(got - exp_t) < 1e-9 || error("cutoff mismatch for $(u.model)|$(arm): fit $(got), report $(exp_t)")
            end
        end
    else
        push!(unverified, "cutoffs")
    end
    return unverified
end

# ---------------------------------------------------------------------------------------
# Prediction records
# ---------------------------------------------------------------------------------------

"One predicted curve: observations, noisy draws, latent (noise-free) draws, per-draw σ."
struct CurveRecord
    model::String
    tumour_type::String
    arm::Symbol
    t::Vector{Float64}
    y::Vector{Float64}
    sims::Matrix{Float64}
    latent::Matrix{Float64}
    σ::Vector{Float64}
end

model_seed(seed, model) = Int(hash((seed, String(model))) % 2^31)

# Noisy and latent paths use the same rng seed: the (new-PDX) random effects are drawn before the
# observation noise, so both calls share random effects; the seed depends on the PDX only, so its
# control and treated arms share them too. `draws` are common to every call.
function predict_curve(pd, model, tumour_type, arm::Symbol, t, y, w0; newpdx, draws, seed)
    simf = arm === :control ? (; kw...) -> M.simulate_control(pd; kw...) :
                              (; kw...) -> M.simulate_regimen(pd, pd.doses; kw...)
    kw = (; tgrid = t, w0, draws, details = true, (newpdx ? (; newpdx = true) : (; model = model))...)
    noisy = simf(; kw..., obs_noise = true, rng = MersenneTwister(model_seed(seed, model)))
    lat = simf(; kw..., obs_noise = false, rng = MersenneTwister(model_seed(seed, model)))
    return CurveRecord(model, tumour_type, arm, collect(Float64, t), collect(Float64, y),
                       noisy.y, lat.y, lat.σ)
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
            w0 = baseline_volume(c, arm === :treated ? med_t : med_c)
            push!(recs, predict_curve(pd, u.model, u.tumour_type, arm,
                                      c.t[mask], c.v[mask], w0; newpdx, draws, seed))
        end
    end
    return recs
end

# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------

mean_finite(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))

function curve_metrics(recs::Vector{CurveRecord}; level = 0.9, lod, nboot = 500, seed = 1)
    isempty(recs) && return nothing
    pos(r) = r.y .> 0
    y = reduce(vcat, [r.y[pos(r)] for r in recs])
    S = reduce(hcat, [r.sims[:, pos(r)] for r in recs])
    pdx = reduce(vcat, [fill(r.model, count(pos(r))) for r in recs])      # tumour = PDX identity
    cg = M.coverage_by_group(y, S, pdx; level)
    fail = M.ensemble_failures(S)
    wlog = M.interval_width(S; level, transform = log)
    lpd = reduce(vcat, [M.log_predictive_density(r.y, r.latent, r.σ; limit = lod) for r in recs])
    crps = M.crps(y, S)
    ncens = sum(count(<=(0), r.y) for r in recs)
    cov_boot = M.cluster_bootstrap(idx -> M.coverage(y[idx], S[:, idx]; level), pdx;
                                   nboot, rng = MersenneTwister(seed))
    return (n_points = length(y), n_censored = ncens, n_curves = length(recs),
            n_pdx = length(unique(r.model for r in recs)), level, lod,
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
Response metrics for held-out PDX. Every model is simulated with the same `draws`, so row s of the
responder matrix is one population draw across models. Predicted categories use the paper variant
of the PDXE rules. AUROC and its bootstrap CI both use P(responder); Kendall τ_b uses the expected
ordinal score. `obs_cat` (model => published category) is the primary truth when given.
"""
function response_metrics(pd, units; draws, seed = 7, obs_cat = nothing, classifier = M.pdxe_response_paper)
    names = String[]; ocat = String[]; opub = String[]; pscore = Float64[]; pres = Float64[]
    R = Vector{Vector{Union{Missing,Bool}}}()
    for u in units
        c = u.treated
        (issorted(c.t) && c.t[1] == 0) || continue
        oc = classifier(c.t, c.v).category
        oc == "NA" && continue
        w0 = baseline_volume(c, median(pd.w0_treated))
        sims = M.simulate_regimen(pd, pd.doses; newpdx = true, tgrid = c.t, w0, draws,
                                  rng = MersenneTwister(model_seed(seed, u.model)))
        cats = M.response_categories(c.t, sims; classifier)
        push!(names, u.model); push!(ocat, oc)
        push!(pscore, M.expected_score(cats)); push!(pres, M.p_responder(cats))
        push!(R, [x == "NA" ? missing : M.is_responder(x) for x in cats])
        push!(opub, obs_cat === nothing ? "NA" : get(obs_cat, u.model, "NA"))
    end
    isempty(names) && return nothing
    truth = any(!=("NA"), opub) ? [o == "NA" ? c : o for (o, c) in zip(opub, ocat)] : ocat
    ag = M.ordinal_agreement(pscore, truth; responder_scores = pres)
    ag_recomputed = M.ordinal_agreement(pscore, ocat; responder_scores = pres)
    Rm = Matrix{Union{Missing,Bool}}(reduce(hcat, R))              # ndraw × nmodel, same draw per row
    rf = M.responder_fraction(Rm)
    obs_rf = mean(M.is_responder.(truth))
    boot = M.cluster_bootstrap(names; nboot = 500, rng = MersenneTwister(seed)) do idx
        M.auroc(pres[idx], M.is_responder.(truth[idx]))
    end
    bf = filter(isfinite, boot)
    auc_ci = isempty(bf) ? (NaN, NaN) : (quantile(bf, 0.025), quantile(bf, 0.975))
    return (names = names, obs_cat = truth, obs_cat_recomputed = ocat, truth_is_published = any(!=("NA"), opub),
            pred_score = pscore, p_resp = pres, agreement = ag, agreement_recomputed = ag_recomputed,
            auroc_ci = auc_ci, pred_resp_frac = rf, obs_resp_frac = obs_rf,
            cat_counts = [count(==(c), truth) for c in M.RESPONSE_CATEGORIES])
end

# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------

function make_figures(drug, outdir, recs, cm, rm)
    paths = String[]
    mkpath(outdir)
    isempty(recs) || begin
        using_cairo()
        CM = Base.invokelatest(() -> getfield(Main, :CairoMakie))
        Base.invokelatest(figure_bands, CM, drug, outdir, recs, paths)
        cm === nothing || Base.invokelatest(figure_calibration, CM, drug, outdir, cm, paths)
    end
    rm === nothing || begin
        using_cairo()
        CM = Base.invokelatest(() -> getfield(Main, :CairoMakie))
        Base.invokelatest(figure_response, CM, drug, outdir, rm, paths)
    end
    return paths
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

function figure_calibration(CM, drug, outdir, cm, paths)
    fig = CM.Figure(size = (420, 420))
    ax = CM.Axis(fig[1, 1], xlabel = "nominal level", ylabel = "empirical coverage",
                 title = "Held-out calibration ($(drug))", limits = (0, 1, 0, 1))
    CM.lines!(ax, [0, 1], [0, 1], color = :gray, linestyle = :dash)
    CM.scatterlines!(ax, first.(cm.calib), last.(cm.calib), color = :steelblue)
    p = joinpath(outdir, "fig_$(drug)_calibration.png"); CM.save(p, fig); push!(paths, p)
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

function metrics_table(io, name, cm)
    cm === nothing && return println(io, "$name: not available (no fit file / no held-out points).\n")
    println(io, "**$name**: $(cm.n_points) scored points ($(cm.n_censored) undetectable, scored as left-censored at ",
            "LOD $(cm.lod) mm³ in the LPD only), $(cm.n_curves) curves, $(cm.n_pdx) PDX models; nominal level $(pct(cm.level)).\n")
    println(io, "| metric | value |\n|---|---|")
    println(io, "| coverage, pooled | $(pct(cm.coverage_pooled)) (PDX-cluster bootstrap 95% CI $(pct(cm.coverage_pdx_ci[1]))–$(pct(cm.coverage_pdx_ci[2]))) |")
    println(io, "| coverage, PDX-averaged (each PDX weighted equally) | $(pct(cm.coverage_tumour_avg)) |")
    println(io, "| mean interval width (log-volume) | $(f3(cm.width_log_mean)) (NaN for $(cm.n_width_nan) points) |")
    println(io, "| mean log predictive density, finite scores (volume scale) | $(f3(cm.lpd_mean_finite)) over $(cm.lpd_n - cm.n_lpd_neginf) points |")
    println(io, "| LPD scores equal to -Inf (no valid draw), kept | $(cm.n_lpd_neginf); mean over all scored points incl. these: $(cm.lpd_mean_all) |")
    println(io, "| mean CRPS (mm³) | $(f3(cm.crps_mean)) (NaN for $(cm.n_crps_nan) points) |")
    println(io, "| failed solves | $(cm.n_failed_draw_rows) draw rows with non-finite values; $(cm.n_empty_ensembles) empty ensembles (counted as coverage misses) |\n")
end

function write_report(drug, outdir; late = nothing, newpdx = nothing, rm = nothing, loo_c = nothing,
                      loo_m = nothing, figs = String[], notes = String[], settings = "", unverified = String[])
    mkpath(outdir)
    path = joinpath(outdir, "validation_$(drug).md")
    open(path, "w") do io
        println(io, "# Validation: $(drug)\n")
        println(io, "Generated $(Dates.format(now(), "yyyy-mm-dd HH:MM")). $(settings)\n")
        for n in notes; println(io, "> ", n); end
        isempty(unverified) || println(io, "> Fit metadata not recorded, so these were NOT verified: ", join(unverified, ", "), ".")
        println(io, "\n## Forecasting known tumours (late timepoints held out)\n")
        metrics_table(io, "Known-tumour forecast", late)
        println(io, "## New PDX prediction (whole models held out, fresh random effects)\n")
        metrics_table(io, "New-PDX prediction", newpdx)
        println(io, "## Response categories on held-out PDX (PDXE rules, Gao et al. 2015)\n")
        if rm === nothing
            println(io, "Not available.\n")
        else
            println(io, "Predicted categories use `pdxe_response_paper` (windows t > 10, short curves on all points); ",
                    "this reproduces the published BestAvgResponse in 98.4% of PDXE curves but its BestResponse is ",
                    "systematically lower than the published one in ~40% of curves (published mPD scored mSD by the rule), ",
                    "so 4-way category agreement with the published labels is ~91%, responder vs non-responder ~99.9%.\n")
            println(io, "Truth labels: ", rm.truth_is_published ? "published PDXE categories (recomputed where missing)" : "recomputed from the raw curves",
                    ". Observed category counts (mCR/mPR/mSD/mPD): ", join(rm.cat_counts, "/"), " over ", length(rm.names), " models.\n")
            println(io, "| metric | value |\n|---|---|")
            println(io, "| Kendall τ_b (predicted expected ordinal score vs observed ordinal) | $(f3(rm.agreement.tau_b)) |")
            println(io, "| AUROC responder (mCR/mPR) vs rest, ranked by predicted P(responder) | $(f3(rm.agreement.auroc)) (same score; cluster-bootstrap 95% CI $(f3(rm.auroc_ci[1]))–$(f3(rm.auroc_ci[2]))) |")
            rm.truth_is_published && println(io,
                "| same, truth = recomputed categories | τ_b $(f3(rm.agreement_recomputed.tau_b)), AUROC $(f3(rm.agreement_recomputed.auroc)) |")
            println(io, "| predicted responder fraction (common posterior draws across models) | $(pct(rm.pred_resp_frac.mean)) ($(pct(rm.pred_resp_frac.lo))–$(pct(rm.pred_resp_frac.hi))); $(rm.pred_resp_frac.n_missing) failed model-draws excluded |")
            println(io, "| observed responder fraction | $(pct(rm.obs_resp_frac)) |\n")
        end
        println(io, "## PSIS-LOO\n")
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
    end
    return path
end

# ---------------------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------------------

load_if(path) = isfile(path) ? M.load_fit(path) : nothing

function run_real(drug; late_frac = 0.3, model_frac = 0.2, seed = 1, ndraws = 200, marginal_loo = false,
                  fitdir = joinpath(ROOT, "fits"), outdir = joinpath(ROOT, "docs"))
    isdefined(M, :drug_dataset) || error("data.jl (drug_dataset, load_metrics) is not included in TumourPKPD yet")
    units = M.drug_dataset(drug)
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
    # known-tumour forecast: score the held-out tail of each curve under the *_late fit
    pd_late = load_if(joinpath(fitdir, "$(drug)_late.jld2"))
    cm_late = nothing; recs_late = CurveRecord[]
    if pd_late === nothing
        push!(notes, "No fits/$(drug)_late.jld2: known-tumour forecast skipped.")
    else
        append!(unverified, "late: " .* verify_late_fit(pd_late, drug, units, late_frac))
        sel = c -> begin
            _, te = M.split_late_indices(length(c.t); frac = late_frac)
            m = falses(length(c.t)); m[sortperm(c.t)[te]] .= true; m
        end
        draws = common_draws(pd_late, ndraws, seed)
        recs_late = predict_units(pd_late, [u for u in units if u.model in pd_late.models]; newpdx = false,
                                  select = sel, draws, seed)
        cm_late = curve_metrics(recs_late; lod = M.detection_limit_from_fit(pd_late))
    end
    # new-PDX: held-out models under the *_holdout fit
    pd_ho = load_if(joinpath(fitdir, "$(drug)_holdout.jld2"))
    cm_new = nothing; rm = nothing; recs_new = CurveRecord[]
    if pd_ho === nothing
        push!(notes, "No fits/$(drug)_holdout.jld2: new-PDX prediction skipped.")
    else
        train_ids, test_ids = M.split_models(units; frac = model_frac, seed)
        append!(unverified, "holdout: " .* verify_holdout_fit(pd_ho, drug, test_ids, train_ids; model_frac, seed))
        test_units = [u for u in units if u.model in test_ids]
        draws = common_draws(pd_ho, ndraws, seed)
        recs_new = predict_units(pd_ho, test_units; newpdx = true, select = c -> c.t .> 0, draws, seed)
        cm_new = curve_metrics(recs_new; lod = M.detection_limit_from_fit(pd_ho))
        rm = response_metrics(pd_ho, test_units; draws, seed, obs_cat)
    end
    # PSIS-LOO on the full fit (chain labels from the saved nchains)
    pd_full = load_if(joinpath(fitdir, "$(drug).jld2"))
    loo_c = loo_m = nothing
    if pd_full === nothing
        push!(notes, "No fits/$(drug).jld2: PSIS-LOO skipped.")
    else
        check_drug(pd_full, drug)
        ci = M.chain_index_from_fit(pd_full)
        loo_c = M.loo(M.grouped_loglik(pd_full, units); chain_index = ci)
        marginal_loo && (loo_m = M.loo(M.marginal_unit_loglik(pd_full, units); chain_index = ci))
    end
    figs = make_figures(drug, outdir, isempty(recs_new) ? recs_late : recs_new, cm_new === nothing ? cm_late : cm_new, rm)
    settings = "Late-split frac $(late_frac); model split frac $(model_frac), seed $(seed); $(ndraws) common predictive draws."
    return write_report(drug, outdir; late = cm_late, newpdx = cm_new, rm, loo_c, loo_m, figs, notes, settings, unverified)
end

function run_selftest(; outdir = mktempdir())
    ensure_loaded()
    rng = MersenneTwister(11)
    doses = dosing_schedule(10.0; start = 0.0, interval = 1.0, n = 21)
    pk = OneCmt(CL = 20.0, V = 5.0, ka = 8.0, F = 0.8)
    cref = maximum(conc(pk, doses, 0:0.1:21))
    truth = (μ_λ0 = log(0.12), ω_λ0 = 0.3, μ_k2 = log(0.5), ω_k2 = 0.5, logλ1 = log(8.0), logk1 = log(0.4), σ = 0.15)
    units, tru = M.simulate_units(rng, 80, doses, pk, cref; truth, lod = 1.0)
    S = 200; late_frac = 0.3; model_frac = 0.25; seed = 3; drug = "selftest"
    ids = [u.model for u in units]
    function mk(sel_idx; extra = Dict{String,Any}())
        models = ids[sel_idx]
        diag = Dict{String,Any}("settings" => Dict{String,Any}("nchains" => 4, "lod" => 1.0), merge(Dict{String,Any}(), extra)...)
        M.PosteriorDraws(drug, models, doses, fill(pk, S), cref,
            fill(truth.μ_λ0, S), fill(truth.ω_λ0, S), fill(truth.μ_k2, S), fill(truth.ω_k2, S),
            fill(truth.logλ1, S), fill(truth.logk1, S), fill(truth.σ, S),
            repeat(tru.λ0[sel_idx]', S), repeat(tru.k2[sel_idx]', S),
            fill(180.0, length(models)), fill(180.0, length(models)), diag)
    end
    # known-tumour forecast with oracle per-unit parameters; metadata as the inference side should save it
    cutoffs = Dict{String,Float64}()
    for u in units, (arm, c) in ((:treated, u.treated), (:control, u.control))
        cutoffs["$(u.model)|$(arm)"] = last_train_time(c, late_frac)
    end
    pd_late = mk(1:length(ids); extra = Dict("split" => Dict{String,Any}("late_frac" => late_frac), "cutoffs" => cutoffs))
    unv_late = verify_late_fit(pd_late, drug, units, late_frac)
    sel = function (c)
        m = falses(length(c.t))
        te = M.split_late_indices(length(c.t); frac = late_frac)[2]
        m[te] .= true
        return m
    end
    draws = common_draws(pd_late, S, seed)
    recs_late = predict_units(pd_late, units; newpdx = false, select = sel, draws, seed)
    cm_late = curve_metrics(recs_late; lod = M.detection_limit_from_fit(pd_late))
    # new-PDX with the oracle population fitted on the training models only
    train_ids, test_ids = M.split_models(units; frac = model_frac, seed)
    tr_idx = [i for (i, m) in enumerate(ids) if m in train_ids]
    pd_ho = mk(tr_idx; extra = Dict("train_models" => train_ids, "split" => Dict{String,Any}("model_frac" => model_frac, "seed" => seed)))
    unv_ho = verify_holdout_fit(pd_ho, drug, test_ids, train_ids; model_frac, seed)
    tu = [u for u in units if u.model in test_ids]
    draws = common_draws(pd_ho, S, seed)
    recs_new = predict_units(pd_ho, tu; newpdx = true, select = c -> c.t .> 0, draws, seed)
    cm_new = curve_metrics(recs_new; lod = M.detection_limit_from_fit(pd_ho))
    rm = response_metrics(pd_ho, tu; draws, seed)
    # the verification must reject a leaking fit, a wrong split setting and a wrong cutoff
    leak = mk(1:length(ids); extra = Dict("train_models" => ids))
    leak_err = try; verify_holdout_fit(leak, drug, test_ids, train_ids; model_frac, seed); nothing; catch e; e; end
    badfrac = try; verify_late_fit(pd_late, drug, units, 0.25); nothing; catch e; e; end
    bad_co = merge(cutoffs, Dict(first(keys(cutoffs)) => -1.0))
    badcut = try; verify_late_fit(mk(1:length(ids); extra = Dict("cutoffs" => bad_co)), drug, units, late_frac); nothing; catch e; e; end
    @assert leak_err isa ErrorException && badfrac isa ErrorException && badcut isa ErrorException "verification failed to reject bad fits"
    figs = make_figures("selftest", outdir, recs_new, cm_new, rm)
    p = write_report("selftest", outdir; late = cm_late, newpdx = cm_new, rm, figs,
                     unverified = vcat(unv_late, unv_ho),
                     notes = ["SELFTEST with an oracle posterior (true parameters); pipeline check only."])
    println("selftest report: ", p)
    println("known-tumour coverage (nominal 0.9): ", f3(cm_late.coverage_pooled), "  PDX-avg ", f3(cm_late.coverage_tumour_avg),
            "   new-PDX coverage: ", f3(cm_new.coverage_pooled))
    println("new-PDX response: tau_b=", f3(rm.agreement.tau_b), " AUROC=", f3(rm.agreement.auroc), " n=", rm.agreement.n)
    println("verification rejected leak / late_frac mismatch / cutoff mismatch: ", leak_err.msg[1:40], " | ", badfrac.msg[1:30], " | ", badcut.msg[1:30])
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
