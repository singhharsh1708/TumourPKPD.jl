# Validation metrics: PDXE response rules, predictive scores, PSIS-LOO, ranking, splits.
# Needs from the enclosing module: Random, Statistics, `import ParetoSmooth`.

# ----------------------------------------------------------------------------
# PDXE response classification
# ----------------------------------------------------------------------------

"Ordered PDXE categories, best response first."
const RESPONSE_CATEGORIES = ("mCR", "mPR", "mSD", "mPD")

"Ordinal response score: mCR=3, mPR=2, mSD=1, mPD=0 (higher = better response)."
response_score(cat::AbstractString) =
    cat == "mCR" ? 3 : cat == "mPR" ? 2 : cat == "mSD" ? 1 : cat == "mPD" ? 0 :
    throw(ArgumentError("unknown response category $(repr(cat))"))

"True for mCR and mPR (the 'responder' group used for AUROC / responder fractions)."
is_responder(cat::AbstractString) = cat == "mCR" || cat == "mPR"

"""
    pdxe_response(t, v; min_time=10.0, min_time_avg=min_time, strict_avg=false,
                  strict_best=false, min_obs=1, fallback_all=false) -> (best_response, best_avg_response, category)

PDXE tumour response classification of Gao et al. 2015 (Nat Med 21:1318, Online
Methods), mRECIST-adapted, as implemented in Xeva::mRECIST (bhklab/Xeva, R/mRECIST.R):

- ΔVol_t = 100 (V_t - V_0)/V_0, V_0 the first (day 0) volume. If V_0 == 0 every volume is
  shifted by +1 first (Xeva convention).
- BestResponse = min over t >= `min_time` (10 days) of ΔVol_t.
- BestAvgResponse = min over t >= `min_time_avg` of the running mean of ΔVol over all
  observations from the first up to t (day 0 included).
- mCR: BestResponse < -95 and BestAvgResponse < -40;
  mPR: BestResponse < -50 and BestAvgResponse < -20;
  mSD: BestResponse < 35 and BestAvgResponse < 30; otherwise mPD. All comparisons strict;
  the highest category satisfied wins.

With the defaults this is the literal rule. Cross-checking against the PDXE tables
(test/pdxe_crosscheck.jl) shows the published BestAvgResponse is reproduced only when the
window is `t > 10` (`strict_avg=true`); see [`pdxe_response_paper`](@ref).
`strict_*` switch `>=` to `>` for the time window. Returns category "NA" (and NaN values)
when fewer than `min_obs` observations fall in a window (Xeva requires 2; the PDXE tables
contain values for curves with a single late point, hence default 1) or the baseline is
not finite; with `fallback_all=true` a curve with no observation in the window is scored on
all its post-baseline points instead. Non-finite later volumes are dropped. Times need not be sorted.
"""
function pdxe_response(t::AbstractVector, v::AbstractVector; min_time::Real = 10.0,
                       min_time_avg::Real = min_time, strict_avg::Bool = false,
                       strict_best::Bool = false, min_obs::Integer = 1,
                       fallback_all::Bool = false)
    length(t) == length(v) || throw(DimensionMismatch("t and v differ in length"))
    na = (best_response = NaN, best_avg_response = NaN, category = "NA")
    ord = sortperm(t)
    ts = float.(t[ord]); vs = float.(v[ord])
    isempty(ts) && return na
    isfinite(vs[1]) || return na
    keep = [i == 1 || isfinite(vs[i]) for i in eachindex(vs)]
    ts = ts[keep]; vs = vs[keep]
    if vs[1] == 0
        vs = vs .+ 1
    end
    Δ = 100 .* (vs .- vs[1]) ./ vs[1]
    avg = cumsum(Δ) ./ (1:length(Δ))
    lb = strict_best ? findall(>(min_time), ts) : findall(>=(min_time), ts)
    la = strict_avg ? findall(>(min_time_avg), ts) : findall(>=(min_time_avg), ts)
    if fallback_all && (isempty(lb) || isempty(la))
        # no observation in the window (curve shorter than min_time): use all post-baseline points
        isempty(lb) && (lb = collect(2:length(ts)))
        isempty(la) && (la = collect(2:length(ts)))
    end
    (length(lb) < min_obs || length(la) < min_obs || isempty(lb) || isempty(la)) && return na
    br = minimum(Δ[lb]); bar = minimum(avg[la])
    cat = "mPD"
    if br < 35 && bar < 30
        cat = "mSD"
    end
    if br < -50 && bar < -20
        cat = "mPR"
    end
    if br < -95 && bar < -40
        cat = "mCR"
    end
    return (best_response = br, best_avg_response = bar, category = cat)
end

"""
    pdxe_response_paper(t, v)

`pdxe_response` with the settings that best reproduce the published PDXE tables
(`strict_avg=true, strict_best=true, fallback_all=true`: windows `t > 10`, and curves with no
observation after day 10 scored on all post-baseline points). Reproduces BestAvgResponse in
98.4% of PDXE curves and the 4-way category in ~91% (responder vs non-responder ≈ 99.8%).
The literal rule's BestResponse (min over t ≥ 10) is lower than the published value in ~40% of
curves, which moves some published mPD curves to mSD; see worker-validate.md.
"""
pdxe_response_paper(t, v) = pdxe_response(t, v; strict_avg = true, strict_best = true, fallback_all = true)

"""
    response_categories(tgrid, sims; classifier=pdxe_response_paper) -> Vector{String}

Apply `classifier(tgrid, row)` to each row (draw) of a simulation matrix whose columns are at
`tgrid` (must include day 0 as first column). The default is [`pdxe_response_paper`](@ref), the
variant that best reproduces the published PDXE categories (the literal rule is
[`pdxe_response`](@ref)). Rows containing non-finite volumes (failed solves) give "NA".
"""
function response_categories(tgrid, sims::AbstractMatrix; classifier = pdxe_response_paper)
    return [all(isfinite, @view(sims[r, :])) ? classifier(tgrid, @view(sims[r, :])).category : "NA"
            for r in axes(sims, 1)]
end

"Category probabilities `(mCR, mPR, mSD, mPD, NA)` from a vector of categories."
function category_probs(cats::AbstractVector{<:AbstractString})
    n = length(cats)
    f(c) = count(==(c), cats) / n
    return (mCR = f("mCR"), mPR = f("mPR"), mSD = f("mSD"), mPD = f("mPD"), NA = f("NA"))
end

"Expected ordinal score (mCR=3 … mPD=0) over draws, ignoring NA."
function expected_score(cats::AbstractVector{<:AbstractString})
    s = [response_score(c) for c in cats if c != "NA"]
    return isempty(s) ? NaN : mean(s)
end

"P(responder) = P(mCR or mPR) over draws, ignoring NA."
function p_responder(cats::AbstractVector{<:AbstractString})
    c = [x for x in cats if x != "NA"]
    return isempty(c) ? NaN : count(is_responder, c) / length(c)
end

# ----------------------------------------------------------------------------
# Predictive coverage / scores.  yobs :: Vector (n), sims :: Matrix (ndraw × n)
# ----------------------------------------------------------------------------

function _central_interval(col, level)
    α = (1 - level) / 2
    c = filter(isfinite, col)
    isempty(c) && return (NaN, NaN)
    return (quantile(c, α), quantile(c, 1 - α))
end

"""
    coverage(yobs, simdraws; level=0.9)

Fraction of observations inside the central `level` predictive interval of the matching
column of `simdraws` (`ndraw × n`). Non-finite observations are skipped. An observation whose
ensemble is empty (every draw failed, NaN) counts as a miss, never silently dropped; count them
with [`ensemble_failures`](@ref). Invariant to monotone transforms (log vs raw volume).
"""
function coverage(yobs::AbstractVector, sims::AbstractMatrix; level::Real = 0.9)
    size(sims, 2) == length(yobs) || throw(DimensionMismatch("sims must be ndraw × length(yobs)"))
    hit = 0; tot = 0
    for j in eachindex(yobs)
        isfinite(yobs[j]) || continue
        lo, hi = _central_interval(@view(sims[:, j]), level)
        tot += 1
        hit += isfinite(lo) && (lo <= yobs[j] <= hi)
    end
    return tot == 0 ? NaN : hit / tot
end

"""
    coverage_by_group(yobs, sims, groups; level=0.9)

`(pooled, tumour_avg, per_group)`: pooled coverage over all points, mean of per-group
coverages (each tumour weighted equally), and the per-group Dict.
"""
function coverage_by_group(yobs, sims, groups; level::Real = 0.9)
    gs = unique(groups)
    per = Dict(g => coverage(yobs[groups .== g], sims[:, groups .== g]; level) for g in gs)
    vals = filter(isfinite, collect(values(per)))
    return (pooled = coverage(yobs, sims; level),
            tumour_avg = isempty(vals) ? NaN : mean(vals), per_group = per)
end

"""
    interval_width(sims; level=0.9, transform=identity) -> Vector

Width of the central `level` interval per column, after applying `transform` to draws
(e.g. `log` for width on the log-volume scale).
"""
function interval_width(sims::AbstractMatrix; level::Real = 0.9, transform = identity)
    return [begin
                col = filter(x -> isfinite(x) && (transform === identity || x > 0), @view(sims[:, j]))
                lo, hi = _central_interval(transform.(col), level)
                hi - lo
            end for j in axes(sims, 2)]
end

"""
    interval_score(yobs, sims; level=0.9) -> Vector

Gneiting-Raftery interval score per observation (lower is better):
`(u-l) + (2/α)(l-y)₊ + (2/α)(y-u)₊`, α = 1-level.
"""
function interval_score(yobs::AbstractVector, sims::AbstractMatrix; level::Real = 0.9)
    α = 1 - level
    return [begin
                lo, hi = _central_interval(@view(sims[:, j]), level)
                y = yobs[j]
                (hi - lo) + (2 / α) * max(lo - y, 0) + (2 / α) * max(y - hi, 0)
            end for j in eachindex(yobs)]
end

"""
    ensemble_failures(sims) -> (n_failed_rows, n_empty_columns, n_nonfinite_entries)

Counts of draws (rows) containing any non-finite value, columns with no finite draw at all, and
non-finite entries. Report these next to every score: failed solves are not scored as if they
were absent.
"""
function ensemble_failures(sims::AbstractMatrix)
    rows = count(r -> !all(isfinite, @view(sims[r, :])), axes(sims, 1))
    cols = count(j -> !any(isfinite, @view(sims[:, j])), axes(sims, 2))
    return (n_failed_rows = rows, n_empty_columns = cols, n_nonfinite_entries = count(!isfinite, sims))
end

"""
    crps(yobs, sims) -> Vector

Ensemble CRPS per observation, `E|X-y| - ½E|X-X'|` (sorted-sample form). NaN for an observation
with an empty ensemble (count them with [`ensemble_failures`](@ref)).
"""
function crps(yobs::AbstractVector, sims::AbstractMatrix)
    return [begin
                x = sort!(filter(isfinite, sims[:, j])); n = length(x)
                n == 0 ? NaN : begin
                t1 = mean(abs.(x .- yobs[j]))
                # E|X-X'| = (2/n²) Σ (2i-n-1) x_i  for sorted x
                t2 = 2 * sum((2i - n - 1) * x[i] for i in 1:n) / n^2
                t1 - 0.5 * t2
                end
            end for j in eachindex(yobs)]
end

# log of (sum_i exp(x_i)) / ntotal, with ntotal >= length(x): failed draws (not in x) count as zero density.
function _val_logmeanexp(x, ntotal::Integer = length(x))
    isempty(x) && return -Inf
    m = maximum(x)
    return isfinite(m) ? m + log(sum(exp.(x .- m)) / ntotal) : -Inf
end
_val_lognormpdf(x, μ, σ) = -0.5 * ((x - μ) / σ)^2 - log(σ) - 0.5 * log(2π)
_val_lognormcdf(x, μ, σ) = log(max(0.5 * erfc_(-(x - μ) / (σ * sqrt(2))), floatmin(Float64)))

# Observation-model densities on the log scale: Normal (`d === nothing`) or Student-t (`d::TDist`), scale σ.
_val_obsdist(ν::Real) = isinf(ν) ? nothing : TDist(float(ν))
_val_logpdf(x, μ, σ, ::Nothing) = _val_lognormpdf(x, μ, σ)
_val_logpdf(x, μ, σ, d::TDist) = logpdf(d, (x - μ) / σ) - log(σ)
_val_logcdf(x, μ, σ, ::Nothing) = _val_lognormcdf(x, μ, σ)
_val_logcdf(x, μ, σ, d::TDist) = max(logcdf(d, (x - μ) / σ), log(floatmin(Float64)))

# erfc without a SpecialFunctions dependency (Numerical Recipes erfc Chebyshev, rel. err < 1.2e-7;
# adequate for censored-likelihood scoring).
function erfc_(x)
    z = abs(x)
    t = 1 / (1 + 0.5z)
    r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
        t * (-0.82215223 + t * 0.17087277)))))))))
    return x >= 0 ? r : 2 - r
end

"""
    log_predictive_density(yobs, latent, σ; limit=nothing, jacobian=true, nu=5.0) -> Vector

Pointwise log predictive density of observations under the fitted observation model
`log y = log w + σ ε`, `ε ~ Student-t(nu)` (`nu = nu_from_fit(pd)`; `nu = Inf` is the Normal special case),
mixed over posterior draws: `log mean_s p(log y_j | log w_sj, σ_s)`. Left-censored observations use the
matching CDF.
`latent` is `ndraw × n` noise-free trajectories (`simulate_regimen(...; obs_noise=false,
details=true).y`), `σ` the matching per-draw σ. `jacobian=true` subtracts `log y` so the
value is a density on the volume scale (comparable across transforms). Observations
`<= 0` (undetectable) are scored as left-censored at `limit` (P(y < limit)) when `limit`
is given, otherwise NaN. Failed draws (non-finite or non-positive latent volume) contribute zero
density but stay in the denominator, and an observation with no valid draw scores `-Inf`; these
`-Inf` values are kept (count them with `count(==(-Inf), lpd)`), never dropped.
"""
function log_predictive_density(yobs::AbstractVector, latent::AbstractMatrix, σ::AbstractVector;
                                limit = nothing, jacobian::Bool = true, nu::Real = 5.0)
    size(latent, 1) == length(σ) || throw(DimensionMismatch("σ must have one entry per draw"))
    dist = _val_obsdist(nu)
    S = size(latent, 1)
    out = Vector{Float64}(undef, length(yobs))
    for j in eachindex(yobs)
        y = yobs[j]
        if !isfinite(y)
            out[j] = NaN
        elseif y <= 0
            if limit === nothing
                out[j] = NaN
            else
                lw = [_val_logcdf(log(limit), log(latent[s, j]), σ[s], dist) for s in 1:S
                      if isfinite(latent[s, j]) && latent[s, j] > 0]
                out[j] = _val_logmeanexp(lw, S)
            end
        else
            ly = log(y)
            lw = [_val_logpdf(ly, log(latent[s, j]), σ[s], dist) for s in 1:S
                  if isfinite(latent[s, j]) && latent[s, j] > 0]
            out[j] = _val_logmeanexp(lw, S) - (jacobian ? ly : 0.0)
        end
    end
    return out
end

"""
    log_predictive_density_kde(yobs, sims; jacobian=true) -> Vector

Model-free alternative: Gaussian kernel density (Silverman bandwidth) of the *noisy*
predictive draws on the log scale, evaluated at `log y`. Use when the per-draw σ is not at
hand. Draws or observations `<= 0` are ignored / NaN.
"""
function log_predictive_density_kde(yobs::AbstractVector, sims::AbstractMatrix; jacobian::Bool = true)
    return [begin
                y = yobs[j]
                x = [log(s) for s in sims[:, j] if isfinite(s) && s > 0]
                if !(isfinite(y) && y > 0) || length(x) < 2
                    NaN
                else
                    h = 1.06 * std(x) * length(x)^(-0.2)
                    h = max(h, 1e-6)
                    ly = log(y)
                    _val_logmeanexp(_val_lognormpdf.(ly, x, h)) - (jacobian ? ly : 0.0)
                end
            end for j in eachindex(yobs)]
end

"""
    cluster_bootstrap(stat, clusters; nboot=1000, rng) -> Vector

Cluster (e.g. PDX-model) bootstrap. `clusters` labels each observation; each replicate
resamples whole clusters with replacement and calls `stat(idx)` with the concatenated
observation indices (so `stat` closes over the data). Returns the `nboot` statistics.
"""
function cluster_bootstrap(stat, clusters::AbstractVector; nboot::Int = 1000,
                           rng::AbstractRNG = Random.default_rng())
    gs = unique(clusters)
    members = [findall(==(g), clusters) for g in gs]
    out = Vector{Float64}(undef, nboot)
    for b in 1:nboot
        pick = rand(rng, 1:length(gs), length(gs))
        idx = reduce(vcat, (members[p] for p in pick))
        out[b] = stat(idx)
    end
    return out
end

# ----------------------------------------------------------------------------
# PSIS-LOO
# ----------------------------------------------------------------------------

"""
    loo(loglik::AbstractMatrix; groups=nothing, chain_index=nothing)

PSIS leave-one-out via ParetoSmooth.jl. `loglik` is `ndraw × nobs` (pointwise
log-likelihood per posterior draw; draws from several chains concatenated, with
`chain_index` giving each draw's chain, default one chain). With `groups` (a label per
column) the columns of each group are summed first, so the unit left out is the whole group
(e.g. one PDX model's likelihood block) — the primary assessment for new-PDX prediction.

Returns `(elpd, se, p_eff, pareto_k, pareto_k_counts, pointwise_elpd, n, groups)`;
`pareto_k_counts` = counts of k in (≤0.5, 0.5–0.7, 0.7–1, >1).
Note ParetoSmooth wants `(data, draws)` orientation; this wrapper transposes.
"""
function loo(loglik::AbstractMatrix; groups = nothing, chain_index = nothing)
    L = Matrix{Float64}(loglik)
    labels = nothing
    if groups !== nothing
        length(groups) == size(L, 2) || throw(DimensionMismatch("groups must label every column"))
        labels = unique(groups)
        G = zeros(size(L, 1), length(labels))
        for (k, g) in enumerate(labels)
            G[:, k] = sum(@view(L[:, groups .== g]); dims = 2)
        end
        L = G
    end
    ci = chain_index === nothing ? ones(Int, size(L, 1)) : Int.(chain_index)
    res = ParetoSmooth.psis_loo(permutedims(L), ; chain_index = ci)
    est = res.estimates
    elpd = est(:cv_elpd, :total)
    se = est(:cv_elpd, :se_total)
    peff = est(:p_eff, :total)
    k = collect(Float64, res.psis_object.pareto_k)
    counts = (good = count(<=(0.5), k), ok = count(x -> 0.5 < x <= 0.7, k),
              bad = count(x -> 0.7 < x <= 1, k), very_bad = count(>(1), k))
    pw = collect(Float64, res.pointwise(:cv_elpd))
    return (elpd = elpd, se = se, p_eff = peff, pareto_k = k, pareto_k_counts = counts,
            pointwise_elpd = pw, n = length(k), groups = labels)
end

# Metadata saved with a fit (PosteriorDraws.diagnostics, or its "settings" sub-dict).
function _fit_meta(pd, key, default = nothing)
    d = pd.diagnostics
    haskey(d, key) && return d[key]
    st = get(d, "settings", nothing)
    st isa AbstractDict && haskey(st, key) && return st[key]
    return default
end

"""
    chain_index_from_fit(pd) -> Vector{Int}

Chain label of every stored draw, from the saved `nchains` (draws are stored chain after chain).
Errors if the draw count is not divisible by `nchains`; use as `loo(ll; chain_index=...)`.
"""
function chain_index_from_fit(pd)
    nch = Int(_fit_meta(pd, "nchains", 1))
    S = length(pd.σ)
    S % nch == 0 || throw(ArgumentError("$(S) stored draws are not divisible by nchains=$(nch)"))
    return repeat(1:nch, inner = S ÷ nch)
end

"""
    detection_limit_from_fit(pd) -> Float64

Detection limit (mm^3) the fit used for left-censoring: `diagnostics["lod"]` (or `settings["lod"]`),
falling back to `detection_limit()` if defined, else 4.0.
"""
function detection_limit_from_fit(pd)
    v = _fit_meta(pd, "lod", nothing)
    v !== nothing && return float(v)
    return isdefined(@__MODULE__, :detection_limit) ? float(detection_limit()) : 4.0
end

"""
    nu_from_fit(pd) -> Float64

Degrees of freedom of the fitted Student-t observation model (`diagnostics["nu"]`, else 5; `Inf` = Normal).
"""
nu_from_fit(pd) = float(_fit_meta(pd, "nu", 5.0))

"Fixed sd of the latent log baselines (`diagnostics[\"baseline_sd\"]`, else 0.1)."
baseline_sd_from_fit(pd) = float(_fit_meta(pd, "baseline_sd", 0.1))

"""
    convergence_summary(pd) -> NamedTuple

Sampler diagnostics of a fit for the validation report: what the fit recorded (`max_rhat`, `min_ess_bulk`,
`min_ess_tail`, `n_divergences`, E-BFMI, tree-depth cap hits, mean acceptance / tree depth / step size /
leapfrog steps, the cross-mode R-hat, MAP log-densities, chains and draws per chain; `nothing` when the
fit did not record an item) and a recomputation from the stored draws of the seven hyperparameters
(`μ_λ0, ω_λ0, μ_k2, ω_k2, logλ1, logk1, σ`, plus `corr_lk`, `mu_logVtr`, `omega_Vtr`, `waning_rate` for v2 / waning fits) using the saved chain layout (`recomputed`: names, rhat,
ess_bulk, ess_tail; `nothing` if the draw count does not match `nchains`). Draws are stored chain after
chain. Keys read: `nchains`, `draws_per_chain`, `max_rhat`, `min_ess_bulk`, `min_ess_tail`, `n_divergences`,
`min_ebfmi`/`ebfmi`, `max_tree_depth_hits`, `mean_tree_depth`, `mean_acceptance`, `mean_step_size`,
`mean_n_steps`, `cross_mode_rhat`/`rhat_cross_mode`, `map_lps`; every other recorded key containing
"rhat", "bfmi" or a `mode` word (e.g. `cross_mode_...`) is returned in `extra`.
"""
function convergence_summary(pd)
    first_meta(keys...) = (for k in keys
                               v = _fit_meta(pd, k, nothing)
                               v === nothing || return v
                           end; nothing)
    nch = Int(_fit_meta(pd, "nchains", 1))
    S = length(pd.σ)
    recomputed = nothing
    if S % nch == 0 && S ÷ nch >= 4
        L = S ÷ nch
        nm = ["μ_λ0", "ω_λ0", "μ_k2", "ω_k2", "logλ1", "logk1", "σ"]
        cols = Any[pd.μ_λ0, pd.ω_λ0, pd.μ_k2, pd.ω_k2, pd.logλ1, pd.logk1, pd.σ]
        # v2 / waning hyperparameters (per draw), when the fit has them
        for (lab, key) in (("corr(log λ0, log k2)", "corr_lk"), ("μ_logVtr", "mu_logVtr"), ("ω_Vtr", "omega_Vtr"), ("waning rate", "waning_rate"))
            v = first_meta(key)
            (v isa AbstractVector && length(v) == S) && (push!(nm, lab); push!(cols, v))
        end
        arr = Array{Float64}(undef, L, length(nm), nch)
        for c in 1:nch, (j, v) in enumerate(cols)
            arr[:, j, c] = v[(c-1)*L+1:c*L]
        end
        ss = MCMCChains.summarystats(MCMCChains.Chains(arr, Symbol.(nm)))
        recomputed = (names = nm, rhat = collect(Float64, ss.nt.rhat), ess_bulk = collect(Float64, ss.nt.ess_bulk),
                      ess_tail = collect(Float64, ss.nt.ess_tail))
    end
    known = Set(["nchains", "draws_per_chain", "max_rhat", "min_ess_bulk", "min_ess_tail", "n_divergences", "min_ebfmi",
                 "ebfmi", "max_tree_depth_hits", "mean_tree_depth", "mean_acceptance", "mean_step_size", "mean_n_steps",
                 "cross_mode_rhat", "rhat_cross_mode", "map_lps"])
    extra = Dict{String,Any}()
    for d in (pd.diagnostics, get(pd.diagnostics, "settings", Dict{String,Any}()))
        d isa AbstractDict || continue
        for (k, v) in d
            ks = String(k)
            ks in known && continue
            occursin(r"rhat|bfmi|(^|_)mode(_|$)", lowercase(ks)) && !(v isa AbstractVector{<:AbstractString}) && (extra[ks] = v)
        end
    end
    return (nchains = nch, draws_per_chain = first_meta("draws_per_chain"), n_draws = S,
            max_rhat = first_meta("max_rhat"), min_ess_bulk = first_meta("min_ess_bulk"),
            min_ess_tail = first_meta("min_ess_tail"), n_divergences = first_meta("n_divergences"),
            ebfmi = first_meta("min_ebfmi", "ebfmi"), max_tree_depth_hits = first_meta("max_tree_depth_hits"),
            mean_tree_depth = first_meta("mean_tree_depth"), mean_acceptance = first_meta("mean_acceptance"),
            mean_step_size = first_meta("mean_step_size"), mean_n_steps = first_meta("mean_n_steps"),
            cross_mode_rhat = first_meta("cross_mode_rhat", "rhat_cross_mode"), map_lps = first_meta("map_lps"),
            recomputed = recomputed, extra = extra)
end

# ----------------------------------------------------------------------------
# Public-output guard (PDXE data must not be redistributed)
# ----------------------------------------------------------------------------

"Minimum number of distinct PDX models any public bin or summary must aggregate."
const PUBLIC_MIN_PDX = 10

"""
    public_output_violations(; bins=[], summaries=[], text="", ids=[], figures=[], min_pdx=PUBLIC_MIN_PDX) -> Vector{String}

Rules for report outputs that may be published (the PDXE data are licensed and must not be
redistributed): every bin (`n_pdx` field) and summary (`n_pdx` field) aggregates at least `min_pdx` distinct
PDX models; the text contains none of the PDX model `ids` (as a whole token); and no individual-level figure
(file names ending `_bands.png` or `_response.png`, which show individual curves / per-PDX categories) is among
`figures`; and the text contains no absolute local path (`/Users/`, `/home/`, `/private/`). Returns the violations
(empty = fine).
"""
function public_output_violations(; bins = NamedTuple[], summaries = NamedTuple[], text::AbstractString = "",
                                  ids = String[], figures = String[], min_pdx::Integer = PUBLIC_MIN_PDX)
    v = String[]
    for b in bins
        b.n_pdx < min_pdx && push!(v, "bin $(get(b, :arm, "")) $(get(b, :lo, "?"))-$(get(b, :hi, "?")) aggregates only $(b.n_pdx) PDX models (< $(min_pdx))")
    end
    for (i, sm) in enumerate(summaries)
        sm.n_pdx < min_pdx && push!(v, "summary #$(i) aggregates only $(sm.n_pdx) PDX models (< $(min_pdx))")
    end
    for id in ids
        isempty(id) && continue
        occursin(Regex("(?<![A-Za-z0-9])\\Q" * String(id) * "\\E(?![A-Za-z0-9])"), text) &&
            push!(v, "text contains the PDX id $(id)")
    end
    for pat in ("/Users/", "/home/", "/private/")
        occursin(pat, text) && push!(v, "text contains an absolute local path ($(pat)...); use repo-relative paths")
    end
    for f in figures
        occursin(r"_(bands|response)\.png$", String(f)) &&
            push!(v, "individual-level figure $(basename(String(f))) must go to the local output directory")
    end
    return v
end

"Throw if [`public_output_violations`](@ref) finds anything; used before writing public outputs."
function assert_public_output(; kwargs...)
    v = public_output_violations(; kwargs...)
    isempty(v) || error("refusing to write public output: " * join(v, "; "))
    return nothing
end

# ----------------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------------

function _ranks(x::AbstractVector)
    n = length(x); p = sortperm(x); r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[p[j+1]] == x[p[i]]
            j += 1
        end
        r[p[i:j]] .= (i + j) / 2
        i = j + 1
    end
    return r
end

"Spearman rank correlation (average ranks for ties)."
spearman(x::AbstractVector, y::AbstractVector) = cor(_ranks(x), _ranks(y))

"""
    kendall_tau_b(x, y)

Kendall's τ_b (tie-corrected), O(n²). NaN if either vector is constant.
"""
function kendall_tau_b(x::AbstractVector, y::AbstractVector)
    n = length(x); n == length(y) || throw(DimensionMismatch("x and y differ in length"))
    conc_ = 0; disc = 0; tx = 0; ty = 0
    for i in 1:n-1, j in i+1:n
        dx = sign(x[i] - x[j]); dy = sign(y[i] - y[j])
        if dx == 0 && dy == 0
            continue
        elseif dx == 0
            tx += 1
        elseif dy == 0
            ty += 1
        elseif dx == dy
            conc_ += 1
        else
            disc += 1
        end
    end
    d = sqrt(float(conc_ + disc + tx) * float(conc_ + disc + ty))
    return d == 0 ? NaN : (conc_ - disc) / d
end

"""
    auroc(scores, labels)

Area under the ROC curve for `labels::Bool` (true = positive) given `scores` (higher =
more likely positive); ties count 1/2 (Mann-Whitney U). NaN if one class is empty.
"""
function auroc(scores::AbstractVector, labels::AbstractVector{Bool})
    r = _ranks(scores)
    n1 = count(labels); n0 = length(labels) - n1
    (n1 == 0 || n0 == 0) && return NaN
    return (sum(r[labels]) - n1 * (n1 + 1) / 2) / (n1 * n0)
end

"""
    ordinal_agreement(pred_scores, obs_categories) -> (tau_b, auroc, n)

Kendall τ_b between a predicted ordinal score `pred_scores` (e.g. [`expected_score`](@ref) over
draws) and the observed PDXE category, and the AUROC of `responder_scores` (default
`pred_scores`; pass [`p_responder`](@ref) per model, the quantity AUROC should rank by) for
observed responders (mCR/mPR) vs the rest. Pairs with observed "NA" or non-finite scores are
dropped.
"""
function ordinal_agreement(pred_scores::AbstractVector, obs_categories::AbstractVector{<:AbstractString};
                           responder_scores::AbstractVector = pred_scores)
    length(responder_scores) == length(pred_scores) || throw(DimensionMismatch("score vectors differ in length"))
    keep = [c != "NA" && isfinite(s) && isfinite(r)
            for (s, r, c) in zip(pred_scores, responder_scores, obs_categories)]
    s = pred_scores[keep]; r = responder_scores[keep]; c = obs_categories[keep]
    return (tau_b = kendall_tau_b(s, [response_score(x) for x in c]),
            auroc = auroc(r, [is_responder(x) for x in c]), n = length(s))
end

"""
    responder_fraction(R; level=0.9)

`R[draw, model]` = predicted responder indicator (`true`/`false`, `missing` for a failed/NA draw)
for one drug on its shared PDX models. **Row `s` must be the same posterior (population) draw for
every model** (simulate all models with the same `draws=` indices). Returns the per-draw fraction
(a posterior sample of the drug's responder fraction, over the non-missing models of that row)
with mean and central interval, and the number of missing entries.
"""
function responder_fraction(R::AbstractMatrix{<:Union{Missing,Bool}}; level::Real = 0.9)
    f = [begin
             v = collect(skipmissing(@view(R[r, :])))
             isempty(v) ? NaN : mean(v)
         end for r in axes(R, 1)]
    ff = filter(isfinite, f)
    lo, hi = _central_interval(ff, level)
    return (mean = isempty(ff) ? NaN : mean(ff), lo = lo, hi = hi, draws = f,
            n_missing = count(ismissing, R))
end

"""
    drug_ranking_agreement(pred_fraction, obs_fraction) -> (spearman, kendall_tau_b, n)

Agreement between predicted and observed per-drug responder fractions (same drug order).
"""
function drug_ranking_agreement(pred::AbstractVector, obs::AbstractVector)
    return (spearman = spearman(pred, obs), kendall_tau_b = kendall_tau_b(pred, obs), n = length(pred))
end

# ----------------------------------------------------------------------------
# Hold-out splits
# ----------------------------------------------------------------------------

"""
    split_late_indices(n; frac=0.3, min_train=3) -> (train_idx, test_idx)

Deterministic: the last `round(frac*n)` timepoints (at least 1, but leaving at least
`min_train`) are held out. Returns `(1:n, Int[])` when the curve is too short.
"""
function split_late_indices(n::Integer; frac::Real = 0.3, min_train::Integer = 3)
    nt = clamp(round(Int, frac * n), 1, max(n - min_train, 0))
    n - min_train < 1 && return (collect(1:n), Int[])
    return (collect(1:n-nt), collect(n-nt+1:n))
end

"""
    split_late(curve; frac=0.3, min_train=3) -> (train, test)

Split a curve (any struct with `model, tumour_type, treatment, t, v` fields, e.g.
`TumourCurve`) into early points for fitting and the last `frac` of timepoints held out.
The split is by observation order (sort by time first), deterministic, day 0 always kept.
"""
function split_late(curve; frac::Real = 0.3, min_train::Integer = 3)
    ord = sortperm(curve.t)
    tr, te = split_late_indices(length(ord); frac, min_train)
    mk(ix) = typeof(curve)(curve.model, curve.tumour_type, curve.treatment,
                           curve.t[ord[ix]], curve.v[ord[ix]])
    return mk(tr), mk(te)
end

"""
    split_models(models; frac=0.2, seed=1, strata=nothing) -> (train_ids, test_ids)

Whole-PDX holdout, stratified by tumour type. `models` is either a vector of model ids
(then pass `strata=` tumour types of the same length) or a vector of objects with `.model`
and `.tumour_type` (e.g. `PDXUnit`, `TumourCurve`). Duplicate ids are collapsed. Within each
stratum ids are sorted, shuffled with `MersenneTwister(seed + stratum-hash)` and
`round(frac*n)` held out (at least 1 when the stratum has ≥ 2 PDX; 0 for singletons).
Output is independent of input order. A model carrying more than one stratum label is an
`ArgumentError` (`on_conflict=:majority` assigns its most frequent label instead).
"""
function split_models(models::AbstractVector; frac::Real = 0.2, seed::Integer = 1, strata = nothing,
                      on_conflict::Symbol = :error)
    if eltype(models) <: AbstractString
        strata === nothing && (strata = fill("all", length(models)))
        length(strata) == length(models) || throw(DimensionMismatch("strata must label every model"))
        ids = String.(models); st = String.(strata)
    else
        ids = [String(m.model) for m in models]; st = [String(m.tumour_type) for m in models]
    end
    # one stratum per PDX: a model must never be in two strata (it could then land in both sets)
    lab = Dict{String,Vector{String}}()
    for (i, s_) in zip(ids, st)
        push!(get!(lab, i, String[]), s_)
    end
    for i in collect(keys(lab))
        ls = lab[i]
        length(unique(ls)) > 1 || continue
        on_conflict === :error && throw(ArgumentError(
            "model $(i) has inconsistent strata $(sort(unique(ls))); fix the labels or use on_conflict=:majority"))
        on_conflict === :majority || throw(ArgumentError("on_conflict must be :error or :majority"))
        u = sort(unique(ls))
        lab[i] = [u[argmax([count(==(x), ls) for x in u])]]   # ties: first in sorted order
    end
    pairs = unique([(i, first(lab[i])) for i in ids])
    train = String[]; test = String[]
    for s in sort(unique(last.(pairs)))
        grp = sort([first(p) for p in pairs if last(p) == s])
        n = length(grp)
        h = foldl((a, c) -> mod(a * 131 + Int(c), 1_000_003), collect(s); init = 7)
        rng = MersenneTwister(seed + h)
        perm = randperm(rng, n)
        nt = n < 2 ? 0 : max(1, round(Int, frac * n))
        append!(test, grp[perm[1:nt]]); append!(train, grp[perm[nt+1:end]])
    end
    return sort!(train), sort!(test)
end
