# Prior predictive check of the growth-side priors (μ_λ0, ω_λ0, transition volume Vtr = λ1/λ0) against observed control curves.
#   julia --project=. scripts/prior_predictive.jl [drug] [window] [n] [--exploratory]
#
# DEFAULT (strict) design: only the frozen 80% TRAINING PDX of `frozen_pdx_split` and only days <= window (default 21, the
# `late21` design, i.e. the least data any fit is allowed to see). No held-out PDX and no day beyond the window is read.
# `--exploratory` reproduces the earlier all-PDX / 42-day look that was used to move the μ_λ0 prior (phase 2b); that check
# used the held-out PDX and post-21-day data and is therefore EXPLORATORY ONLY, not a clean prior-selection step.
using TumourPKPD, Random, Statistics, Printf, Distributions
isdefined(TumourPKPD, :fit_drug) || Base.include(TumourPKPD, joinpath(@__DIR__, "..", "src", "hierarchical.jl"))
const H = TumourPKPD
pos = filter(a -> !startswith(a, "--"), ARGS); expl = "--exploratory" in ARGS
drug = length(pos) >= 1 ? pos[1] : "binimetinib"
window = length(pos) >= 2 ? parse(Float64, pos[2]) : (expl ? 42.0 : 21.0)
nsim = length(pos) >= 3 ? parse(Int, pos[3]) : 4000
if !expl && !(isfinite(window) && 0 < window <= 21)
    error("strict prior-predictive check requires a finite window 0 < window <= 21 days (got $window); use --exploratory for anything else")
end
units = drug_dataset(drug)
if !expl
    train_ids, test_ids, _ = H.frozen_pdx_split()
    keep = Set(train_ids)
    units = [u for u in units if u.model in keep]
end
units = H.window_units(units, window)
ctl = [u.control for u in units]
q(x, ps = (0.05, 0.25, 0.5, 0.75, 0.95)) = round.(quantile(x, collect(ps)); digits = 0)
println("# $drug prior predictive, " * (expl ? "EXPLORATORY (all PDX incl. held-out, " : "STRICT (frozen training PDX only, ") * "$(length(ctl)) control curves, days <= $window)")
v0 = [c.v[1] for c in ctl]; vmax = [maximum(c.v) for c in ctl]; vlast = [c.v[end] for c in ctl]; tlast = [c.t[end] for c in ctl]
println("v0 quantiles (5,25,50,75,95%):       ", q(v0))
println("max observed volume:                 ", q(vmax), "   fraction > 1000: ", round(mean(vmax .> 1000); digits = 2))
println("last observation day:                ", q(tlast))
sl1 = Float64[]; sl2 = Float64[]
for c in ctl
    pos_ = c.v .> 0; t = c.t[pos_]; y = log.(c.v[pos_])
    length(t) >= 6 || continue
    h = length(t) ÷ 2
    slope(a, b) = (m = a .- mean(a); sum(m .* (b .- mean(b))) / max(sum(m .^ 2), eps()))
    push!(sl1, slope(t[1:h], y[1:h])); push!(sl2, slope(t[h:end], y[h:end]))
end
println("median log-slope first / second half of each control (1/day): ", round(median(sl1); digits = 3), " / ", round(median(sl2); digits = 3),
        "  (n=$(length(sl1)) with >= 6 obs)")
long = [c for c in ctl if c.t[end] >= 0.8 * window]
obslast = [c.v[end] for c in long]; hi = maximum(vmax)
println("controls followed to >= $(round(0.8 * window)) days: ", length(long), "; their last volumes: ", q(obslast))

pri = H.default_priors()
function pp(name, μprior, ωsd, vtr; rng = MersenneTwister(1))
    simlast = Float64[]; simmax = Float64[]
    for _ in 1:nsim
        μ = μprior[1] + μprior[2] * randn(rng); ω = abs(ωsd * randn(rng))
        lvtr = vtr[1] + vtr[2] * randn(rng)
        c = long[rand(rng, 1:length(long))]
        λ0 = exp(μ + ω * randn(rng)); λ1 = exp(μ + lvtr)
        w0 = exp(log(c.v[1]) + 0.1 * randn(rng))          # latent baseline: log w0 = log v0 + 0.1 z, as in the fitted model
        w, ok = simulate_tumour(Simeoni(λ0, λ1, 1.0, 0.0, w0), NoDrug(), collect(c.t[2:end]))
        ok || continue
        # observation model: Student-t(5) log-volume noise with σ ~ HalfNormal(prior); the start volume above already carries the latent-baseline draw
        σ = abs(pri.σ * randn(rng))
        push!(simlast, w[end] * exp(σ * rand(rng, TDist(5.0)))); push!(simmax, maximum(w))
    end
    lo5, hi95 = quantile(simlast, 0.05), quantile(simlast, 0.95)
    inside = [quantile(obslast, p) for p in (0.05, 0.5, 0.95)]
    passes = all(lo5 .<= inside .<= hi95) && mean(simlast .> 10hi) < 0.05
    @printf("%-28s prior-pred OBSERVED-volume (with t5 noise) 5/25/50/75/95%%: %s | observed 5/50/95%% inside the 5-95%% band: %s | frac > 3x largest observed (%.0f): %.3f | frac > 10x: %.3f | PASS %s\n",
            name, string(q(simlast)), string(all(lo5 .<= inside .<= hi95)), 3hi, mean(simmax .> 3hi), mean(simlast .> 10hi), passes)
    return passes
end
println("\ncriterion: observed last-volume 5/50/95% quantiles of the controls lie inside the prior-predictive 5-95% band, and < 5% of prior-predictive volumes exceed 10x the largest observed volume")
ok_cur = pp("current  mu_lam0 N(log $(round(exp(pri.μ_λ0[1]); digits = 3)),$(pri.μ_λ0[2]))", pri.μ_λ0, pri.ω_λ0, pri.logVtr)
pp("phase-2b mu_lam0 N(log .06,.5)", (log(0.06), 0.5), pri.ω_λ0, pri.logVtr)
ok_old = pp("earlier  mu_lam0 N(log .12,.7)", (log(0.12), 0.7), pri.ω_λ0, pri.logVtr)
for (m_, s_) in ((0.05, 0.6), (0.055, 0.6), (0.05, 0.5), (0.045, 0.6))
    pp("candidate mu_lam0 N(log $(m_),$(s_))", (log(m_), s_), pri.ω_λ0, pri.logVtr)
end
println("\nVtr prior 95% interval (mm^3): ", round.(exp.(pri.logVtr[1] .+ [-1.96, 1.96] .* pri.logVtr[2]); digits = 0), "; implied λ1 median ", round(exp(pri.μ_λ0[1] + pri.logVtr[1]); digits = 1), " mm^3/day")
println("DECISION: current μ_λ0 prior ", ok_cur ? "KEPT (passes on the strict design)" : "FAILS the strict check")
