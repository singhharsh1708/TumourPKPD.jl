# Tests for src/simulate.jl and src/validate.jl.
# Run: julia --project=TumourPKPD.jl -t 2 TumourPKPD.jl/test/test_validate.jl
using Test, Random, Statistics, Distributions
using TumourPKPD
import StatsBase, ParetoSmooth

const ROOT = normpath(joinpath(@__DIR__, ".."))

# Until the lead includes the files in the package, load them into the module here
# (hierarchical.jl first, then simulate.jl and validate.jl, the intended include order).
if !isdefined(TumourPKPD, :simulate_regimen)
    @eval TumourPKPD begin
        import ParetoSmooth
        # include the real hierarchical.jl (also checks that it and validate.jl coexist in one module)
        isdefined(@__MODULE__, :PosteriorDraws) || include($(joinpath(ROOT, "src", "hierarchical.jl")))
        include($(joinpath(ROOT, "src", "simulate.jl")))
        include($(joinpath(ROOT, "src", "validate.jl")))
    end
end
const TP = TumourPKPD
const PD = TP.PosteriorDraws

# ---------------------------------------------------------------- response rules
@testset "pdxe_response" begin
    t = [0, 3, 7, 10, 14, 21, 28.0]
    resp(v) = TP.pdxe_response(t, v)
    # mCR: best -98, running avg at day 10 = -41.5
    r = resp([100, 80, 50, 4, 3, 2, 2.0])
    @test r.category == "mCR"
    @test r.best_response ≈ -98
    @test r.best_avg_response ≈ -(20 + 50 + 96 + 97 + 98 + 98) / 7
    @test TP.pdxe_response(t, [100, 80, 50, 4, 3, 2, 2.0])[3] == "mCR"   # positional destructuring
    # best response exactly -95 is NOT mCR (strict) -> mPR
    @test resp([100, 80, 50, 5, 5, 5, 5.0]).best_response ≈ -95
    @test resp([100, 80, 50, 5, 5, 5, 5.0]).category == "mPR"
    # mCR needs avg < -40: slow decline, best -96 but average above -40
    @test resp([100, 100, 100, 100, 100, 100, 4.0]).best_response ≈ -96
    @test resp([100, 100, 100, 100, 100, 100, 4.0]).best_avg_response ≈ -96 / 7
    @test resp([100, 100, 100, 100, 100, 100, 4.0]).category == "mSD"       # avg too weak for mPR/mCR
    # mPR
    @test resp([100, 80, 60, 50, 40, 40, 40.0]).category == "mPR"       # best -60, avg ≈ -29
    # best exactly -50 -> not mPR (strict) -> mSD
    @test resp([100, 90, 70, 50, 50, 50, 50.0]).best_response ≈ -50
    @test resp([100, 90, 70, 50, 50, 50, 50.0]).category == "mSD"
    # mSD
    @test resp([100, 105, 110, 115, 120, 118, 119.0]).category == "mSD"
    # best exactly 35 -> mPD (need all t>=10 volumes >= 135, min is 135)
    @test resp([100, 110, 120, 135, 140, 150, 160.0]).best_response ≈ 35
    @test resp([100, 110, 120, 135, 140, 150, 160.0]).category == "mPD"
    # avg exactly 30 -> mPD: constant Δ=30 from day 0 gives avg (0+30*k)/(k+1) < 30, so use
    # best<35 but avg>=30 instead
    @test resp([100, 300, 300, 300, 130, 300, 300.0]).category == "mPD"
    # mPD
    @test resp([100, 150, 200, 300, 400, 500, 600.0]).category == "mPD"
    # large late response preceded by growth -> average blocks the category
    rr = resp([100, 300, 300, 300, 40, 40, 40.0])
    @test rr.best_response ≈ -60
    @test rr.best_avg_response ≈ 60
    @test rr.category == "mPD"
    # dip before day 10 is not counted as best response
    @test resp([100, 5, 5, 200, 250, 300, 400.0]).category == "mPD"
    # no point at t >= 10 -> NA (or scored on all points with fallback_all); Xeva needs >= 2 points (min_obs=2)
    @test TP.pdxe_response([0, 3, 7.0], [1, 1, 1.0]).category == "NA"
    @test isnan(TP.pdxe_response([0, 3, 7.0], [1, 1, 1.0]).best_response)
    @test TP.pdxe_response([0, 3, 7.0], [100, 80, 150.0]; fallback_all = true).best_response ≈ -20
    @test TP.pdxe_response([0, 5, 10.0], [1, 1, 1.0]).category == "mSD"
    @test TP.pdxe_response([0, 5, 10.0], [1, 1, 1.0]; min_obs = 2).category == "NA"
    # strict windows: a day-10 shrinkage counts for t >= 10 but not for t > 10
    @test resp([100, 100, 100, 40, 100, 100, 100.0]).best_response ≈ -60
    @test TP.pdxe_response(t, [100, 100, 100, 40, 100, 100, 100.0]; strict_best = true).best_response ≈ 0
    @test TP.pdxe_response(t, [100, 100, 100, 40, 100, 100, 100.0]; strict_avg = true).best_avg_response ≈ -12
    @test resp([100, 100, 100, 40, 100, 100, 100.0]).best_avg_response ≈ -15
    @test TP.pdxe_response_paper(t, [100, 80, 50, 4, 3, 2, 2.0]).category == "mCR"
    # unsorted input
    @test TP.pdxe_response(reverse(t), reverse([100, 80, 50, 4, 3, 2, 2.0])).category == "mCR"
    # zero baseline is shifted by +1 (Xeva convention): 0 -> 1, then 100 -> 101 : mPD
    @test TP.pdxe_response([0, 10, 20.0], [0, 100, 100.0]).category == "mPD"
    # Xeva documentation example (mRECIST(time, volume, min.time=10)) -> grows ~146% => PD
    xt = [0, 3, 7, 11, 18, 22, 26, 30, 32, 35.0]
    xv = [250.8, 320.4, 402.3, 382.6, 384, 445.9, 460.2, 546.8, 554.3, 617.9]
    @test TP.pdxe_response(xt, xv).category == "mPD"
    # category helpers
    @test TP.response_score("mCR") > TP.response_score("mPR") > TP.response_score("mSD") > TP.response_score("mPD")
    cats = ["mCR", "mPR", "mPD", "mSD", "NA"]
    @test TP.p_responder(cats) == 0.5
    @test TP.expected_score(cats) ≈ (3 + 2 + 0 + 1) / 4
    @test TP.category_probs(cats).NA ≈ 0.2
end

# ---------------------------------------------------------------- coverage & scores
@testset "coverage and scores" begin
    rng = MersenneTwister(1)
    n, S = 3000, 1000
    sims = randn(rng, S, n)
    y = randn(rng, n)
    @test 0.88 < TP.coverage(y, sims; level = 0.9) < 0.92
    @test 0.93 < TP.coverage(y, sims; level = 0.95) < 0.97
    @test TP.coverage(2 .* y, sims; level = 0.9) < 0.75           # over-dispersed truth under-covered
    w = TP.interval_width(sims[:, 1:50]; level = 0.9)
    @test mean(w) ≈ 2 * 1.645 atol = 0.15
    # grouped coverage: tumour-averaged vs pooled
    groups = repeat(1:30, inner = 100)
    cg = TP.coverage_by_group(y, sims, groups; level = 0.9)
    @test 0.86 < cg.pooled < 0.94 && 0.86 < cg.tumour_avg < 0.94 && length(cg.per_group) == 30
    # CRPS of N(0,1) forecast at y=0 is 2φ(0) - 1/√π
    @test TP.crps([0.0], randn(rng, 20000, 1))[1] ≈ 2 / sqrt(2π) - 1 / sqrt(π) atol = 0.01
    # interval score is minimal-ish: width + penalties
    is = TP.interval_score([0.0, 10.0], hcat(randn(rng, 4000), randn(rng, 4000)); level = 0.9)
    @test is[1] ≈ 2 * 1.645 atol = 0.2
    @test is[2] > 20 * (10 - 1.645) * 0.9
    # log predictive density: mixture over identical draws equals the lognormal density on volume scale
    latent = fill(exp(1.0), 50, 1); σ = fill(0.3, 50)
    yv = 3.0
    lp = TP.log_predictive_density([yv], latent, σ; nu = Inf)
    @test lp[1] ≈ (-0.5 * ((log(yv) - 1) / 0.3)^2 - log(0.3) - 0.5 * log(2π)) - log(yv) atol = 1e-12
    # left-censored: P(y < limit) with limit = exp(1) is 0.5
    @test TP.log_predictive_density([0.0], latent, σ; limit = exp(1.0), nu = Inf)[1] ≈ log(0.5) atol = 1e-6
    @test isnan(TP.log_predictive_density([0.0], latent, σ; nu = Inf)[1])
    # Student-t observation model (default nu = 5): scaled-t density and t CDF for the censored mass
    lt = TP.log_predictive_density([yv], latent, σ)[1]
    @test lt ≈ logpdf(TDist(5.0), (log(yv) - 1) / 0.3) - log(0.3) - log(yv) atol = 1e-10
    @test TP.log_predictive_density([yv], latent, σ; nu = 5.0)[1] == lt
    @test TP.log_predictive_density([yv], latent, σ; nu = 1e9)[1] ≈ lp[1] atol = 1e-4                  # large nu -> Normal
    lim = exp(1.0 - 0.3)                                                                  # one scale unit below the median
    @test TP.log_predictive_density([0.0], latent, σ; limit = lim)[1] ≈ logcdf(TDist(5.0), -1.0) atol = 1e-10
    @test TP.log_predictive_density([0.0], latent, σ; limit = lim, nu = Inf)[1] ≈ logcdf(Normal(), -1.0) atol = 1e-6
    # KDE variant approximates the true density
    noisy = exp.(1.0 .+ 0.3 .* randn(rng, 20000, 1))
    @test TP.log_predictive_density_kde([3.0], noisy)[1] ≈ lp[1] atol = 0.05
    # failed solves: empty ensembles are counted, never silently dropped
    sf = hcat(randn(rng, 100, 3), fill(NaN, 100)); sf[1:5, 2] .= NaN
    fl = TP.ensemble_failures(sf)
    @test fl.n_failed_rows == 100 && fl.n_empty_columns == 1 && fl.n_nonfinite_entries == 105
    @test TP.coverage([0.0, 0.0, 0.0, 0.0], sf; level = 0.99) ≤ 0.75          # the empty column is a miss, not dropped
    @test isnan(TP.crps([0.0, 0.0, 0.0, 0.0], sf)[4]) && isfinite(TP.crps([0.0, 0.0, 0.0, 0.0], sf)[1])
    @test isnan(TP.interval_width(sf)[4])
    lf = TP.log_predictive_density([2.0, 2.0], hcat(fill(2.0, 50), fill(NaN, 50)), fill(0.3, 50); nu = Inf)
    @test isfinite(lf[1]) && lf[2] == -Inf                                      # no valid draw: -Inf kept
    half = fill(2.0, 50); half[1:25] .= NaN                                     # half the draws failed: log(1/2) penalty
    lh = TP.log_predictive_density([2.0], reshape(half, :, 1), fill(0.3, 50); nu = Inf)[1]
    @test lh ≈ TP.log_predictive_density([2.0], fill(2.0, 50, 1), fill(0.3, 50); nu = Inf)[1] + log(0.5)
    # cluster bootstrap of a mean: spread ≈ sd/√ngroups
    gv = repeat(1:40, inner = 10); vals = randn(rng, 400)
    bs = TP.cluster_bootstrap(idx -> mean(vals[idx]), gv; nboot = 400, rng = rng)
    @test std(bs) ≈ std(vals) / sqrt(400) rtol = 0.3
end

# ---------------------------------------------------------------- ranking
@testset "ranking" begin
    @test TP.kendall_tau_b([1, 2, 3], [1, 3, 2]) ≈ 1 / 3
    @test TP.kendall_tau_b(1:6, 1:6) ≈ 1
    @test TP.kendall_tau_b(1:6, 6:-1:1) ≈ -1
    @test isnan(TP.kendall_tau_b([1, 1, 1], [1, 2, 3]))
    rng = MersenneTwister(2)
    for _ in 1:20
        x = rand(rng, 1:5, 40); y = rand(rng, 1:4, 40)
        @test TP.kendall_tau_b(x, y) ≈ StatsBase.corkendall(Float64.(x), Float64.(y)) atol = 1e-12
        @test TP.spearman(x, y) ≈ StatsBase.corspearman(Float64.(x), Float64.(y)) atol = 1e-12
    end
    # sklearn doc example: AUROC = 0.75
    @test TP.auroc([0.1, 0.4, 0.35, 0.8], [false, false, true, true]) ≈ 0.75
    @test TP.auroc([1.0, 1.0], [true, false]) ≈ 0.5
    @test TP.auroc([3, 2, 1, 0], [true, true, false, false]) ≈ 1.0
    @test isnan(TP.auroc([1, 2], [true, true]))
    oa = TP.ordinal_agreement([3.0, 2.0, 1.0, 0.0, 5.0], ["mCR", "mPR", "mSD", "mPD", "NA"])
    @test oa.tau_b ≈ 1 && oa.auroc ≈ 1 && oa.n == 4
    # AUROC ranks by responder_scores, tau_b by the ordinal score (they can disagree)
    ord_s = [3.0, 2.9, 0.1, 0.0]; pr = [0.1, 0.9, 0.5, 0.4]; obs = ["mCR", "mPR", "mSD", "mPD"]
    oa2 = TP.ordinal_agreement(ord_s, obs; responder_scores = pr)
    @test oa2.tau_b ≈ 1 && oa2.auroc ≈ 0.5 && TP.auroc(ord_s, [true, true, false, false]) ≈ 1
    @test TP.auroc(pr, [true, true, false, false]) ≈ 0.5
    # responder fraction posterior
    R = [true false true; true true false; false false false; true true true]
    rf = TP.responder_fraction(R)
    @test rf.mean ≈ mean([2/3, 2/3, 0, 1]) && length(rf.draws) == 4 && rf.n_missing == 0
    Rm = Union{Missing,Bool}[true missing true; true true false]          # failed model-draws are excluded per row
    rfm = TP.responder_fraction(Rm)
    @test rfm.draws ≈ [1.0, 2/3] && rfm.n_missing == 1
    da = TP.drug_ranking_agreement([0.1, 0.5, 0.3], [0.2, 0.9, 0.4])
    @test da.spearman ≈ 1 && da.kendall_tau_b ≈ 1 && da.n == 3
end

# ---------------------------------------------------------------- PSIS-LOO
@testset "loo toy normal" begin
    rng = MersenneTwister(3)
    n = 50; y = 0.7 .+ randn(rng, n)
    S = 4000
    μs = mean(y) .+ randn(rng, S) ./ sqrt(n)               # flat-prior posterior of the mean
    ll = [-0.5 * log(2π) - 0.5 * (y[i] - μs[s])^2 for s in 1:S, i in 1:n]
    # analytic LOO: y_i | y_-i ~ N(ȳ_-i, 1 + 1/(n-1))
    exact = sum(begin
                    m = (sum(y) - y[i]) / (n - 1); v = 1 + 1 / (n - 1)
                    -0.5 * log(2π * v) - 0.5 * (y[i] - m)^2 / v
                end for i in 1:n)
    r = TP.loo(ll)
    @test r.elpd ≈ exact atol = 0.25
    @test r.n == n && r.se > 0
    @test all(r.pareto_k .< 0.7)
    @test sum(values(r.pareto_k_counts)) == n
    @test 0.5 < r.p_eff < 2.0
    # chain_index path (two chains)
    r2 = TP.loo(ll; chain_index = repeat([1, 2], inner = S ÷ 2))
    @test r2.elpd ≈ exact atol = 0.3
    # grouped: leave out blocks of m=5; analytic MVN predictive with cov I + c J, c = 1/(n-m)
    m = 5; groups = repeat(1:(n ÷ m), inner = m)
    gexact = sum(begin
                     idx = groups .== g; yo = y[.!idx]
                     mu = mean(yo); c = 1 / (n - m); res = y[idx] .- mu
                     quad = sum(abs2, res) - c * sum(res)^2 / (1 + m * c)
                     -0.5 * m * log(2π) - 0.5 * log(1 + m * c) - 0.5 * quad
                 end for g in unique(groups))
    rg = TP.loo(ll; groups = groups)
    @test rg.n == n ÷ m
    @test rg.elpd ≈ gexact atol = 0.4
    @test rg.elpd < sum(TP.loo(ll).pointwise_elpd) + 1       # blocks are harder than single points
end

# ---------------------------------------------------------------- simulation
function toy_pd(; S = 40, ω = 0.0, σ = 0.1)
    doses = dosing_schedule(10.0; start = 0.0, interval = 1.0, n = 14)
    pk = OneCmt(CL = 20.0, V = 5.0, ka = 8.0, F = 0.8)
    cref = maximum(conc(pk, doses, 0:0.1:14))
    PD("toy", ["A", "B"], doses, fill(pk, S), cref,
       fill(log(0.15), S), fill(ω, S), fill(log(0.6), S), fill(ω, S),
       fill(log(2.0), S), fill(log(0.5), S), fill(σ, S),
       hcat(fill(0.15, S), fill(0.10, S)), hcat(fill(0.6, S), fill(1.5, S)),
       [100.0, 150.0], [120.0, 160.0], Dict{String,Any}("nu" => Inf))      # Normal noise unless a test sets nu
end

@testset "simulate_regimen" begin
    pd = toy_pd()
    tg = collect(0.0:2.0:28.0)
    # known model, deterministic draws, no noise == direct simulate_tumour
    pk = pd.pk[1]
    expo = PKExposure(pk, pd.doses, pd.cref)
    θ = Simeoni(0.15, 2.0, 0.5, 0.6, 120.0)
    wref, ok = simulate_tumour(θ, expo, tg; reltol = 1e-6, abstol = 1e-6)
    @test ok
    sims = TP.simulate_regimen(pd, pd.doses; model = "A", tgrid = tg, ndraws = 10, obs_noise = false)
    @test size(sims) == (10, length(tg))
    for r in 1:10
        @test sims[r, :] ≈ wref rtol = 1e-10
    end
    # model B uses its own λ0 / k2 / baseline
    θB = Simeoni(0.10, 2.0, 0.5, 1.5, 160.0)
    wB, _ = simulate_tumour(θB, expo, tg)
    @test TP.simulate_regimen(pd, pd.doses; model = "B", tgrid = tg, ndraws = 3, obs_noise = false)[1, :] ≈ wB rtol = 1e-10
    # explicit w0 override and a different regimen (half the doses): exposure scales through same cref
    half = pd.doses[1:7]
    w_half, _ = simulate_tumour(Simeoni(0.15, 2.0, 0.5, 0.6, 50.0), PKExposure(pk, half, pd.cref), tg)
    @test TP.simulate_regimen(pd, half; model = "A", w0 = 50.0, tgrid = tg, ndraws = 2, obs_noise = false)[2, :] ≈ w_half rtol = 1e-10
    # control
    wc, _ = simulate_tumour(Simeoni(0.15, 2.0, 0.5, 0.6, 100.0), NoDrug(), tg)
    @test TP.simulate_control(pd; model = "A", tgrid = tg, ndraws = 2, obs_noise = false)[1, :] ≈ wc rtol = 1e-10
    @test all(sims[:, end] .< TP.simulate_control(pd; model = "A", w0 = 120.0, tgrid = tg, ndraws = 2, obs_noise = false)[1, end])
    # observation noise: baseline untouched, log-residual sd ≈ σ
    pdn = toy_pd(S = 4000, σ = 0.2)
    sn = TP.simulate_regimen(pdn, pdn.doses; model = "A", tgrid = tg, ndraws = 4000, obs_noise = true, rng = MersenneTwister(5))
    @test all(sn[:, 1] .== 120.0)
    @test std(log.(sn[:, 5]) .- log(wref[5])) ≈ 0.2 rtol = 0.06
    # reproducible with a seeded rng, errors on bad args
    a = TP.simulate_regimen(pdn, pdn.doses; model = "A", tgrid = tg, ndraws = 50, rng = MersenneTwister(9))
    b = TP.simulate_regimen(pdn, pdn.doses; model = "A", tgrid = tg, ndraws = 50, rng = MersenneTwister(9))
    @test a == b
    # Student-t observation noise (fit's nu; default 5 when the fit does not record it)
    pdt5 = toy_pd(S = 6000, σ = 0.2); pdt5.diagnostics["nu"] = 5.0
    @test TP.nu_from_fit(pdt5) == 5.0 && TP.nu_from_fit(pd) == Inf
    delete!(pdt5.diagnostics, "nu"); @test TP.nu_from_fit(pdt5) == 5.0
    st = TP.simulate_regimen(pdt5, pdt5.doses; model = "A", tgrid = tg, ndraws = 6000, rng = MersenneTwister(8))
    zt = (log.(st[:, 5]) .- log(wref[5])) ./ 0.2
    @test std(zt) ≈ sqrt(5 / 3) rtol = 0.1                                      # t5 has variance 5/3
    @test count(abs.(zt) .> 4) / 6000 > 0.003                                    # heavy tails (Normal: 6e-5)
    @test all(st[:, 1] .== 120.0)
    @test_throws ArgumentError TP.simulate_regimen(pd, pd.doses; tgrid = tg)
    @test_throws ArgumentError TP.simulate_regimen(pd, pd.doses; model = "A", newpdx = true, tgrid = tg)
    @test_throws ArgumentError TP.simulate_regimen(pd, pd.doses; model = "Z", tgrid = tg)
    # new PDX: ω = 0 collapses to the population median; ω > 0 spreads
    pd0 = toy_pd(ω = 0.0)
    s0 = TP.simulate_regimen(pd0, pd0.doses; newpdx = true, tgrid = tg, ndraws = 5, obs_noise = false)
    # new-PDX baselines are the draw-matched baselines of randomly chosen training models (A: 120, B: 160)
    for r in 1:5
        w0r = s0[r, 1]
        @test w0r in (120.0, 160.0)
        wpop, _ = simulate_tumour(Simeoni(0.15, 2.0, 0.5, 0.6, w0r), expo, tg)
        @test s0[r, :] ≈ wpop rtol = 1e-10
    end
    @test length(unique(TP.simulate_regimen(pd0, pd0.doses; newpdx = true, tgrid = tg, ndraws = 40, obs_noise = false, rng = MersenneTwister(2))[:, 1])) == 2
    pdw = toy_pd(S = 300, ω = 0.5)
    sw = TP.simulate_regimen(pdw, pdw.doses; newpdx = true, tgrid = tg, ndraws = 300, obs_noise = false, rng = MersenneTwister(1))
    @test std(log.(sw[:, end])) > 0.1
    # details
    d = TP.simulate_regimen(pdn, pdn.doses; model = "A", tgrid = tg, ndraws = 20, details = true)
    @test length(d.idx) == 20 && d.σ == fill(0.2, 20)
    # common draw indices: row s is the same stored draw for every PDX model
    pdd = toy_pd(S = 30, ω = 0.5)
    pdd.σ .= range(0.05, 0.3, length = 30)                                  # make draws distinguishable
    da = TP.simulate_regimen(pdd, pdd.doses; model = "A", tgrid = tg, draws = [3, 7, 11], details = true)
    db = TP.simulate_regimen(pdd, pdd.doses; model = "B", tgrid = tg, draws = [3, 7, 11], details = true)
    @test da.idx == [3, 7, 11] == db.idx && da.σ == db.σ == pdd.σ[[3, 7, 11]]
    @test_throws ArgumentError TP.simulate_regimen(pdd, pdd.doses; model = "A", tgrid = tg, draws = [0, 1])
    # same seed: noisy and latent calls share the new-PDX random effects (noise is drawn after them)
    n1 = TP.simulate_regimen(pdd, pdd.doses; newpdx = true, tgrid = tg, draws = 1:30, rng = MersenneTwister(4), obs_noise = true)
    l1 = TP.simulate_regimen(pdd, pdd.doses; newpdx = true, tgrid = tg, draws = 1:30, rng = MersenneTwister(4), obs_noise = false)
    @test all(isapprox.(n1[:, 1], l1[:, 1])) && cor(log.(n1[:, end]), log.(l1[:, end])) > 0.9
    # metadata helpers
    pdm = toy_pd(S = 40)
    pdm.diagnostics["settings"] = Dict{String,Any}("nchains" => 4, "lod" => 2.5)
    @test TP.chain_index_from_fit(pdm) == repeat(1:4, inner = 10)
    @test TP.detection_limit_from_fit(pdm) == 2.5
    pdm.diagnostics["lod"] = 3.0; pdm.diagnostics["nchains"] = 5
    @test TP.detection_limit_from_fit(pdm) == 3.0                              # top-level key wins
    @test TP.chain_index_from_fit(pdm) == repeat(1:5, inner = 8)
    pdm.diagnostics["nchains"] = 3
    @test_throws ArgumentError TP.chain_index_from_fit(pdm)                    # 40 draws not divisible by 3
    delete!(pdm.diagnostics, "lod"); delete!(pdm.diagnostics, "settings")
    @test TP.detection_limit_from_fit(pdm) in (4.0, float(TP.detection_limit()))
    # response categories default to the paper variant; failed rows give "NA"
    sc = copy(sn[1:3, :]); sc[2, 4] = NaN
    @test TP.response_categories(tg, sc)[2] == "NA"
    @test TP.response_categories(tg, sc)[1] == TP.pdxe_response_paper(tg, sc[1, :]).category
    # bands
    bands = TP.predictive_bands(sn)
    @test size(bands) == (length(tg), 3)
    @test all(bands[:, 1] .<= bands[:, 2] .<= bands[:, 3])
    # coverage of the model's own predictive: truth drawn from the predictive
    ytrue = TP.simulate_regimen(pdn, pdn.doses; model = "A", tgrid = tg[2:end], ndraws = 1, rng = MersenneTwister(77))[1, :]
    cov_list = Float64[]
    for k in 1:200
        yk = TP.simulate_regimen(pdn, pdn.doses; model = "A", tgrid = tg[2:end], ndraws = 1, rng = MersenneTwister(1000 + k))
        push!(cov_list, TP.coverage(vec(yk), TP.simulate_regimen(pdn, pdn.doses; model = "A", tgrid = tg[2:end], ndraws = 800, rng = MersenneTwister(5))))
    end
    @test 0.85 < mean(cov_list) < 0.95
    # response categories from simulated draws
    cats = TP.response_categories(tg, sn)
    @test length(cats) == 4000 && all(c in ("mCR", "mPR", "mSD", "mPD") for c in cats)
end

# ---------------------------------------------------------------- splits
@testset "splits" begin
    struct _C; model::String; tumour_type::String; treatment::String; t::Vector{Float64}; v::Vector{Float64}; end
    c = _C("X", "BRCA", "drug", collect(0.0:3.0:30.0), collect(1.0:11.0))       # n = 11
    tr, te = TP.split_late(c)
    @test length(te.t) == 3 && length(tr.t) == 8                                # round(0.3*11) = 3
    @test tr.t[end] < te.t[1] && vcat(tr.t, te.t) == c.t && tr.t[1] == 0
    @test TP.split_late_indices(4) == ([1], [2, 3, 4]) || TP.split_late_indices(4) == ([1, 2, 3], [4])
    @test TP.split_late_indices(2) == ([1, 2], Int[])                            # too short to split
    ids = ["M$(i)" for i in 1:50]; ty = repeat(["BRCA", "CRC", "NSCLC", "PAN", "GC"], 10)
    tr, te = TP.split_models(ids; frac = 0.2, seed = 4, strata = ty)
    @test length(te) == 10 && length(tr) == 40 && isempty(intersect(tr, te))
    for s in unique(ty)                                                       # 2 held out per stratum
        @test count(m -> ty[findfirst(==(m), ids)] == s, te) == 2
    end
    p = randperm(MersenneTwister(1), 50)
    @test TP.split_models(ids[p]; frac = 0.2, seed = 4, strata = ty[p]) == (tr, te)   # order independent
    @test TP.split_models(ids; frac = 0.2, seed = 5, strata = ty)[2] != te
    # inconsistent stratum labels for one PDX are rejected (it could land in both train and test)
    @test_throws ArgumentError TP.split_models(["A", "A", "B", "C"]; strata = ["x", "y", "x", "y"])
    trm, tem = TP.split_models(["A", "A", "B", "C"]; strata = ["x", "y", "x", "y"], on_conflict = :majority, frac = 0.5, seed = 1)
    @test isempty(intersect(trm, tem)) && sort(vcat(trm, tem)) == ["A", "B", "C"]
    units = [_C(ids[i], ty[i], "d", [0.0], [1.0]) for i in 1:50]
    @test TP.split_models(units; frac = 0.2, seed = 4) == (tr, te)
end

# ---------------------------------------------------------------- real-data classifier cross-check
@testset "PDXE classifier vs paper (optional)" begin
    cache = joinpath(ROOT, "data", "cache")
    f1 = joinpath(cache, "pct_long.csv"); f2 = joinpath(cache, "pct_metrics.csv")
    if isfile(f1) && isfile(f2)
        include(joinpath(ROOT, "test", "pdxe_crosscheck.jl"))
        res = pdxe_crosscheck(cache)                                     # literal rule
        @test res.n_compared > 1000
        @test res.agreement > 0.85
        resp = pdxe_crosscheck(cache; strict_avg = true, strict_best = true, fallback_all = true)  # = pdxe_response_paper
        @test resp.agreement > 0.9 && resp.responder_agreement > 0.99
    else
        @info "pct_long.csv / pct_metrics.csv not found; skipping classifier cross-check"
    end
end


# ---------------------------------------------------------------- Phase 2: baselines, what-ifs, virtual trial
function synth_pd(; S = 40, μk = log(0.2), ωk = 0.6, μl = log(0.15), ωl = 0.3, σ = 0.15, nm = 30, seed = 1,
                  diag = Dict{String,Any}("settings" => Dict{String,Any}("lod" => 4.0)))
    rng = MersenneTwister(seed)
    doses = dosing_schedule(10.0; start = 0.0, interval = 1.0, n = 28)
    pk = OneCmt(CL = 20.0, V = 5.0, ka = 8.0, F = 0.8)
    cref = maximum(conc(pk, doses, 0:0.1:7))
    w0 = 180 .* exp.(0.25 .* randn(rng, nm))
    PD("syn", ["m$i" for i in 1:nm], doses, fill(pk, S), cref, fill(μl, S), fill(ωl, S), fill(μk, S), fill(ωk, S),
       fill(log(8.0), S), fill(log(0.4), S), fill(σ, S), ones(S, nm), ones(S, nm), w0, w0, diag), doses
end

@testset "draw-matched baselines" begin
    S = 20
    base, doses = synth_pd(S = S, nm = 3)
    wt = [100.0 + s + 1000 * j for s in 1:S, j in 1:3]          # distinct per draw and model
    wc = wt .+ 0.5
    pd = PD(base.drug, base.models, base.doses, base.pk, base.cref, base.μ_λ0, base.ω_λ0, base.μ_k2, base.ω_k2,
            base.logλ1, base.logk1, base.σ, base.λ0, base.k2, vec(mean(wc; dims = 1)), vec(mean(wt; dims = 1)),
            wc, wt, base.diagnostics)
    tg = [0.0, 5.0, 10.0]
    r = TP.simulate_regimen(pd, doses; model = "m2", tgrid = tg, draws = [3, 7], obs_noise = false, details = true)
    @test r.y[:, 1] == [2103.0, 2107.0] && r.w0 == r.y[:, 1]
    c = TP.simulate_control(pd; model = "m3", tgrid = tg, draws = [3, 7], obs_noise = false)
    @test c[:, 1] == [3103.5, 3107.5]
    # new PDX: baseline is the draw-matched value of some training model (all columns for that draw are candidates)
    n = TP.simulate_regimen(pd, doses; newpdx = true, tgrid = tg, draws = repeat([3], 50), obs_noise = false, rng = MersenneTwister(1))
    @test all(w -> w in (1103.0, 2103.0, 3103.0), n[:, 1]) && length(unique(n[:, 1])) == 3
    # restricted pool and explicit override
    n2 = TP.simulate_regimen(pd, doses; newpdx = true, tgrid = tg, draws = repeat([3], 20), obs_noise = false,
                             baseline_models = ["m2"], rng = MersenneTwister(1))
    @test all(==(2103.0), n2[:, 1])
    n3 = TP.simulate_regimen(pd, doses; newpdx = true, tgrid = tg, draws = repeat([3], 4), obs_noise = false, w0 = [1.0, 2.0, 3.0, 4.0])
    @test n3[:, 1] == [1.0, 2.0, 3.0, 4.0]
    @test_throws DimensionMismatch TP.simulate_regimen(pd, doses; newpdx = true, tgrid = tg, draws = [1, 2], w0 = [1.0, 2.0, 3.0])
    @test_throws ArgumentError TP.simulate_regimen(pd, doses; newpdx = true, tgrid = tg, ndraws = 3, baseline_models = ["zzz"])
end

@testset "what-if regimens" begin
    doses = dosing_schedule(30.0; start = 0.0, interval = 1.0, n = 40)
    d2 = TP.scaled_regimen(doses; multiplier = 2.0)
    @test all(d.amt == 60.0 for d in d2) && [d.time for d in d2] == [d.time for d in doses] && doses[1].amt == 30.0
    m = TP.make_regimen(amt_multiplier = 0.5, interval_days = 2.0, duration_days = 11.0, ref_amt = 30.0)
    @test [d.time for d in m] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0] && all(d.amt == 15.0 for d in m) && all(d.route === :oral for d in m)
    @test length(TP.make_regimen(interval_days = 3.0, n_doses = 4, start = 2.0, ref_amt = 10.0, route = :iv)) == 4
    @test TP.make_regimen(interval_days = 3.0, n_doses = 4, start = 2.0, ref_amt = 10.0)[1].time == 2.0
    @test_throws ArgumentError TP.make_regimen(interval_days = 1.0, ref_amt = 1.0)                       # neither n nor duration
    @test_throws ArgumentError TP.make_regimen(interval_days = 1.0, n_doses = 3, duration_days = 3.0, ref_amt = 1.0)
    # extrapolation flags
    ex(d; kw...) = TP.is_extrapolation(doses, d; kw...)
    @test ex(doses) == (false, String[])
    @test ex(doses[1:28]; horizon = 28.0) == (false, String[])                      # same schedule, shorter record is fine
    f, why = ex(d2); @test f && any(occursin("dose amount", x) for x in why) && any(occursin("one dose level", x) for x in why)
    f, why = ex(TP.make_regimen(interval_days = 2.0, n_doses = 20, ref_amt = 30.0)); @test f && any(occursin("interval", x) for x in why)
    f, why = ex(dosing_schedule(30.0; n = 40, interval = 1.0, route = :iv)); @test f && any(occursin("route", x) for x in why)
    f, why = ex(dosing_schedule(30.0; n = 5, interval = 1.0); horizon = 21.0); @test f && any(occursin("stops", x) for x in why)
    f, why = ex(doses; horizon = 60.0, observed_max_day = 45.0); @test f && any(occursin("follow-up", x) for x in why)
    @test ex(doses; horizon = 40.0, observed_max_day = 45.0)[1] == false
    f, why = ex(dosing_schedule(30.0; start = 3.0, n = 40, interval = 1.0)); @test f && any(occursin("first dose", x) for x in why)
    # dose-event comparison: holidays, omissions, shifts and missed final doses must all be flagged
    sched = dosing_schedule(30.0; n = 40, interval = 1.0)
    exs(d) = TP.is_extrapolation(sched, d; horizon = 28.0)
    hol = TP.make_regimen(interval_days = 1.0, days_on = 5, days_off = 2, duration_days = 28.0, ref_amt = 30.0)
    @test [d.time for d in hol[1:8]] == [0.0, 1, 2, 3, 4, 7, 8, 9] && length(hol) == 20
    f, why = exs(hol); @test f && any(occursin("holiday", x) && occursin("omitted", x) for x in why) && !any(occursin("interval differs", x) for x in why)
    omit = [d for (i, d) in enumerate(sched[1:28]) if i != 11]                       # day 10 skipped
    f, why = exs(omit); @test f && any(occursin("omitted", x) && occursin("10", x) for x in why)
    shifted = [Dose(i == 11 ? 10.5 : d.time, d.amt, d.route) for (i, d) in enumerate(sched[1:28])]
    f, why = exs(shifted); @test f && any(occursin("shifted", x) for x in why)
    f, why = exs(sched[1:25]); @test f && any(occursin("stops after day 24", x) && occursin("25", x) for x in why)   # missed final doses
    f, why = exs(vcat(sched[1:28], [Dose(3.5, 30.0, :oral)])); @test f && any(occursin("where the reference has none", x) for x in why)
    bid = TP.make_regimen(interval_days = 0.5, amt_multiplier = 0.5, duration_days = 28.0, ref_amt = 30.0)
    @test length(bid) == 56 && exs(bid)[1]                                           # BID at half amount per administration
    @test exs(sched[1:28]) == (false, String[])
    # make_regimen holiday schedules
    @test_throws ArgumentError TP.make_regimen(interval_days = 1.0, days_off = 2.0, n_doses = 5, ref_amt = 1.0)
    @test length(TP.make_regimen(interval_days = 1.0, days_on = 5, days_off = 2, n_doses = 12, ref_amt = 1.0)) == 12
    @test [d.time for d in TP.make_regimen(interval_days = 1.0, days_on = 5, days_off = 2, n_doses = 7, ref_amt = 1.0)] == [0.0, 1, 2, 3, 4, 7, 8]
    bidh = TP.make_regimen(interval_days = 0.5, days_on = 2, days_off = 1, duration_days = 6.0, ref_amt = 1.0)
    @test [d.time for d in bidh] == [0.0, 0.5, 1.0, 1.5, 3.0, 3.5, 4.0, 4.5]
    @test [d.time for d in TP.make_regimen(interval_days = 1.0, days_on = 3, days_off = 0, n_doses = 5, start = 2.0, ref_amt = 1.0)] == [2.0, 3, 4, 5, 6]
    # grid
    @test TP.pdxe_like_grid(3.5) == [0.0, 3.5]
    @test TP.pdxe_like_grid(28.0) == [0.0, 4, 7, 10, 14, 18, 21, 24, 28]
    for h in (3.5, 10.2, 13.9, 28.0, 41.3)
        g = TP.pdxe_like_grid(h); @test g[1] == 0 && g[end] == h && issorted(g) && allunique(g)
    end
    pd, _ = synth_pd(diag = Dict{String,Any}("max_observed_day" => 50.0))
    @test TP.is_extrapolation(pd, pd.doses; horizon = 28.0) == (false, String[])
    @test TP.is_extrapolation(pd, pd.doses; horizon = 80.0)[1]
    pdn, _ = synth_pd()
    @test TP.is_extrapolation(pdn, pdn.doses; horizon = 28.0)[1]                    # unknown follow-up is flagged, conservatively
    @test !TP.is_extrapolation(pdn, pdn.doses)[1]
end

@testset "virtual PDX trial" begin
    pd, doses = synth_pd(S = 40, diag = Dict{String,Any}("settings" => Dict{String,Any}("lod" => 4.0), "max_observed_day" => 40.0,
                         "median_followup_day" => 28.0))
    # --- validation: predicted responder rate matches the generating truth (independent direct simulation)
    hz = 28.0; tg = TP.pdxe_like_grid(hz)
    @test tg[1] == 0.0 && tg[end] == hz && issorted(tg) && 3 < length(tg) < 15
    rng = MersenneTwister(99)
    N = 6000; hit = 0; cat_counts = zeros(Int, 4)
    pk = pd.pk[1]; ex_ = PKExposure(pk, [d for d in doses if d.time <= hz], pd.cref)
    for _ in 1:N
        λ0 = exp(pd.μ_λ0[1] + pd.ω_λ0[1] * randn(rng)); k2 = exp(pd.μ_k2[1] + pd.ω_k2[1] * randn(rng))
        w0 = pd.w0_treated[rand(rng, 1:length(pd.models))]
        w, ok = simulate_tumour(Simeoni(λ0, exp(pd.logλ1[1]), exp(pd.logk1[1]), k2, w0), ex_, tg)
        v = [t == 0 ? w0 : (x = max(w[i], 0.0) * exp(pd.σ[1] * rand(rng, TDist(5.0))); x < 4.0 ? 0.0 : x) for (i, t) in enumerate(tg)]
        c = TP.pdxe_response_paper(tg, v).category
        hit += TP.is_responder(c); cat_counts[findfirst(==(c), TP.RESPONSE_CATEGORIES)] += 1
    end
    truth = hit / N
    @test 0.2 < truth < 0.7                                                            # an informative, intermediate rate
    t = TP.simulate_trial(pd, doses; n_pdx = 200, ndraws = 40, rng = MersenneTwister(1))
    se = sqrt(truth * (1 - truth) / N + truth * (1 - truth) / (200 * 40))
    @test abs(t.summary.responder_rate.median - truth) < 4 * se + 0.01
    @test all(abs.([t.summary.mCR.median, t.summary.mPR.median, t.summary.mSD.median, t.summary.mPD.median] .- cat_counts ./ N) .< 0.04)
    # --- structure
    @test length(t.responder_rate) == 40 && size(t.fractions) == (40, 4) && size(t.categories) == (200, 40) && t.n_na == 0
    @test all(isapprox.(sum(t.fractions; dims = 2), 1.0)) && t.responder_rate ≈ t.fractions[:, 1] .+ t.fractions[:, 2]
    @test t.summary.responder_rate.lo ≤ t.summary.responder_rate.median ≤ t.summary.responder_rate.hi
    @test t.horizon == 28.0 && t.tgrid == tg && t.extrapolation == (false, String[])
    # reproducible, same result for the same rng; default horizon from the fit's median follow-up
    t2 = TP.simulate_trial(pd, doses; n_pdx = 200, ndraws = 40, rng = MersenneTwister(1))
    @test t2.responder_rate == t.responder_rate && t2.categories == t.categories
    # quick mode caps the work
    tq = TP.simulate_trial(pd, doses; n_pdx = 500, ndraws = 300, quick = true, rng = MersenneTwister(1))
    @test tq.ndraws == 40 && tq.n_pdx == 100                                           # 40 stored draws < 50 cap
    pdb, _ = synth_pd(S = 120)
    tq2 = TP.simulate_trial(pdb, doses; n_pdx = 500, ndraws = 300, quick = true, rng = MersenneTwister(1))
    @test tq2.ndraws == 50 && tq2.n_pdx == 100
    # --- dose response is monotone and flagged as extrapolation
    rates = [TP.simulate_trial(pd, TP.scaled_regimen(doses; multiplier = m); n_pdx = 300, ndraws = 30, rng = MersenneTwister(5)).summary.responder_rate.median
             for m in (0.25, 1.0, 4.0)]
    @test rates[1] < rates[2] < rates[3]
    t4 = TP.simulate_trial(pd, TP.scaled_regimen(doses; multiplier = 4.0); n_pdx = 50, ndraws = 10, rng = MersenneTwister(5))
    @test t4.extrapolation[1] && any(occursin("dose amount", r) for r in t4.extrapolation[2])
    # untreated-like (zero dose): essentially no responders
    t0 = TP.simulate_trial(pd, TP.scaled_regimen(doses; multiplier = 0.0); n_pdx = 100, ndraws = 10, rng = MersenneTwister(5))
    @test t0.summary.responder_rate.median < 0.03
    # --- tumour_type needs fit metadata; restricts baselines
    @test_throws ArgumentError TP.simulate_trial(pd, doses; tumour_type = "BRCA", n_pdx = 5, ndraws = 3)
    types = vcat(fill("BRCA", 10), fill("CRC", 20))
    pdt, _ = synth_pd(diag = Dict{String,Any}("tumour_types" => types, "settings" => Dict{String,Any}("lod" => 4.0)))
    @test_throws ArgumentError TP.simulate_trial(pdt, doses; tumour_type = "GC", n_pdx = 5, ndraws = 3)
    tt = TP.simulate_trial(pdt, doses; tumour_type = "BRCA", n_pdx = 20, ndraws = 5, rng = MersenneTwister(3))
    @test tt.tumour_type == "BRCA" && size(tt.categories) == (20, 5)
    # --- fractional horizon is kept exactly
    th = TP.simulate_trial(pd, doses; n_pdx = 10, ndraws = 3, horizon = 3.5, rng = MersenneTwister(2))
    @test th.horizon == 3.5 && th.tgrid == [0.0, 3.5]
    # --- interval meaning: finite-trial predictive interval vs population credible interval
    @test occursin("posterior predictive interval for a trial of 200 PDX", t.interval_kind) && !t.population
    wid(x) = x.hi - x.lo
    tr_or = TP.simulate_trial(pd, doses; n_pdx = 100, ndraws = 20, rng = MersenneTwister(7))
    po_or = TP.simulate_trial(pd, doses; population = true, n_pdx_population = 800, ndraws = 20, rng = MersenneTwister(7))
    @test po_or.population && po_or.n_pdx == 800 && occursin("population response probability", po_or.interval_kind)
    @test wid(po_or.summary.responder_rate) < 0.7 * wid(tr_or.summary.responder_rate)       # oracle: only simulation error remains
    @test abs(po_or.summary.responder_rate.median - truth) < 0.04
    @test mean(po_or.mc_se) ≈ sqrt(truth * (1 - truth) / 800) rtol = 0.15
    @test wid(po_or.summary.responder_rate) < 4 * 1.645 * mean(po_or.mc_se)                 # width explained by Monte Carlo SE
    pdsp, _ = synth_pd(S = 20)
    pdsp.μ_k2 .= range(log(0.1), log(0.4), length = 20)                                      # real posterior spread in potency
    po_sp = TP.simulate_trial(pdsp, doses; population = true, n_pdx_population = 800, ndraws = 20, rng = MersenneTwister(7))
    @test wid(po_sp.summary.responder_rate) > 2.5 * wid(po_or.summary.responder_rate)
    @test TP.simulate_trial(pd, doses; population = true, quick = true, rng = MersenneTwister(1), ndraws = 5).n_pdx == 400
    # --- compare_trials: common random numbers, paired contrasts
    cmp = TP.compare_trials(pd, Dict("ref" => doses, "double" => TP.scaled_regimen(doses; multiplier = 2.0),
                                     "same" => copy(doses)); n_pdx = 100, ndraws = 20, baseline = "ref")
    @test [r.name for r in cmp.table] == ["double", "ref", "same"]
    @test cmp.results["ref"].responder_rate == cmp.results["same"].responder_rate           # identical regimen, identical draws
    @test cmp.contrasts["same"].median == 0.0 && cmp.contrasts["double"].median > 0
    @test cmp.table[1].extrapolation && !cmp.table[2].extrapolation
    @test_throws ArgumentError TP.compare_trials(pd, Dict("a" => doses); baseline = "zzz", n_pdx = 5, ndraws = 3)
    @test occursin("posterior predictive interval", cmp.contrast_kind)
    cmpp = TP.compare_trials(pd, Dict("ref" => doses, "double" => TP.scaled_regimen(doses; multiplier = 2.0));
                             n_pdx_population = 300, ndraws = 6, population = true, baseline = "ref")
    @test occursin("population response probability", cmpp.contrast_kind) && cmpp.contrasts["double"].median > 0
end

# ---------------------------------------------------------------- Phase 3: fit window, convergence summary
@testset "fit window and convergence helpers" begin
    base = Dict{String,Any}("settings" => Dict{String,Any}("lod" => 4.0), "fit_window_day" => 42.0,
                            "max_observed_day" => 42.0, "median_followup_day" => 50.0)
    pd, doses = synth_pd(diag = copy(base))
    @test TP.fit_window_day(pd) == 42.0
    # default horizon = min(median follow-up, fit window); no flag inside the window
    t = TP.simulate_trial(pd, doses; n_pdx = 20, ndraws = 4, rng = MersenneTwister(1))
    @test t.horizon == 42.0 && t.extrapolation == (false, String[])
    pd21, _ = synth_pd(diag = merge(base, Dict("fit_window_day" => 21.0, "median_followup_day" => 28.0)))
    t21 = TP.simulate_trial(pd21, doses; n_pdx = 20, ndraws = 4, rng = MersenneTwister(1))
    @test t21.horizon == 21.0 && !t21.extrapolation[1]
    # an explicit horizon beyond the window runs but is flagged
    tb = TP.simulate_trial(pd21, doses; n_pdx = 20, ndraws = 4, horizon = 35.0, rng = MersenneTwister(1))
    @test tb.horizon == 35.0 && tb.extrapolation[1] && any(occursin("beyond the fitted 21-day window", r) for r in tb.extrapolation[2])
    @test TP.is_extrapolation(pd21, pd21.doses; horizon = 21.0) == (false, String[])
    # full-follow-up fits (window 0) never trigger it
    pd0, _ = synth_pd(diag = merge(base, Dict("fit_window_day" => 0.0)))
    @test TP.fit_window_day(pd0) === nothing && !TP.is_extrapolation(pd0, pd0.doses; horizon = 30.0)[1]
    # latent baseline draws
    pdl, _ = synth_pd(diag = Dict{String,Any}("baseline_sd" => 0.1))
    wb = TP.latent_baseline(pdl, 200.0, 20000; rng = MersenneTwister(2))
    @test isapprox(std(log.(wb)), 0.1; rtol = 0.05) && isapprox(mean(log.(wb)), log(200.0); atol = 0.01)
    @test TP.baseline_sd_from_fit(pd0) == 0.1                                          # default

    # convergence summary: iid draws in 4 chains -> R-hat ~ 1, ESS ~ draws; recorded items are passed through
    rng = MersenneTwister(5); S = 400
    dg = Dict{String,Any}("nchains" => 4, "max_rhat" => 1.004, "min_ess_bulk" => 310.0, "n_divergences" => 0,
                          "min_ebfmi" => 0.8, "max_tree_depth_hits" => 2, "cross_mode_rhat" => 1.01,
                          "map_lps" => [1.0, 2.0], "rhat_other_mode_check" => 1.2, "train_models" => ["a"], "mode_masses" => [0.5, 0.5], "settings" => Dict{String,Any}("lod" => 4.0))
    b, _ = synth_pd(S = S, diag = dg)
    b.μ_λ0 .= randn(rng, S); b.ω_λ0 .= exp.(0.1 .* randn(rng, S)); b.μ_k2 .= randn(rng, S); b.ω_k2 .= exp.(0.1 .* randn(rng, S))
    b.logλ1 .= randn(rng, S); b.logk1 .= randn(rng, S); b.σ .= exp.(0.1 .* randn(rng, S))
    cs = TP.convergence_summary(b)
    @test cs.nchains == 4 && cs.n_draws == S && cs.max_rhat == 1.004 && cs.ebfmi == 0.8 && cs.cross_mode_rhat == 1.01
    @test cs.recomputed !== nothing && length(cs.recomputed.names) == 7 && all(0.98 .< cs.recomputed.rhat .< 1.03)
    @test all(cs.recomputed.ess_bulk .> 0.5 * S) && all(cs.recomputed.ess_tail .> 0.3 * S)
    @test haskey(cs.extra, "rhat_other_mode_check") && !haskey(cs.extra, "train_models") && haskey(cs.extra, "mode_masses")
    # chains that sit in different places give a large R-hat; a missing item is `nothing`
    b.μ_k2 .= repeat(1:4, inner = S ÷ 4) .+ 0.01 .* randn(rng, S)
    @test maximum(TP.convergence_summary(b).recomputed.rhat) > 2
    cm = TP.convergence_summary(synth_pd(diag = Dict{String,Any}())[1])
    @test cm.ebfmi === nothing && cm.cross_mode_rhat === nothing && cm.nchains == 1
end

@testset "public output guard" begin
    ok_bins = [(arm = :treated, lo = 0.0, hi = 14.0, n_pdx = 12), (arm = :control, lo = 0.0, hi = 14.0, n_pdx = 10)]
    @test TP.PUBLIC_MIN_PDX == 10
    @test isempty(TP.public_output_violations(bins = ok_bins, summaries = [(n_pdx = 25,)], text = "aggregated text only",
                                              ids = ["X-007", "X-010"], figures = ["docs/fig_d_logratio.png", "docs/fig_d_calibration.png", "docs/fig_d_responsecal.png"]))
    # small bins and summaries
    v = TP.public_output_violations(bins = [(arm = :treated, lo = 0.0, hi = 7.0, n_pdx = 9)], summaries = [(n_pdx = 3,)])
    @test length(v) == 2 && any(occursin("9 PDX", x) for x in v) && any(occursin("3 PDX", x) for x in v)
    # ids as whole tokens only
    v = TP.public_output_violations(text = "X-007 treated at day 14", ids = ["X-007"])
    @test length(v) == 1 && occursin("X-007", v[1])
    @test isempty(TP.public_output_violations(text = "group X-0071 and X-0070", ids = ["X-007"]))
    @test !isempty(TP.public_output_violations(text = "(see X-007)", ids = ["X-007"]))
    # individual-level figures
    v = TP.public_output_violations(figures = ["docs/fig_BKM120_bands.png", "docs/fig_BKM120_response.png", "docs/fig_BKM120_responsecal.png"])
    @test length(v) == 2
    @test_throws ErrorException TP.assert_public_output(bins = [(arm = :treated, lo = 0.0, hi = 7.0, n_pdx = 2)])
    @test TP.assert_public_output(bins = ok_bins) === nothing
    @test TP.public_output_violations(summaries = [(n_pdx = 4,)], min_pdx = 3) == String[]
end

# ---------------------------------------------------------------- v2 fits: correlated effects, per-PDX Vtr, waning
@testset "v2 model and waning" begin
    # ---- v1 is unchanged: a "model" => "v1" key gives exactly the stream without the key
    pd, doses = synth_pd(S = 8, nm = 5)
    tg = [0.0, 7.0, 14.0, 28.0]
    a = TP.simulate_regimen(pd, doses; newpdx = true, tgrid = tg, ndraws = 8, rng = MersenneTwister(3))
    pd.diagnostics["model"] = "v1"; pd.diagnostics["waning"] = false
    b = TP.simulate_regimen(pd, doses; newpdx = true, tgrid = tg, ndraws = 8, rng = MersenneTwister(3))
    @test a == b
    d1 = TP.simulate_regimen(pd, doses; newpdx = true, tgrid = tg, draws = repeat([1], 3000), rng = MersenneTwister(4), obs_noise = false, details = true)
    @test abs(cor(log.(d1.λ0), log.(d1.k2))) < 0.06                                     # independent effects
    @test all(d1.λ1 .== exp(pd.logλ1[1]))                                                # pooled λ1
    # ---- v2 new PDX: correlation, Vtr and independence reproduce the target (Monte Carlo)
    S = 4
    v2, _ = synth_pd(S = S, nm = 5, ωl = 0.4, ωk = 0.7)
    ρs = [-0.6, 0.0, 0.5, 0.85]
    v2.diagnostics["model"] = "v2"; v2.diagnostics["corr_lk"] = ρs
    v2.diagnostics["mu_logVtr"] = fill(log(500.0), S); v2.diagnostics["omega_Vtr"] = fill(0.5, S)
    v2.diagnostics["lambda1_draws"] = fill(123.0, S, 5)
    n = 4000
    r = TP.simulate_regimen(v2, doses; newpdx = true, tgrid = [0.0, 1.0], draws = repeat(1:S, inner = n), obs_noise = false,
                            rng = MersenneTwister(11), details = true)
    for s in 1:S
        rows = (s - 1) * n + 1:s * n
        llam0 = log.(r.λ0[rows]); lk2 = log.(r.k2[rows]); lvtr = log.(r.λ1[rows]) .- llam0
        @test cor(llam0, lk2) ≈ ρs[s] atol = 0.045
        @test isapprox(std(llam0), 0.4; rtol = 0.05) && isapprox(std(lk2), 0.7; rtol = 0.05)
        @test isapprox(mean(llam0), v2.μ_λ0[s]; atol = 0.03) && isapprox(mean(lk2), v2.μ_k2[s]; atol = 0.04)
        @test isapprox(mean(lvtr), log(500.0); atol = 0.03) && isapprox(std(lvtr), 0.5; rtol = 0.05)   # λ1 = λ0·Vtr_i, Vtr_i ~ logN
        @test abs(cor(lvtr, llam0)) < 0.05 && abs(cor(lvtr, lk2)) < 0.05                    # Vtr independent of (λ0, k2)
    end
    # seeded reproducibility of the v2 stream
    r2 = TP.simulate_regimen(v2, doses; newpdx = true, tgrid = [0.0, 1.0], draws = repeat(1:S, inner = n), obs_noise = false, rng = MersenneTwister(11), details = true)
    @test r2.y == r.y && r2.λ1 == r.λ1
    # ---- known PDX in a v2 fit: the model's own prediction (pointwise log-likelihood of the hierarchical model)
    rng = MersenneTwister(31); N = 4; S = 6
    ddoses = dosing_schedule(10.0; start = 0.0, interval = 1.0, n = 45)
    pk = OneCmt(CL = 20.0, V = 5.0, ka = 8.0, F = 0.8)
    cref = maximum(conc(pk, ddoses, 0:0.1:7))
    truth = (μ_λ0 = log(0.1), ω_λ0 = 0.3, μ_k2 = log(0.3), ω_k2 = 0.6, logλ1 = log(100.0), logk1 = log(0.7), σ = 0.15)
    units, _ = TP.simulate_units(rng, N, ddoses, pk, cref; truth)
    for waning in (false, true)
        λ0 = 0.06 .+ 0.06 .* rand(rng, S, N); k2 = 0.1 .+ 0.5 .* rand(rng, S, N); λ1m = 40.0 .+ 120.0 .* rand(rng, S, N)
        w0c = 150.0 .+ 60.0 .* rand(rng, S, N); w0t = 150.0 .+ 60.0 .* rand(rng, S, N)
        logk1 = log(0.7) .+ 0.1 .* randn(rng, S); ρw = 0.02 .+ 0.04 .* rand(rng, S)
        dg = Dict{String,Any}("settings" => Dict{String,Any}("lod" => 4.0, "nu" => 5.0, "nchains" => 1), "nu" => 5.0, "model" => "v2",
                              "lambda1_draws" => λ1m, "corr_lk" => fill(0.3, S), "mu_logVtr" => fill(log(500.0), S), "omega_Vtr" => fill(0.5, S),
                              "waning" => waning, "waning_rate" => waning ? ρw : Float64[])
        pdk = PD("t", [u.model for u in units], ddoses, fill(pk, S), cref, fill(log(0.1), S), fill(0.3, S), fill(log(0.3), S), fill(0.6, S),
                 fill(log(100.0), S), logk1, fill(0.15, S), λ0, k2, vec(median(w0c; dims = 1)), vec(median(w0t; dims = 1)), w0c, w0t, dg)
        ll = TP.pointwise_loglik(pdk, units; threaded = false)
        idx = TP.pointwise_index(units)
        dist = TDist(5.0)
        ncheck = 0
        for (m, u) in enumerate(units), arm in (:treated, :control)
            c = arm === :treated ? u.treated : u.control
            cols = [i for i in eachindex(idx.t) if idx.group[i] == m && idx.arm[i] === arm && !idx.censored[i]]
            tt = idx.t[cols]
            w = arm === :treated ?
                TP.simulate_regimen(pdk, ddoses; model = u.model, tgrid = tt, w0 = w0t[:, m], draws = 1:S, obs_noise = false, rtol = 1e-5, atol = 1e-5) :
                TP.simulate_control(pdk; model = u.model, tgrid = tt, w0 = w0c[:, m], draws = 1:S, obs_noise = false, rtol = 1e-5, atol = 1e-5)
            for (q, col) in enumerate(cols), s in 1:S
                y = c.v[findfirst(==(tt[q]), c.t)]
                expected = logpdf(dist, (log(y) - log(max(w[s, q], 1e-3))) / 0.15) - log(0.15)
                @test ll[s, col] ≈ expected atol = 1e-4
                ncheck += 1
            end
        end
        @test ncheck > 100
        # the stored λ1 and (if present) the waning rate really enter the prediction
        base = TP.simulate_regimen(pdk, ddoses; model = units[1].model, tgrid = [0.0, 20.0, 40.0], w0 = w0t[:, 1], draws = 1:S, obs_noise = false)
        pdk.diagnostics["lambda1_draws"] = 0.02 .* λ1m
        @test maximum(abs.(TP.simulate_regimen(pdk, ddoses; model = units[1].model, tgrid = [0.0, 20.0, 40.0], w0 = w0t[:, 1], draws = 1:S, obs_noise = false) .- base)) > 1.0
        pdk.diagnostics["lambda1_draws"] = λ1m
        if waning
            pdk.diagnostics["waning"] = false
            nowane = TP.simulate_regimen(pdk, ddoses; model = units[1].model, tgrid = [0.0, 20.0, 40.0], w0 = w0t[:, 1], draws = 1:S, obs_noise = false)
            @test all(nowane[:, 3] .< base[:, 3])                                       # waning efficacy gives larger late volumes
        end
    end
    # ---- waning lowers predicted response in the virtual trial; v2 trial runs through the same path
    wp, wd = synth_pd(S = 20, μk = log(0.25))
    nw = TP.simulate_trial(wp, wd; n_pdx = 150, ndraws = 20, rng = MersenneTwister(2)).summary.responder_rate.median
    wp.diagnostics["waning"] = true; wp.diagnostics["waning_rate"] = fill(0.15, 20)
    ww = TP.simulate_trial(wp, wd; n_pdx = 150, ndraws = 20, rng = MersenneTwister(2)).summary.responder_rate.median
    @test ww < nw - 0.05
    wp.diagnostics["model"] = "v2"; wp.diagnostics["corr_lk"] = fill(0.5, 20); wp.diagnostics["mu_logVtr"] = fill(log(1000.0), 20)
    wp.diagnostics["omega_Vtr"] = fill(0.5, 20); wp.diagnostics["lambda1_draws"] = fill(120.0, 20, 30)
    tv2 = TP.simulate_trial(wp, wd; n_pdx = 50, ndraws = 10, rng = MersenneTwister(2))
    @test size(tv2.categories) == (50, 10) && tv2.n_na == 0
    # convergence summary includes the v2 hyperparameters
    wp.diagnostics["nchains"] = 4; wp.diagnostics["waning_rate"] = exp.(0.1 .* randn(MersenneTwister(3), 20))
    wp.diagnostics["corr_lk"] = 0.5 .+ 0.1 .* randn(MersenneTwister(4), 20)
    cs = TP.convergence_summary(wp)
    @test cs.recomputed !== nothing && "corr(log λ0, log k2)" in cs.recomputed.names && "waning rate" in cs.recomputed.names
end
