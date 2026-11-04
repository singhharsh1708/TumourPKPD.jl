# Tests for src/simulate.jl and src/validate.jl.
# Run: julia --project=TumourPKPD.jl -t 2 TumourPKPD.jl/test/test_validate.jl
using Test, Random, Statistics
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
    lp = TP.log_predictive_density([yv], latent, σ)
    @test lp[1] ≈ (-0.5 * ((log(yv) - 1) / 0.3)^2 - log(0.3) - 0.5 * log(2π)) - log(yv) atol = 1e-12
    # left-censored: P(y < limit) with limit = exp(1) is 0.5
    @test TP.log_predictive_density([0.0], latent, σ; limit = exp(1.0))[1] ≈ log(0.5) atol = 1e-6
    @test isnan(TP.log_predictive_density([0.0], latent, σ)[1])
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
    lf = TP.log_predictive_density([2.0, 2.0], hcat(fill(2.0, 50), fill(NaN, 50)), fill(0.3, 50))
    @test isfinite(lf[1]) && lf[2] == -Inf                                      # no valid draw: -Inf kept
    half = fill(2.0, 50); half[1:25] .= NaN                                     # half the draws failed: log(1/2) penalty
    lh = TP.log_predictive_density([2.0], reshape(half, :, 1), fill(0.3, 50))[1]
    @test lh ≈ TP.log_predictive_density([2.0], fill(2.0, 50, 1), fill(0.3, 50))[1] + log(0.5)
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
       [100.0, 150.0], [120.0, 160.0], Dict{String,Any}())
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
    @test_throws ArgumentError TP.simulate_regimen(pd, pd.doses; tgrid = tg)
    @test_throws ArgumentError TP.simulate_regimen(pd, pd.doses; model = "A", newpdx = true, tgrid = tg)
    @test_throws ArgumentError TP.simulate_regimen(pd, pd.doses; model = "Z", tgrid = tg)
    # new PDX: ω = 0 collapses to the population median; ω > 0 spreads
    pd0 = toy_pd(ω = 0.0)
    s0 = TP.simulate_regimen(pd0, pd0.doses; newpdx = true, tgrid = tg, ndraws = 5, obs_noise = false)
    wpop, _ = simulate_tumour(Simeoni(0.15, 2.0, 0.5, 0.6, median([100.0, 150.0] .* 0 .+ [120.0, 160.0])), expo, tg)
    @test s0[1, :] ≈ wpop rtol = 1e-10
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
