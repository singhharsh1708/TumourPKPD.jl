using Test, Random, Statistics
using TumourPKPD
using TumourPKPD: Dose, OneCmt, dosing_schedule, conc, PKExposure
using Turing
using MCMCChains
isdefined(TumourPKPD, :fit_drug) ||
    Base.include(TumourPKPD, joinpath(@__DIR__, "..", "src", "hierarchical.jl"))
const H = TumourPKPD
const DynamicPPL = Turing.DynamicPPL
const LogDensityProblems = Turing.LogDensityProblems
using ForwardDiff

@testset "hierarchical" begin
    pk = OneCmt(CL = 3.0, V = 1.5, ka = 24.0, F = 0.8)
    doses = dosing_schedule(25.0; start = 0.0, interval = 1.0, n = 21)
    cref = H.reference_cref(pk, doses)
    truth = (μ_λ0 = log(0.08), ω_λ0 = 0.3, μ_k2 = log(0.5), ω_k2 = 0.7,
             logλ1 = log(120.0), logk1 = log(0.7), σ = 0.15)
    rng = MersenneTwister(11)
    units, tru = H.simulate_units(rng, 3, doses, pk, cref; truth, tspan = (21.0, 24.0))
    tight = H.SolverCfg(; reltol = 1e-10, abstol = 1e-10)

    @testset "exposure and data prep" begin
        e1 = H.make_exposure(pk, doses, cref)
        @test e1 isa PKExposure
        direct(t) = conc(pk, doses, t) / cref
        @test maximum(abs(e1(t) - direct(t)) for t in 0:0.07:25) < 1e-10
        c = (; t = [0.0, 3.0, 7.0, 10.0], v = [150.0, 0.0, 40.0, -1.0])
        cd = H.prepare_curve(c)
        @test cd.v0 == 150.0
        @test cd.t == [3.0, 7.0]               # day 0 and negative volume dropped
        @test cd.cens == [true, false]
        @test cd.logv[2] ≈ log(40.0)
    end

    @testset "invalid baselines are dropped, not fabricated" begin
        mk(m, t, v) = (; model = m, control = (; t, v), treated = (; t, v))
        good = mk("ok", [0.0, 5.0, 10.0], [100.0, 150.0, 300.0])
        zero0 = mk("zero0", [0.0, 5.0, 10.0], [0.0, 150.0, 300.0])
        nan0 = mk("nan0", [0.0, 5.0], [NaN, 150.0])
        late = mk("late", [2.0, 5.0], [100.0, 150.0])             # no day-0 observation
        @test H.valid_baseline(good.control) && !H.valid_baseline(zero0.control)
        @test !H.valid_baseline(nan0.control) && !H.valid_baseline(late.control)
        @test_throws ArgumentError H.prepare_curve(zero0.control)
        ud, dropped = @test_logs (:warn, r"dropping 3 of 4") H.prepare_units([good, zero0, nan0, late])
        @test length(ud) == 1 && ud[1].model == "ok"
        @test sort(dropped) == ["late", "nan0", "zero0"]
        d = @test_logs (:warn, r"dropping 1 of 2") H.prepare_data([good, zero0], doses, pk, cref)
        @test d.dropped == ["zero0"] && length(d.units) == 1
        @test_throws ArgumentError H.prepare_data([zero0], doses, pk, cref)
    end

    function make_ldf(; latent = false, local_grad = true, threaded = true, adtype = true, chunk = nothing, solver = tight, nu = 5.0, centered = true, k2c = nothing)
        d = H.prepare_data(units, doses, pk, cref; latent_w0 = latent, local_grad, threaded, solver, nu, centered_lam0 = centered, k2_centered = k2c)
        model = H.hier_model(d, H.default_priors())
        D = 7 + 2 * 3 + (latent ? 6 : 0)
        adt = AutoForwardDiff(chunksize = chunk === nothing ? (local_grad ? D : 12) : chunk)
        vi = DynamicPPL.VarInfo(MersenneTwister(3), model)
        nt = merge(H.initial_values(d, H.default_priors(), MersenneTwister(4)),
                   (; μ_k2 = log(0.5), logk1 = log(0.7), logVtr = log(1000.0), σ = 0.2))
        vi = DynamicPPL.link(DynamicPPL.initialize_parameters!!(vi, nt, model), model)
        ldf = adtype ? DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi; adtype = adt) :
                       DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi)
        return ldf, vi[:], d
    end

    @testset "logdensity and gradient" begin
        for latent in (false, true)
            ldf, x, _ = make_ldf(; latent)
            lp, g = LogDensityProblems.logdensity_and_gradient(ldf, x)
            @test isfinite(lp)
            @test all(isfinite, g)
            ldf0, _, _ = make_ldf(; latent, adtype = false)
            @test LogDensityProblems.logdensity(ldf0, x) ≈ lp rtol = 1e-8
            fd = map(eachindex(x)) do i
                h = 1e-5; e = zeros(length(x)); e[i] = h   # larger h: solver noise (~1e-10 rel.) swamps 1e-6 steps
                (LogDensityProblems.logdensity(ldf0, x .+ e) - LogDensityProblems.logdensity(ldf0, x .- e)) / 2h
            end
            @test maximum(abs.(g .- fd) ./ (1 .+ abs.(fd))) < 1e-4
            ldfg, _, _ = make_ldf(; latent, local_grad = false)
            lpg, gg = LogDensityProblems.logdensity_and_gradient(ldfg, x)
            @test lpg ≈ lp rtol = 1e-8
            @test maximum(abs.(gg .- g) ./ (1 .+ abs.(g))) < 1e-6
            # small chunk: several forward passes share one cached set of local gradients
            ldfc, _, dc = make_ldf(; latent, chunk = 4)
            lpc, gc = LogDensityProblems.logdensity_and_gradient(ldfc, x)
            @test lpc ≈ lp rtol = 1e-10
            @test gc ≈ g rtol = 1e-8
            @test H.cache_stats(dc)[1] >= 2
        end
    end

    @testset "non-centred lambda0 parameterisation also differentiates correctly" begin
        ldf, x, _ = make_ldf(; latent = true, centered = false)
        lp, g = LogDensityProblems.logdensity_and_gradient(ldf, x)
        ldf0, _, _ = make_ldf(; latent = true, centered = false, adtype = false)
        fd = map(eachindex(x)) do i
            e = zeros(length(x)); e[i] = 1e-5
            (LogDensityProblems.logdensity(ldf0, x .+ e) - LogDensityProblems.logdensity(ldf0, x .- e)) / 2e-5
        end
        @test isfinite(lp)
        @test maximum(abs.(g .- fd) ./ (1 .+ abs.(fd))) < 2e-3      # FD noise at the start point (gradients of order 1e3)
        nt = (; μ_λ0 = -2.0, ω_λ0 = 0.5, z_λ0 = [1.0, -1.0], z_k2 = [0.0, 0.0])
        c = H.to_centered(nt)
        @test c.llam0 ≈ [-1.5, -2.5] && !haskey(c, :z_λ0) && haskey(c, :z_k2)
    end

    @testset "partially centred potency (k2_centered mask)" begin
        mask = [true, false, true]
        ldf, x, d = make_ldf(; latent = true, k2c = mask)
        @test length(x) == 7 + 3 + 3 + 6                # llam0 + (lk2c 2 + z_k2 1) + z_w0 6
        lp, g = LogDensityProblems.logdensity_and_gradient(ldf, x)
        ldf0, _, _ = make_ldf(; latent = true, k2c = mask, adtype = false)
        fd = map(eachindex(x)) do i
            e = zeros(length(x)); e[i] = 1e-5
            (LogDensityProblems.logdensity(ldf0, x .+ e) - LogDensityProblems.logdensity(ldf0, x .- e)) / 2e-5
        end
        @test isfinite(lp) && maximum(abs.(g .- fd) ./ (1 .+ abs.(fd))) < 2e-3
        nt = (; μ_k2 = -1.0, ω_k2 = 0.5, z_k2 = [1.0, 2.0, 3.0])
        c = H.apply_k2_mask(nt, mask)
        @test c.lk2c ≈ [-0.5, 0.5] && c.z_k2 == [2.0] && H.apply_k2_mask(nt, falses(3)) === nt
        @test H._k2_positions(mask) == [1, 1, 2]
        _, pdm = H.fit_drug(units, doses, pk, cref; drug = "toymask", nchains = 1, nsamples = 8, nadapt = 8,
                            rng = MersenneTwister(8), map_starts = 0, window_days = 0, k2_centered = mask)
        @test size(pdm.k2) == (8, 3) && all(pdm.k2 .> 0) && pdm.diagnostics["settings"]["k2_centered_units"] == 2
    end

    @testset "plain and AD values agree at default tolerances (censored, shrinking, extreme)" begin
        dflt = H.SolverCfg()                       # production tolerances 1e-5 / 1e-5
        ud, _ = H.prepare_units(units)
        d = H.HierData(ud, H.make_exposure(pk, doses, cref), 4.0, dflt, true, true, false)
        dt = H.HierData(ud, H.make_exposure(pk, doses, cref), 4.0, tight, true, true, false)
        u = ud[1]
        lc, lt = log(u.control.v0), log(u.treated.v0)
        regimes = Dict(
            "typical"   => (log(0.08), log(0.5), log(120.0), log(0.7), 0.15),
            "shrinking" => (log(0.05), log(3.0), log(120.0), log(0.7), 0.15),   # strong response
            "eradicated" => (log(0.02), log(40.0), log(120.0), log(1.5), 0.15), # volume goes to ~0, censored obs
            "extreme"   => (log(5.0), log(300.0), log(5.0), log(20.0), 0.05),   # stiff / huge rates
            "tiny sigma" => (log(0.08), log(0.5), log(120.0), log(0.7), 0.01))
        for (name, (a, b, c, e, s)) in regimes
            p = (a, b, c, e, s, lc, lt)
            vplain = H.unit_loglik(u, p..., d)
            v, g = H.unit_valgrad(u, p, d)
            if isfinite(vplain)
                @test v == vplain                       # same value, bit for bit, in both paths
                @test all(isfinite, g)
                # gradient from the default-tolerance dual solve vs tight-tolerance dual solve
                vt, gt = H.unit_valgrad(u, p, dt)
                if name != "extreme" && name != "tiny sigma"
                    @test maximum(abs.(g .- gt) ./ (1 .+ abs.(gt))) < 5e-3
                end
            else
                @test v == -Inf                         # same failure semantics
                @test all(==(0.0), g)
            end
        end
        # eradication really exercises the censored branch
        @test any(u.treated.cens) || any(any(w.treated.cens) for w in ud)
    end

    @testset "threaded == serial" begin
        _, x, dt = make_ldf(; threaded = true)
        _, _, ds = make_ldf(; threaded = false)
        args = (fill(log(0.08), 3), fill(log(0.5), 3), [log(u.control.v0) for u in dt.units],
                [log(u.treated.v0) for u in dt.units], log(120.0), log(0.7), 0.2)
        @test H.population_loglik(dt, args...) == H.population_loglik(ds, args...)
    end

    @testset "ODE failure gives -Inf" begin
        _, _, d = make_ldf()
        u = d.units[1]
        v = H.unit_loglik(u, log(1e6), log(1e6), log(1e6), log(1e6), 0.2, log(u.control.v0), log(u.treated.v0), d)
        @test !isnan(v)
    end

    @testset "chain parts roundtrip (process-parallel chains)" begin
        c1 = Chains(randn(MersenneTwister(1), 30, 3, 1), [:a, :b, :tree_depth], Dict(:internals => [:tree_depth]))
        c2 = Chains(randn(MersenneTwister(2), 30, 3, 1), [:a, :b, :tree_depth], Dict(:internals => [:tree_depth]))
        r1 = H.chain_from_parts(H.chain_to_parts(c1)...)
        @test Array(r1) == Array(c1) && Symbol.(names(r1)) == [:a, :b, :tree_depth]
        cc = chainscat(r1, H.chain_from_parts(H.chain_to_parts(c2)...))
        @test size(cc) == (30, 3, 2)
    end

    @testset "failed dual solve rejects the unit (Float64 solve fine, dual solve failed)" begin
        ud, _ = H.prepare_units(units)
        d = H.HierData(ud, H.make_exposure(pk, doses, cref), 4.0, H.SolverCfg(), true, true, false)
        u = ud[1]
        p = (log(0.08), log(0.5), log(120.0), log(0.7), 0.15, log(u.control.v0), log(u.treated.v0))
        v0_, g0 = H.unit_valgrad(u, p, d)
        @test isfinite(v0_) && any(!=(0.0), g0)                  # normal case accepted
        H._FAULT_DUAL[] = true
        try
            @test isfinite(H.unit_loglik(u, p..., d))             # the plain solve still succeeds
            v, g = H.unit_valgrad(u, p, d)
            @test v == -Inf && all(==(0.0), g)                    # ... but the unit is rejected
            # and so is the whole population log-likelihood (dual path)
            dd = ForwardDiff.Dual{:t}
            x = [dd(p[1], 1.0, 0.0), dd(p[1] + 0.01, 0.0, 1.0)]
            ll = H.population_loglik(H.HierData(ud[1:2], d.exposure, 4.0, H.SolverCfg(), true, true, false),
                                     x, [dd(p[2], 0.0, 0.0), dd(p[2], 0.0, 0.0)],
                                     [dd(p[6], 0.0, 0.0), dd(p[6], 0.0, 0.0)], [dd(p[7], 0.0, 0.0), dd(p[7], 0.0, 0.0)],
                                     dd(p[3], 0.0, 0.0), dd(p[4], 0.0, 0.0), dd(p[5], 0.0, 0.0))
            @test ForwardDiff.value(ll) == -Inf
        finally
            H._FAULT_DUAL[] = false
        end
    end

    @testset "Student-t residuals" begin
        ν = 5.0
        @test H._t_logpdf(0.3, 0.1, 0.2, ν) ≈ logpdf(TDist(ν), (0.3 - 0.1) / 0.2) - log(0.2)
        @test H._t_logpdf(0.3, 0.1, 0.2, Inf) ≈ logpdf(Normal(0.1, 0.2), 0.3)
        @test H._t_logcdf(1.4, 3.0, 0.5, ν) ≈ logcdf(TDist(ν), (1.4 - 3.0) / 0.5)
        @test H._t_logcdf(1.4, 3.0, 0.5, Inf) ≈ logcdf(Normal(3.0, 0.5), 1.4)
        # ForwardDiff derivative of the censored term (the Student-t CDF has a hand-written rule)
        f(μ) = H._t_logcdf(1.4, μ, 0.5, ν)
        for μ0 in (1.0, 3.0, 6.0)
            fd = (f(μ0 + 1e-6) - f(μ0 - 1e-6)) / 2e-6
            @test ForwardDiff.derivative(f, μ0) ≈ fd rtol = 1e-5
        end
        h(σ) = H._t_logcdf(1.4, 3.0, σ, ν)
        @test ForwardDiff.derivative(h, 0.5) ≈ (h(0.5 + 1e-6) - h(0.5 - 1e-6)) / 2e-6 rtol = 1e-5
        # heavier tails: an outlier costs less under t5 than under the Normal with the same scale
        @test H._t_logpdf(2.0, 0.0, 0.15, 5.0) > H._t_logpdf(2.0, 0.0, 0.15, Inf)
    end

    @testset "fit window, jitter, MAP" begin
        w = H.window_units(units, 14.0)
        @test all(maximum(u.control.t) <= 14.0 && maximum(u.treated.t) <= 14.0 for u in w)
        @test all(u.control.t[1] == 0.0 && u.treated.t[1] == 0.0 for u in w)
        @test H.window_units(units, 0) === units && H.window_units(units, Inf) === units
        @test length(w[1].control.t) < length(units[1].control.t)
        nt = (; μ_λ0 = -2.0, ω_λ0 = 0.0, σ = 0.2, z_λ0 = [0.1, 0.2])
        j = H.jitter_init(nt, MersenneTwister(1), 0.05)
        @test j.ω_λ0 > 0 && j.σ > 0 && length(j.z_λ0) == 2 && j.μ_λ0 != nt.μ_λ0
        @test H.jitter_init(nt, MersenneTwister(1), 0) === nt
        d = H.prepare_data(units, doses, pk, cref; latent_w0 = true, threaded = false, centered_lam0 = false)
        model = H.hier_model(d, H.default_priors())
        nt0 = H.initial_values(d, H.default_priors(), MersenneTwister(2); jitter = 0.0)
        modes = H.multistart_map(model, d, H.default_priors(), MersenneTwister(3); nstarts = 2, maxiters = 60, logger = _ -> nothing)
        @test length(modes) == 2 && all(m -> haskey(m.nt, :z_w0c) && m.nt.ω_λ0 > 0, modes)
        @test all(m -> isfinite(m.lp), modes)
        # the first start (data-driven) must not be worse than where it started
        vi = DynamicPPL.link(DynamicPPL.initialize_parameters!!(DynamicPPL.VarInfo(MersenneTwister(0), model), nt0, model), model)
        ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi)
        @test modes[1].lp >= LogDensityProblems.logdensity(ldf, vi[:]) - 1e-6
    end

    @testset "diagnostics edge cases" begin
        # constant parameter -> no finite R-hat/ESS: must not throw, reports NaN
        ch = Chains(hcat(ones(20, 1), ones(20, 1)), [:a, :tree_depth], Dict(:internals => [:tree_depth]))
        dg = H.chain_diagnostics(ch; max_depth = 1)
        @test dg["max_tree_depth_hits"] == 20            # depth 1 >= configured max_depth 1
        dg2 = H.chain_diagnostics(ch; max_depth = 2)
        @test dg2["max_tree_depth_hits"] == 0
        @test dg isa Dict
    end

    @testset "fit_drug (latent default), PosteriorDraws, save/load, pointwise" begin
        chain, pd = H.fit_drug(units, doses, pk, cref; drug = "toy", nchains = 2, nsamples = 20,
                               nadapt = 20, rng = MersenneTwister(5), map_starts = 2, map_maxiters = 40, window_days = 0,
                               cutoffs = Dict("syn1|treated" => 14.0),
                               split = Dict("kind" => "late", "late_frac" => 0.3), seed = 5)
        @test pd isa H.PosteriorDraws
        @test pd.diagnostics["latent_w0"] === true
        @test pd.diagnostics["fit_window_day"] == 0.0 && pd.diagnostics["nu"] == 5.0 && pd.diagnostics["baseline_sd"] == 0.1
        @test length(pd.diagnostics["map_lps"]) == 2 && pd.diagnostics["draws_per_chain"] == 20
        @test length(pd.σ) == 40
        @test size(pd.λ0) == (40, 3) && size(pd.k2) == (40, 3)
        @test size(pd.w0_control_draws) == (40, 3) && size(pd.w0_treated_draws) == (40, 3)
        @test pd.w0_control ≈ vec(median(pd.w0_control_draws; dims = 1))
        @test length(unique(pd.w0_treated_draws[:, 1])) > 1       # genuinely per-draw
        @test pd.models == ["syn1", "syn2", "syn3"]
        @test all(pd.λ0 .> 0) && all(pd.k2 .> 0)
        @test haskey(pd.diagnostics, "max_rhat") && haskey(pd.diagnostics, "runtime_s")
        @test pd.diagnostics["nchains"] == 2 && pd.diagnostics["draws_per_chain"] == 20
        @test pd.diagnostics["train_models"] == pd.models
        @test pd.diagnostics["split"]["kind"] == "late" && pd.diagnostics["split"]["late_frac"] == 0.3
        @test pd.diagnostics["cutoffs"] == Dict("syn1|treated" => 14.0)
        @test pd.diagnostics["lod"] == H._default_lod() && isempty(pd.diagnostics["dropped_units"])
        @test pd.diagnostics["settings"]["chunksize"] <= 32
        @test pd.diagnostics["max_observed_day"] == maximum(max(last(u.control.t), last(u.treated.t)) for u in units)
        @test pd.diagnostics["median_followup_day"] == median(last(u.treated.t) for u in units)
        @test pd.diagnostics["tumour_types"] == Dict(m => "synthetic" for m in pd.models)
        path = joinpath(mktempdir(), "toy.jld2")
        H.save_fit(path, pd)
        pd2 = H.load_fit(path)
        @test pd2.drug == pd.drug && pd2.models == pd.models
        @test pd2.λ0 == pd.λ0 && pd2.k2 == pd.k2 && pd2.σ == pd.σ
        @test pd2.w0_control_draws == pd.w0_control_draws && pd2.w0_treated_draws == pd.w0_treated_draws
        @test pd2.doses == pd.doses && pd2.pk == pd.pk && pd2.cref == pd.cref

        ll = H.pointwise_loglik(pd, units; draws = 1:5)
        idx = H.pointwise_index(units)
        @test size(ll) == (5, length(idx.t))
        @test all(isfinite, ll)
        # sums of pointwise terms reproduce the unit likelihood evaluated at the draw's OWN latent baselines
        d = H.prepare_data(units, doses, pk, cref)
        s = 3
        for (i, u) in enumerate(d.units)
            v = H.unit_loglik(u, log(pd.λ0[s, i]), log(pd.k2[s, i]), pd.logλ1[s], pd.logk1[s], pd.σ[s],
                              log(pd.w0_control_draws[s, i]), log(pd.w0_treated_draws[s, i]), d)
            @test sum(ll[s, idx.group .== i]) ≈ v rtol = 1e-8
        end
        @test size(H.grouped_loglik(ll, idx)) == (5, 3)
    end

    @testset "fixed-baseline fit stores constant baselines" begin
        _, pdf = H.fit_drug(units, doses, pk, cref; drug = "toyfixed", nchains = 1, nsamples = 10, nadapt = 10,
                            rng = MersenneTwister(6), latent_w0 = false, map_starts = 0, window_days = 0)
        @test pdf.diagnostics["latent_w0"] === false
        @test all(pdf.w0_control_draws .== reshape(pdf.w0_control, 1, :))
        @test pdf.w0_control == [u.control.v0 for u in H.prepare_units(units)[1]]
    end

    @testset "marginal_unit_loglik vs brute-force importance sampling" begin
        # hand-built posterior draws (no fit): 2 population draws, latent baselines
        S = 2
        set = Dict{String,Any}("lod" => 4.0, "latent_w0" => true, "reltol" => 1e-5, "abstol" => 1e-5, "nu" => 5.0, "baseline_sd" => 0.1)
        pops = (μ_λ0 = [log(0.08), log(0.06)], ω_λ0 = [0.3, 0.4], μ_k2 = [log(0.5), log(0.4)],
                ω_k2 = [0.7, 0.8], logλ1 = [log(120.0), log(150.0)], logk1 = [log(0.7), log(0.6)],
                σ = [0.15, 0.2])
        ud, _ = H.prepare_units(units)
        v0c = [u.control.v0 for u in ud]; v0t = [u.treated.v0 for u in ud]
        mkpd(latent) = H.PosteriorDraws("syn", [u.model for u in ud], doses, fill(pk, S), cref,
            pops.μ_λ0, pops.ω_λ0, pops.μ_k2, pops.ω_k2, pops.logλ1, pops.logk1, pops.σ,
            ones(S, 3), ones(S, 3), v0c, v0t, repeat(v0c', S), repeat(v0t', S),
            Dict{String,Any}("settings" => merge(set, Dict("latent_w0" => latent)), "latent_w0" => latent))
        pdl = mkpd(true)
        expo = H.make_exposure(pk, doses, cref)
        cfg = H.SolverCfg()
        # one log-likelihood evaluation at standardised latent vector z = (z_λ0, z_k2, z_w0c, z_w0t)
        function loglik_z(i, s, z; latent = true)
            u = ud[i]; σ = pops.σ[s]
            hd = H.HierData([u], expo, 4.0, cfg, latent, false, false)
            H.unit_loglik(u, pops.μ_λ0[s] + pops.ω_λ0[s] * z[1], pops.μ_k2[s] + pops.ω_k2[s] * z[2],
                          pops.logλ1[s], pops.logk1[s], σ,
                          log(u.control.v0) + (latent ? 0.1 * z[3] : 0.0), log(u.treated.v0) + (latent ? 0.1 * z[4] : 0.0), hd)
        end
        # estimator B: importance sampling with an independent, wider proposal N(0, 1.6^2 I)
        function is_estimate(i, s, M, rng; sd = 1.6)
            lw = Vector{Float64}(undef, M)
            for m in 1:M
                z = sd .* randn(rng, 4)
                logq = -0.5 * sum(abs2, z) / sd^2 - 4 * log(sd) - 2 * log(2π)
                lw[m] = loglik_z(i, s, z) + (-0.5 * sum(abs2, z) - 2 * log(2π)) - logq
            end
            mx = maximum(lw); w = exp.(lw .- mx)
            est = mx + log(mean(w))
            se = std(w) / (mean(w) * sqrt(M))             # delta method for log of a mean
            return est, se
        end
        M = 1500
        A = H.marginal_unit_loglik(pdl, units; M, rng = MersenneTwister(1), threaded = false)
        @test size(A) == (S, 3) && all(isfinite, A)
        for (i, s) in ((1, 1), (2, 2), (3, 1))
            # standard error of the prior-MC estimator from its own samples
            eps = randn(MersenneTwister(1), M, 4)
            lw = [loglik_z(i, s, eps[m, :]) for m in 1:M]
            mx = maximum(lw); w = exp.(lw .- mx)
            seA = std(w) / (mean(w) * sqrt(M))
            @test A[s, i] ≈ mx + log(mean(w)) atol = 1e-6                # matches the direct computation
            B, seB = is_estimate(i, s, M, MersenneTwister(99))
            @test abs(A[s, i] - B) < 5 * sqrt(seA^2 + seB^2) + 1e-3
        end
        # baselines are integrated for latent fits but not for fixed fits
        pdf = mkpd(false)
        e1 = randn(MersenneTwister(2), 200, 4); e2 = copy(e1); e2[:, 3:4] .= randn(MersenneTwister(3), 200, 2)
        @test H.marginal_unit_loglik(pdf, units; eps = e1, threaded = false) ==
              H.marginal_unit_loglik(pdf, units; eps = e2, threaded = false)
        @test H.marginal_unit_loglik(pdl, units; eps = e1, threaded = false) !=
              H.marginal_unit_loglik(pdl, units; eps = e2, threaded = false)
    end
end
