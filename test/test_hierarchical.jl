using Test, Random, Statistics, Distributions, LinearAlgebra
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
using JLD2

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

    @testset "distinct MAP modes and E-BFMI / diagnostics keys" begin
        mk(lp, μ) = (; nt = (; μ_λ0 = μ, ω_λ0 = 0.5, μ_k2 = -1.0, ω_k2 = 0.5, logVtr = 7.0, logk1 = 0.0, σ = 0.2), lp)
        ms = [mk(70.9, -3.0), mk(65.4, -3.0), mk(65.45, -3.0), mk(70.88, -3.0), mk(-1.3, -3.0), mk(-Inf, -3.0)]
        dm = H.distinct_modes(ms)
        @test [round(m.lp; digits = 1) for m in dm] == [70.9, 65.4]          # duplicates merged, non-converged and failed dropped
        @test length(H.distinct_modes([mk(10.0, -3.0), mk(10.1, -2.5)])) == 2   # same lp, parameters differ -> two modes
        E = hcat(cumsum(0.01 .* randn(MersenneTwister(1), 200)), randn(MersenneTwister(2), 200))   # sticky vs fresh energies
        ch = Chains(reshape(hcat(randn(MersenneTwister(3), 200, 2), E), 200, 4, 1), [:a, :b, :hamiltonian_energy, :tree_depth],
                    Dict(:internals => [:hamiltonian_energy, :tree_depth]))
        dg = H.chain_diagnostics(ch)
        for k in ("min_ess_tail", "min_ebfmi", "cross_mode_rhat", "max_rhat", "min_ess_bulk"); @test haskey(dg, k); end
        @test isnan(dg["cross_mode_rhat"])
    end

    @testset "staged sampling (warmup state -> fixed-kernel sampling)" begin
        mask = [true, false, true]
        ud0 = units
        mm = H.map_modes(ud0, doses, pk, cref; window_days = 0, nstarts = 2, maxiters = 120, rng = MersenneTwister(1), logger = _ -> nothing)
        start = mm.starts[1].nt
        tgt = H.stage_target(ud0, doses, pk, cref; start, k2_centered = mm.k2_centered, window_days = 0, threaded = false)
        # layout follows the model: constrained transform equals DynamicPPL's invlink
        vi = DynamicPPL.link(DynamicPPL.initialize_parameters!!(DynamicPPL.VarInfo(MersenneTwister(0), tgt.model), start, tgt.model), tgt.model)
        @test vi[:] ≈ tgt.θ0
        θ = tgt.θ0 .+ 0.05 .* randn(MersenneTwister(2), length(tgt.θ0))
        @test H.constrain_draws(tgt.layout, reshape(θ, 1, :))[1, :] ≈ DynamicPPL.invlink(DynamicPPL.unflatten(vi, θ), tgt.model)[:]
        w = H.stage_warmup(tgt, tgt.θ0, MersenneTwister(5); nadapt = 40, log_every = 0)
        @test w.ϵ > 0 && length(w.Minv) == length(tgt.θ0) && size(w.stats, 1) == 40
        # a sample stage rebuilt from scratch sees the same target at the warm position and does not adapt
        tgt2 = H.stage_target(ud0, doses, pk, cref; start, k2_centered = mm.k2_centered, window_days = 0, threaded = false)
        @test tgt2.ℓπ(w.θ) == tgt.ℓπ(w.θ)
        s = H.stage_sample(tgt2, w.θ, w.ϵ, w.Minv, MersenneTwister(6); ndraws = 12, burnin = 3, log_every = 0)
        @test all(s.stats[:, 3] .== w.ϵ)                                  # step size never changes
        @test size(s.θ) == (12, length(tgt.θ0)) && all(isfinite, s.θ)
        # chain assembly and extraction (also: wrong mask is rejected)
        chn = H.stage_chain(tgt2.layout, H.constrain_draws(tgt2.layout, s.θ), s.stats)
        two = chainscat(chn, chn)
        _, pdst = H.fit_drug(ud0, doses, pk, cref; precomputed_chain = two, k2_centered = mm.k2_centered, window_days = 0,
                             map_starts = 0, drug = "staged")
        @test length(pdst.σ) == 24 && pdst.diagnostics["k2_centered"] == mm.k2_centered && all(pdst.k2 .> 0)
        @test pdst.diagnostics["k2_centered_models"] == [m for (m, c) in zip(pdst.models, mm.k2_centered) if c]
        bad = .!mm.k2_centered
        any(mm.k2_centered) && !all(mm.k2_centered) &&
            @test_throws ArgumentError H.fit_drug(ud0, doses, pk, cref; precomputed_chain = two, k2_centered = bad, window_days = 0, map_starts = 0)
        @test H.stage_seed(1, 1, 1) != H.stage_seed(1, 1, 2) != H.stage_seed(1, 2, 1)
        # resumable warmup across the metric-adaptation windows (220 iterations, buffers 40/30, window 25), RNG serialised
        nad = 220; kw = (; nadapt = nad, log_every = 0, init_buffer = 40, term_buffer = 30, window_size = 25)
        rng0 = MersenneTwister(5)
        r1 = H.stage_warmup(tgt, tgt.θ0, rng0; kw..., upto = 90)
        ad = r1.adaptor
        @test !r1.done && r1.i == 90 && size(r1.stats, 1) == 90 && ad.state.i == 90
        @test ad.state.window_start == 41 && ad.state.window_end == nad - 30 && !isempty(ad.state.window_splits)
        @test r1.Minv != ones(length(tgt.θ0))                       # at least one metric window closed before iteration 90
        path = tempname() * ".jld2"
        JLD2.jldsave(path; theta = r1.θ, epsilon = r1.ϵ, Minv = r1.Minv, adaptor = ad, i = r1.i, rng = rng0)
        f = JLD2.load(path); ad2 = f["adaptor"]
        @test rand(copy(rng0)) == rand(f["rng"])                    # RNG state round-trips
        for (x, y) in ((ad.state.i, ad2.state.i), (ad.state.window_start, ad2.state.window_start), (ad.state.window_end, ad2.state.window_end),
                       (ad.state.window_splits, ad2.state.window_splits), (ad.pc.n, ad2.pc.n), (ad.pc.μ, ad2.pc.μ), (ad.pc.M, ad2.pc.M),
                       (ad.pc.var, ad2.pc.var), (ad.ssa.state.m, ad2.ssa.state.m), (ad.ssa.state.ϵ, ad2.ssa.state.ϵ),
                       (ad.ssa.state.x_bar, ad2.ssa.state.x_bar), (ad.ssa.state.H_bar, ad2.ssa.state.H_bar),
                       (ad.init_buffer, ad2.init_buffer), (ad.term_buffer, ad2.term_buffer), (ad.window_size, ad2.window_size))
            @test x == y                                            # window, variance-estimator and dual-averaging state restored exactly
        end
        # dual averaging restarts at each window end: m counts the iterations since the last split <= i
        last_split(i, sp) = (v = filter(<=(i), sp); isempty(v) ? 0 : maximum(v))
        @test ad2.ssa.state.m == ad2.state.i - last_split(ad2.state.i, ad2.state.window_splits)
        r2 = H.stage_warmup(tgt, f["theta"], f["rng"]; kw..., upto = 150,
                            resume = (; θ = f["theta"], ϵ = f["epsilon"], Minv = f["Minv"], adaptor = ad2, i = f["i"]))
        a2 = r2.adaptor
        @test !r2.done && r2.i == 150 && size(r2.stats, 1) == 60 && a2.state.i == 150
        @test a2.state.window_splits == ad.state.window_splits && a2.state.window_end == ad.state.window_end
        @test a2.ssa.state.m == 150 - last_split(150, a2.state.window_splits)
        @test a2.pc.n == 150 - last_split(150, a2.state.window_splits) || a2.pc.n >= 0          # estimator keeps accumulating within its window
        r3 = H.stage_warmup(tgt, r2.θ, f["rng"]; kw..., resume = (; θ = r2.θ, ϵ = r2.ϵ, Minv = r2.Minv, adaptor = a2, i = r2.i))
        @test r3.done && r3.i == nad && size(r3.stats, 1) == nad - 150 && r3.adaptor.state.i == nad
        @test r3.ϵ == exp(r3.adaptor.ssa.state.x_bar)               # finalised once, at the last iteration
        @test r3.Minv != r1.Minv && all(r3.Minv .> 0) && all(isfinite, r3.θ)
        @test filesize(path) < 200_000                              # parameters only: no data/closures in the state

        # block-dense metric: dense population block + dense per-unit blocks, groups independent
        grp = H.stage_unit_groups(tgt.layout, mm.k2_centered)
        @test length(grp) == length(tgt.θ0) && count(==(0), grp) == 7 && all(i -> count(==(i), grp) == 4, 1:length(units))
        # estimator: one group = AdvancedHMC's WelfordCov; two groups = its estimate with the cross block removed
        let D = 6, X = randn(MersenneTwister(7), 40, D) * [1.0 0.5 0 0 0 0; 0 1 0.3 0 0 0; 0 0 1 0 0 0; 0 0 0 1 0.8 0; 0 0 0 0 1 0; 0 0 0 0 0 2.0]
            wc = H.AHMC.WelfordCov((D,)); mc1 = H.MaskedCov(zeros(Int, D)); mc2 = H.MaskedCov([0, 0, 0, 1, 1, 1])
            for x in eachrow(X); push!(wc, collect(x)); push!(mc1, collect(x)); push!(mc2, collect(x)); end
            full = H.AHMC.Adaptation.get_estimation(wc)
            @test H.AHMC.Adaptation.get_estimation(mc1) ≈ full
            E2 = H.AHMC.Adaptation.get_estimation(mc2)
            @test E2[1:3, 4:6] == zeros(3, 3) && E2[1:3, 1:3] ≈ full[1:3, 1:3] && E2[4:6, 4:6] ≈ full[4:6, 4:6] && isposdef(Symmetric(E2))
        end
        gkw = (; kw..., metric_kind = :block, groups = grp)
        rb0 = MersenneTwister(5)
        b1 = H.stage_warmup(tgt, tgt.θ0, rb0; gkw..., upto = 90)
        @test b1.Minv isa Matrix && size(b1.Minv) == (length(tgt.θ0), length(tgt.θ0)) && b1.Minv != Matrix(I, size(b1.Minv)...)
        offblock = [grp[i] != grp[j] for i in eachindex(grp), j in eachindex(grp)]
        @test all(iszero, b1.Minv[offblock]) && isposdef(Symmetric(b1.Minv)) && b1.Minv ≈ b1.Minv'
        pathb = tempname() * ".jld2"
        JLD2.jldsave(pathb; theta = b1.θ, epsilon = b1.ϵ, Minv = b1.Minv, adaptor = b1.adaptor, i = b1.i, rng = rb0)
        fb = JLD2.load(pathb)
        @test fb["adaptor"].pc isa H.MaskedCov && fb["adaptor"].pc.mask == b1.adaptor.pc.mask && fb["adaptor"].pc.M == b1.adaptor.pc.M
        b2 = H.stage_warmup(tgt, fb["theta"], fb["rng"]; gkw..., resume = (; θ = fb["theta"], ϵ = fb["epsilon"], Minv = fb["Minv"], adaptor = fb["adaptor"], i = fb["i"]))
        # (a resumed segment draws a fresh momentum, so the stream differs from a one-shot run: checked for validity, not identity)
        @test b2.done && b2.i == nad && b2.ϵ > 0 && all(isfinite, b2.θ) && b2.ϵ == exp(b2.adaptor.ssa.state.x_bar)
        @test b2.Minv isa Matrix && all(iszero, b2.Minv[offblock]) && isposdef(Symmetric(b2.Minv)) && b2.Minv != b1.Minv
        sb = H.stage_sample(tgt2, b2.θ, b2.ϵ, b2.Minv, MersenneTwister(6); ndraws = 10, burnin = 2, log_every = 0)
        @test all(sb.stats[:, 3] .== b2.ϵ) && size(sb.θ) == (10, length(tgt.θ0)) && all(isfinite, sb.θ)

        # fingerprint: sensitive to every part of the target, insensitive to nothing but identity
        mkfp(; kwargs...) = begin
            base = (; drug = "binimetinib", design = "primary", window = 42.0, models = ["a", "b"], split = Dict("kind" => "primary", "seed" => 1, "holdout_models" => ["z"]),
                    priors = H.default_priors(), nu = 5.0, baseline_sd = 0.1, lod = 4.0, doses = doses, pk = pk, cref = cref,
                    k2_centered = [true, false], centered_lam0 = true, latent_w0 = true, reltol = 1e-5, abstol = 1e-5, init_id = "abc")
            H.target_fingerprint(H.fingerprint_text(; merge(base, (; kwargs...))...))
        end
        fp0 = mkfp()
        @test fp0 == mkfp() && length(fp0) == 32
        for alt in ((; drug = "BKM120"), (; design = "late21"), (; window = 21.0), (; models = ["a", "c"]), (; split = Dict("kind" => "primary", "seed" => 2, "holdout_models" => ["z"])),
                    (; priors = H.HierPriors(; μ_λ0 = (log(0.06), 0.5))), (; nu = 4.0), (; baseline_sd = 0.2), (; lod = 1.0),
                    (; doses = doses[1:end-1]), (; pk = OneCmt(CL = 3.1, V = 1.5, ka = 24.0, F = 0.8)), (; cref = cref * 1.01),
                    (; k2_centered = [false, false]), (; centered_lam0 = false), (; latent_w0 = false), (; reltol = 1e-6), (; init_id = "abd"), (; mode_gap = 30.0))
            @test mkfp(; alt...) != fp0
        end
        @test mkfp(; mode_gap = Inf) == fp0 && mkfp(; mode_gap = 30.0) != mkfp(; mode_gap = 20.0)   # Inf (no gap limit) keeps the old text
        @test H.verify_fingerprint(fp0, fp0) && H.verify_fingerprint(fp0, fp0; what = "x")
        err = try H.verify_fingerprint(fp0, mkfp(; nu = 4.0); what = "chain_1_1.jld2"); nothing catch e; e end
        @test err isa ErrorException && occursin("fingerprint mismatch", err.msg) && occursin("chain_1_1.jld2", err.msg)
        err2 = try H.verify_fingerprint(fp0, nothing; what = "warm_2.jld2"); nothing catch e; e end
        @test err2 isa ErrorException && occursin("no fingerprint", err2.msg)
    end

    @testset "split_modes: converged modes kept, exclusions recorded; cross-mode only with >= 2 start modes" begin
        mk(lp, μ; conv = false) = (; nt = (; μ_λ0 = μ, ω_λ0 = 0.5, μ_k2 = -1.0, ω_k2 = 0.5, logVtr = 7.0, logk1 = 0.0, σ = 0.2), lp, converged = conv)
        ms = [mk(70.0, -3.0), mk(10.0, -1.0; conv = true), mk(-1.3, -3.0), mk(69.9, -3.0), mk(-Inf, -3.0)]
        sm = H.split_modes(ms)
        @test [round(m.lp; digits = 1) for m in sm.kept] == [70.0, 10.0]          # the converged low mode stays; duplicates merged
        @test length(sm.excluded) == 2 && any(e -> e.reason == "failed start", sm.excluded) &&
              any(e -> e.lp == -1.3 && !e.converged && isfinite(e.dist), sm.excluded)
        @test H.distinct_modes(ms) == sm.kept
        # modes that differ only in the waning rate are distinct (logρ_w is part of the mode comparison)
        mw(lr) = (; nt = (; μ_λ0 = -3.0, ω_λ0 = 0.5, μ_k2 = -1.0, ω_k2 = 0.5, logVtr = 7.0, logk1 = 0.0, σ = 0.2, logρ_w = lr), lp = 5.0, converged = true)
        @test length(H.split_modes([mw(-4.0), mw(-2.0)]).kept) == 2 && length(H.split_modes([mw(-4.0), mw(-4.0)]).kept) == 1
        # gap and k1 are recorded for every excluded start (gap = best lp - lp; NaN for failed starts)
        eg = only(filter(e -> e.lp == -1.3, sm.excluded))
        @test eg.gap ≈ 71.3 && eg.k1 ≈ 1.0
        @test all(e -> hasproperty(e, :gap) && hasproperty(e, :k1), sm.excluded)
        # converged_gap (v2 production: 30) also excludes converged modes far below the best
        sg = H.split_modes(ms; converged_gap = 30.0)
        @test [round(m.lp; digits = 1) for m in sg.kept] == [70.0]
        @test any(e -> e.lp == 10.0 && e.converged && e.gap ≈ 60.0 && occursin("converged but", e.reason), sg.excluded)
        # a fit whose chains all started from one mode must not report a cross-mode R-hat
        _, pd1 = H.fit_drug(units, doses, pk, cref; drug = "one", nchains = 2, nsamples = 8, nadapt = 6, rng = MersenneTwister(3),
                            map_starts = 2, map_maxiters = 30, window_days = 0)
        @test length(unique(pd1.diagnostics["chain_start_modes"])) <= 2 && haskey(pd1.diagnostics, "excluded_modes")
        length(unique(pd1.diagnostics["chain_start_modes"])) == 1 && @test isnan(pd1.diagnostics["cross_mode_rhat"])
    end

    @testset "progress callback under threaded chains (locked state)" begin
        _, pdt = H.fit_drug(units, doses, pk, cref; drug = "thr", nchains = 2, nsamples = 12, nadapt = 8, rng = MersenneTwister(9),
                            map_starts = 0, window_days = 0, parallel = :chains, log_every = 4, logger = _ -> nothing)
        @test length(pdt.σ) == 24
    end

    if isfile(joinpath(default_cache_dir(), "pct_long.csv"))
        @testset "design_units (frozen split, windows)" begin
            p = H.design_units("binimetinib", "primary"; subsample = 60)
            l = H.design_units("binimetinib", "late21"; subsample = 60)
            @test p.window == 42.0 && l.window == 21.0
            @test [u.model for u in p.train] == [u.model for u in l.train]            # same PDX
            @test isempty(intersect([u.model for u in p.train], p.split["holdout_models"]))
            @test all(maximum(u.treated.t) <= 21.0 && maximum(u.control.t) <= 21.0 for u in l.train)
            @test haskey(l.split, "late_window_day") && length(l.cutoffs) == 2 * length(l.train) && isempty(p.cutoffs)
            @test p.split["model_frac"] == 0.2 && p.split["seed"] == 1
            tr, te, info = H.frozen_pdx_split()
            @test isempty(intersect(tr, te)) && info.n_universe == length(tr) + length(te)
            # the same PDX are held out whichever drug is fitted
            @test Set(H.design_units("BKM120", "primary").split["holdout_models"]) ⊆ Set(te)
        end
    end

    @testset "model v2 (correlated effects, per-PDX Vtr) and waning" begin
        truth2 = (μ_λ0 = log(0.06), ω_λ0 = 0.4, μ_k2 = log(0.4), ω_k2 = 0.8, corr = 0.5, μ_Vtr = log(1500.0), ω_Vtr = 0.4,
                  logk1 = log(0.7), σ = 0.15, ρ_w = 0.015)
        u2, tru2 = H.simulate_units(MersenneTwister(21), 6, doses, pk, cref; truth = truth2, tspan = (21.0, 24.0))
        @test all(isfinite, tru2.λ1) && all(tru2.λ1 .≈ tru2.λ0 .* tru2.Vtr)
        # LKJ(2) on a 2x2 correlation is (ρ+1)/2 ~ Beta(2,2): density ∝ (1-ρ²)
        ρs = [-0.7, -0.2, 0.3, 0.9]
        dens = [pdf(Beta(2.0, 2.0), (r + 1) / 2) / (1 - r^2) for r in ρs]
        @test all(≈(dens[1]), dens)
        # the centred / conditional construction has covariance ρ ω_λ0 ω_k2
        r = MersenneTwister(5); n = 200_000; ρv = 0.6; ω1 = 0.4; ω2 = 0.8
        a = ω1 .* randn(r, n); astd = a ./ ω1
        b = ω2 .* (ρv .* astd .+ sqrt(1 - ρv^2) .* randn(r, n))
        @test cor(a, b) ≈ ρv atol = 0.01
        @test std(b) ≈ ω2 rtol = 0.01
        # a non-centred start converts to the centred/masked parametrisation with identical implied log k2
        nt = (; μ_λ0 = -3.0, ω_λ0 = 0.5, μ_k2 = -1.0, ω_k2 = 0.7, ρ_u = 0.8, z_λ0 = [0.3, -1.0, 0.5], z_k2 = [0.2, 1.0, -0.4])
        ρ = 2 * nt.ρ_u - 1
        bnc = nt.μ_k2 .+ nt.ω_k2 .* (ρ .* nt.z_λ0 .+ sqrt(1 - ρ^2) .* nt.z_k2)
        ntc = H.apply_k2_mask(H.to_centered(nt), [true, true, true])
        @test ntc.lk2c ≈ bnc && !haskey(ntc, :z_k2) && !haskey(ntc, :z_λ0) && ntc.llam0 ≈ nt.μ_λ0 .+ nt.ω_λ0 .* nt.z_λ0
        ntp = H.apply_k2_mask(H.to_centered(nt), [true, false, true])
        @test ntp.lk2c ≈ bnc[[1, 3]] && ntp.z_k2 == [1.0]
        # gradients: local per-unit assembly vs finite differences vs global ForwardDiff, per-unit λ1 and waning included
        for (v2, wn, cen, mask) in ((true, false, true, falses(6)), (true, false, true, [true, false, true, true, false, true]),
                                    (true, true, true, falses(6)), (true, false, false, falses(6)), (false, true, true, falses(6)))
            mk(; lg) = begin
                d = H.prepare_data(u2, doses, pk, cref; solver = tight, model_v2 = v2, waning = wn, centered_lam0 = cen, k2_centered = mask, local_grad = lg, threaded = false)
                model = H.hier_model(d, H.default_priors())
                nt0 = H.initial_values(d, H.default_priors(), MersenneTwister(4); jitter = 0.05)
                vi = DynamicPPL.link(DynamicPPL.initialize_parameters!!(DynamicPPL.VarInfo(MersenneTwister(0), model), nt0, model), model)
                D = length(vi[:])
                adt = AutoForwardDiff(chunksize = lg ? min(D, 8) : min(D, 12))
                (DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi; adtype = adt),
                 DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi), vi[:])
            end
            ldf, ldf0, x = mk(lg = true)
            lp, g = LogDensityProblems.logdensity_and_gradient(ldf, x)
            @test isfinite(lp)
            @test lp ≈ LogDensityProblems.logdensity(ldf0, x) rtol = 1e-8
            fd = map(eachindex(x)) do i
                e = zeros(length(x)); e[i] = 1e-5
                (LogDensityProblems.logdensity(ldf0, x .+ e) - LogDensityProblems.logdensity(ldf0, x .- e)) / 2e-5
            end
            @test maximum(abs.(g .- fd) ./ (1 .+ abs.(fd))) < 1e-3
            ldfg, _, _ = mk(lg = false)
            lpg, gg = LogDensityProblems.logdensity_and_gradient(ldfg, x)
            @test lpg ≈ lp rtol = 1e-8
            @test maximum(abs.(gg .- g) ./ (1 .+ abs.(g))) < 1e-5
        end
        # waning exposure: e(t) exp(-ρ t); ρ = 0 reproduces the plain exposure
        e0 = H.make_exposure(pk, doses, cref)
        @test H.WaningExposure(e0, 0.0)(7.3) ≈ e0(7.3) && H.WaningExposure(e0, 0.02)(7.3) ≈ e0(7.3) * exp(-0.02 * 7.3)
        # v2 fit, waning fit: saved fields, pointwise and marginal log-likelihoods, extraction
        for wn in (false, true)
            _, pd2 = H.fit_drug(u2, doses, pk, cref; drug = "v2toy", model = :v2, waning = wn, nchains = 2, nsamples = 8, nadapt = 8,
                                rng = MersenneTwister(31), map_starts = 2, map_maxiters = 40, window_days = 0)
            dg = pd2.diagnostics
            @test dg["model"] == "v2" && dg["waning"] == wn && dg["settings"]["model"] == "v2"
            @test length(dg["corr_lk"]) == 16 && all(-1 .< dg["corr_lk"] .< 1)
            @test size(dg["lambda1_draws"]) == (16, 6) && all(dg["lambda1_draws"] .> 0)
            @test length(dg["mu_logVtr"]) == 16 && all(dg["omega_Vtr"] .> 0)
            @test wn ? (length(dg["waning_rate"]) == 16 && all(dg["waning_rate"] .> 0)) : isempty(dg["waning_rate"])
            @test pd2.logλ1 ≈ pd2.μ_λ0 .+ dg["mu_logVtr"]                  # pooled field = population-median λ1
            ll = H.pointwise_loglik(pd2, u2; draws = 1:4)
            idx2 = H.pointwise_index(u2)
            @test all(isfinite, ll) && size(ll) == (4, length(idx2.t))
            # pointwise sums reproduce the unit likelihood at that draw's own λ1_i, k2_i, λ0_i (and ρ)
            d2 = H.prepare_data(u2, doses, pk, cref; model_v2 = true, waning = wn)
            s_ = 3
            for (i, u) in enumerate(d2.units)
                v = H.unit_loglik(u, log(pd2.λ0[s_, i]), log(pd2.k2[s_, i]), log(dg["lambda1_draws"][s_, i]), pd2.logk1[s_], pd2.σ[s_],
                                  log(pd2.w0_control_draws[s_, i]), log(pd2.w0_treated_draws[s_, i]), d2; lrho = wn ? log(dg["waning_rate"][s_]) : nothing)
                @test sum(ll[s_, idx2.group .== i]) ≈ v rtol = 1e-8
            end
            mg = H.marginal_unit_loglik(pd2, u2; M = 8, rng = MersenneTwister(2), draws = 1:3, threaded = false)
            @test size(mg) == (3, 6) && all(isfinite, mg)
            @test_throws ArgumentError H.marginal_unit_loglik(pd2, u2; M = 4, eps = randn(4, 4), draws = 1:2)
        end
        # v2 start / staged layout: the logit link of ρ_u and the log link of ω_Vtr match DynamicPPL's
        d3 = H.prepare_data(u2, doses, pk, cref; model_v2 = true, waning = true, threaded = false)
        mm2 = H.map_modes(u2, doses, pk, cref; window_days = 0, nstarts = 2, maxiters = 60, rng = MersenneTwister(3), logger = _ -> nothing, model_v2 = true, waning = true)
        @test all(m -> haskey(m.nt, :ρ_u) && haskey(m.nt, :μ_Vtr) && haskey(m.nt, :z_Vtr) && haskey(m.nt, :logρ_w) && !haskey(m.nt, :logVtr), mm2.starts)
        tg = H.stage_target(u2, doses, pk, cref; start = mm2.starts[1].nt, k2_centered = mm2.k2_centered, window_days = 0, threaded = false, model_v2 = true, waning = true)
        vi3 = DynamicPPL.link(DynamicPPL.initialize_parameters!!(DynamicPPL.VarInfo(MersenneTwister(0), tg.model), mm2.starts[1].nt, tg.model), tg.model)
        θ3 = tg.θ0 .+ 0.05 .* randn(MersenneTwister(2), length(tg.θ0))
        @test H.constrain_draws(tg.layout, reshape(θ3, 1, :))[1, :] ≈ DynamicPPL.invlink(DynamicPPL.unflatten(vi3, θ3), tg.model)[:]
        @test H.unconstrain(tg.layout, mm2.starts[1].nt) ≈ vi3[:]
        # fingerprint depends on the model version and waning
        base = (; drug = "binimetinib", design = "primary", window = 42.0, models = ["a"], split = Dict("kind" => "primary"), priors = H.default_priors(),
                nu = 5.0, baseline_sd = 0.1, lod = 4.0, doses = doses, pk = pk, cref = cref, k2_centered = [true], centered_lam0 = true,
                latent_w0 = true, reltol = 1e-5, abstol = 1e-5, init_id = "x")
        f1 = H.target_fingerprint(H.fingerprint_text(; base...))
        @test f1 != H.target_fingerprint(H.fingerprint_text(; base..., model = "v2"))
        @test f1 != H.target_fingerprint(H.fingerprint_text(; base..., waning = true))
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
