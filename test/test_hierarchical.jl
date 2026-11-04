using Test, Random, Statistics
using TumourPKPD
using TumourPKPD: Dose, OneCmt, dosing_schedule, conc, PKExposure
using Turing
isdefined(TumourPKPD, :fit_drug) ||
    Base.include(TumourPKPD, joinpath(@__DIR__, "..", "src", "hierarchical.jl"))
const H = TumourPKPD
const DynamicPPL = Turing.DynamicPPL
const LogDensityProblems = Turing.LogDensityProblems
using ForwardDiff

@testset "hierarchical" begin
    pk = OneCmt(CL = 3.0, V = 1.5, ka = 24.0, F = 0.8)
    doses = dosing_schedule(25.0; start = 0.0, interval = 1.0, n = 21)
    cref = maximum(conc(pk, doses, 0:0.02:21.0))
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
        # censoring and baseline handling
        c = (; t = [0.0, 3.0, 7.0, 10.0], v = [150.0, 0.0, 40.0, -1.0])
        cd = H.prepare_curve(c)
        @test cd.v0 == 150.0
        @test cd.t == [3.0, 7.0]               # day 0 and negative volume dropped
        @test cd.cens == [true, false]
        @test cd.logv[2] ≈ log(40.0)
    end

    function make_ldf(; latent = false, local_grad = true, threaded = true, adtype = true, chunk = nothing)
        d = H.prepare_data(units, doses, pk, cref; latent_w0 = latent, local_grad, threaded, solver = tight)
        model = H.hier_model(d, H.default_priors())
        D = 7 + 2 * 3 + (latent ? 6 : 0)
        adt = AutoForwardDiff(chunksize = chunk === nothing ? (local_grad ? D : 12) : chunk)
        vi = DynamicPPL.VarInfo(MersenneTwister(3), model)
        nt = merge(H.initial_values(d, H.default_priors(), MersenneTwister(4)),
                   (; μ_k2 = log(0.5), logk1 = log(0.7), logλ1 = log(120.0), σ = 0.2))
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
                h = 1e-6; e = zeros(length(x)); e[i] = h
                (LogDensityProblems.logdensity(ldf0, x .+ e) - LogDensityProblems.logdensity(ldf0, x .- e)) / 2h
            end
            @test maximum(abs.(g .- fd) ./ (1 .+ abs.(fd))) < 1e-4
            # global ForwardDiff (no local-gradient shortcut) gives the same gradient
            ldfg, _, _ = make_ldf(; latent, local_grad = false)
            lpg, gg = LogDensityProblems.logdensity_and_gradient(ldfg, x)
            @test lpg ≈ lp rtol = 1e-8
            @test maximum(abs.(gg .- g) ./ (1 .+ abs.(g))) < 1e-6
            # small chunk: several forward passes share one cached set of local gradients
            ldfc, _, dc = make_ldf(; latent, chunk = 4)
            lpc, gc = LogDensityProblems.logdensity_and_gradient(ldfc, x)
            @test lpc ≈ lp rtol = 1e-10
            @test gc ≈ g rtol = 1e-8
            @test dc.cache.hits >= 2
        end
    end

    @testset "threaded == serial" begin
        _, x, dt = make_ldf(; threaded = true)
        _, _, ds = make_ldf(; threaded = false)
        args = (fill(log(0.08), 3), fill(log(0.5), 3), [log(u.control.v0) for u in dt.units],
                [log(u.treated.v0) for u in dt.units], log(120.0), log(0.7), 0.2)
        @test H.population_loglik(dt, args...) == H.population_loglik(ds, args...)
        @test Threads.nthreads() >= 1
    end

    @testset "ODE failure gives -Inf" begin
        _, _, d = make_ldf()
        u = d.units[1]
        # absurd rates make the solve fail or the likelihood non-finite; must not throw or return NaN
        v = H.unit_loglik(u, log(1e6), log(1e6), log(1e6), log(1e6), 0.2, log(u.control.v0), log(u.treated.v0), d)
        @test !isnan(v)
    end

    @testset "fit_drug, PosteriorDraws, save/load, pointwise" begin
        chain, pd = H.fit_drug(units, doses, pk, cref; drug = "toy", nchains = 2, nsamples = 20,
                               nadapt = 20, rng = MersenneTwister(5), split = "late", cutoffs = Dict("syn1" => 14.0), seed = 5, reltol = 1e-5, abstol = 1e-5)
        @test pd isa H.PosteriorDraws
        @test length(pd.σ) == 40
        @test size(pd.λ0) == (40, 3) && size(pd.k2) == (40, 3)
        @test pd.models == ["syn1", "syn2", "syn3"]
        @test all(pd.λ0 .> 0) && all(pd.k2 .> 0)
        @test haskey(pd.diagnostics, "max_rhat") && haskey(pd.diagnostics, "runtime_s")
        @test pd.diagnostics["settings"]["nchains"] == 2
        @test pd.diagnostics["nchains"] == 2 && pd.diagnostics["draws_per_chain"] == 20
        @test pd.diagnostics["train_models"] == pd.models && pd.diagnostics["split"] == "late"
        @test pd.diagnostics["cutoffs"] == Dict("syn1" => 14.0) && pd.diagnostics["lod"] == H._default_lod()
        path = joinpath(mktempdir(), "toy.jld2")
        H.save_fit(path, pd)
        pd2 = H.load_fit(path)
        @test pd2.drug == pd.drug && pd2.models == pd.models
        @test pd2.λ0 == pd.λ0 && pd2.k2 == pd.k2 && pd2.σ == pd.σ
        @test pd2.doses == pd.doses && pd2.pk == pd.pk && pd2.cref == pd.cref
        @test pd2.diagnostics["settings"]["lod"] == H._default_lod()

        ll = H.pointwise_loglik(pd, units; draws = 1:5)
        idx = H.pointwise_index(units)
        @test size(ll) == (5, length(idx.t))
        @test all(isfinite, ll)
        # sums of pointwise terms reproduce the unit-level likelihood
        d = H.prepare_data(units, doses, pk, cref)
        s = 3
        for (i, u) in enumerate(d.units)
            v = H.unit_loglik(u, log(pd.λ0[s, i]), log(pd.k2[s, i]), pd.logλ1[s], pd.logk1[s], pd.σ[s],
                              log(u.control.v0), log(u.treated.v0), d)
            @test sum(ll[s, idx.group .== i]) ≈ v rtol = 1e-8
        end
        @test size(H.grouped_loglik(ll, idx)) == (5, 3)
    end
end
