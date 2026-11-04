# Limiting-case, conservation and derivative checks for the Simeoni model in src/pd.jl.
# Run: julia --project=. -t 2 test/test_pd.jl
using Test
using LinearAlgebra
using TumourPKPD
using ForwardDiff
using OrdinaryDiffEq, SciMLBase

# Constant exposure, handy for analytic checks (the package ones are NoDrug / PKExposure).
struct ConstExp{T<:Real}
    c::T
end
(e::ConstExp)(t) = e.c
TumourPKPD.dose_times(::ConstExp) = Float64[]
TumourPKPD._exposure_eltype(e::ConstExp) = e.c

const TIGHT = (reltol = 1e-11, abstol = 1e-11, solver = Vern9())

# central finite difference of a scalar function of a vector
function fd_grad(f, x; h = 1e-5)
    g = similar(x, Float64)
    for i in eachindex(x)
        hi = h * max(abs(x[i]), 1e-3)
        xp = copy(x); xp[i] += hi
        xm = copy(x); xm[i] -= hi
        g[i] = (f(xp) - f(xm)) / (2hi)
    end
    return g
end

@testset "Simeoni PD" begin

    @testset "PKExposure equals conc/cref (sorted-dose and one-compartment-state paths)" begin
        ts = sort(vcat(collect(0.0:0.37:60.0), [-1.0, 0.0, 1.0, 5.0, 5.0 + 1e-9, 12.0, 339.5, 400.0]))
        # unsorted doses, duplicated dose time, oral + iv mix
        doses = [Dose(7.0, 5.0), Dose(0.0, 10.0), Dose(3.5, 2.0; route = :iv), Dose(7.0, 1.0),
                 Dose(14.0, 10.0), Dose(1.0, 4.0, :oral)]
        for pk in (OneCmt(CL = 1.0, V = 2.0, ka = 3.0, F = 0.8),
                   OneCmt(CL = 1.0, V = 2.0, ka = 0.2, F = 0.8),            # flip-flop
                   OneCmt(CL = 1.0, V = 2.0, ka = 0.5, F = 0.8),            # ka == k exactly (loop path)
                   OneCmt(CL = 1.0, V = 2.0, ka = 0.5004, F = 0.8),         # ka ≈ k (loop path)
                   TwoCmt(CL = 1.0, V1 = 2.0, Q = 0.7, V2 = 3.0, ka = 2.0, F = 0.8))
            ex = PKExposure(pk, doses, 3.0)
            ref = [conc(pk, doses, t) / 3.0 for t in ts]
            @test [ex(t) for t in ts] ≈ ref rtol = 1e-9 atol = 1e-12
        end
        @test TumourPKPD.dose_times(PKExposure(OneCmt(CL = 1.0, V = 2.0, ka = 3.0), doses)) ==
              [0.0, 1.0, 3.5, 7.0, 14.0]
        # long, fast regimen (bid for 340 days, k = 3.4/day): no overflow, matches the direct sum
        long = [Dose(d + h, 30.0) for d in 0:339 for h in (0.0, 0.5)]
        pkl = OneCmt(CL = 8.2368, V = 2.4, ka = 48.0, F = 0.54)
        exl = PKExposure(pkl, long, 2.0)
        tl = [0.0, 0.2, 0.5, 0.75, 17.3, 100.25, 339.5, 339.9, 345.0]
        @test [exl(t) for t in tl] ≈ [conc(pkl, long, t) / 2.0 for t in tl] rtol = 1e-9
        # derivatives w.r.t. PK parameters and cref agree with the direct formulation
        for t in (0.4, 7.2, 20.0)
            f1(q) = PKExposure(OneCmt(CL = q[1], V = q[2], ka = q[3], F = 0.8), doses, 3.0)(t)
            f2(q) = conc(OneCmt(CL = q[1], V = q[2], ka = q[3], F = 0.8), doses, t) / 3.0
            q0 = [1.0, 2.0, 3.0]
            @test ForwardDiff.gradient(f1, q0) ≈ ForwardDiff.gradient(f2, q0) rtol = 1e-8
        end
    end

    @testset "no drug, small w: exponential growth" begin
        # w_th = λ1/λ0 = 500, stay below ~300 where the saturation factor is 1 - O(1e-6)
        θ = Simeoni(0.2, 100.0, 1.0, 0.5, 1.0)
        t = collect(0.0:2.0:28.0)
        w, ok = simulate_tumour(θ, NoDrug(), t; TIGHT...)
        @test ok
        @test all(isapprox.(w, 1.0 .* exp.(0.2 .* t); rtol = 1e-5))
    end

    @testset "no drug, large w: linear growth at rate λ1" begin
        θ = Simeoni(0.2, 100.0, 1.0, 0.5, 5000.0)   # w >> w_th = 500
        t = collect(0.0:5.0:30.0)
        w, ok = simulate_tumour(θ, NoDrug(), t; TIGHT...)
        @test ok
        @test all(isapprox.(w, 5000.0 .+ 100.0 .* t; rtol = 1e-8))
        # derivative via the right-hand side
        du = zeros(4)
        TumourPKPD.simeoni_rhs!(du, [8000.0, 0, 0, 0], (θ, NoDrug()), 0.0)
        @test sum(du) ≈ 100.0 rtol = 1e-8
    end

    @testset "switch point near λ1/λ0" begin
        λ0, λ1 = 0.2, 100.0
        wth = λ1 / λ0
        θ = Simeoni(λ0, λ1, 1.0, 0.0, 10.0)
        # growth rate exactly at w_th is λ1 * 2^(-1/ψ)
        du = zeros(4)
        TumourPKPD.simeoni_rhs!(du, [wth, 0, 0, 0], (θ, NoDrug()), 0.0)
        @test du[1] ≈ λ1 * 2.0^(-1 / θ.ψ) rtol = 1e-12
        # simulated trajectory follows exponential-then-linear, with the switch where w = w_th
        ts = log(wth / θ.w0) / λ0
        t = [0.0, ts - 5, ts, ts + 5, ts + 20]
        w, ok = simulate_tumour(θ, NoDrug(), t; TIGHT...)
        @test ok
        @test w[2] ≈ θ.w0 * exp(λ0 * (ts - 5)) rtol = 2e-3
        @test w[end] ≈ wth + λ1 * 20 rtol = 0.05
        # growth is exponential below and linear above: dw/dt/w falls from λ0 to ~0
        tt = collect(range(0.0, ts + 30; length = 400))
        ww, _ = simulate_tumour(θ, NoDrug(), tt; TIGHT...)
        slope = diff(ww) ./ diff(tt)
        @test all(slope .> 0)
        @test maximum(slope) <= λ1 * 1.0001      # never faster than the linear rate
    end

    @testset "linear chain when λ0 → 0: exact matrix exponential and mass balance" begin
        # d/dt(x1+…+x4) = -k1 x4 exactly when λ0 = 0 (checked on the RHS for random states)
        θ0 = Simeoni(0.0, 100.0, 0.7, 0.9, 50.0)
        for _ in 1:20
            u = 100 .* rand(4)
            du = zeros(4)
            TumourPKPD.simeoni_rhs!(du, u, (θ0, ConstExp(rand() + 0.1)), 0.0)
            @test sum(du) ≈ -θ0.k1 * u[4] atol = 1e-12
        end
        # full solve vs exp(At) (λ0 tiny but positive: simulate_tumour requires λ0 > 0)
        λ0 = 1e-12; k1 = 0.7; k2 = 0.9; c = 1.3; w0 = 50.0
        a = k2 * c
        A = [-a 0 0 0; a -k1 0 0; 0 k1 -k1 0; 0 0 k1 -k1]
        t = [0.0, 0.5, 1.0, 3.0, 7.0, 15.0]
        w, ok = simulate_tumour(Simeoni(λ0, 100.0, k1, k2, w0), ConstExp(c), t; TIGHT...)
        @test ok
        wex = [sum(exp(A * ti) * [w0, 0, 0, 0]) for ti in t]
        @test w ≈ wex rtol = 1e-8
        # mass balance: w(t) = w0 - k1 ∫ x4 dt  <=> w(T) + k1 ∫x4 = w0 (x4 by quadrature of exp(At))
        T = 15.0
        ts = range(0, T; length = 2001)
        x4 = [(exp(A * s)*[w0, 0, 0, 0])[4] for s in ts]
        h = step(ts)
        integral = h / 3 * (x4[1] + x4[end] + 4 * sum(x4[2:2:end-1]) + 2 * sum(x4[3:2:end-2]))
        @test wex[end] + k1 * integral ≈ w0 rtol = 1e-8
    end

    @testset "large k1: damaged cells clear instantly (w → x1)" begin
        # λ0 tiny: w should follow w0 exp(-k2 c t)
        θ = Simeoni(1e-12, 100.0, 1000.0, 0.8, 100.0)
        t = [0.0, 1.0, 2.0, 5.0]
        w, ok = simulate_tumour(θ, ConstExp(1.0), t; solver = Rodas5P(), reltol = 1e-9, abstol = 1e-9)
        @test ok
        @test w ≈ 100.0 .* exp.(-0.8 .* t) rtol = 2e-3
        # with growth: compare to the single-state model dx = λ0 x g(λ0 x/λ1) - k2 c x
        λ0, λ1, k2 = 0.2, 100.0, 0.5
        f(x, p, s) = λ0 * x * TumourPKPD.growth_factor(λ0 * x / λ1, 20.0) - k2 * 0.4 * x
        sol = solve(ODEProblem(f, 100.0, (0.0, 30.0)), Vern9(); reltol = 1e-11, abstol = 1e-11,
                    saveat = [0.0, 5.0, 15.0, 30.0])
        w2, ok2 = simulate_tumour(Simeoni(λ0, λ1, 1000.0, k2, 100.0), ConstExp(0.4),
                                  [0.0, 5.0, 15.0, 30.0]; solver = Rodas5P(), reltol = 1e-9, abstol = 1e-9)
        @test ok2
        @test w2 ≈ sol.u rtol = 5e-3
    end

    @testset "k2 = 0 equals NoDrug; monotone in k2" begin
        t = collect(0.0:3.0:30.0)
        pk = OneCmt(CL = 1.0, V = 2.0, ka = 3.0, F = 0.8)
        doses = dosing_schedule(10.0; n = 21)
        ex = PKExposure(pk, doses, 5.0)
        θ(k2) = Simeoni(0.15, 80.0, 0.5, k2, 150.0)
        # (tstops at the dose times change the step sequence, so compare at tight tolerance)
        w0, _ = simulate_tumour(θ(0.0), NoDrug(), t; TIGHT...)
        wz, okz = simulate_tumour(θ(0.0), ex, t; TIGHT...)
        @test okz
        @test wz ≈ w0 rtol = 1e-9
        prev = wz
        for k2 in (0.05, 0.2, 0.5, 1.0, 3.0, 10.0)
            w, ok = simulate_tumour(θ(k2), ex, t)
            @test ok
            @test all(w[2:end] .< prev[2:end])
            prev = w
        end
    end

    @testset "ForwardDiff vs finite differences" begin
        t = [0.0, 3.0, 7.0, 10.0, 14.0, 18.0, 21.0, 25.0, 30.0]
        doses = dosing_schedule(10.0; n = 21)
        tight = (reltol = 1e-11, abstol = 1e-11, solver = Vern9())
        # PD parameters, fixed PK
        pk = OneCmt(CL = 1.0, V = 2.0, ka = 3.0, F = 0.8)
        ex = PKExposure(pk, doses, 5.0)
        function ll_pd(p)
            w, ok = simulate_tumour(Simeoni(p[1], p[2], p[3], p[4], 150.0), ex, t; tight...)
            ok || error("solver failed")
            return sum(log.(w))
        end
        p0 = [0.15, 80.0, 0.5, 0.8]
        g = ForwardDiff.gradient(ll_pd, p0)
        gfd = fd_grad(ll_pd, p0)
        @test g ≈ gfd rtol = 1e-5
        # in the exponential-to-linear transition regime too
        p1 = [0.3, 40.0, 1.5, 0.3]
        @test ForwardDiff.gradient(ll_pd, p1) ≈ fd_grad(ll_pd, p1) rtol = 1e-5
        # PK parameters (CL, V, ka) through PKExposure
        function ll_pk(q)
            pk = OneCmt(CL = q[1], V = q[2], ka = q[3], F = 0.8)
            w, ok = simulate_tumour(Simeoni(0.15, 80.0, 0.5, 0.8, 150.0), PKExposure(pk, doses, 5.0), t; tight...)
            ok || error("solver failed")
            return sum(log.(w))
        end
        q0 = [1.0, 2.0, 3.0]
        @test ForwardDiff.gradient(ll_pk, q0) ≈ fd_grad(ll_pk, q0) rtol = 1e-5
        # all seven at once
        function ll_all(r)
            pk = OneCmt(CL = r[5], V = r[6], ka = r[7], F = 0.8)
            w, ok = simulate_tumour(Simeoni(r[1], r[2], r[3], r[4], 150.0), PKExposure(pk, doses, 5.0), t; tight...)
            ok || error("solver failed")
            return sum(log.(w))
        end
        r0 = [p0; q0]
        @test ForwardDiff.gradient(ll_all, r0) ≈ fd_grad(ll_all, r0) rtol = 1e-5
        # default (loose) tolerances: gradient close enough for NUTS
        function ll_loose(p)
            w, _ = simulate_tumour(Simeoni(p[1], p[2], p[3], p[4], 150.0), ex, t)
            return sum(log.(w))
        end
        @test ForwardDiff.gradient(ll_loose, p0) ≈ gfd rtol = 1e-3
    end

    @testset "solver success for extreme-but-plausible parameters" begin
        t = collect(0.0:2.0:40.0)
        pk = OneCmt(CL = 1.0, V = 2.0, ka = 3.0, F = 0.8)
        ex = PKExposure(pk, dosing_schedule(10.0; n = 28), 5.0)
        cases = [
            Simeoni(1.0, 500.0, 1.0, 0.5, 5000.0),        # λ0 = 1/day, huge tumour
            Simeoni(1.0, 5000.0, 1.0, 2.0, 5.0),
            Simeoni(0.5, 2000.0, 10.0, 50.0, 5000.0),     # very large k2 and k1
            Simeoni(0.01, 5.0, 0.05, 1e-3, 100.0),        # slow growth, slow death
            Simeoni(0.3, 1e4, 3.0, 100.0, 1.0),           # near-total kill
            Simeoni(0.2, 100.0, 0.02, 5.0, 300.0),        # very slow clearance
        ]
        # Cases 2, 3, 5 under treatment eradicate the tumour (true w < 1e-6 mm³ before day 40). There
        # the returned w is only accurate to ~abstol and can be slightly NEGATIVE (observed: ~ -2e-7).
        # Callers must floor w before taking logs; see notes/worker-pd.md.
        eradicated(i, expo) = expo isa PKExposure && i in (2, 3, 5)
        for (i, θ) in enumerate(cases), expo in (ex, NoDrug())
            for solver in (Tsit5(), AutoTsit5(Rosenbrock23()))
                w, ok = simulate_tumour(θ, expo, t; solver)
                @test ok
                @test all(isfinite, w)
                if eradicated(i, expo)
                    @test all(w .> -1e-5)          # negative only at the abstol scale
                    @test w[end] < 1e-3
                else
                    @test all(w .> 0)
                end
            end
        end
        # ForwardDiff stays finite in the stiff-ish corner
        f(p) = sum(log.(max.(first(simulate_tumour(Simeoni(p[1], 2000.0, p[2], p[3], 5000.0), ex, t)), 1e-3)))
        @test all(isfinite, ForwardDiff.gradient(f, [0.5, 10.0, 50.0]))
        # invalid parameters report failure instead of throwing
        wbad, okbad = simulate_tumour(Simeoni(-0.1, 100.0, 1.0, 1.0, 100.0), NoDrug(), t)
        @test !okbad
    end

    @testset "observation times: zero, duplicates, empty, bad input" begin
        θ = Simeoni(0.2, 100.0, 1.0, 0.5, 100.0)
        pk = OneCmt(CL = 1.0, V = 2.0, ka = 3.0, F = 0.8)
        ex = PKExposure(pk, dosing_schedule(10.0; n = 21), 5.0)
        wref, _ = simulate_tumour(θ, ex, [0.0, 1.0, 2.0, 3.0])
        w, ok = simulate_tumour(θ, ex, [0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0])
        @test ok
        @test w == wref[[1, 1, 2, 2, 3, 4, 4]]
        @test w[1] == θ.w0
        w, ok = simulate_tumour(θ, ex, [0.0, 0.0])
        @test ok && w == [θ.w0, θ.w0]
        w, ok = simulate_tumour(θ, ex, [2.0, 2.0])
        @test ok && w[1] == w[2] && w[1] ≈ wref[3]
        w, ok = simulate_tumour(θ, ex, Float64[])
        @test ok && isempty(w)
        @test_throws ArgumentError simulate_tumour(θ, ex, [2.0, 1.0])
        @test_throws ArgumentError simulate_tumour(θ, ex, [-1.0, 1.0])
        # dose exactly at an observation time and observation inside a dose-free gap
        ex2 = PKExposure(pk, [Dose(0.0, 10.0), Dose(5.0, 10.0)], 5.0)
        w, ok = simulate_tumour(θ, ex2, [0.0, 4.99, 5.0, 5.01, 20.0])
        @test ok && all(isfinite, w)
        # ForwardDiff with duplicates and zero
        g = ForwardDiff.gradient(p -> sum(log.(first(simulate_tumour(Simeoni(p[1], 100.0, 1.0, 0.5, 100.0), ex,
                                                               [0.0, 0.0, 1.0, 1.0, 3.0])))), [0.2])
        @test isfinite(g[1])
    end
end
