# Analytic and cross-solver checks for the closed-form PK in src/pk.jl.
# Run: julia --project=. -t 2 test/test_pk.jl
using Test
using TumourPKPD
using ForwardDiff
using OrdinaryDiffEq, SciMLBase

# The MTK cross-check models live in src/pk_mtk.jl. If the package module does not
# include them yet, load them into a private module that sees the exported API.
const PKMTK = if isdefined(TumourPKPD, :mtk_onecmt)
    TumourPKPD
else
    let m = Module(:PKMTK)
        Core.eval(m, :(using TumourPKPD))
        Base.include(m, joinpath(@__DIR__, "..", "src", "pk_mtk.jl"))
        m
    end
end

# Composite Simpson rule on [a, b] with n (even) subintervals.
function simpson(f, a, b, n)
    h = (b - a) / n
    s = f(a) + f(b)
    for i in 1:(n - 1)
        s += (iseven(i) ? 2 : 4) * f(a + i * h)
    end
    return s * h / 3
end

relerr(x, y) = abs(x - y) / abs(y)

# Hand-coded linear ODE with depot (independent of both pk.jl and ModelingToolkit).
function ode_conc(pk, doses, ts; reltol = 1e-11, abstol = 1e-13)
    twocmt = pk isa TwoCmt
    function rhs!(du, u, p, t)
        if twocmt
            k10 = pk.CL / pk.V1; k12 = pk.Q / pk.V1; k21 = pk.Q / pk.V2
            du[1] = -pk.ka * u[1]
            du[2] = pk.ka * u[1] - (k10 + k12) * u[2] + k21 * u[3]
            du[3] = k12 * u[2] - k21 * u[3]
        else
            k = pk.CL / pk.V
            du[1] = -pk.ka * u[1]
            du[2] = pk.ka * u[1] - k * u[2]
        end
    end
    n = twocmt ? 3 : 2
    Vc = twocmt ? pk.V1 : pk.V
    u0 = zeros(n)
    dts = Float64[]
    for d in doses
        d.time < 0 && throw(ArgumentError("dose time < 0 not supported"))
        if d.time == 0
            u0[1] += pk.F * d.amt
        else
            push!(dts, d.time)
        end
    end
    function affect!(integ)
        for d in doses
            d.time > 0 && isapprox(integ.t, d.time; atol = 1e-12) && (integ.u[1] += pk.F * d.amt)
        end
        SciMLBase.u_modified!(integ, true)
    end
    cb = DiscreteCallback((u, t, integ) -> any(isapprox(t, x; atol = 1e-12) for x in dts), affect!;
                          save_positions = (false, false))
    prob = ODEProblem(rhs!, u0, (0.0, maximum(ts)))
    sol = solve(prob, Vern9(); callback = cb, tstops = dts, saveat = ts, reltol, abstol)
    return [u[2] / Vc for u in sol.u]
end

@testset "PK closed forms" begin

    @testset "AUC identities" begin
        T = 1500.0
        n = 400_000
        one_cmt = OneCmt(CL = 2.0, V = 20.0, ka = 1.5, F = 0.7)     # k = 0.1
        two_cmt = TwoCmt(CL = 2.0, V1 = 10.0, Q = 3.0, V2 = 25.0, ka = 1.7, F = 0.6)
        dose = 100.0
        for pk in (one_cmt, two_cmt)
            iv = [Dose(0.0, dose; route = :iv)]
            po = [Dose(0.0, dose; route = :oral)]
            auc_iv = simpson(t -> conc(pk, iv, t), 0.0, T, n)
            auc_po = simpson(t -> conc(pk, po, t), 0.0, T, n)
            @test relerr(auc_iv, dose / pk.CL) < 1e-7             # IV: dose / CL
            @test relerr(auc_po, pk.F * dose / pk.CL) < 1e-7      # oral: F dose / CL
        end
    end

    @testset "two-compartment reduces to one-compartment as Q -> 0" begin
        one = OneCmt(CL = 1.5, V = 12.0, ka = 0.9, F = 0.8)
        two = TwoCmt(CL = 1.5, V1 = 12.0, Q = 1e-10, V2 = 30.0, ka = 0.9, F = 0.8)
        ts = collect(0.1:0.37:60.0)
        for route in (:iv, :oral)
            d = [Dose(0.0, 50.0; route), Dose(12.0, 25.0; route)]
            c1 = conc(one, d, ts); c2 = conc(two, d, ts)
            @test maximum(abs.(c1 .- c2) ./ c1) < 1e-6
        end
    end

    @testset "superposition equals sum of single doses" begin
        for pk in (OneCmt(CL = 0.9, V = 7.0, ka = 1.1, F = 0.85),
                   TwoCmt(CL = 0.9, V1 = 7.0, Q = 0.5, V2 = 11.0, ka = 1.1, F = 0.85))
            doses = [Dose(0.0, 10.0; route = :oral), Dose(5.0, 4.0; route = :iv),
                     Dose(8.5, 12.0; route = :oral), Dose(20.0, 3.0; route = :iv)]
            ts = collect(0.0:0.5:48.0)
            total = conc(pk, doses, ts)
            parts = sum(conc(pk, [d], ts) for d in doses)
            @test maximum(abs.(total .- parts)) < 1e-12 * maximum(total)
            # no contribution from future doses
            @test conc(pk, [Dose(10.0, 5.0)], 9.999) == 0.0
        end
    end

    @testset "steady-state accumulation, one compartment" begin
        pk = OneCmt(CL = 2.0, V = 20.0, ka = 1.5, F = 0.7)         # k = 0.1
        k = pk.CL / pk.V
        for τ in (6.0, 12.0, 24.0)
            nd = 400
            # IV bolus: peak(ss)/peak(1st dose) = 1 / (1 - exp(-k τ))
            iv = dosing_schedule(10.0; interval = τ, n = nd, route = :iv)
            tlast = iv[end].time
            R = conc(pk, iv, tlast) / conc(pk, iv[1:1], 0.0)
            @test relerr(R, 1 / (1 - exp(-k * τ))) < 1e-10
            # Oral: mean concentration over the last interval at steady state = F dose / (CL τ)
            po = dosing_schedule(10.0; interval = τ, n = nd, route = :oral)
            auc_ss = simpson(t -> conc(pk, po, t), po[end].time, po[end].time + τ, 20_000)
            @test relerr(auc_ss, pk.F * 10.0 / pk.CL) < 1e-8
        end
    end

    @testset "ka ≈ k limit is continuous (vs 256-bit reference)" begin
        CL, V = 2.0, 20.0
        k = CL / V
        # exact one-compartment oral unit-dose formula in extended precision
        function ref(ka, τ)
            setprecision(BigFloat, 256) do
                kb, kab, τb = big(CL) / big(V), big(ka), big(τ)
                kab == kb && return kb * τb * exp(-kb * τb) / big(V)
                kab / (big(V) * (kab - kb)) * (exp(-kb * τb) - exp(-kab * τb))
            end
        end
        f(ka, τ) = conc(OneCmt(CL = CL, V = V, ka = ka, F = 1.0), [Dose(0.0, 1.0)], τ)
        for τ in (0.5, 3.0, 10.0, 40.0)
            for δ in (0.0, -1e-4, -1e-6, -2e-8, -5e-9, -1e-10, 1e-10, 5e-9, 2e-8, 1e-6, 1e-4)
                ka = k * (1 + δ)
                @test relerr(f(ka, τ), Float64(ref(ka, τ))) < 1e-9
            end
        end
        # sweep through ka = k on a fine grid: no jumps
        τ = 7.0
        vals = [f(k * (1 + δ), τ) for δ in range(-1e-5, 1e-5; length = 2001)]
        @test maximum(abs.(diff(diff(vals)))) < 1e-13 * maximum(vals)   # second differences
    end

    @testset "derivative wrt ka is correct at ka == k" begin
        # C(ka) = ka/(ka-k) (e^{-kτ} - e^{-ka τ}); at CL=V=ka=F=τ=1, dC/dka = e^{-1}/2
        g = ForwardDiff.derivative(
            ka -> conc(OneCmt(CL = 1.0, V = 1.0, ka = ka, F = 1.0), [Dose(0.0, 1.0)], 1.0), 1.0)
        @test g ≈ exp(-1) / 2 rtol = 1e-12
        # same via CL at fixed ka == k: dC/dCL
        h(CL) = conc(OneCmt(CL = CL, V = 1.0, ka = 1.0, F = 1.0), [Dose(0.0, 1.0)], 1.0)
        gh = ForwardDiff.derivative(h, 1.0)
        fdh = (h(1.0 + 1e-6) - h(1.0 - 1e-6)) / 2e-6
        @test gh ≈ fdh rtol = 1e-6
    end

    @testset "two-compartment oral at ka == α, ka == β: value and derivatives" begin
        CL0, V1, Q, V2, F = 2.0, 10.0, 3.0, 25.0, 0.8
        pk0 = TwoCmt(CL = CL0, V1 = V1, Q = Q, V2 = V2, ka = 1.0, F = F)
        α, β, _ = TumourPKPD.twocmt_rates(pk0)
        d = [Dose(0.0, 100.0)]
        ts = [0.3, 2.0, 9.0, 30.0]
        # Independent 256-bit reference: textbook partial-fraction oral solution,
        # evaluated just off the singularity with tiny central differences.
        function ref(CL, ka, τ)
            setprecision(BigFloat, 256) do
                CLb, V1b, Qb, V2b, kab, τb = big(CL), big(V1), big(Q), big(V2), big(ka), big(τ)
                k10, k12, k21 = CLb / V1b, Qb / V1b, Qb / V2b
                s = k10 + k12 + k21
                r = sqrt(s^2 - 4k10 * k21)
                a, b = (s + r) / 2, (s - r) / 2
                ca = (k21 - a) / ((kab - a) * (b - a))
                cb = (k21 - b) / ((kab - b) * (a - b))
                ck = (k21 - kab) / ((a - kab) * (b - kab))
                big(F) * 100 * kab / V1b * (ca * exp(-a * τb) + cb * exp(-b * τb) + ck * exp(-kab * τb))
            end
        end
        function refgrad(CL, ka)
            h = big"1e-30"
            setprecision(BigFloat, 256) do
                gCL = sum((ref(big(CL) * (1 + h), ka, τ) - ref(big(CL) * (1 - h), ka, τ)) / (2h * big(CL)) * big(CL) for τ in ts) / big(CL)
                gka = sum((ref(CL, big(ka) * (1 + h), τ) - ref(CL, big(ka) * (1 - h), τ)) / (2h * big(ka)) * big(ka) for τ in ts) / big(ka)
                Float64.((gCL, gka))
            end
        end
        for (name, target) in (("alpha", α), ("beta", β))
            pk = TwoCmt(CL = CL0, V1 = V1, Q = Q, V2 = V2, ka = target, F = F)
            c = conc(pk, d, ts)
            @test all(isfinite, c)
            # value vs reference just off the singularity (offset 1e-20 relative)
            cref = [Float64(ref(CL0, big(target) * (1 + big"1e-20"), τ)) for τ in ts]
            @test maximum(abs.(c .- cref) ./ cref) < 1e-9
            g = ForwardDiff.gradient(θ -> sum(conc(TwoCmt(CL = θ[1], V1 = V1, Q = Q, V2 = V2, ka = θ[2], F = F), d, ts)),
                                     [CL0, target])
            gr = refgrad(CL0, target)
            @test g[1] ≈ gr[1] rtol = 1e-8
            @test g[2] ≈ gr[2] rtol = 1e-8
        end
        # AUC identity still holds exactly at ka == α
        pk = TwoCmt(CL = CL0, V1 = V1, Q = Q, V2 = V2, ka = α, F = F)
        auc = simpson(t -> conc(pk, d, t), 0.0, 1500.0, 400_000)
        @test relerr(auc, F * 100.0 / CL0) < 1e-7
    end

    @testset "negative dose times are rejected by the ODE cross-check models" begin
        bad = [Dose(-1.0, 10.0), Dose(4.0, 10.0)]
        @test_throws ArgumentError PKMTK.mtk_onecmt(bad)
        @test_throws ArgumentError PKMTK.mtk_twocmt(bad)
        @test_throws ArgumentError ode_conc(OneCmt(CL = 1.0, V = 5.0, ka = 1.0, F = 1.0), bad, [1.0, 5.0])
    end

    @testset "two-compartment with tiny CL keeps the slow rate" begin
        # CL -> 0: beta = k10 k21 / alpha must stay positive and accurate
        for CL in (1e-3, 1e-6, 1e-9)
            pk = TwoCmt(CL = CL, V1 = 10.0, Q = 3.0, V2 = 25.0, ka = 1.0, F = 1.0)
            α, β, k21 = TumourPKPD.twocmt_rates(pk)
            k10 = CL / 10.0; k12 = 0.3
            @test β > 0
            @test relerr(α * β, k10 * k21) < 1e-12           # Vieta product
            @test relerr(α + β, k10 + k12 + k21) < 1e-12     # Vieta sum
            # IV AUC = dose/CL holds even for a very slow terminal phase
            T = 60 / β
            auc = simpson(t -> conc(pk, [Dose(0.0, 1.0; route = :iv)], t), 0.0, T, 2_000_000)
            CL >= 1e-6 && @test relerr(auc, 1 / CL) < 1e-4
            c = conc(pk, [Dose(0.0, 1.0)], [1.0, 10.0, 100.0])
            @test all(isfinite, c) && all(>(0), c)
        end
    end

    @testset "ForwardDiff gradient vs finite differences" begin
        doses = [Dose(0.0, 100.0), Dose(8.0, 100.0), Dose(16.0, 50.0), Dose(30.0, 100.0)]
        ts = [0.5, 3.0, 9.0, 17.0, 25.0, 40.0]
        fd(f, x; h = 1e-6) = [(f(replace_at(x, i, x[i] * (1 + h))) - f(replace_at(x, i, x[i] * (1 - h)))) /
                              (2h * x[i]) for i in eachindex(x)]
        replace_at(x, i, v) = (y = copy(x); y[i] = v; y)

        f1(θ) = sum(conc(OneCmt(CL = θ[1], V = θ[2], ka = θ[3], F = θ[4]), doses, ts) .* (1:6))
        θ1 = [3.0, 25.0, 1.4, 0.8]
        @test maximum(abs.(ForwardDiff.gradient(f1, θ1) .- fd(f1, θ1)) ./ abs.(fd(f1, θ1))) < 1e-6

        f2(θ) = sum(conc(TwoCmt(CL = θ[1], V1 = θ[2], Q = θ[3], V2 = θ[4], ka = θ[5], F = 0.8), doses, ts) .* (1:6))
        θ2 = [3.0, 25.0, 4.0, 40.0, 1.4]
        g2 = ForwardDiff.gradient(f2, θ2)
        @test maximum(abs.(g2 .- fd(f2, θ2)) ./ abs.(fd(f2, θ2))) < 1e-5

        # IV path too (ka, F unused -> zero gradient)
        f3(θ) = sum(conc(OneCmt(CL = θ[1], V = θ[2], ka = θ[3], F = 1.0), [Dose(0.0, 100.0; route = :iv)], ts))
        g3 = ForwardDiff.gradient(f3, [3.0, 25.0, 1.4])
        @test g3[3] == 0.0
        @test maximum(abs.(g3[1:2] .- fd(f3, [3.0, 25.0, 1.4])[1:2]) ./ abs.(g3[1:2])) < 1e-6
    end

    @testset "closed form vs ODE solves (multiple oral doses)" begin
        doses = [Dose(0.0, 100.0), Dose(6.0, 50.0), Dose(12.0, 100.0),
                 Dose(18.5, 75.0), Dose(30.0, 100.0)]
        ts = collect(0.25:0.25:60.0)
        pk1 = OneCmt(CL = 3.0, V = 25.0, ka = 1.4, F = 0.8)
        pk2 = TwoCmt(CL = 3.0, V1 = 25.0, Q = 4.0, V2 = 40.0, ka = 1.4, F = 0.8)
        for (pk, mtk) in ((pk1, PKMTK.mtk_onecmt), (pk2, PKMTK.mtk_twocmt))
            closed = conc(pk, doses, ts)
            hand = ode_conc(pk, doses, ts)
            @test maximum(abs.(hand .- closed) ./ closed) < 1e-6
            built = mtk(doses; F = pk.F)
            viamtk = PKMTK.mtk_conc(built, pk, ts)
            @test maximum(abs.(viamtk .- closed) ./ closed) < 1e-6
        end
    end
end
