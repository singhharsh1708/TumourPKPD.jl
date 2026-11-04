# Cross-check simulate_tumour against ModelingToolkit v11 versions (src/pd_mtk.jl).
# Run: julia --project=. -t 2 test/test_pd_mtk.jl
using Test
using TumourPKPD
using OrdinaryDiffEq

const PDMTK = if isdefined(TumourPKPD, :mtk_simeoni_pk)
    TumourPKPD
else
    let m = Module(:PDMTK)
        Core.eval(m, :(using TumourPKPD))
        Base.include(m, joinpath(@__DIR__, "..", "src", "pd_mtk.jl"))
        m
    end
end

@testset "Simeoni MTK cross-check" begin
    t = [0.0, 1.0, 2.5, 5.0, 8.0, 12.0, 17.0, 21.0, 25.0, 30.0]
    pk = OneCmt(CL = 1.0, V = 2.0, ka = 3.0, F = 0.8)
    doses = dosing_schedule(10.0; n = 21)       # includes a dose at t = 0
    cref = 5.0
    for θ in (Simeoni(0.15, 80.0, 0.5, 0.8, 150.0),      # exponential-linear transition with drug
              Simeoni(0.3, 40.0, 1.5, 0.3, 30.0),
              Simeoni(0.05, 20.0, 0.2, 2.0, 200.0))
        ex = PKExposure(pk, doses, cref)
        w, ok = simulate_tumour(θ, ex, t; solver = Vern9(), reltol = 1e-11, abstol = 1e-11)
        @test ok
        # (a) PK depot/central + Simeoni in one MTK system
        built = PDMTK.mtk_simeoni_pk(doses; F = pk.F)
        wm = PDMTK.mtk_simulate_pk(built, θ, pk, cref, t)
        @test wm ≈ w rtol = 1e-7
        # (b) registered closed-form exposure function
        reg = PDMTK.mtk_simeoni_registered()
        wr = PDMTK.mtk_simulate_registered(reg, θ, ex, t)
        @test wr ≈ w rtol = 1e-7
        # default settings of simulate_tumour agree to the 1e-5 level
        wd, _ = simulate_tumour(θ, ex, t)
        @test wd ≈ wm rtol = 1e-5
    end
    # untreated arm (dose-free): registered variant with NoDrug
    θ = Simeoni(0.15, 80.0, 0.5, 0.8, 150.0)
    reg = PDMTK.mtk_simeoni_registered()
    wr = PDMTK.mtk_simulate_registered(reg, θ, NoDrug(), t)
    wn, _ = simulate_tumour(θ, NoDrug(), t; solver = Vern9(), reltol = 1e-11, abstol = 1e-11)
    @test wr ≈ wn rtol = 1e-7
end
