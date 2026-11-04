# Before/after timing of simulate_tumour with the real PDXE regimens (many doses).
#   JULIA_LOAD_PATH="@:@stdlib:<env with BenchmarkTools>" julia --project=. -t 2 scripts/bench_pd_regimen.jl
using TumourPKPD, ForwardDiff, BenchmarkTools, Printf
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.5

function run(drug, tobs)
    doses, pkp = regimen(drug)
    pk = OneCmt(CL = pkp.CL, V = pkp.V, ka = pkp.ka, F = pkp.F)
    cref = maximum(conc(pk, doses[1:min(end, 60)], range(0, 3; length = 3001)))
    ndose = count(d -> d.time <= last(tobs), doses)
    mk() = PKExposure(pk, doses, cref)
    θ(p) = Simeoni(exp(p[1]), exp(p[2]), exp(p[3]), exp(p[4]), 150.0)
    p0 = log.([0.1, 60.0, 0.5, 1.0])
    ll(p) = sum(log.(max.(first(simulate_tumour(θ(p), mk(), tobs)), 1e-3)))
    b0 = @benchmark PKExposure($pk, $doses, $cref)
    ex = mk()
    b1 = @benchmark ($ex)(17.3)
    b2 = @benchmark first(simulate_tumour(Simeoni(0.1, 60.0, 0.5, 1.0, 150.0), $ex, $tobs))
    cfg = ForwardDiff.GradientConfig(ll, p0)
    b3 = @benchmark ForwardDiff.gradient($ll, $p0, $cfg)
    # PK parameters as duals too
    llpk(q) = sum(log.(max.(first(simulate_tumour(θ(q[1:4]), PKExposure(OneCmt(CL = exp(q[5]), V = exp(q[6]), ka = exp(q[7]), F = pk.F), doses, cref), tobs)), 1e-3)))
    q0 = [p0; log.([pk.CL, pk.V, pk.ka])]
    b4 = @benchmark ForwardDiff.gradient($llpk, $q0)
    @printf("%-12s ndoses(total)=%4d  in-window=%4d  tend=%5.1f\n", drug, length(doses), ndose, last(tobs))
    @printf("  construct PKExposure  %9.2f us\n  e(t=17.3) call        %9.3f us\n  solve (Float64)       %9.1f us\n  grad(4 PD)            %9.1f us\n  grad(4 PD + 3 PK)     %9.1f us\n",
            median(b0).time / 1e3, median(b1).time / 1e3, median(b2).time / 1e3, median(b3).time / 1e3, median(b4).time / 1e3)
    println("  value check: ", first(simulate_tumour(Simeoni(0.1, 60.0, 0.5, 1.0, 150.0), ex, tobs))[end])
end

for drug in ("binimetinib", "BKM120"), tobs in ([0.0, 3, 7, 10, 14, 17, 21, 24, 27, 30], collect(0.0:6.0:120.0))
    run(drug, tobs)
end
