# Benchmark of the Simeoni solve and its ForwardDiff gradient for one tumour.
#
#   JULIA_LOAD_PATH="@:@stdlib:<env with BenchmarkTools, StaticArrays>" \
#     julia --project=. -t 2 scripts/bench_pd.jl > ../logs/2026-11-04-pd-bench.log
#
# BenchmarkTools and StaticArrays are not (yet) direct dependencies of the package, hence the
# extra load-path entry. Everything compared here is implemented locally in this script so the
# comparison does not depend on what src/pd.jl currently does.
using TumourPKPD, ForwardDiff, OrdinaryDiffEq, SciMLBase
using BenchmarkTools, StaticArrays, Printf, Statistics, LinearAlgebra
using TumourPKPD: growth_factor

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.5
BenchmarkTools.DEFAULT_PARAMETERS.samples = 2000

# ---------------------------------------------------------------- problem
const TOBS = [0.0, 3, 7, 10, 14, 17, 21, 24, 27, 30]          # days, 10 obs
const DOSES = dosing_schedule(10.0; n = 21)                    # 21 daily oral doses
const PK0 = (CL = 1.0, V = 2.0, ka = 3.0, F = 0.8)
const W0 = 150.0
const σobs = 0.2
# reference scale: Cmax of a single dose
const CREF = maximum(conc(OneCmt(; PK0...), [Dose(0.0, 10.0)], range(0, 3; length = 3001)))
const TRUE_PD = (λ0 = 0.12, λ1 = 60.0, k1 = 0.5, k2 = 1.5)

# log-parameter vector: [logλ0, logλ1, logk1, logk2, logCL, logV, logka]
const P0 = log.([TRUE_PD.λ0, TRUE_PD.λ1, TRUE_PD.k1, TRUE_PD.k2, PK0.CL, PK0.V, PK0.ka])

struct NoDrugC; cref::Float64; end
NoDrugC() = NoDrugC(1.0)
(::NoDrugC)(t) = 0.0

# ---------------------------------------------------------------- local solver variants
# (a) in-place Vector RHS, via the package's simulate_tumour
# (b) out-of-place SVector{4} RHS
@inline function simeoni_oop(u, p, t)
    θ, ex = p
    x1, x2, x3, x4 = u
    w = x1 + x2 + x3 + x4
    kill = θ.k2 * ex(t) * x1
    return SVector(θ.λ0 * x1 * growth_factor(θ.λ0 * w / θ.λ1, θ.ψ) - kill,
                   kill - θ.k1 * x2,
                   θ.k1 * (x2 - x3),
                   θ.k1 * (x3 - x4))
end

function simeoni_ip!(du, u, p, t)
    θ, ex = p
    x1, x2, x3, x4 = u
    w = x1 + x2 + x3 + x4
    kill = θ.k2 * ex(t) * x1
    du[1] = θ.λ0 * x1 * growth_factor(θ.λ0 * w / θ.λ1, θ.ψ) - kill
    du[2] = kill - θ.k1 * x2
    du[3] = θ.k1 * (x2 - x3)
    du[4] = θ.k1 * (x3 - x4)
    return nothing
end

function sim_ip(θ, ex, stopsrc, tobs; solver = Tsit5(), reltol = 1e-6, abstol = 1e-6, tstops = true)
    T = promote_type(typeof(θ.λ0), typeof(etype(ex)))
    u0 = T[θ.w0, 0, 0, 0]
    tend = last(tobs)
    stops = tstops ? filter(s -> 0 < s < tend, stopsrc) : Float64[]
    prob = ODEProblem{true}(simeoni_ip!, u0, (0.0, tend), (θ, ex))
    sol = solve(prob, solver; saveat = tobs, tstops = stops, reltol, abstol,
                save_everystep = false, dense = false, verbose = false)
    SciMLBase.successful_retcode(sol) || return fill(T(NaN), length(tobs))
    return [sum(u) for u in sol.u]
end

function sim_sv(θ, ex, stopsrc, tobs; solver = Tsit5(), reltol = 1e-6, abstol = 1e-6, tstops = true)
    T = promote_type(typeof(θ.λ0), typeof(etype(ex)))
    u0 = SVector{4,T}(θ.w0, 0, 0, 0)
    tend = last(tobs)
    stops = tstops ? filter(s -> 0 < s < tend, stopsrc) : Float64[]
    prob = ODEProblem{false}(simeoni_oop, u0, (0.0, tend), (θ, ex))
    sol = solve(prob, solver; saveat = tobs, tstops = stops, reltol, abstol,
                save_everystep = false, dense = false, verbose = false)
    SciMLBase.successful_retcode(sol) || return fill(T(NaN), length(tobs))
    return [sum(u) for u in sol.u]
end

etype(e::PKExposure) = e.pk.CL
etype(e) = e.cref

# (c) exposure with precomputed dose arrays: c(t) = s Σ_{td≤t} a_i (e^{-k t} e^{k td_i} - e^{-ka t} e^{ka td_i})
# -> 2 exp calls per RHS evaluation instead of 2 per dose. Oral 1-cmt only (ka != k).
struct FastExp{T,A<:AbstractVector{T}}
    td::Vector{Float64}
    k::T
    ka::T
    s::T          # F ka / (V (ka-k)) / cref
    ek::A         # amt_i * exp(k td_i)
    eka::A        # amt_i * exp(ka td_i)
    cref::T
end
function FastExp(pk::OneCmt, doses, cref)
    k = pk.CL / pk.V
    s = pk.F * pk.ka / (pk.V * (pk.ka - k)) / cref
    td = [d.time for d in doses]
    ek = [d.amt * exp(k * d.time) for d in doses]
    eka = [d.amt * exp(pk.ka * d.time) for d in doses]
    T = typeof(s)
    return FastExp{T,Vector{T}}(td, k, pk.ka, s, ek, eka, T(cref))
end
function (e::FastExp)(t)
    Ek = exp(-e.k * t)
    Eka = exp(-e.ka * t)
    a = zero(Ek); b = zero(Ek)
    @inbounds for i in eachindex(e.td)
        e.td[i] > t && break
        a += e.ek[i]
        b += e.eka[i]
    end
    return e.s * (Ek * a - Eka * b)
end

# ---------------------------------------------------------------- likelihood / gradient
const YOBS = let
    θ = Simeoni(TRUE_PD..., W0)
    ex = PKExposure(OneCmt(; PK0...), DOSES, CREF)
    w, _ = simulate_tumour(θ, ex, TOBS; solver = Vern9(), reltol = 1e-12, abstol = 1e-12)
    # fixed pseudo-noise (deterministic, no RNG dependence)
    exp.(log.(w) .+ σobs .* [0.5, -1.0, 0.3, 1.1, -0.4, 0.8, -0.7, 0.2, -0.2, 0.9])
end

loglik(w) = -sum(abs2, log.(max.(w, 1e-3)) .- log.(YOBS)) / (2σobs^2)

function make_ll(kind::Symbol; npk = 0, kw...)
    function ll(p)
        θ = Simeoni(exp(p[1]), exp(p[2]), exp(p[3]), exp(p[4]), W0)
        pk = npk == 0 ? OneCmt(; PK0...) :
             OneCmt(CL = exp(p[5]), V = exp(p[6]), ka = exp(p[7]), F = PK0.F)
        w = if kind === :pkg
            first(simulate_tumour(θ, PKExposure(pk, DOSES, CREF), TOBS; kw...))
        elseif kind === :ip
            sim_ip(θ, PKExposure(pk, DOSES, CREF), [d.time for d in DOSES], TOBS; kw...)
        elseif kind === :sv
            sim_sv(θ, PKExposure(pk, DOSES, CREF), [d.time for d in DOSES], TOBS; kw...)
        elseif kind === :ip_fast
            sim_ip(θ, FastExp(pk, DOSES, CREF), [d.time for d in DOSES], TOBS; kw...)
        elseif kind === :sv_fast
            sim_sv(θ, FastExp(pk, DOSES, CREF), [d.time for d in DOSES], TOBS; kw...)
        end
        return loglik(w)
    end
    return ll
end

# ---------------------------------------------------------------- timing helpers
function timeit(f)
    b = @benchmark $f()
    return (t = median(b).time / 1e3, tmin = minimum(b).time / 1e3, allocs = b.allocs, mem = b.memory)   # μs
end
fmt(r) = @sprintf("%8.1f us (min %7.1f)  %6d allocs  %8.1f KiB", r.t, r.tmin, r.allocs, r.mem / 1024)

function gradtime(ll, p)
    cfg = ForwardDiff.GradientConfig(ll, p)
    return timeit(() -> ForwardDiff.gradient(ll, p, cfg))
end

# ---------------------------------------------------------------- reference
println("threads = ", Threads.nthreads(), "  julia ", VERSION, "  CREF = ", round(CREF, digits = 4))
const ll_ref7 = make_ll(:pkg; npk = 1, solver = Vern9(), reltol = 1e-13, abstol = 1e-13)
const LL_REF = ll_ref7(P0)
const G_REF = ForwardDiff.gradient(ll_ref7, P0)
println("reference loglik = ", LL_REF, "\nreference grad   = ", round.(G_REF, sigdigits = 6))

function errs(ll, p)
    g = ForwardDiff.gradient(ll, p)
    return abs(ll(p) - LL_REF), norm(g - G_REF) / norm(G_REF)
end

# ---------------------------------------------------------------- 1. solvers
println("\n=== 1. solver comparison (reltol = abstol = 1e-6, tstops at doses) ===")
println("plain solve (Float64), gradient wrt 4 PD params, gradient wrt 4 PD + 3 PK params")
solvers = [("Tsit5", Tsit5()), ("Vern7", Vern7()), ("Rodas5P", Rodas5P()),
           ("AutoTsit5(Rosenbrock23)", AutoTsit5(Rosenbrock23()))]
for (nm, s) in solvers
    println("--- ", nm)
    try
        f4 = make_ll(:pkg; solver = s)
        f7 = make_ll(:pkg; npk = 1, solver = s)
        println("  solve    ", fmt(timeit(() -> f4(P0[1:4]))))
        println("  grad(4)  ", fmt(gradtime(f4, P0[1:4])))
        println("  grad(7)  ", fmt(gradtime(f7, P0)))
        e, ge = errs(f7, P0)
        @printf("  |Δll| = %.2e   rel grad err (norm) = %.2e\n", e, ge)
    catch err
        println("  FAILED: ", sprint(showerror, err)[1:min(end, 200)])
    end
end

# ---------------------------------------------------------------- 2. tolerances
println("\n=== 2. tolerance effect (Tsit5, tstops at doses), error vs Vern9 1e-13 reference ===")
for (rt, at) in [(1e-3, 1e-3), (1e-4, 1e-4), (1e-5, 1e-5), (1e-6, 1e-6), (1e-8, 1e-8), (1e-10, 1e-10), (1e-4, 1e-8), (1e-6, 1e-3)]
    f7 = make_ll(:pkg; npk = 1, solver = Tsit5(), reltol = rt, abstol = at)
    e, ge = errs(f7, P0)
    g = gradtime(f7, P0)
    @printf("  reltol=%.0e abstol=%.0e  |Δll|=%.2e  rel grad err=%.2e   grad(7) %s\n", rt, at, e, ge, fmt(g))
end

# ---------------------------------------------------------------- 3. in-place vs static
println("\n=== 3. in-place Vector vs out-of-place SVector{4} (reltol = abstol = 1e-6) ===")
for (nm, s) in [("Tsit5", Tsit5()), ("Vern7", Vern7())]
    for kind in (:ip, :sv)
        f4 = make_ll(kind; solver = s)
        f7 = make_ll(kind; npk = 1, solver = s)
        println("--- ", nm, "  ", kind)
        println("  solve    ", fmt(timeit(() -> f4(P0[1:4]))))
        println("  grad(4)  ", fmt(gradtime(f4, P0[1:4])))
        println("  grad(7)  ", fmt(gradtime(f7, P0)))
    end
end
# the same on a stiffer solver (Rosenbrock needs a Jacobian; may fail with nested duals)
for kind in (:ip, :sv)
    try
        f4 = make_ll(kind; solver = Rodas5P())
        println("--- Rodas5P ", kind, "\n  solve    ", fmt(timeit(() -> f4(P0[1:4]))),
                "\n  grad(4)  ", fmt(gradtime(f4, P0[1:4])))
    catch err
        println("--- Rodas5P ", kind, " FAILED: ", sprint(showerror, err)[1:min(end, 160)])
    end
end

# ---------------------------------------------------------------- 4. tstops
println("\n=== 4. tstops at dose times (Tsit5, in-place) ===")
for rt in (1e-6, 1e-4)
    for ts in (true, false)
        f7 = make_ll(:ip; npk = 1, solver = Tsit5(), reltol = rt, abstol = rt, tstops = ts)
        e, ge = errs(f7, P0)
        @printf("  reltol=abstol=%.0e tstops=%-5s |Δll|=%.2e rel grad err=%.2e  solve %s\n", rt, ts, e, ge,
                fmt(timeit(() -> f7(P0))))
    end
end
# step counts
let θ = Simeoni(TRUE_PD..., W0), ex = PKExposure(OneCmt(; PK0...), DOSES, CREF)
    for ts in (true, false)
        stops = ts ? [d.time for d in DOSES if 0 < d.time < 30] : Float64[]
        sol = solve(ODEProblem{true}(simeoni_ip!, [W0, 0, 0, 0.0], (0.0, 30.0), (θ, ex)), Tsit5();
                    saveat = TOBS, tstops = stops, reltol = 1e-6, abstol = 1e-6, save_everystep = false, dense = false)
        println("  tstops=", ts, ": nf evals = ", sol.stats.nf, ", accepted = ", sol.stats.naccept, ", rejected = ", sol.stats.nreject)
    end
end

# ---------------------------------------------------------------- 5. precomputed dose arrays
println("\n=== 5. precomputed dose arrays (FastExp: 2 exp per RHS call) ===")
let pk = OneCmt(; PK0...), ex = PKExposure(pk, DOSES, CREF), fe = FastExp(pk, DOSES, CREF)
    ts = range(0, 30; length = 301)
    @printf("  max rel diff FastExp vs conc: %.2e\n", maximum(abs(fe(t) - ex(t)) / max(abs(ex(t)), 1e-3) for t in ts))
    println("  e(t) call, PKExposure: ", fmt(timeit(() -> ex(17.3))))
    println("  e(t) call, FastExp   : ", fmt(timeit(() -> fe(17.3))))
end
for kind in (:ip, :ip_fast, :sv, :sv_fast)
    f4 = make_ll(kind; solver = Tsit5())
    f7 = make_ll(kind; npk = 1, solver = Tsit5())
    e, ge = errs(f7, P0)
    println("--- Tsit5 ", kind, "   |Δll|=", @sprintf("%.1e", e), " rel grad err=", @sprintf("%.1e", ge))
    println("  solve    ", fmt(timeit(() -> f4(P0[1:4]))))
    println("  grad(4)  ", fmt(gradtime(f4, P0[1:4])))
    println("  grad(7)  ", fmt(gradtime(f7, P0)))
end

# ---------------------------------------------------------------- 6. scaling with number of tumours
println("\n=== 6. global ForwardDiff over N tumours vs per-tumour local gradients ===")
# Hierarchical-like parameter vector: shared [logλ1, logk1] + per-tumour [logλ0_i, logk2_i]. Each tumour has a
# control and a treated curve (2 solves). Global gradient = one ForwardDiff call over 2N+2 inputs.
function tumour_ll(λ0, λ1, k1, k2, kind, fast)
    pk = OneCmt(; PK0...)
    sd = [d.time for d in DOSES]
    ex = fast ? FastExp(pk, DOSES, CREF) : PKExposure(pk, DOSES, CREF)
    simf = kind === :sv ? sim_sv : sim_ip
    wc = simf(Simeoni(λ0, λ1, k1, 0.0, W0), NoDrugC(), sd, TOBS)
    wt = simf(Simeoni(λ0, λ1, k1, k2, W0), ex, sd, TOBS)
    return loglik(wc) + loglik(wt)
end
function make_global(N; kind = :ip, fast = false)
    function ll(p)
        s = zero(eltype(p))
        for i in 1:N
            s += tumour_ll(exp(p[2+i]), exp(p[1]), exp(p[2]), exp(p[2+N+i]), kind, fast)
        end
        return s
    end
    return ll
end

for (kind, fast) in ((:ip, false), (:ip, true), (:sv, true))
    println("--- kind = ", kind, ", fast exposure = ", fast, "  (Tsit5, 1e-6)")
    for N in (5, 10, 20, 40)
        ll = make_global(N; kind, fast)
        p = vcat([log(60.0), log(0.5)], fill(log(0.12), N), fill(log(1.5), N))
        ll(p)
        cfg = ForwardDiff.GradientConfig(ll, p)
        ForwardDiff.gradient(ll, p, cfg)
        t = minimum(@elapsed(ForwardDiff.gradient(ll, p, cfg)) for _ in 1:5)
        @printf("  N=%3d  P=%3d  chunk=%d  global gradient %8.2f ms   per tumour %7.2f ms\n", N, length(p),
                typeof(cfg).parameters[3], 1e3 * t, 1e3 * t / N)
    end
    # local per-tumour gradient: 4 inputs [logλ1, logk1, logλ0_i, logk2_i]
    ll1 = make_global(1; kind, fast)
    p = [log(60.0), log(0.5), log(0.12), log(1.5)]
    cfg = ForwardDiff.GradientConfig(ll1, p)
    ForwardDiff.gradient(ll1, p, cfg)
    println("  local per-tumour gradient (4 inputs, 2 solves): ", fmt(timeit(() -> ForwardDiff.gradient(ll1, p, cfg))))
end

# ---------------------------------------------------------------- 7. N tumours, local gradients, threads
println("\n=== 7. N tumours, per-tumour local gradients assembled by hand (fast exposure, SVector or Vector) ===")
function local_grads!(G, lam1, k1, lam0, k2, kind, fast; threaded = false)
    N = length(lam0)
    body(i) = begin
        f(q) = tumour_ll(exp(q[3]), exp(q[1]), exp(q[2]), exp(q[4]), kind, fast)
        G[:, i] .= ForwardDiff.gradient(f, [lam1, k1, lam0[i], k2[i]])
    end
    if threaded
        Threads.@threads for i in 1:N
            body(i)
        end
    else
        for i in 1:N
            body(i)
        end
    end
    return G
end
for N in (100, 300)
    G = zeros(4, N)
    l0 = fill(log(0.12), N); kk = fill(log(1.5), N)
    for (kind, fast) in ((:ip, true), (:sv, true))
        local_grads!(G, log(60.0), log(0.5), l0, kk, kind, fast)
        t1 = minimum(@elapsed(local_grads!(G, log(60.0), log(0.5), l0, kk, kind, fast)) for _ in 1:3)
        t2 = minimum(@elapsed(local_grads!(G, log(60.0), log(0.5), l0, kk, kind, fast; threaded = true)) for _ in 1:3)
        @printf("  N=%3d kind=%s fast: serial %8.2f ms (%.1f us/tumour), %d threads %8.2f ms\n", N, kind, 1e3 * t1,
                1e6 * t1 / N, Threads.nthreads(), 1e3 * t2)
    end
end
println("\ndone")
