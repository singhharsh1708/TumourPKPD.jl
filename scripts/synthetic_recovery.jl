# Synthetic parameter recovery for the hierarchical Simeoni model.
#
#   julia --project=TumourPKPD.jl -t 4 TumourPKPD.jl/scripts/synthetic_recovery.jl N [nchains nsamples nadapt latent_w0 seed tag]
#
# Simulates N PDX units (control + treated) from known population parameters, fits them, and reports
# true vs posterior median vs 90% CrI, per-unit coverage of λ0 and k2, diagnostics and timings.
# Output goes to stdout (redirect to a log file of your choice).

using TumourPKPD, Random, Statistics, Printf
using TumourPKPD: Dose, OneCmt, dosing_schedule, conc
isdefined(TumourPKPD, :fit_drug) ||
    Base.include(TumourPKPD, joinpath(@__DIR__, "..", "src", "hierarchical.jl"))
const H = TumourPKPD

# flush stdout once a second so a redirected log shows progress (and survives a kill)
const _flusher = Timer(_ -> flush(stdout), 1.0; interval = 1.0)

include(joinpath(@__DIR__, "procchains.jl"))
const FLAGS = filter(a -> startswith(a, "--"), ARGS)
const POS = filter(a -> !startswith(a, "--"), ARGS)
const CHILD = let f = filter(a -> startswith(a, "--chain-only="), FLAGS); isempty(f) ? 0 : parse(Int, split(f[1], '=')[2]) end
const CHILD_OUT = let f = filter(a -> startswith(a, "--chain-out="), FLAGS); isempty(f) ? "" : split(f[1], '=')[2] end
const NOPROCS = "--noprocs" in FLAGS
ARGS_ = POS

N          = length(POS) >= 1 ? parse(Int, POS[1]) : 20
nchains    = length(POS) >= 2 ? parse(Int, POS[2]) : 2
nsamples   = length(POS) >= 3 ? parse(Int, POS[3]) : 300
nadapt     = length(POS) >= 4 ? parse(Int, POS[4]) : 300
latent_w0  = length(POS) >= 5 ? parse(Bool, POS[5]) : true
seed       = length(POS) >= 6 ? parse(Int, POS[6]) : 20261104
tag        = length(POS) >= 7 ? POS[7] : "N$(N)"
PRIORS     = H.default_priors()
CTHREADS   = let f = filter(a -> startswith(a, "--cthreads="), FLAGS); isempty(f) ? 1 : parse(Int, split(f[1], '=')[2]) end
INIT_FILE  = let f = filter(a -> startswith(a, "--init-file="), FLAGS); isempty(f) ? "" : split(f[1], '=')[2] end

CHILD > 0 || println("# synthetic recovery  Vtr prior N(log 1000, 0.7), Student-t nu=5, baseline sd 0.1  N=$N chains=$nchains samples=$nsamples adapt=$nadapt latent_w0=$latent_w0 seed=$seed threads=$(Threads.nthreads())")

# --- truth ---------------------------------------------------------------------------------
# mouse-like PK in per-day units: k = CL/V = 2/day (t1/2 ≈ 8 h), fast absorption; daily oral dosing 28 days.
pk = OneCmt(CL = 3.0, V = 1.5, ka = 24.0, F = 0.8)
doses = dosing_schedule(25.0; start = 0.0, interval = 1.0, n = 28)
cref = H.reference_cref(pk, doses)
truth = (μ_λ0 = log(0.06), ω_λ0 = 0.4, μ_k2 = log(0.4), ω_k2 = 0.8,
         logλ1 = log(120.0), logk1 = log(0.7), σ = 0.15)
rng = MersenneTwister(seed)
units, tru = H.simulate_units(rng, N, doses, pk, cref; truth)

nobs = sum(length(u.control.t) + length(u.treated.t) - 2 for u in units)
nzero = sum(count(==(0.0), u.treated.v) + count(==(0.0), u.control.v) for u in units)
nzero_units = count(u -> any(==(0.0), u.treated.v), units)
CHILD > 0 || println("units=$N  post-baseline obs=$nobs  zero (censored) obs=$nzero in $nzero_units units")
CHILD > 0 || println("final control volume (median, max): ",
        round(median(last(u.control.v) for u in units); digits = 0), " ", round(maximum(last(u.control.v) for u in units); digits = 0))

if CHILD > 0                                      # child process: sample one chain, save, exit
    init, k2c = isempty(INIT_FILE) ? (nothing, nothing) : load_init(INIT_FILE; chain = CHILD)
    chain, _ = H.fit_drug(units, doses, pk, cref; nchains = 1, nsamples, nadapt, chain_only = true, init, k2_centered = k2c,
                          map_starts = 0, window_days = 0,
                          rng = MersenneTwister(1000 * seed + CHILD), latent_w0, priors = PRIORS, log_every = 25,
                          logger = m -> (println("[chain $CHILD] ", m); flush(stdout)))
    save_chain_parts(CHILD_OUT, chain)
    println("[chain $CHILD] saved"); flush(stdout)
    exit(0)
end

# --- gradient timing ---------------------------------------------------------------------------
using Turing
const DynamicPPL = Turing.DynamicPPL
const LogDensityProblems = Turing.LogDensityProblems
function grad_timing(local_grad, threaded; reps = 20)
    d = H.prepare_data(units, doses, pk, cref; local_grad, threaded, latent_w0)
    model = H.hier_model(d, PRIORS)
    D = 7 + 2N + (latent_w0 ? 2N : 0)
    adt = AutoForwardDiff(chunksize = min(D, local_grad ? 32 : 12))   # bounded chunk policy (as fit_drug)
    vi = DynamicPPL.VarInfo(MersenneTwister(1), model)
    nt0 = merge(H.initial_values(d, H.default_priors(), MersenneTwister(2)), (; μ_k2 = log(0.4), σ = 0.2))
    vi = DynamicPPL.link(DynamicPPL.initialize_parameters!!(vi, nt0, model), model)
    ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi; adtype = adt)
    x = vi[:]
    LogDensityProblems.logdensity_and_gradient(ldf, x)
    t0 = time()
    rr = MersenneTwister(5)
    for _ in 1:reps      # perturb x: the local-gradient cache would otherwise return the previous result
        LogDensityProblems.logdensity_and_gradient(ldf, x .+ 1e-3 .* randn(rr, length(x)))
    end
    return (time() - t0) / reps
end
println("\n## gradient evaluation time (ms), full logdensity+gradient at a typical point")
for (lg, thr, name) in ((true, true, "local-grad, threaded"), (true, false, "local-grad, serial"),
                        (false, true, "global chunk-12 ForwardDiff, threaded"))
    (N > 60 && !lg) && continue
    @printf("%-40s %8.2f\n", name, 1000 * grad_timing(lg, thr))
end

# --- fit -------------------------------------------------------------------------------------
println("\n## fit")
fitkw = (; drug = "synthetic_$(tag)", nsamples, nadapt, latent_w0, progress = false, priors = PRIORS, seed, window_days = 0)
if nchains > 1 && Threads.nthreads() > 1 && !NOPROCS
    println("multi-start MAP (5 starts) in the parent, then $nchains chains in separate processes"); flush(stdout)
    mm0 = H.map_modes(units, doses, pk, cref; window_days = 0, latent_w0, priors = PRIORS, nstarts = 5,
                      maxiters = 1000, rng = MersenneTwister(seed + 2))
    map_lps = mm0.all_lps; k2c = mm0.k2_centered; nt0 = [m.nt for m in mm0.starts]
    println("MAP log-densities of the starts: ", round.(map_lps; digits = 1)); flush(stdout)
    init_file = joinpath(mktempdir(), "init.jld2"); save_init(init_file, nt0, k2c)
    maxprocs = max(1, Threads.nthreads() ÷ CTHREADS)
    println("sampling: $CTHREADS thread(s) per chain process, max $maxprocs at a time"); flush(stdout)
    chain_, wall = run_chain_procs(@__FILE__, vcat(POS, ["--cthreads=$CTHREADS", "--init-file=$init_file"]), nchains;
                                   maxprocs, threads = CTHREADS, project = Base.active_project())
    chain, pd = H.fit_drug(units, doses, pk, cref; fitkw..., precomputed_chain = chain_, precomputed_runtime = wall, map_lps, k2_centered = k2c)
else
    chain, pd = H.fit_drug(units, doses, pk, cref; fitkw..., nchains, rng = MersenneTwister(seed + 1), log_every = 25)
end
diag = pd.diagnostics
niter = nchains * (nsamples + nadapt)
@printf("runtime %.1f s  (%d iterations total -> %.1f s per 100 iterations)\n",
        diag["runtime_s"], niter, 100 * diag["runtime_s"] / niter)
println("parallel mode: ", diag["settings"]["parallel"], "  chunksize ", diag["settings"]["chunksize"])
println("MAP log-densities (mode multiplicity): ", round.(diag["map_lps"]; digits = 1))
for k in ("max_rhat", "min_ess_bulk", "n_divergences", "mean_acceptance", "mean_tree_depth", "mean_n_steps",
          "max_tree_depth_hits", "mean_step_size")
    println(rpad(k, 22), diag[k])
end
H.save_fit(joinpath(homedir(), "Work/tumour-pkpd/logs/2026-11-04-p3-inference-synth-$(tag).jld2"), pd)

# --- recovery table ----------------------------------------------------------------------------
q(x) = (quantile(x, 0.05), median(x), quantile(x, 0.95))
function row(name, tv, draws)
    lo, md, hi = q(draws)
    inside = lo <= tv <= hi
    @printf("%-9s %9.4f %9.4f   [%8.4f, %8.4f]   %s\n", name, tv, md, lo, hi, inside ? "yes" : "NO")
    return inside
end
println("\n## population parameters (log scale; omega/sigma natural scale)")
println("param        true    median        90% CrI           in CrI")
res = [row("μ_λ0", truth.μ_λ0, pd.μ_λ0), row("ω_λ0", truth.ω_λ0, pd.ω_λ0),
       row("μ_k2", truth.μ_k2, pd.μ_k2), row("ω_k2", truth.ω_k2, pd.ω_k2),
       row("logλ1", truth.logλ1, pd.logλ1), row("log Vtr", truth.logλ1 - truth.μ_λ0, pd.logλ1 .- pd.μ_λ0),
       row("logk1", truth.logk1, pd.logk1),
       row("σ", truth.σ, pd.σ)]
println("population params inside 90% CrI: $(sum(res))/$(length(res))")
println("posterior sd / prior sd of logk1: ", round(std(pd.logk1) / PRIORS.logk1[2]; digits = 2))

println("\n## per-unit coverage (90% CrI)")
for (nm, post, tv) in (("λ0", pd.λ0, tru.λ0), ("k2", pd.k2, tru.k2))
    cov = mean(quantile(post[:, i], 0.05) <= tv[i] <= quantile(post[:, i], 0.95) for i in 1:N)
    lcov50 = mean(quantile(post[:, i], 0.25) <= tv[i] <= quantile(post[:, i], 0.75) for i in 1:N)
    relerr = median(abs.(log.(vec(median(post; dims = 1))) .- log.(tv)))
    @printf("%s: 90%% coverage %.2f (%d/%d), 50%% coverage %.2f, median |log error| of posterior median %.3f\n",
            nm, cov, round(Int, cov * N), N, lcov50, relerr)
end
if latent_w0
    for (nm, post, tv) in (("w0 control", pd.w0_control_draws, tru.w0c), ("w0 treated", pd.w0_treated_draws, tru.w0t))
        cov = mean(quantile(post[:, i], 0.05) <= tv[i] <= quantile(post[:, i], 0.95) for i in 1:N)
        @printf("%s: 90%% coverage %.2f\n", nm, cov)
    end
end
# k2 for censored/strong responders vs others
resp = [any(==(0.0), u.treated.v) for u in units]
if any(resp) && !all(resp)
    for (lab, msk) in (("units with undetectable treated obs", resp), ("other units", .!resp))
        c = mean(quantile(pd.k2[:, i], 0.05) <= tru.k2[i] <= quantile(pd.k2[:, i], 0.95) for i in findall(msk))
        @printf("  k2 coverage, %s (n=%d): %.2f\n", lab, count(msk), c)
    end
end

# --- pointwise log-lik timing -------------------------------------------------------------------
println("\n## pointwise_loglik")
t0 = time()
ll = H.pointwise_loglik(pd, units; draws = 1:min(200, length(pd.σ)))   # uses each draw's latent baselines
@printf("pointwise_loglik 200 draws x %d obs: %.2f s, finite: %s\n", size(ll, 2), time() - t0, all(isfinite, ll))
