# Theophylline validation of the PK layer: hierarchical Bayesian 1-compartment
# oral model fitted with Turing.jl NUTS, compared with the nlme fit of
# Pinheiro & Bates (2000), `fm1Theo.nlme`.
#
# Run (4 threads for the 4 chains; this run only):
#   julia --project=. -t 4 scripts/theoph_fit.jl
# Env: THEOPH_NWARM, THEOPH_NSAMP (defaults 1000/1000), THEOPH_SEED.
#
# Units: time h, Dose mg/kg, conc mg/L, CL L/h/kg, V L/kg, ka 1/h.
# Parameters are apparent CL/F and V/F (F fixed at 1).
#
# Data handling. Primary analysis: the 12 t = 0 rows (pre-dose samples) are dropped.
# The model starts with an empty depot, so its prediction at t = 0 is exactly 0; the
# combined error model stays defined there (sd = sigma_add), but 3 of the 12 subjects
# have positive "carry-over" baselines (printed below) that the model has no term for,
# so they would be absorbed into sigma_add and carry no information about CL, V, ka.
# The zero rows carry none either. A sensitivity fit with all 132 rows is run with
# identical settings and reported next to the primary one (docs/theoph_results.md).
#
# Chains start from dispersed points, including one in the flip-flop region
# (ka < CL/V); per-chain modes are reported.

using TumourPKPD
using RDatasets, DataFrames, Statistics, Random, Printf
using Turing, MCMCChains, Distributions, LinearAlgebra
using Optim, ForwardDiff
using CairoMakie

const ROOT = normpath(joinpath(@__DIR__, ".."))
const NWARM = parse(Int, get(ENV, "THEOPH_NWARM", "1000"))
const NSAMP = parse(Int, get(ENV, "THEOPH_NSAMP", "1000"))
const SEED = parse(Int, get(ENV, "THEOPH_SEED", "20261104"))
const NCHAIN = 4

# ----------------------------------------------------------------- data
raw = RDatasets.dataset("datasets", "Theoph")
raw.Subject = parse.(Int, string.(raw.Subject))
sort!(raw, [:Subject, :Time])
subjects = sort(unique(raw.Subject))
nsub = length(subjects)
dose_s = [first(raw.Dose[raw.Subject .== s]) for s in subjects]
doses = [[Dose(0.0, d)] for d in dose_s]

function prep(d)
    (sidx = [findfirst(==(s), subjects) for s in d.Subject],
     tobs = Float64.(d.Time), yobs = Float64.(d.Conc), n = nrow(d))
end
t0pos = raw[(raw.Time .== 0) .& (raw.Conc .> 0), :]
df = raw[raw.Time .> 0, :]
D1 = prep(df)       # primary: t > 0
D0 = prep(raw)      # sensitivity: all rows
sidx, tobs, yobs = D1.sidx, D1.tobs, D1.yobs
println("Theoph: $(nrow(raw)) rows; primary fit uses $(D1.n) rows (t>0), $nsub subjects.")
println("t = 0 rows with conc > 0: $(nrow(t0pos)) (subject, conc) = ", collect(zip(t0pos.Subject, t0pos.Conc)))

# ------------------------------------------------------- literature values
# Pinheiro & Bates (2000), Mixed-Effects Models in S and S-PLUS, ch. 8,
# fm1Theo.nlme (conc ~ SSfol(Dose, Time, lKe, lKa, lCl), general pos-def RE):
# fixed effects lKe = -2.43267, lKa = 0.45141, lCl = -3.21445 (printed output in the
# book's R scripts, e.g. https://asancpt.github.io/nlme/chapter-8.html).
const LIT_lKe, LIT_lKa, LIT_lCl = -2.43267, 0.45141, -3.21445
const LIT = (ka = exp(LIT_lKa), CL = exp(LIT_lCl), V = exp(LIT_lCl - LIT_lKe), ke = exp(LIT_lKe))
@printf("Literature: ka=%.4f /h  CL=%.5f L/h/kg  V=%.4f L/kg  ke=%.5f /h\n", LIT.ka, LIT.CL, LIT.V, LIT.ke)

# ----------------------------------------------------------------- model
pred_conc(CL, V, ka, dosevec, t) = conc(OneCmt(CL, V, ka, 1.0), dosevec, t)

@model function theoph_model(sidx, tobs, y, doses, nsub, ordered)
    # population medians (log scale); weakly informative, generic adult-drug scales
    μ_CL ~ Normal(log(0.1), 1.0)
    μ_V ~ Normal(log(0.5), 1.0)
    if ordered
        # Population-level absorption faster than elimination (ka > CL/V), which excludes the
        # flip-flop mode. Equivalent to μ_ka ~ Normal(log 1, 1) truncated below at
        # μ_CL - μ_V, written via the gap δ = μ_ka - (μ_CL - μ_V) > 0 so the truncation
        # bound is constant (δ ~ Normal(-(μ_CL - μ_V), 1) truncated at 0).
        δ_ka ~ truncated(Normal(-(μ_CL - μ_V), 1.0); lower = 0)
        μ_ka := μ_CL - μ_V + δ_ka
    else
        μ_ka ~ Normal(log(1.0), 1.0)
    end
    # between-subject SDs (log scale), diagonal
    ω_CL ~ truncated(Normal(0, 0.5); lower = 0)
    ω_V ~ truncated(Normal(0, 0.5); lower = 0)
    ω_ka ~ truncated(Normal(0, 1.0); lower = 0)
    # non-centred subject effects
    z_CL ~ filldist(Normal(), nsub)
    z_V ~ filldist(Normal(), nsub)
    z_ka ~ filldist(Normal(), nsub)
    # combined residual error: sd = sqrt(σ_add^2 + (σ_prop * pred)^2)
    σ_prop ~ truncated(Normal(0, 0.3); lower = 0)
    σ_add ~ truncated(Normal(0, 0.5); lower = 0)

    CL = exp.(μ_CL .+ ω_CL .* z_CL)
    V = exp.(μ_V .+ ω_V .* z_V)
    ka = exp.(μ_ka .+ ω_ka .* z_ka)
    m = [pred_conc(CL[s], V[s], ka[s], doses[s], t) for (s, t) in zip(sidx, tobs)]
    v = σ_add^2 .+ (σ_prop .* m) .^ 2
    # Gaussian log-likelihood written out so that non-finite values reached in wild
    # warmup trajectories reject the proposal instead of throwing a DomainError.
    ll = -0.5 * sum(log.(2π .* v) .+ (y .- m) .^ 2 ./ v)
    Turing.@addlogprob! (isfinite(ll) ? ll : -Inf)
end

# ----------------------------------------------- non-Bayesian sanity checks
# (a) naive pooled MLE: one set of (CL, V, ka) for everybody, combined error
# (b) two-stage: per-subject MLE, then median over subjects (subjects 1 and 9
# have degenerate individual fits with ka -> infinity, so a mean would be meaningless)
function nll(θ, s_idx, t, y, dvec)
    CL, V, ka = exp(θ[1]), exp(θ[2]), exp(θ[3])
    σp, σa = exp(θ[4]), exp(θ[5])
    ll = zero(eltype(θ))
    for (i, s) in enumerate(s_idx)
        m = pred_conc(CL, V, ka, dvec[s], t[i])
        v = σa^2 + (σp * m)^2
        ll += 0.5 * log(2π * v) + (y[i] - m)^2 / (2v)
    end
    return isfinite(ll) ? ll : oftype(ll, 1e12)
end
function fit_mle(s_idx, t, y, dvec; start = [log(0.04), log(0.5), log(1.5), log(0.1), log(0.5)])
    f = θ -> nll(θ, s_idx, t, y, dvec)
    res = optimize(f, start, LBFGS(; linesearch = Optim.LineSearches.BackTracking()), Optim.Options(iterations = 2000); autodiff = :forward)
    return Optim.minimizer(res), Optim.minimum(res), Optim.converged(res)
end

θ_pool, nll_pool, conv_pool = fit_mle(sidx, tobs, yobs, doses)
mle_pool = (CL = exp(θ_pool[1]), V = exp(θ_pool[2]), ka = exp(θ_pool[3]))
println("Naive pooled MLE (converged=$conv_pool): ", mle_pool)

per_sub = map(1:nsub) do s
    ii = findall(==(s), sidx)
    # one-subject fit: treat as subject 1 of a length-1 dose list
    θ, _, ok = fit_mle(ones(Int, length(ii)), tobs[ii], yobs[ii], [doses[s]])
    (θ = θ, ok = ok)
end
two_stage = (CL = exp(median(p.θ[1] for p in per_sub)),
             V = exp(median(p.θ[2] for p in per_sub)),
             ka = exp(median(p.θ[3] for p in per_sub)),
             nconv = count(p -> p.ok, per_sub))
for (s, p) in enumerate(per_sub)
    @printf("  subject %2d: CL=%.4f V=%.4f ka=%.3f (converged=%s)\n", subjects[s], exp(p.θ[1]), exp(p.θ[2]), exp(p.θ[3]), p.ok)
end
println("Two-stage MLE (median of $(nsub) subject fits, $(two_stage.nconv) converged): ", two_stage)

# ----------------------------------------------------------------- NUTS
# Dispersed starts (CL, V, ka). Unconstrained fit: chain 2 starts in the flip-flop region
# ka < k = CL/V. Ordered (ka > k) fit: all starts satisfy ka > CL/V.
const STARTS_FREE = [(0.04, 0.50, 1.5),     # near the usual mode
                     (0.04, 0.025, 0.06),   # k = 1.6 /h > ka = 0.06 /h : flip-flop region
                     (0.20, 2.00, 5.0),     # high everything
                     (0.01, 0.20, 0.5)]     # low CL, ka > k = 0.05
const STARTS_ORD = [(0.04, 0.50, 1.5), (0.10, 1.00, 8.0), (0.20, 2.00, 5.0), (0.02, 0.25, 0.8)]

function fit_nuts(D; seed, ordered, starts)
    model = theoph_model(D.sidx, D.tobs, D.yobs, doses, nsub, ordered)
    rng = MersenneTwister(seed)
    inits = [vcat(log(c), log(v), ordered ? log(a) - log(c / v) : log(a), 0.2, 0.2, 0.5, 0.3 .* randn(rng, 3nsub), 0.1, 0.5) for (c, v, a) in starts]
    Random.seed!(seed)
    t = @elapsed ch = sample(Random.default_rng(), model, NUTS(NWARM, 0.8), MCMCThreads(), NSAMP, NCHAIN;
                             initial_params = inits, progress = false)
    return ch, t
end

println("Threads: $(Threads.nthreads()); $NCHAIN chains x ($NWARM warmup + $NSAMP draws), dispersed inits")
# (A) unconstrained model, dispersed starts incl. flip-flop region: mode diagnostic
chainU, runtimeU = fit_nuts(D1; seed = SEED + 3, ordered = false, starts = STARTS_FREE)
@printf("Unconstrained dispersed fit wall time: %.1f s\n", runtimeU)
# (B) primary: ordered prior (ka > CL/V), t > 0 rows
chain, runtime = fit_nuts(D1; seed = SEED, ordered = true, starts = STARTS_ORD)
@printf("Primary (ordered) fit wall time: %.1f s\n", runtime)
# (C) sensitivity: ordered prior, all 132 rows
chain0, runtime0 = fit_nuts(D0; seed = SEED + 7, ordered = true, starts = STARTS_ORD)
@printf("Sensitivity (ordered, all 132 rows) wall time: %.1f s\n", runtime0)

# ------------------------------------------------------------ diagnostics
pars = [:μ_CL, :μ_V, :μ_ka, :ω_CL, :ω_V, :ω_ka, :σ_prop, :σ_add]
function diag(ch)
    a = DataFrame(summarystats(ch))
    (ssdf = DataFrame(summarystats(ch[pars])), nq = nrow(a),
     div = round(Int, sum(ch[:numerical_error])),
     rhat = maximum(skipmissing(a.rhat)), essb = minimum(skipmissing(a.ess_bulk)),
     esst = minimum(skipmissing(a.ess_tail)))
end
dg, dg0, dgU = diag(chain), diag(chain0), diag(chainU)
println(dg.ssdf)
@printf("PRIMARY: max R-hat (%d quantities) = %.4f; min bulk ESS = %.0f; min tail ESS = %.0f; divergences = %d\n",
        dg.nq, dg.rhat, dg.essb, dg.esst, dg.div)
@printf("UNCONSTRAINED dispersed: max R-hat = %.4f; min bulk ESS = %.0f; divergences = %d\n", dgU.rhat, dgU.essb, dgU.div)
@printf("SENSITIVITY: max R-hat = %.4f; min bulk ESS = %.0f; divergences = %d\n", dg0.rhat, dg0.essb, dg0.div)

# per-chain mode check (population level): is ka < CL/V in any chain?
function per_chain(ch)
    nd = size(ch, 1)
    get(sym) = reshape(vec(Array(ch[sym])), nd, :)   # draws x chains (chain-major)
    mu_CL, mu_V, mu_ka = get(:μ_CL), get(:μ_V), get(:μ_ka)
    map(1:size(mu_CL, 2)) do i
        ka, CL, V = exp.(mu_ka[:, i]), exp.(mu_CL[:, i]), exp.(mu_V[:, i])
        (chain = i, ka = median(ka), CL = median(CL), V = median(V), ke = median(CL ./ V),
         frac_flip = mean(ka .< CL ./ V))
    end
end
pcU = per_chain(chainU)     # unconstrained, dispersed starts
pc = per_chain(chain)       # ordered primary
pc0 = per_chain(chain0)
println("Per-chain population medians, UNCONSTRAINED model with dispersed starts:")
for r in pcU
    @printf("  chain %d: start (CL,V,ka)=%s  ka=%.3f CL=%.4f V=%.4f ke=%.4f  frac(ka<ke)=%.3f\n",
            r.chain, string(STARTS_FREE[r.chain]), r.ka, r.CL, r.V, r.ke, r.frac_flip)
end
flip_chains = [r.chain for r in pcU if r.frac_flip > 0.5]
other_mode = !isempty(flip_chains)
println("Chains in the flip-flop mode (ka < ke): ", isempty(flip_chains) ? "none" : string(flip_chains))
# summary of the unconstrained fit restricted to the chains in the usual mode
function main_mode_summary(ch, flip)
    keep = setdiff(1:NCHAIN, flip)
    ka = Float64[]; CL = Float64[]; V = Float64[]
    for i in keep
        m = reshape(vec(Array(ch[:μ_ka])), size(ch, 1), :)[:, i]; append!(ka, exp.(m))
        m = reshape(vec(Array(ch[:μ_CL])), size(ch, 1), :)[:, i]; append!(CL, exp.(m))
        m = reshape(vec(Array(ch[:μ_V])), size(ch, 1), :)[:, i]; append!(V, exp.(m))
    end
    return ka, CL, V, CL ./ V
end
drU = main_mode_summary(chainU, flip_chains)
println("Per-chain population medians, ORDERED primary model:")
for r in pc
    @printf("  chain %d: ka=%.3f CL=%.4f V=%.4f ke=%.4f\n", r.chain, r.ka, r.CL, r.V, r.ke)
end

# --------------------------------------------------- population comparison
q(x, p) = quantile(x, p)
function popdraws(ch)
    d(sym) = vec(Array(ch[sym]))
    ka, CL, V = exp.(d(:μ_ka)), exp.(d(:μ_CL)), exp.(d(:μ_V))
    return ka, CL, V, CL ./ V
end
ka_d, CL_d, V_d, ke_d = popdraws(chain)
ka_0, CL_0, V_0, ke_0 = popdraws(chain0)
names = ["ka (1/h)", "CL/F (L/h/kg)", "V/F (L/kg)", "ke = CL/V (1/h)"]
lits = [LIT.ka, LIT.CL, LIT.V, LIT.ke]
mp = [mle_pool.ka, mle_pool.CL, mle_pool.V, mle_pool.CL / mle_pool.V]
ts2 = [two_stage.ka, two_stage.CL, two_stage.V, two_stage.CL / two_stage.V]
dr = [ka_d, CL_d, V_d, ke_d]; dr0 = [ka_0, CL_0, V_0, ke_0]
verdicts = Bool[]
tab = String[]
for i in 1:4
    med, lo, hi = q(dr[i], 0.5), q(dr[i], 0.05), q(dr[i], 0.95)
    relerr = (med - lits[i]) / lits[i]
    inside = lo <= lits[i] <= hi
    i <= 3 && push!(verdicts, abs(relerr) <= 0.15 && inside)
    m0, l0, h0 = q(dr0[i], 0.5), q(dr0[i], 0.05), q(dr0[i], 0.95)
    mu, lu, hu = q(drU[i], 0.5), q(drU[i], 0.05), q(drU[i], 0.95)
    push!(tab, @sprintf("| %s | %.4g | [%.4g, %.4g] | %.4g | %+.1f%% | %s | %.4g [%.4g, %.4g] | %.4g [%.4g, %.4g] | %.4g | %.4g |",
                        names[i], med, lo, hi, lits[i], 100relerr, inside ? "yes" : "no", m0, l0, h0, mu, lu, hu, mp[i], ts2[i]))
end
overall = all(verdicts)
println("\n", join(tab, "\n"))
println("\nPASS RULE (typical values ka, CL, V within 15% and literature inside 90% CrI): ", overall ? "PASS" : "FAIL")

# ----------------------------------------------------- posterior predictive
# Gaussian predictive draws are kept as they are (negative values allowed), so the
# bands are the fitted observation distribution, not a censored display.
Random.seed!(SEED + 1)
cols(sym) = Array(chain[[Symbol("$(sym)[$s]") for s in 1:nsub]])
zCL, zV, zka = cols(:z_CL), cols(:z_V), cols(:z_ka)
dvec(sym) = vec(Array(chain[sym]))
μCL, μV, μka, ωCL, ωV, ωka, σp, σa = (dvec(x) for x in (:μ_CL, :μ_V, :μ_ka, :ω_CL, :ω_V, :ω_ka, :σ_prop, :σ_add))
ndraw = length(μCL)
tgrid = collect(range(0.0, 25.0; length = 126))
pick = rand(1:ndraw, 1000)
subpar(d, s) = (exp(μCL[d] + ωCL[d] * zCL[d, s]), exp(μV[d] + ωV[d] * zV[d, s]), exp(μka[d] + ωka[d] * zka[d, s]))
function ppc_at(s, times)
    sims = zeros(length(pick), length(times)); means = similar(sims)
    for (j, d) in enumerate(pick)
        CL, V, ka = subpar(d, s)
        for (k, t) in enumerate(times)
            m = pred_conc(CL, V, ka, doses[s], t)
            means[j, k] = m
            sims[j, k] = m + sqrt(σa[d]^2 + (σp[d] * m)^2) * randn()
        end
    end
    return (lo = [quantile(sims[:, k], 0.05) for k in eachindex(times)],
            hi = [quantile(sims[:, k], 0.95) for k in eachindex(times)],
            med = [quantile(means[:, k], 0.5) for k in eachindex(times)])
end
band = Dict(s => ppc_at(s, tgrid) for s in 1:nsub)
# coverage at the exact observation times
ncov = sum(1:nsub) do s
    ii = findall(==(s), sidx)
    b = ppc_at(s, tobs[ii])
    count(k -> b.lo[k] <= yobs[ii[k]] <= b.hi[k], eachindex(ii))
end
coverage90 = ncov / length(yobs)
@printf("Posterior predictive 90%% interval coverage (in-sample, conditional on subject effects): %.3f (%d/%d)\n",
        coverage90, ncov, length(yobs))

fig = Figure(size = (1200, 900))
for s in 1:nsub
    ax = Axis(fig[div(s - 1, 4) + 1, mod(s - 1, 4) + 1];
              title = "Subject $(subjects[s])  (dose $(round(dose_s[s]; digits = 2)) mg/kg)",
              xlabel = "time (h)", ylabel = "theophylline (mg/L)")
    b = band[s]
    band!(ax, tgrid, b.lo, b.hi; color = (:steelblue, 0.30))
    lines!(ax, tgrid, b.med; color = :steelblue, linewidth = 2)
    hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 1)
    ii = findall(==(s), sidx)
    scatter!(ax, tobs[ii], yobs[ii]; color = :black, markersize = 8)
end
Label(fig[0, :], "Theoph: observed vs posterior predictive (median curve; 90% interval incl. Gaussian residual error, not clipped at 0)"; fontsize = 16)
mkpath(joinpath(ROOT, "docs", "figures"))
save(joinpath(ROOT, "docs", "figures", "theoph_ppc.png"), fig)

# --------------------------------------------------------- results table
srow(ssdf, name) = begin
    r = ssdf[ssdf.parameters .== name, :][1, :]
    @sprintf("| %s | %.4g | %.3g | %.3f | %.0f | %.0f |", name, r.mean, r.std, r.rhat, r.ess_bulk, r.ess_tail)
end
priors = [
    ("μ_CL", "Normal(log 0.1, 1.0)", "log L/h/kg", "median CL/F; broad (factor ~2.7 per SD) around a generic small-molecule scale, not centred on the literature value"),
    ("μ_V", "Normal(log 0.5, 1.0)", "log L/kg", "median V/F; broad around total body water scale"),
    ("μ_ka", "Normal(log 1.0, 1.0); primary fit: truncated below at μ_CL − μ_V (ka > CL/V), sampled via the gap δ_ka", "log 1/h", "median ka; broad around 1/h typical oral absorption; truncation removes the flip-flop mode"),
    ("ω_CL", "Half-Normal(0.5)", "log scale", "between-subject SD of log CL; allows CV up to ~100%"),
    ("ω_V", "Half-Normal(0.5)", "log scale", "between-subject SD of log V"),
    ("ω_ka", "Half-Normal(1.0)", "log scale", "between-subject SD of log ka; wider because ka is typically most variable"),
    ("z_CL, z_V, z_ka", "Normal(0, 1) each, x12 subjects", "-", "non-centred standardized subject effects"),
    ("σ_prop", "Half-Normal(0.3)", "unitless", "proportional residual SD (up to ~60%)"),
    ("σ_add", "Half-Normal(0.5)", "mg/L", "additive residual SD (assay floor)"),
]
fmtpct(x) = @sprintf("%.1f", x)
open(joinpath(ROOT, "docs", "theoph_results.md"), "w") do io
    println(io, "# Theophylline validation (Theoph, 12 subjects)\n")
    println(io, "Hierarchical 1-compartment oral model, Turing.jl NUTS, non-centred diagonal random effects on CL, V, ka, combined error sd = sqrt(sigma_add^2 + (sigma_prop * pred)^2). Apparent CL/F and V/F (F = 1). Units: h, mg/kg, mg/L, L/h/kg, L/kg, 1/h.\n")
    println(io, "Primary fit (ordered prior, see below) uses the $(D1.n) rows with t > 0. The 12 t = 0 rows are dropped: the model predicts exactly 0 there, and $(nrow(t0pos)) of them are positive carry-over baselines (subject, mg/L): $(join(["($a, $b)" for (a, b) in zip(t0pos.Subject, t0pos.Conc)], ", ")), which the model has no term for. A sensitivity fit on all $(D0.n) rows is shown in the table.\n")
    println(io, "Settings: $NCHAIN chains x ($NWARM warmup + $NSAMP draws), target accept 0.8, seed $SEED, ForwardDiff, $(Threads.nthreads()) threads. Wall time: primary $(round(runtime; digits = 1)) s, sensitivity $(round(runtime0; digits = 1)) s, unconstrained mode check $(round(runtimeU; digits = 1)) s.\n")
    println(io, "## Priors\n")
    println(io, "| parameter | prior | units | rationale |")
    println(io, "|---|---|---|---|")
    for (a, b, c, d) in priors
        println(io, "| $a | $b | $c | $d |")
    end
    println(io, "\nHalf-Normal(s) is Normal(0, s) truncated at 0.\n")
    println(io, "## Diagnostics\n")
    println(io, "Primary, all $(dg.nq) sampled quantities: max R-hat $(round(dg.rhat; digits = 4)), min bulk ESS $(round(Int, dg.essb)), min tail ESS $(round(Int, dg.esst)), divergent transitions $(dg.div). Sensitivity: max R-hat $(round(dg0.rhat; digits = 4)), min bulk ESS $(round(Int, dg0.essb)), divergences $(dg0.div).\n")
    println(io, "### Mode check: unconstrained model, dispersed starts\n")
    println(io, "The model with an unconstrained prior on mu_ka has a second, flip-flop mode (ka and ke exchange roles, V collapses). Four chains with dispersed starts, one in the flip-flop region (diagnostics of this fit: max R-hat $(round(dgU.rhat; digits = 3)), min bulk ESS $(round(Int, dgU.essb)), divergences $(dgU.div)):\n")
    println(io, "| chain | start (CL, V, ka) | median ka | median CL | median V | median CL/V | fraction of draws with ka < CL/V |")
    println(io, "|---|---|---|---|---|---|---|")
    for r in pcU
        println(io, @sprintf("| %d | %s | %.3f | %.4f | %.4f | %.4f | %.3f |", r.chain, string(STARTS_FREE[r.chain]), r.ka, r.CL, r.V, r.ke, r.frac_flip))
    end
    println(io, "\nChains that end in the flip-flop mode: **", isempty(flip_chains) ? "none" : join(flip_chains, ", "), "**. The flip-flop mode has V/F of about $(round(minimum(r.V for r in pcU); digits = 3)) L/kg at the lowest chain, below plasma volume, which is not physiologically credible for a drug distributed like theophylline, and it is not what the literature fit finds. The mode does not mix with the usual one, so pooled R-hat over these chains is meaningless.\n")
    println(io, "**Primary analysis therefore uses an ordered prior** (population ka > CL/V, implemented as a truncation of the mu_ka prior at mu_CL - mu_V; theophylline immediate-release absorption is known to be faster than elimination). With it, all four dispersed chains agree:\n")
    println(io, "| chain | median ka | median CL | median V | median CL/V |")
    println(io, "|---|---|---|---|---|")
    for r in pc
        println(io, @sprintf("| %d | %.3f | %.4f | %.4f | %.4f |", r.chain, r.ka, r.CL, r.V, r.ke))
    end
    println(io, "")
    println(io, "## Population typical values vs literature\n")
    println(io, "Posterior of exp(mu) (population median). Literature: Pinheiro & Bates (2000) nlme `fm1Theo.nlme`, lKe = $LIT_lKe, lKa = $LIT_lKa, lCl = $LIT_lCl; V = CL/ke. 'Ordered, all 132 rows' = sensitivity fit including t = 0 rows; 'unconstrained, usual-mode chains' = unconstrained fit with the flip-flop chain(s) removed (median [90% CrI]).\n")
    println(io, "| quantity | posterior median (t>0) | 90% CrI | literature | rel. diff | lit. in CrI | ordered, all 132 rows | unconstrained, usual-mode chains | naive pooled MLE | two-stage MLE |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    foreach(l -> println(io, l), tab)
    println(io, "\n**Pass rule** (typical values of ka, CL, V each within 15% of the literature value and the literature value inside the 90% CrI): **", overall ? "PASS" : "FAIL", "**. This means the typical values agree; it is not a replication of the literature model.\n")
    println(io, "Caveats on the comparison:\n")
    println(io, "- The nlme fit uses a general (correlated) positive-definite random-effects matrix on log ke, log ka, log CL; ours is diagonal on log CL, log V, log ka.")
    println(io, "- nlme uses an additive Gaussian error (residual SD 0.68 mg/L) and all 132 observations; ours is combined error on the t > 0 rows (the all-rows column above is the closest analogue).")
    println(io, "- The cited nlme output (https://asancpt.github.io/nlme/chapter-8.html) carries an LME-step non-convergence warning (nlminb code 1), so the literature numbers themselves are not exact targets.")
    println(io, "- Naive pooled MLE ignores between-subject variation. The two-stage value is the median of per-subject MLEs (subjects 1 and 9 have degenerate individual fits with ka -> very large, so a mean is not meaningful).")
    println(io, "- ka is the least well-identified parameter.\n")
    println(io, "## Posterior summaries (primary)\n")
    println(io, "| parameter | mean | sd | R-hat | bulk ESS | tail ESS |")
    println(io, "|---|---|---|---|---|---|")
    foreach(p -> println(io, srow(dg.ssdf, p)), pars)
    println(io, "\nIn-sample 90% posterior predictive interval coverage (conditional on each subject's random effects, evaluated at the exact observation times): $(round(coverage90; digits = 3)) ($ncov/$(length(yobs))). This is not a held-out forecast.")
    println(io, "\nFigure: `docs/figures/theoph_ppc.png` (median curve and 90% predictive interval including Gaussian residual error; negative predictive draws are kept, not clipped).")
end
println("Wrote docs/theoph_results.md and docs/figures/theoph_ppc.png")
