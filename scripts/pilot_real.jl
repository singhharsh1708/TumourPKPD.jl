# Pilot fit on real PDXE data (subsample of units), PK fixed at the regimen prior medians.
#
#   julia --project=TumourPKPD.jl -t 4 TumourPKPD.jl/scripts/pilot_real.jl drug nunits [nchains nsamples nadapt seed latent_w0]
using TumourPKPD, Random, Statistics, Printf
using TumourPKPD: OneCmt
isdefined(TumourPKPD, :fit_drug) ||
    Base.include(TumourPKPD, joinpath(@__DIR__, "..", "src", "hierarchical.jl"))
const H = TumourPKPD

const _flusher = Timer(_ -> flush(stdout), 1.0; interval = 1.0)

drug     = length(ARGS) >= 1 ? ARGS[1] : "binimetinib"
nunits   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30
nchains  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
nsamples = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 300
nadapt   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 300
seed     = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1
latent   = length(ARGS) >= 7 ? parse(Bool, ARGS[7]) : true

all_units = drug_dataset(drug)
rng = MersenneTwister(seed)
units = length(all_units) <= nunits ? all_units : all_units[sort(randperm(rng, length(all_units))[1:nunits])]
doses, pkp = regimen(drug)
pk = OneCmt(CL = pkp.CL, V = pkp.V, ka = pkp.ka, F = pkp.F)
cref = H.reference_cref(pk, doses)
println("# pilot $drug: $(length(units)) of $(length(all_units)) units, $(length(doses)) doses, cref=$(round(cref; digits=3)), LOD=$(detection_limit())")
println("# threads=$(Threads.nthreads()) chains=$nchains samples=$nsamples adapt=$nadapt latent_w0=$latent")
nobs = sum(length(u.control.t) + length(u.treated.t) - 2 for u in units)
nz = sum(count(==(0.0), u.treated.v) + count(==(0.0), u.control.v) for u in units)
println("post-baseline obs=$nobs  zero obs=$nz  max curve length (days)=",
        maximum(max(last(u.control.t), last(u.treated.t)) for u in units))

chain, pd = H.fit_drug(units, doses, pk, cref; drug, nchains, nsamples, nadapt,
                       rng = MersenneTwister(seed + 100), progress = false, latent_w0 = latent, seed, log_every = 100)
dg = pd.diagnostics
niter = nchains * (nsamples + nadapt)
@printf("runtime %.1f s (%d iterations -> %.1f s per 100 iterations)\n", dg["runtime_s"], niter, 100 * dg["runtime_s"] / niter)
for k in ("max_rhat", "min_ess_bulk", "n_divergences", "mean_acceptance", "mean_tree_depth", "max_tree_depth_hits", "mean_step_size")
    println(rpad(k, 22), dg[k])
end
q(x) = (quantile(x, 0.05), median(x), quantile(x, 0.95))
println("\nparam   median   [90% CrI]")
for (nm, x) in (("μ_λ0", pd.μ_λ0), ("ω_λ0", pd.ω_λ0), ("μ_k2", pd.μ_k2), ("ω_k2", pd.ω_k2),
                ("logλ1", pd.logλ1), ("logk1", pd.logk1), ("σ", pd.σ))
    lo, md, hi = q(x)
    @printf("%-7s %8.3f  [%8.3f, %8.3f]\n", nm, md, lo, hi)
end
H.save_fit(joinpath(homedir(), "Work/tumour-pkpd/logs/2026-11-04-p2-inference-pilot-$(drug)-$(length(units)).jld2"), pd)
