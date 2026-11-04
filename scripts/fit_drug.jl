# Fit the hierarchical Simeoni model for one drug and save it for validation.
#
#   julia --project=. -t 4 scripts/fit_drug.jl <drug> <full|late|holdout> [nchains warmup draws] [--seed=1] [--units=N]
#                                                [--target=0.8] [--nolatent] [--noprocs] [--out=fits]
#
#   full     all PDX models, all points                       -> fits/<drug>.jld2
#   late     every curve cut with split_late(frac=0.3)         -> fits/<drug>_late.jld2
#            (cutoffs saved as "<model>|treated" / "<model>|control" = last training day)
#   holdout  PDX models left out by split_models(frac=0.2, seed=1) -> fits/<drug>_holdout.jld2
#
# These are exactly the splits scripts/validation_report.jl re-derives and verifies against the saved
# metadata (diagnostics keys: split, cutoffs, train_models, lod, nchains, draws_per_chain, seed, latent_w0).
# Defaults: 4 chains, 500 warmup, 500 draws, latent baselines. stdout is flushed every second.
# With -t N (N > 1) and nchains > 1 the chains run as separate single-threaded processes, at most N at a time
# (scripts/procchains.jl; in-process chains scale badly). `--noprocs` samples chains in sequence in this process
# with threaded units instead.
using TumourPKPD, Random, Statistics, Printf, Dates
using TumourPKPD: OneCmt
const M = TumourPKPD
isdefined(M, :fit_drug) || Base.include(M, joinpath(@__DIR__, "..", "src", "hierarchical.jl"))
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "procchains.jl"))
const _flusher = Timer(_ -> flush(stdout), 1.0; interval = 1.0)

const LATE_FRAC = 0.3
const MODEL_FRAC = 0.2
const SPLIT_SEED = 1

function parse_cli(args)
    pos = String[]; opts = Dict{String,String}()
    for a in args
        if startswith(a, "--")
            kv = split(a[3:end], '='; limit = 2)
            opts[kv[1]] = length(kv) == 2 ? kv[2] : "true"
        else
            push!(pos, a)
        end
    end
    length(pos) >= 2 || error("usage: fit_drug.jl <drug> <full|late|holdout> [nchains warmup draws] [--seed=1] [--units=N] [--target=0.8] [--nolatent] [--noprocs] [--out=fits]")
    return pos, opts
end

const TAG = Ref("")
stamp(msg) = println("[", Dates.format(now(), "HH:MM:SS"), "]", TAG[], " ", msg)

function main(args)
    pos, opts = parse_cli(args)
    drug, mode = pos[1], pos[2]
    mode in ("full", "late", "holdout") || error("mode must be full, late or holdout")
    nchains = length(pos) >= 3 ? parse(Int, pos[3]) : 4
    nadapt  = length(pos) >= 4 ? parse(Int, pos[4]) : 500
    ndraws  = length(pos) >= 5 ? parse(Int, pos[5]) : 500
    seed    = parse(Int, get(opts, "seed", "1"))
    target  = parse(Float64, get(opts, "target", "0.8"))
    latent  = !haskey(opts, "nolatent")
    outdir  = abspath(get(opts, "out", joinpath(ROOT, "fits")))

    haskey(opts, "chain-only") && (TAG[] = "[chain " * opts["chain-only"] * "]")
    stamp("fit_drug $drug $mode chains=$nchains warmup=$nadapt draws=$ndraws seed=$seed latent_w0=$latent threads=$(Threads.nthreads())")
    units = drug_dataset(drug)
    if haskey(opts, "units")                       # subsample for smoke tests, keeps split logic on the subsample
        n = parse(Int, opts["units"])
        units = units[sort(randperm(MersenneTwister(0), length(units))[1:min(n, length(units))])]
    end
    _, dropped = M.prepare_units(units; warn = false)
    stamp("$drug: $(length(units)) paired PDX models, $(length(dropped)) dropped for an invalid day-0 volume $(dropped)")
    doses, pkp = regimen(drug)
    pk = OneCmt(CL = pkp.CL, V = pkp.V, ka = pkp.ka, F = pkp.F)
    cref = M.reference_cref(pk, doses)
    stamp("regimen: $(length(doses)) doses, cref=$(round(cref; digits = 4)), LOD=$(detection_limit()) mm^3")

    cutoffs = Dict{String,Float64}()
    split = Dict{String,Any}("kind" => mode)
    path = joinpath(outdir, drug * (mode == "full" ? "" : "_" * mode) * ".jld2")
    train = units
    if mode == "late"
        train = map(units) do u
            tr_c, _ = split_late(u.treated; frac = LATE_FRAC)
            tr_k, _ = split_late(u.control; frac = LATE_FRAC)
            cutoffs["$(u.model)|treated"] = last(tr_c.t)
            cutoffs["$(u.model)|control"] = last(tr_k.t)
            PDXUnit(u.model, u.tumour_type, tr_k, tr_c)
        end
        split["late_frac"] = LATE_FRAC
        stamp("late split: $(sum(length(u.treated.t) for u in train)) of $(sum(length(u.treated.t) for u in units)) treated points kept")
    elseif mode == "holdout"
        train_ids, test_ids = split_models(units; frac = MODEL_FRAC, seed = SPLIT_SEED)
        keep = Set(train_ids)
        train = [u for u in units if u.model in keep]
        split["model_frac"] = MODEL_FRAC; split["seed"] = SPLIT_SEED
        stamp("holdout split: train $(length(train)) PDX, held out $(length(test_ids)) PDX")
    end

    kw = (; drug, nsamples = ndraws, nadapt, target_accept = target, latent_w0 = latent, split, cutoffs, seed)
    chain_only = haskey(opts, "chain-only")
    if chain_only                                  # child process: one chain, save it, done
        k = parse(Int, opts["chain-only"])
        chain, _ = M.fit_drug(train, doses, pk, cref; kw..., nchains = 1, chain_only = true,
                              rng = MersenneTwister(1000 * seed + k), log_every = 250)
        save_chain_parts(opts["chain-out"], chain)
        stamp("chain $k saved")
        return nothing
    end
    use_procs = nchains > 1 && Threads.nthreads() > 1 && !haskey(opts, "noprocs")
    if use_procs
        childargs = String[drug, mode, "1", string(nadapt), string(ndraws), "--seed=$seed", "--target=$target"]
        latent || push!(childargs, "--nolatent")
        haskey(opts, "units") && push!(childargs, "--units=" * opts["units"])
        stamp("sampling $nchains chains in separate processes (max $(Threads.nthreads()) at a time)")
        chain, wall = run_chain_procs(joinpath(@__DIR__, "fit_drug.jl"), childargs, nchains;
                                      maxprocs = Threads.nthreads(), project = Base.active_project())
        _, pd = M.fit_drug(train, doses, pk, cref; kw..., precomputed_chain = chain, precomputed_runtime = wall)
    else
        _, pd = M.fit_drug(train, doses, pk, cref; kw..., nchains, rng = MersenneTwister(seed), log_every = 250)
    end
    dg = pd.diagnostics
    niter = nchains * (ndraws + nadapt)
    stamp(@sprintf("done: %.1f min wall (%d iterations, %.1f s per 100 iterations, parallel=%s)",
                   dg["runtime_s"] / 60, niter, 100 * dg["runtime_s"] / niter, dg["settings"]["parallel"]))
    stamp(@sprintf("max R-hat %.4f   min bulk ESS %.0f   divergences %d   mean tree depth %.2f   depth-limit hits %d   step size %.4f",
                   dg["max_rhat"], dg["min_ess_bulk"], dg["n_divergences"], dg["mean_tree_depth"],
                   dg["max_tree_depth_hits"], dg["mean_step_size"]))
    q(x) = (quantile(x, 0.05), median(x), quantile(x, 0.95))
    for (nm, x) in (("μ_λ0", pd.μ_λ0), ("ω_λ0", pd.ω_λ0), ("μ_k2", pd.μ_k2), ("ω_k2", pd.ω_k2),
                    ("logλ1", pd.logλ1), ("logk1", pd.logk1), ("σ", pd.σ))
        lo, md, hi = q(x)
        stamp(@sprintf("  %-6s %8.3f  [%8.3f, %8.3f]", nm, md, lo, hi))
    end
    M.save_fit(path, pd)
    stamp("saved $path")
    return pd
end

main(ARGS)
