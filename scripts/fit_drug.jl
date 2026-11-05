# Fit the hierarchical Simeoni model for one drug and save it for validation.
#
#   julia --project=. -t 4 scripts/fit_drug.jl <drug> <full|late|holdout> [nchains warmup draws] [--seed=1] [--units=N]
#                                                [--target=0.8] [--window=42] [--nu=5] [--baseline-sd=0.1]
#                                                [--cthreads=2] [--mapstarts=5] [--mapiters=1000] [--nolatent] [--noprocs] [--out=fits] [--model=v1|v2] [--waning]
#
#   full     all PDX models, all points                       -> fits/<drug>.jld2
#   late     every curve cut with split_late(frac=0.3)         -> fits/<drug>_late.jld2
#            (cutoffs saved as "<model>|treated" / "<model>|control" = last training day)
#   holdout  PDX models left out by split_models(frac=0.2, seed=1) -> fits/<drug>_holdout.jld2
#
# --window=42 (default) is the prespecified fit window: both arms are truncated to day <= 42 BEFORE any late/holdout
# split (0 = full follow-up); the late-split cutoffs are therefore taken on the windowed curves.
# Residuals are Student-t with --nu=5 (Inf = Normal); baselines are latent with fixed sd --baseline-sd=0.1.
# A multi-start MAP (--mapstarts, default 5) runs once in the parent; every chain starts with small jitter around
# the best mode; all MAP log-densities are logged and saved in diagnostics["map_lps"].
# --cthreads=T gives each chain process T threads (units are threaded inside a chain); at most (-t N)/T run at once.
#
# Production design (2 fits per drug, 6 in total):
#   primary  fit on the frozen stratified 80% of PDX models (frozen_pdx_split: union over drugs, seed 1, stratified by tumour
#            type, so the SAME PDX identities are held out for every drug); window 42 d     -> fits/<drug>_primary.jld2
#   late21   same 80%, both arms truncated at day 21 (init sees no later data)               -> fits/<drug>_late21.jld2
# Saved split Dict: kind, model_frac, seed, stratified, universe, holdout_models (this drug's held-out PDX), late_window_day.
#
# The phase-2 modes full / late / holdout remain available.
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
    length(pos) >= 2 || error("usage: fit_drug.jl <drug> <full|late|holdout> [nchains warmup draws] [--seed=1] [--units=N] [--target=0.8] [--window=42] [--nu=5] [--baseline-sd=0.1]
#                                                [--cthreads=2] [--mapstarts=5] [--mapiters=1000] [--nolatent] [--noprocs] [--out=fits] [--model=v1|v2] [--waning]")
    return pos, opts
end

getopt_(o, k, d) = get(o, k, d)
const TAG = Ref("")
stamp(msg) = println("[", Dates.format(now(), "HH:MM:SS"), "]", TAG[], " ", msg)

function main(args)
    pos, opts = parse_cli(args)
    drug, mode = pos[1], pos[2]
    mode in ("full", "late", "holdout", "primary", "late21") || error("mode must be primary, late21 (production design) or full, late, holdout (phase-2 splits)")
    nchains = length(pos) >= 3 ? parse(Int, pos[3]) : 4
    nadapt  = length(pos) >= 4 ? parse(Int, pos[4]) : 500
    ndraws  = length(pos) >= 5 ? parse(Int, pos[5]) : 500
    seed    = parse(Int, get(opts, "seed", "1"))
    target  = parse(Float64, get(opts, "target", "0.8"))
    latent  = !haskey(opts, "nolatent")
    model   = getopt_(opts, "model", "v1"); model in ("v1", "v2") || error("--model must be v1 or v2")
    waning  = haskey(opts, "waning") && opts["waning"] != "false"
    mode_gap = parse(Float64, get(opts, "mode-gap", model == "v2" ? "30" : "Inf"))   # v2 production default 30
    outdir  = abspath(get(opts, "out", joinpath(ROOT, "fits")))
    window  = parse(Float64, get(opts, "window", mode == "late21" ? "21" : "42"))
    nu      = parse(Float64, get(opts, "nu", "5"))
    bsd     = parse(Float64, get(opts, "baseline-sd", "0.1"))
    cthreads = parse(Int, get(opts, "cthreads", "1"))
    mapstarts = parse(Int, get(opts, "mapstarts", "5"))
    mapiters = parse(Int, get(opts, "mapiters", "1000"))

    haskey(opts, "chain-only") && (TAG[] = "[chain " * opts["chain-only"] * "]")
    stamp("fit_drug $drug $mode chains=$nchains warmup=$nadapt draws=$ndraws seed=$seed latent_w0=$latent threads=$(Threads.nthreads())")
    units = drug_dataset(drug)
    if haskey(opts, "units")                       # subsample for smoke tests, keeps split logic on the subsample
        n = parse(Int, opts["units"])
        units = units[sort(randperm(MersenneTwister(0), length(units))[1:min(n, length(units))])]
    end
    units = M.window_units(units, window)          # fit window first, then any split
    stamp("fit window: " * (window > 0 ? "day <= $(window)" : "full follow-up") * "; Student-t nu=$(nu), baseline sd=$(bsd)")
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
    elseif mode in ("primary", "late21")
        # Production design: fit on the frozen stratified 80% of PDX models (same identities for every drug), whole-PDX
        # holdout of the other 20%. `late21` additionally truncates both arms at the late window (default 21 d) so the
        # fit and its start never see later data; the 22-42 d forecast is checked by the validation side.
        train_all, test_all, uinfo = frozen_pdx_split(; frac = MODEL_FRAC, seed = SPLIT_SEED)
        keep = Set(train_all)
        train = [u for u in units if u.model in keep]
        hold = sort([u.model for u in units if !(u.model in keep)])
        split["model_frac"] = MODEL_FRAC; split["seed"] = SPLIT_SEED; split["stratified"] = "tumour_type"
        split["universe"] = "union of paired PDX models over " * join(uinfo.drugs, ",") * " ($(uinfo.n_universe))"
        split["holdout_models"] = hold
        if mode == "late21"
            split["late_window_day"] = window
            for u in train                          # last training day per curve (<= window), for verification
                cutoffs["$(u.model)|treated"] = maximum(u.treated.t)
                cutoffs["$(u.model)|control"] = maximum(u.control.t)
            end
        end
        stamp("$mode design: train $(length(train)) PDX, whole-PDX holdout $(length(hold)) (frozen split over $(uinfo.n_universe) PDX of $(join(uinfo.drugs, ",")))")
    elseif mode == "holdout"
        train_ids, test_ids = split_models(units; frac = MODEL_FRAC, seed = SPLIT_SEED)
        keep = Set(train_ids)
        train = [u for u in units if u.model in keep]
        split["model_frac"] = MODEL_FRAC; split["seed"] = SPLIT_SEED
        stamp("holdout split: train $(length(train)) PDX, held out $(length(test_ids)) PDX")
    end

    kw = (; drug, nsamples = ndraws, nadapt, target_accept = target, latent_w0 = latent, split, cutoffs, seed,
          model = Symbol(model), waning, mode_gap,
          window_days = window, nu, baseline_sd = bsd)
    chain_only = haskey(opts, "chain-only")
    if chain_only                                  # child process: one chain from the shared start, save it, done
        k = parse(Int, opts["chain-only"])
        init, k2c = haskey(opts, "init-file") ? load_init(opts["init-file"]; chain = k) : (nothing, nothing)
        chain, _ = M.fit_drug(train, doses, pk, cref; kw..., nchains = 1, chain_only = true, init, k2_centered = k2c,
                              map_starts = init === nothing ? mapstarts : 0, map_maxiters = mapiters,
                              rng = MersenneTwister(1000 * seed + k), log_every = 25, logger = stamp)
        save_chain_parts(opts["chain-out"], chain)
        stamp("chain $k saved")
        return nothing
    end
    use_procs = nchains > 1 && Threads.nthreads() > 1 && !haskey(opts, "noprocs")
    if use_procs
        init_file = ""
        map_lps = nothing
        k2c = nothing
        mode_idx = nothing; mode_lps = nothing; excluded = nothing
        if mapstarts > 0
            stamp("multi-start MAP: $mapstarts starts, <= $mapiters iterations each")
            mm = M.map_modes(train, doses, pk, cref; window_days = window, nu, baseline_sd = bsd,
                             latent_w0 = latent, nstarts = mapstarts, maxiters = mapiters, model_v2 = model == "v2", waning, mode_gap,
                             rng = MersenneTwister(seed), logger = stamp)
            map_lps = mm.all_lps; k2c = mm.k2_centered; excluded = mm.excluded
            mode_idx = [mod1(k, length(mm.starts)) for k in 1:nchains]; mode_lps = [m.lp for m in mm.starts]
            stamp("MAP log-densities of all starts: " * join(round.(map_lps; digits = 1), ", ") *
                  "; $(length(mm.starts)) distinct mode(s), best " * string(round(maximum(map_lps); digits = 1)) *
                  "; chain k starts at mode mod1(k, $(length(mm.starts)))")
            init_file = joinpath(mktempdir(), "init.jld2")
            save_init(init_file, [m.nt for m in mm.starts], k2c)
        end
        childargs = String[drug, mode, "1", string(nadapt), string(ndraws), "--seed=$seed", "--target=$target",
                           "--window=$window", "--nu=$nu", "--baseline-sd=$bsd", "--mapstarts=$mapstarts"]
        isempty(init_file) || push!(childargs, "--init-file=" * init_file)
        latent || push!(childargs, "--nolatent")
        push!(childargs, "--model=$model", "--mode-gap=$mode_gap"); waning && push!(childargs, "--waning")
        haskey(opts, "units") && push!(childargs, "--units=" * opts["units"])
        maxprocs = max(1, Threads.nthreads() ÷ cthreads)
        stamp("sampling $nchains chains in separate processes ($cthreads thread(s) each, max $maxprocs at a time)")
        chain, wall = run_chain_procs(joinpath(@__DIR__, "fit_drug.jl"), childargs, nchains;
                                      maxprocs, threads = cthreads, project = Base.active_project())
        worker_info = (; worker_threads = cthreads, chain_seeds = [1000 * seed + k for k in 1:nchains],
                       worker_processes = true)
        _, pd = M.fit_drug(train, doses, pk, cref; kw..., precomputed_chain = chain, precomputed_runtime = wall,
                           map_lps, worker_info, k2_centered = k2c,
                           chain_start_modes = mode_idx, distinct_mode_lps = mode_lps, excluded_modes = excluded)
    else
        _, pd = M.fit_drug(train, doses, pk, cref; kw..., nchains, rng = MersenneTwister(seed), log_every = 25, logger = stamp,
                           map_starts = mapstarts, map_maxiters = mapiters)
    end
    dg = pd.diagnostics
    niter = nchains * (ndraws + nadapt)
    stamp(@sprintf("done: %.1f min wall (%d iterations, %.1f s per 100 iterations, parallel=%s)",
                   dg["runtime_s"] / 60, niter, 100 * dg["runtime_s"] / niter, dg["settings"]["parallel"]))
    stamp(@sprintf("mean n_steps %.1f   MAP lps %s", dg["mean_n_steps"], string(round.(dg["map_lps"]; digits = 1))))
    stamp(@sprintf("max R-hat %.4f   min bulk ESS %.0f   divergences %d   mean tree depth %.2f   depth-limit hits %d   step size %.4f",
                   dg["max_rhat"], dg["min_ess_bulk"], dg["n_divergences"], dg["mean_tree_depth"],
                   dg["max_tree_depth_hits"], dg["mean_step_size"]))
    q(x) = (quantile(x, 0.05), median(x), quantile(x, 0.95))
    for (nm, x) in (("μ_λ0", pd.μ_λ0), ("ω_λ0", pd.ω_λ0), ("μ_k2", pd.μ_k2), ("ω_k2", pd.ω_k2),
                    ("logλ1", pd.logλ1), ("Vtr=λ1/λ0", exp.(pd.logλ1 .- pd.μ_λ0)), ("logk1", pd.logk1), ("σ", pd.σ))
        lo, md, hi = q(x)
        stamp(@sprintf("  %-6s %8.3f  [%8.3f, %8.3f]", nm, md, lo, hi))
    end
    M.save_fit(path, pd)
    stamp("saved $path")
    return pd
end

main(ARGS)
