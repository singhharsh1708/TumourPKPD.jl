# Staged fit (init -> warmup -> sample -> merge), built for CI jobs of a few hours each.
#
#   julia --project=. -t 4 scripts/fit_stage.jl init   <drug> <design> --out=DIR
#   julia --project=. -t 4 scripts/fit_stage.jl warmup <drug> <design> --chain=K --dir=DIR [--nadapt=500] [--seglen=L --seg=S]
#   julia --project=. -t 4 scripts/fit_stage.jl sample <drug> <design> --chain=K --part=P --dir=DIR [--draws=250] [--burnin=25]
#   julia --project=. -t 4 scripts/fit_stage.jl merge  <drug> <design> --dir=DIR --out=fits/
#
# <design> is `primary` (frozen stratified 80% of PDX, days <= 42) or `late21` (same PDX, days <= 21).
#
# Resumable warmup (optional): with --seglen=L the adaptation runs in segments of L iterations: `--seg=S` runs iterations
# (S-1)L+1 .. min(nadapt, S L), reading DIR/warmup_K_(S-1).jld2 (adaptor + position + RNG state) and writing DIR/warmup_K_S.jld2;
# the segment that reaches nadapt also writes DIR/warm_K.jld2. Without --seglen one job does all of it.
#
# Artifacts (parameters and sampler state only, never data or closures):
#   DIR/init.jld2        MAP modes (sampling coordinates), k2-centring mask with model ids, split/cutoff metadata, model settings
#   DIR/warm_K.jld2      warmed state of chain K: unconstrained position, adapted step size, adapted diagonal metric, seed, stats
#   DIR/chain_K_P.jld2   `draws` post-warmup draws of (K,P), no adaptation (seed from (K,P)); constrained values + sampler stats
#   <out>/<drug>_<design>.jld2 and <out>/<drug>_<design>_diagnostics.md   merged PosteriorDraws (a standard `save_fit` file)
#
# Every stage rebuilds the target from the cleaned data cache and the init artifact, so the model settings (priors, nu,
# baseline sd, window, mask, ...) come from init.jld2, not from flags. Each (K,P) is one chain in the diagnostics; the
# report also gives R-hat per warmup chain K (its parts concatenated) and the cross-mode check. Chains K start from the
# DIFFERENT distinct MAP modes (cycled) when init found more than one.
#
# Optional flags: --seed=1 --units=N (smoke tests: subsample N paired PDX) --mapstarts=5 --mapiters=1000 --target=0.8
#                 --maxdepth=10 --nu=5 --baseline-sd=0.1 --window=<days> --burnin=25
#                 init: --model=v1|v2 (default v1; v2 = correlated (log λ0, log k2) + per-PDX transition volume) --waning (pooled efficacy decay)
#                       --mode-gap=G (default Inf: every converged distinct MAP mode starts chains; G=30 drops converged modes > 30 log-density units below the best)
#                 warmup: --init-buffer=75 --term-buffer=50 --window-size=25 (Stan adaptation windows)
# stdout is flushed every second; progress lines every 25 iterations.
using TumourPKPD, Random, Statistics, Printf, Dates, JLD2, MCMCChains
using TumourPKPD: OneCmt
const M = TumourPKPD
isdefined(M, :stage_warmup) || Base.include(M, joinpath(@__DIR__, "..", "src", "hierarchical.jl"))
const _flusher = Timer(_ -> flush(stdout), 1.0; interval = 1.0)
stamp(msg) = println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)

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
    length(pos) >= 3 || error("usage: fit_stage.jl <init|warmup|sample|merge> <drug> <design> [--flags]")
    return pos, opts
end
getopt(opts, k, d) = get(opts, k, d)
need(opts, k) = haskey(opts, k) ? opts[k] : error("missing --$k")

# ----------------------------------------------------------------------------------------------------------------------
# Shared context
# ----------------------------------------------------------------------------------------------------------------------

priors_dict(p::M.HierPriors) = Dict{String,Any}(string(k) => getfield(p, k) for k in fieldnames(M.HierPriors))
priors_from(d) = M.HierPriors(; (Symbol(k) => (v isa Tuple ? Tuple(v) : v) for (k, v) in d)...)

function pk_and_doses(drug)
    doses, pkp = regimen(drug)
    pk = OneCmt(CL = pkp.CL, V = pkp.V, ka = pkp.ka, F = pkp.F)
    return pk, doses, M.reference_cref(pk, doses)
end

function target_fp(init, du, models, pri, pk, doses, cref, mask)
    txt = M.fingerprint_text(; drug = init["drug"], design = init["design"], window = init["window"], models, split = du.split,
                             priors = pri, nu = init["nu"], baseline_sd = init["baseline_sd"], lod = init["lod"], doses, pk, cref,
                             k2_centered = mask, centered_lam0 = init["centered_lam0"], latent_w0 = init["latent_w0"],
                             reltol = init["reltol"], abstol = init["abstol"], init_id = init["init_id"],
                             model = get(init, "model", "v1"), waning = get(init, "waning", false),
                             mode_gap = get(init, "mode_gap", Inf))
    return M.target_fingerprint(txt)
end

"Rebuild units and the sampling target from init.jld2 (+ data). Returns a context NamedTuple."
function load_context(dir, drug, design; start_mode::Union{Nothing,Int} = nothing)
    init = JLD2.load(joinpath(dir, "init.jld2"))
    init["drug"] == drug && init["design"] == design || error("init.jld2 is for $(init["drug"]) $(init["design"]), not $drug $design")
    du = M.design_units(drug, design; frac = init["model_frac"], seed = init["split_seed"], window = init["window"],
                        subsample = init["subsample"])
    models = [u.model for u in du.train]
    # the prepared (valid-baseline) unit order must be exactly the one init used
    prepared = [u.model for u in first(M.prepare_units(du.train; warn = false))]
    prepared == init["train_models"] || error("data cache differs from the one used by init (train_models mismatch)")
    pk, doses, cref = pk_and_doses(drug)
    pri = priors_from(init["priors"])
    mask = collect(Bool, init["k2_centered"])
    # the target rebuilt now (data, regimen, split, priors, settings, mask, init identity) must be the one init was made for
    fp = target_fp(init, du, prepared, pri, pk, doses, cref, mask)
    M.verify_fingerprint(init["fingerprint"], fp; what = "the target rebuilt from the current data/settings", against = "init.jld2")
    nt = start_mode === nothing ? init["modes"][1].nt : init["modes"][start_mode].nt
    tgt = M.stage_target(du.train, doses, pk, cref; start = nt, lod = init["lod"], priors = pri, latent_w0 = init["latent_w0"],
                         reltol = init["reltol"], abstol = init["abstol"], nu = init["nu"], baseline_sd = init["baseline_sd"],
                         centered_lam0 = init["centered_lam0"], k2_centered = mask, window_days = 0,
                         model_v2 = get(init, "model", "v1") == "v2", waning = get(init, "waning", false))
    return (; init, du, pk, doses, cref, pri, mask, tgt, models = prepared, fingerprint = init["fingerprint"])
end

# ----------------------------------------------------------------------------------------------------------------------
# init
# ----------------------------------------------------------------------------------------------------------------------

function do_init(drug, design, opts)
    out = need(opts, "out"); mkpath(out)
    seed = parse(Int, getopt(opts, "seed", "1"))
    sub = parse(Int, getopt(opts, "units", "0"))
    nu = parse(Float64, getopt(opts, "nu", "5")); bsd = parse(Float64, getopt(opts, "baseline-sd", "0.1"))
    nstarts = parse(Int, getopt(opts, "mapstarts", "5")); mapiters = parse(Int, getopt(opts, "mapiters", "1000"))
    window = haskey(opts, "window") ? parse(Float64, opts["window"]) : nothing
    model = getopt(opts, "model", "v1"); model in ("v1", "v2") || error("--model must be v1 or v2")
    waning = haskey(opts, "waning") && opts["waning"] != "false"
    # v2 production default 30 (a converged mode ~90 log-density units below the best also has a data log-likelihood ~72 lower:
    # negligible posterior mass, and chains started there do not mix back); v1 default Inf keeps every converged mode.
    mode_gap = parse(Float64, getopt(opts, "mode-gap", model == "v2" ? "30" : "Inf"))      # converged MAP modes further than this below the best are not used as chain starts (recorded as excluded)
    du = M.design_units(drug, design; frac = 0.2, seed = 1, window, subsample = sub)
    prepared, dropped = M.prepare_units(du.train; warn = false)
    stamp("init $drug $design: $(length(prepared)) training PDX (window <= $(du.window) d), holdout $(length(du.split["holdout_models"])), dropped $(length(dropped))")
    pk, doses, cref = pk_and_doses(drug)
    pri = M.default_priors()
    stamp("model $model" * (waning ? " with waning efficacy" : ""))
    mm = M.map_modes(du.train, doses, pk, cref; window_days = 0, nu, baseline_sd = bsd, nstarts, maxiters = mapiters,
                     priors = pri, rng = MersenneTwister(seed), logger = stamp, model_v2 = model == "v2", waning, mode_gap)
    stamp("MAP log-densities of all $(length(mm.all_lps)) starts: " * join(round.(mm.all_lps; digits = 1), ", ") *
          "; $(length(mm.starts)) distinct mode(s): " * join(round.([m.lp for m in mm.starts]; digits = 1), ", "))
    modes_nt = [(; nt = m.nt, lp = m.lp) for m in mm.starts]
    init_id = M.target_fingerprint(string(now()) * "|" * join([string(m.lp) * ":" * join(string.(collect(Iterators.flatten(
        [v isa Real ? [v] : collect(v) for v in values(m.nt)]))), ",") for m in mm.starts], ";"))
    excluded = [Dict{String,Any}("lp" => e.lp, "converged" => e.converged, "dist" => e.dist, "gap" => e.gap, "k1" => e.k1, "reason" => e.reason) for e in mm.excluded]
    meta = Dict{String,Any}("drug" => drug, "design" => design, "window" => du.window, "nu" => nu, "baseline_sd" => bsd, "lod" => detection_limit(),
                            "reltol" => 1e-5, "abstol" => 1e-5, "latent_w0" => true, "centered_lam0" => true, "init_id" => init_id)
    fp = M.target_fingerprint(M.fingerprint_text(; drug, design, window = du.window, models = [u.model for u in prepared], split = du.split,
                                                 priors = pri, nu, baseline_sd = bsd, lod = detection_limit(), doses, pk, cref,
                                                 k2_centered = mm.k2_centered, centered_lam0 = true, latent_w0 = true, reltol = 1e-5,
                                                 abstol = 1e-5, init_id, model, waning, mode_gap))
    jldsave(joinpath(out, "init.jld2");
            drug, design, window = du.window, subsample = sub, model_frac = 0.2, split_seed = 1, seed,
            nu, baseline_sd = bsd, lod = detection_limit(), reltol = 1e-5, abstol = 1e-5, latent_w0 = true, centered_lam0 = true,
            priors = priors_dict(pri), train_models = [u.model for u in prepared], dropped = dropped,
            k2_centered = mm.k2_centered, k2_centered_models = [u.model for (u, c) in zip(prepared, mm.k2_centered) if c],
            modes = modes_nt, all_lps = mm.all_lps, excluded_modes = excluded,
            split = du.split, cutoffs = du.cutoffs, created = string(now()), init_id, fingerprint = fp, model, waning, mode_gap)
    stamp("wrote $(joinpath(out, "init.jld2")) (k2 centred for $(count(mm.k2_centered)) of $(length(mm.k2_centered)) units)")
end

# ----------------------------------------------------------------------------------------------------------------------
# warmup
# ----------------------------------------------------------------------------------------------------------------------

function do_warmup(drug, design, opts)
    dir = need(opts, "dir"); K = parse(Int, need(opts, "chain"))
    nadapt = parse(Int, getopt(opts, "nadapt", "500"))
    seglen = parse(Int, getopt(opts, "seglen", "0")); seg = parse(Int, getopt(opts, "seg", "1"))
    target = parse(Float64, getopt(opts, "target", "0.8")); maxdepth = parse(Int, getopt(opts, "maxdepth", "10"))
    init0 = JLD2.load(joinpath(dir, "init.jld2"))
    nm = length(init0["modes"]); mode = mod1(K, nm)
    ctx = load_context(dir, drug, design; start_mode = mode)
    seed = M.stage_seed(init0["seed"], K, 0)
    segmented = seglen > 0
    prev = nothing
    if segmented && seg > 1
        prev = JLD2.load(joinpath(dir, "warmup_$(K)_$(seg - 1).jld2"))
        M.verify_fingerprint(ctx.fingerprint, get(prev, "fingerprint", nothing); what = "previous warmup segment warmup_$(K)_$(seg - 1).jld2")
        prev["mode"] == mode && prev["nadapt"] == nadapt && prev["seglen"] == seglen || error("previous warmup segment is inconsistent with the arguments")
        prev["i"] < nadapt || error("warmup of chain $K is already complete")
    end
    rng = prev === nothing ? MersenneTwister(seed) : prev["rng"]
    i0 = prev === nothing ? 0 : prev["i"]
    upto = segmented ? min(nadapt, seg * seglen) : nadapt
    stamp("warmup chain $K" * (segmented ? " segment $seg (iterations $(i0 + 1)-$upto of $nadapt)" : "") *
          ": start mode $mode of $nm (lp $(round(init0["modes"][mode].lp; digits = 1))), dim $(length(ctx.tgt.θ0)), seed $seed, $(Threads.nthreads()) threads")
    θ0 = M.unconstrain(ctx.tgt.layout, M.jitter_init(init0["modes"][mode].nt, MersenneTwister(seed), 0.02))
    resume = prev === nothing ? nothing : (; θ = prev["theta"], ϵ = prev["epsilon"], Minv = prev["Minv"], adaptor = prev["adaptor"], i = prev["i"])
    w = M.stage_warmup(ctx.tgt, θ0, rng; nadapt, δ = target, max_depth = maxdepth, label = " $K", resume, upto,
                       init_buffer = parse(Int, getopt(opts, "init-buffer", "75")), term_buffer = parse(Int, getopt(opts, "term-buffer", "50")),
                       window_size = parse(Int, getopt(opts, "window-size", "25")))
    stats = prev === nothing ? w.stats : vcat(prev["warm_stats"], w.stats)
    time_s = (prev === nothing ? 0.0 : prev["warm_time_s"]) + w.time
    common = (; chain = K, mode, mode_lp = init0["modes"][mode].lp, seed, theta = w.θ, epsilon = w.ϵ, Minv = w.Minv, nadapt, target,
              max_depth = maxdepth, warm_stats = stats, stat_names = collect(String.(M.STAT_NAMES)), warm_time_s = time_s,
              names = collect(String.(ctx.tgt.layout.names)), k2_centered = ctx.mask, models = ctx.models, threads = Threads.nthreads(),
              fingerprint = ctx.fingerprint)
    if segmented && !w.done        # resumable segment: parameters + adaptor state (no data), then stop
        jldsave(joinpath(dir, "warmup_$(K)_$(seg).jld2"); common..., seglen, seg, i = w.i, adaptor = w.adaptor, rng)
        stamp("warmup chain $K segment $seg: $(w.i)/$nadapt iterations done, step size $(round(w.ϵ; sigdigits = 4)); wrote $(joinpath(dir, "warmup_$(K)_$(seg).jld2")) (run segment $(seg + 1) next)")
        return
    end
    st = stats
    stamp(@sprintf("warmup chain %d done in %.1f min (all segments): step size %.4f, mean n_steps %.1f (last 100: %.1f), divergences %d, metric range %.3g..%.3g",
                   K, time_s / 60, w.ϵ, mean(st[:, 1]), mean(st[max(1, end-99):end, 1]), Int(sum(st[:, 5])), minimum(w.Minv), maximum(w.Minv)))
    jldsave(joinpath(dir, "warm_$(K).jld2"); common..., lp_final = ctx.tgt.ℓπ(w.θ))
    stamp("wrote $(joinpath(dir, "warm_$(K).jld2"))")
end

# ----------------------------------------------------------------------------------------------------------------------
# sample
# ----------------------------------------------------------------------------------------------------------------------

function do_sample(drug, design, opts)
    dir = need(opts, "dir"); K = parse(Int, need(opts, "chain")); P = parse(Int, need(opts, "part"))
    ndraws = parse(Int, getopt(opts, "draws", "250")); burnin = parse(Int, getopt(opts, "burnin", "25"))
    maxdepth = parse(Int, getopt(opts, "maxdepth", "10"))
    w = JLD2.load(joinpath(dir, "warm_$(K).jld2"))
    ctx = load_context(dir, drug, design; start_mode = w["mode"])
    M.verify_fingerprint(ctx.fingerprint, get(w, "fingerprint", nothing); what = "warm_$(K).jld2")
    collect(String.(ctx.tgt.layout.names)) == w["names"] || error("parameter layout differs from the warmup artifact")
    ctx.mask == w["k2_centered"] && ctx.models == w["models"] || error("k2 mask / model ids differ from the warmup artifact")
    seed = M.stage_seed(ctx.init["seed"], K, P)
    stamp("sample chain $K part $P: $ndraws draws (+$burnin burn-in, no adaptation), step size $(round(w["epsilon"]; sigdigits = 4)), seed $seed, $(Threads.nthreads()) threads")
    # the warm state: same log density at the saved position as at the end of warmup
    lp = ctx.tgt.ℓπ(w["theta"])
    abs(lp - w["lp_final"]) <= 1e-8 * max(1.0, abs(lp)) || error("log density at the warm position differs from the warmup stage ($lp vs $(w["lp_final"]))")
    s = M.stage_sample(ctx.tgt, w["theta"], w["epsilon"], w["Minv"], MersenneTwister(seed); ndraws, burnin, max_depth = maxdepth, label = " $K.$P")
    all(s.stats[:, 3] .== w["epsilon"]) || error("step size changed during sampling (adaptation must be off)")
    st = s.stats
    stamp(@sprintf("sample chain %d part %d done in %.1f min: mean n_steps %.1f, tree depth %.2f, divergences %d, acceptance %.3f",
                   K, P, s.time / 60, mean(st[:, 1]), mean(st[:, 2]), Int(sum(st[:, 5])), mean(st[:, 4])))
    jldsave(joinpath(dir, "chain_$(K)_$(P).jld2"); chain = K, part = P, seed, mode = w["mode"], epsilon = w["epsilon"],
            draws = M.constrain_draws(ctx.tgt.layout, s.θ), stats = st, stat_names = collect(String.(M.STAT_NAMES)),
            names = collect(String.(ctx.tgt.layout.names)), k2_centered = ctx.mask, models = ctx.models, burnin, ndraws,
            sample_time_s = s.time, final_theta = s.final, threads = Threads.nthreads(), fingerprint = ctx.fingerprint)
    stamp("wrote $(joinpath(dir, "chain_$(K)_$(P).jld2"))")
end

# ----------------------------------------------------------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------------------------------------------------------

function rhat_ess(ch)
    ss = MCMCChains.summarystats(ch)
    prm = [n for n in ss.nt.parameters]
    rh = collect(ss.nt.rhat); es = collect(ss.nt.ess_bulk)
    return prm, rh, es
end

function do_merge(drug, design, opts)
    dir = need(opts, "dir"); out = getopt(opts, "out", "fits"); mkpath(out)
    ctx = load_context(dir, drug, design)
    init = ctx.init
    files = sort(filter(f -> occursin(r"^chain_\d+_\d+\.jld2$", f), readdir(dir)))
    isempty(files) && error("no chain_K_P.jld2 files in $dir")
    parts = [JLD2.load(joinpath(dir, f)) for f in files]
    sort!(parts; by = p -> (p["chain"], p["part"]))
    Ks = sort(unique(p["chain"] for p in parts))
    for p in parts
        M.verify_fingerprint(ctx.fingerprint, get(p, "fingerprint", nothing); what = "chain_$(p["chain"])_$(p["part"]).jld2")
        collect(String.(ctx.tgt.layout.names)) == p["names"] || error("layout mismatch in chain $(p["chain"]) part $(p["part"])")
        ctx.mask == p["k2_centered"] && ctx.models == p["models"] || error("k2 mask / model ids differ in chain $(p["chain"]) part $(p["part"])")
        p["draws"] isa AbstractMatrix || error("bad draws")
    end
    nd = minimum(size(p["draws"], 1) for p in parts)
    chains = [M.stage_chain(ctx.tgt.layout, p["draws"][1:nd, :], p["stats"][1:nd, :]) for p in parts]
    chain = chainscat(chains...)
    labels = ["$(p["chain"]).$(p["part"])" for p in parts]
    warm = Dict(K => JLD2.load(joinpath(dir, "warm_$(K).jld2")) for K in Ks)
    for K in Ks
        M.verify_fingerprint(ctx.fingerprint, get(warm[K], "fingerprint", nothing); what = "warm_$(K).jld2")
    end
    nmodes = length(init["modes"])
    mode_of = [mod1(p["chain"], nmodes) for p in parts]
    wall_core = sum(p["sample_time_s"] for p in parts) + sum(w["warm_time_s"] for w in values(warm))
    stamp("merging $(length(parts)) chains (K = $(Ks), $(nd) draws each) for $drug $design")

    pk, doses, cref = ctx.pk, ctx.doses, ctx.cref
    _, pd = M.fit_drug(ctx.du.train, doses, pk, cref; drug = drug, nchains = length(parts), nsamples = nd,
                       nadapt = first(values(warm))["nadapt"], target_accept = first(values(warm))["target"],
                       max_depth = first(values(warm))["max_depth"], lod = init["lod"], priors = ctx.pri,
                       latent_w0 = init["latent_w0"], nu = init["nu"], baseline_sd = init["baseline_sd"],
                       centered_lam0 = init["centered_lam0"], reltol = init["reltol"], abstol = init["abstol"],
                       window_days = 0, precomputed_chain = chain, precomputed_runtime = wall_core, k2_centered = ctx.mask,
                       model = Symbol(get(init, "model", "v1")), waning = get(init, "waning", false),
                       split = init["split"], cutoffs = init["cutoffs"], seed = init["seed"], map_lps = init["all_lps"],
                       worker_info = (; staged = true, threads_per_job = first(parts)["threads"],
                                      chain_seeds = [p["seed"] for p in parts], warmup_seeds = [warm[K]["seed"] for K in Ks]))
    dg = pd.diagnostics
    dg["fit_window_day"] = float(init["window"])
    dg["stage"] = Dict{String,Any}("warmup_chains" => Ks, "parts" => sort(unique(p["part"] for p in parts)), "draws_per_part" => nd,
                                   "burnin" => first(parts)["burnin"], "nadapt" => first(values(warm))["nadapt"],
                                   "chain_labels" => labels, "warmup_chain_of_chain" => [p["chain"] for p in parts],
                                   "core_seconds" => wall_core,
                                   "warm_step_size" => Dict(K => warm[K]["epsilon"] for K in Ks),
                                   "warm_mean_n_steps" => Dict(K => mean(warm[K]["warm_stats"][:, 1]) for K in Ks))
    dg["chain_start_modes"] = mode_of
    dg["mode_lps"] = [m.lp for m in init["modes"]]
    dg["n_distinct_modes"] = nmodes
    dg["map_lps"] = collect(Float64, init["all_lps"])

    # ---- diagnostics: all (K,P) chains, per warmup chain K (parts concatenated), cross-mode
    prm_all, rh_all, es_all = rhat_ess(chain)
    pop = get(init, "model", "v1") == "v2" ? [:μ_λ0, :ω_λ0, :μ_k2, :ω_k2, :ρ_u, :μ_Vtr, :ω_Vtr, :logk1, :σ] :
                                             [:μ_λ0, :ω_λ0, :μ_k2, :ω_k2, :logVtr, :logk1, :σ]
    get(init, "waning", false) && push!(pop, :logρ_w)
    keep = [Symbol(n) for n in MCMCChains.names(chain, :parameters)]
    byK = Chains[]
    for K in Ks
        cs = [chains[i] for i in eachindex(parts) if parts[i]["chain"] == K]
        cat = vcat([Array(c[:, keep, 1]) for c in cs]...)
        push!(byK, Chains(reshape(cat, size(cat, 1), size(cat, 2), 1), keep))
    end
    chainK = length(byK) > 1 ? chainscat(byK...) : nothing
    rhK = chainK === nothing ? nothing : rhat_ess(chainK)
    finite(x) = filter(isfinite, x)
    maxrh(v) = isempty(finite(v)) ? NaN : maximum(finite(v)); minesspop(v) = isempty(finite(v)) ? NaN : minimum(finite(v))
    dg["rhat_by_warmup_chain"] = rhK === nothing ? NaN : maxrh(rhK[2])
    nstart_modes = length(unique(mode_of))
    dg["cross_mode_rhat"] = (nstart_modes > 1 && rhK !== nothing) ? maxrh(rhK[2]) : NaN      # only if chains started from >= 2 distinct modes
    dg["n_start_modes"] = nstart_modes
    dg["excluded_modes"] = collect(init["excluded_modes"])
    dg["fingerprint"] = ctx.fingerprint

    md = IOBuffer()
    println(md, "# Fit diagnostics: $drug $design\n")
    println(md, "Staged fit (`scripts/fit_stage.jl`). No data in this file; parameters and sampler statistics only.\n")
    println(md, "- training PDX: $(length(ctx.models)) (window <= $(init["window"]) d); frozen split seed $(init["split_seed"]), holdout $(length(init["split"]["holdout_models"])) PDX")
    println(md, "- chains: $(length(parts)) = warmup chains K = $(join(Ks, ",")) x parts; $nd draws each ($(nd * length(parts)) draws), burn-in per part $(first(parts)["burnin"]), warmup $(first(values(warm))["nadapt"]) iterations")
    println(md, "- model: **$(get(init, "model", "v1"))**$(get(init, "waning", false) ? " + waning efficacy" : "") (v2 = correlated log λ0/log k2 with LKJ(2), per-PDX transition volume); Student-t nu=$(init["nu"]) residuals, latent baselines (sd $(init["baseline_sd"])), LOD $(init["lod"]) mm^3, k2 centred for $(count(ctx.mask)) of $(length(ctx.mask)) units, centred lambda0")
    println(md, "- target fingerprint: `$(ctx.fingerprint)` (init id `$(init["init_id"])`); chains started from $(nstart_modes) distinct recorded mode(s)")
    println(md, "- mode gap limit: " * (isfinite(get(init, "mode_gap", Inf)) ? "$(init["mode_gap"]) log-density units (converged modes further below the best are not used as chain starts)" : "none (every converged distinct mode starts chains)"))
    println(md, "- excluded MAP starts: " * (isempty(init["excluded_modes"]) ? "none" : join([@sprintf("lp %.1f, gap %.1f, k1 %.3g (%s, max param distance to best %.2f)", e["lp"], get(e, "gap", NaN), get(e, "k1", NaN), e["reason"], e["dist"]) for e in init["excluded_modes"]], "; ")))
    any(e -> haskey(e, "gap") && isfinite(e["gap"]) && e["converged"], init["excluded_modes"]) &&
        println(md, "  - justification: an excluded converged mode lies about 90 log-density units below the best mode and its data log-likelihood is about 72 lower, so its posterior mass is negligible; chains started there stay in the poor mode and only inflate R-hat. Recorded here (lp, gap, k1) instead of being silently dropped.")
    println(md, "- MAP: log-densities of the starts $(join(round.(init["all_lps"]; digits = 1), ", ")); $nmodes distinct mode(s) $(join(round.([m.lp for m in init["modes"]]; digits = 1), ", ")); chains K start at mode `mod1(K, $nmodes)`\n")
    println(md, "## Convergence\n")
    println(md, @sprintf("| statistic | value |\n|---|---|\n| max R-hat, all %d (K,P) chains | %.4f |\n| min bulk ESS, all chains | %.0f |\n| max R-hat, per warmup chain K (%d chains) | %s |\n| cross-mode R-hat (chains started from different modes) | %s |\n| divergences | %d |\n| mean acceptance | %.3f |\n| mean tree depth | %.2f |\n| mean n_steps | %.1f |\n| depth-limit hits | %d |",
                         length(parts), dg["max_rhat"], dg["min_ess_bulk"], length(Ks), rhK === nothing ? "n/a (one warmup chain)" : @sprintf("%.4f", maxrh(rhK[2])),
                         nstart_modes > 1 && rhK !== nothing ? @sprintf("%.4f", maxrh(rhK[2])) : "n/a (single start mode)", dg["n_divergences"], dg["mean_acceptance"],
                         dg["mean_tree_depth"], dg["mean_n_steps"], dg["max_tree_depth_hits"]))
    println(md, "\n### Population parameters\n")
    println(md, "| parameter | median | 90% CrI | R-hat (all) | ESS bulk (all) |" * (rhK === nothing ? "" : " R-hat (by K) |") * "\n|---|---|---|---|---|" * (rhK === nothing ? "" : "---|"))
    q(x) = (quantile(x, 0.05), median(x), quantile(x, 0.95))
    for nmk in pop
        x = vec(Array(chain[nmk])); lo, md_, hi = q(x)
        i = findfirst(==(nmk), Symbol.(prm_all))
        row = @sprintf("| %s | %.3f | [%.3f, %.3f] | %.3f | %.0f |", nmk, md_, lo, hi, rh_all[i], es_all[i])
        if rhK !== nothing
            j = findfirst(==(nmk), Symbol.(rhK[1])); row *= @sprintf(" %.3f |", rhK[2][j])
        end
        println(md, row)
    end
    println(md, "\n### Warmup chains\n\n| K | start mode (lp) | step size | mean n_steps (warmup) | warmup min | sample min per part |\n|---|---|---|---|---|---|")
    for K in Ks
        w = warm[K]; ps = [p for p in parts if p["chain"] == K]
        println(md, @sprintf("| %d | %d (%.1f) | %.4f | %.1f | %.1f | %s |", K, w["mode"], w["mode_lp"], w["epsilon"], mean(w["warm_stats"][:, 1]),
                             w["warm_time_s"] / 60, join([@sprintf("%.1f", p["sample_time_s"] / 60) for p in ps], ", ")))
    end
    if nstart_modes > 1
        println(md, "\n### Cross-mode check\n\nPosterior means of the population parameters per warmup chain (start mode in brackets); a mode that kept its chain away from the others shows as a column that differs by more than the within-chain spread.\n")
        println(md, "| K (mode) | " * join(string.(pop), " | ") * " |\n|---|" * repeat("---|", length(pop)))
        for (iK, K) in enumerate(Ks)
            vals = [mean(Array(byK[iK][nmk])) for nmk in pop]
            println(md, "| $K ($(warm[K]["mode"])) | " * join([@sprintf("%.3f", v) for v in vals], " | ") * " |")
        end
    end
    println(md, "\n### Chain table\n\n| chain (K.P) | seed | mean n_steps | divergences | time (min) |\n|---|---|---|---|---|")
    for p in parts
        println(md, @sprintf("| %s.%s | %d | %.1f | %d | %.1f |", p["chain"], p["part"], p["seed"], mean(p["stats"][:, 1]), Int(sum(p["stats"][:, 5])), p["sample_time_s"] / 60))
    end
    println(md, "\nCore-time of all stages (sum of job times): $(round(wall_core / 3600; digits = 2)) h.")
    # v1 keeps the contract names; v2 / waning fits get a suffix so they sit next to the v1 fits (also set by CI as SUFFIX)
    suffix = (get(init, "model", "v1") == "v2" ? "_v2" : "") * (get(init, "waning", false) ? "_waning" : "")
    path = joinpath(out, "$(drug)_$(design)$(suffix).jld2")
    M.save_fit(path, pd)
    write(joinpath(out, "$(drug)_$(design)$(suffix)_diagnostics.md"), String(take!(md)))
    stamp(@sprintf("merged: max R-hat %.4f, min ESS %.0f, divergences %d; saved %s and the diagnostics md", dg["max_rhat"], dg["min_ess_bulk"], dg["n_divergences"], path))
    return pd
end

function main(args)
    pos, opts = parse_cli(args)
    stage, drug, design = pos[1], pos[2], pos[3]
    design in ("primary", "late21") || error("design must be primary or late21")
    stage == "init" ? do_init(drug, design, opts) :
    stage == "warmup" ? do_warmup(drug, design, opts) :
    stage == "sample" ? do_sample(drug, design, opts) :
    stage == "merge" ? do_merge(drug, design, opts) : error("unknown stage $stage")
    return nothing
end

main(ARGS)
