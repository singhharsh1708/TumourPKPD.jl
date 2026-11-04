# Headless smoke test: builds the explorer state and app without a browser and exercises every update path.
#
#   julia --project=TumourPKPD.jl/app -t 2 TumourPKPD.jl/app/smoke_test.jl [fits_dir]
#
# With no argument (or a directory lacking a drug) demo fits are written to app/demo_fits and used.

include(joinpath(@__DIR__, "explorer.jl"))
using .Explorer
using .Explorer.ExplorerCore
using Statistics, Bonito, Printf

const fitsdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "demo_fits")
const demodir = joinpath(@__DIR__, "demo_fits")

nfail = Ref(0); npass = Ref(0)
function check(cond, msg)
    if cond
        npass[] += 1
    else
        nfail[] += 1
        println("FAIL: ", msg)
    end
    return cond
end

fits = load_fits(fitsdir; demo_dir = demodir)
println("fits: ", join(["$d ($(fits[d].label))" for d in sort(collect(keys(fits)))], ", "))
check(length(fits) == 3, "three drugs loaded")

function sane(v)
    ok = size(v.treated) == (length(v.t), 5) && size(v.control) == (length(v.t), 5)
    ok &= all(isfinite, v.treated) && all(isfinite, v.control)
    # quantile columns are ordered
    ok &= all(v.treated[:, 1] .<= v.treated[:, 3] .+ 1e-9) && all(v.treated[:, 3] .<= v.treated[:, 5] .+ 1e-9)
    ok &= all(v.control[:, 1] .<= v.control[:, 3] .+ 1e-9) && all(v.control[:, 3] .<= v.control[:, 5] .+ 1e-9)
    return ok
end

# 1. every drug x schedule x (known PDX | new PDX): main view + banner reasons
for d in DRUGS
    st = ExplorerState(fits; drug = d)
    h = fits[d]
    for sched in SCHEDULES, model in (h.pd.models[min(3, end)], nothing)
        v = set_params!(st; trial = false, schedule = sched, model = model, mult = 1.5, duration = 21, horizon = 70)
        check(sane(v), "$d $sched $(model === nothing ? "new PDX" : model): finite, ordered bands")
        check(v.extrap, "$d $sched $(model === nothing ? "new PDX" : model): 1.5x / 21 d flagged as extrapolation")
        check(!isempty(v.reasons), "$d $sched: reasons listed")
        check(v.t[end] == 70.0 && v.window > 0, "$d: horizon and observed window set")
        check(model === nothing || length(v.obs.tt) > 0, "$d $model: observed points present")
    end
    # the reference regimen over the PDXE window is not extrapolation
    v = set_params!(st; trial = false, schedule = h.ref_schedule, model = h.pd.models[Explorer.default_model_index(h)],
                    mult = 1.0, duration = h.eval_window, horizon = h.eval_window)
    check(!v.extrap || any(occursin("follow-up", r) for r in v.reasons), "$d: reference regimen unflagged (or only the follow-up window)")
    # more drug must not predict more tumour (median at the end)
    lo = set_params!(st; trial = false, schedule = h.ref_schedule, mult = 0.5, duration = 60, horizon = 60)
    hi = set_params!(st; trial = false, schedule = h.ref_schedule, mult = 3.0, duration = 60, horizon = 60)
    check(hi.treated[end, 3] <= lo.treated[end, 3], "$d: median volume falls with dose multiplier")
    # control does not depend on the regimen
    check(lo.control == hi.control, "$d: control band independent of regimen")
    # uncertainty widens beyond the data: new-PDX 90% band (log10) wider at the end than at day 7
    v = set_params!(st; trial = false, schedule = h.ref_schedule, model = nothing, mult = 1.0, duration = 90, horizon = 90)
    wid = log10.(v.treated[:, 5]) .- log10.(max.(v.treated[:, 1], 1e-3))
    check(wid[end] > wid[findfirst(>=(7.0), v.t)], "$d: band widens with time")
end

# 2. virtual trial for each drug
for d in DRUGS
    st = ExplorerState(fits; drug = d)
    h = fits[d]
    set_params!(st; trial = false, schedule = h.ref_schedule, model = nothing)
    tv = trial_view(h, st.raw[]; quick = true)
    for r in (tv.current, tv.reference)
        check(0 <= r.responder.lo <= r.responder.median <= r.responder.hi <= 1, "$d trial: responder CrI ordered")
        s = sum(r.cats[c].median for c in TRIAL_CATS)
        check(0.8 < s < 1.2, "$d trial: category medians sum to ~1 ($(round(s; digits = 3)))")
    end
    check(tv.horizon == h.eval_window && tv.current.horizon == h.eval_window, "$d trial: horizon reported")
    check(tv.n_observed > 100, "$d trial: observed PDXE counts present ($(tv.n_observed))")
    # a much stronger regimen should raise the predicted responder rate
    tv2 = set_params!(st; trial = true, quick = true, mult = 3.0, schedule = h.ref_schedule, duration = h.eval_window, horizon = h.eval_window)
    check(st.tview[].current.responder.median >= tv.reference.responder.median - 0.02, "$d trial: 3x dose responder rate not below reference")
end

# 3. widget flow with debouncing: several quick slider moves give one update
let st = ExplorerState(fits)
    n0 = length(st.timings)
    for k in (30, 35, 40, 45)
        st.dur.value[] = st.dur.values[][k]        # simulates dragging the slider
        sleep(0.02)
    end
    sleep(0.6)
    check(length(st.timings) == n0 + 1, "debounce collapsed 4 slider moves into 1 update (got $(length(st.timings) - n0))")
    check(st.request[].duration == Float64(st.dur.values[][45]), "debounced request holds the last slider value")
    wait_trial(st)
    # drug switch via the dropdown resets PDX list and schedule
    st.drug_dd.option_index[] = 2
    sleep(1.3); wait_trial(st)
    check(st.request[].drug == "BKM120" && st.request[].model !== nothing, "drug dropdown switches fit and picks a default PDX")
    check(length(st.model_dd.options[]) == length(fits["BKM120"].pd.models) + 1, "PDX list rebuilt for the new drug")
    st.model_filter.value[] = "NSCLC"
    check(all(occursin("NSCLC", string(o)) for o in st.model_dd.options[][2:end]) && length(st.model_dd.options[]) > 2, "PDX filter narrows the list")
    sleep(1.3); wait_trial(st)
    check(st.request[].model !== nothing && fits["BKM120"].units[st.request[].model].tumour_type == "NSCLC", "filtered selection is an NSCLC model")
end

# 3b. trial provenance, serialized publication, bounded queue, regimen independence from the viewing horizon
let st = ExplorerState(fits; drug = "BKM120")
    h = fits["BKM120"]
    ExplorerCore = Explorer.ExplorerCore
    set_params!(st; trial = true, schedule = "qd", mult = 1.0, duration = 30, horizon = 60)
    tv = st.tview[]
    check(tv !== nothing && ExplorerCore.trial_matches(tv, h, st.raw[]), "trial shown with matching provenance")
    check(tv.prov.drug == "BKM120" && tv.prov.demo == h.demo && tv.prov.duration == 30 && tv.prov.mult == 1.0, "provenance records drug, demo flag, regimen")
    check(!isempty(tv.current.interval_kind), "interval_kind text carried: " * tv.current.interval_kind[1:min(end, 60)])
    check(tv.extrapolation_current isa NamedTuple && !isempty(tv.extrapolation_current.reasons), "trial extrapolation reasons present for a stopped regimen")
    # viewing horizon does not change the tested regimen
    tv_h = ExplorerCore.trial_view(h, Params(; drug = "BKM120", schedule = "qd", mult = 1.0, duration = 30, horizon = 14))
    check(tv_h.current === tv.current, "trial identical for horizon 14 and 60 at the same treatment duration")
    # a change of inputs hides the old trial at once (before any debounce)
    st.mult.value[] = 1.5
    sleep(0.1)
    check(st.tview[] === nothing && !st.tp.shown[], "trial hidden immediately when an input changes")
    sleep(1.4); wait_trial(st)
    check(st.tview[] !== nothing && st.tview[].prov.mult == 1.5, "trial recomputed for the new inputs")
    # a late result for old inputs is dropped by the UI task
    old = tv
    Explorer.ui!(st; wait = true) do
        Explorer.publish_trial!(st, st.trial_gen[], old, h, time())
    end
    check(st.tview[].prov.mult == 1.5, "late result with other provenance is not published")
    # a model-only change keeps the trial (not an input of the trial)
    keep = st.tview[]
    st.model_dd.option_index[] = 1
    sleep(0.3)
    check(st.tview[] === keep, "model change keeps the trial that does not depend on it")
    # queue: one running, one replaceable pending per owner
    sc = Explorer.TRIAL_SCHED
    got = Int[]
    lk = ReentrantLock()
    slow = Base.Event()
    owner = :testowner
    Explorer.submit_trial_job!(:blocker, () -> wait(slow))
    sleep(0.2)
    for i in 1:5
        Explorer.submit_trial_job!(owner, () -> lock(() -> push!(got, i), lk))
    end
    check(count(x -> x.first === owner, sc.pending) == 1, "at most one pending request per owner")
    notify(slow); sleep(1.0)
    check(got == [5], "only the latest pending request ran (ran: $got)")
    # population option (quick caps keep it short)
    set_params!(st; trial = true, quick = true, population = true, mult = 1.0, duration = h.eval_window)
    check(st.tview[] !== nothing && st.tview[].population && occursin("population", lowercase(st.tview[].current.interval_kind)), "population mode result and interval text")
end

# 4. app object + static render
app = build_app(fits)
check(app isa Bonito.App, "build_app returns a Bonito.App")
html = joinpath(mktempdir(), "explorer.html")
Bonito.export_static(html, app)
sz = filesize(html)
txt = read(html, String)
check(sz > 50_000, "static export rendered ($(round(sz / 1e6; digits = 1)) MB)")
check(occursin("Fit diagnostics", txt) && occursin("Data provenance", txt) && occursin("10.1038/nm.3954", txt), "info panel rendered")
check(occursin("PDXE reference dose", txt), "dose multiplier labelled as x PDXE reference dose")

# 5. timings (warmed up): vary the regimen so every call is a cache miss
println("\nupdate times (seconds), uncached main plot, 200 draws:")
for d in DRUGS
    st = ExplorerState(fits; drug = d)
    h = fits[d]
    set_params!(st; trial = false)
    ts = Float64[]
    for (i, m) in enumerate(0.3:0.05:2.9)
        sched = SCHEDULES[mod1(i, 4)]
        t0 = time()
        v = update_view(h, Params(; drug = d, model = h.pd.models[mod1(i, length(h.pd.models))], mult = m, schedule = sched,
                                 duration = 20 + i, horizon = 60 + i))
        push!(ts, time() - t0)
    end
    cached = Float64[]
    for _ in 1:5
        t0 = time(); update_view(h, Params(; drug = d, model = h.pd.models[1], mult = 0.3, schedule = "qd", duration = 21, horizon = 61)); push!(cached, time() - t0)
    end
    @printf("  %-12s n=%d  median %.3f  p90 %.3f  max %.3f  | cached median %.4f\n", d, length(ts), median(ts), quantile(ts, 0.9), maximum(ts), median(cached))
end
println("\npassed $(npass[]), failed $(nfail[])")
nfail[] == 0 || exit(1)
