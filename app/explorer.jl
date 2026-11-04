# TumourPKPD explorer: interactive Bayesian tumour PK/PD on public PDX data (Bonito + WGLMakie).
#
#   include("explorer.jl"); using .Explorer
#   app = Explorer.build_app(Explorer.ExplorerCore.load_fits("fits"))      # see run.jl
#
# Layers: core.jl (computation, cached, no UI) -> plots.jl (Makie panels, backend agnostic) -> this file
# (widgets, debouncing, layout). `ExplorerState` owns all Observables and works without a browser, which
# is what smoke_test.jl uses.

module Explorer

using Bonito, Observables, Statistics, Printf
import Makie, WGLMakie
using TumourPKPD

include("core.jl")
include("plots.jl")
using .ExplorerCore, .ExplorerPlots

export ExplorerState, build_app, set_params!, wait_trial

const DEBOUNCE_S = 0.12          # slider settle time before recomputing the main plot
const TRIAL_DEBOUNCE_S = 0.8     # the virtual trial waits longer: it is the expensive panel
const DOC_DOI = "10.1038/nm.3954"

# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------

function debounce(dt, src::Observable{T}) where {T}
    out = Observable{T}(src[])
    timer = Ref{Union{Nothing,Timer}}(nothing)
    on(src) do v
        timer[] === nothing || close(timer[])
        timer[] = Timer(_ -> (out[] = v), dt)
    end
    return out
end

model_id(label::AbstractString) = String(first(split(label, " · ")))

# Default PDX for a drug: the model with most treated observations among those followed to the evaluation window.
function default_model_index(h::FitHandle)
    best, bi = -1, 1
    for (i, m) in enumerate(h.pd.models)
        u = h.units[m]
        score = length(u.treated.t) + (maximum(u.treated.t) >= h.eval_window ? 1000 : 0)
        if score > best
            best, bi = score, i
        end
    end
    return bi
end

function model_options(h::FitHandle, filter::AbstractString)
    f = lowercase(strip(filter))
    labels = h.model_labels
    keep = isempty(f) ? labels : [l for l in labels if occursin(f, lowercase(l))]
    return Any[NEWPDX; keep...]
end

# ---------------------------------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------------------------------

mutable struct ExplorerState
    fits::Dict{String,FitHandle}
    drugs::Vector{String}
    # widgets
    drug_dd::Bonito.Dropdown
    model_filter::Bonito.TextField
    model_dd::Bonito.Dropdown
    sched_dd::Bonito.Dropdown
    mult::Bonito.Slider
    dur::Bonito.Slider
    hor::Bonito.Slider
    logscale::Bonito.Checkbox
    noise::Bonito.Checkbox
    refine::Bonito.Button
    popmode::Bonito.Checkbox
    # flow
    raw::Observable{Params}
    request::Observable{Params}
    trial_request::Observable{Params}
    # outputs
    view::Observable{Any}
    tview::Observable{Any}
    banner::Observable{Any}
    status::Observable{String}
    trial_status::Observable{String}
    trial_extrap::Observable{Any}
    fig_main::Makie.Figure
    fig_trial::Makie.Figure
    mp::MainPlot
    tp::TrialPlot
    trial_gen::Base.RefValue{Int}        # bumped when a trial is submitted (UI task only)
    trial_done::Base.RefValue{Int}       # generation whose outcome the UI task has handled
    chan::Channel{Any}                   # every UI mutation goes through this queue, one consumer
    ui_task::Base.RefValue{Any}
    timings::Vector{NamedTuple}      # (; seconds, cached, drug, model) of every main-plot update
    trial_timings::Vector{NamedTuple}
end

ctl_handle(st::ExplorerState) = st.fits[st.drug_dd.value[]]

function current_params(st::ExplorerState)
    d = st.drug_dd.value[]
    h = st.fits[d]
    lab = st.model_dd.value[]
    m = lab == NEWPDX ? nothing : model_id(lab)
    (m !== nothing && !(m in h.pd.models)) && (m = nothing)     # transient state while the drug changes
    return Params(; drug = d, model = m, mult = Float64(st.mult.value[]), schedule = st.sched_dd.value[],
                  duration = Float64(st.dur.value[]), horizon = Float64(st.hor.value[]),
                  obs_noise = st.noise.value[])
end

fmt_ms(s) = s < 1 ? @sprintf("%.0f ms", 1000s) : @sprintf("%.1f s", s)

function title_for(h::FitHandle, p::Params)
    who = p.model === nothing ? "new PDX (population prediction)" :
          string(p.model, " (", h.units[p.model].tumour_type, ")")
    return string(h.drug, " · ", who, h.demo ? " · DEMO, not fitted" : "")
end

function banner_dom(v, h::FitHandle)
    D = Bonito.DOM
    if v.extrap
        items = [D.li(r) for r in v.reasons]
        return D.div(
            D.div("EXTRAPOLATION: this prediction goes beyond what the PDXE data identify",
                  style = "font-weight:700;font-size:15px;margin-bottom:4px;"),
            D.ul(items...; style = "margin:0 0 0 18px;padding:0;font-size:13px;"),
            D.div("The orange region of the plot is beyond the observed data; the bands there are model-based, " *
                  "not data-constrained.", style = "margin-top:4px;font-size:12px;opacity:.85;");
            style = "background:#fef3c7;border:2px solid #f59e0b;color:#78350f;padding:10px 14px;border-radius:6px;")
    else
        items = [D.li(r) for r in v.reasons]
        return D.div(
            D.div("Within the range of the PDXE data: reference dose level and schedule, follow-up inside the observed window",
                  style = "font-weight:600;"),
            isempty(items) ? "" : D.ul(items...; style = "margin:4px 0 0 18px;padding:0;font-size:12px;");
            style = "background:#ecfdf5;border:1px solid #6ee7b7;color:#064e3b;padding:8px 14px;border-radius:6px;font-size:13px;")
    end
end

function trial_caption(tv, h::FitHandle)
    c = tv.current
    q = tv.prov
    return string(h.demo ? "DEMO — not fitted. " : "",
                  "Virtual PDX trial (", tv.population ? "population response rate" : tv.quick ? "quick" : "refined", "): ",
                  c.n_pdx, " new PDX × ", c.ndraws, " posterior draws; tested regimen ", @sprintf("%.2f", q.mult), "× PDXE reference dose, ",
                  q.schedule, ", dosed for ", round(Int, q.duration), " d; response scored on days 0–", round(Int, tv.horizon),
                  " (the drug's median PDXE follow-up, rates depend on it). Responder rate = mCR+mPR. Whiskers: 90% ",
                  "interval. What it means: ", c.interval_kind, ".")
end

function trial_extrap_dom(tv)
    D = Bonito.DOM
    ec, er = tv.extrapolation_current, tv.extrapolation_reference
    if ec.flag || er.flag
        items = [D.li("current regimen: ", r) for r in ec.reasons]
        append!(items, [D.li("reference regimen: ", r) for r in er.reasons])
        return D.div(D.div("Trial extrapolation: the tested regimen or scoring horizon goes beyond the PDXE data";
                           style = "font-weight:700;margin-bottom:2px;"),
                     D.ul(items...; style = "margin:0 0 0 18px;padding:0;");
                     style = "background:#fef3c7;border:1px solid #f59e0b;color:#78350f;padding:6px 12px;border-radius:6px;font-size:12px;margin-top:6px;")
    end
    return D.div("Trial regimen and scoring horizon are inside the PDXE data (same dose level, schedule, follow-up).";
                 style = "background:#ecfdf5;border:1px solid #6ee7b7;color:#064e3b;padding:4px 12px;border-radius:6px;font-size:12px;margin-top:6px;")
end

function ExplorerState(fits::Dict{String,FitHandle}; drug::Union{Nothing,String} = nothing)
    drugs = [d for d in DRUGS if haskey(fits, d)]
    isempty(drugs) && error("no drugs available")
    d0 = drug === nothing ? first(drugs) : drug
    h0 = fits[d0]
    mi = default_model_index(h0)
    drug_dd = Bonito.Dropdown(drugs; index = findfirst(==(d0), drugs))
    filter_tf = Bonito.TextField("")
    opts = model_options(h0, "")
    model_dd = Bonito.Dropdown(opts; index = mi + 1)
    sched_dd = Bonito.Dropdown(SCHEDULES; index = findfirst(==(h0.ref_schedule), SCHEDULES))
    mult = Bonito.Slider(collect(0.25:0.05:3.0); value = 1.0)
    w0 = Int(h0.eval_window)
    dur = Bonito.Slider(collect(1:340); value = w0)
    hor = Bonito.Slider(collect(14:340); value = w0)
    logscale = Bonito.Checkbox(true)
    noise = Bonito.Checkbox(true)
    refine = Bonito.Button("Refine trial (full size)")
    popmode = Bonito.Checkbox(false)

    p0 = Params(; drug = d0, model = h0.pd.models[mi], mult = 1.0, schedule = h0.ref_schedule,
                duration = float(w0), horizon = float(w0))
    v0 = update_view(h0, p0)
    tv0 = trial_view(h0, p0; quick = true)

    fig_main = Makie.Figure(size = (880, 480), backgroundcolor = :white)
    mp = MainPlot(fig_main[1, 1], v0; logscale = true, lod = detection_limit(), title = title_for(h0, p0))
    fig_trial = Makie.Figure(size = (880, 400), backgroundcolor = :white)
    tp = TrialPlot(fig_trial[1, 1], tv0; title = "Virtual PDX trial" * (h0.demo ? " · DEMO, not fitted" : ""))
    tp.ax.xlabel = "scored on days 0–$(round(Int, tv0.horizon)) (median PDXE follow-up)"

    st = ExplorerState(fits, drugs, drug_dd, filter_tf, model_dd, sched_dd, mult, dur, hor, logscale, noise, refine, popmode,
                       Observable(p0), Observable(p0), Observable(p0),
                       Observable{Any}(v0), Observable{Any}(tv0), Observable{Any}(banner_dom(v0, h0)),
                       Observable(""), Observable(trial_caption(tv0, h0) * " Computed at start-up."),
                       Observable{Any}(trial_extrap_dom(tv0)),
                       fig_main, fig_trial, mp, tp, Ref(0), Ref(0), Channel{Any}(Inf), Ref{Any}(nothing),
                       NamedTuple[], NamedTuple[])
    wire!(st)
    return st
end

# ---------------------------------------------------------------------------------------------------
# serialized UI queue and bounded trial scheduler
# ---------------------------------------------------------------------------------------------------

"""
Run `f()` on the session's single UI task (a Channel consumer). All reads and writes of the plots, banner
and status Observables happen there, so a trial result is checked and published atomically with respect to
input changes. With `wait=true` blocks until `f` has run (never call that from inside the UI task).
"""
function ui!(f, st::ExplorerState; wait::Bool = false)
    ev = Base.Event()
    put!(st.chan, () -> try
        f()
    finally
        notify(ev)
    end)
    wait && Base.wait(ev)
    return nothing
end

# One trial runs at a time in the whole process; each session may have one replaceable pending request.
mutable struct TrialScheduler
    cond::Threads.Condition
    pending::Vector{Pair{Any,Function}}
    worker::Union{Nothing,Task}
end
const TRIAL_SCHED = TrialScheduler(Threads.Condition(), Pair{Any,Function}[], nothing)

function submit_trial_job!(owner, job::Function)
    sc = TRIAL_SCHED
    lock(sc.cond) do
        filter!(x -> x.first !== owner, sc.pending)      # replace the owner's pending request
        push!(sc.pending, owner => job)
        if sc.worker === nothing || istaskdone(sc.worker)
            sc.worker = Threads.@spawn trial_worker_loop()
        end
        notify(sc.cond)
    end
    return nothing
end

function trial_worker_loop()
    sc = TRIAL_SCHED
    while true
        job = lock(sc.cond) do
            while isempty(sc.pending)
                wait(sc.cond)
            end
            popfirst!(sc.pending).second
        end
        try
            Base.invokelatest(job)
        catch err
            @error "trial job crashed" exception = (err, catch_backtrace())
        end
    end
end

"Hide the trial result: it was computed for other inputs."
function invalidate_trial!(st::ExplorerState, msg)
    st.tview[] = nothing
    set_trial!(st.tp, nothing)
    st.trial_extrap[] = ""
    st.trial_status[] = msg
    return nothing
end

"""
    submit_trial!(st, p; quick, population)  (UI task only)

Starts a generation, hides any shown trial that does not match `p`, and queues the computation. The worker
only computes; its outcome comes back through the UI queue, where it is published only if its generation is
still the latest and its provenance matches the inputs at that moment.
"""
function submit_trial!(st::ExplorerState, p::Params; quick::Bool = true, population::Bool = false)
    gen = (st.trial_gen[] += 1)
    h = st.fits[p.drug]
    matches = trial_matches(st.tview[], h, p)
    matches || invalidate_trial!(st, "trial hidden: it belongs to other inputs")
    st.trial_status[] = string("computing the ", population ? "population-rate" : quick ? "quick" : "full-size",
                               " virtual trial … (the main plot is already current)")
    job = function ()
        t0 = time()
        outcome = try
            trial_view(h, p; quick, population)
        catch err
            @error "virtual trial failed" exception = (err, catch_backtrace())
            err
        end
        put!(st.chan, () -> publish_trial!(st, gen, outcome, h, t0))     # computing is done; the UI task decides
        return nothing
    end
    submit_trial_job!(st, job)
    return gen
end

function publish_trial!(st::ExplorerState, gen::Int, outcome, h::FitHandle, t0)
    st.trial_done[] = max(st.trial_done[], gen)
    if outcome isa Exception
        gen == st.trial_gen[] && invalidate_trial!(st, "virtual trial failed: $(sprint(showerror, outcome))")
        return nothing
    end
    cur = st.raw[]
    if gen != st.trial_gen[] || !trial_matches(outcome, st.fits[cur.drug], cur)
        return nothing                      # superseded, or computed for inputs that are no longer current
    end
    st.tp.ax.xlabel = "scored on days 0–$(round(Int, outcome.horizon)) (median PDXE follow-up)"
    st.tp.ax.title = string("Virtual PDX trial", h.demo ? " · DEMO, not fitted" : "")
    set_trial!(st.tp, outcome)
    st.tview[] = outcome
    st.trial_extrap[] = trial_extrap_dom(outcome)
    st.trial_status[] = trial_caption(outcome, h) * " Updated in " * fmt_ms(time() - t0) * "."
    push!(st.trial_timings, (; seconds = time() - t0, quick = outcome.quick, population = outcome.population, drug = h.drug))
    return nothing
end

"Block until the outcome of the latest submitted trial has been handled by the UI task (tests, screenshots)."
function wait_trial(st::ExplorerState; timeout = 120.0)
    t0 = time()
    while st.trial_done[] < st.trial_gen[] && time() - t0 < timeout
        sleep(0.02)
    end
    return st.trial_done[] >= st.trial_gen[]
end

function refresh_main!(st::ExplorerState, p::Params)
    t0 = time()
    h = st.fits[p.drug]
    v = update_view(h, p)
    st.mp.title[] = title_for(h, v.params)
    st.mp.view[] = v
    st.view[] = v
    st.banner[] = banner_dom(v, h)
    dt = time() - t0
    push!(st.timings, (; seconds = dt, cached = v.cached, drug = p.drug, model = p.model))
    st.status[] = string("main plot updated in ", fmt_ms(dt), v.cached ? " (cached)" : "",
                         " · ", length(v.t), " time points · ", v.ndoses, " doses · ", length(h.draw_idx), " posterior draws")
    return v
end

"Set the parameters programmatically and update through the UI queue (no browser needed)."
function set_params!(st::ExplorerState; trial::Bool = true, quick::Bool = true, population::Bool = false, kwargs...)
    p0 = st.raw[]
    p = Params(; drug = get(kwargs, :drug, p0.drug), model = haskey(kwargs, :model) ? kwargs[:model] : p0.model,
               mult = get(kwargs, :mult, p0.mult), schedule = get(kwargs, :schedule, p0.schedule),
               duration = float(get(kwargs, :duration, p0.duration)), horizon = float(get(kwargs, :horizon, p0.horizon)),
               obs_noise = get(kwargs, :obs_noise, p0.obs_noise))
    v = nothing
    ui!(st; wait = true) do
        v = refresh_main!(st, p)
        st.raw.val = p
        st.request.val = p
        if trial
            submit_trial!(st, p; quick, population)
        elseif !trial_matches(st.tview[], st.fits[p.drug], p)
            invalidate_trial!(st, "trial hidden: it belongs to other inputs")
        end
    end
    trial && wait_trial(st)
    return v
end

function wire!(st::ExplorerState)
    # drug change: reset schedule and PDX list
    on(st.drug_dd.value) do d
        h = st.fits[d]
        sched = findfirst(==(h.ref_schedule), SCHEDULES)
        st.sched_dd.option_index[] = sched
        st.model_filter.value.val = ""
        st.model_dd.option_index[] = 1
        st.model_dd.options[] = model_options(h, "")
        st.model_dd.option_index[] = default_model_index(h) + 1
        w = Int(h.eval_window)
        for (sl, x) in ((st.dur, w), (st.hor, w), (st.mult, 1.0))
            sl.index[] = findfirst(==(x), sl.values[])    # moves the thumb in the browser
            sl.value.val = x                              # and the value, for headless use
        end
    end
    # PDX text filter
    on(st.model_filter.value) do f
        h = ctl_handle(st)
        opts = model_options(h, f)
        st.model_dd.option_index[] = 1
        st.model_dd.options[] = opts
        length(opts) > 1 && (st.model_dd.option_index[] = 2)
    end
    # any control -> raw params (immediately) -> debounced requests
    onany(st.drug_dd.value, st.model_dd.value, st.sched_dd.value, st.mult.value, st.dur.value, st.hor.value,
          st.noise.value) do args...
        p = current_params(st)
        ui!(st) do
            st.raw.val = p
            h = st.fits[p.drug]
            # a shown trial that does not match the new inputs is hidden at once, not after the debounce
            if !trial_matches(st.tview[], h, p)
                invalidate_trial!(st, "trial hidden: it belongs to other inputs; recomputing when you pause")
            end
        end
        st.raw[] = p
    end
    st.request = debounce(DEBOUNCE_S, st.raw)
    st.trial_request = debounce(TRIAL_DEBOUNCE_S, st.raw)
    on(st.request) do p
        ui!(st) do
            try
                refresh_main!(st, p)
            catch err
                @error "update failed" exception = (err, catch_backtrace())
                st.status[] = "update failed: $(sprint(showerror, err))"
            end
        end
    end
    on(st.trial_request) do _
        ui!(st) do
            p = st.raw[]
            trial_matches(st.tview[], st.fits[p.drug], p) || submit_trial!(st, p; quick = true)
        end
    end
    on(st.logscale.value) do l
        ui!(st) do
            st.mp.logscale[] = l
        end
    end
    on(st.refine.value) do _
        ui!(st) do
            submit_trial!(st, st.raw[]; quick = false, population = st.popmode.value[])
        end
    end
    st.ui_task[] = @async for f in st.chan
        try
            f()
        catch err
            @error "UI task" exception = (err, catch_backtrace())
        end
    end
    return st
end

# ---------------------------------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------------------------------

const PAGE_CSS = Bonito.Styles(
    Bonito.CSS("body", "margin" => "0", "background" => "#f8fafc", "color" => "#111827",
        "font-family" => "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"),
    Bonito.CSS(".tp-wrap", "max-width" => "1320px", "margin" => "0 auto", "padding" => "12px 16px 40px"),
    Bonito.CSS(".tp-row", "display" => "flex", "flex-wrap" => "wrap", "gap" => "16px", "align-items" => "flex-start"),
    Bonito.CSS(".tp-side", "flex" => "0 0 300px", "min-width" => "260px"),
    Bonito.CSS(".tp-main", "flex" => "1 1 640px", "min-width" => "0"),
    Bonito.CSS(".tp-card", "background" => "white", "border" => "1px solid #e5e7eb", "border-radius" => "8px",
        "padding" => "12px 14px", "margin-bottom" => "12px"),
    Bonito.CSS(".tp-card h3", "margin" => "0 0 8px 0", "font-size" => "15px"),
    Bonito.CSS(".tp-label", "font-size" => "13px", "font-weight" => "600", "margin" => "10px 0 2px 0"),
    Bonito.CSS(".tp-note", "font-size" => "12px", "color" => "#4b5563", "line-height" => "1.35"),
    Bonito.CSS(".tp-card table", "border-collapse" => "collapse", "font-size" => "13px"),
    Bonito.CSS(".tp-card td", "padding" => "2px 10px 2px 0", "vertical-align" => "top"),
    Bonito.CSS("input[type=range]", "width" => "100%"),
    Bonito.CSS("select", "width" => "100%"),
)

function info_dom(st::ExplorerState, h::FitHandle)
    D = Bonito.DOM
    d = h.pd.diagnostics
    num(k, f = "%.3g") = haskey(d, k) && d[k] isa Real && isfinite(d[k]) ? Printf.format(Printf.Format(f), d[k]) : "n/a"
    row(a, b) = D.tr(D.td(a; style = "color:#4b5563"), D.td(b))
    demo = h.demo ? D.div("DEMO — not fitted. These draws are synthetic (plausible parameters on the real PDX model ids), " *
                          "made to develop the interface. Diagnostics below do not exist for them.";
                          style = "background:#fee2e2;border:1px solid #f87171;color:#7f1d1d;padding:6px 10px;border-radius:6px;font-size:12px;margin-bottom:8px;") : ""
    nunits = length(h.pd.models)
    pk = h.pkprior
    dose_txt = string(round(h.ref_amt; sigdigits = 3), " mg/kg per dose, ", h.ref_route, ", every ",
                      round(h.ref_interval; digits = 2), " d")
    return D.div(
        D.div(D.h3("Fit diagnostics"), demo,
              D.table(
                  row("source", h.demo ? "DEMO — not fitted" : string("fitted posterior, ", basename(h.path))),
                  row("PDX models in the fit", string(nunits)),
                  row("posterior draws stored", string(length(h.pd.σ))),
                  row("chains × draws/chain", string(get(d, "nchains", "n/a"), " × ", get(d, "draws_per_chain", "n/a"))),
                  row("max R-hat", num("max_rhat", "%.3f")),
                  row("min bulk ESS", num("min_ess_bulk", "%.0f")),
                  row("divergent transitions", h.demo ? "n/a" : string(get(d, "n_divergences", "n/a"))),
                  row("mean acceptance", num("mean_acceptance", "%.2f")),
                  row("runtime", haskey(d, "runtime_s") && d["runtime_s"] isa Real && isfinite(d["runtime_s"]) ?
                                 @sprintf("%.0f s", d["runtime_s"]) : "n/a"));
              class = "tp-card"),
        D.div(D.h3("What is fitted and what is assumed"),
              D.table(
                  row("fitted (posterior)", "per-PDX growth rate λ0 and drug sensitivity k2, their between-PDX spread, pooled λ1, kill-delay k1, residual noise σ"),
                  row("assumed: dose", string(dose_txt, " [", pk.dose_status, "]")),
                  row("assumed: mouse PK", pk.pk_status),
                  row("assumed: schedule", "the schedule options in the sidebar are what-if scenarios; PDXE tested one schedule per drug"),
                  row("assumed: structure", "Simeoni tumour growth inhibition; exposure = concentration ÷ Cmax of the reference regimen; one-compartment oral PK held fixed"),
                  row("assumed: detection limit", string(detection_limit(), " mm³; recorded volume 0 is treated as censored below it"));
              ),
              D.div("The multiplier scales the per-dose amount relative to the PDXE reference regimen; it is not an mg/kg claim. " *
                    "Because exposure is normalised, only the ratio matters to the model.";
                    class = "tp-note", style = "margin-top:6px;");
              class = "tp-card"),
        D.div(D.h3("Data provenance"),
              D.div("Tumour-volume curves and published response categories: Novartis PDX Encyclopedia (PDXE), Gao H et al., ",
                    D.i("Nat Med"), " 21:1318–1325 (2015), doi ", D.a(DOC_DOI; href = "https://doi.org/$DOC_DOI", target = "_blank"),
                    ". Drugs here: binimetinib, BKM120, BYL719 (single agent, each PDX model with a paired untreated arm).";
                    class = "tp-note");
              class = "tp-card");
    )
end

function control_panel(st::ExplorerState)
    D = Bonito.DOM
    session_val(o) = o
    mult_lbl = map(m -> @sprintf("%.2f× PDXE reference dose (per dose)", m), st.mult.value)
    dur_lbl = map(x -> @sprintf("treatment duration: %d d", x), st.dur.value)
    hor_lbl = map(x -> @sprintf("follow-up horizon: %d d", x), st.hor.value)
    return D.div(
        D.div(D.h3("Drug and tumour"),
              D.div("drug"; class = "tp-label"), st.drug_dd,
              D.div("PDX model (type a few characters to filter, e.g. NSCLC or X-1)"; class = "tp-label"), st.model_filter,
              D.div(; style = "height:4px"), st.model_dd;
              class = "tp-card"),
        D.div(D.h3("Regimen"),
              D.div(mult_lbl; class = "tp-label"), st.mult,
              D.div("× PDXE reference dose. The PDXE doses of BKM120 and BYL719 are assumptions (see Info).";
                    class = "tp-note"),
              D.div("schedule"; class = "tp-label"), st.sched_dd,
              D.div(dur_lbl; class = "tp-label"), st.dur,
              D.div(hor_lbl; class = "tp-label"), st.hor;
              class = "tp-card"),
        D.div(D.h3("Display"),
              D.label(st.logscale, " log volume axis"; style = "display:block;font-size:13px;margin:4px 0;"),
              D.label(st.noise, " include observation noise in the bands"; style = "display:block;font-size:13px;margin:4px 0;"),
              D.div(map(identity, st.status); class = "tp-note", style = "margin-top:8px;");
              class = "tp-card"),
    )
end

function layout(st::ExplorerState)
    D = Bonito.DOM
    h0 = ctl_handle(st)
    info = map(_ -> info_dom(st, ctl_handle(st)), st.drug_dd.value)
    demo_any = any(h -> h.demo, values(st.fits))
    head = D.div(
        D.h1("Bayesian tumour PK/PD explorer"; style = "font-size:22px;margin:4px 0 2px 0;"),
        D.div("Simeoni tumour growth inhibition fitted hierarchically (NUTS) to Novartis PDXE curves. " *
              "Open, validated, Bayesian, with uncertainty that widens where the data thin.";
              class = "tp-note", style = "margin-bottom:8px;"),
        demo_any ? D.div(map(_ -> ctl_handle(st).demo ? "DEMO — not fitted: the selected drug uses synthetic parameters, not a fit. Do not read the numbers." : "",
                             st.drug_dd.value);
                         style = "color:#7f1d1d;background:#fee2e2;border-radius:6px;font-weight:700;padding:0 10px;margin-bottom:6px;") : "")
    main = D.div(
        D.div(st.banner; style = "margin-bottom:12px;"),
        D.div(st.fig_main; class = "tp-card", style = "padding:4px;overflow-x:auto;"),
        D.div(D.h3("Virtual PDX trial: current regimen vs the PDXE reference regimen"),
              D.div(st.fig_trial; style = "overflow-x:auto;"),
              D.div(map(identity, st.trial_status); class = "tp-note"),
              D.div(st.trial_extrap),
              D.div(st.refine, D.label(st.popmode, " population response rate (credible interval of the population rate; 50 draws × 1000 PDX, slow)";
                                       style = "font-size:12px;margin-left:10px;"); style = "margin-top:8px;");
              class = "tp-card"),
        D.div(info))
    return D.div(PAGE_CSS, D.div(head, D.div(D.div(control_panel(st); class = "tp-side"), D.div(main; class = "tp-main");
                                             class = "tp-row"); class = "tp-wrap"))
end

"""
    build_app(fits; title) -> Bonito.App

`fits` is a `Dict{String,FitHandle}` from `ExplorerCore.load_fits`. Every browser session gets its own
widgets and figures; the simulation caches are shared.
"""
function build_app(fits::Dict{String,FitHandle}; title = "TumourPKPD explorer")
    WGLMakie.activate!()
    return Bonito.App(; title) do session::Bonito.Session
        st = ExplorerState(fits)
        return layout(st)
    end
end

end # module
