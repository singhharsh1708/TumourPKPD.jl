# Static (CairoMakie) renderings of the explorer panels for the README. They use the same core and plot code as
# the live app. With demo fits the figures carry a DEMO label.
#
#   julia --project=TumourPKPD.jl/app -t 2 TumourPKPD.jl/app/screenshots.jl [fits_dir]

using CairoMakie
include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "plots.jl"))
using .ExplorerCore, .ExplorerPlots
using Statistics, Printf, TumourPKPD

CairoMakie.activate!()
const fitsdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "demo_fits")
const outdir = joinpath(@__DIR__, "screenshots")
mkpath(outdir)
fits = load_fits(fitsdir; demo_dir = joinpath(@__DIR__, "demo_fits"))

function default_model(h)
    best, bi = -1, 1
    for (i, m) in enumerate(h.pd.models)
        u = h.units[m]
        s = length(u.treated.t) + (maximum(u.treated.t) >= h.eval_window ? 1000 : 0)
        s > best && ((best, bi) = (s, i))
    end
    return h.pd.models[bi]
end

function panel(h, p, path; trial = true, quick = true)
    v = update_view(h, p)
    tv = trial_view(h, p; quick)
    fig = Figure(size = (1000, trial ? 1100 : 620), backgroundcolor = :white)
    # banner
    bg = v.extrap ? colorant"#fef3c7" : colorant"#ecfdf5"
    ed = v.extrap ? colorant"#f59e0b" : colorant"#6ee7b7"
    head = v.extrap ? "EXTRAPOLATION: beyond what the PDXE data identify" : "Within the range of the PDXE data"
    lines = [head; ["- " * r for r in v.reasons]]
    wrap(s) = join([length(l) > 135 ? l[1:132] * "..." : l for l in split(s, "\n")], "\n")
    Box(fig[1, 1]; color = bg, strokecolor = ed, strokewidth = 2, cornerradius = 6)
    Label(fig[1, 1], wrap(join(lines, "\n")); halign = :left, justification = :left, fontsize = 12,
          padding = (12, 12, 8, 8), tellwidth = false)
    who = p.model === nothing ? "new PDX (population prediction)" : "$(p.model) ($(h.units[p.model].tumour_type))"
    ttl = string(h.drug, " · ", who, " · ", @sprintf("%.2g× PDXE reference dose, %s, %d d treatment, %d d horizon",
                 p.mult, p.schedule, p.duration, p.horizon), h.demo ? " · DEMO, not fitted" : "")
    MainPlot(fig[2, 1], v; logscale = true, lod = detection_limit(), title = ttl)
    if trial
        tp = TrialPlot(fig[3, 1], tv; title = "Virtual PDX trial (" * (tv.quick ? "quick" : "full") * "): $(tv.current.n_pdx) PDX × $(tv.current.ndraws) draws, scored on days 0–$(round(Int, tv.horizon))" *
                                              (h.demo ? " · DEMO, not fitted" : ""))
        tp.ax.xlabel = "responder = mCR + mPR; whiskers = 90% CrI; scored on days 0–$(round(Int, tv.horizon)) (median PDXE follow-up)"
        ec = tv.extrapolation_current
        note = string("Interval: ", tv.current.interval_kind, ".\n",
                      ec.flag ? "Trial extrapolation: " * join(ec.reasons, "; ") : "Trial regimen and scoring horizon are inside the PDXE data.")
        note = join([length(l) > 150 ? l[1:147] * "..." : l for l in split(note, "\n")], "\n")
        Label(fig[4, 1], note; halign = :left, justification = :left, fontsize = 11, color = :gray25, tellwidth = false)
        rowsize!(fig.layout, 1, Auto(0.14)); rowsize!(fig.layout, 2, Auto(1.0)); rowsize!(fig.layout, 3, Auto(0.85))
    end
    save(path, fig; px_per_unit = 1.5)
    println("wrote ", path)
end

h = fits["binimetinib"]
w = Int(h.eval_window)
panel(h, Params(; drug = "binimetinib", model = default_model(h), schedule = h.ref_schedule, duration = w, horizon = w),
      joinpath(outdir, "reference-regimen.png"))
panel(h, Params(; drug = "binimetinib", model = nothing, mult = 2.0, schedule = "5-on-2-off", duration = 28, horizon = 120),
      joinpath(outdir, "extrapolation-new-pdx.png"))
h2 = fits["BKM120"]
w2 = Int(h2.eval_window)
panel(h2, Params(; drug = "BKM120", model = default_model(h2), schedule = h2.ref_schedule, mult = 1.0, duration = w2, horizon = w2),
      joinpath(outdir, "bkm120-reference.png"); trial = false)
