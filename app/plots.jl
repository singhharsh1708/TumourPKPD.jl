# Backend-agnostic Makie panels. The live app (WGLMakie) and the static screenshots (CairoMakie) share this
# code: whichever backend the caller activated renders it. Panels are driven by Observables, so the live
# app updates them in place.

module ExplorerPlots

using Makie, Statistics

export MainPlot, TrialPlot, set_main!, set_trial!, PALETTE, trial_summary_lines

const PALETTE = (
    control = colorant"#6b7280",
    treated = colorant"#2563eb",
    obs_control = colorant"#374151",
    obs_treated = colorant"#1e3a8a",
    extrap = colorant"#f59e0b",
    current = colorant"#2563eb",
    reference = colorant"#9ca3af",
    observed = colorant"#d97706",
    lod = colorant"#b91c1c",
)

const LOG_FLOOR = 1.0     # mm^3: lower edge of the log axis (volumes below it are drawn on the edge)

_floor(x, lg) = lg ? max.(x, LOG_FLOOR) : x
_pts(t, y, lg) = Point2f.(t, _floor(y, lg))

mutable struct MainPlot
    ax::Axis
    view::Observable{Any}
    logscale::Observable{Bool}
    title::Observable{String}
    lod::Float64
end

"""
    MainPlot(slot, v0; logscale=true, lod=4.0, title="")

Predicted tumour volume (median, 50% and 90% bands) for the treated arm and the untreated control, the
observed PDXE points of the selected model, the observed-data window and the extrapolation shading.
`v0` is a view as returned by `ExplorerCore.update_view`.
"""
function MainPlot(slot, v0; logscale::Bool = true, lod::Float64 = 4.0, title::AbstractString = "")
    view = Observable{Any}(v0)
    lg = Observable(logscale)
    ttl = Observable(String(title))
    ax = Axis(slot; xlabel = "days since treatment start", ylabel = "tumour volume (mm³)",
              title = ttl, titlealign = :left, titlesize = 15, yscale = logscale ? log10 : identity)

    t = lift(v -> v.t, view)
    col(v, k, side) = side === :c ? v.control[:, k] : v.treated[:, k]
    # observed-data window, extrapolation shading
    win = lift(v -> v.window, view)
    hor = lift(v -> v.t[end], view)
    xs0 = lift((w, h) -> min(w, h), win, hor)
    vspan!(ax, win, hor; color = (PALETTE.extrap, 0.13))
    vlines!(ax, win; color = (PALETTE.extrap, 0.9), linestyle = :dash, linewidth = 1.5)
    text!(ax, lift(w -> Point2f(w, LOG_FLOOR), win); text = lift(w -> "extrapolation →", win),
          align = (:left, :bottom), fontsize = 11, color = (PALETTE.extrap, 1.0), space = :data,
          offset = (4, 4), visible = lift((w, h) -> w < h, win, hor))
    # control
    band!(ax, lift((v, l) -> _pts(v.t, col(v, 1, :c), l), view, lg),
              lift((v, l) -> _pts(v.t, col(v, 5, :c), l), view, lg); color = (PALETTE.control, 0.15),
          label = "untreated, 90%")
    band!(ax, lift((v, l) -> _pts(v.t, col(v, 2, :c), l), view, lg),
              lift((v, l) -> _pts(v.t, col(v, 4, :c), l), view, lg); color = (PALETTE.control, 0.25),
          label = "untreated, 50%")
    lines!(ax, lift((v, l) -> _pts(v.t, col(v, 3, :c), l), view, lg); color = PALETTE.control, linewidth = 2,
           label = "untreated, median")
    # treated
    band!(ax, lift((v, l) -> _pts(v.t, col(v, 1, :t), l), view, lg),
              lift((v, l) -> _pts(v.t, col(v, 5, :t), l), view, lg); color = (PALETTE.treated, 0.16),
          label = "treated, 90%")
    band!(ax, lift((v, l) -> _pts(v.t, col(v, 2, :t), l), view, lg),
              lift((v, l) -> _pts(v.t, col(v, 4, :t), l), view, lg); color = (PALETTE.treated, 0.30),
          label = "treated, 50%")
    lines!(ax, lift((v, l) -> _pts(v.t, col(v, 3, :t), l), view, lg); color = PALETTE.treated, linewidth = 2.5,
           label = "treated, median")
    # end of dosing
    dend = lift(v -> v.params.duration, view)
    vlines!(ax, lift(d -> [d], dend); color = (PALETTE.treated, 0.7), linestyle = :dot, linewidth = 1.5,
            visible = lift((v) -> v.params.duration < v.t[end] - 1e-9, view))
    # detection limit
    hlines!(ax, [lod]; color = (PALETTE.lod, 0.6), linestyle = :dot, linewidth = 1)
    # observed PDXE points
    scatter!(ax, lift((v, l) -> _pts(v.obs.tc, v.obs.vc, l), view, lg); color = PALETTE.obs_control,
             marker = :circle, markersize = 8, strokewidth = 1, strokecolor = :white, label = "PDXE observed, untreated")
    scatter!(ax, lift((v, l) -> _pts(v.obs.tt, v.obs.vt, l), view, lg); color = PALETTE.obs_treated,
             marker = :diamond, markersize = 10, strokewidth = 1, strokecolor = :white, label = "PDXE observed, treated")
    axislegend(ax; position = :lt, labelsize = 11, patchsize = (18, 10), rowgap = 1, nbanks = 2,
               framevisible = true, backgroundcolor = (:white, 0.8))

    mp = MainPlot(ax, view, lg, ttl, lod)
    on(_ -> refresh_axes!(mp), view)
    on(l -> (ax.yscale = l ? log10 : identity; refresh_axes!(mp)), lg)
    refresh_axes!(mp)
    return mp
end

function refresh_axes!(mp::MainPlot)
    v = mp.view[]; ax = mp.ax
    lg = mp.logscale[]
    allv = vcat(vec(v.treated), vec(v.control), v.obs.vc, v.obs.vt)
    allv = filter(isfinite, allv)
    xlims!(ax, 0, v.t[end])
    if lg
        top = max(maximum(allv), 10 * mp.lod)
        ylims!(ax, LOG_FLOOR, top * 1.6)
    else
        top = max(maximum(v.treated[:, 5]), maximum(v.control[:, 5]), isempty(v.obs.vc) ? 0.0 : maximum(v.obs.vc),
                  isempty(v.obs.vt) ? 0.0 : maximum(v.obs.vt))
        ylims!(ax, 0, top * 1.1)
    end
    return nothing
end

set_main!(mp::MainPlot, v; title = nothing) = (title === nothing || (mp.title[] = String(title)); mp.view[] = v; mp)

# --------------------------------------------------------------------------------------------------------

const TRIAL_GROUPS = ["responder\n(mCR + mPR)", "mCR", "mPR", "mSD", "mPD"]

mutable struct TrialPlot
    ax::Axis
    tv::Observable{Any}
    shown::Observable{Bool}      # false: the bars belong to other inputs and are hidden
end

# (group x position, value, lo, hi) for one series; `series` is :current | :reference | :observed
function _trial_rows(tv, series)
    cats = ("mCR", "mPR", "mSD", "mPD")
    if series === :observed
        n = max(tv.n_observed, 1)
        f = [tv.observed[c] / n for c in cats]
        vals = [f[1] + f[2]; f]
        return vals, vals, vals
    end
    r = series === :current ? tv.current : tv.reference
    items = [r.responder; [r.cats[c] for c in cats]]
    return [x.median for x in items], [x.lo for x in items], [x.hi for x in items]
end

"""
    TrialPlot(slot, tv0; title="")

Bars (median) with 90% credible-interval whiskers of the predicted responder rate and category fractions of
a virtual PDX trial, for the current and the PDXE reference regimen, next to the observed PDXE fractions.
"""
function TrialPlot(slot, tv0; title::AbstractString = "")
    tv = Observable{Any}(tv0)
    shown = Observable(true)
    ax = Axis(slot; title = title, titlealign = :left, titlesize = 15,
              subtitle = "observed bars are context (all PDX models for the drug, published follow-up), not a matched validation",
              subtitlesize = 11, subtitlecolor = :gray35, subtitlegap = 2,
              xticks = (1:5, TRIAL_GROUPS), ylabel = "fraction of PDX models", xticklabelsize = 12)
    w = 0.26
    series = ((:current, -w, PALETTE.current, "predicted, current regimen"),
              (:reference, 0.0, PALETTE.reference, "predicted, PDXE reference regimen"),
              (:observed, w, PALETTE.observed, "observed PDXE (published categories)"))
    for (s, off, c, lab) in series
        xs = (1:5) .+ off
        vals = lift(t -> _trial_rows(t, s)[1], tv)
        barplot!(ax, xs, vals; width = 0.24, color = c, label = lab, visible = shown)
        if s !== :observed
            rangebars!(ax, xs, lift(t -> _trial_rows(t, s)[2], tv), lift(t -> _trial_rows(t, s)[3], tv);
                       color = :black, whiskerwidth = 6, linewidth = 1.3, visible = shown)
        end
        text!(ax, lift(t -> Point2f.(xs, _trial_rows(t, s)[3]), tv);
              text = lift(t -> [string(round(Int, 100 * x), "%") for x in _trial_rows(t, s)[1]], tv),
              align = (:center, :bottom), fontsize = 9, offset = (0, 2), visible = shown)
    end
    axislegend(ax; position = :lt, labelsize = 11, patchsize = (14, 10))
    text!(ax, Point2f(0.5, 0.5); text = "virtual trial out of date for these inputs: recomputing …", space = :relative,
          align = (:center, :center), fontsize = 14, color = :gray30, visible = lift(!, shown))
    tp = TrialPlot(ax, tv, shown)
    top(t) = 1.15 * maximum(maximum(_trial_rows(t, s)[3]) for s in (:current, :reference, :observed))
    ylims!(ax, 0, top(tv0))
    on(t -> ylims!(ax, 0, top(t)), tv)
    return tp
end

function set_trial!(tp::TrialPlot, tv)
    tv === nothing || (tp.tv[] = tv)
    tp.shown[] = tv !== nothing
    return tp
end

# Text summary of the observed counts, used under the plot.
function trial_summary_lines(tv)
    return join((string(c, " ", tv.observed[c]) for c in ("mCR", "mPR", "mSD", "mPD")), ", ")
end

end # module
