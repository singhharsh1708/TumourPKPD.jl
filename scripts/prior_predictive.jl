# Prior predictive check of the growth-side priors (μ_λ0, ω_λ0, transition volume Vtr = λ1/λ0) against the
# observed binimetinib control curves inside the 42-day fit window.
#   julia --project=. scripts/prior_predictive.jl [drug] [window] [n]
using TumourPKPD, Random, Statistics, Printf
isdefined(TumourPKPD, :fit_drug) || Base.include(TumourPKPD, joinpath(@__DIR__, "..", "src", "hierarchical.jl"))
const H = TumourPKPD
drug = length(ARGS) >= 1 ? ARGS[1] : "binimetinib"
window = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 42.0
nsim = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4000
units = H.window_units(drug_dataset(drug), window)
ctl = [u.control for u in units]
q(x, ps = (0.05, 0.25, 0.5, 0.75, 0.95)) = round.(quantile(x, collect(ps)); digits = 0)
println("# $drug: $(length(ctl)) control curves, window day <= $window")
v0 = [c.v[1] for c in ctl]
vmax = [maximum(c.v) for c in ctl]
vlast = [c.v[end] for c in ctl]
tlast = [c.t[end] for c in ctl]
println("v0 quantiles (5,25,50,75,95%):       ", q(v0))
println("max observed volume:                 ", q(vmax), "   fraction > 1000: ", round(mean(vmax .> 1000); digits = 2), ", > 2000: ", round(mean(vmax .> 2000); digits = 2))
println("last observed volume:                ", q(vlast))
println("last observation day:                ", q(tlast))
# empirical growth deceleration: log-slope in the first and second half of each control (>= 6 obs)
sl1 = Float64[]; sl2 = Float64[]
for c in ctl
    pos = c.v .> 0
    t = c.t[pos]; y = log.(c.v[pos])
    length(t) >= 6 || continue
    h = length(t) ÷ 2
    slope(a, b) = (m = (a .- mean(a)); sum(m .* (b .- mean(b))) / max(sum(m .^ 2), eps()))
    push!(sl1, slope(t[1:h], y[1:h])); push!(sl2, slope(t[h:end], y[h:end]))
end
println("median log-slope first half / second half of each control (1/day): ", round(median(sl1); digits = 3), " / ", round(median(sl2); digits = 3),
        "   fraction decelerating: ", round(mean(sl2 .< sl1); digits = 2))
# volume at which decelerating controls are, at the half-way point
# prior predictive
pri = H.default_priors()
rng = MersenneTwister(1)
long = [c for c in ctl if c.t[end] >= 0.8 * window]
println("controls followed to >= $(round(0.8 * window)) days: ", length(long))
simlast = Float64[]; simmax = Float64[]; inrange = Bool[]
lo, hi = minimum(vmax), maximum(vmax)
for _ in 1:nsim
    μ = pri.μ_λ0[1] + pri.μ_λ0[2] * randn(rng)
    ω = abs(pri.ω_λ0 * randn(rng))
    lvtr = pri.logVtr[1] + pri.logVtr[2] * randn(rng)
    c = long[rand(rng, 1:length(long))]
    λ0 = exp(μ + ω * randn(rng))
    λ1 = exp(μ + lvtr)                      # λ1 = exp(μ_λ0) · Vtr
    w, ok = simulate_tumour(Simeoni(λ0, λ1, 1.0, 0.0, c.v[1]), NoDrug(), collect(c.t[2:end]))
    ok || continue
    push!(simlast, w[end]); push!(simmax, maximum(w)); push!(inrange, maximum(w) <= 2 * hi)
end
obslast = [c.v[end] for c in long]
println("prior predictive volume at the last observation day: ", q(simlast))
println("observed volume at the last observation day (n=$(length(long))): ", q(obslast))
println("fraction of prior-predictive control trajectories with max <= 2x the largest observed volume ($(round(2hi))): ", round(mean(inrange); digits = 3))
println("fraction with last volume inside the observed [min, max] of last volumes ($(round(minimum(obslast))), $(round(maximum(obslast)))): ",
        round(mean(minimum(obslast) .<= simlast .<= maximum(obslast)); digits = 3))
println("fraction with last volume > 10000 mm^3: ", round(mean(simlast .> 1e4); digits = 3), "   < v0/3 (shrinking controls): ", round(mean(simlast .< 60); digits = 3))
println("prior of Vtr 95% interval (mm^3): ", round.(exp.(pri.logVtr[1] .+ [-1.96, 1.96] .* pri.logVtr[2]); digits = 0),
        "   prior λ1 median (mm^3/day): ", round(exp(pri.μ_λ0[1] + pri.logVtr[1]); digits = 1))
