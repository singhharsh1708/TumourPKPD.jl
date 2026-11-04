# Synthetic stand-in fits for developing the explorer before real fits exist.
#
# These are NOT fitted. Each PosteriorDraws is built directly from plausible parameter values and the
# real PDX model ids of `drug_dataset(drug)`, so the observed PDXE overlays work. Per-model growth rates
# are crude log-linear slopes of the control curve; potencies are random. `diagnostics["demo"] = true`
# makes the app label them "DEMO - not fitted".

using TumourPKPD, Statistics, Random, JLD2

const DEMO_DRUG_SETTINGS = Dict(
    # μ_k2 is on the scale of exposure normalised to the reference-regimen Cmax (see CONTRACT.md).
    "binimetinib" => (μ_k2 = log(0.036), ω_k2 = 0.9),
    "BKM120"      => (μ_k2 = log(0.12), ω_k2 = 0.9),
    "BYL719"      => (μ_k2 = log(0.07), ω_k2 = 0.9),
)

# exponential growth rate of a control curve, from the positive volumes up to day 45
function _crude_growth(c::TumourCurve)
    k = findall(i -> c.v[i] > 0 && c.t[i] <= 45, eachindex(c.t))
    length(k) < 3 && return 0.06
    t = c.t[k]; y = log.(c.v[k])
    tm = mean(t); ym = mean(y)
    s = sum((t .- tm) .* (y .- ym)) / max(sum((t .- tm) .^ 2), eps())
    return clamp(s, 0.01, 0.4)
end

"""
    make_demo_fit(drug; S=400, seed=1) -> PosteriorDraws

Synthetic posterior draws with the structure of a real fit (see `PosteriorDraws`).
"""
function make_demo_fit(drug::AbstractString; S::Int = 400, seed::Int = 1)
    rng = MersenneTwister(seed + hash(drug) % 10_000)
    units = drug_dataset(drug)
    N = length(units)
    doses, pkp = regimen(drug)
    pk = OneCmt(pkp.CL, pkp.V, pkp.ka, pkp.F)
    cref = reference_cref(pk, doses)
    st = DEMO_DRUG_SETTINGS[drug]

    lg = [log(_crude_growth(u.control)) for u in units]
    μ_λ0 = mean(lg); ω_λ0 = max(std(lg), 0.2)
    npts = [length(u.control.t) + length(u.treated.t) for u in units]
    zk = randn(rng, N)                       # fixed "true" potency offsets

    μ_λ0s = μ_λ0 .+ 0.05 .* randn(rng, S)
    ω_λ0s = ω_λ0 .* exp.(0.08 .* randn(rng, S))
    μ_k2s = st.μ_k2 .+ 0.20 .* randn(rng, S)
    ω_k2s = st.ω_k2 .* exp.(0.12 .* randn(rng, S))
    logλ1 = log(300.0) .+ 0.6 .* randn(rng, S)
    logk1 = log(0.7) .+ 0.12 .* randn(rng, S)
    σ = 0.2 .* exp.(0.05 .* randn(rng, S))

    # per-model posteriors: wider where a model has few points (uncertainty widens where data thins)
    λ0 = Matrix{Float64}(undef, S, N); k2 = Matrix{Float64}(undef, S, N)
    for j in 1:N
        sd = clamp(1.6 / sqrt(npts[j]), 0.1, 0.7)
        λ0[:, j] .= exp.(lg[j] .+ 0.6 * sd .* randn(rng, S))
        k2[:, j] .= exp.(st.μ_k2 .+ st.ω_k2 .* zk[j] .+ 2.0 * sd .* randn(rng, S))
    end
    w0c = [max(u.control.v[1], 10.0) for u in units]
    w0t = [max(u.treated.v[1], 10.0) for u in units]
    wcd = repeat(permutedims(w0c), S, 1) .* exp.(0.05 .* randn(rng, S, N))
    wtd = repeat(permutedims(w0t), S, 1) .* exp.(0.05 .* randn(rng, S, N))

    diag = Dict{String,Any}("demo" => true,
        "note" => "synthetic parameters, not a fit",
        "max_rhat" => NaN, "min_ess_bulk" => NaN, "n_divergences" => 0,
        "nchains" => 0, "draws_per_chain" => S, "runtime_s" => NaN,
        "max_observed_day" => maximum(maximum(u.treated.t) for u in units),
        "median_followup_day" => round(median([maximum(u.treated.t) for u in units])),
        "tumour_types" => Dict(u.model => u.tumour_type for u in units),
        "lod" => detection_limit())
    return PosteriorDraws(String(drug), [u.model for u in units], collect(Dose, doses),
        fill(pk, S), float(cref), μ_λ0s, ω_λ0s, μ_k2s, ω_k2s, logλ1, logk1, σ, λ0, k2,
        w0c, w0t, wcd, wtd, diag)
end

"Write demo fits for the given drugs into `dir` (one `<drug>.jld2` each); returns the paths."
function write_demo_fits(dir::AbstractString; drugs = ["binimetinib", "BKM120", "BYL719"], S = 400)
    mkpath(dir)
    return [save_fit(joinpath(dir, "$d.jld2"), make_demo_fit(d; S)) for d in drugs]
end
