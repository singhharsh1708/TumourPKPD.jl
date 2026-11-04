# Posterior-predictive simulation of tumour trajectories from PosteriorDraws.
# Needs (from the enclosing module): Random, Statistics, PosteriorDraws (hierarchical.jl),
# PKExposure/NoDrug/Simeoni/simulate_tumour (pd.jl).

function _draw_indices(rng, nall::Int, n::Int)
    n == nall && return collect(1:nall)
    n > nall && return rand(rng, 1:nall, n)          # more requested than stored: resample
    return sort!(randperm(rng, nall)[1:n])
end

# Core: returns (y, idx, σ, λ0, k2) for the selected draws. `mkexpo(i)` builds the
# exposure object for stored draw `i`.
function _simulate(pd, mkexpo, w0default::Float64, tgrid, ndraws, model, newpdx, obs_noise,
                   rng, rtol, atol, draws = nothing)
    tg = collect(Float64, tgrid)
    (!isempty(tg) && issorted(tg) && first(tg) >= 0) ||
        throw(ArgumentError("tgrid must be non-empty, sorted and >= 0"))
    if model === nothing && !newpdx
        throw(ArgumentError("give `model=<id>` for a known PDX or `newpdx=true` for a new one"))
    elseif model !== nothing && newpdx
        throw(ArgumentError("`model` and `newpdx=true` are mutually exclusive"))
    end
    nall = length(pd.σ)
    if draws === nothing
        n = Int(ndraws)
        idx = _draw_indices(rng, nall, n)
    else
        idx = collect(Int, draws)
        all(1 .<= idx .<= nall) || throw(ArgumentError("draws must index stored posterior draws"))
        n = length(idx)
    end
    nt = length(tg)

    # All random numbers are drawn serially so results do not depend on thread count.
    zλ = newpdx ? randn(rng, n) : Float64[]
    zk = newpdx ? randn(rng, n) : Float64[]
    ε = obs_noise ? randn(rng, n, nt) : Matrix{Float64}(undef, 0, 0)

    j = 0
    if model !== nothing
        j = findfirst(==(String(model)), pd.models)
        j === nothing && throw(ArgumentError("model $(model) not in the fit; use newpdx=true"))
    end
    w0s = fill(w0default, n)

    λ0 = Vector{Float64}(undef, n)
    k2 = Vector{Float64}(undef, n)
    for r in 1:n
        i = idx[r]
        if newpdx
            λ0[r] = exp(pd.μ_λ0[i] + pd.ω_λ0[i] * zλ[r])
            k2[r] = exp(pd.μ_k2[i] + pd.ω_k2[i] * zk[r])
        else
            λ0[r] = pd.λ0[i, j]
            k2[r] = pd.k2[i, j]
        end
    end

    y = Matrix{Float64}(undef, n, nt)
    Threads.@threads for r in 1:n
        i = idx[r]
        θ = Simeoni(λ0[r], exp(pd.logλ1[i]), exp(pd.logk1[i]), k2[r], w0s[r])
        w, ok = simulate_tumour(θ, mkexpo(i), tg; reltol = rtol, abstol = atol)
        if ok
            if obs_noise
                σ = pd.σ[i]
                for c in 1:nt
                    # baseline (t = 0) is conditioned on, so it carries no noise
                    y[r, c] = tg[c] == 0 ? w[c] : w[c] * exp(σ * ε[r, c])
                end
            else
                y[r, :] .= w
            end
        else
            y[r, :] .= NaN
        end
    end
    return (; y, idx, σ = pd.σ[idx], λ0, k2)
end

function _w0default(w0, ws::Vector{Float64}, pd, model, newpdx)
    w0 !== nothing && return float(w0)
    if model !== nothing
        j = findfirst(==(String(model)), pd.models)
        j === nothing && throw(ArgumentError("model $(model) not in the fit; use newpdx=true"))
        return ws[j]
    end
    return float(median(ws))
end

"""
    simulate_regimen(pd::PosteriorDraws, doses; model=nothing, w0=nothing, tgrid,
                     ndraws=200, newpdx=false, obs_noise=true, rng=Random.default_rng(),
                     rtol=1e-6, atol=1e-6, details=false)

Posterior-predictive treated-arm tumour volumes (mm^3), a `ndraws × length(tgrid)`
matrix. Each row uses one stored posterior draw: its PK (`pd.pk[i]`), `cref`, `logλ1`,
`logk1`, `σ`, and either the PDX model's own `λ0`/`k2` (`model="X-1234"`) or fresh
random effects `exp(μ + ω z)`, `z ~ N(0,1)` (`newpdx=true`, a population prediction for
an unseen PDX). `doses` are given in the same units as the fit (mg/kg, days); exposure is
`conc/cref` so `doses` may differ from the fitted reference regimen.

`w0` defaults to the model's fitted treated baseline (`pd.w0_treated`), or the median
over models for a new PDX. `obs_noise=true` multiplies by lognormal noise
`exp(σ ε)` at every `t > 0` (the baseline is conditioned on). Rows whose ODE solve fails
are filled with NaN. With `details=true` returns `(y, idx, σ, λ0, k2)` where `idx` are the
stored-draw indices used (needed for log predictive density with `obs_noise=false`).

`draws` (a vector of stored-draw indices) overrides `ndraws`/random selection: pass the same
`draws` for every PDX model so that row `s` is the same population draw everywhere (needed for
joint quantities such as the responder fraction); fresh new-PDX random effects are still drawn
per call from `rng`. Using the same `rng` seed with `obs_noise=true` and `false` gives the same
latent paths (random effects are drawn before the noise).
"""
function simulate_regimen(pd, doses::AbstractVector{Dose}; model = nothing, w0 = nothing,
                          tgrid, ndraws = 200, newpdx::Bool = false, obs_noise::Bool = true,
                          rng::AbstractRNG = Random.default_rng(), rtol = 1e-6, atol = 1e-6,
                          details::Bool = false, draws = nothing)
    w0d = _w0default(w0, pd.w0_treated, pd, model, newpdx)
    ds = collect(Dose, doses)
    cref = pd.cref
    mk = i -> PKExposure(pd.pk[i], ds, cref)
    r = _simulate(pd, mk, w0d, tgrid, ndraws, model, newpdx, obs_noise, rng, rtol, atol, draws)
    return details ? r : r.y
end

"""
    simulate_control(pd::PosteriorDraws; model=nothing, w0=nothing, tgrid, ndraws=200,
                     newpdx=false, obs_noise=true, rng, rtol, atol, details=false)

Untreated (NoDrug) predictive trajectories; same conventions as [`simulate_regimen`](@ref).
`w0` defaults to `pd.w0_control`. `draws` as in `simulate_regimen`.
"""
function simulate_control(pd; model = nothing, w0 = nothing, tgrid, ndraws = 200,
                          newpdx::Bool = false, obs_noise::Bool = true,
                          rng::AbstractRNG = Random.default_rng(), rtol = 1e-6, atol = 1e-6,
                          details::Bool = false, draws = nothing)
    w0d = _w0default(w0, pd.w0_control, pd, model, newpdx)
    r = _simulate(pd, _ -> NoDrug(), w0d, tgrid, ndraws, model, newpdx, obs_noise, rng, rtol, atol, draws)
    return details ? r : r.y
end

"""
    predictive_bands(sims; levels=(0.05, 0.5, 0.95))

Column-wise quantiles (ignoring NaN rows) of a `ndraw × ntime` simulation matrix.
Returns a `ntime × length(levels)` matrix.
"""
function predictive_bands(sims::AbstractMatrix; levels = (0.05, 0.5, 0.95))
    out = Matrix{Float64}(undef, size(sims, 2), length(levels))
    for c in axes(sims, 2)
        col = filter(isfinite, @view sims[:, c])
        for (k, q) in enumerate(levels)
            out[c, k] = isempty(col) ? NaN : quantile(col, q)
        end
    end
    return out
end
