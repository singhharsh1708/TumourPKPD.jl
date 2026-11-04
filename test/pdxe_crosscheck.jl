# Recompute PDXE response categories from raw curves and compare with the paper's reported
# values (data/cache/pct_long.csv vs pct_metrics.csv). Included from test_validate.jl, or run
# directly:  julia --project=TumourPKPD.jl test/pdxe_crosscheck.jl
using CSV, DataFrames, Printf, Statistics

function pdxe_crosscheck(cache; verbose = true, daycol = :day, kw...)
    TP = TumourPKPD
    long = CSV.read(joinpath(cache, "pct_long.csv"), DataFrame)
    met = CSV.read(joinpath(cache, "pct_metrics.csv"), DataFrame)
    rows = NamedTuple[]
    nskip_dup = 0
    # replicate animals (replicate > 1, a few dozen rows) are ambiguous w.r.t. the per-(model, treatment)
    # paper metrics; classify replicate 1 only
    hasrep = "replicate" in names(long)
    hasrep && (long = long[long.replicate .== 1, :])
    for g in groupby(long, [:model, :treatment])
        d = g[.!ismissing.(g.volume_mm3) .& .!ismissing.(g[!, daycol]), :]
        nrow(d) == 0 && continue
        if length(unique(d[!, daycol])) != nrow(d)          # replicate animals: ambiguous
            nskip_dup += 1
            continue
        end
        r = TP.pdxe_response(Float64.(d[!, daycol]), Float64.(d.volume_mm3); kw...)
        push!(rows, (model = String(g.model[1]), treatment = String(g.treatment[1]),
                     our_cat = r.category, our_best = r.best_response, our_bavg = r.best_avg_response))
    end
    ours = DataFrame(rows)
    met2 = select(met, :model, :treatment, :response_category => :paper_cat,
                  :best_response => :paper_best, :best_avg_response => :paper_bavg)
    j = innerjoin(ours, met2; on = [:model, :treatment])
    j = j[.!ismissing.(j.paper_cat), :]
    j.paper_cat = String.(j.paper_cat)
    agree = mean(j.our_cat .== j.paper_cat)
    resp_agree = mean(TP.is_responder.(j.our_cat[j.our_cat .!= "NA"]) .== TP.is_responder.(j.paper_cat[j.our_cat .!= "NA"]))
    cats = TP.RESPONSE_CATEGORIES
    conf = [count((j.paper_cat .== a) .& (j.our_cat .== b)) for a in cats, b in cats]
    ok = .!ismissing.(j.paper_best) .& .!ismissing.(j.paper_bavg) .& isfinite.(j.our_best)
    dbest = abs.(Float64.(j.paper_best[ok]) .- j.our_best[ok])
    dbavg = abs.(Float64.(j.paper_bavg[ok]) .- j.our_bavg[ok])
    res = (n_compared = nrow(j), responder_agreement = resp_agree, n_curves = nrow(ours), n_skipped_replicates = nskip_dup,
           agreement = agree, confusion = conf, categories = cats,
           median_abs_diff_best = isempty(dbest) ? NaN : median(dbest),
           frac_best_within_1pct = isempty(dbest) ? NaN : mean(dbest .< 1),
           median_abs_diff_bavg = isempty(dbavg) ? NaN : median(dbavg),
           frac_bavg_within_1pct = isempty(dbavg) ? NaN : mean(dbavg .< 1),
           mismatches = j[j.our_cat .!= j.paper_cat, :])
    if verbose
        @printf("PDXE classifier cross-check %s: %d (model,treatment) curves compared, 4-way agreement = %.2f%%, responder/non-responder agreement = %.2f%%\n",
                string(kw), res.n_compared, 100 * agree, 100 * resp_agree)
        println("rows = paper category, cols = ours, order ", cats); display(conf)
        @printf("best response within 1 pct-point: %.1f%%, best avg response: %.1f%%\n",
                100 * res.frac_best_within_1pct, 100 * res.frac_bavg_within_1pct)
    end
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    using TumourPKPD
    pdxe_crosscheck(joinpath(@__DIR__, "..", "data", "cache"))
end
