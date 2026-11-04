# Clean the Novartis PDX Encyclopedia (Gao et al. 2015, Nat Med 21:1318) PCT sheets.
#
#   julia --project=. data/clean.jl [raw_xlsx_or_dir] [out_dir]
#
# Reads only the two PCT sheets of nm3954_S2.xlsx (the other sheets are ~600 MB of XML and are never touched)
# and writes, into out_dir (default data/cache):
#   pct_long.csv         one row per measurement, snake_case columns, replicate and day-0 handling made explicit
#   pct_metrics.csv      curve metrics, response categories mapped to mCR/mPR/mSD/mPD
#   treatment_audit.csv  per single-agent treatment: paired models, tumour types, response counts
#
# Every decision taken here is documented in data/README.md.

using XLSX, DataFrames, CSV, Statistics

const HERE = @__DIR__
const DEFAULT_OUT = joinpath(HERE, "cache")
const RAW_NAME = "nm3954_S2.xlsx"
const MIN_POINTS = 3   # a curve needs >= 3 observations to enter the audit pairing / drug datasets

function find_raw(arg::Union{Nothing,String})
    cands = String[]
    if arg !== nothing
        push!(cands, isdir(arg) ? joinpath(arg, RAW_NAME) : arg)
    end
    haskey(ENV, "TUMOURPKPD_RAW_DIR") && push!(cands, joinpath(ENV["TUMOURPKPD_RAW_DIR"], RAW_NAME))
    push!(cands, joinpath(HERE, "cache", "raw", RAW_NAME))
    push!(cands, joinpath(homedir(), "Work", "tumour-pkpd", "downloads", "pdxe", RAW_NAME))
    for c in cands
        isfile(c) && return c
    end
    error("could not find $RAW_NAME; run data/download.jl or pass the path. Tried: $(join(cands, ", "))")
end

# -- reading ---------------------------------------------------------------------------------------------
function read_sheet(path, sheet)
    XLSX.openxlsx(path; enable_cache = false) do xf
        DataFrame(XLSX.gettable(xf[sheet]; infer_eltypes = false))
    end
end

fixname(x) = replace(strip(String(x)), "\"" => "")   # raw sheet has a stray quote: figitumumab"
asfloat(x) = x isa Real ? Float64(x) : (x === missing ? NaN : parse(Float64, strip(String(x))))
asstr(x) = x === missing ? "" : String(x)

# -- long table ---------------------------------------------------------------------------------------------
function clean_long(raw::DataFrame)
    df = DataFrame(
        model = [String(x) for x in raw[!, "Model"]],
        tumor_type_raw = [asstr(x) for x in raw[!, "Tumor Type"]],
        treatment_raw = [String(x) for x in raw[!, "Treatment"]],
        day_raw = [asfloat(x) for x in raw[!, "Days Post T0"]],
        volume_mm3 = [asfloat(x) for x in raw[!, "Volume (mm3)"]],
        body_weight_g = [asfloat(x) for x in raw[!, "body weight (g)"]],
        pct_tvol_diff = [asfloat(x) for x in raw[!, "% TVol Difference"]],
        pct_bw_diff = [asfloat(x) for x in raw[!, "% BW Difference"]],
    )
    df.tumour_type = [isempty(x) ? "UNKNOWN" : x for x in df.tumor_type_raw]
    df.treatment = fixname.(df.treatment_raw)
    n = nrow(df)

    # replicate index: file order inside (model, treatment); a day that does not increase starts a new animal
    rep = ones(Int, n)
    t0 = zeros(n)
    for g in groupby(df, [:model, :treatment_raw])
        idx = parentindices(g)[1]
        r = 1
        start = df.day_raw[idx[1]]
        rep[idx[1]] = 1
        t0[idx[1]] = start
        for k in 2:length(idx)
            if df.day_raw[idx[k]] <= df.day_raw[idx[k-1]]
                r += 1
                start = df.day_raw[idx[k]]
            end
            rep[idx[k]] = r
            t0[idx[k]] = start
        end
    end
    df.replicate = rep
    df.day = df.day_raw .- t0                  # re-based so each animal's first observation is day 0
    df.t0_shifted = t0 .!= 0.0
    df.detected = df.volume_mm3 .> 0
    df.arm = [t == "untreated" ? "untreated" : occursin('+', t) ? "combo" : "single" for t in df.treatment]
    df.row_id = 1:n
    select!(df, :model, :tumour_type, :treatment, :arm, :replicate, :day, :day_raw, :t0_shifted,
        :volume_mm3, :detected, :body_weight_g, :pct_tvol_diff, :pct_bw_diff, :treatment_raw, :row_id)
    return df
end

# -- metrics ---------------------------------------------------------------------------------------------
function clean_metrics(rawm::DataFrame, tt::Dict{String,String}, curvekeys)
    m = DataFrame(
        model = [String(x) for x in rawm[!, "Model"]],
        treatment = fixname.(rawm[!, "Treatment"]),
        treatment_type = [String(x) for x in rawm[!, "Treatment type"]],
        target = [asstr(x) for x in rawm[!, "Treatment target"]],
        best_response = [asfloat(x) for x in rawm[!, "BestResponse"]],
        day_best_response = [asfloat(x) for x in rawm[!, "Day_BestResponse"]],
        best_avg_response = [asfloat(x) for x in rawm[!, "BestAvgResponse"]],
        day_best_avg_response = [asfloat(x) for x in rawm[!, "Day_BestAvgResponse"]],
        time_to_double = [asfloat(x) for x in rawm[!, "TimeToDouble"]],
        day_last = [asfloat(x) for x in rawm[!, "Day_Last"]],
        response_trajectory = [String(x) for x in rawm[!, "ResponseCategory"]],
    )
    # sheet categories look like "SD", "PR-->PD", "CR-->-->PD": first token = mRECIST class from
    # (BestResponse, BestAvgResponse); the "-->PD" tail = later progression. Verified to equal the rule below
    # for all rows (see data/README.md).
    first_tok = [String(split(x, "-->")[1]) for x in m.response_trajectory]
    m.response_category = ["m" * x for x in first_tok]
    m.progressed = [occursin("-->", x) for x in m.response_trajectory]
    m.tumour_type = [get(tt, x, "UNKNOWN") for x in m.model]
    m.has_curve = [(a, b) in curvekeys for (a, b) in zip(m.model, m.treatment)]
    select!(m, :model, :tumour_type, :treatment, :treatment_type, :response_category, :best_response, :best_avg_response,
        :response_trajectory, :progressed, :day_best_response, :day_best_avg_response, :time_to_double, :day_last,
        :target, :has_curve)
    return m
end

mrecist(b, a) = b < -95 && a < -40 ? "mCR" : b < -50 && a < -20 ? "mPR" : b < 35 && a < 30 ? "mSD" : "mPD"

# -- pairing + audit ---------------------------------------------------------------------------------------------
"""
First replicate of every (model, treatment) curve that has >= MIN_POINTS observations and starts at raw day 0.
Returns Dict((model, treatment) => (tumour_type, days, volumes)).
"""
function usable_curves(long::DataFrame)
    out = Dict{Tuple{String,String},Tuple{String,Vector{Float64},Vector{Float64}}}()
    for g in groupby(long, [:model, :treatment])
        g1 = g[g.replicate .== 1, :]
        (nrow(g1) >= MIN_POINTS && !g1.t0_shifted[1]) || continue
        out[(g1.model[1], g1.treatment[1])] = (g1.tumour_type[1], g1.day, g1.volume_mm3)
    end
    return out
end

function treatment_audit(long::DataFrame, met::DataFrame)
    uc = usable_curves(long)
    metk = Dict((r.model, r.treatment) => r for r in eachrow(met))
    treatments = sort(unique(long.treatment[long.arm .== "single"]))
    rows = NamedTuple[]
    allctl = Set(k[1] for k in keys(uc) if k[2] == "untreated")
    anyctl = Set(long.model[long.treatment .== "untreated"])
    for tr in treatments
        anyt = Set(long.model[long.treatment .== tr])
        both_any = intersect(anyt, anyctl)
        paired = sort(collect(intersect(Set(k[1] for k in keys(uc) if k[2] == tr), allctl)))
        tts = Dict{String,Int}()
        cats = Dict("mCR" => 0, "mPR" => 0, "mSD" => 0, "mPD" => 0)
        nzero = 0
        fu = Float64[]
        for m in paired
            tt, d, v = uc[(m, tr)]
            tts[tt] = get(tts, tt, 0) + 1
            any(==(0.0), v) && (nzero += 1)
            push!(fu, maximum(d))
            haskey(metk, (m, tr)) && (cats[metk[(m, tr)].response_category] += 1)
        end
        push!(rows, (treatment = tr, n_models_any = length(anyt), n_with_both_arms_any = length(both_any),
            n_paired = length(paired),
            tumour_types = join(["$k:$(tts[k])" for k in sort(collect(keys(tts)))], ";"),
            n_mCR = cats["mCR"], n_mPR = cats["mPR"], n_mSD = cats["mSD"], n_mPD = cats["mPD"],
            n_models_with_zero_volume = nzero,
            median_followup_day = isempty(fu) ? NaN : median(fu)))
    end
    a = DataFrame(rows)
    sort!(a, :n_paired, rev = true)
    return a
end

function main(args = ARGS)
    rawpath = find_raw(length(args) >= 1 ? args[1] : nothing)
    outdir = length(args) >= 2 ? args[2] : DEFAULT_OUT
    mkpath(outdir)
    @info "reading" rawpath
    rawpct = read_sheet(rawpath, "PCT raw data")
    rawmet = read_sheet(rawpath, "PCT curve metrics")
    long = clean_long(rawpct)
    tt = Dict{String,String}()
    for r in eachrow(long)
        tt[r.model] = r.tumour_type
    end
    curvekeys = Set(zip(long.model, long.treatment))
    met = clean_metrics(rawmet, tt, curvekeys)

    # consistency check of the sheet's category against the published thresholds
    bad = count(r -> r.response_category != mrecist(r.best_response, r.best_avg_response), eachrow(met))
    bad == 0 || @warn "response_category disagrees with mRECIST thresholds" bad

    audit = treatment_audit(long, met)
    CSV.write(joinpath(outdir, "pct_long.csv"), long)
    CSV.write(joinpath(outdir, "pct_metrics.csv"), met)
    CSV.write(joinpath(outdir, "treatment_audit.csv"), audit)
    @info "wrote" outdir nrow(long) nrow(met) nrow(audit)
    return long, met, audit
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
