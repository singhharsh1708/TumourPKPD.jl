using Test, TumourPKPD, DataFrames, Statistics

# The lead includes src/data.jl in the module; until then load it into the module for this test.
if !isdefined(TumourPKPD, :load_pct)
    Core.eval(TumourPKPD, :(include(joinpath($(pkgdir(TumourPKPD)), "src", "data.jl"))))
end
const TP = TumourPKPD

const CACHE = TP.default_cache_dir()
const HAVE_CACHE = isfile(joinpath(CACHE, "pct_long.csv")) && isfile(joinpath(CACHE, "pct_metrics.csv"))
HAVE_CACHE || @warn "cleaned PDXE cache missing; run data/download.jl and data/clean.jl. Data tests skipped." CACHE

@testset "regimens.toml" begin
    @test TP.detection_limit() == TP.DETECTION_LIMIT_MM3
    @test sort(TP.list_drugs()) == ["BKM120", "BYL719", "binimetinib"]
    for d in TP.list_drugs()
        doses, pr = TP.regimen(d)
        @test !isempty(doses)
        @test all(x -> x.route === :oral && x.amt > 0, doses)
        @test issorted([x.time for x in doses]) && doses[1].time == 0.0
        @test all(>(0), (pr.CL, pr.V, pr.ka, pr.F, pr.CL_logsd, pr.V_logsd, pr.ka_logsd, pr.F_logitsd))
        @test 0 < pr.F <= 1
        @test pr.dose_status in ("assumption", "verified for NSCLC, inferred for other indications")
    end
    d1, _ = TP.regimen("BKM120"; duration = 10)
    @test length(d1) == 10 && d1[2].time == 1.0
    d2, _ = TP.regimen("binimetinib"; duration = 10)
    @test length(d2) == 20 && d2[2].time == 0.5 && d2[1].amt == 30.0
    # unit sanity: BKM120 CL 11 mL/min/kg = 15.84 L/day/kg; BYL719 0.48 L/h/kg = 11.52 L/day/kg
    @test TP.regimen("BKM120")[2].CL ≈ 11 * 60 * 24 / 1000
    @test TP.regimen("BYL719")[2].CL ≈ 0.48 * 24
    @test_throws ArgumentError TP.regimen("nonexistent")
    # PK prior can drive the lead's PK code
    pk = TP.OneCmt(; CL = TP.regimen("BYL719")[2].CL, V = TP.regimen("BYL719")[2].V,
        ka = TP.regimen("BYL719")[2].ka, F = TP.regimen("BYL719")[2].F)
    dd, _ = TP.regimen("BYL719"; duration = 5)
    c = TP.conc(pk, dd, 2.5)
    @test isfinite(c) && c > 0
end

if HAVE_CACHE
    @testset "pct_long.csv schema" begin
        long = TP._read_long(CACHE)
        for c in ("model", "tumour_type", "treatment", "day", "volume_mm3", "replicate", "detected", "t0_shifted")
            @test c in names(long)
        end
        @test nrow(long) == 67445
        @test !any(isnan, long.volume_mm3) && !any(isnan, long.day) && !any(isnan, long.pct_tvol_diff)
        @test all(long.volume_mm3 .>= 0) && all(long.day .>= 0)
        @test count(==(0.0), long.volume_mm3) == 2255
        @test all(long.detected .== (long.volume_mm3 .> 0))
        @test !any(occursin("\"", t) for t in long.treatment)
        @test "untreated" in long.treatment
        @test Set(long.tumour_type) ⊆ Set(["BRCA", "CM", "CRC", "GC", "NSCLC", "PDAC", "UNKNOWN"])
        @test length(unique(long.model)) == 277
        # no duplicate (model, treatment, replicate, day)
        @test nrow(unique(long[:, [:model, :treatment, :replicate, :day]])) == nrow(long)
        # replicates only for the three multi-animal control sets
        multi = unique(long[long.replicate .> 1, [:model, :treatment]])
        @test sort(multi.model) == ["X-025", "X-158", "X-166"] && all(multi.treatment .== "untreated")
        # day 0 volume present and positive for all curves that start at day 0 as recorded
        for g in groupby(long, [:model, :treatment, :replicate])
            @test g.day[1] == 0.0                       # re-based: every curve starts at 0
            @test issorted(g.day)
            g.day_raw[1] == 0.0 && @test g.volume_mm3[1] > 0
        end
        # curves whose first recorded day is not 0 (documented in data/README.md): 7 curves, all re-based to 0
        shifted = unique(long[long.t0_shifted, [:model, :treatment, :replicate]])
        @test nrow(shifted) == 7
    end

    @testset "load_pct / TumourCurve" begin
        cs = TP.load_pct(; dir = CACHE)
        @test eltype(cs) == TP.TumourCurve
        @test length(cs) == 4706
        @test all(c -> c.t[1] == 0.0 && issorted(c.t) && length(c.t) == length(c.v), cs)
        @test all(c -> all(isfinite, c.v) && all(>=(0), c.v), cs)
        # X-025, X-158, X-166 each have 3 control animals: 6 extra curves with :all
        @test length(TP.load_pct(; dir = CACHE, replicates = :all)) == 4706 + 6
        @test length(TP.load_pct(; dir = CACHE, treatments = ["BKM120"])) == 224
        @test length(TP.load_pct(; dir = CACHE, min_points = 3)) < length(cs)
    end

    @testset "load_metrics" begin
        m = TP.load_metrics(; dir = CACHE)
        for c in ("model", "tumour_type", "treatment", "response_category", "best_response", "best_avg_response")
            @test c in names(m)
        end
        @test nrow(m) == 4758
        @test Set(m.response_category) == Set(["mCR", "mPR", "mSD", "mPD"])
        @test !any(isnan, m.best_response) && !any(isnan, m.best_avg_response)
        # category must follow the published mRECIST thresholds (Gao 2015 / Xeva)
        rule(b, a) = b < -95 && a < -40 ? "mCR" : b < -50 && a < -20 ? "mPR" : b < 35 && a < 30 ? "mSD" : "mPD"
        @test all(rule.(m.best_response, m.best_avg_response) .== m.response_category)
        @test all(m.progressed .== occursin.("-->", m.response_trajectory))
        @test count(.!m.has_curve) == 52   # abraxane + gemcitabine rows without raw curves
    end

    @testset "drug_dataset" begin
        audit = TP.CSV.read(joinpath(CACHE, "treatment_audit.csv"), DataFrame)
        expect = Dict("BKM120" => 197, "BYL719" => 184, "binimetinib" => 200)
        for d in TP.list_drugs()
            ds = TP.drug_dataset(d; dir = CACHE)
            @test length(ds) == expect[d]
            @test length(ds) == audit.n_paired[findfirst(==(d), audit.treatment)]
            @test length(unique(u.model for u in ds)) == length(ds)
            for u in ds
                @test u.control.model == u.model == u.treated.model
                @test u.control.treatment == "untreated" && u.treated.treatment == d
                @test u.control.t[1] == 0.0 && u.treated.t[1] == 0.0       # treatment start, no shifted curves allowed
                @test length(u.control.t) >= 3 && length(u.treated.t) >= 3
                @test issorted(u.control.t) && issorted(u.treated.t)
                @test u.control.v[1] > 0 && u.treated.v[1] > 0              # day-0 volume always measurable
                @test !any(isnan, u.control.v) && !any(isnan, u.treated.v) && !any(isnan, u.control.t) && !any(isnan, u.treated.t)
                @test all(>=(0), u.control.v) && all(>=(0), u.treated.v)
                @test u.tumour_type == u.treated.tumour_type && u.tumour_type != "UNKNOWN"
            end
            # at least some responding units exist per drug
            mt = TP.load_metrics(; dir = CACHE)
            cats = [mt.response_category[findfirst((mt.model .== u.model) .& (mt.treatment .== d))] for u in ds]
            @test count(in(("mCR", "mPR")), cats) >= 5
            @test count(==("mPD"), cats) >= 50
        end
        # volumes of exactly 0 exist in treated arms and are flagged as censored by the caller
        @test any(any(==(0.0), u.treated.v) for u in TP.drug_dataset("binimetinib"; dir = CACHE))
        # unknown treatment: warns, returns empty
        @test isempty(@test_logs (:warn, r"no paired") TP.drug_dataset("nonexistent"; dir = CACHE))
    end
end
