# PDXE data

Source: Gao H, Korn JM, Ferretti S, et al. High-throughput screening using patient-derived tumor xenografts to predict
clinical trial drug response. Nat Med 21:1318-1325 (2015), doi:10.1038/nm.3954. Supplementary Table 1
(`nm3954_S2.xlsx`, sheets `PCT raw data` and `PCT curve metrics`) and the Supplementary Information PDF.
Raw files are not redistributed here; `data/download.jl` fetches them from the Springer static-content URLs and
verifies SHA-256.

```
julia --project=. data/download.jl     # raw xlsx + pdf -> ~/Work/tumour-pkpd/downloads/pdxe (or data/cache/raw)
julia --project=. data/clean.jl        # -> data/cache/{pct_long,pct_metrics,treatment_audit}.csv   (~10 s)
```

`data/cache/` is gitignored. `data/regimens.toml` is committed.

## The PCT design

One animal per PDX model per arm (the paper's "1 x 1 x 1" design: 1 animal x 1 model x 1 treatment). Each model has
at most one untreated control animal that is shared by all its treatment arms. Exceptions: three models (X-025, X-158,
X-166) have three control animals each.

## Raw columns (sheet `PCT raw data`, 67,445 rows)

| raw column | cleaned | meaning |
|---|---|---|
| Model | `model` | PDX model id, `X-nnn[n]`; 277 models |
| Tumor Type | `tumour_type` | BRCA, CM (melanoma), CRC, GC, NSCLC, PDAC. Blank for models X-1468 and X-3782 (38 rows) -> `UNKNOWN`. A model has one type everywhere. |
| Treatment | `treatment` | 61 treatment labels + `untreated`. 37 single-agent labels, 24 combinations (contain ` + `). |
| Volume (mm3) | `volume_mm3` | tumour volume, caliper-based, mm^3 |
| body weight (g) | `body_weight_g` | missing in 128 rows |
| Days Post T0 | `day_raw`, `day` | integer days since T0; `day` is re-based so each animal's first row is day 0 |
| % TVol Difference | `pct_tvol_diff` | 100 (V - V_first)/V_first, relative to the animal's first row. Matches the volumes to within 0.05 percentage points of rounding. |
| % BW Difference | `pct_bw_diff` | same for body weight |

Added columns: `arm` (untreated/single/combo), `replicate` (animal index within a model/treatment), `t0_shifted`,
`detected` (`volume_mm3 > 0`), `treatment_raw`, `row_id` (row order in the sheet).

## Audit findings

**Identifiers and labels.**
- `untreated` is the control label (223 models, 229 animals).
- One label has a stray double quote in the sheet (`figitumumab"`); it is stripped (`treatment_raw` keeps the original).
- Two labels encode the dose: `binimetinib-3.5mpk` and `gemcitabine-50mpk`. They are separate arms from `binimetinib` / the other gemcitabine use; they are NOT merged.
- The metrics sheet has 52 rows for `abraxane + gemcitabine` that have no raw curve, and 4 models (X-055, X-080, X-152, X-2220) that appear only there. They are kept in `pct_metrics.csv` with `has_curve=false`.
- A model has one tumour type throughout, but some arms contain a one-off model of another type (e.g. one CRC model inside the PDAC-only arms).

**Day 0 and time.**
- 4,705 of 4,712 animal curves (4,706 model/treatment pairs) start at `Days Post T0 = 0` with `% TVol Difference = 0`. Day 0 is treated as treatment start: the paper's metrics (BestResponse, BestAvgResponse) are relative to this baseline, and baseline volumes are tightly distributed (median 220 mm^3, 5th-95th percentile 187-298; range 106-1,180), consistent with size-triggered enrolment. The sheet itself does not define T0; this reading is an assumption.
- 7 curves start later (first raw day 7-29): X-1870 (BKM120 + binimetinib), X-2042 (binimetinib, paclitaxel), X-2088 (CGM097, one point), X-2524, X-4018 (the baseline is not even 0%), X-4087 (BYL719 + cetuximab). They are re-based to start at 0 and flagged `t0_shifted`; `drug_dataset` excludes them (this removes only X-2042 binimetinib, 2 points, from binimetinib).
- Sampling is every 3-4 days (median 3.5); curves have 1-72 points (median 11). Follow-up: control animals median 33 days (max 165), treated animals median 39 days (max 404). Animals leave the study at progression/size limit, so end of follow-up is informative (controls end earlier than treated).
- Days are non-decreasing in every curve except in the three multi-animal control sets (days restart at 0 for each animal, handled by `replicate`). No negative days. No fractional days.

**Duplicates.** 33 (model, treatment, day) keys are duplicated, all from the three multi-animal control sets (X-025, X-158, X-166; 3 animals each). With the `replicate` index there are no duplicates. `load_pct`/`drug_dataset` use replicate 1 (arbitrary but deterministic; `load_pct(replicates=:all)` returns all).

**Volumes.** No missing volumes. No negative volumes. 2,255 rows (3.3%) have volume exactly 0: in 134 curves (2.8% of curves), none at day 0, 13 in a single control animal (X-1823 untreated, which grows to 553 mm^3 then regresses to 0 by day 42 -- spontaneous regression; flag for the inference worker), 1,167 in single-agent arms, 1,075 in combinations. `% TVol Difference` is -100 for all of them (and for two rows of 0.1 mm^3).
- Zeros are not NA codes: they are the recorded "no measurable tumour". 44 of the 134 curves later have a positive volume again, so a zero means undetectable, not cured. In 106 curves the last observation is 0.
- Around zeros the volumes are small but not tiny: the last positive reading before the first zero has median 25 mm^3 (10th percentile 8.6); the first positive reading after a zero has median 23 mm^3 (10th percentile 2.8).
- Small positive values exist: 27 of 65,190 positive readings are < 4 mm^3, 10 are exactly 4.0.
- **Detection limit: 4 mm^3 (assumption).** Rationale: (i) a lesion of 2 x 2 x 2 mm has caliper volume (L W^2)/2 = 4 mm^3 and 10 readings equal exactly 4.0, a visible spike in the lowest decade (the formula is the usual caliper one; the paper's Methods are paywalled so I could not confirm it); (ii) it sits below the 10th percentile of readings adjacent to zeros (about 3-9 mm^3), so zeros are plausibly lower than anything the study measured; (iii) it is under 2% of the median baseline (220 mm^3), and only 3% of rows are zero, so the choice between 1 and 4 mm^3 moves the censoring point by log(4) = 1.4 on those rows only. Treat `v == 0` as "volume < 4 mm^3" (left-censored), do not replace by a pseudocount. Run a sensitivity check at 1 and 8 mm^3. In the three selected drugs, 12 of 581 paired treated curves contain any zeros.

**Combinations.** 24 labels contain ` + ` (1,238 animals). They are kept in `pct_long.csv` (`arm == "combo"`) but are out of scope for the single-agent drug datasets.

**Response metrics sheet (`PCT curve metrics`, 4,758 rows).**
- `ResponseCategory` takes the values `CR, PR, SD, PD` plus progression strings (`SD-->PD`, `PR-->PD`, `CR-->-->PD`, ...). The first token equals the mRECIST class computed from `BestResponse` and `BestAvgResponse` with the thresholds mCR: best < -95 and avg < -40; mPR: best < -50 and avg < -20; mSD: best < 35 and avg < 30; else mPD. I checked all 4,758 rows (0 disagree). The `-->PD` tail means the tumour later progressed (1,320 rows). The sheet uses `CR/PR/SD/PD` without the `m` prefix; `pct_metrics.csv` adds it: `response_category` in `mCR|mPR|mSD|mPD` (the best class), `response_trajectory` keeps the original string, `progressed` is the tail flag.
- `BestResponse` = minimum of % volume change over time; `BestAvgResponse` = minimum of the running mean of % volume change; the sheet's earliest `Day_BestResponse` is day 3 (the Xeva implementation of mRECIST restricts to day >= 10; the sheet does not).
- `TimeToDouble` appears to be set to `Day_Last` when the tumour never doubled; `Day_Last` is the last observation day.
- Untreated animals have metrics rows too (226).

## Cleaned files

`pct_long.csv`: model, tumour_type, treatment, arm, replicate, day, day_raw, t0_shifted, volume_mm3, detected, body_weight_g, pct_tvol_diff, pct_bw_diff, treatment_raw, row_id.

`pct_metrics.csv`: model, tumour_type, treatment, treatment_type, response_category, best_response, best_avg_response, response_trajectory, progressed, day_best_response, day_best_avg_response, time_to_double, day_last, target, has_curve.

`treatment_audit.csv`: per single-agent treatment: models with any curve, models with both arms, `n_paired` (both arms with >= 3 points and starting at day 0), tumour-type counts among paired models, mCR/mPR/mSD/mPD counts of the treated arm among paired models, models with any zero volume, median follow-up (days).

## Regimens and PK priors

`data/regimens.toml` has one block per selected drug (BKM120, BYL719, binimetinib). Each value is labelled
verified / derived / assumption with its source. Read the status fields before trusting a number: **the PDXE dose,
route and schedule for BKM120 and BYL719 could not be verified** (the Online Methods table is paywalled); the
binimetinib dose comes from the open Supplementary Information (30 mg/kg bid MTD in NSCLC).
