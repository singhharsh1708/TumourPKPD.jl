# TumourPKPD.jl

Bayesian tumour PK/PD in Julia: Simeoni tumour growth inhibition driven by mouse PK, fitted hierarchically
(Turing.jl NUTS) to the Novartis PDX Encyclopedia tumour curves (Gao et al. 2015, *Nat Med* 21:1318,
doi:10.1038/nm.3954), with predictions for new doses, schedules and PDX models.

**Status: work in progress.** Real-data fits and the validation report are not in yet.

## What is here
- `src/pk.jl`: closed-form 1- and 2-compartment PK (IV bolus, oral), dose superposition, ForwardDiff-safe.
- `src/pd.jl`: Simeoni 2004 model with PK-driven kill term.
- `src/data.jl`, `data/`: download and cleaning scripts for PDXE (raw data is not redistributed), regimens and
  mouse PK priors with sources and their status (verified / derived / assumption).
- `src/hierarchical.jl`: hierarchical model (per-PDX growth and potency, latent baselines, censoring below the
  detection limit) with per-unit gradient assembly.
- `src/simulate.jl`, `src/validate.jl`: posterior simulation of new regimens, virtual PDX trials, coverage,
  PSIS-LOO, PDXE response classification, held-out splits.
- `app/`: interactive explorer (Bonito + WGLMakie).
- `docs/theoph_results.md`: PK check against the theophylline data (typical values agree with nlme).

## Quick start
```julia
julia --project -e 'include("data/download.jl"); include("data/clean.jl")'   # fetch + clean PDXE
julia --project -t 4 test/runtests.jl
julia --project=app -t 4 app/run.jl fits                                      # explorer
```

## Assumptions to know about
- PDXE doses for BKM120 and BYL719 are not public in open sources; exposure is normalised to the reference
  regimen, so dose is expressed as a multiple of the PDXE dose.
- Any change from the PDXE regimen (dose, schedule, follow-up) is extrapolation and is flagged as such.

License: MIT.
