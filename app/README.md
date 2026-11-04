# TumourPKPD explorer

Interactive Bayesian tumour PK/PD on public PDX data: Simeoni tumour growth inhibition, fitted hierarchically
with NUTS to Novartis PDX Encyclopedia (PDXE) tumour curves for binimetinib, BKM120 and BYL719. Pick a drug and
a PDX model (or a new PDX drawn from the population), change the regimen, and see the predictive bands, the
observed points, and a virtual PDX trial.

Built with Bonito and WGLMakie. Everything it computes comes from `TumourPKPD`: `simulate_regimen`,
`simulate_control`, `is_extrapolation`, `simulate_trial`, `make_regimen`.

![reference regimen](screenshots/reference-regimen.png)
![extrapolation](screenshots/extrapolation-new-pdx.png)

(Static renderings of the same panels. The screenshots in this folder use **demo fits**, so every panel is
labelled "DEMO, not fitted". `screenshots/live-browser.png` is the live app in Chrome.)

## Run

```sh
# from the repository root
julia --project=app -t 2 app/run.jl [fits_dir] [--port 8081] [--demo]
```

Open `http://127.0.0.1:8081`.

- `fits_dir` holds `<drug>.jld2` files written by `save_fit` (default `fits/`). A drug without a file falls back
  to a synthetic demo fit, written to `app/demo_fits/` (gitignored) and labelled "DEMO — not fitted" in the page
  header, the plot title and the info panel. Pointing the app at real fits needs no code change.
- `--demo` uses demo fits for every drug.
- Use `-t 4` or more if the machine is idle: the main plot and the virtual trial both thread over posterior draws.

Other scripts:

| script | what it does |
|---|---|
| `smoke_test.jl [fits_dir]` | headless: builds the state and the app, drives every drug x schedule, new PDX and known PDX, the debounced widget flow, the virtual trial, a static HTML export; prints update times |
| `screenshots.jl [fits_dir]` | CairoMakie renderings of the panels into `screenshots/` |
| `browser_test.mjs URL [png]` | optional, Node 22+ and Chrome: drives the running page through the DevTools protocol and times input to update |
| `demo_fits.jl` | how the synthetic fits are built (parameters are plausible, not fitted) |

Install: the app has its own environment (`app/Project.toml`: Bonito, WGLMakie, CairoMakie, Makie, JLD2,
Observables, Statistics, and `TumourPKPD` from `..`). `julia --project=app -e 'using Pkg; Pkg.instantiate()'`.

## What the page shows

1. **Drug and PDX.** Dropdown for the drug; a text box filters the PDX list by id or tumour type; the first
   entry of the list is "New PDX (population)", a prediction for an unseen tumour.
2. **Regimen.** Dose as a multiple of the PDXE reference dose (0.25 to 3), schedule (qd, bid, q2d,
   5-on-2-off), treatment duration, follow-up horizon. Sliders are debounced (120 ms) and results cached per
   (drug, model, regimen, horizon).
3. **Main plot.** Median with 50% and 90% bands for the treated arm and the untreated control, the observed
   PDXE points of the selected model, the end of dosing (dotted), the assumed detection limit, and the region
   beyond the observed data shaded as extrapolation. Log-scale toggle.
4. **Extrapolation banner**, from `is_extrapolation`, listing every reason (dose level, interval, early stop,
   horizon beyond the longest observed follow-up).
5. **Virtual trial.** Responder rate (mCR+mPR) and the four category fractions with 90% CrIs for the current
   and the reference regimen, next to the published PDXE category fractions of the drug. The horizon of the
   scoring window is shown with the result. The default is `quick` (50 draws x 100 PDX); the *Refine* button
   runs 200 x 200.
6. **Info.** R-hat, ESS, divergences from `pd.diagnostics`, the PK and dose status strings from
   `data/regimens.toml`, data provenance.

## Fit versus assumption

| | |
|---|---|
| **Fitted** (posterior) | per-PDX growth rate lambda0 and drug sensitivity k2, their between-PDX spread, pooled lambda1, kill-delay k1, residual noise sigma |
| **Assumed: dose** | BKM120 and BYL719 doses are assumptions (the PDXE Online Methods were not accessible); binimetinib 30 mg/kg bid is stated for NSCLC and inferred for the other indications. The slider is a multiplier, not an mg/kg claim; the model only sees exposure divided by the Cmax of the reference regimen. |
| **Assumed: PK** | literature mouse PK held fixed (one compartment); each parameter carries its own status string in the info panel |
| **Assumed: schedule** | PDXE tested one schedule per drug; every other schedule is a what-if |
| **Assumed: detection limit** | recorded volume 0 is censored below 4 mm^3 |

Data: Gao H et al., *Nat Med* 21:1318-1325 (2015), doi 10.1038/nm.3954.

What the bands mean: they are posterior predictive and include observation noise by default (checkbox). For a
known PDX they are conditional on that tumour's fitted growth and sensitivity; for a new PDX they also include
between-PDX variation. Past the observed window they are model-based, not data-constrained, which is what the
orange region and the banner say.
