# Running the fits

The production fits (3 drugs x 2 designs, 4 chains each) are too long for one machine and one job, so they run on
GitHub Actions as a staged pipeline (`.github/workflows/fits.yml`) built on the `scripts/fit_stage.jl` command line.
Each stage is a separate job, so a failed or timed-out piece can be re-run alone. The same stages run locally with the
commands further down.

Designs: `primary` (80% of PDX models, days <= 42) and `late21` (same models, fit to days <= 21, forecast days 22-42).

## One-time setup: the STAGE_KEY secret

The repository is public. The PDXE tables are not redistributed by it, and anything derived from them that could be
used to reconstruct observations must not be downloadable. The fits workflow therefore passes the cleaned tables and
the sampler state between jobs only as encrypted artifacts (AES-256-CBC, PBKDF2, salted, via `openssl enc`), keyed by a
repository secret. Create it once:

```
openssl rand -base64 48 | gh secret set STAGE_KEY
```

Without the secret every job fails at its first step with a message saying so. Rotating the key does not affect old
merged fits (those are not encrypted), only runs in flight.

## What goes where

| thing | where it lives | in clear? |
|---|---|---|
| raw and cleaned PDXE tables | downloaded from the Springer source by `prep` (SHA-256 verified, retries with backoff), handed to later jobs as the artifact `pdxe-data` | encrypted, kept 2 days |
| `init`, `warm`, `chain` state | artifacts `init-...`, `warm-...`, `chain-...` | encrypted, kept 3 days |
| merged fit and its diagnostics | artifact `fit-<drug>-<design>` | **in clear**, kept 90 days |
| reports and figures | made locally only; the workflow does not produce or upload them | - |
| Julia depot (code only) | `actions/cache` | yes, contains no data |

No PDXE data is ever stored in `actions/cache`, because workflows run from forks can restore caches. `CI.yml` downloads
PDXE from the source inside the test job on every run and keeps it on that runner only; forks get no secrets.

Why the stage state is encrypted: the chain files hold the baseline coordinates and the merged fit holds baseline
draws built from them, so together they allow recovering the observed day-0 volumes. The merged fit alone does not. Before it
is uploaded, `scripts/ci_check_upload.jl --kind=fit` verifies that it has exactly the expected `PosteriorDraws`
fields, only allow-listed diagnostics keys (no matrices, no per-unit coordinate vectors, no raw chain arrays), and latent
(non-constant) baselines. What the merged fit still contains: per-PDX posterior draws of lambda0, k2 and baselines
(posterior medians within about 10% of the observed day-0 volumes, not equal to them), model ids, tumour types, and
the last observed day of every training curve (`diagnostics["cutoffs"]`).

## Trigger a run

```
gh workflow run fits.yml \
  -f drugs=binimetinib,BKM120,BYL719 -f designs=primary,late21 \
  -f nchains=4 -f nadapt=500 -f parts=2 -f draws_per_part=250 -f seglen=125 -f burnin=25
```

All inputs are optional; the values above are the defaults. Draws per chain = `parts x draws_per_part`.

| input | meaning |
|---|---|
| `drugs` | comma-separated, names as in `data/regimens.toml` |
| `designs` | `primary`, `late21` |
| `nchains` | chains per drug and design (1-based chain index K) |
| `nadapt` | warmup iterations per chain |
| `parts` | sampling parts per chain (1-based part index P), one job each |
| `draws_per_part` | retained draws per part |
| `seglen` | warmup segment length. `125` (default) splits each chain's warmup into `ceil(nadapt / seglen)` chained jobs (`warmup`, `warmup2`, `warmup3`, `warmup4`) of up to `seglen` iterations; at most 4 segments, so `nadapt <= 4 x seglen` (checked in `prep`); `0`, or a value `>= nadapt`, runs the whole warmup in one job |
| `burnin` | iterations discarded at the start of each sampling part (no adaptation), default 25 |
| `units` | optional: subsample N PDX units (smoke tests only; recorded in `init.jld2`) |
| `model` | `v1` (default; independent log lambda0 / log k2 effects, pooled transition volume) or `v2` (correlated effects with an LKJ(2) prior on their 2x2 correlation and a per-PDX transition volume). v2 fits are saved as `fits/<drug>_<design>_v2.jld2` (+ `_diagnostics.md`), so they sit next to the v1 fits. `fit_stage.jl init` uses `--mode-gap=30` for v2 (a converged MAP mode more than 30 log-density units below the best does not start chains; each excluded mode is recorded with lp, gap and k1 in the diagnostics) and no gap limit for v1 |
| `waning` | `false` (default) or `true`: pooled waning efficacy, kill term `k2 exp(-rho t)`; a sensitivity fit, saved with a `_waning` suffix (`..._v2_waning`) |

Smoke test, a few minutes of sampling and the whole plumbing:

```
gh workflow run fits.yml -f drugs=BKM120 -f designs=late21 -f nchains=2 -f nadapt=50 \
  -f seglen=25 -f parts=1 -f draws_per_part=25 -f burnin=5 -f units=12
gh run watch
```

The `prep` job validates the inputs and prints the plan in the run summary. Matrices are limited to 256 jobs; the
defaults give 6 init, 24 warmup, 48 sample and 6 merge jobs. About 20 jobs run at once, the rest queue. A second
dispatch with the same ref, drugs and designs waits for the first instead of running in parallel.

## Stages

| job | matrix | what it does | uploads (encrypted unless noted) |
|---|---|---|---|
| `prep` | - | validates inputs, builds the matrices, restores the Julia depot and precompiles, downloads and cleans PDXE | `pdxe-data` |
| `init` | drug x design | `fit_stage.jl init`: data prep, multi-start MAP, split metadata | `init-<drug>-<design>` |
| `warmup` | drug x design x chain | `fit_stage.jl warmup`: NUTS adaptation from init. With `seglen` and more than one segment: segment 1 only (`--seglen --seg=1`), writes the resumable state (adaptor, position, RNG; parameters only). Otherwise the whole warmup, writes `warm_K` | segmented: `warmseg-<drug>-<design>-<K>-1`; otherwise `warm-<drug>-<design>-<K>` |
| `warmup2`, `warmup3`, `warmup4` | drug x design x chain | segment S = 2, 3, 4, each skipped when the warmup has fewer than S segments (`nseg = ceil(nadapt / seglen)`): downloads init and its own segment S-1 state, runs `--seg=S`. Segments before the last write the resumable state, the last one (S = nseg) writes the warm state (position, step size, metric, seed) | `warmseg-<drug>-<design>-<K>-S`; the last one `warm-<drug>-<design>-<K>` |
| `sample` | drug x design x chain x part | `fit_stage.jl sample --draws --burnin`: fixed-setting sampling from the warm state, no adaptation | `chain-<drug>-<design>-<K>-<P>` |
| `merge` | drug x design | checks that all chains and parts are there, `fit_stage.jl merge`: R-hat/ESS across all parts, `PosteriorDraws`, diagnostics; upload check | `fit-<drug>-<design>` (**clear**: `<drug>_<design>.jld2`, `<drug>_<design>_diagnostics.md`) |

Every job decrypts what it needs with `STAGE_KEY` and fails clearly if the secret is missing.

GitHub cannot make a job depend on one element of a matrix, so `warmupS` starts only after every `warmup(S-1)` job is done,
`sample` after every warmup job of every segment, and `merge` after every `sample` job. `sample` always takes the artifact
`warm-<drug>-<design>-<K>`, whichever job produced it. A combination whose upstream job failed fails within a minute on its artifact
download; the other combinations carry on. The merge job refuses to run on an incomplete set of chains or parts.

Every job uses `julia -t 4` and a 330-minute timeout (the limit is 360); `prep` has 60.

## Expected times

All figures are **projections** from the inference worker's measurements at N = 159 binimetinib primary (4 threads,
65-99 ms per gradient on an M-series Mac), not measurements on a hosted runner; a hosted 4-vCPU runner is probably
slower (3-5 h for a whole 500-iteration warmup was the lead's estimate). Replace with run times once there are some.
late21 fits cost about half. Each job adds 5-10 min of JIT.

| stage | projected (primary, N~160) | measured on a runner |
|---|---|---|
| prep, cold depot cache | - | 15 min (run 41873171934) |
| prep, warm depot cache | - | not yet measured |
| init (5 starts x 1000 iterations) | about 30-40 min (2 starts x 300 its took 3 min) | primary 32-65 min, late21 24-52 min |
| warmup, nadapt=500, whole chain, M-series | about 2.0 h (range 1.7-2.6 h); the first ~100 iterations at tree depth 9 cost ~55 min | not run unsegmented on a hosted runner |
| warmup segmented, seglen=250 (v1; the default is now 125) | segment 1 about 1.3 h, segment 2 about 0.6 h | segment 1: primary 90-323 min (median 201), late21 34-135; segment 2: primary 58-190, late21 17-100. **Primary segment 1 came within 7 min of the 330-min timeout: use `seglen=125` (up to 4 segments are now supported).** |
| sample, 250 draws + 25 burn-in, per part | 18-39 min | primary 47-155 min (median 104), late21 18-121 (median 44) |
| merge | minutes | _TBD_ |
| whole run (default inputs, with queueing) | | _TBD_ |

If a `sample` job approaches 330 minutes, raise `parts` and lower `draws_per_part`. If a warmup segment is too long, lower
`seglen` (down to `nadapt / 4`: four segments at most) or lower `nadapt`.

## Download the fits

```
gh run list --workflow=fits.yml --limit 5
gh run download <run-id> --pattern 'fit-*' --dir fits/downloaded
```

With several matching artifacts `gh` creates one directory per artifact (`fit-<drug>-<design>/`). Move the files into
`fits/` so that `fits/<drug>_<design>.jld2` exists, which is where the validation script and the app look:

```
mv fits/downloaded/fit-*/* fits/
```

The `.jld2` files are not tracked by git (`fits/*.jld2` is ignored). Validation reports and figures are produced
locally from them (`julia --project=. -t 4 scripts/validation_report.jl <drug>`); they show observed PDXE curves and
are not uploaded anywhere.

Stage artifacts (`init-*`, `warm-*`, `chain-*`) are encrypted; to inspect one locally, decrypt with the same secret
value: `openssl enc -d -aes-256-cbc -pbkdf2 -pass env:STAGE_KEY -in chain_1_1.jld2.enc -out chain_1_1.jld2`.

## Re-running a failed piece

```
gh run rerun <run-id> --failed
```

re-runs the failed jobs and the jobs that depend on them, reusing the artifacts of the jobs that succeeded. This works
while the artifacts exist: 2 days for `pdxe-data`, 3 days for stage state (including `warmseg-*`). After that, start a fresh dispatch.

## Run the same stages locally

From the repository root, with the PDXE cache built (`julia --project=. data/download.jl --only=xlsx` and
`julia --project=. data/clean.jl`):

```
DRUG=BKM120; DESIGN=late21; DIR=stage/${DRUG}_${DESIGN}; mkdir -p $DIR fits

julia --project=. -t 4 scripts/fit_stage.jl init   $DRUG $DESIGN --out=$DIR            # add --units=12 to subsample
for K in 1 2 3 4; do
  julia --project=. -t 4 scripts/fit_stage.jl warmup $DRUG $DESIGN --chain=$K --dir=$DIR --nadapt=500   # one job ...
  # ... or up to four resumable segments (as the workflow does with seglen=125):
  #   for S in 1 2 3 4; do ... warmup $DRUG $DESIGN --chain=$K --dir=$DIR --nadapt=500 --seglen=125 --seg=$S; done
done
for K in 1 2 3 4; do for P in 1 2; do
  julia --project=. -t 4 scripts/fit_stage.jl sample $DRUG $DESIGN --chain=$K --part=$P --dir=$DIR --draws=250 --burnin=25
done; done
julia --project=. -t 4 scripts/fit_stage.jl merge  $DRUG $DESIGN --dir=$DIR --out=fits/
julia --project=. scripts/ci_check_upload.jl --kind=fit fits/${DRUG}_${DESIGN}.jld2 fits/${DRUG}_${DESIGN}_diagnostics.md
```

Chains and parts are independent, so run them in parallel if the machine has cores to spare (keep `-t` times the
number of simultaneous processes within the core count). `stage/` holds sampler state that can be combined with the
merged fit to recover observed baselines: keep it out of git and do not publish it.

To see the matrices a dispatch would create, without running anything:

```
julia scripts/ci_matrix.jl --drugs=BKM120 --designs=late21 --nchains=2 --parts=1
```

## Caches

Only the Julia depot (code, no data) is cached: `julia-depot-v1-<os>-<julia version>-<hash of Project.toml and
Manifest.toml>`, written by `prep` only, size _TBD_. GitHub keeps 10 GB of caches per repository and evicts entries
unused for 7 days; a missing depot cache only makes the jobs slower. `CI.yml` keeps its own depot cache through
`julia-actions/cache`.

## Action pinning

Every external action is pinned to a full commit SHA with the tag in a comment. To update one, resolve the new tag
(dereferencing annotated tags) and review the diff before changing the SHA:

```
gh api repos/actions/checkout/git/ref/tags/v4.4.0 --jq .object
```
