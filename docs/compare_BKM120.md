# Model comparison: BKM120, v1 vs v2

Generated 2026-11-08 12:04. v1: `fits`; v2: `fits`. Both fit sets passed the frozen-split, window, cutoff and held-out-leakage checks against one shared split (the script errors otherwise) and are scored by the same code. Aggregated metrics only: each pools at least 10 contributing PDX models, and metrics with fewer are left out.

Shared split: model fraction 0.2, seed 1, 40 held-out PDX for this drug (identical for both fit sets; declared in all fits' metadata). Sampling plan: each fit's predictive draws are `randperm(MersenneTwister(seed), S)[1:n]` over its own stored draws with the same seed rule and the same counts n = 200 (held-out curve check), 200 (late21 forecast), 1000 (response section; seed + 1); stored draws v1: primary 2000, late21 2000; v2: primary 2000, late21 2000.


| metric | v1 | v2 | v2 − v1 |
|---|---|---|---|
| held-out 90% coverage | 0.9562 | 0.9431 | -0.01314 |
| held-out mean log predictive density | -6.833 | -6.718 | +0.1146 |
| held-out mean CRPS (mm³) | 167 | 153.1 | -13.95 |
| held-out treated below-5% / above-95% tails | 0.5% / 0.0% | 1.1% / 2.1% |  |
| observed responder fraction (held-out) | 0.025 | 0.025 | +0 |
| predicted responder fraction, recorded (held-out) | 0.1789 | 0.1074 | -0.07152 |
| predicted responder fraction, latent (held-out) | 0.1726 | 0.1018 | -0.07082 |
| Kendall τ_b (held-out) | 0.4738 | 0.2263 | -0.2475 |
| AUROC (held-out) | 0.8077 | 1 | +0.1923 |
| Brier skill vs constant risk (held-out) | 0.007317 | 0.03906 | +0.03174 |
| observed responder fraction (training) | 0.07643 | 0.07643 | +0 |
| predicted responder fraction, recorded (training, in-sample) | 0.06367 | 0.06268 | -0.0009873 |
| late21 90% coverage | 0.7446 | 0.8115 | +0.06692 |
| late21 mean log predictive density | -6.476 | -6.234 | +0.242 |
| late21 mean CRPS (mm³) | 103.3 | 78.67 | -24.64 |
| late21 treated below-5% / above-95% tails | 7.8% / 19.4% | 5.1% / 19.8% |  |
| late21 treated median residual, last day bin | 0.07606 | 0.1157 | +0.03967 |
| primary max R-hat | 1.028 | 1.608 | +0.5803 |
| primary min bulk ESS | 294.9 | 13.85 | -281.1 |
| primary cross-mode R-hat | 1.026 | NaN | +NaN |
| late21 max R-hat | 1.625 | 1.186 | -0.4392 |
| late21 min bulk ESS | 13.81 | 32.01 | +18.21 |
| late21 cross-mode R-hat | 1.609 | 1.084 | -0.5252 |
| predictive draws, held-out curve check | 200 | 200 | +0 |
| predictive draws, late21 forecast | 200 | 200 | +0 |
| predictive draws, response section | 1000 | 1000 | +0 |

Coverage closer to 0.90 is better; log predictive density higher is better; CRPS lower is better; tails nominal 5% / 5% (below the 5% quantile = smaller than predicted, above the 95% quantile = larger than predicted, i.e. excess regrowth). Convergence rows show whether each fit set is fit for comparison: do not interpret metric differences for a fit with R-hat > 1.01 as model differences.

