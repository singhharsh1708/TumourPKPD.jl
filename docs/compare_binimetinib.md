# Model comparison: binimetinib, v1 vs v2

Generated 2026-11-08 12:21. v1: `fits`; v2: `fits`. Both fit sets passed the frozen-split, window, cutoff and held-out-leakage checks against one shared split (the script errors otherwise) and are scored by the same code. Aggregated metrics only: each pools at least 10 contributing PDX models, and metrics with fewer are left out.

Shared split: model fraction 0.2, seed 1, 41 held-out PDX for this drug (identical for both fit sets; declared in all fits' metadata). Sampling plan: each fit's predictive draws are `randperm(MersenneTwister(seed), S)[1:n]` over its own stored draws with the same seed rule and the same counts n = 200 (held-out curve check), 200 (late21 forecast), 1000 (response section; seed + 1); stored draws v1: primary 2000, late21 2000; v2: primary 1000, late21 2000.


| metric | v1 | v2 | v2 − v1 |
|---|---|---|---|
| held-out 90% coverage | 0.953 | 0.953 | +0 |
| held-out mean log predictive density | -6.842 | -6.774 | +0.06824 |
| held-out mean CRPS (mm³) | 170.2 | 163.5 | -6.701 |
| held-out treated below-5% / above-95% tails | 0.5% / 1.2% | 1.2% / 3.0% |  |
| observed responder fraction (held-out) | 0.122 | 0.122 | +0 |
| predicted responder fraction, recorded (held-out) | 0.2305 | 0.1692 | -0.06129 |
| predicted responder fraction, latent (held-out) | 0.2223 | 0.1603 | -0.06205 |
| Kendall τ_b (held-out) | 0.2923 | 0.3486 | +0.05624 |
| AUROC (held-out) | 0.6139 | 0.7222 | +0.1083 |
| Brier skill vs constant risk (held-out) | 0.02352 | 0.02983 | +0.006315 |
| observed responder fraction (training) | 0.1132 | 0.1132 | +0 |
| predicted responder fraction, recorded (training, in-sample) | 0.09482 | 0.09442 | -0.0004088 |
| late21 90% coverage | 0.7401 | 0.7849 | +0.0448 |
| late21 mean log predictive density | -6.616 | -6.406 | +0.2095 |
| late21 mean CRPS (mm³) | 108.8 | 89.33 | -19.51 |
| late21 treated below-5% / above-95% tails | 11.2% / 18.1% | 9.5% / 17.9% |  |
| late21 treated median residual, last day bin | 0.04523 | 0.06155 | +0.01632 |
| primary max R-hat | 1.183 | 1.731 | +0.5475 |
| primary min bulk ESS | 32.47 | 37.77 | +5.292 |
| primary cross-mode R-hat | 1.143 | 1.69 | +0.5462 |
| late21 max R-hat | 1.014 | 1.1 | +0.08614 |
| late21 min bulk ESS | 924.5 | 65.16 | -859.3 |
| late21 cross-mode R-hat | 1.009 | 1.056 | +0.04665 |
| predictive draws, held-out curve check | 200 | 200 | +0 |
| predictive draws, late21 forecast | 200 | 200 | +0 |
| predictive draws, response section | 1000 | 1000 | +0 |

Coverage closer to 0.90 is better; log predictive density higher is better; CRPS lower is better; tails nominal 5% / 5% (below the 5% quantile = smaller than predicted, above the 95% quantile = larger than predicted, i.e. excess regrowth). Convergence rows show whether each fit set is fit for comparison: do not interpret metric differences for a fit with R-hat > 1.01 as model differences.

