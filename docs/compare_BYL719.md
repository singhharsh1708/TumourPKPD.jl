# Model comparison: BYL719, v1 vs v2

Generated 2026-11-08 12:11. v1: `fits`; v2: `fits`. Both fit sets passed the frozen-split, window, cutoff and held-out-leakage checks against one shared split (the script errors otherwise) and are scored by the same code. Aggregated metrics only: each pools at least 10 contributing PDX models, and metrics with fewer are left out.

Shared split: model fraction 0.2, seed 1, 38 held-out PDX for this drug (identical for both fit sets; declared in all fits' metadata). Sampling plan: each fit's predictive draws are `randperm(MersenneTwister(seed), S)[1:n]` over its own stored draws with the same seed rule and the same counts n = 200 (held-out curve check), 200 (late21 forecast), 1000 (response section; seed + 1); stored draws v1: primary 2000, late21 2000; v2: primary 1000, late21 2000.


| metric | v1 | v2 | v2 − v1 |
|---|---|---|---|
| held-out 90% coverage | 0.9382 | 0.925 | -0.01324 |
| held-out mean log predictive density | -6.725 | -6.672 | +0.05259 |
| held-out mean CRPS (mm³) | 155.2 | 153.1 | -2.062 |
| held-out treated below-5% / above-95% tails | 0.6% / 1.9% | 0.6% / 4.7% |  |
| observed responder fraction (held-out) | 0.02632 | 0.02632 | +0 |
| predicted responder fraction, recorded (held-out) | 0.1574 | 0.1058 | -0.05155 |
| predicted responder fraction, latent (held-out) | 0.1511 | 0.1006 | -0.05047 |
| Kendall τ_b (held-out) | 0.3883 | 0.3063 | -0.08196 |
| AUROC (held-out) | 0.7703 | 0.2432 | -0.527 |
| Brier skill vs constant risk (held-out) | 0.004054 | -0.02767 | -0.03172 |
| observed responder fraction (training) | 0.06849 | 0.06849 | +0 |
| predicted responder fraction, recorded (training, in-sample) | 0.06493 | 0.06411 | -0.0008219 |
| late21 90% coverage | 0.7038 | 0.7428 | +0.039 |
| late21 mean log predictive density | -6.755 | -6.535 | +0.2206 |
| late21 mean CRPS (mm³) | 114 | 89.87 | -24.1 |
| late21 treated below-5% / above-95% tails | 5.3% / 25.5% | 4.5% / 27.6% |  |
| late21 treated median residual, last day bin | 0.1807 | 0.2104 | +0.02972 |
| primary max R-hat | 1.015 | 1.57 | +0.5544 |
| primary min bulk ESS | 661.8 | 42.93 | -618.8 |
| primary cross-mode R-hat | 1.013 | 1.537 | +0.5249 |
| late21 max R-hat | 1.015 | 1.068 | +0.05313 |
| late21 min bulk ESS | 780.9 | 82.93 | -697.9 |
| late21 cross-mode R-hat | 1.011 | 1.048 | +0.03704 |
| predictive draws, held-out curve check | 200 | 200 | +0 |
| predictive draws, late21 forecast | 200 | 200 | +0 |
| predictive draws, response section | 1000 | 1000 | +0 |

Coverage closer to 0.90 is better; log predictive density higher is better; CRPS lower is better; tails nominal 5% / 5% (below the 5% quantile = smaller than predicted, above the 95% quantile = larger than predicted, i.e. excess regrowth). Convergence rows show whether each fit set is fit for comparison: do not interpret metric differences for a fit with R-hat > 1.01 as model differences.

