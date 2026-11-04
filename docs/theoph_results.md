# Theophylline validation (Theoph, 12 subjects)

Hierarchical 1-compartment oral model, Turing.jl NUTS, non-centred diagonal random effects on CL, V, ka, combined error sd = sqrt(sigma_add^2 + (sigma_prop * pred)^2). Apparent CL/F and V/F (F = 1). Units: h, mg/kg, mg/L, L/h/kg, L/kg, 1/h.

Primary fit (ordered prior, see below) uses the 120 rows with t > 0. The 12 t = 0 rows are dropped: the model predicts exactly 0 there, and 3 of them are positive carry-over baselines (subject, mg/L): (1, 0.74), (7, 0.15), (10, 0.24), which the model has no term for. A sensitivity fit on all 132 rows is shown in the table.

Settings: 4 chains x (1000 warmup + 1000 draws), target accept 0.8, seed 20261104, ForwardDiff, 4 threads. Wall time: primary 34.9 s, sensitivity 27.4 s, unconstrained mode check 43.1 s.

## Priors

| parameter | prior | units | rationale |
|---|---|---|---|
| μ_CL | Normal(log 0.1, 1.0) | log L/h/kg | median CL/F; broad (factor ~2.7 per SD) around a generic small-molecule scale, not centred on the literature value |
| μ_V | Normal(log 0.5, 1.0) | log L/kg | median V/F; broad around total body water scale |
| μ_ka | Normal(log 1.0, 1.0); primary fit: truncated below at μ_CL − μ_V (ka > CL/V), sampled via the gap δ_ka | log 1/h | median ka; broad around 1/h typical oral absorption; truncation removes the flip-flop mode |
| ω_CL | Half-Normal(0.5) | log scale | between-subject SD of log CL; allows CV up to ~100% |
| ω_V | Half-Normal(0.5) | log scale | between-subject SD of log V |
| ω_ka | Half-Normal(1.0) | log scale | between-subject SD of log ka; wider because ka is typically most variable |
| z_CL, z_V, z_ka | Normal(0, 1) each, x12 subjects | - | non-centred standardized subject effects |
| σ_prop | Half-Normal(0.3) | unitless | proportional residual SD (up to ~60%) |
| σ_add | Half-Normal(0.5) | mg/L | additive residual SD (assay floor) |

Half-Normal(s) is Normal(0, s) truncated at 0.

## Diagnostics

Primary, all 45 sampled quantities: max R-hat 1.0036, min bulk ESS 1273, min tail ESS 1538, divergent transitions 0. Sensitivity: max R-hat 1.0052, min bulk ESS 1135, divergences 0.

### Mode check: unconstrained model, dispersed starts

The model with an unconstrained prior on mu_ka has a second, flip-flop mode (ka and ke exchange roles, V collapses). Four chains with dispersed starts, one in the flip-flop region (diagnostics of this fit: max R-hat 1.531, min bulk ESS 14, divergences 8):

| chain | start (CL, V, ka) | median ka | median CL | median V | median CL/V | fraction of draws with ka < CL/V |
|---|---|---|---|---|---|---|
| 1 | (0.04, 0.5, 1.5) | 1.536 | 0.0404 | 0.4585 | 0.0879 | 0.000 |
| 2 | (0.04, 0.025, 0.06) | 0.087 | 0.0406 | 0.0288 | 1.3972 | 1.000 |
| 3 | (0.2, 2.0, 5.0) | 1.533 | 0.0403 | 0.4581 | 0.0881 | 0.000 |
| 4 | (0.01, 0.2, 0.5) | 1.515 | 0.0404 | 0.4578 | 0.0883 | 0.000 |

Chains that end in the flip-flop mode: **2**. The flip-flop mode has V/F of about 0.029 L/kg at the lowest chain, below plasma volume, which is not physiologically credible for a drug distributed like theophylline, and it is not what the literature fit finds. The mode does not mix with the usual one, so pooled R-hat over these chains is meaningless.

**Primary analysis therefore uses an ordered prior** (population ka > CL/V, implemented as a truncation of the mu_ka prior at mu_CL - mu_V; theophylline immediate-release absorption is known to be faster than elimination). With it, all four dispersed chains agree:

| chain | median ka | median CL | median V | median CL/V |
|---|---|---|---|---|
| 1 | 1.503 | 0.0400 | 0.4557 | 0.0880 |
| 2 | 1.524 | 0.0404 | 0.4574 | 0.0888 |
| 3 | 1.511 | 0.0402 | 0.4582 | 0.0878 |
| 4 | 1.514 | 0.0406 | 0.4575 | 0.0889 |

## Population typical values vs literature

Posterior of exp(mu) (population median). Literature: Pinheiro & Bates (2000) nlme `fm1Theo.nlme`, lKe = -2.43267, lKa = 0.45141, lCl = -3.21445; V = CL/ke. 'Ordered, all 132 rows' = sensitivity fit including t = 0 rows; 'unconstrained, usual-mode chains' = unconstrained fit with the flip-flop chain(s) removed (median [90% CrI]).

| quantity | posterior median (t>0) | 90% CrI | literature | rel. diff | lit. in CrI | ordered, all 132 rows | unconstrained, usual-mode chains | naive pooled MLE | two-stage MLE |
|---|---|---|---|---|---|---|---|---|---|
| ka (1/h) | 1.512 | [1.042, 2.216] | 1.571 | -3.7% | yes | 1.485 [1.019, 2.149] | 1.528 [1.058, 2.191] | 1.491 | 1.438 |
| CL/F (L/h/kg) | 0.04031 | [0.03437, 0.04712] | 0.04018 | +0.3% | yes | 0.0405 [0.03472, 0.04716] | 0.0404 [0.03445, 0.04756] | 0.03884 | 0.04302 |
| V/F (L/kg) | 0.4572 | [0.4195, 0.4983] | 0.4576 | -0.1% | yes | 0.4581 [0.4199, 0.4997] | 0.458 [0.422, 0.5003] | 0.4848 | 0.4622 |
| ke = CL/V (1/h) | 0.08844 | [0.07294, 0.1061] | 0.0878 | +0.7% | yes | 0.0882 [0.07359, 0.1053] | 0.08808 [0.07305, 0.1073] | 0.08012 | 0.09308 |

**Pass rule** (typical values of ka, CL, V each within 15% of the literature value and the literature value inside the 90% CrI): **PASS**. This means the typical values agree; it is not a replication of the literature model.

Caveats on the comparison:

- The nlme fit uses a general (correlated) positive-definite random-effects matrix on log ke, log ka, log CL; ours is diagonal on log CL, log V, log ka.
- nlme uses an additive Gaussian error (residual SD 0.68 mg/L) and all 132 observations; ours is combined error on the t > 0 rows (the all-rows column above is the closest analogue).
- The cited nlme output (https://asancpt.github.io/nlme/chapter-8.html) carries an LME-step non-convergence warning (nlminb code 1), so the literature numbers themselves are not exact targets.
- Naive pooled MLE ignores between-subject variation. The two-stage value is the median of per-subject MLEs (subjects 1 and 9 have degenerate individual fits with ka -> very large, so a mean is not meaningful).
- ka is the least well-identified parameter.

## Posterior summaries (primary)

| parameter | mean | sd | R-hat | bulk ESS | tail ESS |
|---|---|---|---|---|---|
| μ_CL | -3.211 | 0.0988 | 1.002 | 1659 | 2131 |
| μ_V | -0.7824 | 0.0534 | 1.001 | 2184 | 2545 |
| μ_ka | 0.4113 | 0.232 | 1.004 | 1273 | 1738 |
| ω_CL | 0.3111 | 0.0816 | 1.001 | 1641 | 2734 |
| ω_V | 0.1555 | 0.0488 | 1.001 | 1846 | 2500 |
| ω_ka | 0.7694 | 0.207 | 1.002 | 1657 | 2161 |
| σ_prop | 0.07899 | 0.0337 | 1.001 | 2370 | 1538 |
| σ_add | 0.5615 | 0.131 | 1.001 | 2836 | 2164 |

In-sample 90% posterior predictive interval coverage (conditional on each subject's random effects, evaluated at the exact observation times): 0.925 (111/120). This is not a held-out forecast.

Figure: `docs/figures/theoph_ppc.png` (median curve and 90% predictive interval including Gaussian residual error; negative predictive draws are kept, not clipped).
