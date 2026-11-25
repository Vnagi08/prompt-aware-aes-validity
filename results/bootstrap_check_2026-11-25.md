# Essay-level paired bootstrap, gate vs ungated baseline (seeds 32, 42, 52) — 2026-11-25
Script: code/22_boot_from_preds.py (QWK = ProTACT metrics.metrics.kappa, quadratic). Kaggle CPU notebook notebooka6ff507d91 v1, 56.2 min.
Inputs: per-essay predictions from code/21_gate_bootstrap.py (notebook65479248d7 v2 = seed 52, v3 = seed 32, v4 = seed 42), routed via dataset gate-boot-preds (v1 = s52, v2 = s32) and copy notebook notebook2d8aa18291 (s42 + s52).
Log line 13: "prediction files: 48". All assertions (same essays, same order, same gold scores across models and seeds) passed. B = 1,000 resamples, rng seed 0; the same resampled essays for both models and all three seeds in every replicate; within a replicate, mean over seeds, then over prompts.
Files: results/bootstrap_gate_vs_base.json, results/boot_log.txt.

## Check 1: full-sample point estimates vs 18_multiseed_stats.py (Table V inputs)
Max absolute difference after rounding to 3 decimals: 0 (48/48 values identical).
| P | seed | base avg (bootstrap) | base (stored) | gate avg (bootstrap) | gate (stored) |
|---|---|---|---|---|---|
| 1 | 32 | 0.6399 | 0.640 | 0.6458 | 0.646 |
| 1 | 42 | 0.6367 | 0.637 | 0.6529 | 0.653 |
| 1 | 52 | 0.6483 | 0.648 | 0.6434 | 0.643 |
| 2 | 32 | 0.5533 | 0.553 | 0.5517 | 0.552 |
| 2 | 42 | 0.5632 | 0.563 | 0.5491 | 0.549 |
| 2 | 52 | 0.5617 | 0.562 | 0.5448 | 0.545 |
| 3 | 32 | 0.5701 | 0.570 | 0.5670 | 0.567 |
| 3 | 42 | 0.5731 | 0.573 | 0.5552 | 0.555 |
| 3 | 52 | 0.5665 | 0.566 | 0.5412 | 0.541 |
| 4 | 32 | 0.5347 | 0.535 | 0.5280 | 0.528 |
| 4 | 42 | 0.5365 | 0.537 | 0.5244 | 0.524 |
| 4 | 52 | 0.5387 | 0.539 | 0.5281 | 0.528 |
| 5 | 32 | 0.6330 | 0.633 | 0.6120 | 0.612 |
| 5 | 42 | 0.6405 | 0.640 | 0.6181 | 0.618 |
| 5 | 52 | 0.6382 | 0.638 | 0.6107 | 0.611 |
| 6 | 32 | 0.5585 | 0.559 | 0.5415 | 0.541 |
| 6 | 42 | 0.5615 | 0.561 | 0.5419 | 0.542 |
| 6 | 52 | 0.5654 | 0.565 | 0.5524 | 0.552 |
| 7 | 32 | 0.3679 | 0.368 | 0.3243 | 0.324 |
| 7 | 42 | 0.3756 | 0.376 | 0.3323 | 0.332 |
| 7 | 52 | 0.3553 | 0.355 | 0.3225 | 0.323 |
| 8 | 32 | 0.4350 | 0.435 | 0.4349 | 0.435 |
| 8 | 42 | 0.4582 | 0.458 | 0.4435 | 0.444 |
| 8 | 52 | 0.4551 | 0.455 | 0.4481 | 0.448 |
## Result: gate - baseline, three-seed mean with 95% essay-level bootstrap interval
| P | essays | avg QWK | Content QWK | PA QWK |
|---|---|---|---|---|
| 1 | 1783 | +0.0057 [+0.0019, +0.0094] | -0.0376 [-0.0506, -0.0248] | - |
| 2 | 1798 | -0.0109 [-0.0149, -0.0070] | -0.1139 [-0.1306, -0.0961] | - |
| 3 | 1726 | -0.0154 [-0.0215, -0.0089] | -0.0067 [-0.0202, +0.0062] | -0.0926 [-0.1111, -0.0749] |
| 4 | 1772 | -0.0098 [-0.0144, -0.0047] | -0.0222 [-0.0316, -0.0123] | -0.0245 [-0.0377, -0.0094] |
| 5 | 1805 | -0.0237 [-0.0284, -0.0189] | -0.0164 [-0.0286, -0.0037] | -0.0882 [-0.1049, -0.0708] |
| 6 | 1800 | -0.0166 [-0.0219, -0.0114] | -0.0228 [-0.0338, -0.0120] | -0.0638 [-0.0869, -0.0425] |
| 7 | 1569 | -0.0399 [-0.0441, -0.0359] | -0.1865 [-0.2010, -0.1718] | - |
| 8 | 723 | -0.0073 [-0.0139, -0.0013] | +0.1108 [+0.0730, +0.1433] | - |
| mean | | -0.0147 [-0.0166, -0.0129] | -0.0369 [-0.0433, -0.0307] | -0.0673 [-0.0765, -0.0581] (p3-6) |

Five-seed means for comparison (Table V / 18_multiseed_stats.py): avg -0.016; p1 -0.001, p8 +0.004 (seed-level intervals containing 0).
With seeds 32/42/52 only, p1 is +0.006 and p8 -0.007, both with essay-level intervals excluding 0: the sign on these two prompts depends on which seeds are included (seeds 12 and 22 not in this bootstrap).
