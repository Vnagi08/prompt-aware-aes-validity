# Fixed similarity rule, seed 32 (Kaggle CPU notebook notebookc58d6dec7c, version 5 "rule seed32")
Script: code/20_rule_stage4.py, the same as for seed 42 (rule_baseline_seed42.md). Ungated ProTACT weights of seed 32 (training notebook V11), attached as dataset protact-seed42-outputs v2. No training.
The kagglehub fallback in the log ("New Notebooks cannot be attached in non-interactive sessions") is harmless: the weights were found in the attached dataset.

## Per prompt. Wrong prompt = mean over the 7 other ASAP prompts. % lower = share of essays scored lower.
| P | rule | avg QWK (base) | mean g | Content QWK (base) | Content chg / % lower | PA QWK (base) | PA chg / % lower |
|---|---|---|---|---|---|---|---|
| 1 | hard | 0.638 (0.640) | 0.988 | 0.630 (0.644) | -2.759 / 98.1 | - | - |
| 1 | ramp | 0.640 | 0.956 | 0.642 | -2.528 / 99.0 | - | - |
| 2 | hard | 0.554 (0.553) | 0.989 | 0.613 (0.609) | -1.948 / 89.6 | - | - |
| 2 | ramp | 0.555 | 0.948 | 0.619 | -1.848 / 94.1 | - | - |
| 3 | hard | 0.561 (0.570) | 0.904 | 0.573 (0.596) | -1.315 / 86.4 | 0.532 (0.555) | -1.202 / 85.1 |
| 3 | ramp | 0.550 | 0.790 | 0.542 | -1.070 / 83.0 | 0.509 | -0.981 / 79.2 |
| 4 | hard | 0.540 (0.535) | 0.967 | 0.564 (0.550) | -1.272 / 89.6 | 0.557 (0.544) | -1.314 / 90.3 |
| 4 | ramp | 0.559 | 0.918 | 0.609 | -1.158 / 85.7 | 0.605 | -1.197 / 86.7 |
| 5 | hard | 0.637 (0.633) | 0.962 | 0.660 (0.647) | -1.826 / 86.9 | 0.607 (0.602) | -1.533 / 84.8 |
| 5 | ramp | 0.640 | 0.896 | 0.690 | -1.511 / 90.5 | 0.592 | -1.282 / 85.4 |
| 6 | hard | 0.561 (0.559) | 0.991 | 0.534 (0.527) | -1.875 / 98.4 | 0.562 (0.557) | -1.835 / 97.8 |
| 6 | ramp | 0.568 | 0.966 | 0.553 | -1.833 / 97.1 | 0.579 | -1.796 / 96.3 |
| 7 | hard | 0.267 (0.368) | 0.293 | -0.057 (0.345) | -0.609 / 29.2 | - | - |
| 7 | ramp | 0.274 | 0.354 | -0.031 | -0.705 / 56.5 | - | - |
| 8 | hard | 0.381 (0.435) | 0.537 | 0.080 (0.407) | -3.741 / 53.4 | - | - |
| 8 | ramp | 0.390 | 0.563 | 0.135 | -3.332 / 86.5 | - | - |

Mean avg QWK over 8 prompts: ungated baseline 0.537, hard rule 0.517, ramp rule 0.522 (seed 42: 0.543 / 0.522 / 0.526).
Pass criterion (Content and PA <= -0.25 points and >= 50% lower; language traits unchanged by construction), hard rule: passes on 7 of 8 prompts, fails only on prompt 7 (Content 29% lower) - same as seed 42. Prompt 8 passes the test but Content QWK is 0.08.

## Further tests (hard rule; ramp in brackets). Content mean change vs the rule with the correct prompt.
| P | same-passage / nearby: change (% lower) | Content signed error |
|---|---|---|
| 1 | -2.77 (98.5%) [-2.61 (99.2%)] | -0.08 [-0.16] |
| 2 | -1.55 (73.9%) [-1.25 (91.5%)] | -0.07 [-0.15] |
| 3 | +0.01 (2.3%) [-0.02 (5.7%)] | -0.08 [-0.25] |
| 4 | -0.15 (13.9%) [-0.17 (17.5%)] | +0.22 [+0.15] |
| 5 | -0.10 (7.0%) [-0.19 (20.1%)] | +0.15 [+0.03] |
| 6 | -0.02 (1.7%) [-0.10 (10.3%)] | +0.03 [-0.01] |
| 7 | +0.45 (2.6%) [+0.37 (4.5%)] | -3.07 [-2.93] |
| 8 | -2.62 (38.9%) [-1.58 (68.0%)] | -1.85 [-1.68] |
Prompt 1: paraphrases -0.68 / -1.08 / -0.03 (hard), unseen -2.77, splice 25/50/75/100% -0.01 / -0.40 / -2.09 / -2.57.
Prompt 2: paraphrases -0.13 / -0.12 / -0.05, unseen -1.98 to -2.15, splice -0.02 / -0.20 / -1.46 / -2.12.
Prompt 3: paraphrases -0.00 / -0.01 / 0.00, unseen -1.26 to -1.36, splice -0.11 / -0.53 / -1.19 / -1.35.

Reading: seed 32 reproduces seed 42 closely on every prompt. Same-passage questions on prompts 3-6 lower Content for 2-14% of essays (hard), the rule passes the test on 7 of 8 prompts, and it under-scores on-topic narrative essays (signed error -3.1 / -1.9 on prompts 7 / 8).
