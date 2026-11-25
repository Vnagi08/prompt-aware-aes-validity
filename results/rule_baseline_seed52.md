# Fixed similarity rule, seed 52 (Kaggle CPU notebook notebookc58d6dec7c, version 6 "rule seed52")
Script: code/20_rule_stage4.py. Ungated ProTACT weights of seed 52 (training notebook V8), attached as dataset protact-seed42-outputs v3. No training.
Format per prompt: avg QWK (base) | mean g | Content QWK (base) | Content change / % lower under a wrong prompt | PA QWK (base) | PA change / % lower.

| P | rule | avg QWK (base) | mean g | Content QWK (base) | Content chg / % lower | PA QWK (base) | PA chg / % lower |
|---|---|---|---|---|---|---|---|
| 1 | hard | 0.646 (0.648) | 0.988 | 0.654 (0.667) | -2.886 / 98.1 | - | - |
| 1 | ramp | 0.649 | 0.956 | 0.668 | -2.646 / 99.0 | - | - |
| 2 | hard | 0.562 (0.562) | 0.989 | 0.598 (0.595) | -1.795 / 89.1 | - | - |
| 2 | ramp | 0.561 | 0.948 | 0.591 | -1.710 / 93.2 | - | - |
| 3 | hard | 0.557 (0.566) | 0.904 | 0.563 (0.586) | -1.295 / 86.3 | 0.522 (0.545) | -1.177 / 84.9 |
| 3 | ramp | 0.548 | 0.790 | 0.539 | -1.051 / 82.1 | 0.500 | -0.963 / 78.4 |
| 4 | hard | 0.544 (0.539) | 0.967 | 0.560 (0.547) | -1.143 / 86.8 | 0.562 (0.551) | -1.236 / 89.2 |
| 4 | ramp | 0.560 | 0.918 | 0.598 | -1.044 / 82.5 | 0.604 | -1.132 / 85.6 |
| 5 | hard | 0.642 (0.638) | 0.962 | 0.660 (0.647) | -1.843 / 86.9 | 0.616 (0.612) | -1.586 / 85.2 |
| 5 | ramp | 0.646 | 0.896 | 0.690 | -1.524 / 90.8 | 0.608 | -1.319 / 86.0 |
| 6 | hard | 0.568 (0.565) | 0.991 | 0.524 (0.517) | -1.863 / 98.4 | 0.543 (0.538) | -2.090 / 98.7 |
| 6 | ramp | 0.577 | 0.966 | 0.546 | -1.819 / 97.1 | 0.567 | -2.046 / 97.3 |
| 7 | hard | 0.253 (0.355) | 0.293 | -0.059 (0.350) | -0.615 / 29.2 | - | - |
| 7 | ramp | 0.260 | 0.354 | -0.032 | -0.715 / 57.1 | - | - |
| 8 | hard | 0.391 (0.455) | 0.537 | 0.084 (0.468) | -3.574 / 53.4 | - | - |
| 8 | ramp | 0.400 | 0.563 | 0.139 | -3.204 / 86.3 | - | - |

Mean avg QWK: baseline 0.541, hard 0.520, ramp 0.525. Hard rule passes the test on 7 of 8 prompts (fails on prompt 7).
Further tests, hard rule: same-passage / nearby (% lower) p1 98.5, p2 73.4, p3 2.3, p4 12.2, p5 7.0, p6 1.7, p7 2.6, p8 38.9; Content signed error p1..p8 +0.05, -0.24, -0.10, +0.08, +0.16, +0.02, -3.07, -2.02.
p1: paraphrases -0.71 / -1.13 / -0.03, unseen -2.90, splice 25/50/75/100% -0.01 / -0.42 / -2.19 / -2.69. p2: paraphrases -0.12 / -0.10 / -0.05, unseen -1.83 to -1.98, splice -0.02 / -0.18 / -1.35 / -1.95. p3: unseen -1.24 to -1.34, splice -0.10 / -0.52 / -1.17 / -1.33.

## Three seeds (32, 42, 52), mean average QWK per prompt (Table VII of the paper)
| | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | mean |
|---|---|---|---|---|---|---|---|---|---|
| ungated baseline | .642 | .559 | .570 | .537 | .637 | .562 | .366 | .449 | .540 |
| learned gate | .647 | .549 | .554 | .527 | .614 | .545 | .326 | .442 | .526 |
| hard rule | .639 | .560 | .560 | .542 | .641 | .564 | .263 | .388 | .520 |
| ramp rule | .642 | .560 | .551 | .560 | .645 | .573 | .270 | .397 | .525 |
Content % lower: gate 89 58 51 44 63 79 56 92 (67); hard rule 98 89 86 88 87 99 29 53 (79). Sample SD across seeds (ddof=1, as in Table V) of the rule entries: at most 0.008 in the QWK rows (prompt 7) and 1.4 points in the Content % lower row (prompt 4).
