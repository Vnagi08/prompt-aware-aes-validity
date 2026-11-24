# Gate without passage summaries, seed 12, prompts 3-6 (Kaggle Version 12, session 373240184)
Started 2026-11-24 ~04:57 IST, finished in 10,253 s (2.8 h), 4/4 OK. Output: /kaggle/working/seed12_results/nosum25_*.json.
Setup: notebook 16_kaggle_seed.py (V11) with SEED='12', BASE_PROMPTS='', GATE_PROMPTS='3 4 5 6', tag nosum25, SEM_SUMMARIES default '0' (embedding build + Stage 4). Mismatch share 0.25.
Whole-essay cosine picks own prompt for 85.2% of essays (95.0% with summaries).

## Main results, seed 12 (no summaries vs with summaries)
| P | avg QWK | Content QWK | PA QWK | Content chg / % lower / QWK wrong | PA chg / % lower / QWK wrong |
|---|---|---|---|---|---|
| 3 | 0.545 (0.552) | 0.558 (0.579) | 0.426 (0.442) | -0.477 / 48 / 0.318 (-0.528 / 52.8) | -0.381 / 38 / 0.305 (-0.421 / 42.2) |
| 4 | 0.523 (0.524) | 0.522 (0.524) | 0.505 (0.504) | -0.386 / 39 / 0.393 (-0.433 / 43.3) | -0.392 / 39 / 0.375 (-0.439 / 43.9) |
| 5 | 0.600 (0.602) | 0.624 (0.620) | 0.503 (0.512) | -0.575 / 56 / 0.470 (-0.681 / 66.0) | -0.428 / 43 / 0.308 (-0.495 / 49.3) |
| 6 | 0.548 (0.551) | 0.503 (0.511) | 0.474 (0.481) | -0.882 / 80 / 0.178 (-0.909 / 81.4) | -0.629 / 61 / 0.220 (-0.654 / 63.9) |
Language traits change 0.000 everywhere.

## Stage 4, seed 12, no summaries (with-summaries seed 12 in brackets)
| P | mean g | signed err | paraphrases | same-passage (% lower) | unseen range | splice 25/50/75/100 |
|---|---|---|---|---|---|---|
| 3 | 0.929 (0.939) | -0.04 (-0.01) | -0.01/-0.02/-0.07 | -0.03 (5.3) [-0.043 (4.6)] | -0.46 to -0.72 | -0.06/-0.14/-0.31/-0.59 |
| 4 | 0.977 (0.978) | +0.14 (+0.15) | -0.03/-0.03/-0.06 | -0.155 (15.7) [-0.028 (3.0)] | -0.22 to -0.84 | -0.01/-0.05/-0.22/-0.46 |
| 5 | 0.944 (0.962) | +0.17 (+0.25) | +0.01/-0.04/+0.02 | -0.116 (15.5) [-0.091 (9.4)] | -0.52 to -1.13 | -0.06/-0.16/-0.36/-0.83 |
| 6 | 0.970 (0.974) | +0.04 (+0.04) | +0.01/-0.01/0.00 | -0.266 (27.1) [-0.061 (6.1)] | -0.65 to -1.01 | -0.04/-0.11/-0.29/-0.87 |

Reading: summaries are not doing the work. Without them accuracy is almost the same (avg QWK -0.001 to -0.007), the wrong-prompt response is somewhat weaker (Content % lower 4-10 points lower), and the same-passage/different-question test reacts MORE (5-27% lowered vs 3-9%) — the summaries (appended to every variant of a source prompt) made different questions on the same passage look alike.
