# Stage 3/4: relevance gate v4 (raw prompts + our passage summaries for prompts 3-6), seed 12 — 2026-11-19

Kaggle committed run "stage3 gate v4 seed12" (Version 4), 8 prompts x ~48 min, 15 epochs, mismatch ratio 0.5.
Script: code/10_stage3_gate.py. Whole-essay cosine own-prompt top-1: 95.0% (was 86.4% with stop-word-stripped prompts).

## Accuracy (test QWK, correct prompt) and wrong-prompt test (seen training prompts as wrong prompts)

| Prompt | Avg v3 | Avg v4 | Content v3 | Content v4 | PA v3 | PA v4 | Content % lower, wrong prompt v3 -> v4 | Content change v4 |
|---|---|---|---|---|---|---|---|---|
| 1 | .628 | .630 | .458 | .468 | – | – | 74 -> 82 | -0.880 |
| 2 | .547 | .546 | .478 | .476 | – | – | 57 -> 61 | -0.637 |
| 3 | .518 | .530 | .528 | .559 | .356 | .379 | 45 -> 49 | -0.491 |
| 4 | .507 | .512 | .491 | .506 | .434 | .447 | 40 -> 44 | -0.441 |
| 5 | .568 | .576 | .620 | .628 | .378 | .399 | 54 -> 67 | -0.690 |
| 6 | .528 | .529 | .479 | .475 | .408 | .421 | 77 -> 78 | -0.840 |
| 7 | .345 | .330 | .195 | .143 | – | – | 64 -> 55 | -0.594 |
| 8 | .454 | .444 | .591 | .564 | – | – | 92 -> 89 | -1.681 |
| Mean | .512 | .512 | .480 | .477 | .394 | .412 | 62.9 -> 65.6 | |

ProTACT baseline (same seed): avg .538, Content .543, PA .554. Ungated traits change by 0.000 (by construction).

## Wrong-prompt test, full v4 numbers (mean over 7 wrong prompts)

| Prompt | Content change | Content % lower | Content QWK wrong prompt | PA change | PA % lower | PA QWK wrong prompt |
|---|---|---|---|---|---|---|
| 1 | -0.880 | 81.8 | .155 | – | – | – |
| 2 | -0.637 | 60.7 | .219 | – | – | – |
| 3 | -0.491 | 49.2 | .284 | -0.366 | 36.7 | .303 |
| 4 | -0.441 | 44.1 | .355 | -0.402 | 40.2 | .348 |
| 5 | -0.690 | 66.8 | .347 | -0.451 | 44.8 | .210 |
| 6 | -0.840 | 78.1 | .138 | -0.577 | 57.2 | .208 |
| 7 | -0.594 | 55.3 | .099 | – | – | – |
| 8 | -1.681 | 88.6 | .241 | – | – | – |
| Mean | -0.782 | 65.6 | .230 | -0.449 | 44.7 | .267 |

Paper (main.tex) updated 2026-11-19: Table 5 now reports gate v4; new Section "Beyond the wrong-prompt test" with Table 6 (stage 4, prompts 1-6); abstract, contribution 5, limitations and conclusion updated.

## Stage 4 tests (prompts 1-6; Content mean change in score points, % essays lower in brackets)

Stage 4 output files for prompts 7 and 8 were not produced (cause not yet diagnosed; log files not previewable in the viewer).

| Prompt | Paraphrases (3) | Near-miss | Unseen prompts (6), range | Splice 25 / 50 / 75 / 100 % off-topic | Stop-word-stripped prompt |
|---|---|---|---|---|---|
| 1 | -0.21, -0.16, -0.04 | -0.86 (81) | -0.87 to -1.09 | -0.04 / -0.11 / -0.23 / -0.91 | -0.03 |
| 2 | -0.08, -0.09, -0.03 | -0.49 (48) | -0.58 to -0.85 | -0.03 / -0.09 / -0.20 / -0.66 | -0.04 |
| 3 | -0.02, -0.02, -0.05 | -0.04 (5) | -0.47 to -0.62 | -0.04 / -0.12 / -0.26 / -0.55 | -0.10 |
| 4 | -0.03, -0.03, -0.03 | -0.04 (4) | -0.34 to -0.70 | -0.01 / -0.07 / -0.26 / -0.49 | -0.08 |
| 5 | -0.02, -0.06, -0.03 | -0.12 (12) | -0.66 to -1.02 | -0.04 / -0.13 / -0.32 / -0.83 | -0.16 |
| 6 | -0.01, -0.02, -0.02 | -0.07 (7) | -0.72 to -0.89 | -0.04 / -0.11 / -0.28 / -0.84 | -0.08 |

Near-miss prompts: 1 = television instead of computers, 2 = library opening hours; 3-6 = a different question about the same source passage (with the same summary); see stage4_prompts.json.

## Gate diagnostics on on-topic test essays (correct prompt)

| Prompt | median g | 5th pct g | rho(g, #sentences) | rho(g, gold Content) | mean signed error Content (pred - gold) | PA signed error |
|---|---|---|---|---|---|---|
| 1 | .957 | .861 | -.15 | .10 | -0.66 | – |
| 2 | .965 | .842 | -.04 | .17 | -0.52 | – |
| 3 | .964 | .730 | -.17 | .05 | -0.06 | -0.41 |
| 4 | .982 | .857 | -.13 | .16 | +0.11 | +0.06 |
| 5 | .964 | .810 | .00 | .13 | -0.01 | -0.73 |
| 6 | .981 | .902 | -.12 | .09 | -0.06 | -0.38 |

Baseline ProTACT signed errors not yet computed (weights saved in Version 3 output), so the bias cannot yet be attributed to the gate.

## Reading
- Unseen prompts: the gate responds as strongly as to seen wrong prompts, so it is not memorising the 7 training prompts.
- Paraphrases: small changes (mostly <= 0.1 points, <= 9% of essays), except prompt 1 (up to -0.21, 21%).
- Near-miss: for source prompts 3-6, a different question about the same passage barely changes Content (<= 12%): the gate detects the topic/passage, not the question asked.
- Splice: monotone in all prompts, but convex: small until most of the essay is off-topic.
