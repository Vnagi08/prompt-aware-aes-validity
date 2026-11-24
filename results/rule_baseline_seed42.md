# Rule-based (non-learned) relevance baseline, seed 42 (Kaggle CPU notebook notebookc58d6dec7c)
Script: code/19_rule_baseline.py. Baseline ProTACT weights from seed-42 run (main notebook V9, copied to dataset protact-seed42-outputs). No training.
c = cos(mean essay sentence embedding, mean prompt sentence embedding), all-MiniLM-L6-v2, prompts 3-6 with passage summaries.
Thresholds from the 7 training prompts only: lo = median c(essay, wrong prompt), hi = median c(essay, own prompt).
hard: g = 1[c >= (lo+hi)/2]; ramp: g = clip((c-lo)/(hi-lo), 0, 1). Content and PA multiplied by g (normalised space).
V2 (2026-11-24, 4,448 s): prompts 3-6 OK; prompts 1, 2, 7, 8 failed (KeyError 'prompt_adherence' - no PA label there). V3 reruns 1, 2, 7, 8 with the fix.

## Prompts 3-6 (seed 42). Wrong prompt = mean over the 7 other ASAP prompts.
| P | rule | avg QWK (base) | mean g | Content QWK (base) | Content chg / % lower / QWK wrong | PA QWK (base) | PA chg / % lower / QWK wrong |
|---|---|---|---|---|---|---|---|
| 3 | hard | 0.563 (0.573) | 0.904 | 0.570 (0.597) | -1.319 / 86.4 / 0.015 | 0.525 (0.547) | -1.193 / 85.2 / 0.013 |
| 3 | ramp | 0.554 | 0.790 | 0.544 | -1.068 / 82.6 / 0.052 | 0.505 | -0.974 / 79.0 / 0.044 |
| 4 | hard | 0.543 (0.537) | 0.967 | 0.565 (0.551) | -1.231 / 88.7 / 0.022 | 0.544 (0.528) | -1.416 / 91.9 / 0.025 |
| 4 | ramp | 0.560 | 0.918 | 0.607 | -1.125 / 84.9 / 0.063 | 0.589 | -1.271 / 88.1 / 0.076 |
| 5 | hard | 0.644 (0.640) | 0.962 | 0.659 (0.645) | -1.841 / 87.0 / 0.058 | 0.625 (0.620) | -1.610 / 85.4 / 0.052 |
| 5 | ramp | 0.648 | 0.896 | 0.687 | -1.518 / 90.7 / 0.125 | 0.616 | -1.341 / 86.7 / 0.105 |
| 6 | hard | 0.564 (0.561) | 0.991 | 0.528 (0.521) | -1.932 / 98.7 / -0.000 | 0.529 (0.522) | -2.138 / 98.9 / -0.000 |
| 6 | ramp | 0.573 | 0.966 | 0.550 | -1.890 / 97.4 / 0.001 | 0.552 | -2.091 / 97.8 / 0.001 |
Learned gate, seed 42 (kaggle_seed42.md): avg QWK 0.555 / 0.524 / 0.618 / 0.542; Content QWK 0.588 / 0.527 / 0.632 / 0.494, % lower 52 / 43 / 62 / 80; PA QWK 0.473 / 0.515 / 0.507 / 0.463, % lower 42 / 44 / 50 / 60.
Ungated baseline seed 42: avg QWK 0.573 / 0.537 / 0.640 / 0.561.
Reading (prompts 3-6): a fixed rule on the same similarity is at least as accurate as the ungated baseline (avg QWK -0.010 to +0.023) and responds far more strongly to wrong prompts (79-99% of essays lowered) than the learned gate.

## Prompts 1, 2, 7, 8 (V3, 2,873 s, fix applied; Content only - no PA label)
| P | rule | avg QWK (base) | mean g | Content QWK (base) | Content chg / % lower / QWK wrong |
|---|---|---|---|---|---|
| 1 | hard | 0.634 (0.637) | 0.988 | 0.637 (0.650) | -2.776 / 98.1 / 0.001 |
| 1 | ramp | 0.636 | 0.956 | 0.647 | -2.544 / 99.0 / 0.016 |
| 2 | hard | 0.564 (0.563) | 0.989 | 0.602 (0.598) | -1.861 / 89.3 / 0.044 |
| 2 | ramp | 0.564 | 0.948 | 0.604 | -1.769 / 93.5 / 0.051 |
| 7 | hard | 0.268 (0.376) | 0.293 | -0.061 (0.370) | -0.638 / 29.2 / 0.000 |
| 7 | ramp | 0.275 | 0.354 | -0.034 | -0.739 / 57.8 / 0.005 |
| 8 | hard | 0.392 (0.458) | 0.537 | 0.084 (0.482) | -3.532 / 53.4 / 0.001 |
| 8 | ramp | 0.401 | 0.563 | 0.138 | -3.171 / 86.2 / 0.022 |
Learned gate seed 42: avg QWK p1 0.653, p2 0.549, p7 0.332, p8 0.444; Content QWK 0.642 / 0.503 / 0.177 / 0.565; Content % lower 90 / 60 / 54 / 91.

## All 8 prompts, seed 42, average QWK
| | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | mean |
|---|---|---|---|---|---|---|---|---|---|
| ungated baseline | .637 | .563 | .573 | .537 | .640 | .561 | .376 | .458 | .543 |
| learned gate | .653 | .549 | .555 | .524 | .618 | .542 | .332 | .444 | .527 |
| hard rule | .634 | .564 | .563 | .543 | .644 | .564 | .268 | .392 | .522 |
| ramp rule | .636 | .564 | .554 | .560 | .648 | .573 | .275 | .401 | .526 |

Reading: on prompts 1-6 the fixed rule matches or beats the ungated baseline and the learned gate in accuracy and lowers 79-99% of essays under a wrong prompt. On the narrative prompts 7 and 8 the rule fails: on-topic personal stories have low similarity to the prompt, so the rule zeroes many of them (mean g 0.29-0.56) and Content QWK collapses (-0.06 / 0.08). The learned gate is much better there (Content QWK 0.18 / 0.57), so learning helps exactly where raw similarity is unreliable. Averaged over all prompts the three fixes are within 0.005 average QWK.

## Further tests on the rule (V4, 4,526 s, seed 42) — Content mean change vs the rule with the correct prompt
Format: paraphrases 1/2/3 | same-passage or nearby (% lower) | unseen range | splice 25/50/75/100% | Content signed error
hard p1 para -0.69/-1.09/-0.03 | near -2.79 (98.5%) | unseen -2.78..-2.79 | splice -0.01/-0.40/-2.10/-2.59 | se -0.06
ramp p1 para -0.96/-1.21/-0.05 | near -2.63 (99.1%) | unseen -2.49..-2.70 | splice -0.06/-0.68/-1.73/-2.44 | se -0.14
hard p2 para -0.12/-0.11/-0.05 | near -1.48 (73.6%) | unseen -1.89..-2.05 | splice -0.02/-0.19/-1.40/-2.02 | se -0.17
ramp p2 para -0.23/-0.31/-0.10 | near -1.20 (90.6%) | unseen -1.36..-1.98 | splice -0.04/-0.39/-1.19/-1.81 | se -0.24
hard p3 para -0.00/-0.01/-0.00 | near +0.01 (2.3%) | unseen -1.26..-1.36 | splice -0.11/-0.53/-1.19/-1.36 | se -0.07
ramp p3 para -0.02/0.00/-0.02 | near -0.01 (5.4%) | unseen -0.84..-1.18 | splice -0.12/-0.43/-0.86/-1.15 | se -0.25
hard p4 para -0.02/-0.03/-0.08 | near -0.14 (13.4%) | unseen -0.99..-1.28 | splice -0.02/-0.19/-0.81/-1.21 | se +0.17
ramp p4 para -0.05/-0.06/-0.12 | near -0.17 (17.2%) | unseen -0.80..-1.22 | splice -0.05/-0.29/-0.73/-1.10 | se +0.11
hard p5 para -0.02/-0.03/-0.01 | near -0.10 (7.1%) | unseen -1.91..-2.05 | splice -0.04/-0.47/-1.53/-1.99 | se +0.16
ramp p5 para -0.05/-0.08/-0.05 | near -0.18 (19.9%) | unseen -1.30..-1.92 | splice -0.13/-0.56/-1.25/-1.75 | se +0.04
hard p6 para -0.01/-0.02/-0.01 | near -0.02 (1.8%) | unseen -1.93..-1.93 | splice -0.02/-0.40/-1.77/-1.93 | se +0.09
ramp p6 para -0.04/-0.09/-0.04 | near -0.11 (11.6%) | unseen -1.82..-1.90 | splice -0.09/-0.61/-1.48/-1.89 | se +0.05
hard p7 para -0.19/+0.07/-0.06 | near +0.48 (2.6%) | unseen -0.10..-0.64 | splice +0.01/-0.24/-0.57/-0.64 | se -3.04
ramp p7 para -0.05/+0.11/-0.00 | near +0.37 (5.5%) | unseen -0.00..-0.80 | splice +0.04/-0.13/-0.44/-0.68 | se -2.89
hard p8 para +0.32/+0.19/-0.34 | near -2.46 (38.9%) | unseen -0.93..-3.56 | splice -0.06/-1.20/-3.12/-3.55 | se -2.06
ramp p8 para +0.05/+0.09/-0.43 | near -1.51 (67.6%) | unseen -0.91..-3.75 | splice -0.09/-0.95/-2.47/-3.30 | se -1.87

Comparison with the learned gate (Table VI, 5 seeds):
- Same passage, different question (p3-6): rule lowers 2-13% (hard) / 5-20% (ramp) of essays vs 4-9% for the gate -> the rule also detects the topic/passage, not the question.
- Partly off-topic: the rule reacts much earlier, e.g. 50% splice p3 -0.53 vs -0.09 (gate), p5 -0.47 vs -0.10; 75% splice -1.2 to -2.1 on p1-6 vs -0.23 to -0.34.
- Paraphrases: stable on p3-6 like the gate; less stable on p1 (-0.69, -1.09 vs at most -0.23) and p8 (+0.32 / -0.34).
- Unseen prompts: rule lowers more (-0.8 to -2.8 on p1-6).
- Narrative p7/p8: rule under-scores on-topic essays (signed error -3.0 / -2.1 vs gate -1.84 / +0.37).
