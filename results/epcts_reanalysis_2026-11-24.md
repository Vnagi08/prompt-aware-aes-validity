# EPCTS PR re-analysis: n+1 variant and bootstrap CIs (2026-11-24)

Data: Kaggle run "EPCTS rerun per-essay metrics" (V5, session 371190038), epcts_results/epcts_pr_v3.csv, 1,200 essays x 8 prompts = 9,600 rows (no failed segmentations). Computed in the browser from the CSV.
Variants: PR (paper formula, n+2 divisor, with alpha); PR w/o alpha (n+2 divisor); PR mean = (sim_e + sum sim_i)/(n+1) (true mean of the n+1 similarities); sim_e.
CIs: essay-level bootstrap (resample essays with all 8 pairs), 1,000 reps, percentile 95%.

## Pooled own-vs-wrong AUC
| Score | AUC | 95% CI |
|---|---|---|
| PR | 0.907 | [0.902, 0.918] |
| PR w/o alpha (n+2) | 0.950 | [0.945, 0.958] |
| PR mean (n+1) | 0.944 | [0.939, 0.952] |
| sim_e | 0.964 | [0.959, 0.971] |

Paired differences (bootstrap 95% CI): w/o alpha - PR = [+0.037, +0.046]; (n+1) - (n+2) = [-0.006, -0.005]; sim_e - (n+1) = [+0.017, +0.021]. All exclude 0.

## Per prompt (AUC; Spearman rho of own-prompt score with n)
| P | AUC PR | AUC n+2 | AUC n+1 | AUC sim_e | rho PR | rho n+2 | rho n+1 | rho sim_e |
|---|---|---|---|---|---|---|---|---|
| 1 | .993 | .993 | .993 | .994 | -0.61 | -0.41 | -0.53 | 0.00 |
| 2 | .996 | .999 | .999 | 1.000 | -0.56 | -0.27 | -0.43 | 0.04 |
| 3 | .905 | .962 | .960 | .972 | 0.27 | -0.27 | -0.51 | -0.15 |
| 4 | .955 | .997 | .997 | .998 | 0.62 | 0.23 | -0.10 | 0.33 |
| 5 | .921 | .977 | .977 | .985 | 0.21 | -0.01 | -0.28 | 0.01 |
| 6 | .991 | 1.000 | 1.000 | 1.000 | -0.10 | -0.02 | -0.30 | 0.04 |
| 7 | .768 | .859 | .853 | .899 | -0.51 | -0.33 | -0.43 | -0.16 |
| 8 | .841 | .877 | .876 | .910 | -0.55 | -0.30 | -0.35 | -0.07 |

Short wrong (n<=3) beats long own (n>=8): PR 23.1%, n+2 9.5%, n+1 13.0%, sim_e 6.6%.
Within-essay top-1 is 88.83% for PR, n+2 and n+1 (identical, as expected: same divisor for all prompts of an essay).

Reading: removing alpha is what helps; the choice of divisor barely matters (n+1 is 0.006 lower). The (n+1) mean is still length-dependent (rho -0.10 to -0.53), because segment similarities are lower than sim_e and more segments pull the mean down. sim_e beats all PR variants on every prompt.

# Seed 22, prompts 6-8: values pulled from Google Drive (MyDrive/protact_seed22_results), 2026-11-24
gatev4r25_p6_s22.json: Content change -0.941, % lower 83.09, wrong-prompt QWK 0.166; PA change -0.730, % lower 69.94, QWK 0.210; other traits 0.000.
gatev4r25_p7_s22.json: Content change -0.596, % lower 56.09, wrong-prompt QWK 0.109.
gatev4r25_p8_s22.json: Content change -2.024, % lower 91.58, wrong-prompt QWK 0.321.
gatev4r25_stage4_p8_s22.json: mean g 0.860, signed error +0.414; paraphrases -0.187/-0.015/-0.245; near-miss -1.353 (86.2% lower); unseen U1-U6 -2.777/-2.672/-2.651/-3.133/-1.354/-1.480; splice 25/50/75/100% -0.131/-0.320/-0.732/-2.368.
Stage 4 for gate p6, p7 seed 22: none (OOM on Colab).

Five-seed Table V after filling: Content % lower 89 59 52 43 64 80 56 92 (mean 67); Content wrong-prompt QWK p6 0.170, p8 0.342, mean 0.280; PA % lower 42 45 50 64 (mean 50).
Table VI prompt 8 (5 seeds): para -0.14/+0.00/-0.20; near-miss -1.18 (83); unseen -1.17 to -2.92; splice -0.11/-0.27/-0.68/-2.25; signed error +0.37. Max SD of a Table VI entry now 0.12 (p8 near-miss/unseen).
Paper updated accordingly (main.tex), plus the EPCTS n+1 and bootstrap-CI sentence.
