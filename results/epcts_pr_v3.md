# EPCTS prompt-relevance feature, rerun with per-essay ranking metrics (v3) — 2026-11-19

Kaggle committed run "EPCTS rerun per-essay metrics" (Version 5), 22 min, 150 essays per prompt (1,200 essays x 8 prompts = 9,600 rows).
Script: code/epcts_pr.py (--per_prompt 150 --out results/epcts_pr_v3). Segmentation Qwen2.5-3B-Instruct, embeddings all-MiniLM-L6-v2, lambda = 0.

## Per-essay ranking of the 8 candidate prompts (pooled over all essays)

| Score | Top-1 (own prompt ranked first) | MRR | Pairwise (= per-essay AUC) | Pooled AUC |
|---|---|---|---|---|
| PR | 0.8883 | 0.9288 | 0.9648 | 0.907 |
| PR without alpha | 0.8883 | 0.9288 | 0.9648 | 0.950 |
| sim_e (whole essay) | 0.9200 | 0.9477 | 0.9725 | 0.964 |
| mean segment similarity | 0.8717 | 0.9181 | 0.9598 | 0.938 |

Identity check: per-essay metrics identical with and without alpha = True (confirms Eq. prdiff on real data).
Per-prompt table (rho, AUCs, wrong-prompt wins) matches epcts_pr_v2.

Paper updated: EPCTS section now reports these numbers.
