# Stage 2: ProTACT baseline with saved weights, ablations on all 8 prompts, weight check — 2026-11-18

Kaggle committed run "stage2 baseline ablations" (Version 3, 11,150 s, 2x T4, two prompts in parallel), seed 12, 15 epochs.
Script: code/09_stage2_baseline.py. Weights saved: results/protact_p*_s12.weights.h5.

Avg QWK reproduces the earlier baseline exactly on every prompt (0.658, 0.560, 0.578, 0.541, 0.625, 0.557, 0.383, 0.403; mean 0.538).

## Input ablations (average QWK over traits, test set of the target prompt)

| Prompt | None | Prompt removed | Essay POS shuffled | Linguistic shuffled | Readability shuffled | Both shuffled |
|---|---|---|---|---|---|---|
| 1 | .658 | .658 | .658 | .519 | .189 | -.021 |
| 2 | .560 | .560 | .560 | .435 | .175 | .014 |
| 3 | .578 | .578 | .578 | .415 | .220 | -.017 |
| 4 | .541 | .541 | .541 | .405 | .210 | .003 |
| 5 | .625 | .625 | .625 | .446 | .287 | .002 |
| 6 | .557 | .557 | .557 | .424 | .198 | .012 |
| 7 | .383 | .383 | .383 | .284 | .137 | -.009 |
| 8 | .403 | .403 | .403 | .339 | .151 | .024 |

Wrong-prompt test: max |mean change| = 0.000000 for every trait, every wrong prompt, every target prompt.

## Weight check (trained vs initial weights)

On every prompt, 82 of 176 weight tensors are bit-identical to their initial values (max |diff| = 0):
- prompt encoder (19): prompt/embeddings, pos_prompt/embeddings, prompt_zcnn (2), prompt_avg_zcnn (2), multi_head_attention_9 (8), lstm_9 (3), attention_11 (2)
- query and key projections of the 9 prompt-essay attention layers (multi_head_attention_pe*, dense kernel+bias for query and key: 36)
- after that attention (length-one sequences): recurrent_kernel of lstm_10..18 (9) and att_v/att_W of attention_12..20 (18)
Value and output projections of the prompt-essay attention, and the LSTM input kernels, do change.
