# EPCTS relevance (PR) check, run 2 (final, 2026-11-17)
Setup: 150 essays per prompt (1,200), Qwen2.5-3B-Instruct segmentation (lenient parse: 0 failed, 230 lightly repaired), all-MiniLM-L6-v2 embeddings, lambda = 0.
AUC = how well the score separates (essay, own prompt) from (essay, wrong prompt); 1.0 = perfect.

| Prompt | mean n_seg | rho(PR, n_seg) | rho(PR, words) | rho(PR w/o alpha, n_seg) | mean alpha/(n+2) | mean PR true | mean PR wrong | AUC PR | AUC PR w/o alpha | AUC sim_e only | % essays where a wrong prompt gets higher PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 15.11 | -0.606 | -0.346 | -0.411 | 0.072 | 0.432 | 0.129 | 0.993 | 0.993 | 0.994 | 0.7 |
| 2 | 10.61 | -0.561 | -0.202 | -0.270 | 0.095 | 0.523 | 0.156 | 0.996 | 0.999 | 1.000 | 0.7 |
| 3 | 2.85 | 0.267 | 0.227 | -0.273 | 0.126 | 0.411 | 0.211 | 0.905 | 0.962 | 0.972 | 15.3 |
| 4 | 2.29 | 0.616 | 0.718 | 0.235 | 0.102 | 0.515 | 0.185 | 0.955 | 0.997 | 0.998 | 2.0 |
| 5 | 3.25 | 0.211 | 0.288 | -0.007 | 0.136 | 0.464 | 0.245 | 0.921 | 0.977 | 0.985 | 2.7 |
| 6 | 3.65 | -0.096 | 0.191 | -0.019 | 0.153 | 0.592 | 0.184 | 0.991 | 1.000 | 1.000 | 0.0 |
| 7 | 6.88 | -0.507 | -0.468 | -0.335 | 0.118 | 0.267 | 0.162 | 0.768 | 0.859 | 0.899 | 36.7 |
| 8 | 20.25 | -0.552 | -0.384 | -0.300 | 0.061 | 0.268 | 0.121 | 0.841 | 0.877 | 0.910 | 31.3 |

Pooled: rho(PR, n_seg) = -0.256; AUC PR 0.907, PR w/o alpha 0.950; short off-topic (n_seg<=3) beats long on-topic (n_seg>=8) in 23.1% of pairs.
