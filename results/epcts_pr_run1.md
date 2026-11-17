# EPCTS relevance (PR) check, run 1 (2026-11-17)
Setup: 150 essays per prompt (1,200), Qwen2.5-3B-Instruct segmentation, all-MiniLM-L6-v2 embeddings, lambda=0.
CAVEAT: 442/1200 segmentations failed to parse and fell back to sentence-level segments (inflates n_seg, esp. P1, P2, P8). Rerun needed.

| Prompt | mean n_seg | rho(PR, n_seg) | rho(PR, words) | rho(PR w/o alpha, n_seg) | mean PR true | mean PR wrong | mean alpha/(n+2) | AUC PR | AUC PR w/o alpha | AUC sim_e only | % essays where a wrong prompt gets higher PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20.45 | -0.685 | -0.426 | -0.502 | 0.400 | 0.120 | 0.062 | 0.990 | 0.993 | 0.994 | 0.7 |
| 2 | 13.69 | -0.686 | -0.308 | -0.410 | 0.497 | 0.149 | 0.087 | 0.995 | 0.999 | 1.000 | 0.7 |
| 3 | 3.47 | -0.090 | -0.090 | -0.220 | 0.426 | 0.232 | 0.148 | 0.907 | 0.958 | 0.972 | 16.7 |
| 4 | 2.95 | 0.330 | 0.525 | 0.282 | 0.551 | 0.230 | 0.147 | 0.959 | 0.997 | 0.998 | 2.7 |
| 5 | 3.65 | -0.035 | 0.080 | 0.012 | 0.477 | 0.259 | 0.151 | 0.942 | 0.977 | 0.985 | 2.7 |
| 6 | 4.17 | -0.318 | 0.064 | -0.059 | 0.586 | 0.187 | 0.155 | 0.991 | 1.000 | 1.000 | 0.0 |
| 7 | 8.17 | -0.690 | -0.527 | -0.349 | 0.270 | 0.166 | 0.122 | 0.769 | 0.860 | 0.899 | 36.7 |
| 8 | 32.16 | -0.606 | -0.457 | -0.340 | 0.239 | 0.107 | 0.048 | 0.824 | 0.865 | 0.910 | 36.0 |

Pooled: rho(PR, n_seg) = -0.449; AUC PR 0.893 vs w/o alpha 0.947; short off-topic (n_seg<=3) beats long on-topic (n_seg>=8) in 31.3% of pairs.
