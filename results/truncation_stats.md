# How much of each essay the sentence encoder sees (review item 11) — 2026-11-25
Script: code/23_truncation_stats.py, Kaggle CPU notebook notebookf6bcb45702 v1. All 12,976 essays in ProTACT's preprocessed files (content_text), NLTK sentence split, all-MiniLM-L6-v2 tokenizer (limit 256 word pieces incl. [CLS]/[SEP]). The gate embeds at most 40 sentences per essay; Section VI's sim_e embeds the whole essay as one input.

| prompt | essays | essays >40 sentences (%) | sentences beyond 40 (%) | seen sentences >256 wp (%) | whole essay >256 wp (%) | median whole-essay wp |
|---|---|---|---|---|---|---|
| 1 | 1783 | 2.6 | 0.59 | 0.00 | 91.0 | 452 |
| 2 | 1798 | 3.0 | 0.95 | 0.02 | 87.8 | 445 |
| 3 | 1726 | 0.0 | 0.00 | 0.00 | 3.2 | 122 |
| 4 | 1772 | 0.0 | 0.00 | 0.00 | 2.8 | 106 |
| 5 | 1805 | 0.0 | 0.00 | 0.00 | 11.1 | 154 |
| 6 | 1800 | 0.0 | 0.00 | 0.00 | 20.3 | 201 |
| 7 | 1569 | 0.9 | 0.56 | 0.01 | 29.3 | 195 |
| 8 | 723 | 33.6 | 12.28 | 0.04 | 97.0 | 781 |

Reading: per-sentence truncation is negligible for the gate (<0.05% of embedded sentences). The 40-sentence cap matters only on prompt 8 (34% of essays, 12% of sentences unseen). sim_e in Table IV sees only the first 256 word pieces for 88-97% of essays on prompts 1, 2 and 8; Table IV was not repeated with chunked averaging.
