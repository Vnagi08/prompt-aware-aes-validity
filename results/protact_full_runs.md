# ProTACT full runs (seed 12, 15 epochs, Kaggle 2x T4) — started 2026-11-17

| Prompt | Best dev epoch | Avg QWK | Overall | Content | Org | WC | SF | Conv | PA | Lang | Narr | Max wrong-prompt change | Wrong-prompt QWK change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 15 | 0.658 | 0.809 | 0.661 | 0.618 | 0.642 | 0.614 | 0.603 | – | – | – | 0.0 | 0.0 |
| 2 | 15 | 0.560 | 0.623 | 0.599 | 0.565 | 0.530 | 0.525 | 0.517 | – | – | – | 0.0 | 0.0 |
| 3 | – | 0.578 | 0.606 | 0.592 | – | – | – | – | 0.565 | 0.544 | 0.581 | 0.0 | 0.0 |
| 4 | – | 0.541 | 0.586 | 0.552 | – | – | – | – | 0.546 | 0.472 | 0.551 | 0.0 | 0.0 |
| 5 | – | 0.625 | 0.740 | 0.639 | – | – | – | – | 0.580 | 0.544 | 0.624 | 0.0 | 0.0 |
| 6 | – | 0.557 | 0.615 | 0.514 | – | – | – | – | 0.523 | 0.555 | 0.577 | 0.0 | 0.0 |
| 7 | – | 0.383 | 0.727 | 0.379 | 0.257 | – | – | 0.168 | – | – | – | 0.0 | 0.0 |
| 8 | – | 0.403 | 0.448 | 0.409 | 0.406 | 0.491 | 0.391 | 0.273 | – | – | – | 0.0 | 0.0 |

Mean over 8 prompts: 0.538. Reference (Li & Ng 2026, Table 17, ProTACT, 5 seeds): .647 .587 .623 .632 .674 .584 .446 .541, avg .592±.016.

Wrong-prompt test: prompt text replaced by each of the other 7 prompts. The LDA feature is a property of the essay alone (probability of its dominant topic), so it does not change.
