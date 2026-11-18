# Fix runs (seed 12, Kaggle 2x T4) — 2026-11-18

## Fix v1 (05_fix_mismatch.py): multi-key attention over POS sentence states + mismatch training
2-epoch test, prompt 1: 9513 originals + 4756 mismatched copies.
Wrong-prompt mean change (score points): score +0.0005, content 0.0000, organization 0.0000,
word choice +0.0012, sentence fluency +0.0004, conventions 0.0000. Test QWK: content 0.222.
Conclusion: the essay is represented only by POS tags, which carry no meaning, so the prompt
cannot be compared with it. Fixing the single key alone is not enough.

## Fix v2 (06_semantic_fix.py): semantic relevance branch (MiniLM sentence embeddings) + mismatch training
Sanity check: whole-essay cosine picks the essay's own prompt for 86.4% of essays (8-way).
Prompt texts in data/prompt_info_pp.csv are preprocessed (stop words removed).

2-epoch test, prompt 1 (best dev epoch 2, avg test QWK 0.579):

| Trait | QWK | Wrong prompt: mean change | % essays lower | QWK with wrong prompt |
|---|---|---|---|---|
| Overall | 0.801 | +0.318 | 0 | 0.818 |
| Content | 0.365 | -1.008 | 90 | 0.095 |
| Organization | 0.574 | -0.610 | 61 | 0.489 |
| Word choice | 0.616 | -0.508 | 51 | 0.555 |
| Sentence fluency | 0.586 | -0.507 | 51 | 0.466 |
| Conventions | 0.531 | -0.540 | 54 | 0.541 |

The prompt now changes Content strongly, but after 2 epochs the language traits also drop (about half
as much), so the response is not yet trait-selective. Full 15-epoch run on all 8 prompts started 02:20 IST
(tag `sem`, results/sem_p*_s12.json).

### v2 full 15-epoch run, prompt 1 (memory-leak fix applied; best dev epoch 13, avg test QWK 0.603; ProTACT baseline 0.658)

| Trait | QWK (true prompt) | ProTACT QWK | Wrong prompt: mean change | % essays lower | QWK with wrong prompt |
|---|---|---|---|---|---|
| Overall | 0.762 | 0.809 | +0.262 | 0 | 0.797 |
| Content | 0.464 | 0.661 | -1.111 | 93 | 0.129 |
| Organization | 0.590 | 0.618 | -0.710 | 71 | 0.414 |
| Word choice | 0.621 | 0.642 | -0.448 | 45 | 0.443 |
| Sentence fluency | 0.601 | 0.614 | -0.464 | 46 | 0.499 |
| Conventions | 0.576 | 0.603 | -0.577 | 58 | 0.529 |

Not trait-selective, and Content QWK drops by 0.20. Likely cause: the relevance vector enters every trait and also
spreads through ProTACT's trait-relation attention. v2 stopped after prompt 1 (kept as an ablation).

## Fix v3 (07_gate_fix.py): relevance gate on Content and Prompt Adherence only
ProTACT unchanged; Content and PA outputs multiplied by g = sigmoid(MLP(semantic stats)), bias initialised so g≈0.95.
Language traits receive no prompt information by construction.

Infrastructure notes: running two training jobs in parallel exceeded Kaggle's 31 GB RAM (the per-epoch fit()
call re-copies the training set). Fixed by building the tf.data dataset once; runs use one GPU queue.

### v3 2-epoch test, prompt 1
| Trait | QWK | Wrong prompt: mean change | % essays lower | QWK with wrong prompt |
|---|---|---|---|---|
| Content | 0.461 | -0.796 | 77 | 0.178 |
| Organization | 0.610 | 0.000 | 0 | 0.610 |
| Word choice | 0.602 | 0.000 | 0 | 0.602 |
| Sentence fluency | 0.545 | 0.000 | 0 | 0.545 |
| Conventions | 0.586 | 0.000 | 0 | 0.586 |
Trait-selective as designed. Full run (tag `gate`, all 8 prompts, one GPU queue) started 04:58 IST.

## Fix v3 full run — all 8 prompts (Kaggle committed run "fix v3 gate all prompts", Version 2, 23,736 s, 15 epochs, seed 12)

Wrong-prompt test (each test essay paired with each of the other 7 prompts). Gated traits only; every
ungated trait (Overall, Organization, Word choice, Sentence fluency, Conventions, Language, Narrativity)
changed by exactly 0.000 on every prompt.

| Prompt | Avg QWK ProTACT | Avg QWK v3 | Content QWK ProTACT | Content QWK v3 | PA QWK ProTACT | PA QWK v3 | Content change (wrong prompt) | % essays lower | Content QWK with wrong prompt | PA change | PA % lower |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | .658 | .628 | .661 | .458 | – | – | -0.776 | 74 | .181 | – | – |
| 2 | .560 | .547 | .599 | .478 | – | – | -0.589 | 57 | .247 | – | – |
| 3 | .578 | .518 | .592 | .528 | .565 | .356 | -0.445 | 45 | .307 | -0.335 | 34 |
| 4 | .541 | .507 | .552 | .491 | .546 | .434 | -0.399 | 40 | .364 | -0.360 | 36 |
| 5 | .625 | .568 | .639 | .620 | .580 | .378 | -0.560 | 54 | .381 | -0.364 | 36 |
| 6 | .557 | .528 | .514 | .479 | .523 | .408 | -0.823 | 77 | .141 | -0.553 | 55 |
| 7 | .383 | .345 | .379 | .195 | – | – | -0.711 | 64 | .106 | – | – |
| 8 | .403 | .454 | .409 | .591 | – | – | -1.840 | 92 | .306 | – | – |
| Mean | .538 | .512 | .543 | .480 | .553 | .394 | -0.768 | 62.9 | .254 | -0.403 | 40.3 |

Best dev epochs: 14, 15, 13, 11, 15, 10, 14, 14 (prompts 1–8).

Summary: the fix makes the scorer pass the wrong-prompt test (Content and PA fall; language traits exactly unchanged),
at a cost in accuracy on the gated traits: Content QWK -0.063 and PA QWK -0.159 on average (avg over all traits -0.026).
Content improves on prompt 8 (+0.18). The gate likely also lowers some on-topic essays whose embedding similarity is low.
