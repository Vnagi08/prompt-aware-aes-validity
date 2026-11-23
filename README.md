# Do Prompt-Aware Essay Trait Scorers Really Read the Prompt?

Code and results for the paper *"Do Prompt-Aware Essay Trait Scorers Really Read the Prompt? A Trait-Level Validity Test and Fix"* (Piyush Parihar and Satish Kumar Singh, Department of Information Technology, Netaji Subhas University of Technology, New Delhi).

## Main findings

1. **ProTACT ignores the prompt.** The essay is pooled to a single vector before the prompt attention, so the attention has one key and its weight is always 1. Replacing the prompt with any of the other 7 ASAP prompts changes no trait score, including Prompt Adherence, on all 8 prompts (56 wrong-prompt conditions, maximum change 0.0).
2. **Scores come from hand-crafted features.** Comparing each trained model with its initialisation, 82 of 176 weight tensors (the whole prompt encoder among them) are bit-identical on all 8 prompts: they never receive a gradient. Removing the prompt or shuffling the essay POS sequences leaves the average QWK unchanged on all 8 prompts; shuffling both hand-crafted feature sets drops it to between -0.021 and 0.024.
3. **The EPCTS relevance feature has a length-dependent, prompt-independent term.** For a fixed essay the term is the same for every candidate prompt, so it cannot change which prompt the essay ranks highest; it adds a length-dependent offset that makes the feature less comparable across essays. Removing it raises the pooled separation of own-prompt from wrong-prompt pairs (AUC 0.907 to 0.950). A plain whole-essay similarity does better still.
4. **What a fix requires.** Making ProTACT's attention multi-key is not enough: the essay is represented only by POS tags, so a wrong prompt still changes nothing. A sentence-embedding relevance signal fed to every trait makes Content respond, but it leaks into language traits through ProTACT's trait-relation attention. A relevance gate on Content and Prompt Adherence only, trained with deliberately mismatched prompts (a quarter of the training essays), passes the wrong-prompt test on all 8 prompts in all 40 runs (5 seeds x 8 prompts). Averaged over five seeds, Content falls for 66% of essays under a wrong prompt (mean -0.83 points), Prompt Adherence for 50% (mean -0.51), and language traits do not change. Average QWK over all traits is 0.524 for the gate against 0.540 for ProTACT (paired difference -0.016, 95% bootstrap interval [-0.018, -0.013]); mean Content QWK 0.508 against 0.543.
5. **What the gate does and does not detect (prompts 1-6).** It reacts to prompts it never saw in training about as strongly as to seen wrong prompts, and paraphrasing the correct prompt changes little. But for source-based prompts a different question about the same reading passage barely changes Content (4-12% of essays lower): the gate detects the topic, not the question asked. Partly off-topic essays are lowered only when most of the essay is off-topic. On the narrative prompt 7 the gate also partly closes for on-topic essays (median gate 0.70), which explains its Content QWK drop there.
6. **The mismatch share matters.** With seed 12, training with mismatched copies of 25% instead of 50% of the essays raised Content QWK on prompt 1 from 0.468 to 0.634 (ProTACT: 0.661) and made the wrong-prompt response stronger; on prompt 2 it helped only slightly (0.476 to 0.489). The five-seed runs use 25%. Results vary little across seeds (SD of average QWK at most 0.026 per prompt).

## Repository layout

| Path | Contents |
|---|---|
| `code/01_setup_protact.py` | Clones [ProTACT](https://github.com/doheejin/ProTACT), installs tf_keras 2.21, applies three compatibility fixes (sklearn `get_scorer`, NumPy `np.cast` shim, NLTK data) and sets 15 epochs. |
| `code/02_make_wrong_prompt_script.py` | Downloads GloVe 6B 50d if needed and builds `train_pilot.py`: ProTACT training plus the wrong-prompt test. |
| `code/03_run_all_prompts.py` | Trains all 8 target prompts (seed 12) on two GPUs and runs the wrong-prompt test after each. |
| `code/04_diag_ablations_v1.py`, `code/04_diag_ablations_v2.py` | Input ablations on prompt 1: prompt swap, empty prompt, and shuffles of essay text and feature sets. |
| `code/epcts_pr.py` | Re-implementation of the EPCTS prompt-relevance feature and its length and wrong-prompt checks. |
| `code/05_fix_mismatch.py` | Fix variant 1: multi-key prompt attention over the essay's sentence states, with mismatch training (fails). |
| `code/06_semantic_fix.py` | Fix variant 2: MiniLM sentence-embedding relevance branch fed to every trait, with mismatch training (leaks into language traits). Includes a fix for a per-epoch memory leak in training. |
| `code/07_gate_fix.py` | Fix variant 3 (final): relevance gate on Content and Prompt Adherence only, with mismatch training. |
| `code/08_commit_run_gate.py` | Single cell that sets everything up and runs variant 3 on all 8 prompts; run it with Kaggle's "Save & Run All (Commit)" so it is not stopped by the interactive-session idle timeout. |
| `code/09_stage2_baseline.py` | Single commit-run cell: retrains ProTACT on all 8 prompts, compares weights with their initialisation, and runs the input ablations on every prompt. |
| `code/10_stage3_gate.py` | Single commit-run cell: relevance gate v4 (full prompt text, passage summaries for prompts 3-6) on all 8 prompts, plus the further tests (paraphrases, same-passage questions, unseen prompts, splicing, gate diagnostics). |
| `code/12_stage4_fill_and_ratio.py` | Single commit-run cell: the further tests for prompts 7 and 8, and the gate on prompt 1 with a mismatch share of 25%. |
| (run 13) | Prompt 2 with a mismatch share of 25%: script 12 with `JOBS = [('gatev4r25', '2', '0.25')]`. |
| `code/16_kaggle_seed.py` | Single commit-run cell for one seed (set `SEED`): ProTACT baseline on 8 prompts (two GPUs in parallel), then the gate with a 25% mismatch share and the further tests. About 8.5 hours on Kaggle 2x T4. The Colab variant used for part of seed 22 differs only in paths (Google Drive output), one GPU and the prompt lists. |
| `code/18_multiseed_stats.py` | Five-seed means, SDs, paired bootstrap intervals and Stage 4 means (reads `results/multiseed_stage4_raw.txt`). |
| `results/` | All numbers reported in the paper. Five-seed results: `multiseed_5seeds.md` (per-run numbers, statistics and Stage 4 means) and `multiseed_stage4_raw.txt`; single-seed analyses: `stage2_baseline.md`, `stage3_gate_v4.md`, `epcts_pr_v3.md`. |

## How to run

Run in a Kaggle notebook with 2x T4 GPU and Internet on, as notebook cells in the order 01 → 02 → 03. Each prompt takes about 41 minutes. For the fix, run `08_commit_run_gate.py` as the only cell of a notebook with "Save & Run All (Commit)"; it trains one prompt at a time (about 50 minutes each, about 6.6 hours in total), because two parallel jobs exceed Kaggle's 31 GB RAM. `09_stage2_baseline.py` (about 3 hours on two GPUs) and `10_stage3_gate.py` (about 6.5 hours) are run the same way. The ablation scripts and `epcts_pr.py` (which also reports per-essay ranking metrics with and without the alpha term) can be run afterwards:

```
CUDA_VISIBLE_DEVICES=0 python epcts_pr.py --per_prompt 150 --out results/epcts_pr_v3
```

## Data

ASAP/ASAP++ in the preprocessed form shipped with the ProTACT repository. The essays come from the Kaggle ASAP competition and are not redistributed here.

## Acknowledgements

Built on the released code of ProTACT (Do et al., Findings of ACL 2023). EPCTS is by Xu et al. (Neurocomputing 621, 2025).
