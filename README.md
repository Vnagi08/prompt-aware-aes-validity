# Do Prompt-Aware Essay Trait Scorers Really Read the Prompt?

Code and results for the paper *"Do Prompt-Aware Essay Trait Scorers Really Read the Prompt? A Trait-Level Validity Test and Fix"* by Vishal Nagi and Ritu Sibal, Department of Computer Science, Netaji Subhas University of Technology (NSUT), New Delhi (vishal.nagi.pg25@nsut.ac.in, rsb@nsut.ac.in).

## The test

Each test essay is paired with a wrong prompt (each of the other 7 ASAP prompts), and the change in every trait score is measured. A prompt-aware scorer should lower the prompt-dependent traits (Content, Prompt Adherence) and leave the language traits (Word Choice, Sentence Fluency, Conventions, Language) unchanged.

**Pass criterion.** On a target prompt, averaged over the 7 wrong prompts, each prompt-dependent trait must fall by at least 0.25 score points, with at least 50% of essays scored lower after rounding to the score scale. Each language trait must change by at most 0.05 points.

## Main findings

1. **ProTACT ignores the prompt.** In the released code the essay is pooled to a single vector before the prompt attention, so the attention has one key and its weight is always 1. Replacing the prompt with any of the other 7 ASAP prompts changes no trait score on any of the 8 prompts, including Prompt Adherence (56 wrong-prompt conditions, maximum change 0.0). The published equations of GAPS describe the same single-key pattern.
2. **Scores come from hand-crafted features.** We compared each trained model with its initialisation on all 8 prompts: 82 of 176 weight tensors are bit-identical, because they never receive a gradient. These include the whole prompt encoder. Removing the prompt or shuffling the essay POS sequences leaves the average QWK unchanged. Shuffling both hand-crafted feature sets drops it to between -0.021 and 0.024.
3. **The EPCTS relevance feature has a length-dependent term that ignores the prompt.** For a fixed essay the alpha term is the same for every candidate prompt, so it cannot change which prompt the essay ranks highest. It does add a length-dependent offset, which makes the feature less comparable across essays. Removing it raises the pooled own-prompt vs wrong-prompt AUC from 0.907 [0.902, 0.918] to 0.950 [0.945, 0.958]. A plain whole-essay similarity does better still (0.964). A short essay with a wrong prompt gets a higher relevance score than a long essay with its own prompt in 23% of pairs, against 9.5% without alpha.
4. **What a fix requires.**
   - **Multi-key attention alone is not enough.** We made ProTACT's attention multi-key, with one seed, prompt 1 and only 2 epochs of training. The essay is represented only by POS tags, so a wrong prompt still changed nothing.
   - **Relevance fed to every trait leaks.** A sentence-embedding relevance signal fed to all traits makes Content respond. It also leaks into the language traits through ProTACT's trait-relation attention.
   - **The relevance gate.** It acts on Content and Prompt Adherence only and is trained with deliberately mismatched prompts (copies of a quarter of the training essays). These traits respond to a wrong prompt on all 8 prompts in all 40 runs (5 seeds x 8 prompts). The gate meets the pass criterion on 6 prompts. It fails on the source-dependent prompts 3 and 4, where Prompt Adherence is lowered for only 42% and 45% of essays.
   - **Gate results over five seeds.** Under a wrong prompt, Content falls for 67% of essays (mean -0.83 points) and Prompt Adherence for 50% (mean -0.51). Language traits do not change.
   - **Cost.** Average QWK over all traits is 0.524 for the gate, against 0.540 for ProTACT. The paired difference is -0.016, with a bootstrap interval of [-0.018, -0.013] over the five seeds. Mean Content QWK is 0.508 against 0.543, and mean Prompt Adherence QWK is 0.491 against 0.558.
5. **What the gate does and does not detect (all 8 prompts, 5 seeds).**
   - **Unseen prompts and paraphrases.** The gate reacts to prompts it never saw in training about as strongly as to the seen wrong prompts. Paraphrasing the correct prompt changes little.
   - **Topic, not question.** For the source-dependent prompts, a different question about the same reading passage barely changes Content (4-9% of essays lower). The gate detects the topic, not the question asked.
   - **Partly off-topic essays.** These are lowered only when most of the essay is off-topic.
   - **Narrative prompt 7.** The gate partly closes for on-topic essays (mean gate 0.71), which explains the drop in Content QWK there.
   - **Passage summaries.** Without our summaries for prompts 3-6, the gate still works but responds less to wrong prompts (`results/kaggle_nosum_seed12.md`).
6. **The mismatch share matters.** With seed 12, using mismatched copies of 25% instead of 50% of the essays raised Content QWK on prompt 1 from 0.468 to 0.634 (ProTACT: 0.661) and strengthened the wrong-prompt response. On prompt 2 it helped only slightly (0.476 to 0.489). The five-seed runs use 25%.
7. **A fixed rule without training is a strong baseline (seed 42).**
   - **The rule.** Content and Prompt Adherence are multiplied by a threshold on the cosine similarity between the mean essay embedding and the mean prompt embedding.
   - **Pass criterion.** The rule passes on 7 of the 8 prompts. It fails only on prompt 7, where Content is lowered for 29% of essays.
   - **Accuracy.** On prompts 2-6 it matches or beats the learned gate in average QWK, and under a wrong prompt it lowers Content for 86-99% of essays. On prompt 8 it passes the test, but Content QWK falls to 0.08. On the narrative prompts, on-topic stories share little vocabulary with the prompt.
   - **Shared limit.** Like the gate, the rule does not distinguish a different question about the same passage (`results/rule_baseline_seed42.md`).
   - **More seeds.** Runs for seeds 32 and 52 are in progress.

## Repository layout

| Path | Contents |
|---|---|
| `code/01_setup_protact.py` | Clones [ProTACT](https://github.com/doheejin/ProTACT), installs tf_keras 2.21, applies three compatibility fixes (sklearn `get_scorer`, NumPy `np.cast` shim, NLTK data) and sets 15 epochs. Training otherwise uses ProTACT's defaults (GloVe 50d, 2 attention heads, RMSprop, batch size 10). |
| `code/02_make_wrong_prompt_script.py` | Downloads GloVe 6B 50d if needed and builds `train_pilot.py`, which trains ProTACT and runs the wrong-prompt test. |
| `code/03_run_all_prompts.py` | Trains all 8 target prompts (seed 12) on two GPUs and runs the wrong-prompt test after each. |
| `code/04_diag_ablations_v1.py`, `code/04_diag_ablations_v2.py` | Input ablations on prompt 1: prompt swap, empty prompt, and shuffles of the essay text and feature sets. |
| `code/epcts_pr.py` | Re-implementation of the EPCTS prompt-relevance feature and its length and wrong-prompt checks. |
| `code/05_fix_mismatch.py` | Fix variant 1: multi-key prompt attention over the essay's sentence states, with mismatch training (2 epochs; fails). |
| `code/06_semantic_fix.py` | Fix variant 2: MiniLM sentence-embedding relevance branch fed to every trait, with mismatch training (leaks into the language traits). Also fixes a per-epoch memory leak in training. |
| `code/07_gate_fix.py` | Fix variant 3 (final): relevance gate on Content and Prompt Adherence only, with mismatch training. |
| `code/08_commit_run_gate.py` | Single cell that sets everything up and runs variant 3 on all 8 prompts. Run it with Kaggle's "Save & Run All (Commit)" so the interactive-session idle timeout does not stop it. |
| `code/09_stage2_baseline.py` | Single commit-run cell: retrains ProTACT on all 8 prompts, compares the weights with their initialisation, and runs the input ablations on every prompt. |
| `code/10_stage3_gate.py` | Single commit-run cell: relevance gate v4 (full prompt text, passage summaries for prompts 3-6) on all 8 prompts, plus the further tests (paraphrases, same-passage questions, unseen prompts, splicing, gate diagnostics). |
| `code/12_stage4_fill_and_ratio.py` | Single commit-run cell: the further tests for prompts 7 and 8, and the gate on prompt 1 with a mismatch share of 25%. |
| (run 13) | Prompt 2 with a mismatch share of 25%: script 12 with `JOBS = [('gatev4r25', '2', '0.25')]`. |
| `code/16_kaggle_seed.py` | Single commit-run cell for one seed (set `SEED`): the ProTACT baseline on 8 prompts (two GPUs in parallel), then the gate with a 25% mismatch share and the further tests. About 8.5 hours on Kaggle 2x T4. The Colab variant used for part of seed 22 differs only in paths (Google Drive output), one GPU and the prompt lists. |
| `code/18_multiseed_stats.py` | Five-seed means, SDs, paired bootstrap intervals and Stage 4 means (reads `results/multiseed_stage4_raw.txt`). |
| `code/19_rule_baseline.py` | Fixed similarity rule (no training) applied to the trained ungated ProTACT weights (Kaggle CPU): threshold and ramp rules on the mean-embedding cosine, with the wrong-prompt test on all 8 prompts. |
| `code/20_rule_stage4.py` | The same rule plus the further tests (paraphrases, same-passage/nearby, unseen prompts, splice). It runs on every seed whose weights are attached (42, 32, 52). |
| `results/` | All numbers reported in the paper. Five seeds: `multiseed_5seeds.md` (per-run numbers, statistics and Stage 4 means) and `multiseed_stage4_raw.txt`. Single-seed analyses: `stage2_baseline.md`, `stage3_gate_v4.md`, `epcts_pr_v3.md`, `epcts_reanalysis_2026-11-24.md`. Rule baseline: `rule_baseline_seed42.md`. Gate without passage summaries: `kaggle_nosum_seed12.md`. |

## How to run

1. **Wrong-prompt test.** In a Kaggle notebook with 2x T4 GPU and Internet on, run 01 → 02 → 03 as notebook cells. Each prompt takes about 41 minutes.
2. **The fix.** Run `08_commit_run_gate.py` as the only cell of a notebook with "Save & Run All (Commit)". It trains one prompt at a time, because two parallel jobs exceed Kaggle's 31 GB RAM. Each prompt takes about 50 minutes, about 6.6 hours in total.
3. **Other commit-run cells.** Run `09_stage2_baseline.py` (about 3 hours on two GPUs), `10_stage3_gate.py` (about 6.5 hours) and `16_kaggle_seed.py` the same way.
4. **Fixed-rule scripts.** Scripts 19 and 20 need only a CPU. Attach the GloVe file and the trained weights as notebook inputs.
5. **Ablations and the EPCTS check.** The ablation scripts and `epcts_pr.py` can be run afterwards. `epcts_pr.py` also reports per-essay ranking metrics with and without the alpha term:

```
CUDA_VISIBLE_DEVICES=0 python epcts_pr.py --per_prompt 150 --out results/epcts_pr_v3
```

## Data

ASAP/ASAP++, in the preprocessed form shipped with the ProTACT repository. The essays come from the Kaggle ASAP competition and are not redistributed here.

## Acknowledgements

Built on the released code of ProTACT (Do et al., Findings of ACL 2023). EPCTS is by Xu et al. (Neurocomputing 621, 2025).
