# Do Prompt-Aware Essay Trait Scorers Really Read the Prompt?

Code and results for the paper *"Do Prompt-Aware Essay Trait Scorers Really Read the Prompt? A Trait-Level Validity Test and Fix"* (Piyush Parihar and Satish Kumar Singh, Department of Information Technology, Netaji Subhas University of Technology, New Delhi).

## Main findings

1. **ProTACT ignores the prompt.** The essay is pooled to a single vector before the prompt attention, so the attention has one key and its weight is always 1. Replacing the prompt with any of the other 7 ASAP prompts changes no trait score, including Prompt Adherence, on all 8 prompts (56 wrong-prompt conditions, maximum change 0.0).
2. **Scores come from hand-crafted features.** Shuffling the essay text between essays barely changes predictions. Shuffling the linguistic and readability features drops the average QWK on prompt 1 from 0.658 to -0.021.
3. **The EPCTS relevance feature has a length-dependent, prompt-independent term.** Removing its reward term alpha raises its ability to tell an essay's own prompt from a wrong one (AUC 0.907 to 0.950). A plain whole-essay similarity does better still.

## Repository layout

| Path | Contents |
|---|---|
| `code/01_setup_protact.py` | Clones [ProTACT](https://github.com/doheejin/ProTACT), installs tf_keras 2.21, applies three compatibility fixes (sklearn `get_scorer`, NumPy `np.cast` shim, NLTK data) and sets 15 epochs. |
| `code/02_make_wrong_prompt_script.py` | Downloads GloVe 6B 50d if needed and builds `train_pilot.py`: ProTACT training plus the wrong-prompt test. |
| `code/03_run_all_prompts.py` | Trains all 8 target prompts (seed 12) on two GPUs and runs the wrong-prompt test after each. |
| `code/04_diag_ablations_v1.py`, `code/04_diag_ablations_v2.py` | Input ablations on prompt 1: prompt swap, empty prompt, and shuffles of essay text and feature sets. |
| `code/epcts_pr.py` | Re-implementation of the EPCTS prompt-relevance feature and its length and wrong-prompt checks. |
| `results/` | All numbers reported in the paper. |

## How to run

Run in a Kaggle notebook with 2x T4 GPU and Internet on, as notebook cells in the order 01 → 02 → 03. Each prompt takes about 41 minutes. The ablation scripts and `epcts_pr.py` can be run afterwards:

```
CUDA_VISIBLE_DEVICES=0 python epcts_pr.py --per_prompt 150 --out results/epcts_pr_v2
```

## Data

ASAP/ASAP++ in the preprocessed form shipped with the ProTACT repository. The essays come from the Kaggle ASAP competition and are not redistributed here.

## Acknowledgements

Built on the released code of ProTACT (Do et al., Findings of ACL 2023). EPCTS is by Xu et al. (Neurocomputing 621, 2025).
