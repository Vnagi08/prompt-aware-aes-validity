# 22: essay-level paired bootstrap of the relevance gate vs the ungated baseline, from saved per-essay predictions.
# Input: the per-essay prediction files written by 21_gate_bootstrap.py (boot/base_p{P}_s{S}.json, boot/gate_p{P}_s{S}.json),
# either as separate files anywhere under /kaggle/input or bundled in one JSON {file name: file content}.
# QWK is ProTACT's own metrics.metrics.kappa (quadratic), the function behind every QWK in the paper: it rounds the
# scores and uses every integer between the lowest and the highest score present, so a resample that lacks a middle
# score gets the same weights as the full sample. No model inference; a few minutes on CPU.
import os, re, glob, json, subprocess, sys, time
import numpy as np

REPO = '/kaggle/working/ProTACT'
if not os.path.exists(REPO):
    subprocess.run('git clone --depth 1 https://github.com/doheejin/ProTACT.git ' + REPO, shell=True, check=True)
p = REPO + '/metrics/metrics.py'
s = open(p).read()
s = re.sub(r'SCORERS\[([^\]]+)\]', r'get_scorer(\1)', s).replace('SCORERS', 'get_scorer')   # removed from scikit-learn
open(p, 'w').write(s)
sys.path.insert(0, REPO)
from metrics.metrics import kappa


def qwk(true, pred):
    return kappa(true, pred, 'quadratic')


# ---- load predictions ----
P = {}
for f in glob.glob('/kaggle/input/**/*.json', recursive=True):
    name = os.path.basename(f)
    if re.fullmatch(r'(base|gate)_p\d_s\d+\.json', name):
        P[name] = json.load(open(f))
    else:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, dict):
            for k, v in d.items():
                if re.fullmatch(r'(base|gate)_p\d_s\d+\.json', k):
                    P[k] = v if isinstance(v, dict) else json.loads(v)
print('prediction files:', len(P), sorted(P)[:6], '...', flush=True)

SEEDS = ['32', '42', '52']
PROMPTS = [str(i) for i in range(1, 9)]
B = 1000
rng = np.random.RandomState(0)
OUT = '/kaggle/working/boot_results'
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

rep = {'B': B, 'rng_seed': 0, 'qwk': 'ProTACT metrics.metrics.kappa, quadratic', 'seeds': SEEDS, 'prompts': {}}
reps_by_prompt = {}
for p in PROMPTS:
    seeds = [s for s in SEEDS if 'base_p%s_s%s.json' % (p, s) in P and 'gate_p%s_s%s.json' % (p, s) in P]
    if not seeds:
        print('prompt', p, ': no predictions', flush=True)
        continue
    D = {s: (P['base_p%s_s%s.json' % (p, s)], P['gate_p%s_s%s.json' % (p, s)]) for s in seeds}
    traits = list(next(iter(D.values()))[0]['pred'].keys())
    n = len(next(iter(D.values()))[0]['true'][traits[0]])
    for s, (b, g) in D.items():   # same essays, same order, same gold scores in every file
        for t in traits:
            assert len(b['true'][t]) == n and b['true'][t] == g['true'][t], (p, s, t)
    arr = {s: {m: {t: (np.array(d['true'][t], float), np.array(d['pred'][t], float)) for t in traits}
               for m, d in zip(['base', 'gate'], D[s])} for s in seeds}

    def scores(idx):
        # per seed and model: QWK per trait on the essays idx
        return {s: {m: {t: qwk(arr[s][m][t][0][idx], arr[s][m][t][1][idx]) for t in traits} for m in arr[s]} for s in seeds}

    def summary(sc):
        out = {}
        for key in ['avg', 'content', 'prompt_adherence']:
            if key != 'avg' and key not in traits:
                continue
            diffs = []
            for s in seeds:
                if key == 'avg':
                    diffs.append(np.mean(list(sc[s]['gate'].values())) - np.mean(list(sc[s]['base'].values())))
                else:
                    diffs.append(sc[s]['gate'][key] - sc[s]['base'][key])
            out[key] = float(np.mean(diffs))
        return out

    full = np.arange(n)
    sc = scores(full)
    point = {s: {'base_avg': float(np.mean(list(sc[s]['base'].values()))), 'gate_avg': float(np.mean(list(sc[s]['gate'].values()))),
                 'base_trait': {t: float(v) for t, v in sc[s]['base'].items()},
                 'gate_trait': {t: float(v) for t, v in sc[s]['gate'].items()}} for s in seeds}
    for s in seeds:
        print('point p%s s%s: base avg %.3f, gate avg %.3f, base content %.3f, gate content %.3f'
              % (p, s, point[s]['base_avg'], point[s]['gate_avg'], point[s]['base_trait']['content'], point[s]['gate_trait']['content']), flush=True)
    boots = []
    for _ in range(B):
        idx = rng.randint(0, n, n)      # the same resampled essays for both models and all seeds
        boots.append(summary(scores(idx)))
    keys = list(boots[0].keys())
    reps_by_prompt[p] = {k: np.array([b[k] for b in boots]) for k in keys}
    rep['prompts'][p] = {'seeds': seeds, 'n_essays': n, 'traits': traits, 'point_per_seed': point,
                         'mean_diff_over_seeds': summary(sc),
                         'ci95': {k: [float(np.percentile(reps_by_prompt[p][k], 2.5)), float(np.percentile(reps_by_prompt[p][k], 97.5))] for k in keys}}
    print('prompt %s (%d essays, seeds %s): %s  [%.0f s]' % (p, n, ','.join(seeds),
          {k: (round(rep['prompts'][p]['mean_diff_over_seeds'][k], 4), [round(x, 4) for x in rep['prompts'][p]['ci95'][k]]) for k in keys},
          time.time() - T0), flush=True)

# mean over prompts: each replicate averages the per-prompt replicates (prompts are resampled independently)
for k in ['avg', 'content', 'prompt_adherence']:
    ps = [p for p in reps_by_prompt if k in reps_by_prompt[p]]
    if not ps:
        continue
    m = np.mean(np.vstack([reps_by_prompt[p][k] for p in ps]), axis=0)
    rep['mean_over_prompts_' + k] = {'prompts': ps,
                                     'point': float(np.mean([rep['prompts'][p]['mean_diff_over_seeds'][k] for p in ps])),
                                     'ci95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
    print('mean over prompts', k, rep['mean_over_prompts_' + k], flush=True)
json.dump(rep, open(OUT + '/bootstrap_gate_vs_base.json', 'w'), indent=1)
print('ALL DONE in %.1f min' % ((time.time() - T0) / 60))
