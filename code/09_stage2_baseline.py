import os, re, subprocess, glob, sys, time
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-1500:])
REPO = '/kaggle/working/ProTACT'
if not os.path.exists(REPO):
    sh('git clone --depth 1 https://github.com/doheejin/ProTACT.git ' + REPO)
os.chdir(REPO)
sh('pip install -q "tf_keras==2.21.*" 2>&1 | tail -3; python -c "import tf_keras; print(\'tf_keras\', tf_keras.__version__)"')
sh('python -c "import nltk; [nltk.download(x, quiet=True) for x in [\'averaged_perceptron_tagger_eng\', \'averaged_perceptron_tagger\', \'punkt\', \'punkt_tab\']]; print(\'nltk ok\')"')
p = 'metrics/metrics.py'; s = open(p).read()
s = re.sub(r'SCORERS\[([^\]]+)\]', r'get_scorer(\1)', s).replace('SCORERS', 'get_scorer')
open(p, 'w').write(s)
p = 'utils/general_utils.py'; s = open(p).read()
if '_np_cast' not in s:
    s = s.replace('np.cast[', '_np_cast[')
    s = s.replace('import numpy as np', "import numpy as np\n_np_cast = type('_C', (), {'__getitem__': lambda self, t: (lambda x: np.asarray(x, dtype=t))})()", 1)
    open(p, 'w').write(s)
p = 'configs/configs.py'; s = open(p).read()
s = re.sub(r'EPOCHS\s*=\s*\d+', 'EPOCHS = 15', s)
open(p, 'w').write(s)
sh('grep -n -E "EPOCHS|EMBEDDING|DATA_PATH" configs/configs.py; grep -n -E "get_scorer|SCORERS" metrics/metrics.py; grep -n "_np_cast" utils/general_utils.py | head -3')
os.makedirs('embeddings', exist_ok=True)
g = 'embeddings/glove.6B.50d.txt'
cands = glob.glob('/kaggle/input/**/glove.6B.50d.txt', recursive=True)
if not os.path.exists(g) and cands:
    os.symlink(cands[0], g)
print('glove present:', os.path.exists(g), cands[:2])
print('embedding refs:'); sh('grep -rn -i "glove\\|EMBEDDING_PATH\\|load_word_embedding" --include=*.py . | head -12')
print(sorted(os.listdir('data')), sorted(os.listdir('.')))
sh('nvidia-smi -L')

import os, glob, subprocess
os.chdir('/kaggle/working/ProTACT')
G = 'embeddings/glove.6B.50d.txt'
if not os.path.exists(G):
    try:
        import kagglehub
        d = kagglehub.dataset_download('watts2/glove6b50dtxt')
        f = glob.glob(d + '/**/glove.6B.50d.txt', recursive=True)[0]
        os.symlink(f, G)
    except Exception as e:
        print('kagglehub failed:', e, '- downloading from Stanford')
        subprocess.run('cd embeddings && wget -q https://nlp.stanford.edu/data/glove.6B.zip && unzip -o glove.6B.zip glove.6B.50d.txt && rm glove.6B.zip', shell=True)
print('glove ok:', os.path.exists(G), os.path.getsize(G) // 10**6 if os.path.exists(G) else 0, 'MB')
assert os.path.exists(G)
import re, subprocess, os, time
os.chdir('/kaggle/working/ProTACT')
src = open('train_ProTACT.py').read()
src = src.replace('from evaluators.multitask_evaluator_all_attributes import Evaluator as AllAttEvaluator',
                  'from evaluators.multitask_evaluator_all_attributes import Evaluator as AllAttEvaluator\nfrom utils.general_utils import separate_and_rescale_attributes_for_scoring\nfrom metrics.metrics import kappa')
src = re.sub(r'^(\s*)evaluator\.evaluate\(model, ii \+ 1\)\s*$',
             lambda m: m.group(1) + 'evaluator.evaluate(model, ii + 1)\n' + m.group(1) + 'if evaluator.best_dev_epoch == ii + 1:\n' + m.group(1) + '    best_weights = model.get_weights()',
             src, count=1, flags=re.M)
src = re.sub(r'^(\s*)evaluator\.print_final_info\(\)\s*$',
             lambda m: m.group(1) + 'evaluator.print_final_info()\n' + m.group(1) + 'run_mismatch(locals())',
             src, count=1, flags=re.M)
src = src.replace("custom_hist.init()", "custom_hist.init()\n    best_weights = None", 1)
block = "\n\ndef run_mismatch(L, n_note='prompt text swapped; LDA topic feature not recomputed'):\n    import json\n    model = L['model']\n    ev = L['evaluator']\n    tp = L['test_prompt_id']\n    seed = L['seed']\n    if L.get('best_weights') is not None:\n        model.set_weights(L['best_weights'])\n    os.makedirs('results', exist_ok=True)\n    try:\n        model.save_weights('results/protact_p%d_s%d.weights.h5' % (tp, seed))\n    except Exception as e:\n        print('could not save weights:', e)\n    test_feats = L['test_features_list']\n    test_ids = L['test_data']['prompt_ids']\n    Y_test = L['Y_test']\n    true_org = separate_and_rescale_attributes_for_scoring(Y_test, test_ids)\n    base_pred = model.predict(test_feats, batch_size=32)\n    base_dict = separate_and_rescale_attributes_for_scoring(base_pred, test_ids)\n    n = len(test_ids)\n    src = []\n    for name in ['train', 'dev']:\n        d = L[name + '_data']\n        src.append((d['prompt_ids'], L['X_' + name + '_prompt'], L['X_' + name + '_prompt_pos']))\n    out = {'test_prompt': tp, 'seed': seed, 'best_dev_epoch': ev.best_dev_epoch,\n           'best_test_qwk': {k: float(v) for k, v in ev.best_test_kappa_set.items()},\n           'best_test_avg_qwk': float(ev.best_test_kappa_mean), 'note': n_note,\n           'base_qwk_reloaded': {k: float(Evaluator_kappa(base_dict[k], true_org[k])) for k in base_dict},\n           'swaps': {}}\n    for k in range(1, 9):\n        if k == tp:\n            continue\n        row = None\n        for ids, xp, xpp in src:\n            for i, pid in enumerate(ids):\n                if str(pid) == str(k):\n                    row = (xp[i], xpp[i])\n                    break\n            if row is not None:\n                break\n        if row is None:\n            print('prompt', k, 'not found in train/dev; skipping')\n            continue\n        feats = list(test_feats)\n        feats[1] = np.repeat(row[0][None, :], n, axis=0)\n        feats[2] = np.repeat(row[1][None, :], n, axis=0)\n        pred = model.predict(feats, batch_size=32)\n        pdict = separate_and_rescale_attributes_for_scoring(pred, test_ids)\n        res = {}\n        for t in pdict:\n            b = np.array(base_dict[t], dtype=float)\n            s = np.array(pdict[t], dtype=float)\n            res[t] = {'mean_true_prompt': float(b.mean()), 'mean_wrong_prompt': float(s.mean()),\n                      'mean_change': float((s - b).mean()),\n                      'pct_essays_lower': float((s < b).mean() * 100),\n                      'qwk_wrong_prompt': float(Evaluator_kappa(pdict[t], true_org[t]))}\n        out['swaps'][str(k)] = res\n    with open('results/pilot_p%d_s%d.json' % (tp, seed), 'w') as f:\n        json.dump(out, f, indent=1)\n    print('\\n===== PILOT MISMATCH RESULTS =====')\n    print('best dev epoch', out['best_dev_epoch'], '| best test avg QWK', round(out['best_test_avg_qwk'], 3))\n    traits = list(base_dict.keys())\n    print('trait'.ljust(18), 'true-prompt QWK', 'avg change with wrong prompt (score points)', 'avg QWK wrong prompt')\n    for t in traits:\n        ch = np.mean([out['swaps'][k][t]['mean_change'] for k in out['swaps']])\n        qw = np.mean([out['swaps'][k][t]['qwk_wrong_prompt'] for k in out['swaps']])\n        print(t.ljust(18), str(round(out['base_qwk_reloaded'][t], 3)).ljust(15), str(round(ch, 3)).ljust(44), round(qw, 3))\n    print('saved results/pilot_p%d_s%d.json' % (tp, seed))\n\n\ndef Evaluator_kappa(pred, original):\n    return kappa(original, pred, 'quadratic')\n"
src = src.replace("if __name__ == '__main__':", block + "\n\nif __name__ == '__main__':", 1)
open('train_pilot.py', 'w').write(src)
r = subprocess.run('python -c "import ast; ast.parse(open(\'train_pilot.py\').read()); print(\'pilot script OK\')"; grep -n -E "best_weights|run_mismatch\\(locals" train_pilot.py', shell=True, capture_output=True, text=True)

# ===================== Stage 2: baseline with saved weights, ablations on all prompts, weight check =====================
# Adds to train_pilot.py: (1) a copy of the initial weights, (2) run_extra(), called after run_mismatch(), which
# reports which weight tensors never changed during training and runs the input ablations on the target prompt.
import os, re, subprocess, time, json, glob, shutil
os.chdir('/kaggle/working/ProTACT')
s = open('train_pilot.py').read()
s, n1 = re.subn(r'^(\s*)(evaluator = AllAttEvaluator\()', lambda m: m.group(1) + 'init_weights = [w.numpy().copy() for w in model.weights]\n' + m.group(1) + m.group(2), s, count=1, flags=re.M)
EXTRA = '''

def run_extra(L):
    import json
    model, tp, seed = L['model'], L['test_prompt_id'], L['seed']
    if L.get('best_weights') is not None:
        model.set_weights(L['best_weights'])
    # (1) weight check: which tensors are bit-identical to their initial values after training
    init = L['init_weights']
    unchanged, changed = [], []
    for w, w0 in zip(model.weights, init):
        d = float(np.max(np.abs(w.numpy() - w0)))
        (unchanged if d == 0.0 else changed).append([w.name, list(w.shape), d])
    # (2) input ablations on the test set (changes in the model's normalised output)
    feats = list(L['test_features_list'])
    ids = L['test_data']['prompt_ids']
    true_org = separate_and_rescale_attributes_for_scoring(L['Y_test'], ids)
    def avg_qwk(pred):
        d = separate_and_rescale_attributes_for_scoring(pred, ids)
        q = {k: float(Evaluator_kappa(d[k], true_org[k])) for k in d}
        return float(np.mean(list(q.values()))), q
    base = model.predict(feats, batch_size=32)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(ids))
    def variant(idx_list, mode):
        f = list(feats)
        for i in idx_list:
            f[i] = np.zeros_like(f[i]) if mode == 'zero' else f[i][perm]
        return f
    conds = {'prompt_removed_zeros': variant([1, 2], 'zero'),
             'essay_pos_shuffled': variant([0], 'shuffle'),
             'linguistic_shuffled': variant([3], 'shuffle'),
             'readability_shuffled': variant([4], 'shuffle'),
             'both_feature_sets_shuffled': variant([3, 4], 'shuffle')}
    q0, q0t = avg_qwk(base)
    abl = {'none': {'avg_qwk': q0, 'qwk': q0t}}
    for name, f in conds.items():
        p = model.predict(f, batch_size=32)
        q, qt = avg_qwk(p)
        diff = np.abs(p - base)
        abl[name] = {'max_abs_change': float(diff.max()), 'mean_abs_change': float(diff.mean()), 'avg_qwk': q, 'qwk': qt}
    out = {'test_prompt': tp, 'seed': seed, 'n_weights': len(model.weights),
           'unchanged_weights': unchanged, 'changed_weight_names': [c[0] for c in changed], 'ablations': abl}
    json.dump(out, open('results/extra_p%d_s%d.json' % (tp, seed), 'w'), indent=1)
    print('\\n===== EXTRA =====')
    print('unchanged weight tensors (%d of %d):' % (len(unchanged), len(model.weights)), [u[0] for u in unchanged])
    for k, v in abl.items():
        print(k.ljust(28), {kk: (round(vv, 6) if isinstance(vv, float) else '') for kk, vv in v.items() if kk != 'qwk'})
'''
s = s.replace("\n\nif __name__ == '__main__':", EXTRA + "\n\nif __name__ == '__main__':", 1)
s, n2 = re.subn(r'^(\s*)run_mismatch\(locals\(\)\)\s*$', lambda m: m.group(1) + 'run_mismatch(locals())\n' + m.group(1) + 'run_extra(locals())', s, count=1, flags=re.M)
open('train_pilot.py', 'w').write(s)
print('patches:', n1, n2)
assert n1 == 1 and n2 == 1
subprocess.run('python -c "import ast; ast.parse(open(\'train_pilot.py\').read()); print(\'train_pilot OK\')"', shell=True)

SEED = os.environ.get('STAGE2_SEED', '12')
ARGS = ['--model_name', 'ProTACT', '--seed', SEED, '--num_heads', '2', '--features_path', 'data/LDA/hand_crafted_final_']
os.makedirs('logs', exist_ok=True); os.makedirs('results', exist_ok=True)
OUT = '/kaggle/working/stage2_results'
os.makedirs(OUT, exist_ok=True)

def run_queue(gpu, prompts):
    procs = []
    return subprocess.Popen(['python', '-c', '''
import subprocess, os, sys, time
gpu, prompts, args = sys.argv[1], sys.argv[2].split(','), sys.argv[3:]
env = dict(os.environ, TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES=gpu)
for p in prompts:
    t = time.time()
    with open('logs/base_p%s.log' % p, 'w') as f:
        subprocess.run(['python', '-u', 'train_pilot.py', '--test_prompt_id', p] + args, stdout=f, stderr=subprocess.STDOUT, env=env)
    print('prompt', p, 'done in %.0f min' % ((time.time() - t) / 60), flush=True)
''', gpu, prompts] + ARGS)

t0 = time.time()
# Two queues, one per T4; the plain baseline fits in RAM twice (it ran this way before).
NG = len([l for l in subprocess.run('nvidia-smi -L', shell=True, capture_output=True, text=True).stdout.splitlines() if l.startswith('GPU')])
print('GPUs:', NG)
qs = [run_queue('0', '1,3,5,7'), run_queue('1', '2,4,6,8')] if NG >= 2 else [run_queue('0', '1,3,5,7,2,4,6,8')]
for q in qs:
    q.wait()
print('all prompts done in %.0f min' % ((time.time() - t0) / 60))
for f in glob.glob('results/pilot_p*_s%s.json' % SEED) + glob.glob('results/extra_p*_s%s.json' % SEED) + glob.glob('results/protact_p*_s%s.weights.h5' % SEED) + glob.glob('logs/base_p*.log'):
    shutil.copy(f, OUT)
# Summary table
for p in range(1, 9):
    try:
        e = json.load(open('results/extra_p%d_s%s.json' % (p, SEED)))
        m = json.load(open('results/pilot_p%d_s%s.json' % (p, SEED)))
        maxswap = max(abs(v['mean_change']) for k in m['swaps'] for v in m['swaps'][k].values())
        a = e['ablations']
        print('P%d avgQWK %.3f | swap max|mean change| %.6f | unchanged tensors %d | ' % (p, m['best_test_avg_qwk'], maxswap, len(e['unchanged_weights'])) +
              ' '.join('%s %.3f' % (k[:12], a[k]['avg_qwk']) for k in a))
    except Exception as ex:
        print('P%d missing:' % p, ex)
print('ALL DONE')
