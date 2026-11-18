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
print(r.stdout, r.stderr)



"""Fix v3: relevance gate on prompt-dependent traits + trait-selective mismatch training.

v2 (semantic branch fed to every trait) made Content respond to a wrong prompt, but language traits
also dropped (prompt 1: content -1.11, word choice -0.45, sentence fluency -0.46, conventions -0.58)
and Content QWK fell from 0.661 to 0.464: the relevance signal leaks into all traits through
ProTACT's trait-relation attention. v3 keeps ProTACT unchanged and multiplies only the Content and
Prompt Adherence outputs by a learned relevance gate g in (0, 1) computed from the semantic statistics.
Language traits cannot depend on the prompt by construction.

(v2 description follows.)

Why v2: fix v1 (05_fix_mismatch.py) made ProTACT's prompt attention multi-key over the essay's sentence
states. With mismatch training, a wrong prompt still changed Content by 0.000 score points on prompt 1
(2-epoch test). ProTACT represents the essay only by POS tags, which carry no meaning, so nothing can
compare it with the prompt. v2 therefore compares essay and prompt in a shared semantic space.

1. Embeddings (sem/): every essay sentence and every prompt sentence is embedded with all-MiniLM-L6-v2
   (up to 40 essay sentences, 30 prompt sentences, 384 dims).
Memory: essay embeddings are float16; prompt embeddings sit in a frozen lookup table inside the
model, so each example carries only its prompt id.
2. Model (models/ProTACT_fix.py, from the original ProTACT.py): a branch computes, inside the network,
   the cosine similarity of every essay sentence with every prompt sentence, and summarises it as
   [mean over essay sentences of best-match similarity, max of it, share of sentences with best match
   > 0.3, share > 0.5, whole-essay cosine]. A Dense(16, tanh) layer maps this to a relevance vector,
   which is concatenated to each trait's representation next to the hand-crafted features.
3. Training: same trait-selective mismatch augmentation as v1 (Content and Prompt Adherence -> lowest
   score; language traits keep their labels; Overall, Organization, Narrativity masked).
4. Evaluation: same wrong-prompt test; the prompt embeddings are swapped along with prompt words/POS.
"""
import os, re, subprocess, time, json, glob, pickle
import numpy as np
os.chdir('/kaggle/working/ProTACT')
S_MAX, P_MAX, D = 40, 30, 384

# ---- 1. embeddings ----
def build_embeddings():
    if os.path.exists('sem/prompt_sem.npy'):
        print('embeddings already built'); return
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        subprocess.run('pip install -q sentence-transformers', shell=True)
        from sentence_transformers import SentenceTransformer
    import nltk, pandas as pd
    from configs.configs import Configs
    os.makedirs('sem', exist_ok=True)
    rows = {}
    for split in ['train', 'dev', 'test']:
        for r in pickle.load(open('data/cross_prompt_attributes/1/%s.pk' % split, 'rb')):
            rows[str(r['essay_id'])] = r['content_text']
    ids = sorted(rows)
    print('essays:', len(ids))
    enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    sents, owner = [], []
    for i, e in enumerate(ids):
        ss = [s for s in nltk.sent_tokenize(rows[e]) if s.strip()][:S_MAX] or [rows[e][:500]]
        sents += ss; owner += [i] * len(ss)
    emb = enc.encode(sents, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
    E = np.zeros((len(ids), S_MAX, D), dtype=np.float16)
    pos = np.zeros(len(ids), dtype=int)
    for v, o in zip(emb, owner):
        E[o, pos[o]] = v; pos[o] += 1
    np.save('sem/essay_sem.npy', E)
    json.dump({e: i for i, e in enumerate(ids)}, open('sem/essay_index.json', 'w'))
    pp = Configs().PROMPT_PATH
    df = pd.read_csv(pp) if pp.endswith('.csv') else pd.read_excel(pp)
    idc = [c for c in df.columns if set(pd.to_numeric(df[c], errors='coerce').dropna().astype(int)) == set(range(1, 9))][0]
    txc = max([c for c in df.columns if df[c].dtype == object], key=lambda c: df[c].astype(str).str.len().mean())
    print('prompt file', pp, '| id column', idc, '| text column', txc)
    P = np.zeros((9, P_MAX, D), dtype=np.float32)
    for _, r in df.iterrows():
        k = int(r[idc]); ss = [s for s in nltk.sent_tokenize(str(r[txc])) if s.strip()][:P_MAX]
        P[k, :len(ss)] = enc.encode(ss, convert_to_numpy=True)
        print('prompt', k, len(ss), 'sentences |', str(r[txc])[:80])
    np.save('sem/prompt_sem.npy', P)
    del enc
    import torch, gc; gc.collect(); torch.cuda.empty_cache()
    # sanity check: whole-essay cosine with own vs other prompts
    pv = P.sum(1); pv /= np.linalg.norm(pv, axis=1, keepdims=True) + 1e-9
    ev = E.astype(np.float32).sum(1); ev /= np.linalg.norm(ev, axis=1, keepdims=True) + 1e-9
    own = {}
    for split in ['train', 'dev', 'test']:
        for r in pickle.load(open('data/cross_prompt_attributes/1/%s.pk' % split, 'rb')):
            own[str(r['essay_id'])] = int(r['prompt_id'])
    sims = ev @ pv.T
    acc = np.mean([np.argmax(sims[i, 1:]) + 1 == own[e] for i, e in enumerate(ids)])
    print('whole-essay cosine picks the own prompt for %.1f%% of essays' % (100 * acc))

build_embeddings()

# ---- 2. model: relevance gate on prompt-dependent traits only ----
m = open('models/ProTACT.py').read()
SEM_FN = """
SEM_S, SEM_P, SEM_D, SEM_UNITS = %d, %d, %d, 16
GATED_TRAITS = (1, 6)   # content, prompt_adherence (positions from get_score_vector_positions)


def _sem_stats(x):
    e, p = x
    e = tf.cast(e, tf.float32)
    p = tf.cast(p, tf.float32)
    em = tf.cast(tf.reduce_any(tf.not_equal(e, 0.), -1), tf.float32)
    pm = tf.cast(tf.reduce_any(tf.not_equal(p, 0.), -1), tf.float32)
    sim = tf.matmul(tf.math.l2_normalize(e, -1), tf.math.l2_normalize(p, -1), transpose_b=True)
    sim = sim - 2.0 * (1.0 - pm[:, None, :])
    best = tf.reduce_max(sim, -1)
    cnt = tf.reduce_sum(em, -1) + 1e-6
    mean_best = tf.reduce_sum(best * em, -1) / cnt
    max_best = tf.reduce_max(best * em - 2.0 * (1.0 - em), -1)
    f3 = tf.reduce_sum(tf.cast(best > 0.3, tf.float32) * em, -1) / cnt
    f5 = tf.reduce_sum(tf.cast(best > 0.5, tf.float32) * em, -1) / cnt
    ev = tf.reduce_sum(e * em[..., None], 1)
    pv = tf.reduce_sum(p * pm[..., None], 1)
    whole = tf.reduce_sum(tf.math.l2_normalize(ev, -1) * tf.math.l2_normalize(pv, -1), -1)
    return tf.stack([mean_best, max_best, f3, f5, whole], -1)

""" % (S_MAX, P_MAX, D)
m, n1 = re.subn(r'^def build_ProTACT', SEM_FN + 'def build_ProTACT', m, count=1, flags=re.M)
GATE = (r"\1essay_sem_input = layers.Input(shape=(SEM_S, SEM_D), dtype='float16', name='essay_sem_input')\n"
        r"\1prompt_sem_input = layers.Input(shape=(1,), dtype='int32', name='prompt_sem_input')\n"
        r"\1prompt_sem = layers.Embedding(9, SEM_P * SEM_D, weights=[np.load('sem/prompt_sem.npy').reshape(9, -1)], trainable=False, name='prompt_sem_table')(prompt_sem_input)\n"
        r"\1prompt_sem = layers.Reshape((SEM_P, SEM_D), name='prompt_sem_reshape')(prompt_sem)\n"
        r"\1sem_stats = layers.Lambda(_sem_stats, name='sem_stats')([essay_sem_input, prompt_sem])\n"
        r"\1rel_gate = layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(3.0), name='rel_gate')(layers.Dense(SEM_UNITS, activation='tanh', name='sem_rep')(sem_stats))\n"
        r"\1final_preds = [layers.Multiply(name='gated_%d' % gi)([gp, rel_gate]) if gi in GATED_TRAITS else gp for gi, gp in enumerate(final_preds)]\n"
        r"\1\2")
m, n2 = re.subn(r'^(\s*)(y\s*=\s*layers\.Concatenate\(\)\(\[pred for pred in final_preds\]\))', GATE, m, count=1, flags=re.M)
m, n5 = re.subn(r'readability_input\],\s*outputs\s*=\s*y', 'readability_input, essay_sem_input, prompt_sem_input], outputs=y', m)
print('model patches:', n1, n2, n5)
assert n1 == 1 and n2 == 1 and n5 == 1
open('models/ProTACT_fix.py', 'w').write(m)

# ---- 3. training script ----
s = open('train_pilot.py').read()
s = s.replace('from models.ProTACT import build_ProTACT', 'from models.ProTACT_fix import build_ProTACT', 1)
WRAP = '''
# ---- memory-leak fix: build the training dataset once instead of on every per-epoch fit() call ----
_orig_fit = keras.Model.fit
_fit_cache = {}
def _cached_fit(self, x=None, y=None, batch_size=None, epochs=1, verbose='auto', callbacks=None,
                validation_data=None, shuffle=True, **kw):
    if isinstance(x, (list, tuple)) and y is not None and 'sample_weight' not in kw:
        key = ('t', id(x), id(y), batch_size)
        if key not in _fit_cache:
            for k in [k for k in _fit_cache if k[0] == 't']:
                del _fit_cache[k]
            ds = tf.data.Dataset.from_tensor_slices((tuple(x), np.asarray(y, dtype='float32')))
            if shuffle:
                ds = ds.shuffle(len(y), reshuffle_each_iteration=True)
            _fit_cache[key] = ds.batch(batch_size or 32).prefetch(2)
        vd = validation_data
        if isinstance(vd, (list, tuple)) and len(vd) == 2 and isinstance(vd[0], (list, tuple)):
            vkey = ('v', id(vd[0]), id(vd[1]))
            if vkey not in _fit_cache:
                _fit_cache[vkey] = tf.data.Dataset.from_tensor_slices((tuple(vd[0]), np.asarray(vd[1], dtype='float32'))).batch(batch_size or 32)
            vd = _fit_cache[vkey]
        return _orig_fit(self, _fit_cache[key], epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data=vd, **kw)
    return _orig_fit(self, x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                     validation_data=validation_data, shuffle=shuffle, **kw)
keras.Model.fit = _cached_fit
'''
s, _n = re.subn(r'^from tensorflow import keras\s*$', 'from tensorflow import keras\n' + WRAP, s, count=1, flags=re.M)
assert _n == 1
s = s.replace('epochs = configs.EPOCHS', "epochs = int(os.environ.get('EPOCHS_OVERRIDE', configs.EPOCHS))", 1)
s = s.replace("'results/pilot_p%d_s%d.json'", "'results/' + os.environ.get('RUN_TAG', 'gate') + '_p%d_s%d.json'")
s = s.replace("'results/protact_p%d_s%d.weights.h5'", "'results/' + os.environ.get('RUN_TAG', 'gate') + '_p%d_s%d.weights.h5'")
s = s.replace("print('saved results/pilot_p%d_s%d.json' % (tp, seed))", "print('saved results json for prompt', tp)")
old = "        feats[2] = np.repeat(row[1][None, :], n, axis=0)\n"
assert old in s
s = s.replace(old, old + "        if len(feats) > 6:\n            feats[6] = np.full((n, 1), int(k), dtype='int32')\n", 1)
SEMLOAD = '''
    # ---- semantic sentence embeddings (fix v2) ----
    import json as _json, pickle as _pk
    _E = np.load('sem/essay_sem.npy', mmap_mode='r')
    _I = _json.load(open('sem/essay_index.json'))
    def _sem(path, d):
        rows = _pk.load(open(path, 'rb'))
        assert len(rows) == len(d['prompt_ids']), (path, len(rows), len(d['prompt_ids']))
        assert all(str(r['prompt_id']) == str(q) for r, q in zip(rows, d['prompt_ids']))
        return (np.stack([_E[_I[str(r['essay_id'])]] for r in rows]).astype('float16'),
                np.array([[int(r['prompt_id'])] for r in rows], dtype='int32'))
    X_train_esem, X_train_psem = _sem(train_path, train_data)
    X_dev_esem, X_dev_psem = _sem(dev_path, dev_data)
    X_test_esem, X_test_psem = _sem(test_path, test_data)
    print('semantic inputs:', X_train_esem.shape, X_train_psem.shape, X_test_esem.shape)
'''
AUG = '''
    # ---- mismatch augmentation ----
    ratio = float(os.environ.get('MISMATCH_RATIO', '0.5'))
    if ratio > 0:
        from utils.general_utils import get_score_vector_positions
        posd = get_score_vector_positions()
        rng = np.random.RandomState(seed)
        pids = np.array([str(p) for p in train_data['prompt_ids']])
        uniq = sorted(set(pids))
        first = {p: int(np.where(pids == p)[0][0]) for p in uniq}
        n0 = len(pids)
        idx = rng.choice(n0, int(n0 * ratio), replace=False)
        src = np.array([first[rng.choice([q for q in uniq if q != pids[j]])] for j in idx])
        X_train_pos = np.concatenate([X_train_pos, X_train_pos[idx]])
        X_train_prompt = np.concatenate([X_train_prompt, X_train_prompt[src]])
        X_train_prompt_pos = np.concatenate([X_train_prompt_pos, X_train_prompt_pos[src]])
        X_train_linguistic_features = np.concatenate([X_train_linguistic_features, X_train_linguistic_features[idx]])
        X_train_readability = np.concatenate([X_train_readability, X_train_readability[idx]])
        X_train_esem = np.concatenate([X_train_esem, X_train_esem[idx]])
        X_train_psem = np.concatenate([X_train_psem, X_train_psem[src]])
        Yn = np.array(Y_train[idx], dtype=float).copy()
        for t in ['content', 'prompt_adherence']:
            Yn[:, posd[t]] = 0.0
        for t in ['score', 'organization', 'narrativity']:
            Yn[:, posd[t]] = -1.0
        Y_train = np.concatenate([np.array(Y_train, dtype=float), Yn])
        print('mismatch augmentation: %d originals + %d mismatched copies' % (n0, len(idx)))
'''
s, k1 = re.subn(r'^(\s*)X_train_attribute_rel = get_attribute_masks\(Y_train\)',
                (SEMLOAD + AUG).rstrip('\n') + r'\n\1X_train_attribute_rel = get_attribute_masks(Y_train)', s, count=1, flags=re.M)
s, k2 = re.subn(r'^(\s*)(train|dev|test)_features_list = \[(.*)\]\s*$', r'\1\2_features_list = [\3, X_\2_esem, X_\2_psem]', s, flags=re.M)
print('train patches:', k1, k2)
assert k1 == 1 and k2 == 3
open('train_fix.py', 'w').write(s)
r = subprocess.run("python -c \"import ast; ast.parse(open('train_fix.py').read()); ast.parse(open('models/ProTACT_fix.py').read()); print('v2 scripts OK')\"; grep -n -E '_features_list = ' train_fix.py | cut -c1-160",
                   shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr[-2000:])

os.makedirs('logs', exist_ok=True); os.makedirs('results', exist_ok=True)
ARGS = ['--model_name', 'ProTACT', '--seed', '12', '--num_heads', '2', '--features_path', 'data/LDA/hand_crafted_final_']


def summary(path):
    d = json.load(open(path))
    print('best dev epoch', d['best_dev_epoch'], '| best test avg QWK', round(d['best_test_avg_qwk'], 3))
    for t in d['base_qwk_reloaded']:
        ch = [d['swaps'][k][t]['mean_change'] for k in d['swaps']]
        lo = [d['swaps'][k][t]['pct_essays_lower'] for k in d['swaps']]
        qw = [d['swaps'][k][t]['qwk_wrong_prompt'] for k in d['swaps']]
        print(t.ljust(18), 'QWK %.3f | wrong prompt: mean change %+.3f, %% lower %.0f, QWK %.3f'
              % (d['base_qwk_reloaded'][t], np.mean(ch), np.mean(lo), np.mean(qw)))


def smoke(prompt='1', epochs='2', gpu='0', ratio='0.5'):
    env = dict(os.environ, TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES=gpu, EPOCHS_OVERRIDE=epochs,
               RUN_TAG='smoke3', MISMATCH_RATIO=ratio)
    t = time.time()
    r = subprocess.run(['python', '-u', 'train_fix.py', '--test_prompt_id', prompt] + ARGS, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    open('logs/smoke3_p%s.log' % prompt, 'w').write(out)
    print('\n'.join([l for l in out.splitlines() if any(k in l for k in ['semantic inputs', 'mismatch augmentation', 'Traceback', 'Error'])][-15:]))
    if r.returncode != 0:
        print(out[-3000:])
    else:
        summary('results/smoke3_p%s_s12.json' % prompt)
    print('smoke finished in %.1f min, return code %d' % ((time.time() - t) / 60, r.returncode))


RUNNER = r'''
import subprocess, os, sys, time
gpu, prompts, tag, ratio = sys.argv[1], sys.argv[2].split(','), sys.argv[3], sys.argv[4]
env = dict(os.environ, TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES=gpu, RUN_TAG=tag, MISMATCH_RATIO=ratio)
for p in prompts:
    if os.path.exists('results/%s_p%s_s12.json' % (tag, p)):
        continue
    t = time.time()
    with open('logs/%s_p%s_s12.log' % (tag, p), 'w') as f:
        subprocess.run(['python', '-u', 'train_fix.py', '--test_prompt_id', p, '--model_name', 'ProTACT', '--seed', '12',
                        '--num_heads', '2', '--features_path', 'data/LDA/hand_crafted_final_'],
                       stdout=f, stderr=subprocess.STDOUT, env=env)
    with open('logs/%s_queue_gpu%s.txt' % (tag, gpu), 'a') as f:
        f.write('prompt %s done in %.0f min\n' % (p, (time.time() - t) / 60))
'''
# One queue: two parallel jobs need more than Kaggle's 31 GB RAM.
Q = {'0': '1,3,5,7,2,4,6,8'}


def start(tag='gate', ratio='0.5'):
    open('runner_fix.py', 'w').write(RUNNER)
    for g, q in Q.items():
        subprocess.Popen('nohup python -u runner_fix.py %s %s %s %s > logs/runner_%s_gpu%s.out 2>&1 &' % (g, q, tag, ratio, tag, g), shell=True)
    print(tag, 'queues started:', Q)


def st(tag='gate'):
    for g in Q:
        f = 'logs/%s_queue_gpu%s.txt' % (tag, g)
        print('GPU', g, open(f).read().strip() if os.path.exists(f) else '(none finished yet)')
    for lg in sorted(glob.glob('logs/%s_p*_s12.log' % tag)):
        lines = [l for l in open(lg).read().splitlines() if 'Epoch' in l or 'QWK' in l or 'Error' in l]
        print(lg, '|', lines[-1][:160] if lines else '(starting)')
    for r in sorted(glob.glob('results/%s_p*_s12.json' % tag)):
        print('==', r); summary(r)


# ---- run all 8 prompts one after another (blocking, for a committed Kaggle run) ----
import shutil
for p in ['1', '3', '5', '7', '2', '4', '6', '8']:
    if os.path.exists('results/gate_p%s_s12.json' % p):
        continue
    env = dict(os.environ, TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES='0', RUN_TAG='gate', MISMATCH_RATIO='0.5')
    t = time.time()
    with open('logs/gate_p%s_s12.log' % p, 'w') as f:
        subprocess.run(['python', '-u', 'train_fix.py', '--test_prompt_id', p] + ARGS, stdout=f, stderr=subprocess.STDOUT, env=env)
    print('\n#### prompt', p, 'finished in %.0f min' % ((time.time() - t) / 60), flush=True)
    try:
        summary('results/gate_p%s_s12.json' % p)
    except Exception as e:
        print('no result for prompt', p, e)
        print(open('logs/gate_p%s_s12.log' % p).read()[-3000:])
    os.makedirs('/kaggle/working/gate_results', exist_ok=True)
    for f in glob.glob('results/gate_p*_s12.json') + glob.glob('logs/gate_p*_s12.log'):
        shutil.copy(f, '/kaggle/working/gate_results/')
print('ALL DONE')
