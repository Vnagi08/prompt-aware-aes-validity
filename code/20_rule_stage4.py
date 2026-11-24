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

# ======================= Rule-based (non-learned) relevance baseline =======================
# Baseline ProTACT weights (trained, ungated) are loaded; no training happens here (EPOCHS = 0).
# Content and Prompt Adherence are multiplied by a fixed rule on the mean-embedding cosine
# c = cos(mean essay sentence embedding, mean prompt sentence embedding) (all-MiniLM-L6-v2,
# prompts 3-6 with our passage summaries, as for the gate in Table V).
# Thresholds come from the 7 training prompts only: lo = median c(essay, wrong prompt),
# hi = median c(essay, own prompt). hard: g = 1[c >= (lo+hi)/2]; ramp: g = clip((c-lo)/(hi-lo), 0, 1).
import json, pickle, numpy as np, shutil
os.chdir('/kaggle/working/ProTACT')
SEED = os.environ.get('RULE_SEED', '42')
SUMMARIES = {
 3: "Summary of the source essay 'Rough Road Ahead: Do Not Exceed Posted Speed Limit' by Joe Kurmaskie: a cyclist on a long summer ride follows a shortcut suggested by old men whose directions are out of date. The road is rough and hilly, the heat is intense, the towns he passes are abandoned, and his water runs low until he is dehydrated and exhausted before he finally reaches a store.",
 4: "Summary of the story 'Winter Hibiscus' by Minfong Ho: Saeng, a young immigrant girl, fails her driving test and stops at a greenhouse where she finds a hibiscus and other plants from her homeland. The plants bring back memories of her old home, and she buys a hibiscus to plant with her mother. In the final paragraph she vows to take the test again in the spring, when the geese return and the hibiscus buds.",
 5: "Summary of the memoir 'Narciso Rodriguez' from Home: The Blueprints of Our Lives: the fashion designer describes growing up in Newark, New Jersey, in the home of his Cuban immigrant parents. He remembers the warmth of the kitchen, Cuban music and food, neighbours treated as family, and his gratitude for the sacrifices his parents made so that he could have a better life.",
 6: "Summary of the excerpt 'The Mooring Mast' by Marcia Amidon Lusted: the builders of the Empire State Building planned a mast at the top where airships, or dirigibles, could dock. They had to strengthen the steel frame for the stress of a moored airship, and they faced strong winds at the top, the danger of flammable hydrogen over a crowded city, and laws against airships flying low over urban areas, so the mast was never used as planned.",
}

PARAPHRASES = {
 1: ["Write a letter to the editor of your local newspaper explaining what you believe about how computers affect people, and try to convince readers that your view is correct.",
     "Do computers do more good than harm for people? State your position in a letter to a newspaper and support it with reasons and examples.",
     "Some people say computers help us learn and stay in touch; others worry that people spend too much time on them instead of exercising or being with family. In a letter to your newspaper, argue for your opinion on the effects of computers."],
 2: ["Write an essay for a newspaper giving your opinion on censorship in libraries: should books, music, movies or magazines be taken off the shelves if some people find them offensive?",
     "Should libraries remove materials that some readers find offensive? Write a persuasive newspaper essay that takes a position and supports it with reasons.",
     "Think about whether offensive books, films, music and magazines should be removed from library shelves. Persuade the readers of a newspaper to share your view on library censorship."],
 3: ["Explain how the setting of 'Rough Road Ahead' affects the cyclist, using examples from the essay.",
     "Using details from 'Rough Road Ahead', describe how features of the setting change what happens to the cyclist.",
     "How do the road, the heat and the places along the way affect the cyclist in 'Rough Road Ahead'? Support your answer with examples from the essay."],
 4: ["Explain why Minfong Ho ends 'Winter Hibiscus' with the paragraph in which Saeng vows to take the test again in the spring. Use details from the story.",
     "Why does the author close the story 'Winter Hibiscus' with Saeng's promise to retake the test when the geese return and the hibiscus buds? Support your answer with details from the story.",
     "Write a response explaining the purpose of the final paragraph of 'Winter Hibiscus', using details from the story."],
 5: ["Describe the mood the author creates in the memoir 'Narciso Rodriguez', using relevant and specific details from the memoir.",
     "What feeling does the memoir 'Narciso Rodriguez' create for the reader? Explain with specific examples from the text.",
     "Explain the mood of the memoir 'Narciso Rodriguez' and support your answer with details from the memoir."],
 6: ["Using the excerpt 'The Mooring Mast', describe the problems the builders of the Empire State Building faced in letting dirigibles dock there.",
     "What obstacles stood in the way of docking airships at the Empire State Building? Answer with details from 'The Mooring Mast'.",
     "Based on 'The Mooring Mast', explain the difficulties the builders faced when they tried to make the Empire State Building a dock for dirigibles."],
 7: ["Write about patience: tell about a time when you, or someone you know, was patient, or write about patience in your own way.",
     "Being patient means being understanding and tolerant. Write a story about a time you or someone you know showed patience.",
     "Tell about a time when you had to wait calmly and patiently for something, or write about patience in any way you like."],
 8: ["Tell a true story in which laughter played a part.",
     "Laughter brings people together. Write a true story in which laughter was one of the elements.",
     "Write a true story about a time when laughter was part of what happened."],
}
NEARMISS = {
 1: "Write a letter to your local newspaper about the effects television has on people, and persuade readers to agree with your opinion.",
 2: "Write a persuasive essay for a newspaper arguing whether public libraries should stay open longer hours.",
 3: "In 'Rough Road Ahead', describe how the cyclist's attitude toward the old men who gave him directions changes during his ride.",
 4: "In 'Winter Hibiscus', describe the relationship between Saeng and her mother.",
 5: "In the memoir 'Narciso Rodriguez', explain what the author's parents gave up when they came to the United States.",
 6: "Based on 'The Mooring Mast', explain why dirigibles were seen as the transportation of the future in the 1920s and 1930s.",
 7: "Write about impatience. Tell about a time when you or someone you know was impatient and what happened.",
 8: "Tell a true story in which sadness was one of the elements.",
}
UNSEEN = {
 'U1_persuasive': "Write a letter to your school board arguing whether students should be required to wear school uniforms.",
 'U2_persuasive': "Should cities ban cars from their downtown areas? Write an essay persuading readers to accept your view.",
 'U3_source': "After reading the short story 'The Lighthouse Keeper's Daughter', explain how the storm changes the main character's view of her father.",
 'U4_source': "Using details from the article 'Saving the Honeybees', describe the steps farmers have taken to protect bee colonies.",
 'U5_narrative': "Write about a time when you had to be brave. Tell the story of what happened.",
 'U6_narrative': "Tell a true story about a time you helped a friend solve a problem.",
}


def build_means():
    if os.path.exists('sem/essay_sents.npy'):
        return
    subprocess.run('pip install -q sentence-transformers', shell=True)
    from sentence_transformers import SentenceTransformer
    import nltk, pandas as pd
    os.makedirs('sem', exist_ok=True)
    rows = {}
    for split in ['train', 'dev', 'test']:
        for r in pickle.load(open('data/cross_prompt_attributes/1/%s.pk' % split, 'rb')):
            rows[str(r['essay_id'])] = r['content_text']
    ids = sorted(rows)
    enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    sents, owner = [], []
    for i, e in enumerate(ids):
        ss = [s for s in nltk.sent_tokenize(rows[e]) if s.strip()][:40] or [rows[e][:500]]
        sents += ss; owner += [i] * len(ss)
    t = time.time()
    emb = enc.encode(sents, batch_size=128, show_progress_bar=False, convert_to_numpy=True)
    print('encoded', len(sents), 'sentences in %.0f s' % (time.time() - t), flush=True)
    M = np.zeros((len(ids), emb.shape[1])); cnt = np.zeros(len(ids))
    for v, o in zip(emb, owner):
        M[o] += v; cnt[o] += 1
    M /= cnt[:, None]
    np.save('sem/essay_mean.npy', M); json.dump({e: i for i, e in enumerate(ids)}, open('sem/essay_mean_index.json', 'w'))
    np.save('sem/essay_sents.npy', emb.astype('float16')); np.save('sem/essay_owner.npy', np.array(owner, dtype='int32'))
    json.dump({'PARAPHRASES': {str(k): v for k, v in PARAPHRASES.items()}, 'NEARMISS': {str(k): v for k, v in NEARMISS.items()}, 'UNSEEN': UNSEEN, 'SUMMARIES': {str(k): v for k, v in SUMMARIES.items()}}, open('sem/stage4_prompts.json', 'w'))
    df = pd.read_csv('data/prompt_info.csv')
    P = np.zeros((9, emb.shape[1]))
    for _, r in df.iterrows():
        k = int(r['prompt_id']); text = str(r['prompt']) + (' ' + SUMMARIES[k] if k in SUMMARIES else '')
        ss = [s for s in nltk.sent_tokenize(text) if s.strip()][:30]
        P[k] = enc.encode(ss, convert_to_numpy=True).mean(0)
    np.save('sem/prompt_mean.npy', P)

build_means()

RULE_FN = """

def rule_eval(L):
    import json, glob, pickle
    from utils.general_utils import get_score_vector_positions
    model, tp = L['model'], L['test_prompt_id']
    seed = os.environ.get('RULE_SEED', '42')
    wf = sorted(glob.glob('/kaggle/input/**/protact_p%d_s%s.weights.h5' % (tp, seed), recursive=True) + glob.glob('/kaggle/working/base_weights/**/protact_p%d_s%s.weights.h5' % (tp, seed), recursive=True))
    print('weights:', wf[:1], flush=True)
    model.load_weights(wf[0])
    M = np.load('sem/essay_mean.npy'); I = json.load(open('sem/essay_mean_index.json')); P = np.load('sem/prompt_mean.npy')
    def cos(a, b):
        return (a @ b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b) + 1e-9)
    def emb(path):
        rows = pickle.load(open(path, 'rb'))
        return M[[I[str(r['essay_id'])] for r in rows]], np.array([int(r['prompt_id']) for r in rows])
    Etr, ptr = emb(L['train_path'])
    trp = sorted(set(ptr.tolist()))
    own = np.array([cos(Etr[i], P[ptr[i]]) for i in range(len(ptr))])
    wrong = np.concatenate([cos(Etr[ptr != k], P[k]) for k in trp])
    lo, hi = float(np.median(wrong)), float(np.median(own))
    tau = (lo + hi) / 2
    rules = {'hard': lambda c: (c >= tau).astype(float), 'ramp': lambda c: np.clip((c - lo) / (hi - lo), 0, 1)}
    Ete, pte = emb(L['test_path'])
    assert (pte == tp).all()
    feats = L['test_features_list']; ids = L['test_data']['prompt_ids']
    true_org = separate_and_rescale_attributes_for_scoring(L['Y_test'], ids)
    base = model.predict(feats, batch_size=64, verbose=0)
    posd = get_score_vector_positions()
    gated = [t for t in ['content', 'prompt_adherence'] if t in true_org]
    def apply(g):
        y = base.copy()
        for t in gated:
            y[:, posd[t]] = y[:, posd[t]] * g
        return separate_and_rescale_attributes_for_scoring(y, ids)
    def q(d):
        return {t: float(kappa(true_org[t], d[t], 'quadratic')) for t in d}
    bd = separate_and_rescale_attributes_for_scoring(base, ids)
    out = {'test_prompt': tp, 'seed': seed, 'lo': lo, 'hi': hi, 'tau': tau, 'base_qwk': q(bd)}
    c_own = cos(Ete, P[tp])
    for rn, rf in rules.items():
        g0 = rf(c_own); d0 = apply(g0); q0 = q(d0)
        r = {'mean_g': float(g0.mean()), 'share_g0': float((g0 == 0).mean()), 'qwk': q0, 'avg_qwk': float(np.mean([q0[t] for t in true_org])), 'swaps': {}}
        for k in range(1, 9):
            if k == tp:
                continue
            dk = apply(rf(cos(Ete, P[k])))
            r['swaps'][str(k)] = {t: {'mean_change': float((np.array(dk[t], float) - np.array(d0[t], float)).mean()),
                                      'pct_lower': float((np.array(dk[t], float) < np.array(d0[t], float)).mean() * 100),
                                      'qwk_wrong': float(kappa(true_org[t], dk[t], 'quadratic'))} for t in gated}
        out[rn] = r
        msg = ['%s p%d: avg QWK %.3f (base %.3f) g_mean %.3f' % (rn, tp, r['avg_qwk'], np.mean(list(out['base_qwk'].values())), r['mean_g'])]
        for t in gated:
            sw = [r['swaps'][k][t] for k in r['swaps']]
            msg.append('%s QWK %.3f (base %.3f) wrong: chg %.3f lower %.1f%% qwk %.3f' % (t, q0[t], out['base_qwk'][t], np.mean([s['mean_change'] for s in sw]), np.mean([s['pct_lower'] for s in sw]), np.mean([s['qwk_wrong'] for s in sw])))
        print(' | '.join(msg), flush=True)
    # ---- further tests (as Section beyond): paraphrases, same passage/nearby, unseen prompts, splice ----
    import nltk
    from sentence_transformers import SentenceTransformer
    V = json.load(open('sem/stage4_prompts.json'))
    enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    summ = V['SUMMARIES'].get(str(tp), '')
    def pmean(text):
        ss = [x for x in nltk.sent_tokenize(text) if x.strip()][:30]
        return enc.encode(ss, convert_to_numpy=True).mean(0)
    variants = [('paraphrase_%d' % (j + 1), p + (' ' + summ if summ else '')) for j, p in enumerate(V['PARAPHRASES'][str(tp)])]
    variants += [('near_miss', V['NEARMISS'][str(tp)] + (' ' + summ if summ else ''))]
    variants += [(k, v) for k, v in V['UNSEEN'].items()]
    S = np.load('sem/essay_sents.npy').astype(np.float32); O = np.load('sem/essay_owner.npy')
    starts = np.searchsorted(O, np.arange(len(M))); ends = np.searchsorted(O, np.arange(len(M)), side='right')
    te_rows = [I[str(r['essay_id'])] for r in pickle.load(open(L['test_path'], 'rb'))]
    tr_rows = [I[str(r['essay_id'])] for r in pickle.load(open(L['train_path'], 'rb'))]
    gold_c = np.array(true_org['content'], float)
    for rn, rf in rules.items():
        g0 = rf(c_own); d0 = apply(g0)
        def comp(dk):
            r = {}
            for t in gated:
                b = np.array(d0[t], float); x = np.array(dk[t], float)
                r[t] = {'mean_change': float((x - b).mean()), 'pct_lower': float((x < b).mean() * 100), 'qwk': float(kappa(true_org[t], dk[t], 'quadratic'))}
            return r
        out[rn]['signed_error_content'] = float((np.array(d0['content'], float) - gold_c).mean())
        out[rn]['variants'] = {}
        for name, text in variants:
            out[rn]['variants'][name] = comp(apply(rf(cos(Ete, pmean(text)))))
        rng = np.random.RandomState(0)
        out[rn]['splice'] = {}
        for frac in [0.25, 0.5, 0.75, 1.0]:
            Em = Ete.copy()
            for ii, row in enumerate(te_rows):
                own_s = S[starts[row]:ends[row]]
                k = int(round(frac * len(own_s)))
                if k == 0:
                    continue
                drow = tr_rows[rng.randint(len(tr_rows))]
                dv = S[starts[drow]:ends[drow]]
                keep = np.ones(len(own_s), bool); keep[rng.choice(len(own_s), k, replace=False)] = False
                Em[ii] = np.concatenate([own_s[keep], dv[rng.randint(len(dv), size=k)]]).mean(0)
            out[rn]['splice'][str(frac)] = comp(apply(rf(cos(Em, P[tp]))))
        vv = out[rn]['variants']
        print('%s p%d further: para %s | near %.3f (%.1f%%) | unseen %.2f..%.2f | splice %s | signed err %.2f' % (rn, tp,
              '/'.join('%.2f' % vv['paraphrase_%d' % j]['content']['mean_change'] for j in (1, 2, 3)),
              vv['near_miss']['content']['mean_change'], vv['near_miss']['content']['pct_lower'],
              max(vv[k]['content']['mean_change'] for k in V['UNSEEN']), min(vv[k]['content']['mean_change'] for k in V['UNSEEN']),
              '/'.join('%.2f' % out[rn]['splice'][f]['content']['mean_change'] for f in out[rn]['splice']), out[rn]['signed_error_content']), flush=True)
    os.makedirs('/kaggle/working/rule_results', exist_ok=True)
    json.dump(out, open('/kaggle/working/rule_results/rule_p%d_s%s.json' % (tp, seed), 'w'), indent=1)
"""
s = open('train_pilot.py').read()
s = s.replace('run_mismatch(locals())', 'rule_eval(locals())', 1)
s = s.replace('evaluator.print_final_info()', 'pass  # no training in the rule baseline', 1)
s = s.replace("if __name__ == '__main__':", RULE_FN + "\n\nif __name__ == '__main__':", 1)
open('rule_pilot.py', 'w').write(s)
c = open('configs/configs.py').read(); c = re.sub(r'EPOCHS\s*=\s*\d+', 'EPOCHS = 0', c); open('configs/configs.py', 'w').write(c)
print(subprocess.run('python -c "import ast; ast.parse(open(\'rule_pilot.py\').read()); print(\'rule script OK\')"', shell=True, capture_output=True, text=True).stdout)
for SEED in [x for x in ['42', '32', '52'] if glob.glob('/kaggle/input/**/protact_p1_s%s.weights.h5' % x, recursive=True)]:
    print('=== SEED', SEED, flush=True)
    WDIR = '/kaggle/working/base_weights'
    if not glob.glob(WDIR + '/**/protact_p1_s%s.weights.h5' % SEED, recursive=True):
        import kagglehub
        ver = {'42': 9, '52': 8, '32': 11}[SEED]
        for h in ['vishalnagi/notebook6942c2a9d0/versions/%d' % ver]:
            try:
                d = kagglehub.notebook_output_download(h)
                print('downloaded', h, '->', d, flush=True)
                if not os.path.exists(WDIR):
                    os.symlink(d, WDIR)
                break
            except Exception as e:
                print('kagglehub failed for', h, repr(e)[:300], flush=True)
    print('weights found:', sorted(glob.glob('/kaggle/input/**/protact_p*_s*.weights.h5', recursive=True) + glob.glob(WDIR + '/**/protact_p*_s*.weights.h5', recursive=True))[:20], flush=True)
    env = dict(os.environ, TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES='', RULE_SEED=SEED)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('/kaggle/working/rule_results', exist_ok=True)
    for p in '1 2 3 4 5 6 7 8'.split():
        with open('logs/rule_p%s_s%s.log' % (p, SEED), 'w') as f:
            rc = subprocess.run(['python', '-u', 'rule_pilot.py', '--test_prompt_id', p, '--model_name', 'ProTACT', '--seed', SEED, '--num_heads', '2', '--features_path', 'data/LDA/hand_crafted_final_'], stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        lines = [l for l in open('logs/rule_p%s_s%s.log' % (p, SEED)).read().splitlines() if l.startswith(('hard', 'ramp', 'weights', 'Traceback', 'Error')) or 'Error' in l]
        print('prompt', p, 'rc', rc, *lines[-6:], sep='\n', flush=True)
        shutil.copy('logs/rule_p%s_s%s.log' % (p, SEED), '/kaggle/working/rule_results/')
    print('RULE BASELINE DONE')
