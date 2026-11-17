"""Downloads GloVe if needed and builds train_pilot.py: ProTACT training + wrong-prompt test (run_mismatch)."""
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
