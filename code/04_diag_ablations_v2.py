# Diagnostic 2: verify the reloaded model's QWK, and see which inputs drive the scores.
diag2_block = r'''

def diag2(L):
    import json
    model = L['model']
    model.load_weights('results/protact_p1_s12.weights.h5')
    ev = L['evaluator']
    tf_ = L['test_features_list']
    ids = L['test_data']['prompt_ids']
    Y = L['Y_test']
    n = len(ids)
    true_org = separate_and_rescale_attributes_for_scoring(Y, ids)
    def qwk_avg(pred):
        d = separate_and_rescale_attributes_for_scoring(pred, ids)
        q = {t: float(kappa(true_org[t], d[t], 'quadratic')) for t in d}
        return float(np.mean(list(q.values()))), q
    base = model.predict(tf_, batch_size=32, verbose=0)
    out = {}
    b_avg, b_q = qwk_avg(base)
    out['reloaded_model'] = {'avg_qwk': b_avg, 'per_trait': b_q}
    print('RELOADED model test avg QWK:', round(b_avg, 3), {k: round(v, 3) for k, v in b_q.items()})
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    tests = {
        'swap prompt (to prompt 2)': None,
        'shuffle essay POS text': [0],
        'shuffle linguistic features': [3],
        'shuffle readability features': [4],
        'shuffle both feature sets': [3, 4],
        'shuffle essay text + features': [0, 3, 4],
    }
    for name, cols in tests.items():
        f = list(tf_)
        if cols is None:
            idx = [i for i, p in enumerate(L['train_data']['prompt_ids']) if str(p) == '2'][0]
            f[1] = np.repeat(L['X_train_prompt'][idx][None, :], n, axis=0)
            f[2] = np.repeat(L['X_train_prompt_pos'][idx][None, :], n, axis=0)
        else:
            for c in cols:
                f[c] = tf_[c][perm]
        p = model.predict(f, batch_size=32, verbose=0)
        a, q = qwk_avg(p)
        out[name] = {'max_abs_change': float(np.abs(p - base).max()), 'mean_abs_change': float(np.abs(p - base).mean()), 'avg_qwk': a}
        print(name.ljust(32), '| max change', '%.6f' % out[name]['max_abs_change'], '| mean change', '%.6f' % out[name]['mean_abs_change'], '| avg QWK', round(a, 3))
    json.dump(out, open('results/diag2_p1_s12.json', 'w'), indent=1)
    print('saved results/diag2_p1_s12.json')
'''
src = open('train_pilot.py').read()
src = src.replace('run_mismatch(locals())', 'diag2(locals())', 1)
src = src.replace("if __name__ == '__main__':", diag2_block + "\n\nif __name__ == '__main__':", 1)
open('diag2_pilot.py', 'w').write(src)
print(subprocess.run('grep EPCHS configs/configs.py; grep EPOCHS configs/configs.py; python -c "import ast; ast.parse(open(\'diag2_pilot.py\').read()); print(\'diag2 OK\')"', shell=True, capture_output=True, text=True).stdout)
LOG = 'logs/diag2_p1_s12.log'
T0 = time.time()
proc = subprocess.Popen('python -u diag2_pilot.py --test_prompt_id 1 --model_name ProTACT --seed 12 --num_heads 2 --features_path data/LDA/hand_crafted_final_ > ' + LOG + ' 2>&1', shell=True, env=env)
print('started diagnostic 2, pid', proc.pid)
