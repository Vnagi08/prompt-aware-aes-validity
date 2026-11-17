# Diagnostic: reload the saved pilot model, check the swapped prompt inputs really differ,
# and measure raw (unrounded) prediction changes.
diag_block = r'''

def diag(L):
    model = L['model']
    model.load_weights('results/protact_p1_s12.weights.h5')
    tf_ = L['test_features_list']
    ids = L['test_data']['prompt_ids']
    n = len(ids)
    base = model.predict(tf_, batch_size=32, verbose=0)
    print('base pred shape', base.shape, '| first essay', np.round(base[0], 4))
    print('test prompt-word rows identical across essays:', bool((tf_[1] == tf_[1][0]).all()), '| nonzero tokens:', int((tf_[1][0] != 0).sum()))
    trD, dvD = L['train_data'], L['dev_data']
    for k in range(2, 9):
        idx = [i for i, p in enumerate(trD['prompt_ids']) if str(p) == str(k)]
        if not idx:
            print('prompt', k, 'missing'); continue
        rw, rp = L['X_train_prompt'][idx[0]], L['X_train_prompt_pos'][idx[0]]
        same_w = bool(np.array_equal(rw, tf_[1][0])); same_p = bool(np.array_equal(rp, tf_[2][0]))
        f = list(tf_)
        f[1] = np.repeat(rw[None, :], n, axis=0); f[2] = np.repeat(rp[None, :], n, axis=0)
        sw = model.predict(f, batch_size=32, verbose=0)
        d = sw - base
        print('swap to prompt', k, '| words same as P1:', same_w, '| pos same:', same_p, '| nonzero tokens:', int((rw != 0).sum()),
              '| max |diff|:', float(np.abs(d).max()), '| mean diff per output:', np.round(d.mean(axis=0), 5))
    # also: zero out the prompt entirely
    f = list(tf_); f[1] = np.zeros_like(tf_[1]); f[2] = np.zeros_like(tf_[2])
    z = model.predict(f, batch_size=32, verbose=0)
    print('EMPTY prompt | max |diff|:', float(np.abs(z - base).max()), '| mean diff per output:', np.round((z - base).mean(axis=0), 5))
    # and: shuffle the essay POS input (sanity: model must react to the essay)
    f = list(tf_); f[0] = tf_[0][np.random.permutation(n)]
    s = model.predict(f, batch_size=32, verbose=0)
    print('SHUFFLED essays (sanity) | max |diff|:', float(np.abs(s - base).max()))
'''
src = open('train_pilot.py').read()
src = src.replace('run_mismatch(locals())', 'diag(locals())', 1)
src = src.replace("if __name__ == '__main__':", diag_block + "\n\nif __name__ == '__main__':", 1)
open('diag_pilot.py', 'w').write(src)
cfg = open('configs/configs.py').read()
open('configs/configs.py', 'w').write(cfg.replace('EPOCHS = 15', 'EPOCHS = 0'))
print(subprocess.run('grep EPOCHS configs/configs.py; python -c "import ast; ast.parse(open(\'diag_pilot.py\').read()); print(\'diag OK\')"', shell=True, capture_output=True, text=True))
LOG = 'logs/diag_p1_s12.log'
T0 = time.time()
proc = subprocess.Popen('python -u diag_pilot.py --test_prompt_id 1 --model_name ProTACT --seed 12 --num_heads 2 --features_path data/LDA/hand_crafted_final_ > ' + LOG + ' 2>&1', shell=True, env=env)
print('started diagnostic, pid', proc.pid)
