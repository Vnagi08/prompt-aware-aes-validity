"""The fix: a ProTACT variant that can use the prompt, trained with deliberately mismatched prompts.

1. Architecture (models/ProTACT_fix.py). ProTACT pools the essay and the prompt to one vector each before
   its prompt attention, so the attention has a single key. The fix replaces that layer: every essay
   sentence state attends over the prompt sentence states, and the result is combined ESIM-style
   [e, a, e*a, e-a] -> Dense(100, tanh). The output keeps one vector per essay sentence, so the rest of
   the network is unchanged.
2. Training (train_fix.py). A random MISMATCH_RATIO share of the training essays is copied and paired
   with the prompt of a different training prompt. For these copies Content and Prompt Adherence are set
   to the lowest score (0 after scaling); Word Choice, Sentence Fluency, Conventions and Language keep
   their true labels (they do not depend on the prompt); Overall, Organization and Narrativity are
   masked (-1) because their dependence on the prompt is unclear. The test prompt is never used.
3. Evaluation: the same wrong-prompt test as the baseline (run_mismatch), written to
   results/<RUN_TAG>_p<P>_s<seed>.json.

Run after 01 and 02. smoke() trains 2 epochs on prompt 1; start() runs all 8 prompts on two GPUs.
"""
import os, re, subprocess, time, json, glob
os.chdir('/kaggle/working/ProTACT')

# ---- 1. model ----
m = open('models/ProTACT.py').read()
new_line = (r"\1es_pr_MA_list = [layers.Dense(100, activation='tanh', name='pe_comb_%d' % i)("
            r"layers.Concatenate(name='pe_cat_%d' % i)([e_, a_, layers.Multiply()([e_, a_]), layers.Subtract()([e_, a_])])) "
            r"for i, (e_, a_) in enumerate((e0, layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(1, lstm_units // num_heads), "
            r"name='pe_xattn_%d' % j)(query=e0, value=prompt_MA_lstm_list, key=prompt_MA_lstm_list)) "
            r"for j, e0 in enumerate(pos_MA_lstm_list))]")
m2, n = re.subn(r'^(\s*)es_pr_MA_list\s*=.*$', new_line, m, count=1, flags=re.M)
assert n == 1, 'es_pr_MA_list line not found'
assert 'prompt_MA_lstm_list' in m2 and 'pos_MA_lstm_list' in m2
open('models/ProTACT_fix.py', 'w').write(m2)

# ---- 2. training script (built from train_pilot.py, which already has run_mismatch) ----
s = open('train_pilot.py').read()
s = s.replace('from models.ProTACT import build_ProTACT', 'from models.ProTACT_fix import build_ProTACT', 1)
s = s.replace('epochs = configs.EPOCHS', "epochs = int(os.environ.get('EPOCHS_OVERRIDE', configs.EPOCHS))", 1)
s = s.replace("'results/pilot_p%d_s%d.json'", "'results/' + os.environ.get('RUN_TAG', 'fix') + '_p%d_s%d.json'")
s = s.replace("'results/protact_p%d_s%d.weights.h5'", "'results/' + os.environ.get('RUN_TAG', 'fix') + '_p%d_s%d.weights.h5'")
s = s.replace("print('saved results/pilot_p%d_s%d.json' % (tp, seed))", "print('saved results json for prompt', tp)")
AUG = '''
    # ---- mismatch augmentation (the fix) ----
    ratio = float(os.environ.get('MISMATCH_RATIO', '0.5'))
    if ratio > 0:
        from utils.general_utils import get_score_vector_positions
        posd = get_score_vector_positions()
        print('trait positions:', posd)
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
        Yn = np.array(Y_train[idx], dtype=float).copy()
        for t in ['content', 'prompt_adherence']:
            if t in posd:
                Yn[:, posd[t]] = 0.0
        for t in ['score', 'organization', 'narrativity']:
            if t in posd:
                Yn[:, posd[t]] = -1.0
        Y_train = np.concatenate([np.array(Y_train, dtype=float), Yn])
        print('mismatch augmentation: %d originals + %d mismatched copies' % (n0, len(idx)))
'''
s2, n = re.subn(r'^(\s*)X_train_attribute_rel = get_attribute_masks\(Y_train\)', AUG.rstrip('\n') + r'\n\1X_train_attribute_rel = get_attribute_masks(Y_train)', s, count=1, flags=re.M)
assert n == 1, 'mask line not found'
open('train_fix.py', 'w').write(s2)
r = subprocess.run("python -c \"import ast; ast.parse(open('train_fix.py').read()); ast.parse(open('models/ProTACT_fix.py').read()); print('fix scripts OK')\"; grep -n -E 'ProTACT_fix|EPOCHS_OVERRIDE|mismatch augmentation|RUN_TAG' train_fix.py | cut -c1-120; grep -n 'pe_xattn' models/ProTACT_fix.py | cut -c1-80",
                   shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr[-2000:])

os.makedirs('logs', exist_ok=True); os.makedirs('results', exist_ok=True)
ARGS = ['--model_name', 'ProTACT', '--seed', '12', '--num_heads', '2', '--features_path', 'data/LDA/hand_crafted_final_']


def smoke(prompt='1', epochs='2', gpu='0'):
    env = dict(os.environ, TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES=gpu, EPOCHS_OVERRIDE=epochs, RUN_TAG='smoke')
    t = time.time()
    r = subprocess.run(['python', '-u', 'train_fix.py', '--test_prompt_id', prompt] + ARGS, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    open('logs/smoke_p%s.log' % prompt, 'w').write(out)
    keep = [l for l in out.splitlines() if any(k in l for k in ['trait positions', 'mismatch augmentation', 'Error', 'error', 'PILOT', 'trait', 'content', 'score ', 'organization', 'word_choice', 'sentence', 'conventions', 'best dev', 'X_train_pos'])]
    print('\n'.join(keep[-40:]))
    print('smoke finished in %.1f min, return code %d' % ((time.time() - t) / 60, r.returncode))


RUNNER = r'''
import subprocess, os, sys, time
gpu, prompts = sys.argv[1], sys.argv[2].split(',')
env = dict(os.environ, TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES=gpu, RUN_TAG='fix')
for p in prompts:
    if os.path.exists('results/fix_p%s_s12.json' % p):
        continue
    t = time.time()
    with open('logs/fix_p%s_s12.log' % p, 'w') as f:
        subprocess.run(['python', '-u', 'train_fix.py', '--test_prompt_id', p, '--model_name', 'ProTACT', '--seed', '12',
                        '--num_heads', '2', '--features_path', 'data/LDA/hand_crafted_final_'],
                       stdout=f, stderr=subprocess.STDOUT, env=env)
    with open('logs/fixqueue_gpu%s.txt' % gpu, 'a') as f:
        f.write('prompt %s done in %.0f min\n' % (p, (time.time() - t) / 60))
'''
Q = {'0': '1,3,5,7', '1': '2,4,6,8'}


def start():
    open('runner_fix.py', 'w').write(RUNNER)
    for g, q in Q.items():
        subprocess.Popen('nohup python -u runner_fix.py %s %s > logs/runner_fix_gpu%s.out 2>&1 &' % (g, q, g), shell=True)
    print('fix queues started:', Q)


def st():
    for g in Q:
        f = 'logs/fixqueue_gpu%s.txt' % g
        print('GPU', g, open(f).read().strip() if os.path.exists(f) else '(none finished yet)')
    for lg in sorted(glob.glob('logs/fix_p*_s12.log')):
        lines = [l for l in open(lg).read().splitlines() if 'Epoch' in l or 'QWK' in l or 'Error' in l]
        print(lg, '|', lines[-1][:160] if lines else '(starting)')
    for r in sorted(glob.glob('results/fix_p*_s12.json')):
        d = json.load(open(r))
        ch = {t: round(sum(d['swaps'][k][t]['mean_change'] for k in d['swaps']) / len(d['swaps']), 3) for t in d['base_qwk_reloaded']}
        print(r, 'avg QWK', round(d['best_test_avg_qwk'], 3), '| mean change with wrong prompt:', ch)
