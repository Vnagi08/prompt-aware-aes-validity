"""Runs train_pilot.py for all 8 target prompts, two queues on two GPUs; st() prints progress."""
import os, subprocess, time, json, glob
os.chdir('/kaggle/working/ProTACT')
os.makedirs('logs', exist_ok=True); os.makedirs('results', exist_ok=True)
RUNNER = r'''
import subprocess, os, sys, time
gpu, prompts = sys.argv[1], sys.argv[2].split(',')
env = dict(os.environ, TF_USE_LEGACY_KERAS='1', CUDA_VISIBLE_DEVICES=gpu)
for p in prompts:
    if os.path.exists('results/pilot_p%s_s12.json' % p):
        continue
    t = time.time()
    with open('logs/full_p%s_s12.log' % p, 'w') as f:
        subprocess.run(['python', '-u', 'train_pilot.py', '--test_prompt_id', p, '--model_name', 'ProTACT',
                        '--seed', '12', '--num_heads', '2', '--features_path', 'data/LDA/hand_crafted_final_'],
                       stdout=f, stderr=subprocess.STDOUT, env=env)
    with open('logs/queue_gpu%s.txt' % gpu, 'a') as f:
        f.write('prompt %s done in %.0f min\n' % (p, (time.time() - t) / 60))
'''
open('runner.py', 'w').write(RUNNER)
Q = {'0': '1,3,5,7', '1': '2,4,6,8'}
procs = {g: subprocess.Popen('nohup python -u runner.py %s %s > logs/runner_gpu%s.out 2>&1 &' % (g, q, g), shell=True) for g, q in Q.items()}
print('queues started:', Q)

def st():
    for g in Q:
        print('GPU', g, open('logs/queue_gpu%s.txt' % g).read().strip() if os.path.exists('logs/queue_gpu%s.txt' % g) else '(none finished yet)')
    for lg in sorted(glob.glob('logs/full_p*_s12.log')):
        lines = [l for l in open(lg).read().splitlines() if 'Epoch' in l or 'QWK' in l or 'kappa' in l.lower() or 'Error' in l]
        print(lg, '|', lines[-1][:160] if lines else '(starting)')
    for r in sorted(glob.glob('results/pilot_p*_s12.json')):
        d = json.load(open(r))
        print(r, 'best test avg QWK', round(d['best_test_avg_qwk'], 3))
time.sleep(60); st()
