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
