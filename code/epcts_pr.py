"""EPCTS prompt-relevance (PR) feature: re-implementation and validity checks.

PR = (sim_e + sum_i sim_i + alpha + lambda) / (n + 2),  alpha = (1 - 1/n^3) / (1 + 1/n^3)   (Xu et al., 2025, Eqs. 13-16)
Segmentation: open-source Qwen2.5 instruct model (EPCTS used the commercial Qwen-max; its segmentation prompt is not published).
Embeddings: sentence-transformers (EPCTS does not report its embedding model).
lambda (random noise in +-0.03) is set to 0 for analysis.

Usage: python epcts_pr.py --per_prompt 150 --llm Qwen/Qwen2.5-3B-Instruct --gpu 0
"""
import argparse, json, os, re, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--per_prompt', type=int, default=150)
ap.add_argument('--llm', default='Qwen/Qwen2.5-3B-Instruct')
ap.add_argument('--emb', default='sentence-transformers/all-MiniLM-L6-v2')
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='results/epcts_pr')
args = ap.parse_args()
os.makedirs('results', exist_ok=True)
T0 = time.time()
def log(*a):
    print('[%5.0fs]' % (time.time() - T0), *a, flush=True)

# ---------- data ----------
ess = pd.read_csv('data/training_set_rel3.tsv', sep='\t', encoding='latin-1')
ess = ess[['essay_id', 'essay_set', 'essay']]
sample = (ess.groupby('essay_set', group_keys=False)
             .apply(lambda g: g.sample(min(len(g), args.per_prompt), random_state=args.seed))
             .reset_index(drop=True))
log('essays sampled:', len(sample), dict(sample.essay_set.value_counts().sort_index()))

def load_prompts():
    best = None
    for f in ['data/prompt_info.csv', 'data/prompt_info_pp.csv']:
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        log(f, 'columns:', list(df.columns))
        idc = [c for c in df.columns if 'id' in c.lower()][0]
        txt = max([c for c in df.columns if df[c].dtype == object], key=lambda c: df[c].astype(str).str.len().mean())
        cand = {int(r[idc]): str(r[txt]) for _, r in df.iterrows()}
        if best is None or np.mean([len(v) for v in cand.values()]) > np.mean([len(v) for v in best.values()]):
            best = cand
    return best
PROMPTS = load_prompts()
for k in sorted(PROMPTS):
    log('prompt', k, repr(PROMPTS[k][:120]))

# ---------- sentence split ----------
import nltk
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
from nltk.tokenize import sent_tokenize
sample['sents'] = sample.essay.apply(lambda t: [s.strip() for s in sent_tokenize(str(t)) if s.strip()])
sample['n_words'] = sample.essay.apply(lambda t: len(str(t).split()))
sample['n_sents'] = sample.sents.apply(len)

# ---------- LLM segmentation ----------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(args.llm)
tok.padding_side = 'left'
llm = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.float16, device_map='cuda:0')
llm.eval()
SYS = ('You split student essays into semantic segments. Each segment is a run of consecutive sentences '
       'that develops one idea. Reply only with the segments as sentence-number ranges separated by semicolons, '
       'for example: 1-3; 4-4; 5-8. Cover every sentence exactly once, in order.')
def make_msg(sents):
    body = '\n'.join('%d. %s' % (i + 1, s) for i, s in enumerate(sents))
    return tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': body}],
                                   tokenize=False, add_generation_prompt=True)
def parse(ans, n):
    """Lenient parse: accepts 'a-b' and single numbers, any separators; repairs gaps/overlaps.
    Returns (segments, status) with status in {'clean', 'repaired'} or (None, 'failed')."""
    toks = re.findall(r'(\d+)\s*(?:-|to|–)\s*(\d+)|(\d+)', ans)
    starts = []
    for a, b, c in toks:
        a = int(a or c)
        if 1 <= a <= n:
            starts.append(a)
    starts = sorted(set(starts))
    if not starts:
        return None, 'failed'
    clean = parse_strict(ans, n)
    if clean:
        return clean, 'clean'
    if starts[0] != 1:
        starts = [1] + starts
    segs = [(a, (starts[k + 1] - 1) if k + 1 < len(starts) else n) for k, a in enumerate(starts)]
    return segs, 'repaired'
def parse_strict(ans, n):
    rng = re.findall(r'(\d+)\s*-\s*(\d+)', ans)
    segs, nxt = [], 1
    for a, b in rng:
        a, b = int(a), int(b)
        if a != nxt or b < a or b > n:
            return None
        segs.append((a, b)); nxt = b + 1
    return segs if segs and nxt == n + 1 else None

seg_out, fails, repaired, seg_status = [], 0, 0, []
rows = list(sample.itertuples())
for i in range(0, len(rows), args.batch):
    batch = rows[i:i + args.batch]
    enc = tok([make_msg(r.sents) for r in batch], return_tensors='pt', padding=True, truncation=True, max_length=3072).to('cuda:0')
    with torch.no_grad():
        out = llm.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    for r, o in zip(batch, out):
        ans = tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True)
        segs, status = parse(ans, len(r.sents))
        if segs is None:
            fails += 1
            segs = [(k + 1, k + 1) for k in range(len(r.sents))]
        elif status == 'repaired':
            repaired += 1
        seg_status.append(status)
        seg_out.append([' '.join(r.sents[a - 1:b]) for a, b in segs])
    if (i // args.batch) % 10 == 0:
        log('segmented', i + len(batch), '/', len(rows), 'failed', fails, 'repaired', repaired)
sample['segments'] = seg_out
sample['n_seg'] = sample.segments.apply(len)
log('segmentation done; failed (fell back to sentences):', fails, 'repaired:', repaired)
sample['seg_status'] = seg_status
del llm; torch.cuda.empty_cache()

# ---------- embeddings + PR ----------
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer(args.emb, device='cuda:0')
E = lambda texts: emb.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
P = {k: E([v])[0] for k, v in PROMPTS.items()}
ess_vec = E(sample.essay.astype(str).tolist())
recs = []
for j, r in enumerate(sample.itertuples()):
    sv = E(r.segments)
    n = len(r.segments)
    alpha = (1 - 1 / n ** 3) / (1 + 1 / n ** 3)
    for p, pv in P.items():
        sim_e = float(ess_vec[j] @ pv)
        sims = sv @ pv
        recs.append(dict(essay_id=r.essay_id, essay_set=r.essay_set, prompt=p, true=int(p == r.essay_set),
                         n_seg=n, n_sents=r.n_sents, n_words=r.n_words, sim_e=sim_e, mean_sim=float(sims.mean()),
                         alpha=alpha, alpha_term=alpha / (n + 2),
                         PR=(sim_e + float(sims.sum()) + alpha) / (n + 2),
                         PR_no_alpha=(sim_e + float(sims.sum())) / (n + 2)))
df = pd.DataFrame(recs)
stat = dict(zip(sample.essay_id, sample.seg_status))
df['seg_status'] = df.essay_id.map(stat)
df_all = df
df = df[df.seg_status != 'failed']
log('essays kept for analysis (segmentation not failed):', df.essay_id.nunique())
df_all.to_csv(args.out + '.csv', index=False)
sample[['essay_id', 'essay_set', 'n_words', 'n_sents', 'n_seg', 'seg_status']].to_csv(args.out + '_segments.csv', index=False)
log('saved', args.out + '.csv', len(df), 'rows')

# ---------- analysis ----------
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
summ = {}
for s in sorted(df.essay_set.unique()):
    d = df[df.essay_set == s]; t = d[d.true == 1]; w = d[d.true == 0]
    beat = d.groupby('essay_id')[['true', 'PR']].apply(lambda g: g[g.true == 0].PR.max() > g[g.true == 1].PR.iloc[0]).mean()
    summ[int(s)] = dict(
        n_essays=len(t), mean_nseg=round(t.n_seg.mean(), 2),
        rho_PR_nseg=round(spearmanr(t.PR, t.n_seg)[0], 3), rho_PR_words=round(spearmanr(t.PR, t.n_words)[0], 3),
        rho_noalpha_nseg=round(spearmanr(t.PR_no_alpha, t.n_seg)[0], 3),
        mean_PR_true=round(t.PR.mean(), 3), mean_PR_wrong=round(w.PR.mean(), 3),
        mean_alpha_term=round(t.alpha_term.mean(), 3),
        auc_PR=round(roc_auc_score(d.true, d.PR), 3), auc_noalpha=round(roc_auc_score(d.true, d.PR_no_alpha), 3),
        auc_sim_e=round(roc_auc_score(d.true, d.sim_e), 3),
        pct_wrong_beats_true=round(100 * beat, 1))
# cross-essay check: short wrong-prompt pairs vs long true-prompt pairs
t_all, w_all = df[df.true == 1], df[df.true == 0]
short_w = w_all[w_all.n_seg <= 3].PR.values; long_t = t_all[t_all.n_seg >= 8].PR.values
cross = float((short_w[:, None] > long_t[None, :]).mean()) if len(short_w) and len(long_t) else None
summary = dict(per_prompt=summ, pooled=dict(
    rho_PR_nseg=round(spearmanr(t_all.PR, t_all.n_seg)[0], 3),
    auc_PR=round(roc_auc_score(df.true, df.PR), 3), auc_noalpha=round(roc_auc_score(df.true, df.PR_no_alpha), 3),
    pct_short_offtopic_gt_long_ontopic=None if cross is None else round(100 * cross, 1),
    seg_parse_failures=fails, seg_repaired=repaired, llm=args.llm, emb=args.emb, per_prompt=args.per_prompt))
# ---------- per-essay ranking metrics (alpha cannot change these; see Eq. prdiff) ----------
def per_essay(df, col):
    top1, rr, pair, auc_e = [], [], [], []
    for eid, g in df.groupby('essay_id'):
        t = g[g.true == 1][col]
        if len(t) != 1:
            continue
        tv = t.iloc[0]; w = g[g.true == 0][col].values
        rank = 1 + int((w > tv).sum())
        top1.append(rank == 1); rr.append(1.0 / rank)
        pair.append(float((w < tv).mean())); auc_e.append(float((w < tv).mean() + 0.5 * (w == tv).mean()))
    return dict(top1=round(float(np.mean(top1)), 4), mrr=round(float(np.mean(rr)), 4),
                pairwise=round(float(np.mean(pair)), 4), per_essay_auc=round(float(np.mean(auc_e)), 4))
cols = ['PR', 'PR_no_alpha', 'sim_e', 'mean_sim']
rank_tab = {'pooled': {c: per_essay(df, c) for c in cols}}
for s_ in sorted(df.essay_set.unique()):
    rank_tab[int(s_)] = {c: per_essay(df[df.essay_set == s_], c) for c in cols}
for c in cols:
    rank_tab['pooled'][c]['pooled_auc'] = round(float(roc_auc_score(df.true, df[c])), 4)
summary['per_essay_ranking'] = rank_tab
print('\nPer-essay ranking (pooled):')
for c in cols:
    print(c.ljust(12), rank_tab['pooled'][c])
same = all(rank_tab[k]['PR'][m] == rank_tab[k]['PR_no_alpha'][m] for k in rank_tab for m in ['top1', 'mrr', 'pairwise', 'per_essay_auc'])
print('per-essay metrics identical with and without alpha:', same)
summary['per_essay_identical_with_without_alpha'] = bool(same)
os.makedirs('/kaggle/working/epcts_results', exist_ok=True) if os.path.exists('/kaggle') else None
json.dump(summary, open(args.out + '_summary.json', 'w'), indent=1)
if os.path.exists('/kaggle/working'):
    import shutil
    for f in [args.out + '.csv', args.out + '_segments.csv', args.out + '_summary.json']:
        shutil.copy(f, '/kaggle/working/epcts_results/')
print(pd.DataFrame(summ).T.to_string())
print(json.dumps(summary['pooled']))
log('done')
