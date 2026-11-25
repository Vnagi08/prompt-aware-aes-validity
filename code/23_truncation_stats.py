# 23: how much of each essay the sentence encoder sees (review item 11). CPU only, no model inference.
# all-MiniLM-L6-v2 (SentenceTransformer) truncates every input at 256 word pieces including [CLS]/[SEP];
# the relevance gate embeds at most 40 sentences per essay. Section VI's sim_e embeds the whole essay as one input.
# Essays: ProTACT's preprocessed ASAP files (content_text), the same text the gate embeds.
import os, glob, pickle, subprocess, collections, json
import numpy as np

REPO = '/kaggle/working/ProTACT'
if not os.path.exists(REPO):
    subprocess.run('git clone --depth 1 https://github.com/doheejin/ProTACT.git ' + REPO, shell=True, check=True)
import nltk
for pkg in ['punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
LIMIT, S_MAX = 256, 40

rows = {}
for split in ['train', 'dev', 'test']:
    for r in pickle.load(open(REPO + '/data/cross_prompt_attributes/1/%s.pk' % split, 'rb')):
        p = r.get('prompt_id', r.get('essay_set'))
        rows[str(r['essay_id'])] = (int(p), r['content_text'])
print('essays:', len(rows), flush=True)

S = collections.defaultdict(lambda: collections.Counter())
WP = collections.defaultdict(list)
for eid, (p, text) in rows.items():
    sents = [s for s in nltk.sent_tokenize(text) if s.strip()]
    c = S[p]
    c['essays'] += 1
    c['sents'] += len(sents)
    c['essays_over_40'] += len(sents) > S_MAX
    c['sents_dropped'] += max(0, len(sents) - S_MAX)
    lens = [len(tok(s)['input_ids']) for s in sents[:S_MAX]]
    c['sents_seen'] += len(lens)
    c['sents_truncated'] += sum(l > LIMIT for l in lens)
    whole = len(tok(text)['input_ids'])
    WP[p].append(whole)
    c['essays_whole_over_256'] += whole > LIMIT

out = {}
print('prompt | essays | essays >40 sentences (%) | sentences beyond 40 (%) | seen sentences >256 wp (%) | whole essay >256 wp (%) | median whole-essay wp')
for p in sorted(S):
    c = S[p]
    out[p] = dict(essays=c['essays'],
                  essays_over_40_pct=100 * c['essays_over_40'] / c['essays'],
                  sents_beyond_40_pct=100 * c['sents_dropped'] / max(1, c['sents']),
                  seen_sents_over_256_pct=100 * c['sents_truncated'] / max(1, c['sents_seen']),
                  whole_essay_over_256_pct=100 * c['essays_whole_over_256'] / c['essays'],
                  median_whole_wp=float(np.median(WP[p])))
    o = out[p]
    print('%d | %d | %.1f | %.2f | %.2f | %.1f | %.0f' % (p, o['essays'], o['essays_over_40_pct'], o['sents_beyond_40_pct'],
          o['seen_sents_over_256_pct'], o['whole_essay_over_256_pct'], o['median_whole_wp']), flush=True)
os.makedirs('/kaggle/working/trunc', exist_ok=True)
json.dump(out, open('/kaggle/working/trunc/truncation_stats.json', 'w'), indent=1)
print('DONE')
