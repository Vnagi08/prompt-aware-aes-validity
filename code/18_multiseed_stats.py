import re, numpy as np
from collections import defaultdict
R=defaultdict(dict)
for l in open('results/multiseed_stage4_raw.txt'):
    m=re.match(r's(\d+)p(\d) g([\d.]+) se([-\d.]+)/(\S+) para(\S+) nm([-\d.]+)\(([\d.]+)\) un(\S+) sp(\S+)',l)
    s,p=int(m[1]),int(m[2]); f=lambda x:[float(v) for v in x.split('/')]
    R[p][s]=dict(g=float(m[3]),se=float(m[4]),para=np.mean(f(m[6])),nm=float(m[7]),nmp=float(m[8]),un=np.mean(f(m[9])),sp1=f(m[10])[3],sp50=f(m[10])[1])
ms=lambda a:'%.3f ± %.3f'%(np.mean(a),np.std(a,ddof=1)) if len(a)>1 else '%.3f'%a[0]
print('| Prompt | n | gate g | paraphrase Δ | near-miss Δ (% lower) | unseen Δ | splice 50% | splice 100% | signed err (Content) |'); print('|'+'---|'*9)
for p in range(1,9):
    d=R[p]; k=lambda q:[d[s][q] for s in d]
    print('| %d | %d | %s | %s | %s (%.1f) | %s | %s | %s | %s |'%(p,len(d),ms(k('g')),ms(k('para')),ms(k('nm')),np.mean(k('nmp')),ms(k('un')),ms(k('sp50')),ms(k('sp1')),ms(k('se'))))
# QWK
B={12:[.658,.560,.578,.541,.625,.557,.383,.403],22:[.655,.564,.577,.542,.627,.564,.391,.407],32:[.640,.553,.570,.535,.633,.559,.368,.435],42:[.637,.563,.573,.537,.640,.561,.376,.458],52:[.648,.562,.566,.539,.638,.565,.355,.455]}
G={12:[.654,.556,.552,.524,.602,.551,.319,.407],22:[.639,.548,.546,.523,.612,.550,.325,.442],32:[.646,.552,.567,.528,.612,.541,.324,.435],42:[.653,.549,.555,.524,.618,.542,.332,.444],52:[.643,.545,.541,.528,.611,.552,.323,.448]}
b=np.array([B[s] for s in sorted(B)]);g=np.array([G[s] for s in sorted(G)]);D=g-b
rng=np.random.default_rng(0)
def boot(x,n=10000):
    i=rng.integers(0,len(x),(n,len(x))); m=x[i].mean(1); return np.percentile(m,[2.5,97.5])
print('\nAvg QWK, gate - baseline, paired over seeds (95% bootstrap CI over 5 seeds):')
for p in range(8):
    lo,hi=boot(D[:,p]); print('p%d %+.3f [%+.3f, %+.3f]'%(p+1,D[:,p].mean(),lo,hi))
lo,hi=boot(D.mean(1)); print('all %+.3f [%+.3f, %+.3f]  (seed-level means)'%(D.mean(),lo,hi))
print('prompts 1-6 only: %+.3f'%D[:,:6].mean())
print()
for p in range(1,9):
    d=R[p]; 
    import re as _r
L={}
for l in open('results/multiseed_stage4_raw.txt'):
    m=re.match(r's(\d+)p(\d) .* para(\S+) .* un(\S+) sp(\S+)',l); L.setdefault(int(m[2]),[]).append(([float(x) for x in m[3].split('/')],[float(x) for x in m[4].split('/')],[float(x) for x in m[5].split('/')]))
for p in sorted(L):
    pa=np.mean([x[0] for x in L[p]],0); un=np.mean([x[1] for x in L[p]],0); sp=np.mean([x[2] for x in L[p]],0)
    print(p, 'para',' '.join('%+.2f'%v for v in pa),'| unseen %.2f to %.2f'%(un.max(),un.min()),'| splice',' / '.join('%+.2f'%v for v in sp))
