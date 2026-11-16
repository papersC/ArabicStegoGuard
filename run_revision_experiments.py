"""Revision experiments for TIFS reviewer comments. Checkpoints after each section."""
import json, time, os, re, random
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import stego
from models import Rules, NgramLR, Neural

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
os.makedirs('results', exist_ok=True)
OUT = 'results/revision_results.json'
rows = json.load(open('data/bench.json'))
S = {k: [r for r in rows if r['split'] == k] for k in ('train', 'val', 'test')}
X = lambda R: [r['text'] for r in R]
Y = lambda R: np.array([r['label'] for r in R])
METHODS = ['ZW1', 'KHN', 'KSH', 'PSP']
EP = 3
SEEDS = [0, 1, 2]

def save(res):
    json.dump(res, open(OUT, 'w'), indent=1, default=float)

def thr_at_fpr(sc, y, f=0.01):
    neg = np.sort(sc[y == 0])[::-1]
    k = int(np.floor(f * len(neg)))
    return neg[k] + 1e-12 if k < len(neg) else neg[-1]

def thr_at_tpr(sc, y, t=0.99):
    pos = np.sort(sc[y == 1])
    k = int(np.floor((1 - t) * len(pos)))
    return pos[k] - 1e-12 if k < len(pos) else pos[0]

def hardneg_mask(R):
    return np.array([str(r.get('kind','')).startswith('hardneg') for r in R])

def metrics_at(sc, y, thr, hn):
    return dict(
        TPR=float((sc[y == 1] >= thr).mean()) if y.sum() else None,
        FPR=float((sc[y == 0] >= thr).mean()) if (y == 0).sum() else None,
        FPR_hardneg=float((sc[hn] >= thr).mean()) if hn.any() else None,
        FPR_clean=float((sc[(y == 0) & ~hn] >= thr).mean()) if ((y == 0) & ~hn).any() else None,
        AUROC=float(roc_auc_score(y, sc)) if 0 < y.sum() < len(y) else None,
        AUPRC=float(average_precision_score(y, sc)) if 0 < y.sum() < len(y) else None,
    )

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res['meta'] = {'seeds': SEEDS, 'epochs': EP, 'note': 'revision experiments; measured only'}

# ---- 1 Multi-seed E1 ----
if 'E1_seeds' not in res or len(res.get('E1_seeds', {}).get('B6-ArabicStegoGuard', [])) < 3:
    print('=== Multi-seed E1 ===', flush=True)
    res['E1_seeds'] = res.get('E1_seeds', {})
    models_seed0 = {}
    for mode, name, lam in [
        ('raw', 'B3-RawCharCNN', 0.0),
        ('concat', 'B5-ConcatFusion', 1.0),
        ('gated', 'B6-ArabicStegoGuard', 1.0),
    ]:
        res['E1_seeds'].setdefault(name, [])
        done_seeds = {r['seed'] for r in res['E1_seeds'][name]}
        for seed in SEEDS:
            if seed in done_seeds and not (mode == 'gated' and seed == 0 and 'model_needed' ):
                # still need seed0 models in memory — retrain seed0 always if missing models
                if seed != 0:
                    print('skip', name, seed, flush=True); continue
            t0 = time.time()
            m = Neural(mode, EP, lam_loc=lam, seed=seed)
            m.fit(X(S['train']), Y(S['train']), [r['offsets'] for r in S['train']])
            m.calibrate(X(S['val']), Y(S['val']))
            sc_va = m.score(X(S['val'])); y_va = Y(S['val'])
            sc_te = m.score(X(S['test'])); y_te = Y(S['test']); hn = hardneg_mask(S['test'])
            thr_fpr1 = thr_at_fpr(sc_va, y_va, 0.01)
            thr_tpr99 = thr_at_tpr(sc_va, y_va, 0.99)
            out = {
                'seed': seed,
                'at_FPR1': metrics_at(sc_te, y_te, thr_fpr1, hn),
                'at_TPR99': metrics_at(sc_te, y_te, thr_tpr99, hn),
                'train_s': time.time() - t0,
            }
            if mode == 'gated' and seed == 0:
                T = [r for r in S['test'] if r['label'] == 1]
                _, loc = m.logits(X(T)); tp = fp = fn = 0
                for r, q in zip(T, loc):
                    pred = set(np.where(q[:len(r['text'])] >= 0.5)[0])
                    gold = set(o for o in r['offsets'] if o < 1024)
                    tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
                P, Rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
                out['localization'] = dict(P=P, R=Rc, F1=2 * P * Rc / max(1e-9, P + Rc))
            # replace if seed exists
            res['E1_seeds'][name] = [r for r in res['E1_seeds'][name] if r['seed'] != seed] + [out]
            if seed == 0:
                models_seed0[name] = (m, thr_fpr1)
            print(name, 'seed', seed, out['at_FPR1'], flush=True)
            save(res)
    # keep models for later sections
    res['_have_seed0_models'] = True
else:
    print('E1_seeds already complete; retraining seed0 models for downstream', flush=True)
    models_seed0 = {}
    for mode, name, lam in [('raw','B3-RawCharCNN',0.0),('concat','B5-ConcatFusion',1.0),('gated','B6-ArabicStegoGuard',1.0)]:
        m = Neural(mode, EP, lam_loc=lam, seed=0)
        m.fit(X(S['train']), Y(S['train']), [r['offsets'] for r in S['train']])
        m.calibrate(X(S['val']), Y(S['val']))
        models_seed0[name] = (m, thr_at_fpr(m.score(X(S['val'])), Y(S['val'])))

res['E1_agg'] = {}
for name, runs in res['E1_seeds'].items():
    agg = {}
    for key in ('at_FPR1', 'at_TPR99'):
        agg[key] = {}
        for metric in ('TPR', 'FPR', 'FPR_hardneg', 'FPR_clean', 'AUROC'):
            vals = [r[key][metric] for r in runs if r[key].get(metric) is not None]
            agg[key][metric] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), values=[float(v) for v in vals])
    res['E1_agg'][name] = agg
save(res)

# ---- matched OP ----
print('=== Matched OP ===', flush=True)
res['matched_op'] = {}
targets = [0.98, 0.99, 0.995]
hn = hardneg_mask(S['test']); y_te = Y(S['test'])
b1 = Rules().fit(X(S['train']), Y(S['train']))
b2 = NgramLR().fit(X(S['train']), Y(S['train']))
all_models = {
    'B1-Rules': (b1, thr_at_fpr(b1.score(X(S['val'])), Y(S['val']))),
    'B2-CodepointNgramLR': (b2, thr_at_fpr(b2.score(X(S['val'])), Y(S['val']))),
}
all_models.update(models_seed0)
for name, (m, _) in all_models.items():
    sc_va = m.score(X(S['val'])); y_va = Y(S['val']); sc_te = m.score(X(S['test']))
    res['matched_op'][name] = {}
    for t in targets:
        thr = thr_at_tpr(sc_va, y_va, t)
        res['matched_op'][name][f'TPR>={t}'] = metrics_at(sc_te, y_te, thr, hn)
        print(name, t, res['matched_op'][name][f'TPR>={t}'], flush=True)
save(res)

# ---- expanded hardneg ----
print('=== Expanded hardneg ===', flush=True)
rng = random.Random(4242)
covers_all, seen = [], set()
for r in rows:
    c = r.get('cover') or (r['text'] if r['kind'] == 'clean' else None)
    if c and c not in seen:
        seen.add(c); covers_all.append((c, r['source_id']))
test_srcs = {r['source_id'] for r in S['test']}
expanded = []
for c, sid in covers_all:
    if sid in test_srcs:
        s, k = stego.hard_negative_expanded(c, rng)
        expanded.append(dict(text=s, label=0, kind='hardneg_'+k, method='none', source_id=sid))
while len(expanded) < 1500:
    c, sid = rng.choice([(c,s) for c,s in covers_all if s in test_srcs])
    s, k = stego.hard_negative_expanded(c, rng)
    expanded.append(dict(text=s, label=0, kind='hardneg_'+k, method='none', source_id=sid))
res['expanded_hardneg'] = {'n': len(expanded), 'kinds': dict(Counter(r['kind'] for r in expanded)), 'FPR': {}}
for name, (m, thr) in all_models.items():
    sc = m.score(X(expanded)); src = np.array([r['source_id'] for r in expanded])
    fpr = float((sc >= thr).mean())
    rngb = np.random.default_rng(0); us = np.unique(src); g = {u: np.where(src==u)[0] for u in us}
    boots = [float((sc[np.concatenate([g[u] for u in rngb.choice(us, len(us))])] >= thr).mean()) for _ in range(300)]
    res['expanded_hardneg']['FPR'][name] = dict(FPR=fpr, CI95=[float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))], n_flagged=int((sc>=thr).sum()))
    print(name, res['expanded_hardneg']['FPR'][name], flush=True)
save(res)

# ---- domain news ----
print('=== Domain news ===', flush=True)
import pyarrow.parquet as pq
texts = pq.read_table('data/arabic_bbc_news.parquet').column('text').to_pylist()
AR = re.compile(r'[\u0621-\u064A]')
def paras(text, lo=200, hi=800):
    out=[]
    for p in re.split(r'\n+', text or ''):
        p=p.strip()
        if lo<=len(p)<=hi and len(AR.findall(p))/max(1,len(p))>0.6: out.append(p)
    return out
news_paras=[]
for t in texts:
    news_paras.extend(paras(t))
    if len(news_paras)>=2500: break
rngn=random.Random(2026); rngn.shuffle(news_paras); news_paras=news_paras[:2000]
news_rows=[]
for i,t in enumerate(news_paras):
    r=rngn.random(); sid=f'news_{i}'
    if r<0.4:
        news_rows.append(dict(text=t,label=0,kind='clean',method='none',source_id=sid,offsets=[],bits=[]))
    elif r<0.6:
        s,k=stego.hard_negative_expanded(t,rngn)
        news_rows.append(dict(text=s,label=0,kind='hardneg_'+k,method='none',source_id=sid,offsets=[],bits=[]))
    else:
        mth=rngn.choice(METHODS)
        nb=min(rngn.choice([16,32,64]), stego.capacity(mth,t)-(stego.capacity(mth,t)%2))
        if nb<8:
            news_rows.append(dict(text=t,label=0,kind='clean',method='none',source_id=sid,offsets=[],bits=[]))
        else:
            bits=[rngn.randint(0,1) for _ in range(nb)]; s,offs=stego.embed(mth,t,bits)
            news_rows.append(dict(text=s,label=1,kind='stego',method=mth,source_id=sid,offsets=offs,bits=bits))
json.dump(news_rows, open('data/news_domain_bench.json','w'), ensure_ascii=False)
res['domain_news']={'n':len(news_rows),'stego':int(sum(r['label'] for r in news_rows)),
  'kinds':dict(Counter(r['kind'] if r['label']==0 else r['method'] for r in news_rows)), 'zero_shot':{}}
yn=Y(news_rows); hn_n=hardneg_mask(news_rows)
for name,(m,thr) in all_models.items():
    sc=m.score(X(news_rows))
    out=metrics_at(sc,yn,thr,hn_n)
    out['TPR_by_method']={a:float((sc[np.array([r['method']==a for r in news_rows])]>=thr).mean()) for a in METHODS}
    res['domain_news']['zero_shot'][name]=out
    print('news',name,out,flush=True)
save(res)

# ---- E2 remedies (only KSH and PSP are critical; still do all 4) ----
print('=== E2 remedies ===', flush=True)
feat=lambda R: np.array([stego.anomaly_features(r['text']) for r in R], dtype=np.float32)
res['E2_remedies']=res.get('E2_remedies',{})
for held in METHODS:
    if held in res['E2_remedies'] and 'Hybrid-B6-AnomalyLR' in res['E2_remedies'][held]:
        print('skip E2', held, flush=True); continue
    keep=lambda R:[r for r in R if r['method']!=held]
    tr,va=keep(S['train']),keep(S['val'])
    te=[r for r in S['test'] if r['method'] in (held,'none')]
    sc_tr=StandardScaler().fit(feat(tr))
    clf=LogisticRegression(C=1,max_iter=2000).fit(sc_tr.transform(feat(tr)), Y(tr))
    iso_h=IsolationForest(n_estimators=200,contamination=0.02,random_state=0)
    iso_h.fit(sc_tr.transform(feat([r for r in tr if r['label']==0])))
    m6=Neural('gated',EP,lam_loc=1.0,seed=0)
    m6.fit(X(tr),Y(tr),[r['offsets'] for r in tr]); m6.calibrate(X(va),Y(va))
    thr6=thr_at_fpr(m6.score(X(va)),Y(va))
    sc_va_a=clf.predict_proba(sc_tr.transform(feat(va)))[:,1]
    thr_a=thr_at_fpr(sc_va_a,Y(va))
    sc_va_i=-iso_h.decision_function(sc_tr.transform(feat(va)))
    thr_i=thr_at_fpr(sc_va_i,Y(va))
    sc6_va=m6.score(X(va)); sc6_te=m6.score(X(te))
    sc_a_te=clf.predict_proba(sc_tr.transform(feat(te)))[:,1]
    sc_i_te=-iso_h.decision_function(sc_tr.transform(feat(te)))
    def norm(sc,lo,hi): return (sc-lo)/(hi-lo+1e-12)
    def combine(sc_m,sc_a,va_m,va_a):
        return np.maximum(norm(sc_m,*np.percentile(va_m,[1,99])), norm(sc_a,*np.percentile(va_a,[1,99])))
    sc_h_va=combine(sc6_va,sc_va_a,sc6_va,sc_va_a); thr_h=thr_at_fpr(sc_h_va,Y(va))
    sc_h_te=combine(sc6_te,sc_a_te,sc6_va,sc_va_a)
    yte=Y(te)
    def pack(sc,thr):
        return dict(AUROC=float(roc_auc_score(yte,sc)), TPR=float((sc[yte==1]>=thr).mean()), FPR=float((sc[yte==0]>=thr).mean()))
    res['E2_remedies'][held]={
        'B6-LOFO':pack(sc6_te,thr6),
        'AnomalyLR':pack(sc_a_te,thr_a),
        'IsolationForest':pack(sc_i_te,thr_i),
        'Hybrid-B6-AnomalyLR':pack(sc_h_te,thr_h),
        'B1-Rules':pack(b1.score(X(te)), thr_at_fpr(b1.score(X(S['val'])),Y(S['val']))),
    }
    print('E2',held,res['E2_remedies'][held],flush=True); save(res)

# ---- consistency aug (NFC duplicates; not full adversarial training) ----
print('=== Consistency aug ===', flush=True)
aug=[]
for r in S['train']:
    aug.append(r)
    t2=stego.TRANSFORMS['NFC'](r['text'])
    if r['label']==1:
        if stego.decode(r['method'],t2,len(r['bits']))==r['bits']:
            aug.append(dict(r,text=t2))
    else:
        aug.append(dict(r,text=t2))
m_c=Neural('gated',EP,lam_loc=1.0,seed=0)
m_c.fit(X(aug),Y(aug),[r['offsets'] for r in aug]); m_c.calibrate(X(S['val']),Y(S['val']))
thr_c=thr_at_fpr(m_c.score(X(S['val'])),Y(S['val']))
res['consistency_aug']={'train_n':len(aug),'note':'NFC label-preserving augmentation only; not full adversarial training (H3)',
  'E1':metrics_at(m_c.score(X(S['test'])),Y(S['test']),thr_c,hardneg_mask(S['test'])),'transforms':{}}
for tn,f in stego.TRANSFORMS.items():
    R2=[dict(r,text=f(r['text'])) for r in S['test']]
    for r in R2:
        if r['label']==1:
            r['label']=int(stego.decode(r['method'],r['text'],len(r['bits']))==r['bits'])
    res['consistency_aug']['transforms'][tn]=metrics_at(m_c.score(X(R2)),Y(R2),thr_c,hardneg_mask(R2))
print('consistency',res['consistency_aug']['E1'],flush=True)
save(res)
print('DONE', flush=True)
