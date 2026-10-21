"""Run all experiments reported in the paper; writes results/results.json and LaTeX tables."""
import json, time, sys, numpy as np, stego
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from models import Rules, NgramLR, LangLR, Neural
rows = json.load(open('data/bench.json'))
S = {k: [r for r in rows if r['split'] == k] for k in ('train', 'val', 'test')}
X = lambda R: [r['text'] for r in R]; Y = lambda R: np.array([r['label'] for r in R])
QUICK = '--quick' in sys.argv; EP = 1 if QUICK else 3
METHODS = ['ZW1', 'KHN', 'KSH', 'PSP']
def make(name):
    return {'B1-Rules': Rules, 'B2-CodepointNgramLR': NgramLR, 'B4-LanguageOnlyLR': LangLR,
            'B3-RawCharCNN': lambda: Neural('raw', EP), 'B5-ConcatFusion': lambda: Neural('concat', EP),
            'B6-ASG-noLoc': lambda: Neural('gated', EP, lam_loc=0), 'B6-ArabicStegoGuard': lambda: Neural('gated', EP)}[name]()
NAMES = ['B1-Rules', 'B2-CodepointNgramLR', 'B4-LanguageOnlyLR', 'B3-RawCharCNN', 'B5-ConcatFusion', 'B6-ASG-noLoc', 'B6-ArabicStegoGuard']
def fit(m, R):
    if isinstance(m, Neural): m.fit(X(R), Y(R), [r['offsets'] for r in R]); m.calibrate(X(S['val']), Y(S['val']))
    else: m.fit(X(R), Y(R))
    return m
def thr_at_fpr(sc, y, f=0.01):
    neg = np.sort(sc[y == 0])[::-1]; k = int(np.floor(f * len(neg))); return neg[k] + 1e-12 if k < len(neg) else neg[-1]
def ece(p, y, b=10):
    e = 0
    for lo in np.linspace(0, 1, b, endpoint=False):
        i = (p >= lo) & (p < lo + 1 / b)
        if i.any(): e += i.mean() * abs(p[i].mean() - y[i].mean())
    return e
def boot_auc(sc, y, src, n=300, seed=0):
    rng = np.random.default_rng(seed); us = np.unique(src); g = {u: np.where(src == u)[0] for u in us}; a = []
    for _ in range(n):
        idx = np.concatenate([g[u] for u in rng.choice(us, len(us))])
        if len(set(y[idx])) == 2: a.append(roc_auc_score(y[idx], sc[idx]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
def evaluate(m, R, thr):
    sc = m.score(X(R)); y = Y(R); src = np.array([r['source_id'] for r in R])
    hn = np.array([r['kind'].startswith('hardneg') for r in R])
    out = dict(AUROC=roc_auc_score(y, sc), AUPRC=average_precision_score(y, sc),
               TPR=float((sc[y == 1] >= thr).mean()), FPR=float((sc[y == 0] >= thr).mean()),
               FPR_hardneg=float((sc[hn] >= thr).mean()), FPR_clean=float((sc[(y == 0) & ~hn] >= thr).mean()))
    if isinstance(m, Neural): out.update(ECE=ece(sc, y), Brier=brier_score_loss(y, sc))
    out['AUROC_CI'] = boot_auc(sc, y, src)
    out['TPR_by_method'] = {a: float((sc[np.array([r['method'] == a for r in R])] >= thr).mean()) for a in METHODS}
    return out, sc
import ast, os
DONE = {}
if '--resume' in sys.argv and os.path.exists('results/run_log.txt'):
    for ln in open('results/run_log.txt'):
        p = ln.split(' ', 3)
        if p[0] == 'E1': DONE[('E1', p[1])] = ast.literal_eval(ln.split(' ', 2)[2].replace('np.float64', ''))
        if p[0] == 'E2': DONE[('E2', p[1], p[2])] = ast.literal_eval(p[3])
        if p[0] == 'loc': DONE['loc'] = ast.literal_eval(ln[4:])
res = {'counts': {k: {'n': len(v), 'stego': int(Y(v).sum())} for k, v in S.items()}}
# ---------- E1: source-disjoint, all families seen ----------
res['E1'] = {}; trained = {}
for nm in NAMES:
    if ('E1', nm) in DONE and nm not in ('B1-Rules', 'B2-CodepointNgramLR', 'B6-ArabicStegoGuard'):
        res['E1'][nm] = DONE[('E1', nm)]; continue
    t0 = time.time(); m = fit(make(nm), S['train'])
    thr = thr_at_fpr(m.score(X(S['val'])), Y(S['val']))
    r, sc = evaluate(m, S['test'], thr); r['train_s'] = time.time() - t0
    t1 = time.time(); m.score(X(S['test'])); r['ms_per_doc'] = 1000 * (time.time() - t1) / len(S['test'])
    res['E1'][nm] = r; trained[nm] = (m, thr); print('E1' if '--resume' not in sys.argv else 'E1re', nm, {k: v for k, v in r.items() if k != 'TPR_by_method'}, flush=True)
# localization for B6
m, _ = trained['B6-ArabicStegoGuard']; T = [r for r in S['test'] if r['label'] == 1]
_, loc = m.logits(X(T)); tp = fp = fn = 0
for r, q in zip(T, loc):
    pred = set(np.where(q[:len(r['text'])] >= 0.5)[0]); gold = set(o for o in r['offsets'] if o < 1024)
    tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
P, Rc = tp / max(1, tp + fp), tp / max(1, tp + fn); res['localization'] = dict(P=P, R=Rc, F1=2 * P * Rc / max(1e-9, P + Rc))
print('loc', res['localization'], flush=True)
# ---------- E2: leave-one-embedding-family-out ----------
res['E2'] = {}
for nm in ['B1-Rules', 'B2-CodepointNgramLR', 'B3-RawCharCNN', 'B6-ArabicStegoGuard']:
    res['E2'][nm] = {}
    for a in METHODS:
        if ('E2', nm, a) in DONE: res['E2'][nm][a] = DONE[('E2', nm, a)]; continue
        keep = lambda R: [r for r in R if r['method'] != a]
        tr = keep(S['train']); va = keep(S['val'])
        m = make(nm)
        if isinstance(m, Neural): m.fit(X(tr), Y(tr), [r['offsets'] for r in tr]); m.calibrate(X(va), Y(va))
        else: m.fit(X(tr), Y(tr))
        thr = thr_at_fpr(m.score(X(va)), Y(va))
        te = [r for r in S['test'] if r['method'] in (a, 'none')]; sc = m.score(X(te)); y = Y(te)
        res['E2'][nm][a] = dict(AUROC=roc_auc_score(y, sc), TPR=float((sc[y == 1] >= thr).mean()), FPR=float((sc[y == 0] >= thr).mean()))
        print('E2', nm, a, res['E2'][nm][a], flush=True)
# ---------- E4: transformations: payload survival and detection on transformed test set ----------
res['PSR'] = {}; res['E4'] = {}
for tn, f in stego.TRANSFORMS.items():
    res['PSR'][tn] = {a: float(np.mean([stego.decode(a, f(r['text']), len(r['bits'])) == r['bits'] for r in S['test'] if r['method'] == a])) for a in METHODS}
    R2 = [dict(r, text=f(r['text'])) for r in S['test']]
    # active label: payload survives
    for r in R2:
        if r['label'] == 1: r['label'] = int(stego.decode(r['method'], r['text'], len(r['bits'])) == r['bits'])
    res['E4'][tn] = {'n_active': int(Y(R2).sum())}
    for nm in ['B1-Rules', 'B2-CodepointNgramLR', 'B6-ArabicStegoGuard']:
        m, thr = trained[nm]; sc = m.score(X(R2)); y = Y(R2)
        res['E4'][tn][nm] = dict(AUROC=roc_auc_score(y, sc) if 0 < y.sum() < len(y) else None,
                                 TPR=float((sc[y == 1] >= thr).mean()) if y.sum() else None, FPR=float((sc[y == 0] >= thr).mean()))
    print('E4', tn, res['PSR'][tn], res['E4'][tn], flush=True)
import os; os.makedirs('results', exist_ok=True)
json.dump(res, open('results/results%s.json' % ('_quick' if QUICK else ''), 'w'), indent=1, default=float)
print('done')
