"""Detectors: B1 rules, B2 code-point n-gram LR, B4 language-only LR, B3 raw char-CNN, B6 dual-stream gated."""
import numpy as np, torch, torch.nn as nn, unicodedata, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import stego
SUSP = set([stego.ZWNJ, stego.ZWJ, stego.ZWSP, stego.WJ, stego.TATWEEL, stego.THIN])

class Rules:   # B1: count of suspicious code points per 100 chars
    def fit(self, X, y): return self
    def score(self, X): return np.array([100*sum(ch in SUSP for ch in x)/max(1, len(x)) for x in X])

class NgramLR:  # B2: raw code-point 1-3 grams
    def __init__(self): self.v = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False, min_df=2, sublinear_tf=True, max_features=200000)
    def fit(self, X, y): self.m = LogisticRegression(C=10, max_iter=3000).fit(self.v.fit_transform(X), y); return self
    def score(self, X): return self.m.predict_proba(self.v.transform(X))[:, 1]

def ling_view(x):  # typical Arabic NLP preprocessing: NFKC, drop Cf, drop tatweel/diacritics, collapse spaces
    x = unicodedata.normalize('NFKC', x); x = stego.strip_format(x)
    x = re.sub(r'[\u0640\u064B-\u0652]', '', x); return re.sub(r'\s+', ' ', x)

class LangLR:   # B4: language-only (word uni/bigram on normalized view)
    def __init__(self): self.v = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    def fit(self, X, y): self.m = LogisticRegression(C=10, max_iter=3000).fit(self.v.fit_transform([ling_view(x) for x in X]), y); return self
    def score(self, X): return self.m.predict_proba(self.v.transform([ling_view(x) for x in X]))[:, 1]

# ---------- neural ----------
MAXLEN = 1024
def cp_id(ch):  # hashed code-point id (lossless for BMP Arabic/general punct ranges)
    return (ord(ch) % 4093) + 1
CATS = ['Lo', 'Mn', 'Cf', 'Zs', 'Nd', 'Po', 'Ll', 'Lu', 'Lm', 'Cc', 'Sm', 'Ps', 'Pe', 'Pd', 'So']
def cat_id(ch):
    c = unicodedata.category(ch); return CATS.index(c) + 1 if c in CATS else len(CATS) + 1
def bidi_id(ch):
    B = ['AL', 'R', 'L', 'NSM', 'BN', 'WS', 'EN', 'AN', 'ON', 'CS', 'ES', 'ET', 'B', 'S', 'LRM', 'RLE']
    b = unicodedata.bidirectional(ch); return B.index(b) + 1 if b in B else len(B) + 1

def encode(X, offsets=None):
    n = len(X); C = np.zeros((n, MAXLEN), np.int64); K = np.zeros_like(C); D = np.zeros_like(C)
    L = np.zeros((n, MAXLEN), np.float32); M = np.zeros((n, MAXLEN), np.float32)
    for i, x in enumerate(X):
        x = x[:MAXLEN]
        C[i, :len(x)] = [cp_id(c) for c in x]; K[i, :len(x)] = [cat_id(c) for c in x]; D[i, :len(x)] = [bidi_id(c) for c in x]
        M[i, :len(x)] = 1
        if offsets is not None:
            for o in offsets[i]:
                if o < MAXLEN: L[i, o] = 1
    return [torch.tensor(a) for a in (C, K, D, L, M)]

class CharEnc(nn.Module):
    def __init__(self, d=64, feats=True):
        super().__init__(); self.feats = feats
        self.e = nn.Embedding(4100, d, padding_idx=0); self.k = nn.Embedding(20, 16); self.b = nn.Embedding(20, 16)
        din = d + (32 if feats else 0)
        self.conv = nn.Sequential(nn.Conv1d(din, 128, 5, padding=2), nn.ReLU(), nn.Conv1d(128, 128, 5, padding=4, dilation=2), nn.ReLU())
    def forward(self, C, K, D):
        h = self.e(C)
        if self.feats: h = torch.cat([h, self.k(K), self.b(D)], -1)
        return self.conv(h.transpose(1, 2)).transpose(1, 2)  # B x T x 128

class Net(nn.Module):
    """mode='raw' (B3): raw char-CNN only. mode='gated' (B6): raw stream + linguistic-view stream, gated fusion, localization head.
    mode='concat' (B5): same two streams, concatenation fusion."""
    def __init__(self, mode='gated'):
        super().__init__(); self.mode = mode
        self.U = CharEnc(feats=(mode != 'raw'))
        if mode != 'raw':
            self.Lg = CharEnc(feats=False)
            self.g = nn.Linear(256, 128); self.WU = nn.Linear(128, 128); self.WL = nn.Linear(128, 128)
            if mode == 'concat': self.cat = nn.Linear(256, 128)
        self.att = nn.Linear(128, 1); self.doc = nn.Linear(128, 1); self.loc = nn.Linear(128, 1)
    def forward(self, C, K, D, M, CL=None):
        hU = self.U(C, K, D)
        if self.mode == 'raw': h = hU
        else:
            hL = self.Lg(CL, CL * 0, CL * 0)  # normalized view, character-aligned (deleted chars -> padding id)
            if self.mode == 'gated':
                a = torch.sigmoid(self.g(torch.cat([hU, hL], -1))); h = a * self.WU(hU) + (1 - a) * self.WL(hL)
            else: h = torch.relu(self.cat(torch.cat([hU, hL], -1)))
        w = self.att(h).squeeze(-1).masked_fill(M == 0, -1e9); w = torch.softmax(w, 1)
        v = (w.unsqueeze(-1) * h).sum(1)
        return self.doc(v).squeeze(-1), self.loc(h).squeeze(-1)

def aligned_ling_ids(X):
    """Linguistic view aligned to raw positions: each raw char maps to its normalized char id, or 0 if deleted."""
    out = np.zeros((len(X), MAXLEN), np.int64)
    for i, x in enumerate(X):
        for j, ch in enumerate(x[:MAXLEN]):
            n = ling_view(ch) if ch != ' ' else ' '
            out[i, j] = cp_id(n[0]) if n else 0
    return torch.tensor(out)

class Neural:
    def __init__(self, mode='gated', epochs=4, lam_loc=1.0, seed=0):
        self.mode, self.epochs, self.lam, self.seed = mode, epochs, lam_loc, seed
    def fit(self, X, y, offsets=None):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        C, K, D, L, M = encode(X, offsets if offsets is not None else [[]] * len(X)); CL = aligned_ling_ids(X)
        y = torch.tensor(y, dtype=torch.float32); self.net = Net(self.mode)
        opt = torch.optim.Adam(self.net.parameters(), 2e-3); bce = nn.BCEWithLogitsLoss(reduction='none')
        for ep in range(self.epochs):
            perm = torch.randperm(len(X)); self.net.train()
            for b in range(0, len(X), 64):
                idx = perm[b:b+64]
                pd_, pl = self.net(C[idx], K[idx], D[idx], M[idx], CL[idx])
                loss = bce(pd_, y[idx]).mean()
                if self.mode != 'raw' and self.lam > 0 and offsets is not None:
                    lm = bce(pl, L[idx]) * M[idx]; pw = 1 + 20 * L[idx]
                    loss = loss + self.lam * (lm * pw).sum() / M[idx].sum()
                opt.zero_grad(); loss.backward(); opt.step()
        self.T = 1.0; return self
    def logits(self, X):
        C, K, D, L, M = encode(X); CL = aligned_ling_ids(X); self.net.eval(); outs, locs = [], []
        with torch.no_grad():
            for b in range(0, len(X), 256):
                d, l = self.net(C[b:b+256], K[b:b+256], D[b:b+256], M[b:b+256], CL[b:b+256]); outs.append(d); locs.append(torch.sigmoid(l) * M[b:b+256])
        return torch.cat(outs).numpy(), torch.cat(locs).numpy()
    def calibrate(self, X, y):  # temperature scaling on validation
        z, _ = self.logits(X); z = torch.tensor(z); yt = torch.tensor(y, dtype=torch.float32)
        best = min(np.linspace(0.25, 8, 64), key=lambda T: nn.functional.binary_cross_entropy_with_logits(z / T, yt).item()); self.T = float(best)
    def score(self, X):
        z, _ = self.logits(X); return 1 / (1 + np.exp(-z / self.T))
