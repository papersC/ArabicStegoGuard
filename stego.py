"""Embedding families, decoders, benign hard negatives and transformations."""
import random, re, unicodedata
AR = re.compile(r'[\u0621-\u063F\u0641-\u064A]')
# letters that do not join to the following letter
NONJOIN = set('\u0621\u0622\u0623\u0624\u0625\u0627\u062F\u0630\u0631\u0632\u0648\u0629\u0649')
ZWNJ, ZWJ, ZWSP, WJ, TATWEEL, THIN = '\u200C', '\u200D', '\u200B', '\u2060', '\u0640', '\u2009'
KHN_SYMS = [ZWSP, ZWNJ, ZWJ, WJ]          # 2 bits per symbol
INVIS = set(KHN_SYMS)

def letter_positions(t):  # indices i such that t[i] is Arabic letter
    return [i for i, ch in enumerate(t) if AR.match(ch)]

def kashida_positions(t):  # between two Arabic letters where the first joins forward
    return [i for i in range(len(t) - 1) if AR.match(t[i]) and AR.match(t[i+1]) and t[i] not in NONJOIN]

def capacity(method, t):
    if method == 'ZW1': return len(letter_positions(t))
    if method == 'KHN': return 2 * len(letter_positions(t))
    if method == 'KSH': return len(kashida_positions(t))
    if method == 'PSP': return t.count(' ')

def _insert(t, pos_sym):
    """pos_sym: list of (index, symbol) inserted after t[index]. Returns text and new offsets."""
    out, offs, d = [], [], dict(pos_sym)
    for i, ch in enumerate(t):
        out.append(ch)
        if i in d and d[i]:
            offs.append(sum(len(x) for x in out)); out.append(d[i])
    return ''.join(out), offs

def embed(method, t, bits):
    """Returns stego text and list of inserted/modified character offsets (in stego text)."""
    if method == 'ZW1':  # Alshamsi-style non-printable: ZWJ=1, ZWNJ=0 after successive letters
        P = letter_positions(t)[:len(bits)]
        return _insert(t, [(p, ZWJ if b else ZWNJ) for p, b in zip(P, bits)])
    if method == 'KHN':  # Khan-style: one of four invisible characters = 2 bits per letter
        P = letter_positions(t)
        pairs = [bits[i:i+2] for i in range(0, len(bits), 2)]
        return _insert(t, [(p, KHN_SYMS[2*a + b]) for p, (a, b) in zip(P, pairs)])
    if method == 'KSH':  # Kashida: tatweel present at eligible position = 1, absent = 0
        t = t.replace(TATWEEL, '')  # encoder first removes any pre-existing elongation
        P = kashida_positions(t)[:len(bits)]
        return _insert(t, [(p, TATWEEL if b else '') for p, b in zip(P, bits)])
    if method == 'PSP':  # pseudo-space: thin space = 1, normal space = 0
        out, offs, k = list(t), [], 0
        for i, ch in enumerate(t):
            if ch == ' ' and k < len(bits):
                if bits[k]: out[i] = THIN; offs.append(i)
                k += 1
        return ''.join(out), offs

def decode(method, s, nbits):
    if method == 'ZW1':
        b = [1 if ch == ZWJ else 0 for ch in s if ch in (ZWJ, ZWNJ)]
        return b[:nbits] if len(b) >= nbits else None
    if method == 'KHN':
        b = []
        for ch in s:
            if ch in INVIS: k = KHN_SYMS.index(ch); b += [k >> 1, k & 1]
        return b[:nbits] if len(b) >= nbits else None
    if method == 'KSH':
        b, i, letters = [], 0, s
        # walk eligible positions on the tatweel-stripped view
        clean, flags = [], []
        for ch in s:
            if ch == TATWEEL:
                if flags: flags[-1] = 1
            else: clean.append(ch); flags.append(0)
        clean = ''.join(clean)
        b = [flags[p] for p in kashida_positions(clean)]
        return b[:nbits] if len(b) >= nbits else None
    if method == 'PSP':
        b = [1 if ch == THIN else 0 for ch in s if ch in (' ', THIN)]
        return b[:nbits] if len(b) >= nbits else None

def hard_negative(t, rng):
    """Benign, non-payload Unicode phenomena found in legitimate Arabic text."""
    kind = rng.choice(['zwnj_compound', 'decor_tatweel', 'bidi_marks', 'nbsp_numbers'])
    words = t.split(' ')
    if kind == 'zwnj_compound':   # isolated ZWNJ inside a few words (Persian-style compounds / abbreviations)
        for j in rng.sample(range(len(words)), k=min(2, len(words))):
            w = words[j]
            if len(w) > 3: words[j] = w[:2] + ZWNJ + w[2:]
        return ' '.join(words), kind
    if kind == 'decor_tatweel':   # decorative elongation of one or two words
        for j in rng.sample(range(len(words)), k=min(2, len(words))):
            w = words[j]
            if len(w) > 2 and AR.match(w[0]) and w[0] not in NONJOIN and AR.match(w[1]):
                words[j] = w[0] + TATWEEL * rng.randint(1, 3) + w[1:]
        return ' '.join(words), kind
    if kind == 'bidi_marks':      # RLM/LRM around Latin/digit runs, as inserted by editors
        s = re.sub(r'([A-Za-z0-9][A-Za-z0-9 .\-]*)', '\u200E\\1\u200F', t)
        if s == t: s = '\u200F' + t
        return s, kind
    s = re.sub(r'(\d) ', '\\1\u00A0', t)  # NBSP after numbers
    s = re.sub(r' (\d)', '\u00A0\\1', s)
    if s == t: s = t.replace(' ', '\u00A0', 1)
    return s, kind

def strip_format(s):  # remove Unicode Cf characters (common sanitizer)
    return ''.join(ch for ch in s if unicodedata.category(ch) != 'Cf')

TRANSFORMS = {
    'identity': lambda s: s,
    'NFC': lambda s: unicodedata.normalize('NFC', s),
    'NFKC': lambda s: unicodedata.normalize('NFKC', s),
    'strip_Cf': strip_format,
    'ws_collapse': lambda s: re.sub(r'[\s\u00A0\u2000-\u200A]+', ' ', s),
    'strip_tatweel': lambda s: s.replace(TATWEEL, ''),
}

# ---- Expanded curated benign patterns (best-effort; not linguist-annotated) ----
SOFT_HYPHEN = '\u00AD'
ZWSP_LOCAL = ZWSP

def hard_negative_expanded(t, rng):
    """Broader surrogate legitimate Unicode uses for FP evaluation.
    Documented as curated best-effort, not Arabic-specialist reviewed."""
    kind = rng.choice([
        'zwnj_compound', 'decor_tatweel', 'bidi_marks', 'nbsp_numbers',
        'soft_hyphen', 'zwj_ligature_hint', 'mixed_script_bidi',
        'multi_tatweel', 'thin_nbsp_mix', 'rlm_quote'
    ])
    words = t.split(' ')
    if kind in ('zwnj_compound', 'decor_tatweel', 'bidi_marks', 'nbsp_numbers'):
        return hard_negative(t, rng)[0], kind
    if kind == 'soft_hyphen':
        for j in rng.sample(range(len(words)), k=min(3, len(words))):
            w = words[j]
            if len(w) > 4:
                words[j] = w[:len(w)//2] + SOFT_HYPHEN + w[len(w)//2:]
        return ' '.join(words), kind
    if kind == 'zwj_ligature_hint':
        # ZWJ between Arabic letters as sometimes used for presentation control
        for j in rng.sample(range(len(words)), k=min(2, len(words))):
            w = words[j]
            if len(w) > 3 and AR.match(w[1]) and AR.match(w[2]):
                words[j] = w[:2] + ZWJ + w[2:]
        return ' '.join(words), kind
    if kind == 'mixed_script_bidi':
        s = re.sub(r'([A-Za-z]+)', '\u202B\\1\u202C', t)  # RLE/PDF around Latin
        if s == t:
            s = '\u200F' + t + ' (OK)'
        return s, kind
    if kind == 'multi_tatweel':
        for j in rng.sample(range(len(words)), k=min(3, len(words))):
            w = words[j]
            if len(w) > 3 and AR.match(w[0]) and w[0] not in NONJOIN and AR.match(w[1]):
                words[j] = w[0] + TATWEEL * rng.randint(2, 5) + w[1:]
        return ' '.join(words), kind
    if kind == 'thin_nbsp_mix':
        s = t.replace(' ', THIN, 1) if ' ' in t else t + THIN
        s = re.sub(r'(\d)', '\\1\u00A0', s, count=2)
        return s, kind
    # rlm_quote
    return '\u200F«' + t[:80] + '»\u200F' + t[80:], 'rlm_quote'


def anomaly_features(x):
    """Family-agnostic Unicode anomaly features for open-set remedies."""
    n = max(1, len(x))
    cf = sum(1 for ch in x if unicodedata.category(ch) == 'Cf')
    tat = x.count(TATWEEL)
    thin = x.count(THIN)
    nbsp = x.count('\u00A0')
    spaces = sum(1 for ch in x if ch.isspace() or ch in (THIN, '\u00A0'))
    join_opp = max(1, len(kashida_positions(x.replace(TATWEEL, ''))))
    # space-type entropy proxy
    space_types = [ch for ch in x if ch in (' ', THIN, '\u00A0', '\u2000', '\u2001', '\u2002', '\u2003')]
    from collections import Counter
    c = Counter(space_types)
    tot = sum(c.values()) or 1
    ent = -sum((v/tot) * __import__('math').log(v/tot + 1e-12) for v in c.values())
    return [
        cf / n * 100,
        tat / n * 100,
        tat / join_opp,
        thin / max(1, spaces),
        nbsp / n * 100,
        ent,
        sum(1 for ch in x if ch in INVIS) / n * 100,
    ]
