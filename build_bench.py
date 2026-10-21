"""Build the ArabicStegoBench-mini split (source-disjoint)."""
import json, random, stego
rng = random.Random(2026)
covers = json.load(open('data/covers.json'))
srcs = sorted({c['source_id'] for c in covers}); rng.shuffle(srcs)
n = len(srcs); part = {s: ('train' if i < .7*n else 'val' if i < .8*n else 'test') for i, s in enumerate(srcs)}
METHODS = ['ZW1', 'KHN', 'KSH', 'PSP']
rows = []
for c in covers:
    t = c['text']; r = rng.random()
    row = dict(source_id=c['source_id'], split=part[c['source_id']], cover=t)
    if r < 0.4:
        row.update(text=t, label=0, kind='clean', method='none', offsets=[], bits=[])
    elif r < 0.6:
        s, k = stego.hard_negative(t, rng)
        row.update(text=s, label=0, kind='hardneg_' + k, method='none', offsets=[], bits=[])
    else:
        m = rng.choice(METHODS); nb = rng.choice([16, 32, 64])
        nb = min(nb, stego.capacity(m, t) - (stego.capacity(m, t) % 2))
        if nb < 8:
            row.update(text=t, label=0, kind='clean', method='none', offsets=[], bits=[])
        else:
            bits = [rng.randint(0, 1) for _ in range(nb)]
            s, offs = stego.embed(m, t, bits)
            assert stego.decode(m, s, nb) == bits, m
            row.update(text=s, label=1, kind='stego', method=m, offsets=offs, bits=bits)
    rows.append(row)
json.dump(rows, open('data/bench.json', 'w'), ensure_ascii=False)
from collections import Counter
print(Counter((r['split'], r['kind'] if r['label'] == 0 else r['method']) for r in rows))
