"""Fetch Arabic Wikipedia paragraphs (source = article) as cover corpus."""
import json, re, random
from datasets import load_dataset
random.seed(0)
ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)
out=[]; n_art=0
for art in ds:
    paras=[p.strip() for p in art["text"].split("\n") if 200<=len(p.strip())<=800]
    paras=[p for p in paras if len(re.findall(r'[\u0621-\u064A]',p))>0.6*len(p)]
    if not paras: continue
    for p in paras[:4]:
        out.append({"source_id":art["id"],"title":art["title"],"text":p})
    n_art+=1
    if len(out)>=8000: break
json.dump(out,open("data/covers.json","w"),ensure_ascii=False)
print(len(out),"paragraphs from",n_art,"articles")
