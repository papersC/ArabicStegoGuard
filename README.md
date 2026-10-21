# ArabicStegoGuard — reproduction code

Code for "ArabicStegoGuard: A Linguistically Informed and Adversarially Robust Deep Learning Framework for Arabic Text Steganalysis" (IEEE TIFS submission).

* `fetch_data.py` – downloads cover paragraphs from Arabic Wikipedia (`wikimedia/wikipedia`, `20231101.ar`, first 2,391 usable articles, 8,003 paragraphs of 200–800 chars).
* `stego.py` – four embedding families with decoders (ZW1: ZWJ/ZWNJ non-printable; KHN: 4 invisible symbols, 2 bits/letter; KSH: Kashida; PSP: thin-space pseudo-space), benign hard negatives (legitimate ZWNJ, decorative tatweel, LRM/RLM, NBSP), transformations.
* `build_bench.py` – ArabicStegoBench-mini, source(article)-disjoint 70/10/20 split, seed 2026.
* `models.py` – B1 rules, B2 code-point n-gram LR, B4 language-only LR, B3 raw char-CNN, B5 concat fusion, B6 ArabicStegoGuard (gated dual stream + localization head + temperature scaling).
* `run_experiments.py` – E1 (source-disjoint), localization, E2 (leave-one-family-out), PSR and transformed detection.
* `make_tables.py` – LaTeX tables.

Run: `pip install -r requirements.txt && ./reproduce.sh` (CPU, ~40 min on 8 cores). Results: `results/results.json`, `results/run_log.txt`.

Scope note: the linguistic stream is a lightweight character-aligned normalized view, not a pretrained Arabic LM; the benchmark is a synthetic, Wikipedia-only study.
