# ArabicStegoGuard — reproduction code

Code for "ArabicStegoGuard: A Linguistically Informed Dual-Stream Framework for Arabic Unicode Text Steganalysis" (IEEE TIFS submission).

* `fetch_data.py` – downloads cover paragraphs from Arabic Wikipedia (`wikimedia/wikipedia`, `20231101.ar`, first 2,391 usable articles, 8,003 paragraphs of 200–800 chars).
* `stego.py` – four embedding families with decoders (ZW1: ZWJ/ZWNJ non-printable; KHN: 4 invisible symbols, 2 bits/letter; KSH: Kashida; PSP: thin-space pseudo-space), benign hard negatives (legitimate ZWNJ, decorative tatweel, LRM/RLM, NBSP), transformations.
* `build_bench.py` – ArabicStegoBench-mini, source(article)-disjoint 70/10/20 split, seed 2026.
* `models.py` – B1 rules, B2 code-point n-gram LR, B4 language-only LR, B3 raw char-CNN, B5 concat fusion, B6 ArabicStegoGuard (gated dual stream + localization head + temperature scaling).
* `run_experiments.py` – E1 (source-disjoint), localization, E2 (leave-one-family-out), PSR and transformed detection.
* `make_tables.py` – LaTeX tables.
* `run_revision_experiments.py` – multi-seed fusion ablation, validation-matched operating points, expanded hard negatives, BBC Arabic news zero-shot transfer and unseen-family remedies (`results/revision_results.json`).
* `make_figures.py` – regenerates Figs. 2–5 of the paper from `results/results.json` and `results/revision_results.json` (PDFs in `results/`).

Run: `pip install -r requirements.txt && ./reproduce.sh` (CPU, ~40 min on 8 cores). Results: `results/results.json`, `results/revision_results.json`, the run logs and the figure PDFs in `results/`.

Scope note: the linguistic stream is a lightweight character-aligned normalized view, not a pretrained Arabic LM; the benchmark is a synthetic, Wikipedia-only study.
