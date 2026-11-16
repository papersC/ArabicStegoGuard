#!/usr/bin/env bash
# Reproduce all results in the ArabicStegoGuard paper (CPU, ~15-25 min).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data results
python3 fetch_data.py          # Arabic Wikipedia (wikimedia/wikipedia 20231101.ar) covers
python3 build_bench.py         # source-disjoint ArabicStegoBench-mini
python3 -u run_experiments.py | tee results/run_log.txt
python3 make_tables.py         # LaTeX tables -> results/tables.tex
python3 -u run_revision_experiments.py | tee results/revision_run_log.txt
python3 make_figures.py --results-dir results   # Figs. 2-5 -> results/*.pdf
