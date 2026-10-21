#!/usr/bin/env bash
# Reproduce all results in the ArabicStegoGuard paper (CPU, ~15-25 min).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data results
python3 fetch_data.py          # Arabic Wikipedia (wikimedia/wikipedia 20231101.ar) covers
python3 build_bench.py         # source-disjoint ArabicStegoBench-mini
python3 -u run_experiments.py | tee results/run_log.txt
python3 make_tables.py         # LaTeX tables -> results/tables.tex
