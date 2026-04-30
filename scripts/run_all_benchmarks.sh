#!/usr/bin/env bash
# Train IRL-GAD on every benchmark used in the paper.
#
# Usage:
#   bash scripts/run_all_benchmarks.sh
#
# Each run writes a checkpoint into ./experiments/runs/<dataset>_seed0_K2/.
# Results JSON is appended to ./experiments/all_benchmarks.jsonl.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="experiments/all_benchmarks.jsonl"
mkdir -p experiments
: > "$OUT"   # truncate

for cfg in cora citeseer amazon yelpchi jodie ogbn_arxiv; do
    echo
    echo "============================================================"
    echo "  training: $cfg"
    echo "============================================================"
    python main.py train --config "configs/${cfg}.yaml" \
        | tee -a "$OUT"
done

echo
echo "all benchmark runs complete."
echo "log -> $OUT"
