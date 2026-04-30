#!/usr/bin/env bash
# Run the component ablation study (Table 6 in the paper) on YelpChi.
#
# Usage:
#   bash scripts/ablation.sh                   # YelpChi
#   bash scripts/ablation.sh configs/cora.yaml # any config

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG="${1:-configs/yelpchi.yaml}"

python -m experiments.ablation \
    --config "$CFG" \
    --out experiments/ablation_results.json

echo
echo "ablation done -> experiments/ablation_results.json"
