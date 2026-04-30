#!/usr/bin/env bash
# Run the open-set generalization experiment (Table 5).
#
# Trains on structural anomalies, then evaluates the same checkpoint
# on three held-out anomaly types: attribute, contextual, hybrid.
#
# Usage:
#   bash scripts/openset.sh                  # YelpChi
#   bash scripts/openset.sh configs/cora.yaml

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFG="${1:-configs/yelpchi.yaml}"

python -m experiments.openset \
    --config "$CFG" \
    --out experiments/openset_results.json

echo
echo "open-set done -> experiments/openset_results.json"
