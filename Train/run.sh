#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/houston.yaml}"
variant="${2:-full}"

for run in $(seq 1 10); do
    seed=$((202600 + run))
    python Train/train.py --config "$config" --variant "$variant" --run "$run" --seed "$seed"
done
