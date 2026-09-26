#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/houston.yaml}"
variant="${2:-full}"

python Analysis/trace.py --config "$config" --variant "$variant" --run 1 --seed 202601
python Analysis/efficiency.py --config "$config" --variant "$variant"
