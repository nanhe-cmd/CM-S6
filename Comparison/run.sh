#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
    echo "Usage: bash Comparison/run.sh CONFIG BASELINE PACKAGE.MODULE:FUNCTION [RUNS]"
    exit 2
fi

config="$1"
baseline="$2"
factory="$3"
runs="${4:-10}"

for run in $(seq 1 "$runs"); do
    seed=$((202600 + run))
    python Comparison/train.py \
        --config "$config" \
        --baseline "$baseline" \
        --factory "$factory" \
        --run "$run" \
        --seed "$seed"
done
