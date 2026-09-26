#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/houston.yaml}"
selection="${2:-all}"

if [[ "$selection" == "all" ]]; then
    experiments=(A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15)
    if [[ "$(basename "$config")" == augsburg.yaml ]]; then
        experiments+=(A16)
    fi
else
    experiments=("$selection")
fi

for experiment in "${experiments[@]}"; do
    for run in $(seq 1 10); do
        seed=$((202600 + run))
        python Ablation/train.py \
            --config "$config" \
            --experiment "$experiment" \
            --run "$run" \
            --seed "$seed"
    done
done
