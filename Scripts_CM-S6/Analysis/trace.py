import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model, modulation_blocks
from utils import load_config, run_training, set_seed


SCALARS = ("eta", "alpha", "beta", "gamma", "zeta")


def parse_args():
    parser = argparse.ArgumentParser(description="Track the five CM-S6 modulation coefficients.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("full", "lite"), default="full")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def collect(epoch, model, loss):
    row = {"epoch": epoch, "loss": loss}
    blocks = modulation_blocks(model)
    row["blocks"] = len(blocks)
    for name in SCALARS:
        raw = []
        effective = []
        for block in blocks:
            if not hasattr(block, name):
                continue
            raw.append(float(getattr(block, name).detach().cpu()))
            effective.append(float(block.effective_scalar(name).detach().cpu()))
        row[f"{name}_raw_mean"] = float(np.mean(raw)) if raw else np.nan
        row[f"{name}_effective_mean"] = float(np.mean(effective)) if effective else np.nan
        row[f"{name}_effective_min"] = float(np.min(effective)) if effective else np.nan
        row[f"{name}_effective_max"] = float(np.max(effective)) if effective else np.nan
    return row


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    model = build_model(config, args.variant)
    rows = []

    def callback(epoch, current_model, loss):
        rows.append(collect(epoch, current_model, loss))

    run_directory, _ = run_training(
        config,
        model,
        run_name=f"trace/{args.variant}",
        run=args.run,
        seed=args.seed,
        device=args.device,
        output=args.output,
        epochs=args.epochs,
        epoch_callback=callback,
    )
    trace_file = run_directory / "scalar_trace.csv"
    with trace_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"trace={trace_file}")


if __name__ == "__main__":
    main()
