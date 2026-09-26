import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = ("OA", "AA", "Kappa")
STRONG_BASELINES = ("picnet", "mtmixer", "mcamamba")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize CM-S6 experiment runs.")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    return parser.parse_args()


def load_runs(root):
    rows = []
    for path in sorted(root.glob("**/summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        best = record.get("best")
        if not best:
            continue
        row = {
            "dataset": record["dataset"],
            "method": record["run_name"],
            "run": int(record["run"]),
            "seed": int(record["seed"]),
            "parameters_M": float(record["parameters_M"]),
            "path": str(path),
        }
        for metric in METRICS:
            row[metric] = float(best[metric])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No summary.json files found below {root}")
    return rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(runs):
    groups = defaultdict(list)
    for row in runs:
        groups[(row["dataset"], row["method"])].append(row)
    output = []
    for (dataset, method), rows in sorted(groups.items()):
        record = {
            "dataset": dataset,
            "method": method,
            "runs": len(rows),
            "parameters_M": np.mean([row["parameters_M"] for row in rows]),
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(record)
    return output


def holm_adjust(p_values):
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def significance(runs):
    lookup = defaultdict(dict)
    for row in runs:
        lookup[(row["dataset"], row["method"])][row["seed"]] = row
    output = []
    for dataset in sorted({row["dataset"] for row in runs}):
        reference = lookup[(dataset, "full")]
        tests = []
        for baseline in STRONG_BASELINES:
            comparison = lookup[(dataset, f"comparison/{baseline}")]
            seeds = sorted(set(reference) & set(comparison))
            if len(seeds) < 2:
                continue
            differences = np.asarray(
                [reference[seed]["OA"] - comparison[seed]["OA"] for seed in seeds],
                dtype=float,
            )
            standard_error = differences.std(ddof=1) / math.sqrt(len(differences))
            margin = stats.t.ppf(0.975, len(differences) - 1) * standard_error
            raw_p = float(stats.ttest_rel(
                [reference[seed]["OA"] for seed in seeds],
                [comparison[seed]["OA"] for seed in seeds],
            ).pvalue)
            tests.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "paired_runs": len(seeds),
                    "OA_difference_pp": float(differences.mean()),
                    "CI95_low_pp": float(differences.mean() - margin),
                    "CI95_high_pp": float(differences.mean() + margin),
                    "raw_p": raw_p,
                }
            )
        if tests:
            adjusted = holm_adjust(np.asarray([row["raw_p"] for row in tests]))
            for row, value in zip(tests, adjusted):
                row["Holm_p"] = float(value)
                output.append(row)
    return output


def main():
    args = parse_args()
    runs = load_runs(args.root)
    output = args.output.expanduser().resolve()
    write_csv(output / "runs.csv", runs)
    write_csv(output / "summary.csv", aggregate(runs))
    tests = significance(runs)
    write_csv(output / "paired_significance.csv", tests)
    print(f"runs={len(runs)}")
    print(f"results={output}")


if __name__ == "__main__":
    main()
