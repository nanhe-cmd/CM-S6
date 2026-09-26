import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("houston", "augsburg", "muufl")
DATASET_LABELS = {"houston": "Houston 2013", "augsburg": "Augsburg", "muufl": "MUUFL"}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot accuracy-efficiency Pareto fronts.")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/pareto"))
    return parser.parse_args()


def load_results(root):
    records = []
    for path in sorted(root.glob("**/efficiency_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        metrics = record.get("checkpoint_metrics")
        if not metrics:
            continue
        record["OA"] = float(metrics["OA"])
        records.append(record)
    if not records:
        raise FileNotFoundError("No evaluated efficiency JSON files were found")
    return records


def frontier(records, x_key):
    ordered = sorted(records, key=lambda item: (item[x_key], -item["OA"]))
    selected = []
    best_accuracy = -np.inf
    for record in ordered:
        if record["OA"] > best_accuracy:
            selected.append(record)
            best_accuracy = record["OA"]
    return selected


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 6.3,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(records, output):
    figure, axes = plt.subplots(2, 3, figsize=(7.12, 4.15))
    settings = (
        ("FLOPs", "FLOPs", "parameters_M"),
        ("latency_ms", "Latency (ms)", "peak_memory_MiB"),
    )
    colors = plt.get_cmap("tab10")
    model_names = sorted({record["model"] for record in records})
    color_map = {name: colors(index % 10) for index, name in enumerate(model_names)}
    for row, (x_key, x_label, bubble_key) in enumerate(settings):
        for column, dataset in enumerate(DATASETS):
            axis = axes[row, column]
            subset = [
                record for record in records
                if record["dataset"] == dataset and record.get(x_key) is not None
            ]
            if not subset:
                axis.set_visible(False)
                continue
            bubble_values = [record.get(bubble_key) or 0.0 for record in subset]
            scale = max(bubble_values) or 1.0
            for record, bubble in zip(subset, bubble_values):
                size = 25.0 + 95.0 * bubble / scale
                axis.scatter(
                    record[x_key],
                    record["OA"],
                    s=size,
                    color=color_map[record["model"]],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                )
                axis.annotate(
                    record["model"],
                    (record[x_key], record["OA"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=5.2,
                )
            front = frontier(subset, x_key)
            axis.plot(
                [record[x_key] for record in front],
                [record["OA"] for record in front],
                color="#4B5563",
                linewidth=0.8,
                linestyle="--",
                zorder=2,
            )
            axis.set_xscale("log")
            axis.grid(color="#DDE1E4", linewidth=0.45, linestyle="--", alpha=0.75)
            axis.set_title(f"({chr(ord('a') + row * 3 + column)}) {DATASET_LABELS[dataset]}")
            axis.set_xlabel(x_label)
            if column == 0:
                axis.set_ylabel("OA (%)")
    figure.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.10, wspace=0.27, hspace=0.38)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    figure.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def main():
    args = parse_args()
    configure_style()
    plot(load_results(args.root), args.output)
    print(f"figure={args.output.resolve()}")


if __name__ == "__main__":
    main()
