import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("houston", "augsburg", "muufl")
DATASET_LABELS = {
    "houston": "Houston 2013",
    "augsburg": "Augsburg",
    "muufl": "MUUFL",
}
VARIANTS = ("full", "lite")
VARIANT_LABELS = {"full": "CM-S6", "lite": "CM-S6(L)"}
COEFFICIENTS = (
    ("eta", r"$\eta$ (CII)", "#426B82"),
    ("alpha", r"$\alpha$ ($\Delta$)", "#C77C2E"),
    ("beta", r"$\beta$ ($B$)", "#4F7967"),
    ("gamma", r"$\gamma$ ($C$)", "#B85F5F"),
    ("zeta", r"$\zeta$ (output)", "#8C6D91"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CM-S6 coefficient analyses.")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    return parser.parse_args()


def find_trace(root, dataset, variant):
    pattern = f"{dataset}/trace/{variant}/run_01_seed_202601*/scalar_trace.csv"
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No trace matched {root / pattern}")
    return matches[0]


def read_trace(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty trace file: {path}")
    return rows


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 6.4,
            "axes.titlesize": 7.1,
            "axes.labelsize": 6.9,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(figure, output_stem):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(output_stem.with_suffix(".svg"), facecolor="white")


def plot_trajectories(traces, output):
    figure, axes = plt.subplots(3, 2, figsize=(7.12, 5.18), sharex=True, sharey=True)
    legend = []
    panel = 0
    for row_index, dataset in enumerate(DATASETS):
        for column_index, variant in enumerate(VARIANTS):
            axis = axes[row_index, column_index]
            rows = traces[(dataset, variant)]
            epochs = np.asarray([int(row["epoch"]) for row in rows])
            tail_start = max(int(epochs.max()) - 19, 1)
            axis.axvspan(tail_start, epochs.max(), color="#9CA3AF", alpha=0.10, linewidth=0)
            for name, label, color in COEFFICIENTS:
                values = np.asarray([float(row[f"{name}_effective_mean"]) for row in rows])
                line = axis.plot(epochs, values, color=color, linewidth=1.25, label=label)[0]
                axis.scatter(
                    epochs[-1],
                    values[-1],
                    s=13,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=4,
                )
                if panel == 0:
                    legend.append(line)
            axis.set_title(f"{DATASET_LABELS[dataset]} | {VARIANT_LABELS[variant]}", pad=3)
            axis.text(
                0.0,
                1.025,
                f"({chr(ord('a') + panel)})",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
            )
            axis.grid(color="#DDE1E4", linewidth=0.5, linestyle="--", alpha=0.82)
            axis.tick_params(length=2.4, width=0.6, pad=2)
            if column_index == 0:
                axis.set_ylabel("Effective coefficient")
            if row_index == 2:
                axis.set_xlabel("Epoch")
            panel += 1

    figure.legend(
        legend,
        [label for _, label, _ in COEFFICIENTS],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
    )
    figure.subplots_adjust(left=0.083, right=0.988, top=0.925, bottom=0.09, wspace=0.18, hspace=0.35)
    save_figure(figure, output / "coefficient_trajectories")
    plt.close(figure)


def plot_reallocation(traces, output):
    figure, axes = plt.subplots(1, 3, figsize=(7.12, 2.32), sharey=True)
    positions = np.arange(len(COEFFICIENTS))
    for panel, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        full = np.asarray(
            [float(traces[(dataset, "full")][-1][f"{name}_effective_mean"]) for name, _, _ in COEFFICIENTS]
        )
        lite = np.asarray(
            [float(traces[(dataset, "lite")][-1][f"{name}_effective_mean"]) for name, _, _ in COEFFICIENTS]
        )
        for y, full_value, lite_value in zip(positions, full, lite):
            axis.plot([full_value, lite_value], [y, y], color="#9AA1A8", linewidth=1.0)
        axis.scatter(full, positions, s=25, color="#426B82", edgecolor="white", linewidth=0.55, label="CM-S6")
        axis.scatter(lite, positions, s=25, marker="s", color="#B37A3C", edgecolor="white", linewidth=0.55, label="CM-S6(L)")
        axis.set_title(DATASET_LABELS[dataset], pad=3)
        axis.text(
            0.0,
            1.035,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            fontsize=7.0,
            fontweight="bold",
        )
        axis.set_xlabel("Final effective coefficient")
        axis.set_xlim(0.05, 0.53)
        axis.set_ylim(-0.65, 4.65)
        axis.invert_yaxis()
        axis.grid(axis="x", color="#DDE1E4", linewidth=0.5, linestyle="--", alpha=0.82)
        axis.tick_params(length=2.4, width=0.6, pad=1.5)
    axes[0].set_yticks(positions, [label for _, label, _ in COEFFICIENTS])
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=2, frameon=False)
    figure.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.20, wspace=0.17)
    save_figure(figure, output / "coefficient_reallocation")
    plt.close(figure)


def main():
    args = parse_args()
    configure_style()
    traces = {
        (dataset, variant): read_trace(find_trace(args.root, dataset, variant))
        for dataset in DATASETS
        for variant in VARIANTS
    }
    plot_trajectories(traces, args.output)
    plot_reallocation(traces, args.output)
    print(f"figures={args.output.resolve()}")


if __name__ == "__main__":
    main()
