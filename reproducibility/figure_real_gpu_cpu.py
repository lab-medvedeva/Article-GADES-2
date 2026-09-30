#!/usr/bin/env python3
"""GADES sparse mode on the GPU and on the CPU, real datasets.

    python3 reproducibility/figure_real_gpu_cpu.py

Reads results/RealDatasets/<dataset>/sparse_{GPU,CPU}_<metric>.csv and writes
figure5_real_gpu_cpu.png: one panel per metric, x is the dataset, y is log10 of the mean
time in microseconds without the first repetition. A dataset without a result file has
no bar.
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "results" / "RealDatasets"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

METRICS = ["euclidean", "cosine", "pearson", "manhattan", "spearman", "kendall"]
HARDWARE = ["GPU", "CPU"]
PALETTE = {"GPU": "#1f77b4", "CPU": "#d62728"}


def mean_us(path):
    """Mean time in microseconds without the first repetition, NaN without a result."""
    if not os.path.exists(path):
        return np.nan
    try:
        values = pd.read_csv(path).iloc[:, 0].values
    except Exception:
        return np.nan
    values = values[values > 0]
    if len(values) > 1:
        return float(np.mean(values[1:]))
    if len(values):
        return float(values[0])
    return np.nan


def main():
    parser = argparse.ArgumentParser(description="GADES sparse GPU versus CPU on real datasets.")
    parser.add_argument("--real-results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    results = args.real_results
    if not os.path.isdir(results):
        raise SystemExit(f"results directory not found: {results}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in sorted(os.listdir(results)):
        for metric in METRICS:
            for hardware in HARDWARE:
                value = mean_us(f"{results}/{dataset}/sparse_{hardware}_{metric}.csv")
                if value == value:
                    row = {"dataset": dataset, "metric": metric, "hw": hardware, "logt": np.log10(value)}
                    rows.append(row)
    if not rows:
        raise SystemExit(f"no sparse GADES results found in {results}")
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 3, figsize=(26, 13))
    for ax, metric in zip(axes.flat, METRICS):
        subset = table[table["metric"] == metric]
        if not subset.empty:
            sns.barplot(
                ax=ax, data=subset, x="dataset", y="logt", hue="hw",
                hue_order=HARDWARE, palette=PALETTE,
            )
            if ax.get_legend():
                ax.get_legend().remove()
        ax.set_title(metric.capitalize(), fontsize=22)
        ax.set_ylabel("Log10 time, $\\mu$s", fontsize=15)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90, labelsize=13)
        ax.tick_params(axis="y", labelsize=14)
        ax.yaxis.grid(True, alpha=.4)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, fontsize=17,
        bbox_to_anchor=(.5, 1.03), title="GADES-sparse", title_fontsize=17,
    )
    fig.tight_layout(rect=[0, 0, 1, .97])
    out = out_dir / "figure5_real_gpu_cpu.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    pivot = table.pivot_table(index=["dataset", "metric"], columns="hw", values="logt").dropna()
    pivot["speedup"] = 10 ** (pivot["CPU"] - pivot["GPU"])
    medians = pivot.reset_index().groupby("metric")["speedup"].median().round(2).to_dict()
    print(f"saved: {out}")
    print("datasets:", table["dataset"].nunique())
    print("median GPU-over-CPU speedup by metric:", medians)


if __name__ == "__main__":
    main()
