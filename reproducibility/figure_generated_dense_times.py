#!/usr/bin/env python3
"""Execution time on the generated dense datasets.

    python3 reproducibility/figure_generated_dense_times.py

Reads the table written by collect_generated_dense.py and writes figure3_dense_times.png:
one panel per metric, x is the matrix size, y is log10 of the mean time in microseconds
without the first repetition.

The benchmark method `pythonic` is shown as `scipy` with its time multiplied by 24.
Spearman of ArrayFire and Armadillo is left out: both rank without averaging ties.
A method without a result has no box.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "reproducibility" / "data"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

SIZES = [10 ** 5, 10 ** 6, 10 ** 7]
SIZE_LABELS = {10 ** 5: "$10^5$", 10 ** 6: "$10^6$", 10 ** 7: "$10^7$"}
METRICS = ["euclidean", "cosine", "pearson", "manhattan", "spearman", "kendall"]
PYTHON_TIME_FACTOR = 24

TAB20 = sns.color_palette("tab20", 20)
COL = {
    "scipy": TAB20[0],
    "pandas": TAB20[2],
    "factoextra": TAB20[4],
    "amap": TAB20[6],
    "armadillo": TAB20[8],
    "arrayfirecuda": TAB20[10],
    "arrayfirecpu": TAB20[11],
    "raft": TAB20[12],
    "GADES-CPU-dense": TAB20[18],
    "GADES-GPU-dense": TAB20[14],
}
METHODS = [
    "scipy", "pandas", "factoextra", "amap", "armadillo", "arrayfirecuda", "arrayfirecpu",
    "raft", "GADES-CPU-dense", "GADES-GPU-dense",
]
NO_TIE_AVERAGING = ["arrayfirecuda", "arrayfirecpu", "armadillo"]


def keep_order(preferred, present):
    ordered = []
    for item in preferred:
        if item in present:
            ordered.append(item)
    return ordered


def palette_for(methods):
    palette = {}
    for method in methods:
        palette[method] = COL[method]
    return palette


def load_dense(path):
    if not Path(path).exists():
        raise SystemExit(f"table not found: {path}; run collect_generated_dense.py first")
    table = pd.read_csv(path)
    table = table[["metric", "method", "wX", "wY", "|W|", "mean_sequential"]].dropna().copy()
    is_python = table["method"] == "pythonic"
    table.loc[is_python, "mean_sequential"] *= PYTHON_TIME_FACTOR
    table.loc[is_python, "method"] = "scipy"
    no_ties = (table["metric"] == "spearman") & table["method"].isin(NO_TIE_AVERAGING)
    table = table[~no_ties].copy()
    table["method"] = table["method"].replace({"CPU": "GADES-CPU-dense", "GPU": "GADES-GPU-dense"})
    table["logT"] = np.log10(table["mean_sequential"])
    table = table[table["|W|"].isin(SIZES)].copy()
    table["Wlab"] = table["|W|"].map(SIZE_LABELS)
    return table


def main():
    parser = argparse.ArgumentParser(description="Execution time on the generated dense datasets.")
    parser.add_argument("--dense-table", default=str(DATA_DIR / "generated_dense.csv"))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_dense(args.dense_table)
    methods = keep_order(METHODS, set(table["method"]))
    size_labels = ["$10^5$", "$10^6$", "$10^7$"]

    fig, axes = plt.subplots(2, 3, figsize=(30, 14))
    for ax, metric in zip(axes.flat, METRICS):
        subset = table[(table["metric"] == metric) & table["method"].isin(methods)]
        present = keep_order(methods, set(subset["method"]))
        x_order = keep_order(size_labels, set(subset["Wlab"]))
        if present and x_order:
            sns.boxplot(
                ax=ax, data=subset, x="Wlab", y="logT", hue="method",
                hue_order=present, order=x_order, palette=palette_for(present), fliersize=1.5,
            )
            if ax.get_legend():
                ax.get_legend().remove()
        ax.set_title(metric.capitalize(), fontsize=24)
        ax.set_xlabel("$|W|$", fontsize=17)
        ax.set_ylabel("Log10 execution time, $\\mu$s", fontsize=15)
        ax.tick_params(labelsize=16)
        ax.yaxis.grid(True, alpha=.4)

    handles = []
    for method in methods:
        handles.append(Patch(facecolor=COL[method], label=method))
    fig.legend(
        handles=handles, loc="upper center", ncol=len(handles),
        fontsize=16, bbox_to_anchor=(.5, 1.04), frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, .97])
    out = out_dir / "figure3_dense_times.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
