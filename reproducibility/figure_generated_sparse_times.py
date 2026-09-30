#!/usr/bin/env python3
"""Execution time on the generated sparse datasets.

    python3 reproducibility/figure_generated_sparse_times.py

Reads the tables written by collect_generated_dense.py and collect_generated_sparse.py and
writes figure4_sparse_times_W<k>.png for every matrix size: one panel per metric, x is
sparsity, y is log10 of the mean time in microseconds without the first repetition.
Dense methods do not depend on sparsity and are repeated at every sparsity level.

The benchmark method `pythonic` is shown as `scipy` with its time multiplied by 24.
Spearman of ArrayFire and Armadillo is left out: both rank without averaging ties.
Of raft, rapids and cupy only raft is drawn.
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
SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.99]
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
    "raft": TAB20[12],
    "GADES-GPU-dense": TAB20[14],
    "GADES-CPU-sparse": TAB20[3],
    "GADES-GPU-sparse": TAB20[5],
}
METHODS = [
    "scipy", "pandas", "factoextra", "amap", "armadillo", "arrayfirecuda", "raft",
    "GADES-GPU-dense", "GADES-CPU-sparse", "GADES-GPU-sparse",
]
NO_TIE_AVERAGING = ["arrayfirecuda", "armadillo"]


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


def read_table(path, collector):
    if not Path(path).exists():
        raise SystemExit(f"table not found: {path}; run {collector} first")
    return pd.read_csv(path)


def combine(sparse_path, dense_path):
    """Sparse methods plus the dense baselines repeated at every sparsity level."""
    columns = ["metric", "method", "sparsity", "wX", "wY", "mean_sequential"]
    sparse = read_table(sparse_path, "collect_generated_sparse.py")[columns].dropna().copy()
    sparse["method"] = sparse["method"].replace({"CPU": "GADES-CPU-sparse", "GPU": "GADES-GPU-sparse"})
    sparse_kept = ["GADES-CPU-sparse", "GADES-GPU-sparse", "raft", "armadillo", "arrayfirecuda"]
    sparse = sparse[sparse["method"].isin(sparse_kept)]
    no_ties = (sparse["metric"] == "spearman") & sparse["method"].isin(NO_TIE_AVERAGING)
    sparse = sparse[~no_ties].copy()
    sparse["time"] = sparse["mean_sequential"]

    columns = ["metric", "method", "wX", "wY", "mean_sequential"]
    dense = read_table(dense_path, "collect_generated_dense.py")[columns].dropna().copy()
    is_python = dense["method"] == "pythonic"
    dense.loc[is_python, "mean_sequential"] *= PYTHON_TIME_FACTOR
    dense.loc[is_python, "method"] = "scipy"
    dense = dense[dense["method"].isin(["amap", "scipy", "pandas", "factoextra", "GPU"])].copy()
    dense["method"] = dense["method"].replace({"GPU": "GADES-GPU-dense"})
    repeated = []
    for sparsity in SPARSITIES:
        level = dense.copy()
        level["sparsity"] = sparsity
        level["time"] = level["mean_sequential"]
        repeated.append(level)
    dense = pd.concat(repeated, ignore_index=True)

    columns = ["metric", "method", "sparsity", "wX", "wY", "time"]
    combined = pd.concat([sparse[columns], dense[columns]], ignore_index=True)
    combined["|W|"] = combined["wX"] * combined["wY"]
    combined["logT"] = np.log10(combined["time"])
    return combined[combined["|W|"].isin(SIZES)]


def main():
    parser = argparse.ArgumentParser(description="Execution time on the generated sparse datasets.")
    parser.add_argument("--dense-table", default=str(DATA_DIR / "generated_dense.csv"))
    parser.add_argument("--sparse-table", default=str(DATA_DIR / "generated_sparse.csv"))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = combine(args.sparse_table, args.dense_table)
    methods = keep_order(METHODS, set(combined["method"]))
    for size in SIZES:
        frame = combined[combined["|W|"] == size]
        fig, axes = plt.subplots(2, 3, figsize=(30, 14))
        for ax, metric in zip(axes.flat, METRICS):
            subset = frame[(frame["metric"] == metric) & frame["method"].isin(methods)]
            present = keep_order(methods, set(subset["method"]))
            x_order = keep_order(SPARSITIES, set(subset["sparsity"]))
            if present and x_order:
                sns.boxplot(
                    ax=ax, data=subset, x="sparsity", y="logT", hue="method",
                    hue_order=present, order=x_order, palette=palette_for(present), fliersize=1.5,
                )
                if ax.get_legend():
                    ax.get_legend().remove()
            ax.set_title(metric.capitalize(), fontsize=24)
            ax.set_xlabel("sparsity", fontsize=17)
            ax.set_ylabel("Log10 execution time, $\\mu$s", fontsize=15)
            ax.tick_params(labelsize=16)
            ax.yaxis.grid(True, alpha=.4)

        handles = []
        for method in methods:
            handles.append(Patch(facecolor=COL[method], label=method))
        fig.legend(
            handles=handles, loc="upper center", ncol=min(len(methods), 12),
            fontsize=16, bbox_to_anchor=(.5, 1.05), frameon=False,
        )
        title = f"Sparse log10 execution time \u2014 $|W|$ = {SIZE_LABELS[size]}"
        fig.suptitle(title, y=1.10, fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, .96])
        out = out_dir / f"figure4_sparse_times_W{int(np.log10(size))}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
