#!/usr/bin/env python3
"""GADES modes on the generated sparse datasets relative to the dense GPU mode.

    python3 reproducibility/figure_gades_modes.py

Reads the tables written by collect_generated_dense.py and collect_generated_sparse.py and
writes figure4-barplot-sequential.png: one panel per matrix size, x is sparsity,
y = log10(T_GADES-GPU-dense / T_mode), aggregated over metrics and matrix shapes.
A mode above zero is faster than GADES-GPU-dense. GADES-CPU-dense does not depend on
sparsity and is repeated at every sparsity level.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "reproducibility" / "data"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

SIZES = [10 ** 5, 10 ** 6, 10 ** 7]
SIZE_LABELS = {10 ** 5: "$10^5$", 10 ** 6: "$10^6$", 10 ** 7: "$10^7$"}
SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.99]
MODES = ["GADES-CPU-dense", "GADES-CPU-sparse", "GADES-GPU-sparse"]
COL = {
    "GADES-CPU-dense": "#7e4ea3",
    "GADES-CPU-sparse": "#e377c2",
    "GADES-GPU-sparse": "#7f7f7f",
}
REFERENCE_COLOR = "#d62728"


def read_table(path, collector):
    if not Path(path).exists():
        raise SystemExit(f"table not found: {path}; run {collector} first")
    return pd.read_csv(path)


def load(dense_path, sparse_path):
    columns = ["metric", "method", "wX", "wY", "mean_sequential"]
    dense = read_table(dense_path, "collect_generated_dense.py")[columns].dropna()
    gpu_dense = dense[dense["method"] == "GPU"]
    reference = gpu_dense.groupby(["metric", "wX", "wY"])["mean_sequential"].median()
    reference = reference.rename("ref").reset_index()

    cpu_dense = dense[dense["method"] == "CPU"][["metric", "wX", "wY", "mean_sequential"]].copy()
    cpu_dense["method"] = "GADES-CPU-dense"
    repeated = []
    for sparsity in SPARSITIES:
        repeated.append(cpu_dense.assign(sparsity=sparsity))
    cpu_dense = pd.concat(repeated, ignore_index=True)

    columns = ["metric", "method", "sparsity", "wX", "wY", "mean_sequential"]
    sparse = read_table(sparse_path, "collect_generated_sparse.py")[columns].dropna()
    sparse = sparse[sparse["method"].isin(["GPU", "CPU"])].copy()
    sparse["method"] = sparse["method"].replace({"GPU": "GADES-GPU-sparse", "CPU": "GADES-CPU-sparse"})

    frame = pd.concat([cpu_dense, sparse], ignore_index=True)
    frame = frame.merge(reference, on=["metric", "wX", "wY"], how="inner")
    frame["|W|"] = frame["wX"] * frame["wY"]
    frame["logA"] = np.log10(frame["ref"] / frame["mean_sequential"])
    return frame[frame["|W|"].isin(SIZES)]


def main():
    parser = argparse.ArgumentParser(description="GADES modes on the generated sparse datasets.")
    parser.add_argument("--dense-table", default=str(DATA_DIR / "generated_dense.csv"))
    parser.add_argument("--sparse-table", default=str(DATA_DIR / "generated_sparse.csv"))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load(args.dense_table, args.sparse_table)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    for ax, size in zip(axes, SIZES):
        subset = frame[frame["|W|"] == size]
        present = []
        for mode in MODES:
            if not subset[subset["method"] == mode].empty:
                present.append(mode)
        x_order = []
        for sparsity in SPARSITIES:
            if sparsity in set(subset["sparsity"]):
                x_order.append(sparsity)
        if present and x_order:
            palette = {}
            for mode in present:
                palette[mode] = COL[mode]
            sns.boxplot(
                ax=ax, data=subset, x="sparsity", y="logA", hue="method",
                hue_order=present, order=x_order, palette=palette, fliersize=2,
            )
            if ax.get_legend():
                ax.get_legend().remove()
        ax.axhline(0, ls="-", c=REFERENCE_COLOR, lw=1.6)
        ax.set_title(f"$|W| = {SIZE_LABELS[size][1:-1]}$", fontsize=18)
        ax.set_xlabel("Sparsity degree", fontsize=14)
        ax.set_ylabel("Log10 Acceleration", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.yaxis.grid(True, alpha=.4)

    handles = []
    for mode in MODES:
        handles.append(Patch(facecolor=COL[mode], label=mode))
    handles.append(Line2D([0], [0], color=REFERENCE_COLOR, lw=1.6, label="GADES-GPU-dense baseline"))
    fig.legend(
        handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5),
        fontsize=12, frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 0.99, 1])
    out = out_dir / "figure4-barplot-sequential.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    print(frame.groupby(["|W|", "method"])["logA"].median().round(2).to_string())


if __name__ == "__main__":
    main()
