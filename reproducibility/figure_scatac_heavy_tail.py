#!/usr/bin/env python3
"""Number of non-zero features per cell in five real datasets.

    python3 reproducibility/figure_scatac_heavy_tail.py --datasets Datasets/Real

Reads <dataset>.mtx (Matrix Market coordinate format, cells in columns) for HSC, PBMC5K,
CellLines, HLCA_marrow and TCells and writes figure_scatac_heavy_tail.png. The left panel
shows the sorted number of non-zero features per cell, the right panel the share of cells
above the limit of 3072 non-zero features: below it the sparse Kendall kernel of GADES
keeps a cell in the shared memory of the GPU, above it the slower variant is used.

The matrices are not distributed with this repository.
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS = REPO_ROOT / "Datasets" / "Real"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

SHARED_MEMORY_LIMIT = 3072
DATASETS = ["HSC", "PBMC5K", "CellLines", "HLCA_marrow", "TCells"]
COL = {
    "HSC": "#d62728",
    "PBMC5K": "#1f77b4",
    "CellLines": "#2ca02c",
    "HLCA_marrow": "#9467bd",
    "TCells": "#ff7f0e",
}


def nonzeros_per_cell(path):
    """Number of stored entries in every non-empty column of a Matrix Market file."""
    with open(path) as handle:
        while True:
            line = handle.readline()
            if not line.startswith("%"):
                header = line.split()
                break
    cells = int(header[1])
    entries = pd.read_csv(
        path, sep=r"\s+", comment="%", header=None,
        usecols=[1], names=["c"], dtype=np.int32,
    )
    columns = entries.iloc[1:]["c"].values
    counts = np.bincount(columns, minlength=cells + 1)[1:cells + 1]
    return counts[counts > 0]


def main():
    parser = argparse.ArgumentParser(description="Non-zero features per cell in real datasets.")
    parser.add_argument(
        "--datasets",
        default=str(DEFAULT_DATASETS),
        help="directory with <dataset>.mtx files",
    )
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    missing = []
    for dataset in DATASETS:
        path = os.path.join(args.datasets, f"{dataset}.mtx")
        if not os.path.isfile(path):
            missing.append(path)
    if missing:
        listing = "\n  ".join(missing)
        raise SystemExit(f"dataset matrices not found:\n  {listing}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for dataset in DATASETS:
        counts[dataset] = nonzeros_per_cell(os.path.join(args.datasets, f"{dataset}.mtx"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for dataset in DATASETS:
        ranked = np.sort(counts[dataset])[::-1]
        percentile = np.arange(1, len(ranked) + 1) / len(ranked) * 100
        label = f"{dataset} (max {ranked.max()}, med {int(np.median(counts[dataset]))})"
        axes[0].plot(percentile, ranked, lw=2, color=COL[dataset], label=label)
    limit_label = f"variant-B limit ({SHARED_MEMORY_LIMIT})"
    axes[0].axhline(SHARED_MEMORY_LIMIT, ls="--", c="black", lw=1.5, label=limit_label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("cell percentile (by open-peak count)", fontsize=13)
    axes[0].set_ylabel("peaks per cell $k_c$ (log)", fontsize=13)
    axes[0].set_title("scATAC: heavy-tailed peaks-per-cell", fontsize=15)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=.3, which="both")

    shares = []
    colors = []
    for dataset in DATASETS:
        shares.append(100 * (counts[dataset] > SHARED_MEMORY_LIMIT).mean())
        colors.append(COL[dataset])
    axes[1].bar(range(len(DATASETS)), shares, color=colors)
    axes[1].set_xticks(range(len(DATASETS)))
    axes[1].set_xticklabels(DATASETS, rotation=30, ha="right", fontsize=11)
    axes[1].set_ylabel("% of cells above variant-B limit", fontsize=13)
    axes[1].set_title("Cells forcing the slow variant A", fontsize=15)
    axes[1].grid(alpha=.3, axis="y")
    for index, share in enumerate(shares):
        axes[1].annotate(f"{share:.0f}%", (index, share + 1), ha="center", fontsize=11)

    fig.tight_layout()
    out = out_dir / "figure_scatac_heavy_tail.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    for dataset, share in zip(DATASETS, shares):
        maximum = int(counts[dataset].max())
        median = int(np.median(counts[dataset]))
        print(f"  {dataset}: max {maximum}, median {median}, above the limit {share:.1f}%")


if __name__ == "__main__":
    main()
