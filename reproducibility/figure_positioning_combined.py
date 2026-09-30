#!/usr/bin/env python3
"""GADES 1.0 versus GADES 2.0 on the Kendall distance, plain bar charts.

    python3 reproducibility/figure_positioning_combined.py --gades1-results <Article-GADES>/results

Writes figure_positioning_combined.png with three panels: (a) generated dense data against
the matrix size with the amap baseline, (b) generated sparse data of 10000 x 1000 against
sparsity, (c) real datasets, sparse mode on the GPU and on the CPU.

GADES 2.0 times are read from results/ of this repository. GADES 1.0 times are read from
the results/ directory of https://github.com/lab-medvedeva/Article-GADES, passed with
--gades1-results. A time is the mean of the repetitions without the first one. Panel (a)
shows the median over the matrix shapes of one size. GADES 1.0 did not finish PBMC5K
within 24 hours; its bars in panel (c) are drawn at that limit.
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
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

SIZES = [1e5, 1e6, 1e7]
SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.99]
SPARSE_SHAPE = "10000_cells_1000_features"
DATASETS = [
    "B_CD8T", "B_T", "Camp", "CellLines", "Chen", "FibrocardRNA", "HLCA_aorta", "HLCA_lung",
    "HLCA_marrow", "HSC", "PBMC_all", "TCells", "PBMC5K",
]
TIMED_OUT_GADES1 = {"PBMC5K": ["GPU", "CPU"]}
TIMEOUT_US = 86400.0 * 1e6

COLOR_AMAP = "#969696"
COLOR_GPU_1 = "#9ecae1"
COLOR_GPU_2 = "#08519c"
COLOR_CPU_1 = "#fdae6b"
COLOR_CPU_2 = "#a63603"


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


def seconds(value):
    if value:
        return value / 1e6
    return np.nan


def dense_shapes():
    shapes = []
    for cells in [10, 100, 1000, 10000]:
        for features in [10, 100, 1000, 10000, 100000]:
            if cells * features in (10 ** 5, 10 ** 6, 10 ** 7):
                shapes.append((cells, features, cells * features))
    return shapes


def dense_medians(root, file_name):
    """Median over the shapes of each matrix size, in seconds."""
    buckets = {}
    for size in SIZES:
        buckets[size] = []
    for cells, features, size in dense_shapes():
        value = mean_us(f"{root}/{cells}_cells_{features}_features/{file_name}")
        if value:
            buckets[float(size)].append(value)
    medians = []
    for size in SIZES:
        if buckets[size]:
            medians.append(seconds(np.median(buckets[size])))
        else:
            medians.append(np.nan)
    return medians


def sparse_times(directory, template):
    times = []
    for sparsity in SPARSITIES:
        times.append(seconds(mean_us(f"{directory}/{template.format(sparsity=sparsity)}")))
    return times


def real_times(gades1_real, gades2_real):
    rows = []
    for dataset in DATASETS:
        for hardware in ["GPU", "CPU"]:
            for version, root in [("v1", gades1_real), ("v2", gades2_real)]:
                value = mean_us(f"{root}/{dataset}/sparse_{hardware}_kendall.csv")
                series = f"{version}-{hardware}"
                if value == value:
                    rows.append({"ds": dataset, "series": series, "logt": np.log10(value)})
                elif version == "v1" and hardware in TIMED_OUT_GADES1.get(dataset, []):
                    rows.append({"ds": dataset, "series": series, "logt": np.log10(TIMEOUT_US)})
    return pd.DataFrame(rows)


def draw_bars(ax, groups, labels, series, title, xlabel):
    count = len(series)
    positions = np.arange(len(groups))
    width = 0.8 / count
    for index, (name, values, color) in enumerate(series):
        offset = (index - (count - 1) / 2) * width
        heights = []
        for value in values:
            if value == value:
                heights.append(value)
            else:
                heights.append(0)
        ax.bar(positions + offset, heights, width, label=name, color=color)
        for position, value in zip(positions, values):
            if value != value:
                ax.annotate(
                    "T.O.", (position + offset, ax.get_ylim()[0] * 1.5),
                    ha="center", rotation=90, fontsize=8, color="gray",
                )
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Kendall time, s (log)", fontsize=13)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=10, ncol=2)
    ax.yaxis.grid(True)


def main():
    parser = argparse.ArgumentParser(description="GADES 1.0 versus 2.0 on Kendall, bar charts.")
    parser.add_argument(
        "--gades1-results",
        required=True,
        help="results/ directory of https://github.com/lab-medvedeva/Article-GADES",
    )
    parser.add_argument("--generated-dense", default=str(RESULTS_DIR / "GeneratedDense"))
    parser.add_argument("--generated-sparse", default=str(RESULTS_DIR / "GeneratedSparse"))
    parser.add_argument("--real-results", default=str(RESULTS_DIR / "RealDatasets"))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    gades1 = args.gades1_results
    if not os.path.isdir(gades1):
        raise SystemExit(f"GADES 1.0 results not found: {gades1}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gades1_dense = f"{gades1}/GeneratedDense"
    dense_gpu_1 = dense_medians(gades1_dense, "_GPU_kendall.csv")
    dense_gpu_2 = dense_medians(args.generated_dense, "dense_GPU_kendall.csv")
    dense_cpu_1 = dense_medians(gades1_dense, "_CPU_kendall.csv")
    dense_cpu_2 = dense_medians(args.generated_dense, "dense_CPU_kendall.csv")
    dense_amap = dense_medians(gades1_dense, "_amap_kendall.csv")

    gades1_sparse = f"{gades1}/GeneratedSparse/{SPARSE_SHAPE}"
    gades2_sparse = f"{args.generated_sparse}/{SPARSE_SHAPE}"
    sparse_gpu_1 = sparse_times(gades1_sparse, "{sparsity}_GPU_kendall.csv")
    sparse_gpu_2 = sparse_times(gades2_sparse, "{sparsity}/sparse_GPU_kendall.csv")
    sparse_cpu_1 = sparse_times(gades1_sparse, "{sparsity}_CPU_kendall.csv")
    sparse_cpu_2 = sparse_times(gades2_sparse, "{sparsity}/sparse_CPU_kendall.csv")

    real = real_times(f"{gades1}/RealDatasets", args.real_results)

    fig = plt.figure(figsize=(20, 13))
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.05], hspace=0.32, wspace=0.18)
    ax_dense = fig.add_subplot(grid[0, 0])
    ax_sparse = fig.add_subplot(grid[0, 1])
    ax_real = fig.add_subplot(grid[1, :])

    dense_series = [
        ("amap (CPU baseline)", dense_amap, COLOR_AMAP),
        ("1.0 GPU", dense_gpu_1, COLOR_GPU_1),
        ("2.0 GPU", dense_gpu_2, COLOR_GPU_2),
        ("1.0 CPU", dense_cpu_1, COLOR_CPU_1),
        ("2.0 CPU", dense_cpu_2, COLOR_CPU_2),
    ]
    draw_bars(
        ax_dense, SIZES, ["$10^5$", "$10^6$", "$10^7$"], dense_series,
        "(a) Generated dense: GADES 1.0 vs 2.0 vs amap", "$|W|$ (matrix elements)",
    )

    sparse_series = [
        ("1.0 GPU", sparse_gpu_1, COLOR_GPU_1),
        ("2.0 GPU", sparse_gpu_2, COLOR_GPU_2),
        ("1.0 CPU", sparse_cpu_1, COLOR_CPU_1),
        ("2.0 CPU", sparse_cpu_2, COLOR_CPU_2),
    ]
    sparsity_labels = []
    for sparsity in SPARSITIES:
        sparsity_labels.append(str(sparsity))
    draw_bars(
        ax_sparse, SPARSITIES, sparsity_labels, sparse_series,
        "(b) Generated sparse @ $10^4\\times10^3$: GADES 1.0 vs 2.0", "sparsity",
    )

    palette = {
        "v1-GPU": COLOR_GPU_1,
        "v2-GPU": COLOR_GPU_2,
        "v1-CPU": COLOR_CPU_1,
        "v2-CPU": COLOR_CPU_2,
    }
    series_order = ["v1-GPU", "v2-GPU", "v1-CPU", "v2-CPU"]
    sns.barplot(
        ax=ax_real, data=real, x="ds", y="logt", hue="series",
        hue_order=series_order, palette=palette, order=DATASETS,
    )
    ceiling = np.log10(TIMEOUT_US)
    ax_real.axhline(ceiling, ls="--", c="gray", lw=1.4)
    ax_real.annotate(
        "1.0 timed out", (DATASETS.index("PBMC5K"), ceiling + 0.05),
        ha="center", fontsize=10, color="gray", fontweight="bold",
    )
    ax_real.set_title("(c) Real datasets: GADES 1.0 vs 2.0 (RTX 3090)", fontsize=16)
    ax_real.set_ylabel("Log10 execution time, $\\mu$s", fontsize=13)
    ax_real.set_xlabel("")
    ax_real.tick_params(axis="x", rotation=45, labelsize=11)
    ax_real.legend(fontsize=11, title="version / hardware", ncol=2)
    ax_real.yaxis.grid(True, alpha=.4)
    for label in ax_real.get_xticklabels():
        label.set_ha("right")

    out = out_dir / "figure_positioning_combined.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
