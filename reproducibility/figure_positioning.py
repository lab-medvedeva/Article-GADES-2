#!/usr/bin/env python3
"""GADES 1.0 versus GADES 2.0 on the Kendall distance.

    python3 reproducibility/figure_positioning.py --gades1-results <Article-GADES>/results

Writes figure_positioning_canon.png and figure_positioning_canon.pdf with three panels:
(a) generated dense data against the matrix size with the amap baseline, (b) generated
sparse data of 10000 x 1000 against sparsity, (c) real datasets, sparse mode.

GADES 2.0 times are read from results/ of this repository. GADES 1.0 times are read from
the results/ directory of https://github.com/lab-medvedeva/Article-GADES, passed with
--gades1-results. A time is the mean of the repetitions without the first one. Panel (a)
shows the median over the matrix shapes of one size; GADES 1.0 on the CPU has no result
for the size 10^7 and is drawn at the 24 hour limit. In panel (c) a dataset without a
result file is drawn at the 24 hour limit.

Colour is the implementation, solid bars are GADES 2.0, hollow bars are GADES 1.0 and the
baseline, hatched bars ran on the GPU.
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

COL = {
    "amap": "#7f7f7f",
    "1.0 GPU": "#6baed6",
    "1.0 CPU": "#fdae6b",
    "2.0 GPU": "#08519c",
    "2.0 CPU": "#d94801",
}
FADE = 0.22
SOLID = {"2.0 GPU", "2.0 CPU"}
ON_GPU = {"1.0 GPU", "2.0 GPU"}

HATCH = "///"
LW_HOLLOW = 2.0
LW_SOLID = 1.4

FS_PANEL = 68
FS_TITLE = 50
FS_LABEL = 36
FS_TICK = 27
FS_LEGEND = 30

matplotlib.rcParams["hatch.linewidth"] = 2.0

SIZES = [10 ** 5, 10 ** 6, 10 ** 7]
SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.99]
SPARSE_SHAPE = "10000_cells_1000_features"
DATASETS = [
    "B_CD8T", "B_T", "Camp", "CellLines", "Chen", "FibrocardRNA",
    "HLCA_aorta", "HLCA_lung", "HLCA_marrow", "HSC", "PBMC_all",
    "TCells", "PBMC5K",
]
TIMEOUT_US = 86400.0 * 1e6
TIMEOUT_S = 86400.0

# series and matrix sizes that did not finish within 24 hours
DENSE_TIMED_OUT = {("1.0 CPU", 10 ** 7)}


def luminance(color):
    red, green, blue = mcolors.to_rgb(color)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def style_bar(bar, series):
    """Solid for GADES 2.0, hollow for GADES 1.0 and the baseline."""
    color = COL[series]
    if series in SOLID:
        bar.set_facecolor(color)
        bar.set_linewidth(LW_SOLID)
        if luminance(color) < 0.45:
            bar.set_edgecolor("#ffffff")
        else:
            bar.set_edgecolor("#222222")
    else:
        bar.set_facecolor(mcolors.to_rgba(color, FADE))
        bar.set_edgecolor(color)
        bar.set_linewidth(LW_HOLLOW)
    if series in ON_GPU:
        bar.set_hatch(HATCH)


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
    if value and value == value:
        return value / 1e6
    return np.nan


def dense_shapes():
    shapes = []
    for cells in (10, 100, 1000, 10000):
        for features in (10, 100, 1000, 10000, 100000):
            if cells * features in SIZES:
                shapes.append((cells, features, cells * features))
    return shapes


def dense_medians(root, file_name):
    """Median over the shapes of each matrix size, in seconds."""
    buckets = {}
    for size in SIZES:
        buckets[size] = []
    for cells, features, size in dense_shapes():
        value = mean_us(f"{root}/{cells}_cells_{features}_features/{file_name}")
        if value == value:
            buckets[size].append(value)
    medians = []
    for size in SIZES:
        if buckets[size]:
            medians.append(seconds(np.median(buckets[size])))
        else:
            medians.append(np.nan)
    return medians


def collect_dense(gades1_dense, gades2_dense):
    result = {
        "amap": dense_medians(gades1_dense, "_amap_kendall.csv"),
        "1.0 GPU": dense_medians(gades1_dense, "_GPU_kendall.csv"),
        "1.0 CPU": dense_medians(gades1_dense, "_CPU_kendall.csv"),
        "2.0 GPU": dense_medians(gades2_dense, "dense_GPU_kendall.csv"),
        "2.0 CPU": dense_medians(gades2_dense, "dense_CPU_kendall.csv"),
    }
    for name, size in DENSE_TIMED_OUT:
        index = SIZES.index(size)
        if result[name][index] != result[name][index]:
            result[name][index] = TIMEOUT_S
    return result


def sparse_times(directory, template):
    times = []
    for sparsity in SPARSITIES:
        times.append(seconds(mean_us(f"{directory}/{template.format(sparsity=sparsity)}")))
    return times


def collect_sparse(gades1_sparse, gades2_sparse):
    old = f"{gades1_sparse}/{SPARSE_SHAPE}"
    new = f"{gades2_sparse}/{SPARSE_SHAPE}"
    return {
        "1.0 GPU": sparse_times(old, "{sparsity}_GPU_kendall.csv"),
        "1.0 CPU": sparse_times(old, "{sparsity}_CPU_kendall.csv"),
        "2.0 GPU": sparse_times(new, "{sparsity}/sparse_GPU_kendall.csv"),
        "2.0 CPU": sparse_times(new, "{sparsity}/sparse_CPU_kendall.csv"),
    }


def collect_real(gades1_real, gades2_real):
    series = {}
    for name in ("1.0 GPU", "1.0 CPU", "2.0 GPU", "2.0 CPU"):
        series[name] = []
    for dataset in DATASETS:
        for hardware in ("GPU", "CPU"):
            for version, root in (("1.0", gades1_real), ("2.0", gades2_real)):
                value = mean_us(f"{root}/{dataset}/sparse_{hardware}_kendall.csv")
                # a missing result means the run did not finish within 24 hours
                if value != value:
                    value = TIMEOUT_US
                series[f"{version} {hardware}"].append(np.log10(value))
    return series


def draw_panel(ax, groups, series, order, log_scale, xlabel, ylabel, xticklabels, rotation=0):
    count = len(order)
    positions = np.arange(len(groups))
    width = 0.82 / count

    for index, name in enumerate(order):
        heights = []
        for value in series[name]:
            if value == value:
                heights.append(value)
            else:
                heights.append(0)
        offset = (index - (count - 1) / 2) * width
        bars = ax.bar(positions + offset, heights, width, label=name)
        for bar in bars:
            style_bar(bar, name)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(positions)
    if rotation:
        alignment = "right"
        rotation_mode = "anchor"
    else:
        alignment = "center"
        rotation_mode = None
    ax.set_xticklabels(
        xticklabels, rotation=rotation, ha=alignment,
        rotation_mode=rotation_mode, fontsize=FS_TICK,
    )
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    sns.despine(ax=ax)


def main():
    parser = argparse.ArgumentParser(description="GADES 1.0 versus 2.0 on Kendall.")
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

    dense = collect_dense(f"{gades1}/GeneratedDense", args.generated_dense)
    sparse = collect_sparse(f"{gades1}/GeneratedSparse", args.generated_sparse)
    real = collect_real(f"{gades1}/RealDatasets", args.real_results)

    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(34, 26))
    top = GridSpec(
        1, 2, figure=fig, left=0.075, right=0.985,
        top=0.905, bottom=0.58, wspace=0.20,
    )
    bottom = GridSpec(
        1, 1, figure=fig, left=0.075, right=0.985,
        top=0.48, bottom=0.115,
    )

    ax_a = fig.add_subplot(top[0, 0])
    draw_panel(
        ax_a, SIZES, dense, ["amap", "1.0 GPU", "1.0 CPU", "2.0 GPU", "2.0 CPU"],
        True, "|W| (matrix elements)", "Kendall time, s (log)",
        [r"$10^{5}$", r"$10^{6}$", r"$10^{7}$"],
    )
    ax_a.set_title("Generated dense", fontsize=FS_TITLE, fontweight="bold", pad=14)
    ax_a.axhline(TIMEOUT_S, color="#666666", linestyle="--", linewidth=2.0)
    ax_a.text(
        len(SIZES) - 0.55, TIMEOUT_S, "24 h limit", ha="right", va="bottom",
        fontsize=FS_TICK, color="#666666",
    )

    sparsity_labels = []
    for sparsity in SPARSITIES:
        sparsity_labels.append(str(sparsity))
    ax_b = fig.add_subplot(top[0, 1])
    draw_panel(
        ax_b, SPARSITIES, sparse, ["1.0 GPU", "1.0 CPU", "2.0 GPU", "2.0 CPU"],
        True, "Sparsity", "Kendall time, s (log)", sparsity_labels,
    )
    ax_b.set_title(
        r"Generated sparse, $10^{4}\times10^{3}$",
        fontsize=FS_TITLE, fontweight="bold", pad=14,
    )

    ceiling = np.log10(TIMEOUT_US)
    ax_c = fig.add_subplot(bottom[0, 0])
    draw_panel(
        ax_c, DATASETS, real, ["1.0 GPU", "1.0 CPU", "2.0 GPU", "2.0 CPU"],
        False, "", r"Log10 Kendall time, $\mu$s", DATASETS, rotation=57,
    )
    ax_c.set_title("Real datasets, RTX 3090", fontsize=FS_TITLE, fontweight="bold", pad=14)
    ax_c.axhline(ceiling, color="#666666", linestyle="--", linewidth=2.0)
    ax_c.text(
        len(DATASETS) - 0.4, ceiling, "1.0 timed out",
        ha="right", va="bottom", fontsize=FS_TICK, color="#666666",
    )

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.005),
        ncol=5, frameon=False, fontsize=FS_LEGEND,
    )

    fig.text(0.012, 0.900, "a", fontsize=FS_PANEL, fontweight="bold", va="top")
    fig.text(0.512, 0.900, "b", fontsize=FS_PANEL, fontweight="bold", va="top")
    fig.text(0.012, 0.505, "c", fontsize=FS_PANEL, fontweight="bold", va="top")

    out = out_dir / "figure_positioning_canon.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    for name, values in real.items():
        capped = 0
        for value in values:
            if abs(value - ceiling) < 1e-9:
                capped += 1
        if capped:
            print(f"  panel c, {name}: 24 h limit shown for {capped} datasets")


if __name__ == "__main__":
    main()
