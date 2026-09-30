#!/usr/bin/env python3
"""Combined figure for the 22 real datasets.

    python3 reproducibility/figure_real22_combined.py [--log-speedup]

Reads the two tables written by collect_real_results.py and writes
figure_real22_combined.png and figure_real22_combined.pdf.

Panel (a) shows how many datasets each implementation did not finish within the time
limit. Panel (b) shows the speedup of the finished runs against GADES-GPU-dense. A method
that does not support a metric is left out of both panels for that metric. Spearman of
ArrayFire and Armadillo is left out of panel (b): both rank without averaging ties, so
their times are not comparable.

Without --log-speedup the acceleration axis of panel (b) is linear and cut at a fixed
upper bound per metric; with it the axis shows log10 of the acceleration in full.

Colour is the method (dense and sparse modes share a hue, sparse is lighter), filled boxes
are GADES and hollow boxes are baselines, hatched boxes ran on the GPU.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import PathPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "reproducibility" / "data"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

COL = {
    "amap": "#8c564b",
    "factoextra": "#2ca02c",
    "scipy": "#1f77b4",
    "pandas": "#17becf",
    "armadillo-dense": "#7b4ea3",
    "armadillo-sparse": "#c5b0d5",
    "arrayfire-dense": "#d1479e",
    "arrayfire-sparse": "#f7b6d2",
    "raft-dense": "#023e8a",
    "raft-sparse": "#6d9ed6",
    "GADES-CPU-dense": "#e85d04",
    "GADES-CPU-sparse": "#f48c06",
    "GADES-GPU-dense": "#6a040f",
    "GADES-GPU-sparse": "#d00000",
}
GPU_METHODS = {
    "arrayfire-dense", "arrayfire-sparse", "raft-dense", "raft-sparse",
    "GADES-GPU-dense", "GADES-GPU-sparse",
}
METHOD_ORDER = [
    "amap", "factoextra", "scipy", "pandas",
    "armadillo-dense", "armadillo-sparse",
    "arrayfire-dense", "arrayfire-sparse",
    "raft-dense", "raft-sparse",
    "GADES-CPU-dense", "GADES-CPU-sparse",
    "GADES-GPU-dense", "GADES-GPU-sparse",
]
METRICS = ["euclidean", "cosine", "manhattan", "pearson", "spearman", "kendall"]

BASELINE_METHOD = "GADES-GPU-dense"

# upper bounds of the linear acceleration axis: single outliers would stretch the scale
# by orders of magnitude and flatten the boxes
YLIM = {
    "euclidean": 4,
    "cosine": 5,
    "manhattan": 5,
    "pearson": 4,
    "spearman": 6,
    "kendall": 16,
}

# left out of panel (b): the metric is supported, but the result is not comparable by time
EXCLUDE_FROM_SPEEDUP = {
    "spearman": {"armadillo-dense", "armadillo-sparse", "arrayfire-dense", "arrayfire-sparse"},
}

HATCH = "///"
LW_HOLLOW = 2.2
LW_SOLID = 1.4
FRAME_COLOR = "#9a9a9a"

FS_PANEL = 68
FS_PANEL_TITLE = 38
FS_TITLE = 50
FS_LABEL = 36
FS_TICK = 27

matplotlib.rcParams["hatch.linewidth"] = 2.0


def luminance(color):
    red, green, blue = mcolors.to_rgb(color)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def edge_for(method):
    """A dark fill needs a light outline, otherwise the hatch disappears."""
    if luminance(COL[method]) < 0.45:
        return "#ffffff"
    return "#222222"


def style_patch(patch, method):
    on_gpu = method in GPU_METHODS
    if method.startswith("GADES"):
        patch.set_facecolor(COL[method])
        patch.set_linewidth(LW_SOLID)
        if on_gpu:
            patch.set_edgecolor(edge_for(method))
        else:
            patch.set_edgecolor("#222222")
    else:
        patch.set_facecolor(mcolors.to_rgba(COL[method], 0.30))
        patch.set_edgecolor(COL[method])
        patch.set_linewidth(LW_HOLLOW)
    if on_gpu:
        patch.set_hatch(HATCH)


def keep_order(preferred, present):
    ordered = []
    for item in preferred:
        if item in present:
            ordered.append(item)
    return ordered


def squared_distance(first, second):
    total = 0.0
    for a, b in zip(first, second):
        total += (a - b) ** 2
    return total


def draw_walltimed(ax, frame, total, show_ylabel, unsupported=()):
    """One bar per method: how many datasets did not finish within the limit."""
    present = []
    for method in METHOD_ORDER:
        if method not in unsupported:
            present.append(method)
    positions = np.arange(len(present))
    ax.bar(
        positions, [total] * len(present), facecolor="none",
        edgecolor=FRAME_COLOR, linewidth=1.0, zorder=1,
    )

    for position, method in zip(positions, present):
        row = frame[frame["method"] == method]
        if not len(row):
            continue
        height = float(row["wall_timed"].iloc[0])
        if height <= 0:
            continue
        bar = ax.bar(position, height, zorder=2)[0]
        style_patch(bar, method)

    ax.set_xticks(positions)
    ax.set_xticklabels(present, rotation=57, ha="right", rotation_mode="anchor", fontsize=FS_TICK)
    ax.set_ylim(0, total)
    ax.set_yticks(range(0, total + 1, 4))
    ax.tick_params(axis="y", labelsize=FS_TICK)
    if show_ylabel:
        ax.set_ylabel("Wall-timed datasets", fontsize=FS_LABEL)
    else:
        ax.set_ylabel("", fontsize=FS_LABEL)
    sns.despine(ax=ax)


def draw_speedup(ax, frame, show_ylabel, unsupported=(), ylim=6, log=False):
    """Box plot of the speedup against GADES-GPU-dense.

    With log=True the value is log10 of the acceleration on an axis with integer ticks:
    -4 is 10 000 times slower, 0 is the reference, 1 is ten times faster.
    """
    if log:
        frame = frame[frame["speedup"] > 0]
        frame = frame.assign(speedup=np.log10(frame["speedup"]))
    frame = frame[frame["method"] != BASELINE_METHOD]
    if len(unsupported):
        frame = frame[~frame["method"].isin(unsupported)]
    present = keep_order(METHOD_ORDER, set(frame["method"].values))
    if not present:
        ax.set_axis_off()
        return

    palette = []
    for method in present:
        palette.append(COL[method])
    sns.boxplot(
        data=frame, x="method", y="speedup", order=present,
        hue="method", hue_order=present, legend=False,
        palette=palette, saturation=1, width=0.65, fliersize=2.5, ax=ax,
    )

    # boxes are matched to methods by the nearest palette colour
    references = {}
    for method in present:
        references[method] = mcolors.to_rgb(COL[method])
    for patch in ax.findobj(PathPatch):
        face = patch.get_facecolor()[:3]
        nearest = min(references, key=lambda method: squared_distance(references[method], face))
        if squared_distance(references[nearest], face) > 0.25:
            continue
        style_patch(patch, nearest)

    if log:
        # whole units that cover the data of the panel and the reference at zero
        low = int(np.floor(min(frame["speedup"].min(), 0.0)))
        high = int(np.ceil(max(frame["speedup"].max(), 0.0) + 0.01))
        ax.set_ylim(low, high)
        ax.set_yticks(range(low, high + 1))
        ax.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
        reference_level = 0.0
        label = r"$\log_{10}$ acceleration"
    else:
        ax.set_ylim(0, ylim)
        reference_level = 1.0
        label = "Acceleration (x)"

    ax.axhline(reference_level, color="#D55E00", linestyle="--", linewidth=2.0, zorder=5)
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels(present, rotation=57, ha="right", rotation_mode="anchor", fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(label, fontsize=FS_LABEL)
    else:
        ax.set_ylabel("", fontsize=FS_LABEL)
    sns.despine(ax=ax)


def read_table(path):
    if not Path(path).exists():
        raise SystemExit(f"table not found: {path}; run collect_real_results.py first")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Combined figure for the 22 real datasets.")
    parser.add_argument("--walltimed", default=str(DATA_DIR / "real_walltimed_22.csv"))
    parser.add_argument("--speedup", default=str(DATA_DIR / "real_speedup_22.csv"))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    parser.add_argument("--name", default="figure_real22_combined", help="name of the figure file without extension")
    parser.add_argument("--panel-titles", action="store_true", help="draw a title above each panel")
    parser.add_argument(
        "--log-speedup",
        action="store_true",
        help="logarithmic axis in panel (b) instead of the cut linear one",
    )
    args = parser.parse_args()

    walltimed = read_table(args.walltimed)
    speedup = read_table(args.speedup)
    total = int(walltimed["total_datasets"].max())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unsupported = {}
    for _, row in walltimed[walltimed["unsupported"] == 1].iterrows():
        unsupported.setdefault(row["metric"], set()).add(row["method"])

    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(34, 34))
    if args.panel_titles:
        gap = 0.105
        top = 0.925
    else:
        gap = 0.085
        top = 0.955
    middle = 0.50
    grid_a = GridSpec(
        2, 3, figure=fig, left=0.065, right=0.985,
        top=top, bottom=middle + gap, hspace=1.05, wspace=0.20,
    )
    grid_b = GridSpec(
        2, 3, figure=fig, left=0.085, right=0.985,
        top=middle - gap, bottom=0.045, hspace=1.05, wspace=0.20,
    )

    for index, metric in enumerate(METRICS):
        row, col = divmod(index, 3)
        hidden = unsupported.get(metric, set())

        ax_a = fig.add_subplot(grid_a[row, col])
        draw_walltimed(ax_a, walltimed[walltimed["metric"] == metric], total, col == 0, hidden)
        ax_a.set_title(metric.capitalize(), fontsize=FS_TITLE, fontweight="bold", pad=12)

        ax_b = fig.add_subplot(grid_b[row, col])
        hidden_b = hidden | EXCLUDE_FROM_SPEEDUP.get(metric, set())
        draw_speedup(
            ax_b, speedup[speedup["metric"] == metric], col == 0,
            hidden_b, YLIM.get(metric, 6), log=args.log_speedup,
        )
        ax_b.set_title(metric.capitalize(), fontsize=FS_TITLE, fontweight="bold", pad=12)

    fig.text(0.012, 0.972, "a", fontsize=FS_PANEL, fontweight="bold", va="top")
    fig.text(0.012, 0.432, "b", fontsize=FS_PANEL, fontweight="bold", va="top")

    if args.panel_titles:
        title_a = f"Datasets not finished within the wall-clock limit, out of {total}"
        title_b = "Acceleration relative to GADES-GPU-dense (dashed line: reference = 1.0)"
        fig.text(0.46, 0.962, title_a, fontsize=FS_PANEL_TITLE, va="top", ha="center")
        fig.text(0.5, 0.422, title_b, fontsize=FS_PANEL_TITLE, va="top", ha="center")

    out = out_dir / f"{args.name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    print(f"metrics: {len(METRICS)}, methods: {len(METHOD_ORDER)}, datasets: {total}")
    for metric in METRICS:
        hidden = sorted(unsupported.get(metric, set()))
        if hidden:
            print(f"  {metric}: left out of both panels -> {', '.join(hidden)}")
        manual = sorted(EXCLUDE_FROM_SPEEDUP.get(metric, set()))
        if manual:
            print(f"  {metric}: left out of panel b only -> {', '.join(manual)}")


if __name__ == "__main__":
    main()
