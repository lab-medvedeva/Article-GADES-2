#!/usr/bin/env python3
"""Combined figure for the ten largest real datasets.

    python3 reproducibility/figure_huge_combined.py

Reads reproducibility/data/real_datasets_failures.csv (datasets that did not finish, by
cause) and reproducibility/data/superbig_speedup.csv (speedup of the finished runs against
GADES-GPU-dense). Writes figure_huge_combined.png and figure_huge_combined.pdf.

Panel (a) shows for each method how many datasets did not finish: the outline marks all
datasets, the fill pattern tells the cause. Panel (b) shows log10 of the speedup of the
finished runs. A method that supports a metric on no dataset is left out of panel (b)
for that metric.

Colour is the method, filled boxes are GADES and hollow boxes are baselines, hatched boxes
in panel (b) ran on the GPU.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "reproducibility" / "data"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

COL = {
    "armadillo": "#9467bd",
    "arrayfire": "#e377c2",
    "raft": "#023e8a",
    "GADES-CPU-sparse": "#f48c06",
    "GADES-GPU-dense": "#6a040f",
    "GADES-GPU-sparse": "#d00000",
}
GPU_METHODS = {"arrayfire", "raft", "GADES-GPU-dense", "GADES-GPU-sparse"}
METHOD_ORDER = [
    "armadillo", "arrayfire", "raft",
    "GADES-CPU-sparse", "GADES-GPU-dense", "GADES-GPU-sparse",
]
METRICS = ["euclidean", "cosine", "pearson", "manhattan", "spearman", "kendall"]

# the speedup table names the baselines with the mode suffix
ALIASES = {
    "armadillo-sparse": "armadillo",
    "arrayfire-sparse": "arrayfire",
    "raft-sparse": "raft",
}

# reference method of panel (b): its speedup is 1 by construction
BASELINE_METHOD = "GADES-GPU-dense"

HATCH = "///"
LW_HOLLOW = 2.8
LW_SOLID = 1.6

# panel (a) encodes the cause of a failure with the fill pattern; the GPU hatch of
# panel (b) is not reused here
FRAME_COLOR = "#9a9a9a"
FAIL_REASONS = ["time_limit", "memory_limit", "unavailable"]
FAIL_HATCH = {"time_limit": "", "memory_limit": "xxx", "unavailable": "..."}
FAIL_TITLES = {
    "time_limit": "time limit",
    "memory_limit": "memory limit",
    "unavailable": "metric unsupported",
}

FS_PANEL = 68
FS_PANEL_TITLE = 38
FS_TITLE = 50
FS_LABEL = 36
FS_TICK = 27
FS_LEGEND = 30

SPEEDUP_LABEL = "Log10 acceleration\n" + r"($T_{\rm GADES-GPU-dense}/T_{\rm method}$)"

matplotlib.rcParams["hatch.linewidth"] = 2.0


def luminance(color):
    red, green, blue = mcolors.to_rgb(color)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def edge_for(method):
    """A dark fill needs a light outline, otherwise the hatch disappears."""
    if luminance(COL[method]) < 0.45:
        return "#ffffff"
    return "#222222"


def style_box(patch, method):
    on_gpu = method in GPU_METHODS
    if method.startswith("GADES"):
        patch.set_facecolor(COL[method])
        patch.set_linewidth(LW_SOLID)
        if on_gpu:
            patch.set_edgecolor(edge_for(method))
        else:
            patch.set_edgecolor("#222222")
    else:
        patch.set_facecolor("none")
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


def draw_failures(ax, frame, total, show_ylabel):
    """One bar per method: the outline marks all datasets, the fill marks failures."""
    present = keep_order(METHOD_ORDER, set(frame["method"].values))
    positions = np.arange(len(present))

    # thin frame of full height: the reference of "out of N datasets"
    ax.bar(
        positions, [total] * len(present), facecolor="none",
        edgecolor=FRAME_COLOR, linewidth=1.2, zorder=1,
    )

    bottom = np.zeros(len(present))
    for reason in FAIL_REASONS:
        heights = []
        for method in present:
            rows = frame[frame["method"] == method]
            if len(rows):
                heights.append(float(rows[reason].values[0]))
            else:
                heights.append(0.0)
        heights = np.array(heights)
        for position, height, base, method in zip(positions, heights, bottom, present):
            if height <= 0:
                continue
            if method.startswith("GADES"):
                face = COL[method]
                edge = edge_for(method)
                width = LW_SOLID
            else:
                # a baseline segment without a pattern gets a light fill to stay visible
                if FAIL_HATCH[reason]:
                    alpha = 0.0
                else:
                    alpha = 0.30
                face = mcolors.to_rgba(COL[method], alpha)
                edge = COL[method]
                width = LW_HOLLOW
            bar = ax.bar(
                position, height, bottom=base, facecolor=face,
                edgecolor=edge, linewidth=width, zorder=2,
            )[0]
            if FAIL_HATCH[reason]:
                bar.set_hatch(FAIL_HATCH[reason])
        bottom = bottom + heights

    ax.set_xticks(positions)
    ax.set_xticklabels(present, rotation=57, ha="right", rotation_mode="anchor", fontsize=FS_TICK)
    ax.set_ylim(0, total)
    ax.set_yticks(range(0, total + 1, 2))
    ax.tick_params(axis="y", labelsize=FS_TICK)
    if show_ylabel:
        ax.set_ylabel("Datasets", fontsize=FS_LABEL)
    else:
        ax.set_ylabel("", fontsize=FS_LABEL)
    sns.despine(ax=ax)


def draw_speedup(ax, frame, show_ylabel, unsupported=()):
    """Box plot of the log10 speedup against GADES-GPU-dense."""
    frame = frame[frame["method"] != BASELINE_METHOD]
    if len(unsupported):
        frame = frame[~frame["method"].isin(unsupported)]
    present = keep_order(METHOD_ORDER, set(frame["method"].values))
    if not present:
        ax.set_axis_off()
        return

    frame = frame.copy()
    frame["log_speedup"] = np.log10(frame["speedup"])
    palette = []
    for method in present:
        palette.append(COL[method])
    sns.boxplot(
        data=frame, x="method", y="log_speedup", order=present,
        hue="method", hue_order=present, legend=False,
        palette=palette, saturation=1, width=0.6, fliersize=3, ax=ax,
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
        style_box(patch, nearest)

    ax.axhline(0, color="#D55E00", linestyle="--", linewidth=2.2, zorder=5)
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels(present, rotation=57, ha="right", rotation_mode="anchor", fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(SPEEDUP_LABEL, fontsize=FS_LABEL)
    else:
        ax.set_ylabel("", fontsize=FS_LABEL)
    sns.despine(ax=ax)


def failure_handles():
    """Legend of panel (a): the fill pattern of each failure cause."""
    handles = []
    for reason in FAIL_REASONS:
        handle = Patch(
            facecolor="none", edgecolor="#222222", linewidth=1.6,
            hatch=FAIL_HATCH[reason], label=FAIL_TITLES[reason],
        )
        handles.append(handle)
    return handles


def read_table(path):
    if not Path(path).exists():
        raise SystemExit(f"table not found: {path}")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Combined figure for the largest real datasets.")
    parser.add_argument("--failures", default=str(DATA_DIR / "real_datasets_failures.csv"))
    parser.add_argument("--speedup", default=str(DATA_DIR / "superbig_speedup.csv"))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    parser.add_argument("--panel-titles", action="store_true", help="draw a title above each panel")
    args = parser.parse_args()

    failures = read_table(args.failures)
    speedup = read_table(args.speedup)
    failures["method"] = failures["method"].replace(ALIASES)
    speedup["method"] = speedup["method"].replace(ALIASES)
    total = int(failures["total_datasets"].max())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # a metric that a method supports on no dataset
    unsupported = {}
    for _, row in failures.iterrows():
        if row["unavailable"] >= row["total_datasets"]:
            unsupported.setdefault(row["metric"], set()).add(row["method"])

    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(34, 33))
    # panel (a) leaves a strip on the right for its legend, panel (b) takes the full width
    if args.panel_titles:
        gap = 0.085
        top = 0.925
        legend_top = 0.955
    else:
        gap = 0.065
        top = 0.955
        legend_top = 0.985
    middle = 0.50
    grid_a = GridSpec(
        2, 3, figure=fig, left=0.075, right=0.875,
        top=top, bottom=middle + gap, hspace=0.88, wspace=0.28,
    )
    grid_b = GridSpec(
        2, 3, figure=fig, left=0.105, right=0.985,
        top=middle - gap, bottom=0.035, hspace=0.88, wspace=0.28,
    )

    for index, metric in enumerate(METRICS):
        row, col = divmod(index, 3)
        ax_a = fig.add_subplot(grid_a[row, col])
        draw_failures(ax_a, failures[failures["metric"] == metric], total, col == 0)
        ax_a.set_title(metric.capitalize(), fontsize=FS_TITLE, fontweight="bold", pad=12)

        ax_b = fig.add_subplot(grid_b[row, col])
        dropped = unsupported.get(metric, set())
        draw_speedup(ax_b, speedup[speedup["metric"] == metric], col == 0, dropped)
        ax_b.set_title(metric.capitalize(), fontsize=FS_TITLE, fontweight="bold", pad=12)

    fig.legend(
        handles=failure_handles(), loc="upper left", bbox_to_anchor=(0.885, legend_top),
        ncol=1, frameon=False, fontsize=FS_LEGEND,
    )

    if args.panel_titles:
        separator = Line2D(
            [0.02, 0.98], [0.500, 0.500], transform=fig.transFigure,
            color="#c8c8c8", linewidth=2.0,
        )
        fig.add_artist(separator)

    fig.text(0.015, 0.972, "a", fontsize=FS_PANEL, fontweight="bold", va="top")
    fig.text(0.015, 0.452, "b", fontsize=FS_PANEL, fontweight="bold", va="top")

    if args.panel_titles:
        title_a_start = f"Datasets that did not finish, out of {total} "
        title_a_end = "(fill pattern shows the cause; hollow bars are baselines)"
        title_b = "Acceleration relative to GADES-GPU-dense (dashed line: reference = 1.0)"
        fig.text(
            0.5, 0.962, title_a_start + title_a_end,
            fontsize=FS_PANEL_TITLE, va="top", ha="center",
        )
        fig.text(0.5, 0.458, title_b, fontsize=FS_PANEL_TITLE, va="top", ha="center")

    out = out_dir / "figure_huge_combined.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    print(f"metrics: {len(METRICS)}, methods: {len(METHOD_ORDER)}, datasets: {total}")
    for metric in METRICS:
        present = set(speedup[speedup["metric"] == metric]["method"])
        hidden = []
        for method in sorted(unsupported.get(metric, set())):
            if method in present:
                hidden.append(method)
        if hidden:
            print(f"  panel b, {metric}: left out as unsupported -> {', '.join(hidden)}")


if __name__ == "__main__":
    main()
