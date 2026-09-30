#!/usr/bin/env python3
"""Acceleration on the generated datasets relative to the dense GPU mode of GADES.

    python3 reproducibility/figure_generated_acceleration.py

Reads the tables written by collect_generated_dense.py and collect_generated_sparse.py.
Writes figure3_dense_accel.png (dense data against the matrix size) and, for every matrix
size, figure4_sparse_accel_W<k>.png (sparse data against sparsity, sparse-capable methods)
and figure4_sparse_accel_full_W<k>.png (the same with the dense-only baselines).

y = log10(T_GADES-GPU-dense / T_method): a method above zero is faster than GADES-GPU-dense.
Colour is the method, filled boxes are GADES and hollow boxes are baselines, hatched boxes
ran on the GPU. A method without a result has no box. Dense methods do not depend on
sparsity and are repeated at every sparsity level.

The benchmark method `pythonic` is shown as `scipy` with its time multiplied by 24.
Spearman of ArrayFire and Armadillo is left out: both rank without averaging ties.
Of raft, rapids and cupy only raft is drawn in the sparse figures.
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
from matplotlib.patches import Patch, PathPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "reproducibility" / "data"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

SIZES = [10 ** 5, 10 ** 6, 10 ** 7]
SIZE_LABELS = {10 ** 5: "$10^5$", 10 ** 6: "$10^6$", 10 ** 7: "$10^7$"}
SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.99]
METRICS = ["euclidean", "cosine", "pearson", "manhattan", "spearman", "kendall"]
GRID_ROWS = 3
GRID_COLUMNS = 2
FIGURE_SIZE = (20, 21)
PYTHON_TIME_FACTOR = 24

# baselines take cool and neutral hues, GADES takes the warm ones
COL = {
    "scipy": "#1f77b4",
    "pandas": "#17becf",
    "factoextra": "#2ca02c",
    "amap": "#8c564b",
    "armadillo": "#9467bd",
    "arrayfirecuda": "#e377c2",
    "arrayfirecpu": "#c5b0d5",
    "raft": "#023e8a",
    "GADES-CPU-dense": "#e85d04",
    "GADES-GPU-dense": "#6a040f",
    "GADES-CPU-sparse": "#f48c06",
    "GADES-GPU-sparse": "#d00000",
}
GPU_METHODS = {"arrayfirecuda", "raft", "GADES-GPU-dense", "GADES-GPU-sparse"}

DENSE_METHODS = [
    "scipy", "pandas", "factoextra", "amap", "armadillo", "arrayfirecuda", "arrayfirecpu",
    "raft", "GADES-CPU-dense", "GADES-GPU-dense",
]
SPARSE_METHODS = [
    "armadillo", "arrayfirecuda", "raft",
    "GADES-GPU-dense", "GADES-CPU-sparse", "GADES-GPU-sparse",
]
SPARSE_METHODS_FULL = [
    "scipy", "pandas", "factoextra", "amap", "armadillo", "arrayfirecuda", "raft",
    "GADES-GPU-dense", "GADES-CPU-sparse", "GADES-GPU-sparse",
]
NO_TIE_AVERAGING = ["arrayfirecuda", "arrayfirecpu", "armadillo"]

HATCH = "/"
LW_HOLLOW = 2.8
LW_SOLID = 1.6
Y_LABEL = "Log10 acceleration\n($T_{\\mathrm{GADES\\text{-}GPU\\text{-}dense}}/T_{\\mathrm{method}}$)"

matplotlib.rcParams["hatch.linewidth"] = 2.0


def is_gpu(method):
    return method in GPU_METHODS


def is_gades(method):
    return method.startswith("GADES")


def rgb(color):
    return np.array(mcolors.to_rgb(color))


def luminance(color):
    red, green, blue = rgb(color)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def edge_for(method):
    """A dark fill needs a light outline, otherwise the hatch disappears."""
    if luminance(COL[method]) < 0.45:
        return "#ffffff"
    return "#222222"


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


def style_boxes(ax, present):
    """Hollow baselines, solid GADES, hatch for every GPU method.

    Boxes are matched to methods by the nearest palette colour.
    """
    references = {}
    for method in present:
        references[method] = rgb(COL[method])

    def distance(method, face):
        return np.sum((references[method] - face) ** 2)

    for patch in ax.findobj(PathPatch):
        face = np.array(patch.get_facecolor()[:3])
        nearest = min(references, key=lambda method: distance(method, face))
        if distance(nearest, face) > 0.25:
            continue
        if is_gades(nearest):
            patch.set_facecolor(COL[nearest])
            patch.set_edgecolor(edge_for(nearest))
            patch.set_linewidth(LW_SOLID)
        else:
            patch.set_facecolor("none")
            patch.set_edgecolor(COL[nearest])
            patch.set_linewidth(LW_HOLLOW)
        if is_gpu(nearest):
            patch.set_hatch(HATCH)


def method_handles(methods):
    handles = []
    for method in methods:
        hatch = None
        if is_gpu(method):
            hatch = HATCH
        if is_gades(method):
            handle = Patch(
                facecolor=COL[method], edgecolor=edge_for(method),
                linewidth=LW_SOLID, hatch=hatch, label=method,
            )
        else:
            handle = Patch(
                facecolor="none", edgecolor=COL[method],
                linewidth=LW_HOLLOW, hatch=hatch, label=method,
            )
        handles.append(handle)
    return handles


def encoding_handles():
    grey = "#dddddd"
    dark = "#222222"
    return [
        Patch(facecolor=grey, edgecolor=dark, linewidth=1.4, hatch=HATCH, label="GPU (hatched)"),
        Patch(facecolor=grey, edgecolor=dark, linewidth=1.4, label="CPU (plain)"),
        Patch(facecolor=grey, edgecolor=dark, linewidth=LW_SOLID, label="GADES (filled)"),
        Patch(facecolor="none", edgecolor=dark, linewidth=LW_HOLLOW, label="baseline (hollow)"),
    ]


def add_legends(fig, methods):
    methods_legend = fig.legend(
        handles=method_handles(methods), loc="upper center", ncol=5,
        fontsize=16, bbox_to_anchor=(.5, 1.055), frameon=False,
    )
    fig.add_artist(methods_legend)
    fig.legend(
        handles=encoding_handles(), loc="upper center", ncol=4,
        fontsize=13, bbox_to_anchor=(.5, 1.012), frameon=False,
    )


def read_table(path, collector):
    if not Path(path).exists():
        raise SystemExit(f"table not found: {path}; run {collector} first")
    return pd.read_csv(path)


def load_dense(path):
    """Dense table with the Python loop renamed to scipy and scaled."""
    table = read_table(path, "collect_generated_dense.py")
    table = table[["metric", "method", "wX", "wY", "|W|", "mean_sequential"]].dropna().copy()
    is_python = table["method"] == "pythonic"
    table.loc[is_python, "mean_sequential"] *= PYTHON_TIME_FACTOR
    table.loc[is_python, "method"] = "scipy"
    return table


def draw_panels(axes, frame, methods, x_column, x_values, x_label):
    """One box plot per metric; a metric without data keeps an empty panel."""
    for ax, metric in zip(axes.flat, METRICS):
        subset = frame[(frame["metric"] == metric) & frame["method"].isin(methods)]
        present = keep_order(methods, set(subset["method"]))
        x_order = keep_order(x_values, set(subset[x_column]))
        if present and x_order:
            sns.boxplot(
                ax=ax, data=subset, x=x_column, y="logA", hue="method",
                hue_order=present, order=x_order, palette=palette_for(present),
                fliersize=1.5, saturation=1,
            )
            if ax.get_legend():
                ax.get_legend().remove()
            style_boxes(ax, present)
        ax.axhline(0, ls="--", c="crimson", lw=1.3)
        ax.set_title(metric.capitalize(), fontsize=24)
        ax.set_xlabel(x_label, fontsize=17)
        ax.set_ylabel(Y_LABEL, fontsize=14)
        ax.tick_params(labelsize=16)
        ax.yaxis.grid(True, alpha=.4)


def draw_dense(dense, reference, out_dir):
    frame = dense.merge(reference, on=["metric", "wX", "wY"], how="inner")
    frame["logA"] = np.log10(frame["base"] / frame["mean_sequential"])
    frame["method"] = frame["method"].replace({"CPU": "GADES-CPU-dense", "GPU": "GADES-GPU-dense"})
    frame = frame[frame["|W|"].isin(SIZES)].copy()
    frame["Wlab"] = frame["|W|"].map(SIZE_LABELS)

    methods = keep_order(DENSE_METHODS, set(frame["method"]))
    size_labels = ["$10^5$", "$10^6$", "$10^7$"]
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLUMNS, figsize=FIGURE_SIZE)
    draw_panels(axes, frame, methods, "Wlab", size_labels, "$|W|$")
    add_legends(fig, methods)
    fig.tight_layout(rect=[0, 0, 1, .97])
    out = out_dir / "figure3_dense_accel.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def combine_sparse(sparse_path, dense_path, reference):
    """Sparse methods plus the dense baselines repeated at every sparsity level."""
    columns = ["metric", "method", "sparsity", "wX", "wY", "|W|", "mean_sequential"]
    sparse = read_table(sparse_path, "collect_generated_sparse.py")[columns].dropna().copy()
    sparse["method"] = sparse["method"].replace({"CPU": "GADES-CPU-sparse", "GPU": "GADES-GPU-sparse"})
    sparse_kept = ["GADES-CPU-sparse", "GADES-GPU-sparse", "raft", "armadillo", "arrayfirecuda"]
    sparse = sparse[sparse["method"].isin(sparse_kept)]
    no_ties = (sparse["metric"] == "spearman") & sparse["method"].isin(NO_TIE_AVERAGING)
    sparse = sparse[~no_ties].copy()
    sparse["time"] = sparse["mean_sequential"]

    dense = load_dense(dense_path)
    dense = dense[dense["method"].isin(["amap", "scipy", "pandas", "factoextra", "GPU"])].copy()
    dense["method"] = dense["method"].replace({"GPU": "GADES-GPU-dense"})
    repeated = []
    for sparsity in SPARSITIES:
        level = dense.copy()
        level["sparsity"] = sparsity
        level["time"] = level["mean_sequential"]
        repeated.append(level)
    dense = pd.concat(repeated, ignore_index=True)

    columns = ["metric", "method", "sparsity", "wX", "wY", "|W|", "time"]
    combined = pd.concat([sparse[columns], dense[columns]], ignore_index=True)
    combined = combined.merge(reference, on=["metric", "wX", "wY"], how="inner")
    combined["logA"] = np.log10(combined["base"] / combined["time"])
    return combined[combined["|W|"].isin(SIZES)]


def draw_sparse(combined, preferred, suffix, out_dir):
    methods = keep_order(preferred, set(combined["method"]))
    for size in SIZES:
        frame = combined[combined["|W|"] == size]
        fig, axes = plt.subplots(GRID_ROWS, GRID_COLUMNS, figsize=FIGURE_SIZE)
        draw_panels(axes, frame, methods, "sparsity", SPARSITIES, "sparsity")
        add_legends(fig, methods)
        title = f"Sparse log10 acceleration \u2014 $|W|$ = {SIZE_LABELS[size]}"
        fig.suptitle(title, y=1.095, fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, .96])
        out = out_dir / f"figure4_sparse_accel{suffix}_W{int(np.log10(size))}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Acceleration figures of the generated datasets.")
    parser.add_argument("--dense-table", default=str(DATA_DIR / "generated_dense.csv"))
    parser.add_argument("--sparse-table", default=str(DATA_DIR / "generated_sparse.csv"))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dense = load_dense(args.dense_table)
    no_ties = (dense["metric"] == "spearman") & dense["method"].isin(NO_TIE_AVERAGING)
    dense = dense[~no_ties]
    gpu_dense = dense[dense["method"] == "GPU"]
    reference = gpu_dense.groupby(["metric", "wX", "wY"])["mean_sequential"].median()
    reference = reference.rename("base").reset_index()

    draw_dense(dense, reference, out_dir)
    combined = combine_sparse(args.sparse_table, args.dense_table, reference)
    draw_sparse(combined, SPARSE_METHODS, "", out_dir)
    draw_sparse(combined, SPARSE_METHODS_FULL, "_full", out_dir)


if __name__ == "__main__":
    main()
