#!/usr/bin/env python3
"""Size and density of the real datasets.

    python3 reproducibility/figure_datasets_scatter.py

Writes datasets_scatter.png: the number of matrix elements against the share of non-zero
elements, one point per dataset. The numbers are listed in this file.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

# name, rows, columns, density, category
DATASETS = [
    ("PBMC3K", 13714, 2638, 0.061882, "scRNA"),
    ("PBMC3K B/T", 13714, 1806, 0.057955, "scRNA"),
    ("PBMC3K B/CD8T", 13714, 623, 0.055540, "scRNA"),
    ("HLCA: marrow", 23341, 5037, 0.134850, "scRNA"),
    ("HLCA: aorta", 23341, 408, 0.107794, "scRNA"),
    ("HLCA: lung", 23341, 1716, 0.107782, "scRNA"),
    ("HumanCortex", 18927, 734, 0.198877, "scRNA"),
    ("FibrocardRNA", 26128, 27998, 0.050035, "scRNA"),
    ("MouseHypothalamus", 23284, 14437, 0.066434, "scRNA"),
    ("HSC", 237450, 2034, 0.028230, "scATAC"),
    ("PBMC5K", 106935, 10032, 0.067310, "scATAC"),
    ("TCells", 49344, 765, 0.033921, "scATAC"),
    ("CellLines", 125647, 1224, 0.035873, "scATAC"),
    ("Anime", 69600, 9927, 0.009172, "recsys"),
    ("BeerAdvocate", 33388, 66055, 0.000713, "recsys"),
    ("Jester", 23500, 100, 0.727231, "recsys"),
    ("LastFM", 1892, 17632, 0.002783, "recsys"),
    ("ModCloth", 47958, 1378, 0.001246, "recsys"),
    ("MovieLens20M", 138493, 26744, 0.005400, "recsys"),
    ("Pinterest", 55187, 9916, 0.002675, "recsys"),
    ("RateBeer", 29265, 110369, 0.000884, "recsys"),
    ("TaFeng", 32266, 23812, 0.000967, "recsys"),
]

CATEGORIES = ["scRNA", "scATAC", "recsys"]
COLORS = {"recsys": "#D62728", "scATAC": "#1F77B4", "scRNA": "#2CA02C"}
LABELS = {"recsys": "Recommender Systems", "scATAC": "scATAC-seq", "scRNA": "scRNA-seq"}

# label offsets in points for the names that would overlap at the default position
DEFAULT_OFFSET = (8, 5)
OFFSETS = {
    "PBMC3K B/T": (8, -12),
    "PBMC3K B/CD8T": (-5, -12),
    "HLCA: aorta": (8, -12),
    "HLCA: lung": (8, 8),
    "LastFM": (8, -12),
    "ModCloth": (8, -12),
}


def main():
    parser = argparse.ArgumentParser(description="Size and density of the real datasets.")
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    for category in CATEGORIES:
        names = []
        sizes = []
        densities = []
        for name, rows, columns, density, dataset_category in DATASETS:
            if dataset_category == category:
                names.append(name)
                sizes.append(rows * columns)
                densities.append(density)
        ax.scatter(
            sizes, densities, c=COLORS[category], label=LABELS[category],
            s=170, edgecolors="black", linewidths=0.9, alpha=0.85, zorder=3,
        )
        for name, size, density in zip(names, sizes, densities):
            ax.annotate(
                name, (size, density), fontsize=12,
                xytext=OFFSETS.get(name, DEFAULT_OFFSET), textcoords="offset points",
                ha="left", va="bottom", alpha=0.8,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dimensionality", fontsize=17)
    ax.set_ylabel("Density", fontsize=17)
    ax.set_title("Datasets", fontsize=19)
    ax.legend(fontsize=15, loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", ls="--", alpha=0.3, zorder=0)
    ax.tick_params(axis="both", which="major", labelsize=14)

    fig.tight_layout()
    out = out_dir / "datasets_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
