#!/usr/bin/env python3
"""Error of the Spearman distance on data with many tied zeros.

    python3 reproducibility/figure_spearman_ties.py

Reads reproducibility/data/spearman_ties_sweep.csv and writes spearman_ties_correctness.png:
the largest off-diagonal difference from R cor(method = "spearman") against the share of
zero entries. ArrayFire and Armadillo give the same error and share one curve, drawn from
the larger of the two. The table is the recorded output of
scripts/Validation/spearman_ties_sweep.R.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "reproducibility" / "data"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

COLOR_GADES = "#08519c"
COLOR_ORDINAL = "#d62728"
FLOOR = 1e-9


def main():
    parser = argparse.ArgumentParser(description="Spearman error against the share of zeros.")
    parser.add_argument("--sweep", default=str(DATA_DIR / "spearman_ties_sweep.csv"))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    if not Path(args.sweep).exists():
        raise SystemExit(f"table not found: {args.sweep}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = pd.read_csv(args.sweep)
    gades_error = np.maximum(sweep["gades_max"], FLOOR)
    ordinal_error = np.maximum(sweep[["af_max", "arma_max"]].max(axis=1), FLOOR)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(
        sweep["sparsity"], gades_error, "o-", color=COLOR_GADES,
        lw=2.4, ms=8, label="GADES (average ranks)",
    )
    ax.plot(
        sweep["sparsity"], ordinal_error, "s-", color=COLOR_ORDINAL,
        lw=2.4, ms=8, label="ArrayFire & Armadillo (ordinal ranks)",
    )
    ax.set_yscale("log")
    xlabel = "sparsity (fraction of zero entries \u2192 size of the zero-tie group)"
    ylabel = r"max $|D - D_{\mathrm{R\ cor(spearman)}}|$  (off-diagonal)"
    title_first = "Spearman zero-tie correctness vs the R reference (RTX 3090)"
    title_second = "GADES averages tied ranks (exact); ArrayFire & Armadillo assign ordinal ranks and drift"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title_first}\n{title_second}", fontsize=12)
    ax.axhline(1e-6, ls=":", c="gray", lw=1.2)
    ax.text(0.01, 1.3e-6, "float32 machine precision", fontsize=9, color="gray", va="bottom")
    ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=11, loc="center right")
    fig.tight_layout()
    out = out_dir / "spearman_ties_correctness.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
