#!/usr/bin/env python3
"""Collect the generated sparse benchmark into one table.

    python3 reproducibility/collect_generated_sparse.py

Reads results/GeneratedSparse/<C>_cells_<F>_features/<sparsity>/<prefix>_<method>_<metric>.csv:
one column of times in microseconds, one row per repetition. The prefix is `sparse` or the
sparsity value. Files of memory usage and the old layout without the sparsity directory are
skipped. Writes reproducibility/data/generated_sparse.csv with one row per result file.

The first repetition includes the one-off initialisation of the GPU and BLAS libraries,
so mean_sequential, sd, minTime and maxTime are computed without it.
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "results" / "GeneratedSparse"
DEFAULT_OUT = REPO_ROOT / "reproducibility" / "data" / "generated_sparse.csv"

METHODS = {"GPU", "CPU", "armadillo", "arrayfirecuda", "raft", "cupy", "rapids"}
METRICS = {"euclidean", "cosine", "pearson", "manhattan", "spearman", "kendall"}
METRIC_ALIASES = {"l1": "manhattan"}

DIRECTORY_PATTERN = re.compile(r"(?P<cells>\d+)_cells_(?P<features>\d+)_features")
SPARSITY_PATTERN = re.compile(r"^\d+\.\d+$")


def summarise(path, drop_first):
    """Return the table row of one result file, or None when the file is not a timing."""
    sparsity = os.path.basename(os.path.dirname(path))
    if not SPARSITY_PATTERN.match(sparsity):
        return None
    shape_directory = os.path.basename(os.path.dirname(os.path.dirname(path)))
    match = DIRECTORY_PATTERN.search(shape_directory)
    if not match:
        return None
    parts = os.path.basename(path)[:-4].split("_")
    if len(parts) < 2:
        return None
    method = parts[-2]
    metric = METRIC_ALIASES.get(parts[-1], parts[-1])
    if method not in METHODS or metric not in METRICS:
        return None
    cells = int(match.group("cells"))
    features = int(match.group("features"))

    frame = pd.read_csv(path)
    column = frame.columns[0]
    repetitions = frame.loc[:, column]
    first_two = frame.loc[:1, column]
    positive = repetitions[repetitions > 0]
    if len(positive) == 0:
        return None
    if drop_first and len(positive) > 1:
        steady = positive.iloc[1:]
    else:
        steady = positive

    if (first_two > 0).any():
        mean_reinit = np.mean(first_two[first_two > 0])
        log_mean_reinit = np.log10(mean_reinit)
    else:
        mean_reinit = np.nan
        log_mean_reinit = np.nan

    row = {
        "DataFile": Path(os.path.relpath(path, REPO_ROOT)).as_posix(),
        "metric": metric,
        "method": method,
        "sparsity": float(sparsity),
        "wX": cells,
        "wY": features,
        "ID": f"{cells}_{features}",
        "|W|": cells * features,
        "logWX": np.log10(cells),
        "logWY": np.log10(features),
        "logW": np.log10(cells),
        "logMeanTime": np.log10(np.mean(steady)),
        "logMeanTimeReinit": log_mean_reinit,
        "logMinTime": np.log10(np.min(steady)),
        "logMaxTime": np.log10(np.max(steady)),
        "minTime": np.min(steady),
        "maxTime": np.max(steady),
        "mean_reinit": mean_reinit,
        "mean_sequential": np.mean(steady),
        "sd": np.std(steady),
        "Method": f"{metric}_{method}",
    }
    for index, value in enumerate(repetitions.tolist()):
        if value > 0:
            row[f"val_{index + 1}"] = value
    return row


def main():
    parser = argparse.ArgumentParser(description="Collect the generated sparse benchmark.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="directory GeneratedSparse")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="table to write")
    parser.add_argument(
        "--keep-first-repetition",
        action="store_true",
        help="include the first repetition in the statistics",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results):
        raise SystemExit(f"results directory not found: {args.results}")

    pattern = os.path.join(args.results, "*_cells_*_features", "*", "*.csv")
    rows = []
    for path in sorted(glob.glob(pattern)):
        row = summarise(path, not args.keep_first_repetition)
        if row is not None:
            rows.append(row)

    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"collected {len(table)} rows from {args.results}")
    if len(table):
        print("methods:", sorted(table["method"].unique()))
        print("metrics:", sorted(table["metric"].unique()))
        print("sparsities:", sorted(table["sparsity"].unique()))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
