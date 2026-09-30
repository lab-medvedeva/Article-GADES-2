#!/usr/bin/env python3
"""Collect the generated dense benchmark into one table.

    python3 reproducibility/collect_generated_dense.py

Reads results/GeneratedDense/<C>_cells_<F>_features/<density>_<method>_<metric>.csv:
one column of times in microseconds, one row per repetition. Writes
reproducibility/data/generated_dense.csv with one row per result file.

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
DEFAULT_RESULTS = REPO_ROOT / "results" / "GeneratedDense"
DEFAULT_OUT = REPO_ROOT / "reproducibility" / "data" / "generated_dense.csv"

DIRECTORY_PATTERN = re.compile(r"(?P<cells>\d+)_cells_(?P<features>\d+)_features")


def parse_name(path):
    """Return cells, features, density, method and metric encoded in the path."""
    match = DIRECTORY_PATTERN.search(os.path.dirname(path))
    if not match:
        return None
    stem = os.path.basename(path)[:-4]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    density = parts[-3]
    method = parts[-2]
    metric = parts[-1]
    return int(match.group("cells")), int(match.group("features")), density, method, metric


def summarise(path, density_filter, drop_first):
    """Return the table row of one result file, or None when the file is skipped."""
    parsed = parse_name(path)
    if parsed is None:
        return None
    cells, features, density, method, metric = parsed
    if density_filter and density != density_filter:
        return None

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
        "density": density,
        "wX": cells,
        "wY": features,
        "ID": f"{cells}_{features}",
        "|W|": cells * features,
        "logWX": np.log10(cells),
        "logWY": np.log10(features),
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
    parser = argparse.ArgumentParser(description="Collect the generated dense benchmark.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="directory GeneratedDense")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="table to write")
    parser.add_argument("--density", default="dense", help="file prefix to keep; empty keeps all")
    parser.add_argument(
        "--keep-first-repetition",
        action="store_true",
        help="include the first repetition in the statistics",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results):
        raise SystemExit(f"results directory not found: {args.results}")

    pattern = os.path.join(args.results, "*_cells_*_features", "*_*_*.csv")
    rows = []
    for path in sorted(glob.glob(pattern)):
        row = summarise(path, args.density, not args.keep_first_repetition)
        if row is not None:
            rows.append(row)

    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"collected {len(table)} rows from {args.results}")
    if len(table):
        print("methods:", sorted(table["method"].unique()))
        print("metrics:", sorted(table["metric"].unique()))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
