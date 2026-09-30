#!/usr/bin/env python3
"""Collect the benchmark of the huge datasets and draw its figure.

    python3 reproducibility/collect_huge_results.py

Reads results/HugeDatasets/<dataset>/<mode>_<method>_<metric>.csv. A case without a result
file did not finish within the time limit. The time of a case is the median of its
repetitions, without the first one when there are several. Writes
reproducibility/data/huge_walltimed.csv and reproducibility/data/huge_speedup.csv (time of
GADES-GPU-dense divided by the time of the method, per dataset), then draws
figure_huge_combined.png with figure_real22_combined.py on the logarithmic axis.
"""
import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "results" / "HugeDatasets"
DATA_DIR = REPO_ROOT / "reproducibility" / "data"
FIGURES_DIR = REPO_ROOT / "reproducibility" / "figures"

DATASETS = [
    "MouseAtlas", "TabulaMuris", "TabulaSapiensV1", "FibrocardATAC", "AIDA",
    "NYTimes", "Netflix", "BookCrossing", "Pokec", "AmazonVideoGames",
]
ALL_METRICS = ["euclidean", "cosine", "pearson", "manhattan", "spearman", "kendall"]
FOUR_METRICS = ["euclidean", "cosine", "pearson", "manhattan"]
# name in the figure -> mode, method in the file name, supported metrics
SERIES = {
    "GADES-GPU-dense": ("dense", "GPU", ALL_METRICS),
    "GADES-GPU-sparse": ("sparse", "GPU", ALL_METRICS),
    "GADES-CPU-sparse": ("sparse", "CPU", ALL_METRICS),
    "arrayfire-sparse": ("sparse", "arrayfirecuda", FOUR_METRICS),
    "raft-sparse": ("sparse", "raft", FOUR_METRICS),
    "armadillo-sparse": ("sparse", "armadillo", FOUR_METRICS),
}
# methods of the real-dataset figure that the huge suite does not run
NOT_RUN = [
    "GADES-CPU-dense", "arrayfire-dense", "raft-dense", "armadillo-dense",
    "amap", "factoextra", "pandas", "scipy",
]
BASELINE = "GADES-GPU-dense"


def read_seconds(path):
    """Median time of a result file in seconds, None when there is no valid time."""
    if not path.exists():
        return None
    with path.open() as handle:
        rows = []
        for row in csv.reader(handle):
            if row:
                rows.append(row)
    header = rows[0]
    data = rows[1:]
    column = 0
    if data and len(data[0]) == len(header) + 1:
        column = 1
    times = []
    for row in data:
        try:
            value = float(row[column])
        except ValueError:
            continue
        if value > 0:
            times.append(value)
    if not times:
        return None
    if len(times) > 1:
        times = times[1:]
    return statistics.median(times) / 1e6



def write_walltimed(path, seconds):
    header = ["metric", "method", "completed", "wall_timed", "total_datasets", "unsupported"]
    total = len(DATASETS)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for metric in ALL_METRICS:
            for name, (mode, method, metrics) in SERIES.items():
                if metric not in metrics:
                    writer.writerow([metric, name, 0, 0, total, 1])
                    continue
                completed = 0
                for dataset in DATASETS:
                    if seconds[(dataset, name, metric)] is not None:
                        completed += 1
                writer.writerow([metric, name, completed, total - completed, total, 0])
            for name in NOT_RUN:
                writer.writerow([metric, name, 0, 0, total, 1])


def write_speedup(path, seconds):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "method", "speedup", "dataset"])
        for (dataset, name, metric), value in seconds.items():
            reference = seconds[(dataset, BASELINE, metric)]
            if value is None or reference is None:
                continue
            writer.writerow([metric, name, reference / value, dataset])


def main():
    parser = argparse.ArgumentParser(description="Collect the benchmark of the huge datasets.")
    parser.add_argument("--huge-results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="directory for the two tables")
    parser.add_argument("--out-dir", default=str(FIGURES_DIR), help="directory for the figure")
    args = parser.parse_args()

    results = Path(args.huge_results)
    if not results.is_dir():
        raise SystemExit(f"results directory not found: {results}")
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    seconds = {}
    for dataset in DATASETS:
        for name, (mode, method, metrics) in SERIES.items():
            for metric in metrics:
                path = results / dataset / f"{mode}_{method}_{metric}.csv"
                seconds[(dataset, name, metric)] = read_seconds(path)

    walltimed_path = data_dir / "huge_walltimed.csv"
    speedup_path = data_dir / "huge_speedup.csv"
    write_walltimed(walltimed_path, seconds)
    write_speedup(speedup_path, seconds)
    print(f"-> {walltimed_path}")
    print(f"-> {speedup_path}")

    figure_script = Path(__file__).resolve().parent / "figure_real22_combined.py"
    command = [
        sys.executable, str(figure_script),
        "--walltimed", str(walltimed_path),
        "--speedup", str(speedup_path),
        "--log-speedup",
        "--out-dir", args.out_dir,
        "--name", "figure_huge_from_results",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
