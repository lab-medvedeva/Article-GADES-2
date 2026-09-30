# Reproducing the figures

The scripts in this directory rebuild the figures of the article from the benchmark results stored in `results/`. Every script resolves its default paths from its own location, so it can be started from the repository root or from `reproducibility/`. Collected tables are written to `reproducibility/data/` and figures to `reproducibility/figures/`; both locations can be changed with command-line arguments (`--help` lists them).

## Figures of the main text

| Figure file in the article | Command | Reads |
|---|---|---|
| `figure3_dense_accel.png` | `python3 reproducibility/collect_generated_dense.py`, then `python3 reproducibility/figure_generated_acceleration.py` | `results/GeneratedDense` |
| `figure4_sparse_accel_full_W7.png` | `python3 reproducibility/collect_generated_sparse.py`, then `python3 reproducibility/figure_generated_acceleration.py` | `results/GeneratedSparse`, `results/GeneratedDense` |
| `figure_real22_combined.png` | `python3 reproducibility/collect_real_results.py` | `results/RealDatasets` |
| `figure_huge_combined.png` | `python3 reproducibility/figure_huge_combined.py` | `reproducibility/data/real_datasets_failures.csv`, `reproducibility/data/superbig_speedup.csv` |
| `figure_positioning_canon.png` | `python3 reproducibility/figure_positioning.py --gades1-results ../Article-GADES/results` | `results/GeneratedDense`, `results/GeneratedSparse`, `results/RealDatasets`, results of the GADES 1.0 article |

The figures are written to `reproducibility/figures/`. The sections below give the requirements, the inputs and the commands of every figure, including the supplementary ones.

## Requirements

Python 3 with `numpy`, `pandas`, `matplotlib` and `seaborn`. The figures were checked with numpy 2.1.3, pandas 2.2.3, matplotlib 3.10.0 and seaborn 0.13.2.

## Inputs

| Input | Location | Used by |
|---|---|---|
| Generated dense benchmark | `results/GeneratedDense/<C>_cells_<F>_features/dense_<method>_<metric>.csv` | `collect_generated_dense.py`, `figure_positioning*.py` |
| Generated sparse benchmark | `results/GeneratedSparse/<C>_cells_<F>_features/<sparsity>/<prefix>_<method>_<metric>.csv` | `collect_generated_sparse.py`, `figure_positioning*.py` |
| Real-dataset benchmark | `results/RealDatasets/<dataset>/<mode>_<method>_<metric>.csv` | `collect_real_results.py`, `figure_real_gpu_cpu.py`, `figure_positioning*.py` |
| GADES 1.0 benchmark | `results/` of https://github.com/lab-medvedeva/Article-GADES, passed with `--gades1-results` | `figure_positioning*.py` |
| Largest datasets: unfinished runs by cause | `reproducibility/data/real_datasets_failures.csv` | `figure_huge_combined.py` |
| Largest datasets: speedup against GADES-GPU-dense | `reproducibility/data/superbig_speedup.csv` | `figure_huge_combined.py` |
| Spearman error against the share of zeros | `reproducibility/data/spearman_ties_sweep.csv`, the recorded output of `scripts/Validation/spearman_ties_sweep.R` | `figure_spearman_ties.py` |
| Raw matrices `HSC.mtx`, `PBMC5K.mtx`, `CellLines.mtx`, `HLCA_marrow.mtx`, `TCells.mtx` | not distributed with the repository; `Datasets/Real/` by default, or `--datasets <directory>` | `figure_scatac_heavy_tail.py` |

A result file holds one column of times in microseconds, one row per repetition. The time of a case is the mean of the remaining repetitions in the figures of the generated datasets, in `figure_positioning*.py` and in `figure_real_gpu_cpu.py`, and their median in `collect_real_results.py`.

The commands below expect the GADES 1.0 repository next to this one:

```shell
git clone https://github.com/lab-medvedeva/Article-GADES.git ../Article-GADES
```

## Commands

Run the commands from the repository root in this order: the collectors write the tables that the figure scripts read.

| Output | Script | Command | Inputs |
|---|---|---|---|
| `data/generated_dense.csv` | `collect_generated_dense.py` | `python3 reproducibility/collect_generated_dense.py` | `results/GeneratedDense` |
| `data/generated_sparse.csv` | `collect_generated_sparse.py` | `python3 reproducibility/collect_generated_sparse.py` | `results/GeneratedSparse` |
| `figure3_dense_accel.png`, `figure4_sparse_accel_W5.png`, `figure4_sparse_accel_W6.png`, `figure4_sparse_accel_W7.png`, `figure4_sparse_accel_full_W5.png`, `figure4_sparse_accel_full_W6.png`, `figure4_sparse_accel_full_W7.png` | `figure_generated_acceleration.py` | `python3 reproducibility/figure_generated_acceleration.py` | `data/generated_dense.csv`, `data/generated_sparse.csv` |
| `figure3_dense_times.png` | `figure_generated_dense_times.py` | `python3 reproducibility/figure_generated_dense_times.py` | `data/generated_dense.csv` |
| `figure4_sparse_times_W5.png`, `figure4_sparse_times_W6.png`, `figure4_sparse_times_W7.png` | `figure_generated_sparse_times.py` | `python3 reproducibility/figure_generated_sparse_times.py` | `data/generated_dense.csv`, `data/generated_sparse.csv` |
| `figure4-barplot-sequential.png` | `figure_gades_modes.py` | `python3 reproducibility/figure_gades_modes.py` | `data/generated_dense.csv`, `data/generated_sparse.csv` |
| `figure_positioning_canon.png`, `figure_positioning_canon.pdf` | `figure_positioning.py` | `python3 reproducibility/figure_positioning.py --gades1-results ../Article-GADES/results` | `results/GeneratedDense`, `results/GeneratedSparse`, `results/RealDatasets`, GADES 1.0 results |
| `figure_positioning_combined.png` | `figure_positioning_combined.py` | `python3 reproducibility/figure_positioning_combined.py --gades1-results ../Article-GADES/results` | `results/GeneratedDense`, `results/GeneratedSparse`, `results/RealDatasets`, GADES 1.0 results |
| `figure5_real_gpu_cpu.png` | `figure_real_gpu_cpu.py` | `python3 reproducibility/figure_real_gpu_cpu.py` | `results/RealDatasets` |
| `data/real_walltimed_22.csv`, `data/real_speedup_22.csv`, `figure_real22_combined.png`, `figure_real22_combined.pdf` | `collect_real_results.py`, which calls `figure_real22_combined.py --log-speedup` | `python3 reproducibility/collect_real_results.py` | `results/RealDatasets` |
| `figure_real22_combined.png`, `figure_real22_combined.pdf`, redrawn from the tables | `figure_real22_combined.py` | `python3 reproducibility/figure_real22_combined.py --log-speedup` | `data/real_walltimed_22.csv`, `data/real_speedup_22.csv` |
| `figure_huge_combined.png`, `figure_huge_combined.pdf` | `figure_huge_combined.py` | `python3 reproducibility/figure_huge_combined.py` | `data/real_datasets_failures.csv`, `data/superbig_speedup.csv` |
| `spearman_ties_correctness.png` | `figure_spearman_ties.py` | `python3 reproducibility/figure_spearman_ties.py` | `data/spearman_ties_sweep.csv` |
| `datasets_scatter.png` | `figure_datasets_scatter.py` | `python3 reproducibility/figure_datasets_scatter.py` | dataset sizes listed in the script |
| `figure_scatac_heavy_tail.png` | `figure_scatac_heavy_tail.py` | `python3 reproducibility/figure_scatac_heavy_tail.py --datasets Datasets/Real` | raw matrices `<dataset>.mtx` |

Paths in the first and the last column are relative to `reproducibility/` for `data/` and to the repository root for `results/` and `Datasets/`. Figures are written to `reproducibility/figures/`.

`figure_real22_combined.py` draws panel (b) on a logarithmic acceleration axis with `--log-speedup`. Without the option the axis is linear and cut at a fixed upper bound per metric, so boxes and whiskers above the bound are not shown in full.

## Conventions of the figures

- The reference of every acceleration figure is the dense GPU mode of GADES: the plotted value is the time of GADES-GPU-dense divided by the time of the method.
- In the figures of the generated datasets the benchmark method `pythonic` is shown as `scipy` and its time is multiplied by 24. In the figure of the 22 real datasets the same method is shown as `scipy` with its measured time.
- Spearman of ArrayFire and Armadillo is left out of the timing figures: both rank without averaging ties, see `spearman_ties_correctness.png`.
- A method without a result file has no box in the figures of the generated datasets and counts as not finished within the time limit in the figure of the 22 real datasets.
- In `figure_positioning_canon.png` a missing result is drawn at the 24 hour limit.
- The `DataFile` column of the collected tables holds paths relative to the repository root.

The notebooks and images of the GADES 1.0 article are kept in this directory unchanged.
