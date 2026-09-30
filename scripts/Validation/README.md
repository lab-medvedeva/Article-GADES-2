# Validation

Accuracy checks of GADES against reference implementations. Every R script loads GADES from a built source tree and compares the distance matrices of `mtrx_distance()` with a reference. The `validate_*.R` scripts exit with status 1 when a check fails; the two `spearman_ties*.R` scripts report the errors and have no pass criterion.

## Requirements

- A built GADES source tree (Step 1 of the top-level README) and its path in `GADES_ROOT`. The scripts load `$GADES_ROOT/build/mtrx_cpu.so`, `$GADES_ROOT/build/mtrx.so` (the GPU library) and `$GADES_ROOT/R/mtrx.R`. When `build/` does not hold the shared objects, `lib/` is tried.
- R packages `Matrix` and `glue`.
- A CUDA GPU for the scripts marked "required" below.
- ArrayFire with the CUDA backend and Armadillo for the two `spearman_ties*.R` scripts.

A script stops with a message when `GADES_ROOT` is not set. The scripts can be started from any directory; `common.R` has to stay next to them.

## Conventions

The input matrix is features by objects: distances are computed between columns, as in the benchmark driver `scripts/Benchmarking/test.R`. The references follow the definitions of the GADES kernels:

| Metric | Distance between objects x and y | Reference in R |
|---|---|---|
| `euclidean` | square root of the sum of squared differences | `dist(t(M), method = "euclidean")` |
| `manhattan` | sum of absolute differences | `dist(t(M), method = "manhattan")` |
| `cosine` | 1 - cosine similarity | `1 - crossprod(M) / outer(norms, norms)` with `norms <- sqrt(colSums(M * M))` |
| `pearson` | 1 - Pearson correlation | `1 - cor(M, method = "pearson")` |
| `spearman` | 1 - Spearman correlation, average ranks for ties | `1 - cor(M, method = "spearman")` |
| `kendall` | 2 D / (n (n - 1)), where D is the number of feature pairs that the two objects order oppositely and n is the number of features; a pair with a tie is not discordant | `kendall_reference()` in `common.R` |

A block is the part of the distance matrix that one kernel call computes. With a batch size not smaller than the number of objects there is one same-batch block. With a smaller batch size `mtrx_distance()` also computes different-batch blocks, which use separate kernels.

## Scripts

| Script | Validated | Reference | Tolerance | GPU |
|---|---|---|---|---|
| `validate_all_metrics.R` | all six metrics, CPU and GPU, dense and sparse input, one batch and several batches | pure R, table above | relative error at most 1e-4 | optional, `VALIDATE_GPU=1` |
| `validate_kendall_dense.R` | dense Kendall, one batch and two batches | count of discordant pairs in pure R | absolute difference below 1e-9 | optional, `VALIDATE_GPU=1` |
| `validate_kendall_dense_gpu_cpu.R` | dense Kendall on the GPU for 30, 7000 and 20000 features, one batch and three batches | the CPU kernel | absolute difference below 1e-12 | required |
| `validate_kendall_sparse.R` | sparse Kendall on the CPU, layout `per_cell_pair`, one batch and two batches | count of discordant pairs in pure R | absolute difference below 1e-9 | no |
| `validate_kendall_gpu.R` | sparse Kendall on the GPU, all variants of the `per_cell_pair` kernel, non-negative and signed values | pure R for the small inputs, the CPU kernel for the large ones | absolute difference below 1e-9 | required |
| `validate_gpu_dispatch.R` | sparse Kendall on the GPU: choice between shared and global memory for every pair of objects, forced through `HOBO_PCP_CAPSMEM` | the CPU kernel and the GPU variant `per_cell_pair_a` | identical results | required |
| `validate_spearman_sparse.R` | sparse Spearman, layouts `default` and `per_cell_pair`, GPU and CPU, non-negative and signed values | `1 - cor(method = "spearman")`; the `default` layout for the largest input | absolute difference below 2e-3 off the diagonal | required |
| `spearman_ties.R` | Spearman on a matrix with 80 % zeros: GADES in four modes (GPU and CPU, dense and sparse) and ArrayFire | `1 - cor(method = "spearman")` | none, prints the errors | required |
| `spearman_ties_sweep.R` | Spearman against the share of zeros (0 to 0.95): GADES on the GPU with dense input, ArrayFire and Armadillo | `1 - cor(method = "spearman")` | none, writes the errors to `spearman_ties_sweep.csv` | required |
| `kendall_dense_reference.cpp` | the Fenwick-tree count of discordant pairs used by the dense Kendall kernel | naive count over all pairs | equal counts | no, standalone |
| `kendall_sparse_reference.cpp` | the Fenwick-tree count used by the sparse Kendall kernel | naive count over all features and a quadratic merge over the nonzero features | equal counts | no, standalone |

`validate_all_metrics.R` was written for this repository; the other scripts come from the GADES source tree with paths and comments adapted. `validate_kendall_dense.R` and `validate_kendall_dense_gpu_cpu.R` check different things: the first compares the CPU (and, on request, the GPU) with pure R on 200 features, the second compares the GPU with the CPU at feature counts where a count in pure R is too slow and where the GPU kernel switches to its global-memory variant.

`common.R` holds the shared code: loading GADES, the Kendall reference, the generator of sparse test matrices. `arrayfire_spearman_dump.cpp` and `armadillo_spearman_dump.cpp` compute the Spearman distance with the expressions of the benchmark drivers `scripts/Benchmarking/test_arrayfire.cpp` and `test_armadillo.cpp` and write the matrix to a CSV file.

## Build the helper programs

```shell
cd scripts/Validation
g++ -O2 -std=c++17 kendall_dense_reference.cpp -o kendall_dense_reference
g++ -O2 -std=c++17 kendall_sparse_reference.cpp -o kendall_sparse_reference
g++ -O3 -std=c++17 armadillo_spearman_dump.cpp -o armadillo_spearman_dump -larmadillo
g++ -O3 -std=c++17 arrayfire_spearman_dump.cpp -o arrayfire_spearman_dump -laf
```

For an ArrayFire build tree in `$AF` that is not installed system-wide:

```shell
g++ -O3 -std=c++17 -I$AF/include -I$AF/build/include arrayfire_spearman_dump.cpp -o arrayfire_spearman_dump -L$AF/build/src/api/unified -laf
export LD_LIBRARY_PATH=$AF/build/src/api/unified:$AF/build/src/backend/cuda:$AF/build/src/backend/cpu:$LD_LIBRARY_PATH
```

The R scripts look for `arrayfire_spearman_dump` and `armadillo_spearman_dump` next to themselves; `ARRAYFIRE_SPEARMAN_DUMP` and `ARMADILLO_SPEARMAN_DUMP` override the paths.

## Run

```shell
export GADES_ROOT=/path/to/GADES
cd scripts/Validation

# all metrics, CPU only; well under a minute
Rscript validate_all_metrics.R
# all metrics, CPU and GPU
VALIDATE_GPU=1 Rscript validate_all_metrics.R

# Kendall
Rscript validate_kendall_dense.R
VALIDATE_GPU=1 Rscript validate_kendall_dense.R
Rscript validate_kendall_dense_gpu_cpu.R
Rscript validate_kendall_sparse.R
Rscript validate_kendall_gpu.R
Rscript validate_gpu_dispatch.R

# Spearman
Rscript validate_spearman_sparse.R
Rscript spearman_ties.R
Rscript spearman_ties_sweep.R

# counting algorithms of the Kendall kernels, without GADES
./kendall_dense_reference
./kendall_sparse_reference
```

`validate_all_metrics.R` and `spearman_ties_sweep.R` take the path of their output table as an optional first argument.

## Outputs used by the article

- **Error bounds.** The article states that the floating-point distances agree with pure R to a maximum relative error below 1e-4 across CPU and GPU, dense and sparse input and both block types, and that the Kendall distances agree exactly. `validate_all_metrics.R` checks this statement: it prints one row per combination of metric, device, input and batch size with the maximum absolute and the maximum relative error, then the maxima per metric and overall, and fails when a relative error exceeds 1e-4. The same table is written to `validate_all_metrics.csv`.
- **Spearman and ties.** The supplementary figure on Spearman and ties is drawn from `spearman_ties_sweep.csv`, the recorded output of `spearman_ties_sweep.R` (columns: share of zeros, then the maximum and the mean absolute error off the diagonal for GADES, ArrayFire and Armadillo). A rerun of the script overwrites this file unless another path is given. `reproducibility/figure_spearman_ties.py` draws the figure; `--sweep scripts/Validation/spearman_ties_sweep.csv` points it to this copy.

