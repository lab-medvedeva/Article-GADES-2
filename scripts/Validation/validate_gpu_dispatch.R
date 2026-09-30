#!/usr/bin/env Rscript
# Sparse Kendall on the GPU: the per-pair dispatch of the per_cell_pair kernel.
# The kernel keeps the buffers of a pair of cells in shared memory when they
# fit and falls back to global memory otherwise; the environment variable
# HOBO_PCP_CAPSMEM forces the fallback for pairs above a given size.
#
#   Validated:  process_batch(metric = "kendall", sparse = TRUE,
#               sparse_layout = "per_cell_pair") for a same-batch block and a
#               different-batch block, without a cap and with HOBO_PCP_CAPSMEM
#               set to 2 and to 8, on a matrix where every twelfth cell is
#               almost dense.
#   Reference:  process_batch_cpu() with the per_cell_pair layout, and the GPU
#               variant per_cell_pair_a.
#   Tolerance:  none, the results must be identical (maximum difference 0).
#   GPU:        required.
#   Run:        export GADES_ROOT=/path/to/GADES
#               Rscript validate_gpu_dispatch.R

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- getwd()
if (length(file_argument) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_argument[1])))
}
source(file.path(script_dir, "common.R"))
gpu_loaded <- load_gades(gpu = "required")

set.seed(7)
n_genes <- 200
n_cells <- 48
# Most cells have few nonzeros, every twelfth cell is almost dense. The values
# are small integers, so the cells have ties.
M <- matrix(0L, n_genes, n_cells)
for (j in seq_len(n_cells)) {
  if (j %% 12 == 0) {
    k <- sample(120:190, 1)
  } else {
    k <- sample(2:25, 1)
  }
  rows <- sample(n_genes, k)
  M[rows, j] <- sample(1:4, k, replace = TRUE)
}
support <- colSums(M != 0)
cat(sprintf("matrix %dx%d, col-support range [%d, %d]\n", n_genes, n_cells, min(support), max(support)))

gpu_block <- function(fi, si, bs, layout) {
  block <- process_batch(M, fi, si, bs, "kendall", sparse = TRUE, sparse_layout = layout)
  block$correlation_matrix
}
cpu_block <- function(fi, si, bs) {
  block <- process_batch_cpu(M, fi, si, bs, "kendall", sparse = TRUE, sparse_layout = "per_cell_pair")
  block$correlation_matrix
}

report <- function(tag, a, b) {
  d <- max(abs(a - b))
  status <- "*** MISMATCH"
  if (d == 0) {
    status <- "OK (bit-identical)"
  }
  cat(sprintf("  %-46s max|diff| = %.3e   %s\n", tag, d, status))
  invisible(d == 0)
}

ok <- TRUE
run_caps <- function(capval) {
  if (is.na(capval)) {
    Sys.unsetenv("HOBO_PCP_CAPSMEM")
    tag <- "all-shared (no cap)"
  } else {
    Sys.setenv(HOBO_PCP_CAPSMEM = as.character(capval))
    tag <- sprintf("forced cap_smem=%d (fallback path)", capval)
  }
  cat(sprintf("\n=== %s ===\n", tag))

  # same-batch block: the whole matrix in one block
  ref <- cpu_block(0, 0, n_cells)
  disp <- gpu_block(0, 0, n_cells, "per_cell_pair")
  varA <- gpu_block(0, 0, n_cells, "per_cell_pair_a")
  ok <<- report("same_block  GPU-dispatch vs CPU", disp, ref) && ok
  ok <<- report("same_block  GPU-dispatch vs GPU-varA", disp, varA) && ok

  # different-batch block: the first half of the cells against the second half
  bs <- n_cells %/% 2
  refD <- cpu_block(0, bs, bs)
  dispD <- gpu_block(0, bs, bs, "per_cell_pair")
  varAD <- gpu_block(0, bs, bs, "per_cell_pair_a")
  ok <<- report("diff_blocks GPU-dispatch vs CPU", dispD, refD) && ok
  ok <<- report("diff_blocks GPU-dispatch vs GPU-varA", dispD, varAD) && ok
}

# no cap: every pair fits shared memory
run_caps(NA)
# almost every pair goes to the fallback
run_caps(2)
# small pairs stay in shared memory, larger pairs go to the fallback
run_caps(8)
Sys.unsetenv("HOBO_PCP_CAPSMEM")

if (ok) {
  cat("\n==== ALL PASS: dispatch is bit-identical ====\n")
  quit(status = 0)
}
cat("\n==== FAILURE: see mismatches above ====\n")
quit(status = 1)
