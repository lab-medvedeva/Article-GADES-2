#!/usr/bin/env Rscript
# Sparse Kendall distance of GADES on the CPU against a count of discordant
# pairs in pure R.
#
#   Validated:  mtrx_distance(metric = "kendall", type = "cpu", sparse = TRUE)
#               with the per_cell_pair layout, in one batch (same-batch block)
#               and in two batches (same-batch and different-batch blocks), and
#               the agreement of the two runs with each other.
#   Reference:  kendall_reference() in common.R, the definition of the distance.
#   Tolerance:  maximum absolute difference below 1e-9.
#   Printed without a pass criterion: the run with sparse_layout = "default".
#   GPU:        not needed.
#   Run:        export GADES_ROOT=/path/to/GADES
#               Rscript validate_kendall_sparse.R

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- getwd()
if (length(file_argument) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_argument[1])))
}
source(file.path(script_dir, "common.R"))
gpu_loaded <- load_gades(gpu = "none")

set.seed(42)
n_genes <- 300
n_cells <- 40
density <- 0.15
M <- matrix(0, n_genes, n_cells)
mask <- runif(n_genes * n_cells) < density
M[mask] <- sample(1:10, sum(mask), replace = TRUE)
sp <- as(Matrix(M, sparse = TRUE), "CsparseMatrix")

ref <- kendall_reference(M)

run <- function(layout, bs) {
  distances <- quiet(
    mtrx_distance(
      sp,
      batch_size = bs,
      metric = "kendall",
      type = "cpu",
      sparse = TRUE,
      sparse_layout = layout,
      write = TRUE
    )
  )
  unname(distances)
}

# one batch: same-batch block only
pcp_1blk <- run("per_cell_pair", n_cells)
# two batches: same-batch and different-batch blocks
pcp_batch <- run("per_cell_pair", 20)
def_blk <- run("default", n_cells)

d1 <- max(abs(pcp_1blk - ref))
d2 <- max(abs(pcp_batch - ref))
d3 <- max(abs(def_blk - ref))
d12 <- max(abs(pcp_1blk - pcp_batch))

cat(sprintf("matrix: %d genes x %d cells, density %.2f\n", n_genes, n_cells, density))
cat(sprintf("max |per_cell_pair (1 block)  - ref| = %.3e\n", d1))
cat(sprintf("max |per_cell_pair (batched)  - ref| = %.3e\n", d2))
cat(sprintf("max |default sparse           - ref| = %.3e\n", d3))
cat(sprintf("max |per_cell_pair 1blk - batched|   = %.3e\n", d12))

tol <- 1e-9
ok <- (d1 < tol) && (d2 < tol) && (d12 < tol)
if (ok) {
  cat("PASS: per_cell_pair matches the reference\n")
  quit(status = 0)
}
cat("FAIL\n")
quit(status = 1)
