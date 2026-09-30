#!/usr/bin/env Rscript
# Sparse Kendall distance of GADES on the GPU: every variant of the
# per_cell_pair kernel, in one batch and in two batches.
#
#   Validated:  mtrx_distance(metric = "kendall", type = "gpu", sparse = TRUE)
#               with sparse_layout per_cell_pair, per_cell_pair_a,
#               per_cell_pair_b, per_cell_pair_b0 and per_cell_pair_hybrid, on
#               five inputs: non-negative and signed values, 400 to 20000
#               features, densities 0.01 to 0.30.
#   Reference:  kendall_reference() in common.R for cases 1, 2 and 4; the CPU
#               per_cell_pair kernel for cases 3 and 5, where a count in pure R
#               is too slow (that kernel is compared with pure R in case 1 and
#               by validate_kendall_sparse.R).
#   Tolerance:  maximum absolute difference below 1e-9.
#   GPU:        required.
#   Run:        export GADES_ROOT=/path/to/GADES
#               Rscript validate_kendall_gpu.R

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- getwd()
if (length(file_argument) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_argument[1])))
}
source(file.path(script_dir, "common.R"))
gpu_loaded <- load_gades(gpu = "required")

run <- function(sp, layout, bs, type) {
  distances <- quiet(
    mtrx_distance(
      sp,
      batch_size = bs,
      metric = "kendall",
      type = type,
      sparse = TRUE,
      sparse_layout = layout,
      write = TRUE
    )
  )
  unname(distances)
}

fail <- 0
check <- function(label, got, ref) {
  d <- max(abs(got - ref))
  status <- "FAIL"
  if (d < 1e-9) {
    status <- "OK"
  }
  cat(sprintf("  %-34s max|diff| = %.3e  %s\n", label, d, status))
  if (d >= 1e-9) {
    fail <<- fail + 1
  }
}

# ---- Case 1: small, non-negative, reference in pure R ----
cat("Case 1: 400 genes x 48 cells, density 0.15, nonneg (ref = pure-R)\n")
sp1 <- generate_sparse_matrix(400, 48, 0.15, FALSE, 1)
ref1 <- kendall_reference(as.matrix(sp1))
nc1 <- ncol(sp1)
check("cpu fenwick (1 block)", run(sp1, "per_cell_pair", nc1, "cpu"), ref1)
check("gpu a (1 block)", run(sp1, "per_cell_pair_a", nc1, "gpu"), ref1)
check("gpu b (1 block)", run(sp1, "per_cell_pair_b", nc1, "gpu"), ref1)
check("gpu hybrid (1 block)", run(sp1, "per_cell_pair_hybrid", nc1, "gpu"), ref1)
check("gpu a (batched)", run(sp1, "per_cell_pair_a", nc1 / 2, "gpu"), ref1)
check("gpu b (batched)", run(sp1, "per_cell_pair_b", nc1 / 2, "gpu"), ref1)
check("gpu hybrid (batched)", run(sp1, "per_cell_pair_hybrid", nc1 / 2, "gpu"), ref1)
check("gpu BARE per_cell_pair(->B)", run(sp1, "per_cell_pair", nc1, "gpu"), ref1)
check("gpu BARE per_cell_pair batch", run(sp1, "per_cell_pair", nc1 / 2, "gpu"), ref1)

# ---- Case 2: signed values, reference in pure R ----
cat("Case 2: 400 genes x 40 cells, density 0.20, signed (ref = pure-R)\n")
sp2 <- generate_sparse_matrix(400, 40, 0.20, TRUE, 2)
ref2 <- kendall_reference(as.matrix(sp2))
nc2 <- ncol(sp2)
check("gpu a", run(sp2, "per_cell_pair_a", nc2, "gpu"), ref2)
check("gpu b", run(sp2, "per_cell_pair_b", nc2, "gpu"), ref2)
check("gpu hybrid", run(sp2, "per_cell_pair_hybrid", nc2, "gpu"), ref2)

# ---- Case 3: more nonzeros per cell, reference = CPU per_cell_pair ----
cat("Case 3: 3000 genes x 40 cells, density 0.20, nonneg (ref = CPU fenwick)\n")
sp3 <- generate_sparse_matrix(3000, 40, 0.20, FALSE, 3)
nc3 <- ncol(sp3)
ref3 <- run(sp3, "per_cell_pair", nc3, "cpu")
check("gpu a", run(sp3, "per_cell_pair_a", nc3, "gpu"), ref3)
check("gpu b", run(sp3, "per_cell_pair_b", nc3, "gpu"), ref3)
check("gpu hybrid", run(sp3, "per_cell_pair_hybrid", nc3, "gpu"), ref3)
check("gpu a (batched)", run(sp3, "per_cell_pair_a", nc3 / 2, "gpu"), ref3)
check("gpu b (batched)", run(sp3, "per_cell_pair_b", nc3 / 2, "gpu"), ref3)

# ---- Case 4: few nonzeros per cell, the O(k^2) fast path of variant b ----
cat("Case 4: 2000 genes x 60 cells, density 0.01, nonneg (k~40, ref = pure-R)\n")
sp4 <- generate_sparse_matrix(2000, 60, 0.01, FALSE, 4)
ref4 <- kendall_reference(as.matrix(sp4))
nc4 <- ncol(sp4)
check("gpu b  (fast path on)", run(sp4, "per_cell_pair_b", nc4, "gpu"), ref4)
check("gpu b0 (fast path off)", run(sp4, "per_cell_pair_b0", nc4, "gpu"), ref4)

# ---- Case 5: about 6000 nonzeros per cell: the plain per_cell_pair layout
# has to fall back from variant b to variant a ----
case5_title <- "Case 5: 20000 genes x 32 cells, density 0.30, nonneg -> bare must route to A (ref = CPU fenwick)\n"
cat(case5_title)
sp5 <- generate_sparse_matrix(20000, 32, 0.30, FALSE, 5)
nc5 <- ncol(sp5)
ref5 <- run(sp5, "per_cell_pair", nc5, "cpu")
check("gpu BARE per_cell_pair(->A)", run(sp5, "per_cell_pair", nc5, "gpu"), ref5)
check("gpu a (cross-check)", run(sp5, "per_cell_pair_a", nc5, "gpu"), ref5)

if (fail == 0) {
  cat("\nALL PASS\n")
  quit(status = 0)
}
cat(sprintf("\n%d FAILURES\n", fail))
quit(status = 1)
