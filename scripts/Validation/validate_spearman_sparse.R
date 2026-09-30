#!/usr/bin/env Rscript
# Sparse Spearman distance of GADES, GPU and CPU, against cor() of R.
#
#   Validated:  mtrx_distance(metric = "spearman", sparse = TRUE) with the
#               layouts default and per_cell_pair, type gpu and cpu, in one
#               batch; per_cell_pair on the GPU also in two batches (case 3).
#   Reference:  1 - cor(M, method = "spearman") for cases 1 and 2
#               (non-negative and signed values); the GPU run with the default
#               layout for case 3.
#   Tolerance:  maximum absolute difference below 2e-3 off the diagonal
#               (the kernels work in single precision).
#   GPU:        required.
#   Run:        export GADES_ROOT=/path/to/GADES
#               Rscript validate_spearman_sparse.R

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
      metric = "spearman",
      type = type,
      sparse = TRUE,
      sparse_layout = layout,
      write = TRUE
    )
  )
  as.matrix(distances)
}

# 1 - Spearman correlation between cells (columns)
spearman_ref <- function(sp) {
  M <- as.matrix(sp)
  D <- 1 - cor(M, method = "spearman")
  D[is.na(D)] <- 0
  as.matrix(D)
}

tol <- 2e-3
fail <- 0
# compares the entries off the diagonal
check <- function(label, got, ref) {
  g <- got
  r <- ref
  diag(g) <- 0
  diag(r) <- 0
  d <- max(abs(g - r))
  status <- "FAIL"
  if (d < tol) {
    status <- "OK"
  }
  cat(sprintf("  %-36s max|diff| = %.3e  %s\n", label, d, status))
  if (d >= tol) {
    fail <<- fail + 1
  }
}

# ---- Case 1: small, non-negative, reference in pure R ----
cat("Case 1: 300 genes x 40 cells, density 0.20, nonneg (ref = pure-R spearman)\n")
sp1 <- generate_sparse_matrix(300, 40, 0.20, FALSE, 1)
ref1 <- spearman_ref(sp1)
nc1 <- ncol(sp1)
check("default gpu", run(sp1, "default", nc1, "gpu"), ref1)
check("default cpu", run(sp1, "default", nc1, "cpu"), ref1)
check("per_cell_pair gpu", run(sp1, "per_cell_pair", nc1, "gpu"), ref1)
check("per_cell_pair cpu", run(sp1, "per_cell_pair", nc1, "cpu"), ref1)
check("pcp gpu == default gpu", run(sp1, "per_cell_pair", nc1, "gpu"), run(sp1, "default", nc1, "gpu"))

# ---- Case 2: signed values (negative values rank below the zeros) ----
cat("Case 2: 300 genes x 36 cells, density 0.25, signed (ref = pure-R spearman)\n")
sp2 <- generate_sparse_matrix(300, 36, 0.25, TRUE, 2)
ref2 <- spearman_ref(sp2)
nc2 <- ncol(sp2)
check("per_cell_pair gpu", run(sp2, "per_cell_pair", nc2, "gpu"), ref2)
check("per_cell_pair cpu", run(sp2, "per_cell_pair", nc2, "cpu"), ref2)

# ---- Case 3: larger input, reference = default layout (consistency) ----
cat("Case 3: 2000 genes x 48 cells, density 0.15 (ref = default path)\n")
sp3 <- generate_sparse_matrix(2000, 48, 0.15, FALSE, 3)
nc3 <- ncol(sp3)
ref3 <- run(sp3, "default", nc3, "gpu")
check("per_cell_pair gpu", run(sp3, "per_cell_pair", nc3, "gpu"), ref3)
check("per_cell_pair gpu (batch)", run(sp3, "per_cell_pair", nc3 / 2, "gpu"), ref3)
check("per_cell_pair cpu", run(sp3, "per_cell_pair", nc3, "cpu"), ref3)

if (fail == 0) {
  cat("\nALL PASS\n")
  quit(status = 0)
}
cat(sprintf("\n%d FAILURES\n", fail))
quit(status = 1)
