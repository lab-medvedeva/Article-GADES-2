#!/usr/bin/env Rscript
# Dense Kendall distance of GADES: GPU against CPU on inputs with many features.
#
#   Validated:  mtrx_distance(metric = "kendall", type = "gpu", sparse = FALSE)
#               for 30, 7000 and 20000 features in one batch, and for 7000
#               features in three batches (different-batch blocks). 30 features
#               run the shared-memory variant of the GPU kernel, 7000 and 20000
#               features the global-memory variant; each row prints the variant.
#   Reference:  the same call with type = "cpu". The CPU kernel is compared
#               with pure R by validate_kendall_dense.R; a pure-R count is too
#               slow for these feature counts.
#   Tolerance:  maximum absolute difference below 1e-12.
#   GPU:        required.
#   Run:        export GADES_ROOT=/path/to/GADES
#               Rscript validate_kendall_dense_gpu_cpu.R

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- getwd()
if (length(file_argument) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_argument[1])))
}
source(file.path(script_dir, "common.R"))
gpu_loaded <- load_gades(gpu = "required")

status_label <- function(d) {
  if (d < 1e-12) {
    return("OK")
  }
  "*** FAIL"
}

chk <- function(tag, N, M) {
  set.seed(1)
  X <- matrix(rpois(N * M, 3), N, M)
  G <- mtrx_distance(X, metric = "kendall", type = "gpu", sparse = FALSE, batch_size = 1000, write = TRUE)
  C <- mtrx_distance(X, metric = "kendall", type = "cpu", sparse = FALSE, batch_size = 1000, write = TRUE)
  d <- max(abs(G - C))
  variant <- "B (shared memory)"
  if (4 * N * 4 > (99 * 1024 - 2048)) {
    variant <- "G (global memory)"
  }
  line_format <- "%-22s N=%6d M=%d -> variant %s | max|GPU-CPU|=%.2e %s\n"
  cat(sprintf(line_format, tag, N, M, variant, d, status_label(d)))
  d < 1e-12
}

# Batch size below the number of objects: the block loop calls the
# different-batch kernel.
chkdiff <- function(N, M, bs) {
  set.seed(2)
  X <- matrix(rpois(N * M, 3), N, M)
  G <- mtrx_distance(X, metric = "kendall", type = "gpu", sparse = FALSE, batch_size = bs, write = TRUE)
  C <- mtrx_distance(X, metric = "kendall", type = "cpu", sparse = FALSE, batch_size = bs, write = TRUE)
  d <- max(abs(G - C))
  line_format <- "diff-blocks N=%d M=%d bs=%d -> max|GPU-CPU|=%.2e %s\n"
  cat(sprintf(line_format, N, M, bs, d, status_label(d)))
  d < 1e-12
}

ok <- TRUE
ok <- chk("small (variant B)", 30, 8) && ok
ok <- chk("large (variant G)", 7000, 6) && ok
ok <- chk("xlarge (variant G)", 20000, 5) && ok
ok <- chkdiff(7000, 12, 4) && ok
if (ok) {
  cat("\n==== DENSE KENDALL ALL PASS ====\n")
  quit(status = 0)
}
cat("\n==== FAIL ====\n")
quit(status = 1)
