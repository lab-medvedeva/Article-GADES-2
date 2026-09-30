#!/usr/bin/env Rscript
# Dense Kendall distance of GADES against a count of discordant pairs in pure R.
#
#   Validated:  mtrx_distance(metric = "kendall", sparse = FALSE) on the CPU with
#               one batch (same-batch block) and with two batches (same-batch and
#               different-batch blocks); the same on the GPU when VALIDATE_GPU=1.
#   Reference:  kendall_reference() in common.R, the definition of the distance.
#   Tolerance:  maximum absolute difference below 1e-9.
#   GPU:        optional, VALIDATE_GPU=1.
#   Run:        export GADES_ROOT=/path/to/GADES
#               Rscript validate_kendall_dense.R

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- getwd()
if (length(file_argument) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_argument[1])))
}
source(file.path(script_dir, "common.R"))
gpu_loaded <- load_gades(gpu = gpu_mode_from_environment())

set.seed(7)
nfeat <- 200
ncell <- 40
# features x cells
M <- matrix(sample(1:8, nfeat * ncell, replace = TRUE), nfeat, ncell)

ref <- kendall_reference(M)

run <- function(type, bs) {
  distances <- quiet(
    mtrx_distance(M, batch_size = bs, metric = "kendall", type = type, sparse = FALSE, write = TRUE)
  )
  unname(distances)
}

fail <- 0
chk <- function(label, got) {
  d <- max(abs(got - ref))
  status <- "FAIL"
  if (d < 1e-9) {
    status <- "OK"
  }
  cat(sprintf("  %-28s max|diff| = %.3e  %s\n", label, d, status))
  if (d >= 1e-9) {
    fail <<- fail + 1
  }
}

cat(sprintf("dense %d features x %d cells, integer\n", nfeat, ncell))
chk("CPU same_block", run("cpu", ncell))
chk("CPU batched", run("cpu", 20))
if (gpu_loaded) {
  chk("GPU same_block", run("gpu", ncell))
  chk("GPU batched", run("gpu", 20))
}
if (fail == 0) {
  cat("PASS\n")
  quit(status = 0)
}
cat(sprintf("%d FAIL\n", fail))
quit(status = 1)
