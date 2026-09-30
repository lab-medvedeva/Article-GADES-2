#!/usr/bin/env Rscript
# Spearman distance against the share of zeros in the data: GADES, ArrayFire
# and Armadillo. Without zeros there are no ties and all three agree with R;
# as the share of zeros grows, the two libraries that rank without averaging
# over ties move away from the reference. The output table is the source of
# the supplementary figure on Spearman and ties.
#
#   Validated:  mtrx_distance(metric = "spearman", type = "gpu",
#               sparse = FALSE), arrayfire_spearman_dump and
#               armadillo_spearman_dump, on matrices of 60 features x 30
#               objects with the share of zeros 0, 0.30, 0.50, 0.70, 0.85
#               and 0.95.
#   Reference:  1 - cor(M, method = "spearman").
#   Tolerance:  none, the script records the maximum and the mean absolute
#               error off the diagonal.
#   GPU:        required (GADES GPU and the CUDA backend of ArrayFire).
#   Helpers:    arrayfire_spearman_dump and armadillo_spearman_dump next to
#               this script, or the paths in ARRAYFIRE_SPEARMAN_DUMP and
#               ARMADILLO_SPEARMAN_DUMP. The ArrayFire libraries must be on
#               LD_LIBRARY_PATH.
#   Output:     spearman_ties_sweep.csv next to this script, or the path given
#               as the first argument.
#   Run:        export GADES_ROOT=/path/to/GADES
#               Rscript spearman_ties_sweep.R [output.csv]

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- getwd()
if (length(file_argument) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_argument[1])))
}
source(file.path(script_dir, "common.R"))
gpu_loaded <- load_gades(gpu = "required")
arrayfire_dump <- find_helper_binary(script_dir, "arrayfire_spearman_dump", "ARRAYFIRE_SPEARMAN_DUMP")
armadillo_dump <- find_helper_binary(script_dir, "armadillo_spearman_dump", "ARMADILLO_SPEARMAN_DUMP")

output_path <- file.path(script_dir, "spearman_ties_sweep.csv")
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) >= 1) {
  output_path <- arguments[1]
}

# Writes M as objects x features, runs a dumper and reads its distance matrix.
dump_spearman <- function(M, binary) {
  csv_in <- tempfile(fileext = ".csv")
  out <- tempfile(fileext = ".csv")
  write.csv(t(M), csv_in)
  system2(binary, c(csv_in, out), stdout = FALSE, stderr = FALSE)
  as.matrix(read.csv(out, header = FALSE))
}
offmax <- function(D, R) {
  d <- abs(D - R)
  diag(d) <- 0
  max(d)
}
offmean <- function(D, R) {
  d <- abs(D - R)
  diag(d) <- 0
  mean(d[upper.tri(d)])
}

set.seed(7)
n <- 60L
m <- 30L
sparsities <- c(0, 0.30, 0.50, 0.70, 0.85, 0.95)
cat(sprintf("%-9s | %-12s | %-10s | %-10s\n", "sparsity", "GADES max", "AF max", "Arma max"))
cat(strrep("-", 52), "\n")
rows <- list()
for (sparsity in sparsities) {
  M <- matrix(runif(n * m, 1, 100), n, m)
  if (sparsity > 0) {
    M[matrix(runif(n * m) < sparsity, n, m)] <- 0
  }
  # no constant column
  for (j in 1:m) {
    if (length(unique(M[, j])) < 2) {
      M[sample(n, 2), j] <- c(0, 50)
    }
  }
  R <- 1 - cor(M, method = "spearman")
  G <- mtrx_distance(M, metric = "spearman", type = "gpu", sparse = FALSE, write = TRUE)
  A <- dump_spearman(M, arrayfire_dump)
  dimnames(A) <- dimnames(R)
  Ar <- dump_spearman(M, armadillo_dump)
  dimnames(Ar) <- dimnames(R)
  rows[[length(rows) + 1]] <- data.frame(
    sparsity = sparsity,
    gades_max = offmax(G, R),
    gades_mean = offmean(G, R),
    af_max = offmax(A, R),
    af_mean = offmean(A, R),
    arma_max = offmax(Ar, R),
    arma_mean = offmean(Ar, R)
  )
  cat(sprintf("%-9.2f | %-12.2e | %-10.4f | %-10.4f\n", sparsity, offmax(G, R), offmax(A, R), offmax(Ar, R)))
}
df <- do.call(rbind, rows)
write.csv(df, output_path, row.names = FALSE)
cat(sprintf("\n-> %s\n", output_path))
