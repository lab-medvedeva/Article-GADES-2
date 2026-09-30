#!/usr/bin/env Rscript
# Spearman distance on data with many zeros: GADES against ArrayFire.
# GADES gives tied values their average rank, as cor(method = "spearman")
# does. The Spearman expression of the ArrayFire benchmark driver ranks by a
# sort of the sort permutation, which gives tied values distinct ranks.
#
#   Validated:  mtrx_distance(metric = "spearman") in four modes (GPU dense,
#               CPU dense, GPU sparse, CPU sparse) and arrayfire_spearman_dump,
#               on one matrix of 40 features x 8 objects with about 80 % zeros
#               and distinct nonzero values, so the only ties are the zeros.
#   Reference:  1 - cor(M, method = "spearman").
#   Tolerance:  none, the script prints the maximum absolute error off the
#               diagonal for each of the five results.
#   GPU:        required (GADES GPU modes and the CUDA backend of ArrayFire).
#   Helper:     arrayfire_spearman_dump next to this script, or the path in
#               ARRAYFIRE_SPEARMAN_DUMP. The ArrayFire libraries must be on
#               LD_LIBRARY_PATH.
#   Run:        export GADES_ROOT=/path/to/GADES
#               Rscript spearman_ties.R

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- getwd()
if (length(file_argument) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_argument[1])))
}
source(file.path(script_dir, "common.R"))
gpu_loaded <- load_gades(gpu = "required")
arrayfire_dump <- find_helper_binary(script_dir, "arrayfire_spearman_dump", "ARRAYFIRE_SPEARMAN_DUMP")

set.seed(42)
# features (rows)
n <- 40L
# objects (columns), the distance is computed between them
m <- 8L
# share of zeros: a large group of tied values in every column
sparsity <- 0.80

# The nonzero values are distinct, so the only ties are the zeros.
M <- matrix(0.0, n, m)
for (j in seq_len(m)) {
  # at least two nonzero values, so that no column is constant
  k <- max(2L, rbinom(1, n, 1 - sparsity))
  pos <- sample(n, k)
  M[pos, j] <- runif(k, 1, 100)
}
cat(sprintf("matrix %dx%d, actual zero fraction = %.2f\n", n, m, mean(M == 0)))

# ---- reference: Spearman of R, average ranks for ties ----
D_ref <- 1 - cor(M, method = "spearman")

# ---- GADES (write = TRUE returns the m x m matrix) ----
D_gpu_dense <- mtrx_distance(M, metric = "spearman", type = "gpu", sparse = FALSE, write = TRUE)
D_cpu_dense <- mtrx_distance(M, metric = "spearman", type = "cpu", sparse = FALSE, write = TRUE)
Msp <- as(Matrix(M, sparse = TRUE), "CsparseMatrix")
D_gpu_sparse <- mtrx_distance(Msp, metric = "spearman", type = "gpu", sparse = TRUE, write = TRUE)
D_cpu_sparse <- mtrx_distance(Msp, metric = "spearman", type = "cpu", sparse = TRUE, write = TRUE)

# ---- ArrayFire: write a CSV (objects x features) and run the dumper ----
csv_in <- tempfile(fileext = ".csv")
af_out <- tempfile(fileext = ".csv")
write.csv(t(M), csv_in)
rc <- system2(arrayfire_dump, c(csv_in, af_out), stdout = TRUE, stderr = TRUE)
cat(paste(rc, collapse = "\n"), "\n")
D_af <- as.matrix(read.csv(af_out, header = FALSE))
dimnames(D_af) <- dimnames(D_ref)

# ---- maximum absolute error off the diagonal against the reference ----
offdiag_maxerr <- function(D) {
  d <- abs(D - D_ref)
  diag(d) <- 0
  max(d)
}
cat("\n================ max |D - D_ref| (off-diagonal) ================\n")
res <- c(
  "GADES GPU dense" = offdiag_maxerr(D_gpu_dense),
  "GADES CPU dense" = offdiag_maxerr(D_cpu_dense),
  "GADES GPU sparse" = offdiag_maxerr(D_gpu_sparse),
  "GADES CPU sparse" = offdiag_maxerr(D_cpu_sparse),
  "ArrayFire (ordinal ranks)" = offdiag_maxerr(D_af)
)
for (nm in names(res)) {
  cat(sprintf("  %-26s %.6e\n", nm, res[nm]))
}

cat("\n---- reference D_ref[1:4,1:4] ----\n")
print(round(D_ref[1:4, 1:4], 4))
cat("\n---- GADES GPU dense   [1:4,1:4] ----\n")
print(round(D_gpu_dense[1:4, 1:4], 4))
cat("\n---- ArrayFire         [1:4,1:4] ----\n")
print(round(D_af[1:4, 1:4], 4))
cat("\n---- (ArrayFire - ref) [1:4,1:4] ----\n")
print(round((D_af - D_ref)[1:4, 1:4], 4))
