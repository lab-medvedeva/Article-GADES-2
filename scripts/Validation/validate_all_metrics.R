#!/usr/bin/env Rscript
# Every metric of GADES against a reference computed in pure R, with the error
# bounds quoted in the article.
#
# This script is new: it was written for this repository and is not part of
# the GADES source tree. The other scripts of this directory cover Kendall,
# Spearman and single paths of the other metrics; this one runs all six
# metrics through every mode and prints the maximum absolute and the maximum
# relative error of each combination.
#
#   Validated:  mtrx_distance() for euclidean, cosine, pearson, manhattan,
#               spearman and kendall; on the CPU, and on the GPU when
#               VALIDATE_GPU=1; dense and sparse input; one batch covering all
#               objects (same-batch block only) and a batch smaller than the
#               number of objects (same-batch and different-batch blocks).
#               mtrx_distance() is called as the benchmark driver test.R calls
#               it, except for write = TRUE and filename = "", which make it
#               return the distance matrix.
#   Reference:  euclidean, manhattan: dist()
#               pearson, spearman:    1 - cor()
#               cosine:               1 - cosine similarity, from crossprod()
#               kendall:              2 * discordant pairs / (n * (n - 1)),
#                                     kendall_reference() in common.R
#   Inputs:     synthetic, fixed seeds. "small" (150 features x 20 objects, all
#               metrics) and "large" (3000 features x 120 objects, without
#               kendall: the count in pure R is quadratic in the features).
#               The large input makes the CPU kernels of euclidean, cosine and
#               pearson take their BLAS path. Values lie on a 0.1 grid, so
#               every object has ties; the sparse input has 80 % zeros and
#               values of both signs.
#   Tolerance:  maximum relative error 1e-4 (RELATIVE_TOLERANCE). Entries whose
#               reference is zero (the diagonal) are compared by absolute error
#               with the same bound.
#   GPU:        optional, VALIDATE_GPU=1.
#   Output:     a table on the console and the same table in
#               validate_all_metrics.csv next to this script, or in the path
#               given as the first argument.
#   Run:        export GADES_ROOT=/path/to/GADES
#               Rscript validate_all_metrics.R [output.csv]
#   Exit status is 1 when a combination exceeds the tolerance or fails to run.

file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- getwd()
if (length(file_argument) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_argument[1])))
}
source(file.path(script_dir, "common.R"))
gpu_loaded <- load_gades(gpu = gpu_mode_from_environment())

RELATIVE_TOLERANCE <- 1e-4
METRICS <- c("euclidean", "cosine", "pearson", "manhattan", "spearman", "kendall")
# Sparse metrics the benchmark runs with the per_cell_pair layout; the other
# metrics keep the default layout.
PER_CELL_PAIR_METRICS <- c("manhattan", "spearman", "kendall")
SPARSE_DENSITY <- 0.2
MEMORY_LIMIT_GB <- as.numeric(Sys.getenv("MEM_LIMIT_GB", "12"))

# batch_size leaves more than one object in the last batch: mtrx_distance()
# slices the batches without drop = FALSE.
DATASETS <- list(
  list(
    name = "small",
    features = 150L,
    objects = 20L,
    batch_size = 8L,
    metrics = METRICS,
    seed = 101L
  ),
  list(
    name = "large",
    features = 3000L,
    objects = 120L,
    batch_size = 60L,
    metrics = setdiff(METRICS, "kendall"),
    seed = 202L
  )
)

output_path <- file.path(script_dir, "validate_all_metrics.csv")
arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) >= 1) {
  output_path <- arguments[1]
}

devices <- "cpu"
if (gpu_loaded) {
  devices <- c("cpu", "gpu")
}

# ---- inputs: features in rows, objects in columns, as in the benchmark ----

# Dense input: values on a 0.1 grid, so every column has ties.
generate_dense <- function(features, objects, seed) {
  set.seed(seed)
  values <- round(rexp(features * objects, rate = 0.5), 1)
  matrix(values, features, objects)
}

# Sparse input: zeros except for a share SPARSE_DENSITY of the entries, which
# lie on a 0.1 grid and are negative with probability 0.3.
generate_sparse <- function(features, objects, seed) {
  set.seed(seed)
  total <- features * objects
  mask <- runif(total) < SPARSE_DENSITY
  count <- sum(mask)
  magnitudes <- round(rexp(count, rate = 0.5), 1) + 0.1
  signs <- sample(c(-1, 1), count, replace = TRUE, prob = c(0.3, 0.7))
  values <- numeric(total)
  values[mask] <- magnitudes * signs
  matrix(values, features, objects)
}

count_distinct <- function(values) {
  length(unique(values))
}

# The correlation distances need non-constant columns, the rank metrics are
# meant to be checked on ties.
check_input <- function(M) {
  distinct <- apply(M, 2, count_distinct)
  if (any(distinct < 2)) {
    stop("the generated input has a constant column")
  }
  if (all(distinct == nrow(M))) {
    stop("the generated input has no ties")
  }
}

# ---- reference distances between the columns of M, in double precision ----

# dist() compares the rows of its argument, hence t(M); cor() and crossprod()
# compare columns.
# ASSUMPTION to confirm by a run: cosine and pearson distances are
# 1 - similarity. No R script of the GADES tree states it; the definition is
# read from the CPU kernels (src/metrics/cosine.cpp, src/metrics/pearson.cpp).
reference_distance <- function(M, metric) {
  if (metric == "euclidean") {
    reference <- as.matrix(dist(t(M), method = "euclidean"))
  } else if (metric == "manhattan") {
    reference <- as.matrix(dist(t(M), method = "manhattan"))
  } else if (metric == "cosine") {
    norms <- sqrt(colSums(M * M))
    reference <- 1 - crossprod(M) / outer(norms, norms)
  } else if (metric == "pearson") {
    reference <- 1 - cor(M, method = "pearson")
  } else if (metric == "spearman") {
    reference <- 1 - cor(M, method = "spearman")
  } else if (metric == "kendall") {
    reference <- kendall_reference(M)
  } else {
    stop(paste("unknown metric:", metric))
  }
  reference <- unname(reference)
  diag(reference) <- 0
  reference
}

# ---- GADES ----

sparse_layout_of <- function(metric, sparse) {
  if (sparse && metric %in% PER_CELL_PAIR_METRICS) {
    return("per_cell_pair")
  }
  "default"
}

gades_distance <- function(gades_input, metric, device, sparse, batch_size) {
  quiet(
    mtrx_distance(
      gades_input,
      batch_size = batch_size,
      metric = metric,
      type = device,
      sparse = sparse,
      write = TRUE,
      filename = "",
      sparse_layout = sparse_layout_of(metric, sparse),
      memory_limit_gb = MEMORY_LIMIT_GB
    )
  )
}

# ---- comparison ----

empty_errors <- function(note) {
  list(abs_error = NA_real_, rel_error = NA_real_, zero_error = NA_real_, note = note)
}

# abs_error:  maximum |GADES - reference| over all entries
# rel_error:  maximum |GADES - reference| / |reference| over the entries with
#             a nonzero reference
# zero_error: maximum |GADES| over the entries with a zero reference
measure_errors <- function(got, reference) {
  if (!is.matrix(got) || !is.numeric(got)) {
    return(empty_errors("mtrx_distance did not return a numeric matrix"))
  }
  if (!identical(dim(got), dim(reference))) {
    return(empty_errors("the result has the wrong dimensions"))
  }
  got <- unname(got)
  non_finite <- !is.finite(got)
  if (any(non_finite)) {
    note_format <- "the result has %d non-finite values, %d of them on the diagonal"
    return(empty_errors(sprintf(note_format, sum(non_finite), sum(diag(non_finite)))))
  }
  difference <- abs(got - reference)
  nonzero <- reference != 0
  errors <- empty_errors("")
  errors$abs_error <- max(difference)
  errors$rel_error <- max(difference[nonzero] / abs(reference[nonzero]))
  errors$zero_error <- max(difference[!nonzero])
  errors
}

error_text <- function(e) {
  paste("error:", conditionMessage(e))
}

run_case <- function(gades_input, reference, metric, device, sparse, batch_size) {
  got <- tryCatch(
    gades_distance(gades_input, metric, device, sparse, batch_size),
    error = error_text
  )
  if (is.character(got)) {
    return(empty_errors(got[1]))
  }
  measure_errors(got, reference)
}

status_of <- function(errors) {
  if (errors$note != "") {
    return("ERROR")
  }
  if (errors$rel_error > RELATIVE_TOLERANCE) {
    return("FAIL")
  }
  if (errors$zero_error > RELATIVE_TOLERANCE) {
    return("FAIL")
  }
  "OK"
}

# ---- run every combination ----

row_format <- "%-8s %-10s %-6s %-6s %-14s %-9s %12s %12s  %s\n"
header_line <- sprintf(
  row_format,
  "dataset", "metric", "device", "input", "layout", "blocks", "max abs err", "max rel err", "status"
)
cat(header_line)
cat(strrep("-", nchar(header_line) - 1), "\n", sep = "")

rows <- list()
for (dataset in DATASETS) {
  inputs <- list(
    dense = generate_dense(dataset$features, dataset$objects, dataset$seed),
    sparse = generate_sparse(dataset$features, dataset$objects, dataset$seed + 1L)
  )
  batch_sizes <- c("one batch" = dataset$objects, "batched" = dataset$batch_size)
  for (input_kind in names(inputs)) {
    M <- inputs[[input_kind]]
    check_input(M)
    sparse <- input_kind == "sparse"
    gades_input <- M
    if (sparse) {
      # readMM() in the benchmark driver returns a triplet matrix
      gades_input <- as(Matrix(M, sparse = TRUE), "TsparseMatrix")
    }
    for (metric in dataset$metrics) {
      reference <- reference_distance(M, metric)
      layout <- "-"
      if (sparse) {
        layout <- sparse_layout_of(metric, sparse)
      }
      for (device in devices) {
        for (blocks in names(batch_sizes)) {
          batch_size <- batch_sizes[[blocks]]
          errors <- run_case(gades_input, reference, metric, device, sparse, batch_size)
          status <- status_of(errors)
          abs_text <- sprintf("%.3e", errors$abs_error)
          rel_text <- sprintf("%.3e", errors$rel_error)
          row_line <- sprintf(
            row_format,
            dataset$name, metric, device, input_kind, layout, blocks, abs_text, rel_text, status
          )
          cat(row_line)
          if (errors$note != "") {
            cat(sprintf("    %s\n", errors$note))
          }
          rows[[length(rows) + 1]] <- data.frame(
            dataset = dataset$name,
            features = dataset$features,
            objects = dataset$objects,
            metric = metric,
            device = device,
            input = input_kind,
            layout = layout,
            blocks = blocks,
            batch_size = batch_size,
            max_abs_error = errors$abs_error,
            max_rel_error = errors$rel_error,
            max_abs_error_at_zero_reference = errors$zero_error,
            status = status,
            note = errors$note,
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
}
results <- do.call(rbind, rows)
write.csv(results, output_path, row.names = FALSE)

# ---- maxima ----

finite_maximum <- function(values) {
  values <- values[is.finite(values)]
  if (length(values) == 0) {
    return(NA_real_)
  }
  max(values)
}

cat("\nMaximum errors by metric\n")
summary_format <- "%-10s %12s %12s\n"
cat(sprintf(summary_format, "metric", "max abs err", "max rel err"))
for (metric in METRICS) {
  of_metric <- results[results$metric == metric, ]
  abs_text <- sprintf("%.3e", finite_maximum(of_metric$max_abs_error))
  rel_text <- sprintf("%.3e", finite_maximum(of_metric$max_rel_error))
  cat(sprintf(summary_format, metric, abs_text, rel_text))
}
cat(sprintf("\nOverall maximum absolute error: %.3e\n", finite_maximum(results$max_abs_error)))
cat(sprintf("Overall maximum relative error: %.3e\n", finite_maximum(results$max_rel_error)))
cat(sprintf("Tolerance on the relative error: %.0e\n", RELATIVE_TOLERANCE))
cat(sprintf("Devices: %s\n", paste(devices, collapse = ", ")))
cat(sprintf("Table written to %s\n", output_path))

failed <- sum(results$status != "OK")
if (failed == 0) {
  cat(sprintf("\nPASS: %d combinations within the tolerance\n", nrow(results)))
  quit(status = 0)
}
cat(sprintf("\nFAIL: %d of %d combinations\n", failed, nrow(results)))
quit(status = 1)
