# Shared code of the validation scripts. Every script sources this file from
# its own directory.
#
# load_gades(gpu) loads GADES from the built source tree named by the
# environment variable GADES_ROOT: build/mtrx_cpu.so, build/mtrx.so (the GPU
# library) and R/mtrx.R, the files the benchmark driver loads. When build/ does
# not hold a shared object, lib/ is tried (the default output directory of the
# CMake project).
#   gpu = "required": stop when the GPU library is missing
#   gpu = "optional": load the GPU library when it is present
#   gpu = "none":     load the CPU library only
# The value is TRUE when the GPU library has been loaded.

find_shared_object <- function(gades_root, file_name) {
  candidates <- file.path(gades_root, c("build", "lib"), file_name)
  present <- candidates[file.exists(candidates)]
  if (length(present) == 0) {
    return("")
  }
  present[1]
}

ignore_error <- function(e) {
  NULL
}

not_loaded <- function(e) {
  FALSE
}

load_gpu_library <- function(gpu_library) {
  if (!is.loaded("matrix_Kendall_distance_same_block")) {
    dyn.load(gpu_library)
  }
  # The package calls check_gpu when it is loaded; it reports the GPU found.
  tryCatch(.C("check_gpu", PACKAGE = "mtrx"), error = ignore_error)
  TRUE
}

load_gades <- function(gpu = "optional") {
  gades_root <- Sys.getenv("GADES_ROOT")
  if (gades_root == "") {
    not_set_message <- "GADES_ROOT is not set. Point it to a built GADES source tree: export GADES_ROOT=/path/to/GADES"
    stop(not_set_message)
  }
  interface_path <- file.path(gades_root, "R", "mtrx.R")
  if (!file.exists(interface_path)) {
    stop(paste("R/mtrx.R not found in GADES_ROOT:", gades_root))
  }
  cpu_library <- find_shared_object(gades_root, "mtrx_cpu.so")
  if (cpu_library == "") {
    stop(paste("mtrx_cpu.so not found in build/ or lib/ of GADES_ROOT:", gades_root))
  }
  suppressMessages({
    library(Matrix)
    library(glue)
  })
  if (!is.loaded("matrix_Kendall_distance_same_block_cpu")) {
    dyn.load(cpu_library)
  }
  gpu_loaded <- FALSE
  if (gpu != "none") {
    gpu_library <- find_shared_object(gades_root, "mtrx.so")
    if (gpu_library == "" && gpu == "required") {
      stop(paste("mtrx.so (the GPU library) not found in build/ or lib/ of GADES_ROOT:", gades_root))
    }
    if (gpu_library != "" && gpu == "required") {
      gpu_loaded <- load_gpu_library(gpu_library)
    }
    if (gpu_library != "" && gpu == "optional") {
      gpu_loaded <- tryCatch(load_gpu_library(gpu_library), error = not_loaded)
    }
  }
  source(interface_path)
  gpu_loaded
}

# "required" when VALIDATE_GPU=1 is set, otherwise "none".
gpu_mode_from_environment <- function() {
  if (Sys.getenv("VALIDATE_GPU") == "1") {
    return("required")
  }
  "none"
}

# Evaluates an expression with the console output of GADES discarded.
quiet <- function(expr) {
  sink(tempfile())
  on.exit(sink())
  force(expr)
}

# Kendall distance between the columns of M, from the definition:
# 2 * (number of row pairs ordered oppositely by the two columns) / (n * (n - 1)).
# A pair with a tie in either column is not discordant.
kendall_reference <- function(M) {
  n <- nrow(M)
  m <- ncol(M)
  D <- matrix(0, m, m)
  for (a in 1:(m - 1)) {
    for (b in (a + 1):m) {
      dx <- outer(M[, a], M[, a], "-")
      dy <- outer(M[, b], M[, b], "-")
      disc <- sum((dx * dy) < 0) / 2
      d <- disc * 2 / (n * (n - 1))
      D[a, b] <- d
      D[b, a] <- d
    }
  }
  D
}

# Sparse matrix with the given share of nonzero entries: integers 1..10,
# with both signs when signed is TRUE.
generate_sparse_matrix <- function(n_genes, n_cells, density, signed, seed) {
  set.seed(seed)
  M <- matrix(0, n_genes, n_cells)
  mask <- runif(n_genes * n_cells) < density
  if (signed) {
    vals <- sample(c(-10:-1, 1:10), sum(mask), replace = TRUE)
  } else {
    vals <- sample(1:10, sum(mask), replace = TRUE)
  }
  M[mask] <- vals
  as(Matrix(M, sparse = TRUE), "CsparseMatrix")
}

# Path of a helper binary: the environment variable when it is set, otherwise
# the file of that name next to the scripts.
find_helper_binary <- function(script_dir, file_name, variable) {
  binary_path <- Sys.getenv(variable)
  if (binary_path == "") {
    binary_path <- file.path(script_dir, file_name)
  }
  if (!file.exists(binary_path)) {
    stop(paste0(file_name, " not found at ", binary_path, ": build it (see README.md) or set ", variable))
  }
  binary_path
}
