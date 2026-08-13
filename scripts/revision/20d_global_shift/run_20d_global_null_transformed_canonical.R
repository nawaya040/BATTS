#!/usr/bin/env Rscript

parse_cli <- function(args) {
  values <- list()
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) {
      stop("Arguments must have the form --name=value: ", arg)
    }
    pair <- strsplit(substring(arg, 3L), "=", fixed = TRUE)[[1L]]
    values[[pair[1L]]] <- paste(pair[-1L], collapse = "=")
  }
  values
}

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) != 1L) {
    stop("Unable to determine script path")
  }
  normalizePath(sub("^--file=", "", file_arg), winslash = "/", mustWork = TRUE)
}

args <- parse_cli(commandArgs(TRUE))
mode <- if (is.null(args$mode)) "smoke" else args$mode
if (!mode %in% c("smoke", "preflight", "canonical")) {
  stop("--mode must be smoke, preflight, or canonical")
}
if (identical(mode, "canonical") &&
    !identical(args[["confirm-canonical"]], "YES")) {
  stop("Canonical mode requires --confirm-canonical=YES")
}
if (is.null(args[["output-dir"]]) || is.null(args[["code-commit"]])) {
  stop("Required arguments: --output-dir=PATH --code-commit=HASH")
}
output_dir <- normalizePath(args[["output-dir"]], winslash = "/", mustWork = FALSE)
code_commit <- tolower(args[["code-commit"]])
if (!grepl("^[0-9a-f]{40}$", code_commit)) {
  stop("--code-commit must be a full 40-character Git commit")
}
workers <- if (is.null(args$workers)) 4L else as.integer(args$workers)
if (is.na(workers) || workers < 1L || workers > 8L) {
  stop("--workers must be an integer from 1 through 8")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(output_dir)) {
  stop("Unable to create output directory: ", output_dir)
}

script <- script_path()
script_dir <- dirname(script)
project_root <- normalizePath(file.path(script_dir, "..", "..", ".."),
                              winslash = "/", mustWork = TRUE)
if (identical(mode, "canonical")) {
  git_commit <- system2("git", c("-C", shQuote(project_root), "rev-parse", "HEAD"),
                        stdout = TRUE, stderr = FALSE)
  if (!is.null(attr(git_commit, "status")) || length(git_commit) != 1L ||
      !identical(tolower(trimws(git_commit)), code_commit)) {
    stop("--code-commit does not match the checked-out Git commit")
  }
  git_status <- system2("git", c("-C", shQuote(project_root), "status", "--porcelain"),
                        stdout = TRUE, stderr = FALSE)
  if (!is.null(attr(git_status, "status")) || length(git_status)) {
    stop("Canonical mode requires a clean Git worktree")
  }
}
canonical_common_path <- file.path(script_dir, "20d_global_null_transformed_common.R")
base_common_path <- file.path(script_dir, "20d_global_shift_common.R")
transform_path <- file.path(script_dir, "20d_global_null_transform.R")
source(base_common_path, local = globalenv())
source(transform_path, local = globalenv())
source(canonical_common_path, local = globalenv())
assert_required_packages()
assert_batts_version()

make_task <- function(scenario, n0, n1, seed) {
  list(
    scenario = scenario,
    n0 = as.integer(n0),
    n1 = as.integer(n1),
    seed = as.integer(seed),
    d = 20L,
    latent_dim = 4L,
    noise_sd = 0.1,
    transformed = TRUE,
    beta_shape1 = 0.5,
    beta_shape2 = 10,
    num_trees = 200L,
    size_burnin = 2000L,
    size_backfitting = 1000L,
    calibration_levels = seq(0.01, 0.99, by = 0.01)
  )
}

full_tasks <- list()
task_index <- 0L
for (seed in 1:50) {
  for (scenario in c("global_shift", "null")) {
    for (sample_sizes in list(c(5000L, 5000L), c(9000L, 1000L))) {
      task_index <- task_index + 1L
      full_tasks[[task_index]] <- make_task(
        scenario,
        sample_sizes[1L],
        sample_sizes[2L],
        seed
      )
    }
  }
}
full_design <- do.call(rbind, lapply(full_tasks, function(task) {
  data.frame(
    scenario = task$scenario,
    n0 = task$n0,
    n1 = task$n1,
    seed = task$seed,
    d = task$d,
    latent_dim = task$latent_dim,
    noise_sd = task$noise_sd,
    transformed = task$transformed,
    beta_shape1 = task$beta_shape1,
    beta_shape2 = task$beta_shape2,
    num_trees = task$num_trees,
    size_burnin = task$size_burnin,
    size_backfitting = task$size_backfitting,
    stringsAsFactors = FALSE
  )
}))
if (nrow(full_design) != 200L || anyDuplicated(full_design[c("scenario", "n0", "n1", "seed")])) {
  stop("Invalid canonical design")
}

if (mode == "smoke") {
  tasks <- lapply(
    list(
      c("global_shift", 60L, 60L),
      c("null", 60L, 60L)
    ),
    function(specification) {
      task <- make_task(
        specification[1L],
        as.integer(specification[2L]),
        as.integer(specification[3L]),
        1L
      )
      task$num_trees <- 5L
      task$size_burnin <- 10L
      task$size_backfitting <- 20L
      task
    }
  )
} else if (mode == "preflight") {
  tasks <- Filter(function(task) task$seed == 1L, full_tasks)
} else {
  tasks <- full_tasks
}

source_paths <- c(base_common_path, transform_path, canonical_common_path, script)
source_hashes <- data.frame(
  path = normalizePath(source_paths, winslash = "/", mustWork = TRUE),
  sha256 = vapply(source_paths, sha256_file, character(1)),
  stringsAsFactors = FALSE
)
design_hash <- digest::digest(full_design, algo = "sha256")
contract <- list(
  schema_version = "1.1",
  code_commit = code_commit,
  batts_remote_sha = EXPECTED_BATTS_SHA,
  full_design_hash = design_hash,
  workers = workers,
  worker_threads = 1L,
  posterior_draws_saved = TRUE,
  posterior_draw_dimension = c(10000L, 1000L),
  source_hashes = source_hashes
)
contract_hash <- digest::digest(contract, algo = "sha256")
contract$contract_hash <- contract_hash
contract_path <- file.path(output_dir, "run_contract.rds")
design_path <- file.path(output_dir, "canonical_design.csv")
if (file.exists(contract_path)) {
  observed_contract <- readRDS(contract_path)
  if (!identical(observed_contract, contract)) {
    stop("Existing run contract does not match the current code/design")
  }
  observed_design <- utils::read.csv(design_path, stringsAsFactors = FALSE)
  if (!identical(observed_design, full_design)) {
    stop("Existing canonical design does not match")
  }
} else {
  if (length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0L) {
    stop("Refusing to initialize contract in a non-empty output directory")
  }
  atomic_save_rds(contract, contract_path, compress = "gzip")
  utils::write.csv(full_design, design_path, row.names = FALSE)
}

Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1"
)

worker_log <- file.path(output_dir, paste0("workers_", mode, ".log"))
cluster <- parallel::makePSOCKcluster(
  workers,
  outfile = worker_log,
  rscript_args = "--vanilla"
)
on.exit(parallel::stopCluster(cluster), add = TRUE)
parallel::clusterCall(cluster, function() {
  Sys.setenv(
    OMP_NUM_THREADS = "1",
    OPENBLAS_NUM_THREADS = "1",
    MKL_NUM_THREADS = "1"
  )
  NULL
})

status <- parallel::clusterApplyLB(
  cluster,
  tasks,
  function(task, output_dir, base_common_path, transform_path,
           canonical_common_path, source_paths, contract_hash, code_commit) {
    tryCatch({
      source(base_common_path, local = globalenv())
      source(transform_path, local = globalenv())
      source(canonical_common_path, local = globalenv())
      run_transformed_canonical_task(
        task,
        output_dir,
        contract_hash,
        source_paths,
        code_commit
      )
    }, error = function(e) {
      data.frame(
        status = "failed",
        scenario = task$scenario,
        n0 = task$n0,
        n1 = task$n1,
        seed = task$seed,
        elapsed_seconds = NA_real_,
        result_path = NA_character_,
        result_sha256 = NA_character_,
        warnings = conditionMessage(e),
        stringsAsFactors = FALSE
      )
    })
  },
  output_dir = output_dir,
  base_common_path = base_common_path,
  transform_path = transform_path,
  canonical_common_path = canonical_common_path,
  source_paths = source_paths,
  contract_hash = contract_hash,
  code_commit = code_commit
)
parallel::stopCluster(cluster)
on.exit(NULL, add = FALSE)

status_table <- do.call(rbind, status)
status_path <- file.path(
  output_dir,
  paste0("task_status_", mode, "_", format(Sys.time(), "%Y%m%dT%H%M%S"), ".csv")
)
utils::write.csv(status_table, status_path, row.names = FALSE)
if (any(status_table$status == "failed")) {
  stop("At least one task failed. See ", status_path, " and ", worker_log)
}

cat("Mode: ", mode, "\n", sep = "")
cat("Transformed: true\n")
cat("Tasks returned: ", nrow(status_table), "\n", sep = "")
cat("Completed: ", sum(status_table$status == "completed"), "\n", sep = "")
cat("Skipped verified: ", sum(status_table$status == "skipped_verified"), "\n", sep = "")
cat("Output: ", output_dir, "\n", sep = "")
