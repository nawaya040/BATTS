#!/usr/bin/env Rscript

if (!identical(environment(), globalenv())) {
  stop("This script must be run with Rscript.")
}

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
mode <- if (is.null(args$mode)) "pilot" else args$mode
if (!mode %in% c("smoke", "calibration-smoke", "preflight", "pilot",
                 "calibration-unbalanced")) {
  stop(paste(
    "--mode must be smoke, calibration-smoke, preflight, pilot,",
    "or calibration-unbalanced"
  ))
}
if (is.null(args[["output-dir"]])) {
  stop("Missing required argument --output-dir=PATH")
}
output_dir <- normalizePath(
  args[["output-dir"]],
  winslash = "/",
  mustWork = FALSE
)
workers <- if (is.null(args$workers)) {
  if (mode == "smoke") 1L else 2L
} else {
  as.integer(args$workers)
}
if (!is.finite(workers) || workers < 1L) {
  stop("--workers must be a positive integer")
}
if (mode %in% c("pilot", "calibration-unbalanced") && workers != 2L) {
  stop("The pilot design requires exactly 2 workers")
}
if (mode == "preflight" && workers != 2L) {
  stop("The preflight design requires exactly 2 workers")
}

if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0L) {
  stop("Refusing to use a non-empty output directory: ", output_dir)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(output_dir)) {
  stop("Unable to create output directory: ", output_dir)
}

script_dir <- dirname(script_path())
common_path <- file.path(script_dir, "20d_global_shift_common.R")
source(common_path, local = globalenv())
assert_required_packages()
assert_reviewpkg_version()

full_parameters <- list(
  d = 20L,
  latent_dim = 4L,
  shift = 1,
  noise_sd = 0.1,
  num_trees = 200L,
  size_burnin = 2000L,
  size_backfitting = 1000L
)
smoke_parameters <- modifyList(
  full_parameters,
  list(
    num_trees = 5L,
    size_burnin = 10L,
    size_backfitting = 20L
  )
)
calibration_parameters <- modifyList(
  full_parameters,
  list(calibration_levels = seq(0.01, 0.99, by = 0.01))
)
calibration_smoke_parameters <- modifyList(
  smoke_parameters,
  list(calibration_levels = seq(0.01, 0.99, by = 0.01))
)

make_task <- function(n0, n1, seed, parameters) {
  c(list(n0 = as.integer(n0), n1 = as.integer(n1), seed = as.integer(seed)), parameters)
}

if (mode == "smoke") {
  tasks <- list(make_task(60L, 60L, 1L, smoke_parameters))
} else if (mode == "calibration-smoke") {
  tasks <- list(make_task(60L, 60L, 1L, calibration_smoke_parameters))
} else if (mode == "preflight") {
  tasks <- list(
    make_task(5000L, 5000L, 1L, full_parameters),
    make_task(9000L, 1000L, 1L, full_parameters)
  )
} else if (mode == "pilot") {
  tasks <- c(
    lapply(1:10, function(seed) make_task(5000L, 5000L, seed, full_parameters)),
    lapply(1:10, function(seed) make_task(9000L, 1000L, seed, full_parameters))
  )
} else {
  tasks <- lapply(
    1:10,
    function(seed) make_task(9000L, 1000L, seed, calibration_parameters)
  )
}

design <- do.call(rbind, lapply(tasks, function(task) {
  data.frame(
    mode = mode,
    n0 = task$n0,
    n1 = task$n1,
    seed = task$seed,
    d = task$d,
    latent_dim = task$latent_dim,
    shift = task$shift,
    noise_sd = task$noise_sd,
    num_trees = task$num_trees,
    size_burnin = task$size_burnin,
    size_backfitting = task$size_backfitting,
    calibration_levels = if (is.null(task$calibration_levels)) {
      NA_character_
    } else {
      paste(task$calibration_levels, collapse = ";")
    },
    workers = workers,
    stringsAsFactors = FALSE
  )
}))
utils::write.csv(design, file.path(output_dir, "run_design.csv"), row.names = FALSE)

Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1"
)

metadata <- c(
  paste0("mode=", mode),
  paste0("workers=", workers),
  paste0("task_count=", length(tasks)),
  paste0("start_time_utc=", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  paste0("r_version=", R.version.string),
  paste0("reviewpkg_remote_sha=", utils::packageDescription("ReviewPkg")[["Review-Source-ID"]]),
  paste0("rng_kind=", paste(RNGkind(), collapse = ",")),
  "parallel_backend=PSOCK outer parallelism by independent seed",
  "OMP_NUM_THREADS=1",
  "OPENBLAS_NUM_THREADS=1",
  "MKL_NUM_THREADS=1",
  paste0("common_sha256=", sha256_file(common_path)),
  paste0("runner_sha256=", sha256_file(script_path()))
)
writeLines(metadata, file.path(output_dir, "run_metadata.txt"), useBytes = TRUE)

source_paths <- c(common_path, script_path())
worker_log <- file.path(output_dir, "workers.log")
cluster <- parallel::makePSOCKcluster(
  workers,
  outfile = worker_log,
  rscript_args = "--vanilla"
)
on.exit(parallel::stopCluster(cluster), add = TRUE)
parallel::clusterCall(
  cluster,
  function() {
    Sys.setenv(
      OMP_NUM_THREADS = "1",
      OPENBLAS_NUM_THREADS = "1",
      MKL_NUM_THREADS = "1"
    )
    NULL
  }
)

status <- parallel::parLapplyLB(
  cluster,
  tasks,
  function(task, output_dir, common_path, source_paths) {
    tryCatch({
      source(common_path, local = globalenv())
      run_global_shift_fit(task, output_dir, source_paths)
    }, error = function(e) {
      list(
        status = "failed",
        n0 = task$n0,
        n1 = task$n1,
        seed = task$seed,
        elapsed_seconds = NA_real_,
        output = NA_character_,
        sha256 = NA_character_,
        warnings = conditionMessage(e)
      )
    })
  },
  output_dir = output_dir,
  common_path = common_path,
  source_paths = source_paths
)
parallel::stopCluster(cluster)
on.exit(NULL, add = FALSE)

status_table <- do.call(rbind, lapply(status, as.data.frame, stringsAsFactors = FALSE))
utils::write.csv(status_table, file.path(output_dir, "task_status.csv"), row.names = FALSE)

if (any(status_table$status != "completed")) {
  stop(
    "One or more tasks failed. See ",
    file.path(output_dir, "task_status.csv"),
    " and ", worker_log
  )
}

cat("Completed ", nrow(status_table), " task(s).\n", sep = "")
cat("Output: ", output_dir, "\n", sep = "")
