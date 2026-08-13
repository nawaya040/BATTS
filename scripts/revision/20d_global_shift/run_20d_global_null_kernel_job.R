#!/usr/bin/env Rscript

script_path <- normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]),
  winslash = "/", mustWork = TRUE
)
script_dir <- dirname(script_path)
project_root <- normalizePath(file.path(script_dir, "..", "..", ".."),
                              winslash = "/", mustWork = TRUE)
source(file.path(project_root, "scripts", "boosting", "boosting_selection_common.R"))
source(file.path(script_dir, "20d_global_shift_common.R"))
source(file.path(script_dir, "20d_global_null_transform.R"))
source(file.path(script_dir, "global_null_comparator_common.R"))

args <- boosting_parse_args(commandArgs(TRUE))
mode <- boosting_mode(args)
scenario <- boosting_arg(args, "scenario", if (mode == "smoke") "global_shift" else NULL,
                         required = mode != "smoke")
method <- tolower(boosting_arg(args, "method", "kliep"))
seed <- boosting_int(boosting_arg(args, "seed", required = TRUE), "seed")
n0 <- boosting_int(boosting_arg(args, "n0", if (mode == "smoke") "80" else NULL,
                               required = mode != "smoke"), "n0")
n1 <- boosting_int(boosting_arg(args, "n1", if (mode == "smoke") "80" else NULL,
                               required = mode != "smoke"), "n1")
transformed <- boosting_bool(boosting_arg(args, "transformed", "false"), "transformed")
resume <- boosting_bool(boosting_arg(args, "resume", "false"), "resume")
output_dir <- normalizePath(boosting_arg(args, "output-dir", required = TRUE),
                            winslash = "/", mustWork = FALSE)
r_lib <- normalizePath(boosting_arg(args, "r-lib", required = TRUE),
                       winslash = "/", mustWork = TRUE)

if (!scenario %in% c("global_shift", "null")) stop("Unsupported scenario")
if (!method %in% c("kliep", "ulsif")) stop("--method must be kliep or ulsif")
if (identical(mode, "canonical") &&
    !paste(n0, n1, sep = "_") %in% c("5000_5000", "9000_1000")) {
  stop("Unsupported canonical sample sizes")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(output_dir)) stop("Unable to create output directory")
.libPaths(c(r_lib, .libPaths()))
required <- c("densratio", "digest", "pracma")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing packages: ", paste(missing, collapse = ", "))

task <- list(
  scenario = scenario, n0 = n0, n1 = n1, seed = seed,
  d = 20L, latent_dim = 4L, noise_sd = 0.1,
  transformed = transformed, beta_shape1 = 0.5, beta_shape2 = 10
)
simulation <- generate_global_null_20d(task)
truth_error <- validate_global_null_simulation(simulation)
data_hash <- global_null_data_hash(simulation)
labels <- simulation$group_labels
x0 <- simulation$data[labels == 0L, , drop = FALSE]
x1 <- simulation$data[labels == 1L, , drop = FALSE]

job_id <- sprintf(
  "global_null_%s_%s_n0-%d_n1-%d_transformed-%s_seed-%03d",
  method, scenario, n0, n1, tolower(as.character(transformed)), seed
)
result_path <- file.path(output_dir, paste0(job_id, ".rds"))
sidecar_path <- file.path(output_dir, paste0(job_id, ".done.rds"))
if (global_null_resume_state(
  result_path, sidecar_path, resume,
  list(job_id = job_id, method = method, task = task, data_hash = data_hash)
)) {
  message("Verified existing output; skipping: ", result_path)
  quit(save = "no", status = 0L)
}

set_reproducible_rng(700000L + seed + if (method == "ulsif") 10000L else 0L)
warnings <- character()
started <- Sys.time()
fit <- withCallingHandlers(
  if (method == "kliep") {
    densratio::KLIEP(x0, x1, verbose = FALSE)
  } else {
    densratio::uLSIF(x0, x1, verbose = FALSE)
  },
  warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)
ratio <- fit$compute_density_ratio(simulation$data)
estimate <- suppressWarnings(log(ratio))
finished <- Sys.time()
metrics <- global_null_metrics(estimate, simulation$true_log_ratio, labels)

source_files <- c(script_path, file.path(script_dir, "20d_global_shift_common.R"),
                  file.path(script_dir, "20d_global_null_transform.R"),
                  file.path(script_dir, "global_null_comparator_common.R"))
result <- list(
  schema_version = "1.0",
  job_id = job_id,
  method = method,
  task = task,
  data_hash = data_hash,
  estimate = estimate,
  metrics = metrics,
  diagnostics = list(
    symmetric_mse = global_null_symmetric_mse(metrics),
    nonpositive_ratio = sum(!is.finite(ratio) | ratio <= 0),
    truth_validation_max_error = truth_error
  ),
  provenance = list(
    start_time_utc = format(started, tz = "UTC", usetz = TRUE),
    end_time_utc = format(finished, tz = "UTC", usetz = TRUE),
    elapsed_seconds = as.numeric(difftime(finished, started, units = "secs")),
    r_version = R.version.string,
    rng_kind = RNGkind(),
    seed = 700000L + seed + if (method == "ulsif") 10000L else 0L,
    warnings = unique(warnings),
    package_version = as.character(utils::packageVersion("densratio")),
    source_hashes = tools::sha256sum(source_files),
    session_info = capture.output(utils::sessionInfo())
  )
)
global_null_atomic_save(result, result_path)
result_sha <- unname(tools::sha256sum(result_path))
sidecar <- list(
  schema_version = "1.0", job_id = job_id, method = method, task = task,
  data_hash = data_hash, result_path = normalizePath(result_path, winslash = "/"),
  result_bytes = file.info(result_path)$size, result_sha256 = result_sha,
  elapsed_seconds = result$provenance$elapsed_seconds,
  warnings = unique(warnings), metrics = metrics
)
global_null_atomic_save(sidecar, sidecar_path)
cat("Completed: ", result_path, "\n", sep = "")
