#!/usr/bin/env Rscript

script_path <- normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]]),
  winslash = "/", mustWork = TRUE
)
script_dir <- dirname(script_path)
project_root <- normalizePath(file.path(script_dir, "..", "..", ".."),
                              winslash = "/", mustWork = TRUE)
source(file.path(project_root, "scripts", "boosting", "boosting_selection_common.R"))
source(file.path(script_dir, "global_null_comparator_common.R"))

args <- boosting_parse_args(commandArgs(TRUE))
resume <- boosting_bool(boosting_arg(args, "resume", "false"), "resume")
input_path <- normalizePath(boosting_arg(args, "boosting-result", required = TRUE),
                            winslash = "/", mustWork = TRUE)
output_dir <- normalizePath(boosting_arg(args, "output-dir", required = TRUE),
                            winslash = "/", mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (!requireNamespace("digest", quietly = TRUE)) stop("Missing package: digest")

input_sha <- digest::digest(file = input_path, algo = "sha256", serialize = FALSE)
input <- readRDS(input_path)
metadata <- input$metadata
boosting <- input$result
task <- list(
  scenario = metadata$config$scenario,
  n0 = metadata$config$n0,
  n1 = metadata$config$n1,
  seed = metadata$seed,
  transformed = metadata$config$transformed
)
labels <- boosting$design$labels_train
truth <- boosting$design$truth_train
drt <- boosting$adaboost$estimates$train$exponential_loss
if (length(labels) != length(truth) || length(truth) != length(drt)) {
  stop("Boosting output dimensions are inconsistent")
}

logit_score <- drt - log(task$n1 / task$n0)
started <- Sys.time()
fit <- global_null_fit_cdc(logit_score, labels)
prediction <- global_null_predict_cdc(fit, logit_score)
finished <- Sys.time()
submitted_metrics <- global_null_metrics(prediction$submitted, truth, labels)
stable_metrics <- global_null_metrics(prediction$stable, truth, labels)

job_id <- sprintf(
  "global_null_cdc_%s_n0-%d_n1-%d_transformed-%s_seed-%03d",
  task$scenario, task$n0, task$n1,
  tolower(as.character(task$transformed)), task$seed
)
result_path <- file.path(output_dir, paste0(job_id, ".rds"))
sidecar_path <- file.path(output_dir, paste0(job_id, ".done.rds"))
if (global_null_resume_state(
  result_path, sidecar_path, resume,
  list(job_id = job_id, task = task, data_hash = boosting$design$data_hash,
       input = list(path = input_path, sha256 = input_sha))
)) {
  message("Verified existing output; skipping: ", result_path)
  quit(save = "no", status = 0L)
}
result <- list(
  schema_version = "1.0",
  job_id = job_id,
  task = task,
  data_hash = boosting$design$data_hash,
  input = list(path = input_path, sha256 = input_sha),
  estimates = list(submitted = prediction$submitted, stable = prediction$stable),
  metrics = list(submitted = submitted_metrics, stable = stable_metrics),
  diagnostics = prediction$diagnostics,
  provenance = list(
    start_time_utc = format(started, tz = "UTC", usetz = TRUE),
    end_time_utc = format(finished, tz = "UTC", usetz = TRUE),
    elapsed_seconds = as.numeric(difftime(finished, started, units = "secs")),
    r_version = R.version.string,
    source_hash = unname(tools::sha256sum(script_path)),
    session_info = capture.output(utils::sessionInfo())
  )
)
global_null_atomic_save(result, result_path)
result_sha <- unname(tools::sha256sum(result_path))
sidecar <- list(
  schema_version = "1.0", job_id = job_id, task = task,
  data_hash = result$data_hash, input = result$input,
  result_path = normalizePath(result_path, winslash = "/"),
  result_bytes = file.info(result_path)$size, result_sha256 = result_sha,
  elapsed_seconds = result$provenance$elapsed_seconds,
  diagnostics = prediction$diagnostics,
  submitted_metrics = submitted_metrics,
  stable_metrics = stable_metrics
)
global_null_atomic_save(sidecar, sidecar_path)
cat("Completed: ", result_path, "\n", sep = "")
