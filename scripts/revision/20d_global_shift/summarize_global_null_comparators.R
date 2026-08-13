#!/usr/bin/env Rscript

parse_args <- function(args) {
  out <- list()
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) stop("Use --name=value")
    pair <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[pair[[1L]]]] <- paste(pair[-1L], collapse = "=")
  }
  out
}
bool <- function(x) {
  if (tolower(x) %in% c("true", "1", "yes")) return(TRUE)
  if (tolower(x) %in% c("false", "0", "no")) return(FALSE)
  stop("Boolean value required")
}
args <- parse_args(commandArgs(TRUE))
required <- c("boosting-dir", "kernel-dir", "cdc-dir", "output-dir")
missing <- required[vapply(required, function(x) is.null(args[[x]]), logical(1))]
if (length(missing)) stop("Missing: ", paste(missing, collapse = ", "))
expect_complete <- bool(if (is.null(args[["expect-complete"]])) "false" else args[["expect-complete"]])
dirs <- lapply(args[required[1:3]], normalizePath, winslash = "/", mustWork = TRUE)
output_dir <- normalizePath(args[["output-dir"]], winslash = "/", mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (length(list.files(output_dir, all.files = TRUE, no.. = TRUE))) {
  stop("Refusing to write into a non-empty summary directory")
}

result_files <- function(path) {
  files <- list.files(path, pattern = "[.]rds$", full.names = TRUE, recursive = TRUE)
  files[!grepl("[.](manifest|done)[.]rds$", files)]
}
verify_hash <- function(result_path, manifest_path, field) {
  if (!file.exists(manifest_path)) stop("Missing sidecar: ", manifest_path)
  manifest <- readRDS(manifest_path)
  actual <- unname(tools::sha256sum(result_path))
  if (!identical(actual, manifest[[field]])) stop("Hash mismatch: ", result_path)
  manifest
}
task_key <- function(scenario, n0, n1, transformed, seed) {
  sprintf("%s|%d|%d|%s|%03d", scenario, n0, n1,
          tolower(as.character(transformed)), seed)
}

metric_rows <- list()
hash_rows <- list()
index <- 0L
hindex <- 0L

boosting_files <- result_files(dirs[[1L]])
for (path in boosting_files) {
  manifest <- verify_hash(path, sub("[.]rds$", ".manifest.rds", path), "output_sha256")
  object <- readRDS(path)
  metadata <- object$metadata
  result <- object$result
  key <- task_key(metadata$config$scenario, metadata$config$n0, metadata$config$n1,
                  metadata$config$transformed, metadata$seed)
  metrics <- rbind(result$proposed$metrics, result$adaboost$metrics)
  metrics$key <- key
  metrics$scenario <- metadata$config$scenario
  metrics$n0 <- metadata$config$n0
  metrics$n1 <- metadata$config$n1
  metrics$transformed <- metadata$config$transformed
  metrics$seed <- metadata$seed
  index <- index + 1L
  metric_rows[[index]] <- metrics
  hindex <- hindex + 1L
  hash_rows[[hindex]] <- data.frame(key = key, source = "boosting", method = "boosting",
                                    data_hash = result$design$data_hash, stringsAsFactors = FALSE)
}

kernel_files <- result_files(dirs[[2L]])
for (path in kernel_files) {
  manifest <- verify_hash(path, sub("[.]rds$", ".done.rds", path), "result_sha256")
  object <- readRDS(path)
  task <- object$task
  key <- task_key(task$scenario, task$n0, task$n1, task$transformed, task$seed)
  group0 <- object$metrics[object$metrics$group == "group0", ]
  group1 <- object$metrics[object$metrics$group == "group1", ]
  index <- index + 1L
  metric_rows[[index]] <- data.frame(
    method = object$method, selection = "package_default", split = "train",
    mse_group0 = group0$mse, mse_group1 = group1$mse,
    mse_symmetric = mean(c(group0$mse, group1$mse)),
    nonfinite = group0$nonfinite_n + group1$nonfinite_n,
    max_abs_finite = max(group0$max_abs_estimate, group1$max_abs_estimate),
    key = key, scenario = task$scenario, n0 = task$n0, n1 = task$n1,
    transformed = task$transformed, seed = task$seed, stringsAsFactors = FALSE
  )
  hindex <- hindex + 1L
  hash_rows[[hindex]] <- data.frame(key = key, source = "kernel", method = object$method,
                                    data_hash = object$data_hash, stringsAsFactors = FALSE)
}

cdc_files <- result_files(dirs[[3L]])
for (path in cdc_files) {
  manifest <- verify_hash(path, sub("[.]rds$", ".done.rds", path), "result_sha256")
  object <- readRDS(path)
  task <- object$task
  key <- task_key(task$scenario, task$n0, task$n1, task$transformed, task$seed)
  for (variant in c("submitted", "stable")) {
    metrics <- object$metrics[[variant]]
    group0 <- metrics[metrics$group == "group0", ]
    group1 <- metrics[metrics$group == "group1", ]
    index <- index + 1L
    metric_rows[[index]] <- data.frame(
      method = "cdc", selection = variant, split = "train",
      mse_group0 = group0$mse, mse_group1 = group1$mse,
      mse_symmetric = mean(c(group0$mse, group1$mse)),
      nonfinite = group0$nonfinite_n + group1$nonfinite_n,
      max_abs_finite = max(group0$max_abs_estimate, group1$max_abs_estimate),
      key = key, scenario = task$scenario, n0 = task$n0, n1 = task$n1,
      transformed = task$transformed, seed = task$seed, stringsAsFactors = FALSE
    )
  }
  hindex <- hindex + 1L
  hash_rows[[hindex]] <- data.frame(key = key, source = "cdc", method = "cdc",
                                    data_hash = object$data_hash, stringsAsFactors = FALSE)
}

metrics <- do.call(rbind, metric_rows)
hashes <- do.call(rbind, hash_rows)
if (expect_complete &&
    !identical(c(length(boosting_files), length(kernel_files), length(cdc_files)),
               c(400L, 800L, 400L))) {
  stop("Canonical comparator output counts are incomplete")
}
bad_keys <- names(Filter(function(values) length(unique(values)) != 1L,
                         split(hashes$data_hash, hashes$key)))
if (length(bad_keys)) stop("Data hashes disagree across methods for ", length(bad_keys), " tasks")

summary <- aggregate(
  cbind(mse_group0, mse_group1, mse_symmetric, nonfinite) ~
    method + selection + scenario + n0 + n1 + transformed,
  data = metrics, FUN = mean
)
utils::write.csv(metrics, file.path(output_dir, "comparator_metrics_by_job.csv"), row.names = FALSE)
utils::write.csv(summary, file.path(output_dir, "comparator_metrics_summary.csv"), row.names = FALSE)
utils::write.csv(hashes, file.path(output_dir, "data_hash_audit.csv"), row.names = FALSE)
saveRDS(list(
  schema_version = "1.0", input_counts = c(boosting = length(boosting_files),
                                           kernel = length(kernel_files), cdc = length(cdc_files)),
  data_hashes_consistent = TRUE, metrics = metrics, summary = summary,
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
), file.path(output_dir, "comparator_summary.rds"), compress = "xz")
cat(sprintf("Summarized %d boosting, %d kernel, and %d CDC outputs\n",
            length(boosting_files), length(kernel_files), length(cdc_files)))
