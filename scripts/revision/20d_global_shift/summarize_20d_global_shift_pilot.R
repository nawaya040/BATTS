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

args <- parse_cli(commandArgs(TRUE))
if (is.null(args[["input-dir"]]) || is.null(args[["output-dir"]])) {
  stop("Required arguments: --input-dir=PATH --output-dir=PATH")
}
input_dir <- normalizePath(args[["input-dir"]], winslash = "/", mustWork = TRUE)
output_dir <- normalizePath(args[["output-dir"]], winslash = "/", mustWork = FALSE)
expected_results <- if (is.null(args$expected)) 20L else as.integer(args$expected)
if (!is.finite(expected_results) || expected_results < 1L) {
  stop("--expected must be a positive integer")
}
if (!requireNamespace("digest", quietly = TRUE)) {
  stop("The digest package is required")
}
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0L) {
  stop("Refusing to overwrite a non-empty summary directory: ", output_dir)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

result_paths <- sort(list.files(
  input_dir,
  pattern = "^global_shift_n0_[0-9]+_n1_[0-9]+_seed_[0-9]+\\.rds$",
  full.names = TRUE
))
if (length(result_paths) != expected_results) {
  stop("Expected ", expected_results, " result files; found ", length(result_paths))
}

results <- lapply(result_paths, readRDS)
if (any(vapply(results, function(x) !identical(x$schema_version, "1.0"), logical(1)))) {
  stop("Unexpected result schema")
}
if (any(vapply(results, function(x) !identical(x$provenance$batts_remote_sha,
                                                "6f625bad83702b36e5480be1ed1343258a9b075a"),
               logical(1)))) {
  stop("BATTS provenance mismatch")
}

metric_rows <- list()
bin_rows <- list()
for (i in seq_along(results)) {
  result <- results[[i]]
  identity <- data.frame(
    n0 = result$task$n0,
    n1 = result$task$n1,
    seed = result$task$seed,
    elapsed_seconds = result$provenance$elapsed_seconds,
    stringsAsFactors = FALSE
  )
  metric_rows[[i]] <- cbind(identity[rep(1L, nrow(result$metrics)), ], result$metrics)
  bin_rows[[i]] <- cbind(identity[rep(1L, nrow(result$binned_metrics)), ],
                         result$binned_metrics)
}
seed_metrics <- do.call(rbind, metric_rows)
seed_bins <- do.call(rbind, bin_rows)

key <- paste(seed_metrics$n0, seed_metrics$n1, seed_metrics$seed, seed_metrics$group)
if (anyDuplicated(key)) {
  stop("Duplicate seed-level metric keys")
}
if (any(!is.finite(seed_metrics$coverage_95)) ||
    any(seed_metrics$coverage_95 < 0 | seed_metrics$coverage_95 > 1)) {
  stop("Invalid coverage values")
}

metric_names <- c(
  "mse",
  "bias",
  "mean_abs_error",
  "coverage_95",
  "zero_exclusion_rate",
  "mean_interval_width",
  "mean_magnitude_attenuation",
  "toward_zero_miss_rate",
  "toward_zero_share_of_misses"
)

summary_rows <- list()
summary_index <- 0L
groups <- split(
  seq_len(nrow(seed_metrics)),
  interaction(seed_metrics$n0, seed_metrics$n1, seed_metrics$group, drop = TRUE)
)
for (indices in groups) {
  cell <- seed_metrics[indices, , drop = FALSE]
  for (metric_name in metric_names) {
    values <- cell[[metric_name]]
    finite_values <- values[is.finite(values)]
    summary_index <- summary_index + 1L
    summary_rows[[summary_index]] <- data.frame(
      n0 = cell$n0[1L],
      n1 = cell$n1[1L],
      group = cell$group[1L],
      metric = metric_name,
      replicates = length(finite_values),
      mean = if (length(finite_values)) mean(finite_values) else NA_real_,
      sd = if (length(finite_values) > 1L) stats::sd(finite_values) else NA_real_,
      mcse = if (length(finite_values) > 1L) {
        stats::sd(finite_values) / sqrt(length(finite_values))
      } else {
        NA_real_
      },
      min = if (length(finite_values)) min(finite_values) else NA_real_,
      max = if (length(finite_values)) max(finite_values) else NA_real_,
      stringsAsFactors = FALSE
    )
  }
}
cell_summary <- do.call(rbind, summary_rows)
cell_summary <- cell_summary[order(
  cell_summary$n0,
  cell_summary$n1,
  cell_summary$group,
  cell_summary$metric
), ]

manifest <- data.frame(
  path = normalizePath(result_paths, winslash = "/", mustWork = TRUE),
  bytes = file.info(result_paths)$size,
  sha256 = vapply(
    result_paths,
    function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE),
    character(1)
  ),
  stringsAsFactors = FALSE
)

utils::write.csv(seed_metrics, file.path(output_dir, "seed_metrics.csv"), row.names = FALSE)
utils::write.csv(seed_bins, file.path(output_dir, "seed_binned_metrics.csv"), row.names = FALSE)
utils::write.csv(cell_summary, file.path(output_dir, "cell_summary.csv"), row.names = FALSE)
utils::write.csv(manifest, file.path(output_dir, "input_manifest.csv"), row.names = FALSE)

metadata <- c(
  paste0("input_dir=", input_dir),
  paste0("result_count=", length(result_paths)),
  paste0("summary_time_utc=", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  paste0("r_version=", R.version.string),
  "batts_remote_sha=6f625bad83702b36e5480be1ed1343258a9b075a",
  "interval=pointwise posterior quantiles 0.025 and 0.975",
  "pilot_scope=20D matched global shift; seeds 1-10; balanced and unbalanced"
)
writeLines(metadata, file.path(output_dir, "summary_metadata.txt"), useBytes = TRUE)

cat("Summary complete: ", output_dir, "\n", sep = "")
