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
if (!requireNamespace("digest", quietly = TRUE)) {
  stop("The digest package is required")
}
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE, no.. = TRUE)) > 0L) {
  stop("Refusing to overwrite a non-empty output directory: ", output_dir)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

paths <- sort(list.files(
  input_dir,
  pattern = "^global_shift_n0_09000_n1_01000_seed_[0-9]+\\.rds$",
  full.names = TRUE
))
if (length(paths) != 10L) {
  stop("Expected 10 unbalanced result files; found ", length(paths))
}
results <- lapply(paths, readRDS)
seeds <- vapply(results, function(x) x$task$seed, integer(1))
if (!identical(sort(seeds), 1:10)) {
  stop("Expected seeds 1--10 exactly")
}
if (any(vapply(results, function(x) is.null(x$calibration_curve), logical(1)))) {
  stop("At least one result lacks calibration_curve")
}
if (any(vapply(
  results,
  function(x) !identical(
    x$provenance$batts_remote_sha,
    "6f625bad83702b36e5480be1ed1343258a9b075a"
  ),
  logical(1)
))) {
  stop("BATTS provenance mismatch")
}
if (any(vapply(results, function(x) length(x$provenance$warnings) > 0L, logical(1)))) {
  stop("At least one result contains a warning")
}

curves <- do.call(rbind, lapply(results, function(result) {
  cbind(seed = result$task$seed, result$calibration_curve)
}))
if (any(!is.finite(curves$empirical_coverage)) ||
    any(curves$empirical_coverage < 0 | curves$empirical_coverage > 1)) {
  stop("Invalid empirical coverage values")
}

expected_levels <- seq(0.01, 0.99, by = 0.01)
for (seed in 1:10) {
  for (group_name in c("all", "group0", "group1")) {
    observed <- curves$nominal_level[
      curves$seed == seed & curves$group == group_name
    ]
    if (!isTRUE(all.equal(observed, expected_levels, tolerance = 1e-15))) {
      stop("Unexpected nominal levels for seed ", seed, ", group ", group_name)
    }
  }
}

summary_rows <- list()
summary_index <- 0L
for (group_name in c("all", "group0", "group1")) {
  for (level in expected_levels) {
    values <- curves$empirical_coverage[
      curves$group == group_name & abs(curves$nominal_level - level) < 1e-12
    ]
    summary_index <- summary_index + 1L
    summary_rows[[summary_index]] <- data.frame(
      group = group_name,
      nominal_level = level,
      replicates = length(values),
      mean_empirical_coverage = mean(values),
      sd_across_replicates = stats::sd(values),
      mcse = stats::sd(values) / sqrt(length(values)),
      min_empirical_coverage = min(values),
      max_empirical_coverage = max(values),
      stringsAsFactors = FALSE
    )
  }
}
summary_table <- do.call(rbind, summary_rows)

for (i in seq_along(results)) {
  result <- results[[i]]
  curve_95 <- result$calibration_curve[
    result$calibration_curve$group == "all" &
      abs(result$calibration_curve$nominal_level - 0.95) < 1e-12,
    "empirical_coverage"
  ]
  stored_95 <- result$metrics$coverage_95[result$metrics$group == "all"]
  if (length(curve_95) != 1L || !identical(curve_95, stored_95)) {
    stop("95% calibration point mismatch for seed ", result$task$seed)
  }
}

manifest <- data.frame(
  path = normalizePath(paths, winslash = "/", mustWork = TRUE),
  bytes = file.info(paths)$size,
  sha256 = vapply(
    paths,
    function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE),
    character(1)
  ),
  stringsAsFactors = FALSE
)

curve_path <- file.path(output_dir, "unbalanced_calibration_seed_curves.csv")
summary_path <- file.path(output_dir, "unbalanced_calibration_summary.csv")
manifest_path <- file.path(output_dir, "unbalanced_calibration_input_manifest.csv")
plot_path <- file.path(output_dir, "unbalanced_calibration_curve.png")
metadata_path <- file.path(output_dir, "unbalanced_calibration_metadata.txt")
utils::write.csv(curves, curve_path, row.names = FALSE)
utils::write.csv(summary_table, summary_path, row.names = FALSE)
utils::write.csv(manifest, manifest_path, row.names = FALSE)

grDevices::png(plot_path, width = 3000, height = 1100, res = 300, bg = "white")
old_par <- graphics::par(no.readonly = TRUE)
on.exit({
  graphics::par(old_par)
  grDevices::dev.off()
}, add = TRUE)
graphics::par(
  mfrow = c(1, 3),
  mar = c(4.4, 4.6, 3.3, 1.0),
  oma = c(1.5, 1.2, 2.5, 0.5),
  mgp = c(2.6, 0.8, 0),
  tcl = -0.25
)
group_labels <- c(
  all = "All observations",
  group0 = "Group 0 (n = 9000)",
  group1 = "Group 1 (n = 1000)"
)
for (group_name in names(group_labels)) {
  graphics::plot(
    expected_levels,
    rep(NA_real_, length(expected_levels)),
    type = "n",
    xlim = c(0, 1),
    ylim = c(0, 1),
    xaxs = "i",
    yaxs = "i",
    xlab = "Posterior interval mass",
    ylab = "Empirical coverage rate",
    main = group_labels[[group_name]],
    cex.main = 1.05
  )
  graphics::abline(a = 0, b = 1, lty = 2, lwd = 1.2, col = "grey30")
  for (seed in 1:10) {
    seed_curve <- curves[curves$group == group_name & curves$seed == seed, ]
    graphics::lines(
      seed_curve$nominal_level,
      seed_curve$empirical_coverage,
      col = grDevices::adjustcolor("grey60", alpha.f = 0.55),
      lwd = 0.8
    )
  }
  mean_curve <- summary_table[summary_table$group == group_name, ]
  graphics::lines(
    mean_curve$nominal_level,
    mean_curve$mean_empirical_coverage,
    col = "black",
    lwd = 2.5
  )
}
graphics::mtext(
  "20D matched global shift, unbalanced sample sizes",
  side = 3,
  outer = TRUE,
  line = 0.8,
  cex = 1.15
)
graphics::par(old_par)
grDevices::dev.off()
on.exit(NULL, add = FALSE)

metadata <- c(
  paste0("input_dir=", input_dir),
  "scenario=20D matched global shift",
  "sample_sizes=n0=9000,n1=1000",
  "seeds=1:10",
  "nominal_levels=0.01:0.99 by 0.01",
  "intervals=central pointwise posterior intervals, R quantile type 7",
  "aggregation=each seed curve uses all observed points; black curve is seed mean",
  "batts_remote_sha=6f625bad83702b36e5480be1ed1343258a9b075a",
  paste0("created_utc=", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  paste0("r_version=", R.version.string)
)
writeLines(metadata, metadata_path, useBytes = TRUE)

cat("Calibration outputs: ", output_dir, "\n", sep = "")
