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
  stop("Refusing to overwrite a non-empty summary directory: ", output_dir)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

sidecar_paths <- sort(list.files(input_dir, pattern = "[.]done[.]rds$", full.names = TRUE))
if (length(sidecar_paths) != 200L) {
  stop("Expected 200 completed sidecars; found ", length(sidecar_paths))
}
sidecars <- lapply(sidecar_paths, readRDS)
keys <- vapply(
  sidecars,
  function(x) paste(x$scenario, x$n0, x$n1, x$seed, sep = "/"),
  character(1)
)
if (anyDuplicated(keys)) {
  stop("Duplicate canonical task keys")
}
if (any(vapply(sidecars, function(x) length(x$warnings) > 0L, logical(1)))) {
  stop("At least one canonical task has a warning")
}

metric_rows <- list()
curve_rows <- list()
manifest_rows <- list()
for (i in seq_along(sidecars)) {
  x <- sidecars[[i]]
  identity <- data.frame(
    scenario = x$scenario,
    n0 = x$n0,
    n1 = x$n1,
    seed = x$seed,
    elapsed_seconds = x$elapsed_seconds,
    stringsAsFactors = FALSE
  )
  metric_rows[[i]] <- cbind(identity[rep(1L, nrow(x$metrics)), ], x$metrics)
  curve_rows[[i]] <- cbind(identity[rep(1L, nrow(x$calibration_curve)), ],
                           x$calibration_curve)
  manifest_rows[[i]] <- data.frame(
    scenario = x$scenario,
    n0 = x$n0,
    n1 = x$n1,
    seed = x$seed,
    path = x$result_path,
    bytes = x$result_bytes,
    sha256 = x$result_sha256,
    stringsAsFactors = FALSE
  )
}
seed_metrics <- do.call(rbind, metric_rows)
seed_curves <- do.call(rbind, curve_rows)
manifest <- do.call(rbind, manifest_rows)

metric_names <- setdiff(
  names(seed_metrics),
  c("scenario", "n0", "n1", "seed", "elapsed_seconds", "group", "n")
)
summary_rows <- list()
summary_index <- 0L
cells <- split(
  seq_len(nrow(seed_metrics)),
  interaction(
    seed_metrics$scenario,
    seed_metrics$n0,
    seed_metrics$n1,
    seed_metrics$group,
    drop = TRUE
  )
)
for (indices in cells) {
  cell <- seed_metrics[indices, , drop = FALSE]
  for (metric_name in metric_names) {
    values <- cell[[metric_name]]
    finite_values <- values[is.finite(values)]
    summary_index <- summary_index + 1L
    summary_rows[[summary_index]] <- data.frame(
      scenario = cell$scenario[1L],
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

curve_summary <- stats::aggregate(
  empirical_coverage ~ scenario + n0 + n1 + group + nominal_level,
  data = seed_curves,
  FUN = mean
)
names(curve_summary)[names(curve_summary) == "empirical_coverage"] <-
  "mean_empirical_coverage"

utils::write.csv(seed_metrics, file.path(output_dir, "seed_metrics.csv"), row.names = FALSE)
utils::write.csv(cell_summary, file.path(output_dir, "cell_summary.csv"), row.names = FALSE)
utils::write.csv(seed_curves, file.path(output_dir, "seed_calibration_curves.csv"), row.names = FALSE)
utils::write.csv(curve_summary, file.path(output_dir, "calibration_curve_summary.csv"), row.names = FALSE)
utils::write.csv(manifest, file.path(output_dir, "output_manifest.csv"), row.names = FALSE)

plot_path <- file.path(output_dir, "global_null_calibration_curves.png")
grDevices::png(plot_path, width = 2600, height = 2200, res = 300, bg = "white")
old_par <- graphics::par(no.readonly = TRUE)
on.exit({
  graphics::par(old_par)
  grDevices::dev.off()
}, add = TRUE)
graphics::par(
  mfrow = c(2, 2),
  mar = c(4.2, 4.5, 3.2, 1.2),
  oma = c(2.0, 2.0, 3.0, 0.5),
  mgp = c(2.5, 0.8, 0),
  tcl = -0.25
)
for (scenario in c("global_shift", "null")) {
  for (sample_sizes in list(c(5000L, 5000L), c(9000L, 1000L))) {
    curves <- seed_curves[
      seed_curves$scenario == scenario &
        seed_curves$n0 == sample_sizes[1L] &
        seed_curves$n1 == sample_sizes[2L] &
        seed_curves$group == "all",
    ]
    graphics::plot(
      c(0, 1), c(0, 1), type = "n", xlim = c(0, 1), ylim = c(0, 1),
      xaxs = "i", yaxs = "i",
      xlab = "Posterior interval mass",
      ylab = "Empirical coverage rate",
      main = paste0(
        if (scenario == "global_shift") "Global shift" else "Null",
        " (", sample_sizes[1L], "/", sample_sizes[2L], ")"
      )
    )
    graphics::abline(a = 0, b = 1, lty = 2, lwd = 1.2, col = "grey30")
    for (seed in sort(unique(curves$seed))) {
      z <- curves[curves$seed == seed, ]
      graphics::lines(
        z$nominal_level,
        z$empirical_coverage,
        col = grDevices::adjustcolor("grey60", alpha.f = 0.35),
        lwd = 0.6
      )
    }
    z <- curve_summary[
      curve_summary$scenario == scenario &
        curve_summary$n0 == sample_sizes[1L] &
        curve_summary$n1 == sample_sizes[2L] &
        curve_summary$group == "all",
    ]
    graphics::lines(
      z$nominal_level,
      z$mean_empirical_coverage,
      col = "black",
      lwd = 2.5
    )
  }
}
graphics::mtext(
  "20D global-shift and null calibration, 50 replicates",
  side = 3,
  outer = TRUE,
  line = 1,
  cex = 1.15
)
graphics::par(old_par)
grDevices::dev.off()
on.exit(NULL, add = FALSE)

metadata <- c(
  paste0("input_dir=", input_dir),
  "task_count=200",
  "replicates_per_cell=50",
  "posterior_draws_per_observation=1000",
  "nominal_levels=0.01:0.99 by 0.01",
  "batts_remote_sha=6f625bad83702b36e5480be1ed1343258a9b075a",
  paste0("created_utc=", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  paste0("r_version=", R.version.string)
)
writeLines(metadata, file.path(output_dir, "summary_metadata.txt"), useBytes = TRUE)

cat("Canonical summary complete: ", output_dir, "\n", sep = "")
