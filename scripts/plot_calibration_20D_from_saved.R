file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(file_arg)) {
  stop("This script must be run with Rscript.")
}
script_path <- normalizePath(
  sub("^--file=", "", file_arg[[1]]),
  winslash = "/",
  mustWork = TRUE
)

get_arg_value <- function(args, name, default = NULL) {
  prefix <- paste0("--", name, "=")
  matched <- args[startsWith(args, prefix)]
  if (!length(matched)) {
    return(default)
  }
  sub(prefix, "", matched[[1]], fixed = TRUE)
}

require_arg <- function(args, name) {
  value <- get_arg_value(args, name)
  if (is.null(value) || !nzchar(value)) {
    stop("Missing required argument --", name, "=PATH")
  }
  value
}

normalize_existing_dir <- function(path, label) {
  if (!dir.exists(path)) {
    stop(label, " does not exist: ", path)
  }
  normalizePath(path, winslash = "/", mustWork = TRUE)
}

validate_summary <- function(object, expected_rows, expected_columns, path) {
  required_names <- c("coverage_rate_BAT", "is_included_mat")
  missing_names <- setdiff(required_names, names(object))
  if (length(missing_names)) {
    stop("Missing fields in ", path, ": ", paste(missing_names, collapse = ", "))
  }

  coverage <- object$coverage_rate_BAT
  included <- object$is_included_mat

  if (!is.numeric(coverage) || length(coverage) != expected_rows) {
    stop("Invalid coverage_rate_BAT in ", path)
  }
  if (!is.matrix(included) ||
      !identical(dim(included), c(expected_rows, expected_columns))) {
    stop(
      "Invalid is_included_mat dimensions in ", path,
      "; expected ", expected_rows, " x ", expected_columns
    )
  }
  if (any(!is.finite(coverage)) || any(!is.finite(included))) {
    stop("Non-finite coverage values in ", path)
  }
  if (any(coverage < 0 | coverage > 1) ||
      any(!included %in% c(0, 1))) {
    stop("Coverage values outside their valid range in ", path)
  }

  recomputed <- rowMeans(included)
  if (max(abs(coverage - recomputed)) > 1e-12) {
    stop("Stored coverage does not equal rowMeans(is_included_mat) in ", path)
  }

  list(coverage = coverage, included = included)
}

file_manifest_row <- function(path, setting_id, index_repeat, kind) {
  info <- file.info(path)
  data.frame(
    setting_id = setting_id,
    index_repeat = index_repeat,
    file_kind = kind,
    path = normalizePath(path, winslash = "/", mustWork = TRUE),
    bytes = unname(info$size),
    modified = format(info$mtime, "%Y-%m-%dT%H:%M:%S%z"),
    sha256 = digest::digest(file = path, algo = "sha256", serialize = FALSE),
    stringsAsFactors = FALSE
  )
}

plot_calibration <- function(
    output_path,
    curves,
    interval_mass,
    settings,
    overall_title) {
  ordered <- order(interval_mass)

  grDevices::png(
    output_path,
    width = 2400,
    height = 2200,
    res = 300,
    bg = "white"
  )
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

  for (index_setting in seq_along(settings)) {
    setting <- settings[[index_setting]]
    setting_curves <- curves[, ordered, index_setting, drop = FALSE]
    dim(setting_curves) <- c(dim(curves)[1], length(ordered))

    graphics::plot(
      interval_mass[ordered],
      rep(NA_real_, length(ordered)),
      type = "n",
      xlim = c(0, 1),
      ylim = c(0, 1),
      xaxs = "i",
      yaxs = "i",
      xlab = "Posterior interval mass",
      ylab = "Empirical coverage rate",
      main = paste0(setting$label, " (", setting$balance, ")"),
      cex.main = 1.05
    )
    graphics::abline(
      a = 0,
      b = 1,
      lty = 2,
      lwd = 1.2,
      col = "grey30"
    )

    for (index_repeat in seq_len(nrow(setting_curves))) {
      graphics::lines(
        interval_mass[ordered],
        setting_curves[index_repeat, ],
        col = grDevices::adjustcolor("grey60", alpha.f = 0.55),
        lwd = 0.8
      )
    }
    graphics::lines(
      interval_mass[ordered],
      colMeans(setting_curves),
      col = "black",
      lwd = 2.5
    )
  }

  graphics::mtext(overall_title, side = 3, outer = TRUE, line = 1.0, cex = 1.15)
}

args <- commandArgs(trailingOnly = TRUE)
results_root <- normalize_existing_dir(
  require_arg(args, "results-root"),
  "Results root"
)
baseline_arg <- get_arg_value(args, "baseline-root")
baseline_root <- if (is.null(baseline_arg) || !nzchar(baseline_arg)) {
  NULL
} else {
  normalize_existing_dir(baseline_arg, "Baseline root")
}
output_dir <- require_arg(args, "output-dir")
expected_repeats <- as.integer(get_arg_value(args, "expected-repeats", "50"))
lambda_0 <- as.integer(get_arg_value(args, "lambda", "5"))

if (is.na(expected_repeats) || expected_repeats < 1L) {
  stop("--expected-repeats must be a positive integer.")
}
if (is.na(lambda_0)) {
  stop("--lambda must be an integer.")
}
if (!requireNamespace("digest", quietly = TRUE)) {
  stop("The digest package is required to record SHA-256 hashes.")
}

settings <- list(
  list(
    id = "location_balanced",
    scenario = "latent_location_shift",
    label = "Location Shift",
    n0 = 5000L,
    n1 = 5000L,
    balance = "balanced"
  ),
  list(
    id = "location_unbalanced",
    scenario = "latent_location_shift",
    label = "Location Shift",
    n0 = 9000L,
    n1 = 1000L,
    balance = "unbalanced"
  ),
  list(
    id = "dispersion_balanced",
    scenario = "latent_dispersion",
    label = "Dispersion",
    n0 = 5000L,
    n1 = 5000L,
    balance = "balanced"
  ),
  list(
    id = "dispersion_unbalanced",
    scenario = "latent_dispersion",
    label = "Dispersion",
    n0 = 9000L,
    n1 = 1000L,
    balance = "unbalanced"
  )
)

quantile_probs <- seq(0.005, 0.995, by = 0.005)
n_probs <- length(quantile_probs)
lower_indices <- seq_len(floor(n_probs / 2))
lower_probs <- quantile_probs[lower_indices]
upper_probs_used <- quantile_probs[n_probs - lower_indices]
original_axis_mass <- 1 - 2 * lower_probs
actual_interval_mass <- upper_probs_used - lower_probs
n_curve_points <- length(actual_interval_mass)

if (max(abs((original_axis_mass - actual_interval_mass) - 0.005)) > 1e-12) {
  stop("Unexpected interval-mass correction.")
}

output_paths <- c(
  file.path(output_dir, "calib_20D_corrected.png"),
  file.path(output_dir, "calib_20D_s_corrected.png"),
  file.path(output_dir, "calib_20D_summary.csv"),
  file.path(output_dir, "calib_20D_input_manifest.csv"),
  file.path(output_dir, "calib_20D_run_metadata.txt")
)
existing_outputs <- output_paths[file.exists(output_paths)]
if (length(existing_outputs)) {
  stop(
    "Refusing to overwrite existing outputs: ",
    paste(existing_outputs, collapse = ", ")
  )
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, winslash = "/", mustWork = TRUE)

all_curves <- array(
  NA_real_,
  dim = c(expected_repeats, n_curve_points, length(settings))
)
subset_curves <- all_curves
subset_sizes <- matrix(
  NA_integer_,
  nrow = expected_repeats,
  ncol = length(settings)
)
manifest_rows <- list()
manifest_index <- 0L

for (index_setting in seq_along(settings)) {
  setting <- settings[[index_setting]]

  for (index_repeat in seq_len(expected_repeats)) {
    base_name <- sprintf(
      "%s_%d_%d_%d_%d",
      setting$scenario,
      setting$n0,
      setting$n1,
      lambda_0,
      index_repeat
    )
    scenario_dir <- file.path(results_root, setting$scenario)
    summary_path <- file.path(scenario_dir, paste0(base_name, ".rds"))
    true_ratio_path <- file.path(
      scenario_dir,
      paste0(base_name, "_true_log_ratio.rds")
    )

    if (!file.exists(summary_path)) {
      stop("Missing summary RDS: ", summary_path)
    }
    if (!file.exists(true_ratio_path)) {
      stop("Missing true-log-ratio RDS: ", true_ratio_path)
    }

    summary_object <- readRDS(summary_path)
    validated <- validate_summary(
      summary_object,
      expected_rows = n_curve_points,
      expected_columns = setting$n0 + setting$n1,
      path = summary_path
    )
    true_log_ratio <- readRDS(true_ratio_path)
    if (!is.numeric(true_log_ratio) ||
        length(true_log_ratio) != setting$n0 + setting$n1 ||
        any(!is.finite(true_log_ratio))) {
      stop("Invalid true-log-ratio vector in ", true_ratio_path)
    }

    threshold <- stats::quantile(
      abs(true_log_ratio),
      probs = 0.5,
      names = FALSE,
      type = 7
    )
    subset_indices <- which(abs(true_log_ratio) < threshold)
    if (!length(subset_indices)) {
      stop("Empty central subset in ", true_ratio_path)
    }

    all_curves[index_repeat, , index_setting] <- validated$coverage
    subset_curves[index_repeat, , index_setting] <- rowMeans(
      validated$included[, subset_indices, drop = FALSE]
    )
    subset_sizes[index_repeat, index_setting] <- length(subset_indices)

    manifest_index <- manifest_index + 1L
    manifest_rows[[manifest_index]] <- file_manifest_row(
      summary_path,
      setting$id,
      index_repeat,
      "summary"
    )
    manifest_index <- manifest_index + 1L
    manifest_rows[[manifest_index]] <- file_manifest_row(
      true_ratio_path,
      setting$id,
      index_repeat,
      "true_log_ratio"
    )

    if (!is.null(baseline_root)) {
      baseline_path <- file.path(
        baseline_root,
        setting$scenario,
        paste0(base_name, ".rds")
      )
      if (!file.exists(baseline_path)) {
        stop("Missing baseline summary RDS: ", baseline_path)
      }
      baseline_object <- readRDS(baseline_path)
      baseline_validated <- validate_summary(
        baseline_object,
        expected_rows = n_curve_points,
        expected_columns = setting$n0 + setting$n1,
        path = baseline_path
      )
      if (!identical(validated$coverage, baseline_validated$coverage) ||
          !identical(validated$included, baseline_validated$included)) {
        stop("Current result differs from baseline: ", summary_path)
      }

      manifest_index <- manifest_index + 1L
      manifest_rows[[manifest_index]] <- file_manifest_row(
        baseline_path,
        setting$id,
        index_repeat,
        "baseline_summary"
      )
    }
  }
}

if (any(!is.finite(all_curves)) || any(!is.finite(subset_curves))) {
  stop("Non-finite values remain after aggregation.")
}

summary_rows <- list()
summary_index <- 0L
for (index_setting in seq_along(settings)) {
  setting <- settings[[index_setting]]
  for (index_probability in seq_len(n_curve_points)) {
    all_values <- all_curves[, index_probability, index_setting]
    subset_values <- subset_curves[, index_probability, index_setting]
    summary_index <- summary_index + 1L
    summary_rows[[summary_index]] <- data.frame(
      setting_id = setting$id,
      scenario = setting$label,
      balance = setting$balance,
      n0 = setting$n0,
      n1 = setting$n1,
      original_axis_mass = original_axis_mass[index_probability],
      actual_interval_mass = actual_interval_mass[index_probability],
      mean_coverage_all = mean(all_values),
      median_coverage_all = stats::median(all_values),
      min_coverage_all = min(all_values),
      max_coverage_all = max(all_values),
      mean_coverage_subset = mean(subset_values),
      median_coverage_subset = stats::median(subset_values),
      min_coverage_subset = min(subset_values),
      max_coverage_subset = max(subset_values),
      mean_subset_size = mean(subset_sizes[, index_setting]),
      stringsAsFactors = FALSE
    )
  }
}

summary_table <- do.call(rbind, summary_rows)
manifest_table <- do.call(rbind, manifest_rows)

summary_path <- file.path(output_dir, "calib_20D_summary.csv")
manifest_path <- file.path(output_dir, "calib_20D_input_manifest.csv")
all_plot_path <- file.path(output_dir, "calib_20D_corrected.png")
subset_plot_path <- file.path(output_dir, "calib_20D_s_corrected.png")
metadata_path <- file.path(output_dir, "calib_20D_run_metadata.txt")

utils::write.csv(summary_table, summary_path, row.names = FALSE)
utils::write.csv(manifest_table, manifest_path, row.names = FALSE)

plot_calibration(
  all_plot_path,
  all_curves,
  actual_interval_mass,
  settings,
  "20D calibration: all observations"
)
plot_calibration(
  subset_plot_path,
  subset_curves,
  actual_interval_mass,
  settings,
  "20D calibration: central half by |true log-density ratio|"
)

metadata <- c(
  paste0("generated_at=", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
  paste0("script_path=", script_path),
  paste0(
    "script_sha256=",
    digest::digest(file = script_path, algo = "sha256", serialize = FALSE)
  ),
  paste0("results_root=", results_root),
  paste0(
    "baseline_root=",
    if (is.null(baseline_root)) "not supplied" else baseline_root
  ),
  paste0("output_dir=", output_dir),
  paste0("expected_repeats=", expected_repeats),
  paste0("lambda_0=", lambda_0),
  "axis_correction=actual interval mass is original displayed mass minus 0.005",
  "subset_rule=abs(true log-density ratio) below its within-repeat median",
  "baseline_check=coverage_rate_BAT and is_included_mat compared exactly",
  paste0("R_version=", R.version.string),
  "",
  "sessionInfo:",
  capture.output(sessionInfo())
)
writeLines(metadata, metadata_path, useBytes = TRUE)

cat("Calibration plots created.\n")
cat("All observations: ", normalizePath(all_plot_path), "\n", sep = "")
cat("Central subset: ", normalizePath(subset_plot_path), "\n", sep = "")
cat("Summary: ", normalizePath(summary_path), "\n", sep = "")
cat("Manifest: ", normalizePath(manifest_path), "\n", sep = "")
cat("Metadata: ", normalizePath(metadata_path), "\n", sep = "")
