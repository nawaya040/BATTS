#!/usr/bin/env Rscript

# Build the 20D pointwise-localization figure from saved 95% credible
# intervals. No estimator is fitted and no random numbers are consumed.

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(file_arg)) stop("This script must be run with Rscript")
script_path <- normalizePath(
  sub("^--file=", "", file_arg[[1L]]), winslash = "/", mustWork = TRUE
)
repo_root <- normalizePath(
  file.path(dirname(script_path), "..", ".."), winslash = "/", mustWork = TRUE
)

parse_args <- function(x) {
  out <- list()
  for (arg in x) {
    if (!grepl("^--[^=]+=", arg)) stop("Arguments must use --name=value: ", arg)
    pair <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    out[[pair[[1L]]]] <- paste(pair[-1L], collapse = "=")
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
legacy_root_arg <- args[["legacy-root"]]
global_root_arg <- args[["global-root"]]
if (is.null(legacy_root_arg) || !nzchar(legacy_root_arg) ||
    is.null(global_root_arg) || !nzchar(global_root_arg)) {
  stop(paste(
    "Required arguments:",
    "--legacy-root=<saved location/dispersion results>",
    "--global-root=<saved canonical global-shift results>"
  ))
}
legacy_root <- normalizePath(legacy_root_arg, winslash = "/", mustWork = TRUE)
global_root <- normalizePath(global_root_arg, winslash = "/", mustWork = TRUE)
output_png <- file.path(
  repo_root, "output", "figures", "figure_20d_localization_r2_preview.png"
)
output_pdf <- file.path(
  repo_root, "output", "pdf", "figure_20d_localization_r2_preview.pdf"
)
repeat_csv <- file.path(
  repo_root, "output", "figures",
  "figure_20d_localization_r2_preview_by_repeat.csv"
)
summary_csv <- file.path(
  repo_root, "output", "figures",
  "figure_20d_localization_r2_preview_summary.csv"
)
manifest_csv <- file.path(
  repo_root, "output", "figures",
  "figure_20d_localization_r2_preview_input_manifest.csv"
)
metadata_path <- file.path(
  repo_root, "output", "figures",
  "figure_20d_localization_r2_preview_metadata.txt"
)
output_paths <- c(
  output_png, output_pdf, repeat_csv, summary_csv, manifest_csv, metadata_path
)
if (any(file.exists(output_paths))) {
  stop("Refusing to overwrite an existing 20D localization preview output")
}

repeat_ids <- seq.int(1L, 46L, by = 5L)
settings <- list(
  list(id = "global_balanced", scenario = "global_shift",
       label = "Global Shift", balance = "balanced", n0 = 5000L, n1 = 5000L,
       source = "global"),
  list(id = "location_balanced", scenario = "latent_location_shift",
       label = "Location Shift", balance = "balanced", n0 = 5000L, n1 = 5000L,
       source = "legacy"),
  list(id = "dispersion_balanced", scenario = "latent_dispersion",
       label = "Dispersion", balance = "balanced", n0 = 5000L, n1 = 5000L,
       source = "legacy"),
  list(id = "global_unbalanced", scenario = "global_shift",
       label = "Global Shift", balance = "unbalanced", n0 = 9000L, n1 = 1000L,
       source = "global"),
  list(id = "location_unbalanced", scenario = "latent_location_shift",
       label = "Location Shift", balance = "unbalanced", n0 = 9000L, n1 = 1000L,
       source = "legacy"),
  list(id = "dispersion_unbalanced", scenario = "latent_dispersion",
       label = "Dispersion", balance = "unbalanced", n0 = 9000L, n1 = 1000L,
       source = "legacy")
)

bin_plot_values <- seq(0, 2.5, by = 0.25)
bin_breaks <- c(
  0,
  (head(bin_plot_values, -1L) + tail(bin_plot_values, -1L)) / 2,
  Inf
)
bin_labels <- c(
  sprintf("[%.3f,%.3f)", head(bin_breaks, -2L), bin_breaks[2:(length(bin_breaks) - 1L)]),
  sprintf("[%.3f,Inf)", bin_breaks[[length(bin_breaks) - 1L]])
)

validate_vectors <- function(truth, lower, upper, group, setting, path) {
  n <- setting$n0 + setting$n1
  if (!all(vapply(list(truth, lower, upper), is.numeric, logical(1))) ||
      any(vapply(list(truth, lower, upper), length, integer(1)) != n) ||
      any(!is.finite(c(truth, lower, upper))) || any(lower > upper)) {
    stop("Invalid pointwise values in ", path)
  }
  expected_group <- c(rep(0L, setting$n0), rep(1L, setting$n1))
  if (!identical(as.integer(group), expected_group)) {
    stop("Unexpected group order in ", path)
  }
}

read_pointwise <- function(setting, seed) {
  if (setting$source == "global") {
    path <- file.path(
      global_root,
      sprintf(
        "global_shift_n0_%05d_n1_%05d_seed_%02d.rds",
        setting$n0, setting$n1, seed
      )
    )
    if (!file.exists(path)) stop("Missing input: ", path)
    object <- readRDS(path)
    if (!identical(object$scenario, "global_shift") ||
        object$task$n0 != setting$n0 || object$task$n1 != setting$n1 ||
        object$task$seed != seed || is.null(object$pointwise)) {
      stop("Global result identity mismatch: ", path)
    }
    pointwise <- object$pointwise
    truth <- pointwise$true_log_ratio
    lower <- pointwise$lower_95
    upper <- pointwise$upper_95
    group <- pointwise$group_labels
    rm(object, pointwise)
  } else {
    stem <- paste(
      setting$scenario, setting$n0, setting$n1, 5L, seed, sep = "_"
    )
    scenario_dir <- file.path(legacy_root, setting$scenario)
    details_path <- file.path(scenario_dir, paste0(stem, "_details.rds"))
    truth_path <- file.path(scenario_dir, paste0(stem, "_true_log_ratio.rds"))
    if (!file.exists(details_path) || !file.exists(truth_path)) {
      stop("Missing legacy inputs for ", setting$id, " repeat ", seed)
    }
    details <- readRDS(details_path)
    truth <- readRDS(truth_path)
    quantiles <- details$log_ratio_BART_data_quantiles
    if (!is.matrix(quantiles) ||
        !identical(dim(quantiles), c(7L, setting$n0 + setting$n1))) {
      stop("Unexpected quantile matrix: ", details_path)
    }
    lower <- quantiles[1L, ]
    upper <- quantiles[7L, ]
    group <- c(rep(0L, setting$n0), rep(1L, setting$n1))
    path <- details_path
    rm(details, quantiles)
    return(list(
      truth = truth, lower = lower, upper = upper, group = group,
      paths = c(details_path, truth_path)
    ))
  }
  list(truth = truth, lower = lower, upper = upper, group = group, paths = path)
}

weighted_rate <- function(indicator, weights, indices) {
  if (!length(indices)) return(NA_real_)
  sum(weights[indices] * indicator[indices]) / sum(weights[indices])
}

manifest_row <- function(path, setting, seed) {
  info <- file.info(path)
  data.frame(
    setting_id = setting$id,
    simulation_repeat = seed,
    file_kind = if (grepl("_details[.]rds$", path)) {
      "details"
    } else if (grepl("_true_log_ratio[.]rds$", path)) {
      "true_log_ratio"
    } else {
      "canonical_global_result"
    },
    path = normalizePath(path, winslash = "/", mustWork = TRUE),
    bytes = unname(info$size),
    sha256 = unname(tools::sha256sum(path)),
    stringsAsFactors = FALSE
  )
}

repeat_rows <- list()
manifest_rows <- list()
row_index <- 0L
manifest_index <- 0L
max_abs_truth <- setNames(rep(-Inf, length(settings)), vapply(settings, `[[`, "", "id"))

for (setting in settings) {
  n <- setting$n0 + setting$n1
  weights <- c(
    rep(0.5 / setting$n0, setting$n0),
    rep(0.5 / setting$n1, setting$n1)
  )
  for (seed in repeat_ids) {
    pointwise <- read_pointwise(setting, seed)
    validate_vectors(
      pointwise$truth, pointwise$lower, pointwise$upper, pointwise$group,
      setting, pointwise$paths[[1L]]
    )
    abs_truth <- abs(pointwise$truth)
    covered <- pointwise$lower < pointwise$truth &
      pointwise$truth < pointwise$upper
    correct <- (pointwise$truth > 0 & pointwise$lower > 0) |
      (pointwise$truth < 0 & pointwise$upper < 0)
    wrong <- (pointwise$truth > 0 & pointwise$upper < 0) |
      (pointwise$truth < 0 & pointwise$lower > 0)
    if (any(correct & wrong)) stop("Direction classifications overlap")
    bins <- cut(
      abs_truth, breaks = bin_breaks, labels = FALSE,
      right = FALSE, include.lowest = TRUE
    )
    if (anyNA(bins)) stop("Failed to assign an absolute-effect bin")
    max_abs_truth[[setting$id]] <- max(max_abs_truth[[setting$id]], abs_truth)

    for (bin_id in seq_along(bin_plot_values)) {
      indices <- which(bins == bin_id)
      row_index <- row_index + 1L
      repeat_rows[[row_index]] <- data.frame(
        setting_id = setting$id,
        scenario = setting$label,
        balance = setting$balance,
        n0 = setting$n0,
        n1 = setting$n1,
        simulation_repeat = seed,
        bin_id = bin_id,
        bin_label = bin_labels[[bin_id]],
        plot_value = bin_plot_values[[bin_id]],
        n_points = length(indices),
        coverage_rate = weighted_rate(covered, weights, indices),
        correct_detection_rate = weighted_rate(correct, weights, indices),
        wrong_direction_rate = weighted_rate(wrong, weights, indices),
        stringsAsFactors = FALSE
      )
    }
    for (path in pointwise$paths) {
      manifest_index <- manifest_index + 1L
      manifest_rows[[manifest_index]] <- manifest_row(path, setting, seed)
    }
    rm(pointwise, abs_truth, covered, correct, wrong, bins)
    invisible(gc(FALSE))
  }
}

repeat_data <- do.call(rbind, repeat_rows)
manifest <- do.call(rbind, manifest_rows)

summarize_metric <- function(values) {
  values <- values[is.finite(values)]
  if (!length(values)) {
    return(c(mean = NA_real_, q10 = NA_real_, q90 = NA_real_, n = 0))
  }
  c(
    mean = mean(values),
    q10 = unname(stats::quantile(values, 0.10, type = 7)),
    q90 = unname(stats::quantile(values, 0.90, type = 7)),
    n = length(values)
  )
}

split_rows <- split(
  repeat_data,
  interaction(repeat_data$setting_id, repeat_data$bin_id, drop = TRUE)
)
summary_rows <- lapply(split_rows, function(cell) {
  coverage <- summarize_metric(cell$coverage_rate)
  correct <- summarize_metric(cell$correct_detection_rate)
  wrong <- summarize_metric(cell$wrong_direction_rate)
  data.frame(
    cell[1L, c(
      "setting_id", "scenario", "balance", "n0", "n1", "bin_id",
      "bin_label", "plot_value"
    )],
    mean_coverage_rate = coverage[["mean"]],
    q10_coverage_rate = coverage[["q10"]],
    q90_coverage_rate = coverage[["q90"]],
    n_coverage_repeats = as.integer(coverage[["n"]]),
    mean_correct_detection_rate = correct[["mean"]],
    q10_correct_detection_rate = correct[["q10"]],
    q90_correct_detection_rate = correct[["q90"]],
    n_correct_repeats = as.integer(correct[["n"]]),
    mean_wrong_direction_rate = wrong[["mean"]],
    q10_wrong_direction_rate = wrong[["q10"]],
    q90_wrong_direction_rate = wrong[["q90"]],
    n_wrong_repeats = as.integer(wrong[["n"]]),
    stringsAsFactors = FALSE
  )
})
summary_data <- do.call(rbind, summary_rows)
setting_order <- vapply(settings, `[[`, "", "id")
summary_data <- summary_data[order(
  match(summary_data$setting_id, setting_order), summary_data$bin_id
), , drop = FALSE]
rownames(summary_data) <- NULL

dir.create(dirname(output_png), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_pdf), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(repeat_data, repeat_csv, row.names = FALSE)
utils::write.csv(summary_data, summary_csv, row.names = FALSE)
utils::write.csv(manifest, manifest_csv, row.names = FALSE)

draw_ribbon <- function(x, lower, upper, colour, alpha = 0.10) {
  valid <- is.finite(x) & is.finite(lower) & is.finite(upper)
  if (sum(valid) > 1L) {
    graphics::polygon(
      c(x[valid], rev(x[valid])), c(lower[valid], rev(upper[valid])),
      border = NA, col = grDevices::adjustcolor(colour, alpha.f = alpha)
    )
  }
}

colours <- c(correct = "#111111", coverage = "#2C7FB8", wrong = "#D7301F")
draw_figure <- function() {
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)
  graphics::layout(
    matrix(c(1:6, 7, 7, 7), nrow = 3L, byrow = TRUE),
    heights = c(1, 1, 0.14)
  )
  graphics::par(
    oma = c(5.7, 5.9, 0.8, 6.8),
    mar = c(2.7, 2.9, 3.7, 1.0),
    mgp = c(2.15, 0.68, 0), tcl = -0.25,
    family = "sans", fg = "#333333", xaxs = "i", yaxs = "i"
  )
  y_ticks <- seq(0, 1, by = 0.25)
  x_ticks <- seq(0, 2.5, by = 0.5)
  x_labels <- c("0.0", "0.5", "1.0", "1.5", "2.0", expression(">=" * 2.375))

  for (index in seq_along(settings)) {
    setting <- settings[[index]]
    row_number <- if (setting$balance == "balanced") 1L else 2L
    column_number <- match(setting$label, c("Global Shift", "Location Shift", "Dispersion"))
    panel <- summary_data[summary_data$setting_id == setting$id, , drop = FALSE]
    panel <- panel[order(panel$plot_value), , drop = FALSE]
    x <- panel$plot_value

    graphics::plot.new()
    graphics::plot.window(xlim = c(-0.025, 2.525), ylim = c(-0.055, 1.055))
    graphics::rect(-0.025, -0.055, 2.525, 1.055, col = "white", border = NA)
    graphics::abline(h = y_ticks, v = x_ticks, col = "#E3E3E3", lwd = 0.9)
    graphics::abline(h = 0, col = "#BDBDBD", lwd = 0.9)

    draw_ribbon(
      x, panel$q10_correct_detection_rate, panel$q90_correct_detection_rate,
      colours[["correct"]]
    )
    draw_ribbon(
      x, panel$q10_coverage_rate, panel$q90_coverage_rate,
      colours[["coverage"]], alpha = 0.20
    )
    graphics::lines(
      x, panel$mean_correct_detection_rate,
      col = colours[["correct"]], lwd = 2.7, lty = 1
    )
    graphics::points(
      x, panel$mean_correct_detection_rate,
      col = colours[["correct"]], pch = 16, cex = 1.05
    )
    graphics::lines(
      x, panel$mean_coverage_rate,
      col = colours[["coverage"]], lwd = 2.7, lty = 2
    )
    graphics::points(
      x, panel$mean_coverage_rate,
      col = colours[["coverage"]], pch = 16, cex = 1.05
    )
    graphics::lines(
      x, panel$mean_wrong_direction_rate,
      col = colours[["wrong"]], lwd = 2.4, lty = 3
    )
    graphics::points(
      x, panel$mean_wrong_direction_rate,
      col = colours[["wrong"]], pch = 16, cex = 1.05
    )
    graphics::box(col = "#4D4D4D", lwd = 1.1)

    if (row_number == 2L) {
      graphics::axis(1, at = x_ticks, labels = x_labels, cex.axis = 1.42)
    }
    if (column_number == 1L) {
      graphics::axis(
        2, at = y_ticks, labels = sprintf("%.2f", y_ticks),
        las = 1, cex.axis = 1.48
      )
    }
    if (row_number == 1L) {
      graphics::title(
        main = setting$label, cex.main = 1.45, font.main = 2, line = 1.25
      )
    }
    if (column_number == 3L) {
      row_heading <- if (row_number == 1L) "Balanced" else "Unbalanced"
      sample_heading <- if (row_number == 1L) {
        "n0 = n1 = 5000"
      } else {
        "n0 = 9000, n1 = 1000"
      }
      graphics::mtext(
        row_heading, side = 4, line = 2.2, las = 3,
        cex = 1.25, font = 2
      )
      graphics::mtext(
        sample_heading, side = 4, line = 3.45, las = 3, cex = 1.08
      )
    }
  }

  graphics::mtext(
    "Absolute true log-density ratio",
    side = 1, outer = TRUE, line = 2.8, cex = 1.40
  )
  graphics::mtext("Rate", side = 2, outer = TRUE, line = 3.2, cex = 1.45)
  graphics::par(mar = c(0, 0, 0, 0))
  graphics::plot.new()
  graphics::legend(
    "center",
    legend = c(
      "Correct-direction zero exclusion",
      "Coverage of true log ratio",
      "Wrong-direction zero exclusion"
    ),
    col = colours[c("correct", "coverage", "wrong")],
    lty = c(1, 2, 3), lwd = c(2.7, 2.7, 2.4),
    pch = 16, pt.cex = 1.05,
    horiz = TRUE, bty = "n", cex = 1.65,
    x.intersp = 0.75, seg.len = 2.5
  )
}

grDevices::png(
  output_png, width = 15.8, height = 9.0,
  units = "in", res = 300, bg = "white"
)
draw_figure()
grDevices::dev.off()
grDevices::cairo_pdf(output_pdf, width = 15.8, height = 9.0, bg = "white")
draw_figure()
grDevices::dev.off()

writeLines(c(
  "figure=20D pointwise localization R2 preview",
  "source=saved raw 95% credible intervals",
  "scenarios=global_shift,latent_location_shift,latent_dispersion",
  "sample_sizes=5000_5000,9000_1000",
  paste0("repeat_ids=", paste(repeat_ids, collapse = ",")),
  "repeats_per_cell=10",
  "evaluation_points=all observed sample points",
  "evaluation_weighting=equal mixture: 0.5 total weight per group",
  "coverage_rule=lower < truth < upper",
  "correct_detection=true ratio positive and lower > 0, or true ratio negative and upper < 0",
  "wrong_direction=true ratio positive and upper < 0, or true ratio negative and lower > 0",
  "bin_plot_values=0:2.5 by 0.25",
  "finite_bin_width=0.25 with boundaries halfway between plot values",
  "tail_bin=absolute true log ratio >= 2.375, plotted at 2.5",
  "uncertainty_summary=mean and empirical 0.10/0.90 quantiles across repeats",
  "maximum_simulated_log_ratio_line=removed",
  "x_axis_tick_cex=1.42",
  "y_axis_tick_cex=1.48",
  "legend_cex=1.65",
  "correct_detection_ribbon_alpha=0.10",
  "coverage_ribbon_alpha=0.20",
  "estimator_fitting=false",
  "rng_consumed=false",
  paste0(
    "max_abs_true_log_ratio_by_setting=",
    paste(names(max_abs_truth), sprintf("%.7f", max_abs_truth), sep = ":", collapse = ",")
  ),
  paste0("input_manifest_sha256=", unname(tools::sha256sum(manifest_csv))),
  paste0("repeat_csv_sha256=", unname(tools::sha256sum(repeat_csv))),
  paste0("summary_csv_sha256=", unname(tools::sha256sum(summary_csv))),
  paste0("script_sha256=", unname(tools::sha256sum(script_path))),
  paste0("png=", normalizePath(output_png, winslash = "/", mustWork = TRUE)),
  paste0("pdf=", normalizePath(output_pdf, winslash = "/", mustWork = TRUE)),
  paste0("R_version=", R.version.string)
), metadata_path, useBytes = TRUE)

cat("Saved 20D localization PNG: ", normalizePath(output_png), "\n", sep = "")
cat("Saved 20D localization PDF: ", normalizePath(output_pdf), "\n", sep = "")
cat("Saved repeat summary: ", normalizePath(repeat_csv), "\n", sep = "")
cat("Saved aggregate summary: ", normalizePath(summary_csv), "\n", sep = "")
cat("Saved input manifest: ", normalizePath(manifest_csv), "\n", sep = "")
cat("Saved metadata: ", normalizePath(metadata_path), "\n", sep = "")
