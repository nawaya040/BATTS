#!/usr/bin/env Rscript

# Render the 20D BART calibration figure from saved raw
# summaries. This script only reads existing summaries; it does not fit an
# estimator or consume random numbers.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) stop("This script must be run with Rscript")
script_path <- normalizePath(
  sub("^--file=", "", script_arg[[1L]]), winslash = "/", mustWork = TRUE
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
global_null_dir_arg <- args[["global-null-dir"]]
coverage_dir_arg <- args[["coverage-dir"]]
if (is.null(global_null_dir_arg) || !nzchar(global_null_dir_arg) ||
    is.null(coverage_dir_arg) || !nzchar(coverage_dir_arg)) {
  stop(paste(
    "Required arguments:",
    "--global-null-dir=<canonical summary directory>",
    "--coverage-dir=<canonical coverage summary directory>"
  ))
}
global_null_dir <- normalizePath(
  global_null_dir_arg, winslash = "/", mustWork = TRUE
)
coverage_dir <- normalizePath(
  coverage_dir_arg, winslash = "/", mustWork = TRUE
)
global_null_path <- file.path(global_null_dir, "seed_calibration_curves.csv")
coverage_path <- file.path(coverage_dir, "coverage_by_seed.csv")
global_null_metadata_path <- file.path(global_null_dir, "summary_metadata.txt")
coverage_metadata_path <- file.path(coverage_dir, "summary_metadata.txt")
input_paths <- c(
  global_null_path, coverage_path,
  global_null_metadata_path, coverage_metadata_path
)
for (path in input_paths) {
  if (!file.exists(path)) stop("Missing required input: ", path)
}

output_png <- file.path(
  repo_root, "output", "figures", "figure_20d_calibration_r2_preview.png"
)
output_pdf <- file.path(
  repo_root, "output", "pdf", "figure_20d_calibration_r2_preview.pdf"
)
metadata_path <- file.path(
  repo_root, "output", "figures",
  "figure_20d_calibration_r2_preview_metadata.txt"
)
if (any(file.exists(c(output_png, output_pdf, metadata_path)))) {
  stop("Refusing to overwrite an existing 20D calibration preview output")
}

global_null_all <- utils::read.csv(
  global_null_path, stringsAsFactors = FALSE
)
coverage_all <- utils::read.csv(coverage_path, stringsAsFactors = FALSE)
global_null_required <- c(
  "scenario", "n0", "n1", "seed", "group", "nominal_level",
  "empirical_coverage"
)
coverage_required <- c(
  "family", "scenario", "n0", "n1", "representation", "seed",
  "nominal_mass", "group", "coverage"
)
if (!all(global_null_required %in% names(global_null_all)) ||
    !all(coverage_required %in% names(coverage_all))) {
  stop("Calibration summaries do not have the required columns")
}

global_null <- global_null_all[
  global_null_all$scenario %in% c("global_shift", "null") &
    global_null_all$group == "all",
  global_null_required,
  drop = FALSE
]
names(global_null)[names(global_null) == "nominal_level"] <- "nominal_mass"
names(global_null)[names(global_null) == "empirical_coverage"] <- "coverage"

location_dispersion <- coverage_all[
  coverage_all$family == "20d" &
    coverage_all$scenario %in% c(
      "latent_location_shift", "latent_dispersion"
    ) &
    coverage_all$representation == "raw" &
    coverage_all$group == "all_pooled",
  c("scenario", "n0", "n1", "seed", "nominal_mass", "coverage"),
  drop = FALSE
]

seed_data <- rbind(
  global_null[c(
    "scenario", "n0", "n1", "seed", "nominal_mass", "coverage"
  )],
  location_dispersion
)
# The two independently written CSVs can differ at machine precision even
# though both grids are 0.01, ..., 0.99. Restore their documented grid exactly.
seed_data$nominal_mass <- round(seed_data$nominal_mass, digits = 2L)
scenario_levels <- c(
  "global_shift", "latent_location_shift", "latent_dispersion", "null"
)
expected_sizes <- data.frame(
  n0 = c(5000L, 9000L),
  n1 = c(5000L, 1000L),
  stringsAsFactors = FALSE
)
expected_nominal <- seq(0.01, 0.99, by = 0.01)
expected_rows <- length(scenario_levels) * nrow(expected_sizes) * 50L * 99L
if (nrow(seed_data) != expected_rows ||
    !setequal(unique(seed_data$scenario), scenario_levels) ||
    !identical(sort(unique(seed_data$seed)), 1:50) ||
    max(abs(sort(unique(seed_data$nominal_mass)) - expected_nominal)) > 1e-12 ||
    any(!is.finite(seed_data$coverage)) ||
    any(seed_data$coverage < 0 | seed_data$coverage > 1)) {
  stop("Combined 20D calibration input failed structural validation")
}

cell_counts <- stats::aggregate(
  seed ~ scenario + n0 + n1 + nominal_mass,
  data = seed_data,
  FUN = function(x) length(unique(x))
)
if (nrow(cell_counts) != length(scenario_levels) * 2L * 99L ||
    any(cell_counts$seed != 50L)) {
  stop("Every panel and nominal level must contain exactly 50 seeds")
}
observed_sizes <- unique(seed_data[c("n0", "n1")])
if (nrow(observed_sizes) != 2L ||
    nrow(merge(observed_sizes, expected_sizes, by = c("n0", "n1"))) != 2L) {
  stop("Unexpected 20D sample-size settings")
}

mean_data <- stats::aggregate(
  coverage ~ scenario + n0 + n1 + nominal_mass,
  data = seed_data,
  FUN = mean
)
scenario_labels <- c(
  global_shift = "Global Shift",
  latent_location_shift = "Location Shift",
  latent_dispersion = "Dispersion",
  null = "Null"
)

dir.create(dirname(output_png), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_pdf), recursive = TRUE, showWarnings = FALSE)

draw_figure <- function() {
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)
  graphics::layout(
    matrix(c(1:8, rep(9L, 4L)), nrow = 3L, byrow = TRUE),
    heights = c(1, 1, 0.13)
  )
  graphics::par(
    oma = c(5.5, 5.9, 0.8, 6.7),
    mar = c(2.6, 2.8, 3.7, 1.0),
    mgp = c(2.1, 0.65, 0), tcl = -0.25,
    family = "sans", fg = "#333333", xaxs = "i", yaxs = "i"
  )
  ticks <- seq(0, 1, by = 0.25)
  repeat_colour <- "#C7C7C7"

  for (row_index in seq_len(2L)) {
    current_n0 <- expected_sizes$n0[[row_index]]
    current_n1 <- expected_sizes$n1[[row_index]]
    for (column_index in seq_along(scenario_levels)) {
      current_scenario <- scenario_levels[[column_index]]
      graphics::plot.new()
      graphics::plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1)
      graphics::rect(0, 0, 1, 1, col = "white", border = NA)
      graphics::abline(v = ticks, h = ticks, col = "#E3E3E3", lwd = 0.9)
      graphics::abline(a = 0, b = 1, col = "#555555", lty = 2, lwd = 1.8)

      cell_seed <- seed_data[
        seed_data$scenario == current_scenario &
          seed_data$n0 == current_n0 & seed_data$n1 == current_n1,
        , drop = FALSE
      ]
      for (seed in seq_len(50L)) {
        curve <- cell_seed[cell_seed$seed == seed, , drop = FALSE]
        curve <- curve[order(curve$nominal_mass), , drop = FALSE]
        graphics::lines(
          curve$nominal_mass, curve$coverage,
          col = repeat_colour, lwd = 1.15
        )
      }
      cell_mean <- mean_data[
        mean_data$scenario == current_scenario &
          mean_data$n0 == current_n0 & mean_data$n1 == current_n1,
        , drop = FALSE
      ]
      cell_mean <- cell_mean[order(cell_mean$nominal_mass), , drop = FALSE]
      graphics::lines(
        cell_mean$nominal_mass, cell_mean$coverage,
        col = "#111111", lwd = 2.7
      )
      graphics::box(col = "#4D4D4D", lwd = 1.1)
      if (row_index == 2L) {
        graphics::axis(
          1, at = ticks, labels = sprintf("%.2f", ticks), cex.axis = 1.25
        )
      }
      if (column_index == 1L) {
        graphics::axis(
          2, at = ticks, labels = sprintf("%.2f", ticks),
          las = 1, cex.axis = 1.25
        )
      }
      if (row_index == 1L) {
        graphics::title(
          main = unname(scenario_labels[[current_scenario]]),
          cex.main = 1.45, font.main = 2, line = 1.25
        )
      }
      if (column_index == length(scenario_levels)) {
        row_heading <- if (row_index == 1L) "Balanced" else "Unbalanced"
        sample_heading <- if (row_index == 1L) {
          "n0 = n1 = 5000"
        } else {
          "n0 = 9000, n1 = 1000"
        }
        graphics::mtext(
          row_heading, side = 4, line = 2.2, las = 3,
          cex = 1.25, font = 2
        )
        graphics::mtext(
          sample_heading, side = 4, line = 3.45, las = 3,
          cex = 1.08
        )
      }
    }
  }

  graphics::mtext(
    "Nominal credible level", side = 1, outer = TRUE,
    line = 2.6, cex = 1.45
  )
  graphics::mtext(
    "Empirical coverage rate", side = 2, outer = TRUE,
    line = 3.25, cex = 1.45
  )
  graphics::par(mar = c(0, 0, 0, 0))
  graphics::plot.new()
  graphics::legend(
    "center",
    legend = c("Average", "Repeats", "Nominal"),
    col = c("#111111", "#A6A6A6", "#555555"),
    lty = c(1, 1, 2), lwd = c(2.7, 2.0, 1.8),
    horiz = TRUE, bty = "n", cex = 1.50,
    x.intersp = 0.8, seg.len = 2.6
  )
}

grDevices::png(
  output_png, width = 16.8, height = 8.8,
  units = "in", res = 300, bg = "white"
)
draw_figure()
grDevices::dev.off()
grDevices::cairo_pdf(output_pdf, width = 16.8, height = 8.8, bg = "white")
draw_figure()
grDevices::dev.off()

input_hashes <- tools::sha256sum(input_paths)
writeLines(c(
  "figure=20D calibration R2 preview",
  "source=canonical saved raw BART calibration summaries",
  "family=20d",
  "representation=raw",
  "scenarios=global_shift,latent_location_shift,latent_dispersion,null",
  "sample_sizes=5000_5000,9000_1000",
  "seeds_per_cell=50",
  "nominal_levels=0.01:0.99 by 0.01",
  "coverage_group=all observations",
  "endpoint_rule_note=global/null use inclusive endpoints; location/dispersion use strict endpoints",
  "estimator_fitting=false",
  "rng_consumed=false",
  "repeat_colour=#C7C7C7",
  "repeat_legend_colour=#A6A6A6",
  "repeat_linewidth=1.15",
  "average_colour=#111111",
  "average_linewidth=2.7",
  "nominal_colour=#555555",
  "nominal_linewidth=1.8",
  "nominal_linetype=dashed",
  "panel_background=white",
  "scenario_heading_cex=1.45",
  "axis_tick_cex=1.25",
  "axis_title_cex=1.45",
  "row_heading_cex=1.25",
  "sample_size_cex=1.08",
  "legend_cex=1.50",
  paste0("input_global_null_sha256=", unname(input_hashes[[1L]])),
  paste0("input_location_dispersion_sha256=", unname(input_hashes[[2L]])),
  paste0("input_global_null_metadata_sha256=", unname(input_hashes[[3L]])),
  paste0("input_coverage_metadata_sha256=", unname(input_hashes[[4L]])),
  paste0("script_sha256=", unname(tools::sha256sum(script_path))),
  paste0("png=", normalizePath(output_png, winslash = "/", mustWork = TRUE)),
  paste0("pdf=", normalizePath(output_pdf, winslash = "/", mustWork = TRUE))
), metadata_path, useBytes = TRUE)

cat("Saved 20D calibration PNG: ", normalizePath(output_png), "\n", sep = "")
cat("Saved 20D calibration PDF: ", normalizePath(output_pdf), "\n", sep = "")
cat("Saved metadata: ", normalizePath(metadata_path), "\n", sep = "")
