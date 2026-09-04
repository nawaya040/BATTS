#!/usr/bin/env Rscript

# Render the Supplementary Figure S4 revision preview from the corrected,
# canonical 2D BART coverage summaries. This script only reads saved results;
# it does not fit an estimator or consume random numbers.

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
input_dir_arg <- args[["input-dir"]]
if (is.null(input_dir_arg) || !nzchar(input_dir_arg)) {
  stop("Required argument: --input-dir=<canonical coverage summary directory>")
}
input_dir <- normalizePath(input_dir_arg, winslash = "/", mustWork = TRUE)
seed_path <- file.path(input_dir, "coverage_by_seed.csv")
summary_path <- file.path(input_dir, "coverage_by_nominal_mass.csv")
metadata_input_path <- file.path(input_dir, "summary_metadata.txt")
for (path in c(seed_path, summary_path, metadata_input_path)) {
  if (!file.exists(path)) stop("Missing required input: ", path)
}

output_png <- file.path(
  repo_root, "output", "figures", "supplement_s4_r2_preview.png"
)
output_pdf <- file.path(
  repo_root, "output", "pdf", "supplement_s4_r2_preview.pdf"
)
metadata_path <- file.path(
  repo_root, "output", "figures", "supplement_s4_r2_preview_metadata.txt"
)
if (any(file.exists(c(output_png, output_pdf, metadata_path)))) {
  stop("Refusing to overwrite an existing S4 preview output")
}
seed_all <- utils::read.csv(seed_path, stringsAsFactors = FALSE)
summary_all <- utils::read.csv(summary_path, stringsAsFactors = FALSE)
seed_required <- c(
  "family", "scenario", "n0", "n1", "representation", "seed",
  "nominal_mass", "group", "coverage"
)
summary_required <- c(
  "family", "scenario", "n0", "n1", "representation", "group",
  "nominal_mass", "n_seeds", "mean_coverage"
)
if (!all(seed_required %in% names(seed_all)) ||
    !all(summary_required %in% names(summary_all))) {
  stop("Coverage summaries do not have the required columns")
}

keep_seed <- seed_all$family == "2d" &
  seed_all$representation == "not_applicable" &
  seed_all$group == "all_pooled"
keep_summary <- summary_all$family == "2d" &
  summary_all$representation == "not_applicable" &
  summary_all$group == "all_pooled"
seed_data <- seed_all[keep_seed, seed_required, drop = FALSE]
mean_data <- summary_all[keep_summary, summary_required, drop = FALSE]

scenario_levels <- c("global_shift", "local_shift", "local_dispersion")
expected_sizes <- data.frame(
  n0 = c(5000L, 9000L), n1 = c(5000L, 1000L),
  size_label = c(
    "Balanced (n0 = n1 = 5000)",
    "Unbalanced (n0 = 9000, n1 = 1000)"
  ),
  stringsAsFactors = FALSE
)
expected_nominal <- seq(0.01, 0.99, by = 0.01)
if (nrow(seed_data) != 3L * 2L * 50L * 99L ||
    nrow(mean_data) != 3L * 2L * 99L ||
    !setequal(unique(seed_data$scenario), scenario_levels) ||
    !identical(sort(unique(seed_data$seed)), 1:50) ||
    max(abs(sort(unique(seed_data$nominal_mass)) - expected_nominal)) > 1e-12 ||
    any(!is.finite(seed_data$coverage)) ||
    any(seed_data$coverage < 0 | seed_data$coverage > 1) ||
    any(mean_data$n_seeds != 50L)) {
  stop("Corrected 2D coverage input failed structural validation")
}

cell_sizes <- unique(seed_data[c("n0", "n1")])
if (nrow(merge(cell_sizes, expected_sizes[c("n0", "n1")])) != 2L ||
    nrow(cell_sizes) != 2L) {
  stop("Unexpected 2D sample-size settings")
}

# Independently reproduce every saved mean from the 50 seed-level curves.
recomputed <- stats::aggregate(
  coverage ~ scenario + n0 + n1 + nominal_mass,
  data = seed_data,
  FUN = mean
)
comparison <- merge(
  recomputed,
  mean_data[c("scenario", "n0", "n1", "nominal_mass", "mean_coverage")],
  by = c("scenario", "n0", "n1", "nominal_mass"),
  all = TRUE
)
mean_max_abs_difference <- max(abs(comparison$coverage - comparison$mean_coverage))
if (nrow(comparison) != 3L * 2L * 99L ||
    !is.finite(mean_max_abs_difference) || mean_max_abs_difference > 1e-12) {
  stop("Saved mean curves do not match the seed-level results")
}

scenario_labels <- c(
  global_shift = "Global Shift",
  local_shift = "Local Shift",
  local_dispersion = "Local Dispersion"
)
add_facet_labels <- function(x) {
  x$scenario_label <- factor(
    scenario_labels[x$scenario], levels = unname(scenario_labels)
  )
  size_key <- paste(x$n0, x$n1, sep = "_")
  x$size_label <- factor(
    c(
      "5000_5000" = expected_sizes$size_label[[1L]],
      "9000_1000" = expected_sizes$size_label[[2L]]
    )[size_key],
    levels = expected_sizes$size_label
  )
  if (anyNA(x$scenario_label) || anyNA(x$size_label)) {
    stop("Failed to construct facet labels")
  }
  x
}
seed_data <- add_facet_labels(seed_data)
mean_data <- add_facet_labels(mean_data)
seed_data$curve_id <- interaction(
  seed_data$scenario, seed_data$n0, seed_data$n1, seed_data$seed,
  drop = TRUE
)

dir.create(dirname(output_png), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_pdf), recursive = TRUE, showWarnings = FALSE)

draw_figure <- function() {
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)
  graphics::layout(
    matrix(c(1:6, 7, 7, 7), nrow = 3L, byrow = TRUE),
    heights = c(1, 1, 0.13)
  )
  graphics::par(
    oma = c(5.4, 5.8, 0.8, 6.6),
    mar = c(2.6, 2.8, 3.6, 1.1),
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
        cell_mean$nominal_mass, cell_mean$mean_coverage,
        col = "#111111", lwd = 2.7
      )
      graphics::box(col = "#4D4D4D", lwd = 1.1)
      if (row_index == 2L) {
        graphics::axis(1, at = ticks, labels = sprintf("%.2f", ticks), cex.axis = 1.25)
      }
      if (column_index == 1L) {
        graphics::axis(2, at = ticks, labels = sprintf("%.2f", ticks), las = 1, cex.axis = 1.25)
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
  output_png, width = 13.2, height = 8.8,
  units = "in", res = 300, bg = "white"
)
draw_figure()
grDevices::dev.off()
grDevices::cairo_pdf(output_pdf, width = 13.2, height = 8.8, bg = "white")
draw_figure()
grDevices::dev.off()

input_hashes <- tools::sha256sum(c(seed_path, summary_path, metadata_input_path))
writeLines(c(
  "figure=Supplementary Figure S4 R2 preview",
  "source=corrected canonical 2D BART coverage summary",
  "family=2d",
  "scenarios=global_shift,local_shift,local_dispersion",
  "sample_sizes=5000_5000,9000_1000",
  "seeds_per_cell=50",
  "nominal_levels=0.01:0.99 by 0.01",
  "coverage_group=all_pooled",
  "estimator_fitting=false",
  "rng_consumed=false",
  "repeat_colour=#C7C7C7",
  "repeat_legend_colour=#A6A6A6",
  "repeat_alpha=1",
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
  paste0("mean_recalculation_max_abs_difference=", format(mean_max_abs_difference, scientific = TRUE)),
  paste0("input_seed_sha256=", unname(input_hashes[[1L]])),
  paste0("input_summary_sha256=", unname(input_hashes[[2L]])),
  paste0("input_metadata_sha256=", unname(input_hashes[[3L]])),
  paste0("script_sha256=", unname(tools::sha256sum(script_path))),
  paste0("png=", normalizePath(output_png, winslash = "/", mustWork = TRUE)),
  paste0("pdf=", normalizePath(output_pdf, winslash = "/", mustWork = TRUE))
), metadata_path, useBytes = TRUE)

cat("Saved S4 PNG: ", normalizePath(output_png), "\n", sep = "")
cat("Saved S4 PDF: ", normalizePath(output_pdf), "\n", sep = "")
cat("Saved metadata: ", normalizePath(metadata_path), "\n", sep = "")
