#!/usr/bin/env Rscript

# Render the revision candidate for Figure 3 from completed, checksummed
# results. This script does not fit or tune any estimator.

parse_args <- function(args) {
  result <- list()
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) {
      stop("Arguments must have the form --name=value: ", arg)
    }
    pair <- strsplit(substring(arg, 3L), "=", fixed = TRUE)[[1L]]
    result[[pair[[1L]]]] <- paste(pair[-1L], collapse = "=")
  }
  result
}

script_path <- function() {
  hit <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(hit)) stop("This script must be run with Rscript")
  normalizePath(sub("^--file=", "", hit[[1L]]), winslash = "/", mustWork = TRUE)
}

normalize_input <- function(path, label) {
  if (!file.exists(path)) stop("Missing ", label, ": ", path)
  normalizePath(path, winslash = "/", mustWork = TRUE)
}

assert_finite_vector <- function(x, expected_length, label) {
  x <- as.numeric(x)
  if (length(x) != expected_length) {
    stop(label, " has length ", length(x), "; expected ", expected_length)
  }
  if (any(!is.finite(x))) stop(label, " contains non-finite values")
  x
}

pick_quantile <- function(x, probability) {
  if (is.null(dim(x)) || length(dim(x)) != 2L) {
    stop("BAT quantiles must be a two-dimensional matrix")
  }
  labels <- rownames(x)
  if (is.null(labels)) {
    stop("BAT quantiles lack probability row names")
  }
  probabilities <- suppressWarnings(as.numeric(sub("%", "", labels, fixed = TRUE)) / 100)
  if (anyNA(probabilities)) stop("BAT quantile row names are not probabilities")
  as.numeric(x[which.min(abs(probabilities - probability)), , drop = TRUE])
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
this_script <- script_path()
repo_root <- normalizePath(
  file.path(dirname(this_script), "..", ".."), winslash = "/", mustWork = TRUE
)

for (key in c("legacy-detail", "boosting-result", "boosting-checksums")) {
  if (is.null(args[[key]])) stop("Required input argument: --", key)
}
legacy_path <- normalize_input(
  args[["legacy-detail"]],
  "legacy Figure 3 detail RDS"
)
boosting_path <- normalize_input(
  args[["boosting-result"]],
  "canonical boosting RDS"
)
checksums_path <- normalize_input(
  args[["boosting-checksums"]],
  "canonical boosting checksum table"
)
model_path <- normalize_input(
  file.path(repo_root, "scripts", "coverage", "models", "section41_2d_models.R"),
  "2D simulation model"
)

outdir <- if (is.null(args[["output-dir"]])) file.path(repo_root,"output/figures/figure3-compact-preview") else args[["output-dir"]]
if (dir.exists(outdir)) stop("Output directory already exists; choose a new candidate directory")
output_png <- file.path(outdir,"figure3_compact.png")
output_pdf <- file.path(outdir,"figure3_compact.pdf")
metadata_path <- file.path(outdir,"validation.txt")

if (!requireNamespace("ggplot2", quietly = TRUE) ||
    !requireNamespace("mvtnorm", quietly = TRUE)) {
  stop("Packages ggplot2 and mvtnorm are required")
}

expected_job <- paste0(
  "boosting_selection_2d_local_shift_n0-5000_n1-5000_",
  "transformed-false_seed-001"
)
checksum_table <- utils::read.csv(checksums_path, stringsAsFactors = FALSE)
required_columns <- c("job_id", "output_file", "output_sha256", "output_bytes")
if (!all(required_columns %in% names(checksum_table))) {
  stop("Boosting checksum table has an unexpected schema")
}
checksum_row <- checksum_table[checksum_table$job_id == expected_job, , drop = FALSE]
if (nrow(checksum_row) != 1L) stop("Expected one checksum row for ", expected_job)
actual_hash <- tolower(unname(tools::sha256sum(boosting_path)))
actual_bytes <- unname(file.info(boosting_path)$size)
if (!identical(actual_hash, tolower(checksum_row$output_sha256[[1L]])) ||
    !identical(as.numeric(actual_bytes), as.numeric(checksum_row$output_bytes[[1L]]))) {
  stop("Canonical boosting RDS does not match its checksum record")
}

expected_model_hash <- "ed19303c802057751b6a28e4ce63be8570bbe29da0da277c4a24c5f05f0f3790"
model_hash <- tolower(unname(tools::sha256sum(model_path)))
if (!identical(model_hash, expected_model_hash)) {
  stop("2D model source hash differs from the canonical source hash")
}

legacy <- readRDS(legacy_path)
boosting <- readRDS(boosting_path)
config <- boosting$metadata$config
if (!identical(boosting$metadata$status, "CANONICAL") ||
    !identical(boosting$metadata$job_id, expected_job) ||
    !identical(config$family, "2d") ||
    !identical(config$scenario, "local_shift") ||
    !identical(config$n0, 5000L) ||
    !identical(config$n1, 5000L) ||
    !identical(config$transformed, FALSE) ||
    !identical(config$seed_map$data, 1L)) {
  stop("Canonical boosting configuration does not match Figure 3")
}

source(model_path, local = TRUE)
set.seed(1L)
generated <- simulation_2d(5000L, 5000L, "local_shift", 100L)
coordinates <- as.matrix(generated$data)
if (!identical(dim(coordinates), c(10000L, 2L))) {
  stop("Regenerated Figure 3 coordinates have unexpected dimensions")
}

truth <- assert_finite_vector(boosting$result$design$truth_train, 10000L, "Truth")
generated_truth <- assert_finite_vector(generated$true_log_w_obs, 10000L, "Regenerated truth")
truth_difference <- max(abs(truth - generated_truth))
if (truth_difference > 1e-12) {
  stop("Regenerated data do not align with canonical truth; max difference = ", truth_difference)
}

drt <- assert_finite_vector(
  boosting$result$adaboost$estimates$train$exponential_loss,
  10000L, "DRT (AdaBoost), exponential-loss selection"
)
gb <- assert_finite_vector(
  boosting$result$proposed$estimates$gb$train, 10000L, "GB"
)
kliep <- assert_finite_vector(legacy$log_ratio_KLIEP_data, 10000L, "KLIEP")
ulsif <- assert_finite_vector(legacy$log_ratio_uLSIF_data, 10000L, "uLSIF")
bat_mean <- assert_finite_vector(legacy$log_ratio_BART_data_mean, 10000L, "BAT mean")
bat_lower <- assert_finite_vector(
  pick_quantile(legacy$log_ratio_BART_data_quantiles, 0.025),
  10000L, "BAT 2.5% quantile"
)
bat_upper <- assert_finite_vector(
  pick_quantile(legacy$log_ratio_BART_data_quantiles, 0.975),
  10000L, "BAT 97.5% quantile"
)
if (any(bat_lower > bat_upper)) stop("BAT lower quantile exceeds upper quantile")

continuous_values <- list(
  "Truth" = truth,
  "DRT (AdaBoost)" = drt,
  "KLIEP" = kliep,
  "uLSIF" = ulsif,
  "GB" = gb,
  "Bayesian additive trees" = bat_mean,
  "Lower (2.5%)" = bat_lower,
  "Upper (97.5%)" = bat_upper
)

# Preserve the scale construction used by visualize_result_2D.R. The legacy
# GB vector is used only to recover the original display scale; the plotted GB
# values come from the canonical revision output above.
legacy_gb_for_scale <- assert_finite_vector(
  legacy$log_ratio_boosting_grad_data, 10000L, "Legacy GB display reference"
)
minimum_true <- min(truth)
maximum_true <- max(truth)
minimum_estimate <- min(legacy_gb_for_scale)
maximum_estimate <- max(legacy_gb_for_scale)
threshold <- 0.55
minimum_display <- min((minimum_true + minimum_estimate) / 2, -threshold)
maximum_display <- max((maximum_true + maximum_estimate) / 2, threshold)
minimum_palette <- min(minimum_true, minimum_estimate, -threshold) - 1e-10
maximum_palette <- max(maximum_true, maximum_estimate, threshold) + 1e-10

palette_colours <- c(
  "darkblue", "blue", "blue", "deepskyblue", "cyan", "#7FFFFF",
  "white", "white", "white", "#FFFF99", "yellow", "orange", "red",
  "red", "darkred"
)
palette_locations <- scales::rescale(c(
  minimum_palette,
  minimum_display,
  minimum_display * 8 / 10,
  minimum_display * 6 / 10,
  minimum_display * 4 / 10,
  minimum_display * 2 / 10,
  minimum_display * 1 / 10,
  0,
  maximum_display * 1 / 10,
  maximum_display * 2 / 10,
  maximum_display * 4 / 10,
  maximum_display * 6 / 10,
  maximum_display * 8 / 10,
  maximum_display,
  maximum_palette
))

plot_data <- do.call(rbind, lapply(names(continuous_values), function(method) {
  data.frame(
    x1 = coordinates[, 1L],
    x2 = coordinates[, 2L],
    log_weight = continuous_values[[method]],
    method = method,
    stringsAsFactors = FALSE
  )
}))
plot_data$method <- factor(plot_data$method, levels = names(continuous_values))
plot_data <- plot_data[sample.int(nrow(plot_data)), , drop = FALSE]

pad_range <- function(values, fraction = 0.04) {
  limits <- range(values)
  padding <- diff(limits) * fraction
  limits + c(-padding, padding)
}

axis_limits <- list(
  x = pad_range(coordinates[, 1L]),
  y = pad_range(coordinates[, 2L])
)
base_theme <- ggplot2::theme_gray(base_size = 9, base_family = "sans") +
  ggplot2::theme(
    panel.grid.minor = ggplot2::element_blank(),
    panel.border = ggplot2::element_rect(
      fill = NA, colour = "#4D4D4D", linewidth = 0.25
    ),
    panel.background = ggplot2::element_rect(fill = "#EBEBEB", colour = NA),
    strip.background = ggplot2::element_rect(fill = "#D9D9D9", colour = "#4D4D4D", linewidth = 0.25),
    strip.text = ggplot2::element_text(size = 9, face = "bold", margin = ggplot2::margin(3, 2, 3, 2)),
    axis.title = ggplot2::element_text(size = 9),
    axis.text = ggplot2::element_text(size = 8, colour = "#333333"),
    legend.title = ggplot2::element_text(size = 9),
    legend.text = ggplot2::element_text(size = 8),
    plot.margin = ggplot2::margin(3, 3, 3, 3)
  )

final_plot <- ggplot2::ggplot(
  plot_data,
  ggplot2::aes(x = x1, y = x2, colour = log_weight)
) +
  ggplot2::geom_point(size = 0.7, alpha = 0.75, stroke = 0) +
  ggplot2::scale_colour_gradientn(
    colours = palette_colours,
    values = palette_locations,
    limits = c(minimum_display, maximum_display),
    oob = scales::squish,
    name = "log(density\nratio)",
    guide = ggplot2::guide_colourbar(
      barwidth = grid::unit(0.35, "cm"),
      barheight = grid::unit(2.6, "cm"),
      title.position = "top",
      title.hjust = 0.5
    )
  ) +
  ggplot2::facet_wrap(~method, nrow = 2L, ncol = 4L,
    labeller = ggplot2::as_labeller(c("Truth"="Truth", "DRT (AdaBoost)"="DRT\n(AdaBoost)",
      "KLIEP"="KLIEP", "uLSIF"="uLSIF", "GB"="GB",
      "Bayesian additive trees"="Bayesian\nadditive trees",
      "Lower (2.5%)"="Lower (2.5%)", "Upper (97.5%)"="Upper (97.5%)"))) +
  ggplot2::coord_equal(
    xlim = axis_limits$x, ylim = axis_limits$y, expand = FALSE, clip = "on"
  ) +
  ggplot2::labs(x = expression(x[1]), y = expression(x[2])) +
  base_theme +
  ggplot2::theme(legend.position = "right")

dir.create(dirname(output_png), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_pdf), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(metadata_path), recursive = TRUE, showWarnings = FALSE)

saveRDS(list(plot_data=plot_data, axis_limits=axis_limits, palette_colours=palette_colours, palette_locations=palette_locations, limits=c(minimum_display,maximum_display)), file.path(outdir,"plotted_data.rds"))
ggplot2::ggsave(
  output_png, final_plot, width = 13.5, height = 9.3, units = "cm",
  dpi = 300, bg = "white"
)
ggplot2::ggsave(
  output_pdf, final_plot, width = 13.5, height = 9.3, units = "cm",
  device = grDevices::cairo_pdf, bg = "white"
)

selection_row <- boosting$result$adaboost$selections[
  boosting$result$adaboost$selections$criterion == "exponential_loss", , drop = FALSE
]
metadata <- c(
  "figure=Figure 3 compact preview",
  "width_cm=13.5", "height_cm=9.3", "ticks_and_colorbar_pt=8", "headings_and_axis_titles_pt=9",
  "scenario=local_shift",
  "n0=5000",
  "n1=5000",
  "simulation_repeat=1",
  paste0("boosting_job_id=", expected_job),
  paste0("boosting_sha256=", actual_hash),
  paste0("legacy_detail_sha256=", tolower(unname(tools::sha256sum(legacy_path)))),
  paste0("model_sha256=", model_hash),
  paste0("drt_selection=exponential_loss"),
  paste0("drt_selected_trees=", selection_row$final_selected_trees[[1L]]),
  paste0("continuous_color_limits=", minimum_display, ",", maximum_display),
  "continuous_oob=scales::squish",
  "palette=legacy visualize_result_2D.R 15-colour gradient",
  "point_size=0.7",
  "point_alpha=0.75",
  "axis_margin_fraction=0.04_per_side",
  "panel_background=#EBEBEB",
  paste0("truth_alignment_max_abs_difference=", format(truth_difference, scientific = TRUE)),
  paste0("png=", normalizePath(output_png, winslash = "/", mustWork = TRUE)),
  paste0("pdf=", normalizePath(output_pdf, winslash = "/", mustWork = TRUE))
)
writeLines(metadata, metadata_path, useBytes = TRUE)

cat("Saved Figure 3 PNG: ", normalizePath(output_png), "\n", sep = "")
cat("Saved Figure 3 PDF: ", normalizePath(output_pdf), "\n", sep = "")
cat("Saved metadata: ", normalizePath(metadata_path), "\n", sep = "")
