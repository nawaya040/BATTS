#!/usr/bin/env Rscript

# Render the Figure 4 revision candidate from completed, checksummed results.
# This script reconstructs plotting coordinates but does not fit or tune models.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) stop("This script must be run with Rscript")
script_path <- normalizePath(sub("^--file=", "", script_arg[[1L]]), winslash = "/")
repo_root <- normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/")

legacy_path <- paste0(
  "C:/Users/user/Dropbox/Rcpp_experiments/active_programs/balancePM_results/",
  "experiments_multi/latent_location_shift/",
  "latent_location_shift_5000_5000_5_21_details.rds"
)
boosting_path <- paste0(
  "C:/Users/user/Dropbox/balancePM_experiment_backup_0806/",
  "boosting-selection-full-20260804/outputs/canonical/",
  "boosting_selection_20d_latent_location_shift_n0-5000_n1-5000_",
  "transformed-false_seed-021.rds"
)
checksums_path <- paste0(
  "C:/Users/user/Dropbox/Rcpp_experiments/active_programs/balancePM_backup/",
  "tmp/boosting-full-launcher-20260804/github-result-bundle/results/reference/",
  "boosting-selection-full-20260804/output_checksums.csv"
)
model_path <- file.path(repo_root, "scripts", "coverage", "models", "section42_multi_models.R")
output_png <- file.path(repo_root, "output", "figures", "figure4_r2_preview.png")
output_pdf <- file.path(repo_root, "output", "pdf", "figure4_r2_preview.pdf")
metadata_path <- file.path(repo_root, "output", "figures", "figure4_r2_preview_metadata.txt")

for (path in c(legacy_path, boosting_path, checksums_path, model_path)) {
  if (!file.exists(path)) stop("Missing required input: ", path)
}
for (package in c("ggplot2", "scales", "mvtnorm", "pracma")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
}

expected_job <- paste0(
  "boosting_selection_20d_latent_location_shift_n0-5000_n1-5000_",
  "transformed-false_seed-021"
)
checksums <- utils::read.csv(checksums_path, stringsAsFactors = FALSE)
row <- checksums[checksums$job_id == expected_job, , drop = FALSE]
if (nrow(row) != 1L) stop("Expected one checksum row for ", expected_job)
actual_hash <- tolower(unname(tools::sha256sum(boosting_path)))
actual_bytes <- unname(file.info(boosting_path)$size)
if (!identical(actual_hash, tolower(row$output_sha256[[1L]])) ||
    !identical(as.numeric(actual_bytes), as.numeric(row$output_bytes[[1L]]))) {
  stop("Canonical boosting RDS does not match its checksum record")
}
expected_model_hash <- "ad2090da5fbc01123e65687b1364a6064f2e5b3b17824ad3b9c4df7152e6eb40"
model_hash <- tolower(unname(tools::sha256sum(model_path)))
if (!identical(model_hash, expected_model_hash)) stop("20D model source hash mismatch")

legacy <- readRDS(legacy_path)
boosting <- readRDS(boosting_path)
config <- boosting$metadata$config
if (!identical(boosting$metadata$status, "CANONICAL") ||
    !identical(boosting$metadata$job_id, expected_job) ||
    !identical(config$family, "20d") ||
    !identical(config$scenario, "latent_location_shift") ||
    !identical(config$n0, 5000L) || !identical(config$n1, 5000L) ||
    !identical(config$transformed, FALSE) ||
    !identical(config$seed_map$data, 21L)) stop("Canonical Figure 4 configuration mismatch")

source(model_path, local = TRUE)
set.seed(21L)
generated <- simulation_multi_latent(5000L, 5000L, 20L, "latent_location_shift", 0.2, FALSE)
coordinates <- as.matrix(generated$data_u)
if (nrow(coordinates) != 10000L || ncol(coordinates) < 2L) {
  stop("Unexpected latent-coordinate dimensions")
}

finite_vector <- function(x, label) {
  x <- as.numeric(x)
  if (length(x) != 10000L || any(!is.finite(x))) stop("Invalid vector: ", label)
  x
}
pick_quantile <- function(x, probability) {
  probabilities <- as.numeric(sub("%", "", rownames(x), fixed = TRUE)) / 100
  as.numeric(x[which.min(abs(probabilities - probability)), ])
}
truth <- finite_vector(boosting$result$design$truth_train, "truth")
generated_truth <- finite_vector(generated$true_log_w_obs, "regenerated truth")
truth_difference <- max(abs(truth - generated_truth))
if (truth_difference > 1e-12) stop("Regenerated Figure 4 data do not align with canonical truth")

values <- list(
  "Truth" = truth,
  "DRT (AdaBoost)" = finite_vector(boosting$result$adaboost$estimates$train$exponential_loss, "DRT"),
  "KLIEP" = finite_vector(legacy$log_ratio_KLIEP_data, "KLIEP"),
  "uLSIF" = finite_vector(legacy$log_ratio_uLSIF_data, "uLSIF"),
  "GB" = finite_vector(boosting$result$proposed$estimates$gb$train, "GB"),
  "Bayesian additive trees" = finite_vector(legacy$log_ratio_BART_data_mean, "BAT mean"),
  "Lower (2.5%)" = finite_vector(pick_quantile(legacy$log_ratio_BART_data_quantiles, 0.025), "BAT lower"),
  "Upper (97.5%)" = finite_vector(pick_quantile(legacy$log_ratio_BART_data_quantiles, 0.975), "BAT upper")
)
if (any(values[[7L]] > values[[8L]])) stop("BAT lower quantile exceeds upper quantile")

legacy_gb <- finite_vector(legacy$log_ratio_boosting_grad_data, "legacy GB scale reference")
minimum_true <- min(truth); maximum_true <- max(truth)
minimum_estimate <- min(legacy_gb); maximum_estimate <- max(legacy_gb)
threshold <- 0.5
minimum_display <- min((minimum_true + minimum_estimate) / 2, -threshold)
maximum_display <- max((maximum_true + maximum_estimate) / 2, threshold)
minimum_palette <- min(minimum_true, minimum_estimate, -threshold) - 1e-10
maximum_palette <- max(maximum_true, maximum_estimate, threshold) + 1e-10
palette_colours <- c("darkblue", "blue", "blue", "deepskyblue", "cyan", "#7FFFFF",
  "white", "white", "white", "#FFFF99", "yellow", "orange", "red", "red", "darkred")
palette_locations <- scales::rescale(c(minimum_palette, minimum_display,
  minimum_display * 8/10, minimum_display * 6/10, minimum_display * 4/10,
  minimum_display * 2/10, minimum_display * 1/10, 0, maximum_display * 1/10,
  maximum_display * 2/10, maximum_display * 4/10, maximum_display * 6/10,
  maximum_display * 8/10, maximum_display, maximum_palette))

plot_data <- do.call(rbind, lapply(names(values), function(method) data.frame(
  z1 = coordinates[, 1L], z2 = coordinates[, 2L], log_weight = values[[method]], method = method
)))
plot_data$method <- factor(plot_data$method, levels = names(values))
plot_data <- plot_data[sample.int(nrow(plot_data)), , drop = FALSE]
pad <- function(x) { r <- range(x); r + c(-1, 1) * diff(r) * 0.04 }

final_plot <- ggplot2::ggplot(plot_data, ggplot2::aes(z1, z2, colour = log_weight)) +
  ggplot2::geom_point(size = 2, alpha = 0.75, stroke = 0) +
  ggplot2::scale_colour_gradientn(
    colours = palette_colours, values = palette_locations,
    limits = c(minimum_display, maximum_display), oob = scales::squish,
    name = "log(density ratio)", guide = ggplot2::guide_colourbar(
      barwidth = grid::unit(1.05, "cm"), barheight = grid::unit(6.4, "cm"),
      title.position = "top", title.hjust = 0.5)) +
  ggplot2::facet_wrap(~method, nrow = 2L, ncol = 4L) +
  ggplot2::coord_equal(xlim = pad(coordinates[,1L]), ylim = pad(coordinates[,2L]), expand = FALSE) +
  ggplot2::labs(x = expression(z[1]), y = expression(z[2])) +
  ggplot2::theme_gray(base_size = 15, base_family = "sans") +
  ggplot2::theme(
    panel.grid.minor = ggplot2::element_blank(),
    panel.border = ggplot2::element_rect(fill = NA, colour = "#4D4D4D", linewidth = 0.45),
    panel.background = ggplot2::element_rect(fill = "#EBEBEB", colour = NA),
    strip.background = ggplot2::element_rect(fill = "#D9D9D9", colour = "#4D4D4D", linewidth = 0.45),
    strip.text = ggplot2::element_text(size = 14, face = "bold", margin = ggplot2::margin(5,3,5,3)),
    axis.title = ggplot2::element_text(size = 18), axis.text = ggplot2::element_text(size = 11, colour = "#333333"),
    legend.title = ggplot2::element_text(size = 16), legend.text = ggplot2::element_text(size = 14),
    legend.position = "right", plot.margin = ggplot2::margin(7,7,7,7))

dir.create(dirname(output_png), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_pdf), recursive = TRUE, showWarnings = FALSE)
ggplot2::ggsave(output_png, final_plot, width = 13.2, height = 9.2, units = "in", dpi = 300, bg = "white")
ggplot2::ggsave(output_pdf, final_plot, width = 13.2, height = 9.2, units = "in", device = grDevices::cairo_pdf, bg = "white")
selection <- boosting$result$adaboost$selections
selection <- selection[selection$criterion == "exponential_loss", , drop = FALSE]
writeLines(c(
  "figure=Figure 4 R2 preview", "scenario=latent_location_shift", "n0=5000", "n1=5000",
  "simulation_repeat=21", paste0("boosting_job_id=", expected_job), paste0("boosting_sha256=", actual_hash),
  paste0("legacy_detail_sha256=", tolower(unname(tools::sha256sum(legacy_path)))),
  paste0("model_sha256=", model_hash), "drt_selection=exponential_loss",
  paste0("drt_selected_trees=", selection$final_selected_trees[[1L]]), "point_size=2", "point_alpha=0.75",
  "axis_title_size=18", "axis_margin_fraction=0.04_per_side", "panel_background=#EBEBEB",
  paste0("truth_alignment_max_abs_difference=", format(truth_difference, scientific = TRUE))
), metadata_path)
cat("Saved Figure 4 PNG: ", normalizePath(output_png), "\n", sep = "")
cat("Saved Figure 4 PDF: ", normalizePath(output_pdf), "\n", sep = "")
