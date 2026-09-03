#!/usr/bin/env Rscript

# Render the Supplementary Figure S1 revision preview from the completed
# fixed-100-tree AdaBoost and GB results. No estimator is fitted here.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) stop("This script must be run with Rscript")
script_path <- normalizePath(sub("^--file=", "", script_arg[[1L]]), winslash = "/")
repo_root <- normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/")

input_root <- paste0(
  "C:/Users/user/Dropbox/Rcpp_experiments/active_programs/",
  "balancePM_results/experiments_1D/boosting"
)
output_png <- file.path(repo_root, "output", "figures", "supplement_s1_r2_preview.png")
output_pdf <- file.path(repo_root, "output", "pdf", "supplement_s1_r2_preview.pdf")
metadata_path <- file.path(
  repo_root, "output", "figures", "supplement_s1_r2_preview_metadata.txt"
)
manifest_path <- file.path(
  repo_root, "output", "figures", "supplement_s1_r2_preview_inputs.csv"
)

for (package in c("ggplot2", "patchwork", "scales")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
}

settings <- list(
  c(n0 = 500L, n1 = 500L),
  c(n0 = 300L, n1 = 700L),
  c(n0 = 200L, n1 = 800L),
  c(n0 = 100L, n1 = 900L)
)
n_repeat <- 50L
grid <- seq(-2.5, 3.5, length.out = 100L)
display <- grid <= 3
grid_display <- grid[display]
truth <- stats::dnorm(grid, 0, 1, log = TRUE) -
  stats::dnorm(grid, 1, 1.5, log = TRUE)

input_rows <- list()
setting_data <- vector("list", length(settings))
for (setting_index in seq_along(settings)) {
  setting <- settings[[setting_index]]
  n0 <- unname(setting[["n0"]])
  n1 <- unname(setting[["n1"]])
  ada_grid <- matrix(NA_real_, nrow = length(grid), ncol = n_repeat)
  ada_probability <- matrix(NA_real_, nrow = length(grid), ncol = n_repeat)
  gb_grid <- matrix(NA_real_, nrow = length(grid), ncol = n_repeat)

  for (repeat_index in seq_len(n_repeat)) {
    ada_path <- file.path(
      input_root, sprintf("Adaboost_%d_%d_%d.rds", n0, n1, repeat_index)
    )
    gb_path <- file.path(
      input_root, sprintf("Proposed_%d_%d_%d.rds", n0, n1, repeat_index)
    )
    if (!file.exists(ada_path) || !file.exists(gb_path)) {
      stop("Missing saved S1 result for setting ", n0, "/", n1,
           ", repeat ", repeat_index)
    }
    ada <- readRDS(ada_path)
    gb <- readRDS(gb_path)
    if (!identical(as.numeric(ada$num_trees_selected), 100) ||
        length(ada$log_w_hat_ada_grid) != length(grid) ||
        !identical(dim(ada$prob_pred_grid), c(length(grid), 2L)) ||
        length(gb) != length(grid)) {
      stop("Unexpected saved-result structure for setting ", n0, "/", n1,
           ", repeat ", repeat_index)
    }
    ada_grid[, repeat_index] <- as.numeric(ada$log_w_hat_ada_grid)
    ada_probability[, repeat_index] <- as.numeric(ada$prob_pred_grid[, 1L])
    gb_grid[, repeat_index] <- as.numeric(gb)
    input_rows[[length(input_rows) + 1L]] <- data.frame(
      setting = sprintf("%d_%d", n0, n1), simulation_repeat = repeat_index,
      method = "AdaBoost", path = normalizePath(ada_path, winslash = "/")
    )
    input_rows[[length(input_rows) + 1L]] <- data.frame(
      setting = sprintf("%d_%d", n0, n1), simulation_repeat = repeat_index,
      method = "GB", path = normalizePath(gb_path, winslash = "/")
    )
  }
  if (any(!is.finite(ada_grid)) || any(!is.finite(ada_probability)) ||
      any(!is.finite(gb_grid)) || any(ada_probability <= 0) ||
      any(ada_probability >= 1)) {
    stop("Non-finite or invalid saved values for setting ", n0, "/", n1)
  }
  setting_data[[setting_index]] <- list(
    n0 = n0,
    n1 = n1,
    posterior_truth = truth + log(n0 / n1),
    posterior_ada = rowMeans(log(ada_probability / (1 - ada_probability))),
    density_truth = truth,
    density_ada = rowMeans(ada_grid),
    density_gb = rowMeans(gb_grid)
  )
}

input_manifest <- do.call(rbind, input_rows)
input_hashes <- tools::sha256sum(input_manifest$path)
input_manifest$sha256 <- tolower(unname(input_hashes))
input_manifest$bytes <- as.numeric(file.info(input_manifest$path)$size)
if (anyNA(input_manifest$sha256) || anyNA(input_manifest$bytes)) {
  stop("Failed to hash one or more S1 input files")
}

method_levels <- c("DRT (AdaBoost)", "Truth", "GB")
method_colours <- c("DRT (AdaBoost)" = "black", "Truth" = "red", "GB" = "blue")
method_linetypes <- c("DRT (AdaBoost)" = "solid", "Truth" = "dotted", "GB" = "dashed")
column_titles <- c(
  "Log posterior odds ratio",
  "Log-density ratio",
  "Estimation error"
)
pad_range <- function(values, fraction = 0.04) {
  limits <- range(values, finite = TRUE)
  limits + c(-1, 1) * diff(limits) * fraction
}
y_limits <- list(
  pad_range(unlist(lapply(setting_data, function(setting) c(
    setting$posterior_truth[display], setting$posterior_ada[display]
  )))),
  pad_range(unlist(lapply(setting_data, function(setting) c(
    setting$density_truth[display], setting$density_ada[display],
    setting$density_gb[display]
  )))),
  pad_range(unlist(lapply(setting_data, function(setting) c(
    (setting$density_ada - setting$density_truth)[display],
    (setting$density_gb - setting$density_truth)[display]
  ))))
)
x_limits <- pad_range(c(-2.5, 3), fraction = 0.04)

make_panel_data <- function(setting, column_index) {
  if (column_index == 1L) {
    result <- rbind(
      data.frame(x = grid, value = setting$posterior_ada, method = "DRT (AdaBoost)"),
      data.frame(x = grid, value = setting$posterior_truth, method = "Truth")
    )
  } else if (column_index == 2L) {
    result <- rbind(
      data.frame(x = grid, value = setting$density_ada, method = "DRT (AdaBoost)"),
      data.frame(x = grid, value = setting$density_truth, method = "Truth"),
      data.frame(x = grid, value = setting$density_gb, method = "GB")
    )
  } else {
    result <- rbind(
      data.frame(
        x = grid, value = setting$density_ada - setting$density_truth,
        method = "DRT (AdaBoost)"
      ),
      data.frame(
        x = grid, value = setting$density_gb - setting$density_truth,
        method = "GB"
      )
    )
  }
  result <- result[result$x <= 3, , drop = FALSE]
  missing_methods <- setdiff(method_levels, as.character(unique(result$method)))
  if (length(missing_methods)) {
    result <- rbind(
      result,
      data.frame(x = NA_real_, value = NA_real_, method = missing_methods)
    )
  }
  result$method <- factor(result$method, levels = method_levels)
  result
}

base_theme <- ggplot2::theme_gray(base_size = 15, base_family = "sans") +
  ggplot2::theme(
    panel.grid.major = ggplot2::element_line(colour = "#E3E3E3", linewidth = 0.5),
    panel.grid.minor = ggplot2::element_blank(),
    panel.border = ggplot2::element_rect(
      fill = NA, colour = "#4D4D4D", linewidth = 0.45
    ),
    panel.background = ggplot2::element_rect(fill = "white", colour = NA),
    plot.title = ggplot2::element_text(
      size = 15, face = "bold", hjust = 0.5, margin = ggplot2::margin(b = 4)
    ),
    plot.subtitle = ggplot2::element_text(
      size = 14, face = "bold", hjust = 0.5, margin = ggplot2::margin(b = 4)
    ),
    axis.title = ggplot2::element_text(size = 18),
    axis.text = ggplot2::element_text(size = 11, colour = "#333333"),
    legend.title = ggplot2::element_blank(),
    legend.text = ggplot2::element_text(size = 14),
    legend.key.width = grid::unit(1.5, "cm"),
    plot.margin = ggplot2::margin(6, 6, 6, 6)
  )

plots <- list()
plot_index <- 1L
for (setting_index in seq_along(setting_data)) {
  setting <- setting_data[[setting_index]]
  for (column_index in seq_len(3L)) {
    panel_data <- make_panel_data(setting, column_index)
    title <- if (setting_index == 1L) column_titles[[column_index]] else NULL
    subtitle <- if (column_index == 1L) {
      bquote(paste(
        "(", n[0], ", ", n[1], ") = (", .(setting$n0), ", ", .(setting$n1), ")"
      ))
    } else {
      " "
    }
    x_label <- if (setting_index == length(setting_data)) "x" else NULL
    panel <- ggplot2::ggplot(
      panel_data, ggplot2::aes(x = x, y = value, colour = method, linetype = method)
    ) +
      ggplot2::geom_hline(
        yintercept = 0, colour = "#777777", linetype = "dotted", linewidth = 0.55
      ) +
      ggplot2::geom_line(linewidth = 1.25, lineend = "round", na.rm = TRUE) +
      ggplot2::scale_colour_manual(values = method_colours, drop = FALSE) +
      ggplot2::scale_linetype_manual(values = method_linetypes, drop = FALSE) +
      ggplot2::coord_cartesian(
        xlim = x_limits, ylim = y_limits[[column_index]], expand = FALSE
      ) +
      ggplot2::labs(title = title, subtitle = subtitle, x = x_label, y = NULL) +
      base_theme
    if (setting_index < length(setting_data)) {
      panel <- panel + ggplot2::theme(axis.title.x = ggplot2::element_blank())
    }
    plots[[plot_index]] <- panel
    plot_index <- plot_index + 1L
  }
}

final_plot <- patchwork::wrap_plots(plots, ncol = 3L, guides = "collect") &
  ggplot2::theme(legend.position = "bottom")

dir.create(dirname(output_png), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_pdf), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(input_manifest, manifest_path, row.names = FALSE, na = "")
ggplot2::ggsave(
  output_png, final_plot, width = 13.2, height = 14, units = "in",
  dpi = 300, bg = "white"
)
ggplot2::ggsave(
  output_pdf, final_plot, width = 13.2, height = 14, units = "in",
  device = grDevices::cairo_pdf, bg = "white"
)

writeLines(c(
  "figure=Supplementary Figure S1 R2 preview",
  "input=completed fixed-100-tree AdaBoost and GB results",
  "estimator_fitting=false",
  "n_repeat=50",
  "settings=500_500,300_700,200_800,100_900",
  "ada_trees=100_fixed",
  "grid_full=-2.5_to_3.5_100_points",
  "display_x=-2.5_to_3.0_matching_submitted_figure",
  "axis_margin_fraction=0.04_per_side",
  "line_width=1.25",
  "panel_background=white",
  "major_grid_colour=#E3E3E3",
  "axis_title_size=18",
  paste0("input_file_count=", nrow(input_manifest)),
  paste0("script_sha256=", tolower(unname(tools::sha256sum(script_path)))),
  paste0("png=", normalizePath(output_png, winslash = "/")),
  paste0("pdf=", normalizePath(output_pdf, winslash = "/"))
), metadata_path)

cat("Saved S1 PNG: ", normalizePath(output_png), "\n", sep = "")
cat("Saved S1 PDF: ", normalizePath(output_pdf), "\n", sep = "")
cat("Saved input manifest: ", normalizePath(manifest_path), "\n", sep = "")
