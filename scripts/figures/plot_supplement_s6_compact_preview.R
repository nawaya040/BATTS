#!/usr/bin/env Rscript

# Display-only compact candidate for current Figure S6, using the verified
# fixed-seed simulation. No estimator is fitted or tuned here.

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_arg)) stop("This script must be run with Rscript")
script_path <- normalizePath(
  sub("^--file=", "", script_arg[[1L]]), winslash = "/", mustWork = TRUE
)
repo_root <- normalizePath(
  file.path(dirname(script_path), "..", ".."), winslash = "/", mustWork = TRUE
)

model_path <- file.path(
  repo_root, "scripts", "coverage", "models", "section42_multi_models.R"
)
args <- commandArgs(TRUE)
outdir <- if (length(args)) args[1] else file.path(repo_root, "output/figures/figureS6-compact-preview")
if (dir.exists(outdir)) stop("Output directory exists; choose a new candidate directory")
output_png <- file.path(outdir, "figureS6_compact.png")
output_pdf <- file.path(outdir, "figureS6_compact.pdf")
metadata_path <- file.path(outdir, "validation.txt")

for (package in c("ggplot2", "patchwork", "mvtnorm", "pracma")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
}
if (!file.exists(model_path)) stop("Missing 20D simulation model: ", model_path)
expected_model_hash <- "ad2090da5fbc01123e65687b1364a6064f2e5b3b17824ad3b9c4df7152e6eb40"
model_hash <- tolower(unname(tools::sha256sum(model_path)))
if (!identical(model_hash, expected_model_hash)) stop("20D model source hash mismatch")

model_environment <- new.env(parent = globalenv())
sys.source(model_path, envir = model_environment)
seed <- 2L
n0 <- 500L
n1 <- 500L
set.seed(seed)
generated <- model_environment$simulation_multi_latent(
  n0 = n0, n1 = n1, d = 20L, scenario = "latent_location_shift",
  unif_w = 0.2, transform = TRUE
)
data <- as.matrix(generated$data)
labels <- as.integer(generated$group_labels)
if (!identical(dim(data), c(1000L, 20L)) ||
    !identical(as.integer(table(labels)), c(n0, n1)) ||
    any(!is.finite(data)) || any(data < 0) || any(data > 1)) {
  stop("Generated S5 data failed structural validation")
}

# Alternate groups deterministically so neither colour is systematically
# drawn above the other throughout the dense region near the origin.
draw_order <- as.vector(rbind(which(labels == 0L), which(labels == 1L)))
group <- factor(
  ifelse(labels[draw_order] == 0L, "Group 0", "Group 1"),
  levels = c("Group 0", "Group 1")
)
make_pair_data <- function(first_dimension, second_dimension, panel_label) {
  data.frame(
    horizontal = data[draw_order, first_dimension],
    vertical = data[draw_order, second_dimension],
    group = group,
    panel = panel_label,
    stringsAsFactors = FALSE
  )
}

base_theme <- ggplot2::theme_gray(base_size = 9, base_family = "sans") +
  ggplot2::theme(
    panel.grid.major = ggplot2::element_line(colour = "#E3E3E3", linewidth = 0.2),
    panel.grid.minor = ggplot2::element_blank(),
    panel.border = ggplot2::element_rect(
      fill = NA, colour = "#4D4D4D", linewidth = 0.25
    ),
    panel.background = ggplot2::element_rect(fill = "white", colour = NA),
    strip.background = ggplot2::element_rect(
      fill = "white", colour = "#4D4D4D", linewidth = 0.25
    ),
    strip.text = ggplot2::element_text(
      size = 9, face = "bold", margin = ggplot2::margin(3, 2, 3, 2)
    ),
    axis.title = ggplot2::element_text(size = 9),
    axis.text = ggplot2::element_text(size = 8, colour = "#333333"),
    legend.title = ggplot2::element_blank(),
    legend.text = ggplot2::element_text(size = 8),
    legend.key.width = grid::unit(0.65, "cm"),
    plot.margin = ggplot2::margin(3, 3, 3, 3)
  )

make_plot <- function(plot_data, x_label, y_label) {
  ggplot2::ggplot(
    plot_data,
    ggplot2::aes(x = horizontal, y = vertical, colour = group)
  ) +
    ggplot2::geom_point(shape = 1, size = 0.8, alpha = 0.48, stroke = 0.25) +
    ggplot2::scale_colour_manual(
      values = c("Group 0" = "black", "Group 1" = "red"),
      name = NULL,
      guide = ggplot2::guide_legend(
        override.aes = list(alpha = 1, size = 1.8, stroke = 0.4),
        title.position = "left"
      )
    ) +
    ggplot2::scale_x_continuous(
      breaks = seq(0, 0.5, by = 0.1), expand = c(0, 0)
    ) +
    ggplot2::scale_y_continuous(
      breaks = seq(0, 0.5, by = 0.1), expand = c(0, 0)
    ) +
    ggplot2::coord_equal(
      xlim = c(-0.02, 0.52), ylim = c(-0.02, 0.52),
      expand = FALSE, clip = "on"
    ) +
    ggplot2::facet_wrap(~panel, nrow = 1L, labeller = ggplot2::label_parsed) +
    ggplot2::labs(x = x_label, y = y_label) +
    base_theme
}

x12 <- make_pair_data(
  1L, 2L, "paste('(', X[1], ', ', X[2], ')')"
)
x34 <- make_pair_data(
  3L, 4L, "paste('(', X[3], ', ', X[4], ')')"
)
plot_x12 <- make_plot(x12, expression(X[1]), expression(X[2]))
plot_x34 <- make_plot(x34, expression(X[3]), expression(X[4]))
final_plot <- patchwork::wrap_plots(
  plot_x12, plot_x34, nrow = 1L, guides = "collect"
) & ggplot2::theme(legend.position = "bottom")

dir.create(dirname(output_png), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_pdf), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(data=data, labels=labels, draw_order=draw_order), file.path(outdir, "plotted_data.rds"))
ggplot2::ggsave(
  output_png, final_plot, width = 11.8, height = 6.6,
  units = "cm", dpi = 300, bg = "white"
)
ggplot2::ggsave(
  output_pdf, final_plot, width = 11.8, height = 6.6,
  units = "cm", device = grDevices::cairo_pdf, bg = "white"
)

writeLines(c(
  "figure=Supplementary Figure S6 compact preview",
  "width_cm=11.8", "height_cm=6.6", "ticks_and_legend_pt=8", "axis_and_panel_titles_pt=9",
  "scenario=latent_location_shift",
  "n0=500",
  "n1=500",
  "dimension=20",
  "uniform_weight=0.2",
  "transformed=true",
  "seed=2",
  paste0("model_sha256=", model_hash),
  "estimator_fitting=false",
  "group0=black_open_circles",
  "group1=red_open_circles",
  "point_size=0.8", "point_stroke=0.25",
  "point_alpha=0.48",
  "axis_limits=0,0.5_with_0.02_visual_margin",
  "panel_background=white",
  "strip_background=white",
  "major_grid_colour=#E3E3E3",
  paste0("png=", normalizePath(output_png, winslash = "/", mustWork = TRUE)),
  paste0("pdf=", normalizePath(output_pdf, winslash = "/", mustWork = TRUE))
), metadata_path, useBytes = TRUE)

cat("Saved S6 PNG: ", normalizePath(output_png), "\n", sep = "")
cat("Saved S6 PDF: ", normalizePath(output_pdf), "\n", sep = "")
cat("Saved metadata: ", normalizePath(metadata_path), "\n", sep = "")
